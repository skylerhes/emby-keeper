import asyncio
import base64
import os
import random
import time
from typing import List, Optional

import emoji
import httpx
from pyrogram.errors import RPCError
from pyrogram.types import Message

from . import AnswerBotCheckin

# 内置的公共智谱 Key: 历史上被硬编码在代码中, 所有未配置 ZHIPU_API_KEY 的用户共用它.
# 免费额度按分钟限流, 多人同时使用必然触发 429, 强烈建议自备 Key.
_PUBLIC_ZHIPU_KEY = "2eade9f1396646bfb64a0fe3ab8afba3.6H8UfxWTsqeGMsFQ"
_ZHIPU_ENDPOINT = "https://open.bigmodel.cn/api/paas/v4/chat/completions"


class VisionRateLimited(Exception):
    """视觉识别接口被限流 (HTTP 429).

    与"识别错误"区分开: 限流是可恢复的, 应等待后重试, 而不能直接判定签到失败.
    """

    def __init__(self, wait: float, detail: str = ""):
        super().__init__(detail or f"被限流, 建议等待 {wait:.0f} 秒")
        self.wait = wait


class _Throttle:
    """进程级固定间隔限速器.

    多个账号并发签到时, 视觉请求会在同一时刻集中发出, 共用额度必然撞 429.
    这里保证任意两次请求之间至少间隔 interval 秒 (所有实例共享同一计时).
    """

    def __init__(self):
        self._lock: Optional[asyncio.Lock] = None
        self._last = 0.0

    async def acquire(self, interval: float):
        if interval <= 0:
            return
        # 事件循环相关对象懒创建, 避免模块导入时绑定错误的 loop
        if self._lock is None:
            self._lock = asyncio.Lock()
        async with self._lock:
            wait = self._last + interval - time.monotonic()
            if wait > 0:
                await asyncio.sleep(wait)
            self._last = time.monotonic()


_vision_throttle = _Throttle()


class TerminusCheckin(AnswerBotCheckin):
    name = "终点站"
    bot_username = "EmbyPublicBot"
    bot_checkin_cmd = ["/cancel", "/checkin"]
    bot_text_ignore = ["会话已取消", "没有活跃的会话"]
    bot_checked_keywords = ["今天已签到"]
    max_retries = 1
    bot_use_history = 3

    # ---- 视觉识别配置 (仅「终点站」使用; 均可通过环境变量覆盖, 默认值即当前配置) ----
    zhipu_api_key = os.environ.get("ZHIPU_API_KEY", _PUBLIC_ZHIPU_KEY)
    zhipu_model = os.environ.get("ZHIPU_VISION_MODEL", "glm-4v-flash")
    # 智谱 API 端点, 换厂商时设置 VISION_BASE_URL (需 OpenAI 兼容的 /chat/completions)
    vision_base_url = os.environ.get("VISION_BASE_URL", _ZHIPU_ENDPOINT).rstrip("/")
    vision_timeout = float(os.environ.get("VISION_TIMEOUT", "30"))
    vision_max_retries = max(int(os.environ.get("VISION_MAX_RETRIES", "3")), 1)
    # 同一进程内两次视觉请求的最小间隔 (秒), 用于规避免费版按分钟限流
    vision_min_interval = float(os.environ.get("ZHIPU_MIN_INTERVAL", "1.5"))
    # 429 退避基数 (秒), 实际等待为 base * 2**第几次 + 抖动, 上限 30 秒
    vision_backoff_base = float(os.environ.get("VISION_BACKOFF_BASE", "4"))

    _warned_public_key = False

    # ---------- 通用工具 ----------

    def _check_public_key(self):
        """使用内置公共 Key 时提示一次, 避免每次签到都刷屏."""
        cls = type(self)
        if self.zhipu_api_key == _PUBLIC_ZHIPU_KEY and not cls._warned_public_key:
            cls._warned_public_key = True
            self.log.warning(
                "正在使用内置的公共智谱 API Key, 该 Key 由所有用户共享, 极易触发 429 限流. "
                "建议到 https://open.bigmodel.cn 免费申请自己的 Key 并设置环境变量 ZHIPU_API_KEY."
            )

    @staticmethod
    def _build_prompt(options_cleaned: List[str]) -> str:
        return (
            "请根据图片内容，从候选项中选出最匹配的一项。\n"
            "候选项如下：\n" + "\n".join("- " + x for x in options_cleaned) + "\n\n"
            "要求：\n"
            "1. 只能输出候选项中的一个；\n"
            "2. 不要解释；\n"
            "3. 不要输出多余文字；\n"
            "4. 如果不确定，也必须从候选项中选最可能的一项。"
        )

    @staticmethod
    def _clean_result(text: str) -> str:
        """清理模型输出中的标点与空白, 便于与候选文本比对."""
        for ch in ("“", "”", '"', "'", "。", "，", ",", "：", ":", " ", "\n", "\t"):
            text = text.replace(ch, "")
        return text.strip()

    def _match_option(self, content: str, options_cleaned: List[str], source: str) -> Optional[str]:
        """把模型返回的文本映射回某个候选项, 失败返回 None."""
        result = self._clean_result(content or "")
        self.log.debug("%s提取出的结果: %s", source, result)
        if result in options_cleaned:
            return result
        # 模型偶尔会带前缀 (如 "答案: A"), 退化为子串匹配
        for option in options_cleaned:
            if option and option in result:
                return option
        self.log.warning("%s返回了无效结果: %s", source, result)
        return None

    async def _image_data_url(self, message: Message) -> Optional[str]:
        """把验证码图片转成 data URL (Telegram 照片固定为 JPEG)."""
        try:
            image_data = await self.client.download_media(message, in_memory=True)
        except Exception as e:
            self.log.warning("下载验证码图片失败: %s", e)
            return None
        if not image_data:
            self.log.warning("验证码图片为空, 无法识别.")
            return None
        image_bytes = image_data.getvalue() if hasattr(image_data, "getvalue") else image_data
        return "data:image/jpeg;base64," + base64.b64encode(image_bytes).decode("utf-8")

    def _retry_after(self, resp: httpx.Response) -> float:
        """解析 429 的建议等待时间: 优先 Retry-After, 否则退回退避基数."""
        retry_after = resp.headers.get("retry-after")
        if retry_after:
            try:
                return max(float(retry_after), 1.0)
            except ValueError:
                pass
        return self.vision_backoff_base

    @staticmethod
    def _rate_limit_detail(resp: httpx.Response) -> str:
        """提取 429 的具体原因 (智谱会在 body 里返回 code/message)."""
        try:
            body = resp.json()
            err = body.get("error") or {}
            code = err.get("code") or body.get("code")
            msg = err.get("message") or body.get("message") or ""
            if code or msg:
                return f"code={code} {msg}".strip()
        except Exception:
            pass
        return (resp.text or "")[:160]

    # ---------- 识别后端 ----------

    async def _zhipu_visual(self, message: Message, options_cleaned: List[str]) -> Optional[str]:
        """调用智谱视觉模型识别, 从候选项中返回最匹配的一项.

        抛出 VisionRateLimited 表示被限流 (可重试), 其他失败返回 None.
        """
        if not self.zhipu_api_key:
            self.log.warning("未设置 ZHIPU_API_KEY 环境变量.")
            return None
        self._check_public_key()

        image_url = await self._image_data_url(message)
        if not image_url:
            return None

        payload = {
            "model": self.zhipu_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_url}},
                        {"type": "text", "text": self._build_prompt(options_cleaned)},
                    ],
                }
            ],
            # 取 0.01 而非 0, 避免部分模型对极端值报错或返回空内容
            "temperature": 0.01,
            "max_tokens": 32,
        }
        headers = {
            "Authorization": "Bearer " + self.zhipu_api_key,
            "Content-Type": "application/json",
        }

        await _vision_throttle.acquire(self.vision_min_interval)
        try:
            async with httpx.AsyncClient(timeout=self.vision_timeout) as client:
                resp = await client.post(self.vision_base_url, headers=headers, json=payload)
                if resp.status_code == 429:
                    raise VisionRateLimited(self._retry_after(resp), self._rate_limit_detail(resp))
                resp.raise_for_status()
                data = resp.json()
                self.log.debug("智谱接口完整返回内容: %s", data)
                content = data["choices"][0]["message"]["content"]
                return self._match_option(content, options_cleaned, "智谱")
        except VisionRateLimited:
            raise
        except Exception as e:
            self.log.warning("智谱视觉识别失败: %s", e)
            return None

    async def _fallback_visual(self, message: Message, options_cleaned: List[str]) -> Optional[str]:
        """智谱不可用时, 回落到任意 OpenAI 兼容的视觉端点 (可选).

        通过 VISION_FALLBACK_BASE_URL / VISION_FALLBACK_API_KEY / VISION_FALLBACK_MODEL 启用.
        """
        base = os.environ.get("VISION_FALLBACK_BASE_URL", "").rstrip("/")
        key = os.environ.get("VISION_FALLBACK_API_KEY", "")
        model = os.environ.get("VISION_FALLBACK_MODEL", "")
        if not (base and key and model):
            return None

        image_url = await self._image_data_url(message)
        if not image_url:
            return None

        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_url}},
                        {"type": "text", "text": self._build_prompt(options_cleaned)},
                    ],
                }
            ],
            "temperature": 0.01,
            "max_tokens": 32,
        }
        try:
            async with httpx.AsyncClient(timeout=self.vision_timeout) as client:
                resp = await client.post(
                    f"{base}/chat/completions",
                    headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                    json=payload,
                )
                if resp.status_code == 429:
                    raise VisionRateLimited(self._retry_after(resp), self._rate_limit_detail(resp))
                resp.raise_for_status()
                content = resp.json()["choices"][0]["message"]["content"]
                self.log.debug("备用视觉端点返回: %s", content)
                return self._match_option(content, options_cleaned, "备用视觉端点")
        except VisionRateLimited:
            raise
        except Exception as e:
            self.log.warning("备用视觉端点识别失败: %s", e)
            return None

    # ---------- 带重试的识别流程 ----------

    async def _detect(self, message: Message, options_cleaned: List[str]) -> Optional[str]:
        """先走智谱, 全部失败再回落备用端点.

        429 与识别失败分开处理: 限流按 Retry-After / 指数退避后再试,
        不直接判定为"验证码识别错误"而放弃本次签到.
        """
        total = self.vision_max_retries

        for i in range(total):
            try:
                result = await self._zhipu_visual(message, options_cleaned)
            except VisionRateLimited as e:
                wait = min(e.wait * (2**i) + random.uniform(0, 1), 30)
                self.log.warning("智谱接口限流 (429), 等待 %.1f 秒后重试 (%s/%s): %s", wait, i + 1, total, e)
                if i < total - 1:
                    await asyncio.sleep(wait)
                continue
            if result:
                self.log.debug("已通过智谱解析答案: %s.", result)
                return result
            self.log.warning("智谱解析失败, 正在重试解析 (%s/%s).", i + 1, total)
            if i < total - 1:
                await asyncio.sleep(3)

        # 智谱彻底不可用: 尝试备用视觉端点
        if os.environ.get("VISION_FALLBACK_MODEL"):
            self.log.info("智谱不可用, 尝试回落备用视觉端点 (模型: %s).", os.environ.get("VISION_FALLBACK_MODEL"))
            for i in range(2):
                try:
                    result = await self._fallback_visual(message, options_cleaned)
                except VisionRateLimited as e:
                    wait = min(e.wait * (2**i) + random.uniform(0, 1), 30)
                    self.log.warning("备用视觉端点限流 (429), 等待 %.1f 秒后重试 (%s/2): %s", wait, i + 1, e)
                    if i < 1:
                        await asyncio.sleep(wait)
                    continue
                if result:
                    self.log.debug("已通过备用视觉端点解析答案: %s.", result)
                    return result
                if i < 1:
                    await asyncio.sleep(2)
        return None

    async def on_photo(self, message: Message):
        """分析传入的验证码图片并点击对应按钮."""
        if not message.reply_markup:
            return

        clean = lambda o: emoji.replace_emoji(o, "").replace(" ", "")
        keys = [k for r in message.reply_markup.inline_keyboard for k in r]
        options = [k.text for k in keys]
        options_cleaned = [clean(o) for o in options]

        if len(options) < 2:
            return

        result = await self._detect(message, options_cleaned)

        if not result:
            self.log.warning("签到失败: 验证码识别错误.")
            return await self.fail()

        if result not in options_cleaned:
            self.log.warning("签到失败: 返回了无效结果: %s", result)
            return await self.fail()

        result = options[options_cleaned.index(result)]

        await asyncio.sleep(random.uniform(0.5, 1.5))
        try:
            await message.click(result)
        except RPCError:
            self.log.warning("按钮点击失败.")
