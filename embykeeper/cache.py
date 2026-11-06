import concurrent.futures
import json
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, List

from loguru import logger

from .utils import CachedFuncProxy
from .config import config


# 硬超时: pymongo 的 socketTimeoutMS 在回包极慢时仍可能让调用方一直等
# （2026-11-06 py-spy 实测: 主线程卡在 SSLSocket.read，事件循环冻住，
# 所有面板页面一直转圈）。所以所有 MongoDB 操作都在工作线程里跑，
# 调用方最多等 _MONGO_TIMEOUT 秒；卡住的 socket 读留在工作线程里，
# 不再占用事件循环。缓存只是 best-effort，超时/失败时返回默认值。
_MONGO_TIMEOUT = 3.0
_mongo_executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="cache-mongo")

# 同一类缓存失败 60 秒内只记一次日志，避免 MongoDB 挂掉时日志刷屏。
_WARN_COOLDOWN = 60.0
_last_warn_at: dict = {}


def _warn_throttled(tag: str, message: str) -> None:
    now = time.monotonic()
    if now - _last_warn_at.get(tag, 0.0) >= _WARN_COOLDOWN:
        _last_warn_at[tag] = now
        logger.warning(message)


def _mongo_call(tag: str, fn, default):
    """在工作线程里执行一次 pymongo 调用，最多等待 _MONGO_TIMEOUT 秒。

    超时或任何异常时返回 default，调用方绝不因此阻塞或抛错。
    """
    future = _mongo_executor.submit(fn)
    try:
        return future.result(timeout=_MONGO_TIMEOUT)
    except concurrent.futures.TimeoutError:
        _warn_throttled(tag, f"缓存 MongoDB 操作超时 ({tag})，已跳过并返回默认值。")
        return default
    except Exception as exc:
        _warn_throttled(tag, f"缓存 MongoDB 操作失败 ({tag}): {exc}，已跳过并返回默认值。")
        return default


class Cache:
    def __init__(self):
        self._mongo_client = None
        if hasattr(config, "mongodb") and config.mongodb:
            try:
                from pymongo import MongoClient

                # 缓存只是 best-effort: MongoDB 连不上时要快速失败, 不能让
                # 面板接口卡住 (默认 serverSelectionTimeoutMS 是 30 秒,
                # 曾导致账号列表接口 30 秒后才报 500, 前端一直转圈)。
                self._mongo_client = MongoClient(
                    config.mongodb,
                    serverSelectionTimeoutMS=3000,
                    connectTimeoutMS=3000,
                    socketTimeoutMS=5000,
                )
                self._db = self._mongo_client.embykeeper
                self._collection = self._db.cache
            except ImportError:
                logger.warning("没有安装 pymongo 包, 将使用 JSON 存储缓存.")
                self._setup_json_cache()
        else:
            self._setup_json_cache()

    def _setup_json_cache(self):
        self._cache_file = config.basedir / "cache.json"
        self._data = {}
        if self._cache_file.exists():
            try:
                with open(self._cache_file, "r", encoding="utf-8") as f:
                    self._data = json.load(f)
            except json.JSONDecodeError:
                logger.warning("缓存文件损坏, 将使用全新缓存.")

    def get(self, key: str, default: Any = None) -> Any:
        if self._mongo_client:
            result = _mongo_call("get", lambda: self._collection.find_one({"_id": key}), None)
            return result["value"] if result else default
        else:
            value = self._data
            try:
                for part in key.split("."):
                    value = value.get(part, {})
                return default if value == {} else value
            except (AttributeError, TypeError):
                return default

    def set(self, key: str, value: Any) -> None:
        if self._mongo_client:
            _mongo_call(
                "set",
                lambda: self._collection.update_one(
                    {"_id": key}, {"$set": {"value": value}}, upsert=True
                ),
                None,
            )
        else:
            parts = key.split(".")
            current = self._data
            for part in parts[:-1]:
                current = current.setdefault(part, {})
            current[parts[-1]] = value
            with open(self._cache_file, "w", encoding="utf-8") as f:
                json.dump(self._data, f, ensure_ascii=False)

    def delete(self, key: str) -> None:
        if self._mongo_client:
            _mongo_call("delete", lambda: self._collection.delete_one({"_id": key}), None)
        else:
            parts = key.split(".")
            current = self._data
            path = []

            # 遍历路径, 检查每一层
            for part in parts[:-1]:
                if not isinstance(current, dict) or part not in current:
                    return
                current = current[part]
                path.append((part, current))

            # 检查并删除最后一个键
            if isinstance(current, dict) and parts[-1] in current:
                del current[parts[-1]]

                # 清理空字典
                for part, parent in reversed(path):
                    if isinstance(parent, dict) and part in parent and not parent[part]:
                        del parent[part]
                    else:
                        break

            with open(self._cache_file, "w", encoding="utf-8") as f:
                json.dump(self._data, f, ensure_ascii=False, indent=2)

    def find_by_prefix(self, prefix: str) -> List[str]:
        if self._mongo_client:
            # 游标是惰性的，必须在工作线程里就 list() 物化，
            # 否则网络 I/O 会回到调用方线程上做。
            docs = _mongo_call(
                "find_by_prefix",
                lambda: list(
                    self._collection.find({"_id": {"$regex": f"^{prefix}"}}, {"_id": 1})
                ),
                [],
            )
            return [doc["_id"] for doc in docs]
        else:

            def get_keys_with_prefix(d, current_path="", keys=None):
                if keys is None:
                    keys = []
                for k, v in d.items():
                    path = f"{current_path}.{k}" if current_path else k
                    if isinstance(v, dict):
                        get_keys_with_prefix(v, path, keys)
                    else:
                        if path.startswith(prefix):
                            keys.append(path)
                return keys

            return get_keys_with_prefix(self._data)

    def delete_by_prefix(self, prefix: str) -> None:
        keys = self.find_by_prefix(prefix)
        for key in keys:
            self.delete(key)

    def delete_many(self, keys: List[str]) -> None:
        """批量删除多个键的缓存

        Args:
            keys: 要删除的键列表
        """
        if self._mongo_client:
            _mongo_call(
                "delete_many",
                lambda: self._collection.delete_many({"_id": {"$in": keys}}),
                None,
            )
        else:
            # 批量删除所有键, 只写入一次文件
            changed = False
            for key in keys:
                parts = key.split(".")
                current = self._data
                path = []

                # 遍历路径, 检查每一层
                for part in parts[:-1]:
                    if not isinstance(current, dict) or part not in current:
                        break
                    current = current[part]
                    path.append((part, current))

                # 检查并删除最后一个键
                if isinstance(current, dict) and parts[-1] in current:
                    del current[parts[-1]]
                    changed = True

                    # 清理空字典
                    for part, parent in reversed(path):
                        if isinstance(parent, dict) and part in parent and not parent[part]:
                            del parent[part]
                        else:
                            break

            # 只在有改动时写入一次文件
            if changed:
                with open(self._cache_file, "w", encoding="utf-8") as f:
                    json.dump(self._data, f, ensure_ascii=False, indent=2)


cache: Cache = CachedFuncProxy(lambda: Cache())
