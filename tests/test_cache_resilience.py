"""Tests for panel resilience when the run-info cache is unavailable.

回归 2026-11-06: 部署在 VPS 的 dev 镜像, 面板所有读配置的页面一直
转圈, 后端 8001 curl 超时。py-spy 确认主线程（asyncio 事件循环）
卡在 ``SSLSocket.read``，调用链是 ``list_telegram_accounts ->
_serialize_account -> _get_last_checkin_status -> RunContext.list_all()
-> RunContext.get -> Cache.get -> Collection.find_one``。MongoDB
（Atlas）通但回包极慢，而 dev 的 ``MongoClient`` 没设超时
（``socketTimeoutMS`` 默认无限等），于是同步 pymongo 查询冻住整个
后端 API。缓存只是运行记录的来源之一，绝不能让账号列表、新增账号
这类接口跟着它一起失败。
"""

import unittest
from unittest.mock import patch

try:
    from embykeeper.runinfo import RunContext
    from embykeeper.api.telegram import _get_last_checkin_status

    _BACKEND_AVAILABLE = True
except Exception:  # pragma: no cover - missing backend deps
    _BACKEND_AVAILABLE = False


@unittest.skipUnless(_BACKEND_AVAILABLE, "backend dependencies not installed")
class TestListAllCacheFailure(unittest.TestCase):
    def test_list_all_returns_in_memory_runs_when_cache_raises(self):
        """find_by_prefix_values 抛错时不抛到调用方, 只返回内存中的运行任务。
        （list_all 自 ab85765 之后的批量改动起走 find_by_prefix_values
        一次取回全部记录，不再调 find_by_prefix。）"""
        import embykeeper.runinfo as runinfo
        from unittest.mock import MagicMock

        # 注意: runinfo.cache 是懒加载代理, 未加载配置时直接碰它会先报
        # "config not loaded", 所以把模块级的 cache 整个换成假对象。
        fake_cache = MagicMock()
        fake_cache.find_by_prefix_values.side_effect = RuntimeError("Connection refused")
        with patch.object(runinfo, "cache", fake_cache):
            self.assertEqual(RunContext.list_all(), [])

    def test_list_all_skips_unreadable_run(self):
        """某条运行记录解析不了时跳过它, 而不是让整个列表失败。
        （list_all 自批量改动起不再逐条调 RunContext.get，而是直接解析
        批量取回的记录；这里用一条坏 JSON 模拟读不到的记录。）"""
        import embykeeper.runinfo as runinfo
        from unittest.mock import MagicMock

        fake_cache = MagicMock()
        fake_cache.find_by_prefix_values.return_value = {
            "runinfo.deadbeef": "{not-json",
        }
        with patch.object(runinfo, "cache", fake_cache):
            self.assertEqual(RunContext.list_all(), [])


@unittest.skipUnless(_BACKEND_AVAILABLE, "backend dependencies not installed")
class TestLastCheckinStatusCacheFailure(unittest.TestCase):
    def test_returns_unknown_when_list_all_raises(self):
        """list_all 抛错时上次签到状态返回未知, 而不是把异常抛上去。"""
        with patch.object(RunContext, "list_all", side_effect=RuntimeError("boom")):
            self.assertEqual(_get_last_checkin_status("+8613800000000"), (None, None))


@unittest.skipUnless(_BACKEND_AVAILABLE, "backend dependencies not installed")
class TestTelegramAccountsEndpointCacheFailure(unittest.TestCase):
    def test_list_returns_accounts_when_cache_down(self):
        """回归: 缓存挂了, GET /telegram/accounts 也应 200 返回账号,
        上次签到状态为空, 而不是一直转圈/报 500。"""
        from fastapi.testclient import TestClient

        import embykeeper.api.telegram as telegram_api
        from embykeeper.api import create_app
        from embykeeper.api.auth import require_auth
        from embykeeper.schema import Config, TelegramAccount, TelegramConfig

        cfg = Config(
            telegram=TelegramConfig(
                account=[TelegramAccount(phone="+8613800000000")]
            )
        )
        app = create_app()
        app.dependency_overrides[require_auth] = lambda: True
        # 在真实故障点打补丁: 缓存层抛错, list_all 内部应兜住并返回空历史。
        # runinfo.cache 是懒加载代理, 直接碰它会先报 "config not loaded",
        # 所以把模块级的 cache 整个换成假对象。
        import embykeeper.runinfo as runinfo
        from unittest.mock import MagicMock

        fake_cache = MagicMock()
        fake_cache.find_by_prefix_values.side_effect = RuntimeError("Connection refused")
        with (
            patch.object(telegram_api, "_load_config", return_value=cfg),
            patch.object(runinfo, "cache", fake_cache),
        ):
            with TestClient(app, raise_server_exceptions=False) as client:
                resp = client.get("/api/telegram/accounts")

        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(len(body["accounts"]), 1)
        self.assertEqual(body["accounts"][0]["phone"], "+8613800000000")
        self.assertIsNone(body["accounts"][0]["last_checkin_status"])
        self.assertIsNone(body["accounts"][0]["last_checkin_time"])


if __name__ == "__main__":
    unittest.main()
