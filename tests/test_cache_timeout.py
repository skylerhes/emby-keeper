"""Tests for the hard timeout on MongoDB cache operations.

回归 2026-11-06: 45879da 已给 MongoClient 加了 socketTimeoutMS=5000，
但 py-spy 证明仍不够——主线程照样卡在 Cache.get -> Collection.find_one
-> SSLSocket.read（list_telegram_accounts -> RunContext.list_all ->
RunContext.get），事件循环冻住，所有面板页面一直转圈。pymongo 的
socket 超时在回包极慢的连接上并不可靠，所以缓存操作改在工作线程
里跑，调用方最多等 1 秒；卡住的 socket 读留在工作线程里，缓存
是 best-effort，超时/失败时返回默认值。
"""

import pathlib
import tempfile
import threading
import time
import unittest
from unittest.mock import MagicMock

from embykeeper.cache import Cache

try:
    from embykeeper.runinfo import RunContext  # noqa: F401
    from embykeeper.api.telegram import _get_last_checkin_status  # noqa: F401

    _BACKEND_AVAILABLE = True
except Exception:  # pragma: no cover - missing backend deps
    _BACKEND_AVAILABLE = False


def _mongo_cache(collection):
    """构造一个走 MongoDB 路径的 Cache，不连真实 MongoDB。"""
    cache = Cache.__new__(Cache)
    cache._mongo_client = object()  # 仅作 truthy 标记
    cache._collection = collection
    return cache


def _json_cache(tmp_path):
    cache = Cache.__new__(Cache)
    cache._mongo_client = None
    cache._cache_file = tmp_path / "cache.json"
    cache._data = {}
    return cache


class TestMongoTimeout(unittest.TestCase):
    def test_get_returns_default_when_find_one_blocks(self):
        """find_one 卡住时 get 在约 1 秒后返回 default，不阻塞调用方。"""
        release = threading.Event()

        def blocked_find_one(query):
            release.wait(timeout=30)
            return None

        collection = MagicMock()
        collection.find_one.side_effect = blocked_find_one
        cache = _mongo_cache(collection)
        start = time.monotonic()
        try:
            self.assertEqual(cache.get("runinfo.x", "dflt"), "dflt")
        finally:
            release.set()  # 放行工作线程，避免进程退出时等它
        self.assertLess(time.monotonic() - start, 2.5)

    def test_find_by_prefix_returns_empty_when_find_blocks(self):
        """find 卡住时 find_by_prefix 在约 1 秒后返回 []。"""
        release = threading.Event()

        def blocked_find(*args, **kwargs):
            release.wait(timeout=30)
            return []

        collection = MagicMock()
        collection.find.side_effect = blocked_find
        cache = _mongo_cache(collection)
        start = time.monotonic()
        try:
            self.assertEqual(cache.find_by_prefix("runinfo."), [])
        finally:
            release.set()
        self.assertLess(time.monotonic() - start, 2.5)

    def test_write_ops_return_quickly_when_mongo_blocks(self):
        """set/delete/delete_many 卡住时也不阻塞调用方超过约 1 秒。"""
        release = threading.Event()

        def blocked(*args, **kwargs):
            release.wait(timeout=30)
            return None

        collection = MagicMock()
        collection.update_one.side_effect = blocked
        collection.delete_one.side_effect = blocked
        collection.delete_many.side_effect = blocked
        cache = _mongo_cache(collection)
        start = time.monotonic()
        try:
            cache.set("a.b", 1)
            cache.delete("a.b")
            cache.delete_many(["a.b"])
        finally:
            release.set()
        # 三次写操作各最多等 1 秒；顺序执行，这里放宽到 10 秒，
        # 关键是每次都不无限等（无硬超时时这里会卡 90 秒）。
        self.assertLess(time.monotonic() - start, 10.0)


class TestMongoFailure(unittest.TestCase):
    def test_get_returns_default_when_find_one_raises(self):
        """find_one 抛错时 get 返回 default，而不是把异常抛上去。"""
        collection = MagicMock()
        collection.find_one.side_effect = RuntimeError("Connection refused")
        cache = _mongo_cache(collection)
        self.assertEqual(cache.get("a.b", "dflt"), "dflt")

    def test_find_by_prefix_returns_empty_when_find_raises(self):
        """find 抛错时 find_by_prefix 返回 []。"""
        collection = MagicMock()
        collection.find.side_effect = RuntimeError("Connection refused")
        cache = _mongo_cache(collection)
        self.assertEqual(cache.find_by_prefix("runinfo."), [])

    def test_write_ops_silently_skip_when_mongo_raises(self):
        """set/delete/delete_many/delete_by_prefix 失败时不抛错。"""
        collection = MagicMock()
        collection.update_one.side_effect = RuntimeError("boom")
        collection.delete_one.side_effect = RuntimeError("boom")
        collection.delete_many.side_effect = RuntimeError("boom")
        collection.find.side_effect = RuntimeError("boom")
        cache = _mongo_cache(collection)
        cache.set("a.b", 1)  # 不应抛错
        cache.delete("a.b")  # 不应抛错
        cache.delete_many(["a.b"])  # 不应抛错
        cache.delete_by_prefix("a.")  # find 失败 -> [] -> 无事可做，不应抛错


class TestMongoHappyPath(unittest.TestCase):
    def test_get_and_find_by_prefix_return_real_values(self):
        """MongoDB 正常时读写仍走工作线程并返回真实结果。"""
        collection = MagicMock()

        def find_one(query):
            return {"_id": "a.b", "value": 42} if query == {"_id": "a.b"} else None

        collection.find_one.side_effect = find_one
        collection.find.return_value = [{"_id": "runinfo.1"}, {"_id": "runinfo.2"}]
        cache = _mongo_cache(collection)
        self.assertEqual(cache.get("a.b"), 42)
        self.assertEqual(cache.get("missing", "dflt"), "dflt")
        self.assertEqual(cache.find_by_prefix("runinfo."), ["runinfo.1", "runinfo.2"])
        # find_one 被调了两次（"a.b" 和 "missing"），只断言其中一次的参数。
        collection.find_one.assert_any_call({"_id": "a.b"})


class TestJsonPathUnchanged(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.cache = _json_cache(pathlib.Path(self._tmp.name))

    def tearDown(self):
        self._tmp.cleanup()

    def test_set_get_delete_roundtrip(self):
        """JSON 路径的读写删行为不变，且仍落盘。"""
        c = self.cache
        c.set("a.b", 1)
        c.set("a.c", 2)
        c.set("x.y", 3)
        self.assertEqual(c.get("a.b"), 1)
        self.assertEqual(c.get("a.c"), 2)
        self.assertEqual(c.get("missing", "dflt"), "dflt")
        self.assertEqual(sorted(c.find_by_prefix("a.")), ["a.b", "a.c"])
        c.delete("a.b")
        self.assertEqual(c.get("a.b", "dflt"), "dflt")
        c.delete_many(["a.c", "x.y"])
        self.assertEqual(c.find_by_prefix(""), [])
        self.assertTrue((pathlib.Path(self._tmp.name) / "cache.json").exists())

    def test_delete_by_prefix(self):
        c = self.cache
        c.set("a.b", 1)
        c.set("a.c", 2)
        c.set("x.y", 3)
        c.delete_by_prefix("a.")
        self.assertEqual(c.get("a.b", "dflt"), "dflt")
        self.assertEqual(c.get("x.y"), 3)


@unittest.skipUnless(_BACKEND_AVAILABLE, "backend dependencies not installed")
class TestTelegramAccountsEndpointMongoBlocked(unittest.TestCase):
    def test_endpoint_returns_200_when_mongo_blocks(self):
        """端到端回归：缓存 find 卡住时，GET /telegram/accounts 仍在
        约 1 秒内返回 200，而不是像 2026-11-06 的故障那样一直转圈。
        （list_telegram_accounts 只调一次 list_all，即一次批量
        find_by_prefix_values 调用。）"""
        from unittest.mock import patch

        from fastapi.testclient import TestClient

        import embykeeper.api.telegram as telegram_api
        import embykeeper.runinfo as runinfo
        from embykeeper.api import create_app
        from embykeeper.api.auth import require_auth
        from embykeeper.schema import Config, TelegramAccount, TelegramConfig

        release = threading.Event()

        def blocked_find(*args, **kwargs):
            release.wait(timeout=30)
            return []

        def blocked_find_one(*args, **kwargs):
            release.wait(timeout=30)
            return None

        collection = MagicMock()
        collection.find.side_effect = blocked_find
        collection.find_one.side_effect = blocked_find_one
        blocking_cache = _mongo_cache(collection)

        cfg = Config(telegram=TelegramConfig(account=[TelegramAccount(phone="+8613800000000")]))
        app = create_app()
        app.dependency_overrides[require_auth] = lambda: True
        with (
            patch.object(telegram_api, "_load_config", return_value=cfg),
            patch.object(runinfo, "cache", blocking_cache),
        ):
            with TestClient(app, raise_server_exceptions=False) as client:
                start = time.monotonic()
                try:
                    resp = client.get("/api/telegram/accounts")
                finally:
                    release.set()
                elapsed = time.monotonic() - start

        self.assertEqual(resp.status_code, 200)
        self.assertLess(elapsed, 6.0)
        body = resp.json()
        self.assertEqual(len(body["accounts"]), 1)
        self.assertIsNone(body["accounts"][0]["last_checkin_status"])
        self.assertIsNone(body["accounts"][0]["last_checkin_time"])


if __name__ == "__main__":
    unittest.main()
