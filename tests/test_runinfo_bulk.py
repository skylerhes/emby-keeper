"""任务记录接口 (runinfo) 一次取回运行记录，不再逐条查缓存。

背景: 任务记录页 (GET /runinfo) 原先是 N+1 查询——先
cache.find_by_prefix("runinfo.") 拿 keys，再对每个 id 逐个
RunContext.get（即一次 find_one），每个最多等一个 1 秒硬超时；
_context_to_dict 还对每个任务再调一次 cache.get 取子任务数。
任务一多就是几十次 × 1 秒串行累加，页面几分钟加载不出来；
每次超时还占住一个线程池工作线程（卡在 SSLSocket.read），
占满 8 个后其它页面也跟着慢/加载不出来。
修复：列表接口只调一次 RunContext.list_all()（已批量取回），
子任务关系直接从已取回的上下文按 parent_ids 推算，不再逐条查缓存；
运行中任务接口改用只读内存的 RunContext.list_running()。
"""

import re
import threading
import time
import unittest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from embykeeper.cache import Cache
from embykeeper.runinfo import LogRecord, RunContext, RunStatus

try:
    import embykeeper.api.runinfo as runinfo_api
    import embykeeper.runinfo as runinfo

    _BACKEND_AVAILABLE = True
except Exception:  # pragma: no cover - missing backend deps
    _BACKEND_AVAILABLE = False


def _now(hour=12, minute=0):
    return datetime(2026, 11, 6, hour, minute, 0, tzinfo=timezone.utc)


def _mongo_cache(collection):
    """构造一个走 MongoDB 路径的 Cache，不连真实 MongoDB。"""
    cache = Cache.__new__(Cache)
    cache._mongo_client = object()  # 仅作 truthy 标记
    cache._collection = collection
    return cache


def _run_json(run_id, description, status=RunStatus.SUCCESS, parent_ids=None,
              start_hour=12, start_minute=0, log_message=None):
    logs = []
    if log_message is not None:
        logs = [LogRecord(level="INFO", message=log_message, time=_now(start_hour, start_minute))]
    return RunContext(
        id=run_id,
        description=description,
        status=status,
        parent_ids=parent_ids or [],
        start_time=_now(start_hour, start_minute),
        end_time=_now(start_hour, start_minute),
        log=logs,
    ).model_dump_json()


def _fake_find(all_docs):
    """像 MongoDB 一样按 _id 前缀正则过滤。"""
    def fake_find(filter_, projection):
        pattern = filter_["_id"]["$regex"]
        return [d for d in all_docs if re.search(pattern, d["_id"])]
    return fake_find


def _make_client():
    from fastapi.testclient import TestClient

    from embykeeper.api import create_app
    from embykeeper.api.auth import require_auth

    app = create_app()
    app.dependency_overrides[require_auth] = lambda: True
    return TestClient(app, raise_server_exceptions=False)


def _patch_caches(test_case, collection):
    """把 runinfo 与 api.runinfo 两个模块的 cache 都换成 mock 集合的 Cache。"""
    cache = _mongo_cache(collection)
    p1 = patch.object(runinfo, "cache", cache)
    p2 = patch.object(runinfo_api, "cache", cache)
    p1.start()
    p2.start()
    test_case.addCleanup(p1.stop)
    test_case.addCleanup(p2.stop)
    return cache


@unittest.skipUnless(_BACKEND_AVAILABLE, "backend dependencies not installed")
class TestListTopLevelRunsBulk(unittest.TestCase):
    def _docs(self):
        return [
            {"_id": "runinfo.TOPAAA", "value": _run_json("TOPAAA", "top one", start_hour=12, start_minute=0)},
            {"_id": "runinfo.TOPBBB", "value": _run_json("TOPBBB", "top two", start_hour=12, start_minute=10)},
            {"_id": "runinfo.CHILD1", "value": _run_json(
                "CHILD1", "child of top one", parent_ids=["TOPAAA"],
                start_hour=12, start_minute=5, log_message="child log")},
            {"_id": "runinfo.CHILD2", "value": _run_json(
                "CHILD2", "child of top one", parent_ids=["TOPAAA"],
                start_hour=12, start_minute=6)},
            # children 键应被 list_all 跳过，child_count 从 parent_ids 推算
            {"_id": "runinfo.children.TOPAAA", "value": '["CHILD1", "CHILD2"]'},
        ]

    def test_single_bulk_query_filter_sort_and_child_count(self):
        """多个任务时 GET /runinfo 只发一次 find，find_one 为 0 次。"""
        collection = MagicMock()
        collection.find.side_effect = _fake_find(self._docs())
        _patch_caches(self, collection)
        with patch.dict(runinfo._running_runs, {}, clear=True):
            client = _make_client()
            with client:
                resp = client.get("/api/runinfo")

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(collection.find.call_count, 1)
        self.assertEqual(collection.find_one.call_count, 0)
        runs = resp.json()["runs"]
        # 只返回顶层任务（CHILD1/CHILD2 不在列表里），按开始时间倒序
        self.assertEqual([r["id"] for r in runs], ["TOPBBB", "TOPAAA"])
        by_id = {r["id"]: r for r in runs}
        # child_count 从 parent_ids 推算：TOPAAA 有 2 个子任务，TOPBBB 有 0 个
        self.assertEqual(by_id["TOPAAA"]["child_count"], 2)
        self.assertEqual(by_id["TOPBBB"]["child_count"], 0)

    def test_status_filter_applied(self):
        """状态过滤仍生效。"""
        docs = [
            {"_id": "runinfo.OKAAAA", "value": _run_json("OKAAAA", "ok", status=RunStatus.SUCCESS)},
            {"_id": "runinfo.FAILAA", "value": _run_json("FAILAA", "bad", status=RunStatus.FAIL)},
        ]
        collection = MagicMock()
        collection.find.side_effect = _fake_find(docs)
        _patch_caches(self, collection)
        with patch.dict(runinfo._running_runs, {}, clear=True):
            client = _make_client()
            with client:
                resp = client.get("/api/runinfo?status=SUCCESS")

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(collection.find.call_count, 1)
        self.assertEqual([r["id"] for r in resp.json()["runs"]], ["OKAAAA"])

    def test_returns_quickly_when_cache_blocks(self):
        """缓存卡住时 GET /runinfo 在约 1 秒内返回空列表。"""
        release = threading.Event()

        def blocked_find(*args, **kwargs):
            release.wait(timeout=30)
            return []

        collection = MagicMock()
        collection.find.side_effect = blocked_find
        _patch_caches(self, collection)
        with patch.dict(runinfo._running_runs, {}, clear=True):
            client = _make_client()
            with client:
                start = time.monotonic()
                try:
                    resp = client.get("/api/runinfo")
                finally:
                    release.set()

        self.assertLess(time.monotonic() - start, 2.5)
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["runs"], [])
        self.assertEqual(collection.find.call_count, 1)


@unittest.skipUnless(_BACKEND_AVAILABLE, "backend dependencies not installed")
class TestListRunningTasksNoCache(unittest.TestCase):
    def test_running_comes_from_memory_only(self):
        """运行中任务只读内存，不碰缓存；缓存里的旧 RUNNING 记录不报。"""
        live = RunContext(
            id="LIVE01", description="running", status=RunStatus.RUNNING,
            start_time=_now(),
        )
        done = RunContext(
            id="DONE01", description="done", status=RunStatus.SUCCESS,
            start_time=_now(), end_time=_now(),
        )
        # 缓存里有一条陈旧的 RUNNING 记录（进程中途死掉留下的），不应再报成运行中
        stale = {"_id": "runinfo.STALE1", "value": _run_json(
            "STALE1", "stale running", status=RunStatus.RUNNING)}
        collection = MagicMock()
        collection.find.side_effect = _fake_find([stale])
        _patch_caches(self, collection)
        with patch.dict(runinfo._running_runs, {"LIVE01": live, "DONE01": done}, clear=True):
            client = _make_client()
            with client:
                resp = client.get("/api/runinfo/running")

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(collection.find.call_count, 0)
        self.assertEqual(collection.find_one.call_count, 0)
        body = resp.json()
        self.assertEqual(body["total_tasks"], 1)
        self.assertEqual([t["id"] for t in body["running_tasks"]], ["LIVE01"])

    def test_returns_immediately_when_cache_blocks(self):
        """缓存卡住时 GET /runinfo/running 也不等，直接返回内存任务。"""
        release = threading.Event()

        def blocked_find(*args, **kwargs):
            release.wait(timeout=30)
            return []

        live = RunContext(
            id="LIVE01", description="running", status=RunStatus.RUNNING,
            start_time=_now(),
        )
        collection = MagicMock()
        collection.find.side_effect = blocked_find
        _patch_caches(self, collection)
        with patch.dict(runinfo._running_runs, {"LIVE01": live}, clear=True):
            client = _make_client()
            with client:
                start = time.monotonic()
                try:
                    resp = client.get("/api/runinfo/running")
                finally:
                    release.set()

        # 不碰缓存，应远快于 1 秒硬超时
        self.assertLess(time.monotonic() - start, 2.0)
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["total_tasks"], 1)


@unittest.skipUnless(_BACKEND_AVAILABLE, "backend dependencies not installed")
class TestOtherRuninfoEndpointsBulk(unittest.TestCase):
    def _docs(self):
        return [
            {"_id": "runinfo.TOPAAA", "value": _run_json(
                "TOPAAA", "top one", start_hour=12, start_minute=0, log_message="top log")},
            {"_id": "runinfo.CHILD1", "value": _run_json(
                "CHILD1", "child of top one", parent_ids=["TOPAAA"],
                start_hour=12, start_minute=5, log_message="child log")},
            {"_id": "runinfo.TOPBBB", "value": _run_json(
                "TOPBBB", "top two", start_hour=12, start_minute=10, log_message="other log")},
            {"_id": "runinfo.children.TOPAAA", "value": '["CHILD1"]'},
        ]

    def test_get_logs_single_bulk_query(self):
        """GET /runinfo/logs 只发一次 find，且子任务日志按需包含。"""
        collection = MagicMock()
        collection.find.side_effect = _fake_find(self._docs())
        _patch_caches(self, collection)
        with patch.dict(runinfo._running_runs, {}, clear=True):
            client = _make_client()
            with client:
                resp = client.get("/api/runinfo/logs?limit=10&include_children=true")

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(collection.find.call_count, 1)
        self.assertEqual(collection.find_one.call_count, 0)
        messages = [item["message"] for item in resp.json()["logs"]]
        self.assertIn("top log", messages)
        self.assertIn("child log", messages)
        self.assertIn("other log", messages)

    def test_get_logs_without_children(self):
        """include_children=false 时父任务不额外带上子任务日志。

        子任务本身也是 runs 列表里的一条记录，它的日志本来就会出现；
        要区分 include_children，需让子任务被 limit 排除在 runs 之外，
        同时它的日志又新到能进 logs[:limit]：父任务最新、子任务日志
        次新、其它任务再次之，limit=2 只取到父任务和其它任务。
        """
        def ctx_with_log(run_id, description, parent_ids, start, log_time, message):
            return RunContext(
                id=run_id, description=description, status=RunStatus.SUCCESS,
                parent_ids=parent_ids, start_time=start, end_time=start,
                log=[LogRecord(level="INFO", message=message, time=log_time)],
            ).model_dump_json()

        docs = [
            {"_id": "runinfo.TOPAAA", "value": ctx_with_log(
                "TOPAAA", "top one", [], _now(12, 10), _now(12, 10), "top log")},
            {"_id": "runinfo.CHILD1", "value": ctx_with_log(
                "CHILD1", "child of top one", ["TOPAAA"], _now(12, 0), _now(12, 9), "child log")},
            {"_id": "runinfo.TOPBBB", "value": ctx_with_log(
                "TOPBBB", "top two", [], _now(12, 5), _now(12, 5), "other log")},
        ]
        collection = MagicMock()
        collection.find.side_effect = _fake_find(docs)
        _patch_caches(self, collection)
        with patch.dict(runinfo._running_runs, {}, clear=True):
            client = _make_client()
            with client:
                resp = client.get("/api/runinfo/logs?limit=2&include_children=false")

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(collection.find.call_count, 1)
        messages = [item["message"] for item in resp.json()["logs"]]
        self.assertEqual(messages, ["top log", "other log"])

        collection.find.reset_mock()
        with patch.dict(runinfo._running_runs, {}, clear=True):
            with client:
                resp = client.get("/api/runinfo/logs?limit=2&include_children=true")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(collection.find.call_count, 1)
        messages = [item["message"] for item in resp.json()["logs"]]
        self.assertEqual(messages, ["top log", "child log"])

    def test_list_children_runs_single_bulk_query(self):
        """GET /runinfo/{parent}/children 只发一次 find。"""
        collection = MagicMock()
        collection.find.side_effect = _fake_find(self._docs())
        _patch_caches(self, collection)
        with patch.dict(runinfo._running_runs, {}, clear=True):
            client = _make_client()
            with client:
                resp = client.get("/api/runinfo/TOPAAA/children")

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(collection.find.call_count, 1)
        self.assertEqual(collection.find_one.call_count, 0)
        runs = resp.json()["runs"]
        self.assertEqual([r["id"] for r in runs], ["CHILD1"])
        self.assertEqual(runs[0]["child_count"], 0)

    def test_get_run_and_parents_single_bulk_query(self):
        """GET /runinfo/{id} 与 /parents 各只发一次 find。"""
        collection = MagicMock()
        collection.find.side_effect = _fake_find(self._docs())
        _patch_caches(self, collection)
        with patch.dict(runinfo._running_runs, {}, clear=True):
            client = _make_client()
            with client:
                resp = client.get("/api/runinfo/CHILD1")
                self.assertEqual(resp.status_code, 200)
                self.assertEqual(collection.find.call_count, 1)
                self.assertEqual(collection.find_one.call_count, 0)
                self.assertEqual(resp.json()["run"]["id"], "CHILD1")

                collection.find.reset_mock()
                collection.find_one.reset_mock()
                resp = client.get("/api/runinfo/CHILD1/parents")
                self.assertEqual(resp.status_code, 200)
                self.assertEqual(collection.find.call_count, 1)
                self.assertEqual(collection.find_one.call_count, 0)
                self.assertEqual([r["id"] for r in resp.json()["runs"]], ["TOPAAA"])

    def test_get_run_logs_single_bulk_query(self):
        """GET /runinfo/{id}/logs 只发一次 find，且含子任务日志。"""
        collection = MagicMock()
        collection.find.side_effect = _fake_find(self._docs())
        _patch_caches(self, collection)
        with patch.dict(runinfo._running_runs, {}, clear=True):
            client = _make_client()
            with client:
                resp = client.get("/api/runinfo/TOPAAA/logs?include_children=true")

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(collection.find.call_count, 1)
        self.assertEqual(collection.find_one.call_count, 0)
        messages = [item["message"] for item in resp.json()["logs"]]
        self.assertIn("top log", messages)
        self.assertIn("child log", messages)

    def test_get_run_summary_single_bulk_query(self):
        """GET /runinfo/summary 只发一次 find。"""
        collection = MagicMock()
        collection.find.side_effect = _fake_find(self._docs())
        _patch_caches(self, collection)
        with patch.dict(runinfo._running_runs, {}, clear=True):
            client = _make_client()
            with client:
                resp = client.get("/api/runinfo/summary")

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(collection.find.call_count, 1)
        self.assertEqual(collection.find_one.call_count, 0)
        body = resp.json()
        self.assertEqual(body["total"], 2)
        self.assertEqual([r["id"] for r in body["recent"]], ["TOPBBB", "TOPAAA"])
        by_id = {r["id"]: r for r in body["recent"]}
        self.assertEqual(by_id["TOPAAA"]["child_count"], 1)
        self.assertEqual(by_id["TOPBBB"]["child_count"], 0)


if __name__ == "__main__":
    unittest.main()
