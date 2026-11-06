"""Tests for the bulk run-info fetch (N+1 fix).

回归 2026-11-06: 硬超时（ab85765）把单次缓存调用限制在 3 秒，但
RunContext.list_all 是 N+1 查询——先 find_by_prefix 拿 keys，再对
每个 key 逐个调 cls.get（即一次 find_one）。py-spy 实测：主线程卡在
Future.result <- Cache.get <- RunContext.get <- RunContext.list_all，
工作线程 cache-mongo_0 卡在 SSLSocket.read <- find_one；N 条记录就
是 N 次 × 每人最多 3 秒，串行累加成几分钟，页面一直转圈。
修复：Cache.find_by_prefix_values 一次 find 取回全部键值，
list_all 不再逐条调 get。
"""

import pathlib
import tempfile
import threading
import time
import unittest
from unittest.mock import MagicMock, patch

from embykeeper.cache import Cache

try:
    from embykeeper.runinfo import RunContext
    import embykeeper.runinfo as runinfo

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


def _run_json(run_id, description="desc"):
    return RunContext(id=run_id, description=description).model_dump_json()


@unittest.skipUnless(_BACKEND_AVAILABLE, "backend dependencies not installed")
class TestListAllBulkQuery(unittest.TestCase):
    def test_single_find_and_no_find_one_for_many_records(self):
        """多条记录时 list_all 只发一次 find，find_one 为 0 次。"""
        docs = [
            {"_id": "runinfo.AAAAAA", "value": _run_json("AAAAAA", "first")},
            {"_id": "runinfo.BBBBBB", "value": _run_json("BBBBBB", "second")},
            {"_id": "runinfo.CCCCCC", "value": _run_json("CCCCCC", "third")},
            # children 键应被跳过
            {"_id": "runinfo.children.PARENT", "value": '["AAAAAA"]'},
            # 坏记录应被跳过，不影响其他记录
            {"_id": "runinfo.BADBAD", "value": "{not-json"},
        ]
        collection = MagicMock()
        collection.find.return_value = docs
        cache = _mongo_cache(collection)
        with (
            patch.object(runinfo, "cache", cache),
            patch.dict(runinfo._running_runs, {}, clear=True),
        ):
            contexts = RunContext.list_all()

        self.assertEqual(collection.find.call_count, 1)
        self.assertEqual(collection.find_one.call_count, 0)
        # find 一次取回 _id 和 value（两个位置参数：过滤条件、投影）
        self.assertEqual(
            collection.find.call_args.args[1], {"_id": 1, "value": 1}
        )
        got = {ctx.id: ctx for ctx in contexts}
        self.assertEqual(sorted(got), ["AAAAAA", "BBBBBB", "CCCCCC"])
        self.assertEqual(got["AAAAAA"].description, "first")
        self.assertEqual(got["BBBBBB"].description, "second")
        self.assertEqual(got["CCCCCC"].description, "third")

    def test_in_memory_run_wins_over_cached_copy(self):
        """内存中的运行任务优先，缓存里同 id 的记录不再重复返回。"""
        live = RunContext(id="AAAAAA", description="live")
        docs = [{"_id": "runinfo.AAAAAA", "value": _run_json("AAAAAA", "stale")}]
        collection = MagicMock()
        collection.find.return_value = docs
        cache = _mongo_cache(collection)
        with (
            patch.object(runinfo, "cache", cache),
            patch.dict(runinfo._running_runs, {"AAAAAA": live}, clear=True),
        ):
            contexts = RunContext.list_all()

        self.assertEqual(collection.find.call_count, 1)
        self.assertEqual(collection.find_one.call_count, 0)
        self.assertEqual(len(contexts), 1)
        self.assertIs(contexts[0], live)
        self.assertEqual(contexts[0].description, "live")

    def test_list_all_returns_quickly_when_bulk_find_blocks(self):
        """批量 find 卡住时 list_all 在约 3 秒内返回，只含内存任务。"""
        release = threading.Event()

        def blocked_find(*args, **kwargs):
            release.wait(timeout=30)
            return []

        live = RunContext(id="LIVE01", description="live")
        collection = MagicMock()
        collection.find.side_effect = blocked_find
        cache = _mongo_cache(collection)
        with (
            patch.object(runinfo, "cache", cache),
            patch.dict(runinfo._running_runs, {"LIVE01": live}, clear=True),
        ):
            start = time.monotonic()
            try:
                contexts = RunContext.list_all()
            finally:
                release.set()

        self.assertLess(time.monotonic() - start, 4.5)
        self.assertEqual(collection.find.call_count, 1)
        self.assertEqual(contexts, [live])

    def test_list_all_returns_empty_when_bulk_find_raises(self):
        """批量 find 抛错时 list_all 只返回内存任务（此处为空）。"""
        collection = MagicMock()
        collection.find.side_effect = RuntimeError("Connection refused")
        cache = _mongo_cache(collection)
        with (
            patch.object(runinfo, "cache", cache),
            patch.dict(runinfo._running_runs, {}, clear=True),
        ):
            self.assertEqual(RunContext.list_all(), [])


class TestFindByPrefixValues(unittest.TestCase):
    def test_mongo_returns_key_value_dict(self):
        """Mongo 路径一次 find 返回 {key: value}。"""
        import re

        all_docs = [
            {"_id": "runinfo.AAAAAA", "value": "v1"},
            {"_id": "runinfo.BBBBBB", "value": "v2"},
            {"_id": "other.X", "value": "v3"},
        ]

        def fake_find(filter_, projection):
            # 像 MongoDB 一样按 _id 前缀正则过滤
            pattern = filter_["_id"]["$regex"]
            return [d for d in all_docs if re.search(pattern, d["_id"])]

        collection = MagicMock()
        collection.find.side_effect = fake_find
        cache = _mongo_cache(collection)
        self.assertEqual(
            cache.find_by_prefix_values("runinfo."),
            {"runinfo.AAAAAA": "v1", "runinfo.BBBBBB": "v2"},
        )
        self.assertEqual(collection.find.call_count, 1)
        self.assertEqual(collection.find_one.call_count, 0)

    def test_mongo_returns_empty_dict_when_find_blocks(self):
        """find 卡住时 find_by_prefix_values 在约 3 秒后返回 {}。"""
        release = threading.Event()

        def blocked_find(*args, **kwargs):
            release.wait(timeout=30)
            return []

        collection = MagicMock()
        collection.find.side_effect = blocked_find
        cache = _mongo_cache(collection)
        start = time.monotonic()
        try:
            self.assertEqual(cache.find_by_prefix_values("runinfo."), {})
        finally:
            release.set()
        self.assertLess(time.monotonic() - start, 4.5)

    def test_mongo_returns_empty_dict_when_find_raises(self):
        """find 抛错时 find_by_prefix_values 返回 {}，不抛错。"""
        collection = MagicMock()
        collection.find.side_effect = RuntimeError("boom")
        cache = _mongo_cache(collection)
        self.assertEqual(cache.find_by_prefix_values("runinfo."), {})

    def test_json_path_returns_key_value_dict(self):
        """JSON 路径行为与 find_by_prefix 一致，但同时返回 value。"""
        with tempfile.TemporaryDirectory() as tmp:
            cache = _json_cache(pathlib.Path(tmp))
            cache.set("runinfo.AAAAAA", "v1")
            cache.set("runinfo.BBBBBB", "v2")
            cache.set("other.X", "v3")
            self.assertEqual(
                cache.find_by_prefix_values("runinfo."),
                {"runinfo.AAAAAA": "v1", "runinfo.BBBBBB": "v2"},
            )
            self.assertEqual(cache.find_by_prefix_values("missing."), {})


if __name__ == "__main__":
    unittest.main()
