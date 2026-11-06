"""Tests for the run-info retention policy (7 days).

回归 2026-11-06: 容器里实测 cache 集合 11517 条, 其中 runinfo 11317 条
(从未清理过), 一次取回全部要 1.86 秒, 数据转圈久; ping MongoDB 本身
就要 1.56 秒。find_by_prefix_values 一次取回全部 11317 条, 每条的
value 还是整份 RunContext JSON (含日志)。run_id 是随机 6 字符, 不是
时间有序的, 所以不能按 _id 排序取最新; start_time 存在 value 的 JSON
字符串里, MongoDB 里没有可查的时间字段。
修复: 运行记录只保留最近 7 天。list_all 取回后把 7 天前的旧记录批量
删掉 (以后不再越积越多); 另提供一次性清理脚本
embykeeper.prune_runinfo_cache, 先把存量旧记录清掉。
"""

import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

try:
    from embykeeper.runinfo import RunContext
    import embykeeper.runinfo as runinfo

    _BACKEND_AVAILABLE = True
except Exception:  # pragma: no cover - missing backend deps
    _BACKEND_AVAILABLE = False

try:
    from embykeeper.prune_runinfo_cache import (
        _is_stale,
        _parse_start_time,
        prune_old_runs,
    )

    _SCRIPT_AVAILABLE = True
except Exception:  # pragma: no cover - script import failed
    _SCRIPT_AVAILABLE = False


def _now():
    return datetime.now(timezone.utc)


def _run_json(run_id, description="desc", start_time=None):
    ctx = RunContext(id=run_id, description=description)
    if start_time is not None:
        ctx.start_time = start_time
    return ctx.model_dump_json()


@unittest.skipUnless(_BACKEND_AVAILABLE, "backend dependencies not installed")
class TestListAllRetention(unittest.TestCase):
    def test_old_records_deleted_recent_kept(self):
        """list_all 取回时, 7 天前的旧记录被删掉、最近的保留, 返回值正确。"""
        now = _now()
        records = {
            "runinfo.RECENT": _run_json("RECENT", "recent", now - timedelta(days=1)),
            "runinfo.OLD001": _run_json("OLD001", "old", now - timedelta(days=8)),
            "runinfo.OLD002": _run_json("OLD002", "old", now - timedelta(days=30)),
            "runinfo.NOSTAR": _run_json("NOSTAR", "no start time"),
            "runinfo.children.PARENT": '["RECENT"]',
            # 旧记录的父子关系键随父记录一起删; 保留的父记录的父子关系键不动。
            "runinfo.children.OLD001": '["OLD002"]',
            "runinfo.children.RECENT": '["NOSTAR"]',
        }
        mock_cache = MagicMock()
        mock_cache.find_by_prefix_values.return_value = records
        with (
            patch.object(runinfo, "cache", mock_cache),
            patch.dict(runinfo._running_runs, {}, clear=True),
        ):
            contexts = RunContext.list_all()

        returned_ids = {ctx.id for ctx in contexts}
        self.assertIn("RECENT", returned_ids)
        self.assertIn("NOSTAR", returned_ids)
        self.assertNotIn("OLD001", returned_ids)
        self.assertNotIn("OLD002", returned_ids)
        # 只删旧的 runinfo 记录及其父子关系键, 不动其它键。
        mock_cache.delete_many.assert_called_once()
        deleted_keys = mock_cache.delete_many.call_args.args[0]
        self.assertCountEqual(
            deleted_keys,
            ["runinfo.OLD001", "runinfo.OLD002", "runinfo.children.OLD001"],
        )

    def test_bad_json_not_deleted_and_return_unaffected(self):
        """旧记录解析失败 (坏 JSON) 时不影响返回、也不误删。"""
        now = _now()
        records = {
            "runinfo.RECENT": _run_json("RECENT", "recent", now - timedelta(days=1)),
            "runinfo.BADJSN": "not valid json {",
            "runinfo.OLD001": _run_json("OLD001", "old", now - timedelta(days=8)),
        }
        mock_cache = MagicMock()
        mock_cache.find_by_prefix_values.return_value = records
        with (
            patch.object(runinfo, "cache", mock_cache),
            patch.dict(runinfo._running_runs, {}, clear=True),
        ):
            contexts = RunContext.list_all()

        returned_ids = {ctx.id for ctx in contexts}
        self.assertIn("RECENT", returned_ids)
        self.assertNotIn("BADJSN", returned_ids)
        self.assertNotIn("OLD001", returned_ids)
        # 坏 JSON 的键不在删除列表里。
        deleted_keys = mock_cache.delete_many.call_args.args[0]
        self.assertCountEqual(deleted_keys, ["runinfo.OLD001"])

    def test_in_memory_old_task_not_deleted(self):
        """内存里的旧任务不删 (只删缓存里的旧记录)。"""
        now = _now()
        live_old = RunContext(id="LIVE01", description="live old")
        live_old.start_time = now - timedelta(days=30)
        records = {
            "runinfo.LIVE01": _run_json("LIVE01", "live old", now - timedelta(days=30)),
        }
        mock_cache = MagicMock()
        mock_cache.find_by_prefix_values.return_value = records
        with (
            patch.object(runinfo, "cache", mock_cache),
            patch.dict(runinfo._running_runs, {"LIVE01": live_old}, clear=True),
        ):
            contexts = RunContext.list_all()

        returned_ids = {ctx.id for ctx in contexts}
        self.assertIn("LIVE01", returned_ids)
        # 内存任务的同 id 缓存记录被跳过 (seen_ids), 所以不删。
        mock_cache.delete_many.assert_not_called()

    def test_delete_failure_does_not_block_return(self):
        """清理旧记录失败时 list_all 仍正常返回。"""
        now = _now()
        records = {
            "runinfo.RECENT": _run_json("RECENT", "recent", now - timedelta(days=1)),
            "runinfo.OLD001": _run_json("OLD001", "old", now - timedelta(days=8)),
        }
        mock_cache = MagicMock()
        mock_cache.find_by_prefix_values.return_value = records
        mock_cache.delete_many.side_effect = RuntimeError("mongo down")
        with (
            patch.object(runinfo, "cache", mock_cache),
            patch.dict(runinfo._running_runs, {}, clear=True),
        ):
            contexts = RunContext.list_all()

        returned_ids = {ctx.id for ctx in contexts}
        self.assertIn("RECENT", returned_ids)
        self.assertNotIn("OLD001", returned_ids)


@unittest.skipUnless(_SCRIPT_AVAILABLE, "prune script not importable")
class TestPruneScriptFiltering(unittest.TestCase):
    def test_parse_start_time(self):
        """清理脚本解析 start_time: 正常、缺失、坏 JSON、非法时间。"""
        now = _now()
        good = _run_json("AAAAAA", "x", now - timedelta(days=1))
        parsed = _parse_start_time(good)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.date(), (now - timedelta(days=1)).date())
        # 缺失 start_time
        self.assertIsNone(_parse_start_time(_run_json("BBBBBB", "x")))
        # 坏 JSON / 非 dict / 非法时间
        self.assertIsNone(_parse_start_time("not json {"))
        self.assertIsNone(_parse_start_time("[1, 2, 3]"))
        self.assertIsNone(_parse_start_time('{"start_time": "not-a-time"}'))

    def test_is_stale(self):
        """按 start_time 判断新旧: 早于 cutoff 为过期, 没有 start_time 的保留。"""
        now = _now()
        cutoff = now - timedelta(days=7)
        self.assertTrue(_is_stale(now - timedelta(days=8), cutoff))
        self.assertTrue(_is_stale(now - timedelta(days=30), cutoff))
        self.assertFalse(_is_stale(now - timedelta(days=1), cutoff))
        self.assertFalse(_is_stale(now, cutoff))
        self.assertFalse(_is_stale(None, cutoff))

    def test_prune_old_runs_deletes_only_stale(self):
        """清理脚本只删 7 天前的 runinfo 记录及其父子关系键, 不动其它键。"""
        now = _now()
        docs = [
            {"_id": "runinfo.RECENT", "value": _run_json("RECENT", "x", now - timedelta(days=1))},
            {"_id": "runinfo.OLD001", "value": _run_json("OLD001", "x", now - timedelta(days=8))},
            {"_id": "runinfo.OLD002", "value": _run_json("OLD002", "x", now - timedelta(days=30))},
            {"_id": "runinfo.NOSTAR", "value": _run_json("NOSTAR", "x")},
            {"_id": "runinfo.BADJSN", "value": "not json {"},
            {"_id": "runinfo.children.PARENT", "value": '["RECENT"]'},
            # 旧记录的父子关系键随父记录一起删; 保留的父记录的父子关系键不动。
            {"_id": "runinfo.children.OLD001", "value": '["OLD002"]'},
            {"_id": "runinfo.children.RECENT", "value": '["NOSTAR"]'},
        ]
        collection = MagicMock()
        collection.find.return_value = docs
        deleted, kept = prune_old_runs(collection, retention_days=7)

        self.assertEqual(deleted, 3)  # 2 条旧记录 + 1 个父子关系键
        self.assertEqual(kept, 3)  # RECENT + NOSTAR + BADJSN (坏 JSON 不删, 计为保留)
        collection.delete_many.assert_called_once()
        delete_arg = collection.delete_many.call_args.args[0]
        self.assertEqual(
            set(delete_arg["_id"]["$in"]),
            {"runinfo.OLD001", "runinfo.OLD002", "runinfo.children.OLD001"},
        )

    def test_prune_old_runs_no_stale_no_delete_call(self):
        """没有旧记录时不调 delete_many。"""
        now = _now()
        docs = [
            {"_id": "runinfo.RECENT", "value": _run_json("RECENT", "x", now - timedelta(days=1))},
        ]
        collection = MagicMock()
        collection.find.return_value = docs
        deleted, kept = prune_old_runs(collection, retention_days=7)

        self.assertEqual((deleted, kept), (0, 1))
        collection.delete_many.assert_not_called()


if __name__ == "__main__":
    unittest.main()
