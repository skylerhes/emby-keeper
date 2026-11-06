"""Emby/Subsonic 账号列表批量取回缓存，不再逐账号逐条查。

背景: list_emby_accounts / list_subsonic_accounts 虽已只调一次
RunContext.list_all()，但 _serialize_account 里每个账号还要单独调
多次 cache.get：Emby 是站点标题、观看历史、设备信息各 1 次（每账号
3 次），每个 run 转字典时还要调一次 cache.get 取 child_count（每条
run 1 次）；Subsonic 是观看历史每账号 1 次、child_count 每条 run 1 次。
N 个账号就是 3N + N×M 次单独 find_one，每次都要等 MongoDB（VPS 到
Atlas 的 ping 就要 1.56 秒），串行累加很慢。
修复：三类键各一次 find_by_prefix_values 批量取回后内存查；
child_count 从已取回的 contexts 按 parent_ids 推算（api/runinfo.py
的 _index_contexts 就是这么做的）。列表接口的 MongoDB 查询次数从
随账号数/记录数增长变成常数级。
"""

import unittest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from embykeeper.cache import Cache
from embykeeper.runinfo import LogRecord, RunContext, RunStatus
from embykeeper.schema import Config, EmbyAccount, SubsonicAccount

import embykeeper.runinfo as runinfo


def _now():
    return datetime(2026, 11, 6, 12, 0, 0, tzinfo=timezone.utc)


def _run_json(run_id, description, parent_ids=None, log_message=None):
    logs = []
    if log_message is not None:
        logs = [LogRecord(level="INFO", message=log_message, time=_now())]
    return RunContext(
        id=run_id,
        description=description,
        status=RunStatus.SUCCESS,
        start_time=_now(),
        end_time=_now(),
        parent_ids=parent_ids or [],
        log=logs,
    ).model_dump_json()


def _collection_with_store(store):
    """MagicMock collection：find 按前缀正则过滤，find_one 按 _id 查。

    find/find_one 的调用次数由 MagicMock 自己计数；find 里不真的
    连 MongoDB，只读内存里的 store。
    """

    def _find(query, projection=None):
        pattern = query["_id"]["$regex"]
        prefix = pattern[1:] if pattern.startswith("^") else pattern
        return [{"_id": k, "value": v} for k, v in store.items() if k.startswith(prefix)]

    def _find_one(query):
        key = query["_id"]
        if key in store:
            return {"_id": key, "value": store[key]}
        return None

    collection = MagicMock()
    collection.find.side_effect = _find
    collection.find_one.side_effect = _find_one
    return collection


def _mongo_cache(collection):
    """构造一个走 MongoDB 路径的 Cache，不连真实 MongoDB。"""
    cache = Cache.__new__(Cache)
    cache._mongo_client = object()  # 仅作 truthy 标记
    cache._collection = collection
    return cache


def _make_client():
    from fastapi.testclient import TestClient

    from embykeeper.api import create_app
    from embykeeper.api.auth import require_auth

    app = create_app()
    app.dependency_overrides[require_auth] = lambda: True
    return TestClient(app, raise_server_exceptions=False)


class TestEmbyListBulkCache(unittest.TestCase):
    def _store(self):
        return {
            "runinfo.P1": _run_json("P1", "Emby 保活任务 - u1@127.0.0.1"),
            "runinfo.C1": _run_json("C1", "Emby 其它任务", parent_ids=["P1"]),
            "runinfo.P2": _run_json("P2", "Emby 保活任务 - u2@127.0.0.1"),
            # children 键应被 list_all 跳过，child_count 只按 parent_ids 推算
            "runinfo.children.P1": '["C1"]',
            "emby.site_title.127.0.0.1": "My Emby Server",
            "emby.account_history.127.0.0.1.u1": [
                {"status": "success", "time": "2026-11-06T10:00:00+00:00"}
            ],
            "emby.account_history.127.0.0.1.u2": [
                {"status": "failed", "time": "2026-11-06T09:00:00+00:00"}
            ],
            "emby.env.127.0.0.1.u1": {
                "client": "Emby Web",
                "device": "Chrome",
                "device_id": "dev1",
                "client_version": "4.8",
                "useragent": "Mozilla/5.0",
            },
            "emby.env.127.0.0.1.u2": {
                "client": "Emby Theater",
                "device": "TV",
                "device_id": "dev2",
            },
        }

    def _accounts(self):
        return [
            EmbyAccount(url="http://127.0.0.1:8096", username="u1", password="p1", id="a1"),
            EmbyAccount(url="http://127.0.0.1:8096", username="u2", password="p1", id="a2"),
            EmbyAccount(url="http://127.0.0.1:8096", username="u3", password="p1", id="a3"),
        ]

    def test_list_many_accounts_constant_mongo_queries(self):
        """3 个 Emby 账号时列表接口只发 4 次 find，find_one 为 0 次。"""
        import embykeeper.api.emby as emby_api

        collection = _collection_with_store(self._store())
        cache = _mongo_cache(collection)
        with (
            patch.object(runinfo, "cache", cache),
            patch.object(emby_api, "cache", cache),
            patch.dict(runinfo._running_runs, {}, clear=True),
            patch.object(
                emby_api, "_load_emby_accounts", return_value=(Config(), {}, self._accounts())
            ),
        ):
            client = _make_client()
            with client:
                resp = client.get("/api/emby/accounts")
        self.assertEqual(resp.status_code, 200)
        # 1 次 runinfo + 3 次三类键批量取回；不再逐账号逐条 find_one。
        self.assertEqual(collection.find.call_count, 4)
        self.assertEqual(collection.find_one.call_count, 0)
        accounts = resp.json()["accounts"]
        self.assertEqual(len(accounts), 3)
        by_id = {a["id"]: a for a in accounts}
        # 站点标题三账号共用同一 host，只取回一次却都正确对应。
        for item in accounts:
            self.assertEqual(item["site_title"], "My Emby Server")
            self.assertEqual(item["card_title"], "My Emby Server")
        # 观看历史各账号正确对应。
        self.assertEqual(
            by_id["a1"]["watch_history"],
            [{"status": "success", "time": "2026-11-06T10:00:00+00:00"}],
        )
        self.assertEqual(
            by_id["a2"]["watch_history"],
            [{"status": "failed", "time": "2026-11-06T09:00:00+00:00"}],
        )
        self.assertEqual(by_id["a3"]["watch_history"], [])
        # 设备信息各账号正确对应。
        self.assertEqual(
            by_id["a1"]["device_cache"],
            {
                "client": "Emby Web",
                "device": "Chrome",
                "device_id": "dev1",
                "client_version": "4.8",
                "useragent": "Mozilla/5.0",
            },
        )
        self.assertEqual(
            by_id["a2"]["device_cache"],
            {
                "client": "Emby Theater",
                "device": "TV",
                "device_id": "dev2",
                "client_version": None,
                "useragent": None,
            },
        )
        self.assertIsNone(by_id["a3"]["device_cache"])
        # child_count 从 parent_ids 推算：P1 有子任务 C1，P2 没有。
        self.assertEqual(by_id["a1"]["run_history"][0]["id"], "P1")
        self.assertEqual(by_id["a1"]["run_history"][0]["child_count"], 1)
        self.assertEqual(by_id["a2"]["run_history"][0]["id"], "P2")
        self.assertEqual(by_id["a2"]["run_history"][0]["child_count"], 0)
        self.assertEqual(by_id["a3"]["run_history"], [])
        # 子任务本身不出现在任何账号的 run_history 里。
        self.assertNotIn(
            "C1", [r["id"] for a in accounts for r in a["run_history"]]
        )

    def test_serialize_without_maps_falls_back_to_single_gets(self):
        """不传批量取回的字典时，单账号序列化回退到逐个 cache.get。"""
        import embykeeper.api.emby as emby_api

        account = EmbyAccount(url="http://127.0.0.1:8096", username="u1", password="p1", id="a1")
        ctx_parent = RunContext(
            id="P1",
            description="Emby 保活任务 - u1@127.0.0.1",
            status=RunStatus.SUCCESS,
            start_time=_now(),
            end_time=_now(),
            log=[],
        )
        ctx_child = RunContext(
            id="C1",
            description="Emby 其它任务",
            status=RunStatus.SUCCESS,
            start_time=_now(),
            end_time=_now(),
            parent_ids=["P1"],
            log=[],
        )
        store = {
            "emby.site_title.127.0.0.1": "My Emby Server",
            "emby.account_history.127.0.0.1.u1": [
                {"status": "success", "time": "2026-11-06T10:00:00+00:00"}
            ],
            "emby.env.127.0.0.1.u1": {"client": "Emby Web", "device": "Chrome"},
        }
        stub = MagicMock()
        stub.get.side_effect = lambda key, default=None: store.get(key, default)
        stub.find_by_prefix_values.side_effect = AssertionError(
            "find_by_prefix_values must not be called in fallback"
        )
        with (
            patch.object(RunContext, "list_all", return_value=[ctx_parent, ctx_child]),
            patch.object(emby_api, "cache", stub),
        ):
            item = emby_api._serialize_account(account)
        self.assertEqual(item["site_title"], "My Emby Server")
        self.assertEqual(item["card_title"], "My Emby Server")
        self.assertEqual(
            item["watch_history"],
            [{"status": "success", "time": "2026-11-06T10:00:00+00:00"}],
        )
        self.assertEqual(
            item["device_cache"],
            {
                "client": "Emby Web",
                "device": "Chrome",
                "device_id": None,
                "client_version": None,
                "useragent": None,
            },
        )
        self.assertEqual([r["id"] for r in item["run_history"]], ["P1"])
        self.assertEqual(item["run_history"][0]["child_count"], 1)
        # 回退时每个字段各调一次 get（站点标题在函数内只取一次），
        # 且没有批量取回；find_one 不存在于 stub，不用断言。
        self.assertEqual(stub.find_by_prefix_values.call_count, 0)
        self.assertEqual(stub.get.call_count, 3)

    def test_list_returns_200_when_bulk_fetch_fails(self):
        """批量取回抛错时列表仍返回 200，空值回退，不再调 find_one。"""
        import embykeeper.api.emby as emby_api

        collection = MagicMock()
        collection.find.side_effect = RuntimeError("Connection refused")
        cache = _mongo_cache(collection)
        with (
            patch.object(runinfo, "cache", cache),
            patch.object(emby_api, "cache", cache),
            patch.dict(runinfo._running_runs, {}, clear=True),
            patch.object(
                emby_api, "_load_emby_accounts", return_value=(Config(), {}, self._accounts())
            ),
        ):
            client = _make_client()
            with client:
                resp = client.get("/api/emby/accounts")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(collection.find_one.call_count, 0)
        accounts = resp.json()["accounts"]
        self.assertEqual(len(accounts), 3)
        for item in accounts:
            # 没取到站点标题时 site_title 为空、card_title 回退到 host，
            # 历史/设备为空。
            self.assertIsNone(item["site_title"])
            self.assertEqual(item["card_title"], "127.0.0.1")
            self.assertEqual(item["watch_history"], [])
            self.assertIsNone(item["device_cache"])
            self.assertEqual(item["run_history"], [])


class TestSubsonicListBulkCache(unittest.TestCase):
    def _store(self):
        return {
            "runinfo.Q1": _run_json("Q1", "Subsonic 保活任务 - s1@127.0.0.1"),
            "runinfo.D1": _run_json("D1", "Subsonic 其它任务", parent_ids=["Q1"]),
            "runinfo.Q2": _run_json("Q2", "Subsonic 保活任务 - s2@127.0.0.1"),
            "subsonic.account_history.127.0.0.1.s1": [
                {"status": "success", "time": "2026-11-06T10:00:00+00:00"}
            ],
            "subsonic.account_history.127.0.0.1.s2": [
                {"status": "failed", "time": "2026-11-06T09:00:00+00:00"}
            ],
        }

    def _accounts(self):
        return [
            SubsonicAccount(url="http://127.0.0.1:4040", username="s1", password="p1", id="s1"),
            SubsonicAccount(url="http://127.0.0.1:4040", username="s2", password="p1", id="s2"),
            SubsonicAccount(url="http://127.0.0.1:4040", username="s3", password="p1", id="s3"),
        ]

    def test_list_many_accounts_constant_mongo_queries(self):
        """3 个 Subsonic 账号时列表接口只发 2 次 find，find_one 为 0 次。"""
        import embykeeper.api.subsonic as subsonic_api

        collection = _collection_with_store(self._store())
        cache = _mongo_cache(collection)
        with (
            patch.object(runinfo, "cache", cache),
            patch.object(subsonic_api, "cache", cache),
            patch.dict(runinfo._running_runs, {}, clear=True),
            patch.object(
                subsonic_api,
                "_load_subsonic_accounts",
                return_value=(Config(), {}, self._accounts()),
            ),
        ):
            client = _make_client()
            with client:
                resp = client.get("/api/subsonic/accounts")
        self.assertEqual(resp.status_code, 200)
        # 1 次 runinfo + 1 次观看历史批量取回。
        self.assertEqual(collection.find.call_count, 2)
        self.assertEqual(collection.find_one.call_count, 0)
        accounts = resp.json()["accounts"]
        self.assertEqual(len(accounts), 3)
        by_id = {a["id"]: a for a in accounts}
        self.assertEqual(
            by_id["s1"]["watch_history"],
            [{"status": "success", "time": "2026-11-06T10:00:00+00:00"}],
        )
        self.assertEqual(
            by_id["s2"]["watch_history"],
            [{"status": "failed", "time": "2026-11-06T09:00:00+00:00"}],
        )
        self.assertEqual(by_id["s3"]["watch_history"], [])
        self.assertEqual(by_id["s1"]["run_history"][0]["id"], "Q1")
        self.assertEqual(by_id["s1"]["run_history"][0]["child_count"], 1)
        self.assertEqual(by_id["s2"]["run_history"][0]["id"], "Q2")
        self.assertEqual(by_id["s2"]["run_history"][0]["child_count"], 0)
        self.assertEqual(by_id["s3"]["run_history"], [])

    def test_serialize_without_maps_falls_back_to_single_gets(self):
        """不传批量取回的字典时，单账号序列化回退到逐个 cache.get。"""
        import embykeeper.api.subsonic as subsonic_api

        account = SubsonicAccount(
            url="http://127.0.0.1:4040", username="s1", password="p1", id="s1"
        )
        ctx_parent = RunContext(
            id="Q1",
            description="Subsonic 保活任务 - s1@127.0.0.1",
            status=RunStatus.SUCCESS,
            start_time=_now(),
            end_time=_now(),
            log=[],
        )
        ctx_child = RunContext(
            id="D1",
            description="Subsonic 其它任务",
            status=RunStatus.SUCCESS,
            start_time=_now(),
            end_time=_now(),
            parent_ids=["Q1"],
            log=[],
        )
        store = {
            "subsonic.account_history.127.0.0.1.s1": [
                {"status": "success", "time": "2026-11-06T10:00:00+00:00"}
            ],
        }
        stub = MagicMock()
        stub.get.side_effect = lambda key, default=None: store.get(key, default)
        stub.find_by_prefix_values.side_effect = AssertionError(
            "find_by_prefix_values must not be called in fallback"
        )
        with (
            patch.object(RunContext, "list_all", return_value=[ctx_parent, ctx_child]),
            patch.object(subsonic_api, "cache", stub),
        ):
            item = subsonic_api._serialize_account(account)
        self.assertEqual(
            item["watch_history"],
            [{"status": "success", "time": "2026-11-06T10:00:00+00:00"}],
        )
        self.assertEqual([r["id"] for r in item["run_history"]], ["Q1"])
        self.assertEqual(item["run_history"][0]["child_count"], 1)
        self.assertEqual(stub.find_by_prefix_values.call_count, 0)
        self.assertEqual(stub.get.call_count, 1)

    def test_list_returns_200_when_store_empty(self):
        """缓存为空时列表仍返回 200，空值回退，不再调 find_one。"""
        import embykeeper.api.subsonic as subsonic_api

        collection = _collection_with_store({})
        cache = _mongo_cache(collection)
        with (
            patch.object(runinfo, "cache", cache),
            patch.object(subsonic_api, "cache", cache),
            patch.dict(runinfo._running_runs, {}, clear=True),
            patch.object(
                subsonic_api,
                "_load_subsonic_accounts",
                return_value=(Config(), {}, self._accounts()),
            ),
        ):
            client = _make_client()
            with client:
                resp = client.get("/api/subsonic/accounts")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(collection.find.call_count, 2)
        self.assertEqual(collection.find_one.call_count, 0)
        accounts = resp.json()["accounts"]
        self.assertEqual(len(accounts), 3)
        for item in accounts:
            self.assertEqual(item["watch_history"], [])
            self.assertEqual(item["run_history"], [])


if __name__ == "__main__":
    unittest.main()
