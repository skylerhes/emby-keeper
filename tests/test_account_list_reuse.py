"""Emby/Subsonic 账号列表只查一次运行记录并复用.

背景: list_emby_accounts / list_subsonic_accounts 原先逐账号调
RunContext.list_all(), 每个账号一次批量查询 (每次最多等一个 3 秒
硬超时), 账号一多就串行累加, 超时一次后后面的请求也跟着慢。
telegram.py 在 fdd3321 里已修成只调一次并复用; 这里把同一做法
用到 Emby / Subsonic / ops 概览上, 轮询新任务则改用只读内存的
RunContext.list_running(), 不再每次轮询都查 MongoDB。
"""

import unittest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from embykeeper.runinfo import LogRecord, RunContext, RunStatus
from embykeeper.schema import Config, EmbyAccount, SubsonicAccount, TelegramAccount


def _now():
    return datetime(2026, 11, 6, 12, 0, 0, tzinfo=timezone.utc)


def _make_cache_stub():
    """cache.get 返回传入的 default, set/delete 为空操作。"""
    stub = MagicMock()
    stub.get.side_effect = lambda key, default=None: default
    return stub


def _make_ctx(run_id, description, status=RunStatus.SUCCESS, log_message=None):
    logs = []
    if log_message is not None:
        logs = [LogRecord(level="INFO", message=log_message, time=_now())]
    return RunContext(
        id=run_id,
        description=description,
        status=status,
        start_time=_now(),
        end_time=_now(),
        log=logs,
    )


def _make_client():
    from fastapi.testclient import TestClient

    from embykeeper.api import create_app
    from embykeeper.api.auth import require_auth

    app = create_app()
    app.dependency_overrides[require_auth] = lambda: True
    return TestClient(app, raise_server_exceptions=False)


class TestEmbyListReuse(unittest.TestCase):
    def test_list_calls_list_all_once_for_many_accounts(self):
        """多个 Emby 账号时 list_emby_accounts 只调一次 list_all。"""
        import embykeeper.api.emby as emby_api

        a1 = EmbyAccount(url="http://127.0.0.1:8096", username="u1", password="p1", id="a1")
        a2 = EmbyAccount(url="http://127.0.0.1:8096", username="u2", password="p1", id="a2")
        a3 = EmbyAccount(url="http://127.0.0.1:8096", username="u3", password="p1", id="a3")
        ctx1 = _make_ctx("RUN001", "Emby 保活任务 - u1@127.0.0.1", log_message="u1@127.0.0.1 保活成功")
        ctx2 = _make_ctx("RUN002", "Emby 保活任务 - u2@127.0.0.1", log_message="u2@127.0.0.1 保活成功")
        ctx_other = _make_ctx("RUN999", "Emby 保活任务 - nobody@127.0.0.1")
        list_mock = MagicMock(return_value=[ctx1, ctx2, ctx_other])
        with (
            patch.object(RunContext, "list_all", list_mock),
            patch.object(emby_api, "_load_emby_accounts", return_value=(Config(), {}, [a1, a2, a3])),
            patch.object(emby_api, "cache", _make_cache_stub()),
        ):
            client = _make_client()
            with client:
                resp = client.get("/api/emby/accounts")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(list_mock.call_count, 1)
        accounts = resp.json()["accounts"]
        self.assertEqual(len(accounts), 3)
        by_id = {a["id"]: a for a in accounts}
        self.assertEqual([r["id"] for r in by_id["a1"]["run_history"]], ["RUN001"])
        self.assertEqual([r["id"] for r in by_id["a2"]["run_history"]], ["RUN002"])
        self.assertEqual(by_id["a3"]["run_history"], [])

    def test_account_run_history_uses_passed_contexts(self):
        """传入 contexts 时 _account_run_history 不调 list_all, 且正确过滤。"""
        import embykeeper.api.emby as emby_api

        account = EmbyAccount(url="http://127.0.0.1:8096", username="u1", password="p1", id="a1")
        ctx_match = _make_ctx("RUN001", "Emby 保活任务 - u1@127.0.0.1")
        ctx_other = _make_ctx("RUN002", "Emby 保活任务 - u2@127.0.0.1")
        with (
            patch.object(
                RunContext, "list_all", side_effect=AssertionError("list_all must not be called")
            ),
            patch.object(emby_api, "cache", _make_cache_stub()),
        ):
            runs = emby_api._account_run_history(account, contexts=[ctx_match, ctx_other])
        self.assertEqual([r["id"] for r in runs], ["RUN001"])

    def test_serialize_uses_passed_contexts_without_calling_list_all(self):
        """传入 contexts 时 _serialize_account 不调 list_all。"""
        import embykeeper.api.emby as emby_api

        account = EmbyAccount(url="http://127.0.0.1:8096", username="u1", password="p1", id="a1")
        ctx_match = _make_ctx("RUN001", "Emby 保活任务 - u1@127.0.0.1", log_message="u1@127.0.0.1 保活成功")
        ctx_other = _make_ctx("RUN002", "Emby 保活任务 - u2@127.0.0.1")
        with (
            patch.object(
                RunContext, "list_all", side_effect=AssertionError("list_all must not be called")
            ),
            patch.object(emby_api, "cache", _make_cache_stub()),
        ):
            item = emby_api._serialize_account(account, contexts=[ctx_match, ctx_other])
        self.assertEqual([r["id"] for r in item["run_history"]], ["RUN001"])

    def test_serialize_without_contexts_calls_list_all_once(self):
        """不传 contexts 时 _serialize_account 只调一次 list_all 并共享给两个子调用。"""
        import embykeeper.api.emby as emby_api

        account = EmbyAccount(url="http://127.0.0.1:8096", username="u1", password="p1", id="a1")
        ctx_match = _make_ctx("RUN001", "Emby 保活任务 - u1@127.0.0.1", log_message="u1@127.0.0.1 保活成功")
        list_mock = MagicMock(return_value=[ctx_match])
        with (
            patch.object(RunContext, "list_all", list_mock),
            patch.object(emby_api, "cache", _make_cache_stub()),
        ):
            item = emby_api._serialize_account(account)
        # _hydrate 和 _account_run_history 各需一次, 但 _serialize_account
        # 只取回一次并复用, 所以总共 1 次而不是 2 次。
        self.assertEqual(list_mock.call_count, 1)
        self.assertEqual([r["id"] for r in item["run_history"]], ["RUN001"])


class TestSubsonicListReuse(unittest.TestCase):
    def test_list_calls_list_all_once_for_many_accounts(self):
        """多个 Subsonic 账号时 list_subsonic_accounts 只调一次 list_all。"""
        import embykeeper.api.subsonic as subsonic_api

        a1 = SubsonicAccount(url="http://127.0.0.1:4040", username="s1", password="p1", id="s1")
        a2 = SubsonicAccount(url="http://127.0.0.1:4040", username="s2", password="p1", id="s2")
        a3 = SubsonicAccount(url="http://127.0.0.1:4040", username="s3", password="p1", id="s3")
        ctx1 = _make_ctx("RUN001", "Subsonic 保活任务 - s1@127.0.0.1")
        ctx2 = _make_ctx("RUN002", "Subsonic 保活任务 - s2@127.0.0.1")
        ctx_other = _make_ctx("RUN999", "Subsonic 保活任务 - nobody@127.0.0.1")
        list_mock = MagicMock(return_value=[ctx1, ctx2, ctx_other])
        with (
            patch.object(RunContext, "list_all", list_mock),
            patch.object(
                subsonic_api, "_load_subsonic_accounts", return_value=(Config(), {}, [a1, a2, a3])
            ),
            patch.object(subsonic_api, "cache", _make_cache_stub()),
        ):
            client = _make_client()
            with client:
                resp = client.get("/api/subsonic/accounts")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(list_mock.call_count, 1)
        accounts = resp.json()["accounts"]
        self.assertEqual(len(accounts), 3)
        by_id = {a["id"]: a for a in accounts}
        self.assertEqual([r["id"] for r in by_id["s1"]["run_history"]], ["RUN001"])
        self.assertEqual([r["id"] for r in by_id["s2"]["run_history"]], ["RUN002"])
        self.assertEqual(by_id["s3"]["run_history"], [])

    def test_account_run_history_uses_passed_contexts(self):
        """传入 contexts 时 _account_run_history 不调 list_all, 且正确过滤。"""
        import embykeeper.api.subsonic as subsonic_api

        account = SubsonicAccount(url="http://127.0.0.1:4040", username="s1", password="p1", id="s1")
        ctx_match = _make_ctx("RUN001", "Subsonic 保活任务 - s1@127.0.0.1")
        ctx_other = _make_ctx("RUN002", "Subsonic 保活任务 - s2@127.0.0.1")
        with (
            patch.object(
                RunContext, "list_all", side_effect=AssertionError("list_all must not be called")
            ),
            patch.object(subsonic_api, "cache", _make_cache_stub()),
        ):
            runs = subsonic_api._account_run_history(account, contexts=[ctx_match, ctx_other])
        self.assertEqual([r["id"] for r in runs], ["RUN001"])

    def test_serialize_without_contexts_calls_list_all_once(self):
        """不传 contexts 时 _serialize_account 只调一次 list_all。"""
        import embykeeper.api.subsonic as subsonic_api

        account = SubsonicAccount(url="http://127.0.0.1:4040", username="s1", password="p1", id="s1")
        ctx_match = _make_ctx("RUN001", "Subsonic 保活任务 - s1@127.0.0.1")
        list_mock = MagicMock(return_value=[ctx_match])
        with (
            patch.object(RunContext, "list_all", list_mock),
            patch.object(subsonic_api, "cache", _make_cache_stub()),
        ):
            item = subsonic_api._serialize_account(account)
        self.assertEqual(list_mock.call_count, 1)
        self.assertEqual([r["id"] for r in item["run_history"]], ["RUN001"])


class TestListRunning(unittest.TestCase):
    def test_list_running_returns_only_in_memory_runs(self):
        """list_running 只返回内存中的运行任务, 不碰缓存也不调 list_all。"""
        import embykeeper.runinfo as runinfo

        live1 = RunContext(id="LIVE01", description="live one")
        live2 = RunContext(id="LIVE02", description="live two")
        with (
            patch.dict(runinfo._running_runs, {"LIVE01": live1, "LIVE02": live2}, clear=True),
            patch.object(
                RunContext, "list_all", side_effect=AssertionError("list_all must not be called")
            ),
        ):
            running = RunContext.list_running()
        self.assertEqual({c.id for c in running}, {"LIVE01", "LIVE02"})


class TestOpsOverviewReuse(unittest.TestCase):
    def test_overview_fallback_reuses_all_contexts(self):
        """ops 概览里无当日记录时的回退不再逐账号调 list_all。"""
        import embykeeper.api.ops as ops_api

        t1 = TelegramAccount(phone="+8613800000001", api_id="123", api_hash="hash", checkiner=True)
        t2 = TelegramAccount(phone="+8613800000002", api_id="123", api_hash="hash", checkiner=True)
        cfg = Config(telegram={"account": [t1.model_dump(mode="json"), t2.model_dump(mode="json")]})
        list_mock = MagicMock(return_value=[])
        with (
            patch.object(RunContext, "list_all", list_mock),
            patch.object(ops_api, "_load_config", return_value=cfg),
            patch.object(ops_api, "cache", _make_cache_stub()),
        ):
            client = _make_client()
            with client:
                resp = client.get("/api/ops/overview")
        self.assertEqual(resp.status_code, 200)
        # 初始一次取回 + 两个账号的回退都复用它, 总共 1 次。
        self.assertEqual(list_mock.call_count, 1)


if __name__ == "__main__":
    unittest.main()
