import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query
from loguru import logger
from pydantic import BaseModel, Field, ValidationError

from .auth import require_auth
from .config import _read_config_dict, _write_config_dict
from .runinfo import _index_contexts, _sort_datetime
from ..cache import cache
from ..schema import Config, EmbyAccount, format_errors
from ..config import config as config_manager
from ..runinfo import RunContext, RunStatus
from ..emby.api import Emby, EmbyError
from ..emby.main import EmbyManager

router = APIRouter()

EMBY_ACCOUNT_HISTORY_LIMIT = 30
EMBY_ACCOUNT_RUN_HISTORY_LIMIT = 20


class EmbyAccountPayload(BaseModel):
    url: str
    username: str
    password: str = ""
    name: Optional[str] = None
    enabled: bool = True
    use_proxy: bool = True
    allow_stream: bool = False
    cf_challenge: bool = True
    allow_multiple: bool = True
    play_id: Optional[str] = None
    time: List[int] = Field(default_factory=lambda: [300, 600])
    interval_days: Optional[str] = None
    time_range: Optional[str] = None


class EmbySettingsPayload(BaseModel):
    enabled: bool = True


class EmbyPlayIdProbePayload(BaseModel):
    url: str
    username: str
    password: str = ""
    name: Optional[str] = None
    use_proxy: bool = True
    cf_challenge: bool = True
    play_id: str


def _load_config() -> Config:
    return Config(**_read_config_dict())


def _generate_account_id(existing_ids: Set[str]) -> str:
    while True:
        account_id = uuid4().hex[:12]
        if account_id not in existing_ids:
            return account_id


def _account_key(account: EmbyAccount) -> str:
    return str(account.id or "")


def _account_signature(account: EmbyAccount) -> str:
    return f"{account.username}@{account.url}"


def _account_history_cache_key(account: EmbyAccount) -> str:
    return f"emby.account_history.{account.url.host}.{account.username}"


def _site_title_cache_key(account: EmbyAccount) -> str:
    return f"emby.site_title.{account.url.host}"


def _device_cache_key(account: EmbyAccount) -> str:
    return f"emby.env.{account.url.host}.{account.username}"


def _normalize_history_records(records: Any) -> List[Dict[str, str]]:
    normalized: List[Dict[str, str]] = []
    if not isinstance(records, list):
        return normalized

    for record in records:
        if not isinstance(record, dict):
            continue

        status = record.get("status")
        time_value = record.get("time")
        if status not in {"success", "failed"} or not isinstance(time_value, str):
            continue

        normalized.append({"status": status, "time": time_value})

    normalized.sort(key=lambda item: item["time"], reverse=True)
    return normalized[:EMBY_ACCOUNT_HISTORY_LIMIT]


def _hostname_from_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.hostname:
        return parsed.hostname
    return url.split("://", 1)[-1].split("/", 1)[0]


def _get_cached_site_title(
    account: EmbyAccount, site_titles: Optional[Dict[str, Any]] = None
) -> Optional[str]:
    if site_titles is not None:
        cached_title = site_titles.get(_site_title_cache_key(account))
    else:
        cached_title = cache.get(_site_title_cache_key(account))
    return cached_title if isinstance(cached_title, str) and cached_title.strip() else None


def _card_subtitle(account: EmbyAccount) -> str:
    return account.name or account.username


def _read_account_history(
    account: EmbyAccount, histories: Optional[Dict[str, Any]] = None
) -> List[Dict[str, str]]:
    if histories is not None:
        raw = histories.get(_account_history_cache_key(account), [])
    else:
        raw = cache.get(_account_history_cache_key(account), [])
    return _normalize_history_records(raw)


def _account_spec(account: EmbyAccount) -> str:
    return f"{account.username}@{account.name or account.url.host}"


def _account_run_descriptions(account: EmbyAccount) -> Set[str]:
    spec = _account_spec(account)
    return {
        f"Emby 保活任务 - {spec}",
        f"Emby 测试运行 - {spec}",
        "使用全局设置的 Emby 统一保活",
    }


def _account_log_target(account: EmbyAccount) -> str:
    return _account_spec(account)


def _match_account_run(ctx: RunContext, account: EmbyAccount) -> bool:
    if ctx.description not in _account_run_descriptions(account):
        return False

    if ctx.description != "使用全局设置的 Emby 统一保活":
        return True

    target = _account_log_target(account)
    for log in ctx.yield_logs(reverse=True):
        message = str(log.message or "")
        if target in message and (
            "保活成功" in message or "保活失败" in message or "正在登陆" in message or "成功登陆" in message
        ):
            return True
    return False


def _serialize_run_summary(
    ctx: RunContext, child_counts: Optional[Dict[str, int]] = None
) -> Dict[str, Any]:
    if child_counts is not None:
        # 列表接口已从取回的 contexts 里按 parent_ids 推算过子任务数，
        # 这里直接查字典，不再逐条调缓存。
        child_count = child_counts.get(ctx.id, 0)
    else:
        # 单条记录才走这里：只有一次缓存调用，不会累加。
        child_count = len(cache.get(f"runinfo.children.{ctx.id}", []) or [])
    return {
        "id": ctx.id,
        "description": ctx.description,
        "status": ctx.status.name if ctx.status is not None else None,
        "status_code": int(ctx.status) if ctx.status is not None else None,
        "status_info": ctx.status_info,
        "duration": ctx.duration,
        "start_time": ctx.start_time.isoformat() if ctx.start_time else None,
        "end_time": ctx.end_time.isoformat() if ctx.end_time else None,
        "parent_ids": ctx.parent_ids or [],
        "child_count": child_count,
    }


def _account_run_history(
    account: EmbyAccount,
    limit: int = EMBY_ACCOUNT_RUN_HISTORY_LIMIT,
    contexts: Optional[List[RunContext]] = None,
    child_counts: Optional[Dict[str, int]] = None,
) -> List[Dict[str, Any]]:
    all_contexts = contexts if contexts is not None else RunContext.list_all()
    if child_counts is None:
        _, _, child_counts = _index_contexts(all_contexts)
    contexts = [ctx for ctx in all_contexts if _match_account_run(ctx, account)]
    contexts.sort(key=lambda ctx: _sort_datetime(ctx.end_time or ctx.start_time), reverse=True)
    return [_serialize_run_summary(ctx, child_counts) for ctx in contexts[:limit]]


def _read_cached_device(
    account: EmbyAccount, devices: Optional[Dict[str, Any]] = None
) -> Optional[Dict[str, Any]]:
    if devices is not None:
        cached = devices.get(_device_cache_key(account), {})
    else:
        cached = cache.get(_device_cache_key(account), {})
    if not isinstance(cached, dict):
        return None

    cleaned = {
        "client": cached.get("client"),
        "device": cached.get("device"),
        "device_id": cached.get("device_id"),
        "client_version": cached.get("client_version"),
        "useragent": cached.get("useragent") or cached.get("ua"),
    }
    if not any(cleaned.values()):
        return None
    return cleaned


def _find_account_by_key(account_key: str) -> EmbyAccount:
    _, _, accounts = _load_emby_accounts()
    for account in accounts:
        if _account_key(account) == account_key:
            return account
    raise HTTPException(status_code=404, detail="Emby account not found")


async def _run_single_account(account: EmbyAccount, description: Optional[str] = None):
    manager = EmbyManager()
    test_account = account.model_copy(update={"enabled": True})
    result = await manager._watch_main(
        [test_account], instant=True, description=description, connect_only=True
    )
    if result is None:
        raise HTTPException(status_code=500, detail="Failed to start Emby test run")
    return result


def _record_account_watch_result(account: EmbyAccount, success: bool, run_time: Optional[datetime] = None):
    timestamp = (run_time or datetime.now()).isoformat()
    history = _read_account_history(account)
    history.insert(0, {"status": "success" if success else "failed", "time": timestamp})
    cache.set(_account_history_cache_key(account), history[:EMBY_ACCOUNT_HISTORY_LIMIT])


def _hydrate_account_history_from_runinfo(
    account: EmbyAccount,
    contexts: Optional[List[RunContext]] = None,
    histories: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, str]]:
    history = _read_account_history(account, histories)
    if len(history) >= EMBY_ACCOUNT_HISTORY_LIMIT:
        return history

    target_spec = f"{account.username}@{account.name or account.url.host}"
    collected = list(history)
    seen_times = {record["time"] for record in collected}

    all_contexts = contexts if contexts is not None else RunContext.list_all()
    contexts = sorted(
        all_contexts,
        key=lambda ctx: _sort_datetime(ctx.end_time or ctx.start_time),
        reverse=True,
    )

    for ctx in contexts:
        if len(collected) >= EMBY_ACCOUNT_HISTORY_LIMIT:
            break
        if ctx.description not in {"使用全局设置的 Emby 统一保活", f"Emby 保活任务 - {target_spec}"}:
            continue
        if ctx.status in {RunStatus.PENDING, RunStatus.INITIALIZING, RunStatus.RUNNING, RunStatus.CATAGORY}:
            continue

        logs = [log.message for log in ctx.yield_logs(reverse=True)]
        matched = False
        status = None
        for message in logs:
            if target_spec not in message:
                continue
            if "保活成功" in message:
                status = "success"
                matched = True
                break
            if "保活失败" in message:
                status = "failed"
                matched = True
                break

        if not matched:
            continue

        run_time = ctx.end_time or ctx.start_time
        if not run_time:
            continue

        time_value = run_time.isoformat()
        if time_value in seen_times:
            continue

        collected.append({"status": status, "time": time_value})
        seen_times.add(time_value)

    collected.sort(key=lambda item: item["time"], reverse=True)
    trimmed = collected[:EMBY_ACCOUNT_HISTORY_LIMIT]
    if trimmed:
        cache.set(_account_history_cache_key(account), trimmed)
    return trimmed


def _load_account_cache_maps() -> Dict[str, Dict[str, Any]]:
    """账号列表一次取回三类缓存：站点标题、观看历史、设备信息。

    原先每个账号各调 3 次 cache.get（每人一次 find_one），N 个账号
    就是 3N 次 MongoDB 查询，串行累加很慢。这里每类键一次批量取回，
    查询次数与账号数无关。注意不用 "emby." 一个前缀全取——那会连
    emby.credential.*（登录凭据）一起取回来，没必要也不安全。
    """
    return {
        "site_titles": cache.find_by_prefix_values("emby.site_title."),
        "histories": cache.find_by_prefix_values("emby.account_history."),
        "devices": cache.find_by_prefix_values("emby.env."),
    }


def _serialize_account(
    account: EmbyAccount,
    contexts: Optional[List[RunContext]] = None,
    cache_maps: Optional[Dict[str, Dict[str, Any]]] = None,
    child_counts: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    if contexts is None:
        contexts = RunContext.list_all()
    if child_counts is None:
        _, _, child_counts = _index_contexts(contexts)
    maps = cache_maps or {}
    # 站点标题只取一次，site_title 和 card_title 共用（之前各调一次）。
    site_title = _get_cached_site_title(account, maps.get("site_titles"))
    watch_history = _hydrate_account_history_from_runinfo(
        account, contexts, histories=maps.get("histories")
    )
    return {
        "id": _account_key(account),
        "url": str(account.url),
        "username": account.username,
        "name": account.name,
        "site_title": site_title,
        "card_title": site_title or account.url.host,
        "card_subtitle": _card_subtitle(account),
        "enabled": bool(account.enabled),
        "allow_stream": bool(account.allow_stream),
        "use_proxy": bool(account.use_proxy),
        "cf_challenge": bool(account.cf_challenge),
        "allow_multiple": bool(account.allow_multiple),
        "play_id": account.play_id,
        "time": account.time,
        "interval_days": str(account.interval_days) if account.interval_days is not None else None,
        "time_range": account.time_range,
        "watch_history": watch_history,
        "hostname": _hostname_from_url(str(account.url)),
        "device_cache": _read_cached_device(account, maps.get("devices")),
        "run_history": _account_run_history(account, contexts=contexts, child_counts=child_counts),
    }


def _replace_accounts(raw: Dict[str, Any], accounts: List[EmbyAccount]) -> Dict[str, Any]:
    merged = dict(raw)
    emby_section = dict(merged.get("emby") or {})
    emby_section["account"] = [account.model_dump(exclude_none=True) for account in accounts]
    if "concurrency" not in emby_section:
        emby_section["concurrency"] = 1
    merged["emby"] = emby_section
    return merged


def _ensure_emby_account_ids(
    model: Config, raw: Dict[str, Any], accounts: List[EmbyAccount]
) -> Tuple[Config, Dict[str, Any], List[EmbyAccount]]:
    existing_ids = {_account_key(account) for account in accounts if _account_key(account)}
    normalized: List[EmbyAccount] = []
    changed = False

    for account in accounts:
        if _account_key(account):
            normalized.append(account)
            continue

        account_id = _generate_account_id(existing_ids)
        existing_ids.add(account_id)
        normalized.append(account.model_copy(update={"id": account_id}))
        changed = True

    if not changed:
        return model, raw, accounts

    merged = _replace_accounts(raw, normalized)
    validated = Config(**merged)
    serialized = validated.model_dump(mode="json", exclude_none=True, exclude_unset=True)
    _write_config_dict(serialized)
    persisted_accounts = list(validated.emby.account if validated.emby and validated.emby.account else [])
    return validated, serialized, persisted_accounts


def _load_emby_accounts() -> Tuple[Config, Dict[str, Any], List[EmbyAccount]]:
    raw = _read_config_dict()
    model = Config(**raw)
    accounts = list(model.emby.account if model.emby and model.emby.account else [])
    return _ensure_emby_account_ids(model, raw, accounts)


@router.get("/emby/accounts")
async def list_emby_accounts(_: bool = Depends(require_auth)):
    try:
        cfg, _, accounts = _load_emby_accounts()
        items: List[Dict] = []
        summary = {
            "total": len(accounts),
            "enabled": 0,
            "use_proxy": 0,
            "allow_stream": 0,
            "cf_challenge": 0,
            "module_enabled": bool(cfg.emby.enabled) if cfg.emby and cfg.emby.enabled is not None else True,
            "concurrency": cfg.emby.concurrency if cfg.emby and cfg.emby.concurrency is not None else 1,
            "time_range": str(cfg.emby.time_range) if cfg.emby and cfg.emby.time_range is not None else None,
            "interval_days": (
                str(cfg.emby.interval_days) if cfg.emby and cfg.emby.interval_days is not None else None
            ),
        }
        # 运行记录只读一次传给各账号: 每个账号各读一次的话, 缓存出问题时
        # 延迟会按账号数叠加 (之前每个账号都要等一次 MongoDB 超时)。
        # 三类账号缓存（站点标题/观看历史/设备信息）也一样：每类键一次
        # 批量取回传给各账号，不再每个账号各调 3 次 cache.get；子任务数
        # 从取回的 contexts 按 parent_ids 推算，不再逐条调缓存。
        contexts = RunContext.list_all()
        cache_maps = _load_account_cache_maps()
        _, _, child_counts = _index_contexts(contexts)
        for a in accounts:
            status = _serialize_account(a, contexts, cache_maps=cache_maps, child_counts=child_counts)
            items.append(status)
            if status["enabled"]:
                summary["enabled"] += 1
            if status["use_proxy"]:
                summary["use_proxy"] += 1
            if status["allow_stream"]:
                summary["allow_stream"] += 1
            if status["cf_challenge"]:
                summary["cf_challenge"] += 1
        return {"accounts": items, "summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list emby accounts: {e}")


@router.post("/emby/settings")
async def save_emby_settings(payload: EmbySettingsPayload, _: bool = Depends(require_auth)):
    try:
        raw = _read_config_dict()
        merged = dict(raw)
        emby_section = dict(merged.get("emby") or {})
        emby_section["enabled"] = payload.enabled
        merged["emby"] = emby_section

        validated = Config(**merged)
        _write_config_dict(validated.model_dump(mode="json", exclude_none=True, exclude_unset=True))
        return {"success": True}
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=format_errors(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save emby settings: {e}")


@router.get("/emby/accounts/detail")
async def get_emby_account(key: str = Query(...), _: bool = Depends(require_auth)):
    try:
        account = _find_account_by_key(key)
        return {"account": _serialize_account(account)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load emby account: {e}")


@router.get("/emby/accounts/runs")
async def get_emby_account_runs(key: str = Query(...), limit: int = 20, _: bool = Depends(require_auth)):
    try:
        account = _find_account_by_key(key)
        safe_limit = max(1, min(limit, 100))
        return {"runs": _account_run_history(account, limit=safe_limit)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load emby account runs: {e}")


@router.get("/emby/accounts/device")
async def get_emby_account_device(key: str = Query(...), _: bool = Depends(require_auth)):
    try:
        account = _find_account_by_key(key)
        return {"device": _read_cached_device(account)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load emby account device cache: {e}")


@router.post("/emby/accounts/test-run")
async def start_emby_account_test_run(
    key: Optional[str] = Query(None),
    payload: Optional[EmbyAccountPayload] = None,
    _: bool = Depends(require_auth),
):
    try:
        if key:
            account = _find_account_by_key(key)
            if payload is not None:
                payload_data = payload.model_dump(exclude_none=True)
                if not payload_data.get("password"):
                    payload_data["password"] = account.password
                account = EmbyAccount(id=_account_key(account), **payload_data)
        elif payload is not None:
            account = EmbyAccount(id=_generate_account_id(set()), **payload.model_dump(exclude_none=True))
        else:
            raise HTTPException(status_code=400, detail="key or account payload required")

        description = f"Emby 测试运行 - {_account_spec(account)}"
        existing_ids = {ctx.id for ctx in RunContext.list_all()}
        task = asyncio.create_task(_run_single_account(account, description=description))

        run_id = None
        for _ in range(30):
            for ctx in RunContext.list_running():
                if ctx.id not in existing_ids and ctx.description == description:
                    run_id = ctx.id
                    break

            if run_id:
                break

            if task.done():
                break

            await asyncio.sleep(0.1)

        if not run_id and task.done() and not task.cancelled():
            result = task.result()
            run_id = result.id if result else None

        return {"success": True, "run_id": run_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start emby account test run: {e}")


@router.post("/emby/accounts/keepalive")
async def start_emby_account_keepalive(
    key: str = Query(...),
    _: bool = Depends(require_auth),
):
    """手动保活: 对单个 Emby 账号立即执行一次完整播放保活, 后台执行, 凭 run_id 轮询结果."""
    try:
        account = _find_account_by_key(key)
        if not account.enabled:
            raise HTTPException(status_code=400, detail="该账号未启用")

        description = f"手动保活 - {_account_spec(account)}"
        manager = EmbyManager()
        task = asyncio.create_task(
            manager._watch_main(
                [account.model_copy(update={"enabled": True})],
                instant=True,
                description=description,
            )
        )

        existing_ids = {ctx.id for ctx in RunContext.list_all()}
        run_id = None
        for _ in range(30):
            for ctx in RunContext.list_running():
                if ctx.id not in existing_ids and ctx.description == description:
                    run_id = ctx.id
                    break

            if run_id:
                break

            if task.done():
                break

            await asyncio.sleep(0.1)

        if not run_id and task.done() and not task.cancelled():
            result = task.result()
            run_id = result.id if result else None

        if not run_id:
            raise HTTPException(status_code=500, detail="保活任务已提交, 但暂时还没有可读取的运行记录")

        return {"success": True, "run_id": run_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"启动手动保活失败: {e}")


@router.post("/emby/accounts/probe-item")
async def probe_emby_play_item(
    payload: EmbyPlayIdProbePayload,
    key: Optional[str] = Query(None),
    _: bool = Depends(require_auth),
):
    """Log in with the supplied credentials and resolve the given play ID to an item.

    This is the lightweight counterpart of the test-run endpoint: it only verifies
    that the account can log in and that the configured video ID exists, without
    actually starting a playback task.
    """
    play_id = (payload.play_id or "").strip()
    if not play_id:
        raise HTTPException(status_code=400, detail="请先填写播放 ID")

    try:
        password = payload.password
        account_id: Optional[str] = None

        if key:
            existing = _find_account_by_key(key)
            account_id = _account_key(existing)
            if not password:
                password = existing.password

        if not account_id:
            account_id = _generate_account_id(set())

        account_data: Dict[str, Any] = {
            "id": account_id,
            "url": payload.url,
            "username": payload.username,
            "password": password,
            "use_proxy": payload.use_proxy,
            "cf_challenge": payload.cf_challenge,
            "play_id": play_id,
        }
        if payload.name:
            account_data["name"] = payload.name

        account = EmbyAccount(**account_data)

        emby = Emby(account)
        emby.log = logger.bind(server=account.name or emby.hostname, username=account.username)

        if not emby.user_id and not await emby.login():
            raise HTTPException(status_code=400, detail="登录失败, 无法访问该站点")

        item = await emby.get_item(play_id)
        if not item or not item.get("Id"):
            raise HTTPException(status_code=404, detail="未找到该视频项目, 请确认播放 ID 是否正确")

        try:
            duration_seconds = round(int(item.get("RunTimeTicks") or 0) / 10_000_000)
        except (TypeError, ValueError):
            duration_seconds = 0

        return {
            "success": True,
            "item": {
                "id": item.get("Id"),
                "name": item.get("Name"),
                "type": item.get("Type"),
                "series": item.get("SeriesName"),
                "year": item.get("ProductionYear"),
                "duration_seconds": duration_seconds,
            },
        }
    except HTTPException:
        raise
    except EmbyError as e:
        raise HTTPException(status_code=400, detail=f"测试播放 ID 失败: {e}")
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=format_errors(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"测试播放 ID 失败: {e}")


@router.post("/emby/accounts")
async def create_emby_account(payload: EmbyAccountPayload, _: bool = Depends(require_auth)):
    try:
        _, raw, accounts = _load_emby_accounts()

        if any(_account_signature(existing) == f"{payload.username}@{payload.url}" for existing in accounts):
            raise HTTPException(status_code=409, detail="Emby account already exists")

        existing_ids = {_account_key(existing) for existing in accounts if _account_key(existing)}
        account = EmbyAccount(id=_generate_account_id(existing_ids), **payload.model_dump(exclude_none=True))

        accounts.append(account)
        merged = _replace_accounts(raw, accounts)
        validated = Config(**merged)
        _write_config_dict(validated.model_dump(mode="json", exclude_none=True, exclude_unset=True))
        return {"success": True, "account": _serialize_account(account)}
    except HTTPException:
        raise
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=format_errors(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create emby account: {e}")


@router.put("/emby/accounts")
async def update_emby_account(
    key: str = Query(...), payload: EmbyAccountPayload = None, _: bool = Depends(require_auth)
):
    try:
        _, raw, accounts = _load_emby_accounts()
        updated = False
        next_accounts: List[EmbyAccount] = []
        next_account: Optional[EmbyAccount] = None

        for account in accounts:
            if _account_key(account) == key:
                payload_data = payload.model_dump(exclude_unset=True)
                if not payload_data.get("password"):
                    payload_data["password"] = account.password
                account_data = account.model_dump(exclude_none=False)
                account_data.update(payload_data)
                account_data["id"] = _account_key(account)
                next_account = EmbyAccount(**account_data)
                next_accounts.append(next_account)
                updated = True
            else:
                next_accounts.append(account)

        if not updated:
            raise HTTPException(status_code=404, detail="Emby account not found")

        if (
            sum(
                1
                for account in next_accounts
                if _account_signature(account) == _account_signature(next_account)
            )
            > 1
        ):
            raise HTTPException(status_code=409, detail="Emby account already exists")

        merged = _replace_accounts(raw, next_accounts)
        validated = Config(**merged)
        _write_config_dict(validated.model_dump(mode="json", exclude_none=True, exclude_unset=True))
        return {"success": True, "account": _serialize_account(next_account)}
    except HTTPException:
        raise
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=format_errors(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update emby account: {e}")


@router.delete("/emby/accounts")
async def delete_emby_account(key: str = Query(...), _: bool = Depends(require_auth)):
    try:
        _, raw, accounts = _load_emby_accounts()
        next_accounts = [account for account in accounts if _account_key(account) != key]

        if len(next_accounts) == len(accounts):
            raise HTTPException(status_code=404, detail="Emby account not found")

        merged = _replace_accounts(raw, next_accounts)
        validated = Config(**merged)
        _write_config_dict(validated.model_dump(mode="json", exclude_none=True, exclude_unset=True))
        return {"success": True}
    except HTTPException:
        raise
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=format_errors(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete emby account: {e}")
