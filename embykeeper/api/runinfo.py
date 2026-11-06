from datetime import datetime, timezone
from typing import Dict, List, Optional, Set
from pathlib import Path

import asyncio
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from ..cache import cache
from ..runinfo import RunContext, RunStatus

router = APIRouter()


EPOCH = datetime.min.replace(tzinfo=timezone.utc)


def _ensure_aware(value: Optional[datetime]) -> Optional[datetime]:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _sort_datetime(value: Optional[datetime]) -> datetime:
    if value is None:
        return EPOCH
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _parse_status_filter(status_param: Optional[str]) -> Optional[Set[RunStatus]]:
    """Parse comma-separated status names or int values into RunStatus set."""
    if not status_param:
        return None
    tokens = [t.strip() for t in status_param.split(",") if t and t.strip()]
    statuses: Set[RunStatus] = set()
    for token in tokens:
        # try enum name (case-insensitive)
        upper = token.upper()
        try:
            statuses.add(RunStatus[upper])
            continue
        except Exception:
            pass
        # try integer value
        try:
            num = int(token)
            statuses.add(RunStatus(num))
        except Exception:
            continue
    return statuses or None


def _effective_duration_seconds(ctx: RunContext) -> Optional[float]:
    """Return duration seconds, compute live duration for running tasks if needed."""
    if ctx.duration is not None:
        return ctx.duration
    if ctx.start_time and not ctx.end_time:
        start_time = _ensure_aware(ctx.start_time)
        if start_time is None:
            return None
        return (datetime.now(timezone.utc) - start_time).total_seconds()
    return None


def _is_reschedule(ctx: RunContext) -> bool:
    return (ctx.status == RunStatus.RESCHEDULE) or (
        ctx.reschedule is not None and isinstance(ctx.reschedule, int) and ctx.reschedule > 0
    )


def _context_matches_filters(
    ctx: RunContext,
    statuses: Optional[Set[RunStatus]],
    start_time_min: Optional[datetime],
    start_time_max: Optional[datetime],
    end_time_min: Optional[datetime],
    end_time_max: Optional[datetime],
    duration_min: Optional[float],
    duration_max: Optional[float],
    has_next_time: Optional[bool],
    is_reschedule: Optional[bool],
) -> bool:
    if statuses is not None and ctx.status not in statuses:
        return False

    if start_time_min is not None:
        ctx_start_time = _ensure_aware(ctx.start_time)
        if ctx_start_time is None or ctx_start_time < _ensure_aware(start_time_min):
            return False
    if start_time_max is not None:
        ctx_start_time = _ensure_aware(ctx.start_time)
        if ctx_start_time is None or ctx_start_time > _ensure_aware(start_time_max):
            return False

    if end_time_min is not None:
        ctx_end_time = _ensure_aware(ctx.end_time)
        if ctx_end_time is None or ctx_end_time < _ensure_aware(end_time_min):
            return False
    if end_time_max is not None:
        ctx_end_time = _ensure_aware(ctx.end_time)
        if ctx_end_time is None or ctx_end_time > _ensure_aware(end_time_max):
            return False

    eff_duration = _effective_duration_seconds(ctx)
    if duration_min is not None:
        if eff_duration is None or eff_duration < duration_min:
            return False
    if duration_max is not None:
        if eff_duration is None or eff_duration > duration_max:
            return False

    if has_next_time is not None:
        if has_next_time and ctx.next_time is None:
            return False
        if not has_next_time and ctx.next_time is not None:
            return False

    if is_reschedule is not None:
        if is_reschedule and not _is_reschedule(ctx):
            return False
        if not is_reschedule and _is_reschedule(ctx):
            return False

    return True


def _index_contexts(contexts: List[RunContext]):
    """给一次取回的上下文建索引：按 id、按父任务查子任务、子任务数。

    子任务关系直接从各任务的 parent_ids 推算，不再逐任务查缓存：
    prepare() 建子任务时同时写 parent_ids 和缓存里的 children 列表，
    两边是一致的；这里只用已取回的列表，列表接口就只有一次批量查询，
    超时时也最多等一个 3 秒硬超时。
    """
    by_id: Dict[str, RunContext] = {}
    children: Dict[str, List[RunContext]] = {}
    child_counts: Dict[str, int] = {}
    for ctx in contexts:
        by_id[ctx.id] = ctx
        for pid in (ctx.parent_ids or []):
            children.setdefault(pid, []).append(ctx)
            child_counts[pid] = child_counts.get(pid, 0) + 1
    return by_id, children, child_counts


def _context_to_dict(ctx: RunContext, child_counts: Optional[Dict[str, int]] = None) -> dict:
    if child_counts is not None:
        child_count = child_counts.get(ctx.id, 0)
    else:
        # 单条记录接口才走这里：只有一次缓存调用，不会累加。
        child_ids = cache.get(f"runinfo.children.{ctx.id}", []) or []
        child_count = len(child_ids)
    return {
        "id": ctx.id,
        "parent_ids": ctx.parent_ids or [],
        "description": ctx.description,
        "status": ctx.status.name if ctx.status is not None else None,
        "status_code": int(ctx.status) if ctx.status is not None else None,
        "status_info": ctx.status_info,
        "duration": _effective_duration_seconds(ctx),
        "start_time": ctx.start_time.isoformat() if ctx.start_time else None,
        "end_time": ctx.end_time.isoformat() if ctx.end_time else None,
        "next_time": ctx.next_time.isoformat() if ctx.next_time else None,
        "reschedule": ctx.reschedule,
        "has_next_time": ctx.next_time is not None,
        "is_reschedule": _is_reschedule(ctx),
        "child_count": child_count,
    }


def _context_log_items(ctx: RunContext) -> List[dict]:
    items: List[dict] = []
    for log in ctx.yield_logs(include_children=False):
        items.append(
            {
                "run_id": ctx.id,
                "parent_ids": ctx.parent_ids or [],
                "description": ctx.description,
                "level": log.level,
                "message": log.message,
                "time": log.time.isoformat() if log.time else None,
                "status": ctx.status.name if ctx.status else None,
            }
        )
    return items


def _all_contexts() -> List[RunContext]:
    # 直接用已批量取回的 list_all()：一次 find 拿全部记录（最多等一个
    # 3 秒硬超时），不要再 find_by_prefix 拿 keys 再逐个 RunContext.get。
    return RunContext.list_all()


@router.get("/runinfo")
async def list_top_level_runs(
    status: Optional[str] = Query(None, description="Comma-separated status names or numeric codes"),
    start_time_min: Optional[datetime] = None,
    start_time_max: Optional[datetime] = None,
    end_time_min: Optional[datetime] = None,
    end_time_max: Optional[datetime] = None,
    duration_min: Optional[float] = None,
    duration_max: Optional[float] = None,
    has_next_time: Optional[bool] = None,
    is_reschedule: Optional[bool] = None,
):
    try:
        statuses = _parse_status_filter(status)
        # 一次批量取回全部上下文再过滤，不要再逐 id 调 RunContext.get。
        # 子任务数按全量列表建索引：子任务本身不是顶层任务，过滤后再建
        # 索引会把 child_count 全算成 0。
        all_contexts = RunContext.list_all()
        _, _, child_counts = _index_contexts(all_contexts)
        contexts = [ctx for ctx in all_contexts if not ctx.parent_ids]

        filtered = [
            ctx
            for ctx in contexts
            if _context_matches_filters(
                ctx,
                statuses,
                start_time_min,
                start_time_max,
                end_time_min,
                end_time_max,
                duration_min,
                duration_max,
                has_next_time,
                is_reschedule,
            )
        ]

        filtered.sort(key=lambda c: _sort_datetime(c.start_time), reverse=True)
        return {"runs": [_context_to_dict(c, child_counts) for c in filtered]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list top-level runs: {e}")


@router.get("/runinfo/running")
async def list_running_tasks():
    try:
        # 只读内存的 list_running()，不碰缓存：运行中的任务从 prepare()
        # 起就在 _running_runs 里，finish() 时才移除并写入缓存；缓存里
        # 残留的 RUNNING 记录只能是进程中途死掉的旧数据，不应再报成运行中。
        running = []
        for ctx in RunContext.list_running():
            if ctx.status in [RunStatus.PENDING, RunStatus.INITIALIZING, RunStatus.RUNNING]:
                running.append(
                    {
                        "id": ctx.id,
                        "description": ctx.description,
                        "status": ctx.status.name,
                        "start_time": ctx.start_time.isoformat() if ctx.start_time else None,
                    }
                )
        return {"running_tasks": running, "total_tasks": len(running), "cache_status": "connected"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list running tasks: {e}")


@router.get("/runinfo/logs")
async def get_logs(
    limit: int = 100,
    include_children: bool = True,
    level: Optional[str] = None,
    keyword: Optional[str] = None,
):
    try:
        logs: List[dict] = []
        # 一次批量取回全部上下文，按开始时间倒序取最近 limit 条——以前是
        # 先 find_by_prefix 拿 keys 再逐个 get，key 的顺序本身是任意的，
        # 取“最后 limit 个 key”再按时间排并不能保证拿到最近的记录。
        contexts = RunContext.list_all()
        contexts.sort(key=lambda x: _sort_datetime(x.start_time), reverse=True)
        runs = contexts[:limit]
        _, children_by_parent, _ = _index_contexts(contexts)

        seen_log_keys: Set[tuple] = set()
        for run in runs:
            log_contexts = [run]
            if include_children:
                log_contexts.extend(children_by_parent.get(run.id, []))

            for log_ctx in log_contexts:
                for item in _context_log_items(log_ctx):
                    log_key = (item["run_id"], item["time"], item["level"], item["message"])
                    if log_key in seen_log_keys:
                        continue
                    seen_log_keys.add(log_key)

                    if level and str(item["level"] or "").upper() != level.upper():
                        continue

                    if keyword:
                        haystack = " ".join(
                            [
                                str(item["message"] or ""),
                                str(item["description"] or ""),
                                str(item["run_id"] or ""),
                            ]
                        ).lower()
                        if keyword.lower() not in haystack:
                            continue

                    logs.append(item)

        logs.sort(key=lambda x: x["time"] or "", reverse=True)
        return {"logs": logs[:limit]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get logs: {e}")


# 注意：固定路径的 /runinfo/summary 必须在 /runinfo/{run_id} 之前注册，
# 否则 "summary" 会被当成 run_id 吃掉，汇总接口永远 404（原版 bug）。
@router.get("/runinfo/summary")
async def get_run_summary():
    try:
        all_contexts = _all_contexts()
        top_level_runs = [ctx for ctx in all_contexts if not ctx.parent_ids]
        top_level_runs.sort(key=lambda ctx: _sort_datetime(ctx.start_time), reverse=True)
        _, _, child_counts = _index_contexts(all_contexts)

        summary = {
            "total": len(top_level_runs),
            "running": 0,
            "success": 0,
            "failed": 0,
            "cancelled": 0,
            "recent": [],
        }

        for ctx in top_level_runs:
            status_name = ctx.status.name if ctx.status else "UNKNOWN"
            if status_name in {"PENDING", "INITIALIZING", "RUNNING"}:
                summary["running"] += 1
            elif status_name == "SUCCESS":
                summary["success"] += 1
            elif status_name == "CANCELLED":
                summary["cancelled"] += 1
            else:
                summary["failed"] += 1

        summary["recent"] = [_context_to_dict(ctx, child_counts) for ctx in top_level_runs[:10]]
        return summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get run summary: {e}")


@router.get("/runinfo/{run_id}")
async def get_run(run_id: str):
    try:
        # 单条记录也从一次批量取回里查，不要再 find_one + 再查 children。
        by_id, _, child_counts = _index_contexts(RunContext.list_all())
        ctx = by_id.get(run_id)
        if not ctx:
            raise HTTPException(status_code=404, detail="Run not found")
        return {"run": _context_to_dict(ctx, child_counts)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get run: {e}")


@router.get("/runinfo/{parent_id}/children")
async def list_children_runs(
    parent_id: str,
    status: Optional[str] = Query(None, description="Comma-separated status names or numeric codes"),
    start_time_min: Optional[datetime] = None,
    start_time_max: Optional[datetime] = None,
    end_time_min: Optional[datetime] = None,
    end_time_max: Optional[datetime] = None,
    duration_min: Optional[float] = None,
    duration_max: Optional[float] = None,
    has_next_time: Optional[bool] = None,
    is_reschedule: Optional[bool] = None,
):
    try:
        # 子任务直接从一次批量取回的上下文里按 parent_ids 找，不再查缓存。
        # 运行中的子任务也在 list_all() 的内存部分里，不会漏掉。
        contexts = RunContext.list_all()
        _, children_by_parent, child_counts = _index_contexts(contexts)
        contexts = children_by_parent.get(parent_id, [])

        statuses = _parse_status_filter(status)
        filtered = [
            ctx
            for ctx in contexts
            if _context_matches_filters(
                ctx,
                statuses,
                start_time_min,
                start_time_max,
                end_time_min,
                end_time_max,
                duration_min,
                duration_max,
                has_next_time,
                is_reschedule,
            )
        ]

        filtered.sort(key=lambda c: _sort_datetime(c.start_time), reverse=True)
        return {"runs": [_context_to_dict(c, child_counts) for c in filtered]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list children runs: {e}")


@router.get("/runinfo/logs/stream")
async def stream_logs(
    include_children: bool = True, level: Optional[str] = None, keyword: Optional[str] = None
):
    async def event_generator():
        import asyncio

        last_time = None
        while True:
            try:
                # reuse get_logs logic with higher limit
                logs_data = await get_logs(
                    limit=200,
                    include_children=include_children,
                    level=level,
                    keyword=keyword,
                )
                logs = logs_data.get("logs", [])
                # filter new logs by time
                new = []
                for l in logs:
                    t = l.get("time") or ""
                    if not last_time or t > last_time:
                        new.append(l)
                if new:
                    last_time = new[0].get("time") or last_time
                    for l in reversed(new):
                        yield f"data: {__import__('json').dumps(l, ensure_ascii=False)}\n\n"
                await asyncio.sleep(2)
            except asyncio.CancelledError:
                break
            except Exception:
                await asyncio.sleep(2)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.get("/runinfo/{run_id}/logs")
async def get_run_logs(run_id: str, include_children: bool = True, limit: Optional[int] = None):
    try:
        # 先一次批量取回再按 id 查，单条记录也不再单独 find_one。
        by_id, children_by_parent, _ = _index_contexts(RunContext.list_all())
        ctx = by_id.get(run_id)
        if not ctx:
            raise HTTPException(status_code=404, detail="Run not found")
        items: List[dict] = []
        # 子任务从已取回的上下文里按 parent_ids 找，不要再逐个 get。
        log_contexts = [ctx]
        if include_children:
            log_contexts.extend(children_by_parent.get(ctx.id, []))

        for log_ctx in log_contexts:
            items.extend(_context_log_items(log_ctx))
        items.sort(key=lambda x: x["time"] or "", reverse=True)
        if limit is not None and limit > 0:
            items = items[:limit]
        return {"run_id": run_id, "logs": items}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get run logs: {e}")


@router.get("/runinfo/{run_id}/parents")
async def list_parents(run_id: str):
    try:
        # 先一次批量取回再按 id 查，单条记录也不再单独 find_one。
        by_id, _, child_counts = _index_contexts(RunContext.list_all())
        ctx = by_id.get(run_id)
        if not ctx:
            raise HTTPException(status_code=404, detail="Run not found")
        # 父任务从已取回的上下文里按 id 查，不要再逐个 RunContext.get。
        parents = [by_id[pid] for pid in (ctx.parent_ids or []) if pid in by_id]
        return {"runs": [_context_to_dict(p, child_counts) for p in parents]}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list parents: {e}")


@router.post("/runinfo/{run_id}/cancel")
async def cancel_run(run_id: str):
    try:
        ctx = RunContext.get(run_id)
        if not ctx:
            raise HTTPException(status_code=404, detail="Run not found")
        ctx.cancel_tree()
        return {"success": True}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to cancel run: {e}")


@router.post("/runinfo/cancel_all")
async def cancel_all_runs():
    try:
        RunContext.cancel_all()
        return {"success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to cancel all runs: {e}")


@router.post("/runinfo/cache/clear")
async def clear_cache():
    try:
        keys = cache.find_by_prefix("runinfo.")
        cache.delete_many(keys)
        return {"success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear cache: {e}")


def _active_config_file(config_manager) -> Path:
    conf_file = getattr(config_manager, "_conf_file", None)
    if conf_file:
        return Path(conf_file)
    return config_manager.basedir / "config.toml"


async def _trigger_checkin():
    from ..telegram.checkin_main import CheckinerManager
    from ..config import config as config_manager

    # 触发前确保 config_manager 与 config.toml 同步，避免读到已删除的账号
    await config_manager.reload_conf(_active_config_file(config_manager))

    manager = CheckinerManager()
    ctx = RunContext.prepare(description="立即触发签到")
    ctx.start(RunStatus.INITIALIZING)
    accounts = [a for a in (config_manager.telegram.account or []) if a.enabled and a.checkiner]
    if not accounts:
        ctx.finish(RunStatus.NONEED, "没有启用签到的账号")
        return ctx

    tasks = []
    for account in accounts:
        account_ctx = RunContext.prepare(
            description=f"{account.phone} 账号签到",
            parent_ids=[ctx.id],
        )
        account_ctx.start(RunStatus.RUNNING)
        task = manager._start_account_task(manager._ensure_state(account), account_ctx, instant=True)
        if task:
            tasks.append(task)

    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)

    ctx.finish(RunStatus.SUCCESS, "签到任务已触发")
    return ctx


async def _trigger_emby_keepalive():
    from ..emby.main import EmbyManager
    from ..config import config as config_manager

    # 触发前确保 config_manager 与 config.toml 同步，避免读到已删除的账号
    await config_manager.reload_conf(_active_config_file(config_manager))

    manager = EmbyManager()
    accounts = [a for a in (config_manager.emby.account or []) if a.enabled]
    if not accounts:
        ctx = RunContext.prepare(description="立即触发 Emby 保活")
        ctx.start(RunStatus.INITIALIZING)
        ctx.finish(RunStatus.NONEED, "没有启用的 Emby 账号")
        return ctx

    return await manager._watch_main(accounts, instant=True, description="立即触发 Emby 保活")


async def _trigger_subsonic_keepalive():
    from ..subsonic.main import SubsonicManager
    from ..config import config as config_manager

    # 触发前确保 config_manager 与 config.toml 同步，避免读到已删除的账号
    await config_manager.reload_conf(_active_config_file(config_manager))

    manager = SubsonicManager()
    accounts = [a for a in (config_manager.subsonic.account or []) if a.enabled]
    if not accounts:
        ctx = RunContext.prepare(description="立即触发 Subsonic 保活")
        ctx.start(RunStatus.INITIALIZING)
        ctx.finish(RunStatus.NONEED, "没有启用的 Subsonic 账号")
        return ctx

    return await manager.run_all(instant=True)


@router.post("/runinfo/trigger")
async def trigger_immediate(
    type: str = Query("all", description="触发类型: checkin, emby, subsonic, all"),
):
    try:
        valid_types = {"checkin", "emby", "subsonic", "all"}
        if type not in valid_types:
            raise HTTPException(status_code=400, detail=f"Invalid type: {type}. Must be one of {valid_types}")

        run_ids = []

        async def _run_and_collect(coro):
            ctx = await coro
            if ctx:
                run_ids.append(ctx.id)

        tasks = []
        if type in ("checkin", "all"):
            tasks.append(_run_and_collect(_trigger_checkin()))
        if type in ("emby", "all"):
            tasks.append(_run_and_collect(_trigger_emby_keepalive()))
        if type in ("subsonic", "all"):
            tasks.append(_run_and_collect(_trigger_subsonic_keepalive()))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        return {"success": True, "run_ids": run_ids}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to trigger: {e}")
