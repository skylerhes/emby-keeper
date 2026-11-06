"""一次性清理运行记录缓存里的旧记录。

cache 里的 runinfo 从未清理, 已积到上万条 (2026-11-06 实测 11317 条),
一次取回全部要 1.86 秒, 面板数据转圈久。RunContext.list_all 现在每次
取回时自动删掉 7 天前的旧记录, 但存量旧记录要先用这个脚本清一次。

用法 (在容器里):
    docker exec embykeeper python -m embykeeper.prune_runinfo_cache

只依赖 pymongo 和标准库, 不读面板配置、不启动任何服务, 直接连
EK_MONGODB 指定的 MongoDB。保留期与
embykeeper.runinfo._RUNINFO_RETENTION_DAYS 保持一致 (7 天)。
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple

# 与 embykeeper.runinfo._RUNINFO_RETENTION_DAYS 保持一致。
RETENTION_DAYS = 7


def _parse_start_time(run_json: str) -> Optional[datetime]:
    """从 RunContext 的 JSON 字符串里取出 start_time, 解析失败返回 None。"""
    try:
        data = json.loads(run_json)
    except (ValueError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    raw = data.get("start_time")
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _is_stale(start_time: Optional[datetime], cutoff: datetime) -> bool:
    """start_time 早于 cutoff 即为过期; 没有 start_time 的保留。"""
    if start_time is None:
        return False
    return start_time < cutoff


def prune_old_runs(collection, retention_days: int = RETENTION_DAYS) -> Tuple[int, int]:
    """删掉 collection 里早于 retention_days 的 runinfo 记录。

    返回 (删掉的键数, 剩下的 runinfo 条数)。
    只删 cache 集合里 _id 以 runinfo. 开头的键: 旧的运行记录
    runinfo.{id}, 以及这些旧记录的父子关系键 runinfo.children.{id}。
    绝不碰配置 (配置在另一个集合) 和其它非 runinfo 的缓存键。
    删前打印要删的数量, 方便核对。
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
    docs = list(
        collection.find({"_id": {"$regex": "^runinfo."}}, {"_id": 1, "value": 1})
    )
    stale_ids = []
    kept = 0
    for doc in docs:
        key = doc.get("_id", "")
        if key.startswith("runinfo.children."):
            continue
        if _is_stale(_parse_start_time(doc.get("value") or ""), cutoff):
            stale_ids.append(key)
        else:
            kept += 1
    # 旧记录的父子关系键随父记录一起删掉。
    stale_id_set = {key.replace("runinfo.", "") for key in stale_ids}
    stale_children_keys = []
    for doc in docs:
        key = doc.get("_id", "")
        if key.startswith("runinfo.children."):
            parent_id = key.replace("runinfo.children.", "")
            if parent_id in stale_id_set:
                stale_children_keys.append(key)
    to_delete = stale_ids + stale_children_keys
    if to_delete:
        print(
            f"将删除 {len(to_delete)} 个键 "
            f"(运行记录 {len(stale_ids)} 条, 父子关系键 {len(stale_children_keys)} 个)..."
        )
        collection.delete_many({"_id": {"$in": to_delete}})
    return len(to_delete), kept


def main() -> int:
    mongodb_url = os.environ.get("EK_MONGODB", "").strip()
    if not mongodb_url:
        print("未设置 EK_MONGODB, 不知道连哪个 MongoDB, 已退出。", file=sys.stderr)
        return 1
    try:
        from pymongo import MongoClient
    except ImportError:
        print("没有安装 pymongo, 无法连接 MongoDB。", file=sys.stderr)
        return 1
    client = MongoClient(
        mongodb_url,
        serverSelectionTimeoutMS=5000,
        connectTimeoutMS=5000,
        socketTimeoutMS=30000,
    )
    # 与 embykeeper.cache.Cache 的 MongoDB 路径一致: embykeeper 库的 cache 集合。
    # 只删这个集合里 runinfo. 前缀的键, 绝不碰配置 (配置在另一个集合)。
    collection = client["embykeeper"]["cache"]
    deleted, kept = prune_old_runs(collection, RETENTION_DAYS)
    print(f"删掉 {deleted} 个键, 还剩 {kept} 条 runinfo 记录。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
