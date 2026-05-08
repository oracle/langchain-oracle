from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import ChannelVersions, Checkpoint, CheckpointMetadata
from langgraph.checkpoint.base.id import uuid6


def generate_checkpoint(
    *,
    checkpoint_id: str | None = None,
    channel_values: dict[str, Any] | None = None,
    channel_versions: ChannelVersions | None = None,
    versions_seen: dict[str, ChannelVersions] | None = None,
) -> Checkpoint:
    return Checkpoint(
        v=1,
        id=checkpoint_id or str(uuid6(clock_seq=-1)),
        ts=datetime.now(timezone.utc).isoformat(),
        channel_values=channel_values if channel_values is not None else {},
        channel_versions=channel_versions if channel_versions is not None else {},
        versions_seen=versions_seen if versions_seen is not None else {},
        pending_sends=[],
        updated_channels=None,
    )


def generate_config(
    thread_id: str | None = None,
    *,
    checkpoint_ns: str = "",
    checkpoint_id: str | None = None,
) -> RunnableConfig:
    configurable: dict[str, Any] = {
        "thread_id": thread_id or str(uuid4()),
        "checkpoint_ns": checkpoint_ns,
    }
    if checkpoint_id is not None:
        configurable["checkpoint_id"] = checkpoint_id
    return {"configurable": configurable}


def generate_metadata(
    source: str = "loop",
    step: int = 0,
    **extra: Any,
) -> CheckpointMetadata:
    metadata: dict[str, Any] = {"source": source, "step": step, "parents": {}}
    metadata.update(extra)
    return metadata


def version_matches(actual: Any, expected: Any) -> bool:
    return str(actual).split(".")[0] == str(expected).split(".")[0]
