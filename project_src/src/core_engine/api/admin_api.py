"""核心引擎对外接口背后的内部辅助函数。"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.core_engine.api.contracts import EngineStatus


_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _get_attr(target: Any, name: str, default: Any = None) -> Any:
    if hasattr(target, name):
        return getattr(target, name)
    persona = getattr(target, "_persona", None)
    if persona is not None and hasattr(persona, name):
        return getattr(persona, name)
    return default


def _is_hierarchical_memory(memory: Any) -> bool:
    return (
        memory is not None
        and hasattr(memory, "working_memory")
        and hasattr(memory, "_memory_lock")
        and hasattr(memory, "get_system_context")
    )


def _recount_l1_tokens(turns: List[Any]) -> int:
    try:
        from src.memory.memory_manager import count_tokens, turn_to_text

        return sum(count_tokens(turn_to_text(turn)) for turn in turns)
    except Exception:
        total = 0
        for turn in turns:
            total += len(getattr(turn, "user_input", "") or "")
            total += len(getattr(turn, "response", "") or "")
        return total


def get_persona_name(target: Any) -> str:
    personality = _get_attr(target, "personality")
    if personality is not None and hasattr(personality, "name"):
        return personality.name
    return "unknown"


def get_llm_model(target: Any) -> Optional[str]:
    llm_client = _get_attr(target, "llm_client")
    return getattr(llm_client, "model", None)


def get_memory(target: Any) -> Any:
    return _get_attr(target, "memory")


def get_attention_tracker(target: Any) -> Any:
    return _get_attr(target, "attention_tracker")


def get_emotion_tracker(target: Any) -> Any:
    return _get_attr(target, "emotion_state_tracker")


def snapshot_recent_turns(
    target: Any,
    *,
    limit: int = 8,
    context_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """提取最近若干轮对话，供主动决策或状态查询使用。"""
    memory = get_memory(target)
    if memory is None:
        return []

    turns = list(getattr(memory, "working_memory", getattr(memory, "short_term", [])))
    if context_id is not None:
        turns = [turn for turn in turns if getattr(turn, "context_id", None) == context_id]

    recent = turns[-limit:] if limit > 0 else turns
    return [
        {
            "user_input": getattr(turn, "user_input", ""),
            "response": getattr(turn, "response", ""),
            "emotion": getattr(turn, "emotion", ""),
            "intensity": getattr(turn, "intensity", 0.0),
            "behavior": getattr(turn, "behavior", ""),
            "tone": getattr(turn, "tone", ""),
            "timestamp": getattr(turn, "timestamp", None),
            "context_id": getattr(turn, "context_id", None),
        }
        for turn in recent
    ]


def snapshot_emotion_state(target: Any) -> Dict[str, Any]:
    """读取当前人格的情绪状态快照。"""
    tracker = get_emotion_tracker(target)
    if tracker is not None and hasattr(tracker, "to_dict"):
        try:
            return tracker.to_dict()
        except Exception:
            pass

    memory = get_memory(target)
    if _is_hierarchical_memory(memory) and hasattr(memory, "config"):
        path = Path(memory.config.vector_store_path).parent / "emotion_state.json"
    else:
        path = _PROJECT_ROOT / "data" / "emotion_state.json"

    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def search_l4_memories(
    target: Any,
    *,
    query: str = "情感 情绪 心情",
    limit: int = 3,
) -> List[str]:
    """搜索与当前查询相关的 L4 长期记忆。"""
    memory = get_memory(target)
    if memory is None or not hasattr(memory, "mem0"):
        return []

    user_id = getattr(getattr(memory, "config", None), "user_id", None) or getattr(
        memory,
        "user_id",
        None,
    )
    if not user_id:
        return []

    try:
        results = memory.mem0.search(query=query, limit=limit, filters={"user_id": user_id})
    except Exception:
        return []

    raw_items = results.get("results", results) if isinstance(results, dict) else results
    memories: List[str] = []
    for item in raw_items or []:
        if isinstance(item, str):
            memories.append(item)
        elif isinstance(item, dict) and "memory" in item:
            memories.append(item["memory"])
    return memories


def clear_context_memory(target: Any, context_id: Optional[str] = None) -> int:
    """按 context_id 清理工作记忆。None 表示清空全部。"""
    memory = get_memory(target)
    if memory is None:
        return 0

    if _is_hierarchical_memory(memory):
        with memory._memory_lock:
            original = list(memory.working_memory)
            if context_id is None:
                kept = []
            else:
                kept = [
                    turn
                    for turn in original
                    if getattr(turn, "context_id", None) != context_id
                ]
            cleared = len(original) - len(kept)
            if cleared <= 0:
                return 0
            memory.working_memory = kept
            if hasattr(memory, "_l1_tokens"):
                memory._l1_tokens = _recount_l1_tokens(kept)
            return cleared

    if hasattr(memory, "short_term"):
        original = list(memory.short_term)
        if context_id is None:
            kept = []
        else:
            kept = [
                turn for turn in original if getattr(turn, "context_id", None) != context_id
            ]
        cleared = len(original) - len(kept)
        if cleared > 0:
            memory.short_term = kept
        return cleared

    if hasattr(memory, "working_memory"):
        original = list(memory.working_memory)
        if context_id is None:
            kept = []
        else:
            kept = [
                turn
                for turn in original
                if getattr(turn, "context_id", None) != context_id
            ]
        cleared = len(original) - len(kept)
        if cleared > 0:
            memory.working_memory = kept
        return cleared

    return 0


def build_engine_status(
    target: Any,
    *,
    loop: Any = None,
    contexts: Optional[List[str]] = None,
    patterns: Optional[List[str]] = None,
    config_source: str = "",
    uptime_seconds: Optional[int] = None,
) -> EngineStatus:
    """构造对外暴露的统一状态对象。"""
    memory = get_memory(target)
    attention_tracker = get_attention_tracker(target)
    attention = {}
    if attention_tracker is not None and hasattr(attention_tracker, "get_status"):
        try:
            attention = attention_tracker.get_status()
        except Exception:
            attention = {}

    working_memory = getattr(memory, "working_memory", getattr(memory, "short_term", []))
    idle_seconds = None
    queue_size = None
    running = False
    proactive_state = None
    if loop is not None:
        running = bool(getattr(loop, "running", False))
        proactive = getattr(loop, "proactive_state", None)
        proactive_state = getattr(proactive, "value", None)
        last_user_time = getattr(loop, "last_user_time", None)
        if last_user_time:
            idle_seconds = max(0, int(time.time() - last_user_time))
        event_queue = getattr(loop, "event_queue", None)
        if event_queue is not None and hasattr(event_queue, "qsize"):
            queue_size = event_queue.qsize()

    return EngineStatus(
        persona_name=get_persona_name(target),
        llm_model=get_llm_model(target),
        running=running,
        proactive_state=proactive_state,
        idle_seconds=idle_seconds,
        queue_size=queue_size,
        uptime_seconds=uptime_seconds,
        working_memory_turns=len(working_memory),
        attention=attention,
        emotion_state=snapshot_emotion_state(target),
        contexts=contexts or [],
        patterns=patterns or [],
        config_source=config_source,
    )
