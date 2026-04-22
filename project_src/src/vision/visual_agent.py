from __future__ import annotations

from typing import Optional, TYPE_CHECKING

from .visual_text import visual_event_to_agent_text
from .visual_types import VisualEvent

if TYPE_CHECKING:
    from src.agent.agent_loop import AgentEvent, AgentLoop
    from src.core_engine.runtime_types import ChatMode


def visual_event_to_agent_event(
    event: VisualEvent,
    *,
    chat_mode: Optional["ChatMode"] = None,
    is_mentioned: bool = True,
    reply_context: Optional[dict] = None,
    context_id: Optional[str] = None,
) -> "AgentEvent":
    from src.agent.agent_loop import AgentEvent
    from src.core_engine.runtime_types import ChatMode

    metadata = {
        "event_id": event.event_id,
        "peak_frame_index": event.peak_frame_index,
        "representative_frame_index": event.representative_frame_index,
        "peak_score": event.peak_score,
        "metrics": event.metrics,
        "rate_limited": event.rate_limited,
    }
    if event.analysis:
        metadata["analysis"] = {
            "scene": event.analysis.scene,
            "facts": event.analysis.facts,
            "weak_interpretations": event.analysis.weak_interpretations,
            "memory_candidate": event.analysis.memory_candidate,
            "agent_hint": event.analysis.agent_hint,
            "raw_text": event.analysis.raw_text,
        }

    return AgentEvent(
        type="visual",
        content=visual_event_to_agent_text(event),
        chat_mode=chat_mode or ChatMode.PRIVATE,
        is_mentioned=is_mentioned,
        images=list(event.keyframes),
        reply_context=reply_context or {},
        context_id=context_id,
        metadata=metadata,
    )


def push_visual_event_to_agent_loop(
    loop: "AgentLoop",
    event: VisualEvent,
    *,
    chat_mode: Optional["ChatMode"] = None,
    is_mentioned: bool = True,
    reply_context: Optional[dict] = None,
    context_id: Optional[str] = None,
) -> None:
    loop.push(
        visual_event_to_agent_event(
            event,
            chat_mode=chat_mode,
            is_mentioned=is_mentioned,
            reply_context=reply_context,
            context_id=context_id,
        )
    )


__all__ = [
    "push_visual_event_to_agent_loop",
    "visual_event_to_agent_event",
]
