from __future__ import annotations

from typing import Optional

from .visual_semantics import infer_visual_emotion_signal
from .visual_types import VisualEvent, VisualPerceptionConfig


def default_visual_event_text(event: VisualEvent) -> str:
    score = event.peak_score
    area = float(event.metrics.get("foreground_ratio", 0.0))
    hist = float(event.metrics.get("histogram_distance", 0.0))
    edge = float(event.metrics.get("edge_change", 0.0))

    if hist >= 0.35 and area >= 0.15:
        return "画面发生了明显切换"
    if area >= 0.12 and edge >= 0.05:
        return "检测到明显动作或物体变化"
    if area >= 0.04:
        return "检测到局部视觉变化"
    if score >= 0.3:
        return "检测到视觉事件"
    return "画面出现轻微变化"


def derive_visual_emotion_signal(
    event: VisualEvent,
    config: Optional[VisualPerceptionConfig] = None,
) -> dict:
    cfg = config or VisualPerceptionConfig()
    text_parts = []
    if event.analysis:
        text_parts.append(event.summary_text())
        text_parts.extend(event.analysis.facts)
        text_parts.extend(event.analysis.weak_interpretations)
    return infer_visual_emotion_signal(
        text_parts,
        peak_score=event.peak_score,
        scale=cfg.visual_emotion_scale,
    )


def visual_event_memory_text(event: VisualEvent) -> str:
    if event.analysis and event.analysis.memory_candidate:
        return event.analysis.memory_candidate
    if event.analysis and event.analysis.facts:
        return "；".join(event.analysis.facts[:2])
    return default_visual_event_text(event)


def visual_event_to_agent_text(event: VisualEvent) -> str:
    content = event.summary_text().strip()
    if not content:
        content = default_visual_event_text(event)
    return f"[视觉事件] {content}"


__all__ = [
    "default_visual_event_text",
    "derive_visual_emotion_signal",
    "visual_event_memory_text",
    "visual_event_to_agent_text",
]
