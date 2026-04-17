from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Protocol

import numpy as np

from src.media.image_utils import ImageResult

from ._shared import _extract_json_object


DEFAULT_VISION_ANALYSIS_PROMPT = """
你将看到多张按时间顺序排列的图片，它们来自同一段短时间片段。
请比较画面之间的变化，只根据可见内容输出一个 JSON 对象，不要输出额外文字。
如果画面中有清晰的人脸、头肩或主体人物，请优先描述该人物本身、该人物附近的物体，
尤其是脸部、手部、嘴部周边的物体和人与物的交互。
如果背景中也有人或有轻微变化，除非它们构成主要事件，否则不要把背景干扰当成描述重点。

字段要求：
{
  “scene”: “10字以内的场景概括，如'书桌前''客厅沙发上''户外路边'”,
  “facts”: [“最多3条可直接观察到的事实”],
  “weak_interpretations”: [“最多2条弱解释，必须使用”可能”这类低强度表述”],
  “memory_candidate”: “1条适合长期记忆的压缩观察，没有则为空字符串”,
  “agent_hint”: “1条适合发给主 Agent 的简短视觉事件描述”
}

规则：
1. scene 只描述可见环境和位置，不要包含人物动作，控制在 10 字以内。
2. facts 只能描述直接可见的动作、物体变化、人与物的交互或场景变化，不要脑补心理状态。
3. weak_interpretations 只能做低强度推断，避免”焦虑、压力大、犹豫”等强心理结论。
4. memory_candidate 只在该片段具有跨时间意义时填写，否则返回空字符串。
5. agent_hint 控制在 30 字以内，适合作为”[视觉事件] ...”的正文。
6. 若画面中存在主体人物，优先总结主体人物的连续动作，不要把背景人物衣着差异、次要遮挡或细小光照波动当作主事件。
""".strip()


def _normalize_analysis_text(value: object) -> str:
    text = str(value).strip()
    if text.startswith("[视觉事件]"):
        text = text[len("[视觉事件]") :].strip()
    return text


def _extract_json_like_string_field(text: str, key: str) -> str:
    pattern = re.compile(rf'"{re.escape(key)}"\s*:\s*"((?:\\.|[^"\\])*)"', flags=re.DOTALL)
    matched = pattern.search(text)
    if matched is None:
        return ""
    try:
        value = json.loads(f'"{matched.group(1)}"')
    except json.JSONDecodeError:
        value = matched.group(1)
    return _normalize_analysis_text(value)


def _extract_json_like_list_field(text: str, key: str) -> List[str]:
    pattern = re.compile(
        rf'"{re.escape(key)}"\s*:\s*(\[[\s\S]*?\])(?:\s*,|\s*\}}|$)',
        flags=re.DOTALL,
    )
    matched = pattern.search(text)
    if matched is None:
        return []

    raw_list = matched.group(1)
    try:
        items = json.loads(raw_list)
        if isinstance(items, list):
            return [
                _normalize_analysis_text(item)
                for item in items
                if _normalize_analysis_text(item)
            ]
    except json.JSONDecodeError:
        pass

    return [
        _normalize_analysis_text(item)
        for item in re.findall(r'"((?:\\.|[^"\\])*)"', raw_list)
        if _normalize_analysis_text(item)
    ]


def _recover_analysis_from_json_like_text(text: str) -> Optional[dict]:
    recovered = {
        "scene": _extract_json_like_string_field(text, "scene"),
        "facts": _extract_json_like_list_field(text, "facts"),
        "weak_interpretations": _extract_json_like_list_field(text, "weak_interpretations"),
        "memory_candidate": _extract_json_like_string_field(text, "memory_candidate"),
        "agent_hint": _extract_json_like_string_field(text, "agent_hint"),
    }
    if any(
        recovered["scene"]
        or recovered["facts"]
        or recovered["weak_interpretations"]
        or recovered["memory_candidate"]
        or recovered["agent_hint"]
    ):
        return recovered
    return None


@dataclass
class VisualPerceptionConfig:
    """动态视觉感知原型的运行时配置。"""

    enabled: bool = True
    vision_analysis_mode: str = "triggered"  # "none" | "triggered" | "per_event"
    resize_width: int = 320
    gaussian_kernel_size: int = 5
    mog2_history: int = 500
    mog2_var_threshold: float = 16.0
    mog2_detect_shadows: bool = False
    morphology_kernel_size: int = 3
    min_component_area_ratio: float = 0.005
    temporal_vote_window: int = 3
    temporal_vote_required: int = 2
    area_weight: float = 0.5
    histogram_weight: float = 0.2
    edge_weight: float = 0.3
    ema_alpha: float = 0.3
    peak_threshold: float = 0.15
    peak_cooldown_seconds: float = 0.5
    event_window_seconds: float = 1.0
    peak_neighborhood_frames: int = 3
    max_keyframes_per_event: int = 3
    frame_queue_size: int = 5
    observation_queue_size: int = 32
    event_queue_size: int = 16
    vision_calls_per_minute: int = 2
    vision_rate_limit_wait_for_file_source: bool = True
    vision_rate_limit_max_wait_seconds: float = 90.0
    area_sigmoid_center: float = 0.02
    area_sigmoid_scale: float = 0.01
    histogram_sigmoid_center: float = 0.08
    histogram_sigmoid_scale: float = 0.04
    edge_sigmoid_center: float = 0.03
    edge_sigmoid_scale: float = 0.015
    canny_threshold1: int = 100
    canny_threshold2: int = 200
    jpeg_quality: int = 85
    route_visual_events_to_chat: bool = True
    inject_to_emotion_state: bool = True
    persist_to_memory: bool = True
    weak_monitor_buffer_seconds: float = 30.0
    segment_merge_gap_seconds: float = 2.0
    trigger_analysis_enabled: bool = True
    trigger_window_seconds: float = 10.0
    trigger_accumulated_score_threshold: float = 1.2
    trigger_peak_score_threshold: float = 0.35
    trigger_min_strong_events: int = 2
    trigger_refractory_seconds: float = 8.0
    explicit_request_top_k: int = 2
    summary_enabled: bool = True
    summary_window_seconds: float = 30.0
    summary_top_k: int = 3
    visual_emotion_scale: float = 0.2
    memory_peak_score_threshold: float = 0.22
    clip_duration_seconds: float = 0.0    # 事件视频片段时长（秒），0=单帧模式
    clip_max_frames: int = 8              # 片段模式下最多采样帧数
    # ── 自适应帧采样（FFT 频域驱动） ──
    adaptive_sampling_enabled: bool = False
    adaptive_stft_window_size: int = 32
    adaptive_stft_hop_size: int = 4
    adaptive_highfreq_cutoff_ratio: float = 0.3
    adaptive_fps_min: float = 4.0
    adaptive_fps_max: float = 15.0
    adaptive_gamma: float = 0.7
    adaptive_spike_threshold: float = 0.4
    adaptive_spike_boost_seconds: float = 2.0
    adaptive_fps_smoothing_alpha: float = 0.3
    adaptive_precheck_diff_threshold: float = 15.0
    adaptive_precheck_resize_width: int = 160
    analysis_prompt: str = DEFAULT_VISION_ANALYSIS_PROMPT

    @classmethod
    def from_settings(cls, settings: object) -> "VisualPerceptionConfig":
        """从配置数据对象构造运行时配置。"""

        if settings is None:
            return cls()
        kwargs = {}
        for field_name in cls.__dataclass_fields__.keys():
            if hasattr(settings, field_name):
                kwargs[field_name] = getattr(settings, field_name)
        return cls(**kwargs)


@dataclass
class FramePacket:
    frame_index: int
    timestamp: float
    frame_bgr: np.ndarray


@dataclass
class FrameObservation:
    frame_index: int
    timestamp: float
    frame_bgr: np.ndarray
    frame_width: int
    frame_height: int
    foreground_ratio: float
    histogram_distance: float
    edge_change: float
    change_score: float
    smoothed_score: float
    sharpness: float
    motion_vote: bool


@dataclass
class PeakSelection:
    peak: FrameObservation
    representative: FrameObservation
    window: List[FrameObservation]


@dataclass
class VisualAnalysis:
    scene: str = ""
    facts: List[str] = field(default_factory=list)
    weak_interpretations: List[str] = field(default_factory=list)
    memory_candidate: str = ""
    agent_hint: str = ""
    raw_text: str = ""

    @classmethod
    def from_llm_text(cls, text: str) -> "VisualAnalysis":
        data = _extract_json_object(text)
        if not data:
            data = _recover_analysis_from_json_like_text(text)
        if not data:
            cleaned = _normalize_analysis_text(text)
            return cls(raw_text=text.strip(), agent_hint=cleaned)
        return cls(
            scene=_normalize_analysis_text(data.get("scene", "")),
            facts=[
                _normalize_analysis_text(item)
                for item in data.get("facts", [])
                if _normalize_analysis_text(item)
            ],
            weak_interpretations=[
                _normalize_analysis_text(item)
                for item in data.get("weak_interpretations", [])
                if _normalize_analysis_text(item)
            ],
            memory_candidate=_normalize_analysis_text(data.get("memory_candidate", "")),
            agent_hint=_normalize_analysis_text(data.get("agent_hint", "")),
            raw_text=text.strip(),
        )

    def to_summary_text(self) -> str:
        if self.agent_hint:
            return self.agent_hint
        if self.facts:
            return "；".join(self.facts[:2])
        if self.raw_text:
            return self.raw_text
        return "检测到视觉事件"


@dataclass
class VisualEvent:
    event_id: str
    peak_frame_index: int
    timestamp: float
    peak_score: float
    representative_frame_index: int
    keyframes: List[ImageResult] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    analysis: Optional[VisualAnalysis] = None
    rate_limited: bool = False

    def summary_text(self) -> str:
        if self.analysis:
            return self.analysis.to_summary_text()
        return "检测到视觉事件"


class VisualEventAnalyzer(Protocol):
    def analyze(self, event: VisualEvent) -> Optional[VisualAnalysis]:
        ...


__all__ = [
    "DEFAULT_VISION_ANALYSIS_PROMPT",
    "FrameObservation",
    "FramePacket",
    "PeakSelection",
    "VisualAnalysis",
    "VisualEvent",
    "VisualEventAnalyzer",
    "VisualPerceptionConfig",
]
