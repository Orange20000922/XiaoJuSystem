from __future__ import annotations

import json
import math
import queue
import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Deque, List, Optional, Protocol, Sequence, TYPE_CHECKING, Union

import numpy as np

from src.logger import logger
from src.media.image_utils import ImageResult

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    cv2 = None
    CV2_AVAILABLE = False

if TYPE_CHECKING:
    from src.agent.agent_loop import AgentEvent, AgentLoop
    from src.core.inference_pipeline import ChatMode, NeuroLikePipeline
    from src.llm.client import LLMClient


DEFAULT_VISION_ANALYSIS_PROMPT = """
请根据提供的关键帧，输出一个 JSON 对象，不要输出额外文字。

字段要求：
{
  "facts": ["最多3条可直接观察到的事实"],
  "weak_interpretations": ["最多2条弱解释，必须使用“可能”这类低强度表述"],
  "memory_candidate": "1条适合长期记忆的压缩观察，没有则为空字符串",
  "agent_hint": "1条适合发给主 Agent 的简短视觉事件描述"
}

规则：
1. facts 只能描述可见事实，不要脑补心理状态。
2. weak_interpretations 只能做低强度推断，避免“焦虑、压力大、犹豫”等强心理结论。
3. memory_candidate 只在事件具有跨时间意义时填写，否则返回空字符串。
4. agent_hint 控制在 30 字以内，适合作为“[视觉事件] ...”的正文。
""".strip()


def _ensure_cv2() -> None:
    if not CV2_AVAILABLE:
        raise RuntimeError(
            "动态视觉模块需要 OpenCV。"
            "请先安装 opencv-python-headless 或 opencv-python。"
        )


def _odd_kernel(value: int) -> int:
    value = max(1, int(value))
    return value if value % 2 == 1 else value + 1


def sigmoid_normalize(value: float, center: float, scale: float) -> float:
    scale = max(scale, 1e-6)
    return 1.0 / (1.0 + math.exp(-((value - center) / scale)))


def _strip_json_fence(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def _extract_json_object(text: str) -> Optional[dict]:
    cleaned = _strip_json_fence(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _fit_width(frame: np.ndarray, width: int) -> np.ndarray:
    if width <= 0:
        return frame
    h, w = frame.shape[:2]
    if w <= width:
        return frame
    scale = width / float(w)
    resized_h = max(1, int(round(h * scale)))
    return cv2.resize(frame, (width, resized_h), interpolation=cv2.INTER_AREA)


def _encode_frame_as_image_result(
    frame_bgr: np.ndarray,
    frame_index: int,
    timestamp: float,
    jpeg_quality: int,
) -> ImageResult:
    _ensure_cv2()
    success, encoded = cv2.imencode(
        ".jpg",
        frame_bgr,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
    )
    if not success:
        raise RuntimeError(f"Failed to encode keyframe {frame_index} as JPEG")

    import base64

    payload = base64.standard_b64encode(encoded.tobytes()).decode("ascii")
    return ImageResult(
        base64_data=payload,
        media_type="image/jpeg",
        original_url=f"visual://frame/{frame_index}?ts={timestamp:.3f}",
    )


@dataclass
class VisualPerceptionConfig:
    """动态视觉感知原型的运行时配置。"""
    enabled: bool = True
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
    visual_emotion_scale: float = 0.2
    memory_peak_score_threshold: float = 0.22
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
    facts: List[str] = field(default_factory=list)
    weak_interpretations: List[str] = field(default_factory=list)
    memory_candidate: str = ""
    agent_hint: str = ""
    raw_text: str = ""

    @classmethod
    def from_llm_text(cls, text: str) -> "VisualAnalysis":
        data = _extract_json_object(text)
        if not data:
            return cls(raw_text=text.strip(), agent_hint=text.strip())
        return cls(
            facts=[str(item) for item in data.get("facts", []) if str(item).strip()],
            weak_interpretations=[
                str(item)
                for item in data.get("weak_interpretations", [])
                if str(item).strip()
            ],
            memory_candidate=str(data.get("memory_candidate", "")).strip(),
            agent_hint=str(data.get("agent_hint", "")).strip(),
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


@dataclass
class _DetectorState:
    subtractor: object
    morphology_kernel: np.ndarray
    temporal_votes: Deque[bool]
    previous_gray: Optional[np.ndarray] = None
    previous_histogram: Optional[np.ndarray] = None
    previous_edge_density: Optional[float] = None
    previous_smoothed_score: float = 0.0


class TemporalPeakSelector:
    def __init__(
        self,
        peak_threshold: float,
        cooldown_seconds: float,
        window_seconds: float,
        peak_neighborhood_frames: int,
    ):
        self.peak_threshold = peak_threshold
        self.cooldown_seconds = cooldown_seconds
        self.window_seconds = window_seconds
        self.peak_neighborhood_frames = peak_neighborhood_frames
        self._window: Deque[FrameObservation] = deque()
        self._history: Deque[FrameObservation] = deque(maxlen=3)
        self._last_peak_time = -1e9

    def consume(self, observation: FrameObservation) -> Optional[PeakSelection]:
        self._window.append(observation)
        self._history.append(observation)
        self._trim_window(observation.timestamp)

        if len(self._history) < 3:
            return None

        left, center, right = self._history
        is_local_max = (
            center.smoothed_score > left.smoothed_score
            and center.smoothed_score >= right.smoothed_score
        )
        if not is_local_max:
            return None
        if center.smoothed_score < self.peak_threshold:
            return None
        if not center.motion_vote:
            return None
        if (center.timestamp - self._last_peak_time) < self.cooldown_seconds:
            return None

        representative = self._select_representative(center.frame_index)
        self._last_peak_time = center.timestamp
        return PeakSelection(
            peak=center,
            representative=representative,
            window=list(self._window),
        )

    def _trim_window(self, now: float) -> None:
        cutoff = now - self.window_seconds
        while self._window and self._window[0].timestamp < cutoff:
            self._window.popleft()

    def _select_representative(self, peak_frame_index: int) -> FrameObservation:
        candidates = [
            item
            for item in self._window
            if abs(item.frame_index - peak_frame_index) <= self.peak_neighborhood_frames
        ]
        if not candidates:
            candidates = list(self._window)
        return max(
            candidates,
            key=lambda item: (item.sharpness, item.smoothed_score, -abs(item.frame_index - peak_frame_index)),
        )


class LLMVisualEventAnalyzer:
    def __init__(
        self,
        llm_client: "LLMClient",
        prompt: str = DEFAULT_VISION_ANALYSIS_PROMPT,
        max_tokens: int = 400,
    ):
        self.llm_client = llm_client
        self.prompt = prompt
        self.max_tokens = max_tokens

    def analyze(self, event: VisualEvent) -> Optional[VisualAnalysis]:
        if not event.keyframes:
            return None
        text = self.llm_client.generate(
            system_prompt=self.prompt,
            user_input="请分析这些关键帧对应的视觉事件。",
            images=event.keyframes,
            max_tokens=self.max_tokens,
            temperature=0.2,
        )
        return VisualAnalysis.from_llm_text(text)


class VisualPerceptionPipeline:
    def __init__(
        self,
        config: Optional[VisualPerceptionConfig] = None,
        analyzer: Optional[VisualEventAnalyzer] = None,
        event_callback: Optional[Callable[[VisualEvent], None]] = None,
    ):
        self.config = config or VisualPerceptionConfig()
        self.analyzer = analyzer
        self.event_callback = event_callback
        self._stop_event = threading.Event()
        self._events_lock = threading.Lock()
        self._collected_events: List[VisualEvent] = []

    def stop(self) -> None:
        self._stop_event.set()

    def run(self, source: Union[int, str, Path], max_frames: Optional[int] = None) -> List[VisualEvent]:
        _ensure_cv2()
        self._stop_event.clear()
        self._collected_events = []

        frame_queue: "queue.Queue[Optional[FramePacket]]" = queue.Queue(
            maxsize=self.config.frame_queue_size
        )
        observation_queue: "queue.Queue[Optional[FrameObservation]]" = queue.Queue(
            maxsize=self.config.observation_queue_size
        )
        event_queue: "queue.Queue[Optional[VisualEvent]]" = queue.Queue(
            maxsize=self.config.event_queue_size
        )

        producer = threading.Thread(
            target=self._producer_loop,
            args=(source, max_frames, frame_queue),
            daemon=True,
            name="visual-producer",
        )
        detector = threading.Thread(
            target=self._detector_loop,
            args=(frame_queue, observation_queue),
            daemon=True,
            name="visual-detector",
        )
        selector = threading.Thread(
            target=self._selector_loop,
            args=(observation_queue, event_queue),
            daemon=True,
            name="visual-selector",
        )
        uploader = threading.Thread(
            target=self._uploader_loop,
            args=(event_queue,),
            daemon=True,
            name="visual-uploader",
        )

        threads = [producer, detector, selector, uploader]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        return list(self._collected_events)

    def _producer_loop(
        self,
        source: Union[int, str, Path],
        max_frames: Optional[int],
        frame_queue: "queue.Queue[Optional[FramePacket]]",
    ) -> None:
        capture_source: Union[int, str] = source if isinstance(source, int) else str(source)
        capture = cv2.VideoCapture(capture_source)
        if not capture.isOpened():
            frame_queue.put(None)
            raise RuntimeError(f"Unable to open video source: {source}")

        try:
            fps = capture.get(cv2.CAP_PROP_FPS)
            fps = fps if fps and fps > 0 else 30.0
            frame_index = 0
            started_at = time.monotonic()

            while not self._stop_event.is_set():
                if max_frames is not None and frame_index >= max_frames:
                    break

                ok, frame = capture.read()
                if not ok:
                    break

                timestamp = frame_index / fps
                if isinstance(source, int):
                    timestamp = time.monotonic() - started_at

                packet = FramePacket(
                    frame_index=frame_index,
                    timestamp=timestamp,
                    frame_bgr=frame,
                )
                try:
                    frame_queue.put(packet, timeout=0.5)
                except queue.Full:
                    logger.warning("Visual frame queue is full, dropping a frame")
                frame_index += 1
        finally:
            capture.release()
            frame_queue.put(None)

    def _detector_loop(
        self,
        frame_queue: "queue.Queue[Optional[FramePacket]]",
        observation_queue: "queue.Queue[Optional[FrameObservation]]",
    ) -> None:
        detector_state = self._build_detector_state()

        while not self._stop_event.is_set():
            packet = frame_queue.get()
            if packet is None:
                observation_queue.put(None)
                return

            try:
                observation = self._analyze_frame(packet, detector_state)
                observation_queue.put(observation)
            except Exception as exc:
                logger.error(f"Visual detector failed on frame {packet.frame_index}: {exc}")

    def _selector_loop(
        self,
        observation_queue: "queue.Queue[Optional[FrameObservation]]",
        event_queue: "queue.Queue[Optional[VisualEvent]]",
    ) -> None:
        selector = TemporalPeakSelector(
            peak_threshold=self.config.peak_threshold,
            cooldown_seconds=self.config.peak_cooldown_seconds,
            window_seconds=self.config.event_window_seconds,
            peak_neighborhood_frames=self.config.peak_neighborhood_frames,
        )

        while not self._stop_event.is_set():
            observation = observation_queue.get()
            if observation is None:
                event_queue.put(None)
                return

            selection = selector.consume(observation)
            if selection is None:
                continue

            try:
                event = self._build_visual_event(selection)
                event_queue.put(event)
            except Exception as exc:
                logger.error(
                    f"Visual selector failed on peak frame {selection.peak.frame_index}: {exc}"
                )

    def _uploader_loop(
        self,
        event_queue: "queue.Queue[Optional[VisualEvent]]",
    ) -> None:
        recent_calls: Deque[float] = deque()

        while not self._stop_event.is_set():
            event = event_queue.get()
            if event is None:
                return

            now = time.monotonic()
            while recent_calls and (now - recent_calls[0]) > 60.0:
                recent_calls.popleft()

            if self.analyzer is not None and len(recent_calls) < self.config.vision_calls_per_minute:
                try:
                    event.analysis = self.analyzer.analyze(event)
                    recent_calls.append(now)
                except Exception as exc:
                    logger.warning(f"Visual event analysis failed: {exc}")
            elif self.analyzer is not None:
                event.rate_limited = True
                logger.info(
                    "Visual event analysis skipped because the Vision LLM rate limit was reached"
                )

            with self._events_lock:
                self._collected_events.append(event)

            if self.event_callback is not None:
                try:
                    self.event_callback(event)
                except Exception as exc:
                    logger.warning(f"Visual event callback failed: {exc}")

    def _build_detector_state(self) -> _DetectorState:
        morphology_kernel = np.ones(
            (
                _odd_kernel(self.config.morphology_kernel_size),
                _odd_kernel(self.config.morphology_kernel_size),
            ),
            dtype=np.uint8,
        )
        subtractor = cv2.createBackgroundSubtractorMOG2(
            history=self.config.mog2_history,
            varThreshold=self.config.mog2_var_threshold,
            detectShadows=self.config.mog2_detect_shadows,
        )
        return _DetectorState(
            subtractor=subtractor,
            morphology_kernel=morphology_kernel,
            temporal_votes=deque(maxlen=self.config.temporal_vote_window),
        )

    def _analyze_frame(
        self,
        packet: FramePacket,
        state: _DetectorState,
    ) -> FrameObservation:
        gaussian_kernel = _odd_kernel(self.config.gaussian_kernel_size)
        resized = _fit_width(packet.frame_bgr, self.config.resize_width)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (gaussian_kernel, gaussian_kernel), 0)

        foreground = state.subtractor.apply(blurred)
        if self.config.mog2_detect_shadows:
            _, foreground = cv2.threshold(foreground, 200, 255, cv2.THRESH_BINARY)

        foreground = cv2.erode(foreground, state.morphology_kernel, iterations=1)
        foreground = cv2.dilate(foreground, state.morphology_kernel, iterations=1)
        _, foreground_ratio = self._filter_small_components(foreground)

        histogram = cv2.calcHist([blurred], [0], None, [32], [0, 256])
        cv2.normalize(histogram, histogram)

        edges = cv2.Canny(
            blurred,
            self.config.canny_threshold1,
            self.config.canny_threshold2,
        )
        edge_density = float(np.count_nonzero(edges)) / float(edges.size)

        histogram_distance = 0.0
        edge_change = 0.0
        if state.previous_histogram is not None:
            histogram_distance = float(
                cv2.compareHist(
                    histogram,
                    state.previous_histogram,
                    cv2.HISTCMP_BHATTACHARYYA,
                )
            )
        if state.previous_edge_density is not None:
            edge_change = abs(edge_density - state.previous_edge_density)

        s1 = sigmoid_normalize(
            foreground_ratio,
            self.config.area_sigmoid_center,
            self.config.area_sigmoid_scale,
        )
        s2 = sigmoid_normalize(
            histogram_distance,
            self.config.histogram_sigmoid_center,
            self.config.histogram_sigmoid_scale,
        )
        s3 = sigmoid_normalize(
            edge_change,
            self.config.edge_sigmoid_center,
            self.config.edge_sigmoid_scale,
        )

        change_score = (
            self.config.area_weight * s1
            + self.config.histogram_weight * s2
            + self.config.edge_weight * s3
        )
        smoothed_score = (
            self.config.ema_alpha * change_score
            + (1.0 - self.config.ema_alpha) * state.previous_smoothed_score
        )

        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        motion_flag = foreground_ratio >= self.config.min_component_area_ratio
        state.temporal_votes.append(motion_flag)
        motion_vote = sum(1 for item in state.temporal_votes if item) >= self.config.temporal_vote_required

        state.previous_gray = blurred
        state.previous_histogram = histogram
        state.previous_edge_density = edge_density
        state.previous_smoothed_score = smoothed_score

        height, width = packet.frame_bgr.shape[:2]
        return FrameObservation(
            frame_index=packet.frame_index,
            timestamp=packet.timestamp,
            frame_bgr=packet.frame_bgr,
            frame_width=width,
            frame_height=height,
            foreground_ratio=foreground_ratio,
            histogram_distance=histogram_distance,
            edge_change=edge_change,
            change_score=change_score,
            smoothed_score=smoothed_score,
            sharpness=sharpness,
            motion_vote=motion_vote,
        )

    def _filter_small_components(self, mask: np.ndarray) -> tuple[np.ndarray, float]:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        total_pixels = float(mask.shape[0] * mask.shape[1])
        min_area_pixels = max(1, int(total_pixels * self.config.min_component_area_ratio))
        filtered = np.zeros_like(mask)
        kept_pixels = 0

        for label_index in range(1, num_labels):
            area = int(stats[label_index, cv2.CC_STAT_AREA])
            if area < min_area_pixels:
                continue
            filtered[labels == label_index] = 255
            kept_pixels += area

        return filtered, kept_pixels / total_pixels

    def _build_visual_event(self, selection: PeakSelection) -> VisualEvent:
        keyframe_observations = self._select_keyframes(selection)
        keyframes = [
            _encode_frame_as_image_result(
                observation.frame_bgr,
                observation.frame_index,
                observation.timestamp,
                self.config.jpeg_quality,
            )
            for observation in keyframe_observations
        ]

        metrics = {
            "foreground_ratio": selection.peak.foreground_ratio,
            "histogram_distance": selection.peak.histogram_distance,
            "edge_change": selection.peak.edge_change,
            "change_score": selection.peak.change_score,
            "smoothed_score": selection.peak.smoothed_score,
            "sharpness": selection.representative.sharpness,
            "window_size": len(selection.window),
        }

        return VisualEvent(
            event_id=f"visual-{selection.peak.frame_index}",
            peak_frame_index=selection.peak.frame_index,
            timestamp=selection.peak.timestamp,
            peak_score=selection.peak.smoothed_score,
            representative_frame_index=selection.representative.frame_index,
            keyframes=keyframes,
            metrics=metrics,
        )

    def _select_keyframes(self, selection: PeakSelection) -> List[FrameObservation]:
        unique: List[FrameObservation] = []
        earliest = selection.window[0]
        latest = selection.window[-1]
        representative = selection.representative

        for candidate in (earliest, representative, latest):
            if any(item.frame_index == candidate.frame_index for item in unique):
                continue
            unique.append(candidate)

        unique.sort(key=lambda item: item.frame_index)
        return unique[: self.config.max_keyframes_per_event]


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
    text_parts = [event.summary_text()]
    if event.analysis:
        text_parts.extend(event.analysis.facts)
        text_parts.extend(event.analysis.weak_interpretations)
    text = " ".join(part for part in text_parts if part).lower()

    valence = 0.0
    arousal = min(0.18, event.peak_score * 0.18)
    label = "neutral"

    keyword_rules = (
        (("微笑", "笑", "开心", "高兴"), 0.14, 0.06, "joy"),
        (("温柔", "亲近"), 0.10, 0.03, "tenderness"),
        (("皱眉", "困惑", "苦恼"), -0.10, 0.05, "sadness"),
        (("惊讶", "突然"), 0.02, 0.12, "surprise"),
        (("离开座位", "离开"), -0.03, 0.06, "neutral"),
        (("回到座位", "回到桌前", "返回"), 0.05, 0.05, "joy"),
        (("阅读", "学习", "看书"), 0.03, 0.03, "curiosity"),
    )

    for keywords, v_delta, a_delta, detected_label in keyword_rules:
        if any(keyword in text for keyword in keywords):
            valence += v_delta
            arousal += a_delta
            label = detected_label

    scale = max(0.0, cfg.visual_emotion_scale)
    valence *= scale
    arousal *= scale
    valence = max(-0.25, min(0.25, valence))
    arousal = max(-0.25, min(0.25, arousal))

    intensity = min(
        1.0,
        abs(valence) * 2.5 + abs(arousal) * 2.0 + min(event.peak_score, 0.4) * 0.5,
    )

    return {
        "valence_delta": round(valence, 4),
        "arousal_delta": round(arousal, 4),
        "emotion": label,
        "intensity": round(intensity, 4),
        "source_text": text.strip(),
    }


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


def visual_event_to_agent_event(
    event: VisualEvent,
    *,
    chat_mode: Optional["ChatMode"] = None,
    is_mentioned: bool = True,
    reply_context: Optional[dict] = None,
    context_id: Optional[str] = None,
) -> "AgentEvent":
    from src.agent.agent_loop import AgentEvent
    from src.core.inference_pipeline import ChatMode

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


def build_visual_analyzer_from_pipeline(
    pipeline: "NeuroLikePipeline",
    *,
    prompt: str = DEFAULT_VISION_ANALYSIS_PROMPT,
) -> Optional[LLMVisualEventAnalyzer]:
    llm_client = pipeline.llm_client_vision or pipeline.llm_client
    if llm_client is None:
        return None
    return LLMVisualEventAnalyzer(llm_client=llm_client, prompt=prompt)
