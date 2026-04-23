import importlib.abc
import importlib.util
import sys
import tempfile
import time
import types
import unittest
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np

MODULE_PATH = Path(__file__).resolve().parents[1] / "vision" / "__init__.py"
SRC_ROOT = MODULE_PATH.parents[2]


class _DummyLogger:
    def debug(self, *args, **kwargs):
        return None

    info = warning = error = debug


@dataclass
class _ImageResult:
    base64_data: str
    media_type: str
    original_url: str


class _ChatMode(Enum):
    PRIVATE = "private"
    GROUP = "group"


@dataclass
class _AgentEvent:
    type: str = ""
    content: str = ""
    chat_mode: _ChatMode = _ChatMode.PRIVATE
    is_mentioned: bool = True
    timestamp: float = field(default_factory=time.time)
    reply_context: dict = field(default_factory=dict)
    images: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    user_id: Optional[int] = None
    user_name: Optional[str] = None
    context_id: Optional[str] = None


src_pkg = types.ModuleType("src")
src_pkg.__path__ = [str(SRC_ROOT)]
logger_module = types.ModuleType("src.logger")
logger_module.logger = _DummyLogger()
media_pkg = types.ModuleType("src.media")
media_pkg.__path__ = []
image_utils_module = types.ModuleType("src.media.image_utils")
image_utils_module.ImageResult = _ImageResult
vision_pkg = types.ModuleType("src.vision")
vision_pkg.__path__ = [str(MODULE_PATH.parent)]

agent_pkg = types.ModuleType("src.agent")
agent_pkg.__path__ = [str(Path(SRC_ROOT) / "agent")]
agent_loop_module = types.ModuleType("src.agent.agent_loop")
agent_loop_module.AgentEvent = _AgentEvent
agent_loop_module.AgentLoop = None
agent_loop_module.ProactiveState = None

core_pkg = types.ModuleType("src.core")
core_pkg.__path__ = [str(Path(SRC_ROOT) / "core")]
inference_pipeline_module = types.ModuleType("src.core.inference_pipeline")
inference_pipeline_module.ChatMode = _ChatMode

sys.modules.setdefault("src", src_pkg)
sys.modules["src.logger"] = logger_module
sys.modules["src.media"] = media_pkg
sys.modules["src.media.image_utils"] = image_utils_module
sys.modules["src.vision"] = vision_pkg
sys.modules["src.agent"] = agent_pkg
sys.modules["src.agent.agent_loop"] = agent_loop_module
sys.modules["src.core"] = core_pkg
sys.modules["src.core.inference_pipeline"] = inference_pipeline_module

spec = importlib.util.spec_from_file_location("src.vision", MODULE_PATH,
                                               submodule_search_locations=[str(MODULE_PATH.parent)])
visual_module = importlib.util.module_from_spec(spec)
visual_module.__path__ = [str(MODULE_PATH.parent)]
assert spec.loader is not None
sys.modules["src.vision"] = visual_module
spec.loader.exec_module(visual_module)
visual_pipeline_module = sys.modules["src.vision.visual_pipeline"]

FrameObservation = visual_module.FrameObservation
TemporalPeakSelector = visual_module.TemporalPeakSelector
VisualAnalysis = visual_module.VisualAnalysis
VisualEventCluster = visual_module.VisualEventCluster
VisualEventMonitor = visual_module.VisualEventMonitor
VisualEvent = visual_module.VisualEvent
VisualPerceptionConfig = visual_module.VisualPerceptionConfig
VisualPerceptionPipeline = visual_module.VisualPerceptionPipeline
LLMVisualEventAnalyzer = visual_module.LLMVisualEventAnalyzer
VisualSkillDetector = visual_module.VisualSkillDetector
VisualSkillExecutor = visual_module.VisualSkillExecutor
CV2_AVAILABLE = visual_module.CV2_AVAILABLE
DEFAULT_VISION_ANALYSIS_PROMPT = visual_module.DEFAULT_VISION_ANALYSIS_PROMPT
build_visual_analysis_user_input = visual_module.build_visual_analysis_user_input
cluster_visual_events = visual_module.cluster_visual_events
cv2 = visual_module.cv2
default_visual_event_text = visual_module.default_visual_event_text
derive_visual_emotion_signal = visual_module.derive_visual_emotion_signal
sigmoid_normalize = visual_module.sigmoid_normalize
summarize_visual_event_windows = visual_module.summarize_visual_event_windows
visual_event_to_agent_text = visual_module.visual_event_to_agent_text
visual_event_to_agent_event = visual_module.visual_event_to_agent_event


def _make_observation(frame_index: int, timestamp: float, score: float, sharpness: float, motion_vote: bool = True):
    return FrameObservation(
        frame_index=frame_index,
        timestamp=timestamp,
        frame_bgr=np.zeros((8, 8, 3), dtype=np.uint8),
        frame_width=8,
        frame_height=8,
        foreground_ratio=0.08,
        histogram_distance=0.1,
        edge_change=0.04,
        change_score=score,
        smoothed_score=score,
        sharpness=sharpness,
        motion_vote=motion_vote,
    )


def _make_visual_event(
    event_id: str,
    timestamp: float,
    peak_score: float,
    sharpness: float = 10.0,
    frame_index: int = 0,
):
    return VisualEvent(
        event_id=event_id,
        peak_frame_index=frame_index,
        timestamp=timestamp,
        peak_score=peak_score,
        representative_frame_index=frame_index,
        metrics={"sharpness": sharpness},
    )


class VisualPerceptionLogicTests(unittest.TestCase):
    def test_sigmoid_normalize_is_monotonic(self):
        low = sigmoid_normalize(0.01, center=0.02, scale=0.01)
        mid = sigmoid_normalize(0.02, center=0.02, scale=0.01)
        high = sigmoid_normalize(0.05, center=0.02, scale=0.01)
        self.assertLess(low, mid)
        self.assertLess(mid, high)

    def test_temporal_peak_selector_prefers_sharpest_candidate_near_peak(self):
        selector = TemporalPeakSelector(
            peak_threshold=0.15,
            cooldown_seconds=0.5,
            window_seconds=1.0,
            peak_neighborhood_frames=2,
        )
        observations = [
            _make_observation(0, 0.0, 0.05, 10.0),
            _make_observation(1, 0.1, 0.20, 15.0),
            _make_observation(2, 0.2, 0.40, 20.0),
            _make_observation(3, 0.3, 0.30, 90.0),
        ]

        selection = None
        for item in observations:
            selection = selector.consume(item) or selection

        self.assertIsNotNone(selection)
        self.assertEqual(selection.peak.frame_index, 2)
        self.assertEqual(selection.representative.frame_index, 3)

    def test_visual_analysis_parses_json_response(self):
        raw = """
        ```json
        {
          "facts": ["用户回到桌前", "桌面上有一本打开的书"],
          "weak_interpretations": ["可能继续在阅读"],
          "memory_candidate": "用户回到书桌前继续阅读",
          "agent_hint": "用户回到座位并继续看书"
        }
        ```
        """
        analysis = VisualAnalysis.from_llm_text(raw)
        self.assertEqual(analysis.facts[0], "用户回到桌前")
        self.assertEqual(analysis.weak_interpretations[0], "可能继续在阅读")
        self.assertEqual(analysis.memory_candidate, "用户回到书桌前继续阅读")
        self.assertEqual(analysis.to_summary_text(), "用户回到座位并继续看书")

    def test_visual_analysis_strips_visual_event_prefix_from_llm_fields(self):
        raw = """
        {
          "facts": ["[视觉事件] 用户拿起杯子"],
          "weak_interpretations": ["[视觉事件] 可能准备喝水"],
          "memory_candidate": "[视觉事件] 用户拿起杯子准备喝水",
          "agent_hint": "[视觉事件] 用户拿起杯子喝水"
        }
        """

        analysis = VisualAnalysis.from_llm_text(raw)
        event = VisualEvent(
            event_id="visual-prefixed",
            peak_frame_index=1,
            timestamp=0.1,
            peak_score=0.4,
            representative_frame_index=1,
            analysis=analysis,
        )

        self.assertEqual(analysis.facts[0], "用户拿起杯子")
        self.assertEqual(analysis.weak_interpretations[0], "可能准备喝水")
        self.assertEqual(analysis.memory_candidate, "用户拿起杯子准备喝水")
        self.assertEqual(analysis.agent_hint, "用户拿起杯子喝水")
        self.assertEqual(visual_event_to_agent_text(event), "[视觉事件] 用户拿起杯子喝水")

    def test_visual_analysis_recovers_fields_from_incomplete_json_like_text(self):
        raw = """
        {
          "facts": ["用户举起饮料瓶", "饮料瓶靠近嘴部"],
          "weak_interpretations": ["可能正在喝饮料"],
          "memory_candidate": "",
          "agent_hint": "用户举起饮料瓶准备饮用"
        """

        analysis = VisualAnalysis.from_llm_text(raw)

        self.assertEqual(analysis.facts[0], "用户举起饮料瓶")
        self.assertEqual(analysis.weak_interpretations[0], "可能正在喝饮料")
        self.assertEqual(analysis.agent_hint, "用户举起饮料瓶准备饮用")

    def test_prompt_builder_distinguishes_short_segment_and_summary(self):
        short_event = VisualEvent(
            event_id="short-1",
            peak_frame_index=1,
            timestamp=1.0,
            peak_score=0.42,
            representative_frame_index=1,
            metrics={"mode": "trigger"},
        )
        summary_event = VisualEvent(
            event_id="summary-1",
            peak_frame_index=2,
            timestamp=30.0,
            peak_score=0.55,
            representative_frame_index=2,
            metrics={"mode": "summary"},
        )

        short_prompt = build_visual_analysis_user_input(short_event)
        summary_prompt = build_visual_analysis_user_input(summary_event)

        self.assertIn("同一段短时间片段", short_prompt)
        self.assertIn("时间窗口", summary_prompt)
        self.assertIn("彼此不一定连续", summary_prompt)
        self.assertIn("人脸", short_prompt)
        self.assertIn("主体人物", summary_prompt)
        self.assertIn("背景", short_prompt)

    def test_system_prompt_prioritizes_person_and_filters_background_noise(self):
        self.assertIn("人脸", DEFAULT_VISION_ANALYSIS_PROMPT)
        self.assertIn("主体人物", DEFAULT_VISION_ANALYSIS_PROMPT)
        self.assertIn("背景", DEFAULT_VISION_ANALYSIS_PROMPT)
        self.assertIn("嘴部", DEFAULT_VISION_ANALYSIS_PROMPT)

    def test_llm_visual_event_analyzer_uses_mode_specific_user_prompt(self):
        class _FakeVisionClient:
            def __init__(self):
                self.calls = []

            def generate(self, **kwargs):
                self.calls.append(kwargs)
                return '{"facts":["用户抬起杯子"],"weak_interpretations":[],"memory_candidate":"","agent_hint":"用户拿起杯子"}'

        client = _FakeVisionClient()
        analyzer = LLMVisualEventAnalyzer(client)
        event = VisualEvent(
            event_id="summary-2",
            peak_frame_index=3,
            timestamp=31.0,
            peak_score=0.6,
            representative_frame_index=3,
            keyframes=[_ImageResult(base64_data="x", media_type="image/jpeg", original_url="visual://1")],
            metrics={"mode": "summary"},
        )

        analysis = analyzer.analyze(event)

        self.assertIsNotNone(analysis)
        self.assertEqual(analysis.agent_hint, "用户拿起杯子")
        self.assertEqual(len(client.calls), 1)
        self.assertIn("时间窗口", client.calls[0]["user_input"])

    def test_visual_event_summary_falls_back_when_analysis_missing(self):
        event = VisualEvent(
            event_id="visual-1",
            peak_frame_index=1,
            timestamp=0.1,
            peak_score=0.35,
            representative_frame_index=1,
            metrics={
                "foreground_ratio": 0.18,
                "histogram_distance": 0.42,
                "edge_change": 0.08,
            },
        )
        self.assertEqual(default_visual_event_text(event), "画面发生了明显切换")
        self.assertTrue(visual_event_to_agent_text(event).startswith("[视觉事件] "))

    def test_visual_emotion_signal_uses_text_and_peak_score(self):
        event = VisualEvent(
            event_id="visual-2",
            peak_frame_index=2,
            timestamp=0.2,
            peak_score=0.4,
            representative_frame_index=2,
            analysis=VisualAnalysis(
                facts=["用户回到桌前"],
                weak_interpretations=["可能继续在阅读"],
                agent_hint="用户回到座位并继续看书",
            ),
        )
        signal = derive_visual_emotion_signal(event)
        self.assertGreater(signal["valence_delta"], 0.0)
        self.assertGreater(signal["arousal_delta"], 0.0)
        self.assertIn(signal["emotion"], {"joy", "curiosity"})
        self.assertGreaterEqual(signal["confidence"], 0.0)

    def test_visual_emotion_signal_keeps_plain_drinking_action_neutral(self):
        event = VisualEvent(
            event_id="visual-drink",
            peak_frame_index=3,
            timestamp=0.3,
            peak_score=0.52,
            representative_frame_index=3,
            analysis=VisualAnalysis(
                facts=["用户拿起杯子并喝水"],
                weak_interpretations=["可能刚喝了一口水"],
                agent_hint="用户手持杯子进行饮水动作",
            ),
        )

        signal = derive_visual_emotion_signal(event)

        self.assertEqual(signal["emotion"], "neutral")
        self.assertAlmostEqual(signal["valence_delta"], 0.0, places=4)
        self.assertGreater(signal["arousal_delta"], 0.0)
        self.assertEqual(signal["matches"], [])

    def test_cluster_visual_events_merges_close_events_into_segments(self):
        events = [
            _make_visual_event("e1", timestamp=0.0, peak_score=0.22, frame_index=1),
            _make_visual_event("e2", timestamp=0.8, peak_score=0.35, frame_index=2),
            _make_visual_event("e3", timestamp=3.5, peak_score=0.28, frame_index=3),
        ]

        clusters = cluster_visual_events(events, merge_gap_seconds=2.0)

        self.assertEqual(len(clusters), 2)
        self.assertEqual(clusters[0].event_count, 2)
        self.assertEqual(clusters[0].peak_event.event_id, "e2")
        self.assertEqual(clusters[1].event_count, 1)
        self.assertAlmostEqual(clusters[0].accumulated_score, 0.57, places=3)

    def test_monitor_promotes_triggered_segment_only_after_short_window_crosses_threshold(self):
        monitor = VisualEventMonitor(
            VisualPerceptionConfig(
                vision_analysis_mode="triggered",
                weak_monitor_buffer_seconds=30.0,
                segment_merge_gap_seconds=2.0,
                trigger_analysis_enabled=True,
                trigger_window_seconds=10.0,
                trigger_accumulated_score_threshold=0.6,
                trigger_peak_score_threshold=0.25,
                trigger_min_strong_events=2,
                trigger_refractory_seconds=5.0,
                summary_enabled=False,
            )
        )

        def _analyze(event):
            return VisualAnalysis(agent_hint="用户拿起杯子并喝水"), False

        first = monitor.consume_candidate(
            _make_visual_event("e1", timestamp=1.0, peak_score=0.28, frame_index=1),
            analyze_callback=_analyze,
        )
        second = monitor.consume_candidate(
            _make_visual_event("e2", timestamp=2.0, peak_score=0.36, frame_index=2),
            analyze_callback=_analyze,
        )

        self.assertEqual(len(first.promoted_events), 0)
        self.assertEqual(len(second.promoted_events), 1)
        promoted = second.promoted_events[0]
        self.assertEqual(promoted.metrics["mode"], "trigger")
        self.assertEqual(promoted.metrics["segment_event_count"], 2)
        self.assertEqual(promoted.analysis.agent_hint, "用户拿起杯子并喝水")

    def test_pipeline_waits_for_file_source_rate_limit_slot_before_skipping(self):
        class _FakeAnalyzer:
            def __init__(self):
                self.calls = 0

            def analyze(self, event):
                self.calls += 1
                return VisualAnalysis(agent_hint="用户拿起杯子")

        analyzer = _FakeAnalyzer()
        pipeline = VisualPerceptionPipeline(
            config=VisualPerceptionConfig(
                vision_calls_per_minute=1,
                vision_rate_limit_wait_for_file_source=True,
                vision_rate_limit_max_wait_seconds=90.0,
            ),
            analyzer=analyzer,
        )
        pipeline._allow_analysis_wait = True
        pipeline._analysis_call_times.append(0.0)

        event = VisualEvent(
            event_id="visual-wait",
            peak_frame_index=1,
            timestamp=1.0,
            peak_score=0.4,
            representative_frame_index=1,
        )

        original_monotonic = visual_pipeline_module.time.monotonic
        original_sleep = visual_pipeline_module.time.sleep
        monotonic_values = iter([10.0, 60.2])
        sleep_calls = []

        visual_pipeline_module.time.monotonic = lambda: next(monotonic_values)
        visual_pipeline_module.time.sleep = lambda seconds: sleep_calls.append(seconds)
        try:
            analysis, rate_limited = pipeline._analyze_event_if_allowed(event)
        finally:
            visual_pipeline_module.time.monotonic = original_monotonic
            visual_pipeline_module.time.sleep = original_sleep

        self.assertFalse(rate_limited)
        self.assertIsNotNone(analysis)
        self.assertEqual(analyzer.calls, 1)
        self.assertEqual(len(sleep_calls), 1)
        self.assertGreater(sleep_calls[0], 49.0)

    def test_summarize_visual_event_windows_keeps_top_k_clusters_per_window(self):
        events = [
            _make_visual_event("w1-a", timestamp=1.0, peak_score=0.20, frame_index=1),
            _make_visual_event("w1-b", timestamp=2.0, peak_score=0.45, frame_index=2),
            _make_visual_event("w1-c", timestamp=12.0, peak_score=0.30, frame_index=3),
            _make_visual_event("w2-a", timestamp=31.0, peak_score=0.25, frame_index=4),
            _make_visual_event("w2-b", timestamp=34.0, peak_score=0.40, frame_index=5),
        ]

        summaries = summarize_visual_event_windows(
            events,
            window_seconds=30.0,
            top_k=1,
            merge_gap_seconds=2.0,
        )

        self.assertEqual(len(summaries), 2)
        self.assertEqual(len(summaries[0].top_clusters), 1)
        self.assertEqual(summaries[0].top_clusters[0].peak_event.event_id, "w1-b")
        self.assertEqual(len(summaries[1].top_clusters), 1)
        self.assertEqual(summaries[1].top_clusters[0].peak_event.event_id, "w2-b")

    def test_pipeline_raises_when_video_source_cannot_open(self):
        if not CV2_AVAILABLE:
            self.skipTest("OpenCV is not available")

        pipeline = VisualPerceptionPipeline(config=VisualPerceptionConfig())
        missing = Path(__file__).with_name("__missing_visual_source__.mp4")
        with self.assertRaises(RuntimeError):
            pipeline.run(missing, max_frames=1)

    def test_pipeline_can_save_trimmed_video_segment(self):
        if not CV2_AVAILABLE:
            self.skipTest("OpenCV is not available")

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.avi"
            output_path = Path(tmpdir) / "output.avi"

            writer = cv2.VideoWriter(
                str(input_path),
                cv2.VideoWriter_fourcc(*"MJPG"),
                10.0,
                (64, 48),
            )
            self.assertTrue(writer.isOpened())
            for index in range(10):
                frame = np.full((48, 64, 3), index * 20, dtype=np.uint8)
                writer.write(frame)
            writer.release()

            pipeline = VisualPerceptionPipeline(config=VisualPerceptionConfig())
            events = pipeline.run(
                input_path,
                duration_seconds=0.35,
                save_video_path=output_path,
            )

            self.assertIsInstance(events, list)
            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 0)

            capture = cv2.VideoCapture(str(output_path))
            self.assertTrue(capture.isOpened())
            saved_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            capture.release()

            self.assertGreater(saved_frames, 0)
            self.assertLessEqual(saved_frames, 4)

    # ── clip mode tests ──

    def test_clip_duration_zero_behaves_like_original(self):
        """clip_duration_seconds=0 should produce the same peak detection as before."""
        selector = TemporalPeakSelector(
            peak_threshold=0.15,
            cooldown_seconds=0.5,
            window_seconds=1.0,
            peak_neighborhood_frames=2,
            clip_half_duration=0.0,
        )
        observations = [
            _make_observation(0, 0.0, 0.05, 10.0),
            _make_observation(1, 0.1, 0.20, 15.0),
            _make_observation(2, 0.2, 0.40, 20.0),
            _make_observation(3, 0.3, 0.30, 90.0),
        ]

        selection = None
        for item in observations:
            selection = selector.consume(item) or selection

        self.assertIsNotNone(selection)
        self.assertEqual(selection.peak.frame_index, 2)
        self.assertGreater(len(selection.window), 0)

    def test_clip_mode_delays_emission_until_deadline(self):
        """In clip mode, peak should not emit immediately but wait for deadline."""
        selector = TemporalPeakSelector(
            peak_threshold=0.15,
            cooldown_seconds=2.0,
            window_seconds=2.0,
            peak_neighborhood_frames=2,
            clip_half_duration=1.0,  # 2s total clip
        )
        observations = [
            _make_observation(0, 0.0, 0.05, 10.0),
            _make_observation(1, 0.1, 0.20, 15.0),
            _make_observation(2, 0.2, 0.40, 20.0),  # peak at t=0.2
            _make_observation(3, 0.3, 0.30, 12.0),
        ]

        # Feed frames up to peak + some after — all before deadline (0.2 + 1.0 = 1.2)
        results = []
        for item in observations:
            result = selector.consume(item)
            if result is not None:
                results.append(result)

        self.assertEqual(len(results), 0, "Peak should be pending, not emitted yet")

        # Feed frames past the deadline
        more_observations = [
            _make_observation(i, 0.4 + (i - 4) * 0.1, 0.05, 10.0)
            for i in range(4, 15)
        ]
        for item in more_observations:
            result = selector.consume(item)
            if result is not None:
                results.append(result)

        self.assertEqual(len(results), 1, "Peak should be emitted after deadline")
        self.assertEqual(results[0].peak.frame_index, 2)

    def test_clip_mode_window_contains_frames_around_peak(self):
        """Clip window should contain frames within [peak_t - half, peak_t + half]."""
        selector = TemporalPeakSelector(
            peak_threshold=0.15,
            cooldown_seconds=2.0,
            window_seconds=3.0,
            peak_neighborhood_frames=2,
            clip_half_duration=0.5,  # 1s total clip
        )
        # Build a sequence: low ... peak at t=1.0 ... low
        observations = []
        for i in range(30):
            t = i * 0.1
            score = 0.40 if i == 10 else (0.20 if i in (9, 11) else 0.05)
            observations.append(_make_observation(i, t, score, 10.0 + i))

        results = []
        for item in observations:
            result = selector.consume(item)
            if result is not None:
                results.append(result)

        self.assertEqual(len(results), 1)
        selection = results[0]
        peak_t = selection.peak.timestamp  # 1.0
        for obs in selection.window:
            self.assertGreaterEqual(obs.timestamp, peak_t - 0.5 - 1e-6)
            self.assertLessEqual(obs.timestamp, peak_t + 0.5 + 1e-6)

    def test_clip_mode_flush_emits_pending_at_stream_end(self):
        """flush() should emit pending peak even if deadline not reached."""
        selector = TemporalPeakSelector(
            peak_threshold=0.15,
            cooldown_seconds=5.0,
            window_seconds=5.0,
            peak_neighborhood_frames=2,
            clip_half_duration=2.5,  # 5s clip — deadline far in the future
        )
        observations = [
            _make_observation(0, 0.0, 0.05, 10.0),
            _make_observation(1, 0.1, 0.20, 15.0),
            _make_observation(2, 0.2, 0.40, 20.0),
            _make_observation(3, 0.3, 0.30, 12.0),
        ]

        for item in observations:
            selector.consume(item)

        flushed = selector.flush()
        self.assertIsNotNone(flushed)
        self.assertEqual(flushed.peak.frame_index, 2)

    def test_select_clip_frames_uniform_sampling(self):
        """_select_clip_frames should produce at most clip_max_frames, uniformly spaced."""
        if not CV2_AVAILABLE:
            self.skipTest("OpenCV is not available")

        from src.vision.visual_pipeline import PeakSelection

        config = VisualPerceptionConfig(
            clip_duration_seconds=2.0,
            clip_max_frames=4,
        )
        pipeline = VisualPerceptionPipeline(config=config)

        # 20 frames over 2 seconds
        window = [_make_observation(i, i * 0.1, 0.2, 10.0 + i) for i in range(20)]
        selection = PeakSelection(
            peak=window[10],
            representative=window[10],
            window=window,
        )

        selected = pipeline._select_clip_frames(selection)

        self.assertLessEqual(len(selected), 4)
        self.assertGreater(len(selected), 0)
        # Should be time-sorted
        timestamps = [o.timestamp for o in selected]
        self.assertEqual(timestamps, sorted(timestamps))

    def test_clip_mode_stores_resized_frame_in_observation(self):
        """In clip mode, _analyze_frame should store resized frame, not original."""
        if not CV2_AVAILABLE:
            self.skipTest("OpenCV is not available")

        from src.vision.visual_types import FramePacket

        config = VisualPerceptionConfig(
            clip_duration_seconds=2.0,
            resize_width=64,
        )
        pipeline = VisualPerceptionPipeline(config=config)
        detector_state = pipeline._build_detector_state()

        # Create a large frame
        large_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        packet = FramePacket(frame_index=0, timestamp=0.0, frame_bgr=large_frame)
        obs = pipeline._analyze_frame(packet, detector_state)

        # Should be resized, not original
        self.assertLessEqual(obs.frame_width, 64)
        self.assertLess(obs.frame_bgr.shape[1], 640)

    def test_clip_mode_preserves_original_frame_when_disabled(self):
        """When clip_duration_seconds=0, _analyze_frame should keep original resolution."""
        if not CV2_AVAILABLE:
            self.skipTest("OpenCV is not available")

        from src.vision.visual_types import FramePacket

        config = VisualPerceptionConfig(
            clip_duration_seconds=0.0,
            resize_width=64,
        )
        pipeline = VisualPerceptionPipeline(config=config)
        detector_state = pipeline._build_detector_state()

        large_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        packet = FramePacket(frame_index=0, timestamp=0.0, frame_bgr=large_frame)
        obs = pipeline._analyze_frame(packet, detector_state)

        self.assertEqual(obs.frame_width, 640)
        self.assertEqual(obs.frame_bgr.shape[1], 640)

    def test_prompt_builder_adds_temporal_hint_in_clip_mode(self):
        """build_visual_analysis_user_input should mention temporal continuity in clip mode."""
        clip_event = VisualEvent(
            event_id="clip-1",
            peak_frame_index=1,
            timestamp=1.0,
            peak_score=0.42,
            representative_frame_index=1,
            metrics={"clip_duration": 2.0},
        )
        no_clip_event = VisualEvent(
            event_id="noclip-1",
            peak_frame_index=1,
            timestamp=1.0,
            peak_score=0.42,
            representative_frame_index=1,
            metrics={"clip_duration": 0},
        )

        clip_prompt = build_visual_analysis_user_input(clip_event)
        no_clip_prompt = build_visual_analysis_user_input(no_clip_event)

        self.assertIn("连续帧序列", clip_prompt)
        self.assertIn("时序变化", clip_prompt)
        self.assertNotIn("连续帧序列", no_clip_prompt)
        self.assertNotIn("时序变化", no_clip_prompt)


class TestVisualAgentKeyframePassthrough(unittest.TestCase):
    """visual_event_to_agent_event 应把 keyframes 映射到 AgentEvent.images。"""

    def test_keyframes_passed_as_images(self):
        kf1 = _ImageResult(base64_data="abc", media_type="image/jpeg", original_url="")
        kf2 = _ImageResult(base64_data="def", media_type="image/jpeg", original_url="")
        event = VisualEvent(
            event_id="img-pass-1",
            peak_frame_index=0,
            timestamp=1.0,
            peak_score=0.5,
            representative_frame_index=0,
            keyframes=[kf1, kf2],
        )
        agent_event = visual_event_to_agent_event(event)
        self.assertEqual(len(agent_event.images), 2)
        self.assertIs(agent_event.images[0], kf1)
        self.assertIs(agent_event.images[1], kf2)

    def test_empty_keyframes_give_empty_images(self):
        event = VisualEvent(
            event_id="img-pass-2",
            peak_frame_index=0,
            timestamp=1.0,
            peak_score=0.3,
            representative_frame_index=0,
        )
        agent_event = visual_event_to_agent_event(event)
        self.assertEqual(agent_event.images, [])

    def test_scene_field_in_metadata(self):
        event = VisualEvent(
            event_id="scene-pass-1",
            peak_frame_index=0,
            timestamp=1.0,
            peak_score=0.4,
            representative_frame_index=0,
            analysis=VisualAnalysis(
                scene="书桌前",
                facts=["人物在打字"],
                agent_hint="人物坐在书桌前打字",
            ),
        )
        agent_event = visual_event_to_agent_event(event)
        self.assertEqual(agent_event.metadata["analysis"]["scene"], "书桌前")


class TestVisualSkillDetector(unittest.TestCase):
    """VisualSkillDetector regex 关键词检测。"""

    def setUp(self):
        self.detector = VisualSkillDetector()

    def test_detects_kan_kan(self):
        self.assertTrue(self.detector.detect("你能看看我在做什么吗"))

    def test_detects_kan_dao(self):
        self.assertTrue(self.detector.detect("你看到了什么"))

    def test_detects_hua_mian(self):
        self.assertTrue(self.detector.detect("画面上有什么"))

    def test_detects_zuo_shenme(self):
        self.assertTrue(self.detector.detect("我在做什么"))

    def test_detects_she_xiang_tou(self):
        self.assertTrue(self.detector.detect("摄像头能看到吗"))

    def test_no_match_for_unrelated(self):
        self.assertFalse(self.detector.detect("今天天气怎么样"))

    def test_no_match_for_empty(self):
        self.assertFalse(self.detector.detect(""))


class TestVisualSkillExecutor(unittest.TestCase):
    """VisualSkillExecutor 格式化摘要输出。"""

    def test_returns_summary_from_events(self):
        events = [
            VisualEvent(
                event_id="skill-1",
                peak_frame_index=0,
                timestamp=1.0,
                peak_score=0.5,
                representative_frame_index=0,
                analysis=VisualAnalysis(agent_hint="人物在喝水"),
            ),
            VisualEvent(
                event_id="skill-2",
                peak_frame_index=5,
                timestamp=2.0,
                peak_score=0.4,
                representative_frame_index=5,
                analysis=VisualAnalysis(agent_hint="人物放下杯子"),
            ),
        ]
        executor = VisualSkillExecutor(handler=lambda top_k=2: events[:top_k])
        result = executor.execute(top_k=2)
        self.assertIn("人物在喝水", result)
        self.assertIn("人物放下杯子", result)

    def test_returns_fallback_when_no_events(self):
        executor = VisualSkillExecutor(handler=lambda top_k=2: [])
        result = executor.execute()
        self.assertEqual(result, "当前画面暂无显著变化")

    def test_returns_string_message_directly(self):
        executor = VisualSkillExecutor(
            handler=lambda top_k=2: "视觉已启动，正在观察当前画面，请稍后再问一次。"
        )
        result = executor.execute()
        self.assertEqual(result, "视觉已启动，正在观察当前画面，请稍后再问一次。")

    def test_handles_exception_gracefully(self):
        def _fail(**kwargs):
            raise RuntimeError("pipeline error")

        executor = VisualSkillExecutor(handler=_fail)
        result = executor.execute()
        self.assertEqual(result, "")


# ── AdaptiveFrameSampler tests ──

class _SamplerBase(unittest.TestCase):
    """Shared setup for AdaptiveFrameSampler tests."""

    @classmethod
    def setUpClass(cls):
        sampler_path = MODULE_PATH.parent / "adaptive_sampler.py"
        spec = importlib.util.spec_from_file_location("src.vision.adaptive_sampler", sampler_path)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        cls._AdaptiveFrameSampler = mod.AdaptiveFrameSampler
        cls._AdaptiveFrameSamplerConfig = mod.AdaptiveFrameSamplerConfig

    def _make_sampler(self, **overrides):
        defaults = dict(
            stft_window_size=16,
            stft_hop_size=2,
            smoothing_alpha=0.5,
        )
        defaults.update(overrides)
        cfg = self._AdaptiveFrameSamplerConfig(**defaults)
        return self._AdaptiveFrameSampler(cfg, source_fps=30.0)


class _BlockTorchFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise ImportError(f"blocked import for {fullname}")
        return None


class TestAdaptiveFrameSamplerOptionalTorch(unittest.TestCase):
    def test_import_and_fft_fallback_work_without_torch(self):
        sampler_path = MODULE_PATH.parent / "adaptive_sampler.py"
        module_name = "src.vision.adaptive_sampler_no_torch"
        spec = importlib.util.spec_from_file_location(module_name, sampler_path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)

        blocker = _BlockTorchFinder()
        module = importlib.util.module_from_spec(spec)
        saved_torch_modules = {
            name: value
            for name, value in list(sys.modules.items())
            if name == "torch" or name.startswith("torch.")
        }
        sys.modules.pop(module_name, None)
        sys.modules[module_name] = module
        for name in saved_torch_modules:
            sys.modules.pop(name, None)
        sys.modules["torch"] = None
        sys.meta_path.insert(0, blocker)
        try:
            spec.loader.exec_module(module)
        finally:
            sys.meta_path.remove(blocker)
            sys.modules.pop(module_name, None)
            sys.modules.pop("torch", None)
            sys.modules.update(saved_torch_modules)

        self.assertFalse(module.TORCH_AVAILABLE)
        config = module.AdaptiveFrameSamplerConfig(
            stft_window_size=16,
            stft_hop_size=2,
            smoothing_alpha=0.5,
        )
        sampler = module.AdaptiveFrameSampler(config, source_fps=30.0)

        for i in range(32):
            score = 0.8 if i % 2 == 0 else 0.1
            sampler.update(change_score=score, smoothed_score=score, timestamp=i / 30.0)

        self.assertGreater(sampler.current_target_fps, 8.0)


class TestAdaptiveFrameSamplerStaticScene(_SamplerBase):
    def test_static_scene_reduces_fps(self):
        sampler = self._make_sampler()
        for i in range(32):
            sampler.update(change_score=0.02, smoothed_score=0.03, timestamp=i / 30.0)
        fps = sampler.current_target_fps
        self.assertLess(fps, 5.0)

    def test_active_scene_increases_fps(self):
        sampler = self._make_sampler()
        for i in range(32):
            score = 0.8 if i % 2 == 0 else 0.1
            sampler.update(change_score=score, smoothed_score=score, timestamp=i / 30.0)
        fps = sampler.current_target_fps
        self.assertGreater(fps, 8.0)


class TestAdaptiveFrameSamplerSpikeGuard(_SamplerBase):
    def test_spike_triggers_boost(self):
        sampler = self._make_sampler(spike_threshold=0.5)
        for i in range(16):
            sampler.update(0.02, 0.03, timestamp=i / 30.0)
        sampler.update(change_score=0.8, smoothed_score=0.7, timestamp=16 / 30.0)
        eff = sampler.effective_fps
        self.assertEqual(eff, 15.0)

    def test_spike_boost_decays(self):
        sampler = self._make_sampler(spike_threshold=0.5, spike_boost_seconds=1.0, smoothing_alpha=1.0)
        # Feed static data to drive FFT target_fps down
        for i in range(20):
            sampler.update(0.02, 0.03, timestamp=i / 30.0)
        # FFT target fps should now be low
        base_fps = sampler.current_target_fps
        self.assertLess(base_fps, 10.0)
        # Trigger spike
        spike_time = 1.0
        sampler.update(0.8, 0.7, timestamp=spike_time)
        with sampler._lock:
            self.assertEqual(sampler._effective_fps_locked(spike_time), 15.0)
        # After boost window expires
        with sampler._lock:
            fps_after = sampler._effective_fps_locked(spike_time + 2.0)
        self.assertLess(fps_after, 10.0)


class TestAdaptiveFrameSamplerWindow(_SamplerBase):
    def test_window_not_full_uses_max_fps(self):
        sampler = self._make_sampler()
        for i in range(5):
            sampler.update(0.02, 0.03, timestamp=i / 30.0)
        fps = sampler.current_target_fps
        self.assertEqual(fps, 15.0)

    def test_spectral_activity_ratio_zero_when_no_data(self):
        sampler = self._make_sampler()
        self.assertEqual(sampler.spectral_activity_ratio, 0.0)


class TestAdaptiveFrameSamplerShouldSample(_SamplerBase):
    def test_should_sample_respects_interval(self):
        sampler = self._make_sampler(fps_min=5.0, fps_max=5.0, smoothing_alpha=1.0)
        # Fill window to force FFT computation
        for i in range(20):
            sampler.update(0.02, 0.03, timestamp=i / 30.0)
        # At 5 fps, interval = 0.2s. At 30fps source, sample every 6th frame.
        sampled = 0
        for i in range(30):
            if sampler.should_sample(i, i / 30.0):
                sampled += 1
        # Expect roughly 5 samples in 1 second
        self.assertGreaterEqual(sampled, 3)
        self.assertLessEqual(sampled, 7)


if __name__ == "__main__":
    unittest.main()
