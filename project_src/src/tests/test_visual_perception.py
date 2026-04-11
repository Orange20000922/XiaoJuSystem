import importlib.util
import sys
import types
import unittest
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MODULE_PATH = Path(__file__).resolve().parents[1] / "vision" / "visual_perception.py"


class _DummyLogger:
    def debug(self, *args, **kwargs):
        return None

    info = warning = error = debug


@dataclass
class _ImageResult:
    base64_data: str
    media_type: str
    original_url: str


src_pkg = types.ModuleType("src")
src_pkg.__path__ = []
logger_module = types.ModuleType("src.logger")
logger_module.logger = _DummyLogger()
media_pkg = types.ModuleType("src.media")
media_pkg.__path__ = []
image_utils_module = types.ModuleType("src.media.image_utils")
image_utils_module.ImageResult = _ImageResult

sys.modules.setdefault("src", src_pkg)
sys.modules["src.logger"] = logger_module
sys.modules["src.media"] = media_pkg
sys.modules["src.media.image_utils"] = image_utils_module

spec = importlib.util.spec_from_file_location("visual_perception_under_test", MODULE_PATH)
visual_module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = visual_module
spec.loader.exec_module(visual_module)

FrameObservation = visual_module.FrameObservation
TemporalPeakSelector = visual_module.TemporalPeakSelector
VisualAnalysis = visual_module.VisualAnalysis
VisualEvent = visual_module.VisualEvent
default_visual_event_text = visual_module.default_visual_event_text
derive_visual_emotion_signal = visual_module.derive_visual_emotion_signal
sigmoid_normalize = visual_module.sigmoid_normalize
visual_event_to_agent_text = visual_module.visual_event_to_agent_text


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


if __name__ == "__main__":
    unittest.main()
