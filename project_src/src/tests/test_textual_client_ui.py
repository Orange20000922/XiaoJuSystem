import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

try:
    from rich.text import Text
    from src.client.vision_runtime import VisionRuntimeController
    from src.client.tui.pixel_art import AnsiPixelArtRenderer, PixelArtRegistry
    from src.client.tui.textual_app import TextualClientApp
    from src.core_engine.api import DirectRuntime
    from src.vision import VisualAnalysis, VisualEvent, VisualWindowSummary
except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
    Text = None
    _UI_IMPORT_ERROR = exc
else:
    _UI_IMPORT_ERROR = None


class _FakeMemory:
    def __init__(self):
        self.working_memory = []
        self.visual_summaries = []
        self.emotion_snapshots = []

    def add_visual_session_summary_l3(self, content: str):
        self.visual_summaries.append(content)

    def add_emotion_snapshot_l4(self, snapshot, source: str = "runtime"):
        self.emotion_snapshots.append((snapshot, source))


class _FakeTarget:
    def __init__(self):
        self.personality = SimpleNamespace(name="retro-unit")
        self.llm_client = SimpleNamespace(model="debug-model")
        self.memory = _FakeMemory()
        self.emotion_state_tracker = SimpleNamespace(to_dict=lambda: {"valence": 0.2, "arousal": 0.4})
        self.visual_perception_config = None
        self.registered_visual_handler = None
        self.chat_calls = []

    def chat(self, text, **kwargs):
        self.chat_calls.append((text, kwargs))
        return {
            "response": f"echo:{text}",
            "should_respond": True,
            "emotion": {"primary": "neutral", "intensity": 0.0},
            "behavior": {"type": "respond_positive", "tone": "calm"},
        }

    def register_visual_skill(self, handler):
        self.registered_visual_handler = handler

    def close(self):
        return None


class _FakeSlowTarget(_FakeTarget):
    def __init__(self):
        super().__init__()
        self.chat_started = False
        self.chat_released = False

    def chat(self, text, **kwargs):
        self.chat_started = True
        time.sleep(0.2)
        self.chat_released = True
        return super().chat(text, **kwargs)


class TextualClientUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if _UI_IMPORT_ERROR is not None:
            raise unittest.SkipTest(f"UI dependencies unavailable: {_UI_IMPORT_ERROR}")

    def test_ansi_pixel_renderer_returns_rich_text(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "portrait.png"
            image = Image.new("RGBA", (4, 4), (0, 0, 0, 0))
            image.putpixel((0, 0), (255, 0, 0, 255))
            image.putpixel((0, 1), (0, 255, 0, 255))
            image.save(path)

            registry = PixelArtRegistry()
            registry.register("portrait", path)

            renderable = registry.preview_renderable(AnsiPixelArtRenderer(), width=8)

            self.assertIsInstance(renderable, Text)
            self.assertGreater(len(renderable.plain), 0)

    def test_textual_app_can_be_constructed_with_fake_runtime(self):
        runtime = DirectRuntime(_FakeTarget())
        app = TextualClientApp(runtime)

        self.assertEqual(app.session.persona_name, "retro-unit")
        self.assertEqual(app.session.llm_model, "debug-model")
        self.assertIsNotNone(app.runtime.target.registered_visual_handler)

    def test_vision_runtime_registers_visual_skill_handler(self):
        target = _FakeTarget()
        runtime = DirectRuntime(target)
        app = TextualClientApp(runtime)

        self.assertIsNotNone(target.registered_visual_handler)
        self.assertEqual(target.registered_visual_handler.__self__, app.vision)

    def test_vision_runtime_promoted_event_uses_visual_direct_chat(self):
        target = _FakeSlowTarget()
        runtime = DirectRuntime(target)
        app = TextualClientApp(runtime)
        event = VisualEvent(
            event_id="visual-1",
            peak_frame_index=10,
            timestamp=1.5,
            peak_score=0.9,
            representative_frame_index=10,
            analysis=VisualAnalysis(agent_hint="画面里有人抬手"),
        )

        app.vision._handle_promoted_event(event)

        self.assertEqual(len(target.chat_calls), 0)
        deadline = time.time() + 1.0
        while not target.chat_released and time.time() < deadline:
            time.sleep(0.02)

        self.assertEqual(len(target.chat_calls), 1)
        _, kwargs = target.chat_calls[0]
        self.assertTrue(kwargs["visual_direct"])
        self.assertEqual(kwargs["metadata"]["event_id"], "visual-1")

    def test_vision_runtime_visual_skill_can_auto_start(self):
        target = _FakeTarget()
        runtime = DirectRuntime(target)
        controller = VisionRuntimeController.from_runtime(
            runtime,
            TextualClientApp(runtime).session,
        )
        controller._skill_wait_timeout_seconds = 0.01
        controller._skill_poll_interval_seconds = 0.001

        start_calls = []

        def _fake_start(source, **kwargs):
            start_calls.append((source, kwargs))
            return True

        controller.start = _fake_start
        result = target.registered_visual_handler(top_k=2)

        self.assertEqual(len(start_calls), 1)
        self.assertEqual(start_calls[0][0], 0)
        self.assertEqual(result, "视觉已启动，正在观察当前画面，请稍后再问一次。")

    def test_vision_runtime_flushes_summary_and_emotion_snapshot_on_stop(self):
        target = _FakeTarget()
        runtime = DirectRuntime(target)
        controller = VisionRuntimeController.from_runtime(
            runtime,
            TextualClientApp(runtime).session,
        )
        controller._last_summaries = [
            VisualWindowSummary(
                window_index=0,
                start_timestamp=0.0,
                end_timestamp=30.0,
                total_events=3,
                total_clusters=1,
            )
        ]

        controller._flush_visual_shutdown_memory()

        self.assertEqual(len(target.memory.visual_summaries), 1)
        self.assertEqual(len(target.memory.emotion_snapshots), 1)
        self.assertEqual(target.memory.emotion_snapshots[0][1], "vision_runtime_stop")


if __name__ == "__main__":
    unittest.main()
