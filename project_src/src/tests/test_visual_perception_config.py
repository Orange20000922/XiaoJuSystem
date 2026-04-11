import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from configs.config_loader import load_visual_perception_config


class VisualPerceptionConfigTests(unittest.TestCase):
    def test_load_visual_perception_config_defaults(self):
        cfg = load_visual_perception_config({})
        self.assertFalse(cfg.enabled)
        self.assertEqual(cfg.resize_width, 320)
        self.assertEqual(cfg.vision_calls_per_minute, 2)
        self.assertTrue(cfg.route_visual_events_to_chat)
        self.assertTrue(cfg.inject_to_emotion_state)
        self.assertTrue(cfg.persist_to_memory)

    def test_load_visual_perception_config_custom_values(self):
        cfg = load_visual_perception_config(
            {
                "visual_perception": {
                    "enabled": True,
                    "resize_width": 256,
                    "mog2_history": 240,
                    "peak_threshold": 0.2,
                    "route_visual_events_to_chat": False,
                    "inject_to_emotion_state": True,
                    "persist_to_memory": False,
                    "visual_emotion_scale": 0.15,
                    "memory_peak_score_threshold": 0.3,
                }
            }
        )
        self.assertTrue(cfg.enabled)
        self.assertEqual(cfg.resize_width, 256)
        self.assertEqual(cfg.mog2_history, 240)
        self.assertAlmostEqual(cfg.peak_threshold, 0.2)
        self.assertFalse(cfg.route_visual_events_to_chat)
        self.assertTrue(cfg.inject_to_emotion_state)
        self.assertFalse(cfg.persist_to_memory)
        self.assertAlmostEqual(cfg.visual_emotion_scale, 0.15)
        self.assertAlmostEqual(cfg.memory_peak_score_threshold, 0.3)


if __name__ == "__main__":
    unittest.main()
