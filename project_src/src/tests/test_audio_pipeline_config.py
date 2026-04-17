import logging
import sys
import types
import unittest
from pathlib import Path

_project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_project_root))

_src_root = str(_project_root / "src")
if "src" not in sys.modules:
    _pkg = types.ModuleType("src")
    _pkg.__path__ = [_src_root]
    sys.modules["src"] = _pkg
if "src.logger" not in sys.modules:
    _logger_mod = types.ModuleType("src.logger")
    _logger_mod.logger = logging.getLogger("test_stub")
    sys.modules["src.logger"] = _logger_mod
if "src.media" not in sys.modules:
    _media = types.ModuleType("src.media")
    _media.__path__ = [str(_project_root / "src" / "media")]
    sys.modules["src.media"] = _media

from configs.config_loader import load_audio_config
from src.media.audio_pipeline import AudioConfig, AudioPipeline, _normalize_tts_provider


class AudioPipelineConfigTests(unittest.TestCase):
    def test_provider_normalization(self):
        self.assertEqual(_normalize_tts_provider("edgetts"), "edge-tts")
        self.assertEqual(_normalize_tts_provider("edge_tts"), "edge-tts")
        self.assertEqual(_normalize_tts_provider("index-tts2"), "indextts2")
        self.assertEqual(_normalize_tts_provider("IndexTTS2"), "indextts2")
        self.assertEqual(_normalize_tts_provider("cosyvoice2"), "cosyvoice")

    def test_load_audio_config_parses_new_provider_sections(self):
        cfg = {
            "audio": {
                "enabled": True,
                "tts_provider": "edgetts",
                "cache_dir": "./cache/audio",
                "cache_enabled": True,
                "auto_play": False,
                "emotion_ref_map": {
                    "joy": {"audio": "happy.wav", "text": "开心一点"}
                },
                "cosyvoice": {
                    "repo_dir": "D:/repos/CosyVoice",
                    "model_dir": "./models/cosy",
                    "ref_audio_dir": "./refs",
                    "default_ref_audio": "default.wav",
                    "default_ref_text": "参考文本",
                    "sample_rate": 24000,
                    "speed": 1.1,
                },
                "edge_tts": {
                    "voice": "zh-CN-YunxiNeural",
                    "rate": "+10%",
                    "volume": "+5%",
                    "pitch": "+2Hz",
                    "proxy": "http://127.0.0.1:7890",
                },
                "indextts2": {
                    "repo_dir": "D:/repos/index-tts",
                    "model_dir": "./models/IndexTTS2",
                    "cfg_path": "./models/IndexTTS2/config.yaml",
                    "speaker_audio": "./refs/speaker.wav",
                    "emotion_audio": "./refs/emotion.wav",
                    "emo_text": "平静、温柔",
                    "emo_vector": [0.1, 0.2],
                    "emo_alpha": 0.8,
                    "use_emo_text": True,
                    "use_random": False,
                    "use_fp16": False,
                    "use_cuda_kernel": False,
                    "use_deepspeed": False,
                },
            }
        }

        audio_cfg = load_audio_config(cfg)
        self.assertEqual(audio_cfg.tts_provider, "edgetts")
        self.assertEqual(audio_cfg.edge_tts_voice, "zh-CN-YunxiNeural")
        self.assertEqual(audio_cfg.edge_tts_proxy, "http://127.0.0.1:7890")
        self.assertEqual(audio_cfg.indextts2_repo_dir, "D:/repos/index-tts")
        self.assertEqual(audio_cfg.indextts2_cfg_path, "./models/IndexTTS2/config.yaml")
        self.assertEqual(audio_cfg.indextts2_speaker_audio, "./refs/speaker.wav")
        self.assertEqual(audio_cfg.indextts2_emotion_audio, "./refs/emotion.wav")
        self.assertEqual(audio_cfg.indextts2_emo_text, "平静、温柔")
        self.assertEqual(audio_cfg.indextts2_emo_vector, [0.1, 0.2])
        self.assertEqual(audio_cfg.sample_rate, 24000)
        self.assertAlmostEqual(audio_cfg.speed, 1.1)
        self.assertIn("joy", audio_cfg.emotion_ref_map)

    def test_pipeline_builds_expected_client_from_provider_alias(self):
        edge_pipeline = AudioPipeline(AudioConfig(tts_provider="edgetts", enabled=True))
        index_pipeline = AudioPipeline(AudioConfig(tts_provider="index-tts2", enabled=True))
        cosy_pipeline = AudioPipeline(AudioConfig(tts_provider="cosyvoice2", enabled=True))

        self.assertEqual(edge_pipeline._build_tts_client().provider_name, "edge-tts")
        self.assertEqual(index_pipeline._build_tts_client().provider_name, "indextts2")
        self.assertEqual(cosy_pipeline._build_tts_client().provider_name, "cosyvoice")


if __name__ == "__main__":
    unittest.main()
