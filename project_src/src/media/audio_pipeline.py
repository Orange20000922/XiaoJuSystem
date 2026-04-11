"""TTS audio pipeline with pluggable providers.

Supported providers:
- CosyVoice / CosyVoice2
- edge-tts
- IndexTTS2

The pipeline keeps the original queueing and optional autoplay behavior,
but no longer assumes every backend returns WAV bytes. Each provider reports
its preferred output extension so cache files and temp playback files match the
actual audio format.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import os
import sys
import tempfile
import time
import uuid
import wave
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.logger import logger


_cv_repo = os.environ.get("COSYVOICE_REPO_DIR", "")
if _cv_repo and _cv_repo not in sys.path:
    sys.path.insert(0, _cv_repo)
    sys.path.insert(0, str(Path(_cv_repo) / "third_party" / "Matcha-TTS"))

try:
    import torch
    import torchaudio
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

try:
    from cosyvoice.cli.cosyvoice import CosyVoice2
    COSYVOICE_AVAILABLE = True
except ImportError:
    try:
        from cosyvoice.cli.cosyvoice import CosyVoice as CosyVoice2
        COSYVOICE_AVAILABLE = True
    except ImportError:
        COSYVOICE_AVAILABLE = False


def _normalize_tts_provider(provider: str) -> str:
    normalized = (provider or "cosyvoice").strip().lower().replace("_", "-")
    aliases = {
        "cosyvoice": "cosyvoice",
        "cosyvoice2": "cosyvoice",
        "edge-tts": "edge-tts",
        "edgetts": "edge-tts",
        "index-tts2": "indextts2",
        "index-tts": "indextts2",
        "indextts2": "indextts2",
    }
    return aliases.get(normalized, normalized)


class PlaybackBehavior(Enum):
    """Playback behavior for queued TTS requests."""

    QUEUE = "queue"
    INTERRUPT = "interrupt"
    REPLACE = "replace"


@dataclass
class SpeechIntent:
    """Queued speech request."""

    intent_id: str
    text: str
    emotion: str = "neutral"
    priority: int = 0
    behavior: PlaybackBehavior = PlaybackBehavior.QUEUE
    created_at: float = field(default_factory=time.time)


@dataclass
class AudioConfig:
    """Local audio/TTS configuration."""

    enabled: bool = True
    tts_provider: str = "cosyvoice"

    cosyvoice_repo_dir: str = ""
    cosyvoice_model_dir: str = "./models/CosyVoice2-0.5B"
    ref_audio_dir: str = "./data/audio_refs"
    default_ref_audio: str = "default.wav"
    default_ref_text: str = ""

    edge_tts_voice: str = "zh-CN-XiaoxiaoNeural"
    edge_tts_rate: str = "+0%"
    edge_tts_volume: str = "+0%"
    edge_tts_pitch: str = "+0Hz"
    edge_tts_proxy: Optional[str] = None

    indextts2_repo_dir: str = ""
    indextts2_model_dir: str = "./models/IndexTTS2"
    indextts2_cfg_path: str = ""
    indextts2_speaker_audio: str = ""
    indextts2_emotion_audio: str = ""
    indextts2_emo_text: str = ""
    indextts2_emo_vector: List[float] = field(default_factory=list)
    indextts2_emo_alpha: float = 0.9
    indextts2_use_emo_text: bool = False
    indextts2_use_random: bool = False
    indextts2_use_fp16: bool = False
    indextts2_use_cuda_kernel: bool = False
    indextts2_use_deepspeed: bool = False

    sample_rate: int = 22050
    speed: float = 1.0
    emotion_ref_map: Dict[str, Dict[str, str]] = field(default_factory=dict)
    cache_dir: str = "./data/audio_cache"
    cache_enabled: bool = True
    auto_play: bool = False


class BaseTTSClient:
    """Shared interface for pluggable TTS backends."""

    provider_name: str = "base"
    output_extension: str = ".wav"

    def __init__(self, config: AudioConfig):
        self.config = config
        self._loaded = False

    def load(self):
        raise NotImplementedError

    def synthesize(self, text: str, emotion: str = "neutral") -> bytes:
        raise NotImplementedError

    @property
    def available(self) -> bool:
        return self._loaded


class CosyVoiceClient(BaseTTSClient):
    """CosyVoice / CosyVoice2 backend."""

    provider_name = "cosyvoice"
    output_extension = ".wav"

    def __init__(self, config: AudioConfig):
        super().__init__(config)
        self.model: Optional["CosyVoice2"] = None
        self.ref_paths: Dict[str, str] = {}
        self.ref_texts: Dict[str, str] = {}
        self._sample_rate: int = config.sample_rate

    def load(self):
        repo_dir = self.config.cosyvoice_repo_dir
        if repo_dir:
            repo_path = Path(repo_dir)
            for path_entry in [str(repo_path), str(repo_path / "third_party" / "Matcha-TTS")]:
                if path_entry not in sys.path:
                    sys.path.insert(0, path_entry)

        global COSYVOICE_AVAILABLE, CosyVoice2
        if not COSYVOICE_AVAILABLE:
            try:
                from cosyvoice.cli.cosyvoice import CosyVoice2 as _CV2
                CosyVoice2 = _CV2
                COSYVOICE_AVAILABLE = True
            except ImportError:
                try:
                    from cosyvoice.cli.cosyvoice import CosyVoice as _CV2
                    CosyVoice2 = _CV2
                    COSYVOICE_AVAILABLE = True
                except ImportError:
                    pass

        if not COSYVOICE_AVAILABLE:
            raise ImportError(
                "CosyVoice is unavailable. Set audio.cosyvoice.repo_dir to the local "
                "CosyVoice repository path first."
            )
        if not TORCH_AVAILABLE:
            raise ImportError("CosyVoice requires torch and torchaudio.")

        model_dir = Path(self.config.cosyvoice_model_dir)
        if not model_dir.exists():
            raise FileNotFoundError(f"CosyVoice model directory does not exist: {model_dir}")

        logger.info(f"Loading CosyVoice model from {model_dir}")
        self.model = CosyVoice2(str(model_dir))
        self._sample_rate = getattr(self.model, "sample_rate", self.config.sample_rate)
        self._load_ref_audios()
        self._loaded = True
        logger.info(f"CosyVoice ready at sample_rate={self._sample_rate}")

    def _load_ref_audios(self):
        ref_dir = Path(self.config.ref_audio_dir)
        if not ref_dir.exists():
            logger.warning(f"Reference audio directory does not exist: {ref_dir}; creating it")
            ref_dir.mkdir(parents=True, exist_ok=True)
            return

        default_path = ref_dir / self.config.default_ref_audio
        if default_path.exists():
            self.ref_paths["neutral"] = str(default_path)
            self.ref_texts["neutral"] = self.config.default_ref_text
        else:
            logger.warning(f"Default reference audio is missing: {default_path}")

        for emotion, ref_info in self.config.emotion_ref_map.items():
            audio_file = ref_info.get("audio", "")
            if not audio_file:
                continue
            audio_path = ref_dir / audio_file
            if audio_path.exists():
                self.ref_paths[emotion] = str(audio_path)
                self.ref_texts[emotion] = ref_info.get("text", "")
            else:
                logger.warning(f"Emotion reference audio is missing: {audio_path}")

    def synthesize(self, text: str, emotion: str = "neutral") -> bytes:
        if not self._loaded or self.model is None:
            raise RuntimeError("CosyVoice client is not loaded")

        ref_text = self.ref_texts.get(emotion) or self.ref_texts.get("neutral", "")
        ref_path = self.ref_paths.get(emotion) or self.ref_paths.get("neutral")
        if ref_path is None:
            raise RuntimeError(
                "No usable reference audio found. Put at least one reference clip under "
                f"{self.config.ref_audio_dir}."
            )

        import numpy as np

        audio_chunks = []
        for chunk in self.model.inference_zero_shot(
            tts_text=text,
            prompt_text=ref_text,
            prompt_wav=ref_path,
            stream=False,
        ):
            audio_chunks.append(chunk["tts_speech"].numpy())

        if not audio_chunks:
            raise RuntimeError("CosyVoice returned no audio data")

        audio_array = np.concatenate(audio_chunks, axis=-1)
        if audio_array.ndim > 1:
            audio_array = audio_array.squeeze(0)
        return self._numpy_to_wav(audio_array, self._sample_rate)

    @staticmethod
    def _numpy_to_wav(audio, sample_rate: int) -> bytes:
        import numpy as np

        if audio.dtype in (np.float32, np.float64):
            audio = np.clip(audio, -1.0, 1.0)
            audio = (audio * 32767).astype(np.int16)

        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio.tobytes())
        return buffer.getvalue()


class EdgeTTSClient(BaseTTSClient):
    """edge-tts backend."""

    provider_name = "edge-tts"
    output_extension = ".mp3"

    def __init__(self, config: AudioConfig):
        super().__init__(config)
        self._edge_tts = None

    def load(self):
        try:
            import edge_tts as edge_tts_module
        except ImportError as exc:
            raise ImportError(
                "edge-tts is not installed. Install it in this repo virtual environment first."
            ) from exc

        self._edge_tts = edge_tts_module
        self._loaded = True
        logger.info(f"edge-tts ready with voice={self.config.edge_tts_voice}")

    def synthesize(self, text: str, emotion: str = "neutral") -> bytes:
        if not self._loaded or self._edge_tts is None:
            raise RuntimeError("edge-tts client is not loaded")

        with tempfile.NamedTemporaryFile(suffix=self.output_extension, delete=False) as handle:
            temp_path = handle.name

        try:
            communicate = self._edge_tts.Communicate(
                text=text,
                voice=self.config.edge_tts_voice,
                rate=self.config.edge_tts_rate,
                volume=self.config.edge_tts_volume,
                pitch=self.config.edge_tts_pitch,
                proxy=self.config.edge_tts_proxy,
            )
            communicate.save_sync(temp_path)
            return Path(temp_path).read_bytes()
        finally:
            try:
                os.unlink(temp_path)
            except OSError:
                pass


class IndexTTS2Client(BaseTTSClient):
    """IndexTTS2 backend."""

    provider_name = "indextts2"
    output_extension = ".wav"

    def __init__(self, config: AudioConfig):
        super().__init__(config)
        self.model = None

    def load(self):
        repo_dir = self.config.indextts2_repo_dir
        if repo_dir:
            repo_path = str(Path(repo_dir))
            if repo_path not in sys.path:
                sys.path.insert(0, repo_path)

        try:
            from indextts.infer_v2 import IndexTTS2
        except ImportError as exc:
            raise ImportError(
                "IndexTTS2 is unavailable. Install the package in this repo virtual environment, "
                "or set audio.indextts2.repo_dir to the local IndexTTS2 repository."
            ) from exc

        model_dir = Path(self.config.indextts2_model_dir)
        if not model_dir.exists():
            raise FileNotFoundError(f"IndexTTS2 model directory does not exist: {model_dir}")

        cfg_path = Path(self.config.indextts2_cfg_path) if self.config.indextts2_cfg_path else model_dir / "config.yaml"
        if not cfg_path.exists():
            raise FileNotFoundError(f"IndexTTS2 config file does not exist: {cfg_path}")

        logger.info(f"Loading IndexTTS2 model from {model_dir}")
        self.model = IndexTTS2(
            cfg_path=str(cfg_path),
            model_dir=str(model_dir),
            use_fp16=bool(self.config.indextts2_use_fp16),
            use_cuda_kernel=bool(self.config.indextts2_use_cuda_kernel),
            use_deepspeed=bool(self.config.indextts2_use_deepspeed),
        )
        self._loaded = True
        logger.info("IndexTTS2 ready")

    def synthesize(self, text: str, emotion: str = "neutral") -> bytes:
        if not self._loaded or self.model is None:
            raise RuntimeError("IndexTTS2 client is not loaded")

        speaker_audio = self._resolve_audio_path(self.config.indextts2_speaker_audio, required=True)
        emotion_audio, emotion_text = self._resolve_emotion_prompt(emotion)

        kwargs = {
            "spk_audio_prompt": speaker_audio,
            "text": text,
            "verbose": False,
            "use_random": bool(self.config.indextts2_use_random),
        }

        emo_alpha = float(self.config.indextts2_emo_alpha)
        if emotion_audio:
            kwargs["emo_audio_prompt"] = emotion_audio
            kwargs["emo_alpha"] = emo_alpha
        elif self.config.indextts2_emo_vector:
            kwargs["emo_vector"] = list(self.config.indextts2_emo_vector)
        elif emotion_text:
            kwargs["use_emo_text"] = True
            kwargs["emo_text"] = emotion_text
            kwargs["emo_alpha"] = emo_alpha
        elif self.config.indextts2_use_emo_text:
            kwargs["use_emo_text"] = True
            if self.config.indextts2_emo_text:
                kwargs["emo_text"] = self.config.indextts2_emo_text
                kwargs["emo_alpha"] = emo_alpha

        with tempfile.NamedTemporaryFile(suffix=self.output_extension, delete=False) as handle:
            temp_path = handle.name

        try:
            kwargs["output_path"] = temp_path
            self.model.infer(**kwargs)
            return Path(temp_path).read_bytes()
        finally:
            try:
                os.unlink(temp_path)
            except OSError:
                pass

    def _resolve_emotion_prompt(self, emotion: str) -> tuple[Optional[str], str]:
        prompt_audio = self._resolve_audio_path(self.config.indextts2_emotion_audio)
        prompt_text = self.config.indextts2_emo_text

        ref_info = self.config.emotion_ref_map.get(emotion, {})
        ref_audio = self._resolve_audio_path(ref_info.get("audio", ""))
        if ref_audio:
            prompt_audio = ref_audio
        if ref_info.get("text"):
            prompt_text = ref_info["text"]
        return prompt_audio, prompt_text

    def _resolve_audio_path(self, value: str, *, required: bool = False) -> Optional[str]:
        raw = (value or "").strip()
        if not raw:
            if required:
                raise RuntimeError("IndexTTS2 requires audio.indextts2.speaker_audio")
            return None

        path = Path(raw)
        if not path.is_absolute():
            ref_candidate = Path(self.config.ref_audio_dir) / raw
            path = ref_candidate if ref_candidate.exists() else Path(raw)

        if required and not path.exists():
            raise FileNotFoundError(f"Audio prompt file does not exist: {path}")
        if not required and not path.exists():
            logger.warning(f"Optional audio prompt file does not exist: {path}")
            return None
        return str(path)


class AudioPipeline:
    """Queued TTS pipeline with provider-based synthesis."""

    def __init__(self, config: AudioConfig):
        self.config = config
        self.tts_client: Optional[BaseTTSClient] = None
        self._intent_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self._active_intent: Optional[SpeechIntent] = None
        self._is_playing: bool = False
        self._process_task: Optional[asyncio.Task] = None
        self._stop_event: asyncio.Event = asyncio.Event()
        self._cache_dir = Path(config.cache_dir)
        self._cache_extension = ".wav"
        self._provider_key = _normalize_tts_provider(config.tts_provider)

        if config.cache_enabled:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
        if config.auto_play and PYGAME_AVAILABLE:
            pygame.mixer.init()

    async def start(self):
        if not self.config.enabled:
            logger.info("Audio pipeline disabled")
            return

        loop = asyncio.get_running_loop()
        self.tts_client = self._build_tts_client()
        await loop.run_in_executor(None, self.tts_client.load)
        self._cache_extension = self.tts_client.output_extension

        self._stop_event.clear()
        self._process_task = asyncio.create_task(self._process_loop())
        logger.info(f"Audio pipeline started with provider={self._provider_key}")

    async def stop(self):
        self._stop_event.set()
        if self._process_task:
            self._process_task.cancel()
            try:
                await self._process_task
            except asyncio.CancelledError:
                pass
        if self.config.auto_play and PYGAME_AVAILABLE:
            pygame.mixer.quit()
        logger.info("Audio pipeline stopped")

    async def speak(
        self,
        text: str,
        emotion: str = "neutral",
        priority: int = 0,
        behavior: PlaybackBehavior = PlaybackBehavior.QUEUE,
    ) -> str:
        intent = SpeechIntent(
            intent_id=uuid.uuid4().hex[:12],
            text=text,
            emotion=emotion,
            priority=priority,
            behavior=behavior,
        )

        if behavior == PlaybackBehavior.REPLACE:
            while not self._intent_queue.empty():
                try:
                    self._intent_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

        await self._intent_queue.put((-priority, time.time(), intent))
        return intent.intent_id

    async def synthesize(self, text: str, emotion: str = "neutral") -> bytes:
        if not self.tts_client or not self.tts_client.available:
            raise RuntimeError("TTS client is not initialized")

        cache_key = self._cache_key(text, emotion)
        cached = self._load_cache(cache_key)
        if cached:
            return cached

        loop = asyncio.get_running_loop()
        audio_bytes = await loop.run_in_executor(None, self.tts_client.synthesize, text, emotion)
        self._save_cache(cache_key, audio_bytes)
        return audio_bytes

    async def _process_loop(self):
        while not self._stop_event.is_set():
            try:
                _, _, intent = await asyncio.wait_for(self._intent_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            self._active_intent = intent
            try:
                audio_bytes = await self.synthesize(intent.text, intent.emotion)
                if self.config.auto_play and PYGAME_AVAILABLE:
                    await self._play_audio(audio_bytes)
            except Exception as exc:
                logger.error(f"TTS synthesis failed [{intent.intent_id}]: {exc}")
            finally:
                self._active_intent = None

    async def _play_audio(self, audio_bytes: bytes):
        with tempfile.NamedTemporaryFile(
            suffix=self._cache_extension,
            delete=False,
            dir=str(self._cache_dir),
        ) as handle:
            handle.write(audio_bytes)
            temp_path = handle.name

        try:
            self._is_playing = True
            pygame.mixer.music.load(temp_path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                await asyncio.sleep(0.05)
        finally:
            self._is_playing = False
            try:
                os.unlink(temp_path)
            except OSError:
                pass

    def _build_tts_client(self) -> BaseTTSClient:
        provider = self._provider_key
        if provider == "cosyvoice":
            return CosyVoiceClient(self.config)
        if provider == "edge-tts":
            return EdgeTTSClient(self.config)
        if provider == "indextts2":
            return IndexTTS2Client(self.config)
        raise ValueError(f"Unsupported TTS provider: {self.config.tts_provider}")

    def _cache_key(self, text: str, emotion: str) -> str:
        raw = "|".join(
            [
                self._provider_key,
                text,
                emotion,
                str(self.config.speed),
                self.config.edge_tts_voice,
                self.config.edge_tts_rate,
                self.config.edge_tts_volume,
                self.config.edge_tts_pitch,
                self.config.indextts2_model_dir,
                self.config.indextts2_cfg_path,
                self.config.indextts2_speaker_audio,
                self.config.indextts2_emotion_audio,
                self.config.indextts2_emo_text,
                self.config.cosyvoice_model_dir,
                self.config.default_ref_audio,
            ]
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def _load_cache(self, key: str) -> Optional[bytes]:
        if not self.config.cache_enabled:
            return None
        cache_path = self._cache_dir / f"{key}{self._cache_extension}"
        if cache_path.exists():
            return cache_path.read_bytes()
        return None

    def _save_cache(self, key: str, data: bytes):
        if not self.config.cache_enabled:
            return
        cache_path = self._cache_dir / f"{key}{self._cache_extension}"
        try:
            cache_path.write_bytes(data)
        except OSError as exc:
            logger.warning(f"Failed to write audio cache: {exc}")

    @property
    def is_speaking(self) -> bool:
        return self._is_playing

    @property
    def queue_size(self) -> int:
        return self._intent_queue.qsize()

    @property
    def available(self) -> bool:
        return self.config.enabled and self.tts_client is not None and self.tts_client.available

