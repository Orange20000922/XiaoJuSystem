"""
TTS 语音合成管道

基于 CosyVoice 的 zero-shot 音色克隆语音合成。
支持情绪参考音频切换、优先级队列、播放控制。

CosyVoice 支持两种模式：
  1. zero-shot: 使用 3-10 秒参考音频克隆音色（无需训练）
  2. sft: 使用微调后的模型（需要训练）

用法：
    from src.audio_pipeline import AudioPipeline

    pipeline = AudioPipeline(config)
    await pipeline.start()

    # 合成并播放
    await pipeline.speak("你好呀！", emotion="joy")

    # 仅合成（返回音频字节）
    audio_bytes = await pipeline.synthesize("你好呀！", emotion="joy")

    await pipeline.stop()
"""

import asyncio
import io
import os
import time
import wave
import uuid
import hashlib
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Optional

import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.logger import logger

# CosyVoice 可选导入
# CosyVoice 是源码包，需要把仓库路径加入 sys.path
# 路径通过 AudioConfig.cosyvoice_repo_dir 配置，或环境变量 COSYVOICE_REPO_DIR
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
    from cosyvoice.cli.cosyvoice import CosyVoice2
    COSYVOICE_AVAILABLE = True
except ImportError:
    try:
        from cosyvoice.cli.cosyvoice import CosyVoice as CosyVoice2
        COSYVOICE_AVAILABLE = True
    except ImportError:
        COSYVOICE_AVAILABLE = False

# 音频播放（可选）
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False


# ============== 数据结构 ==============

class PlaybackBehavior(Enum):
    """播放行为"""
    QUEUE = "queue"           # 排队播放
    INTERRUPT = "interrupt"   # 中断当前，优先播放
    REPLACE = "replace"       # 清空队列并替换


@dataclass
class SpeechIntent:
    """语音合成意图"""
    intent_id: str
    text: str
    emotion: str = "neutral"
    priority: int = 0
    behavior: PlaybackBehavior = PlaybackBehavior.QUEUE
    created_at: float = field(default_factory=time.time)


@dataclass
class AudioConfig:
    """TTS 音频配置"""
    enabled: bool = True
    tts_provider: str = "cosyvoice"

    # CosyVoice 源码仓库路径（因为 CosyVoice 是源码包，不能直接 pip install）
    cosyvoice_repo_dir: str = ""              # 例: "D:/path/to/CosyVoice"
    cosyvoice_model_dir: str = "./models/CosyVoice2-0.5B"
    ref_audio_dir: str = "./data/audio_refs"
    default_ref_audio: str = "default.wav"
    default_ref_text: str = ""
    sample_rate: int = 22050
    speed: float = 1.0

    # 情绪 -> 参考音频映射 {"joy": {"audio": "happy.wav", "text": "..."}}
    emotion_ref_map: Dict[str, Dict[str, str]] = field(default_factory=dict)

    # 音频缓存
    cache_dir: str = "./data/audio_cache"
    cache_enabled: bool = True

    # 播放
    auto_play: bool = False


# ============== CosyVoice 客户端 ==============

class CosyVoiceClient:
    """
    CosyVoice TTS 客户端。

    封装 CosyVoice 模型加载和推理，
    支持 zero-shot 音色克隆和情绪参考音频切换。
    """

    def __init__(self, config: AudioConfig):
        self.config = config
        self.model: Optional['CosyVoice2'] = None
        self.ref_paths: Dict[str, str] = {}   # emotion -> 音频文件绝对路径
        self.ref_texts: Dict[str, str] = {}
        self._sample_rate: int = config.sample_rate
        self._loaded = False

    def load(self):
        """加载 CosyVoice 模型和参考音频"""
        # 动态注入 repo 路径（比模块级的环境变量优先）
        repo_dir = self.config.cosyvoice_repo_dir
        if repo_dir:
            repo_path = Path(repo_dir)
            for p in [str(repo_path), str(repo_path / "third_party" / "Matcha-TTS")]:
                if p not in sys.path:
                    sys.path.insert(0, p)

        # 路径注入后再尝试 import（模块级 import 可能已失败）
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
                "CosyVoice 未找到。请在 config.json 的 audio.cosyvoice.repo_dir 中\n"
                "填写 CosyVoice 仓库的本地路径，例如:\n"
                '  "repo_dir": "D:/repos/CosyVoice"'
            )
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch 未安装，CosyVoice 需要 torch 和 torchaudio。")

        model_dir = Path(self.config.cosyvoice_model_dir)
        if not model_dir.exists():
            raise FileNotFoundError(
                f"CosyVoice 模型目录不存在: {model_dir}\n"
                f"请下载模型到该目录。"
            )

        logger.info(f"加载 CosyVoice 模型: {model_dir}")
        self.model = CosyVoice2(str(model_dir))
        # 从模型配置读取实际采样率（CosyVoice2 为 24000 Hz）
        self._sample_rate = getattr(self.model, "sample_rate", self.config.sample_rate)
        logger.info(f"CosyVoice 模型加载完成，采样率 {self._sample_rate} Hz")

        # 加载参考音频
        self._load_ref_audios()
        self._loaded = True

    def _load_ref_audios(self):
        """记录参考音频路径（CosyVoice2 直接接受文件路径）"""
        ref_dir = Path(self.config.ref_audio_dir)
        if not ref_dir.exists():
            logger.warning(f"参考音频目录不存在: {ref_dir}，将创建")
            ref_dir.mkdir(parents=True, exist_ok=True)
            return

        # 默认参考音频
        default_path = ref_dir / self.config.default_ref_audio
        if default_path.exists():
            self.ref_paths["neutral"] = str(default_path)
            self.ref_texts["neutral"] = self.config.default_ref_text
            logger.info(f"已注册默认参考音频: {default_path.name}")
        else:
            logger.warning(f"默认参考音频不存在: {default_path}")

        # 情绪参考音频
        for emotion, ref_info in self.config.emotion_ref_map.items():
            audio_file = ref_info.get("audio", "")
            ref_text = ref_info.get("text", "")
            audio_path = ref_dir / audio_file

            if audio_path.exists():
                self.ref_paths[emotion] = str(audio_path)
                self.ref_texts[emotion] = ref_text
                logger.info(f"已注册情绪参考音频: {emotion} -> {audio_file}")
            else:
                logger.warning(f"情绪参考音频不存在: {audio_path}")

        logger.info(f"参考音频注册完成，共 {len(self.ref_paths)} 个")

    def synthesize(self, text: str, emotion: str = "neutral") -> bytes:
        """
        合成语音（同步方法）。

        Args:
            text: 要合成的文本
            emotion: 情绪标签（用于选择参考音频）

        Returns:
            WAV 格式音频字节
        """
        if not self._loaded:
            raise RuntimeError("CosyVoice 模型未加载，请先调用 load()")

        # 选择参考路径（优先情绪对应的，fallback 到 neutral）
        ref_text = self.ref_texts.get(emotion) or self.ref_texts.get("neutral", "")
        ref_path = self.ref_paths.get(emotion) or self.ref_paths.get("neutral")

        if ref_path is None:
            raise RuntimeError(
                "没有可用的参考音频。请在 config.json 的 audio.ref_audio_dir 中放置参考音频。"
            )

        logger.debug(f"TTS 合成: emotion={emotion}, text={text[:50]}...")

        # CosyVoice zero-shot 推理
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
            raise RuntimeError("CosyVoice 合成未返回音频数据")

        # 拼接所有音频块
        audio_array = np.concatenate(audio_chunks, axis=-1)

        # 如果是二维的 (1, N)，取第一个通道
        if audio_array.ndim > 1:
            audio_array = audio_array.squeeze(0)

        # 转换为 WAV 字节
        return self._numpy_to_wav(audio_array, self._sample_rate)

    def _numpy_to_wav(self, audio: 'np.ndarray', sample_rate: int) -> bytes:
        """将 numpy 音频数组转换为 WAV 字节"""
        import numpy as np

        # 归一化到 int16 范围
        if audio.dtype in (np.float32, np.float64):
            audio = np.clip(audio, -1.0, 1.0)
            audio = (audio * 32767).astype(np.int16)

        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sample_rate)
            wf.writeframes(audio.tobytes())

        return buffer.getvalue()

    @property
    def available_emotions(self) -> List[str]:
        """已注册的情绪参考音频列表"""
        return list(self.ref_paths.keys())


# ============== 音频管道 ==============

class AudioPipeline:
    """
    TTS 语音合成管道。

    管理语音合成请求队列，支持优先级调度和播放控制。
    """

    def __init__(self, config: AudioConfig):
        self.config = config
        self.tts_client: Optional[CosyVoiceClient] = None
        self._intent_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self._active_intent: Optional[SpeechIntent] = None
        self._is_playing: bool = False
        self._process_task: Optional[asyncio.Task] = None
        self._stop_event: asyncio.Event = asyncio.Event()

        # 音频缓存
        self._cache_dir = Path(config.cache_dir)
        if config.cache_enabled:
            self._cache_dir.mkdir(parents=True, exist_ok=True)

        # 音频播放初始化
        if config.auto_play and PYGAME_AVAILABLE:
            pygame.mixer.init()

    async def start(self):
        """启动音频管道"""
        if not self.config.enabled:
            logger.info("音频管道已禁用")
            return

        if not COSYVOICE_AVAILABLE:
            logger.warning("CosyVoice 未安装，音频管道不可用")
            return

        # 在线程池中加载模型（避免阻塞事件循环）
        loop = asyncio.get_running_loop()
        self.tts_client = CosyVoiceClient(self.config)
        await loop.run_in_executor(None, self.tts_client.load)

        # 启动处理循环
        self._stop_event.clear()
        self._process_task = asyncio.create_task(self._process_loop())
        logger.info("音频管道已启动")

    async def stop(self):
        """停止音频管道"""
        self._stop_event.set()
        if self._process_task:
            self._process_task.cancel()
            try:
                await self._process_task
            except asyncio.CancelledError:
                pass
        if self.config.auto_play and PYGAME_AVAILABLE:
            pygame.mixer.quit()
        logger.info("音频管道已停止")

    async def speak(
        self,
        text: str,
        emotion: str = "neutral",
        priority: int = 0,
        behavior: PlaybackBehavior = PlaybackBehavior.QUEUE,
    ) -> str:
        """
        发起语音合成请求。

        Args:
            text: 要合成的文本
            emotion: 情绪标签
            priority: 优先级（越大越优先）
            behavior: 播放行为

        Returns:
            intent_id
        """
        intent = SpeechIntent(
            intent_id=uuid.uuid4().hex[:12],
            text=text,
            emotion=emotion,
            priority=priority,
            behavior=behavior,
        )

        if behavior == PlaybackBehavior.REPLACE:
            # 清空队列
            while not self._intent_queue.empty():
                try:
                    self._intent_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

        # 负数实现大顶堆（priority 越大越优先）
        await self._intent_queue.put((-priority, time.time(), intent))
        return intent.intent_id

    async def synthesize(
        self,
        text: str,
        emotion: str = "neutral",
    ) -> bytes:
        """
        仅合成语音（不排队播放），返回 WAV 字节。

        适用于需要自行处理音频的场景（如发送语音消息）。
        """
        if not self.tts_client or not self.tts_client._loaded:
            raise RuntimeError("TTS 客户端未初始化")

        # 检查缓存
        cache_key = self._cache_key(text, emotion)
        cached = self._load_cache(cache_key)
        if cached:
            return cached

        # 在线程池中合成（CosyVoice 推理是 CPU/GPU 密集型）
        loop = asyncio.get_running_loop()
        audio_bytes = await loop.run_in_executor(
            None, self.tts_client.synthesize, text, emotion
        )

        # 写入缓存
        self._save_cache(cache_key, audio_bytes)

        return audio_bytes

    async def _process_loop(self):
        """处理合成队列"""
        while not self._stop_event.is_set():
            try:
                _, _, intent = await asyncio.wait_for(
                    self._intent_queue.get(),
                    timeout=1.0,
                )
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            self._active_intent = intent

            try:
                audio_bytes = await self.synthesize(intent.text, intent.emotion)

                if self.config.auto_play and PYGAME_AVAILABLE:
                    await self._play_audio(audio_bytes, intent)

            except Exception as e:
                logger.error(f"语音合成失败 [{intent.intent_id}]: {e}")
            finally:
                self._active_intent = None

    async def _play_audio(self, audio_bytes: bytes, intent: SpeechIntent):
        """播放音频"""
        import tempfile

        # 写入临时文件
        with tempfile.NamedTemporaryFile(
            suffix=".wav", delete=False, dir=str(self._cache_dir)
        ) as f:
            f.write(audio_bytes)
            temp_path = f.name

        try:
            self._is_playing = True
            pygame.mixer.music.load(temp_path)
            pygame.mixer.music.play()

            # 等待播放完成
            while pygame.mixer.music.get_busy():
                await asyncio.sleep(0.05)
        finally:
            self._is_playing = False
            try:
                os.unlink(temp_path)
            except OSError:
                pass

    # ── 缓存 ──────────────────────────────────────────────────

    def _cache_key(self, text: str, emotion: str) -> str:
        """生成缓存键"""
        raw = f"{text}|{emotion}|{self.config.speed}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def _load_cache(self, key: str) -> Optional[bytes]:
        """从缓存加载音频"""
        if not self.config.cache_enabled:
            return None
        cache_path = self._cache_dir / f"{key}.wav"
        if cache_path.exists():
            return cache_path.read_bytes()
        return None

    def _save_cache(self, key: str, data: bytes):
        """保存音频到缓存"""
        if not self.config.cache_enabled:
            return
        cache_path = self._cache_dir / f"{key}.wav"
        try:
            cache_path.write_bytes(data)
        except OSError as e:
            logger.warning(f"写入音频缓存失败: {e}")

    # ── 状态查询 ──────────────────────────────────────────────

    @property
    def is_speaking(self) -> bool:
        return self._is_playing

    @property
    def queue_size(self) -> int:
        return self._intent_queue.qsize()

    @property
    def available(self) -> bool:
        return (
            self.config.enabled
            and COSYVOICE_AVAILABLE
            and self.tts_client is not None
            and self.tts_client._loaded
        )
