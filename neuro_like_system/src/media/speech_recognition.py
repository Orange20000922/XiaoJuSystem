"""
ASR 语音识别模块

基于 SenseVoice（阿里 FunAudioLLM）的语音识别，
自带语音情感识别（SER）和音频事件检测（AED）。

SenseVoice 特点：
  - 中文识别最强（超过 Whisper）
  - 10 秒音频仅需 70ms 推理
  - 自带情感标签（happy/sad/angry/neutral）
  - 自带音频事件检测（笑声/掌声/哭泣等）
  - 支持 50+ 语言

与 BERT 情绪系统的融合：
  SenseVoice 的语音情感标签可与 BERT 文本情绪标签交叉验证，
  提高情绪判断的准确性。语音情感 -> 文本情绪的映射：
    happy  -> joy / excitement
    sad    -> sadness
    angry  -> anger
    neutral -> neutral

用法：
    from src.media.speech_recognition import SenseVoiceASR, SenseVoiceConfig

    asr = SenseVoiceASR(config)
    asr.load()

    result = asr.transcribe("audio.wav")
    print(result.text)       # "你好呀"
    print(result.emotion)    # "happy"
    print(result.language)   # "zh"
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.logger import logger

# SenseVoice 通过 funasr 包使用，安装命令：
#   pip install funasr>=1.1.3
#   完整依赖: pip install "torch<=2.3" torchaudio modelscope huggingface_hub "funasr>=1.1.3" "numpy<=1.26.4"
try:
    from funasr import AutoModel
    SENSEVOICE_AVAILABLE = True
except ImportError:
    SENSEVOICE_AVAILABLE = False

try:
    import torch
    import torchaudio
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ============== 语音情感 -> 文本情绪映射 ==============

VOICE_EMOTION_TO_TEXT_EMOTION: Dict[str, str] = {
    "happy": "joy",
    "sad": "sadness",
    "angry": "anger",
    "neutral": "neutral",
    # SenseVoice 可能输出的其他标签
    "surprised": "surprise",
    "fearful": "fear",
    "disgusted": "disgust",
}


# ============== 数据结构 ==============

@dataclass
class ASRResult:
    """语音识别结果"""
    text: str                              # 识别文本
    emotion: Optional[str] = None          # 语音情感标签（SenseVoice 原始）
    mapped_emotion: Optional[str] = None   # 映射到文本情绪系统的标签
    language: Optional[str] = None         # 检测到的语言
    audio_events: List[str] = field(default_factory=list)  # 音频事件（笑声等）
    duration_seconds: float = 0.0          # 音频时长
    inference_time_ms: float = 0.0         # 推理耗时


@dataclass
class SenseVoiceConfig:
    """SenseVoice 配置"""
    enabled: bool = True
    model_id: str = "iic/SenseVoiceSmall"   # ModelScope 路径（funasr 默认源）
    # 也可用 HuggingFace 路径: "FunAudioLLM/SenseVoiceSmall"
    model_dir: Optional[str] = None   # 本地模型路径（优先于 model_id）
    device: str = "cuda"              # "cuda" | "cpu"
    language: str = "auto"            # "zh" | "en" | "auto"
    use_emotion: bool = True          # 是否启用情感识别
    use_vad: bool = True              # 是否启用 VAD（语音活动检测）
    batch_size: int = 1


# ============== SenseVoice ASR ==============

class SenseVoiceASR:
    """
    SenseVoice 语音识别。

    集成了 ASR + 语音情感识别 + 音频事件检测。
    """

    def __init__(self, config: SenseVoiceConfig):
        self.config = config
        self.model = None
        self._loaded = False

    def load(self):
        """加载 SenseVoice 模型"""
        if not SENSEVOICE_AVAILABLE:
            raise ImportError(
                "FunASR 未安装。请运行: pip install funasr\n"
                "SenseVoice 模型会在首次使用时自动下载。"
            )

        model_source = self.config.model_dir or self.config.model_id
        logger.info(f"加载 SenseVoice 模型: {model_source}")

        # 确定设备
        device = self.config.device
        if device == "cuda" and TORCH_AVAILABLE and not torch.cuda.is_available():
            logger.warning("CUDA 不可用，回退到 CPU")
            device = "cpu"

        self.model = AutoModel(
            model=model_source,
            trust_remote_code=True,
            device=device,
        )

        self._loaded = True
        logger.info(f"SenseVoice 模型加载完成 (device={device})")

    def transcribe(
        self,
        audio_input: Union[str, Path, bytes],
        language: Optional[str] = None,
    ) -> ASRResult:
        """
        识别语音。

        Args:
            audio_input: 音频文件路径、或音频字节
            language: 语言代码（None 使用配置默认值）

        Returns:
            ASRResult 识别结果
        """
        if not self._loaded:
            raise RuntimeError("SenseVoice 模型未加载，请先调用 load()")

        lang = language or self.config.language
        start_time = time.time()

        # 处理输入
        if isinstance(audio_input, bytes):
            audio_input = self._bytes_to_temp_file(audio_input)
        audio_path = str(audio_input)

        # 推理
        result = self.model.generate(
            input=audio_path,
            cache={},
            language=lang if lang != "auto" else None,
            use_itn=True,  # 逆文本正则化（数字、日期等）
            batch_size_s=self.config.batch_size,
        )

        inference_ms = (time.time() - start_time) * 1000

        # 解析结果
        return self._parse_result(result, inference_ms, audio_path)

    async def transcribe_async(
        self,
        audio_input: Union[str, Path, bytes],
        language: Optional[str] = None,
    ) -> ASRResult:
        """异步版本的 transcribe（在线程池中运行）"""
        import asyncio
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, self.transcribe, audio_input, language
        )

    def _parse_result(
        self,
        raw_result: list,
        inference_ms: float,
        audio_path: str,
    ) -> ASRResult:
        """解析 SenseVoice 原始输出"""
        if not raw_result:
            return ASRResult(text="", inference_time_ms=inference_ms)

        # SenseVoice 输出格式: [{"key": ..., "text": "<|zh|><|HAPPY|><|Speech|>你好呀"}]
        first = raw_result[0] if isinstance(raw_result, list) else raw_result
        raw_text = first.get("text", "") if isinstance(first, dict) else str(first)

        # 解析标签
        text, emotion, language, audio_events = self._parse_tagged_text(raw_text)

        # 映射语音情感到文本情绪
        mapped_emotion = None
        if emotion:
            emotion_lower = emotion.lower()
            mapped_emotion = VOICE_EMOTION_TO_TEXT_EMOTION.get(
                emotion_lower, emotion_lower
            )

        # 计算音频时长
        duration = self._get_audio_duration(audio_path)

        return ASRResult(
            text=text.strip(),
            emotion=emotion,
            mapped_emotion=mapped_emotion,
            language=language,
            audio_events=audio_events,
            duration_seconds=duration,
            inference_time_ms=inference_ms,
        )

    def _parse_tagged_text(self, raw_text: str):
        """
        解析 SenseVoice 的标签文本。

        SenseVoice 输出格式:
            <|zh|><|HAPPY|><|Speech|>你好呀
            <|en|><|NEUTRAL|><|BGM|><|Speech|>Hello
        """
        import re

        # 提取所有标签
        tag_pattern = re.compile(r'<\|([^|]+)\|>')
        tags = tag_pattern.findall(raw_text)

        # 移除标签，得到纯文本
        text = tag_pattern.sub('', raw_text)

        # 分类标签
        emotion = None
        language = None
        audio_events = []

        # 已知情感标签
        emotion_tags = {"HAPPY", "SAD", "ANGRY", "NEUTRAL", "SURPRISED", "FEARFUL", "DISGUSTED"}
        # 已知语言标签
        language_tags = {"zh", "en", "ja", "ko", "yue", "auto"}
        # 已知音频事件标签
        event_tags = {"Speech", "BGM", "Laughter", "Applause", "Cry", "Cough", "Music"}

        for tag in tags:
            tag_upper = tag.upper()
            tag_lower = tag.lower()

            if tag_upper in emotion_tags:
                emotion = tag_lower
            elif tag_lower in language_tags:
                language = tag_lower
            elif tag in event_tags and tag != "Speech":
                audio_events.append(tag.lower())

        return text, emotion, language, audio_events

    def _get_audio_duration(self, audio_path: str) -> float:
        """获取音频时长（秒）"""
        if not TORCH_AVAILABLE:
            return 0.0
        try:
            info = torchaudio.info(audio_path)
            return info.num_frames / info.sample_rate
        except Exception:
            return 0.0

    def _bytes_to_temp_file(self, audio_bytes: bytes) -> str:
        """将音频字节写入临时文件"""
        import tempfile
        with tempfile.NamedTemporaryFile(
            suffix=".wav", delete=False
        ) as f:
            f.write(audio_bytes)
            return f.name

    @property
    def available(self) -> bool:
        return self.config.enabled and SENSEVOICE_AVAILABLE and self._loaded


# ============== 情绪融合工具 ==============

def fuse_voice_and_text_emotion(
    voice_emotion: Optional[str],
    text_emotion: str,
    text_confidence: float,
    voice_weight: float = 0.3,
    text_weight: float = 0.7,
) -> str:
    """
    融合语音情感和文本情绪。

    当两个信号一致时，增强置信度。
    当两个信号冲突时，以文本情绪为主（因为 BERT 在文本情绪上更准确），
    但降低置信度。

    Args:
        voice_emotion: SenseVoice 语音情感（已映射到文本情绪标签）
        text_emotion: BERT 文本情绪标签
        text_confidence: BERT 文本情绪置信度
        voice_weight: 语音情感权重
        text_weight: 文本情绪权重

    Returns:
        最终情绪标签
    """
    if voice_emotion is None:
        return text_emotion

    # 信号一致 -> 直接返回
    if voice_emotion == text_emotion:
        return text_emotion

    # 信号冲突 -> 高置信度时以文本为主，低置信度时参考语音
    if text_confidence >= 0.7:
        return text_emotion

    # 低置信度且语音有明确情绪 -> 使用语音
    if voice_emotion != "neutral":
        return voice_emotion

    return text_emotion
