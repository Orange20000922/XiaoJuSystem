"""
模型配置文件
定义情绪标签、行为标签、人格特征、API配置等
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum


# ============== 情绪标签定义 ==============
class EmotionType(Enum):
    """基础情绪类型"""
    JOY = "joy"                    # 喜悦
    SADNESS = "sadness"            # 悲伤
    ANGER = "anger"                # 愤怒
    FEAR = "fear"                  # 恐惧
    SURPRISE = "surprise"          # 惊讶
    DISGUST = "disgust"            # 厌恶
    NEUTRAL = "neutral"            # 中性
    EXCITEMENT = "excitement"      # 兴奋
    TENDERNESS = "tenderness"      # 温柔
    CURIOSITY = "curiosity"        # 好奇


# 情绪ID映射
EMOTION_TO_ID = {e.value: i for i, e in enumerate(EmotionType)}
ID_TO_EMOTION = {i: e.value for i, e in enumerate(EmotionType)}
NUM_EMOTIONS = len(EmotionType)


# ============== 行为标签定义 ==============
class BehaviorType(Enum):
    """行为类型"""
    RESPOND_POSITIVE = "respond_positive"      # 积极回应
    RESPOND_NEGATIVE = "respond_negative"      # 消极回应
    ASK_QUESTION = "ask_question"              # 提问
    SHARE_EXPERIENCE = "share_experience"      # 分享经历
    GIVE_ADVICE = "give_advice"                # 给建议
    EXPRESS_EMPATHY = "express_empathy"        # 表达共情
    MAKE_JOKE = "make_joke"                    # 开玩笑
    CHANGE_TOPIC = "change_topic"              # 转移话题
    SEEK_CLARIFICATION = "seek_clarification"  # 寻求澄清
    AGREE = "agree"                            # 同意
    DISAGREE = "disagree"                      # 不同意
    NEUTRAL_ACKNOWLEDGE = "neutral_acknowledge" # 中性确认


BEHAVIOR_TO_ID = {b.value: i for i, b in enumerate(BehaviorType)}
ID_TO_BEHAVIOR = {i: b.value for i, b in enumerate(BehaviorType)}
NUM_BEHAVIORS = len(BehaviorType)


# ============== 语气标签定义 ==============
class ToneType(Enum):
    """语气类型"""
    ENTHUSIASTIC = "enthusiastic"  # 热情
    CALM = "calm"                  # 平静
    PLAYFUL = "playful"            # 俏皮
    SERIOUS = "serious"            # 严肃
    WARM = "warm"                  # 温暖
    COLD = "cold"                  # 冷淡
    SARCASTIC = "sarcastic"        # 讽刺
    SUPPORTIVE = "supportive"      # 支持


TONE_TO_ID = {t.value: i for i, t in enumerate(ToneType)}
ID_TO_TONE = {i: t.value for i, t in enumerate(ToneType)}
NUM_TONES = len(ToneType)


# ============== 人格配置 ==============
@dataclass
class PersonalityConfig:
    """人格配置"""
    name: str = "Neuro"

    # 基础特质 (Big Five)
    openness: float = 0.8           # 开放性
    conscientiousness: float = 0.5  # 尽责性
    extraversion: float = 0.7       # 外向性
    agreeableness: float = 0.8      # 宜人性
    neuroticism: float = 0.3        # 神经质

    # 情绪基线
    emotional_baseline: str = "positive"
    emotional_volatility: float = 0.4  # 情绪波动性

    # 行为倾向
    humor_tendency: float = 0.7     # 幽默倾向
    empathy_level: float = 0.8      # 共情能力
    curiosity_level: float = 0.9    # 好奇心

    # 语言风格
    formality: float = 0.3          # 正式程度 (0=casual, 1=formal)
    verbosity: float = 0.5          # 话多程度

    # 特殊标签
    traits: List[str] = field(default_factory=lambda: ["活泼", "好奇", "善良"])

    # 自由文本人格描述（直接注入 prompt，优先级高于数值字段）
    # 填写后将替代数值参数作为人格描述，支持详细的行为规范、语言习惯、负向约束等
    description: str = ""

    def to_embedding_vector(self) -> List[float]:
        """转换为嵌入向量"""
        return [
            self.openness,
            self.conscientiousness,
            self.extraversion,
            self.agreeableness,
            self.neuroticism,
            self.emotional_volatility,
            self.humor_tendency,
            self.empathy_level,
            self.curiosity_level,
            self.formality,
            self.verbosity
        ]

    @property
    def embedding_dim(self) -> int:
        return len(self.to_embedding_vector())


# ============== 模型配置 ==============
@dataclass
class EmotionModelConfig:
    """情绪识别模型配置"""
    # 基座模型
    pretrained_model: str = "hfl/chinese-roberta-wwm-ext"

    # 模型结构
    hidden_size: int = 768
    num_emotions: int = NUM_EMOTIONS
    dropout: float = 0.1

    # 人格融合
    personality_dim: int = 11  # PersonalityConfig.embedding_dim
    use_personality: bool = True

    # 训练参数
    max_length: int = 128
    batch_size: int = 32
    learning_rate: float = 2e-5
    num_epochs: int = 10
    warmup_ratio: float = 0.1


@dataclass
class BehaviorModelConfig:
    """行为生成模型配置"""
    # 基座模型 (Encoder-Decoder)
    pretrained_model: str = "uer/t5-base-chinese-cluecorpussmall"

    # 模型结构
    hidden_size: int = 768
    num_behaviors: int = NUM_BEHAVIORS
    num_tones: int = NUM_TONES
    dropout: float = 0.1

    # 输入维度
    emotion_dim: int = NUM_EMOTIONS
    personality_dim: int = 11

    # 训练参数
    max_input_length: int = 256
    max_output_length: int = 64
    batch_size: int = 16
    learning_rate: float = 3e-5
    num_epochs: int = 15
    warmup_ratio: float = 0.1


@dataclass
class JointModelConfig:
    """联合模型配置 (情绪+行为一体)"""
    # 基座模型
    pretrained_model: str = "hfl/chinese-roberta-wwm-ext"

    # 模型结构
    hidden_size: int = 768
    intermediate_size: int = 512
    num_emotions: int = NUM_EMOTIONS
    num_behaviors: int = NUM_BEHAVIORS
    num_tones: int = NUM_TONES
    dropout: float = 0.1

    # 人格融合
    personality_dim: int = 11
    use_personality: bool = True

    # 训练参数
    max_length: int = 128
    batch_size: int = 16              # 8GB显存建议16
    learning_rate: float = 2e-5
    num_epochs: int = 10
    warmup_ratio: float = 0.1
    gradient_accumulation_steps: int = 2  # 梯度累积，等效batch=32
    fp16: bool = True                 # 混合精度训练，省显存

    # 多任务权重
    emotion_loss_weight: float = 1.0
    behavior_loss_weight: float = 1.0
    tone_loss_weight: float = 0.5
    intensity_loss_weight: float = 0.5


# ============== 8GB显存优化配置 ==============
@dataclass
class JointModelConfigLowVRAM(JointModelConfig):
    """8GB显存优化配置"""
    batch_size: int = 8
    gradient_accumulation_steps: int = 4  # 等效batch=32
    max_length: int = 96              # 稍微缩短，省显存
    fp16: bool = True


# ============== 默认配置实例 ==============
DEFAULT_PERSONALITY = PersonalityConfig()
DEFAULT_EMOTION_CONFIG = EmotionModelConfig()
DEFAULT_BEHAVIOR_CONFIG = BehaviorModelConfig()
DEFAULT_JOINT_CONFIG = JointModelConfig()
DEFAULT_JOINT_CONFIG_LOW_VRAM = JointModelConfigLowVRAM()


# ============== LLM API 配置 ==============
class LLMProvider(Enum):
    """支持的LLM提供商"""
    OPENAI = "openai"
    DEEPSEEK = "deepseek"
    ANTHROPIC = "anthropic"
    CUSTOM = "custom"  # 自定义API（兼容OpenAI格式）


@dataclass
class LLMConfig:
    """大模型API配置"""
    # 提供商
    provider: LLMProvider = LLMProvider.OPENAI

    # API密钥（优先从环境变量读取）
    api_key: Optional[str] = None

    # API Base URL
    base_url: Optional[str] = None

    # 模型名称
    model: str = "gpt-5.2-instant"

    # 生成参数
    temperature: float = 0.8
    max_tokens: int = 200
    top_p: float = 0.95
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0

    # 超时设置（秒）
    timeout: int = 30

    # 重试设置
    max_retries: int = 3
    retry_delay: float = 1.0        # 首次重试等待秒数
    retry_backoff: float = 2.0      # 指数退避乘数
    retry_max_delay: float = 60.0   # 单次等待上限（秒）

    # 使用 OpenAI Responses API（/v1/responses）而非 Chat Completions API
    use_responses_api: bool = False

    def __post_init__(self):
        """初始化后处理：从环境变量读取密钥"""
        if self.api_key is None:
            env_key_map = {
                LLMProvider.OPENAI: "OPENAI_API_KEY",
                LLMProvider.DEEPSEEK: "DEEPSEEK_API_KEY",
                LLMProvider.ANTHROPIC: "ANTHROPIC_API_KEY",
                LLMProvider.CUSTOM: "CUSTOM_API_KEY",
            }
            env_key = env_key_map.get(self.provider, "OPENAI_API_KEY")
            self.api_key = os.environ.get(env_key)

        # 设置默认base_url
        if self.base_url is None:
            self.base_url = self._get_default_base_url()

    def _get_default_base_url(self) -> Optional[str]:
        """获取默认的API Base URL"""
        url_map = {
            LLMProvider.OPENAI: "https://api.openai.com/v1",
            LLMProvider.DEEPSEEK: "https://api.deepseek.com/v1",
            LLMProvider.ANTHROPIC: None,  # Anthropic SDK自带
            LLMProvider.CUSTOM: None,
        }
        return url_map.get(self.provider)

    @classmethod
    def from_env(cls, provider: str = "openai") -> "LLMConfig":
        """从环境变量创建配置"""
        provider_enum = LLMProvider(provider.lower())
        return cls(provider=provider_enum)

    @classmethod
    def openai(
        cls,
        api_key: Optional[str] = None,
        model: str = "gpt-5.2-instant",
        base_url: Optional[str] = None
    ) -> "LLMConfig":
        """创建OpenAI配置"""
        return cls(
            provider=LLMProvider.OPENAI,
            api_key=api_key,
            model=model,
            base_url=base_url or "https://api.openai.com/v1"
        )

    @classmethod
    def deepseek(
        cls,
        api_key: Optional[str] = None,
        model: str = "deepseek-chat"
    ) -> "LLMConfig":
        """创建DeepSeek配置"""
        return cls(
            provider=LLMProvider.DEEPSEEK,
            api_key=api_key,
            model=model,
            base_url="https://api.deepseek.com/v1"
        )

    @classmethod
    def anthropic(
        cls,
        api_key: Optional[str] = None,
        model: str = "claude-3-5-haiku-20241022"
    ) -> "LLMConfig":
        """创建Anthropic配置"""
        return cls(
            provider=LLMProvider.ANTHROPIC,
            api_key=api_key,
            model=model,
            base_url=None
        )

    @classmethod
    def custom(
        cls,
        base_url: str,
        api_key: Optional[str] = None,
        model: str = "gpt-5.2-instant"
    ) -> "LLMConfig":
        """创建自定义API配置（兼容OpenAI格式）"""
        return cls(
            provider=LLMProvider.CUSTOM,
            api_key=api_key,
            model=model,
            base_url=base_url
        )


# ============== 预设LLM配置 ==============
# GPT-5 系列
DEFAULT_LLM_CONFIG = LLMConfig.openai(model="gpt-5")

# GPT-5.2 系列(默认)
GPT5_INSTANT_CONFIG = LLMConfig.openai(model="gpt-5.2-instant")
GPT5_THINKING_CONFIG = LLMConfig.openai(model="gpt-5.2-thinking")

# Claude 4.5 系列
CLAUDE_OPUS_CONFIG = LLMConfig.anthropic(model="claude-opus-4-5-20251101")
CLAUDE_SONNET_CONFIG = LLMConfig.anthropic(model="claude-sonnet-4-5-20241022")
CLAUDE_HAIKU_CONFIG = LLMConfig.anthropic(model="claude-3-5-haiku-20241022")

# DeepSeek
DEEPSEEK_CONFIG = LLMConfig.deepseek()


# ============== 数据标注API配置 ==============
@dataclass
class AnnotationAPIConfig:
    """数据标注API配置"""
    # 主要标注模型
    primary_provider: LLMProvider = LLMProvider.OPENAI
    primary_model: str = "gpt-5.2-instant"
    primary_api_key: Optional[str] = None
    primary_base_url: Optional[str] = None

    # 备用标注模型（用于对比或降级）
    fallback_provider: LLMProvider = LLMProvider.DEEPSEEK
    fallback_model: str = "deepseek-chat"
    fallback_api_key: Optional[str] = None
    fallback_base_url: Optional[str] = None

    # 标注参数
    batch_size: int = 10
    temperature: float = 0.3  # 标注用低温度，提高一致性
    max_retries: int = 3

    def __post_init__(self):
        """从环境变量读取密钥"""
        if self.primary_api_key is None:
            self.primary_api_key = os.environ.get("OPENAI_API_KEY")
        if self.fallback_api_key is None:
            self.fallback_api_key = os.environ.get("DEEPSEEK_API_KEY")
        if self.primary_base_url is None:
            self.primary_base_url = "https://api.openai.com/v1"
        if self.fallback_base_url is None:
            self.fallback_base_url = "https://api.deepseek.com/v1"


DEFAULT_ANNOTATION_CONFIG = AnnotationAPIConfig()


# ============== 分级记忆系统配置 ==============
@dataclass
class MemoryConfig:
    """Token 计数导向的分级记忆系统配置"""

    # Mem0 向量存储
    vector_store_path: str = "./data/qdrant_db"
    collection_name: str = "neuro_memory"

    # Mem0 内部 LLM（压缩摘要/事实抽取用，复用对话 LLM 的 key/endpoint）
    mem0_llm_model: str = "gpt-5.2-instant"
    mem0_llm_temperature: float = 0.1
    mem0_api_key: Optional[str] = None
    mem0_base_url: Optional[str] = None     # 第三方供应商 endpoint

    # LLM 上下文窗口大小（按实际使用的模型填写）
    context_window_tokens: int = 128_000    # GPT-5.2 Instant: 128K

    # L1 压缩触发阈值（占上下文窗口比例），达到后将最旧部分压缩到 L2
    compression_threshold: float = 0.75

    # 每次压缩 L1 最旧的比例
    compression_ratio: float = 0.5

    # Mem0 检索参数
    l3_search_limit: int = 5
    l4_search_limit: int = 3
    relevance_threshold: float = 0.7

    @property
    def compression_trigger_tokens(self) -> int:
        return int(self.context_window_tokens * self.compression_threshold)

    def __post_init__(self):
        if self.mem0_api_key is None:
            self.mem0_api_key = os.environ.get("OPENAI_API_KEY")


DEFAULT_MEMORY_CONFIG = MemoryConfig()

