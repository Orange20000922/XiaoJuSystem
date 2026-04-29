"""
模型配置文件。

定义情绪标签、行为标签、人格特征，以及各类运行时配置。
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum


# ============== 情绪标签定义 ==============
class EmotionType(Enum):
    """基础情绪类型。"""
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


# 情绪 ID 映射
EMOTION_TO_ID = {e.value: i for i, e in enumerate(EmotionType)}
ID_TO_EMOTION = {i: e.value for i, e in enumerate(EmotionType)}
NUM_EMOTIONS = len(EmotionType)


# ============== 行为标签定义 ==============
class BehaviorType(Enum):
    """行为类型。"""
    RESPOND_POSITIVE = "respond_positive"       # 积极回应
    RESPOND_NEGATIVE = "respond_negative"       # 消极回应
    ASK_QUESTION = "ask_question"               # 提问
    SHARE_EXPERIENCE = "share_experience"       # 分享经历
    GIVE_ADVICE = "give_advice"                 # 给出建议
    EXPRESS_EMPATHY = "express_empathy"         # 表达共情
    MAKE_JOKE = "make_joke"                     # 开玩笑
    CHANGE_TOPIC = "change_topic"               # 转移话题
    SEEK_CLARIFICATION = "seek_clarification"   # 寻求澄清
    AGREE = "agree"                             # 同意
    DISAGREE = "disagree"                       # 不同意
    NEUTRAL_ACKNOWLEDGE = "neutral_acknowledge" # 中性确认


BEHAVIOR_TO_ID = {b.value: i for i, b in enumerate(BehaviorType)}
ID_TO_BEHAVIOR = {i: b.value for i, b in enumerate(BehaviorType)}
NUM_BEHAVIORS = len(BehaviorType)


# ============== 语气标签定义 ==============
class ToneType(Enum):
    """语气类型。"""
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
    """人格配置。"""
    name: str = "Neuro"

    # Big Five 基础特质
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
    formality: float = 0.3          # 正式程度（0=casual, 1=formal）
    verbosity: float = 0.5          # 话多程度

    # 标签式人格描述
    traits: List[str] = field(default_factory=lambda: ["活泼", "好奇", "善良"])

    # 主人格描述。填写后可直接注入 prompt，
    # 优先级高于数值型人格字段，适合承载更细的行为规范和说话习惯。
    description: str = ""

    def to_embedding_vector(self) -> List[float]:
        """转换为人格嵌入向量。"""
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


# ============== BERT 输出到 Prompt 的映射配置 ==============
@dataclass
class EmotionPromptConfig:
    """BERT 输出到 Prompt 提示的映射配置。"""
    emotion_map: Dict[str, str] = field(default_factory=dict)
    intensity_levels: Dict[str, float] = field(
        default_factory=lambda: {"low_max": 0.4, "high_min": 0.7}
    )
    # 各情绪标签的可靠度，可用验证集 F1 估计。
    emotion_reliability: Dict[str, float] = field(default_factory=dict)
    # effective_confidence = BERT_prob * reliability
    # > strong: 注入确定性提示
    # > weak:   注入不确定提示
    # 其余情况跳过，交给 LLM 自行判断。
    confidence_thresholds: Dict[str, float] = field(
        default_factory=lambda: {"strong": 0.2, "weak": 0.15}
    )


# ============== 小模型运行时配置 ==============
@dataclass
class SmallModelConfig:
    """小模型推理后端配置。"""

    backend: str = "pytorch"              # "pytorch" | "onnx_grpc"
    checkpoint_path: str = "./checkpoints/joint_model/best.pt"
    device: str = "auto"                  # PyTorch 后端使用；ONNX gRPC 后端固定由 Python 侧发起请求
    tokenizer_path: Optional[str] = None  # ONNX gRPC 后端 tokenizer 路径，为空时自动推断
    onnx_target: str = "127.0.0.1:50051"  # C++ ONNX gRPC 服务地址
    grpc_timeout_seconds: float = 30.0
    request_timeout_seconds: float = 35.0
    max_length: int = 128
    batching_enabled: bool = True
    batch_size: int = 8
    batch_wait_ms: float = 4.0
    max_queue_size: int = 256


# ============== 模型配置 ==============
@dataclass
class EmotionModelConfig:
    """情绪识别模型配置。"""
    # 基座模型
    pretrained_model: str = "hfl/chinese-roberta-wwm-ext"

    # 模型结构
    hidden_size: int = 768
    num_emotions: int = NUM_EMOTIONS
    dropout: float = 0.1

    # 人格特征融合
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
    """行为生成模型配置。"""
    # 基座模型（Encoder-Decoder）
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
    """联合模型配置（情绪 + 行为）。"""
    # 基座模型
    pretrained_model: str = "hfl/chinese-roberta-wwm-ext"

    # 模型结构
    hidden_size: int = 768
    intermediate_size: int = 512
    num_emotions: int = NUM_EMOTIONS
    num_behaviors: int = NUM_BEHAVIORS
    num_tones: int = NUM_TONES
    dropout: float = 0.1

    # 人格特征融合
    personality_dim: int = 11
    use_personality: bool = True

    # 训练参数
    max_length: int = 128
    batch_size: int = 16              # 8GB 显存建议 16
    learning_rate: float = 2e-5
    num_epochs: int = 10
    warmup_ratio: float = 0.1
    gradient_accumulation_steps: int = 2  # 梯度累积，等效 batch=32
    fp16: bool = True                 # 混合精度训练，节省显存

    # 多任务损失权重
    emotion_loss_weight: float = 1.0
    behavior_loss_weight: float = 1.0
    tone_loss_weight: float = 0.5
    intensity_loss_weight: float = 0.5


# ============== 8GB 显存优化配置 ==============
@dataclass
class JointModelConfigLowVRAM(JointModelConfig):
    """8GB 显存优化配置。"""
    batch_size: int = 8
    gradient_accumulation_steps: int = 4  # 等效 batch=32
    max_length: int = 96              # 适度缩短序列，节省显存
    fp16: bool = True


# ============== 默认配置实例 ==============
DEFAULT_PERSONALITY = PersonalityConfig()
DEFAULT_SMALL_MODEL_CONFIG = SmallModelConfig()
DEFAULT_EMOTION_CONFIG = EmotionModelConfig()
DEFAULT_BEHAVIOR_CONFIG = BehaviorModelConfig()
DEFAULT_JOINT_CONFIG = JointModelConfig()
DEFAULT_JOINT_CONFIG_LOW_VRAM = JointModelConfigLowVRAM()


# ============== LLM API 配置 ==============
class LLMProvider(Enum):
    """支持的 LLM 提供商。"""
    OPENAI = "openai"
    DEEPSEEK = "deepseek"
    ANTHROPIC = "anthropic"
    CUSTOM = "custom"  # 自定义 API（兼容 OpenAI 格式）


@dataclass
class LLMConfig:
    """大模型 API 配置。"""
    # 提供商
    provider: LLMProvider = LLMProvider.OPENAI

    # API 密钥，优先从环境变量读取
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

    # 是否使用 OpenAI Responses API（/v1/responses）
    use_responses_api: bool = False

    def __post_init__(self):
        """初始化后处理：从环境变量补全密钥。"""
        if self.api_key is None:
            env_key_map = {
                LLMProvider.OPENAI: "OPENAI_API_KEY",
                LLMProvider.DEEPSEEK: "DEEPSEEK_API_KEY",
                LLMProvider.ANTHROPIC: "ANTHROPIC_API_KEY",
                LLMProvider.CUSTOM: "CUSTOM_API_KEY",
            }
            env_key = env_key_map.get(self.provider, "OPENAI_API_KEY")
            self.api_key = os.environ.get(env_key)

        # 补全默认 base_url
        if self.base_url is None:
            self.base_url = self._get_default_base_url()

    def _get_default_base_url(self) -> Optional[str]:
        """获取默认 API Base URL。"""
        url_map = {
            LLMProvider.OPENAI: "https://api.openai.com/v1",
            LLMProvider.DEEPSEEK: "https://api.deepseek.com/v1",
            LLMProvider.ANTHROPIC: None,  # Anthropic SDK 自带
            LLMProvider.CUSTOM: None,
        }
        return url_map.get(self.provider)

    @classmethod
    def from_env(cls, provider: str = "openai") -> "LLMConfig":
        """从环境变量创建配置。"""
        provider_enum = LLMProvider(provider.lower())
        return cls(provider=provider_enum)

    @classmethod
    def openai(
        cls,
        api_key: Optional[str] = None,
        model: str = "gpt-5.2-instant",
        base_url: Optional[str] = None
    ) -> "LLMConfig":
        """创建 OpenAI 配置。"""
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
        """创建 DeepSeek 配置。"""
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
        """创建 Anthropic 配置。"""
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
        """创建自定义 API 配置（兼容 OpenAI 格式）。"""
        return cls(
            provider=LLMProvider.CUSTOM,
            api_key=api_key,
            model=model,
            base_url=base_url
        )


# ============== 注意力系统配置 ==============
@dataclass
class AttentionConfig:
    """注意力系统配置。"""
    intensity_threshold: float = 0.7      # 情绪强度阈值，超过时更倾向回复
    cooldown_seconds: int = 60            # 回复冷却时间，避免刷屏
    track_mentioned_users: bool = True    # 是否追踪被 @ 过的用户
    mentioned_user_ttl: int = 300         # 被 @ 用户的注意力保持时间（秒）
    context_window_messages: int = 10     # 最近 N 条消息内的用户视为仍在上下文中

    # 非焦点回复控制（群聊中未被 @ 且不在焦点内的用户）
    non_focus_reply_interval: int = 180   # 非焦点回复最小间隔（秒）
    non_focus_max_token_ratio: float = 0.5  # 非焦点回复的 max_tokens 比例


# ============== Agent 事件循环配置 ==============
@dataclass
class AgentConfig:
    """Agent 事件循环配置。"""
    proactive_level: str = "off"          # "off" | "low" | "medium"
    idle_threshold_seconds: int = 300
    proactive_interval_seconds: int = 30
    time_awareness: bool = True
    tick_interval: float = 2.0
    max_concurrent_chats: int = 3         # 每个 persona 最大并发聊天数


# ============== 主动决策模块配置 ==============
@dataclass
class ProactiveConfig:
    """主动决策模块配置。"""
    enabled: bool = True

    # 决策层 LLM
    decision_provider: str = "deepseek"
    decision_model: str = "deepseek-chat"
    decision_temperature: float = 0.3
    decision_timeout: float = 5.0

    # 决策阈值
    confidence_threshold: float = 0.6  # should_respond 的置信度阈值

    # 上下文窗口
    recent_turns_limit: int = 8        # 从 L1 取最近 N 轮
    l4_memory_limit: int = 3           # 从 L4 取 N 条情绪相关记忆

    # 时间驱动触发
    idle_trigger_hours: float = 2.0    # 空闲 N 小时后触发主动决策
    response_wait_minutes: int = 30    # 主动发言后等待用户回应的时间

    # 冷却时间（秒）
    min_interval_seconds: int = 30


# ============== 预设 LLM 配置 ==============
# GPT-5 系列
DEFAULT_LLM_CONFIG = LLMConfig.openai(model="gpt-5")

# GPT-5.2 系列（默认）
GPT5_INSTANT_CONFIG = LLMConfig.openai(model="gpt-5.2-instant")
GPT5_THINKING_CONFIG = LLMConfig.openai(model="gpt-5.2-thinking")

# Claude 4.5 系列
CLAUDE_OPUS_CONFIG = LLMConfig.anthropic(model="claude-opus-4-5-20251101")
CLAUDE_SONNET_CONFIG = LLMConfig.anthropic(model="claude-sonnet-4-5-20241022")
CLAUDE_HAIKU_CONFIG = LLMConfig.anthropic(model="claude-3-5-haiku-20241022")

# DeepSeek
DEEPSEEK_CONFIG = LLMConfig.deepseek()


# ============== 数据标注 API 配置 ==============
@dataclass
class AnnotationAPIConfig:
    """数据标注 API 配置。"""
    # 主标注模型
    primary_provider: LLMProvider = LLMProvider.OPENAI
    primary_model: str = "gpt-5.2-instant"
    primary_api_key: Optional[str] = None
    primary_base_url: Optional[str] = None

    # 备用标注模型
    fallback_provider: LLMProvider = LLMProvider.DEEPSEEK
    fallback_model: str = "deepseek-chat"
    fallback_api_key: Optional[str] = None
    fallback_base_url: Optional[str] = None

    # 标注参数
    batch_size: int = 10
    temperature: float = 0.3  # 标注任务使用低温度，提高一致性
    max_retries: int = 3

    def __post_init__(self):
        """从环境变量读取密钥。"""
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
    """基于 token 计数的分级记忆系统配置。"""

    # 用户身份标识，用于 Mem0 user_id
    user_id: str = "owner"

    # Mem0 向量存储
    vector_store_path: str = "./data/qdrant_db"
    collection_name: str = "neuro_memory"

    # Mem0 内部使用的 LLM，用于压缩摘要和事实抽取
    mem0_llm_provider: str = "openai"   # "anthropic" | "openai"
    mem0_llm_model: str = "gpt-5.2-instant"
    mem0_llm_temperature: float = 0.1
    mem0_api_key: Optional[str] = None
    mem0_base_url: Optional[str] = None     # 第三方供应商 endpoint

    # LLM 上下文窗口大小，按实际使用模型填写
    context_window_tokens: int = 128_000    # GPT-5.2 Instant: 128K

    # L1 压缩触发阈值，占上下文窗口比例
    compression_threshold: float = 0.75

    # 每次压缩 L1 最旧部分的比例
    compression_ratio: float = 0.5

    # Mem0 检索参数
    l3_search_limit: int = 5
    l4_search_limit: int = 3
    # 缺少生命周期元数据的旧记忆使用该值作为兼容兜底。
    relevance_threshold: float = 0.50
    memory_recall_overfetch: int = 3
    l2_relevance_threshold: float = 0.40
    l2_history_threshold: float = 0.30
    l3_recent_dialog_threshold: float = 0.35
    l3_recent_dialog_history_threshold: float = 0.20
    l3_fact_threshold: float = 0.50
    l3_visual_threshold: float = 0.50
    l3_default_threshold: float = 0.50
    l4_relevance_threshold: float = 0.70

    # 遗忘机制 MVP：元数据 / 逻辑遗忘 / 延迟删除配置
    lifecycle_store_path: Optional[str] = None
    enable_forgetting: bool = False
    forgetting_interval_hours: int = 24
    memory_count_trigger: int = 5000
    min_retention_days: float = 1.0
    physical_deletion_delay_days: float = 7.0

    lambda_l2: float = 0.30
    lambda_l3: float = 0.10
    base_weight_l2: float = 0.70
    base_weight_l3: float = 1.00
    alpha_recall: float = 0.20

    mood_coeff_v: float = 0.10
    mood_coeff_a: float = 0.10
    encoding_intensity_coeff: float = 0.30
    encoding_arousal_coeff: float = 0.20

    configured_W_ref: float = 1.50
    max_prune_prob: float = 0.80
    depth_bias: float = 0.05
    random_jitter_sigma: float = 0.10
    random_seed: Optional[int] = None

    recovery_enabled: bool = True
    recovery_threshold: float = 0.70
    recovery_recent_dialog_threshold: Optional[float] = None
    recovery_rate_limit_per_day: int = 10
    hard_delete_enabled: bool = False

    @property
    def compression_trigger_tokens(self) -> int:
        return int(self.context_window_tokens * self.compression_threshold)

    def __post_init__(self):
        if self.mem0_api_key is None:
            self.mem0_api_key = os.environ.get("OPENAI_API_KEY")


DEFAULT_MEMORY_CONFIG = MemoryConfig()


# ============== 情绪融合配置 ==============
@dataclass
class EmotionFusionConfig:
    """BERT + LLM 双信号情绪融合配置。"""
    enabled: bool = True
    use_by_default: bool = True  # 全局默认是否启用融合
    w_bert: float = 0.6
    w_llm: float = 0.4
    bias: float = 0.0
    skip_llm_threshold: float = 0.85
    llm_timeout: float = 5.0
    llm_temperature: float = 0.3


DEFAULT_EMOTION_FUSION_CONFIG = EmotionFusionConfig()


# ============== 情绪状态机配置 ==============
@dataclass
class EmotionStateConfig:
    """情绪状态机配置（二位耦合 OU 过程 + tanh 非线性）。"""
    alpha: float = 0.75             # 状态惯性
    beta: float = 0.25              # AI 输出影响权重
    gamma: float = 0.25             # 用户情绪影响权重（EKF-MLE 估计）
    delta: float = 0.15             # 均值回归强度
    baseline_valence: float = 0.15  # 人格基线 valence
    baseline_arousal: float = 0.28  # 人格基线 arousal
    kappa: float = 0.05             # V-A 耦合系数
    negativity_bias: float = 1.3    # 负面情绪衰减减速因子
    noise_sigma: float = 0.05       # 过程噪声
    injection_threshold: float = 0.12  # 超过该阈值才注入 prompt
    save_interval_turns: int = 5    # 每 N 轮保存一次到 L4
    persist_to_l4: bool = True


DEFAULT_EMOTION_STATE_CONFIG = EmotionStateConfig()


# ============== 图片识别配置 ==============
@dataclass
class ImageConfig:
    """图片识别配置。"""
    enabled: bool = True
    cache_dir: str = "./data/image_cache"
    max_download_size_bytes: int = 10_485_760  # 10MB
    max_dimension: int = 1568                  # Anthropic 推荐最大边长
    max_images_per_message: int = 5
    cache_ttl_seconds: int = 86400             # 24h
    download_timeout: float = 15.0


@dataclass
class ImageGenerationConfig:
    """OpenAI-compatible image generation config."""

    enabled: bool = False
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model: str = "gpt-image-2"
    timeout: int = 120
    max_retries: int = 2
    size: str = "1024x1024"
    quality: str = "auto"
    moderation: str = "auto"
    background: str = "auto"
    output_format: str = "png"
    response_format: str = "b64_json"
    output_dir: str = "./data/generated_images"
    output_prefix: str = "gpt_image"
    n: int = 1
    output_compression: Optional[int] = None
    user: Optional[str] = None
    default_prompt: str = ""

    def __post_init__(self):
        if self.api_key is None:
            self.api_key = (
                os.environ.get("OPENAI_IMAGE_API_KEY")
                or os.environ.get("OPENAI_API_KEY")
            )

        if self.base_url is None:
            self.base_url = (
                os.environ.get("OPENAI_IMAGE_BASE_URL")
                or os.environ.get("OPENAI_BASE_URL")
                or "https://api.openai.com/v1"
            )


@dataclass
class VisualPerceptionSettings:
    """动态视觉感知配置。"""
    enabled: bool = False
    vision_analysis_mode: str = "triggered"  # "none" | "triggered" | "per_event"
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
    vision_rate_limit_wait_for_file_source: bool = True
    vision_rate_limit_max_wait_seconds: float = 90.0
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
    weak_monitor_buffer_seconds: float = 30.0
    segment_merge_gap_seconds: float = 2.0
    trigger_analysis_enabled: bool = True
    trigger_window_seconds: float = 10.0
    trigger_accumulated_score_threshold: float = 1.2
    trigger_peak_score_threshold: float = 0.35
    trigger_min_strong_events: int = 2
    trigger_refractory_seconds: float = 8.0
    explicit_request_top_k: int = 2
    summary_enabled: bool = True
    summary_window_seconds: float = 30.0
    summary_top_k: int = 3
    visual_emotion_scale: float = 0.2         # 视觉弱刺激注入情绪状态机时的缩放系数
    memory_peak_score_threshold: float = 0.22 # 写入长期记忆的最低峰值分数
    clip_duration_seconds: float = 0.0        # 事件视频片段时长（秒），0=单帧模式
    clip_max_frames: int = 8                  # 片段模式下最多采样帧数
    # ── 自适应帧采样（FFT 频域驱动） ──
    adaptive_sampling_enabled: bool = False
    adaptive_stft_window_size: int = 32
    adaptive_stft_hop_size: int = 4
    adaptive_highfreq_cutoff_ratio: float = 0.3
    adaptive_fps_min: float = 4.0
    adaptive_fps_max: float = 15.0
    adaptive_gamma: float = 0.7
    adaptive_spike_threshold: float = 0.4
    adaptive_spike_boost_seconds: float = 2.0
    adaptive_fps_smoothing_alpha: float = 0.3
    adaptive_precheck_diff_threshold: float = 15.0
    adaptive_precheck_resize_width: int = 160



# ============== QQ 机器人配置 ==============
@dataclass
class QQBotConfig:
    """QQ 机器人适配层配置。"""
    ws_host: str = "0.0.0.0"         # WebSocket 监听地址
    ws_port: int = 8080              # WebSocket 监听端口
    ws_path: str = "/xm"             # WebSocket 路径，需与 NapCat 配置一致
    bot_qq: int = 0                  # 机器人 QQ 号，用于检测 @
    owner_qq: int = 0                # 主人 QQ 号，始终拥有管理权限
    owner_name: str = ""             # 对话中的主人称呼，留空则用 QQ 昵称
    admin_qq: List[int] = field(default_factory=list)  # 其他管理员 QQ 列表
    command_prefix: str = "/"        # 命令前缀
    reply_with_at: bool = True       # 群聊回复时是否 @ 对方
    max_message_length: int = 500    # 单条消息最大长度，超出则分条发送


# ============== TTS 音频配置 ==============
@dataclass
class AudioConfig:
    """TTS 语音合成配置"""
    enabled: bool = True
    tts_provider: str = "cosyvoice"   # "cosyvoice" | "edge-tts" | "indextts2"

    # CosyVoice
    cosyvoice_repo_dir: str = "D:\\Users\\21405\\source\\repos\\MyNeuroLikeSystem\\CosyVoice2\\CosyVoice"
    cosyvoice_model_dir: str = "./models/CosyVoice2-0.5B"
    ref_audio_dir: str = "./data/audio_refs"
    default_ref_audio: str = "test2.wav"
    default_ref_text: str = ""

    # edge-tts
    edge_tts_voice: str = "zh-CN-XiaoxiaoNeural"
    edge_tts_rate: str = "+0%"
    edge_tts_volume: str = "+0%"
    edge_tts_pitch: str = "+0Hz"
    edge_tts_proxy: Optional[str] = None

    # IndexTTS2
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

    # 情绪 -> 参考提示映射。对 CosyVoice 使用 audio/text；对 IndexTTS2 可复用为 emotion audio/text prompt。
    emotion_ref_map: Dict[str, Dict[str, str]] = field(default_factory=dict)

    cache_dir: str = "./data/audio_cache"
    cache_enabled: bool = True
    auto_play: bool = False


# ============== ASR 语音识别配置 ==============
@dataclass
class SenseVoiceConfig:
    """SenseVoice 语音识别配置。"""
    enabled: bool = True
    model_id: str = "FunAudioLLM/SenseVoiceSmall"
    model_dir: Optional[str] = None   # 本地模型路径，优先于 model_id
    device: str = "cuda"              # "cuda" | "cpu"
    language: str = "auto"            # "zh" | "en" | "auto"
    use_emotion: bool = True          # 是否启用情感识别
    use_vad: bool = True              # 是否启用 VAD
    batch_size: int = 1


# ============== 调度器配置 ==============
@dataclass
class SchedulerConfig:
    """PersonaScheduler 配置。"""
    max_concurrent_llm: int = 3            # LLM 并发信号量上限
    llm_acquire_timeout: float = 30.0      # 信号量获取超时（秒）
    health_check_interval: float = 60.0    # 健康检查间隔（秒）
    default_persona: Optional[str] = None  # 路由 miss 时的兜底 persona 名称


# ============== API 安全配置 ==============
@dataclass
class SecurityConfig:
    """API 服务安全配置，涵盖认证、限流、封禁、体积限制和 HTTPS。"""
    enabled: bool = False                      # 认证总开关；False 表示向后兼容
    api_keys: List[str] = field(default_factory=list)

    rate_limit_per_minute: int = 30            # 每 IP / 分钟
    rate_limit_per_key_per_minute: int = 60    # 每 Key / 分钟
    max_concurrent_chat: int = 5               # 同时 pipeline.chat() 上限
    max_request_body_bytes: int = 10_485_760   # 10 MB
    max_messages_per_request: int = 50

    auth_fail_ban_threshold: int = 10          # 连续认证失败 N 次后封 IP
    auth_fail_ban_duration_seconds: int = 600  # 封禁 10 分钟
    ip_whitelist: List[str] = field(default_factory=list)

    cors_enabled: bool = False
    cors_origins: List[str] = field(default_factory=lambda: ["*"])

    ssl_certfile: Optional[str] = None
    ssl_keyfile: Optional[str] = None

    debug: bool = False  # API debug 日志开关
    include_debug_in_response: bool = False  # 是否在 API 响应中包含 debug 信息

    def __post_init__(self):
        env_keys = os.environ.get("API_SERVICE_KEYS", "")
        if env_keys:
            self.api_keys = [k.strip() for k in env_keys.split(",") if k.strip()]
