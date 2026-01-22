"""
模型配置文件
定义情绪标签、行为标签、人格特征等
"""

from dataclasses import dataclass, field
from typing import List, Dict
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
    batch_size: int = 32
    learning_rate: float = 2e-5
    num_epochs: int = 10
    warmup_ratio: float = 0.1

    # 多任务权重
    emotion_loss_weight: float = 1.0
    behavior_loss_weight: float = 1.0
    tone_loss_weight: float = 0.5
    intensity_loss_weight: float = 0.5


# ============== 默认配置实例 ==============
DEFAULT_PERSONALITY = PersonalityConfig()
DEFAULT_EMOTION_CONFIG = EmotionModelConfig()
DEFAULT_BEHAVIOR_CONFIG = BehaviorModelConfig()
DEFAULT_JOINT_CONFIG = JointModelConfig()
