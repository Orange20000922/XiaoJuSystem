"""
联合模型 (Joint Model) - 推荐方案
将情绪识别和行为生成合并到一个模型中，更简单高效
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

import sys
sys.path.append("..")
from configs.model_config import (
    JointModelConfig,
    NUM_EMOTIONS,
    NUM_BEHAVIORS,
    NUM_TONES,
    ID_TO_EMOTION,
    ID_TO_BEHAVIOR,
    ID_TO_TONE
)


@dataclass
class JointOutput:
    """联合模型输出"""
    # 情绪相关
    emotion_logits: torch.Tensor      # [batch, num_emotions]
    emotion_probs: torch.Tensor       # [batch, num_emotions]
    intensity: torch.Tensor           # [batch, 1]
    primary_emotion: torch.Tensor     # [batch]

    # 行为相关
    behavior_logits: torch.Tensor     # [batch, num_behaviors]
    behavior_probs: torch.Tensor      # [batch, num_behaviors]
    tone_logits: torch.Tensor         # [batch, num_tones]
    tone_probs: torch.Tensor          # [batch, num_tones]
    response_length: torch.Tensor     # [batch, 3]
    primary_behavior: torch.Tensor    # [batch]
    primary_tone: torch.Tensor        # [batch]

    # 共享表示
    shared_hidden: torch.Tensor       # [batch, hidden_size]

    loss: Optional[torch.Tensor] = None


class SharedEncoder(nn.Module):
    """共享编码器 (文本 + 人格)"""

    def __init__(
        self,
        pretrained_model: str,
        personality_dim: int,
        hidden_size: int,
        dropout: float = 0.1
    ):
        super().__init__()

        # 预训练编码器
        self.text_encoder = AutoModel.from_pretrained(pretrained_model)
        self.hidden_size = self.text_encoder.config.hidden_size

        # 人格融合
        self.personality_proj = nn.Sequential(
            nn.Linear(personality_dim, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size // 2, self.hidden_size)
        )

        # 门控融合
        self.gate = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.Sigmoid()
        )

        # 共享表示层
        self.shared_layer = nn.Sequential(
            nn.Linear(self.hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_size)
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        personality: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
            personality: [batch, personality_dim]

        Returns:
            shared_hidden: [batch, hidden_size]
        """
        # 编码文本
        text_output = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        text_hidden = text_output.last_hidden_state[:, 0, :]  # CLS

        # 投影人格
        personality_hidden = self.personality_proj(personality)

        # 门控融合
        concat = torch.cat([text_hidden, personality_hidden], dim=-1)
        gate = self.gate(concat)
        fused = gate * text_hidden + (1 - gate) * personality_hidden

        # 共享表示
        shared_hidden = self.shared_layer(fused)

        return shared_hidden


class EmotionHead(nn.Module):
    """情绪识别头"""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_emotions: int,
        dropout: float = 0.1
    ):
        super().__init__()

        # 情绪分类器
        self.emotion_classifier = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_size, num_emotions)
        )

        # 强度回归器
        self.intensity_regressor = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_size // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        emotion_logits = self.emotion_classifier(hidden)
        intensity = self.intensity_regressor(hidden)
        return emotion_logits, intensity


class BehaviorHead(nn.Module):
    """行为决策头"""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_behaviors: int,
        num_tones: int,
        dropout: float = 0.1
    ):
        super().__init__()

        # 行为分类器
        self.behavior_classifier = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_size, num_behaviors)
        )

        # 语气分类器
        self.tone_classifier = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_size, num_tones)
        )

        # 回复长度预测
        self.length_predictor = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_size // 2, 3)
        )

    def forward(
        self,
        hidden: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        behavior_logits = self.behavior_classifier(hidden)
        tone_logits = self.tone_classifier(hidden)
        length_logits = self.length_predictor(hidden)
        return behavior_logits, tone_logits, length_logits


class EmotionBehaviorInteraction(nn.Module):
    """情绪-行为交互模块

    让情绪信息影响行为决策
    """

    def __init__(self, hidden_size: int, num_emotions: int, dropout: float = 0.1):
        super().__init__()

        # 情绪条件化
        self.emotion_condition = nn.Sequential(
            nn.Linear(num_emotions + hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_size)
        )

    def forward(
        self,
        shared_hidden: torch.Tensor,
        emotion_probs: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            shared_hidden: [batch, hidden_size] 共享表示
            emotion_probs: [batch, num_emotions] 情绪概率

        Returns:
            conditioned_hidden: [batch, hidden_size] 条件化后的表示
        """
        concat = torch.cat([shared_hidden, emotion_probs], dim=-1)
        conditioned = self.emotion_condition(concat)
        return conditioned


class JointEmotionBehaviorModel(nn.Module):
    """
    联合情绪-行为模型 (推荐方案)

    架构:
    [用户输入 + 人格] -> [Shared Encoder] -> [共享表示]
                                              ↓
                                    ┌─────────┴─────────┐
                                    ↓                   ↓
                              [Emotion Head]      [Interaction]
                                    ↓                   ↓
                              情绪+强度          [Behavior Head]
                                                        ↓
                                                  行为+语气+长度

    优势:
    1. 参数共享，模型更小 (~100M vs 200M)
    2. 情绪和行为联合训练，一致性更好
    3. 推理更快 (一次前向传播)
    4. 训练更简单 (一个模型)
    """

    def __init__(self, config: JointModelConfig):
        super().__init__()
        self.config = config

        # 共享编码器
        self.shared_encoder = SharedEncoder(
            pretrained_model=config.pretrained_model,
            personality_dim=config.personality_dim,
            hidden_size=config.hidden_size,
            dropout=config.dropout
        )

        # 情绪头
        self.emotion_head = EmotionHead(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_emotions=config.num_emotions,
            dropout=config.dropout
        )

        # 情绪-行为交互
        self.interaction = EmotionBehaviorInteraction(
            hidden_size=config.hidden_size,
            num_emotions=config.num_emotions,
            dropout=config.dropout
        )

        # 行为头
        self.behavior_head = BehaviorHead(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_behaviors=config.num_behaviors,
            num_tones=config.num_tones,
            dropout=config.dropout
        )

        # 损失函数
        self.emotion_loss_fn = nn.CrossEntropyLoss()
        self.intensity_loss_fn = nn.MSELoss()
        self.behavior_loss_fn = nn.CrossEntropyLoss()
        self.tone_loss_fn = nn.CrossEntropyLoss()
        self.length_loss_fn = nn.CrossEntropyLoss()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        personality: torch.Tensor,
        emotion_labels: Optional[torch.Tensor] = None,
        intensity_labels: Optional[torch.Tensor] = None,
        behavior_labels: Optional[torch.Tensor] = None,
        tone_labels: Optional[torch.Tensor] = None,
        length_labels: Optional[torch.Tensor] = None
    ) -> JointOutput:
        """
        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
            personality: [batch, personality_dim]
            emotion_labels: [batch] (训练时)
            intensity_labels: [batch] (训练时)
            behavior_labels: [batch] (训练时)
            tone_labels: [batch] (训练时)
            length_labels: [batch] (训练时)

        Returns:
            JointOutput
        """
        # 共享编码
        shared_hidden = self.shared_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            personality=personality
        )

        # 情绪预测
        emotion_logits, intensity = self.emotion_head(shared_hidden)
        emotion_probs = F.softmax(emotion_logits, dim=-1)
        primary_emotion = torch.argmax(emotion_probs, dim=-1)

        # 情绪条件化的行为预测
        conditioned_hidden = self.interaction(shared_hidden, emotion_probs)
        behavior_logits, tone_logits, length_logits = self.behavior_head(conditioned_hidden)

        behavior_probs = F.softmax(behavior_logits, dim=-1)
        tone_probs = F.softmax(tone_logits, dim=-1)
        length_probs = F.softmax(length_logits, dim=-1)

        primary_behavior = torch.argmax(behavior_probs, dim=-1)
        primary_tone = torch.argmax(tone_probs, dim=-1)

        # 计算损失
        loss = None
        if emotion_labels is not None:
            # 情绪损失
            emotion_loss = self.emotion_loss_fn(emotion_logits, emotion_labels)
            loss = self.config.emotion_loss_weight * emotion_loss

            # 强度损失
            if intensity_labels is not None:
                intensity_loss = self.intensity_loss_fn(
                    intensity.squeeze(-1),
                    intensity_labels
                )
                loss = loss + self.config.intensity_loss_weight * intensity_loss

            # 行为损失
            if behavior_labels is not None:
                behavior_loss = self.behavior_loss_fn(behavior_logits, behavior_labels)
                loss = loss + self.config.behavior_loss_weight * behavior_loss

            # 语气损失
            if tone_labels is not None:
                tone_loss = self.tone_loss_fn(tone_logits, tone_labels)
                loss = loss + self.config.tone_loss_weight * tone_loss

            # 长度损失
            if length_labels is not None:
                length_loss = self.length_loss_fn(length_logits, length_labels)
                loss = loss + 0.3 * length_loss

        return JointOutput(
            emotion_logits=emotion_logits,
            emotion_probs=emotion_probs,
            intensity=intensity,
            primary_emotion=primary_emotion,
            behavior_logits=behavior_logits,
            behavior_probs=behavior_probs,
            tone_logits=tone_logits,
            tone_probs=tone_probs,
            response_length=length_probs,
            primary_behavior=primary_behavior,
            primary_tone=primary_tone,
            shared_hidden=shared_hidden,
            loss=loss
        )

    def predict(
        self,
        text: str,
        personality: torch.Tensor,
        tokenizer: AutoTokenizer,
        device: str = "cpu"
    ) -> Dict:
        """
        单条预测 (一次前向传播得到所有结果)

        Args:
            text: 输入文本
            personality: 人格向量
            tokenizer: 分词器
            device: 设备

        Returns:
            完整预测结果
        """
        self.eval()

        # 分词
        encoding = tokenizer(
            text,
            max_length=self.config.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)
        personality = personality.unsqueeze(0).to(device)

        # 推理
        with torch.no_grad():
            output = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                personality=personality
            )

        # 转换结果
        emotion_id = output.primary_emotion.item()
        behavior_id = output.primary_behavior.item()
        tone_id = output.primary_tone.item()
        length_id = torch.argmax(output.response_length, dim=-1).item()

        length_map = {0: "short", 1: "medium", 2: "long"}

        return {
            # 情绪信息
            "emotion": {
                "primary": ID_TO_EMOTION[emotion_id],
                "primary_prob": output.emotion_probs[0, emotion_id].item(),
                "intensity": output.intensity.item(),
                "all_probs": {
                    ID_TO_EMOTION[i]: output.emotion_probs[0, i].item()
                    for i in range(len(ID_TO_EMOTION))
                }
            },
            # 行为信息
            "behavior": {
                "type": ID_TO_BEHAVIOR[behavior_id],
                "type_prob": output.behavior_probs[0, behavior_id].item(),
                "tone": ID_TO_TONE[tone_id],
                "tone_prob": output.tone_probs[0, tone_id].item(),
                "response_length": length_map[length_id],
                "all_behaviors": {
                    ID_TO_BEHAVIOR[i]: output.behavior_probs[0, i].item()
                    for i in range(len(ID_TO_BEHAVIOR))
                }
            }
        }

    def to_json_format(self, prediction: Dict) -> Dict:
        """
        转换为JSON格式，供大模型使用

        Returns:
            {
                "emotion": {"primary": "joy", "intensity": 0.8},
                "behavior": {"type": "respond_positive", "tone": "enthusiastic"},
                "response_length": "medium"
            }
        """
        return {
            "emotion": {
                "primary": prediction["emotion"]["primary"],
                "intensity": prediction["emotion"]["intensity"]
            },
            "behavior": {
                "type": prediction["behavior"]["type"],
                "tone": prediction["behavior"]["tone"]
            },
            "response_length": prediction["behavior"]["response_length"]
        }


def create_joint_model(
    config: Optional[JointModelConfig] = None
) -> Tuple[JointEmotionBehaviorModel, AutoTokenizer]:
    """
    创建联合模型和分词器

    Args:
        config: 模型配置

    Returns:
        (model, tokenizer)
    """
    if config is None:
        config = JointModelConfig()

    model = JointEmotionBehaviorModel(config)
    tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model)

    return model, tokenizer


# ============== 测试代码 ==============
if __name__ == "__main__":
    from configs.model_config import DEFAULT_JOINT_CONFIG, DEFAULT_PERSONALITY

    print("创建联合模型...")
    model, tokenizer = create_joint_model(DEFAULT_JOINT_CONFIG)

    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 测试推理
    test_text = "今天天气真好，心情很愉快！"
    personality_vec = torch.tensor(DEFAULT_PERSONALITY.to_embedding_vector())

    result = model.predict(test_text, personality_vec, tokenizer)

    print(f"\n输入: {test_text}")
    print("\n情绪分析:")
    print(f"  主要情绪: {result['emotion']['primary']} (置信度: {result['emotion']['primary_prob']:.3f})")
    print(f"  情绪强度: {result['emotion']['intensity']:.3f}")

    print("\n行为决策:")
    print(f"  行为类型: {result['behavior']['type']} (置信度: {result['behavior']['type_prob']:.3f})")
    print(f"  语气: {result['behavior']['tone']} (置信度: {result['behavior']['tone_prob']:.3f})")
    print(f"  回复长度: {result['behavior']['response_length']}")

    print("\nJSON格式 (供大模型使用):")
    import json
    print(json.dumps(model.to_json_format(result), ensure_ascii=False, indent=2))
