"""
情绪识别模型 (Encoder-only Transformer)
基于BERT架构，输入用户文本+人格配置，输出情绪标签和强度
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

import sys
sys.path.append("..")
from configs.model_config import EmotionModelConfig, NUM_EMOTIONS


@dataclass
class EmotionOutput:
    """情绪模型输出"""
    emotion_logits: torch.Tensor      # [batch, num_emotions] 情绪分类logits
    emotion_probs: torch.Tensor       # [batch, num_emotions] 情绪概率
    intensity: torch.Tensor           # [batch, 1] 情绪强度 0-1
    primary_emotion: torch.Tensor     # [batch] 主要情绪ID
    secondary_emotions: torch.Tensor  # [batch, 3] 次要情绪ID (top3)
    hidden_state: torch.Tensor        # [batch, hidden_size] 隐藏状态，供下游使用
    loss: Optional[torch.Tensor] = None


class PersonalityFusion(nn.Module):
    """人格特征融合模块"""

    def __init__(self, hidden_size: int, personality_dim: int, dropout: float = 0.1):
        super().__init__()
        self.personality_proj = nn.Sequential(
            nn.Linear(personality_dim, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size)
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Sigmoid()
        )
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, text_hidden: torch.Tensor, personality: torch.Tensor) -> torch.Tensor:
        """
        Args:
            text_hidden: [batch, hidden_size] 文本编码
            personality: [batch, personality_dim] 人格向量
        Returns:
            fused: [batch, hidden_size] 融合后的表示
        """
        # 投影人格向量
        personality_hidden = self.personality_proj(personality)

        # 门控融合
        concat = torch.cat([text_hidden, personality_hidden], dim=-1)
        gate = self.gate(concat)

        # 加权融合
        fused = gate * text_hidden + (1 - gate) * personality_hidden
        fused = self.layer_norm(fused)

        return fused


class EmotionClassifier(nn.Module):
    """情绪分类头"""

    def __init__(self, hidden_size: int, num_emotions: int, dropout: float = 0.1):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_emotions)
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.classifier(hidden)


class IntensityRegressor(nn.Module):
    """情绪强度回归头"""

    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()  # 输出0-1
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.regressor(hidden)


class EmotionRecognitionModel(nn.Module):
    """
    情绪识别模型

    架构:
    [用户输入] -> [BERT Encoder] -> [CLS Token]
                                        ↓
    [人格向量] -> [PersonalityFusion] -> [融合表示]
                                        ↓
                            ┌──────────┴──────────┐
                            ↓                     ↓
                    [EmotionClassifier]   [IntensityRegressor]
                            ↓                     ↓
                      情绪分类logits           情绪强度
    """

    def __init__(self, config: EmotionModelConfig):
        super().__init__()
        self.config = config

        # 加载预训练编码器
        self.encoder = AutoModel.from_pretrained(config.pretrained_model)
        self.hidden_size = self.encoder.config.hidden_size

        # 人格融合模块
        if config.use_personality:
            self.personality_fusion = PersonalityFusion(
                hidden_size=self.hidden_size,
                personality_dim=config.personality_dim,
                dropout=config.dropout
            )
        else:
            self.personality_fusion = None

        # 分类头
        self.emotion_classifier = EmotionClassifier(
            hidden_size=self.hidden_size,
            num_emotions=config.num_emotions,
            dropout=config.dropout
        )

        # 强度回归头
        self.intensity_regressor = IntensityRegressor(
            hidden_size=self.hidden_size,
            dropout=config.dropout
        )

        # 损失函数
        self.emotion_loss_fn = nn.CrossEntropyLoss()
        self.intensity_loss_fn = nn.MSELoss()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        personality: Optional[torch.Tensor] = None,
        emotion_labels: Optional[torch.Tensor] = None,
        intensity_labels: Optional[torch.Tensor] = None
    ) -> EmotionOutput:
        """
        Args:
            input_ids: [batch, seq_len] 输入token IDs
            attention_mask: [batch, seq_len] 注意力掩码
            personality: [batch, personality_dim] 人格向量 (可选)
            emotion_labels: [batch] 情绪标签 (训练时)
            intensity_labels: [batch] 强度标签 (训练时)

        Returns:
            EmotionOutput
        """
        # 编码文本
        encoder_output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        cls_hidden = encoder_output.last_hidden_state[:, 0, :]  # [batch, hidden]

        # 融合人格特征
        if self.personality_fusion is not None and personality is not None:
            fused_hidden = self.personality_fusion(cls_hidden, personality)
        else:
            fused_hidden = cls_hidden

        # 情绪分类
        emotion_logits = self.emotion_classifier(fused_hidden)
        emotion_probs = F.softmax(emotion_logits, dim=-1)

        # 强度回归
        intensity = self.intensity_regressor(fused_hidden)

        # 获取主要和次要情绪
        primary_emotion = torch.argmax(emotion_probs, dim=-1)
        secondary_emotions = torch.topk(emotion_probs, k=3, dim=-1).indices

        # 计算损失
        loss = None
        if emotion_labels is not None:
            emotion_loss = self.emotion_loss_fn(emotion_logits, emotion_labels)
            loss = emotion_loss

            if intensity_labels is not None:
                intensity_loss = self.intensity_loss_fn(
                    intensity.squeeze(-1),
                    intensity_labels
                )
                loss = loss + 0.5 * intensity_loss

        return EmotionOutput(
            emotion_logits=emotion_logits,
            emotion_probs=emotion_probs,
            intensity=intensity,
            primary_emotion=primary_emotion,
            secondary_emotions=secondary_emotions,
            hidden_state=fused_hidden,
            loss=loss
        )

    def predict(
        self,
        text: str,
        tokenizer: AutoTokenizer,
        personality: Optional[torch.Tensor] = None,
        device: str = "cpu"
    ) -> Dict:
        """
        单条文本预测

        Args:
            text: 输入文本
            tokenizer: 分词器
            personality: 人格向量
            device: 设备

        Returns:
            预测结果字典
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

        if personality is not None:
            personality = personality.unsqueeze(0).to(device)

        # 推理
        with torch.no_grad():
            output = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                personality=personality
            )

        # 转换为字典
        from configs.model_config import ID_TO_EMOTION

        primary_id = output.primary_emotion.item()
        secondary_ids = output.secondary_emotions[0].tolist()

        return {
            "primary_emotion": ID_TO_EMOTION[primary_id],
            "primary_prob": output.emotion_probs[0, primary_id].item(),
            "secondary_emotions": [ID_TO_EMOTION[i] for i in secondary_ids],
            "intensity": output.intensity.item(),
            "all_probs": {
                ID_TO_EMOTION[i]: output.emotion_probs[0, i].item()
                for i in range(len(ID_TO_EMOTION))
            }
        }


def create_emotion_model(config: Optional[EmotionModelConfig] = None) -> Tuple[EmotionRecognitionModel, AutoTokenizer]:
    """
    创建情绪识别模型和分词器

    Args:
        config: 模型配置，None则使用默认配置

    Returns:
        (model, tokenizer)
    """
    if config is None:
        config = EmotionModelConfig()

    model = EmotionRecognitionModel(config)
    tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model)

    return model, tokenizer


# ============== 测试代码 ==============
if __name__ == "__main__":
    from configs.model_config import DEFAULT_EMOTION_CONFIG, DEFAULT_PERSONALITY

    print("创建情绪识别模型...")
    model, tokenizer = create_emotion_model(DEFAULT_EMOTION_CONFIG)

    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 测试推理
    test_text = "今天天气真好，心情很愉快！"
    personality_vec = torch.tensor(DEFAULT_PERSONALITY.to_embedding_vector())

    result = model.predict(test_text, tokenizer, personality_vec)

    print(f"\n输入: {test_text}")
    print(f"主要情绪: {result['primary_emotion']} (置信度: {result['primary_prob']:.3f})")
    print(f"情绪强度: {result['intensity']:.3f}")
    print(f"次要情绪: {result['secondary_emotions']}")
