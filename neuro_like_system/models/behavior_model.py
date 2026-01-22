"""
行为生成模型 (Encoder-Decoder Transformer)
基于T5架构，输入用户文本+情绪标签+人格配置，输出行为决策
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, T5EncoderModel, T5Config
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass

import sys
sys.path.append("..")
from configs.model_config import (
    BehaviorModelConfig,
    NUM_BEHAVIORS,
    NUM_TONES,
    NUM_EMOTIONS,
    ID_TO_BEHAVIOR,
    ID_TO_TONE
)


@dataclass
class BehaviorOutput:
    """行为模型输出"""
    behavior_logits: torch.Tensor     # [batch, num_behaviors] 行为分类logits
    behavior_probs: torch.Tensor      # [batch, num_behaviors] 行为概率
    tone_logits: torch.Tensor         # [batch, num_tones] 语气分类logits
    tone_probs: torch.Tensor          # [batch, num_tones] 语气概率
    response_length: torch.Tensor     # [batch, 3] 回复长度 (short/medium/long)
    primary_behavior: torch.Tensor    # [batch] 主要行为ID
    primary_tone: torch.Tensor        # [batch] 主要语气ID
    hidden_state: torch.Tensor        # [batch, hidden_size] 隐藏状态
    loss: Optional[torch.Tensor] = None


class EmotionEncoder(nn.Module):
    """情绪信息编码器"""

    def __init__(self, num_emotions: int, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        # 情绪嵌入
        self.emotion_embedding = nn.Embedding(num_emotions, hidden_size // 2)
        # 强度编码
        self.intensity_encoder = nn.Linear(1, hidden_size // 2)
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_size)
        )

    def forward(
        self,
        emotion_ids: torch.Tensor,
        intensity: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            emotion_ids: [batch] 情绪ID
            intensity: [batch, 1] 情绪强度

        Returns:
            emotion_hidden: [batch, hidden_size]
        """
        emotion_emb = self.emotion_embedding(emotion_ids)  # [batch, hidden/2]
        intensity_emb = self.intensity_encoder(intensity)   # [batch, hidden/2]

        concat = torch.cat([emotion_emb, intensity_emb], dim=-1)
        return self.fusion(concat)


class ContextEncoder(nn.Module):
    """上下文编码器 (处理对话历史)"""

    def __init__(self, hidden_size: int, max_history: int = 5, dropout: float = 0.1):
        super().__init__()
        self.max_history = max_history

        # 历史轮次位置编码
        self.position_embedding = nn.Embedding(max_history, hidden_size)

        # 注意力聚合
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )

        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        history_hidden: torch.Tensor,
        history_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            history_hidden: [batch, num_turns, hidden_size] 历史对话编码
            history_mask: [batch, num_turns] 历史掩码

        Returns:
            context: [batch, hidden_size] 聚合的上下文表示
        """
        batch_size, num_turns, hidden_size = history_hidden.shape

        # 添加位置编码
        positions = torch.arange(num_turns, device=history_hidden.device)
        pos_emb = self.position_embedding(positions)  # [num_turns, hidden]
        history_hidden = history_hidden + pos_emb.unsqueeze(0)

        # 自注意力聚合
        if history_mask is not None:
            # 转换为attention mask格式
            key_padding_mask = ~history_mask.bool()
        else:
            key_padding_mask = None

        # 使用最后一轮作为query
        query = history_hidden[:, -1:, :]  # [batch, 1, hidden]

        attn_output, _ = self.attention(
            query=query,
            key=history_hidden,
            value=history_hidden,
            key_padding_mask=key_padding_mask
        )

        context = self.layer_norm(attn_output.squeeze(1))
        return context


class BehaviorDecisionHead(nn.Module):
    """行为决策头"""

    def __init__(
        self,
        hidden_size: int,
        num_behaviors: int,
        num_tones: int,
        dropout: float = 0.1
    ):
        super().__init__()

        # 行为分类器
        self.behavior_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_behaviors)
        )

        # 语气分类器
        self.tone_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_tones)
        )

        # 回复长度预测 (short=0, medium=1, long=2)
        self.length_predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, 3)
        )

    def forward(self, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        behavior_logits = self.behavior_classifier(hidden)
        tone_logits = self.tone_classifier(hidden)
        length_logits = self.length_predictor(hidden)

        return behavior_logits, tone_logits, length_logits


class MultiModalFusion(nn.Module):
    """多模态融合模块 (文本 + 情绪 + 人格 + 上下文)"""

    def __init__(self, hidden_size: int, personality_dim: int, dropout: float = 0.1):
        super().__init__()

        # 人格投影
        self.personality_proj = nn.Linear(personality_dim, hidden_size)

        # 跨模态注意力
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )

        # 融合MLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(hidden_size * 4, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size)
        )

    def forward(
        self,
        text_hidden: torch.Tensor,
        emotion_hidden: torch.Tensor,
        personality: torch.Tensor,
        context_hidden: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            text_hidden: [batch, hidden_size] 文本编码
            emotion_hidden: [batch, hidden_size] 情绪编码
            personality: [batch, personality_dim] 人格向量
            context_hidden: [batch, hidden_size] 上下文编码 (可选)

        Returns:
            fused: [batch, hidden_size] 融合表示
        """
        batch_size = text_hidden.shape[0]
        hidden_size = text_hidden.shape[1]

        # 投影人格
        personality_hidden = self.personality_proj(personality)

        # 如果没有上下文，用零向量
        if context_hidden is None:
            context_hidden = torch.zeros_like(text_hidden)

        # 拼接所有模态
        all_modalities = torch.stack([
            text_hidden,
            emotion_hidden,
            personality_hidden,
            context_hidden
        ], dim=1)  # [batch, 4, hidden]

        # 跨模态注意力 (以文本为query)
        query = text_hidden.unsqueeze(1)  # [batch, 1, hidden]
        attn_output, _ = self.cross_attention(
            query=query,
            key=all_modalities,
            value=all_modalities
        )
        attn_output = attn_output.squeeze(1)  # [batch, hidden]

        # MLP融合
        concat = torch.cat([
            text_hidden,
            emotion_hidden,
            personality_hidden,
            attn_output
        ], dim=-1)

        fused = self.fusion_mlp(concat)
        return fused


class BehaviorGenerationModel(nn.Module):
    """
    行为生成模型

    架构:
    [用户输入] -> [Text Encoder] ─────────────────┐
                                                  │
    [情绪标签+强度] -> [Emotion Encoder] ─────────┤
                                                  ├─> [MultiModal Fusion] -> [Behavior Decision]
    [人格向量] -> [Personality Proj] ─────────────┤
                                                  │
    [对话历史] -> [Context Encoder] ──────────────┘

    输出: 行为类型 + 语气 + 回复长度
    """

    def __init__(self, config: BehaviorModelConfig):
        super().__init__()
        self.config = config

        # 文本编码器 (使用BERT而非T5，更简单)
        self.text_encoder = AutoModel.from_pretrained(
            "hfl/chinese-roberta-wwm-ext"  # 使用BERT作为编码器
        )
        self.hidden_size = self.text_encoder.config.hidden_size

        # 情绪编码器
        self.emotion_encoder = EmotionEncoder(
            num_emotions=config.emotion_dim,
            hidden_size=self.hidden_size,
            dropout=config.dropout
        )

        # 上下文编码器
        self.context_encoder = ContextEncoder(
            hidden_size=self.hidden_size,
            max_history=5,
            dropout=config.dropout
        )

        # 多模态融合
        self.fusion = MultiModalFusion(
            hidden_size=self.hidden_size,
            personality_dim=config.personality_dim,
            dropout=config.dropout
        )

        # 行为决策头
        self.decision_head = BehaviorDecisionHead(
            hidden_size=self.hidden_size,
            num_behaviors=config.num_behaviors,
            num_tones=config.num_tones,
            dropout=config.dropout
        )

        # 损失函数
        self.behavior_loss_fn = nn.CrossEntropyLoss()
        self.tone_loss_fn = nn.CrossEntropyLoss()
        self.length_loss_fn = nn.CrossEntropyLoss()

    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """编码文本"""
        output = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        return output.last_hidden_state[:, 0, :]  # CLS token

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        emotion_ids: torch.Tensor,
        emotion_intensity: torch.Tensor,
        personality: torch.Tensor,
        history_hidden: Optional[torch.Tensor] = None,
        history_mask: Optional[torch.Tensor] = None,
        behavior_labels: Optional[torch.Tensor] = None,
        tone_labels: Optional[torch.Tensor] = None,
        length_labels: Optional[torch.Tensor] = None
    ) -> BehaviorOutput:
        """
        Args:
            input_ids: [batch, seq_len] 输入token IDs
            attention_mask: [batch, seq_len] 注意力掩码
            emotion_ids: [batch] 情绪ID (来自情绪模型)
            emotion_intensity: [batch, 1] 情绪强度
            personality: [batch, personality_dim] 人格向量
            history_hidden: [batch, num_turns, hidden_size] 历史对话编码 (可选)
            history_mask: [batch, num_turns] 历史掩码
            behavior_labels: [batch] 行为标签 (训练时)
            tone_labels: [batch] 语气标签 (训练时)
            length_labels: [batch] 长度标签 (训练时)

        Returns:
            BehaviorOutput
        """
        # 编码各模态
        text_hidden = self.encode_text(input_ids, attention_mask)
        emotion_hidden = self.emotion_encoder(emotion_ids, emotion_intensity)

        # 编码上下文
        if history_hidden is not None:
            context_hidden = self.context_encoder(history_hidden, history_mask)
        else:
            context_hidden = None

        # 多模态融合
        fused_hidden = self.fusion(
            text_hidden=text_hidden,
            emotion_hidden=emotion_hidden,
            personality=personality,
            context_hidden=context_hidden
        )

        # 行为决策
        behavior_logits, tone_logits, length_logits = self.decision_head(fused_hidden)

        # 计算概率
        behavior_probs = F.softmax(behavior_logits, dim=-1)
        tone_probs = F.softmax(tone_logits, dim=-1)
        length_probs = F.softmax(length_logits, dim=-1)

        # 获取主要预测
        primary_behavior = torch.argmax(behavior_probs, dim=-1)
        primary_tone = torch.argmax(tone_probs, dim=-1)

        # 计算损失
        loss = None
        if behavior_labels is not None:
            behavior_loss = self.behavior_loss_fn(behavior_logits, behavior_labels)
            loss = behavior_loss

            if tone_labels is not None:
                tone_loss = self.tone_loss_fn(tone_logits, tone_labels)
                loss = loss + tone_loss

            if length_labels is not None:
                length_loss = self.length_loss_fn(length_logits, length_labels)
                loss = loss + 0.5 * length_loss

        return BehaviorOutput(
            behavior_logits=behavior_logits,
            behavior_probs=behavior_probs,
            tone_logits=tone_logits,
            tone_probs=tone_probs,
            response_length=length_probs,
            primary_behavior=primary_behavior,
            primary_tone=primary_tone,
            hidden_state=fused_hidden,
            loss=loss
        )

    def predict(
        self,
        text: str,
        emotion_id: int,
        emotion_intensity: float,
        personality: torch.Tensor,
        tokenizer: AutoTokenizer,
        device: str = "cpu"
    ) -> Dict:
        """
        单条预测

        Args:
            text: 输入文本
            emotion_id: 情绪ID (来自情绪模型)
            emotion_intensity: 情绪强度
            personality: 人格向量
            tokenizer: 分词器
            device: 设备

        Returns:
            预测结果字典
        """
        self.eval()

        # 分词
        encoding = tokenizer(
            text,
            max_length=self.config.max_input_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        emotion_ids = torch.tensor([emotion_id], device=device)
        emotion_intensity_tensor = torch.tensor([[emotion_intensity]], device=device)
        personality = personality.unsqueeze(0).to(device)

        # 推理
        with torch.no_grad():
            output = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                emotion_ids=emotion_ids,
                emotion_intensity=emotion_intensity_tensor,
                personality=personality
            )

        # 转换结果
        behavior_id = output.primary_behavior.item()
        tone_id = output.primary_tone.item()
        length_id = torch.argmax(output.response_length, dim=-1).item()

        length_map = {0: "short", 1: "medium", 2: "long"}

        return {
            "behavior": ID_TO_BEHAVIOR[behavior_id],
            "behavior_prob": output.behavior_probs[0, behavior_id].item(),
            "tone": ID_TO_TONE[tone_id],
            "tone_prob": output.tone_probs[0, tone_id].item(),
            "response_length": length_map[length_id],
            "all_behaviors": {
                ID_TO_BEHAVIOR[i]: output.behavior_probs[0, i].item()
                for i in range(len(ID_TO_BEHAVIOR))
            },
            "all_tones": {
                ID_TO_TONE[i]: output.tone_probs[0, i].item()
                for i in range(len(ID_TO_TONE))
            }
        }


def create_behavior_model(
    config: Optional[BehaviorModelConfig] = None
) -> Tuple[BehaviorGenerationModel, AutoTokenizer]:
    """
    创建行为生成模型和分词器

    Args:
        config: 模型配置

    Returns:
        (model, tokenizer)
    """
    if config is None:
        config = BehaviorModelConfig()

    model = BehaviorGenerationModel(config)
    tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")

    return model, tokenizer


# ============== 测试代码 ==============
if __name__ == "__main__":
    from configs.model_config import DEFAULT_BEHAVIOR_CONFIG, DEFAULT_PERSONALITY, EMOTION_TO_ID

    print("创建行为生成模型...")
    model, tokenizer = create_behavior_model(DEFAULT_BEHAVIOR_CONFIG)

    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 测试推理
    test_text = "今天天气真好，心情很愉快！"
    emotion_id = EMOTION_TO_ID["joy"]
    emotion_intensity = 0.8
    personality_vec = torch.tensor(DEFAULT_PERSONALITY.to_embedding_vector())

    result = model.predict(
        text=test_text,
        emotion_id=emotion_id,
        emotion_intensity=emotion_intensity,
        personality=personality_vec,
        tokenizer=tokenizer
    )

    print(f"\n输入: {test_text}")
    print(f"情绪: joy (强度: {emotion_intensity})")
    print(f"行为: {result['behavior']} (置信度: {result['behavior_prob']:.3f})")
    print(f"语气: {result['tone']} (置信度: {result['tone_prob']:.3f})")
    print(f"回复长度: {result['response_length']}")
