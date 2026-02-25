"""
标注模型 - 用于大规模数据标注的轻量级模型

功能:
- 情绪分类 (10类)
- 情绪强度回归 (0-1)

设计原则:
- 轻量级，推理速度快
- 专注于标注任务，不需要行为生成
- 支持批量推理
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple


@dataclass
class AnnotationModelConfig:
    """标注模型配置"""
    # 基座模型 (使用小一点的模型加速推理)
    pretrained_model: str = "hfl/chinese-roberta-wwm-ext"

    # 模型结构
    hidden_size: int = 768
    num_emotions: int = 10
    dropout: float = 0.1

    # 训练参数
    max_length: int = 64  # 弹幕较短，64够用
    batch_size: int = 64  # 标注时可以用大batch
    learning_rate: float = 2e-5
    num_epochs: int = 5
    warmup_ratio: float = 0.1

    # 推理参数
    inference_batch_size: int = 128  # 推理时更大的batch


@dataclass
class AnnotationOutput:
    """标注模型输出"""
    emotion_logits: torch.Tensor      # [batch, num_emotions]
    emotion_probs: torch.Tensor       # [batch, num_emotions]
    predicted_emotion: torch.Tensor   # [batch]
    intensity: torch.Tensor           # [batch]
    confidence: torch.Tensor          # [batch] 预测置信度
    loss: Optional[torch.Tensor] = None


class AnnotationModel(nn.Module):
    """
    标注模型

    轻量级设计，专注于:
    1. 情绪分类
    2. 强度回归
    """

    def __init__(self, config: AnnotationModelConfig):
        super().__init__()
        self.config = config

        # 加载预训练编码器
        self.encoder = AutoModel.from_pretrained(config.pretrained_model)

        # 情绪分类头
        self.emotion_classifier = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size // 2, config.num_emotions)
        )

        # 强度回归头
        self.intensity_regressor = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.hidden_size // 4),
            nn.GELU(),
            nn.Linear(config.hidden_size // 4, 1),
            nn.Sigmoid()  # 输出0-1
        )

        # 损失函数
        self.emotion_loss_fn = nn.CrossEntropyLoss()
        self.intensity_loss_fn = nn.MSELoss()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        emotion_labels: Optional[torch.Tensor] = None,
        intensity_labels: Optional[torch.Tensor] = None
    ) -> AnnotationOutput:
        """
        前向传播

        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
            emotion_labels: [batch] 情绪标签 (训练时)
            intensity_labels: [batch] 强度标签 (训练时)
        """
        # 编码
        encoder_output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # 使用[CLS] token的表示
        cls_hidden = encoder_output.last_hidden_state[:, 0, :]  # [batch, hidden]

        # 情绪分类
        emotion_logits = self.emotion_classifier(cls_hidden)  # [batch, num_emotions]
        emotion_probs = torch.softmax(emotion_logits, dim=-1)
        predicted_emotion = torch.argmax(emotion_logits, dim=-1)

        # 置信度 = 最大概率
        confidence, _ = torch.max(emotion_probs, dim=-1)

        # 强度回归
        intensity = self.intensity_regressor(cls_hidden).squeeze(-1)  # [batch]

        # 计算损失
        loss = None
        if emotion_labels is not None:
            emotion_loss = self.emotion_loss_fn(emotion_logits, emotion_labels)

            if intensity_labels is not None:
                intensity_loss = self.intensity_loss_fn(intensity, intensity_labels)
                loss = emotion_loss + 0.5 * intensity_loss
            else:
                loss = emotion_loss

        return AnnotationOutput(
            emotion_logits=emotion_logits,
            emotion_probs=emotion_probs,
            predicted_emotion=predicted_emotion,
            intensity=intensity,
            confidence=confidence,
            loss=loss
        )

    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        推理接口

        Returns:
            (predicted_emotion, intensity, confidence)
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(input_ids, attention_mask)
            return output.predicted_emotion, output.intensity, output.confidence


class AnnotationInferencer:
    """
    标注推理器 - 封装批量推理逻辑
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        batch_size: int = 128
    ):
        """
        Args:
            model_path: 模型检查点路径
            device: 推理设备
            batch_size: 批量大小
        """
        self.device = device
        self.batch_size = batch_size

        # 加载检查点
        checkpoint = torch.load(model_path, map_location=device)

        # 恢复配置
        if "config" in checkpoint:
            self.config = checkpoint["config"]
        else:
            self.config = AnnotationModelConfig()

        # 创建模型
        self.model = AnnotationModel(self.config)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(device)
        self.model.eval()

        # 加载tokenizer
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.pretrained_model)

        # 情绪ID映射
        self.id_to_emotion = {
            0: "joy",
            1: "sadness",
            2: "anger",
            3: "fear",
            4: "surprise",
            5: "disgust",
            6: "neutral",
            7: "excitement",
            8: "tenderness",
            9: "curiosity"
        }

    def annotate_texts(
        self,
        texts: List[str],
        show_progress: bool = True
    ) -> List[Dict]:
        """
        批量标注文本

        Args:
            texts: 文本列表
            show_progress: 是否显示进度条

        Returns:
            标注结果列表
        """
        from tqdm import tqdm

        results = []

        # 分批处理
        iterator = range(0, len(texts), self.batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="模型标注")

        for i in iterator:
            batch_texts = texts[i:i + self.batch_size]

            # Tokenize
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
                return_tensors="pt"
            )

            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)

            # 推理
            with torch.no_grad():
                emotions, intensities, confidences = self.model.predict(
                    input_ids, attention_mask
                )

            # 转换结果
            for j, text in enumerate(batch_texts):
                emotion_id = emotions[j].item()
                results.append({
                    "text": text,
                    "emotion": self.id_to_emotion[emotion_id],
                    "intensity": round(intensities[j].item(), 3),
                    "confidence": round(confidences[j].item(), 3),
                    "annotator": "annotation_model"
                })

        return results

    def annotate_file(
        self,
        input_path: str,
        output_path: str,
        text_field: str = "text"
    ):
        """
        标注整个文件

        Args:
            input_path: 输入文件路径
            output_path: 输出文件路径
            text_field: 文本字段名
        """
        import json
        from pathlib import Path

        # 加载数据
        input_path = Path(input_path)
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 提取文本
        if isinstance(data[0], str):
            texts = data
        else:
            texts = [item.get(text_field, "") for item in data]

        print(f"加载了 {len(texts)} 条数据")

        # 标注
        results = self.annotate_texts(texts)

        # 保存
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"标注完成，保存到: {output_path}")

        # 统计
        self._print_stats(results)

    def _print_stats(self, results: List[Dict]):
        """打印统计信息"""
        from collections import Counter

        emotions = Counter(r["emotion"] for r in results)
        avg_confidence = sum(r["confidence"] for r in results) / len(results)
        avg_intensity = sum(r["intensity"] for r in results) / len(results)

        print("\n" + "=" * 40)
        print("标注统计")
        print("=" * 40)
        print(f"总数: {len(results)}")
        print(f"平均置信度: {avg_confidence:.3f}")
        print(f"平均强度: {avg_intensity:.3f}")
        print("\n情绪分布:")
        for emotion, count in emotions.most_common():
            print(f"  {emotion}: {count} ({count/len(results)*100:.1f}%)")


def create_annotation_model(
    config: Optional[AnnotationModelConfig] = None
) -> Tuple[AnnotationModel, "AutoTokenizer"]:
    """
    创建标注模型

    Returns:
        (model, tokenizer)
    """
    from transformers import AutoTokenizer

    if config is None:
        config = AnnotationModelConfig()

    model = AnnotationModel(config)
    tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model)

    return model, tokenizer
