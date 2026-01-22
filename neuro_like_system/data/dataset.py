"""
数据集定义
用于训练情绪识别和行为生成模型
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import sys
sys.path.append("..")
from configs.model_config import (
    EMOTION_TO_ID,
    BEHAVIOR_TO_ID,
    TONE_TO_ID,
    PersonalityConfig
)


class EmotionBehaviorDataset(Dataset):
    """
    情绪-行为数据集

    数据格式 (JSON):
    {
        "text": "用户输入文本",
        "emotion": "joy",
        "intensity": 0.8,
        "behavior": "respond_positive",
        "tone": "enthusiastic",
        "response_length": "medium"  # short/medium/long
    }
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: AutoTokenizer,
        personality: PersonalityConfig,
        max_length: int = 128,
        is_training: bool = True
    ):
        self.tokenizer = tokenizer
        self.personality = personality
        self.max_length = max_length
        self.is_training = is_training

        # 加载数据
        self.data = self._load_data(data_path)

        # 人格向量
        self.personality_vector = torch.tensor(
            personality.to_embedding_vector(),
            dtype=torch.float32
        )

        # 长度映射
        self.length_to_id = {"short": 0, "medium": 1, "long": 2}

    def _load_data(self, data_path: str) -> List[Dict]:
        """加载数据文件"""
        path = Path(data_path)

        if path.suffix == ".json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        elif path.suffix == ".jsonl":
            data = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
        else:
            raise ValueError(f"不支持的文件格式: {path.suffix}")

        return data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]

        # 分词
        encoding = self.tokenizer(
            item["text"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        result = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "personality": self.personality_vector,
        }

        # 标签
        if self.is_training:
            # 情绪标签
            emotion = item.get("emotion", "neutral")
            result["emotion_labels"] = torch.tensor(
                EMOTION_TO_ID.get(emotion, EMOTION_TO_ID["neutral"])
            )

            # 强度标签
            intensity = item.get("intensity", 0.5)
            result["intensity_labels"] = torch.tensor(intensity, dtype=torch.float32)

            # 行为标签
            behavior = item.get("behavior", "neutral_acknowledge")
            result["behavior_labels"] = torch.tensor(
                BEHAVIOR_TO_ID.get(behavior, BEHAVIOR_TO_ID["neutral_acknowledge"])
            )

            # 语气标签
            tone = item.get("tone", "calm")
            result["tone_labels"] = torch.tensor(
                TONE_TO_ID.get(tone, TONE_TO_ID["calm"])
            )

            # 长度标签
            length = item.get("response_length", "medium")
            result["length_labels"] = torch.tensor(
                self.length_to_id.get(length, 1)
            )

        return result


class StreamingDataset(Dataset):
    """
    流式数据集 - 用于大规模数据
    逐行读取，节省内存
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: AutoTokenizer,
        personality: PersonalityConfig,
        max_length: int = 128
    ):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.personality = personality
        self.max_length = max_length

        self.personality_vector = torch.tensor(
            personality.to_embedding_vector(),
            dtype=torch.float32
        )

        self.length_to_id = {"short": 0, "medium": 1, "long": 2}

        # 预先计算行偏移
        self.line_offsets = self._build_index()

    def _build_index(self) -> List[int]:
        """构建行索引"""
        offsets = []
        with open(self.data_path, "r", encoding="utf-8") as f:
            offset = 0
            for line in f:
                if line.strip():
                    offsets.append(offset)
                offset += len(line.encode("utf-8"))
        return offsets

    def __len__(self) -> int:
        return len(self.line_offsets)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # 读取指定行
        with open(self.data_path, "r", encoding="utf-8") as f:
            f.seek(self.line_offsets[idx])
            line = f.readline()
            item = json.loads(line)

        # 分词
        encoding = self.tokenizer(
            item["text"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        result = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "personality": self.personality_vector,
        }

        # 标签
        emotion = item.get("emotion", "neutral")
        result["emotion_labels"] = torch.tensor(
            EMOTION_TO_ID.get(emotion, EMOTION_TO_ID["neutral"])
        )

        intensity = item.get("intensity", 0.5)
        result["intensity_labels"] = torch.tensor(intensity, dtype=torch.float32)

        behavior = item.get("behavior", "neutral_acknowledge")
        result["behavior_labels"] = torch.tensor(
            BEHAVIOR_TO_ID.get(behavior, BEHAVIOR_TO_ID["neutral_acknowledge"])
        )

        tone = item.get("tone", "calm")
        result["tone_labels"] = torch.tensor(
            TONE_TO_ID.get(tone, TONE_TO_ID["calm"])
        )

        length = item.get("response_length", "medium")
        result["length_labels"] = torch.tensor(
            self.length_to_id.get(length, 1)
        )

        return result


def create_dataloader(
    data_path: str,
    tokenizer: AutoTokenizer,
    personality: PersonalityConfig,
    batch_size: int = 32,
    max_length: int = 128,
    shuffle: bool = True,
    num_workers: int = 0,
    streaming: bool = False
) -> DataLoader:
    """
    创建数据加载器

    Args:
        data_path: 数据文件路径
        tokenizer: 分词器
        personality: 人格配置
        batch_size: 批次大小
        max_length: 最大序列长度
        shuffle: 是否打乱
        num_workers: 工作进程数
        streaming: 是否使用流式加载

    Returns:
        DataLoader
    """
    if streaming:
        dataset = StreamingDataset(
            data_path=data_path,
            tokenizer=tokenizer,
            personality=personality,
            max_length=max_length
        )
    else:
        dataset = EmotionBehaviorDataset(
            data_path=data_path,
            tokenizer=tokenizer,
            personality=personality,
            max_length=max_length,
            is_training=True
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )


def generate_sample_data(output_path: str, num_samples: int = 100):
    """
    生成示例数据 (用于测试)

    Args:
        output_path: 输出路径
        num_samples: 样本数量
    """
    import random

    emotions = ["joy", "sadness", "anger", "fear", "surprise", "neutral", "excitement"]
    behaviors = ["respond_positive", "respond_negative", "ask_question", "share_experience",
                 "express_empathy", "make_joke", "agree", "neutral_acknowledge"]
    tones = ["enthusiastic", "calm", "playful", "warm", "supportive"]
    lengths = ["short", "medium", "long"]

    sample_texts = [
        "今天天气真好！",
        "我感觉有点累...",
        "你觉得这个怎么样？",
        "太棒了，我很喜欢！",
        "这也太难了吧",
        "哈哈哈笑死我了",
        "嗯，我知道了",
        "能帮我解释一下吗？",
        "我今天遇到了一件有趣的事",
        "感觉有点无聊",
        "这个真的很有意思",
        "我不太确定...",
        "你说得对",
        "让我想想",
        "好的没问题",
    ]

    data = []
    for i in range(num_samples):
        text = random.choice(sample_texts) + f" (样本{i})"
        emotion = random.choice(emotions)
        intensity = round(random.uniform(0.3, 1.0), 2)
        behavior = random.choice(behaviors)
        tone = random.choice(tones)
        length = random.choice(lengths)

        data.append({
            "text": text,
            "emotion": emotion,
            "intensity": intensity,
            "behavior": behavior,
            "tone": tone,
            "response_length": length
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"生成了 {num_samples} 条示例数据: {output_path}")


# ============== 测试代码 ==============
if __name__ == "__main__":
    from transformers import AutoTokenizer
    from configs.model_config import DEFAULT_PERSONALITY

    # 生成示例数据
    sample_path = "../data/sample_data.json"
    generate_sample_data(sample_path, num_samples=100)

    # 测试数据加载
    tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")

    dataloader = create_dataloader(
        data_path=sample_path,
        tokenizer=tokenizer,
        personality=DEFAULT_PERSONALITY,
        batch_size=8
    )

    # 查看一个批次
    batch = next(iter(dataloader))
    print("\n数据批次:")
    for key, value in batch.items():
        print(f"  {key}: {value.shape}")
