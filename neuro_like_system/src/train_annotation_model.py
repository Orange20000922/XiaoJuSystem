"""
标注模型训练脚本

用于训练用于大规模数据标注的轻量级模型
"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
from pathlib import Path
import argparse
from typing import Dict, List, Optional

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.annotation_model import (
    AnnotationModel,
    AnnotationModelConfig,
    create_annotation_model
)


# 情绪映射
EMOTION_TO_ID = {
    "joy": 0,
    "sadness": 1,
    "anger": 2,
    "fear": 3,
    "surprise": 4,
    "disgust": 5,
    "neutral": 6,
    "excitement": 7,
    "tenderness": 8,
    "curiosity": 9
}


class AnnotationDataset(Dataset):
    """标注数据集"""

    def __init__(
        self,
        data: List[Dict],
        tokenizer,
        max_length: int = 64
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # 文本
        text = item.get("text", "")

        # 情绪标签
        emotion = item.get("emotion", "neutral")
        emotion_id = EMOTION_TO_ID.get(emotion, 6)  # 默认neutral

        # 强度标签
        intensity = item.get("intensity", 0.5)

        # Tokenize
        encoded = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "emotion_labels": torch.tensor(emotion_id, dtype=torch.long),
            "intensity_labels": torch.tensor(intensity, dtype=torch.float)
        }


def load_annotation_data(data_path: str) -> List[Dict]:
    """加载标注数据"""
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 过滤无效数据
    valid_data = []
    for item in data:
        if "text" in item and "emotion" in item:
            if item["emotion"] in EMOTION_TO_ID:
                valid_data.append(item)

    print(f"加载了 {len(valid_data)}/{len(data)} 条有效数据")
    return valid_data


def split_data(data: List[Dict], val_ratio: float = 0.1):
    """划分训练集和验证集"""
    import random
    random.shuffle(data)

    val_size = int(len(data) * val_ratio)
    val_data = data[:val_size]
    train_data = data[val_size:]

    return train_data, val_data


class AnnotationTrainer:
    """标注模型训练器"""

    def __init__(
        self,
        model: AnnotationModel,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        config: AnnotationModelConfig,
        device: str = "cuda",
        output_dir: str = "./checkpoints/annotation_model"
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 混合精度
        self.use_fp16 = device == "cuda"
        self.scaler = GradScaler() if self.use_fp16 else None

        # 优化器
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=0.01
        )

        # 学习率调度
        total_steps = len(train_loader) * config.num_epochs
        warmup_steps = int(total_steps * config.warmup_ratio)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        # 训练状态
        self.best_val_acc = 0
        self.train_losses = []
        self.val_accs = []

    def train_epoch(self, epoch: int) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}")

        for batch in progress_bar:
            # 移动到设备
            batch = {k: v.to(self.device) for k, v in batch.items()}

            # 前向传播
            if self.use_fp16:
                with autocast():
                    output = self.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        emotion_labels=batch["emotion_labels"],
                        intensity_labels=batch["intensity_labels"]
                    )
                    loss = output.loss

                # 反向传播
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                output = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    emotion_labels=batch["emotion_labels"],
                    intensity_labels=batch["intensity_labels"]
                )
                loss = output.loss

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

            self.scheduler.step()
            total_loss += loss.item()

            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{self.scheduler.get_last_lr()[0]:.2e}"
            })

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """验证"""
        if self.val_loader is None:
            return {}

        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        intensity_mae = 0

        for batch in tqdm(self.val_loader, desc="Validating"):
            batch = {k: v.to(self.device) for k, v in batch.items()}

            if self.use_fp16:
                with autocast():
                    output = self.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        emotion_labels=batch["emotion_labels"],
                        intensity_labels=batch["intensity_labels"]
                    )
            else:
                output = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    emotion_labels=batch["emotion_labels"],
                    intensity_labels=batch["intensity_labels"]
                )

            total_loss += output.loss.item()

            # 准确率
            correct += (output.predicted_emotion == batch["emotion_labels"]).sum().item()
            total += batch["emotion_labels"].size(0)

            # 强度MAE
            intensity_mae += torch.abs(
                output.intensity - batch["intensity_labels"]
            ).sum().item()

        return {
            "val_loss": total_loss / len(self.val_loader),
            "val_acc": correct / total,
            "intensity_mae": intensity_mae / total
        }

    def save_checkpoint(self, epoch: int, metrics: Dict[str, float]):
        """保存检查点"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": self.config,
            "metrics": metrics
        }

        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        # 保存最新
        latest_path = self.output_dir / "latest.pt"
        torch.save(checkpoint, latest_path)

        # 保存最佳
        val_acc = metrics.get("val_acc", 0)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            best_path = self.output_dir / "best.pt"
            torch.save(checkpoint, best_path)
            print(f"保存最佳模型: {best_path} (acc={val_acc:.4f})")

    def train(self):
        """完整训练流程"""
        print("=" * 50)
        print("开始训练标注模型")
        print("=" * 50)
        print(f"设备: {self.device}")
        print(f"训练样本: {len(self.train_loader.dataset)}")
        if self.val_loader:
            print(f"验证样本: {len(self.val_loader.dataset)}")
        print(f"批次大小: {self.config.batch_size}")
        print(f"训练轮数: {self.config.num_epochs}")
        print(f"学习率: {self.config.learning_rate}")
        print("=" * 50)

        for epoch in range(1, self.config.num_epochs + 1):
            print(f"\nEpoch {epoch}/{self.config.num_epochs}")

            # 训练
            train_loss = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            print(f"训练损失: {train_loss:.4f}")

            # 验证
            if self.val_loader:
                metrics = self.validate()
                self.val_accs.append(metrics["val_acc"])
                print(f"验证损失: {metrics['val_loss']:.4f}")
                print(f"验证准确率: {metrics['val_acc']:.4f}")
                print(f"强度MAE: {metrics['intensity_mae']:.4f}")
            else:
                metrics = {"train_loss": train_loss}

            # 保存
            self.save_checkpoint(epoch, metrics)

            # 显存
            if self.device == "cuda":
                allocated = torch.cuda.memory_allocated() / 1024**3
                print(f"显存使用: {allocated:.2f}GB")

        print("\n" + "=" * 50)
        print("训练完成!")
        print(f"最佳验证准确率: {self.best_val_acc:.4f}")
        print(f"模型保存在: {self.output_dir}")
        print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="训练标注模型")
    parser.add_argument("--data", type=str, required=True, help="标注数据路径 (GPT标注的JSON)")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/annotation_model", help="输出目录")
    parser.add_argument("--batch_size", type=int, default=64, help="批次大小")
    parser.add_argument("--num_epochs", type=int, default=5, help="训练轮数")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="学习率")
    parser.add_argument("--max_length", type=int, default=64, help="最大序列长度")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="验证集比例")
    parser.add_argument("--device", type=str, default="cuda", help="设备")
    parser.add_argument("--num_workers", type=int, default=0, help="数据加载进程数")

    args = parser.parse_args()

    # 配置
    config = AnnotationModelConfig(
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        max_length=args.max_length
    )

    # 加载数据
    print("加载数据...")
    data = load_annotation_data(args.data)
    train_data, val_data = split_data(data, args.val_ratio)
    print(f"训练集: {len(train_data)}, 验证集: {len(val_data)}")

    # 创建模型
    print("创建模型...")
    model, tokenizer = create_annotation_model(config)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 创建数据集
    train_dataset = AnnotationDataset(train_data, tokenizer, config.max_length)
    val_dataset = AnnotationDataset(val_data, tokenizer, config.max_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    # 训练
    trainer = AnnotationTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=args.device,
        output_dir=args.output_dir
    )

    trainer.train()


if __name__ == "__main__":
    main()
