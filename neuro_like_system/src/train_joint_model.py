"""
训练脚本 - 联合模型 (推荐)
"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
from pathlib import Path
import argparse
from typing import Dict, Optional

import sys
sys.path.append("..")
from models.joint_model import create_joint_model
from data.dataset import create_dataloader
from configs.model_config import (
    JointModelConfig,
    PersonalityConfig,
    DEFAULT_JOINT_CONFIG,
    DEFAULT_PERSONALITY
)


class Trainer:
    """训练器"""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        config: JointModelConfig,
        device: str = "cuda",
        output_dir: str = "./checkpoints"
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 优化器
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=0.01
        )

        # 学习率调度器
        total_steps = len(train_loader) * config.num_epochs
        warmup_steps = int(total_steps * config.warmup_ratio)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        # 训练状态
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.train_losses = []
        self.val_losses = []

    def train_epoch(self, epoch: int) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}")

        for batch in progress_bar:
            # 移动到设备
            batch = {k: v.to(self.device) for k, v in batch.items()}

            # 前向传播
            output = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                personality=batch["personality"],
                emotion_labels=batch["emotion_labels"],
                intensity_labels=batch["intensity_labels"],
                behavior_labels=batch["behavior_labels"],
                tone_labels=batch["tone_labels"],
                length_labels=batch["length_labels"]
            )

            loss = output.loss

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()

            # 记录
            total_loss += loss.item()
            self.global_step += 1

            # 更新进度条
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{self.scheduler.get_last_lr()[0]:.2e}"
            })

        avg_loss = total_loss / len(self.train_loader)
        return avg_loss

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """验证"""
        if self.val_loader is None:
            return {}

        self.model.eval()
        total_loss = 0
        emotion_correct = 0
        behavior_correct = 0
        total_samples = 0

        for batch in tqdm(self.val_loader, desc="Validating"):
            batch = {k: v.to(self.device) for k, v in batch.items()}

            output = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                personality=batch["personality"],
                emotion_labels=batch["emotion_labels"],
                intensity_labels=batch["intensity_labels"],
                behavior_labels=batch["behavior_labels"],
                tone_labels=batch["tone_labels"],
                length_labels=batch["length_labels"]
            )

            total_loss += output.loss.item()

            # 计算准确率
            emotion_correct += (output.primary_emotion == batch["emotion_labels"]).sum().item()
            behavior_correct += (output.primary_behavior == batch["behavior_labels"]).sum().item()
            total_samples += batch["input_ids"].size(0)

        metrics = {
            "val_loss": total_loss / len(self.val_loader),
            "emotion_acc": emotion_correct / total_samples,
            "behavior_acc": behavior_correct / total_samples
        }

        return metrics

    def save_checkpoint(self, epoch: int, metrics: Dict[str, float]):
        """保存检查点"""
        checkpoint = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "metrics": metrics,
            "config": self.config
        }

        # 保存最新检查点
        latest_path = self.output_dir / "latest.pt"
        torch.save(checkpoint, latest_path)

        # 保存最佳检查点
        if metrics.get("val_loss", float("inf")) < self.best_val_loss:
            self.best_val_loss = metrics["val_loss"]
            best_path = self.output_dir / "best.pt"
            torch.save(checkpoint, best_path)
            print(f"✓ 保存最佳模型: {best_path}")

        # 保存epoch检查点
        epoch_path = self.output_dir / f"epoch_{epoch}.pt"
        torch.save(checkpoint, epoch_path)

    def train(self):
        """完整训练流程"""
        print("=" * 50)
        print("开始训练")
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
                self.val_losses.append(metrics["val_loss"])
                print(f"验证损失: {metrics['val_loss']:.4f}")
                print(f"情绪准确率: {metrics['emotion_acc']:.2%}")
                print(f"行为准确率: {metrics['behavior_acc']:.2%}")
            else:
                metrics = {"train_loss": train_loss}

            # 保存检查点
            self.save_checkpoint(epoch, metrics)

        # 保存训练历史
        history = {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses
        }
        with open(self.output_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

        print("\n" + "=" * 50)
        print("训练完成！")
        print(f"最佳验证损失: {self.best_val_loss:.4f}")
        print(f"检查点保存在: {self.output_dir}")
        print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="训练联合情绪-行为模型")
    parser.add_argument("--train_data", type=str, required=True, help="训练数据路径")
    parser.add_argument("--val_data", type=str, default=None, help="验证数据路径")
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="输出目录")
    parser.add_argument("--batch_size", type=int, default=32, help="批次大小")
    parser.add_argument("--num_epochs", type=int, default=10, help="训练轮数")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="学习率")
    parser.add_argument("--max_length", type=int, default=128, help="最大序列长度")
    parser.add_argument("--device", type=str, default="cuda", help="设备")
    parser.add_argument("--num_workers", type=int, default=0, help="数据加载工作进程数")

    args = parser.parse_args()

    # 配置
    config = JointModelConfig(
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        max_length=args.max_length
    )

    # 创建模型
    print("创建模型...")
    model, tokenizer = create_joint_model(config)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 创建数据加载器
    print("加载数据...")
    train_loader = create_dataloader(
        data_path=args.train_data,
        tokenizer=tokenizer,
        personality=DEFAULT_PERSONALITY,
        batch_size=args.batch_size,
        max_length=args.max_length,
        shuffle=True,
        num_workers=args.num_workers
    )

    val_loader = None
    if args.val_data:
        val_loader = create_dataloader(
            data_path=args.val_data,
            tokenizer=tokenizer,
            personality=DEFAULT_PERSONALITY,
            batch_size=args.batch_size,
            max_length=args.max_length,
            shuffle=False,
            num_workers=args.num_workers
        )

    # 训练
    trainer = Trainer(
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
