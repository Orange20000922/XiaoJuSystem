"""
训练脚本 - 联合模型 (推荐)
支持混合精度训练和梯度累积，适配8GB显存
"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
from pathlib import Path
import argparse
from typing import Dict, Optional

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.joint_model import create_joint_model
from data.dataset import create_dataloader
from configs.model_config import (
    JointModelConfig,
    JointModelConfigLowVRAM,
    DEFAULT_PERSONALITY
)


class Trainer:
    """训练器 - 支持混合精度和梯度累积"""

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

        # 混合精度训练
        self.use_fp16 = config.fp16 and device == "cuda"
        self.scaler = GradScaler() if self.use_fp16 else None

        # 梯度累积
        self.gradient_accumulation_steps = getattr(config, 'gradient_accumulation_steps', 1)
        self.effective_batch_size = config.batch_size * self.gradient_accumulation_steps

        # 优化器
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=0.01
        )

        # 学习率调度器 (基于有效步数)
        total_steps = (len(train_loader) // self.gradient_accumulation_steps) * config.num_epochs
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
        accumulated_loss = 0
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}")

        self.optimizer.zero_grad()

        for step, batch in enumerate(progress_bar):
            # 移动到设备
            batch = {k: v.to(self.device) for k, v in batch.items()}

            # 前向传播 (混合精度)
            if self.use_fp16:
                with autocast():
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
                    loss = output.loss / self.gradient_accumulation_steps

                # 反向传播 (混合精度)
                self.scaler.scale(loss).backward()
            else:
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
                loss = output.loss / self.gradient_accumulation_steps
                loss.backward()

            accumulated_loss += loss.item() * self.gradient_accumulation_steps
            total_loss += loss.item() * self.gradient_accumulation_steps

            # 梯度累积后更新
            if (step + 1) % self.gradient_accumulation_steps == 0:
                if self.use_fp16:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()

                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1

                # 更新进度条
                progress_bar.set_postfix({
                    "loss": f"{accumulated_loss / self.gradient_accumulation_steps:.4f}",
                    "lr": f"{self.scheduler.get_last_lr()[0]:.2e}"
                })
                accumulated_loss = 0

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

            if self.use_fp16:
                with autocast():
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
            else:
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

        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        # 保存最新检查点
        latest_path = self.output_dir / "latest.pt"
        torch.save(checkpoint, latest_path)

        # 保存最佳检查点
        if metrics.get("val_loss", float("inf")) < self.best_val_loss:
            self.best_val_loss = metrics["val_loss"]
            best_path = self.output_dir / "best.pt"
            torch.save(checkpoint, best_path)
            print(f"保存最佳模型: {best_path}")

        # 保存epoch检查点
        epoch_path = self.output_dir / f"epoch_{epoch}.pt"
        torch.save(checkpoint, epoch_path)

    def train(self):
        """完整训练流程"""
        print("=" * 50)
        print("开始训练")
        print("=" * 50)
        print(f"设备: {self.device}")
        print(f"混合精度(FP16): {self.use_fp16}")
        print(f"训练样本: {len(self.train_loader.dataset)}")
        if self.val_loader:
            print(f"验证样本: {len(self.val_loader.dataset)}")
        print(f"批次大小: {self.config.batch_size}")
        print(f"梯度累积: {self.gradient_accumulation_steps}")
        print(f"有效批次大小: {self.effective_batch_size}")
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

            # 显示显存使用情况
            if self.device == "cuda":
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                print(f"显存使用: {allocated:.2f}GB / {reserved:.2f}GB")

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
    parser.add_argument("--batch_size", type=int, default=16, help="批次大小")
    parser.add_argument("--gradient_accumulation", type=int, default=2, help="梯度累积步数")
    parser.add_argument("--num_epochs", type=int, default=10, help="训练轮数")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="学习率")
    parser.add_argument("--max_length", type=int, default=128, help="最大序列长度")
    parser.add_argument("--device", type=str, default="cuda", help="设备")
    parser.add_argument("--num_workers", type=int, default=0, help="数据加载工作进程数")
    parser.add_argument("--fp16", action="store_true", default=True, help="使用混合精度训练")
    parser.add_argument("--no_fp16", action="store_false", dest="fp16", help="禁用混合精度")
    parser.add_argument("--low_vram", action="store_true", help="8GB显存优化模式")

    args = parser.parse_args()

    # 根据显存选择配置
    if args.low_vram:
        print("使用8GB显存优化配置")
        config = JointModelConfigLowVRAM()
        config.num_epochs = args.num_epochs
        config.learning_rate = args.learning_rate
    else:
        config = JointModelConfig(
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            max_length=args.max_length,
            gradient_accumulation_steps=args.gradient_accumulation,
            fp16=args.fp16
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
        batch_size=config.batch_size,
        max_length=config.max_length,
        shuffle=True,
        num_workers=args.num_workers
    )

    val_loader = None
    if args.val_data:
        val_loader = create_dataloader(
            data_path=args.val_data,
            tokenizer=tokenizer,
            personality=DEFAULT_PERSONALITY,
            batch_size=config.batch_size,
            max_length=config.max_length,
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
