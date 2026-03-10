"""
共享基础设施层

单例，线程安全。所有 persona 实例共享：
- BERTInferenceEngine: BERT 推理（threading.Lock 保护）
- LLMClientPool: 主/副/视觉 LLM 客户端
- EmotionFusionEngine: LLM 情绪分类 + 融合（初始化后无状态）
"""

import os
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Optional

import torch

import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.logger import logger
from configs.model_config import (
    LLMConfig,
    EmotionPromptConfig,
    EmotionFusionConfig,
)
from configs.config_loader import AppConfig
from src.llm.client import LLMClient
from src.core.emotion_fusion import LLMEmotionClassifier, EmotionNeuronFusion


# ── HuggingFace 离线模式 ─────────────────────────────────────────────────────

def _auto_set_hf_offline():
    """如果 HuggingFace 模型缓存已存在，自动启用离线模式"""
    model_name = "hfl/chinese-roberta-wwm-ext"
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    model_dir = cache_dir / ("models--" + model_name.replace("/", "--"))
    if model_dir.exists() and "HF_HUB_OFFLINE" not in os.environ:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"


# ── BERT 推理引擎 ─────────────────────────────────────────────────────────────

class BERTInferenceEngine:
    """线程安全的 BERT 推理引擎"""

    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        self._lock = threading.Lock()
        self.device = device
        self.model = None
        self.tokenizer = None

        _auto_set_hf_offline()

        try:
            from models.joint_model import create_joint_model
            logger.info("加载小模型...")
            self.model, self.tokenizer = create_joint_model()
            self._load_checkpoint(checkpoint_path)
            self.model.to(device)
            self.model.eval()
            logger.info("小模型加载成功")
        except Exception as e:
            logger.warning(f"小模型加载失败，进入纯 LLM 模式: {e}")
            self.model = None
            self.tokenizer = None

    @property
    def available(self) -> bool:
        return self.model is not None

    def _load_checkpoint(self, checkpoint_path: str):
        """加载模型检查点"""
        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"加载检查点: {checkpoint_path}")
        if "metrics" in checkpoint:
            metrics = checkpoint["metrics"]
            if "val_loss" in metrics:
                logger.info(f"  验证损失: {metrics['val_loss']:.4f}")
            if "emotion_acc" in metrics:
                logger.info(f"  情绪准确率: {metrics['emotion_acc']:.2%}")

    def predict(self, text: str, personality_vector: torch.Tensor) -> Optional[Dict]:
        """
        线程安全的 BERT 推理。

        Args:
            text: 输入文本
            personality_vector: 人格向量（由 PersonaInstance 提供）

        Returns:
            BERT 预测结果 dict，或 None（模型不可用时）
        """
        if not self.available:
            return None

        with self._lock:
            return self.model.predict(
                text=text,
                personality=personality_vector,
                tokenizer=self.tokenizer,
                device=self.device,
            )


# ── LLM 客户端池 ──────────────────────────────────────────────────────────────

class LLMClientPool:
    """LLM 客户端池：主/副/视觉三路由"""

    def __init__(
        self,
        primary_config: LLMConfig,
        secondary_config: Optional[LLMConfig] = None,
        vision_config: Optional[LLMConfig] = None,
    ):
        logger.info(
            f"初始化主 LLM 客户端 "
            f"({primary_config.provider.value}: {primary_config.model})"
        )
        self.primary = LLMClient(primary_config)

        if secondary_config is not None:
            logger.info(
                f"初始化副 LLM 客户端 "
                f"({secondary_config.provider.value}: {secondary_config.model})"
            )
            self.secondary = LLMClient(secondary_config)
        else:
            self.secondary = None

        if vision_config is not None:
            logger.info(
                f"初始化图片专用 LLM 客户端 "
                f"({vision_config.provider.value}: {vision_config.model})"
            )
            self.vision = LLMClient(vision_config)
        else:
            self.vision = None


# ── 共享基础设施 ──────────────────────────────────────────────────────────────

class SharedInfra:
    """
    共享基础设施 — 创建一次，所有 persona 引用。

    包含：
    - BERT 推理引擎（线程安全）
    - LLM 客户端池
    - 情绪融合引擎（无状态）
    """

    def __init__(
        self,
        bert: BERTInferenceEngine,
        llm_pool: LLMClientPool,
        llm_emotion_classifier: Optional[LLMEmotionClassifier] = None,
        emotion_fusion: Optional[EmotionNeuronFusion] = None,
        emotion_fusion_config: Optional[EmotionFusionConfig] = None,
    ):
        self.bert = bert
        self.llm_pool = llm_pool
        self.llm_emotion_classifier = llm_emotion_classifier
        self.emotion_fusion = emotion_fusion
        self.emotion_fusion_config = emotion_fusion_config
        self.llm_semaphore: Optional[threading.BoundedSemaphore] = None

    @contextmanager
    def llm_gate(self):
        """
        LLM 并发控制门。

        由 PersonaScheduler 注入 llm_semaphore 后生效，
        限制多 persona 同时调用 LLM API 的数量。
        未设置 semaphore 时为 no-op。
        """
        if self.llm_semaphore is None:
            yield
            return
        acquired = self.llm_semaphore.acquire(timeout=30.0)
        if not acquired:
            logger.warning("LLM 信号量获取超时，不阻塞继续执行")
        try:
            yield
        finally:
            if acquired:
                self.llm_semaphore.release()

    @classmethod
    def from_app_config(cls, config: AppConfig) -> "SharedInfra":
        """从 AppConfig 创建共享基础设施"""
        bert = BERTInferenceEngine(
            checkpoint_path=config.small_model_checkpoint,
            device=config.device,
        )

        llm_pool = LLMClientPool(
            primary_config=config.llm,
            secondary_config=config.llm_secondary,
            vision_config=config.llm_vision,
        )

        # 情绪融合（BERT + LLM）
        llm_emotion_classifier = None
        emotion_fusion = None
        ef_cfg = config.emotion_fusion
        ep_cfg = config.emotion_prompts

        if ef_cfg and ef_cfg.enabled and ep_cfg:
            llm_for_emotion = llm_pool.secondary or llm_pool.primary
            llm_emotion_classifier = LLMEmotionClassifier(
                llm_client=llm_for_emotion,
                temperature=ef_cfg.llm_temperature,
            )
            emotion_fusion = EmotionNeuronFusion(
                config=ef_cfg,
                emotion_reliability=ep_cfg.emotion_reliability,
            )
            logger.info("情绪融合系统已启用")

        return cls(
            bert=bert,
            llm_pool=llm_pool,
            llm_emotion_classifier=llm_emotion_classifier,
            emotion_fusion=emotion_fusion,
            emotion_fusion_config=ef_cfg,
        )

    def close(self):
        """关闭共享资源（可选）"""
        pass
