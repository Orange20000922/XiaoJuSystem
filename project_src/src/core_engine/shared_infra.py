"""Shared infrastructure for multi-persona runtime components."""

import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.logger import logger
from configs.model_config import LLMConfig, EmotionFusionConfig
from configs.config_loader import AppConfig
from src.llm.client import LLMClient
from src.core_engine.emotion_fusion import LLMEmotionClassifier, EmotionNeuronFusion
from src.core_engine.small_model import BERTInferenceEngine


class LLMClientPool:
    """Process-wide LLM client pool."""

    def __init__(
        self,
        primary_config: LLMConfig,
        secondary_config: Optional[LLMConfig] = None,
        vision_config: Optional[LLMConfig] = None,
    ):
        logger.info(
            f"Initializing primary LLM client "
            f"({primary_config.provider.value}: {primary_config.model})"
        )
        self.primary = LLMClient(primary_config)

        if secondary_config is not None:
            logger.info(
                f"Initializing secondary LLM client "
                f"({secondary_config.provider.value}: {secondary_config.model})"
            )
            self.secondary = LLMClient(secondary_config)
        else:
            self.secondary = None

        if vision_config is not None:
            logger.info(
                f"Initializing vision LLM client "
                f"({vision_config.provider.value}: {vision_config.model})"
            )
            self.vision = LLMClient(vision_config)
        else:
            self.vision = None


class SharedInfra:
    """Shared runtime dependencies used by all personas."""

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
        self.llm_acquire_timeout: float = 30.0

    @contextmanager
    def llm_gate(self):
        if self.llm_semaphore is None:
            yield
            return

        timeout = max(0.0, float(self.llm_acquire_timeout))
        acquired = self.llm_semaphore.acquire(timeout=timeout)
        if not acquired:
            logger.warning(
                f"Timed out acquiring LLM semaphore after {timeout:.1f}s; continuing without backpressure"
            )
        try:
            yield
        finally:
            if acquired:
                self.llm_semaphore.release()

    @classmethod
    def from_app_config(cls, config: AppConfig) -> "SharedInfra":
        bert = BERTInferenceEngine(
            small_model_config=config.small_model,
        )

        llm_pool = LLMClientPool(
            primary_config=config.llm,
            secondary_config=config.llm_secondary,
            vision_config=config.llm_vision,
        )

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
            logger.info("Emotion fusion subsystem enabled")

        return cls(
            bert=bert,
            llm_pool=llm_pool,
            llm_emotion_classifier=llm_emotion_classifier,
            emotion_fusion=emotion_fusion,
            emotion_fusion_config=ef_cfg,
        )

    def close(self):
        self.bert.close()
