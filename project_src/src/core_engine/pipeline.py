"""
核心推理管线实现。

由 SharedInfra + PersonaInstance 组成，是当前核心引擎的主实现入口。
"""

from __future__ import annotations

from typing import Dict, List, Optional

from configs.config_loader import AppConfig
from configs.model_config import (
    AttentionConfig,
    EmotionFusionConfig,
    EmotionPromptConfig,
    EmotionStateConfig,
    LLMConfig,
    MemoryConfig,
    PersonalityConfig,
    SmallModelConfig,
    VisualPerceptionSettings,
)
from src.core_engine.runtime_types import MemoryManager
from src.llm.client import LLMClient
from src.logger import logger


class NeuroLikePipeline:
    """
    核心推理管线。

    内部由 SharedInfra + PersonaInstance 组成，负责组合共享基础设施与单人格状态。
    """

    def __init__(
        self,
        small_model_path: str,
        personality: PersonalityConfig,
        llm_config: Optional[LLMConfig] = None,
        llm_secondary_config: Optional[LLMConfig] = None,
        llm_vision_config: Optional[LLMConfig] = None,
        memory_config: Optional[MemoryConfig] = None,
        emotion_prompt_config: Optional[EmotionPromptConfig] = None,
        emotion_fusion_config: Optional[EmotionFusionConfig] = None,
        emotion_state_config: Optional[EmotionStateConfig] = None,
        attention_config: Optional[AttentionConfig] = None,
        visual_perception_config: Optional[VisualPerceptionSettings] = None,
        # 向后兼容的参数
        llm_provider: Optional[str] = None,
        llm_api_key: Optional[str] = None,
        llm_model: Optional[str] = None,
        llm_base_url: Optional[str] = None,
        device: str = "cpu",
        time_awareness: bool = True,
        small_model_config: Optional[SmallModelConfig] = None,
    ):
        from configs.model_config import LLMProvider as _LLMProvider
        from src.core_engine.persona import PersonaInstance
        from src.core_engine.shared_infra import BERTInferenceEngine, LLMClientPool, SharedInfra

        # ── 构建 SharedInfra ──────────────────────────────────────────
        if small_model_config is None:
            small_model_config = SmallModelConfig(
                checkpoint_path=small_model_path,
                device=device,
            )

        bert = BERTInferenceEngine(small_model_config=small_model_config)

        # 兼容旧式参数（llm_provider / llm_api_key / ...）
        if llm_config is None and llm_provider is not None:
            llm_config = LLMConfig(
                provider=_LLMProvider(llm_provider),
                api_key=llm_api_key,
                model=llm_model,
                base_url=llm_base_url,
            )

        if llm_config is None:
            raise ValueError(
                "未提供 LLM 配置。请使用 NeuroLikePipeline.from_config() 从 config.json 加载，"
                "或通过 llm_config 参数传入 LLMConfig 对象。"
            )

        llm_pool = LLMClientPool(
            primary_config=llm_config,
            secondary_config=llm_secondary_config,
            vision_config=llm_vision_config,
        )

        # 情绪融合引擎
        from src.core_engine.emotion_fusion import EmotionNeuronFusion, LLMEmotionClassifier

        llm_emotion_classifier = None
        emotion_fusion = None
        if emotion_fusion_config and emotion_fusion_config.enabled and emotion_prompt_config:
            llm_for_emotion = llm_pool.secondary or llm_pool.primary
            llm_emotion_classifier = LLMEmotionClassifier(
                llm_client=llm_for_emotion,
                temperature=emotion_fusion_config.llm_temperature,
            )
            emotion_fusion = EmotionNeuronFusion(
                config=emotion_fusion_config,
                emotion_reliability=emotion_prompt_config.emotion_reliability,
            )
            logger.info("情绪融合系统已启用")

        self._infra = SharedInfra(
            bert=bert,
            llm_pool=llm_pool,
            llm_emotion_classifier=llm_emotion_classifier,
            emotion_fusion=emotion_fusion,
            emotion_fusion_config=emotion_fusion_config,
        )

        # ── 构建 PersonaInstance ──────────────────────────────────────
        self._persona = PersonaInstance(
            infra=self._infra,
            personality=personality,
            memory_config=memory_config,
            emotion_prompt_config=emotion_prompt_config,
            emotion_fusion_config=emotion_fusion_config,
            emotion_state_config=emotion_state_config,
            attention_config=attention_config,
            visual_perception_config=visual_perception_config,
            time_awareness=time_awareness,
        )

    @classmethod
    def from_config(cls, config_path: Optional[str] = None) -> "NeuroLikePipeline":
        """从 config.json 创建 Pipeline（推荐入口）"""
        app_cfg = AppConfig.load(config_path)
        logger.info(f"加载配置: {app_cfg}")
        return cls(
            small_model_path=app_cfg.small_model_checkpoint,
            personality=app_cfg.personality,
            llm_config=app_cfg.llm,
            llm_secondary_config=app_cfg.llm_secondary,
            llm_vision_config=app_cfg.llm_vision,
            memory_config=app_cfg.memory,
            emotion_prompt_config=app_cfg.emotion_prompts,
            emotion_fusion_config=app_cfg.emotion_fusion,
            emotion_state_config=app_cfg.emotion_state_config,
            attention_config=app_cfg.attention,
            visual_perception_config=app_cfg.visual_perception,
            device=app_cfg.device,
            time_awareness=app_cfg.agent.time_awareness,
            small_model_config=app_cfg.small_model,
        )

    # ── 委托属性（AgentLoop 等需要访问）──────────────────────────────────

    @property
    def personality(self) -> PersonalityConfig:
        return self._persona.personality

    @property
    def memory(self):
        return self._persona.memory

    @property
    def attention_tracker(self):
        return self._persona.attention_tracker

    @property
    def llm_client(self) -> "LLMClient":
        return self._persona.llm_client

    @property
    def llm_client_secondary(self) -> Optional["LLMClient"]:
        return self._persona.llm_client_secondary

    @property
    def llm_client_vision(self) -> Optional["LLMClient"]:
        return self._persona.llm_client_vision

    @property
    def small_model(self):
        return self._persona.small_model

    @property
    def emotion_state_tracker(self):
        return self._persona.emotion_state_tracker

    @property
    def emotion_prompt_config(self):
        return self._persona.emotion_prompt_config

    @property
    def visual_perception_config(self):
        return self._persona.visual_perception_config

    @property
    def emotion_fusion_config(self):
        return self._persona.emotion_fusion_config

    @property
    def device(self) -> str:
        return self._infra.bert.device

    # ── 委托方法 ──────────────────────────────────────────────────────────

    def chat(self, user_input: str, **kwargs) -> Dict:
        return self._persona.chat(user_input, **kwargs)

    def generate_response(self, user_input: str, **kwargs) -> str:
        return self._persona.generate_response(user_input, **kwargs)

    def generate_proactive(self, trigger: str, **kwargs) -> Optional[str]:
        return self._persona.generate_proactive(trigger, **kwargs)

    def analyze_emotion_behavior(self, text: str) -> Optional[Dict]:
        return self._persona.analyze_emotion_behavior(text)

    def should_respond(self, emotion_behavior: Dict, is_mentioned: bool = False) -> bool:
        return self._persona.should_respond(emotion_behavior, is_mentioned)

    def build_system_prompt(
        self,
        recalled_context: str = "",
        emotion_analysis: Optional[Dict] = None,
    ) -> str:
        hint = None
        if self._persona.emotion_state_tracker:
            hint = self._persona.emotion_state_tracker.get_prompt_hint()
        return self._persona.prompt_builder.build_system_prompt(
            recalled_context,
            emotion_analysis,
            hint,
        )

    def build_system_prompt_blocks(
        self,
        recalled_context: str = "",
        emotion_analysis: Optional[Dict] = None,
    ) -> List[Dict]:
        hint = None
        if self._persona.emotion_state_tracker:
            hint = self._persona.emotion_state_tracker.get_prompt_hint()
        return self._persona.prompt_builder.build_system_prompt_blocks(
            recalled_context,
            emotion_analysis,
            hint,
        )

    def save_conversation(self, output_path: str):
        self._persona.save_conversation(output_path)

    def close(self):
        self._persona.close()

    def handle_visual_event(self, event) -> Dict:
        return self._persona.handle_visual_event(event)

    def _handle_group_passive_message(
        self,
        user_input: str,
        context_id: Optional[str] = None,
    ) -> Dict:
        return self._persona._handle_group_passive_message(user_input, context_id)


__all__ = ["NeuroLikePipeline", "MemoryManager"]
