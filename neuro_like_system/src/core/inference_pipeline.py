"""
推理 Pipeline — 向后兼容的瘦包装层

原 NeuroLikePipeline 的全部职责已拆分到：
  - SharedInfra:     BERT 推理引擎、LLM 客户端池、情绪融合引擎（共享、线程安全）
  - PersonaInstance:  记忆管理器、情绪状态机、注意力追踪、prompt 构建（每 persona 独立）

本文件保留 NeuroLikePipeline 类，内部创建 SharedInfra + PersonaInstance，
所有方法和属性委托到 PersonaInstance，保证 AgentLoop、QQ 适配层等外部代码无需改动。

同时保留 ChatMode、ConversationTurn、MemoryManager 这些被外部 import 的公共类型。
"""

import json
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, TYPE_CHECKING

import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.logger import logger

from configs.model_config import (
    PersonalityConfig,
    EmotionPromptConfig,
    EmotionFusionConfig,
    EmotionStateConfig,
    AttentionConfig,
    LLMConfig,
    MemoryConfig,
)
from configs.config_loader import AppConfig

# LLMClient 作为运行时 re-export（add_memory.py / memory_manager.py 从此模块 import）
from src.llm.client import LLMClient  
if TYPE_CHECKING:
    from src.attention.proactive_decision import ProactiveDecision


# ── 公共类型（被外部 import，必须保留在此模块）────────────────────────────────

class ChatMode(Enum):
    """对话模式：决定 LLM 路由策略"""
    PRIVATE = "private"
    GROUP = "group"


@dataclass
class ConversationTurn:
    """单轮对话"""
    user_input: str
    emotion: str
    intensity: float
    behavior: str
    tone: str
    response: str
    timestamp: Optional[str] = None
    context_id: Optional[str] = None


class MemoryManager:
    """简单的记忆管理器（fallback，无 Qdrant 时使用）"""

    def __init__(self, max_short_term: int = 10):
        self.short_term: List[ConversationTurn] = []
        self.max_short_term = max_short_term

    def add(self, turn: ConversationTurn):
        self.short_term.append(turn)
        if len(self.short_term) > self.max_short_term:
            self.short_term.pop(0)

    def get_context(self, num_turns: int = 5) -> List[ConversationTurn]:
        return self.short_term[-num_turns:]

    def format_context(self, num_turns: int = 5) -> str:
        context = self.get_context(num_turns)
        if not context:
            return ""
        lines = []
        for turn in context:
            lines.append(f"用户: {turn.user_input}")
            lines.append(f"助手: {turn.response}")
        return "\n".join(lines)


# ── 瘦包装层 ─────────────────────────────────────────────────────────────────

class NeuroLikePipeline:
    """
    向后兼容的 Pipeline 包装层。

    内部由 SharedInfra + PersonaInstance 组成，所有方法和属性
    委托到 PersonaInstance，外部代码（AgentLoop 等）无需改动。
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
        # 向后兼容的参数
        llm_provider: Optional[str] = None,
        llm_api_key: Optional[str] = None,
        llm_model: Optional[str] = None,
        llm_base_url: Optional[str] = None,
        device: str = "cpu",
        time_awareness: bool = True,
    ):
        from src.core.shared_infra import SharedInfra, BERTInferenceEngine, LLMClientPool
        from src.core.persona import PersonaInstance
        from configs.model_config import LLMProvider as _LLMProvider

        # ── 构建 SharedInfra ──────────────────────────────────────────
        bert = BERTInferenceEngine(
            checkpoint_path=small_model_path,
            device=device,
        )

        # 兼容旧式参数（llm_provider / llm_api_key / ...）
        if llm_config is None and llm_provider is not None:
            from src.llm.client import LLMClient as _LC
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
        from src.core.emotion_fusion import LLMEmotionClassifier, EmotionNeuronFusion
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
            device=app_cfg.device,
            time_awareness=app_cfg.agent.time_awareness,
        )

    # ── 委托属性（AgentLoop 需要访问）─────────────────────────────────────

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

    def build_system_prompt(self, recalled_context: str = "",
                            emotion_analysis: Optional[Dict] = None) -> str:
        hint = None
        if self._persona.emotion_state_tracker:
            hint = self._persona.emotion_state_tracker.get_prompt_hint()
        return self._persona.prompt_builder.build_system_prompt(
            recalled_context, emotion_analysis, hint
        )

    def build_system_prompt_blocks(self, recalled_context: str = "",
                                   emotion_analysis: Optional[Dict] = None) -> List[Dict]:
        hint = None
        if self._persona.emotion_state_tracker:
            hint = self._persona.emotion_state_tracker.get_prompt_hint()
        return self._persona.prompt_builder.build_system_prompt_blocks(
            recalled_context, emotion_analysis, hint
        )

    def save_conversation(self, output_path: str):
        self._persona.save_conversation(output_path)

    def close(self):
        self._persona.close()

    def _handle_group_passive_message(self, user_input: str,
                                       context_id: Optional[str] = None) -> Dict:
        return self._persona._handle_group_passive_message(user_input, context_id)


# ── 交互式对话入口 ────────────────────────────────────────────────────────────

def interactive_chat(pipeline: NeuroLikePipeline):
    """交互式对话"""
    print("\n" + "=" * 50)
    print(f"欢迎与 {pipeline.personality.name} 对话！")
    print(f"模型: {pipeline.llm_client.model}")
    print("输入 'quit' 退出，'save' 保存对话历史")
    print("=" * 50 + "\n")

    while True:
        try:
            user_input = input("你: ").strip()

            if not user_input:
                continue

            if user_input.lower() == "quit":
                pipeline.close()
                print("再见！")
                break

            if user_input.lower() == "save":
                pipeline.save_conversation("conversation_history.json")
                continue

            result = pipeline.chat(user_input, verbose=True)
            print(f"\n{pipeline.personality.name}: {result['response']}\n")

        except KeyboardInterrupt:
            print("\n\n再见！")
            break
        except Exception as e:
            print(f"\n错误: {e}\n")


# ============== 入口 ==============
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Neuro-Like对话系统")
    parser.add_argument("--config", type=str, default=None,
                        help="config.json 路径（默认：项目根目录下的 config.json）")
    args = parser.parse_args()

    pipeline = NeuroLikePipeline.from_config(args.config)
    interactive_chat(pipeline)
