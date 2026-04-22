"""核心引擎的稳定运行时类型。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


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


@runtime_checkable
class PipelineLike(Protocol):
    """核心推理管线的最小协议接口。"""

    personality: Any
    memory: Any
    attention_tracker: Any
    llm_client: Any
    llm_client_secondary: Any
    llm_client_vision: Any
    small_model: Any
    emotion_state_tracker: Any
    emotion_prompt_config: Any
    visual_perception_config: Any
    emotion_fusion_config: Any
    device: str

    def chat(self, user_input: str, **kwargs) -> Dict: ...

    def generate_response(self, user_input: str, **kwargs) -> str: ...

    def generate_proactive(self, trigger: str, **kwargs) -> Optional[str]: ...

    def analyze_emotion_behavior(self, text: str) -> Optional[Dict]: ...

    def should_respond(self, emotion_behavior: Dict, is_mentioned: bool = False) -> bool: ...

    def build_system_prompt(
        self,
        recalled_context: str = "",
        emotion_analysis: Optional[Dict] = None,
    ) -> str: ...

    def build_system_prompt_blocks(
        self,
        recalled_context: str = "",
        emotion_analysis: Optional[Dict] = None,
    ) -> List[Dict]: ...

    def save_conversation(self, output_path: str): ...

    def close(self): ...

    def handle_visual_event(self, event) -> Dict: ...


__all__ = [
    "ChatMode",
    "ConversationTurn",
    "MemoryManager",
    "PipelineLike",
]
