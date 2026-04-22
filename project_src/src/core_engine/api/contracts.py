"""核心引擎边界契约。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional


ChatModeValue = Literal["private", "group"]
EventTypeValue = Literal["message", "system", "visual"]


@dataclass
class ChatRequest:
    """核心引擎接受的标准化对话输入。"""

    text: str
    context_id: str = "default"
    mode: ChatModeValue = "private"
    user_id: Optional[int] = None
    user_name: Optional[str] = None
    is_mentioned: bool = True
    images: List[Any] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    verbose: bool = False
    use_fusion: Optional[bool] = None
    visual_direct: bool = False


@dataclass
class ChatResponse:
    """核心引擎返回的标准化对话输出。"""

    reply: Optional[str]
    should_respond: bool
    emotion: Dict[str, Any] = field(default_factory=dict)
    behavior: Dict[str, Any] = field(default_factory=dict)
    debug: Dict[str, Any] = field(default_factory=dict)
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EngineEvent:
    """与协议无关的标准化事件。"""

    type: EventTypeValue
    content: str
    context_id: str = "default"
    chat_mode: ChatModeValue = "private"
    is_mentioned: bool = True
    reply_context: Dict[str, Any] = field(default_factory=dict)
    images: List[Any] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    user_id: Optional[int] = None
    user_name: Optional[str] = None


@dataclass
class OutboundMessage:
    """发往客户端/服务端传输层的标准化输出。"""

    context_id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EngineStatus:
    """核心引擎对外暴露的运行状态快照。"""

    persona_name: str
    llm_model: Optional[str] = None
    running: bool = False
    proactive_state: Optional[str] = None
    idle_seconds: Optional[int] = None
    queue_size: Optional[int] = None
    uptime_seconds: Optional[int] = None
    working_memory_turns: int = 0
    attention: Dict[str, Any] = field(default_factory=dict)
    emotion_state: Dict[str, Any] = field(default_factory=dict)
    contexts: List[str] = field(default_factory=list)
    patterns: List[str] = field(default_factory=list)
    config_source: str = ""


@dataclass
class ClearContextResult:
    """清理某个上下文工作记忆后的结果。"""

    context_id: Optional[str]
    cleared_turns: int
    persona_name: Optional[str] = None
