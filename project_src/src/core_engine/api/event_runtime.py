"""事件循环运行时封装。"""

from __future__ import annotations

from typing import Any, Optional

from src.core_engine.api.admin_api import (
    build_engine_status,
    clear_context_memory,
    get_persona_name,
)
from src.core_engine.api.contracts import ClearContextResult, EngineEvent
from src.core_engine.runtime_types import ChatMode


class EventRuntime:
    """面向传输层事件投递的 AgentLoop 封装。"""

    def __init__(self, loop: Any):
        self._loop = loop

    @property
    def loop(self) -> Any:
        return self._loop

    @property
    def persona_name(self) -> str:
        return get_persona_name(self._loop.pipeline)

    def start(self) -> None:
        self._loop.start()

    def push(self, event: Any) -> None:
        if isinstance(event, EngineEvent):
            from src.agent.agent_loop import AgentEvent

            chat_mode = ChatMode.GROUP if event.chat_mode == "group" else ChatMode.PRIVATE
            event = AgentEvent(
                type=event.type,
                content=event.content,
                chat_mode=chat_mode,
                is_mentioned=event.is_mentioned,
                reply_context=event.reply_context,
                images=event.images,
                metadata=event.metadata,
                user_id=event.user_id,
                user_name=event.user_name,
                context_id=event.context_id,
            )
        self._loop.push(event)

    def stop(self) -> None:
        self._loop.stop()

    def shutdown(self) -> None:
        self.stop()

    def get_status(self):
        return build_engine_status(self._loop.pipeline, loop=self._loop)

    def clear_context(self, context_id: Optional[str] = None) -> ClearContextResult:
        cleared = clear_context_memory(self._loop.pipeline, context_id=context_id)
        return ClearContextResult(
            context_id=context_id,
            cleared_turns=cleared,
            persona_name=self.persona_name,
        )
