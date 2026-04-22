"""单人格直连运行时封装。"""

from __future__ import annotations

from typing import Any, Optional

from src.core_engine.api.admin_api import (
    build_engine_status,
    clear_context_memory,
    get_llm_model,
    get_persona_name,
    snapshot_recent_turns,
)
from src.core_engine.api.contracts import ChatRequest, ChatResponse, ClearContextResult
from src.core_engine.runtime_types import ChatMode


class DirectRuntime:
    """面向同步对话调用的单人格封装。"""

    def __init__(self, target: Any):
        self._target = target

    @classmethod
    def from_config(cls, config_path: Optional[str] = None) -> "DirectRuntime":
        from src.core_engine.pipeline import NeuroLikePipeline

        return cls(NeuroLikePipeline.from_config(config_path))

    @property
    def target(self) -> Any:
        return self._target

    @property
    def persona_name(self) -> str:
        return get_persona_name(self._target)

    @property
    def llm_model(self) -> Optional[str]:
        return get_llm_model(self._target)

    def chat(self, request: ChatRequest) -> ChatResponse:
        chat_mode = ChatMode.GROUP if request.mode == "group" else ChatMode.PRIVATE
        result = self._target.chat(
            request.text,
            verbose=request.verbose,
            is_mentioned=request.is_mentioned,
            chat_mode=chat_mode,
            use_fusion=request.use_fusion,
            images=request.images or None,
            user_id=request.user_id,
            user_name=request.user_name,
            context_id=request.context_id,
            visual_direct=request.visual_direct,
            **request.metadata,
        )
        return ChatResponse(
            reply=result.get("response"),
            should_respond=bool(result.get("should_respond")),
            emotion=result.get("emotion") or {},
            behavior=result.get("behavior") or {},
            debug=result.get("_debug") or {},
            raw=result,
        )

    def get_status(self):
        return build_engine_status(self._target)

    def recent_turns(self, *, limit: int = 8, context_id: Optional[str] = None):
        return snapshot_recent_turns(self._target, limit=limit, context_id=context_id)

    def clear_context(self, context_id: Optional[str] = None) -> ClearContextResult:
        cleared = clear_context_memory(self._target, context_id=context_id)
        return ClearContextResult(
            context_id=context_id,
            cleared_turns=cleared,
            persona_name=self.persona_name,
        )

    def flush_memory(self) -> None:
        memory = getattr(self._target, "memory", None)
        if memory is not None and hasattr(memory, "close_session"):
            memory.close_session()

    def save_conversation(self, output_path: str) -> None:
        if hasattr(self._target, "save_conversation"):
            self._target.save_conversation(output_path)

    def shutdown(self) -> None:
        if hasattr(self._target, "close"):
            self._target.close()
