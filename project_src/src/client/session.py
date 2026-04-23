"""Client-side session helpers built on top of core_engine APIs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

from src.core_engine.api import ChatRequest, ChatResponse, ClearContextResult, DirectRuntime


def coerce_runtime(runtime_or_target: Any) -> DirectRuntime:
    """Return a DirectRuntime regardless of the caller's input shape."""

    if isinstance(runtime_or_target, DirectRuntime):
        return runtime_or_target
    return DirectRuntime(runtime_or_target)


@dataclass
class ClientChatSession:
    """Thin client-facing chat session wrapper."""

    runtime: DirectRuntime
    context_id: str = "client_default"
    mode: str = "private"
    verbose: bool = True
    default_user_id: Optional[int] = None
    default_user_name: Optional[str] = None

    @classmethod
    def from_target(
        cls,
        runtime_or_target: Any,
        *,
        context_id: str = "client_default",
        mode: str = "private",
        verbose: bool = True,
        default_user_id: Optional[int] = None,
        default_user_name: Optional[str] = None,
    ) -> "ClientChatSession":
        return cls(
            runtime=coerce_runtime(runtime_or_target),
            context_id=context_id,
            mode=mode,
            verbose=verbose,
            default_user_id=default_user_id,
            default_user_name=default_user_name,
        )

    @property
    def persona_name(self) -> str:
        return self.runtime.persona_name

    @property
    def llm_model(self) -> Optional[str]:
        return self.runtime.llm_model

    def send_message(
        self,
        text: str,
        *,
        user_id: Optional[int] = None,
        user_name: Optional[str] = None,
        is_mentioned: bool = True,
        images: Optional[Sequence[Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        use_fusion: Optional[bool] = None,
        visual_direct: bool = False,
    ) -> ChatResponse:
        request = ChatRequest(
            text=text,
            context_id=self.context_id,
            mode=self.mode,
            user_id=self.default_user_id if user_id is None else user_id,
            user_name=self.default_user_name if user_name is None else user_name,
            is_mentioned=is_mentioned,
            images=list(images or []),
            metadata=dict(metadata or {}),
            verbose=self.verbose,
            use_fusion=use_fusion,
            visual_direct=visual_direct,
        )
        return self.runtime.chat(request)

    def get_status(self):
        return self.runtime.get_status()

    def clear_context(self) -> ClearContextResult:
        return self.runtime.clear_context(self.context_id)

    def save_conversation(self, output_path: str) -> None:
        self.runtime.save_conversation(output_path)

    def shutdown(self) -> None:
        self.runtime.shutdown()
