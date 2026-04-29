"""ClientChatSession 是一个面向客户端的聊天会话包装器，提供了一个简化的接口来与核心引擎进行交互。它负责管理会话状态、发送消息、获取状态、清理上下文以及保存对话记录等功能。该类设计为线程安全，以支持在多线程环境中使用，并确保在会话关闭时正确处理正在进行的消息发送操作。"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence

from src.core_engine.api import ChatRequest, ChatResponse, ClearContextResult, DirectRuntime


def coerce_runtime(runtime_or_target: Any) -> DirectRuntime:
    """将输入的 runtime_or_target 转换为 DirectRuntime 实例。"""

    if isinstance(runtime_or_target, DirectRuntime):
        return runtime_or_target
    return DirectRuntime(runtime_or_target)


@dataclass
class ClientChatSession:

    runtime: DirectRuntime
    context_id: str = "client_default"
    mode: str = "private"
    verbose: bool = True
    default_user_id: Optional[int] = None
    default_user_name: Optional[str] = None
    _send_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _state_condition: threading.Condition = field(
        default_factory=lambda: threading.Condition(threading.Lock()),
        init=False,
        repr=False,
    )
    _closed: bool = field(default=False, init=False, repr=False)
    _active_sends: int = field(default=0, init=False, repr=False)

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
        with self._state_condition:
            if self._closed:
                raise RuntimeError("ClientChatSession is closed")
            self._active_sends += 1

        try:
            with self._send_lock:
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
        finally:
            with self._state_condition:
                self._active_sends -= 1
                if self._active_sends == 0:
                    self._state_condition.notify_all()

    def get_status(self):
        return self.runtime.get_status()

    def clear_context(self) -> ClearContextResult:
        return self.runtime.clear_context(self.context_id)

    def save_conversation(self, output_path: str) -> None:
        self.runtime.save_conversation(output_path)

    def shutdown(self) -> None:
        should_shutdown = False
        with self._state_condition:
            if not self._closed:
                self._closed = True
                should_shutdown = True
            while self._active_sends > 0:
                self._state_condition.wait()

        if should_shutdown:
            self.runtime.shutdown()
