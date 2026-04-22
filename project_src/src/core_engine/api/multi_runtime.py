"""多人格调度运行时封装。"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from src.core_engine.api.admin_api import build_engine_status
from src.core_engine.api.contracts import ClearContextResult, EngineEvent
from src.core_engine.api.event_runtime import EventRuntime
from src.core_engine.runtime_types import ChatMode


class MultiRuntime:
    """面向传输层的 PersonaScheduler 封装。"""

    def __init__(self, scheduler: Any):
        self._scheduler = scheduler

    @property
    def scheduler(self) -> Any:
        return self._scheduler

    def list_personas(self) -> List[str]:
        return list(self._scheduler.persona_names)

    def get_runtime(self, name: str) -> Optional[EventRuntime]:
        managed = self._scheduler.get_persona(name)
        if managed is None:
            return None
        return EventRuntime(managed.agent_loop)

    def resolve_runtime(self, context_id: str) -> Optional[EventRuntime]:
        managed = self._scheduler.resolve_persona(context_id)
        if managed is None:
            return None
        return EventRuntime(managed.agent_loop)

    def push(self, event: Any) -> bool:
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
        return bool(self._scheduler.dispatch(event))

    def clear_context(self, context_id: str) -> ClearContextResult:
        managed = self._scheduler.resolve_persona(context_id)
        if managed is None:
            return ClearContextResult(
                context_id=context_id,
                cleared_turns=0,
                persona_name=None,
            )
        runtime = EventRuntime(managed.agent_loop)
        result = runtime.clear_context(context_id)
        result.persona_name = managed.name
        return result

    def get_health_report(self) -> Dict[str, Any]:
        report: Dict[str, Any] = {}
        now = time.time()
        for name in self.list_personas():
            managed = self._scheduler.get_persona(name)
            if managed is None:
                continue
            uptime_seconds = int(now - managed.started_at) if managed.started_at else 0
            report[name] = build_engine_status(
                managed.persona,
                loop=managed.agent_loop,
                contexts=sorted(managed.context_ids),
                patterns=list(managed.context_patterns),
                config_source=managed.config_source,
                uptime_seconds=uptime_seconds,
            )
        return report

    def shutdown(self) -> None:
        self._scheduler.stop_all()
