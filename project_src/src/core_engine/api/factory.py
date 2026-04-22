"""客户端/服务端启动阶段使用的工厂函数。"""

from __future__ import annotations

from typing import Optional

from src.core_engine.api.direct_runtime import DirectRuntime
from src.core_engine.api.event_runtime import EventRuntime
from src.core_engine.api.multi_runtime import MultiRuntime


def create_direct_runtime(config_path: Optional[str] = None) -> DirectRuntime:
    return DirectRuntime.from_config(config_path)


def wrap_agent_loop(loop) -> EventRuntime:
    return EventRuntime(loop)


def wrap_scheduler(scheduler) -> MultiRuntime:
    return MultiRuntime(scheduler)
