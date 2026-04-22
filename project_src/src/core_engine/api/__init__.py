"""核心引擎对外 API。"""

from __future__ import annotations

import importlib
from typing import Dict, Tuple


_EXPORTS: Dict[str, Tuple[str, str]] = {
    "ChatRequest": ("src.core_engine.api.contracts", "ChatRequest"),
    "ChatResponse": ("src.core_engine.api.contracts", "ChatResponse"),
    "ClearContextResult": ("src.core_engine.api.contracts", "ClearContextResult"),
    "EngineEvent": ("src.core_engine.api.contracts", "EngineEvent"),
    "EngineStatus": ("src.core_engine.api.contracts", "EngineStatus"),
    "OutboundMessage": ("src.core_engine.api.contracts", "OutboundMessage"),
    "DirectRuntime": ("src.core_engine.api.direct_runtime", "DirectRuntime"),
    "EventRuntime": ("src.core_engine.api.event_runtime", "EventRuntime"),
    "MultiRuntime": ("src.core_engine.api.multi_runtime", "MultiRuntime"),
    "create_direct_runtime": ("src.core_engine.api.factory", "create_direct_runtime"),
    "wrap_agent_loop": ("src.core_engine.api.factory", "wrap_agent_loop"),
    "wrap_scheduler": ("src.core_engine.api.factory", "wrap_scheduler"),
}


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module 'src.core_engine.api' has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    module = importlib.import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(list(globals().keys()) + list(_EXPORTS.keys()))


__all__ = sorted(_EXPORTS.keys())
