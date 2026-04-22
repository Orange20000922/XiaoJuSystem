"""核心引擎公共导出层。"""

from __future__ import annotations

import importlib
from typing import Dict, Tuple


_EXPORTS: Dict[str, Tuple[str, str]] = {
    "BERTInferenceEngine": ("src.core_engine.small_model", "BERTInferenceEngine"),
    "ChatMode": ("src.core_engine.runtime_types", "ChatMode"),
    "ChatRequest": ("src.core_engine.api", "ChatRequest"),
    "ChatResponse": ("src.core_engine.api", "ChatResponse"),
    "ConversationTurn": ("src.core_engine.runtime_types", "ConversationTurn"),
    "ClearContextResult": ("src.core_engine.api", "ClearContextResult"),
    "DirectRuntime": ("src.core_engine.api", "DirectRuntime"),
    "EmotionNeuronFusion": ("src.core_engine.emotion_fusion", "EmotionNeuronFusion"),
    "EmotionState": ("src.core_engine.emotion_state", "EmotionState"),
    "EmotionStateConfig": ("src.core_engine.emotion_state", "EmotionStateConfig"),
    "EmotionStateTracker": ("src.core_engine.emotion_state", "EmotionStateTracker"),
    "EngineEvent": ("src.core_engine.api", "EngineEvent"),
    "EngineStatus": ("src.core_engine.api", "EngineStatus"),
    "EventRuntime": ("src.core_engine.api", "EventRuntime"),
    "LLMClientPool": ("src.core_engine.shared_infra", "LLMClientPool"),
    "LLMEmotionClassifier": ("src.core_engine.emotion_fusion", "LLMEmotionClassifier"),
    "MemoryManager": ("src.core_engine.runtime_types", "MemoryManager"),
    "ManagedPersona": ("src.core_engine.scheduler", "ManagedPersona"),
    "MultiRuntime": ("src.core_engine.api", "MultiRuntime"),
    "NeuroLikePipeline": ("src.core_engine.pipeline", "NeuroLikePipeline"),
    "OutboundMessage": ("src.core_engine.api", "OutboundMessage"),
    "PipelineLike": ("src.core_engine.runtime_types", "PipelineLike"),
    "PersonaInstance": ("src.core_engine.persona", "PersonaInstance"),
    "PersonaRegistry": ("src.core_engine.registry", "PersonaRegistry"),
    "PersonaScheduler": ("src.core_engine.scheduler", "PersonaScheduler"),
    "PromptBuilder": ("src.core_engine.prompt_builder", "PromptBuilder"),
    "SharedInfra": ("src.core_engine.shared_infra", "SharedInfra"),
    "create_direct_runtime": ("src.core_engine.api", "create_direct_runtime"),
    "wrap_agent_loop": ("src.core_engine.api", "wrap_agent_loop"),
    "wrap_scheduler": ("src.core_engine.api", "wrap_scheduler"),
}


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module 'src.core_engine' has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    module = importlib.import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(list(globals().keys()) + list(_EXPORTS.keys()))


__all__ = sorted(_EXPORTS.keys())
