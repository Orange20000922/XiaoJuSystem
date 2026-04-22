"""核心推理引擎旧入口的懒加载兼容层。"""

from __future__ import annotations

import importlib
from typing import Dict, Tuple

from src._deprecation import warn_deprecated_path


_EXPORTS: Dict[str, Tuple[str, str]] = {
    "NeuroLikePipeline": ("src.core_engine.pipeline", "NeuroLikePipeline"),
    "ChatMode": ("src.core_engine.runtime_types", "ChatMode"),
    "ConversationTurn": ("src.core_engine.runtime_types", "ConversationTurn"),
    "MemoryManager": ("src.core_engine.runtime_types", "MemoryManager"),
    "LLMClient": ("src.llm.client", "LLMClient"),
    "interactive_chat": ("src.client.cli.inference_pipeline", "interactive_chat"),
    "SharedInfra": ("src.core_engine.shared_infra", "SharedInfra"),
    "BERTInferenceEngine": ("src.core_engine.small_model", "BERTInferenceEngine"),
    "LLMClientPool": ("src.core_engine.shared_infra", "LLMClientPool"),
    "PersonaInstance": ("src.core_engine.persona", "PersonaInstance"),
    "PromptBuilder": ("src.core_engine.prompt_builder", "PromptBuilder"),
    "LLMEmotionClassifier": ("src.core_engine.emotion_fusion", "LLMEmotionClassifier"),
    "EmotionNeuronFusion": ("src.core_engine.emotion_fusion", "EmotionNeuronFusion"),
    "EmotionStateTracker": ("src.core_engine.emotion_state", "EmotionStateTracker"),
    "EmotionState": ("src.core_engine.emotion_state", "EmotionState"),
    "EmotionStateConfig": ("src.core_engine.emotion_state", "EmotionStateConfig"),
    "PersonaRegistry": ("src.core_engine.registry", "PersonaRegistry"),
    "PersonaScheduler": ("src.core_engine.scheduler", "PersonaScheduler"),
    "ManagedPersona": ("src.core_engine.scheduler", "ManagedPersona"),
}

_DEPRECATED_EXPORTS = {
    "NeuroLikePipeline": "src.core_engine.NeuroLikePipeline",
    "ChatMode": "src.core_engine.ChatMode",
    "ConversationTurn": "src.core_engine.ConversationTurn",
    "MemoryManager": "src.core_engine.MemoryManager",
    "interactive_chat": "src.client.cli.inference_pipeline.interactive_chat",
    "SharedInfra": "src.core_engine.SharedInfra",
    "BERTInferenceEngine": "src.core_engine.BERTInferenceEngine",
    "LLMClientPool": "src.core_engine.LLMClientPool",
    "PersonaInstance": "src.core_engine.PersonaInstance",
    "PromptBuilder": "src.core_engine.PromptBuilder",
    "LLMEmotionClassifier": "src.core_engine.LLMEmotionClassifier",
    "EmotionNeuronFusion": "src.core_engine.EmotionNeuronFusion",
    "EmotionStateTracker": "src.core_engine.EmotionStateTracker",
    "EmotionState": "src.core_engine.EmotionState",
    "EmotionStateConfig": "src.core_engine.EmotionStateConfig",
    "PersonaRegistry": "src.core_engine.PersonaRegistry",
    "PersonaScheduler": "src.core_engine.PersonaScheduler",
    "ManagedPersona": "src.core_engine.ManagedPersona",
}


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module 'src.core' has no attribute {name!r}")
    if name in _DEPRECATED_EXPORTS:
        warn_deprecated_path(f"src.core.{name}", _DEPRECATED_EXPORTS[name])
    module_name, attr_name = _EXPORTS[name]
    module = importlib.import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(list(globals().keys()) + list(_EXPORTS.keys()))


__all__ = sorted(_EXPORTS.keys())
