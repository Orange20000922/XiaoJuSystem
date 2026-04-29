"""
src 模块向后兼容层。

"""

from __future__ import annotations

import importlib
from typing import Dict, Tuple


_EXPORTS: Dict[str, Tuple[str, str]] = {
    # 核心模块
    "NeuroLikePipeline": ("src.core_engine", "NeuroLikePipeline"),
    "ChatMode": ("src.core_engine", "ChatMode"),
    "ConversationTurn": ("src.core_engine", "ConversationTurn"),
    "MemoryManager": ("src.core_engine", "MemoryManager"),
    "LLMClient": ("src.llm.client", "LLMClient"),
    "interactive_chat": ("src.client.cli.inference_pipeline", "interactive_chat"),
    "SharedInfra": ("src.core_engine", "SharedInfra"),
    "BERTInferenceEngine": ("src.core_engine", "BERTInferenceEngine"),
    "LLMClientPool": ("src.core_engine", "LLMClientPool"),
    "PersonaInstance": ("src.core_engine", "PersonaInstance"),
    "PromptBuilder": ("src.core_engine", "PromptBuilder"),
    "LLMEmotionClassifier": ("src.core_engine", "LLMEmotionClassifier"),
    "EmotionNeuronFusion": ("src.core_engine", "EmotionNeuronFusion"),
    "EmotionStateTracker": ("src.core_engine", "EmotionStateTracker"),
    "EmotionState": ("src.core_engine", "EmotionState"),
    "EmotionStateConfig": ("src.core_engine", "EmotionStateConfig"),
    # 调度模块
    "PersonaScheduler": ("src.core_engine", "PersonaScheduler"),
    "ManagedPersona": ("src.core_engine", "ManagedPersona"),
    # 记忆模块
    "HierarchicalMemoryManager": ("src.memory.memory_manager", "HierarchicalMemoryManager"),
    # 注意力模块
    "AttentionTracker": ("src.attention.attention_tracker", "AttentionTracker"),
    "ProactiveDecisionModule": ("src.attention.proactive_decision", "ProactiveDecisionModule"),
    # Agent 模块
    "AgentLoop": ("src.agent.agent_loop", "AgentLoop"),
    "AgentEvent": ("src.agent.agent_loop", "AgentEvent"),
    "ProactiveState": ("src.agent.agent_loop", "ProactiveState"),
    # 适配器模块
    "QQBotAdapter": ("src.server.qq", "QQBotAdapter"),
    # 核心引擎统一接口
    "ChatRequest": ("src.core_engine.api", "ChatRequest"),
    "ChatResponse": ("src.core_engine.api", "ChatResponse"),
    "ClearContextResult": ("src.core_engine.api", "ClearContextResult"),
    "DirectRuntime": ("src.core_engine.api", "DirectRuntime"),
    "EngineEvent": ("src.core_engine.api", "EngineEvent"),
    "EngineStatus": ("src.core_engine.api", "EngineStatus"),
    "EventRuntime": ("src.core_engine.api", "EventRuntime"),
    "MultiRuntime": ("src.core_engine.api", "MultiRuntime"),
    # 多媒体模块
    "process_image_url": ("src.media.image_utils", "process_image_url"),
    "ImageResult": ("src.media.image_utils", "ImageResult"),
    "PILLOW_AVAILABLE": ("src.media.image_utils", "PILLOW_AVAILABLE"),
    # 视觉模块
    "CV2_AVAILABLE": ("src.vision", "CV2_AVAILABLE"),
    "VisualPerceptionConfig": ("src.vision", "VisualPerceptionConfig"),
    "VisualPerceptionPipeline": ("src.vision", "VisualPerceptionPipeline"),
    "VisualEvent": ("src.vision", "VisualEvent"),
    "VisualAnalysis": ("src.vision", "VisualAnalysis"),
    "VisualEventCluster": ("src.vision", "VisualEventCluster"),
    "VisualWindowSummary": ("src.vision", "VisualWindowSummary"),
    "VisualEventMonitor": ("src.vision", "VisualEventMonitor"),
    "build_visual_analyzer_from_pipeline": ("src.vision", "build_visual_analyzer_from_pipeline"),
    "cluster_visual_events": ("src.vision", "cluster_visual_events"),
    "derive_visual_emotion_signal": ("src.vision", "derive_visual_emotion_signal"),
    "push_visual_event_to_agent_loop": ("src.vision", "push_visual_event_to_agent_loop"),
    "summarize_visual_event_windows": ("src.vision", "summarize_visual_event_windows"),
    "visual_event_memory_text": ("src.vision", "visual_event_memory_text"),
    "visual_event_to_agent_event": ("src.vision", "visual_event_to_agent_event"),
    # 日志模块
    "logger": ("src.logger", "logger"),
}


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module 'src' has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    module = importlib.import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(list(globals().keys()) + list(_EXPORTS.keys()))


__all__ = sorted(_EXPORTS.keys())
