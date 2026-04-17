"""
src 模块向后兼容层

为了不破坏现有代码，从新的子模块 re-export 常用类和函数。
这样旧的 import 路径（如 `from src.inference_pipeline import ...`）仍然可以工作。
"""

# ── 核心模块 re-exports ──────────────────────────────────────────────────────
from src.core.inference_pipeline import (
    NeuroLikePipeline,
    ChatMode,
    ConversationTurn,
    MemoryManager,
    LLMClient,
    interactive_chat,
)

from src.core.shared_infra import (
    SharedInfra,
    BERTInferenceEngine,
    LLMClientPool,
)

from src.core.persona import PersonaInstance

from src.core.prompt_builder import PromptBuilder

from src.core.emotion_fusion import (
    LLMEmotionClassifier,
    EmotionNeuronFusion,
)

from src.core.emotion_state import (
    EmotionStateTracker,
    EmotionState,
)

# ── 调度器 re-exports ──────────────────────────────────────────────────────
from src.core.scheduler import PersonaScheduler, ManagedPersona

# ── 记忆系统 re-exports ──────────────────────────────────────────────────────
from src.memory.memory_manager import HierarchicalMemoryManager

# ── 注意力系统 re-exports ────────────────────────────────────────────────────
from src.attention.attention_tracker import AttentionTracker
from src.attention.proactive_decision import ProactiveDecisionModule

# ── Agent 层 re-exports ──────────────────────────────────────────────────────
from src.agent.agent_loop import AgentLoop, AgentEvent, ProactiveState

# ── 适配器层 re-exports ──────────────────────────────────────────────────────
from src.adapters.qq_adapter import QQBotAdapter

# ── 多媒体 re-exports ────────────────────────────────────────────────────────
from src.media.image_utils import (
    process_image_url,
    ImageResult,
    PILLOW_AVAILABLE,
)

from src.vision import (
    CV2_AVAILABLE,
    VisualPerceptionConfig,
    VisualPerceptionPipeline,
    VisualEvent,
    VisualAnalysis,
    VisualEventCluster,
    VisualWindowSummary,
    VisualEventMonitor,
    build_visual_analyzer_from_pipeline,
    cluster_visual_events,
    derive_visual_emotion_signal,
    push_visual_event_to_agent_loop,
    summarize_visual_event_windows,
    visual_event_memory_text,
    visual_event_to_agent_event,
)

# ── LLM 客户端 re-exports ────────────────────────────────────────────────────
from src.llm.client import LLMClient as _LLMClient  # 避免重复定义

# ── 日志 re-export ───────────────────────────────────────────────────────────
from src.logger import logger


__all__ = [
    # Core
    "NeuroLikePipeline",
    "ChatMode",
    "ConversationTurn",
    "MemoryManager",
    "LLMClient",
    "interactive_chat",
    "SharedInfra",
    "BERTInferenceEngine",
    "LLMClientPool",
    "PersonaInstance",
    "PromptBuilder",
    "LLMEmotionClassifier",
    "EmotionNeuronFusion",
    "EmotionStateTracker",
    "EmotionState",
    # Scheduler
    "PersonaScheduler",
    "ManagedPersona",
    # Memory
    "HierarchicalMemoryManager",
    # Attention
    "AttentionTracker",
    "ProactiveDecisionModule",
    # Agent
    "AgentLoop",
    "AgentEvent",
    "ProactiveState",
    # Adapters
    "QQBotAdapter",
    # Media
    "process_image_url",
    "ImageResult",
    "PILLOW_AVAILABLE",
    "CV2_AVAILABLE",
    "VisualPerceptionConfig",
    "VisualPerceptionPipeline",
    "VisualEvent",
    "VisualAnalysis",
    "VisualEventCluster",
    "VisualWindowSummary",
    "VisualEventMonitor",
    "build_visual_analyzer_from_pipeline",
    "cluster_visual_events",
    "derive_visual_emotion_signal",
    "push_visual_event_to_agent_loop",
    "summarize_visual_event_windows",
    "visual_event_memory_text",
    "visual_event_to_agent_event",
    # Logger
    "logger",
]
