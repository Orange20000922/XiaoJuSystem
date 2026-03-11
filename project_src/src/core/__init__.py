"""核心推理引擎模块"""

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

from src.core.scheduler import (
    PersonaScheduler,
    ManagedPersona,
)

__all__ = [
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
    "PersonaScheduler",
    "ManagedPersona",
]
