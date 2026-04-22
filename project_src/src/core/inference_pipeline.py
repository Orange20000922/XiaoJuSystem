"""
已过时的推理管线兼容入口。

请改用：
  - src.core_engine.pipeline.NeuroLikePipeline
  - src.core_engine.runtime_types.ChatMode / ConversationTurn / MemoryManager
  - src.client.cli.inference_pipeline.interactive_chat
"""

from __future__ import annotations

from src._deprecation import warn_deprecated_path
from src.client.cli.inference_pipeline import interactive_chat
from src.core_engine.pipeline import NeuroLikePipeline
from src.core_engine.runtime_types import ChatMode, ConversationTurn, MemoryManager
from src.llm.client import LLMClient


warn_deprecated_path("src.core.inference_pipeline", "src.core_engine")


__all__ = [
    "ChatMode",
    "ConversationTurn",
    "LLMClient",
    "MemoryManager",
    "NeuroLikePipeline",
    "interactive_chat",
]


if __name__ == "__main__":
    from src.client.cli.inference_pipeline import main

    main()
