"""已过时的情绪状态机入口。"""

from src._deprecation import warn_deprecated_path
from src.core_engine.emotion_state import *  # noqa: F401,F403


warn_deprecated_path("src.core.emotion_state", "src.core_engine.emotion_state")
