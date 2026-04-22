"""已过时的情绪融合入口。"""

from src._deprecation import warn_deprecated_path
from src.core_engine.emotion_fusion import *  # noqa: F401,F403


warn_deprecated_path("src.core.emotion_fusion", "src.core_engine.emotion_fusion")
