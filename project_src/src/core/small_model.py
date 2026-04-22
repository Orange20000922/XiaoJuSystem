"""已过时的小模型推理入口。"""

from src._deprecation import warn_deprecated_path
from src.core_engine.small_model import *  # noqa: F401,F403


warn_deprecated_path("src.core.small_model", "src.core_engine.small_model")
