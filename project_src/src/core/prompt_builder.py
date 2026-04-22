"""已过时的提示词构建入口。"""

from src._deprecation import warn_deprecated_path
from src.core_engine.prompt_builder import *  # noqa: F401,F403


warn_deprecated_path("src.core.prompt_builder", "src.core_engine.prompt_builder")
