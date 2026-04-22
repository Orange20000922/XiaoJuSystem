"""已过时的人格注册表入口。"""

from src._deprecation import warn_deprecated_path
from src.core_engine.registry import *  # noqa: F401,F403


warn_deprecated_path("src.core.registry", "src.core_engine.registry")
