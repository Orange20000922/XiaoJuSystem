"""已过时的人格实现入口。"""

from src._deprecation import warn_deprecated_path
from src.core_engine.persona import *  # noqa: F401,F403


warn_deprecated_path("src.core.persona", "src.core_engine.persona")
