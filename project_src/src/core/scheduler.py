"""已过时的人格调度器入口。"""

from src._deprecation import warn_deprecated_path
from src.core_engine.scheduler import *  # noqa: F401,F403


warn_deprecated_path("src.core.scheduler", "src.core_engine.scheduler")
