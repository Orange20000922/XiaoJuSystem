"""已过时的共享基础设施入口。"""

from src._deprecation import warn_deprecated_path
from src.core_engine.shared_infra import *  # noqa: F401,F403


warn_deprecated_path("src.core.shared_infra", "src.core_engine.shared_infra")
