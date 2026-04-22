"""已过时的人格资源清理入口。"""

from src._deprecation import warn_deprecated_path
from src.core_engine.persona_resource_cleanup import *  # noqa: F401,F403


warn_deprecated_path(
    "src.core.persona_resource_cleanup",
    "src.core_engine.persona_resource_cleanup",
)
