"""已过时的 QQ 协议适配器入口。"""

from src._deprecation import warn_deprecated_path
from src.server.qq.qq_adapter import *  # noqa: F401,F403


warn_deprecated_path("src.adapters.qq_adapter", "src.server.qq.qq_adapter")
