"""已过时的 QQ 启动入口。"""

from src._deprecation import warn_deprecated_path
from src.server.qq.run_qq_bot import *  # noqa: F401,F403


warn_deprecated_path("src.adapters.run_qq_bot", "src.server.qq.run_qq_bot")
