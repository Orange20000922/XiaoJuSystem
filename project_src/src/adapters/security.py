"""已过时的 HTTP 安全中间件入口。"""

from src._deprecation import warn_deprecated_path
from src.server.http.security import *  # noqa: F401,F403


warn_deprecated_path("src.adapters.security", "src.server.http.security")
