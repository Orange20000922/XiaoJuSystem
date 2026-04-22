"""已过时的 OpenAI 兼容 API 启动入口。"""

from src._deprecation import warn_deprecated_path
from src.server.http.run_api_server import *  # noqa: F401,F403


warn_deprecated_path("src.adapters.run_api_server", "src.server.http.run_api_server")
