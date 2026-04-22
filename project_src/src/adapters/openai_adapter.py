"""已过时的 OpenAI 兼容 API 适配器入口。"""

from src._deprecation import warn_deprecated_path
from src.server.http.openai_adapter import *  # noqa: F401,F403


warn_deprecated_path("src.adapters.openai_adapter", "src.server.http.openai_adapter")
