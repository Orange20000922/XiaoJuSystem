"""HTTP 服务入口。"""

from src.server.http.openai_adapter import create_app
from src.server.http.run_api_server import main

__all__ = ["create_app", "main"]
