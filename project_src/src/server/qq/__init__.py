"""QQ / OneBot 服务入口。"""

from src.server.qq.qq_adapter import QQBotAdapter
from src.server.qq.run_qq_bot import main

__all__ = ["QQBotAdapter", "main"]
