"""
日志模块

使用 loguru 作为后端，同时输出到控制台和滚动文件。
所有模块统一 from src.logger import logger 使用同一个实例。
"""

import sys
from pathlib import Path
from loguru import logger

# 日志目录
_LOG_DIR = Path(__file__).parent.parent / "logs"
_LOG_DIR.mkdir(exist_ok=True)

# 移除 loguru 默认的 stderr handler
logger.remove()

# ── 控制台：INFO 及以上，彩色输出 ────────────────────────────────────
logger.add(
    sys.stderr,
    level="INFO",
    colorize=True,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - <level>{message}</level>",
)

# ── 文件：DEBUG 及以上，按天滚动，保留 30 天 ─────────────────────────
logger.add(
    _LOG_DIR / "neuro_{time:YYYY-MM-DD}.log",
    level="DEBUG",
    rotation="00:00",       # 每天午夜滚动
    retention="30 days",
    compression="zip",
    encoding="utf-8",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
)

# ── 错误单独归档：WARNING 及以上 ─────────────────────────────────────
logger.add(
    _LOG_DIR / "neuro_errors.log",
    level="WARNING",
    rotation="10 MB",
    retention="90 days",
    compression="zip",
    encoding="utf-8",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
)

__all__ = ["logger"]
