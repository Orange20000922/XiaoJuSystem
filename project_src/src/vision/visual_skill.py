from __future__ import annotations

import re
from typing import Callable, List, Optional

from src.logger import logger
from .visual_types import VisualEvent


_VISUAL_REQUEST_PATTERNS: tuple[str, ...] = (
    r"看看",
    r"看一下",
    r"看到",
    r"你看[看到见]",
    r"你能看",
    r"画面",
    r"摄像头",
    r"你的眼",
    r"周围",
    r"在做什么",
    r"在干什么",
    r"你见",
    r"你注意到",
    r"屏幕",
    r"你面前",
    r"我的样子",
)


class VisualSkillDetector:
    """检测用户消息中的视觉请求意图。"""

    def __init__(self, patterns: Optional[tuple[str, ...]] = None):
        raw = patterns or _VISUAL_REQUEST_PATTERNS
        self._compiled = re.compile("|".join(raw))

    def detect(self, text: str) -> bool:
        return self._compiled.search(text) is not None


class VisualSkillExecutor:
    """调用视觉管线获取最近画面分析摘要。"""

    def __init__(
        self,
        handler: Callable[..., object],
    ):
        self._handler = handler

    def execute(self, top_k: int = 2) -> str:
        try:
            result = self._handler(top_k=top_k)
        except Exception as exc:
            logger.warning(f"视觉技能执行失败: {exc}")
            return ""

        if isinstance(result, str):
            return result

        if result is None:
            return "当前画面暂无显著变化"

        events = list(result)
        if not events:
            return "当前画面暂无显著变化"
        parts = [e.summary_text() for e in events if e.summary_text()]
        return "；".join(parts) if parts else "当前画面暂无显著变化"


__all__ = [
    "VisualSkillDetector",
    "VisualSkillExecutor",
]
