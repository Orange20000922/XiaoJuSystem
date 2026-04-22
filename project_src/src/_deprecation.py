"""兼容层过时提示工具。"""

from __future__ import annotations

import warnings
from typing import Set, Tuple


_WARNED: Set[Tuple[str, str, str]] = set()


def warn_deprecated_path(old_path: str, new_path: str, *, removal: str = "后续版本") -> None:
    """按旧路径发出一次性过时提示。"""
    key = (old_path, new_path, removal)
    if key in _WARNED:
        return
    warnings.warn(
        f"`{old_path}` 已过时，请改用 `{new_path}`，兼容入口预计在{removal}删除。",
        DeprecationWarning,
        stacklevel=2,
    )
    _WARNED.add(key)
