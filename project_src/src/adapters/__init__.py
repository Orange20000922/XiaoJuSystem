"""外部平台适配器旧入口的懒加载兼容层。"""

from __future__ import annotations

import importlib

from src._deprecation import warn_deprecated_path


_EXPORTS = {
    "QQBotAdapter": ("src.adapters.qq_adapter", "QQBotAdapter"),
}


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module 'src.adapters' has no attribute {name!r}")
    warn_deprecated_path("src.adapters.QQBotAdapter", "src.server.qq.QQBotAdapter")
    module_name, attr_name = _EXPORTS[name]
    module = importlib.import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(list(globals().keys()) + list(_EXPORTS.keys()))


__all__ = sorted(_EXPORTS.keys())
