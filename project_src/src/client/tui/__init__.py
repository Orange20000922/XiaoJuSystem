"""Textual-based terminal UI helpers."""

from __future__ import annotations

import importlib
from typing import Dict, Tuple


_EXPORTS: Dict[str, Tuple[str, str]] = {
    "RetroTheme": ("src.client.tui.theme", "RetroTheme"),
    "AMBER_CRT_THEME": ("src.client.tui.theme", "AMBER_CRT_THEME"),
    "PixelArtAsset": ("src.client.tui.pixel_art", "PixelArtAsset"),
    "PixelArtRegistry": ("src.client.tui.pixel_art", "PixelArtRegistry"),
    "PixelArtRenderer": ("src.client.tui.pixel_art", "PixelArtRenderer"),
    "MetadataPixelArtRenderer": ("src.client.tui.pixel_art", "MetadataPixelArtRenderer"),
    "TextualClientApp": ("src.client.tui.textual_app", "TextualClientApp"),
    "create_textual_app": ("src.client.tui.textual_app", "create_textual_app"),
    "run_textual_client": ("src.client.tui.run_textual", "run_textual_client"),
}


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module 'src.client.tui' has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    module = importlib.import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(list(globals().keys()) + list(_EXPORTS.keys()))


__all__ = sorted(_EXPORTS.keys())
