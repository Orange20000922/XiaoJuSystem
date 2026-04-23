"""Retro terminal theme definitions for the Textual client."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RetroTheme:
    """Client-facing theme metadata."""

    theme_id: str
    display_name: str
    title: str
    subtitle: str
    prompt_placeholder: str
    help_lines: tuple[str, ...]
    boot_lines: tuple[str, ...]


AMBER_CRT_THEME = RetroTheme(
    theme_id="amber-crt",
    display_name="柔和复古终端",
    title="小橘 TUI 终端",
    subtitle="复古 TUI / 柔和荧屏 / Textual 前端",
    prompt_placeholder="输入消息或斜杠命令。使用 /help 查看命令列表。",
    help_lines=(
        "/help",
        "/status",
        "/vision-start [source]",
        "/vision-stop",
        "/vision-status",
        "/save [path]",
        "/clear-log",
        "/clear-context",
        "/sprite <id> <path>",
        "/sprite-use <id>",
        "/sprite-list",
        "/quit",
    ),
    boot_lines=(
        "小橘 TUI 终端启动序列",
        "Textual 前端已上线。",
        "立绘相对路径默认从 resource/ 解析。",
        "立绘面板支持 ANSI 像素渲染，并预留图像组件回退接口。",
    ),
)
