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
    display_name="Amber CRT",
    title="Neuro Terminal",
    subtitle="retro TUI / amber phosphor / textual frontend",
    prompt_placeholder="Type a message or a slash command. /help for command list.",
    help_lines=(
        "/help",
        "/status",
        "/save [path]",
        "/clear-log",
        "/clear-context",
        "/sprite <id> <path>",
        "/sprite-use <id>",
        "/sprite-list",
        "/quit",
    ),
    boot_lines=(
        "NEURO TERMINAL BOOT SEQUENCE",
        "Textual frontend online.",
        "Pixel-art portrait panel is wired as a placeholder interface.",
    ),
)
