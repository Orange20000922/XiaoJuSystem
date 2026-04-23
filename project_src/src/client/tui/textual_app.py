"""Textual application shell for the terminal client."""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container
from textual.widgets import Footer, Header, Input, RichLog, Static

from src.client.session import ClientChatSession
from src.client.tui.pixel_art import PixelArtRegistry
from src.client.tui.theme import AMBER_CRT_THEME, RetroTheme
from src.core_engine.api import ChatResponse


class TextualClientApp(App[None]):
    """Retro-styled Textual frontend for the core engine."""

    CSS_PATH = "textual_app.tcss"
    BINDINGS = [
        Binding("ctrl+c", "request_quit", "Quit"),
        Binding("ctrl+s", "save_transcript", "Save"),
        Binding("ctrl+g", "show_status", "Status"),
    ]

    def __init__(
        self,
        runtime_or_target: Any,
        *,
        context_id: str = "tui_default",
        theme: RetroTheme = AMBER_CRT_THEME,
        pixel_registry: PixelArtRegistry | None = None,
    ):
        super().__init__()
        self.session = ClientChatSession.from_target(
            runtime_or_target,
            context_id=context_id,
            verbose=True,
        )
        self.theme_spec = theme
        self.pixel_registry = pixel_registry or PixelArtRegistry(base_dir=Path.cwd())
        self._busy = False
        self._last_reply_chars = 0
        self._last_error = ""

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Container(id="body"):
            with Container(id="sidebar"):
                yield Static("", id="boot-panel", classes="panel")
                yield Static("", id="status-panel", classes="panel")
                yield Static("", id="sprite-panel", classes="panel")
                yield Static("", id="help-panel", classes="panel")
            with Container(id="chat-column"):
                yield RichLog(id="chat-log", wrap=True, markup=False, highlight=False)
                yield Input(
                    placeholder=self.theme_spec.prompt_placeholder,
                    id="chat-input",
                )
        yield Footer()

    def on_mount(self) -> None:
        self.title = self.theme_spec.title
        self.sub_title = self.theme_spec.subtitle
        self._update_boot_panel()
        self._update_status_panel()
        self._update_sprite_panel()
        self._update_help_panel()
        self._append_system("Terminal client ready.")
        self._chat_log().write(f"persona      {self.session.persona_name}")
        if self.session.llm_model:
            self._chat_log().write(f"model        {self.session.llm_model}")
        self._input().focus()

    def action_request_quit(self) -> None:
        self.session.shutdown()
        self.exit()

    def action_save_transcript(self) -> None:
        output_path = "conversation_history.json"
        self.session.save_conversation(output_path)
        self._append_system(f"Conversation saved to {output_path}.")

    def action_show_status(self) -> None:
        self._append_system(self._build_status_text())

    def on_input_submitted(self, event: Input.Submitted) -> None:
        raw_text = event.value.strip()
        event.input.value = ""
        if not raw_text:
            return
        if self._busy:
            self._append_system("A request is already running.")
            return
        if raw_text.startswith("/"):
            self._handle_command(raw_text)
            return

        self._busy = True
        self._last_error = ""
        self._append_user(raw_text)
        self._input().disabled = True
        self._update_status_panel()
        self._dispatch_chat(raw_text)

    @work(thread=True, exclusive=True, exit_on_error=False)
    def _dispatch_chat(self, user_text: str) -> None:
        try:
            response = self.session.send_message(user_text)
        except Exception as exc:
            self.call_from_thread(self._finish_chat_error, str(exc))
            return
        self.call_from_thread(self._finish_chat_success, response)

    def _finish_chat_success(self, response: ChatResponse) -> None:
        reply = (response.reply or "").strip()
        self._last_reply_chars = len(reply)
        if response.should_respond and reply:
            self._append_assistant(reply)
        elif response.should_respond:
            self._append_system("Engine returned an empty reply.")
        else:
            self._append_system("Engine suppressed the reply for this turn.")
        self._complete_request()

    def _finish_chat_error(self, error_text: str) -> None:
        self._last_error = error_text
        self._append_system(f"Request failed: {error_text}")
        self._complete_request()

    def _complete_request(self) -> None:
        self._busy = False
        self._input().disabled = False
        self._input().focus()
        self._update_status_panel()

    def _handle_command(self, raw_text: str) -> None:
        parts = self._split_command(raw_text)
        if not parts:
            return

        command = parts[0].lower()
        if command == "/help":
            self._append_system(self._build_help_text())
            return
        if command == "/status":
            self._append_system(self._build_status_text())
            return
        if command == "/save":
            output_path = parts[1] if len(parts) > 1 else "conversation_history.json"
            self.session.save_conversation(output_path)
            self._append_system(f"Conversation saved to {output_path}.")
            return
        if command == "/clear-log":
            self._chat_log().clear()
            self._append_system("Chat log cleared.")
            return
        if command == "/clear-context":
            result = self.session.clear_context()
            self._append_system(
                f"Context cleared: {result.cleared_turns} turns removed from {result.context_id}."
            )
            return
        if command == "/sprite":
            if len(parts) < 3:
                self._append_system("Usage: /sprite <id> <path>")
                return
            try:
                asset = self.pixel_registry.register(parts[1], parts[2])
            except Exception as exc:
                self._append_system(f"Sprite registration failed: {exc}")
                return
            self._append_system(f"Sprite registered: {asset.asset_id} -> {asset.source_path}")
            self._update_sprite_panel()
            return
        if command == "/sprite-use":
            if len(parts) < 2:
                self._append_system("Usage: /sprite-use <id>")
                return
            try:
                asset = self.pixel_registry.set_active(parts[1])
            except Exception as exc:
                self._append_system(f"Sprite switch failed: {exc}")
                return
            self._append_system(f"Active sprite set to {asset.asset_id}.")
            self._update_sprite_panel()
            return
        if command == "/sprite-list":
            assets = self.pixel_registry.list_assets()
            if not assets:
                self._append_system("No sprite assets registered.")
                return
            listing = "\n".join(f"{asset.asset_id} -> {asset.source_path}" for asset in assets)
            self._append_system(listing)
            return
        if command == "/quit":
            self.action_request_quit()
            return

        self._append_system(f"Unknown command: {command}")

    def _split_command(self, raw_text: str) -> list[str]:
        try:
            parts = shlex.split(raw_text, posix=False)
        except ValueError as exc:
            self._append_system(f"Command parse error: {exc}")
            return []
        return [part.strip('"') for part in parts]

    def _update_boot_panel(self) -> None:
        self.query_one("#boot-panel", Static).update("\n".join(self.theme_spec.boot_lines))

    def _update_status_panel(self) -> None:
        self.query_one("#status-panel", Static).update(self._build_status_text())

    def _update_sprite_panel(self) -> None:
        self.query_one("#sprite-panel", Static).update(self.pixel_registry.preview_text())

    def _update_help_panel(self) -> None:
        self.query_one("#help-panel", Static).update(self._build_help_text())

    def _build_help_text(self) -> str:
        lines = ["commands"] + [f"  {line}" for line in self.theme_spec.help_lines]
        return "\n".join(lines)

    def _build_status_text(self) -> str:
        status = self.session.get_status()
        state = "busy" if self._busy else ("running" if getattr(status, "running", False) else "idle")
        lines = [
            f"persona      {self.session.persona_name}",
            f"model        {self.session.llm_model or '-'}",
            f"context      {self.session.context_id}",
            f"frontend     textual",
            f"state        {state}",
            f"wm turns     {getattr(status, 'working_memory_turns', 0)}",
        ]
        proactive_state = getattr(status, "proactive_state", None)
        if proactive_state:
            lines.append(f"proactive    {proactive_state}")
        idle_seconds = getattr(status, "idle_seconds", None)
        if idle_seconds is not None:
            lines.append(f"idle secs    {idle_seconds}")
        lines.append(f"last reply   {self._last_reply_chars} chars")
        if self._last_error:
            lines.append(f"last error   {self._last_error}")
        return "\n".join(lines)

    def _append_user(self, text: str) -> None:
        self._write_block("user", text)

    def _append_assistant(self, text: str) -> None:
        self._write_block("assistant", text)

    def _append_system(self, text: str) -> None:
        self._write_block("system", text)

    def _write_block(self, speaker: str, text: str) -> None:
        prefix = f"{speaker:<12}"
        lines = text.splitlines() or [""]
        self._chat_log().write(f"{prefix}{lines[0]}")
        for line in lines[1:]:
            self._chat_log().write(f"{'':12}{line}")

    def _chat_log(self) -> RichLog:
        return self.query_one("#chat-log", RichLog)

    def _input(self) -> Input:
        return self.query_one("#chat-input", Input)


def create_textual_app(
    runtime_or_target: Any,
    *,
    context_id: str = "tui_default",
    theme: RetroTheme = AMBER_CRT_THEME,
    pixel_registry: PixelArtRegistry | None = None,
) -> TextualClientApp:
    """Create a Textual client app without running it."""

    return TextualClientApp(
        runtime_or_target,
        context_id=context_id,
        theme=theme,
        pixel_registry=pixel_registry,
    )
