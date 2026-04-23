"""Textual application shell for the terminal client."""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.widgets import Footer, Header, Input, Static

from src.client.session import ClientChatSession
from src.client.vision_runtime import (
    VisionCallbacks,
    VisionRuntimeController,
    VisionRuntimeState,
)
from src.client.tui.pixel_art import (
    AnsiPixelArtRenderer,
    MetadataPixelArtRenderer,
    PixelArtRegistry,
    build_textual_image_widget,
)
from src.client.tui.theme import AMBER_CRT_THEME, RetroTheme
from src.core_engine.api import ChatResponse, DirectRuntime
from src.vision import VisualEvent, VisualWindowSummary


class MessageBubble(Static):
    """Single chat bubble in the conversation column."""

    DEFAULT_CLASSES = "message"

    def __init__(self, role: str, text: str):
        super().__init__("", classes=f"message {role}")
        self.role = role
        self.text = text
        self.border_title = {
            "user": "用户",
            "assistant": "助手",
            "system": "系统",
            "visual": "视觉",
        }.get(role, role)

    def on_mount(self) -> None:
        self.update(self.text or " ")


class PixelArtPanel(Static):
    """Portrait panel with image-widget hook and ANSI fallback."""

    def __init__(self, registry: PixelArtRegistry):
        super().__init__("", id="portrait-panel", classes="panel portrait-shell")
        self.registry = registry
        self.ansi_renderer = AnsiPixelArtRenderer(default_width=26, max_width=30)
        self.metadata_renderer = MetadataPixelArtRenderer()

    async def refresh_asset(self) -> None:
        await self.remove_children()
        asset = self.registry.get_active()
        if asset is None:
            self.border_title = "立绘"
            self.update(
                "尚未加载立绘资源。\n"
                "使用 /sprite <id> <path> 挂载像素立绘。"
            )
            return

        self.border_title = f"立绘 / {asset.asset_id}"

        widget = build_textual_image_widget(asset)
        if widget is not None:
            self.update("")
            await self.mount(widget)
            return

        fallback = Static(
            self.registry.preview_renderable(self.ansi_renderer, width=28),
            classes="portrait-ansi",
        )
        metadata = Static(
            self.metadata_renderer.render_preview(asset),
            classes="portrait-meta muted",
        )
        await self.mount(fallback, metadata)


class TextualClientApp(App[None]):
    """Retro-styled Textual frontend for the core engine."""

    CSS_PATH = "textual_app.tcss"
    BINDINGS = [
        Binding("ctrl+c", "request_quit", "退出"),
        Binding("ctrl+s", "save_transcript", "保存"),
        Binding("ctrl+g", "show_status", "状态"),
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
        self.runtime = (
            runtime_or_target
            if isinstance(runtime_or_target, DirectRuntime)
            else DirectRuntime(runtime_or_target)
        )
        self.session = ClientChatSession.from_target(
            self.runtime,
            context_id=context_id,
            verbose=True,
        )
        self.theme_spec = theme
        self.pixel_registry = pixel_registry or PixelArtRegistry(base_dir=Path.cwd())
        self.vision = VisionRuntimeController.from_runtime(
            self.runtime,
            self.session,
            callbacks=VisionCallbacks(
                on_event=self._on_visual_event,
                on_summary=self._on_visual_summary,
                on_visual_reply=self._on_visual_reply,
                on_status_change=self._on_vision_status_change,
            ),
        )
        self._busy = False
        self._last_reply_chars = 0
        self._last_error = ""
        self._vision_state = VisionRuntimeState()

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="shell"):
            with Vertical(id="left-column"):
                yield PixelArtPanel(self.pixel_registry)
                yield Static("", id="boot-panel", classes="panel compact")
                yield Static("", id="help-panel", classes="panel compact muted")
            with Vertical(id="center-column"):
                yield Static("", id="scene-bar", classes="panel compact scene-bar")
                with VerticalScroll(id="chat-scroll"):
                    yield Container(id="chat-thread")
                yield Input(
                    placeholder=self.theme_spec.prompt_placeholder,
                    id="chat-input",
                )
            with Vertical(id="right-column"):
                yield Static("", id="status-panel", classes="panel compact")
                yield Static("", id="sensor-panel", classes="panel compact")
                yield Static("", id="context-panel", classes="panel compact muted")
        yield Footer()

    def on_mount(self) -> None:
        self.title = self.theme_spec.title
        self.sub_title = self.theme_spec.subtitle
        self._update_boot_panel()
        self._update_help_panel()
        self._update_status_panel()
        self._update_sensor_panel()
        self._update_context_panel()
        self._update_scene_bar()
        self.call_after_refresh(self._refresh_portrait_panel)
        self._append_system("终端客户端已就绪。")
        self._append_assistant(
            f"人格已上线：{self.session.persona_name}\n"
            f"模型：{self.session.llm_model or '-'}"
        )
        self._input().focus()

    def action_request_quit(self) -> None:
        self.vision.shutdown()
        self.session.shutdown()
        self.exit()

    def action_save_transcript(self) -> None:
        output_path = "conversation_history.json"
        self.session.save_conversation(output_path)
        self._append_system(f"对话已保存到 {output_path}。")

    def action_show_status(self) -> None:
        self._append_system(self._build_status_text())

    def on_input_submitted(self, event: Input.Submitted) -> None:
        raw_text = event.value.strip()
        event.input.value = ""
        if not raw_text:
            return
        if self._busy:
            self._append_system("已有请求正在处理中。")
            return
        if raw_text.startswith("/"):
            self._handle_command(raw_text)
            return

        self._busy = True
        self._last_error = ""
        self._append_user(raw_text)
        self._input().disabled = True
        self._refresh_side_panels()
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
            self._append_system("引擎返回了空回复。")
        else:
            self._append_system("引擎在本轮抑制了回复。")
        self._complete_request()

    def _finish_chat_error(self, error_text: str) -> None:
        self._last_error = error_text
        self._append_system(f"请求失败：{error_text}")
        self._complete_request()

    def _complete_request(self) -> None:
        self._busy = False
        self._input().disabled = False
        self._input().focus()
        self._refresh_side_panels()

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
        if command == "/vision-start":
            source = parts[1] if len(parts) > 1 else "0"
            parsed_source: int | str = int(source) if source.isdigit() else source
            started = self.vision.start(parsed_source)
            if started:
                self._append_system(f"视觉运行时已启动，输入源：{source}。")
            else:
                self._append_system("视觉运行时已处于活动状态。")
            self._refresh_side_panels()
            return
        if command == "/vision-stop":
            self.vision.stop()
            self._append_system("视觉运行时已停止。")
            self._refresh_side_panels()
            return
        if command == "/vision-status":
            self._append_system(self._build_vision_text())
            return
        if command == "/save":
            output_path = parts[1] if len(parts) > 1 else "conversation_history.json"
            self.session.save_conversation(output_path)
            self._append_system(f"对话已保存到 {output_path}。")
            return
        if command == "/clear-log":
            self._chat_thread().remove_children()
            self._append_system("对话区已清空。")
            return
        if command == "/clear-context":
            result = self.session.clear_context()
            self._append_system(
                f"上下文已清除：已从 {result.context_id} 移除 {result.cleared_turns} 轮。"
            )
            self._refresh_side_panels()
            return
        if command == "/sprite":
            if len(parts) < 3:
                self._append_system("用法：/sprite <id> <path>")
                return
            try:
                asset = self.pixel_registry.register(parts[1], parts[2])
            except Exception as exc:
                self._append_system(f"立绘注册失败：{exc}")
                return
            self._append_system(f"立绘已注册：{asset.asset_id} -> {asset.source_path}")
            self.call_after_refresh(self._refresh_portrait_panel)
            return
        if command == "/sprite-use":
            if len(parts) < 2:
                self._append_system("用法：/sprite-use <id>")
                return
            try:
                asset = self.pixel_registry.set_active(parts[1])
            except Exception as exc:
                self._append_system(f"立绘切换失败：{exc}")
                return
            self._append_system(f"当前立绘已切换为 {asset.asset_id}。")
            self.call_after_refresh(self._refresh_portrait_panel)
            return
        if command == "/sprite-list":
            assets = self.pixel_registry.list_assets()
            if not assets:
                self._append_system("当前没有已注册的立绘资源。")
                return
            listing = "\n".join(f"{asset.asset_id} -> {asset.source_path}" for asset in assets)
            self._append_system(listing)
            return
        if command == "/quit":
            self.action_request_quit()
            return

        self._append_system(f"未知命令：{command}")

    def _split_command(self, raw_text: str) -> list[str]:
        try:
            parts = shlex.split(raw_text, posix=False)
        except ValueError as exc:
            self._append_system(f"命令解析失败：{exc}")
            return []
        return [part.strip('"') for part in parts]

    def _refresh_side_panels(self) -> None:
        self._update_status_panel()
        self._update_sensor_panel()
        self._update_context_panel()
        self._update_scene_bar()

    def _update_boot_panel(self) -> None:
        self.query_one("#boot-panel", Static).update("\n".join(self.theme_spec.boot_lines))

    def _update_help_panel(self) -> None:
        self.query_one("#help-panel", Static).update(self._build_help_text())

    def _update_status_panel(self) -> None:
        self.query_one("#status-panel", Static).update(self._build_status_text())

    def _update_sensor_panel(self) -> None:
        self.query_one("#sensor-panel", Static).update(self._build_sensor_text())

    def _update_context_panel(self) -> None:
        self.query_one("#context-panel", Static).update(self._build_context_text())

    def _update_scene_bar(self) -> None:
        status = self.session.get_status()
        running = "运行中" if getattr(status, "running", False) or self._busy else "空闲"
        wm_turns = getattr(status, "working_memory_turns", 0)
        queue_size = getattr(status, "queue_size", None)
        queue_text = "-" if queue_size is None else str(queue_size)
        vision_text = "视觉在线" if self._vision_state.running else "视觉关闭"
        self.query_one("#scene-bar", Static).update(
            "  ".join(
                [
                    f"人格 {self.session.persona_name}",
                    f"上下文 {self.session.context_id}",
                    f"状态 {running}",
                    vision_text,
                    f"记忆 {wm_turns}",
                    f"队列 {queue_text}",
                ]
            )
        )

    async def _refresh_portrait_panel(self) -> None:
        await self.query_one(PixelArtPanel).refresh_asset()

    def _build_help_text(self) -> str:
        lines = ["命令"] + [f"  {line}" for line in self.theme_spec.help_lines]
        return "\n".join(lines)

    def _build_status_text(self) -> str:
        status = self.session.get_status()
        state = "忙碌" if self._busy else ("运行中" if getattr(status, "running", False) else "空闲")
        lines = [
            "状态",
            f"  人格      {self.session.persona_name}",
            f"  模型      {self.session.llm_model or '-'}",
            f"  上下文    {self.session.context_id}",
            f"  运行      {state}",
            f"  工作记忆  {getattr(status, 'working_memory_turns', 0)}",
        ]
        proactive_state = getattr(status, "proactive_state", None)
        if proactive_state:
            lines.append(f"  主动状态  {proactive_state}")
        idle_seconds = getattr(status, "idle_seconds", None)
        if idle_seconds is not None:
            lines.append(f"  空闲时长  {idle_seconds}s")
        if getattr(status, "queue_size", None) is not None:
            lines.append(f"  队列      {status.queue_size}")
        lines.append(f"  视觉      {'运行中' if self._vision_state.running else '空闲'}")
        lines.append(f"  上次回复  {self._last_reply_chars} 字")
        if self._last_error:
            lines.append(f"  上次错误  {self._last_error}")
        return "\n".join(lines)

    def _build_sensor_text(self) -> str:
        status = self.session.get_status()
        attention = getattr(status, "attention", {}) or {}
        emotion = getattr(status, "emotion_state", {}) or {}

        focused = attention.get("focused_users", 0)
        context_users = attention.get("context_users", 0)
        cooldown = "开启" if attention.get("cooldown_active") else "关闭"
        reply_age = attention.get("last_reply_ago")
        reply_age_text = "-" if reply_age is None else f"{int(reply_age)}s"

        valence = emotion.get("valence")
        arousal = emotion.get("arousal")
        mood = self._localize_emotion_label(emotion.get("last_emotion") or "neutral")
        valence_text = "-" if valence is None else f"{float(valence):+.2f}"
        arousal_text = "-" if arousal is None else f"{float(arousal):.2f}"

        lines = [
            "传感",
            f"  焦点人数  {focused}",
            f"  上下文    {context_users}",
            f"  冷却      {cooldown}",
            f"  上次发送  {reply_age_text}",
            "",
            "情绪",
            f"  心境      {mood}",
            f"  愉悦度    {valence_text}",
            f"  唤醒度    {arousal_text}",
            "",
            "视觉",
            f"  状态      {'运行中' if self._vision_state.running else '空闲'}",
            f"  来源      {self._vision_state.source}",
            f"  事件数    {self._vision_state.candidate_count}",
            f"  晋升数    {self._vision_state.promoted_count}",
        ]
        return "\n".join(lines)

    def _build_context_text(self) -> str:
        status = self.session.get_status()
        contexts = getattr(status, "contexts", []) or []
        patterns = getattr(status, "patterns", []) or []

        lines = ["运行时"]
        lines.append("  前端      textual")
        lines.append(f"  配置      {getattr(status, 'config_source', '') or '-'}")
        if contexts:
            lines.append(f"  绑定      {', '.join(contexts[:3])}")
        if patterns:
            lines.append(f"  模式      {', '.join(patterns[:3])}")
        if self.pixel_registry.active_asset_id:
            lines.append(f"  立绘      {self.pixel_registry.active_asset_id}")
        else:
            lines.append("  立绘      无")
        lines.append(f"  视觉      {'活动' if self._vision_state.running else '待机'}")
        lines.append(f"  总结      {self._trim_panel_line(self._vision_state.last_summary_text)}")
        return "\n".join(lines)

    def _build_vision_text(self) -> str:
        state = self._vision_state
        lines = [
            "视觉",
            f"  状态      {'运行中' if state.running else '空闲'}",
            f"  来源      {state.source}",
            f"  事件数    {state.candidate_count}",
            f"  晋升数    {state.promoted_count}",
            f"  总结数    {state.summary_count}",
            f"  最新事件  {self._trim_panel_line(state.last_event_text, width=56)}",
            f"  最新总结  {self._trim_panel_line(state.last_summary_text, width=56)}",
        ]
        if state.error:
            lines.append(f"  错误      {self._trim_panel_line(state.error, width=56)}")
        return "\n".join(lines)

    def _trim_panel_line(self, text: str, *, width: int = 28) -> str:
        value = (text or "-").strip()
        if len(value) <= width:
            return value
        return value[: max(0, width - 3)] + "..."

    def _localize_emotion_label(self, emotion: str) -> str:
        return {
            "neutral": "平静",
            "joy": "喜悦",
            "sadness": "悲伤",
            "anger": "愤怒",
            "fear": "恐惧",
            "surprise": "惊讶",
            "disgust": "厌恶",
            "excitement": "兴奋",
            "tenderness": "温柔",
            "curiosity": "好奇",
        }.get((emotion or "").strip().lower(), emotion or "-")

    def _append_user(self, text: str) -> None:
        self._append_message("user", text)

    def _append_assistant(self, text: str) -> None:
        self._append_message("assistant", text)

    def _append_system(self, text: str) -> None:
        self._append_message("system", text)

    def _append_visual(self, text: str) -> None:
        self._append_message("visual", text)

    def _append_message(self, role: str, text: str) -> None:
        bubble = MessageBubble(role, text)
        self._chat_thread().mount(bubble)
        self.call_after_refresh(self._scroll_chat_to_end)

    def _scroll_chat_to_end(self) -> None:
        self.query_one("#chat-scroll", VerticalScroll).scroll_end(animate=False)

    def _chat_thread(self) -> Container:
        return self.query_one("#chat-thread", Container)

    def _input(self) -> Input:
        return self.query_one("#chat-input", Input)

    def _on_visual_event(self, event: VisualEvent) -> None:
        self.call_from_thread(self._render_visual_event, event)

    def _on_visual_summary(self, summary: VisualWindowSummary) -> None:
        self.call_from_thread(self._render_visual_summary, summary)

    def _on_visual_reply(self, event: VisualEvent, reply: str) -> None:
        self.call_from_thread(self._render_visual_reply, event, reply)

    def _on_vision_status_change(self, state: VisionRuntimeState) -> None:
        self.call_from_thread(self._apply_vision_state, state)

    def _apply_vision_state(self, state: VisionRuntimeState) -> None:
        self._vision_state = state
        self._refresh_side_panels()

    def _render_visual_event(self, event: VisualEvent) -> None:
        self._append_visual(f"[视觉事件] {event.summary_text()}")
        self._refresh_side_panels()

    def _render_visual_summary(self, summary: VisualWindowSummary) -> None:
        self._append_system(f"[视觉总结] {summary.to_text()}")
        self._refresh_side_panels()

    def _render_visual_reply(self, event: VisualEvent, reply: str) -> None:
        self._append_visual(f"[视觉直连] {event.summary_text()}")
        self._append_assistant(reply)
        self._refresh_side_panels()


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
