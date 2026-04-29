

from __future__ import annotations

import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

from src.client.session import ClientChatSession, coerce_runtime
from src.core_engine.api import DirectRuntime
from src.logger import logger
from src.vision import (
    VisualEvent,
    VisualPerceptionConfig,
    VisualPerceptionPipeline,
    VisualWindowSummary,
    build_visual_analyzer_from_pipeline,
    visual_event_to_agent_text,
)


@dataclass
class VisionRuntimeState:
    """TUI视觉管线的状态快照。"""

    running: bool = False
    source: str = "-"
    error: str = ""
    last_event_text: str = "-"
    last_summary_text: str = "-"
    candidate_count: int = 0
    promoted_count: int = 0
    summary_count: int = 0


@dataclass
class VisionCallbacks:
    """TUI视觉管线的回调函数类"""

    on_event: Optional[Callable[[VisualEvent], None]] = None
    on_summary: Optional[Callable[[VisualWindowSummary], None]] = None
    on_visual_reply: Optional[Callable[[VisualEvent, str], None]] = None
    on_status_change: Optional[Callable[[VisionRuntimeState], None]] = None


@dataclass
class VisionRuntimeController:
    """TUI视觉管线控制器。"""

    runtime: DirectRuntime
    session: ClientChatSession
    callbacks: VisionCallbacks = field(default_factory=VisionCallbacks)

    def __post_init__(self) -> None:
        self._pipeline: Optional[VisualPerceptionPipeline] = None
        self._thread: Optional[threading.Thread] = None
        self._reply_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="client-vision-reply",
        )
        self._reply_futures: List[Future] = []
        self._reply_shutdown = False
        self._lock = threading.RLock()
        self._state = VisionRuntimeState()
        self._last_events: List[VisualEvent] = []
        self._last_summaries: List[VisualWindowSummary] = []
        self._last_source = "-"
        self._skill_wait_timeout_seconds = 2.5
        self._skill_poll_interval_seconds = 0.2
        self._register_visual_skill()

    @classmethod
    def from_runtime(
        cls,
        runtime_or_target,
        session: ClientChatSession,
        *,
        callbacks: Optional[VisionCallbacks] = None,
    ) -> "VisionRuntimeController":
        return cls(
            runtime=coerce_runtime(runtime_or_target),
            session=session,
            callbacks=callbacks or VisionCallbacks(),
        )

    @property
    def state(self) -> VisionRuntimeState:
        with self._lock:
            return VisionRuntimeState(**self._state.__dict__)

    @property
    def running(self) -> bool:
        return self.state.running

    def recent_events(self, top_k: int = 2) -> List[VisualEvent]:
        with self._lock:
            events = list(self._last_events)
        if not events:
            return []
        return sorted(
            events,
            key=lambda item: (item.peak_score, item.timestamp),
            reverse=True,
        )[: max(0, top_k)]

    def start(
        self,
        source: int | str,
        *,
        duration_seconds: Optional[float] = None,
        camera_width: Optional[int] = None,
        camera_height: Optional[int] = None,
        camera_fps: Optional[float] = None,
    ) -> bool:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return False
            self._last_events = []
            self._last_summaries = []
            self._last_source = str(source)
            self._state = VisionRuntimeState(running=True, source=str(source))
            self._pipeline = self._build_pipeline()
            self._thread = threading.Thread(
                target=self._run_pipeline,
                args=(source, duration_seconds, camera_width, camera_height, camera_fps),
                daemon=True,
                name="client-vision-runtime",
            )
            self._thread.start()
        self._notify_status()
        return True

    def stop(self) -> None:
        pipeline = None
        thread = None
        with self._lock:
            pipeline = self._pipeline
            thread = self._thread
        if pipeline is not None:
            pipeline.stop()
        if thread is not None and thread.is_alive():
            thread.join(timeout=5.0)

    def shutdown(self) -> None:
        with self._lock:
            self._reply_shutdown = True
        self.stop()
        self._reply_executor.shutdown(wait=True, cancel_futures=True)

    def _build_pipeline(self) -> VisualPerceptionPipeline:
        target = self.runtime.target
        config = VisualPerceptionConfig.from_settings(
            getattr(target, "visual_perception_config", None)
        )
        analyzer = build_visual_analyzer_from_pipeline(target)
        return VisualPerceptionPipeline(
            config=config,
            analyzer=analyzer,
            event_callback=self._handle_candidate_event,
            promoted_event_callback=self._handle_promoted_event,
            summary_callback=self._handle_summary,
        )

    def _register_visual_skill(self) -> None:
        target = self.runtime.target
        if hasattr(target, "register_visual_skill"):
            try:
                target.register_visual_skill(self._execute_visual_skill)
            except Exception as exc:
                logger.warning(f"Failed to register visual skill: {exc}")

    def _run_pipeline(
        self,
        source: int | str,
        duration_seconds: Optional[float],
        camera_width: Optional[int],
        camera_height: Optional[int],
        camera_fps: Optional[float],
    ) -> None:
        error_text = ""
        try:
            assert self._pipeline is not None
            self._pipeline.run(
                source,
                duration_seconds=duration_seconds,
                camera_width=camera_width,
                camera_height=camera_height,
                camera_fps=camera_fps,
            )
        except Exception as exc:
            error_text = str(exc)
            logger.warning(f"Vision runtime stopped with error: {exc}")
        finally:
            self._flush_visual_shutdown_memory()
            with self._lock:
                self._state.running = False
                self._state.error = error_text
                self._pipeline = None
                self._thread = None
            self._notify_status()

    def _handle_candidate_event(self, event: VisualEvent) -> None:
        with self._lock:
            self._last_events.append(event)
            self._last_events = self._last_events[-24:]
            self._state.last_event_text = event.summary_text()
            self._state.candidate_count += 1
        if self.callbacks.on_event is not None:
            self.callbacks.on_event(event)
        self._notify_status()

    def _handle_promoted_event(self, event: VisualEvent) -> None:
        with self._lock:
            if self._reply_shutdown:
                return
            self._state.promoted_count += 1
            future = self._reply_executor.submit(self._send_visual_direct_reply, event)
            self._reply_futures.append(future)
            self._reply_futures = [item for item in self._reply_futures if not item.done()]
        self._notify_status()

    def _handle_summary(self, summary: VisualWindowSummary) -> None:
        with self._lock:
            self._last_summaries.append(summary)
            self._last_summaries = self._last_summaries[-12:]
            self._state.last_summary_text = summary.to_text()
            self._state.summary_count += 1
        if self.callbacks.on_summary is not None:
            self.callbacks.on_summary(summary)
        self._notify_status()

    def _analysis_payload(self, event: VisualEvent) -> dict:
        if event.analysis is None:
            return {}
        return {
            "scene": event.analysis.scene,
            "facts": list(event.analysis.facts),
            "weak_interpretations": list(event.analysis.weak_interpretations),
            "memory_candidate": event.analysis.memory_candidate,
            "agent_hint": event.analysis.agent_hint,
            "raw_text": event.analysis.raw_text,
        }

    def _flush_visual_shutdown_memory(self) -> None:
        target = self.runtime.target
        memory = getattr(target, "memory", None)
        if memory is None:
            return

        summary_text = self._build_session_summary()
        if summary_text and hasattr(memory, "add_visual_session_summary_l3"):
            try:
                memory.add_visual_session_summary_l3(summary_text)
            except Exception as exc:
                logger.warning(f"Failed to write visual summary to L3: {exc}")

        emotion_tracker = getattr(target, "emotion_state_tracker", None)
        if (
            emotion_tracker is not None
            and hasattr(emotion_tracker, "to_dict")
            and hasattr(memory, "add_emotion_snapshot_l4")
        ):
            try:
                memory.add_emotion_snapshot_l4(
                    emotion_tracker.to_dict(),
                    source="vision_runtime_stop",
                )
            except Exception as exc:
                logger.warning(f"Failed to write emotion snapshot to L4: {exc}")

    def _build_session_summary(self) -> str:
        with self._lock:
            summaries = list(self._last_summaries)
            promoted_events = [
                event for event in self._last_events
                if str(event.metrics.get("mode", "")).strip().lower() in {"trigger", "summary", "manual"}
            ]

        if summaries:
            parts = [summary.to_text() for summary in summaries[-3:] if summary.to_text()]
            if parts:
                return " | ".join(parts)

        if promoted_events:
            parts = [event.summary_text() for event in promoted_events[-3:] if event.summary_text()]
            if parts:
                return "；".join(parts)

        return ""

    def _execute_visual_skill(self, top_k: int = 2) -> str | List[VisualEvent]:
        events = self.recent_events(top_k=top_k)
        if events:
            return events

        started = self.ensure_started()
        if not started:
            return "视觉管线暂时无法启动，请检查摄像头或输入源配置。"

        deadline = time.time() + self._skill_wait_timeout_seconds
        while time.time() < deadline:
            events = self.recent_events(top_k=top_k)
            if events:
                return events
            time.sleep(self._skill_poll_interval_seconds)

        return "视觉已启动，正在观察当前画面，请稍后再问一次。"

    def ensure_started(
        self,
        source: int | str = 0,
        *,
        duration_seconds: Optional[float] = None,
        camera_width: Optional[int] = None,
        camera_height: Optional[int] = None,
        camera_fps: Optional[float] = None,
    ) -> bool:
        if self.running:
            return True

        try:
            started = self.start(
                source,
                duration_seconds=duration_seconds,
                camera_width=camera_width,
                camera_height=camera_height,
                camera_fps=camera_fps,
            )
        except Exception as exc:
            logger.warning(f"Failed to ensure vision runtime start: {exc}")
            with self._lock:
                self._state.error = str(exc)
            self._notify_status()
            return False

        return started or self.running

    def _send_visual_direct_reply(self, event: VisualEvent) -> None:
        try:
            response = self.session.send_message(
                visual_event_to_agent_text(event),
                images=list(event.keyframes),
                metadata={
                    "event_id": event.event_id,
                    "analysis": self._analysis_payload(event),
                    "peak_score": event.peak_score,
                    "peak_frame_index": event.peak_frame_index,
                    "representative_frame_index": event.representative_frame_index,
                    "metrics": dict(event.metrics),
                    "timestamp": event.timestamp,
                    "rate_limited": event.rate_limited,
                },
                visual_direct=True,
            )
        except Exception as exc:
            logger.warning(f"Vision direct reply failed: {exc}")
            with self._lock:
                self._state.error = str(exc)
            self._notify_status()
            return

        reply = (response.reply or "").strip()
        if reply and self.callbacks.on_visual_reply is not None:
            self.callbacks.on_visual_reply(event, reply)
        self._notify_status()

    def _notify_status(self) -> None:
        if self.callbacks.on_status_change is None:
            return
        try:
            self.callbacks.on_status_change(self.state)
        except Exception as exc:
            logger.warning(f"Vision status callback failed: {exc}")


__all__ = [
    "VisionCallbacks",
    "VisionRuntimeController",
    "VisionRuntimeState",
]
