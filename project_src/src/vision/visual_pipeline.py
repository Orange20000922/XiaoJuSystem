from __future__ import annotations

import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Deque, List, Optional, Union

import numpy as np

from src.logger import logger

from ._shared import (
    _encode_frame_as_image_result,
    _ensure_cv2,
    _fit_width,
    _odd_kernel,
    _open_video_writer,
    _push_queue_sentinel,
    cv2,
    sigmoid_normalize,
)
from .visual_monitor import (
    VisualEventCluster,
    VisualEventMonitor,
    VisualMonitorUpdate,
    VisualWindowSummary,
)
from .visual_types import (
    FrameObservation,
    FramePacket,
    PeakSelection,
    VisualAnalysis,
    VisualEvent,
    VisualEventAnalyzer,
    VisualPerceptionConfig,
)
from .adaptive_sampler import AdaptiveFrameSampler, AdaptiveFrameSamplerConfig


@dataclass
class _DetectorState:
    subtractor: object
    morphology_kernel: np.ndarray
    temporal_votes: Deque[bool]
    previous_gray: Optional[np.ndarray] = None
    previous_histogram: Optional[np.ndarray] = None
    previous_edge_density: Optional[float] = None
    previous_smoothed_score: float = 0.0


class TemporalPeakSelector:
    def __init__(
        self,
        peak_threshold: float,
        cooldown_seconds: float,
        window_seconds: float,
        peak_neighborhood_frames: int,
        clip_half_duration: float = 0.0,
    ):
        self.peak_threshold = peak_threshold
        self.cooldown_seconds = cooldown_seconds
        self.peak_neighborhood_frames = peak_neighborhood_frames
        self.clip_half_duration = max(0.0, clip_half_duration)
        self.window_seconds = (
            max(window_seconds, clip_half_duration * 2)
            if self.clip_half_duration > 0
            else window_seconds
        )
        self._window: Deque[FrameObservation] = deque()
        self._history: Deque[FrameObservation] = deque(maxlen=3)
        self._last_peak_time = -1e9
        self._pending_peak: Optional[PeakSelection] = None
        self._pending_deadline: float = 0.0

    def consume(self, observation: FrameObservation) -> Optional[PeakSelection]:
        self._window.append(observation)
        self._history.append(observation)
        self._trim_window(observation.timestamp)

        # Check if a pending clip peak is ready to emit.
        result = self._try_emit_pending(observation.timestamp)

        if len(self._history) >= 3:
            left, center, right = self._history
            is_local_max = (
                center.smoothed_score > left.smoothed_score
                and center.smoothed_score >= right.smoothed_score
            )
            if (
                is_local_max
                and center.smoothed_score >= self.peak_threshold
                and center.motion_vote
                and (center.timestamp - self._last_peak_time) >= self.cooldown_seconds
            ):
                representative = self._select_representative(center.frame_index)
                self._last_peak_time = center.timestamp

                if self.clip_half_duration > 0:
                    self._pending_peak = PeakSelection(
                        peak=center,
                        representative=representative,
                        window=[],  # filled at emit time
                    )
                    self._pending_deadline = center.timestamp + self.clip_half_duration
                else:
                    return PeakSelection(
                        peak=center,
                        representative=representative,
                        window=list(self._window),
                    )

        return result

    def flush(self) -> Optional[PeakSelection]:
        """Emit any pending peak regardless of deadline (called at stream end)."""
        if self._pending_peak is None:
            return None
        return self._emit_pending()

    def _try_emit_pending(self, now: float) -> Optional[PeakSelection]:
        if self._pending_peak is None:
            return None
        if now < self._pending_deadline:
            return None
        return self._emit_pending()

    def _emit_pending(self) -> Optional[PeakSelection]:
        pending = self._pending_peak
        self._pending_peak = None
        if pending is None:
            return None
        peak_time = pending.peak.timestamp
        half = self.clip_half_duration
        clip_window = [
            obs for obs in self._window
            if (peak_time - half) <= obs.timestamp <= (peak_time + half)
        ]
        if not clip_window:
            clip_window = list(self._window)
        pending.window = clip_window
        return pending

    def _trim_window(self, now: float) -> None:
        cutoff = now - self.window_seconds
        while self._window and self._window[0].timestamp < cutoff:
            self._window.popleft()

    def _select_representative(self, peak_frame_index: int) -> FrameObservation:
        candidates = [
            item
            for item in self._window
            if abs(item.frame_index - peak_frame_index) <= self.peak_neighborhood_frames
        ]
        if not candidates:
            candidates = list(self._window)
        return max(
            candidates,
            key=lambda item: (
                item.sharpness,
                item.smoothed_score,
                -abs(item.frame_index - peak_frame_index),
            ),
        )


class VisualPerceptionPipeline:
    def __init__(
        self,
        config: Optional[VisualPerceptionConfig] = None,
        analyzer: Optional[VisualEventAnalyzer] = None,
        event_callback: Optional[Callable[[VisualEvent], None]] = None,
        promoted_event_callback: Optional[Callable[[VisualEvent], None]] = None,
        summary_callback: Optional[Callable[[VisualWindowSummary], None]] = None,
    ):
        self.config = config or VisualPerceptionConfig()
        self.analyzer = analyzer
        self.event_callback = event_callback
        self.promoted_event_callback = promoted_event_callback
        self.summary_callback = summary_callback
        self._stop_event = threading.Event()
        self._events_lock = threading.Lock()
        self._collected_events: List[VisualEvent] = []
        self._promoted_events: List[VisualEvent] = []
        self._window_summaries: List[VisualWindowSummary] = []
        self._thread_error: Optional[Exception] = None
        self._thread_error_lock = threading.Lock()
        self._analysis_lock = threading.Lock()
        self._analysis_call_times: Deque[float] = deque()
        self._allow_analysis_wait = False
        self._monitor = VisualEventMonitor(self.config)
        self._adaptive_sampler = None

    def stop(self) -> None:
        self._stop_event.set()

    @property
    def candidate_events(self) -> List[VisualEvent]:
        return list(self._collected_events)

    @property
    def promoted_events(self) -> List[VisualEvent]:
        return list(self._promoted_events)

    @property
    def window_summaries(self) -> List[VisualWindowSummary]:
        return list(self._window_summaries)

    @property
    def recent_candidate_events(self) -> List[VisualEvent]:
        return self._monitor.recent_events

    def analyze_recent_buffer(
        self,
        top_k: Optional[int] = None,
    ) -> List[VisualEvent]:
        """对最近缓冲中的事件做聚类分析，返回按显著度排序的 top-k 事件。"""
        return self._monitor.analyze_recent_buffer(
            top_k=top_k,
            analyze_callback=self._analyze_event_if_allowed,
        )

    def run(
        self,
        source: Union[int, str, Path],
        max_frames: Optional[int] = None,
        duration_seconds: Optional[float] = None,
        save_video_path: Optional[Union[str, Path]] = None,
        camera_width: Optional[int] = None,
        camera_height: Optional[int] = None,
        camera_fps: Optional[float] = None,
    ) -> List[VisualEvent]:
        _ensure_cv2()
        if duration_seconds is not None and duration_seconds <= 0:
            raise ValueError("duration_seconds must be positive")
        if camera_width is not None and camera_width <= 0:
            raise ValueError("camera_width must be positive")
        if camera_height is not None and camera_height <= 0:
            raise ValueError("camera_height must be positive")
        if camera_fps is not None and camera_fps <= 0:
            raise ValueError("camera_fps must be positive")

        self._stop_event.clear()
        self._collected_events = []
        self._promoted_events = []
        self._window_summaries = []
        self._thread_error = None
        self._analysis_call_times.clear()
        self._allow_analysis_wait = (
            not isinstance(source, int)
            and self.config.vision_rate_limit_wait_for_file_source
        )
        self._monitor = VisualEventMonitor(self.config)
        self._adaptive_sampler = None

        frame_queue: "queue.Queue[Optional[FramePacket]]" = queue.Queue(
            maxsize=self.config.frame_queue_size
        )
        observation_queue: "queue.Queue[Optional[FrameObservation]]" = queue.Queue(
            maxsize=self.config.observation_queue_size
        )
        event_queue: "queue.Queue[Optional[VisualEvent]]" = queue.Queue(
            maxsize=self.config.event_queue_size
        )

        producer = threading.Thread(
            target=self._producer_loop,
            args=(
                source,
                max_frames,
                duration_seconds,
                save_video_path,
                camera_width,
                camera_height,
                camera_fps,
                frame_queue,
            ),
            daemon=True,
            name="visual-producer",
        )
        detector = threading.Thread(
            target=self._detector_loop,
            args=(frame_queue, observation_queue),
            daemon=True,
            name="visual-detector",
        )
        selector = threading.Thread(
            target=self._selector_loop,
            args=(observation_queue, event_queue),
            daemon=True,
            name="visual-selector",
        )
        uploader = threading.Thread(
            target=self._uploader_loop,
            args=(event_queue,),
            daemon=True,
            name="visual-uploader",
        )

        threads = [producer, detector, selector, uploader]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self._flush_monitor_state()
        self._raise_thread_error()
        return list(self._collected_events)

    def _record_thread_error(self, exc: Exception) -> None:
        with self._thread_error_lock:
            if self._thread_error is None:
                self._thread_error = exc
        self._stop_event.set()

    def _raise_thread_error(self) -> None:
        if self._thread_error is not None:
            raise self._thread_error

    def _flush_monitor_state(self) -> None:
        update = self._monitor.finalize()
        self._publish_monitor_update(update)

    def _publish_monitor_update(self, update: VisualMonitorUpdate) -> None:
        if update.promoted_events:
            self._promoted_events.extend(update.promoted_events)
            if self.promoted_event_callback is not None:
                for event in update.promoted_events:
                    try:
                        self.promoted_event_callback(event)
                    except Exception as exc:
                        logger.warning(f"Visual promoted-event callback failed: {exc}")

        if update.completed_summaries:
            self._window_summaries.extend(update.completed_summaries)
            if self.summary_callback is not None:
                for summary in update.completed_summaries:
                    try:
                        self.summary_callback(summary)
                    except Exception as exc:
                        logger.warning(f"Visual summary callback failed: {exc}")

    def _analyze_event_if_allowed(
        self,
        event: VisualEvent,
    ) -> tuple[Optional[VisualAnalysis], bool]:
        if self.analyzer is None or self.config.vision_analysis_mode == "none":
            return None, False

        if not self._reserve_analysis_slot(wait_for_slot=self._allow_analysis_wait):
            return None, True

        try:
            return self.analyzer.analyze(event), False
        except Exception as exc:
            logger.warning(f"Visual event analysis failed: {exc}")
            return None, False

    def _reserve_analysis_slot(self, *, wait_for_slot: bool) -> bool:
        if self.config.vision_calls_per_minute <= 0:
            return False

        waited_seconds = 0.0
        max_wait_seconds = max(0.0, self.config.vision_rate_limit_max_wait_seconds)

        while not self._stop_event.is_set():
            with self._analysis_lock:
                now = time.monotonic()
                while self._analysis_call_times and (now - self._analysis_call_times[0]) > 60.0:
                    self._analysis_call_times.popleft()
                if len(self._analysis_call_times) < self.config.vision_calls_per_minute:
                    self._analysis_call_times.append(now)
                    return True

                oldest_call = self._analysis_call_times[0]
                wait_seconds = max(0.01, 60.0 - (now - oldest_call) + 0.01)

            if not wait_for_slot:
                return False

            remaining_wait_seconds = max_wait_seconds - waited_seconds
            if remaining_wait_seconds <= 0:
                return False

            sleep_seconds = min(wait_seconds, remaining_wait_seconds)
            logger.info(
                f"Visual analysis is waiting {sleep_seconds:.2f}s for an available Vision LLM slot"
            )
            time.sleep(sleep_seconds)
            waited_seconds += sleep_seconds

        return False

    def recent_clusters(
        self,
        window_seconds: Optional[float] = None,
    ) -> List[VisualEventCluster]:
        return self._monitor.recent_clusters(window_seconds=window_seconds)

    def analyze_recent_activity(self, top_k: Optional[int] = None) -> List[VisualEvent]:
        return self._monitor.analyze_recent_buffer(
            top_k=top_k,
            analyze_callback=self._analyze_event_if_allowed,
        )

    def _configure_camera_capture(
        self,
        capture,
        *,
        camera_width: Optional[int],
        camera_height: Optional[int],
        camera_fps: Optional[float],
    ) -> None:
        if camera_width is not None:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(camera_width))
        if camera_height is not None:
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(camera_height))
        if camera_fps is not None:
            capture.set(cv2.CAP_PROP_FPS, float(camera_fps))

    def _producer_loop(
        self,
        source: Union[int, str, Path],
        max_frames: Optional[int],
        duration_seconds: Optional[float],
        save_video_path: Optional[Union[str, Path]],
        camera_width: Optional[int],
        camera_height: Optional[int],
        camera_fps: Optional[float],
        frame_queue: "queue.Queue[Optional[FramePacket]]",
    ) -> None:
        capture = None
        writer = None
        try:
            capture_source: Union[int, str] = source if isinstance(source, int) else str(source)
            capture = cv2.VideoCapture(capture_source)
            if not capture.isOpened():
                raise RuntimeError(f"Unable to open video source: {source}")

            if isinstance(source, int):
                self._configure_camera_capture(
                    capture,
                    camera_width=camera_width,
                    camera_height=camera_height,
                    camera_fps=camera_fps,
                )

            fps = capture.get(cv2.CAP_PROP_FPS)
            fallback_fps = float(camera_fps) if camera_fps is not None else 30.0
            fps = fps if fps and fps > 0 else fallback_fps
            frame_index = 0
            started_at = time.monotonic()

            if self.config.adaptive_sampling_enabled:
                sampler_cfg = AdaptiveFrameSamplerConfig(
                    stft_window_size=self.config.adaptive_stft_window_size,
                    stft_hop_size=self.config.adaptive_stft_hop_size,
                    highfreq_cutoff_ratio=self.config.adaptive_highfreq_cutoff_ratio,
                    fps_min=self.config.adaptive_fps_min,
                    fps_max=self.config.adaptive_fps_max,
                    gamma=self.config.adaptive_gamma,
                    spike_threshold=self.config.adaptive_spike_threshold,
                    spike_boost_seconds=self.config.adaptive_spike_boost_seconds,
                    smoothing_alpha=self.config.adaptive_fps_smoothing_alpha,
                    precheck_diff_threshold=self.config.adaptive_precheck_diff_threshold,
                    precheck_resize_width=self.config.adaptive_precheck_resize_width,
                )
                self._adaptive_sampler = AdaptiveFrameSampler(sampler_cfg, source_fps=fps)
                logger.info(
                    f"自适应帧采样已启用: fps_range=[{sampler_cfg.fps_min}, {sampler_cfg.fps_max}] "
                    f"stft_window={sampler_cfg.stft_window_size} source_fps={fps:.1f}"
                )

            if isinstance(source, int):
                actual_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                actual_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
                actual_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
                logger.info(
                    f"摄像头已打开: source={source} "
                    f"width={actual_width} height={actual_height} fps={actual_fps:.2f}"
                )

            while not self._stop_event.is_set():
                if max_frames is not None and frame_index >= max_frames:
                    break

                ok, frame = capture.read()
                if not ok:
                    break

                timestamp = frame_index / fps
                if isinstance(source, int):
                    timestamp = time.monotonic() - started_at

                if duration_seconds is not None and timestamp >= duration_seconds:
                    break

                if self._adaptive_sampler is not None and not self._adaptive_sampler.should_sample(frame_index, timestamp):
                    if not self._adaptive_sampler.precheck_skipped_frame(frame):
                        frame_index += 1
                        continue
                    self._adaptive_sampler.mark_sampled(frame, timestamp)

                if save_video_path is not None:
                    if writer is None:
                        writer = _open_video_writer(
                            save_video_path,
                            frame.shape[1],
                            frame.shape[0],
                            fps,
                        )
                        logger.info(f"正在保存视频到: {Path(save_video_path)}")
                    writer.write(frame)

                packet = FramePacket(
                    frame_index=frame_index,
                    timestamp=timestamp,
                    frame_bgr=frame,
                )
                try:
                    frame_queue.put(packet, timeout=0.5)
                except queue.Full:
                    logger.warning("Visual frame queue is full, dropping a frame")
                if self._adaptive_sampler is not None:
                    self._adaptive_sampler.mark_sampled(frame, timestamp)
                frame_index += 1
        except Exception as exc:
            self._record_thread_error(exc)
        finally:
            if writer is not None:
                writer.release()
            if capture is not None:
                capture.release()
            _push_queue_sentinel(frame_queue)

    def _detector_loop(
        self,
        frame_queue: "queue.Queue[Optional[FramePacket]]",
        observation_queue: "queue.Queue[Optional[FrameObservation]]",
    ) -> None:
        try:
            detector_state = self._build_detector_state()
            while not self._stop_event.is_set():
                packet = frame_queue.get()
                if packet is None:
                    return
                observation = self._analyze_frame(packet, detector_state)
                if self._adaptive_sampler is not None:
                    self._adaptive_sampler.update(
                        change_score=observation.change_score,
                        smoothed_score=observation.smoothed_score,
                        timestamp=observation.timestamp,
                    )
                observation_queue.put(observation)
        except Exception as exc:
            self._record_thread_error(exc)
        finally:
            _push_queue_sentinel(observation_queue)

    def _selector_loop(
        self,
        observation_queue: "queue.Queue[Optional[FrameObservation]]",
        event_queue: "queue.Queue[Optional[VisualEvent]]",
    ) -> None:
        clip_half = self.config.clip_duration_seconds / 2.0
        cooldown = self.config.peak_cooldown_seconds
        if clip_half > 0:
            cooldown = max(cooldown, self.config.clip_duration_seconds)
        selector = TemporalPeakSelector(
            peak_threshold=self.config.peak_threshold,
            cooldown_seconds=cooldown,
            window_seconds=self.config.event_window_seconds,
            peak_neighborhood_frames=self.config.peak_neighborhood_frames,
            clip_half_duration=clip_half,
        )
        try:
            while not self._stop_event.is_set():
                observation = observation_queue.get()
                if observation is None:
                    # Stream ended — flush any pending clip peak.
                    flushed = selector.flush()
                    if flushed is not None:
                        event = self._build_visual_event(flushed)
                        event_queue.put(event)
                    return

                selection = selector.consume(observation)
                if selection is None:
                    continue

                event = self._build_visual_event(selection)
                event_queue.put(event)
        except Exception as exc:
            self._record_thread_error(exc)
        finally:
            _push_queue_sentinel(event_queue)

    def _uploader_loop(
        self,
        event_queue: "queue.Queue[Optional[VisualEvent]]",
    ) -> None:
        try:
            while not self._stop_event.is_set():
                event = event_queue.get()
                if event is None:
                    return

                if self.config.vision_analysis_mode == "per_event":
                    analysis, rate_limited = self._analyze_event_if_allowed(event)
                    if analysis is not None:
                        event.analysis = analysis
                    event.rate_limited = rate_limited
                    if rate_limited:
                        logger.info(
                            "Visual event analysis skipped because the Vision LLM rate limit was reached"
                        )

                with self._events_lock:
                    self._collected_events.append(event)

                if self.event_callback is not None:
                    try:
                        self.event_callback(event)
                    except Exception as exc:
                        logger.warning(f"Visual event callback failed: {exc}")

                update = self._monitor.consume_candidate(
                    event,
                    analyze_callback=self._analyze_event_if_allowed,
                )
                if any(item.rate_limited for item in update.promoted_events):
                    logger.info(
                        "Visual trigger analysis skipped because the Vision LLM rate limit was reached"
                    )
                self._publish_monitor_update(update)
        except Exception as exc:
            self._record_thread_error(exc)

    def _build_detector_state(self) -> _DetectorState:
        morphology_kernel = np.ones(
            (
                _odd_kernel(self.config.morphology_kernel_size),
                _odd_kernel(self.config.morphology_kernel_size),
            ),
            dtype=np.uint8,
        )
        subtractor = cv2.createBackgroundSubtractorMOG2(
            history=self.config.mog2_history,
            varThreshold=self.config.mog2_var_threshold,
            detectShadows=self.config.mog2_detect_shadows,
        )
        return _DetectorState(
            subtractor=subtractor,
            morphology_kernel=morphology_kernel,
            temporal_votes=deque(maxlen=self.config.temporal_vote_window),
        )

    def _analyze_frame(
        self,
        packet: FramePacket,
        state: _DetectorState,
    ) -> FrameObservation:
        gaussian_kernel = _odd_kernel(self.config.gaussian_kernel_size)
        resized = _fit_width(packet.frame_bgr, self.config.resize_width)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (gaussian_kernel, gaussian_kernel), 0)

        foreground = state.subtractor.apply(blurred)
        if self.config.mog2_detect_shadows:
            _, foreground = cv2.threshold(foreground, 200, 255, cv2.THRESH_BINARY)

        foreground = cv2.erode(foreground, state.morphology_kernel, iterations=1)
        foreground = cv2.dilate(foreground, state.morphology_kernel, iterations=1)
        _, foreground_ratio = self._filter_small_components(foreground)

        histogram = cv2.calcHist([blurred], [0], None, [32], [0, 256])
        cv2.normalize(histogram, histogram)

        edges = cv2.Canny(
            blurred,
            self.config.canny_threshold1,
            self.config.canny_threshold2,
        )
        edge_density = float(np.count_nonzero(edges)) / float(edges.size)

        histogram_distance = 0.0
        edge_change = 0.0
        if state.previous_histogram is not None:
            histogram_distance = float(
                cv2.compareHist(
                    histogram,
                    state.previous_histogram,
                    cv2.HISTCMP_BHATTACHARYYA,
                )
            )
        if state.previous_edge_density is not None:
            edge_change = abs(edge_density - state.previous_edge_density)

        s1 = sigmoid_normalize(
            foreground_ratio,
            self.config.area_sigmoid_center,
            self.config.area_sigmoid_scale,
        )
        s2 = sigmoid_normalize(
            histogram_distance,
            self.config.histogram_sigmoid_center,
            self.config.histogram_sigmoid_scale,
        )
        s3 = sigmoid_normalize(
            edge_change,
            self.config.edge_sigmoid_center,
            self.config.edge_sigmoid_scale,
        )

        change_score = (
            self.config.area_weight * s1
            + self.config.histogram_weight * s2
            + self.config.edge_weight * s3
        )
        smoothed_score = (
            self.config.ema_alpha * change_score
            + (1.0 - self.config.ema_alpha) * state.previous_smoothed_score
        )

        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        motion_flag = foreground_ratio >= self.config.min_component_area_ratio
        state.temporal_votes.append(motion_flag)
        motion_vote = (
            sum(1 for item in state.temporal_votes if item)
            >= self.config.temporal_vote_required
        )

        state.previous_gray = blurred
        state.previous_histogram = histogram
        state.previous_edge_density = edge_density
        state.previous_smoothed_score = smoothed_score

        stored_frame = resized if self.config.clip_duration_seconds > 0 else packet.frame_bgr
        height, width = stored_frame.shape[:2]
        return FrameObservation(
            frame_index=packet.frame_index,
            timestamp=packet.timestamp,
            frame_bgr=stored_frame,
            frame_width=width,
            frame_height=height,
            foreground_ratio=foreground_ratio,
            histogram_distance=histogram_distance,
            edge_change=edge_change,
            change_score=change_score,
            smoothed_score=smoothed_score,
            sharpness=sharpness,
            motion_vote=motion_vote,
        )

    def _filter_small_components(self, mask: np.ndarray) -> tuple[np.ndarray, float]:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        total_pixels = float(mask.shape[0] * mask.shape[1])
        min_area_pixels = max(1, int(total_pixels * self.config.min_component_area_ratio))
        filtered = np.zeros_like(mask)
        kept_pixels = 0

        for label_index in range(1, num_labels):
            area = int(stats[label_index, cv2.CC_STAT_AREA])
            if area < min_area_pixels:
                continue
            filtered[labels == label_index] = 255
            kept_pixels += area

        return filtered, kept_pixels / total_pixels

    def _build_visual_event(self, selection: PeakSelection) -> VisualEvent:
        if self.config.clip_duration_seconds > 0:
            keyframe_observations = self._select_clip_frames(selection)
        else:
            keyframe_observations = self._select_keyframes(selection)
        keyframes = [
            _encode_frame_as_image_result(
                observation.frame_bgr,
                observation.frame_index,
                observation.timestamp,
                self.config.jpeg_quality,
            )
            for observation in keyframe_observations
        ]

        metrics = {
            "foreground_ratio": selection.peak.foreground_ratio,
            "histogram_distance": selection.peak.histogram_distance,
            "edge_change": selection.peak.edge_change,
            "change_score": selection.peak.change_score,
            "smoothed_score": selection.peak.smoothed_score,
            "sharpness": selection.representative.sharpness,
            "window_size": len(selection.window),
            "clip_duration": self.config.clip_duration_seconds,
        }

        return VisualEvent(
            event_id=f"visual-{selection.peak.frame_index}",
            peak_frame_index=selection.peak.frame_index,
            timestamp=selection.peak.timestamp,
            peak_score=selection.peak.smoothed_score,
            representative_frame_index=selection.representative.frame_index,
            keyframes=keyframes,
            metrics=metrics,
        )

    def _select_keyframes(self, selection: PeakSelection) -> List[FrameObservation]:
        unique: List[FrameObservation] = []
        earliest = selection.window[0]
        latest = selection.window[-1]
        representative = selection.representative

        for candidate in (earliest, representative, latest):
            if any(item.frame_index == candidate.frame_index for item in unique):
                continue
            unique.append(candidate)

        unique.sort(key=lambda item: item.frame_index)
        return unique[: self.config.max_keyframes_per_event]

    def _select_clip_frames(self, selection: PeakSelection) -> List[FrameObservation]:
        window = selection.window
        if not window:
            return []
        max_frames = min(self.config.clip_max_frames, len(window))
        if max_frames >= len(window):
            return sorted(window, key=lambda o: o.timestamp)

        t_start = window[0].timestamp
        t_end = window[-1].timestamp
        if t_end <= t_start:
            return window[:max_frames]

        step = (t_end - t_start) / max_frames
        selected: List[FrameObservation] = []
        seen_indices: set = set()
        for i in range(max_frames):
            target_t = t_start + step * (i + 0.5)
            best = min(
                window,
                key=lambda o: (abs(o.timestamp - target_t), -o.sharpness),
            )
            if best.frame_index not in seen_indices:
                seen_indices.add(best.frame_index)
                selected.append(best)

        selected.sort(key=lambda o: o.timestamp)
        return selected


__all__ = [
    "TemporalPeakSelector",
    "VisualPerceptionPipeline",
]
