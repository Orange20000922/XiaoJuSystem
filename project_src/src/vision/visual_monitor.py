from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, List, Optional, Sequence

from .visual_text import default_visual_event_text
from .visual_types import VisualAnalysis, VisualEvent, VisualPerceptionConfig


@dataclass
class VisualEventCluster:
    """时间上相邻的候选视觉事件聚类结果。"""

    cluster_id: str
    start_timestamp: float
    end_timestamp: float
    events: List[VisualEvent] = field(default_factory=list)
    peak_event: Optional[VisualEvent] = None
    analysis: Optional[VisualAnalysis] = None
    rate_limited: bool = False

    @property
    def event_count(self) -> int:
        return len(self.events)

    @property
    def peak_score(self) -> float:
        if self.peak_event is None:
            return 0.0
        return float(self.peak_event.peak_score)

    @property
    def accumulated_score(self) -> float:
        return float(sum(item.peak_score for item in self.events))

    @property
    def duration_seconds(self) -> float:
        return max(0.0, float(self.end_timestamp - self.start_timestamp))

    @property
    def salience_score(self) -> float:
        peak = self.peak_score
        accumulated = min(self.accumulated_score, 3.0)
        event_bonus = min(float(self.event_count), 5.0) * 0.05
        return peak + 0.35 * accumulated + event_bonus

    def summary_text(self) -> str:
        if self.analysis is not None:
            return self.analysis.to_summary_text()
        if self.peak_event is not None and self.peak_event.analysis is not None:
            return self.peak_event.analysis.to_summary_text()
        if self.peak_event is not None:
            return default_visual_event_text(self.peak_event)
        return "检测到视觉片段"

    def representative_event(self) -> Optional[VisualEvent]:
        if not self.events:
            return None
        return max(
            self.events,
            key=lambda item: (
                float(item.metrics.get("sharpness", 0.0)),
                item.peak_score,
            ),
        )

    def merged_keyframes(self, max_keyframes: int):
        if max_keyframes <= 0:
            return []

        candidates: List[VisualEvent] = []
        if self.peak_event is not None:
            candidates.append(self.peak_event)

        representative = self.representative_event()
        if representative is not None and representative not in candidates:
            candidates.append(representative)

        ranked_events = sorted(
            self.events,
            key=lambda item: (
                item.peak_score,
                float(item.metrics.get("sharpness", 0.0)),
            ),
            reverse=True,
        )
        for event in ranked_events:
            if event not in candidates:
                candidates.append(event)

        merged = []
        seen_urls = set()
        for event in candidates:
            for image in event.keyframes:
                key = getattr(image, "original_url", None) or id(image)
                if key in seen_urls:
                    continue
                merged.append(image)
                seen_urls.add(key)
                if len(merged) >= max_keyframes:
                    return merged
        return merged

    def to_visual_event(
        self,
        *,
        mode: str,
        config: Optional[VisualPerceptionConfig] = None,
        analysis: Optional[VisualAnalysis] = None,
        rate_limited: bool = False,
    ) -> VisualEvent:
        cfg = config or VisualPerceptionConfig()
        peak = self.peak_event
        if peak is None:
            raise ValueError("cluster has no peak_event")

        merged_metrics = dict(peak.metrics)
        merged_metrics.update(
            {
                "mode": mode,
                "segment_start_timestamp": self.start_timestamp,
                "segment_end_timestamp": self.end_timestamp,
                "segment_duration_seconds": self.duration_seconds,
                "segment_event_count": self.event_count,
                "segment_accumulated_score": round(self.accumulated_score, 4),
                "segment_salience_score": round(self.salience_score, 4),
                "segment_source_event_ids": [item.event_id for item in self.events],
            }
        )

        return VisualEvent(
            event_id=f"{mode}-{self.cluster_id}",
            peak_frame_index=peak.peak_frame_index,
            timestamp=peak.timestamp,
            peak_score=peak.peak_score,
            representative_frame_index=peak.representative_frame_index,
            keyframes=self.merged_keyframes(cfg.max_keyframes_per_event),
            metrics=merged_metrics,
            analysis=analysis or self.analysis or peak.analysis,
            rate_limited=rate_limited or self.rate_limited,
        )


@dataclass
class VisualWindowSummary:
    """长时间窗内的视觉摘要结果。"""

    window_index: int
    start_timestamp: float
    end_timestamp: float
    top_clusters: List[VisualEventCluster] = field(default_factory=list)
    total_events: int = 0
    total_clusters: int = 0
    total_accumulated_score: float = 0.0

    def to_text(self) -> str:
        if not self.top_clusters:
            return f"{self.start_timestamp:.1f}-{self.end_timestamp:.1f}s 内未检测到显著视觉片段"

        parts = []
        for index, cluster in enumerate(self.top_clusters, start=1):
            parts.append(
                f"{index}. {cluster.summary_text()} "
                f"(峰值={cluster.peak_score:.2f}, 次数={cluster.event_count})"
            )
        return (
            f"{self.start_timestamp:.1f}-{self.end_timestamp:.1f}s "
            f"共 {self.total_clusters} 个片段，保留 {len(self.top_clusters)} 个："
            + " ".join(parts)
        )


@dataclass
class VisualMonitorUpdate:
    promoted_events: List[VisualEvent] = field(default_factory=list)
    completed_summaries: List[VisualWindowSummary] = field(default_factory=list)


def cluster_visual_events(
    events: Sequence[VisualEvent],
    merge_gap_seconds: float,
) -> List[VisualEventCluster]:
    if not events:
        return []

    sorted_events = sorted(events, key=lambda item: item.timestamp)
    clusters: List[VisualEventCluster] = []
    current_events: List[VisualEvent] = [sorted_events[0]]

    def _build_cluster(items: List[VisualEvent]) -> VisualEventCluster:
        peak_event = max(
            items,
            key=lambda item: (
                item.peak_score,
                float(item.metrics.get("sharpness", 0.0)),
            ),
        )
        start_timestamp = items[0].timestamp
        end_timestamp = items[-1].timestamp
        return VisualEventCluster(
            cluster_id=f"{int(start_timestamp * 1000)}-{peak_event.peak_frame_index}",
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
            events=list(items),
            peak_event=peak_event,
        )

    for event in sorted_events[1:]:
        gap = event.timestamp - current_events[-1].timestamp
        if gap <= merge_gap_seconds:
            current_events.append(event)
            continue
        clusters.append(_build_cluster(current_events))
        current_events = [event]

    clusters.append(_build_cluster(current_events))
    return clusters


def summarize_visual_event_windows(
    events: Sequence[VisualEvent],
    *,
    window_seconds: float,
    top_k: int,
    merge_gap_seconds: float,
) -> List[VisualWindowSummary]:
    if not events or window_seconds <= 0 or top_k <= 0:
        return []

    windows: dict[int, List[VisualEvent]] = {}
    for event in sorted(events, key=lambda item: item.timestamp):
        window_index = int(event.timestamp // window_seconds)
        windows.setdefault(window_index, []).append(event)

    summaries: List[VisualWindowSummary] = []
    for window_index in sorted(windows.keys()):
        window_events = windows[window_index]
        clusters = cluster_visual_events(window_events, merge_gap_seconds)
        ranked_clusters = sorted(
            clusters,
            key=lambda item: (
                item.salience_score,
                item.peak_score,
                item.accumulated_score,
            ),
            reverse=True,
        )
        start_timestamp = window_index * window_seconds
        end_timestamp = start_timestamp + window_seconds
        summaries.append(
            VisualWindowSummary(
                window_index=window_index,
                start_timestamp=start_timestamp,
                end_timestamp=end_timestamp,
                top_clusters=ranked_clusters[:top_k],
                total_events=len(window_events),
                total_clusters=len(clusters),
                total_accumulated_score=sum(item.peak_score for item in window_events),
            )
        )
    return summaries


class VisualEventMonitor:
    """弱监控缓冲、触发式升级和窗口摘要的上层策略层。"""

    def __init__(self, config: Optional[VisualPerceptionConfig] = None):
        self.config = config or VisualPerceptionConfig()
        self._recent_events: Deque[VisualEvent] = deque()
        self._promoted_peak_event_ids: Deque[str] = deque(maxlen=128)
        self._promoted_events: List[VisualEvent] = []
        self._completed_summaries: List[VisualWindowSummary] = []
        self._current_summary_window_index: Optional[int] = None
        self._current_summary_events: List[VisualEvent] = []
        self._last_trigger_time: float = -1e9

    @property
    def recent_events(self) -> List[VisualEvent]:
        return list(self._recent_events)

    @property
    def promoted_events(self) -> List[VisualEvent]:
        return list(self._promoted_events)

    @property
    def completed_summaries(self) -> List[VisualWindowSummary]:
        return list(self._completed_summaries)

    def consume_candidate(
        self,
        event: VisualEvent,
        *,
        analyze_callback: Optional[
            Callable[[VisualEvent], tuple[Optional[VisualAnalysis], bool]]
        ] = None,
    ) -> VisualMonitorUpdate:
        update = VisualMonitorUpdate()
        self._recent_events.append(event)
        self._prune_recent_events(event.timestamp)

        if self.config.summary_enabled:
            update.completed_summaries.extend(self._consume_summary_window(event))

        if (
            self.config.trigger_analysis_enabled
            and self.config.vision_analysis_mode == "triggered"
        ):
            promoted = self._maybe_promote_recent_segment(
                now=event.timestamp,
                mode="trigger",
                analyze_callback=analyze_callback,
            )
            if promoted is not None:
                self._promoted_events.append(promoted)
                update.promoted_events.append(promoted)

        return update

    def recent_clusters(
        self,
        window_seconds: Optional[float] = None,
    ) -> List[VisualEventCluster]:
        events = self.recent_events
        if window_seconds is not None:
            if not events:
                return []
            cutoff = events[-1].timestamp - window_seconds
            events = [item for item in events if item.timestamp >= cutoff]
        return cluster_visual_events(events, self.config.segment_merge_gap_seconds)

    def analyze_recent_buffer(
        self,
        *,
        top_k: Optional[int] = None,
        analyze_callback: Optional[
            Callable[[VisualEvent], tuple[Optional[VisualAnalysis], bool]]
        ] = None,
    ) -> List[VisualEvent]:
        clusters = sorted(
            self.recent_clusters(),
            key=lambda item: (
                item.salience_score,
                item.peak_score,
                item.accumulated_score,
            ),
            reverse=True,
        )
        limit = top_k or self.config.explicit_request_top_k
        promoted: List[VisualEvent] = []
        for cluster in clusters[: max(0, limit)]:
            promoted_event = self._cluster_to_promoted_event(
                cluster,
                mode="manual",
                analyze_callback=analyze_callback,
            )
            promoted.append(promoted_event)
        return promoted

    def finalize(self) -> VisualMonitorUpdate:
        update = VisualMonitorUpdate()
        if self.config.summary_enabled:
            summary = self._finalize_current_summary_window()
            if summary is not None:
                self._completed_summaries.append(summary)
                update.completed_summaries.append(summary)
        return update

    def _consume_summary_window(self, event: VisualEvent) -> List[VisualWindowSummary]:
        window_seconds = self.config.summary_window_seconds
        if window_seconds <= 0:
            return []

        window_index = int(event.timestamp // window_seconds)
        completed: List[VisualWindowSummary] = []
        if self._current_summary_window_index is None:
            self._current_summary_window_index = window_index
        elif window_index != self._current_summary_window_index:
            summary = self._finalize_current_summary_window()
            if summary is not None:
                self._completed_summaries.append(summary)
                completed.append(summary)
            self._current_summary_window_index = window_index

        self._current_summary_events.append(event)
        return completed

    def _finalize_current_summary_window(self) -> Optional[VisualWindowSummary]:
        if (
            self._current_summary_window_index is None
            or not self._current_summary_events
            or self.config.summary_top_k <= 0
        ):
            self._current_summary_events = []
            return None

        window_seconds = self.config.summary_window_seconds
        window_index = self._current_summary_window_index
        window_events = list(self._current_summary_events)
        clusters = cluster_visual_events(window_events, self.config.segment_merge_gap_seconds)
        ranked_clusters = sorted(
            clusters,
            key=lambda item: (
                item.salience_score,
                item.peak_score,
                item.accumulated_score,
            ),
            reverse=True,
        )

        summary = VisualWindowSummary(
            window_index=window_index,
            start_timestamp=window_index * window_seconds,
            end_timestamp=(window_index + 1) * window_seconds,
            top_clusters=ranked_clusters[: self.config.summary_top_k],
            total_events=len(window_events),
            total_clusters=len(clusters),
            total_accumulated_score=sum(item.peak_score for item in window_events),
        )
        self._current_summary_events = []
        return summary

    def _prune_recent_events(self, now: float) -> None:
        cutoff = now - max(0.0, self.config.weak_monitor_buffer_seconds)
        while self._recent_events and self._recent_events[0].timestamp < cutoff:
            self._recent_events.popleft()

    def _maybe_promote_recent_segment(
        self,
        *,
        now: float,
        mode: str,
        analyze_callback: Optional[
            Callable[[VisualEvent], tuple[Optional[VisualAnalysis], bool]]
        ],
    ) -> Optional[VisualEvent]:
        if (now - self._last_trigger_time) < self.config.trigger_refractory_seconds:
            return None

        cutoff = now - self.config.trigger_window_seconds
        window_events = [item for item in self._recent_events if item.timestamp >= cutoff]
        if not window_events:
            return None

        clusters = cluster_visual_events(window_events, self.config.segment_merge_gap_seconds)
        strong_clusters = [
            item
            for item in clusters
            if item.peak_score >= self.config.trigger_peak_score_threshold
        ]
        if not strong_clusters:
            return None

        accumulated_score = sum(item.peak_score for item in window_events)
        if (
            accumulated_score < self.config.trigger_accumulated_score_threshold
            and len(strong_clusters) < self.config.trigger_min_strong_events
        ):
            return None

        best_cluster = max(
            strong_clusters,
            key=lambda item: (
                item.salience_score,
                item.peak_score,
                item.accumulated_score,
            ),
        )
        peak_event_id = best_cluster.peak_event.event_id if best_cluster.peak_event else ""
        if peak_event_id in self._promoted_peak_event_ids:
            return None

        promoted_event = self._cluster_to_promoted_event(
            best_cluster,
            mode=mode,
            analyze_callback=analyze_callback,
        )
        self._promoted_peak_event_ids.append(peak_event_id)
        self._last_trigger_time = now
        return promoted_event

    def _cluster_to_promoted_event(
        self,
        cluster: VisualEventCluster,
        *,
        mode: str,
        analyze_callback: Optional[
            Callable[[VisualEvent], tuple[Optional[VisualAnalysis], bool]]
        ],
    ) -> VisualEvent:
        promoted_event = cluster.to_visual_event(mode=mode, config=self.config)
        if analyze_callback is not None:
            analysis, rate_limited = analyze_callback(promoted_event)
            if analysis is not None:
                cluster.analysis = analysis
                promoted_event.analysis = analysis
            promoted_event.rate_limited = rate_limited
            cluster.rate_limited = rate_limited
        return promoted_event


__all__ = [
    "VisualEventCluster",
    "VisualEventMonitor",
    "VisualMonitorUpdate",
    "VisualWindowSummary",
    "cluster_visual_events",
    "summarize_visual_event_windows",
]
