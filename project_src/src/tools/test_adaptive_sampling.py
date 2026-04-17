"""自适应帧采样 dry-run：对比开启/关闭 adaptive 的帧数、事件数、fps 曲线。

不调用 GLM-4V，纯本地 CV 分析。输出时序数据用于验证 FFT 行为。
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from typing import List, Tuple

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from src.vision import (
    CV2_AVAILABLE,
    VisualPerceptionConfig,
    VisualPerceptionPipeline,
)

VIDEO = project_root / "data" / "test_adaptive.avi"


def _make_config(adaptive: bool, spike_threshold: float = 0.6, fps_min: float = 2.0,
                 precheck_diff_threshold: float = 30.0) -> VisualPerceptionConfig:
    cfg = VisualPerceptionConfig()
    cfg.vision_analysis_mode = "none"
    cfg.summary_enabled = False
    cfg.clip_duration_seconds = 0.0
    cfg.adaptive_sampling_enabled = adaptive
    cfg.adaptive_stft_window_size = 32
    cfg.adaptive_fps_min = fps_min
    cfg.adaptive_fps_max = 15.0
    cfg.adaptive_spike_threshold = spike_threshold
    cfg.adaptive_precheck_diff_threshold = precheck_diff_threshold
    return cfg


def _run(adaptive: bool, spike_threshold: float = 0.6, fps_min: float = 2.0,
         label_suffix: str = "", precheck_diff_threshold: float = 30.0) -> dict:
    cfg = _make_config(adaptive, spike_threshold=spike_threshold, fps_min=fps_min,
                       precheck_diff_threshold=precheck_diff_threshold)

    events_detail: List[dict] = []

    def _on_event(event):
        events_detail.append({
            "timestamp": event.timestamp,
            "frame_index": event.peak_frame_index,
            "score": event.peak_score,
        })

    runner = VisualPerceptionPipeline(
        config=cfg,
        analyzer=None,
        event_callback=_on_event,
    )

    # Poll sampler state in background for fps trace
    fps_trace: List[Tuple[float, float, float]] = []  # (wallclock, target_fps, R_high)
    stop_probe = threading.Event()

    def _probe():
        t_start = time.time()
        while not stop_probe.is_set():
            if runner._adaptive_sampler is not None:
                fps_trace.append((
                    time.time() - t_start,
                    runner._adaptive_sampler.current_target_fps,
                    runner._adaptive_sampler.spectral_activity_ratio,
                ))
            time.sleep(0.1)

    label = ("adaptive" if adaptive else "baseline") + label_suffix
    print(f"\n{'=' * 60}\n  {label} 模式 (adaptive={adaptive} spike_th={spike_threshold} fps_min={fps_min})\n{'=' * 60}")

    probe_thread = threading.Thread(target=_probe, daemon=True)
    if adaptive:
        probe_thread.start()

    t0 = time.time()
    events = runner.run(str(VIDEO))
    elapsed = time.time() - t0
    stop_probe.set()

    sampler = runner._adaptive_sampler
    result = {
        "mode": label,
        "elapsed_seconds": elapsed,
        "events": len(events),
        "events_detail": events_detail,
        "fps_trace": fps_trace if adaptive else [],
    }
    if sampler is not None:
        result["final_target_fps"] = sampler.current_target_fps
        result["final_spectral_ratio"] = sampler.spectral_activity_ratio
        result["precheck_force_count"] = sampler.precheck_force_count

    print(f"  总处理事件数: {len(events)}")
    print(f"  运行耗时: {elapsed:.1f}s")
    if sampler is not None:
        print(f"  最终 target_fps: {sampler.current_target_fps:.2f}")
        print(f"  最终 R_high: {sampler.spectral_activity_ratio:.3f}")
        print(f"  预检测强制采样次数: {sampler.precheck_force_count}")

    return result


def _compare_events(baseline_events: List[dict], adaptive_events: List[dict], tolerance_sec: float = 1.5):
    """找出 baseline 中有但 adaptive 没对应的事件（可能的误伤）。"""
    matched = set()
    missed = []
    extra_adaptive = []

    for b_event in baseline_events:
        b_ts = b_event["timestamp"]
        best_match = None
        best_delta = float("inf")
        for i, a_event in enumerate(adaptive_events):
            if i in matched:
                continue
            delta = abs(a_event["timestamp"] - b_ts)
            if delta < best_delta and delta <= tolerance_sec:
                best_delta = delta
                best_match = i
        if best_match is not None:
            matched.add(best_match)
        else:
            missed.append(b_event)

    for i, a_event in enumerate(adaptive_events):
        if i not in matched:
            extra_adaptive.append(a_event)

    return missed, extra_adaptive


def _print_fps_timeline(fps_trace: List[Tuple[float, float, float]], buckets: int = 20):
    """将 fps_trace 聚合为 N 个桶打印"""
    if not fps_trace:
        print("  (无 fps 轨迹数据)")
        return
    total = fps_trace[-1][0]
    if total <= 0:
        return
    print(f"\n  fps 时序 (每桶 ~{total / buckets:.1f}s, 横线=fps 范围 [2, 15]):")
    bucket_size = total / buckets
    for i in range(buckets):
        t_start = i * bucket_size
        t_end = (i + 1) * bucket_size
        samples = [row for row in fps_trace if t_start <= row[0] < t_end]
        if not samples:
            continue
        avg_fps = sum(s[1] for s in samples) / len(samples)
        avg_rhigh = sum(s[2] for s in samples) / len(samples)
        bar_width = int((avg_fps - 2) / 13 * 30)
        bar = "█" * max(bar_width, 0) + "·" * (30 - max(bar_width, 0))
        print(f"  t={t_start:>5.1f}s  fps={avg_fps:>5.2f}  R_high={avg_rhigh:.3f}  |{bar}|")


def _print_events(events_detail: List[dict], label: str, limit: int = None):
    print(f"\n  {label} 事件列表 (共 {len(events_detail)} 个):")
    to_print = events_detail if limit is None else events_detail[:limit]
    for e in to_print:
        print(f"    t={e['timestamp']:>6.2f}s  frame={e['frame_index']:>5}  score={e['score']:.3f}")
    if limit is not None and len(events_detail) > limit:
        print(f"    ... (省略 {len(events_detail) - limit} 条)")


def main() -> int:
    if not CV2_AVAILABLE:
        print("ERROR: OpenCV 不可用")
        return 1
    if not VIDEO.exists():
        print(f"ERROR: 视频不存在: {VIDEO}")
        return 1

    print(f"视频源: {VIDEO}")
    baseline = _run(adaptive=False)
    runs = [
        ("default",          _run(adaptive=True, spike_threshold=0.6, fps_min=2.0, label_suffix="-default")),
        ("spike=0.4",        _run(adaptive=True, spike_threshold=0.4, fps_min=2.0, label_suffix="-spike04")),
        ("spike=0.4+fps4",   _run(adaptive=True, spike_threshold=0.4, fps_min=4.0, label_suffix="-spike04-fps4")),
        ("precheck=15+fps4", _run(adaptive=True, spike_threshold=0.4, fps_min=4.0, label_suffix="-pc15-fps4",
                                  precheck_diff_threshold=15.0)),
    ]

    print(f"\n{'=' * 60}\n  对比汇总\n{'=' * 60}")
    print(f"  {'配置':<22}  {'events':>6}  {'elapsed':>9}  {'加速比':>8}  {'保留率':>8}")
    print(f"  {'baseline':<22}  {baseline['events']:>6}  {baseline['elapsed_seconds']:>7.1f}s  {'1.00x':>8}  {'100.0%':>8}")
    for name, r in runs:
        sp = baseline['elapsed_seconds'] / max(r['elapsed_seconds'], 0.01)
        er = r['events'] / max(baseline['events'], 1)
        print(f"  adaptive-{name:<14}  {r['events']:>6}  {r['elapsed_seconds']:>7.1f}s  {sp:>7.2f}x  {er*100:>7.1f}%")

    # 对最激进的配置做详细诊断
    name, adaptive = runs[-1]
    print(f"\n{'=' * 60}\n  详细诊断: adaptive-{name}\n{'=' * 60}")

    print(f"\n{'=' * 60}\n  fps 变化轨迹 (adaptive 模式)\n{'=' * 60}")
    _print_fps_timeline(adaptive["fps_trace"])

    print(f"\n{'=' * 60}\n  事件时间分布对比\n{'=' * 60}")

    # 按显著度分层看 baseline
    bl_events = baseline["events_detail"]
    ad_events = adaptive["events_detail"]

    missed, extra = _compare_events(bl_events, ad_events, tolerance_sec=1.5)
    print(f"\n  adaptive 丢失的 baseline 事件 (baseline 有但 adaptive 没匹配到, 1.5s 容差):")
    if not missed:
        print("    (无)")
    else:
        # 按 score 分组显示
        high_missed = [e for e in missed if e["score"] >= 0.3]
        mid_missed = [e for e in missed if 0.2 <= e["score"] < 0.3]
        low_missed = [e for e in missed if e["score"] < 0.2]
        print(f"    高显著 (score>=0.3): {len(high_missed)} 个 ← 关键！")
        for e in high_missed:
            print(f"      t={e['timestamp']:>6.2f}s  frame={e['frame_index']:>5}  score={e['score']:.3f}")
        print(f"    中显著 (0.2<=score<0.3): {len(mid_missed)} 个")
        for e in mid_missed[:5]:
            print(f"      t={e['timestamp']:>6.2f}s  frame={e['frame_index']:>5}  score={e['score']:.3f}")
        if len(mid_missed) > 5:
            print(f"      ... (省略 {len(mid_missed) - 5} 条)")
        print(f"    低显著 (score<0.2): {len(low_missed)} 个 ← 可能为冗余")

    print(f"\n  adaptive 独有事件 (可能因为采样点不同导致的新事件):")
    if not extra:
        print("    (无)")
    else:
        for e in extra[:10]:
            print(f"    t={e['timestamp']:>6.2f}s  frame={e['frame_index']:>5}  score={e['score']:.3f}")
        if len(extra) > 10:
            print(f"    ... (省略 {len(extra) - 10} 条)")

    # 事件 peak_score 分布对比
    print(f"\n{'=' * 60}\n  显著度分布对比\n{'=' * 60}")
    def _score_stats(events, name):
        if not events:
            return
        scores = [e["score"] for e in events]
        scores.sort()
        n = len(scores)
        mean = sum(scores) / n
        p50 = scores[n // 2]
        p90 = scores[min(n - 1, int(n * 0.9))]
        p_max = scores[-1]
        print(f"  {name}: n={n}  mean={mean:.3f}  p50={p50:.3f}  p90={p90:.3f}  max={p_max:.3f}")
    _score_stats(bl_events, "baseline")
    _score_stats(ad_events, "adaptive")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
