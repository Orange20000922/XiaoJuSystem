import argparse
import json
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.logger import logger
from configs.config_loader import AppConfig
from src.core_engine import NeuroLikePipeline
from src.vision import (
    CV2_AVAILABLE,
    VisualPerceptionConfig,
    VisualPerceptionPipeline,
    build_visual_analyzer_from_pipeline,
    derive_visual_emotion_signal,
    visual_event_to_agent_text,
)


def _parse_source(raw: str):
    if raw.isdigit():
        return int(raw)
    return raw


def main() -> int:
    parser = argparse.ArgumentParser(description="运行动态视觉感知 MVP 原型。")
    parser.add_argument(
        "--source",
        default="0",
        help="视频源。可传摄像头索引（如 0）或视频文件路径。",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="可选的 config.json 路径。提供后可启用 Vision LLM 分析。",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="可选的最大处理帧数，便于离线调试。",
    )
    parser.add_argument(
        "--duration-seconds",
        type=float,
        default=None,
        help="可选的处理时长，适合摄像头源。与 --max-frames 同时提供时，任一条件满足即停止。",
    )
    parser.add_argument(
        "--save-video",
        default=None,
        help="可选的输出视频路径。提供后会把摄像头或输入视频的原始帧另存一份。",
    )
    parser.add_argument(
        "--camera-width",
        type=int,
        default=None,
        help="可选的摄像头目标宽度，只对整数型摄像头 source 生效。",
    )
    parser.add_argument(
        "--camera-height",
        type=int,
        default=None,
        help="可选的摄像头目标高度，只对整数型摄像头 source 生效。",
    )
    parser.add_argument(
        "--camera-fps",
        type=float,
        default=None,
        help="可选的摄像头目标 FPS，只对整数型摄像头 source 生效。",
    )
    parser.add_argument(
        "--analysis-mode",
        choices=("none", "triggered", "per_event"),
        default=None,
        help="视觉分析策略覆盖。none=只跑本地 CV，triggered=触发式升级，per_event=逐事件分析。",
    )
    parser.add_argument(
        "--summary-window-seconds",
        type=float,
        default=None,
        help="可选的摘要窗口长度覆盖，适合长视频摘要。",
    )
    parser.add_argument(
        "--summary-top-k",
        type=int,
        default=None,
        help="每个摘要窗口保留的 top-k 片段数。",
    )
    parser.add_argument(
        "--print-candidates",
        action="store_true",
        help="打印原始候选事件。默认只打印触发升级事件和摘要。",
    )
    parser.add_argument(
        "--print-raw-analysis",
        action="store_true",
        help="调试模式：打印云端视觉模型返回的完整原始文本/JSON。",
    )
    parser.add_argument(
        "--print-emotion-signal",
        action="store_true",
        help="调试模式：打印由视觉文本推导出的情绪注入信号。",
    )
    parser.add_argument(
        "--no-vision",
        action="store_true",
        help="即使提供了 config 文件，也禁用 Vision LLM 分析。",
    )
    args = parser.parse_args()

    if not CV2_AVAILABLE:
        logger.error(
            "未检测到 OpenCV，请先安装 opencv-python-headless 或 opencv-python。"
        )
        return 1

    source = _parse_source(args.source)
    analyzer = None
    pipeline = None
    config = VisualPerceptionConfig()
    if args.config:
        app_config = AppConfig.load(args.config)
        config = VisualPerceptionConfig.from_settings(app_config.visual_perception)
        if not args.no_vision:
            pipeline = NeuroLikePipeline.from_config(args.config)
            analyzer = build_visual_analyzer_from_pipeline(pipeline)

    if args.analysis_mode is not None:
        config.vision_analysis_mode = args.analysis_mode
    if args.summary_window_seconds is not None:
        config.summary_window_seconds = args.summary_window_seconds
    if args.summary_top_k is not None:
        config.summary_top_k = args.summary_top_k

    def _print_analysis_debug(event):
        if args.print_raw_analysis and event.analysis and event.analysis.raw_text:
            print("  raw_analysis=")
            print(event.analysis.raw_text)
        if args.print_emotion_signal:
            signal = derive_visual_emotion_signal(event, config=config)
            print(
                "  emotion_signal="
                + json.dumps(signal, ensure_ascii=False)
            )

    def _print_candidate_event(event):
        print(
            f"[{event.event_id}] frame={event.peak_frame_index} "
            f"score={event.peak_score:.3f} text={visual_event_to_agent_text(event)}"
        )
        if event.analysis and event.analysis.memory_candidate:
            print(f"  memory_candidate={event.analysis.memory_candidate}")
        _print_analysis_debug(event)
        if event.rate_limited:
            print("  vision_skipped=rate_limited")

    def _print_promoted_event(event):
        mode = str(event.metrics.get("mode", "trigger"))
        duration_seconds = float(event.metrics.get("segment_duration_seconds", 0.0))
        event_count = int(event.metrics.get("segment_event_count", 1))
        print(
            f"[{mode}] peak={event.peak_score:.3f} "
            f"duration={duration_seconds:.1f}s events={event_count} "
            f"text={visual_event_to_agent_text(event)}"
        )
        if event.analysis and event.analysis.memory_candidate:
            print(f"  memory_candidate={event.analysis.memory_candidate}")
        _print_analysis_debug(event)
        if event.rate_limited:
            print("  vision_skipped=rate_limited")

    def _print_summary(summary):
        print(f"[summary] {summary.to_text()}")

    runner = VisualPerceptionPipeline(
        config=config,
        analyzer=analyzer,
        event_callback=_print_candidate_event if args.print_candidates else None,
        promoted_event_callback=_print_promoted_event,
        summary_callback=_print_summary if config.summary_enabled else None,
    )

    if isinstance(source, int) and args.duration_seconds is None and args.max_frames is None:
        logger.info("当前使用摄像头实时模式；未设置时长或帧数上限，将持续运行直到手动中断。")

    try:
        runner.run(
            source,
            max_frames=args.max_frames,
            duration_seconds=args.duration_seconds,
            save_video_path=args.save_video,
            camera_width=args.camera_width,
            camera_height=args.camera_height,
            camera_fps=args.camera_fps,
        )
    except KeyboardInterrupt:
        runner.stop()
        logger.info("收到中断信号，视觉管线已停止。")
        return 130
    except Exception as exc:
        logger.error(f"视觉管线运行失败: {exc}")
        return 1
    finally:
        if pipeline is not None:
            pipeline.close()

    print(
        f"[done] candidates={len(runner.candidate_events)} "
        f"promoted={len(runner.promoted_events)} "
        f"summaries={len(runner.window_summaries)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
