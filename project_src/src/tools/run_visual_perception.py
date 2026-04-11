import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.logger import logger
from configs.config_loader import AppConfig
from src.core.inference_pipeline import NeuroLikePipeline
from src.vision import (
    CV2_AVAILABLE,
    VisualPerceptionConfig,
    VisualPerceptionPipeline,
    build_visual_analyzer_from_pipeline,
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

    analyzer = None
    pipeline = None
    config = VisualPerceptionConfig()
    if args.config:
        app_config = AppConfig.load(args.config)
        config = VisualPerceptionConfig.from_settings(app_config.visual_perception)
        if not args.no_vision:
            pipeline = NeuroLikePipeline.from_config(args.config)
            analyzer = build_visual_analyzer_from_pipeline(pipeline)

    def _print_event(event):
        print(
            f"[{event.event_id}] frame={event.peak_frame_index} "
            f"score={event.peak_score:.3f} text={visual_event_to_agent_text(event)}"
        )
        if event.analysis and event.analysis.memory_candidate:
            print(f"  memory_candidate={event.analysis.memory_candidate}")
        if event.rate_limited:
            print("  vision_skipped=rate_limited")

    runner = VisualPerceptionPipeline(
        config=config,
        analyzer=analyzer,
        event_callback=_print_event,
    )

    try:
        runner.run(_parse_source(args.source), max_frames=args.max_frames)
    finally:
        if pipeline is not None:
            pipeline.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
