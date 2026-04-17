"""端到端测试：自适应帧采样 + GLM-4V 分析，用 test_adaptive.avi 跑完整管线。

对比 adaptive vs baseline：
  - 事件检出数、GLM 调用次数、总耗时
  - 最佳事件的 GLM 分析结果（scene/facts/agent_hint）
  - 三模型回复（GLM 直出 / Claude / DeepSeek）
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from src.logger import logger
from configs.config_loader import AppConfig
from src.core.inference_pipeline import NeuroLikePipeline
from src.vision import (
    CV2_AVAILABLE,
    VisualPerceptionConfig,
    VisualPerceptionPipeline,
    build_visual_analyzer_from_pipeline,
    derive_visual_emotion_signal,
    visual_event_to_agent_text,
)

VIDEO = project_root / "data" / "test_adaptive.avi"
CONFIG = project_root / "config.json"


def _sep(title: str):
    print(f"\n{'=' * 60}\n  {title}\n{'=' * 60}")


def _run_pipeline(adaptive: bool, pipeline: NeuroLikePipeline, analyzer):
    label = "adaptive" if adaptive else "baseline"
    _sep(f"视觉管线 ({label}, analysis=triggered)")

    app_config = AppConfig.load(str(CONFIG))
    vp_config = VisualPerceptionConfig.from_settings(app_config.visual_perception)
    vp_config.vision_analysis_mode = "triggered"
    vp_config.vision_calls_per_minute = 5
    vp_config.adaptive_sampling_enabled = adaptive
    if adaptive:
        vp_config.adaptive_fps_min = 4.0
        vp_config.adaptive_spike_threshold = 0.4
        vp_config.adaptive_precheck_diff_threshold = 15.0

    runner = VisualPerceptionPipeline(config=vp_config, analyzer=analyzer)

    t0 = time.time()
    events = runner.run(str(VIDEO))
    elapsed = time.time() - t0

    analyzed = [e for e in events if e.analysis and e.analysis.agent_hint]
    promoted = runner.promoted_events
    promoted_analyzed = [e for e in promoted if e.analysis and e.analysis.agent_hint]

    sampler = runner._adaptive_sampler
    print(f"  总事件数: {len(events)}")
    print(f"  已分析候选事件: {len(analyzed)}")
    print(f"  触发升级事件: {len(promoted)} (已分析: {len(promoted_analyzed)})")
    print(f"  总耗时: {elapsed:.1f}s")
    if sampler:
        print(f"  target_fps: {sampler.current_target_fps:.2f}")
        print(f"  precheck_force: {sampler.precheck_force_count}")

    all_analyzed = promoted_analyzed or analyzed
    best = max(all_analyzed, key=lambda e: e.peak_score) if all_analyzed else None
    if best:
        a = best.analysis
        print(f"\n  最佳事件:")
        print(f"    t={best.timestamp:.2f}s  score={best.peak_score:.3f}")
        print(f"    scene:      {a.scene}")
        print(f"    facts:      {a.facts}")
        print(f"    agent_hint: {a.agent_hint}")
        signal = derive_visual_emotion_signal(best, config=vp_config)
        print(f"    emotion:    {signal['emotion']} (conf={signal['confidence']:.2f})")

    return {
        "label": label,
        "elapsed": elapsed,
        "events": len(events),
        "analyzed": len(analyzed) + len(promoted_analyzed),
        "promoted": len(promoted),
        "best": best,
    }


def _test_responses(pipeline: NeuroLikePipeline, best_event):
    """用最佳事件的 agent_text 测三个模型的回复。"""
    _sep("三模型回复对比")

    prompt_builder = pipeline._persona.prompt_builder
    system_prompt = prompt_builder.build_system_prompt()
    visual_text = visual_event_to_agent_text(best_event)
    user_input = f"[你通过视觉感知到了以下画面变化] {visual_text}"

    clients = [
        ("GLM-4V (直出)", pipeline.llm_client_vision or pipeline.llm_client, best_event.keyframes),
        ("Claude Opus", pipeline.llm_client, []),
        ("DeepSeek", pipeline.llm_client_secondary, []),
    ]

    for name, client, images in clients:
        if client is None:
            print(f"\n  {name}: (无可用客户端)")
            continue
        t0 = time.time()
        try:
            kwargs = dict(system_prompt=system_prompt, user_input=user_input,
                          max_tokens=300, temperature=0.8)
            if images:
                kwargs["images"] = images
            resp = client.generate(**kwargs)
        except Exception as exc:
            resp = f"[ERROR] {exc}"
        print(f"\n  {name} ({client.model}, {time.time()-t0:.1f}s):")
        print(f"    {resp}")


def main() -> int:
    if not CV2_AVAILABLE:
        print("ERROR: OpenCV 不可用")
        return 1
    if not VIDEO.exists():
        print(f"ERROR: 视频不存在: {VIDEO}")
        return 1

    pipeline = NeuroLikePipeline.from_config(str(CONFIG))
    analyzer = build_visual_analyzer_from_pipeline(pipeline)

    baseline = _run_pipeline(adaptive=False, pipeline=pipeline, analyzer=analyzer)
    adaptive = _run_pipeline(adaptive=True, pipeline=pipeline, analyzer=analyzer)

    _sep("管线对比汇总")
    for r in [baseline, adaptive]:
        print(f"  {r['label']:<10}  events={r['events']:>3}  analyzed={r['analyzed']:>3}  "
              f"promoted={r['promoted']:>3}  elapsed={r['elapsed']:>6.1f}s")
    if baseline['elapsed'] > 0 and adaptive['elapsed'] > 0:
        print(f"\n  加速比: {baseline['elapsed']/adaptive['elapsed']:.2f}x")

    # 用 adaptive 的最佳事件做三模型回复
    best = adaptive.get("best") or baseline.get("best")
    if best:
        _test_responses(pipeline, best)
    else:
        print("\n  无可分析事件，跳过回复测试")

    try:
        pipeline.close()
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
