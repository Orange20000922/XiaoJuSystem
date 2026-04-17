"""远端 LLM 集成测试：验证 GLM-4V 返回的视觉分析 JSON 信号。"""
from __future__ import annotations

import json
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
    VisualAnalysis,
    VisualEvent,
    VisualPerceptionConfig,
    VisualPerceptionPipeline,
    build_visual_analyzer_from_pipeline,
)
from src.vision.visual_skill import VisualSkillDetector

VIDEO = project_root / "data" / "test_adaptive.avi"
CONFIG = project_root / "config.json"


def _sep(title: str):
    print(f"\n{'=' * 60}\n  {title}\n{'=' * 60}")


def main() -> int:
    _sep("1. GLM-4V 视觉分析 JSON 测试")

    pipeline = NeuroLikePipeline.from_config(str(CONFIG))
    analyzer = build_visual_analyzer_from_pipeline(pipeline)

    # 用 adaptive 模式跑管线，取 promoted 事件
    app_config = AppConfig.load(str(CONFIG))
    vp_config = VisualPerceptionConfig.from_settings(app_config.visual_perception)
    vp_config.vision_analysis_mode = "triggered"
    vp_config.vision_calls_per_minute = 10
    vp_config.adaptive_sampling_enabled = True

    runner = VisualPerceptionPipeline(config=vp_config, analyzer=analyzer)
    events = runner.run(str(VIDEO))

    promoted = runner.promoted_events
    analyzed = [e for e in promoted if e.analysis]

    print(f"  候选事件: {len(events)}  升级事件: {len(promoted)}  已分析: {len(analyzed)}")

    if not analyzed:
        print("  ERROR: 无已分析事件")
        try: pipeline.close()
        except: pass
        return 1

    # 验证每个分析的 JSON 结构
    _sep("2. JSON 结构验证")
    for i, event in enumerate(analyzed):
        a = event.analysis
        print(f"\n  事件 {i+1}: t={event.timestamp:.1f}s  score={event.peak_score:.3f}")
        print(f"    scene:           {repr(a.scene)}")
        print(f"    facts:           {a.facts}")
        print(f"    weak_interp:     {a.weak_interpretations}")
        print(f"    memory_cand:     {repr(a.memory_candidate)}")
        print(f"    agent_hint:      {repr(a.agent_hint)}")

        issues = []
        if not a.scene:
            issues.append("scene 为空")
        if not a.facts:
            issues.append("facts 为空")
        if not a.agent_hint:
            issues.append("agent_hint 为空")
        if a.raw_text and a.raw_text.startswith("{"):
            try:
                json.loads(a.raw_text)
            except json.JSONDecodeError as e:
                issues.append(f"raw_text JSON 解析失败: {e}")

        if issues:
            print(f"    问题: {', '.join(issues)}")
        else:
            print(f"    结构完整")

    # 用最佳事件测试 skill 检测 + 回复生成
    best = max(analyzed, key=lambda e: e.peak_score)
    _sep("3. Skill 检测 + LLM 回复生成")

    detector = VisualSkillDetector()
    test_queries = [
        "你能看看我在做什么吗",
        "你看到了什么",
        "画面上有什么",
        "你好呀",
    ]

    prompt_builder = pipeline._persona.prompt_builder
    system_prompt = prompt_builder.build_system_prompt()

    for q in test_queries:
        triggered = detector.detect(q)
        if triggered:
            from src.vision.visual_text import visual_event_to_agent_text
            visual_text = visual_event_to_agent_text(best)
            user_input = f"[视觉感知] {visual_text}\n{q}"
        else:
            user_input = q

        print(f"\n  Query: \"{q}\"  triggered={triggered}")
        if triggered:
            t0 = time.time()
            try:
                resp = pipeline.llm_client.generate(
                    system_prompt=system_prompt, user_input=user_input,
                    max_tokens=200, temperature=0.8,
                )
                print(f"    Claude Opus ({time.time()-t0:.1f}s): {resp}")
            except Exception as exc:
                print(f"    Claude ERROR: {exc}")

    try: pipeline.close()
    except: pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
