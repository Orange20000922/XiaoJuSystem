"""
端到端集成测试：视觉事件直接响应效果对比

用 drink_test.avi 跑视觉管线 → 选取最佳事件 → 用三种 LLM 生成人格化回复：
  1. GLM-4V (vision LLM)  — 带关键帧的直出模式
  2. Claude Sonnet         — 仅文本描述（无图片）
  3. DeepSeek              — 仅文本描述（无图片）

同时对比 GLM 的结构化分析输出（JSON facts/agent_hint）和三者的对话回复。
"""

import json
import sys
import time
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

# Windows 控制台 UTF-8 输出
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from src.logger import logger
from configs.config_loader import AppConfig
from src.core.inference_pipeline import NeuroLikePipeline
from src.core.prompt_builder import PromptBuilder
from src.vision import (
    CV2_AVAILABLE,
    VisualPerceptionConfig,
    VisualPerceptionPipeline,
    build_visual_analyzer_from_pipeline,
    derive_visual_emotion_signal,
    visual_event_to_agent_text,
)
from src.vision.visual_agent import visual_event_to_agent_event

DATA_DIR = project_root / "data"
VIDEO_FILE = DATA_DIR / "drink_test.avi"
CONFIG_CLAUDE = project_root / "config.json"
CONFIG_DEEPSEEK = project_root / "config_test_deepseek.json"


def _sep(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print('=' * 60)


def _subsep(title: str):
    print(f"\n--- {title} ---")


def run_visual_pipeline(config_path: str, clip_mode: bool = True):
    """跑视觉管线，返回最佳事件和 pipeline 引用。"""
    app_config = AppConfig.load(config_path)
    vp_config = VisualPerceptionConfig.from_settings(app_config.visual_perception)
    vp_config.vision_analysis_mode = "per_event"
    vp_config.vision_calls_per_minute = 10
    if clip_mode:
        vp_config.clip_duration_seconds = 2.0
        vp_config.clip_max_frames = 8

    pipeline = NeuroLikePipeline.from_config(config_path)
    analyzer = build_visual_analyzer_from_pipeline(pipeline)

    runner = VisualPerceptionPipeline(
        config=vp_config,
        analyzer=analyzer,
    )

    print(f"运行视觉管线: source={VIDEO_FILE}")
    print(f"  clip_mode={'ON' if clip_mode else 'OFF'}")
    print(f"  analysis_mode={vp_config.vision_analysis_mode}")

    events = runner.run(str(VIDEO_FILE))
    print(f"  候选事件数: {len(events)}")

    analyzed = [e for e in events if e.analysis and e.analysis.agent_hint]
    if not analyzed:
        analyzed = [e for e in events if e.analysis]
    if not analyzed:
        analyzed = events

    best = max(analyzed, key=lambda e: e.peak_score) if analyzed else None
    return best, pipeline, vp_config


def print_event_analysis(event):
    """打印事件的结构化分析结果。"""
    _subsep("GLM-4V 结构化分析 (JSON)")
    if event.analysis:
        a = event.analysis
        print(f"  scene:      {a.scene}")
        print(f"  facts:      {a.facts}")
        print(f"  weak_interp: {a.weak_interpretations}")
        print(f"  agent_hint: {a.agent_hint}")
        print(f"  memory_cand: {a.memory_candidate}")
    else:
        print("  (无分析结果)")

    print(f"  peak_score: {event.peak_score:.3f}")
    print(f"  keyframes:  {len(event.keyframes)} 张")

    signal = derive_visual_emotion_signal(event)
    print(f"  emotion:    {signal['emotion']} (conf={signal['confidence']:.2f})")
    print(f"  valence_Δ:  {signal['valence_delta']:+.4f}")
    print(f"  arousal_Δ:  {signal['arousal_delta']:+.4f}")


def generate_direct_response(pipeline: NeuroLikePipeline, event, label: str):
    """用指定 pipeline 的 LLM 生成人格化回复。"""
    persona = pipeline._persona
    prompt_builder = persona.prompt_builder

    system_prompt = prompt_builder.build_system_prompt()

    visual_text = visual_event_to_agent_text(event)
    user_input = f"[你通过视觉感知到了以下画面变化] {visual_text}"

    _subsep(f"{label} 回复")
    print(f"  user_input: {user_input[:100]}...")

    start_time = time.time()
    try:
        if label.startswith("GLM") and event.keyframes:
            client = pipeline.llm_client_vision or pipeline.llm_client
            response = client.generate(
                system_prompt=system_prompt,
                user_input=user_input,
                images=event.keyframes,
                max_tokens=300,
                temperature=0.8,
            )
        elif label.startswith("Claude"):
            client = pipeline.llm_client
            response = client.generate(
                system_prompt=system_prompt,
                user_input=user_input,
                max_tokens=300,
                temperature=0.8,
            )
        elif label.startswith("DeepSeek"):
            client = pipeline.llm_client_secondary or pipeline.llm_client
            response = client.generate(
                system_prompt=system_prompt,
                user_input=user_input,
                max_tokens=300,
                temperature=0.8,
            )
        else:
            response = "(unknown label)"
    except Exception as exc:
        response = f"[ERROR] {exc}"

    elapsed = time.time() - start_time
    print(f"  model:    {getattr(client, 'model', '?')}")
    print(f"  latency:  {elapsed:.1f}s")
    print(f"  response: {response}")
    return response


def main():
    if not CV2_AVAILABLE:
        print("ERROR: OpenCV 不可用")
        return 1
    if not VIDEO_FILE.exists():
        print(f"ERROR: 测试视频不存在: {VIDEO_FILE}")
        return 1

    # ── Step 1: 跑视觉管线获取最佳事件 ──
    _sep("Step 1: 视觉管线 (GLM-4V clip mode)")
    best_event, claude_pipeline, vp_config = run_visual_pipeline(
        str(CONFIG_CLAUDE), clip_mode=True
    )
    if best_event is None:
        print("ERROR: 未检测到任何视觉事件")
        return 1

    print_event_analysis(best_event)

    # ── Step 2: AgentEvent 转换验证 ──
    _sep("Step 2: AgentEvent 转换验证")
    agent_event = visual_event_to_agent_event(best_event)
    print(f"  type:      {agent_event.type}")
    print(f"  content:   {agent_event.content[:80]}...")
    print(f"  images:    {len(agent_event.images)} 张")
    print(f"  scene:     {agent_event.metadata.get('analysis', {}).get('scene', '(none)')}")
    has_images = bool(agent_event.images)
    visual_direct = agent_event.type == "visual" and has_images
    print(f"  visual_direct: {visual_direct}")

    # ── Step 3: GLM 直出回复（带关键帧）──
    _sep("Step 3: 三模型回复对比")

    glm_response = generate_direct_response(
        claude_pipeline, best_event, "GLM-4V (images + persona)"
    )

    # ── Step 4: Claude 回复（仅文本描述）──
    claude_response = generate_direct_response(
        claude_pipeline, best_event, "Claude Sonnet (text only)"
    )

    # ── Step 5: DeepSeek 回复（仅文本描述）──
    deepseek_response = None
    if CONFIG_DEEPSEEK.exists():
        try:
            ds_pipeline = NeuroLikePipeline.from_config(str(CONFIG_DEEPSEEK))
            deepseek_response = generate_direct_response(
                ds_pipeline, best_event, "DeepSeek (text only)"
            )
            ds_pipeline.close()
        except Exception as exc:
            print(f"  DeepSeek 初始化失败: {exc}")

    # ── Step 6: 视觉技能调用测试 ──
    _sep("Step 4: 视觉技能调用模拟")
    from src.vision.visual_skill import VisualSkillDetector, VisualSkillExecutor

    detector = VisualSkillDetector()
    test_messages = [
        "你能看看我在做什么吗？",
        "你看到了什么",
        "今天天气怎么样",
        "画面上有什么东西",
        "你好呀",
    ]
    for msg in test_messages:
        matched = detector.detect(msg)
        print(f"  {'✓' if matched else '✗'} \"{msg}\"")

    mock_events = [best_event] if best_event else []
    executor = VisualSkillExecutor(handler=lambda top_k=2: mock_events[:top_k])
    skill_result = executor.execute(top_k=2)
    print(f"\n  技能执行结果: {skill_result}")

    # ── 汇总 ──
    _sep("汇总对比")
    print(f"\n{'─' * 40}")
    print(f"GLM-4V (直出+图片):")
    print(f"  {glm_response[:200]}")
    print(f"\n{'─' * 40}")
    print(f"Claude Sonnet (仅文本):")
    print(f"  {claude_response[:200]}")
    if deepseek_response:
        print(f"\n{'─' * 40}")
        print(f"DeepSeek (仅文本):")
        print(f"  {deepseek_response[:200]}")

    # 清理（DeepSeek 关闭了共享 Qdrant 后 Claude pipeline 不能再 close_session）
    try:
        claude_pipeline.close()
    except Exception:
        pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
