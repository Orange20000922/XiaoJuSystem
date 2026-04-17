"""单独测试 Claude Sonnet 对视觉事件的文本回复。"""
import sys, time
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from configs.config_loader import AppConfig
from src.core.inference_pipeline import NeuroLikePipeline
from src.vision import (
    CV2_AVAILABLE, VisualPerceptionConfig, VisualPerceptionPipeline,
    build_visual_analyzer_from_pipeline, derive_visual_emotion_signal,
    visual_event_to_agent_text,
)

VIDEO = project_root / "data" / "drink_test.avi"

# ── Step 1: 视觉管线获取最佳事件 ──
print("=" * 60)
print("  Step 1: 视觉管线 (GLM-4V clip mode)")
print("=" * 60)

pipeline = NeuroLikePipeline.from_config(str(project_root / "config.json"))
analyzer = build_visual_analyzer_from_pipeline(pipeline)

vp_config = VisualPerceptionConfig()
vp_config.vision_analysis_mode = "per_event"
vp_config.vision_calls_per_minute = 10
vp_config.clip_duration_seconds = 2.0
vp_config.clip_max_frames = 8

runner = VisualPerceptionPipeline(config=vp_config, analyzer=analyzer)
events = runner.run(str(VIDEO))
print(f"  候选事件: {len(events)}")

analyzed = [e for e in events if e.analysis and e.analysis.agent_hint]
best = max(analyzed, key=lambda e: e.peak_score) if analyzed else None
if best is None:
    print("ERROR: 无可用事件")
    raise SystemExit(1)

a = best.analysis
print(f"  scene:      {a.scene}")
print(f"  facts:      {a.facts}")
print(f"  agent_hint: {a.agent_hint}")
print(f"  keyframes:  {len(best.keyframes)} 张")

# ── Step 2: GLM 直出 ──
print("\n" + "=" * 60)
print("  Step 2: GLM-4V 直出回复 (images + persona)")
print("=" * 60)

prompt_builder = pipeline._persona.prompt_builder
system_prompt = prompt_builder.build_system_prompt()
visual_text = visual_event_to_agent_text(best)
user_input = f"[你通过视觉感知到了以下画面变化] {visual_text}"

glm_client = pipeline.llm_client_vision or pipeline.llm_client
t0 = time.time()
glm_resp = glm_client.generate(
    system_prompt=system_prompt, user_input=user_input,
    images=best.keyframes, max_tokens=300, temperature=0.8,
)
print(f"  model:   {glm_client.model}")
print(f"  latency: {time.time() - t0:.1f}s")
print(f"  回复:    {glm_resp}")

# ── Step 3: Claude 仅文本 ──
print("\n" + "=" * 60)
print("  Step 3: Claude Sonnet 回复 (text only)")
print("=" * 60)

claude_client = pipeline.llm_client
t0 = time.time()
claude_resp = claude_client.generate(
    system_prompt=system_prompt, user_input=user_input,
    max_tokens=300, temperature=0.8,
)
print(f"  model:   {claude_client.model}")
print(f"  latency: {time.time() - t0:.1f}s")
print(f"  回复:    {claude_resp}")

# ── Step 4: DeepSeek 仅文本 ──
print("\n" + "=" * 60)
print("  Step 4: DeepSeek 回复 (text only)")
print("=" * 60)

ds_client = pipeline.llm_client_secondary
if ds_client:
    t0 = time.time()
    ds_resp = ds_client.generate(
        system_prompt=system_prompt, user_input=user_input,
        max_tokens=300, temperature=0.8,
    )
    print(f"  model:   {ds_client.model}")
    print(f"  latency: {time.time() - t0:.1f}s")
    print(f"  回复:    {ds_resp}")
else:
    ds_resp = "(无 secondary LLM)"
    print(f"  {ds_resp}")

# ── 汇总 ──
print("\n" + "=" * 60)
print("  汇总对比")
print("=" * 60)
print(f"\nGLM-4V (直出+图片):\n  {glm_resp}")
print(f"\nClaude Sonnet (仅文本):\n  {claude_resp}")
if ds_client:
    print(f"\nDeepSeek (仅文本):\n  {ds_resp}")

try:
    pipeline.close()
except Exception:
    pass
