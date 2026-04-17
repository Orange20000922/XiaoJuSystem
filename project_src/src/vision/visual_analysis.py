from __future__ import annotations

from typing import Optional, TYPE_CHECKING

from .visual_types import (
    DEFAULT_VISION_ANALYSIS_PROMPT,
    VisualAnalysis,
    VisualEvent,
)

if TYPE_CHECKING:
    from src.core.inference_pipeline import NeuroLikePipeline
    from src.llm.client import LLMClient


_PERSON_FOCUS_INSTRUCTION = (
    "如果画面中有清晰的人脸、头肩或主体人物，请优先围绕该人物、"
    "该人物脸部/手部/嘴部附近的物体，以及人与物的交互来描述。"
)

_BACKGROUND_FILTER_INSTRUCTION = (
    "若同时出现多人，优先描述最显著、最居中、持续出现或离镜头最近的人；"
    "不要把背景里的次要人物、衣着细节、轻微光照变化或远处扰动当作主事件。"
)

_CLIP_TEMPORAL_INSTRUCTION = (
    "这些图片来自一段连续视频片段中均匀采样的帧，请特别关注动作的时序变化和连续性，"
    "描述人物或场景在这段时间内的动态过程，而非孤立描述每一帧。"
)


def build_visual_analysis_user_input(event: VisualEvent) -> str:
    mode = str(event.metrics.get("mode", "")).strip().lower()
    clip_duration = float(event.metrics.get("clip_duration", 0))
    is_clip = clip_duration > 0

    temporal_hint = _CLIP_TEMPORAL_INSTRUCTION if is_clip else ""
    frame_desc = "连续帧序列" if is_clip else "代表画面"

    if mode == "summary":
        return (
            f"以下图片来自同一较长时间窗口内选出的{frame_desc}，"
            "顺序与时间顺序一致，但彼此不一定连续。"
            "请概括这个时间窗口内最值得关注的可见活动、动作变化或场景变化。"
            f"{_PERSON_FOCUS_INSTRUCTION}"
            f"{_BACKGROUND_FILTER_INSTRUCTION}"
            f"{temporal_hint}"
            "只描述你能从图像直接支持的内容，只输出 JSON。"
        )

    if mode in {"trigger", "manual"}:
        return (
            f"以下图片按时间顺序来自同一段短时间片段，是该片段的{frame_desc}。"
            "请根据画面之间的变化，概括这段时间里最显著的可见动作、"
            "物体变化或人与物的交互。"
            f"{_PERSON_FOCUS_INSTRUCTION}"
            f"{_BACKGROUND_FILTER_INSTRUCTION}"
            f"{temporal_hint}"
            "只描述你能从图像直接支持的内容，只输出 JSON。"
        )

    return (
        f"以下图片按时间顺序来自一次短暂变化附近的{frame_desc}。"
        "请简洁描述其中最明显的可见变化或动作。"
        f"{_PERSON_FOCUS_INSTRUCTION}"
        f"{_BACKGROUND_FILTER_INSTRUCTION}"
        f"{temporal_hint}"
        "只描述你能从图像直接支持的内容，只输出 JSON。"
    )


class LLMVisualEventAnalyzer:
    def __init__(
        self,
        llm_client: "LLMClient",
        prompt: str = DEFAULT_VISION_ANALYSIS_PROMPT,
        max_tokens: int = 400,
    ):
        self.llm_client = llm_client
        self.prompt = prompt
        self.max_tokens = max_tokens

    def analyze(self, event: VisualEvent) -> Optional[VisualAnalysis]:
        if not event.keyframes:
            return None
        text = self.llm_client.generate(
            system_prompt=self.prompt,
            user_input=build_visual_analysis_user_input(event),
            images=event.keyframes,
            max_tokens=self.max_tokens,
            temperature=0.2,
        )
        return VisualAnalysis.from_llm_text(text)


def build_visual_analyzer_from_pipeline(
    pipeline: "NeuroLikePipeline",
    *,
    prompt: str = DEFAULT_VISION_ANALYSIS_PROMPT,
) -> Optional[LLMVisualEventAnalyzer]:
    llm_client = pipeline.llm_client_vision or pipeline.llm_client
    if llm_client is None:
        return None
    return LLMVisualEventAnalyzer(llm_client=llm_client, prompt=prompt)


__all__ = [
    "LLMVisualEventAnalyzer",
    "build_visual_analysis_user_input",
    "build_visual_analyzer_from_pipeline",
]
