"""
Prompt 构建器

每个 persona 独立的 prompt 构建逻辑。从 NeuroLikePipeline 提取，
负责：人格描述 + 情绪指令注入 + 记忆上下文 + 情绪状态暗示。

PromptBuilder 本身无状态，emotion_state_hint 作为字符串参数传入。
"""

from datetime import datetime
from typing import Dict, List, Optional

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.logger import logger
from configs.model_config import PersonalityConfig, EmotionPromptConfig


class PromptBuilder:
    """每个 persona 独立的 prompt 构建器"""

    def __init__(
        self,
        personality: PersonalityConfig,
        emotion_prompt_config: Optional[EmotionPromptConfig] = None,
        time_awareness: bool = True,
    ):
        self.personality = personality
        self.emotion_prompt_config = emotion_prompt_config
        self.time_awareness = time_awareness

    def _personality_section(self) -> str:
        """生成人格描述文本"""
        p = self.personality
        if p.description.strip():
            return p.description.strip()

        formality_str = (
            "口语" if p.formality < 0.4 else
            "适中" if p.formality < 0.7 else "正式"
        )
        return (
            f"[标签]{','.join(p.traits)} "
            f"[开放]{p.openness:.1f} [外向]{p.extraversion:.1f} "
            f"[幽默]{p.humor_tendency:.1f} [共情]{p.empathy_level:.1f} "
            f"[好奇]{p.curiosity_level:.1f} [风格]{formality_str}"
        )

    def build_system_prompt(
        self,
        recalled_context: str = "",
        emotion_analysis: Optional[Dict] = None,
        emotion_state_hint: Optional[str] = None,
    ) -> str:
        """
        构建 system prompt（单块模式，向后兼容）。
        包含：人格 + 情感分析结果（强制接受）+ L2/L3/L4 跨会话召回。
        L1 原文通过 messages 数组单独传入，不在这里。
        """
        p = self.personality
        personality_section = self._personality_section()

        prompt = f"你是{p.name}。\n<persona>{personality_section}</persona>"

        # 时间感知注入
        if self.time_awareness:
            now = datetime.now().strftime("%Y年%m月%d日 %H:%M")
            prompt += f"\n<current_time>{now}</current_time>"

        # 情感指令注入
        if emotion_analysis and self.emotion_prompt_config:
            directives = self._build_emotion_directives(emotion_analysis)
            if directives:
                prompt += f"\n<mood>\n{directives}\n</mood>"

        if recalled_context:
            prompt += f"\n{recalled_context}"

        # 情绪状态暗示
        if emotion_state_hint:
            prompt += f"\n<feeling>{emotion_state_hint}</feeling>"

        prompt += "\n自然回复就好。"
        return prompt

    def build_system_prompt_blocks(
        self,
        recalled_context: str = "",
        emotion_analysis: Optional[Dict] = None,
        emotion_state_hint: Optional[str] = None,
    ) -> List[Dict]:
        """
        构建多块 system prompt（用于 Anthropic 缓存优化）。

        缓存策略：
        - Block 1（静态，可缓存）：人格描述 + 通用指令
        - Block 2（半静态，可缓存）：跨会话记忆召回（L2/L3/L4）
        - Block 3（动态，不缓存）：情感分析结果（每轮变化）
        - Block 4（动态，不缓存）：情绪状态暗示
        """
        p = self.personality
        blocks = []

        # ── Block 1: 静态人格 + 通用指令（可缓存）────────────────────
        personality_section = self._personality_section()
        static_prompt = f"你是{p.name}。\n<persona>{personality_section}</persona>"

        if self.time_awareness:
            now = datetime.now().strftime("%Y年%m月%d日 %H:%M")
            static_prompt += f"\n<current_time>{now}</current_time>"

        static_prompt += "\n自然回复就好。"

        blocks.append({
            "type": "text",
            "text": static_prompt,
            "cache_control": {"type": "ephemeral"}
        })

        # ── Block 2: 跨会话记忆召回（半静态，可缓存）────────────────
        if recalled_context:
            blocks.append({
                "type": "text",
                "text": recalled_context,
                "cache_control": {"type": "ephemeral"}
            })

        # ── Block 3: 情感指令（动态，不缓存）─────────────────────────────
        if emotion_analysis and self.emotion_prompt_config:
            directives = self._build_emotion_directives(emotion_analysis)
            if directives:
                blocks.append({
                    "type": "text",
                    "text": f"<mood>\n{directives}\n</mood>"
                })

        # ── Block 4: 情绪状态暗示（可选，动态，不缓存）──────────────────
        if emotion_state_hint:
            blocks.append({
                "type": "text",
                "text": f"<feeling>{emotion_state_hint}</feeling>"
            })

        return blocks

    def _build_emotion_directives(self, emotion_analysis: Dict) -> str:
        """
        从 BERT 输出 + config 映射生成指令文本。

        置信度门控：
          effective_confidence = BERT_prob * emotion_reliability[label]
          > strong  → 注入确定性指令
          > weak    → 注入不确定指令（"用户可能…"）
          ≤ weak   → 跳过情绪指令，让 LLM 自行判断
        """
        cfg = self.emotion_prompt_config
        parts = []

        emotion = emotion_analysis["emotion"]["primary"]
        intensity = emotion_analysis["emotion"]["intensity"]
        bert_prob = emotion_analysis["emotion"].get("primary_prob", 1.0)

        # 有效置信度 = BERT 输出概率 × 该标签的历史可靠度
        reliability = cfg.emotion_reliability.get(emotion, 0.7)
        effective_conf = bert_prob * reliability
        strong_thr = cfg.confidence_thresholds.get("strong", 0.5)
        weak_thr = cfg.confidence_thresholds.get("weak", 0.3)

        # 查 emotion_map
        emotion_directive = cfg.emotion_map.get(emotion, "")
        if emotion_directive:
            if effective_conf >= strong_thr:
                if intensity >= cfg.intensity_levels.get("high_min", 0.7):
                    emotion_directive = f"（强烈）{emotion_directive}"
                parts.append(emotion_directive)
            elif effective_conf >= weak_thr:
                stripped = emotion_directive.lstrip("用户")
                parts.append(f"好像{stripped}，但也说不准")

        directive = "。".join(parts) if parts else ""
        logger.debug(
            f"[情感层] 指令注入: 情绪={emotion}(eff_conf={effective_conf:.2f}) "
            f"{'「' + directive + '」' if directive else '(跳过，置信度不足)'}"
        )
        return directive
