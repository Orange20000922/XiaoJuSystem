"""
主动决策模块

使用 DeepSeek 判断是否需要主动发言，并输出结构化的意图指导。
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.logger import logger
from configs.model_config import LLMConfig, LLMProvider, ProactiveConfig


@dataclass
class ProactiveDecision:
    """主动决策结果"""
    should_respond: bool
    confidence: float
    intent: str           # "关心对方" | "分享想法" | "调节气氛" 等
    topic_hint: str       # "可以聊聊刚才提到的..."
    tone: str             # "warm" | "playful" | "calm" 等
    reason: str           # 决策理由（调试用）


class ProactiveDecisionModule:
    """主动决策模块"""

    def __init__(self, llm_config: LLMConfig, config: ProactiveConfig):
        self.config = config
        self.llm_config = llm_config
        self._init_client()

    def _init_client(self):
        """初始化 LLM 客户端"""
        from openai import OpenAI
        # 使用 LLMConfig 的 timeout，如果没有则使用 ProactiveConfig 的 decision_timeout
        timeout = self.llm_config.timeout if hasattr(self.llm_config, 'timeout') else self.config.decision_timeout
        self.client = OpenAI(
            api_key=self.llm_config.api_key,
            base_url=self.llm_config.base_url,
            timeout=timeout
        )

    def decide(
        self,
        recent_conversations: List[Dict],
        emotion_state: Dict,
        l4_memories: List[str],
        current_time: str = "",
    ) -> Optional[ProactiveDecision]:
        """
        调用 DeepSeek 判断是否需要主动发言。

        Args:
            recent_conversations: L1/L3 最近对话记录
            emotion_state: emotion_state.json 内容
            l4_memories: L4 情感记忆片段
            current_time: 当前时间（可选）

        Returns:
            ProactiveDecision 或 None（不需要发言时）
        """
        # 构造 prompt
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(
            recent_conversations, emotion_state, l4_memories, current_time
        )

        try:
            # 显式传递 timeout，确保生效
            timeout = self.llm_config.timeout if hasattr(self.llm_config, 'timeout') else 60

            logger.debug(f"开始调用 DeepSeek API，timeout={timeout}秒")
            import time
            start_time = time.time()

            response = self.client.chat.completions.create(
                model=self.llm_config.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.config.decision_temperature,
                max_tokens=500,
                response_format={"type": "json_object"},
                timeout=timeout  # 显式传递 timeout
            )

            elapsed = time.time() - start_time
            logger.debug(f"DeepSeek API 响应成功，耗时 {elapsed:.2f}秒")

            content = response.choices[0].message.content
            result = json.loads(content)

            decision = ProactiveDecision(
                should_respond=result.get("should_respond", False),
                confidence=result.get("confidence", 0.0),
                intent=result.get("intent", ""),
                topic_hint=result.get("topic_hint", ""),
                tone=result.get("tone", "calm"),
                reason=result.get("reason", "")
            )

            logger.debug(
                f"主动决策: should_respond={decision.should_respond} "
                f"confidence={decision.confidence:.2f} intent={decision.intent}"
            )

            return decision

        except Exception as e:
            logger.error(f"主动决策失败: {e}")
            return None

    def _build_system_prompt(self) -> str:
        return """你是 Neuro 的主动性判断助手。你的任务是根据对话上下文、情绪状态和历史记忆，判断 Neuro 是否应该主动发言。

你不生成实际回复，只输出结构化的决策。

输出格式（JSON）：
{
  "should_respond": true/false,
  "confidence": 0.0-1.0,
  "intent": "关心对方" | "分享想法" | "调节气氛" | "继续话题" | "无需发言",
  "topic_hint": "可以聊聊刚才提到的..." (如果 should_respond=true),
  "tone": "warm" | "playful" | "calm" | "supportive",
  "reason": "决策理由（简短）"
}

判断原则：
1. 对话自然结束（对方说"好的"、"嗯"、"拜拜"等）→ should_respond=false
2. 对方情绪低落且对话中断 → should_respond=true, intent="关心对方"
3. 对话话题有趣但中断 → should_respond=true, intent="继续话题"
4. 对方明确表示不想聊 → should_respond=false
5. 空闲时间很长但没有明显话题 → should_respond=false (除非有特殊情感记忆)
6. 情绪状态显示 Neuro 自己情绪波动较大 → 可以主动分享感受

confidence 计算：
- 对话上下文明确 → 高置信度
- 情绪信号强烈 → 高置信度
- 缺乏上下文或信号模糊 → 低置信度"""

    def _build_user_prompt(
        self,
        recent_conversations: List[Dict],
        emotion_state: Dict,
        l4_memories: List[str],
        current_time: str
    ) -> str:
        # 格式化对话历史
        conv_text = self._format_conversations(recent_conversations)

        # 格式化情绪状态
        emotion_text = self._format_emotion_state(emotion_state)

        # 格式化 L4 记忆
        memory_text = "\n".join(l4_memories) if l4_memories else "（无相关记忆）"

        prompt = f"""请根据以下信息判断 Neuro 是否应该主动发言：

【最近对话】
{conv_text}

【当前情绪氛围】
{emotion_text}

【过往情感记忆】
{memory_text}"""

        if current_time:
            prompt += f"\n\n【当前时间】\n{current_time}"

        return prompt

    def _format_conversations(self, conversations: List[Dict]) -> str:
        if not conversations:
            return "（无对话记录）"

        lines = []
        for turn in conversations:
            user_msg = turn.get("user_input", "")
            ai_msg = turn.get("response", "")
            if user_msg:
                lines.append(f"用户: {user_msg}")
            if ai_msg:
                lines.append(f"Neuro: {ai_msg}")

        return "\n".join(lines) if lines else "（无对话记录）"

    def _format_emotion_state(self, state: Dict) -> str:
        if not state:
            return "（无情绪状态记录）"

        valence = state.get("valence", 0.0)
        arousal = state.get("arousal", 0.0)
        last_emotion = state.get("last_emotion", "neutral")

        valence_desc = "积极" if valence > 0.2 else "消极" if valence < -0.2 else "中性"
        arousal_desc = "激动" if arousal > 0.3 else "平静"

        return f"情绪倾向: {valence_desc}（{valence:.2f}），激活度: {arousal_desc}（{arousal:.2f}），上次情绪: {last_emotion}"
