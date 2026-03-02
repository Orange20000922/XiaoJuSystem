"""
情绪状态机（EmotionStateTracker）

持续演化的内部情绪状态（valence + arousal 二维连续值），
由用户输入的 BERT 情绪标签和 Neuro 自身输出的 BERT 情绪标签共同驱动。
通过 tanh 非线性状态更新 + 小噪声，产生自然的情绪惯性和随机波动。
"""

import json
import math
import random
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


# ============== 离散标签 → 连续 (valence, arousal) ==============
# 基于心理学 Circumplex 模型
EMOTION_VA_MAP: Dict[str, Tuple[float, float]] = {
    "neutral":    ( 0.0,  0.1),
    "joy":        ( 0.8,  0.6),
    "sadness":    (-0.7,  0.2),
    "anger":      (-0.4,  0.8),
    "fear":       (-0.6,  0.7),
    "surprise":   ( 0.1,  0.8),
    "disgust":    (-0.5,  0.4),
    "excitement": ( 0.7,  0.9),
    "tenderness": ( 0.6,  0.3),
    "curiosity":  ( 0.5,  0.6),
}

# ============== 状态 → 自然语言暗示 ==============
STATE_HINT_MAP: Dict[str, str] = {
    "joy":        "最近聊得不错，可以自然地带一点轻松愉快",
    "sadness":    "聊了一些沉重的话题，用陈述句回应，给予陪伴和理解就好",
    "anger":      "对话里有些让你不太舒服的东西，直接表达感受，不展开追问",
    "fear":       "话题让你有点不安，给予安全感，用肯定句而不是疑问句",
    "excitement": "对话让你很有活力，可以热情一些",
    "tenderness": "聊到了让你心软的内容，可以用更温柔的方式回应",
    "curiosity":  "被这个话题激发了好奇，可以表现出想深入了解的兴趣",
    "disgust":    "对话让你有点不舒服，直接说出感受，不追问原因",
}


@dataclass
class EmotionStateConfig:
    """情绪状态机配置"""
    alpha: float = 0.80             # 状态惯性
    beta: float = 0.25              # AI 自身输出影响权重
    gamma: float = 0.12             # 用户情绪影响权重
    noise_sigma: float = 0.05       # 随机噪声强度
    injection_threshold: float = 0.35  # |state| 超过此值才注入 prompt
    save_interval_turns: int = 5    # 每 N 轮保存一次到 L4
    persist_to_l4: bool = True


@dataclass
class EmotionState:
    """当前情绪状态"""
    valence: float = 0.0
    arousal: float = 0.1
    turn_count: int = 0
    last_emotion: str = "neutral"


class EmotionStateTracker:
    """
    情绪状态追踪器

    维护一个 (valence, arousal) 二维连续状态，
    每轮由用户和 AI 的情绪信号共同驱动更新。
    """

    def __init__(
        self,
        config: EmotionStateConfig,
        initial_state: Optional[EmotionState] = None,
    ):
        self.config = config
        self.state = initial_state or EmotionState()

    def update(
        self,
        user_emotion: str,
        user_intensity: float,
        ai_emotion: str,
        ai_intensity: float,
    ) -> EmotionState:
        """
        非线性状态更新，返回新状态。

        v_new = tanh(α * v_old + β * ai_v + γ * user_v + ε)
        a_new = tanh(α * a_old + β * ai_a + γ * user_a + ε)
        """
        cfg = self.config

        # 投影到 (v, a) 并乘以强度
        user_v, user_a = self._project(user_emotion, user_intensity)
        ai_v, ai_a = self._project(ai_emotion, ai_intensity)

        # 噪声
        eps_v = random.gauss(0, cfg.noise_sigma)
        eps_a = random.gauss(0, cfg.noise_sigma)

        # tanh 更新
        v_new = math.tanh(
            cfg.alpha * self.state.valence
            + cfg.beta * ai_v
            + cfg.gamma * user_v
            + eps_v
        )
        a_new = math.tanh(
            cfg.alpha * self.state.arousal
            + cfg.beta * ai_a
            + cfg.gamma * user_a
            + eps_a
        )

        self.state.valence = v_new
        self.state.arousal = a_new
        self.state.turn_count += 1
        self.state.last_emotion = self._state_to_label()

        return self.state

    def get_param_adjustments(
        self,
        base_temp: float,
        base_tokens: int,
        base_top_p: float,
    ) -> Dict:
        """
        返回根据情绪状态调节后的 LLM 参数。

        temperature: arousal 驱动，负 valence 微调
        max_tokens:  valence 负向延长
        top_p:       arousal 偏离平静基线时增大多样性
        """
        v = self.state.valence
        a = self.state.arousal

        # temperature
        adj_temp = base_temp + 0.20 * a - 0.08 * max(0.0, -v)
        adj_temp = max(base_temp * 0.75, min(adj_temp, base_temp * 1.35))

        # max_tokens
        adj_tokens = int(base_tokens * (1 + 0.25 * max(0.0, -v)))
        adj_tokens = max(base_tokens, min(adj_tokens, int(base_tokens * 1.25)))

        # top_p
        arousal_dev = abs(a - 0.3)
        adj_top_p = max(0.80, min(base_top_p + 0.08 * arousal_dev, 0.99))

        return {
            "temperature": round(adj_temp, 4),
            "max_tokens": adj_tokens,
            "top_p": round(adj_top_p, 4),
        }

    def get_prompt_hint(self) -> Optional[str]:
        """
        返回自然语言情绪暗示，注入 system prompt。
        |state| < threshold 时返回 None（不注入）。
        绝不暴露 valence/arousal 数值。
        """
        magnitude = math.sqrt(
            self.state.valence ** 2 + self.state.arousal ** 2
        )
        if magnitude < self.config.injection_threshold:
            return None

        label = self.state.last_emotion
        return STATE_HINT_MAP.get(label)

    def to_dict(self) -> dict:
        return {
            "valence": round(self.state.valence, 4),
            "arousal": round(self.state.arousal, 4),
            "turn_count": self.state.turn_count,
            "last_emotion": self.state.last_emotion,
            "config": {
                "alpha": self.config.alpha,
                "beta": self.config.beta,
                "gamma": self.config.gamma,
                "noise_sigma": self.config.noise_sigma,
                "injection_threshold": self.config.injection_threshold,
                "save_interval_turns": self.config.save_interval_turns,
                "persist_to_l4": self.config.persist_to_l4,
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EmotionStateTracker":
        cfg_data = data.get("config", {})
        config = EmotionStateConfig(
            alpha=cfg_data.get("alpha", 0.80),
            beta=cfg_data.get("beta", 0.25),
            gamma=cfg_data.get("gamma", 0.12),
            noise_sigma=cfg_data.get("noise_sigma", 0.05),
            injection_threshold=cfg_data.get("injection_threshold", 0.35),
            save_interval_turns=cfg_data.get("save_interval_turns", 5),
            persist_to_l4=cfg_data.get("persist_to_l4", True),
        )
        state = EmotionState(
            valence=data.get("valence", 0.0),
            arousal=data.get("arousal", 0.1),
            turn_count=data.get("turn_count", 0),
            last_emotion=data.get("last_emotion", "neutral"),
        )
        return cls(config=config, initial_state=state)

    def _project(
        self, emotion: str, intensity: float
    ) -> Tuple[float, float]:
        """将情绪标签 × 强度投影到 (v, a) 坐标"""
        base_v, base_a = EMOTION_VA_MAP.get(emotion, (0.0, 0.1))
        return base_v * intensity, base_a * intensity

    def _state_to_label(self) -> str:
        """反查最近邻情绪标签"""
        v, a = self.state.valence, self.state.arousal
        best_label = "neutral"
        best_dist = float("inf")
        for label, (lv, la) in EMOTION_VA_MAP.items():
            dist = (v - lv) ** 2 + (a - la) ** 2
            if dist < best_dist:
                best_dist = dist
                best_label = label
        return best_label
