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
    """
    情绪状态机配置

    数学模型：二维耦合 Ornstein-Uhlenbeck 过程 + tanh 非线性

    状态方程（向量形式）：
      x_{t+1} = tanh(A·x_t + B·u_t + δ·(μ - x_t) + ε_t)

    其中：
      x = (valence, arousal)       — 内部情绪状态
      u = (user_va, ai_va)         — 外部输入（用户+AI 情绪）
      μ = (μ_v, μ_a)              — 人格基线（set point）
      A = [[α, κ], [κ, α]]        — 状态转移（含 V-A 耦合 κ）
      ε ~ N(0, σ²I)               — 过程噪声

    理论基础：
      - Circumplex VA 空间 (Russell, 1980)
      - Ornstein-Uhlenbeck 均值回归 (hedonic adaptation / set point theory)
      - 情绪惯性 (Kuppens et al., 2010)
      - negativity bias: 负面情绪衰减比正面慢 (Baumeister et al., 2001)
      - V-A 耦合: 高唤醒增强 valence 偏移 (Lang, 1995)

    参数辨识：
      gamma=0.25 由 EKF-MLE 在真实对话日志上估计，
      其余为理论驱动的设计参数。
    """
    alpha: float = 0.75             # 状态惯性（自相关强度）
    beta: float = 0.25              # AI 自身输出影响权重
    gamma: float = 0.25             # 用户情绪影响权重（EKF-MLE 估计）
    delta: float = 0.15             # 均值回归强度（OU 回弹力 θ）
    baseline_valence: float = 0.15  # 人格情绪基线 valence（positive → 略正）
    baseline_arousal: float = 0.28  # 人格情绪基线 arousal（适度活跃）
    kappa: float = 0.05             # V-A 耦合系数（高 arousal 放大 valence 偏移）
    negativity_bias: float = 1.3    # 负面情绪衰减减速因子（>1 表示负面衰减更慢）
    noise_sigma: float = 0.05       # 过程噪声 σ
    injection_threshold: float = 0.12  # |state| 超过此值才注入 prompt
    save_interval_turns: int = 5    # 每 N 轮保存一次到 L4
    persist_to_l4: bool = True


@dataclass
class EmotionState:
    """当前情绪状态"""
    valence: float = 0.0
    arousal: float = 0.1
    turn_count: int = 0
    last_emotion: str = "neutral"
    prev_valence: float = 0.0       # 上一轮 valence（计算趋势用）
    sustained_label: str = "neutral" # 持续停留的情绪标签
    sustained_turns: int = 0         # 在该标签区间停留了多少轮


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
        二维耦合 OU 过程状态更新。

        连续时间模型：
          dv = θ_v·(μ_v - v)dt + κ·(a - μ_a)dt + γ·user_v·dt + β·ai_v·dt + σ·dW_v
          da = θ_a·(μ_a - a)dt + κ·(v - μ_v)dt + γ·user_a·dt + β·ai_a·dt + σ·dW_a

        Euler-Maruyama 离散化（Δt=1 轮）：
          φ = α - δ        (persistence coefficient, 状态记忆强度)
          θ = 1 - φ        (mean-reversion rate, 基线回归速率)

          v_{t+1} = tanh(φ_v·v + θ_v·μ_v + κ·(a - μ_a) + γ·user_v + β·ai_v + ε_v)
          a_{t+1} = tanh(φ·a  + θ·μ_a   + κ·(v - μ_v) + γ·user_a + β·ai_a + ε_a)

        不动点（零输入、线性近似）：
          v* = μ_v,  a* = μ_a  （恰好等于人格基线）

        Negativity bias (Baumeister et al., 2001):
          v < μ_v 时，θ_v 减小为 θ/negativity_bias，使负面状态衰减更慢。

        V-A 耦合 κ·(a - μ_a) (Lang, 1995):
          高 arousal 偏移放大 valence 变化，低 arousal 偏移抑制。
          使用偏差形式 (a-μ_a) 确保耦合不影响不动点位置。
        """
        cfg = self.config
        v, a = self.state.valence, self.state.arousal

        # 投影到 (v, a) 并乘以强度
        user_v, user_a = self._project(user_emotion, user_intensity)
        ai_v, ai_a = self._project(ai_emotion, ai_intensity)

        # 噪声
        eps_v = random.gauss(0, cfg.noise_sigma)
        eps_a = random.gauss(0, cfg.noise_sigma)

        # φ = α - δ (persistence), θ = 1 - φ (mean-reversion rate)
        phi = cfg.alpha - cfg.delta
        theta = 1.0 - phi       # = 1 - α + δ

        # Negativity bias: v < baseline 时回归减慢
        theta_v = theta
        if v < cfg.baseline_valence and cfg.negativity_bias > 1.0:
            theta_v = theta / cfg.negativity_bias
        phi_v = 1.0 - theta_v

        # 耦合 OU 更新（偏差形式）
        v_new = math.tanh(
            phi_v * v
            + theta_v * cfg.baseline_valence
            + cfg.kappa * (a - cfg.baseline_arousal)    # 偏差耦合
            + cfg.beta * ai_v
            + cfg.gamma * user_v
            + eps_v
        )
        a_new = math.tanh(
            phi * a
            + theta * cfg.baseline_arousal
            + cfg.kappa * (v - cfg.baseline_valence)    # 偏差耦合
            + cfg.beta * ai_a
            + cfg.gamma * user_a
            + eps_a
        )

        # 轨迹追踪（供 get_prompt_hint 使用）
        self.state.prev_valence = v
        self.state.valence = v_new
        self.state.arousal = a_new
        self.state.turn_count += 1

        new_label = self._state_to_label()
        if new_label == self.state.sustained_label:
            self.state.sustained_turns += 1
        else:
            self.state.sustained_label = new_label
            self.state.sustained_turns = 1
        self.state.last_emotion = new_label

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
        返回动态自然语言情绪暗示，注入 system prompt。

        利用连续状态的三个维度生成上下文感知的暗示：
          1. 强度（intensity）：距基线的偏离程度 → 轻微/明显
          2. 轨迹（trajectory）：Δv 方向 → 正在好转/正在恶化/刚转变
          3. 持续性（duration）：在同一情绪区间停留的轮次 → 短暂/持续

        绝不暴露 valence/arousal 数值。
        |state - baseline| < threshold 时返回 None（不注入）。
        """
        v = self.state.valence
        a = self.state.arousal
        cfg = self.config

        # 偏离基线的距离（而非原点距离）
        dev_v = v - cfg.baseline_valence
        dev_a = a - cfg.baseline_arousal
        deviation = math.sqrt(dev_v ** 2 + dev_a ** 2)
        if deviation < cfg.injection_threshold:
            return None

        label = self.state.last_emotion
        base_hint = STATE_HINT_MAP.get(label)
        if not base_hint:
            return None

        # --- 强度修饰 ---
        if deviation > 0.5:
            intensity_mod = "（情绪比较强烈）"
        elif deviation > 0.3:
            intensity_mod = ""
        else:
            intensity_mod = "（只是淡淡的）"

        # --- 轨迹修饰 ---
        delta_v = v - self.state.prev_valence
        trajectory_mod = ""
        if self.state.turn_count > 1:
            if abs(delta_v) > 0.08:
                if delta_v > 0 and v < cfg.baseline_valence:
                    trajectory_mod = "，不过情绪正在慢慢好转"
                elif delta_v < 0 and v > cfg.baseline_valence:
                    trajectory_mod = "，但情绪有些往下走"
                elif delta_v > 0 and v >= cfg.baseline_valence:
                    trajectory_mod = "，而且越来越高兴"
                else:
                    trajectory_mod = "，情绪还在继续低落"

        # --- 持续性修饰 ---
        duration_mod = ""
        if self.state.sustained_turns >= 5:
            duration_mod = "。这种状态已经持续了一段时间"
        elif self.state.sustained_turns == 1 and self.state.turn_count > 1:
            duration_mod = "。刚刚情绪发生了转变"

        return base_hint + intensity_mod + trajectory_mod + duration_mod

    def to_dict(self) -> dict:
        return {
            "valence": round(self.state.valence, 4),
            "arousal": round(self.state.arousal, 4),
            "turn_count": self.state.turn_count,
            "last_emotion": self.state.last_emotion,
            "prev_valence": round(self.state.prev_valence, 4),
            "sustained_label": self.state.sustained_label,
            "sustained_turns": self.state.sustained_turns,
            "config": {
                "alpha": self.config.alpha,
                "beta": self.config.beta,
                "gamma": self.config.gamma,
                "delta": self.config.delta,
                "baseline_valence": self.config.baseline_valence,
                "baseline_arousal": self.config.baseline_arousal,
                "kappa": self.config.kappa,
                "negativity_bias": self.config.negativity_bias,
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
            alpha=cfg_data.get("alpha", 0.75),
            beta=cfg_data.get("beta", 0.25),
            gamma=cfg_data.get("gamma", 0.25),
            delta=cfg_data.get("delta", 0.15),
            baseline_valence=cfg_data.get("baseline_valence", 0.15),
            baseline_arousal=cfg_data.get("baseline_arousal", 0.28),
            kappa=cfg_data.get("kappa", 0.05),
            negativity_bias=cfg_data.get("negativity_bias", 1.3),
            noise_sigma=cfg_data.get("noise_sigma", 0.05),
            injection_threshold=cfg_data.get("injection_threshold", 0.12),
            save_interval_turns=cfg_data.get("save_interval_turns", 5),
            persist_to_l4=cfg_data.get("persist_to_l4", True),
        )
        state = EmotionState(
            valence=data.get("valence", 0.0),
            arousal=data.get("arousal", 0.1),
            turn_count=data.get("turn_count", 0),
            last_emotion=data.get("last_emotion", "neutral"),
            prev_valence=data.get("prev_valence", 0.0),
            sustained_label=data.get("sustained_label", "neutral"),
            sustained_turns=data.get("sustained_turns", 0),
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
