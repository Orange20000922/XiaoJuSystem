"""
基于 Extended Kalman Filter (EKF) 的情绪状态机参数辨识

两阶段流程：
  Stage 1: 用 SMP2020 标定 BERT 观测噪声 R（需要 BERT 模型）
  Stage 2: 用真实对话日志（有时序结构）辨识动力学参数 (α, γ, δ, μ_v, μ_a)

数学框架：
  过程方程:  x_t = f(x_{t-1}, u_t; θ) + w_t       x = (valence, arousal)
  观测方程:  z_t = H · x_t + v_t                    z = BERT 输出 → VA

  f(x, u; θ):
    v_new = tanh((α-δ)·v + γ·user_v + δ·μ_v)
    a_new = tanh((α-δ)·a + γ·user_a + δ·μ_a)

  最优 θ* = argmin_θ  Σ_t [log|S_t| + ν_t' S_t⁻¹ ν_t]

用法:
  # Stage 1: 标定 R（需要 BERT checkpoint）
  python ekf_tuner.py calibrate-R --smp_path ../data_set/smp2020/usual_test_labeled.txt

  # Stage 2: 用日志调参
  python ekf_tuner.py optimize --log_dir ../logs

  # 用经验 R 直接跑日志调参（跳过 Stage 1）
  python ekf_tuner.py optimize --log_dir ../logs --empirical-R
"""

import json
import math
import re
import os
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


# ============== VA 映射 ==============

SMP_LABEL_TO_VA: Dict[str, Tuple[float, float]] = {
    "angry":    (-0.4,  0.8),
    "happy":    ( 0.8,  0.6),
    "sad":      (-0.7,  0.2),
    "neutral":  ( 0.0,  0.1),
    "surprise": ( 0.1,  0.8),
    "fear":     (-0.6,  0.7),
}

SYSTEM_LABEL_TO_VA: Dict[str, Tuple[float, float]] = {
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


# ============== 数据加载 ==============

@dataclass
class LogTurn:
    """从日志提取的单轮数据"""
    emotion: str        # BERT 情绪标签
    intensity: float    # BERT 强度
    state_v: float      # 状态机 valence 输出
    state_a: float      # 状态机 arousal 输出


def load_log_sessions(log_dir: str) -> List[List[LogTurn]]:
    """
    从运行日志中提取 BERT 输出 + 状态机轨迹。

    每个 .log 文件视为一个 session。只保留同时有 BERT 和状态机数据的轮次。
    自动解压 .log.zip 文件。
    """
    import zipfile

    bert_pattern = re.compile(
        r'BERT 情绪=(\w+) 强度=([0-9.]+)'
    )
    state_pattern = re.compile(
        r'\[情绪状态机\] v=(-?[0-9.]+) a=([0-9.]+)'
    )

    # 解压 .zip 文件
    for fname in os.listdir(log_dir):
        if fname.endswith('.log.zip'):
            zpath = os.path.join(log_dir, fname)
            with zipfile.ZipFile(zpath) as zf:
                for name in zf.namelist():
                    target = os.path.join(log_dir, name)
                    if not os.path.exists(target):
                        zf.extract(name, log_dir)

    sessions = []
    for fname in sorted(os.listdir(log_dir)):
        if not fname.endswith('.log'):
            continue
        fpath = os.path.join(log_dir, fname)
        if os.path.getsize(fpath) == 0:
            continue

        turns = []
        pending_bert = None

        with open(fpath, encoding='utf-8', errors='ignore') as f:
            for line in f:
                bm = bert_pattern.search(line)
                if bm:
                    pending_bert = (bm.group(1), float(bm.group(2)))
                    continue

                sm = state_pattern.search(line)
                if sm and pending_bert is not None:
                    turns.append(LogTurn(
                        emotion=pending_bert[0],
                        intensity=pending_bert[1],
                        state_v=float(sm.group(1)),
                        state_a=float(sm.group(2)),
                    ))
                    pending_bert = None

        if len(turns) >= 3:
            sessions.append(turns)

    return sessions


def load_smp2020(path: str) -> List[Dict]:
    """加载 SMP2020 数据，返回 {content, label} 列表"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ============== Stage 1: 标定 R ==============

def calibrate_R_empirical() -> np.ndarray:
    """
    基于 BERT emotion_reliability F1 估算 R（不需要跑模型）。

    VA 空间的观测方差通过 F1 → 混淆概率 → VA 误差推导：
    - 高 F1 标签（neutral=0.76）：分类正确率高，VA 投影误差小
    - 低 F1 标签（anger=0.45）：易混淆，VA 投影误差大
    - 加权平均给出整体观测噪声
    """
    reliability = {
        "joy": 0.66, "sadness": 0.66, "anger": 0.45,
        "fear": 0.69, "surprise": 0.62, "disgust": 0.58,
        "neutral": 0.76, "excitement": 0.62,
        "tenderness": 0.68, "curiosity": 0.74,
    }
    # 误差标准差 ≈ (1 - F1) * VA 空间最大距离 / 2
    # VA 空间对角线约 sqrt(2) ≈ 1.41
    errors_v = []
    errors_a = []
    for label, f1 in reliability.items():
        va = SYSTEM_LABEL_TO_VA.get(label, (0, 0.1))
        err = (1 - f1) * 0.7  # 经验缩放
        errors_v.append(err * abs(va[0]) if abs(va[0]) > 0.1 else err * 0.3)
        errors_a.append(err * abs(va[1]) if abs(va[1]) > 0.1 else err * 0.3)

    r_v = np.mean(np.array(errors_v) ** 2)
    r_a = np.mean(np.array(errors_a) ** 2)

    R = np.array([[r_v, 0.0], [0.0, r_a]])
    return R


def calibrate_R_from_smp2020(
    smp_path: str,
    bert_predict_fn,
    max_samples: int = 2000,
) -> np.ndarray:
    """
    Stage 1: 用 SMP2020 标定 BERT → VA 的观测噪声 R。

    对每条 SMP2020 数据：
    - 真值 z_true = SMP 标签 → VA
    - 预测 z_pred = BERT(text) → VA
    - 残差 e = z_true - z_pred

    R = Cov(e) = E[e·e'] (样本协方差)
    """
    data = load_smp2020(smp_path)[:max_samples]

    residuals = []
    for item in data:
        label = item["label"]
        if label not in SMP_LABEL_TO_VA:
            continue

        z_true = np.array(SMP_LABEL_TO_VA[label])

        # BERT 预测
        pred = bert_predict_fn(item["content"])
        if pred is None:
            continue
        pred_label = pred["emotion"]
        pred_va = SYSTEM_LABEL_TO_VA.get(pred_label, (0.0, 0.1))
        z_pred = np.array(pred_va) * pred.get("intensity", 0.5)

        residuals.append(z_true - z_pred)

    if len(residuals) < 10:
        print(f"Warning: only {len(residuals)} valid residuals, using empirical R")
        return calibrate_R_empirical()

    residuals = np.array(residuals)
    R = np.cov(residuals.T)

    print(f"Calibrated R from {len(residuals)} samples:")
    print(f"  R_vv = {R[0, 0]:.4f} (valence noise variance)")
    print(f"  R_aa = {R[1, 1]:.4f} (arousal noise variance)")
    print(f"  R_va = {R[0, 1]:.4f} (cross-covariance)")

    return R


# ============== EKF 核心 ==============

def _f(x: np.ndarray, u_va: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """状态转移 f(x, u; θ)"""
    alpha, gamma, delta, mu_v, mu_a = theta
    v, a = x
    v_new = math.tanh((alpha - delta) * v + gamma * u_va[0] + delta * mu_v)
    a_new = math.tanh((alpha - delta) * a + gamma * u_va[1] + delta * mu_a)
    return np.array([v_new, a_new])


def _jacobian_f(x: np.ndarray, u_va: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """∂f/∂x 雅可比矩阵"""
    alpha, gamma, delta, mu_v, mu_a = theta
    v, a = x
    v_arg = (alpha - delta) * v + gamma * u_va[0] + delta * mu_v
    a_arg = (alpha - delta) * a + gamma * u_va[1] + delta * mu_a
    dv = (alpha - delta) * (1 - math.tanh(v_arg) ** 2)
    da = (alpha - delta) * (1 - math.tanh(a_arg) ** 2)
    return np.array([[dv, 0.0], [0.0, da]])


def ekf_nll_on_logs(
    theta: np.ndarray,
    sessions: List[List[LogTurn]],
    Q: np.ndarray,
    R: np.ndarray,
    persona_prior: Optional[Dict] = None,
) -> float:
    """
    在真实日志数据上计算 EKF 负对数似然 + 人格先验正则化。

    与 SMP2020 版本的关键区别：
    - u_t (用户输入) = BERT 标签 × 强度（独立于观测）
    - z_t (观测) = 状态机实际输出的 (v, a)
    - 这样 u 和 z 是不同信号，EKF 辨识才有意义

    人格先验正则化（选项 C）：
    用高斯先验把 baseline_valence 拉向人格设定值，
    将设计意图编码进目标函数。

    总损失 = -log L(θ|data) + λ_v·(μ_v - μ_v*)² + λ_a·(μ_a - μ_a*)²
                              + λ_γ·(γ - γ*)²

    Args:
        persona_prior: {
            "target_valence": 期望基线 valence（小橘=positive → 0.15）,
            "target_arousal": 期望基线 arousal（适度活跃 → 0.25）,
            "target_gamma":   期望用户影响力（不被绑架 → 0.12~0.18）,
            "lambda_valence":  valence 先验强度,
            "lambda_arousal":  arousal 先验强度,
            "lambda_gamma":    gamma 先验强度,
        }
    """
    alpha, gamma, delta, mu_v, mu_a = theta

    # 参数边界软惩罚
    if not (0.2 <= alpha <= 0.9 and 0.03 <= gamma <= 0.4
            and 0.02 <= delta <= 0.55
            and -0.3 <= mu_v <= 0.5 and 0.05 <= mu_a <= 0.6):
        return 1e8

    H = np.eye(2)
    total_nll = 0.0

    for session in sessions:
        x = np.array([mu_v, mu_a])
        P = np.eye(2) * 0.3

        for turn in session:
            # 用户输入：BERT 标签 → VA × 强度
            va_base = SYSTEM_LABEL_TO_VA.get(turn.emotion, (0.0, 0.1))
            u_va = np.array(va_base) * turn.intensity

            # 观测值：状态机实际输出
            z = np.array([turn.state_v, turn.state_a])

            # Predict
            x_pred = _f(x, u_va, theta)
            F = _jacobian_f(x, u_va, theta)
            P_pred = F @ P @ F.T + Q

            # Update
            nu = z - H @ x_pred
            S = H @ P_pred @ H.T + R
            det_S = np.linalg.det(S)
            if det_S < 1e-12:
                total_nll += 1e6
                continue

            S_inv = np.linalg.inv(S)
            K = P_pred @ H.T @ S_inv
            x = x_pred + K @ nu
            P = (np.eye(2) - K @ H) @ P_pred

            total_nll += 0.5 * (math.log(det_S) + nu @ S_inv @ nu)

    # ── 人格先验正则化 ──
    if persona_prior:
        p = persona_prior
        total_nll += p.get("lambda_valence", 0) * (mu_v - p.get("target_valence", 0.15)) ** 2
        total_nll += p.get("lambda_arousal", 0) * (mu_a - p.get("target_arousal", 0.25)) ** 2
        total_nll += p.get("lambda_gamma", 0)   * (gamma - p.get("target_gamma", 0.15)) ** 2

    return total_nll


# ============== Stage 2: 参数优化 ==============

def optimize_from_logs(
    log_dir: str,
    R: Optional[np.ndarray] = None,
    persona_prior: Optional[Dict] = None,
) -> Dict:
    """
    Stage 2: 用真实对话日志辨识动力学参数。

    日志提供：
    - u_t: BERT 情绪标签 × 强度（用户输入信号）
    - z_t: 状态机 (v, a) 输出（观测值）

    EKF-MLE + 人格先验正则化，寻找使 p(z_1:T | θ) · p(θ) 最大的参数 θ。
    """
    from scipy.optimize import minimize
    from collections import Counter

    sessions = load_log_sessions(log_dir)
    total_turns = sum(len(s) for s in sessions)
    print(f"Loaded {len(sessions)} sessions, {total_turns} turns with paired data")

    if total_turns < 10:
        print("ERROR: Not enough data for optimization (need >= 10 turns)")
        return {}

    for i, s in enumerate(sessions):
        emotions = [t.emotion for t in s]
        print(f"  Session {i}: {len(s)} turns | {dict(Counter(emotions))}")

    Q = np.eye(2) * 0.05 ** 2
    if R is None:
        R = calibrate_R_empirical()

    prior_desc = "OFF" if not persona_prior else (
        f"mu_v→{persona_prior['target_valence']}"
        f"(λ={persona_prior['lambda_valence']}), "
        f"mu_a→{persona_prior['target_arousal']}"
        f"(λ={persona_prior['lambda_arousal']}), "
        f"γ→{persona_prior['target_gamma']}"
        f"(λ={persona_prior['lambda_gamma']})"
    )
    print(f"\nPersona prior: {prior_desc}")

    bounds = [
        (0.3, 0.85),   # alpha
        (0.05, 0.35),  # gamma
        (0.03, 0.50),  # delta
        (-0.2, 0.4),   # mu_v
        (0.05, 0.5),   # mu_a
    ]

    print(f"Starting EKF-MLE optimization...")

    # 多起点搜索
    best_result = None
    best_nll = float("inf")

    starts = [
        np.array([0.7, 0.15, 0.15, 0.15, 0.25]),
        np.array([0.5, 0.10, 0.10, 0.10, 0.30]),
        np.array([0.8, 0.20, 0.08, 0.20, 0.20]),
        np.array([0.6, 0.25, 0.20, 0.05, 0.35]),
    ]

    for i, t0 in enumerate(starts):
        result = minimize(
            ekf_nll_on_logs, t0,
            args=(sessions, Q, R, persona_prior),
            method="L-BFGS-B", bounds=bounds,
            options={"maxiter": 300, "ftol": 1e-8},
        )
        if result.fun < best_nll:
            best_nll = result.fun
            best_result = result
        print(f"  Start {i}: nll={result.fun:.4f} success={result.success}")

    alpha, gamma, delta, mu_v, mu_a = best_result.x

    # 计算纯数据似然（不含先验惩罚），用于评估拟合质量
    data_nll = ekf_nll_on_logs(
        best_result.x, sessions, Q, R, persona_prior=None
    )
    prior_penalty = best_nll - data_nll

    print(f"\n{'=' * 60}")
    print(f"  Optimized parameters ({total_turns} turns, {len(sessions)} sessions)")
    print(f"{'=' * 60}")
    print(f"  alpha (inertia):              {alpha:.4f}")
    print(f"  gamma (user influence):       {gamma:.4f}")
    print(f"  delta (mean reversion):       {delta:.4f}")
    print(f"  baseline_valence (mu_v):      {mu_v:.4f}")
    print(f"  baseline_arousal (mu_a):      {mu_a:.4f}")
    print(f"  effective_inertia (alpha-delta): {alpha - delta:.4f}")
    print(f"  ---")
    print(f"  total_loss:                   {best_nll:.4f}")
    print(f"    data_nll:                   {data_nll:.4f}")
    print(f"    persona_prior_penalty:      {prior_penalty:.4f}")
    print(f"{'=' * 60}")

    return {
        "alpha": round(float(alpha), 4),
        "gamma": round(float(gamma), 4),
        "delta": round(float(delta), 4),
        "baseline_valence": round(float(mu_v), 4),
        "baseline_arousal": round(float(mu_a), 4),
        "neg_log_likelihood": round(float(data_nll), 4),
        "prior_penalty": round(float(prior_penalty), 4),
        "total_loss": round(float(best_nll), 4),
        "data_turns": total_turns,
        "data_sessions": len(sessions),
    }


def main():
    parser = argparse.ArgumentParser(
        description="EKF-based emotion state machine parameter tuning"
    )
    sub = parser.add_subparsers(dest="command")

    # Stage 1
    cal = sub.add_parser("calibrate-R", help="Calibrate observation noise R from SMP2020")
    cal.add_argument("--smp_path", type=str,
                     default="D:/Users/21405/source/repos/MyNeuroLikeSystem/data_set/smp2020/usual_test_labeled.txt")
    cal.add_argument("--output", type=str, default=None)

    # Stage 2
    opt = sub.add_parser("optimize", help="Optimize dynamics params from real logs")
    opt.add_argument("--log_dir", type=str, default="logs")
    opt.add_argument("--empirical-R", action="store_true",
                     help="Use empirical R instead of calibrated R")
    opt.add_argument("--R_path", type=str, default=None,
                     help="Path to calibrated R matrix (JSON)")
    opt.add_argument("--no-prior", action="store_true",
                     help="Disable persona prior regularization")
    opt.add_argument("--lambda-v", type=float, default=50.0,
                     help="Persona prior strength for baseline_valence")
    opt.add_argument("--lambda-a", type=float, default=20.0,
                     help="Persona prior strength for baseline_arousal")
    opt.add_argument("--lambda-g", type=float, default=30.0,
                     help="Persona prior strength for gamma")
    opt.add_argument("--output", type=str, default=None)

    args = parser.parse_args()

    if args.command == "calibrate-R":
        R = calibrate_R_empirical()
        print("Empirical R (no BERT model loaded):")
        print(f"  R = diag({R[0,0]:.4f}, {R[1,1]:.4f})")
        if args.output:
            with open(args.output, "w") as f:
                json.dump({"R_vv": float(R[0, 0]), "R_aa": float(R[1, 1]),
                           "R_va": float(R[0, 1])}, f, indent=2)
            print(f"Saved to {args.output}")

    elif args.command == "optimize":
        R = None
        if args.R_path:
            with open(args.R_path) as f:
                rd = json.load(f)
            R = np.array([[rd["R_vv"], rd.get("R_va", 0)],
                          [rd.get("R_va", 0), rd["R_aa"]]])
        elif args.empirical_R:
            R = calibrate_R_empirical()

        persona_prior = None
        if not args.no_prior:
            persona_prior = {
                "target_valence": 0.15,
                "target_arousal": 0.25,
                "target_gamma": 0.15,
                "lambda_valence": args.lambda_v,
                "lambda_arousal": args.lambda_a,
                "lambda_gamma": args.lambda_g,
            }

        result = optimize_from_logs(args.log_dir, R=R, persona_prior=persona_prior)
        if args.output and result:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"\nResults saved to {args.output}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
