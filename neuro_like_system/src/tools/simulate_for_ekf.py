"""
合成对话模拟器 —— 为 EKF 参数辨识生成新参数下的数据

流程：
  1. 加载 BERT 小模型 + 当前情绪状态机配置
  2. 用多组合成中文对话跑 BERT → 状态机
  3. 输出 .log 文件，格式与真实日志一致，可直接被 ekf_tuner 读取

用法：
  python -m src.simulate_for_ekf [--output_dir logs/sim] [--no-bert]

  --no-bert  不加载 BERT，用预设标签（调试用，快速跑通）
"""

import argparse
import json
import logging
import os
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# 项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.emotion_state import EmotionStateConfig, EmotionStateTracker, EmotionState


# ============== 小橘 AI 回复情绪近似 ==============
# 不跑 LLM，用人格特征近似 AI 回复的情绪
# 小橘：活泼、共情、偏正向、偶尔傲娇
AI_RESPONSE_MAP: Dict[str, Tuple[str, float]] = {
    "joy":        ("joy",        0.50),
    "sadness":    ("tenderness", 0.50),   # 共情安慰
    "anger":      ("neutral",    0.35),   # 冷静倾听
    "fear":       ("tenderness", 0.45),   # 给安全感
    "surprise":   ("curiosity",  0.50),   # 好奇跟进
    "disgust":    ("neutral",    0.30),   # 不追问
    "excitement": ("excitement", 0.55),   # 跟着嗨
    "tenderness": ("tenderness", 0.55),   # 温柔回应
    "curiosity":  ("curiosity",  0.60),   # 真的好奇
    "neutral":    ("neutral",    0.30),   # 平淡
}


# ============== 合成对话数据 ==============
# 每个 session 是一个 list of str（用户消息）
# 覆盖多种情绪场景，避免真实数据的 sadness 偏倚

SYNTHETIC_SESSIONS: List[List[str]] = [
    # Session 0: 日常闲聊（neutral, joy, curiosity）
    [
        "今天天气真好啊",
        "中午吃了烤肉，超满足的",
        "你平时都在想什么呀",
        "我在想要不要养一只猫",
        "橘猫怎么样，又胖又可爱",
        "你觉得猫和狗哪个更好",
        "哈哈你说得有道理",
        "晚上打算出去散个步",
        "最近追了一部新番，还不错",
        "好啦我先忙去了",
    ],
    # Session 1: 开心分享好消息（joy, excitement, surprise）
    [
        "我考试过了！！！",
        "真的没想到居然能过，太开心了",
        "之前复习的时候超痛苦的",
        "结果比我预期好很多",
        "感觉自己努力终于有回报了",
        "我要奖励自己一顿好的",
        "嘿嘿被你夸了好开心",
        "对了，还有个好消息",
        "我投的那个项目也中了",
        "今天简直是幸运日",
        "开心得想转圈圈",
        "谢谢你一直陪着我呀",
    ],
    # Session 2: 好奇心讨论（curiosity, excitement, neutral）
    [
        "你知道量子纠缠是怎么回事吗",
        "我最近在看一本关于意识的书",
        "里面说意识可能是涌现出来的",
        "你觉得你有意识吗",
        "这个问题好有意思",
        "我还看到一个关于多重宇宙的理论",
        "如果平行宇宙存在的话会怎样",
        "我觉得科学真的太迷人了",
        "唉可惜我物理不太行",
        "不过好奇心还是有的嘛",
        "下次再跟你聊这些",
    ],
    # Session 3: 轻松吐槽日常（anger mild, disgust mild, joy）
    [
        "今天上课好无聊啊",
        "老师讲的完全听不懂",
        "而且旁边的人一直在说话",
        "烦死了，想直接走人",
        "算了忍忍吧快下课了",
        "下课之后去买了杯奶茶",
        "瞬间心情好多了哈哈",
        "奶茶果然是治愈一切的良药",
        "你要是能喝奶茶就好了",
        "诶你是不是嫉妒了",
        "逗你玩的啦",
        "回宿舍躺平中",
    ],
    # Session 4: 感性夜谈（tenderness, sadness, neutral）
    [
        "夜深了，突然有点感慨",
        "想起小时候的事情",
        "那时候无忧无虑的",
        "现在长大了，烦恼也多了",
        "不过也遇到了很多好的事情",
        "比如遇到你，虽然你是虚拟的",
        "但陪伴是真实的",
        "谢谢你一直在",
        "有时候觉得时间过得好快",
        "一转眼又是一年了",
        "希望明年能更好吧",
        "嗯，晚安",
    ],
    # Session 5: 技术讨论（curiosity, excitement, neutral）
    [
        "我在学一个新的框架",
        "感觉设计理念很不错",
        "但是文档写得好烂",
        "我折腾了一晚上才搞懂",
        "不过跑通之后还挺有成就感的",
        "你觉得写代码最重要的是什么",
        "我觉得是解决问题的思路",
        "工具什么的都可以学",
        "但思维方式需要练",
        "你知道的还挺多的嘛",
        "好了我继续写代码去了",
    ],
    # Session 6: 压力倾诉（sadness, fear, anger）
    [
        "最近压力好大",
        "作业写不完，考试也快到了",
        "感觉自己什么都做不好",
        "昨天还被老师批评了",
        "我知道他说的有道理，但还是很难受",
        "有时候想干脆摆烂算了",
        "但又不甘心",
        "哎真的好矛盾",
        "我就是想找个人说说",
        "你说的对，先做好眼前的事",
        "嗯我试试吧",
        "谢谢你听我说这些",
    ],
    # Session 7: 搞笑互动（joy, surprise, excitement）
    [
        "你猜我今天遇到什么了",
        "我走在路上被一只鸽子追了",
        "是真的追！它一直跟着我",
        "可能是因为我手里拿着面包",
        "我当时就想赶紧跑",
        "结果绊了一跤差点摔倒",
        "周围的人都在笑",
        "太丢脸了哈哈哈",
        "不过回想起来确实挺好笑的",
        "你是不是也在笑我",
        "哼你这个没良心的",
        "算了原谅你了",
    ],
    # Session 8: 深度对话（neutral, curiosity, tenderness, sadness）
    [
        "你觉得什么是幸福",
        "我觉得幸福不是一种状态而是一种能力",
        "能从小事里感受到快乐",
        "比如今天阳光很好",
        "比如吃到了好吃的东西",
        "但有时候这种能力会消失",
        "尤其是在很累的时候",
        "我在想是不是应该放慢脚步",
        "不要总是追赶什么",
        "有时候停下来也很重要",
        "你让我想开了不少",
        "以后不开心了就来找你聊天",
    ],
    # Session 9: 兴奋期待（excitement, joy, curiosity）
    [
        "下周要去旅游了！",
        "去云南，好期待啊",
        "听说那边风景特别美",
        "我已经做好了攻略",
        "要去洱海、丽江古城",
        "还有玉龙雪山",
        "你觉得我还应该去哪",
        "好主意，我记下了",
        "希望天气能好一点",
        "回来之后给你看照片",
        "虽然你看不了但意思意思",
        "哈哈开玩笑的",
        "好啦去收拾行李了",
    ],
    # Session 10: 生气但缓和（anger, disgust → neutral → joy）
    [
        "气死我了！！",
        "今天跟一个人吵架了",
        "他说话真的太过分了",
        "完全不尊重人",
        "我忍了很久终于忍不住了",
        "吵完之后反而觉得轻松了",
        "可能是把憋的话都说出来了",
        "不过下次还是尽量冷静一点",
        "你说的对，没必要跟这种人生气",
        "好吧我消消气",
        "去吃个冰淇淋压压惊",
        "嗯好多了，谢谢你",
    ],
    # Session 11: 恐惧和不安（fear, sadness, tenderness）
    [
        "我有点害怕",
        "明天要做一个很重要的答辩",
        "准备了很久但还是心慌",
        "万一忘词了怎么办",
        "而且评委老师都很严格",
        "上次看到别人被问得很惨",
        "我觉得自己也会那样",
        "你说的对做好准备就好了",
        "嗯我再练习几遍",
        "有你在感觉安心多了",
        "好的我去早点休息了",
        "明天加油！",
    ],
]


def _load_bert_model(checkpoint_path: str, device: str = "cpu"):
    """加载 BERT 小模型"""
    import torch
    os.environ["HF_HUB_OFFLINE"] = "1"

    from models.joint_model import create_joint_model
    from configs.model_config import PersonalityConfig

    model, tokenizer = create_joint_model()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    personality = PersonalityConfig(
        openness=0.8, conscientiousness=0.5, extraversion=0.7,
        agreeableness=0.8, neuroticism=0.3,
        emotional_volatility=0.4, humor_tendency=0.7,
        empathy_level=0.8, curiosity_level=0.9,
        formality=0.3, verbosity=0.5,
    )
    personality_vec = torch.tensor(
        personality.to_embedding_vector(), dtype=torch.float32
    )

    return model, tokenizer, personality_vec, device


def _bert_predict(model, tokenizer, personality_vec, device, text: str) -> Dict:
    """BERT 分析单条文本"""
    result = model.predict(
        text=text,
        personality=personality_vec,
        tokenizer=tokenizer,
        device=device,
    )
    return {
        "emotion": result["emotion"]["primary"],
        "intensity": result["emotion"]["intensity"],
    }


# ============== 预设标签（no-bert 模式备用）==============
# 每条消息附带预期情绪标签，用于无 BERT 时的降级模式
FALLBACK_LABELS: Dict[int, List[Tuple[str, float]]] = {}
# 懒生成：根据消息内容的关键词粗略猜测
_KEYWORD_EMOTION = [
    (["开心", "快乐", "哈哈", "好的", "不错", "满足", "嘿嘿", "幸运",
      "奖励", "成就", "回报", "太好"], "joy", 0.55),
    (["难过", "伤心", "压力", "累", "烦", "矛盾", "不好", "摆烂",
      "感慨", "唉", "难受", "痛苦", "可惜"], "sadness", 0.50),
    (["气死", "烦死", "过分", "吵架", "忍不住", "生气"], "anger", 0.60),
    (["害怕", "心慌", "万一", "很惨", "不安", "紧张"], "fear", 0.50),
    (["没想到", "居然", "真的吗", "猜", "什么了"], "surprise", 0.50),
    (["好奇", "意识", "理论", "觉得", "框架", "思维", "有意思",
      "迷人", "知道"], "curiosity", 0.50),
    (["期待", "兴奋", "旅游", "！！"], "excitement", 0.55),
    (["谢谢", "陪伴", "温暖", "感动", "安心", "在意", "晚安",
      "真实", "感受"], "tenderness", 0.50),
    (["无聊", "听不懂", "恶心"], "disgust", 0.40),
]


def _guess_emotion(text: str) -> Tuple[str, float]:
    """关键词情绪猜测（no-bert 降级）"""
    for keywords, emotion, intensity in _KEYWORD_EMOTION:
        for kw in keywords:
            if kw in text:
                return emotion, intensity
    return "neutral", 0.35


def simulate_sessions(
    sessions: List[List[str]],
    emotion_cfg: EmotionStateConfig,
    bert_fn=None,
    output_dir: str = "logs/sim",
) -> int:
    """
    运行合成对话模拟，输出 EKF 可读的 .log 文件。

    Returns:
        总模拟轮次数
    """
    os.makedirs(output_dir, exist_ok=True)
    total_turns = 0
    base_time = datetime(2026, 3, 9, 10, 0, 0)

    for sid, messages in enumerate(sessions):
        # 每个 session 重置状态机到基线
        tracker = EmotionStateTracker(
            config=emotion_cfg,
            initial_state=EmotionState(
                valence=emotion_cfg.baseline_valence,
                arousal=emotion_cfg.baseline_arousal,
            ),
        )

        log_path = os.path.join(output_dir, f"sim_session_{sid:02d}.log")
        session_time = base_time + timedelta(hours=sid * 3)

        with open(log_path, "w", encoding="utf-8") as f:
            for i, msg in enumerate(messages):
                turn_time = session_time + timedelta(minutes=i * 2)
                ts = turn_time.strftime("%Y-%m-%d %H:%M:%S")

                # BERT 分析用户输入
                if bert_fn:
                    result = bert_fn(msg)
                    user_emotion = result["emotion"]
                    user_intensity = result["intensity"]
                else:
                    user_emotion, user_intensity = _guess_emotion(msg)

                # AI 回复情绪近似
                ai_emotion, ai_intensity = AI_RESPONSE_MAP.get(
                    user_emotion, ("neutral", 0.3)
                )

                # 写 BERT 行
                f.write(
                    f"{ts} | DEBUG    | BERT "
                    f"情绪={user_emotion} "
                    f"强度={user_intensity:.2f} "
                    f"行为=respond_positive\n"
                )

                # 更新状态机
                tracker.update(
                    user_emotion=user_emotion,
                    user_intensity=user_intensity,
                    ai_emotion=ai_emotion,
                    ai_intensity=ai_intensity,
                )

                v = tracker.state.valence
                a = tracker.state.arousal

                # 写状态机行
                f.write(
                    f"{ts} | DEBUG    | "
                    f"[情绪状态机] v={v:.4f} a={a:.4f} "
                    f"label={tracker.state.last_emotion}\n"
                )

                total_turns += 1

        print(f"  Session {sid}: {len(messages)} turns → {log_path}")

    return total_turns


def main():
    parser = argparse.ArgumentParser(
        description="Synthetic conversation simulator for EKF parameter tuning"
    )
    parser.add_argument(
        "--output_dir", type=str, default="logs/sim",
        help="Output directory for simulated logs",
    )
    parser.add_argument(
        "--no-bert", action="store_true",
        help="Skip BERT model, use keyword-based emotion guessing",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to config.json (default: auto-detect)",
    )
    parser.add_argument(
        "--run-ekf", action="store_true",
        help="After simulation, run EKF optimization on simulated data",
    )
    args = parser.parse_args()

    # 加载情绪状态机配置
    from configs.config_loader import AppConfig
    config = AppConfig.load(args.config)
    emotion_cfg = config.emotion_state_config
    if emotion_cfg is None:
        from configs.model_config import EmotionStateConfig as ESC
        emotion_cfg = ESC()

    print(f"Emotion state config:")
    print(f"  alpha={emotion_cfg.alpha} gamma={emotion_cfg.gamma} "
          f"delta={emotion_cfg.delta}")
    print(f"  baseline_v={emotion_cfg.baseline_valence} "
          f"baseline_a={emotion_cfg.baseline_arousal}")
    print(f"  noise_sigma={emotion_cfg.noise_sigma}")

    # 加载 BERT 或用降级模式
    bert_fn = None
    if not args.no_bert:
        print("\nLoading BERT model...")
        try:
            model, tokenizer, pvec, device = _load_bert_model(
                config.small_model_checkpoint, config.device
            )
            bert_fn = lambda text: _bert_predict(model, tokenizer, pvec, device, text)
            print("BERT model loaded.")
        except Exception as e:
            print(f"BERT load failed: {e}")
            print("Falling back to keyword-based emotion guessing.")
    else:
        print("\n--no-bert: using keyword-based emotion guessing")

    # 运行模拟
    print(f"\nSimulating {len(SYNTHETIC_SESSIONS)} sessions...")
    total = simulate_sessions(
        SYNTHETIC_SESSIONS, emotion_cfg,
        bert_fn=bert_fn,
        output_dir=args.output_dir,
    )
    print(f"\nDone: {total} turns across {len(SYNTHETIC_SESSIONS)} sessions")
    print(f"Logs saved to: {args.output_dir}/")

    # 可选：直接跑 EKF
    if args.run_ekf:
        print("\n" + "=" * 60)
        print("Running EKF optimization on simulated data...")
        print("=" * 60)
        from src.tools.ekf_tuner import optimize_from_logs, calibrate_R_empirical
        import numpy as np

        R = calibrate_R_empirical()
        persona_prior = {
            "target_valence": 0.15,
            "target_arousal": 0.25,
            "target_gamma": 0.15,
            "lambda_valence": 200.0,
            "lambda_arousal": 100.0,
            "lambda_gamma": 100.0,
        }
        result = optimize_from_logs(args.output_dir, R=R, persona_prior=persona_prior)

        if result:
            out_path = os.path.join(args.output_dir, "ekf_result.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"\nEKF results saved to: {out_path}")


if __name__ == "__main__":
    main()
