"""集成测试：视觉技能检测 + 执行 + persona 注入 + 情绪推断。

测试覆盖：
  1. VisualSkillDetector 正则匹配（正例/负例/边界）
  2. VisualSkillExecutor mock 执行
  3. PersonaInstance 中 skill 注入到 user_input 的完整链路
  4. VisualEmotionInferer 正则 + 向量相似度匹配
"""
from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from src.vision.visual_skill import VisualSkillDetector, VisualSkillExecutor
from src.vision.visual_types import VisualAnalysis, VisualEvent
from src.vision.visual_semantics import VisualEmotionInferer


def _sep(title: str):
    print(f"\n{'=' * 60}\n  {title}\n{'=' * 60}")


def _check(ok: bool, label: str):
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {label}")
    return ok


# ── 1. Skill Detector regex 匹配 ──

def test_detector():
    _sep("1. VisualSkillDetector 正则匹配")
    d = VisualSkillDetector()
    total, passed = 0, 0

    positives = [
        ("你能看看我在做什么吗？", "看看"),
        ("你看到了什么", "看到"),
        ("画面上有什么东西", "画面"),
        ("你能看见周围有人吗", "你能看 + 周围"),
        ("摄像头能拍到我吗", "摄像头"),
        ("你的眼睛看到了啥", "你的眼"),
        ("我现在在做什么", "在做什么"),
        ("你注意到什么变化了吗", "你注意到"),
        ("屏幕上显示了什么", "屏幕"),
        ("你面前有什么", "你面前"),
        ("我的样子怎么样", "我的样子"),
        ("帮我看看这是什么", "看看"),
        ("你有没有看到刚才发生了什么", "看到"),
        ("你在干什么呀", "在干什么"),
    ]

    for text, hint in positives:
        total += 1
        result = d.detect(text)
        if _check(result, f"正例: \"{text}\" ({hint})"):
            passed += 1

    negatives = [
        "今天天气怎么样",
        "你好呀",
        "帮我写一段代码",
        "最近有什么新闻",
        "我想吃火锅",
        "给我讲个笑话",
        "你觉得AI会取代人类吗",
        "谢谢你的帮助",
        "晚安",
        "明天几点开会",
    ]

    for text in negatives:
        total += 1
        result = d.detect(text)
        if _check(not result, f"负例: \"{text}\""):
            passed += 1

    # 边界用例
    edge_cases = [
        ("", False, "空字符串"),
        ("看", False, "单字不应匹配"),
        ("好看", False, "'好看' 不是视觉请求"),
        ("你看起来不错", False, "'你看起' 不再匹配"),
        ("我看了一本书", False, "'我看' 不在 pattern 中"),
        ("看到你很开心", True, "'看到' 匹配"),
        ("能不能看一下我桌上的东西", True, "'看一下' 新增匹配"),
    ]

    for text, expected, desc in edge_cases:
        total += 1
        result = d.detect(text)
        if _check(result == expected, f"边界: \"{text}\" → {'匹配' if expected else '不匹配'} ({desc})"):
            passed += 1

    print(f"\n  Detector 总计: {passed}/{total}")
    return passed, total


# ── 2. Skill Executor mock 执行 ──

def test_executor():
    _sep("2. VisualSkillExecutor 执行")
    total, passed = 0, 0

    # mock events
    def _make_event(hint: str, score: float) -> VisualEvent:
        analysis = VisualAnalysis(
            scene="测试场景",
            facts=["测试事实"],
            weak_interpretations=[],
            memory_candidate="",
            agent_hint=hint,
        )
        return VisualEvent(
            event_id="test",
            peak_frame_index=0,
            timestamp=0.0,
            peak_score=score,
            representative_frame_index=0,
            analysis=analysis,
        )

    e1 = _make_event("人物在喝水", 0.5)
    e2 = _make_event("人物放下杯子", 0.4)

    # 正常返回
    total += 1
    executor = VisualSkillExecutor(handler=lambda top_k=2: [e1, e2])
    result = executor.execute(top_k=2)
    if _check("人物在喝水" in result and "人物放下杯子" in result,
              f"正常执行: \"{result[:60]}...\""):
        passed += 1

    # 空事件
    total += 1
    executor = VisualSkillExecutor(handler=lambda top_k=2: [])
    result = executor.execute()
    if _check(result == "当前画面暂无显著变化", f"空事件回退: \"{result}\""):
        passed += 1

    # handler 异常
    total += 1
    def _fail(**kwargs): raise RuntimeError("boom")
    executor = VisualSkillExecutor(handler=_fail)
    result = executor.execute()
    if _check(result == "", f"异常处理: 返回空字符串"):
        passed += 1

    # top_k=1 只返回一个
    total += 1
    executor = VisualSkillExecutor(handler=lambda top_k=2: [e1, e2][:top_k])
    result = executor.execute(top_k=1)
    if _check("人物在喝水" in result and "人物放下杯子" not in result,
              f"top_k=1: \"{result}\""):
        passed += 1

    print(f"\n  Executor 总计: {passed}/{total}")
    return passed, total


# ── 3. VisualEmotionInferer 情绪推断 ──

def test_emotion_inferer():
    _sep("3. VisualEmotionInferer 情绪推断")
    total, passed = 0, 0

    inferer = VisualEmotionInferer()
    has_embedder = False

    # regex 匹配测试
    regex_cases = [
        ("人物微笑着挥手", "joy"),
        ("人物看起来非常沮丧和失落", "sadness"),
        ("人物猛地站起来，表情愤怒", "anger"),
        ("人物惊讶地后退了一步", "fear"),  # "后退" 同时匹配 fear regex
        ("人物紧张地四处张望", "fear"),
        ("人物安静地坐在桌前看书", "curiosity"),  # "看书" 语义匹配好奇
        ("人物好奇地凑近屏幕", "curiosity"),
    ]

    for text, expected_emotion in regex_cases:
        total += 1
        result = inferer.infer([text], peak_score=0.5, scale=0.2)
        emotion = result["emotion"]
        method = result.get("method", "")
        if expected_emotion is None:
            ok = emotion == "neutral"
            label = f"regex 无匹配 → neutral: \"{text}\" → {emotion}"
        else:
            ok = emotion == expected_emotion
            label = f"regex \"{text}\" → {emotion} (期望 {expected_emotion}, method={method})"
        if _check(ok, label):
            passed += 1

    # 向量匹配测试（如果 embedder 可用）
    inferer._ensure_embedder()
    has_embedder = inferer._embedder is not None

    if has_embedder:
        print(f"\n  sentence-transformer embedder 已加载，测试向量匹配:")
        vector_cases = [
            ("画面中的人散发出温暖幸福的感觉", "joy"),
            ("人物一动不动地呆坐着，眼神空洞", "neutral"),  # 描述太间接，未达 threshold
            ("场面非常热烈，气氛高涨", "excitement"),
            ("人物温柔地抚摸小猫", "tenderness"),
        ]
        for text, expected in vector_cases:
            total += 1
            result = inferer.infer([text], peak_score=0.5, scale=0.2)
            emotion = result["emotion"]
            confidence = result["confidence"]
            method = result.get("method", "")
            ok = emotion == expected
            if _check(ok, f"向量 \"{text}\" → {emotion} (conf={confidence:.2f}, method={method}, 期望 {expected})"):
                passed += 1
    else:
        print(f"\n  sentence-transformer 不可用，跳过向量匹配测试")
        # 验证 fallback 正常
        total += 1
        result = inferer.infer(["这是一个复杂的场景"], peak_score=0.3, scale=0.2)
        if _check(result["emotion"] == "neutral", f"embedder 不可用时 fallback → neutral"):
            passed += 1

    print(f"\n  EmotionInferer 总计: {passed}/{total}")
    return passed, total


# ── 4. Persona skill 注入链路 (mock) ──

def test_persona_skill_injection():
    _sep("4. Persona skill 注入链路 (模拟)")
    total, passed = 0, 0

    detector = VisualSkillDetector()

    def _make_event(hint: str) -> VisualEvent:
        return VisualEvent(
            event_id="test",
            peak_frame_index=0,
            timestamp=0.0,
            peak_score=0.5,
            representative_frame_index=0,
            analysis=VisualAnalysis(
                scene="书桌前", facts=["人物坐在椅子上"], weak_interpretations=[],
                memory_candidate="", agent_hint=hint,
            ),
        )

    mock_events = [_make_event("人物坐在书桌前敲键盘")]
    executor = VisualSkillExecutor(handler=lambda top_k=2: mock_events)

    # 模拟 persona.chat() 中的检测+注入逻辑
    test_messages = [
        ("你能看看我在做什么吗", True, "应触发"),
        ("今天天气不错", False, "不应触发"),
        ("画面上有什么", True, "应触发"),
        ("帮我写代码", False, "不应触发"),
    ]

    for user_input, should_trigger, desc in test_messages:
        total += 1
        triggered = detector.detect(user_input)

        if triggered:
            context = executor.execute(top_k=2)
            if context:
                modified_input = f"[视觉感知] {context}\n{user_input}"
            else:
                modified_input = user_input
        else:
            modified_input = user_input

        if should_trigger:
            ok = "[视觉感知]" in modified_input and "人物坐在书桌前" in modified_input
            if _check(ok, f"{desc}: \"{user_input}\" → 注入 \"{modified_input[:50]}...\""):
                passed += 1
        else:
            ok = modified_input == user_input
            if _check(ok, f"{desc}: \"{user_input}\" → 未修改"):
                passed += 1

    print(f"\n  Persona 注入 总计: {passed}/{total}")
    return passed, total


def main() -> int:
    results = []
    results.append(test_detector())
    results.append(test_executor())
    results.append(test_emotion_inferer())
    results.append(test_persona_skill_injection())

    total_passed = sum(r[0] for r in results)
    total_all = sum(r[1] for r in results)

    _sep("汇总")
    print(f"  总计: {total_passed}/{total_all}")
    if total_passed == total_all:
        print(f"  ALL PASSED")
        return 0
    else:
        print(f"  FAILED: {total_all - total_passed} 个失败")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
