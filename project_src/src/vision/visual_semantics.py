from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from src.logger import logger


_EMOTION_VA_BASES: Dict[str, tuple[float, float]] = {
    "neutral": (0.0, 0.1),
    "joy": (0.8, 0.6),
    "sadness": (-0.7, 0.2),
    "anger": (-0.4, 0.8),
    "fear": (-0.6, 0.7),
    "surprise": (0.1, 0.8),
    "disgust": (-0.5, 0.4),
    "excitement": (0.7, 0.9),
    "tenderness": (0.6, 0.3),
    "curiosity": (0.5, 0.6),
}


@dataclass(frozen=True)
class _EmotionRule:
    emotion: str
    regexes: tuple[str, ...]
    prototypes: tuple[str, ...]
    vector_threshold: float = 0.62


_EMOTION_RULES: tuple[_EmotionRule, ...] = (
    _EmotionRule(
        emotion="joy",
        regexes=(
            r"微笑",
            r"笑着",
            r"开心",
            r"高兴",
            r"轻松",
            r"愉快",
            r"放松",
            r"满意",
        ),
        prototypes=(
            "人物显得轻松愉快，带有明显笑意",
            "画面呈现开心放松的互动氛围",
            "人物表现出明显的喜悦和轻松状态",
        ),
    ),
    _EmotionRule(
        emotion="sadness",
        regexes=(
            r"难过",
            r"失落",
            r"沮丧",
            r"低落",
            r"伤心",
            r"落寞",
            r"低头沉默",
            r"神情黯淡",
        ),
        prototypes=(
            "人物显得失落低沉，情绪偏悲伤",
            "画面中的姿态和氛围带有明显低落感",
            "人物状态偏安静沉闷，像是有些难过",
        ),
    ),
    _EmotionRule(
        emotion="anger",
        regexes=(
            r"生气",
            r"愤怒",
            r"不满",
            r"恼火",
            r"激烈",
            r"用力",
            r"甩开",
            r"拍打",
            r"皱眉",
        ),
        prototypes=(
            "人物动作显得激烈而带有不满",
            "画面中带有明显恼火或愤怒的迹象",
            "人物表现出强烈不耐烦和对抗感",
        ),
    ),
    _EmotionRule(
        emotion="fear",
        regexes=(
            r"害怕",
            r"紧张",
            r"不安",
            r"警惕",
            r"后退",
            r"躲避",
            r"受惊",
            r"惊慌",
        ),
        prototypes=(
            "人物显得紧张警惕，像是在回避什么",
            "画面中带有受惊或不安的状态",
            "人物动作偏收缩和防御，显得有些害怕",
        ),
    ),
    _EmotionRule(
        emotion="surprise",
        regexes=(
            r"惊讶",
            r"吃惊",
            r"突然",
            r"猛地",
            r"一下子",
        ),
        prototypes=(
            "画面里出现突然变化，带有惊讶感",
            "人物像是被突发情况打断，表现出意外",
            "动作变化非常突然，给人明显的惊讶印象",
        ),
    ),
    _EmotionRule(
        emotion="disgust",
        regexes=(
            r"厌恶",
            r"嫌弃",
            r"反感",
            r"皱鼻",
            r"避开",
        ),
        prototypes=(
            "人物显得排斥和嫌弃",
            "动作中带有明显回避和反感",
            "画面呈现厌恶和躲开的姿态",
        ),
    ),
    _EmotionRule(
        emotion="excitement",
        regexes=(
            r"兴奋",
            r"激动",
            r"活跃",
            r"热烈",
            r"雀跃",
            r"高举",
            r"快速挥动",
        ),
        prototypes=(
            "人物显得兴奋活跃，动作幅度较大",
            "画面氛围偏激动和高唤醒",
            "人物表现出明显的兴奋和热烈状态",
        ),
    ),
    _EmotionRule(
        emotion="tenderness",
        regexes=(
            r"温柔",
            r"轻柔",
            r"安抚",
            r"拥抱",
            r"抚摸",
            r"依偎",
            r"亲近",
        ),
        prototypes=(
            "人物动作显得温柔而亲近",
            "画面带有安抚和柔和互动的感觉",
            "姿态和距离显得亲密而柔和",
        ),
    ),
    _EmotionRule(
        emotion="curiosity",
        regexes=(
            r"观察",
            r"查看",
            r"阅读",
            r"看书",
            r"研究",
            r"端详",
            r"检查",
            r"仔细看",
            r"翻看",
            r"注视",
        ),
        prototypes=(
            "人物像是在认真观察或研究某个对象",
            "画面体现出查看和探索的好奇状态",
            "人物处于阅读、端详或进一步了解的状态",
        ),
    ),
)


class VisualEmotionInferer:
    def __init__(self):
        self._compiled: Dict[str, re.Pattern[str]] = {}
        for rule in _EMOTION_RULES:
            self._compiled[rule.emotion] = re.compile("|".join(rule.regexes))

        self._embedder = None
        self._embedder_failed = False
        self._prototype_vecs: Dict[str, Optional[np.ndarray]] = {
            rule.emotion: None for rule in _EMOTION_RULES
        }
        self._rules_by_emotion = {rule.emotion: rule for rule in _EMOTION_RULES}

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        parts = re.split(r"[。！？；\n]+", text)
        result: List[str] = []
        for part in parts:
            part = part.strip()
            if not part:
                continue
            if len(part) > 60:
                subparts = re.split(r"[，、]+", part)
                result.extend(item.strip() for item in subparts if item.strip())
            else:
                result.append(part)
        return result

    def _ensure_embedder(self) -> None:
        if self._embedder is not None or self._embedder_failed:
            return

        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        try:
            from sentence_transformers import SentenceTransformer

            model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            self._embedder = SentenceTransformer(model_name, local_files_only=True)
            for rule in _EMOTION_RULES:
                self._prototype_vecs[rule.emotion] = np.array(
                    self._embedder.encode(
                        list(rule.prototypes),
                        normalize_embeddings=True,
                    )
                )
        except Exception as exc:
            logger.warning(
                f"Visual emotion embedder is unavailable; fallback to regex-only matching: {exc}"
            )
            self._embedder_failed = True

    def _regex_matches(self, sentences: Sequence[str]) -> List[dict]:
        matches: List[dict] = []
        for sentence in sentences:
            for emotion, pattern in self._compiled.items():
                matched = pattern.search(sentence)
                if matched is None:
                    continue
                matches.append(
                    {
                        "sentence": sentence,
                        "emotion": emotion,
                        "method": "regex",
                        "matched": matched.group(0),
                        "score": 1.0,
                    }
                )
        return matches

    def _vector_matches(self, sentences: Sequence[str]) -> List[dict]:
        self._ensure_embedder()
        if self._embedder is None:
            return []

        sentence_vecs = np.array(
            self._embedder.encode(list(sentences), normalize_embeddings=True)
        )
        matches: List[dict] = []
        for index, sentence in enumerate(sentences):
            sent_vec = sentence_vecs[index]
            for emotion, rule in self._rules_by_emotion.items():
                proto_vecs = self._prototype_vecs.get(emotion)
                if proto_vecs is None or len(proto_vecs) == 0:
                    continue
                sims = proto_vecs @ sent_vec
                best_idx = int(np.argmax(sims))
                best_sim = float(sims[best_idx])
                if best_sim < rule.vector_threshold:
                    continue
                matches.append(
                    {
                        "sentence": sentence,
                        "emotion": emotion,
                        "method": "vector",
                        "matched": rule.prototypes[best_idx],
                        "score": round(best_sim, 4),
                    }
                )
        return matches

    @staticmethod
    def _aggregate_matches(matches: Sequence[dict]) -> tuple[str, float]:
        if not matches:
            return "neutral", 0.0

        stats: Dict[str, dict] = {}
        for match in matches:
            emotion = str(match["emotion"])
            entry = stats.setdefault(
                emotion,
                {"best": 0.0, "count": 0},
            )
            entry["best"] = max(entry["best"], float(match["score"]))
            entry["count"] += 1

        ranked = sorted(
            stats.items(),
            key=lambda item: (
                item[1]["best"] + min(max(item[1]["count"] - 1, 0), 2) * 0.12,
                item[1]["best"],
                item[1]["count"],
            ),
            reverse=True,
        )
        emotion, payload = ranked[0]
        confidence = min(
            1.0,
            float(payload["best"]) + min(max(payload["count"] - 1, 0), 2) * 0.12,
        )
        return emotion, confidence

    def infer(
        self,
        text_parts: Sequence[str],
        *,
        peak_score: float,
        scale: float,
    ) -> dict:
        source_text = " ".join(part.strip() for part in text_parts if part and part.strip()).strip()
        sentences = self._split_sentences(source_text) if source_text else []

        regex_matches = self._regex_matches(sentences)
        matches = list(regex_matches)
        best_emotion, confidence = self._aggregate_matches(regex_matches)
        if confidence < 0.75 and sentences:
            vector_matches = self._vector_matches(sentences)
            matches.extend(vector_matches)
            best_emotion, confidence = self._aggregate_matches(matches)

        if best_emotion == "neutral" and confidence <= 0.0:
            intensity = min(1.0, max(peak_score, 0.0))
            arousal_delta = max(0.0, min(0.18, peak_score * 0.18)) * max(0.0, scale)
            return {
                "valence_delta": 0.0,
                "arousal_delta": round(arousal_delta, 4),
                "emotion": "neutral",
                "intensity": round(intensity, 4),
                "confidence": 0.0,
                "source_text": source_text,
                "matches": [],
            }

        base_valence, base_arousal = _EMOTION_VA_BASES.get(best_emotion, (0.0, 0.1))
        semantic_intensity = min(
            1.0,
            confidence * 0.75 + min(max(peak_score, 0.0), 1.0) * 0.25,
        )
        applied_scale = max(0.0, scale)
        valence_delta = base_valence * semantic_intensity * applied_scale
        arousal_delta = max(
            base_arousal * semantic_intensity,
            min(0.18, peak_score * 0.18),
        ) * applied_scale

        return {
            "valence_delta": round(max(-0.25, min(0.25, valence_delta)), 4),
            "arousal_delta": round(max(-0.25, min(0.25, arousal_delta)), 4),
            "emotion": best_emotion,
            "intensity": round(semantic_intensity, 4),
            "confidence": round(confidence, 4),
            "source_text": source_text,
            "matches": matches,
        }


_INFERER = VisualEmotionInferer()


def infer_visual_emotion_signal(
    text_parts: Sequence[str],
    *,
    peak_score: float,
    scale: float,
) -> dict:
    return _INFERER.infer(
        text_parts,
        peak_score=peak_score,
        scale=scale,
    )


__all__ = [
    "infer_visual_emotion_signal",
]
