"""记忆生命周期元数据的储存、解析与遗忘机制实现。"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence


SECONDS_PER_DAY = 86_400.0


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def stable_content_hash(user_id: str, content: str) -> str:
    raw = f"{user_id}\n{content}".encode("utf-8", errors="replace")
    return hashlib.sha256(raw).hexdigest()


def timestamp_value(value, default: Optional[float] = None) -> Optional[float]:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return default
        try:
            return float(text)
        except ValueError:
            pass
        try:
            if text.endswith("Z"):
                text = text[:-1] + "+00:00"
            return datetime.fromisoformat(text).timestamp()
        except ValueError:
            return default
    return default


def timestamp_to_iso(value: float) -> str:
    return datetime.fromtimestamp(float(value), timezone.utc).isoformat()


def stable_memory_hash(
    *,
    user_id: str,
    content: str,
    memory_level: str,
    memory_type: str,
    run_id: Optional[str] = None,
    agent_id: Optional[str] = None,
) -> str:
    raw = "\n".join(
        [
            user_id,
            memory_level,
            memory_type,
            run_id or "",
            agent_id or "",
            content,
        ]
    ).encode("utf-8", errors="replace")
    return hashlib.sha256(raw).hexdigest()


def default_lifecycle_path(
    *,
    vector_store_path: str,
    collection_name: str,
    user_id: str,
) -> Path:
    base = Path(vector_store_path).expanduser()
    parent = base.parent if base.name else base
    safe_user = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in user_id)
    safe_collection = "".join(
        ch if ch.isalnum() or ch in "-_" else "_" for ch in collection_name
    )
    return parent / f"{safe_collection}_{safe_user}_lifecycle.json"


@dataclass
class ForgettingDecision:
    memory_hash: str
    memory_level: str
    memory_type: str
    retention_weight: float
    forget_pressure: float
    tree_depth: int
    prune_prob: float
    final_action: str
    secondary_filter_result: str = ""


class MemoryLifecycleStore:
    """用于保存mem0没有完整支持的记忆生命周期元数据的回退路径。"""

    def __init__(self, path: Path):
        self.path = path
        self._memory_only = str(path) == ":memory:"
        self.write_failed = False
        self._lock = threading.RLock()
        self._data: Dict = {"version": 1, "records": {}, "last_scan_at": 0.0}
        self._load()

    def _load(self) -> None:
        with self._lock:
            if self._memory_only:
                return
            if not self.path.exists():
                return
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    data.setdefault("records", {})
                    data.setdefault("last_scan_at", 0.0)
                    self._data = data
            except Exception:
                # 元数据文件损坏时按空数据处理；向量库仍然是记忆文本的来源。
                # 新写入的记忆会重新建立生命周期元数据。
                self._data = {"version": 1, "records": {}, "last_scan_at": 0.0}

    def _save(self) -> None:
        if self._memory_only:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            tmp.write_text(
                json.dumps(self._data, ensure_ascii=False, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            os.replace(tmp, self.path)
        except OSError:
            # 文件系统受限时不影响实时对话；保留内存数据做兜底。
            self.write_failed = True
            self._memory_only = True

    def register(self, metadata: Dict) -> None:
        memory_hash = metadata.get("memory_hash")
        if not memory_hash:
            return
        with self._lock:
            records = self._data.setdefault("records", {})
            existing = records.get(memory_hash)
            if existing:
                for key, value in metadata.items():
                    if key not in existing or existing[key] is None:
                        existing[key] = value
                existing.setdefault("forgotten", False)
                existing.setdefault("recall_count", 0)
            else:
                record = dict(metadata)
                record.setdefault("forgotten", False)
                record.setdefault("recall_count", 0)
                records[memory_hash] = record
            self._save()

    def update(self, memory_hash: str, updates: Dict) -> Optional[Dict]:
        with self._lock:
            record = self._data.setdefault("records", {}).get(memory_hash)
            if not record:
                return None
            record.update(updates)
            self._save()
            return dict(record)

    def get(self, memory_hash: str) -> Optional[Dict]:
        with self._lock:
            record = self._data.get("records", {}).get(memory_hash)
            return dict(record) if isinstance(record, dict) else None

    def find_by_content(self, user_id: str, content: str) -> Optional[Dict]:
        content_hash = stable_content_hash(user_id, content)
        with self._lock:
            matches = [
                record
                for record in self._data.get("records", {}).values()
                if record.get("content_hash") == content_hash
            ]
        if not matches:
            return None
        matches.sort(
            key=lambda r: timestamp_value(
                r.get("lifecycle_created_at", r.get("created_at")),
                0.0,
            ),
            reverse=True,
        )
        return dict(matches[0])

    def mark_recalled(self, memory_hashes: Iterable[str], now: Optional[float] = None) -> List[Dict]:
        now = now or time.time()
        changed = False
        updated = []
        with self._lock:
            records = self._data.setdefault("records", {})
            for memory_hash in set(memory_hashes):
                record = records.get(memory_hash)
                if not record:
                    continue
                record["recall_count"] = int(record.get("recall_count") or 0) + 1
                record["last_recalled_at"] = now
                updated.append(dict(record))
                changed = True
            if changed:
                self._save()
        return updated

    def mark_forgotten(
        self,
        memory_hashes: Iterable[str],
        *,
        now: Optional[float] = None,
        delay_days: float = 7.0,
    ) -> List[Dict]:
        now = now or time.time()
        deleted_after = now + delay_days * SECONDS_PER_DAY
        updated = []
        with self._lock:
            records = self._data.setdefault("records", {})
            for memory_hash in set(memory_hashes):
                record = records.get(memory_hash)
                if not record or record.get("forgotten"):
                    continue
                record["forgotten"] = True
                record["forgotten_at"] = now
                record["deleted_after"] = deleted_after
                record["forget_epoch"] = int(record.get("forget_epoch") or 0) + 1
                updated.append(dict(record))
            if updated:
                self._save()
        return updated

    def reactivate(
        self,
        memory_hashes: Iterable[str],
        *,
        now: Optional[float] = None,
    ) -> List[Dict]:
        now = now or time.time()
        updated = []
        with self._lock:
            records = self._data.setdefault("records", {})
            for memory_hash in set(memory_hashes):
                record = records.get(memory_hash)
                if not record:
                    continue
                record["forgotten"] = False
                record["forgotten_at"] = None
                record["deleted_after"] = None
                record["recall_count"] = int(record.get("recall_count") or 0) + 1
                record["last_recalled_at"] = now
                updated.append(dict(record))
            if updated:
                self._save()
        return updated

    def set_last_scan_at(self, when: Optional[float] = None) -> None:
        with self._lock:
            self._data["last_scan_at"] = when or time.time()
            self._save()

    def all_records(self) -> List[Dict]:
        with self._lock:
            return [dict(record) for record in self._data.get("records", {}).values()]

    def active_candidates(self, levels: Sequence[str] = ("L2", "L3")) -> List[Dict]:
        level_set = set(levels)
        return [
            record
            for record in self.all_records()
            if record.get("memory_level") in level_set and not record.get("forgotten", False)
        ]


class ForgettingEngine:

    def __init__(self, config):
        self.config = config

    def retention_weight(
        self,
        record: Dict,
        *,
        now: float,
        current_valence: float = 0.0,
        current_arousal: float = 0.0,
    ) -> float:
        level = str(record.get("memory_level") or "L3").upper()
        base = (
            self.config.base_weight_l2
            if level == "L2"
            else self.config.base_weight_l3
        )
        decay = (
            self.config.lambda_l2
            if level == "L2"
            else self.config.lambda_l3
        )
        anchor = timestamp_value(record.get("last_recalled_at"))
        if anchor is None:
            anchor = timestamp_value(record.get("lifecycle_created_at"))
        if anchor is None:
            anchor = timestamp_value(record.get("created_at"), now)
        delta_days = max(0.0, (now - anchor) / SECONDS_PER_DAY)
        recall_count = max(0, int(record.get("recall_count") or 0))
        emotion_intensity = clamp(float(record.get("emotion_intensity") or 0.0), 0.0, 1.0)
        state_arousal = float(record.get("state_arousal") or 0.0)

        encode_factor = 1.0
        encode_factor += self.config.encoding_intensity_coeff * emotion_intensity
        encode_factor += self.config.encoding_arousal_coeff * abs(state_arousal)
        encode_factor = clamp(encode_factor, 0.7, 1.5)

        scan_factor = 1.0
        scan_factor += self.config.mood_coeff_v * current_valence
        scan_factor += self.config.mood_coeff_a * current_arousal
        scan_factor = clamp(scan_factor, 0.8, 1.2)

        recall_factor = 1.0 + self.config.alpha_recall * math.log1p(recall_count)
        return base * math.exp(-decay * delta_days) * recall_factor * encode_factor * scan_factor

    def build_forgetting_plan(
        self,
        records: Sequence[Dict],
        *,
        now: Optional[float] = None,
        current_valence: float = 0.0,
        current_arousal: float = 0.0,
    ) -> Dict:
        """根据当前记忆记录和情绪状态计算每条记忆的保留权重，并产生遗忘决策逻辑，供memory manager执行。"""
        now = now or time.time()
        rng = random.Random(self.config.random_seed)
        protected_seconds = self.config.min_retention_days * SECONDS_PER_DAY

        weighted: List[Dict] = []
        protected: List[ForgettingDecision] = []
        for record in records:
            weight = self.retention_weight(
                record,
                now=now,
                current_valence=current_valence,
                current_arousal=current_arousal,
            )
            enriched = dict(record)
            enriched["retention_weight"] = weight
            created_at = timestamp_value(record.get("lifecycle_created_at"))
            if created_at is None:
                created_at = timestamp_value(record.get("created_at"), now)
            age = now - created_at
            keep_reason = self._hard_keep_reason(record, age, protected_seconds)
            if keep_reason:
                protected.append(
                    ForgettingDecision(
                        memory_hash=str(record.get("memory_hash")),
                        memory_level=str(record.get("memory_level")),
                        memory_type=str(record.get("memory_type")),
                        retention_weight=weight,
                        forget_pressure=0.0,
                        tree_depth=0,
                        prune_prob=0.0,
                        final_action="keep",
                        secondary_filter_result=keep_reason,
                    )
                )
                continue
            weighted.append(enriched)

        if not weighted:
            return self._summary([], protected)

        scan_max = max(float(record["retention_weight"]) for record in weighted)
        w_ref = max(scan_max, float(self.config.configured_W_ref))
        decisions: List[ForgettingDecision] = []

        for depth, bucket in enumerate(self._weight_buckets(weighted, w_ref), start=1):
            if not bucket:
                continue
            avg_w = sum(float(record["retention_weight"]) for record in bucket) / len(bucket)
            forget_pressure = 1.0 - clamp(avg_w / w_ref, 0.0, 1.0)
            jitter = rng.gauss(0.0, self.config.random_jitter_sigma)
            prune_prob = clamp(
                forget_pressure + self.config.depth_bias * depth + jitter,
                0.0,
                self.config.max_prune_prob,
            )
            prune_bucket = rng.random() < prune_prob
            for record in bucket:
                action = "forget" if prune_bucket else "keep"
                decisions.append(
                    ForgettingDecision(
                        memory_hash=str(record.get("memory_hash")),
                        memory_level=str(record.get("memory_level")),
                        memory_type=str(record.get("memory_type")),
                        retention_weight=float(record["retention_weight"]),
                        forget_pressure=forget_pressure,
                        tree_depth=depth,
                        prune_prob=prune_prob,
                        final_action=action,
                    )
                )

        return self._summary(decisions, protected)

    def _hard_keep_reason(
        self,
        record: Dict,
        age_seconds: float,
        protected_seconds: float,
    ) -> str:
        if str(record.get("memory_level") or "").upper() == "L4":
            return "level_l4"
        if age_seconds < protected_seconds:
            return "min_retention"
        memory_type = str(record.get("memory_type") or "")
        if memory_type in {"capability", "profile", "safety", "active_task"}:
            return f"memory_type:{memory_type}"
        return ""

    def _weight_buckets(self, records: Sequence[Dict], w_ref: float) -> List[List[Dict]]:
        buckets: List[List[Dict]] = [[], [], [], []]
        for record in sorted(records, key=lambda r: float(r["retention_weight"]), reverse=True):
            norm = clamp(float(record["retention_weight"]) / w_ref, 0.0, 1.0)
            if norm >= 0.75:
                buckets[0].append(record)
            elif norm >= 0.50:
                buckets[1].append(record)
            elif norm >= 0.25:
                buckets[2].append(record)
            else:
                buckets[3].append(record)
        return buckets

    def _summary(
        self,
        decisions: Sequence[ForgettingDecision],
        protected: Sequence[ForgettingDecision],
    ) -> Dict:
        all_decisions = list(protected) + list(decisions)
        weights = [decision.retention_weight for decision in all_decisions]
        selected_forget = [
            decision for decision in all_decisions
            if decision.final_action == "forget"
        ]
        return {
            "candidate_count": len(all_decisions),
            "selected_forget_count": len(selected_forget),
            "avg_weight": sum(weights) / len(weights) if weights else 0.0,
            "min_weight": min(weights) if weights else 0.0,
            "max_weight": max(weights) if weights else 0.0,
            "decisions": [decision.__dict__ for decision in all_decisions],
        }
