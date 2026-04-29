import sys
import types
import unittest
from collections import defaultdict
from pathlib import Path


_project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_project_root))
_src_root = str(_project_root / "src")

if "src" not in sys.modules:
    _pkg = types.ModuleType("src")
    _pkg.__path__ = [_src_root]
    sys.modules["src"] = _pkg
if "src.memory" not in sys.modules:
    _memory_pkg = types.ModuleType("src.memory")
    _memory_pkg.__path__ = [str(_project_root / "src" / "memory")]
    sys.modules["src.memory"] = _memory_pkg

from configs.model_config import MemoryConfig
from src.memory.forgetting import (
    ForgettingEngine,
    MemoryLifecycleStore,
    SECONDS_PER_DAY,
    stable_content_hash,
    stable_memory_hash,
    timestamp_to_iso,
)


class MemoryForgettingDistributionTests(unittest.TestCase):
    REQUIRED_METADATA_KEYS = {
        "memory_level",
        "memory_type",
        "created_at",
        "lifecycle_created_at",
        "last_recalled_at",
        "recall_count",
        "forgotten",
        "forgotten_at",
        "deleted_after",
        "emotion",
        "emotion_intensity",
        "state_label",
        "state_valence",
        "state_arousal",
        "sustained_label",
        "sustained_turns",
        "behavior",
        "tone",
        "context_id",
        "user_id",
        "run_id",
        "agent_id",
        "memory_hash",
        "content_hash",
        "mem0_id",
    }

    def _synthetic_records(self, *, now: float, user_id: str):
        age_buckets = [
            ("0-1d", 0.5),
            ("1-3d", 2.0),
            ("3-14d", 7.0),
            ("14-45d", 21.0),
            ("45-90d", 60.0),
        ]
        memory_types = [
            "state_snapshot",
            "recent_dialog",
            "fact",
            "visual",
            "visual_summary",
        ]
        records = []
        for bucket_index, (bucket_name, age_days) in enumerate(age_buckets):
            for item_index in range(30):
                memory_level = "L2" if item_index % 2 == 0 else "L3"
                memory_type = memory_types[item_index % len(memory_types)]
                run_id = f"synthetic-run-{bucket_index}" if memory_level == "L2" else None
                content = (
                    f"[synthetic][{bucket_name}] memory item {item_index} "
                    f"level={memory_level} type={memory_type}"
                )
                created_at = (
                    now
                    - age_days * SECONDS_PER_DAY
                    - (item_index % 3) * 3600.0
                )
                record = {
                    "memory_level": memory_level,
                    "memory_type": memory_type,
                    "created_at": timestamp_to_iso(created_at),
                    "lifecycle_created_at": created_at,
                    "last_recalled_at": None,
                    "recall_count": item_index % 3,
                    "forgotten": False,
                    "forgotten_at": None,
                    "deleted_after": None,
                    "emotion": ["neutral", "curiosity", "sadness"][item_index % 3],
                    "emotion_intensity": 0.10 + (item_index % 4) * 0.05,
                    "state_label": ["neutral", "focused"][item_index % 2],
                    "state_valence": -0.05 + (item_index % 5) * 0.025,
                    "state_arousal": 0.10 + (item_index % 5) * 0.05,
                    "sustained_label": ["neutral", "curiosity"][item_index % 2],
                    "sustained_turns": item_index % 4,
                    "behavior": ["respond_positive", "seek_clarification"][item_index % 2],
                    "tone": ["calm", "focused"][item_index % 2],
                    "context_id": f"synthetic-context-{bucket_index}",
                    "user_id": user_id,
                    "run_id": run_id,
                    "agent_id": None,
                    "mem0_id": f"synthetic-mem0-{bucket_index}-{item_index}",
                    "_age_bucket": bucket_name,
                    "_age_bucket_index": bucket_index,
                    "_age_days": age_days,
                }
                record["memory_hash"] = stable_memory_hash(
                    user_id=user_id,
                    content=content,
                    memory_level=memory_level,
                    memory_type=memory_type,
                    run_id=run_id,
                    agent_id=None,
                )
                record["content_hash"] = stable_content_hash(user_id, content)
                records.append(record)
        return records

    def test_forgetting_rate_and_time_distribution_on_full_metadata(self):
        now = 1_800_000_000.0
        user_id = "synthetic-distribution-user"
        store = MemoryLifecycleStore(Path(":memory:"))
        records = self._synthetic_records(now=now, user_id=user_id)
        for record in records:
            self.assertFalse(self.REQUIRED_METADATA_KEYS - set(record))
            store.register(record)

        candidates = store.active_candidates(levels=("L2", "L3"))
        self.assertEqual(len(candidates), 150)

        config = MemoryConfig(
            min_retention_days=0,
            base_weight_l2=0.95,
            base_weight_l3=1.00,
            lambda_l2=0.25,
            lambda_l3=0.12,
            alpha_recall=0.05,
            encoding_intensity_coeff=0.05,
            encoding_arousal_coeff=0.03,
            configured_W_ref=1.20,
            max_prune_prob=0.95,
            depth_bias=0.10,
            random_jitter_sigma=0.0,
            random_seed=20,
        )
        summary = ForgettingEngine(config).build_forgetting_plan(
            candidates,
            now=now,
        )

        decision_by_hash = {
            decision["memory_hash"]: decision
            for decision in summary["decisions"]
        }
        forgotten_hashes = {
            memory_hash
            for memory_hash, decision in decision_by_hash.items()
            if decision["final_action"] == "forget"
        }
        forgotten_rate = len(forgotten_hashes) / len(candidates)
        self.assertGreaterEqual(forgotten_rate, 0.35)
        self.assertLessEqual(forgotten_rate, 0.75)

        record_by_hash = {record["memory_hash"]: record for record in candidates}
        kept_ages = [
            record_by_hash[memory_hash]["_age_days"]
            for memory_hash, decision in decision_by_hash.items()
            if decision["final_action"] == "keep"
        ]
        forgotten_ages = [
            record_by_hash[memory_hash]["_age_days"]
            for memory_hash in forgotten_hashes
        ]
        self.assertGreater(
            sum(forgotten_ages) / len(forgotten_ages),
            sum(kept_ages) / len(kept_ages),
        )

        bucket_totals = defaultdict(int)
        bucket_forgotten = defaultdict(int)
        for memory_hash, decision in decision_by_hash.items():
            bucket_index = record_by_hash[memory_hash]["_age_bucket_index"]
            bucket_totals[bucket_index] += 1
            if decision["final_action"] == "forget":
                bucket_forgotten[bucket_index] += 1

        bucket_rates = [
            bucket_forgotten[index] / bucket_totals[index]
            for index in range(5)
        ]
        self.assertEqual(bucket_rates, sorted(bucket_rates))
        self.assertEqual(bucket_rates[0], 0.0)
        self.assertGreaterEqual(bucket_rates[-1], 0.9)

        updated = store.mark_forgotten(
            forgotten_hashes,
            now=now,
            delay_days=7,
        )
        self.assertEqual(len(updated), len(forgotten_hashes))
        self.assertEqual(
            len(store.active_candidates(levels=("L2", "L3"))),
            len(candidates) - len(forgotten_hashes),
        )


if __name__ == "__main__":
    unittest.main()
