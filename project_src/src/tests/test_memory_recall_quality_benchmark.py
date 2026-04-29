import importlib
import json
import os
import sys
import types
import unittest
from itertools import product
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


class _FakeVectorClient:
    def close(self):
        return None


class _QueryAwareMemoryBackend:
    instances = []

    def __init__(self):
        self.records = []
        client = _FakeVectorClient()
        self.vector_store = types.SimpleNamespace(
            client=client,
            get=self._vector_get,
            update=self._vector_update,
        )
        self._telemetry_vector_store = types.SimpleNamespace(client=client)

    @classmethod
    def from_config(cls, _cfg):
        inst = cls()
        cls.instances.append(inst)
        return inst

    def add(
        self,
        messages,
        user_id=None,
        run_id=None,
        agent_id=None,
        metadata=None,
        infer=True,
        **kwargs,
    ):
        del kwargs
        record_id = f"bench-mem-{len(self.records) + 1}"
        self.records.append(
            {
                "id": record_id,
                "memory": messages[0]["content"],
                "user_id": user_id,
                "run_id": run_id,
                "agent_id": agent_id,
                "metadata": metadata or {},
                "score": 1.0,
                "infer": infer,
            }
        )
        return {
            "results": [
                {
                    "id": record_id,
                    "memory": messages[0]["content"],
                    "event": "ADD",
                }
            ]
        }

    def _vector_get(self, vector_id):
        for record in self.records:
            if record["id"] == vector_id:
                payload = dict(record.get("metadata", {}))
                payload["data"] = record["memory"]
                return types.SimpleNamespace(id=record["id"], payload=payload)
        return None

    def _vector_update(self, vector_id, vector=None, payload=None):
        del vector
        for record in self.records:
            if record["id"] == vector_id:
                payload = dict(payload or {})
                record["memory"] = payload.get("data", record["memory"])
                payload.pop("data", None)
                record["metadata"] = payload
                return None

    def _matches_filters(self, record, filters):
        filters = filters or {}
        metadata = record.get("metadata", {})
        for key, expected in filters.items():
            if key in {"user_id", "run_id", "agent_id"}:
                actual = record.get(key)
            else:
                actual = metadata.get(key)
            if actual != expected:
                return False
        return True

    def _score_for_query(self, record, query):
        metadata = record.get("metadata", {})
        scores = metadata.get("benchmark_scores") or {}
        default_score = float(metadata.get("benchmark_default_score", 0.05))
        return float(scores.get(query or "", default_score))

    def get_all(self, *, filters=None, top_k=20, **kwargs):
        del kwargs
        results = []
        for record in self.records:
            if not self._matches_filters(record, filters):
                continue
            results.append(
                {
                    "memory": record["memory"],
                    "id": record["id"],
                    "score": record["score"],
                    "metadata": record.get("metadata", {}),
                    "run_id": record["run_id"],
                    "agent_id": record["agent_id"],
                }
            )
        return {"results": results[:top_k]}

    def search(self, query=None, *, filters=None, top_k=20, **kwargs):
        del kwargs
        results = []
        for record in self.records:
            if not self._matches_filters(record, filters):
                continue
            score = self._score_for_query(record, query)
            record["score"] = score
            results.append(
                {
                    "memory": record["memory"],
                    "id": record["id"],
                    "score": score,
                    "metadata": record.get("metadata", {}),
                    "run_id": record["run_id"],
                    "agent_id": record["agent_id"],
                }
            )
        results.sort(key=lambda item: (-item["score"], item["id"]))
        return {"results": results[:top_k]}


_fake_mem0 = types.ModuleType("mem0")
_fake_mem0.Memory = _QueryAwareMemoryBackend

from configs.model_config import MemoryConfig
from src.logger import logger as app_logger

app_logger.remove()
app_logger.add(sys.stderr, level="WARNING")


DEFAULT_THRESHOLD_CONFIG = {
    "l2_relevance_threshold": 0.40,
    "l2_history_threshold": 0.30,
    "l3_recent_dialog_threshold": 0.35,
    "l3_recent_dialog_history_threshold": 0.20,
    "l3_fact_threshold": 0.50,
    "l3_visual_threshold": 0.50,
    "l3_default_threshold": 0.50,
    "l4_relevance_threshold": 0.70,
    "recovery_threshold": 0.70,
}


class _NoopLLMClient:
    def generate(self, system_prompt, user_input, max_tokens, temperature):
        del system_prompt, user_input, max_tokens, temperature
        return "[]"


def _make_manager(memory_module, config_overrides=None):
    _QueryAwareMemoryBackend.instances.clear()
    memory_module._shared_mem0 = None

    thresholds = dict(DEFAULT_THRESHOLD_CONFIG)
    if config_overrides:
        thresholds.update(config_overrides)

    cfg = MemoryConfig(
        mem0_api_key="recall-quality-benchmark-key",
        lifecycle_store_path=":memory:",
        l3_search_limit=5,
        l4_search_limit=2,
        memory_recall_overfetch=3,
        relevance_threshold=0.50,
        recovery_enabled=True,
        **thresholds,
    )
    manager = memory_module.HierarchicalMemoryManager(
        config=cfg,
        llm_client=_NoopLLMClient(),
        user_id="recall-quality-user",
    )
    manager._should_attempt_recovery = lambda query: str(query or "").startswith("history:")
    return manager


def _add_memory(
    manager,
    content,
    *,
    memory_level,
    memory_type,
    scores,
    run_id=None,
    agent_id=None,
    forgotten=False,
):
    result = manager._mem0_add_content(
        content,
        user_id=manager.user_id,
        run_id=run_id,
        agent_id=agent_id,
        memory_level=memory_level,
        memory_type=memory_type,
        extra_metadata={
            "benchmark_scores": scores,
            "benchmark_default_score": 0.05,
        },
    )
    record = manager.mem0.records[-1]
    if forgotten:
        memory_hash = record["metadata"]["memory_hash"]
        for updated in manager.lifecycle.mark_forgotten(
            [memory_hash],
            now=1_800_000_000.0,
            delay_days=7,
        ):
            manager._sync_lifecycle_record_to_mem0(updated)
    return result


def _metadata_by_content(manager):
    return {
        record["memory"]: dict(record.get("metadata", {}))
        for record in manager.mem0.records
    }


def _contents_in_context(context, contents):
    return [content for content in contents if content in context]


def _mean(values):
    return sum(values) / len(values) if values else 0.0


def _run_recall_quality_benchmark(memory_module, config_overrides=None):
    manager = _make_manager(memory_module, config_overrides=config_overrides)
    alpha_query = "alpha websocket adapter recall"
    beta_query = "beta visual memory recall"
    history_query = "history: alpha retry trace"
    recovery_query = "recover alpha notebook"

    alpha_l2 = "BENCH_ALPHA_L2_STATE websocket adapter timeout decision"
    alpha_fact = "BENCH_ALPHA_L3_FACT websocket timeout is thirty five seconds"
    alpha_dialog = "BENCH_ALPHA_L3_DIALOG raw chat mentioned retry trace id"
    alpha_history_dialog = "BENCH_ALPHA_L3_DIALOG_LOW_SCORE earlier retry trace discussion"
    alpha_l4 = "BENCH_ALPHA_L4_KNOWLEDGE websocket adapter invariant"
    beta_fact = "BENCH_BETA_L3_FACT visual pipeline writes cache manifests"
    beta_visual = "BENCH_BETA_L3_VISUAL screenshot contained the red status banner"
    noise_fact = "BENCH_NOISE_L3_FACT unrelated coffee preference"
    noise_dialog = "BENCH_NOISE_L3_DIALOG unrelated raw small talk"
    borderline_l2_noise = "BENCH_NOISE_L2_STATE unrelated low confidence task snapshot"
    borderline_fact_noise = "BENCH_NOISE_L3_FACT borderline unrelated threshold probe"
    borderline_visual_noise = "BENCH_NOISE_L3_VISUAL unrelated low confidence screenshot"
    borderline_dialog_noise = "BENCH_NOISE_L3_DIALOG borderline raw chat threshold probe"
    borderline_history_dialog_noise = (
        "BENCH_NOISE_L3_DIALOG borderline historical raw chat probe"
    )
    forgotten_relevant = "BENCH_FORGOTTEN_L3_FACT alpha notebook path was shelf seven"
    forgotten_noise = "BENCH_FORGOTTEN_L3_FACT unrelated archived shopping note"

    _add_memory(
        manager,
        alpha_l2,
        memory_level="L2",
        memory_type="state_snapshot",
        run_id=manager.session_id,
        scores={alpha_query: 0.66, history_query: 0.32},
    )
    _add_memory(
        manager,
        alpha_fact,
        memory_level="L3",
        memory_type="fact",
        scores={alpha_query: 0.78, history_query: 0.54},
    )
    _add_memory(
        manager,
        alpha_dialog,
        memory_level="L3",
        memory_type="recent_dialog",
        scores={alpha_query: 0.38, history_query: 0.18},
    )
    _add_memory(
        manager,
        alpha_history_dialog,
        memory_level="L3",
        memory_type="recent_dialog",
        scores={alpha_query: 0.19, history_query: 0.24},
    )
    _add_memory(
        manager,
        alpha_l4,
        memory_level="L4",
        memory_type="knowledge",
        agent_id="neuro_agent",
        scores={alpha_query: 0.73, history_query: 0.72},
    )
    _add_memory(
        manager,
        beta_fact,
        memory_level="L3",
        memory_type="fact",
        scores={beta_query: 0.61},
    )
    _add_memory(
        manager,
        beta_visual,
        memory_level="L3",
        memory_type="visual",
        scores={beta_query: 0.57},
    )
    _add_memory(
        manager,
        noise_fact,
        memory_level="L3",
        memory_type="fact",
        scores={alpha_query: 0.49, beta_query: 0.48, history_query: 0.49},
    )
    _add_memory(
        manager,
        noise_dialog,
        memory_level="L3",
        memory_type="recent_dialog",
        scores={alpha_query: 0.34, beta_query: 0.20, history_query: 0.19},
    )
    _add_memory(
        manager,
        borderline_l2_noise,
        memory_level="L2",
        memory_type="state_snapshot",
        run_id=manager.session_id,
        scores={alpha_query: 0.36, history_query: 0.28},
    )
    _add_memory(
        manager,
        borderline_fact_noise,
        memory_level="L3",
        memory_type="fact",
        scores={alpha_query: 0.46, beta_query: 0.47, history_query: 0.48},
    )
    _add_memory(
        manager,
        borderline_visual_noise,
        memory_level="L3",
        memory_type="visual",
        scores={beta_query: 0.46},
    )
    _add_memory(
        manager,
        borderline_dialog_noise,
        memory_level="L3",
        memory_type="recent_dialog",
        scores={alpha_query: 0.31},
    )
    _add_memory(
        manager,
        borderline_history_dialog_noise,
        memory_level="L3",
        memory_type="recent_dialog",
        scores={history_query: 0.16},
    )
    _add_memory(
        manager,
        forgotten_relevant,
        memory_level="L3",
        memory_type="fact",
        scores={alpha_query: 0.96, recovery_query: 0.82},
        forgotten=True,
    )
    _add_memory(
        manager,
        forgotten_noise,
        memory_level="L3",
        memory_type="fact",
        scores={alpha_query: 0.94, recovery_query: 0.69},
        forgotten=True,
    )

    benchmark_queries = [
        {
            "query": alpha_query,
            "expected": {alpha_l2, alpha_fact, alpha_dialog},
            "irrelevant": {
                noise_fact,
                noise_dialog,
                borderline_l2_noise,
                borderline_fact_noise,
                borderline_dialog_noise,
            },
        },
        {
            "query": beta_query,
            "expected": {beta_fact, beta_visual},
            "irrelevant": {
                noise_fact,
                noise_dialog,
                borderline_fact_noise,
                borderline_visual_noise,
            },
        },
        {
            "query": history_query,
            "expected": {alpha_l2, alpha_fact, alpha_history_dialog},
            "irrelevant": {
                noise_fact,
                noise_dialog,
                borderline_l2_noise,
                borderline_fact_noise,
                borderline_history_dialog_noise,
            },
        },
    ]
    metadata_by_content = _metadata_by_content(manager)
    all_contents = set(metadata_by_content)
    forgotten_contents = {
        content
        for content, metadata in metadata_by_content.items()
        if metadata.get("forgotten")
    }

    query_reports = []
    recall_values = []
    noise_counts = []
    retrieved_counts = []
    recent_dialog_count = 0
    history_item_count = 0
    leaked_forgotten = set()

    for spec in benchmark_queries:
        context = manager.get_system_context(spec["query"])
        retrieved = set(_contents_in_context(context, all_contents))
        history_retrieved = {
            content
            for content in retrieved
            if metadata_by_content[content].get("memory_level") in {"L2", "L3"}
            and not metadata_by_content[content].get("forgotten")
        }
        expected_hits = history_retrieved & spec["expected"]
        unexpected = history_retrieved - spec["expected"]
        noise = unexpected & spec["irrelevant"]
        recall = len(expected_hits) / len(spec["expected"])
        recall_values.append(recall)
        noise_counts.append(len(noise))
        retrieved_counts.append(len(history_retrieved))
        leaked_forgotten.update(retrieved & forgotten_contents)

        for content in history_retrieved:
            metadata = metadata_by_content[content]
            history_item_count += 1
            if (
                metadata.get("memory_level") == "L3"
                and metadata.get("memory_type") == "recent_dialog"
            ):
                recent_dialog_count += 1

        query_reports.append(
            {
                "query": spec["query"],
                "retrieved": sorted(history_retrieved),
                "expected_hits": sorted(expected_hits),
                "noise": sorted(noise),
                "recall": round(recall, 4),
            }
        )

    recovered = manager.recovery_search(
        recovery_query,
        top_k=3,
        now=1_800_000_001.0,
    )
    recovered_contents = {item["memory"] for item in recovered}
    recovery_expected = {forgotten_relevant}
    recovery_hits = recovered_contents & recovery_expected
    recovery_precision = (
        len(recovery_hits) / len(recovered_contents)
        if recovered_contents
        else 0.0
    )
    recovery_recall = len(recovery_hits) / len(recovery_expected)

    report = {
        "query_count": len(benchmark_queries),
        "history_k": manager.config.l3_search_limit,
        "l4_k": manager.config.l4_search_limit,
        "thresholds": {
            key: getattr(manager.config, key)
            for key in DEFAULT_THRESHOLD_CONFIG
        },
        "recall_at_k": round(_mean(recall_values), 4),
        "noise_rate": round(
            sum(noise_counts) / sum(retrieved_counts)
            if sum(retrieved_counts)
            else 0.0,
            4,
        ),
        "forgotten_leakage_rate": round(
            len(leaked_forgotten) / len(forgotten_contents)
            if forgotten_contents
            else 0.0,
            4,
        ),
        "recovery_precision": round(recovery_precision, 4),
        "recovery_recall": round(recovery_recall, 4),
        "l3_recent_dialog_injection_ratio": round(
            recent_dialog_count / history_item_count if history_item_count else 0.0,
            4,
        ),
        "query_reports": query_reports,
        "recovered": sorted(recovered_contents),
    }
    return report


def _with_query_aware_mem0(callback):
    previous_mem0 = sys.modules.get("mem0")
    sys.modules["mem0"] = _fake_mem0
    memory_module = None
    try:
        import src.memory.memory_manager as memory_manager_module

        memory_module = importlib.reload(memory_manager_module)
        return callback(memory_module)
    finally:
        if memory_module is not None:
            memory_module._shared_mem0 = None
        if previous_mem0 is not None:
            sys.modules["mem0"] = previous_mem0
            if memory_module is not None:
                importlib.reload(memory_module)
        else:
            sys.modules.pop("mem0", None)
            sys.modules.pop("src.memory.memory_manager", None)


def run_recall_quality_benchmark(config_overrides=None):
    return _with_query_aware_mem0(
        lambda memory_module: _run_recall_quality_benchmark(
            memory_module,
            config_overrides=config_overrides,
        )
    )


def _threshold_score(report):
    thresholds = report["thresholds"]
    target = {
        "l2_relevance_threshold": 0.40,
        "l2_history_threshold": 0.30,
        "l3_recent_dialog_threshold": 0.35,
        "l3_recent_dialog_history_threshold": 0.20,
        "l3_fact_threshold": 0.50,
        "l3_visual_threshold": 0.50,
        "l4_relevance_threshold": 0.70,
        "recovery_threshold": 0.70,
    }
    drift = sum(abs(float(thresholds[key]) - value) for key, value in target.items())
    caution = sum(float(thresholds[key]) for key in target)
    return (
        report["recall_at_k"],
        report["recovery_recall"],
        report["recovery_precision"],
        -report["noise_rate"],
        -report["forgotten_leakage_rate"],
        -abs(report["l3_recent_dialog_injection_ratio"] - 0.25),
        -drift,
        caution,
    )


def run_threshold_sweep(*, full=False):
    if full:
        sweep_space = {
            "l2_relevance_threshold": [0.35, 0.40, 0.45],
            "l2_history_threshold": [0.25, 0.30, 0.35],
            "l3_recent_dialog_threshold": [0.30, 0.35, 0.40],
            "l3_recent_dialog_history_threshold": [0.15, 0.20, 0.25],
            "l3_fact_threshold": [0.45, 0.50, 0.55],
            "l3_visual_threshold": [0.45, 0.50, 0.55],
            "l4_relevance_threshold": [0.65, 0.70, 0.75],
            "recovery_threshold": [0.65, 0.70, 0.75],
        }
    else:
        sweep_space = {
            "l2_relevance_threshold": [0.35, 0.40],
            "l2_history_threshold": [0.25, 0.30],
            "l3_recent_dialog_threshold": [0.30, 0.35],
            "l3_recent_dialog_history_threshold": [0.15, 0.20],
            "l3_fact_threshold": [0.45, 0.50],
            "l3_visual_threshold": [0.45, 0.50],
            "l4_relevance_threshold": [0.65, 0.70],
            "recovery_threshold": [0.65, 0.70],
        }
    keys = list(sweep_space)

    def _run_all(memory_module):
        reports = []
        accepted = []
        for values in product(*(sweep_space[key] for key in keys)):
            overrides = dict(zip(keys, values))
            report = _run_recall_quality_benchmark(
                memory_module,
                config_overrides=overrides,
            )
            reports.append(report)
            if (
                report["recall_at_k"] >= 0.95
                and report["noise_rate"] <= 0.05
                and report["forgotten_leakage_rate"] == 0.0
                and report["recovery_precision"] >= 0.95
                and report["recovery_recall"] >= 0.95
                and report["l3_recent_dialog_injection_ratio"] <= 0.45
            ):
                accepted.append(report)

        best = sorted(accepted, key=_threshold_score, reverse=True)[:10]
        return {
            "full": full,
            "sweep_space": sweep_space,
            "total_runs": len(reports),
            "accepted_runs": len(accepted),
            "best": [
                {
                    "thresholds": report["thresholds"],
                    "recall_at_k": report["recall_at_k"],
                    "noise_rate": report["noise_rate"],
                    "forgotten_leakage_rate": report["forgotten_leakage_rate"],
                    "recovery_precision": report["recovery_precision"],
                    "recovery_recall": report["recovery_recall"],
                    "l3_recent_dialog_injection_ratio": (
                        report["l3_recent_dialog_injection_ratio"]
                    ),
                    "score": _threshold_score(report),
                }
                for report in best
            ],
        }

    return _with_query_aware_mem0(_run_all)


class MemoryRecallQualityBenchmarkTests(unittest.TestCase):
    def test_recall_quality_benchmark_current_thresholds(self):
        report = run_recall_quality_benchmark()
        if os.environ.get("PRINT_RECALL_BENCHMARK") == "1":
            print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))

        self.assertGreaterEqual(report["recall_at_k"], 0.95)
        self.assertLessEqual(report["noise_rate"], 0.05)
        self.assertEqual(report["forgotten_leakage_rate"], 0.0)
        self.assertGreaterEqual(report["recovery_precision"], 0.95)
        self.assertGreaterEqual(report["recovery_recall"], 0.95)
        self.assertLessEqual(report["l3_recent_dialog_injection_ratio"], 0.45)

    def test_threshold_sweep_recommends_current_thresholds(self):
        report = run_threshold_sweep(
            full=os.environ.get("RUN_FULL_THRESHOLD_SWEEP") == "1"
        )
        if os.environ.get("PRINT_THRESHOLD_SWEEP") == "1":
            print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))

        self.assertGreater(report["accepted_runs"], 0)
        recommended = report["best"][0]["thresholds"]
        expected = {
            "l2_relevance_threshold": 0.40,
            "l2_history_threshold": 0.30,
            "l3_recent_dialog_threshold": 0.35,
            "l3_recent_dialog_history_threshold": 0.20,
            "l3_fact_threshold": 0.50,
            "l3_visual_threshold": 0.50,
            "l4_relevance_threshold": 0.70,
            "recovery_threshold": 0.70,
        }
        for key, value in expected.items():
            self.assertEqual(recommended[key], value)


if __name__ == "__main__":
    os.environ.setdefault("PRINT_RECALL_BENCHMARK", "1")
    os.environ.setdefault("PRINT_THRESHOLD_SWEEP", "1")
    unittest.main()
