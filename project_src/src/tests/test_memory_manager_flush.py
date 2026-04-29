import importlib
import sys
import types
import unittest
from dataclasses import dataclass
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


class _FakeMemoryBackend:
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
        record_id = f"mem-{len(self.records) + 1}"
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
        return {"results": [{"id": record_id, "memory": messages[0]["content"], "event": "ADD"}]}

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

    def get_all(self, *, filters=None, top_k=20, **kwargs):
        del kwargs
        filters = filters or {}
        user_id = filters.get("user_id")
        run_id = filters.get("run_id")
        agent_id = filters.get("agent_id")
        memory_level = filters.get("memory_level")
        forgotten = filters.get("forgotten")
        return {
            "results": [
                {
                    "memory": record["memory"],
                    "id": record["id"],
                    "score": record["score"],
                    "metadata": record.get("metadata", {}),
                    "run_id": record["run_id"],
                    "agent_id": record["agent_id"],
                }
                for record in self.records
                if (user_id is None or record["user_id"] == user_id)
                and (run_id is None or record["run_id"] == run_id)
                and (agent_id is None or record["agent_id"] == agent_id)
                and (
                    memory_level is None
                    or record.get("metadata", {}).get("memory_level") == memory_level
                )
                and (
                    forgotten is None
                    or record.get("metadata", {}).get("forgotten") == forgotten
                )
            ][:top_k]
        }

    def search(self, query=None, *, filters=None, top_k=20, **kwargs):
        del query
        del kwargs
        results = self.get_all(filters=filters, top_k=top_k)["results"]
        return {"results": results[:top_k]}


_fake_mem0 = types.ModuleType("mem0")
_fake_mem0.Memory = _FakeMemoryBackend
sys.modules["mem0"] = _fake_mem0

from configs.model_config import MemoryConfig
import src.memory.memory_manager as memory_manager_module


@dataclass
class _FakeTurn:
    user_input: str
    emotion: str
    intensity: float
    behavior: str
    tone: str
    response: str
    timestamp: str = ""
    context_id: str | None = None


class _FakeLLMClient:
    def generate(self, system_prompt, user_input, max_tokens, temperature):
        del user_input, max_tokens, temperature
        if "提取结构化状态" in system_prompt:
            return (
                '{"active_tasks":["task"],"established_facts":["fact"],'
                '"artifacts":[],"open_questions":[],"dead_ends":[]}'
            )
        if "抽取用户的长期偏好" in system_prompt:
            return '["fact-from-state"]'
        if "提取用户的稳定特征" in system_prompt:
            return "[]"
        if "判断用户的情绪状态" in system_prompt:
            return '{"emotion":"neutral","intensity":0.0}'
        return "[]"


class MemoryManagerFlushTests(unittest.TestCase):
    def setUp(self):
        _FakeMemoryBackend.instances.clear()
        self.memory_module = importlib.reload(memory_manager_module)
        self.memory_module._shared_mem0 = None
    def _make_manager(self, **overrides):
        cfg = MemoryConfig(
            mem0_api_key="test-key",
            context_window_tokens=overrides.pop("context_window_tokens", 128_000),
            compression_threshold=overrides.pop("compression_threshold", 0.75),
            compression_ratio=overrides.pop("compression_ratio", 0.5),
            lifecycle_store_path=":memory:",
            **overrides,
        )
        return self.memory_module.HierarchicalMemoryManager(
            config=cfg,
            llm_client=_FakeLLMClient(),
            user_id="tester",
        )

    def _add_turns(self, manager, count):
        for index in range(count):
            manager.add(
                _FakeTurn(
                    user_input=f"user-{index}",
                    emotion="neutral",
                    intensity=0.1,
                    behavior="respond_positive",
                    tone="calm",
                    response=f"assistant-{index}",
                )
            )

    def test_close_session_skips_duplicate_recent_snapshot(self):
        manager = self._make_manager()
        self._add_turns(manager, 3)

        manager.close_session()
        manager.close_session()

        recent = [
            record for record in manager.mem0.records
            if record["memory"].startswith("[最近对话]")
        ]
        self.assertEqual(len(recent), 1)

    def test_close_session_extracts_facts_from_all_compressed_runs(self):
        manager = self._make_manager(
            context_window_tokens=16,
            compression_threshold=0.1,
        )
        self._add_turns(manager, 14)

        compressed = [
            record for record in manager.mem0.records
            if record["run_id"] is not None
        ]
        self.assertGreaterEqual(len(compressed), 1)

        manager.close_session()

        facts = [
            record for record in manager.mem0.records
            if record["memory"].startswith("- fact-from-state")
        ]
        self.assertEqual(len(facts), 1)

    def test_l3_metadata_is_written_and_recall_updates_lifecycle(self):
        manager = self._make_manager()
        self._add_turns(manager, 2)

        manager.close_session()

        recent = next(
            record for record in manager.mem0.records
            if record["memory"].startswith("[最近对话]")
        )
        metadata = recent["metadata"]
        self.assertEqual(metadata["memory_level"], "L3")
        self.assertEqual(metadata["memory_type"], "recent_dialog")
        self.assertFalse(metadata["forgotten"])
        self.assertEqual(metadata["mem0_id"], recent["id"])
        self.assertFalse(recent["infer"])

        context = manager.get_system_context("user-1")
        self.assertIn("[最近对话]", context)

        stored = manager.lifecycle.get(metadata["memory_hash"])
        self.assertIsNotNone(stored)
        self.assertEqual(stored["recall_count"], 1)
        self.assertIsNotNone(stored["last_recalled_at"])
        self.assertEqual(recent["metadata"]["recall_count"], 1)
        self.assertIsNotNone(recent["metadata"]["last_recalled_at"])

    def test_forgotten_l3_memory_is_filtered_from_context(self):
        manager = self._make_manager()
        self._add_turns(manager, 2)
        manager.close_session()
        recent = next(
            record for record in manager.mem0.records
            if record["memory"].startswith("[最近对话]")
        )

        manager.lifecycle.mark_forgotten(
            [recent["metadata"]["memory_hash"]],
            now=1_700_000_000.0,
            delay_days=7,
        )

        context = manager.get_system_context("user-1")
        self.assertNotIn("[最近对话]", context)

    def test_l3_threshold_uses_memory_type(self):
        manager = self._make_manager(
            relevance_threshold=0.90,
            l3_recent_dialog_threshold=0.30,
            l3_fact_threshold=0.50,
            l3_default_threshold=0.50,
            recovery_enabled=False,
        )
        manager._mem0_add_content(
            "recent dialog low score memory",
            user_id=manager.user_id,
            memory_level="L3",
            memory_type="recent_dialog",
        )
        manager._mem0_add_content(
            "fact low score memory",
            user_id=manager.user_id,
            memory_level="L3",
            memory_type="fact",
        )
        for record in manager.mem0.records:
            record["score"] = 0.35

        context = manager.get_system_context("ordinary lookup")

        self.assertIn("recent dialog low score memory", context)
        self.assertNotIn("fact low score memory", context)

    def test_historical_query_lowers_recent_dialog_threshold(self):
        manager = self._make_manager(
            l3_recent_dialog_threshold=0.60,
            l3_recent_dialog_history_threshold=0.20,
            recovery_enabled=False,
        )
        manager._mem0_add_content(
            "historical recent dialog memory",
            user_id=manager.user_id,
            memory_level="L3",
            memory_type="recent_dialog",
        )
        manager.mem0.records[0]["score"] = 0.30

        normal_context = manager.get_system_context("ordinary lookup")
        historical_context = manager.get_system_context("之前聊过什么")

        self.assertNotIn("historical recent dialog memory", normal_context)
        self.assertIn("historical recent dialog memory", historical_context)

    def test_forgetting_plan_reports_candidates_without_mutating(self):
        manager = self._make_manager(
            min_retention_days=0,
            configured_W_ref=10.0,
            base_weight_l3=0.1,
            lambda_l3=0.0,
            random_jitter_sigma=0.0,
            depth_bias=0.0,
            random_seed=1,
        )
        manager._mem0_add_content(
            "old memory",
            user_id=manager.user_id,
            memory_level="L3",
            memory_type="fact",
        )

        summary = manager.forgetting_engine.build_forgetting_plan(
            manager.lifecycle.active_candidates(levels=("L2", "L3")),
            now=10_000_000_000.0,
        )

        self.assertEqual(summary["candidate_count"], 1)
        self.assertEqual(summary["selected_forget_count"], 1)
        self.assertFalse(manager.lifecycle.all_records()[0]["forgotten"])

    def test_forgetting_maintenance_can_mark_logical_forgetting(self):
        manager = self._make_manager(
            enable_forgetting=True,
            min_retention_days=0,
            configured_W_ref=10.0,
            base_weight_l3=0.1,
            lambda_l3=0.0,
            random_jitter_sigma=0.0,
            depth_bias=0.0,
            random_seed=1,
        )
        manager._mem0_add_content(
            "old memory",
            user_id=manager.user_id,
            memory_level="L3",
            memory_type="fact",
        )

        summary = manager.run_forgetting_maintenance(
            now=10_000_000_000.0,
        )

        self.assertEqual(summary["selected_forget_count"], 1)
        self.assertEqual(summary["forgotten_count"], 1)
        self.assertEqual(summary["mem0_synced_count"], 1)
        stored = manager.lifecycle.active_candidates(levels=("L3",))
        self.assertEqual(stored, [])
        all_records = manager.lifecycle.all_records()
        self.assertTrue(all_records[0]["forgotten"])
        self.assertIsNotNone(all_records[0]["forgotten_at"])
        metadata = manager.mem0.records[0]["metadata"]
        self.assertTrue(metadata["forgotten"])
        self.assertIsNotNone(metadata["forgotten_at"])
        self.assertIsNotNone(metadata["deleted_after"])

    def test_recovery_search_reactivates_mem0_metadata(self):
        manager = self._make_manager(
            enable_forgetting=True,
            min_retention_days=0,
            configured_W_ref=10.0,
            base_weight_l3=0.1,
            lambda_l3=0.0,
            random_jitter_sigma=0.0,
            depth_bias=0.0,
            random_seed=1,
            recovery_threshold=0.5,
        )
        manager._mem0_add_content(
            "old recoverable memory",
            user_id=manager.user_id,
            memory_level="L3",
            memory_type="fact",
        )
        manager.run_forgetting_maintenance(
            now=10_000_000_000.0,
        )

        self.assertTrue(manager.mem0.records[0]["metadata"]["forgotten"])
        recovered = manager.recovery_search(
            "还记得 old recoverable memory 吗",
            now=10_000_000_001.0,
        )

        self.assertEqual(len(recovered), 1)
        metadata = manager.mem0.records[0]["metadata"]
        self.assertFalse(metadata["forgotten"])
        self.assertIsNone(metadata["forgotten_at"])
        self.assertIsNone(metadata["deleted_after"])
        self.assertEqual(metadata["recall_count"], 1)
        self.assertIsNotNone(metadata["last_recalled_at"])


if __name__ == "__main__":
    unittest.main()
