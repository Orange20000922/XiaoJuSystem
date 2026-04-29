import importlib
import os
import shutil
import sys
import time
import unittest
import uuid
from pathlib import Path


_project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_project_root))

try:
    import mem0  # noqa: F401

    _MEM0_AVAILABLE = True
except Exception:
    _MEM0_AVAILABLE = False


class _NoopLLMClient:
    def generate(self, system_prompt, user_input, max_tokens, temperature):
        del system_prompt, user_input, max_tokens, temperature
        return "[]"


@unittest.skipUnless(
    os.environ.get("RUN_MEM0_INTEGRATION") == "1" and _MEM0_AVAILABLE,
    "Set RUN_MEM0_INTEGRATION=1 and run with the project venv to test real Mem0.",
)
class RealMem0ForgettingIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.output_root = (
            _project_root
            / "test_output"
            / f"mem0_forgetting_{os.getpid()}_{time.time_ns()}"
        )
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.manager = None
        self.memory_module = None

    def tearDown(self):
        if self.manager is not None:
            self._close_mem0_clients(self.manager.mem0)
        if self.memory_module is not None:
            self.memory_module._shared_mem0 = None
        shutil.rmtree(self.output_root, ignore_errors=True)

    def _close_mem0_clients(self, mem0_instance):
        stores = [
            getattr(mem0_instance, "vector_store", None),
            getattr(mem0_instance, "__dict__", {}).get("_entity_store"),
            getattr(mem0_instance, "_telemetry_vector_store", None),
        ]
        for store in stores:
            client = getattr(store, "client", None)
            close = getattr(client, "close", None)
            if callable(close):
                close()

    def _make_manager(self):
        from configs.model_config import MemoryConfig
        import src.memory.memory_manager as memory_manager_module

        self.memory_module = importlib.reload(memory_manager_module)
        self.memory_module._shared_mem0 = None

        cfg = MemoryConfig(
            user_id="mem0-integration-user",
            mem0_api_key="integration-test-key",
            vector_store_path=str(self.output_root / "qdrant"),
            collection_name=f"forgetting_{uuid.uuid4().hex}",
            lifecycle_store_path=":memory:",
            enable_forgetting=True,
            min_retention_days=0,
            physical_deletion_delay_days=7.0,
            configured_W_ref=10.0,
            base_weight_l3=0.1,
            lambda_l3=0.0,
            random_jitter_sigma=0.0,
            depth_bias=0.0,
            random_seed=1,
            relevance_threshold=0.0,
            recovery_threshold=0.0,
            memory_recall_overfetch=1,
            l3_search_limit=3,
            l4_search_limit=1,
        )
        return self.memory_module.HierarchicalMemoryManager(
            config=cfg,
            llm_client=_NoopLLMClient(),
            user_id=cfg.user_id,
        )

    def _get_by_content(self, result, content):
        for item in result.get("results", []):
            if item.get("memory") == content:
                return item
        return None

    def test_real_mem0_forgetting_and_recovery_metadata_round_trip(self):
        self.manager = self._make_manager()
        content = (
            "integration forgetting memory alpha: the blue notebook is on shelf seven"
        )

        add_result = self.manager._mem0_add_content(
            content,
            user_id=self.manager.user_id,
            memory_level="L3",
            memory_type="fact",
        )
        mem0_id = str(add_result["results"][0]["id"])

        active = self.manager.mem0.get_all(
            filters={
                "user_id": self.manager.user_id,
                "memory_level": "L3",
                "forgotten": False,
            },
            top_k=10,
        )
        active_item = self._get_by_content(active, content)
        self.assertIsNotNone(active_item)
        active_metadata = active_item["metadata"]
        self.assertEqual(active_item["id"], mem0_id)
        self.assertEqual(active_metadata["mem0_id"], mem0_id)
        self.assertEqual(active_metadata["memory_level"], "L3")
        self.assertFalse(active_metadata["forgotten"])
        self.assertIn("memory_hash", active_metadata)

        summary = self.manager.run_forgetting_maintenance(now=10_000_000_000.0)
        self.assertEqual(summary["selected_forget_count"], 1)
        self.assertEqual(summary["forgotten_count"], 1)
        self.assertEqual(summary["mem0_synced_count"], 1)

        hidden = self.manager.mem0.get_all(
            filters={
                "user_id": self.manager.user_id,
                "memory_level": "L3",
                "forgotten": True,
            },
            top_k=10,
        )
        hidden_item = self._get_by_content(hidden, content)
        self.assertIsNotNone(hidden_item)
        hidden_metadata = hidden_item["metadata"]
        self.assertEqual(hidden_item["id"], mem0_id)
        self.assertTrue(hidden_metadata["forgotten"])
        self.assertIsNotNone(hidden_metadata["forgotten_at"])
        self.assertIsNotNone(hidden_metadata["deleted_after"])

        active_after_forget = self.manager.mem0.get_all(
            filters={
                "user_id": self.manager.user_id,
                "memory_level": "L3",
                "forgotten": False,
            },
            top_k=10,
        )
        self.assertIsNone(self._get_by_content(active_after_forget, content))

        recovered = self.manager.recovery_search(
            "blue notebook shelf seven",
            now=10_000_000_001.0,
        )
        self.assertEqual([item["memory"] for item in recovered], [content])

        restored = self.manager.mem0.get_all(
            filters={
                "user_id": self.manager.user_id,
                "memory_level": "L3",
                "forgotten": False,
            },
            top_k=10,
        )
        restored_item = self._get_by_content(restored, content)
        self.assertIsNotNone(restored_item)
        restored_metadata = restored_item["metadata"]
        self.assertEqual(restored_item["id"], mem0_id)
        self.assertFalse(restored_metadata["forgotten"])
        self.assertIsNone(restored_metadata["forgotten_at"])
        self.assertIsNone(restored_metadata["deleted_after"])
        self.assertEqual(restored_metadata["recall_count"], 1)
        self.assertIsNotNone(restored_metadata["last_recalled_at"])


if __name__ == "__main__":
    unittest.main()
