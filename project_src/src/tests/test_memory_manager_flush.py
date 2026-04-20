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
        self.vector_store = types.SimpleNamespace(client=client)
        self._telemetry_vector_store = types.SimpleNamespace(client=client)

    @classmethod
    def from_config(cls, _cfg):
        inst = cls()
        cls.instances.append(inst)
        return inst

    def add(self, messages, user_id=None, run_id=None, agent_id=None):
        self.records.append(
            {
                "memory": messages[0]["content"],
                "user_id": user_id,
                "run_id": run_id,
                "agent_id": agent_id,
                "score": 1.0,
            }
        )

    def get_all(self, user_id=None, run_id=None, agent_id=None):
        return {
            "results": [
                {"memory": record["memory"], "score": record["score"]}
                for record in self.records
                if (user_id is None or record["user_id"] == user_id)
                and (run_id is None or record["run_id"] == run_id)
                and (agent_id is None or record["agent_id"] == agent_id)
            ]
        }

    def search(self, query=None, user_id=None, run_id=None, agent_id=None, limit=5, filters=None):
        del query
        if filters:
            user_id = filters.get("user_id", user_id)
            run_id = filters.get("run_id", run_id)
            agent_id = filters.get("agent_id", agent_id)
        results = self.get_all(user_id=user_id, run_id=run_id, agent_id=agent_id)["results"]
        return {"results": results[:limit]}


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


if __name__ == "__main__":
    unittest.main()
