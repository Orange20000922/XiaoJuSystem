import enum
import sys
import types
import unittest
from datetime import datetime
from pathlib import Path
from queue import Queue
from threading import Lock
from unittest.mock import patch

_project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_project_root))
_src_root = str(_project_root / "src")

if "src" not in sys.modules:
    _pkg = types.ModuleType("src")
    _pkg.__path__ = [_src_root]
    sys.modules["src"] = _pkg
if "src.core" not in sys.modules:
    _core_pkg = types.ModuleType("src.core")
    _core_pkg.__path__ = [str(_project_root / "src" / "core")]
    sys.modules["src.core"] = _core_pkg

_fake_inference_pipeline = types.ModuleType("src.core.inference_pipeline")


class _FakeChatMode(enum.Enum):
    PRIVATE = "private"
    GROUP = "group"


class _FakeNeuroLikePipeline:
    pass


_fake_inference_pipeline.ChatMode = _FakeChatMode
_fake_inference_pipeline.NeuroLikePipeline = _FakeNeuroLikePipeline
sys.modules["src.core.inference_pipeline"] = _fake_inference_pipeline

from src.agent import agent_loop as agent_loop_module


class _FakeMemory:
    def __init__(self):
        self.close_calls = 0

    def close_session(self):
        self.close_calls += 1


class _FixedDatetime(datetime):
    current = datetime(2026, 1, 1, 3, 0, 0)

    @classmethod
    def now(cls, tz=None):
        del tz
        return cls.current


class AgentLoopMemoryFlushTests(unittest.TestCase):
    def _make_loop(self):
        loop = agent_loop_module.AgentLoop.__new__(agent_loop_module.AgentLoop)
        loop.pipeline = types.SimpleNamespace(memory=_FakeMemory())
        loop.event_queue = Queue()
        loop._last_save_date = None
        loop._chat_dispatch_lock = Lock()
        loop._pending_chat_events = {}
        loop._active_chat_contexts = set()
        return loop

    def test_daily_flush_runs_once_after_three_am(self):
        loop = self._make_loop()

        with patch.object(agent_loop_module, "datetime", _FixedDatetime):
            _FixedDatetime.current = datetime(2026, 1, 1, 3, 5, 0)
            loop._save_memory_to_L3_by_time()
            loop._save_memory_to_L3_by_time()

        self.assertEqual(loop.pipeline.memory.close_calls, 1)
        self.assertEqual(loop._last_save_date, datetime(2026, 1, 1).date())

    def test_daily_flush_skips_when_work_is_pending(self):
        loop = self._make_loop()
        loop._active_chat_contexts.add("group-10001")

        with patch.object(agent_loop_module, "datetime", _FixedDatetime):
            _FixedDatetime.current = datetime(2026, 1, 1, 3, 5, 0)
            loop._save_memory_to_L3_by_time()

        self.assertEqual(loop.pipeline.memory.close_calls, 0)
        self.assertIsNone(loop._last_save_date)

    def test_daily_flush_skips_before_three_am(self):
        loop = self._make_loop()

        with patch.object(agent_loop_module, "datetime", _FixedDatetime):
            _FixedDatetime.current = datetime(2026, 1, 1, 2, 59, 59)
            loop._save_memory_to_L3_by_time()

        self.assertEqual(loop.pipeline.memory.close_calls, 0)
        self.assertIsNone(loop._last_save_date)


if __name__ == "__main__":
    unittest.main()
