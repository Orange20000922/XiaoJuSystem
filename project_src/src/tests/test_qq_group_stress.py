import asyncio
import importlib
import json
import random
import sys
import types
import unittest
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import Optional

_project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_project_root))

_src_root = str(_project_root / "src")
for name, rel in {
    "src": "src",
    "src.adapters": "src/adapters",
    "src.agent": "src/agent",
    "src.attention": "src/attention",
    "src.core": "src/core",
    "src.media": "src/media",
    "src.memory": "src/memory",
}.items():
    if name not in sys.modules:
        mod = types.ModuleType(name)
        mod.__path__ = [str(_project_root / rel)]
        sys.modules[name] = mod


class _FakeMemoryBackend:
    instances = []

    def __init__(self):
        self.records = []
        client = types.SimpleNamespace(close=lambda: None)
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

    def get_all(self, *, filters=None, top_k=20, **kwargs):
        del kwargs
        filters = filters or {}
        user_id = filters.get("user_id")
        run_id = filters.get("run_id")
        agent_id = filters.get("agent_id")
        return {
            "results": [
                {"memory": record["memory"], "score": record["score"]}
                for record in self.records
                if (user_id is None or record["user_id"] == user_id)
                and (run_id is None or record["run_id"] == run_id)
                and (agent_id is None or record["agent_id"] == agent_id)
            ][:top_k]
        }

    def search(self, query=None, *, filters=None, top_k=20, **kwargs):
        del query
        return {
            "results": self.get_all(
                filters=filters,
                top_k=top_k,
            )["results"][:top_k]
        }


_fake_mem0 = types.ModuleType("mem0")
_fake_mem0.Memory = _FakeMemoryBackend
sys.modules["mem0"] = _fake_mem0


class _ChatMode(Enum):
    PRIVATE = "private"
    GROUP = "group"


_fake_inference = types.ModuleType("src.core.inference_pipeline")
_fake_inference.ChatMode = _ChatMode
_fake_inference.NeuroLikePipeline = object
sys.modules["src.core.inference_pipeline"] = _fake_inference


_fake_image_utils = types.ModuleType("src.media.image_utils")


@dataclass
class _ImageResult:
    base64_data: str = ""
    media_type: str = "image/jpeg"
    original_url: str = ""


_fake_image_utils.process_image_url = lambda *args, **kwargs: None
_fake_image_utils.ImageResult = _ImageResult
_fake_image_utils.PILLOW_AVAILABLE = False
sys.modules["src.media.image_utils"] = _fake_image_utils


class _ConnectionClosed(Exception):
    def __init__(self, code=1000, reason=""):
        super().__init__(reason)
        self.code = code
        self.reason = reason


_websockets = types.ModuleType("websockets")
_ws_asyncio = types.ModuleType("websockets.asyncio")
_ws_asyncio_server = types.ModuleType("websockets.asyncio.server")
_ws_exceptions = types.ModuleType("websockets.exceptions")
_ws_http11 = types.ModuleType("websockets.http11")


class _ServerConnection:
    async def send(self, _payload):
        return None


async def _serve(*args, **kwargs):  # pragma: no cover - 仅为导入占位
    del args, kwargs
    raise RuntimeError("serve() should not be used in stress test")


class _Request:
    path = "/xm"


class _Response:
    pass


_ws_asyncio_server.serve = _serve
_ws_asyncio_server.ServerConnection = _ServerConnection
_ws_exceptions.ConnectionClosed = _ConnectionClosed
_ws_http11.Request = _Request
_ws_http11.Response = _Response
sys.modules["websockets"] = _websockets
sys.modules["websockets.asyncio"] = _ws_asyncio
sys.modules["websockets.asyncio.server"] = _ws_asyncio_server
sys.modules["websockets.exceptions"] = _ws_exceptions
sys.modules["websockets.http11"] = _ws_http11

from configs.model_config import AgentConfig, AttentionConfig, ImageConfig, MemoryConfig, QQBotConfig

import src.memory.memory_manager as memory_manager_module
import src.attention.attention_tracker as attention_tracker_module
from src.agent.agent_loop import AgentLoop
from src.adapters.qq_adapter import QQBotAdapter


@dataclass
class _FakeTurn:
    user_input: str
    emotion: str
    intensity: float
    behavior: str
    tone: str
    response: str
    timestamp: str = ""
    context_id: Optional[str] = None


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


class _FakePipeline:
    def __init__(self):
        self.personality = types.SimpleNamespace(name="stress-qq")
        memory_cfg = MemoryConfig(
            mem0_api_key="test-key",
            context_window_tokens=128_000,
            compression_threshold=0.95,
        )
        self.memory = memory_manager_module.HierarchicalMemoryManager(
            config=memory_cfg,
            llm_client=_FakeLLMClient(),
            user_id="stress-qq",
        )
        self.attention_tracker = attention_tracker_module.AttentionTracker(
            AttentionConfig(
                cooldown_seconds=0,
                non_focus_reply_interval=0,
                mentioned_user_ttl=300,
                context_window_messages=50,
            )
        )
        self.processed_count = 0
        self.history_inconsistencies = []
        self.reply_records = []
        self._seq = 0
        self._lock = Lock()

    def _next_seq(self) -> int:
        with self._lock:
            self._seq += 1
            return self._seq

    def _mark_processed(self, context_id, user_id, responded, is_mentioned):
        with self._lock:
            self.processed_count += 1
            self.reply_records.append(
                {
                    "context_id": context_id,
                    "user_id": user_id,
                    "responded": responded,
                    "is_mentioned": is_mentioned,
                }
            )

    def chat(
        self,
        user_input: str,
        *,
        is_mentioned: bool,
        chat_mode: _ChatMode,
        use_fusion,
        images,
        user_id: Optional[int],
        user_name: Optional[str],
        context_id: Optional[str],
        visual_direct: bool,
    ):
        del use_fusion, images, visual_direct

        chat_mode_value = getattr(chat_mode, "value", chat_mode)
        is_group_chat = chat_mode_value == _ChatMode.GROUP.value

        in_attention_focus = False
        if is_group_chat and not is_mentioned and user_id is not None:
            user = self.attention_tracker.get_user_attention(
                user_id=user_id,
                context_key=context_id,
            )
            in_attention_focus = (
                user is not None
                and self.attention_tracker.config.track_mentioned_users
                and user.is_mentioned_active(self.attention_tracker.config.mentioned_user_ttl)
            )

        history = self.memory.get_messages_history(context_id=context_id, max_turns=40)
        with self.memory._memory_lock:
            expected_messages = 0
            for turn in self.memory.working_memory:
                if turn.context_id == context_id:
                    if turn.user_input:
                        expected_messages += 1
                    if turn.response:
                        expected_messages += 1
        if len(history) != expected_messages:
            self.history_inconsistencies.append(
                {
                    "context_id": context_id,
                    "expected": expected_messages,
                    "actual": len(history),
                }
            )

        behavior_type = "ask_question" if ("?" in user_input or "？" in user_input) else "respond_positive"
        intensity = 0.9 if is_mentioned else 0.3

        if is_group_chat and user_id is not None:
            respond = self.attention_tracker.should_respond(
                user_id=user_id,
                emotion_intensity=intensity,
                behavior_type=behavior_type,
                is_mentioned=is_mentioned,
                in_attention_focus=in_attention_focus,
                context_key=context_id,
            )
        else:
            respond = True

        if is_group_chat and user_id is not None:
            self.attention_tracker.on_message(
                user_id=user_id,
                user_name=user_name or str(user_id),
                is_mentioned=is_mentioned,
                context_key=context_id,
            )

        response = ""
        if respond:
            response = f"ack:{context_id}:{user_id}:{self._next_seq()}"
            self.memory.add(
                _FakeTurn(
                    user_input=user_input,
                    emotion="neutral",
                    intensity=intensity,
                    behavior=behavior_type,
                    tone="calm",
                    response=response,
                    context_id=context_id,
                )
            )
            if is_group_chat and user_id is not None:
                self.attention_tracker.on_reply(user_id, context_key=context_id)

        self._mark_processed(context_id, user_id, respond, is_mentioned)
        return {
            "should_respond": respond,
            "response": response,
            "emotion": {"primary": "neutral", "intensity": intensity},
            "behavior": {"type": behavior_type, "tone": "calm"},
        }

    def close(self):
        self.memory.close_session()


class _FakeWSConnection:
    def __init__(self):
        self.sent = []

    async def send(self, payload: str):
        self.sent.append(json.loads(payload))


def _make_group_event(bot_qq: int, group_id: int, user_id: int, text: str, *, mentioned: bool):
    raw = f"[CQ:at,qq={bot_qq}] {text}" if mentioned else text
    return {
        "message_type": "group",
        "user_id": user_id,
        "group_id": group_id,
        "raw_message": raw,
        "sender": {
            "card": f"user_{user_id}",
            "nickname": f"user_{user_id}",
        },
    }


class QQGroupStressTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        _FakeMemoryBackend.instances.clear()
        importlib.reload(memory_manager_module)
        memory_manager_module._shared_mem0 = None

    async def _build_runtime(self, *, reply_with_at: bool = False, max_concurrent_chats: int = 12):
        pipeline = _FakePipeline()
        agent_loop = AgentLoop(
            pipeline=pipeline,
            config=AgentConfig(
                proactive_level="off",
                tick_interval=0.01,
                max_concurrent_chats=max_concurrent_chats,
            ),
            output_callback=lambda text, ctx: None,
        )

        adapter = QQBotAdapter(
            QQBotConfig(
                bot_qq=999001,
                owner_qq=10000,
                reply_with_at=reply_with_at,
                max_message_length=200,
            ),
            image_config=ImageConfig(enabled=False),
        )
        adapter.agent_loop = agent_loop
        adapter.async_loop = asyncio.get_running_loop()
        adapter.ws_connection = _FakeWSConnection()
        agent_loop.output_callback = adapter._make_output_callback()
        agent_loop.start()
        return adapter, agent_loop, pipeline

    async def _wait_processed(self, pipeline: _FakePipeline, expected: int, timeout: float = 10.0):
        start = asyncio.get_running_loop().time()
        while pipeline.processed_count < expected:
            if asyncio.get_running_loop().time() - start > timeout:
                raise TimeoutError(
                    f"processed_count={pipeline.processed_count}, expected={expected}"
                )
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.05)

    async def test_group_memory_isolation_stays_stable_under_10_and_20_groups(self):
        for group_count in (10, 20):
            with self.subTest(group_count=group_count):
                adapter, agent_loop, pipeline = await self._build_runtime()
                try:
                    users_per_group = 5
                    rounds = 4
                    events = []
                    for round_index in range(rounds):
                        for group_offset in range(group_count):
                            group_id = 700000 + group_offset
                            for user_offset in range(users_per_group):
                                user_id = group_id * 10 + user_offset
                                events.append(
                                    _make_group_event(
                                        adapter.config.bot_qq,
                                        group_id,
                                        user_id,
                                        f"group={group_id} user={user_id} round={round_index}?",
                                        mentioned=True,
                                    )
                                )

                    random.Random(42).shuffle(events)
                    for index in range(0, len(events), 100):
                        batch = events[index:index + 100]
                        await asyncio.gather(*(adapter._on_message(event) for event in batch))

                    await self._wait_processed(pipeline, len(events))

                    self.assertEqual(pipeline.history_inconsistencies, [])
                    self.assertEqual(len(adapter.ws_connection.sent), len(events))
                    self.assertEqual(len(pipeline.memory.working_memory), len(events))

                    sent_by_group = Counter(
                        action["params"]["group_id"]
                        for action in adapter.ws_connection.sent
                        if action["action"] == "send_group_msg"
                    )
                    self.assertEqual(len(sent_by_group), group_count)
                    for count in sent_by_group.values():
                        self.assertEqual(count, users_per_group * rounds)

                    agent_loop.stop()
                    l3_recent = [
                        record for record in pipeline.memory.mem0.records
                        if record["memory"].startswith("[最近对话]")
                    ]
                    self.assertTrue(l3_recent)
                finally:
                    if agent_loop.running:
                        agent_loop.stop()

    async def test_attention_is_isolated_per_group_for_same_user(self):
        adapter, agent_loop, pipeline = await self._build_runtime()
        try:
            same_user = 424242
            events = [
                _make_group_event(adapter.config.bot_qq, 1001, same_user, "A群先@一下？", mentioned=True),
                _make_group_event(adapter.config.bot_qq, 2002, same_user, "B群未@直接提问？", mentioned=False),
                _make_group_event(adapter.config.bot_qq, 2002, same_user, "B群现在@一下？", mentioned=True),
                _make_group_event(adapter.config.bot_qq, 2002, same_user, "B群同上下文继续提问？", mentioned=False),
            ]

            for event in events:
                await adapter._on_message(event)

            await self._wait_processed(pipeline, len(events))

            sent_groups = [
                action["params"]["group_id"]
                for action in adapter.ws_connection.sent
                if action["action"] == "send_group_msg"
            ]
            self.assertEqual(Counter(sent_groups), Counter({1001: 1, 2002: 2}))

            self.assertEqual(
                Counter(
                    (
                        record["context_id"],
                        record["responded"],
                        record["is_mentioned"],
                    )
                    for record in pipeline.reply_records
                ),
                Counter(
                    {
                        ("group_1001", True, True): 1,
                        ("group_2002", False, False): 1,
                        ("group_2002", True, True): 1,
                        ("group_2002", True, False): 1,
                    }
                ),
            )
        finally:
            if agent_loop.running:
                agent_loop.stop()

    async def test_same_group_messages_preserve_arrival_order(self):
        adapter, agent_loop, pipeline = await self._build_runtime(max_concurrent_chats=12)
        try:
            group_id = 3003
            user_id = 515151
            events = [
                _make_group_event(
                    adapter.config.bot_qq,
                    group_id,
                    user_id,
                    f"顺序消息{i}？",
                    mentioned=(i == 0),
                )
                for i in range(8)
            ]

            for event in events:
                await adapter._on_message(event)

            await self._wait_processed(pipeline, len(events))

            turns = [
                turn for turn in pipeline.memory.working_memory
                if turn.context_id == f"group_{group_id}"
            ]
            self.assertEqual(
                [turn.user_input for turn in turns],
                [f"[user_{user_id}] 顺序消息{i}？" for i in range(8)],
            )
            self.assertEqual(
                [
                    action["params"]["message"]
                    for action in adapter.ws_connection.sent
                    if action["action"] == "send_group_msg"
                ],
                [f"ack:group_{group_id}:{user_id}:{i}" for i in range(1, 9)],
            )
        finally:
            if agent_loop.running:
                agent_loop.stop()


if __name__ == "__main__":
    unittest.main()
