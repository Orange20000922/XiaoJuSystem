import asyncio
import importlib
import json
import os
import random
import shutil
import sys
import time
import types
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_BENCH_ROOT = _PROJECT_ROOT / "data" / "benchmarks" / "qq_memory_integration"
_MEM0_DIR = _BENCH_ROOT / "mem0_state"

os.environ["MEM0_DIR"] = str(_MEM0_DIR)
sys.path.insert(0, str(_PROJECT_ROOT))

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
        mod.__path__ = [str(_PROJECT_ROOT / rel)]
        sys.modules[name] = mod


class _ChatMode(Enum):
    PRIVATE = "private"
    GROUP = "group"


_FAKE_INFERENCE = types.ModuleType("src.core.inference_pipeline")
_FAKE_INFERENCE.ChatMode = _ChatMode
_FAKE_INFERENCE.NeuroLikePipeline = object
sys.modules["src.core.inference_pipeline"] = _FAKE_INFERENCE

from configs.model_config import AgentConfig, AttentionConfig, ImageConfig, MemoryConfig, QQBotConfig

import src.attention.attention_tracker as attention_tracker_module
import src.memory.memory_manager as memory_manager_module
from src.adapters.qq_adapter import QQBotAdapter
from src.agent.agent_loop import AgentLoop
from src.logger import logger as app_logger

app_logger.remove()
app_logger.add(
    sys.stderr,
    level="WARNING",
    format="{time:HH:mm:ss} | {level} | {name} - {message}",
)


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


class _NullLLMClient:
    def generate(self, system_prompt, user_input, max_tokens, temperature):
        del user_input, max_tokens, temperature
        if "active_tasks" in system_prompt:
            return (
                '{"active_tasks":[],"established_facts":[],"artifacts":[],'
                '"open_questions":[],"dead_ends":[]}'
            )
        if "长期偏好" in system_prompt or "稳定特征" in system_prompt:
            return "[]"
        if "情绪状态" in system_prompt:
            return '{"emotion":"neutral","intensity":0.0}'
        return "[]"


class _Metrics:
    def __init__(self):
        self._lock = Lock()
        self._samples = defaultdict(list)

    def add(self, stage: str, duration: float):
        with self._lock:
            self._samples[stage].append(duration)

    def totals(self) -> Dict[str, float]:
        with self._lock:
            return {stage: sum(samples) for stage, samples in self._samples.items()}

    def summary(self) -> Dict[str, Dict[str, float]]:
        def _percentile(values: List[float], pct: float) -> float:
            if not values:
                return 0.0
            ordered = sorted(values)
            idx = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * pct))))
            return ordered[idx]

        with self._lock:
            out = {}
            for stage, samples in self._samples.items():
                total = sum(samples)
                out[stage] = {
                    "count": len(samples),
                    "total_s": round(total, 4),
                    "mean_ms": round((total / len(samples)) * 1000, 3) if samples else 0.0,
                    "p50_ms": round(_percentile(samples, 0.50) * 1000, 3),
                    "p95_ms": round(_percentile(samples, 0.95) * 1000, 3),
                    "max_ms": round(max(samples) * 1000, 3) if samples else 0.0,
                }
            return out


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


class _BasePipeline:
    def __init__(self):
        self.personality = types.SimpleNamespace(name="qq-memory-bench")
        self.attention_tracker = attention_tracker_module.AttentionTracker(
            AttentionConfig(
                cooldown_seconds=0,
                non_focus_reply_interval=0,
                mentioned_user_ttl=300,
                context_window_messages=50,
            )
        )
        self.metrics = _Metrics()
        self.processed_count = 0
        self.reply_records = []
        self.history_inconsistencies = []
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

    def _should_respond(
        self,
        *,
        user_input: str,
        is_mentioned: bool,
        chat_mode: _ChatMode,
        user_id: Optional[int],
        user_name: Optional[str],
        context_id: Optional[str],
    ) -> bool:
        in_attention_focus = False
        if chat_mode == _ChatMode.GROUP and not is_mentioned and user_id is not None:
            user = self.attention_tracker.get_user_attention(
                user_id=user_id,
                context_key=context_id,
            )
            in_attention_focus = (
                user is not None
                and self.attention_tracker.config.track_mentioned_users
                and user.is_mentioned_active(self.attention_tracker.config.mentioned_user_ttl)
            )

        behavior_type = "ask_question" if "?" in user_input else "respond_positive"
        intensity = 0.9 if is_mentioned else 0.3

        if chat_mode == _ChatMode.GROUP and user_id is not None:
            respond = self.attention_tracker.should_respond(
                user_id=user_id,
                emotion_intensity=intensity,
                behavior_type=behavior_type,
                is_mentioned=is_mentioned,
                in_attention_focus=in_attention_focus,
                context_key=context_id,
            )
            self.attention_tracker.on_message(
                user_id=user_id,
                user_name=user_name or str(user_id),
                is_mentioned=is_mentioned,
                context_key=context_id,
            )
            return respond
        return True

    def close(self):
        return None


class _NoMemoryPipeline(_BasePipeline):
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
        started = time.perf_counter()
        respond = self._should_respond(
            user_input=user_input,
            is_mentioned=is_mentioned,
            chat_mode=chat_mode,
            user_id=user_id,
            user_name=user_name,
            context_id=context_id,
        )

        response = ""
        if respond:
            response = f"ack:{context_id}:{user_id}:{self._next_seq()}"
            if chat_mode == _ChatMode.GROUP and user_id is not None:
                self.attention_tracker.on_reply(user_id, context_key=context_id)

        self.metrics.add("chat_total", time.perf_counter() - started)
        self._mark_processed(context_id, user_id, respond, is_mentioned)
        return {
            "should_respond": respond,
            "response": response,
            "emotion": {"primary": "neutral", "intensity": 0.3},
            "behavior": {"type": "respond_positive", "tone": "calm"},
        }


class _MemoryPipeline(_BasePipeline):
    def __init__(self, scenario_name: str, groups: int, seed_memories_per_group: int):
        super().__init__()
        self._vector_lock = Lock()
        bench_dir = _BENCH_ROOT / scenario_name
        shutil.rmtree(bench_dir, ignore_errors=True)
        bench_dir.mkdir(parents=True, exist_ok=True)

        memory_cfg = MemoryConfig(
            user_id=f"bench-user-{scenario_name}",
            vector_store_path=str(bench_dir / "qdrant"),
            collection_name=f"qq_bench_{uuid.uuid4().hex[:12]}",
            mem0_api_key="test-key",
            context_window_tokens=128_000,
            compression_threshold=0.98,
        )
        self.memory = memory_manager_module.HierarchicalMemoryManager(
            config=memory_cfg,
            llm_client=_NullLLMClient(),
            user_id=memory_cfg.user_id,
        )
        self._preseed(groups, seed_memories_per_group)

    def _preseed(self, groups: int, seed_memories_per_group: int):
        started = time.perf_counter()
        batch = []
        for group_offset in range(groups):
            group_id = 910000 + group_offset
            for seed_index in range(seed_memories_per_group):
                batch.append(
                    {
                        "role": "assistant",
                        "content": (
                            f"[seed] group={group_id} topic={seed_index % 9} "
                            f"user_pref=coffee_{seed_index % 7} "
                            f"artifact=file_{seed_index % 5}.py"
                        ),
                    }
                )
                if len(batch) >= 64:
                    with self._vector_lock:
                        self.memory.mem0.add(batch, user_id=self.memory.user_id, infer=False)
                    batch.clear()
        if batch:
            with self._vector_lock:
                self.memory.mem0.add(batch, user_id=self.memory.user_id, infer=False)
        self.metrics.add("preseed", time.perf_counter() - started)

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
        started = time.perf_counter()

        stage_started = time.perf_counter()
        with self._vector_lock:
            recalled = self.memory.get_system_context(user_input)
        self.metrics.add("recall_search", time.perf_counter() - stage_started)

        stage_started = time.perf_counter()
        history = self.memory.get_messages_history(context_id=context_id, max_turns=40)
        self.metrics.add("history_lookup", time.perf_counter() - stage_started)

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

        respond = self._should_respond(
            user_input=user_input,
            is_mentioned=is_mentioned,
            chat_mode=chat_mode,
            user_id=user_id,
            user_name=user_name,
            context_id=context_id,
        )

        response = ""
        if respond:
            response = (
                f"ack:{context_id}:{user_id}:{self._next_seq()}:"
                f"h={len(history)}:r={len(recalled)}"
            )

            stage_started = time.perf_counter()
            with self._vector_lock:
                self.memory.mem0.add(
                    [
                        {
                            "role": "assistant",
                            "content": (
                                f"[bench-memory] context={context_id} user={user_id} "
                                f"query={user_input} response={response}"
                            ),
                        }
                    ],
                    user_id=self.memory.user_id,
                    infer=False,
                )
            self.metrics.add("vector_write", time.perf_counter() - stage_started)

            stage_started = time.perf_counter()
            self.memory.add(
                _FakeTurn(
                    user_input=user_input,
                    emotion="neutral",
                    intensity=0.9 if is_mentioned else 0.3,
                    behavior="respond_positive",
                    tone="calm",
                    response=response,
                    timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    context_id=context_id,
                )
            )
            self.metrics.add("l1_add", time.perf_counter() - stage_started)

            if chat_mode == _ChatMode.GROUP and user_id is not None:
                self.attention_tracker.on_reply(user_id, context_key=context_id)

        self.metrics.add("chat_total", time.perf_counter() - started)
        self._mark_processed(context_id, user_id, respond, is_mentioned)
        return {
            "should_respond": respond,
            "response": response,
            "emotion": {"primary": "neutral", "intensity": 0.9 if is_mentioned else 0.3},
            "behavior": {"type": "respond_positive", "tone": "calm"},
        }

    def close(self):
        mem0 = getattr(self.memory, "mem0", None)
        if mem0:
            for attr in ("vector_store", "_telemetry_vector_store"):
                vs = getattr(mem0, attr, None)
                if vs and hasattr(vs, "client"):
                    try:
                        vs.client.close()
                    except Exception:
                        pass


async def _build_runtime(pipeline):
    agent_loop = AgentLoop(
        pipeline=pipeline,
        config=AgentConfig(
            proactive_level="off",
            tick_interval=0.01,
            max_concurrent_chats=12,
        ),
        output_callback=lambda text, ctx: None,
    )
    agent_loop._last_save_date = datetime.now().date()

    adapter = QQBotAdapter(
        QQBotConfig(
            bot_qq=999001,
            owner_qq=10000,
            reply_with_at=False,
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


async def _wait_processed(pipeline, expected: int, timeout: float = 180.0):
    started = asyncio.get_running_loop().time()
    while pipeline.processed_count < expected:
        if asyncio.get_running_loop().time() - started > timeout:
            return False
        await asyncio.sleep(0.02)
    await asyncio.sleep(0.1)
    return True


def _reset_shared_mem0():
    shared = getattr(memory_manager_module, "_shared_mem0", None)
    if shared:
        for attr in ("vector_store", "_telemetry_vector_store"):
            vs = getattr(shared, attr, None)
            if vs and hasattr(vs, "client"):
                try:
                    vs.client.close()
                except Exception:
                    pass
    memory_manager_module._shared_mem0 = None


async def _run_scenario(
    *,
    scenario_name: str,
    pipeline_type: str,
    groups: int,
    users_per_group: int,
    rounds: int,
    seed_memories_per_group: int,
):
    if pipeline_type == "memory":
        _reset_shared_mem0()
        importlib.reload(memory_manager_module)
        pipeline = _MemoryPipeline(
            scenario_name=scenario_name,
            groups=groups,
            seed_memories_per_group=seed_memories_per_group,
        )
    else:
        pipeline = _NoMemoryPipeline()

    adapter, agent_loop, pipeline = await _build_runtime(pipeline)
    events = []
    for round_index in range(rounds):
        for group_offset in range(groups):
            group_id = 700000 + group_offset
            for user_offset in range(users_per_group):
                user_id = group_id * 10 + user_offset
                events.append(
                    _make_group_event(
                        adapter.config.bot_qq,
                        group_id,
                        user_id,
                        f"group={group_id} user={user_id} round={round_index} topic={round_index % 7}?",
                        mentioned=True,
                    )
                )

    random.Random(42).shuffle(events)
    wall_started = time.perf_counter()
    try:
        for index in range(0, len(events), 100):
            batch = events[index:index + 100]
            await asyncio.gather(*(adapter._on_message(event) for event in batch))
        processed_ok = await _wait_processed(pipeline, len(events))
    finally:
        if agent_loop.running:
            agent_loop.stop()

    wall_elapsed = time.perf_counter() - wall_started
    sent_by_group = Counter(
        action["params"]["group_id"]
        for action in adapter.ws_connection.sent
        if action["action"] == "send_group_msg"
    )
    expected_per_group = users_per_group * rounds
    totals = pipeline.metrics.totals()
    memory_total = sum(
        totals.get(stage, 0.0)
        for stage in ("recall_search", "history_lookup", "vector_write", "l1_add")
    )
    chat_total = totals.get("chat_total", 0.0)

    return {
        "scenario": scenario_name,
        "pipeline": pipeline_type,
        "groups": groups,
        "users_per_group": users_per_group,
        "rounds": rounds,
        "messages": len(events),
        "processed_messages": pipeline.processed_count,
        "processed_ok": processed_ok,
        "wall_time_s": round(wall_elapsed, 3),
        "throughput_msg_s": round(len(events) / wall_elapsed, 3) if wall_elapsed else 0.0,
        "sent_messages": len(adapter.ws_connection.sent),
        "group_routing_ok": (
            len(sent_by_group) == groups and all(count == expected_per_group for count in sent_by_group.values())
        ),
        "history_inconsistencies": len(getattr(pipeline, "history_inconsistencies", [])),
        "metrics": pipeline.metrics.summary(),
        "memory_share_of_chat_time": round((memory_total / chat_total), 4) if chat_total else 0.0,
    }


async def _run_isolation_smoke():
    _reset_shared_mem0()
    importlib.reload(memory_manager_module)
    pipeline = _MemoryPipeline(
        scenario_name="isolation_smoke",
        groups=2,
        seed_memories_per_group=8,
    )
    adapter, agent_loop, pipeline = await _build_runtime(pipeline)
    try:
        same_user = 424242
        events = [
            _make_group_event(adapter.config.bot_qq, 1001, same_user, "group A first mention?", mentioned=True),
            _make_group_event(adapter.config.bot_qq, 2002, same_user, "group B without mention?", mentioned=False),
            _make_group_event(adapter.config.bot_qq, 2002, same_user, "group B mention now?", mentioned=True),
            _make_group_event(adapter.config.bot_qq, 2002, same_user, "group B follow up?", mentioned=False),
        ]
        for event in events:
            await adapter._on_message(event)
        await _wait_processed(pipeline, len(events))
        sent_groups = Counter(
            action["params"]["group_id"]
            for action in adapter.ws_connection.sent
            if action["action"] == "send_group_msg"
        )
        response_mix = Counter(
            (record["context_id"], record["responded"], record["is_mentioned"])
            for record in pipeline.reply_records
        )
        return {
            "sent_groups": dict(sent_groups),
            "response_mix": {
                f"{k[0]}|responded={k[1]}|mentioned={k[2]}": v
                for k, v in response_mix.items()
            },
        }
    finally:
        if agent_loop.running:
            agent_loop.stop()


async def _main():
    _BENCH_ROOT.mkdir(parents=True, exist_ok=True)

    isolation = await _run_isolation_smoke()
    scenarios = []
    for groups in (10, 20):
        scenarios.append(
            await _run_scenario(
                scenario_name=f"baseline_g{groups}",
                pipeline_type="baseline",
                groups=groups,
                users_per_group=5,
                rounds=10,
                seed_memories_per_group=0,
            )
        )
        scenarios.append(
            await _run_scenario(
                scenario_name=f"memory_g{groups}",
                pipeline_type="memory",
                groups=groups,
                users_per_group=5,
                rounds=10,
                seed_memories_per_group=50,
            )
        )

    pairs = {}
    for groups in (10, 20):
        baseline = next(item for item in scenarios if item["scenario"] == f"baseline_g{groups}")
        memory = next(item for item in scenarios if item["scenario"] == f"memory_g{groups}")
        pairs[f"groups_{groups}"] = {
            "baseline_wall_s": baseline["wall_time_s"],
            "memory_wall_s": memory["wall_time_s"],
            "slowdown_x": round(memory["wall_time_s"] / baseline["wall_time_s"], 3)
            if baseline["wall_time_s"]
            else 0.0,
            "baseline_throughput_msg_s": baseline["throughput_msg_s"],
            "memory_throughput_msg_s": memory["throughput_msg_s"],
            "memory_share_of_chat_time": memory["memory_share_of_chat_time"],
        }

    print(
        json.dumps(
            {
                "isolation_smoke": isolation,
                "scenarios": scenarios,
                "comparisons": pairs,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    asyncio.run(_main())
