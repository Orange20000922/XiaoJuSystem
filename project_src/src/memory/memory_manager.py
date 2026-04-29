"""
分级记忆管理器

压缩策略：
  - 保护区（最近 PROTECTED_TURNS 轮）永不压缩
  - 压缩目标是结构化状态提取，而非叙述性摘要
  - 双条件触发：token 压力 OR BERT 检测到话题边界
"""

import json
import os
import time
import threading
from typing import List, Dict, Optional, TYPE_CHECKING

import tiktoken
from mem0 import Memory

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.model_config import MemoryConfig
from src.logger import logger
from src.memory.forgetting import (
    ForgettingEngine,
    MemoryLifecycleStore,
    default_lifecycle_path,
    stable_content_hash,
    stable_memory_hash,
    timestamp_to_iso,
    timestamp_value,
)

if TYPE_CHECKING:
    from src.core_engine.runtime_types import ConversationTurn
    from src.llm.client import LLMClient


# tiktoken 编码器缓存（gpt-4o 编码兼容 GPT-5.2）
_ENCODER = None

# ── Mem0 全进程共享单例 ──────────────────────────────────────────────────────
# Memory 实例（含 QdrantClient + SentenceTransformer）全进程只创建一次。
# 各人格通过不同的 user_id 隔离自己的记忆，共用同一个 Qdrant collection。
_shared_mem0: Optional[Memory] = None
_shared_mem0_lock = threading.Lock()

def _get_encoder():
    global _ENCODER
    if _ENCODER is None:
        try:
            _ENCODER = tiktoken.encoding_for_model("gpt-4o")
        except Exception:
            _ENCODER = tiktoken.get_encoding("cl100k_base")
    return _ENCODER


def count_tokens(text: str) -> int:
    return len(_get_encoder().encode(text))


def turn_to_text(turn: "ConversationTurn") -> str:
    return f"用户: {turn.user_input}\n助手: {turn.response}\n"


# 结构化状态提取 prompt（不是摘要，是状态机快照）
_STATE_EXTRACTION_PROMPT = """从以下对话中提取结构化状态，输出 JSON，不要写叙述性文字。

字段说明：
- active_tasks: 未完成的任务或话题（字符串列表）
- established_facts: 已确认的关键事实、决策、用户偏好（字符串列表）
- artifacts: 产生的代码片段、文件路径、具体结论（字符串列表）
- open_questions: 未解决的问题或悬而未决的分歧（字符串列表）
- dead_ends: 已排除的方向，避免重复尝试（字符串列表）

只输出 JSON，不要任何解释。若某字段无内容则输出空列表。"""

# L3 事实抽取 prompt
_FACT_EXTRACTION_PROMPT = """从以下会话状态记录中抽取用户的长期偏好、重要特征、关键决策。
只输出 JSON 数组，每条字符串不超过 50 字，不要任何解释。
格式：["事实1", "事实2", ...]"""


# L4 用户画像构建 prompt
_PROFILE_COMPACT_PROMPT = """从以下对话片段中提取用户的稳定特征，输出 JSON 数组。
每条标签格式：{"tag": "标签名", "value": "具体内容", "confidence": 0.0-1.0}
标签类别：interests（兴趣爱好）、personality（性格特点）、preferences（偏好习惯）、
          background（背景信息）、relationship（与AI的关系模式）
只输出 JSON 数组，不要任何解释。若无有效信息则输出空数组 []。"""

# L4 情绪推断 prompt（策略 B 回退：从纯文本推断情绪）
_EMOTION_INFER_PROMPT = """分析以下对话片段，判断用户的情绪状态。
输出 JSON：{"emotion": "情绪标签", "intensity": 0.0-1.0}
情绪标签从以下选择：joy, sadness, anger, fear, surprise, disgust, neutral, excitement, tenderness, curiosity
只输出 JSON，不要任何解释。"""


class HierarchicalMemoryManager:
    """
    分级记忆管理器

    L1  工作记忆  — 原始对话轮次，内存存储，直接注入 prompt
                    最近 PROTECTED_TURNS 轮永不压缩
    L2  状态快照  — L1 压缩时结构化状态写入 Mem0 session memory
    L3  情节记忆  — 会话结束时抽取关键事实写入 Mem0 user memory（持久化）
    L4  知识库    — Agent 执行结果等显式写入 Mem0 agent memory
    """

    # 保护区大小：最近这些轮次永远保留在 L1 原文
    PROTECTED_TURNS = 12

    # 话题边界触发的最低 token 占比（避免刚开始就触发）
    BOUNDARY_MIN_THRESHOLD = 0.4

    def __init__(
        self,
        config: MemoryConfig,
        llm_client: "LLMClient",
        maintenance_llm_client: Optional["LLMClient"] = None,
        user_id: str = None,
    ):
        self.config = config
        self.llm_client = llm_client
        self.maintenance_llm_client = maintenance_llm_client or llm_client
        self.user_id = user_id or config.user_id or "owner"
        self.session_id = self._new_session_id()

        # L1
        self.working_memory: List["ConversationTurn"] = []
        self._l1_tokens: int = 0
        self._memory_lock = threading.RLock()
        self._turn_seq_counter: int = 0
        self._last_flushed_turn_seq: int = 0
        self._compressed_run_ids: List[str] = []

        # 最近一次 BERT 分类的行为标签，由外部（Pipeline）在 add() 前设置
        self.last_behavior: str = ""

        # Mem0 后端 ── 全进程共享单例，各人格通过 user_id 隔离记忆
        global _shared_mem0
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        if config.mem0_api_key:
            os.environ["OPENAI_API_KEY"] = config.mem0_api_key
        if config.mem0_base_url and config.mem0_llm_provider == "openai":
            os.environ["OPENAI_BASE_URL"] = config.mem0_base_url

        with _shared_mem0_lock:
            if _shared_mem0 is None:
                llm_cfg: dict = {
                    "model": config.mem0_llm_model,
                    "temperature": config.mem0_llm_temperature,
                    "api_key": config.mem0_api_key,
                }
                if config.mem0_llm_provider == "openai" and config.mem0_base_url:
                    llm_cfg["openai_base_url"] = config.mem0_base_url

                mem0_cfg: Dict = {
                    "vector_store": {
                        "provider": "qdrant",
                        "config": {
                            "collection_name": config.collection_name,
                            "path": config.vector_store_path,
                            "embedding_model_dims": 384,
                            "on_disk": True,
                        },
                    },
                    "llm": {
                        "provider": config.mem0_llm_provider,
                        "config": llm_cfg,
                    },
                    "embedder": {
                        "provider": "huggingface",
                        "config": {
                            "model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                        },
                    },
                }
                _shared_mem0 = Memory.from_config(mem0_cfg)
                logger.info(
                    f"Mem0 共享实例已创建 "
                    f"(collection={config.collection_name}, user_id={self.user_id})"
                )

                # Anthropic 专属修复
                if (config.mem0_llm_provider == "anthropic"
                        and hasattr(_shared_mem0, "llm")
                        and hasattr(_shared_mem0.llm, "client")):
                    import anthropic, httpx

                    def _strip_bearer(request: httpx.Request):
                        if "authorization" in request.headers:
                            del request.headers["authorization"]

                    _shared_mem0.llm.client = anthropic.Anthropic(
                        api_key=config.mem0_api_key,
                        base_url=config.mem0_base_url,
                        http_client=httpx.Client(
                            event_hooks={"request": [_strip_bearer]}
                        ),
                    )
            else:
                logger.info(f"Mem0 共享实例复用 (user_id={self.user_id})")

        self.mem0 = _shared_mem0
        lifecycle_path = (
            Path(config.lifecycle_store_path)
            if config.lifecycle_store_path
            else default_lifecycle_path(
                vector_store_path=config.vector_store_path,
                collection_name=config.collection_name,
                user_id=self.user_id,
            )
        )
        self.lifecycle = MemoryLifecycleStore(lifecycle_path)
        self.forgetting_engine = ForgettingEngine(config)

    def _new_session_id(self) -> str:
        return f"session_{time.time_ns()}"

    def _assign_turn_seq(self, turn: "ConversationTurn") -> int:
        self._turn_seq_counter += 1
        setattr(turn, "_memory_seq", self._turn_seq_counter)
        return self._turn_seq_counter

    def _turn_seq(self, turn: "ConversationTurn") -> int:
        return int(getattr(turn, "_memory_seq", 0))

    def _maintenance_generate(
        self,
        *,
        system_prompt: str,
        user_input: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        return self.maintenance_llm_client.generate(
            system_prompt=system_prompt,
            user_input=user_input,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def _turns_metadata(self, turns: List["ConversationTurn"]) -> Dict:
        if not turns:
            return {}
        strongest = max(
            turns,
            key=lambda turn: float(getattr(turn, "intensity", 0.0) or 0.0),
        )
        return {
            "emotion": getattr(strongest, "emotion", "neutral") or "neutral",
            "emotion_intensity": float(getattr(strongest, "intensity", 0.0) or 0.0),
            "behavior": getattr(strongest, "behavior", "") or "",
            "tone": getattr(strongest, "tone", "") or "",
            "context_id": getattr(strongest, "context_id", None),
        }

    def _memory_metadata(
        self,
        *,
        content: str,
        memory_level: str,
        memory_type: str,
        run_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        extra_metadata: Optional[Dict] = None,
    ) -> Dict:
        now = time.time()
        metadata = {
            "memory_level": memory_level,
            "memory_type": memory_type,
            "created_at": timestamp_to_iso(now),
            "lifecycle_created_at": now,
            "last_recalled_at": None,
            "recall_count": 0,
            "forgotten": False,
            "forgotten_at": None,
            "deleted_after": None,
            "memory_hash": stable_memory_hash(
                user_id=self.user_id,
                content=content,
                memory_level=memory_level,
                memory_type=memory_type,
                run_id=run_id,
                agent_id=agent_id,
            ),
            "content_hash": stable_content_hash(self.user_id, content),
            "user_id": self.user_id,
            "run_id": run_id,
            "agent_id": agent_id,
        }
        if extra_metadata:
            metadata.update({k: v for k, v in extra_metadata.items() if v is not None})
        return metadata

    def _mem0_add_content(
        self,
        content: str,
        *,
        user_id: Optional[str] = None,
        run_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        memory_level: str,
        memory_type: str,
        extra_metadata: Optional[Dict] = None,
    ):
        metadata = self._memory_metadata(
            content=content,
            memory_level=memory_level,
            memory_type=memory_type,
            run_id=run_id,
            agent_id=agent_id,
            extra_metadata=extra_metadata,
        )
        self.lifecycle.register(metadata)
        kwargs = {
            "user_id": user_id,
            "run_id": run_id,
            "agent_id": agent_id,
            "metadata": metadata,
        }
        kwargs = {key: value for key, value in kwargs.items() if value is not None}
        try:
            result = self.mem0.add(
                [{"role": "assistant", "content": content}],
                infer=False,
                **kwargs,
            )
        except TypeError:
            try:
                # 旧版mem0兼容回退
                result = self.mem0.add(
                    [{"role": "assistant", "content": content}],
                    **kwargs,
                )
            except TypeError:
                kwargs.pop("metadata", None)
                result = self.mem0.add(
                    [{"role": "assistant", "content": content}],
                    **kwargs,
                )
        self._record_mem0_ids(result, metadata)
        return result

    def _record_mem0_ids(self, add_result, metadata: Dict):
        results = add_result.get("results", []) if isinstance(add_result, dict) else []
        if not results:
            return
        # 记录第一个记忆 ID 到生命周期存储，供遗忘机制原型使用；同时尝试同步生命周期元数据到 Mem0 以便后续查询。
        first = results[0]
        mem0_id = first.get("id") or first.get("memory_id")
        if mem0_id:
            metadata["mem0_id"] = str(mem0_id)
            self.lifecycle.register(metadata)
            record = self.lifecycle.get(str(metadata.get("memory_hash")))
            if record:
                self._sync_lifecycle_record_to_mem0(record)

    def _sync_lifecycle_record_to_mem0(self, record: Dict) -> bool:
        mem0_id = record.get("mem0_id") or record.get("id")
        if not mem0_id:
            return False
        try:
            stored = self.mem0.vector_store.get(vector_id=str(mem0_id))
            if not stored or not getattr(stored, "payload", None):
                return False
            payload = dict(stored.payload)
            data = payload.get("data") or record.get("memory") or ""
            if not data:
                return False
            for key, value in record.items():
                if key in {"memory", "score"}:
                    continue
                if key in {"created_at", "updated_at"}:
                    ts = timestamp_value(value)
                    value = timestamp_to_iso(ts) if ts is not None else value
                payload[key] = value
            if hasattr(self.mem0.vector_store, "update"):
                self.mem0.vector_store.update(vector_id=str(mem0_id), payload=payload)
                return True
            self.mem0.update(str(mem0_id), data, metadata=payload)
            return True
        except Exception as exc:
            logger.warning(f"同步 lifecycle metadata 到 Mem0 失败: {exc}")
            return False

    # ── L1 操作 ──────────────────────────────────────────────────────────

    def add(self, turn: "ConversationTurn"):
        """
        添加对话轮次，双条件触发压缩：
          1. token 达到阈值（硬性触发）
          2. BERT 检测到话题边界且 token 超过最低占比（软性触发）
        """
        with self._memory_lock:
            self._assign_turn_seq(turn)
            self.working_memory.append(turn)
            self._l1_tokens += count_tokens(turn_to_text(turn))

            token_pressure = self._l1_tokens >= self.config.compression_trigger_tokens

            boundary_trigger = (
                turn.behavior == "change_topic"
                and self._l1_tokens >= int(
                    self.config.context_window_tokens * self.BOUNDARY_MIN_THRESHOLD
                )
                # 保护区内的话题切换不触发（轮次太少压缩没意义）
                and len(self.working_memory) > self.PROTECTED_TURNS + 2
            )

            if token_pressure or boundary_trigger:
                self._compress_l1_to_l2()

    def _compress_l1_to_l2(self):
        """
        压缩 L1 中保护区以外的最旧部分：
          - 保护区（最近 PROTECTED_TURNS 轮）原文保留，永不触碰
          - 可压缩区做结构化状态提取，而非叙述性摘要
        """
        with self._memory_lock:
            n = len(self.working_memory)
            compressible_end = max(0, n - self.PROTECTED_TURNS)

            if compressible_end == 0:
                # 全部在保护区内，无可压缩内容
                return

            # 取可压缩区中最旧的 compression_ratio 部分
            cut = max(1, int(compressible_end * self.config.compression_ratio))
            to_compress = self.working_memory[:cut]

            history_text = "".join(turn_to_text(t) for t in to_compress)
            run_id = self.session_id

            raw = self.llm_client.generate(
                system_prompt=_STATE_EXTRACTION_PROMPT,
                user_input=history_text,
                max_tokens=500,
                temperature=0.1,
            )

            # 尝试解析为 JSON，失败时原文存储（保底）
            try:
                state = json.loads(raw)
                content = f"[状态快照] {json.dumps(state, ensure_ascii=False)}"
            except json.JSONDecodeError:
                content = f"[状态快照] {raw}"

            self._mem0_add_content(
                content,
                user_id=self.user_id,
                run_id=run_id,
                memory_level="L2",
                memory_type="state_snapshot",
                extra_metadata=self._turns_metadata(to_compress),
            )

            if not self._compressed_run_ids or self._compressed_run_ids[-1] != run_id:
                self._compressed_run_ids.append(run_id)

            removed_tokens = sum(count_tokens(turn_to_text(t)) for t in to_compress)
            self.working_memory = self.working_memory[cut:]
            self._l1_tokens = max(0, self._l1_tokens - removed_tokens)

            # 压缩后切换 run_id，避免后续 L2 继续写到已封存的 session。
            old_session = self.session_id
            self.session_id = self._new_session_id()
            from src.logger import logger
            logger.info(
                f"L1 压缩完成：移除 {len(to_compress)} 轮 ({removed_tokens} tokens)，"
                f"保留 {len(self.working_memory)} 轮 ({self._l1_tokens} tokens)，"
                f"session 重置 {old_session[-8:]} → {self.session_id[-8:]}"
            )

    # ── L2 → L3 会话结束 ─────────────────────────────────────────────────

    def close_session(self):
        """
        会话结束时调用：
          1. 压缩保护区以外的剩余 L1
          2. 从本次会话的状态快照中抽取长期事实写入 L3
        """
        from src.logger import logger

        with self._memory_lock:
            # 仅把上次 flush 之后新增的保护区轮次写入 L3，避免定时 flush + stop() 重复写。
            protected = [
                t for t in self.working_memory[-self.PROTECTED_TURNS:]
                if self._turn_seq(t) > self._last_flushed_turn_seq
            ]
            pending_recent = bool(protected)
            logger.info(
                f"close_session: protected turns={len(self.working_memory[-self.PROTECTED_TURNS:])}, "
                f"pending_recent={len(protected)}"
            )
            if protected:
                lines = []
                for t in protected:
                    tag = (
                        f"[emotion={t.emotion},intensity={t.intensity:.2f}]"
                        if t.emotion else ""
                    )
                    lines.append(f"{tag} {turn_to_text(t)}")
                recent = "\n".join(lines)
                logger.info(
                    f"close_session: writing {len(recent)} chars to L3 with emotion tags"
                )
                self._mem0_add_content(
                    f"[最近对话] {recent}",
                    user_id=self.user_id,
                    memory_level="L3",
                    memory_type="recent_dialog",
                    extra_metadata=self._turns_metadata(protected),
                )
                self._last_flushed_turn_seq = max(
                    self._last_flushed_turn_seq,
                    max(self._turn_seq(t) for t in protected),
                )
                logger.info("close_session: L3 write done")

            # 压缩剩余可压缩区，把当前逻辑 session 内所有 L2 状态都归档到 L3。
            n = len(self.working_memory)
            compressible_end = max(0, n - self.PROTECTED_TURNS)
            if compressible_end > 0:
                self._compress_l1_to_l2()

            run_ids = tuple(self._compressed_run_ids)
            pending_states = bool(run_ids)
            if not pending_recent and not pending_states:
                logger.info("close_session: no new L1/L2 content pending, skip")
                return

            all_state_texts = []
            for run_id in run_ids:
                session_mems = self.mem0.get_all(
                    filters={
                        "user_id": self.user_id,
                        "run_id": run_id,
                        "memory_level": "L2",
                        "forgotten": False,
                    },
                )
                session_items = self._filter(
                    session_mems,
                    memory_level="L2",
                    include_forgotten=False,
                )
                if not session_items:
                    continue
                all_state_texts.extend(
                    r["memory"] for r in session_items if "memory" in r
                )

            if all_state_texts:
                all_states = "\n".join(all_state_texts)

                raw_facts = self._maintenance_generate(
                    system_prompt=_FACT_EXTRACTION_PROMPT,
                    user_input=all_states,
                    max_tokens=400,
                    temperature=0.1,
                )

                try:
                    facts = json.loads(raw_facts)
                    content = "\n".join(f"- {f}" for f in facts if isinstance(f, str))
                except json.JSONDecodeError:
                    content = raw_facts

                if content.strip():
                    self._mem0_add_content(
                        content,
                        user_id=self.user_id,  # 无 session_id → L3 持久化
                        memory_level="L3",
                        memory_type="fact",
                    )

            # L4 用户画像构建（基于本次 session 写入的 L3 强情绪片段）
            try:
                logger.info("close_session: triggering L4 profile build...")
                self.build_user_profile_l4()
                logger.info("close_session: L4 profile build done")
            except Exception as e:
                logger.warning(f"L4 画像构建失败（不影响主流程）: {e}")

            self._compressed_run_ids.clear()
            old_session = self.session_id
            self.session_id = self._new_session_id()
            logger.info(
                f"close_session: session rolled over {old_session[-8:]} → {self.session_id[-8:]}"
            )

    # ── L4 显式写入 ───────────────────────────────────────────────────────

    def add_knowledge(self, content: str, memory_type: str = "knowledge"):
        """显式写入 L4 知识库（Agent 执行结果、分析结论等）"""
        if content.startswith("[能力]"):
            memory_type = "capability"
        elif content.startswith("[用户画像]"):
            memory_type = "profile"
        elif content.startswith("[尾端情绪状态]"):
            memory_type = "emotion_snapshot"
        self._mem0_add_content(
            content,
            agent_id="neuro_agent",
            user_id=self.user_id,
            memory_level="L4",
            memory_type=memory_type,
        )

    def add_visual_observation(self, content: str, event_id: Optional[str] = None):
        """将筛选后的高价值视觉观察写入 L4。"""
        prefix = "[视觉观察]"
        if event_id:
            prefix = f"{prefix}[{event_id}]"
        self.add_knowledge(f"{prefix} {content}", memory_type="visual")

    def add_visual_observation_l3(self, content: str, event_id: Optional[str] = None):
        """将视觉观察写入 L3 用户级记忆（非即时事件）。"""
        prefix = "[视觉观察]"
        if event_id:
            prefix = f"{prefix}[{event_id}]"
        self._mem0_add_content(
            f"{prefix} {content}",
            user_id=self.user_id,
            memory_level="L3",
            memory_type="visual",
        )

    def add_visual_session_summary_l3(self, content: str):
        """将实时视觉会话的全局总结写入 L3。"""
        self._mem0_add_content(
            f"[视觉总结] {content}",
            user_id=self.user_id,
            memory_level="L3",
            memory_type="visual_summary",
        )

    def add_emotion_snapshot_l4(self, snapshot, source: str = "runtime"):
        """将尾端情绪状态快照写入 L4。"""
        content = snapshot if isinstance(snapshot, str) else json.dumps(
            snapshot,
            ensure_ascii=False,
        )
        self.add_knowledge(f"[尾端情绪状态][{source}] {content}")

    def build_user_profile_l4(
        self,
        intensity_threshold: float = 0.6,
        max_entries: int = 20,
    ):
        """
        扫描 L3，提取强情绪片段，用 LLM compact 生成用户画像标签写入 L4。

        策略 A：优先使用 L3 中的 [emotion=xxx,intensity=x.xx] 标签过滤。
        策略 B：若某条 L3 记录无情绪标签（旧数据），用 LLM 从文本推断情绪作为回退。

        Args:
            intensity_threshold: 情绪强度过滤阈值（默认 0.6）
            max_entries: 最多处理的 L3 条目数（避免 token 爆炸）
        """
        import re

        # 拉取所有 L3 记录
        all_l3 = self.mem0.get_all(
            filters={
                "user_id": self.user_id,
                "memory_level": "L3",
                "forgotten": False,
            }
        )
        results = self._filter(all_l3, memory_level="L3", include_forgotten=False)
        if not results:
            from src.logger import logger
            logger.info("L4 画像构建：L3 无记录，跳过")
            return

        results = results[:max_entries]

        # ── 过滤强情绪片段 ────────────────────────────────────────────────
        strong_entries = []
        _emotion_tag_re = re.compile(
            r"\[emotion=(\w+),intensity=([0-9.]+)\]"
        )

        for r in results:
            text = r.get("memory", "")
            if not text:
                continue

            # 策略 A：解析情绪标签
            matches = _emotion_tag_re.findall(text)
            if matches:
                # 取最高强度
                max_intensity = max(float(intensity) for _, intensity in matches)
                if max_intensity >= intensity_threshold:
                    strong_entries.append(text)
                continue

            # 策略 B：无标签，用 LLM 推断情绪
            try:
                raw = self._maintenance_generate(
                    system_prompt=_EMOTION_INFER_PROMPT,
                    user_input=text[:500],  # 截断避免 token 过多
                    max_tokens=50,
                    temperature=0.1,
                )
                inferred = json.loads(raw)
                if float(inferred.get("intensity", 0)) >= intensity_threshold:
                    strong_entries.append(text)
            except Exception:
                pass  # 推断失败则跳过该条

        if not strong_entries:
            from src.logger import logger
            logger.info(f"L4 画像构建：无强情绪片段（阈值={intensity_threshold}），跳过")
            return

        # ── 统计词频预处理（过滤低信息量片段）────────────────────────────
        from collections import Counter
        import jieba

        word_freq: Counter = Counter()
        for entry in strong_entries:
            # 去掉标签头，只统计对话正文
            clean = _emotion_tag_re.sub("", entry)
            clean = re.sub(r"\[最近对话\]|\[状态快照\]", "", clean)
            words = [w for w in jieba.cut(clean) if len(w) > 1]
            word_freq.update(words)

        # 高频词（出现 3 次以上）作为关键词提示注入 prompt
        keywords = [w for w, c in word_freq.most_common(20) if c >= 3]

        # ── LLM compact → 结构化画像标签 ─────────────────────────────────
        combined_text = "\n---\n".join(strong_entries[:10])  # 最多 10 条
        keyword_hint = f"\n\n关键词参考（高频出现）：{', '.join(keywords)}" if keywords else ""

        raw_profile = self._maintenance_generate(
            system_prompt=_PROFILE_COMPACT_PROMPT,
            user_input=combined_text + keyword_hint,
            max_tokens=600,
            temperature=0.1,
        )

        try:
            tags = json.loads(raw_profile)
            if not isinstance(tags, list) or not tags:
                return
            # 过滤低置信度标签
            tags = [t for t in tags if isinstance(t, dict) and t.get("confidence", 0) >= 0.5]
        except json.JSONDecodeError:
            tags = []

        if not tags:
            from src.logger import logger
            logger.info("L4 画像构建：LLM 未返回有效标签")
            return

        # ── 写入 L4 ───────────────────────────────────────────────────────
        profile_content = f"[用户画像] {json.dumps(tags, ensure_ascii=False)}"
        self.add_knowledge(profile_content)

        from src.logger import logger
        logger.info(f"L4 画像构建完成：{len(tags)} 条标签写入 L4")


    # ── 召回与 prompt 组装 ────────────────────────────────────────────────

    def get_system_context(self, query: str) -> str:
        """
        返回注入 system prompt 的跨会话上下文（L2/L3/L4 召回）。
        L1 原文不在这里，走 get_messages_history()。
        L2/L3/L4 三次 Qdrant search 并行执行。
        """
        from concurrent.futures import ThreadPoolExecutor

        with self._memory_lock:
            l2_run_ids = tuple(self._compressed_run_ids) or (self.session_id,)

        sections = []
        overfetch = max(1, int(getattr(self.config, "memory_recall_overfetch", 1)))
        l3_top_k = self.config.l3_search_limit * overfetch
        l4_top_k = self.config.l4_search_limit * overfetch
        historical_query = self._should_attempt_recovery(query)

        with ThreadPoolExecutor(max_workers=2 + len(l2_run_ids)) as executor:
            l2_futures = [
                executor.submit(
                    self.mem0.search,
                    query=query,
                    filters={
                        "user_id": self.user_id,
                        "run_id": run_id,
                        "memory_level": "L2",
                        "forgotten": False,
                    },
                    top_k=l3_top_k,
                )
                for run_id in l2_run_ids
            ]
            f_l3 = executor.submit(
                self.mem0.search,
                query=query,
                filters={
                    "user_id": self.user_id,
                    "memory_level": "L3",
                    "forgotten": False,
                },
                top_k=l3_top_k,
            )
            f_l4 = executor.submit(
                self.mem0.search,
                query=query,
                filters={
                    "user_id": self.user_id,
                    "agent_id": "neuro_agent",
                    "memory_level": "L4",
                },
                top_k=l4_top_k,
            )
            l2_results = [future.result() for future in l2_futures]
            l3 = f_l3.result()
            l4 = f_l4.result()

        l2_items: List[Dict] = []
        for result in l2_results:
            l2_items.extend(
                self._filter(
                    result,
                    memory_level="L2",
                    include_forgotten=False,
                    historical_query=historical_query,
                )
            )
        l3_items = self._filter(
            l3,
            memory_level="L3",
            include_forgotten=False,
            historical_query=historical_query,
        )
        combined_items = self._merge_items(l2_items, l3_items)[
            :self.config.l3_search_limit
        ]
        if self.config.recovery_enabled and (not combined_items or historical_query):
            recovered = self.recovery_search(
                query,
                top_k=max(1, self.config.l3_search_limit - len(combined_items)),
            )
            combined_items = self._merge_items(combined_items, recovered)[
                :self.config.l3_search_limit
            ]
        if combined_items:
            self._mark_recalled(combined_items)
            sections.append(
                "<history>" +
                "|".join(item["memory"] for item in combined_items) +
                "</history>"
            )
        elif self._has_l3_records():
            # 语义搜索没命中具体内容，但 L3 确实有历史记录
            sections.append(
                "<has_memory>你们之前聊过天，你记得这个人，但一时想不起具体聊了什么。"
                "不要说'这是第一次对话'，也不要提到记忆系统、数据、调取之类的词。</has_memory>"
            )

        l4_items = self._filter(l4, memory_level="L4", include_forgotten=True)[
            :self.config.l4_search_limit
        ]
        if l4_items:
            sections.append(
                "<knowledge>" +
                "|".join(m["memory"] for m in l4_items) +
                "</knowledge>"
            )

        return "\n".join(sections) if sections else ""

    def _should_attempt_recovery(self, query: str) -> bool:
        if not query:
            return False
        recovery_terms = (
            "还记得",
            "记得",
            "之前",
            "以前",
            "上次",
            "刚才",
            "我们聊过",
            "说过",
        )
        return any(term in query for term in recovery_terms)

    def recovery_search(
        self,
        query: str,
        *,
        top_k: Optional[int] = None,
        now: Optional[float] = None,
    ) -> List[Dict]:
        """搜索已遗忘的 L2/L3 记忆，并重新激活高相关命中。"""
        if not self.config.recovery_enabled:
            return []
        now = now or time.time()
        limit = top_k or self.config.l3_search_limit
        overfetch = max(1, int(getattr(self.config, "memory_recall_overfetch", 1)))
        search_top_k = limit * overfetch
        result_sets = []
        for level in ("L2", "L3"):
            try:
                result_sets.append(
                    self.mem0.search(
                        query=query,
                        filters={
                            "user_id": self.user_id,
                            "memory_level": level,
                            "forgotten": True,
                        },
                        top_k=search_top_k,
                    )
                )
            except Exception as exc:
                logger.warning(f"恢复性召回搜索失败 level={level}: {exc}")

        candidates: List[Dict] = []
        for result in result_sets:
            candidates.extend(
                self._filter(
                    result,
                    include_forgotten=True,
                    recovery=True,
                    historical_query=self._should_attempt_recovery(query),
                )
            )

        recoverable = []
        for item in self._merge_items(candidates):
            lifecycle = item.get("_lifecycle") or self._item_lifecycle(item)
            if not lifecycle or not lifecycle.get("forgotten", False):
                continue
            deleted_after = lifecycle.get("deleted_after")
            if deleted_after is not None and float(deleted_after) <= now:
                continue
            level = str(lifecycle.get("memory_level") or "").upper()
            if level not in {"L2", "L3"}:
                continue
            recoverable.append(item)
            if len(recoverable) >= limit:
                break

        hashes = []
        for item in recoverable:
            lifecycle = item.get("_lifecycle") or self._item_lifecycle(item)
            if lifecycle and lifecycle.get("memory_hash"):
                hashes.append(str(lifecycle["memory_hash"]))

        for record in self.lifecycle.reactivate(hashes, now=now):
            self._sync_lifecycle_record_to_mem0(record)
        for item in recoverable:
            lifecycle = self._item_lifecycle(item)
            if lifecycle:
                item["_lifecycle"] = lifecycle
            item["_skip_recall_mark"] = True
        return recoverable

    def get_messages_history(self, context_id: Optional[str] = None, max_turns: Optional[int] = None) -> List[Dict]:
        """
        返回 L1 working_memory 的 OpenAI messages 格式列表。
        直接传入 API 的 messages 数组，让 API 在真实窗口内维护上下文。

        Args:
            context_id: 对话上下文标识（群号/私聊用户ID），None 表示返回所有记忆
            max_turns: 最多返回最近 N 轮对话（None 表示返回所有），用于限制 messages 数组长度
        """
        with self._memory_lock:
            turns = list(self.working_memory)

        messages = []
        for turn in turns:
            # 如果指定了 context_id，只返回匹配的记忆
            if context_id is not None and turn.context_id != context_id:
                continue

            if turn.user_input:
                messages.append({"role": "user", "content": turn.user_input})
            if turn.response:
                messages.append({"role": "assistant", "content": turn.response})

        # 限制 messages 数量（保留最近的 N 轮，每轮 2 条 message）
        if max_turns is not None and max_turns > 0:
            max_messages = max_turns * 2
            if len(messages) > max_messages:
                from src.logger import logger
                logger.debug(
                    f"messages 数组过长 ({len(messages)} 条)，截断为最近 {max_turns} 轮 ({max_messages} 条)"
                )
                messages = messages[-max_messages:]

        return messages

    # 向后兼容
    def format_context(self, query: str) -> str:
        return self.get_system_context(query)

    # ── 工具方法 ──────────────────────────────────────────────────────────

    def _has_l3_records(self) -> bool:
        """检查 L3（user 级持久记忆）是否有任何记录"""
        try:
            l3 = self.mem0.get_all(
                filters={
                    "user_id": self.user_id,
                    "memory_level": "L3",
                    "forgotten": False,
                }
            )
            return bool(
                self._filter(l3, memory_level="L3", include_forgotten=False)
            )
        except Exception:
            return False

    def _item_lifecycle(self, item: Dict) -> Optional[Dict]:
        metadata = item.get("metadata")
        if not isinstance(metadata, dict):
            metadata = item.get("_lifecycle")
        if isinstance(metadata, dict) and metadata.get("memory_hash"):
            stored = self.lifecycle.get(str(metadata["memory_hash"]))
            merged = dict(metadata)
            if stored:
                merged.update(stored)
            if item.get("id") and not merged.get("mem0_id"):
                merged["mem0_id"] = str(item["id"])
            return merged
        memory_hash = item.get("memory_hash")
        if memory_hash:
            stored = self.lifecycle.get(str(memory_hash))
            if stored:
                return stored
        text = item.get("memory")
        if isinstance(text, str):
            lifecycle = self.lifecycle.find_by_content(self.user_id, text)
            if lifecycle and item.get("id") and not lifecycle.get("mem0_id"):
                lifecycle["mem0_id"] = str(item["id"])
            return lifecycle
        return None

    def _filter(
        self,
        search_result,
        *,
        memory_level: Optional[str] = None,
        include_forgotten: bool = False,
        min_score: Optional[float] = None,
        historical_query: bool = False,
        recovery: bool = False,
    ) -> List[Dict]:
        if isinstance(search_result, dict):
            results = search_result.get("results", [])
        elif isinstance(search_result, list):
            results = search_result
        else:
            return []

        filtered = []
        expected_level = memory_level.upper() if memory_level else None
        for result in results:
            if not isinstance(result, dict):
                continue
            item = dict(result)
            lifecycle = self._item_lifecycle(item)
            item_level = expected_level
            memory_type = None
            if lifecycle:
                item["_lifecycle"] = lifecycle
                item_level = str(lifecycle.get("memory_level") or "").upper()
                memory_type = str(lifecycle.get("memory_type") or "")
                if expected_level and item_level and item_level != expected_level:
                    continue
                if not include_forgotten and lifecycle.get("forgotten", False):
                    continue
            elif expected_level == "L4":
                # 旧版 L4 记录没有生命周期元数据，但依然支持以 agent_id等特定字段进行过滤
                pass
            elif expected_level in {"L2", "L3"}:
                # 兼容生命周期元数据引入前写入的旧记忆。
                if item.get("agent_id") == "neuro_agent":
                    continue
                if expected_level == "L3" and item.get("run_id") is not None:
                    continue
                pass
            threshold = self._score_threshold_for(
                memory_level=item_level,
                memory_type=memory_type,
                min_score=min_score,
                historical_query=historical_query,
                recovery=recovery,
            )
            if result.get("score", 0) < threshold:
                continue
            filtered.append(item)
        return filtered

    def _score_threshold_for(
        self,
        *,
        memory_level: Optional[str] = None,
        memory_type: Optional[str] = None,
        min_score: Optional[float] = None,
        historical_query: bool = False,
        recovery: bool = False,
    ) -> float:
        if min_score is not None:
            return float(min_score)
        level = str(memory_level or "").upper()
        mtype = str(memory_type or "").lower()
        if recovery:
            if mtype == "recent_dialog":
                recent_threshold = getattr(
                    self.config,
                    "recovery_recent_dialog_threshold",
                    None,
                )
                if recent_threshold is not None:
                    return float(recent_threshold)
            return float(
                getattr(
                    self.config,
                    "recovery_threshold",
                    self.config.relevance_threshold,
                )
            )
        if level == "L2":
            field = "l2_history_threshold" if historical_query else "l2_relevance_threshold"
            return float(getattr(self.config, field, self.config.relevance_threshold))
        if level == "L3":
            if mtype == "recent_dialog":
                field = (
                    "l3_recent_dialog_history_threshold"
                    if historical_query
                    else "l3_recent_dialog_threshold"
                )
                return float(getattr(self.config, field, self.config.relevance_threshold))
            if mtype == "fact":
                return float(
                    getattr(
                        self.config,
                        "l3_fact_threshold",
                        self.config.relevance_threshold,
                    )
                )
            if mtype in {"visual", "visual_summary"}:
                return float(
                    getattr(
                        self.config,
                        "l3_visual_threshold",
                        self.config.relevance_threshold,
                    )
                )
            return float(
                getattr(
                    self.config,
                    "l3_default_threshold",
                    self.config.relevance_threshold,
                )
            )
        if level == "L4":
            return float(
                getattr(
                    self.config,
                    "l4_relevance_threshold",
                    self.config.relevance_threshold,
                )
            )
        return float(self.config.relevance_threshold)

    def _merge_items(self, *item_sets: List[Dict]) -> List[Dict]:
        seen, out = set(), []
        for items in item_sets:
            for item in items:
                text = item.get("memory")
                if not text:
                    continue
                if text not in seen:
                    seen.add(text)
                    out.append(item)
        return out

    def _mark_recalled(self, items: List[Dict]):
        hashes = []
        for item in items:
            if item.get("_skip_recall_mark"):
                continue
            lifecycle = item.get("_lifecycle") or self._item_lifecycle(item)
            if not lifecycle:
                continue
            level = str(lifecycle.get("memory_level") or "").upper()
            if level not in {"L2", "L3"}:
                continue
            memory_hash = lifecycle.get("memory_hash")
            if memory_hash:
                hashes.append(str(memory_hash))
        if hashes:
            for record in self.lifecycle.mark_recalled(hashes):
                self._sync_lifecycle_record_to_mem0(record)

    def run_forgetting_maintenance(
        self,
        *,
        current_valence: float = 0.0,
        current_arousal: float = 0.0,
        now: Optional[float] = None,
    ) -> Dict:
        """基于生命周期元数据执行遗忘扫描并写入逻辑遗忘标记。
           MVP 路径暂不实现物理删除。
        """
        now = now or time.time()
        if not self.config.enable_forgetting:
            return {
                "enabled": False,
                "reason": "forgetting_disabled",
                "candidate_count": 0,
                "selected_forget_count": 0,
                "forgotten_count": 0,
            }
        records = self.lifecycle.active_candidates(levels=("L2", "L3"))
        summary = self.forgetting_engine.build_forgetting_plan(
            records,
            now=now,
            current_valence=current_valence,
            current_arousal=current_arousal,
        )
        hashes = [
            decision["memory_hash"]
            for decision in summary.get("decisions", [])
            if decision.get("final_action") == "forget"
        ]
        updated_records = self.lifecycle.mark_forgotten(
            hashes,
            now=now,
            delay_days=self.config.physical_deletion_delay_days,
        )
        synced = sum(
            1 for record in updated_records
            if self._sync_lifecycle_record_to_mem0(record)
        )
        summary["forgotten_count"] = len(updated_records)
        summary["mem0_synced_count"] = synced
        self.lifecycle.set_last_scan_at(now)
        summary["enabled"] = True
        summary["lifecycle_write_failed"] = bool(
            getattr(self.lifecycle, "write_failed", False)
        )
        return summary

    @property
    def l1_usage(self) -> str:
        pct = self._l1_tokens / self.config.context_window_tokens * 100
        protected = min(len(self.working_memory), self.PROTECTED_TURNS)
        return (
            f"{self._l1_tokens}/{self.config.context_window_tokens} tokens "
            f"({pct:.1f}%) | 保护区 {protected}/{self.PROTECTED_TURNS} 轮 "
            f"| trigger={self.config.compression_trigger_tokens}"
        )
