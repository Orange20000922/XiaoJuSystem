"""
分级记忆管理器

压缩策略：
  - 保护区（最近 PROTECTED_TURNS 轮）永不压缩
  - 压缩目标是结构化状态提取，而非叙述性摘要
  - 双条件触发：token 压力 OR BERT 检测到话题边界
"""

import json
import time
from typing import List, Dict, Optional, TYPE_CHECKING

import tiktoken
from mem0 import Memory

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.model_config import MemoryConfig

if TYPE_CHECKING:
    from src.inference_pipeline import ConversationTurn, LLMClient


# tiktoken 编码器缓存（gpt-4o 编码兼容 GPT-5.2）
_ENCODER = None

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
        user_id: str = "default",
    ):
        self.config = config
        self.llm_client = llm_client
        self.user_id = user_id
        self.session_id = f"session_{int(time.time())}"

        # L1
        self.working_memory: List["ConversationTurn"] = []
        self._l1_tokens: int = 0

        # 最近一次 BERT 分类的行为标签，由外部（Pipeline）在 add() 前设置
        self.last_behavior: str = ""

        # Mem0 后端
        mem0_cfg: Dict = {
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "collection_name": config.collection_name,
                    "path": config.vector_store_path,
                },
            },
            "llm": {
                "provider": "openai",
                "config": {
                    "model": config.mem0_llm_model,
                    "temperature": config.mem0_llm_temperature,
                    "api_key": config.mem0_api_key,
                    **({"base_url": config.mem0_base_url}
                       if config.mem0_base_url else {}),
                },
            },
        }
        self.mem0 = Memory.from_config(mem0_cfg)

    # ── L1 操作 ──────────────────────────────────────────────────────────

    def add(self, turn: "ConversationTurn"):
        """
        添加对话轮次，双条件触发压缩：
          1. token 达到阈值（硬性触发）
          2. BERT 检测到话题边界且 token 超过最低占比（软性触发）
        """
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
        n = len(self.working_memory)
        compressible_end = max(0, n - self.PROTECTED_TURNS)

        if compressible_end == 0:
            # 全部在保护区内，无可压缩内容
            return

        # 取可压缩区中最旧的 compression_ratio 部分
        cut = max(1, int(compressible_end * self.config.compression_ratio))
        to_compress = self.working_memory[:cut]

        history_text = "".join(turn_to_text(t) for t in to_compress)

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

        self.mem0.add(
            [{"role": "assistant", "content": content}],
            user_id=self.user_id,
            session_id=self.session_id,
        )

        removed_tokens = sum(count_tokens(turn_to_text(t)) for t in to_compress)
        self.working_memory = self.working_memory[cut:]
        self._l1_tokens = max(0, self._l1_tokens - removed_tokens)

    # ── L2 → L3 会话结束 ─────────────────────────────────────────────────

    def close_session(self):
        """
        会话结束时调用：
          1. 压缩保护区以外的剩余 L1
          2. 从本次会话的状态快照中抽取长期事实写入 L3
        """
        # 先压缩剩余可压缩区（保护区内容丢弃，它们是最新的已在对话中体现）
        n = len(self.working_memory)
        compressible_end = max(0, n - self.PROTECTED_TURNS)
        if compressible_end > 0:
            self._compress_l1_to_l2()

        session_mems = self.mem0.get_all(
            filters={"user_id": self.user_id, "session_id": self.session_id}
        )
        if not session_mems or not session_mems.get("results"):
            return

        all_states = "\n".join(r["memory"] for r in session_mems["results"])

        raw_facts = self.llm_client.generate(
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
            self.mem0.add(
                [{"role": "assistant", "content": content}],
                user_id=self.user_id,  # 无 session_id → L3 持久化
            )

    # ── L4 显式写入 ───────────────────────────────────────────────────────

    def add_knowledge(self, content: str):
        """显式写入 L4 知识库（Agent 执行结果、分析结论等）"""
        self.mem0.add(
            [{"role": "assistant", "content": content}],
            agent_id="neuro_agent",
        )

    # ── 召回与 prompt 组装 ────────────────────────────────────────────────

    def format_context(self, query: str) -> str:
        """
        供 build_system_prompt() 调用。
        L1 保护区原文优先，L2/L3 状态快照按语义检索补充。
        """
        sections = []

        # L1 保护区：最近 PROTECTED_TURNS 轮原文（全部注入，不受 budget 限制）
        protected = self.working_memory[-self.PROTECTED_TURNS:]
        if protected:
            sections.append(
                "## 最近对话\n" + "".join(turn_to_text(t) for t in protected)
            )

        # L2：当前会话状态快照（语义检索）
        l2 = self.mem0.search(
            query=query,
            filters={"user_id": self.user_id, "session_id": self.session_id},
            limit=self.config.l3_search_limit,
        )
        # L3：跨会话长期记忆（语义检索）
        l3 = self.mem0.search(
            query=query,
            filters={"user_id": self.user_id},
            limit=self.config.l3_search_limit,
        )
        combined = self._merge(l2, l3)
        if combined:
            sections.append(
                "## 历史上下文\n" + "\n".join(f"- {m}" for m in combined)
            )

        # L4：知识库
        l4 = self.mem0.search(
            query=query,
            filters={"agent_id": "neuro_agent"},
            limit=self.config.l4_search_limit,
        )
        l4_items = self._filter(l4)
        if l4_items:
            sections.append(
                "## 知识库\n" + "\n".join(f"- {m['memory']}" for m in l4_items)
            )

        return "\n\n".join(sections) if sections else "(首次对话)"

    # ── 工具方法 ──────────────────────────────────────────────────────────

    def _filter(self, search_result: Optional[Dict]) -> List[Dict]:
        if not search_result or "results" not in search_result:
            return []
        return [
            r for r in search_result["results"]
            if r.get("score", 0) >= self.config.relevance_threshold
        ]

    def _merge(self, *result_sets) -> List[str]:
        seen, out = set(), []
        for rs in result_sets:
            for item in self._filter(rs):
                text = item["memory"]
                if text not in seen:
                    seen.add(text)
                    out.append(text)
        return out

    @property
    def l1_usage(self) -> str:
        pct = self._l1_tokens / self.config.context_window_tokens * 100
        protected = min(len(self.working_memory), self.PROTECTED_TURNS)
        return (
            f"{self._l1_tokens}/{self.config.context_window_tokens} tokens "
            f"({pct:.1f}%) | 保护区 {protected}/{self.PROTECTED_TURNS} 轮 "
            f"| trigger={self.config.compression_trigger_tokens}"
        )
