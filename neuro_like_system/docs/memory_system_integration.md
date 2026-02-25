# 分级记忆系统集成方案

## 1. 现有架构分析

### 1.1 当前工作流

```
用户输入 → BERT分类器(情绪/行为/语气) → 简单10轮缓冲(MemoryManager) → 构建Prompt → LLM API → 输出
```

### 1.2 现有 MemoryManager 的局限

| 问题 | 说明 |
|---|---|
| 仅保留最近10轮 | 超出即丢弃，无法回溯早期对话 |
| 无语义检索 | 只能按时间顺序获取，无法按相关性召回 |
| 无压缩/摘要 | 每轮完整存储，占用上下文窗口 |
| 无持久化 | 进程结束即丢失所有记忆 |
| 无分级机制 | 所有信息同等对待，无重要性区分 |

### 1.3 相关源码位置

- `inference_pipeline.py:40-68` — `MemoryManager` 类（替换目标）
- `inference_pipeline.py:228-448` — `NeuroLikePipeline` 类（需适配）
- `inference_pipeline.py:324-357` — `build_system_prompt()` （prompt组装，需扩展）
- `configs/model_config.py` — 配置定义（需增加记忆系统配置）

---

## 2. 目标架构

### 2.1 新工作流

```
用户输入
  → BERT 分类器（情绪/行为/语气/强度）
  → 分级记忆系统
      ├── 召回相关记忆（L2/L3/L4）
      ├── 融合分类器输出 + 人格参数 + 历史上下文
      └── 组装 prompt（含输出格式约束：对话 or Agent调用）
  → GPT-5.2 API
      ├── 普通对话输出
      └── Agent 工具调用（结构化JSON）
  → 输出 / 执行工具后回环
```

### 2.2 记忆分级定义

```
┌─────────────────────────────────────────────────────┐
│ L1 工作记忆（In-Context）                            │
│   完整原始对话，直接拼入 prompt                      │
│   存储方式：内存列表（ConversationTurn[]）            │
│   触发上限：token 数达到 context_window × threshold  │
│   → 溢出时触发 L1→L2 压缩                           │
├─────────────────────────────────────────────────────┤
│ L2 会话摘要（Session Summary）                       │
│   L1 溢出部分经 LLM 压缩后的摘要文本                 │
│   存储方式：Mem0 session memory（向量DB）             │
│   触发上限：会话结束 → 触发 L2→L3 事实抽取           │
├─────────────────────────────────────────────────────┤
│ L3 情节记忆（Episodic Memory）                       │
│   跨会话的用户偏好、关键事件、重要决策                │
│   存储方式：Mem0 user memory（持久化向量DB）          │
│   触发条件：会话结束时由 LLM 自动抽取写入             │
├─────────────────────────────────────────────────────┤
│ L4 知识库（Knowledge Base）                          │
│   代码库结构、工具使用记录、分析结论                  │
│   存储方式：Mem0 agent memory（持久化向量DB）         │
│   触发条件：显式写入（Agent 执行完成后手动存储）       │
└─────────────────────────────────────────────────────┘
```

### 2.3 Token 计数导向的分级触发机制（初版策略）

初版采用最直接的方案：**以 LLM 上下文窗口边界作为分级标准**，类似 Claude 的 Compact 机制。

```
L1 token 计数
    │
    ├─ < threshold（如 80% × 128K = 102K）
    │     → 正常追加，不触发压缩
    │
    └─ ≥ threshold
          → 取 L1 最旧的 50% 轮次
          → LLM 压缩为摘要文本
          → 摘要写入 Mem0 L2（session memory）
          → 从 L1 删除已压缩的轮次
          → 继续追加新轮次

会话结束（程序退出 / 显式调用 close_session）
    → 从 Mem0 L2 召回本次会话所有摘要
    → LLM 抽取关键事实/用户偏好
    → 写入 Mem0 L3（user memory，持久化跨会话）
```

**主流 LLM 上下文窗口参考值：**

| 模型 | 上下文窗口 | 建议 threshold |
|---|---|---|
| GPT-5.2 Instant | 128K tokens | 80%（≈ 102K） |
| GPT-5.2 | 256K tokens | 80%（≈ 205K） |
| Claude Opus 4.6 | 1M tokens | 75%（≈ 768K） |
| Claude Sonnet 4.6 | 200K tokens | 80%（≈ 160K） |

**Token 计数方式：**

```python
# 推荐：使用 tiktoken（精确，适用于 GPT 系列）
import tiktoken
enc = tiktoken.encoding_for_model("gpt-4o")   # GPT-5.2 暂用 gpt-4o 编码
token_count = len(enc.encode(text))

# 备选：粗略估算（无需额外依赖）
# 中文约 1.5-2 char/token，英文约 4 char/token
def estimate_tokens(text: str) -> int:
    chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    other_chars = len(text) - chinese_chars
    return int(chinese_chars / 1.5 + other_chars / 4)
```

---

## 3. Mem0 API 概览

### 3.1 安装

```bash
pip install mem0ai
```

要求 Python 3.10+，默认依赖 OpenAI API（用于内部的事实抽取和嵌入）。

### 3.2 基础用法

```python
from mem0 import Memory

# 默认初始化（使用 OpenAI gpt-4.1-nano 做事实抽取）
m = Memory()

# 存储记忆
messages = [
    {"role": "user", "content": "我是小明，我喜欢打篮球和编程。"},
    {"role": "assistant", "content": "记住了，你喜欢篮球和编程！"}
]
m.add(messages, user_id="xiaoming")

# 检索记忆
results = m.search("我有什么爱好？", filters={"user_id": "xiaoming"}, limit=3)
# 返回结构:
# {
#   "results": [
#     {
#       "id": "mem_123abc",
#       "memory": "名字是小明，喜欢打篮球和编程",
#       "user_id": "xiaoming",
#       "categories": ["personal_info"],
#       "score": 0.89
#     }
#   ]
# }

# 获取所有记忆
all_memories = m.get_all(filters={"user_id": "xiaoming"})

# 删除记忆
m.delete(memory_id="mem_123abc")
```

### 3.3 自定义配置

Mem0 的四个可配置组件：向量存储、LLM、嵌入器、重排器。

```python
from mem0 import Memory

config = {
    # 向量存储：默认 Qdrant（本地磁盘），可选 Postgres/Pinecone/Chroma/Milvus 等
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "collection_name": "neuro_memory",
            "path": "./data/qdrant_db",     # 本地持久化路径
        }
    },

    # LLM：用于事实抽取和记忆更新（不是你的对话LLM，是Mem0内部用的）
    "llm": {
        "provider": "openai",
        "config": {
            "model": "gpt-5.2-instant",     # 可以复用你的GPT-5.2
            "temperature": 0.1,              # 事实抽取用低温度
            "api_key": "your-api-key",
        }
    },

    # 嵌入器：用于向量化记忆内容
    "embedder": {
        "provider": "openai",
        "config": {
            "model": "text-embedding-3-small",
        }
    },

    # 重排器（可选）：对检索结果二次排序
    "reranker": {
        "provider": "cohere",
        "config": {
            "model": "rerank-english-v3.0",
            "top_k": 10,
        }
    }
}

memory = Memory.from_config(config)
```

也支持 YAML 配置文件：

```python
memory = Memory.from_config_file("config.yaml")
```

### 3.4 三级记忆隔离

Mem0 原生支持通过 `user_id` / `session_id` / `agent_id` 实现记忆隔离：

```python
# L2 会话级记忆
m.add(messages, user_id="user_1", session_id="session_20260222")
m.search(query, filters={"user_id": "user_1", "session_id": "session_20260222"})

# L3 用户级记忆（跨会话持久化）
m.add(messages, user_id="user_1")
m.search(query, filters={"user_id": "user_1"})

# L4 Agent级记忆（工具使用记录、知识库）
m.add(messages, agent_id="neuro_agent")
m.search(query, filters={"agent_id": "neuro_agent"})
```

### 3.5 默认存储

| 组件 | 默认值 |
|---|---|
| LLM | OpenAI `gpt-4.1-nano-2025-04-14` |
| Embeddings | OpenAI `text-embedding-3-small` (1536维) |
| Vector Store | Qdrant 本地磁盘 `/tmp/qdrant` |
| History | SQLite `~/.mem0/history.db` |
| Reranker | 无 |

---

## 4. 集成方案

### 4.1 新增配置项（model_config.py）

在 `configs/model_config.py` 中增加：

```python
@dataclass
class MemoryConfig:
    """分级记忆系统配置"""

    # Mem0 后端配置
    vector_store_path: str = "./data/qdrant_db"
    collection_name: str = "neuro_memory"

    # Mem0 内部 LLM（事实抽取/压缩用，复用对话LLM的key和endpoint）
    mem0_llm_model: str = "gpt-5.2-instant"
    mem0_llm_temperature: float = 0.1       # 压缩摘要用低温度
    mem0_api_key: Optional[str] = None      # 默认从环境变量读取
    mem0_base_url: Optional[str] = None     # 第三方供应商时需要设置

    # ── Token 计数导向的分级参数 ──────────────────────────
    # LLM 上下文窗口大小（按实际使用模型填写）
    context_window_tokens: int = 128_000    # GPT-5.2 Instant: 128K

    # L1 压缩触发阈值（占上下文窗口的比例）
    # 达到此比例时触发 L1→L2 压缩，留出空间给新对话和召回内容
    compression_threshold: float = 0.75    # 建议 0.7-0.85

    # L1 每次压缩的比例（压缩最旧的这部分轮次）
    compression_ratio: float = 0.5         # 压缩最旧的50%

    # 召回参数
    l3_search_limit: int = 5               # 情节记忆每次召回条数
    l4_search_limit: int = 3               # 知识库每次召回条数
    relevance_threshold: float = 0.7       # 最低相关性分数

    @property
    def compression_trigger_tokens(self) -> int:
        """触发压缩的 token 绝对值"""
        return int(self.context_window_tokens * self.compression_threshold)

    def __post_init__(self):
        if self.mem0_api_key is None:
            self.mem0_api_key = os.environ.get("OPENAI_API_KEY")
```

### 4.2 替换 MemoryManager（inference_pipeline.py）

用新的 `HierarchicalMemoryManager` 替换现有的 `MemoryManager`：

```python
import tiktoken
from mem0 import Memory

def _count_tokens(text: str, model: str = "gpt-4o") -> int:
    """计算文本 token 数（精确版，用 tiktoken）"""
    try:
        enc = tiktoken.encoding_for_model(model)
        return len(enc.encode(text))
    except Exception:
        # 备用粗略估算
        chinese = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        return int(chinese / 1.5 + (len(text) - chinese) / 4)

def _turn_to_text(turn: ConversationTurn) -> str:
    """将对话轮次序列化为文本（用于 token 计数）"""
    return f"用户: {turn.user_input}\n助手: {turn.response}\n"


class HierarchicalMemoryManager:
    """
    Token 计数导向的分级记忆管理器

    触发逻辑：
      L1 token数 ≥ compression_trigger_tokens
        → 压缩最旧的 compression_ratio 部分 → 写入 Mem0 L2
      会话结束（close_session）
        → 从 L2 抽取关键事实 → 写入 Mem0 L3（持久化）
    """

    def __init__(self, config: MemoryConfig, llm_client, user_id: str = "default"):
        self.config = config
        self.user_id = user_id
        self.session_id = f"session_{int(time_module.time())}"
        self.llm_client = llm_client    # 复用 NeuroLikePipeline 的 LLMClient

        # L1 工作记忆：完整原始轮次
        self.working_memory: List[ConversationTurn] = []
        self._l1_token_count: int = 0

        # Mem0 后端（L2/L3/L4）
        mem0_config = {
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "collection_name": config.collection_name,
                    "path": config.vector_store_path,
                }
            },
            "llm": {
                "provider": "openai",
                "config": {
                    "model": config.mem0_llm_model,
                    "temperature": config.mem0_llm_temperature,
                    "api_key": config.mem0_api_key,
                    **({"base_url": config.mem0_base_url}
                       if config.mem0_base_url else {})
                }
            }
        }
        self.mem0 = Memory.from_config(mem0_config)

    # ── L1 操作 ─────────────────────────────────────────

    def add(self, turn: ConversationTurn):
        """添加对话轮次，自动触发分级压缩"""
        text = _turn_to_text(turn)
        self.working_memory.append(turn)
        self._l1_token_count += _count_tokens(text)

        # 检查是否需要压缩
        if self._l1_token_count >= self.config.compression_trigger_tokens:
            self._compress_l1_to_l2()

    def _compress_l1_to_l2(self):
        """
        L1 → L2 压缩：取最旧的 compression_ratio 部分，
        用 LLM 压缩为摘要，写入 Mem0 session memory
        """
        n = len(self.working_memory)
        compress_count = max(1, int(n * self.config.compression_ratio))
        to_compress = self.working_memory[:compress_count]

        # 构造压缩 prompt
        history_text = "".join(_turn_to_text(t) for t in to_compress)
        summary = self.llm_client.generate(
            system_prompt=(
                "你是一个对话摘要助手。请将以下对话压缩为简洁的要点摘要，"
                "保留：用户的关键偏好、重要事件、情绪变化趋势、未解决的问题。"
                "摘要应简洁，不超过200字。"
            ),
            user_input=history_text,
            max_tokens=300,
            temperature=0.1
        )

        # 写入 Mem0 L2（session 级）
        self.mem0.add(
            [{"role": "assistant", "content": f"[会话摘要] {summary}"}],
            user_id=self.user_id,
            session_id=self.session_id
        )

        # 从 L1 移除已压缩的轮次，更新 token 计数
        removed_tokens = sum(
            _count_tokens(_turn_to_text(t)) for t in to_compress
        )
        self.working_memory = self.working_memory[compress_count:]
        self._l1_token_count = max(0, self._l1_token_count - removed_tokens)

    # ── L2 → L3 会话结束处理 ────────────────────────────

    def close_session(self):
        """
        会话结束时调用：将本次会话摘要抽取为长期记忆写入 L3
        应在程序退出或用户主动结束会话时调用
        """
        # 先压缩当前 L1 剩余部分
        if self.working_memory:
            self._compress_l1_to_l2()

        # 召回本次会话的所有 L2 摘要
        session_memories = self.mem0.get_all(
            filters={"user_id": self.user_id, "session_id": self.session_id}
        )
        if not session_memories or not session_memories.get("results"):
            return

        all_summaries = "\n".join(
            r["memory"] for r in session_memories["results"]
        )

        # LLM 抽取关键事实 → 写入 L3（user 级，跨会话持久化）
        key_facts = self.llm_client.generate(
            system_prompt=(
                "请从以下会话摘要中抽取用户的长期偏好、重要特征、关键决策，"
                "用简洁的条目格式输出，每条不超过50字。"
            ),
            user_input=all_summaries,
            max_tokens=400,
            temperature=0.1
        )

        self.mem0.add(
            [{"role": "assistant", "content": key_facts}],
            user_id=self.user_id    # 无 session_id → 写入 L3 用户级记忆
        )

    # ── 召回与格式化 ─────────────────────────────────────

    def format_context(self, query: str) -> str:
        """
        格式化完整上下文（供 prompt 组装使用），替代原 format_context()

        返回结构：
          ## 最近对话    ← L1 最近几轮（直接展示）
          ## 会话记忆    ← L2 召回的摘要
          ## 历史记忆    ← L3 召回的跨会话事实
          ## 知识库      ← L4 召回的 Agent 知识
        """
        sections = []

        # L1：最近若干轮（直接来自内存，无需检索）
        # 数量由剩余 token 预算决定：为 L2/L3 召回留 20% 空间
        l1_budget = int(self.config.context_window_tokens * 0.2)
        recent_turns = self._get_recent_turns_within_budget(l1_budget)
        if recent_turns:
            sections.append("## 最近对话\n" + "".join(
                _turn_to_text(t) for t in recent_turns
            ))

        # L2/L3：语义检索（Mem0 自动处理向量检索）
        l2 = self.mem0.search(
            query=query,
            filters={"user_id": self.user_id, "session_id": self.session_id},
            limit=self.config.l3_search_limit
        )
        l3 = self.mem0.search(
            query=query,
            filters={"user_id": self.user_id},
            limit=self.config.l3_search_limit
        )

        combined = self._merge_and_filter(l2, l3)
        if combined:
            sections.append("## 相关记忆\n" + "\n".join(
                f"- {m}" for m in combined
            ))

        # L4：知识库
        l4 = self.mem0.search(
            query=query,
            filters={"agent_id": "neuro_agent"},
            limit=self.config.l4_search_limit
        )
        l4_items = self._filter_by_score(l4)
        if l4_items:
            sections.append("## 知识库\n" + "\n".join(
                f"- {m['memory']}" for m in l4_items
            ))

        return "\n\n".join(sections) if sections else "(首次对话)"

    # ── 工具方法 ─────────────────────────────────────────

    def _get_recent_turns_within_budget(self, token_budget: int
                                        ) -> List[ConversationTurn]:
        """从 L1 尾部取若干轮，不超过 token_budget"""
        result, used = [], 0
        for turn in reversed(self.working_memory):
            cost = _count_tokens(_turn_to_text(turn))
            if used + cost > token_budget:
                break
            result.insert(0, turn)
            used += cost
        return result

    def _filter_by_score(self, search_results) -> List[Dict]:
        if not search_results or "results" not in search_results:
            return []
        return [r for r in search_results["results"]
                if r.get("score", 0) >= self.config.relevance_threshold]

    def _merge_and_filter(self, *result_sets) -> List[str]:
        seen, out = set(), []
        for rs in result_sets:
            for item in self._filter_by_score(rs):
                text = item["memory"]
                if text not in seen:
                    seen.add(text)
                    out.append(text)
        return out

    @property
    def l1_token_usage(self) -> str:
        """当前 L1 token 使用情况（调试用）"""
        pct = self._l1_token_count / self.config.context_window_tokens * 100
        return (f"{self._l1_token_count}/{self.config.context_window_tokens} "
                f"tokens ({pct:.1f}%) | "
                f"trigger at {self.config.compression_trigger_tokens}")
```

### 4.3 适配 NeuroLikePipeline

最小改动，只需修改 `__init__` 和 `chat` 中的记忆相关调用：

```python
class NeuroLikePipeline:
    def __init__(self, ..., memory_config: Optional[MemoryConfig] = None):
        # ... 其他初始化不变 ...

        # 记忆管理器（替换原来的 MemoryManager()）
        if memory_config:
            self.memory = HierarchicalMemoryManager(memory_config)
        else:
            # 向后兼容：无配置时降级为简单缓冲
            self.memory = MemoryManager()

    def chat(self, user_input: str, verbose: bool = False) -> Dict:
        # 1. 分析情绪和行为（不变）
        emotion_behavior = self.analyze_emotion_behavior(user_input)

        # 2. 获取上下文（适配新接口）
        if isinstance(self.memory, HierarchicalMemoryManager):
            context = self.memory.format_context(query=user_input)
        else:
            context = self.memory.format_context(num_turns=5)

        # 3. 生成回复（不变）
        response = self.generate_response(user_input, emotion_behavior, context)

        # 4. 保存记忆（不变，ConversationTurn 接口一致）
        turn = ConversationTurn(...)
        self.memory.add(turn)

        return { ... }
```

### 4.4 Prompt 组装扩展（支持 Agent 调用）

`build_system_prompt()` 末尾增加输出格式约束：

```python
def build_system_prompt(self, emotion_behavior: Dict, context: str = "",
                        enable_agent: bool = False) -> str:
    prompt = f"""你是{self.personality.name}，一个AI虚拟主播。

# 人格特质
...（现有内容不变）

# 当前状态
...（现有内容不变）

# 记忆上下文
{context if context else '(首次对话)'}

# 指令
请根据以上人格设定和当前状态，用自然、符合人设的方式回复用户。
"""

    if enable_agent:
        prompt += """
# 输出格式
当你需要执行工具/代码分析时，使用以下JSON格式：
```json
{
    "type": "agent_call",
    "tool": "工具名称",
    "params": { ... },
    "reason": "调用原因"
}
```
当你正常回复时，直接输出文本即可。
"""

    return prompt
```

---

## 5. 数据流示意

```
用户: "帮我看看这段代码的性能问题"
                │
                ▼
    ┌──── BERT 分类器 ────┐
    │ emotion: curiosity   │
    │ behavior: ask_question│
    │ tone: serious        │
    └──────────┬───────────┘
               │
               ▼
    ┌── 分级记忆系统 ──────────────────────────┐
    │                                           │
    │  L1 (工作记忆):                           │
    │    最近3轮: 用户提到了Python项目...         │
    │                                           │
    │  L2 (会话缓冲):                           │
    │    本次会话: 用户在调试一个Flask应用...      │
    │                                           │
    │  L3 (情节记忆):                           │
    │    历史: 用户偏好使用cProfile分析性能...     │
    │                                           │
    │  L4 (知识库):                             │
    │    工具: stack_tracer 可追踪调用栈...       │
    │                                           │
    │  → 组装 prompt (含Agent输出格式约束)       │
    └──────────┬───────────────────────────────┘
               │
               ▼
    ┌──── GPT-5.2 API ────┐
    │ 输出:                │
    │ {                    │
    │   "type": "agent_call│
    │   "tool": "profiler" │
    │   "params": {...}    │
    │ }                    │
    └──────────┬───────────┘
               │
               ▼
    ┌── 输出解析 ──────────┐
    │ 检测到 agent_call     │
    │ → 执行 profiler 工具  │
    │ → 结果回传 LLM        │
    │ → 生成最终回复        │
    └──────────────────────┘
```

---

## 6. 依赖项

```
# requirements.txt 新增
mem0ai>=1.0.0
qdrant-client>=1.7.0       # Mem0 默认向量存储后端
```

注意：Mem0 内部默认使用 OpenAI API 做事实抽取（gpt-4.1-nano），
这会消耗少量额外 token。可通过配置切换为你的第三方供应商 endpoint。

---

## 7. 后续扩展方向

### 7.1 Agent 工具注册系统

```python
@dataclass
class ToolConfig:
    """Agent 工具配置"""
    tools: Dict[str, Dict] = field(default_factory=lambda: {
        "stack_tracer": {
            "description": "追踪Python调用栈",
            "module": "tools.stack_tracer",   # CPython C扩展
            "params": ["target_function", "depth"]
        },
        "profiler": {
            "description": "性能分析",
            "module": "tools.profiler",
            "params": ["script_path", "duration"]
        },
        "memory_analyzer": {
            "description": "内存分配分析",
            "module": "tools.memory_analyzer",
            "params": ["pid", "top_n"]
        }
    })
```

### 7.2 CPython C ABI 工具（独立开发）

```
neuro_like_system/
└── tools/                        # 新增
    ├── __init__.py
    ├── stack_tracer.c            # C扩展：帧栈追踪
    ├── profiler.c                # C扩展：性能计数器
    ├── memory_analyzer.c         # C扩展：内存分析
    └── setup.py                  # C扩展编译配置
```

### 7.3 记忆压缩策略优化

当前方案使用简单截断（保留最近5轮）。后续可引入：

- **语义压缩**：用LLM将多轮对话压缩为摘要存入L2
- **重要性评分**：基于情绪强度、行为类型判断记忆重要性
- **遗忘曲线**：模拟人类记忆衰减，自动降级/删除低价值记忆

参考论文：
- SimpleMem 的三阶段压缩管线
- A-MEM 的 Zettelkasten 动态链接方法

---

## 8. 参考资料

- [Mem0 GitHub](https://github.com/mem0ai/mem0) — Apache 2.0
- [Mem0 文档](https://docs.mem0.ai)
- [MemGPT / Letta](https://github.com/letta-ai/letta) — 虚拟上下文管理原始论文实现
- [A-MEM (NeurIPS 2025)](https://arxiv.org/abs/2502.12110) — 动态记忆组织
- [SimpleMem](https://github.com/aiming-lab/SimpleMem) — 语义无损压缩
- [MemEngine](https://github.com/nuster1128/MemEngine) — 统一记忆框架
- [Chain of Agents (Google)](https://research.google/blog/chain-of-agents-large-language-models-collaborating-on-long-context-tasks/) — 多Agent长上下文协作
 1. 多智能体协作分治
  - https://research.google/blog/chain-of-agents-large-language-models-collaborating-on-long-context-tasks/ —
  将长文档切片，多个 worker agent 处理后汇总给 manager agent。训练无关、任务无关，优于单纯扩展上下文窗口
  - https://arxiv.org/abs/2505.20096 — Planner/Extractor/QA 等专职 agent 协作推理，在多跳问答上显著超越单模型 RAG

  2. 虚拟内存/分层记忆
  - https://research.memgpt.ai/ — 仿操作系统的分页内存管理，将历史信息在主存/外存间调度，是最早的系统性方案
  - https://arxiv.org/abs/2502.12110 — 基于 Zettelkasten 方法动态构建记忆知识网络，而非静态存储

  3. 综述论文
  - https://www.techrxiv.org/users/1007269/articles/1367390 — 系统梳理多智能体系统中记忆的机制、挑战与集体协作
  - https://dl.acm.org/doi/10.1145/3748302 — ACM 正刊综述，覆盖记忆存储、检索、更新全流程
  - https://github.com/Shichun-Liu/Agent-Memory-Paper-List — 持续更新的论文列表，包含 G-Memory、MAGMA、MIRIX 等 2025
  年最新工作

  4. 2025年的核心争论
  长上下文窗口 vs RAG 的路线之争在 2025 年有了实践答案：混合架构（长窗口 + 分层 RAG + 多 agent）优于单一方案，企业侧 RAG
   并未被大上下文窗口替代。