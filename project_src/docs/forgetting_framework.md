
# 遗忘机制设计文档

## 1. 设计目标

遗忘机制不是为了做简单的“超时就删”，而是给 L2/L3 记忆加一套可解释、可回测、可关掉的生命周期管理。

核心目标：

- 模拟人类那种不精确的遗忘，不是到点就扔。
- 综合时间、有效召回次数、记忆层级、情绪强度和状态标签，计算每条记忆的“保留权重”。
- 先做逻辑遗忘（标记为忘掉，但普通召回看不见），留一个悔过期，如果用户问起类似内容，可以恢复。
- 物理删除必须延迟执行，并且单独用开关控制，防止早期误删。
- 第一版必须先用 dry-run 观察决策分布，确认没问题再允许真正打上 `forgotten=true`。

当前 MVP 已经实现：

- L2/L3 写入时带上生命周期 metadata。
- 这些 metadata 会同步到 Mem0/Qdrant 的 payload 里。
- 普通召回默认过滤掉 `forgotten=true` 的 L2/L3。
- `run_forgetting_maintenance()` 只做真正的逻辑遗忘；dry-run 由测试直接调用决策树。
- `recovery_search()` 支持在悔过期里恢复被遗忘的记忆。
- 已经做了真实的 Mem0/Qdrant metadata 往返集成测试。

还没做的：

- 物理删除。
- 持久化的树结构。
- root-oriented kNN 树。
- recovery 频率限制。
- 多客户端分布式维护锁。

## 2. 适用范围

| 层级 | 含义 | 是否参与随机遗忘 | 说明 |
|---|---:|---|
| L1 | 工作记忆，当前窗口原文 | 否 | 由 `PROTECTED_TURNS` 和上下文窗口控制，不进入 Mem0 遗忘扫描 |
| L2 | 会话状态快照 | 是 | 衰减更快，适合比较激进的生命周期管理 |
| L3 | 长期向量记忆 | 是 | 包含事实、近期原文片段、视觉摘要等，需要保守一点 |
| L4 | 知识库/稳定画像/能力基线 | 否 | 默认不参与随机时间遗忘，只允许显式更新、取代或过期 |

L4 虽然不做随机遗忘，但以后可以支持显式生命周期字段，比如：

```text
superseded_by
valid_until
confidence
memory_type
```

注意：L4 并不是所有内容都永久等价。用户画像、能力描述、尾端情绪快照、视觉观察这些应该用 `memory_type` 区分开，后续按类型处理更新、取代和过期。

## 3. Metadata 约束

遗忘机制不能依赖隐式的层级判断。所有新写入 Mem0 的 L2/L3/L4 记忆都必须带明确的 metadata。

之前的隐式约定：

```text
L2: user_id + run_id
L3: user_id
L4: user_id + agent_id="neuro_agent"
```

现在用显式 metadata：

```json
{
  "memory_level": "L2|L3|L4",
  "memory_type": "state_snapshot|recent_dialog|fact|visual|visual_summary|profile|capability|emotion_snapshot",
  "created_at": "2024-03-09T16:00:00+00:00",
  "lifecycle_created_at": 1710000000.0,
  "last_recalled_at": null,
  "recall_count": 0,
  "forgotten": false,
  "forgotten_at": null,
  "deleted_after": null,
  "emotion": "sadness",
  "emotion_intensity": 0.72,
  "state_label": "sadness",
  "state_valence": -0.31,
  "state_arousal": 0.48,
  "sustained_label": "sadness",
  "sustained_turns": 3,
  "behavior": "seek_clarification",
  "tone": "calm",
  "context_id": "group:123",
  "user_id": "owner",
  "run_id": "session_...",
  "agent_id": null,
  "memory_hash": "...",
  "content_hash": "...",
  "mem0_id": "..."
}
```

几个兼容性注意点：

- `created_at` 用 ISO 字符串，兼容 Mem0 2.0 的 payload 处理。
- `lifecycle_created_at` 用 float 时间戳，供遗忘计算用。
- `memory_hash` 是生命周期里的主键。
- `content_hash` 用于旧记录或返回结果缺少 metadata 时做 fallback 查找。
- `mem0_id` 记下真实向量库的 id，用来把生命周期更新同步回 Mem0 payload。

召回计数的规则：

- 只有最终进了 prompt 的 L2/L3 记忆才算有效召回。
- 那些只被向量库返回、但因为阈值或 forgotten 过滤掉的候选，不更新 `recall_count`。
- `recovery_search()` 恢复的记忆会在 `reactivate()` 里更新一次召回计数，并且通过 `_skip_recall_mark` 避免本轮重复计数。

## 4. 触发时机

推荐触发点：

1. 客户端冷启动后，如果距离上次维护超过 `forgetting_interval_hours`。
2. 客户端退出时。
3. AgentLoop 空闲时。
4. 每天固定低峰时间。
5. L2/L3 总量超过 `memory_count_trigger` 时。

扫描方式：

- 后台低优先级任务。
- 单次扫描只处理 L2/L3 里的 active candidates。
- 同一个 `user_id + collection` 同一时间只允许一个遗忘维护任务运行。

当前 MVP 还没做分布式维护锁。如果以后有多个客户端或多进程同时访问同一个 collection，需要加文件锁、数据库锁或任务租约。

## 5. 保留权重

`W` 表示 `retention_weight`，即保留权重。`W` 越大，记忆越不容易被遗忘。

基础公式：

```text
W = W_base * exp(-lambda * delta_days) * (1 + alpha * log1p(recall_count)) * M_encode * M_scan
```

参数解释：

| 参数 | 含义 |
|---|---|
| `W_base` | 记忆层级的基础权重，L2 < L3 |
| `lambda` | 时间衰减率，L2 > L3 |
| `delta_days` | 距离上次有效召回的天数；从未召回时，用当前时间减创建时间 |
| `alpha` | 召回强化系数 |
| `recall_count` | 有效召回次数，只统计进了 prompt 的那些 |
| `M_encode` | 写入时的情绪/状态显著性 |
| `M_scan` | 扫描时当前状态带来的全局调制 |

当前默认值：

```yaml
base_weight_l2: 0.70
base_weight_l3: 1.00
lambda_l2: 0.30
lambda_l3: 0.10
alpha_recall: 0.20
```

用 `log1p(recall_count)` 而不是线性 `recall_count`，避免高频记忆很快变成几乎忘不掉。

## 6. 情绪与状态机因子

写入时的显著性：

```text
M_encode = 1 + encoding_intensity_coeff * emotion_intensity
             + encoding_arousal_coeff * abs(state_arousal)
```

裁剪范围：

```text
M_encode in [0.7, 1.5]
```

扫描时的全局调制：

```text
M_scan = 1 + mood_coeff_v * current_valence + mood_coeff_a * current_arousal
```

裁剪范围：

```text
M_scan in [0.8, 1.2]
```

设计上的考虑：

- 一条记忆是否显著，优先由写入时的 metadata 决定。
- 当前的 OU 状态只作为本轮扫描的全局偏移，不能过度支配所有历史记忆。
- 第一版可以先用已有的 `emotion_intensity`、`state_arousal` 和 `behavior` 近似，不强依赖完整状态机快照。

后续可以配置特殊规则，例如：

- 当 `arousal > 0.8` 时，跳过本次扫描，或降低全局剪枝概率。
- 当 `valence < -0.6` 且 `arousal < 0.2` 时，将剪枝概率乘以 1.2。

这些规则必须先通过测试 dry-run 记录影响，再允许启用。

## 7. 边界保护

硬保护条件：

- `memory_level == "L4"`
- `memory_type in {"capability", "profile", "safety", "active_task"}`
- `now - lifecycle_created_at < min_retention_days`

当前 MVP 的 `_hard_keep_reason()` 已经实现了上述三类硬保护。

设计上建议后续增加：

- `last_recalled_at` 非常近
- `recall_count >= high_recall_threshold`
- `emotion_intensity >= high_emotion_threshold`
- `behavior in {"ask_question", "seek_clarification"}`
- `protected_until` 未过期

注意：L1 的 `PROTECTED_TURNS` 只保护工作记忆里的近期轮次，不等于 L3 已经有的保护区。L3 写入后仍需要 `min_retention_days` 或 `protected_until` 作为兜底。

## 8. 遗忘决策流程

### 8.1 读取候选记忆

从 lifecycle store 中读取所有 active L2/L3：

```python
records = lifecycle.active_candidates(levels=("L2", "L3"))
```

候选字段：

```text
memory_hash
memory_level
memory_type
created_at
lifecycle_created_at
last_recalled_at
recall_count
emotion_intensity
state_label
state_valence
state_arousal
sustained_label
sustained_turns
behavior
forgotten
mem0_id
```

第一版遗忘判断不使用语义 embedding，只使用生命周期和状态标签特征。

### 8.2 计算生命周期特征

主要特征：

- `age_norm`：距离创建时间
- `idle_norm`：距离上次有效召回时间
- `recall_resistance`：`recall_count` 带来的抗遗忘能力
- `level_pressure`：L2 高于 L3，L4 不进入
- `emotion_salience`：情绪强度或状态机偏离基线
- `state_stability`：`sustained_turns` / `sustained_label`
- `behavior_salience`：行为标签显著性
- `retention_weight`：`W`

当前 MVP 直接计算 `retention_weight`，并按归一化的 `W` 分桶。

### 8.3 临时生命周期树

设计目标是构造一棵临时树，使 `root -> leaf` 的方向整体上越来越容易遗忘。树结构不持久化，每次维护任务临时计算。

当前 MVP 的实现是简化版：

1. 以 `retention_weight` 为主轴。
2. 使用 `configured_W_ref` 和本轮最大 `W` 得到归一化参考值。
3. 按 `W` 分为四个桶：
   - `W_norm >= 0.75`
   - `0.50 <= W_norm < 0.75`
   - `0.25 <= W_norm < 0.50`
   - `W_norm < 0.25`
4. 桶的顺序等价于临时树的深度。
5. 深度越大，剪枝概率越高。

后续可替换为 root-oriented kNN tree：

```text
feature(i) = [
  W_norm,
  idle_norm,
  age_norm,
  recall_norm,
  level_code,
  emotion_salience,
  state_stability,
  behavior_salience
]

forget_distance(i) = weighted_distance(feature(i), root)
parent(i) = nearest node j where forget_distance(j) < forget_distance(i)
```

约束：

- `W_norm` 必须是主维度。
- `root -> leaf` 必须整体越来越容易遗忘。
- kNN tree 只用于生命周期结构，不使用语义 embedding。

## 9. 剪枝概率

先把保留权重转换成遗忘压力：

```text
W_ref = max(scan_W_max, configured_W_ref)
W_norm = clamp(avg_bucket_W / W_ref, 0, 1)
forget_pressure = 1 - W_norm
```

基础剪枝概率：

```text
prune_prob = clamp(
  forget_pressure + depth_bias * depth + random_jitter,
  0,
  max_prune_prob
)
```

参数说明：

| 参数 | 含义 |
|---|---|
| `configured_W_ref` | 参考 W 上限，避免本轮样本整体过旧时相对归一化失真 |
| `depth_bias` | 越深的桶越容易被剪 |
| `random_jitter_sigma` | 高斯随机扰动强度 |
| `max_prune_prob` | 单次维护最大剪枝概率上限 |

不建议直接使用：

```text
prune_prob = 1 - avg_W / W_max
```

原因是如果本轮扫描的样本整体都很旧，纯相对归一化会让遗忘概率过度依赖批次分布。

## 10. 二次筛选

设计上，剪下来的分支不能直接全部标记遗忘，必须做二次筛选。

当前 MVP 的硬保护已在剪枝前完成。后续如果加入更细的二次筛选，建议从剪枝候选中移除：

- 硬保护条件命中的记忆
- `min_retention_days` 内的新记忆
- 高 `recall_count` 的记忆
- 最近有效召回的记录
- 高 `emotion_intensity` 的记忆
- `memory_type` 为 `profile` / `capability` / `safety` / `active_task` 的记忆

二次筛选仍然只使用 metadata，不使用语义内容。

## 11. 逻辑遗忘

逻辑遗忘写入 lifecycle store：

```json
{
  "forgotten": true,
  "forgotten_at": 1710000000.0,
  "deleted_after": 1710604800.0,
  "forget_epoch": 1
}
```

当前实现还会把更新后的 lifecycle record 同步回 Mem0/Qdrant payload，确保真实向量库中也能通过 metadata filter 区分，例如：

```python
filters={
  "user_id": user_id,
  "memory_level": "L3",
  "forgotten": false
}
```

普通召回行为：

- L2/L3 检索带 `forgotten=false` filter。
- `_filter()` 仍会做 lifecycle fallback 检查。
- L4 不参与随机遗忘，召回时 `include_forgotten=True`。

## 12. 恢复性召回

`forgotten=true` 的记忆不会进入普通召回，因此悔过期内的恢复必须通过独立的 `recovery_search()` 实现。

触发条件：

1. active 召回结果不足。
2. 用户明确询问“还记得”“之前”“上次”“我们聊过”等历史查询。
3. 调试或管理接口显式调用 `recovery_search()`。

恢复流程：

1. 在 `forgotten=true` 且未超过 `deleted_after` 的 L2/L3 中额外检索。
2. 使用 `recovery_threshold` 或显式 `min_score` 过滤。
3. 命中后撤销遗忘：
   - `forgotten = false`
   - `forgotten_at = null`
   - `deleted_after = null`
   - `recall_count += 1`
   - `last_recalled_at = now`
4. 同步 lifecycle metadata 回 Mem0 payload。
5. 返回恢复的记忆，可进入本轮 prompt。

当前默认值：

```yaml
recovery_enabled: true
recovery_threshold: 0.70
recovery_recent_dialog_threshold: null
```

后续需要增加：

- `recovery_rate_limit_per_day` 的实际执行。
- recovery 日志。
- 用户可见的“想起来了”式的行为策略。

## 13. 物理删除

当前 MVP 不执行物理删除。`hard_delete_enabled` 只是预留配置。

设计状态流：

```text
active -> forgotten -> deleted
```

后续可扩展为：

```text
active -> forgotten -> archived -> deleted
```


## 14. 配置

当前 `MemoryConfig` 中与遗忘相关的默认配置：

```yaml
enable_forgetting: false
forgetting_interval_hours: 24
memory_count_trigger: 5000
min_retention_days: 1.0
physical_deletion_delay_days: 7.0

lambda_l2: 0.30
lambda_l3: 0.10
base_weight_l2: 0.70
base_weight_l3: 1.00
alpha_recall: 0.20

mood_coeff_v: 0.10
mood_coeff_a: 0.10
encoding_intensity_coeff: 0.30
encoding_arousal_coeff: 0.20

configured_W_ref: 1.50
max_prune_prob: 0.80
depth_bias: 0.05
random_jitter_sigma: 0.10
random_seed: null

recovery_enabled: true
recovery_threshold: 0.70
recovery_recent_dialog_threshold: null
recovery_rate_limit_per_day: 10

hard_delete_enabled: false
```

默认策略：

- `enable_forgetting = false`
- `hard_delete_enabled = false`

只有测试 dry-run 和维护日志验证通过后，才应该开启真正的 `forgotten=true` 标记。物理删除必须单独评审和开关控制。

## 15. 召回阈值联动

引入遗忘后，普通召回会被逻辑遗忘进一步降低召回率，因此向量相关性阈值不能继续只用一个全局值。

当前阈值系统按层级和类型分流：

```yaml
relevance_threshold: 0.50
memory_recall_overfetch: 3

l2_relevance_threshold: 0.40
l2_history_threshold: 0.30

l3_recent_dialog_threshold: 0.35
l3_recent_dialog_history_threshold: 0.20
l3_fact_threshold: 0.50
l3_visual_threshold: 0.50
l3_default_threshold: 0.50

l4_relevance_threshold: 0.70
recovery_threshold: 0.70
```

设计理由：

- L2/L3 的短中程记忆需要更高召回率。
- L3 的 `recent_dialog` 是原文片段，阈值可以低，但必须控制注入占比。
- L3 的 `fact` 和 `visual` 仍保持较高阈值，避免噪声污染上下文。
- L4 是稳定基线，阈值更严格。
- recovery 使用独立阈值，避免 forgotten 记忆在普通召回中泄漏。

详细标定方法见 [`memory_recall_thresholds.md`](./memory_recall_thresholds.md)。

## 16. 实现阶段

### 第一阶段：metadata 与召回过滤

已完成：

- 为 L2/L3/L4 写入 `memory_level`、`memory_type`、`created_at`、`lifecycle_created_at` 等 metadata。
- 使用 `infer=False` 写入 Mem0，保持一条系统记忆对应一条 Mem0 record。
- 记录 `mem0_id` 并同步 lifecycle metadata 到真实 Mem0 payload。
- 普通召回默认过滤 `forgotten=false`。
- 最终进入 prompt 的 L2/L3 记忆更新 `recall_count` 和 `last_recalled_at`。

### 第二阶段：遗忘决策树 dry-run 验证

已完成：

- `ForgettingEngine.retention_weight()`。
- 桶化临时生命周期树。
- `ForgettingEngine.build_forgetting_plan()` 决策。
- 维护摘要和 per-memory decisions。

### 第三阶段：逻辑遗忘

已完成：

- `MemoryLifecycleStore.mark_forgotten()`。
- `run_forgetting_maintenance()`。
- Mem0 payload 同步。
- 普通召回跳过 forgotten 记忆。
- `recovery_search()` 恢复 forgotten 记忆。

### 第四阶段：延迟物理删除

未完成：

- `deleted_after` 到期扫描。
- 向量库 hard delete。
- deletion audit log。
- L2/L3 分级删除策略。

## 17. 验证方法

当前测试：

```powershell
python -B -m unittest project_src/src/tests/test_memory_manager_flush.py -v
python -B -m unittest project_src/src/tests/test_memory_forgetting_distribution.py -v
python -B -m unittest project_src/src/tests/test_memory_recall_quality_benchmark.py -v
```

真实 Mem0 集成测试：

```powershell
$env:RUN_MEM0_INTEGRATION='1'
.\Scripts\python.exe -B -m unittest project_src/src/tests/test_memory_forgetting_mem0_integration.py -v
```

测试覆盖：

- metadata 写入和 `mem0_id` 回写。
- `recall_count` / `last_recalled_at` 同步。
- forgotten 过滤。
- 逻辑遗忘同步到 Mem0 payload。
- `recovery_search()` 恢复 forgotten 记忆。
- 遗忘率随时间区间单调上升。
- recall quality benchmark 和 threshold sweep。

## 18. 维护日志

每次扫描建议记录：

```text
scan_id
started_at
finished_at
user_id
candidate_count_l2
candidate_count_l3
current_valence
current_arousal
selected_forget_count
forgotten_count
recovered_count
deleted_count
avg_weight
min_weight
max_weight
random_seed
```

每条候选建议记录：

```text
memory_id
memory_hash
memory_level
memory_type
created_at
lifecycle_created_at
last_recalled_at
recall_count
retention_weight
forget_pressure
tree_depth
prune_prob
secondary_filter_result
final_action
```

这些日志用于调参、回滚和分析误伤。

## 19. 后续工作

优先级较高：

1. 增加真实 Mem0 样本的 recall quality benchmark。
2. 增加 recovery rate limit。
3. 增加最近召回和高情绪强度的 hard-keep。
4. 增加 maintenance lock，避免多客户端同时扫描。
5. 增加物理删除前的 deletion audit log。

优先级中等：

1. root-oriented kNN tree。
2. L2/L3 不同的 `deleted_after` 策略。
3. forgotten 记忆的管理查询接口。
4. 用户反馈驱动的误伤恢复。

暂不建议：

- 在遗忘决策中引入语义 embedding。
- 默认开启 hard delete。
- 让当前情绪状态强支配单条历史记忆的遗忘概率。