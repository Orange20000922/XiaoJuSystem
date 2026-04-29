
# 记忆召回阈值与 Benchmark 标定

## 1. 背景

早期记忆召回只用一个全局 `relevance_threshold`。引入遗忘机制后，这个设计就太粗糙了：

- 逻辑遗忘会让 L2/L3 的可召回集合变少。
- 短中程对话本来就容易因为向量相似度不够而召回失败。
- L3 同时装了事实、视觉摘要和一部分原文 `recent_dialog`，不能都用同一个阈值。
- L4 存的是长期画像、能力描述和稳定知识，应该比 L2/L3 更严格。
- recovery 需要独立的阈值，不能让 forgotten 记忆在普通召回里漏出来。

所以现在按层级、类型和是否只是要求恢复旧对话历史分流。

## 2. 当前推荐阈值

默认值在 `MemoryConfig` 里：

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
recovery_recent_dialog_threshold: null
```

含义：

| 配置 | 默认值 | 用途 |
|---|---:|---|
| `relevance_threshold` | `0.50` | legacy fallback，给缺少 lifecycle metadata 的旧记录或未知层级用 |
| `l2_relevance_threshold` | `0.40` | 普通 L2 状态快照召回 |
| `l2_history_threshold` | `0.30` | 历史类查询下的 L2 召回 |
| `l3_recent_dialog_threshold` | `0.35` | 普通 L3 原文对话片段 |
| `l3_recent_dialog_history_threshold` | `0.20` | 历史类查询下的 L3 原文对话片段 |
| `l3_fact_threshold` | `0.50` | L3 事实记忆 |
| `l3_visual_threshold` | `0.50` | L3 视觉记忆和视觉摘要 |
| `l3_default_threshold` | `0.50` | 未知 L3 类型 |
| `l4_relevance_threshold` | `0.70` | L4 知识、画像、能力基线 |
| `recovery_threshold` | `0.70` | forgotten 记忆的恢复性召回 |
| `recovery_recent_dialog_threshold` | `null` | 可选，覆盖 recovery 中 `recent_dialog` 的阈值 |

## 3. 召回路径

### 3.1 普通召回

`get_system_context()` 对 L2/L3/L4 并行搜索：

- L2：`user_id + run_id + memory_level="L2" + forgotten=false`
- L3：`user_id + memory_level="L3" + forgotten=false`
- L4：`user_id + agent_id="neuro_agent" + memory_level="L4"`

搜索返回后统一走 `_filter()`：

1. 合并 lifecycle metadata。
2. 确认 memory level。
3. 普通 L2/L3 过滤掉 `forgotten=true`。
4. 调用 `_score_threshold_for()` 选择动态阈值。
5. 过滤低分候选。

最终进入 `<history>` 的 L2/L3 记忆才会更新：

- `recall_count += 1`
- `last_recalled_at = now`

### 3.2 历史查询

`_should_attempt_recovery()` 会识别以下中文触发词：

```text
还记得 记得 之前 以前 上次 刚才 我们聊过 说过
```

命中后：

- L2 使用 `l2_history_threshold`。
- L3 的 `recent_dialog` 使用 `l3_recent_dialog_history_threshold`。
- 如果 active 召回不足，或者查询明显是历史类，会额外尝试 `recovery_search()`。

### 3.3 Recovery

`recovery_search()` 只搜 forgotten 的 L2/L3：

```text
memory_level in {"L2", "L3"}
forgotten = true
deleted_after > now
score >= recovery_threshold
```

命中后撤销遗忘：

- `forgotten = false`
- `forgotten_at = null`
- `deleted_after = null`
- `recall_count += 1`
- `last_recalled_at = now`

恢复后的 lifecycle metadata 会同步回 Mem0/Qdrant payload。

## 4. Benchmark 指标

小型的 benchmark 在：

```text
project_src/src/tests/test_memory_recall_quality_benchmark.py
```

它用 query-aware fake Mem0 后端，按 query 为每条合成记忆设置不同的 score，避免“搜索总是返回全部记录”那种假阳性。

指标：

| 指标 | 含义 | 当前约束 |
|---|---:|---|
| `recall_at_k` | L2/L3 在 prompt 注入范围内的平均召回率 | `>= 0.95` |
| `noise_rate` | 进入 `<history>` 的无关记忆占比 | `<= 0.05` |
| `forgotten_leakage_rate` | 普通召回中 forgotten 记忆泄漏率 | `== 0.0` |
| `recovery_precision` | recovery 结果中相关记忆的占比 | `>= 0.95` |
| `recovery_recall` | 应该恢复的 forgotten 记忆是否被恢复 | `>= 0.95` |
| `l3_recent_dialog_injection_ratio` | L3 原文对话在 history 注入中的占比 | `<= 0.45` |

## 5. Synthetic 数据集

Benchmark 覆盖了这些类型：

- L2 状态快照正例
- L3 fact 正例
- L3 visual 正例
- L3 `recent_dialog` 正例
- L4 knowledge 正例
- 普通不相关的 fact/dialog 噪声
- 贴近阈值边界的 L2 噪声
- 贴近阈值边界的 L3 fact 噪声
- 贴近阈值边界的 L3 visual 噪声
- 贴近阈值边界的 L3 `recent_dialog` 噪声
- forgotten 高分相关项
- forgotten 接近 recovery 阈值但不相关的噪声项

加边界噪声很重要，否则 sweep 会倾向于无限降低阈值只追召回，而暴露不出噪声污染。

## 6. Threshold Sweep

### 6.1 快速 sweep

默认单测跑快速 sweep：

```text
2^8 = 256 次
```

搜索空间：

```yaml
l2_relevance_threshold: [0.35, 0.40]
l2_history_threshold: [0.25, 0.30]
l3_recent_dialog_threshold: [0.30, 0.35]
l3_recent_dialog_history_threshold: [0.15, 0.20]
l3_fact_threshold: [0.45, 0.50]
l3_visual_threshold: [0.45, 0.50]
l4_relevance_threshold: [0.65, 0.70]
recovery_threshold: [0.65, 0.70]
```

快速 sweep 适合日常回归，运行时间约 `0.5s`。

### 6.2 完整 sweep

完整 sweep 需要显式设置环境变量：

```text
3^8 = 6561 次
```

搜索空间：

```yaml
l2_relevance_threshold: [0.35, 0.40, 0.45]
l2_history_threshold: [0.25, 0.30, 0.35]
l3_recent_dialog_threshold: [0.30, 0.35, 0.40]
l3_recent_dialog_history_threshold: [0.15, 0.20, 0.25]
l3_fact_threshold: [0.45, 0.50, 0.55]
l3_visual_threshold: [0.45, 0.50, 0.55]
l4_relevance_threshold: [0.65, 0.70, 0.75]
recovery_threshold: [0.65, 0.70, 0.75]
```

完整 sweep 适合手动标定，运行时间约 `12s`。

## 7. 排序策略

每组阈值先过硬约束：

```text
recall_at_k >= 0.95
noise_rate <= 0.05
forgotten_leakage_rate == 0.0
recovery_precision >= 0.95
recovery_recall >= 0.95
l3_recent_dialog_injection_ratio <= 0.45
```

通过硬约束后再排序：

1. `recall_at_k` 越高越好
2. `recovery_recall` 越高越好
3. `recovery_precision` 越高越好
4. `noise_rate` 越低越好
5. `forgotten_leakage_rate` 越低越好
6. `l3_recent_dialog_injection_ratio` 越接近目标值 `0.25` 越好
7. 越接近当前推荐阈值越好（避免无意义漂移）
8. 同等条件下稍微偏保守

排序函数在测试文件里叫 `_threshold_score()`。

## 8. 当前标定结果

快速 sweep：

```text
total_runs: 256
accepted_runs: 2
best: 当前推荐阈值
```

完整 sweep：

```text
total_runs: 6561
accepted_runs: 24
best: 当前推荐阈值
```

第一推荐结果：

```json
{
  "recall_at_k": 1.0,
  "noise_rate": 0.0,
  "forgotten_leakage_rate": 0.0,
  "recovery_precision": 1.0,
  "recovery_recall": 1.0,
  "l3_recent_dialog_injection_ratio": 0.25,
  "thresholds": {
    "l2_relevance_threshold": 0.4,
    "l2_history_threshold": 0.3,
    "l3_recent_dialog_threshold": 0.35,
    "l3_recent_dialog_history_threshold": 0.2,
    "l3_fact_threshold": 0.5,
    "l3_visual_threshold": 0.5,
    "l4_relevance_threshold": 0.7,
    "recovery_threshold": 0.7
  }
}
```

结论：当前默认阈值在 synthetic benchmark 上是第一推荐的组合。

## 9. 运行命令

普通 benchmark + 快速 sweep：

```powershell
python -B -m unittest project_src/src/tests/test_memory_recall_quality_benchmark.py -v
```

打印 benchmark 和 sweep 报告：

```powershell
$env:PRINT_RECALL_BENCHMARK='1'
$env:PRINT_THRESHOLD_SWEEP='1'
python -B -m unittest project_src/src/tests/test_memory_recall_quality_benchmark.py -v
```

完整 sweep：

```powershell
$env:RUN_FULL_THRESHOLD_SWEEP='1'
$env:PRINT_THRESHOLD_SWEEP='1'
python -B -m unittest project_src/src/tests/test_memory_recall_quality_benchmark.py -v
```

和遗忘相关测试一起跑：

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

## 10. 局限

当前 benchmark 是 synthetic 的，不等价于真实 Mem0 embedding 分布下的最终最优值。

它的价值在于：

- 给阈值系统提供一个可重复、可回归的工程基线。
- 防止后续改动导致召回率下降。
- 防止 forgotten 记忆泄漏。
- 防止 L3 原文片段过度注入 prompt。
- 防止 recovery 过于激进。

下一步更接近真实效果的做法：

1. 构造真实/半真实的中文对话记忆样本。
2. 用真实 Mem0/Qdrant 跑相同指标。
3. 记录不同模型、不同 embedding 后端下的推荐阈值。
4. 把 synthetic 和 real-Mem0 的推荐结果做对比。
5. 建立 threshold changelog，避免无依据调参。

## 11. 调参原则

优先级：

1. 不能泄漏 forgotten 记忆。
2. recovery precision 不能低。
3. L3 原文注入比例不能失控。
4. 在上述约束下尽量提高 L2/L3 的 recall。
5. L4 保持严格，避免污染长期基线。

不建议的做法：

- 全局移除向量分数过滤。
- 对 L3 所有类型用同一个低阈值。
- 让 `recent_dialog` 和 `fact` 共用阈值。
- 为了追 recovery recall 而过度降低 `recovery_threshold`。

可以考虑的方向：

- 对 `recent_dialog` 增加 prompt 注入数量上限。
- 对历史查询引入更细的 intent 分类。
- 为不同 embedding 模型保留不同的阈值 profile。
- 用真实 Mem0 benchmark 逐步替换 synthetic 的结论。
