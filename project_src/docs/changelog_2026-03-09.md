# 变更记录 2026-03-09

## 概述

本次工作涵盖三个方向：情绪状态机的输出接口优化、系统架构简化（移除 behavior/tone 映射）、运行稳定性加固。同时完成了全系统代码审查，并为"数字学生"比赛场景生成了配置模板。

---

## 一、情绪状态机 — 动态 Prompt Hint

### 背景

`get_prompt_hint()` 原先从静态字典 `STATE_HINT_MAP` 查表返回固定字符串，丢弃了 OU 过程计算出的连续状态信息（强度、轨迹、持续性）。无论对话进行到第 1 轮还是第 10 轮，只要 label 相同，注入的暗示完全一样，导致情绪输出刻板化。

### 改动

**文件**: `src/emotion_state.py`

#### 1. `EmotionState` 新增轨迹追踪字段

```python
prev_valence: float = 0.0       # 上一轮 valence（计算趋势用）
sustained_label: str = "neutral" # 持续停留的情绪标签
sustained_turns: int = 0         # 在该标签区间停留了多少轮
```

#### 2. `update()` 填充轨迹字段

- 更新前记录 `prev_valence = v`
- 更新后比较新旧 label，维护 `sustained_label` 和 `sustained_turns` 计数器

#### 3. `get_prompt_hint()` 重写为动态生成

利用三个维度生成上下文感知的暗示：

- **强度**（deviation from baseline）：`> 0.5` → "（情绪比较强烈）"，`< 0.3` → "（只是淡淡的）"
- **轨迹**（delta_v 方向）：正在好转 / 正在恶化 / 越来越高兴 / 继续低落
- **持续性**（sustained_turns）：`>= 5` → "这种状态已经持续了一段时间"，`== 1` → "刚刚情绪发生了转变"

阈值判断改为**距基线的偏离**（而非距原点），与 OU 不动点一致。

#### 4. `to_dict()` / `from_dict()` 序列化新字段

`prev_valence`, `sustained_label`, `sustained_turns` 加入 JSON 持久化。

### 效果示例

| 场景 | 旧输出（固定） | 新输出（动态） |
|------|---------------|---------------|
| 第 1 轮难过 | "聊了一些沉重的话题..." | "聊了一些沉重的话题...（只是淡淡的）。刚刚情绪发生了转变" |
| 第 5 轮持续难过 | "聊了一些沉重的话题..." | "聊了一些沉重的话题...（情绪比较强烈），情绪还在继续低落。这种状态已经持续了一段时间" |
| 第 6 轮开始好转 | "聊了一些沉重的话题..." | "聊了一些沉重的话题...，不过情绪正在慢慢好转。刚刚情绪发生了转变" |

---

## 二、移除 behavior_map / tone_map

### 背景

BERT Joint Model 的 behavior/tone head 未经专项训练，预测为噪声。`_build_emotion_directives()` 中已注释掉 behavior/tone 注入（551-552 行），但配置文件、数据类、加载器中仍保留了死代码。

经分析，`behavior_map` 的 12 个条目中 4 个是空字符串，其余要么和 `emotion_map` 重复，要么是 LLM 不需要被教的常识。

### 改动

| 文件 | 改动 |
|------|------|
| `config.json` | 删除 `emotion_prompts` 下的 `behavior_map` 和 `tone_map` 段（24 行） |
| `configs/model_config.py` | `EmotionPromptConfig` 删除 `behavior_map` 和 `tone_map` 字段 |
| `configs/config_loader.py` | `load_emotion_prompt_config()` 删除对应加载行 |
| `src/inference_pipeline.py` | `_build_emotion_directives()` 删除读取 `behavior` 和 `tone` 变量的代码 |

### 保留的 behavior 用途

behavior label 仍在以下位置使用（不走 map，直接用 BERT 输出的 label）：

- `memory_manager.py:191` — `turn.behavior == "change_topic"` 触发 L1→L2 压缩边界
- `attention_tracker.py:195` — `behavior in ("ask_question", "seek_clarification")` 判断是否回复
- `ConversationTurn.behavior` / `.tone` — 元数据记录

---

## 三、运行稳定性加固

### 3.1 AgentLoop 主循环顶层异常保护

**文件**: `src/agent_loop.py`

**问题**: `_loop()` 的 while 循环体没有顶层 try-except。如果 `_poll_event()` 之外的代码（如 `_check_proactive_triggers()`、注意力清理）抛出未预料的异常，daemon 线程静默死亡，系统看起来正常但不再响应消息。

**改动**: 在 while 循环内部包一层 catch-all：

```python
while not self._stop_event.is_set():
    try:
        # 原有逻辑
    except Exception as e:
        logger.error(f"Agent 循环异常（已恢复）: {e}", exc_info=True)
```

### 3.2 ThreadPoolExecutor 复用

**文件**: `src/inference_pipeline.py`

**问题**: 情绪状态持久化每 5 轮创建一个新的 `ThreadPoolExecutor(max_workers=1)`，不被关闭。40 分钟演示 = 12 个泄漏的 executor 和线程。

**改动**:

- `__init__` 创建共享实例 `self._bg_executor = ThreadPoolExecutor(max_workers=1)`
- 持久化改用 `self._bg_executor.submit(self._save_emotion_state)`
- `close()` 调用 `self._bg_executor.shutdown(wait=False)` 清理
- `ThreadPoolExecutor` import 提升到文件顶部

---

## 四、全系统代码审查

对项目全部 ~8,000 行生产代码进行了三维评估。

### 评估结论

| 维度 | 评分 | 要点 |
|------|------|------|
| 功能完成度 | 8/10 | 核心闭环完整，语音模块待接入 |
| 架构质量 | 7/10 | 分层清晰、降级优雅，但 Pipeline 是 God Class（1245 行） |
| 创新性 | 8.5/10 | OU 情绪状态机 + EKF 参数辨识在同类项目中几乎独一无二 |

### 关键发现

- **融合系统的隐式保护**: `skip_llm_threshold=0.85` 而 `max(reliability)=0.76`，eff_conf 永远无法达到 0.85，因此 LLM 情绪分类每轮都被调用。BERT 和 LLM 分歧时融合后置信度自动降低，触发置信度门控跳过注入。路径 2（prompt 直接注入）实际已被融合 + 门控保护。
- **BERT 准确度接近贝叶斯上限**: macro-F1=0.64 在 10 分类任务上接近标注者间一致性上限（~0.70）。重新训练边际收益极低，系统级优化（时序平滑、混淆矩阵校正、状态机反向校正）性价比更高。
- **Pipeline God Class**: `NeuroLikePipeline` 1245 行，同时负责 BERT 推理、情绪融合、状态机更新、记忆管理、LLM 路由、prompt 构建、响应生成、注意力判断。建议拆分为 EmotionAnalyzer / PromptBuilder / LLMRouter / ResponseGenerator。

---

## 五、教育场景适配 — 数字学生配置模板

**文件**: `config_example_student.json`

为教育智能体场景生成的配置模板，展示了如何将 OU 情绪状态机应用于课堂模拟：

- 所有配置段的中文注解（用 `_comment` / `_note` 字段）
- 教育场景的 `emotion_map`（课堂情绪 → 学生行为指令）
- 四种学生类型的 Big Five 参数推荐值和 OU 基线建议
- `emotion_state` 参数标注了完整的数学含义

### 技术亮点

- OU 情绪状态机 → Pekrun (2006) 学业情绪模型
- EKF 参数辨识 → 数据驱动的参数估计（非手工调参）
- 动态 Prompt Hint → 连续状态信息不被离散化丢弃
- 同一节课三种学生 → 人格参数驱动差异化情绪动力学

---

## 修改文件清单

| 文件 | 类型 | 描述 |
|------|------|------|
| `src/emotion_state.py` | 修改 | 动态 hint + 轨迹追踪 + 序列化 |
| `src/inference_pipeline.py` | 修改 | 移除 behavior/tone 读取 + 共享 executor |
| `src/agent_loop.py` | 修改 | 主循环顶层异常保护 |
| `configs/model_config.py` | 修改 | 移除 behavior_map/tone_map 字段 |
| `configs/config_loader.py` | 修改 | 移除 behavior_map/tone_map 加载 |
| `config.json` | 修改 | 移除 behavior_map/tone_map 配置段 |
| `config_example_student.json` | 新建 | 数字学生配置模板 |
