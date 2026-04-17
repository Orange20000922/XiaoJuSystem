# 主动性系统实现文档

## 概述

主动性系统实现了两阶段决策架构：
1. **判断层（DeepSeek）**：低成本判断是否需要主动发言，输出结构化意图指导
2. **生成层（Claude）**：基于判断层的指导生成实际回复

## 架构设计

### 状态机

```
NORMAL (消息驱动)
    ↓ (空闲 idle_trigger_hours 小时)
    ↓ 调用决策模块
    ↓ (should_respond=True, confidence > threshold)
WAITING_RESPONSE (等待用户回应)
    ↓ (收到用户消息) → NORMAL
    ↓ (超时 response_wait_minutes 分钟无回应)
DORMANT (休眠，仅响应消息)
    ↓ (收到用户消息) → NORMAL
```

### 输入信号

决策模块接收以下输入：
- **L1 最近对话**：从 `memory.working_memory` 获取最近 N 轮对话
- **情绪状态**：从 `data/emotion_state.json` 读取 valence/arousal/last_emotion
- **L4 情感记忆**：从 Mem0 搜索带情感标签的长期记忆

### 决策输出

```python
@dataclass
class ProactiveDecision:
    should_respond: bool    # 是否应该主动发言
    confidence: float       # 置信度 (0.0-1.0)
    intent: str            # 意图："关心对方" | "分享想法" | "调节气氛" | "继续话题"
    topic_hint: str        # 话题提示："可以聊聊刚才提到的..."
    tone: str              # 建议语气："warm" | "playful" | "calm" | "supportive"
    reason: str            # 决策理由（调试用）
```

## 文件结构

### 新增文件

- `src/proactive_decision.py` - 主动决策模块
- `test_proactive.py` - 完整系统测试
- `test_proactive_decision.py` - 决策模块单元测试

### 修改文件

- `configs/model_config.py` - 添加 `ProactiveConfig` 配置类
- `configs/config_loader.py` - 添加 `load_proactive_config()` 加载函数
- `config.json` - 添加 `proactive` 配置段
- `src/agent_loop.py` - 集成决策模块，实现状态机
- `src/inference_pipeline.py` - `generate_proactive()` 接受 `decision_hint` 参数

## 配置说明

### config.json

```json
{
  "proactive": {
    "enabled": true,                      // 是否启用主动决策模块
    "decision_provider": "deepseek",      // 判断层 LLM 提供商
    "decision_model": "deepseek-chat",    // 判断层模型
    "decision_temperature": 0.3,          // 判断层温度
    "decision_timeout": 5.0,              // 判断层超时（秒）
    "confidence_threshold": 0.6,          // 置信度阈值
    "recent_turns_limit": 8,              // 从 L1 取最近 N 轮
    "l4_memory_limit": 3,                 // 从 L4 取 N 条记忆
    "idle_trigger_hours": 2.0,            // 空闲 N 小时后触发
    "response_wait_minutes": 30,          // 主动发言后等待回应时间
    "min_interval_seconds": 30            // 冷却时间（秒）
  }
}
```

### 配置参数说明

- **enabled**: 主开关，false 时回退到原有的简单空闲触发
- **confidence_threshold**: 决策置信度低于此值时不发言
- **idle_trigger_hours**: 空闲多久后触发决策（可设为小值用于测试）
- **response_wait_minutes**: 主动发言后等待用户回应的时间，超时进入 DORMANT 状态
- **min_interval_seconds**: 两次主动发言之间的最小间隔

## 使用方法

### 1. 启动 Agent Loop

```python
from src.agent_loop import AgentLoop, AgentEvent
from src.inference_pipeline import NeuroLikePipeline
from configs.config_loader import AppConfig

config = AppConfig.load("config.json")
pipeline = NeuroLikePipeline(config)

loop = AgentLoop(
    pipeline=pipeline,
    config=config.agent,
    output_callback=print,
    proactive_config=config.proactive,  # 传入主动配置
)

loop.start()

# 推送用户消息
loop.push(AgentEvent(type="message", content="你好"))

# 停止循环
loop.stop()
```

### 2. 运行测试

```bash
# 单元测试（仅测试决策模块）
python test_proactive_decision.py

# 完整系统测试（需要 Claude + DeepSeek API）
python test_proactive.py
```

## 决策逻辑

### Prompt 设计

判断层使用结构化 prompt，要求输出 JSON 格式的决策：

**System Prompt**:
- 说明角色是"主动性判断助手"
- 不生成实际回复，只输出决策
- 定义判断原则（对话自然结束 → 不发言，情绪低落 → 关心对方等）
- 定义 confidence 计算规则

**User Prompt**:
- 格式化最近对话历史
- 格式化情绪状态（valence/arousal/last_emotion）
- 格式化 L4 情感记忆
- 可选：当前时间

### 生成层注入

当决策模块判断需要发言时，将决策指导注入到 Claude 的 system prompt：

```
[主动发言指导]
意图: 关心对方
话题提示: 可以聊聊刚才提到的工作压力
建议语气: warm
```

## 调试与监控

### 日志输出

- `logger.info("空闲 X 小时，触发主动决策")` - 触发时机
- `logger.debug("主动决策: should_respond=... confidence=... intent=...")` - 决策结果
- `logger.info("主动发言成功，进入 WAITING_RESPONSE 状态")` - 状态转换
- `logger.info("收到用户消息，状态从 X 重置为 normal")` - 状态重置

### 状态查询

```python
# 查看当前状态
print(loop.proactive_state)  # ProactiveState.NORMAL / WAITING_RESPONSE / DORMANT

# 查看上次主动发言时间
print(loop.last_proactive_time)
```

## 成本控制

- **DeepSeek 判断**：约 0.001 元/次（500 tokens）
- **Claude 生成**：仅在判断层认为需要时调用
- **冷却时间**：`min_interval_seconds` 防止频繁调用

## 已知限制

1. **L4 记忆检索**：当前使用简单关键词搜索，可能需要优化查询策略
2. **emotion_state.json 同步**：需要确保文件在每轮对话后及时更新
3. **状态持久化**：`proactive_state` 在 AgentLoop 重启后会丢失（可选优化）
4. **DeepSeek API 稳定性**：需要处理超时和错误，失败时回退到原有逻辑

## 后续优化方向

1. **决策缓存**：相似上下文下复用决策结果
2. **多级置信度阈值**：confidence > 0.8 立即发言，0.6-0.8 延迟发言
3. **A/B 测试**：对比有/无决策模块的主动发言质量
4. **决策日志分析**：统计 intent/tone 分布，优化 prompt
5. **状态持久化**：将 `proactive_state` 保存到文件，支持跨 session 恢复

## 故障排查

### 问题：决策模块不触发

- 检查 `config.proactive.enabled` 是否为 `true`
- 检查 `idle_trigger_hours` 是否设置过大
- 检查日志是否有 "空闲 X 小时，触发主动决策" 输出

### 问题：决策模块判断不发言

- 检查 `confidence_threshold` 是否设置过高
- 查看日志中的 `confidence` 值
- 检查对话上下文是否明确（空对话历史会导致低置信度）

### 问题：DeepSeek API 调用失败

- 检查 API key 是否正确配置
- 检查网络连接
- 查看日志中的错误信息
- 确认 `decision_timeout` 是否足够

### 问题：主动发言后一直等待

- 检查 `response_wait_minutes` 是否设置过大
- 查看当前状态：`loop.proactive_state`
- 手动推送消息触发状态重置

## 测试场景

### 场景 1：用户情绪低落

```
用户: 我今天心情不太好
Neuro: 怎么了？发生什么事了吗？
用户: 算了不说了
Neuro: 好吧，如果你想聊的话随时找我

[空闲触发]
决策: should_respond=True, intent="关心对方", confidence=0.75
Neuro: [主动] 还好吗？如果不想说也没关系，我陪着你
```

### 场景 2：对话自然结束

```
用户: 今天天气不错
Neuro: 是啊，挺舒服的
用户: 好的，拜拜
Neuro: 拜拜~

[空闲触发]
决策: should_respond=False, reason="对话自然结束"
[不发言]
```

### 场景 3：话题有趣但中断

```
用户: 你知道量子计算吗？
Neuro: 知道一些，你对这个感兴趣？
用户: 嗯，挺有意思的
Neuro: 确实，量子纠缠的概念特别神奇

[空闲触发]
决策: should_respond=True, intent="继续话题", confidence=0.68
Neuro: [主动] 你想了解量子计算的哪方面？算法还是硬件实现？
```

## 版本历史

- **v1.0** (2026-03-02): 初始实现，包含两阶段决策架构和状态机
