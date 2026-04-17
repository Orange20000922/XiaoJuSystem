# 视觉直出响应 + 视觉技能调用

> 2026-04-17

## 背景

原有视觉事件处理流程存在两个瓶颈：

1. **文本描述瓶颈**：GLM-4V 分析图像后输出 JSON 文本描述，再注入主 LLM prompt 生成回复。图像→文本的转换丢失大量视觉信息。
2. **仅有 push 模式**：视觉管线只能主动推送事件，主 LLM 无法按需请求视觉信息。用户问"你看到什么"时系统没有机制响应。

本次更新实现两个新特性解决上述问题。

---

## Feature 1：GLM 直出响应（visual_direct）

### 核心思路

当视觉事件需要即时响应（`route_to_chat=True`）时，让 GLM-4V 直接带着人格 prompt + 关键帧图片生成对话回复，跳过"图像→文本描述→主 LLM"的中间环节。

### 数据流

```
即时路径 (route_to_chat=True, visual_direct=True):

  VisualEvent(keyframes=[img1, img2, ...])
  → AgentEvent(type="visual", images=[img1, img2, ...], content="[视觉事件] ...")
  → AgentLoop._process_chat(visual_direct=True)
    → persona.chat(images=keyframes, visual_direct=True)
      → 跳过 _extract_image_context（不将图片转为文本描述）
      → _route_llm_client: images 存在 → vision LLM (GLM-4V)
      → GLM 收到完整人格 prompt + 关键帧 + 对话历史
      → 直接生成人格化回复
      → BERT 情绪分析回复 → OU 状态机更新（完全复用现有设施）
      → L1 记忆写入
      → output_callback(response)

记忆路径 (route_to_chat=False):

  → persona.handle_visual_event():
    → 情绪注入（apply_stimulus）
    → L4 知识库写入（保留原有行为）
    → L3 用户记忆写入（新增）
```

### 改动文件

| 文件 | 改动 |
|------|------|
| `src/vision/visual_agent.py` | `visual_event_to_agent_event()` 新增 `images=list(event.keyframes)` 传递关键帧；补充 `scene` 字段到 metadata |
| `src/agent/agent_loop.py` | `_process_chat()` 检测 `visual_direct = event.type == "visual" and bool(event.images)`，透传到 `pipeline.chat()` |
| `src/core/persona.py` | `chat()` 新增 `visual_direct` 参数；`visual_direct=True` 时跳过 `_extract_image_context`，保留 images 直接进入 LLM 生成；`handle_visual_event()` 新增 L3 写入路径 |
| `src/memory/memory_manager.py` | 新增 `add_visual_observation_l3()` 方法，将非即时视觉观察写入 L3 用户级记忆 |

### 效果对比（drink_test.avi）

| 模型 | 模式 | 延迟 | 回复 |
|------|------|------|------|
| GLM-4V | 直出+关键帧 | 18.7s | "欸，你刚才举着的是啥呀？看起来好好吃的样子，是不是在吃好吃的？" |
| Claude Opus | 仅文本描述 | 9.1s | "嗯？你刚举起来那个是什么东西呀，我没看太清楚诶。" |
| DeepSeek | 仅文本描述 | 2.0s | "（注意到你的动作）嗯？刚才想给我看什么吗？" |

**结论**：GLM 直出回复质量最高（能识别出具体物品），但延迟也最高。文本描述方案实时性更好，两条路径可并用：直出用于低频高质量即时事件，文本描述用于高频/技能调用场景。

---

## Feature 2：视觉技能调用模式（Visual Skill）

### 核心思路

在对话管线中增加 pull 模式——检测用户消息中的视觉请求意图，按需调用视觉管线获取最近画面分析，注入对话上下文。

### 数据流

```
用户: "你能看看我在做什么吗？"

→ persona.chat(user_input="你能看看我在做什么吗？")
  → BERT + 记忆并行
  → 视觉技能检测: "看看" 匹配 → trigger
  → executor.execute()
    → vision_pipeline.analyze_recent_buffer(top_k=2)
    → 返回最近 2 个高显著度事件的摘要文本
  → user_input = "[视觉感知] 人物坐在书桌前，手握杯子\n你能看看我在做什么吗？"
  → LLM 生成（主 LLM 已有视觉上下文）
  → L1 记忆写入（视觉上下文随对话轮次自然保留）
```

### 新文件

**`src/vision/visual_skill.py`**

- `VisualSkillDetector`：编译 15 个中文关键词正则（看看、看到、画面、摄像头、做什么...）为单一 pattern，`detect(text) -> bool`
- `VisualSkillExecutor`：接受 handler 回调，调用 `analyze_recent_buffer()` 获取最近事件，格式化为分号分隔的摘要文本

### 改动文件

| 文件 | 改动 |
|------|------|
| `src/vision/visual_skill.py` | **新建**，技能检测器 + 执行器 |
| `src/vision/visual_pipeline.py` | 新增公共方法 `analyze_recent_buffer(top_k)` |
| `src/core/persona.py` | 新增 `register_visual_skill(handler)` 注册方法；`chat()` 中 LLM 生成前插入技能检测/注入逻辑；注册时写入 L4 能力描述 |
| `src/vision/__init__.py` | 导出 `VisualSkillDetector`、`VisualSkillExecutor` |

### 集成接线

在启动代码中（如 `run_qq_bot.py`），管线创建后注册技能：

```python
persona.register_visual_skill(
    handler=lambda top_k=2: vision_pipeline.analyze_recent_buffer(top_k=top_k),
)
```

---

## 其他改动

| 文件 | 改动 |
|------|------|
| `configs/config_loader.py` | `AppConfig.load()` 的 `encoding="utf-8"` 改为 `"utf-8-sig"`，兼容 BOM |
| `src/vision/visual_agent.py` | 补充 `scene` 字段在 AgentEvent 序列化/反序列化中的传递（之前遗漏） |
| `src/core/persona.py` | `_visual_event_from_agent_event()` 补充 `scene` 反序列化 |

---

## 单元测试

新增 3 个测试类、13 个测试用例（全部通过，总计 38 个）：

- `TestVisualAgentKeyframePassthrough`：keyframes 传递、空 keyframes、scene 字段验证
- `TestVisualSkillDetector`：7 个关键词匹配/不匹配/空输入用例
- `TestVisualSkillExecutor`：摘要返回、无事件回退、异常处理
