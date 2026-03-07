# XiaoJu System

小橘 AI 伙伴系统 — BERT 情绪感知 + 分级记忆 + LLM 人格对话 + 语音交互

小橘是一个由 Orange 创作的 AI 伙伴，具有独立人格、情绪感知和长期记忆能力。
可用于 QQ 机器人私聊/群聊、个人博客看板娘等场景。

## 系统架构

```
                  外部输入
                    |
        +-----------+-----------+
        v           v           v
    用户消息    群聊消息流    语音输入(SenseVoice)
        |           |           |
        +-----------+-----------+
                    v
             +-----------+
             | 事件队列  |   thread-safe Queue
             +-----+-----+
                   |
    Agent 循环 (独立线程, 持续运行)
    +-------------------------------+
    |              v                |
    |   +---------------------+    |
    |   |  collect_events()   |    |
    |   |  . 队列中的消息     |    |
    |   |  . 空闲计时器       |    |
    |   |  . 时间段变化       |    |
    |   +----------+----------+    |
    |              v               |
    |   +---------------------+    |
    |   |  decide()           |    |
    |   |  . 有消息 -> chat() |    |
    |   |  . 空闲超时 -> 主动说|    |
    |   |  . 都没有 -> 继续等  |    |
    |   +----------+----------+    |
    |              v               |
    |       output_callback(msg)   |
    |       (CLI / QQ / TTS / ...) |
    +-------------------------------+
                   |
                   v
+----------------------------------------------------------------------+
|                        XiaoJu Pipeline                               |
|                                                                      |
|  +--------------+   +--------------------------------------------+  |
|  |  BERT 小模型  |   |          分级记忆系统                       |  |
|  |  (~15ms GPU)  |   |                                            |  |
|  |              |   |  L1 工作记忆    原文 messages（保护区不压缩）  |  |
|  |  情绪分类     |   |       | 压缩触发                            |  |
|  |  .emotion     |   |  L2 会话摘要    结构化状态提取               |  |
|  |  .intensity   |   |       | close_session                      |  |
|  |  .confidence  |   |  L3 长期记忆    Mem0 + Qdrant 向量持久化    |  |
|  |              |   |                                            |  |
|  +------+-------+   |  L4 知识库      用户画像 / Agent 结果        |  |
|         |           +--------------------+-----+-----------------+  |
|         v                                v     |                    |
|  +-----------------------------------------------------+           |
|  |                    System Prompt 组装                  |          |
|  |                                                       |          |
|  |  <persona> 人格描述 </persona>                        |          |
|  |  <current_time> 时间感知 </current_time>              |          |
|  |  <emotion_context> BERT 置信度门控指令 </emotion_context>        |
|  |  <memory> L2/L3/L4 召回上下文 </memory>               |          |
|  |  + L1 原文 messages 数组                              |          |
|  +----------------------------+--------------------------+          |
|                               |                                     |
|                       +-------+-------+                             |
|                       |  LLM 路由     |                             |
|          +------------+---+---+-------+-----+                       |
|          v                v                 v                       |
|  +---------------+  +----------+   +--------------+                 |
|  | 主 LLM        |  | 副 LLM   |   | 图片 LLM     |                |
|  | (Claude)      |  | (DeepSeek)|   | (GLM-4V)    |                |
|  | 私聊 always   |  | 群聊默认  |   | 有图片时优先  |                |
|  +-------+-------+  +----+-----+   +------+-------+                |
|          +-------+--------+--------+-------+                        |
|                  v                                                   |
|              文本回复                                                |
|                  |                                                   |
|                  v                                                   |
|  +-------------------------------+                                  |
|  |  语音合成 (CosyVoice, 可选)  |                                  |
|  |  . zero-shot 音色克隆         |                                  |
|  |  . 情绪参考音频切换           |                                  |
|  |  . 优先级队列 + 播放控制      |                                  |
|  +-------------------------------+                                  |
+----------------------------------------------------------------------+
```

### BERT 置信度门控

BERT 的情绪分类结果不直接注入 LLM，而是经过三级置信度门控：

```
effective_confidence = BERT_prob * emotion_reliability[label]

 > strong (0.5)  ->  确定性指令注入："用户心情不错，自然地一起开心就好"
 > weak   (0.3)  ->  不确定注入："用户可能心情不错，但不确定，结合上下文判断"
 <= weak         ->  跳过，让 LLM 自行判断
```

每个情绪标签有独立的可靠度权重（基于验证集 F1），可在 `config.json` 的 `emotion_reliability` 中调整。

### 双 LLM 群聊路由

通过 `ChatMode` 控制 LLM 选择策略，私聊保质量、群聊省费用：

| 模式 | 默认 LLM | 说明 |
|---|---|---|
| `PRIVATE` | 主 LLM (Claude) | 私聊 / CLI，始终用高质量模型 |
| `GROUP` | 副 LLM (DeepSeek) | 群聊默认用廉价模型，BERT 路由决定是否升级 |

群聊升级到主 LLM 的条件：
- 被 @ 提及（用户直接对话，期望高质量回复）
- BERT 检测到强情绪且高置信度（sadness / fear / anger / tenderness，需要细腻情感处理）

```python
# 私聊（默认）
result = pipeline.chat(msg)

# 群聊（QQ 机器人接口）
result = pipeline.chat(msg, is_mentioned=is_at, chat_mode=ChatMode.GROUP)
```

### 自适应 max_tokens

根据 BERT 情绪类型动态调整 LLM 的 `max_tokens`，平衡回复质量和 API 费用：

| 情绪 | 权重 | 实际 tokens (base=10000) | 理由 |
|---|---|---|---|
| neutral | 0.6 | 6,000 | 日常闲聊简短 |
| curiosity | 1.4 | 14,000 | 好奇心需要详细回答 |
| sadness | 1.2 | 12,000 | 安慰需要更多话 |
| joy | 0.8 | 8,000 | 自然回应 |

高强度情绪（intensity >= 0.7）额外 +0.15 权重。

### Agent 事件循环

将小橘从纯请求-响应升级为持续运行的 Agent。用户输入只是事件队列中的一种事件，小橘可以根据空闲时间等条件主动发言。

**主动性档位** (`proactive_level`)：

| 档位 | 行为 | 适用场景 |
|---|---|---|
| `off` | 纯被动，只处理消息队列 | 默认，与旧模式兼容 |
| `low` | 仅响应外部系统事件（群消息流、工具结果等） | QQ 机器人接入 |
| `medium` | 系统事件 + 空闲兜底（超过阈值未交互则主动搭话） | 私聊陪伴 |

**时间感知**：启用 `time_awareness` 后，system prompt 自动注入 `<current_time>` 标签，小橘能感知当前时间（凌晨、深夜等）。

```python
from src.agent_loop import AgentLoop, AgentEvent

loop = AgentLoop(pipeline, agent_config, output_callback=print)
loop.start()

# 用户消息 -> 事件队列
loop.push(AgentEvent(type="message", content="你好"))

# 系统事件 -> 事件队列（low/medium 档位响应）
loop.push(AgentEvent(type="system", content="群里有人在讨论 AI"))

loop.stop()  # 写入长期记忆后退出
```

### 语音交互

**TTS（CosyVoice）**：支持 zero-shot 音色克隆，使用 3-10 秒参考音频即可复刻声音。可根据 BERT 情绪标签切换不同情绪的参考音频。

**ASR（SenseVoice）**：阿里开源语音识别模型，中文识别最强，自带语音情感识别（SER），可与 BERT 情绪系统交叉验证。

```python
# TTS 语音合成
from src.audio_pipeline import AudioPipeline

audio = AudioPipeline(config.audio)
await audio.speak("你好呀！", emotion="joy")

# ASR 语音识别
from src.speech_recognition import SenseVoiceASR

asr = SenseVoiceASR(config.sensevoice)
result = asr.transcribe("audio.wav")
# result.text = "你好呀"
# result.emotion = "happy"  # 语音情感标签
```

## 项目结构

```
neuro_like_system/
+-- config.json                # 主配置（API、人格、记忆、情绪映射、音频）
+-- configs/
|   +-- __init__.py
|   +-- config_loader.py       # config.json -> AppConfig 加载器
|   +-- model_config.py        # 数据类定义（LLMConfig, MemoryConfig 等）
|
+-- models/
|   +-- joint_model.py         # 联合模型（BERT 情绪+行为+语气+强度）
|   +-- emotion_model.py       # 独立情绪模型
|   +-- behavior_model.py      # 独立行为模型
|   +-- annotation_model.py    # 标注模型（知识蒸馏）
|
+-- src/
|   +-- inference_pipeline.py  # 核心：LLMClient + NeuroLikePipeline
|   +-- agent_loop.py          # Agent 事件循环（AgentLoop + AgentEvent）
|   +-- memory_manager.py      # 分级记忆管理器（L1-L4, Mem0, Qdrant）
|   +-- audio_pipeline.py      # TTS 语音合成管道（CosyVoice）
|   +-- speech_recognition.py  # ASR 语音识别（SenseVoice）
|   +-- qq_adapter.py          # QQ 机器人适配层（OneBot v11 WebSocket）
|   +-- image_utils.py         # 图片处理工具（下载/验证/缩放/base64/缓存）
|   +-- emotion_fusion.py      # BERT + LLM 双信号情绪融合
|   +-- emotion_state.py       # 情绪状态机（valence-arousal 二维连续状态）
|   +-- proactive_decision.py  # 主动决策模块
|   +-- logger.py              # loguru 日志配置
|   +-- content_filter.py      # 内容过滤器（敏感词清洗）
|   +-- quick_test.py          # 测试脚本（多模式）
|   +-- run_qq_bot.py          # QQ 机器人启动入口
|   +-- export_onnx.py         # ONNX 导出
|
+-- data/
|   +-- dataset.py             # PyTorch Dataset 定义
|   +-- qdrant_db/             # Qdrant 向量数据库（L3 持久化）
|   +-- audio_refs/            # TTS 参考音频
|   +-- image_cache/           # 图片缓存
|
+-- checkpoints/
|   +-- joint_model/best.pt    # BERT 联合模型检查点
|
+-- docs/                      # 设计文档
+-- logs/                      # 运行日志
+-- requirements.txt
```

## 配置

所有配置集中在 `config.json`，通过 `AppConfig.load()` 加载：

```jsonc
{
  "llm": {
    "provider": "anthropic",       // 主 LLM：私聊 + 群聊升级
    "model": "claude-sonnet-4-6",
    "api_key": "...",
    "base_url": "https://...",     // 支持第三方代理
    "max_tokens": 10000,           // 自适应 max_tokens 的基准值
    "temperature": 0.8
  },

  "llm_secondary": {
    "provider": "deepseek",        // 副 LLM：群聊默认（廉价模型）
    "model": "deepseek-chat",
    "api_key": "...",
    "base_url": "https://api.deepseek.com/v1",
    "max_tokens": 10000,
    "temperature": 0.8
  },

  "llm_vision": {
    "provider": "custom",          // 图片 LLM：智谱 GLM-4V
    "model": "glm-4.6v",
    "api_key": "...",
    "base_url": "https://open.bigmodel.cn/api/paas/v4/"
  },

  "personality": {
    "name": "小橘",
    "traits": ["活泼", "好奇", "善良"],
    "description": "人格描述（直接写入 system prompt）..."
    // Big Five 参数：openness, extraversion, humor_tendency 等
  },

  "memory": {
    "vector_store_path": "./data/qdrant_db",
    "context_window_tokens": 400000,
    "compression_threshold": 0.75,
    "relevance_threshold": 0.3
  },

  "audio": {
    "enabled": true,
    "tts_provider": "cosyvoice",
    "cosyvoice": {
      "model_dir": "./models/CosyVoice2-0.5B",
      "ref_audio_dir": "./data/audio_refs",
      "default_ref_audio": "default.wav",
      "default_ref_text": "你好，我是小橘，很高兴认识你。",
      "sample_rate": 22050,
      "speed": 1.0
    },
    "emotion_ref_map": {
      "joy": { "audio": "happy.wav", "text": "..." },
      "sadness": { "audio": "sad.wav", "text": "..." }
    }
  },

  "sensevoice": {
    "enabled": true,
    "model_id": "FunAudioLLM/SenseVoiceSmall",
    "device": "cuda",
    "language": "zh",
    "use_emotion": true
  },

  "agent": {
    "proactive_level": "off",
    "idle_threshold_seconds": 300,
    "time_awareness": true
  }
}
```

## 快速开始

### 环境准备

```bash
# 安装基础依赖
pip install -r requirements.txt

# 安装语音模块（可选）
pip install cosyvoice funasr sensevoice
```

### 运行测试

```bash
cd neuro_like_system

# 1. 测试 API 连通性
python src/quick_test.py --ping

# 2. 纯 LLM 对话测试（无 BERT）
python src/quick_test.py --msg "你好|||最近在干嘛|||聊聊AI吧"

# 3. BERT + LLM 联合测试
python src/quick_test.py --bert-test

# 4. 记忆系统测试
python src/quick_test.py --memory-test --compress-at 300 --protected-turns 2

# 5. Agent 事件循环测试（交互式，输入 quit 退出）
python src/quick_test.py --agent-test
```

### QQ 机器人

```bash
# 启动（NapCat 反向 WS 连接 ws://localhost:8080/xm）
python run_qq_bot.py
```

### 交互式对话

```bash
python src/inference_pipeline.py --config config.json
```

## 记忆系统

四层分级记忆架构：

| 层级 | 名称 | 存储 | 生命周期 | 用途 |
|---|---|---|---|---|
| L1 | 工作记忆 | 原文 messages 数组 | 当前窗口 | LLM 上下文输入 |
| L2 | 会话摘要 | 结构化 JSON 状态 | 单次会话 | L1 压缩后的状态快照 |
| L3 | 长期记忆 | Mem0 + Qdrant 向量 | 跨会话持久化 | 语义搜索召回 |
| L4 | 知识库 | Qdrant 向量 | 永久 | 用户画像、Agent 结果 |

关键机制：
- **保护区**：最近 N 轮（默认 12）原文永不压缩，保证对话连贯性
- **压缩策略**：结构化状态提取（话题/事实/情绪/待处理），而非叙述性摘要
- **跨 session 召回**：`close_session()` 时保护区原文写入 L3，新 session 通过语义搜索召回
- **自然提示**：语义搜索无结果但 L3 有记录时，注入 `<has_memory>` 标签让 LLM 自然提及

## BERT 模型

### 联合模型 (joint_model.py)

- 基座：`hfl/chinese-roberta-wwm-ext` (~100M 参数)
- 输出头：情绪分类（10 类） + 行为分类 + 语气分类 + 强度回归
- 推理速度：~15ms/条 (GPU)
- 当前状态：情绪头可用（68% 准确率），行为/语气头未专项训练

### 情绪标签 (10 类)

| ID | 标签 | 可靠度 | 说明 |
|---|---|---|---|
| 0 | joy | 0.66 | 喜悦 |
| 1 | sadness | 0.66 | 悲伤 |
| 2 | anger | 0.45 | 愤怒（可靠度低） |
| 3 | fear | 0.69 | 恐惧 |
| 4 | surprise | 0.62 | 惊讶 |
| 5 | disgust | 0.58 | 厌恶 |
| 6 | neutral | 0.76 | 中性（最可靠） |
| 7 | excitement | 0.62 | 兴奋 |
| 8 | tenderness | 0.68 | 温柔 |
| 9 | curiosity | 0.74 | 好奇 |

## 已知问题与技巧

### Anthropic 代理 401 双 header 冲突

环境变量 `ANTHROPIC_AUTH_TOKEN` 会让 SDK 同时发送 `x-api-key` 和 `Authorization` header，导致代理报 401。已通过 httpx event hook 在请求发出前删除 `authorization` header 解决。

### Qdrant Windows 锁文件

跨 session 测试时需在创建新 session 前关闭 Qdrant 客户端，否则 `.lock` 文件被占用。`NeuroLikePipeline.close()` 已处理。

## 硬件要求

| 配置 | GPU | RAM | 说明 |
|---|---|---|---|
| 最低 | 8GB VRAM (RTX 3060/4060) | 16GB | `--low_vram` 模式训练 |
| 推荐 | 12GB+ VRAM | 32GB | 标准训练 + CosyVoice |
| 纯推理 | CPU 可用 | 8GB+ | BERT ~15ms/条，LLM 依赖 API |

## 远期规划

- [x] Agent 事件循环（主动发言、时间感知）
- [x] QQ 机器人接入（群聊注意力判断）
- [x] 图片识别（GLM-4V 多模态）
- [ ] CosyVoice TTS 语音合成
- [ ] SenseVoice ASR 语音识别
- [ ] Live2D 渲染 + 口型同步
- [ ] behavior/tone 标签补全并重新训练
- [ ] 个人博客看板娘前端
- [ ] 本地 LLM 替代 API（Qwen2-7B 蒸馏）

## License

MIT License

## 致谢

本项目灵感来源于 Neuro-sama，感谢 Vedal 的创新工作。
