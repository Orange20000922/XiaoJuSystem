# XiaoJu System

小橘 AI 伙伴系统 — BERT 情绪感知 + 分级记忆 + LLM 人格对话 + 动态视觉感知 + 语音交互

小橘是一个由 Orange 创作的 AI 伙伴，具有独立人格、情绪感知和长期记忆能力。
可用于 QQ 机器人私聊/群聊、个人博客看板娘等场景。

## 系统架构

```
                  外部输入
                    |
        +-----------+-----------+-----------+
        v           v           v           v
    用户消息    群聊消息流    语音输入      视频流
        |           |       (SenseVoice)  (摄像头/文件)
        |           |           |           |
        +-----------+-----------+     +-----+-------+
                    v                 |  视觉感知管线 |
             +-----------+           |  (OpenCV +   |
             | 事件队列  |   <----   |   GLM-4V)    |
             +-----+-----+           +-------------+
                   |
    Agent 循环 (独立线程, 持续运行)
    +-------------------------------+
    |              v                |
    |   +---------------------+    |
    |   |  collect_events()   |    |
    |   |  . 队列中的消息     |    |
    |   |  . 视觉事件(push)   |    |
    |   |  . 空闲计时器       |    |
    |   |  . 时间段变化       |    |
    |   +----------+----------+    |
    |              v               |
    |   +---------------------+    |
    |   |  decide()           |    |
    |   |  . 有消息 -> chat() |    |
    |   |  . 视觉事件 -> 直出  |    |
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
|          +-------+-------+                                          |
|          |               |                                          |
|          v               v                                          |
|  +---------------+  +-------------------------------+               |
|  | 视觉技能注入  |  |  语音合成 (CosyVoice, 可选)  |               |
|  | (pull 模式)   |  |  . zero-shot 音色克隆         |               |
|  | 用户问"你看到 |  |  . 情绪参考音频切换           |               |
|  |  什么" → 注入 |  |  . 优先级队列 + 播放控制      |               |
|  | 最近画面分析  |  +-------------------------------+               |
|  +---------------+                                                  |
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

### BERT + LLM 情绪融合

BERT 快速分类（~10ms）与 LLM 高精度分类（~500ms）通过单神经元 softmax 融合：

```
对每个情绪 i:
  z[i] = w_bert * bert_prob[i] * reliability[i] + w_llm * llm_conf[i] + bias
scores = softmax(z)
```

- BERT 和 LLM **一致**时：置信度叠加，指令果断注入
- BERT 和 LLM **分歧**时：softmax 分散概率，置信度降低，门控自动拦截错误指令

默认配置下（`skip_llm_threshold=0.85`，`max(reliability)=0.76`），LLM 情绪分类每轮都被调用，确保最大准确度。可降低阈值跳过高置信 BERT 预测以减少延迟。

### 情绪状态机

系统维护一个 **(valence, arousal) 二维连续情绪状态**，由 Ornstein-Uhlenbeck 随机过程驱动演化，而非简单的离散标签切换。

#### 数学模型

二维耦合 OU 过程，Euler-Maruyama 离散化（Δt = 1 轮）：

```
φ = α - δ             (persistence, 状态记忆强度)
θ = 1 - φ             (mean-reversion rate, 基线回归速率)

v_{t+1} = tanh(φ_v·v + θ_v·μ_v + κ·(a - μ_a) + β·ai_v + γ·user_v + ε_v)
a_{t+1} = tanh(φ·a   + θ·μ_a   + κ·(v - μ_v) + β·ai_a + γ·user_a + ε_a)
```

#### 核心特征

| 特征 | 机制 | 理论基础 |
|------|------|---------|
| **情绪惯性** | φ=0.60 的状态记忆系数 | Kuppens et al. (2010) |
| **均值回归** | θ·μ 拉回人格基线 | Hedonic adaptation / Set point theory |
| **负面偏差** | v < baseline 时 θ 减小为 θ/1.3 | Baumeister et al. (2001) negativity bias |
| **V-A 耦合** | κ·(a-μ_a) 偏差形式 | Lang (1995) 动机维度理论 |
| **自然波动** | ε ~ N(0, 0.05²) 过程噪声 | 情绪的随机性 |
| **有界性** | tanh 非线性 | 防止状态爆炸 |

#### 不动点与稳定性

零输入时不动点恰好等于人格基线 (μ_v, μ_a)。状态转移矩阵 Φ 的特征值 |λ| < 1，保证全局渐近稳定。

#### 参数辨识

gamma=0.25 由 **Extended Kalman Filter (EKF) 最大似然估计**从真实对话数据中辨识。其余参数为理论驱动的设计参数，配合 persona prior 正则化确保估计结果符合人格设计意图。

#### 动态 Prompt Hint

状态机的输出不是固定字符串，而是根据三个维度动态生成自然语言暗示：

- **强度**：距基线的偏离程度（"淡淡的" / 默认 / "比较强烈"）
- **轨迹**：Δv 方向（"正在好转" / "继续低落" / "越来越高兴"）
- **持续性**：在同一情绪区间停留的轮次（"刚刚转变" / "持续了一段时间"）

```
第 1 轮难过: "聊了一些沉重的话题...（只是淡淡的）。刚刚情绪发生了转变"
第 5 轮持续难过: "聊了一些沉重的话题...（情绪比较强烈），情绪还在继续低落。这种状态已经持续了一段时间"
第 6 轮开始好转: "聊了一些沉重的话题...，不过情绪正在慢慢好转。刚刚情绪发生了转变"
```

#### LLM 参数自适应调节

状态机还根据 (valence, arousal) 动态调节 LLM 生成参数：

- **temperature**：arousal 驱动（高唤醒 → 更高多样性）
- **max_tokens**：负 valence 延长回复（安慰需要更多话）
- **top_p**：arousal 偏离基线时增大

### 双 LLM 群聊路由

通过 `ChatMode` 控制 LLM 选择策略，私聊保质量、群聊省费用：

| 模式 | 默认 LLM | 说明 |
|---|---|---|
| `PRIVATE` | 主 LLM (Claude Opus) | 私聊 / CLI，始终用高质量模型 |
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

### 动态视觉感知

基于 OpenCV 的实时视频流事件检测 + GLM-4V 云端多模态理解，采用分层策略：本地做轻量实时的感知与筛选，云端做深度语义化的理解。

#### 视觉管线流程

```
视频流 → 帧采样(5fps) → 多通道显著度检测 → 时序峰值筛选 → 关键帧提取(clip mode)
                                                                      |
                                               +----------------------+
                                               v
                                    GLM-4V 结构化分析 (JSON)
                                    scene / facts / agent_hint
                                               |
                              +----------------+----------------+
                              v                                 v
                    即时路径 (push)                     记忆路径
                    GLM 直出人格回复                   L3 + L4 写入
                    (keyframes + persona prompt)      压缩观察存入记忆
```

#### 显著度检测通道

| 通道 | 方法 | 检测目标 |
|------|------|---------|
| 像素差分 | 帧间差分 + 形态学滤波 | 运动、物体进出画面 |
| 结构变化 | SSIM 结构相似度 | 场景切换、光照突变 |
| 光流密度 | Farneback 光流 | 大幅动作、手势 |
| 人脸表情 | Haar 级联 + 表情变化 | 面部表情变化（可选） |

各通道输出经 sigmoid 归一化后加权融合为综合显著度分数，通过 `TemporalPeakSelector` 时序峰值检测筛选出事件。

#### 双路径响应模式

**即时路径（visual_direct）**：GLM-4V 直接带着人格 prompt + 关键帧图片生成对话回复，跳过文本描述中间环节。回复仍经 BERT 情绪分析 → OU 状态机更新，完全复用现有情感设施。

**技能调用（pull 模式）**：用户问"你看到什么"时，系统自动检测视觉请求意图（15 个中文关键词正则匹配），按需调用 `analyze_recent_buffer()` 获取最近画面分析结果注入对话上下文。

```python
# 注册视觉技能（启动时）
persona.register_visual_skill(
    handler=lambda top_k=2: vision_pipeline.analyze_recent_buffer(top_k=top_k),
)

# 之后用户说"你能看看我在做什么吗"时自动触发
# user_input 变为: "[视觉感知] 人物坐在书桌前，手握杯子\n你能看看我在做什么吗？"
```

#### 延迟特性

| 模式 | 延迟 | 适用场景 |
|------|------|---------|
| GLM 直出（8 帧） | ~18s | 低频高质量即时事件 |
| 文本描述 + Claude | ~9s | 私聊高质量回复 |
| 文本描述 + DeepSeek | ~2s | 高频/群聊实时响应 |

## 项目结构

```
project_src/
+-- config.json                # 主配置（API、人格、记忆、情绪映射、音频、视觉）
+-- configs/
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
|   +-- core/
|   |   +-- inference_pipeline.py  # 核心：LLMClient + NeuroLikePipeline
|   |   +-- persona.py             # 人格实例（对话、情绪、记忆、视觉技能）
|   |   +-- prompt_builder.py      # System Prompt 组装
|   |   +-- shared_infra.py        # 共享基础设施（LLM 池、小模型、记忆）
|   |   +-- small_model.py         # BERT 小模型推理
|   |   +-- emotion_state.py       # 情绪状态机（OU 过程）
|   |   +-- emotion_fusion.py      # BERT + LLM 情绪融合
|   |   +-- scheduler.py           # 定时任务调度
|   |
|   +-- agent/
|   |   +-- agent_loop.py          # Agent 事件循环（AgentLoop + AgentEvent）
|   |
|   +-- memory/
|   |   +-- memory_manager.py      # 分级记忆管理器（L1-L4, Mem0, Qdrant）
|   |
|   +-- llm/
|   |   +-- client.py              # 多提供商 LLM 客户端（Anthropic/OpenAI 兼容）
|   |
|   +-- vision/
|   |   +-- visual_perception.py   # 视觉感知主模块（检测、事件、配置）
|   |   +-- visual_pipeline.py     # 视觉管线编排（采集→检测→分析）
|   |   +-- visual_monitor.py      # 实时监控器（帧缓冲、峰值检测）
|   |   +-- visual_analysis.py     # GLM-4V 结构化分析
|   |   +-- visual_agent.py        # VisualEvent ↔ AgentEvent 桥接
|   |   +-- visual_skill.py        # 视觉技能检测器 + 执行器（pull 模式）
|   |   +-- visual_semantics.py    # 语义工具（情绪信号、记忆文本）
|   |   +-- visual_text.py         # 文本格式化（agent_text、memory_text）
|   |   +-- visual_types.py        # 数据类型定义
|   |   +-- _shared.py             # 共享常量与工具
|   |
|   +-- adapters/
|   |   +-- qq_adapter.py          # QQ 机器人适配层（OneBot v11 WebSocket）
|   |
|   +-- media/
|   |   +-- audio_pipeline.py      # TTS 语音合成管道（CosyVoice）
|   |   +-- speech_recognition.py  # ASR 语音识别（SenseVoice）
|   |   +-- image_utils.py         # 图片处理工具（下载/验证/缩放/base64/缓存）
|   |
|   +-- attention/
|   |   +-- attention_tracker.py   # 群聊注意力追踪器
|   |
|   +-- tests/                     # 单元测试
|   +-- tools/                     # 集成测试 / 工具脚本
|
+-- data/
|   +-- qdrant_db/             # Qdrant 向量数据库（L3 持久化）
|   +-- audio_refs/            # TTS 参考音频
|   +-- image_cache/           # 图片缓存
|
+-- checkpoints/
|   +-- joint_model/best.pt    # BERT 联合模型检查点
|
+-- docs/                      # 设计文档与变更日志
+-- logs/                      # 运行日志
+-- requirements.txt
```

## 配置

所有配置集中在 `config.json`，通过 `AppConfig.load()` 加载：

```jsonc
{
  "llm": {
    "provider": "anthropic",       // 主 LLM：私聊 + 群聊升级
    "model": "claude-opus-4-6",
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
  },

  "visual_perception": {
    "enabled": true,
    "source": "camera",                 // "camera" | "file"
    "sample_fps": 5,                    // 帧采样率
    "clip_duration_seconds": 2.0,       // 事件 clip 时长
    "clip_max_frames": 8,               // clip 内最大关键帧数
    "vision_analysis_mode": "per_event", // "per_event" | "batch"
    "vision_calls_per_minute": 5,       // GLM API 调用频率限制
    "persist_to_memory": true,          // 非即时事件写入 L3/L4
    "channel_weights": {
      "pixel_diff": 0.35,
      "structural": 0.25,
      "optical_flow": 0.25,
      "face": 0.15
    }
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
- 当前状态：情绪头可用（macro-F1=0.64，接近标注数据贝叶斯上限），行为头用于记忆压缩边界检测

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
- [x] 情绪状态机（OU 过程 + EKF 参数辨识 + 动态 Prompt Hint）
- [x] BERT + LLM 情绪融合（单神经元 softmax + reliability 加权）
- [x] 动态视觉感知管线（OpenCV 多通道显著度检测 + GLM-4V 云端分析）
- [x] 视觉直出响应（GLM-4V 带人格 prompt + 关键帧直接生成对话回复）
- [x] 视觉技能调用（pull 模式，用户请求时按需获取画面分析）
- [ ] CosyVoice TTS 语音合成集成到 AgentLoop
- [ ] SenseVoice ASR 语音识别集成到 AgentLoop
- [ ] 自适应帧采样密度（根据显著度动态调节采样率）
- [ ] Live2D 渲染 + 口型同步
- [ ] 个人博客看板娘前端
- [ ] 本地 LLM 替代 API（Qwen2-7B 蒸馏）

## License

MIT License

## 致谢

本项目灵感来源于 Neuro-sama，感谢 Vedal 的创新工作。
