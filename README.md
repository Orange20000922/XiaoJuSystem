# Neuro-Like System

类 Neuro 虚拟主播 AI 系统 — BERT 情绪感知 + 分级记忆 + LLM 人格对话

## 系统架构

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        Neuro-Like System                                │
│                                                                         │
│  用户输入                                                                │
│    │                                                                    │
│    ├──────────────────────┐                                             │
│    ▼                      ▼                                             │
│  ┌──────────────┐   ┌──────────────────────────────────────────────┐   │
│  │  BERT 小模型  │   │          分级记忆系统                         │   │
│  │  (~15ms GPU)  │   │                                              │   │
│  │              │   │  L1 工作记忆    原文 messages（保护区不压缩）   │   │
│  │  情绪分类     │   │       ↓ 压缩触发                              │   │
│  │  ·emotion     │   │  L2 会话摘要    结构化状态提取（非叙述性）     │   │
│  │  ·intensity   │   │       ↓ close_session                        │   │
│  │  ·confidence  │   │  L3 长期记忆    Mem0 + Qdrant 向量持久化      │   │
│  │              │   │                                              │   │
│  └──────┬───────┘   │  L4 知识库      用户画像 / Agent 结果          │   │
│         │           └────────────────────┬─────────────────────────┘   │
│         │                                │                             │
│         ▼                                ▼                             │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    System Prompt 组装                            │   │
│  │                                                                 │   │
│  │  <persona> 人格描述 </persona>                                  │   │
│  │  <emotion_context> BERT 置信度门控指令 </emotion_context>        │   │
│  │  <memory> L2/L3/L4 召回上下文 </memory>                         │   │
│  │  + L1 原文 messages 数组                                        │   │
│  └──────────────────────────┬──────────────────────────────────────┘   │
│                             │                                          │
│                     ┌───────┴───────┐                                  │
│                     │  LLM 路由     │                                  │
│                     │               │                                  │
│          ┌──────────┴──────────┐                                       │
│          ▼                     ▼                                       │
│  ┌───────────────┐   ┌────────────────┐                                │
│  │ 主 LLM        │   │ 副 LLM         │                               │
│  │ (Claude)      │   │ (DeepSeek)     │                               │
│  │               │   │                │                               │
│  │ 私聊 always   │   │ 群聊 default   │                               │
│  │ 群聊 escalate │   │                │                               │
│  └───────┬───────┘   └───────┬────────┘                                │
│          └───────┬───────────┘                                         │
│                  ▼                                                      │
│              回复输出                                                    │
└──────────────────────────────────────────────────────────────────────────┘
```

### BERT 置信度门控

BERT 的情绪分类结果不直接注入 LLM，而是经过三级置信度门控：

```
effective_confidence = BERT_prob × emotion_reliability[label]

 > strong (0.5)  →  确定性指令注入：「用户心情不错，自然地一起开心就好」
 > weak   (0.3)  →  不确定注入：「用户可能心情不错，但不确定，结合上下文判断」
 ≤ weak          →  跳过，让 LLM 自行判断
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

高强度情绪（intensity ≥ 0.7）额外 +0.15 权重。

## 项目结构

```
neuro_like_system/
├── config.json                # 主配置（API、人格、记忆、情绪映射）
├── configs/
│   ├── __init__.py
│   ├── config_loader.py       # config.json → AppConfig 加载器
│   └── model_config.py        # 数据类定义（LLMConfig, MemoryConfig, PersonalityConfig 等）
│
├── models/
│   ├── joint_model.py         # 联合模型（BERT 情绪+行为+语气+强度）
│   ├── emotion_model.py       # 独立情绪模型
│   ├── behavior_model.py      # 独立行为模型
│   └── annotation_model.py    # 标注模型（知识蒸馏）
│
├── src/
│   ├── inference_pipeline.py  # 核心：LLMClient + NeuroLikePipeline
│   ├── memory_manager.py      # 分级记忆管理器（L1-L4, Mem0, Qdrant）
│   ├── quick_test.py          # 测试脚本（多模式：API/BERT/记忆/跨session）
│   ├── logger.py              # loguru 日志配置
│   ├── content_filter.py      # 内容过滤器（敏感词清洗）
│   ├── danmaku_annotation.py  # 弹幕标注工具（API + 模式匹配）
│   ├── data_annotation.py     # 通用数据标注工具
│   ├── train_joint_model.py   # 联合模型训练
│   ├── train_annotation_model.py  # 标注模型训练
│   └── export_onnx.py         # ONNX 导出
│
├── data/
│   ├── dataset.py             # PyTorch Dataset 定义
│   └── qdrant_db/             # Qdrant 向量数据库（L3 持久化）
│
├── checkpoints/
│   └── joint_model/best.pt    # BERT 联合模型检查点
│
├── docs/                      # 设计文档
├── logs/                      # 运行日志
└── requirements.txt
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

  "personality": {
    "name": "Neuro",
    "traits": ["活泼", "好奇", "善良"],
    "description": "人格描述（直接写入 system prompt）..."
    // Big Five 参数：openness, extraversion, humor_tendency 等
  },

  "memory": {
    "vector_store_path": "./data/qdrant_db",
    "context_window_tokens": 400000,
    "compression_threshold": 0.75,   // L1 token 占比超过此值触发压缩
    "relevance_threshold": 0.3       // L3 语义搜索阈值
  },

  "emotion_prompts": {
    "emotion_map": { "joy": "用户心情不错...", ... },
    "emotion_reliability": { "joy": 0.66, "anger": 0.45, ... },
    "confidence_thresholds": { "strong": 0.5, "weak": 0.3 }
  },

  "annotation": {
    "primary_provider": "deepseek",  // 标注用的 LLM（低成本）
    "fallback_provider": "openai"    // 备用标注 LLM
  }
}
```

## 快速开始

### 环境准备

```bash
# 安装依赖
pip install -r requirements.txt

# 主要依赖：torch, transformers, anthropic, openai, mem0ai, qdrant-client,
#           tiktoken, loguru, tqdm
```

### 运行测试

```bash
cd neuro_like_system

# 1. 测试 API 连通性
python src/quick_test.py --ping

# 2. 纯 LLM 对话测试（无 BERT）
python src/quick_test.py --msg "你好|||最近在干嘛|||聊聊AI吧"

# 3. BERT + LLM 联合测试（显示情绪分析 + 指令注入 + 回复）
python src/quick_test.py --bert-test

# 4. 记忆系统测试（Token 计数 + L1→L2 压缩 + 向量 DB 状态）
python src/quick_test.py --memory-test --compress-at 300 --protected-turns 2

# 5. 跨 session 记忆召回测试
python src/quick_test.py --memory-test --cross-session --compress-at 300
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
- **压缩触发**：`L1_tokens > context_window × compression_threshold`
- **跨 session 召回**：`close_session()` 时保护区原文写入 L3，新 session 通过语义搜索召回
- **自然提示**：语义搜索无结果但 L3 有记录时，注入 `<has_memory>` 标签让 LLM 自然提及

## BERT 模型

### 联合模型 (joint_model.py)

- 基座：`hfl/chinese-roberta-wwm-ext` (~100M 参数)
- 输出头：情绪分类（10 类） + 行为分类 + 语气分类 + 强度回归
- 推理速度：~15ms/条 (GPU)
- 当前状态：情绪头可用（68% 准确率），行为/语气头未专项训练（权重=0 导致预测为噪声，已禁用注入）

### 情绪标签 (10 类)

| ID | 标签 | 可靠度 | 说明 |
|---|---|---|---|
| 0 | joy | 0.66 | 喜悦 |
| 1 | sadness | 0.66 | 悲伤 |
| 2 | anger | 0.45 | 愤怒（可靠度低，中置信度注入） |
| 3 | fear | 0.69 | 恐惧 |
| 4 | surprise | 0.62 | 惊讶 |
| 5 | disgust | 0.58 | 厌恶 |
| 6 | neutral | 0.76 | 中性（最可靠） |
| 7 | excitement | 0.62 | 兴奋 |
| 8 | tenderness | 0.68 | 温柔 |
| 9 | curiosity | 0.74 | 好奇 |

### 训练数据

弹幕标注数据 ~80K 条（`data_set/neuro_danmakus_annotated.json`）：
- 来源：B 站 Neuro/AI 虚拟主播相关弹幕
- 标注流程：模式匹配预标注（~70%）→ DeepSeek API 标注剩余 → 人工审核
- 已知问题：anger / fear / disgust 样本不足（< 2000），存在类别不平衡

## 数据流水线

### 采集 → 清洗 → 标注 → 训练

```
弹幕爬虫 → 内容过滤器 → 模式匹配预标注 → LLM API 标注 → BERT 训练
              │                                   │
              ▼                                   ▼
         敏感词过滤                          知识蒸馏（可选）：
         · 脏话                             GPT 标注种子数据
         · 政治敏感                          → 训练标注小模型
         · 色情/广告/垃圾                    → 小模型大规模标注
```

```bash
# 过滤
python src/content_filter.py filter --input raw.json --output cleaned.json

# 标注
python src/danmaku_annotation.py annotate --input cleaned.json --output annotated.json \
    --provider deepseek --batch_size 50

# 训练
python src/train_joint_model.py --train_data annotated.json \
    --output_dir ./checkpoints/joint_model --device cuda
```

## 已知问题与技巧

### Anthropic 代理 401 双 header 冲突

环境变量 `ANTHROPIC_AUTH_TOKEN` 会让 SDK 同时发送 `x-api-key` 和 `Authorization` header，导致代理报 401。已通过 httpx event hook 在请求发出前删除 `authorization` header 解决。

### HuggingFace 离线模式

BERT 模型缓存到本地后，`inference_pipeline.py` 启动时自动检测缓存目录并设置 `HF_HUB_OFFLINE=1`，跳过联网检查。

### Qdrant Windows 锁文件

跨 session 测试时需在创建新 session 前关闭 Qdrant 客户端（包括 `vector_store` 和 `_telemetry_vector_store`），否则 `.lock` 文件被占用。`NeuroLikePipeline.close()` 已处理。

### PyTorch 2.10+ weights_only 默认值

`torch.load` 默认 `weights_only=True`，加载含自定义类的 checkpoint 时需显式传 `weights_only=False`。

## 硬件要求

| 配置 | GPU | RAM | 说明 |
|---|---|---|---|
| 最低 | 8GB VRAM (RTX 3060/4060) | 16GB | `--low_vram` 模式训练 |
| 推荐 | 12GB+ VRAM | 32GB | 标准训练 |
| 纯推理 | CPU 可用 | 8GB+ | BERT ~15ms/条，LLM 依赖 API |

## 远期规划

- [ ] QQ 机器人接入（群聊注意力判断）
- [ ] behavior/tone 标签补全并重新训练
- [ ] 外部数据集补充（SMP2020 EWECT 解决类别不平衡）
- [ ] 本地 LLM 替代 API（Qwen2-7B 蒸馏）
- [ ] TTS 语音合成 + Live2D 表情驱动
- [ ] 多模态输入（图片、语音）

## License

MIT License

## 致谢

本项目灵感来源于 Neuro-sama，感谢 Vedal 的创新工作。
