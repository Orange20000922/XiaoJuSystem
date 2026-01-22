# Neuro-Like System

类Neuro虚拟主播AI系统 - 使用两个小模型+大模型API实现风格稳定的对话生成

## 项目结构

```
neuro_like_system/
├── configs/                # 配置文件
│   ├── __init__.py
│   └── model_config.py    # 模型配置、标签定义、人格配置
├── models/                 # 模型架构
│   ├── __init__.py
│   ├── emotion_model.py   # 情绪识别模型 (Encoder)
│   ├── behavior_model.py  # 行为生成模型 (Encoder-Decoder)
│   └── joint_model.py     # 联合模型 (推荐)
├── data/                   # 数据处理
│   └── dataset.py         # 数据集定义
├── scripts/                # 脚本
│   ├── data_annotation.py # 数据标注工具
│   ├── train_joint_model.py # 训练脚本
│   └── inference_pipeline.py # 推理Pipeline
└── utils/                  # 工具函数
```

## 系统架构

### 方案A: 两个独立模型 (原始方案)

```
用户输入 → [情绪识别模型] → 情绪标签+强度
                              ↓
                        [行为生成模型] → 行为+语气+长度
                              ↓
                        [大模型API] → 自然语言回复
```

### 方案B: 联合模型 (推荐)

```
用户输入 + 人格配置 → [联合模型] → {情绪, 行为, 语气}
                                      ↓
                                [大模型API] → 自然语言回复
```

**推荐使用方案B的原因:**
- 参数更少 (~100M vs 200M)
- 训练更简单 (一个模型)
- 推理更快 (一次前向传播)
- 情绪和行为联合训练，一致性更好

## 核心组件

### 1. 情绪标签 (10种)
- joy (喜悦)
- sadness (悲伤)
- anger (愤怒)
- fear (恐惧)
- surprise (惊讶)
- disgust (厌恶)
- neutral (中性)
- excitement (兴奋)
- tenderness (温柔)
- curiosity (好奇)

### 2. 行为标签 (12种)
- respond_positive (积极回应)
- respond_negative (消极回应)
- ask_question (提问)
- share_experience (分享经历)
- give_advice (给建议)
- express_empathy (表达共情)
- make_joke (开玩笑)
- change_topic (转移话题)
- seek_clarification (寻求澄清)
- agree (同意)
- disagree (不同意)
- neutral_acknowledge (中性确认)

### 3. 语气标签 (8种)
- enthusiastic (热情)
- calm (平静)
- playful (俏皮)
- serious (严肃)
- warm (温暖)
- cold (冷淡)
- sarcastic (讽刺)
- supportive (支持)

### 4. 人格配置
基于Big Five人格模型:
- openness (开放性)
- conscientiousness (尽责性)
- extraversion (外向性)
- agreeableness (宜人性)
- neuroticism (神经质)

额外特质:
- humor_tendency (幽默倾向)
- empathy_level (共情能力)
- curiosity_level (好奇心)

## 快速开始

### 1. 安装依赖

```bash
pip install torch transformers tqdm openai anthropic
```

### 2. 数据标注

#### 方案A: 使用GPT-4o-mini标注 (推荐)

```bash
python scripts/data_annotation.py annotate \
    --input raw_texts.json \
    --output annotated_data.json \
    --provider openai \
    --model gpt-4o-mini \
    --api_key YOUR_API_KEY \
    --batch_size 10
```

#### 方案B: 使用DeepSeek标注 (低成本)

```bash
python scripts/data_annotation.py annotate \
    --input raw_texts.json \
    --output annotated_data.json \
    --provider deepseek \
    --api_key YOUR_DEEPSEEK_KEY \
    --batch_size 20
```

#### 对比不同API的标注质量

```bash
python scripts/data_annotation.py compare \
    --input test_texts.json \
    --output comparison.json \
    --openai_key YOUR_OPENAI_KEY \
    --deepseek_key YOUR_DEEPSEEK_KEY
```

**输入数据格式 (raw_texts.json):**
```json
[
    {"text": "今天天气真好！"},
    {"text": "我感觉有点累..."},
    {"text": "哈哈哈笑死我了"}
]
```

**输出数据格式 (annotated_data.json):**
```json
[
    {
        "text": "今天天气真好！",
        "emotion": "joy",
        "intensity": 0.8,
        "behavior": "respond_positive",
        "tone": "enthusiastic",
        "response_length": "medium",
        "confidence": 0.95
    }
]
```

### 3. 训练模型

```bash
python scripts/train_joint_model.py \
    --train_data annotated_data.json \
    --val_data val_data.json \
    --output_dir ./checkpoints \
    --batch_size 32 \
    --num_epochs 10 \
    --learning_rate 2e-5 \
    --device cuda
```

**训练参数说明:**
- `batch_size`: 根据显存调整 (16GB显存建议32)
- `num_epochs`: 10-15轮通常足够
- `learning_rate`: 2e-5 是BERT微调的标准学习率

**预期训练时间:**
- 5万条数据，RTX 4090: ~2-3小时
- 5万条数据，RTX 3060: ~6-8小时
- 5万条数据，CPU: 不推荐 (太慢)

### 4. 推理对话

```bash
python scripts/inference_pipeline.py \
    --checkpoint ./checkpoints/best.pt \
    --provider openai \
    --api_key YOUR_API_KEY \
    --model gpt-4o-mini \
    --device cuda
```

**支持的大模型API:**
- OpenAI: gpt-4o, gpt-4o-mini, gpt-3.5-turbo
- DeepSeek: deepseek-chat
- Claude: claude-3-haiku-20240307, claude-3-sonnet-20240229

## 模型详细说明

### 联合模型架构 (joint_model.py)

```python
from models import create_joint_model
from configs import DEFAULT_JOINT_CONFIG, DEFAULT_PERSONALITY

# 创建模型
model, tokenizer = create_joint_model(DEFAULT_JOINT_CONFIG)

# 推理
import torch
personality_vec = torch.tensor(DEFAULT_PERSONALITY.to_embedding_vector())
result = model.predict("今天天气真好！", personality_vec, tokenizer)

print(result)
# {
#     "emotion": {
#         "primary": "joy",
#         "intensity": 0.85,
#         "primary_prob": 0.92
#     },
#     "behavior": {
#         "type": "respond_positive",
#         "tone": "enthusiastic",
#         "response_length": "medium"
#     }
# }
```

### 模型参数量

| 模型 | 参数量 | 推理速度 (CPU) | 推理速度 (GPU) |
|------|--------|---------------|---------------|
| 情绪识别模型 | ~110M | ~100ms | ~20ms |
| 行为生成模型 | ~120M | ~150ms | ~30ms |
| **联合模型 (推荐)** | **~100M** | **~80ms** | **~15ms** |

## 成本估算

### 数据标注成本 (5万条)

| 方案 | 模型 | 成本 | 质量 |
|------|------|------|------|
| GPT-4 | gpt-4 | ~$100-150 | 最高 (90%+) |
| **GPT-4o-mini (推荐)** | **gpt-4o-mini** | **~$7-10** | **高 (85-88%)** |
| DeepSeek | deepseek-chat | ~$70 | 中等 (80-85%) |

### 训练成本

| 方案 | 成本 |
|------|------|
| 本地GPU (RTX 3060/4060) | $0 (电费忽略) |
| 云GPU (RunPod RTX 4090) | ~$30-40 |
| 云GPU (AWS A100) | ~$80-100 |

### 运行成本 (每月，个人使用)

假设每天对话100轮，每轮200 tokens:

| 大模型 | 月成本 |
|--------|--------|
| GPT-4o | ~$30-50 |
| **GPT-4o-mini (推荐)** | **~$1-3** |
| DeepSeek | ~$1-2 |
| Claude Haiku | ~$0.5-1 |
| 本地部署 (Qwen2-7B) | $0 |

### 总预算 (200-400美元)

**推荐配置 ($250-300):**
```
数据标注:
- GPT-4o-mini标注4万条: $6-8
- GPT-4标注1万条复杂样本: $20-30
- 人工审核500条: 你的时间

模型训练:
- 云GPU租用 (RunPod RTX 4090): $30-40

运行成本 (首年):
- Claude Haiku API: $15-30/月 × 12 = $180-360

总计: $236-438 (在预算内)
```

## 性能预期

### 小模型准确率

| 指标 | 使用GPT-4o-mini标注 | 使用GPT-4标注 |
|------|-------------------|--------------|
| 情绪识别准确率 | 80-85% | 85-90% |
| 行为识别准确率 | 75-82% | 82-88% |
| 推理速度 (GPU) | ~15ms | ~15ms |

### 与Neuro对比

| 能力 | Neuro | 你的系统 | 差距 |
|------|-------|---------|------|
| 对话质量 | 95分 | 80-85分 | 模型规模 |
| **风格一致性** | **90分** | **85-90分** | **架构优势** |
| 反应速度 | 快 | 相当 | 小模型快 |
| 记忆能力 | 强 | 中等 | 简化设计 |

**结论:** 对于个人使用，80-85分的水平完全够用，风格一致性这个核心目标可以做得很好。

## 高级功能

### 自定义人格

```python
from configs import PersonalityConfig

# 创建自定义人格
my_personality = PersonalityConfig(
    name="小助手",
    extraversion=0.6,      # 中等外向
    humor_tendency=0.5,    # 适度幽默
    empathy_level=0.9,     # 高共情
    traits=["温柔", "耐心", "专业"]
)

# 使用自定义人格训练/推理
```

### 流式数据加载 (大规模数据)

```python
from data.dataset import create_dataloader

# 使用流式加载，节省内存
dataloader = create_dataloader(
    data_path="large_dataset.jsonl",  # JSONL格式
    tokenizer=tokenizer,
    personality=personality,
    streaming=True  # 启用流式加载
)
```

### 多人格切换

```python
# 在推理时动态切换人格
personalities = {
    "活泼": PersonalityConfig(extraversion=0.9, humor_tendency=0.8),
    "严肃": PersonalityConfig(extraversion=0.3, humor_tendency=0.2),
}

result = model.predict(text, personalities["活泼"].to_embedding_vector(), tokenizer)
```

## 常见问题

### Q1: 我应该用两个独立模型还是联合模型？

**A:** 强烈推荐联合模型 (joint_model.py)，原因:
- 更简单: 只需训练一个模型
- 更快: 推理速度快一倍
- 更小: 参数量少一半
- 更好: 情绪和行为联合训练，一致性更好

### Q2: GPT-4o-mini和DeepSeek哪个更好？

**A:** 建议先对比测试 (使用compare命令)，一般来说:
- GPT-4o-mini: 质量更高，中文理解好，性价比极高
- DeepSeek: 成本更低，中文也不错，但一致性稍差

### Q3: 需要多少训练数据？

**A:**
- 最少: 1万条 (可以跑起来，质量一般)
- 推荐: 3-5万条 (质量不错)
- 理想: 10万条+ (接近商业水平)

### Q4: 训练需要多久？

**A:** 5万条数据:
- RTX 4090: 2-3小时
- RTX 3060: 6-8小时
- 云GPU (A100): 1-2小时

### Q5: 可以用CPU训练吗？

**A:** 技术上可以，但不推荐:
- CPU训练5万条数据需要几天
- 建议租用云GPU (RunPod/Vast.ai很便宜)

### Q6: 如何提高模型质量？

**A:**
1. 提高标注质量 (使用GPT-4标注关键样本)
2. 增加训练数据量
3. 人工审核并修正错误标注
4. 使用主动学习 (挑选低置信度样本重新标注)
5. 在真实场景中收集badcase，持续优化

## 下一步计划

- [ ] 添加TTS语音合成
- [ ] 添加Live2D表情驱动
- [ ] 实现长期记忆系统 (向量数据库)
- [ ] 支持多模态输入 (图片、语音)
- [ ] 添加情绪轨迹可视化
- [ ] 实现在线学习 (从用户反馈中学习)

## 参考资料

- Neuro-sama: https://www.twitch.tv/vedal987
- Transformer论文: "Attention is All You Need"
- BERT: "BERT: Pre-training of Deep Bidirectional Transformers"
- Big Five人格模型: https://en.wikipedia.org/wiki/Big_Five_personality_traits

## License

MIT License

## 致谢

本项目灵感来源于Neuro-sama，感谢Vedal的创新工作。
