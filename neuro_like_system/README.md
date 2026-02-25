# Neuro-Like System

类Neuro虚拟主播AI系统 - 使用两个小模型+大模型API实现风格稳定的对话生成

## 系统架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Neuro-Like System 架构图                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐    ┌──────────────────┐    ┌─────────────────────────┐    │
│  │  弹幕/评论   │───▶│   内容过滤器      │───▶│   标注模型 (轻量级)      │    │
│  │  数据采集    │    │  content_filter  │    │   annotation_model      │    │
│  └─────────────┘    └──────────────────┘    └───────────┬─────────────┘    │
│                                                         │                   │
│                                                         ▼                   │
│                                              ┌─────────────────────┐        │
│                                              │   标注数据 (JSON)    │        │
│                                              └───────────┬─────────┘        │
│                                                         │                   │
│                                                         ▼                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        训练阶段                                      │   │
│  │  ┌─────────────────┐              ┌─────────────────────────────┐   │   │
│  │  │  联合模型训练     │              │  标注模型训练 (知识蒸馏)      │   │   │
│  │  │  joint_model     │              │  GPT标注 → 小模型 → 大规模   │   │   │
│  │  └────────┬────────┘              └─────────────────────────────┘   │   │
│  └───────────┼─────────────────────────────────────────────────────────┘   │
│              │                                                              │
│              ▼                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        推理阶段                                      │   │
│  │                                                                      │   │
│  │  用户输入 ──▶ [联合模型] ──▶ {情绪, 行为, 语气} ──▶ [LLM API] ──▶ 回复  │   │
│  │              (15ms GPU)                           (GPT/Claude)       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 项目结构

```
neuro_like_system/
├── configs/                    # 配置文件
│   ├── __init__.py
│   └── model_config.py        # 模型配置、标签定义、人格配置、API配置
│
├── models/                     # 模型架构
│   ├── __init__.py
│   ├── emotion_model.py       # 情绪识别模型 (Encoder)
│   ├── behavior_model.py      # 行为生成模型 (Encoder-Decoder)
│   ├── joint_model.py         # 联合模型 (推荐用于推理)
│   └── annotation_model.py    # 标注模型 (用于知识蒸馏)
│
├── data/                       # 数据处理
│   └── dataset.py             # 数据集定义
│
├── src/                        # 核心脚本
│   ├── content_filter.py      # 内容过滤器 (清洗负面内容)
│   ├── danmaku_annotation.py  # 弹幕标注工具 (支持API和模型标注)
│   ├── data_annotation.py     # 通用数据标注工具
│   ├── train_annotation_model.py  # 标注模型训练
│   ├── train_joint_model.py   # 联合模型训练
│   ├── inference_pipeline.py  # 推理Pipeline
│   └── export_onnx.py         # ONNX导出 (用于C++部署)
│
├── src/wordlists/             # 敏感词词库 (自动生成)
│   ├── profanity.txt
│   ├── political.txt
│   └── ...
│
├── checkpoints/               # 模型检查点 (训练后生成)
│   ├── annotation_model/
│   └── joint_model/
│
├── README.md
└── requirements.txt
```

## 完整工作流

### 阶段1: 环境准备

```bash
# 安装依赖
pip install torch transformers tqdm openai anthropic onnx onnxruntime

# 或使用requirements.txt
pip install -r requirements.txt
```

### 阶段2: 数据采集

使用你的弹幕爬虫采集数据:
- 推荐关键词: Neuro、AI虚拟主播、Vedal 等
- 弹幕优势: 无风控限制，单视频可获取上万条
- 目标数量: 10万条原始数据

### 阶段3: 数据清洗

```bash
# 初始化敏感词词库 (首次运行)
python -c "from src.content_filter import init_wordlists; init_wordlists('./src/wordlists')"

# 过滤负面内容
python src/content_filter.py filter \
    --input raw_danmaku.json \
    --output cleaned_danmaku.json \
    --wordlist_dir ./src/wordlists
```

过滤类别:
- profanity (脏话)
- political (政治敏感)
- pornographic (色情)
- advertisement (广告)
- toxic (人身攻击)
- spam (垃圾信息)

### 阶段4: 数据标注 (知识蒸馏)

#### 4.1 GPT标注种子数据

```bash
# 使用模式匹配预标注 + GPT标注剩余数据
python src/danmaku_annotation.py annotate \
    --input cleaned_danmaku.json \
    --output seed_annotated.json \
    --provider openai \
    --model gpt-4o-mini \
    --api_key YOUR_API_KEY \
    --batch_size 50
```

预期: 模式匹配覆盖70%+，API只需标注30%，节省大量成本

#### 4.2 训练标注模型

```bash
python src/train_annotation_model.py \
    --data seed_annotated.json \
    --output_dir ./checkpoints/annotation_model \
    --batch_size 64 \
    --num_epochs 5 \
    --device cuda
```

#### 4.3 使用标注模型大规模标注

```bash
python src/danmaku_annotation.py annotate-model \
    --input large_dataset.json \
    --output full_annotated.json \
    --model_path ./checkpoints/annotation_model/best.pt \
    --batch_size 128 \
    --device cuda
```

推理速度: ~10000条/分钟 (RTX 5060)

### 阶段5: 训练联合模型

```bash
# 标准训练
python src/train_joint_model.py \
    --train_data full_annotated.json \
    --output_dir ./checkpoints/joint_model \
    --batch_size 32 \
    --num_epochs 10

# 低显存模式 (8GB VRAM)
python src/train_joint_model.py \
    --train_data full_annotated.json \
    --output_dir ./checkpoints/joint_model \
    --low_vram \
    --device cuda
```

### 阶段6: 推理部署

```bash
# 交互式对话 (完整模式)
python src/inference_pipeline.py \
    --checkpoint ./checkpoints/joint_model/best.pt \
    --provider openai \
    --model gpt-4o-mini \
    --api_key YOUR_API_KEY
```

#### 测试CLI

提供独立的测试CLI，支持多种运行模式：

```bash
# 1. 纯本地测试 (无需模型、无需API，快速验证流程)
python src/test_cli.py --mock --offline

# 2. 测试LLM API连接 (无需训练模型)
python src/test_cli.py --mock --provider openai --api_key YOUR_KEY

# 3. 测试小模型推理 (需要训练模型，不调用API)
python src/test_cli.py --checkpoint ./checkpoints/joint_model/best.pt --offline

# 4. 完整流程测试 (需要模型+API)
python src/test_cli.py --checkpoint ./checkpoints/joint_model/best.pt \
    --provider openai --api_key YOUR_KEY
```

**测试模式对比:**

| 模式 | 小模型 | LLM | 用途 |
|------|--------|-----|------|
| `--mock --offline` | 规则模拟 | 模拟回复 | 快速验证流程 |
| `--mock` | 规则模拟 | 真实API | 测试API连接 |
| `--offline` | 真实模型 | 模拟回复 | 测试模型推理 |
| 完整模式 | 真实模型 | 真实API | 生产环境 |

**CLI命令:**
- `quit/exit` - 退出
- `clear` - 清空对话历史
- `debug` - 切换详细输出
- `status` - 显示当前状态

#### ONNX导出 (可选，用于C++部署)

```bash
# 导出标注模型
python src/export_onnx.py annotation \
    --checkpoint ./checkpoints/annotation_model/best.pt \
    --output_dir ./onnx/annotation_model

# 导出联合模型
python src/export_onnx.py joint \
    --checkpoint ./checkpoints/joint_model/best.pt \
    --output_dir ./onnx/joint_model
```

导出文件:
- `model.onnx` - ONNX模型
- `model.json` - 元数据和标签映射
- `vocab.txt` - 词表
- `tokenizer_config.json` - 分词器配置

## 标签定义

### 情绪标签 (10种)

| ID | 英文 | 中文 | 示例 |
|----|------|------|------|
| 0 | joy | 喜悦 | "太棒了！" |
| 1 | sadness | 悲伤 | "好难过..." |
| 2 | anger | 愤怒 | "气死我了" |
| 3 | fear | 恐惧 | "好害怕" |
| 4 | surprise | 惊讶 | "什么？！" |
| 5 | disgust | 厌恶 | "恶心" |
| 6 | neutral | 中性 | "好的" |
| 7 | excitement | 兴奋 | "冲冲冲！" |
| 8 | tenderness | 温柔 | "辛苦了~" |
| 9 | curiosity | 好奇 | "这是什么？" |

### 行为标签 (12种)

| 英文 | 中文 | 说明 |
|------|------|------|
| respond_positive | 积极回应 | 正面肯定的回复 |
| respond_negative | 消极回应 | 负面否定的回复 |
| ask_question | 提问 | 向用户提问 |
| share_experience | 分享经历 | 分享自己的经历 |
| give_advice | 给建议 | 提供建议或指导 |
| express_empathy | 表达共情 | 表示理解和同情 |
| make_joke | 开玩笑 | 幽默调侃 |
| change_topic | 转移话题 | 切换到其他话题 |
| seek_clarification | 寻求澄清 | 请求更多信息 |
| agree | 同意 | 表示赞同 |
| disagree | 不同意 | 表示反对 |
| neutral_acknowledge | 中性确认 | 简单确认收到 |

### 语气标签 (8种)

| 英文 | 中文 |
|------|------|
| enthusiastic | 热情 |
| calm | 平静 |
| playful | 俏皮 |
| serious | 严肃 |
| warm | 温暖 |
| cold | 冷淡 |
| sarcastic | 讽刺 |
| supportive | 支持 |

## 人格配置

基于Big Five人格模型 + 额外特质:

```python
from configs import PersonalityConfig

# 默认人格 (类Neuro风格)
personality = PersonalityConfig(
    name="Neuro",
    # Big Five
    openness=0.8,           # 开放性
    conscientiousness=0.5,  # 尽责性
    extraversion=0.7,       # 外向性
    agreeableness=0.6,      # 宜人性
    neuroticism=0.4,        # 神经质
    # 额外特质
    humor_tendency=0.8,     # 幽默倾向
    empathy_level=0.7,      # 共情能力
    curiosity_level=0.9,    # 好奇心
    traits=["活泼", "可爱", "有点傲娇"]
)
```

## 模型说明

### 标注模型 (annotation_model.py)

用于知识蒸馏的轻量级模型:
- 基座: `hfl/chinese-roberta-wwm-ext`
- 参数量: ~100M
- 输出: 情绪分类 + 强度回归
- 推理速度: ~10000条/分钟 (GPU)

### 联合模型 (joint_model.py)

用于最终推理的主模型:
- 基座: `hfl/chinese-roberta-wwm-ext`
- 参数量: ~100M
- 输出: 情绪 + 行为 + 语气 + 强度 + 回复长度
- 推理速度: ~15ms/条 (GPU)

### 模型对比

| 模型 | 用途 | 参数量 | 输出 |
|------|------|--------|------|
| 标注模型 | 大规模数据标注 | ~100M | 情绪+强度 |
| 联合模型 | 推理部署 | ~100M | 情绪+行为+语气 |

## API配置

支持多种LLM API:

```python
from configs import LLMConfig

# OpenAI
config = LLMConfig.openai(
    api_key="YOUR_KEY",
    model="gpt-4o-mini",
    base_url="https://api.openai.com/v1"  # 可选，支持代理
)

# DeepSeek
config = LLMConfig.deepseek(
    api_key="YOUR_KEY",
    model="deepseek-chat"
)

# Claude
config = LLMConfig.claude(
    api_key="YOUR_KEY",
    model="claude-3-haiku-20240307"
)
```

## 成本估算

### 数据标注成本 (10万条弹幕)

| 阶段 | 方法 | 成本 |
|------|------|------|
| 模式匹配预标注 | 本地规则 | $0 |
| GPT标注种子数据 | gpt-4o-mini (3万条) | ~$5-8 |
| 训练标注模型 | 本地GPU | $0 |
| 模型标注剩余数据 | 本地推理 (7万条) | $0 |
| **总计** | | **~$5-8** |

### 训练成本

| 硬件 | 成本 |
|------|------|
| 本地GPU (RTX 5060 8GB) | $0 |
| 云GPU (RunPod RTX 4090) | ~$30-40 |

### 运行成本 (每月)

| LLM | 成本 (100轮/天) |
|-----|-----------------|
| GPT-4o-mini | ~$1-3 |
| Claude Haiku | ~$0.5-1 |
| DeepSeek | ~$1-2 |
| 本地LLM (未来) | $0 |

### 总预算 ($200-400)

```
数据标注: $5-10
云GPU训练 (可选): $30-40
LLM API (首年): $12-36
─────────────────
总计: $47-86 (远低于预算)

剩余预算可用于:
- 更多GPT-4标注提升质量
- 更长时间的API使用
- 未来本地LLM部署
```

## 硬件要求

### 最低配置
- GPU: 8GB VRAM (RTX 3060/4060/5060)
- RAM: 16GB
- 存储: 10GB

### 推荐配置
- GPU: 12GB+ VRAM (RTX 4070/4080)
- RAM: 32GB
- 存储: 50GB

### 8GB显存优化

使用 `--low_vram` 参数:
- batch_size=8
- gradient_accumulation=4
- FP16混合精度

## 常见问题

### Q1: 知识蒸馏的意义是什么？

**A:**
- GPT标注质量高但成本高 (~$0.0002/条)
- 训练小模型后，标注成本降为0
- 10万条数据: GPT全标注~$20 vs 蒸馏~$5

### Q2: 为什么用弹幕而不是评论？

**A:**
- 弹幕无风控限制，采集效率高
- 单视频可获取上万条
- 弹幕更口语化，适合训练对话模型

### Q3: C++ ONNX部署值得吗？

**A:**
- 当前: 不值得 (LLM API是瓶颈)
- 未来 (本地LLM): 值得 (可提升40-50%性能)

### Q4: 如何提升标注质量？

**A:**
1. 用GPT-4标注难样本
2. 人工审核低置信度样本
3. 增加模式匹配规则
4. 迭代训练标注模型

### Q5: 支持哪些语言？

**A:** 当前专注中文，但架构支持多语言:
- 更换预训练模型 (如 `bert-base-multilingual`)
- 调整模式匹配规则

## 远期规划

- [ ] 本地LLM替代API (Qwen2-7B蒸馏)
- [ ] C++ ONNX Runtime部署
- [ ] TTS语音合成集成
- [ ] Live2D表情驱动
- [ ] 长期记忆系统 (向量数据库)
- [ ] 多模态输入 (图片、语音)

## License

MIT License

## 致谢

本项目灵感来源于Neuro-sama，感谢Vedal的创新工作。
