# src/ 目录结构说明

## 概览

`src/` 目录已按功能模块重新组织，提高代码可维护性和可扩展性。

## 目录结构

```
src/
├── core/                    # 核心推理引擎
│   ├── inference_pipeline.py    # Pipeline 瘦包装层（向后兼容入口）
│   ├── shared_infra.py          # 共享基础设施（BERT、LLM 客户端池）
│   ├── persona.py               # 人格实例（独立状态）
│   ├── prompt_builder.py        # Prompt 构建器
│   ├── emotion_fusion.py        # BERT+LLM 情绪融合
│   └── emotion_state.py         # 情绪状态机（OU 过程）
│
├── memory/                  # 记忆系统
│   └── memory_manager.py        # 分级记忆管理器（L1-L4）
│
├── attention/               # 注意力系统
│   ├── attention_tracker.py     # 注意力追踪器（群聊场景）
│   └── proactive_decision.py    # 主动决策模块
│
├── agent/                   # Agent 事件循环
│   └── agent_loop.py            # Agent 事件循环（daemon thread）
│
├── adapters/                # 外部适配器
│   ├── qq_adapter.py            # QQ 机器人适配器（OneBot v11）
│   └── run_qq_bot.py            # QQ 机器人启动入口
│
├── llm/                     # LLM 客户端
│   └── client.py                # 统一 LLM 客户端（支持多 provider）
│
├── media/                   # 多模态处理
│   ├── image_utils.py           # 图片处理（下载/缩放/base64）
│   ├── audio_pipeline.py        # TTS 语音合成（CosyVoice）
│   └── speech_recognition.py    # ASR 语音识别（SenseVoice）
│
├── training/                # 训练相关脚本
│   ├── data_annotation.py       # 数据标注
│   ├── danmaku_annotation.py    # 弹幕标注
│   ├── content_filter.py        # 内容过滤
│   └── export_onnx.py           # ONNX 导出
│
├── tools/                   # 工具脚本
│   ├── add_memory.py            # 手动写入 L4 记忆
│   ├── ekf_tuner.py             # EKF 参数调优
│   ├── simulate_for_ekf.py      # EKF 仿真
│   └── debug_api.py             # API 调试
│
├── tests/                   # 测试脚本
│   └── (测试文件)
│
└── logger.py                # 日志模块（顶层工具）
```

## Import 路径

### 新路径（推荐）

```python
# 核心模块
from src.core.inference_pipeline import NeuroLikePipeline, ChatMode
from src.core.shared_infra import SharedInfra
from src.core.persona import PersonaInstance

# 记忆系统
from src.memory.memory_manager import HierarchicalMemoryManager

# 注意力系统
from src.attention.attention_tracker import AttentionTracker
from src.attention.proactive_decision import ProactiveDecisionModule

# Agent 层
from src.agent.agent_loop import AgentLoop, AgentEvent

# 适配器
from src.adapters.qq_adapter import QQBotAdapter

# 多媒体
from src.media.image_utils import process_image_url, ImageResult

# LLM 客户端
from src.llm.client import LLMClient

# 日志
from src.logger import logger
```

### 向后兼容路径（仍然可用）

```python
# 顶层 re-export，旧代码无需修改
from src import (
    NeuroLikePipeline,
    ChatMode,
    AgentLoop,
    QQBotAdapter,
    HierarchicalMemoryManager,
    logger,
)
```

## 模块职责

### core/ - 核心推理引擎
- **inference_pipeline.py**: 向后兼容的瘦包装层，内部创建 SharedInfra + PersonaInstance
- **shared_infra.py**: 单例共享资源（BERT 推理引擎、LLM 客户端池、情绪融合引擎）
- **persona.py**: 每个人格的独立实例（记忆、情绪状态机、注意力追踪）
- **prompt_builder.py**: 无状态的 prompt 构建器
- **emotion_fusion.py**: BERT + LLM 双信号情绪融合
- **emotion_state.py**: 二维 OU 过程情绪状态机

### memory/ - 记忆系统
- 分级记忆管理（L1 工作记忆、L2 压缩摘要、L3 保护区、L4 长期记忆）
- 基于 Qdrant + Mem0 的向量检索

### attention/ - 注意力系统
- 群聊场景的注意力追踪（用户级冷却、@ 提及追踪）
- 主动发言决策模块（基于 LLM 的意图判断）

### agent/ - Agent 事件循环
- 持续运行的 daemon thread
- 事件队列处理（用户消息、系统事件）
- 主动发言触发（空闲检测、时间驱动）

### adapters/ - 外部适配器
- QQ 机器人适配层（NapCat + OneBot v11 反向 WebSocket）
- 未来可扩展：Discord、Telegram、微信等

### media/ - 多模态处理
- 图片识别（下载、验证、缩放、base64 编码、缓存）
- TTS 语音合成（CosyVoice zero-shot 音色克隆）
- ASR 语音识别（SenseVoice + 语音情感识别）

### training/ - 训练脚本
- 模型训练、数据标注、内容过滤等
- 与推理系统解耦

### tools/ - 工具脚本
- 独立的命令行工具
- 参数调优、记忆管理、调试等

## 架构优势

### 1. 模块化
- 每个模块职责单一，易于理解和维护
- 新功能可独立开发，不影响其他模块

### 2. 可扩展性
- 新增适配器：在 `adapters/` 下添加新文件
- 新增多媒体处理：在 `media/` 下扩展
- 新增工具：在 `tools/` 下添加脚本

### 3. 向后兼容
- 所有旧代码无需修改
- `src/__init__.py` 提供顶层 re-export
- 各子模块 `__init__.py` 提供模块级 re-export

### 4. 多人格支持（Phase 5）
- `SharedInfra` 单例，所有人格共享 BERT 和 LLM 客户端
- `PersonaInstance` 独立实例，每个人格独立状态
- 线程安全设计，支持并发运行

## 迁移指南

### 对于新代码
使用新的模块化路径：
```python
from src.core.inference_pipeline import NeuroLikePipeline
from src.agent.agent_loop import AgentLoop
```

### 对于旧代码
无需修改，向后兼容路径仍然可用：
```python
from src.inference_pipeline import NeuroLikePipeline  # 仍然有效
from src.agent_loop import AgentLoop                  # 仍然有效
```

### 逐步迁移
建议在新功能开发时使用新路径，旧代码可逐步迁移。

## 测试验证

所有模块已通过以下测试：
- ✅ 新路径 import 测试
- ✅ 向后兼容 import 测试
- ✅ Pipeline 实例化测试
- ✅ QQ bot 启动测试
- ✅ 工具脚本 import 测试

## 相关文档

- [Pipeline 拆分设计文档](../docs/pipeline_refactoring.md)
- [多人格架构设计](../docs/multi_persona_architecture.md)
- [API 文档](../docs/api_reference.md)
