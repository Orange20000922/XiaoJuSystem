# 音频处理与 Live2D 集成方案

基于 Airi 项目的架构分析，为 NeuroLikeSystem 设计 Python 版本的音频处理和 Live2D 集成方案。

---

## 一、Airi 架构分析

### 1.1 音频处理架构 (pipelines-audio)

**核心组件**：
- **SpeechPipeline**：语音合成管道编排器
  - TTS 请求队列管理
  - 优先级调度（priority-based）
  - 播放控制（queue/interrupt/replace 三种行为）
  - 事件驱动架构（Eventa 事件系统）

- **关键特性**：
  - **Intent 系统**：每个发言意图（intent）有独立的优先级、所有者、行为策略
  - **流式处理**：TextToken → TextSegment → TTS → Playback
  - **中断控制**：支持按 intent/owner 停止播放
  - **生命周期管理**：onStart/onEnd/onInterrupt/onReject 事件

**数据流**：
```
TextToken Stream → Segmenter → TTS Chunker → Audio Playback
                                    ↓
                            Priority Resolver
                                    ↓
                        Queue / Interrupt / Replace
```

### 1.2 Live2D 架构

**核心组件**：
- **model-driver-lipsync**：口型同步驱动
  - 基于 `wlipsync` (Web Audio Worklet)
  - 实时音频分析 → AEIOUS 元音权重
  - 音量加权 + 平滑处理

- **stage-ui-live2d**：Live2D 渲染层
  - 基于 `pixi-live2d-display` (PixiJS v6)
  - Vue 3 组件封装
  - 表情常量映射（emotions.ts）
  - 眼动控制（eye-motions.ts）
  - ZIP 模型加载器

**口型同步算法**：
```typescript
// 1. wLipSync 输出 AEIOUS 权重（S=静音）
// 2. 映射到 AEIOU（S → I，避免硬切）
// 3. 音量加权：weight = min(cap, raw_weight * (volume * scale)^exponent)
// 4. 平滑处理：lerp(prev, curr, alpha) 每 40ms 更新
// 5. 输出 getMouthOpen() 和 getVowelWeights()
```

---

## 二、Python 实现方案

### 2.1 技术选型

#### 音频处理
| 功能 | Python 库 | 说明 |
|---|---|---|
| TTS | `edge-tts` / `pyttsx3` / `coqui-tts` | edge-tts 免费且质量高 |
| 音频播放 | `pygame.mixer` / `sounddevice` | pygame 简单，sounddevice 低延迟 |
| 音频分析 | `librosa` / `pyaudio` | 实时频谱分析 |
| 口型同步 | `librosa.feature.mfcc` | 提取元音特征 |
| 异步队列 | `asyncio.Queue` | 原生异步支持 |

#### Live2D 渲染
| 方案 | 技术栈 | 优缺点 |
|---|---|---|
| **方案 A：Web 前端** | FastAPI + WebSocket + Vue3 | ✅ 可复用 Airi 的 pixi-live2d-display<br>✅ 跨平台<br>❌ 需要浏览器 |
| **方案 B：PyQt/PySide** | Qt WebEngine + pixi-live2d | ✅ 独立应用<br>❌ 打包体积大 |
| **方案 C：纯 Python** | `live2d-py` (社区库) | ✅ 纯 Python<br>❌ 功能有限，维护少 |

**推荐**：方案 A（Web 前端），理由：
- 可直接参考 Airi 的成熟实现
- 适合 QQ 机器人场景（可做 Web 控制面板）
- 未来可扩展为直播推流

### 2.2 模块设计

#### 2.2.1 音频处理模块 (`src/audio_pipeline.py`)

```python
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional
import asyncio

class PlaybackBehavior(Enum):
    QUEUE = "queue"        # 排队播放
    INTERRUPT = "interrupt"  # 中断当前
    REPLACE = "replace"    # 替换队列

@dataclass
class AudioIntent:
    intent_id: str
    text: str
    priority: int
    owner_id: str
    behavior: PlaybackBehavior
    created_at: float

class AudioPipeline:
    """语音合成与播放管道"""

    def __init__(self, tts_provider: str = "edge-tts"):
        self.tts_provider = tts_provider
        self.intent_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self.active_intent: Optional[AudioIntent] = None
        self.is_playing = False

    async def speak(
        self,
        text: str,
        priority: int = 0,
        behavior: PlaybackBehavior = PlaybackBehavior.QUEUE,
        owner_id: str = "default"
    ) -> str:
        """发起语音合成请求"""
        intent = AudioIntent(
            intent_id=self._generate_id(),
            text=text,
            priority=priority,
            owner_id=owner_id,
            behavior=behavior,
            created_at=time.time()
        )

        if behavior == PlaybackBehavior.INTERRUPT:
            await self.stop_current("interrupted by higher priority")
        elif behavior == PlaybackBehavior.REPLACE:
            await self.stop_all("replaced by new intent")

        await self.intent_queue.put((-priority, intent))  # 负数实现大顶堆
        return intent.intent_id

    async def _process_loop(self):
        """处理队列循环"""
        while True:
            _, intent = await self.intent_queue.get()
            self.active_intent = intent

            try:
                # TTS 合成
                audio_data = await self._synthesize(intent.text)

                # 播放音频
                await self._play_audio(audio_data, intent)

            except Exception as e:
                logger.error(f"Intent {intent.intent_id} failed: {e}")
            finally:
                self.active_intent = None

    async def _synthesize(self, text: str) -> bytes:
        """TTS 合成（edge-tts 示例）"""
        import edge_tts

        communicate = edge_tts.Communicate(text, voice="zh-CN-XiaoxiaoNeural")
        audio_chunks = []

        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_chunks.append(chunk["data"])

        return b"".join(audio_chunks)

    async def _play_audio(self, audio_data: bytes, intent: AudioIntent):
        """播放音频并触发口型同步"""
        # 保存临时文件
        temp_path = f"temp_audio_{intent.intent_id}.mp3"
        with open(temp_path, "wb") as f:
            f.write(audio_data)

        # 播放音频（pygame 示例）
        pygame.mixer.music.load(temp_path)
        pygame.mixer.music.play()

        # 实时分析音频用于口型同步
        await self._analyze_for_lipsync(temp_path, intent)

        # 等待播放完成
        while pygame.mixer.music.get_busy():
            await asyncio.sleep(0.1)

        os.remove(temp_path)
```

#### 2.2.2 口型同步模块 (`src/lipsync_driver.py`)

```python
import librosa
import numpy as np
from typing import Dict

class LipSyncDriver:
    """实时口型同步驱动"""

    def __init__(self,
                 cap: float = 0.7,
                 volume_scale: float = 0.9,
                 volume_exponent: float = 0.7,
                 update_interval_ms: int = 40):
        self.cap = cap
        self.volume_scale = volume_scale
        self.volume_exponent = volume_exponent
        self.update_interval = update_interval_ms / 1000.0

        self.last_mouth_open = 0.0
        self.smoothed_mouth_open = 0.0

    async def analyze_audio_stream(self, audio_path: str):
        """分析音频流并生成口型数据"""
        # 加载音频
        y, sr = librosa.load(audio_path, sr=22050)

        # 计算帧数（每 40ms 一帧）
        hop_length = int(sr * self.update_interval)

        # 提取 MFCC 特征（用于元音识别）
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_length)

        # 计算 RMS 能量（音量）
        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]

        # 逐帧生成口型数据
        for i in range(mfcc.shape[1]):
            vowel_weights = self._mfcc_to_vowels(mfcc[:, i])
            volume = rms[i]

            mouth_open = self._compute_mouth_open(vowel_weights, volume)

            # 发送到 Live2D 渲染层
            await self._emit_lipsync_data({
                "vowels": vowel_weights,
                "mouth_open": mouth_open,
                "timestamp": i * self.update_interval
            })

            await asyncio.sleep(self.update_interval)

    def _mfcc_to_vowels(self, mfcc_frame: np.ndarray) -> Dict[str, float]:
        """MFCC 特征映射到 AEIOU 元音权重（简化版）"""
        # 这里需要训练一个小模型或使用启发式规则
        # 简化示例：基于 MFCC 系数的线性映射

        # 归一化
        mfcc_norm = (mfcc_frame - mfcc_frame.mean()) / (mfcc_frame.std() + 1e-8)

        # 简单映射（实际需要更复杂的算法）
        vowels = {
            "A": max(0, mfcc_norm[1] * 0.5),
            "E": max(0, mfcc_norm[2] * 0.5),
            "I": max(0, mfcc_norm[3] * 0.5),
            "O": max(0, mfcc_norm[4] * 0.5),
            "U": max(0, mfcc_norm[5] * 0.5),
        }

        # 归一化到 [0, 1]
        total = sum(vowels.values()) + 1e-8
        return {k: v / total for k, v in vowels.items()}

    def _compute_mouth_open(self, vowels: Dict[str, float], volume: float) -> float:
        """计算嘴巴张开度"""
        # 音量加权
        amp = min((volume * self.volume_scale), 1.0) ** self.volume_exponent

        # 取最大元音权重
        max_vowel = max(vowels.values())
        raw_mouth = min(self.cap, max_vowel * amp)

        # 平滑处理（简单 lerp）
        alpha = 0.3
        self.smoothed_mouth_open += (raw_mouth - self.smoothed_mouth_open) * alpha

        return self.smoothed_mouth_open

    async def _emit_lipsync_data(self, data: dict):
        """发送口型数据到渲染层（通过 WebSocket）"""
        # 这里会连接到 Live2D 前端
        pass
```

#### 2.2.3 Live2D 渲染层（Web 前端）

**技术栈**：FastAPI + WebSocket + Vue 3 + pixi-live2d-display

**架构**：
```
Python Backend (FastAPI)
    ↓ WebSocket
Vue 3 Frontend
    ↓
pixi-live2d-display (直接复用 Airi 的实现)
```

**后端 WebSocket 服务** (`src/live2d_server.py`):
```python
from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
import asyncio

app = FastAPI()

# 挂载前端静态文件
app.mount("/static", StaticFiles(directory="frontend/dist"), name="static")

class Live2DManager:
    def __init__(self):
        self.connections: list[WebSocket] = []

    async def broadcast(self, data: dict):
        """广播到所有连接的前端"""
        for ws in self.connections:
            try:
                await ws.send_json(data)
            except:
                self.connections.remove(ws)

live2d_manager = Live2DManager()

@app.websocket("/ws/live2d")
async def live2d_websocket(websocket: WebSocket):
    await websocket.accept()
    live2d_manager.connections.append(websocket)

    try:
        while True:
            # 保持连接
            await websocket.receive_text()
    except:
        live2d_manager.connections.remove(websocket)

# 从 AudioPipeline 调用
async def send_lipsync_to_frontend(data: dict):
    await live2d_manager.broadcast({
        "type": "lipsync",
        "data": data
    })
```

**前端 Vue 组件**（直接参考 Airi 的 `Live2D.vue`）:
```vue
<template>
  <div ref="canvasContainer" class="live2d-container"></div>
</template>

<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { Application } from '@pixi/app'
import { Live2DModel } from 'pixi-live2d-display'

const canvasContainer = ref<HTMLDivElement>()
let app: Application
let model: Live2DModel
let ws: WebSocket

onMounted(async () => {
  // 初始化 PixiJS
  app = new Application({
    view: document.createElement('canvas'),
    transparent: true,
    width: 800,
    height: 600
  })
  canvasContainer.value?.appendChild(app.view)

  // 加载 Live2D 模型
  model = await Live2DModel.from('/models/neuro/neuro.model3.json')
  app.stage.addChild(model)

  // 连接 WebSocket
  ws = new WebSocket('ws://localhost:8000/ws/live2d')
  ws.onmessage = (event) => {
    const msg = JSON.parse(event.data)
    if (msg.type === 'lipsync') {
      updateLipSync(msg.data)
    }
  }
})

function updateLipSync(data: any) {
  // 更新口型参数
  model.internalModel.coreModel.setParameterValueById(
    'ParamMouthOpenY',
    data.mouth_open
  )

  // 更新元音形状（如果模型支持）
  const vowels = data.vowels
  if (vowels) {
    model.internalModel.coreModel.setParameterValueById('ParamMouthForm',
      vowels.A * 1.0 + vowels.O * 0.5)  // 简化映射
  }
}
</script>
```

### 2.3 与现有系统集成

#### 修改 `inference_pipeline.py`

```python
class NeuroLikePipeline:
    def __init__(self, config: AppConfig):
        # ... 现有初始化 ...

        # 新增音频管道
        self.audio_pipeline = AudioPipeline(tts_provider="edge-tts")
        self.lipsync_driver = LipSyncDriver()

    async def chat_with_voice(self, user_input: str, **kwargs) -> dict:
        """带语音输出的对话"""
        # 1. 文本生成（现有逻辑）
        result = self.chat(user_input, **kwargs)

        # 2. 语音合成
        intent_id = await self.audio_pipeline.speak(
            text=result["response"],
            priority=1,
            behavior=PlaybackBehavior.INTERRUPT if kwargs.get("urgent") else PlaybackBehavior.QUEUE
        )

        return {
            **result,
            "audio_intent_id": intent_id
        }
```

#### 修改 `qq_adapter.py`

```python
class QQBotAdapter:
    def __init__(self, pipeline: NeuroLikePipeline, config: QQBotConfig):
        # ... 现有初始化 ...

        # 启动 Live2D 服务器（可选）
        if config.enable_live2d:
            self.live2d_server = Live2DServer(port=8000)
            asyncio.create_task(self.live2d_server.start())

    async def _on_message(self, data: dict):
        # ... 现有逻辑 ...

        # 如果启用语音，使用 chat_with_voice
        if self.config.enable_voice:
            result = await self.pipeline.chat_with_voice(
                user_input=message,
                is_mentioned=is_at,
                chat_mode=ChatMode.GROUP
            )
        else:
            result = self.pipeline.chat(...)
```

---

## 三、语音识别（STT/ASR）方案调研

### 3.1 开源方案

#### SenseVoice（阿里，推荐 — 中文最强）

- **项目**：https://github.com/FunAudioLLM/SenseVoice
- **模型**：SenseVoice-Small（HuggingFace: `FunAudioLLM/SenseVoiceSmall`）
- **中文支持**：在 AISHELL-1/2、WenetSpeech 上超过 Whisper
- **速度**：处理 10 秒音频仅需 **70ms**，比 Whisper-Large 快 **15 倍**
- **特殊能力**：
  - 自带**语音情感识别（SER）** — 可与 BERT 情绪系统交叉验证/融合
  - 自带**音频事件检测（AED）** — 识别笑声、掌声、哭泣、咳嗽等
  - 支持 50+ 语言
- **许可**：开源免费

#### FunASR（阿里，全栈工具包）

- **项目**：https://github.com/modelscope/FunASR
- **功能**：ASR + VAD（语音端点检测）+ 标点恢复 + 说话人分离
- **最新**：Fun-ASR-Nano-2512，支持 31 种语言低延迟实时转写
- **代表模型**：Paraformer（非自回归，高精度高效率）
- **与 SenseVoice 的关系**：FunASR 可集成 SenseVoice 模型，SenseVoice 负责理解，FunASR 负责工程化部署

#### OpenAI Whisper

- **项目**：https://github.com/openai/whisper
- **模型大小**：tiny(39M) ~ large-v3(1.5B)
- **推荐变体**：
  - **Whisper Large V3 Turbo**：比原版快 5.4x，解码器层 32→4
  - **WhisperX**：加了实时能力 + 词级时间戳 + 说话人分离，速度 4x
- **中文表现**：可用但不如 SenseVoice

#### Voxtral（Mistral AI，新秀）

- 多项指标超 Whisper 和 GPT-4o-mini-transcribe
- Apache 2.0 许可，32K token 上下文窗口
- 可处理最长 30 分钟音频

#### 其他

| 模型 | 特点 | 适合场景 |
|---|---|---|
| NVIDIA Parakeet TDT | 极致速度（RTFx > 2000），仅英文 | 英文实时 |
| Moonshine | 超小模型 | 边缘设备、离线 |
| RapidASR | 基于 FunASR + ONNX，开箱即用 | 快速集成 |

### 3.2 付费 API

| 服务商 | 价格 | 特点 |
|---|---|---|
| OpenAI gpt-4o-mini-transcribe | ~$0.006/分钟 | 错误率最低，OpenAI 官方推荐 |
| Deepgram Nova-3 | ~$4.3/千分钟 | 延迟 < 300ms，实时 |
| AssemblyAI Universal-2 | ~$0.012/分钟 | 流式准确率最高（14.5% WER），自带情感分析 |
| Google Chirp 2 | 按用量计费 | 准确率基准第一，100+ 语言 |
| Azure Speech | 按用量计费 | 140+ 语言，企业级 |
| 讯飞 ASR | 国内有免费额度 | 中文体验好 |

### 3.3 推荐方案：SenseVoice

**理由**：
1. 中文识别最好 — QQ 机器人场景以中文为主
2. 自带情感识别 — 与 BERT 情绪系统互补，双信号融合更准确
3. 极低延迟（70ms/10s 音频）— 适合实时对话
4. 开源免费 — 本地部署，无 API 费用
5. 阿里生态 — FunASR + SenseVoice + CosyVoice（TTS），一整套方案

**集成架构**：
```
语音输入 → SenseVoice（ASR + 情感识别）
               ↓                ↓
           文字文本         语音情感标签
               ↓                ↓
           现有 pipeline    与 BERT 情绪融合
               ↓
           LLM 生成回复
               ↓
           edge-tts 语音输出
```

### 3.4 参考链接

- SenseVoice: https://github.com/FunAudioLLM/SenseVoice
- SenseVoice HuggingFace: https://huggingface.co/FunAudioLLM/SenseVoiceSmall
- FunASR: https://github.com/modelscope/FunASR
- OpenAI Whisper: https://github.com/openai/whisper
- Voxtral: https://apidog.com/blog/voxtral-open-source-whisper-alternative/
- RapidASR: https://github.com/RapidAI/RapidASR
- Best STT APIs 2026: https://deepgram.com/learn/best-speech-to-text-apis-2026
- Open Source STT Benchmarks: https://northflank.com/blog/best-open-source-speech-to-text-stt-model-in-2026-benchmarks

---

## 五、实施路线图

### Phase 1：音频处理基础（1-2 周）
- [ ] 实现 `AudioPipeline` 基础框架
- [ ] 集成 edge-tts
- [ ] 实现优先级队列和播放控制
- [ ] 单元测试

### Phase 2：口型同步（2-3 周）
- [ ] 实现 `LipSyncDriver`
- [ ] MFCC → 元音映射算法
- [ ] 实时音频分析
- [ ] 与 AudioPipeline 集成

### Phase 3：Live2D 渲染（2-3 周）
- [ ] FastAPI WebSocket 服务器
- [ ] Vue 3 前端项目搭建
- [ ] 集成 pixi-live2d-display
- [ ] 口型数据实时传输

### Phase 4：系统集成（1 周）
- [ ] 修改 `inference_pipeline.py`
- [ ] 修改 `qq_adapter.py`
- [ ] 配置文件扩展
- [ ] 端到端测试

### Phase 5：优化与扩展（持续）
- [ ] 表情驱动（基于 BERT 情绪）
- [ ] 眼动控制
- [ ] 多模型支持
- [ ] 性能优化

---

## 六、配置扩展

在 `config.json` 中新增：

```json
{
  "audio": {
    "enabled": true,
    "tts_provider": "edge-tts",
    "voice": "zh-CN-XiaoxiaoNeural",
    "default_priority": 0,
    "default_behavior": "queue"
  },

  "lipsync": {
    "enabled": true,
    "cap": 0.7,
    "volume_scale": 0.9,
    "volume_exponent": 0.7,
    "update_interval_ms": 40,
    "lerp_window_ms": 120
  },

  "live2d": {
    "enabled": true,
    "server_port": 8000,
    "model_path": "./models/neuro/neuro.model3.json",
    "canvas_width": 800,
    "canvas_height": 600
  }
}
```

---

## 七、依赖库

新增 `requirements.txt`：

```txt
# 音频处理
edge-tts>=6.1.0
pygame>=2.5.0
librosa>=0.10.0
soundfile>=0.12.0

# Live2D 服务器
fastapi>=0.109.0
uvicorn[standard]>=0.27.0
websockets>=12.0

# 前端构建（可选，如果本地开发）
# npm install -g @vue/cli
```

---

## 八、注意事项

### 6.1 性能考虑
- **TTS 延迟**：edge-tts 网络延迟 ~500ms，考虑本地 TTS（coqui-tts）
- **口型同步精度**：MFCC 方法是简化版，精度不如 wLipSync，可考虑训练专用模型
- **WebSocket 带宽**：40ms 更新频率，每秒 25 帧，数据量小（~1KB/s）

### 6.2 兼容性
- **Live2D SDK**：pixi-live2d-display 仅支持 Cubism 3.x/4.x 模型
- **浏览器要求**：需要支持 WebGL 的现代浏览器
- **跨平台**：Web 方案天然跨平台，但需要用户打开浏览器

### 6.3 替代方案
如果不想用 Web 前端，可以考虑：
- **VTube Studio API**：通过 WebSocket 控制 VTube Studio（需要购买软件）
- **VSeeFace**：开源虚拟主播软件，支持 OSC 协议控制
- **纯 Python**：使用 `live2d-py` + PyQt，但功能有限

---

## 九、参考资源

- Airi 项目：https://github.com/moeru-ai/airi
- pixi-live2d-display：https://github.com/guansss/pixi-live2d-display
- wLipSync：https://github.com/hecomi/wLipSync
- edge-tts：https://github.com/rany2/edge-tts
- librosa 文档：https://librosa.org/doc/latest/index.html
