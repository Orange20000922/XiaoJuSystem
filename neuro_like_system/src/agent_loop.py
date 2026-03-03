"""
Agent 事件循环

将 Neuro 从请求-响应模式升级为持续运行的 Agent。
用户输入只是事件队列中的一种事件，Agent 可以根据配置主动发言。

用法：
    from src.agent_loop import AgentLoop, AgentEvent
    loop = AgentLoop(pipeline, agent_config, output_callback=print, proactive_config=proactive_config)
    loop.start()
    loop.push(AgentEvent(type="message", content="你好"))
    # ... 用户不说话时，Agent 可能主动发言
    loop.stop()
"""

import gc
import json
import time
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from queue import Queue, Empty
from threading import Thread, Event
from typing import Callable, List, Dict, Optional

import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.logger import logger
from src.inference_pipeline import ChatMode, NeuroLikePipeline
from configs.model_config import AgentConfig, ProactiveConfig, LLMConfig, LLMProvider


class ProactiveState(Enum):
    """主动性状态"""
    NORMAL = "normal"              # 正常消息驱动
    TRIGGERED = "triggered"        # 已触发主动决策
    WAITING_RESPONSE = "waiting"   # 等待用户回应
    DORMANT = "dormant"            # 休眠（不再主动）


@dataclass
class AgentEvent:
    """Agent 事件"""
    type: str          # "message" | "system"
    content: str       # 消息内容 / 系统事件描述
    chat_mode: ChatMode = ChatMode.PRIVATE
    is_mentioned: bool = True
    timestamp: float = field(default_factory=time.time)
    reply_context: dict = field(default_factory=dict)  # 适配层回复路由信息


class AgentLoop:
    """
    Agent 事件循环。

    在独立线程中持续运行，处理事件队列中的消息，
    并根据 proactive_level 配置决定是否主动发言。
    """

    def __init__(
        self,
        pipeline: NeuroLikePipeline,
        config: AgentConfig,
        output_callback: Callable[[str], None],
        proactive_config: Optional[ProactiveConfig] = None,
    ):
        self.pipeline = pipeline
        self.config = config
        self.output_callback = output_callback
        self.proactive_config = proactive_config or ProactiveConfig(enabled=False)

        self.event_queue: Queue[AgentEvent] = Queue()
        self.last_user_time: float = time.time()
        self.last_proactive_time: float = 0.0
        self.last_proactive_attempt_time: float = 0.0
        self.proactive_state = ProactiveState.NORMAL
        self._stop_event = Event()
        self._thread: Optional[Thread] = None
        self._current_reply_context: dict = {}  # 当前正在处理的事件的回复路由

        # 初始化主动决策模块
        if self.proactive_config.enabled:
            from src.proactive_decision import ProactiveDecisionModule

            # 构造 DeepSeek LLM 配置
            decision_llm_config = LLMConfig(
                provider=LLMProvider(self.proactive_config.decision_provider),
                model=self.proactive_config.decision_model,
                temperature=self.proactive_config.decision_temperature,
                timeout=int(self.proactive_config.decision_timeout),
            )

            self.decision_module = ProactiveDecisionModule(
                llm_config=decision_llm_config,
                config=self.proactive_config,
            )
            logger.info("主动决策模块已启用")
        else:
            self.decision_module = None
            logger.info("主动决策模块未启用")

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self):
        """启动 Agent 事件循环线程"""
        if self.running:
            logger.warning("Agent 循环已在运行")
            return
        self._stop_event.clear()
        self._thread = Thread(target=self._loop, daemon=True, name="agent-loop")
        self._thread.start()
        logger.info(
            f"Agent 循环已启动 "
            f"proactive_level={self.config.proactive_level} "
            f"tick={self.config.tick_interval}s"
        )

    def stop(self):
        """停止 Agent 事件循环并写入长期记忆"""
        if not self.running:
            return
        logger.info("正在停止 Agent 循环...")
        self._stop_event.set()
        self._thread.join(timeout=self.config.tick_interval + 5)
        self._thread = None
        # 写入长期记忆
        self.pipeline.close()
        logger.info("Agent 循环已停止")

    def push(self, event: AgentEvent):
        """外部接口：往事件队列塞事件"""
        self.event_queue.put(event)

    # ── 内部方法 ──────────────────────────────────────────────────────────

    def _loop(self):
        """事件循环主体（在独立线程运行）"""
        while not self._stop_event.is_set():
            event = self._poll_event()

            if event:
                if event.type == "message":
                    self._handle_message(event)
                elif event.type == "system":
                    self._handle_system_event(event)
            else:
                self._check_proactive_triggers()

    def _poll_event(self) -> Optional[AgentEvent]:
        """从队列取事件，带 timeout"""
        try:
            return self.event_queue.get(timeout=self.config.tick_interval)
        except Empty:
            return None

    def _handle_message(self, event: AgentEvent):
        """处理用户消息事件"""
        self.last_user_time = time.time()
        self._current_reply_context = event.reply_context

        # 状态机转换：收到用户消息后重置为 NORMAL
        if self.proactive_state in (ProactiveState.WAITING_RESPONSE, ProactiveState.DORMANT):
            logger.info(f"收到用户消息，状态从 {self.proactive_state.value} 重置为 normal")
            self.proactive_state = ProactiveState.NORMAL

        # 群聊场景强制启用融合（注意力判断需要高准确率）
        # 私聊场景使用配置默认值（None 会让 chat() 自动读取 config.use_by_default）
        use_fusion = True if event.chat_mode == ChatMode.GROUP else None

        try:
            result = self.pipeline.chat(
                event.content,
                is_mentioned=event.is_mentioned,
                chat_mode=event.chat_mode,
                use_fusion=use_fusion,
            )

            if result["should_respond"] and result["response"]:
                self.output_callback(result["response"])

        except Exception as e:
            logger.error(f"处理消息失败: {e}")

    def _handle_system_event(self, event: AgentEvent):
        """处理系统事件（低级主动：由外部事件触发）"""
        if self.config.proactive_level == "off":
            return

        now = time.time()
        if now - self.last_proactive_time < self.config.proactive_interval_seconds:
            logger.debug("主动发言冷却中，跳过系统事件触发")
            return

        response = self.pipeline.generate_proactive(
            trigger=event.content,
            chat_mode=event.chat_mode,
        )
        if response:
            self.last_proactive_time = now
            self.output_callback(response)

    def _check_proactive_triggers(self):
        """根据 proactive_level 和状态机检查是否需要主动发言"""
        # 如果启用了主动决策模块，优先使用新的决策逻辑（忽略 proactive_level）
        if self.decision_module:
            self._check_proactive_with_decision()
            return

        # 否则使用原有的简单逻辑
        level = self.config.proactive_level

        if level == "off":
            return

        if level == "low":
            # low 级别只响应系统事件（已在 _handle_system_event 中处理），
            # 不做空闲兜底
            return

        if level == "medium":
            self._check_idle_trigger()

    def _check_proactive_with_decision(self):
        """使用主动决策模块的触发逻辑"""
        if self.proactive_state == ProactiveState.DORMANT:
            # 休眠状态不再主动
            return

        if self.proactive_state == ProactiveState.WAITING_RESPONSE:
            # 检查是否超时无回应
            self._check_response_timeout()
            return

        # NORMAL 状态：检查时间驱动触发
        if self.proactive_state == ProactiveState.NORMAL:
            self._check_time_driven_trigger()

    def _check_time_driven_trigger(self):
        """时间驱动：空闲 N 小时后触发主动决策"""
        now = time.time()
        idle_seconds = now - self.last_user_time
        idle_hours = idle_seconds / 3600

        if idle_hours < self.proactive_config.idle_trigger_hours:
            return

        # 冷却检查
        if now - self.last_proactive_time < self.proactive_config.min_interval_seconds:
            return

        logger.info(f"空闲 {idle_hours:.1f} 小时，触发主动决策")

        # 立即更新时间，避免并发请求
        self.last_proactive_time = now

        # 收集输入信号
        recent_convs = self._get_recent_conversations()
        emotion_state = self._load_emotion_state()
        l4_memories = self._get_l4_memories()

        # 调用决策模块
        decision = self.decision_module.decide(
            recent_conversations=recent_convs,
            emotion_state=emotion_state,
            l4_memories=l4_memories,
            current_time=time.strftime("%Y-%m-%d %H:%M"),
        )

        if not decision or not decision.should_respond:
            logger.info("决策模块判断不需要主动发言")
            return

        if decision.confidence < self.proactive_config.confidence_threshold:
            logger.debug(f"决策置信度不足: {decision.confidence:.2f}")
            return

        # 生成回复（注入决策指导）
        response = self.pipeline.generate_proactive(
            trigger=f"[主动发言] {decision.reason}",
            decision_hint=decision,
        )

        if response:
            self.last_proactive_time = now
            self.last_proactive_attempt_time = now
            self.proactive_state = ProactiveState.WAITING_RESPONSE
            logger.info(f"主动发言成功，进入 WAITING_RESPONSE 状态")
            self.output_callback(response)

    def _check_response_timeout(self):
        """检查主动发言后是否超时无回应"""
        now = time.time()
        wait_minutes = (now - self.last_proactive_attempt_time) / 60

        if wait_minutes < self.proactive_config.response_wait_minutes:
            return

        logger.info(
            f"主动发言后 {wait_minutes:.1f} 分钟无回应，"
            f"进入 DORMANT 状态（仅响应用户消息）"
        )
        self.proactive_state = ProactiveState.DORMANT

    def _get_recent_conversations(self) -> List[Dict]:
        """从 pipeline.memory.working_memory 获取最近对话"""
        limit = self.proactive_config.recent_turns_limit
        if hasattr(self.pipeline.memory, 'working_memory'):
            return [
                {
                    "user_input": turn.user_input,
                    "response": turn.response,
                    "emotion": turn.emotion,
                    "intensity": turn.intensity,
                }
                for turn in self.pipeline.memory.working_memory[-limit:]
            ]
        return []

    def _load_emotion_state(self) -> Dict:
        """读取 data/emotion_state.json"""
        path = project_root / "data" / "emotion_state.json"
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception as e:
                logger.warning(f"读取 emotion_state.json 失败: {e}")
        return {}

    def _get_l4_memories(self) -> List[str]:
        """从 Mem0 搜索情感相关记忆"""
        if not hasattr(self.pipeline.memory, 'mem0'):
            return []

        try:
            # 搜索带情感标签的记忆
            results = self.pipeline.memory.mem0.search(
                query="情感 情绪 心情",
                limit=self.proactive_config.l4_memory_limit,
                user_id=self.pipeline.memory.config.user_id,
            )
            # Mem0 返回的是字符串列表或字典列表，需要兼容处理
            memories = []
            for r in results:
                if isinstance(r, str):
                    memories.append(r)
                elif isinstance(r, dict) and "memory" in r:
                    memories.append(r["memory"])
            return memories
        except Exception as e:
            logger.warning(f"搜索 L4 记忆失败: {e}")
            return []

    def _check_idle_trigger(self):
        """medium 级别：空闲超时触发主动发言"""
        now = time.time()
        idle_seconds = now - self.last_user_time

        if idle_seconds < self.config.idle_threshold_seconds:
            return

        if now - self.last_proactive_time < self.config.proactive_interval_seconds:
            return

        idle_minutes = int(idle_seconds // 60)
        if idle_minutes > 0:
            trigger = f"对话已空闲{idle_minutes}分钟"
        else:
            trigger = f"对话已空闲{int(idle_seconds)}秒"

        logger.info(f"空闲触发主动发言: {trigger}")
        response = self.pipeline.generate_proactive(trigger=trigger)

        if response:
            self.last_proactive_time = now
            self.output_callback(response)
