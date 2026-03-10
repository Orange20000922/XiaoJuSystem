"""
PersonaScheduler — 多人格调度器

管理多个 (PersonaInstance + AgentLoop) 对的生命周期、事件路由和 LLM 并发控制。
支持运行时热插拔（register / unregister）。

用法：
    infra = SharedInfra.from_app_config(primary_config)
    scheduler = PersonaScheduler(infra, SchedulerConfig(max_concurrent_llm=3))

    # 注册人格
    scheduler.register(config_a, context_ids={"group_123"})
    scheduler.register_default(config_b)  # catch-all

    # 启动
    scheduler.start_all()

    # 路由事件
    scheduler.dispatch(event)  # 自动路由到正确的 persona

    # 运行时热插拔
    scheduler.register(config_c, context_ids={"group_456"})  # 立即上线
    scheduler.unregister("小明")  # 优雅关闭

    # 停止
    scheduler.stop_all()
"""

import fnmatch
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from threading import Thread, Event, BoundedSemaphore
from typing import Callable, Dict, FrozenSet, List, Optional, Set, Tuple

import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.logger import logger
from src.core.shared_infra import SharedInfra
from src.core.persona import PersonaInstance
from src.agent.agent_loop import AgentLoop, AgentEvent
from configs.model_config import SchedulerConfig
from configs.config_loader import AppConfig


@dataclass
class ManagedPersona:
    """被调度器管理的人格实例"""
    name: str                          # personality.name（唯一键）
    persona: PersonaInstance
    agent_loop: AgentLoop
    context_ids: Set[str]              # 精确匹配 context_id（"group_123"）
    context_patterns: List[str]        # fnmatch 通配符（"private_*"）
    config_source: str = ""            # 来源 config 路径（调试用）
    started_at: float = 0.0

    def matches(self, context_id: str) -> bool:
        """检查此 persona 是否负责处理给定 context_id"""
        if context_id in self.context_ids:
            return True
        return any(fnmatch.fnmatch(context_id, p) for p in self.context_patterns)


class PersonaScheduler:
    """
    多人格调度器。

    职责：
    - 管理多个 PersonaInstance + AgentLoop 的生命周期
    - 按 context_id 路由事件到正确的 persona
    - 通过 BoundedSemaphore 限制 LLM 并发调用数
    - 健康监控（检测死线程）
    - 支持运行时热插拔
    """

    def __init__(
        self,
        infra: SharedInfra,
        config: Optional[SchedulerConfig] = None,
    ):
        self._infra = infra
        self._config = config or SchedulerConfig()

        # persona 注册表（name → ManagedPersona）
        self._personas: Dict[str, ManagedPersona] = {}
        # 路由缓存（context_id → persona name）
        self._context_cache: Dict[str, str] = {}
        # 保护 _personas 和 _context_cache 的锁（热插拔线程安全）
        self._lock = threading.Lock()

        # LLM 并发信号量，注入到 SharedInfra
        self._llm_semaphore = BoundedSemaphore(self._config.max_concurrent_llm)
        infra.llm_semaphore = self._llm_semaphore

        # 健康检查
        self._stop_event = Event()
        self._health_thread: Optional[Thread] = None
        self._alert_callback: Optional[Callable[[str], None]] = None

        # 状态
        self._running = False

        logger.info(
            f"PersonaScheduler 初始化: "
            f"max_concurrent_llm={self._config.max_concurrent_llm} "
            f"health_interval={self._config.health_check_interval}s"
        )

    # ── 注册 / 注销 ──────────────────────────────────────────────────────

    def register(
        self,
        app_config: AppConfig,
        context_ids: Optional[Set[str]] = None,
        context_patterns: Optional[List[str]] = None,
        callback_factory: Optional[Callable[[AgentLoop], Callable[[str], None]]] = None,
        config_source: str = "",
    ) -> ManagedPersona:
        """
        创建并注册一个 persona。

        Args:
            app_config: 该 persona 的完整配置
            context_ids: 精确匹配的 context_id 集合
            context_patterns: fnmatch 通配符列表
            callback_factory: 接受 AgentLoop 返回 output_callback 的工厂函数
            config_source: 配置文件路径（调试用）

        Returns:
            ManagedPersona 实例

        Raises:
            ValueError: persona 名称已存在
        """
        name = app_config.personality.name

        with self._lock:
            if name in self._personas:
                raise ValueError(f"persona '{name}' 已注册，请先 unregister")

        # 创建 PersonaInstance（可能耗时，在锁外执行）
        persona = PersonaInstance.from_app_config(self._infra, app_config)

        # 创建 AgentLoop（先用 dummy callback，后面替换）
        dummy_cb = lambda text: logger.warning(f"[{name}] 无 output_callback: {text[:50]}")
        loop = AgentLoop(
            pipeline=persona,
            config=app_config.agent,
            output_callback=dummy_cb,
            proactive_config=app_config.proactive,
        )

        # 用 callback_factory 创建真正的 callback
        if callback_factory is not None:
            loop.output_callback = callback_factory(loop)

        managed = ManagedPersona(
            name=name,
            persona=persona,
            agent_loop=loop,
            context_ids=context_ids or set(),
            context_patterns=context_patterns or [],
            config_source=config_source,
        )

        with self._lock:
            if name in self._personas:
                # 竞争检查
                persona.close()
                raise ValueError(f"persona '{name}' 已注册（竞争）")
            self._personas[name] = managed
            self._context_cache.clear()

        logger.info(
            f"注册 persona '{name}': "
            f"contexts={context_ids or set()} "
            f"patterns={context_patterns or []} "
            f"source={config_source}"
        )

        # 热插拔：如果调度器已在运行，立即启动新 persona 的 AgentLoop
        if self._running:
            managed.started_at = time.time()
            loop.start()
            logger.info(f"热插拔: persona '{name}' 的 AgentLoop 已启动")

        return managed

    def register_default(
        self,
        app_config: AppConfig,
        callback_factory: Optional[Callable[[AgentLoop], Callable[[str], None]]] = None,
        config_source: str = "",
    ) -> ManagedPersona:
        """注册一个 catch-all persona（处理所有未匹配的 context_id）"""
        return self.register(
            app_config=app_config,
            context_patterns=["*"],
            callback_factory=callback_factory,
            config_source=config_source,
        )

    def unregister(self, name: str) -> bool:
        """
        注销并优雅关闭一个 persona（热插拔）。

        会调用 AgentLoop.stop()（内部调 persona.close() 持久化记忆）。

        Args:
            name: persona 名称

        Returns:
            True 成功，False 名称不存在
        """
        with self._lock:
            managed = self._personas.pop(name, None)
            if managed is None:
                logger.warning(f"unregister: persona '{name}' 不存在")
                return False
            self._context_cache.clear()

        # 在锁外执行 stop（可能耗时：写入记忆、关闭连接）
        logger.info(f"正在注销 persona '{name}'...")
        try:
            managed.agent_loop.stop()
        except Exception as e:
            logger.error(f"注销 persona '{name}' 时 AgentLoop.stop() 失败: {e}")

        logger.info(f"persona '{name}' 已注销")
        return True

    # ── 事件路由 ──────────────────────────────────────────────────────────

    def dispatch(self, event: AgentEvent) -> bool:
        """
        将事件路由到正确 persona 的 AgentLoop。

        Returns:
            True 路由成功，False 无匹配 persona
        """
        context_id = event.context_id or ""

        # 1. 查缓存
        with self._lock:
            cached_name = self._context_cache.get(context_id)
            if cached_name and cached_name in self._personas:
                self._personas[cached_name].agent_loop.push(event)
                return True

        # 2. 遍历匹配
        matched = self.resolve_persona(context_id)
        if matched:
            with self._lock:
                self._context_cache[context_id] = matched.name
            matched.agent_loop.push(event)
            return True

        # 3. 尝试 default_persona
        if self._config.default_persona:
            with self._lock:
                default = self._personas.get(self._config.default_persona)
            if default:
                with self._lock:
                    self._context_cache[context_id] = default.name
                default.agent_loop.push(event)
                return True

        logger.warning(
            f"路由失败: context_id='{context_id}' 无匹配 persona"
        )
        return False

    def resolve_persona(self, context_id: str) -> Optional[ManagedPersona]:
        """查找负责某 context_id 的 persona（供命令处理器等外部使用）"""
        with self._lock:
            # 精确匹配优先
            for mp in self._personas.values():
                if context_id in mp.context_ids:
                    return mp
            # 通配符匹配
            for mp in self._personas.values():
                if any(fnmatch.fnmatch(context_id, p) for p in mp.context_patterns):
                    return mp
        return None

    # ── 生命周期 ─────────────────────────────────────────────────────────

    def start_all(self):
        """启动所有已注册 persona 的 AgentLoop + 健康监控线程"""
        if self._running:
            logger.warning("PersonaScheduler 已在运行")
            return

        self._running = True
        self._stop_event.clear()

        with self._lock:
            for name, mp in self._personas.items():
                if not mp.agent_loop.running:
                    mp.started_at = time.time()
                    mp.agent_loop.start()
                    logger.info(f"启动 AgentLoop: '{name}'")

        # 启动健康监控线程
        self._health_thread = Thread(
            target=self._health_loop, daemon=True, name="scheduler-health"
        )
        self._health_thread.start()
        logger.info("PersonaScheduler 健康监控已启动")

        with self._lock:
            names = list(self._personas.keys())
        logger.info(f"PersonaScheduler 已启动，管理 {len(names)} 个 persona: {names}")

    def stop_all(self):
        """优雅关闭所有 persona 和健康监控"""
        if not self._running:
            return

        logger.info("PersonaScheduler 正在关闭...")
        self._running = False

        # 1. 停止健康监控
        self._stop_event.set()
        if self._health_thread and self._health_thread.is_alive():
            self._health_thread.join(timeout=5)
        self._health_thread = None

        # 2. 逐个停止 persona 的 AgentLoop
        with self._lock:
            personas_snapshot = list(self._personas.values())

        for mp in personas_snapshot:
            logger.info(f"正在停止 persona '{mp.name}'...")
            try:
                mp.agent_loop.stop()
            except Exception as e:
                logger.error(f"停止 persona '{mp.name}' 失败: {e}")

        # 3. 关闭共享资源
        self._infra.close()

        logger.info("PersonaScheduler 已完全关闭")

    # ── 健康监控 ─────────────────────────────────────────────────────────

    def get_health_report(self) -> Dict[str, Dict]:
        """返回各 persona 的健康状态"""
        report = {}
        now = time.time()
        with self._lock:
            for name, mp in self._personas.items():
                report[name] = {
                    "alive": mp.agent_loop.running,
                    "uptime_seconds": int(now - mp.started_at) if mp.started_at else 0,
                    "queue_size": mp.agent_loop.event_queue.qsize(),
                    "proactive_state": mp.agent_loop.proactive_state.value,
                    "contexts": sorted(mp.context_ids),
                    "patterns": mp.context_patterns,
                    "config_source": mp.config_source,
                }
        return report

    def set_alert_callback(self, fn: Callable[[str], None]):
        """设置健康报警回调（例如通知 owner）"""
        self._alert_callback = fn

    def _health_loop(self):
        """健康监控线程主循环"""
        while not self._stop_event.wait(self._config.health_check_interval):
            with self._lock:
                personas_snapshot = list(self._personas.items())

            for name, mp in personas_snapshot:
                if not mp.agent_loop.running:
                    msg = (
                        f"[Health] AgentLoop '{name}' 已停止 "
                        f"(运行了 {time.time() - mp.started_at:.0f}s)"
                    )
                    logger.error(msg)
                    if self._alert_callback:
                        try:
                            self._alert_callback(msg)
                        except Exception:
                            pass
                else:
                    qsize = mp.agent_loop.event_queue.qsize()
                    if qsize > 50:
                        logger.warning(
                            f"[Health] persona '{name}' 事件队列积压: {qsize} 个事件"
                        )

    # ── 属性和查询 ────────────────────────────────────────────────────────

    @property
    def persona_names(self) -> List[str]:
        with self._lock:
            return list(self._personas.keys())

    def get_persona(self, name: str) -> Optional[ManagedPersona]:
        with self._lock:
            return self._personas.get(name)

    def __len__(self) -> int:
        with self._lock:
            return len(self._personas)

    def __repr__(self) -> str:
        with self._lock:
            names = list(self._personas.keys())
        return (
            f"PersonaScheduler(running={self._running}, "
            f"personas={names}, "
            f"max_concurrent_llm={self._config.max_concurrent_llm})"
        )
