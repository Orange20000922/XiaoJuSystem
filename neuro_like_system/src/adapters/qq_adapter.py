"""
QQ 机器人适配层

通过反向 WebSocket（OneBot v11 协议）对接 NapCat，
将 QQ 消息桥接到 AgentLoop / NeuroLikePipeline。

架构：
    NapCat (QQ 协议端)
        ↕ 反向 WebSocket (OneBot v11)
    QQBotAdapter (WebSocket Server, asyncio 主线程)
        ↕ push() / output_callback
    AgentLoop (daemon thread)
        ↕
    NeuroLikePipeline (BERT + Memory + LLM)

用法：
    adapter = QQBotAdapter(config)
    adapter.agent_loop = loop
    asyncio.run(adapter.start())
"""

import asyncio
import html
import json
import re
import time
from datetime import date
from pathlib import Path
from typing import Callable, Dict, List, Optional

import websockets
from websockets.asyncio.server import serve as ws_serve, ServerConnection
from websockets.exceptions import ConnectionClosed
from websockets.http11 import Request, Response

from src.logger import logger
from src.agent.agent_loop import AgentLoop, AgentEvent
from src.core.inference_pipeline import ChatMode
from configs.model_config import QQBotConfig, ImageConfig
from src.media.image_utils import process_image_url, ImageResult, PILLOW_AVAILABLE


# ============== CQ 码解析工具 ==============

CQ_PATTERN = re.compile(r'\[CQ:(\w+)(?:,([^\]]*))?\]')


def parse_cq_codes(message: str) -> tuple:
    """
    解析消息中的 CQ 码。

    Args:
        message: 原始消息字符串（含 CQ 码）

    Returns:
        (plain_text, cq_codes): 纯文本和 CQ 码列表
            cq_codes 每个元素为 {"type": "at", "qq": "12345", ...}
            图片 CQ 码包含 "url" 字段
    """
    cq_codes = []
    for match in CQ_PATTERN.finditer(message):
        cq_type = match.group(1)
        params = {}
        if match.group(2):
            for pair in match.group(2).split(','):
                k, _, v = pair.partition('=')
                params[k.strip()] = v.strip()
        cq_codes.append({"type": cq_type, **params})

    plain_text = CQ_PATTERN.sub('', message).strip()
    return plain_text, cq_codes


def is_at_me(cq_codes: list, bot_qq: int) -> bool:
    """检查 CQ 码中是否有 @机器人"""
    return any(
        c["type"] == "at" and str(c.get("qq")) == str(bot_qq)
        for c in cq_codes
    )


def extract_image_urls(cq_codes: list, max_images: int = 5) -> List[str]:
    """
    从 CQ 码中提取图片 URL。

    Args:
        cq_codes: CQ 码列表
        max_images: 最多提取几张图片

    Returns:
        图片 URL 列表
    """
    urls = []
    for c in cq_codes:
        if c["type"] == "image" and "url" in c:
            # CQ 码中的 URL 可能包含 HTML 转义（如 &amp; → &）
            urls.append(html.unescape(c["url"]))
            if len(urls) >= max_images:
                break
    return urls


# ============== 命令定义 ==============

# (命令, 描述, 是否需要管理员权限)
COMMANDS = {
    "/help":   ("显示帮助信息", False),
    "/status": ("查看小橘状态", True),
    "/clear":  ("清空当前对话历史", True),
    "/logs":   ("查看最近日志", True),
    "/exit":   ("关闭 AgentLoop 并退出", True),
}

# 日志目录（与 logger.py 中 _LOG_DIR 保持一致）
_LOG_DIR = Path(__file__).parent.parent / "logs"


# ============== QQ 机器人适配器 ==============

class QQBotAdapter:
    """
    QQ 机器人适配器。

    作为反向 WebSocket Server，接受 NapCat 连接，
    将 QQ 消息转发给 AgentLoop，并将 AI 回复发回 QQ。
    """

    def __init__(self, config: QQBotConfig, image_config: Optional[ImageConfig] = None):
        self.config = config
        self.image_config = image_config or ImageConfig()
        self.agent_loop: Optional[AgentLoop] = None
        self.scheduler = None  # Optional[PersonaScheduler]，运行时设置
        self.ws_connection: Optional[ServerConnection] = None
        self.async_loop: Optional[asyncio.AbstractEventLoop] = None
        self._server_task: Optional[asyncio.Task] = None
        self._stop_event: asyncio.Event = asyncio.Event()

        # 回复目标跟踪（AgentLoop 线程通过 output_callback 访问）
        self._last_reply_target: Dict = {}

    def _check_path(self, connection: ServerConnection, request: Request):
        """process_request 回调：拒绝路径不匹配的连接"""
        if request.path != self.config.ws_path:
            logger.warning(f"收到未知路径连接: {request.path}，预期 {self.config.ws_path}")
            return connection.respond(404, f"预期路径 {self.config.ws_path}\n")
        return None

    async def start(self):
        """启动 WebSocket Server（在 asyncio 事件循环中运行）"""
        self.async_loop = asyncio.get_running_loop()
        self._stop_event = asyncio.Event()

        async with ws_serve(
            self._handler,
            self.config.ws_host,
            self.config.ws_port,
            process_request=self._check_path,
        ) as server:
            logger.info(
                f"QQ 适配器已启动，监听 "
                f"ws://{self.config.ws_host}:{self.config.ws_port}{self.config.ws_path}"
            )
            # 阻塞直到收到 stop 信号
            await self._stop_event.wait()

    async def _handler(self, websocket: ServerConnection):
        """处理 NapCat 的 WebSocket 连接"""
        # 记录连接
        if self.ws_connection is not None:
            logger.warning("已有 NapCat 连接，新连接将替换旧连接")
        self.ws_connection = websocket
        logger.info("NapCat 已连接")
        await self.notify_owner("NapCat 已连接，小橘上线了~")

        try:
            async for raw in websocket:
                try:
                    event = json.loads(raw)
                    await self._on_event(event)
                except json.JSONDecodeError:
                    logger.warning(f"收到非 JSON 消息: {raw[:200]}")
                except Exception as e:
                    logger.error(f"处理事件异常: {e}", exc_info=True)
        except ConnectionClosed as e:
            logger.info(f"NapCat 断开连接: code={e.code} reason={e.reason}")
        finally:
            if self.ws_connection is websocket:
                self.ws_connection = None
                logger.info("NapCat 连接已清理")
                # 断线无法通知（WS 已断），只记日志

    async def _on_event(self, event: dict):
        """分发 OneBot v11 事件"""
        post_type = event.get("post_type")

        if post_type == "message":
            await self._on_message(event)
        elif post_type == "meta_event":
            sub_type = event.get("meta_event_type")
            if sub_type == "lifecycle":
                logger.info(f"NapCat 生命周期事件: {event.get('sub_type')}")
            elif sub_type == "heartbeat":
                pass  # 心跳，静默处理
        elif post_type == "notice":
            logger.debug(f"通知事件: {event.get('notice_type')}")
        elif post_type == "request":
            logger.debug(f"请求事件: {event.get('request_type')}")
        else:
            # 可能是 action 响应（echo 字段存在）
            if "echo" in event:
                retcode = event.get("retcode", -1)
                if retcode != 0:
                    logger.warning(
                        f"Action 失败: echo={event['echo']} "
                        f"retcode={retcode} msg={event.get('msg')}"
                    )

    async def _on_message(self, event: dict):
        """处理消息事件 → push 到 AgentLoop"""
        msg_type = event.get("message_type")  # "private" | "group"
        user_id = event.get("user_id")
        group_id = event.get("group_id")
        raw_message = event.get("raw_message", event.get("message", ""))

        # 忽略自己发的消息
        if user_id == self.config.bot_qq:
            return

        # 解析 CQ 码
        plain_text, cq_codes = parse_cq_codes(str(raw_message))

        # 提取图片 URL
        image_urls = []
        if self.image_config.enabled and PILLOW_AVAILABLE:
            image_urls = extract_image_urls(
                cq_codes,
                max_images=self.image_config.max_images_per_message
            )

        # 纯图片消息：plain_text 为空但有图片
        if not plain_text and not image_urls:
            return  # 纯表情等，跳过

        # 检测 @
        if msg_type == "group":
            mentioned = is_at_me(cq_codes, self.config.bot_qq)
        else:
            mentioned = True  # 私聊始终视为被提及

        # 命令处理（@ 或私聊时才响应命令）
        if mentioned and plain_text.startswith(self.config.command_prefix):
            handled = await self._handle_command(plain_text, event)
            if handled:
                return

        # 记录回复目标（也作为主动发言 fallback）
        reply_target = {
            "type": msg_type,
            "user_id": user_id,
            "group_id": group_id,
        }
        self._last_reply_target = reply_target

        # 构造发送者标识
        sender = event.get("sender", {})
        raw_name = sender.get("card") or sender.get("nickname") or str(user_id)
        # owner_qq 映射到 owner_name，便于 AI 识别和回复
        if user_id == self.config.owner_qq and self.config.owner_name:
            sender_name = self.config.owner_name
        else:
            sender_name = raw_name

        # 异步下载图片（在 asyncio 上下文中，不阻塞 AgentLoop 线程）
        images = []
        if image_urls:
            logger.info(f"检测到 {len(image_urls)} 张图片，开始下载...")
            images = await self._download_images_async(image_urls)
            if images:
                logger.info(f"成功下载 {len(images)} 张图片")
            else:
                logger.warning("所有图片下载失败")

        # 纯图片且下载全失败 → 跳过
        if not plain_text and not images:
            logger.debug("纯图片消息但下载全失败，跳过")
            return

        # 构造消息内容（图片下载完成后，才能判断是否为纯图片）
        if plain_text:
            content = f"[{sender_name}] {plain_text}"
        else:
            content = f"[{sender_name}] [用户发送了图片]"

        logger.info(
            f"{'群聊' if msg_type == 'group' else '私聊'} "
            f"来自 {sender_name}({user_id})"
            f"{f' 群{group_id}' if group_id else ''}: "
            f"{plain_text[:80] if plain_text else '[纯图片]'}"
            f"{f' +{len(images)}张图' if images else ''}"
        )

        # 推送到 AgentLoop / Scheduler
        # 构造 context_id：群聊用 group_{group_id}，私聊用 private_{user_id}
        if msg_type == "group":
            context_id = f"group_{group_id}"
        else:
            context_id = f"private_{user_id}"

        event = AgentEvent(
            type="message",
            content=content,
            chat_mode=ChatMode.GROUP if msg_type == "group" else ChatMode.PRIVATE,
            is_mentioned=mentioned,
            reply_context=reply_target,
            images=images,
            user_id=user_id,
            user_name=sender_name,
            context_id=context_id,
        )

        if self.scheduler:
            if not self.scheduler.dispatch(event):
                logger.warning(f"无 persona 处理 context_id={context_id}")
        elif self.agent_loop:
            self.agent_loop.push(event)

    def _make_output_callback(self) -> Callable[[str], None]:
        """
        创建 output_callback 闭包。

        从 AgentLoop 线程中安全地发送 QQ 消息，
        使用 asyncio.run_coroutine_threadsafe() 桥接到主线程的 asyncio 事件循环。

        回复路由优先级：
        1. AgentLoop._current_reply_context（当前正在处理的事件绑定的目标）
        2. _last_reply_target（fallback，用于主动发言）
        3. 如果都没有 → 发给 owner（系统通知兜底）
        """
        adapter = self

        def callback(text: str):
            # 优先从当前事件的 reply_context 取目标
            target = (
                adapter.agent_loop._current_reply_context
                if adapter.agent_loop and adapter.agent_loop._current_reply_context
                else adapter._last_reply_target
            )

            # 如果没有回复目标（系统启动时的主动发言等），发给 owner
            if not target:
                if adapter.config.owner_qq:
                    logger.info("无回复目标，系统消息发送给 owner")
                    adapter.notify_owner_sync(text)
                else:
                    logger.warning("无法发送消息: 无回复目标且未配置 owner_qq")
                return

            if not adapter.ws_connection:
                logger.warning("无法发送消息: 无 WS 连接")
                return

            # 长消息分条发送
            chunks = adapter._split_message(text)
            for chunk in chunks:
                at_user = None
                if target["type"] == "group" and adapter.config.reply_with_at:
                    at_user = target["user_id"]

                coro = adapter._send_message(
                    target["type"],
                    target.get("group_id") or target["user_id"],
                    chunk,
                    at_user=at_user,
                )
                future = asyncio.run_coroutine_threadsafe(coro, adapter.async_loop)
                try:
                    future.result(timeout=10)
                except Exception as e:
                    logger.error(f"发送消息失败: {e}")
                    adapter.notify_owner_sync(f"[异常] 发送消息失败: {e}")

        return callback

    def _make_output_callback_for_loop(self, loop: AgentLoop) -> Callable[[str], None]:
        """
        为指定 AgentLoop 创建 output_callback（多 persona 模式使用）。

        与 _make_output_callback 逻辑相同，但闭包捕获的是参数 loop
        而非 self.agent_loop，确保每个 persona 的回复路由到正确的目标。
        """
        adapter = self

        def callback(text: str):
            target = (
                loop._current_reply_context
                if loop._current_reply_context
                else adapter._last_reply_target
            )

            if not target:
                if adapter.config.owner_qq:
                    logger.info("无回复目标，系统消息发送给 owner")
                    adapter.notify_owner_sync(text)
                else:
                    logger.warning("无法发送消息: 无回复目标且未配置 owner_qq")
                return

            if not adapter.ws_connection:
                logger.warning("无法发送消息: 无 WS 连接")
                return

            chunks = adapter._split_message(text)
            for chunk in chunks:
                at_user = None
                if target["type"] == "group" and adapter.config.reply_with_at:
                    at_user = target["user_id"]

                coro = adapter._send_message(
                    target["type"],
                    target.get("group_id") or target["user_id"],
                    chunk,
                    at_user=at_user,
                )
                future = asyncio.run_coroutine_threadsafe(coro, adapter.async_loop)
                try:
                    future.result(timeout=10)
                except Exception as e:
                    logger.error(f"发送消息失败: {e}")
                    adapter.notify_owner_sync(f"[异常] 发送消息失败: {e}")

        return callback

    async def _download_images_async(self, urls: List[str]) -> List[ImageResult]:
        """
        异步下载图片（在 asyncio 上下文中调用）。

        Args:
            urls: 图片 URL 列表

        Returns:
            成功下载的 ImageResult 列表
        """
        if not self.image_config.enabled or not PILLOW_AVAILABLE:
            return []

        # 在 executor 中并行下载（process_image_url 是同步函数）
        loop = asyncio.get_running_loop()
        tasks = []
        for url in urls:
            task = loop.run_in_executor(
                None,
                process_image_url,
                url,
                self.image_config.cache_dir,
                self.image_config.max_download_size_bytes,
                self.image_config.max_dimension,
                self.image_config.cache_ttl_seconds,
                self.image_config.download_timeout,
            )
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 过滤失败的下载
        images = []
        for i, r in enumerate(results):
            if isinstance(r, ImageResult):
                images.append(r)
            elif isinstance(r, Exception):
                logger.warning(f"图片下载失败: {urls[i][:80]} — {r}")
            # None 表示 process_image_url 内部已处理并 log 了

        return images

    async def _send_message(
        self,
        msg_type: str,
        target_id: int,
        text: str,
        at_user: int = None,
    ):
        """通过 WebSocket 发送 OneBot v11 action"""
        if not self.ws_connection:
            logger.warning("无 WS 连接，跳过发送")
            return

        if msg_type == "group":
            message = text
            if at_user:
                message = f"[CQ:at,qq={at_user}] {text}"
            action = {
                "action": "send_group_msg",
                "params": {"group_id": target_id, "message": message},
                "echo": f"send_{time.time()}",
            }
        else:
            action = {
                "action": "send_private_msg",
                "params": {"user_id": target_id, "message": text},
                "echo": f"send_{time.time()}",
            }

        try:
            await self.ws_connection.send(json.dumps(action, ensure_ascii=False))
        except ConnectionClosed:
            logger.warning("发送消息时 WS 连接已断开")

    async def _send_reply(self, event: dict, text: str):
        """根据原始事件直接回复（用于命令响应等）"""
        msg_type = event.get("message_type", "private")
        if msg_type == "group":
            target_id = event.get("group_id")
        else:
            target_id = event.get("user_id")

        chunks = self._split_message(text)
        for chunk in chunks:
            await self._send_message(msg_type, target_id, chunk)

    def _split_message(self, text: str) -> List[str]:
        """将长消息按 max_message_length 分条，优先在换行处断开"""
        max_len = self.config.max_message_length
        if len(text) <= max_len:
            return [text]

        chunks = []
        while text:
            if len(text) <= max_len:
                chunks.append(text)
                break

            # 在 max_len 范围内找最后一个换行
            cut = text.rfind('\n', 0, max_len)
            if cut <= 0:
                # 没有合适的换行，在句号/问号/感叹号处断
                for sep in ('。', '！', '？', '；', '.', '!', '?', '，', ','):
                    cut = text.rfind(sep, 0, max_len)
                    if cut > 0:
                        cut += 1  # 包含标点
                        break
            if cut <= 0:
                cut = max_len  # 硬切

            chunks.append(text[:cut].rstrip())
            text = text[cut:].lstrip()

        return [c for c in chunks if c]

    # ── 权限 & 命令系统 ─────────────────────────────────────────────────

    def _is_admin(self, user_id: int) -> bool:
        """检查用户是否为管理员（owner 或 admin_qq 列表中的成员）"""
        if self.config.owner_qq and user_id == self.config.owner_qq:
            return True
        return user_id in self.config.admin_qq

    async def _handle_command(self, command_text: str, event: dict) -> bool:
        """处理 / 开头的命令，返回 True 表示已处理"""
        parts = command_text.strip().split(maxsplit=1)
        cmd = parts[0].lower()
        user_id = event.get("user_id")

        if cmd not in COMMANDS:
            return False

        _, requires_admin = COMMANDS[cmd]

        # 权限检查
        if requires_admin and not self._is_admin(user_id):
            await self._send_reply(event, "权限不足，该命令仅管理员可用")
            return True

        if cmd == "/help":
            await self._cmd_help(event)
        elif cmd == "/status":
            await self._cmd_status(event)
        elif cmd == "/clear":
            await self._cmd_clear(event)
        elif cmd == "/logs":
            # /logs 或 /logs 50（指定行数）
            n = 30
            if len(parts) > 1:
                try:
                    n = int(parts[1])
                except ValueError:
                    pass
            await self._cmd_logs(event, n)
        elif cmd == "/exit":
            await self._cmd_exit(event)

        return True

    async def _cmd_help(self, event: dict):
        user_id = event.get("user_id")
        is_admin = self._is_admin(user_id)
        lines = ["小橘可用命令:"]
        for k, (desc, admin_only) in COMMANDS.items():
            if admin_only and not is_admin:
                continue
            tag = " [管理员]" if admin_only else ""
            lines.append(f"  {k} — {desc}{tag}")
        await self._send_reply(event, "\n".join(lines))

    async def _cmd_status(self, event: dict):
        # 多 persona 模式：显示所有 persona 状态
        if self.scheduler:
            report = self.scheduler.get_health_report()
            status_lines = [f"调度器状态 ({len(report)} 个 persona):"]
            for name, info in report.items():
                alive = "运行中" if info["alive"] else "已停止"
                uptime = info["uptime_seconds"]
                if uptime < 60:
                    uptime_str = f"{uptime}秒"
                else:
                    uptime_str = f"{uptime // 60}分钟"
                status_lines.append(
                    f"\n  [{name}] {alive} | "
                    f"运行: {uptime_str} | "
                    f"队列: {info['queue_size']} | "
                    f"主动性: {info['proactive_state']}"
                )
                if info["contexts"]:
                    status_lines.append(f"    绑定: {', '.join(info['contexts'][:5])}")
                if info["patterns"]:
                    status_lines.append(f"    通配: {', '.join(info['patterns'])}")
            status_lines.append(f"\nWS 连接: {'已连接' if self.ws_connection else '未连接'}")
            await self._send_reply(event, "\n".join(status_lines))
            return

        # 单 persona 模式（向后兼容）
        status_lines = ["小橘状态:"]
        if self.agent_loop:
            state = self.agent_loop.proactive_state.value
            status_lines.append(f"  主动性状态: {state}")
            idle = time.time() - self.agent_loop.last_user_time
            if idle < 60:
                status_lines.append(f"  空闲时间: {int(idle)}秒")
            else:
                status_lines.append(f"  空闲时间: {int(idle // 60)}分钟")
            if hasattr(self.agent_loop.pipeline, 'memory'):
                wm_len = len(self.agent_loop.pipeline.memory.working_memory)
                status_lines.append(f"  工作记忆轮数: {wm_len}")

            # 注意力追踪器状态
            if hasattr(self.agent_loop.pipeline, 'attention_tracker'):
                att_status = self.agent_loop.pipeline.attention_tracker.get_status()
                status_lines.append(f"  注意力焦点用户: {att_status['focused_users']}")
                status_lines.append(f"  上下文活跃用户: {att_status['context_users']}")
                if att_status['cooldown_active']:
                    status_lines.append(f"  回复冷却: 进行中")
                elif att_status['last_reply_ago'] is not None:
                    status_lines.append(f"  上次回复: {int(att_status['last_reply_ago'])}秒前")

        status_lines.append(f"  WS 连接: {'已连接' if self.ws_connection else '未连接'}")
        await self._send_reply(event, "\n".join(status_lines))

    async def _cmd_clear(self, event: dict):
        # 多 persona 模式
        if self.scheduler:
            msg_type = event.get("message_type")
            user_id = event.get("user_id")
            group_id = event.get("group_id")
            if msg_type == "group":
                ctx = f"group_{group_id}"
            else:
                ctx = f"private_{user_id}"
            mp = self.scheduler.resolve_persona(ctx)
            if mp and hasattr(mp.persona, 'memory'):
                mp.persona.memory.working_memory.clear()
                await self._send_reply(event, f"[{mp.name}] 对话历史已清空~")
            else:
                await self._send_reply(event, "没有可清空的对话历史")
            return

        # 单 persona 模式
        if self.agent_loop and hasattr(self.agent_loop.pipeline, 'memory'):
            self.agent_loop.pipeline.memory.working_memory.clear()
            await self._send_reply(event, "对话历史已清空~")
        else:
            await self._send_reply(event, "没有可清空的对话历史")

    async def _cmd_logs(self, event: dict, n: int = 30):
        """读取今日日志文件最后 n 行，私聊发给请求者"""
        n = max(1, min(n, 100))  # 限制 1~100 行
        user_id = event.get("user_id")

        log_file = _LOG_DIR / f"neuro_{date.today():%Y-%m-%d}.log"
        if not log_file.exists():
            await self._send_private(user_id, "今日暂无日志文件")
            return

        try:
            lines = log_file.read_text(encoding="utf-8").splitlines()
            tail = lines[-n:]
            text = f"最近 {len(tail)} 行日志:\n" + "\n".join(tail)
        except Exception as e:
            text = f"读取日志失败: {e}"

        # 始终通过私聊发送，避免日志泄漏到群
        await self._send_private(user_id, text)
        # 如果原消息来自群，在群里提示一下
        if event.get("message_type") == "group":
            await self._send_reply(event, "日志已私聊发送~")

    async def _cmd_exit(self, event: dict):
        """关闭调度器/AgentLoop 并退出进程"""
        user_id = event.get("user_id")
        await self._send_reply(event, "正在关闭...")
        logger.warning(f"收到 /exit 命令 (来自 {user_id})，正在关闭")

        if self.scheduler:
            self.scheduler.stop_all()
            logger.info("PersonaScheduler 已停止")
        elif self.agent_loop:
            self.agent_loop.stop()
            logger.info("AgentLoop 已停止")

        await self._send_reply(event, "AgentLoop 已关闭，进程即将退出")
        # 如果命令来自群聊，额外私聊通知 owner
        if event.get("message_type") == "group" and user_id != self.config.owner_qq:
            await self.notify_owner(f"管理员 {user_id} 执行了 /exit，进程即将退出")
        # 触发主线程 asyncio 退出
        await self.stop()

    # ── 私聊接口 & Owner 通知 ────────────────────────────────────────────

    async def _send_private(self, user_id: int, text: str):
        """直接向指定用户发送私聊消息"""
        chunks = self._split_message(text)
        for chunk in chunks:
            await self._send_message("private", user_id, chunk)

    async def notify_owner(self, text: str):
        """
        向 owner 私聊发送通知消息。

        用于系统事件（启动、连接变动、异常等）的实时推送。
        若 owner_qq 未配置或无 WS 连接，静默跳过。
        """
        if not self.config.owner_qq or not self.ws_connection:
            return
        try:
            await self._send_private(self.config.owner_qq, text)
        except Exception as e:
            logger.warning(f"通知 owner 失败: {e}")

    def notify_owner_sync(self, text: str):
        """
        notify_owner 的线程安全同步版。

        供 AgentLoop 线程或其他非 asyncio 上下文调用。
        """
        if not self.config.owner_qq or not self.ws_connection or not self.async_loop:
            return
        try:
            future = asyncio.run_coroutine_threadsafe(
                self.notify_owner(text), self.async_loop
            )
            future.result(timeout=10)
        except Exception as e:
            logger.warning(f"同步通知 owner 失败: {e}")

    # ── 生命周期 ──────────────────────────────────────────────────────────

    async def stop(self):
        """优雅关闭 WebSocket Server"""
        self._stop_event.set()
        logger.info("QQ 适配器已关闭")
