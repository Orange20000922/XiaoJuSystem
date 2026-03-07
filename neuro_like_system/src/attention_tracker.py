"""
注意力追踪器

管理群聊场景下的用户级注意力状态：
- 追踪被 @ 过的用户（注意力焦点）
- 追踪最近活跃的用户（上下文窗口）
- 回复冷却机制（避免刷屏）
"""

import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, Optional, Set

from src.logger import logger
from configs.model_config import AttentionConfig


@dataclass
class UserAttention:
    """单个用户的注意力状态"""
    user_id: int
    user_name: str
    mentioned_at: Optional[float] = None  # 最后一次被 @ 的时间戳
    last_message_at: float = 0.0          # 最后一次发言时间戳

    def is_mentioned_active(self, ttl: int) -> bool:
        """检查 @ 注意力是否仍然有效"""
        if self.mentioned_at is None:
            return False
        return (time.time() - self.mentioned_at) < ttl


class AttentionTracker:
    """
    注意力追踪器（群聊场景）

    功能：
    1. 追踪被 @ 过的用户（注意力焦点）
    2. 追踪最近活跃的用户（上下文窗口）
    3. 回复冷却机制（避免刷屏）
    """

    def __init__(self, config: AttentionConfig):
        self.config = config

        # 用户注意力状态表 {user_id: UserAttention}
        self.users: Dict[int, UserAttention] = {}

        # 最近消息窗口（用于判断上下文活跃用户）
        # 存储 (user_id, timestamp) 元组
        self.recent_messages: deque = deque(maxlen=config.context_window_messages)

        # 最后一次回复时间（全局冷却）
        self.last_reply_time: float = 0.0

        # 每个用户的最后回复时间（用户级冷却）
        self.user_last_reply: Dict[int, float] = {}

        # 非焦点回复的最后时间（全局，用于控制非焦点回复频率）
        self.last_non_focus_reply_time: float = 0.0

    def on_message(self, user_id: int, user_name: str, is_mentioned: bool):
        """
        记录用户消息事件

        Args:
            user_id: 用户 QQ 号
            user_name: 用户昵称
            is_mentioned: 是否 @ 了机器人
        """
        now = time.time()

        # 更新或创建用户注意力状态
        if user_id not in self.users:
            self.users[user_id] = UserAttention(
                user_id=user_id,
                user_name=user_name,
                last_message_at=now,
            )
        else:
            self.users[user_id].last_message_at = now
            # 更新昵称（可能变化）
            self.users[user_id].user_name = user_name

        # 如果被 @，更新注意力焦点
        if is_mentioned:
            self.users[user_id].mentioned_at = now
            logger.debug(f"注意力焦点: {user_name}({user_id}) @ 了机器人")

        # 记录到最近消息窗口
        self.recent_messages.append((user_id, now))

    def on_reply(self, user_id: Optional[int] = None):
        """
        记录回复事件（更新冷却时间）

        Args:
            user_id: 回复的目标用户（None 表示全局回复）
        """
        now = time.time()
        self.last_reply_time = now
        if user_id is not None:
            self.user_last_reply[user_id] = now

    def should_respond(
        self,
        user_id: int,
        emotion_intensity: float,
        behavior_type: str,
        is_mentioned: bool,
        in_attention_focus: bool = False,
    ) -> bool:
        """
        综合判断是否应该回复该用户

        判断逻辑：
        1. 被 @ → 必回复（除非在冷却中）
        2. 用户在注意力焦点内（最近被 @ 过）→ 降低回复阈值
        3. 用户在上下文窗口内（最近活跃）→ 适度降低阈值
        4. 高强度情绪或提问行为 → 回复
        5. 冷却机制：避免短时间内连续回复
        6. 非焦点回复间隔：避免 bot 过于吵闹

        Args:
            user_id: 用户 QQ 号
            emotion_intensity: BERT 检测的情绪强度
            behavior_type: BERT 检测的行为类型
            is_mentioned: 是否 @ 了机器人
            in_attention_focus: 是否在注意力焦点内（由外部传入）

        Returns:
            True 表示应该回复
        """
        logger.debug(
            f"[注意力判断] user={user_id} @={is_mentioned} "
            f"focus={in_attention_focus} "
            f"emotion={emotion_intensity:.2f} behavior={behavior_type}"
        )

        # 1. 被 @ → 必回复（但检查冷却）
        if is_mentioned:
            if self._is_in_cooldown(user_id):
                logger.info(f"[注意力判断] 用户 {user_id} 被 @ 但在冷却中，跳过回复")
                return False
            logger.info(f"[注意力判断] 用户 {user_id} @ 了机器人，触发回复")
            return True

        # 2. 检查用户是否在注意力焦点内（使用外部传入的值）
        in_focus = in_attention_focus

        # 3. 检查用户是否在上下文窗口内（最近活跃）
        in_context = self._is_in_context_window(user_id)

        # 4. 根据注意力状态调整阈值
        if in_focus:
            # 注意力焦点内：大幅降低阈值（0.7 → 0.42）
            threshold = self.config.intensity_threshold * 0.6
            logger.debug(f"用户 {user_id} 在注意力焦点内，阈值降低至 {threshold:.2f}")
        elif in_context:
            # 上下文窗口内：适度降低阈值（0.7 → 0.56）
            threshold = self.config.intensity_threshold * 0.8
            logger.debug(f"用户 {user_id} 在上下文窗口内，阈值降低至 {threshold:.2f}")
        else:
            # 普通用户（非焦点）：使用标准阈值
            threshold = self.config.intensity_threshold

            # 非焦点回复间隔检查（避免 bot 过于吵闹）
            if self.config.non_focus_reply_interval > 0:
                now = time.time()
                time_since_last = now - self.last_non_focus_reply_time
                if time_since_last < self.config.non_focus_reply_interval:
                    logger.debug(
                        f"[注意力判断] 非焦点回复间隔未到 "
                        f"({time_since_last:.0f}s < {self.config.non_focus_reply_interval}s)，跳过回复"
                    )
                    return False

        # 5. 情绪强度判断
        if emotion_intensity >= threshold:
            if self._is_in_cooldown(user_id):
                logger.debug(f"用户 {user_id} 在冷却中，跳过回复")
                return False
            logger.debug(
                f"用户 {user_id} 情绪强度 {emotion_intensity:.2f} >= {threshold:.2f}，触发回复"
            )

            # 如果是非焦点回复，更新非焦点回复时间
            if not in_focus and not in_context:
                self.last_non_focus_reply_time = time.time()

            return True

        # 6. 提问行为判断（焦点用户或上下文用户的提问必回复）
        if behavior_type in ("ask_question", "seek_clarification"):
            if in_focus or in_context:
                if self._is_in_cooldown(user_id):
                    logger.debug(f"用户 {user_id} 在冷却中，跳过回复")
                    return False
                logger.debug(f"用户 {user_id} 提问，且在注意力范围内，触发回复")
                return True

        logger.debug(f"[注意力判断] 用户 {user_id} 不满足回复条件")
        return False

    def _is_in_cooldown(self, user_id: Optional[int] = None) -> bool:
        """
        检查是否在冷却期内

        Args:
            user_id: 用户 QQ 号（None 表示检查全局冷却）

        Returns:
            True 表示在冷却中
        """
        now = time.time()
        cooldown = self.config.cooldown_seconds

        # 全局冷却检查
        if (now - self.last_reply_time) < cooldown:
            return True

        # 用户级冷却检查
        if user_id is not None:
            last_reply = self.user_last_reply.get(user_id, 0.0)
            if (now - last_reply) < cooldown:
                return True

        return False

    def _is_in_context_window(self, user_id: int) -> bool:
        """
        检查用户是否在最近消息窗口内（上下文活跃）

        Args:
            user_id: 用户 QQ 号

        Returns:
            True 表示在窗口内
        """
        return any(uid == user_id for uid, _ in self.recent_messages)

    def get_focused_users(self) -> Set[int]:
        """
        获取当前注意力焦点内的所有用户

        Returns:
            用户 QQ 号集合
        """
        if not self.config.track_mentioned_users:
            return set()

        focused = set()
        for user_id, user in self.users.items():
            if user.is_mentioned_active(self.config.mentioned_user_ttl):
                focused.add(user_id)

        return focused

    def get_context_users(self) -> Set[int]:
        """
        获取当前上下文窗口内的所有用户

        Returns:
            用户 QQ 号集合
        """
        return {uid for uid, _ in self.recent_messages}

    def cleanup_expired(self):
        """清理过期的注意力状态（定期调用，避免内存泄漏）"""
        now = time.time()
        ttl = self.config.mentioned_user_ttl

        # 清理过期的用户注意力状态
        expired = []
        for user_id, user in self.users.items():
            # 如果用户既不在注意力焦点，也不在上下文窗口，且超过 TTL*2 未活跃
            if (
                not user.is_mentioned_active(ttl)
                and not self._is_in_context_window(user_id)
                and (now - user.last_message_at) > ttl * 2
            ):
                expired.append(user_id)

        for user_id in expired:
            del self.users[user_id]
            if user_id in self.user_last_reply:
                del self.user_last_reply[user_id]

        if expired:
            logger.debug(f"清理 {len(expired)} 个过期用户注意力状态")

    def get_status(self) -> Dict:
        """获取注意力追踪器状态（用于调试和 /status 命令）"""
        focused = self.get_focused_users()
        context = self.get_context_users()

        return {
            "total_users": len(self.users),
            "focused_users": len(focused),
            "context_users": len(context),
            "cooldown_active": self._is_in_cooldown(),
            "last_reply_ago": time.time() - self.last_reply_time if self.last_reply_time > 0 else None,
        }
