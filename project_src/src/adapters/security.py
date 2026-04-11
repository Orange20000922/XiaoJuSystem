"""
API 安全中间件

提供认证、限流、IP 封禁和请求体积限制，全部基于 stdlib + starlette。
"""

import asyncio
import json
import threading
import time
from typing import Dict, List, Optional, Set

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from configs.model_config import SecurityConfig


# ── OpenAI 格式错误响应构建 ───────────────────────────────────────────────────

def _error_response(status_code: int, message: str, error_type: str,
                    code: Optional[str] = None) -> Response:
    body = {"error": {"message": message, "type": error_type}}
    if code:
        body["error"]["code"] = code
    return Response(
        content=json.dumps(body, ensure_ascii=False),
        status_code=status_code,
        media_type="application/json",
    )


# ── 3a. InMemoryRateLimiter ──────────────────────────────────────────────────

class InMemoryRateLimiter:
    """滑动窗口限流器（线程安全）"""

    def __init__(self):
        self._windows: Dict[str, List[float]] = {}
        self._lock = threading.Lock()

    def is_allowed(self, key: str, limit: int, window: float = 60.0) -> bool:
        now = time.monotonic()
        cutoff = now - window
        with self._lock:
            timestamps = self._windows.get(key)
            if timestamps is None:
                self._windows[key] = [now]
                return True
            # 移除过期条目
            timestamps[:] = [t for t in timestamps if t > cutoff]
            if len(timestamps) >= limit:
                return False
            timestamps.append(now)
            return True

    def cleanup(self, max_age: float = 120.0):
        """清理超过 max_age 秒无活动的 key"""
        now = time.monotonic()
        cutoff = now - max_age
        with self._lock:
            stale = [k for k, ts in self._windows.items()
                     if not ts or ts[-1] < cutoff]
            for k in stale:
                del self._windows[k]


# ── 3b. IPBanManager ─────────────────────────────────────────────────────────

class IPBanManager:
    """追踪连续认证失败，超阈值自动封禁"""

    def __init__(self, threshold: int, duration: float):
        self._threshold = threshold
        self._duration = duration
        self._failures: Dict[str, int] = {}          # ip -> 连续失败次数
        self._banned: Dict[str, float] = {}           # ip -> 封禁到期时间
        self._lock = threading.Lock()

    def record_failure(self, ip: str) -> bool:
        """记录一次认证失败，返回是否触发封禁"""
        with self._lock:
            count = self._failures.get(ip, 0) + 1
            self._failures[ip] = count
            if count >= self._threshold:
                self._banned[ip] = time.monotonic() + self._duration
                self._failures.pop(ip, None)
                return True
            return False

    def clear_failures(self, ip: str):
        """认证成功时清除失败计数"""
        with self._lock:
            self._failures.pop(ip, None)

    def is_banned(self, ip: str) -> bool:
        with self._lock:
            expire = self._banned.get(ip)
            if expire is None:
                return False
            if time.monotonic() > expire:
                del self._banned[ip]
                return False
            return True


# ── 3c. AuthMiddleware ────────────────────────────────────────────────────────

class AuthMiddleware(BaseHTTPMiddleware):
    """认证 + 限流 + 封禁，按 OpenAI 协议返回错误"""

    # 这些路径免认证、免限流
    EXEMPT_PATHS: Set[str] = {"/health", "/v1/models"}

    def __init__(self, app, *, config: SecurityConfig,
                 rate_limiter: InMemoryRateLimiter,
                 ban_manager: IPBanManager):
        super().__init__(app)
        self._config = config
        self._rate_limiter = rate_limiter
        self._ban_manager = ban_manager
        self._api_keys: Set[str] = set(config.api_keys)
        self._whitelist: Set[str] = set(config.ip_whitelist)

    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        # 1. 豁免路径
        if path in self.EXEMPT_PATHS:
            return await call_next(request)

        ip = request.client.host if request.client else "unknown"

        # 2. 白名单 IP 跳过所有检查
        if ip in self._whitelist:
            return await call_next(request)

        # 3. 封禁检查
        if self._ban_manager.is_banned(ip):
            return _error_response(
                403,
                "IP temporarily banned due to repeated auth failures",
                "permission_error",
                "ip_banned",
            )

        # 4. IP 限流
        if not self._rate_limiter.is_allowed(
            f"ip:{ip}", self._config.rate_limit_per_minute
        ):
            return _error_response(
                429,
                "Rate limit exceeded for this IP",
                "rate_limit_error",
                "rate_limit_exceeded",
            )

        # 5. 认证（仅在 enabled=True 时校验）
        if self._config.enabled:
            auth_header = request.headers.get("authorization", "")
            if not auth_header.startswith("Bearer "):
                return _error_response(
                    401,
                    "Missing API key. Include 'Authorization: Bearer <key>' header.",
                    "auth_error",
                    "missing_api_key",
                )

            key = auth_header[7:]  # len("Bearer ") == 7
            if key not in self._api_keys:
                triggered = self._ban_manager.record_failure(ip)
                msg = "Invalid API key"
                if triggered:
                    msg += ". IP has been temporarily banned."
                return _error_response(401, msg, "auth_error", "invalid_api_key")

            # 认证成功，清除失败计数
            self._ban_manager.clear_failures(ip)

            # 6. Key 限流
            if not self._rate_limiter.is_allowed(
                f"key:{key}", self._config.rate_limit_per_key_per_minute
            ):
                return _error_response(
                    429,
                    "Rate limit exceeded for this API key",
                    "rate_limit_error",
                    "rate_limit_exceeded",
                )

        # 全部通过
        return await call_next(request)


# ── 3d. RequestSizeLimitMiddleware ────────────────────────────────────────────

class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    """检查 Content-Length，超限直接拒绝"""

    def __init__(self, app, *, max_bytes: int):
        super().__init__(app)
        self._max_bytes = max_bytes

    async def dispatch(self, request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                if int(content_length) > self._max_bytes:
                    return _error_response(
                        413,
                        f"Request body too large (max {self._max_bytes} bytes)",
                        "invalid_request_error",
                        "request_too_large",
                    )
            except ValueError:
                pass
        return await call_next(request)


# ── 3e. apply_security 入口 ──────────────────────────────────────────────────

def apply_security(app, config: SecurityConfig):
    """按序挂载所有安全中间件并启动后台清理任务"""
    rate_limiter = InMemoryRateLimiter()
    ban_manager = IPBanManager(
        threshold=config.auth_fail_ban_threshold,
        duration=config.auth_fail_ban_duration_seconds,
    )

    # Starlette 中间件按添加顺序逆序执行（最后添加的最先执行）
    # 执行顺序：RequestSizeLimit → Auth（含限流/封禁）→ 应用

    app.add_middleware(
        AuthMiddleware,
        config=config,
        rate_limiter=rate_limiter,
        ban_manager=ban_manager,
    )
    app.add_middleware(
        RequestSizeLimitMiddleware,
        max_bytes=config.max_request_body_bytes,
    )

    # CORS（可选）
    if config.cors_enabled:
        from starlette.middleware.cors import CORSMiddleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=config.cors_origins,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # 后台定期清理限流窗口
    async def _cleanup_loop():
        while True:
            await asyncio.sleep(120)
            rate_limiter.cleanup()

    @app.on_event("startup")
    async def _start_cleanup():
        asyncio.create_task(_cleanup_loop())
