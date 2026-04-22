"""
OpenAI 兼容 API 适配器

将 NeuroLikeSystem 推理管线封装为 OpenAI `/v1/chat/completions` 兼容的 HTTP 服务，
供腾讯云 ADP 智能体开发平台通过「自定义模型」方式调用。

内部完整运行：BERT 情绪分类 → 情绪融合 → OU 状态机 → 分级记忆 → LLM 生成。
对外表现为标准 OpenAI chat completion 接口，支持文本和图片多模态输入。
"""

import asyncio
import json
import time
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from configs.model_config import SecurityConfig
from src.core_engine.api import ChatRequest
from src.logger import logger
from src.server.http.security import apply_security


# ── 请求模型 ──────────────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str
    # content 可以是纯文本字符串，或 OpenAI 多模态内容数组
    content: Union[str, List[Dict[str, Any]]]


class ChatCompletionRequest(BaseModel):
    model: str = "educational-agent"
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    stream: bool = False
    user: Optional[str] = None

    model_config = {"extra": "allow"}


# ── 图片解析 ──────────────────────────────────────────────────────────────────

def _parse_images_from_content(content: Union[str, List[Dict]]) -> tuple:
    """
    从 OpenAI 多模态 content 中提取文本和图片列表。

    支持的图片格式：
      - {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
      - {"type": "image_url", "image_url": {"url": "https://..."}}

    返回：
        (text: str, images: List[ImageResult])
    """
    from src.media.image_utils import ImageResult

    if isinstance(content, str):
        return content, []

    text_parts = []
    images = []

    for part in content:
        part_type = part.get("type", "")

        if part_type == "text":
            text_parts.append(part.get("text", ""))

        elif part_type == "image_url":
            image_url_obj = part.get("image_url", {})
            url = image_url_obj.get("url", "")

            if url.startswith("data:"):
                # base64 内嵌图片：data:image/jpeg;base64,<data>
                try:
                    header, data = url.split(",", 1)
                    media_type = header.split(":")[1].split(";")[0]
                    images.append(
                        ImageResult(
                            base64_data=data,
                            media_type=media_type,
                            original_url=url[:50],  # 只记前缀用于日志
                        )
                    )
                except Exception as exc:
                    logger.warning(f"解析 base64 图片失败: {exc}")

            elif url.startswith("http"):
                # URL 图片：下载并处理
                try:
                    from src.media.image_utils import process_image_url

                    result = process_image_url(url)
                    if result:
                        images.append(result)
                except Exception as exc:
                    logger.warning(f"下载图片失败 ({url[:60]}...): {exc}")

    text = " ".join(text_parts).strip()
    return text, images


# ── 响应构建 ──────────────────────────────────────────────────────────────────

def _make_completion_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:24]}"


def _build_response(
    completion_id: str,
    model: str,
    content: str,
    finish_reason: str = "stop",
) -> Dict:
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish_reason,
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def _build_chunk(
    completion_id: str,
    model: str,
    delta: Dict,
    finish_reason: Optional[str] = None,
) -> str:
    chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }
    return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"


# ── FastAPI 应用工厂 ──────────────────────────────────────────────────────────

def create_app(
    runtime,
    model_name: Optional[str] = None,
    security_config: Optional[SecurityConfig] = None,
) -> FastAPI:
    """
    创建 FastAPI 应用。

    参数：
        runtime: DirectRuntime 或兼容封装对象（已初始化）
        model_name: 对外展示的模型名称
        security_config: 安全配置（认证/限流/并发控制）
    """
    resolved_model_name = model_name or getattr(runtime, "persona_name", "educational-agent")
    app = FastAPI(title="NeuroLike Educational Agent API", version="1.0.0")

    sec = security_config or SecurityConfig()
    semaphore = asyncio.Semaphore(sec.max_concurrent_chat)
    max_messages = sec.max_messages_per_request
    debug = sec.debug

    # ── POST /v1/chat/completions ──────────────────────────────────────

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest):
        # 0. messages 数量校验
        if len(req.messages) > max_messages:
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "message": f"Too many messages ({len(req.messages)} > {max_messages})",
                        "type": "invalid_request_error",
                        "code": "too_many_messages",
                    }
                },
            )

        # 1. 提取最后一条 user 消息（含多模态内容）
        user_input = ""
        images = []
        for msg in reversed(req.messages):
            if msg.role == "user":
                user_input, images = _parse_images_from_content(msg.content)
                break

        if not user_input and not images:
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "message": "No user message found",
                        "type": "invalid_request_error",
                    }
                },
            )

        if images:
            logger.info(f"收到 {len(images)} 张图片")

        # 2. context_id
        context_id = req.user or "default"

        # 3. 调用核心引擎封装层（并发控制）
        loop = asyncio.get_running_loop()
        try:
            async with semaphore:
                response = await loop.run_in_executor(
                    None,
                    lambda: runtime.chat(
                        ChatRequest(
                            text=user_input,
                            context_id=context_id,
                            mode="private",
                            images=images or [],
                            verbose=debug,
                        )
                    ),
                )
        except Exception as exc:
            logger.error(f"Runtime 调用失败: {exc}", exc_info=True)
            return JSONResponse(
                status_code=500,
                content={"error": {"message": str(exc), "type": "server_error"}},
            )

        response_text = response.reply or ""
        completion_id = _make_completion_id()

        # debug 日志输出
        if debug:
            dbg = response.debug or {}
            logger.info(
                f"[DEBUG] user={context_id} input={user_input!r}\n"
                f"  response={response_text!r}\n"
                f"  emotion={response.emotion}\n"
                f"  behavior={response.behavior}\n"
                f"  emotion_state={dbg.get('emotion_state')}\n"
                f"  recalled={dbg.get('recalled_context', '')[:200]!r}"
            )

        # 4. 非流式响应
        if not req.stream:
            return JSONResponse(
                content=_build_response(completion_id, resolved_model_name, response_text)
            )

        # 5. 流式响应（SSE）
        async def stream_generator() -> AsyncGenerator[str, None]:
            yield _build_chunk(completion_id, resolved_model_name, {"role": "assistant"})
            for char in response_text:
                yield _build_chunk(completion_id, resolved_model_name, {"content": char})
                await asyncio.sleep(0.02)
            yield _build_chunk(completion_id, resolved_model_name, {}, finish_reason="stop")
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            stream_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    # ── GET /v1/models ─────────────────────────────────────────────────

    @app.get("/v1/models")
    async def list_models():
        return {
            "object": "list",
            "data": [
                {
                    "id": resolved_model_name,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "neuro-like-system",
                }
            ],
        }

    # ── GET /health ────────────────────────────────────────────────────

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "model": resolved_model_name,
            "persona": getattr(runtime, "persona_name", resolved_model_name),
        }

    # ── 安全中间件 ──────────────────────────────────────────────────────

    apply_security(app, sec)

    return app
