"""
LLM 客户端基类

支持多种 LLM 提供商的统一接口，包含重试逻辑和错误处理
"""

import time
from typing import Dict, List, Optional
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.logger import logger
from configs.model_config import LLMConfig, LLMProvider


class LLMClient:
    """大模型API客户端 (支持多种API)"""

    def __init__(self, config: LLMConfig):
        """
        初始化LLM客户端

        Args:
            config: LLMConfig配置对象
        """
        self.config = config
        self.provider = config.provider
        self.model = config.model
        self.base_url = config.base_url
        self.api_key = config.api_key

        # 验证API密钥
        if not self.api_key:
            raise ValueError(
                f"API密钥未设置。请设置环境变量或在配置中提供api_key。\n"
                f"Provider: {self.provider.value}"
            )

        # 初始化客户端
        self._init_client()

    def _init_client(self):
        """初始化API客户端"""
        if self.provider in [LLMProvider.OPENAI, LLMProvider.DEEPSEEK, LLMProvider.CUSTOM]:
            from openai import OpenAI
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.config.timeout
            )
        elif self.provider == LLMProvider.ANTHROPIC:
            import anthropic, httpx

            kwargs = {"api_key": self.api_key, "timeout": self.config.timeout}
            if self.base_url:
                kwargs["base_url"] = self.base_url
                # 代理模式：环境变量 ANTHROPIC_AUTH_TOKEN（Claude Code 登录 token）
                # 会让 SDK 同时发送 x-api-key 和 Authorization 两个 header，
                # 导致代理报 401 "冲突的 API 密钥"。
                # 用 httpx event hook 在请求发出前移除 Authorization header。
                def _strip_bearer(request: httpx.Request):
                    if "authorization" in request.headers:
                        del request.headers["authorization"]

                kwargs["http_client"] = httpx.Client(
                    event_hooks={"request": [_strip_bearer]}
                )
            self.client = anthropic.Anthropic(**kwargs)
        else:
            raise ValueError(f"不支持的provider: {self.provider}")

    def generate(
        self,
        system_prompt: str = None,
        user_input: str = "",
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        history: Optional[List[Dict]] = None,
        system_blocks: Optional[List[Dict]] = None,
        images: Optional[List] = None,
    ) -> str:
        """
        生成回复，带指数退避重试。

        Args:
            system_prompt: 单块 system prompt（向后兼容）
            system_blocks: 多块 system prompt（优先使用，用于 Anthropic 缓存优化）
            user_input: 用户输入
            max_tokens: 最大 token 数
            temperature: 温度
            history: 对话历史（OpenAI 格式）
            images: 图片列表（ImageResult 对象）

        可重试错误（网络/限流/服务端临时故障）：最多 max_retries 次。
        不可重试错误（认证失败/请求参数非法）：立即抛出，不重试。
        """
        max_tokens = max_tokens or self.config.max_tokens
        temperature = temperature or self.config.temperature
        top_p = top_p or self.config.top_p

        delay = self.config.retry_delay

        for attempt in range(self.config.max_retries):
            try:
                logger.debug(
                    f"LLM 请求 attempt={attempt + 1}/{self.config.max_retries} "
                    f"model={self.model} max_tokens={max_tokens}"
                )
                if self.provider in [LLMProvider.OPENAI, LLMProvider.DEEPSEEK,
                                     LLMProvider.CUSTOM]:
                    if self.config.use_responses_api:
                        result = self._generate_responses_api(
                            system_prompt, user_input, max_tokens, temperature,
                            top_p=top_p,
                            history=history,
                        )
                    else:
                        result = self._generate_openai_compatible(
                            system_prompt, user_input, max_tokens, temperature,
                            top_p=top_p,
                            history=history,
                            images=images,
                        )
                else:
                    result = self._generate_anthropic(
                        system_prompt, user_input, max_tokens, temperature,
                        top_p=top_p,
                        system_blocks=system_blocks,
                        history=history,
                        images=images,
                    )
                logger.debug(f"LLM 响应成功 长度={len(result)} chars")
                return result

            except Exception as e:
                err_str = str(e)
                status_code = getattr(e, "status_code", None)

                # ── 不可重试：认证/权限/请求参数错误 ──────────────────
                if status_code in (401, 403, 422):
                    logger.error(
                        f"LLM 不可重试错误 status={status_code}: {err_str}"
                    )
                    raise

                # ── 不可重试：上下文超长 ────────────────────────────────
                if status_code == 400 and "context" in err_str.lower():
                    logger.error(f"LLM 上下文超长 status=400: {err_str}")
                    raise

                # ── 可重试：限流 / 服务不可用 / 网络超时 ────────────────
                is_last = attempt >= self.config.max_retries - 1
                if is_last:
                    logger.error(
                        f"LLM 请求失败，已达最大重试次数 {self.config.max_retries}: {err_str}"
                    )
                    raise

                logger.warning(
                    f"LLM 请求失败 attempt={attempt + 1}/{self.config.max_retries} "
                    f"status={status_code} error={err_str} "
                    f"等待 {delay:.1f}s 后重试..."
                )
                time.sleep(delay)
                delay = min(delay * self.config.retry_backoff,
                            self.config.retry_max_delay)

    def _generate_openai_compatible(
        self,
        system_prompt: str,
        user_input: str,
        max_tokens: int,
        temperature: float,
        top_p: Optional[float] = None,
        history: Optional[List[Dict]] = None,
        images: Optional[List] = None,
    ) -> str:
        """
        OpenAI 兼容 API 生成。
        history 为 L1 原文 messages 列表，直接拼入上下文窗口。
        images 为 ImageResult 列表，转换为 OpenAI 格式的 image_url content block。
        """
        messages = [{"role": "system", "content": system_prompt}]
        if history:
            messages.extend(history)

        # 有图片时构建多块 content（OpenAI Vision 格式）
        if images:
            user_content = []
            for img in images:
                user_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{img.media_type};base64,{img.base64_data}",
                    },
                })
            user_content.append({"type": "text", "text": user_input})
            messages.append({"role": "user", "content": user_content})
        else:
            messages.append({"role": "user", "content": user_input})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p if top_p is not None else self.config.top_p,
            frequency_penalty=self.config.frequency_penalty,
            presence_penalty=self.config.presence_penalty,
        )
        return response.choices[0].message.content

    def _generate_responses_api(
        self,
        system_prompt: str,
        user_input: str,
        max_tokens: int,
        temperature: float,
        top_p: Optional[float] = None,
        history: Optional[List[Dict]] = None,
    ) -> str:
        """OpenAI Responses API（/v1/responses）格式生成"""
        # 构造 input：历史 messages + 当前用户输入
        input_messages = list(history) if history else []
        input_messages.append({"role": "user", "content": user_input})

        response = self.client.responses.create(
            model=self.model,
            instructions=system_prompt,
            input=input_messages,
            max_output_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p if top_p is not None else self.config.top_p,
        )
        return response.output[0].content[0].text

    def _generate_anthropic(
        self,
        system_prompt: str,
        user_input: str,
        max_tokens: int,
        temperature: float,
        top_p: Optional[float] = None,
        system_blocks: Optional[List[Dict]] = None,
        history: Optional[List[Dict]] = None,
        images: Optional[List] = None,
    ) -> str:
        """
        Anthropic API生成（支持多块 system prompt 以优化缓存）

        Args:
            system_prompt: 单块 system prompt（向后兼容）
            system_blocks: 多块 system prompt（优先使用，用于缓存优化）
            history: 对话历史（OpenAI 格式 messages）
            images: 图片列表（ImageResult 对象）
        """
        if system_blocks:
            # 使用多块 system prompt（缓存优化）
            system_content = system_blocks
        else:
            # 向后兼容：单块 system prompt
            system_content = [
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ]

        # 构建 messages 数组：history + 当前用户输入
        messages = []
        if history:
            # 将历史消息添加到 messages 数组
            # 对于 Anthropic，可以在最后一条历史消息上添加缓存标记
            for i, msg in enumerate(history):
                if i == len(history) - 1:
                    # 最后一条历史消息标记为可缓存（Anthropic Prompt Caching）
                    # 这样历史对话部分可以被缓存，节省 90% 的 input 成本
                    msg_with_cache = msg.copy()
                    if isinstance(msg_with_cache.get("content"), str):
                        # 字符串 content 需要转换为 content blocks 格式
                        msg_with_cache["content"] = [
                            {
                                "type": "text",
                                "text": msg["content"],
                                "cache_control": {"type": "ephemeral"}
                            }
                        ]
                    elif isinstance(msg_with_cache.get("content"), list):
                        # 已经是 content blocks 格式，在最后一个 block 添加缓存标记
                        msg_with_cache["content"] = msg_with_cache["content"].copy()
                        if msg_with_cache["content"]:
                            msg_with_cache["content"][-1] = {
                                **msg_with_cache["content"][-1],
                                "cache_control": {"type": "ephemeral"}
                            }
                    messages.append(msg_with_cache)
                else:
                    messages.append(msg)

        # 当前用户消息：图片在前，文字在后
        user_content = []
        if images:
            for img in images:
                user_content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": img.media_type,
                        "data": img.base64_data,
                    },
                })
        user_content.append({"type": "text", "text": user_input})

        messages.append({"role": "user", "content": user_content})

        response = self.client.messages.create(
            model=self.model,
            system=system_content,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p if top_p is not None else self.config.top_p,
        )
        import re
        # 优先取 text 类型 block，过滤 thinking block
        for block in response.content:
            if block.type == "text":
                # 部分供应商把 <thinking>...</thinking> 混在 text 里，过滤掉
                text = re.sub(r"<thinking>.*?</thinking>\s*", "", block.text,
                              flags=re.DOTALL).strip()
                return text if text else block.text
        return response.content[0].text

    @classmethod
    def from_config(cls, config: LLMConfig) -> "LLMClient":
        """从配置创建客户端"""
        return cls(config)

    @classmethod
    def from_args(
        cls,
        provider: str = "openai",
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None
    ) -> "LLMClient":
        """从参数创建客户端（向后兼容）"""
        provider_enum = LLMProvider(provider.lower())

        # 设置默认模型
        default_models = {
            LLMProvider.OPENAI: "gpt-5",
            LLMProvider.DEEPSEEK: "deepseek-chat",
            LLMProvider.ANTHROPIC: "claude-3-5-haiku-20241022",
            LLMProvider.CUSTOM: "gpt-5"
        }
        model = model or default_models.get(provider_enum, "gpt-5")

        config = LLMConfig(
            provider=provider_enum,
            api_key=api_key,
            model=model,
            base_url=base_url
        )
        return cls(config)
