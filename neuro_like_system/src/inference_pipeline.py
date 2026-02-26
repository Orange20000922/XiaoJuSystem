"""
完整推理Pipeline

架构分工：
  BERT 小模型 — 情绪/行为/强度分析，用于：
    · L4 用户画像写入触发（高强度情绪）
    · 群聊注意力判断（是否需要回复）
    · 话题边界检测（触发记忆压缩）
    · 元数据记录（ConversationTurn）

  LLM — 完整人格 prompt + 记忆上下文，自主判断如何回复
    · 不再接收 BERT 分类结果，避免重复判断浪费 token
"""

import torch
import json
import time
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.logger import logger

from models.joint_model import JointEmotionBehaviorModel, create_joint_model
from configs.model_config import (
    PersonalityConfig,
    DEFAULT_PERSONALITY,
    LLMConfig,
    LLMProvider,
    MemoryConfig,
)
from configs.config_loader import AppConfig
from src.memory_manager import HierarchicalMemoryManager


@dataclass
class ConversationTurn:
    """单轮对话"""
    user_input: str
    emotion: str
    intensity: float
    behavior: str
    tone: str
    response: str
    timestamp: Optional[str] = None


class MemoryManager:
    """简单的记忆管理器"""

    def __init__(self, max_short_term: int = 10):
        self.short_term: List[ConversationTurn] = []
        self.max_short_term = max_short_term

    def add(self, turn: ConversationTurn):
        """添加对话轮次"""
        self.short_term.append(turn)
        if len(self.short_term) > self.max_short_term:
            self.short_term.pop(0)

    def get_context(self, num_turns: int = 5) -> List[ConversationTurn]:
        """获取最近的对话上下文"""
        return self.short_term[-num_turns:]

    def format_context(self, num_turns: int = 5) -> str:
        """格式化上下文为文本"""
        context = self.get_context(num_turns)
        if not context:
            return ""

        lines = []
        for turn in context:
            lines.append(f"用户: {turn.user_input}")
            lines.append(f"助手: {turn.response}")

        return "\n".join(lines)


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
        system_prompt: str,
        user_input: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        history: Optional[List[Dict]] = None,
    ) -> str:
        """
        生成回复，带指数退避重试。

        可重试错误（网络/限流/服务端临时故障）：最多 max_retries 次。
        不可重试错误（认证失败/请求参数非法）：立即抛出，不重试。
        """
        max_tokens = max_tokens or self.config.max_tokens
        temperature = temperature or self.config.temperature

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
                            history=history,
                        )
                    else:
                        result = self._generate_openai_compatible(
                            system_prompt, user_input, max_tokens, temperature,
                            history=history,
                        )
                else:
                    result = self._generate_anthropic(
                        system_prompt, user_input, max_tokens, temperature,
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
        history: Optional[List[Dict]] = None,
    ) -> str:
        """
        OpenAI 兼容 API 生成。
        history 为 L1 原文 messages 列表，直接拼入上下文窗口。
        """
        messages = [{"role": "system", "content": system_prompt}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_input})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=self.config.top_p,
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
            top_p=self.config.top_p,
        )
        return response.output[0].content[0].text

    def _generate_anthropic(
        self,
        system_prompt: str,
        user_input: str,
        max_tokens: int,
        temperature: float
    ) -> str:
        """Anthropic API生成（system prompt 标记 cache_control 启用缓存）"""
        response = self.client.messages.create(
            model=self.model,
            system=[
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": user_input}],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=self.config.top_p,
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


class NeuroLikePipeline:
    """
    完整的Neuro-Like推理Pipeline

    流程:
    1. 用户输入 -> 小模型 (情绪+行为识别)
    2. 小模型输出 -> 构建Prompt
    3. Prompt + 用户输入 -> 大模型API
    4. 大模型输出 -> 返回给用户
    """

    def __init__(
        self,
        small_model_path: str,
        personality: PersonalityConfig,
        llm_config: Optional[LLMConfig] = None,
        memory_config: Optional[MemoryConfig] = None,
        # 向后兼容的参数
        llm_provider: Optional[str] = None,
        llm_api_key: Optional[str] = None,
        llm_model: Optional[str] = None,
        llm_base_url: Optional[str] = None,
        device: str = "cpu"
    ):
        """
        初始化Pipeline

        Args:
            small_model_path: 小模型检查点路径
            personality: 人格配置
            llm_config: LLM配置对象（推荐使用）
            llm_provider: LLM提供商（向后兼容）
            llm_api_key: API密钥（向后兼容）
            llm_model: 模型名称（向后兼容）
            llm_base_url: API Base URL（向后兼容）
            device: 运行设备
        """
        self.personality = personality
        self.device = device

        # 加载小模型
        logger.info("加载小模型...")
        self.small_model, self.tokenizer = create_joint_model()
        self._load_checkpoint(small_model_path)
        self.small_model.to(device)
        self.small_model.eval()

        # 初始化大模型客户端
        if llm_config is not None:
            logger.info(f"初始化大模型客户端 ({llm_config.provider.value}: {llm_config.model})")
            self.llm_client = LLMClient(llm_config)
        elif llm_provider is not None:
            logger.info(f"初始化大模型客户端 ({llm_provider})")
            self.llm_client = LLMClient.from_args(
                provider=llm_provider,
                api_key=llm_api_key,
                model=llm_model,
                base_url=llm_base_url
            )
        else:
            raise ValueError(
                "未提供 LLM 配置。请使用 NeuroLikePipeline.from_config() 从 config.json 加载，"
                "或通过 llm_config 参数传入 LLMConfig 对象。"
            )

        # 记忆管理器
        if memory_config is not None:
            self.memory = HierarchicalMemoryManager(
                config=memory_config,
                llm_client=self.llm_client,
            )
        else:
            self.memory = MemoryManager()

        # 人格向量
        self.personality_vector = torch.tensor(
            personality.to_embedding_vector(),
            dtype=torch.float32
        )

    @classmethod
    def from_config(cls, config_path: Optional[str] = None) -> "NeuroLikePipeline":
        """从 config.json 创建 Pipeline（推荐入口）"""
        app_cfg = AppConfig.load(config_path)
        logger.info(f"加载配置: {app_cfg}")
        return cls(
            small_model_path=app_cfg.small_model_checkpoint,
            personality=app_cfg.personality,
            llm_config=app_cfg.llm,
            memory_config=app_cfg.memory,
            device=app_cfg.device,
        )

    def _load_checkpoint(self, checkpoint_path: str):
        """加载模型检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.small_model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"加载检查点: {checkpoint_path}")
        if "metrics" in checkpoint:
            metrics = checkpoint['metrics']
            if 'val_loss' in metrics:
                logger.info(f"  验证损失: {metrics['val_loss']:.4f}")
            if 'emotion_acc' in metrics:
                logger.info(f"  情绪准确率: {metrics['emotion_acc']:.2%}")

    def analyze_emotion_behavior(self, text: str) -> Dict:
        """使用小模型分析情绪和行为"""
        result = self.small_model.predict(
            text=text,
            personality=self.personality_vector,
            tokenizer=self.tokenizer,
            device=self.device
        )
        return result

    def build_system_prompt(self, recalled_context: str = "") -> str:
        """
        构建 system prompt。
        只包含：人格 + L2/L3/L4 跨会话召回。
        L1 原文通过 messages 数组单独传入，不在这里。
        """
        p = self.personality

        if p.description.strip():
            personality_section = p.description.strip()
        else:
            formality_str = (
                "口语" if p.formality < 0.4 else
                "适中" if p.formality < 0.7 else "正式"
            )
            personality_section = (
                f"[标签]{','.join(p.traits)} "
                f"[开放]{p.openness:.1f} [外向]{p.extraversion:.1f} "
                f"[幽默]{p.humor_tendency:.1f} [共情]{p.empathy_level:.1f} "
                f"[好奇]{p.curiosity_level:.1f} [风格]{formality_str}"
            )

        prompt = f"你是{p.name}。\n<persona>{personality_section}</persona>"
        if recalled_context:
            prompt += f"\n{recalled_context}"
        prompt += "\n根据人格和对话历史自然回复，保持一致性。"
        return prompt

    def generate_response(self, user_input: str,
                          recalled_context: str = "",
                          history: Optional[List[Dict]] = None) -> str:
        """
        使用大模型生成回复。

        Args:
            user_input: 当前用户输入
            recalled_context: L2/L3/L4 召回（进 system prompt）
            history: L1 原文 messages 列表（进 messages 数组）
        """
        system_prompt = self.build_system_prompt(recalled_context)
        return self.llm_client.generate(
            system_prompt=system_prompt,
            user_input=user_input,
            history=history,
        )

    def should_respond(self, emotion_behavior: Dict, is_mentioned: bool = False) -> bool:
        """
        注意力判断：是否需要回复（群聊场景使用）

        Args:
            emotion_behavior: BERT 分析结果
            is_mentioned: 是否被 @ 提及

        Returns:
            True 表示需要回复
        """
        if is_mentioned:
            return True

        intensity = emotion_behavior["emotion"]["intensity"]
        behavior = emotion_behavior["behavior"]["type"]

        # 高强度情绪或问句行为触发回复
        if intensity >= 0.7:
            return True
        if behavior in ("ask_question", "seek_clarification"):
            return True

        return False

    def chat(self, user_input: str, verbose: bool = False,
             is_mentioned: bool = True) -> Dict:
        """
        完整对话流程

        Args:
            user_input: 用户输入
            verbose: 是否输出详细信息
            is_mentioned: 是否被 @ 提及（群聊场景传入）

        Returns:
            {
                "response": 回复文本，若 should_respond=False 则为 None,
                "emotion": BERT 情绪分析结果（旁路元数据）,
                "behavior": BERT 行为分析结果（旁路元数据）,
                "should_respond": bool,
                "debug_info": {...}
            }
        """
        # ── 旁路：BERT 分析（元数据，不影响 LLM 生成）──────────────────
        emotion_behavior = self.analyze_emotion_behavior(user_input)
        intensity = emotion_behavior["emotion"]["intensity"]
        behavior_type = emotion_behavior["behavior"]["type"]

        logger.debug(
            f"BERT 情绪={emotion_behavior['emotion']['primary']} "
            f"强度={intensity:.2f} 行为={behavior_type}"
        )

        # ── 注意力判断 ────────────────────────────────────────────────────
        respond = self.should_respond(emotion_behavior, is_mentioned)
        if not respond:
            logger.debug("注意力判断：跳过回复，仅记录记忆")

        # ── 记忆上下文 ────────────────────────────────────────────────────
        if isinstance(self.memory, HierarchicalMemoryManager):
            # L2/L3/L4 召回 → system prompt
            recalled = self.memory.get_system_context(query=user_input)
            # L1 原文 → messages 数组
            history = self.memory.get_messages_history()
            logger.debug(f"记忆 {self.memory.l1_usage} L1轮次={len(self.memory.working_memory)}")
        else:
            recalled = self.memory.format_context(num_turns=5)
            history = None

        # ── LLM 生成 ──────────────────────────────────────────────────────
        response = None
        if respond:
            response = self.generate_response(user_input, recalled, history)

        # ── 记忆写入（无论是否回复都记录，群聊静默观察也积累上下文）──────
        turn = ConversationTurn(
            user_input=user_input,
            emotion=emotion_behavior["emotion"]["primary"],
            intensity=intensity,
            behavior=behavior_type,
            tone=emotion_behavior["behavior"]["tone"],
            response=response or "",
        )
        self.memory.add(turn)

        return {
            "response": response,
            "emotion": emotion_behavior["emotion"],
            "behavior": emotion_behavior["behavior"],
            "should_respond": respond,
            "debug_info": {
                "model_provider": self.llm_client.provider.value,
                "model_name": self.llm_client.model,
            },
        }

    def save_conversation(self, output_path: str):
        """保存对话历史"""
        if isinstance(self.memory, HierarchicalMemoryManager):
            turns = self.memory.working_memory
        else:
            turns = self.memory.short_term
        data = [asdict(turn) for turn in turns]
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"对话历史已保存: {output_path}")

    def close(self):
        """会话结束：触发 L2→L3 长期记忆写入（仅分级记忆模式有效）"""
        if isinstance(self.memory, HierarchicalMemoryManager):
            logger.info("正在写入长期记忆...")
            self.memory.close_session()
            logger.info("长期记忆已保存。")


def interactive_chat(pipeline: NeuroLikePipeline):
    """交互式对话"""
    print("\n" + "=" * 50)
    print(f"欢迎与 {pipeline.personality.name} 对话！")
    print(f"模型: {pipeline.llm_client.model}")
    print("输入 'quit' 退出，'save' 保存对话历史")
    print("=" * 50 + "\n")

    while True:
        try:
            user_input = input("你: ").strip()

            if not user_input:
                continue

            if user_input.lower() == "quit":
                pipeline.close()
                print("再见！")
                break

            if user_input.lower() == "save":
                pipeline.save_conversation("conversation_history.json")
                continue

            # 生成回复
            result = pipeline.chat(user_input, verbose=True)

            print(f"\n{pipeline.personality.name}: {result['response']}\n")

        except KeyboardInterrupt:
            print("\n\n再见！")
            break
        except Exception as e:
            print(f"\n错误: {e}\n")


# ============== 入口 ==============
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Neuro-Like对话系统")
    parser.add_argument("--config", type=str, default=None,
                        help="config.json 路径（默认：项目根目录下的 config.json）")
    args = parser.parse_args()

    pipeline = NeuroLikePipeline.from_config(args.config)
    interactive_chat(pipeline)
