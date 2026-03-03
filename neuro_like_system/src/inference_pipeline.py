"""
完整推理Pipeline

架构分工：
  BERT 小模型 — 情绪/行为/强度分析，用于：
    · 情绪感知指令注入 system prompt（emotion_prompts 配置映射）
    · L4 用户画像写入触发（高强度情绪）
    · 群聊注意力判断（是否需要回复）
    · 话题边界检测（触发记忆压缩）
    · 元数据记录（ConversationTurn）

  LLM — 完整人格 prompt + 情绪指令 + 记忆上下文

  BERT 不可用时自动回退为纯 LLM 模式（默认 neutral 情绪）。
"""

import os
import torch
import json
import time
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

# 如果 HuggingFace 模型缓存已存在，自动启用离线模式，跳过联网检查
def _auto_set_hf_offline():
    model_name = "hfl/chinese-roberta-wwm-ext"
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    model_dir = cache_dir / ("models--" + model_name.replace("/", "--"))
    if model_dir.exists() and "HF_HUB_OFFLINE" not in os.environ:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

_auto_set_hf_offline()
from dataclasses import dataclass, asdict

import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.logger import logger

from models.joint_model import JointEmotionBehaviorModel, create_joint_model
from configs.model_config import (
    PersonalityConfig,
    DEFAULT_PERSONALITY,
    EmotionPromptConfig,
    EmotionFusionConfig,
    EmotionStateConfig,
    LLMConfig,
    LLMProvider,
    MemoryConfig,
)
from configs.config_loader import AppConfig
from src.memory_manager import HierarchicalMemoryManager
from src.emotion_fusion import LLMEmotionClassifier, EmotionNeuronFusion
from src.emotion_state import EmotionStateTracker, EmotionState


class ChatMode(Enum):
    """对话模式：决定 LLM 路由策略"""
    PRIVATE = "private"   # 私聊：始终用主 LLM (Claude)
    GROUP = "group"       # 群聊：BERT 路由，默认副 LLM，必要时升级到主 LLM


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
        system_prompt: str = None,
        user_input: str = "",
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        history: Optional[List[Dict]] = None,
        system_blocks: Optional[List[Dict]] = None,
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
                        )
                else:
                    result = self._generate_anthropic(
                        system_prompt, user_input, max_tokens, temperature,
                        top_p=top_p,
                        system_blocks=system_blocks,
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
        system_blocks: Optional[List[Dict]] = None
    ) -> str:
        """
        Anthropic API生成（支持多块 system prompt 以优化缓存）

        Args:
            system_prompt: 单块 system prompt（向后兼容）
            system_blocks: 多块 system prompt（优先使用，用于缓存优化）
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

        response = self.client.messages.create(
            model=self.model,
            system=system_content,
            messages=[{"role": "user", "content": user_input}],
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


# 情绪 → max_tokens 权重（相对于 config.max_tokens）
# config.max_tokens=400000 是理论上限，权重控制实际使用量
# 目标：大部分场景在 5K-15K tokens，保证对话质量和自然展开
_EMOTION_TOKEN_WEIGHTS = {
    "neutral":    0.0125,  # 日常闲聊 → 5K tokens
    "joy":        0.02,    # 开心 → 8K tokens
    "excitement": 0.025,   # 兴奋 → 10K tokens
    "sadness":    0.03,    # 安慰需要更多话 → 12K tokens
    "fear":       0.03,    # 给安全感 → 12K tokens
    "anger":      0.025,   # 认可情绪 → 10K tokens
    "disgust":    0.02,    # 简短回应 → 8K tokens
    "surprise":   0.025,   # 视情况 → 10K tokens
    "tenderness": 0.025,   # 温暖回应 → 10K tokens
    "curiosity":  0.0375,  # 好奇心需要详细回答 → 15K tokens
}


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
        llm_secondary_config: Optional[LLMConfig] = None,
        memory_config: Optional[MemoryConfig] = None,
        emotion_prompt_config: Optional[EmotionPromptConfig] = None,
        emotion_fusion_config: Optional[EmotionFusionConfig] = None,
        emotion_state_config: Optional[EmotionStateConfig] = None,
        # 向后兼容的参数
        llm_provider: Optional[str] = None,
        llm_api_key: Optional[str] = None,
        llm_model: Optional[str] = None,
        llm_base_url: Optional[str] = None,
        device: str = "cpu",
        time_awareness: bool = True,
    ):
        """
        初始化Pipeline

        Args:
            small_model_path: 小模型检查点路径
            personality: 人格配置
            llm_config: 主 LLM 配置（Claude，私聊 + 群聊升级）
            llm_secondary_config: 副 LLM 配置（DeepSeek 等廉价模型，群聊默认）
            memory_config: 记忆系统配置
            emotion_prompt_config: BERT 输出 → Prompt 指令映射配置
            emotion_fusion_config: BERT + LLM 情绪融合配置
            device: 运行设备
        """
        self.personality = personality
        self.device = device
        self.emotion_prompt_config = emotion_prompt_config
        self.time_awareness = time_awareness

        # 加载小模型（BERT 不可用时回退为 None，纯 LLM 模式）
        try:
            logger.info("加载小模型...")
            self.small_model, self.tokenizer = create_joint_model()
            self._load_checkpoint(small_model_path)
            self.small_model.to(device)
            self.small_model.eval()
            logger.info("小模型加载成功")
        except Exception as e:
            logger.warning(f"小模型加载失败，进入纯 LLM 模式: {e}")
            self.small_model = None
            self.tokenizer = None

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

        # 副 LLM 客户端（群聊廉价模型，可选）
        if llm_secondary_config is not None:
            logger.info(
                f"初始化副 LLM 客户端 "
                f"({llm_secondary_config.provider.value}: {llm_secondary_config.model})"
            )
            self.llm_client_secondary = LLMClient(llm_secondary_config)
        else:
            self.llm_client_secondary = None

        # 情绪融合（BERT + LLM）
        self.emotion_fusion_config = emotion_fusion_config
        if emotion_fusion_config and emotion_fusion_config.enabled and emotion_prompt_config:
            llm_for_emotion = self.llm_client_secondary or self.llm_client
            self.llm_emotion_classifier = LLMEmotionClassifier(
                llm_client=llm_for_emotion,
                temperature=emotion_fusion_config.llm_temperature
            )
            self.emotion_fusion = EmotionNeuronFusion(
                config=emotion_fusion_config,
                emotion_reliability=emotion_prompt_config.emotion_reliability
            )
            logger.info("情绪融合系统已启用")
        else:
            self.llm_emotion_classifier = None
            self.emotion_fusion = None

        # 记忆管理器
        if memory_config is not None:
            self.memory = HierarchicalMemoryManager(
                config=memory_config,
                llm_client=self.llm_client,
            )
        else:
            self.memory = MemoryManager()

        # 情绪状态机
        self.emotion_state_config = emotion_state_config
        if emotion_state_config:
            initial = self._load_emotion_state()
            self.emotion_state_tracker = EmotionStateTracker(
                config=emotion_state_config, initial_state=initial
            )
            logger.info(
                f"情绪状态机已启用 v={self.emotion_state_tracker.state.valence:.2f} "
                f"a={self.emotion_state_tracker.state.arousal:.2f}"
            )
        else:
            self.emotion_state_tracker = None

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
            llm_secondary_config=app_cfg.llm_secondary,
            memory_config=app_cfg.memory,
            emotion_prompt_config=app_cfg.emotion_prompts,
            emotion_fusion_config=app_cfg.emotion_fusion,
            emotion_state_config=app_cfg.emotion_state_config,
            device=app_cfg.device,
            time_awareness=app_cfg.agent.time_awareness,
        )

    def _load_checkpoint(self, checkpoint_path: str):
        """加载模型检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.small_model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"加载检查点: {checkpoint_path}")
        if "metrics" in checkpoint:
            metrics = checkpoint['metrics']
            if 'val_loss' in metrics:
                logger.info(f"  验证损失: {metrics['val_loss']:.4f}")
            if 'emotion_acc' in metrics:
                logger.info(f"  情绪准确率: {metrics['emotion_acc']:.2%}")

    def analyze_emotion_behavior(self, text: str) -> Optional[Dict]:
        """使用小模型分析情绪和行为，BERT 不可用时返回 None"""
        if self.small_model is None:
            return None
        result = self.small_model.predict(
            text=text,
            personality=self.personality_vector,
            tokenizer=self.tokenizer,
            device=self.device
        )
        return result

    def _analyze_emotion_with_fusion(self, text: str, use_fusion: bool = False) -> Optional[Dict]:
        """
        情绪分析（支持 BERT-only 或 BERT+LLM 融合）。

        Args:
            text: 输入文本
            use_fusion: 是否启用融合（False=BERT-only，True=融合）

        Returns:
            emotion_behavior dict
        """
        if not use_fusion or self.emotion_fusion is None:
            # BERT-only 模式
            return self.analyze_emotion_behavior(text)

        # 融合模式：并行调用 BERT + LLM
        bert_result = None
        llm_result = None

        # 优化：BERT 置信度很高时跳过 LLM
        quick_bert = self.analyze_emotion_behavior(text)
        if quick_bert:
            bert_prob = quick_bert["emotion"].get("primary_prob", 0.0)
            emotion = quick_bert["emotion"]["primary"]
            reliability = self.emotion_prompt_config.emotion_reliability.get(emotion, 0.7)
            eff_conf = bert_prob * reliability

            if eff_conf >= self.emotion_fusion_config.skip_llm_threshold:
                logger.debug(f"Skip LLM: BERT eff_conf={eff_conf:.2f} >= {self.emotion_fusion_config.skip_llm_threshold}")
                return quick_bert

            bert_result = quick_bert

        # 并行调用 LLM
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
        with ThreadPoolExecutor(max_workers=1) as executor:
            llm_future = executor.submit(self.llm_emotion_classifier.classify, text, self.emotion_fusion_config.llm_timeout)
            try:
                llm_result = llm_future.result(timeout=self.emotion_fusion_config.llm_timeout + 1)
            except FutureTimeoutError:
                logger.warning("LLM emotion classification timeout")
            except Exception as e:
                logger.warning(f"LLM emotion classification failed: {e}")

        # 融合
        if bert_result:
            fused = self.emotion_fusion.fuse(bert_result, llm_result)
            if llm_result:
                logger.debug(
                    f"Fused: {fused['emotion']['primary']} "
                    f"(BERT: {bert_result['emotion']['primary']}, "
                    f"LLM: {llm_result['emotion']}, "
                    f"agree: {fused['emotion']['_fusion_meta']['agreement']})"
                )
            return fused

        # Fallback
        return self.analyze_emotion_behavior(text)

    def build_system_prompt(self, recalled_context: str = "",
                            emotion_analysis: Optional[Dict] = None) -> str:
        """
        构建 system prompt（单块模式，向后兼容）。
        包含：人格 + 情感分析结果（强制接受）+ L2/L3/L4 跨会话召回。
        L1 原文通过 messages 数组单独传入，不在这里。

        改进：将情感判断完全从 Claude 中剥离，用强制性语言命令 Claude 接受
        情感系统的标签判断，降低 Claude 自行判断情绪的可能。
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

        # 时间感知注入
        if self.time_awareness:
            now = datetime.now().strftime("%Y年%m月%d日 %H:%M")
            prompt += f"\n<current_time>{now}</current_time>"

        # 情感指令注入（仅自然语言指引，结构化数据走日志不进 prompt）
        if emotion_analysis and self.emotion_prompt_config:
            directives = self._build_emotion_directives(emotion_analysis)
            if directives:
                prompt += f"\n<mood>\n{directives}\n</mood>"

        if recalled_context:
            prompt += f"\n{recalled_context}"

        # 情绪状态暗示
        if self.emotion_state_tracker:
            hint = self.emotion_state_tracker.get_prompt_hint()
            if hint:
                prompt += f"\n<feeling>{hint}</feeling>"

        prompt += "\n自然回复就好。"
        return prompt

    def build_system_prompt_blocks(self, recalled_context: str = "",
                                   emotion_analysis: Optional[Dict] = None) -> List[Dict]:
        """
        构建多块 system prompt（用于 Anthropic 缓存优化）。

        缓存策略：
        - Block 1（静态，可缓存）：人格描述 + 通用指令
        - Block 2（半静态，可缓存）：跨会话记忆召回（L2/L3/L4）
        - Block 3（动态，不缓存）：情感分析结果（每轮变化）

        Returns:
            List[Dict]: Anthropic system blocks with cache_control
        """
        p = self.personality
        blocks = []

        # ── Block 1: 静态人格 + 通用指令（可缓存）────────────────────
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

        static_prompt = f"你是{p.name}。\n<persona>{personality_section}</persona>"

        # 时间感知注入（每分钟变化，但缓存 5 分钟内有效）
        if self.time_awareness:
            now = datetime.now().strftime("%Y年%m月%d日 %H:%M")
            static_prompt += f"\n<current_time>{now}</current_time>"

        static_prompt += "\n自然回复就好。"

        blocks.append({
            "type": "text",
            "text": static_prompt,
            "cache_control": {"type": "ephemeral"}
        })

        # ── Block 2: 跨会话记忆召回（半静态，可缓存）────────────────
        if recalled_context:
            blocks.append({
                "type": "text",
                "text": recalled_context,
                "cache_control": {"type": "ephemeral"}
            })

        # ── Block 3: 情感指令（动态，不缓存）─────────────────────────────
        # 结构化数据（情绪类型/强度/置信度等）走日志，prompt 只注入自然语言指引
        if emotion_analysis and self.emotion_prompt_config:
            directives = self._build_emotion_directives(emotion_analysis)
            if directives:
                blocks.append({
                    "type": "text",
                    "text": f"<mood>\n{directives}\n</mood>"
                })

        # ── Block 4: 情绪状态暗示（可选，动态，不缓存）──────────────────
        if self.emotion_state_tracker:
            hint = self.emotion_state_tracker.get_prompt_hint()
            if hint:
                blocks.append({
                    "type": "text",
                    "text": f"<feeling>{hint}</feeling>"
                })

        return blocks

    def _build_emotion_directives(self, emotion_analysis: Dict) -> str:
        """
        从 BERT 输出 + config 映射生成指令文本。

        置信度门控：
          effective_confidence = BERT_prob * emotion_reliability[label]
          > strong  → 注入确定性指令
          > weak    → 注入不确定指令（"用户可能…"）
          ≤ weak   → 跳过情绪指令，让 LLM 自行判断
        """
        cfg = self.emotion_prompt_config
        parts = []

        emotion = emotion_analysis["emotion"]["primary"]
        intensity = emotion_analysis["emotion"]["intensity"]
        bert_prob = emotion_analysis["emotion"].get("primary_prob", 1.0)
        behavior = emotion_analysis["behavior"]["type"]
        tone = emotion_analysis["behavior"]["tone"]

        # 有效置信度 = BERT 输出概率 × 该标签的历史可靠度
        reliability = cfg.emotion_reliability.get(emotion, 0.7)
        effective_conf = bert_prob * reliability
        strong_thr = cfg.confidence_thresholds.get("strong", 0.5)
        weak_thr = cfg.confidence_thresholds.get("weak", 0.3)

        # 查 emotion_map
        emotion_directive = cfg.emotion_map.get(emotion, "")
        if emotion_directive:
            if effective_conf >= strong_thr:
                # 高置信度：确定性注入
                if intensity >= cfg.intensity_levels.get("high_min", 0.7):
                    emotion_directive = f"（强烈）{emotion_directive}"
                parts.append(emotion_directive)
            elif effective_conf >= weak_thr:
                # 中置信度：不确定注入，去掉"用户"开头避免"用户可能用户..."重复
                stripped = emotion_directive.lstrip("用户")
                parts.append(f"好像{stripped}，但也说不准")

        # behavior/tone head 未经专项训练，预测为噪声，暂不注入
        # 待 behavior/tone 标签补全并重新训练后再启用

        directive = "。".join(parts) if parts else ""
        logger.debug(
            f"[情感层] 指令注入: 情绪={emotion}(eff_conf={effective_conf:.2f}) "
            f"{'「' + directive + '」' if directive else '(跳过，置信度不足)'}"
        )
        return directive

    def _adaptive_max_tokens(
        self,
        emotion_analysis: Optional[Dict],
        client: Optional["LLMClient"] = None
    ) -> int:
        """
        根据情绪动态调整 max_tokens。

        策略：
        - config.max_tokens 是理论上限（400000），权重控制实际使用量
        - 日常对话：0.0125-0.03 → 5K-12K tokens（保证对话质量和自然展开）
        - 详细回答（curiosity）：0.0375 → 15K tokens
        - 高强度情绪：+50% 加成
        - 最终不超过 config.max_tokens（在极端情况下 LLM 可自主决定是否用满）
        """
        c = client or self.llm_client
        ceiling = c.config.max_tokens

        if emotion_analysis is None:
            return int(ceiling * 0.02)  # 默认 8K tokens

        emotion = emotion_analysis["emotion"]["primary"]
        intensity = emotion_analysis["emotion"]["intensity"]

        weight = _EMOTION_TOKEN_WEIGHTS.get(emotion, 0.02)

        # 高强度情绪加成
        if intensity >= 0.7:
            weight *= 1.5

        result = int(ceiling * weight)
        result = min(result, ceiling)  # 不超过配置上限
        result = max(result, 100)      # 最低 100 tokens

        logger.debug(
            f"adaptive max_tokens: ceiling={ceiling} × {weight:.5f} "
            f"(emotion={emotion} intensity={intensity:.2f}) = {result}"
        )
        return result

    # ── 群聊 LLM 路由 ────────────────────────────────────────────────────

    # 需要升级到主 LLM 的情绪（需要细腻情感处理）
    _ESCALATE_EMOTIONS = frozenset({"sadness", "fear", "anger", "tenderness"})

    def _route_llm_client(
        self,
        chat_mode: ChatMode,
        emotion_analysis: Optional[Dict],
        is_mentioned: bool,
    ) -> "LLMClient":
        """
        根据对话模式和 BERT 分析选择 LLM 客户端。

        私聊：始终用主 LLM（Claude）
        群聊：默认副 LLM（DeepSeek），以下情况升级到主 LLM：
          - 被 @ 提及（直接对话，用户期望高质量回复）
          - 高置信度的强情绪（需要细腻的情感处理）
          - 副 LLM 不可用时回退到主 LLM
        """
        # 私聊 or 没有副 LLM → 主 LLM
        if chat_mode == ChatMode.PRIVATE or self.llm_client_secondary is None:
            return self.llm_client

        # 群聊：被 @ → 主 LLM
        if is_mentioned:
            logger.debug("路由: @提及 → 主 LLM")
            return self.llm_client

        # 群聊：检查 BERT 是否检测到需要升级的强情绪
        if emotion_analysis and self.emotion_prompt_config:
            emotion = emotion_analysis["emotion"]["primary"]
            bert_prob = emotion_analysis["emotion"].get("primary_prob", 0.0)
            cfg = self.emotion_prompt_config
            reliability = cfg.emotion_reliability.get(emotion, 0.7)
            eff = bert_prob * reliability
            strong_thr = cfg.confidence_thresholds.get("strong", 0.5)

            if emotion in self._ESCALATE_EMOTIONS and eff >= strong_thr:
                logger.debug(
                    f"路由: 强情绪 {emotion}(eff={eff:.2f}) → 主 LLM"
                )
                return self.llm_client

        # 群聊默认 → 副 LLM
        logger.debug("路由: 群聊默认 → 副 LLM")
        return self.llm_client_secondary

    def generate_response(self, user_input: str,
                          recalled_context: str = "",
                          history: Optional[List[Dict]] = None,
                          emotion_analysis: Optional[Dict] = None,
                          client: Optional["LLMClient"] = None) -> str:
        """
        使用大模型生成回复。

        Args:
            user_input: 当前用户输入
            recalled_context: L2/L3/L4 召回（进 system prompt）
            history: L1 原文 messages 列表（进 messages 数组）
            emotion_analysis: BERT 情绪分析结果（注入 system prompt）
            client: 指定 LLM 客户端（路由选择的结果）
        """
        c = client or self.llm_client
        max_tokens = self._adaptive_max_tokens(emotion_analysis, c)

        # 情感分析全量日志（供研究/调参，结构化数据不进 prompt）
        if emotion_analysis:
            em = emotion_analysis["emotion"]
            bh = emotion_analysis["behavior"]
            parts = [
                f"情绪={em['primary']}(prob={em.get('primary_prob', 0):.2f})",
                f"强度={em['intensity']:.2f}",
                f"行为={bh['type']} 语气={bh['tone']}",
            ]
            fusion_meta = em.get("_fusion_meta")
            if fusion_meta:
                parts.append(
                    f"融合[bert={fusion_meta.get('bert_label')} "
                    f"llm={fusion_meta.get('llm_label')} "
                    f"一致={fusion_meta.get('agreement')}]"
                )
            logger.debug(f"[情感层] {' | '.join(parts)}")

        # 情绪状态机参数调节
        temperature = None
        top_p = None
        if self.emotion_state_tracker:
            adj = self.emotion_state_tracker.get_param_adjustments(
                base_temp=c.config.temperature,
                base_tokens=max_tokens,
                base_top_p=c.config.top_p,
            )
            temperature = adj["temperature"]
            max_tokens = adj["max_tokens"]
            top_p = adj["top_p"]

            st = self.emotion_state_tracker.state
            hint = self.emotion_state_tracker.get_prompt_hint()
            logger.debug(
                f"[情绪状态机] v={st.valence:.2f} a={st.arousal:.2f} "
                f"→ label={st.last_emotion} | "
                f"Δtemp={temperature - c.config.temperature:+.2f} "
                f"Δtop_p={top_p - c.config.top_p:+.2f} | "
                f"注入暗示: {hint or '(无)'}"
            )

        # Anthropic 使用多块 system prompt 优化缓存
        if c.provider == LLMProvider.ANTHROPIC:
            system_blocks = self.build_system_prompt_blocks(recalled_context, emotion_analysis)
            return c.generate(
                system_blocks=system_blocks,
                user_input=user_input,
                history=history,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
        else:
            # 其他 provider 使用单块 system prompt
            system_prompt = self.build_system_prompt(recalled_context, emotion_analysis)
            return c.generate(
                system_prompt=system_prompt,
                user_input=user_input,
                history=history,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
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
             is_mentioned: bool = True,
             chat_mode: ChatMode = ChatMode.PRIVATE,
             use_fusion: bool = None) -> Dict:
        """
        完整对话流程

        Args:
            user_input: 用户输入
            verbose: 是否输出详细信息
            is_mentioned: 是否被 @ 提及（群聊场景传入）
            chat_mode: 对话模式（PRIVATE=私聊纯 Claude，GROUP=群聊 BERT 路由）
            use_fusion: 是否启用情绪融合（None=使用配置默认值，True/False=强制覆盖）

        Returns:
            {
                "response": 回复文本，若 should_respond=False 则为 None,
                "emotion": BERT 情绪分析结果（旁路元数据）,
                "behavior": BERT 行为分析结果（旁路元数据）,
                "should_respond": bool,
                "debug_info": {...}
            }
        """
        # 决定是否启用融合：优先使用显式参数，否则使用配置默认值
        if use_fusion is None:
            use_fusion = (
                self.emotion_fusion_config is not None
                and self.emotion_fusion_config.enabled
                and self.emotion_fusion_config.use_by_default
            )

        # ── 旁路：BERT+LLM 情绪分析 与 记忆召回 并行执行 ───────────────────
        # 两者互不依赖，可同时进行以节省约 1-2s（情绪融合 LLM 调用耗时）
        if isinstance(self.memory, HierarchicalMemoryManager):
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=2) as _pool:
                _f_emotion = _pool.submit(
                    self._analyze_emotion_with_fusion, user_input, use_fusion
                )
                _f_recalled = _pool.submit(
                    self.memory.get_system_context, user_input
                )
                emotion_behavior = _f_emotion.result()
                recalled = _f_recalled.result()
            history = self.memory.get_messages_history()
            logger.debug(f"记忆 {self.memory.l1_usage} L1轮次={len(self.memory.working_memory)}")
        else:
            emotion_behavior = self._analyze_emotion_with_fusion(user_input, use_fusion=use_fusion)
            recalled = self.memory.format_context(num_turns=5)
            history = None

        if emotion_behavior is not None:
            intensity = emotion_behavior["emotion"]["intensity"]
            behavior_type = emotion_behavior["behavior"]["type"]
            emotion_primary = emotion_behavior["emotion"]["primary"]
            tone = emotion_behavior["behavior"]["tone"]
            logger.debug(
                f"BERT 情绪={emotion_primary} "
                f"强度={intensity:.2f} 行为={behavior_type}"
            )
        else:
            # BERT 不可用，使用默认值
            intensity = 0.5
            behavior_type = "respond_positive"
            emotion_primary = "neutral"
            tone = "calm"
            emotion_behavior = {
                "emotion": {"primary": emotion_primary, "intensity": intensity},
                "behavior": {"type": behavior_type, "tone": tone},
            }
            logger.debug("BERT 不可用，使用默认情绪值")

        # ── 注意力判断 ────────────────────────────────────────────────────
        respond = self.should_respond(emotion_behavior, is_mentioned)
        if not respond:
            logger.debug("注意力判断：跳过回复，仅记录记忆")

        # ── LLM 路由 + 生成 ──────────────────────────────────────────────
        response = None
        routed_client = self._route_llm_client(
            chat_mode, emotion_behavior, is_mentioned
        )
        if respond:
            response = self.generate_response(
                user_input, recalled, history,
                emotion_analysis=emotion_behavior,
                client=routed_client,
            )

        # ── 情绪状态机更新 ──────────────────────────────────────────────
        if self.emotion_state_tracker and response:
            # BERT 分析 AI 自身输出
            ai_emotion = emotion_primary
            ai_intensity = intensity
            if self.small_model:
                ai_result = self._analyze_emotion_with_fusion(
                    response, use_fusion=False
                )
                if ai_result:
                    ai_emotion = ai_result["emotion"]["primary"]
                    ai_intensity = ai_result["emotion"]["intensity"]

            self.emotion_state_tracker.update(
                user_emotion=emotion_primary,
                user_intensity=intensity,
                ai_emotion=ai_emotion,
                ai_intensity=ai_intensity,
            )

            # 每 N 轮持久化一次（后台线程，不阻塞响应）
            tc = self.emotion_state_tracker.state.turn_count
            interval = self.emotion_state_tracker.config.save_interval_turns
            if interval > 0 and tc % interval == 0:
                from concurrent.futures import ThreadPoolExecutor
                ThreadPoolExecutor(max_workers=1).submit(self._save_emotion_state)

        # ── 记忆写入（无论是否回复都记录，群聊静默观察也积累上下文）──────
        turn = ConversationTurn(
            user_input=user_input,
            emotion=emotion_primary,
            intensity=intensity,
            behavior=behavior_type,
            tone=tone,
            response=response or "",
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )
        self.memory.add(turn)

        return {
            "response": response,
            "emotion": emotion_behavior["emotion"],
            "behavior": emotion_behavior["behavior"],
            "should_respond": respond,
            "debug_info": {
                "model_provider": routed_client.provider.value,
                "model_name": routed_client.model,
                "chat_mode": chat_mode.value,
            },
        }

    def generate_proactive(
        self,
        trigger: str,
        chat_mode: ChatMode = ChatMode.PRIVATE,
        decision_hint: Optional['ProactiveDecision'] = None
    ) -> Optional[str]:
        """
        主动发言生成。没有用户输入，由 Agent 事件循环触发。

        Args:
            trigger: 触发原因描述（如 "对话已空闲5分钟"）
            chat_mode: 对话模式
            decision_hint: 主动决策模块的指导（可选）
        Returns:
            生成的主动发言文本，或 None（若决定不说话）
        """
        # 记忆上下文
        if isinstance(self.memory, HierarchicalMemoryManager):
            recalled = self.memory.get_system_context(query=trigger)
            history = self.memory.get_messages_history()
        else:
            recalled = self.memory.format_context(num_turns=5)
            history = None

        # 构建 system prompt，注入触发原因
        system_prompt = self.build_system_prompt(recalled)
        system_prompt += (
            f"\n<proactive_trigger>{trigger}</proactive_trigger>"
            "\n你可以主动找话题聊，或者接上之前的对话继续说。"
            "如果实在没什么好说的，回复空字符串即可。"
        )

        # 注入决策指导
        if decision_hint:
            hint_text = (
                f"\n[主动发言指导]\n"
                f"意图: {decision_hint.intent}\n"
                f"话题提示: {decision_hint.topic_hint}\n"
                f"建议语气: {decision_hint.tone}"
            )
            system_prompt += hint_text

        # 主动发言用主 LLM 保质量
        max_tokens = self.llm_client.config.max_tokens
        try:
            response = self.llm_client.generate(
                system_prompt=system_prompt,
                user_input="...",
                history=history,
                max_tokens=max_tokens,
            )
        except Exception as e:
            logger.error(f"主动发言生成失败: {e}")
            return None

        if not response or not response.strip():
            return None

        # 写入记忆
        turn = ConversationTurn(
            user_input="",
            emotion="neutral",
            intensity=0.0,
            behavior="respond_positive",
            tone="calm",
            response=response,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )
        self.memory.add(turn)

        return response

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
        """会话结束：写入长期记忆 + 保存情绪状态 + 关闭 Qdrant 连接"""
        # 情绪状态持久化（在记忆系统关闭前写入）
        if (self.emotion_state_tracker
                and self.emotion_state_tracker.config.persist_to_l4):
            self._save_emotion_state()

        if isinstance(self.memory, HierarchicalMemoryManager):
            logger.info("正在写入长期记忆...")
            self.memory.close_session()
            logger.info("长期记忆已保存。")

            # 主动关闭 Qdrant 客户端，避免 Python 退出时 __del__ 报 ImportError
            mem0 = getattr(self.memory, 'mem0', None)
            if mem0:
                for attr in ('vector_store', '_telemetry_vector_store'):
                    vs = getattr(mem0, attr, None)
                    if vs and hasattr(vs, 'client'):
                        try:
                            vs.client.close()
                        except Exception:
                            pass

    # ── 情绪状态持久化 ────────────────────────────────────────────────

    def _load_emotion_state(self) -> Optional[EmotionState]:
        """从本地 JSON 文件恢复情绪状态（无 LLM 调用）"""
        path = self._emotion_state_path()
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            state = EmotionState(
                valence=data.get("valence", 0.0),
                arousal=data.get("arousal", 0.1),
                turn_count=data.get("turn_count", 0),
                last_emotion=data.get("last_emotion", "neutral"),
            )
            logger.info(
                f"从文件恢复情绪状态: v={state.valence:.2f} "
                f"a={state.arousal:.2f} label={state.last_emotion}"
            )
            return state
        except Exception as e:
            logger.warning(f"加载情绪状态失败（使用默认值）: {e}")
            return None

    def _save_emotion_state(self):
        """将情绪状态写入本地 JSON 文件（快速，无 LLM 调用）"""
        if not self.emotion_state_tracker:
            return
        try:
            path = self._emotion_state_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            data = self.emotion_state_tracker.to_dict()
            path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            logger.debug(
                f"情绪状态已保存: v={data['valence']:.2f} a={data['arousal']:.2f}"
            )
        except Exception as e:
            logger.warning(f"保存情绪状态失败: {e}")

    def _emotion_state_path(self) -> "Path":
        from pathlib import Path
        if isinstance(self.memory, HierarchicalMemoryManager):
            base = Path(self.memory.config.vector_store_path).parent
        else:
            base = project_root
        return base / "emotion_state.json"


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
