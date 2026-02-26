"""
配置加载器

优先级：环境变量 > config.json > 代码默认值
敏感信息（API key）始终优先从环境变量读取，config.json 中的 key 仅作后备。
"""

import json
import os
from pathlib import Path
from typing import Optional

from configs.model_config import (
    AnnotationAPIConfig,
    EmotionPromptConfig,
    LLMConfig,
    LLMProvider,
    MemoryConfig,
    PersonalityConfig,
)


# config.json 默认搜索路径：项目根目录（neuro_like_system/）
_DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config.json"


def _find_config(path: Optional[str] = None) -> Optional[Path]:
    p = Path(path) if path else _DEFAULT_CONFIG_PATH
    return p if p.exists() else None


def _resolve_api_key(env_var: str, json_value: str) -> Optional[str]:
    """环境变量优先，json 值作后备，均为空则返回 None"""
    return os.environ.get(env_var) or json_value or None


def _load_llm_from_dict(llm: dict) -> LLMConfig:
    """从一个 LLM 配置字典加载 LLMConfig（llm / llm_secondary 共用）"""
    provider_str = llm.get("provider", "custom").lower()
    provider = LLMProvider(provider_str)

    env_map = {
        LLMProvider.OPENAI:    "OPENAI_API_KEY",
        LLMProvider.DEEPSEEK:  "DEEPSEEK_API_KEY",
        LLMProvider.ANTHROPIC: "ANTHROPIC_API_KEY",
        LLMProvider.CUSTOM:    "CUSTOM_API_KEY",
    }
    api_key = _resolve_api_key(env_map.get(provider, "OPENAI_API_KEY"),
                               llm.get("api_key", ""))

    return LLMConfig(
        provider=provider,
        api_key=api_key,
        base_url=llm.get("base_url") or None,
        model=llm.get("model", "gpt-5.2-instant"),
        temperature=llm.get("temperature", 0.8),
        max_tokens=llm.get("max_tokens", 500),
        top_p=llm.get("top_p", 0.95),
        frequency_penalty=llm.get("frequency_penalty", 0.0),
        presence_penalty=llm.get("presence_penalty", 0.0),
        timeout=llm.get("timeout", 30),
        max_retries=llm.get("max_retries", 3),
        retry_delay=llm.get("retry_delay", 1.0),
        retry_backoff=llm.get("retry_backoff", 2.0),
        retry_max_delay=llm.get("retry_max_delay", 60.0),
        use_responses_api=llm.get("use_responses_api", False),
    )


def load_llm_config(cfg: dict) -> LLMConfig:
    return _load_llm_from_dict(cfg.get("llm", {}))


def load_llm_secondary_config(cfg: dict) -> Optional[LLMConfig]:
    """加载副 LLM 配置（群聊廉价模型），不存在则返回 None"""
    section = cfg.get("llm_secondary")
    if not section:
        return None
    return _load_llm_from_dict(section)


def load_personality_config(cfg: dict) -> PersonalityConfig:
    p = cfg.get("personality", {})
    return PersonalityConfig(
        name=p.get("name", "Neuro"),
        openness=p.get("openness", 0.8),
        conscientiousness=p.get("conscientiousness", 0.5),
        extraversion=p.get("extraversion", 0.7),
        agreeableness=p.get("agreeableness", 0.8),
        neuroticism=p.get("neuroticism", 0.3),
        emotional_baseline=p.get("emotional_baseline", "positive"),
        emotional_volatility=p.get("emotional_volatility", 0.4),
        humor_tendency=p.get("humor_tendency", 0.7),
        empathy_level=p.get("empathy_level", 0.8),
        curiosity_level=p.get("curiosity_level", 0.9),
        formality=p.get("formality", 0.3),
        verbosity=p.get("verbosity", 0.5),
        traits=p.get("traits", ["活泼", "好奇", "善良"]),
        description=p.get("description", ""),
    )


def load_memory_config(cfg: dict) -> MemoryConfig:
    m = cfg.get("memory", {})
    # Mem0 内部 LLM 的 key 复用主 LLM 的 key
    llm_section = cfg.get("llm", {})
    provider_str = llm_section.get("provider", "custom").lower()
    env_map = {
        "openai": "OPENAI_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "custom": "CUSTOM_API_KEY",
    }
    mem0_api_key = _resolve_api_key(
        env_map.get(provider_str, "OPENAI_API_KEY"),
        m.get("mem0_api_key", "") or llm_section.get("api_key", "")
    )
    mem0_base_url = (m.get("mem0_base_url") or
                     llm_section.get("base_url") or None)

    return MemoryConfig(
        vector_store_path=m.get("vector_store_path", "./data/qdrant_db"),
        collection_name=m.get("collection_name", "neuro_memory"),
        mem0_llm_model=m.get("mem0_llm_model", "gpt-5.2-instant"),
        mem0_llm_temperature=m.get("mem0_llm_temperature", 0.1),
        mem0_api_key=mem0_api_key,
        mem0_base_url=mem0_base_url,
        context_window_tokens=m.get("context_window_tokens", 128_000),
        compression_threshold=m.get("compression_threshold", 0.75),
        compression_ratio=m.get("compression_ratio", 0.5),
        l3_search_limit=m.get("l3_search_limit", 5),
        l4_search_limit=m.get("l4_search_limit", 3),
        relevance_threshold=m.get("relevance_threshold", 0.7),
    )


def load_emotion_prompt_config(cfg: dict) -> EmotionPromptConfig:
    ep = cfg.get("emotion_prompts", {})
    if not ep:
        return EmotionPromptConfig()
    intensity = ep.get("intensity_levels", {})
    ct = ep.get("confidence_thresholds", {})
    return EmotionPromptConfig(
        emotion_map=ep.get("emotion_map", {}),
        behavior_map=ep.get("behavior_map", {}),
        tone_map=ep.get("tone_map", {}),
        intensity_levels={
            "low_max": intensity.get("low_max", 0.4),
            "high_min": intensity.get("high_min", 0.7),
        },
        emotion_reliability=ep.get("emotion_reliability", {}),
        confidence_thresholds={
            "strong": ct.get("strong", 0.5),
            "weak": ct.get("weak", 0.3),
        },
    )


def load_annotation_config(cfg: dict) -> AnnotationAPIConfig:
    a = cfg.get("annotation", {})
    if not a:
        return AnnotationAPIConfig()

    primary_prov = a.get("primary_provider", "openai").lower()
    fallback_prov = a.get("fallback_provider", "deepseek").lower()

    env_map = {
        "openai": "OPENAI_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "custom": "CUSTOM_API_KEY",
    }
    primary_api_key = _resolve_api_key(
        env_map.get(primary_prov, "OPENAI_API_KEY"),
        a.get("primary_api_key", ""),
    )
    fallback_api_key = _resolve_api_key(
        env_map.get(fallback_prov, "DEEPSEEK_API_KEY"),
        a.get("fallback_api_key", ""),
    )

    return AnnotationAPIConfig(
        primary_provider=LLMProvider(primary_prov),
        primary_model=a.get("primary_model", "gpt-5.2-instant"),
        primary_api_key=primary_api_key,
        primary_base_url=a.get("primary_base_url") or None,
        fallback_provider=LLMProvider(fallback_prov),
        fallback_model=a.get("fallback_model", "deepseek-chat"),
        fallback_api_key=fallback_api_key,
        fallback_base_url=a.get("fallback_base_url") or None,
        batch_size=a.get("batch_size", 10),
        temperature=a.get("temperature", 0.3),
        max_retries=a.get("max_retries", 3),
    )


class AppConfig:
    """完整运行时配置，由 config.json 加载"""

    def __init__(
        self,
        llm: LLMConfig,
        personality: PersonalityConfig,
        memory: MemoryConfig,
        emotion_prompts: EmotionPromptConfig,
        annotation: AnnotationAPIConfig,
        small_model_checkpoint: str,
        device: str,
        attention_intensity_threshold: float,
        attention_cooldown_seconds: int,
        llm_secondary: Optional[LLMConfig] = None,
    ):
        self.llm = llm
        self.llm_secondary = llm_secondary
        self.personality = personality
        self.memory = memory
        self.emotion_prompts = emotion_prompts
        self.annotation = annotation
        self.small_model_checkpoint = small_model_checkpoint
        self.device = device
        self.attention_intensity_threshold = attention_intensity_threshold
        self.attention_cooldown_seconds = attention_cooldown_seconds

    @classmethod
    def load(cls, path: Optional[str] = None) -> "AppConfig":
        """
        从 config.json 加载完整配置。

        Args:
            path: 配置文件路径，默认为项目根目录下的 config.json

        Raises:
            FileNotFoundError: config.json 不存在时
        """
        config_path = _find_config(path)
        if config_path is None:
            raise FileNotFoundError(
                "找不到 config.json。\n"
                "请复制 config.example.json 为 config.json 并填写配置。"
            )

        with open(config_path, encoding="utf-8") as f:
            cfg = json.load(f)

        sm = cfg.get("small_model", {})
        att = cfg.get("attention", {})

        return cls(
            llm=load_llm_config(cfg),
            llm_secondary=load_llm_secondary_config(cfg),
            personality=load_personality_config(cfg),
            memory=load_memory_config(cfg),
            emotion_prompts=load_emotion_prompt_config(cfg),
            annotation=load_annotation_config(cfg),
            small_model_checkpoint=sm.get(
                "checkpoint_path",
                "./checkpoints/joint_model/best_model.pt"
            ),
            device=sm.get("device", "cpu"),
            attention_intensity_threshold=att.get("intensity_threshold", 0.7),
            attention_cooldown_seconds=att.get("cooldown_seconds", 60),
        )

    def __repr__(self) -> str:
        return (
            f"AppConfig(\n"
            f"  llm={self.llm.provider.value}:{self.llm.model},\n"
            f"  personality={self.personality.name},\n"
            f"  device={self.device},\n"
            f"  context_window={self.memory.context_window_tokens}\n"
            f")"
        )
