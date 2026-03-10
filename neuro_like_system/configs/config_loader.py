"""
配置加载器

优先级：环境变量 > config.json > 代码默认值
敏感信息（API key）始终优先从环境变量读取，config.json 中的 key 仅作后备。
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from configs.model_config import (
    AgentConfig,
    AnnotationAPIConfig,
    AttentionConfig,
    AudioConfig,
    EmotionFusionConfig,
    EmotionPromptConfig,
    EmotionStateConfig,
    ImageConfig,
    LLMConfig,
    LLMProvider,
    MemoryConfig,
    PersonalityConfig,
    ProactiveConfig,
    QQBotConfig,
    SchedulerConfig,
    SenseVoiceConfig,
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


def load_llm_vision_config(cfg: dict) -> Optional[LLMConfig]:
    """加载图片专用 LLM 配置，不存在则返回 None"""
    section = cfg.get("llm_vision")
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
    llm_section = cfg.get("llm", {})
    mem0_llm_provider = m.get("mem0_llm_provider", "openai").lower()
    # API key：优先用 memory 块自己显式配置的，否则 fallback 到同 provider 的主 LLM key
    mem0_provider_env = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai":    "OPENAI_API_KEY",
    }.get(mem0_llm_provider, "OPENAI_API_KEY")
    mem0_api_key = _resolve_api_key(
        mem0_provider_env,
        m.get("mem0_api_key", "") or llm_section.get("api_key", "")
    )
    mem0_base_url = m.get("mem0_base_url") or None

    return MemoryConfig(
        user_id=m.get("user_id", "owner"),
        vector_store_path=m.get("vector_store_path", "./data/qdrant_db"),
        collection_name=m.get("collection_name", "neuro_memory"),
        mem0_llm_provider=mem0_llm_provider,
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


def load_attention_config(cfg: dict) -> AttentionConfig:
    att = cfg.get("attention", {})
    return AttentionConfig(
        intensity_threshold=att.get("intensity_threshold", 0.7),
        cooldown_seconds=att.get("cooldown_seconds", 60),
        track_mentioned_users=att.get("track_mentioned_users", True),
        mentioned_user_ttl=att.get("mentioned_user_ttl", 300),
        context_window_messages=att.get("context_window_messages", 10),
        non_focus_reply_interval=att.get("non_focus_reply_interval", 180),
        non_focus_max_token_ratio=att.get("non_focus_max_token_ratio", 0.5),
    )


def load_agent_config(cfg: dict) -> AgentConfig:
    a = cfg.get("agent", {})
    return AgentConfig(
        proactive_level=a.get("proactive_level", "off"),
        idle_threshold_seconds=a.get("idle_threshold_seconds", 300),
        proactive_interval_seconds=a.get("proactive_interval_seconds", 30),
        time_awareness=a.get("time_awareness", True),
        tick_interval=a.get("tick_interval", 2.0),
    )


def load_emotion_fusion_config(cfg: dict) -> EmotionFusionConfig:
    ef = cfg.get("emotion_fusion", {})
    return EmotionFusionConfig(
        enabled=ef.get("enabled", True),
        use_by_default=ef.get("use_by_default", True),
        w_bert=ef.get("w_bert", 0.6),
        w_llm=ef.get("w_llm", 0.4),
        bias=ef.get("bias", 0.0),
        skip_llm_threshold=ef.get("skip_llm_threshold", 0.85),
        llm_timeout=ef.get("llm_timeout", 5.0),
        llm_temperature=ef.get("llm_temperature", 0.3),
    )


def load_emotion_state_config(cfg: dict) -> Optional[EmotionStateConfig]:
    es = cfg.get("emotion_state")
    if not es:
        return None
    return EmotionStateConfig(
        alpha=es.get("alpha", 0.75),
        beta=es.get("beta", 0.25),
        gamma=es.get("gamma", 0.25),
        delta=es.get("delta", 0.15),
        baseline_valence=es.get("baseline_valence", 0.15),
        baseline_arousal=es.get("baseline_arousal", 0.28),
        kappa=es.get("kappa", 0.05),
        negativity_bias=es.get("negativity_bias", 1.3),
        noise_sigma=es.get("noise_sigma", 0.05),
        injection_threshold=es.get("injection_threshold", 0.12),
        save_interval_turns=es.get("save_interval_turns", 5),
        persist_to_l4=es.get("persist_to_l4", True),
    )


def load_proactive_config(cfg: dict) -> ProactiveConfig:
    p = cfg.get("proactive", {})
    return ProactiveConfig(
        enabled=p.get("enabled", True),
        decision_provider=p.get("decision_provider", "deepseek"),
        decision_model=p.get("decision_model", "deepseek-chat"),
        decision_temperature=p.get("decision_temperature", 0.3),
        decision_timeout=p.get("decision_timeout", 5.0),
        confidence_threshold=p.get("confidence_threshold", 0.6),
        recent_turns_limit=p.get("recent_turns_limit", 8),
        l4_memory_limit=p.get("l4_memory_limit", 3),
        idle_trigger_hours=p.get("idle_trigger_hours", 2.0),
        response_wait_minutes=p.get("response_wait_minutes", 30),
        min_interval_seconds=p.get("min_interval_seconds", 30),
    )


def load_image_config(cfg: dict) -> ImageConfig:
    im = cfg.get("image", {})
    return ImageConfig(
        enabled=im.get("enabled", True),
        cache_dir=im.get("cache_dir", "./data/image_cache"),
        max_download_size_bytes=im.get("max_download_size_bytes", 10_485_760),
        max_dimension=im.get("max_dimension", 1568),
        max_images_per_message=im.get("max_images_per_message", 5),
        cache_ttl_seconds=im.get("cache_ttl_seconds", 86400),
        download_timeout=im.get("download_timeout", 15.0),
    )


def load_qq_bot_config(cfg: dict) -> QQBotConfig:
    q = cfg.get("qq_bot", {})
    return QQBotConfig(
        ws_host=q.get("ws_host", "0.0.0.0"),
        ws_port=q.get("ws_port", 8080),
        ws_path=q.get("ws_path", "/xm"),
        bot_qq=q.get("bot_qq", 0),
        owner_qq=q.get("owner_qq", 0),
        owner_name=q.get("owner_name", ""),
        admin_qq=q.get("admin_qq", []),
        command_prefix=q.get("command_prefix", "/"),
        reply_with_at=q.get("reply_with_at", True),
        max_message_length=q.get("max_message_length", 500),
    )


def load_audio_config(cfg: dict) -> AudioConfig:
    a = cfg.get("audio", {})
    if not a:
        return AudioConfig(enabled=False)

    # 解析 cosyvoice 子配置
    cv = a.get("cosyvoice", {})

    return AudioConfig(
        enabled=a.get("enabled", True),
        tts_provider=a.get("tts_provider", "cosyvoice"),
        cosyvoice_repo_dir=cv.get("repo_dir", ""),
        cosyvoice_model_dir=cv.get("model_dir", "./models/CosyVoice2-0.5B"),
        ref_audio_dir=cv.get("ref_audio_dir", "./data/audio_refs"),
        default_ref_audio=cv.get("default_ref_audio", "default.wav"),
        default_ref_text=cv.get("default_ref_text", ""),
        sample_rate=cv.get("sample_rate", 22050),
        speed=cv.get("speed", 1.0),
        emotion_ref_map=a.get("emotion_ref_map", {}),
        cache_dir=a.get("cache_dir", "./data/audio_cache"),
        cache_enabled=a.get("cache_enabled", True),
        auto_play=a.get("auto_play", False),
    )


def load_sensevoice_config(cfg: dict) -> SenseVoiceConfig:
    s = cfg.get("sensevoice", {})
    if not s:
        return SenseVoiceConfig(enabled=False)
    return SenseVoiceConfig(
        enabled=s.get("enabled", True),
        model_id=s.get("model_id", "iic/SenseVoiceSmall"),
        model_dir=s.get("model_dir"),
        device=s.get("device", "cuda"),
        language=s.get("language", "auto"),
        use_emotion=s.get("use_emotion", True),
        use_vad=s.get("use_vad", True),
        batch_size=s.get("batch_size", 1),
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
        attention: AttentionConfig,
        llm_secondary: Optional[LLMConfig] = None,
        llm_vision: Optional[LLMConfig] = None,
        agent: Optional[AgentConfig] = None,
        emotion_fusion: Optional[EmotionFusionConfig] = None,
        emotion_state_config: Optional[EmotionStateConfig] = None,
        proactive: Optional[ProactiveConfig] = None,
        qq_bot: Optional[QQBotConfig] = None,
        image: Optional[ImageConfig] = None,
        audio: Optional[AudioConfig] = None,
        sensevoice: Optional[SenseVoiceConfig] = None,
    ):
        self.llm = llm
        self.llm_secondary = llm_secondary
        self.llm_vision = llm_vision
        self.personality = personality
        self.memory = memory
        self.emotion_prompts = emotion_prompts
        self.annotation = annotation
        self.small_model_checkpoint = small_model_checkpoint
        self.device = device
        self.attention = attention
        self.agent = agent or AgentConfig()
        self.emotion_fusion = emotion_fusion or EmotionFusionConfig()
        self.emotion_state_config = emotion_state_config
        self.proactive = proactive or ProactiveConfig()
        self.qq_bot = qq_bot or QQBotConfig()
        self.image = image or ImageConfig()
        self.audio = audio or AudioConfig(enabled=False)
        self.sensevoice = sensevoice or SenseVoiceConfig(enabled=False)

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

        return cls(
            llm=load_llm_config(cfg),
            llm_secondary=load_llm_secondary_config(cfg),
            llm_vision=load_llm_vision_config(cfg),
            personality=load_personality_config(cfg),
            memory=load_memory_config(cfg),
            emotion_prompts=load_emotion_prompt_config(cfg),
            annotation=load_annotation_config(cfg),
            small_model_checkpoint=sm.get(
                "checkpoint_path",
                "./checkpoints/joint_model/best_model.pt"
            ),
            device=sm.get("device", "cpu"),
            attention=load_attention_config(cfg),
            agent=load_agent_config(cfg),
            emotion_fusion=load_emotion_fusion_config(cfg),
            emotion_state_config=load_emotion_state_config(cfg),
            proactive=load_proactive_config(cfg),
            qq_bot=load_qq_bot_config(cfg),
            image=load_image_config(cfg),
            audio=load_audio_config(cfg),
            sensevoice=load_sensevoice_config(cfg),
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


# ============== 调度器配置加载 ==============

def load_scheduler_config(path: str) -> Tuple[SchedulerConfig, List[Dict]]:
    """
    加载调度器配置文件。

    Args:
        path: scheduler_config.json 路径

    Returns:
        (SchedulerConfig, persona_entries)
        persona_entries: [{"config": "config.json", "contexts": [...]}, ...]

    Raises:
        FileNotFoundError: 文件不存在
        ValueError: 配置格式错误
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"调度器配置文件不存在: {path}")

    with open(p, encoding="utf-8") as f:
        raw = json.load(f)

    sched_cfg = SchedulerConfig(
        max_concurrent_llm=raw.get("max_concurrent_llm", 3),
        llm_acquire_timeout=raw.get("llm_acquire_timeout", 30.0),
        health_check_interval=raw.get("health_check_interval", 60.0),
        default_persona=raw.get("default_persona"),
    )

    personas = raw.get("personas", [])
    if not personas:
        raise ValueError("scheduler_config.json 的 personas 列表不能为空")

    entries = []
    for i, item in enumerate(personas):
        cfg_path = item.get("config")
        if not cfg_path:
            raise ValueError(f"personas[{i}] 缺少 'config' 字段")
        entries.append({
            "config": cfg_path,
            "contexts": item.get("contexts", []),
        })

    return sched_cfg, entries
