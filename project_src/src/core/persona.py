"""
人格实例

每个人格（persona）一个实例，持有独立的状态：
- 记忆管理器（独立 Qdrant collection）
- 情绪状态机（独立 OU 过程）
- 注意力追踪器
- Prompt 构建器

共享计算委托给 SharedInfra（BERT、LLM 客户端、融合引擎）。
"""

import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, TYPE_CHECKING

import torch

import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.logger import logger
from src.core.shared_infra import SharedInfra
from src.core.prompt_builder import PromptBuilder
from src.memory.memory_manager import HierarchicalMemoryManager
from src.core.emotion_state import EmotionStateTracker, EmotionState
from src.attention.attention_tracker import AttentionTracker
from src.vision import (
    VisualAnalysis,
    VisualEvent,
    VisualPerceptionConfig,
    derive_visual_emotion_signal,
    visual_event_memory_text,
)
from configs.model_config import (
    PersonalityConfig,
    EmotionPromptConfig,
    EmotionFusionConfig,
    EmotionStateConfig,
    AttentionConfig,
    MemoryConfig,
    LLMProvider,
    VisualPerceptionSettings,
)

if TYPE_CHECKING:
    from src.llm.client import LLMClient
    from src.attention.proactive_decision import ProactiveDecision
    from configs.config_loader import AppConfig


# 重用 inference_pipeline 中定义的公共类型
from src.core.inference_pipeline import ChatMode, ConversationTurn, MemoryManager

# 情绪 → max_tokens 权重
_EMOTION_TOKEN_WEIGHTS = {
    "neutral":    0.0125,
    "joy":        0.02,
    "excitement": 0.025,
    "sadness":    0.03,
    "fear":       0.03,
    "anger":      0.025,
    "disgust":    0.02,
    "surprise":   0.025,
    "tenderness": 0.025,
    "curiosity":  0.0375,
}

# 需要升级到主 LLM 的情绪
_ESCALATE_EMOTIONS = frozenset({"sadness", "fear", "anger", "tenderness"})


class PersonaInstance:
    """
    单个人格实例 — 持有每 persona 独有的状态，
    委托共享计算给 SharedInfra。
    """
    def __init__(
        self,
        infra: SharedInfra,
        personality: PersonalityConfig,
        memory_config: Optional[MemoryConfig] = None,
        emotion_prompt_config: Optional[EmotionPromptConfig] = None,
        emotion_fusion_config: Optional[EmotionFusionConfig] = None,
        emotion_state_config: Optional[EmotionStateConfig] = None,
        attention_config: Optional[AttentionConfig] = None,
        visual_perception_config: Optional[VisualPerceptionSettings] = None,
        time_awareness: bool = True,
    ):
        self.infra = infra
        self.personality = personality
        self.emotion_prompt_config = emotion_prompt_config
        self.emotion_fusion_config = emotion_fusion_config or infra.emotion_fusion_config
        self.attention_config = attention_config or AttentionConfig()
        self.visual_perception_config = visual_perception_config or VisualPerceptionSettings()
        self.visual_runtime_config = VisualPerceptionConfig.from_settings(
            self.visual_perception_config
        )

        # 人格向量（BERT 推理需要，跟随 BERT 设备避免每次推理搬运）
        self.personality_vector = torch.tensor(
            personality.to_embedding_vector(),
            dtype=torch.float32,
            device=torch.device(infra.bert.device) if infra.bert.available else None,
        )

        # Prompt 构建器
        self.prompt_builder = PromptBuilder(
            personality=personality,
            emotion_prompt_config=emotion_prompt_config,
            time_awareness=time_awareness,
        )

        # 记忆管理器（L1 独立，Mem0 后端全进程共享；user_id = 人格名用于记忆隔离）
        if memory_config is not None:
            self.memory = HierarchicalMemoryManager(
                config=memory_config,
                llm_client=infra.llm_pool.primary,
                user_id=personality.name,
            )
        else:
            self.memory = MemoryManager()

        # 情绪状态机（每 persona 独立）
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

        # 注意力追踪器（每 persona 独立）
        self.attention_tracker = AttentionTracker(self.attention_config)
        logger.info("注意力追踪器已启用")
        # 多人格模式标记
        self.muti_persona_mode = False
        # 后台线程池
        self._bg_executor = ThreadPoolExecutor(max_workers=1)
        # 视觉技能（按需注册）
        self._visual_skill_detector = None
        self._visual_skill_executor = None

    @classmethod
    def from_app_config(cls, infra: SharedInfra, config: "AppConfig") -> "PersonaInstance":
        """从 SharedInfra + AppConfig 创建 PersonaInstance（减少外部样板代码）"""
        return cls(
            infra=infra,
            personality=config.personality,
            memory_config=config.memory,
            emotion_prompt_config=config.emotion_prompts,
            emotion_fusion_config=config.emotion_fusion,
            emotion_state_config=config.emotion_state_config,
            attention_config=config.attention,
            visual_perception_config=config.visual_perception,
            time_awareness=config.agent.time_awareness,
        )

    # ── 便捷属性（兼容 AgentLoop 访问路径）────────────────────────────────

    @property
    def llm_client(self) -> "LLMClient":
        return self.infra.llm_pool.primary

    @property
    def llm_client_secondary(self) -> Optional["LLMClient"]:
        return self.infra.llm_pool.secondary

    @property
    def llm_client_vision(self) -> Optional["LLMClient"]:
        return self.infra.llm_pool.vision

    @property
    def small_model(self):
        return self.infra.bert.model

    def register_visual_skill(
        self,
        handler,
    ):
        """注册视觉技能处理器，启用按需视觉感知。"""
        from src.vision.visual_skill import VisualSkillDetector, VisualSkillExecutor

        self._visual_skill_detector = VisualSkillDetector()
        self._visual_skill_executor = VisualSkillExecutor(handler)

        if isinstance(self.memory, HierarchicalMemoryManager):
            try:
                self.memory.add_knowledge(
                    "[能力] 小橘具有实时视觉感知能力。"
                    "当用户询问关于视觉、画面、周围环境的问题时，"
                    "系统会自动获取最近的画面分析结果并注入对话上下文。"
                )
                logger.info("视觉能力描述已写入 L4")
            except Exception as exc:
                logger.warning(f"视觉能力 L4 写入失败（不影响功能）: {exc}")

        logger.info("视觉技能已注册")

    def handle_visual_event(self, event) -> Dict:
        """
        在视觉事件决定是否进入聊天链路前，先处理它对其他模块的影响。

        包括：
        - 作为弱刺激注入情绪状态机
        - 即时事件：按阈值将视觉观察写入 L4 知识库
        - 非即时事件：额外写入 L3 用户级记忆
        """
        cfg = self.visual_perception_config
        route_to_chat = bool(cfg.route_visual_events_to_chat)
        result = {
            "route_to_chat": route_to_chat,
            "emotion_signal": None,
            "memory_written": False,
        }

        visual_event = self._visual_event_from_agent_event(event)

        if cfg.inject_to_emotion_state and self.emotion_state_tracker is not None:
            signal = derive_visual_emotion_signal(
                visual_event,
                config=self.visual_runtime_config,
            )
            result["emotion_signal"] = signal
            if abs(signal["valence_delta"]) > 1e-6 or abs(signal["arousal_delta"]) > 1e-6:
                self.emotion_state_tracker.apply_stimulus(
                    valence_delta=signal["valence_delta"],
                    arousal_delta=signal["arousal_delta"],
                )
                logger.debug(
                    "视觉弱刺激已注入情绪状态机: "
                    f"dv={signal['valence_delta']:+.3f} "
                    f"da={signal['arousal_delta']:+.3f} "
                    f"emotion={signal['emotion']} intensity={signal['intensity']:.2f}"
                )

        if (
            cfg.persist_to_memory
            and isinstance(self.memory, HierarchicalMemoryManager)
            and visual_event.peak_score >= cfg.memory_peak_score_threshold
        ):
            memory_text = visual_event_memory_text(visual_event).strip()
            if memory_text:
                self.memory.add_visual_observation(
                    memory_text,
                    event_id=visual_event.event_id,
                )
                result["memory_written"] = True
                logger.debug(
                    f"视觉观察已写入 L4: event={visual_event.event_id} "
                    f"peak={visual_event.peak_score:.3f}"
                )

                if not route_to_chat:
                    try:
                        self.memory.add_visual_observation_l3(
                            memory_text,
                            event_id=visual_event.event_id,
                        )
                        logger.debug(
                            f"非即时视觉观察已写入 L3: event={visual_event.event_id}"
                        )
                    except Exception as exc:
                        logger.warning(f"视觉观察 L3 写入失败: {exc}")

        return result

    def _visual_event_from_agent_event(self, event) -> VisualEvent:
        metadata = getattr(event, "metadata", {}) or {}
        analysis_data = metadata.get("analysis") or {}
        analysis = None
        if analysis_data:
            analysis = VisualAnalysis(
                scene=str(analysis_data.get("scene", "")).strip(),
                facts=list(analysis_data.get("facts", [])),
                weak_interpretations=list(analysis_data.get("weak_interpretations", [])),
                memory_candidate=str(analysis_data.get("memory_candidate", "")).strip(),
                agent_hint=str(analysis_data.get("agent_hint", "")).strip(),
                raw_text=str(analysis_data.get("raw_text", "")).strip(),
            )

        content = str(getattr(event, "content", "")).strip()
        if content.startswith("[视觉事件]"):
            content = content[len("[视觉事件]"):].strip()

        if analysis is None and content:
            analysis = VisualAnalysis(agent_hint=content, raw_text=content)

        return VisualEvent(
            event_id=str(metadata.get("event_id", f"visual-agent-{int(time.time())}")),
            peak_frame_index=int(metadata.get("peak_frame_index", -1)),
            timestamp=float(metadata.get("timestamp", getattr(event, "timestamp", time.time()))),
            peak_score=float(metadata.get("peak_score", 0.0)),
            representative_frame_index=int(
                metadata.get("representative_frame_index", metadata.get("peak_frame_index", -1))
            ),
            keyframes=list(getattr(event, "images", []) or []),
            metrics=dict(metadata.get("metrics", {})),
            analysis=analysis,
            rate_limited=bool(metadata.get("rate_limited", False)),
        )

    # ── BERT + 融合 ─────────────────────────────────────────────────────────

    def analyze_emotion_behavior(self, text: str) -> Optional[Dict]:
        """BERT 情绪分析（委托给 SharedInfra）"""
        return self.infra.bert.predict(text, self.personality_vector)

    def _analyze_emotion_with_fusion(
        self, text: str, use_fusion: bool = False
    ) -> Optional[Dict]:
        """情绪分析（支持 BERT-only 或 BERT+LLM 融合）"""
        if not use_fusion or self.infra.emotion_fusion is None:
            return self.analyze_emotion_behavior(text)

        # 融合模式：先尝试 BERT 快速路径
        quick_bert = self.analyze_emotion_behavior(text)
        if quick_bert:
            bert_prob = quick_bert["emotion"].get("primary_prob", 0.0)
            emotion = quick_bert["emotion"]["primary"]
            reliability = self.emotion_prompt_config.emotion_reliability.get(
                emotion, 0.7
            )
            eff_conf = bert_prob * reliability

            if eff_conf >= self.emotion_fusion_config.skip_llm_threshold:
                logger.debug(
                    f"Skip LLM: BERT eff_conf={eff_conf:.2f} "
                    f">= {self.emotion_fusion_config.skip_llm_threshold}"
                )
                return quick_bert

            bert_result = quick_bert
        else:
            bert_result = None

        # 并行调用 LLM
        llm_result = None
        from concurrent.futures import (
            ThreadPoolExecutor,
            TimeoutError as FutureTimeoutError,
        )

        def _classify_with_gate():
            with self.infra.llm_gate():
                return self.infra.llm_emotion_classifier.classify(
                    text, self.emotion_fusion_config.llm_timeout,
                )

        with ThreadPoolExecutor(max_workers=1) as executor:
            llm_future = executor.submit(_classify_with_gate)
            try:
                llm_result = llm_future.result(
                    timeout=self.emotion_fusion_config.llm_timeout + 1
                )
            except FutureTimeoutError:
                logger.warning("LLM emotion classification timeout")
            except Exception as e:
                logger.warning(f"LLM emotion classification failed: {e}")

        # 融合
        if bert_result:
            fused = self.infra.emotion_fusion.fuse(bert_result, llm_result)
            if llm_result:
                logger.debug(
                    f"Fused: {fused['emotion']['primary']} "
                    f"(BERT: {bert_result['emotion']['primary']}, "
                    f"LLM: {llm_result['emotion']}, "
                    f"agree: {fused['emotion']['_fusion_meta']['agreement']})"
                )
            return fused

        return self.analyze_emotion_behavior(text)

    # ── LLM 路由 ───────────────────────────────────────────────────────────

    def _route_llm_client(
        self,
        chat_mode: ChatMode,
        emotion_analysis: Optional[Dict],
        is_mentioned: bool,
        images: Optional[list] = None,
        in_attention_focus: bool = False,
    ) -> "LLMClient":
        """根据对话模式和 BERT 分析选择 LLM 客户端"""
        pool = self.infra.llm_pool

        # 有图片且 vision client 存在 → vision LLM
        if images and pool.vision is not None:
            logger.debug("路由: 图片 → vision LLM")
            return pool.vision

        # 私聊 or 没有副 LLM → 主 LLM
        if chat_mode == ChatMode.PRIVATE or pool.secondary is None:
            return pool.primary

        # 群聊：被 @ → 主 LLM
        if is_mentioned:
            logger.debug("路由: @提及 → 主 LLM")
            return pool.primary

        # 群聊：在注意力焦点内 → 主 LLM
        if in_attention_focus:
            logger.debug("路由: 注意力焦点内 → 主 LLM")
            return pool.primary

        # 群聊：检查 BERT 是否检测到需要升级的强情绪
        if emotion_analysis and self.emotion_prompt_config:
            emotion = emotion_analysis["emotion"]["primary"]
            bert_prob = emotion_analysis["emotion"].get("primary_prob", 0.0)
            cfg = self.emotion_prompt_config
            reliability = cfg.emotion_reliability.get(emotion, 0.7)
            eff = bert_prob * reliability
            strong_thr = cfg.confidence_thresholds.get("strong", 0.5)

            if emotion in _ESCALATE_EMOTIONS and eff >= strong_thr:
                logger.debug(
                    f"路由: 强情绪 {emotion}(eff={eff:.2f}) → 主 LLM"
                )
                return pool.primary

        logger.debug("路由: 群聊默认 → 副 LLM")
        return pool.secondary

    # ── 图片信息提取 ──────────────────────────────────────────────────────

    def _extract_image_context(self, images: list, user_input: str = "") -> str:
        """调用 vision LLM 提取图片关键信息，返回描述文本"""
        vision_client = self.infra.llm_pool.vision
        if vision_client is None:
            return ""

        extraction_prompt = (
            "请简洁描述这张图片的关键内容和特征（100字以内）。"
            "只描述你看到的事实，不要做推测，不要回复其他内容。"
        )

        with self.infra.llm_gate():
            description = vision_client.generate(
                system_prompt=extraction_prompt,
                user_input=user_input or "请描述这张图片",
                images=images,
                max_tokens=200,
                temperature=0.3,
            )
        return description.strip()

    # ── max_tokens 自适应 ──────────────────────────────────────────────────

    def _adaptive_max_tokens(
        self,
        emotion_analysis: Optional[Dict],
        client: Optional["LLMClient"] = None,
    ) -> int:
        """根据情绪动态调整 max_tokens"""
        c = client or self.llm_client
        ceiling = c.config.max_tokens

        if emotion_analysis is None:
            return int(ceiling * 0.02)

        emotion = emotion_analysis["emotion"]["primary"]
        intensity = emotion_analysis["emotion"]["intensity"]
        weight = _EMOTION_TOKEN_WEIGHTS.get(emotion, 0.02)

        if intensity >= 0.7:
            weight *= 1.5

        result = int(ceiling * weight)
        result = min(result, ceiling)
        result = max(result, 100)

        logger.debug(
            f"adaptive max_tokens: ceiling={ceiling} × {weight:.5f} "
            f"(emotion={emotion} intensity={intensity:.2f}) = {result}"
        )
        return result

    # ── 注意力判断 ─────────────────────────────────────────────────────────

    def should_respond(
        self, emotion_behavior: Dict, is_mentioned: bool = False
    ) -> bool:
        """注意力判断：是否需要回复（群聊场景使用）"""
        if is_mentioned:
            return True

        intensity = emotion_behavior["emotion"]["intensity"]
        behavior = emotion_behavior["behavior"]["type"]

        if intensity >= self.attention_config.intensity_threshold:
            return True
        if behavior in ("ask_question", "seek_clarification"):
            return True

        return False

    # ── 生成回复 ──────────────────────────────────────────────────────────

    def generate_response(
        self,
        user_input: str,
        recalled_context: str = "",
        history: Optional[List[Dict]] = None,
        emotion_analysis: Optional[Dict] = None,
        client: Optional["LLMClient"] = None,
        images: Optional[List] = None,
        in_attention_focus: bool = True,
    ) -> str:
        """使用大模型生成回复"""
        c = client or self.llm_client
        max_tokens = self._adaptive_max_tokens(emotion_analysis, c)

        # 非焦点回复：降低 max_tokens
        if not in_attention_focus:
            max_tokens = int(
                max_tokens * self.attention_config.non_focus_max_token_ratio
            )
            logger.debug(
                f"非焦点回复：max_tokens 降低至 {max_tokens} "
                f"(ratio={self.attention_config.non_focus_max_token_ratio})"
            )

        # 情感分析全量日志
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
        emotion_state_hint = None
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
            emotion_state_hint = self.emotion_state_tracker.get_prompt_hint()
            logger.debug(
                f"[情绪状态机] v={st.valence:.2f} a={st.arousal:.2f} "
                f"→ label={st.last_emotion} | "
                f"Δtemp={temperature - c.config.temperature:+.2f} "
                f"Δtop_p={top_p - c.config.top_p:+.2f} | "
                f"注入暗示: {emotion_state_hint or '(无)'}"
            )

        # Anthropic 使用多块 system prompt 优化缓存
        if c.provider == LLMProvider.ANTHROPIC:
            system_blocks = self.prompt_builder.build_system_prompt_blocks(
                recalled_context, emotion_analysis, emotion_state_hint
            )
            with self.infra.llm_gate():
                return c.generate(
                    system_blocks=system_blocks,
                    user_input=user_input,
                    history=history,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    images=images,
                )
        else:
            system_prompt = self.prompt_builder.build_system_prompt(
                recalled_context, emotion_analysis, emotion_state_hint
            )
            with self.infra.llm_gate():
                return c.generate(
                    system_prompt=system_prompt,
                    user_input=user_input,
                    history=history,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    images=images,
                )

    # ── 主对话流程 ────────────────────────────────────────────────────────

    def chat(
        self,
        user_input: str,
        verbose: bool = False,
        is_mentioned: bool = True,
        chat_mode: ChatMode = ChatMode.PRIVATE,
        use_fusion: bool = None,
        images: Optional[List] = None,
        user_id: Optional[int] = None,
        user_name: Optional[str] = None,
        context_id: Optional[str] = None,
        visual_direct: bool = False,
    ) -> Dict:
        """完整对话流程（签名与原 NeuroLikePipeline.chat 完全兼容）"""

        # ── 注意力焦点判断 ────────────────────────────────────────────
        in_attention_focus = False
        if chat_mode == ChatMode.GROUP and not is_mentioned and user_id is not None:
            user = self.attention_tracker.get_user_attention(
                user_id=user_id,
                context_key=context_id,
            )
            in_attention_focus = (
                user is not None
                and self.attention_config.track_mentioned_users
                and user.is_mentioned_active(self.attention_config.mentioned_user_ttl)
            )
            logger.debug(
                f"[注意力路由] user_id={user_id} in_focus={in_attention_focus}"
            )

        # ── 融合开关 ──────────────────────────────────────────────────
        if use_fusion is None:
            use_fusion = (
                self.emotion_fusion_config is not None
                and self.emotion_fusion_config.enabled
                and self.emotion_fusion_config.use_by_default
            )

        # ── BERT + 记忆并行 ──────────────────────────────────────────
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
            history = self.memory.get_messages_history(
                context_id=context_id, max_turns=40
            )
            logger.debug(
                f"记忆 {self.memory.l1_usage} L1轮次={len(self.memory.working_memory)} "
                f"context={context_id} messages={len(history) if history else 0}"
            )
        else:
            emotion_behavior = self._analyze_emotion_with_fusion(
                user_input, use_fusion=use_fusion
            )
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
            intensity = 0.5
            behavior_type = "respond_positive"
            emotion_primary = "neutral"
            tone = "calm"
            emotion_behavior = {
                "emotion": {"primary": emotion_primary, "intensity": intensity},
                "behavior": {"type": behavior_type, "tone": tone},
            }
            logger.debug("BERT 不可用，使用默认情绪值")

        # ── 注意力判断 ────────────────────────────────────────────────
        if chat_mode == ChatMode.GROUP and user_id is not None:
            respond = self.attention_tracker.should_respond(
                user_id=user_id,
                emotion_intensity=intensity,
                behavior_type=behavior_type,
                is_mentioned=is_mentioned,
                in_attention_focus=in_attention_focus,
                context_key=context_id,
            )
        else:
            respond = self.should_respond(emotion_behavior, is_mentioned)

        if chat_mode == ChatMode.GROUP and user_id is not None:
            logger.debug(
                f"[注意力追踪] 记录消息: user_id={user_id} "
                f"user_name={user_name} is_mentioned={is_mentioned}"
            )
            self.attention_tracker.on_message(
                user_id=user_id,
                user_name=user_name or str(user_id),
                is_mentioned=is_mentioned,
                context_key=context_id,
            )

        if not respond:
            logger.debug("注意力判断：跳过回复，仅记录记忆")

        # ── LLM 路由 + 生成 ──────────────────────────────────────────

        # ── 视觉技能检测（按需获取画面分析注入上下文）────
        if (
            not visual_direct
            and self._visual_skill_detector is not None
            and self._visual_skill_executor is not None
            and self._visual_skill_detector.detect(user_input)
        ):
            try:
                visual_context = self._visual_skill_executor.execute(top_k=2)
                if visual_context:
                    user_input = f"[视觉感知] {visual_context}\n{user_input}"
                    logger.info(f"视觉技能触发: {visual_context[:80]}...")
            except Exception as exc:
                logger.warning(f"视觉技能执行失败: {exc}")

        # ── 图片信息提取（vision → 描述 → 注入 user_input）────
        if images and self.infra.llm_pool.vision is not None and not visual_direct:
            try:
                image_context = self._extract_image_context(images, user_input)
                if image_context:
                    count = len(images)
                    tag = "图片" if count == 1 else f"{count}张图片"
                    user_input = f"[对方发送了{tag}，内容：{image_context}]\n{user_input}"
                    logger.info(f"图片描述已提取: {image_context[:80]}...")
            except Exception as e:
                logger.warning(f"图片描述提取失败，回退到 vision 直出: {e}")
            else:
                images = None  # 提取成功，图片已消费，后续路由跳过 vision

        response = None
        routed_client = self._route_llm_client(
            chat_mode, emotion_behavior, is_mentioned,
            images=images,
            in_attention_focus=in_attention_focus,
        )

        # 根据路由调整 history 长度
        if history and routed_client:
            if routed_client == self.llm_client_secondary:
                max_turns_for_api = 15
                if len(history) > max_turns_for_api * 2:
                    logger.debug(
                        f"DeepSeek API 限制：截断 history 为最近 {max_turns_for_api} 轮 "
                        f"({len(history)} → {max_turns_for_api * 2} 条)"
                    )
                    history = history[-(max_turns_for_api * 2):]
            else:
                max_turns_for_api = 30
                if len(history) > max_turns_for_api * 2:
                    history = history[-(max_turns_for_api * 2):]

        if respond:
            response = self.generate_response(
                user_input, recalled, history,
                emotion_analysis=emotion_behavior,
                client=routed_client,
                images=images,
                in_attention_focus=in_attention_focus or is_mentioned,
            )

        # ── 情绪状态机更新 ───────────────────────────────────────────
        if self.emotion_state_tracker and response:
            ai_emotion = emotion_primary
            ai_intensity = intensity
            if self.infra.bert.available:
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

            tc = self.emotion_state_tracker.state.turn_count
            interval = self.emotion_state_tracker.config.save_interval_turns
            if interval > 0 and tc % interval == 0:
                self._bg_executor.submit(self._save_emotion_state)

        # ── 记忆写入 ────────────────────────────────────────────────
        if respond and response:
            turn = ConversationTurn(
                user_input=user_input,
                emotion=emotion_primary,
                intensity=intensity,
                behavior=behavior_type,
                tone=tone,
                response=response,
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                context_id=context_id,
            )
            self.memory.add(turn)

            if chat_mode == ChatMode.GROUP and user_id is not None:
                self.attention_tracker.on_reply(user_id, context_key=context_id)

        result = {
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

        if verbose:
            es = self.emotion_state_tracker.state if self.emotion_state_tracker else None
            result["_debug"] = {
                "emotion_state": {
                    "valence": es.valence,
                    "arousal": es.arousal,
                    "label": es.last_emotion,  # V-A 映射到的离散标签
                } if es else None,
                "recalled_context": recalled[:500] if recalled else "",
                "emotion_analysis": emotion_behavior,
            }

        return result

    # ── 群聊被动消息 ─────────────────────────────────────────────────────

    def _handle_group_passive_message(
        self, user_input: str, context_id: Optional[str] = None
    ) -> Dict:
        """群聊被动消息处理（未被 @）：仅写入 L1 记忆"""
        logger.debug(f"群聊被动消息（未 @）：仅记录 L1 → {user_input[:50]}")

        turn = ConversationTurn(
            user_input=user_input,
            emotion="neutral",
            intensity=0.0,
            behavior="neutral_acknowledge",
            tone="calm",
            response="",
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            context_id=context_id,
        )
        self.memory.add(turn)

        return {
            "response": None,
            "emotion": {"primary": "neutral", "intensity": 0.0},
            "behavior": {"type": "neutral_acknowledge", "tone": "calm"},
            "should_respond": False,
            "debug_info": {
                "model_provider": "none",
                "model_name": "none",
                "chat_mode": "group_passive",
            },
        }

    # ── 主动发言 ──────────────────────────────────────────────────────────

    def generate_proactive(
        self,
        trigger: str,
        chat_mode: ChatMode = ChatMode.PRIVATE,
        decision_hint: Optional["ProactiveDecision"] = None,
    ) -> Optional[str]:
        """主动发言生成"""
        if isinstance(self.memory, HierarchicalMemoryManager):
            recalled = self.memory.get_system_context(query=trigger)
            history = self.memory.get_messages_history()
        else:
            recalled = self.memory.format_context(num_turns=5)
            history = None

        system_prompt = self.prompt_builder.build_system_prompt(recalled)
        system_prompt += (
            f"\n<proactive_trigger>{trigger}</proactive_trigger>"
            "\n你可以主动找话题聊，或者接上之前的对话继续说。"
            "如果实在没什么好说的，回复空字符串即可。"
        )

        if decision_hint:
            hint_text = (
                f"\n[主动发言指导]\n"
                f"意图: {decision_hint.intent}\n"
                f"话题提示: {decision_hint.topic_hint}\n"
                f"建议语气: {decision_hint.tone}"
            )
            system_prompt += hint_text
        max_tokens = self._adaptive_max_tokens(emotion_analysis=self.analyze_emotion_behavior(trigger), client=self.llm_client)
        try:
            with self.infra.llm_gate():
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

    # ── 生命周期 ──────────────────────────────────────────────────────────

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
        if (
            self.emotion_state_tracker
            and self.emotion_state_tracker.config.persist_to_l4
        ):
            self._save_emotion_state()
        
        if isinstance(self.memory, HierarchicalMemoryManager):
            logger.info("正在写入长期记忆...")
            self.memory.close_session()
            logger.info("长期记忆已保存。")
        
            mem0 = getattr(self.memory, "mem0", None)
            if mem0:
                for attr in ("vector_store", "_telemetry_vector_store"):
                    vs = getattr(mem0, attr, None)
                    if vs and hasattr(vs, "client"):
                        try:
                          if not self.muti_persona_mode:
                            vs.client.close()
                          else:
                            pass
                        except Exception:
                            pass
        if not self.muti_persona_mode:
            self._bg_executor.shutdown(wait=False)

    # ── 情绪状态持久化 ────────────────────────────────────────────────────

    def _load_emotion_state(self) -> Optional[EmotionState]:
        """从本地 JSON 文件恢复情绪状态"""
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
        """将情绪状态写入本地 JSON 文件"""
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

    def _emotion_state_path(self) -> Path:
        if isinstance(self.memory, HierarchicalMemoryManager):
            base = Path(self.memory.config.vector_store_path).parent
        else:
            base = project_root
        return base / "emotion_state.json"
