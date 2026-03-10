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
from configs.model_config import (
    PersonalityConfig,
    EmotionPromptConfig,
    EmotionFusionConfig,
    EmotionStateConfig,
    AttentionConfig,
    MemoryConfig,
    LLMProvider,
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
        time_awareness: bool = True,
    ):
        self.infra = infra
        self.personality = personality
        self.emotion_prompt_config = emotion_prompt_config
        self.emotion_fusion_config = emotion_fusion_config or infra.emotion_fusion_config
        self.attention_config = attention_config or AttentionConfig()

        # 人格向量（BERT 推理需要）
        self.personality_vector = torch.tensor(
            personality.to_embedding_vector(),
            dtype=torch.float32,
        )

        # Prompt 构建器
        self.prompt_builder = PromptBuilder(
            personality=personality,
            emotion_prompt_config=emotion_prompt_config,
            time_awareness=time_awareness,
        )

        # 记忆管理器（每 persona 独立）
        if memory_config is not None:
            self.memory = HierarchicalMemoryManager(
                config=memory_config,
                llm_client=infra.llm_pool.primary,
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

        # 后台线程池
        self._bg_executor = ThreadPoolExecutor(max_workers=1)

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
    ) -> Dict:
        """完整对话流程（签名与原 NeuroLikePipeline.chat 完全兼容）"""

        # ── 群聊注意力追踪 ─────────────────────────────────────────────
        if chat_mode == ChatMode.GROUP and user_id is not None:
            logger.debug(
                f"[注意力追踪] 记录消息: user_id={user_id} "
                f"user_name={user_name} is_mentioned={is_mentioned}"
            )
            self.attention_tracker.on_message(
                user_id=user_id,
                user_name=user_name or str(user_id),
                is_mentioned=is_mentioned,
            )

        # ── 注意力焦点判断 ────────────────────────────────────────────
        in_attention_focus = False
        if chat_mode == ChatMode.GROUP and not is_mentioned and user_id is not None:
            user = self.attention_tracker.users.get(user_id)
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
            )
        else:
            respond = self.should_respond(emotion_behavior, is_mentioned)

        if not respond:
            logger.debug("注意力判断：跳过回复，仅记录记忆")

        # ── LLM 路由 + 生成 ──────────────────────────────────────────
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
                self.attention_tracker.on_reply(user_id)

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

        max_tokens = self.llm_client.config.max_tokens
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
                            vs.client.close()
                        except Exception:
                            pass

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
