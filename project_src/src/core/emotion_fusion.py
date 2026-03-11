"""
双信号情绪融合：BERT + LLM 神经元融合

架构：
  BERT（~15ms，68% 准确率）+ LLM（~1-2s，高准确率）→ 单神经元前向传播融合

使用场景：
  - 日常对话：BERT-only（保留"人味"）
  - 主动发言、L4 记忆写入、群聊注意力判断：启用融合（需要高准确率）
"""

import copy
import math
import json
from typing import Dict, Optional
from concurrent.futures import ThreadPoolExecutor, TimeoutError

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.logger import logger
from configs.model_config import EmotionFusionConfig


class LLMEmotionClassifier:
    """轻量级 LLM 情绪分类器（实时单条）"""

    def __init__(self, llm_client, temperature: float = 0.3):
        """
        Args:
            llm_client: LLMClient 实例（通常是副 LLM，如 DeepSeek）
            temperature: 分类温度（低温保证一致性）
        """
        self.llm_client = llm_client
        self.temperature = temperature
        self.prompt_template = """分析这段文本的情绪，从以下10种中选择一个：
joy（喜悦）, sadness（悲伤）, anger（愤怒）, fear（恐惧）, surprise（惊讶）,
disgust（厌恶）, neutral（中性）, excitement（兴奋）, tenderness（温柔）, curiosity（好奇）

文本：{text}

返回 JSON：
{{
  "emotion": "joy",
  "confidence": 0.85
}}"""

    def classify(self, text: str, timeout: float = 5.0) -> Optional[Dict]:
        """
        分类单条文本。

        Args:
            text: 输入文本
            timeout: 超时时间（秒）

        Returns:
            {"emotion": "joy", "confidence": 0.85} or None（失败时）
        """
        prompt = self.prompt_template.format(text=text)

        try:
            # 使用 OpenAI-compatible API 的 JSON mode
            from openai import OpenAI

            # 检查 llm_client 是否是 OpenAI-compatible
            if hasattr(self.llm_client, 'client') and isinstance(self.llm_client.client, OpenAI):
                response = self.llm_client.client.chat.completions.create(
                    model=self.llm_client.model,
                    messages=[
                        {"role": "system", "content": "你是情绪分类专家，只返回 JSON 格式结果。"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=50,
                    response_format={"type": "json_object"},
                    timeout=timeout,
                )
                result_text = response.choices[0].message.content
            else:
                # Anthropic 或其他 provider，用普通生成
                result_text = self.llm_client.generate(
                    system_prompt="你是情绪分类专家，只返回 JSON 格式结果。",
                    user_input=prompt,
                    max_tokens=50,
                    temperature=self.temperature,
                )

            # 解析 JSON
            result = json.loads(result_text)
            emotion = result.get("emotion", "").lower()
            confidence = float(result.get("confidence", 0.0))

            # 验证情绪标签
            valid_emotions = {
                "joy", "sadness", "anger", "fear", "surprise",
                "disgust", "neutral", "excitement", "tenderness", "curiosity"
            }
            if emotion not in valid_emotions:
                logger.warning(f"LLM 返回无效情绪标签: {emotion}")
                return None

            return {"emotion": emotion, "confidence": confidence}

        except TimeoutError:
            logger.warning(f"LLM 情绪分类超时 ({timeout}s)")
            return None
        except json.JSONDecodeError as e:
            logger.warning(f"LLM 返回非 JSON 格式: {e}")
            return None
        except Exception as e:
            logger.warning(f"LLM 情绪分类失败: {e}")
            return None


class EmotionNeuronFusion:
    """单神经元前向传播融合 BERT 和 LLM 情绪信号"""

    def __init__(self, config: EmotionFusionConfig,
                 emotion_reliability: Dict[str, float]):
        """
        Args:
            config: 融合配置（权重、偏置等）
            emotion_reliability: BERT 各情绪标签的可靠度（F1 分数）
        """
        self.config = config
        self.emotion_reliability = emotion_reliability
        self.emotion_labels = list(emotion_reliability.keys())  # 10 emotions

    def fuse(self, bert_result: Dict, llm_result: Optional[Dict]) -> Dict:
        """
        神经元融合。

        For each emotion label i:
            x_bert = bert_prob[i] × reliability[i]
            x_llm = llm_confidence if llm_label == i else 0
            z[i] = w_bert × x_bert + w_llm × x_llm + bias

        Final: scores = softmax(z), label = argmax(scores)

        改进：使用 softmax 替代 sigmoid，避免信号压缩到 [0.5, 1.0] 区间。

        Args:
            bert_result: BERT predict() 输出
            llm_result: LLM classify() 输出，可为 None

        Returns:
            修改后的 emotion_behavior dict（保持原结构）
        """
        if llm_result is None:
            # LLM 失败，返回 BERT 原结果
            return bert_result

        # 提取信号
        bert_probs = bert_result["emotion"]["all_probs"]
        llm_label = llm_result["emotion"]
        llm_conf = llm_result["confidence"]

        # 神经元计算（线性加权）
        z_scores = {}
        for label in self.emotion_labels:
            x_bert = bert_probs[label] * self.emotion_reliability.get(label, 0.7)
            x_llm = llm_conf if label == llm_label else 0.0
            z_scores[label] = self.config.w_bert * x_bert + self.config.w_llm * x_llm + self.config.bias

        # Softmax 归一化
        scores = self._softmax(z_scores)

        # 最终预测
        fused_label = max(scores, key=scores.get)
        fused_conf = scores[fused_label]

        # 更新 bert_result 结构（深拷贝，避免污染原始 BERT 结果）
        result = copy.deepcopy(bert_result)
        result["emotion"]["primary"] = fused_label
        result["emotion"]["primary_prob"] = fused_conf
        result["emotion"]["all_probs"] = scores
        result["emotion"]["_fusion_meta"] = {
            "bert_label": bert_result["emotion"]["primary"],
            "llm_label": llm_label,
            "agreement": (bert_result["emotion"]["primary"] == llm_label)
        }

        return result

    @staticmethod
    def _softmax(z_scores: Dict[str, float]) -> Dict[str, float]:
        """
        Softmax 归一化。

        Args:
            z_scores: 原始分数字典 {label: z}

        Returns:
            归一化后的概率分布 {label: prob}
        """
        # 数值稳定性：减去最大值
        max_z = max(z_scores.values())
        exp_scores = {label: math.exp(z - max_z) for label, z in z_scores.items()}
        sum_exp = sum(exp_scores.values())
        return {label: exp_z / sum_exp for label, exp_z in exp_scores.items()}

    @staticmethod
    def _sigmoid(x: float) -> float:
        """Sigmoid 激活函数（已弃用，保留用于向后兼容）"""
        try:
            return 1.0 / (1.0 + math.exp(-x))
        except OverflowError:
            # x 太小时 exp(-x) 溢出，sigmoid(x) ≈ 0
            return 0.0 if x < 0 else 1.0
