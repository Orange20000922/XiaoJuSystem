"""
完整推理Pipeline
整合小模型 + 大模型API，实现端到端对话生成
"""

import torch
import json
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

import sys
sys.path.append("..")
from models.joint_model import JointEmotionBehaviorModel, create_joint_model
from configs.model_config import PersonalityConfig, DEFAULT_PERSONALITY


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

    def __init__(
        self,
        provider: str = "openai",  # openai, deepseek, claude
        api_key: str = None,
        model: str = None,
        base_url: str = None
    ):
        self.provider = provider
        self.api_key = api_key
        self.model = model or self._get_default_model()
        self.base_url = base_url

        # 初始化客户端
        if provider in ["openai", "deepseek"]:
            from openai import OpenAI
            self.client = OpenAI(
                api_key=api_key,
                base_url=base_url or self._get_default_base_url()
            )
        elif provider == "claude":
            import anthropic
            self.client = anthropic.Anthropic(api_key=api_key)
        else:
            raise ValueError(f"不支持的provider: {provider}")

    def _get_default_model(self) -> str:
        """获取默认模型"""
        defaults = {
            "openai": "gpt-4o-mini",
            "deepseek": "deepseek-chat",
            "claude": "claude-3-haiku-20240307"
        }
        return defaults.get(self.provider, "gpt-4o-mini")

    def _get_default_base_url(self) -> str:
        """获取默认base URL"""
        if self.provider == "deepseek":
            return "https://api.deepseek.com/v1"
        return None

    def generate(
        self,
        system_prompt: str,
        user_input: str,
        max_tokens: int = 200,
        temperature: float = 0.8
    ) -> str:
        """生成回复"""
        if self.provider in ["openai", "deepseek"]:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input}
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content

        elif self.provider == "claude":
            response = self.client.messages.create(
                model=self.model,
                system=system_prompt,
                messages=[{"role": "user", "content": user_input}],
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.content[0].text


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
        llm_provider: str = "openai",
        llm_api_key: str = None,
        llm_model: str = None,
        device: str = "cpu"
    ):
        self.personality = personality
        self.device = device

        # 加载小模型
        print("加载小模型...")
        self.small_model, self.tokenizer = create_joint_model()
        self._load_checkpoint(small_model_path)
        self.small_model.to(device)
        self.small_model.eval()

        # 初始化大模型客户端
        print(f"初始化大模型客户端 ({llm_provider})...")
        self.llm_client = LLMClient(
            provider=llm_provider,
            api_key=llm_api_key,
            model=llm_model
        )

        # 记忆管理器
        self.memory = MemoryManager()

        # 人格向量
        self.personality_vector = torch.tensor(
            personality.to_embedding_vector(),
            dtype=torch.float32
        )

    def _load_checkpoint(self, checkpoint_path: str):
        """加载模型检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.small_model.load_state_dict(checkpoint["model_state_dict"])
        print(f"✓ 加载检查点: {checkpoint_path}")
        if "metrics" in checkpoint:
            print(f"  验证损失: {checkpoint['metrics'].get('val_loss', 'N/A')}")
            print(f"  情绪准确率: {checkpoint['metrics'].get('emotion_acc', 'N/A'):.2%}")

    def analyze_emotion_behavior(self, text: str) -> Dict:
        """使用小模型分析情绪和行为"""
        result = self.small_model.predict(
            text=text,
            personality=self.personality_vector,
            tokenizer=self.tokenizer,
            device=self.device
        )
        return result

    def build_system_prompt(
        self,
        emotion_behavior: Dict,
        context: str = ""
    ) -> str:
        """构建系统Prompt"""
        emotion = emotion_behavior["emotion"]
        behavior = emotion_behavior["behavior"]

        prompt = f"""你是{self.personality.name}，一个AI虚拟主播。

# 人格特质
- 性格: {', '.join(self.personality.traits)}
- 外向性: {self.personality.extraversion:.1f}/1.0
- 幽默感: {self.personality.humor_tendency:.1f}/1.0
- 共情能力: {self.personality.empathy_level:.1f}/1.0

# 当前状态
- 情绪: {emotion['primary']} (强度: {emotion['intensity']:.1f})
- 行为倾向: {behavior['type']}
- 语气: {behavior['tone']}
- 回复长度: {behavior['response_length']}

# 对话历史
{context if context else '(首次对话)'}

# 指令
请根据以上人格设定和当前状态，用自然、符合人设的方式回复用户。
- 保持风格一致，避免突兀的语气转变
- 回复要像真人聊天，不要太正式
- 根据情绪强度调整表达方式
- 根据行为倾向选择回复策略
"""
        return prompt

    def generate_response(
        self,
        user_input: str,
        emotion_behavior: Dict,
        context: str = ""
    ) -> str:
        """使用大模型生成回复"""
        system_prompt = self.build_system_prompt(emotion_behavior, context)

        # 根据回复长度设置max_tokens
        length_map = {
            "short": 50,
            "medium": 150,
            "long": 300
        }
        max_tokens = length_map.get(
            emotion_behavior["behavior"]["response_length"],
            150
        )

        # 根据情绪强度调整temperature
        intensity = emotion_behavior["emotion"]["intensity"]
        temperature = 0.7 + (intensity * 0.3)  # 0.7-1.0

        response = self.llm_client.generate(
            system_prompt=system_prompt,
            user_input=user_input,
            max_tokens=max_tokens,
            temperature=temperature
        )

        return response

    def chat(self, user_input: str, verbose: bool = False) -> Dict:
        """
        完整对话流程

        Args:
            user_input: 用户输入
            verbose: 是否输出详细信息

        Returns:
            {
                "response": "回复文本",
                "emotion": {...},
                "behavior": {...},
                "debug_info": {...}
            }
        """
        # 1. 分析情绪和行为
        emotion_behavior = self.analyze_emotion_behavior(user_input)

        if verbose:
            print("\n[小模型分析]")
            print(f"情绪: {emotion_behavior['emotion']['primary']} "
                  f"(强度: {emotion_behavior['emotion']['intensity']:.2f})")
            print(f"行为: {emotion_behavior['behavior']['type']}")
            print(f"语气: {emotion_behavior['behavior']['tone']}")

        # 2. 获取对话上下文
        context = self.memory.format_context(num_turns=5)

        # 3. 生成回复
        response = self.generate_response(
            user_input=user_input,
            emotion_behavior=emotion_behavior,
            context=context
        )

        # 4. 保存到记忆
        turn = ConversationTurn(
            user_input=user_input,
            emotion=emotion_behavior["emotion"]["primary"],
            intensity=emotion_behavior["emotion"]["intensity"],
            behavior=emotion_behavior["behavior"]["type"],
            tone=emotion_behavior["behavior"]["tone"],
            response=response
        )
        self.memory.add(turn)

        return {
            "response": response,
            "emotion": emotion_behavior["emotion"],
            "behavior": emotion_behavior["behavior"],
            "debug_info": {
                "context_turns": len(self.memory.short_term),
                "model_provider": self.llm_client.provider,
                "model_name": self.llm_client.model
            }
        }

    def save_conversation(self, output_path: str):
        """保存对话历史"""
        data = [asdict(turn) for turn in self.memory.short_term]
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"对话历史已保存: {output_path}")


def interactive_chat(pipeline: NeuroLikePipeline):
    """交互式对话"""
    print("\n" + "=" * 50)
    print(f"欢迎与 {pipeline.personality.name} 对话！")
    print("输入 'quit' 退出，'save' 保存对话历史")
    print("=" * 50 + "\n")

    while True:
        try:
            user_input = input("你: ").strip()

            if not user_input:
                continue

            if user_input.lower() == "quit":
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


# ============== 测试代码 ==============
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Neuro-Like对话系统")
    parser.add_argument("--checkpoint", type=str, required=True, help="小模型检查点路径")
    parser.add_argument("--provider", type=str, default="openai",
                       choices=["openai", "deepseek", "claude"], help="大模型提供商")
    parser.add_argument("--api_key", type=str, required=True, help="API密钥")
    parser.add_argument("--model", type=str, default=None, help="大模型名称")
    parser.add_argument("--device", type=str, default="cpu", help="设备")

    args = parser.parse_args()

    # 创建Pipeline
    pipeline = NeuroLikePipeline(
        small_model_path=args.checkpoint,
        personality=DEFAULT_PERSONALITY,
        llm_provider=args.provider,
        llm_api_key=args.api_key,
        llm_model=args.model,
        device=args.device
    )

    # 开始交互
    interactive_chat(pipeline)
