"""
测试CLI - 用于测试对话系统的各个组件

支持两种模式:
1. 模拟模式 (--mock): 无需训练模型，使用随机/规则生成情绪行为
2. 真实模式: 使用训练好的模型进行推理

用法:
    # 模拟模式 (测试LLM API连接)
    python test_cli.py --mock --provider openai --api_key YOUR_KEY

    # 真实模式 (需要训练好的模型)
    python test_cli.py --checkpoint ./checkpoints/joint_model/best.pt --provider openai --api_key YOUR_KEY

    # 纯本地测试 (不调用API)
    python test_cli.py --mock --offline
"""

import argparse
import random
import time
from typing import Dict, Optional
from dataclasses import dataclass

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from configs.model_config import (
    PersonalityConfig,
    DEFAULT_PERSONALITY,
    LLMConfig,
    LLMProvider,
    EMOTION_TO_ID,
    ID_TO_EMOTION,
    BEHAVIOR_TO_ID,
    ID_TO_BEHAVIOR,
    TONE_TO_ID,
    ID_TO_TONE
)


# ============== 模拟器 ==============

class MockEmotionBehaviorAnalyzer:
    """
    模拟情绪行为分析器
    用于在没有训练模型时测试系统
    """

    def __init__(self):
        # 简单的关键词规则
        self.emotion_keywords = {
            "joy": ["哈哈", "开心", "太棒了", "好耶", "nice", "赞", "喜欢", "爱"],
            "sadness": ["难过", "伤心", "哭", "呜呜", "唉", "可惜", "遗憾"],
            "anger": ["生气", "愤怒", "气死", "烦", "讨厌", "滚", "傻"],
            "surprise": ["什么", "啊", "卧槽", "天哪", "居然", "竟然", "不会吧"],
            "excitement": ["冲", "燃", "激动", "期待", "太强了", "牛", "厉害"],
            "curiosity": ["为什么", "怎么", "什么是", "好奇", "想知道", "?", "？"],
            "tenderness": ["辛苦", "加油", "抱抱", "心疼", "温柔", "乖"],
            "fear": ["害怕", "恐怖", "吓", "怕", "慌"],
            "disgust": ["恶心", "讨厌", "呕", "无语"],
        }

        self.behavior_map = {
            "joy": ["respond_positive", "make_joke", "share_experience"],
            "sadness": ["express_empathy", "give_advice", "neutral_acknowledge"],
            "anger": ["express_empathy", "change_topic", "seek_clarification"],
            "surprise": ["ask_question", "seek_clarification", "respond_positive"],
            "excitement": ["respond_positive", "make_joke", "agree"],
            "curiosity": ["ask_question", "share_experience", "give_advice"],
            "tenderness": ["express_empathy", "respond_positive", "give_advice"],
            "fear": ["express_empathy", "give_advice", "change_topic"],
            "disgust": ["change_topic", "disagree", "neutral_acknowledge"],
            "neutral": ["neutral_acknowledge", "ask_question", "respond_positive"],
        }

        self.tone_map = {
            "joy": ["enthusiastic", "playful", "warm"],
            "sadness": ["warm", "supportive", "calm"],
            "anger": ["calm", "supportive", "serious"],
            "surprise": ["enthusiastic", "playful", "calm"],
            "excitement": ["enthusiastic", "playful", "warm"],
            "curiosity": ["enthusiastic", "playful", "calm"],
            "tenderness": ["warm", "supportive", "calm"],
            "fear": ["warm", "supportive", "calm"],
            "disgust": ["calm", "serious", "cold"],
            "neutral": ["calm", "warm", "playful"],
        }

    def analyze(self, text: str) -> Dict:
        """分析文本的情绪和行为"""
        text_lower = text.lower()

        # 检测情绪
        detected_emotion = "neutral"
        max_matches = 0

        for emotion, keywords in self.emotion_keywords.items():
            matches = sum(1 for kw in keywords if kw in text_lower)
            if matches > max_matches:
                max_matches = matches
                detected_emotion = emotion

        # 生成强度 (基于匹配数和文本长度)
        intensity = min(0.5 + max_matches * 0.15, 1.0)
        if "!" in text or "！" in text:
            intensity = min(intensity + 0.1, 1.0)

        # 选择行为和语气
        behavior = random.choice(self.behavior_map.get(detected_emotion, ["neutral_acknowledge"]))
        tone = random.choice(self.tone_map.get(detected_emotion, ["calm"]))

        # 回复长度
        if len(text) < 10:
            response_length = "short"
        elif len(text) < 30:
            response_length = "medium"
        else:
            response_length = "long"

        return {
            "emotion": {
                "primary": detected_emotion,
                "intensity": round(intensity, 2),
                "primary_prob": round(0.6 + random.random() * 0.3, 2)
            },
            "behavior": {
                "type": behavior,
                "tone": tone,
                "response_length": response_length
            }
        }


class MockLLMClient:
    """
    模拟LLM客户端
    用于离线测试，不调用真实API
    """

    def __init__(self, personality: PersonalityConfig):
        self.personality = personality
        self.responses = {
            "joy": [
                "哈哈，是呀是呀~",
                "开心！这种感觉真好呢~",
                "嘿嘿，我也觉得超棒的！"
            ],
            "sadness": [
                "唔...听起来有点难过呢，没关系的~",
                "抱抱，会好起来的！",
                "我理解你的感受..."
            ],
            "anger": [
                "冷静冷静，深呼吸~",
                "唔，确实挺让人生气的...",
                "要不我们聊点别的？"
            ],
            "surprise": [
                "诶？！真的假的！",
                "哇，这也太神奇了吧！",
                "什么什么，快给我讲讲！"
            ],
            "excitement": [
                "冲冲冲！！！",
                "太燃了吧！我也好激动！",
                "哇啊啊啊超期待的！"
            ],
            "curiosity": [
                "嗯嗯，这个问题很有意思呢~",
                "让我想想...其实是这样的~",
                "好问题！我来解释一下~"
            ],
            "tenderness": [
                "谢谢你~你也辛苦了呢",
                "嘿嘿，你真温柔~",
                "感觉心里暖暖的~"
            ],
            "neutral": [
                "嗯嗯，我知道了~",
                "好的好的~",
                "原来如此呢~"
            ]
        }

    def generate(self, system_prompt: str, user_input: str, **kwargs) -> str:
        """生成模拟回复"""
        # 从system_prompt中提取情绪
        emotion = "neutral"
        for e in self.responses.keys():
            if e in system_prompt.lower():
                emotion = e
                break

        responses = self.responses.get(emotion, self.responses["neutral"])
        return random.choice(responses)


# ============== 记忆管理 ==============

@dataclass
class ConversationTurn:
    """对话轮次"""
    user_input: str
    emotion: str
    intensity: float
    behavior: str
    tone: str
    response: str


class MemoryManager:
    """记忆管理器"""

    def __init__(self, max_turns: int = 10):
        self.history = []
        self.max_turns = max_turns

    def add(self, turn: ConversationTurn):
        self.history.append(turn)
        if len(self.history) > self.max_turns:
            self.history.pop(0)

    def format_context(self, num_turns: int = 5) -> str:
        recent = self.history[-num_turns:]
        if not recent:
            return ""
        lines = []
        for turn in recent:
            lines.append(f"用户: {turn.user_input}")
            lines.append(f"助手: {turn.response}")
        return "\n".join(lines)

    def clear(self):
        self.history = []


# ============== 测试Pipeline ==============

class TestPipeline:
    """测试Pipeline"""

    def __init__(
        self,
        personality: PersonalityConfig,
        mock_mode: bool = True,
        offline_mode: bool = False,
        checkpoint_path: Optional[str] = None,
        llm_config: Optional[LLMConfig] = None,
        device: str = "cpu"
    ):
        self.personality = personality
        self.mock_mode = mock_mode
        self.offline_mode = offline_mode
        self.device = device
        self.memory = MemoryManager()

        # 初始化分析器
        if mock_mode:
            print("[模拟模式] 使用规则分析器")
            self.analyzer = MockEmotionBehaviorAnalyzer()
        else:
            print("[真实模式] 加载训练模型...")
            self._load_real_model(checkpoint_path)

        # 初始化LLM客户端
        if offline_mode:
            print("[离线模式] 使用模拟LLM")
            self.llm_client = MockLLMClient(personality)
            self.llm_info = "MockLLM (离线)"
        else:
            print(f"[在线模式] 连接LLM API...")
            self._init_llm_client(llm_config)

    def _load_real_model(self, checkpoint_path: str):
        """加载真实模型"""
        import torch
        from models.joint_model import create_joint_model

        self.model, self.tokenizer = create_joint_model()
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        self.personality_vector = torch.tensor(
            self.personality.to_embedding_vector(),
            dtype=torch.float32
        )
        print(f"模型加载完成: {checkpoint_path}")

    def _init_llm_client(self, llm_config: LLMConfig):
        """初始化LLM客户端"""
        from inference_pipeline import LLMClient
        self.llm_client = LLMClient(llm_config)
        self.llm_info = f"{llm_config.provider.value}: {llm_config.model}"
        print(f"LLM客户端初始化完成: {self.llm_info}")

    def analyze(self, text: str) -> Dict:
        """分析情绪和行为"""
        if self.mock_mode:
            return self.analyzer.analyze(text)
        else:
            import torch
            return self.model.predict(
                text=text,
                personality=self.personality_vector,
                tokenizer=self.tokenizer,
                device=self.device
            )

    def build_prompt(self, emotion_behavior: Dict, context: str) -> str:
        """构建系统提示"""
        emotion = emotion_behavior["emotion"]
        behavior = emotion_behavior["behavior"]

        return f"""你是{self.personality.name}，一个AI虚拟主播。

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
- 保持风格一致，像真人聊天
- 根据情绪强度调整表达
- 回复要简洁自然
"""

    def chat(self, user_input: str, verbose: bool = True) -> Dict:
        """对话"""
        # 分析
        start_time = time.time()
        emotion_behavior = self.analyze(user_input)
        analyze_time = time.time() - start_time

        if verbose:
            print(f"\n[分析] ({analyze_time*1000:.1f}ms)")
            print(f"  情绪: {emotion_behavior['emotion']['primary']} "
                  f"(强度: {emotion_behavior['emotion']['intensity']:.2f})")
            print(f"  行为: {emotion_behavior['behavior']['type']}")
            print(f"  语气: {emotion_behavior['behavior']['tone']}")

        # 生成回复
        context = self.memory.format_context()
        system_prompt = self.build_prompt(emotion_behavior, context)

        length_map = {"short": 50, "medium": 150, "long": 300}
        max_tokens = length_map.get(emotion_behavior["behavior"]["response_length"], 150)

        start_time = time.time()
        if self.offline_mode:
            response = self.llm_client.generate(system_prompt, user_input)
        else:
            response = self.llm_client.generate(
                system_prompt=system_prompt,
                user_input=user_input,
                max_tokens=max_tokens,
                temperature=0.7 + emotion_behavior["emotion"]["intensity"] * 0.3
            )
        generate_time = time.time() - start_time

        if verbose:
            print(f"[生成] ({generate_time*1000:.1f}ms)")

        # 保存记忆
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
            "timing": {
                "analyze_ms": analyze_time * 1000,
                "generate_ms": generate_time * 1000
            }
        }


# ============== CLI ==============

def print_banner(personality: PersonalityConfig, llm_info: str, mode: str):
    """打印欢迎信息"""
    print("\n" + "=" * 60)
    print(f"  Neuro-Like System 测试CLI")
    print("=" * 60)
    print(f"  人格: {personality.name}")
    print(f"  特质: {', '.join(personality.traits)}")
    print(f"  LLM: {llm_info}")
    print(f"  模式: {mode}")
    print("=" * 60)
    print("  命令:")
    print("    quit/exit  - 退出")
    print("    clear      - 清空对话历史")
    print("    debug      - 切换详细输出")
    print("    status     - 显示当前状态")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Neuro-Like System 测试CLI")

    # 模式选择
    parser.add_argument("--mock", action="store_true",
                       help="模拟模式 (无需训练模型)")
    parser.add_argument("--offline", action="store_true",
                       help="离线模式 (不调用LLM API)")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="模型检查点路径 (真实模式)")

    # LLM配置
    parser.add_argument("--provider", type=str, default="openai",
                       choices=["openai", "deepseek", "anthropic", "custom"],
                       help="LLM提供商")
    parser.add_argument("--api_key", type=str, default=None,
                       help="API密钥")
    parser.add_argument("--model", type=str, default=None,
                       help="模型名称")
    parser.add_argument("--base_url", type=str, default=None,
                       help="API Base URL")

    # 其他
    parser.add_argument("--device", type=str, default="cpu",
                       help="设备 (cpu/cuda)")
    parser.add_argument("--verbose", action="store_true", default=True,
                       help="详细输出")

    args = parser.parse_args()

    # 验证参数
    if not args.mock and not args.checkpoint:
        print("错误: 真实模式需要指定 --checkpoint")
        print("提示: 使用 --mock 进入模拟模式")
        return

    if not args.offline and not args.api_key:
        print("错误: 在线模式需要指定 --api_key")
        print("提示: 使用 --offline 进入离线模式")
        return

    # 创建LLM配置
    llm_config = None
    if not args.offline:
        default_models = {
            "openai": "gpt-4o-mini",
            "deepseek": "deepseek-chat",
            "anthropic": "claude-3-5-haiku-20241022",
            "custom": "gpt-4o-mini"
        }
        model = args.model or default_models.get(args.provider, "gpt-4o-mini")

        llm_config = LLMConfig(
            provider=LLMProvider(args.provider),
            api_key=args.api_key,
            model=model,
            base_url=args.base_url
        )

    # 创建Pipeline
    pipeline = TestPipeline(
        personality=DEFAULT_PERSONALITY,
        mock_mode=args.mock,
        offline_mode=args.offline,
        checkpoint_path=args.checkpoint,
        llm_config=llm_config,
        device=args.device
    )

    # 确定模式描述
    if args.mock and args.offline:
        mode = "模拟+离线 (纯本地测试)"
    elif args.mock:
        mode = "模拟+在线 (测试LLM API)"
    elif args.offline:
        mode = "真实+离线 (测试小模型)"
    else:
        mode = "真实+在线 (完整流程)"

    llm_info = pipeline.llm_info if hasattr(pipeline, 'llm_info') else "Unknown"

    # 打印欢迎信息
    print_banner(DEFAULT_PERSONALITY, llm_info, mode)

    verbose = args.verbose

    # 主循环
    while True:
        try:
            user_input = input(f"你: ").strip()

            if not user_input:
                continue

            # 命令处理
            cmd = user_input.lower()
            if cmd in ["quit", "exit", "q"]:
                print("再见！")
                break
            elif cmd == "clear":
                pipeline.memory.clear()
                print("[对话历史已清空]\n")
                continue
            elif cmd == "debug":
                verbose = not verbose
                print(f"[详细输出: {'开启' if verbose else '关闭'}]\n")
                continue
            elif cmd == "status":
                print(f"\n[状态]")
                print(f"  对话轮数: {len(pipeline.memory.history)}")
                print(f"  模式: {mode}")
                print(f"  LLM: {llm_info}")
                print()
                continue

            # 对话
            result = pipeline.chat(user_input, verbose=verbose)
            print(f"\n{DEFAULT_PERSONALITY.name}: {result['response']}\n")

        except KeyboardInterrupt:
            print("\n\n再见！")
            break
        except Exception as e:
            print(f"\n[错误] {e}\n")


if __name__ == "__main__":
    main()
