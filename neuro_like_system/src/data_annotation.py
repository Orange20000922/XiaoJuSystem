"""
GPT/DeepSeek 数据标注脚本
用于批量标注训练数据
"""

import json
import time
import random
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm
from dataclasses import dataclass
import argparse


@dataclass
class AnnotationConfig:
    """标注配置"""
    provider: str = "openai"  # openai, deepseek
    model: str = "gpt-4o-mini"
    api_key: str = ""
    base_url: Optional[str] = None
    batch_size: int = 10  # 每批标注数量
    max_retries: int = 3
    retry_delay: float = 1.0
    temperature: float = 0.3


# 标注Prompt模板
ANNOTATION_PROMPT = """你是数据标注专家。请分析以下文本的情绪和行为倾向。

# 情绪类型 (选择一个)
- joy: 喜悦、开心
- sadness: 悲伤、难过
- anger: 愤怒、生气
- fear: 恐惧、害怕
- surprise: 惊讶
- disgust: 厌恶
- neutral: 中性
- excitement: 兴奋
- tenderness: 温柔
- curiosity: 好奇

# 行为类型 (选择一个)
- respond_positive: 积极回应
- respond_negative: 消极回应
- ask_question: 提问
- share_experience: 分享经历
- give_advice: 给建议
- express_empathy: 表达共情
- make_joke: 开玩笑
- change_topic: 转移话题
- seek_clarification: 寻求澄清
- agree: 同意
- disagree: 不同意
- neutral_acknowledge: 中性确认

# 语气类型 (选择一个)
- enthusiastic: 热情
- calm: 平静
- playful: 俏皮
- serious: 严肃
- warm: 温暖
- cold: 冷淡
- sarcastic: 讽刺
- supportive: 支持

# 回复长度
- short: 简短回复 (1-2句)
- medium: 中等回复 (3-5句)
- long: 详细回复 (5句以上)

# 注意事项
1. 网络用语映射：
   - yyds/绝绝子 → excitement, 高强度
   - 爷青回 → joy + tenderness
   - 破防了 → sadness 或 tenderness
   - 笑死/哈哈哈 → joy, 高强度
   - 呵呵/行吧 → 可能是讽刺

2. 强度标准 (0.0-1.0)：
   - 0.3-0.5: 轻微情绪
   - 0.6-0.8: 明显情绪
   - 0.9-1.0: 强烈情绪

# 待标注文本
{text}

# 输出格式 (严格JSON)
{{
    "emotion": "情绪类型",
    "intensity": 0.0-1.0,
    "behavior": "行为类型",
    "tone": "语气类型",
    "response_length": "short/medium/long",
    "confidence": 0.0-1.0,
    "reasoning": "简短解释"
}}"""


BATCH_ANNOTATION_PROMPT = """你是数据标注专家。请批量分析以下文本的情绪和行为倾向。

# 标签定义
情绪: joy, sadness, anger, fear, surprise, disgust, neutral, excitement, tenderness, curiosity
行为: respond_positive, respond_negative, ask_question, share_experience, give_advice, express_empathy, make_joke, change_topic, seek_clarification, agree, disagree, neutral_acknowledge
语气: enthusiastic, calm, playful, serious, warm, cold, sarcastic, supportive
长度: short, medium, long

# 待标注文本列表
{texts}

# 输出格式 (JSON数组，顺序与输入一致)
[
    {{"id": 0, "emotion": "...", "intensity": 0.8, "behavior": "...", "tone": "...", "response_length": "...", "confidence": 0.9}},
    ...
]"""


class DataAnnotator:
    """数据标注器"""

    def __init__(self, config: AnnotationConfig):
        self.config = config

        # 初始化API客户端
        from openai import OpenAI

        base_url = config.base_url
        if config.provider == "deepseek" and not base_url:
            base_url = "https://api.deepseek.com/v1"

        self.client = OpenAI(
            api_key=config.api_key,
            base_url=base_url
        )

    def annotate_single(self, text: str) -> Optional[Dict]:
        """标注单条文本"""
        prompt = ANNOTATION_PROMPT.format(text=text)

        for attempt in range(self.config.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.config.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.config.temperature,
                    response_format={"type": "json_object"}
                )

                result = json.loads(response.choices[0].message.content)
                result["text"] = text
                return result

            except Exception as e:
                print(f"标注失败 (尝试 {attempt + 1}/{self.config.max_retries}): {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay * (attempt + 1))

        return None

    def annotate_batch(self, texts: List[str]) -> List[Optional[Dict]]:
        """批量标注"""
        # 格式化文本列表
        formatted_texts = "\n".join([
            f"{i}. {text}" for i, text in enumerate(texts)
        ])

        prompt = BATCH_ANNOTATION_PROMPT.format(texts=formatted_texts)

        for attempt in range(self.config.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.config.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.config.temperature,
                    response_format={"type": "json_object"}
                )

                content = response.choices[0].message.content

                # 解析JSON
                try:
                    results = json.loads(content)
                    if isinstance(results, dict) and "annotations" in results:
                        results = results["annotations"]
                except json.JSONDecodeError:
                    # 尝试提取JSON数组
                    import re
                    match = re.search(r'\[.*\]', content, re.DOTALL)
                    if match:
                        results = json.loads(match.group())
                    else:
                        raise ValueError("无法解析JSON")

                # 添加原始文本
                for i, result in enumerate(results):
                    if i < len(texts):
                        result["text"] = texts[i]

                return results

            except Exception as e:
                print(f"批量标注失败 (尝试 {attempt + 1}/{self.config.max_retries}): {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay * (attempt + 1))

        # 批量失败，回退到单条标注
        print("批量标注失败，回退到单条标注...")
        return [self.annotate_single(text) for text in texts]

    def annotate_file(
        self,
        input_path: str,
        output_path: str,
        text_field: str = "text",
        use_batch: bool = True,
        resume: bool = True
    ):
        """
        标注整个文件

        Args:
            input_path: 输入文件路径 (JSON/JSONL)
            output_path: 输出文件路径
            text_field: 文本字段名
            use_batch: 是否使用批量标注
            resume: 是否从断点续传
        """
        # 加载输入数据
        input_path = Path(input_path)
        if input_path.suffix == ".json":
            with open(input_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        elif input_path.suffix == ".jsonl":
            data = []
            with open(input_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
        else:
            raise ValueError(f"不支持的文件格式: {input_path.suffix}")

        print(f"加载了 {len(data)} 条数据")

        # 检查断点
        output_path = Path(output_path)
        annotated = []
        start_idx = 0

        if resume and output_path.exists():
            with open(output_path, "r", encoding="utf-8") as f:
                annotated = json.load(f)
            start_idx = len(annotated)
            print(f"从断点续传: 已完成 {start_idx} 条")

        # 开始标注
        texts_to_annotate = [item[text_field] for item in data[start_idx:]]

        if use_batch:
            # 批量标注
            for i in tqdm(range(0, len(texts_to_annotate), self.config.batch_size),
                         desc="批量标注"):
                batch = texts_to_annotate[i:i + self.config.batch_size]
                results = self.annotate_batch(batch)

                for result in results:
                    if result:
                        annotated.append(result)

                # 定期保存
                if len(annotated) % 100 == 0:
                    self._save_results(annotated, output_path)

                # 避免API限流
                time.sleep(0.5)
        else:
            # 单条标注
            for text in tqdm(texts_to_annotate, desc="单条标注"):
                result = self.annotate_single(text)
                if result:
                    annotated.append(result)

                # 定期保存
                if len(annotated) % 50 == 0:
                    self._save_results(annotated, output_path)

                # 避免API限流
                time.sleep(0.2)

        # 最终保存
        self._save_results(annotated, output_path)
        print(f"\n标注完成！共 {len(annotated)} 条，保存到: {output_path}")

        # 统计
        self._print_statistics(annotated)

    def _save_results(self, results: List[Dict], output_path: Path):
        """保存结果"""
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    def _print_statistics(self, results: List[Dict]):
        """打印统计信息"""
        from collections import Counter

        emotions = Counter(r.get("emotion", "unknown") for r in results)
        behaviors = Counter(r.get("behavior", "unknown") for r in results)

        print("\n" + "=" * 40)
        print("标注统计")
        print("=" * 40)

        print("\n情绪分布:")
        for emotion, count in emotions.most_common():
            print(f"  {emotion}: {count} ({count/len(results)*100:.1f}%)")

        print("\n行为分布:")
        for behavior, count in behaviors.most_common(5):
            print(f"  {behavior}: {count} ({count/len(results)*100:.1f}%)")

        # 平均置信度
        confidences = [r.get("confidence", 0) for r in results if "confidence" in r]
        if confidences:
            print(f"\n平均置信度: {sum(confidences)/len(confidences):.2f}")


def compare_providers(
    texts: List[str],
    openai_key: str,
    deepseek_key: str,
    output_path: str = "comparison_results.json"
):
    """
    对比不同API的标注质量

    Args:
        texts: 测试文本列表
        openai_key: OpenAI API密钥
        deepseek_key: DeepSeek API密钥
        output_path: 输出路径
    """
    results = []

    # OpenAI GPT-4o-mini
    print("\n标注中: GPT-4o-mini")
    gpt_annotator = DataAnnotator(AnnotationConfig(
        provider="openai",
        model="gpt-4o-mini",
        api_key=openai_key
    ))

    # DeepSeek
    print("\n标注中: DeepSeek")
    deepseek_annotator = DataAnnotator(AnnotationConfig(
        provider="deepseek",
        model="deepseek-chat",
        api_key=deepseek_key
    ))

    for text in tqdm(texts, desc="对比标注"):
        gpt_result = gpt_annotator.annotate_single(text)
        time.sleep(0.5)
        deepseek_result = deepseek_annotator.annotate_single(text)
        time.sleep(0.5)

        results.append({
            "text": text,
            "gpt4o_mini": gpt_result,
            "deepseek": deepseek_result,
            "agreement": {
                "emotion": gpt_result.get("emotion") == deepseek_result.get("emotion") if gpt_result and deepseek_result else False,
                "behavior": gpt_result.get("behavior") == deepseek_result.get("behavior") if gpt_result and deepseek_result else False
            }
        })

    # 保存结果
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 计算一致性
    emotion_agree = sum(1 for r in results if r["agreement"]["emotion"])
    behavior_agree = sum(1 for r in results if r["agreement"]["behavior"])

    print("\n" + "=" * 40)
    print("对比结果")
    print("=" * 40)
    print(f"情绪一致性: {emotion_agree}/{len(results)} ({emotion_agree/len(results)*100:.1f}%)")
    print(f"行为一致性: {behavior_agree}/{len(results)} ({behavior_agree/len(results)*100:.1f}%)")
    print(f"结果保存到: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="数据标注工具")
    subparsers = parser.add_subparsers(dest="command", help="命令")

    # 标注命令
    annotate_parser = subparsers.add_parser("annotate", help="标注数据")
    annotate_parser.add_argument("--input", type=str, required=True, help="输入文件")
    annotate_parser.add_argument("--output", type=str, required=True, help="输出文件")
    annotate_parser.add_argument("--provider", type=str, default="openai",
                                choices=["openai", "deepseek"], help="API提供商")
    annotate_parser.add_argument("--model", type=str, default=None, help="模型名称")
    annotate_parser.add_argument("--api_key", type=str, required=True, help="API密钥")
    annotate_parser.add_argument("--batch_size", type=int, default=10, help="批次大小")
    annotate_parser.add_argument("--no_batch", action="store_true", help="禁用批量标注")

    # 对比命令
    compare_parser = subparsers.add_parser("compare", help="对比不同API")
    compare_parser.add_argument("--input", type=str, required=True, help="测试文本文件")
    compare_parser.add_argument("--output", type=str, default="comparison.json", help="输出文件")
    compare_parser.add_argument("--openai_key", type=str, required=True, help="OpenAI API密钥")
    compare_parser.add_argument("--deepseek_key", type=str, required=True, help="DeepSeek API密钥")

    args = parser.parse_args()

    if args.command == "annotate":
        config = AnnotationConfig(
            provider=args.provider,
            model=args.model or ("gpt-4o-mini" if args.provider == "openai" else "deepseek-chat"),
            api_key=args.api_key,
            batch_size=args.batch_size
        )

        annotator = DataAnnotator(config)
        annotator.annotate_file(
            input_path=args.input,
            output_path=args.output,
            use_batch=not args.no_batch
        )

    elif args.command == "compare":
        # 加载测试文本
        with open(args.input, "r", encoding="utf-8") as f:
            if args.input.endswith(".json"):
                data = json.load(f)
                texts = [item["text"] if isinstance(item, dict) else item for item in data]
            else:
                texts = [line.strip() for line in f if line.strip()]

        compare_providers(
            texts=texts[:100],  # 限制100条
            openai_key=args.openai_key,
            deepseek_key=args.deepseek_key,
            output_path=args.output
        )


if __name__ == "__main__":
    main()
