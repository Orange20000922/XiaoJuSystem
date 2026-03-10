"""
弹幕专用标注脚本 - 针对弹幕特点优化

优化策略:
1. 大批量处理 (50条/批，弹幕短小)
2. 简化标注 (只标注情绪，不需要行为/回复长度)
3. 预处理过滤 (去除无意义内容)
4. 模式预标注 (常见表达直接标注，不调用API)
5. 去重聚类 (相似弹幕只标注一次)
6. 支持小模型标注 (知识蒸馏后的模型)
"""

import json
import re
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import argparse

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from configs.model_config import (
    AnnotationAPIConfig,
    DEFAULT_ANNOTATION_CONFIG,
)
from configs.config_loader import AppConfig


# ============== 弹幕专用Prompt (简化版) ==============
DANMAKU_ANNOTATION_PROMPT = """你是弹幕情绪标注专家。请分析以下弹幕的情绪。

# 情绪类型
- joy: 开心、喜悦、满足
- sadness: 难过、伤感、失落
- anger: 愤怒、生气、不满
- fear: 害怕、担忧、紧张
- surprise: 惊讶、震惊、意外
- disgust: 厌恶、反感、鄙视
- neutral: 中性、客观陈述
- excitement: 兴奋、激动、热血
- tenderness: 温柔、感动、温馨
- curiosity: 好奇、疑问、探索

# 强度标准 (0.0-1.0)
- 0.2-0.4: 轻微 (如"还行"、"可以")
- 0.5-0.7: 中等 (如"不错"、"笑了")
- 0.8-1.0: 强烈 (如"哈哈哈哈"、"破防了"、"绝了")

# 弹幕列表 (每行一条)
{danmakus}

# 输出格式 (JSON数组，顺序对应)
[
    {{"emotion": "joy", "intensity": 0.8}},
    {{"emotion": "neutral", "intensity": 0.3}},
    ...
]

只输出JSON数组，不要其他内容。"""


# ============== 模式预标注规则 ==============
# 这些常见模式可以直接标注，不需要调用API
PATTERN_RULES: List[Tuple[str, str, float]] = [
    # (正则模式, 情绪, 强度)

    # 喜悦/开心
    (r'^[哈]{3,}$', 'joy', 0.9),
    (r'^哈哈+', 'joy', 0.8),
    (r'^笑死', 'joy', 0.85),
    (r'^乐了', 'joy', 0.7),
    (r'^好笑', 'joy', 0.6),
    (r'^xswl', 'joy', 0.85),
    (r'^233+', 'joy', 0.8),
    (r'^hhh+', 'joy', 0.75),
    (r'^哈哈哈', 'joy', 0.8),
    (r'笑出[声猪]', 'joy', 0.85),
    (r'^可爱$', 'joy', 0.7),
    (r'^太可爱了', 'joy', 0.8),

    # 兴奋
    (r'^[!！]{2,}$', 'excitement', 0.7),
    (r'^awsl', 'excitement', 0.9),
    (r'^啊+[!！]*$', 'excitement', 0.8),
    (r'^冲[!！]*$', 'excitement', 0.85),
    (r'^来了', 'excitement', 0.7),
    (r'^开始了', 'excitement', 0.6),
    (r'yyds', 'excitement', 0.85),
    (r'绝绝子', 'excitement', 0.8),
    (r'^绝了', 'excitement', 0.85),
    (r'^爱了', 'excitement', 0.8),
    (r'^太强了', 'excitement', 0.85),
    (r'^牛[逼批b]', 'excitement', 0.8),
    (r'^666+', 'excitement', 0.75),
    (r'^厉害', 'excitement', 0.7),
    (r'^猛', 'excitement', 0.7),

    # 温柔/感动
    (r'^泪目', 'tenderness', 0.85),
    (r'^破防', 'tenderness', 0.9),
    (r'^呜呜', 'tenderness', 0.8),
    (r'^[感好]动', 'tenderness', 0.75),
    (r'^爷青回', 'tenderness', 0.8),
    (r'^DNA动了', 'tenderness', 0.75),
    (r'^暖', 'tenderness', 0.6),
    (r'^治愈', 'tenderness', 0.7),
    (r'^太温柔', 'tenderness', 0.8),

    # 惊讶
    (r'^[?？]{2,}$', 'surprise', 0.7),
    (r'^[?？]+$', 'surprise', 0.6),
    (r'^啊[?？]', 'surprise', 0.7),
    (r'^什么', 'surprise', 0.6),
    (r'^卧槽', 'surprise', 0.8),
    (r'^我[草槽艹操]', 'surprise', 0.8),
    (r'^woc', 'surprise', 0.75),
    (r'^我去', 'surprise', 0.7),
    (r'^好家伙', 'surprise', 0.75),
    (r'^离谱', 'surprise', 0.75),
    (r'^[牛这]什么', 'surprise', 0.7),

    # 好奇
    (r'^为什么', 'curiosity', 0.7),
    (r'^怎么', 'curiosity', 0.6),
    (r'^这是', 'curiosity', 0.5),
    (r'^[?？]$', 'curiosity', 0.5),
    (r'什么意思', 'curiosity', 0.6),
    (r'^求[解科]', 'curiosity', 0.7),

    # 愤怒/不满
    (r'^草$', 'anger', 0.7),
    (r'^[艹操]$', 'anger', 0.75),
    (r'^滚', 'anger', 0.85),
    (r'^闭嘴', 'anger', 0.8),
    (r'^傻[逼b]', 'anger', 0.9),
    (r'^sb', 'anger', 0.85),
    (r'^烦死', 'anger', 0.8),
    (r'^恶心', 'disgust', 0.85),

    # 厌恶
    (r'^呕', 'disgust', 0.75),
    (r'^吐了', 'disgust', 0.8),
    (r'^无语', 'disgust', 0.6),
    (r'^服了', 'disgust', 0.65),

    # 悲伤
    (r'^难过', 'sadness', 0.75),
    (r'^伤心', 'sadness', 0.75),
    (r'^哭了', 'sadness', 0.8),
    (r'^[呜哇][呜哇]+', 'sadness', 0.7),
    (r'^悲', 'sadness', 0.65),
    (r'^惨', 'sadness', 0.6),

    # 恐惧
    (r'^害怕', 'fear', 0.75),
    (r'^怕了', 'fear', 0.7),
    (r'^瑟瑟发抖', 'fear', 0.65),
    (r'^慌', 'fear', 0.6),

    # 中性
    (r'^[.。]+$', 'neutral', 0.3),
    (r'^确实', 'neutral', 0.4),
    (r'^是的', 'neutral', 0.3),
    (r'^对$', 'neutral', 0.3),
    (r'^嗯', 'neutral', 0.3),
    (r'^ok', 'neutral', 0.3),
    (r'^好$', 'neutral', 0.4),
    (r'^行$', 'neutral', 0.3),
    (r'^前排', 'neutral', 0.3),
    (r'^打卡', 'neutral', 0.3),
    (r'^火钳刘明', 'neutral', 0.3),
]

# 编译正则表达式
COMPILED_PATTERNS = [(re.compile(p, re.IGNORECASE), e, i) for p, e, i in PATTERN_RULES]


# ============== 过滤规则 ==============
@dataclass
class FilterStats:
    """过滤统计"""
    total: int = 0
    too_short: int = 0
    too_long: int = 0
    pure_emoji: int = 0
    pure_number: int = 0
    meaningless: int = 0
    duplicate: int = 0
    passed: int = 0


def should_filter(text: str) -> Tuple[bool, str]:
    """
    判断弹幕是否应该被过滤

    Returns:
        (是否过滤, 过滤原因)
    """
    text = text.strip()

    # 太短 (少于2个字符)
    if len(text) < 2:
        return True, "too_short"

    # 太长 (超过50字符，可能是复制粘贴的)
    if len(text) > 50:
        return True, "too_long"

    # 纯数字
    if text.isdigit():
        return True, "pure_number"

    # 纯emoji (简单检测)
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map
        "\U0001F700-\U0001F77F"  # alchemical
        "\U0001F780-\U0001F7FF"  # Geometric
        "\U0001F800-\U0001F8FF"  # arrows
        "\U0001F900-\U0001F9FF"  # Supplemental
        "\U0001FA00-\U0001FA6F"  # Chess
        "\U0001FA70-\U0001FAFF"  # Symbols
        "\U00002702-\U000027B0"  # Dingbats
        "]+"
    )
    text_without_emoji = emoji_pattern.sub('', text)
    if len(text_without_emoji) == 0:
        return True, "pure_emoji"

    # 无意义内容
    meaningless_patterns = [
        r'^[1一][1一][4四][5五][1一][4四]$',  # 114514
        r'^[dD][yY][jJ][sS]',  # dyjs
        r'^[来莱][了啦]$',  # 只有"来了"
        r'^第?[一二三四五六七八九十\d]+$',  # 纯序号
        r'^[好嗯啊哦噢]+$',  # 纯语气词
    ]
    for pattern in meaningless_patterns:
        if re.match(pattern, text):
            return True, "meaningless"

    return False, ""


def preprocess_danmaku(text: str) -> str:
    """预处理弹幕文本"""
    text = text.strip()

    # 统一全角转半角
    text = text.replace('！', '!').replace('？', '?')

    # 移除多余空格
    text = re.sub(r'\s+', ' ', text)

    # 限制重复字符 (如 "哈哈哈哈哈哈哈" -> "哈哈哈哈")
    text = re.sub(r'(.)\1{4,}', r'\1\1\1\1', text)

    return text


def get_text_hash(text: str) -> str:
    """获取文本的哈希值（用于去重）"""
    normalized = text.lower().strip()
    return hashlib.md5(normalized.encode()).hexdigest()[:8]


# ============== 弹幕标注器 ==============
class DanmakuAnnotator:
    """弹幕专用标注器"""

    def __init__(
        self,
        config: Optional[AnnotationAPIConfig] = None,
        batch_size: int = 50,  # 弹幕短，可以大批量
        use_pattern_matching: bool = True,
        deduplicate: bool = True,
        max_workers: int = 5,  # API 并发数
    ):
        self.config = config or DEFAULT_ANNOTATION_CONFIG
        self.batch_size = batch_size
        self.use_pattern_matching = use_pattern_matching
        self.deduplicate = deduplicate
        self.max_workers = max_workers

        # API客户端
        from openai import OpenAI
        self.client = OpenAI(
            api_key=self.config.primary_api_key,
            base_url=self.config.primary_base_url
        )
        self.model = self.config.primary_model

        # 统计
        self.stats = {
            "total": 0,
            "filtered": 0,
            "pattern_matched": 0,
            "api_called": 0,
            "deduplicated": 0,
            "api_tokens": 0
        }

    def match_pattern(self, text: str) -> Optional[Dict]:
        """尝试模式匹配标注"""
        for pattern, emotion, intensity in COMPILED_PATTERNS:
            if pattern.search(text):
                return {
                    "text": text,
                    "emotion": emotion,
                    "intensity": intensity,
                    "annotator": "pattern",
                    "confidence": 0.95  # 模式匹配的置信度高
                }
        return None

    def annotate_batch_api(self, texts: List[str]) -> List[Dict]:
        """调用API批量标注"""
        if not texts:
            return []

        # 格式化弹幕列表
        danmaku_list = "\n".join([f"{i+1}. {t}" for i, t in enumerate(texts)])
        prompt = DANMAKU_ANNOTATION_PROMPT.format(danmakus=danmaku_list)

        for attempt in range(self.config.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,  # 低温度提高一致性
                    max_tokens=len(texts) * 50  # 每条弹幕约50 token
                )

                # 代理可能返回原始字符串而非 ChatCompletion 对象
                if isinstance(response, str):
                    content = response.strip()
                    if not content.startswith("["):
                        raise ValueError(f"代理返回非JSON字符串: {content[:100]}")
                else:
                    content = response.choices[0].message.content.strip()
                    # 记录token使用
                    if hasattr(response, 'usage') and response.usage:
                        self.stats["api_tokens"] += response.usage.total_tokens

                # 解析JSON
                # 尝试直接解析
                try:
                    results = json.loads(content)
                except json.JSONDecodeError:
                    # 提取JSON数组
                    match = re.search(r'\[.*\]', content, re.DOTALL)
                    if match:
                        results = json.loads(match.group())
                    else:
                        raise ValueError(f"无法解析JSON: {content[:200]}")

                # 添加原始文本
                annotated = []
                for i, result in enumerate(results):
                    if i < len(texts):
                        result["text"] = texts[i]
                        result["annotator"] = self.model
                        result["confidence"] = result.get("confidence", 0.8)
                        annotated.append(result)

                self.stats["api_called"] += len(texts)
                return annotated

            except Exception as e:
                print(f"API调用失败 (尝试 {attempt+1}/{self.config.max_retries}): {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(2 ** attempt)  # 指数退避

        # 全部失败，返回空结果
        return [{"text": t, "emotion": "neutral", "intensity": 0.5,
                 "annotator": "fallback", "confidence": 0.3} for t in texts]

    def annotate_file(
        self,
        input_path: str,
        output_path: str,
        text_field: str = "text",
        resume: bool = True,
        max_samples: Optional[int] = None
    ) -> List[Dict]:
        """
        标注弹幕文件

        Args:
            input_path: 输入文件 (JSON/JSONL/CSV)
            output_path: 输出文件
            text_field: 文本字段名
            resume: 是否断点续传
            max_samples: 最大标注数量
        """
        # 加载数据
        input_path = Path(input_path)
        data = self._load_data(input_path, text_field)

        print(f"加载了 {len(data)} 条弹幕")
        self.stats["total"] = len(data)

        if max_samples:
            data = data[:max_samples]
            print(f"限制为 {max_samples} 条")

        # 检查断点
        output_path = Path(output_path)
        annotated = []
        processed_hashes = set()

        if resume and output_path.exists():
            with open(output_path, "r", encoding="utf-8") as f:
                annotated = json.load(f)
            processed_hashes = {get_text_hash(a["text"]) for a in annotated}
            print(f"从断点续传: 已完成 {len(annotated)} 条")

        # 第一阶段：过滤 + 预处理 + 去重
        print("\n[阶段1] 过滤和预处理...")
        filter_stats = FilterStats()
        valid_items = []
        seen_hashes = set(processed_hashes)

        for item in tqdm(data, desc="预处理"):
            text = item if isinstance(item, str) else item.get(text_field, "")
            filter_stats.total += 1

            # 预处理
            text = preprocess_danmaku(text)

            # 过滤
            should_skip, reason = should_filter(text)
            if should_skip:
                setattr(filter_stats, reason, getattr(filter_stats, reason) + 1)
                continue

            # 去重
            text_hash = get_text_hash(text)
            if self.deduplicate and text_hash in seen_hashes:
                filter_stats.duplicate += 1
                continue

            seen_hashes.add(text_hash)
            filter_stats.passed += 1
            valid_items.append(text)

        self._print_filter_stats(filter_stats)
        self.stats["filtered"] = filter_stats.total - filter_stats.passed

        # 第二阶段：模式匹配
        print("\n[阶段2] 模式匹配...")
        pattern_matched = []
        need_api = []

        if self.use_pattern_matching:
            for text in tqdm(valid_items, desc="模式匹配"):
                result = self.match_pattern(text)
                if result:
                    pattern_matched.append(result)
                else:
                    need_api.append(text)

            print(f"模式匹配: {len(pattern_matched)} 条")
            print(f"需要API: {len(need_api)} 条")
            self.stats["pattern_matched"] = len(pattern_matched)
        else:
            need_api = valid_items

        annotated.extend(pattern_matched)

        # 第三阶段：API标注（并发）
        if need_api:
            print(f"\n[阶段3] API标注 ({len(need_api)} 条, 并发={self.max_workers})...")

            batches = [need_api[i:i + self.batch_size]
                       for i in range(0, len(need_api), self.batch_size)]
            save_every = max(1, 500 // self.batch_size)  # 约每500条保存

            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {executor.submit(self.annotate_batch_api, b): idx
                           for idx, b in enumerate(batches)}
                with tqdm(total=len(batches), desc="API标注") as pbar:
                    for future in as_completed(futures):
                        results = future.result()
                        annotated.extend(results)
                        pbar.update(1)
                        if pbar.n % save_every == 0:
                            self._save_results(annotated, output_path)

        # 最终保存
        self._save_results(annotated, output_path)

        # 统计
        self._print_final_stats(annotated)

        return annotated

    def _load_data(self, path: Path, text_field: str) -> List:
        """加载数据文件"""
        if path.suffix == ".json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                # 处理爬虫输出格式 (包含video/comments/danmakus)
                if isinstance(data, list) and data and "danmakus" in data[0]:
                    danmakus = []
                    for item in data:
                        for d in item.get("danmakus", []):
                            danmakus.append(d.get(text_field, d.get("text", "")))
                    return danmakus
                return data

        elif path.suffix == ".jsonl":
            data = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        data.append(item.get(text_field, item))
            return data

        elif path.suffix == ".csv":
            import csv
            data = []
            with open(path, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    data.append(row.get(text_field, ""))
            return data

        elif path.suffix == ".txt":
            with open(path, "r", encoding="utf-8") as f:
                return [line.strip() for line in f if line.strip()]

        else:
            raise ValueError(f"不支持的文件格式: {path.suffix}")

    def _save_results(self, results: List[Dict], output_path: Path):
        """保存结果"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    def _print_filter_stats(self, stats: FilterStats):
        """打印过滤统计"""
        print("\n过滤统计:")
        print(f"  总数: {stats.total}")
        print(f"  太短: {stats.too_short}")
        print(f"  太长: {stats.too_long}")
        print(f"  纯emoji: {stats.pure_emoji}")
        print(f"  纯数字: {stats.pure_number}")
        print(f"  无意义: {stats.meaningless}")
        print(f"  重复: {stats.duplicate}")
        print(f"  通过: {stats.passed} ({stats.passed/stats.total*100:.1f}%)")

    def _print_final_stats(self, results: List[Dict]):
        """打印最终统计"""
        emotions = Counter(r.get("emotion", "unknown") for r in results)
        annotators = Counter(r.get("annotator", "unknown") for r in results)

        print("\n" + "=" * 50)
        print("标注完成!")
        print("=" * 50)

        print(f"\n总计: {len(results)} 条")
        print(f"过滤: {self.stats['filtered']} 条")
        print(f"模式匹配: {self.stats['pattern_matched']} 条")
        print(f"API调用: {self.stats['api_called']} 条")
        print(f"API Token: {self.stats['api_tokens']:,}")

        # 估算成本 (GPT-5.2-instant 约 $0.15/1M tokens)
        cost = self.stats['api_tokens'] / 1_000_000 * 0.15
        print(f"预估成本: ${cost:.4f}")

        print("\n情绪分布:")
        for emotion, count in emotions.most_common():
            print(f"  {emotion}: {count} ({count/len(results)*100:.1f}%)")

        print("\n标注来源:")
        for annotator, count in annotators.most_common():
            print(f"  {annotator}: {count} ({count/len(results)*100:.1f}%)")


# ============== 小模型标注器 ==============
class ModelAnnotator:
    """
    使用训练好的小模型进行标注

    用于知识蒸馏流程的第二阶段：
    1. GPT标注种子数据 → 训练小模型
    2. 小模型标注大规模数据 ← 这个类
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        batch_size: int = 128,
        use_pattern_matching: bool = True,
        deduplicate: bool = True
    ):
        """
        Args:
            model_path: 训练好的标注模型路径
            device: 推理设备
            batch_size: 批量大小
            use_pattern_matching: 是否先用模式匹配
            deduplicate: 是否去重
        """
        self.device = device
        self.batch_size = batch_size
        self.use_pattern_matching = use_pattern_matching
        self.deduplicate = deduplicate

        # 加载模型
        print(f"加载标注模型: {model_path}")
        from models.annotation_model import AnnotationInferencer
        self.inferencer = AnnotationInferencer(
            model_path=model_path,
            device=device,
            batch_size=batch_size
        )

        # 统计
        self.stats = {
            "total": 0,
            "filtered": 0,
            "pattern_matched": 0,
            "model_annotated": 0
        }

    def match_pattern(self, text: str) -> Optional[Dict]:
        """尝试模式匹配标注"""
        for pattern, emotion, intensity in COMPILED_PATTERNS:
            if pattern.search(text):
                return {
                    "text": text,
                    "emotion": emotion,
                    "intensity": intensity,
                    "annotator": "pattern",
                    "confidence": 0.95
                }
        return None

    def annotate_file(
        self,
        input_path: str,
        output_path: str,
        text_field: str = "text",
        resume: bool = True,
        max_samples: Optional[int] = None,
        min_confidence: float = 0.0  # 最低置信度阈值
    ) -> List[Dict]:
        """
        使用小模型标注文件

        Args:
            input_path: 输入文件
            output_path: 输出文件
            text_field: 文本字段名
            resume: 是否断点续传
            max_samples: 最大标注数量
            min_confidence: 最低置信度阈值，低于此值的标注会被标记
        """
        # 加载数据
        input_path = Path(input_path)
        data = self._load_data(input_path, text_field)

        print(f"加载了 {len(data)} 条数据")
        self.stats["total"] = len(data)

        if max_samples:
            data = data[:max_samples]
            print(f"限制为 {max_samples} 条")

        # 检查断点
        output_path = Path(output_path)
        annotated = []
        processed_hashes = set()

        if resume and output_path.exists():
            with open(output_path, "r", encoding="utf-8") as f:
                annotated = json.load(f)
            processed_hashes = {get_text_hash(a["text"]) for a in annotated}
            print(f"从断点续传: 已完成 {len(annotated)} 条")

        # 第一阶段：过滤 + 预处理 + 去重
        print("\n[阶段1] 过滤和预处理...")
        filter_stats = FilterStats()
        valid_items = []
        seen_hashes = set(processed_hashes)

        for item in tqdm(data, desc="预处理"):
            text = item if isinstance(item, str) else item.get(text_field, "")
            filter_stats.total += 1

            # 预处理
            text = preprocess_danmaku(text)

            # 过滤
            should_skip, reason = should_filter(text)
            if should_skip:
                setattr(filter_stats, reason, getattr(filter_stats, reason) + 1)
                continue

            # 去重
            text_hash = get_text_hash(text)
            if self.deduplicate and text_hash in seen_hashes:
                filter_stats.duplicate += 1
                continue

            seen_hashes.add(text_hash)
            filter_stats.passed += 1
            valid_items.append(text)

        self._print_filter_stats(filter_stats)
        self.stats["filtered"] = filter_stats.total - filter_stats.passed

        # 第二阶段：模式匹配
        pattern_matched = []
        need_model = []

        if self.use_pattern_matching:
            print("\n[阶段2] 模式匹配...")
            for text in tqdm(valid_items, desc="模式匹配"):
                result = self.match_pattern(text)
                if result:
                    pattern_matched.append(result)
                else:
                    need_model.append(text)

            print(f"模式匹配: {len(pattern_matched)} 条")
            print(f"需要模型: {len(need_model)} 条")
            self.stats["pattern_matched"] = len(pattern_matched)
        else:
            need_model = valid_items

        annotated.extend(pattern_matched)

        # 第三阶段：小模型标注
        if need_model:
            print(f"\n[阶段3] 小模型标注 ({len(need_model)} 条)...")

            model_results = self.inferencer.annotate_texts(
                need_model,
                show_progress=True
            )

            # 标记低置信度
            for result in model_results:
                if result["confidence"] < min_confidence:
                    result["low_confidence"] = True

            annotated.extend(model_results)
            self.stats["model_annotated"] = len(model_results)

            # 定期保存
            self._save_results(annotated, output_path)

        # 最终保存
        self._save_results(annotated, output_path)

        # 统计
        self._print_final_stats(annotated)

        return annotated

    def _load_data(self, path: Path, text_field: str) -> List:
        """加载数据文件"""
        if path.suffix == ".json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list) and data and "danmakus" in data[0]:
                    danmakus = []
                    for item in data:
                        for d in item.get("danmakus", []):
                            danmakus.append(d.get(text_field, d.get("text", "")))
                    return danmakus
                return data

        elif path.suffix == ".jsonl":
            data = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        data.append(item.get(text_field, item))
            return data

        elif path.suffix == ".csv":
            import csv
            data = []
            with open(path, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    data.append(row.get(text_field, ""))
            return data

        elif path.suffix == ".txt":
            with open(path, "r", encoding="utf-8") as f:
                return [line.strip() for line in f if line.strip()]

        else:
            raise ValueError(f"不支持的文件格式: {path.suffix}")

    def _save_results(self, results: List[Dict], output_path: Path):
        """保存结果"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    def _print_filter_stats(self, stats: FilterStats):
        """打印过滤统计"""
        print("\n过滤统计:")
        print(f"  总数: {stats.total}")
        print(f"  太短: {stats.too_short}")
        print(f"  太长: {stats.too_long}")
        print(f"  纯emoji: {stats.pure_emoji}")
        print(f"  纯数字: {stats.pure_number}")
        print(f"  无意义: {stats.meaningless}")
        print(f"  重复: {stats.duplicate}")
        print(f"  通过: {stats.passed} ({stats.passed/stats.total*100:.1f}%)")

    def _print_final_stats(self, results: List[Dict]):
        """打印最终统计"""
        emotions = Counter(r.get("emotion", "unknown") for r in results)
        annotators = Counter(r.get("annotator", "unknown") for r in results)

        # 统计低置信度
        low_conf_count = sum(1 for r in results if r.get("low_confidence", False))

        print("\n" + "=" * 50)
        print("标注完成!")
        print("=" * 50)

        print(f"\n总计: {len(results)} 条")
        print(f"过滤: {self.stats['filtered']} 条")
        print(f"模式匹配: {self.stats['pattern_matched']} 条")
        print(f"模型标注: {self.stats['model_annotated']} 条")
        if low_conf_count > 0:
            print(f"低置信度: {low_conf_count} 条")

        # 平均置信度
        confidences = [r.get("confidence", 0) for r in results]
        avg_conf = sum(confidences) / len(confidences) if confidences else 0
        print(f"平均置信度: {avg_conf:.3f}")

        print("\n情绪分布:")
        for emotion, count in emotions.most_common():
            print(f"  {emotion}: {count} ({count/len(results)*100:.1f}%)")

        print("\n标注来源:")
        for annotator, count in annotators.most_common():
            print(f"  {annotator}: {count} ({count/len(results)*100:.1f}%)")


# ============== 数据转换工具 ==============
def convert_crawler_output(
    crawler_output: str,
    output_path: str,
    extract_type: str = "danmakus"  # "danmakus" 或 "comments"
):
    """
    将爬虫输出转换为标注器输入格式

    Args:
        crawler_output: 爬虫输出的JSON文件
        output_path: 转换后的输出文件
        extract_type: 提取类型
    """
    with open(crawler_output, "r", encoding="utf-8") as f:
        data = json.load(f)

    extracted = []
    for item in data:
        if extract_type == "danmakus":
            for d in item.get("danmakus", []):
                extracted.append({
                    "text": d.get("text", ""),
                    "video_bvid": item.get("video", {}).get("bvid", ""),
                    "dm_time": d.get("dm_time", 0)
                })
        else:  # comments
            for c in item.get("comments", []):
                extracted.append({
                    "text": c.get("content", ""),
                    "video_bvid": item.get("video", {}).get("bvid", ""),
                    "uname": c.get("uname", ""),
                    "like_count": c.get("like_count", 0)
                })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(extracted, f, ensure_ascii=False, indent=2)

    print(f"提取了 {len(extracted)} 条{extract_type}")
    print(f"保存到: {output_path}")


# ============== 主函数 ==============
def main():
    parser = argparse.ArgumentParser(description="弹幕专用标注工具")
    parser.add_argument("--config", type=str, default=None,
                        help="config.json 路径（未指定 CLI 参数时从中读取标注配置）")
    subparsers = parser.add_subparsers(dest="command", help="命令")

    # API标注命令
    annotate_parser = subparsers.add_parser("annotate", help="使用API标注弹幕")
    annotate_parser.add_argument("--input", type=str, required=True, help="输入文件")
    annotate_parser.add_argument("--output", type=str, required=True, help="输出文件")
    annotate_parser.add_argument("--model", type=str, default=None,
                                help="API模型（默认从 config.json 读取）")
    annotate_parser.add_argument("--api_key", type=str, default=None, help="API密钥")
    annotate_parser.add_argument("--base_url", type=str, default=None, help="API URL")
    annotate_parser.add_argument("--batch_size", type=int, default=50, help="批次大小")
    annotate_parser.add_argument("--workers", type=int, default=5, help="API并发数 (默认: 5)")
    annotate_parser.add_argument("--max_samples", type=int, default=None, help="最大标注数")
    annotate_parser.add_argument("--no_pattern", action="store_true", help="禁用模式匹配")
    annotate_parser.add_argument("--no_dedup", action="store_true", help="禁用去重")

    # 小模型标注命令 (新增)
    model_parser = subparsers.add_parser("annotate-model", help="使用训练好的小模型标注")
    model_parser.add_argument("--input", type=str, required=True, help="输入文件")
    model_parser.add_argument("--output", type=str, required=True, help="输出文件")
    model_parser.add_argument("--model_path", type=str, required=True, help="小模型检查点路径")
    model_parser.add_argument("--device", type=str, default="cuda", help="推理设备")
    model_parser.add_argument("--batch_size", type=int, default=128, help="批次大小")
    model_parser.add_argument("--max_samples", type=int, default=None, help="最大标注数")
    model_parser.add_argument("--min_confidence", type=float, default=0.5, help="最低置信度阈值")
    model_parser.add_argument("--no_pattern", action="store_true", help="禁用模式匹配")
    model_parser.add_argument("--no_dedup", action="store_true", help="禁用去重")

    # 转换命令
    convert_parser = subparsers.add_parser("convert", help="转换爬虫输出")
    convert_parser.add_argument("--input", type=str, required=True, help="爬虫输出文件")
    convert_parser.add_argument("--output", type=str, required=True, help="输出文件")
    convert_parser.add_argument("--type", type=str, default="danmakus",
                               choices=["danmakus", "comments"], help="提取类型")

    # 测试命令
    test_parser = subparsers.add_parser("test", help="测试模式匹配")
    test_parser.add_argument("--text", type=str, required=True, help="测试文本")

    args = parser.parse_args()

    # 从 config.json 加载标注默认配置
    try:
        app_cfg = AppConfig.load(args.config)
        ann_defaults = app_cfg.annotation
    except FileNotFoundError:
        ann_defaults = AnnotationAPIConfig()

    if args.command == "annotate":
        # CLI 参数覆盖 config.json 值
        config = AnnotationAPIConfig(
            primary_model=args.model or ann_defaults.primary_model,
            primary_api_key=args.api_key or ann_defaults.primary_api_key,
            primary_base_url=args.base_url or ann_defaults.primary_base_url,
            fallback_provider=ann_defaults.fallback_provider,
            fallback_model=ann_defaults.fallback_model,
            fallback_api_key=ann_defaults.fallback_api_key,
            fallback_base_url=ann_defaults.fallback_base_url,
            temperature=ann_defaults.temperature,
            max_retries=ann_defaults.max_retries,
        )

        annotator = DanmakuAnnotator(
            config=config,
            batch_size=args.batch_size,
            use_pattern_matching=not args.no_pattern,
            deduplicate=not args.no_dedup,
            max_workers=args.workers,
        )

        annotator.annotate_file(
            input_path=args.input,
            output_path=args.output,
            max_samples=args.max_samples
        )

    elif args.command == "annotate-model":
        # 小模型标注
        annotator = ModelAnnotator(
            model_path=args.model_path,
            device=args.device,
            batch_size=args.batch_size,
            use_pattern_matching=not args.no_pattern,
            deduplicate=not args.no_dedup
        )

        annotator.annotate_file(
            input_path=args.input,
            output_path=args.output,
            max_samples=args.max_samples,
            min_confidence=args.min_confidence
        )

    elif args.command == "convert":
        convert_crawler_output(
            crawler_output=args.input,
            output_path=args.output,
            extract_type=args.type
        )

    elif args.command == "test":
        # 测试模式匹配
        for pattern, emotion, intensity in COMPILED_PATTERNS:
            if pattern.search(args.text):
                print(f"模式匹配成功:")
                print(f"  情绪: {emotion}")
                print(f"  强度: {intensity}")
                return
        print("未匹配到模式，需要API/模型标注")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
