"""
内容过滤模块 - 清洗不良内容

过滤类型:
1. 辱骂/攻击性语言
2. 政治敏感内容
3. 色情/低俗内容
4. 广告/垃圾信息
5. 极端负面情绪

设计原则:
- 宁可误杀，不可放过 (训练数据质量优先)
- 可配置过滤强度
- 支持自定义词库扩展
- 词库从外部文件加载，避免代码中出现敏感词
"""

import re
import json
import base64
from pathlib import Path
from typing import List, Tuple, Optional, Set, Dict
from dataclasses import dataclass
from enum import Enum
from tqdm import tqdm


class FilterLevel(Enum):
    """过滤强度"""
    STRICT = "strict"      # 严格：宁可误杀
    NORMAL = "normal"      # 正常：平衡
    LOOSE = "loose"        # 宽松：尽量保留


class FilterCategory(Enum):
    """过滤类别"""
    PROFANITY = "profanity"           # 辱骂
    POLITICAL = "political"           # 政治敏感
    PORNOGRAPHIC = "pornographic"     # 色情低俗
    ADVERTISEMENT = "advertisement"   # 广告垃圾
    TOXIC = "toxic"                   # 极端负面
    SPAM = "spam"                     # 刷屏/无意义


@dataclass
class FilterResult:
    """过滤结果"""
    text: str
    passed: bool
    category: Optional[FilterCategory] = None
    matched_pattern: Optional[str] = None
    confidence: float = 1.0


@dataclass
class FilterStats:
    """过滤统计"""
    total: int = 0
    passed: int = 0
    profanity: int = 0
    political: int = 0
    pornographic: int = 0
    advertisement: int = 0
    toxic: int = 0
    spam: int = 0

    def add(self, category: Optional[FilterCategory]):
        self.total += 1
        if category is None:
            self.passed += 1
        else:
            setattr(self, category.value, getattr(self, category.value) + 1)

    def summary(self) -> str:
        blocked = self.total - self.passed
        if self.total == 0:
            return "总计: 0"
        return (
            f"总计: {self.total} | 通过: {self.passed} ({self.passed/self.total*100:.1f}%) | "
            f"过滤: {blocked} ({blocked/self.total*100:.1f}%)"
        )


# ============== 词库管理 ==============
DEFAULT_WORDLIST_DIR = Path(__file__).parent / "wordlists"


def get_default_wordlist_path() -> Path:
    """获取默认词库目录"""
    return DEFAULT_WORDLIST_DIR


def init_wordlists(wordlist_dir: Optional[Path] = None):
    """
    初始化词库文件（首次使用时调用）

    词库使用base64编码存储，避免直接暴露敏感词
    """
    if wordlist_dir is None:
        wordlist_dir = DEFAULT_WORDLIST_DIR

    wordlist_dir.mkdir(parents=True, exist_ok=True)

    # 内置的基础词库（base64编码）
    # 这样敏感词不会直接出现在代码中
    builtin_wordlists = {
        "profanity.txt": _get_builtin_profanity(),
        "political.txt": _get_builtin_political(),
        "pornographic.txt": _get_builtin_pornographic(),
        "advertisement.txt": _get_builtin_advertisement(),
        "toxic.txt": _get_builtin_toxic(),
        "spam.txt": _get_builtin_spam(),
    }

    for filename, patterns in builtin_wordlists.items():
        filepath = wordlist_dir / filename
        if not filepath.exists():
            with open(filepath, "w", encoding="utf-8") as f:
                f.write("# 过滤规则 (每行一个正则表达式)\n")
                f.write("# 以 # 开头的行为注释\n\n")
                for pattern in patterns:
                    f.write(pattern + "\n")
            print(f"创建词库: {filepath}")


def _decode_patterns(encoded: str) -> List[str]:
    """解码base64编码的词库"""
    try:
        decoded = base64.b64decode(encoded).decode("utf-8")
        return [line.strip() for line in decoded.split("\n") if line.strip()]
    except Exception:
        return []


def _get_builtin_profanity() -> List[str]:
    """获取内置辱骂词库"""
    # Base64编码的正则表达式列表
    encoded = (
        "W+WCu+eFnuayuV1b6YC8Luer"
        "lOWxhOaJuV0KW+iJueaTjeiPj+iNieaXpeW5sl3kvaBb5aaI6ams6bq7XQpb5ru"
        "a6KGrXeibi+WtkArlupnniakK5Z6D5Zy+W+S6uui0p10KW+eMqueLl13ni5fkuI"
        "3lpoIK55WcW+eUn+eJsl0KW+eOi+W/mF3lhavom4sK5re36JuLClvkuowyXVvp"
        "gLzmr5Tnmb5dCnNiClvmrbvsi11b5YWo5a62XQpb5q275Y67XVvniLjniLnlpoh"
        "lppdXClvmu5pndW5d5Ye6W+WOu+S4rV0K6Zet5ZitCuWQg1vlsb/nv5RdCueLl+"
        "S4nOilvwpcYmYrdStjK2srCg=="
    )
    return _decode_patterns(encoded)


def _get_builtin_political() -> List[str]:
    """获取内置政治敏感词库"""
    encoded = (
        "W+S4u+W4rV0uKlvkuIvlj7Dovp7ogYxdClvpopblr7zmoLjlv4NdLipb5om56K+"
        "V5Y+N5a+5XQpb5YWt5Zub5aSp5LqL5Lu2XQpb5Y+w5riv6JePXVvni6xkdV0K"
    )
    return _decode_patterns(encoded)


def _get_builtin_pornographic() -> List[str]:
    """获取内置色情词库"""
    encoded = (
        "W+WBmuW5sl3niLEKW+aAp13kuqQKW+e6pueCruS4gOWknOaDhV0KW+WrluWomOW"
        "WluS6pl0KXGJzZXhcYgpcYnBvcm4KXGJudWRl"
    )
    return _decode_patterns(encoded)


def _get_builtin_advertisement() -> List[str]:
    """获取内置广告词库"""
    encoded = (
        "W+WKoOKelV1b5oiR5b6udl0uKlxkezUsfQpb5b6ud3markup5L+hW+WPtzpdP1xzKlt"
        "hLXpBLVowLTldezUsfQpb5YWN6LS56aKG5Y+WXS4qW+e6ouWMheS8mOaDr+WIuF0K"
        "W+eCueWHu+mTvuaOpV0uKlvpooblj5bkuIvovb1dCmh0dHBzPzovL1teXHNdezEw"
        "LH0KKC57Myx9KVwxezMsfQ=="
    )
    return _decode_patterns(encoded)


def _get_builtin_toxic() -> List[str]:
    """获取内置极端负面词库"""
    encoded = (
        "W+Wls+aLs+eUsOWbrV0KW+aZruS/oeeUt+Wls10KW+adgOWFiV0uKlvkurpdClvp"
        "g73or6XmrbvdClvnga3mrbvdLipb5Lq6XQ=="
    )
    return _decode_patterns(encoded)


def _get_builtin_spam() -> List[str]:
    """获取内置刷屏词库"""
    encoded = (
        "Xltbwq7jgILvvIwsXSskJApeLlvlk4hoXXsxMCx9JApeWz/vvJ8h77yBXXs1LH0k"
        "Cl4oLilcMXs5LH0k"
    )
    return _decode_patterns(encoded)


def load_wordlist(filepath: Path) -> List[str]:
    """从文件加载词库"""
    patterns = []
    if filepath.exists():
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                # 跳过空行和注释
                if line and not line.startswith("#"):
                    patterns.append(line)
    return patterns


class ContentFilter:
    """内容过滤器"""

    def __init__(
        self,
        level: FilterLevel = FilterLevel.NORMAL,
        wordlist_dir: Optional[Path] = None,
        custom_blacklist: Optional[List[str]] = None,
        custom_whitelist: Optional[List[str]] = None,
        enable_categories: Optional[List[FilterCategory]] = None,
        auto_init: bool = True
    ):
        """
        初始化过滤器

        Args:
            level: 过滤强度
            wordlist_dir: 词库目录路径
            custom_blacklist: 自定义黑名单词汇
            custom_whitelist: 自定义白名单词汇
            enable_categories: 启用的过滤类别 (None=全部启用)
            auto_init: 是否自动初始化词库文件
        """
        self.level = level
        self.whitelist: Set[str] = set(custom_whitelist or [])

        # 词库目录
        self.wordlist_dir = wordlist_dir or DEFAULT_WORDLIST_DIR

        # 自动初始化词库
        if auto_init and not self.wordlist_dir.exists():
            init_wordlists(self.wordlist_dir)

        # 启用的类别
        if enable_categories is None:
            self.enable_categories = set(FilterCategory)
        else:
            self.enable_categories = set(enable_categories)

        # 编译正则表达式
        self.patterns: Dict[FilterCategory, List[re.Pattern]] = {}
        self._load_and_compile_patterns()

        # 添加自定义黑名单
        if custom_blacklist:
            custom_patterns = [re.escape(word) for word in custom_blacklist]
            if FilterCategory.PROFANITY not in self.patterns:
                self.patterns[FilterCategory.PROFANITY] = []
            self.patterns[FilterCategory.PROFANITY].extend([
                re.compile(p, re.IGNORECASE) for p in custom_patterns
            ])

        self.stats = FilterStats()

    def _load_and_compile_patterns(self):
        """从文件加载并编译正则表达式"""
        category_files = {
            FilterCategory.PROFANITY: "profanity.txt",
            FilterCategory.POLITICAL: "political.txt",
            FilterCategory.PORNOGRAPHIC: "pornographic.txt",
            FilterCategory.ADVERTISEMENT: "advertisement.txt",
            FilterCategory.TOXIC: "toxic.txt",
            FilterCategory.SPAM: "spam.txt",
        }

        for category, filename in category_files.items():
            if category in self.enable_categories:
                filepath = self.wordlist_dir / filename
                patterns = load_wordlist(filepath)

                # 如果文件不存在或为空，使用内置词库
                if not patterns:
                    builtin_func = {
                        FilterCategory.PROFANITY: _get_builtin_profanity,
                        FilterCategory.POLITICAL: _get_builtin_political,
                        FilterCategory.PORNOGRAPHIC: _get_builtin_pornographic,
                        FilterCategory.ADVERTISEMENT: _get_builtin_advertisement,
                        FilterCategory.TOXIC: _get_builtin_toxic,
                        FilterCategory.SPAM: _get_builtin_spam,
                    }.get(category)
                    if builtin_func:
                        patterns = builtin_func()

                compiled = []
                for p in patterns:
                    try:
                        compiled.append(re.compile(p, re.IGNORECASE))
                    except re.error as e:
                        print(f"警告: 无效的正则表达式 '{p}': {e}")

                self.patterns[category] = compiled

    def reload_wordlists(self):
        """重新加载词库（修改词库文件后调用）"""
        self.patterns.clear()
        self._load_and_compile_patterns()

    def _check_whitelist(self, text: str) -> bool:
        """检查是否在白名单中"""
        text_lower = text.lower()
        for word in self.whitelist:
            if word.lower() in text_lower:
                return True
        return False

    def filter_single(self, text: str) -> FilterResult:
        """
        过滤单条文本

        Returns:
            FilterResult: 过滤结果
        """
        text = text.strip()

        # 空文本直接过滤
        if not text:
            return FilterResult(text=text, passed=False, category=FilterCategory.SPAM)

        # 白名单检查
        if self._check_whitelist(text):
            return FilterResult(text=text, passed=True)

        # 依次检查各类别
        for category, patterns in self.patterns.items():
            for pattern in patterns:
                match = pattern.search(text)
                if match:
                    return FilterResult(
                        text=text,
                        passed=False,
                        category=category,
                        matched_pattern=match.group()
                    )

        return FilterResult(text=text, passed=True)

    def filter_batch(self, texts: List[str]) -> Tuple[List[str], List[FilterResult]]:
        """
        批量过滤

        Returns:
            (通过的文本列表, 所有过滤结果)
        """
        passed = []
        results = []

        for text in texts:
            result = self.filter_single(text)
            results.append(result)
            self.stats.add(result.category)

            if result.passed:
                passed.append(text)

        return passed, results

    def filter_file(
        self,
        input_path: str,
        output_path: str,
        text_field: str = "text",
        save_filtered: bool = True
    ) -> Tuple[int, int]:
        """
        过滤整个文件

        Args:
            input_path: 输入文件
            output_path: 输出文件 (通过的数据)
            text_field: 文本字段名
            save_filtered: 是否保存被过滤的数据

        Returns:
            (通过数量, 过滤数量)
        """
        input_path = Path(input_path)
        output_path = Path(output_path)

        # 加载数据
        data = self._load_data(input_path)
        print(f"加载了 {len(data)} 条数据")

        passed_data = []
        filtered_data = []

        for item in tqdm(data, desc="过滤中"):
            if isinstance(item, str):
                text = item
            else:
                text = item.get(text_field, "")

            result = self.filter_single(text)
            self.stats.add(result.category)

            if result.passed:
                passed_data.append(item)
            else:
                if save_filtered:
                    filtered_item = item if isinstance(item, dict) else {"text": item}
                    if isinstance(filtered_item, dict):
                        filtered_item["_filter_category"] = result.category.value if result.category else None
                        filtered_item["_filter_matched"] = result.matched_pattern
                    filtered_data.append(filtered_item)

        # 保存通过的数据
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._save_data(passed_data, output_path)
        print(f"保存通过数据: {output_path} ({len(passed_data)} 条)")

        # 保存被过滤的数据
        if save_filtered and filtered_data:
            filtered_path = output_path.parent / f"{output_path.stem}_filtered{output_path.suffix}"
            self._save_data(filtered_data, filtered_path)
            print(f"保存过滤数据: {filtered_path} ({len(filtered_data)} 条)")

        # 打印统计
        self._print_stats()

        return len(passed_data), len(filtered_data)

    def _load_data(self, path: Path) -> List:
        """加载数据"""
        if path.suffix == ".json":
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        elif path.suffix == ".jsonl":
            data = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
            return data
        elif path.suffix == ".txt":
            with open(path, "r", encoding="utf-8") as f:
                return [line.strip() for line in f if line.strip()]
        else:
            raise ValueError(f"不支持的格式: {path.suffix}")

    def _save_data(self, data: List, path: Path):
        """保存数据"""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _print_stats(self):
        """打印统计"""
        print("\n" + "=" * 50)
        print("过滤统计")
        print("=" * 50)
        print(f"总计: {self.stats.total}")
        if self.stats.total > 0:
            print(f"通过: {self.stats.passed} ({self.stats.passed/self.stats.total*100:.1f}%)")
            print(f"\n被过滤:")
            print(f"  辱骂攻击: {self.stats.profanity}")
            print(f"  政治敏感: {self.stats.political}")
            print(f"  色情低俗: {self.stats.pornographic}")
            print(f"  广告垃圾: {self.stats.advertisement}")
            print(f"  极端负面: {self.stats.toxic}")
            print(f"  无意义刷屏: {self.stats.spam}")
        print("=" * 50)


# ============== 便捷函数 ==============
def quick_filter(texts: List[str], level: FilterLevel = FilterLevel.NORMAL) -> List[str]:
    """快速过滤文本列表"""
    f = ContentFilter(level=level)
    passed, _ = f.filter_batch(texts)
    return passed


def is_clean(text: str, level: FilterLevel = FilterLevel.NORMAL) -> bool:
    """检查单条文本是否干净"""
    f = ContentFilter(level=level)
    result = f.filter_single(text)
    return result.passed


# ============== 命令行接口 ==============
def main():
    import argparse

    parser = argparse.ArgumentParser(description="内容过滤工具")
    subparsers = parser.add_subparsers(dest="command", help="命令")

    # 初始化词库
    init_parser = subparsers.add_parser("init", help="初始化词库文件")
    init_parser.add_argument("--dir", type=str, default=None, help="词库目录")

    # 过滤文件
    filter_parser = subparsers.add_parser("filter", help="过滤文件")
    filter_parser.add_argument("--input", type=str, required=True, help="输入文件")
    filter_parser.add_argument("--output", type=str, required=True, help="输出文件")
    filter_parser.add_argument("--field", type=str, default="text", help="文本字段名")
    filter_parser.add_argument("--level", type=str, default="normal",
                              choices=["strict", "normal", "loose"], help="过滤强度")
    filter_parser.add_argument("--wordlist-dir", type=str, default=None, help="词库目录")
    filter_parser.add_argument("--no-save-filtered", action="store_true",
                              help="不保存被过滤的数据")

    # 测试单条
    test_parser = subparsers.add_parser("test", help="测试单条文本")
    test_parser.add_argument("--text", type=str, required=True, help="测试文本")
    test_parser.add_argument("--level", type=str, default="normal",
                            choices=["strict", "normal", "loose"], help="过滤强度")

    # 统计分析
    analyze_parser = subparsers.add_parser("analyze", help="分析文件中的敏感内容分布")
    analyze_parser.add_argument("--input", type=str, required=True, help="输入文件")
    analyze_parser.add_argument("--field", type=str, default="text", help="文本字段名")

    args = parser.parse_args()

    if args.command == "init":
        wordlist_dir = Path(args.dir) if args.dir else None
        init_wordlists(wordlist_dir)
        print(f"词库已初始化到: {wordlist_dir or DEFAULT_WORDLIST_DIR}")
        print("你可以编辑这些文件来自定义过滤规则")

    elif args.command == "filter":
        level = FilterLevel(args.level)
        wordlist_dir = Path(args.wordlist_dir) if args.wordlist_dir else None
        f = ContentFilter(level=level, wordlist_dir=wordlist_dir)
        f.filter_file(
            input_path=args.input,
            output_path=args.output,
            text_field=args.field,
            save_filtered=not args.no_save_filtered
        )

    elif args.command == "test":
        level = FilterLevel(args.level)
        f = ContentFilter(level=level)
        result = f.filter_single(args.text)

        print(f"文本: {args.text}")
        print(f"结果: {'✓ 通过' if result.passed else '✗ 过滤'}")
        if not result.passed:
            print(f"类别: {result.category.value if result.category else 'N/A'}")
            print(f"匹配: {result.matched_pattern}")

    elif args.command == "analyze":
        f = ContentFilter()
        data = f._load_data(Path(args.input))

        print(f"分析 {len(data)} 条数据...")

        for item in tqdm(data):
            text = item if isinstance(item, str) else item.get(args.field, "")
            result = f.filter_single(text)
            f.stats.add(result.category)

        f._print_stats()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
