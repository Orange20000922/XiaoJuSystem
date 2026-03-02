"""
手动写入 L4 长期记忆的工具脚本

用法：
    python add_memory.py "主人的名字是XXX"
    python add_memory.py "主人喜欢吃辣，不喜欢甜食"
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from configs.config_loader import AppConfig
from src.memory_manager import HierarchicalMemoryManager
from src.inference_pipeline import LLMClient


def add_memory(memory_text: str):
    """添加一条长期记忆到 L4"""
    # 加载配置
    config = AppConfig.load("config.json")

    # 初始化 LLM 客户端（Mem0 需要）
    llm_client = LLMClient(config.llm)

    # 初始化记忆管理器
    memory_manager = HierarchicalMemoryManager(
        config=config.memory,
        llm_client=llm_client,
        user_id=config.memory.user_id
    )

    # 写入记忆
    print(f"正在写入记忆: {memory_text}")

    result = memory_manager.mem0.add(
        messages=[{"role": "user", "content": memory_text}],
        user_id=config.memory.user_id
    )

    print(f"✓ 记忆已写入 L4")
    print(f"  Memory ID: {result}")

    # 验证写入
    print("\n验证召回...")
    recalled = memory_manager.mem0.search(
        query=memory_text[:20],  # 用前20个字符搜索
        user_id=config.memory.user_id,
        limit=3
    )

    if recalled:
        print(f"✓ 成功召回 {len(recalled)} 条相关记忆:")
        for i, mem in enumerate(recalled, 1):
            if isinstance(mem, str):
                print(f"  {i}. {mem}")
            elif isinstance(mem, dict):
                print(f"  {i}. {mem.get('memory', mem)}")
    else:
        print("⚠ 未能召回记忆（可能需要等待索引更新）")


def list_memories():
    """列出所有 L4 记忆"""
    config = AppConfig.load("config.json")
    llm_client = LLMClient(config.llm)
    memory_manager = HierarchicalMemoryManager(
        config=config.memory,
        llm_client=llm_client,
        user_id=config.memory.user_id
    )

    print("正在获取所有记忆...")
    memories = memory_manager.mem0.get_all(user_id=config.memory.user_id)

    if memories:
        print(f"\n共有 {len(memories)} 条记忆:\n")
        for i, mem in enumerate(memories, 1):
            if isinstance(mem, dict):
                print(f"{i}. {mem.get('memory', mem)}")
            else:
                print(f"{i}. {mem}")
    else:
        print("暂无记忆")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法:")
        print("  添加记忆: python add_memory.py \"记忆内容\"")
        print("  列出记忆: python add_memory.py --list")
        sys.exit(1)

    if sys.argv[1] == "--list":
        list_memories()
    else:
        memory_text = " ".join(sys.argv[1:])
        add_memory(memory_text)
