"""测试 Mem0 初始化和嵌入 API"""
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.config_loader import AppConfig
from mem0 import Memory

cfg = AppConfig.load()

os.environ["OPENAI_API_KEY"] = cfg.memory.mem0_api_key
os.environ["OPENAI_BASE_URL"] = cfg.memory.mem0_base_url

print(f"OPENAI_BASE_URL={os.environ['OPENAI_BASE_URL']}")
print(f"OPENAI_API_KEY={os.environ['OPENAI_API_KEY'][:10]}...")

mem0_cfg = {
    "vector_store": {"provider": "qdrant", "config": {
        "collection_name": "test", "path": "./data/test_qdrant"
    }},
    "llm": {"provider": "openai", "config": {
        "model": cfg.memory.mem0_llm_model,
        "temperature": 0.1,
        "api_key": cfg.memory.mem0_api_key,
    }},
    "embedder": {"provider": "openai", "config": {
        "model": "text-embedding-3-small",
        "api_key": cfg.memory.mem0_api_key,
    }},
}

print("初始化 Mem0...")
try:
    m = Memory.from_config(mem0_cfg)
    print("Mem0 初始化成功")

    print("测试 add()...")
    m.add([{"role": "user", "content": "测试消息"}], user_id="test_user")
    print("add() 成功")

    print("测试 search()...")
    r = m.search("测试", filters={"user_id": "test_user"}, limit=3)
    print(f"search() 成功，结果数={len(r.get('results', []))}")
except Exception as e:
    print(f"失败: {type(e).__name__}: {e}")
