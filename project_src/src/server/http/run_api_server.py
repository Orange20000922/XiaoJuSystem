"""
启动 OpenAI 兼容 API 服务

用法：
    python run_api_server.py
    python run_api_server.py --config config.json
    python run_api_server.py --config config.json --host 0.0.0.0 --port 8000
"""

import os
import sys
from pathlib import Path

# 项目根目录 = agent/  (src/server/http/run_api_server.py → .parent.parent.parent.parent)
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# 固定 CWD 到项目根，确保 config.json 中的相对路径可用
os.chdir(project_root)

# 阻止 HuggingFace 联网
os.environ["HF_HUB_OFFLINE"] = "1"

from src.logger import logger


def main():
    import argparse

    parser = argparse.ArgumentParser(description="启动 OpenAI 兼容 API 服务")
    parser.add_argument("--config", default=None, help="config.json 路径")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址（默认 0.0.0.0）")
    parser.add_argument("--port", type=int, default=8000, help="监听端口（默认 8000）")
    parser.add_argument("--model-name", default=None, help="对外展示的模型名（默认用 persona 名）")
    parser.add_argument("--ssl-certfile", default=None, help="SSL 证书文件路径")
    parser.add_argument("--ssl-keyfile", default=None, help="SSL 私钥文件路径")
    args = parser.parse_args()

    # 1. 加载配置并初始化核心引擎封装层
    from configs.config_loader import AppConfig
    from src.core_engine.api import DirectRuntime
    from src.server.http.openai_adapter import create_app

    logger.info("正在加载配置...")
    app_cfg = AppConfig.load(args.config)
    runtime = DirectRuntime.from_config(args.config)
    persona_name = runtime.persona_name
    model_name = args.model_name or persona_name

    logger.info(
        f"Runtime 初始化完成: persona={persona_name}, "
        f"model={runtime.llm_model}"
    )

    # 2. 创建 FastAPI 应用（传入安全配置）
    security_config = app_cfg.security
    app = create_app(runtime, model_name=model_name, security_config=security_config)

    # 3. SSL 参数（命令行优先，其次配置文件）
    ssl_certfile = args.ssl_certfile or security_config.ssl_certfile
    ssl_keyfile = args.ssl_keyfile or security_config.ssl_keyfile

    # 4. 启动日志
    import uvicorn

    protocol = "https" if ssl_certfile else "http"
    sec_status = []
    if security_config.enabled:
        sec_status.append(f"认证: ON ({len(security_config.api_keys)} keys)")
    else:
        sec_status.append("认证: OFF")
    sec_status.append(f"IP限流: {security_config.rate_limit_per_minute}/min")
    sec_status.append(f"并发上限: {security_config.max_concurrent_chat}")
    if ssl_certfile:
        sec_status.append("HTTPS: ON")

    logger.info(
        f"API 服务启动中: {protocol}://{args.host}:{args.port}\n"
        f"  POST /v1/chat/completions  — 对话\n"
        f"  GET  /v1/models            — 模型列表\n"
        f"  GET  /health               — 健康检查\n"
        f"  模型名: {model_name}\n"
        f"  安全: {' | '.join(sec_status)}"
    )

    # 5. 启动服务
    uvicorn_kwargs = dict(host=args.host, port=args.port, log_level="info")
    if ssl_certfile and ssl_keyfile:
        uvicorn_kwargs["ssl_certfile"] = ssl_certfile
        uvicorn_kwargs["ssl_keyfile"] = ssl_keyfile

    try:
        uvicorn.run(app, **uvicorn_kwargs)
    except KeyboardInterrupt:
        logger.info("收到 Ctrl+C，正在关闭...")
    finally:
        runtime.shutdown()
        logger.info("API 服务已关闭")


if __name__ == "__main__":
    main()
