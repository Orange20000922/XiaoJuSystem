"""
启动小橘 QQ 机器人

用法：
    python run_qq_bot.py
    python src/run_qq_bot.py --config config.json

NapCat 配置：
    反向 WebSocket 地址填 ws://localhost:8080/xm
    （端口和路径在 config.json 的 qq_bot 段配置）
"""

import asyncio
import os
import sys
import traceback
from pathlib import Path

# 项目根目录 = neuro_like_system/
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 固定 CWD 到项目根，确保 config.json 中的相对路径（./data/、./checkpoints/）可用
os.chdir(project_root)

from src.logger import logger


def main():
    import argparse

    parser = argparse.ArgumentParser(description="启动小橘 QQ 机器人")
    parser.add_argument("--config", default=None, help="config.json 路径")
    args = parser.parse_args()

    # 1. 加载配置
    from configs.config_loader import AppConfig
    config = AppConfig.load(args.config)
    logger.info(f"配置加载完成: {config}")

    # 2. 初始化 Pipeline
    from src.inference_pipeline import NeuroLikePipeline
    pipeline = NeuroLikePipeline(
        small_model_path=config.small_model_checkpoint,
        personality=config.personality,
        llm_config=config.llm,
        llm_secondary_config=config.llm_secondary,
        llm_vision_config=config.llm_vision,
        memory_config=config.memory,
        emotion_prompt_config=config.emotion_prompts,
        emotion_fusion_config=config.emotion_fusion,
        emotion_state_config=config.emotion_state_config,
        attention_config=config.attention,
        device=config.device,
        time_awareness=config.agent.time_awareness,
    )
    logger.info("Pipeline 初始化完成")

    # 3. 创建 QQ 适配器（先创建，获取 output_callback）
    from src.qq_adapter import QQBotAdapter
    adapter = QQBotAdapter(config.qq_bot, image_config=config.image)

    # 4. 初始化 AgentLoop（用适配器的 callback）
    from src.agent_loop import AgentLoop
    loop = AgentLoop(
        pipeline=pipeline,
        config=config.agent,
        output_callback=adapter._make_output_callback(),
        proactive_config=config.proactive,
    )
    adapter.agent_loop = loop

    # 5. 启动 AgentLoop（daemon thread）
    loop.start()
    logger.info("AgentLoop 已启动")

    # 6. 运行 WebSocket Server（主线程 asyncio，阻塞）
    logger.info(
        f"QQ 机器人启动中... "
        f"NapCat 请连接 ws://localhost:{config.qq_bot.ws_port}{config.qq_bot.ws_path}"
    )
    try:
        asyncio.run(adapter.start())
    except KeyboardInterrupt:
        logger.info("收到 Ctrl+C，正在关闭...")
    except Exception as e:
        logger.critical(f"未捕获异常: {e}", exc_info=True)
        # 尝试通知 owner（asyncio 循环可能已停止，用新循环发送）
        if config.qq_bot.owner_qq:
            tb = traceback.format_exc()[-500:]
            try:
                asyncio.run(adapter.notify_owner(f"[严重] 进程崩溃:\n{tb}"))
            except Exception:
                pass  # 通知失败也不能阻止退出
    finally:
        loop.stop()
        logger.info("QQ 机器人已关闭")


if __name__ == "__main__":
    main()
