"""
启动小橘 QQ 机器人

用法：
    # 单 persona 模式（向后兼容）
    python run_qq_bot.py
    python run_qq_bot.py --config config.json

    # 多 persona 模式
    python run_qq_bot.py --scheduler scheduler_config.json

NapCat 配置：
    反向 WebSocket 地址填 ws://localhost:8080/xm
    （端口和路径在 config.json 的 qq_bot 段配置）
"""

import asyncio
import os
import sys
import traceback
from pathlib import Path

# 项目根目录 (src/adapters/run_qq_bot.py → .parent.parent.parent)
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 固定 CWD 到项目根，确保 config.json 中的相对路径（./data/、./checkpoints/）可用
os.chdir(project_root)

from src.logger import logger


def _start_single_persona(args):
    """单 persona 模式（通过 PersonaScheduler 统一管理）"""
    from configs.config_loader import AppConfig
    from src.core.shared_infra import SharedInfra
    from src.core.scheduler import PersonaScheduler
    from src.adapters.qq_adapter import QQBotAdapter
    from configs.model_config import SchedulerConfig

    # 1. 加载配置
    config = AppConfig.load(args.config)
    logger.info(f"配置加载完成: {config}")

    # 2. 创建共享基础设施（单例）
    infra = SharedInfra.from_app_config(config)
    logger.info("SharedInfra 初始化完成")

    # 3. 创建调度器
    scheduler = PersonaScheduler(infra, SchedulerConfig())

    # 4. 创建 QQ 适配器
    adapter = QQBotAdapter(config.qq_bot, image_config=config.image)

    # 5. 注册为默认 persona（catch-all，处理所有 context）
    scheduler.register_default(
        app_config=config,
        callback_factory=adapter._make_output_callback_for_loop,
        config_source=str(args.config or "config.json"),
    )

    # 6. 接线
    adapter.scheduler = scheduler
    scheduler.set_alert_callback(adapter.notify_owner_sync)

    return config, scheduler, adapter


def _start_multi_persona(args):
    """多 persona 模式（从 scheduler_config.json 加载）"""
    from configs.config_loader import AppConfig, load_scheduler_config
    from src.core.shared_infra import SharedInfra
    from src.core.scheduler import PersonaScheduler
    from src.adapters.qq_adapter import QQBotAdapter

    # 1. 加载调度器配置
    sched_cfg, entries = load_scheduler_config(args.scheduler)
    logger.info(
        f"调度器配置加载完成: {len(entries)} 个 persona, "
        f"max_concurrent_llm={sched_cfg.max_concurrent_llm}"
    )

    # 2. 用第一个 config 创建共享基础设施（BERT + LLM 客户端池）
    primary_config = AppConfig.load(entries[0]["config"])
    infra = SharedInfra.from_app_config(primary_config)
    logger.info("SharedInfra 初始化完成")

    # 3. 创建调度器
    scheduler = PersonaScheduler(infra, sched_cfg)

    # 4. 创建 QQ 适配器（用主 config 的 qq_bot 配置）
    adapter = QQBotAdapter(primary_config.qq_bot, image_config=primary_config.image)

    # 5. 逐个注册 persona
    for entry in entries:
        app_cfg = AppConfig.load(entry["config"])
        contexts = entry.get("contexts", [])
        exact = {c for c in contexts if "*" not in c and "?" not in c}
        patterns = [c for c in contexts if "*" in c or "?" in c]

        scheduler.register(
            app_config=app_cfg,
            context_ids=exact or None,
            context_patterns=patterns or None,
            callback_factory=adapter._make_output_callback_for_loop,
            config_source=entry["config"],
        )

    # 6. 接线
    adapter.scheduler = scheduler
    scheduler.set_alert_callback(adapter.notify_owner_sync)

    return primary_config, scheduler, adapter


def main():
    import argparse

    parser = argparse.ArgumentParser(description="启动小橘 QQ 机器人")
    parser.add_argument("--config", default=None, help="config.json 路径（单 persona 模式）")
    parser.add_argument("--scheduler", default=None, help="scheduler_config.json 路径（多 persona 模式）")
    args = parser.parse_args()

    # 初始化
    if args.scheduler:
        config, scheduler, adapter = _start_multi_persona(args)
    else:
        config, scheduler, adapter = _start_single_persona(args)

    # 启动所有 AgentLoop + 健康监控
    scheduler.start_all()

    # 运行 WebSocket Server（主线程 asyncio，阻塞）
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
        if config.qq_bot.owner_qq:
            tb = traceback.format_exc()[-500:]
            try:
                asyncio.run(adapter.notify_owner(f"[严重] 进程崩溃:\n{tb}"))
            except Exception:
                pass
    finally:
        scheduler.stop_all()
        logger.info("QQ 机器人已关闭")


if __name__ == "__main__":
    main()
