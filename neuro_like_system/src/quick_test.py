"""
快速测试脚本（无 BERT 模式）

用法：
  # 单句测试
  python src/quick_test.py --msg "你好"

  # 多轮对话（用分隔符）
  python src/quick_test.py --msg "你好|||最近在干嘛|||聊聊AI吧"

  # 只测试 API 连通性
  python src/quick_test.py --ping

  # 记忆系统测试（跨上下文窗口，显示 Token 计数和向量 DB 状态）
  python src/quick_test.py --memory-test

  # 记忆系统测试 + 自定义压缩阈值（token 数，调小方便测试）
  python src/quick_test.py --memory-test --compress-at 300
"""

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.logger import logger
from configs.config_loader import AppConfig, load_emotion_prompt_config
from configs.model_config import MemoryConfig
from src.inference_pipeline import LLMClient, NeuroLikePipeline
from src.memory_manager import HierarchicalMemoryManager, count_tokens


# ── 轻量 Pipeline（无 BERT，无记忆系统）────────────────────────────────────

class LightPipeline:
    """无 BERT、无记忆系统，用于基础 API + Prompt 测试"""

    def __init__(self, config: AppConfig):
        self.personality = config.personality
        self.llm_client = LLMClient(config.llm)
        self.messages: List[dict] = []

    def build_system_prompt(self) -> str:
        p = self.personality
        persona = p.description.strip() if p.description.strip() else \
            f"[标签]{','.join(p.traits)} [风格]口语"
        return f"你是{p.name}。\n<persona>{persona}</persona>\n根据人格和对话历史自然回复，保持一致性。"

    def chat(self, user_input: str) -> str:
        response = self.llm_client.generate(
            system_prompt=self.build_system_prompt(),
            user_input=user_input,
            history=self.messages,
        )
        self.messages.append({"role": "user", "content": user_input})
        self.messages.append({"role": "assistant", "content": response})
        return response


# ── 记忆 Pipeline（无 BERT，含 HierarchicalMemoryManager）─────────────────

class MemoryPipeline:
    """无 BERT，接入完整分级记忆系统，用于跨窗口记忆测试"""

    def __init__(self, config: AppConfig, compress_at_tokens: Optional[int] = None):
        self.personality = config.personality
        self.llm_client = LLMClient(config.llm)

        # 测试时可以把压缩阈值调小，快速触发 L1→L2 压缩
        mem_cfg = config.memory
        if compress_at_tokens:
            # 直接覆盖 context_window_tokens，threshold 保持 0.75
            # 使得 compress_at_tokens ≈ context_window * 0.75
            mem_cfg.context_window_tokens = int(compress_at_tokens / 0.75)
            logger.info(
                f"测试模式：压缩触发阈值设为 {compress_at_tokens} tokens "
                f"(context_window={mem_cfg.context_window_tokens})"
            )

        self.memory = HierarchicalMemoryManager(
            config=mem_cfg,
            llm_client=self.llm_client,
        )
        logger.info(
            f"记忆系统初始化完成 "
            f"compress_trigger={self.memory.config.compression_trigger_tokens} tokens"
        )

    def build_system_prompt(self, recalled: str = "") -> str:
        p = self.personality
        persona = p.description.strip() if p.description.strip() else \
            f"[标签]{','.join(p.traits)} [风格]口语"
        prompt = f"你是{p.name}。\n<persona>{persona}</persona>"
        if recalled:
            prompt += f"\n{recalled}"
        prompt += "\n根据人格和对话历史自然回复，保持一致性。"
        return prompt

    def chat(self, user_input: str) -> str:
        recalled = self.memory.get_system_context(query=user_input)
        history = self.memory.get_messages_history()

        response = self.llm_client.generate(
            system_prompt=self.build_system_prompt(recalled),
            user_input=user_input,
            history=history,
        )

        from src.inference_pipeline import ConversationTurn
        turn = ConversationTurn(
            user_input=user_input,
            emotion="neutral",
            intensity=0.5,
            behavior="respond_positive",
            tone="calm",
            response=response,
        )
        self.memory.add(turn)
        return response

    def print_memory_status(self):
        """打印当前记忆系统状态"""
        m = self.memory
        logger.info(f"  [L1] {m.l1_usage}")
        logger.info(f"  [L1] working_memory 轮次={len(m.working_memory)}")

        # 查询 Mem0 L2（当前 session）
        try:
            l2 = m.mem0.get_all(user_id=m.user_id, run_id=m.session_id)
            l2_count = len(l2.get("results", [])) if l2 else 0
            logger.info(f"  [L2] Mem0 session 记录数={l2_count} session={m.session_id}")
        except Exception as e:
            logger.warning(f"  [L2] Mem0 查询失败: {e}")

        # 查询 Mem0 L3（user 级）
        try:
            l3 = m.mem0.get_all(user_id=m.user_id)
            l3_count = len(l3.get("results", [])) if l3 else 0
            logger.info(f"  [L3] Mem0 user 记录数={l3_count} user={m.user_id}")
        except Exception as e:
            logger.warning(f"  [L3] Mem0 查询失败: {e}")


# ── 通用工具函数 ────────────────────────────────────────────────────────────

def ping_api(config: AppConfig) -> bool:
    logger.info(f"测试 API 连通性: {config.llm.base_url} model={config.llm.model}")
    try:
        client = LLMClient(config.llm)
        t0 = time.time()
        resp = client.generate(
            system_prompt="你是测试助手。",
            user_input="回复'OK'即可。",
            max_tokens=10,
            temperature=0.0,
        )
        logger.info(f"API 连通正常 响应={repr(resp)} 耗时={time.time() - t0:.2f}s")
        return True
    except Exception as e:
        logger.error(f"API 连通失败: {e}")
        return False


def run_bert_test(pipeline: NeuroLikePipeline, messages: List[str], ep_cfg):
    """完整 BERT+LLM pipeline 测试：显示情绪分析 + 注入指令 + 回复"""
    SEP = "─" * 60
    logger.info("=" * 60)
    logger.info(f"BERT+LLM 联合测试，共 {len(messages)} 轮")
    logger.info("=" * 60)

    for i, msg in enumerate(messages, 1):
        print(SEP)
        print(f"用户: {msg}")

        # 显示 BERT 分析 + 指令
        if pipeline.small_model:
            analysis = pipeline.analyze_emotion_behavior(msg)
            e = analysis["emotion"]["primary"]
            p = analysis["emotion"]["primary_prob"]
            rel = ep_cfg.emotion_reliability.get(e, 0.7)
            eff = p * rel
            directive = pipeline._build_emotion_directives(analysis)
            print(f"[BERT] {e}  prob={p:.2f}  eff={eff:.3f}  →  {directive or '(无指令)'}")

        t0 = time.time()
        result = pipeline.chat(msg)
        elapsed = time.time() - t0

        print(f"Neuro: {result['response']}")
        print(f"[{elapsed:.1f}s]")

    print(SEP)


def run_conversation(pipeline: LightPipeline, messages: List[str]):
    logger.info("=" * 50)
    logger.info(f"开始对话测试，共 {len(messages)} 轮")
    logger.info("=" * 50)
    for i, msg in enumerate(messages, 1):
        logger.info(f"[第{i}轮] 用户: {msg}")
        t0 = time.time()
        try:
            response = pipeline.chat(msg)
            logger.info(f"[第{i}轮] {pipeline.personality.name}: {response}")
            logger.info(f"[第{i}轮] 耗时={time.time() - t0:.2f}s")
        except Exception as e:
            logger.error(f"[第{i}轮] 生成失败: {e}")
            break
        logger.info("-" * 40)
    logger.info("对话测试完成")


def run_memory_test(pipeline: MemoryPipeline, messages: List[str]):
    """
    记忆系统测试：每轮显示 Token 计数和向量 DB 状态，
    观察 L1→L2 压缩触发和跨窗口连贯性。
    """
    logger.info("=" * 60)
    logger.info(f"开始记忆系统测试，共 {len(messages)} 轮")
    logger.info(f"压缩触发阈值: {pipeline.memory.config.compression_trigger_tokens} tokens")
    logger.info("=" * 60)

    for i, msg in enumerate(messages, 1):
        logger.info(f"\n{'─' * 50}")
        logger.info(f"[第{i}轮] 用户: {msg}")

        # 记录压缩前状态
        l1_before = pipeline.memory._l1_tokens
        turns_before = len(pipeline.memory.working_memory)

        t0 = time.time()
        try:
            # 详细 debug：记录每个子步骤
            logger.debug(f"[第{i}轮] 开始 get_system_context...")
            recalled = pipeline.memory.get_system_context(query=msg)
            logger.debug(f"[第{i}轮] get_system_context 完成，长度={len(recalled)} chars")

            logger.debug(f"[第{i}轮] 开始 get_messages_history...")
            history = pipeline.memory.get_messages_history()
            logger.debug(f"[第{i}轮] get_messages_history 完成，轮次={len(history)//2}")

            logger.debug(f"[第{i}轮] 开始构建 system_prompt...")
            system_prompt = pipeline.build_system_prompt(recalled)
            prompt_tokens = count_tokens(system_prompt)
            msg_tokens = count_tokens(msg)
            history_tokens = sum(count_tokens(m["content"]) for m in history)
            logger.debug(
                f"[第{i}轮] prompt tokens: system={prompt_tokens} "
                f"history={history_tokens} user={msg_tokens} "
                f"合计={prompt_tokens + history_tokens + msg_tokens}"
            )

            logger.debug(f"[第{i}轮] 开始 LLM 调用...")
            response = pipeline.llm_client.generate(
                system_prompt=system_prompt,
                user_input=msg,
                history=history,
            )
            elapsed = time.time() - t0
            logger.debug(f"[第{i}轮] LLM 调用完成，耗时={elapsed:.2f}s")

            # 写入记忆
            from src.inference_pipeline import ConversationTurn
            turn = ConversationTurn(
                user_input=msg, emotion="neutral", intensity=0.5,
                behavior="respond_positive", tone="calm", response=response,
            )
            logger.debug(f"[第{i}轮] 开始写入 L1 记忆...")
            pipeline.memory.add(turn)
            logger.debug(f"[第{i}轮] L1 写入完成")

            logger.info(f"[第{i}轮] {pipeline.personality.name}: {response}")
            logger.info(f"[第{i}轮] 耗时={elapsed:.2f}s")

            resp_tokens = count_tokens(response)
            logger.info(
                f"[第{i}轮] Token消耗: system={prompt_tokens} "
                f"history={history_tokens} user={msg_tokens} output={resp_tokens}"
            )

            l1_after = pipeline.memory._l1_tokens
            turns_after = len(pipeline.memory.working_memory)
            if turns_after < turns_before + 1:
                compressed = turns_before + 1 - turns_after
                logger.info(
                    f"[第{i}轮] *** L1→L2 压缩触发 *** "
                    f"压缩了 {compressed} 轮，L1 tokens: {l1_before} → {l1_after}"
                )

            logger.info(f"[第{i}轮] 记忆状态:")
            pipeline.print_memory_status()

        except Exception as e:
            import traceback
            logger.error(f"[第{i}轮] 生成失败: {type(e).__name__}: {e}")
            logger.error(f"[第{i}轮] 完整堆栈:\n{traceback.format_exc()}")
            break

    logger.info("\n" + "=" * 60)
    logger.info("记忆系统测试完成，触发 close_session 写入 L3...")
    pipeline.memory.close_session()
    logger.info("close_session 完成，最终 L3 状态:")
    pipeline.print_memory_status()


def run_cross_session_test(pipeline: MemoryPipeline, config: AppConfig,
                           messages: List[str], args):
    """
    跨 session 记忆测试：
      Session 1: 对话若干轮 → 触发 L1→L2 压缩 → close_session 写入 L3
      Session 2: 新建 pipeline（新 session_id）→ 验证 L3 召回
    """
    # ── Session 1 ──────────────────────────────────────────────────────
    session1_msgs = messages[:6] if len(messages) >= 6 else messages
    logger.info("=" * 60)
    logger.info("Session 1: 建立对话历史并写入长期记忆")
    logger.info("=" * 60)
    run_memory_test(pipeline, session1_msgs)

    logger.info("\n" + "=" * 60)
    logger.info("Session 1 结束，等待 L3 写入...")
    logger.info("=" * 60)

    # 释放 Session 1 的 Qdrant 连接（Windows 上 .lock 文件占用）
    if hasattr(pipeline.memory, 'mem0'):
        mem0 = pipeline.memory.mem0
        for attr in ('vector_store', '_telemetry_vector_store'):
            vs = getattr(mem0, attr, None)
            if vs and hasattr(vs, 'client'):
                vs.client.close()
    del pipeline
    import gc; gc.collect()

    # ── Session 2 ──────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("Session 2: 新建会话，测试跨 session L3 记忆召回")
    logger.info("=" * 60)

    pipeline2 = MemoryPipeline(config, compress_at_tokens=args.compress_at)
    if args.protected_turns is not None:
        pipeline2.memory.PROTECTED_TURNS = args.protected_turns

    # 用和 Session 1 相关的问题来验证是否能召回
    recall_messages = [
        "你还记得我之前跟你聊过什么吗",
        "我之前提到的那个AI系统，你还有印象吗",
    ]

    for i, msg in enumerate(recall_messages, 1):
        logger.info(f"\n{'─' * 50}")
        logger.info(f"[Session2 第{i}轮] 用户: {msg}")

        try:
            # 先看召回了什么
            recalled = pipeline2.memory.get_system_context(query=msg)
            logger.info(f"[Session2 第{i}轮] L2/L3 召回内容: {recalled if recalled else '(无)'}")

            response = pipeline2.chat(msg)
            logger.info(f"[Session2 第{i}轮] Neuro: {response}")
            logger.info(f"[Session2 第{i}轮] 记忆状态:")
            pipeline2.print_memory_status()
        except Exception as e:
            import traceback
            logger.error(f"[Session2 第{i}轮] 失败: {e}\n{traceback.format_exc()}")
            break

    logger.info("\n" + "=" * 60)
    logger.info("跨 session 测试完成")


# ── 入口 ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Neuro 快速测试（无 BERT）")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--ping", action="store_true", help="只测试 API 连通性")
    parser.add_argument("--msg", type=str, default=None,
                        help="对话内容，多轮用 ||| 分隔")
    parser.add_argument("--memory-test", action="store_true",
                        help="记忆系统测试（跨窗口 + Token 计数 + 向量 DB 状态）")
    parser.add_argument("--compress-at", type=int, default=None,
                        help="记忆测试时的 L1 压缩触发 token 数（默认使用 config.json）")
    parser.add_argument("--protected-turns", type=int, default=None,
                        help="记忆测试时的保护区轮次数（默认12，测试建议2-3）")
    parser.add_argument("--cross-session", action="store_true",
                        help="跨 session 记忆召回测试（先跑第一轮对话写入 L3，再跑第二轮验证召回）")
    parser.add_argument("--bert-test", action="store_true",
                        help="完整 BERT+LLM 联合测试，显示情绪分析和注入指令")
    args = parser.parse_args()

    try:
        config = AppConfig.load(args.config)
        logger.info(f"配置加载成功: provider={config.llm.provider.value} model={config.llm.model}")
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    if args.ping:
        sys.exit(0 if ping_api(config) else 1)

    # ── BERT+LLM 联合测试模式 ───────────────────────────────────────────────
    if args.bert_test:
        import os
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        test_messages = (
            [m.strip() for m in args.msg.split("|||")] if args.msg else [
                "哈哈哈哈笑死我了",
                "好可怕，我一个人在家",
                "这也太离谱了吧！！",
                "呜呜呜好感动",
                "你们觉得AI有没有意识？",
                "气死我了，今天被坑了",
                "大家好",
                "666前方高能！",
            ]
        )
        import json
        with open(Path(args.config) if args.config else
                  Path(__file__).parent.parent / "config.json", encoding="utf-8") as f:
            raw_cfg = json.load(f)
        ep_cfg = load_emotion_prompt_config(raw_cfg)

        bert_pipeline = NeuroLikePipeline.from_config(args.config)
        run_bert_test(bert_pipeline, test_messages, ep_cfg)
        return

    if not ping_api(config):
        logger.error("API 不可用，终止测试")
        sys.exit(1)

    # ── 记忆系统测试模式 ──────────────────────────────────────────────────
    if args.memory_test:
        # 默认测试对话：足够长以触发压缩
        if args.msg:
            test_messages = [m.strip() for m in args.msg.split("|||")]
        else:
            test_messages = [
                "你好，我们开始聊吧",
                "我最近在做一个AI对话系统，用的是BERT做情绪分类",
                "对，就是用来判断用户情绪和行为的，然后配合LLM生成回复",
                "现在架构大概是BERT分类器加分级记忆系统加LLM API",
                "记忆系统分四层，L1是工作记忆，L2是会话摘要，L3是跨会话持久化",
                "L4是知识库，用来存Agent执行结果和用户画像",
                "你觉得这个架构有什么问题吗",
                "对了，我还打算接QQ机器人，群聊场景需要注意力判断",
                "就是判断这条消息要不要回复，不能每条都回",
                "现在基本跑通了，正在测试记忆系统的跨窗口效果",
            ]
            logger.info(f"使用默认测试对话，共 {len(test_messages)} 轮")

        pipeline = MemoryPipeline(config, compress_at_tokens=args.compress_at)

        # 测试时覆盖保护区大小
        if args.protected_turns is not None:
            pipeline.memory.PROTECTED_TURNS = args.protected_turns
            logger.info(f"测试模式：保护区设为 {args.protected_turns} 轮")

        if args.cross_session:
            run_cross_session_test(pipeline, config, test_messages, args)
        else:
            run_memory_test(pipeline, test_messages)

    # ── 普通对话模式 ──────────────────────────────────────────────────────
    else:
        if not args.msg:
            test_messages = [
                "你好，我们来聊聊吧",
                "你平时都对什么话题感兴趣？",
                "我最近在做一个AI对话系统，有点复杂",
            ]
            logger.info("未指定 --msg，使用默认测试对话")
        else:
            test_messages = [m.strip() for m in args.msg.split("|||")]

        pipeline = LightPipeline(config)
        run_conversation(pipeline, test_messages)


if __name__ == "__main__":
    main()
