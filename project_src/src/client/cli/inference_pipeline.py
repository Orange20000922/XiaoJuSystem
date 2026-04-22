"""基于核心引擎封装的命令行交互入口。"""

from __future__ import annotations

from typing import Any

from src.core_engine.api import ChatRequest, DirectRuntime


def _coerce_runtime(runtime_or_target: Any) -> DirectRuntime:
    if isinstance(runtime_or_target, DirectRuntime):
        return runtime_or_target
    return DirectRuntime(runtime_or_target)


def interactive_chat(runtime_or_target: Any):
    runtime = _coerce_runtime(runtime_or_target)

    print("\n" + "=" * 50)
    print(f"欢迎与 {runtime.persona_name} 对话！")
    if runtime.llm_model:
        print(f"模型: {runtime.llm_model}")
    print("输入 'quit' 退出，'save' 保存对话历史")
    print("=" * 50 + "\n")

    while True:
        try:
            user_input = input("你: ").strip()

            if not user_input:
                continue

            if user_input.lower() == "quit":
                runtime.shutdown()
                print("再见！")
                break

            if user_input.lower() == "save":
                runtime.save_conversation("conversation_history.json")
                continue

            result = runtime.chat(
                ChatRequest(
                    text=user_input,
                    context_id="cli_default",
                    mode="private",
                    verbose=True,
                )
            )
            print(f"\n{runtime.persona_name}: {result.reply}\n")

        except KeyboardInterrupt:
            print("\n\n再见！")
            break
        except Exception as exc:
            print(f"\n错误: {exc}\n")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Neuro-Like 对话系统命令行入口")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="config.json 路径（默认：项目根目录下的 config.json）",
    )
    args = parser.parse_args()

    runtime = DirectRuntime.from_config(args.config)
    interactive_chat(runtime)


if __name__ == "__main__":
    main()
