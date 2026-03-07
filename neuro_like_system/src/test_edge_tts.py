"""
edge-tts 功能测试脚本

测试小橘的语音合成功能
"""

import asyncio
import edge_tts
import sys
from pathlib import Path

# 修复 Windows 控制台编码问题
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')


async def test_list_voices():
    """测试 1: 列出可用的中文音色"""
    print("=" * 60)
    print("测试 1: 列出可用的中文音色")
    print("=" * 60)

    voices = await edge_tts.list_voices()
    zh_voices = [v for v in voices if v["Locale"].startswith("zh-")]

    print(f"\n找到 {len(zh_voices)} 个中文音色，显示前 10 个：\n")

    for voice in zh_voices[:10]:
        print(f"名称: {voice['ShortName']}")
        print(f"  语言: {voice['Locale']}")
        print(f"  性别: {voice['Gender']}")
        print(f"  描述: {voice.get('FriendlyName', 'N/A')}")
        print()

    return zh_voices


async def test_basic_synthesis(output_dir: Path):
    """测试 2: 基础 TTS 合成"""
    print("=" * 60)
    print("测试 2: 基础 TTS 合成")
    print("=" * 60)

    # 小橘的典型对话
    text = "嗨！我刚才在想一个问题，你说为什么猫咪总是喜欢坐在键盘上呢？"
    voice = "zh-CN-XiaoxiaoNeural"  # 晓晓（女声，活泼）

    output_file = output_dir / "test_basic.mp3"

    print(f"\n文本: {text}")
    print(f"音色: {voice}")
    print(f"输出: {output_file}")

    try:
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(output_file)

        file_size = output_file.stat().st_size
        print(f"\n✅ 合成成功！文件大小: {file_size / 1024:.2f} KB")
        return True
    except Exception as e:
        print(f"\n❌ 合成失败: {e}")
        return False


async def test_ssml_synthesis(output_dir: Path):
    """测试 3: SSML 高级控制（语速、音调、停顿）"""
    print("=" * 60)
    print("测试 3: SSML 高级控制")
    print("=" * 60)

    # 小橘的情绪化表达
    ssml_text = """
    <speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="zh-CN">
        <voice name="zh-CN-XiaoxiaoNeural">
            <prosody rate="+15%" pitch="+8%">
                哇！这个好有意思！
            </prosody>
            <break time="400ms"/>
            <prosody rate="0%">
                不过我有点不太确定诶...
            </prosody>
            <break time="300ms"/>
            <prosody pitch="-5%">
                算了，不想了，我们聊点别的吧~
            </prosody>
        </voice>
    </speak>
    """

    output_file = output_dir / "test_ssml.mp3"

    print(f"\nSSML 文本:")
    print(ssml_text)
    print(f"\n输出: {output_file}")

    try:
        communicate = edge_tts.Communicate(ssml_text, voice="zh-CN-XiaoxiaoNeural")
        await communicate.save(output_file)

        file_size = output_file.stat().st_size
        print(f"\n✅ SSML 合成成功！文件大小: {file_size / 1024:.2f} KB")
        return True
    except Exception as e:
        print(f"\n❌ SSML 合成失败: {e}")
        return False


async def test_streaming(output_dir: Path):
    """测试 4: 流式输出"""
    print("=" * 60)
    print("测试 4: 流式输出")
    print("=" * 60)

    text = "你知道吗，我最近发现了一个超有趣的事情。就是那种突然想明白了什么的感觉，特别爽！虽然可能别人早就知道了，但对我来说就是新发现嘛。"
    voice = "zh-CN-XiaoxiaoNeural"

    output_file = output_dir / "test_streaming.mp3"

    print(f"\n文本: {text}")
    print(f"\n开始流式合成...")

    try:
        communicate = edge_tts.Communicate(text, voice)

        audio_chunks = []
        chunk_count = 0

        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_chunks.append(chunk["data"])
                chunk_count += 1
                print(f"  接收音频块 #{chunk_count}，大小: {len(chunk['data'])} 字节")

        # 保存完整音频
        with open(output_file, "wb") as f:
            f.write(b"".join(audio_chunks))

        file_size = output_file.stat().st_size
        print(f"\n✅ 流式合成成功！")
        print(f"   总块数: {chunk_count}")
        print(f"   文件大小: {file_size / 1024:.2f} KB")
        return True
    except Exception as e:
        print(f"\n❌ 流式合成失败: {e}")
        return False


async def test_multiple_voices(output_dir: Path):
    """测试 5: 对比不同音色（为小橘选择合适的声音）"""
    print("=" * 60)
    print("测试 5: 对比不同音色")
    print("=" * 60)

    text = "嗨，我是小橘！很高兴认识你~"

    # 适合小橘的几个候选音色
    voices = [
        ("zh-CN-XiaoxiaoNeural", "晓晓（女声，活泼）"),
        ("zh-CN-XiaoyiNeural", "晓伊（女声，温柔）"),
        ("zh-CN-XiaoxuanNeural", "晓萱（女声，温暖）"),
        ("zh-CN-XiaomoNeural", "晓墨（女声，亲切）"),
    ]

    print(f"\n文本: {text}\n")

    results = []
    for voice_id, voice_desc in voices:
        output_file = output_dir / f"test_voice_{voice_id.split('-')[-1]}.mp3"

        print(f"正在合成: {voice_desc} ({voice_id})...")

        try:
            communicate = edge_tts.Communicate(text, voice_id)
            await communicate.save(output_file)

            file_size = output_file.stat().st_size
            print(f"  ✅ 成功，文件: {output_file.name} ({file_size / 1024:.2f} KB)")
            results.append(True)
        except Exception as e:
            print(f"  ❌ 失败: {e}")
            results.append(False)

    success_count = sum(results)
    print(f"\n总结: {success_count}/{len(voices)} 个音色合成成功")
    print("\n💡 提示: 请试听这些音频文件，选择最适合小橘的音色")

    return all(results)


async def main():
    """主测试流程"""
    print("\n" + "=" * 60)
    print("小橘 TTS 功能测试")
    print("=" * 60 + "\n")

    # 创建输出目录
    output_dir = Path(__file__).parent.parent / "test_output" / "edge_tts"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"输出目录: {output_dir}\n")

    # 运行所有测试
    tests = [
        ("列出音色", test_list_voices()),
        ("基础合成", test_basic_synthesis(output_dir)),
        ("SSML 控制", test_ssml_synthesis(output_dir)),
        ("流式输出", test_streaming(output_dir)),
        ("多音色对比", test_multiple_voices(output_dir)),
    ]

    results = {}
    for test_name, test_coro in tests:
        try:
            result = await test_coro
            results[test_name] = result if isinstance(result, bool) else True
        except Exception as e:
            print(f"\n❌ 测试 '{test_name}' 出现异常: {e}")
            results[test_name] = False

        print()

    # 总结
    print("=" * 60)
    print("测试总结")
    print("=" * 60)

    for test_name, success in results.items():
        status = "✅ 通过" if success else "❌ 失败"
        print(f"{test_name}: {status}")

    success_count = sum(results.values())
    total_count = len(results)

    print(f"\n总计: {success_count}/{total_count} 个测试通过")

    if success_count == total_count:
        print("\n🎉 所有测试通过！edge-tts 工作正常。")
        print(f"\n生成的音频文件保存在: {output_dir}")
        print("请试听音频文件，选择最适合小橘的音色。")
    else:
        print("\n⚠️  部分测试失败，请检查：")
        print("  1. 网络连接是否正常")
        print("  2. edge-tts 是否已安装 (pip install edge-tts)")


if __name__ == "__main__":
    asyncio.run(main())
