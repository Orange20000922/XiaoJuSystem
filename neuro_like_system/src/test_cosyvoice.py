"""
CosyVoice TTS 测试脚本

测试 zero-shot 音色克隆合成，输出 WAV 文件到 test_output/cosyvoice/。

用法：
    python src/test_cosyvoice.py
    python src/test_cosyvoice.py --ref path/to/ref.wav --text "你好呀！"
"""

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

# 固定 CWD 到项目根
project_root = Path(__file__).parent.parent
os.chdir(project_root)
sys.path.insert(0, str(project_root))

# Windows 控制台编码
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from src.logger import logger


# ── 直接调用 CosyVoice2，不走 AudioPipeline ────────────────────────────────

COSYVOICE_REPO = r"D:\Users\21405\source\repos\MyNeuroLikeSystem\CosyVoice2\CosyVoice"
MODEL_DIR      = r"D:\Users\21405\source\repos\MyNeuroLikeSystem\CosyVoice2\CosyVoice\pretrained_models\CosyVoice2-0.5B"
DEFAULT_REF    = r"D:\Users\21405\source\repos\MyNeuroLikeSystem\CosyVoice2\CosyVoice\data\audio_refs\test2.wav"

OUTPUT_DIR = project_root / "test_output" / "cosyvoice"

TEST_TEXTS = [
    "你好，我是小橘，很高兴认识你！",
    "哇，这个好有意思，我之前完全没想到呢。",
    "嗯……我不太确定诶，你再说清楚一点？",
    "哈哈，你说的这个我完全同意，就是这个感觉！",
]


def _inject_cosyvoice_path():
    """把 CosyVoice 仓库路径注入 sys.path"""
    repo = Path(COSYVOICE_REPO)
    for p in [str(repo), str(repo / "third_party" / "Matcha-TTS")]:
        if p not in sys.path:
            sys.path.insert(0, p)


def test_import():
    """测试 1：确认 CosyVoice2 可以 import"""
    print("=" * 60)
    print("测试 1：import CosyVoice2")
    print("=" * 60)

    _inject_cosyvoice_path()

    try:
        from cosyvoice.cli.cosyvoice import CosyVoice2
        print(f"✅ CosyVoice2 import 成功")
        return True
    except ImportError as e:
        print(f"❌ import 失败: {e}")
        return False


def test_load_model():
    """测试 2：加载模型"""
    print("\n" + "=" * 60)
    print("测试 2：加载 CosyVoice2-0.5B 模型")
    print("=" * 60)

    _inject_cosyvoice_path()
    from cosyvoice.cli.cosyvoice import CosyVoice2

    model_path = Path(MODEL_DIR)
    if not model_path.exists():
        print(f"❌ 模型目录不存在: {model_path}")
        return None

    print(f"模型目录: {model_path}")
    print("加载中（首次较慢）...")

    t0 = time.time()
    try:
        model = CosyVoice2(str(model_path))
        elapsed = time.time() - t0
        print(f"✅ 模型加载成功，耗时 {elapsed:.1f}s")
        return model
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return None


def test_ref_audio(ref_path: str):
    """测试 3：加载参考音频"""
    print("\n" + "=" * 60)
    print("测试 3：加载参考音频")
    print("=" * 60)

    import torchaudio

    path = Path(ref_path)
    if not path.exists():
        print(f"❌ 参考音频不存在: {path}")
        return None

    try:
        audio, sr = torchaudio.load(str(path))
        duration = audio.shape[-1] / sr

        print(f"文件: {path.name}")
        print(f"采样率: {sr} Hz")
        print(f"时长: {duration:.2f}s")
        print(f"声道: {audio.shape[0]}")

        # 转单声道 + 重采样到 16kHz
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        if sr != 16000:
            audio = torchaudio.functional.resample(audio, sr, 16000)
            print(f"已重采样到 16kHz")

        print(f"✅ 参考音频加载成功")
        return audio

    except Exception as e:
        print(f"❌ 加载参考音频失败: {e}")
        return None


def test_synthesis(model, ref_path: str, ref_text: str, texts: list, output_dir: Path):
    """测试 4：zero-shot 合成"""
    print("\n" + "=" * 60)
    print("测试 4：zero-shot 语音合成")
    print("=" * 60)

    import torchaudio
    import torch

    sr = getattr(model, "sample_rate", 22050)
    print(f"模型采样率: {sr} Hz")

    output_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for i, text in enumerate(texts, 1):
        print(f"\n[{i}/{len(texts)}] 合成: {text}")
        output_path = output_dir / f"test_{i:02d}.wav"

        t0 = time.time()
        try:
            audio_chunks = []
            for chunk in model.inference_zero_shot(
                tts_text=text,
                prompt_text=ref_text,
                prompt_wav=ref_path,
                stream=False,
            ):
                audio_chunks.append(chunk["tts_speech"])

            if not audio_chunks:
                print(f"  ❌ 未返回音频数据")
                results.append(False)
                continue

            audio_out = torch.cat(audio_chunks, dim=-1)
            # 转 PCM 16-bit 保存（float32 WAV 很多播放器不支持）
            audio_out = audio_out.clamp(-1.0, 1.0)
            audio_pcm = (audio_out * 32767).to(torch.int16)
            torchaudio.save(str(output_path), audio_pcm, sr)

            elapsed = time.time() - t0
            duration = audio_out.shape[-1] / sr
            rtf = elapsed / duration

            print(f"  ✅ 成功 | 时长 {duration:.2f}s | 耗时 {elapsed:.2f}s | RTF {rtf:.3f}")
            print(f"  输出: {output_path}")
            results.append(True)

        except Exception as e:
            elapsed = time.time() - t0
            print(f"  ❌ 失败 ({elapsed:.2f}s): {e}")
            results.append(False)

    return results


def test_via_audio_pipeline(ref_path: str, text: str, output_dir: Path):
    """测试 5：通过 AudioPipeline 合成（集成测试）"""
    print("\n" + "=" * 60)
    print("测试 5：通过 AudioPipeline 合成（集成测试）")
    print("=" * 60)

    from configs.model_config import AudioConfig
    from src.audio_pipeline import AudioPipeline

    config = AudioConfig(
        enabled=True,
        cosyvoice_repo_dir=COSYVOICE_REPO,
        cosyvoice_model_dir=MODEL_DIR,
        ref_audio_dir=str(Path(ref_path).parent),
        default_ref_audio=Path(ref_path).name,
        default_ref_text="",
        cache_dir=str(output_dir / "cache"),
        cache_enabled=True,
        auto_play=False,
    )

    async def run():
        pipeline = AudioPipeline(config)
        await pipeline.start()

        if not pipeline.available:
            print("❌ AudioPipeline 不可用")
            return False

        print(f"合成: {text}")
        t0 = time.time()
        audio_bytes = await pipeline.synthesize(text, emotion="neutral")
        elapsed = time.time() - t0

        output_path = output_dir / "pipeline_test.wav"
        output_path.write_bytes(audio_bytes)
        print(f"✅ 成功 | {len(audio_bytes) / 1024:.1f} KB | 耗时 {elapsed:.2f}s")
        print(f"输出: {output_path}")

        await pipeline.stop()
        return True

    return asyncio.run(run())


def _load_ref_text_from_config() -> str:
    """尝试从 config.json 读取 default_ref_text"""
    try:
        from configs.config_loader import load_audio_config
        import json
        config_path = project_root / "config.json"
        if config_path.exists():
            with open(config_path, encoding="utf-8") as f:
                cfg = json.load(f)
            audio_cfg = load_audio_config(cfg)
            if audio_cfg.default_ref_text:
                return audio_cfg.default_ref_text
    except Exception:
        pass
    return ""


def main():
    # 从 config.json 读取默认 ref_text
    config_ref_text = _load_ref_text_from_config()

    parser = argparse.ArgumentParser(description="CosyVoice TTS 测试")
    parser.add_argument("--ref",  default=DEFAULT_REF, help="参考音频路径")
    parser.add_argument("--ref-text", default=config_ref_text or "", help="参考音频对应的文本（留空从config读取）")
    parser.add_argument("--text", default=None, help="指定合成文本（默认跑全部测试文本）")
    parser.add_argument("--skip-pipeline", action="store_true", help="跳过 AudioPipeline 集成测试")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    texts = [args.text] if args.text else TEST_TEXTS

    print(f"\n{'=' * 60}")
    print("小橘 CosyVoice TTS 测试")
    print(f"{'=' * 60}")
    print(f"参考音频: {args.ref}")
    print(f"参考文本: {args.ref_text or '(空)'}")
    print(f"输出目录: {OUTPUT_DIR}\n")

    # 1. import
    if not test_import():
        print("\n❌ import 失败，请检查 COSYVOICE_REPO 路径配置")
        return

    # 2. 加载模型
    model = test_load_model()
    if model is None:
        print("\n❌ 模型加载失败，请确认模型已下载到 MODEL_DIR")
        return

    # 3. 参考音频
    ref_ok = test_ref_audio(args.ref)
    if ref_ok is None:
        print("\n❌ 参考音频加载失败")
        return

    # 4. 合成测试
    results = test_synthesis(model, args.ref, args.ref_text, texts, OUTPUT_DIR)

    # 5. AudioPipeline 集成测试
    pipeline_ok = True
    if not args.skip_pipeline:
        pipeline_ok = test_via_audio_pipeline(args.ref, texts[0], OUTPUT_DIR)

    # 总结
    print(f"\n{'=' * 60}")
    print("测试总结")
    print(f"{'=' * 60}")
    ok = sum(results)
    total = len(results)
    print(f"零样本合成:    {ok}/{total} 成功")
    if not args.skip_pipeline:
        print(f"Pipeline 集成: {'✅ 通过' if pipeline_ok else '❌ 失败'}")

    if ok == total and (args.skip_pipeline or pipeline_ok):
        print(f"\n🎉 全部测试通过！")
        print(f"音频文件保存在: {OUTPUT_DIR}")
        print("请试听确认音色克隆效果。")
    else:
        print(f"\n⚠️  部分测试失败，请检查日志。")


if __name__ == "__main__":
    main()
