"""
ONNX 导出脚本

支持导出:
1. 标注模型 (AnnotationModel)
2. 联合模型 (JointModel)
3. Tokenizer词表 (供C++使用)

导出格式:
- model.onnx: ONNX模型文件
- model.json: 元数据 (标签映射、配置等)
- vocab.txt: 词表文件
- tokenizer_config.json: Tokenizer配置
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn


def export_annotation_model(
    checkpoint_path: str,
    output_dir: str,
    opset_version: int = 14,
    dynamic_batch: bool = True,
    simplify: bool = True
) -> Tuple[str, str]:
    """
    导出标注模型到ONNX

    Args:
        checkpoint_path: PyTorch检查点路径
        output_dir: 输出目录
        opset_version: ONNX opset版本
        dynamic_batch: 是否支持动态batch
        simplify: 是否简化ONNX图

    Returns:
        (onnx_path, metadata_path)
    """
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    from models.annotation_model import AnnotationModel, AnnotationModelConfig

    print(f"加载检查点: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # 恢复配置
    if "config" in checkpoint:
        config = checkpoint["config"]
    else:
        config = AnnotationModelConfig()

    # 创建模型
    model = AnnotationModel(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 准备输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 导出ONNX
    onnx_path = output_dir / "annotation_model.onnx"

    # 创建示例输入
    batch_size = 1
    seq_length = config.max_length
    dummy_input_ids = torch.randint(0, 21128, (batch_size, seq_length))  # BERT vocab size
    dummy_attention_mask = torch.ones(batch_size, seq_length, dtype=torch.long)

    # 动态轴配置
    if dynamic_batch:
        dynamic_axes = {
            "input_ids": {0: "batch_size", 1: "seq_length"},
            "attention_mask": {0: "batch_size", 1: "seq_length"},
            "emotion_logits": {0: "batch_size"},
            "intensity": {0: "batch_size"}
        }
    else:
        dynamic_axes = None

    # 包装模型以只输出需要的张量
    class AnnotationModelWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, input_ids, attention_mask):
            output = self.model(input_ids, attention_mask)
            return output.emotion_logits, output.intensity

    wrapped_model = AnnotationModelWrapper(model)
    wrapped_model.eval()

    print(f"导出ONNX: {onnx_path}")
    torch.onnx.export(
        wrapped_model,
        (dummy_input_ids, dummy_attention_mask),
        str(onnx_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["emotion_logits", "intensity"],
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        do_constant_folding=True
    )

    # 简化ONNX (可选)
    if simplify:
        try:
            import onnx
            from onnxsim import simplify as onnx_simplify

            print("简化ONNX模型...")
            onnx_model = onnx.load(str(onnx_path))
            simplified_model, check = onnx_simplify(onnx_model)
            if check:
                onnx.save(simplified_model, str(onnx_path))
                print("简化成功")
            else:
                print("简化验证失败，保留原模型")
        except ImportError:
            print("未安装onnxsim，跳过简化 (pip install onnxsim)")

    # 导出元数据
    metadata_path = output_dir / "annotation_model.json"
    metadata = {
        "model_type": "annotation_model",
        "config": {
            "pretrained_model": config.pretrained_model,
            "hidden_size": config.hidden_size,
            "num_emotions": config.num_emotions,
            "max_length": config.max_length,
            "dropout": config.dropout
        },
        "label_map": {
            "0": "joy",
            "1": "sadness",
            "2": "anger",
            "3": "fear",
            "4": "surprise",
            "5": "disgust",
            "6": "neutral",
            "7": "excitement",
            "8": "tenderness",
            "9": "curiosity"
        },
        "outputs": {
            "emotion_logits": {
                "shape": ["batch_size", config.num_emotions],
                "description": "情绪分类logits，需要softmax"
            },
            "intensity": {
                "shape": ["batch_size"],
                "description": "情绪强度，范围0-1"
            }
        }
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"元数据保存到: {metadata_path}")

    # 导出Tokenizer
    export_tokenizer(config.pretrained_model, output_dir)

    return str(onnx_path), str(metadata_path)


def export_joint_model(
    checkpoint_path: str,
    output_dir: str,
    opset_version: int = 14,
    dynamic_batch: bool = True,
    simplify: bool = True
) -> Tuple[str, str]:
    """
    导出联合模型到ONNX

    Args:
        checkpoint_path: PyTorch检查点路径
        output_dir: 输出目录
        opset_version: ONNX opset版本
        dynamic_batch: 是否支持动态batch
        simplify: 是否简化ONNX图

    Returns:
        (onnx_path, metadata_path)
    """
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    from models.joint_model import JointEmotionBehaviorModel, create_joint_model
    from configs.model_config import JointModelConfig

    print(f"加载检查点: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # 恢复配置
    if "config" in checkpoint:
        config = checkpoint["config"]
    else:
        config = JointModelConfig()

    # 创建模型
    model, tokenizer = create_joint_model(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 准备输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 导出ONNX
    onnx_path = output_dir / "joint_model.onnx"

    # 创建示例输入
    batch_size = 1
    seq_length = config.max_length
    personality_dim = config.personality_dim

    dummy_input_ids = torch.randint(0, 21128, (batch_size, seq_length))
    dummy_attention_mask = torch.ones(batch_size, seq_length, dtype=torch.long)
    dummy_personality = torch.randn(batch_size, personality_dim)

    # 动态轴配置
    if dynamic_batch:
        dynamic_axes = {
            "input_ids": {0: "batch_size", 1: "seq_length"},
            "attention_mask": {0: "batch_size", 1: "seq_length"},
            "personality": {0: "batch_size"},
            "emotion_logits": {0: "batch_size"},
            "behavior_logits": {0: "batch_size"},
            "tone_logits": {0: "batch_size"},
            "intensity": {0: "batch_size"},
            "response_length": {0: "batch_size"}
        }
    else:
        dynamic_axes = None

    # 包装模型
    class JointModelWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, input_ids, attention_mask, personality):
            output = self.model(input_ids, attention_mask, personality)
            return (
                output.emotion_logits,
                output.behavior_logits,
                output.tone_logits,
                output.intensity,
                output.response_length
            )

    wrapped_model = JointModelWrapper(model)
    wrapped_model.eval()

    print(f"导出ONNX: {onnx_path}")
    torch.onnx.export(
        wrapped_model,
        (dummy_input_ids, dummy_attention_mask, dummy_personality),
        str(onnx_path),
        input_names=["input_ids", "attention_mask", "personality"],
        output_names=["emotion_logits", "behavior_logits", "tone_logits", "intensity", "response_length"],
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        do_constant_folding=True
    )

    # 简化ONNX
    if simplify:
        try:
            import onnx
            from onnxsim import simplify as onnx_simplify

            print("简化ONNX模型...")
            onnx_model = onnx.load(str(onnx_path))
            simplified_model, check = onnx_simplify(onnx_model)
            if check:
                onnx.save(simplified_model, str(onnx_path))
                print("简化成功")
            else:
                print("简化验证失败，保留原模型")
        except ImportError:
            print("未安装onnxsim，跳过简化")

    # 导出元数据
    metadata_path = output_dir / "joint_model.json"
    metadata = {
        "model_type": "joint_model",
        "config": {
            "pretrained_model": config.pretrained_model,
            "hidden_size": config.hidden_size,
            "num_emotions": config.num_emotions,
            "num_behaviors": config.num_behaviors,
            "num_tones": config.num_tones,
            "personality_dim": config.personality_dim,
            "max_length": config.max_length
        },
        "label_map": {
            "emotions": {
                "0": "joy", "1": "sadness", "2": "anger", "3": "fear",
                "4": "surprise", "5": "disgust", "6": "neutral",
                "7": "excitement", "8": "tenderness", "9": "curiosity"
            },
            "behaviors": {
                "0": "respond_positive", "1": "respond_negative", "2": "ask_question",
                "3": "share_experience", "4": "give_advice", "5": "express_empathy",
                "6": "make_joke", "7": "change_topic", "8": "seek_clarification",
                "9": "agree", "10": "disagree", "11": "neutral_acknowledge"
            },
            "tones": {
                "0": "enthusiastic", "1": "calm", "2": "playful", "3": "serious",
                "4": "warm", "5": "cold", "6": "sarcastic", "7": "supportive"
            }
        },
        "outputs": {
            "emotion_logits": {"shape": ["batch_size", config.num_emotions]},
            "behavior_logits": {"shape": ["batch_size", config.num_behaviors]},
            "tone_logits": {"shape": ["batch_size", config.num_tones]},
            "intensity": {"shape": ["batch_size"], "range": [0, 1]},
            "response_length": {"shape": ["batch_size", 3], "labels": ["short", "medium", "long"]}
        }
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"元数据保存到: {metadata_path}")

    # 导出Tokenizer
    export_tokenizer(config.pretrained_model, output_dir)

    return str(onnx_path), str(metadata_path)


def export_tokenizer(pretrained_model: str, output_dir: Path):
    """
    导出Tokenizer词表和配置 (供C++使用)

    Args:
        pretrained_model: HuggingFace模型名
        output_dir: 输出目录
    """
    from transformers import AutoTokenizer

    print(f"导出Tokenizer: {pretrained_model}")
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

    # 保存vocab.txt
    vocab_path = output_dir / "vocab.txt"
    vocab = tokenizer.get_vocab()

    # 按ID排序
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])

    with open(vocab_path, "w", encoding="utf-8") as f:
        for token, idx in sorted_vocab:
            f.write(f"{token}\n")

    print(f"词表保存到: {vocab_path} ({len(vocab)} tokens)")

    # 保存tokenizer配置
    tokenizer_config = {
        "vocab_size": len(vocab),
        "pad_token": tokenizer.pad_token,
        "pad_token_id": tokenizer.pad_token_id,
        "unk_token": tokenizer.unk_token,
        "unk_token_id": tokenizer.unk_token_id,
        "cls_token": tokenizer.cls_token,
        "cls_token_id": tokenizer.cls_token_id,
        "sep_token": tokenizer.sep_token,
        "sep_token_id": tokenizer.sep_token_id,
        "mask_token": tokenizer.mask_token,
        "mask_token_id": tokenizer.mask_token_id,
        "do_lower_case": getattr(tokenizer, "do_lower_case", True),
        "model_max_length": tokenizer.model_max_length
    }

    config_path = output_dir / "tokenizer_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(tokenizer_config, f, ensure_ascii=False, indent=2)

    print(f"Tokenizer配置保存到: {config_path}")


def verify_onnx(onnx_path: str, checkpoint_path: str, model_type: str = "annotation"):
    """
    验证ONNX模型输出与PyTorch一致

    Args:
        onnx_path: ONNX模型路径
        checkpoint_path: PyTorch检查点路径
        model_type: 模型类型 ("annotation" 或 "joint")
    """
    import numpy as np
    import onnxruntime as ort

    print(f"\n验证ONNX模型: {onnx_path}")

    # 加载ONNX
    ort_session = ort.InferenceSession(onnx_path)

    # 加载PyTorch模型
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if model_type == "annotation":
        from models.annotation_model import AnnotationModel, AnnotationModelConfig
        config = checkpoint.get("config", AnnotationModelConfig())
        model = AnnotationModel(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        # 创建测试输入
        batch_size = 2
        seq_length = config.max_length
        input_ids = torch.randint(0, 21128, (batch_size, seq_length))
        attention_mask = torch.ones(batch_size, seq_length, dtype=torch.long)

        # PyTorch推理
        with torch.no_grad():
            pt_output = model(input_ids, attention_mask)
            pt_emotion_logits = pt_output.emotion_logits.numpy()
            pt_intensity = pt_output.intensity.numpy()

        # ONNX推理
        ort_inputs = {
            "input_ids": input_ids.numpy(),
            "attention_mask": attention_mask.numpy()
        }
        ort_outputs = ort_session.run(None, ort_inputs)
        ort_emotion_logits = ort_outputs[0]
        ort_intensity = ort_outputs[1]

        # 比较
        emotion_diff = np.abs(pt_emotion_logits - ort_emotion_logits).max()
        intensity_diff = np.abs(pt_intensity - ort_intensity).max()

        print(f"emotion_logits 最大差异: {emotion_diff:.6f}")
        print(f"intensity 最大差异: {intensity_diff:.6f}")

        if emotion_diff < 1e-4 and intensity_diff < 1e-4:
            print("✓ 验证通过!")
        else:
            print("⚠ 存在较大差异，请检查")

    else:  # joint
        from models.joint_model import JointEmotionBehaviorModel, create_joint_model
        from configs.model_config import JointModelConfig

        config = checkpoint.get("config", JointModelConfig())
        model, _ = create_joint_model(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        # 创建测试输入
        batch_size = 2
        seq_length = config.max_length
        input_ids = torch.randint(0, 21128, (batch_size, seq_length))
        attention_mask = torch.ones(batch_size, seq_length, dtype=torch.long)
        personality = torch.randn(batch_size, config.personality_dim)

        # PyTorch推理
        with torch.no_grad():
            pt_output = model(input_ids, attention_mask, personality)

        # ONNX推理
        ort_inputs = {
            "input_ids": input_ids.numpy(),
            "attention_mask": attention_mask.numpy(),
            "personality": personality.numpy()
        }
        ort_outputs = ort_session.run(None, ort_inputs)

        # 比较主要输出
        emotion_diff = np.abs(pt_output.emotion_logits.numpy() - ort_outputs[0]).max()
        behavior_diff = np.abs(pt_output.behavior_logits.numpy() - ort_outputs[1]).max()

        print(f"emotion_logits 最大差异: {emotion_diff:.6f}")
        print(f"behavior_logits 最大差异: {behavior_diff:.6f}")

        if emotion_diff < 1e-4 and behavior_diff < 1e-4:
            print("✓ 验证通过!")
        else:
            print("⚠ 存在较大差异，请检查")


def benchmark_onnx(onnx_path: str, batch_sizes: list = [1, 8, 32, 128], num_runs: int = 100):
    """
    ONNX模型性能测试

    Args:
        onnx_path: ONNX模型路径
        batch_sizes: 测试的batch大小列表
        num_runs: 每个batch大小的运行次数
    """
    import numpy as np
    import onnxruntime as ort
    import time

    print(f"\n性能测试: {onnx_path}")

    # 尝试使用GPU
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    try:
        ort_session = ort.InferenceSession(onnx_path, providers=providers)
        provider = ort_session.get_providers()[0]
        print(f"使用: {provider}")
    except Exception:
        ort_session = ort.InferenceSession(onnx_path)
        print("使用: CPU")

    # 获取输入信息
    input_info = ort_session.get_inputs()
    seq_length = 64  # 默认序列长度

    print(f"\n{'Batch':<10} {'延迟(ms)':<15} {'吞吐量(samples/s)':<20}")
    print("-" * 50)

    for batch_size in batch_sizes:
        # 准备输入
        inputs = {}
        for inp in input_info:
            shape = [batch_size if dim is None or dim == "batch_size" else
                     seq_length if dim == "seq_length" else dim
                     for dim in inp.shape]

            if inp.type == "tensor(int64)":
                inputs[inp.name] = np.random.randint(0, 21128, shape).astype(np.int64)
            else:
                inputs[inp.name] = np.random.randn(*shape).astype(np.float32)

        # 预热
        for _ in range(10):
            ort_session.run(None, inputs)

        # 计时
        start = time.perf_counter()
        for _ in range(num_runs):
            ort_session.run(None, inputs)
        elapsed = time.perf_counter() - start

        avg_latency = (elapsed / num_runs) * 1000  # ms
        throughput = (batch_size * num_runs) / elapsed

        print(f"{batch_size:<10} {avg_latency:<15.2f} {throughput:<20.1f}")


def main():
    parser = argparse.ArgumentParser(description="ONNX导出工具")
    subparsers = parser.add_subparsers(dest="command", help="命令")

    # 导出标注模型
    ann_parser = subparsers.add_parser("annotation", help="导出标注模型")
    ann_parser.add_argument("--checkpoint", type=str, required=True, help="检查点路径")
    ann_parser.add_argument("--output", type=str, default="./onnx/annotation", help="输出目录")
    ann_parser.add_argument("--opset", type=int, default=14, help="ONNX opset版本")
    ann_parser.add_argument("--no-simplify", action="store_true", help="不简化ONNX")
    ann_parser.add_argument("--verify", action="store_true", help="验证导出结果")
    ann_parser.add_argument("--benchmark", action="store_true", help="性能测试")

    # 导出联合模型
    joint_parser = subparsers.add_parser("joint", help="导出联合模型")
    joint_parser.add_argument("--checkpoint", type=str, required=True, help="检查点路径")
    joint_parser.add_argument("--output", type=str, default="./onnx/joint", help="输出目录")
    joint_parser.add_argument("--opset", type=int, default=14, help="ONNX opset版本")
    joint_parser.add_argument("--no-simplify", action="store_true", help="不简化ONNX")
    joint_parser.add_argument("--verify", action="store_true", help="验证导出结果")
    joint_parser.add_argument("--benchmark", action="store_true", help="性能测试")

    # 仅导出Tokenizer
    tok_parser = subparsers.add_parser("tokenizer", help="仅导出Tokenizer")
    tok_parser.add_argument("--model", type=str, default="hfl/chinese-roberta-wwm-ext",
                           help="HuggingFace模型名")
    tok_parser.add_argument("--output", type=str, default="./onnx", help="输出目录")

    # 性能测试
    bench_parser = subparsers.add_parser("benchmark", help="ONNX性能测试")
    bench_parser.add_argument("--onnx", type=str, required=True, help="ONNX模型路径")
    bench_parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32, 128],
                             help="测试的batch大小")

    args = parser.parse_args()

    if args.command == "annotation":
        onnx_path, _ = export_annotation_model(
            checkpoint_path=args.checkpoint,
            output_dir=args.output,
            opset_version=args.opset,
            simplify=not args.no_simplify
        )

        if args.verify:
            verify_onnx(onnx_path, args.checkpoint, "annotation")

        if args.benchmark:
            benchmark_onnx(onnx_path)

    elif args.command == "joint":
        onnx_path, _ = export_joint_model(
            checkpoint_path=args.checkpoint,
            output_dir=args.output,
            opset_version=args.opset,
            simplify=not args.no_simplify
        )

        if args.verify:
            verify_onnx(onnx_path, args.checkpoint, "joint")

        if args.benchmark:
            benchmark_onnx(onnx_path)

    elif args.command == "tokenizer":
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        export_tokenizer(args.model, output_dir)

    elif args.command == "benchmark":
        benchmark_onnx(args.onnx, args.batch_sizes)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
