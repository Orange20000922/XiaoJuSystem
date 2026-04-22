"""Benchmark small-model backends under concurrent load."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import psutil
import torch

import sys

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

from configs.config_loader import AppConfig
from src.core_engine import BERTInferenceEngine


DEFAULT_TEXTS = [
    "老师，我这一步有点没跟上，可以再讲一遍吗？",
    "这个公式为什么这里要先平方再约分？",
    "我感觉这个知识点和上节课讲的函数有点像。",
    "等一下，我想确认一下定义域是不是这里限制了。",
    "如果考试里这么出题，我可能会先这样想。",
    "老师我懂前两步了，最后一步为什么能直接推出这个结论？",
    "这个地方我有点紧张，怕自己算错。",
    "原来是这样，我刚才把条件看漏了。",
    "那是不是也可以换一种方法来证明？",
    "我现在大概明白了，但还想再做一道类似的题。",
]


def bytes_to_mb(value: int) -> float:
    return float(value) / (1024.0 * 1024.0)


def percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = fraction * (len(ordered) - 1)
    lower = int(index)
    upper = min(len(ordered) - 1, lower + 1)
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize(latencies_ms: List[float], failed_requests: int, elapsed_seconds: float) -> Dict:
    total_requests = len(latencies_ms)
    success_requests = total_requests - failed_requests
    average_ms = sum(latencies_ms) / total_requests if total_requests else 0.0
    throughput_rps = success_requests / elapsed_seconds if elapsed_seconds > 0 else 0.0
    return {
        "total_requests": total_requests,
        "success_requests": success_requests,
        "failed_requests": failed_requests,
        "elapsed_seconds": elapsed_seconds,
        "throughput_rps": throughput_rps,
        "average_ms": average_ms,
        "p50_ms": percentile(latencies_ms, 0.50),
        "p95_ms": percentile(latencies_ms, 0.95),
        "p99_ms": percentile(latencies_ms, 0.99),
    }


def query_gpu_stats(gpu_index: int = 0) -> Optional[Dict]:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            encoding="utf-8",
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return None

    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if gpu_index >= len(lines):
        return None

    parts = [part.strip() for part in lines[gpu_index].split(",")]
    if len(parts) < 3:
        return None

    try:
        return {
            "gpu_index": gpu_index,
            "name": parts[0],
            "memory_used_mb": float(parts[1]),
            "utilization_pct": float(parts[2]),
        }
    except ValueError:
        return None


def resolve_processes_by_name(names: Sequence[str]) -> List[psutil.Process]:
    wanted = {name.lower() for name in names if name}
    if not wanted:
        return []

    processes: List[psutil.Process] = []
    seen_pids = set()
    for proc in psutil.process_iter(["pid", "name"]):
        try:
            name = (proc.info.get("name") or "").lower()
            if name in wanted and proc.pid not in seen_pids:
                processes.append(proc)
                seen_pids.add(proc.pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return processes


class ResourceSampler:
    def __init__(
        self,
        extra_process_names: Sequence[str],
        sample_interval_ms: float,
        gpu_index: int,
        enable_gpu_sampling: bool = True,
    ):
        self._extra_process_names = list(extra_process_names)
        self._sample_interval_s = max(0.05, float(sample_interval_ms) / 1000.0)
        self._gpu_index = gpu_index
        self._enable_gpu_sampling = enable_gpu_sampling
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._baseline_processes: Dict[int, Dict] = {}
        self._peak_processes: Dict[int, Dict] = {}
        self._tracked_rss_baseline_mb: Optional[float] = None
        self._tracked_rss_peak_mb: float = 0.0
        self._gpu_baseline: Optional[Dict] = None
        self._gpu_peak: Optional[Dict] = None

    def start(self):
        self._sample(store_baseline=True)
        self._thread = threading.Thread(
            target=self._worker,
            name="resource-sampler",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> Dict:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._sample(store_baseline=False)
        return self.result()

    def result(self) -> Dict:
        processes = []
        baseline_total = 0.0
        peak_total = 0.0
        python_rss_baseline = 0.0
        python_rss_peak = 0.0
        extra_rss_baseline = 0.0
        extra_rss_peak = 0.0

        for pid in sorted(self._peak_processes):
            peak_info = self._peak_processes[pid]
            baseline_info = self._baseline_processes.get(pid, peak_info)
            role = peak_info["role"]
            processes.append(
                {
                    "pid": pid,
                    "name": peak_info["name"],
                    "role": role,
                    "rss_mb_baseline": baseline_info["rss_mb"],
                    "rss_mb_peak": peak_info["rss_mb"],
                }
            )
            baseline_total += baseline_info["rss_mb"]
            peak_total += peak_info["rss_mb"]
            if role == "python":
                python_rss_baseline += baseline_info["rss_mb"]
                python_rss_peak += peak_info["rss_mb"]
            else:
                extra_rss_baseline += baseline_info["rss_mb"]
                extra_rss_peak += peak_info["rss_mb"]

        gpu = None
        if self._gpu_peak is not None:
            gpu = {
                "gpu_index": self._gpu_peak["gpu_index"],
                "name": self._gpu_peak["name"],
                "memory_used_mb_baseline": (
                    self._gpu_baseline["memory_used_mb"] if self._gpu_baseline else None
                ),
                "memory_used_mb_peak": self._gpu_peak["memory_used_mb"],
                "utilization_pct_peak": self._gpu_peak["utilization_pct"],
            }

        return {
            "tracked_processes": processes,
            "tracked_rss_mb_baseline": (
                self._tracked_rss_baseline_mb if self._tracked_rss_baseline_mb is not None else baseline_total
            ),
            "tracked_rss_mb_peak": max(self._tracked_rss_peak_mb, peak_total),
            "python_rss_mb_baseline": python_rss_baseline,
            "python_rss_mb_peak": python_rss_peak,
            "extra_rss_mb_baseline": extra_rss_baseline,
            "extra_rss_mb_peak": extra_rss_peak,
            "gpu": gpu,
        }

    def _worker(self):
        while not self._stop_event.wait(self._sample_interval_s):
            self._sample(store_baseline=False)

    def _sample(self, store_baseline: bool):
        tracked = [{"role": "python", "process": psutil.Process(os.getpid())}]
        tracked.extend(
            {"role": "extra", "process": proc}
            for proc in resolve_processes_by_name(self._extra_process_names)
        )

        unique = {}
        for item in tracked:
            proc = item["process"]
            unique[proc.pid] = {"role": item["role"], "process": proc}

        total_rss_mb = 0.0
        for pid, item in unique.items():
            proc = item["process"]
            try:
                rss_mb = bytes_to_mb(proc.memory_info().rss)
                name = proc.name()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

            total_rss_mb += rss_mb
            record = {
                "pid": pid,
                "name": name,
                "role": item["role"],
                "rss_mb": rss_mb,
            }

            if store_baseline and pid not in self._baseline_processes:
                self._baseline_processes[pid] = record

            peak = self._peak_processes.get(pid)
            if peak is None or rss_mb > peak["rss_mb"]:
                self._peak_processes[pid] = record

        if store_baseline and self._tracked_rss_baseline_mb is None:
            self._tracked_rss_baseline_mb = total_rss_mb
        self._tracked_rss_peak_mb = max(self._tracked_rss_peak_mb, total_rss_mb)

        if not self._enable_gpu_sampling:
            return

        gpu_stats = query_gpu_stats(self._gpu_index)
        if gpu_stats is None:
            return

        if store_baseline and self._gpu_baseline is None:
            self._gpu_baseline = gpu_stats

        if (
            self._gpu_peak is None
            or gpu_stats["memory_used_mb"] > self._gpu_peak["memory_used_mb"]
        ):
            self._gpu_peak = gpu_stats
        elif gpu_stats["utilization_pct"] > self._gpu_peak["utilization_pct"]:
            self._gpu_peak = {
                **self._gpu_peak,
                "utilization_pct": gpu_stats["utilization_pct"],
            }


def run_benchmark_round(
    engine: BERTInferenceEngine,
    personality_vector: torch.Tensor,
    texts: Sequence[str],
    concurrency: int,
    requests_per_worker: int,
    sample_interval_ms: float = 200.0,
    extra_process_names: Optional[Sequence[str]] = None,
    gpu_index: int = 0,
    enable_gpu_sampling: bool = True,
) -> Dict:
    start_event = threading.Event()
    device = torch.device(engine.device)
    sampler = ResourceSampler(
        extra_process_names=extra_process_names or [],
        sample_interval_ms=sample_interval_ms,
        gpu_index=gpu_index,
        enable_gpu_sampling=enable_gpu_sampling,
    )
    pytorch_cuda_metrics = None

    def worker(worker_index: int):
        local_latencies: List[float] = []
        local_failures = 0

        start_event.wait()

        for request_index in range(requests_per_worker):
            text = texts[(worker_index * requests_per_worker + request_index) % len(texts)]
            start = time.perf_counter()
            try:
                result = engine.predict(text, personality_vector)
                if not result:
                    raise RuntimeError("predict() returned no result")
            except Exception:
                local_failures += 1
            finally:
                end = time.perf_counter()
                local_latencies.append((end - start) * 1000.0)

        return local_latencies, local_failures

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        pytorch_cuda_metrics = {
            "memory_allocated_mb_baseline": bytes_to_mb(torch.cuda.memory_allocated(device)),
            "memory_reserved_mb_baseline": bytes_to_mb(torch.cuda.memory_reserved(device)),
        }
        torch.cuda.reset_peak_memory_stats(device)

    sampler.start()

    try:
        with ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="small-model-bench") as pool:
            futures = [pool.submit(worker, index) for index in range(concurrency)]
            start = time.perf_counter()
            start_event.set()

            all_latencies: List[float] = []
            failed_requests = 0
            for future in futures:
                latencies, failures = future.result()
                all_latencies.extend(latencies)
                failed_requests += failures

            elapsed = time.perf_counter() - start
    finally:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        resources = sampler.stop()

    summary = summarize(all_latencies, failed_requests, elapsed)

    if pytorch_cuda_metrics is not None:
        pytorch_cuda_metrics["memory_allocated_mb_peak"] = bytes_to_mb(
            torch.cuda.max_memory_allocated(device)
        )
        pytorch_cuda_metrics["memory_reserved_mb_peak"] = bytes_to_mb(
            torch.cuda.max_memory_reserved(device)
        )
        resources["pytorch_cuda"] = pytorch_cuda_metrics
        resources["gpu_note"] = (
            "Per-process GPU memory is unavailable on Windows WDDM via nvidia-smi; "
            "PyTorch allocator peaks are reported separately."
        )
    elif resources.get("gpu") is not None:
        resources["gpu_note"] = (
            "GPU metrics are whole-device samples from nvidia-smi. "
            "Per-process GPU memory is unavailable on Windows WDDM."
        )

    summary["resources"] = resources
    return summary


def warmup(engine: BERTInferenceEngine, personality_vector: torch.Tensor, texts: Sequence[str], count: int):
    if count <= 0:
        return
    for index in range(count):
        engine.predict(texts[index % len(texts)], personality_vector)


def print_table(results: List[Dict]):
    header = (
        f"{'backend':<12} {'conc':>5} {'req':>8} {'ok':>8} {'fail':>8} "
        f"{'rps':>10} {'avg_ms':>10} {'p95_ms':>10} {'rss_mb':>10} {'gpu_mb':>10}"
    )
    print(header)
    print("-" * len(header))
    for item in results:
        resources = item.get("resources") or {}
        gpu = resources.get("gpu") or {}
        print(
            f"{item['backend']:<12} "
            f"{item['concurrency']:>5} "
            f"{item['total_requests']:>8} "
            f"{item['success_requests']:>8} "
            f"{item['failed_requests']:>8} "
            f"{item['throughput_rps']:>10.2f} "
            f"{item['average_ms']:>10.2f} "
            f"{item['p95_ms']:>10.2f} "
            f"{resources.get('tracked_rss_mb_peak', 0.0):>10.2f} "
            f"{gpu.get('memory_used_mb_peak', 0.0):>10.2f}"
        )


def main():
    parser = argparse.ArgumentParser(description="Benchmark PyTorch vs ONNX gRPC small-model backends")
    parser.add_argument("--config", type=str, default="config.json", help="Path to config.json")
    parser.add_argument(
        "--backend",
        nargs="+",
        default=["pytorch", "onnx_grpc"],
        help="Backends to benchmark: pytorch onnx_grpc",
    )
    parser.add_argument(
        "--concurrency",
        nargs="+",
        type=int,
        default=[1, 4, 8, 16],
        help="Concurrency levels to benchmark",
    )
    parser.add_argument("--requests-per-worker", type=int, default=100, help="Requests per worker")
    parser.add_argument("--warmup", type=int, default=20, help="Warmup requests before each backend")
    parser.add_argument("--onnx-target", type=str, default=None, help="Override ONNX gRPC target")
    parser.add_argument("--tokenizer-path", type=str, default=None, help="Override tokenizer path for ONNX gRPC")
    parser.add_argument(
        "--grpc-timeout-seconds",
        type=float,
        default=None,
        help="Override ONNX gRPC timeout in seconds",
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Override micro-batch size")
    parser.add_argument("--batch-wait-ms", type=float, default=None, help="Override micro-batch wait time in ms")
    parser.add_argument("--max-length", type=int, default=None, help="Override tokenizer max_length")
    parser.add_argument(
        "--sample-interval-ms",
        type=float,
        default=200.0,
        help="Resource sampling interval in milliseconds",
    )
    parser.add_argument(
        "--gpu-index",
        type=int,
        default=0,
        help="GPU index to sample via nvidia-smi",
    )
    parser.add_argument(
        "--extra-process-name",
        nargs="+",
        default=None,
        help="Additional process names to include in RSS sampling",
    )
    parser.add_argument(
        "--disable-gpu-sampling",
        action="store_true",
        help="Disable nvidia-smi GPU sampling",
    )
    parser.add_argument("--output", type=str, default=None, help="Optional JSON output path")
    args = parser.parse_args()

    app_cfg = AppConfig.load(args.config)
    results: List[Dict] = []

    for backend_name in args.backend:
        small_model_cfg = replace(app_cfg.small_model, backend=backend_name)
        if args.onnx_target:
            small_model_cfg = replace(small_model_cfg, onnx_target=args.onnx_target)
        if args.tokenizer_path:
            small_model_cfg = replace(small_model_cfg, tokenizer_path=args.tokenizer_path)
        if args.grpc_timeout_seconds is not None:
            small_model_cfg = replace(
                small_model_cfg,
                grpc_timeout_seconds=args.grpc_timeout_seconds,
            )
        if args.batch_size is not None:
            small_model_cfg = replace(small_model_cfg, batch_size=args.batch_size)
        if args.batch_wait_ms is not None:
            small_model_cfg = replace(small_model_cfg, batch_wait_ms=args.batch_wait_ms)
        if args.max_length is not None:
            small_model_cfg = replace(small_model_cfg, max_length=args.max_length)

        engine = BERTInferenceEngine(small_model_config=small_model_cfg)
        if not engine.available:
            raise RuntimeError(f"Backend {backend_name!r} is unavailable")

        personality_vector = torch.tensor(
            app_cfg.personality.to_embedding_vector(),
            dtype=torch.float32,
            device=torch.device(engine.device),
        )

        try:
            warmup(engine, personality_vector, DEFAULT_TEXTS, args.warmup)

            for concurrency in args.concurrency:
                extra_process_names = []
                if backend_name == "onnx_grpc":
                    extra_process_names = args.extra_process_name or ["bert_inference_server.exe"]

                summary = run_benchmark_round(
                    engine=engine,
                    personality_vector=personality_vector,
                    texts=DEFAULT_TEXTS,
                    concurrency=concurrency,
                    requests_per_worker=args.requests_per_worker,
                    sample_interval_ms=args.sample_interval_ms,
                    extra_process_names=extra_process_names,
                    gpu_index=args.gpu_index,
                    enable_gpu_sampling=not args.disable_gpu_sampling,
                )
                summary["backend"] = backend_name
                summary["concurrency"] = concurrency
                summary["small_model"] = asdict(small_model_cfg)
                results.append(summary)
        finally:
            engine.close()

    print_table(results)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(results, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
