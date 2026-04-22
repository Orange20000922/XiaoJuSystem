"""Small-model inference backends with optional micro-batching."""

from __future__ import annotations

import os
import queue
import threading
import time
from concurrent.futures import Future, TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import torch

from configs.model_config import (
    ID_TO_BEHAVIOR,
    ID_TO_EMOTION,
    ID_TO_TONE,
    SmallModelConfig,
)
from src.logger import logger
from src.grpc_contract import (
    PREDICT_BATCH_METHOD,
    predict_batch_request_class,
    predict_batch_response_class,
)


_LENGTH_MAP = {0: "short", 1: "medium", 2: "long"}


def _auto_set_hf_offline():
    """Enable offline mode when the local HuggingFace cache is already present."""
    model_name = "hfl/chinese-roberta-wwm-ext"
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    model_dir = cache_dir / ("models--" + model_name.replace("/", "--"))
    if model_dir.exists() and "HF_HUB_OFFLINE" not in os.environ:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"


def _resolve_device(device: str) -> str:
    """Resolve torch device from config."""
    if device == "auto":
        if torch.cuda.is_available():
            resolved = "cuda"
            logger.info(f"Auto-detected CUDA: {torch.cuda.get_device_name(0)}")
        else:
            resolved = "cpu"
            logger.info("CUDA unavailable, falling back to CPU")
        return resolved

    if device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning(f"Configured {device!r} but CUDA is unavailable, falling back to CPU")
        return "cpu"

    return device


def _softmax(values: Sequence[float]) -> List[float]:
    tensor = torch.tensor(list(values), dtype=torch.float32)
    return torch.softmax(tensor, dim=-1).tolist()


def _build_prediction(
    emotion_probs: Sequence[float],
    behavior_probs: Sequence[float],
    tone_probs: Sequence[float],
    intensity: float,
    response_length_probs: Sequence[float],
) -> Dict:
    emotion_probs = [float(x) for x in emotion_probs]
    behavior_probs = [float(x) for x in behavior_probs]
    tone_probs = [float(x) for x in tone_probs]
    response_length_probs = [float(x) for x in response_length_probs]

    emotion_id = max(range(len(emotion_probs)), key=emotion_probs.__getitem__)
    behavior_id = max(range(len(behavior_probs)), key=behavior_probs.__getitem__)
    tone_id = max(range(len(tone_probs)), key=tone_probs.__getitem__)
    length_id = max(range(len(response_length_probs)), key=response_length_probs.__getitem__)

    return {
        "emotion": {
            "primary": ID_TO_EMOTION[emotion_id],
            "primary_prob": emotion_probs[emotion_id],
            "intensity": float(intensity),
            "all_probs": {
                ID_TO_EMOTION[i]: emotion_probs[i]
                for i in range(len(emotion_probs))
            },
        },
        "behavior": {
            "type": ID_TO_BEHAVIOR[behavior_id],
            "type_prob": behavior_probs[behavior_id],
            "tone": ID_TO_TONE[tone_id],
            "tone_prob": tone_probs[tone_id],
            "response_length": _LENGTH_MAP[length_id],
            "all_behaviors": {
                ID_TO_BEHAVIOR[i]: behavior_probs[i]
                for i in range(len(behavior_probs))
            },
        },
    }


def _build_prediction_from_logits(
    emotion_logits: Sequence[float],
    behavior_logits: Sequence[float],
    tone_logits: Sequence[float],
    intensity: float,
    response_length_logits: Sequence[float],
) -> Dict:
    return _build_prediction(
        emotion_probs=_softmax(emotion_logits),
        behavior_probs=_softmax(behavior_logits),
        tone_probs=_softmax(tone_logits),
        intensity=float(intensity),
        response_length_probs=_softmax(response_length_logits),
    )


@dataclass
class _PendingRequest:
    text: str
    personality_vector: torch.Tensor
    future: Future
    submitted_at: float
    deadline_at: Optional[float]


class _MicroBatcher:
    """Collect concurrent requests into short micro-batches."""

    def __init__(
        self,
        name: str,
        max_batch_size: int,
        batch_wait_ms: float,
        request_timeout_seconds: float,
        max_queue_size: int,
        process_batch: Callable[[List[_PendingRequest]], List[Dict]],
    ):
        self._name = name
        self._max_batch_size = max(1, int(max_batch_size))
        self._batch_wait_seconds = max(0.0, float(batch_wait_ms) / 1000.0)
        self._request_timeout_seconds = max(0.0, float(request_timeout_seconds))
        self._max_queue_size = max(1, int(max_queue_size))
        self._process_batch = process_batch
        self._queue: "queue.Queue[Optional[_PendingRequest]]" = queue.Queue(
            maxsize=self._max_queue_size
        )
        self._closed = threading.Event()
        self._metrics_lock = threading.Lock()
        self._rejected_requests = 0
        self._timed_out_requests = 0
        self._thread = threading.Thread(
            target=self._worker,
            name=f"{name}-batcher",
            daemon=True,
        )
        self._thread.start()

    def submit(self, text: str, personality_vector: torch.Tensor) -> Dict:
        if self._closed.is_set():
            raise RuntimeError(f"{self._name} batcher is closed")

        future = Future()
        submitted_at = time.perf_counter()
        deadline_at = None
        if self._request_timeout_seconds > 0:
            deadline_at = submitted_at + self._request_timeout_seconds

        pending_request = _PendingRequest(
            text=text,
            personality_vector=personality_vector,
            future=future,
            submitted_at=submitted_at,
            deadline_at=deadline_at,
        )

        try:
            self._queue.put_nowait(pending_request)
        except queue.Full as exc:
            with self._metrics_lock:
                self._rejected_requests += 1
            raise RuntimeError(
                f"{self._name} queue is full "
                f"(pending={self._queue.qsize()} max={self._max_queue_size})"
            ) from exc

        wait_timeout = None
        if deadline_at is not None:
            wait_timeout = max(0.0, deadline_at - time.perf_counter())

        try:
            return future.result(timeout=wait_timeout)
        except FutureTimeoutError as exc:
            with self._metrics_lock:
                self._timed_out_requests += 1
            future.cancel()
            raise RuntimeError(
                f"{self._name} request timed out after "
                f"{self._request_timeout_seconds:.3f}s"
            ) from exc

    def close(self):
        if self._closed.is_set():
            return
        self._closed.set()

        deadline = time.perf_counter() + 5.0
        while time.perf_counter() < deadline:
            try:
                self._queue.put(None, timeout=0.1)
                break
            except queue.Full:
                if not self._thread.is_alive():
                    break
        else:
            logger.warning(
                f"{self._name} batcher close timed out waiting to enqueue shutdown sentinel"
            )

        self._thread.join(timeout=5.0)

    def stats(self) -> Dict[str, int]:
        with self._metrics_lock:
            return {
                "pending_requests": self._queue.qsize(),
                "max_queue_size": self._max_queue_size,
                "rejected_requests": self._rejected_requests,
                "timed_out_requests": self._timed_out_requests,
            }

    def _is_expired(self, request: _PendingRequest) -> bool:
        return request.deadline_at is not None and time.perf_counter() >= request.deadline_at

    def _fail_request(self, request: _PendingRequest, exc: Exception):
        if request.future.done():
            return
        request.future.set_exception(exc)

    def _worker(self):
        stop_after_batch = False

        while True:
            item = self._queue.get()
            if item is None:
                break

            if item.future.cancelled():
                continue
            if self._is_expired(item):
                self._fail_request(
                    item,
                    RuntimeError(
                        f"{self._name} request expired before batch execution "
                        f"after {self._request_timeout_seconds:.3f}s"
                    ),
                )
                continue

            batch = [item]
            deadline = time.perf_counter() + self._batch_wait_seconds

            while len(batch) < self._max_batch_size:
                timeout = deadline - time.perf_counter()
                if timeout <= 0:
                    break
                try:
                    next_item = self._queue.get(timeout=timeout)
                except queue.Empty:
                    break

                if next_item is None:
                    stop_after_batch = True
                    break

                if next_item.future.cancelled():
                    continue
                if self._is_expired(next_item):
                    self._fail_request(
                        next_item,
                        RuntimeError(
                            f"{self._name} request expired before batch execution "
                            f"after {self._request_timeout_seconds:.3f}s"
                        ),
                    )
                    continue

                batch.append(next_item)

            if not batch:
                if stop_after_batch:
                    break
                continue

            try:
                results = self._process_batch(batch)
                if len(results) != len(batch):
                    raise RuntimeError(
                        f"{self._name} batch processor returned {len(results)} results "
                        f"for {len(batch)} requests"
                    )
            except Exception as exc:
                for request in batch:
                    if not request.future.done():
                        request.future.set_exception(exc)
            else:
                for request, result in zip(batch, results):
                    if not request.future.done():
                        request.future.set_result(result)

            if stop_after_batch:
                break


def _load_tokenizer(tokenizer_source: str):
    from transformers import AutoTokenizer, BertTokenizerFast

    try:
        return AutoTokenizer.from_pretrained(tokenizer_source)
    except Exception as exc:
        source_path = Path(tokenizer_source)
        vocab_path = source_path / "vocab.txt"
        if not source_path.is_dir() or not vocab_path.exists():
            raise

        logger.warning(
            "AutoTokenizer could not load local tokenizer assets from "
            f"{tokenizer_source!r}; falling back to BertTokenizerFast: {exc}"
        )
        return BertTokenizerFast.from_pretrained(str(source_path))


class _PyTorchBackend:
    backend = "pytorch"

    def __init__(self, config: SmallModelConfig):
        self.config = config
        self.device = _resolve_device(config.device)
        self.model = None
        self.tokenizer = None
        self._predict_lock = threading.Lock()
        self._batcher: Optional[_MicroBatcher] = None

        _auto_set_hf_offline()

        try:
            from models.joint_model import create_joint_model

            logger.info(
                "Loading small model backend=pytorch "
                f"device={self.device} checkpoint={config.checkpoint_path}"
            )
            self.model, self.tokenizer = create_joint_model()
            self._load_checkpoint(config.checkpoint_path)
            self.model.to(self.device)
            self.model.eval()

            if config.batching_enabled and config.batch_size > 1:
                self._batcher = _MicroBatcher(
                    name="pytorch-small-model",
                    max_batch_size=config.batch_size,
                    batch_wait_ms=config.batch_wait_ms,
                    request_timeout_seconds=config.request_timeout_seconds,
                    max_queue_size=config.max_queue_size,
                    process_batch=self._process_batch,
                )

            logger.info(
                "Small model backend ready: "
                f"backend=pytorch device={self.device} "
                f"batching={bool(self._batcher)} batch_size={config.batch_size}"
            )
        except Exception as exc:
            logger.warning(f"Failed to load PyTorch small model, entering LLM-only mode: {exc}")
            self.model = None
            self.tokenizer = None

    @property
    def available(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    def _load_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Loaded checkpoint: {checkpoint_path}")

    def _predict_internal(
        self,
        texts: List[str],
        personalities: List[torch.Tensor],
    ) -> List[Dict]:
        encoding = self.tokenizer(
            texts,
            max_length=self.config.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        personality = torch.stack(
            [
                vector.detach().to(self.device, dtype=torch.float32).view(-1)
                for vector in personalities
            ],
            dim=0,
        )

        with torch.inference_mode():
            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                personality=personality,
            )

        results = []
        batch_size = len(texts)
        for index in range(batch_size):
            results.append(
                _build_prediction(
                    emotion_probs=output.emotion_probs[index].detach().cpu().tolist(),
                    behavior_probs=output.behavior_probs[index].detach().cpu().tolist(),
                    tone_probs=output.tone_probs[index].detach().cpu().tolist(),
                    intensity=float(output.intensity[index].item()),
                    response_length_probs=output.response_length[index].detach().cpu().tolist(),
                )
            )
        return results

    def _process_batch(self, batch: List[_PendingRequest]) -> List[Dict]:
        return self._predict_internal(
            texts=[request.text for request in batch],
            personalities=[request.personality_vector for request in batch],
        )

    def predict(self, text: str, personality_vector: torch.Tensor) -> Optional[Dict]:
        if not self.available:
            return None
        if self._batcher is not None:
            return self._batcher.submit(text, personality_vector)
        with self._predict_lock:
            return self._predict_internal([text], [personality_vector])[0]

    def close(self):
        if self._batcher is not None:
            self._batcher.close()


class _OnnxGrpcBackend:
    backend = "onnx_grpc"

    def __init__(self, config: SmallModelConfig):
        self.config = config
        self.device = "cpu"
        self.model = None
        self.tokenizer = None
        self._predict_lock = threading.Lock()
        self._batcher: Optional[_MicroBatcher] = None
        self._channel = None
        self._predict_batch = None

        _auto_set_hf_offline()

        try:
            import grpc

            self._grpc = grpc
            tokenizer_source = self._resolve_tokenizer_source()
            logger.info(
                "Loading small model backend=onnx_grpc "
                f"target={config.onnx_target} tokenizer={tokenizer_source}"
            )
            self.tokenizer = _load_tokenizer(tokenizer_source)

            self._request_cls = predict_batch_request_class()
            self._response_cls = predict_batch_response_class()

            self._channel = grpc.insecure_channel(
                config.onnx_target,
                options=[
                    ("grpc.max_send_message_length", 16 * 1024 * 1024),
                    ("grpc.max_receive_message_length", 16 * 1024 * 1024),
                ],
            )
            self._predict_batch = self._channel.unary_unary(
                PREDICT_BATCH_METHOD,
                request_serializer=lambda message: message.SerializeToString(),
                response_deserializer=self._response_cls.FromString,
            )

            try:
                grpc.channel_ready_future(self._channel).result(
                    timeout=min(config.grpc_timeout_seconds, 1.0)
                )
            except Exception:
                logger.warning(
                    "ONNX gRPC channel is not ready yet; first request may fail until the server starts"
                )

            if config.batching_enabled and config.batch_size > 1:
                self._batcher = _MicroBatcher(
                    name="onnx-grpc-small-model",
                    max_batch_size=config.batch_size,
                    batch_wait_ms=config.batch_wait_ms,
                    request_timeout_seconds=config.request_timeout_seconds,
                    max_queue_size=config.max_queue_size,
                    process_batch=self._process_batch,
                )

            logger.info(
                "Small model backend ready: "
                f"backend=onnx_grpc target={config.onnx_target} "
                f"batching={bool(self._batcher)} batch_size={config.batch_size}"
            )
        except Exception as exc:
            logger.warning(f"Failed to load ONNX gRPC small model backend: {exc}")
            self.tokenizer = None
            self._channel = None
            self._predict_batch = None

    @property
    def available(self) -> bool:
        return self.tokenizer is not None and self._predict_batch is not None

    def _resolve_tokenizer_source(self) -> str:
        if self.config.tokenizer_path:
            return self.config.tokenizer_path

        checkpoint_path = Path(self.config.checkpoint_path)
        if checkpoint_path.exists():
            if checkpoint_path.is_dir():
                return str(checkpoint_path)
            if checkpoint_path.suffix.lower() == ".onnx":
                return str(checkpoint_path.parent)

        default_candidate = Path("./onnx/joint")
        if default_candidate.exists():
            return str(default_candidate)

        return self.config.checkpoint_path

    def _predict_internal(
        self,
        texts: List[str],
        personalities: List[torch.Tensor],
    ) -> List[Dict]:
        encoding = self.tokenizer(
            texts,
            max_length=self.config.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].to(torch.int64).reshape(-1).tolist()
        attention_mask = encoding["attention_mask"].to(torch.int64).reshape(-1).tolist()

        personality_values: List[float] = []
        for vector in personalities:
            personality_values.extend(
                vector.detach().to("cpu", dtype=torch.float32).view(-1).tolist()
            )

        request = self._request_cls()
        request.input_ids.extend(input_ids)
        request.attention_mask.extend(attention_mask)
        request.personality.extend(personality_values)
        request.batch_size = len(texts)
        request.seq_length = int(encoding["input_ids"].shape[1])

        try:
            response = self._predict_batch(
                request,
                timeout=self.config.grpc_timeout_seconds,
            )
        except self._grpc.RpcError as exc:
            raise RuntimeError(
                f"ONNX gRPC request failed: code={exc.code()} details={exc.details()}"
            ) from exc

        if response.error:
            raise RuntimeError(f"ONNX backend returned error: {response.error}")

        batch_size = len(texts)
        emotion_logits = list(response.emotion_logits)
        behavior_logits = list(response.behavior_logits)
        tone_logits = list(response.tone_logits)
        intensities = list(response.intensity)
        response_length_logits = list(response.response_length_logits)

        expected_lengths = {
            "emotion_logits": batch_size * len(ID_TO_EMOTION),
            "behavior_logits": batch_size * len(ID_TO_BEHAVIOR),
            "tone_logits": batch_size * len(ID_TO_TONE),
            "intensity": batch_size,
            "response_length_logits": batch_size * len(_LENGTH_MAP),
        }
        actual_lengths = {
            "emotion_logits": len(emotion_logits),
            "behavior_logits": len(behavior_logits),
            "tone_logits": len(tone_logits),
            "intensity": len(intensities),
            "response_length_logits": len(response_length_logits),
        }
        if actual_lengths != expected_lengths:
            raise RuntimeError(
                f"Unexpected ONNX response size: expected={expected_lengths} actual={actual_lengths}"
            )

        results = []
        emotion_stride = len(ID_TO_EMOTION)
        behavior_stride = len(ID_TO_BEHAVIOR)
        tone_stride = len(ID_TO_TONE)
        length_stride = len(_LENGTH_MAP)

        for index in range(batch_size):
            results.append(
                _build_prediction_from_logits(
                    emotion_logits=emotion_logits[
                        index * emotion_stride:(index + 1) * emotion_stride
                    ],
                    behavior_logits=behavior_logits[
                        index * behavior_stride:(index + 1) * behavior_stride
                    ],
                    tone_logits=tone_logits[
                        index * tone_stride:(index + 1) * tone_stride
                    ],
                    intensity=float(intensities[index]),
                    response_length_logits=response_length_logits[
                        index * length_stride:(index + 1) * length_stride
                    ],
                )
            )

        return results

    def _process_batch(self, batch: List[_PendingRequest]) -> List[Dict]:
        return self._predict_internal(
            texts=[request.text for request in batch],
            personalities=[request.personality_vector for request in batch],
        )

    def predict(self, text: str, personality_vector: torch.Tensor) -> Optional[Dict]:
        if not self.available:
            return None
        if self._batcher is not None:
            return self._batcher.submit(text, personality_vector)
        with self._predict_lock:
            return self._predict_internal([text], [personality_vector])[0]

    def close(self):
        if self._batcher is not None:
            self._batcher.close()
        if self._channel is not None:
            self._channel.close()


class BERTInferenceEngine:
    """Facade for different small-model inference backends."""

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: str = "cpu",
        small_model_config: Optional[SmallModelConfig] = None,
    ):
        if small_model_config is None:
            config = SmallModelConfig(
                checkpoint_path=checkpoint_path or SmallModelConfig().checkpoint_path,
                device=device,
            )
        else:
            config = small_model_config

        backend = (config.backend or "pytorch").lower()
        if backend == "onnx_grpc":
            self._impl = _OnnxGrpcBackend(config)
        else:
            if backend != "pytorch":
                logger.warning(f"Unknown small model backend {backend!r}, falling back to pytorch")
            self._impl = _PyTorchBackend(config)

    @property
    def backend(self) -> str:
        return self._impl.backend

    @property
    def available(self) -> bool:
        return self._impl.available

    @property
    def device(self) -> str:
        return self._impl.device

    @property
    def model(self):
        return self._impl.model

    @property
    def tokenizer(self):
        return self._impl.tokenizer

    def predict(self, text: str, personality_vector: torch.Tensor) -> Optional[Dict]:
        return self._impl.predict(text, personality_vector)

    def close(self):
        self._impl.close()
