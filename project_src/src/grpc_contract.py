from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from google.protobuf import descriptor_pb2, descriptor_pool, message_factory

PROTO_DIR = Path(__file__).resolve().parents[1] / "proto"
DESCRIPTOR_PATH = PROTO_DIR / "bert_inference.desc"
PACKAGE = "bert_inference"
SERVICE_NAME = "BERTInference"
SERVICE_FULL_NAME = f"{PACKAGE}.{SERVICE_NAME}"

PREDICT_METHOD = f"/{SERVICE_FULL_NAME}/Predict"
PREDICT_BATCH_METHOD = f"/{SERVICE_FULL_NAME}/PredictBatch"


@lru_cache(maxsize=1)
def _descriptor_pool() -> descriptor_pool.DescriptorPool:
    if not DESCRIPTOR_PATH.exists():
        raise FileNotFoundError(
            f"gRPC descriptor not found at {DESCRIPTOR_PATH}. "
            "Run `python project_src/scripts/generate_grpc_contract.py` first."
        )

    file_set = descriptor_pb2.FileDescriptorSet()
    file_set.ParseFromString(DESCRIPTOR_PATH.read_bytes())

    pool = descriptor_pool.DescriptorPool()
    for file_proto in file_set.file:
        pool.Add(file_proto)
    return pool


def _message_class(full_name: str):
    descriptor = _descriptor_pool().FindMessageTypeByName(full_name)
    return message_factory.GetMessageClass(descriptor)


def predict_request_class():
    return _message_class(f"{PACKAGE}.PredictRequest")


def predict_response_class():
    return _message_class(f"{PACKAGE}.PredictResponse")


def predict_batch_request_class():
    return _message_class(f"{PACKAGE}.PredictBatchRequest")


def predict_batch_response_class():
    return _message_class(f"{PACKAGE}.PredictBatchResponse")


__all__ = [
    "DESCRIPTOR_PATH",
    "PREDICT_METHOD",
    "PREDICT_BATCH_METHOD",
    "predict_request_class",
    "predict_response_class",
    "predict_batch_request_class",
    "predict_batch_response_class",
]

