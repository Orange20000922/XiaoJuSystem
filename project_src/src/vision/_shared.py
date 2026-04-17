from __future__ import annotations

import json
import math
import queue
import re
from pathlib import Path
from typing import Optional, Union

import numpy as np

from src.media.image_utils import ImageResult

try:
    import cv2

    CV2_AVAILABLE = True
except ImportError:
    cv2 = None
    CV2_AVAILABLE = False


def _ensure_cv2() -> None:
    if not CV2_AVAILABLE:
        raise RuntimeError(
            "动态视觉模块需要 OpenCV。"
            "请先安装 opencv-python-headless 或 opencv-python。"
        )


def _odd_kernel(value: int) -> int:
    value = max(1, int(value))
    return value if value % 2 == 1 else value + 1


def sigmoid_normalize(value: float, center: float, scale: float) -> float:
    scale = max(scale, 1e-6)
    return 1.0 / (1.0 + math.exp(-((value - center) / scale)))


def _strip_json_fence(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def _extract_json_object(text: str) -> Optional[dict]:
    cleaned = _strip_json_fence(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _fit_width(frame: np.ndarray, width: int) -> np.ndarray:
    if width <= 0:
        return frame
    h, w = frame.shape[:2]
    if w <= width:
        return frame
    scale = width / float(w)
    resized_h = max(1, int(round(h * scale)))
    return cv2.resize(frame, (width, resized_h), interpolation=cv2.INTER_AREA)


def _encode_frame_as_image_result(
    frame_bgr: np.ndarray,
    frame_index: int,
    timestamp: float,
    jpeg_quality: int,
) -> ImageResult:
    _ensure_cv2()
    success, encoded = cv2.imencode(
        ".jpg",
        frame_bgr,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
    )
    if not success:
        raise RuntimeError(f"Failed to encode keyframe {frame_index} as JPEG")

    import base64

    payload = base64.standard_b64encode(encoded.tobytes()).decode("ascii")
    return ImageResult(
        base64_data=payload,
        media_type="image/jpeg",
        original_url=f"visual://frame/{frame_index}?ts={timestamp:.3f}",
    )


def _push_queue_sentinel(target_queue: "queue.Queue[object]") -> None:
    while True:
        try:
            target_queue.put(None, timeout=0.1)
            return
        except queue.Full:
            try:
                target_queue.get_nowait()
            except queue.Empty:
                continue


def _open_video_writer(
    output_path: Union[str, Path],
    frame_width: int,
    frame_height: int,
    fps: float,
):
    _ensure_cv2()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    suffix = output.suffix.lower()
    codec = "MJPG" if suffix == ".avi" else "mp4v"
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*codec),
        max(1.0, float(fps)),
        (int(frame_width), int(frame_height)),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Unable to open video writer: {output}")
    return writer


__all__ = [
    "CV2_AVAILABLE",
    "cv2",
    "sigmoid_normalize",
    "_encode_frame_as_image_result",
    "_ensure_cv2",
    "_extract_json_object",
    "_fit_width",
    "_odd_kernel",
    "_open_video_writer",
    "_push_queue_sentinel",
]
