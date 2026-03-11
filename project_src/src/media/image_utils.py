"""
图片处理工具

为 Anthropic Vision API 提供图片下载、验证、缩放和 base64 编码。

特性：
  - 基于 URL SHA256 哈希的文件缓存，24h TTL 自动过期
  - Pillow 可选依赖：未安装时 disable 图片功能并 log 警告
  - 超过 max_dimension 自动等比缩放（Anthropic 推荐 ≤1568px）
  - 支持 jpeg/png/gif/webp 格式
"""

import base64
import hashlib
import io
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import httpx

from src.logger import logger

# Pillow 可选依赖
try:
    from PIL import Image
    PILLOW_AVAILABLE = True
except ImportError:
    PILLOW_AVAILABLE = False
    logger.warning(
        "Pillow 未安装，图片功能已禁用。安装方式: pip install Pillow"
    )

# 支持的图片格式 → MIME 映射
_FORMAT_TO_MIME = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "GIF": "image/gif",
    "WEBP": "image/webp",
}


@dataclass
class ImageResult:
    """处理后的图片数据，可直接用于 Anthropic API"""
    base64_data: str
    media_type: str       # "image/jpeg", "image/png", etc.
    original_url: str


def download_image(
    url: str,
    timeout: float = 15.0,
    max_size_bytes: int = 10_485_760,
) -> bytes:
    """
    同步下载图片。

    Args:
        url: 图片 URL
        timeout: 下载超时（秒）
        max_size_bytes: 最大下载大小（字节），默认 10MB

    Returns:
        图片原始字节

    Raises:
        ValueError: 图片超过大小限制
        httpx.HTTPError: 网络错误
    """
    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        response = client.get(url)
        response.raise_for_status()

        if len(response.content) > max_size_bytes:
            raise ValueError(
                f"图片大小 {len(response.content)} 字节超过限制 {max_size_bytes} 字节"
            )

        return response.content


def validate_image(data: bytes) -> str:
    """
    验证图片格式。

    Args:
        data: 图片原始字节

    Returns:
        Pillow 格式名（"JPEG", "PNG", "GIF", "WEBP"）

    Raises:
        ValueError: 格式不支持或数据损坏
    """
    if not PILLOW_AVAILABLE:
        raise ValueError("Pillow 未安装，无法验证图片")

    try:
        img = Image.open(io.BytesIO(data))
        img.verify()  # 只验证不解码
    except Exception as e:
        raise ValueError(f"图片数据无效: {e}") from e

    fmt = img.format
    if fmt not in _FORMAT_TO_MIME:
        raise ValueError(
            f"不支持的图片格式: {fmt}，"
            f"支持: {', '.join(_FORMAT_TO_MIME.keys())}"
        )

    return fmt


def resize_if_needed(data: bytes, max_dimension: int = 1568) -> bytes:
    """
    超过 max_dimension 时等比缩放。

    Args:
        data: 图片原始字节
        max_dimension: 最大边长（Anthropic 推荐 1568）

    Returns:
        缩放后的字节（JPEG 或原格式）
    """
    if not PILLOW_AVAILABLE:
        return data

    img = Image.open(io.BytesIO(data))
    w, h = img.size

    if w <= max_dimension and h <= max_dimension:
        return data

    # 等比缩放
    ratio = min(max_dimension / w, max_dimension / h)
    new_w = int(w * ratio)
    new_h = int(h * ratio)

    # GIF 动图只取第一帧缩放
    if img.format == "GIF":
        img = img.convert("RGBA")

    img = img.resize((new_w, new_h), Image.LANCZOS)

    buf = io.BytesIO()
    # 保持原格式，JPEG 不支持 alpha 通道
    save_format = img.format if img.format in _FORMAT_TO_MIME else "PNG"
    if save_format == "JPEG" and img.mode in ("RGBA", "P"):
        img = img.convert("RGB")

    img.save(buf, format=save_format, quality=85)
    logger.debug(f"图片缩放: {w}x{h} → {new_w}x{new_h}")
    return buf.getvalue()


def _cache_key(url: str) -> str:
    """URL → SHA256 哈希文件名"""
    return hashlib.sha256(url.encode()).hexdigest()


def _clean_expired_cache(cache_dir: Path, ttl_seconds: int):
    """删除过期缓存文件"""
    if not cache_dir.exists():
        return
    now = time.time()
    for f in cache_dir.iterdir():
        if f.is_file() and (now - f.stat().st_mtime) > ttl_seconds:
            try:
                f.unlink()
            except OSError:
                pass


def process_image_url(
    url: str,
    cache_dir: str = "./data/image_cache",
    max_download_size_bytes: int = 10_485_760,
    max_dimension: int = 1568,
    cache_ttl_seconds: int = 86400,
    download_timeout: float = 15.0,
) -> Optional[ImageResult]:
    """
    完整图片处理流程：缓存检查 → 下载 → 验证 → 缩放 → base64 编码。

    任何步骤失败都返回 None 并 log WARNING。

    Args:
        url: 图片 URL
        cache_dir: 缓存目录
        max_download_size_bytes: 最大下载大小
        max_dimension: 最大边长
        cache_ttl_seconds: 缓存 TTL（秒）
        download_timeout: 下载超时（秒）

    Returns:
        ImageResult 或 None（失败时）
    """
    if not PILLOW_AVAILABLE:
        logger.warning("Pillow 未安装，跳过图片处理")
        return None

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    # 清理过期缓存（惰性清理，每次处理时检查）
    try:
        _clean_expired_cache(cache_path, cache_ttl_seconds)
    except Exception:
        pass

    key = _cache_key(url)
    cached_file = cache_path / key

    # 缓存命中
    if cached_file.exists():
        age = time.time() - cached_file.stat().st_mtime
        if age < cache_ttl_seconds:
            try:
                data = cached_file.read_bytes()
                fmt = validate_image(data)
                media_type = _FORMAT_TO_MIME[fmt]
                b64 = base64.standard_b64encode(data).decode("ascii")
                logger.debug(f"图片缓存命中: {url[:80]}")
                return ImageResult(
                    base64_data=b64,
                    media_type=media_type,
                    original_url=url,
                )
            except Exception as e:
                logger.warning(f"缓存图片读取失败，重新下载: {e}")
                try:
                    cached_file.unlink()
                except OSError:
                    pass

    # 下载
    try:
        raw_data = download_image(
            url,
            timeout=download_timeout,
            max_size_bytes=max_download_size_bytes,
        )
    except Exception as e:
        logger.warning(f"图片下载失败: {url[:80]} — {e}")
        return None

    # 验证
    try:
        fmt = validate_image(raw_data)
    except ValueError as e:
        logger.warning(f"图片验证失败: {url[:80]} — {e}")
        return None

    # 缩放
    try:
        processed_data = resize_if_needed(raw_data, max_dimension)
    except Exception as e:
        logger.warning(f"图片缩放失败: {url[:80]} — {e}")
        processed_data = raw_data  # 缩放失败用原图

    # 如果缩放后格式变了，重新验证
    if processed_data is not raw_data:
        try:
            fmt = validate_image(processed_data)
        except ValueError:
            fmt = validate_image(raw_data)
            processed_data = raw_data

    media_type = _FORMAT_TO_MIME[fmt]

    # 写入缓存
    try:
        cached_file.write_bytes(processed_data)
    except OSError as e:
        logger.warning(f"缓存写入失败: {e}")

    b64 = base64.standard_b64encode(processed_data).decode("ascii")
    logger.debug(
        f"图片处理完成: {url[:80]} format={fmt} "
        f"size={len(processed_data)} bytes"
    )

    return ImageResult(
        base64_data=b64,
        media_type=media_type,
        original_url=url,
    )
