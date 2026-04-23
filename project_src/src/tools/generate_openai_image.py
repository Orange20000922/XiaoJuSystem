"""Generate images through an OpenAI-compatible Images API."""

from __future__ import annotations

import argparse
import base64
import sys
import urllib.request
from pathlib import Path
from typing import Optional

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from openai import OpenAI

from configs.config_loader import AppConfig
from src.logger import logger


def _normalize_base_url(base_url: str) -> str:
    normalized = base_url.strip().rstrip("/")
    for suffix in ("/images/generations", "/images/edits"):
        if normalized.endswith(suffix):
            return normalized[: -len(suffix)]
    return normalized


def _load_prompt(
    prompt_arg: Optional[str],
    prompt_file: Optional[str],
    default_prompt: str,
) -> str:
    if prompt_arg:
        return prompt_arg.strip()

    if prompt_file:
        return Path(prompt_file).read_text(encoding="utf-8").strip()

    return default_prompt.strip()


def _suffix_for_output_format(output_format: str) -> str:
    if output_format == "jpeg":
        return ".jpg"
    return f".{output_format}"


def _resolve_output_paths(
    output: Optional[str],
    output_dir: str,
    output_prefix: str,
    count: int,
    output_format: str,
) -> list[Path]:
    suffix = _suffix_for_output_format(output_format)

    if output:
        output_path = Path(output)
        if count == 1:
            if output_path.suffix:
                return [output_path]
            return [output_path.with_suffix(suffix)]

        stem = output_path.stem or output_prefix
        parent = output_path.parent if str(output_path.parent) != "" else Path(".")
        actual_suffix = output_path.suffix or suffix
        return [
            parent / f"{stem}_{index:02d}{actual_suffix}"
            for index in range(1, count + 1)
        ]

    output_root = Path(output_dir)
    return [
        output_root / f"{output_prefix}{'' if count == 1 else f'_{index:02d}'}{suffix}"
        for index in range(1, count + 1)
    ]


def _validate_gpt_image_2_request(model: str, size: str, background: str) -> None:
    if model != "gpt-image-2":
        return

    if background == "transparent":
        raise ValueError("gpt-image-2 does not support transparent backgrounds.")

    if size == "auto":
        return

    try:
        width_text, height_text = size.lower().split("x", 1)
        width = int(width_text)
        height = int(height_text)
    except ValueError as exc:
        raise ValueError(f"Invalid size format: {size!r}. Expected WIDTHxHEIGHT or auto.") from exc

    if width % 16 != 0 or height % 16 != 0:
        raise ValueError("gpt-image-2 size must use 16-pixel increments.")

    pixels = width * height
    if pixels < 655_360 or pixels > 8_294_400:
        raise ValueError("gpt-image-2 total pixels must be between 655,360 and 8,294,400.")

    ratio = max(width / height, height / width)
    if ratio > 3.0:
        raise ValueError("gpt-image-2 width/height ratio must stay between 1:3 and 3:1.")


def _download_to_path(url: str, path: Path, timeout_seconds: float) -> None:
    with urllib.request.urlopen(url, timeout=timeout_seconds) as response:
        data = response.read()
    path.write_bytes(data)


def _save_response_item(item, path: Path, timeout_seconds: float) -> None:
    encoded = getattr(item, "b64_json", None)
    if encoded:
        path.write_bytes(base64.b64decode(encoded))
        return

    url = getattr(item, "url", None)
    if url:
        _download_to_path(url, path, timeout_seconds=timeout_seconds)
        return

    raise ValueError("Image response did not contain b64_json or url.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate an image through an OpenAI-compatible Images API.",
    )
    parser.add_argument("prompt", nargs="?", help="Image prompt text.")
    parser.add_argument("--prompt-file", type=str, default=None, help="Read the prompt from a UTF-8 text file.")
    parser.add_argument("--config", type=str, default=None, help="Path to config.json.")
    parser.add_argument("--output", type=str, default=None, help="Exact output file path. For n>1, a numbered suffix is added.")
    parser.add_argument("--output-dir", type=str, default=None, help="Override the configured output directory.")
    parser.add_argument("--prefix", type=str, default=None, help="Override the configured output file prefix.")
    parser.add_argument("--model", type=str, default=None, help="Override the configured image model.")
    parser.add_argument("--api-key", type=str, default=None, help="Override the configured API key.")
    parser.add_argument("--base-url", type=str, default=None, help="Override the configured OpenAI-compatible base URL.")
    parser.add_argument("--size", type=str, default=None, help="Image size, for example 1024x1024 or auto.")
    parser.add_argument("--quality", type=str, default=None, help="quality: low / medium / high / auto.")
    parser.add_argument("--moderation", type=str, default=None, help="moderation: auto / low.")
    parser.add_argument("--background", type=str, default=None, help="background: auto / opaque / transparent.")
    parser.add_argument("--output-format", type=str, default=None, help="output format: png / jpeg / webp.")
    parser.add_argument("--response-format", type=str, default=None, help="response format: b64_json / url.")
    parser.add_argument("--n", type=int, default=None, help="How many images to generate.")
    parser.add_argument("--output-compression", type=int, default=None, help="Compression level for jpeg/webp outputs.")
    parser.add_argument("--timeout", type=float, default=None, help="Request timeout in seconds.")
    parser.add_argument("--user", type=str, default=None, help="Optional end-user identifier sent to the API.")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    app_config = AppConfig.load(args.config)
    cfg = app_config.image_generation

    prompt = _load_prompt(args.prompt, args.prompt_file, cfg.default_prompt)
    if not prompt:
        raise SystemExit("Prompt is required. Pass it directly, use --prompt-file, or set image_generation.default_prompt.")

    api_key = args.api_key or cfg.api_key
    if not api_key:
        raise SystemExit("API key is missing. Set image_generation.api_key or OPENAI_IMAGE_API_KEY / OPENAI_API_KEY.")

    model = args.model or cfg.model
    base_url = _normalize_base_url(args.base_url or cfg.base_url or "https://api.openai.com/v1")
    size = args.size or cfg.size
    quality = args.quality or cfg.quality
    moderation = args.moderation or cfg.moderation
    background = args.background or cfg.background
    output_format = args.output_format or cfg.output_format
    response_format = args.response_format or cfg.response_format
    image_count = args.n if args.n is not None else cfg.n
    output_compression = args.output_compression if args.output_compression is not None else cfg.output_compression
    timeout = args.timeout if args.timeout is not None else cfg.timeout
    user = args.user or cfg.user
    output_dir = args.output_dir or cfg.output_dir
    output_prefix = args.prefix or cfg.output_prefix

    if image_count < 1:
        raise SystemExit("n must be >= 1.")

    _validate_gpt_image_2_request(model=model, size=size, background=background)

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
        max_retries=cfg.max_retries,
    )

    request_kwargs = {
        "model": model,
        "prompt": prompt,
        "n": image_count,
        "size": size,
        "quality": quality,
        "moderation": moderation,
        "background": background,
        "output_format": output_format,
        "response_format": response_format,
    }
    if output_compression is not None:
        request_kwargs["output_compression"] = output_compression
    if user:
        request_kwargs["user"] = user

    logger.info(f"Generating image with model={model}, size={size}, base_url={base_url}")
    response = client.images.generate(**request_kwargs)
    items = list(getattr(response, "data", []) or [])
    if not items:
        raise SystemExit("The image API returned no image data.")

    output_paths = _resolve_output_paths(
        output=args.output,
        output_dir=output_dir,
        output_prefix=output_prefix,
        count=len(items),
        output_format=output_format,
    )
    
    # Ensure we don't overwrite existing files by adding a numeric suffix if needed
    count = 0
    output_count = 0
    for path in output_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        while path.exists():
          token_list = str(path).rsplit(".", 1)
          str_path = token_list[0] + "_" + str(count) + "." + token_list[1]
          path = Path(str_path)
          count += 1
        output_paths[output_count] = path
        output_count += 1
     
    for item, path in zip(items, output_paths):
        _save_response_item(item, path, timeout_seconds=float(timeout))
        logger.info(f"Saved image to {path}")
        revised_prompt = getattr(item, "revised_prompt", None)
        if revised_prompt:
            logger.info(f"Revised prompt: {revised_prompt}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
