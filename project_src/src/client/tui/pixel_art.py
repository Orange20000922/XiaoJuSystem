"""Pixel-art asset interfaces for the terminal client."""

from __future__ import annotations

from importlib import import_module
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Protocol, Sequence

from rich.console import RenderableType
from rich.style import Style
from rich.text import Text


_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_DEFAULT_RESOURCE_DIR = _PROJECT_ROOT / "resource"

try:
    from PIL import Image

    PILLOW_AVAILABLE = True
    _NEAREST = getattr(Image, "Resampling", Image).NEAREST
except ImportError:
    Image = None
    PILLOW_AVAILABLE = False
    _NEAREST = None


TEXTUAL_IMAGE_AVAILABLE = False
try:
    import_module("textual_image.widget")
except ImportError:
    pass
else:
    TEXTUAL_IMAGE_AVAILABLE = True


@dataclass(frozen=True)
class PixelArtAsset:
    """像素立绘资源的元数据定义。每个 asset_id 代表一个可被绑定和渲染的立绘槽位。"""

    asset_id: str
    source_path: Path
    alt_text: str = ""
    tags: tuple[str, ...] = ()
    columns: Optional[int] = None
    rows: Optional[int] = None
    backend_hint: str = "textual-image"


class MetadataPixelArtRenderer:
    """基于文本描述的像素立绘渲染器。当无法使用更高级的图像渲染方案时，可以使用该渲染器生成包含立绘元数据的文本预览。"""

    def render_preview(
        self,
        asset: PixelArtAsset,
        *,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> str:
        del width, height
        lines = [
            f"当前槽位   : {asset.asset_id}",
            f"资源路径   : {asset.source_path}",
            f"渲染后端   : {asset.backend_hint}",
        ]
        if asset.columns is not None or asset.rows is not None:
            lines.append(f"尺寸提示   : {asset.columns or '?'} x {asset.rows or '?'}")
        if asset.tags:
            lines.append(f"标签       : {', '.join(asset.tags)}")
        if asset.alt_text:
            lines.append(f"说明       : {asset.alt_text}")
        return "\n".join(lines)


class AnsiPixelArtRenderer:
    """基于 ANSI 转义序列的像素立绘渲染器。使用半块字符（▀ 和 ▄）结合前景色和背景色来模拟像素图像的效果。需要 Pillow 库支持图像处理。"""

    def __init__(
        self,
        *,
        default_width: int = 24,
        max_width: int = 34,
        background_hex: str = "#101410",
        alpha_cutoff: int = 24,
    ):
        self.default_width = default_width
        self.max_width = max_width
        self.background_hex = background_hex
        self.alpha_cutoff = alpha_cutoff

    def render_preview(
        self,
        asset: PixelArtAsset,
        *,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> RenderableType:
        if not PILLOW_AVAILABLE:
            return MetadataPixelArtRenderer().render_preview(asset)

        try:
            with Image.open(asset.source_path) as source:
                image = source.convert("RGBA")
        except Exception as exc:
            return (
                "立绘渲染失败\n"
                f"资源  : {asset.asset_id}\n"
                f"原因  : {exc}"
            )

        target_width = self._resolve_width(asset, image.width, width)
        target_height = self._resolve_height(asset, image, target_width, height)
        preview = image.resize((target_width, target_height), _NEAREST)
        return self._render_rgba(preview)

    def _resolve_width(
        self,
        asset: PixelArtAsset,
        source_width: int,
        width: Optional[int],
    ) -> int:
        hint_width = asset.columns or source_width
        target_width = width or hint_width or self.default_width
        return max(8, min(int(target_width), self.max_width))

    def _resolve_height(
        self,
        asset: PixelArtAsset,
        image: Any,
        target_width: int,
        height: Optional[int],
    ) -> int:
        if height is not None:
            return max(8, int(height))
        if asset.rows is not None:
            return max(8, int(asset.rows))

        aspect_ratio = image.height / max(image.width, 1)
        target_height = round(target_width * aspect_ratio)
        if target_height % 2 != 0:
            target_height += 1
        return max(8, target_height)

    def _render_rgba(self, image: Any) -> Text:
        text = Text(no_wrap=True)
        pixels = image.load()
        width, height = image.size

        for y in range(0, height, 2):
            for x in range(width):
                top = pixels[x, y]
                bottom = pixels[x, y + 1] if y + 1 < height else (0, 0, 0, 0)
                char, style = self._pixel_pair(top, bottom)
                text.append(char, style)
            if y + 2 < height:
                text.append("\n")

        return text

    def _pixel_pair(
        self,
        top: tuple[int, int, int, int],
        bottom: tuple[int, int, int, int],
    ) -> tuple[str, Optional[Style]]:
        top_visible = top[3] >= self.alpha_cutoff
        bottom_visible = bottom[3] >= self.alpha_cutoff

        if not top_visible and not bottom_visible:
            return " ", None
        if top_visible and bottom_visible:
            return "▀", Style(color=self._hex(top), bgcolor=self._hex(bottom))
        if top_visible:
            return "▀", Style(color=self._hex(top), bgcolor=self.background_hex)
        return "▄", Style(color=self._hex(bottom), bgcolor=self.background_hex)

    @staticmethod
    def _hex(pixel: tuple[int, int, int, int]) -> str:
        return f"#{pixel[0]:02x}{pixel[1]:02x}{pixel[2]:02x}"


def build_textual_image_widget(asset: PixelArtAsset) -> Any | None:
    """尝试构建一个基于 textual-image 库的图像小部件以渲染给定的 PixelArtAsset。如果 textual-image 不可用或构造失败，则返回 None。
    该函数尝试多种构造方式以兼容不同版本的 textual-image 库，确保在库更新时仍有较高的成功率。
    """

    if not TEXTUAL_IMAGE_AVAILABLE:
        return None

    try:
        module = import_module("textual_image.widget")
    except Exception:
        return None

    widget_cls = getattr(module, "Image", None)
    if widget_cls is None:
        return None

    constructor_attempts = [
        ((str(asset.source_path),), {}),
        ((asset.source_path,), {}),
        ((), {"path": str(asset.source_path)}),
        ((), {"path": asset.source_path}),
        ((), {"image": str(asset.source_path)}),
    ]
    for args, kwargs in constructor_attempts:
        try:
            return widget_cls(*args, **kwargs)
        except Exception:
            continue
    return None


class PixelArtRegistry:
    """像素立绘资源注册表。负责管理多个 PixelArtAsset 实例，支持注册、查询和预览功能。客户端应用可以通过该注册表维护当前可用的立绘资源，并根据需要切换和展示不同的立绘。"""

    def __init__(self, base_dir: str | Path | None = None):
        resolved_base = Path(base_dir) if base_dir is not None else _DEFAULT_RESOURCE_DIR
        self._base_dir = resolved_base.resolve()
        self._assets: Dict[str, PixelArtAsset] = {}
        self._active_asset_id: Optional[str] = None

    @property
    def active_asset_id(self) -> Optional[str]:
        return self._active_asset_id

    def register(
        self,
        asset_id: str,
        source_path: str | Path,
        *,
        alt_text: str = "",
        tags: Sequence[str] = (),
        columns: Optional[int] = None,
        rows: Optional[int] = None,
        backend_hint: str = "textual-image",
        activate: bool = True,
    ) -> PixelArtAsset:
        resolved_path = self._resolve_path(source_path)
        if not resolved_path.exists():
            raise FileNotFoundError(f"未找到像素立绘资源：{resolved_path}")

        asset = PixelArtAsset(
            asset_id=asset_id,
            source_path=resolved_path,
            alt_text=alt_text,
            tags=tuple(tags),
            columns=columns,
            rows=rows,
            backend_hint=backend_hint,
        )
        self._assets[asset_id] = asset
        if activate or self._active_asset_id is None:
            self._active_asset_id = asset_id
        return asset

    def set_active(self, asset_id: str) -> PixelArtAsset:
        asset = self.get(asset_id)
        if asset is None:
            raise KeyError(f"未知的像素立绘资源：{asset_id}")
        self._active_asset_id = asset_id
        return asset

    def get(self, asset_id: str) -> Optional[PixelArtAsset]:
        return self._assets.get(asset_id)

    def get_active(self) -> Optional[PixelArtAsset]:
        if self._active_asset_id is None:
            return None
        return self._assets.get(self._active_asset_id)

    def list_assets(self) -> tuple[PixelArtAsset, ...]:
        return tuple(self._assets.values())

    def preview_text(self, renderer: Optional[PixelArtRenderer] = None) -> str:
        asset = self.get_active()
        if asset is None:
            return (
                "尚未注册像素立绘资源。\n"
                "使用 /sprite <id> <path> 绑定立绘槽位。\n"
                "相对路径默认从 resource/ 解析。"
            )
        resolved_renderer = renderer or MetadataPixelArtRenderer()
        preview = resolved_renderer.render_preview(asset)
        return preview if isinstance(preview, str) else MetadataPixelArtRenderer().render_preview(asset)

    def preview_renderable(
        self,
        renderer: Optional[PixelArtRenderer] = None,
        *,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> RenderableType:
        asset = self.get_active()
        if asset is None:
            return self.preview_text()
        resolved_renderer = renderer or MetadataPixelArtRenderer()
        return resolved_renderer.render_preview(asset, width=width, height=height)

    def _resolve_path(self, source_path: str | Path) -> Path:
        path = Path(source_path)
        if path.is_absolute():
            return path.resolve()
        if self._base_dir is not None:
            return (self._base_dir / path).resolve()
        return path.resolve()
