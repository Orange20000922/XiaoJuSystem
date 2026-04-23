"""Pixel-art asset interfaces for the terminal client."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Protocol, Sequence


@dataclass(frozen=True)
class PixelArtAsset:
    """Registered pixel-art metadata."""

    asset_id: str
    source_path: Path
    alt_text: str = ""
    tags: tuple[str, ...] = ()
    columns: Optional[int] = None
    rows: Optional[int] = None
    backend_hint: str = "textual-image"


class PixelArtRenderer(Protocol):
    """Renderer contract for future image-capable terminal backends."""

    def render_preview(self, asset: PixelArtAsset) -> str:
        """Return a textual preview for the current frontend."""


class MetadataPixelArtRenderer:
    """Fallback renderer that exposes metadata until image widgets arrive."""

    def render_preview(self, asset: PixelArtAsset) -> str:
        lines = [
            f"active slot : {asset.asset_id}",
            f"source      : {asset.source_path}",
            f"backend     : {asset.backend_hint}",
        ]
        if asset.columns is not None or asset.rows is not None:
            lines.append(f"size hint   : {asset.columns or '?'} x {asset.rows or '?'}")
        if asset.tags:
            lines.append(f"tags        : {', '.join(asset.tags)}")
        if asset.alt_text:
            lines.append(f"alt         : {asset.alt_text}")
        return "\n".join(lines)


class PixelArtRegistry:
    """Registry for pixel-art resources used by the Textual client."""

    def __init__(self, base_dir: str | Path | None = None):
        self._base_dir = Path(base_dir).resolve() if base_dir is not None else None
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
            raise FileNotFoundError(f"Pixel-art asset was not found: {resolved_path}")

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
            raise KeyError(f"Unknown pixel-art asset: {asset_id}")
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
                "No pixel-art asset registered.\n"
                "Use /sprite <id> <path> to reserve a portrait slot."
            )
        resolved_renderer = renderer or MetadataPixelArtRenderer()
        return resolved_renderer.render_preview(asset)

    def _resolve_path(self, source_path: str | Path) -> Path:
        path = Path(source_path)
        if path.is_absolute():
            return path.resolve()
        if self._base_dir is not None:
            return (self._base_dir / path).resolve()
        return path.resolve()
