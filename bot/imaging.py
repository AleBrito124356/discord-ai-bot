"""Make any uploaded image fit what a NIM vision model accepts.

NVIDIA's hosted Llama 3.2 Vision examples accept JPEG/PNG passed inline as a
base64 data URI, and cap that payload at 180 000 base64 characters (larger
images need the separate assets API). Discord users upload whatever they have:
4000x3000 phone photos, animated GIFs, WEBP stickers, BMP screenshots.

:func:`normalize_image` therefore:

* passes a JPEG/PNG through **unchanged** when it already fits;
* otherwise decodes it with Pillow, applies EXIF rotation, takes the first frame
  of an animation, flattens transparency onto white, downscales so the longest
  side is at most ``max_side`` and re-encodes as JPEG, lowering quality and then
  size until the base64 payload fits ``max_b64``.

It returns the bytes to send, their MIME type, and a human-readable note that
the bot shows next to the answer when anything was changed.
"""
from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from typing import Optional, Tuple

try:  # Pillow is a runtime dependency, but keep imports of the bot working without it.
    from PIL import Image, ImageOps, UnidentifiedImageError

    PIL_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without Pillow installed
    Image = ImageOps = None  # type: ignore[assignment]
    UnidentifiedImageError = OSError  # type: ignore[assignment,misc]
    PIL_AVAILABLE = False

PASSTHROUGH_FORMATS = {"JPEG": "image/jpeg", "PNG": "image/png"}
MIN_SIDE = 64
_QUALITIES = (88, 80, 72, 64, 56, 48)


class ImageError(ValueError):
    """The upload can not be turned into a usable image. Message is user-safe."""


@dataclass
class NormalizedImage:
    data: bytes
    mime: str
    width: int
    height: int
    original_format: str
    original_width: int
    original_height: int
    original_bytes: int
    changed: bool
    note: str
    frames: int = 1
    mode: str = "RGB"

    @property
    def b64_len(self) -> int:
        return b64_length(len(self.data))


def b64_length(raw_bytes: int) -> int:
    """Length of the base64 encoding of ``raw_bytes`` bytes."""
    return 4 * ((raw_bytes + 2) // 3)


def _flatten(img: "Image.Image") -> "Image.Image":
    """RGB copy with any transparency composited onto white."""
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.getchannel("A"))
        return background
    return img.convert("RGB")


def _encode_jpeg(img: "Image.Image", quality: int) -> bytes:
    out = io.BytesIO()
    img.save(out, format="JPEG", quality=quality, optimize=True)
    return out.getvalue()


def _fit(img: "Image.Image", max_side: int) -> "Image.Image":
    if max(img.size) <= max_side:
        return img
    scale = max_side / float(max(img.size))
    size = (max(1, round(img.width * scale)), max(1, round(img.height * scale)))
    return img.resize(size, Image.LANCZOS)


def open_image(data: bytes) -> "Image.Image":
    if not PIL_AVAILABLE:
        raise ImageError("Image support needs Pillow: pip install Pillow")
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Image.DecompressionBombError as exc:
        raise ImageError("That image has far too many pixels to process safely.") from exc
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImageError(
            "That file is not an image format I can read (try PNG, JPG, WEBP or GIF)."
        ) from exc
    return img


def normalize_image(
    data: bytes,
    mime: Optional[str] = None,
    *,
    max_side: int = 1568,
    max_b64: int = 180_000,
) -> NormalizedImage:
    """Return an upload-ready version of ``data`` (see the module docstring)."""
    img = open_image(data)
    fmt = (img.format or "UNKNOWN").upper()
    frames = int(getattr(img, "n_frames", 1) or 1)
    original: Tuple[int, int] = img.size

    if (
        fmt in PASSTHROUGH_FORMATS
        and frames == 1
        and max(original) <= max_side
        and b64_length(len(data)) <= max_b64
    ):
        return NormalizedImage(
            data=data,
            mime=PASSTHROUGH_FORMATS[fmt],
            width=original[0],
            height=original[1],
            original_format=fmt,
            original_width=original[0],
            original_height=original[1],
            original_bytes=len(data),
            changed=False,
            note="",
            frames=frames,
            mode=img.mode,
        )

    if frames > 1:
        img.seek(0)
    oriented = ImageOps.exif_transpose(img) or img
    rgb = _flatten(oriented)
    rgb = _fit(rgb, max_side)

    encoded = b""
    while True:
        for quality in _QUALITIES:
            encoded = _encode_jpeg(rgb, quality)
            if b64_length(len(encoded)) <= max_b64:
                break
        if b64_length(len(encoded)) <= max_b64:
            break
        if min(rgb.size) * 0.75 < MIN_SIDE:
            raise ImageError(
                "That image can not be compressed enough for the vision model."
            )
        rgb = rgb.resize(
            (max(1, round(rgb.width * 0.75)), max(1, round(rgb.height * 0.75))),
            Image.LANCZOS,
        )

    steps = []
    if fmt not in PASSTHROUGH_FORMATS:
        steps.append(f"converted {fmt}{' (first frame)' if frames > 1 else ''} to JPEG")
    elif frames > 1:
        steps.append("used the first frame")
    else:
        steps.append(f"re-encoded {fmt} as JPEG")
    if rgb.size != original:
        steps.append(f"resized {original[0]}x{original[1]} -> {rgb.width}x{rgb.height}")
    steps.append(f"{len(data) // 1024} KB -> {len(encoded) // 1024} KB")
    return NormalizedImage(
        data=encoded,
        mime="image/jpeg",
        width=rgb.width,
        height=rgb.height,
        original_format=fmt,
        original_width=original[0],
        original_height=original[1],
        original_bytes=len(data),
        changed=True,
        note="Image " + ", ".join(steps) + " to fit the vision model's limits.",
        frames=frames,
        mode=rgb.mode,
    )


def to_data_uri(data: bytes, mime: str) -> str:
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


# ---------------------------------------------------------- offline analysis
_NAMED_COLOURS = {
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "grey": (128, 128, 128),
    "red": (200, 30, 30),
    "orange": (240, 140, 20),
    "yellow": (235, 220, 40),
    "green": (40, 160, 60),
    "teal": (20, 150, 150),
    "blue": (40, 80, 210),
    "purple": (130, 50, 170),
    "pink": (240, 130, 180),
    "brown": (120, 75, 40),
}


def _colour_name(rgb: Tuple[int, int, int]) -> str:
    return min(
        _NAMED_COLOURS,
        key=lambda n: sum((a - b) ** 2 for a, b in zip(rgb, _NAMED_COLOURS[n])),
    )


def describe_pixels(data: bytes) -> dict:
    """Cheap, deterministic facts about an image (used by the offline backend)."""
    img = open_image(data)
    fmt = (img.format or "UNKNOWN").upper()
    frames = int(getattr(img, "n_frames", 1) or 1)
    rgb = _flatten(img)
    raw = rgb.resize((32, 32), Image.BILINEAR).tobytes()
    pixels = [tuple(raw[i : i + 3]) for i in range(0, len(raw), 3)]
    avg = tuple(sum(p[i] for p in pixels) // len(pixels) for i in range(3))
    counts: dict = {}
    for p in pixels:
        name = _colour_name(p)
        counts[name] = counts.get(name, 0) + 1
    dominant = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:3]
    brightness = sum(avg) / (3 * 255)
    width, height = img.size
    orientation = (
        "square" if abs(width - height) <= 0.05 * max(width, height)
        else ("landscape" if width > height else "portrait")
    )
    return {
        "format": fmt,
        "width": width,
        "height": height,
        "frames": frames,
        "orientation": orientation,
        "brightness": round(brightness, 2),
        "average_rgb": avg,
        "dominant_colours": [(name, round(n / len(pixels), 2)) for name, n in dominant],
    }
