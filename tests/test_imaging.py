"""Image normalisation for the vision model (bot/imaging.py)."""
from __future__ import annotations

import io

import pytest
from PIL import Image

from bot.imaging import ImageError, b64_length, normalize_image

MAX_SIDE, MAX_B64 = 1568, 180_000


def _noise(size, mode="RGB") -> Image.Image:
    """Detailed (hard to compress) but deterministic content."""
    small = Image.merge(
        "RGB", [Image.effect_noise((size[0] // 4, size[1] // 4), 90 + 20 * i) for i in range(3)]
    )
    img = small.resize(size, Image.NEAREST)
    return img.convert(mode) if mode != "RGB" else img


def _encode(img: Image.Image, fmt: str, **kwargs) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt, **kwargs)
    return buf.getvalue()


def _check_fits(result, max_side=MAX_SIDE, max_b64=MAX_B64):
    decoded = Image.open(io.BytesIO(result.data))
    assert decoded.format in ("JPEG", "PNG")
    assert max(decoded.size) <= max_side
    assert (result.width, result.height) == decoded.size
    assert b64_length(len(result.data)) <= max_b64
    return decoded


def _photo_like(size) -> Image.Image:
    """Smooth gradients: compresses like a photo, unlike pure noise."""
    red = Image.linear_gradient("L").resize(size)
    green = Image.linear_gradient("L").rotate(90).resize(size)
    blue = Image.radial_gradient("L").resize(size)
    return Image.merge("RGB", (red, green, blue))


def test_large_png_is_downscaled_to_max_side():
    data = _encode(_photo_like((4000, 3000)), "PNG")
    result = normalize_image(data, "image/png")
    decoded = _check_fits(result)
    assert result.changed and result.mime == "image/jpeg" and decoded.format == "JPEG"
    assert (result.original_width, result.original_height) == (4000, 3000)
    assert decoded.size == (1568, 1176)  # aspect ratio kept
    assert "resized 4000x3000 -> 1568x1176" in result.note


def test_worst_case_noise_still_fits_the_base64_budget():
    data = _encode(_noise((4000, 3000)), "PNG")
    assert b64_length(len(data)) > MAX_B64
    result = normalize_image(data, "image/png")
    decoded = _check_fits(result)
    # Quality alone was not enough: it shrank further, keeping the aspect ratio.
    assert max(decoded.size) < 1568
    assert abs(decoded.width / decoded.height - 4 / 3) < 0.01


def test_animated_gif_uses_the_first_frame():
    frames = [Image.new("RGB", (400, 300), c) for c in [(250, 200, 0), (0, 200, 0), (200, 0, 0)]]
    buf = io.BytesIO()
    frames[0].save(buf, format="GIF", save_all=True, append_images=frames[1:], duration=80, loop=0)
    result = normalize_image(buf.getvalue(), "image/gif")
    decoded = _check_fits(result)
    assert result.frames == 3 and result.changed
    r, g, b = decoded.convert("RGB").getpixel((200, 150))
    assert r > 200 and g > 150 and b < 80  # yellow: frame 0, not green/red
    assert "converted GIF (first frame) to JPEG" in result.note


def test_webp_is_converted():
    result = normalize_image(_encode(_noise((800, 600)), "WEBP"), "image/webp")
    _check_fits(result)
    assert result.original_format == "WEBP" and result.mime == "image/jpeg"


def test_transparent_png_is_flattened_on_white():
    img = Image.new("RGBA", (2000, 1000), (0, 0, 0, 0))
    img.paste((0, 0, 255, 255), (0, 0, 1000, 1000))
    result = normalize_image(_encode(img, "PNG"), "image/png")
    decoded = _check_fits(result).convert("RGB")
    assert decoded.getpixel((10, 10))[2] > 200  # blue half
    assert min(decoded.getpixel((decoded.width - 10, 10))) > 240  # was transparent -> white


def test_small_png_passes_through_unchanged():
    data = _encode(Image.new("RGB", (64, 48), (10, 20, 30)), "PNG")
    result = normalize_image(data, "image/png")
    assert not result.changed and result.data == data and result.mime == "image/png"
    assert result.note == ""


def test_tight_budget_lowers_quality_then_size():
    data = _encode(_noise((1500, 1000)), "PNG")
    result = normalize_image(data, "image/png", max_side=1568, max_b64=20_000)
    _check_fits(result, max_b64=20_000)
    assert max(result.width, result.height) < 1500  # had to shrink, not just re-encode


def test_non_images_are_rejected_with_a_clear_message():
    with pytest.raises(ImageError, match="not an image format"):
        normalize_image(b"%PDF-1.7 definitely not an image", "application/pdf")
