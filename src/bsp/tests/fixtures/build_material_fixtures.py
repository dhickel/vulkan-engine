#!/usr/bin/env python3
"""
build_material_fixtures.py — generate CC0 PBR companion textures for BSP
material evidence tests.

Produces 64×64 8-bit-per-channel PNG companions for project-authored WAD
texture identities. Each companion file carries the expected PBR suffix so
the discover_pbr_texture_companions() path in resources.rs can resolve them.

Outputs (written to --output-dir, by default ../textures/):
  - WALL01_basecolor.png   : CC0 procedural brick-like albedo
  - WALL01_roughness.png   : uniform roughness (inverse of gloss)
  - WALL01_norm.png        : flat tangent-space normal (128, 128, 255)
  - WALL01_gloss.png       : medium gloss (complement of roughness)

Usage:
    python3 build_material_fixtures.py [--output-dir DIR]

Requirements:
    - Python 3.8+
    - No third-party packages (pure Python struct + zlib PNG writer)
"""

from __future__ import annotations

import argparse
import struct
import zlib
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "textures"

# ── Minimal PNG writer (no external deps) ──────────────────────────────────


def _chunk(chunk_type: bytes, data: bytes) -> bytes:
    """Return a complete PNG chunk (length + type + data + crc)."""
    raw = chunk_type + data
    crc = struct.pack(">I", zlib.crc32(raw) & 0xFFFFFFFF)
    return struct.pack(">I", len(data)) + raw + crc


def write_png(
    path: Path,
    pixels: bytes,
    width: int = 64,
    height: int = 64,
    bit_depth: int = 8,
    color_type: int = 2,
) -> None:
    """Write a minimal 8-bit RGB or grayscale PNG file."""
    assert len(pixels) == width * height * (3 if color_type == 2 else 1)

    signature = b"\x89PNG\r\n\x1a\n"
    ihdr_data = struct.pack(
        ">IIBBBBB",
        width,
        height,
        bit_depth,
        color_type,
        0,  # compression
        0,  # filter
        0,  # interlace
    )
    ihdr = _chunk(b"IHDR", ihdr_data)

    # Filter byte (0 = None) per row
    raw = b""
    row_bytes = width * (3 if color_type == 2 else 1)
    for row_start in range(0, len(pixels), row_bytes):
        raw += b"\x00" + pixels[row_start : row_start + row_bytes]

    idat = _chunk(b"IDAT", zlib.compress(raw))
    iend = _chunk(b"IEND", b"")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(signature + ihdr + idat + iend)


# ── Procedural texture generators ──────────────────────────────────────────


def generate_basecolor(width: int = 64, height: int = 64) -> bytes:
    """CC0 procedural brick-like albedo: warm beige/brown checker with noise."""
    import random

    rng = random.Random(42)
    pixels = bytearray()
    for y in range(height):
        for x in range(width):
            # Brick pattern with subtle variation
            is_mortar = (x % 16 < 2) or (y % 16 < 2)
            if is_mortar:
                r, g, b = 80, 75, 70  # dark mortar
            else:
                r = 180 + rng.randint(-15, 15)
                g = 150 + rng.randint(-15, 15)
                b = 120 + rng.randint(-15, 15)
            pixels.extend([r, g, b])
    return bytes(pixels)


def generate_normal_flat(width: int = 64, height: int = 64) -> bytes:
    """Flat tangent-space normal map: (128, 128, 255) for every pixel."""
    pixels = bytearray()
    for _ in range(width * height):
        pixels.extend([128, 128, 255])
    return bytes(pixels)


def generate_gloss(width: int = 64, height: int = 64) -> bytes:
    """Medium gloss: uniform gray level (inverse of roughness)."""
    pixels = bytearray()
    for _ in range(width * height):
        # gloss ~0.5 → 128 in linear-ish sRGB space
        pixels.extend([128, 128, 128])
    return bytes(pixels)


def generate_roughness(width: int = 64, height: int = 64) -> bytes:
    """Uniform roughness: complement of gloss, a separate single-channel test
    companion for future __roughness suffix discovery."""
    # Grayscale PNG (color_type=0), single channel, value ~128 (medium roughness)
    return bytes([128] * (width * height))


# ── Texture identity ───────────────────────────────────────────────────────


TEXTURE_ID = "WALL01"
PBR_COMPANIONS = [
    ("_basecolor.png", generate_basecolor, 2),  # RGB
    ("_roughness.png", generate_roughness, 0),  # Grayscale
    ("_norm.png", generate_normal_flat, 2),  # RGB
    ("_gloss.png", generate_gloss, 2),  # RGB
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate CC0 PBR companion textures for BSP material fixtures"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for PNG textures (default: ../textures/)",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for suffix, generator_fn, color_type in PBR_COMPANIONS:
        filename = f"{TEXTURE_ID}{suffix}"
        path = args.output_dir / filename
        pixels = generator_fn()
        write_png(
            path,
            pixels,
            width=64,
            height=64,
            color_type=color_type,
        )
        print(f"  wrote {path} ({len(pixels)} bytes pixel data)")

    print(f"\nDone: {len(PBR_COMPANIONS)} PBR companions for '{TEXTURE_ID}'")


if __name__ == "__main__":
    main()
