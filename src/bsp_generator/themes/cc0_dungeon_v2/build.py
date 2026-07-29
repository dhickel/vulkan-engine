#!/usr/bin/env python3
"""Deterministic procedural CC0 Dungeon v2 theme asset generator.

Generates a multi-palette WAD2 theme with three CC0 room palettes
(base_stone, crypt, treasury), a dedicated connector palette, and a
compiler-only skip texture.  Every visible texture has matching
basecolor, normal, and gloss companions at 1024x1024.

Outputs (placed in target directory, default CWD):
- palette.lmp                  — 256-colour project-authored palette (768 bytes)
- cc0_dungeon_v2.wad           — WAD2 archive
- theme.toml                   — palette declarations
- LICENSE                      — CC0 dedication
- textures/<name>_basecolor.png, <name>_norm.png, <name>_gloss.png

Pillow is required: pip install Pillow

Usage:
    python3 build.py [output_directory]
"""

from __future__ import annotations

import math
from pathlib import Path
import random
import struct
import sys

try:
    from PIL import Image, ImageChops, ImageFilter, ImageOps
except ImportError as error:  # pragma: no cover
    raise SystemExit(
        "CC0 Dungeon v2 generation requires Pillow: pip install Pillow"
    ) from error


TEXTURE_SIZE = 1024
SKIP_TEXTURE_SIZE = 64
MASTER_SEED = 0x43324447  # "C2DG" in ASCII (CC0 Dungeon v2 Generator)
PNG_SAVE_OPTIONS = {"format": "PNG", "compress_level": 9, "optimize": False}

# ── Texture definitions ────────────────────────────────────────────────────
# (name, base-RGB, height-gain, normal-strength, mean-gloss)

BASE_STONE = (
    ("bs_floor",   (91, 70, 51),   0.72, 1.30, 100),
    ("bs_wall",    (126, 128, 124), 0.60, 1.10, 96),
    ("bs_ceil",    (181, 178, 167), 0.44, 0.92, 104),
    ("bs_accent",  (168, 133, 94),  0.66, 1.20, 112),
)

CRYPT = (
    ("crypt_floor",   (48, 44, 40),   0.78, 1.35, 88),
    ("crypt_wall",    (76, 78, 81),   0.66, 1.18, 76),
    ("crypt_ceil",    (98, 100, 104), 0.50, 0.96, 92),
    ("crypt_accent",  (92, 84, 72),   0.70, 1.24, 96),
)

TREASURY = (
    ("treas_floor",   (132, 98, 62),  0.68, 1.26, 108),
    ("treas_wall",    (168, 152, 120), 0.56, 1.06, 100),
    ("treas_ceil",    (198, 188, 162), 0.40, 0.88, 112),
    ("treas_accent",  (186, 144, 82),  0.62, 1.16, 104),
)

CONNECTOR = (
    ("conn_floor",   (102, 98, 92),  0.64, 1.14, 96),
    ("conn_wall",    (140, 142, 138), 0.52, 1.02, 92),
    ("conn_ceil",    (188, 185, 176), 0.40, 0.90, 100),
)

ALL_VISIBLE = BASE_STONE + CRYPT + TREASURY + CONNECTOR

# ── Helpers ───────────────────────────────────────────────────────────────

def clamp(value: float | int) -> int:
    return max(0, min(255, int(round(value))))


# ═══════════════════════════════════════════════════════════════════════════
# Palette
# ═══════════════════════════════════════════════════════════════════════════

def make_palette() -> bytes:
    """Return a project-authored 256-entry dungeon palette.

    Entries 0..223 are muted stone ramps spanning greys, browns, and
    cool tones.  Entries 224..255 are vivid fullbrights and are avoided
    by generated albedo.
    """
    ramps = (
        ((9, 9, 9), (226, 224, 216)),       # neutral greys
        ((29, 22, 17), (150, 111, 76)),     # dark floor earth
        ((47, 33, 22), (183, 132, 86)),     # warm floor stone
        ((65, 47, 32), (204, 159, 112)),    # weathered brown
        ((38, 36, 34), (98, 96, 92)),       # dark crypt grey
        ((52, 54, 58), (108, 110, 116)),    # cool crypt blue-grey
        ((77, 70, 62), (164, 163, 156)),    # charcoal-grey wall
        ((88, 92, 91), (187, 190, 185)),    # cool wall slate
        ((109, 105, 96), (208, 204, 190)),  # limestone
        ((128, 119, 103), (220, 205, 176)), # warm limestone
        ((98, 72, 44), (186, 144, 82)),     # treasury gold
        ((127, 98, 69), (204, 161, 111)),   # carved warm accent
        ((144, 137, 123), (224, 218, 202)), # pale ceiling stone
        ((159, 151, 139), (238, 232, 216)), # light worn highlights
    )

    palette = bytearray()
    for start, end in ramps:
        for step in range(16):
            fraction = step / 15
            palette.extend(
                clamp(start[channel] + (end[channel] - start[channel]) * fraction)
                for channel in range(3)
            )

    for step in range(32):
        hue = step * 256 // 32
        if hue < 85:
            rgb = (255, hue * 3, 0)
        elif hue < 170:
            rgb = (255 - (hue - 85) * 3, 255, 0)
        else:
            rgb = (0, 255, (hue - 170) * 3)
        palette.extend(clamp(c) for c in rgb)

    assert len(palette) == 768
    return bytes(palette)


def palette_image(palette: bytes) -> Image.Image:
    image = Image.new("P", (1, 1))
    image.putpalette(palette)
    return image


# ═══════════════════════════════════════════════════════════════════════════
# Procedural height fields
# ═══════════════════════════════════════════════════════════════════════════

def periodic_noise(rng: random.Random, cells: int) -> Image.Image:
    """Return a tileable smooth noise field with a fixed 1024-pixel period."""
    values = [rng.randrange(256) for _ in range(cells * cells)]
    dim = cells + 1
    repeated = bytearray(dim * dim)
    for y in range(dim):
        src_y = (y % cells) * cells
        tgt_y = y * dim
        for x in range(dim):
            repeated[tgt_y + x] = values[src_y + (x % cells)]

    source = Image.frombytes("L", (dim, dim), bytes(repeated))
    return source.resize(
        (TEXTURE_SIZE + 1, TEXTURE_SIZE + 1), Image.Resampling.BICUBIC
    ).crop((0, 0, TEXTURE_SIZE, TEXTURE_SIZE))


def centred(image: Image.Image, gain: float) -> Image.Image:
    return image.point(
        [clamp(128 + (v - 128) * gain) for v in range(256)]
    )


def average(left: Image.Image, right: Image.Image) -> Image.Image:
    return ImageChops.add(left, right, scale=2)


def base_height(rng: random.Random, gain: float) -> Image.Image:
    """Combine large, medium, and fine periodic noise into a height field."""
    large = centred(periodic_noise(rng, 8), gain * 1.05)
    medium = centred(periodic_noise(rng, 32), gain * 0.65)
    fine = centred(periodic_noise(rng, 128), gain * 0.35)
    return average(average(large, medium), fine)


# ═══════════════════════════════════════════════════════════════════════════
# PBR maps
# ═══════════════════════════════════════════════════════════════════════════

def colourise(
    height: Image.Image,
    rng: random.Random,
    base_rgb: tuple[int, int, int],
    gain: float,
) -> Image.Image:
    """Turn the height field into restrained, role-specific stone albedo."""
    mottling = centred(periodic_noise(rng, 16), 0.33)
    tone = average(centred(height, 0.92), mottling)
    channel_gains = (gain * 1.06, gain, gain * 0.86)
    channels = [
        tone.point(
            [
                clamp(base_rgb[ch] + (v - 128) * channel_gains[ch])
                for v in range(256)
            ]
        )
        for ch in range(3)
    ]
    return Image.merge("RGB", tuple(channels))


def normal_map(height: Image.Image, strength: float) -> Image.Image:
    """Encode a tangent-space normal map from the height field."""
    source = height.tobytes()
    normal = bytearray(TEXTURE_SIZE * TEXTURE_SIZE * 3)
    dst = 0
    for y in range(TEXTURE_SIZE):
        prev_row = ((y - 1) % TEXTURE_SIZE) * TEXTURE_SIZE
        row = y * TEXTURE_SIZE
        next_row = ((y + 1) % TEXTURE_SIZE) * TEXTURE_SIZE
        for x in range(TEXTURE_SIZE):
            left = row + ((x - 1) % TEXTURE_SIZE)
            right = row + ((x + 1) % TEXTURE_SIZE)
            gx = source[right] - source[left]
            gy = source[next_row + x] - source[prev_row + x]
            normal[dst] = clamp(128 - gx * strength)
            normal[dst + 1] = clamp(128 - gy * strength)
            normal[dst + 2] = clamp(255 - (abs(gx) + abs(gy)) * strength * 0.65)
            dst += 3
    return Image.frombytes("RGB", (TEXTURE_SIZE, TEXTURE_SIZE), bytes(normal))


def gloss_map(
    height: Image.Image,
    rng: random.Random,
    mean_gloss: int,
) -> Image.Image:
    """Create spatially useful 0.30–0.50 stone gloss from the height field."""
    broad = height.filter(ImageFilter.GaussianBlur(radius=18))
    pores = periodic_noise(rng, 64)
    contour = ImageChops.subtract(height, broad, scale=1, offset=128)
    worn = ImageOps.autocontrast(broad)
    relief = ImageOps.autocontrast(contour)
    pore_detail = ImageOps.autocontrast(pores)
    variation = ImageOps.autocontrast(
        average(average(worn, relief), pore_detail)
    )

    curve = 1.0 + (102 - mean_gloss) / 50.0
    gloss = variation.point(
        [clamp(76 + 52 * ((v / 255.0) ** curve)) for v in range(256)]
    )
    # Verify range contract
    lo, hi = gloss.getextrema()
    from PIL import ImageStat
    dev = ImageStat.Stat(gloss).var[0] ** 0.5
    if not (76 <= lo and hi <= 128):
        print(f"  WARNING: gloss range [{lo}, {hi}] outside [76, 128]; clamping")
        gloss = gloss.point(
            [clamp(max(76, min(128, v))) for v in range(256)]
        )
    if dev < 4.0:
        print(f"  WARNING: gloss stddev={dev:.2f} is low (target >= 4.0)")

    return Image.merge("RGB", (gloss, gloss, gloss))


# ═══════════════════════════════════════════════════════════════════════════
# WAD2 writer
# ═══════════════════════════════════════════════════════════════════════════

def make_miptex(name: str, image: Image.Image, pal_img: Image.Image) -> bytes:
    """Build a Quake miptex from an RGB image using four indexed mip levels."""
    w, h = image.size
    if w != h or w < 8 or w & (w - 1):
        raise ValueError(f"{name}: miptex must be square power-of-two >= 8, got {image.size}")

    mips: list[bytes] = []
    level = image
    for _ in range(4):
        indexed = level.quantize(palette=pal_img, dither=Image.Dither.NONE)
        mips.append(indexed.tobytes())
        level = level.resize((level.width // 2, level.height // 2), Image.Resampling.BOX)

    offsets = []
    offset = 40
    for mip in mips:
        offsets.append(offset)
        offset += len(mip)

    encoded = name.encode("ascii")
    if len(encoded) > 15:
        raise ValueError(f"texture name too long: {name}")
    header = struct.pack(
        "<16sIIIIII",
        encoded.ljust(16, b"\0"),
        w,
        h,
        *offsets,
    )
    return header + b"".join(mips)


def make_wad2(entries: list[tuple[str, bytes]]) -> bytes:
    """Build an uncompressed WAD2 archive from named miptex byte blobs."""
    dir_offset = 12 + sum(len(data) for _, data in entries)
    header = struct.pack("<4sii", b"WAD2", len(entries), dir_offset)
    directory = bytearray()
    filepos = 12
    for name, data in entries:
        encoded = name.encode("ascii")
        if len(encoded) > 15:
            raise ValueError(f"texture name too long: {name}")
        directory.extend(
            struct.pack(
                "<iiIBBH16s",
                filepos,
                len(data),
                len(data),
                0x44,
                0,
                0,
                encoded.ljust(16, b"\0"),
            )
        )
        filepos += len(data)
    return header + b"".join(d for _, d in entries) + bytes(directory)


# ═══════════════════════════════════════════════════════════════════════════
# Static files
# ═══════════════════════════════════════════════════════════════════════════

def write_static_files(out_dir: Path) -> None:
    (out_dir / "theme.toml").write_text(
        "[package]\n"
        "name = \"cc0_dungeon_v2\"\n"
        "version = \"1.0.0\"\n"
        "wad = \"cc0_dungeon_v2.wad\"\n"
        "\n"
        "[palettes]\n"
        "base = \"base_stone\"\n"
        "connector = \"connector\"\n"
        "\n"
        "[palettes.entries]\n"
        "base_stone = [\"bs_floor\", \"bs_wall\", "
        "\"bs_ceil\", \"bs_accent\"]\n"
        "crypt = [\"crypt_floor\", \"crypt_wall\", "
        "\"crypt_ceil\", \"crypt_accent\"]\n"
        "treasury = [\"treas_floor\", \"treas_wall\", "
        "\"treas_ceil\", \"treas_accent\"]\n"
        "connector = [\"conn_floor\", \"conn_wall\", "
        "\"conn_ceil\"]\n"
        "\n"
        "[roles]\n"
        "floor = 0\n"
        "wall = 1\n"
        "ceiling = 2\n"
        "accent = 3\n"
        "\n"
        "[compiler]\n"
        "skip = \"skip\"\n"
        "skip_size = 64\n",
        encoding="utf-8",
    )
    (out_dir / "LICENSE").write_text(
        "CC0 1.0 Universal (CC0 1.0) Public Domain Dedication\n\n"
        "The person who associated a work with this deed has dedicated the work to the\n"
        "public domain by waiving all of his or her rights to the work worldwide under\n"
        "copyright law, including all related and neighboring rights, to the extent\n"
        "allowed by law.\n\n"
        "You can copy, modify, distribute and perform the work, even for commercial\n"
        "purposes, all without asking permission.\n\n"
        "Full license text: https://creativecommons.org/publicdomain/zero/1.0/legalcode\n\n"
        "Attribution (not required but appreciated):\n"
        '  "CC0 Dungeon v2 Theme" generated by the bsp_generator project.\n',
        encoding="utf-8",
    )


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    texture_dir = out_dir / "textures"
    texture_dir.mkdir(parents=True, exist_ok=True)

    palette = make_palette()
    (out_dir / "palette.lmp").write_bytes(palette)
    pal_for_quant = palette_image(palette)

    wad_entries: list[tuple[str, bytes]] = []
    for index, (name, base_rgb, colour_gain, normal_strength, mean_gloss) in enumerate(ALL_VISIBLE):
        role_rng = random.Random(MASTER_SEED + index * 0x9E3779B1)
        height = base_height(role_rng, colour_gain)
        base = colourise(height, role_rng, base_rgb, colour_gain)
        normal = normal_map(height, normal_strength)
        gloss = gloss_map(height, role_rng, mean_gloss)

        base.save(texture_dir / f"{name}_basecolor.png", **PNG_SAVE_OPTIONS)
        normal.save(texture_dir / f"{name}_norm.png", **PNG_SAVE_OPTIONS)
        gloss.save(texture_dir / f"{name}_gloss.png", **PNG_SAVE_OPTIONS)
        wad_entries.append((name, make_miptex(name, base, pal_for_quant)))

    # Compiler-only skip texture (64x64, black)
    skip = Image.new("RGB", (SKIP_TEXTURE_SIZE, SKIP_TEXTURE_SIZE), (0, 0, 0))
    wad_entries.append(("skip", make_miptex("skip", skip, pal_for_quant)))

    (out_dir / "cc0_dungeon_v2.wad").write_bytes(make_wad2(wad_entries))
    write_static_files(out_dir)

    print(f"CC0 Dungeon v2 theme generated in {out_dir.resolve()}")


if __name__ == "__main__":
    main()
