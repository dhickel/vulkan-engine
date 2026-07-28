#!/usr/bin/env python3
"""Deterministic procedural CC0 Stone Beta theme asset generator.

The generator creates every theme asset from fixed seeds; it does not sample or
transform third-party artwork. The generated pixels and the accompanying WAD,
palette, and PBR maps are dedicated to CC0 by ``LICENSE``.

Outputs (placed in the target directory, default CWD):
- palette.lmp              — 256-colour project-authored palette (768 bytes)
- cc0_stone_beta.wad       — WAD2 archive with 1024² visual miptex entries
                              and one 64² compiler-only ``skip`` miptex
- textures/<role>_basecolor.png, <role>_norm.png, <role>_gloss.png

Pillow is required to generate and palette-quantise the PNG and WAD images.

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
    from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageOps, ImageStat
except ImportError as error:  # pragma: no cover - exercised by build hosts
    raise SystemExit("CC0 Stone Beta generation requires Pillow: pip install Pillow") from error


TEXTURE_SIZE = 1024
SKIP_TEXTURE_SIZE = 64
MASTER_SEED = 0x43433053  # "CC0S" in ASCII
PNG_SAVE_OPTIONS = {"format": "PNG", "compress_level": 9, "optimize": False}

# name, RGB base, height gain, normal strength, mean gloss
#
# The BSP PBR shader converts gloss to roughness with ``1 - gloss``. Keep the
# stone roles inside the useful 0.30–0.50 gloss band (0.50–0.70 roughness),
# rather than the old near-black 0.10–0.25 gloss range that blurred specular
# IBL into an indistinguishable flat wash.
TEXTURE_DEFS = (
    ("stone_floor", (91, 70, 51), 0.72, 1.30, 100),
    ("stone_wall", (126, 128, 124), 0.60, 1.10, 96),
    ("stone_ceiling", (181, 178, 167), 0.44, 0.92, 104),
    ("stone_accent", (168, 133, 94), 0.66, 1.20, 112),
)


def clamp(value: float | int) -> int:
    return max(0, min(255, int(round(value))))


# ═══════════════════════════════════════════════════════════════════════════
# Palette
# ═══════════════════════════════════════════════════════════════════════════

def make_palette() -> bytes:
    """Return a project-authored 256-entry stone palette.

    Entries 0..223 are deliberately arranged as muted stone ramps. Entries
    224..255 remain vivid fullbrights and are avoided by the generated albedo.
    """
    ramps = (
        ((9, 9, 9), (226, 224, 216)),       # neutral greys
        ((29, 22, 17), (150, 111, 76)),     # dark floor earth
        ((47, 33, 22), (183, 132, 86)),     # warm floor stone
        ((65, 47, 32), (204, 159, 112)),    # weathered brown
        ((77, 70, 62), (164, 163, 156)),    # charcoal-grey wall
        ((88, 92, 91), (187, 190, 185)),    # cool wall slate
        ((109, 105, 96), (208, 204, 190)),  # limestone
        ((128, 119, 103), (220, 205, 176)), # warm limestone
        ((46, 54, 50), (119, 133, 119)),    # subdued moss tint
        ((65, 70, 71), (145, 151, 151)),    # blue-grey stone
        ((102, 88, 72), (181, 157, 126)),   # accent sandstone
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

    # Keep the final two rows reserved for fullbright material semantics.
    for step in range(32):
        hue = step * 256 // 32
        if hue < 85:
            rgb = (255, hue * 3, 0)
        elif hue < 170:
            rgb = (255 - (hue - 85) * 3, 255, 0)
        else:
            rgb = (0, 255, (hue - 170) * 3)
        palette.extend(clamp(component) for component in rgb)

    assert len(palette) == 768
    return bytes(palette)


def palette_image(palette: bytes) -> Image.Image:
    image = Image.new("P", (1, 1))
    image.putpalette(palette)
    return image


# ═══════════════════════════════════════════════════════════════════════════
# Tileable procedural height fields
# ═══════════════════════════════════════════════════════════════════════════

def periodic_noise(rng: random.Random, cells: int) -> Image.Image:
    """Return a tileable smooth noise field with a fixed 1024px period."""
    values = [rng.randrange(256) for _ in range(cells * cells)]
    dimension = cells + 1
    repeated = bytearray(dimension * dimension)
    for y in range(dimension):
        source_y = (y % cells) * cells
        target_y = y * dimension
        for x in range(dimension):
            repeated[target_y + x] = values[source_y + (x % cells)]

    # The matching final row/column makes the resampled field periodic at the
    # texture boundary. Crop the duplicate edge back to the WAD dimensions.
    source = Image.frombytes("L", (dimension, dimension), bytes(repeated))
    return source.resize(
        (TEXTURE_SIZE + 1, TEXTURE_SIZE + 1), Image.Resampling.BICUBIC
    ).crop((0, 0, TEXTURE_SIZE, TEXTURE_SIZE))


def centred(image: Image.Image, gain: float) -> Image.Image:
    return image.point([clamp(128 + (value - 128) * gain) for value in range(256)])


def average(left: Image.Image, right: Image.Image) -> Image.Image:
    return ImageChops.add(left, right, scale=2)


def base_height(rng: random.Random, gains: tuple[float, float, float]) -> Image.Image:
    """Combine large, medium, and fine periodic noise into a stone field."""
    large = centred(periodic_noise(rng, 8), gains[0])
    medium = centred(periodic_noise(rng, 32), gains[1])
    fine = centred(periodic_noise(rng, 128), gains[2])
    return average(average(large, medium), fine)


def wrapped_line(draw: ImageDraw.ImageDraw, points: list[tuple[int, int]], fill: int, width: int) -> None:
    """Draw a line and its neighbouring tile copies for seam-safe cracks."""
    for offset_x in (-TEXTURE_SIZE, 0, TEXTURE_SIZE):
        for offset_y in (-TEXTURE_SIZE, 0, TEXTURE_SIZE):
            draw.line(
                [(x + offset_x, y + offset_y) for x, y in points],
                fill=fill,
                width=width,
                joint="curve",
            )


def wrapped_ellipse(
    draw: ImageDraw.ImageDraw, x: int, y: int, radius: int, fill: int
) -> None:
    for offset_x in (-TEXTURE_SIZE, 0, TEXTURE_SIZE):
        for offset_y in (-TEXTURE_SIZE, 0, TEXTURE_SIZE):
            draw.ellipse(
                (
                    x - radius + offset_x,
                    y - radius + offset_y,
                    x + radius + offset_x,
                    y + radius + offset_y,
                ),
                fill=fill,
            )


def crack_path(rng: random.Random, start: tuple[int, int], segments: int, step: int) -> list[tuple[int, int]]:
    points = [start]
    angle = rng.uniform(0.0, math.tau)
    x, y = start
    for _ in range(segments):
        angle += rng.uniform(-0.75, 0.75)
        distance = rng.randint(step // 2, step)
        x += int(math.cos(angle) * distance)
        y += int(math.sin(angle) * distance)
        points.append((x, y))
    return points


def add_floor_masonry(height: Image.Image, rng: random.Random) -> None:
    """Lay irregular broad flagstones, recessed grout, and surface chips."""
    draw = ImageDraw.Draw(height)
    row_height = 256

    for row, y in enumerate(range(0, TEXTURE_SIZE + 1, row_height)):
        if y in (0, TEXTURE_SIZE):
            points = [(0, y), (TEXTURE_SIZE, y)]
        else:
            points = [(0, y)]
            for x in range(96, TEXTURE_SIZE, 96):
                points.append((x, y + rng.randint(-11, 11)))
            points.append((TEXTURE_SIZE, y))
        draw.line(points, fill=42, width=13, joint="curve")
        draw.line(points, fill=70, width=3, joint="curve")

        if row == 4:
            continue
        offset = 58 if row % 2 else 176
        for x in range(offset, TEXTURE_SIZE, 250):
            seam_x = x + rng.randint(-24, 24)
            points = [(seam_x, y), (seam_x + rng.randint(-10, 10), y + row_height)]
            draw.line(points, fill=45, width=12)
            draw.line([(px + 4, py) for px, py in points], fill=83, width=2)

    for _ in range(115):
        points = crack_path(
            rng,
            (rng.randrange(TEXTURE_SIZE), rng.randrange(TEXTURE_SIZE)),
            rng.randint(2, 5),
            rng.randint(14, 36),
        )
        wrapped_line(draw, points, fill=rng.randint(37, 62), width=rng.choice((1, 1, 2)))

    for _ in range(850):
        wrapped_ellipse(
            draw,
            rng.randrange(TEXTURE_SIZE),
            rng.randrange(TEXTURE_SIZE),
            rng.choice((1, 1, 2, 3)),
            rng.randint(72, 112),
        )


def add_wall_masonry(height: Image.Image, rng: random.Random) -> None:
    """Build weathered ashlar courses with chips and vertical water marks."""
    draw = ImageDraw.Draw(height)
    course_height = 146

    for row, y in enumerate(range(0, TEXTURE_SIZE + 1, course_height)):
        y = min(y, TEXTURE_SIZE)
        draw.line([(0, y), (TEXTURE_SIZE, y)], fill=49, width=9)
        draw.line([(0, y + 3), (TEXTURE_SIZE, y + 3)], fill=91, width=2)
        if row == 7:
            continue
        offset = 90 if row % 2 else 235
        for x in range(offset, TEXTURE_SIZE, 275):
            seam_x = x + rng.randint(-22, 22)
            top = y
            bottom = min(TEXTURE_SIZE, y + course_height)
            draw.line([(seam_x, top), (seam_x + rng.randint(-6, 6), bottom)], fill=52, width=8)
            draw.line([(seam_x + 3, top + 4), (seam_x + 3, bottom - 4)], fill=94, width=1)

    for _ in range(320):
        x = rng.randrange(TEXTURE_SIZE)
        y = rng.randrange(TEXTURE_SIZE)
        width = rng.randint(2, 8)
        height_px = rng.randint(2, 11)
        draw.ellipse((x - width, y - height_px, x + width, y + height_px), fill=rng.randint(48, 95))

    for _ in range(32):
        x = rng.randrange(TEXTURE_SIZE)
        top = rng.randrange(TEXTURE_SIZE)
        length = rng.randint(35, 190)
        wrapped_line(
            draw,
            [(x, top), (x + rng.randint(-8, 8), top + length)],
            fill=rng.randint(60, 82),
            width=rng.choice((1, 2, 3)),
        )


def add_ceiling_roughness(height: Image.Image, rng: random.Random) -> None:
    """Create rough pale stone with broad slab seams, pores, and fine fractures."""
    draw = ImageDraw.Draw(height)

    for y in (0, 340, 680, TEXTURE_SIZE):
        points = [(0, y)]
        for x in range(128, TEXTURE_SIZE, 128):
            points.append((x, y + (0 if y in (0, TEXTURE_SIZE) else rng.randint(-18, 18))))
        points.append((TEXTURE_SIZE, y))
        draw.line(points, fill=77, width=5, joint="curve")

    for _ in range(95):
        points = crack_path(
            rng,
            (rng.randrange(TEXTURE_SIZE), rng.randrange(TEXTURE_SIZE)),
            rng.randint(2, 5),
            rng.randint(17, 48),
        )
        wrapped_line(draw, points, fill=rng.randint(66, 92), width=1)

    for _ in range(1_450):
        wrapped_ellipse(
            draw,
            rng.randrange(TEXTURE_SIZE),
            rng.randrange(TEXTURE_SIZE),
            rng.choice((1, 1, 2, 2, 3, 4)),
            rng.randint(84, 119),
        )


def add_accent_carving(height: Image.Image, rng: random.Random) -> None:
    """Create warm dressed stone with inset borders and worn chisel marks."""
    draw = ImageDraw.Draw(height)
    for offset in (0, TEXTURE_SIZE):
        draw.rectangle((offset - 12, 0, offset + 12, TEXTURE_SIZE), fill=49)
        draw.rectangle((0, offset - 12, TEXTURE_SIZE, offset + 12), fill=49)

    for y in range(128, TEXTURE_SIZE, 256):
        for x in range(128, TEXTURE_SIZE, 256):
            inset = rng.randint(22, 36)
            draw.rounded_rectangle(
                (x - 112, y - 112, x + 112, y + 112),
                radius=18,
                outline=62,
                width=10,
            )
            draw.rounded_rectangle(
                (x - 112 + inset, y - 112 + inset, x + 112 - inset, y + 112 - inset),
                radius=10,
                outline=92,
                width=3,
            )

    for _ in range(310):
        points = crack_path(
            rng,
            (rng.randrange(TEXTURE_SIZE), rng.randrange(TEXTURE_SIZE)),
            rng.randint(1, 3),
            rng.randint(10, 28),
        )
        wrapped_line(draw, points, fill=rng.randint(45, 73), width=rng.choice((1, 1, 2)))


def role_height(name: str, rng: random.Random) -> Image.Image:
    if name == "stone_floor":
        height = base_height(rng, (0.68, 0.45, 0.28))
        add_floor_masonry(height, rng)
    elif name == "stone_wall":
        height = base_height(rng, (0.54, 0.40, 0.32))
        add_wall_masonry(height, rng)
    elif name == "stone_ceiling":
        height = base_height(rng, (0.48, 0.35, 0.24))
        add_ceiling_roughness(height, rng)
    elif name == "stone_accent":
        height = base_height(rng, (0.57, 0.38, 0.26))
        add_accent_carving(height, rng)
    else:  # pragma: no cover - guarded by TEXTURE_DEFS
        raise ValueError(f"unknown stone role: {name}")
    return height


# ═══════════════════════════════════════════════════════════════════════════
# PBR maps
# ═══════════════════════════════════════════════════════════════════════════

def colourise(height: Image.Image, rng: random.Random, base: tuple[int, int, int], gain: float) -> Image.Image:
    """Turn the shared relief map into a restrained, role-specific stone albedo."""
    mottling = centred(periodic_noise(rng, 16), 0.33)
    tone = average(centred(height, 0.92), mottling)
    channel_gains = (gain * 1.06, gain, gain * 0.86)
    channels = [
        tone.point(
            [clamp(base[channel] + (value - 128) * channel_gains[channel]) for value in range(256)]
        )
        for channel in range(3)
    ]
    return Image.merge("RGB", tuple(channels))


def normal_map(height: Image.Image, strength: float) -> Image.Image:
    """Encode a tangent-space normal map from the procedural height field."""
    source = height.tobytes()
    normal = bytearray(TEXTURE_SIZE * TEXTURE_SIZE * 3)
    destination = 0
    for y in range(TEXTURE_SIZE):
        previous_row = ((y - 1) % TEXTURE_SIZE) * TEXTURE_SIZE
        row = y * TEXTURE_SIZE
        next_row = ((y + 1) % TEXTURE_SIZE) * TEXTURE_SIZE
        for x in range(TEXTURE_SIZE):
            left = row + ((x - 1) % TEXTURE_SIZE)
            right = row + ((x + 1) % TEXTURE_SIZE)
            gradient_x = source[right] - source[left]
            gradient_y = source[next_row + x] - source[previous_row + x]
            encoded_x = clamp(128 - gradient_x * strength)
            encoded_y = clamp(128 - gradient_y * strength)
            normal[destination] = encoded_x
            normal[destination + 1] = encoded_y
            # The renderer reconstructs +Z from R/G. Keep B meaningful for
            # external texture viewers while preserving the tangent-space slope.
            normal[destination + 2] = clamp(255 - (abs(gradient_x) + abs(gradient_y)) * strength * 0.65)
            destination += 3
    return Image.frombytes("RGB", (TEXTURE_SIZE, TEXTURE_SIZE), bytes(normal))


def gloss_map(height: Image.Image, rng: random.Random, mean_gloss: int) -> Image.Image:
    """Create spatially useful 0.30–0.50 stone gloss from the relief field.

    Dark cracks, pores, and high-frequency relief remain rougher while broad,
    worn stone gets a modestly glossier response.  Autocontrast is intentional:
    the procedural height fields have role-dependent variance, but the PBR
    range is an authored material contract rather than an accidental output of
    a particular height-map contrast level.
    """
    broad = height.filter(ImageFilter.GaussianBlur(radius=18))
    pores = periodic_noise(rng, 64)
    contour = ImageChops.subtract(height, broad, scale=1, offset=128)
    worn = ImageOps.autocontrast(broad)
    relief = ImageOps.autocontrast(contour)
    pore_detail = ImageOps.autocontrast(pores)
    variation = ImageOps.autocontrast(average(average(worn, relief), pore_detail))

    # 76/255 ≈ 0.30 and 128/255 ≈ 0.50. A gentle role-specific curve shifts
    # the average without allowing any texel to leave the authored PBR range.
    curve = 1.0 + (102 - mean_gloss) / 50.0
    gloss = variation.point(
        [clamp(76 + 52 * ((value / 255.0) ** curve)) for value in range(256)]
    )
    low, high = gloss.getextrema()
    deviation = ImageStat.Stat(gloss).var[0] ** 0.5
    assert 76 <= low <= high <= 128, (low, high)
    assert deviation >= 8.0, f"gloss variation unexpectedly flat: stddev={deviation:.2f}"
    return Image.merge("RGB", (gloss, gloss, gloss))


# ═══════════════════════════════════════════════════════════════════════════
# WAD2 writer
# ═══════════════════════════════════════════════════════════════════════════

def make_miptex(name: str, image: Image.Image, palette: Image.Image) -> bytes:
    """Build a Quake miptex from an RGB image using four indexed mip levels."""
    width, height = image.size
    if width != height or width < 8 or width & (width - 1):
        raise ValueError(f"{name}: miptex size must be square power-of-two >= 8, got {image.size}")

    mips: list[bytes] = []
    level = image
    for _ in range(4):
        indexed = level.quantize(palette=palette, dither=Image.Dither.NONE)
        mips.append(indexed.tobytes())
        level = level.resize((level.width // 2, level.height // 2), Image.Resampling.BOX)

    offsets = []
    offset = 40
    for mip in mips:
        offsets.append(offset)
        offset += len(mip)

    encoded_name = name.encode("ascii")
    if len(encoded_name) > 15:
        raise ValueError(f"texture name too long: {name}")
    header = struct.pack(
        "<16sIIIIII",
        encoded_name.ljust(16, b"\0"),
        width,
        height,
        *offsets,
    )
    return header + b"".join(mips)


def make_wad2(entries: list[tuple[str, bytes]]) -> bytes:
    """Build an uncompressed WAD2 archive from named miptex byte blobs."""
    directory_offset = 12 + sum(len(data) for _, data in entries)
    header = struct.pack("<4sii", b"WAD2", len(entries), directory_offset)
    directory = bytearray()
    file_position = 12
    for name, data in entries:
        encoded_name = name.encode("ascii")
        if len(encoded_name) > 15:
            raise ValueError(f"texture name too long: {name}")
        directory.extend(
            struct.pack(
                "<iiIBBH16s",
                file_position,
                len(data),
                len(data),
                0x44,  # WAD2 miptex lump
                0,     # no compression
                0,
                encoded_name.ljust(16, b"\0"),
            )
        )
        file_position += len(data)
    return header + b"".join(data for _, data in entries) + bytes(directory)


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def write_static_files(out_dir: Path) -> None:
    (out_dir / "theme.toml").write_text(
        "[roles]\n"
        "floor = \"stone_floor\"\n"
        "wall = \"stone_wall\"\n"
        "ceiling = \"stone_ceiling\"\n"
        "accent = \"stone_accent\"\n",
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
        "  \"CC0 Stone Beta Theme\" generated by the bsp_generator project.\n",
        encoding="utf-8",
    )


def main() -> None:
    out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    texture_dir = out_dir / "textures"
    texture_dir.mkdir(parents=True, exist_ok=True)

    palette = make_palette()
    (out_dir / "palette.lmp").write_bytes(palette)
    palette_for_quantisation = palette_image(palette)

    wad_entries: list[tuple[str, bytes]] = []
    for index, (name, base_rgb, colour_gain, normal_strength, mean_gloss) in enumerate(TEXTURE_DEFS):
        role_rng = random.Random(MASTER_SEED + index * 0x9E3779B1)
        height = role_height(name, role_rng)
        base = colourise(height, role_rng, base_rgb, colour_gain)
        normal = normal_map(height, normal_strength)
        gloss = gloss_map(height, role_rng, mean_gloss)

        base.save(texture_dir / f"{name}_basecolor.png", **PNG_SAVE_OPTIONS)
        normal.save(texture_dir / f"{name}_norm.png", **PNG_SAVE_OPTIONS)
        gloss.save(texture_dir / f"{name}_gloss.png", **PNG_SAVE_OPTIONS)
        wad_entries.append((name, make_miptex(name, base, palette_for_quantisation)))

    # ericw-tools requests the conventional compiler-only `skip` material
    # while creating hulls. It remains a compact non-rendered 64² miptex.
    skip = Image.new("RGB", (SKIP_TEXTURE_SIZE, SKIP_TEXTURE_SIZE), (0, 0, 0))
    wad_entries.append(("skip", make_miptex("skip", skip, palette_for_quantisation)))
    (out_dir / "cc0_stone_beta.wad").write_bytes(make_wad2(wad_entries))
    write_static_files(out_dir)

    print(f"CC0 Stone Beta theme generated in {out_dir.resolve()}")


if __name__ == "__main__":
    main()
