#!/usr/bin/env python3
"""Deterministic procedural Richness Ancient v1 theme asset generator.

Generates a complete CC0 Ancient theme with cyclopean masonry,
post-and-lintel portal, tholos/megaron accents, hearth/bench/cist prop,
restrained warm emissive, and a compiler-only skip texture. Every visible
texture has matching basecolor, normal, and gloss companions.

Outputs (placed in target directory, default CWD):
- palette.lmp                     — 256-colour project-authored palette (768 bytes)
- richness_ancient_v1.wad         — WAD2 archive with 9 visible miptex entries
                                    and one compiler-only ``skip`` miptex
- theme.toml                      — semantic role declarations
- LICENSE                         — CC0 dedication
- provenance.toml                 — deterministic build inputs and hashes
- textures/<role>_basecolor.png, <role>_norm.png, <role>_gloss.png  (27 PNGs)

Pillow is required: pip install Pillow

Usage:
    python3 build.py [output_directory]
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
import random
import struct
import sys

try:
    from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageOps
except ImportError as error:  # pragma: no cover
    raise SystemExit(
        "Richness Ancient v1 generation requires Pillow: pip install Pillow"
    ) from error


TEXTURE_SIZE = 256
SKIP_TEXTURE_SIZE = 64
# "RAV1" in ASCII — Richness Ancient V1
MASTER_SEED = 0x52415631
PNG_SAVE_OPTIONS = {"format": "PNG", "compress_level": 9, "optimize": False}

# ── Texture definitions ────────────────────────────────────────────────────
# (name, base-RGB, height-gain, normal-strength, mean-gloss)

ANCIENT_DEFS = (
    ("wall",     (148, 136, 118), 0.72, 1.30, 92),
    ("floor",    (128, 112, 92),  0.78, 1.35, 84),
    ("ceiling",  (172, 164, 150), 0.48, 0.92, 100),
    ("accent",   (176, 154, 120), 0.60, 1.12, 104),
    ("portal",   (156, 142, 124), 0.68, 1.20, 88),
    ("vertical", (140, 130, 118), 0.64, 1.16, 96),
    ("cave",     (96, 88, 78),    0.84, 1.40, 72),
    ("prop",     (162, 140, 112), 0.54, 1.06, 108),
    ("emissive", (176, 148, 98),  0.58, 1.10, 100),
)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def clamp(value: float | int) -> int:
    return max(0, min(255, int(round(value))))


# ═══════════════════════════════════════════════════════════════════════════
# Palette
# ═══════════════════════════════════════════════════════════════════════════

def make_palette() -> bytes:
    """Return a project-authored 256-entry ancient stone palette.

    Entries 0..223 are warm ancient-stone ramps (ochre, limestone, umber,
    sandstone, weathered grey).  Entries 224..255 are warm fullbrights
    (amber through gold to warm white) for emissive glow.
    """
    ramps = (
        ((12, 10, 8),   (220, 212, 198)),  # neutral warm greys
        ((36, 28, 20),  (156, 118, 80)),   # dark umber
        ((54, 38, 26),  (188, 138, 90)),   # warm earth brown
        ((72, 52, 36),  (208, 164, 114)),  # weathered ochre
        ((64, 58, 52),  (168, 162, 150)),  # charcoal warm grey
        ((92, 86, 78),  (192, 186, 174)),  # mid warm stone
        ((114, 108, 96), (216, 210, 194)), # pale limestone
        ((132, 122, 104),(224, 216, 188)), # warm sandstone
        ((82, 78, 72),  (178, 172, 160)),  # cool-toned ancient grey
        ((104, 96, 84), (196, 186, 164)),  # olive-toned stone
        ((120, 106, 88),(210, 192, 158)),  # golden limestone
        ((144, 128, 104),(228, 212, 178)), # carved warm accent
        ((158, 146, 128),(236, 226, 204)), # pale dressed stone
        ((168, 158, 142),(242, 234, 216)), # light worn cap
    )

    palette = bytearray()
    for start, end in ramps:
        for step in range(16):
            fraction = step / 15
            palette.extend(
                clamp(start[channel] + (end[channel] - start[channel]) * fraction)
                for channel in range(3)
            )

    # Warm fullbright entries (224–255): amber through gold to warm white.
    # These are the emissive glow colours that Quake lightmapping leaves
    # unmodulated (fullbright).
    fullbright_ramp = (
        (192, 96, 16),    # deep amber
        (224, 128, 32),   # amber
        (240, 160, 48),   # golden amber
        (248, 184, 64),   # warm gold
        (252, 200, 80),   # light gold
        (254, 216, 104),  # pale gold
        (255, 228, 128),  # warm yellow
        (255, 240, 176),  # warm cream
    )
    for i in range(32):
        seg = i * (len(fullbright_ramp) - 1) / 31
        lo_idx = int(seg)
        frac = seg - lo_idx
        if lo_idx >= len(fullbright_ramp) - 1:
            r, g, b = fullbright_ramp[-1]
        else:
            r = clamp(fullbright_ramp[lo_idx][0] + (fullbright_ramp[lo_idx + 1][0] - fullbright_ramp[lo_idx][0]) * frac)
            g = clamp(fullbright_ramp[lo_idx][1] + (fullbright_ramp[lo_idx + 1][1] - fullbright_ramp[lo_idx][1]) * frac)
            b = clamp(fullbright_ramp[lo_idx][2] + (fullbright_ramp[lo_idx + 1][2] - fullbright_ramp[lo_idx][2]) * frac)
        palette.extend((r, g, b))

    assert len(palette) == 768
    return bytes(palette)


def palette_image(palette: bytes) -> Image.Image:
    image = Image.new("P", (1, 1))
    image.putpalette(palette)
    return image


# ═══════════════════════════════════════════════════════════════════════════
# Periodic noise and utility
# ═══════════════════════════════════════════════════════════════════════════

def periodic_noise(rng: random.Random, cells: int, size: int = TEXTURE_SIZE) -> Image.Image:
    """Return a tileable smooth noise field with a fixed period."""
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
        (size + 1, size + 1), Image.Resampling.BICUBIC
    ).crop((0, 0, size, size))


def centred(image: Image.Image, gain: float) -> Image.Image:
    return image.point(
        [clamp(128 + (v - 128) * gain) for v in range(256)]
    )


def average(left: Image.Image, right: Image.Image) -> Image.Image:
    return ImageChops.add(left, right, scale=2)


def base_height(rng: random.Random, gain: float, size: int = TEXTURE_SIZE) -> Image.Image:
    """Combine large, medium, and fine periodic noise into a height field."""
    large = centred(periodic_noise(rng, 8, size), gain * 1.05)
    medium = centred(periodic_noise(rng, 32, size), gain * 0.65)
    fine = centred(periodic_noise(rng, 128, size), gain * 0.35)
    return average(average(large, medium), fine)


def seamless_draw(draw: ImageDraw.ImageDraw, element_type: str, **kwargs) -> None:
    """Draw a primitive replicated across the 3×3 wrap grid for seamlessness."""
    S = TEXTURE_SIZE
    for ox in (-S, 0, S):
        for oy in (-S, 0, S):
            if element_type == "line":
                pts = [(x + ox, y + oy) for x, y in kwargs["points"]]
                draw.line(pts, fill=kwargs.get("fill", 0),
                          width=kwargs.get("width", 1),
                          joint=kwargs.get("joint", "curve"))
            elif element_type == "ellipse":
                draw.ellipse(
                    (kwargs["x0"] + ox, kwargs["y0"] + oy,
                     kwargs["x1"] + ox, kwargs["y1"] + oy),
                    fill=kwargs.get("fill"))
            elif element_type == "rectangle":
                draw.rectangle(
                    (kwargs["x0"] + ox, kwargs["y0"] + oy,
                     kwargs["x1"] + ox, kwargs["y1"] + oy),
                    fill=kwargs.get("fill"),
                    outline=kwargs.get("outline"),
                    width=kwargs.get("width", 0))
            elif element_type == "rounded_rectangle":
                draw.rounded_rectangle(
                    (kwargs["x0"] + ox, kwargs["y0"] + oy,
                     kwargs["x1"] + ox, kwargs["y1"] + oy),
                    radius=kwargs.get("radius", 0),
                    fill=kwargs.get("fill"),
                    outline=kwargs.get("outline"),
                    width=kwargs.get("width", 1))


# ═══════════════════════════════════════════════════════════════════════════
# Role height generators
# ═══════════════════════════════════════════════════════════════════════════

def add_cyclopean_wall(height: Image.Image, rng: random.Random) -> None:
    """Cyclopean masonry: large irregular polygonal blocks with thick grout.

    Uses a wrapping Voronoi-like pattern: each pixel measures distance to
    its nearest jittered cell centre.  Near boundaries between cells we
    recess deep grout lines.  Surface roughness and edge chips complete
    the ancient monumental feel.
    """
    S = TEXTURE_SIZE
    cells = 6
    # Deterministic jittered cell centres (wrapping)
    centres = []
    for cy in range(cells):
        for cx in range(cells):
            rx = rng.randint(-18, 18)
            ry = rng.randint(-18, 18)
            centres.append((cx * S / cells + rx + S / (cells * 2),
                            cy * S / cells + ry + S / (cells * 2)))

    # Compute per-pixel distance to nearest and second-nearest centre
    pixels = height.load()
    for y in range(S):
        for x in range(S):
            best = 1e9
            second = 1e9
            for cx, cy in centres:
                # Minimum distance considering 3x3 wrap
                dx = min(abs(x - cx), abs(x - cx - S), abs(x - cx + S))
                dy = min(abs(y - cy), abs(y - cy - S), abs(y - cy + S))
                d = dx * dx + dy * dy
                if d < best:
                    second = best
                    best = d
                elif d < second:
                    second = d
            # Boundary metric: how close is second-nearest relative to nearest
            boundary = 1.0 - best / max(second, 1.0)
            # Deep recess at strong boundaries, subtle at weak ones
            if boundary > 0.55:
                recess = int(60 * (boundary - 0.55) / 0.45)
                v = max(0, pixels[x, y] - recess)
            elif boundary > 0.25:
                recess = int(25 * (boundary - 0.25) / 0.30)
                v = max(0, pixels[x, y] - recess)
            else:
                v = pixels[x, y]
            pixels[x, y] = clamp(v)

    # Edge chips along block boundaries
    draw = ImageDraw.Draw(height)
    for _ in range(rng.randint(60, 90)):
        cx = rng.randrange(S)
        cy = rng.randrange(S)
        rx = rng.randint(1, 4)
        ry = rng.randint(1, 4)
        seamless_draw(draw, "ellipse",
                      x0=cx - rx, y0=cy - ry,
                      x1=cx + rx, y1=cy + ry,
                      fill=rng.randint(40, 75))


def add_cyclopean_floor(height: Image.Image, rng: random.Random) -> None:
    """Broad irregular flagstones with recessed grout and surface wear.

    Larger cells than the wall, horizontal emphasis, and scattered pitting
    give a cyclopean floor distinct from vertical surfaces.
    """
    S = TEXTURE_SIZE
    cells = 4
    centres = []
    for cy in range(cells):
        for cx in range(cells):
            rx = rng.randint(-28, 28)
            ry = rng.randint(-22, 22)
            centres.append((cx * S / cells + rx + S / (cells * 2),
                            cy * S / cells + ry + S / (cells * 2)))

    pixels = height.load()
    for y in range(S):
        for x in range(S):
            best = 1e9
            second = 1e9
            for cx, cy in centres:
                dx = min(abs(x - cx), abs(x - cx - S), abs(x - cx + S))
                dy = min(abs(y - cy), abs(y - cy - S), abs(y - cy + S))
                d = dx * dx + dy * dy
                if d < best:
                    second = best
                    best = d
                elif d < second:
                    second = d
            boundary = 1.0 - best / max(second, 1.0)
            if boundary > 0.50:
                recess = int(72 * (boundary - 0.50) / 0.50)
                v = max(0, pixels[x, y] - recess)
            elif boundary > 0.20:
                recess = int(30 * (boundary - 0.20) / 0.30)
                v = max(0, pixels[x, y] - recess)
            else:
                v = pixels[x, y]
            pixels[x, y] = clamp(v)

    # Surface wear pits
    draw = ImageDraw.Draw(height)
    for _ in range(rng.randint(120, 180)):
        cx = rng.randrange(S)
        cy = rng.randrange(S)
        rx = rng.randint(2, 6)
        ry = rng.randint(2, 6)
        seamless_draw(draw, "ellipse",
                      x0=cx - rx, y0=cy - ry,
                      x1=cx + rx, y1=cy + ry,
                      fill=rng.randint(50, 90))


def add_slab_ceiling(height: Image.Image, rng: random.Random) -> None:
    """Massive stone slab ceiling with broad horizontal seams and pores.

    The post-and-lintel structural logic puts the weight on the lintels;
    the ceiling expresses this with thick slab divisions, subtle roughness,
    and scattered pores from ancient stone.
    """
    S = TEXTURE_SIZE
    draw = ImageDraw.Draw(height)

    # Horizontal slab divisions
    for row in range(1, 5):
        y = row * S // 5 + rng.randint(-8, 8)
        y = max(8, min(S - 8, y))
        seamless_draw(draw, "line",
                      points=[(0, y), (S, y)],
                      fill=68, width=5, joint="curve")
        seamless_draw(draw, "line",
                      points=[(0, y + 2), (S, y + 2)],
                      fill=94, width=1, joint="curve")

    # Occasional vertical seams between slabs
    for col in range(1, 3):
        x = col * S // 3 + rng.randint(-20, 20)
        x = max(8, min(S - 8, x))
        # Staggered vertical seams - not full height
        for seg in range(3):
            y0 = seg * S // 3 + rng.randint(-10, 10)
            y1 = y0 + rng.randint(S // 6, S // 4)
            y0 = max(0, min(S, y0))
            y1 = max(0, min(S, y1))
            pts = [(x, y0), (x + rng.randint(-4, 4), y1)]
            seamless_draw(draw, "line", points=pts, fill=72, width=3)

    # Pores
    for _ in range(rng.randint(80, 130)):
        cx = rng.randrange(S)
        cy = rng.randrange(S)
        rr = rng.choice((1, 1, 2))
        seamless_draw(draw, "ellipse",
                      x0=cx - rr, y0=cy - rr,
                      x1=cx + rr, y1=cy + rr,
                      fill=rng.randint(72, 108))


def add_accent_dressed_stone(height: Image.Image, rng: random.Random) -> None:
    """Tholos/megaron dressed stone: refined courses with decorative panels.

    Combines horizontal ashlar courses with inset rectangular panels
    (megaron) or concentric circular motifs (tholos) for a dressed,
    ceremonial character.
    """
    S = TEXTURE_SIZE
    draw = ImageDraw.Draw(height)

    # Horizontal dressed courses
    course_h = S // 6
    for row in range(0, 7):
        y = row * course_h
        y = min(y, S)
        if y == 0 or y >= S:
            continue
        seamless_draw(draw, "line",
                      points=[(0, y), (S, y)],
                      fill=55, width=4, joint="curve")
        seamless_draw(draw, "line",
                      points=[(0, y + 1), (S, y + 1)],
                      fill=102, width=1, joint="curve")

    # Vertical joints (staggered per course)
    for row in range(6):
        y0 = row * course_h
        y1 = min(S, y0 + course_h)
        offset = rng.randint(16, 48) if row % 2 == 0 else rng.randint(64, 96)
        for col in range(0, 6):
            x = (offset + col * S // 5) % S
            x = max(4, min(S - 4, x))
            seamless_draw(draw, "line",
                          points=[(x, y0), (x + rng.randint(-3, 3), y1)],
                          fill=58, width=3)

    # Inset decorative panels (megaron rectangular + tholos concentric)
    # Central rectangular panel
    margin = 32
    seamless_draw(draw, "rounded_rectangle",
                  x0=margin, y0=margin, x1=S - margin, y1=S - margin,
                  radius=8, outline=64, fill=None, width=6)
    seamless_draw(draw, "rounded_rectangle",
                  x0=margin + 6, y0=margin + 6,
                  x1=S - margin - 6, y1=S - margin - 6,
                  radius=6, outline=96, fill=None, width=2)

    # Circular tholos medallion at centre
    cx, cy = S // 2, S // 2
    for r in (S // 6, S // 8, S // 12):
        seamless_draw(draw, "ellipse",
                      x0=cx - r, y0=cy - r,
                      x1=cx + r, y1=cy + r,
                      fill=None)
        # We draw circles manually for control
        for angle in range(0, 360, 15):
            rad = math.radians(angle)
            px = int(cx + r * math.cos(rad))
            py = int(cy + r * math.sin(rad))
            for ox in (-S, 0, S):
                for oy in (-S, 0, S):
                    draw.ellipse((px + ox - 2, py + oy - 2,
                                  px + ox + 2, py + oy + 2),
                                 fill=70 if r > S // 7 else 98)

    # Chisel marks
    for _ in range(rng.randint(40, 70)):
        x = rng.randrange(S)
        y = rng.randrange(S)
        angle = rng.uniform(0, math.tau)
        length = rng.randint(4, 14)
        ex = x + int(math.cos(angle) * length)
        ey = y + int(math.sin(angle) * length)
        for ox in (-S, 0, S):
            for oy in (-S, 0, S):
                draw.line([(x + ox, y + oy), (ex + ox, ey + oy)],
                          fill=rng.randint(50, 78), width=1)


def add_portal_post_lintel(height: Image.Image, rng: random.Random) -> None:
    """Post-and-lintel portal: massive horizontal lintel over vertical posts.

    A strong horizontal band across the upper third, vertical post divisions
    below, and deep shadow lines create the monumental gateway character.
    """
    S = TEXTURE_SIZE
    draw = ImageDraw.Draw(height)

    # Horizontal lintel band (upper third)
    lintel_y = S * 2 // 5 + rng.randint(-6, 6)
    lintel_y = max(S // 5, min(S * 3 // 5, lintel_y))
    lintel_bottom = lintel_y + S // 10

    # Deep shadow under lintel
    seamless_draw(draw, "line",
                  points=[(0, lintel_y), (S, lintel_y)],
                  fill=38, width=8, joint="curve")
    seamless_draw(draw, "line",
                  points=[(0, lintel_bottom), (S, lintel_bottom)],
                  fill=42, width=6, joint="curve")

    # Lintel top highlight
    seamless_draw(draw, "line",
                  points=[(0, lintel_y - 3), (S, lintel_y - 3)],
                  fill=108, width=2, joint="curve")

    # Vertical post divisions below lintel
    for post in range(3):
        if post == 0:
            x = S // 6 + rng.randint(-8, 8)
        elif post == 2:
            x = S * 5 // 6 + rng.randint(-8, 8)
        else:
            x = S // 2 + rng.randint(-12, 12)
        x = max(6, min(S - 6, x))
        seamless_draw(draw, "line",
                      points=[(x, lintel_bottom), (x + rng.randint(-3, 3), S)],
                      fill=40, width=7)
        seamless_draw(draw, "line",
                      points=[(x + 2, lintel_bottom + 4), (x + 2, S - 4)],
                      fill=100, width=1)

    # Vertical post highlight edge
    for post in range(3):
        if post == 0:
            x = S // 6 + rng.randint(-8, 8)
        elif post == 2:
            x = S * 5 // 6 + rng.randint(-8, 8)
        else:
            x = S // 2 + rng.randint(-12, 12)
        seamless_draw(draw, "line",
                      points=[(x - 4, lintel_bottom), (x - 4, S)],
                      fill=106, width=2)

    # Block divisions within lintel
    for bx in range(S // 40, S, S // 4):
        bx = bx + rng.randint(-10, 10)
        bx = max(4, min(S - 4, bx))
        seamless_draw(draw, "line",
                      points=[(bx, 0), (bx + rng.randint(-3, 3), lintel_y)],
                      fill=48, width=4)

    # Surface chips (portal)
    for _ in range(rng.randint(50, 80)):
        cx = rng.randrange(S)
        cy = rng.randrange(S)
        rr = rng.randint(1, 3)
        seamless_draw(draw, "ellipse",
                      x0=cx - rr, y0=cy - rr,
                      x1=cx + rr, y1=cy + rr,
                      fill=rng.randint(45, 82))


def add_vertical_pillar(height: Image.Image, rng: random.Random) -> None:
    """Vertical stone shaft/pillar with fluting-like vertical grooves.

    Strong vertical linear divisions, narrow tall block courses, and
    subtle fluting create a columnar, weight-bearing character.
    """
    S = TEXTURE_SIZE
    draw = ImageDraw.Draw(height)

    # Vertical fluting grooves
    for groove in range(5, S - 4, S // 7):
        gx = groove + rng.randint(-6, 6)
        gx = max(2, min(S - 2, gx))
        # Deep central groove
        seamless_draw(draw, "line",
                      points=[(gx, 0), (gx, S)],
                      fill=42, width=4)
        # Highlight edge
        seamless_draw(draw, "line",
                      points=[(gx + 2, 0), (gx + 2, S)],
                      fill=100, width=1)
        seamless_draw(draw, "line",
                      points=[(gx - 2, 0), (gx - 2, S)],
                      fill=108, width=1)

    # Horizontal block courses (narrow)
    course_h = S // 8
    for row in range(1, 8):
        y = row * course_h + rng.randint(-4, 4)
        y = max(4, min(S - 4, y))
        seamless_draw(draw, "line",
                      points=[(0, y), (S, y)],
                      fill=56, width=3, joint="curve")
        seamless_draw(draw, "line",
                      points=[(0, y + 1), (S, y + 1)],
                      fill=102, width=1, joint="curve")

    # Staggered vertical block joints
    for row in range(8):
        y0 = row * course_h
        y1 = min(S, y0 + course_h)
        offset = rng.randint(8, 24) if row % 2 == 0 else rng.randint(40, 60)
        for col in range(1, 6):
            x = (offset + col * S // 6) % S
            x = max(3, min(S - 3, x))
            seamless_draw(draw, "line",
                          points=[(x, y0), (x + rng.randint(-2, 2), y1)],
                          fill=50, width=2)

    # Surface chips (vertical)
    for _ in range(rng.randint(50, 80)):
        cx = rng.randrange(S)
        cy = rng.randrange(S)
        rr = rng.randint(1, 3)
        seamless_draw(draw, "ellipse",
                      x0=cx - rr, y0=cy - rr,
                      x1=cx + rr, y1=cy + rr,
                      fill=rng.randint(48, 84))


def add_cave_surface(height: Image.Image, rng: random.Random) -> None:
    """Rough natural cavern surface: organic, irregular, no masonry.

    Uses heavy multi-octave noise, cellular blobs, and deep crevices
    without any structured divisions.  The result is a raw excavated feel.
    """
    S = TEXTURE_SIZE
    draw = ImageDraw.Draw(height)

    # Organic blob-like recesses
    for _ in range(rng.randint(12, 20)):
        cx = rng.randrange(S)
        cy = rng.randrange(S)
        rx = rng.randint(16, 48)
        ry = rng.randint(16, 48)
        for ox in (-S, 0, S):
            for oy in (-S, 0, S):
                draw.ellipse(
                    (cx - rx + ox, cy - ry + oy,
                     cx + rx + ox, cy + ry + oy),
                    fill=rng.randint(38, 62))

    # Deep crevices
    for _ in range(rng.randint(24, 40)):
        sx = rng.randrange(S)
        sy = rng.randrange(S)
        angle = rng.uniform(0, math.tau)
        length = rng.randint(12, 60)
        points = [(sx, sy)]
        for seg in range(rng.randint(2, 5)):
            angle += rng.uniform(-1.0, 1.0)
            dist = rng.randint(length // 4, length // 2)
            sx = int(sx + math.cos(angle) * dist)
            sy = int(sy + math.sin(angle) * dist)
            points.append((sx % S, sy % S))
        for ox in (-S, 0, S):
            for oy in (-S, 0, S):
                draw.line([(px + ox, py + oy) for px, py in points],
                          fill=rng.randint(30, 52), width=rng.choice((2, 3, 4)),
                          joint="curve")

    # Surface nodules and bumps (lighter spots)
    for _ in range(rng.randint(60, 100)):
        cx = rng.randrange(S)
        cy = rng.randrange(S)
        rx = rng.randint(2, 8)
        ry = rng.randint(2, 8)
        for ox in (-S, 0, S):
            for oy in (-S, 0, S):
                draw.ellipse(
                    (cx - rx + ox, cy - ry + oy,
                     cx + rx + ox, cy + ry + oy),
                    fill=rng.randint(132, 172))

    # Scattered pits
    for _ in range(rng.randint(100, 160)):
        cx = rng.randrange(S)
        cy = rng.randrange(S)
        for ox in (-S, 0, S):
            for oy in (-S, 0, S):
                draw.ellipse(
                    (cx - 2 + ox, cy - 2 + oy,
                     cx + 2 + ox, cy + 2 + oy),
                    fill=rng.randint(30, 58))


def add_prop_worked_stone(height: Image.Image, rng: random.Random) -> None:
    """Hearth/bench/cist prop: smooth worked stone with tool marks.

    Refined surface with subtle parallel tool-mark grooves, gentle
    edge chamfers, and restrained wear — suitable for crafted furnishings.
    """
    S = TEXTURE_SIZE
    draw = ImageDraw.Draw(height)

    # Gentle edge chamfer (inset border)
    chamfer = 12
    seamless_draw(draw, "rectangle",
                  x0=chamfer, y0=chamfer,
                  x1=S - chamfer, y1=S - chamfer,
                  fill=0, outline=62, width=5)
    seamless_draw(draw, "rectangle",
                  x0=chamfer + 3, y0=chamfer + 3,
                  x1=S - chamfer - 3, y1=S - chamfer - 3,
                  fill=0, outline=104, width=1)

    # Parallel tool marks (subtle horizontal/vertical scoring)
    for _ in range(rng.randint(20, 35)):
        x = rng.randrange(chamfer + 4, S - chamfer - 4)
        y = rng.randrange(chamfer + 4, S - chamfer - 4)
        direction = rng.choice(("h", "v", "h", "v", "d"))
        length = rng.randint(8, 32)
        if direction == "h":
            ex, ey = x + length, y + rng.randint(-2, 2)
        elif direction == "v":
            ex, ey = x + rng.randint(-2, 2), y + length
        else:
            ex = x + length
            ey = y + length * rng.choice((-1, 1))
        ex = max(0, min(S - 1, ex))
        ey = max(0, min(S - 1, ey))
        for ox in (-S, 0, S):
            for oy in (-S, 0, S):
                draw.line([(x + ox, y + oy), (ex + ox, ey + oy)],
                          fill=rng.randint(48, 76), width=1)

    # Subtle surface wear
    for _ in range(rng.randint(40, 70)):
        cx = rng.randrange(S)
        cy = rng.randrange(S)
        for ox in (-S, 0, S):
            for oy in (-S, 0, S):
                draw.ellipse(
                    (cx - 1 + ox, cy - 1 + oy,
                     cx + 1 + ox, cy + 1 + oy),
                    fill=rng.randint(62, 98))


def add_emissive_glow(height: Image.Image, rng: random.Random) -> None:
    """Restrained warm emissive: subtle crack network with glow.

    Similar structure to dressed stone but with glowing veins in warm
    fullbright colours.  The cracks are lighter (high palette indices)
    rather than darker, inverting the typical recess pattern.
    """
    S = TEXTURE_SIZE
    draw = ImageDraw.Draw(height)

    # Subtle stone courses as backdrop
    course_h = S // 5
    for row in range(1, 5):
        y = row * course_h + rng.randint(-3, 3)
        y = max(4, min(S - 4, y))
        for ox in (-S, 0, S):
            for oy in (-S, 0, S):
                draw.line([(0 + ox, y + oy), (S + ox, y + oy)],
                          fill=72, width=3, joint="curve")

    # Glowing crack network (lighter = more glow)
    for _ in range(rng.randint(18, 28)):
        sx = rng.randrange(S)
        sy = rng.randrange(S)
        angle = rng.uniform(0, math.tau)
        segments = rng.randint(3, 7)
        points = [(sx, sy)]
        for seg in range(segments):
            angle += rng.uniform(-1.2, 1.2)
            dist = rng.randint(8, 30)
            sx = int(sx + math.cos(angle) * dist)
            sy = int(sy + math.sin(angle) * dist)
            points.append((sx % S, sy % S))
        # Glow core (bright)
        for ox in (-S, 0, S):
            for oy in (-S, 0, S):
                draw.line([(px + ox, py + oy) for px, py in points],
                          fill=rng.randint(190, 230), width=2, joint="curve")
        # Glow halo (medium)
        for ox in (-S, 0, S):
            for oy in (-S, 0, S):
                draw.line([(px + ox, py + oy) for px, py in points],
                          fill=rng.randint(155, 195), width=5, joint="curve")

    # Glowing ember dots
    for _ in range(rng.randint(30, 50)):
        cx = rng.randrange(S)
        cy = rng.randrange(S)
        for ox in (-S, 0, S):
            for oy in (-S, 0, S):
                draw.ellipse(
                    (cx - 1 + ox, cy - 1 + oy,
                     cx + 1 + ox, cy + 1 + oy),
                    fill=rng.randint(180, 240))

    # Dark recesses for contrast
    for _ in range(rng.randint(15, 25)):
        cx = rng.randrange(S)
        cy = rng.randrange(S)
        rr = rng.randint(2, 5)
        for ox in (-S, 0, S):
            for oy in (-S, 0, S):
                draw.ellipse(
                    (cx - rr + ox, cy - rr + oy,
                     cx + rr + ox, cy + rr + oy),
                    fill=rng.randint(24, 48))


# ═══════════════════════════════════════════════════════════════════════════
# PBR maps
# ═══════════════════════════════════════════════════════════════════════════

def colourise(
    height: Image.Image,
    rng: random.Random,
    base_rgb: tuple[int, int, int],
    gain: float,
    size: int = TEXTURE_SIZE,
) -> Image.Image:
    """Turn the height field into restrained, role-specific stone albedo."""
    mottling = centred(periodic_noise(rng, 16, size), 0.33)
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


def normal_map(height: Image.Image, strength: float, size: int = TEXTURE_SIZE) -> Image.Image:
    """Encode a tangent-space normal map from the procedural height field."""
    source = height.tobytes()
    normal = bytearray(size * size * 3)
    dst = 0
    for y in range(size):
        prev_row = ((y - 1) % size) * size
        row = y * size
        next_row = ((y + 1) % size) * size
        for x in range(size):
            left = row + ((x - 1) % size)
            right = row + ((x + 1) % size)
            gx = source[right] - source[left]
            gy = source[next_row + x] - source[prev_row + x]
            normal[dst] = clamp(128 - gx * strength)
            normal[dst + 1] = clamp(128 - gy * strength)
            normal[dst + 2] = clamp(255 - (abs(gx) + abs(gy)) * strength * 0.65)
            dst += 3
    return Image.frombytes("RGB", (size, size), bytes(normal))


def gloss_map(
    height: Image.Image,
    rng: random.Random,
    mean_gloss: int,
    size: int = TEXTURE_SIZE,
) -> Image.Image:
    """Create spatially useful 0.30–0.50 stone gloss from the height field.

    Dark cracks and pores remain rougher while broad worn stone gets a
    modestly glossier response.
    """
    broad = height.filter(ImageFilter.GaussianBlur(radius=max(2, size // 14)))
    pores = periodic_noise(rng, max(8, size // 4), size)
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
    # Clamp to valid PBR range
    gloss = gloss.point(
        [clamp(max(76, min(128, v))) for v in range(256)]
    )
    return Image.merge("RGB", (gloss, gloss, gloss))


# ═══════════════════════════════════════════════════════════════════════════
# Emissive colourise — uses warm fullbright palette entries for glow
# ═══════════════════════════════════════════════════════════════════════════

def colourise_emissive(
    height: Image.Image,
    rng: random.Random,
    base_rgb: tuple[int, int, int],
    gain: float,
) -> Image.Image:
    """Colourise the emissive height field with warm glow colours.

    Brighter pixels (the glow cracks) map into the fullbright palette range
    (entries 224–255) which Quake renders without lightmap modulation.
    Darker pixels map to warm dark stone.
    """
    S = TEXTURE_SIZE
    source = height.tobytes()
    mottling = centred(periodic_noise(rng, 16), 0.33)
    mottling_bytes = mottling.tobytes()

    rgb = bytearray(S * S * 3)
    dst = 0
    for i in range(S * S):
        h = source[i] / 255.0
        m = mottling_bytes[i] / 255.0

        # Blend base stone and glow based on height value
        # Higher height values → more glow (fullbright warm)
        # Lower height values → dark warm stone
        glow_factor = max(0.0, (h - 0.45) / 0.55)  # 0..1

        # Dark stone base (not glowing)
        dark_r = clamp(base_rgb[0] * 0.55 + (m - 0.5) * 40)
        dark_g = clamp(base_rgb[1] * 0.55 + (m - 0.5) * 35)
        dark_b = clamp(base_rgb[2] * 0.50 + (m - 0.5) * 28)

        # Glow colours (amber through warm gold)
        glow_r = clamp(220 + h * 35 + m * 20)
        glow_g = clamp(140 + h * 80 + m * 30)
        glow_b = clamp(40 + h * 60 + m * 20)

        rgb[dst] = clamp(dark_r + (glow_r - dark_r) * glow_factor)
        rgb[dst + 1] = clamp(dark_g + (glow_g - dark_g) * glow_factor)
        rgb[dst + 2] = clamp(dark_b + (glow_b - dark_b) * glow_factor)
        dst += 3

    return Image.frombytes("RGB", (S, S), bytes(rgb))


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
        "[theme]\n"
        "name = \"richness_ancient_v1\"\n"
        "version = \"1.0.0\"\n"
        "wad = \"richness_ancient_v1.wad\"\n"
        "\n"
        "[roles]\n"
        "wall = \"wall\"\n"
        "floor = \"floor\"\n"
        "ceiling = \"ceiling\"\n"
        "accent = \"accent\"\n"
        "portal = \"portal\"\n"
        "vertical = \"vertical\"\n"
        "cave = \"cave\"\n"
        "prop = \"prop\"\n"
        "emissive = \"emissive\"\n"
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
        '  "Richness Ancient v1 Theme" generated by the bsp_generator project.\n',
        encoding="utf-8",
    )

    (out_dir / "provenance.toml").write_text(
        "[theme]\n"
        "name = \"richness_ancient_v1\"\n"
        "license = \"CC0-1.0\"\n"
        "generator = \"build.py\"\n"
        "generator_language = \"Python 3\"\n"
        "generator_dependency = \"Pillow (PIL)\"\n"
        "texture_method = \"procedural\"\n"
        "texture_size = 256\n"
        "master_seed = \"0x52415631\"\n"
        "\n"
        "[source_assets]\n"
        "note = \"All textures are procedurally generated by build.py from fixed seeds. \"\n"
        "note_cont = \"No third-party artwork is sampled, transformed, or embedded.\"\n"
        "\n"
        "[build_inputs]\n"
        "build_script = \"build.py\"\n"
        "deterministic = true\n"
        "random_module = \"Python stdlib random with fixed seed per role\"\n"
        "pillow_version_hint = \">=9.0\"\n"
        "zlib_compression = \"PNG compress_level=9\"\n"
        "\n"
        "[outputs]\n"
        "png_count = 27\n"
        "wad_identities = [\"wall\", \"floor\", \"ceiling\", \"accent\", \"portal\", \"vertical\", \"cave\", \"prop\", \"emissive\", \"skip\"]\n"
        "palette_size_bytes = 768\n",
        encoding="utf-8",
    )

    # Provenance hashes cover every generated output except this document
    # itself; self-hashing would be cyclic. build.py is an input, not output.
    generated = ["theme.toml", "LICENSE", "palette.lmp"]
    generated.append(next(out_dir.glob("*.wad")).name)
    generated.extend(f"textures/{path.name}" for path in sorted((out_dir / "textures").glob("*.png")))
    with (out_dir / "provenance.toml").open("a", encoding="utf-8") as provenance:
        provenance.write("\n[hashes]\n")
        for filename in generated:
            digest = hashlib.sha256((out_dir / filename).read_bytes()).hexdigest()
            provenance.write(f'"{filename}" = "{digest}"\n')


# ═══════════════════════════════════════════════════════════════════════════
# Per-role height generator dispatch
# ═══════════════════════════════════════════════════════════════════════════

def role_height(name: str, rng: random.Random, gain: float) -> Image.Image:
    """Generate the height field for a texture role."""
    height = base_height(rng, gain)

    if name == "wall":
        add_cyclopean_wall(height, rng)
    elif name == "floor":
        add_cyclopean_floor(height, rng)
    elif name == "ceiling":
        add_slab_ceiling(height, rng)
    elif name == "accent":
        add_accent_dressed_stone(height, rng)
    elif name == "portal":
        add_portal_post_lintel(height, rng)
    elif name == "vertical":
        add_vertical_pillar(height, rng)
    elif name == "cave":
        add_cave_surface(height, rng)
    elif name == "prop":
        add_prop_worked_stone(height, rng)
    elif name == "emissive":
        add_emissive_glow(height, rng)
    else:
        raise ValueError(f"unknown role: {name}")

    return height


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
    for index, (name, base_rgb, colour_gain, normal_strength, mean_gloss) in enumerate(ANCIENT_DEFS):
        role_rng = random.Random(MASTER_SEED + index * 0x9E3779B1)
        height = role_height(name, role_rng, colour_gain)

        if name == "emissive":
            base = colourise_emissive(height, role_rng, base_rgb, colour_gain)
        else:
            base = colourise(height, role_rng, base_rgb, colour_gain)

        normal = normal_map(height, normal_strength)
        gloss = gloss_map(height, role_rng, mean_gloss)

        base.save(texture_dir / f"{name}_basecolor.png", **PNG_SAVE_OPTIONS)
        normal.save(texture_dir / f"{name}_norm.png", **PNG_SAVE_OPTIONS)
        gloss.save(texture_dir / f"{name}_gloss.png", **PNG_SAVE_OPTIONS)
        wad_entries.append((name, make_miptex(name, base, pal_for_quant)))

    # Compiler-only skip texture (64x64, black)
    skip_img = Image.new("RGB", (SKIP_TEXTURE_SIZE, SKIP_TEXTURE_SIZE), (0, 0, 0))
    wad_entries.append(("skip", make_miptex("skip", skip_img, pal_for_quant)))

    (out_dir / "richness_ancient_v1.wad").write_bytes(make_wad2(wad_entries))
    write_static_files(out_dir)

    print(f"Richness Ancient v1 theme generated in {out_dir.resolve()}")


if __name__ == "__main__":
    main()
