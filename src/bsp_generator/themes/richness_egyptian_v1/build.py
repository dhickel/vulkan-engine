#!/usr/bin/env python3
"""Deterministic procedural Richness Egyptian v1 theme asset generator.

Generates a complete CC0 Egyptian theme with stepped pylon/mastaba walls,
hypostyle hall column rhythms, obelisk/cavetto-like stepped capitals,
shrine/sarcophagus/canopic prop atlas, amber warm emissive identity,
and a compiler-only skip texture. Every visible texture has matching
basecolor, normal, and gloss companions.

Outputs (placed in target directory, default CWD):
- palette.lmp                     — 256-colour project-authored palette (768 bytes)
- richness_egyptian_v1.wad        — WAD2 archive with 9 visible miptex entries
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
        "Richness Egyptian v1 generation requires Pillow: pip install Pillow"
    ) from error


TEXTURE_SIZE = 256
SKIP_TEXTURE_SIZE = 64
# "REV1" in ASCII — Richness Egyptian V1
MASTER_SEED = 0x52455631
PNG_SAVE_OPTIONS = {"format": "PNG", "compress_level": 9, "optimize": False}

# ── Texture definitions ────────────────────────────────────────────────────
# (name, base-RGB, height-gain, normal-strength, mean-gloss)

EGYPTIAN_DEFS = (
    ("wall",     (196, 172, 136), 0.68, 1.24, 90),
    ("floor",    (180, 154, 118), 0.74, 1.28, 82),
    ("ceiling",  (208, 190, 158), 0.50, 0.96, 98),
    ("accent",   (212, 180, 130), 0.56, 1.08, 106),
    ("portal",   (190, 166, 130), 0.64, 1.18, 86),
    ("vertical", (188, 164, 128), 0.60, 1.14, 94),
    ("cave",     (130, 114, 92),  0.80, 1.36, 70),
    ("prop",     (198, 168, 124), 0.52, 1.04, 110),
    ("emissive", (200, 160, 90),  0.54, 1.06, 102),
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
    """Return a project-authored 256-entry Egyptian stone palette.

    Entries 0..223 are warm Egyptian-stone ramps (sandstone, limestone,
    golden ochre, umber, weathered buff).  Entries 224..255 are amber
    fullbrights (amber through gold to warm white) for emissive glow.
    """
    ramps = (
        ((16, 12, 8),   (224, 210, 188)),  # warm buff greys
        ((40, 30, 18),  (168, 128, 82)),    # dark umber
        ((56, 40, 24),  (196, 146, 90)),    # warm earth
        ((74, 54, 32),  (212, 170, 110)),   # weathered golden ochre
        ((66, 60, 50),  (176, 168, 150)),   # charcoal warm grey
        ((94, 88, 76),  (198, 190, 174)),   # mid warm sandstone
        ((116, 110, 94),(220, 214, 194)),   # pale limestone
        ((136, 124, 102),(228, 218, 186)),  # warm sand
        ((84, 80, 70),  (182, 176, 160)),   # cool-toned stone grey
        ((106, 98, 82), (200, 190, 162)),   # olive-tinged stone
        ((122, 108, 86),(214, 196, 156)),   # golden limestone
        ((146, 130, 102),(230, 216, 176)),  # carved warm accent
        ((160, 148, 126),(238, 228, 202)),  # pale dressed stone
        ((170, 160, 140),(244, 236, 214)),  # light worn cap
    )

    palette = bytearray()
    for start, end in ramps:
        for step in range(16):
            fraction = step / 15
            palette.extend(
                clamp(start[channel] + (end[channel] - start[channel]) * fraction)
                for channel in range(3)
            )

    # Amber fullbright entries (224–255): amber through gold to warm white.
    # These are the emissive glow colours that Quake lightmapping leaves
    # unmodulated (fullbright).
    fullbright_ramp = (
        (180, 80, 12),     # deep amber
        (210, 110, 24),    # rich amber
        (232, 140, 40),    # bright amber
        (244, 168, 56),    # warm gold
        (250, 190, 72),    # light gold
        (254, 208, 96),    # pale gold
        (255, 224, 120),   # warm yellow
        (255, 238, 168),   # warm cream
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
            elif element_type == "polygon":
                pts = [(x + ox, y + oy) for x, y in kwargs["points"]]
                draw.polygon(pts, fill=kwargs.get("fill"),
                             outline=kwargs.get("outline"))


# ═══════════════════════════════════════════════════════════════════════════
# Role height generators
# ═══════════════════════════════════════════════════════════════════════════

def add_stepped_pylon_wall(height: Image.Image, rng: random.Random) -> None:
    """Stepped pylon/mastaba wall: horizontal stepped batter courses.

    Each course steps inward slightly creating the characteristic battered
    profile of Egyptian monumental walls.  Trapezoidal massing is implied
    by advancing horizontal bands with angled shadow lines at each step.
    Large stone blocks with deeply recessed mortar complete the look.
    """
    S = TEXTURE_SIZE
    draw = ImageDraw.Draw(height)

    # Stepped batter courses: horizontal bands that step
    num_courses = 7
    course_h = S // num_courses
    for row in range(num_courses):
        y0 = row * course_h
        y1 = min(S, y0 + course_h)
        # Step shadow: each course casts a shadow band at its bottom edge
        step_y = y0 + rng.randint(-2, 2)
        step_y = max(2, min(S - 2, step_y))
        # Deep shadow line for the step overhang
        seamless_draw(draw, "line",
                      points=[(0, step_y), (S, step_y)],
                      fill=36, width=6, joint="curve")
        # Subtle highlight on the step face below
        seamless_draw(draw, "line",
                      points=[(0, step_y + 4), (S, step_y + 4)],
                      fill=110, width=2, joint="curve")
        # Angled shadow hint: slight diagonal offsets to suggest batter
        for col_seg in range(0, S, S // 6):
            sx = col_seg + rng.randint(-8, 8)
            sx = max(0, min(S, sx))
            ex = min(S, sx + rng.randint(4, 12))
            seamless_draw(draw, "line",
                          points=[(sx, step_y), (ex, step_y + rng.randint(3, 7))],
                          fill=42, width=3)

    # Vertical block joints within each course
    for row in range(num_courses):
        y0 = row * course_h
        y1 = min(S, y0 + course_h)
        offset = rng.randint(12, 48) if row % 2 == 0 else rng.randint(56, 100)
        for col in range(0, 8):
            x = (offset + col * S // 7) % S
            x = max(3, min(S - 3, x))
            seamless_draw(draw, "line",
                          points=[(x, y0 + 6), (x + rng.randint(-3, 3), y1 - 2)],
                          fill=46, width=4)
        # Mortar highlight beside joints
        for col in range(0, 8):
            x = (offset + col * S // 7) % S
            x = max(3, min(S - 3, x))
            seamless_draw(draw, "line",
                          points=[(x - 3, y0 + 6), (x - 3, y1 - 2)],
                          fill=104, width=1)

    # Surface chips and erosion
    for _ in range(rng.randint(55, 85)):
        cx = rng.randrange(S)
        cy = rng.randrange(S)
        rx = rng.randint(1, 4)
        ry = rng.randint(1, 4)
        seamless_draw(draw, "ellipse",
                      x0=cx - rx, y0=cy - ry,
                      x1=cx + rx, y1=cy + ry,
                      fill=rng.randint(42, 78))


def add_hypostyle_floor(height: Image.Image, rng: random.Random) -> None:
    """Hypostyle hall column rhythm: grid of column base impressions.

    Regular flagstone paving punctuated by circular column-base shadows
    creating a rhythmic hypostyle hall pattern.  Sand-worn edges and
    subtle directional wear between columns complete the floor.
    """
    S = TEXTURE_SIZE
    draw = ImageDraw.Draw(height)

    # Column base grid — circular impressions at regular intervals
    cols = 4
    rows = 4
    cell_w = S // cols
    cell_h = S // rows

    for cy in range(rows):
        for cx in range(cols):
            # Column centre with slight jitter
            centre_x = cx * cell_w + cell_w // 2 + rng.randint(-6, 6)
            centre_y = cy * cell_h + cell_h // 2 + rng.randint(-6, 6)
            base_radius = min(cell_w, cell_h) // 4 + rng.randint(-4, 4)

            # Deep shadow ring for column base
            for angle in range(0, 360, 5):
                rad = math.radians(angle)
                px = int(centre_x + base_radius * math.cos(rad))
                py = int(centre_y + base_radius * math.sin(rad))
                for ox in (-S, 0, S):
                    for oy in (-S, 0, S):
                        draw.ellipse((px - 3 + ox, py - 3 + oy,
                                      px + 3 + ox, py + 3 + oy),
                                     fill=48)

            # Inner highlight ring
            inner_r = base_radius - 4
            for angle in range(0, 360, 8):
                rad = math.radians(angle)
                px = int(centre_x + inner_r * math.cos(rad))
                py = int(centre_y + inner_r * math.sin(rad))
                for ox in (-S, 0, S):
                    for oy in (-S, 0, S):
                        draw.ellipse((px - 1 + ox, py - 1 + oy,
                                      px + 1 + ox, py + 1 + oy),
                                     fill=102)

            # Central column core
            core_r = base_radius // 2
            seamless_draw(draw, "ellipse",
                          x0=centre_x - core_r, y0=centre_y - core_r,
                          x1=centre_x + core_r, y1=centre_y + core_r,
                          fill=52)

    # Flagstone joints — irregular horizontal/vertical lines
    for row in range(1, 6):
        y = row * S // 6 + rng.randint(-8, 8)
        y = max(6, min(S - 6, y))
        seamless_draw(draw, "line",
                      points=[(0, y), (S, y)],
                      fill=58, width=5, joint="curve")
        seamless_draw(draw, "line",
                      points=[(0, y + 2), (S, y + 2)],
                      fill=98, width=1, joint="curve")

    for col in range(1, 6):
        x = col * S // 6 + rng.randint(-8, 8)
        x = max(6, min(S - 6, x))
        seamless_draw(draw, "line",
                      points=[(x, 0), (x + rng.randint(-3, 3), S)],
                      fill=56, width=4)

    # Wear paths between columns
    for _ in range(rng.randint(20, 35)):
        cx = rng.randrange(S)
        cy = rng.randrange(S)
        angle = rng.uniform(0, math.tau)
        length = rng.randint(16, 48)
        ex = cx + int(math.cos(angle) * length)
        ey = cy + int(math.sin(angle) * length)
        for ox in (-S, 0, S):
            for oy in (-S, 0, S):
                draw.line([(cx + ox, cy + oy), (ex + ox, ey + oy)],
                          fill=rng.randint(68, 96), width=2)

    # Sand pits
    for _ in range(rng.randint(80, 120)):
        cx = rng.randrange(S)
        cy = rng.randrange(S)
        rr = rng.randint(1, 3)
        seamless_draw(draw, "ellipse",
                      x0=cx - rr, y0=cy - rr,
                      x1=cx + rr, y1=cy + rr,
                      fill=rng.randint(52, 88))


def add_obelisk_cavetto_ceiling(height: Image.Image, rng: random.Random) -> None:
    """Obelisk/cavetto-like stepped capitals: upward converging stepped forms.

    Stepped, angular ceiling pattern suggesting the stepped capitals of
    obelisks and cavetto cornices.  Concentric stepped rectangles converging
    upward, with papyrus/palm capital suggestions rendered as stepped
    geometric forms rather than smooth curves.
    """
    S = TEXTURE_SIZE
    draw = ImageDraw.Draw(height)

    # Concentric stepped rectangular forms converging toward centre
    steps = 6
    for step_idx in range(steps):
        margin = 12 + step_idx * (S // 16)
        margin = min(margin, S // 2 - 8)

        # Each step has a shadow edge and highlight
        seamless_draw(draw, "rectangle",
                      x0=margin, y0=margin,
                      x1=S - margin, y1=S - margin,
                      outline=40 + step_idx * 6, width=4)

        # Highlight on inner edge of each step
        inner_m = margin + 4
        if inner_m < S // 2:
            seamless_draw(draw, "rectangle",
                          x0=inner_m, y0=inner_m,
                          x1=S - inner_m, y1=S - inner_m,
                          outline=96 + step_idx * 4, width=2)

    # Papyrus/palm capital suggestion: stepped diagonal ribs from corners
    for quadrant in range(4):
        if quadrant == 0:
            cx_s, cy_s = 0, 0
            dx_s, dy_s = 1, 1
        elif quadrant == 1:
            cx_s, cy_s = S, 0
            dx_s, dy_s = -1, 1
        elif quadrant == 2:
            cx_s, cy_s = 0, S
            dx_s, dy_s = 1, -1
        else:
            cx_s, cy_s = S, S
            dx_s, dy_s = -1, -1

        for rib in range(3):
            # Stepped lines radiating from corners toward centre
            offset = 16 + rib * 20
            pts = []
            for t in range(0, S // 2, 16):
                px = cx_s + dx_s * (offset + t)
                py = cy_s + dy_s * (offset + t)
                pts.append((px, py))
            if pts:
                seamless_draw(draw, "line",
                              points=pts, fill=52 + rib * 6,
                              width=3)

    # Subtle horizontal seam lines
    for row in range(1, 4):
        y = row * S // 4 + rng.randint(-6, 6)
        y = max(4, min(S - 4, y))
        seamless_draw(draw, "line",
                      points=[(0, y), (S, y)],
                      fill=66, width=3, joint="curve")

    # Pores and weathering
    for _ in range(rng.randint(70, 110)):
        cx = rng.randrange(S)
        cy = rng.randrange(S)
        rr = rng.choice((1, 1, 2))
        seamless_draw(draw, "ellipse",
                      x0=cx - rr, y0=cy - rr,
                      x1=cx + rr, y1=cy + rr,
                      fill=rng.randint(70, 105))


def add_shrine_accent(height: Image.Image, rng: random.Random) -> None:
    """Shrine/sarcophagus/canopic accent: refined dressed stone.

    Incised horizontal bands suggesting hieroglyphic registers, cartouche
    panel shapes, and polished ceremonial surfaces.  Combines dressed stone
    courses with decorative inset panels.
    """
    S = TEXTURE_SIZE
    draw = ImageDraw.Draw(height)

    # Horizontal dressed courses
    course_h = S // 5
    for row in range(0, 6):
        y = row * course_h
        y = min(y, S)
        if y == 0 or y >= S:
            continue
        seamless_draw(draw, "line",
                      points=[(0, y), (S, y)],
                      fill=50, width=4, joint="curve")
        seamless_draw(draw, "line",
                      points=[(0, y + 1), (S, y + 1)],
                      fill=106, width=1, joint="curve")

    # Hieroglyphic register bands — thin incised lines within courses
    for row in range(5):
        y0 = row * course_h
        y1 = min(S, y0 + course_h)
        # Top register line
        reg_y = y0 + course_h // 4 + rng.randint(-2, 2)
        reg_y = max(y0 + 3, min(y1 - 3, reg_y))
        seamless_draw(draw, "line",
                      points=[(6, reg_y), (S - 6, reg_y)],
                      fill=58, width=2, joint="curve")
        # Bottom register line
        reg_y2 = y0 + 3 * course_h // 4 + rng.randint(-2, 2)
        reg_y2 = max(y0 + 3, min(y1 - 3, reg_y2))
        seamless_draw(draw, "line",
                      points=[(6, reg_y2), (S - 6, reg_y2)],
                      fill=58, width=2, joint="curve")

    # Vertical joints (staggered per course)
    for row in range(5):
        y0 = row * course_h
        y1 = min(S, y0 + course_h)
        offset = rng.randint(16, 40) if row % 2 == 0 else rng.randint(60, 90)
        for col in range(0, 5):
            x = (offset + col * S // 4) % S
            x = max(4, min(S - 4, x))
            seamless_draw(draw, "line",
                          points=[(x, y0), (x + rng.randint(-2, 2), y1)],
                          fill=52, width=3)

    # Cartouche panel — central oval/rounded rectangle inset
    margin = 28
    cartouche_x0 = margin
    cartouche_y0 = margin + 8
    cartouche_x1 = S - margin
    cartouche_y1 = S - margin - 8

    # Cartouche outer border
    seamless_draw(draw, "rounded_rectangle",
                  x0=cartouche_x0, y0=cartouche_y0,
                  x1=cartouche_x1, y1=cartouche_y1,
                  radius=14, outline=60, fill=None, width=5)
    seamless_draw(draw, "rounded_rectangle",
                  x0=cartouche_x0 + 3, y0=cartouche_y0 + 3,
                  x1=cartouche_x1 - 3, y1=cartouche_y1 - 3,
                  radius=12, outline=98, fill=None, width=2)

    # Horizontal rule at cartouche bottom (sarcophagus lid suggestion)
    lid_y = cartouche_y0 + (cartouche_y1 - cartouche_y0) * 2 // 3
    seamless_draw(draw, "line",
                  points=[(cartouche_x0 + 10, lid_y),
                          (cartouche_x1 - 10, lid_y)],
                  fill=54, width=4)

    # Canopic jar lid suggestion — smaller oval at top of cartouche
    jar_cx = S // 2
    jar_cy = cartouche_y0 + (cartouche_y1 - cartouche_y0) // 3
    jar_rx = (cartouche_x1 - cartouche_x0) // 3
    jar_ry = jar_rx * 3 // 4
    seamless_draw(draw, "ellipse",
                  x0=jar_cx - jar_rx, y0=jar_cy - jar_ry,
                  x1=jar_cx + jar_rx, y1=jar_cy + jar_ry,
                  fill=55)

    # Chisel marks
    for _ in range(rng.randint(35, 60)):
        x = rng.randrange(S)
        y = rng.randrange(S)
        angle = rng.uniform(0, math.tau)
        length = rng.randint(3, 12)
        ex = x + int(math.cos(angle) * length)
        ey = y + int(math.sin(angle) * length)
        for ox in (-S, 0, S):
            for oy in (-S, 0, S):
                draw.line([(x + ox, y + oy), (ex + ox, ey + oy)],
                          fill=rng.randint(48, 76), width=1)


def add_pylon_portal(height: Image.Image, rng: random.Random) -> None:
    """Pylon gateway: massive battered trapezoidal doorway with stepped reveals.

    A trapezoidal opening narrowing inward, stepped reveals on each side,
    lintel with solar disk suggestion, and deep shadow lines create the
    monumental Egyptian gateway character.
    """
    S = TEXTURE_SIZE
    draw = ImageDraw.Draw(height)

    # Trapezoidal doorway: wider at bottom, narrower at top
    top_width = S // 3
    bottom_width = S * 2 // 5
    top_y = S // 5
    bottom_y = S - S // 6

    top_left = (S - top_width) // 2
    top_right = top_left + top_width
    bottom_left = (S - bottom_width) // 2
    bottom_right = bottom_left + bottom_width

    # Outer stepped reveal — wider, deeper shadow
    outer_inset = 8
    outer_pts_top = [
        (top_left + outer_inset, top_y + outer_inset),
        (top_right - outer_inset, top_y + outer_inset),
        (bottom_right - outer_inset, bottom_y - outer_inset),
        (bottom_left + outer_inset, bottom_y - outer_inset),
    ]
    seamless_draw(draw, "polygon",
                  points=outer_pts_top, fill=34)

    # Inner stepped reveal — brighter (lit face)
    inner_inset = 6
    inner_pts = [
        (top_left + inner_inset, top_y + inner_inset),
        (top_right - inner_inset, top_y + inner_inset),
        (bottom_right - inner_inset, bottom_y - inner_inset),
        (bottom_left + inner_inset, bottom_y - inner_inset),
    ]
    seamless_draw(draw, "polygon",
                  points=inner_pts, fill=48)

    # Central dark opening (deep portal interior)
    open_inset = 4
    open_pts = [
        (top_left + open_inset, top_y + open_inset),
        (top_right - open_inset, top_y + open_inset),
        (bottom_right - open_inset, bottom_y - open_inset),
        (bottom_left + open_inset, bottom_y - open_inset),
    ]
    seamless_draw(draw, "polygon",
                  points=open_pts, fill=20)

    # Lintel — massive horizontal band above portal
    lintel_y0 = top_y - S // 12 + rng.randint(-3, 3)
    lintel_y0 = max(4, lintel_y0)
    lintel_y1 = top_y + rng.randint(-2, 2)
    # Shadow under lintel
    seamless_draw(draw, "line",
                  points=[(0, lintel_y1), (S, lintel_y1)],
                  fill=32, width=8, joint="curve")
    # Lintel body highlight
    seamless_draw(draw, "line",
                  points=[(0, lintel_y0), (S, lintel_y0)],
                  fill=112, width=3, joint="curve")

    # Solar disk suggestion — circle above lintel centre
    disk_cx = S // 2
    disk_cy = lintel_y0 - S // 16
    disk_cy = max(8, disk_cy)
    disk_r = S // 10
    seamless_draw(draw, "ellipse",
                  x0=disk_cx - disk_r, y0=disk_cy - disk_r,
                  x1=disk_cx + disk_r, y1=disk_cy + disk_r,
                  fill=60)
    # Solar rays (stepped)
    for angle in range(0, 360, 30):
        rad = math.radians(angle)
        px = int(disk_cx + (disk_r + 4) * math.cos(rad))
        py = int(disk_cy + (disk_r + 4) * math.sin(rad))
        for ox in (-S, 0, S):
            for oy in (-S, 0, S):
                draw.ellipse((px - 2 + ox, py - 2 + oy,
                              px + 2 + ox, py + 2 + oy),
                             fill=72)

    # Pylon wall blocks flanking portal
    for side in (0, 1):
        if side == 0:
            bx0, bx1 = 4, top_left - 2
        else:
            bx0, bx1 = top_right + 2, S - 4
        for row in range(0, 5):
            y0 = row * S // 5
            y1 = min(S, y0 + S // 5)
            seamless_draw(draw, "line",
                          points=[(bx0, y0), (bx1, y0)],
                          fill=40, width=4, joint="curve")

    # Surface distress
    for _ in range(rng.randint(45, 70)):
        cx = rng.randrange(S)
        cy = rng.randrange(S)
        rr = rng.randint(1, 3)
        seamless_draw(draw, "ellipse",
                      x0=cx - rr, y0=cy - rr,
                      x1=cx + rr, y1=cy + rr,
                      fill=rng.randint(44, 80))


def add_obelisk_vertical(height: Image.Image, rng: random.Random) -> None:
    """Obelisk shaft / column: tall tapered vertical lines.

    Strong vertical divisions suggesting obelisk shafts with subtle
    horizontal hieroglyphic register bands, fluting, and tapered edges.
    Stepped battery creates the characteristic narrowing profile.
    """
    S = TEXTURE_SIZE
    draw = ImageDraw.Draw(height)

    # Vertical tapered divisions — obelisk shaft edges
    num_shafts = 3
    shaft_w = S // num_shafts
    for s_idx in range(1, num_shafts):
        sx = s_idx * shaft_w + rng.randint(-8, 8)
        sx = max(4, min(S - 4, sx))
        # Deep central groove with taper
        seamless_draw(draw, "line",
                      points=[(sx, 0), (sx, S)],
                      fill=40, width=5)
        # Highlight edges
        seamless_draw(draw, "line",
                      points=[(sx + 2, 0), (sx + 2, S)],
                      fill=100, width=1)
        seamless_draw(draw, "line",
                      points=[(sx - 3, 0), (sx - 3, S)],
                      fill=108, width=2)

    # Subtle fluting within each shaft
    for s_idx in range(num_shafts):
        sx0 = s_idx * shaft_w
        # Fluting lines within shaft
        for f_idx in range(1, 4):
            fx = sx0 + f_idx * shaft_w // 4 + rng.randint(-4, 4)
            fx = max(2, min(S - 2, fx))
            seamless_draw(draw, "line",
                          points=[(fx, 0), (fx, S)],
                          fill=50, width=2)

    # Hieroglyphic register bands — horizontal interruptions
    for band_row in range(3):
        by0 = band_row * S // 3 + S // 8 + rng.randint(-6, 6)
        by0 = max(8, min(S - 8, by0))
        by1 = by0 + S // 14

        # Band shadow
        seamless_draw(draw, "line",
                      points=[(4, by0), (S - 4, by0)],
                      fill=42, width=5, joint="curve")
        seamless_draw(draw, "line",
                      points=[(4, by1), (S - 4, by1)],
                      fill=46, width=4, joint="curve")
        # Band surface highlight
        seamless_draw(draw, "line",
                      points=[(4, by0 + 2), (S - 4, by0 + 2)],
                      fill=104, width=1, joint="curve")

        # Small incised marks within band (hieroglyphic suggestion)
        for _ in range(rng.randint(6, 10)):
            mx = rng.randint(10, S - 10)
            my = rng.randint(by0 + 3, by1 - 3)
            ml = rng.randint(4, 14)
            mangle = rng.choice((0, math.pi / 2))
            mex = mx + int(math.cos(mangle) * ml)
            mey = my + int(math.sin(mangle) * ml)
            for ox in (-S, 0, S):
                for oy in (-S, 0, S):
                    draw.line([(mx + ox, my + oy), (mex + ox, mey + oy)],
                              fill=56, width=1)

    # Surface chips (vertical)
    for _ in range(rng.randint(50, 80)):
        cx = rng.randrange(S)
        cy = rng.randrange(S)
        rr = rng.randint(1, 3)
        seamless_draw(draw, "ellipse",
                      x0=cx - rr, y0=cy - rr,
                      x1=cx + rr, y1=cy + rr,
                      fill=rng.randint(46, 82))


def add_tomb_cave(height: Image.Image, rng: random.Random) -> None:
    """Tomb passage: rough hewn rock with occasional straight-cut surfaces.

    Organic, irregular cavern rock punctuated by chisel marks and occasional
    straight-cut tooled surfaces suggesting worked tomb passages.  Deep
    crevices and natural nodules create a raw excavated feel.
    """
    S = TEXTURE_SIZE
    draw = ImageDraw.Draw(height)

    # Organic blob-like recesses
    for _ in range(rng.randint(10, 18)):
        cx = rng.randrange(S)
        cy = rng.randrange(S)
        rx = rng.randint(14, 44)
        ry = rng.randint(14, 44)
        for ox in (-S, 0, S):
            for oy in (-S, 0, S):
                draw.ellipse(
                    (cx - rx + ox, cy - ry + oy,
                     cx + rx + ox, cy + ry + oy),
                    fill=rng.randint(36, 60))

    # Deep crevices
    for _ in range(rng.randint(20, 36)):
        sx = rng.randrange(S)
        sy = rng.randrange(S)
        angle = rng.uniform(0, math.tau)
        length = rng.randint(10, 56)
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
                          fill=rng.randint(28, 50), width=rng.choice((2, 3, 4)),
                          joint="curve")

    # Straight-cut tooled surfaces — occasional chisel-straight lines
    for _ in range(rng.randint(8, 16)):
        sx = rng.randrange(S)
        sy = rng.randrange(S)
        angle = rng.choice((0, math.pi / 2, math.pi / 4, -math.pi / 4))
        length = rng.randint(20, 70)
        ex = sx + int(math.cos(angle) * length)
        ey = sy + int(math.sin(angle) * length)
        # Tooled surface: parallel chisel marks
        for offset in range(-6, 7, 3):
            ox_s = sx + offset * int(math.cos(angle + math.pi / 2))
            oy_s = sy + offset * int(math.sin(angle + math.pi / 2))
            ox_e = ex + offset * int(math.cos(angle + math.pi / 2))
            oy_e = ey + offset * int(math.sin(angle + math.pi / 2))
            for ox in (-S, 0, S):
                for oy in (-S, 0, S):
                    draw.line([(ox_s + ox, oy_s + oy),
                               (ox_e + ox, oy_e + oy)],
                              fill=52, width=1)

    # Surface nodules and bumps (lighter spots)
    for _ in range(rng.randint(50, 90)):
        cx = rng.randrange(S)
        cy = rng.randrange(S)
        rx = rng.randint(2, 7)
        ry = rng.randint(2, 7)
        for ox in (-S, 0, S):
            for oy in (-S, 0, S):
                draw.ellipse(
                    (cx - rx + ox, cy - ry + oy,
                     cx + rx + ox, cy + ry + oy),
                    fill=rng.randint(136, 178))

    # Scattered pits
    for _ in range(rng.randint(90, 140)):
        cx = rng.randrange(S)
        cy = rng.randrange(S)
        for ox in (-S, 0, S):
            for oy in (-S, 0, S):
                draw.ellipse(
                    (cx - 2 + ox, cy - 2 + oy,
                     cx + 2 + ox, cy + 2 + oy),
                    fill=rng.randint(28, 56))


def add_canopic_prop(height: Image.Image, rng: random.Random) -> None:
    """Canopic/shrine prop detail: refined stone with box/lid shapes.

    Smooth worked stone with banded detail suggesting canopic chests,
    jar lids, and shrine boxes.  Gentle edge chamfers, subtle parallel
    tool marks, and restrained wear.
    """
    S = TEXTURE_SIZE
    draw = ImageDraw.Draw(height)

    # Gentle edge chamfer (inset border)
    chamfer = 10
    seamless_draw(draw, "rectangle",
                  x0=chamfer, y0=chamfer,
                  x1=S - chamfer, y1=S - chamfer,
                  fill=0, outline=56, width=5)
    seamless_draw(draw, "rectangle",
                  x0=chamfer + 3, y0=chamfer + 3,
                  x1=S - chamfer - 3, y1=S - chamfer - 3,
                  fill=0, outline=108, width=1)

    # Banded detail — horizontal divisions suggesting canopic chest tiers
    for band in range(3):
        by = chamfer + 10 + band * (S - 2 * chamfer - 20) // 3 + rng.randint(-3, 3)
        by = max(chamfer + 4, min(S - chamfer - 4, by))
        seamless_draw(draw, "line",
                      points=[(chamfer + 4, by), (S - chamfer - 4, by)],
                      fill=50, width=4, joint="curve")
        seamless_draw(draw, "line",
                      points=[(chamfer + 4, by + 1), (S - chamfer - 4, by + 1)],
                      fill=104, width=1, joint="curve")

    # Canopic jar lid shapes — small ovals in top section
    for lid_idx in range(4):
        lx = chamfer + 14 + lid_idx * (S - 2 * chamfer - 28) // 4 + rng.randint(-4, 4)
        ly = chamfer + 14 + rng.randint(-4, 4)
        lrx = rng.randint(8, 14)
        lry = rng.randint(6, 10)
        seamless_draw(draw, "ellipse",
                      x0=lx - lrx, y0=ly - lry,
                      x1=lx + lrx, y1=ly + lry,
                      fill=58)
        # Lid highlight
        seamless_draw(draw, "ellipse",
                      x0=lx - lrx + 2, y0=ly - lry + 2,
                      x1=lx + lrx - 2, y1=ly + lry - 2,
                      fill=94)

    # Parallel tool marks (subtle scoring)
    for _ in range(rng.randint(16, 28)):
        x = rng.randrange(chamfer + 4, S - chamfer - 4)
        y = rng.randrange(chamfer + 4, S - chamfer - 4)
        direction = rng.choice(("h", "v", "h", "v", "d"))
        length = rng.randint(6, 28)
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
                          fill=rng.randint(46, 74), width=1)

    # Subtle surface wear
    for _ in range(rng.randint(30, 55)):
        cx = rng.randrange(S)
        cy = rng.randrange(S)
        for ox in (-S, 0, S):
            for oy in (-S, 0, S):
                draw.ellipse(
                    (cx - 1 + ox, cy - 1 + oy,
                     cx + 1 + ox, cy + 1 + oy),
                    fill=rng.randint(60, 96))


def add_amber_emissive(height: Image.Image, rng: random.Random) -> None:
    """Amber warm emissive: stepped crack network with amber fullbright glow.

    Angular, stepped crack network inspired by Egyptian architectural lines
    with amber/golden glow in warm fullbright colours.  Torch sconce glow
    hints create bright focal points.  The cracks are lighter (high palette
    indices) rather than darker, inverting the typical recess pattern.
    """
    S = TEXTURE_SIZE
    draw = ImageDraw.Draw(height)

    # Subtle stepped course backdrop
    course_h = S // 5
    for row in range(1, 5):
        y = row * course_h + rng.randint(-3, 3)
        y = max(4, min(S - 4, y))
        for ox in (-S, 0, S):
            for oy in (-S, 0, S):
                draw.line([(0 + ox, y + oy), (S + ox, y + oy)],
                          fill=70, width=3, joint="curve")

    # Angular stepped crack network (Egyptian linear quality)
    for _ in range(rng.randint(14, 24)):
        sx = rng.randrange(S)
        sy = rng.randrange(S)
        # Egyptian-style: cracks tend to follow horizontal/vertical/diagonal axes
        base_angle = rng.choice((0, math.pi / 2, math.pi / 4, -math.pi / 4,
                                 3 * math.pi / 4, -3 * math.pi / 4))
        angle = base_angle + rng.uniform(-0.3, 0.3)
        segments = rng.randint(3, 6)
        points = [(sx, sy)]
        for seg in range(segments):
            # Occasional stepped turns
            if rng.random() < 0.3:
                angle = base_angle + rng.choice((-math.pi / 2, math.pi / 2,
                                                  math.pi / 4, -math.pi / 4))
            angle += rng.uniform(-0.4, 0.4)
            dist = rng.randint(8, 36)
            sx = int(sx + math.cos(angle) * dist)
            sy = int(sy + math.sin(angle) * dist)
            points.append((sx % S, sy % S))

        # Glow core (bright — high palette index)
        for ox in (-S, 0, S):
            for oy in (-S, 0, S):
                draw.line([(px + ox, py + oy) for px, py in points],
                          fill=rng.randint(195, 235), width=2, joint="curve")
        # Glow halo (medium)
        for ox in (-S, 0, S):
            for oy in (-S, 0, S):
                draw.line([(px + ox, py + oy) for px, py in points],
                          fill=rng.randint(152, 195), width=5, joint="curve")

    # Torch sconce glow hints — bright focal spots
    for _ in range(rng.randint(4, 8)):
        tx = rng.randint(30, S - 30)
        ty = rng.randint(20, S - 20)
        # Central bright spot
        for r in (2, 4, 7, 11):
            for ox in (-S, 0, S):
                for oy in (-S, 0, S):
                    draw.ellipse(
                        (tx - r + ox, ty - r + oy,
                         tx + r + ox, ty + r + oy),
                        fill=clamp(240 - r * 6))

    # Glowing ember dots
    for _ in range(rng.randint(28, 48)):
        cx = rng.randrange(S)
        cy = rng.randrange(S)
        for ox in (-S, 0, S):
            for oy in (-S, 0, S):
                draw.ellipse(
                    (cx - 1 + ox, cy - 1 + oy,
                     cx + 1 + ox, cy + 1 + oy),
                    fill=rng.randint(178, 242))

    # Dark recesses for contrast
    for _ in range(rng.randint(12, 22)):
        cx = rng.randrange(S)
        cy = rng.randrange(S)
        rr = rng.randint(2, 5)
        for ox in (-S, 0, S):
            for oy in (-S, 0, S):
                draw.ellipse(
                    (cx - rr + ox, cy - rr + oy,
                     cx + rr + ox, cy + rr + oy),
                    fill=rng.randint(22, 46))


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
# Emissive colourise — uses amber fullbright palette entries for glow
# ═══════════════════════════════════════════════════════════════════════════

def colourise_emissive(
    height: Image.Image,
    rng: random.Random,
    base_rgb: tuple[int, int, int],
    gain: float,
) -> Image.Image:
    """Colourise the emissive height field with amber glow colours.

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
        glow_factor = max(0.0, (h - 0.45) / 0.55)

        # Dark stone base (not glowing)
        dark_r = clamp(base_rgb[0] * 0.52 + (m - 0.5) * 38)
        dark_g = clamp(base_rgb[1] * 0.52 + (m - 0.5) * 32)
        dark_b = clamp(base_rgb[2] * 0.48 + (m - 0.5) * 24)

        # Glow colours (amber through warm gold)
        glow_r = clamp(215 + h * 40 + m * 18)
        glow_g = clamp(130 + h * 88 + m * 32)
        glow_b = clamp(32 + h * 65 + m * 18)

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
        "name = \"richness_egyptian_v1\"\n"
        "version = \"1.0.0\"\n"
        "wad = \"richness_egyptian_v1.wad\"\n"
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
        '  "Richness Egyptian v1 Theme" generated by the bsp_generator project.\n',
        encoding="utf-8",
    )

    (out_dir / "provenance.toml").write_text(
        "[theme]\n"
        "name = \"richness_egyptian_v1\"\n"
        "license = \"CC0-1.0\"\n"
        "generator = \"build.py\"\n"
        "generator_language = \"Python 3\"\n"
        "generator_dependency = \"Pillow (PIL)\"\n"
        "texture_method = \"procedural\"\n"
        "texture_size = 256\n"
        "master_seed = \"0x52455631\"\n"
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
        add_stepped_pylon_wall(height, rng)
    elif name == "floor":
        add_hypostyle_floor(height, rng)
    elif name == "ceiling":
        add_obelisk_cavetto_ceiling(height, rng)
    elif name == "accent":
        add_shrine_accent(height, rng)
    elif name == "portal":
        add_pylon_portal(height, rng)
    elif name == "vertical":
        add_obelisk_vertical(height, rng)
    elif name == "cave":
        add_tomb_cave(height, rng)
    elif name == "prop":
        add_canopic_prop(height, rng)
    elif name == "emissive":
        add_amber_emissive(height, rng)
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
    for index, (name, base_rgb, colour_gain, normal_strength, mean_gloss) in enumerate(EGYPTIAN_DEFS):
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

    (out_dir / "richness_egyptian_v1.wad").write_bytes(make_wad2(wad_entries))
    write_static_files(out_dir)

    print(f"Richness Egyptian v1 theme generated in {out_dir.resolve()}")


if __name__ == "__main__":
    main()
