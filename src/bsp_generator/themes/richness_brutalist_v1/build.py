#!/usr/bin/env python3
"""Deterministic procedural Richness Brutalist v1 theme asset generator.

Generates a complete CC0 Brutalist theme with raw concrete formwork walls
(board marks, tie holes), waffle-slab ceiling, brise-soleil grid accent,
pier portal with inward cantilever, ribbed pier vertical, sprayed-concrete
cave, industrial prop atlas, cold-flood fluorescent emissive identity,
and a compiler-only skip texture. Every visible texture has matching
basecolor, normal, and gloss companions.

Outputs (placed in target directory, default CWD):
- palette.lmp                     — 256-colour project-authored palette (768 bytes)
- richness_brutalist_v1.wad       — WAD2 archive with 9 visible miptex entries
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
        "Richness Brutalist v1 generation requires Pillow: pip install Pillow"
    ) from error


TEXTURE_SIZE = 256
SKIP_TEXTURE_SIZE = 64
# "RBV1" in ASCII — Richness Brutalist V1
MASTER_SEED = 0x52425631
PNG_SAVE_OPTIONS = {"format": "PNG", "compress_level": 9, "optimize": False}

# ── Texture definitions ────────────────────────────────────────────────────
# (name, base-RGB, height-gain, normal-strength, mean-gloss)

BRUTALIST_DEFS = (
    ("wall",     (158, 156, 160), 0.70, 1.32, 88),
    ("floor",    (142, 140, 148), 0.76, 1.38, 80),
    ("ceiling",  (168, 166, 174), 0.52, 0.98, 102),
    ("accent",   (164, 162, 168), 0.58, 1.08, 100),
    ("portal",   (152, 150, 158), 0.66, 1.24, 84),
    ("vertical", (148, 146, 154), 0.62, 1.16, 92),
    ("cave",     (120, 118, 126), 0.82, 1.42, 68),
    ("prop",     (156, 154, 162), 0.54, 1.04, 106),
    ("emissive", (154, 156, 168), 0.56, 1.08, 98),
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
    """Return a project-authored 256-entry Brutalist concrete palette.

    Entries 0..223 are cool concrete ramps (blue-grey, cement grey,
    charcoal, weathered industrial grey).  Entries 224..255 are cold
    fullbrights (cool fluorescent white through arctic blue-white)
    for the cold-flood emissive identity.
    """
    ramps = (
        ((10, 10, 12),   (218, 216, 222)),  # neutral cool greys
        ((32, 30, 36),   (160, 156, 166)),   # dark charcoal blue-grey
        ((48, 46, 52),   (184, 180, 188)),   # cement blue-grey
        ((64, 62, 70),   (200, 196, 206)),   # mid cool concrete
        ((56, 54, 62),   (170, 166, 176)),   # weathered industrial grey
        ((80, 78, 86),   (192, 188, 196)),   # light cool concrete
        ((100, 98, 106), (212, 208, 216)),   # pale cement
        ((118, 116, 124), (224, 220, 226)),  # bright concrete
        ((74, 72, 80),   (178, 174, 184)),   # cool formwork grey
        ((92, 90, 98),   (196, 192, 200)),   # aggregate grey
        ((108, 106, 114), (208, 204, 212)),  # washed concrete
        ((130, 128, 136), (228, 224, 230)),  # sun-bleached concrete
        ((144, 142, 150), (236, 232, 238)),  # pale dressed concrete
        ((160, 158, 166), (244, 240, 246)),  # light cap concrete
    )

    palette = bytearray()
    for start, end in ramps:
        for step in range(16):
            fraction = step / 15
            palette.extend(
                clamp(start[channel] + (end[channel] - start[channel]) * fraction)
                for channel in range(3)
            )

    # Cold fullbright entries (224–255): cool fluorescent white through
    # arctic blue-white.  These are the emissive glow colours that Quake
    # lightmapping leaves unmodulated (fullbright), producing the cold-flood
    # industrial fluorescent identity.
    fullbright_ramp = (
        (140, 148, 180),   # deep cool blue
        (160, 170, 200),   # cool blue
        (180, 190, 216),   # pale cool blue
        (196, 206, 228),   # cool sky
        (210, 218, 236),   # cool white-blue
        (222, 228, 242),   # bright cool white
        (234, 238, 248),   # fluorescent white
        (244, 246, 252),   # arctic white
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

def add_formwork_wall(height: Image.Image, rng: random.Random) -> None:
    """Raw concrete formwork wall: board marks, tie holes, panel joints.

    Béton brut aesthetic with horizontal board marks from plywood formwork,
    a regular grid of form-tie holes, and deep shadow reveals at panel
    boundaries.  The surface aggregate texture completes the raw concrete
    feel — monumental and oppressive without any suggestion of lightness
    or escape.
    """
    S = TEXTURE_SIZE
    draw = ImageDraw.Draw(height)

    # Panel grid: formwork panels divide the surface into large rectangles.
    # Typical formwork panels were 4'×8' or similar.  We use ~3 columns
    # and ~4 rows of panels.
    panel_cols = 3
    panel_rows = 4
    col_w = S // panel_cols
    row_h = S // panel_rows

    # Draw panel boundary reveals — deep shadow at formwork joints
    for col in range(1, panel_cols):
        x = col * col_w + rng.randint(-6, 6)
        x = max(4, min(S - 4, x))
        seamless_draw(draw, "line",
                      points=[(x, 0), (x, S)],
                      fill=30, width=6)
        # Highlight edge on one side (cast-light feel)
        seamless_draw(draw, "line",
                      points=[(x + 3, 0), (x + 3, S)],
                      fill=118, width=1)

    for row in range(1, panel_rows):
        y = row * row_h + rng.randint(-6, 6)
        y = max(4, min(S - 4, y))
        seamless_draw(draw, "line",
                      points=[(0, y), (S, y)],
                      fill=30, width=6)
        seamless_draw(draw, "line",
                      points=[(0, y + 3), (S, y + 3)],
                      fill=118, width=1)

    # Horizontal board marks within each panel — subtle parallel lines
    # from plywood formwork grain.  We add several per panel.
    for panel_row in range(panel_rows):
        y0 = panel_row * row_h + 8
        y1 = min(S, y0 + row_h - 8)
        # 6–10 board marks per panel row
        num_marks = rng.randint(6, 10)
        for _ in range(num_marks):
            by = y0 + rng.randint(4, max(5, (y1 - y0) // num_marks * 2))
            by = max(y0 + 2, min(y1 - 2, by))
            # Slight horizontal offset per board
            ox = rng.randint(-3, 3)
            seamless_draw(draw, "line",
                          points=[(ox, by), (S + ox, by)],
                          fill=72, width=2, joint="curve")
            # Subtle highlight below board mark
            seamless_draw(draw, "line",
                          points=[(ox, by + 1), (S + ox, by + 1)],
                          fill=108, width=1, joint="curve")

    # Form-tie holes — regular grid of small circular recesses.
    # Ties were typically spaced 16–32" apart; at our scale ~16–32 pixels.
    tie_spacing = S // 9  # ~28 pixels
    for ty in range(tie_spacing // 2, S, tie_spacing):
        for tx in range(tie_spacing // 2, S, tie_spacing):
            # Slight jitter from exact grid
            jx = tx + rng.randint(-3, 3)
            jy = ty + rng.randint(-3, 3)
            # Tie hole: small dark circle
            seamless_draw(draw, "ellipse",
                          x0=jx - 2, y0=jy - 2,
                          x1=jx + 2, y1=jy + 2,
                          fill=rng.randint(16, 32))
            # Faint halo of discolouration around tie
            seamless_draw(draw, "ellipse",
                          x0=jx - 4, y0=jy - 4,
                          x1=jx + 4, y1=jy + 4,
                          fill=rng.randint(50, 70))

    # Surface chips and aggregate pops
    for _ in range(rng.randint(50, 80)):
        cx = rng.randrange(S)
        cy = rng.randrange(S)
        rr = rng.randint(1, 3)
        seamless_draw(draw, "ellipse",
                      x0=cx - rr, y0=cy - rr,
                      x1=cx + rr, y1=cy + rr,
                      fill=rng.randint(38, 68))


def add_slab_floor(height: Image.Image, rng: random.Random) -> None:
    """Concrete slab floor: expansion joints, exposed aggregate, wear.

    Saw-cut control joints divide the slab into large panels.  Within each,
    subtle exposed aggregate texture and faint trowel marks.  Industrial
    scuffs and wear marks suggest heavy use without implying anything
    beyond the sealed envelope.
    """
    S = TEXTURE_SIZE
    draw = ImageDraw.Draw(height)

    # Control/expansion joints — saw-cut grid
    joint_cols = 4
    joint_rows = 4
    col_w = S // joint_cols
    row_h = S // joint_rows

    for col in range(1, joint_cols):
        x = col * col_w + rng.randint(-10, 10)
        x = max(6, min(S - 6, x))
        seamless_draw(draw, "line",
                      points=[(x, 0), (x + rng.randint(-2, 2), S)],
                      fill=38, width=5)
        seamless_draw(draw, "line",
                      points=[(x + 2, 0), (x + 2, S)],
                      fill=112, width=1)

    for row in range(1, joint_rows):
        y = row * row_h + rng.randint(-10, 10)
        y = max(6, min(S - 6, y))
        seamless_draw(draw, "line",
                      points=[(0, y), (S, y)],
                      fill=38, width=5)
        seamless_draw(draw, "line",
                      points=[(0, y + 2), (S, y + 2)],
                      fill=112, width=1)

    # Exposed aggregate — small pebble dots from worn surface
    for _ in range(rng.randint(140, 200)):
        cx = rng.randrange(8, S - 8)
        cy = rng.randrange(8, S - 8)
        rr = rng.choice((1, 1, 1, 2))
        # Aggregate pebbles are lighter than matrix
        seamless_draw(draw, "ellipse",
                      x0=cx - rr, y0=cy - rr,
                      x1=cx + rr, y1=cy + rr,
                      fill=rng.randint(128, 182))

    # Trowel marks — faint sweeping arcs
    for _ in range(rng.randint(8, 14)):
        sx = rng.randrange(S)
        sy = rng.randrange(S)
        angle = rng.uniform(0, math.tau)
        length = rng.randint(24, 60)
        ex = int(sx + math.cos(angle) * length)
        ey = int(sy + math.sin(angle) * length)
        for ox in (-S, 0, S):
            for oy in (-S, 0, S):
                draw.line([(sx + ox, sy + oy), (ex + ox, ey + oy)],
                          fill=rng.randint(54, 84), width=2)

    # Industrial wear scuffs
    for _ in range(rng.randint(20, 40)):
        sx = rng.randrange(S)
        sy = rng.randrange(S)
        angle = rng.uniform(0, math.tau)
        length = rng.randint(10, 36)
        ex = int(sx + math.cos(angle) * length)
        ey = int(sy + math.sin(angle) * length)
        for ox in (-S, 0, S):
            for oy in (-S, 0, S):
                draw.line([(sx + ox, sy + oy), (ex + ox, ey + oy)],
                          fill=rng.randint(44, 66), width=3)

    # Surface pits
    for _ in range(rng.randint(80, 120)):
        cx = rng.randrange(S)
        cy = rng.randrange(S)
        rr = rng.randint(1, 2)
        seamless_draw(draw, "ellipse",
                      x0=cx - rr, y0=cy - rr,
                      x1=cx + rr, y1=cy + rr,
                      fill=rng.randint(44, 76))


def add_waffle_ceiling(height: Image.Image, rng: random.Random) -> None:
    """Waffle-slab / coffered ceiling: deep beam grid with recessed panels.

    Exposed structural ribs form a coffered grid overhead.  Deep beam
    shadows at each rib edge, darker coffer interiors, and occasional
    service penetrations (pipe/form-tie circular holes) create the
    oppressive weight of structural concrete pressing down.
    """
    S = TEXTURE_SIZE
    draw = ImageDraw.Draw(height)

    # Waffle grid: structural beams dividing the ceiling into coffers
    num_beams_h = 4
    num_beams_v = 4
    beam_w_h = S // (num_beams_h + 1)
    beam_w_v = S // (num_beams_v + 1)

    for b_idx in range(1, num_beams_h + 1):
        x = b_idx * beam_w_h + rng.randint(-8, 8)
        x = max(8, min(S - 8, x))
        # Deep beam shadow (left side of rib)
        seamless_draw(draw, "line",
                      points=[(x - 2, 0), (x - 2, S)],
                      fill=22, width=6)
        # Beam face highlight (right side catches light)
        seamless_draw(draw, "line",
                      points=[(x + 3, 0), (x + 3, S)],
                      fill=124, width=2)
        # Beam core
        seamless_draw(draw, "line",
                      points=[(x, 0), (x, S)],
                      fill=42, width=4)

    for b_idx in range(1, num_beams_v + 1):
        y = b_idx * beam_w_v + rng.randint(-8, 8)
        y = max(8, min(S - 8, y))
        seamless_draw(draw, "line",
                      points=[(0, y - 2), (S, y - 2)],
                      fill=22, width=6)
        seamless_draw(draw, "line",
                      points=[(0, y + 3), (S, y + 3)],
                      fill=124, width=2)
        seamless_draw(draw, "line",
                      points=[(0, y), (S, y)],
                      fill=42, width=4)

    # Coffer interiors — darker recessed panels
    for cy in range(num_beams_v + 1):
        for cx in range(num_beams_h + 1):
            # Centre of this coffer cell
            cell_x0 = cx * beam_w_h + 12
            cell_y0 = cy * beam_w_v + 12
            cell_x1 = min(S, (cx + 1) * beam_w_h - 10)
            cell_y1 = min(S, (cy + 1) * beam_w_v - 10)
            if cell_x1 <= cell_x0 or cell_y1 <= cell_y0:
                continue
            # Darken coffer interior
            cell_cx = (cell_x0 + cell_x1) // 2
            cell_cy = (cell_y0 + cell_y1) // 2
            cell_rx = (cell_x1 - cell_x0) // 3
            cell_ry = (cell_y1 - cell_y0) // 3
            seamless_draw(draw, "ellipse",
                          x0=cell_cx - cell_rx, y0=cell_cy - cell_ry,
                          x1=cell_cx + cell_rx, y1=cell_cy + cell_ry,
                          fill=rng.randint(28, 48))

    # Service penetrations — small circular holes in some coffers
    for _ in range(rng.randint(6, 14)):
        hx = rng.randint(20, S - 20)
        hy = rng.randint(20, S - 20)
        hr = rng.randint(2, 4)
        seamless_draw(draw, "ellipse",
                      x0=hx - hr, y0=hy - hr,
                      x1=hx + hr, y1=hy + hr,
                      fill=14)

    # Pores and air bubbles (concrete casting)
    for _ in range(rng.randint(60, 100)):
        cx = rng.randrange(S)
        cy = rng.randrange(S)
        seamless_draw(draw, "ellipse",
                      x0=cx - 1, y0=cy - 1,
                      x1=cx + 1, y1=cy + 1,
                      fill=rng.randint(36, 66))


def add_brise_soleil_accent(height: Image.Image, rng: random.Random) -> None:
    """Brise-soleil grid accent: geometric block grid with deep reveals.

    Alternating-height rectangular blocks arranged in a dense grid,
    with deep shadow channels between them.  The blocks suggest a
    structural sun-breaking grid repurposed as an interior architectural
    element — no light passes through, only oppressive depth.  Stepped
    cantilevered masses project inward.
    """
    S = TEXTURE_SIZE
    draw = ImageDraw.Draw(height)

    # Grid of blocks
    grid_x = 5
    grid_y = 5
    cell_w = S // grid_x
    cell_h = S // grid_y

    # Draw deep channel shadows first (between all blocks)
    for gx in range(grid_x):
        for gy in range(grid_y):
            bx = gx * cell_w + 4
            by = gy * cell_h + 4
            bw = cell_w - 8
            bh = cell_h - 8

            # Random height variation: some blocks project more
            projection = rng.choice((0, 1, 2))
            inset = rng.choice((0, 1, 1, 2))

            # Block shadow (deep reveal on left/top sides)
            seamless_draw(draw, "rectangle",
                          x0=bx + inset, y0=by + inset,
                          x1=bx + bw, y1=by + bh,
                          fill=24 + projection * 4,
                          outline=34 + projection * 6, width=3)
            # Block highlight (right/bottom edges catch light)
            if projection > 0:
                seamless_draw(draw, "rectangle",
                              x0=bx + inset + 2, y0=by + inset + 2,
                              x1=bx + bw - 1, y1=by + bh - 1,
                              fill=0,
                              outline=108 + projection * 8, width=1)

    # Channel cross-shadows — deeper where channels intersect
    for gx in range(1, grid_x):
        x = gx * cell_w + rng.randint(-3, 3)
        x = max(4, min(S - 4, x))
        seamless_draw(draw, "line",
                      points=[(x, 0), (x, S)],
                      fill=18, width=7)
    for gy in range(1, grid_y):
        y = gy * cell_h + rng.randint(-3, 3)
        y = max(4, min(S - 4, y))
        seamless_draw(draw, "line",
                      points=[(0, y), (S, y)],
                      fill=18, width=7)

    # Stepped inner cantilever — some blocks have an inner step
    for gx in range(grid_x):
        for gy in range(grid_y):
            if rng.random() < 0.4:
                bx = gx * cell_w + 8
                by = gy * cell_h + 8
                bw = cell_w - 16
                bh = cell_h - 16
                if bw <= 4 or bh <= 4:
                    continue
                seamless_draw(draw, "rectangle",
                              x0=bx, y0=by,
                              x1=bx + bw, y1=by + bh,
                              fill=0, outline=52, width=2)
                seamless_draw(draw, "rectangle",
                              x0=bx + 1, y0=by + 1,
                              x1=bx + bw - 1, y1=by + bh - 1,
                              fill=0, outline=94, width=1)

    # Surface aggregate texture on block faces
    for _ in range(rng.randint(40, 70)):
        cx = rng.randrange(S)
        cy = rng.randrange(S)
        seamless_draw(draw, "ellipse",
                      x0=cx - 1, y0=cy - 1,
                      x1=cx + 1, y1=cy + 1,
                      fill=rng.randint(60, 100))


def add_pier_portal(height: Image.Image, rng: random.Random) -> None:
    """Pier portal: massive inward-cantilevered concrete gateway.

    Two massive piers flank a deep central opening.  A heavy lintel
    spans across, projecting inward with a stepped cantilever detail.
    Deep shadow reveals at every joint.  Horizontal construction joints
    on the pier faces complete the monumental, oppressive gateway.
    No opening implies exterior — this is pure structural mass.
    """
    S = TEXTURE_SIZE
    draw = ImageDraw.Draw(height)

    # Central opening — trapezoidal, wider at bottom
    opening_top_w = S // 5
    opening_bot_w = S // 3
    opening_top_y = S // 4
    opening_bot_y = S - S // 6

    top_left = (S - opening_top_w) // 2
    top_right = top_left + opening_top_w
    bot_left = (S - opening_bot_w) // 2
    bot_right = bot_left + opening_bot_w

    # Deep portal interior — stepped inward cantilever
    # Outermost reveal (darkest, deepest)
    outer_pts = [
        (top_left + 10, opening_top_y + 10),
        (top_right - 10, opening_top_y + 10),
        (bot_right - 12, opening_bot_y - 12),
        (bot_left + 12, opening_bot_y - 12),
    ]
    seamless_draw(draw, "polygon", points=outer_pts, fill=16)

    # Mid reveal
    mid_pts = [
        (top_left + 6, opening_top_y + 6),
        (top_right - 6, opening_top_y + 6),
        (bot_right - 7, opening_bot_y - 7),
        (bot_left + 7, opening_bot_y - 7),
    ]
    seamless_draw(draw, "polygon", points=mid_pts, fill=26)

    # Inner reveal
    inner_pts = [
        (top_left + 3, opening_top_y + 3),
        (top_right - 3, opening_top_y + 3),
        (bot_right - 4, opening_bot_y - 4),
        (bot_left + 4, opening_bot_y - 4),
    ]
    seamless_draw(draw, "polygon", points=inner_pts, fill=40)

    # Lintel — massive horizontal band above opening
    lintel_y0 = opening_top_y - S // 10 + rng.randint(-3, 3)
    lintel_y0 = max(6, lintel_y0)
    lintel_y1 = opening_top_y + rng.randint(-2, 2)

    # Deep shadow under lintel (overhanging inward cantilever)
    seamless_draw(draw, "line",
                  points=[(0, lintel_y1), (S, lintel_y1)],
                  fill=14, width=10, joint="curve")
    # Lintel bottom edge
    seamless_draw(draw, "line",
                  points=[(0, lintel_y1 + 2), (S, lintel_y1 + 2)],
                  fill=38, width=3, joint="curve")
    # Lintel top highlight (catches cold light)
    seamless_draw(draw, "line",
                  points=[(0, lintel_y0), (S, lintel_y0)],
                  fill=126, width=3, joint="curve")
    seamless_draw(draw, "line",
                  points=[(0, lintel_y0 - 2), (S, lintel_y0 - 2)],
                  fill=96, width=1, joint="curve")

    # Inward cantilever step on lintel — projecting mass
    cantilever_y = lintel_y1 + S // 16
    seamless_draw(draw, "line",
                  points=[(0, cantilever_y), (S, cantilever_y)],
                  fill=20, width=8, joint="curve")
    seamless_draw(draw, "line",
                  points=[(0, cantilever_y + 1), (S, cantilever_y + 1)],
                  fill=112, width=1, joint="curve")

    # Pier mass — blocks flanking portal
    pier_cols = 3
    for side in (0, 1):
        if side == 0:
            px0, px1 = 4, top_left - 2
        else:
            px0, px1 = top_right + 2, S - 4

        # Horizontal construction joints on piers
        for row in range(0, 6):
            jy = row * S // 6 + rng.randint(-4, 4)
            jy = max(4, min(S - 4, jy))
            seamless_draw(draw, "line",
                          points=[(px0, jy), (px1, jy)],
                          fill=28, width=5, joint="curve")
            seamless_draw(draw, "line",
                          points=[(px0, jy + 1), (px1, jy + 1)],
                          fill=108, width=1, joint="curve")

        # Vertical tie-hole columns on pier faces
        for col in range(pier_cols):
            cx = px0 + (col + 1) * (px1 - px0) // (pier_cols + 1) + rng.randint(-4, 4)
            cx = max(px0 + 2, min(px1 - 2, cx))
            for ty in range(12, S - 12, S // 8):
                jy = ty + rng.randint(-4, 4)
                seamless_draw(draw, "ellipse",
                              x0=cx - 2, y0=jy - 2,
                              x1=cx + 2, y1=jy + 2,
                              fill=rng.randint(18, 34))

    # Surface distress
    for _ in range(rng.randint(45, 70)):
        cx = rng.randrange(S)
        cy = rng.randrange(S)
        rr = rng.randint(1, 3)
        seamless_draw(draw, "ellipse",
                      x0=cx - rr, y0=cy - rr,
                      x1=cx + rr, y1=cy + rr,
                      fill=rng.randint(40, 72))


def add_ribbed_pier_vertical(height: Image.Image, rng: random.Random) -> None:
    """Ribbed pier vertical: deep vertical ribs with formwork seams.

    Exposed structural pier with deep vertical rib/fluting creating strong
    shadow lines.  Horizontal formwork board marks cross the ribs, and
    tie-hole columns run vertically.  The rhythm of ribs, seams, and ties
    creates the brutalist structural rhythm without any opening or relief.
    """
    S = TEXTURE_SIZE
    draw = ImageDraw.Draw(height)

    # Vertical ribs — deep protruding fins
    num_ribs = 5
    rib_spacing = S // (num_ribs + 1)

    for rib_idx in range(1, num_ribs + 1):
        rx = rib_idx * rib_spacing + rng.randint(-6, 6)
        rx = max(6, min(S - 6, rx))
        rib_w = rng.randint(8, 14)

        # Deep shadow on left side of rib
        seamless_draw(draw, "line",
                      points=[(rx - 1, 0), (rx - 1, S)],
                      fill=18, width=4)
        # Rib body shadow
        seamless_draw(draw, "rectangle",
                      x0=rx, y0=0, x1=rx + rib_w, y1=S,
                      fill=34)
        # Highlight on right edge of rib
        seamless_draw(draw, "line",
                      points=[(rx + rib_w + 1, 0), (rx + rib_w + 1, S)],
                      fill=126, width=2)

    # Horizontal formwork seams across ribs
    num_seams = 6
    for seam_idx in range(1, num_seams):
        sy = seam_idx * S // num_seams + rng.randint(-5, 5)
        sy = max(4, min(S - 4, sy))
        # Seam shadow
        seamless_draw(draw, "line",
                      points=[(0, sy), (S, sy)],
                      fill=24, width=5, joint="curve")
        # Seam highlight
        seamless_draw(draw, "line",
                      points=[(0, sy + 2), (S, sy + 2)],
                      fill=114, width=1, joint="curve")

    # Vertical tie-hole columns in the valleys between ribs
    for col_idx in range(num_ribs + 1):
        if col_idx == 0:
            cx = rib_spacing // 2 + rng.randint(-4, 4)
        elif col_idx == num_ribs:
            cx = num_ribs * rib_spacing + rib_spacing // 2 + rng.randint(-4, 4)
        else:
            cx = col_idx * rib_spacing + rib_spacing // 2 + rng.randint(-4, 4)
        cx = max(4, min(S - 4, cx))

        for ty in range(10, S - 10, S // 6):
            jy = ty + rng.randint(-4, 4)
            seamless_draw(draw, "ellipse",
                          x0=cx - 2, y0=jy - 2,
                          x1=cx + 2, y1=jy + 2,
                          fill=rng.randint(16, 30))
            # Halo
            seamless_draw(draw, "ellipse",
                          x0=cx - 4, y0=jy - 4,
                          x1=cx + 4, y1=jy + 4,
                          fill=rng.randint(48, 68))

    # Surface chips
    for _ in range(rng.randint(45, 75)):
        cx = rng.randrange(S)
        cy = rng.randrange(S)
        rr = rng.randint(1, 3)
        seamless_draw(draw, "ellipse",
                      x0=cx - rr, y0=cy - rr,
                      x1=cx + rr, y1=cy + rr,
                      fill=rng.randint(42, 76))


def add_sprayed_concrete_cave(height: Image.Image, rng: random.Random) -> None:
    """Sprayed concrete cave: raw unformed shotcrete, organic roughness.

    No formwork, no straight lines — this is tunnel-form or sprayed
    shotcrete with organic blob accumulations, drip forms, deep irregular
    crevices, and exposed aggregate.  The sealed, claustrophobic character
    emphasises oppressive mass with no suggestion of natural rock or
    exterior connection.
    """
    S = TEXTURE_SIZE
    draw = ImageDraw.Draw(height)

    # Large organic blob accumulations (shotcrete build-up)
    for _ in range(rng.randint(8, 14)):
        cx = rng.randrange(S)
        cy = rng.randrange(S)
        rx = rng.randint(20, 52)
        ry = rng.randint(16, 44)
        for ox in (-S, 0, S):
            for oy in (-S, 0, S):
                draw.ellipse(
                    (cx - rx + ox, cy - ry + oy,
                     cx + rx + ox, cy + ry + oy),
                    fill=rng.randint(56, 90))

    # Drip forms — elongated vertical blobs (shotcrete sag)
    for _ in range(rng.randint(15, 25)):
        dx = rng.randrange(S)
        dy = rng.randrange(S)
        drip_len = rng.randint(16, 40)
        drip_w = rng.randint(3, 7)
        for seg in range(3):
            sy = (dy + seg * drip_len // 3) % S
            ey = min(sy + drip_len // 3, S)
            for ox in (-S, 0, S):
                for oy in (-S, 0, S):
                    draw.ellipse(
                        (dx - drip_w + ox, sy + oy,
                         dx + drip_w + ox, ey + oy),
                        fill=rng.randint(48, 78))

    # Deep irregular crevices
    for _ in range(rng.randint(18, 30)):
        sx = rng.randrange(S)
        sy = rng.randrange(S)
        angle = rng.uniform(0, math.tau)
        length = rng.randint(14, 64)
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
                          fill=rng.randint(22, 44), width=rng.choice((2, 3, 4, 5)),
                          joint="curve")

    # Exposed aggregate bumps
    for _ in range(rng.randint(70, 120)):
        cx = rng.randrange(S)
        cy = rng.randrange(S)
        rx = rng.randint(1, 5)
        ry = rng.randint(1, 5)
        for ox in (-S, 0, S):
            for oy in (-S, 0, S):
                draw.ellipse(
                    (cx - rx + ox, cy - ry + oy,
                     cx + rx + ox, cy + ry + oy),
                    fill=rng.randint(138, 182))

    # Surface pits and air bubbles
    for _ in range(rng.randint(90, 150)):
        cx = rng.randrange(S)
        cy = rng.randrange(S)
        for ox in (-S, 0, S):
            for oy in (-S, 0, S):
                draw.ellipse(
                    (cx - 2 + ox, cy - 2 + oy,
                     cx + 2 + ox, cy + 2 + oy),
                    fill=rng.randint(24, 52))


def add_industrial_prop(height: Image.Image, rng: random.Random) -> None:
    """Industrial prop atlas: grating, bolted plates, pipe flanges.

    Diamond-plate/chequer-plate grid pattern with industrial bolt/rivet
    circles at regular intervals.  Pipe flange suggestions and mechanical
    hatch detail.  Smooth-finished concrete with industrial fixtures
    attached — the prop atlas for pipe galleries, machine mounts, and
    mechanical infrastructure within the sealed envelope.
    """
    S = TEXTURE_SIZE
    draw = ImageDraw.Draw(height)

    # Edge chamfer border (machined plate edge)
    chamfer = 10
    seamless_draw(draw, "rectangle",
                  x0=chamfer, y0=chamfer,
                  x1=S - chamfer, y1=S - chamfer,
                  fill=0, outline=48, width=4)
    seamless_draw(draw, "rectangle",
                  x0=chamfer + 2, y0=chamfer + 2,
                  x1=S - chamfer - 2, y1=S - chamfer - 2,
                  fill=0, outline=110, width=1)

    # Diamond plate / raised pattern grid
    # Diagonal cross-hatch lines forming diamond pattern
    diamond_spacing = S // 6
    for offset in range(-S, S * 2, diamond_spacing):
        # Diagonal lines one way
        seamless_draw(draw, "line",
                      points=[(offset, 0), (offset + S, S)],
                      fill=54, width=2, joint="curve")
        # Diagonal lines other way
        seamless_draw(draw, "line",
                      points=[(offset, S), (offset + S, 0)],
                      fill=50, width=2, joint="curve")

    # Raised diamond bumps at intersections
    for d1 in range(-1, 8):
        for d2 in range(-1, 8):
            x = (d1 - d2) * diamond_spacing // 2 + S // 2
            y = (d1 + d2) * diamond_spacing // 2
            if 4 < x < S - 4 and 4 < y < S - 4:
                seamless_draw(draw, "ellipse",
                              x0=x - 2, y0=y - 2,
                              x1=x + 2, y1=y + 2,
                              fill=98)

    # Bolted flange circles — industrial pipe connections
    flange_positions = [
        (S // 5, S // 5),
        (S * 4 // 5, S // 5),
        (S // 5, S * 4 // 5),
        (S * 4 // 5, S * 4 // 5),
        (S // 2, S // 2),
    ]
    for fx, fy in flange_positions:
        fx = fx + rng.randint(-8, 8)
        fy = fy + rng.randint(-8, 8)
        fr = rng.randint(14, 20)
        # Outer flange ring
        seamless_draw(draw, "ellipse",
                      x0=fx - fr, y0=fy - fr,
                      x1=fx + fr, y1=fy + fr,
                      fill=0)
        for angle in range(0, 360, 15):
            rad = math.radians(angle)
            px = int(fx + fr * math.cos(rad))
            py = int(fy + fr * math.sin(rad))
            draw.ellipse((px - 1, py - 1, px + 1, py + 1), fill=48)
        # Inner flange ring
        ir = fr - 4
        seamless_draw(draw, "ellipse",
                      x0=fx - ir, y0=fy - ir,
                      x1=fx + ir, y1=fy + ir,
                      fill=0)
        for angle in range(0, 360, 20):
            rad = math.radians(angle)
            px = int(fx + ir * math.cos(rad))
            py = int(fy + ir * math.sin(rad))
            draw.ellipse((px - 1, py - 1, px + 1, py + 1), fill=68)
        # Central bolt hole
        seamless_draw(draw, "ellipse",
                      x0=fx - 3, y0=fy - 3,
                      x1=fx + 3, y1=fy + 3,
                      fill=20)

    # Bolt/rivet grid — regular mechanical fasteners
    bolt_spacing = S // 5
    for bx in range(bolt_spacing // 2, S, bolt_spacing):
        for by in range(bolt_spacing // 2, S, bolt_spacing):
            jx = bx + rng.randint(-3, 3)
            jy = by + rng.randint(-3, 3)
            # Bolt head: small raised circle
            seamless_draw(draw, "ellipse",
                          x0=jx - 2, y0=jy - 2,
                          x1=jx + 2, y1=jy + 2,
                          fill=86)
            # Bolt shadow
            seamless_draw(draw, "ellipse",
                          x0=jx - 3, y0=jy - 3,
                          x1=jx + 1, y1=jy + 1,
                          fill=46)

    # Parallel tool/grinding marks on plate surfaces
    for _ in range(rng.randint(12, 20)):
        x = rng.randrange(chamfer + 4, S - chamfer - 4)
        y = rng.randrange(chamfer + 4, S - chamfer - 4)
        direction = rng.choice(("h", "v"))
        length = rng.randint(12, 36)
        if direction == "h":
            ex, ey = x + length, y + rng.randint(-1, 1)
        else:
            ex, ey = x + rng.randint(-1, 1), y + length
        ex = max(0, min(S - 1, ex))
        ey = max(0, min(S - 1, ey))
        for ox in (-S, 0, S):
            for oy in (-S, 0, S):
                draw.line([(x + ox, y + oy), (ex + ox, ey + oy)],
                          fill=rng.randint(66, 94), width=1)

    # Subtle surface wear
    for _ in range(rng.randint(30, 55)):
        cx = rng.randrange(S)
        cy = rng.randrange(S)
        for ox in (-S, 0, S):
            for oy in (-S, 0, S):
                draw.ellipse(
                    (cx - 1 + ox, cy - 1 + oy,
                     cx + 1 + ox, cy + 1 + oy),
                    fill=rng.randint(58, 96))


def add_cold_flood_emissive(height: Image.Image, rng: random.Random) -> None:
    """Cold-flood fluorescent emissive: cool industrial glow identity.

    Subtle concrete formwork backdrop with cool fluorescent tube ghost
    hints, industrial crack network with cold blue-white fullbright glow,
    and scattered fixture points.  The cold-flood palette entries produce
    an unmodulated fluorescent industrial lighting character — harsh,
    institutional, sealed.  No warm tones; pure cold industrial light.
    """
    S = TEXTURE_SIZE
    draw = ImageDraw.Draw(height)

    # Subtle formwork backdrop — faint panel lines
    num_panels = 4
    panel_h = S // num_panels
    for row in range(1, num_panels):
        y = row * panel_h + rng.randint(-3, 3)
        y = max(4, min(S - 4, y))
        for ox in (-S, 0, S):
            for oy in (-S, 0, S):
                draw.line([(0 + ox, y + oy), (S + ox, y + oy)],
                          fill=62, width=3, joint="curve")

    # Fluorescent tube ghost hints — horizontal bright bars
    for _ in range(rng.randint(3, 6)):
        tx = rng.randint(20, S - 20)
        ty = rng.randint(16, S - 16)
        tube_len = rng.randint(40, 120)
        tube_w = rng.randint(2, 4)

        # Bright tube core (high palette → fullbright)
        for ox in (-S, 0, S):
            for oy in (-S, 0, S):
                draw.ellipse(
                    (tx + ox, ty - tube_w + oy,
                     tx + tube_len + ox, ty + tube_w + oy),
                    fill=rng.randint(220, 248))
                # Tube halo (medium fullbright)
                draw.ellipse(
                    (tx - 2 + ox, ty - tube_w - 4 + oy,
                     tx + tube_len + 2 + ox, ty + tube_w + 4 + oy),
                    fill=rng.randint(175, 215))

    # Cool crack network with blue-white fullbright glow
    for _ in range(rng.randint(16, 26)):
        sx = rng.randrange(S)
        sy = rng.randrange(S)
        # Industrial cracks: tend toward orthogonal (concrete stress)
        base_angle = rng.choice((0, math.pi / 2, math.pi / 4, -math.pi / 4))
        angle = base_angle + rng.uniform(-0.3, 0.3)
        segments = rng.randint(3, 6)
        points = [(sx, sy)]
        for seg in range(segments):
            if rng.random() < 0.3:
                angle = base_angle + rng.choice((-math.pi / 2, math.pi / 2,
                                                  math.pi / 4, -math.pi / 4))
            angle += rng.uniform(-0.4, 0.4)
            dist = rng.randint(8, 34)
            sx = int(sx + math.cos(angle) * dist)
            sy = int(sy + math.sin(angle) * dist)
            points.append((sx % S, sy % S))

        # Glow core (bright fullbright — cool white)
        for ox in (-S, 0, S):
            for oy in (-S, 0, S):
                draw.line([(px + ox, py + oy) for px, py in points],
                          fill=rng.randint(200, 244), width=2, joint="curve")
        # Glow halo (medium fullbright — cool blue)
        for ox in (-S, 0, S):
            for oy in (-S, 0, S):
                draw.line([(px + ox, py + oy) for px, py in points],
                          fill=rng.randint(155, 198), width=6, joint="curve")

    # Industrial fixture glow points — bright cold dots
    for _ in range(rng.randint(25, 45)):
        cx = rng.randrange(S)
        cy = rng.randrange(S)
        # Bright core
        for ox in (-S, 0, S):
            for oy in (-S, 0, S):
                draw.ellipse(
                    (cx - 1 + ox, cy - 1 + oy,
                     cx + 1 + ox, cy + 1 + oy),
                    fill=rng.randint(210, 248))
                # Halo
                draw.ellipse(
                    (cx - 3 + ox, cy - 3 + oy,
                     cx + 3 + ox, cy + 3 + oy),
                    fill=rng.randint(165, 205))

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
                    fill=rng.randint(18, 40))


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
    """Turn the height field into restrained, role-specific concrete albedo.

    Cool-toned concrete with blue-grey bias: R channel slightly suppressed
    relative to G and B to produce the cold institutional concrete feel.
    """
    mottling = centred(periodic_noise(rng, 16, size), 0.33)
    tone = average(centred(height, 0.92), mottling)
    # Cool bias: R gain slightly lower, B gain slightly higher
    channel_gains = (gain * 0.94, gain, gain * 1.06)
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
    """Create spatially useful 0.30–0.50 concrete gloss from the height field.

    Dark cracks, tie holes, and formwork seams remain rougher while broad
    smooth concrete faces get a modestly glossier response.  The cold
    institutional concrete has slightly higher gloss in smooth areas than
    natural stone.
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
# Emissive colourise — uses cold fullbright palette entries for glow
# ═══════════════════════════════════════════════════════════════════════════

def colourise_emissive(
    height: Image.Image,
    rng: random.Random,
    base_rgb: tuple[int, int, int],
    gain: float,
) -> Image.Image:
    """Colourise the emissive height field with cold fluorescent glow.

    Brighter pixels (the glow cracks and tube ghosts) map into the cold
    fullbright palette range (entries 224–255) which Quake renders without
    lightmap modulation.  Darker pixels map to cool dark concrete.
    Blue channel is emphasised over red for the cold industrial feel.
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

        # Blend base concrete and glow based on height value
        glow_factor = max(0.0, (h - 0.45) / 0.55)

        # Dark cool concrete base (not glowing)
        dark_r = clamp(base_rgb[0] * 0.48 + (m - 0.5) * 34)
        dark_g = clamp(base_rgb[1] * 0.52 + (m - 0.5) * 36)
        dark_b = clamp(base_rgb[2] * 0.56 + (m - 0.5) * 40)

        # Glow colours (cool fluorescent: blue-weighted)
        glow_r = clamp(175 + h * 65 + m * 22)
        glow_g = clamp(190 + h * 58 + m * 22)
        glow_b = clamp(210 + h * 45 + m * 18)

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
        "name = \"richness_brutalist_v1\"\n"
        "version = \"1.0.0\"\n"
        "wad = \"richness_brutalist_v1.wad\"\n"
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
        '  "Richness Brutalist v1 Theme" generated by the bsp_generator project.\n',
        encoding="utf-8",
    )

    (out_dir / "provenance.toml").write_text(
        "[theme]\n"
        "name = \"richness_brutalist_v1\"\n"
        "license = \"CC0-1.0\"\n"
        "generator = \"build.py\"\n"
        "generator_language = \"Python 3\"\n"
        "generator_dependency = \"Pillow (PIL)\"\n"
        "texture_method = \"procedural\"\n"
        "texture_size = 256\n"
        "master_seed = \"0x52425631\"\n"
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
        add_formwork_wall(height, rng)
    elif name == "floor":
        add_slab_floor(height, rng)
    elif name == "ceiling":
        add_waffle_ceiling(height, rng)
    elif name == "accent":
        add_brise_soleil_accent(height, rng)
    elif name == "portal":
        add_pier_portal(height, rng)
    elif name == "vertical":
        add_ribbed_pier_vertical(height, rng)
    elif name == "cave":
        add_sprayed_concrete_cave(height, rng)
    elif name == "prop":
        add_industrial_prop(height, rng)
    elif name == "emissive":
        add_cold_flood_emissive(height, rng)
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
    for index, (name, base_rgb, colour_gain, normal_strength, mean_gloss) in enumerate(BRUTALIST_DEFS):
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

    (out_dir / "richness_brutalist_v1.wad").write_bytes(make_wad2(wad_entries))
    write_static_files(out_dir)

    print(f"Richness Brutalist v1 theme generated in {out_dir.resolve()}")


if __name__ == "__main__":
    main()
