#!/usr/bin/env python3
"""Deterministic CC0 Stone Beta theme asset generator.

Generates all theme assets from a fixed seed with zero external dependencies
beyond the Python stdlib. Two independent runs produce byte-identical output.

Outputs (placed in the target directory, default CWD):
- palette.lmp              — 256-color Quake-style palette (768 bytes)
- cc0_stone_beta.wad       — WAD2 archive with 4 role miptex entries plus `skip`
- theme.toml               — texture role bindings
- LICENSE                  — CC0-1.0 public domain dedication
- textures/
    stone_floor_basecolor.png
    stone_floor_norm.png
    stone_floor_gloss.png
    stone_wall_basecolor.png
    stone_wall_norm.png
    stone_wall_gloss.png
    stone_ceiling_basecolor.png
    stone_ceiling_norm.png
    stone_ceiling_gloss.png
    stone_accent_basecolor.png
    stone_accent_norm.png
    stone_accent_gloss.png

Usage:
    python3 build.py [output_directory]
"""

import struct
import zlib
import os
import sys


# ═══════════════════════════════════════════════════════════════════════════
# Deterministic RNG
# ═══════════════════════════════════════════════════════════════════════════

class RNG:
    """31-bit LCG matching glibc rand() behaviour."""

    def __init__(self, seed):
        self.state = seed & 0x7FFFFFFF

    def next(self):
        self.state = (self.state * 1103515245 + 12345) & 0x7FFFFFFF
        return self.state

    def range(self, lo, hi):
        """Return a deterministic integer in [lo, hi)."""
        return lo + (self.next() % (hi - lo))


# ═══════════════════════════════════════════════════════════════════════════
# PNG writer (pure-Python, no Pillow dependency)
# ═══════════════════════════════════════════════════════════════════════════

def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    """Return a complete PNG chunk: length + type + data + CRC32."""
    raw = chunk_type + data
    length = struct.pack('>I', len(data))
    crc = struct.pack('>I', zlib.crc32(raw) & 0xFFFFFFFF)
    return length + raw + crc


def make_png_rgb(width: int, height: int, pixels: bytes) -> bytes:
    """Encode an 8-bit RGB PNG from raw (R,G,B,...) pixel bytes."""
    sig = b'\x89PNG\r\n\x1a\n'
    # IHDR: bit depth 8, colour type 2 (RGB)
    ihdr = struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0)
    out = sig + _png_chunk(b'IHDR', ihdr)
    # IDAT: one filter-byte (0=None) per row, then row data
    row_bytes = width * 3
    raw = b''
    for y in range(height):
        raw += b'\x00'
        raw += pixels[y * row_bytes : (y + 1) * row_bytes]
    out += _png_chunk(b'IDAT', zlib.compress(raw))
    out += _png_chunk(b'IEND', b'')
    return out


def make_png_solid(width: int, height: int, r: int, g: int, b: int) -> bytes:
    """PNG filled with a single RGB colour."""
    return make_png_rgb(width, height, bytes([r, g, b]) * (width * height))


# ═══════════════════════════════════════════════════════════════════════════
# Palette — 256-entry Quake-style .lmp (768 bytes RGB)
# ═══════════════════════════════════════════════════════════════════════════

def make_palette() -> bytes:
    """Return a 768-byte RGB palette with brown/grey ramps and fullbrights."""
    pal = bytearray()

    # Row  0 (  0- 15): grey ramp  — black → white (16 steps)
    for i in range(16):
        v = i * 255 // 15
        pal.extend([v, v, v])

    # Row  1 ( 16- 31): brown ramp  — dark brown → light tan
    for i in range(16):
        r = 60 + i * 120 // 15
        g = 35 + i * 100 // 15
        b = 20 + i * 80 // 15
        pal.extend([r, g, b])

    # Row  2 ( 32- 47): warm beige ramp
    for i in range(16):
        r = 160 + i * 80 // 15
        g = 130 + i * 70 // 15
        b = 100 + i * 60 // 15
        pal.extend([r, g, b])

    # Row  3 ( 48- 63): cool grey ramp
    for i in range(16):
        v = 32 + i * 200 // 15
        pal.extend([v, v, v + 10 if v + 10 <= 255 else 255])

    # Rows 4-13 ( 64-223): graduated variation ramps (10 rows × 16 cols)
    for row in range(10):
        # Each row sweeps a slightly different hue family
        base_r = 35 + row * 20
        base_g = 25 + row * 15
        base_b = 15 + row * 12
        for i in range(16):
            r = min(255, base_r + i * 13)
            g = min(255, base_g + i * 11)
            b = min(255, base_b + i * 9)
            pal.extend([r, g, b])

    # Rows 14-15 (224-255): fullbrights — bright saturated hues
    for i in range(32):
        hue = i * 256 // 32
        if hue < 85:
            r, g, b = 255, hue * 3, 0
        elif hue < 170:
            r, g, b = 255 - (hue - 85) * 3, 255, 0
        else:
            r, g, b = 0, 255, (hue - 170) * 3
        pal.extend([min(255, r), min(255, g), min(255, b)])

    return bytes(pal)


# ═══════════════════════════════════════════════════════════════════════════
# Nearest-neighbour palette quantisation
# ═══════════════════════════════════════════════════════════════════════════

def quantize(rgb_pixels: bytes, palette: bytes) -> list:
    """Map every RGB triple to the closest palette index (Euclidean)."""
    indices = []
    for i in range(0, len(rgb_pixels), 3):
        r, g, b = rgb_pixels[i], rgb_pixels[i + 1], rgb_pixels[i + 2]
        best_idx = 0
        best_dist = 256 * 256 * 3 + 1
        for idx in range(256):
            pr = palette[idx * 3]
            pg = palette[idx * 3 + 1]
            pb = palette[idx * 3 + 2]
            dr = r - pr
            dg = g - pg
            db = b - pb
            dist = dr * dr + dg * dg + db * db
            if dist < best_dist:
                best_dist = dist
                best_idx = idx
        indices.append(best_idx)
    return indices


# ═══════════════════════════════════════════════════════════════════════════
# Mipmap downsampling
# ═══════════════════════════════════════════════════════════════════════════

def downsample_mip(indices: list, w: int, h: int, palette: bytes) -> list:
    """Average each 2×2 block in RGB space, then re-quantise to palette."""
    nw, nh = w // 2, h // 2
    result = []
    for y in range(nh):
        for x in range(nw):
            i00 = indices[(y * 2) * w + (x * 2)]
            i01 = indices[(y * 2) * w + (x * 2 + 1)]
            i10 = indices[(y * 2 + 1) * w + (x * 2)]
            i11 = indices[(y * 2 + 1) * w + (x * 2 + 1)]
            r = (palette[i00 * 3]     + palette[i01 * 3]     +
                 palette[i10 * 3]     + palette[i11 * 3])     // 4
            g = (palette[i00 * 3 + 1] + palette[i01 * 3 + 1] +
                 palette[i10 * 3 + 1] + palette[i11 * 3 + 1]) // 4
            b = (palette[i00 * 3 + 2] + palette[i01 * 3 + 2] +
                 palette[i10 * 3 + 2] + palette[i11 * 3 + 2]) // 4
            # Nearest palette match
            best = 0
            best_d = 999999
            for idx in range(256):
                dr = r - palette[idx * 3]
                dg = g - palette[idx * 3 + 1]
                db = b - palette[idx * 3 + 2]
                d = dr * dr + dg * dg + db * db
                if d < best_d:
                    best_d = d
                    best = idx
            result.append(best)
    return result


# ═══════════════════════════════════════════════════════════════════════════
# WAD2 builder
# ═══════════════════════════════════════════════════════════════════════════

def make_miptex(name: str, width: int, height: int,
                indices: list, palette: bytes) -> bytes:
    """Build a single Quake miptex lump (header + 4 mip levels)."""
    assert width == 64 and height == 64, "only 64×64 supported"

    mip0 = bytes(indices)                              # 4096 B
    mip1_idx = downsample_mip(indices, 64, 64, palette) # 1024 B
    mip2_idx = downsample_mip(mip1_idx, 32, 32, palette) # 256 B
    mip3_idx = downsample_mip(mip2_idx, 16, 16, palette) #  64 B

    mip1 = bytes(mip1_idx)
    mip2 = bytes(mip2_idx)
    mip3 = bytes(mip3_idx)

    HDR = 40
    off0 = HDR
    off1 = off0 + len(mip0)
    off2 = off1 + len(mip1)
    off3 = off2 + len(mip2)

    name_bytes = name.encode('ascii')
    if len(name_bytes) > 15:
        raise ValueError(f"texture name too long: {name}")
    name_padded = name_bytes.ljust(16, b'\x00')

    header = struct.pack('<16sIIIIII',
                         name_padded, width, height,
                         off0, off1, off2, off3)
    return header + mip0 + mip1 + mip2 + mip3


def make_wad2(entries: list) -> bytes:
    """Build a WAD2 file from a list of (name, miptex_bytes) tuples.

    WAD2 layout:
      - Header: magic(4) + numlumps(4 LE) + infotableofs(4 LE)      [12 B]
      - Lump data: concatenated miptex blobs
      - Info table: numlumps × 32 B directory entries
    """
    num = len(entries)
    HDR_SIZE = 12
    ENTRY_SIZE = 32

    lumps = [e[1] for e in entries]
    lump_start = HDR_SIZE
    infotableofs = lump_start + sum(len(l) for l in lumps)

    header = struct.pack('<4sii', b'WAD2', num, infotableofs)

    info = b''
    cur = lump_start
    for (name, data) in entries:
        name_bytes = name.encode('ascii')
        if len(name_bytes) > 15:
            raise ValueError(f"texture name too long: {name}")
        name_padded = name_bytes.ljust(16, b'\x00')
        sz = len(data)
        # Quake WAD2 lumpinfo_t layout:
        # filepos, disksize, size, type, compression, pad1, pad2, name[16]
        info += struct.pack('<iiIBBH16s',
                            cur,            # filepos
                            sz,             # disksize
                            sz,             # size (uncompressed)
                            0x44,           # type = miptex
                            0,              # compression = none
                            0,              # padding
                            name_padded)
        cur += sz

    return header + b''.join(lumps) + info


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

# Fixed master seed: "CC0S" in ASCII → 0x43_43_30_53
MASTER_SEED = 0x43433053

TEXTURE_DEFS = [
    # (name,          base-R, base-G, base-B)
    ('stone_floor',   90,     65,     45),    # dark grey-brown
    ('stone_wall',    140,    135,    130),   # medium grey
    ('stone_ceiling', 190,    185,    180),   # light grey
    ('stone_accent',  200,    170,    140),   # warm beige
]


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else '.'
    tex_dir = os.path.join(out_dir, 'textures')
    os.makedirs(tex_dir, exist_ok=True)

    rng = RNG(MASTER_SEED)

    # ── Palette ────────────────────────────────────────────────────────
    palette = make_palette()
    with open(os.path.join(out_dir, 'palette.lmp'), 'wb') as f:
        f.write(palette)

    # ── Textures ───────────────────────────────────────────────────────
    wad_entries = []

    for name, base_r, base_g, base_b in TEXTURE_DEFS:
        # Generate noisy RGB pixels (deterministic via master RNG)
        pixels = bytearray()
        for _y in range(64):
            for _x in range(64):
                noise = rng.range(-12, 13)
                r = base_r + noise + rng.range(-3, 4)
                g = base_g + noise + rng.range(-3, 4)
                b = base_b + noise + rng.range(-3, 4)
                pixels.extend([
                    max(0, min(255, r)),
                    max(0, min(255, g)),
                    max(0, min(255, b)),
                ])
        pixels = bytes(pixels)

        # Base-colour PNG
        png_base = make_png_rgb(64, 64, pixels)
        with open(os.path.join(tex_dir, f'{name}_basecolor.png'), 'wb') as f:
            f.write(png_base)

        # Normal map  — flat blue (tangent-space "no detail")
        png_norm = make_png_solid(64, 64, 128, 128, 255)
        with open(os.path.join(tex_dir, f'{name}_norm.png'), 'wb') as f:
            f.write(png_norm)

        # Gloss map  — medium roughness (0.5 linear ≈ 128 sRGB grey)
        png_gloss = make_png_solid(64, 64, 128, 128, 128)
        with open(os.path.join(tex_dir, f'{name}_gloss.png'), 'wb') as f:
            f.write(png_gloss)

        # Quantise to palette for WAD miptex
        indices = quantize(pixels, palette)
        miptex = make_miptex(name, 64, 64, indices, palette)
        wad_entries.append((name, miptex))

    # ericw-tools reserves the conventional `skip` material while building
    # hulls even when no source face names it. Keep a compiler-only miptex in
    # the WAD so a clean generated-map compile has no missing-texture warning.
    # It has no theme role and no renderer companion textures.
    skip_indices = [0] * (64 * 64)
    wad_entries.append(('skip', make_miptex('skip', 64, 64, skip_indices, palette)))

    # ── WAD2 archive ───────────────────────────────────────────────────
    wad_data = make_wad2(wad_entries)
    with open(os.path.join(out_dir, 'cc0_stone_beta.wad'), 'wb') as f:
        f.write(wad_data)

    # ── theme.toml ─────────────────────────────────────────────────────
    theme_toml = (
        '[roles]\n'
        'floor = "stone_floor"\n'
        'wall = "stone_wall"\n'
        'ceiling = "stone_ceiling"\n'
        'accent = "stone_accent"\n'
    )
    with open(os.path.join(out_dir, 'theme.toml'), 'w') as f:
        f.write(theme_toml)

    # ── LICENSE ────────────────────────────────────────────────────────
    license_text = (
        'CC0 1.0 Universal (CC0 1.0) Public Domain Dedication\n'
        '\n'
        'The person who associated a work with this deed has dedicated '
        'the work to the\n'
        'public domain by waiving all of his or her rights to the work '
        'worldwide under\n'
        'copyright law, including all related and neighboring rights, '
        'to the extent\n'
        'allowed by law.\n'
        '\n'
        'You can copy, modify, distribute and perform the work, even '
        'for commercial\n'
        'purposes, all without asking permission.\n'
        '\n'
        'Full license text: '
        'https://creativecommons.org/publicdomain/zero/1.0/legalcode\n'
        '\n'
        'Attribution (not required but appreciated):\n'
        '  "CC0 Stone Beta Theme" generated by the bsp_generator project.\n'
    )
    with open(os.path.join(out_dir, 'LICENSE'), 'w') as f:
        f.write(license_text)

    print(f'CC0 Stone Beta theme generated in {os.path.abspath(out_dir)}')


if __name__ == '__main__':
    main()
