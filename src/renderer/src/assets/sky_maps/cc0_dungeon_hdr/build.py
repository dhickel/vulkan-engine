#!/usr/bin/env python3
"""Build the project-authored CC0 Dungeon HDR cubemap.

The six Radiance RGBE faces form the renderer's default IBL source. They are
procedurally generated from fixed equations only; no external artwork is
sampled or transformed. Run from this directory or pass an output directory:

    python3 build.py [output_directory]
"""

from __future__ import annotations

from math import atan2, cos, pi, sin, sqrt
from pathlib import Path
import sys

FACE_SIZE = 256
FACES = ("px", "nx", "py", "ny", "pz", "nz")


def normalize(v: tuple[float, float, float]) -> tuple[float, float, float]:
    length = sqrt(sum(component * component for component in v))
    return tuple(component / length for component in v)


def cube_direction(face: str, u: float, v: float) -> tuple[float, float, float]:
    # Vulkan cubemap face orientation, with v increasing down the scanline.
    directions = {
        "px": (1.0, -v, -u),
        "nx": (-1.0, -v, u),
        "py": (u, 1.0, v),
        "ny": (u, -1.0, -v),
        "pz": (u, -v, 1.0),
        "nz": (-u, -v, -1.0),
    }
    return normalize(directions[face])


def add_scaled(
    target: list[float], color: tuple[float, float, float], scale: float
) -> None:
    for channel in range(3):
        target[channel] += color[channel] * scale


def radiance(direction: tuple[float, float, float]) -> tuple[float, float, float]:
    """Return a warm, directional dungeon-hall HDR signal.

    Broad window/lantern lobes survive the 0.5–0.7 roughness prefilter range,
    while their bright cores provide recognisable sharper reflections. The
    low-level stone/ceiling pattern keeps the cubemap spatially non-flat even
    away from those highlights.
    """
    x, y, z = direction
    azimuth = atan2(z, x)
    horizon = max(0.0, 1.0 - abs(y) * 1.75)
    arch_pattern = 0.72 + 0.16 * sin(azimuth * 12.0 + y * 4.0) + 0.12 * cos(y * 21.0)

    if y >= 0.0:
        base = [0.022, 0.027, 0.038]
        add_scaled(base, (0.035, 0.028, 0.020), horizon * arch_pattern)
    else:
        base = [0.018, 0.011, 0.006]
        add_scaled(base, (0.045, 0.026, 0.011), horizon * arch_pattern)
        # The downward face is mostly floor, so give it broad masonry variation
        # too; a cubemap face may be dark but must not regress into a flat color.
        floor_pattern = 0.50 + 0.25 * sin((x + z) * 28.0) + 0.25 * cos((x - z) * 23.0)
        add_scaled(base, (0.014, 0.008, 0.003), max(0.0, floor_pattern))

    # Fixed, deliberately asymmetric ceiling lanterns and warm portal light.
    lights = (
        ((0.72, 0.34, -0.61), (18.0, 10.0, 3.0)),
        ((-0.38, 0.42, -0.82), (12.0, 6.5, 2.0)),
        ((-0.86, 0.18, 0.47), (8.0, 3.5, 1.0)),
        ((0.21, 0.07, 0.97), (5.5, 7.5, 11.0)),
    )
    for source, color in lights:
        alignment = max(0.0, sum(a * b for a, b in zip(direction, normalize(source))))
        add_scaled(base, color, alignment**44 * 0.20)
        add_scaled(base, color, alignment**180)

    return tuple(max(0.0, component) for component in base)


def rgbe(color: tuple[float, float, float]) -> bytes:
    maximum = max(color)
    if maximum <= 1.0e-32:
        return b"\0\0\0\0"
    exponent = 0
    mantissa = maximum
    while mantissa >= 1.0:
        mantissa *= 0.5
        exponent += 1
    while mantissa < 0.5:
        mantissa *= 2.0
        exponent -= 1
    scale = mantissa * 256.0 / maximum
    return bytes(
        min(255, int(channel * scale)) for channel in color
    ) + bytes((exponent + 128,))


def encode_rle(channel: bytes) -> bytes:
    """Encode one RGBE scanline component using Radiance new-style RLE."""
    encoded = bytearray()
    index = 0
    size = len(channel)
    while index < size:
        run = 1
        while index + run < size and channel[index + run] == channel[index] and run < 127:
            run += 1
        if run >= 4:
            encoded.extend((128 + run, channel[index]))
            index += run
            continue

        literal = bytearray()
        while index < size and len(literal) < 128:
            run = 1
            while index + run < size and channel[index + run] == channel[index] and run < 127:
                run += 1
            if run >= 4:
                break
            literal.append(channel[index])
            index += 1
        encoded.append(len(literal))
        encoded.extend(literal)
    return bytes(encoded)


def write_face(path: Path, face: str) -> None:
    header = (
        b"#?RADIANCE\n"
        b"# Project-authored CC0 Dungeon HDR cubemap\n"
        b"FORMAT=32-bit_rle_rgbe\n\n"
        + f"-Y {FACE_SIZE} +X {FACE_SIZE}\n".encode("ascii")
    )
    encoded = bytearray(header)
    for row in range(FACE_SIZE):
        scanline = bytearray()
        v = 2.0 * (row + 0.5) / FACE_SIZE - 1.0
        for column in range(FACE_SIZE):
            u = 2.0 * (column + 0.5) / FACE_SIZE - 1.0
            scanline.extend(rgbe(radiance(cube_direction(face, u, v))))
        encoded.extend((2, 2, FACE_SIZE >> 8, FACE_SIZE & 0xFF))
        for component in range(4):
            encoded.extend(encode_rle(scanline[component::4]))
    path.write_bytes(bytes(encoded))


def write_license(path: Path) -> None:
    path.write_text(
        "CC0 1.0 Universal (CC0 1.0) Public Domain Dedication\n\n"
        "The CC0 Dungeon HDR cubemap is generated exclusively by build.py from "
        "fixed procedural equations. The project dedicates both the source and "
        "generated Radiance RGBE faces to the public domain under CC0 1.0.\n\n"
        "https://creativecommons.org/publicdomain/zero/1.0/legalcode\n",
        encoding="utf-8",
    )


def main() -> None:
    out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    out_dir.mkdir(parents=True, exist_ok=True)
    for face in FACES:
        write_face(out_dir / f"{face}.hdr", face)
    write_license(out_dir / "LICENSE")
    print(f"CC0 Dungeon HDR cubemap generated in {out_dir.resolve()}")


if __name__ == "__main__":
    main()
