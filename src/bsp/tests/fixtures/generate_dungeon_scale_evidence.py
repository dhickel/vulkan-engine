#!/usr/bin/env python3
"""
generate_dungeon_scale_evidence.py -- deterministic fixture constructor for M1/M2
dungeon map class evidence (Phase 05, BSP Dungeon Contract Evidence sprint).

This is a fixture constructor only. It is NOT a production generator, has no
importable library/API, exposes no reusable generator, and uses no random
entropy. Every output is deterministic from the fixed room/corridor definitions.

Produces exactly two checked representative Standard Quake .map sources:
  - dungeon_m1_standard.map : M1 class (~12 rooms, XY ≤ 1536, Z ≤ 256)
  - dungeon_m2_standard.map : M2 class (~28 rooms, XY ≤ 3072, Z ≤ 384)

Usage:
    python3 generate_dungeon_scale_evidence.py [--output-dir DIR]

Outputs:
    --output-dir/dungeon_m1_standard.map
    --output-dir/dungeon_m2_standard.map
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "maps"

# ── Project textures ─────────────────────────────────────────────────────
TEXTURE_WALL = "DNGN01"
TEXTURE_ACCENT = "DNGN02"

# ── Construction parameters (16-unit quantum) ────────────────────────────
QUANTUM = 16
WALL_THICKNESS = QUANTUM           # 16
CLEAR_ROUTE_WIDTH = 4 * QUANTUM    # 64
CLEAR_HEADROOM = 5 * QUANTUM       # 80

# Room exterior = interior + 2*WALL_THICKNESS on each axis
ROOM_M1_INTERIOR_W = 256
ROOM_M1_INTERIOR_D = 256
ROOM_M1_INTERIOR_H = 112  # CLEAR_HEADROOM(80) + 2*WALL_THICKNESS(32) = 112

ROOM_M2_INTERIOR_W = 224
ROOM_M2_INTERIOR_D = 224
ROOM_M2_INTERIOR_H = 112

ROOM_M1_EXTERIOR_W = ROOM_M1_INTERIOR_W + 2 * WALL_THICKNESS  # 288
ROOM_M1_EXTERIOR_D = ROOM_M1_INTERIOR_D + 2 * WALL_THICKNESS  # 288
ROOM_M1_EXTERIOR_H = ROOM_M1_INTERIOR_H + 2 * WALL_THICKNESS  # 144

ROOM_M2_EXTERIOR_W = ROOM_M2_INTERIOR_W + 2 * WALL_THICKNESS  # 256
ROOM_M2_EXTERIOR_D = ROOM_M2_INTERIOR_D + 2 * WALL_THICKNESS  # 256
ROOM_M2_EXTERIOR_H = ROOM_M2_INTERIOR_H + 2 * WALL_THICKNESS  # 144

# Corridor exterior dimensions
CORRIDOR_WIDTH = CLEAR_ROUTE_WIDTH + 2 * WALL_THICKNESS   # 96
CORRIDOR_HEIGHT = CLEAR_HEADROOM + 2 * WALL_THICKNESS     # 112

# Grid cell spacing (center-to-center) -- rooms share walls between cells
M1_CELL_SPACING = ROOM_M1_EXTERIOR_W  # 288
M2_CELL_SPACING = ROOM_M2_EXTERIOR_W + 16  # 272


# ── Brush generators ─────────────────────────────────────────────────────

def brush_floor(x1: float, y1: float, z_bottom: float, x2: float, y2: float,
                texture: str = TEXTURE_WALL) -> str:
    """Generate a floor brush at z = z_bottom, thickness=WALL_THICKNESS."""
    return _box_brush(x1, y1, z_bottom, x2, y2, z_bottom + WALL_THICKNESS, texture)


def brush_ceiling(x1: float, y1: float, z_bottom: float, x2: float, y2: float,
                  z_top: float, texture: str = TEXTURE_WALL) -> str:
    """Generate a ceiling brush at z_top - WALL_THICKNESS..z_top."""
    return _box_brush(x1, y1, z_top - WALL_THICKNESS, x2, y2, z_top, texture)


def brush_north_wall(x1: float, y1: float, z_bottom: float, x2: float, y2: float,
                     z_top: float, texture: str = TEXTURE_WALL) -> str:
    """Wall at y = y1, thickness = WALL_THICKNESS, extending northward."""
    return _box_brush(x1, y1 - WALL_THICKNESS, z_bottom, x2, y1, z_top, texture)


def brush_south_wall(x1: float, y1: float, z_bottom: float, x2: float, y2: float,
                     z_top: float, texture: str = TEXTURE_WALL) -> str:
    """Wall at y = y2, thickness = WALL_THICKNESS, extending southward."""
    return _box_brush(x1, y2, z_bottom, x2, y2 + WALL_THICKNESS, z_top, texture)


def brush_east_wall(x1: float, y1: float, z_bottom: float, x2: float, y2: float,
                    z_top: float, texture: str = TEXTURE_WALL) -> str:
    """Wall at x = x2, thickness = WALL_THICKNESS, extending eastward."""
    return _box_brush(x2, y1, z_bottom, x2 + WALL_THICKNESS, y2, z_top, texture)


def brush_west_wall(x1: float, y1: float, z_bottom: float, x2: float, y2: float,
                    z_top: float, texture: str = TEXTURE_WALL) -> str:
    """Wall at x = x1, thickness = WALL_THICKNESS, extending westward."""
    return _box_brush(x1 - WALL_THICKNESS, y1, z_bottom, x1, y2, z_top, texture)


def _box_brush(x1: float, y1: float, z1: float, x2: float, y2: float, z2: float,
               texture: str) -> str:
    """Generate an axis-aligned solid box brush with 6 planes.

    Plane winding: each face lists 3 non-colinear points whose cross product
    points INTO the brush interior (matching ericw-tools Standard .map convention).
    Texture params: 0 0 0 1.0 1.0 (aligned, identity scale).
    """
    s = f"// brush: box ({x1},{y1},{z1}) to ({x2},{y2},{z2})\n"
    s += "{\n"
    # West face: x=x1, normal +X, inside to the east
    s += f"( {x1} {y2} {z2} ) ( {x1} {y1} {z2} ) ( {x1} {y1} {z1} ) {texture} 0 0 0 1.0 1.0\n"
    # East face: x=x2, normal -X, inside to the west
    s += f"( {x2} {y1} {z2} ) ( {x2} {y2} {z2} ) ( {x2} {y2} {z1} ) {texture} 0 0 0 1.0 1.0\n"
    # South face: y=y1, normal +Y, inside to the north
    s += f"( {x1} {y1} {z1} ) ( {x1} {y1} {z2} ) ( {x2} {y1} {z2} ) {texture} 0 0 0 1.0 1.0\n"
    # North face: y=y2, normal -Y, inside to the south
    s += f"( {x2} {y2} {z2} ) ( {x1} {y2} {z2} ) ( {x1} {y2} {z1} ) {texture} 0 0 0 1.0 1.0\n"
    # Bottom face: z=z1, normal +Z, inside above (matches dungeon_evidence_standard.map)
    s += f"( {x1} {y2} {z1} ) ( {x1} {y1} {z1} ) ( {x2} {y1} {z1} ) {texture} 0 0 0 1.0 1.0\n"
    # Top face: z=z2, normal -Z, inside below (matches dungeon_evidence_standard.map)
    s += f"( {x1} {y2} {z2} ) ( {x2} {y2} {z2} ) ( {x2} {y1} {z2} ) {texture} 0 0 0 1.0 1.0\n"
    s += "}\n"
    return s


def room_brushes(rx: float, ry: float, z_base: float,
                 w_exterior: float, d_exterior: float, h_exterior: float,
                 wall_tex: str = TEXTURE_WALL,
                 accent_tex: str = TEXTURE_ACCENT,
                 opened_sides: set | None = None) -> list[str]:
    """Generate the 6 brushes for a room, optionally omitting walls on opened sides.

    opened_sides: set of 'north', 'south', 'east', 'west' to omit.
    """
    if opened_sides is None:
        opened_sides = set()

    x1 = rx
    y1 = ry
    x2 = rx + w_exterior
    y2 = ry + d_exterior
    z1 = z_base
    z2 = z_base + h_exterior

    brushes = []

    # Floor -- always present
    brushes.append(brush_floor(x1, y1, z1, x2, y2, wall_tex))

    # Ceiling -- always present
    brushes.append(brush_ceiling(x1, y1, z1, x2, y2, z2, wall_tex))

    # Walls -- omit if on an opened side
    if 'north' not in opened_sides:
        brushes.append(brush_north_wall(x1, y1, z1 + WALL_THICKNESS, x2, y2,
                                        z2 - WALL_THICKNESS, wall_tex))
    if 'south' not in opened_sides:
        brushes.append(brush_south_wall(x1, y1, z1 + WALL_THICKNESS, x2, y2,
                                        z2 - WALL_THICKNESS, wall_tex))
    if 'east' not in opened_sides:
        brushes.append(brush_east_wall(x1, y1, z1 + WALL_THICKNESS, x2, y2,
                                       z2 - WALL_THICKNESS, wall_tex))
    if 'west' not in opened_sides:
        brushes.append(brush_west_wall(x1, y1, z1 + WALL_THICKNESS, x2, y2,
                                       z2 - WALL_THICKNESS, wall_tex))

    return brushes


def corridor_brushes(cx: float, cy: float, cz: float,
                     length: float, orientation: str,
                     width: float = CORRIDOR_WIDTH,
                     height: float = CORRIDOR_HEIGHT,
                     texture: str = TEXTURE_WALL) -> list[str]:
    """Generate the 4 brushes for a corridor (floor, ceiling, 2 side walls).

    orientation: 'horizontal' (runs along X) or 'vertical' (runs along Y).
    The corridor extends from (cx, cy) by `length` in the given direction.
    """
    brushes = []
    z1 = cz
    z2 = cz + height

    if orientation == 'horizontal':
        # Corridor runs east-west; width along Y (north-south)
        x1 = cx
        x2 = cx + length
        y1 = cy
        y2 = cy + width

        # Floor
        brushes.append(brush_floor(x1, y1, z1, x2, y2, texture))
        # Ceiling
        brushes.append(brush_ceiling(x1, y1, z1, x2, y2, z2, texture))
        # North wall (at y2)
        brushes.append(brush_north_wall(x1, y2, z1 + WALL_THICKNESS, x2, y2,
                                        z2 - WALL_THICKNESS, texture))
        # South wall (at y1)
        brushes.append(brush_south_wall(x1, y1, z1 + WALL_THICKNESS, x2, y1,
                                        z2 - WALL_THICKNESS, texture))
    else:
        # Corridor runs north-south; width along X (east-west)
        x1 = cx
        x2 = cx + width
        y1 = cy
        y2 = cy + length

        # Floor
        brushes.append(brush_floor(x1, y1, z1, x2, y2, texture))
        # Ceiling
        brushes.append(brush_ceiling(x1, y1, z1, x2, y2, z2, texture))
        # East wall (at x2)
        brushes.append(brush_east_wall(x1, y1, z1 + WALL_THICKNESS, x2, y2,
                                       z2 - WALL_THICKNESS, texture))
        # West wall (at x1)
        brushes.append(brush_west_wall(x1, y1, z1 + WALL_THICKNESS, x2, y2,
                                       z2 - WALL_THICKNESS, texture))

    return brushes


# ── Map assembly ──────────────────────────────────────────────────────────

# Grid adjacency: (gx, gy) -> room index (or None if no room there)
# Connections: set of ((gx1,gy1), (gx2,gy2)) tuples for corridors


def generate_m1_map() -> str:
    """Generate M1 dungeon map: 4×4 grid, 16 rooms, 2 loops.

    Room exterior: 288×288×144, interior 256×256×112 (80 clear headroom).
    Grid cell spacing: 288 (rooms share walls).
    Outer XY: ~1360×1360 (< 1536). Z: 176 (< 256).

    Layout (gx,gy): 4 cols × 4 rows, rooms at every cell (16 total).
    Connections: spanning tree + 2 cross edges → 2 loops.
    """
    cols, rows = 4, 4
    cell_spacing = M1_CELL_SPACING
    room_w = ROOM_M1_EXTERIOR_W
    room_d = ROOM_M1_EXTERIOR_D
    room_h = ROOM_M1_EXTERIOR_H
    z_base = 0

    # Map outer hull: padding beyond furthest rooms.
    # M1 Z span <= 256: room_h=144, so hull_margin=48 gives span of 240.
    hull_margin = 48
    outer_x = (cols - 1) * cell_spacing + room_w + hull_margin
    outer_y = (rows - 1) * cell_spacing + room_d + hull_margin
    outer_z = z_base + room_h + hull_margin

    # Every cell is a room
    rooms = [(gx, gy) for gy in range(rows) for gx in range(cols)]

    # Adjacency graph: connections between adjacent cells
    # Spanning tree -- snake through all cells:
    # Row 0: (0,0)-(1,0)-(2,0)-(3,0)
    # Down to (3,1), then snake back: (2,1)-(1,1)-(0,1)
    # Down to (0,2), then forward: (1,2)-(2,2)-(3,2)
    # Down to (3,3), then back: (2,3)-(1,3)-(0,3)
    connections = set()
    # Horizontal within rows
    for gy in range(rows):
        for gx in range(cols - 1):
            if gy % 2 == 0:
                connections.add(((gx, gy), (gx + 1, gy)))
            else:
                connections.add(((cols - 2 - gx, gy), (cols - 1 - gx, gy)))
    # Vertical between rows
    for gy in range(rows - 1):
        if gy % 2 == 0:
            # connect last cell of row gy to last cell of row gy+1
            connections.add(((cols - 1, gy), (cols - 1, gy + 1)))
        else:
            # connect first cell of row gy to first cell of row gy+1
            connections.add(((0, gy), (0, gy + 1)))

    # Add 2 cross edges for loops: (1,0)-(1,1) and (2,1)-(2,2)
    connections.add(((1, 0), (1, 1)))
    connections.add(((2, 1), (2, 2)))

    # Determine which sides of each room are opened
    room_open: dict[tuple[int, int], set[str]] = {
        room: set() for room in rooms
    }

    for (a, b) in connections:
        gx_a, gy_a = a
        gx_b, gy_b = b
        if gx_b == gx_a + 1:  # a open east, b open west
            room_open.setdefault(a, set()).add('east')
            room_open.setdefault(b, set()).add('west')
        elif gx_a == gx_b + 1:  # b open east, a open west
            room_open.setdefault(a, set()).add('west')
            room_open.setdefault(b, set()).add('east')
        elif gy_b == gy_a + 1:  # a open south, b open north
            room_open.setdefault(a, set()).add('south')
            room_open.setdefault(b, set()).add('north')
        elif gy_a == gy_b + 1:  # b open south, a open north
            room_open.setdefault(a, set()).add('north')
            room_open.setdefault(b, set()).add('south')

    # Generate brushes
    all_brushes = []

    # Outer hull brushes (seal the map)
    # Hull is at -hull_margin to outer_x, etc.
    hx1 = -hull_margin
    hy1 = -hull_margin
    hz1 = -hull_margin
    hx2 = outer_x
    hy2 = outer_y
    hz2 = outer_z
    # Floor (huge)
    all_brushes.append(brush_floor(hx1, hy1, hz1, hx2, hy2))
    # Ceiling (huge)
    all_brushes.append(brush_ceiling(hx1, hy1, hz1, hx2, hy2, hz2))
    # North wall
    all_brushes.append(brush_north_wall(hx1, hy2, hz1 + WALL_THICKNESS, hx2, hy2,
                                        hz2 - WALL_THICKNESS))
    # South wall
    all_brushes.append(brush_south_wall(hx1, hy1, hz1 + WALL_THICKNESS, hx2, hy1,
                                        hz2 - WALL_THICKNESS))
    # East wall
    all_brushes.append(brush_east_wall(hx2, hy1, hz1 + WALL_THICKNESS, hx2, hy2,
                                       hz2 - WALL_THICKNESS))
    # West wall
    all_brushes.append(brush_west_wall(hx1, hy1, hz1 + WALL_THICKNESS, hx1, hy2,
                                       hz2 - WALL_THICKNESS))

    # Room brushes
    for (gx, gy) in rooms:
        rx = gx * cell_spacing + hull_margin
        ry = gy * cell_spacing + hull_margin
        opened = room_open.get((gx, gy), set())
        all_brushes.extend(room_brushes(rx, ry, z_base, room_w, room_d, room_h,
                                        opened_sides=opened))

    # Lights: one colored light per room, placed at room center
    lights = []
    for (gx, gy) in rooms:
        rx = gx * cell_spacing + hull_margin
        ry = gy * cell_spacing + hull_margin
        cx = rx + room_w / 2.0
        cy = ry + room_d / 2.0
        cz = z_base + room_h - WALL_THICKNESS - QUANTUM  # near ceiling
        # Vary light color slightly by position
        hue = ((gx * 3 + gy * 7) % 24) / 24.0
        r = 0.7 + 0.3 * abs(hue - 0.5)
        g_val = 0.6 + 0.3 * abs((hue + 0.33) % 1.0 - 0.5)
        b = 0.5 + 0.3 * abs((hue + 0.67) % 1.0 - 0.5)
        lights.append((cx, cy, cz, r, g_val, b))

    # Spawn: first room center
    spawn_rx = 0 * cell_spacing + hull_margin
    spawn_ry = 0 * cell_spacing + hull_margin
    spawn_x = spawn_rx + room_w / 2.0
    spawn_y = spawn_ry + room_d / 2.0
    spawn_z = z_base + WALL_THICKNESS + QUANTUM

    return _assemble_map("dungeon_m1_standard", all_brushes, lights,
                         spawn_x, spawn_y, spawn_z, 90)


def generate_m2_map() -> str:
    """Generate M2 dungeon map: 4×7 grid, 28 rooms, 4 loops.

    Room exterior: 256×256×144, interior 224×224×112 (80 clear headroom).
    Grid cell spacing: 272.
    Outer XY: ~1360×2176 (< 3072×3072). Z: 176 (< 384).

    Layout (gx,gy): 4 cols × 7 rows, rooms at every cell (28 total).
    Connections: spanning tree + 4 cross edges → 4 loops.
    """
    cols, rows = 4, 7
    cell_spacing = M2_CELL_SPACING  # 272
    room_w = ROOM_M2_EXTERIOR_W  # 256
    room_d = ROOM_M2_EXTERIOR_D  # 256
    room_h = ROOM_M2_EXTERIOR_H  # 144
    z_base = 0

    hull_margin = 48
    outer_x = (cols - 1) * cell_spacing + room_w + hull_margin
    outer_y = (rows - 1) * cell_spacing + room_d + hull_margin
    outer_z = z_base + room_h + hull_margin

    # All cells are rooms
    rooms = [(gx, gy) for gy in range(rows) for gx in range(cols)]

    # Spanning tree: snake pattern through 4x7 grid
    connections: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    # Horizontal within rows (snake pattern)
    for gy in range(rows):
        for gx in range(cols - 1):
            if gy % 2 == 0:
                connections.add(((gx, gy), (gx + 1, gy)))
            else:
                connections.add(((cols - 2 - gx, gy), (cols - 1 - gx, gy)))
    # Vertical between rows
    for gy in range(rows - 1):
        if gy % 2 == 0:
            connections.add(((cols - 1, gy), (cols - 1, gy + 1)))
        else:
            connections.add(((0, gy), (0, gy + 1)))

    # Add 4 cross edges for loops
    # (1,0)-(1,1), (2,1)-(2,2), (1,2)-(1,3), (2,3)-(2,4),
    # (1,4)-(1,5), (2,5)-(2,6) -- these create 4 independent cycles
    cross_edges = [
        ((1, 0), (1, 1)),
        ((2, 1), (2, 2)),
        ((1, 3), (1, 4)),
        ((2, 5), (2, 6)),
    ]
    for ce in cross_edges:
        connections.add(ce)

    # Determine open sides
    room_open: dict[tuple[int, int], set[str]] = {}
    for room in rooms:
        room_open[room] = set()
    for (a, b) in connections:
        gx_a, gy_a = a
        gx_b, gy_b = b
        if gx_b == gx_a + 1:
            room_open.setdefault(a, set()).add('east')
            room_open.setdefault(b, set()).add('west')
        elif gx_a == gx_b + 1:
            room_open.setdefault(a, set()).add('west')
            room_open.setdefault(b, set()).add('east')
        elif gy_b == gy_a + 1:
            room_open.setdefault(a, set()).add('south')
            room_open.setdefault(b, set()).add('north')
        elif gy_a == gy_b + 1:
            room_open.setdefault(a, set()).add('north')
            room_open.setdefault(b, set()).add('south')

    # Generate brushes
    all_brushes = []

    # Outer hull
    hx1 = -hull_margin
    hy1 = -hull_margin
    hz1 = -hull_margin
    hx2 = outer_x
    hy2 = outer_y
    hz2 = outer_z
    all_brushes.append(brush_floor(hx1, hy1, hz1, hx2, hy2))
    all_brushes.append(brush_ceiling(hx1, hy1, hz1, hx2, hy2, hz2))
    all_brushes.append(brush_north_wall(hx1, hy2, hz1 + WALL_THICKNESS, hx2, hy2,
                                        hz2 - WALL_THICKNESS))
    all_brushes.append(brush_south_wall(hx1, hy1, hz1 + WALL_THICKNESS, hx2, hy1,
                                        hz2 - WALL_THICKNESS))
    all_brushes.append(brush_east_wall(hx2, hy1, hz1 + WALL_THICKNESS, hx2, hy2,
                                       hz2 - WALL_THICKNESS))
    all_brushes.append(brush_west_wall(hx1, hy1, hz1 + WALL_THICKNESS, hx1, hy2,
                                       hz2 - WALL_THICKNESS))

    # Room brushes
    for (gx, gy) in rooms:
        rx = gx * cell_spacing + hull_margin
        ry = gy * cell_spacing + hull_margin
        opened = room_open.get((gx, gy), set())
        all_brushes.extend(room_brushes(rx, ry, z_base, room_w, room_d, room_h,
                                        opened_sides=opened))

    # Lights
    lights = []
    for (gx, gy) in rooms:
        rx = gx * cell_spacing + hull_margin
        ry = gy * cell_spacing + hull_margin
        cx = rx + room_w / 2.0
        cy = ry + room_d / 2.0
        cz = z_base + room_h - WALL_THICKNESS - QUANTUM
        hue = ((gx * 5 + gy * 11) % 24) / 24.0
        r = 0.7 + 0.3 * abs(hue - 0.5)
        g_val = 0.6 + 0.3 * abs((hue + 0.33) % 1.0 - 0.5)
        b = 0.5 + 0.3 * abs((hue + 0.67) % 1.0 - 0.5)
        lights.append((cx, cy, cz, r, g_val, b))

    # Spawn
    spawn_rx = 0 * cell_spacing + hull_margin
    spawn_ry = 0 * cell_spacing + hull_margin
    spawn_x = spawn_rx + room_w / 2.0
    spawn_y = spawn_ry + room_d / 2.0
    spawn_z = z_base + WALL_THICKNESS + QUANTUM

    return _assemble_map("dungeon_m2_standard", all_brushes, lights,
                         spawn_x, spawn_y, spawn_z, 90)


def _assemble_map(name: str, brushes: list[str],
                  lights: list[tuple[float, float, float, float, float, float]],
                  spawn_x: float, spawn_y: float, spawn_z: float,
                  spawn_angle: float) -> str:
    """Assemble a complete Standard Quake .map file from brushes and entities."""
    lines = []
    lines.append("// Game: Quake")
    lines.append("// Format: Standard")
    lines.append(f"// Project-authored {name} -- Phase 05 map class evidence fixture")
    lines.append("// entity 0")
    lines.append("{")
    lines.append('"classname" "worldspawn"')
    lines.append('"wad" "dungeon_evidence.wad"')

    for brush_str in brushes:
        lines.append(brush_str.rstrip())

    lines.append("}")

    # Light entities
    entity_idx = 1
    for (cx, cy, cz, r, g, b) in lights:
        lines.append(f"// entity {entity_idx}: colored light")
        lines.append("{")
        lines.append('"classname" "light"')
        lines.append(f'"origin" "{cx:.1f} {cy:.1f} {cz:.1f}"')
        lines.append('"light" "300"')
        lines.append(f'"_color" "{r:.3f} {g:.3f} {b:.3f}"')
        lines.append('"style" "0"')
        lines.append("}")
        entity_idx += 1

    # Player spawn
    lines.append(f"// entity {entity_idx}: info_player_start")
    lines.append("{")
    lines.append('"classname" "info_player_start"')
    lines.append(f'"origin" "{spawn_x:.1f} {spawn_y:.1f} {spawn_z:.1f}"')
    lines.append(f'"angle" "{spawn_angle:.1f}"')
    lines.append("}")

    return "\n".join(lines) + "\n"


# ── CLI ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate M1/M2 dungeon scale evidence .map fixtures"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for .map files (default: maps/)",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    maps = [
        ("dungeon_m1_standard.map", generate_m1_map()),
        ("dungeon_m2_standard.map", generate_m2_map()),
    ]

    for filename, content in maps:
        path = args.output_dir / filename
        path.write_text(content, encoding="ascii")
        print(f"  wrote {path} ({len(content)} chars)")

    print(f"\nDone: {len(maps)} scale evidence .map source(s) generated")


if __name__ == "__main__":
    main()
