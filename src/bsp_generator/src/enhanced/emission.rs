//! Enhanced v2 emission — produce canonical .map text from placement, topology,
//! theme, and feature results.
//!
//! Every room shell has proper wall apertures for corridors and stairs.
//! Stairwells are sealed shells connecting both room layers. Textures come
//! from theme assignment records. Entities use Phase-06 spawn/light origins.

use std::collections::{BTreeMap, BTreeSet};

use crate::config::CONSTRUCTION_QUANTUM;
use crate::junction;

use super::config::EnhancedConfig;
use super::error::EnhancedError;
use super::features::FeatureResult;
use super::intent::RoomId;
use super::placement::{CandidateSocket, PlacedRoom, PlacementResult, WallDirection};
use super::theme::{cc0_dungeon_v2_theme, TextureRole, ThemeAssignment};
use super::topology::TopologyResult;

const Q: i32 = CONSTRUCTION_QUANTUM as i32;
const CORRIDOR_WIDTH: i32 = 64;
const CORRIDOR_HEIGHT: i32 = 80;

// ── Opening descriptor ─────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct Opening {
    /// Minimum coordinate along the wall tangent axis.
    tangent_min: i32,
    /// Maximum coordinate along the wall tangent axis.
    tangent_max: i32,
    /// Bottom Z of the opening.
    bottom: i32,
    /// Top Z of the opening.
    top: i32,
}

// ── Cell-based helpers ─────────────────────────────────────────────────────

type Cell = (i32, i32); // (cell_x, cell_y)

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct CorridorCell {
    floor_z: i32,
    ceiling_bottom: i32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct WallSpan {
    floor_z: i32,
    shell_top: i32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct GridRect<T> {
    x0: i32,
    x1: i32,
    y0: i32,
    y1: i32,
    value: T,
}

// ── Public API ─────────────────────────────────────────────────────────────

/// Emit a complete Enhanced v2 .map file from the full pipeline results.
pub fn emit_map(
    _config: &EnhancedConfig,
    placement: &PlacementResult,
    topology: &TopologyResult,
    theme: &ThemeAssignment,
    features: &FeatureResult,
) -> Result<String, EnhancedError> {
    let wad = &theme.wad_basename;
    let room_map: BTreeMap<RoomId, &PlacedRoom> =
        placement.rooms.iter().map(|r| (r.id, r)).collect();
    let socket_map: BTreeMap<super::intent::SocketId, &CandidateSocket> =
        placement.sockets.iter().map(|s| (s.id, s)).collect();

    // Build room-owned cell set for corridor slab culling
    let room_owned_cells = build_room_owned_cells(&placement.rooms);

    // Build corridor cell map from route envelopes
    let corridor_cells = build_corridor_cells(topology, &room_map);

    // ── Emit ──────────────────────────────────────────────────────────
    let mut out = String::new();
    out.push_str("{\n\"classname\" \"worldspawn\"\n");
    out.push_str(&format!("\"wad\" \"{wad}\"\n"));

    // Emit room brushes
    for room in &placement.rooms {
        emit_room_brushes(
            &mut out,
            room,
            topology,
            &room_map,
            &socket_map,
            &corridor_cells,
            theme,
        )?;
    }

    // Emit corridor geometry (only in non-room cells)
    emit_corridor_brushes(&mut out, topology, &room_map, &room_owned_cells, theme);

    // Emit stair transitions
    emit_stair_transitions(&mut out, topology, &room_map, &socket_map, theme);

    out.push_str("}\n");

    // ── Entities ──────────────────────────────────────────────────────
    // info_player_start
    let spawn = &features.spawn_point;
    out.push_str(&format!(
        "{{\n\"classname\" \"info_player_start\"\n\"origin\" \"{} {} {}\"\n}}\n",
        spawn.origin.0, spawn.origin.1, spawn.origin.2
    ));

    // light entities
    for light in &features.light_origins {
        out.push_str(&format!(
            "{{\n\"classname\" \"light\"\n\"light\" \"300\"\n\"origin\" \"{} {} {}\"\n}}\n",
            light.origin.0, light.origin.1, light.origin.2
        ));
    }

    Ok(out)
}

// ── Room emission ──────────────────────────────────────────────────────────

fn emit_room_brushes(
    out: &mut String,
    room: &PlacedRoom,
    topology: &TopologyResult,
    _room_map: &BTreeMap<RoomId, &PlacedRoom>,
    socket_map: &BTreeMap<super::intent::SocketId, &CandidateSocket>,
    corridor_cells: &BTreeMap<Cell, CorridorCell>,
    theme: &ThemeAssignment,
) -> Result<(), EnhancedError> {
    let (x0, y0, x1, y1) = room.shell;
    let z0 = room.floor_z;
    let zh = room.dims.2 as i32;
    let z1 = z0 + zh;

    // Texture names from theme assignment
    let palette_name = theme
        .room_palettes
        .get(&room.id)
        .map(|pa| pa.palette_name.as_str())
        .unwrap_or("base_stone");
    let floor_tex = texture_for(palette_name, TextureRole::Floor);
    let wall_tex = texture_for(palette_name, TextureRole::Wall);
    let ceiling_tex = texture_for(palette_name, TextureRole::Ceiling);

    // Floor
    emit_solid_brush(out, x0, y0, z0, x1, y1, z0 + Q, floor_tex);
    // Ceiling
    emit_solid_brush(out, x0, y0, z1 - Q, x1, y1, z1, ceiling_tex);

    // Walls with apertures — openings collected then walls emitted with cutouts

    // Collect openings for each wall
    let openings = collect_room_openings(room, topology, socket_map);

    // Emit wall pieces for each direction
    for wall_dir in &[
        WallDirection::North,
        WallDirection::South,
        WallDirection::East,
        WallDirection::West,
    ] {
        let wall_openings = openings.get(wall_dir).map(|v| v.as_slice()).unwrap_or(&[]);
        emit_split_wall(
            out,
            room,
            *wall_dir,
            wall_openings,
            corridor_cells,
            wall_tex,
        );
    }

    Ok(())
}

fn collect_room_openings(
    room: &PlacedRoom,
    topology: &TopologyResult,
    socket_map: &BTreeMap<super::intent::SocketId, &CandidateSocket>,
) -> BTreeMap<WallDirection, Vec<Opening>> {
    let mut openings: BTreeMap<WallDirection, Vec<Opening>> = BTreeMap::new();
    let z0 = room.floor_z + Q;
    let zh = room.dims.2 as i32;
    let z1 = z0 + zh - Q;

    let mut add_opening = |wall: WallDirection, anchor: (i32, i32, i32)| {
        let half = CORRIDOR_WIDTH / 2;
        let op = match wall {
            WallDirection::North | WallDirection::South => Opening {
                tangent_min: (anchor.0 - half).max(room.shell.0 + Q),
                tangent_max: (anchor.0 + half).min(room.shell.2 - Q),
                bottom: z0,
                top: (z0 + CORRIDOR_HEIGHT).min(z1),
            },
            WallDirection::East | WallDirection::West => Opening {
                tangent_min: (anchor.1 - half).max(room.shell.1 + Q),
                tangent_max: (anchor.1 + half).min(room.shell.3 - Q),
                bottom: z0,
                top: (z0 + CORRIDOR_HEIGHT).min(z1),
            },
        };
        if op.tangent_min < op.tangent_max && op.bottom < op.top {
            openings.entry(wall).or_default().push(op);
        }
    };

    // Corridor route apertures
    for route in &topology.routes {
        for &(socket_id, is_source) in &[(route.source_socket, true), (route.target_socket, false)]
        {
            if let Some(socket) = socket_map.get(&socket_id) {
                let expected_room = if is_source {
                    route.source_room
                } else {
                    route.target_room
                };
                if socket.room == room.id && socket.room == expected_room {
                    add_opening(socket.wall, socket.anchor);
                }
            }
        }
    }

    // Transition (stair) apertures
    for t in &topology.transitions {
        for &(socket_id, is_lower) in &[(t.lower_socket, true), (t.upper_socket, false)] {
            if let Some(socket) = socket_map.get(&socket_id) {
                let expected_room = if is_lower { t.lower_room } else { t.upper_room };
                if socket.room == room.id && socket.room == expected_room {
                    add_opening(socket.wall, socket.anchor);
                }
            }
        }
    }

    // Deduplicate and sort
    for ops in openings.values_mut() {
        ops.sort_unstable_by_key(|o| (o.tangent_min, o.tangent_max, o.bottom, o.top));
        ops.dedup();
    }
    openings
}

fn emit_split_wall(
    out: &mut String,
    room: &PlacedRoom,
    wall: WallDirection,
    openings: &[Opening],
    corridor_cells: &BTreeMap<Cell, CorridorCell>,
    texture: &str,
) {
    let (x0, y0, x1, y1) = room.shell;
    let z0 = room.floor_z;
    let zh = room.dims.2 as i32;
    let wall_z0 = z0 + Q;
    let wall_z1 = z0 + zh - Q;

    let (tangent_min, tangent_max, _wall_x0, _wall_x1, _wall_y0, _wall_y1) = match wall {
        WallDirection::North => (x0, x1, x0, x1 - Q, y1 - Q, y1),
        WallDirection::South => (x0, x1, x0, x1 - Q, y0, y0 + Q),
        WallDirection::East => (y0, y1, x1 - Q, x1, y0 + Q, y1 - Q),
        WallDirection::West => (y0, y1, x0, x0 + Q, y0 + Q, y1 - Q),
    };

    // Build solid cell mask
    let mut solid_cells: BTreeMap<Cell, ()> = BTreeMap::new();
    for t_cell in tangent_min.div_euclid(Q)..tangent_max.div_euclid(Q) {
        for z_cell in wall_z0.div_euclid(Q)..wall_z1.div_euclid(Q) {
            solid_cells.insert((t_cell, z_cell), ());
        }
    }

    // Remove opening cells
    for op in openings {
        for z_cell in op.bottom.div_euclid(Q)..op.top.div_euclid(Q) {
            for t_cell in op.tangent_min.div_euclid(Q)..op.tangent_max.div_euclid(Q) {
                solid_cells.remove(&(t_cell, z_cell));
            }
        }
    }

    // Remove cells that corridor wall cells also claim (for turning chambers)
    let opening_bottom = z0 + Q;
    let opening_limit = room.floor_z + room.dims.2 as i32 - Q;
    for t_cell in tangent_min.div_euclid(Q)..tangent_max.div_euclid(Q) {
        let wall_cell = match wall {
            WallDirection::West => (x0.div_euclid(Q), t_cell),
            WallDirection::East => (x1.div_euclid(Q) - 1, t_cell),
            WallDirection::South => (t_cell, y0.div_euclid(Q)),
            WallDirection::North => (t_cell, y1.div_euclid(Q) - 1),
        };
        if let Some(cc) = corridor_cells.get(&wall_cell) {
            let op_top = cc.ceiling_bottom.min(opening_limit);
            for z_cell in opening_bottom.div_euclid(Q)..op_top.div_euclid(Q) {
                solid_cells.remove(&(t_cell, z_cell));
            }
        }
    }

    // Merge solid cells into rectangles
    for rect in merge_cells_void(&solid_cells) {
        let t0 = rect.x0 * Q;
        let t1 = rect.x1 * Q;
        let bz0 = rect.y0 * Q;
        let bz1 = rect.y1 * Q;

        let (bx0, by0, bx1, by1) = match wall {
            WallDirection::North => (t0, y1 - Q, t1, y1),
            WallDirection::South => (t0, y0, t1, y0 + Q),
            WallDirection::East => (x1 - Q, t0, x1, t1),
            WallDirection::West => (x0, t0, x0 + Q, t1),
        };

        emit_solid_brush(out, bx0, by0, bz0, bx1, by1, bz1, texture);
    }

    // Emit corner columns for E/W walls (the corners are covered by N/S walls)
    // No-op: corners are already handled by N/S wall spans extending full width
}

// ── Corridor emission ──────────────────────────────────────────────────────

fn build_room_owned_cells(rooms: &[PlacedRoom]) -> BTreeSet<Cell> {
    let mut cells = BTreeSet::new();
    for room in rooms {
        let (x0, y0, x1, y1) = room.shell;
        for gy in y0.div_euclid(Q)..y1.div_euclid(Q) {
            for gx in x0.div_euclid(Q)..x1.div_euclid(Q) {
                cells.insert((gx, gy));
            }
        }
    }
    cells
}

fn build_corridor_cells(
    topology: &TopologyResult,
    _room_map: &BTreeMap<RoomId, &PlacedRoom>,
) -> BTreeMap<Cell, CorridorCell> {
    let mut cells = BTreeMap::new();

    for route in &topology.routes {
        // Determine floor_z from the route's headroom
        let floor_z = route.headroom.0 - Q;
        let ceiling_bottom = floor_z + Q + CORRIDOR_HEIGHT;

        let span = CorridorCell {
            floor_z,
            ceiling_bottom,
        };

        for &(ex0, ey0, ex1, ey1) in &route.envelopes {
            let qx0 = ex0.div_euclid(Q);
            let qy0 = ey0.div_euclid(Q);
            let qx1 = ex1.div_euclid(Q);
            let qy1 = ey1.div_euclid(Q);

            for gy in qy0..qy1 {
                for gx in qx0..qx1 {
                    cells.insert((gx, gy), span);
                }
            }
        }
    }

    cells
}

fn emit_corridor_brushes(
    out: &mut String,
    topology: &TopologyResult,
    room_map: &BTreeMap<RoomId, &PlacedRoom>,
    room_owned_cells: &BTreeSet<Cell>,
    _theme: &ThemeAssignment,
) {
    let corridor_cells = build_corridor_cells(topology, room_map);

    // Only emit corridor geometry in cells NOT owned by rooms
    let shell_cells: BTreeMap<Cell, CorridorCell> = corridor_cells
        .iter()
        .filter(|(cell, _)| !room_owned_cells.contains(cell))
        .map(|(&c, &s)| (c, s))
        .collect();

    if shell_cells.is_empty() {
        return;
    }

    // Get the connector palette for corridor textures
    let conn_floor = "conn_floor";
    let conn_wall = "conn_wall";
    let conn_ceil = "conn_ceil";

    // Floor brushes
    let floor_cells: BTreeMap<Cell, i32> = shell_cells
        .iter()
        .map(|(&cell, span)| (cell, span.floor_z))
        .collect();
    for rect in merge_cells(&floor_cells) {
        emit_solid_brush(
            out,
            rect.x0 * Q,
            rect.y0 * Q,
            rect.value,
            rect.x1 * Q,
            rect.y1 * Q,
            rect.value + Q,
            conn_floor,
        );
    }

    // Ceiling brushes
    let ceil_z: BTreeMap<Cell, i32> = shell_cells
        .iter()
        .map(|(&cell, span)| (cell, span.ceiling_bottom))
        .collect();
    for rect in merge_cells(&ceil_z) {
        emit_solid_brush(
            out,
            rect.x0 * Q,
            rect.y0 * Q,
            rect.value,
            rect.x1 * Q,
            rect.y1 * Q,
            rect.value + Q,
            conn_ceil,
        );
    }

    // Boundary walls
    let mut wall_cells: BTreeMap<Cell, WallSpan> = BTreeMap::new();
    for (&cell, &cc) in &shell_cells {
        let span = WallSpan {
            floor_z: cc.floor_z,
            shell_top: cc.ceiling_bottom + Q,
        };
        for neighbor in [
            (cell.0 - 1, cell.1),
            (cell.0 + 1, cell.1),
            (cell.0, cell.1 - 1),
            (cell.0, cell.1 + 1),
        ] {
            if shell_cells.contains_key(&neighbor) || room_owned_cells.contains(&neighbor) {
                continue;
            }
            wall_cells
                .entry(neighbor)
                .and_modify(|existing| {
                    existing.shell_top = existing.shell_top.max(span.shell_top);
                })
                .or_insert(span);
        }
    }

    for rect in merge_cells(&wall_cells) {
        emit_solid_brush(
            out,
            rect.x0 * Q,
            rect.y0 * Q,
            rect.value.floor_z,
            rect.x1 * Q,
            rect.y1 * Q,
            rect.value.shell_top,
            conn_wall,
        );
    }
}

// ── Stair transition emission ──────────────────────────────────────────────

fn emit_stair_transitions(
    out: &mut String,
    topology: &TopologyResult,
    room_map: &BTreeMap<RoomId, &PlacedRoom>,
    socket_map: &BTreeMap<super::intent::SocketId, &CandidateSocket>,
    _theme: &ThemeAssignment,
) {
    let conn_floor = "conn_floor";
    let conn_wall = "conn_wall";
    let conn_ceil = "conn_ceil";

    for t in &topology.transitions {
        let Some(lower_room) = room_map.get(&t.lower_room) else {
            continue;
        };
        let Some(upper_room) = room_map.get(&t.upper_room) else {
            continue;
        };

        let (fx0, fy0, fx1, fy1) = t.footprint;
        let lower_z = lower_room.floor_z;
        let upper_top = upper_room.floor_z + upper_room.dims.2 as i32;

        // Stairwell floor at lower room level
        emit_solid_brush(out, fx0, fy0, lower_z, fx1, fy1, lower_z + Q, conn_floor);

        // Stairwell ceiling at upper room ceiling level
        emit_solid_brush(out, fx0, fy0, upper_top - Q, fx1, fy1, upper_top, conn_ceil);

        // Stairwell walls (with apertures at socket positions)
        let wall_z0 = lower_z + Q;
        let wall_z1 = upper_top - Q;

        // Collect stairwell wall openings from both sockets
        let mut north_ops: Vec<Opening> = Vec::new();
        let mut south_ops: Vec<Opening> = Vec::new();
        let mut east_ops: Vec<Opening> = Vec::new();
        let mut west_ops: Vec<Opening> = Vec::new();

        for &socket_id in &[t.lower_socket, t.upper_socket] {
            if let Some(socket) = socket_map.get(&socket_id) {
                let half = CORRIDOR_WIDTH / 2;
                match socket.wall {
                    WallDirection::North => north_ops.push(Opening {
                        tangent_min: socket.anchor.0 - half,
                        tangent_max: socket.anchor.0 + half,
                        bottom: lower_z + Q,
                        top: upper_top - Q,
                    }),
                    WallDirection::South => south_ops.push(Opening {
                        tangent_min: socket.anchor.0 - half,
                        tangent_max: socket.anchor.0 + half,
                        bottom: lower_z + Q,
                        top: upper_top - Q,
                    }),
                    WallDirection::East => east_ops.push(Opening {
                        tangent_min: socket.anchor.1 - half,
                        tangent_max: socket.anchor.1 + half,
                        bottom: lower_z + Q,
                        top: upper_top - Q,
                    }),
                    WallDirection::West => west_ops.push(Opening {
                        tangent_min: socket.anchor.1 - half,
                        tangent_max: socket.anchor.1 + half,
                        bottom: lower_z + Q,
                        top: upper_top - Q,
                    }),
                }
            }
        }

        // Emit each stairwell wall with apertures
        emit_wall_with_openings(
            out,
            fx0,
            fy0,
            fx1,
            fy1,
            wall_z0,
            wall_z1,
            &north_ops,
            WallDirection::North,
            fx0,
            fy0,
            fx1,
            fy1,
            conn_wall,
        );
        emit_wall_with_openings(
            out,
            fx0,
            fy0,
            fx1,
            fy1,
            wall_z0,
            wall_z1,
            &south_ops,
            WallDirection::South,
            fx0,
            fy0,
            fx1,
            fy1,
            conn_wall,
        );
        emit_wall_with_openings(
            out,
            fx0,
            fy0,
            fx1,
            fy1,
            wall_z0,
            wall_z1,
            &east_ops,
            WallDirection::East,
            fx0,
            fy0,
            fx1,
            fy1,
            conn_wall,
        );
        emit_wall_with_openings(
            out,
            fx0,
            fy0,
            fx1,
            fy1,
            wall_z0,
            wall_z1,
            &west_ops,
            WallDirection::West,
            fx0,
            fy0,
            fx1,
            fy1,
            conn_wall,
        );

        // Stair steps (treads)
        for &(tx, ty, tz) in &t.treads {
            let tread_depth = t.tread_depth;
            let step_width = 64;
            let half_w = step_width / 2;
            // Determine step orientation from the socket positions
            let lower_sock = socket_map.get(&t.lower_socket);
            let upper_sock = socket_map.get(&t.upper_socket);
            let (sx0, sy0, sx1, sy1) = if let (Some(ls), Some(us)) = (lower_sock, upper_sock) {
                let dx = (us.anchor.0 - ls.anchor.0).abs();
                let dy = (us.anchor.1 - ls.anchor.1).abs();
                if dx >= dy {
                    // Horizontal stair — steps span in Y
                    (tx, ty - half_w, tx + tread_depth, ty + half_w)
                } else {
                    // Vertical stair — steps span in X
                    (tx - half_w, ty, tx + half_w, ty + tread_depth)
                }
            } else {
                // Fallback: horizontal
                (tx, ty - half_w, tx + tread_depth, ty + half_w)
            };
            emit_solid_brush(out, sx0, sy0, tz, sx1, sy1, tz + Q, conn_floor);
        }
    }
}

fn emit_wall_with_openings(
    out: &mut String,
    fx0: i32,
    fy0: i32,
    fx1: i32,
    fy1: i32,
    z0: i32,
    z1: i32,
    openings: &[Opening],
    wall: WallDirection,
    _bound_x0: i32,
    _bound_y0: i32,
    _bound_x1: i32,
    _bound_y1: i32,
    texture: &str,
) {
    let (tangent_min, tangent_max, _bx0, _by0, _bx1, _by1) = match wall {
        WallDirection::North => (fx0, fx1, fx0, fy1 - Q, fx1, fy1),
        WallDirection::South => (fx0, fx1, fx0, fy0, fx1, fy0 + Q),
        WallDirection::East => (fy0, fy1, fx1 - Q, fy0 + Q, fx1, fy1 - Q),
        WallDirection::West => (fy0, fy1, fx0, fy0 + Q, fx0 + Q, fy1 - Q),
    };

    let mut solid_cells: BTreeMap<Cell, ()> = BTreeMap::new();
    for t_cell in tangent_min.div_euclid(Q)..tangent_max.div_euclid(Q) {
        for z_cell in z0.div_euclid(Q)..z1.div_euclid(Q) {
            solid_cells.insert((t_cell, z_cell), ());
        }
    }

    for op in openings {
        if op.tangent_min >= op.tangent_max || op.bottom >= op.top {
            continue;
        }
        for z_cell in op.bottom.div_euclid(Q)..op.top.div_euclid(Q) {
            for t_cell in op.tangent_min.div_euclid(Q)..op.tangent_max.div_euclid(Q) {
                solid_cells.remove(&(t_cell, z_cell));
            }
        }
    }

    for rect in merge_cells_void(&solid_cells) {
        let t0 = rect.x0 * Q;
        let t1 = rect.x1 * Q;
        let bz0 = rect.y0 * Q;
        let bz1 = rect.y1 * Q;

        let (rbx0, rby0, rbx1, rby1) = match wall {
            WallDirection::North => (t0, fy1 - Q, t1, fy1),
            WallDirection::South => (t0, fy0, t1, fy0 + Q),
            WallDirection::East => (fx1 - Q, t0, fx1, t1),
            WallDirection::West => (fx0, t0, fx0 + Q, t1),
        };

        emit_solid_brush(out, rbx0, rby0, bz0, rbx1, rby1, bz1, texture);
    }
}

// ── Grid rectangle merging ─────────────────────────────────────────────────

fn merge_cells<T: Copy + Ord>(cells: &BTreeMap<Cell, T>) -> Vec<GridRect<T>> {
    let mut rows: BTreeMap<i32, Vec<(i32, T)>> = BTreeMap::new();
    for (&(x, y), &value) in cells {
        rows.entry(y).or_default().push((x, value));
    }

    let mut active: BTreeMap<(i32, i32, T), GridRect<T>> = BTreeMap::new();
    let mut finished = Vec::new();
    let mut previous_y: Option<i32> = None;

    for (y, mut row) in rows {
        row.sort_unstable();
        if previous_y.is_some_and(|prev| y != prev + 1) {
            finished.extend(active.into_values());
            active = BTreeMap::new();
        }

        let mut runs = Vec::new();
        let mut index = 0;
        while index < row.len() {
            let (x0, value) = row[index];
            let mut x1 = x0 + 1;
            index += 1;
            while index < row.len() && row[index] == (x1, value) {
                x1 += 1;
                index += 1;
            }
            runs.push((x0, x1, value));
        }

        let mut next = BTreeMap::new();
        for (x0, x1, value) in runs {
            let key = (x0, x1, value);
            let rect = if let Some(mut rect) = active.remove(&key) {
                rect.y1 = y + 1;
                rect
            } else {
                GridRect {
                    x0,
                    x1,
                    y0: y,
                    y1: y + 1,
                    value,
                }
            };
            next.insert(key, rect);
        }
        finished.extend(active.into_values());
        active = next;
        previous_y = Some(y);
    }
    finished.extend(active.into_values());
    finished.sort_unstable();
    finished
}

fn merge_cells_void(cells: &BTreeMap<Cell, ()>) -> Vec<GridRect<()>> {
    let valued: BTreeMap<Cell, u8> = cells.iter().map(|(&k, _)| (k, 0u8)).collect();
    merge_cells(&valued)
        .into_iter()
        .map(|r| GridRect {
            x0: r.x0,
            x1: r.x1,
            y0: r.y0,
            y1: r.y1,
            value: (),
        })
        .collect()
}

// ── Solid brush emission ───────────────────────────────────────────────────

fn emit_solid_brush(
    out: &mut String,
    x0: i32,
    y0: i32,
    z0: i32,
    x1: i32,
    y1: i32,
    z1: i32,
    texture: &str,
) {
    if x0 >= x1 || y0 >= y1 || z0 >= z1 {
        return;
    }
    let brush = junction::make_brush((x0, y0, z0), (x1, y1, z1), texture);
    emit_brush(out, &brush);
}

fn emit_brush(out: &mut String, brush: &crate::intent::Brush) {
    out.push_str("{\n");
    for face in &brush.faces {
        let (p0, p1, p2) = (
            face.plane_points[0],
            face.plane_points[1],
            face.plane_points[2],
        );
        use std::fmt::Write;
        write!(
            out,
            "( {} {} {} ) ( {} {} {} ) ( {} {} {} ) \"{}\" 0 0 0 0.25 0.25\n",
            p0.0, p0.1, p0.2, p1.0, p1.1, p1.2, p2.0, p2.1, p2.2, face.texture,
        )
        .unwrap();
    }
    out.push_str("}\n");
}

// ── Texture lookup ─────────────────────────────────────────────────────────

fn texture_for(palette_name: &str, role: TextureRole) -> &'static str {
    let theme = cc0_dungeon_v2_theme();
    let palette = theme
        .palettes
        .iter()
        .find(|p| p.name == palette_name)
        .unwrap_or_else(|| theme.base_palette());
    match role {
        TextureRole::Floor => palette.floor,
        TextureRole::Wall => palette.wall,
        TextureRole::Ceiling => palette.ceiling,
        TextureRole::Accent => palette.accent.unwrap_or(palette.wall),
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::super::config::EnhancedConfig;
    use super::super::features::apply_features;
    use super::super::placement::place_rooms;
    use super::super::seed::{tags, EnhancedSeed};
    use super::super::theme::{assign_uniform, cc0_dungeon_v2_theme};
    use super::super::topology::build_topology;
    use super::*;

    fn build_full_pipeline(
        seed_val: u64,
    ) -> (
        EnhancedConfig,
        PlacementResult,
        TopologyResult,
        ThemeAssignment,
        FeatureResult,
    ) {
        let cfg = EnhancedConfig::nominal();
        let eseed = EnhancedSeed::new(seed_val);
        let placement = place_rooms(&cfg, eseed.stage_seed(tags::LAYER_PLACEMENT)).unwrap();
        let mut topo_rng = eseed.stage_seed(tags::VERTICAL_TOPOLOGY).rng();
        let topology = build_topology(&cfg, &placement, &mut topo_rng).unwrap();
        let theme = cc0_dungeon_v2_theme();
        let assignment = assign_uniform(&theme, &placement.rooms, &topology);
        let corridor_rng = eseed.stage_seed(tags::CORRIDOR_VARIANCE).rng();
        let feature_rng = eseed.stage_seed(tags::FEATURE_PLACEMENT).rng();
        let features = apply_features(
            &cfg,
            &placement,
            &topology,
            &assignment,
            feature_rng,
            corridor_rng,
        )
        .unwrap();
        (cfg, placement, topology, assignment, features)
    }

    #[test]
    fn emit_nominal_map() {
        let (cfg, placement, topology, theme, features) = build_full_pipeline(99);
        let map = emit_map(&cfg, &placement, &topology, &theme, &features).unwrap();
        assert!(!map.is_empty());
        assert!(map.contains("worldspawn"));
        assert!(map.contains("info_player_start"));
        assert!(map.contains("\"wad\""));
        // Should have light entities
        assert!(map.contains("\"classname\" \"light\""));
        // Should have brushes (at least floor/ceiling per room + walls)
        let brush_count = map.matches("}\n{").count();
        assert!(brush_count > 10, "expected many brushes, got {brush_count}");
    }

    #[test]
    fn emit_map_deterministic() {
        let (cfg, placement, topology, theme, features) = build_full_pipeline(42);
        let a = emit_map(&cfg, &placement, &topology, &theme, &features).unwrap();
        let b = emit_map(&cfg, &placement, &topology, &theme, &features).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn emit_map_contains_spawn_at_feature_origin() {
        let (cfg, placement, topology, theme, features) = build_full_pipeline(42);
        let map = emit_map(&cfg, &placement, &topology, &theme, &features).unwrap();
        let spawn_str = format!(
            "\"origin\" \"{} {} {}\"",
            features.spawn_point.origin.0,
            features.spawn_point.origin.1,
            features.spawn_point.origin.2,
        );
        assert!(
            map.contains(&spawn_str),
            "spawn origin not found: {spawn_str}"
        );
    }

    #[test]
    fn emit_map_rooms_have_correct_textures() {
        let (cfg, placement, topology, theme, features) = build_full_pipeline(42);
        let map = emit_map(&cfg, &placement, &topology, &theme, &features).unwrap();
        // All rooms use base_stone palette under uniform assignment
        assert!(map.contains("bs_floor"));
        assert!(map.contains("bs_wall"));
        assert!(map.contains("bs_ceil"));
    }

    #[test]
    fn emit_map_no_hardcoded_legacy_textures() {
        let (cfg, placement, topology, theme, features) = build_full_pipeline(42);
        let map = emit_map(&cfg, &placement, &topology, &theme, &features).unwrap();
        // Should NOT contain the hardcoded legacy texture names
        assert!(!map.contains("\"stone_floor\""));
        assert!(!map.contains("\"stone_wall\""));
        assert!(!map.contains("\"stone_ceiling\""));
    }

    #[test]
    fn emit_map_stairs_present() {
        let (cfg, placement, topology, theme, features) = build_full_pipeline(42);
        let map = emit_map(&cfg, &placement, &topology, &theme, &features).unwrap();
        // Should have stairwell geometry with connector textures
        assert!(map.contains("conn_floor") || map.contains("conn_wall"));
    }

    #[test]
    fn cell_merge_empty() {
        let cells: BTreeMap<Cell, u8> = BTreeMap::new();
        let result = merge_cells(&cells);
        assert!(result.is_empty());
    }

    #[test]
    fn cell_merge_single() {
        let cells: BTreeMap<Cell, u8> = BTreeMap::from([((0, 0), 5)]);
        let result = merge_cells(&cells);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].x0, 0);
        assert_eq!(result[0].x1, 1);
        assert_eq!(result[0].y0, 0);
        assert_eq!(result[0].y1, 1);
        assert_eq!(result[0].value, 5);
    }

    #[test]
    fn cell_merge_rect() {
        let mut cells = BTreeMap::new();
        for y in 0..4 {
            for x in 0..3 {
                cells.insert((x, y), 1u8);
            }
        }
        let result = merge_cells(&cells);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].x0, 0);
        assert_eq!(result[0].x1, 3);
        assert_eq!(result[0].y0, 0);
        assert_eq!(result[0].y1, 4);
    }
}
