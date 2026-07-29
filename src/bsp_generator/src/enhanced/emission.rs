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
use super::intent::{RoomId, RouteId};
use super::placement::{CandidateSocket, PlacedRoom, PlacementResult, WallDirection};
use super::theme::{cc0_dungeon_v2_theme, TextureRole, ThemeAssignment};
use super::topology::TopologyResult;

const Q: i32 = CONSTRUCTION_QUANTUM as i32;
const CORRIDOR_WIDTH: i32 = 64;
const CORRIDOR_HEIGHT: i32 = 80;

/// Minimum baked light level added to worldspawn so that ericw-tools `light`
/// always produces lightmap data for sealed connector and stair surfaces.
/// Without this, completely unilluminated faces receive `lightofs < 0` and
/// the renderer falls back to a frozen neutral-albedo modulation, making
/// dark connector geometry appear full-bright. Value 16 matches the proven
/// source-map experiment (same seed, same face count, 0 missing lightmaps).
const WORLDSPAWN_MINLIGHT: i32 = 16;

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

#[derive(Debug, Clone, PartialEq, Eq)]
struct PortalThroat {
    wall: WallDirection,
    wall_rect: (i32, i32, i32, i32),
    exterior_rect: (i32, i32, i32, i32),
    opening: Opening,
    floor_z: i32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct RouteCellSet {
    route_id: RouteId,
    span: CorridorCell,
    cells: BTreeSet<Cell>,
    portals: Vec<PortalThroat>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct GridRect<T> {
    x0: i32,
    x1: i32,
    y0: i32,
    y1: i32,
    value: T,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct GridBox {
    x0: i32,
    x1: i32,
    y0: i32,
    y1: i32,
    z0: i32,
    z1: i32,
}

#[derive(Debug, Clone, Copy)]
struct ShellTextures<'a> {
    floor: &'a str,
    wall: &'a str,
    ceiling: &'a str,
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

    // Preserve route ownership until emission. Horizontal routes are unioned
    // only after their vertical spans have separated the two layers.
    let corridor_routes = build_corridor_route_cells(topology, &room_map, &socket_map);

    // ── Emit ──────────────────────────────────────────────────────────
    let mut out = String::new();
    out.push_str("{\n\"classname\" \"worldspawn\"\n");
    out.push_str(&format!("\"wad\" \"{wad}\"\n"));
    out.push_str(&format!("\"_minlight\" \"{WORLDSPAWN_MINLIGHT}\"\n"));

    // Emit room brushes
    for room in &placement.rooms {
        emit_room_brushes(&mut out, room, topology, &room_map, &socket_map, theme)?;
    }

    // Emit corridor geometry (only in non-room cells)
    emit_corridor_brushes(&mut out, &corridor_routes, &room_owned_cells, theme);

    // Emit stair transitions
    emit_stair_transitions(
        &mut out,
        topology,
        &room_map,
        &socket_map,
        &room_owned_cells,
        theme,
    );

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
        emit_split_wall(out, room, *wall_dir, wall_openings, wall_tex);
    }

    Ok(())
}

fn collect_room_openings(
    room: &PlacedRoom,
    topology: &TopologyResult,
    socket_map: &BTreeMap<super::intent::SocketId, &CandidateSocket>,
) -> BTreeMap<WallDirection, Vec<Opening>> {
    let mut openings: BTreeMap<WallDirection, Vec<Opening>> = BTreeMap::new();
    let mut add_socket = |socket: &CandidateSocket| {
        if let Some(portal) = portal_throat(room, socket) {
            openings
                .entry(portal.wall)
                .or_default()
                .push(portal.opening);
        }
    };

    // Corridor route apertures
    for route in &topology.routes {
        for &(socket_id, expected_room) in &[
            (route.source_socket, route.source_room),
            (route.target_socket, route.target_room),
        ] {
            if let Some(socket) = socket_map.get(&socket_id) {
                if socket.room == room.id && socket.room == expected_room {
                    add_socket(socket);
                }
            }
        }
    }

    // Transition (stair) apertures
    for transition in &topology.transitions {
        for &(socket_id, expected_room) in &[
            (transition.lower_socket, transition.lower_room),
            (transition.upper_socket, transition.upper_room),
        ] {
            if let Some(socket) = socket_map.get(&socket_id) {
                if socket.room == room.id && socket.room == expected_room {
                    add_socket(socket);
                }
            }
        }
    }

    for ops in openings.values_mut() {
        ops.sort_unstable_by_key(|o| (o.tangent_min, o.tangent_max, o.bottom, o.top));
        ops.dedup();
    }
    openings
}

/// Derive the one canonical 64×80 portal throat used by both the room wall
/// mask and the connector shell. Sharing this descriptor prevents the two
/// sides of a socket from drifting by one wall cell or one vertical quantum.
fn portal_throat(room: &PlacedRoom, socket: &CandidateSocket) -> Option<PortalThroat> {
    if socket.room != room.id {
        return None;
    }
    let (x0, y0, x1, y1) = room.shell;
    let width = i32::try_from(socket.width).ok()?;
    let bottom = room.floor_z + Q;
    let top = (bottom + CORRIDOR_HEIGHT).min(room.floor_z + room.dims.2 as i32 - Q);
    let (anchor_tangent, inner_min, inner_max) = match socket.wall {
        WallDirection::North | WallDirection::South => (socket.anchor.0, x0 + Q, x1 - Q),
        WallDirection::East | WallDirection::West => (socket.anchor.1, y0 + Q, y1 - Q),
    };
    // Odd-quantum room spans place the semantic midpoint eight units off the
    // construction grid. Canonicalize the whole aperture to four complete
    // cells; independently flooring each bound in the old wall mask shifted
    // the room opening away from the connector throat and left an 8-unit slit.
    let tangent_min =
        ((anchor_tangent - width / 2).div_euclid(Q) * Q).clamp(inner_min, inner_max - width);
    let tangent_max = tangent_min + width;
    if tangent_min >= tangent_max || bottom >= top {
        return None;
    }

    let (wall_rect, exterior_rect) = match socket.wall {
        WallDirection::North => (
            (tangent_min, y1 - Q, tangent_max, y1),
            (tangent_min, y1, tangent_max, y1 + Q),
        ),
        WallDirection::South => (
            (tangent_min, y0, tangent_max, y0 + Q),
            (tangent_min, y0 - Q, tangent_max, y0),
        ),
        WallDirection::East => (
            (x1 - Q, tangent_min, x1, tangent_max),
            (x1, tangent_min, x1 + Q, tangent_max),
        ),
        WallDirection::West => (
            (x0, tangent_min, x0 + Q, tangent_max),
            (x0 - Q, tangent_min, x0, tangent_max),
        ),
    };

    Some(PortalThroat {
        wall: socket.wall,
        wall_rect,
        exterior_rect,
        opening: Opening {
            tangent_min,
            tangent_max,
            bottom,
            top,
        },
        floor_z: room.floor_z,
    })
}

fn emit_split_wall(
    out: &mut String,
    room: &PlacedRoom,
    wall: WallDirection,
    openings: &[Opening],
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

fn build_corridor_route_cells(
    topology: &TopologyResult,
    room_map: &BTreeMap<RoomId, &PlacedRoom>,
    socket_map: &BTreeMap<super::intent::SocketId, &CandidateSocket>,
) -> Vec<RouteCellSet> {
    topology
        .routes
        .iter()
        .map(|route| {
            let floor_z = route.headroom.0 - Q;
            let span = CorridorCell {
                floor_z,
                ceiling_bottom: route.headroom.1,
            };
            let mut cells = BTreeSet::new();
            for &envelope in &route.envelopes {
                insert_rect_cells(&mut cells, envelope);
            }

            let mut portals = Vec::with_capacity(2);
            for &(socket_id, room_id) in &[
                (route.source_socket, route.source_room),
                (route.target_socket, route.target_room),
            ] {
                if let (Some(room), Some(socket)) =
                    (room_map.get(&room_id), socket_map.get(&socket_id))
                {
                    if let Some(portal) = portal_throat(room, socket) {
                        // Every route owns an explicit exterior approach cell.
                        // The separate throat slab below continues across the
                        // full 16-unit room-wall thickness.
                        insert_rect_cells(&mut cells, portal.exterior_rect);
                        portals.push(portal);
                    }
                }
            }

            RouteCellSet {
                route_id: route.id,
                span,
                cells,
                portals,
            }
        })
        .collect()
}

fn insert_rect_cells(cells: &mut BTreeSet<Cell>, rect: (i32, i32, i32, i32)) {
    let (x0, y0, x1, y1) = rect;
    for gy in y0.div_euclid(Q)..y1.div_euclid(Q) {
        for gx in x0.div_euclid(Q)..x1.div_euclid(Q) {
            cells.insert((gx, gy));
        }
    }
}

fn emit_corridor_brushes(
    out: &mut String,
    routes: &[RouteCellSet],
    room_owned_cells: &BTreeSet<Cell>,
    _theme: &ThemeAssignment,
) {
    // Same-layer routes intentionally form junction unions. The span is the
    // outer key, so projected overlap on the other layer can never overwrite
    // or inherit this layer's floor/ceiling values.
    let mut layers: BTreeMap<CorridorCell, (BTreeSet<Cell>, Vec<PortalThroat>)> = BTreeMap::new();
    for route in routes {
        let (cells, portals) = layers.entry(route.span).or_default();
        cells.extend(
            route
                .cells
                .iter()
                .copied()
                .filter(|cell| !room_owned_cells.contains(cell)),
        );
        portals.extend(route.portals.iter().cloned());
    }

    for (span, (cells, mut portals)) in layers {
        portals.sort_by_key(|portal| {
            (
                portal.floor_z,
                portal.wall,
                portal.wall_rect,
                portal.opening.clone(),
            )
        });
        portals.dedup();
        emit_cell_shell(
            out,
            &cells,
            span.floor_z,
            span.ceiling_bottom,
            &portals,
            ShellTextures {
                floor: "conn_floor",
                wall: "conn_wall",
                ceiling: "conn_ceil",
            },
        );
    }
}

/// Emit a sealed shell around a set of open XY cells. Boundary walls occupy
/// the adjacent solid cell, including endpoint room-wall cells. Portal masks
/// are then removed from those walls and replaced by floor/ceiling throat
/// slabs spanning exactly the room's 16-unit wall thickness.
fn emit_cell_shell(
    out: &mut String,
    cells: &BTreeSet<Cell>,
    floor_z: i32,
    ceiling_bottom: i32,
    portals: &[PortalThroat],
    textures: ShellTextures<'_>,
) {
    if cells.is_empty() {
        return;
    }

    let valued: BTreeMap<Cell, u8> = cells.iter().map(|&cell| (cell, 0)).collect();
    for rect in merge_cells(&valued) {
        emit_solid_brush(
            out,
            rect.x0 * Q,
            rect.y0 * Q,
            floor_z,
            rect.x1 * Q,
            rect.y1 * Q,
            floor_z + Q,
            textures.floor,
        );
        emit_solid_brush(
            out,
            rect.x0 * Q,
            rect.y0 * Q,
            ceiling_bottom,
            rect.x1 * Q,
            rect.y1 * Q,
            ceiling_bottom + Q,
            textures.ceiling,
        );
    }

    let mut wall_voxels = BTreeSet::new();
    let shell_top = ceiling_bottom + Q;
    for &cell in cells {
        for neighbor in [
            (cell.0 - 1, cell.1),
            (cell.0 + 1, cell.1),
            (cell.0, cell.1 - 1),
            (cell.0, cell.1 + 1),
        ] {
            if cells.contains(&neighbor) {
                continue;
            }
            for gz in floor_z.div_euclid(Q)..shell_top.div_euclid(Q) {
                wall_voxels.insert((neighbor.0, neighbor.1, gz));
            }
        }
    }

    for portal in portals {
        let (x0, y0, x1, y1) = portal.wall_rect;
        // Remove the clear opening plus the two connector-owned slab cells.
        // The slabs are emitted below with their role-specific textures.
        for gy in y0.div_euclid(Q)..y1.div_euclid(Q) {
            for gx in x0.div_euclid(Q)..x1.div_euclid(Q) {
                for gz in portal.floor_z.div_euclid(Q)..(portal.floor_z + Q).div_euclid(Q) {
                    wall_voxels.remove(&(gx, gy, gz));
                }
                for gz in portal.opening.bottom.div_euclid(Q)..portal.opening.top.div_euclid(Q) {
                    wall_voxels.remove(&(gx, gy, gz));
                }
                for gz in portal.opening.top.div_euclid(Q)..(portal.opening.top + Q).div_euclid(Q) {
                    wall_voxels.remove(&(gx, gy, gz));
                }
            }
        }
    }

    for cube in merge_voxels(&wall_voxels) {
        emit_solid_brush(
            out,
            cube.x0 * Q,
            cube.y0 * Q,
            cube.z0 * Q,
            cube.x1 * Q,
            cube.y1 * Q,
            cube.z1 * Q,
            textures.wall,
        );
    }

    for portal in portals {
        let (x0, y0, x1, y1) = portal.wall_rect;
        emit_solid_brush(
            out,
            x0,
            y0,
            portal.floor_z,
            x1,
            y1,
            portal.floor_z + Q,
            textures.floor,
        );
        emit_solid_brush(
            out,
            x0,
            y0,
            portal.opening.top,
            x1,
            y1,
            portal.opening.top + Q,
            textures.ceiling,
        );
    }
}

// ── Stair transition emission ──────────────────────────────────────────────

fn emit_stair_transitions(
    out: &mut String,
    topology: &TopologyResult,
    room_map: &BTreeMap<RoomId, &PlacedRoom>,
    socket_map: &BTreeMap<super::intent::SocketId, &CandidateSocket>,
    room_owned_cells: &BTreeSet<Cell>,
    _theme: &ThemeAssignment,
) {
    for transition in &topology.transitions {
        let Some(lower_room) = room_map.get(&transition.lower_room) else {
            continue;
        };
        let Some(upper_room) = room_map.get(&transition.upper_room) else {
            continue;
        };

        // The protected footprint describes transition-owned open volume, not
        // four unconditional outer walls. Remove room projections and let the
        // cell-shell boundary follow each actual room wall instead. That makes
        // both stair apertures true abutting interfaces even when the sockets
        // lie on different sides of the footprint bounding rectangle.
        let mut cells = BTreeSet::new();
        insert_rect_cells(&mut cells, transition.footprint);
        cells.retain(|cell| !room_owned_cells.contains(cell));

        let mut portals = Vec::with_capacity(2);
        for &(room, socket_id) in &[
            (*lower_room, transition.lower_socket),
            (*upper_room, transition.upper_socket),
        ] {
            if let Some(socket) = socket_map.get(&socket_id) {
                if let Some(portal) = portal_throat(room, socket) {
                    insert_rect_cells(&mut cells, portal.exterior_rect);
                    portals.push(portal);
                }
            }
        }

        let upper_top = upper_room.floor_z + upper_room.dims.2 as i32;
        emit_cell_shell(
            out,
            &cells,
            lower_room.floor_z,
            upper_top - Q,
            &portals,
            ShellTextures {
                floor: "conn_floor",
                wall: "conn_wall",
                ceiling: "conn_ceil",
            },
        );

        // Stair steps (treads)
        for &(tx, ty, tz) in &transition.treads {
            let tread_depth = transition.tread_depth;
            let half_w = CORRIDOR_WIDTH / 2;
            let lower_socket = socket_map.get(&transition.lower_socket);
            let upper_socket = socket_map.get(&transition.upper_socket);
            let (sx0, sy0, sx1, sy1) =
                if let (Some(lower), Some(upper)) = (lower_socket, upper_socket) {
                    let dx = (upper.anchor.0 - lower.anchor.0).abs();
                    let dy = (upper.anchor.1 - lower.anchor.1).abs();
                    if dx >= dy {
                        (tx, ty - half_w, tx + tread_depth, ty + half_w)
                    } else {
                        (tx - half_w, ty, tx + half_w, ty + tread_depth)
                    }
                } else {
                    (tx, ty - half_w, tx + tread_depth, ty + half_w)
                };
            emit_solid_brush(out, sx0, sy0, tz, sx1, sy1, tz + Q, "conn_floor");
        }
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

fn merge_voxels(voxels: &BTreeSet<(i32, i32, i32)>) -> Vec<GridBox> {
    let mut slices: BTreeMap<i32, BTreeMap<Cell, u8>> = BTreeMap::new();
    for &(x, y, z) in voxels {
        slices.entry(z).or_default().insert((x, y), 0);
    }

    let mut active: BTreeMap<(i32, i32, i32, i32), GridBox> = BTreeMap::new();
    let mut finished = Vec::new();
    let mut previous_z = None;
    for (z, cells) in slices {
        if previous_z.is_some_and(|previous| z != previous + 1) {
            finished.extend(active.into_values());
            active = BTreeMap::new();
        }

        let mut next = BTreeMap::new();
        for rect in merge_cells(&cells) {
            let key = (rect.x0, rect.x1, rect.y0, rect.y1);
            let cube = if let Some(mut cube) = active.remove(&key) {
                cube.z1 = z + 1;
                cube
            } else {
                GridBox {
                    x0: rect.x0,
                    x1: rect.x1,
                    y0: rect.y0,
                    y1: rect.y1,
                    z0: z,
                    z1: z + 1,
                }
            };
            next.insert(key, cube);
        }
        finished.extend(active.into_values());
        active = next;
        previous_z = Some(z);
    }
    finished.extend(active.into_values());
    finished.sort_unstable();
    finished
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
    fn route_cells_preserve_identity_and_vertical_layer() {
        let (_cfg, placement, topology, _theme, _features) = build_full_pipeline(42);
        let room_map: BTreeMap<RoomId, &PlacedRoom> =
            placement.rooms.iter().map(|room| (room.id, room)).collect();
        let socket_map: BTreeMap<super::super::intent::SocketId, &CandidateSocket> = placement
            .sockets
            .iter()
            .map(|socket| (socket.id, socket))
            .collect();
        let routes = build_corridor_route_cells(&topology, &room_map, &socket_map);

        assert_eq!(routes.len(), topology.routes.len());
        assert_eq!(
            routes
                .iter()
                .map(|route| route.route_id)
                .collect::<BTreeSet<_>>(),
            topology
                .routes
                .iter()
                .map(|route| route.id)
                .collect::<BTreeSet<_>>()
        );
        assert_eq!(
            routes
                .iter()
                .map(|route| route.span.floor_z)
                .collect::<BTreeSet<_>>(),
            BTreeSet::from([0, 192])
        );
        assert!(routes
            .iter()
            .all(|route| !route.cells.is_empty() && route.portals.len() == 2));
    }

    #[test]
    fn odd_quantum_room_portal_uses_the_same_four_cells_on_both_sides() {
        let room = PlacedRoom {
            id: RoomId(1),
            layer: super::super::intent::LayerId(0),
            floor_z: 0,
            shell: (1632, 544, 1776, 672),
            dims: (144, 128, 176),
        };
        let socket = CandidateSocket {
            id: super::super::intent::SocketId(5),
            room: room.id,
            wall: WallDirection::South,
            anchor: (1704, 544, 56),
            width: 64,
            transition_capable: true,
        };

        let portal = portal_throat(&room, &socket).unwrap();
        assert_eq!(portal.wall_rect, (1664, 544, 1728, 560));
        assert_eq!(portal.exterior_rect, (1664, 528, 1728, 544));
        assert_eq!(portal.opening.tangent_min, 1664);
        assert_eq!(portal.opening.tangent_max, 1728);
        assert_eq!((portal.opening.bottom, portal.opening.top), (16, 96));
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
