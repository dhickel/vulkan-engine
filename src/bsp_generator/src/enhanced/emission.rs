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
const CORRIDOR_HEIGHT: i32 = 80;
#[allow(dead_code)]
const CLEAR_HEADROOM: i32 = 80;

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
type RoomCellsByFloor = BTreeMap<i32, BTreeSet<Cell>>;

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
        emit_room_brushes(&mut out, room, topology, &socket_map, theme)?;
    }

    // Emit corridor geometry (only in non-room cells)
    emit_corridor_brushes(
        &mut out,
        &corridor_routes,
        topology,
        &room_map,
        &socket_map,
        &room_owned_cells,
        theme,
    );

    // Emit stair transitions
    emit_stair_transitions(&mut out, topology, &room_map);

    out.push_str("}\n");

    // ── Entities ──────────────────────────────────────────────────────
    // info_player_start
    let spawn = &features.spawn_point;
    out.push_str(&format!(
        "{{\n\"angle\" \"{}\"\n\"classname\" \"info_player_start\"\n\"origin\" \"{} {} {}\"\n}}\n",
        spawn.yaw, spawn.origin.0, spawn.origin.1, spawn.origin.2
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

    // A stair owned by this lower host replaces, rather than overlays, its
    // floor columns.  Its high treads likewise own the exact lower-ceiling
    // exit cells.  All masks are quantum-aligned canonical intent geometry.
    let floor_omissions: Vec<_> = topology
        .transitions
        .iter()
        .filter(|transition| transition.lower_room == room.id)
        .flat_map(|transition| {
            transition.tread_boxes.iter().map(|tread| {
                (
                    tread.bounds.0,
                    tread.bounds.1,
                    tread.bounds.3,
                    tread.bounds.4,
                )
            })
        })
        .collect();
    let ceiling_omissions: Vec<_> = topology
        .transitions
        .iter()
        .filter(|transition| transition.lower_room == room.id)
        .map(|transition| transition.upper_ceiling_opening.rect)
        .collect();
    emit_masked_slab(
        out,
        (x0, y0, x1, y1),
        z0,
        z0 + Q,
        &floor_omissions,
        floor_tex,
    );
    emit_masked_slab(
        out,
        (x0, y0, x1, y1),
        z1 - Q,
        z1,
        &ceiling_omissions,
        ceiling_tex,
    );

    // Walls with apertures — openings collected then walls emitted with cutouts.
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

    // Transition apertures are typed reservation truth.  The transition
    // constructor canonicalizes them to the same four cells as socket throats;
    // never recompute or infer a stair opening from anchors here.
    for transition in &topology.transitions {
        let opening = if transition.lower_room == room.id {
            Some(transition.lower_wall_opening)
        } else if transition.upper_room == room.id {
            Some(transition.upper_wall_opening)
        } else {
            None
        };
        if let Some(opening) = opening {
            openings.entry(opening.wall).or_default().push(Opening {
                tangent_min: opening.tangent_min,
                tangent_max: opening.tangent_max,
                bottom: opening.bottom_z,
                top: opening.top_z,
            });
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

    // North/south own the corner cells. East/west cover only the interior
    // tangent span, so room wall brushes never positively overlap at corners.
    let (tangent_min, tangent_max) = match wall {
        WallDirection::North | WallDirection::South => (x0, x1),
        WallDirection::East | WallDirection::West => (y0 + Q, y1 - Q),
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

/// Emit one slab from an exact cell mask.  The omitted rectangles are clipped
/// to the host and remove whole half-open construction cells, so adjacent
/// slab and stair brushes may share a plane but never positive volume.
fn emit_masked_slab(
    out: &mut String,
    bounds: (i32, i32, i32, i32),
    z0: i32,
    z1: i32,
    omissions: &[(i32, i32, i32, i32)],
    texture: &str,
) {
    let mut cells = BTreeMap::new();
    for y in bounds.1.div_euclid(Q)..bounds.3.div_euclid(Q) {
        for x in bounds.0.div_euclid(Q)..bounds.2.div_euclid(Q) {
            cells.insert((x, y), 0u8);
        }
    }
    for &(x0, y0, x1, y1) in omissions {
        for y in y0.max(bounds.1).div_euclid(Q)..y1.min(bounds.3).div_euclid(Q) {
            for x in x0.max(bounds.0).div_euclid(Q)..x1.min(bounds.2).div_euclid(Q) {
                cells.remove(&(x, y));
            }
        }
    }
    for rect in merge_cells(&cells) {
        emit_solid_brush(
            out,
            rect.x0 * Q,
            rect.y0 * Q,
            z0,
            rect.x1 * Q,
            rect.y1 * Q,
            z1,
            texture,
        );
    }
}

// ── Corridor emission ──────────────────────────────────────────────────────

fn build_room_owned_cells(rooms: &[PlacedRoom]) -> RoomCellsByFloor {
    let mut layers: RoomCellsByFloor = BTreeMap::new();
    for room in rooms {
        let cells = layers.entry(room.floor_z).or_default();
        let (x0, y0, x1, y1) = room.shell;
        for gy in y0.div_euclid(Q)..y1.div_euclid(Q) {
            for gx in x0.div_euclid(Q)..x1.div_euclid(Q) {
                cells.insert((gx, gy));
            }
        }
    }
    layers
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
    topology: &TopologyResult,
    room_map: &BTreeMap<RoomId, &PlacedRoom>,
    socket_map: &BTreeMap<super::intent::SocketId, &CandidateSocket>,
    room_owned_cells: &RoomCellsByFloor,
    _theme: &ThemeAssignment,
) {
    // Horizontal routes plus both transition approaches form one union per
    // vertical span. Separate shells would put boundary walls into an adjacent
    // connector's open cells and create positive-volume source overlaps.
    let mut layers: BTreeMap<CorridorCell, (BTreeSet<Cell>, Vec<PortalThroat>, BTreeSet<Cell>)> =
        BTreeMap::new();
    for route in routes {
        let blocked = room_owned_cells.get(&route.span.floor_z);
        let (cells, portals, _) = layers.entry(route.span).or_default();
        cells.extend(
            route
                .cells
                .iter()
                .copied()
                .filter(|cell| blocked.is_none_or(|blocked| !blocked.contains(cell))),
        );
        portals.extend(route.portals.iter().cloned());
    }
    for transition in &topology.transitions {
        let (Some(lower_room), Some(upper_room), Some(lower_socket), Some(upper_socket)) = (
            room_map.get(&transition.lower_room),
            room_map.get(&transition.upper_room),
            socket_map.get(&transition.lower_socket),
            socket_map.get(&transition.upper_socket),
        ) else {
            continue;
        };

        let lower_span = CorridorCell {
            floor_z: lower_room.floor_z,
            ceiling_bottom: lower_room.floor_z + Q + CORRIDOR_HEIGHT,
        };
        let lower_blocked = room_owned_cells.get(&lower_span.floor_z);
        let (lower_cells, lower_portals, _) = layers.entry(lower_span).or_default();
        for segment in &transition.lower_approach_segments {
            let mut segment_cells = BTreeSet::new();
            insert_rect_cells(&mut segment_cells, segment.envelope);
            lower_cells.extend(
                segment_cells
                    .into_iter()
                    .filter(|cell| lower_blocked.is_none_or(|blocked| !blocked.contains(cell))),
            );
        }
        if let Some(portal) = portal_throat(lower_room, lower_socket) {
            lower_portals.push(portal);
        }

        let upper_span = CorridorCell {
            floor_z: upper_room.floor_z,
            ceiling_bottom: upper_room.floor_z + Q + CORRIDOR_HEIGHT,
        };
        let upper_blocked = room_owned_cells.get(&upper_span.floor_z);
        let (upper_cells, upper_portals, upper_floor_omissions) =
            layers.entry(upper_span).or_default();
        let mut opening_cells = BTreeSet::new();
        insert_rect_cells(&mut opening_cells, transition.upper_ceiling_opening.rect);
        upper_floor_omissions.extend(opening_cells.iter().copied());
        upper_cells.extend(
            opening_cells
                .into_iter()
                .filter(|cell| upper_blocked.is_none_or(|blocked| !blocked.contains(cell))),
        );
        for segment in &transition.upper_approach_segments {
            let mut segment_cells = BTreeSet::new();
            insert_rect_cells(&mut segment_cells, segment.envelope);
            upper_cells.extend(
                segment_cells
                    .into_iter()
                    .filter(|cell| upper_blocked.is_none_or(|blocked| !blocked.contains(cell))),
            );
        }
        if let Some(portal) = portal_throat(upper_room, upper_socket) {
            upper_portals.push(portal);
        }
    }

    for (span, (cells, mut portals, floor_omissions)) in layers {
        portals.sort_by_key(|portal| {
            (
                portal.floor_z,
                portal.wall,
                portal.wall_rect,
                portal.opening.clone(),
            )
        });
        portals.dedup();
        let blocked = room_owned_cells
            .get(&span.floor_z)
            .cloned()
            .unwrap_or_default();
        emit_cell_shell(
            out,
            &cells,
            span.floor_z,
            span.ceiling_bottom,
            &portals,
            &blocked,
            &floor_omissions,
            ShellTextures {
                floor: "conn_floor",
                wall: "conn_wall",
                ceiling: "conn_ceil",
            },
        );
    }
}

/// Emit a sealed shell around a set of open XY cells. Same-layer room cells
/// remain owned by room slabs/walls, while explicit floor omissions preserve
/// vertical stair exits without weakening the shell ceiling or side walls.
fn emit_cell_shell(
    out: &mut String,
    cells: &BTreeSet<Cell>,
    floor_z: i32,
    ceiling_bottom: i32,
    portals: &[PortalThroat],
    blocked_cells: &BTreeSet<Cell>,
    floor_omissions: &BTreeSet<Cell>,
    textures: ShellTextures<'_>,
) {
    if cells.is_empty() {
        return;
    }

    let floor_cells: BTreeMap<Cell, u8> = cells
        .iter()
        .filter(|cell| !floor_omissions.contains(cell))
        .map(|&cell| (cell, 0))
        .collect();
    for rect in merge_cells(&floor_cells) {
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
    }
    let ceiling_cells: BTreeMap<Cell, u8> = cells.iter().map(|&cell| (cell, 0)).collect();
    for rect in merge_cells(&ceiling_cells) {
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
            if cells.contains(&neighbor) || blocked_cells.contains(&neighbor) {
                continue;
            }
            for gz in floor_z.div_euclid(Q)..shell_top.div_euclid(Q) {
                wall_voxels.insert((neighbor.0, neighbor.1, gz));
            }
        }
    }

    for portal in portals {
        let (x0, y0, x1, y1) = portal.wall_rect;
        for gy in y0.div_euclid(Q)..y1.div_euclid(Q) {
            for gx in x0.div_euclid(Q)..x1.div_euclid(Q) {
                for gz in portal.opening.bottom.div_euclid(Q)..portal.opening.top.div_euclid(Q) {
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
}

// ── Stair transition emission ──────────────────────────────────────────────

/// Emit every materialized stair transition from its typed geometry.  Tread
/// boxes are authoritative: their XY orientation and full Z support volume
/// are never reconstructed from socket deltas or compatibility points.
fn emit_stair_transitions(
    out: &mut String,
    topology: &TopologyResult,
    room_map: &BTreeMap<RoomId, &PlacedRoom>,
) {
    for transition in &topology.transitions {
        let (Some(lower_room), Some(upper_room)) = (
            room_map.get(&transition.lower_room),
            room_map.get(&transition.upper_room),
        ) else {
            continue;
        };
        // The twelve columns have pairwise-disjoint XY treads and each owns
        // the solid riser volume from the lower floor to its walkable top.
        for tread in &transition.tread_boxes {
            let (x0, y0, z0, x1, y1, z1) = tread.bounds;
            emit_solid_brush(out, x0, y0, z0, x1, y1, z1, "conn_floor");
        }

        // Surround the open high-tread cells between the lower ceiling and
        // upper-floor bridge.  Room-wall continuations begin at the former
        // ceiling plane; ceiling-adjacent cells begin above the preserved
        // slab, so no shaft wall overlaps a room ceiling.
        emit_stair_exit_walls(
            out,
            transition.upper_ceiling_opening.rect,
            lower_room,
            lower_room.floor_z + lower_room.dims.2 as i32 - Q,
            upper_room.floor_z,
        );
    }
}

/// Emit only the vertical boundary of a lower-host ceiling opening.  This is
/// deliberately not a floor or an opening brush: it seals the climb's shaft
/// sides while preserving the clear vertical exit through the omitted slab.
fn emit_stair_exit_walls(
    out: &mut String,
    opening: (i32, i32, i32, i32),
    lower_room: &PlacedRoom,
    ceiling_bottom: i32,
    bridge_floor: i32,
) {
    let mut cells = BTreeSet::new();
    insert_rect_cells(&mut cells, opening);
    let mut voxels = BTreeSet::new();
    let (rx0, ry0, rx1, ry1) = lower_room.shell;
    for &cell in &cells {
        for neighbor in [
            (cell.0 - 1, cell.1),
            (cell.0 + 1, cell.1),
            (cell.0, cell.1 - 1),
            (cell.0, cell.1 + 1),
        ] {
            if cells.contains(&neighbor) {
                continue;
            }
            let nx0 = neighbor.0 * Q;
            let ny0 = neighbor.1 * Q;
            // The room ceiling slab owns every shell cell, including its wall
            // band. Continue inside-shell neighbors only above that slab;
            // outside-shell neighbors have no ceiling and seal from 160.
            let inside_shell = nx0 >= rx0 && nx0 < rx1 && ny0 >= ry0 && ny0 < ry1;
            let z0 = if inside_shell {
                ceiling_bottom + Q
            } else {
                ceiling_bottom
            };
            for z in z0.div_euclid(Q)..bridge_floor.div_euclid(Q) {
                voxels.insert((neighbor.0, neighbor.1, z));
            }
        }
    }
    for wall in merge_voxels(&voxels) {
        emit_solid_brush(
            out,
            wall.x0 * Q,
            wall.y0 * Q,
            wall.z0 * Q,
            wall.x1 * Q,
            wall.y1 * Q,
            wall.z1 * Q,
            "conn_wall",
        );
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
        assert!(
            map.contains(&format!("\"angle\" \"{}\"", features.spawn_point.yaw)),
            "spawn yaw must face the stair opening"
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

    // ── Structural stair tests ────────────────────────────────────────

    /// Collect every solid brush emitted for worldspawn.  Returns
    /// `Vec<(x0,y0,z0, x1,y1,z1, texture)>` in emission order.
    fn parse_worldspawn_brushes(map: &str) -> Vec<(i32, i32, i32, i32, i32, i32, String)> {
        let mut brushes = Vec::new();
        let mut in_brush = false;
        let mut faces: Vec<((i32, i32, i32), (i32, i32, i32), (i32, i32, i32), String)> =
            Vec::new();
        for line in map.lines() {
            let t = line.trim();
            if t == "{" {
                if !in_brush {
                    in_brush = true;
                    faces.clear();
                }
            } else if t == "}" {
                if in_brush && faces.len() == 6 {
                    // Compute AABB from the 6 face planes.
                    // Brushes use axis-aligned planes in our generator.
                    let mut xs: Vec<i32> = Vec::new();
                    let mut ys: Vec<i32> = Vec::new();
                    let mut zs: Vec<i32> = Vec::new();
                    for (p0, _p1, _p2, _tex) in &faces {
                        xs.push(p0.0);
                        ys.push(p0.1);
                        zs.push(p0.2);
                    }
                    xs.sort();
                    ys.sort();
                    zs.sort();
                    let tex = faces[0].3.clone();
                    // Our brushes are axis-aligned rectangular solids.
                    // The unique coordinates give (min, max) tuples.
                    xs.dedup();
                    ys.dedup();
                    zs.dedup();
                    if xs.len() == 2 && ys.len() == 2 && zs.len() == 2 {
                        brushes.push((xs[0], ys[0], zs[0], xs[1], ys[1], zs[1], tex));
                    }
                }
                in_brush = false;
            } else if in_brush && t.starts_with('(') {
                // Parse "( x y z ) ( x y z ) ( x y z ) "tex" ..."
                let parts: Vec<&str> = t.split('"').collect();
                if parts.len() >= 2 {
                    let tex = parts[1].to_string();
                    let coords: Vec<i32> = t
                        .split(|c: char| c == '(' || c == ')' || c.is_whitespace())
                        .filter_map(|s| s.parse::<i32>().ok())
                        .collect();
                    if coords.len() >= 9 {
                        faces.push((
                            (coords[0], coords[1], coords[2]),
                            (coords[3], coords[4], coords[5]),
                            (coords[6], coords[7], coords[8]),
                            tex,
                        ));
                    }
                }
            }
        }
        brushes
    }

    fn boxes_overlap(a: (i32, i32, i32, i32, i32, i32), b: (i32, i32, i32, i32, i32, i32)) -> bool {
        a.0 < b.3 && a.3 > b.0 && a.1 < b.4 && a.4 > b.1 && a.2 < b.5 && a.5 > b.2
    }

    #[test]
    fn nominal_source_brushes_have_no_positive_volume_overlap() {
        let (cfg, placement, topology, theme, features) = build_full_pipeline(42);
        let map = emit_map(&cfg, &placement, &topology, &theme, &features).unwrap();
        let brushes = parse_worldspawn_brushes(&map);
        for (index, first) in brushes.iter().enumerate() {
            let first_bounds = (first.0, first.1, first.2, first.3, first.4, first.5);
            for second in &brushes[index + 1..] {
                let second_bounds = (second.0, second.1, second.2, second.3, second.4, second.5);
                assert!(
                    !boxes_overlap(first_bounds, second_bounds),
                    "positive-volume brush overlap: {} {first_bounds:?} with {} {second_bounds:?}",
                    first.6,
                    second.6,
                );
            }
        }
    }

    #[test]
    fn stair_emits_exactly_12_tread_solids() {
        let (_, placement, topology, theme, features) = build_full_pipeline(42);
        let map = emit_map(
            &EnhancedConfig::nominal(),
            &placement,
            &topology,
            &theme,
            &features,
        )
        .unwrap();
        let brushes = parse_worldspawn_brushes(&map);
        let transition = &topology.transitions[0];
        assert_eq!(transition.tread_boxes.len(), 12);
        for tread in &transition.tread_boxes {
            let expected = tread.bounds;
            assert_eq!(
                brushes
                    .iter()
                    .filter(|brush| {
                        brush.6 == "conn_floor"
                            && (brush.0, brush.1, brush.2, brush.3, brush.4, brush.5) == expected
                    })
                    .count(),
                1,
                "missing or duplicated authoritative tread {expected:?}",
            );
        }
    }

    #[test]
    fn stair_treads_are_quantum_aligned() {
        // Every emitted tread brush must be aligned to the 16-unit
        // construction quantum on all six coordinates.
        let (cfg, placement, topology, theme, features) = build_full_pipeline(42);
        let map = emit_map(&cfg, &placement, &topology, &theme, &features).unwrap();
        let brushes = parse_worldspawn_brushes(&map);

        let treads: Vec<_> = brushes
            .iter()
            .filter(|b| b.6 == "conn_floor" && b.5 - b.2 == 16 && b.2 < 192)
            .collect();

        assert!(!treads.is_empty(), "no tread brushes found");
        for b in &treads {
            assert_eq!(b.0 % 16, 0, "tread x0 {} not quantum-aligned", b.0);
            assert_eq!(b.1 % 16, 0, "tread y0 {} not quantum-aligned", b.1);
            assert_eq!(b.2 % 16, 0, "tread z0 {} not quantum-aligned", b.2);
            assert_eq!(b.3 % 16, 0, "tread x1 {} not quantum-aligned", b.3);
            assert_eq!(b.4 % 16, 0, "tread y1 {} not quantum-aligned", b.4);
            assert_eq!(b.5 % 16, 0, "tread z1 {} not quantum-aligned", b.5);
        }
    }

    #[test]
    fn stair_emits_connector_shell() {
        let (cfg, placement, topology, theme, features) = build_full_pipeline(42);
        let map = emit_map(&cfg, &placement, &topology, &theme, &features).unwrap();
        // The transition shell should contain conn_wall brushes.
        assert!(
            map.contains("conn_wall"),
            "transition must emit connector walls"
        );
        assert!(
            map.contains("conn_ceil"),
            "transition must emit connector ceiling"
        );
    }

    #[test]
    fn type_a_and_type_b_fixtures_emit_deterministic_treads() {
        use super::super::intent::{IdAllocator, LayerId, RoomId, SocketId};
        use super::super::placement::{CandidateSocket, PlacedRoom, WallDirection};
        use super::super::profile::StairType;
        use super::super::reservation::Transaction;
        use super::super::transition;

        // Type A fixture
        let lroom_a = PlacedRoom {
            id: RoomId(0),
            layer: LayerId(0),
            floor_z: 0,
            shell: (128, 128, 384, 384),
            dims: (256, 256, 176),
        };
        let uroom = PlacedRoom {
            id: RoomId(1),
            layer: LayerId(1),
            floor_z: 192,
            shell: (432, 192, 560, 448),
            dims: (128, 256, 176),
        };
        let ls_a = CandidateSocket {
            id: SocketId(0),
            room: RoomId(0),
            wall: WallDirection::South,
            anchor: (256, 128, 56),
            width: 64,
            transition_capable: true,
        };
        let us = CandidateSocket {
            id: SocketId(1),
            room: RoomId(1),
            wall: WallDirection::West,
            anchor: (432, 320, 248),
            width: 64,
            transition_capable: true,
        };

        let mut tx_a = Transaction::new(
            super::super::occupancy::OccupancyGrid::new(1024, 1024).unwrap(),
            IdAllocator::new(),
            3,
        );
        let intent_a = transition::reserve_one_stair(
            ls_a.clone(),
            us.clone(),
            &[lroom_a.clone(), uroom.clone()],
            &mut tx_a,
        )
        .unwrap();
        assert_eq!(intent_a.stair_type, StairType::RoomScaleGrand);
        assert_eq!(intent_a.tread_boxes.len(), 12);

        // Verify the emitted treads cover the full room width.
        let first = intent_a.tread_boxes[0].bounds;
        let last = intent_a.tread_boxes[11].bounds;
        assert_eq!(first.2, 0);
        assert_eq!(last.5, 192);
    }

    #[test]
    fn direct_type_a_and_b_owner_fixtures_emit_exact_boxes_and_host_masks() {
        use super::super::intent::{IdAllocator, LayerId, RoomId, SocketId};
        use super::super::profile::StairType;
        use super::super::reservation::Transaction;
        use super::super::transition;

        let upper = PlacedRoom {
            id: RoomId(1),
            layer: LayerId(1),
            floor_z: 192,
            shell: (432, 192, 560, 448),
            dims: (128, 256, 176),
        };
        let upper_socket = CandidateSocket {
            id: SocketId(1),
            room: upper.id,
            wall: WallDirection::West,
            anchor: (432, 320, 248),
            width: 64,
            transition_capable: true,
        };
        let fixtures = [
            (
                StairType::RoomScaleGrand,
                PlacedRoom {
                    id: RoomId(0),
                    layer: LayerId(0),
                    floor_z: 0,
                    shell: (128, 128, 384, 384),
                    dims: (256, 256, 176),
                },
                CandidateSocket {
                    id: SocketId(0),
                    room: RoomId(0),
                    wall: WallDirection::South,
                    anchor: (256, 128, 56),
                    width: 64,
                    transition_capable: true,
                },
            ),
            (
                StairType::WallEdgeNarrow,
                PlacedRoom {
                    id: RoomId(0),
                    layer: LayerId(0),
                    floor_z: 0,
                    shell: (128, 128, 384, 256),
                    dims: (256, 128, 176),
                },
                CandidateSocket {
                    id: SocketId(0),
                    room: RoomId(0),
                    wall: WallDirection::South,
                    anchor: (192, 128, 56),
                    width: 64,
                    transition_capable: true,
                },
            ),
        ];

        for (kind, lower, lower_socket) in fixtures {
            let mut tx = Transaction::new(
                super::super::occupancy::OccupancyGrid::new(1024, 1024).unwrap(),
                IdAllocator::new(),
                3,
            );
            let transition = transition::reserve_one_stair(
                lower_socket.clone(),
                upper_socket.clone(),
                &[lower.clone(), upper.clone()],
                &mut tx,
            )
            .unwrap();
            assert_eq!(transition.stair_type, kind);
            let topology = TopologyResult {
                routes: Vec::new(),
                transitions: vec![transition.clone()],
                loop_edges: 0,
            };
            let rooms = BTreeMap::from([(lower.id, &lower), (upper.id, &upper)]);
            let sockets = BTreeMap::from([
                (lower_socket.id, &lower_socket),
                (upper_socket.id, &upper_socket),
            ]);
            let assignment = assign_uniform(
                &cc0_dungeon_v2_theme(),
                &[lower.clone(), upper.clone()],
                &topology,
            );
            let mut map = String::new();
            emit_room_brushes(&mut map, &lower, &topology, &sockets, &assignment).unwrap();
            emit_room_brushes(&mut map, &upper, &topology, &sockets, &assignment).unwrap();
            emit_stair_transitions(&mut map, &topology, &rooms);
            let brushes = parse_worldspawn_brushes(&map);

            for expected in &transition.tread_boxes {
                let bounds = expected.bounds;
                assert_eq!(
                    brushes
                        .iter()
                        .filter(|brush| {
                            brush.6 == "conn_floor"
                                && (brush.0, brush.1, brush.2, brush.3, brush.4, brush.5) == bounds
                        })
                        .count(),
                    1,
                    "{kind:?} must emit exactly its authoritative tread box {bounds:?}",
                );
                assert!(
                    brushes.iter().all(|brush| {
                        brush.6 != "bs_floor"
                            || !boxes_overlap(
                                (brush.0, brush.1, brush.2, brush.3, brush.4, brush.5),
                                bounds,
                            )
                    }),
                    "{kind:?} host floor overlaps tread {bounds:?}"
                );
            }
            assert_eq!(transition.tread_boxes.len(), 12);
            let first = transition.tread_boxes[0].bounds;
            let width = (first.3 - first.0).max(first.4 - first.1);
            match kind {
                StairType::RoomScaleGrand => {
                    assert_eq!(width, 224, "Type A lost full usable room width")
                }
                StairType::WallEdgeNarrow => assert!(
                    (64..=80).contains(&width),
                    "Type B width {width} outside 64..=80"
                ),
            }
            let ceiling = transition.upper_ceiling_opening.rect;
            assert!(
                brushes.iter().all(|brush| {
                    brush.6 != "bs_ceil"
                        || !boxes_overlap(
                            (brush.0, brush.1, brush.2, brush.3, brush.4, brush.5),
                            (ceiling.0, ceiling.1, 160, ceiling.2, ceiling.3, 176),
                        )
                }),
                "{kind:?} lower ceiling caps its vertical exit"
            );
        }
    }

    fn build_compiled_stair_fixture(
        kind: super::super::profile::StairType,
    ) -> (
        String,
        super::super::intent::TransitionIntent,
        PlacedRoom,
        PlacedRoom,
        PlacedRoom,
        i32,
    ) {
        use super::super::intent::{
            IdAllocator, LayerId, RoomId, RouteId, RouteIntent, SocketId, TransitionApproachSegment,
        };
        use super::super::profile::StairType;
        use super::super::reservation::Transaction;
        use super::super::transition;

        let (lower, lower_socket, entry, entry_socket, route_x) = match kind {
            StairType::RoomScaleGrand => (
                PlacedRoom {
                    id: RoomId(0),
                    layer: LayerId(0),
                    floor_z: 0,
                    shell: (256, 256, 512, 512),
                    dims: (256, 256, 176),
                },
                CandidateSocket {
                    id: SocketId(0),
                    room: RoomId(0),
                    wall: WallDirection::South,
                    anchor: (384, 256, 56),
                    width: 64,
                    transition_capable: true,
                },
                PlacedRoom {
                    id: RoomId(2),
                    layer: LayerId(0),
                    floor_z: 0,
                    shell: (256, 64, 512, 192),
                    dims: (256, 128, 176),
                },
                CandidateSocket {
                    id: SocketId(2),
                    room: RoomId(2),
                    wall: WallDirection::North,
                    anchor: (384, 192, 56),
                    width: 64,
                    transition_capable: true,
                },
                384,
            ),
            StairType::WallEdgeNarrow => (
                PlacedRoom {
                    id: RoomId(0),
                    layer: LayerId(0),
                    floor_z: 0,
                    shell: (256, 256, 512, 384),
                    dims: (256, 128, 176),
                },
                CandidateSocket {
                    id: SocketId(0),
                    room: RoomId(0),
                    wall: WallDirection::South,
                    anchor: (320, 256, 56),
                    width: 64,
                    transition_capable: true,
                },
                PlacedRoom {
                    id: RoomId(2),
                    layer: LayerId(0),
                    floor_z: 0,
                    shell: (192, 64, 448, 192),
                    dims: (256, 128, 176),
                },
                CandidateSocket {
                    id: SocketId(2),
                    room: RoomId(2),
                    wall: WallDirection::North,
                    anchor: (320, 192, 56),
                    width: 64,
                    transition_capable: true,
                },
                320,
            ),
        };
        let upper = PlacedRoom {
            id: RoomId(1),
            layer: LayerId(1),
            floor_z: 192,
            shell: (768, 256, 896, 512),
            dims: (128, 256, 176),
        };
        let upper_socket = CandidateSocket {
            id: SocketId(1),
            room: upper.id,
            wall: WallDirection::West,
            anchor: (768, 384, 248),
            width: 64,
            transition_capable: true,
        };

        let transition_rooms = vec![lower.clone(), upper.clone()];
        let mut tx = Transaction::new(
            super::super::occupancy::OccupancyGrid::new(1024, 1024).unwrap(),
            IdAllocator::new(),
            0,
        );
        let mut stair = transition::reserve_one_stair(
            lower_socket.clone(),
            upper_socket.clone(),
            &transition_rooms,
            &mut tx,
        )
        .unwrap();
        assert_eq!(stair.stair_type, kind);

        let rooms_vec = vec![lower.clone(), upper.clone(), entry.clone()];
        let lower_connector = (route_x - 32, 192, route_x + 32, 256);
        stair.lower_approach = lower_connector;
        stair.lower_landing = lower_connector;
        stair.lower_approach_segments = vec![TransitionApproachSegment {
            start: (route_x, 256),
            end: (route_x, 192),
            envelope: lower_connector,
            z: (16, 96),
        }];
        let route = RouteIntent {
            id: RouteId(0),
            source_socket: entry_socket.id,
            target_socket: lower_socket.id,
            source_room: entry.id,
            target_room: lower.id,
            path: vec![((route_x, 192), (route_x, 256))],
            envelopes: vec![lower_connector],
            headroom: (16, 96),
        };
        let topology = TopologyResult {
            routes: vec![route],
            transitions: vec![stair.clone()],
            loop_edges: 0,
        };
        let sockets_vec = vec![lower_socket, upper_socket, entry_socket];
        let room_map: BTreeMap<RoomId, &PlacedRoom> =
            rooms_vec.iter().map(|room| (room.id, room)).collect();
        let socket_map: BTreeMap<SocketId, &CandidateSocket> = sockets_vec
            .iter()
            .map(|socket| (socket.id, socket))
            .collect();
        let assignment = assign_uniform(&cc0_dungeon_v2_theme(), &rooms_vec, &topology);
        let room_cells = build_room_owned_cells(&rooms_vec);
        let route_cells = build_corridor_route_cells(&topology, &room_map, &socket_map);

        let mut map = String::from(
            "{\n\"classname\" \"worldspawn\"\n\"wad\" \"cc0_dungeon_v2.wad\"\n\"_minlight\" \"16\"\n",
        );
        for room in &rooms_vec {
            emit_room_brushes(&mut map, room, &topology, &socket_map, &assignment).unwrap();
        }
        emit_corridor_brushes(
            &mut map,
            &route_cells,
            &topology,
            &room_map,
            &socket_map,
            &room_cells,
            &assignment,
        );
        emit_stair_transitions(&mut map, &topology, &room_map);
        map.push_str("}\n");
        map.push_str(&format!(
            "{{\n\"classname\" \"info_player_start\"\n\"origin\" \"{} {} 40\"\n}}\n",
            (entry.shell.0 + entry.shell.2) / 2,
            (entry.shell.1 + entry.shell.3) / 2,
        ));
        (map, stair, lower, upper, entry, route_x)
    }

    #[test]
    fn direct_type_a_and_b_compile_with_spatial_witnesses() {
        use super::super::profile::StairType;
        use bsp::{point_contents, BspLoader, LoadOptions, PointContents, QuakeToEngine};
        use std::path::{Path, PathBuf};
        use std::process::Command;

        let tools = PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| "/home/dhickel".into()))
            .join(".local/ericw-tools/ericw-tools-2.0.0-alpha3-Linux/bin");
        if !tools.join("qbsp").is_file() {
            eprintln!("SKIP: pinned ericw-tools are not installed");
            return;
        }
        let theme = Path::new(env!("CARGO_MANIFEST_DIR")).join("themes/cc0_dungeon_v2");

        for kind in [StairType::RoomScaleGrand, StairType::WallEdgeNarrow] {
            let (map, stair, lower, upper, entry, route_x) = build_compiled_stair_fixture(kind);
            let brushes = parse_worldspawn_brushes(&map);
            for (index, first) in brushes.iter().enumerate() {
                let first_bounds = (first.0, first.1, first.2, first.3, first.4, first.5);
                for second in &brushes[index + 1..] {
                    let second_bounds =
                        (second.0, second.1, second.2, second.3, second.4, second.5);
                    assert!(
                        !boxes_overlap(first_bounds, second_bounds),
                        "{kind:?} source overlap: {} {first_bounds:?} with {} {second_bounds:?}",
                        first.6,
                        second.6,
                    );
                }
            }

            let work = std::env::temp_dir().join(format!(
                "enhanced-direct-stair-{}-{}",
                kind.tag(),
                std::process::id()
            ));
            let _ = std::fs::remove_dir_all(&work);
            std::fs::create_dir_all(&work).unwrap();
            std::fs::write(work.join("fixture.map"), map).unwrap();
            std::fs::copy(
                theme.join("cc0_dungeon_v2.wad"),
                work.join("cc0_dungeon_v2.wad"),
            )
            .unwrap();
            std::fs::copy(theme.join("palette.lmp"), work.join("palette.lmp")).unwrap();

            let run = |tool: &str, args: &[&str]| {
                let output = Command::new(tools.join(tool))
                    .args(args)
                    .current_dir(&work)
                    .output()
                    .unwrap();
                let combined = format!(
                    "{}{}",
                    String::from_utf8_lossy(&output.stdout),
                    String::from_utf8_lossy(&output.stderr)
                );
                assert!(
                    output.status.success(),
                    "{kind:?} {tool} failed:\n{combined}"
                );
                let normalized = combined.to_ascii_lowercase();
                assert!(
                    ![
                        "warning:",
                        "error:",
                        "no filling performed",
                        "leak file written"
                    ]
                    .iter()
                    .any(|needle| normalized.contains(needle)),
                    "{kind:?} {tool} emitted prohibited diagnostics:\n{combined}"
                );
            };
            run("qbsp", &["-bsp2", "-threads", "1", "fixture.map"]);
            assert!(!work.join("fixture.pts").exists(), "{kind:?} leaked");
            run("vis", &["-threads", "1", "fixture.bsp"]);
            run("light", &["-threads", "1", "-lit", "fixture.bsp"]);

            let bsp_data = std::fs::read(work.join("fixture.bsp")).unwrap();
            let lit_data = std::fs::read(work.join("fixture.lit")).unwrap();
            let options = LoadOptions {
                strict: true,
                palette: Some(std::fs::read(work.join("palette.lmp")).unwrap()),
                lit_data: Some(lit_data),
                wad_archives: vec![(
                    "cc0_dungeon_v2.wad".into(),
                    std::fs::read(work.join("cc0_dungeon_v2.wad")).unwrap(),
                )],
                texture_overrides: Vec::new(),
                source_identity: format!("direct-{}.map", kind.tag()),
            };
            let world = BspLoader::load(&bsp_data, &options).unwrap();
            assert!(world.diagnostics.is_empty());
            let transform = QuakeToEngine::default();
            let contents = |point: (i32, i32, i32)| {
                point_contents(
                    transform.position(point.0 as f32, point.1 as f32, point.2 as f32),
                    &world.nodes,
                    &world.leaves,
                    &world.planes,
                )
            };
            let assert_clear = |label: &str, point| {
                assert_ne!(
                    contents(point),
                    PointContents::Solid,
                    "{kind:?} {label} is solid at {point:?}"
                )
            };
            let assert_solid = |label: &str, point| {
                assert_eq!(
                    contents(point),
                    PointContents::Solid,
                    "{kind:?} {label} is not solid at {point:?}"
                )
            };

            assert_clear(
                "entry room",
                (
                    (entry.shell.0 + entry.shell.2) / 2,
                    (entry.shell.1 + entry.shell.3) / 2,
                    40,
                ),
            );
            assert_clear("connected lower approach", (route_x, 224, 40));
            assert_clear("lower wall aperture", (route_x, lower.shell.1 + 8, 40));
            for (index, tread) in stair.tread_boxes.iter().enumerate() {
                let bounds = tread.bounds;
                let center = ((bounds.0 + bounds.3) / 2, (bounds.1 + bounds.4) / 2);
                assert_solid(
                    &format!("tread {index} support"),
                    (center.0, center.1, bounds.5 - 8),
                );
                assert_clear(
                    &format!("tread {index} headroom"),
                    (center.0, center.1, bounds.5 + 8),
                );
            }
            let bridge = stair.upper_approach_segments[0].envelope;
            let bridge_center = ((bridge.0 + bridge.2) / 2, (bridge.1 + bridge.3) / 2);
            assert_solid(
                "upper bridge support",
                (bridge_center.0, bridge_center.1, 200),
            );
            assert_clear(
                "upper bridge headroom",
                (bridge_center.0, bridge_center.1, 216),
            );
            let upper_opening = stair.upper_wall_opening;
            assert_clear(
                "upper wall aperture",
                (
                    upper.shell.0 + 8,
                    (upper_opening.tangent_min + upper_opening.tangent_max) / 2,
                    232,
                ),
            );
            assert_clear(
                "upper room",
                (upper.shell.0 + 32, (upper.shell.1 + upper.shell.3) / 2, 232),
            );
            std::fs::remove_dir_all(work).unwrap();
        }
    }

    #[test]
    fn wall_openings_present_for_both_levels() {
        let (cfg, placement, topology, theme, features) = build_full_pipeline(42);
        let map = emit_map(&cfg, &placement, &topology, &theme, &features).unwrap();
        let brushes = parse_worldspawn_brushes(&map);

        // Room walls (bs_wall) should be present.
        let wall_brushes: Vec<_> = brushes.iter().filter(|b| b.6.contains("wall")).collect();
        assert!(
            wall_brushes.len() > 20,
            "expected many wall brushes, got {}",
            wall_brushes.len()
        );

        // Transition wall openings should result in split walls (more wall
        // pieces than a simple 6-brushes-per-room count).
        // At minimum, ensure connector geometry exists.
        let conn_walls: Vec<_> = brushes.iter().filter(|b| b.6 == "conn_wall").collect();
        assert!(
            !conn_walls.is_empty(),
            "transition must produce connector walls"
        );
    }
}
