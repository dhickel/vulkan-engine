//! Builds the final [`EmissionIntent`] from the layout and routed corridor
//! network. Produces sealed solid brushes for rooms, corridors, and junctions,
//! plus point entities (spawn, lights).
//!
//! Every brush is built via [`crate::junction::make_brush`], which produces
//! axis-aligned boxes with exactly 6 faces in canonical face order (bottom,
//! top, north, south, west, east).

use crate::config::CONSTRUCTION_QUANTUM;
use crate::intent::{
    Brush, Corridor, EmissionIntent, EntityIntent, LayoutIntent, RoomIntent, RoutedIntent,
};
use crate::junction;

// ── Texture role names ────────────────────────────────────────────────────
//
// Bound from the CC0 Stone Beta theme (themes/cc0_stone_beta/theme.toml).

/// Texture for room floors.  Theme role: `floor`.
const FLOOR_TEXTURE: &str = "stone_floor";
/// Texture for room and corridor ceilings.  Theme role: `ceiling`.
const CEILING_TEXTURE: &str = "stone_ceiling";
/// Texture for room walls.  Theme role: `wall`.
const WALL_TEXTURE: &str = "stone_wall";
/// Texture for corridor walls.  Theme role: `wall`.
const CORRIDOR_WALL_TEXTURE: &str = "stone_wall";
/// Texture for corridor floors.  Theme role: `floor`.
const CORRIDOR_FLOOR_TEXTURE: &str = "stone_floor";

// ── Construction parameters ───────────────────────────────────────────────

/// Wall / floor / ceiling slab thickness in Quake units.
const SLAB: i32 = CONSTRUCTION_QUANTUM as i32; // 16

// ── Public API ────────────────────────────────────────────────────────────

/// Build the complete [`EmissionIntent`] from a placed layout and its routed
/// corridor network.
///
/// # Brush categories (all worldspawn)
///
/// | category      | brushes per element | total            |
/// |---------------|---------------------|------------------|
/// | room          | 6                   | 6 × room count   |
/// | corridor      | 4                   | 4 × seg count    |
/// | junction      | 0–N                 | variable         |
///
/// # Entities
///
/// - One `info_player_start` at the first room centre.
/// - One `light` at every room centre (intensity 300).
///
/// # Panics
///
/// Debug-panics if any brush does not have exactly 6 faces.
pub fn build_emission(layout: &LayoutIntent, routed: &RoutedIntent) -> EmissionIntent {
    let mut brushes: Vec<Brush> = Vec::new();

    // ── Room brushes ──────────────────────────────────────────────────
    for room in &layout.rooms {
        brushes.extend(build_room_brushes(room));
    }

    // ── Corridor brushes ──────────────────────────────────────────────
    for corr in &routed.corridors {
        brushes.extend(build_corridor_brushes(corr));
    }

    // ── Junction closure brushes ──────────────────────────────────────
    brushes.extend(junction::build_junction_closures(&routed.corridors));

    // ── Validation ────────────────────────────────────────────────────
    for (i, brush) in brushes.iter().enumerate() {
        debug_assert_eq!(
            brush.faces.len(),
            6,
            "brush {} has {} faces, expected 6",
            i,
            brush.faces.len()
        );
    }

    // ── Entities ──────────────────────────────────────────────────────
    let mut entities: Vec<EntityIntent> = Vec::new();

    // info_player_start at first room centre
    if let Some(first) = layout.rooms.first() {
        let centre = room_centre(first);
        entities.push(EntityIntent {
            classname: "info_player_start".to_string(),
            origin: centre,
            properties: vec![
                ("classname".to_string(), "info_player_start".to_string()),
                (
                    "origin".to_string(),
                    format!("{} {} {}", centre.0, centre.1, centre.2),
                ),
            ],
            brushes: Vec::new(),
        });
    }

    // light at every room centre
    for room in &layout.rooms {
        let centre = room_centre(room);
        entities.push(EntityIntent {
            classname: "light".to_string(),
            origin: centre,
            properties: vec![
                ("classname".to_string(), "light".to_string()),
                (
                    "origin".to_string(),
                    format!("{} {} {}", centre.0, centre.1, centre.2),
                ),
                ("light".to_string(), "300".to_string()),
            ],
            brushes: Vec::new(),
        });
    }

    EmissionIntent {
        brushes,
        entities,
        wad: "cc0_stone_beta.wad".to_string(),
    }
}

// ── Room geometry ─────────────────────────────────────────────────────────

/// Build 6 solid brushes that enclose a room: floor, ceiling, north wall,
/// south wall, east wall, west wall. Each brush is an axis-aligned slab
/// `SLAB` units thick.
fn build_room_brushes(room: &RoomIntent) -> Vec<Brush> {
    let x = room.position.0;
    let y = room.position.1;
    let z = room.position.2;
    let dx = room.dimensions.0 as i32;
    let dy = room.dimensions.1 as i32;
    let dz = room.dimensions.2 as i32;

    vec![
        // floor
        junction::make_brush((x, y, z), (x + dx, y + dy, z + SLAB), FLOOR_TEXTURE),
        // ceiling
        junction::make_brush(
            (x, y, z + dz - SLAB),
            (x + dx, y + dy, z + dz),
            CEILING_TEXTURE,
        ),
        // north wall  (positive Y)
        junction::make_brush(
            (x, y + dy - SLAB, z),
            (x + dx, y + dy, z + dz),
            WALL_TEXTURE,
        ),
        // south wall  (negative Y)
        junction::make_brush((x, y, z), (x + dx, y + SLAB, z + dz), WALL_TEXTURE),
        // east wall   (positive X)
        junction::make_brush(
            (x + dx - SLAB, y, z),
            (x + dx, y + dy, z + dz),
            WALL_TEXTURE,
        ),
        // west wall   (negative X)
        junction::make_brush((x, y, z), (x + SLAB, y + dy, z + dz), WALL_TEXTURE),
    ]
}

// ── Corridor geometry ─────────────────────────────────────────────────────

/// Build 4 solid brushes for a corridor segment: floor, ceiling, and two
/// side walls. No end-cap brushes are emitted — the open ends face rooms
/// or junctions.
fn build_corridor_brushes(corr: &Corridor) -> Vec<Brush> {
    let hw = (corr.width / 2) as i32;
    let h = corr.height as i32;
    let z = corr.start.2;

    let (x0, x1) = if corr.start.0 <= corr.end.0 {
        (corr.start.0, corr.end.0)
    } else {
        (corr.end.0, corr.start.0)
    };
    let (y0, y1) = if corr.start.1 <= corr.end.1 {
        (corr.start.1, corr.end.1)
    } else {
        (corr.end.1, corr.start.1)
    };

    let is_horizontal = (corr.end.0 - corr.start.0).abs() >= (corr.end.1 - corr.start.1).abs();

    if is_horizontal {
        // East-West: walls are north and south
        vec![
            // floor
            junction::make_brush(
                (x0, y0 - hw, z),
                (x1, y0 + hw, z + SLAB),
                CORRIDOR_FLOOR_TEXTURE,
            ),
            // ceiling
            junction::make_brush(
                (x0, y0 - hw, z + h - SLAB),
                (x1, y0 + hw, z + h),
                CEILING_TEXTURE,
            ),
            // north wall
            junction::make_brush(
                (x0, y0 + hw - SLAB, z),
                (x1, y0 + hw, z + h),
                CORRIDOR_WALL_TEXTURE,
            ),
            // south wall
            junction::make_brush(
                (x0, y0 - hw, z),
                (x1, y0 - hw + SLAB, z + h),
                CORRIDOR_WALL_TEXTURE,
            ),
        ]
    } else {
        // North-South: walls are east and west
        vec![
            // floor
            junction::make_brush(
                (x0 - hw, y0, z),
                (x0 + hw, y1, z + SLAB),
                CORRIDOR_FLOOR_TEXTURE,
            ),
            // ceiling
            junction::make_brush(
                (x0 - hw, y0, z + h - SLAB),
                (x0 + hw, y1, z + h),
                CEILING_TEXTURE,
            ),
            // east wall
            junction::make_brush(
                (x0 + hw - SLAB, y0, z),
                (x0 + hw, y1, z + h),
                CORRIDOR_WALL_TEXTURE,
            ),
            // west wall
            junction::make_brush(
                (x0 - hw, y0, z),
                (x0 - hw + SLAB, y1, z + h),
                CORRIDOR_WALL_TEXTURE,
            ),
        ]
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────

/// Compute the centre point of a room (floor-level, midpoint of XY).
fn room_centre(room: &RoomIntent) -> (i32, i32, i32) {
    (
        room.position.0 + room.dimensions.0 as i32 / 2,
        room.position.1 + room.dimensions.1 as i32 / 2,
        room.position.2,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intent::Junction;

    fn room(x: i32, y: i32, z: i32, dx: u32, dy: u32, dz: u32) -> RoomIntent {
        RoomIntent {
            position: (x, y, z),
            dimensions: (dx, dy, dz),
        }
    }

    fn corridor_h(x: i32, y: i32, z: i32, length: i32) -> Corridor {
        Corridor {
            start: (x, y, z),
            end: (x + length, y, z),
            width: 64,
            height: 80,
        }
    }

    // ── Room brushes ──────────────────────────────────────────────────

    #[test]
    fn room_produces_six_brushes() {
        let room = room(0, 0, 0, 64, 64, 128);
        let brushes = build_room_brushes(&room);
        assert_eq!(brushes.len(), 6);
    }

    #[test]
    fn each_room_brush_has_six_faces() {
        let room = room(0, 0, 0, 64, 64, 128);
        for b in &build_room_brushes(&room) {
            assert_eq!(b.faces.len(), 6);
        }
    }

    #[test]
    fn room_brush_floor_is_at_bottom() {
        let room = room(0, 0, 0, 64, 64, 128);
        let brushes = build_room_brushes(&room);
        // Floor brush has 6 faces; bottom face has z = room.position.2 = 0
        let floor = &brushes[0];
        let bottom_face = &floor.faces[0]; // first face is bottom
                                           // All 3 points of bottom face have same Z = 0
        for p in &bottom_face.plane_points {
            assert_eq!(p.2, 0);
        }
        assert_eq!(bottom_face.texture, FLOOR_TEXTURE);
    }

    // ── Corridor brushes ──────────────────────────────────────────────

    #[test]
    fn corridor_produces_four_brushes() {
        let corr = corridor_h(0, 0, 0, 128);
        let brushes = build_corridor_brushes(&corr);
        assert_eq!(brushes.len(), 4);
    }

    #[test]
    fn each_corridor_brush_has_six_faces() {
        let corr = corridor_h(0, 0, 0, 128);
        for b in &build_corridor_brushes(&corr) {
            assert_eq!(b.faces.len(), 6);
        }
    }

    #[test]
    fn vertical_corridor_is_oriented_north_south() {
        let corr = Corridor {
            start: (64, 0, 0),
            end: (64, 128, 0),
            width: 64,
            height: 80,
        };
        let brushes = build_corridor_brushes(&corr);
        assert_eq!(brushes.len(), 4);
        for b in &brushes {
            assert_eq!(b.faces.len(), 6);
        }
    }

    // ── build_emission integration ────────────────────────────────────

    #[test]
    fn emission_has_worldspawn_wad_reference() {
        let layout = LayoutIntent {
            rooms: vec![room(0, 0, 0, 64, 64, 128)],
            edges: Vec::new(),
            loop_count: 0,
        };
        let routed = RoutedIntent {
            corridors: Vec::new(),
            junctions: Vec::new(),
        };
        let emission = build_emission(&layout, &routed);
        assert_eq!(emission.wad, "cc0_stone_beta.wad");
    }

    #[test]
    fn emission_has_spawn_and_lights() {
        let layout = LayoutIntent {
            rooms: vec![
                room(0, 0, 0, 64, 64, 128),
                room(160, 0, 0, 64, 64, 128),
                room(0, 160, 0, 64, 64, 128),
            ],
            edges: Vec::new(),
            loop_count: 0,
        };
        let routed = RoutedIntent {
            corridors: Vec::new(),
            junctions: Vec::new(),
        };
        let emission = build_emission(&layout, &routed);
        // 1 spawn + 3 lights = 4 entities
        assert_eq!(emission.entities.len(), 4);
        assert_eq!(emission.entities[0].classname, "info_player_start");
        assert_eq!(emission.entities[1].classname, "light");
        assert_eq!(emission.entities[2].classname, "light");
        assert_eq!(emission.entities[3].classname, "light");
    }

    #[test]
    fn emission_counts_match_layout() {
        let rooms = vec![room(0, 0, 0, 64, 64, 128), room(160, 0, 0, 64, 64, 128)];
        let corr = corridor_h(64, 0, 0, 96);
        let layout = LayoutIntent {
            rooms,
            edges: vec![(0, 1)],
            loop_count: 0,
        };
        let routed = RoutedIntent {
            corridors: vec![corr],
            junctions: Vec::new(),
        };
        let emission = build_emission(&layout, &routed);
        // 2 rooms × 6 + 1 corridor × 4 = 16 brushes
        assert_eq!(emission.brushes.len(), 16);
        // 1 spawn + 2 lights = 3 entities
        assert_eq!(emission.entities.len(), 3);
        // All brushes have 6 faces
        for b in &emission.brushes {
            assert_eq!(b.faces.len(), 6);
        }
    }

    #[test]
    fn emission_with_junctions() {
        let rooms = vec![
            room(0, 0, 0, 64, 64, 128),
            room(160, 0, 0, 64, 64, 128),
            room(0, 160, 0, 64, 64, 128),
        ];
        let layout = LayoutIntent {
            rooms,
            edges: vec![(0, 1), (0, 2)],
            loop_count: 0,
        };
        // Two corridors meeting at a junction
        let corr_h = corridor_h(64, 32, 0, 96);
        let corr_v = Corridor {
            start: (32, 64, 0),
            end: (32, 160, 0),
            width: 64,
            height: 80,
        };
        let routed = RoutedIntent {
            corridors: vec![corr_h, corr_v],
            junctions: vec![
                Junction {
                    position: (64, 32, 0),
                },
                Junction {
                    position: (32, 64, 0),
                },
            ],
        };
        let emission = build_emission(&layout, &routed);
        // All brushes must have 6 faces
        for b in &emission.brushes {
            assert_eq!(b.faces.len(), 6);
        }
    }
}
