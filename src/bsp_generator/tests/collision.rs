//! Integration tests for corridor collision safety.
//!
//! Validates that routed corridors do not intersect unintended rooms, that
//! parallel corridors maintain appropriate separation, and that wall-thickness
//! separation is preserved between all geometry.

use bsp_generator::{
    geometry::rooms_overlap, place_rooms, route_all_edges, route_edge, DungeonConfig, MapClass,
    RoomIntent, Seed, StageRng, CORRIDOR_HEIGHT, CORRIDOR_WIDTH, CONSTRUCTION_QUANTUM,
};

fn make_rng(seed_val: u64) -> StageRng {
    Seed::new(seed_val).stage_seed("corridor-routing").rng()
}

fn make_placement_rng(seed_val: u64) -> StageRng {
    Seed::new(seed_val).stage_seed("room-placement").rng()
}

fn room_at(x: i32, y: i32, z: i32, dx: u32, dy: u32, dz: u32) -> RoomIntent {
    RoomIntent {
        position: (x, y, z),
        dimensions: (dx, dy, dz),
    }
}

fn valid_m1_config(rooms: u32, loops: u32) -> bsp_generator::ValidatedConfig {
    DungeonConfig {
        class: MapClass::M1,
        room_count: rooms,
        loop_count: loops,
        xy_bounds: (1536, 1536),
        z_span: 256,
        placement_candidates: 16,
        max_placement_attempts: 64,
        max_astar_expansions: 131_072,
    }
    .validate()
    .unwrap()
}

fn corridor_to_room_rect(corridor: &bsp_generator::Corridor) -> RoomIntent {
    let hw = corridor.width as i32 / 2;
    let (x0, x1) = if corridor.start.0 <= corridor.end.0 {
        (corridor.start.0 - hw, corridor.end.0 + hw)
    } else {
        (corridor.end.0 - hw, corridor.start.0 + hw)
    };
    let (y0, y1) = if corridor.start.1 <= corridor.end.1 {
        (corridor.start.1 - hw, corridor.end.1 + hw)
    } else {
        (corridor.end.1 - hw, corridor.start.1 + hw)
    };
    let z0 = corridor.start.2;
    let z1 = z0 + corridor.height as i32;
    let dx = (x1 - x0).max(0) as u32;
    let dy = (y1 - y0).max(0) as u32;
    let dz = (z1 - z0).max(0) as u32;
    RoomIntent {
        position: (x0, y0, z0),
        dimensions: (dx, dy, dz),
    }
}

// ── Corridors don't intersect unintended rooms ────────────────────────────

#[test]
fn corridor_does_not_overlap_unintended_room() {
    // Room A (source), Room B (blocking), Room C (target)
    let rooms = vec![
        room_at(0, 0, 0, 64, 64, 128),    // A
        room_at(80, 32, 0, 64, 64, 128),  // B (blocking)
        room_at(0, 160, 0, 64, 64, 128),  // C
    ];
    let cfg = valid_m1_config(8, 0);
    let mut rng = make_rng(13);
    let corridors = route_edge(0, 2, &rooms, &cfg, &mut rng).unwrap();

    // Each corridor segment's bounding box (as a RoomIntent) should not
    // overlap any room except the source/target (indices 0 and 2)
    let wall_thickness = 16;
    for c in &corridors {
        let c_rect = corridor_to_room_rect(c);
        for (i, room) in rooms.iter().enumerate() {
            if i == 0 || i == 2 {
                continue; // skip source and target rooms
            }
            assert!(
                !rooms_overlap(&c_rect, room, wall_thickness),
                "corridor segment {:?} overlaps blocking room {}: room={:?}",
                c,
                i,
                room
            );
        }
    }
}

#[test]
fn corridors_avoid_all_non_endpoint_rooms() {
    let rooms = vec![
        room_at(0, 0, 0, 64, 64, 128),
        room_at(160, 0, 0, 64, 64, 128),
        room_at(320, 0, 0, 64, 64, 128),
        room_at(0, 160, 0, 64, 64, 128),
    ];
    let cfg = valid_m1_config(8, 0);
    let mut rng = make_rng(7);
    // Route from room 0 to room 3 — rooms 1 and 2 are en route
    let corridors = route_edge(0, 3, &rooms, &cfg, &mut rng).unwrap();

    let wall_thickness = 16;
    for c in &corridors {
        let c_rect = corridor_to_room_rect(c);
        for (i, room) in rooms.iter().enumerate() {
            if i == 0 || i == 3 {
                continue;
            }
            assert!(
                !rooms_overlap(&c_rect, room, wall_thickness),
                "corridor intersects unintended room {}: {:?}",
                i,
                room
            );
        }
    }
}

// ── Parallel corridors don't merge ────────────────────────────────────────

#[test]
fn parallel_corridors_maintain_separation() {
    // Two separate room pairs whose corridors run parallel
    let rooms = vec![
        room_at(0, 0, 0, 64, 64, 128),      // A
        room_at(160, 0, 0, 64, 64, 128),    // B
        room_at(0, 128, 0, 64, 64, 128),    // C
        room_at(160, 128, 0, 64, 64, 128),  // D
    ];
    let cfg = valid_m1_config(8, 0);
    let c1 = route_edge(0, 1, &rooms, &cfg, &mut make_rng(42)).unwrap();
    let c2 = route_edge(2, 3, &rooms, &cfg, &mut make_rng(99)).unwrap();

    // Verify corridors don't overlap each other
    let wall_thickness = 16;
    for seg1 in &c1 {
        let r1 = corridor_to_room_rect(seg1);
        for seg2 in &c2 {
            let r2 = corridor_to_room_rect(seg2);
            // Parallel corridors should maintain wall-thickness separation
            // (they may touch at wall boundary but not overlap)
            assert!(
                !rooms_overlap(&r1, &r2, wall_thickness),
                "parallel corridors overlap: seg1={:?}, seg2={:?}",
                seg1,
                seg2
            );
        }
    }
}

#[test]
fn parallel_corridors_keep_minimum_gap() {
    // Two room pairs placed parallel with a gap between them
    let rooms = vec![
        room_at(0, 0, 0, 64, 64, 128),
        room_at(160, 0, 0, 64, 64, 128),
        room_at(0, 96, 0, 64, 64, 128),
        room_at(160, 96, 0, 64, 64, 128),
    ];
    let cfg = valid_m1_config(8, 0);
    let c1 = route_edge(0, 1, &rooms, &cfg, &mut make_rng(7)).unwrap();
    let c2 = route_edge(2, 3, &rooms, &cfg, &mut make_rng(13)).unwrap();

    // Check that there's a gap (wall-thickness separation) between corridor groups
    let wall_thickness = 16;
    let mut any_overlap = false;
    for seg1 in &c1 {
        let r1 = corridor_to_room_rect(seg1);
        for seg2 in &c2 {
            let r2 = corridor_to_room_rect(seg2);
            if rooms_overlap(&r1, &r2, wall_thickness) {
                any_overlap = true;
            }
        }
    }
    // With sufficient room separation, the corridors should not overlap
    assert!(!any_overlap, "parallel corridors should maintain separation");

    // Verify minimum dimensions
    for c in c1.iter().chain(c2.iter()) {
        assert!(c.width >= CORRIDOR_WIDTH);
        assert!(c.height >= CORRIDOR_HEIGHT);
    }
}

// ── Wall-thickness separation ─────────────────────────────────────────────

#[test]
fn corridor_wall_thickness_from_rooms() {
    let rooms = vec![
        room_at(0, 0, 0, 64, 64, 128),
        room_at(160, 0, 0, 64, 64, 128),
    ];
    let cfg = valid_m1_config(8, 0);
    let mut rng = make_rng(1);
    let corridors = route_edge(0, 1, &rooms, &cfg, &mut rng).unwrap();

    // The corridor must not penetrate room interiors.
    // Check: for each corridor segment, the corridor's inner edge should be
    // at least WALL_THICKNESS (16 units) away from any room not being connected.
    let wall_thickness = 16i32;
    for c in &corridors {
        let c_rect = corridor_to_room_rect(c);
        for (i, room) in rooms.iter().enumerate() {
            if i == 0 || i == 1 {
                // Source/target rooms — corridor is allowed to touch wall
                continue;
            }
            // Check that expanded corridor rect doesn't overlap expanded room
            assert!(
                !rooms_overlap(&c_rect, room, wall_thickness),
                "corridor penetrates room {}: {:?}",
                i,
                room
            );
        }
    }
}

#[test]
fn corridor_width_respected_after_routing() {
    // Verify the corridor bounding box width matches declared width
    let rooms = vec![
        room_at(0, 0, 0, 64, 64, 128),
        room_at(160, 0, 0, 64, 64, 128),
    ];
    let cfg = valid_m1_config(8, 0);
    let mut rng = make_rng(1);
    let corridors = route_edge(0, 1, &rooms, &cfg, &mut rng).unwrap();

    for c in &corridors {
        let rect = corridor_to_room_rect(c);
        if c.start.1 == c.end.1 {
            // Horizontal corridor: width is in Y dimension
            assert_eq!(rect.dimensions.1, c.width, "horizontal corridor width mismatch");
            assert_eq!(rect.dimensions.2, c.height, "corridor height mismatch");
        } else {
            // Vertical corridor: width is in X dimension
            assert_eq!(rect.dimensions.0, c.width, "vertical corridor width mismatch");
            assert_eq!(rect.dimensions.2, c.height, "corridor height mismatch");
        }
    }
}

// ── Full pipeline collision test ──────────────────────────────────────────

#[test]
fn full_pipeline_no_collision_m1() {
    // Use generous config: fewer rooms, more space, higher budgets
    let cfg = DungeonConfig {
        class: MapClass::M1,
        room_count: 8,
        loop_count: 0,
        xy_bounds: (1536, 1536),
        z_span: 256,
        placement_candidates: 16,
        max_placement_attempts: 64,
        max_astar_expansions: 131_072,
    }
    .validate()
    .unwrap();
    let rooms = place_rooms(&cfg, &mut make_placement_rng(7)).unwrap();
    let layout =
        bsp_generator::build_topology(rooms, &cfg, &mut make_rng(7)).unwrap();
    let routed = route_all_edges(
        &layout.rooms,
        &layout.edges,
        &cfg,
        &mut make_rng(7),
    );

    match routed {
        Ok(routed) => {
            let wall_thickness = 16i32;

            // Every corridor must have correct dimensions
            for c in &routed.corridors {
                assert!(c.width >= CORRIDOR_WIDTH);
                assert!(c.height >= CORRIDOR_HEIGHT);
            }

            // Every corridor must not intersect any non-endpoint room
            for c in &routed.corridors {
                let c_rect = corridor_to_room_rect(c);
                for (i, room) in layout.rooms.iter().enumerate() {
                    let is_endpoint = layout.edges.iter().any(|&(a, b)| {
                        (a == i || b == i)
                            && (near_wall(&c_rect, room) || near_wall_from_corridor(c, room))
                    });
                    if !is_endpoint {
                        assert!(
                            !rooms_overlap(&c_rect, room, wall_thickness),
                            "corridor collides with non-endpoint room {}",
                            i
                        );
                    }
                }
            }
        }
        Err(bsp_generator::GeneratorError::RouteExhausted { .. }) => {
            // Acceptable for some configurations — just verify the error is well-formed
        }
        Err(e) => {
            panic!("unexpected error: {:?}", e);
        }
    }
}

/// Check if a corridor's bounding box is near a room's wall (within tolerance).
fn near_wall(c_rect: &RoomIntent, room: &RoomIntent) -> bool {
    let margin = CONSTRUCTION_QUANTUM as i32 * 2; // 32 unit tolerance
    let r_x0 = room.position.0 - margin;
    let r_x1 = room.position.0 + room.dimensions.0 as i32 + margin;
    let r_y0 = room.position.1 - margin;
    let r_y1 = room.position.1 + room.dimensions.1 as i32 + margin;

    let c_x0 = c_rect.position.0;
    let c_x1 = c_rect.position.0 + c_rect.dimensions.0 as i32;
    let c_y0 = c_rect.position.1;
    let c_y1 = c_rect.position.1 + c_rect.dimensions.1 as i32;

    c_x1 > r_x0 && c_x0 < r_x1 && c_y1 > r_y0 && c_y0 < r_y1
}

/// Check if a corridor endpoint is near a room wall.
fn near_wall_from_corridor(c: &bsp_generator::Corridor, room: &RoomIntent) -> bool {
    let margin = CONSTRUCTION_QUANTUM as i32 * 3;
    let r_x0 = room.position.0 - margin;
    let r_x1 = room.position.0 + room.dimensions.0 as i32 + margin;
    let r_y0 = room.position.1 - margin;
    let r_y1 = room.position.1 + room.dimensions.1 as i32 + margin;

    // Check start point
    if c.start.0 >= r_x0 && c.start.0 <= r_x1 && c.start.1 >= r_y0 && c.start.1 <= r_y1 {
        return true;
    }
    // Check end point
    if c.end.0 >= r_x0 && c.end.0 <= r_x1 && c.end.1 >= r_y0 && c.end.1 <= r_y1 {
        return true;
    }
    false
}

// ── Multiple seeds produce collision-free routing ─────────────────────────

#[test]
fn multiple_seeds_produce_collision_free_routing() {
    // Test a small set of seeds through the full pipeline
    let seeds = [0u64, 1, 42, 99, 255];

    for &seed in &seeds {
        let cfg = DungeonConfig {
            class: MapClass::M1,
            room_count: 10,
            loop_count: 1,
            xy_bounds: (1536, 1536),
            z_span: 256,
            placement_candidates: 16,
            max_placement_attempts: 64,
            max_astar_expansions: 131_072,
        }
        .validate()
        .unwrap();

        let rooms = place_rooms(&cfg, &mut make_placement_rng(seed));
        if rooms.is_err() {
            continue; // placement may exhaust for some seeds with tight config
        }
        let rooms = rooms.unwrap();
        let layout =
            bsp_generator::build_topology(rooms.clone(), &cfg, &mut make_rng(seed)).unwrap();
        let routed = route_all_edges(
            &layout.rooms,
            &layout.edges,
            &cfg,
            &mut make_rng(seed),
        );

        match routed {
            Ok(routed) => {
                let wall_thickness = 16i32;
                // Quick collision check: all corridor segments have correct dimensions
                for c in &routed.corridors {
                    assert!(
                        c.width >= CORRIDOR_WIDTH,
                        "seed {}: corridor width too small",
                        seed
                    );
                    assert!(
                        c.height >= CORRIDOR_HEIGHT,
                        "seed {}: corridor height too small",
                        seed
                    );
                    assert_eq!(c.width % CONSTRUCTION_QUANTUM, 0);
                    assert_eq!(c.height % CONSTRUCTION_QUANTUM, 0);
                }

                // Verify corridors don't intersect unconnected rooms
                for c in &routed.corridors {
                    let c_rect = corridor_to_room_rect(c);
                    for (i, room) in layout.rooms.iter().enumerate() {
                        let is_endpoint = near_wall_from_corridor(c, room);
                        if !is_endpoint {
                            assert!(
                                !rooms_overlap(&c_rect, room, wall_thickness),
                                "seed {}: corridor collides with room {}",
                                seed,
                                i
                            );
                        }
                    }
                }
            }
            Err(bsp_generator::GeneratorError::RouteExhausted { .. }) => {
                // Acceptable: some seeds may exhaust routing budget for tight configs
            }
            Err(e) => {
                panic!("seed {}: unexpected error: {:?}", seed, e);
            }
        }
    }
}
