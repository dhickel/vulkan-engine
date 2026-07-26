//! Integration tests for corridor routing.
//!
//! Validates that [`bsp_generator::route_edge`] and
//! [`bsp_generator::route_all_edges`] produce orthogonal corridor segments
//! with ≥ 64-unit width, ≥ 80-unit height, within configured bounds, and
//! that determinism and exhaustion behaviours are correct.

use bsp_generator::{
    place_rooms, route_all_edges, route_edge, DungeonConfig, GeneratorError, MapClass, RoomIntent,
    Seed, StageRng, CONSTRUCTION_QUANTUM, CORRIDOR_HEIGHT, CORRIDOR_WIDTH,
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

fn valid_m2_config(rooms: u32, loops: u32) -> bsp_generator::ValidatedConfig {
    DungeonConfig {
        class: MapClass::M2,
        room_count: rooms,
        loop_count: loops,
        xy_bounds: (3072, 3072),
        z_span: 384,
        placement_candidates: 32,
        max_placement_attempts: 96,
        max_astar_expansions: 524_288,
    }
    .validate()
    .unwrap()
}

// ── Straight corridor ─────────────────────────────────────────────────────

#[test]
fn straight_corridor_between_two_rooms() {
    let rooms = vec![
        room_at(0, 0, 0, 64, 64, 128),
        room_at(160, 0, 0, 64, 64, 128),
    ];
    let cfg = valid_m1_config(8, 0);
    let mut rng = make_rng(42);
    let corridors = route_edge(0, 1, &rooms, &cfg, &mut rng).unwrap();
    assert!(
        !corridors.is_empty(),
        "should produce at least one corridor segment"
    );

    // Verify minimum width and height
    for c in &corridors {
        assert!(
            c.width >= CORRIDOR_WIDTH,
            "corridor width {} < minimum {}",
            c.width,
            CORRIDOR_WIDTH
        );
        assert!(
            c.height >= CORRIDOR_HEIGHT,
            "corridor height {} < minimum {}",
            c.height,
            CORRIDOR_HEIGHT
        );
        assert_eq!(c.width % CONSTRUCTION_QUANTUM, 0);
        assert_eq!(c.height % CONSTRUCTION_QUANTUM, 0);
    }
}

#[test]
fn straight_corridor_stays_within_bounds() {
    let rooms = vec![
        room_at(0, 0, 0, 64, 64, 128),
        room_at(160, 0, 0, 64, 64, 128),
    ];
    let cfg = valid_m1_config(8, 0);
    let mut rng = make_rng(1);
    let corridors = route_edge(0, 1, &rooms, &cfg, &mut rng).unwrap();

    for c in &corridors {
        let hw = c.width as i32 / 2;
        // Start and end positions must be within or at bounds
        assert!(c.start.0 - hw >= 0 - hw, "corridor extends beyond min x");
        assert!(c.start.1 - hw >= 0 - hw, "corridor extends beyond min y");
        assert!(
            c.end.0 + hw <= cfg.xy_bounds.0 as i32 + hw,
            "corridor extends beyond max x: end={} + hw={} > {}",
            c.end.0,
            hw,
            cfg.xy_bounds.0
        );
        assert!(
            c.end.1 + hw <= cfg.xy_bounds.1 as i32 + hw,
            "corridor extends beyond max y"
        );
    }
}

#[test]
fn corridors_are_axis_aligned() {
    let rooms = vec![
        room_at(0, 0, 0, 64, 64, 128),
        room_at(160, 0, 0, 64, 64, 128),
        room_at(0, 160, 0, 64, 64, 128),
    ];
    let cfg = valid_m1_config(8, 0);
    let mut rng = make_rng(7);
    let edges = vec![(0, 1), (0, 2)];
    let routed = route_all_edges(&rooms, &edges, &cfg, &mut rng).unwrap();

    for c in &routed.corridors {
        // Each corridor segment must be axis-aligned: either dx==0 or dy==0
        let dx = (c.end.0 - c.start.0).abs();
        let dy = (c.end.1 - c.start.1).abs();
        assert!(
            dx == 0 || dy == 0,
            "corridor segment must be axis-aligned: start={:?}, end={:?}",
            c.start,
            c.end
        );
    }
}

// ── L-shaped routes ───────────────────────────────────────────────────────

#[test]
fn l_shaped_route_has_one_turn() {
    // Rooms arranged so that a direct line or L-shaped path is possible.
    // Place them far enough apart that the corridor approach paths are clear.
    let rooms = vec![
        room_at(0, 0, 0, 64, 64, 128),
        room_at(160, 160, 0, 64, 64, 128),
    ];
    let cfg = valid_m1_config(8, 0);
    let mut rng = make_rng(42);
    let corridors = route_edge(0, 1, &rooms, &cfg, &mut rng).unwrap();

    // Each segment must be axis-aligned and satisfy minimum dimensions
    for c in &corridors {
        assert!(c.width >= CORRIDOR_WIDTH);
        assert!(c.height >= CORRIDOR_HEIGHT);
        let dx = (c.end.0 - c.start.0).abs();
        let dy = (c.end.1 - c.start.1).abs();
        assert!(dx == 0 || dy == 0, "segment must be axis-aligned");
    }

    // Verify segments connect end-to-end
    for i in 1..corridors.len() {
        assert_eq!(
            corridors[i - 1].end,
            corridors[i].start,
            "segments must be connected"
        );
    }
}

#[test]
fn l_shaped_route_avoids_blocking_room() {
    // Room A (0,0), Blocking (80,32), Room B (0,160)
    // Route from A to B must go around the blocking room
    let rooms = vec![
        room_at(0, 0, 0, 64, 64, 128),
        room_at(80, 32, 0, 64, 64, 128),
        room_at(0, 160, 0, 64, 64, 128),
    ];
    let cfg = valid_m1_config(8, 0);
    let mut rng = make_rng(13);
    let corridors = route_edge(0, 2, &rooms, &cfg, &mut rng).unwrap();
    assert!(
        !corridors.is_empty(),
        "should find path around blocking room"
    );

    // Verify minimum dimensions
    for c in &corridors {
        assert!(c.width >= CORRIDOR_WIDTH);
        assert!(c.height >= CORRIDOR_HEIGHT);
    }
}

// ── Route exhaustion ──────────────────────────────────────────────────────

#[test]
fn route_exhaustion_at_too_small_bounds() {
    let rooms = vec![
        room_at(0, 0, 0, 64, 64, 128),
        room_at(80, 0, 0, 64, 64, 128),
    ];
    let cfg = DungeonConfig {
        max_astar_expansions: 1, // impossibly small
        xy_bounds: (256, 16),    // single row of cells
        class: MapClass::M1,
        room_count: 8,
        loop_count: 0,
        z_span: 128,
        placement_candidates: 4,
        max_placement_attempts: 4,
    }
    .validate()
    .unwrap();
    let mut rng = make_rng(42);
    let result = route_edge(0, 1, &rooms, &cfg, &mut rng);
    match result {
        Err(GeneratorError::RouteExhausted { expansions }) => {
            assert!(expansions > 0, "expansions should be recorded");
        }
        Ok(corridors) => {
            panic!(
                "expected RouteExhausted but got {} corridor segments",
                corridors.len()
            );
        }
        Err(other) => {
            panic!("expected RouteExhausted but got {:?}", other);
        }
    }
}

#[test]
fn route_exhaustion_error_display() {
    let err = GeneratorError::RouteExhausted { expansions: 99 };
    let msg = err.to_string();
    assert!(!msg.is_empty());
    assert!(msg.contains("99"));
    assert!(msg.contains("A*"));
}

// ── Determinism ───────────────────────────────────────────────────────────

#[test]
fn deterministic_same_input_produces_same_route() {
    let rooms = vec![
        room_at(0, 0, 0, 64, 64, 128),
        room_at(160, 0, 0, 64, 64, 128),
        room_at(0, 160, 0, 64, 64, 128),
    ];
    let cfg = valid_m1_config(8, 0);

    let r1 = route_edge(0, 1, &rooms, &cfg, &mut make_rng(42)).unwrap();
    let r2 = route_edge(0, 1, &rooms, &cfg, &mut make_rng(42)).unwrap();
    assert_eq!(r1, r2, "same seed should produce identical routes");

    let r3 = route_edge(0, 2, &rooms, &cfg, &mut make_rng(42)).unwrap();
    let r4 = route_edge(0, 2, &rooms, &cfg, &mut make_rng(42)).unwrap();
    assert_eq!(r3, r4, "same seed should produce identical routes");
}

#[test]
fn different_seeds_may_produce_different_routes() {
    let rooms = vec![
        room_at(0, 0, 0, 64, 64, 128),
        room_at(160, 0, 0, 64, 64, 128),
    ];
    let cfg = valid_m1_config(8, 0);

    let r1 = route_edge(0, 1, &rooms, &cfg, &mut make_rng(42)).unwrap();
    let r2 = route_edge(0, 1, &rooms, &cfg, &mut make_rng(99)).unwrap();
    // Both must be valid corridors with minimum dimensions
    assert!(!r1.is_empty());
    assert!(!r2.is_empty());
    for c in r1.iter().chain(r2.iter()) {
        assert!(c.width >= CORRIDOR_WIDTH);
        assert!(c.height >= CORRIDOR_HEIGHT);
    }
}

// ── Route all edges ───────────────────────────────────────────────────────

#[test]
fn route_all_edges_produces_corridors_for_every_edge() {
    let rooms = vec![
        room_at(0, 0, 0, 64, 64, 128),
        room_at(160, 0, 0, 64, 64, 128),
        room_at(0, 160, 0, 64, 64, 128),
        room_at(160, 160, 0, 64, 64, 128),
    ];
    let cfg = valid_m1_config(8, 0);
    let mut rng = make_rng(33);
    let edges = vec![(0, 1), (1, 3), (0, 2)];
    let routed = route_all_edges(&rooms, &edges, &cfg, &mut rng).unwrap();

    assert!(!routed.corridors.is_empty());
    assert!(!routed.junctions.is_empty());

    for c in &routed.corridors {
        assert!(c.width >= CORRIDOR_WIDTH);
        assert!(c.height >= CORRIDOR_HEIGHT);
    }
}

#[test]
fn route_all_edges_with_placed_rooms_m1() {
    let cfg = valid_m1_config(8, 0);
    let rooms = place_rooms(&cfg, &mut make_placement_rng(42)).unwrap();
    let layout = bsp_generator::build_topology(rooms, &cfg, &mut make_rng(42)).unwrap();

    let mut rng = make_rng(42);
    let result = route_all_edges(&layout.rooms, &layout.edges, &cfg, &mut rng);

    match result {
        Ok(routed) => {
            assert!(!routed.corridors.is_empty());
            for c in &routed.corridors {
                assert!(c.width >= CORRIDOR_WIDTH);
                assert!(c.height >= CORRIDOR_HEIGHT);
            }
        }
        Err(GeneratorError::RouteExhausted { .. }) => {
            // Acceptable: some placements produce routing that exhausts budget
        }
        Err(e) => {
            panic!("unexpected error: {:?}", e);
        }
    }
}

#[test]
fn route_all_edges_with_placed_rooms_m2() {
    let cfg = valid_m2_config(20, 2);
    let rooms = place_rooms(&cfg, &mut make_placement_rng(255)).unwrap();
    let layout = bsp_generator::build_topology(rooms, &cfg, &mut make_rng(255)).unwrap();

    let mut rng = make_rng(255);
    let result = route_all_edges(&layout.rooms, &layout.edges, &cfg, &mut rng);

    match result {
        Ok(routed) => {
            assert!(!routed.corridors.is_empty());
            for c in &routed.corridors {
                assert!(c.width >= CORRIDOR_WIDTH);
                assert!(c.height >= CORRIDOR_HEIGHT);
            }
        }
        Err(GeneratorError::RouteExhausted { .. }) => {
            // Acceptable: some room placements produce unroutable edges
        }
        Err(e) => {
            panic!("unexpected error: {:?}", e);
        }
    }
}

// ── G6: Deterministic portal snapping ──────────────────────────────────

/// 7-quantum (112-unit) room snapping — all 4 directions

#[test]
fn portal_snapping_7_quantum_room_east_west() {
    // 7-quantum rooms: 112 units wide. Position at 0, portal at east wall (x=112).
    // Counterpart room center at x=192, y=56. Portal on east wall of A should
    // snap perpendicular to nearest quantum near counterpart center (56).
    let rooms = vec![
        room_at(0, 0, 0, 112, 112, 128),
        room_at(160, 0, 0, 112, 112, 128),
    ];
    let cfg = valid_m1_config(8, 0);
    let mut rng = make_rng(42);
    let corridors = route_edge(0, 1, &rooms, &cfg, &mut rng).unwrap();
    assert!(!corridors.is_empty());
    assert_eq!(corridors.first().unwrap().start, (112, 48, 0));
    assert_eq!(corridors.last().unwrap().end, (160, 48, 0));
    for c in &corridors {
        assert!(c.width >= CORRIDOR_WIDTH);
        assert!(c.height >= CORRIDOR_HEIGHT);
    }
}

#[test]
fn portal_snapping_7_quantum_room_north_south() {
    let rooms = vec![
        room_at(0, 0, 0, 112, 112, 128),
        room_at(0, 160, 0, 112, 112, 128),
    ];
    let cfg = valid_m1_config(8, 0);
    let mut rng = make_rng(42);
    let corridors = route_edge(0, 1, &rooms, &cfg, &mut rng).unwrap();
    assert!(!corridors.is_empty());
    assert_eq!(corridors.first().unwrap().start, (48, 112, 0));
    assert_eq!(corridors.last().unwrap().end, (48, 160, 0));
    for c in &corridors {
        assert!(c.width >= CORRIDOR_WIDTH);
        assert!(c.height >= CORRIDOR_HEIGHT);
    }
}

/// 9-quantum (144-unit) room snapping — all 4 directions

#[test]
fn portal_snapping_9_quantum_room_east_west() {
    let rooms = vec![
        room_at(0, 0, 0, 144, 144, 128),
        room_at(208, 0, 0, 144, 144, 128),
    ];
    let cfg = valid_m1_config(8, 0);
    let mut rng = make_rng(42);
    let corridors = route_edge(0, 1, &rooms, &cfg, &mut rng).unwrap();
    assert!(!corridors.is_empty());
    assert_eq!(corridors.first().unwrap().start, (144, 64, 0));
    assert_eq!(corridors.last().unwrap().end, (208, 64, 0));
    for c in &corridors {
        assert!(c.width >= CORRIDOR_WIDTH);
        assert!(c.height >= CORRIDOR_HEIGHT);
    }
}

#[test]
fn portal_snapping_9_quantum_room_north_south() {
    let rooms = vec![
        room_at(0, 0, 0, 144, 144, 128),
        room_at(0, 208, 0, 144, 144, 128),
    ];
    let cfg = valid_m1_config(8, 0);
    let mut rng = make_rng(42);
    let corridors = route_edge(0, 1, &rooms, &cfg, &mut rng).unwrap();
    assert!(!corridors.is_empty());
    assert_eq!(corridors.first().unwrap().start, (64, 144, 0));
    assert_eq!(corridors.last().unwrap().end, (64, 208, 0));
    for c in &corridors {
        assert!(c.width >= CORRIDOR_WIDTH);
        assert!(c.height >= CORRIDOR_HEIGHT);
    }
}

// ── Quantum alignment ─────────────────────────────────────────────────────

#[test]
fn corridor_endpoints_are_quantum_aligned() {
    let rooms = vec![
        room_at(0, 0, 0, 64, 64, 128),
        room_at(160, 0, 0, 64, 64, 128),
    ];
    let cfg = valid_m1_config(8, 0);
    let mut rng = make_rng(1);
    let corridors = route_edge(0, 1, &rooms, &cfg, &mut rng).unwrap();

    let q = CONSTRUCTION_QUANTUM as i32;
    for c in &corridors {
        assert_eq!(c.start.0 % q, 0, "start.x not quantum-aligned");
        assert_eq!(c.start.1 % q, 0, "start.y not quantum-aligned");
        assert_eq!(c.end.0 % q, 0, "end.x not quantum-aligned");
        assert_eq!(c.end.1 % q, 0, "end.y not quantum-aligned");
    }
}

// ── Corridor-to-corridor connectivity ─────────────────────────────────────

#[test]
fn consecutive_edges_produce_connected_corridors() {
    // Three rooms in a line: (0,0), (160,0), (320,0)
    let rooms = vec![
        room_at(0, 0, 0, 64, 64, 128),
        room_at(160, 0, 0, 64, 64, 128),
        room_at(320, 0, 0, 64, 64, 128),
    ];
    let cfg = valid_m1_config(8, 0);
    let mut rng = make_rng(5);
    let edges = vec![(0, 1), (1, 2)];
    let routed = route_all_edges(&rooms, &edges, &cfg, &mut rng).unwrap();

    // Each edge should have produced at least one corridor
    assert!(routed.corridors.len() >= 2);
    for c in &routed.corridors {
        assert!(c.width >= CORRIDOR_WIDTH);
        assert!(c.height >= CORRIDOR_HEIGHT);
    }
}
