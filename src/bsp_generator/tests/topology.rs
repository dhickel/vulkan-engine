//! Integration tests for topology construction.
//!
//! Validates that [`bsp_generator::build_topology`] produces a connected
//! graph with exactly the requested number of cycles, that edge indices are
//! valid, and that results are deterministic.

use bsp_generator::{
    build_topology, geometry, place_rooms, DungeonConfig, MapClass, RoomIntent, Seed, StageRng,
};

fn make_rng(seed_val: u64, tag: &str) -> StageRng {
    Seed::new(seed_val).stage_seed(tag).rng()
}

fn place_and_build(
    seed: u64,
    room_count: u32,
    loop_count: u32,
    class: MapClass,
) -> (bsp_generator::LayoutIntent, bsp_generator::ValidatedConfig) {
    let (xy, z, candidates, attempts, astar) = match class {
        MapClass::M1 => ((1536, 1536), 256u32, 16u32, 64u32, 131_072u32),
        MapClass::M2 => ((3072, 3072), 384, 32, 96, 524_288),
    };
    let cfg = DungeonConfig {
        class,
        room_count,
        loop_count,
        xy_bounds: xy,
        z_span: z,
        placement_candidates: candidates,
        max_placement_attempts: attempts,
        max_astar_expansions: astar,
    }
    .validate()
    .unwrap();

    let rooms = place_rooms(&cfg, &mut make_rng(seed, "room-placement")).unwrap();
    let layout = build_topology(rooms, &cfg, &mut make_rng(seed, "corridor-routing")).unwrap();
    (layout, cfg)
}

// ── Connectedness guarantee ────────────────────────────────────────────────

#[test]
fn all_rooms_reachable_m1() {
    let (layout, _) = place_and_build(42, 12, 1, MapClass::M1);
    assert!(geometry::validate_connectedness(
        &layout.edges,
        layout.rooms.len()
    ));
}

#[test]
fn all_rooms_reachable_m2() {
    let (layout, _) = place_and_build(255, 28, 3, MapClass::M2);
    assert!(geometry::validate_connectedness(
        &layout.edges,
        layout.rooms.len()
    ));
}

#[test]
fn all_rooms_reachable_with_loops() {
    // M1 with max loops
    let (layout, _) = place_and_build(13, 16, 2, MapClass::M1);
    assert!(geometry::validate_connectedness(
        &layout.edges,
        layout.rooms.len()
    ));
}

#[test]
fn all_rooms_reachable_m2_max_loops() {
    let (layout, _) = place_and_build(100, 35, 6, MapClass::M2);
    assert!(geometry::validate_connectedness(
        &layout.edges,
        layout.rooms.len()
    ));
}

// ── Exact loop count ───────────────────────────────────────────────────────

#[test]
fn loop_count_0_produces_mst_only() {
    let (layout, _) = place_and_build(10, 12, 0, MapClass::M1);
    let n = layout.rooms.len();
    assert_eq!(layout.edges.len(), n - 1);
    assert_eq!(layout.loop_count, 0);
    geometry::validate_cycle_count(&layout.edges, n, 0).unwrap();
}

#[test]
fn loop_count_1_produces_mst_plus_1() {
    let (layout, _) = place_and_build(5, 10, 1, MapClass::M1);
    let n = layout.rooms.len();
    assert_eq!(layout.edges.len(), (n - 1) + 1);
    assert_eq!(layout.loop_count, 1);
    geometry::validate_cycle_count(&layout.edges, n, 1).unwrap();
}

#[test]
fn loop_count_2_produces_mst_plus_2() {
    let (layout, _) = place_and_build(7, 16, 2, MapClass::M1);
    let n = layout.rooms.len();
    assert_eq!(layout.edges.len(), (n - 1) + 2);
    assert_eq!(layout.loop_count, 2);
    geometry::validate_cycle_count(&layout.edges, n, 2).unwrap();
}

#[test]
fn loop_count_6_m2() {
    let (layout, _) = place_and_build(45, 40, 6, MapClass::M2);
    let n = layout.rooms.len();
    assert_eq!(layout.edges.len(), (n - 1) + 6);
    assert_eq!(layout.loop_count, 6);
    geometry::validate_cycle_count(&layout.edges, n, 6).unwrap();
}

// ── Minimal graph ──────────────────────────────────────────────────────────

#[test]
fn two_rooms_zero_loops_has_exactly_one_edge() {
    // Manually construct rooms since M1 config requires 8+ rooms
    let rooms = vec![
        RoomIntent {
            position: (0, 0, 0),
            dimensions: (64, 64, 128),
        },
        RoomIntent {
            position: (80, 0, 0),
            dimensions: (64, 64, 128),
        },
    ];
    let cfg = DungeonConfig {
        class: MapClass::M1,
        room_count: 8,
        loop_count: 0, // zero loops for minimal test
        ..DungeonConfig::nominal_m1()
    }
    .validate()
    .unwrap();
    let mut rng = make_rng(99, "corridor-routing");
    let layout = build_topology(rooms, &cfg, &mut rng).unwrap();
    assert_eq!(layout.edges.len(), 1);
    assert_eq!(layout.edges[0], (0, 1));
    assert!(geometry::validate_connectedness(&layout.edges, 2));
}

#[test]
fn three_rooms_zero_loops_has_two_edges() {
    let rooms = vec![
        RoomIntent {
            position: (0, 0, 0),
            dimensions: (64, 64, 128),
        },
        RoomIntent {
            position: (80, 0, 0),
            dimensions: (64, 64, 128),
        },
        RoomIntent {
            position: (0, 80, 0),
            dimensions: (64, 64, 128),
        },
    ];
    let cfg = DungeonConfig {
        class: MapClass::M1,
        room_count: 8,
        loop_count: 0, // zero loops for minimal test
        ..DungeonConfig::nominal_m1()
    }
    .validate()
    .unwrap();
    let mut rng = make_rng(33, "corridor-routing");
    let layout = build_topology(rooms, &cfg, &mut rng).unwrap();
    assert_eq!(layout.edges.len(), 2);
    assert!(geometry::validate_connectedness(&layout.edges, 3));
}

// ── Edge validity ──────────────────────────────────────────────────────────

#[test]
fn all_edge_indices_are_valid_m1() {
    let (layout, _) = place_and_build(1, 12, 1, MapClass::M1);
    let n = layout.rooms.len();
    for &(a, b) in &layout.edges {
        assert!(a < n);
        assert!(b < n);
        assert_ne!(a, b);
    }
}

#[test]
fn all_edge_indices_are_valid_m2() {
    let (layout, _) = place_and_build(17, 28, 3, MapClass::M2);
    let n = layout.rooms.len();
    for &(a, b) in &layout.edges {
        assert!(a < n);
        assert!(b < n);
        assert_ne!(a, b);
    }
}

// ── Determinism ────────────────────────────────────────────────────────────

#[test]
fn deterministic_topology_from_same_seed() {
    let seed = 42;
    let cfg = DungeonConfig::nominal_m1().validate().unwrap();
    let rooms_a = place_rooms(&cfg, &mut make_rng(seed, "room-placement")).unwrap();
    let rooms_b = rooms_a.clone();

    let layout_a = build_topology(rooms_a, &cfg, &mut make_rng(seed, "corridor-routing")).unwrap();
    let layout_b = build_topology(rooms_b, &cfg, &mut make_rng(seed, "corridor-routing")).unwrap();

    assert_eq!(layout_a.edges, layout_b.edges);
    assert_eq!(layout_a.loop_count, layout_b.loop_count);
    assert_eq!(layout_a.rooms.len(), layout_b.rooms.len());
}

// ── Room preservation ──────────────────────────────────────────────────────

#[test]
fn layout_preserves_all_rooms() {
    let (layout, cfg) = place_and_build(77, 12, 2, MapClass::M1);
    assert_eq!(layout.rooms.len(), cfg.room_count as usize);
}

// ── Multiple seeds produce valid topology ──────────────────────────────────

#[test]
fn multiple_seeds_produce_valid_topology() {
    for seed in [0u64, 1, 2, 3, 42, 99, 255, 1024, u64::MAX] {
        let (layout, _) = place_and_build(seed, 12, 1, MapClass::M1);
        let n = layout.rooms.len();
        assert!(
            geometry::validate_connectedness(&layout.edges, n),
            "seed {} produced disconnected graph",
            seed
        );
        geometry::validate_cycle_count(&layout.edges, n, 1).unwrap_or_else(|e| {
            panic!("seed {} cycle count invalid: {}", seed, e);
        });
    }
}
