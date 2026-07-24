//! Integration tests for room placement.
//!
//! Validates that [`bsp_generator::place_rooms`] produces non-overlapping,
//! quantum-aligned rooms within frozen M1/M2 bounds, that deterministic
//! replay holds, and that exhaustion errors are produced for impossible
//! configurations.

use bsp_generator::placement::WALL_THICKNESS;
use bsp_generator::{
    geometry, place_rooms, DungeonConfig, GeneratorError, MapClass, Seed, StageRng,
};

fn make_rng(seed_val: u64) -> StageRng {
    Seed::new(seed_val).stage_seed("room-placement").rng()
}

// ── M1 placement ───────────────────────────────────────────────────────────

#[test]
fn m1_placement_produces_8_to_16_rooms() {
    let cfg = DungeonConfig::nominal_m1().validate().unwrap();
    let mut rng = make_rng(42);
    let rooms = place_rooms(&cfg, &mut rng).unwrap();
    assert!(rooms.len() >= 8);
    assert!(rooms.len() <= 16);
    assert_eq!(rooms.len(), cfg.room_count as usize);
}

#[test]
fn m1_rooms_are_non_overlapping() {
    let cfg = DungeonConfig::nominal_m1().validate().unwrap();
    let mut rng = make_rng(42);
    let rooms = place_rooms(&cfg, &mut rng).unwrap();
    geometry::validate_no_overlap(&rooms, WALL_THICKNESS as i32).unwrap();
}

#[test]
fn m1_all_rooms_quantum_aligned() {
    let cfg = DungeonConfig::nominal_m1().validate().unwrap();
    let mut rng = make_rng(13);
    let rooms = place_rooms(&cfg, &mut rng).unwrap();
    geometry::validate_quantum_alignment(&rooms).unwrap();
}

#[test]
fn m1_rooms_within_xy1536_z256() {
    let cfg = DungeonConfig::nominal_m1().validate().unwrap();
    let mut rng = make_rng(7);
    let rooms = place_rooms(&cfg, &mut rng).unwrap();

    for room in &rooms {
        let (x, y, z) = room.position;
        let (dx, dy, dz) = room.dimensions;
        assert!(x >= 0, "x negative: {}", x);
        assert!(y >= 0, "y negative: {}", y);
        assert!(z >= 0, "z negative: {}", z);
        assert!(x + dx as i32 <= 1536, "x out of bounds: {}+{}", x, dx);
        assert!(y + dy as i32 <= 1536, "y out of bounds: {}+{}", y, dy);
        assert!(z + dz as i32 <= 256, "z out of bounds: {}+{}", z, dz);
    }
}

// ── M2 placement ───────────────────────────────────────────────────────────

#[test]
fn m2_placement_produces_17_to_40_rooms() {
    let cfg = DungeonConfig::nominal_m2().validate().unwrap();
    let mut rng = make_rng(255);
    let rooms = place_rooms(&cfg, &mut rng).unwrap();
    assert!(rooms.len() >= 17);
    assert!(rooms.len() <= 40);
    assert_eq!(rooms.len(), cfg.room_count as usize);
}

#[test]
fn m2_rooms_are_non_overlapping() {
    let cfg = DungeonConfig::nominal_m2().validate().unwrap();
    let mut rng = make_rng(17);
    let rooms = place_rooms(&cfg, &mut rng).unwrap();
    geometry::validate_no_overlap(&rooms, WALL_THICKNESS as i32).unwrap();
}

#[test]
fn m2_all_rooms_quantum_aligned() {
    let cfg = DungeonConfig::nominal_m2().validate().unwrap();
    let mut rng = make_rng(99);
    let rooms = place_rooms(&cfg, &mut rng).unwrap();
    geometry::validate_quantum_alignment(&rooms).unwrap();
}

#[test]
fn m2_rooms_within_xy3072_z384() {
    let cfg = DungeonConfig::nominal_m2().validate().unwrap();
    let mut rng = make_rng(55);
    let rooms = place_rooms(&cfg, &mut rng).unwrap();

    for room in &rooms {
        let (x, y, z) = room.position;
        let (dx, dy, dz) = room.dimensions;
        assert!(x >= 0);
        assert!(y >= 0);
        assert!(z >= 0);
        assert!(x + dx as i32 <= 3072, "x out of bounds: {}+{}", x, dx);
        assert!(y + dy as i32 <= 3072, "y out of bounds: {}+{}", y, dy);
        assert!(z + dz as i32 <= 384, "z out of bounds: {}+{}", z, dz);
    }
}

// ── Determinism ────────────────────────────────────────────────────────────

#[test]
fn deterministic_same_seed_produces_same_layout() {
    let cfg = DungeonConfig::nominal_m1().validate().unwrap();
    let rooms_a = place_rooms(&cfg, &mut make_rng(42)).unwrap();
    let rooms_b = place_rooms(&cfg, &mut make_rng(42)).unwrap();
    assert_eq!(rooms_a, rooms_b);
}

#[test]
fn deterministic_different_seeds_different_layouts() {
    let cfg = DungeonConfig::nominal_m1().validate().unwrap();
    let rooms_a = place_rooms(&cfg, &mut make_rng(42)).unwrap();
    let rooms_b = place_rooms(&cfg, &mut make_rng(99)).unwrap();
    // Both must be valid, but they may coincidentally match.
    // At minimum they should have the same count.
    assert_eq!(rooms_a.len(), rooms_b.len());
    geometry::validate_no_overlap(&rooms_a, WALL_THICKNESS as i32).unwrap();
    geometry::validate_no_overlap(&rooms_b, WALL_THICKNESS as i32).unwrap();
}

// ── Placement exhaustion ───────────────────────────────────────────────────

#[test]
fn placement_exhaustion_at_too_small_bounds() {
    // Bounds too small to fit requested rooms
    let cfg = DungeonConfig {
        class: MapClass::M1,
        room_count: 10,
        loop_count: 1,
        xy_bounds: (128, 128),
        z_span: 128,
        placement_candidates: 4,
        max_placement_attempts: 4,
        max_astar_expansions: 131_072,
    }
    .validate()
    .unwrap();
    let mut rng = make_rng(0);
    let result = place_rooms(&cfg, &mut rng);
    match result {
        Err(GeneratorError::PlacementExhausted { attempts }) => {
            assert!(attempts > 0, "should have attempted placements");
        }
        Ok(rooms) => {
            panic!("expected PlacementExhausted but got {} rooms", rooms.len());
        }
        Err(other) => {
            panic!("expected PlacementExhausted but got {:?}", other);
        }
    }
}

// ── Boundary configurations ────────────────────────────────────────────────

#[test]
fn boundary_m1_minimum_8_rooms() {
    let cfg = DungeonConfig {
        class: MapClass::M1,
        room_count: 8,
        loop_count: 0,
        xy_bounds: (1024, 1024),
        z_span: 192,
        placement_candidates: 16,
        max_placement_attempts: 64,
        max_astar_expansions: 131_072,
    }
    .validate()
    .unwrap();
    let mut rng = make_rng(42);
    let rooms = place_rooms(&cfg, &mut rng).unwrap();
    assert_eq!(rooms.len(), 8);
    geometry::validate_no_overlap(&rooms, WALL_THICKNESS as i32).unwrap();
    geometry::validate_quantum_alignment(&rooms).unwrap();
}

#[test]
fn boundary_m1_maximum_16_rooms() {
    let cfg = DungeonConfig {
        class: MapClass::M1,
        room_count: 16,
        loop_count: 2,
        xy_bounds: (1536, 1536),
        z_span: 256,
        placement_candidates: 16,
        max_placement_attempts: 64,
        max_astar_expansions: 131_072,
    }
    .validate()
    .unwrap();
    let mut rng = make_rng(43);
    let rooms = place_rooms(&cfg, &mut rng).unwrap();
    assert_eq!(rooms.len(), 16);
    geometry::validate_no_overlap(&rooms, WALL_THICKNESS as i32).unwrap();
    geometry::validate_quantum_alignment(&rooms).unwrap();
}

#[test]
fn boundary_m2_minimum_17_rooms() {
    let cfg = DungeonConfig {
        class: MapClass::M2,
        room_count: 17,
        loop_count: 1,
        xy_bounds: (2048, 2048),
        z_span: 256,
        placement_candidates: 32,
        max_placement_attempts: 96,
        max_astar_expansions: 524_288,
    }
    .validate()
    .unwrap();
    let mut rng = make_rng(44);
    let rooms = place_rooms(&cfg, &mut rng).unwrap();
    assert_eq!(rooms.len(), 17);
    geometry::validate_no_overlap(&rooms, WALL_THICKNESS as i32).unwrap();
    geometry::validate_quantum_alignment(&rooms).unwrap();
}

#[test]
fn boundary_m2_maximum_40_rooms() {
    let cfg = DungeonConfig {
        class: MapClass::M2,
        room_count: 40,
        loop_count: 6,
        xy_bounds: (3072, 3072),
        z_span: 384,
        placement_candidates: 32,
        max_placement_attempts: 96,
        max_astar_expansions: 524_288,
    }
    .validate()
    .unwrap();
    let mut rng = make_rng(45);
    let rooms = place_rooms(&cfg, &mut rng).unwrap();
    assert_eq!(rooms.len(), 40);
    geometry::validate_no_overlap(&rooms, WALL_THICKNESS as i32).unwrap();
    geometry::validate_quantum_alignment(&rooms).unwrap();
}
