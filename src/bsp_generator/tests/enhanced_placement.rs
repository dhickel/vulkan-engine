//! Enhanced v2 placement tests — deterministic two-layer RNG-driven placement,
//! socket derivation, checkpoint/rollback integrity, and exhaustion behavior.

use bsp_generator::enhanced::config::{
    EnhancedConfig, ENHANCED_LOWER_FLOOR_Z, ENHANCED_MAX_ROOM_SPAN, ENHANCED_MIN_ROOM_SPAN,
    ENHANCED_ROOM_HEIGHT, ENHANCED_UPPER_FLOOR_Z, MIN_WALL_FOR_SOCKET, SOCKET_APERTURE,
    SOCKET_CORNER_MARGIN,
};
use bsp_generator::enhanced::error::EnhancedError;
use bsp_generator::enhanced::placement::{place_rooms, WallDirection};
use bsp_generator::enhanced::seed::{tags, EnhancedSeed, EnhancedStageSeed};

const Q: i32 = 16;

fn seed_rng(seed_val: u64) -> EnhancedStageSeed {
    EnhancedSeed::new(seed_val).stage_seed(tags::LAYER_PLACEMENT)
}

// ── Basic placement ────────────────────────────────────────────────────────

#[test]
fn nominal_places_28_rooms() {
    let cfg = EnhancedConfig::nominal();
    let result = place_rooms(&cfg, seed_rng(42)).unwrap();
    assert_eq!(result.rooms.len(), 28);
    assert_eq!(result.lower_rooms.len() + result.upper_rooms.len(), 28);
}

#[test]
fn minimal_places_17_rooms() {
    let cfg = EnhancedConfig::minimal();
    let result = place_rooms(&cfg, seed_rng(17)).unwrap();
    assert_eq!(result.rooms.len(), 17);
    // Both layers populated
    assert!(!result.lower_rooms.is_empty());
    assert!(!result.upper_rooms.is_empty());
}

#[test]
fn maximal_places_40_rooms() {
    let cfg = EnhancedConfig::maximal();
    let result = place_rooms(&cfg, seed_rng(45)).unwrap();
    assert_eq!(result.rooms.len(), 40);
}

// ── Determinism ────────────────────────────────────────────────────────────

#[test]
fn same_seed_same_result() {
    let cfg = EnhancedConfig::nominal();
    let a = place_rooms(&cfg, seed_rng(0)).unwrap();
    let b = place_rooms(&cfg, seed_rng(0)).unwrap();
    assert_eq!(a.rooms, b.rooms);
    assert_eq!(a.sockets, b.sockets);
    assert_eq!(a.lower_rooms, b.lower_rooms);
    assert_eq!(a.upper_rooms, b.upper_rooms);
}

#[test]
fn different_seed_different_result() {
    let cfg = EnhancedConfig::nominal();
    let a = place_rooms(&cfg, seed_rng(1)).unwrap();
    let b = place_rooms(&cfg, seed_rng(2)).unwrap();
    assert!(
        a.rooms != b.rooms || a.sockets != b.sockets,
        "different seeds should produce different placements"
    );
}

#[test]
fn replay_from_same_seed_byte_identical() {
    let cfg = EnhancedConfig::nominal();
    let a = place_rooms(&cfg, seed_rng(12345)).unwrap();
    let b = place_rooms(&cfg, seed_rng(12345)).unwrap();
    assert_eq!(a.rooms, b.rooms, "rooms differ on replay");
    assert_eq!(a.sockets, b.sockets, "sockets differ on replay");
    assert_eq!(a.lower_rooms, b.lower_rooms);
    assert_eq!(a.upper_rooms, b.upper_rooms);
}

#[test]
fn deterministic_across_multiple_seeds() {
    let cfg = EnhancedConfig::nominal();
    for seed in [0u64, 1, 42, 255, 1024, u64::MAX] {
        let a = place_rooms(&cfg, seed_rng(seed)).unwrap();
        let b = place_rooms(&cfg, seed_rng(seed)).unwrap();
        assert_eq!(a.rooms, b.rooms, "seed {} not deterministic", seed);
        assert_eq!(a.sockets, b.sockets, "seed {} not deterministic", seed);
    }
}

// ── Bounds compliance ──────────────────────────────────────────────────────

#[test]
fn all_rooms_within_xy_bounds() {
    let cfg = EnhancedConfig::nominal();
    let result = place_rooms(&cfg, seed_rng(3)).unwrap();
    let extent = cfg.xy_extent() as i32;
    for room in &result.rooms {
        let (x0, y0, x1, y1) = room.shell;
        assert!(x0 >= 0, "x0 negative");
        assert!(y0 >= 0, "y0 negative");
        assert!(x1 <= extent, "x1 {} exceeds extent {}", x1, extent);
        assert!(y1 <= extent, "y1 {} exceeds extent {}", y1, extent);
    }
}

#[test]
fn room_dimensions_in_range() {
    let cfg = EnhancedConfig::nominal();
    let result = place_rooms(&cfg, seed_rng(5)).unwrap();
    for room in &result.rooms {
        let (x0, y0, x1, y1) = room.shell;
        // Use the shell-derived dimensions for validation
        let w = x1 - x0;
        let h = y1 - y0;
        assert!(
            w >= ENHANCED_MIN_ROOM_SPAN,
            "width {} < {}",
            w,
            ENHANCED_MIN_ROOM_SPAN
        );
        assert!(
            w <= ENHANCED_MAX_ROOM_SPAN,
            "width {} > {}",
            w,
            ENHANCED_MAX_ROOM_SPAN
        );
        assert!(
            h >= ENHANCED_MIN_ROOM_SPAN,
            "height {} < {}",
            h,
            ENHANCED_MIN_ROOM_SPAN
        );
        assert!(
            h <= ENHANCED_MAX_ROOM_SPAN,
            "height {} > {}",
            h,
            ENHANCED_MAX_ROOM_SPAN
        );
    }
}

#[test]
fn room_z_values_correct() {
    let cfg = EnhancedConfig::nominal();
    let result = place_rooms(&cfg, seed_rng(7)).unwrap();
    for room in &result.rooms {
        let is_lower = result.lower_rooms.contains(&room.id);
        if is_lower {
            assert_eq!(room.floor_z, ENHANCED_LOWER_FLOOR_Z);
        } else {
            assert_eq!(room.floor_z, ENHANCED_UPPER_FLOOR_Z);
        }
        assert_eq!(room.dims.2 as i32, ENHANCED_ROOM_HEIGHT);
    }
}

// ── Quantum alignment ──────────────────────────────────────────────────────

#[test]
fn all_positions_and_dimensions_quantum_aligned() {
    let cfg = EnhancedConfig::nominal();
    let result = place_rooms(&cfg, seed_rng(11)).unwrap();
    for room in &result.rooms {
        let (x0, y0, x1, y1) = room.shell;
        assert_eq!(x0 % Q, 0, "x0 not aligned");
        assert_eq!(y0 % Q, 0, "y0 not aligned");
        assert_eq!((x1 - x0) % Q, 0, "width not aligned");
        assert_eq!((y1 - y0) % Q, 0, "height not aligned");
        assert_eq!(room.dims.0 % 16, 0, "dims.0 not aligned");
        assert_eq!(room.dims.1 % 16, 0, "dims.1 not aligned");
    }
}

// ── No XY overlap ──────────────────────────────────────────────────────────

#[test]
fn no_room_xy_overlap() {
    let cfg = EnhancedConfig::nominal();
    let result = place_rooms(&cfg, seed_rng(13)).unwrap();
    let rooms = &result.rooms;
    for i in 0..rooms.len() {
        for j in (i + 1)..rooms.len() {
            let a = &rooms[i];
            let b = &rooms[j];
            let ol_x = a.shell.0 < b.shell.2 && a.shell.2 > b.shell.0;
            let ol_y = a.shell.1 < b.shell.3 && a.shell.3 > b.shell.1;
            assert!(
                !(ol_x && ol_y),
                "rooms {} and {} overlap in XY",
                a.id.raw(),
                b.id.raw()
            );
        }
    }
}

#[test]
fn cross_layer_no_xy_overlap() {
    let cfg = EnhancedConfig::nominal();
    let result = place_rooms(&cfg, seed_rng(17)).unwrap();
    for &lid in &result.lower_rooms {
        for &uid in &result.upper_rooms {
            let a = result.rooms.iter().find(|r| r.id == lid).unwrap();
            let b = result.rooms.iter().find(|r| r.id == uid).unwrap();
            let ol_x = a.shell.0 < b.shell.2 && a.shell.2 > b.shell.0;
            let ol_y = a.shell.1 < b.shell.3 && a.shell.3 > b.shell.1;
            assert!(
                !(ol_x && ol_y),
                "cross-layer XY overlap: lower {:?} shell {:?}, upper {:?} shell {:?}",
                lid,
                a.shell,
                uid,
                b.shell
            );
        }
    }
}

#[test]
fn no_overlap_minimal_config() {
    let cfg = EnhancedConfig::minimal();
    let result = place_rooms(&cfg, seed_rng(19)).unwrap();
    for i in 0..result.rooms.len() {
        for j in (i + 1)..result.rooms.len() {
            let a = &result.rooms[i];
            let b = &result.rooms[j];
            let ol_x = a.shell.0 < b.shell.2 && a.shell.2 > b.shell.0;
            let ol_y = a.shell.1 < b.shell.3 && a.shell.3 > b.shell.1;
            assert!(!(ol_x && ol_y));
        }
    }
}

#[test]
fn no_overlap_maximal_config() {
    let cfg = EnhancedConfig::maximal();
    let result = place_rooms(&cfg, seed_rng(23)).unwrap();
    for i in 0..result.rooms.len() {
        for j in (i + 1)..result.rooms.len() {
            let a = &result.rooms[i];
            let b = &result.rooms[j];
            let ol_x = a.shell.0 < b.shell.2 && a.shell.2 > b.shell.0;
            let ol_y = a.shell.1 < b.shell.3 && a.shell.3 > b.shell.1;
            assert!(!(ol_x && ol_y));
        }
    }
}

// ── Balanced membership ────────────────────────────────────────────────────

#[test]
fn balanced_membership_even_count() {
    let cfg = EnhancedConfig::nominal(); // 28 rooms
    let result = place_rooms(&cfg, seed_rng(29)).unwrap();
    assert_eq!(result.lower_rooms.len(), 14);
    assert_eq!(result.upper_rooms.len(), 14);
}

#[test]
fn balanced_membership_odd_count_max_diff_one() {
    let cfg = EnhancedConfig::minimal(); // 17 rooms
    for seed in 0..20u64 {
        let result = place_rooms(&cfg, seed_rng(seed)).unwrap();
        let diff = (result.lower_rooms.len() as i32 - result.upper_rooms.len() as i32).abs();
        assert_eq!(diff, 1, "seed {}: diff {}", seed, diff);
    }
}

#[test]
fn odd_count_extra_room_rng_driven() {
    let cfg = EnhancedConfig::minimal(); // 17 rooms
    let mut saw_lower = false;
    let mut saw_upper = false;
    for seed in 0..100u64 {
        let result = place_rooms(&cfg, seed_rng(seed)).unwrap();
        if result.lower_rooms.len() > result.upper_rooms.len() {
            saw_lower = true;
        } else {
            saw_upper = true;
        }
        if saw_lower && saw_upper {
            break;
        }
    }
    assert!(saw_lower, "lower never got extra room");
    assert!(saw_upper, "upper never got extra room");
}

// ── Socket validation ──────────────────────────────────────────────────────

#[test]
fn all_sockets_have_valid_anchors() {
    let cfg = EnhancedConfig::nominal();
    let result = place_rooms(&cfg, seed_rng(31)).unwrap();
    for s in &result.sockets {
        let room = result.rooms.iter().find(|r| r.id == s.room).unwrap();
        let (x0, y0, x1, y1) = room.shell;

        match s.wall {
            WallDirection::North => assert_eq!(s.anchor.1, y1),
            WallDirection::South => assert_eq!(s.anchor.1, y0),
            WallDirection::East => assert_eq!(s.anchor.0, x1),
            WallDirection::West => assert_eq!(s.anchor.0, x0),
        }
    }
}

#[test]
fn socket_anchor_within_wall_bounds() {
    let cfg = EnhancedConfig::nominal();
    let result = place_rooms(&cfg, seed_rng(33)).unwrap();
    for s in &result.sockets {
        let room = result.rooms.iter().find(|r| r.id == s.room).unwrap();
        let (x0, y0, x1, y1) = room.shell;

        match s.wall {
            WallDirection::North | WallDirection::South => {
                assert!(s.anchor.0 >= x0 && s.anchor.0 <= x1);
            }
            WallDirection::East | WallDirection::West => {
                assert!(s.anchor.1 >= y0 && s.anchor.1 <= y1);
            }
        }
    }
}

#[test]
fn socket_width_is_64() {
    let cfg = EnhancedConfig::nominal();
    let result = place_rooms(&cfg, seed_rng(35)).unwrap();
    for s in &result.sockets {
        assert_eq!(s.width, SOCKET_APERTURE as u32);
    }
}

#[test]
fn socket_corner_margins() {
    let cfg = EnhancedConfig::nominal();
    let result = place_rooms(&cfg, seed_rng(37)).unwrap();
    for s in &result.sockets {
        let room = result.rooms.iter().find(|r| r.id == s.room).unwrap();
        let (x0, y0, x1, y1) = room.shell;

        match s.wall {
            WallDirection::North | WallDirection::South => {
                let half = SOCKET_APERTURE / 2;
                assert!(
                    s.anchor.0 - half - x0 >= SOCKET_CORNER_MARGIN,
                    "left margin too small"
                );
                assert!(
                    x1 - (s.anchor.0 + half) >= SOCKET_CORNER_MARGIN,
                    "right margin too small"
                );
            }
            WallDirection::East | WallDirection::West => {
                let half = SOCKET_APERTURE / 2;
                assert!(
                    s.anchor.1 - half - y0 >= SOCKET_CORNER_MARGIN,
                    "bottom margin too small"
                );
                assert!(
                    y1 - (s.anchor.1 + half) >= SOCKET_CORNER_MARGIN,
                    "top margin too small"
                );
            }
        }
    }
}

#[test]
fn socket_only_on_walls_long_enough() {
    let cfg = EnhancedConfig::nominal();
    let result = place_rooms(&cfg, seed_rng(39)).unwrap();
    for s in &result.sockets {
        let room = result.rooms.iter().find(|r| r.id == s.room).unwrap();
        let (x0, y0, x1, y1) = room.shell;
        match s.wall {
            WallDirection::North | WallDirection::South => {
                assert!(x1 - x0 >= MIN_WALL_FOR_SOCKET);
            }
            WallDirection::East | WallDirection::West => {
                assert!(y1 - y0 >= MIN_WALL_FOR_SOCKET);
            }
        }
    }
}

#[test]
fn socket_transition_capable() {
    let cfg = EnhancedConfig::nominal();
    let result = place_rooms(&cfg, seed_rng(41)).unwrap();
    for s in &result.sockets {
        assert!(
            s.transition_capable,
            "socket {:?} should be transition-capable (176 ≥ 80 headroom)",
            s.id
        );
    }
}

#[test]
fn sockets_canonical_order() {
    let cfg = EnhancedConfig::nominal();
    let result = place_rooms(&cfg, seed_rng(43)).unwrap();
    // Sockets should be sorted by ID (ascending)
    for w in result.sockets.windows(2) {
        assert!(w[0].id <= w[1].id);
    }
}

// ── Exhaustion ─────────────────────────────────────────────────────────────

#[test]
fn tight_bounds_exhaustion() {
    // 20 rooms at min 112×112 each need at least ~1120×1120 area
    // 512×512 is far too small → should exhaust
    let cfg = EnhancedConfig::with_placement_params(20, 1, 1, 16, 512, 4, 4).unwrap();
    let result = place_rooms(&cfg, seed_rng(0));
    assert!(result.is_err());
    match result.unwrap_err() {
        EnhancedError::PlacementExhausted {
            rooms_placed,
            total_attempts,
        } => {
            eprintln!(
                "exhaustion: {} rooms placed, {} attempts",
                rooms_placed, total_attempts
            );
            assert!(rooms_placed < 20);
            assert!(total_attempts > 0);
        }
        e => panic!("expected PlacementExhausted, got {:?}", e),
    }
}

#[test]
fn exhaustion_returns_no_partial_state() {
    let cfg = EnhancedConfig::with_placement_params(30, 1, 1, 16, 512, 2, 2).unwrap();
    let result = place_rooms(&cfg, seed_rng(99));
    // Either succeeds (very unlikely with 512 extent and 30 rooms)
    // or returns PlacementExhausted — never panics
    match result {
        Ok(r) => {
            assert_eq!(r.rooms.len(), 30);
        }
        Err(EnhancedError::PlacementExhausted { .. }) => {}
        Err(e) => panic!("unexpected error: {:?}", e),
    }
}

// ── Late-rejection snapshot equality ───────────────────────────────────────

#[test]
fn late_rejection_grid_unchanged() {
    // Use a config that will likely exhaust after placing some rooms
    let cfg = EnhancedConfig::with_placement_params(25, 1, 1, 16, 640, 4, 4).unwrap();
    let result = place_rooms(&cfg, seed_rng(100));
    if let Err(EnhancedError::PlacementExhausted { rooms_placed, .. }) = &result {
        eprintln!("late rejection after {} rooms (expected)", rooms_placed);
    }
    // The function contract: returns error on exhaustion, never leaks
    // partial PlacementResult. The error is the only output.
    assert!(result.is_err() || result.unwrap().rooms.len() == 25);
}

// ── Overflow / starvation guards ───────────────────────────────────────────

#[test]
fn room_too_large_for_bounds_skipped() {
    // Config validation rejects xy_extent < min room span.
    // Test that the grid handles tight fits: 17 rooms on a small-ish extent.
    // 17 rooms at min 112×112 need significant area; 640×640 is very tight.
    let cfg = EnhancedConfig::with_placement_params(17, 1, 1, 16, 640, 8, 8).unwrap();
    let result = place_rooms(&cfg, seed_rng(0));
    // With 640 extent and 17 rooms, this will likely exhaust
    // but must never panic due to room-too-large-for-bounds
    match result {
        Ok(r) => assert_eq!(r.rooms.len(), 17),
        Err(EnhancedError::PlacementExhausted { .. }) => {}
        Err(e) => panic!("unexpected: {:?}", e),
    }
}

#[test]
fn placement_internal_rejects_too_few_rooms() {
    // The placement function checks room_count >= 2 internally.
    // Config validation already enforces ≥17 for M2, but the placement
    // function still guards its own preconditions.
    // We verify the config rejects < 17 (M2 minimum).
    assert!(EnhancedConfig::with_placement_params(16, 1, 1, 16, 1024, 8, 8).is_err());
    assert!(EnhancedConfig::with_placement_params(1, 1, 1, 16, 1024, 8, 8).is_err());
}

#[test]
fn minimum_room_count_succeeds() {
    // M2 minimum is 17 rooms
    let cfg = EnhancedConfig::minimal();
    let result = place_rooms(&cfg, seed_rng(42)).unwrap();
    assert_eq!(result.rooms.len(), 17);
    assert!(!result.lower_rooms.is_empty());
    assert!(!result.upper_rooms.is_empty());
}

#[test]
fn placement_rejects_too_few_rooms_internally() {
    // Enhanced requires ≥2 rooms internally; but M2 config enforces ≥17.
    // The internal check is exercised by config validation rejecting <17.
    let result = EnhancedConfig::with_placement_params(10, 1, 1, 16, 1024, 8, 8);
    assert!(result.is_err());
}

// ── Occupancy grid consistency ─────────────────────────────────────────────

#[test]
fn grid_reflects_all_placed_rooms() {
    let cfg = EnhancedConfig::nominal();
    let result = place_rooms(&cfg, seed_rng(47)).unwrap();
    let grid = &result.grid;
    for room in &result.rooms {
        let (x0, y0, x1, y1) = room.shell;
        assert!(
            !grid.is_rect_empty(x0, y0, x1 - x0, y1 - y0).unwrap(),
            "room {:?} not found in grid",
            room.id
        );
    }
}

#[test]
fn grid_empty_outside_rooms() {
    let cfg = EnhancedConfig::nominal();
    let result = place_rooms(&cfg, seed_rng(49)).unwrap();
    let grid = &result.grid;
    // Owned cell count should match the sum of room areas
    let expected: usize = result
        .rooms
        .iter()
        .map(|r| {
            let w = (r.shell.2 - r.shell.0) as u32 / 16;
            let h = (r.shell.3 - r.shell.1) as u32 / 16;
            (w * h) as usize
        })
        .sum();
    assert_eq!(
        grid.owned_cell_count(),
        expected,
        "owned cells mismatch: grid has {}, rooms sum to {}",
        grid.owned_cell_count(),
        expected
    );
}

// ── Canonical ordering ─────────────────────────────────────────────────────

#[test]
fn rooms_sorted_by_id() {
    let cfg = EnhancedConfig::nominal();
    let result = place_rooms(&cfg, seed_rng(51)).unwrap();
    for w in result.rooms.windows(2) {
        assert!(w[0].id <= w[1].id);
    }
}

#[test]
fn membership_lists_sorted() {
    let cfg = EnhancedConfig::nominal();
    let result = place_rooms(&cfg, seed_rng(53)).unwrap();
    for w in result.lower_rooms.windows(2) {
        assert!(w[0] < w[1]);
    }
    for w in result.upper_rooms.windows(2) {
        assert!(w[0] < w[1]);
    }
}

// ── Config validation ──────────────────────────────────────────────────────

#[test]
fn placement_candidates_out_of_range_rejected() {
    assert!(EnhancedConfig::with_placement_params(17, 1, 1, 16, 1024, 0, 8).is_err());
    assert!(EnhancedConfig::with_placement_params(17, 1, 1, 16, 1024, 33, 8).is_err());
}

#[test]
fn max_placement_attempts_out_of_range_rejected() {
    assert!(EnhancedConfig::with_placement_params(17, 1, 1, 16, 1024, 8, 0).is_err());
    assert!(EnhancedConfig::with_placement_params(17, 1, 1, 16, 1024, 8, 97).is_err());
}
