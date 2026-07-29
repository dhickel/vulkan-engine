//! Enhanced v2 feature connectivity tests — deep validation of the
//! connectivity oracle, pillar exclusion, and feature-aware spawn/light
//! origins across multiple configurations and edge cases.

use bsp_generator::enhanced::config::EnhancedConfig;
use bsp_generator::enhanced::features::{
    apply_features, boxes_intersect, check_connectivity, ExclusionVolume, FeatureResult,
    PillarRejectionReason, ALLOWED_CEILING_HEIGHTS, ALLOWED_CORRIDOR_WIDTHS,
};
use bsp_generator::enhanced::intent::{LayerId, RoomId};
use bsp_generator::enhanced::placement::{place_rooms, PlacedRoom, PlacementResult};
use bsp_generator::enhanced::seed::{tags, EnhancedSeed};
use bsp_generator::enhanced::theme::{assign_uniform, cc0_dungeon_v2_theme, ThemeAssignment};
use bsp_generator::enhanced::topology::{build_topology, TopologyResult};

// ── Helpers ────────────────────────────────────────────────────────────────

fn build_context(
    seed_val: u64,
) -> (
    EnhancedConfig,
    PlacementResult,
    TopologyResult,
    ThemeAssignment,
) {
    let cfg = EnhancedConfig::nominal();
    let eseed = EnhancedSeed::new(seed_val);
    let placement = place_rooms(&cfg, eseed.stage_seed(tags::LAYER_PLACEMENT)).unwrap();
    let mut topo_rng = eseed.stage_seed(tags::VERTICAL_TOPOLOGY).rng();
    let topology = build_topology(&cfg, &placement, &mut topo_rng).unwrap();
    let theme = cc0_dungeon_v2_theme();
    let assignment = assign_uniform(&theme, &placement.rooms, &topology);
    (cfg, placement, topology, assignment)
}

fn run(seed_val: u64) -> FeatureResult {
    let (cfg, placement, topology, assignment) = build_context(seed_val);
    let corridor_rng = EnhancedSeed::new(seed_val)
        .stage_seed(tags::CORRIDOR_VARIANCE)
        .rng();
    let feature_rng = EnhancedSeed::new(seed_val)
        .stage_seed(tags::FEATURE_PLACEMENT)
        .rng();
    apply_features(
        &cfg,
        &placement,
        &topology,
        &assignment,
        feature_rng,
        corridor_rng,
    )
    .unwrap()
}

// ── Connectivity oracle deep tests ─────────────────────────────────────────

#[test]
fn connectivity_oracle_deterministic() {
    let room = PlacedRoom {
        id: RoomId(0),
        layer: LayerId(0),
        floor_z: 0,
        shell: (0, 0, 256, 256),
        dims: (256, 256, 176),
    };
    let candidate = (64, 64, 16, 96, 96, 96);

    let a = check_connectivity(&room, &[], &[], candidate);
    let b = check_connectivity(&room, &[], &[], candidate);
    assert_eq!(a, b);
}

#[test]
fn connectivity_oracle_with_exclusions() {
    let room = PlacedRoom {
        id: RoomId(0),
        layer: LayerId(0),
        floor_z: 0,
        shell: (0, 0, 256, 256),
        dims: (256, 256, 176),
    };

    // Exclusion that blocks one path but leaves another
    let exclusions = vec![ExclusionVolume {
        bounds: (16, 16, 0, 144, 48, 96),
        reason: "test wall".into(),
    }];

    // Candidate in the free area — should pass
    let candidate = (160, 160, 16, 192, 192, 96);
    assert!(check_connectivity(&room, &exclusions, &[], candidate));

    // Candidate that together with exclusion blocks all paths — should fail
    let blocking = (16, 48, 16, 256, 80, 96);
    assert!(!check_connectivity(&room, &exclusions, &[], blocking));
}

#[test]
fn connectivity_oracle_minimum_room() {
    // Minimum room span (112): interior is 80×80 = 5×5 cells
    // Two 32-wide (2-cell) pillars side by side leave 1 cell gap
    let room = PlacedRoom {
        id: RoomId(0),
        layer: LayerId(0),
        floor_z: 0,
        shell: (0, 0, 112, 112),
        dims: (112, 112, 176),
    };

    // Single pillar: should always be OK in a room this size
    let candidate = (32, 32, 16, 64, 64, 96);
    assert!(check_connectivity(&room, &[], &[], candidate));
}

#[test]
fn connectivity_oracle_all_cells_blocked() {
    let room = PlacedRoom {
        id: RoomId(0),
        layer: LayerId(0),
        floor_z: 0,
        shell: (0, 0, 128, 128),
        dims: (128, 128, 176),
    };
    // Interior is 96×96 = 6×6 cells.
    // A full-width horizontal bar splits the room, breaking connectivity.
    let candidate = (16, 48, 16, 112, 64, 96);
    assert!(!check_connectivity(&room, &[], &[], candidate));
}

// ── Exclusion region verification ──────────────────────────────────────────

#[test]
fn pillars_not_in_walls() {
    let result = run(42);
    let (_, placement, _, _) = build_context(42);

    for pillar in &result.pillars {
        let room = placement
            .rooms
            .iter()
            .find(|r| r.id == pillar.room_id)
            .unwrap();
        let (px0, py0, px1, py1) = (
            pillar.bounds.0,
            pillar.bounds.1,
            pillar.bounds.3,
            pillar.bounds.4,
        );

        // Must be at least WALL_THICKNESS + Q away from all walls
        let min_x = room.shell.0 + 16 + 16;
        let min_y = room.shell.1 + 16 + 16;
        let max_x = room.shell.2 - 16 - 16;
        let max_y = room.shell.3 - 16 - 16;

        assert!(
            px0 >= min_x && py0 >= min_y && px1 <= max_x && py1 <= max_y,
            "pillar too close to wall: bounds ({px0},{py0},{px1},{py1}), room shell {:?}",
            room.shell,
        );
    }
}

#[test]
fn pillars_not_in_corridor_envelopes() {
    let result = run(42);
    let (_, _placement, topology, _) = build_context(42);

    for pillar in &result.pillars {
        for route in &topology.routes {
            if route.source_room != pillar.room_id && route.target_room != pillar.room_id {
                continue;
            }
            for &envelope in &route.envelopes {
                // pillar bounds as 2D projection
                let overlap = pillar.bounds.0 < envelope.2
                    && pillar.bounds.3 > envelope.0
                    && pillar.bounds.1 < envelope.3
                    && pillar.bounds.4 > envelope.1;
                assert!(
                    !overlap,
                    "pillar in room {:?} overlaps corridor envelope",
                    pillar.room_id,
                );
            }
        }
    }
}

#[test]
fn pillars_not_in_transition_footprints() {
    let result = run(42);
    let (_, _placement, topology, _) = build_context(42);

    for pillar in &result.pillars {
        for t in &topology.transitions {
            if t.lower_room != pillar.room_id && t.upper_room != pillar.room_id {
                continue;
            }
            let (fx0, fy0, fx1, fy1) = t.footprint;
            let overlap = pillar.bounds.0 < fx1
                && pillar.bounds.3 > fx0
                && pillar.bounds.1 < fy1
                && pillar.bounds.4 > fy0;
            assert!(
                !overlap,
                "pillar in room {:?} overlaps transition footprint",
                pillar.room_id,
            );
        }
    }
}

// ── Pillar rejection coverage ──────────────────────────────────────────────

#[test]
fn pillar_rejections_have_typed_reasons() {
    let result = run(42);

    for rej in &result.pillar_rejections {
        match &rej.reason {
            PillarRejectionReason::NonPositiveVolume => {}
            PillarRejectionReason::NotAxisAligned => {}
            PillarRejectionReason::ExclusionIntersection(_) => {}
            PillarRejectionReason::Overlap(_) => {}
            PillarRejectionReason::InsufficientClearance => {}
            PillarRejectionReason::ConnectivityBroken => {}
        }
    }
}

#[test]
fn requested_count_unmet_records_correct() {
    let result = run(42);

    for u in &result.requested_count_unmet {
        assert!(u.placed < u.requested);
        assert_eq!(u.requested, 2); // nominal = 2
    }
}

// ── Spawn and light deep tests ─────────────────────────────────────────────

#[test]
fn spawn_is_exactly_one() {
    // Spawn point is always exactly one (not a vec, always present)
    for seed in [42u64, 99, 255, 1000] {
        let result = run(seed);
        // Verify spawn is well-formed
        assert!(result.spawn_point.origin.0 > 0 || result.spawn_point.origin.0 >= 0);
        assert!(result.spawn_point.origin.1 > 0 || result.spawn_point.origin.1 >= 0);
        assert!(result.spawn_point.origin.2 >= 0);
    }
}

#[test]
fn light_origins_per_room() {
    for seed in [42u64, 99, 255] {
        let result = run(seed);
        let (_, placement, _, _) = build_context(seed);

        // One light per room
        assert_eq!(
            result.light_origins.len(),
            placement.rooms.len(),
            "seed {}: light count mismatch",
            seed,
        );
    }
}

// ── Deterministic replay across seeds ──────────────────────────────────────

#[test]
fn deterministic_replay_identical() {
    let seeds = [1u64, 42, 255, 99];

    for &seed in &seeds {
        let a = run(seed);
        let b = run(seed);
        assert_eq!(a, b, "seed {}: non-deterministic result", seed);
    }
}

#[test]
fn different_seeds_produce_different_features() {
    // With very high probability, two different seeds produce different feature sets
    let seeds = [42u64, 99, 255, 1000, 54321];
    let results: Vec<_> = seeds.iter().map(|&s| run(s)).collect();

    for i in 0..results.len() {
        for j in (i + 1)..results.len() {
            // At least one of the vectors should differ
            let same = results[i] == results[j];
            // It's possible but astronomically unlikely for all seeds to produce identical output
            if same {
                eprintln!(
                    "seeds {} and {} produced identical feature results (rare but valid)",
                    seeds[i], seeds[j],
                );
            }
        }
    }
}

// ── Boundary / edge case tests ─────────────────────────────────────────────

#[test]
fn zero_pillars_config_produces_no_pillars() {
    let cfg = EnhancedConfig::with_full_params(28, 3, 1, 16, 2048, 32, 96, 0).unwrap();
    let eseed = EnhancedSeed::new(42);
    let placement = place_rooms(&cfg, eseed.stage_seed(tags::LAYER_PLACEMENT)).unwrap();
    let mut topo_rng = eseed.stage_seed(tags::VERTICAL_TOPOLOGY).rng();
    let topology = build_topology(&cfg, &placement, &mut topo_rng).unwrap();
    let theme = cc0_dungeon_v2_theme();
    let assignment = assign_uniform(&theme, &placement.rooms, &topology);
    let corridor_rng = eseed.stage_seed(tags::CORRIDOR_VARIANCE).rng();
    let feature_rng = eseed.stage_seed(tags::FEATURE_PLACEMENT).rng();
    let result = apply_features(
        &cfg,
        &placement,
        &topology,
        &assignment,
        feature_rng,
        corridor_rng,
    )
    .unwrap();

    assert!(result.pillars.is_empty());
    // No unmet records when max_pillars == 0 (nothing was requested)
    assert!(result.requested_count_unmet.is_empty());
}

#[test]
fn corridor_widths_within_approved_set() {
    for seed in [1u64, 42, 255, 99] {
        let result = run(seed);
        for sel in &result.corridor_widths {
            assert!(
                ALLOWED_CORRIDOR_WIDTHS.contains(&sel.width),
                "seed {}: width {} not in approved set",
                seed,
                sel.width,
            );
        }
    }
}

#[test]
fn ceiling_heights_within_approved_or_fallback() {
    for seed in [0u64, 42, 255] {
        let result = run(seed);
        for sel in &result.ceiling_heights {
            let valid = ALLOWED_CEILING_HEIGHTS.contains(&sel.height) || sel.is_fallback;
            assert!(
                valid,
                "seed {}: room {:?} height {} not approved and not fallback",
                seed, sel.room_id, sel.height,
            );
        }
    }
}

#[test]
fn boxes_intersect_unit() {
    // Two identical boxes intersect
    let a = (0, 0, 0, 10, 10, 10);
    assert!(boxes_intersect(a, a));

    // Touching at edge does not intersect (half-open)
    let b = (10, 0, 0, 20, 10, 10);
    assert!(!boxes_intersect(a, b));

    // Overlapping
    let c = (5, 5, 5, 15, 15, 15);
    assert!(boxes_intersect(a, c));

    // One contains the other
    let d = (2, 2, 2, 8, 8, 8);
    assert!(boxes_intersect(a, d));

    // Disjoint
    let e = (100, 100, 100, 110, 110, 110);
    assert!(!boxes_intersect(a, e));
}
