//! Enhanced v2 feature tests — corridor width, ceiling height, pillars,
//! spawn, and light origins.

use bsp_generator::enhanced::config::EnhancedConfig;
use bsp_generator::enhanced::features::{
    apply_features, boxes_intersect, check_connectivity, CorridorWidthRejectionReason,
    FeatureResult, ALLOWED_CEILING_HEIGHTS, ALLOWED_CORRIDOR_WIDTHS, DEFAULT_SAFE_CEILING,
    MIN_HEADROOM,
};
use bsp_generator::enhanced::intent::{LayerId, RoomId};
use bsp_generator::enhanced::placement::{place_rooms, PlacedRoom};
use bsp_generator::enhanced::seed::{tags, EnhancedSeed};
use bsp_generator::enhanced::theme::{assign_uniform, cc0_dungeon_v2_theme};
use bsp_generator::enhanced::topology::build_topology;

// ── Helpers ────────────────────────────────────────────────────────────────

fn build_test_input(
    seed_val: u64,
) -> (
    EnhancedConfig,
    bsp_generator::enhanced::placement::PlacementResult,
    bsp_generator::enhanced::topology::TopologyResult,
    bsp_generator::enhanced::theme::ThemeAssignment,
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

fn run_features(seed_val: u64) -> FeatureResult {
    let (cfg, placement, topology, assignment) = build_test_input(seed_val);
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

// ── Width selection ────────────────────────────────────────────────────────

#[test]
fn width_selection_all_routes_have_width() {
    let result = run_features(42);
    let topology = {
        let (_, _, topo, _) = build_test_input(42);
        topo
    };
    assert_eq!(result.corridor_widths.len(), topology.routes.len());
    for sel in &result.corridor_widths {
        assert!(ALLOWED_CORRIDOR_WIDTHS.contains(&sel.width));
    }
}

#[test]
fn width_selection_default_at_least_64() {
    let result = run_features(42);
    for sel in &result.corridor_widths {
        assert!(sel.width >= 64);
    }
}

#[test]
fn width_selection_records_unreserved_wider_preferences() {
    let result = run_features(42);
    for selection in &result.corridor_widths {
        assert_eq!(selection.width, 64);
        assert_eq!(selection.rejections.len(), 2);
        assert!(selection.rejections.iter().all(|rejection| matches!(
            rejection.reason,
            CorridorWidthRejectionReason::CapacityUnavailable
        )));

        let mut rejected_widths: Vec<_> = selection
            .rejections
            .iter()
            .map(|rejection| rejection.width)
            .collect();
        rejected_widths.sort_unstable();
        assert_eq!(rejected_widths, vec![80, 96]);
    }
}

#[test]
fn width_selection_deterministic() {
    let a = run_features(42).corridor_widths;
    let b = run_features(42).corridor_widths;
    assert_eq!(a, b);
}

#[test]
fn width_selection_different_seeds_may_differ() {
    // Different seeds produce different RNG streams; widths may or may not differ
    let a = run_features(42).corridor_widths;
    let b = run_features(99).corridor_widths;
    // Just check they are both valid
    for sel in &a {
        assert!(ALLOWED_CORRIDOR_WIDTHS.contains(&sel.width));
    }
    for sel in &b {
        assert!(ALLOWED_CORRIDOR_WIDTHS.contains(&sel.width));
    }
}

// ── Ceiling selection ──────────────────────────────────────────────────────

#[test]
fn ceiling_selection_all_rooms_have_height() {
    let result = run_features(42);
    let (_, placement, _, _) = build_test_input(42);
    assert_eq!(result.ceiling_heights.len(), placement.rooms.len());
    for sel in &result.ceiling_heights {
        assert!(sel.height >= MIN_HEADROOM);
        assert!(ALLOWED_CEILING_HEIGHTS.contains(&sel.height) || sel.is_fallback);
    }
}

#[test]
fn ceiling_selection_min_headroom() {
    let result = run_features(42);
    for sel in &result.ceiling_heights {
        assert!(
            sel.height >= MIN_HEADROOM,
            "room {:?} ceiling {} < min headroom",
            sel.room_id,
            sel.height,
        );
    }
}

#[test]
fn ceiling_selection_not_exceeds_room_height() {
    let (_, placement, _, _) = build_test_input(42);
    let result = run_features(42);
    for sel in &result.ceiling_heights {
        let room = placement
            .rooms
            .iter()
            .find(|r| r.id == sel.room_id)
            .unwrap();
        assert!(
            sel.height <= room.dims.2 as i32,
            "room {:?} ceiling {} > room height {}",
            sel.room_id,
            sel.height,
            room.dims.2,
        );
    }
}

#[test]
fn ceiling_selection_deterministic() {
    let a = run_features(42).ceiling_heights;
    let b = run_features(42).ceiling_heights;
    assert_eq!(a, b);
}

#[test]
fn ceiling_fallback_uses_default() {
    let result = run_features(42);
    for sel in &result.ceiling_heights {
        if sel.is_fallback {
            assert_eq!(sel.height, DEFAULT_SAFE_CEILING);
            assert!(sel.fallback_reason.is_some());
        }
    }
}

// ── Pillar placement ───────────────────────────────────────────────────────

#[test]
fn pillars_within_room_shell() {
    let result = run_features(42);
    let (_, placement, _, _) = build_test_input(42);

    for pillar in &result.pillars {
        let room = placement
            .rooms
            .iter()
            .find(|r| r.id == pillar.room_id)
            .unwrap();
        let (px0, py0, pz0, px1, py1, pz1) = pillar.bounds;

        assert!(px0 >= room.shell.0 && py0 >= room.shell.1);
        assert!(px1 <= room.shell.2 && py1 <= room.shell.3);
        assert!(pz0 >= room.floor_z + 16, "pillar below walkable floor");
        assert!(pz1 - pz0 <= 80, "pillar too tall");
        assert!(px0 < px1 && py0 < py1 && pz0 < pz1, "no volume");
    }
}

#[test]
fn pillars_dont_overlap_each_other() {
    let result = run_features(42);

    for i in 0..result.pillars.len() {
        for j in (i + 1)..result.pillars.len() {
            if result.pillars[i].room_id != result.pillars[j].room_id {
                continue;
            }
            let a = result.pillars[i].bounds;
            let b = result.pillars[j].bounds;
            let overlap =
                a.0 < b.3 && a.3 > b.0 && a.1 < b.4 && a.4 > b.1 && a.2 < b.5 && a.5 > b.2;
            assert!(!overlap, "pillars overlap in same room");
        }
    }
}

#[test]
fn pillar_rejections_valid() {
    let result = run_features(42);
    for rej in &result.pillar_rejections {
        assert!(rej.reason.to_string().len() > 0);
    }
}

#[test]
fn exhausted_pillar_quota() {
    // Use a config with high pillar quota so many rooms can't fit them all
    let cfg = EnhancedConfig::with_full_params(28, 3, 1, 16, 2048, 32, 96, 5).unwrap();
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

    // All unmet entries must have placed < requested
    for u in &result.requested_count_unmet {
        assert_eq!(u.requested, cfg.max_pillars_per_room());
        assert!(u.placed < u.requested);
    }
}

// ── Spawn origin ───────────────────────────────────────────────────────────

#[test]
fn spawn_origin_on_entry_layer() {
    let result = run_features(42);
    let (_, placement, _, _) = build_test_input(42);

    let spawn_room = placement
        .rooms
        .iter()
        .find(|r| r.id == result.spawn_point.room_id)
        .unwrap();
    assert!(placement.lower_rooms.contains(&spawn_room.id));
    assert_eq!(result.spawn_point.layer, 0);
}

#[test]
fn spawn_origin_within_room() {
    let result = run_features(42);
    let (_, placement, _, _) = build_test_input(42);

    let spawn_room = placement
        .rooms
        .iter()
        .find(|r| r.id == result.spawn_point.room_id)
        .unwrap();
    let (sx, sy, sz) = result.spawn_point.origin;
    assert!(sx >= spawn_room.shell.0 && sx <= spawn_room.shell.2);
    assert!(sy >= spawn_room.shell.1 && sy <= spawn_room.shell.3);
    assert!(sz >= spawn_room.floor_z + 16);
}

// ── Light origins ──────────────────────────────────────────────────────────

#[test]
fn light_origins_present() {
    let result = run_features(42);
    let (_, placement, _, _) = build_test_input(42);

    assert!(!result.light_origins.is_empty());
    assert_eq!(result.light_origins.len(), placement.rooms.len());
}

#[test]
fn light_origins_within_rooms() {
    let result = run_features(42);
    let (_, placement, _, _) = build_test_input(42);

    for light in &result.light_origins {
        let room = placement
            .rooms
            .iter()
            .find(|r| r.id == light.room_id)
            .unwrap();
        let (lx, ly, lz) = light.origin;
        assert!(lx >= room.shell.0 && lx <= room.shell.2);
        assert!(ly >= room.shell.1 && ly <= room.shell.3);
        assert!(lz >= room.floor_z + 16);
    }
}

#[test]
fn light_origins_avoid_pillars() {
    let result = run_features(42);

    for light in &result.light_origins {
        let light_box = (
            light.origin.0,
            light.origin.1,
            light.origin.2,
            light.origin.0 + 16,
            light.origin.1 + 16,
            light.origin.2 + 16,
        );
        for pillar in &result.pillars {
            if pillar.room_id == light.room_id {
                let overlap = light_box.0 < pillar.bounds.3
                    && light_box.3 > pillar.bounds.0
                    && light_box.1 < pillar.bounds.4
                    && light_box.4 > pillar.bounds.1
                    && light_box.2 < pillar.bounds.5
                    && light_box.5 > pillar.bounds.2;
                assert!(
                    !overlap,
                    "light in room {:?} intersects pillar",
                    light.room_id
                );
            }
        }
    }
}

// ── Determinism ────────────────────────────────────────────────────────────

#[test]
fn required_origins_are_protected_from_pillars() {
    let result = run_features(42);
    let mut required = vec![(result.spawn_point.room_id, result.spawn_point.origin)];
    required.extend(
        result
            .light_origins
            .iter()
            .map(|light| (light.room_id, light.origin)),
    );
    for (room_id, origin) in required {
        let volume = (
            origin.0,
            origin.1,
            origin.2,
            origin.0 + 16,
            origin.1 + 16,
            origin.2 + 16,
        );
        assert!(!result
            .pillars
            .iter()
            .any(|pillar| pillar.room_id == room_id && boxes_intersect(volume, pillar.bounds)));
    }
}

#[test]
fn full_feature_result_deterministic() {
    let a = run_features(42);
    let b = run_features(42);
    assert_eq!(a, b);
}

#[test]
fn different_seeds_different_results() {
    let a = run_features(42);
    let b = run_features(255);
    // Different seeds should (with very high probability) produce different results
    let same = a == b;
    assert!(
        !same || a.corridor_widths.is_empty(),
        "same results across seeds is statistically impossible for non-trivial input"
    );
}

// ── Config validation ──────────────────────────────────────────────────────

#[test]
fn config_max_pillars_range() {
    assert!(EnhancedConfig::with_full_params(28, 3, 1, 16, 2048, 32, 96, 0).is_ok());
    assert!(EnhancedConfig::with_full_params(28, 3, 1, 16, 2048, 32, 96, 8).is_ok());
    assert!(EnhancedConfig::with_full_params(28, 3, 1, 16, 2048, 32, 96, 9).is_err());
}

#[test]
fn nominal_has_pillars_enabled() {
    let cfg = EnhancedConfig::nominal();
    assert_eq!(cfg.max_pillars_per_room(), 2);
}

#[test]
fn minimal_has_pillars_enabled() {
    let cfg = EnhancedConfig::minimal();
    assert_eq!(cfg.max_pillars_per_room(), 1);
}

#[test]
fn maximal_has_pillars_enabled() {
    let cfg = EnhancedConfig::maximal();
    assert_eq!(cfg.max_pillars_per_room(), 4);
}

// ── Legacy unchanged ───────────────────────────────────────────────────────

#[test]
fn legacy_generate_still_works() {
    // Prove Legacy v1 is untouched by Phase 06
    let cfg = bsp_generator::DungeonConfig::nominal_m1();
    let (map, meta) = bsp_generator::generate(0, cfg).unwrap();
    assert!(!map.is_empty());
    assert_eq!(meta.room_count, 12);
}

#[test]
fn legacy_v1_determinism_unchanged() {
    let cfg = bsp_generator::DungeonConfig::nominal_m1();
    let (a, _) = bsp_generator::generate(0, cfg).unwrap();
    let (b, _) = bsp_generator::generate(0, bsp_generator::DungeonConfig::nominal_m1()).unwrap();
    assert_eq!(a, b);
}

// ── Connectivity oracle unit tests ─────────────────────────────────────────

#[test]
fn connectivity_empty_room_connected() {
    let room = PlacedRoom {
        id: RoomId(0),
        layer: LayerId(0),
        floor_z: 0,
        shell: (0, 0, 256, 256),
        dims: (256, 256, 176),
    };
    let candidate = (64, 64, 16, 96, 96, 96);
    assert!(check_connectivity(&room, &[], &[], candidate));
}

#[test]
fn connectivity_full_block_breaks() {
    let room = PlacedRoom {
        id: RoomId(0),
        layer: LayerId(0),
        floor_z: 0,
        shell: (0, 0, 160, 160),
        dims: (160, 160, 176),
    };
    // Interior is 128×128 = 8×8 cells.
    // Full-width bar that splits the room into two disconnected halves
    let candidate = (16, 64, 16, 144, 80, 96);
    assert!(!check_connectivity(&room, &[], &[], candidate));
}

#[test]
fn connectivity_two_small_pillars_ok() {
    let room = PlacedRoom {
        id: RoomId(0),
        layer: LayerId(0),
        floor_z: 0,
        shell: (0, 0, 256, 256),
        dims: (256, 256, 176),
    };
    let existing = vec![(32, 32, 16, 64, 64, 96)];
    let candidate = (160, 160, 16, 192, 192, 96);
    assert!(check_connectivity(&room, &[], &existing, candidate));
}
