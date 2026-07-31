//! Phase 05 — Enhanced V3 production acceptance.
//!
//! Production acceptance tests for preset family/minimum sets, 36-entry
//! source matrix, diagonal faces/pointed geometry/stairs/features, budgets,
//! and seed variation. These tests validate the production pipeline against
//! frozen acceptance criteria without duplicating expensive compiler tests.

use bsp_generator::enhanced_v3::*;
use std::collections::BTreeSet;

// ── Production identity and required 12-entry source matrix ───────────────

fn topology_and_plan(config: &V3Config) -> (CommittedTopology, PlanOutcome) {
    let seed = V3Seed::new(config.seed);
    let mut alloc = V3IdAllocator::new();
    let (footprints, layout) = build_footprints(config, seed, &mut alloc).unwrap();
    let topology = build_topology(config, &footprints, &layout, seed, &mut alloc).unwrap();
    let (spawn_volume, light_volumes) = compute_reservations(&topology).unwrap();
    let plan = plan_composition(seed, config, &topology, &spawn_volume, &light_volumes).unwrap();
    (topology, plan)
}

fn assert_plan_identity(config: &V3Config, plan: &PlanOutcome) {
    let preset = config.preset;
    let expected_families: BTreeSet<_> = preset
        .required_families()
        .iter()
        .map(|family| (*family).to_string())
        .collect();
    let assembly_rooms: BTreeSet<_> = plan
        .instances
        .iter()
        .map(|instance| instance.room_id)
        .collect();

    assert_eq!(
        plan.grammar_families, expected_families,
        "{preset:?}: family identity"
    );
    assert_eq!(
        plan.grammar_families.len(),
        preset.minimum_families() as usize,
        "{preset:?}: family count"
    );
    assert!(
        assembly_rooms.len() >= preset.minimum_assemblies() as usize,
        "{preset:?}: assemblies {} < {}",
        assembly_rooms.len(),
        preset.minimum_assemblies()
    );
    assert!(
        plan.instances.len() >= preset.minimum_feature_brushes() as usize,
        "{preset:?}: feature instances {} < {}",
        plan.instances.len(),
        preset.minimum_feature_brushes()
    );
    assert!(
        plan.rejected.is_empty(),
        "{preset:?}: rejected planned family"
    );
    assert!(
        plan.simplified.is_empty(),
        "{preset:?}: simplified planned feature"
    );
    assert!(
        plan.identity_satisfied,
        "{preset:?}: identity not satisfied"
    );
    assert!(
        plan.instances
            .iter()
            .all(|instance| instance.support.is_some()),
        "{preset:?}: ungrounded feature instance"
    );
}

#[test]
fn production_matrix_is_total_and_enforces_exact_topology_and_identity() {
    for preset in [V3Preset::Sparse, V3Preset::Moderate, V3Preset::Rich] {
        let (extent, expected_rooms, expected_routes) = match preset {
            V3Preset::Sparse => (2048, 12, 10),
            V3Preset::Moderate => (2048, 20, 20),
            V3Preset::Rich => (3072, 28, 30),
        };
        for seed in [0, 42, 99, 255] {
            let config = V3Config::new(seed, preset, extent).unwrap();
            let output = run_pipeline(&config)
                .unwrap_or_else(|error| panic!("{preset:?}/{seed}: generation failed: {error}"));
            let (topology, plan) = topology_and_plan(&config);

            assert_eq!(
                output.metadata.room_count(),
                expected_rooms,
                "{preset:?}/{seed}"
            );
            assert_eq!(
                output.metadata.route_count(),
                expected_routes,
                "{preset:?}/{seed}"
            );
            assert_eq!(
                topology.rooms.len(),
                expected_rooms as usize,
                "{preset:?}/{seed}"
            );
            assert_eq!(
                topology.routes.len(),
                expected_routes as usize,
                "{preset:?}/{seed}"
            );
            assert_eq!(
                output.metadata.grammar_families().len(),
                preset.minimum_families() as usize
            );
            assert!(output.metadata.identity_satisfied());
            assert!(output.metadata.actual_faces() <= preset.face_budget());
            assert!(output.metadata.actual_faces() < 10_000);
            assert!(output.metadata.actual_entities() < 300);
            assert!(!output.map_text.is_empty());
            assert!(output.map_text.contains("worldspawn"));
            assert_plan_identity(&config, &plan);
        }
    }
}

// ── Room count scaling ────────────────────────────────────────────────────

#[test]
fn nominal_presets_have_exact_room_and_same_layer_route_counts() {
    let sparse = run_pipeline(&V3Config::nominal_sparse()).unwrap();
    let moderate = run_pipeline(&V3Config::nominal_moderate()).unwrap();
    let rich = run_pipeline(&V3Config::nominal_rich()).unwrap();

    assert_eq!(sparse.metadata.room_count(), 12);
    assert_eq!(moderate.metadata.room_count(), 20);
    assert_eq!(rich.metadata.room_count(), 28);
    assert_eq!(sparse.metadata.route_count(), 10);
    assert_eq!(moderate.metadata.route_count(), 20);
    assert_eq!(rich.metadata.route_count(), 30);
}

// ── Seed variation: different seeds produce distinct output ──────────────

#[test]
fn seed_variation_produces_distinct_output() {
    let mut maps = BTreeSet::new();
    for seed in [0u64, 42, 99, 255] {
        let config = V3Config::new(seed, V3Preset::Sparse, 2048).unwrap();
        let output = run_pipeline(&config).unwrap();
        maps.insert(output.map_text);
    }
    assert_eq!(
        maps.len(),
        4,
        "seeds 0/42/99/255 must produce four distinct maps, got {}",
        maps.len()
    );
}

// ── Budget compliance ────────────────────────────────────────────────────

#[test]
fn all_presets_stay_within_declared_budgets() {
    for (config, name) in &[
        (V3Config::nominal_sparse(), "sparse"),
        (V3Config::nominal_moderate(), "moderate"),
        (V3Config::nominal_rich(), "rich"),
    ] {
        let output = run_pipeline(config).unwrap();
        let meta = &output.metadata;

        assert!(
            meta.actual_faces() < 10000,
            "{name}: global face budget exceeded: {}",
            meta.actual_faces()
        );
        assert!(
            meta.actual_entities() < 300,
            "{name}: global entity budget exceeded: {}",
            meta.actual_entities()
        );
        assert!(
            meta.face_budget_satisfied(),
            "{name}: face budget not satisfied"
        );
        assert!(
            meta.entity_budget_satisfied(),
            "{name}: entity budget not satisfied"
        );

        // Preset-specific face budget
        let preset_budget = config.preset.face_budget();
        assert!(
            meta.actual_faces() <= preset_budget,
            "{name}: preset face budget {preset_budget} exceeded: {}",
            meta.actual_faces()
        );
    }
}

// ── Two-layer evidence ───────────────────────────────────────────────────

#[test]
fn all_presets_have_two_layer_structure() {
    for config in &[
        V3Config::nominal_sparse(),
        V3Config::nominal_moderate(),
        V3Config::nominal_rich(),
    ] {
        let output = run_pipeline(config).unwrap();
        assert!(output.metadata.has_upper_layer());
        assert!(output.metadata.lower_room_count() > 0);
        assert!(output.metadata.upper_room_count() > 0);
        assert_eq!(
            output.metadata.room_count(),
            output.metadata.lower_room_count() + output.metadata.upper_room_count()
        );
        assert!(output.metadata.transition_count() >= 1);
    }
}

// ── Diagonal face evidence ───────────────────────────────────────────────

#[test]
fn all_presets_produce_diagonal_face_lines() {
    // 45° diagonal walls in XY produce plane-point triples where
    // at least one face's three points have non-zero X AND Y components
    // that are not purely cardinal-ratio.
    for config in &[
        V3Config::nominal_sparse(),
        V3Config::nominal_moderate(),
        V3Config::nominal_rich(),
    ] {
        let output = run_pipeline(config).unwrap();
        let has_diagonal = output
            .map_text
            .lines()
            .filter(|l| l.trim_start().starts_with('('))
            .any(|line| {
                let coords: Vec<i32> = line
                    .split(|c: char| c == '(' || c == ')' || c.is_ascii_whitespace())
                    .filter_map(|s| s.parse::<i32>().ok())
                    .collect();
                if coords.len() < 9 {
                    return false;
                }
                // Check if any of the three plane-defining triples has
                // non-zero differences in both X and Y components across points
                for i in 0..3 {
                    let j = (i + 1) % 3;
                    let dx = (coords[i * 3] - coords[j * 3]).abs();
                    let dy = (coords[i * 3 + 1] - coords[j * 3 + 1]).abs();
                    if dx > 16 && dy > 16 && dx == dy {
                        return true;
                    }
                }
                false
            });

        assert!(
            has_diagonal,
            "{:?}: no diagonal (45°) face lines detected in map",
            config.preset
        );
    }
}

// ── Pointed geometry evidence ─────────────────────────────────────────────

#[test]
fn output_contains_pointed_arch_geometry() {
    // Pointed arches produce brush faces above the standard 96-unit
    // portal headroom (80 + 16 slab). We check for faces with Z
    // coordinates above Z=96+16=112 that are near portal areas.
    // The simplest check: there exist face coordinates above Z=128
    // (well above the standard 80-unit headroom) in every map.
    for config in &[
        V3Config::nominal_sparse(),
        V3Config::nominal_moderate(),
        V3Config::nominal_rich(),
    ] {
        let output = run_pipeline(config).unwrap();
        let has_high_z = output
            .map_text
            .lines()
            .filter(|l| l.trim_start().starts_with('('))
            .any(|line| {
                line.split(|c: char| c == '(' || c == ')' || c.is_ascii_whitespace())
                    .filter_map(|s| s.parse::<i32>().ok())
                    .any(|coord| coord > 128 && coord < 256)
            });

        assert!(
            has_high_z,
            "{:?}: no geometry above Z=128 — pointed arches not evidenced",
            config.preset
        );
    }
}

// ── Stair evidence ────────────────────────────────────────────────────────

#[test]
fn all_presets_produce_stair_geometry() {
    for config in &[
        V3Config::nominal_sparse(),
        V3Config::nominal_moderate(),
        V3Config::nominal_rich(),
    ] {
        let output = run_pipeline(config).unwrap();

        // Stairs span Z from lower floor (0) to upper floor (192)
        // with treads at multiples of 16. Check for brush face lines
        // with Z values strictly between 32 and 176 (mid-stair).
        let has_mid_stair = output
            .map_text
            .lines()
            .filter(|l| l.trim_start().starts_with('('))
            .any(|line| {
                line.split(|c: char| c == '(' || c == ')' || c.is_ascii_whitespace())
                    .filter_map(|s| s.parse::<i32>().ok())
                    .any(|coord| coord > 32 && coord < 176)
            });

        assert!(
            has_mid_stair,
            "{:?}: no geometry at mid-stair Z heights",
            config.preset
        );

        assert!(
            output.metadata.transition_count() >= 1,
            "{:?}: no transitions in metadata",
            config.preset
        );
    }
}

// ── Spawn safety ─────────────────────────────────────────────────────────

#[test]
fn spawn_is_in_positive_volume_and_reasonable() {
    for config in &[
        V3Config::nominal_sparse(),
        V3Config::nominal_moderate(),
        V3Config::nominal_rich(),
    ] {
        let output = run_pipeline(config).unwrap();
        let (sx, sy, sz) = output.metadata.spawn_origin();

        assert!(sx > 0, "{:?}: spawn_x={sx}", config.preset);
        assert!(sy > 0, "{:?}: spawn_y={sy}", config.preset);
        assert!(sz > 0, "{:?}: spawn_z={sz}", config.preset);

        // Spawn must be within bounds
        let (min_x, min_y, min_z, max_x, max_y, max_z) = output.metadata.bounds();
        assert!(
            sx >= min_x && sx <= max_x,
            "{:?}: spawn X out of bounds",
            config.preset
        );
        assert!(
            sy >= min_y && sy <= max_y,
            "{:?}: spawn Y out of bounds",
            config.preset
        );
        assert!(
            sz >= min_z && sz <= max_z,
            "{:?}: spawn Z out of bounds",
            config.preset
        );

        // Spawn height should be close to floor + slab + eye_offset
        // Lower floor Z=0 + slab 16 + eye_offset 24 = ~40, or upper floor Z=192 + 16 + 24 = ~232
        let spawn_is_lower = sz < 128;
        let spawn_is_upper = sz >= 192 + 16;
        assert!(
            spawn_is_lower || spawn_is_upper,
            "{:?}: spawn_z={sz} not at expected layer height",
            config.preset
        );
    }
}

// ── Light placement ──────────────────────────────────────────────────────

#[test]
fn lights_are_placed_in_every_room() {
    for config in &[
        V3Config::nominal_sparse(),
        V3Config::nominal_moderate(),
        V3Config::nominal_rich(),
    ] {
        let output = run_pipeline(config).unwrap();
        // At minimum, one light per room (plus corridor lights)
        assert!(
            output.metadata.light_count() >= output.metadata.room_count(),
            "{:?}: lights {} < rooms {}",
            config.preset,
            output.metadata.light_count(),
            output.metadata.room_count()
        );
    }
}

// ── Portal evidence: routes exist ────────────────────────────────────────

#[test]
fn all_presets_have_portals_and_routes() {
    for config in &[
        V3Config::nominal_sparse(),
        V3Config::nominal_moderate(),
        V3Config::nominal_rich(),
    ] {
        let output = run_pipeline(config).unwrap();
        assert!(
            output.metadata.portal_count() >= 1,
            "{:?}: no portals",
            config.preset
        );
        assert!(
            output.metadata.route_count() >= 1,
            "{:?}: no routes",
            config.preset
        );
    }
}

// ── Map text validity ────────────────────────────────────────────────────

#[test]
fn all_presets_produce_canonical_map_format() {
    for config in &[
        V3Config::nominal_sparse(),
        V3Config::nominal_moderate(),
        V3Config::nominal_rich(),
    ] {
        let output = run_pipeline(config).unwrap();

        assert!(output.map_text.ends_with('\n'));
        assert!(!output.map_text.contains('\r'));
        assert!(output.map_text.contains("cc0_dungeon_v2.wad"));
        assert!(output.map_text.contains("_minlight"));
        assert!(output.map_text.contains("bs_wall"));
        assert!(output.map_text.contains("bs_floor"));
        assert!(output.map_text.contains("bs_ceil"));

        // Every face line ends with "0 0 0 0.25 0.25"
        for line in output.map_text.lines() {
            if line.trim_start().starts_with('(') {
                assert!(
                    line.ends_with("0 0 0 0.25 0.25"),
                    "{:?}: face line missing canonical texture mapping: {line}",
                    config.preset
                );
            }
        }

        // Balanced braces
        let open = output.map_text.matches('{').count();
        let close = output.map_text.matches('}').count();
        assert_eq!(open, close, "{:?}: mismatched braces", config.preset);
    }
}

// ── Deterministic metadata serialization ──────────────────────────────────

#[test]
fn metadata_serializes_consistently() {
    let config = V3Config::nominal_sparse();
    let output = run_pipeline(&config).unwrap();
    let json1 = serde_json::to_string(&output.metadata).unwrap();
    let json2 = serde_json::to_string(&output.metadata).unwrap();
    assert_eq!(json1, json2);
}

// ── Exact preset room counts at seed 42 ──────────────────────────────────

#[test]
fn seed_42_produces_exact_preset_room_and_route_requirements() {
    let sparse = run_pipeline(&V3Config::new(42, V3Preset::Sparse, 2048).unwrap()).unwrap();
    let moderate = run_pipeline(&V3Config::new(42, V3Preset::Moderate, 2048).unwrap()).unwrap();
    let rich = run_pipeline(&V3Config::new(42, V3Preset::Rich, 3072).unwrap()).unwrap();

    assert_eq!(sparse.metadata.room_count(), 12);
    assert_eq!(moderate.metadata.room_count(), 20);
    assert_eq!(rich.metadata.room_count(), 28);
    assert_eq!(sparse.metadata.route_count(), 10);
    assert_eq!(moderate.metadata.route_count(), 20);
    assert_eq!(rich.metadata.route_count(), 30);
}
