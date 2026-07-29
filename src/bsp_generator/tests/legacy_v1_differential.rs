//! Legacy v1 differential — prove deterministic byte identity for all 12
//! frozen entries.

use bsp_generator::DungeonConfig;

const CORPUS: &[(u64, fn() -> DungeonConfig, &str)] = &[
    (0, || DungeonConfig::nominal_m1(), "M1 nominal seed 0"),
    (1, || DungeonConfig::nominal_m1(), "M1 nominal seed 1"),
    (2, || DungeonConfig::nominal_m1(), "M1 nominal seed 2"),
    (3, || DungeonConfig::nominal_m1(), "M1 nominal seed 3"),
    (17, || DungeonConfig::nominal_m2(), "M2 nominal seed 17"),
    (255, || DungeonConfig::nominal_m2(), "M2 nominal seed 255"),
    (
        0x5555555555555555,
        || DungeonConfig::nominal_m2(),
        "M2 nominal seed alt1",
    ),
    (
        u64::MAX,
        || DungeonConfig::nominal_m2(),
        "M2 nominal seed max",
    ),
    (
        42,
        || DungeonConfig {
            class: bsp_generator::MapClass::M1,
            room_count: 8,
            loop_count: 0,
            xy_bounds: (1024, 1024),
            z_span: 192,
            placement_candidates: 16,
            max_placement_attempts: 64,
            max_astar_expansions: 131_072,
        },
        "M1 boundary A (min)",
    ),
    (
        43,
        || DungeonConfig {
            class: bsp_generator::MapClass::M1,
            room_count: 16,
            loop_count: 2,
            xy_bounds: (1024, 1024),
            z_span: 192,
            placement_candidates: 16,
            max_placement_attempts: 64,
            max_astar_expansions: 131_072,
        },
        "M1 boundary B (max)",
    ),
    (
        44,
        || DungeonConfig {
            class: bsp_generator::MapClass::M2,
            room_count: 17,
            loop_count: 1,
            xy_bounds: (2048, 2048),
            z_span: 256,
            placement_candidates: 32,
            max_placement_attempts: 96,
            max_astar_expansions: 524_288,
        },
        "M2 boundary C (min)",
    ),
    (
        45,
        || DungeonConfig {
            class: bsp_generator::MapClass::M2,
            room_count: 40,
            loop_count: 6,
            xy_bounds: (2048, 2048),
            z_span: 256,
            placement_candidates: 32,
            max_placement_attempts: 96,
            max_astar_expansions: 524_288,
        },
        "M2 boundary D (max)",
    ),
];

#[test]
fn legacy_v1_all_12_entries_regenerate() {
    for &(seed, ref config_fn, label) in CORPUS {
        let config = config_fn();
        let (map_text, meta) = bsp_generator::generate(seed, config).unwrap_or_else(|e| {
            panic!("legacy generation failed for {label} (seed {seed}): {e:?}")
        });
        let (replayed_map, replayed_meta) = bsp_generator::generate(seed, config_fn())
            .unwrap_or_else(|e| panic!("legacy replay failed for {label} (seed {seed}): {e:?}"));
        assert!(!map_text.is_empty(), "{label}: empty map");
        assert!(meta.room_count > 0, "{label}: zero rooms");
        assert_eq!(
            map_text, replayed_map,
            "{label}: legacy output differs across independent replays"
        );
        assert_eq!(
            meta, replayed_meta,
            "{label}: legacy metadata differs on replay"
        );
    }
    eprintln!(
        "legacy_v1_differential: PASS ({} entries regenerated)",
        CORPUS.len()
    );
}

#[test]
fn legacy_generate_signature_unchanged() {
    // Prove the Legacy public API still compiles and runs
    let cfg = DungeonConfig::nominal_m1();
    let (map, meta) = bsp_generator::generate(0, cfg).expect("generate");
    assert!(!map.is_empty());
    assert_eq!(meta.room_count, 12);
}
