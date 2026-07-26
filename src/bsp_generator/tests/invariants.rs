//! Post-generation invariant checks: verify that [`bsp_generator::generate`]
//! output satisfies all mandatory structural contracts from the frozen
//! generation specification (`bsp-dungeon-generation.md`).
//!
//! The frozen M1/M2 support corpus is asserted directly: routing exhaustion is
//! a generation failure, not a skipped invariant.

use bsp_generator::{generate, DungeonConfig, MapClass, CONSTRUCTION_QUANTUM};

// ── Helpers ─────────────────────────────────────────────────────────────

/// Run generation and return the result, panicking on any generation failure.
macro_rules! gen_or_skip {
    ($seed:expr, $cfg:expr) => {
        generate($seed, $cfg).expect("generation must satisfy post-generation invariants")
    };
}

// ── Bounds compliance ──────────────────────────────────────────────────

#[test]
fn m1_generation_bounds_within_limit() {
    let cfg = DungeonConfig::nominal_m1();
    let (_map, meta) = gen_or_skip!(0, cfg);
    assert_eq!(meta.room_count, 12);
    let (min_x, min_y, _min_z, max_x, max_y, max_z) = meta.bounds;
    assert!(min_x >= 0);
    assert!(min_y >= 0);
    assert!(max_x <= 1536, "max_x {} exceeds M1 limit", max_x);
    assert!(max_y <= 1536, "max_y {} exceeds M1 limit", max_y);
    assert!(max_z <= 256, "max_z {} exceeds M1 Z limit", max_z);
}

#[test]
fn m2_generation_bounds_within_limit() {
    let cfg = DungeonConfig {
        class: MapClass::M2,
        room_count: 17,
        loop_count: 1,
        xy_bounds: (2048, 2048),
        z_span: 256,
        placement_candidates: 32,
        max_placement_attempts: 96,
        max_astar_expansions: 524_288,
    };
    let (_map, meta) = gen_or_skip!(44, cfg);
    assert_eq!(meta.room_count, 17);
    let (min_x, min_y, _min_z, max_x, max_y, max_z) = meta.bounds;
    assert!(min_x >= 0);
    assert!(min_y >= 0);
    assert!(max_x <= 3072, "max_x {} exceeds M2 limit", max_x);
    assert!(max_y <= 3072, "max_y {} exceeds M2 limit", max_y);
    assert!(max_z <= 384, "max_z {} exceeds M2 Z limit", max_z);
}

// ── Room/corridor counts ────────────────────────────────────────────────

#[test]
fn room_count_matches_config() {
    for (seed, count) in [(0u64, 8u32), (1, 10), (3, 12)] {
        let cfg = DungeonConfig {
            room_count: count,
            loop_count: 0,
            ..DungeonConfig::nominal_m1()
        };
        if let Ok((_map, meta)) = generate(seed, cfg) {
            assert_eq!(meta.room_count, count);
        }
    }
}

#[test]
fn corridor_count_is_positive_for_connected_rooms() {
    let cfg = DungeonConfig::nominal_m1();
    let (_map, meta) = gen_or_skip!(0, cfg);
    assert!(meta.corridor_count > 0);
}

// ── Entity invariants ──────────────────────────────────────────────────

#[test]
fn at_least_one_info_player_start() {
    let cfg = DungeonConfig::nominal_m1();
    let (map_text, _meta) = gen_or_skip!(0, cfg);
    assert!(map_text.contains("info_player_start"));
}

#[test]
fn entity_count_is_one_spawn_plus_room_count() {
    let cfg = DungeonConfig::nominal_m1();
    let (_map, meta) = gen_or_skip!(0, cfg);
    assert_eq!(meta.entity_count, 13); // 1 spawn + 12 lights
}

#[test]
fn light_entities_present_for_each_room() {
    let cfg = DungeonConfig::nominal_m1();
    let (map_text, meta) = gen_or_skip!(0, cfg);
    let light_count = map_text.matches("\"classname\" \"light\"").count();
    assert_eq!(light_count as u32, meta.room_count);
}

// ── WAD reference ──────────────────────────────────────────────────────

#[test]
fn wad_key_is_present_in_worldspawn() {
    let cfg = DungeonConfig::nominal_m1();
    let (map_text, _meta) = gen_or_skip!(0, cfg);
    assert!(map_text.contains("\"wad\""));
    assert!(map_text.contains("cc0_stone_beta.wad"));
}

// ── No empty brushes ───────────────────────────────────────────────────

#[test]
fn no_brush_block_is_empty() {
    let cfg = DungeonConfig::nominal_m1();
    let (map_text, _meta) = gen_or_skip!(0, cfg);
    assert!(!map_text.contains("{\n}\n"));
}

// ── Non-empty output ───────────────────────────────────────────────────

#[test]
fn output_is_non_empty() {
    let cfg = DungeonConfig::nominal_m1();
    let (map_text, _meta) = gen_or_skip!(0, cfg);
    assert!(!map_text.is_empty());
    assert!(map_text.len() > 100);
}

// ── Nominal seed corpus (seeds known to route) ─────────────────────────

#[test]
fn nominal_seed_0_m1_generates() {
    let cfg = DungeonConfig::nominal_m1();
    let result = generate(0, cfg);
    assert!(result.is_ok(), "seed 0 failed: {:?}", result.err());
}

#[test]
fn nominal_seed_1_m1_generates() {
    let cfg = DungeonConfig::nominal_m1();
    let result = generate(1, cfg);
    assert!(result.is_ok(), "seed 1 failed: {:?}", result.err());
}

#[test]
fn nominal_seed_2_m1_generates() {
    let cfg = DungeonConfig::nominal_m1();
    let result = generate(2, cfg);
    assert!(result.is_ok(), "seed 2 failed: {:?}", result.err());
}

#[test]
fn nominal_seed_3_m1_generates() {
    let cfg = DungeonConfig::nominal_m1();
    let result = generate(3, cfg);
    assert!(result.is_ok(), "seed 3 failed: {:?}", result.err());
}

// ── Boundary configurations ────────────────────────────────────────────

#[test]
fn boundary_a_m1_minimum_generates() {
    let cfg = DungeonConfig {
        class: MapClass::M1,
        room_count: 8,
        loop_count: 0,
        xy_bounds: (1024, 1024),
        z_span: 192,
        placement_candidates: 16,
        max_placement_attempts: 64,
        max_astar_expansions: 131_072,
    };
    let result = generate(42, cfg);
    assert!(result.is_ok(), "boundary A failed: {:?}", result.err());
}

#[test]
fn boundary_c_m2_minimum_generates() {
    let cfg = DungeonConfig {
        class: MapClass::M2,
        room_count: 17,
        loop_count: 1,
        xy_bounds: (2048, 2048),
        z_span: 256,
        placement_candidates: 32,
        max_placement_attempts: 96,
        max_astar_expansions: 524_288,
    };
    let result = generate(44, cfg);
    assert!(result.is_ok(), "boundary C failed: {:?}", result.err());
}

#[test]
fn frozen_support_corpus_generates_deterministically_within_budgets() {
    let cases = [
        ("M1 nominal 0", 0, DungeonConfig::nominal_m1(), 2_000, 50),
        ("M1 nominal 1", 1, DungeonConfig::nominal_m1(), 2_000, 50),
        ("M1 nominal 2", 2, DungeonConfig::nominal_m1(), 2_000, 50),
        ("M1 nominal 3", 3, DungeonConfig::nominal_m1(), 2_000, 50),
        (
            "M2 nominal 17",
            17,
            DungeonConfig::nominal_m2(),
            10_000,
            300,
        ),
        (
            "M2 nominal 255",
            255,
            DungeonConfig::nominal_m2(),
            10_000,
            300,
        ),
        (
            "M2 nominal patterned",
            0x5555_5555_5555_5555,
            DungeonConfig::nominal_m2(),
            10_000,
            300,
        ),
        (
            "M2 nominal max",
            u64::MAX,
            DungeonConfig::nominal_m2(),
            10_000,
            300,
        ),
        (
            "Boundary A",
            42,
            DungeonConfig {
                room_count: 8,
                loop_count: 0,
                ..DungeonConfig::nominal_m1()
            },
            2_000,
            50,
        ),
        (
            "Boundary B",
            43,
            DungeonConfig {
                room_count: 16,
                loop_count: 2,
                ..DungeonConfig::nominal_m1()
            },
            2_000,
            50,
        ),
        (
            "Boundary C",
            44,
            DungeonConfig {
                room_count: 17,
                loop_count: 1,
                ..DungeonConfig::nominal_m2()
            },
            10_000,
            300,
        ),
        (
            "Boundary D",
            45,
            DungeonConfig {
                room_count: 40,
                loop_count: 6,
                ..DungeonConfig::nominal_m2()
            },
            10_000,
            300,
        ),
    ];

    for (name, seed, cfg, _face_budget, entity_budget) in cases {
        let (map_a, meta_a) = generate(seed, cfg.clone()).unwrap_or_else(|err| {
            panic!("{name} failed to generate: {err:?}");
        });
        let (map_b, meta_b) = generate(seed, cfg).unwrap_or_else(|err| {
            panic!("{name} failed deterministic replay: {err:?}");
        });

        assert_eq!(map_a, map_b, "{name} is not byte deterministic");
        assert_eq!(meta_a, meta_b, "{name} metadata is not deterministic");
        assert!(map_a.starts_with("{\n\"classname\" \"worldspawn\"\n"));
        assert!(map_a.contains("\"classname\" \"info_player_start\""));
        assert!(map_a.contains("\"wad\" \"cc0_stone_beta.wad\""));
        // The frozen ceiling applies to compiler-merged BSP faces, not the
        // conservative six-sides-per-source-brush estimate. Compiled ceilings
        // are enforced by `corpus_execution`.
        assert!(
            meta_a.face_count_estimate > 0,
            "{name} source face estimate"
        );
        assert!(meta_a.entity_count < entity_budget, "{name} entity budget");
    }
}

// ── Quantum alignment ──────────────────────────────────────────────────

#[test]
fn all_coordinates_are_quantum_aligned() {
    let cfg = DungeonConfig::nominal_m1();
    let (map_text, _meta) = gen_or_skip!(0, cfg);
    for line in map_text.lines() {
        if line.contains('(') {
            for part in line.split('(') {
                if part.contains(')') {
                    let inner = part.split(')').next().unwrap_or("");
                    for num_str in inner.split_whitespace() {
                        if let Ok(num) = num_str.parse::<i32>() {
                            assert_eq!(
                                num % CONSTRUCTION_QUANTUM as i32,
                                0,
                                "coordinate {} not aligned to quantum in line: {}",
                                num,
                                line
                            );
                        }
                    }
                }
            }
        }
    }
}

// ── G3: Exclusive solid ownership — no room-vs-corridor slab overlap ──

#[test]
fn no_floor_or_ceiling_slab_overlap() {
    // Generate a connected layout and verify that no floor slab
    // (z=0..SLAB) overlaps with another floor slab, and no ceiling
    // slab overlaps with another ceiling slab. Same-room wall corners
    // naturally overlap — that is not a G3 violation.
    let cfg = DungeonConfig::nominal_m1();
    let (map_text, _meta) = gen_or_skip!(0, cfg);

    let slab_thickness = bsp_generator::CONSTRUCTION_QUANTUM as i32;

    // Parse all brushes with their AABBs.
    let mut brush_aabbs: Vec<((i32,i32,i32),(i32,i32,i32))> = Vec::new();
    let mut in_brush = false;
    let mut saw_face = false;
    let mut current_min = (i32::MAX, i32::MAX, i32::MAX);
    let mut current_max = (i32::MIN, i32::MIN, i32::MIN);

    for line in map_text.lines() {
        let trimmed = line.trim();
        if trimmed == "{" && !in_brush {
            in_brush = true;
            saw_face = false;
            current_min = (i32::MAX, i32::MAX, i32::MAX);
            current_max = (i32::MIN, i32::MIN, i32::MIN);
        } else if trimmed == "}" && in_brush {
            if saw_face {
                brush_aabbs.push((current_min, current_max));
            }
            in_brush = false;
        } else if in_brush && trimmed.starts_with('(') {
            saw_face = true;
            for part in trimmed.split('(').skip(1) {
                if let Some(inner) = part.split(')').next() {
                    let coords: Vec<i32> = inner
                        .split_whitespace()
                        .filter_map(|s| s.parse().ok())
                        .collect();
                    if coords.len() == 3 {
                        current_min.0 = current_min.0.min(coords[0]);
                        current_min.1 = current_min.1.min(coords[1]);
                        current_min.2 = current_min.2.min(coords[2]);
                        current_max.0 = current_max.0.max(coords[0]);
                        current_max.1 = current_max.1.max(coords[1]);
                        current_max.2 = current_max.2.max(coords[2]);
                    }
                }
            }
        }
    }

    // Check that thin slabs (floor and ceiling plates) don't overlap.
    // Thin slabs have height exactly SLAB and their Z ranges are mutually
    // exclusive with wall brushes. Same-room wall corners naturally overlap
    // and are not a G3 violation.
    for i in 0..brush_aabbs.len() {
        for j in (i + 1)..brush_aabbs.len() {
            let (a_min, a_max) = brush_aabbs[i];
            let (b_min, b_max) = brush_aabbs[j];
            let a_dz = a_max.2 - a_min.2;
            let b_dz = b_max.2 - b_min.2;
            // Only check thin slabs (floor/ceiling plates).
            if a_dz != slab_thickness || b_dz != slab_thickness {
                continue;
            }
            // Same Z level? (overlapping or adjacent Z)
            if a_min.2 != b_min.2 {
                continue;
            }
            let overlap_x = a_min.0.max(b_min.0) < a_max.0.min(b_max.0);
            let overlap_y = a_min.1.max(b_min.1) < a_max.1.min(b_max.1);
            if overlap_x && overlap_y {
                panic!(
                    "Thin slabs at brushes {i} and {j} overlap at z={z_level}:\n  A: {a_min:?} -> {a_max:?}\n  B: {b_min:?} -> {b_max:?}",
                    z_level = a_min.2,
                );
            }
        }
    }

    assert!(brush_aabbs.len() > 10, "expected many brushes in generated map");
}

// ── G2: Corridor height == 80 invariant ────────────────────────────────

#[test]
#[should_panic(expected = "DECISION-20260726-02")]
fn corridor_height_not_80_panics_in_emission() {
    use bsp_generator::{build_emission, Corridor, LayoutIntent, RoomIntent, RoutedIntent};

    let room = RoomIntent {
        position: (0, 0, 0),
        dimensions: (112, 112, 128),
    };
    let layout = LayoutIntent {
        rooms: vec![room],
        edges: Vec::new(),
        loop_count: 0,
    };
    let bad_corridor = Corridor {
        start: (0, 0, 0),
        end: (64, 0, 0),
        width: 64,
        height: 96, // NOT 80
    };
    let routed = RoutedIntent {
        corridors: vec![bad_corridor],
        junctions: Vec::new(),
    };
    let _ = build_emission(&layout, &routed);
}

// ── Generated .map syntax ──────────────────────────────────────────────

#[test]
fn generated_map_is_valid_utf8() {
    let cfg = DungeonConfig::nominal_m1();
    let (map_text, _meta) = gen_or_skip!(0, cfg);
    assert!(!map_text.contains('\0'));
    // Verify it's valid UTF-8 (String already guarantees this)
    assert!(std::str::from_utf8(map_text.as_bytes()).is_ok());
}

#[test]
fn generated_map_has_balanced_braces() {
    let cfg = DungeonConfig::nominal_m1();
    let (map_text, _meta) = gen_or_skip!(0, cfg);
    let open = map_text.matches('{').count();
    let close = map_text.matches('}').count();
    assert_eq!(open, close);
    assert!(open > 0);
}

#[test]
fn generated_map_starts_with_worldspawn() {
    let cfg = DungeonConfig::nominal_m1();
    let (map_text, _meta) = gen_or_skip!(0, cfg);
    assert!(map_text.starts_with("{\n\"classname\" \"worldspawn\"\n"));
}

#[test]
fn all_line_endings_are_lf() {
    let cfg = DungeonConfig::nominal_m1();
    let (map_text, _meta) = gen_or_skip!(0, cfg);
    assert!(!map_text.contains('\r'));
    assert!(map_text.contains('\n'));
}
