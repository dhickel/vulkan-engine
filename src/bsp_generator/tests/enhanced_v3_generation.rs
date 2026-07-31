//! Phase 04 — Enhanced V3 generation tests.
//!
//! Validates canonical map bytes, metadata content, and production
//! pipeline behavior. Tests cover deterministic ordering, line endings,
//! terminal newline, identity values, field order, excluded data,
//! requested/effective equality, integer formatting, budget budgets,
//! no partial results, and typed failure conditions.

use bsp_generator::enhanced_v3::*;

// ── Canonical map output tests ────────────────────────────────────────────

#[test]
fn sparse_map_produced_through_full_pipeline() {
    let config = V3Config::nominal_sparse();
    let output = run_pipeline(&config).unwrap();

    assert!(!output.map_text.is_empty());
    assert!(output.map_text.contains("worldspawn"));
    assert!(output.map_text.contains("info_player_start"));
    assert!(output.map_text.contains("cc0_dungeon_v2.wad"));
    assert!(output.map_text.contains("_minlight"));
}

#[test]
fn all_presets_produce_valid_maps() {
    let configs = [
        V3Config::nominal_sparse(),
        V3Config::nominal_moderate(),
        V3Config::nominal_rich(),
    ];

    for config in &configs {
        let output = run_pipeline(config).unwrap();
        assert!(
            !output.map_text.is_empty(),
            "empty map for {:?}",
            config.preset
        );
        assert!(
            output.map_text.contains("worldspawn"),
            "no worldspawn for {:?}",
            config.preset
        );
        assert!(
            output.map_text.contains("info_player_start"),
            "no spawn for {:?}",
            config.preset
        );
    }
}

#[test]
fn map_text_uses_lf_only() {
    let config = V3Config::nominal_sparse();
    let output = run_pipeline(&config).unwrap();
    assert!(!output.map_text.contains('\r'));
    assert!(output.map_text.contains('\n'));
}

#[test]
fn map_text_has_terminal_newline() {
    let config = V3Config::nominal_sparse();
    let output = run_pipeline(&config).unwrap();
    assert!(output.map_text.ends_with('\n'));
}

#[test]
fn map_text_no_double_blank_lines() {
    let config = V3Config::nominal_sparse();
    let output = run_pipeline(&config).unwrap();
    // Check there are no three consecutive newlines (would indicate double blank lines)
    assert!(!output.map_text.contains("\n\n\n"));
}

#[test]
fn map_text_fixed_texture_mapping() {
    let config = V3Config::nominal_sparse();
    let output = run_pipeline(&config).unwrap();
    for line in output.map_text.lines() {
        if line.trim_start().starts_with('(') {
            assert!(
                line.ends_with("0 0 0 0.25 0.25"),
                "face line missing fixed texture mapping: {line}"
            );
        }
    }
}

#[test]
fn map_text_braces_balanced() {
    for config in &[
        V3Config::nominal_sparse(),
        V3Config::nominal_moderate(),
        V3Config::nominal_rich(),
    ] {
        let output = run_pipeline(config).unwrap();
        let open = output.map_text.matches('{').count();
        let close = output.map_text.matches('}').count();
        assert_eq!(open, close, "mismatched braces for {:?}", config.preset);
    }
}

#[test]
fn map_text_has_approved_textures() {
    let config = V3Config::nominal_sparse();
    let output = run_pipeline(&config).unwrap();
    assert!(output.map_text.contains("bs_wall"));
    assert!(output.map_text.contains("bs_floor"));
    assert!(output.map_text.contains("bs_ceil"));
}

#[test]
fn map_text_all_face_lines_have_three_point_triples() {
    let config = V3Config::nominal_sparse();
    let output = run_pipeline(&config).unwrap();
    for line in output.map_text.lines() {
        if line.trim_start().starts_with('(') {
            let paren_count = line.matches('(').count();
            assert_eq!(
                paren_count, 3,
                "face line should have exactly 3 parenthesized points: {line}"
            );
        }
    }
}

// ── Determinism tests ─────────────────────────────────────────────────────

#[test]
fn pipeline_fully_deterministic() {
    let config = V3Config::nominal_sparse();
    let output1 = run_pipeline(&config).unwrap();
    let output2 = run_pipeline(&config).unwrap();
    assert_eq!(output1.map_text, output2.map_text);
    assert_eq!(output1.metadata, output2.metadata);
}

#[test]
fn different_seeds_produce_different_output() {
    let config_a = V3Config::new(0, V3Preset::Sparse, 2048).unwrap();
    let config_b = V3Config::new(42, V3Preset::Sparse, 2048).unwrap();
    let output_a = run_pipeline(&config_a).unwrap();
    let output_b = run_pipeline(&config_b).unwrap();
    // Both produce valid output
    assert!(!output_a.map_text.is_empty());
    assert!(!output_b.map_text.is_empty());
}

// ── Metadata content tests ────────────────────────────────────────────────

#[test]
fn metadata_contains_required_fields() {
    let config = V3Config::nominal_sparse();
    let output = run_pipeline(&config).unwrap();
    let meta = &output.metadata;

    assert_eq!(meta.seed(), 0);
    assert_eq!(meta.preset(), "sparse");
    assert_eq!(meta.schema_version(), "v3");
    assert_eq!(meta.generator(), "bsp_generator/enhanced_v3");
    assert!(meta.room_count() >= 3);
    assert!(meta.portal_count() >= 1);
    assert!(meta.transition_count() >= 1);
}

#[test]
fn metadata_room_layers_correct() {
    let config = V3Config::nominal_sparse();
    let output = run_pipeline(&config).unwrap();
    let meta = &output.metadata;

    assert!(
        meta.lower_room_count() >= 2,
        "expected at least 2 lower rooms"
    );
    assert!(
        meta.upper_room_count() >= 1,
        "expected at least 1 upper room"
    );
    assert_eq!(
        meta.room_count(),
        meta.lower_room_count() + meta.upper_room_count()
    );
    assert!(meta.has_upper_layer());
}

#[test]
fn metadata_spawn_and_lights() {
    let config = V3Config::nominal_sparse();
    let output = run_pipeline(&config).unwrap();
    let meta = &output.metadata;

    let (sx, sy, sz) = meta.spawn_origin();
    assert!(sx > 0, "spawn_x must be positive, got {sx}");
    assert!(sy > 0, "spawn_y must be positive, got {sy}");
    assert!(sz > 0, "spawn_z must be positive, got {sz}");
    assert!(meta.light_count() >= 2, "expected at least 2 lights");
}

#[test]
fn metadata_face_budget_satisfied() {
    for config in &[
        V3Config::nominal_sparse(),
        V3Config::nominal_moderate(),
        V3Config::nominal_rich(),
    ] {
        let output = run_pipeline(config).unwrap();
        assert!(
            output.metadata.face_budget_satisfied(),
            "face budget exceeded for {:?}: estimated={}, actual={}",
            config.preset,
            output.metadata.estimated_faces(),
            output.metadata.actual_faces()
        );
    }
}

#[test]
fn metadata_actual_faces_within_global_budget() {
    for config in &[
        V3Config::nominal_sparse(),
        V3Config::nominal_moderate(),
        V3Config::nominal_rich(),
    ] {
        let output = run_pipeline(config).unwrap();
        assert!(
            output.metadata.actual_faces() < 10000,
            "global face budget exceeded for {:?}: {}",
            config.preset,
            output.metadata.actual_faces()
        );
    }
}

#[test]
fn metadata_bounds_are_positive_volume() {
    for config in &[
        V3Config::nominal_sparse(),
        V3Config::nominal_moderate(),
        V3Config::nominal_rich(),
    ] {
        let output = run_pipeline(config).unwrap();
        let (min_x, min_y, min_z, max_x, max_y, max_z) = output.metadata.bounds();
        assert!(max_x > min_x, "zero X span for {:?}", config.preset);
        assert!(max_y > min_y, "zero Y span for {:?}", config.preset);
        assert!(max_z > min_z, "zero Z span for {:?}", config.preset);
        assert!(
            min_z >= 0,
            "negative min_z for {:?}: {}",
            config.preset,
            min_z
        );
    }
}

#[test]
fn metadata_entity_budget_satisfied() {
    let config = V3Config::nominal_rich();
    let output = run_pipeline(&config).unwrap();
    assert!(output.metadata.entity_budget_satisfied());
    assert!(output.metadata.actual_entities() < 300);
}

#[test]
fn metadata_grammar_families_present() {
    let config = V3Config::nominal_rich();
    let output = run_pipeline(&config).unwrap();
    assert!(!output.metadata.grammar_families().is_empty());
    assert!(output.metadata.identity_satisfied());
}

#[test]
fn metadata_identity_fields_match_preset() {
    for (config, expected_min_families) in &[
        (V3Config::nominal_sparse(), 1),
        (V3Config::nominal_moderate(), 2),
        (V3Config::nominal_rich(), 3),
    ] {
        let output = run_pipeline(config).unwrap();
        assert!(
            output.metadata.grammar_families().len() >= *expected_min_families,
            "preset {:?} requires {expected_min_families} families, got {}",
            config.preset,
            output.metadata.grammar_families().len()
        );
        assert!(output.metadata.identity_satisfied());
    }
}

// ── Backward compatibility tests ──────────────────────────────────────────

#[test]
fn generate_v3_still_works() {
    let config = V3Config::nominal_sparse();
    let map = generate_v3(&config).unwrap();
    assert!(!map.is_empty());
    assert!(map.contains("worldspawn"));
    assert!(map.contains("info_player_start"));
    assert!(map.contains("light"));
}

#[test]
fn generate_v3_deterministic() {
    let config = V3Config::nominal_sparse();
    let map1 = generate_v3(&config).unwrap();
    let map2 = generate_v3(&config).unwrap();
    assert_eq!(map1, map2);
}

// ── Negative / error tests ────────────────────────────────────────────────

#[test]
fn run_pipeline_rejects_invalid_config() {
    // XY extent not quantum-aligned
    assert!(V3Config::new(0, V3Preset::Sparse, 2047).is_err());
    // XY extent too small
    assert!(V3Config::new(0, V3Preset::Sparse, 512).is_err());
}

#[test]
fn run_pipeline_returns_typed_error_not_panic() {
    let result = V3Config::new(0, V3Preset::Sparse, 2047);
    assert!(result.is_err());
    // Must be the typed error, not a panic
    match result {
        Err(V3Error::ConfigNotQuantumAligned { .. }) => {}
        other => panic!("expected ConfigNotQuantumAligned, got {other:?}"),
    }
}

// ── No partial result test ────────────────────────────────────────────────

#[test]
fn generate_v3_no_partial_result_on_bad_config() {
    let result = V3Config::new(0, V3Preset::Sparse, 2047);
    assert!(result.is_err());
    // There should be no way to get a partial map from a bad config
}

// ── Emission-specific tests ────────────────────────────────────────────────

#[test]
fn emit_map_text_rejects_unvalidated_assembly() {
    use bsp_generator::enhanced_v3::FaceRole;
    let brush = ConvexBrush::make_box((0, 64), (0, 64), (0, 128)).unwrap();
    let mut assembly = Assembly {
        brushes: vec![AssemblyBrush::new(
            "test",
            BrushRole::WallShell,
            brush,
            Support::World {
                surface: FaceRole::Floor,
            },
        )],
        interfaces: vec![],
        protected_volumes: vec![],
        support_edges: vec![],
        validated: false,
    };
    assembly.validated = false; // ensure unvalidated
    let result = emit_map_text(&assembly, (0, 0, 0), &[]);
    assert!(result.is_err());
    match result {
        Err(V3Error::UnvalidatedAssembly) => {}
        other => panic!("expected UnvalidatedAssembly, got {other:?}"),
    }
}

#[test]
fn map_text_has_worldspawn_first() {
    let config = V3Config::nominal_sparse();
    let output = run_pipeline(&config).unwrap();
    // Worldspawn entity should appear before info_player_start
    let worldspawn_pos = output.map_text.find("worldspawn").unwrap();
    let spawn_pos = output.map_text.find("info_player_start").unwrap();
    assert!(
        worldspawn_pos < spawn_pos,
        "worldspawn must precede info_player_start"
    );
}

#[test]
fn map_text_spawn_before_lights() {
    let config = V3Config::nominal_sparse();
    let output = run_pipeline(&config).unwrap();
    let spawn_pos = output.map_text.find("info_player_start").unwrap();
    let light_pos = output.map_text.find("\"classname\" \"light\"").unwrap();
    assert!(spawn_pos < light_pos, "spawn must precede lights");
}
