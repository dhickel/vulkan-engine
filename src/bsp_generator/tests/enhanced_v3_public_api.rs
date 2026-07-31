//! Phase 05 — Enhanced V3 public API tests.
//!
//! Validates that the EnhancedV3 types are properly re-exported at the crate
//! root, that `GenerationProfile::from_tag("m3")` works, that legacy and
//! enhanced-v2 profiles are unchanged, and that `generate_enhanced_v3` produces
//! valid output.

use bsp_generator::enhanced::profile::GenerationProfile;
use bsp_generator::enhanced_v3::run_pipeline;
use bsp_generator::{generate_enhanced_v3, V3Config, V3Error, V3PipelineOutput, V3Preset};

// ── Profile dispatch tests ────────────────────────────────────────────────

#[test]
fn m3_tag_is_recognized() {
    assert_eq!(
        GenerationProfile::from_tag("m3"),
        Some(GenerationProfile::EnhancedV3)
    );
    assert_eq!(GenerationProfile::EnhancedV3.tag(), "m3");
}

#[test]
fn enhanced_v3_and_v3_tags_are_unrecognized() {
    assert_eq!(GenerationProfile::from_tag("enhanced-v3"), None);
    assert_eq!(GenerationProfile::from_tag("v3"), None);
}

#[test]
fn legacy_and_enhanced_v2_tags_unchanged() {
    assert_eq!(
        GenerationProfile::from_tag("legacy-v1"),
        Some(GenerationProfile::LegacyV1)
    );
    assert_eq!(GenerationProfile::LegacyV1.tag(), "legacy-v1");

    assert_eq!(
        GenerationProfile::from_tag("enhanced-v2"),
        Some(GenerationProfile::EnhancedV2)
    );
    assert_eq!(GenerationProfile::EnhancedV2.tag(), "enhanced-v2");
}

#[test]
fn profile_roundtrip_includes_v3() {
    for p in [
        GenerationProfile::LegacyV1,
        GenerationProfile::EnhancedV2,
        GenerationProfile::EnhancedV3,
    ] {
        let tag = p.tag();
        let back = GenerationProfile::from_tag(tag).unwrap();
        assert_eq!(p, back);
    }
}

#[test]
fn unknown_profile_tags_still_none() {
    assert_eq!(GenerationProfile::from_tag("legacy"), None);
    assert_eq!(GenerationProfile::from_tag(""), None);
    assert_eq!(GenerationProfile::from_tag("enhanced"), None);
    assert_eq!(GenerationProfile::from_tag("m4"), None);
}

// ── Public API re-export tests ────────────────────────────────────────────

#[test]
fn v3_types_available_at_crate_root() {
    // These must compile — proving public re-exports exist
    let _cfg = V3Config::nominal_sparse();
    let _preset = V3Preset::Sparse;
    let _output: Result<V3PipelineOutput, V3Error>;
}

#[test]
fn generate_enhanced_v3_public_entry_point_works() {
    let config = V3Config::nominal_sparse();
    let (map_text, meta) = generate_enhanced_v3(&config).unwrap();

    assert!(!map_text.is_empty());
    assert!(map_text.contains("worldspawn"));
    assert!(map_text.contains("info_player_start"));
    assert!(map_text.contains("light"));
    assert_eq!(meta.seed(), 0);
    assert_eq!(meta.preset(), "sparse");
    assert_eq!(meta.schema_version(), "v3");
}

#[test]
fn generate_enhanced_v3_deterministic() {
    let config = V3Config::nominal_sparse();
    let (map1, meta1) = generate_enhanced_v3(&config).unwrap();
    let (map2, meta2) = generate_enhanced_v3(&config).unwrap();
    assert_eq!(map1, map2);
    assert_eq!(meta1, meta2);
}

#[test]
fn generate_enhanced_v3_all_presets() {
    for (preset, extent) in &[
        (V3Preset::Sparse, 2048),
        (V3Preset::Moderate, 2048),
        (V3Preset::Rich, 3072),
    ] {
        let config = V3Config::new(0, *preset, *extent).unwrap();
        let (map_text, meta) = generate_enhanced_v3(&config).unwrap();
        assert!(!map_text.is_empty(), "empty map for {:?}", preset);
        assert!(
            map_text.contains("worldspawn"),
            "no worldspawn for {:?}",
            preset
        );
        // Metadata should reflect the preset
        assert_eq!(meta.preset(), preset.tag());
    }
}

#[test]
fn generate_enhanced_v3_with_different_seeds() {
    let config_a = V3Config::new(0, V3Preset::Sparse, 2048).unwrap();
    let config_b = V3Config::new(42, V3Preset::Sparse, 2048).unwrap();
    let (map_a, _) = generate_enhanced_v3(&config_a).unwrap();
    let (map_b, _) = generate_enhanced_v3(&config_b).unwrap();
    // Both produce valid output
    assert!(!map_a.is_empty());
    assert!(!map_b.is_empty());
}

#[test]
fn generate_enhanced_v3_metadata_has_spawn_and_lights() {
    let config = V3Config::nominal_sparse();
    let (_, meta) = generate_enhanced_v3(&config).unwrap();
    let (sx, sy, sz) = meta.spawn_origin();
    assert!(sx > 0);
    assert!(sy > 0);
    assert!(sz > 0);
    assert!(meta.light_count() > 0);
}

#[test]
fn generate_enhanced_v3_metadata_has_layer_info() {
    let config = V3Config::nominal_sparse();
    let (_, meta) = generate_enhanced_v3(&config).unwrap();
    assert!(meta.lower_room_count() >= 2);
    assert!(meta.upper_room_count() >= 1);
    assert_eq!(
        meta.room_count(),
        meta.lower_room_count() + meta.upper_room_count()
    );
    assert!(meta.has_upper_layer());
}

#[test]
fn generate_enhanced_v3_rejects_invalid_config() {
    let result = V3Config::new(0, V3Preset::Sparse, 2047);
    assert!(result.is_err());
    match result {
        Err(V3Error::ConfigNotQuantumAligned { .. }) => {}
        other => panic!("expected ConfigNotQuantumAligned, got {other:?}"),
    }
}

#[test]
fn run_pipeline_still_accessible_through_re_export() {
    let config = V3Config::nominal_sparse();
    let output = run_pipeline(&config).unwrap();
    assert!(!output.map_text.is_empty());
    assert_eq!(output.metadata.seed(), 0);
}

// ── Legacy/enhanced-v2 unchanged tests ────────────────────────────────────

#[test]
fn legacy_generate_still_works() {
    let cfg = bsp_generator::DungeonConfig::nominal_m1();
    let (map, meta) = bsp_generator::generate(0, cfg).unwrap();
    assert!(!map.is_empty());
    assert_eq!(meta.room_count, 12);
}

#[test]
fn enhanced_v2_generate_still_works() {
    let config = bsp_generator::enhanced::config::EnhancedConfig::nominal();
    let (map, meta) = bsp_generator::generate_enhanced(0, config).unwrap();
    assert!(!map.is_empty());
    assert!(meta.room_count > 0);
}
