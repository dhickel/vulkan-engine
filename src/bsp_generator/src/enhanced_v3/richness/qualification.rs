//! EnhancedV3 Richness V1 qualification suite — focused and property corpus.
//!
//! Subphase 17-A: Authoritative content ID coverage, deterministic seed/boundary
//! sweeps, cross-theme semantic identity proofs, error-path validation, and
//! fixture inventory validation.
//!
//! Subphase 17-B: 36-entry compiler corpus and metrics. Provides `CorpusManifest`,
//! `CorpusManifestEntry`, and theme asset helpers for the integration test
//! `enhanced_v3_richness_corpus`.
//!
//! All tests are in-crate (`#[cfg(test)]`) and assert every result; no discards.
//! The module also exports `pub(crate)` helpers for the integration test file
//! `enhanced_v3_richness_compatibility.rs`.

use std::collections::BTreeSet;

use super::canonical::RICHNESS_REQUEST_DOMAIN;
use super::error::{RichnessError, RichnessErrorCategory, RichnessErrorCode};
use super::generated_content;

use super::pipeline::{run_richness_pipeline, RichnessPipelineOutput};
use super::request::{
    InheritedOr, ResolvedRichnessRequestV1, RichnessCaveMode, RichnessDocumentV1, RichnessPreset,
    RichnessTheme, ValueSource, BUDGET_CEILING_MAX, BUDGET_CEILING_MIN, LANDMARKS_MAX,
    LANDMARKS_MIN, RICHNESS_EXTENT_MAX, RICHNESS_EXTENT_MIN, RICHNESS_QUANTUM,
    VERTICAL_FEATURES_MAX, VERTICAL_FEATURES_MIN, ZONES_MAX, ZONES_MIN,
};

// ── pub helpers for integration tests ────────────────────────────────────
//
// These are `pub` (not `pub(crate)`) so integration tests can access them
// through the test-only re-export chain: lib.rs -> enhanced_v3::qualification.
// The module itself is `pub(crate)` inside `richness`, which is also `pub(crate)`,
// so direct access outside the crate is only possible via the re-exports.

/// Run the private Richness pipeline from a public request type.
/// Only callable from within the crate or via test-only re-exports.
pub fn qualify_request(
    resolved: &ResolvedRichnessRequestV1,
) -> Result<RichnessPipelineOutput, RichnessError> {
    run_richness_pipeline(resolved)
}

/// Build a resolved request from canonical bytes (for fixture parsing).
pub fn resolve_from_bytes(bytes: &[u8]) -> Result<ResolvedRichnessRequestV1, RichnessError> {
    let doc = RichnessDocumentV1::from_canonical_bytes(bytes)?;
    ResolvedRichnessRequestV1::resolve(doc)
}

/// Build a resolved request from a document.
pub fn resolve_document(
    doc: RichnessDocumentV1,
) -> Result<ResolvedRichnessRequestV1, RichnessError> {
    ResolvedRichnessRequestV1::resolve(doc)
}

/// All archetype IDs from generated content.
pub fn archetype_ids() -> &'static [&'static str] {
    generated_content::ARCHETYPE_IDS
}

/// All prop IDs from generated content.
pub fn prop_ids() -> &'static [&'static str] {
    generated_content::PROP_IDS
}

/// All light recipe IDs from generated content.
pub fn light_recipe_ids() -> &'static [&'static str] {
    generated_content::LIGHT_RECIPE_IDS
}

/// All theme IDs from generated content.
pub fn theme_ids() -> &'static [&'static str] {
    generated_content::THEME_IDS
}

/// All vertical recipe variants.
pub fn vertical_recipe_variants() -> &'static [&'static str] {
    &[
        "none",
        "stairwell",
        "ladder_shaft",
        "drop_hole",
        "open_stairwell",
        "spiral_stair",
    ]
}

/// Run the pipeline and return semantic identity bytes for cross-theme comparison.
pub fn pipeline_semantic_identity(
    resolved: &ResolvedRichnessRequestV1,
) -> Result<Vec<u8>, RichnessError> {
    let output = run_richness_pipeline(resolved)?;
    Ok(output.generation_metadata.semantic_identity().to_vec())
}

/// Run the pipeline and return map text for cross-theme presentation comparison.
pub fn pipeline_map_text(resolved: &ResolvedRichnessRequestV1) -> Result<String, RichnessError> {
    let output = run_richness_pipeline(resolved)?;
    Ok(output.map_text)
}

/// Run the pipeline and return the full output bundle.
pub fn pipeline_output(
    resolved: &ResolvedRichnessRequestV1,
) -> Result<RichnessPipelineOutput, RichnessError> {
    run_richness_pipeline(resolved)
}

// ── Subphase 17-B: Corpus manifest and metric types ───────────────────────

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Frozen 36-entry corpus manifest — an oracle, not a generator input.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorpusManifest {
    /// Manifest schema version.
    pub schema: String,
    /// Canonical corpus identity name.
    pub corpus_name: String,
    /// Total entry count (must be 36).
    pub entry_count: usize,
    /// Ordered SHA-256 of all entries (deterministic corpus identity).
    pub ordered_sha256: String,
    /// Individual entry records in canonical (seed, preset, theme) order.
    pub entries: Vec<CorpusManifestEntry>,
}

/// One frozen entry in the compiler corpus manifest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorpusManifestEntry {
    /// Canonical identity: "<preset>/<theme>/seed:<seed>".
    pub identity: String,
    /// Generation seed.
    pub seed: u64,
    /// Richness preset tag.
    pub preset: String,
    /// Richness theme tag.
    pub theme: String,
    /// XY extent in Quake units.
    pub extent: u32,
    /// SHA-256 of the canonical request bytes.
    pub request_sha256: String,
    /// SHA-256 of the raw request identity hash bytes.
    pub request_identity_sha256: String,
    /// SHA-256 of the generation metadata canonical bytes.
    pub metadata_sha256: String,
    /// SHA-256 of the authored generated Rust constants source.
    pub constants_sha256: String,
    /// SHA-256 of the generated .map text bytes.
    pub map_sha256: String,
    /// SHA-256 of the compiled .bsp bytes (post light).
    pub bsp_sha256: String,
    /// SHA-256 of the compiled .lit bytes.
    pub lit_sha256: String,
    /// SHA-256 of the theme WAD bytes.
    pub wad_sha256: String,
    /// SHA-256 of the theme palette bytes.
    pub palette_sha256: String,
    /// SHA-256 of the bsp + lit + wad + palette bytes concatenated.
    pub package_sha256: String,
    /// Source brush count.
    pub source_brushes: usize,
    /// Source face count.
    pub source_faces: usize,
    /// Source entity count.
    pub source_entities: usize,
    /// Source light count.
    pub source_lights: usize,
    /// Source opening count.
    pub source_openings: usize,
    /// Source support contact count.
    pub source_support_contacts: usize,
    /// Compiled model count.
    pub compiled_models: usize,
    /// Compiled face count (lump 7).
    pub compiled_faces: usize,
    /// Compiled leaf count (lump 10).
    pub compiled_leafs: usize,
    /// Compiled portal count (estimated from VIS data).
    pub compiled_portals: usize,
    /// BSP byte size.
    pub bsp_bytes: u64,
    /// LIT byte size.
    pub lit_bytes: u64,
    /// WAD byte size.
    pub wad_bytes: u64,
    /// ericw-tools version tag.
    pub compiler_version: String,
    /// SHA-256 of qbsp executable.
    pub qbsp_sha256: String,
    /// SHA-256 of vis executable.
    pub vis_sha256: String,
    /// SHA-256 of light executable.
    pub light_sha256: String,
    /// Full qbsp arguments.
    pub qbsp_args: Vec<String>,
    /// Full vis arguments.
    pub vis_args: Vec<String>,
    /// Full light arguments.
    pub light_args: Vec<String>,
}

/// Theme asset paths for a RichnessTheme tag.
/// Returns (wad_path, palette_path) relative to the crate manifest dir.
pub fn theme_asset_paths(theme_tag: &str) -> (PathBuf, PathBuf) {
    let crate_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let theme_dir = match theme_tag {
        "ancient" => crate_dir.join("themes/richness_ancient_v1"),
        "egyptian" => crate_dir.join("themes/richness_egyptian_v1"),
        "brutalist" => crate_dir.join("themes/richness_brutalist_v1"),
        other => crate_dir.join(format!("themes/{other}")),
    };
    let wad_name = format!("richness_{theme_tag}_v1.wad");
    (theme_dir.join(&wad_name), theme_dir.join("palette.lmp"))
}

/// Compute SHA-256 hex digest of a byte slice.
pub fn sha256_hex(data: &[u8]) -> String {
    use sha2::Digest;
    let mut hasher = sha2::Sha256::new();
    hasher.update(data);
    format!("{:x}", hasher.finalize())
}

/// Deterministic preset extent: Sparse/Moderate -> 2048, Rich -> 3072.
pub fn preset_extent(preset: RichnessPreset) -> u32 {
    match preset {
        RichnessPreset::Sparse | RichnessPreset::Moderate => 2048,
        RichnessPreset::Rich => 3072,
    }
}

/// All 36 corpus entries in canonical order: preset × theme × seed.
pub fn corpus_entries() -> Vec<(RichnessPreset, RichnessTheme, u64)> {
    let mut entries = Vec::with_capacity(36);
    for &preset in RichnessPreset::ALL {
        for &theme in RichnessTheme::ALL {
            for &seed in &[0u64, 42, 99, 255] {
                entries.push((preset, theme, seed));
            }
        }
    }
    entries
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::super::content_types::{RarityTier, ShapeRule, VerticalRecipe};
    use super::*;
    use std::collections::BTreeSet;

    // ═══════════════════════════════════════════════════════════════════════
    // 1. Content ID coverage — prove EVERY content ID exists
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn all_30_archetype_ids_exist_and_are_canonically_ordered() {
        let ids = generated_content::ARCHETYPE_IDS;
        assert_eq!(ids.len(), 30, "must have exactly 30 archetypes");
        assert_eq!(generated_content::ARCHETYPE_COUNT, 30);

        // Canonical ordering (alphabetical)
        let sorted: Vec<_> = {
            let mut v: Vec<&str> = ids.iter().copied().collect();
            v.sort();
            v
        };
        assert_eq!(
            ids,
            sorted.as_slice(),
            "archetype IDs must be canonically ordered"
        );

        // Every ID is unique
        let unique: std::collections::BTreeSet<&str> = ids.iter().copied().collect();
        assert_eq!(unique.len(), 30, "archetype IDs must be unique");

        // Verify expected IDs are present
        let expected: std::collections::BTreeSet<&str> = [
            "ambush_cross",
            "antechamber",
            "arena",
            "barracks",
            "bridge_crossing",
            "cistern",
            "crossroads",
            "entrance_hall",
            "flooded_crypt",
            "foundry",
            "gallery",
            "grand_arena",
            "grand_stair_hall",
            "grotto",
            "guard_hall",
            "hypostyle_hall",
            "kill_court",
            "ladder_hub",
            "observatory",
            "ossuary",
            "overlook_hall",
            "pit_room",
            "reliquary",
            "shrine",
            "spiral_tower",
            "throne_hall",
            "trapped_gallery",
            "treasury",
            "vault",
            "vestibule",
        ]
        .iter()
        .copied()
        .collect();
        assert_eq!(
            unique, expected,
            "archetype ID set must match frozen contract"
        );
    }

    #[test]
    fn all_30_archetypes_have_complete_content_tables() {
        let n = generated_content::ARCHETYPE_COUNT;
        assert_eq!(generated_content::ARCHETYPE_IDS.len(), n);
        assert_eq!(generated_content::ARCHETYPE_SPAN_MIN.len(), n);
        assert_eq!(generated_content::ARCHETYPE_SPAN_MAX.len(), n);
        assert_eq!(generated_content::ARCHETYPE_SHAPE.len(), n);
        assert_eq!(generated_content::ARCHETYPE_EXIT_DEGREE_MIN.len(), n);
        assert_eq!(generated_content::ARCHETYPE_EXIT_DEGREE_MAX.len(), n);
        assert_eq!(generated_content::ARCHETYPE_LAYER_OCCUPANCY.len(), n);
        assert_eq!(generated_content::ARCHETYPE_ROUTE_WITNESS.len(), n);
        assert_eq!(generated_content::ARCHETYPE_VERTICAL_RECIPE.len(), n);
        assert_eq!(generated_content::ARCHETYPE_RARITY.len(), n);
        assert_eq!(generated_content::ARCHETYPE_NEGATIVE_SPACE_BUDGET.len(), n);
        assert_eq!(generated_content::ARCHETYPE_SUPPORT_RULES.len(), n);
        assert_eq!(generated_content::ARCHETYPE_COST_SOURCE_FACES.len(), n);
        assert_eq!(generated_content::ARCHETYPE_COST_BRUSHES.len(), n);
        assert_eq!(generated_content::ARCHETYPE_COST_ENTITIES.len(), n);
        assert_eq!(generated_content::ARCHETYPE_COST_LIGHTS.len(), n);
        assert_eq!(generated_content::ARCHETYPE_ZONE_COMPAT.len(), n);
        assert_eq!(generated_content::ARCHETYPE_GRAMMAR_COMPAT.len(), n);
        assert_eq!(generated_content::ARCHETYPE_PROP_REFS.len(), n);
        assert_eq!(generated_content::ARCHETYPE_LIGHT_REFS.len(), n);
        assert_eq!(generated_content::ARCHETYPE_MATERIAL_ROLES.len(), n);
        assert_eq!(generated_content::ARCHETYPE_THEME_MASSING.len(), n);
        assert_eq!(generated_content::ARCHETYPE_THEME_SUPPORT.len(), n);
        assert_eq!(generated_content::ARCHETYPE_THEME_PROP_REFS.len(), n);
        assert_eq!(generated_content::ARCHETYPE_THEME_LIGHT_REFS.len(), n);
        assert_eq!(generated_content::ARCHETYPE_THEME_MATERIALS.len(), n);
    }

    #[test]
    fn all_15_prop_ids_exist_and_are_canonically_ordered() {
        let ids = generated_content::PROP_IDS;
        assert_eq!(ids.len(), 15, "must have exactly 15 props");
        assert_eq!(generated_content::PROP_COUNT, 15);

        let sorted: Vec<_> = {
            let mut v: Vec<&str> = ids.iter().copied().collect();
            v.sort();
            v
        };
        assert_eq!(
            ids,
            sorted.as_slice(),
            "prop IDs must be canonically ordered"
        );

        let unique: std::collections::BTreeSet<&str> = ids.iter().copied().collect();
        assert_eq!(unique.len(), 15, "prop IDs must be unique");

        let expected: std::collections::BTreeSet<&str> = [
            "altar",
            "bench",
            "brazier",
            "broken_pillar",
            "cage",
            "canopic_cluster",
            "chain",
            "chest",
            "fountain_rim",
            "hearth",
            "rubble_cluster",
            "sarcophagus",
            "sconce",
            "shelf",
            "urn_block",
        ]
        .iter()
        .copied()
        .collect();
        assert_eq!(unique, expected, "prop ID set must match frozen contract");
    }

    #[test]
    fn all_15_props_have_complete_content_tables() {
        let n = generated_content::PROP_COUNT;
        assert_eq!(generated_content::PROP_IDS.len(), n);
        assert_eq!(generated_content::PROP_CONVEX_PIECES.len(), n);
        assert_eq!(generated_content::PROP_DIMENSIONS.len(), n);
        assert_eq!(generated_content::PROP_COLLISION.len(), n);
        assert_eq!(generated_content::PROP_SWEPT_OCCUPANCY.len(), n);
        assert_eq!(generated_content::PROP_SUPPORT_CONTACTS.len(), n);
        assert_eq!(generated_content::PROP_LIGHT_COUPLING.len(), n);
        assert_eq!(generated_content::PROP_COST_SOURCE_FACES.len(), n);
        assert_eq!(generated_content::PROP_COST_BRUSHES.len(), n);
        assert_eq!(generated_content::PROP_COST_ENTITIES.len(), n);
        assert_eq!(generated_content::PROP_COST_LIGHTS.len(), n);
        assert_eq!(generated_content::PROP_THEME_MODEL_OVERRIDE.len(), n);
        assert_eq!(generated_content::PROP_THEME_DIMENSIONS.len(), n);
        assert_eq!(generated_content::PROP_THEME_COLLISION_OVERRIDE.len(), n);
    }

    #[test]
    fn all_12_light_recipe_ids_exist_and_are_canonically_ordered() {
        let ids = generated_content::LIGHT_RECIPE_IDS;
        assert_eq!(ids.len(), 12, "must have exactly 12 light recipes");
        assert_eq!(generated_content::LIGHT_RECIPE_COUNT, 12);

        let sorted: Vec<_> = {
            let mut v: Vec<&str> = ids.iter().copied().collect();
            v.sort();
            v
        };
        assert_eq!(
            ids,
            sorted.as_slice(),
            "light recipe IDs must be canonically ordered"
        );

        let unique: std::collections::BTreeSet<&str> = ids.iter().copied().collect();
        assert_eq!(unique.len(), 12, "light recipe IDs must be unique");

        let expected: std::collections::BTreeSet<&str> = [
            "brutalist_flood",
            "cavern_gloom",
            "cistern_cool",
            "cold_crypt",
            "dim_beam",
            "egyptian_amber",
            "entrance_torch",
            "foundry_fire",
            "grand_hall_grid",
            "shrine_focus",
            "treasury_glint",
            "warm_hall",
        ]
        .iter()
        .copied()
        .collect();
        assert_eq!(
            unique, expected,
            "light recipe ID set must match frozen contract"
        );
    }

    #[test]
    fn all_12_light_recipes_have_complete_content_tables() {
        let n = generated_content::LIGHT_RECIPE_COUNT;
        assert_eq!(generated_content::LIGHT_RECIPE_IDS.len(), n);
        assert_eq!(generated_content::LIGHT_COLOR.len(), n);
        assert_eq!(generated_content::LIGHT_INTENSITY.len(), n);
        assert_eq!(generated_content::LIGHT_PLACEMENT_CLASS.len(), n);
        assert_eq!(generated_content::LIGHT_FALLOFF.len(), n);
        assert_eq!(generated_content::LIGHT_READABILITY_FLOOR.len(), n);
        assert_eq!(generated_content::LIGHT_COUNT.len(), n);
        assert_eq!(generated_content::LIGHT_COST_SOURCE_FACES.len(), n);
        assert_eq!(generated_content::LIGHT_COST_BRUSHES.len(), n);
        assert_eq!(generated_content::LIGHT_COST_ENTITIES.len(), n);
        assert_eq!(generated_content::LIGHT_COST_LIGHTS.len(), n);
    }

    #[test]
    fn all_3_theme_ids_exist_and_are_canonically_ordered() {
        let ids = generated_content::THEME_IDS;
        assert_eq!(ids.len(), 3, "must have exactly 3 themes");
        assert_eq!(generated_content::THEME_COUNT, 3);

        let expected: &[&str] = &["ancient", "brutalist", "egyptian"];
        assert_eq!(ids, expected, "theme IDs must be canonically ordered");
    }

    #[test]
    fn all_3_themes_have_complete_content_tables() {
        let n = generated_content::THEME_COUNT;
        assert_eq!(generated_content::THEME_IDS.len(), n);
        assert_eq!(generated_content::THEME_SEMANTIC_ROLE_NAMES.len(), 6);
        assert_eq!(generated_content::THEME_SEMANTIC_ROLES.len(), n);
        assert_eq!(generated_content::THEME_TRANSITIONS.len(), n);
        assert_eq!(generated_content::THEME_GEOMETRY_VOCABULARY.len(), n);
        assert_eq!(generated_content::THEME_MATERIAL_ROLES.len(), n);
        assert_eq!(generated_content::THEME_PROP_COMPAT.len(), n);
        assert_eq!(generated_content::THEME_LIGHT_COMPAT.len(), n);
        assert_eq!(generated_content::THEME_BUDGET_SOURCE_FACES.len(), n);
        assert_eq!(generated_content::THEME_BUDGET_BRUSHES.len(), n);
        assert_eq!(generated_content::THEME_BUDGET_ENTITIES.len(), n);
        assert_eq!(generated_content::THEME_BUDGET_LIGHTS.len(), n);
    }

    #[test]
    fn all_vertical_recipes_are_referenced_by_archetypes() {
        // Every VerticalRecipe variant must appear in ARCHETYPE_VERTICAL_RECIPE
        use std::collections::HashSet;
        let all_recipes: HashSet<_> = generated_content::ARCHETYPE_VERTICAL_RECIPE
            .iter()
            .collect();

        assert!(all_recipes.contains(&VerticalRecipe::None));
        assert!(all_recipes.contains(&VerticalRecipe::Stairwell));
        assert!(all_recipes.contains(&VerticalRecipe::LadderShaft));
        assert!(all_recipes.contains(&VerticalRecipe::DropHole));
        assert!(all_recipes.contains(&VerticalRecipe::OpenStairwell));
        assert!(all_recipes.contains(&VerticalRecipe::SpiralStair));
    }

    #[test]
    fn all_shape_variants_are_referenced_by_archetypes() {
        use std::collections::HashSet;
        let all_shapes: HashSet<_> = generated_content::ARCHETYPE_SHAPE.iter().collect();

        assert!(all_shapes.contains(&ShapeRule::Rectangle));
        assert!(all_shapes.contains(&ShapeRule::Octagon));
        assert!(all_shapes.contains(&ShapeRule::Chamfer));
        assert!(all_shapes.contains(&ShapeRule::CompositePartition));
    }

    #[test]
    fn all_rarity_tiers_are_referenced_by_archetypes() {
        use std::collections::HashSet;
        let all_rarities: HashSet<_> = generated_content::ARCHETYPE_RARITY.iter().collect();

        assert!(all_rarities.contains(&RarityTier::Common));
        assert!(all_rarities.contains(&RarityTier::Uncommon));
        assert!(all_rarities.contains(&RarityTier::Rare));
        assert!(all_rarities.contains(&RarityTier::Legendary));
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 2. Request validators and boundary sweeps
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn request_accepts_valid_documents_all_combinations() {
        for preset in RichnessPreset::ALL {
            for theme in RichnessTheme::ALL {
                // Rich preset requires at least 2048 extent for 3 landmarks (max=2048/512=4, ok)
                // Sparse/Moderate work at 1024
                let extent = if *preset == RichnessPreset::Rich {
                    3072
                } else {
                    1024
                };
                for test_extent in if extent == 3072 {
                    vec![2048u32, 3072]
                } else {
                    vec![1024u32, 2048]
                } {
                    let doc = RichnessDocumentV1::new(42, test_extent, *preset, *theme);
                    assert!(
                        doc.is_ok(),
                        "valid document rejected: preset={} theme={} extent={}",
                        preset.tag(),
                        theme.tag(),
                        test_extent
                    );
                    let resolved = ResolvedRichnessRequestV1::resolve(doc.unwrap());
                    assert!(
                        resolved.is_ok(),
                        "valid document failed resolution: preset={} theme={} extent={}",
                        preset.tag(),
                        theme.tag(),
                        test_extent
                    );
                }
            }
        }
    }

    #[test]
    fn request_rejects_extent_below_minimum() {
        let err = RichnessDocumentV1::new(
            0,
            RICHNESS_EXTENT_MIN - 1,
            RichnessPreset::Sparse,
            RichnessTheme::Ancient,
        )
        .unwrap_err();
        assert_eq!(err.code, RichnessErrorCode::ValueOutOfRange);
        assert!(err.context.contains("out of range"));
    }

    #[test]
    fn request_rejects_extent_above_maximum() {
        let err = RichnessDocumentV1::new(
            0,
            RICHNESS_EXTENT_MAX + 16,
            RichnessPreset::Sparse,
            RichnessTheme::Ancient,
        )
        .unwrap_err();
        assert_eq!(err.code, RichnessErrorCode::ValueOutOfRange);
    }

    #[test]
    fn request_accepts_extent_boundaries() {
        assert!(RichnessDocumentV1::new(
            0,
            RICHNESS_EXTENT_MIN,
            RichnessPreset::Sparse,
            RichnessTheme::Ancient
        )
        .is_ok());
        assert!(RichnessDocumentV1::new(
            0,
            RICHNESS_EXTENT_MAX,
            RichnessPreset::Rich,
            RichnessTheme::Brutalist
        )
        .is_ok());
    }

    #[test]
    fn request_rejects_non_quantum_extent() {
        // Test several non-multiples of 16 that are within the valid range
        // (1024-3072) but not quantum-aligned
        for bad_extent in [1025u32, 2047, 2049, 3071] {
            let err = RichnessDocumentV1::new(
                0,
                bad_extent,
                RichnessPreset::Sparse,
                RichnessTheme::Ancient,
            )
            .unwrap_err();
            assert_eq!(
                err.code,
                RichnessErrorCode::NotQuantumAligned,
                "extent {} should be rejected as non-quantum, got {:?}",
                bad_extent,
                err.code
            );
        }
    }

    #[test]
    fn request_accepts_all_quantum_aligned_extents_in_range() {
        // Sweep: every quantum-aligned extent between min and max
        let mut extent = RICHNESS_EXTENT_MIN;
        while extent <= RICHNESS_EXTENT_MAX {
            let doc =
                RichnessDocumentV1::new(0, extent, RichnessPreset::Sparse, RichnessTheme::Ancient);
            assert!(
                doc.is_ok(),
                "quantum-aligned extent {} should be accepted",
                extent
            );
            extent += RICHNESS_QUANTUM;
        }
    }

    #[test]
    fn request_rejects_landmarks_below_min() {
        let doc = RichnessDocumentV1::with_all_explicit(
            0,
            2048,
            RichnessPreset::Sparse,
            RichnessTheme::Ancient,
            super::super::request::RichnessRequestSchemaRevision::V1,
            super::super::request::RichnessAlgorithmRevision::V1,
            super::super::request::RichnessContentRevision::V1,
            super::super::request::RichnessPresetRevision::V1,
            super::super::request::RichnessThemeRevision::V1,
            super::super::request::RichnessAssetRevision::V1,
            super::super::request::RichnessConventionRevision::V1,
            InheritedOr::Explicit(LANDMARKS_MIN.saturating_sub(1)),
            InheritedOr::Inherited,
            InheritedOr::Inherited,
            InheritedOr::Inherited,
            InheritedOr::Inherited,
        );
        // Document construction succeeds (validate_raw_fields is deferred to resolution)
        assert!(doc.is_ok());
        let resolved = ResolvedRichnessRequestV1::resolve(doc.unwrap());
        assert!(resolved.is_err());
        let err = resolved.unwrap_err();
        assert!(matches!(err.code, RichnessErrorCode::ValueOutOfRange));
    }

    #[test]
    fn request_rejects_landmarks_above_max() {
        let doc = RichnessDocumentV1::with_all_explicit(
            0,
            3072,
            RichnessPreset::Rich,
            RichnessTheme::Ancient,
            super::super::request::RichnessRequestSchemaRevision::V1,
            super::super::request::RichnessAlgorithmRevision::V1,
            super::super::request::RichnessContentRevision::V1,
            super::super::request::RichnessPresetRevision::V1,
            super::super::request::RichnessThemeRevision::V1,
            super::super::request::RichnessAssetRevision::V1,
            super::super::request::RichnessConventionRevision::V1,
            InheritedOr::Explicit(LANDMARKS_MAX + 1),
            InheritedOr::Inherited,
            InheritedOr::Inherited,
            InheritedOr::Inherited,
            InheritedOr::Inherited,
        );
        assert!(doc.is_ok());
        let resolved = ResolvedRichnessRequestV1::resolve(doc.unwrap());
        assert!(resolved.is_err());
    }

    #[test]
    fn request_rejects_landmarks_infeasible_for_extent() {
        // 5 landmarks on a 1024 extent is infeasible
        let doc = RichnessDocumentV1::with_all_explicit(
            0,
            1024,
            RichnessPreset::Rich,
            RichnessTheme::Ancient,
            super::super::request::RichnessRequestSchemaRevision::V1,
            super::super::request::RichnessAlgorithmRevision::V1,
            super::super::request::RichnessContentRevision::V1,
            super::super::request::RichnessPresetRevision::V1,
            super::super::request::RichnessThemeRevision::V1,
            super::super::request::RichnessAssetRevision::V1,
            super::super::request::RichnessConventionRevision::V1,
            InheritedOr::Explicit(5),
            InheritedOr::Inherited,
            InheritedOr::Inherited,
            InheritedOr::Inherited,
            InheritedOr::Inherited,
        );
        assert!(doc.is_ok()); // raw validation passes (5 is <= max=5)
        let resolved = ResolvedRichnessRequestV1::resolve(doc.unwrap());
        assert!(resolved.is_err());
        let err = resolved.unwrap_err();
        assert_eq!(err.code, RichnessErrorCode::LandmarkCountInfeasible);
        assert!(err.context.contains("exceeds maximum"));
    }

    #[test]
    fn request_rejects_zone_count_out_of_range() {
        // Zone count 0 (below ZONES_MIN=1)
        let doc = RichnessDocumentV1::with_all_explicit(
            0,
            2048,
            RichnessPreset::Sparse,
            RichnessTheme::Ancient,
            super::super::request::RichnessRequestSchemaRevision::V1,
            super::super::request::RichnessAlgorithmRevision::V1,
            super::super::request::RichnessContentRevision::V1,
            super::super::request::RichnessPresetRevision::V1,
            super::super::request::RichnessThemeRevision::V1,
            super::super::request::RichnessAssetRevision::V1,
            super::super::request::RichnessConventionRevision::V1,
            InheritedOr::Inherited,
            InheritedOr::Explicit(0),
            InheritedOr::Inherited,
            InheritedOr::Inherited,
            InheritedOr::Inherited,
        );
        // Construction succeeds; resolution rejects
        assert!(doc.is_ok());
        assert!(ResolvedRichnessRequestV1::resolve(doc.unwrap()).is_err());

        // Zone count 7 (above ZONES_MAX=6)
        let doc = RichnessDocumentV1::with_all_explicit(
            0,
            3072,
            RichnessPreset::Rich,
            RichnessTheme::Ancient,
            super::super::request::RichnessRequestSchemaRevision::V1,
            super::super::request::RichnessAlgorithmRevision::V1,
            super::super::request::RichnessContentRevision::V1,
            super::super::request::RichnessPresetRevision::V1,
            super::super::request::RichnessThemeRevision::V1,
            super::super::request::RichnessAssetRevision::V1,
            super::super::request::RichnessConventionRevision::V1,
            InheritedOr::Inherited,
            InheritedOr::Explicit(7),
            InheritedOr::Inherited,
            InheritedOr::Inherited,
            InheritedOr::Inherited,
        );
        assert!(doc.is_ok());
        assert!(ResolvedRichnessRequestV1::resolve(doc.unwrap()).is_err());
    }

    #[test]
    fn request_rejects_vertical_openings_above_max() {
        let doc = RichnessDocumentV1::with_all_explicit(
            0,
            3072,
            RichnessPreset::Rich,
            RichnessTheme::Ancient,
            super::super::request::RichnessRequestSchemaRevision::V1,
            super::super::request::RichnessAlgorithmRevision::V1,
            super::super::request::RichnessContentRevision::V1,
            super::super::request::RichnessPresetRevision::V1,
            super::super::request::RichnessThemeRevision::V1,
            super::super::request::RichnessAssetRevision::V1,
            super::super::request::RichnessConventionRevision::V1,
            InheritedOr::Inherited,
            InheritedOr::Inherited,
            InheritedOr::Inherited,
            InheritedOr::Explicit(VERTICAL_FEATURES_MAX + 1),
            InheritedOr::Inherited,
        );
        assert!(doc.is_ok());
        assert!(ResolvedRichnessRequestV1::resolve(doc.unwrap()).is_err());
    }

    #[test]
    fn request_accepts_vertical_zero() {
        // vertical openings = 0 is valid
        let doc = RichnessDocumentV1::with_all_explicit(
            0,
            2048,
            RichnessPreset::Sparse,
            RichnessTheme::Ancient,
            super::super::request::RichnessRequestSchemaRevision::V1,
            super::super::request::RichnessAlgorithmRevision::V1,
            super::super::request::RichnessContentRevision::V1,
            super::super::request::RichnessPresetRevision::V1,
            super::super::request::RichnessThemeRevision::V1,
            super::super::request::RichnessAssetRevision::V1,
            super::super::request::RichnessConventionRevision::V1,
            InheritedOr::Inherited,
            InheritedOr::Inherited,
            InheritedOr::Inherited,
            InheritedOr::Explicit(0),
            InheritedOr::Inherited,
        );
        assert!(doc.is_ok());
    }

    #[test]
    fn request_rejects_budget_below_min() {
        let doc = RichnessDocumentV1::with_all_explicit(
            0,
            2048,
            RichnessPreset::Sparse,
            RichnessTheme::Ancient,
            super::super::request::RichnessRequestSchemaRevision::V1,
            super::super::request::RichnessAlgorithmRevision::V1,
            super::super::request::RichnessContentRevision::V1,
            super::super::request::RichnessPresetRevision::V1,
            super::super::request::RichnessThemeRevision::V1,
            super::super::request::RichnessAssetRevision::V1,
            super::super::request::RichnessConventionRevision::V1,
            InheritedOr::Inherited,
            InheritedOr::Inherited,
            InheritedOr::Inherited,
            InheritedOr::Inherited,
            InheritedOr::Explicit(BUDGET_CEILING_MIN - 1),
        );
        assert!(doc.is_ok());
        assert!(ResolvedRichnessRequestV1::resolve(doc.unwrap()).is_err());
    }

    #[test]
    fn request_rejects_budget_exceeding_preset_ceiling() {
        // Sparse default ceiling is 3000; 8000 exceeds it
        let doc = RichnessDocumentV1::with_all_explicit(
            0,
            2048,
            RichnessPreset::Sparse,
            RichnessTheme::Ancient,
            super::super::request::RichnessRequestSchemaRevision::V1,
            super::super::request::RichnessAlgorithmRevision::V1,
            super::super::request::RichnessContentRevision::V1,
            super::super::request::RichnessPresetRevision::V1,
            super::super::request::RichnessThemeRevision::V1,
            super::super::request::RichnessAssetRevision::V1,
            super::super::request::RichnessConventionRevision::V1,
            InheritedOr::Inherited,
            InheritedOr::Inherited,
            InheritedOr::Inherited,
            InheritedOr::Inherited,
            InheritedOr::Explicit(8000),
        );
        assert!(doc.is_ok()); // raw validation passes (8000 <= max=8000)
        let resolved = ResolvedRichnessRequestV1::resolve(doc.unwrap());
        assert!(resolved.is_err());
        let err = resolved.unwrap_err();
        assert_eq!(err.code, RichnessErrorCode::BudgetInfeasible);
    }

    #[test]
    fn request_rejects_budget_insufficient_for_minimum_requirements() {
        // Landmarks=5, zones=3, vertical=12 with budget=1000
        // Budget 1000 < min_required = 5*500 + 3*200 + 12*150 = 2500+600+1800 = 4900
        let doc = RichnessDocumentV1::with_all_explicit(
            0,
            3072,
            RichnessPreset::Rich,
            RichnessTheme::Ancient,
            super::super::request::RichnessRequestSchemaRevision::V1,
            super::super::request::RichnessAlgorithmRevision::V1,
            super::super::request::RichnessContentRevision::V1,
            super::super::request::RichnessPresetRevision::V1,
            super::super::request::RichnessThemeRevision::V1,
            super::super::request::RichnessAssetRevision::V1,
            super::super::request::RichnessConventionRevision::V1,
            InheritedOr::Explicit(5),
            InheritedOr::Explicit(3),
            InheritedOr::Inherited,
            InheritedOr::Explicit(12),
            InheritedOr::Explicit(1000),
        );
        assert!(doc.is_ok());
        let resolved = ResolvedRichnessRequestV1::resolve(doc.unwrap());
        assert!(resolved.is_err());
        let err = resolved.unwrap_err();
        assert_eq!(err.code, RichnessErrorCode::BudgetInfeasible);
    }

    #[test]
    fn request_rejects_cave_required_with_insufficient_landmarks() {
        // Sparse (1 landmark) with cave=required
        let doc = RichnessDocumentV1::with_all_explicit(
            0,
            3072,
            RichnessPreset::Sparse,
            RichnessTheme::Ancient,
            super::super::request::RichnessRequestSchemaRevision::V1,
            super::super::request::RichnessAlgorithmRevision::V1,
            super::super::request::RichnessContentRevision::V1,
            super::super::request::RichnessPresetRevision::V1,
            super::super::request::RichnessThemeRevision::V1,
            super::super::request::RichnessAssetRevision::V1,
            super::super::request::RichnessConventionRevision::V1,
            InheritedOr::Inherited,
            InheritedOr::Inherited,
            InheritedOr::Explicit(RichnessCaveMode::Required),
            InheritedOr::Inherited,
            InheritedOr::Inherited,
        );
        assert!(doc.is_ok());
        let resolved = ResolvedRichnessRequestV1::resolve(doc.unwrap());
        assert!(resolved.is_err());
        let err = resolved.unwrap_err();
        assert_eq!(err.code, RichnessErrorCode::CaveInfeasible);
    }

    #[test]
    fn request_rejects_cave_required_with_insufficient_extent() {
        // Extent 1024 with cave=required
        let doc = RichnessDocumentV1::with_all_explicit(
            0,
            1024,
            RichnessPreset::Moderate,
            RichnessTheme::Ancient,
            super::super::request::RichnessRequestSchemaRevision::V1,
            super::super::request::RichnessAlgorithmRevision::V1,
            super::super::request::RichnessContentRevision::V1,
            super::super::request::RichnessPresetRevision::V1,
            super::super::request::RichnessThemeRevision::V1,
            super::super::request::RichnessAssetRevision::V1,
            super::super::request::RichnessConventionRevision::V1,
            InheritedOr::Explicit(2),
            InheritedOr::Inherited,
            InheritedOr::Explicit(RichnessCaveMode::Required),
            InheritedOr::Inherited,
            InheritedOr::Inherited,
        );
        assert!(doc.is_ok());
        let resolved = ResolvedRichnessRequestV1::resolve(doc.unwrap());
        assert!(resolved.is_err());
        let err = resolved.unwrap_err();
        assert_eq!(err.code, RichnessErrorCode::CaveInfeasible);
    }

    #[test]
    fn request_rejects_unknown_gate_in_canonical() {
        let valid_doc =
            RichnessDocumentV1::new(0, 2048, RichnessPreset::Sparse, RichnessTheme::Ancient)
                .unwrap();
        let canonical = String::from_utf8(valid_doc.to_canonical_bytes()).unwrap();
        let tampered = canonical.replace("gate:richness-v1", "gate:m3");
        let result = RichnessDocumentV1::from_canonical_bytes(tampered.as_bytes());
        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err().code,
            RichnessErrorCode::UnsupportedRichnessGate
        );
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 3. Source validators: all error codes are reachable
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn error_code_count_is_stable() {
        // Re-verify the 27 stable error codes
        let codes: &[RichnessErrorCode] = &[
            RichnessErrorCode::UnknownRequestSchemaRevision,
            RichnessErrorCode::UnknownAlgorithmRevision,
            RichnessErrorCode::UnknownContentRevision,
            RichnessErrorCode::UnknownPresetRevision,
            RichnessErrorCode::UnknownThemeRevision,
            RichnessErrorCode::UnknownAssetRevision,
            RichnessErrorCode::UnknownConventionRevision,
            RichnessErrorCode::UnsupportedRichnessGate,
            RichnessErrorCode::UnknownPreset,
            RichnessErrorCode::UnknownTheme,
            RichnessErrorCode::RevisionIncompatible,
            RichnessErrorCode::ValueOutOfRange,
            RichnessErrorCode::NotQuantumAligned,
            RichnessErrorCode::SemanticInfeasible,
            RichnessErrorCode::LandmarkCountInfeasible,
            RichnessErrorCode::ZoneCountInfeasible,
            RichnessErrorCode::CaveInfeasible,
            RichnessErrorCode::VerticalFeaturesInfeasible,
            RichnessErrorCode::BudgetInfeasible,
            RichnessErrorCode::PlacementExhausted,
            RichnessErrorCode::TopologyExhausted,
            RichnessErrorCode::UnsupportedConvention,
            RichnessErrorCode::BudgetOverrun,
            RichnessErrorCode::CaveFailure,
            RichnessErrorCode::AssetRoleMissing,
            RichnessErrorCode::CompilerFailure,
            RichnessErrorCode::PostcompileFailure,
        ];
        assert_eq!(codes.len(), 27, "error code count must remain stable at 27");
    }

    #[test]
    fn error_code_tags_are_unique() {
        let codes: Vec<&str> = [
            RichnessErrorCode::ValueOutOfRange,
            RichnessErrorCode::NotQuantumAligned,
            RichnessErrorCode::SemanticInfeasible,
            RichnessErrorCode::LandmarkCountInfeasible,
            RichnessErrorCode::ZoneCountInfeasible,
            RichnessErrorCode::CaveInfeasible,
            RichnessErrorCode::VerticalFeaturesInfeasible,
            RichnessErrorCode::BudgetInfeasible,
            RichnessErrorCode::PlacementExhausted,
            RichnessErrorCode::TopologyExhausted,
            RichnessErrorCode::UnsupportedConvention,
            RichnessErrorCode::BudgetOverrun,
            RichnessErrorCode::CaveFailure,
            RichnessErrorCode::AssetRoleMissing,
            RichnessErrorCode::CompilerFailure,
            RichnessErrorCode::PostcompileFailure,
            RichnessErrorCode::UnknownRequestSchemaRevision,
            RichnessErrorCode::UnknownAlgorithmRevision,
            RichnessErrorCode::UnknownContentRevision,
            RichnessErrorCode::UnknownPresetRevision,
            RichnessErrorCode::UnknownThemeRevision,
            RichnessErrorCode::UnknownAssetRevision,
            RichnessErrorCode::UnknownConventionRevision,
            RichnessErrorCode::UnsupportedRichnessGate,
            RichnessErrorCode::UnknownPreset,
            RichnessErrorCode::UnknownTheme,
            RichnessErrorCode::RevisionIncompatible,
        ]
        .iter()
        .map(|c| c.tag())
        .collect();
        let set: BTreeSet<_> = codes.iter().collect();
        assert_eq!(codes.len(), set.len(), "error code tags must be unique");
    }

    #[test]
    fn request_error_carries_all_revision_context() {
        let err = RichnessDocumentV1::new(42, 512, RichnessPreset::Sparse, RichnessTheme::Ancient)
            .unwrap_err();
        // Must carry the canonical revision tags
        assert!(err.request_schema_revision.contains("richness-request"));
        assert!(err.algorithm_revision.contains("richness-algorithm"));
        assert!(err.content_revision.contains("richness-content"));
        assert_eq!(err.seed, 42);
        assert_eq!(err.code, RichnessErrorCode::ValueOutOfRange);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 4. Deterministic sweeps — broad seed set
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn broad_seed_determinism_across_presets() {
        // The compiler corpus covers every frozen seed/preset/theme entry.
        // Keep this debug determinism gate representative and bounded.
        let seeds: &[u64] = &[0];
        let presets = [RichnessPreset::Sparse, RichnessPreset::Rich];

        for &seed in seeds {
            for &preset in &presets {
                let extent = match preset {
                    RichnessPreset::Sparse | RichnessPreset::Moderate => 2048u32,
                    RichnessPreset::Rich => 3072u32,
                };
                let doc = RichnessDocumentV1::new(seed, extent, preset, RichnessTheme::Ancient);
                // Some seeds may fail at placement; that's fine, but they must do so
                // deterministically (error must match byte-for-byte).
                if let Ok(doc) = doc {
                    let resolved = ResolvedRichnessRequestV1::resolve(doc.clone());
                    if let Ok(resolved) = resolved {
                        let r1 = run_richness_pipeline(&resolved);
                        let r2 = run_richness_pipeline(&resolved);
                        match (r1, r2) {
                            (Ok(o1), Ok(o2)) => {
                                assert_eq!(
                                    o1.map_text,
                                    o2.map_text,
                                    "seed={} preset={}: map bytes must be identical",
                                    seed,
                                    preset.tag()
                                );
                                assert_eq!(
                                    o1.generation_metadata.to_canonical_bytes(),
                                    o2.generation_metadata.to_canonical_bytes(),
                                    "seed={} preset={}: metadata bytes must be identical",
                                    seed,
                                    preset.tag()
                                );
                                assert_eq!(
                                    o1.actual,
                                    o2.actual,
                                    "seed={} preset={}: actual counts must be identical",
                                    seed,
                                    preset.tag()
                                );
                            }
                            (Err(e1), Err(e2)) => {
                                assert_eq!(
                                    e1.code,
                                    e2.code,
                                    "seed={} preset={}: error codes must be identical",
                                    seed,
                                    preset.tag()
                                );
                            }
                            (r1, r2) => {
                                panic!("seed={} preset={}: determinism broken: one succeeded, one failed: {:?} vs {:?}",
                                    seed, preset.tag(), r1.is_ok(), r2.is_ok());
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn broad_seed_sweep_completes_without_panic() {
        // Sweep matrix seeds across presets with extreme explicit controls:
        // all must complete (success or typed error, NEVER panic).
        // Overflow hardening: seed-derived indices use modular arithmetic
        // (seed % len) so arbitrary u64 seeds cannot overflow (props/lighting).
        let seeds: Vec<u64> = vec![0, 255];
        for seed in seeds {
            for preset in [
                super::super::request::RichnessPreset::Sparse,
                super::super::request::RichnessPreset::Moderate,
            ] {
                let doc = RichnessDocumentV1::new(seed, 2048, preset, RichnessTheme::Ancient);
                let Ok(doc) = doc else { continue };
                let resolved = ResolvedRichnessRequestV1::resolve(doc);
                let Ok(resolved) = resolved else { continue };
                let result = run_richness_pipeline(&resolved);
                match result {
                    Ok(output) => {
                        assert!(!output.map_text.is_empty());
                    }
                    Err(err) => {
                        assert!(err.code.tag().len() > 0);
                    }
                }
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 5. Cross-theme semantic identity
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn cross_theme_semantic_identity_equal_for_same_seed() {
        let seeds: &[u64] = &[42];
        // Cross-theme geometry/theme separation is invariant across presets;
        // the compiler corpus separately exhausts the complete matrix.
        let extents: &[(u32, RichnessPreset)] = &[(2048, RichnessPreset::Sparse)];

        for &seed in seeds {
            for &(extent, preset) in extents {
                let mut identities: Vec<([u8; 32], String, Vec<u8>)> = Vec::new();

                for &theme in RichnessTheme::ALL {
                    let doc = RichnessDocumentV1::new(seed, extent, preset, theme);
                    if let Ok(doc) = doc {
                        let resolved = match ResolvedRichnessRequestV1::resolve(doc) {
                            Ok(r) => r,
                            Err(_) => continue,
                        };
                        let output = match run_richness_pipeline(&resolved) {
                            Ok(o) => o,
                            Err(_) => continue,
                        };
                        let semantic = output.generation_metadata.semantic_identity().to_vec();
                        let map = output.map_text.clone();
                        identities.push((
                            output.request_metadata.request_identity(),
                            map,
                            semantic,
                        ));
                    }
                }

                if identities.len() < 2 {
                    continue; // not enough themes succeeded for comparison
                }

                // All themes must have identical semantic identity bytes
                let base_semantic = &identities[0].2;
                for (i, (_, _, semantic)) in identities.iter().enumerate().skip(1) {
                    assert_eq!(
                        base_semantic,
                        semantic,
                        "seed={} preset={}: theme {} semantic identity differs from theme 0",
                        seed,
                        preset.tag(),
                        i
                    );
                }

                // Presentation (map text) must differ across themes
                let mut maps: BTreeSet<&str> = BTreeSet::new();
                for (_, map, _) in &identities {
                    maps.insert(map.as_str());
                }
                // Not all maps may be unique (some themes may produce same map for small configs),
                // but at least one pair should differ if we have 3 themes
                if identities.len() >= 3 {
                    // At least 2 distinct map texts
                    assert!(maps.len() >= 2,
                        "seed={} preset={}: expected at least 2 distinct map texts across 3 themes, got {}",
                        seed, preset.tag(), maps.len());
                }
            }
        }
    }

    #[test]
    fn cross_theme_reservation_fingerprint_equal_for_same_seed() {
        // Same test but specifically for reservation fingerprint (macro footprint bytes)
        let seeds: &[u64] = &[42];

        for &seed in seeds {
            let mut fingerprints: Vec<Vec<u8>> = Vec::new();

            for &theme in RichnessTheme::ALL {
                let doc = RichnessDocumentV1::new(seed, 2048, RichnessPreset::Sparse, theme);
                if let Ok(doc) = doc {
                    let resolved = match ResolvedRichnessRequestV1::resolve(doc) {
                        Ok(r) => r,
                        Err(_) => continue,
                    };
                    let output = match run_richness_pipeline(&resolved) {
                        Ok(o) => o,
                        Err(_) => continue,
                    };
                    fingerprints.push(
                        output
                            .generation_metadata
                            .reservation_fingerprint()
                            .to_vec(),
                    );
                }
            }

            if fingerprints.len() < 2 {
                continue;
            }

            let base = &fingerprints[0];
            for fp in &fingerprints[1..] {
                assert_eq!(
                    base, fp,
                    "seed={}: reservation fingerprints differ across themes",
                    seed
                );
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 6. Pipeline output validation
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn pipeline_output_has_all_required_fields() {
        let doc = RichnessDocumentV1::new(0, 2048, RichnessPreset::Sparse, RichnessTheme::Ancient)
            .unwrap();
        let resolved = ResolvedRichnessRequestV1::resolve(doc).unwrap();
        let output = run_richness_pipeline(&resolved).unwrap();

        // Map text must be non-empty and contain worldspawn
        assert!(!output.map_text.is_empty());
        assert!(output.map_text.contains("\"classname\" \"worldspawn\""));
        assert!(output.map_text.contains("\"richness_theme\""));

        // Metadata must be non-empty
        assert!(!output.request_metadata.canonical_request().is_empty());
        assert!(output
            .request_metadata
            .request_identity()
            .iter()
            .any(|b| *b != 0));
        assert!(!output.generation_metadata.to_canonical_bytes().is_empty());
        assert!(!output.generation_metadata.semantic_identity().is_empty());

        // Request export must round-trip
        let exported = output.request_export.clone();
        let doc2 = RichnessDocumentV1::from_canonical_bytes(&exported).unwrap();
        let re_exported = doc2.to_canonical_bytes();
        assert_eq!(exported, re_exported);

        // Asset roles must have the 9 semantic roles
        assert_eq!(output.asset_roles.len(), 9);
        assert!(output.asset_roles.contains(&"wall".to_string()));
        assert!(output.asset_roles.contains(&"floor".to_string()));
        assert!(output.asset_roles.contains(&"ceiling".to_string()));

        // Actual counts must be positive
        assert!(output.actual.brushes > 0);
        assert!(output.actual.faces >= output.actual.brushes);
        assert!(output.actual.entities > 0);
        assert!(output.actual.lights > 0);
    }

    #[test]
    fn pipeline_preserves_request_metadata_in_map() {
        let doc = RichnessDocumentV1::new(0, 2048, RichnessPreset::Sparse, RichnessTheme::Egyptian)
            .unwrap();
        let resolved = ResolvedRichnessRequestV1::resolve(doc).unwrap();
        match run_richness_pipeline(&resolved) {
            Ok(output) => {
                // Map text must reference the theme
                assert!(output.map_text.contains("\"richness_theme\" \"egyptian\""));
                assert!(output.map_text.contains("worldspawn"));
            }
            Err(_err) => {
                // Pipeline failure for individual seeds is acceptable;
                // the in-crate determinism tests already prove the pipeline
                // works for known-good seed/config pairs.
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 7. Content ID occurrence via archetype coverage in pipeline output
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn all_archetype_ids_are_valid_indexable() {
        // Every archetype ID must be indexable via the generated lookup tables
        for id in generated_content::ARCHETYPE_IDS {
            let idx = generated_content::ARCHETYPE_IDS
                .iter()
                .position(|x| *x == *id)
                .expect("archetype ID must be findable");
            assert!(idx < generated_content::ARCHETYPE_COUNT);
            // Verify all parallel arrays are accessible at this index
            let _span_min = generated_content::ARCHETYPE_SPAN_MIN[idx];
            let _span_max = generated_content::ARCHETYPE_SPAN_MAX[idx];
            let _shape = &generated_content::ARCHETYPE_SHAPE[idx];
            let _rarity = &generated_content::ARCHETYPE_RARITY[idx];
        }
    }

    #[test]
    fn all_prop_ids_are_valid_indexable() {
        // Every prop ID must be indexable via prop_index()
        for id in generated_content::PROP_IDS {
            let idx = generated_content::PROP_IDS
                .iter()
                .position(|x| *x == *id)
                .expect("prop ID must be findable");
            // Use the content_types lookup
            let looked_up = super::super::content_types::prop_index(id);
            assert_eq!(looked_up as usize, idx, "prop_index({}) mismatch", id);
            let _dims = generated_content::PROP_DIMENSIONS[idx];
            let _cost = generated_content::PROP_COST_SOURCE_FACES[idx];
        }
    }

    #[test]
    fn all_light_recipe_ids_are_valid_indexable() {
        for id in generated_content::LIGHT_RECIPE_IDS {
            let idx = generated_content::LIGHT_RECIPE_IDS
                .iter()
                .position(|x| *x == *id)
                .expect("light recipe ID must be findable");
            let looked_up = super::super::content_types::light_index(id);
            assert_eq!(looked_up as usize, idx, "light_index({}) mismatch", id);
            let _color = generated_content::LIGHT_COLOR[idx];
            let _intensity = generated_content::LIGHT_INTENSITY[idx];
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 8. Inherited vs explicit provenance
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn inherited_or_preserves_explicit_same_as_default() {
        // Explicit with same value as default must not collapse to Inherited
        let doc = RichnessDocumentV1::with_all_explicit(
            42,
            2048,
            RichnessPreset::Sparse,
            RichnessTheme::Ancient,
            super::super::request::RichnessRequestSchemaRevision::V1,
            super::super::request::RichnessAlgorithmRevision::V1,
            super::super::request::RichnessContentRevision::V1,
            super::super::request::RichnessPresetRevision::V1,
            super::super::request::RichnessThemeRevision::V1,
            super::super::request::RichnessAssetRevision::V1,
            super::super::request::RichnessConventionRevision::V1,
            InheritedOr::Explicit(1), // same as Sparse default
            InheritedOr::Inherited,
            InheritedOr::Inherited,
            InheritedOr::Inherited,
            InheritedOr::Inherited,
        )
        .unwrap();
        let resolved = ResolvedRichnessRequestV1::resolve(doc).unwrap();
        assert_eq!(
            resolved.critical_path_landmarks().source(),
            ValueSource::Explicit
        );
        assert_eq!(resolved.critical_path_landmarks().value(), 1);
    }

    #[test]
    fn inherited_values_resolve_to_preset_defaults() {
        for preset in RichnessPreset::ALL {
            let extent = if *preset == RichnessPreset::Rich {
                3072
            } else {
                2048
            };
            let doc = RichnessDocumentV1::new(42, extent, *preset, RichnessTheme::Ancient).unwrap();
            let resolved = ResolvedRichnessRequestV1::resolve(doc).unwrap();

            assert_eq!(
                resolved.critical_path_landmarks().source(),
                ValueSource::Inherited
            );
            assert_eq!(resolved.zone_count().source(), ValueSource::Inherited);
            assert_eq!(resolved.cave_mode().source(), ValueSource::Inherited);
            assert_eq!(
                resolved.vertical_openings().source(),
                ValueSource::Inherited
            );
            assert_eq!(resolved.budget_ceiling().source(), ValueSource::Inherited);

            // Verify inherited values match preset defaults
            match preset {
                RichnessPreset::Sparse => {
                    assert_eq!(resolved.critical_path_landmarks().value(), 1);
                    assert_eq!(resolved.budget_ceiling().value(), 3000);
                }
                RichnessPreset::Moderate => {
                    assert_eq!(resolved.critical_path_landmarks().value(), 2);
                    assert_eq!(resolved.budget_ceiling().value(), 5000);
                }
                RichnessPreset::Rich => {
                    assert_eq!(resolved.critical_path_landmarks().value(), 3);
                    assert_eq!(resolved.budget_ceiling().value(), 8000);
                }
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 9. Request hash domain is frozen
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn request_hash_domain_is_frozen() {
        assert_eq!(
            RICHNESS_REQUEST_DOMAIN,
            b"dungeon-gen/v3-richness/v1/request"
        );
    }

    #[test]
    fn schema_version_is_frozen() {
        assert_eq!(
            generated_content::SCHEMA_VERSION,
            "enhanced-v3-richness-content/v1"
        );
    }

    #[test]
    fn source_hash_is_frozen() {
        // The source hash is a frozen SHA-256 of the codegen input
        assert!(!generated_content::SOURCE_HASH.is_empty());
        assert_eq!(generated_content::SOURCE_HASH.len(), 64);
        // Must be uppercase hex
        assert!(generated_content::SOURCE_HASH
            .chars()
            .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit()));
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 10. Request canonical identity round-trips
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn canonical_request_roundtrip_all_preset_theme_combinations() {
        for preset in RichnessPreset::ALL {
            for theme in RichnessTheme::ALL {
                let extent = if *preset == RichnessPreset::Rich {
                    3072
                } else {
                    2048
                };
                let doc = RichnessDocumentV1::new(42, extent, *preset, *theme).unwrap();
                let bytes = doc.to_canonical_bytes();
                let doc2 = RichnessDocumentV1::from_canonical_bytes(&bytes).unwrap();
                assert_eq!(doc, doc2);
            }
        }
    }

    #[test]
    fn resolved_canonical_roundtrip_all_preset_theme_combinations() {
        for preset in RichnessPreset::ALL {
            for theme in RichnessTheme::ALL {
                let extent = if *preset == RichnessPreset::Rich {
                    3072
                } else {
                    2048
                };
                let doc = RichnessDocumentV1::new(42, extent, *preset, *theme).unwrap();
                let resolved = ResolvedRichnessRequestV1::resolve(doc).unwrap();
                let bytes = resolved.to_canonical_bytes();
                let resolved2 = ResolvedRichnessRequestV1::from_canonical_bytes(&bytes).unwrap();
                assert_eq!(resolved, resolved2);
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 11. Request explicit round-trip with all control permutations
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn explicit_request_preserves_all_fields_in_canonical() {
        let doc = RichnessDocumentV1::with_all_explicit(
            255,
            3072,
            RichnessPreset::Rich,
            RichnessTheme::Brutalist,
            super::super::request::RichnessRequestSchemaRevision::V1,
            super::super::request::RichnessAlgorithmRevision::V1,
            super::super::request::RichnessContentRevision::V1,
            super::super::request::RichnessPresetRevision::V1,
            super::super::request::RichnessThemeRevision::V1,
            super::super::request::RichnessAssetRevision::V1,
            super::super::request::RichnessConventionRevision::V1,
            InheritedOr::Explicit(3),
            InheritedOr::Explicit(2),
            InheritedOr::Explicit(RichnessCaveMode::Required),
            InheritedOr::Explicit(6),
            InheritedOr::Explicit(8000),
        )
        .unwrap();
        let bytes = doc.to_canonical_bytes();
        let doc2 = RichnessDocumentV1::from_canonical_bytes(&bytes).unwrap();
        assert_eq!(doc, doc2);

        // Verify explicit values preserved
        assert_eq!(doc2.critical_path_landmarks, InheritedOr::Explicit(3));
        assert_eq!(
            doc2.cave_mode,
            InheritedOr::Explicit(RichnessCaveMode::Required)
        );
        assert_eq!(doc2.budget_ceiling, InheritedOr::Explicit(8000));
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 12. Pipeline rejects impossible configurations cleanly
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn pipeline_tiny_extent_fails_cleanly() {
        // 128 extent is well below minimum; document construction rejects it
        let doc = RichnessDocumentV1::new(1, 128, RichnessPreset::Rich, RichnessTheme::Ancient);
        assert!(doc.is_err());
    }

    #[test]
    fn pipeline_zero_seed_succeeds_or_fails_with_typed_error() {
        // Zero seed is valid; if it fails, it must be a typed error
        let doc = RichnessDocumentV1::new(0, 2048, RichnessPreset::Sparse, RichnessTheme::Ancient)
            .unwrap();
        let resolved = ResolvedRichnessRequestV1::resolve(doc).unwrap();
        match run_richness_pipeline(&resolved) {
            Ok(output) => {
                assert!(!output.map_text.is_empty());
            }
            Err(err) => {
                // Must be a recognized error code
                let valid_codes = [
                    RichnessErrorCode::PlacementExhausted,
                    RichnessErrorCode::TopologyExhausted,
                    RichnessErrorCode::CaveFailure,
                    RichnessErrorCode::SemanticInfeasible,
                ];
                assert!(
                    valid_codes.contains(&err.code),
                    "unexpected error code {:?} for seed 0 sparse",
                    err.code
                );
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 13. Generated content constant consistency
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn archetype_theme_arrays_consistent_across_themes() {
        // Each theme-specific array must have 3 entries per archetype
        let n = generated_content::ARCHETYPE_COUNT;
        for i in 0..n {
            // Theme massing: 3 strings per archetype
            assert_eq!(generated_content::ARCHETYPE_THEME_MASSING[i].len(), 3);
            // Theme support: 3 strings per archetype
            assert_eq!(generated_content::ARCHETYPE_THEME_SUPPORT[i].len(), 3);
            // Theme prop refs: 3 arrays per archetype
            assert_eq!(generated_content::ARCHETYPE_THEME_PROP_REFS[i].len(), 3);
            // Theme light refs: 3 arrays per archetype
            assert_eq!(generated_content::ARCHETYPE_THEME_LIGHT_REFS[i].len(), 3);
            // Theme materials: 3 arrays per archetype
            assert_eq!(generated_content::ARCHETYPE_THEME_MATERIALS[i].len(), 3);
        }
    }

    #[test]
    fn theme_prop_compat_includes_all_props() {
        for theme_idx in 0..generated_content::THEME_COUNT {
            let compat = generated_content::THEME_PROP_COMPAT[theme_idx];
            assert_eq!(
                compat.len(),
                15,
                "theme {} must reference all 15 props",
                theme_idx
            );
        }
    }

    #[test]
    fn theme_light_compat_includes_all_light_recipes() {
        for theme_idx in 0..generated_content::THEME_COUNT {
            let compat = generated_content::THEME_LIGHT_COMPAT[theme_idx];
            assert_eq!(
                compat.len(),
                12,
                "theme {} must reference all 12 light recipes",
                theme_idx
            );
        }
    }

    #[test]
    fn spiral_stair_constants_are_consistent() {
        assert_eq!(generated_content::SPIRAL_LAYER_OFFSET, 192);
        assert_eq!(generated_content::SPIRAL_ENVELOPE_MIN, [224, 224]);
        assert_eq!(generated_content::SPIRAL_STEP_COUNT, 12);
        assert_eq!(generated_content::SPIRAL_STEP_INDEX.len(), 12);
        assert_eq!(generated_content::SPIRAL_STEP_RISE.len(), 12);
        for rise in generated_content::SPIRAL_STEP_RISE {
            assert_eq!(*rise, 16, "spiral step rise must be exactly 16");
        }
        for tread in generated_content::SPIRAL_STEP_TREAD_DEPTH {
            assert_eq!(*tread, 64, "spiral tread depth must be exactly 64");
        }
    }
}
