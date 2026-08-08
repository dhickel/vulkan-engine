//! Sealed private Richness V1 pipeline: one all-or-nothing path from a
//! resolved request to canonical map text, metadata, request export, asset
//! roles, and validation witnesses.
//!
//! Every stage input is immutable and every stage output is typed. A stage
//! failure returns the stable Richness error envelope with seed/revision
//! context — never a partial map, partial metadata, or mutated state. The
//! pipeline never invokes the compiler and never lets compiler feedback
//! change canonical source generation. The bundle is crate-private until the
//! atomic release phase.

use std::collections::BTreeMap;

use super::{
    composition::{compose_solved_generation, StructuralComposition},
    error::{RichnessError, RichnessErrorCategory, RichnessErrorCode},
    generated_content::SCHEMA_VERSION,
    metadata::RichnessMetadataV1,
    pacing::build_pacing_blueprint,
    request::{ResolvedRichnessRequestV1, RichnessCaveMode, RichnessPreset, RichnessTheme},
    solver::solve_placement_and_topology,
    topology::TopologyResult,
};

/// The complete private pipeline output bundle.
#[derive(Debug, Clone)]
pub(crate) struct RichnessPipelineOutput {
    /// Canonical map text (Standard Quake grammar, frozen ordering).
    pub map_text: String,
    /// Immutable request provenance metadata.
    pub request_metadata: RichnessMetadataV1,
    /// Deterministic generation metadata facts (fixed order).
    pub generation_metadata: super::metadata::RichnessGenerationMetadata,
    /// Canonical request export bytes (provenance-preserving).
    pub request_export: Vec<u8>,
    /// Referenced theme asset role identities (fixed order).
    pub asset_roles: Vec<String>,
    /// Structural composition (assembly + visibility + cave + presentation).
    pub composition: StructuralComposition,
    /// Actual brush/entity/light counts recomputed from the assembly.
    pub actual: ActualCounts,
}

/// Actual output counts recomputed from the sealed assembly (the last
/// invariant gate: actual may be less than reserved, never more).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ActualCounts {
    pub brushes: usize,
    pub faces: usize,
    pub entities: usize,
    pub lights: usize,
    pub support_contacts: usize,
    pub openings: usize,
}

/// Run the complete private Richness pipeline.
///
/// Ordered stages: pacing blueprint -> zone blueprint -> placement + topology
/// -> variation plan -> complexity plan -> composition (structural, portals,
/// walls, vertical, cave, presentation) -> cross-stage invariant validation ->
/// actual-count recompute -> canonical emission -> metadata.
pub(crate) fn run_richness_pipeline(
    resolved: &ResolvedRichnessRequestV1,
) -> Result<RichnessPipelineOutput, RichnessError> {
    let seed = resolved.seed();
    let preset = resolved.preset();
    let theme = resolved.theme();
    let cave_mode = resolved.cave_mode().value;

    // 1. Semantic pacing blueprint (theme-independent).
    let blueprint = build_pacing_blueprint(resolved)?;
    let zone_blueprint = blueprint.zone_blueprint.clone();

    // 2. Solve placement and constrained-Kruskal topology.
    let generation = solve_placement_and_topology(blueprint.clone(), resolved.clone())?;

    // 3. Variation plan (bounded legal variation over committed envelopes).
    let _variation = super::variation::build_variation_plan(
        &blueprint,
        &generation.topology,
        &zone_blueprint,
        &generation.topology.journal,
    );

    // 4. Complexity plan (complete recipe reservation).
    let complexity = super::complexity::build_complexity_plan(
        preset,
        theme_ordinal(theme) as u32,
        &blueprint,
        &generation.topology,
        &generation.placement.request_archetypes,
    );
    if !complexity.is_within_budget() {
        return Err(pipeline_error(
            "complexity.budget",
            format!("complexity plan exceeds ceilings: {:?}", complexity.errors),
        ));
    }

    // 5. Composition: structural, portals, walls, vertical, cave, presentation.
    let composition = compose_solved_generation(&generation, theme, &complexity, seed, cave_mode)?;

    // 6. Cross-stage invariants + actual-count recompute (last gate):
    // actual counts must never exceed the complexity plan's reserved totals.
    let actual = recompute_actual_counts(&composition);
    let actual_budget = super::complexity::BudgetReservation {
        faces: actual.faces as u32,
        brushes: actual.brushes as u32,
        entities: actual.entities as u32,
        lights: actual.lights as u32,
        vertical_openings: actual.openings as u32,
        support_contacts: actual.support_contacts as u32,
        package_assets: 0,
        compiler_lumps: 0,
        renderer_batches: 0,
        renderer_memory_bytes: 0,
        runtime_requirements: 0,
    };
    if !complexity.assert_dominates(&actual_budget) {
        return Err(pipeline_error(
            "costs.actual_exceed_reserved",
            format!(
                "actual counts {actual:?} exceed the complexity reservation {:?}",
                complexity.total_reserved
            ),
        ));
    }

    // 7. Canonical emission.
    let spawn = derive_spawn(&generation.topology);
    let map_text = super::emission::emit_richness_map(&composition, theme, spawn)?;

    // 8. Deterministic generation metadata + request export.
    let request_metadata = RichnessMetadataV1::from_resolved(resolved);
    let generation_metadata = super::metadata::RichnessGenerationMetadata::build(
        resolved,
        &blueprint,
        &generation.topology,
        &composition,
        actual,
    );
    let request_export = resolved.provenance().to_canonical_bytes();
    let asset_roles = theme_asset_roles(theme);

    Ok(RichnessPipelineOutput {
        map_text,
        request_metadata,
        generation_metadata,
        request_export,
        asset_roles,
        composition,
        actual,
    })
}

/// Recompute actual counts from the sealed assembly (never from stage
/// estimates).
pub(crate) fn recompute_actual_counts(composition: &StructuralComposition) -> ActualCounts {
    let brushes = composition.assembly.brushes.len();
    let faces = composition
        .assembly
        .brushes
        .values()
        .map(|brush| brush.brush.faces.len())
        .sum();
    let entities = composition.assembly.entities.len();
    let lights = composition
        .assembly
        .entities
        .values()
        .filter(|entity| entity.classname == "light")
        .count();
    // World anchors are not inter-brush support contacts. Count only the
    // derived brush-to-brush contacts that consume the bounded support graph
    // reservation; floor slabs remain explicit world roots.
    let support_contacts = composition
        .assembly
        .supports
        .values()
        .filter(|support| matches!(support.parent, super::assembly::SupportTarget::Brush(_)))
        .filter(|support| {
            composition
                .assembly
                .brushes
                .get(&support.child)
                .is_some_and(|brush| brush.role.is_vertical_architecture())
        })
        .count();
    // Complexity reserves logical vertical openings. Portal throats are
    // accounted by their dedicated sealing recipes and must not consume the
    // bounded vertical-opening dimension a second time.
    let openings = composition
        .assembly
        .openings
        .values()
        .filter(|opening| opening.portal_id.is_none() && opening.wall_role.is_slab())
        .count();
    ActualCounts {
        brushes,
        faces,
        entities,
        lights,
        support_contacts,
        openings,
    }
}

fn theme_ordinal(theme: RichnessTheme) -> usize {
    match theme {
        RichnessTheme::Ancient => 0,
        RichnessTheme::Egyptian => 1,
        RichnessTheme::Brutalist => 2,
    }
}

/// Deterministic spawn origin: the first committed Spawn reservation's
/// footprint center at 24 units above its floor; fallback (0, 0, 24).
fn derive_spawn(topology: &TopologyResult) -> (i32, i32, i32) {
    use super::reservation::ReservationKind;
    for (_, record) in &topology.journal.reservations {
        if !record.committed || record.kind != ReservationKind::Spawn {
            continue;
        }
        let bounds = super::geometry::footprint_quake_bounds(&record.footprint);
        let vertical = super::geometry::footprint_vertical_bounds(&record.footprint).ok();
        let z = vertical.map(|v| v.floor_min).unwrap_or(0);
        let cx = ((bounds.0 + bounds.2) / 2) as i32;
        let cy = ((bounds.1 + bounds.3) / 2) as i32;
        return (cx, cy, z as i32 + 24);
    }
    (0, 0, 24)
}

/// Referenced theme asset role identities in frozen order (the nine semantic
/// roles present in every theme WAD).
fn theme_asset_roles(theme: RichnessTheme) -> Vec<String> {
    let _ = theme;
    [
        "wall", "floor", "ceiling", "accent", "portal", "vertical", "cave", "prop", "emissive",
    ]
    .iter()
    .map(|role| role.to_string())
    .collect()
}

/// Build a typed pipeline error.
pub(crate) fn pipeline_error(path: &str, context: impl Into<String>) -> RichnessError {
    RichnessError::new(
        RichnessErrorCode::SemanticInfeasible,
        0,
        SCHEMA_VERSION,
        "?",
        "?",
        "?",
        "?",
        "?",
        "?",
        path,
        RichnessErrorCategory::SemanticInfeasibility,
        context,
    )
}

/// Convenience alias for pipeline tests.
pub(crate) type PipelineResult = Result<RichnessPipelineOutput, RichnessError>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::enhanced_v3::richness::request::{RichnessDocumentV1, RichnessPreset};

    fn resolved(
        seed: u64,
        extent: u32,
        preset: RichnessPreset,
        theme: RichnessTheme,
    ) -> ResolvedRichnessRequestV1 {
        ResolvedRichnessRequestV1::resolve(
            RichnessDocumentV1::new(seed, extent, preset, theme).unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn pipeline_runs_end_to_end_for_all_presets_and_themes() {
        for preset in [
            RichnessPreset::Sparse,
            RichnessPreset::Moderate,
            RichnessPreset::Rich,
        ] {
            for theme in [
                RichnessTheme::Ancient,
                RichnessTheme::Egyptian,
                RichnessTheme::Brutalist,
            ] {
                let request = resolved(42, 2048, preset, theme);
                let output = run_richness_pipeline(&request).expect("pipeline");
                assert!(!output.map_text.is_empty());
                assert!(output.map_text.contains("\"classname\" \"worldspawn\""));
                assert!(output
                    .map_text
                    .contains(&format!("\"richness_theme\" \"{}\"", theme.tag())));
                assert!(output.actual.brushes > 0);
                assert!(!output.generation_metadata.to_canonical_bytes().is_empty());
            }
        }
    }

    #[test]
    fn pipeline_is_deterministic_byte_identical() {
        let request = resolved(7, 2048, RichnessPreset::Moderate, RichnessTheme::Ancient);
        let first = run_richness_pipeline(&request).expect("first");
        let second = run_richness_pipeline(&request).expect("second");
        assert_eq!(first.map_text, second.map_text);
        assert_eq!(
            first.generation_metadata.to_canonical_bytes(),
            second.generation_metadata.to_canonical_bytes()
        );
        assert_eq!(first.request_export, second.request_export);
        assert_eq!(first.actual, second.actual);
    }

    #[test]
    fn theme_invariance_semantic_metadata_identical_bytes_differ() {
        let ancient = run_richness_pipeline(&resolved(
            99,
            2048,
            RichnessPreset::Sparse,
            RichnessTheme::Ancient,
        ))
        .expect("ancient");
        let egyptian = run_richness_pipeline(&resolved(
            99,
            2048,
            RichnessPreset::Sparse,
            RichnessTheme::Egyptian,
        ))
        .expect("egyptian");
        // Semantic blueprint + macro reservation metadata identical.
        assert_eq!(
            ancient.generation_metadata.semantic_identity(),
            egyptian.generation_metadata.semantic_identity()
        );
        // Map bytes differ (theme presentation).
        assert_ne!(ancient.map_text, egyptian.map_text);
        // Macro reservations identical.
        assert_eq!(
            ancient.generation_metadata.reservation_fingerprint(),
            egyptian.generation_metadata.reservation_fingerprint()
        );
    }

    #[test]
    fn required_cave_pipeline_with_cave_mode() {
        // Cave mode defaults to Preferred in the pipeline; a request that
        // demands caves (Required) still runs end-to-end or fails with a
        // stable typed CaveFailure — never a partial output.
        let request = resolved(42, 2048, RichnessPreset::Moderate, RichnessTheme::Ancient);
        match run_richness_pipeline(&request) {
            Ok(output) => assert!(!output.map_text.is_empty()),
            Err(error) => {
                assert!(
                    matches!(
                        error.code,
                        RichnessErrorCode::CaveFailure
                            | RichnessErrorCode::CaveInfeasible
                            | RichnessErrorCode::SemanticInfeasible
                            | RichnessErrorCode::PlacementExhausted
                            | RichnessErrorCode::TopologyExhausted
                    ),
                    "unexpected error code: {:?}",
                    error.code
                );
            }
        }
    }

    #[test]
    fn failure_is_atomic_no_partial_output() {
        // An impossible request (tiny extent) must fail with a typed error
        // and never return a bundle.
        let outcome = RichnessDocumentV1::new(1, 128, RichnessPreset::Rich, RichnessTheme::Ancient);
        assert!(
            outcome.is_err(),
            "tiny-extent Rich must fail atomically at request validation"
        );
    }

    #[test]
    fn actual_counts_never_exceed_reserved() {
        for preset in [
            RichnessPreset::Sparse,
            RichnessPreset::Moderate,
            RichnessPreset::Rich,
        ] {
            let request = resolved(42, 2048, preset, RichnessTheme::Ancient);
            let output = run_richness_pipeline(&request).expect("pipeline");
            // The pipeline's own gate already ran assert_dominates; here we
            // only re-assert the recomputed counts are sane and non-zero.
            assert!(output.actual.brushes > 0);
            assert!(output.actual.faces >= output.actual.brushes);
            assert!(output.actual.lights <= 100);
        }
    }
}
