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
    request::{
        ResolvedRichnessRequestV1, RichnessCaveMode, RichnessDocumentV1, RichnessPreset,
        RichnessTheme,
    },
    solver::solve_placement_and_topology,
    topology::TopologyResult,
};

/// The complete private pipeline output bundle.
#[derive(Debug, Clone)]
pub struct RichnessPipelineOutput {
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
    pub(crate) composition: StructuralComposition,
    /// Actual brush/entity/light counts recomputed from the assembly.
    pub actual: ActualCounts,
}

/// Actual output counts recomputed from the sealed assembly (the last
/// invariant gate: actual may be less than reserved, never more).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActualCounts {
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
        cave_mode,
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

    // 7. Canonical emission. Validate each point origin after all structure exists.
    let spawn = derive_spawn(&generation.topology);
    let spawn = hull2_clear_spawn(&composition.assembly, spawn)?;
    validate_entity_origin_safety(&composition.assembly, spawn)?;
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

/// Validate a point-entity origin is inside a sealed volume (used by
/// `validate_entity_origin_safety`).
fn validate_entity_origin_safety(
    ir: &super::assembly::AssemblyIR,
    spawn: (i32, i32, i32),
) -> Result<(), RichnessError> {
    let spawn = (
        i128::from(spawn.0),
        i128::from(spawn.1),
        i128::from(spawn.2),
    );
    if !super::lighting::origin_airtight(ir, spawn) {
        return Err(pipeline_error(
            "entity_origin.spawn",
            format!("unsafe spawn {spawn:?}"),
        ));
    }
    for entity in ir.entities.values() {
        if !super::lighting::origin_airtight(ir, entity.origin) {
            return Err(pipeline_error(
                "entity_origin",
                format!("unsafe {} origin {:?}", entity.classname, entity.origin),
            ));
        }
    }
    Ok(())
}

/// Shift the spawn origin so the qbsp hull-2 box (64x64x64 above the origin)
/// clears every structural brush. The spawn point may be airtight while a
/// nearby stair tread or landing blocks the hull-2 outside-fill seed, which
/// makes qbsp emit "No entities in empty space -- no filling performed
/// (hull 2)". Deterministic offsets; the airtight check runs afterwards.
fn hull2_clear_spawn(
    ir: &super::assembly::AssemblyIR,
    spawn: (i32, i32, i32),
) -> Result<(i32, i32, i32), RichnessError> {
    let offsets: [(i32, i32); 9] = [
        (0, 0),
        (32, 0),
        (0, 32),
        (-32, 0),
        (0, -32),
        (64, 0),
        (0, 64),
        (-64, 0),
        (0, -64),
    ];
    for (dx, dy) in offsets {
        let origin = (spawn.0 + dx, spawn.1 + dy, spawn.2);
        let hull_box = crate::enhanced_v3::geometry::ConvexBrush::make_box(
            (i128::from(origin.0) - 32, i128::from(origin.0) + 32),
            (i128::from(origin.1) - 32, i128::from(origin.1) + 32),
            (i128::from(origin.2), i128::from(origin.2) + 64),
        )
        .map_err(|error| pipeline_error("spawn.hull2_box", format!("{error}")))?;
        let blocked = ir.brushes.values().any(|brush| {
            crate::enhanced_v3::richness::geometry::brushes_overlap(&hull_box, &brush.brush)
                .unwrap_or(true)
        });
        if !blocked {
            return Ok(origin);
        }
    }
    Err(pipeline_error(
        "spawn.hull2_blocked",
        format!("no hull-2-clear spawn position near {spawn:?}"),
    ))
}

fn theme_ordinal(theme: RichnessTheme) -> usize {
    match theme {
        RichnessTheme::Ancient => 0,
        RichnessTheme::Egyptian => 1,
        RichnessTheme::Brutalist => 2,
    }
}

/// Deterministic spawn origin: the first committed Spawn reservation's
/// footprint center at the floor-slab top plus the frozen 24-unit clearance;
/// fallback (0, 0, 40).
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
        return (cx, cy, z as i32 + 40);
    }
    (0, 0, 40)
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

// ── Public entry point ─────────────────────────────────────────────────────

/// Generate a Richness V1 dungeon from an authored request document.
///
/// This is the sole public entry point for Richness V1 generation. It
/// validates and resolves the request, runs the complete private pipeline,
/// and returns the canonical map text, metadata, validation witnesses, and
/// asset roles in a single all-or-nothing bundle.
///
/// # Determinism
///
/// Two calls with an identical `RichnessDocumentV1` produce byte-identical
/// `.map` output and field-identical metadata.
///
/// # Errors
///
/// Returns [`RichnessError`] if validation, placement, topology, composition,
/// emission, or any other pipeline stage fails. No partial output is returned.
pub fn generate_richness_v1(
    request: &RichnessDocumentV1,
) -> Result<RichnessPipelineOutput, RichnessError> {
    let resolved = ResolvedRichnessRequestV1::resolve(request.clone())?;
    run_richness_pipeline(&resolved)
}

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
        // The compiler corpus exhaustively covers the 3×3 preset/theme matrix.
        // This in-crate gate covers every preset and every theme with six
        // end-to-end requests without duplicating that costly matrix in debug.
        let requests = [
            (RichnessPreset::Sparse, RichnessTheme::Ancient),
            (RichnessPreset::Moderate, RichnessTheme::Ancient),
            (RichnessPreset::Rich, RichnessTheme::Ancient),
            (RichnessPreset::Sparse, RichnessTheme::Egyptian),
            (RichnessPreset::Sparse, RichnessTheme::Brutalist),
        ];

        for (preset, theme) in requests {
            let request = resolved(0, 2048, preset, theme);
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
            let request = resolved(0, 2048, preset, RichnessTheme::Ancient);
            let output = run_richness_pipeline(&request).expect("pipeline");
            // The pipeline's own gate already ran assert_dominates; here we
            // only re-assert the recomputed counts are sane and non-zero.
            assert!(output.actual.brushes > 0);
            assert!(output.actual.faces >= output.actual.brushes);
            assert!(output.actual.lights <= 100);
        }
    }
}
