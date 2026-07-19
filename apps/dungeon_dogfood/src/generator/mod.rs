pub(crate) mod ascii;
mod config;
pub mod determinism;
mod diagnostics;
mod error;
pub(crate) mod ir;
pub(crate) mod markers;
pub(crate) mod placement;
pub mod prefab;
pub(crate) mod ramps;
pub(crate) mod repair;
pub(crate) mod resources;
pub(crate) mod routing;
pub(crate) mod topology;
pub(crate) mod validation;
pub mod capture_views;

use std::path::Path;

use crate::layout::{ParsedLevel, TileCoord};

use self::capture_views::derive_capture_views;
use self::config::NormalizedGeneratorConfig;
use self::determinism::{AttemptIdentity, GeneratorIdentity, SemanticStage, SemanticStreamFactory};
use self::diagnostics::GeneratorDiagnostics;
use self::markers::place_all_markers;
use self::placement::place_regions;
use self::prefab::PrefabCatalog;
use self::repair::RepairEngine;
use self::resources::{count_resources, enforce_budgets};
use self::routing::materialize_topology;
use self::topology::{build_candidate_graph, select_topology};
use self::validation::reconstruct_movement_graph;

// ─── Public API ────────────────────────────────────────────────────────────

// Re-export public types.
pub use self::config::GeneratorConfig;
pub use self::config::QualifiedProfile;
pub use self::error::{ErrorStage, GeneratorError};
pub use self::resources::ResourceCounts;
pub use self::capture_views::{CaptureView, CaptureViewCategory};

/// Complete result of a successful generation attempt.
#[derive(Debug, Clone)]
pub struct GenerationResult {
    /// The materialized, validated level with placed markers.
    pub level: ParsedLevel,
    /// Canonical diagnostics for reproducibility and debugging.
    pub diagnostics: Vec<u8>,
    /// Deterministic capture-site camera views.
    pub capture_views: Vec<CaptureView>,
    /// Resource accounting.
    pub resource_counts: resources::ResourceCounts,
    /// The seed used for generation.
    pub seed: u64,
}

/// Run the complete generator pipeline and return a validated level.
///
/// This is the single public entrypoint for the app. All generation phases —
/// placement, topology, materialization, repair, markers, resources,
/// and capture views — are executed transactionally.
pub fn generate(
    config: GeneratorConfig,
    catalog: &PrefabCatalog,
    seed: u64,
) -> Result<GenerationResult, GeneratorError> {
    let normalized = config.normalize()?;
    let identity = GeneratorIdentity::new(&normalized, catalog.identity_bytes(), seed);
    let factory = SemanticStreamFactory::new(AttemptIdentity::new(identity, 0));
    let diagnostics = GeneratorDiagnostics::new(&normalized, catalog.identity_bytes(), seed);

    // Phase 02 — Placement.
    let mut roles_rng = factory.stream(SemanticStage::Roles, &[]);
    let (placed_topology, grid) = place_regions(&normalized, catalog, &mut roles_rng, factory)?;

    // Phase 03 — Topology.
    let candidate_graph = build_candidate_graph(&placed_topology, &grid)?;
    let mut topology_rng = factory.stream(SemanticStage::Topology, &[]);
    let selected_topology = select_topology(
        placed_topology,
        &normalized,
        &candidate_graph,
        &mut topology_rng,
    )?;

    // Phase 04 — Materialization.
    let tile_buffer = materialize_topology(&selected_topology, catalog, &normalized)?;

    // Build a bare ParsedLevel (no markers yet).
    let spawn_region = selected_topology
        .regions
        .iter()
        .find(|r| matches!(r.role, self::ir::RegionRole::Spawn))
        .ok_or(GeneratorError::IrInvariant {
            stage: self::error::ErrorStage::Ir,
            detail: "generate_no_spawn_region".into(),
        })?;
    let initial_spawn_x = spawn_region.footprint.0.checked_add(spawn_region.footprint.2.checked_div(2).unwrap_or(1)).unwrap_or(spawn_region.footprint.0);
    let initial_spawn_y = spawn_region.footprint.1.checked_add(spawn_region.footprint.3.checked_div(2).unwrap_or(1)).unwrap_or(spawn_region.footprint.1);
    let bare_level = tile_buffer.clone().into_parsed_level((initial_spawn_x, initial_spawn_y));

    // Phase 05 — Repair + validation.
    let mut repair_engine = RepairEngine::new(&normalized, factory);
    let accepted = repair_engine.repair_until_valid(
        selected_topology,
        tile_buffer,
        &bare_level,
    )?;
    let post_repair_topology = accepted.topology().clone();
    let level = accepted.lower_to_parsed_level();

    // Phase 05a — Reconstruct movement graph from final level.
    let (movement, _inferred) = reconstruct_movement_graph(&level, &post_repair_topology)?;

    // Phase 06 — Marker placement.
    let envelopes: Vec<_> = std::iter::empty().collect();
    let marker_placement = place_all_markers(
        &level,
        &post_repair_topology,
        &movement,
        &envelopes,
        &normalized,
    )?;

    // Write markers back into the ParsedLevel.
    let mut final_level = level;
    final_level.spawn = TileCoord {
        layer: usize::from(marker_placement.spawn.layer),
        x: usize::from(marker_placement.spawn.x),
        y: usize::from(marker_placement.spawn.y),
    };
    final_level.light_markers = marker_placement
        .lights
        .iter()
        .map(|light| TileCoord {
            layer: usize::from(light.coord.layer),
            x: usize::from(light.coord.x),
            y: usize::from(light.coord.y),
        })
        .collect();
    final_level.model_markers = marker_placement
        .models
        .iter()
        .map(|model| TileCoord {
            layer: usize::from(model.coord.layer),
            x: usize::from(model.coord.x),
            y: usize::from(model.coord.y),
        })
        .collect();

    // Phase 06 — Resource counting.
    let resource_counts = count_resources(
        &final_level,
        &post_repair_topology,
        marker_placement.lights.len() as u32,
        marker_placement.models.len() as u32,
        &normalized,
    )?;
    enforce_budgets(&resource_counts, &normalized)?;

    // Phase 07 — Capture views.
    let capture_views = derive_capture_views(
        &final_level,
        &post_repair_topology,
        &movement,
        &normalized,
    )?;

    // Canonical diagnostics.
    let diagnostics_bytes = diagnostics
        .with_attempt(AttemptIdentity::new(identity, 0))
        .canonical_json_bytes()?;

    Ok(GenerationResult {
        level: final_level,
        diagnostics: diagnostics_bytes,
        capture_views,
        resource_counts,
        seed,
    })
}

/// Convenience: generate with the default primary profile and given seed.
pub fn generate_default(
    catalog: &PrefabCatalog,
    seed: u64,
) -> Result<GenerationResult, GeneratorError> {
    generate(GeneratorConfig::qualified(QualifiedProfile::Primary), catalog, seed)
}
