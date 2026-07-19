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
    /// Zero-based index of the winning attempt.
    pub attempt_index: u32,
}

/// Run the complete generator pipeline and return a validated level.
///
/// This is the single public entrypoint for the app. Each call runs up to
/// `generation_attempts` clean deterministic attempts. A terminal integrity
/// failure (configuration, prefab, serialization, binding, IR invariant)
/// is returned immediately. Exhaustion produces a single
/// `GenerationExhausted` error with aggregate failure-category counts.
pub fn generate(
    config: GeneratorConfig,
    catalog: &PrefabCatalog,
    seed: u64,
) -> Result<GenerationResult, GeneratorError> {
    let normalized = config.normalize()?;
    let identity = GeneratorIdentity::new(&normalized, catalog.identity_bytes(), seed);
    run_generation_attempts(normalized.generation_attempts(), identity, |attempt, factory| {
        generate_attempt(&normalized, catalog, seed, attempt, factory)
    })
}

fn run_generation_attempts<T>(
    attempts: u32,
    identity: GeneratorIdentity,
    mut run: impl FnMut(AttemptIdentity, SemanticStreamFactory) -> Result<T, GeneratorError>,
) -> Result<T, GeneratorError> {
    debug_assert!(attempts > 0, "normalized attempt budget must be nonzero");
    let mut last_error: Option<GeneratorError> = None;
    let mut category_counts = std::collections::BTreeMap::new();

    for index in 0..attempts {
        let attempt_identity = AttemptIdentity::new(identity, index);
        let factory = SemanticStreamFactory::new(attempt_identity);
        match run(attempt_identity, factory) {
            Ok(result) => return Ok(result),
            Err(error) if !error.is_retryable() => return Err(error),
            Err(error) => {
                *category_counts
                    .entry(error.reason_code().to_owned())
                    .or_insert(0) += 1;
                last_error = Some(error);
            }
        }
    }

    let last = last_error.expect("normalized attempt budget must be nonzero");
    Err(GeneratorError::GenerationExhausted {
        attempts,
        last_stage: last.stage(),
        last_reason: last.reason_code().to_owned(),
        category_counts,
    })
}

/// Single clean attempt at the full generator pipeline.
fn generate_attempt(
    config: &NormalizedGeneratorConfig,
    catalog: &PrefabCatalog,
    seed: u64,
    attempt_identity: AttemptIdentity,
    factory: SemanticStreamFactory,
) -> Result<GenerationResult, GeneratorError> {
    let diagnostics = GeneratorDiagnostics::new(config, catalog.identity_bytes(), seed);

    // Phase 02 — Placement.
    let mut roles_rng = factory.stream(SemanticStage::Roles, &[]);
    let (placed_topology, grid) = place_regions(config, catalog, &mut roles_rng, factory)?;

    // Phase 03 — Topology.
    let candidate_graph = build_candidate_graph(&placed_topology, &grid)?;
    let mut topology_rng = factory.stream(SemanticStage::Topology, &[]);
    let selected_topology = select_topology(
        placed_topology,
        config,
        &candidate_graph,
        &mut topology_rng,
    )?;

    // Phase 04 — Materialization.
    let tile_buffer = materialize_topology(&selected_topology, catalog, config)?;

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
    let mut repair_engine = RepairEngine::new(config, factory);
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
        config,
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
        config,
    )?;
    enforce_budgets(&resource_counts, config)?;

    // Phase 07 — Capture views.
    let capture_views = derive_capture_views(
        &final_level,
        &post_repair_topology,
        &movement,
        config,
    )?;

    // Canonical diagnostics.
    let diagnostics_bytes = diagnostics
        .with_attempt(attempt_identity)
        .canonical_json_bytes()?;

    Ok(GenerationResult {
        level: final_level,
        diagnostics: diagnostics_bytes,
        capture_views,
        resource_counts,
        seed,
        attempt_index: attempt_identity.index(),
    })
}

/// Convenience: generate with the default primary profile and given seed.
pub fn generate_default(
    catalog: &PrefabCatalog,
    seed: u64,
) -> Result<GenerationResult, GeneratorError> {
    generate(GeneratorConfig::qualified(QualifiedProfile::Primary), catalog, seed)
}

#[cfg(test)]
mod attempt_tests {
    use super::*;

    fn identity() -> GeneratorIdentity {
        let config = GeneratorConfig::qualified(QualifiedProfile::Minimum)
            .normalize()
            .unwrap();
        GeneratorIdentity::new(&config, [0x5a; 32], 17)
    }

    fn retryable(reason: &'static str) -> GeneratorError {
        GeneratorError::PlacementExhausted {
            stage: ErrorStage::Placement,
            reason,
            attempted: 1,
            placed: 0,
            target: 1,
        }
    }

    #[test]
    fn retry_loop_uses_fresh_deterministic_attempt_identities_and_records_winner() {
        let generator = identity();
        let mut observed = Vec::new();
        let winning = run_generation_attempts(3, generator, |attempt, factory| {
            let mut stream = factory.stream(SemanticStage::Roles, &[]);
            observed.push((attempt.index(), attempt.bytes(), stream.next_u32()));
            if attempt.index() < 2 {
                Err(retryable("placement_retry"))
            } else {
                Ok(attempt.index())
            }
        })
        .unwrap();

        assert_eq!(winning, 2);
        assert_eq!(observed.len(), 3);
        for (index, bytes, first_roll) in observed {
            let expected = AttemptIdentity::new(generator, index);
            let mut expected_stream =
                SemanticStreamFactory::new(expected).stream(SemanticStage::Roles, &[]);
            assert_eq!(bytes, expected.bytes());
            assert_eq!(first_roll, expected_stream.next_u32());
        }
    }

    #[test]
    fn index_zero_success_preserves_single_attempt_behavior() {
        let mut calls = 0;
        let winner = run_generation_attempts(4, identity(), |attempt, _| {
            calls += 1;
            Ok(attempt.index())
        })
        .unwrap();
        assert_eq!(winner, 0);
        assert_eq!(calls, 1);
    }

    #[test]
    fn terminal_error_stops_without_concealing_integrity_failure() {
        let mut calls = 0;
        let error = run_generation_attempts::<()>(4, identity(), |_, _| {
            calls += 1;
            Err(GeneratorError::IrInvariant {
                stage: ErrorStage::Ir,
                detail: "broken_ir".into(),
            })
        })
        .unwrap_err();
        assert_eq!(calls, 1);
        assert!(matches!(error, GeneratorError::IrInvariant { .. }));
    }

    #[test]
    fn exhaustion_has_exact_attempt_and_ordered_category_counts() {
        let mut calls = 0;
        let error = run_generation_attempts::<()>(3, identity(), |attempt, _| {
            calls += 1;
            Err(if attempt.index() == 1 {
                retryable("candidate_disconnected")
            } else {
                retryable("placement_retry")
            })
        })
        .unwrap_err();

        assert_eq!(calls, 3);
        assert_eq!(error.stage(), ErrorStage::Generation);
        assert_eq!(error.reason_code(), "generation_exhausted");
        match error {
            GeneratorError::GenerationExhausted {
                attempts,
                last_stage,
                last_reason,
                category_counts,
            } => {
                assert_eq!(attempts, 3);
                assert_eq!(last_stage, ErrorStage::Placement);
                assert_eq!(last_reason, "placement_retry");
                assert_eq!(category_counts.values().copied().sum::<u32>(), attempts);
                assert_eq!(
                    category_counts.into_iter().collect::<Vec<_>>(),
                    vec![
                        ("candidate_disconnected".into(), 1),
                        ("placement_retry".into(), 2),
                    ]
                );
            }
            other => panic!("expected exhaustion, got {other:?}"),
        }
    }
}
