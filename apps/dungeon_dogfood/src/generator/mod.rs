pub(crate) mod alloc_metrics;
pub(crate) mod ascii;
pub mod capture_views;
mod config;
pub(crate) mod context;
pub mod determinism;
mod diagnostics;
mod error;
pub(crate) mod ir;
pub(crate) mod markers;
pub(crate) mod placement;
pub mod prefab;
pub(crate) mod ramps;
pub(crate) mod repair;
pub(crate) mod replay;
pub(crate) mod resources;
pub(crate) mod routing;
pub(crate) mod telemetry;
pub(crate) mod topology;
pub(crate) mod validation;

use std::path::Path;

use self::prefab::PrefabCatalog;
use crate::layout::{ParsedLevel, TileCoord};

use self::capture_views::derive_capture_views;
use self::config::NormalizedGeneratorConfig;
use self::context::{AttemptContext, TelemetryMode};
use self::determinism::{AttemptIdentity, GeneratorIdentity, SemanticStage, SemanticStreamFactory};
use self::diagnostics::GeneratorDiagnostics;
use self::markers::place_all_markers;
use self::placement::place_regions;
use self::repair::RepairEngine;
use self::resources::{count_resources, enforce_budgets};
use self::routing::materialize_topology;
use self::topology::{build_candidate_graph, select_topology};
use self::validation::reconstruct_movement_graph;

// ─── Public API ────────────────────────────────────────────────────────────

// Re-export public types.
pub use self::capture_views::{CaptureView, CaptureViewCategory};
pub use self::config::GeneratorConfig;
pub use self::config::QualifiedProfile;
pub use self::error::{ErrorStage, GeneratorError};
pub use self::resources::ResourceCounts;

/// Compute the canonical hash of the normalized configuration.
pub(crate) fn compute_config_hash(config: &GeneratorConfig) -> Result<String, GeneratorError> {
    let normalized = config.normalize()?;
    Ok(normalized.canonical_hash())
}

/// Return the error stage code string for a generator error.
pub(crate) fn error_stage_code(error: &GeneratorError) -> &'static str {
    error.stage().code()
}

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
    /// Topology region count from the selected topology.
    pub(crate) topology_region_count: u32,
    /// Topology edge count from the selected topology.
    pub(crate) topology_edge_count: u32,
    /// Route distance from spawn to distant landmark.
    pub(crate) route_distance: u64,
    /// Maximum branch depth from spawn.
    pub(crate) max_branch_depth: u32,
    /// Number of intentional dead-end regions.
    pub(crate) dead_end_count: u32,
    /// Number of articulation points.
    pub(crate) articulation_count: u32,
    /// Number of edge-crossings not sharing a region.
    pub(crate) crossing_count: u32,
    /// Per-layer cycle counts.
    pub(crate) per_layer_cycles: Vec<u32>,
}

/// Run the complete generator pipeline and return a validated level.
///
/// This is the single public entrypoint for the app. Each call runs up to
/// `generation_attempts` clean deterministic attempts. A terminal integrity
/// failure (configuration, prefab, serialization, binding, IR invariant)
/// is returned immediately. Exhaustion produces a single
/// `GenerationExhausted` error with aggregate failure-category counts.
///
/// Telemetry is disabled (Off) through the public API. The benchmark
/// path may select Counters or Timing via the package-local entrypoint.
pub fn generate(
    config: GeneratorConfig,
    catalog: &PrefabCatalog,
    seed: u64,
) -> Result<GenerationResult, GeneratorError> {
    generate_with_telemetry(config, catalog, seed, TelemetryMode::Off).map(|(result, _ctx)| result)
}

/// Internal entrypoint that accepts a telemetry mode. The context is returned
/// alongside the result so the benchmark path can serialize it.
pub(crate) fn generate_with_telemetry(
    config: GeneratorConfig,
    catalog: &PrefabCatalog,
    seed: u64,
    telemetry_mode: TelemetryMode,
) -> Result<(GenerationResult, AttemptContext), GeneratorError> {
    let normalized = config.normalize()?;
    let identity = GeneratorIdentity::new(&normalized, catalog.identity_bytes(), seed);
    run_generation_attempts(
        normalized.generation_attempts(),
        identity,
        |attempt, factory| {
            generate_attempt(&normalized, catalog, seed, attempt, factory, telemetry_mode)
        },
    )
}

fn run_generation_attempts<T>(
    attempts: u32,
    identity: GeneratorIdentity,
    mut run: impl FnMut(
        AttemptIdentity,
        SemanticStreamFactory,
    ) -> (Result<T, GeneratorError>, AttemptContext),
) -> Result<(T, AttemptContext), GeneratorError> {
    debug_assert!(attempts > 0, "normalized attempt budget must be nonzero");
    let mut last_error: Option<GeneratorError> = None;
    let mut _last_ctx: Option<AttemptContext> = None;
    let mut category_counts = std::collections::BTreeMap::new();

    for index in 0..attempts {
        let attempt_identity = AttemptIdentity::new(identity, index);
        let factory = SemanticStreamFactory::new(attempt_identity);
        let (result, mut ctx) = run(attempt_identity, factory);
        match result {
            Ok(result) => return Ok((result, ctx)),
            Err(error) if !error.is_retryable() => {
                ctx.finish_attempt();
                return Err(error);
            }
            Err(error) => {
                *category_counts
                    .entry(error.reason_code().to_owned())
                    .or_insert(0) += 1;
                last_error = Some(error);
                _last_ctx = Some(ctx);
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
/// Always returns the `AttemptContext` even on failure (already finalized).
fn generate_attempt(
    config: &NormalizedGeneratorConfig,
    catalog: &PrefabCatalog,
    seed: u64,
    attempt_identity: AttemptIdentity,
    factory: SemanticStreamFactory,
    telemetry_mode: TelemetryMode,
) -> (Result<GenerationResult, GeneratorError>, AttemptContext) {
    let mut ctx = AttemptContext::new(telemetry_mode);

    // Helper to convert ? errors into early return with finalized context.
    macro_rules! try_ctx {
        ($expr:expr) => {
            match $expr {
                Ok(v) => v,
                Err(e) => {
                    ctx.finish_attempt();
                    return (Err(e), ctx);
                }
            }
        };
    }

    let diagnostics = GeneratorDiagnostics::new(config, catalog.identity_bytes(), seed);

    // Phase 02 — Placement.
    ctx.begin_scope(context::TelemetryScope::Placement);
    let mut roles_rng = factory.stream(SemanticStage::Roles, &[]);
    let (placed_topology, grid) = try_ctx!(place_regions(
        config,
        catalog,
        &mut roles_rng,
        factory,
        &mut ctx
    ));
    ctx.end_scope(context::TelemetryScope::Placement);

    // Phase 03 — Topology.
    ctx.begin_scope(context::TelemetryScope::CandidateConstruction);
    let candidate_graph = try_ctx!(build_candidate_graph(&placed_topology, &grid, &mut ctx));
    ctx.candidate_edges = u64::try_from(candidate_graph.edges.len()).unwrap_or(u64::MAX);
    ctx.end_scope(context::TelemetryScope::CandidateConstruction);

    ctx.begin_scope(context::TelemetryScope::TopologySelection);
    let mut topology_rng = factory.stream(SemanticStage::Topology, &[]);
    let selected_topology = try_ctx!(select_topology(
        placed_topology,
        config,
        &candidate_graph,
        &mut topology_rng,
        &mut ctx,
    ));
    ctx.end_scope(context::TelemetryScope::TopologySelection);

    // Phase 04 — Materialization.
    ctx.begin_scope(context::TelemetryScope::Materialization);
    let tile_buffer = try_ctx!(materialize_topology(
        &selected_topology,
        catalog,
        config,
        &mut ctx
    ));
    ctx.end_scope(context::TelemetryScope::Materialization);

    // Build a bare ParsedLevel (no markers yet).
    let spawn_region = selected_topology
        .regions
        .iter()
        .find(|r| matches!(r.role, self::ir::RegionRole::Spawn))
        .ok_or(GeneratorError::IrInvariant {
            stage: self::error::ErrorStage::Ir,
            detail: "generate_no_spawn_region".into(),
        });
    let spawn_region = try_ctx!(spawn_region);
    let initial_spawn_x = spawn_region
        .footprint
        .0
        .checked_add(spawn_region.footprint.2.checked_div(2).unwrap_or(1))
        .unwrap_or(spawn_region.footprint.0);
    let initial_spawn_y = spawn_region
        .footprint
        .1
        .checked_add(spawn_region.footprint.3.checked_div(2).unwrap_or(1))
        .unwrap_or(spawn_region.footprint.1);
    let bare_level = tile_buffer
        .clone()
        .into_parsed_level((initial_spawn_x, initial_spawn_y));

    // Phase 05 — Repair + validation.
    // Capture topology data before selected_topology is moved into repair.
    let topology_region_count = selected_topology.regions.len() as u32;
    let topology_edge_count = selected_topology.edges.len() as u32;
    let route_distance = selected_topology.route_distance;
    let max_branch_depth = selected_topology.max_branch_depth;
    let dead_end_count = selected_topology.dead_end_count;
    let articulation_count = selected_topology.articulation_count;
    let crossing_count = selected_topology.crossing_count;
    let per_layer_cycles = selected_topology.per_layer_cycles.clone();

    ctx.begin_scope(context::TelemetryScope::Repair);
    let mut repair_engine = RepairEngine::new(config, factory, &mut ctx);
    let accepted =
        try_ctx!(repair_engine.repair_until_valid(selected_topology, tile_buffer, &bare_level,));
    ctx.end_scope(context::TelemetryScope::Repair);
    let post_repair_topology = accepted.topology().clone();
    let level = accepted.lower_to_parsed_level();

    // Phase 05a — Reconstruct movement graph from final level.
    let (movement, _inferred) = try_ctx!(reconstruct_movement_graph(&level, &post_repair_topology));

    // Phase 06 — Marker placement.
    ctx.begin_scope(context::TelemetryScope::MarkerPlacement);
    let envelopes: Vec<_> = std::iter::empty().collect();
    let marker_placement = try_ctx!(place_all_markers(
        &level,
        &post_repair_topology,
        &movement,
        &envelopes,
        config,
    ));
    ctx.markers_placed =
        u64::try_from(marker_placement.lights.len() + marker_placement.models.len() + 1)
            .unwrap_or(u64::MAX);
    ctx.lights_count(u64::try_from(marker_placement.lights.len()).unwrap_or(u64::MAX));
    ctx.models_count(u64::try_from(marker_placement.models.len()).unwrap_or(u64::MAX));
    ctx.end_scope(context::TelemetryScope::MarkerPlacement);

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
    ctx.begin_scope(context::TelemetryScope::ResourceCounting);
    let resource_counts = try_ctx!(count_resources(
        &final_level,
        &post_repair_topology,
        marker_placement.lights.len() as u32,
        marker_placement.models.len() as u32,
        config,
    ));
    ctx.resource_tiles_count(resource_counts.total_tiles);
    ctx.resource_chunks_count(u64::from(resource_counts.non_empty_chunks));
    ctx.resource_static_bodies_count(u64::from(resource_counts.static_body_count));
    ctx.resource_total_bodies_count(u64::from(resource_counts.total_body_count));
    ctx.resource_vertices_count(resource_counts.estimated_vertices);
    ctx.resource_indices_count(resource_counts.estimated_indices);
    try_ctx!(enforce_budgets(&resource_counts, config));
    ctx.end_scope(context::TelemetryScope::ResourceCounting);

    // Phase 07 — Capture views.
    ctx.begin_scope(context::TelemetryScope::CaptureViewDerivation);
    let capture_views = try_ctx!(derive_capture_views(
        &final_level,
        &post_repair_topology,
        &movement,
        config,
    ));
    ctx.capture_views_generated_count(u64::try_from(capture_views.len()).unwrap_or(u64::MAX));
    ctx.end_scope(context::TelemetryScope::CaptureViewDerivation);

    // Canonical diagnostics.
    ctx.begin_scope(context::TelemetryScope::Diagnostics);
    let diagnostics_bytes = try_ctx!(diagnostics
        .with_attempt(attempt_identity)
        .canonical_json_bytes());
    ctx.diagnostics_bytes_count(u64::try_from(diagnostics_bytes.len()).unwrap_or(u64::MAX));
    ctx.end_scope(context::TelemetryScope::Diagnostics);

    // Finalize telemetry AFTER canonical outcome is determined.
    ctx.finish_attempt();

    (
        Ok(GenerationResult {
            level: final_level,
            diagnostics: diagnostics_bytes,
            capture_views,
            resource_counts,
            seed,
            attempt_index: attempt_identity.index(),
            topology_region_count,
            topology_edge_count,
            route_distance,
            max_branch_depth,
            dead_end_count,
            articulation_count,
            crossing_count,
            per_layer_cycles,
        }),
        ctx,
    )
}

/// Convenience: generate with the default primary profile and given seed.
pub fn generate_default(
    catalog: &PrefabCatalog,
    seed: u64,
) -> Result<GenerationResult, GeneratorError> {
    generate(
        GeneratorConfig::qualified(QualifiedProfile::Primary),
        catalog,
        seed,
    )
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

    fn ctx() -> AttemptContext {
        AttemptContext::new(TelemetryMode::Off)
    }

    #[test]
    fn retry_loop_uses_fresh_deterministic_attempt_identities_and_records_winner() {
        let generator = identity();
        let mut observed = Vec::new();
        let (winning, _) = run_generation_attempts(3, generator, |attempt, factory| {
            let mut stream = factory.stream(SemanticStage::Roles, &[]);
            observed.push((attempt.index(), attempt.bytes(), stream.next_u32()));
            if attempt.index() < 2 {
                (Err(retryable("placement_retry")), ctx())
            } else {
                (Ok(attempt.index()), ctx())
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
        let (winner, _) = run_generation_attempts(4, identity(), |attempt, _| {
            calls += 1;
            (Ok(attempt.index()), ctx())
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
            (
                Err(GeneratorError::IrInvariant {
                    stage: ErrorStage::Ir,
                    detail: "broken_ir".into(),
                }),
                ctx(),
            )
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
            (
                Err(if attempt.index() == 1 {
                    retryable("candidate_disconnected")
                } else {
                    retryable("placement_retry")
                }),
                ctx(),
            )
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

#[cfg(test)]
mod qualification_tests {
    use super::*;
    use std::path::PathBuf;

    fn prefab_catalog() -> PrefabCatalog {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets/prefabs");
        PrefabCatalog::load(&path).expect("prefab catalog must load")
    }

    #[test]
    fn primary_seed_77_known_feasible_attempt_is_accepted() {
        const SEED: u64 = 77;
        const FEASIBLE_ATTEMPT: u32 = 79;

        let catalog = prefab_catalog();
        let config = GeneratorConfig::qualified(QualifiedProfile::Primary)
            .normalize()
            .expect("primary config");
        let identity = GeneratorIdentity::new(&config, catalog.identity_bytes(), SEED);
        let attempt = AttemptIdentity::new(identity, FEASIBLE_ATTEMPT);
        let (gen_result, _ctx) = generate_attempt(
            &config,
            &catalog,
            SEED,
            attempt,
            SemanticStreamFactory::new(attempt),
            TelemetryMode::Off,
        );
        let result =
            gen_result.expect("seed 77 feasible attempt must pass generation and acceptance");

        super::ascii::round_trip_exact(&result.level)
            .expect("marker-complete seed 77 output must round-trip exactly");
        assert_eq!(result.seed, SEED);
        assert_eq!(result.attempt_index, FEASIBLE_ATTEMPT);
        assert_eq!(result.level.width, usize::from(config.width()));
        assert_eq!(result.level.height, usize::from(config.height()));
        assert_eq!(result.level.layer_count(), usize::from(config.layers().2));
        assert!(!result.level.light_markers.is_empty());
        assert!(
            result.level.light_markers.len()
                <= usize::try_from(config.max_lights()).expect("light cap fits usize")
        );
    }

    /// Full seed matrix: unrelaxed Primary profile, seeds 0..99.
    /// Ignored by default; run manually with:
    ///   cargo test -p dungeon_dogfood --release -- --ignored --nocapture seed_matrix_primary_0_99
    ///
    /// NOTE: As of 2026-07-19, the unrelaxed path does not pass the acceptance
    /// gate. The workaround flags `single_bottleneck=true` and
    /// `relax_transition_redundancy=true` (or `relax_route_redundancy=true`)
    /// are still required in `build_generator_config()`. This test reports
    /// results without asserting pass/fail so it can be used to track
    /// progress toward gate closure.
    #[test]
    #[ignore]
    fn seed_matrix_primary_0_99() {
        let catalog = prefab_catalog();
        let config = GeneratorConfig::qualified(QualifiedProfile::Primary);
        let mut successes = 0u32;
        let mut failures: Vec<(u64, String)> = Vec::new();
        let mut attempt_zero = 0u32;
        let mut attempt_distribution = std::collections::BTreeMap::<u32, u32>::new();
        let mut failure_categories = std::collections::BTreeMap::<String, u32>::new();

        for seed in 0..100u64 {
            match generate(config.clone(), &catalog, seed) {
                Ok(result) => {
                    successes += 1;
                    assert!(result.level.width > 0, "seed {seed}: zero width");
                    assert!(result.level.height > 0, "seed {seed}: zero height");
                    assert!(result.level.layer_count() > 0, "seed {seed}: zero layers");
                    assert!(
                        !result.level.light_markers.is_empty(),
                        "seed {seed}: no lights"
                    );
                    if result.attempt_index == 0 {
                        attempt_zero += 1;
                    }
                    *attempt_distribution
                        .entry(result.attempt_index)
                        .or_insert(0) += 1;
                }
                Err(e) => {
                    let reason = e.reason_code().to_owned();
                    *failure_categories.entry(reason.clone()).or_insert(0) += 1;
                    failures.push((seed, reason));
                }
            }
        }

        let total = successes as usize + failures.len();
        eprintln!("\n=== Seed Matrix Results (Unrelaxed Primary, seeds 0..99) ===");
        eprintln!("  Successes:        {successes}/{total}");
        eprintln!("  Attempt-zero:     {attempt_zero}/{total}");
        eprintln!("  Failures:         {}/{total}", failures.len());
        if !attempt_distribution.is_empty() {
            eprintln!("  Attempt distribution:");
            for (idx, count) in &attempt_distribution {
                eprintln!("    attempt {idx}: {count}");
            }
        }
        if !failure_categories.is_empty() {
            eprintln!("\n  Failure categories:");
            for (cat, count) in &failure_categories {
                eprintln!("    {cat}: {count}");
            }
        }
        if !failures.is_empty() {
            eprintln!("\n  Failure seeds:");
            for (seed, reason) in &failures {
                eprintln!("    seed={seed}: {reason}");
            }
        }
        eprintln!("=== End Matrix ===\n");

        // Report the gate status but do not assert — the unrelaxed path is
        // not yet passing. When this test reports 100 successes, the
        // workarounds in build_generator_config() can be removed.
        if !failures.is_empty() {
            eprintln!(
                "GATE NOT MET: {}/100 failed. Workarounds still required.",
                failures.len()
            );
        } else if attempt_zero < 80 {
            eprintln!("GATE PARTIALLY MET: all accepted but only {attempt_zero}/100 attempt-zero.");
        } else {
            eprintln!("GATE MET: 100/100 accepted, {attempt_zero}/100 attempt-zero.");
        }
    }

    /// Off/Counters/Timing equivalence: all three modes produce exact same
    /// canonical artifacts for a known-success Primary configuration.
    /// Ignored by default due to execution time; run with:
    ///   cargo test --release -p dungeon_dogfood -- --ignored telemetry_modes
    #[test]
    #[ignore]
    fn telemetry_modes_produce_identical_canonical_artifacts() {
        let catalog = prefab_catalog();

        const SEED: u64 = 77;

        let modes = [
            TelemetryMode::Off,
            TelemetryMode::Counters,
            TelemetryMode::Timing,
        ];

        let mut results: Vec<(TelemetryMode, GenerationResult)> = Vec::new();
        for &mode in &modes {
            let (result, _ctx) = generate_with_telemetry(
                GeneratorConfig::qualified(QualifiedProfile::Primary),
                &catalog,
                SEED,
                mode,
            )
            .expect(&format!("seed {} must succeed in {:?} mode", SEED, mode));
            results.push((mode, result));
        }

        // Compare all pairs — confirm exact equality of canonical artifacts.
        for i in 0..results.len() {
            for j in (i + 1)..results.len() {
                let (mode_a, ref a) = results[i];
                let (mode_b, ref b) = results[j];

                assert_eq!(
                    a.seed, b.seed,
                    "seed differs between {:?} and {:?}",
                    mode_a, mode_b
                );
                assert_eq!(
                    a.attempt_index, b.attempt_index,
                    "attempt_index differs between {:?} and {:?}",
                    mode_a, mode_b
                );
                assert_eq!(
                    a.diagnostics, b.diagnostics,
                    "diagnostics differ between {:?} and {:?}",
                    mode_a, mode_b
                );

                // Also compare the level ASCII
                let ascii_a = super::ascii::serialize_level(&a.level).expect("serialize a");
                let ascii_b = super::ascii::serialize_level(&b.level).expect("serialize b");
                assert_eq!(
                    ascii_a, ascii_b,
                    "level ASCII differs between {:?} and {:?}",
                    mode_a, mode_b
                );
            }
        }

        // Timing mode must produce timing data.
        let (_timing_result, timing_ctx) = generate_with_telemetry(
            GeneratorConfig::qualified(QualifiedProfile::Primary),
            &catalog,
            SEED,
            TelemetryMode::Timing,
        )
        .expect("timing re-run");
        assert!(
            timing_ctx.timing_entries().next().is_some(),
            "timing mode should have timing data"
        );
    }
}
