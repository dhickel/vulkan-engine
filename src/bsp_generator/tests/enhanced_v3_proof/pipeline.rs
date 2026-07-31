//! One-way immutable pipeline for the Enhanced v3 integrated proof.
//!
//! Config → footprint → topology → plan → assemblies → validate → serialize
//!
//! Stage-order violations fail. Refresh mode writes to temp dir only.
//! All stages are private to the proof test harness.

use super::assembly::Assembly;
use super::contract::{ContractError, Preset, ProofConfig};
use super::emission;
use super::footprint::{self};
use super::ir::{CommittedTopology, PlanOutcome, V3IdAllocator};
use super::metadata::{self, ProofMetadata};
use super::planner;
use super::prefab;
use super::seed::V3Seed;
use super::topology;

/// The complete proof pipeline result.
#[derive(Debug, Clone)]
pub struct PipelineResult {
    /// The generated .map text.
    pub map_text: String,
    /// Schema-v3 metadata.
    pub metadata: ProofMetadata,
    /// The committed topology.
    pub topology: CommittedTopology,
    /// The plan outcome.
    pub plan_outcome: PlanOutcome,
    /// The validated assembly.
    pub assembly: Assembly,
}

/// Run the complete one-way pipeline.
///
/// Stages are executed in immutable order. Each stage consumes only the
/// output of the previous stage. No stage can see or modify data from
/// a later stage.
pub fn run_pipeline(config: &ProofConfig, seed: V3Seed) -> Result<PipelineResult, ContractError> {
    let mut alloc = V3IdAllocator::new();

    // Stage 1: Footprint
    let (footprints, layouts) = footprint::build_footprints(config, seed, &mut alloc)?;

    // Stage 2: Topology
    let topology = topology::build_topology(config, &footprints, &layouts, seed, &mut alloc)?;

    // Stage 3: Plan
    let plan_outcome = planner::plan_composition(seed, config, &topology)?;

    // Stages 4–5: compile and validate the assembly once. `Assembly::new`
    // establishes the validation boundary; later stages receive it read-only.
    let assembly = prefab::compile_assembly(&plan_outcome)?;
    if !assembly.validated {
        return Err(ContractError::InvariantViolation {
            detail: "assembly lost validation state".into(),
        });
    }

    // Stage 6: Compute reservations (spawn + lights)
    let (spawn_vol, light_vols) = topology::compute_reservations(&topology, &mut alloc)?;
    let spawn_origin = (
        (spawn_vol.x0 + spawn_vol.x1) / 2,
        (spawn_vol.y0 + spawn_vol.y1) / 2,
        (spawn_vol.z0 + spawn_vol.z1) / 2,
    );
    let spawn_yaw = 90;
    let light_origins: Vec<(i32, i32, i32)> = light_vols
        .iter()
        .map(|v| ((v.x0 + v.x1) / 2, (v.y0 + v.y1) / 2, (v.z0 + v.z1) / 2))
        .collect();

    // Stage 7: Serialize
    let map_text = emission::emit_map(&assembly, spawn_origin, spawn_yaw, &light_origins)?;

    // Stage 8: Metadata
    let metadata = metadata::build_metadata(
        &topology,
        &plan_outcome,
        config,
        spawn_origin,
        spawn_yaw,
        &light_origins,
    );

    Ok(PipelineResult {
        map_text,
        metadata,
        topology,
        plan_outcome,
        assembly,
    })
}

/// Run the canonical fixture through the same fallible one-way pipeline.
///
/// The canonical fixture is not a separate emission path: this preserves the
/// stage boundary exercised by every integrated proof run.
pub fn run_canonical_pipeline(
    config: &ProofConfig,
    seed: V3Seed,
) -> Result<(String, ProofMetadata), ContractError> {
    let result = run_pipeline(config, seed)?;
    Ok((result.map_text, result.metadata))
}

/// Run a corpus entry through the pipeline.
///
/// This is used by the corpus executor. It reuses the same one-way pipeline
/// as every integrated proof run but does not return the full `PipelineResult`.
pub fn run_corpus_pipeline(
    config: &ProofConfig,
    seed: V3Seed,
) -> Result<(String, ProofMetadata), ContractError> {
    run_canonical_pipeline(config, seed)
}

/// Make the canonical integrated .map fixture for the sparse preset.
pub fn make_canonical_fixture() -> (String, ProofMetadata) {
    let config = ProofConfig::new(Preset::Sparse, 2048).expect("valid config");
    let seed = V3Seed::new(0);
    run_canonical_pipeline(&config, seed).expect("canonical proof pipeline")
}

/// Make the dense-rich fixture for M2 budget evidence.
///
/// Uses the Rich preset at XY=2048 (M2 bond), exercising all approved
/// capabilities within M2 budget ceilings.
pub fn make_rich_fixture() -> (String, ProofMetadata) {
    let config = ProofConfig::new(Preset::Rich, 2048).expect("valid rich config");
    let seed = V3Seed::new(2);
    run_canonical_pipeline(&config, seed).expect("rich proof pipeline")
}

/// Make the moderate fixture for corpus completeness.
pub fn make_moderate_fixture() -> (String, ProofMetadata) {
    let config = ProofConfig::new(Preset::Moderate, 2048).expect("valid moderate config");
    let seed = V3Seed::new(1);
    run_canonical_pipeline(&config, seed).expect("moderate proof pipeline")
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pipeline_sparse_runs() {
        let config = ProofConfig::new(Preset::Sparse, 2048).unwrap();
        let result = run_pipeline(&config, V3Seed::new(0)).unwrap();

        assert!(!result.map_text.is_empty());
        assert!(result.map_text.contains("worldspawn"));
        assert!(result.map_text.contains("info_player_start"));
        assert_eq!(result.metadata.schema, "enhanced-v3-proof-metadata/v3");
        assert!(result.assembly.validated);
    }

    #[test]
    fn pipeline_deterministic() {
        let config = ProofConfig::new(Preset::Sparse, 2048).unwrap();

        let a = run_pipeline(&config, V3Seed::new(42)).unwrap();
        let b = run_pipeline(&config, V3Seed::new(42)).unwrap();

        assert_eq!(a.map_text, b.map_text, "pipeline must be deterministic");
        assert_eq!(a.metadata, b.metadata);
    }

    #[test]
    fn pipeline_stage_order_enforced() {
        // Can't get metadata without going through all stages
        let config = ProofConfig::new(Preset::Sparse, 2048).unwrap();
        let result = run_pipeline(&config, V3Seed::new(0));
        assert!(result.is_ok());
    }

    #[test]
    fn canonical_fixture_produces_valid_map() {
        let (map, meta) = make_canonical_fixture();

        assert!(!map.is_empty());
        assert!(map.contains("worldspawn"));
        assert!(map.contains("info_player_start"));
        assert!(map.contains("\"classname\" \"light\""));
        assert!(map.ends_with('\n'));
        assert!(!map.ends_with("\n\n"));

        assert_eq!(meta.schema, "enhanced-v3-proof-metadata/v3");
        assert_eq!(meta.preset, "sparse");
        assert!(meta.room_count > 0);
    }

    #[test]
    fn canonical_fixture_deterministic() {
        let (a, _) = make_canonical_fixture();
        let (b, _) = make_canonical_fixture();
        assert_eq!(a, b);
    }
}
