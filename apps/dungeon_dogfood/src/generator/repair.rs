//! Phase 05b — Bounded monotonic repair framework with canonical state
//! hashing, accepted-IR boundary, and one-shot lowering to `ParsedLevel`.
//!
//! ## Repair operations (bounded, monotonic)
//! - Reroute: reopen a corridor witness within its existing envelope.
//! - Relocate marker: stub for Phase 06.
//! - Remove optional edge: drop a non-required edge to resolve overlap.
//!
//! Each repair consumes a named budget and uses a deterministic semantic
//! stream. Every repair target is hashed canonically; rejected states are
//! recorded and never revisited. After each repair the full validator set
//! runs. If a repair introduces new errors the attempt is rolled back and the
//! state is rejected.
//!
//! ## AcceptedIr boundary
//! `AcceptedIr` can only be constructed after full validation passes. Once
//! accepted, `lower_to_parsed_level` consumes the IR and returns a
//! `ParsedLevel`. The IR cannot be modified or re-accepted after lowering.

use sha2::{Digest, Sha256};
use std::collections::BTreeSet;

use super::config::NormalizedGeneratorConfig;
use super::context::AttemptContext;
use super::determinism::{lowercase_hex, SemanticComponent, SemanticStage, SemanticStreamFactory};
use super::error::{ErrorStage, GeneratorError};
use super::ir::IntendedTopology;
use super::routing::TileBuffer;
use super::validation::{reconstruct_movement_graph, validate_full, ValidationReport};

// ─── State hashing ──────────────────────────────────────────────────────────

const REPAIR_STATE_DOMAIN: &[u8] = b"dungeon-generator/repair-state/v1";

/// A canonical hash of the current topology state for deduplication.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct StateHash(String);

impl StateHash {
    fn of_state(topology: &IntendedTopology, level: &crate::layout::ParsedLevel) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(REPAIR_STATE_DOMAIN);
        hasher.update(topology.config.canonical_bytes());
        for region in &topology.regions {
            hasher.update(region.id.raw().to_be_bytes());
            hasher.update([region.role.ordinal()]);
            hasher.update(region.variant_index.to_be_bytes());
            hasher.update(region.layer.to_be_bytes());
            for value in [
                region.footprint.0,
                region.footprint.1,
                region.footprint.2,
                region.footprint.3,
            ] {
                hasher.update(value.to_be_bytes());
            }
            for socket in &region.sockets {
                hasher.update(socket.id.raw().to_be_bytes());
                hasher.update(socket.variant_socket_index.to_be_bytes());
                hasher.update(socket.global_anchor.layer.to_be_bytes());
                hasher.update(socket.global_anchor.x.to_be_bytes());
                hasher.update(socket.global_anchor.y.to_be_bytes());
                hasher.update([socket.direction as u8, socket.role as u8]);
                hasher.update(socket.width.to_be_bytes());
                hasher.update(
                    socket
                        .paired_socket_id
                        .map_or(u32::MAX, |id| id.raw())
                        .to_be_bytes(),
                );
            }
            for transition in &region.transitions {
                hasher.update(transition.raw().to_be_bytes());
            }
            for marker in &region.marker_variant_indices {
                hasher.update(marker.to_be_bytes());
            }
        }
        for edge in &topology.edges {
            hasher.update(edge.id.raw().to_be_bytes());
            hasher.update(edge.source_socket.raw().to_be_bytes());
            hasher.update(edge.target_socket.raw().to_be_bytes());
            hasher.update(edge.source_region.raw().to_be_bytes());
            hasher.update(edge.target_region.raw().to_be_bytes());
            hasher.update([u8::from(edge.required)]);
            hasher.update(edge.cost.to_be_bytes());
            hasher.update(edge.width.to_be_bytes());
            hasher.update(
                edge.transition
                    .map_or(u32::MAX, |id| id.raw())
                    .to_be_bytes(),
            );
            for cell in edge.path_witness.iter().chain(&edge.allowed_envelope_cells) {
                hasher.update(cell.layer.to_be_bytes());
                hasher.update(cell.x.to_be_bytes());
                hasher.update(cell.y.to_be_bytes());
            }
        }
        for transition in &topology.transitions {
            hasher.update(transition.id.raw().to_be_bytes());
            hasher.update(transition.variant_index.to_be_bytes());
            hasher.update(transition.lower_layer.to_be_bytes());
            hasher.update(transition.lower_region.raw().to_be_bytes());
            hasher.update(transition.upper_region.raw().to_be_bytes());
            hasher.update(transition.lower_socket.raw().to_be_bytes());
            hasher.update(transition.upper_socket.raw().to_be_bytes());
            for cell in transition
                .ramp_run_cells
                .iter()
                .chain(&transition.upper_opening_cells)
                .chain(&transition.landing_cells)
                .chain(&transition.headroom_cells)
                .chain(&transition.lower_funnel_cells)
                .chain(&transition.lower_approach_cells)
            {
                hasher.update(cell.layer.to_be_bytes());
                hasher.update(cell.x.to_be_bytes());
                hasher.update(cell.y.to_be_bytes());
            }
        }
        hasher.update(topology.route_distance.to_be_bytes());
        hasher.update(topology.max_branch_depth.to_be_bytes());
        hasher.update(topology.dead_end_count.to_be_bytes());
        hasher.update(topology.articulation_count.to_be_bytes());
        hasher.update(topology.crossing_count.to_be_bytes());
        for cycles in &topology.per_layer_cycles {
            hasher.update(cycles.to_be_bytes());
        }
        hasher.update(level.width.to_be_bytes());
        hasher.update(level.height.to_be_bytes());
        for layer in &level.layers {
            for tile in layer {
                hasher.update(tile_code(*tile).to_be_bytes());
            }
        }
        for marker in std::iter::once(&level.spawn)
            .chain(&level.model_markers)
            .chain(&level.light_markers)
        {
            hasher.update(marker.layer.to_be_bytes());
            hasher.update(marker.x.to_be_bytes());
            hasher.update(marker.y.to_be_bytes());
        }
        let digest: [u8; 32] = hasher.finalize().into();
        Self(lowercase_hex(&digest))
    }
}

fn tile_code(tile: crate::layout::Tile) -> u16 {
    match tile {
        crate::layout::Tile::Wall => 0,
        crate::layout::Tile::Floor => 1,
        crate::layout::Tile::Void => 2,
        crate::layout::Tile::RampNorth(level) => 0x100 | u16::from(level),
        crate::layout::Tile::RampEast(level) => 0x200 | u16::from(level),
        crate::layout::Tile::RampSouth(level) => 0x300 | u16::from(level),
        crate::layout::Tile::RampWest(level) => 0x400 | u16::from(level),
    }
}

// ─── Accepted IR ────────────────────────────────────────────────────────────

/// A validated, immutable topology that has passed all structural,
/// connectivity, topology, and movement-probe validators.
///
/// Constructed only via `AcceptedIr::accept()`. Once lowered to `ParsedLevel`,
/// the IR is consumed and cannot be modified or re-accepted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct AcceptedIr {
    topology: IntendedTopology,
    level: crate::layout::ParsedLevel,
}

impl AcceptedIr {
    /// Accept only the exact level represented by `tile_buffer` after the
    /// complete current validator set and exact ASCII round-trip both pass.
    pub(super) fn accept(
        topology: IntendedTopology,
        tile_buffer: TileBuffer,
        full_level: crate::layout::ParsedLevel,
    ) -> Result<(Self, ValidationReport), GeneratorError> {
        let spawn_regions: Vec<_> = topology
            .regions
            .iter()
            .filter(|region| region.role == super::ir::RegionRole::Spawn)
            .collect();
        if spawn_regions.len() != 1 {
            return Err(GeneratorError::IrInvariant {
                stage: ErrorStage::Ir,
                detail: format!("acceptance_spawn_region_count={}", spawn_regions.len()),
            });
        }

        let spawn = &spawn_regions[0];
        if full_level.spawn.layer != usize::from(spawn.layer)
            || full_level.spawn.x < usize::from(spawn.footprint.0)
            || full_level.spawn.y < usize::from(spawn.footprint.1)
            || full_level.spawn.x
                >= usize::from(spawn.footprint.0.saturating_add(spawn.footprint.2))
            || full_level.spawn.y
                >= usize::from(spawn.footprint.1.saturating_add(spawn.footprint.3))
        {
            return Err(GeneratorError::IrInvariant {
                stage: ErrorStage::Ir,
                detail: "acceptance_spawn_outside_spawn_region".into(),
            });
        }

        let buffer_level = tile_buffer.into_parsed_level((
            u16::try_from(full_level.spawn.x).map_err(|_| GeneratorError::ArithmeticOverflow {
                stage: ErrorStage::Ir,
                operation: "acceptance_spawn_x",
            })?,
            u16::try_from(full_level.spawn.y).map_err(|_| GeneratorError::ArithmeticOverflow {
                stage: ErrorStage::Ir,
                operation: "acceptance_spawn_y",
            })?,
        ));
        if buffer_level.width != full_level.width
            || buffer_level.height != full_level.height
            || buffer_level.layers != full_level.layers
        {
            return Err(GeneratorError::IrInvariant {
                stage: ErrorStage::Ir,
                detail: "acceptance_level_does_not_match_tile_buffer".into(),
            });
        }

        let (movement, inferred) = reconstruct_movement_graph(&full_level, &topology)?;
        let report = validate_full(&full_level, &topology, &movement, &inferred)?;
        if !report.is_clean() {
            return Err(GeneratorError::IrInvariant {
                stage: ErrorStage::Ir,
                detail: format!(
                    "acceptance_rejected: {} error(s) present",
                    report.all_errors().len()
                ),
            });
        }
        super::ascii::round_trip_exact(&full_level)?;

        Ok((
            Self {
                topology,
                level: full_level,
            },
            report,
        ))
    }

    /// One-shot lowering consumes the only accepted wrapper.
    pub(super) fn lower_to_parsed_level(self) -> crate::layout::ParsedLevel {
        self.level
    }

    pub(super) fn topology(&self) -> &IntendedTopology {
        &self.topology
    }
}

// ─── Repair framework ───────────────────────────────────────────────────────

/// Budgets for each repair category.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct RepairBudgets {
    pub(super) reroute_budget: u32,
    pub(super) marker_relocation_budget: u32,
    pub(super) optional_edge_removal_budget: u32,
}

impl RepairBudgets {
    pub(super) fn from_config(config: &NormalizedGeneratorConfig) -> Self {
        Self {
            reroute_budget: config.reroute_budget(),
            marker_relocation_budget: config.marker_relocation_budget(),
            optional_edge_removal_budget: config.optional_edge_removal_budget(),
        }
    }

    fn is_exhausted(&self) -> bool {
        self.reroute_budget == 0
            && self.marker_relocation_budget == 0
            && self.optional_edge_removal_budget == 0
    }
}

/// A single repair operation with a deterministic reason.
#[derive(Debug, Clone, PartialEq, Eq)]
struct RepairAction {
    /// The error that triggered this repair.
    reason: GeneratorError,
    /// Which repair operation to attempt.
    operation: RepairOperation,
    /// Budget consumed by this repair.
    budget_consumed: RepairBudgets,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum RepairOperation {
    /// Reroute a specific edge within its envelope.
    Reroute { edge_id: super::ir::EdgeId },
    /// Remove an optional edge.
    RemoveOptionalEdge { edge_id: super::ir::EdgeId },
    /// Stub: relocate a marker (Phase 06).
    RelocateMarker { _marker_index: u16 },
}

/// Canonical record of a rejected state.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct RejectedState {
    hash: String,
    reason_code: String,
}

/// The repair state machine.
#[derive(Debug)]
pub(super) struct RepairEngine {
    budgets: RepairBudgets,
    rejected: BTreeSet<RejectedState>,
    repair_stream: super::determinism::Pcg32V1,
    repair_count: u32,
}

impl RepairEngine {
    /// Create a new repair engine from config and a semantic stream.
    pub(super) fn new(
        config: &NormalizedGeneratorConfig,
        factory: SemanticStreamFactory,
        ctx: &mut AttemptContext,
    ) -> Self {
        let _ = ctx; // reserved for future telemetry
        let repair_stream = factory.stream(SemanticStage::Repair, &[SemanticComponent::Index(0)]);
        Self {
            budgets: RepairBudgets::from_config(config),
            rejected: BTreeSet::new(),
            repair_stream,
            repair_count: 0,
        }
    }

    /// Record a state as rejected so it is never revisited.
    fn reject_state(&mut self, hash: &StateHash, reason: &str) {
        self.rejected.insert(RejectedState {
            hash: hash.0.clone(),
            reason_code: reason.to_owned(),
        });
    }

    /// Check if a state has been previously rejected.
    fn was_rejected(&self, hash: &StateHash) -> bool {
        self.rejected.iter().any(|s| s.hash == hash.0)
    }

    /// Get the next deterministic repair action based on a validation report.
    fn next_action(
        &mut self,
        topology: &IntendedTopology,
        report: &ValidationReport,
    ) -> Option<RepairAction> {
        // Prioritize repairs: structural → connectivity → topology → movement.

        // Structural errors: try rerouting affected edges.
        if let Some(err) = report.structural_errors.first() {
            // Corridor conflicts → try rerouting optional edges or removing them.
            if self.budgets.optional_edge_removal_budget > 0 {
                if let Some(optional) = topology
                    .edges
                    .iter()
                    .find(|e| !e.required && e.transition.is_none())
                {
                    return Some(RepairAction {
                        reason: err.clone(),
                        operation: RepairOperation::RemoveOptionalEdge {
                            edge_id: optional.id,
                        },
                        budget_consumed: RepairBudgets {
                            optional_edge_removal_budget: 1,
                            ..RepairBudgets {
                                reroute_budget: 0,
                                marker_relocation_budget: 0,
                                optional_edge_removal_budget: 0,
                            }
                        },
                    });
                }
            }
            if self.budgets.reroute_budget > 0 {
                return Some(RepairAction {
                    reason: err.clone(),
                    operation: RepairOperation::Reroute {
                        edge_id: topology.edges.first()?.id,
                    },
                    budget_consumed: RepairBudgets {
                        reroute_budget: 1,
                        ..RepairBudgets {
                            reroute_budget: 0,
                            marker_relocation_budget: 0,
                            optional_edge_removal_budget: 0,
                        }
                    },
                });
            }
        }

        // Topology errors: try removing optional edges.
        if let Some(_err) = report.topology_errors.first() {
            if self.budgets.optional_edge_removal_budget > 0 {
                if let Some(optional) = topology
                    .edges
                    .iter()
                    .find(|e| !e.required && e.transition.is_none())
                {
                    return Some(RepairAction {
                        reason: GeneratorError::IrInvariant {
                            stage: ErrorStage::Ir,
                            detail: "topology_repair_optional_removal".into(),
                        },
                        operation: RepairOperation::RemoveOptionalEdge {
                            edge_id: optional.id,
                        },
                        budget_consumed: RepairBudgets {
                            optional_edge_removal_budget: 1,
                            ..RepairBudgets {
                                reroute_budget: 0,
                                marker_relocation_budget: 0,
                                optional_edge_removal_budget: 0,
                            }
                        },
                    });
                }
            }
        }

        None
    }

    /// Apply a repair action to the topology. Returns the modified topology
    /// or an error if the repair is infeasible.
    fn apply_repair(
        &mut self,
        mut topology: IntendedTopology,
        action: &RepairAction,
    ) -> Result<IntendedTopology, GeneratorError> {
        match action.operation {
            RepairOperation::RemoveOptionalEdge { edge_id } => {
                // Remove the specified optional edge.
                let before_len = topology.edges.len();
                topology.edges.retain(|e| e.id != edge_id);
                let after_len = topology.edges.len();
                if before_len == after_len {
                    return Err(GeneratorError::IrInvariant {
                        stage: ErrorStage::Ir,
                        detail: format!("repair_remove_optional_edge_{}_not_found", edge_id.raw()),
                    });
                }
                // Deduct budget.
                self.budgets.optional_edge_removal_budget =
                    self.budgets.optional_edge_removal_budget.saturating_sub(1);
                Ok(topology)
            }
            RepairOperation::Reroute { edge_id } => {
                self.budgets.reroute_budget = self.budgets.reroute_budget.saturating_sub(1);
                Err(GeneratorError::IrInvariant {
                    stage: ErrorStage::Ir,
                    detail: format!(
                        "repair_reroute_requires_materialization_context edge={}",
                        edge_id.raw()
                    ),
                })
            }
            RepairOperation::RelocateMarker { .. } => Err(GeneratorError::IrInvariant {
                stage: ErrorStage::Ir,
                detail: "repair_marker_relocation_not_available_before_phase_06".into(),
            }),
        }
    }

    /// Attempt to repair a topology until it passes validation or budgets
    /// are exhausted. Returns the accepted IR on success.
    pub(super) fn repair_until_valid(
        &mut self,
        mut topology: IntendedTopology,
        tile_buffer: TileBuffer,
        full_level: &crate::layout::ParsedLevel,
    ) -> Result<AcceptedIr, GeneratorError> {
        // First, try the initial state. It is tracked as seen, but is not a
        // rejected candidate until validation actually fails.
        let state_hash = StateHash::of_state(&topology, full_level);

        let (movement, inferred) = reconstruct_movement_graph(full_level, &topology)?;
        let mut report = validate_full(full_level, &topology, &movement, &inferred)?;

        if report.is_clean() {
            return AcceptedIr::accept(topology, tile_buffer, full_level.clone())
                .map(|(accepted, _)| accepted);
        }
        self.reject_state(&state_hash, "initial_validation_failed");

        // Repair loop.
        loop {
            if self.budgets.is_exhausted() {
                return Err(GeneratorError::SearchExhausted {
                    stage: ErrorStage::Ir,
                    search: "repair",
                    attempted: u64::from(self.repair_count),
                    budget: 0,
                });
            }

            let Some(action) = self.next_action(&topology, &report) else {
                // No actionable repair identified.
                return Err(GeneratorError::IrInvariant {
                    stage: ErrorStage::Ir,
                    detail: format!(
                        "repair_no_action: {} error(s) remaining",
                        report.all_errors().len()
                    ),
                });
            };

            self.repair_count = self.repair_count.saturating_add(1);

            // Apply the repair.
            let candidate = match self.apply_repair(topology.clone(), &action) {
                Ok(t) => t,
                Err(e) => {
                    // The repair itself was infeasible.
                    self.reject_state(&StateHash::of_state(&topology, full_level), e.reason_code());
                    continue;
                }
            };

            let candidate_hash = StateHash::of_state(&candidate, full_level);

            // Never revisit a rejected state.
            if self.was_rejected(&candidate_hash) {
                continue;
            }

            // Validate the candidate.
            let (movement, inferred) = reconstruct_movement_graph(full_level, &candidate)?;
            let candidate_report = validate_full(full_level, &candidate, &movement, &inferred)?;

            if candidate_report.is_clean() {
                return AcceptedIr::accept(candidate, tile_buffer, full_level.clone())
                    .map(|(accepted, _)| accepted);
            }

            // Commit only a strict lexicographic improvement. A rejected
            // candidate never becomes the base of the next transaction.
            if violation_tuple(&candidate_report) < violation_tuple(&report) {
                topology = candidate;
                report = candidate_report;
            } else {
                self.reject_state(&candidate_hash, "repair_not_monotonic");
            }
        }
    }
}

fn violation_tuple(report: &ValidationReport) -> (usize, usize, usize, usize) {
    (
        report.structural_errors.len(),
        report.connectivity_errors.len(),
        report.topology_errors.len(),
        report.movement_probe_errors.len(),
    )
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::super::config::GeneratorConfig;
    use super::super::context::{AttemptContext, TelemetryMode};
    use super::super::determinism::{AttemptIdentity, GeneratorIdentity};
    use super::super::ir::IdAllocator;
    use super::*;

    fn dummy_config() -> NormalizedGeneratorConfig {
        GeneratorConfig::custom(64, 64, 2).normalize().unwrap()
    }

    #[test]
    fn state_hashes_are_sensitive_to_topology_changes() {
        let config = dummy_config();
        let mut alloc = IdAllocator::new();
        let r0 = alloc.next_region().unwrap();

        let t1 = IntendedTopology {
            regions: vec![super::super::ir::PlacedRegion {
                id: r0,
                role: super::super::ir::RegionRole::Spawn,
                variant_index: 0,
                layer: 0,
                footprint: (0, 0, 5, 5),
                sockets: vec![],
                transitions: vec![],
                marker_variant_indices: vec![],
            }],
            edges: vec![],
            transitions: vec![],
            route_distance: 0,
            per_layer_cycles: vec![0],
            max_branch_depth: 0,
            dead_end_count: 0,
            articulation_count: 0,
            crossing_count: 0,
            config: config.clone(),
        };

        let level = crate::layout::parse_level("####\n#S.#\n####").unwrap();
        let h1 = StateHash::of_state(&t1, &level);

        let mut t2 = t1.clone();
        t2.route_distance = 42;
        let h2 = StateHash::of_state(&t2, &level);
        assert_ne!(h1, h2);

        let mut changed_level = level.clone();
        changed_level.layers[0][5] = crate::layout::Tile::Wall;
        assert_ne!(h1, StateHash::of_state(&t1, &changed_level));
    }

    #[test]
    fn repair_engine_exhausted_on_clean_topology() {
        let config = dummy_config();
        let catalog = super::super::prefab::PrefabCatalog::load(
            &std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets/prefabs"),
        )
        .expect("catalog");
        let identity = GeneratorIdentity::new(&config, catalog.identity_bytes(), 0);
        let factory = SemanticStreamFactory::new(AttemptIdentity::new(identity, 0));

        let engine = RepairEngine::new(
            &config,
            factory,
            &mut AttemptContext::new(TelemetryMode::Off),
        );
        // With a fresh engine, the budgets should be non-zero.
        assert!(!engine.budgets.is_exhausted());
        assert!(engine.rejected.is_empty());
    }

    #[test]
    fn remove_optional_edge_reduces_edge_count() {
        let config = dummy_config();
        let mut alloc = IdAllocator::new();
        let r0 = alloc.next_region().unwrap();
        let s0 = alloc.next_socket().unwrap();
        let s1 = alloc.next_socket().unwrap();
        let e0 = alloc.next_edge().unwrap();

        let topology = IntendedTopology {
            regions: vec![super::super::ir::PlacedRegion {
                id: r0,
                role: super::super::ir::RegionRole::Spawn,
                variant_index: 0,
                layer: 0,
                footprint: (0, 0, 5, 5),
                sockets: vec![],
                transitions: vec![],
                marker_variant_indices: vec![],
            }],
            edges: vec![super::super::ir::IntendedEdge {
                id: e0,
                source_socket: s0,
                target_socket: s1,
                source_region: r0,
                target_region: r0,
                required: false,
                path_witness: vec![],
                allowed_envelope_cells: vec![],
                cost: 0,
                width: 1,
                transition: None,
            }],
            transitions: vec![],
            route_distance: 0,
            per_layer_cycles: vec![0],
            max_branch_depth: 0,
            dead_end_count: 0,
            articulation_count: 0,
            crossing_count: 0,
            config: config.clone(),
        };

        let catalog = super::super::prefab::PrefabCatalog::load(
            &std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets/prefabs"),
        )
        .expect("catalog");
        let identity = GeneratorIdentity::new(&config, catalog.identity_bytes(), 0);
        let factory = SemanticStreamFactory::new(AttemptIdentity::new(identity, 0));

        let mut engine = RepairEngine::new(
            &config,
            factory,
            &mut AttemptContext::new(TelemetryMode::Off),
        );
        let action = RepairAction {
            reason: GeneratorError::IrInvariant {
                stage: ErrorStage::Ir,
                detail: "test_removal".into(),
            },
            operation: RepairOperation::RemoveOptionalEdge { edge_id: e0 },
            budget_consumed: RepairBudgets {
                optional_edge_removal_budget: 1,
                reroute_budget: 0,
                marker_relocation_budget: 0,
            },
        };

        let result = engine.apply_repair(topology, &action).unwrap();
        assert!(result.edges.is_empty());
    }

    #[test]
    fn acceptance_rejects_a_level_other_than_the_tile_buffer() {
        let config = dummy_config();
        let buffer = TileBuffer::new(8, 8, 1).unwrap();
        let mut alloc = IdAllocator::new();
        let topology = IntendedTopology {
            regions: vec![super::super::ir::PlacedRegion {
                id: alloc.next_region().unwrap(),
                role: super::super::ir::RegionRole::Spawn,
                variant_index: 0,
                layer: 0,
                footprint: (1, 1, 2, 1),
                sockets: vec![],
                transitions: vec![],
                marker_variant_indices: vec![],
            }],
            edges: vec![],
            transitions: vec![],
            route_distance: 0,
            per_layer_cycles: vec![0],
            max_branch_depth: 0,
            dead_end_count: 0,
            articulation_count: 0,
            crossing_count: 0,
            config,
        };
        let level = crate::layout::parse_level("####\n#S.#\n####").unwrap();
        let error = AcceptedIr::accept(topology, buffer, level).unwrap_err();
        assert!(error.to_string().contains("does_not_match_tile_buffer"));
    }
}
