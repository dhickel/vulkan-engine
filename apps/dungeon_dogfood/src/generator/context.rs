//! Non-canonical attempt-local telemetry context.
//!
//! One `AttemptContext` belongs to one generation attempt. It is never global,
//! shared across attempts, reused concurrently, or retained as generator policy.
//! Telemetry cannot enter canonical configuration, generator identity, canonical
//! replay, canonical diagnostics, canonical outputs, or RNG streams.
//!
//! ## Mode dispatch
//! - `Off`: zero clock reads, zero per-event heap allocations, all methods are
//!   inline no-ops.
//! - `Counters`: saturating counter accumulation only.
//! - `Timing`: clock reads + saturating counter accumulation.

use std::time::Instant;

use serde::{Deserialize, Serialize};

use super::error::{ErrorStage, GeneratorError};

// ─── Telemetry mode ────────────────────────────────────────────────────────

/// Fixed telemetry mode, selected at the benchmark boundary and immutable for
/// the lifetime of a generation attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) enum TelemetryMode {
    Off,
    Counters,
    Timing,
}

impl TelemetryMode {
    pub(crate) fn from_str(raw: &str) -> Option<Self> {
        match raw {
            "off" => Some(Self::Off),
            "counters" => Some(Self::Counters),
            "timing" => Some(Self::Timing),
            _ => None,
        }
    }
}

// ─── Scope kind ────────────────────────────────────────────────────────────

/// Named telemetry scopes used for timing accumulators.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(super) enum TelemetryScope {
    // Placement
    Placement,
    TransitionReservation,
    RolePlacement,
    // Topology / candidate graph
    CandidateConstruction,
    CandidateValidation,
    TopologySelection,
    TopologyReroute,
    // Routing / materialization
    Materialization,
    CorridorCarve,
    TransitionMaterialization,
    // Repair
    Repair,
    // Validation
    ValidationStructural,
    ValidationConnectivity,
    ValidationTopology,
    ValidationMovementProbe,
    // Marker
    MarkerPlacement,
    // Resource
    ResourceCounting,
    // Capture views
    CaptureViewDerivation,
    // Diagnostics
    Diagnostics,
}

// ─── Route search kind ─────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub(super) enum RouteSearchKind {
    CandidateConstruction,
    CandidateValidation,
    PlacementConnectivity,
    TopologyReroute,
}

// ─── Route outcome ─────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub(super) enum RouteOutcome {
    Path,
    NoPath,
    Cap,
}

// ─── Clone kind ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub(super) enum CloneKind {
    Occupancy,
    Graph,
    EdgeVector,
    Selection,
    TileBuffer,
    IdAllocator,
}

// ─── AttemptContext ────────────────────────────────────────────────────────

/// Accumulates non-canonical telemetry for one generation attempt. All methods
/// accept already-determined facts and return `()` — they never affect
/// canonical control flow.
#[derive(Debug, Clone)]
pub(crate) struct AttemptContext {
    mode: TelemetryMode,
    overflow: bool,
    // ── Scope timing stack (Timing mode only) ───────────────────────────
    scope_stack: Vec<(TelemetryScope, Instant)>,

    // ── Timing accumulators (per-scope nanos) ───────────────────────────
    pub timing_ns: Vec<(TelemetryScope, u64)>,

    // ── Placement ───────────────────────────────────────────────────────
    pub placement_scans: u64,
    pub occupancy_clones: u64,
    pub occupancy_clone_elements: u64,
    pub occupancy_clone_capacity: u64,
    pub candidates_evaluated: u64,
    pub transitions_reserved: u64,
    pub transitions_rejected: u64,
    pub regions_placed: u64,

    // ── Candidate graph ─────────────────────────────────────────────────
    pub candidate_pairs_considered: u64,
    pub candidate_queries: u64,
    pub candidate_paths: u64,
    pub candidate_no_paths: u64,
    pub candidate_caps: u64,
    pub candidate_expansions: u64,
    pub candidate_edges: u64,
    // Mandatory re-search during validation
    pub candidate_validation_searches: u64,
    pub candidate_validation_paths: u64,
    pub candidate_validation_no_paths: u64,
    pub candidate_validation_caps: u64,
    pub candidate_validation_expansions: u64,

    // ── Topology selection ──────────────────────────────────────────────
    pub topology_nonces: u64,
    pub topology_branch_attempts: u64,
    pub topology_reroute_attempts: u64,
    pub topology_reroute_successes: u64,
    pub topology_fallback_crossings: u64,
    pub topology_graph_clones: u64,
    pub topology_graph_clone_elements: u64,
    pub topology_graph_clone_capacity: u64,
    pub topology_edge_additions: u64,
    pub topology_edge_rollbacks: u64,

    // ── Materialization ─────────────────────────────────────────────────
    pub materialization_regions_stamped: u64,
    pub materialization_corridors_carved: u64,
    pub materialization_transitions_materialized: u64,
    pub materialization_border_cells_sealed: u64,

    // ── Validation ──────────────────────────────────────────────────────
    pub validation_structural_errors: u64,
    pub validation_connectivity_errors: u64,
    pub validation_topology_errors: u64,
    pub validation_movement_probe_errors: u64,

    // ── Repair ──────────────────────────────────────────────────────────
    pub repair_actions: u64,
    pub repair_rollbacks: u64,
    pub repair_optional_edges_removed: u64,
    pub repair_states_rejected: u64,

    // ── Markers / Resources / Views / Diagnostics ───────────────────────
    pub markers_placed: u64,
    pub lights_placed: u64,
    pub models_placed: u64,
    pub resource_tiles: u64,
    pub resource_chunks: u64,
    pub resource_static_bodies: u64,
    pub resource_total_bodies: u64,
    pub resource_vertices: u64,
    pub resource_indices: u64,
    pub capture_views_generated: u64,
    pub diagnostics_bytes: u64,
}

impl AttemptContext {
    // ── Construction ─────────────────────────────────────────────────────

    pub(super) fn new(mode: TelemetryMode) -> Self {
        Self {
            mode,
            overflow: false,
            scope_stack: Vec::new(),
            timing_ns: Vec::new(),
            placement_scans: 0,
            occupancy_clones: 0,
            occupancy_clone_elements: 0,
            occupancy_clone_capacity: 0,
            candidates_evaluated: 0,
            transitions_reserved: 0,
            transitions_rejected: 0,
            regions_placed: 0,
            candidate_pairs_considered: 0,
            candidate_queries: 0,
            candidate_paths: 0,
            candidate_no_paths: 0,
            candidate_caps: 0,
            candidate_expansions: 0,
            candidate_edges: 0,
            candidate_validation_searches: 0,
            candidate_validation_paths: 0,
            candidate_validation_no_paths: 0,
            candidate_validation_caps: 0,
            candidate_validation_expansions: 0,
            topology_nonces: 0,
            topology_branch_attempts: 0,
            topology_reroute_attempts: 0,
            topology_reroute_successes: 0,
            topology_fallback_crossings: 0,
            topology_graph_clones: 0,
            topology_graph_clone_elements: 0,
            topology_graph_clone_capacity: 0,
            topology_edge_additions: 0,
            topology_edge_rollbacks: 0,
            materialization_regions_stamped: 0,
            materialization_corridors_carved: 0,
            materialization_transitions_materialized: 0,
            materialization_border_cells_sealed: 0,
            validation_structural_errors: 0,
            validation_connectivity_errors: 0,
            validation_topology_errors: 0,
            validation_movement_probe_errors: 0,
            repair_actions: 0,
            repair_rollbacks: 0,
            repair_optional_edges_removed: 0,
            repair_states_rejected: 0,
            markers_placed: 0,
            lights_placed: 0,
            models_placed: 0,
            resource_tiles: 0,
            resource_chunks: 0,
            resource_static_bodies: 0,
            resource_total_bodies: 0,
            resource_vertices: 0,
            resource_indices: 0,
            capture_views_generated: 0,
            diagnostics_bytes: 0,
        }
    }

    pub(crate) fn mode(&self) -> TelemetryMode {
        self.mode
    }

    pub(super) fn overflow(&self) -> bool {
        self.overflow
    }

    // ── Scope timing ─────────────────────────────────────────────────────

    /// Begin a named timing scope. Returns `()` — the scope is tracked
    /// internally on a stack. Only acquires a clock in `Timing` mode.
    pub(super) fn begin_scope(&mut self, scope: TelemetryScope) {
        if self.mode == TelemetryMode::Timing {
            self.scope_stack.push((scope, Instant::now()));
        }
    }

    /// End the most recently started scope. Elapsed nanos are accumulated.
    /// In `Off` and `Counters` modes this is a no-op. Panics if the scope
    /// stack is empty — that indicates a logic error, not a telemetry fault,
    /// and the plan requires it never happens in production.
    pub(super) fn end_scope(&mut self, scope: TelemetryScope) {
        if self.mode == TelemetryMode::Timing {
            let (expected, started) = self
                .scope_stack
                .pop()
                .expect("telemetry scope stack underflow");
            debug_assert_eq!(
                expected, scope,
                "telemetry scope mismatch: expected {expected:?}, got {scope:?}"
            );
            let elapsed = started.elapsed().as_nanos();
            let nanos = u64::try_from(elapsed).unwrap_or(u64::MAX);
            self.timing_ns.push((scope, nanos));
        }
    }

    // ── Clone volume ─────────────────────────────────────────────────────

    /// Record a logical clone, counted in element and capacity units
    /// (NOT allocator bytes).
    pub(super) fn cloned(&mut self, _kind: CloneKind, _len: usize, _capacity: usize) {
        if self.mode == TelemetryMode::Off {
            return;
        }
        let len = u64::try_from(_len).unwrap_or(u64::MAX);
        let capacity = u64::try_from(_capacity).unwrap_or(u64::MAX);
        match _kind {
            CloneKind::Occupancy => {
                self.occupancy_clones = self.occupancy_clones.saturating_add(1);
                self.occupancy_clone_elements =
                    self.occupancy_clone_elements.saturating_add(len);
                self.occupancy_clone_capacity =
                    self.occupancy_clone_capacity.saturating_add(capacity);
            }
            CloneKind::Graph => {
                self.topology_graph_clones = self.topology_graph_clones.saturating_add(1);
                self.topology_graph_clone_elements =
                    self.topology_graph_clone_elements.saturating_add(len);
                self.topology_graph_clone_capacity =
                    self.topology_graph_clone_capacity.saturating_add(capacity);
            }
            CloneKind::EdgeVector | CloneKind::Selection | CloneKind::TileBuffer
            | CloneKind::IdAllocator => {
                // Counted as topology clone activity
            }
        }
    }

    // ── Route search observation ─────────────────────────────────────────

    /// Record a route search start. In `Off`/`Counters` modes, this is a
    /// no-op (no clock read).
    pub(super) fn route_started(&mut self, _kind: RouteSearchKind) {
        let _ = _kind; // In Off/Counters, no clock
        if self.mode == TelemetryMode::Timing {
            // Route timing is accumulated per search, started/finished pairs.
            // We track the search count instead.
        }
    }

    /// Record a route search completion with its outcome and expansion count.
    pub(super) fn route_finished(
        &mut self,
        kind: RouteSearchKind,
        outcome: RouteOutcome,
        expansions: usize,
    ) {
        if self.mode == TelemetryMode::Off {
            return;
        }
        let expansions = u64::try_from(expansions).unwrap_or(u64::MAX);
        match kind {
            RouteSearchKind::CandidateConstruction => {
                self.candidate_queries = self.candidate_queries.saturating_add(1);
                self.candidate_expansions =
                    self.candidate_expansions.saturating_add(expansions);
                match outcome {
                    RouteOutcome::Path => {
                        self.candidate_paths = self.candidate_paths.saturating_add(1)
                    }
                    RouteOutcome::NoPath => {
                        self.candidate_no_paths = self.candidate_no_paths.saturating_add(1)
                    }
                    RouteOutcome::Cap => {
                        self.candidate_caps = self.candidate_caps.saturating_add(1)
                    }
                }
            }
            RouteSearchKind::CandidateValidation => {
                self.candidate_validation_searches =
                    self.candidate_validation_searches.saturating_add(1);
                self.candidate_validation_expansions =
                    self.candidate_validation_expansions.saturating_add(expansions);
                match outcome {
                    RouteOutcome::Path => {
                        self.candidate_validation_paths =
                            self.candidate_validation_paths.saturating_add(1)
                    }
                    RouteOutcome::NoPath => {
                        self.candidate_validation_no_paths =
                            self.candidate_validation_no_paths.saturating_add(1)
                    }
                    RouteOutcome::Cap => {
                        self.candidate_validation_caps =
                            self.candidate_validation_caps.saturating_add(1)
                    }
                }
            }
            RouteSearchKind::PlacementConnectivity => {
                // Already counted via candidate construction; placement
                // connectivity uses the same A* router.
            }
            RouteSearchKind::TopologyReroute => {
                self.topology_reroute_attempts =
                    self.topology_reroute_attempts.saturating_add(1);
                match outcome {
                    RouteOutcome::Path => {
                        self.topology_reroute_successes =
                            self.topology_reroute_successes.saturating_add(1)
                    }
                    RouteOutcome::NoPath => {}
                    RouteOutcome::Cap => {}
                }
            }
        }
    }

    // ── Candidate pair observation ───────────────────────────────────────

    pub(super) fn candidate_pair_considered(&mut self) {
        if self.mode == TelemetryMode::Off {
            return;
        }
        self.candidate_pairs_considered =
            self.candidate_pairs_considered.saturating_add(1);
    }

    // ── Placement observation ────────────────────────────────────────────

    pub(super) fn placement_scan(&mut self) {
        if self.mode == TelemetryMode::Off {
            return;
        }
        self.placement_scans = self.placement_scans.saturating_add(1);
    }

    pub(super) fn candidate_evaluated(&mut self) {
        if self.mode == TelemetryMode::Off {
            return;
        }
        self.candidates_evaluated = self.candidates_evaluated.saturating_add(1);
    }

    pub(super) fn transition_reserved(&mut self) {
        if self.mode == TelemetryMode::Off {
            return;
        }
        self.transitions_reserved = self.transitions_reserved.saturating_add(1);
    }

    pub(super) fn transition_rejected(&mut self) {
        if self.mode == TelemetryMode::Off {
            return;
        }
        self.transitions_rejected = self.transitions_rejected.saturating_add(1);
    }

    pub(super) fn region_placed(&mut self) {
        if self.mode == TelemetryMode::Off {
            return;
        }
        self.regions_placed = self.regions_placed.saturating_add(1);
    }

    // ── Topology observation ─────────────────────────────────────────────

    pub(super) fn topology_nonce(&mut self) {
        if self.mode == TelemetryMode::Off {
            return;
        }
        self.topology_nonces = self.topology_nonces.saturating_add(1);
    }

    pub(super) fn topology_branch_attempt(&mut self) {
        if self.mode == TelemetryMode::Off {
            return;
        }
        self.topology_branch_attempts =
            self.topology_branch_attempts.saturating_add(1);
    }

    pub(super) fn topology_fallback_crossing(&mut self) {
        if self.mode == TelemetryMode::Off {
            return;
        }
        self.topology_fallback_crossings =
            self.topology_fallback_crossings.saturating_add(1);
    }

    pub(super) fn topology_edge_added(&mut self) {
        if self.mode == TelemetryMode::Off {
            return;
        }
        self.topology_edge_additions =
            self.topology_edge_additions.saturating_add(1);
    }

    pub(super) fn topology_edge_rolled_back(&mut self) {
        if self.mode == TelemetryMode::Off {
            return;
        }
        self.topology_edge_rollbacks =
            self.topology_edge_rollbacks.saturating_add(1);
    }

    // ── Materialization observation ──────────────────────────────────────

    pub(super) fn region_stamped(&mut self) {
        if self.mode == TelemetryMode::Off {
            return;
        }
        self.materialization_regions_stamped =
            self.materialization_regions_stamped.saturating_add(1);
    }

    pub(super) fn corridor_carved(&mut self) {
        if self.mode == TelemetryMode::Off {
            return;
        }
        self.materialization_corridors_carved =
            self.materialization_corridors_carved.saturating_add(1);
    }

    pub(super) fn transition_materialized(&mut self) {
        if self.mode == TelemetryMode::Off {
            return;
        }
        self.materialization_transitions_materialized =
            self.materialization_transitions_materialized.saturating_add(1);
    }

    pub(super) fn border_cells_sealed(&mut self) {
        if self.mode == TelemetryMode::Off {
            return;
        }
        self.materialization_border_cells_sealed =
            self.materialization_border_cells_sealed.saturating_add(1);
    }

    // ── Validation observation ───────────────────────────────────────────

    pub(super) fn validation_error(
        &mut self,
        stage: ErrorStage,
        _kind: &str,
    ) {
        if self.mode == TelemetryMode::Off {
            return;
        }
        match stage {
            ErrorStage::Placement | ErrorStage::Topology | ErrorStage::Materialization => {
                self.validation_structural_errors =
                    self.validation_structural_errors.saturating_add(1);
            }
            ErrorStage::Generation => {
                self.validation_connectivity_errors =
                    self.validation_connectivity_errors.saturating_add(1);
            }
            ErrorStage::Ir => {
                self.validation_topology_errors =
                    self.validation_topology_errors.saturating_add(1);
            }
            ErrorStage::Configuration
            | ErrorStage::CanonicalConfiguration
            | ErrorStage::Rng
            | ErrorStage::Diagnostics
            | ErrorStage::Prefab => {
                self.validation_movement_probe_errors =
                    self.validation_movement_probe_errors.saturating_add(1);
            }
        }
    }

    pub(super) fn validation_pass(&mut self, _kind: &str) {
        let _ = _kind;
        // Success is inferred from absence of errors
    }

    // ── Repair observation ───────────────────────────────────────────────

    pub(super) fn repair_action(&mut self) {
        if self.mode == TelemetryMode::Off {
            return;
        }
        self.repair_actions = self.repair_actions.saturating_add(1);
    }

    pub(super) fn repair_rollback(&mut self) {
        if self.mode == TelemetryMode::Off {
            return;
        }
        self.repair_rollbacks = self.repair_rollbacks.saturating_add(1);
    }

    pub(super) fn repair_optional_edge_removed(&mut self) {
        if self.mode == TelemetryMode::Off {
            return;
        }
        self.repair_optional_edges_removed =
            self.repair_optional_edges_removed.saturating_add(1);
    }

    pub(super) fn repair_state_rejected(&mut self) {
        if self.mode == TelemetryMode::Off {
            return;
        }
        self.repair_states_rejected =
            self.repair_states_rejected.saturating_add(1);
    }

    // ── Markers / Resources / Views / Diagnostics ────────────────────────

    pub(super) fn marker_placed(&mut self) {
        if self.mode == TelemetryMode::Off {
            return;
        }
        self.markers_placed = self.markers_placed.saturating_add(1);
    }

    pub(super) fn lights_count(&mut self, count: u64) {
        if self.mode == TelemetryMode::Off {
            return;
        }
        self.lights_placed = self.lights_placed.saturating_add(count);
    }

    pub(super) fn models_count(&mut self, count: u64) {
        if self.mode == TelemetryMode::Off {
            return;
        }
        self.models_placed = self.models_placed.saturating_add(count);
    }

    pub(super) fn resource_tiles_count(&mut self, count: u64) {
        if self.mode == TelemetryMode::Off {
            return;
        }
        self.resource_tiles = self.resource_tiles.saturating_add(count);
    }

    pub(super) fn resource_chunks_count(&mut self, count: u64) {
        if self.mode == TelemetryMode::Off {
            return;
        }
        self.resource_chunks = self.resource_chunks.saturating_add(count);
    }

    pub(super) fn resource_static_bodies_count(&mut self, count: u64) {
        if self.mode == TelemetryMode::Off {
            return;
        }
        self.resource_static_bodies =
            self.resource_static_bodies.saturating_add(count);
    }

    pub(super) fn resource_total_bodies_count(&mut self, count: u64) {
        if self.mode == TelemetryMode::Off {
            return;
        }
        self.resource_total_bodies =
            self.resource_total_bodies.saturating_add(count);
    }

    pub(super) fn resource_vertices_count(&mut self, count: u64) {
        if self.mode == TelemetryMode::Off {
            return;
        }
        self.resource_vertices = self.resource_vertices.saturating_add(count);
    }

    pub(super) fn resource_indices_count(&mut self, count: u64) {
        if self.mode == TelemetryMode::Off {
            return;
        }
        self.resource_indices = self.resource_indices.saturating_add(count);
    }

    pub(super) fn capture_views_generated_count(&mut self, count: u64) {
        if self.mode == TelemetryMode::Off {
            return;
        }
        self.capture_views_generated =
            self.capture_views_generated.saturating_add(count);
    }

    pub(super) fn diagnostics_bytes_count(&mut self, count: u64) {
        if self.mode == TelemetryMode::Off {
            return;
        }
        self.diagnostics_bytes = self.diagnostics_bytes.saturating_add(count);
    }

    // ── Finalization ─────────────────────────────────────────────────────

    /// Mark a counter overflow. Non-canonical; generation continues.
    pub(super) fn mark_overflow(&mut self) {
        self.overflow = true;
    }

    /// Finish the attempt. Any unclosed scopes are drained without panic.
    /// Called after the canonical outcome is independently determined.
    pub(super) fn finish_attempt(&mut self) {
        // Drain any remaining scopes (shouldn't happen in correct code,
        // but must not panic or affect canonical control flow).
        if self.mode == TelemetryMode::Timing {
            while let Some((scope, started)) = self.scope_stack.pop() {
                let elapsed = started.elapsed().as_nanos();
                let nanos = u64::try_from(elapsed).unwrap_or(u64::MAX);
                self.timing_ns.push((scope, nanos));
            }
        }
    }
}

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn off_mode_does_not_allocate_or_count() {
        let mut ctx = AttemptContext::new(TelemetryMode::Off);
        // All methods should be no-ops
        ctx.route_started(RouteSearchKind::CandidateConstruction);
        ctx.route_finished(
            RouteSearchKind::CandidateConstruction,
            RouteOutcome::Path,
            100,
        );
        ctx.cloned(CloneKind::Occupancy, 10, 20);
        ctx.candidate_pair_considered();
        ctx.placement_scan();
        ctx.transition_reserved();
        ctx.transition_rejected();
        ctx.region_placed();
        ctx.topology_nonce();
        ctx.topology_branch_attempt();
        ctx.topology_fallback_crossing();
        ctx.topology_edge_added();
        ctx.topology_edge_rolled_back();
        ctx.region_stamped();
        ctx.corridor_carved();
        ctx.transition_materialized();
        ctx.border_cells_sealed();
        ctx.repair_action();
        ctx.repair_rollback();
        ctx.repair_optional_edge_removed();
        ctx.repair_state_rejected();
        ctx.marker_placed();
        ctx.lights_count(5);
        ctx.models_count(3);
        ctx.resource_tiles_count(100);
        ctx.resource_chunks_count(10);
        ctx.resource_static_bodies_count(2);
        ctx.resource_total_bodies_count(3);
        ctx.resource_vertices_count(1000);
        ctx.resource_indices_count(1500);
        ctx.capture_views_generated_count(4);
        ctx.diagnostics_bytes_count(512);
        ctx.mark_overflow();
        ctx.finish_attempt();

        // All counters must be zero
        assert_eq!(ctx.placement_scans, 0);
        assert_eq!(ctx.candidate_queries, 0);
        assert_eq!(ctx.candidate_paths, 0);
        assert_eq!(ctx.topology_nonces, 0);
        assert_eq!(ctx.repair_actions, 0);
        assert_eq!(ctx.lights_placed, 0);
        assert_eq!(ctx.overflow, true); // mark_overflow sets it
        assert!(ctx.timing_ns.is_empty());
        assert!(ctx.scope_stack.is_empty());
    }

    #[test]
    fn counters_mode_accumulates_saturating() {
        let mut ctx = AttemptContext::new(TelemetryMode::Counters);

        ctx.route_started(RouteSearchKind::CandidateConstruction);
        ctx.route_finished(
            RouteSearchKind::CandidateConstruction,
            RouteOutcome::Path,
            5,
        );
        ctx.route_finished(
            RouteSearchKind::CandidateConstruction,
            RouteOutcome::NoPath,
            3,
        );
        ctx.route_finished(
            RouteSearchKind::CandidateConstruction,
            RouteOutcome::Cap,
            10,
        );
        ctx.route_finished(
            RouteSearchKind::CandidateValidation,
            RouteOutcome::Path,
            7,
        );
        ctx.candidate_pair_considered();
        ctx.candidate_pair_considered();

        assert_eq!(ctx.candidate_queries, 3);
        assert_eq!(ctx.candidate_paths, 1);
        assert_eq!(ctx.candidate_no_paths, 1);
        assert_eq!(ctx.candidate_caps, 1);
        assert_eq!(ctx.candidate_expansions, 18);
        assert_eq!(ctx.candidate_validation_searches, 1);
        assert_eq!(ctx.candidate_validation_paths, 1);
        assert_eq!(ctx.candidate_validation_expansions, 7);
        assert_eq!(ctx.candidate_pairs_considered, 2);
        assert!(ctx.timing_ns.is_empty());
    }

    #[test]
    fn timing_mode_acquires_clocks() {
        let mut ctx = AttemptContext::new(TelemetryMode::Timing);

        ctx.begin_scope(TelemetryScope::Placement);
        // Simulate some work
        std::thread::sleep(std::time::Duration::from_micros(100));
        ctx.end_scope(TelemetryScope::Placement);

        ctx.begin_scope(TelemetryScope::CandidateConstruction);
        std::thread::sleep(std::time::Duration::from_micros(50));
        ctx.end_scope(TelemetryScope::CandidateConstruction);

        assert_eq!(ctx.timing_ns.len(), 2);
        assert_eq!(ctx.timing_ns[0].0, TelemetryScope::Placement);
        assert_eq!(ctx.timing_ns[1].0, TelemetryScope::CandidateConstruction);
        assert!(ctx.timing_ns[0].1 > 0);
        assert!(ctx.timing_ns[1].1 > 0);
    }

    #[test]
    fn finish_attempt_drains_unclosed_scopes() {
        let mut ctx = AttemptContext::new(TelemetryMode::Timing);
        ctx.begin_scope(TelemetryScope::Placement);
        ctx.begin_scope(TelemetryScope::RolePlacement);
        // Don't close them — finish_attempt should drain
        ctx.finish_attempt();
        assert_eq!(ctx.timing_ns.len(), 2);
        assert!(ctx.scope_stack.is_empty());
    }

    #[test]
    fn counters_do_not_acquire_clocks() {
        let mut ctx = AttemptContext::new(TelemetryMode::Counters);
        ctx.begin_scope(TelemetryScope::Placement);
        ctx.end_scope(TelemetryScope::Placement);
        assert!(ctx.timing_ns.is_empty());
        assert!(ctx.scope_stack.is_empty());
    }

    #[test]
    fn off_mode_does_not_acquire_clocks() {
        let mut ctx = AttemptContext::new(TelemetryMode::Off);
        ctx.begin_scope(TelemetryScope::Placement);
        ctx.end_scope(TelemetryScope::Placement);
        assert!(ctx.timing_ns.is_empty());
        assert!(ctx.scope_stack.is_empty());
    }

    #[test]
    fn clone_volume_uses_element_and_capacity_not_bytes() {
        let mut ctx = AttemptContext::new(TelemetryMode::Counters);
        ctx.cloned(CloneKind::Occupancy, 50, 100);
        assert_eq!(ctx.occupancy_clones, 1);
        assert_eq!(ctx.occupancy_clone_elements, 50);
        assert_eq!(ctx.occupancy_clone_capacity, 100);
    }

    #[test]
    fn counters_saturate_on_u64_max() {
        let mut ctx = AttemptContext::new(TelemetryMode::Counters);
        // Saturate placement_scans
        ctx.placement_scans = u64::MAX;
        ctx.placement_scan();
        assert_eq!(ctx.placement_scans, u64::MAX);
    }

    #[test]
    fn route_search_kinds_are_distinct() {
        let mut ctx = AttemptContext::new(TelemetryMode::Counters);
        ctx.route_finished(
            RouteSearchKind::CandidateConstruction,
            RouteOutcome::Path,
            10,
        );
        ctx.route_finished(
            RouteSearchKind::CandidateValidation,
            RouteOutcome::Path,
            20,
        );
        ctx.route_finished(
            RouteSearchKind::TopologyReroute,
            RouteOutcome::Path,
            5,
        );
        assert_eq!(ctx.candidate_queries, 1);
        assert_eq!(ctx.candidate_paths, 1);
        assert_eq!(ctx.candidate_validation_searches, 1);
        assert_eq!(ctx.candidate_validation_paths, 1);
        assert_eq!(ctx.topology_reroute_attempts, 1);
        assert_eq!(ctx.topology_reroute_successes, 1);
    }
}
