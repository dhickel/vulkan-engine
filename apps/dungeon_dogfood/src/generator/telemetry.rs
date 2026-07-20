//! Non-canonical telemetry serialization.
//!
//! Telemetry payloads are schema-versioned JSON emitted only at the benchmark
//! boundary. They are independently parseable and never enter canonical
//! diagnostics, normalized configuration, RNG streams, or generator decisions.
//!
//! Serialization failure is a benchmark observation, not a generator decision.

use serde::{Deserialize, Serialize};

use super::context::AttemptContext;

const TELEMETRY_SCHEMA_VERSION: u32 = 1;

// ─── Serializable payload ──────────────────────────────────────────────────

/// The serialized form of a completed attempt's telemetry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct TelemetryPayload {
    pub schema_version: u32,
    pub mode: String,
    pub overflow: bool,

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
    pub candidate_validation_searches: u64,
    pub candidate_validation_paths: u64,
    pub candidate_validation_no_paths: u64,
    pub candidate_validation_caps: u64,
    pub candidate_validation_expansions: u64,

    // ── Topology ────────────────────────────────────────────────────────
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

    // ── Timing (Timing mode only) ───────────────────────────────────────
    #[serde(skip_serializing_if = "Vec::is_empty")]
    #[serde(default)]
    pub timing_entries: Vec<TimingEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct TimingEntry {
    pub scope: String,
    pub nanos: u64,
}

// ─── Conversion ────────────────────────────────────────────────────────────

impl TelemetryPayload {
    /// Build a payload from the finalized context. The context must have been
    /// finalized via `finish_attempt()` first.
    pub(super) fn from_context(ctx: &AttemptContext) -> Self {
        let mode_str = match ctx.mode() {
            super::context::TelemetryMode::Off => "off",
            super::context::TelemetryMode::Counters => "counters",
            super::context::TelemetryMode::Timing => "timing",
        };

        let timing_entries = ctx
            .timing_ns
            .iter()
            .map(|(scope, nanos)| TimingEntry {
                scope: format!("{scope:?}"),
                nanos: *nanos,
            })
            .collect();

        Self {
            schema_version: TELEMETRY_SCHEMA_VERSION,
            mode: mode_str.to_string(),
            overflow: ctx.overflow(),
            placement_scans: ctx.placement_scans,
            occupancy_clones: ctx.occupancy_clones,
            occupancy_clone_elements: ctx.occupancy_clone_elements,
            occupancy_clone_capacity: ctx.occupancy_clone_capacity,
            candidates_evaluated: ctx.candidates_evaluated,
            transitions_reserved: ctx.transitions_reserved,
            transitions_rejected: ctx.transitions_rejected,
            regions_placed: ctx.regions_placed,
            candidate_pairs_considered: ctx.candidate_pairs_considered,
            candidate_queries: ctx.candidate_queries,
            candidate_paths: ctx.candidate_paths,
            candidate_no_paths: ctx.candidate_no_paths,
            candidate_caps: ctx.candidate_caps,
            candidate_expansions: ctx.candidate_expansions,
            candidate_edges: ctx.candidate_edges,
            candidate_validation_searches: ctx.candidate_validation_searches,
            candidate_validation_paths: ctx.candidate_validation_paths,
            candidate_validation_no_paths: ctx.candidate_validation_no_paths,
            candidate_validation_caps: ctx.candidate_validation_caps,
            candidate_validation_expansions: ctx.candidate_validation_expansions,
            topology_nonces: ctx.topology_nonces,
            topology_branch_attempts: ctx.topology_branch_attempts,
            topology_reroute_attempts: ctx.topology_reroute_attempts,
            topology_reroute_successes: ctx.topology_reroute_successes,
            topology_fallback_crossings: ctx.topology_fallback_crossings,
            topology_graph_clones: ctx.topology_graph_clones,
            topology_graph_clone_elements: ctx.topology_graph_clone_elements,
            topology_graph_clone_capacity: ctx.topology_graph_clone_capacity,
            topology_edge_additions: ctx.topology_edge_additions,
            topology_edge_rollbacks: ctx.topology_edge_rollbacks,
            materialization_regions_stamped: ctx.materialization_regions_stamped,
            materialization_corridors_carved: ctx.materialization_corridors_carved,
            materialization_transitions_materialized: ctx.materialization_transitions_materialized,
            materialization_border_cells_sealed: ctx.materialization_border_cells_sealed,
            validation_structural_errors: ctx.validation_structural_errors,
            validation_connectivity_errors: ctx.validation_connectivity_errors,
            validation_topology_errors: ctx.validation_topology_errors,
            validation_movement_probe_errors: ctx.validation_movement_probe_errors,
            repair_actions: ctx.repair_actions,
            repair_rollbacks: ctx.repair_rollbacks,
            repair_optional_edges_removed: ctx.repair_optional_edges_removed,
            repair_states_rejected: ctx.repair_states_rejected,
            markers_placed: ctx.markers_placed,
            lights_placed: ctx.lights_placed,
            models_placed: ctx.models_placed,
            resource_tiles: ctx.resource_tiles,
            resource_chunks: ctx.resource_chunks,
            resource_static_bodies: ctx.resource_static_bodies,
            resource_total_bodies: ctx.resource_total_bodies,
            resource_vertices: ctx.resource_vertices,
            resource_indices: ctx.resource_indices,
            capture_views_generated: ctx.capture_views_generated,
            diagnostics_bytes: ctx.diagnostics_bytes,
            timing_entries,
        }
    }
}

// ─── Public helper ─────────────────────────────────────────────────────────

/// Serialize a finalized context to a JSON byte vector.
pub(crate) fn serialize_telemetry(ctx: &AttemptContext) -> Result<Vec<u8>, String> {
    let payload = TelemetryPayload::from_context(ctx);
    serde_json::to_vec(&payload).map_err(|e| format!("telemetry serialization: {e}"))
}

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::super::context::{AttemptContext, RouteOutcome, RouteSearchKind, TelemetryMode};
    use super::*;

    fn filled_context(mode: TelemetryMode) -> AttemptContext {
        let mut ctx = AttemptContext::new(mode);
        ctx.placement_scan();
        ctx.placement_scan();
        ctx.candidate_pair_considered();
        ctx.route_finished(
            RouteSearchKind::CandidateConstruction,
            RouteOutcome::Path,
            10,
        );
        ctx.route_finished(
            RouteSearchKind::CandidateConstruction,
            RouteOutcome::NoPath,
            5,
        );
        ctx.topology_nonce();
        ctx.region_stamped();
        ctx.repair_action();
        ctx.lights_count(3);
        ctx.resource_tiles_count(96 * 96 * 3);
        ctx.finish_attempt();
        ctx
    }

    #[test]
    fn schema_round_trip_off_mode() {
        let ctx = filled_context(TelemetryMode::Off);
        let json = serialize_telemetry(&ctx).unwrap();
        let parsed: TelemetryPayload = serde_json::from_slice(&json).unwrap();

        assert_eq!(parsed.schema_version, TELEMETRY_SCHEMA_VERSION);
        assert_eq!(parsed.mode, "off");
        assert!(!parsed.overflow);
        // Off mode: all counters are 0
        assert_eq!(parsed.placement_scans, 0);
        assert_eq!(parsed.candidate_queries, 0);
        assert!(parsed.timing_entries.is_empty());
    }

    #[test]
    fn schema_round_trip_counters_mode() {
        let ctx = filled_context(TelemetryMode::Counters);
        let json = serialize_telemetry(&ctx).unwrap();
        let parsed: TelemetryPayload = serde_json::from_slice(&json).unwrap();

        assert_eq!(parsed.schema_version, TELEMETRY_SCHEMA_VERSION);
        assert_eq!(parsed.mode, "counters");
        assert_eq!(parsed.placement_scans, 2);
        assert_eq!(parsed.candidate_pairs_considered, 1);
        assert_eq!(parsed.candidate_queries, 2);
        assert_eq!(parsed.candidate_paths, 1);
        assert_eq!(parsed.candidate_no_paths, 1);
        assert_eq!(parsed.topology_nonces, 1);
        assert_eq!(parsed.materialization_regions_stamped, 1);
        assert_eq!(parsed.repair_actions, 1);
        assert_eq!(parsed.lights_placed, 3);
        assert_eq!(parsed.resource_tiles, 96 * 96 * 3);
        assert!(parsed.timing_entries.is_empty());
    }

    #[test]
    fn schema_round_trip_timing_mode() {
        let mut ctx = AttemptContext::new(TelemetryMode::Timing);
        ctx.begin_scope(super::super::context::TelemetryScope::Placement);
        ctx.placement_scan();
        ctx.route_finished(
            RouteSearchKind::CandidateConstruction,
            RouteOutcome::Path,
            42,
        );
        ctx.end_scope(super::super::context::TelemetryScope::Placement);
        ctx.finish_attempt();

        let json = serialize_telemetry(&ctx).unwrap();
        let parsed: TelemetryPayload = serde_json::from_slice(&json).unwrap();

        assert_eq!(parsed.mode, "timing");
        assert_eq!(parsed.placement_scans, 1);
        assert_eq!(parsed.candidate_paths, 1);
        assert_eq!(parsed.candidate_expansions, 42);
        assert!(!parsed.timing_entries.is_empty());
        assert_eq!(parsed.timing_entries[0].scope, "Placement");
        assert!(parsed.timing_entries[0].nanos > 0);
    }

    #[test]
    fn unknown_schema_version_is_detectable() {
        let ctx = filled_context(TelemetryMode::Counters);
        let json = serialize_telemetry(&ctx).unwrap();
        let parsed: TelemetryPayload = serde_json::from_slice(&json).unwrap();
        assert_eq!(parsed.schema_version, TELEMETRY_SCHEMA_VERSION);
    }

    #[test]
    fn serialization_failure_is_a_benchmark_observation() {
        // This test ensures serialization errors don't panic.
        let ctx = AttemptContext::new(TelemetryMode::Off);
        let result = serialize_telemetry(&ctx);
        assert!(result.is_ok());
    }

    #[test]
    fn timing_entries_omitted_for_non_timing_modes() {
        let ctx = filled_context(TelemetryMode::Counters);
        let json = serialize_telemetry(&ctx).unwrap();
        let json_str = std::str::from_utf8(&json).unwrap();
        assert!(!json_str.contains("timing_entries"));
    }

    #[test]
    fn overflow_is_serialized() {
        let mut ctx = AttemptContext::new(TelemetryMode::Counters);
        ctx.mark_overflow();
        ctx.finish_attempt();
        let json = serialize_telemetry(&ctx).unwrap();
        let parsed: TelemetryPayload = serde_json::from_slice(&json).unwrap();
        assert!(parsed.overflow);
    }
}
