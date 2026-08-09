//! Immutable Richness V1 request metadata.
//!
//! This is the request-bound metadata contract available before geometry,
//! assets, compiler evidence, or publication exist. It records exact canonical
//! authored and resolved request bytes plus their independently framed
//! identities; later phases add their own immutable result records rather than
//! mutating this provenance snapshot.

use super::request::ResolvedRichnessRequestV1;

/// Frozen schema tag for request provenance metadata.
pub const RICHNESS_METADATA_SCHEMA_VERSION: &str = "richness-v1";

/// Immutable metadata snapshot for one validated Richness request.
///
/// The canonical byte vectors retain inherited-versus-explicit provenance and
/// the resolved controls, while the hashes give downstream artifacts stable
/// compact references without involving baseline V3 metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RichnessMetadataV1 {
    schema_version: &'static str,
    request_identity: [u8; 32],
    resolved_request_identity: [u8; 32],
    canonical_request: Vec<u8>,
    canonical_resolved_request: Vec<u8>,
}

impl RichnessMetadataV1 {
    /// Capture immutable provenance from an already validated request.
    pub fn from_resolved(request: &ResolvedRichnessRequestV1) -> Self {
        Self {
            schema_version: RICHNESS_METADATA_SCHEMA_VERSION,
            request_identity: request.provenance().identity_hash(),
            resolved_request_identity: request.identity_hash(),
            canonical_request: request.provenance().to_canonical_bytes(),
            canonical_resolved_request: request.to_canonical_bytes(),
        }
    }

    /// Frozen schema tag for this provenance record.
    pub fn schema_version(&self) -> &'static str {
        self.schema_version
    }

    /// Identity of the authored request, including explicit-state markers.
    pub fn request_identity(&self) -> [u8; 32] {
        self.request_identity
    }

    /// Identity of the resolved request, including resolved values and sources.
    pub fn resolved_request_identity(&self) -> [u8; 32] {
        self.resolved_request_identity
    }

    /// Exact canonical authored request bytes.
    pub fn canonical_request(&self) -> &[u8] {
        &self.canonical_request
    }

    /// Exact canonical resolved request bytes.
    pub fn canonical_resolved_request(&self) -> &[u8] {
        &self.canonical_resolved_request
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::enhanced_v3::richness::request::{
        ResolvedRichnessRequestV1, RichnessDocumentV1, RichnessPreset, RichnessTheme,
    };

    #[test]
    fn metadata_is_an_immutable_request_provenance_snapshot() {
        let request = ResolvedRichnessRequestV1::resolve(
            RichnessDocumentV1::new(42, 2048, RichnessPreset::Sparse, RichnessTheme::Ancient)
                .unwrap(),
        )
        .unwrap();
        let metadata = RichnessMetadataV1::from_resolved(&request);

        assert_eq!(metadata.schema_version(), RICHNESS_METADATA_SCHEMA_VERSION);
        assert_eq!(
            metadata.request_identity(),
            request.provenance().identity_hash()
        );
        assert_eq!(
            metadata.resolved_request_identity(),
            request.identity_hash()
        );
        assert_eq!(
            metadata.canonical_request(),
            request.provenance().to_canonical_bytes()
        );
        assert_eq!(
            metadata.canonical_resolved_request(),
            request.to_canonical_bytes()
        );
    }
}

// ── Generation metadata ────────────────────────────────────────────────────

use std::fmt::Write as _;

/// Deterministic generation metadata: every semantic, topological, vertical,
/// cave, theme, presentation, budget, support, and visibility fact in fixed
/// canonical order, with reproduction revisions and request provenance.
///
/// Excludes timestamps, host paths, unordered diagnostics, and random draw
/// logs by construction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RichnessGenerationMetadata {
    /// Fixed-order canonical fact lines.
    facts: Vec<String>,
    /// Semantic identity: blueprint + macro reservations (theme-independent).
    semantic_identity: Vec<u8>,
    /// Macro reservation fingerprint (theme-independent).
    reservation_fingerprint: Vec<u8>,
}

impl RichnessGenerationMetadata {
    pub(crate) fn build(
        resolved: &ResolvedRichnessRequestV1,
        blueprint: &super::ids::PacingBlueprint,
        topology: &super::topology::TopologyResult,
        composition: &super::composition::StructuralComposition,
        actual: crate::enhanced_v3::richness::pipeline::ActualCounts,
    ) -> Self {
        let mut facts: Vec<String> = Vec::new();
        let mut line = |key: &str, value: String| facts.push(format!("{key}: {value}"));

        // Revisions + provenance (frozen order).
        line(
            "schema",
            crate::enhanced_v3::richness::generated_content::SCHEMA_VERSION.to_string(),
        );
        line("seed", resolved.seed().to_string());
        line("extent", resolved.extent().to_string());
        line("preset", resolved.preset().tag().to_string());
        line("theme", resolved.theme().tag().to_string());
        line("cave_mode", resolved.cave_mode().value.tag().to_string());
        line(
            "request_identity",
            hex(resolved.provenance().identity_hash()),
        );
        line("resolved_identity", hex(resolved.identity_hash()));

        // Blueprint facts.
        line("beats", blueprint.beat_order.len().to_string());
        line("landmarks", blueprint.forced_landmarks.len().to_string());
        line("zones", blueprint.zone_blueprint.zones.len().to_string());
        line("branches", blueprint.branch_payoffs.len().to_string());
        line("shortcuts", blueprint.shortcut_intents.len().to_string());
        line(
            "mandatory_edges",
            blueprint.mandatory_edges.len().to_string(),
        );

        // Topology facts.
        line("rooms", topology.journal.reservations.len().to_string());
        line("routes", topology.routes.len().to_string());
        line(
            "vertical_routes",
            topology.vertical_routes.len().to_string(),
        );
        let loop_count = topology.routes.len().saturating_sub(
            topology
                .beat_to_reservations
                .len()
                .min(topology.routes.len()),
        );
        line("loops_estimate", loop_count.to_string());

        // Composition facts.
        line("brushes", actual.brushes.to_string());
        line("faces", actual.faces.to_string());
        line("entities", actual.entities.to_string());
        line("lights", actual.lights.to_string());
        line("support_contacts", actual.support_contacts.to_string());
        line("openings", actual.openings.to_string());
        line(
            "props",
            composition
                .presentation
                .rooms
                .values()
                .map(|room| room.props.len())
                .sum::<usize>()
                .to_string(),
        );
        line(
            "skipped_props",
            composition
                .presentation
                .skipped_room_props
                .values()
                .map(Vec::len)
                .sum::<usize>()
                .to_string(),
        );
        line(
            "skipped_lights",
            composition
                .presentation
                .skipped_room_lights
                .values()
                .map(Vec::len)
                .sum::<usize>()
                .to_string(),
        );
        line(
            "quiet_rooms",
            composition.presentation.quiet_rooms.len().to_string(),
        );
        line(
            "dense_rooms",
            composition.presentation.dense_rooms.len().to_string(),
        );
        line(
            "broken_rooms",
            composition.presentation.broken_rooms.len().to_string(),
        );
        line(
            "cave",
            match &composition.cave {
                Some(cave) => format!(
                    "selected:cells={} boxes={} key={}",
                    cave.empty_cells.len(),
                    cave.solid_boxes.len(),
                    cave.candidate_key
                ),
                None => "none".to_string(),
            },
        );
        line(
            "visibility_merges",
            composition.visibility.semantic_leaves.len().to_string(),
        );

        // Semantic identity: blueprint bytes + macro reservation bounds in
        // fixed order (theme never participates).
        let mut semantic = Vec::new();
        semantic.extend_from_slice(&blueprint.beat_order.len().to_le_bytes());
        semantic.extend_from_slice(&blueprint.forced_landmarks.len().to_le_bytes());
        let mut reservations: Vec<_> = topology.journal.reservations.values().collect();
        reservations.sort_by_key(|record| record.id);
        for record in reservations {
            if !record.committed {
                continue;
            }
            let bounds = super::geometry::footprint_quake_bounds(&record.footprint);
            for value in [bounds.0, bounds.1, bounds.2, bounds.3] {
                semantic.extend_from_slice(&value.to_le_bytes());
            }
        }
        let reservation_fingerprint = semantic.clone();

        Self {
            facts,
            semantic_identity: semantic,
            reservation_fingerprint,
        }
    }

    /// Canonical fixed-order fact bytes (deterministic reproduction record).
    pub fn to_canonical_bytes(&self) -> Vec<u8> {
        let mut out = String::new();
        for fact in &self.facts {
            let _ = writeln!(out, "{fact}");
        }
        out.into_bytes()
    }

    /// Theme-independent semantic identity (blueprint + macro reservations).
    pub fn semantic_identity(&self) -> &[u8] {
        &self.semantic_identity
    }

    /// Macro reservation fingerprint (theme-independent).
    pub fn reservation_fingerprint(&self) -> &[u8] {
        &self.reservation_fingerprint
    }
}

fn hex(bytes: [u8; 32]) -> String {
    let mut out = String::with_capacity(64);
    for byte in bytes {
        let _ = write!(out, "{byte:02x}");
    }
    out
}
