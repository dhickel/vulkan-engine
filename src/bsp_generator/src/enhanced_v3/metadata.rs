//! Schema-v3 production metadata for Enhanced V3 generation.
//!
//! `EnhancedV3Metadata` captures deterministic semantic facts about a
//! completed generation run. All fields are private with read-only
//! accessors. The metadata is serialized in a fixed field order and
//! deliberately excludes random draws, candidate internals, compiler
//! diagnostics, timestamps, paths, and platform-specific provenance.
//!
//! # Contract
//!
//! - Private fields only; read-only accessors expose semantic facts.
//! - Serialized field order is frozen and deterministic.
//! - No root-level export — accessed through the pipeline return value.
//! - Excluded data: random draws, candidate enumeration, compiler diagnostics,
//!   timestamps, host paths, executable paths, platform-specific provenance.

use super::config::V3Config;
use super::ids::CommittedTopology;

/// Canonical production metadata for a completed Enhanced V3 generation.
///
/// Contains only approved deterministic semantic data. Compiler, renderer,
/// package, and runtime evidence belongs in separate evidence artifacts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnhancedV3Metadata {
    // ── Configuration identity ─────────────────────────────────────────
    /// The master seed used for this run.
    seed: u64,
    /// The density preset name.
    preset: String,
    /// The requested XY extent in Quake units.
    xy_extent: u32,

    // ── Output identities ──────────────────────────────────────────────
    /// Schema version identifier (frozen at "v3").
    schema_version: String,
    /// Generator identity string.
    generator: String,

    // ── Topology facts ─────────────────────────────────────────────────
    /// Total number of committed rooms.
    room_count: u32,
    /// Number of rooms on the lower layer (Z=0).
    lower_room_count: u32,
    /// Number of rooms on the upper layer (Z=192).
    upper_room_count: u32,
    /// Total number of committed portals.
    portal_count: u32,
    /// Total number of committed transitions (stairs).
    transition_count: u32,
    /// Total number of committed routes.
    route_count: u32,

    // ── Composition facts ──────────────────────────────────────────────
    /// Grammar families represented in the output.
    grammar_families: Vec<String>,
    /// Whether minimum-identity constraints are satisfied.
    identity_satisfied: bool,

    // ── Source budget facts ────────────────────────────────────────────
    /// Estimated total face count (conservative upper bound).
    estimated_faces: u32,
    /// Actual emitted face count (computed from serialized .map).
    actual_faces: u32,
    /// Estimated total entity count.
    estimated_entities: u32,
    /// Actual emitted entity count.
    actual_entities: u32,
    /// Actual emitted world brush count.
    actual_brushes: u32,

    // ── Spawn and lights ───────────────────────────────────────────────
    /// Spawn origin in Quake units.
    spawn_origin: (i32, i32, i32),
    /// Number of light entities.
    light_count: u32,

    // ── Spatial bounds ─────────────────────────────────────────────────
    /// Axis-aligned bounding box of all rooms:
    /// `(min_x, min_y, min_z, max_x, max_y, max_z)` in Quake units.
    bounds: (i32, i32, i32, i32, i32, i32),

    // ── Layer classification ───────────────────────────────────────────
    /// Whether any upper-layer rooms exist.
    has_upper_layer: bool,
}

impl EnhancedV3Metadata {
    /// Build metadata from the completed pipeline outcome.
    ///
    /// All values are computed from the frozen configuration, topology,
    /// and serialized map text.
    pub fn new(
        config: &V3Config,
        topology: &CommittedTopology,
        grammar_families: Vec<String>,
        identity_satisfied: bool,
        estimated_faces: u32,
        estimated_entities: u32,
        actual_faces: u32,
        actual_entities: u32,
        actual_brushes: u32,
        spawn_origin: (i32, i32, i32),
        light_count: u32,
    ) -> Self {
        let lower_room_count = topology.rooms.iter().filter(|r| r.layer == 0).count() as u32;
        let upper_room_count = topology.rooms.iter().filter(|r| r.layer == 1).count() as u32;

        let mut min_x = i32::MAX;
        let mut min_y = i32::MAX;
        let mut min_z = i32::MAX;
        let mut max_x = i32::MIN;
        let mut max_y = i32::MIN;
        let mut max_z = i32::MIN;

        for room in &topology.rooms {
            let x0 = room.shell.0;
            let y0 = room.shell.1;
            let x1 = room.shell.2;
            let y1 = room.shell.3;
            let z0 = room.floor_z;
            let z1 = z0 + room.dims.2 as i32;

            min_x = min_x.min(x0);
            min_y = min_y.min(y0);
            min_z = min_z.min(z0);
            max_x = max_x.max(x1);
            max_y = max_y.max(y1);
            max_z = max_z.max(z1);
        }

        // If no rooms, use zero bounds
        if topology.rooms.is_empty() {
            min_x = 0;
            min_y = 0;
            min_z = 0;
            max_x = 0;
            max_y = 0;
            max_z = 0;
        }

        Self {
            seed: config.seed,
            preset: config.preset.tag().to_string(),
            xy_extent: config.xy_extent,
            schema_version: "v3".to_string(),
            generator: "bsp_generator/enhanced_v3".to_string(),
            room_count: topology.rooms.len() as u32,
            lower_room_count,
            upper_room_count,
            portal_count: topology.portals.len() as u32,
            transition_count: topology.transitions.len() as u32,
            route_count: topology.routes.len() as u32,
            grammar_families,
            identity_satisfied,
            estimated_faces,
            actual_faces,
            estimated_entities,
            actual_entities,
            actual_brushes,
            spawn_origin,
            light_count,
            bounds: (min_x, min_y, min_z, max_x, max_y, max_z),
            has_upper_layer: upper_room_count > 0,
        }
    }

    // ── Read-only accessors ────────────────────────────────────────────

    /// The master seed.
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// The density preset name.
    pub fn preset(&self) -> &str {
        &self.preset
    }

    /// The requested XY extent.
    pub fn xy_extent(&self) -> u32 {
        self.xy_extent
    }

    /// Schema version.
    pub fn schema_version(&self) -> &str {
        &self.schema_version
    }

    /// Generator identity.
    pub fn generator(&self) -> &str {
        &self.generator
    }

    /// Total room count.
    pub fn room_count(&self) -> u32 {
        self.room_count
    }

    /// Lower-layer room count.
    pub fn lower_room_count(&self) -> u32 {
        self.lower_room_count
    }

    /// Upper-layer room count.
    pub fn upper_room_count(&self) -> u32 {
        self.upper_room_count
    }

    /// Portal count.
    pub fn portal_count(&self) -> u32 {
        self.portal_count
    }

    /// Transition (stair) count.
    pub fn transition_count(&self) -> u32 {
        self.transition_count
    }

    /// Route count.
    pub fn route_count(&self) -> u32 {
        self.route_count
    }

    /// Grammar families in output.
    pub fn grammar_families(&self) -> &[String] {
        &self.grammar_families
    }

    /// Whether minimum identity satisfied.
    pub fn identity_satisfied(&self) -> bool {
        self.identity_satisfied
    }

    /// Estimated face count.
    pub fn estimated_faces(&self) -> u32 {
        self.estimated_faces
    }

    /// Actual face count.
    pub fn actual_faces(&self) -> u32 {
        self.actual_faces
    }

    /// Estimated entity count.
    pub fn estimated_entities(&self) -> u32 {
        self.estimated_entities
    }

    /// Actual entity count.
    pub fn actual_entities(&self) -> u32 {
        self.actual_entities
    }

    /// Actual brush count.
    pub fn actual_brushes(&self) -> u32 {
        self.actual_brushes
    }

    /// Spawn origin.
    pub fn spawn_origin(&self) -> (i32, i32, i32) {
        self.spawn_origin
    }

    /// Light count.
    pub fn light_count(&self) -> u32 {
        self.light_count
    }

    /// Spatial bounds: `(min_x, min_y, min_z, max_x, max_y, max_z)`.
    pub fn bounds(&self) -> (i32, i32, i32, i32, i32, i32) {
        self.bounds
    }

    /// Whether the map has an upper layer.
    pub fn has_upper_layer(&self) -> bool {
        self.has_upper_layer
    }

    /// Check that estimated faces ≥ actual faces (budget constraint).
    pub fn face_budget_satisfied(&self) -> bool {
        self.estimated_faces >= self.actual_faces
    }

    /// Check that estimated entities ≥ actual entities.
    pub fn entity_budget_satisfied(&self) -> bool {
        self.estimated_entities >= self.actual_entities
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::super::config::V3Config;
    use super::*;

    #[test]
    fn metadata_all_accessors_return_expected_types() {
        let topology = CommittedTopology {
            rooms: vec![],
            surfaces: vec![],
            portals: vec![],
            routes: vec![],
            transitions: vec![],
        };
        let config = V3Config::nominal_sparse();
        let meta = EnhancedV3Metadata::new(
            &config,
            &topology,
            vec!["test".to_string()],
            true,
            100,
            10,
            80,
            8,
            12,
            (32, 32, 48),
            3,
        );

        assert_eq!(meta.seed(), 0);
        assert_eq!(meta.preset(), "sparse");
        assert_eq!(meta.schema_version(), "v3");
        assert_eq!(meta.generator(), "bsp_generator/enhanced_v3");
        assert_eq!(meta.room_count(), 0);
        assert!(meta.identity_satisfied());
        assert!(meta.face_budget_satisfied());
        assert!(meta.entity_budget_satisfied());
        assert_eq!(meta.spawn_origin(), (32, 32, 48));
    }

    #[test]
    fn metadata_face_underrun_detected() {
        let topology = CommittedTopology {
            rooms: vec![],
            surfaces: vec![],
            portals: vec![],
            routes: vec![],
            transitions: vec![],
        };
        let config = V3Config::nominal_sparse();
        let meta = EnhancedV3Metadata::new(
            &config,
            &topology,
            vec![],
            true,
            50,
            5,
            100,
            5,
            20,
            (0, 0, 0),
            0,
        );
        assert!(!meta.face_budget_satisfied());
    }
}
