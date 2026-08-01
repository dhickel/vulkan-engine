//! Enhanced V3 semantic core — isolated exact deterministic pipeline.
//!
//! This module implements the V3 dungeon generation pipeline with exact
//! i128 geometry, SHA-256 domain-separated randomness, immutable intent
//! declarations, and deterministic .map emission.
//!
//! # Architecture
//!
//! ```text
//! V3Config → Footprints → CommittedTopology → Assembly → .map bytes
//! ```
//!
//! Each stage is a validated data structure with typed construction.
//! The pipeline is a pure function from `(V3Config)` to canonical `.map` bytes.
//!
//! # Production status
//!
//! This is the first production release of Enhanced V3. It does not change
//! existing v1/v2 behavior, profile dispatch, `GenerationProfile`, or the
//! public API (Phase 05 responsibility).

pub mod assembly;
pub mod composition;
pub mod config;
pub mod emission;
pub mod error;
pub mod footprint;
pub mod geometry;
pub mod ids;
pub mod intent;
pub mod metadata;
pub mod pipeline;
pub mod reservation;
pub mod rng;
pub mod topology;

// ── Re-exports ────────────────────────────────────────────────────────────

pub use assembly::{Assembly, AssemblyBrush, BrushRole, Interface, ProtectedVolume, Support};
pub use composition::{CompositionPlan, GrammarDescriptor};
pub use config::{
    ArchType, FeatureFlags, GrammarMode, V3Config, V3Preset, CONSTRUCTION_QUANTUM,
    GRAMMAR_FAMILIES, HEADROOM, ROUTE_WIDTH,
};
pub use emission::{emit_map_text, emit_map_text_with_minlight, texture_for_role};
pub use error::V3Error;
pub use footprint::{build_footprints, Footprint, FootprintLayout};
pub use geometry::{
    half_space_vertices, CanonicalPlane, ConvexBrush, FaceRole, Point3, Rational, Vector3,
};
pub use ids::{
    CommittedPortal, CommittedRoom, CommittedRoute, CommittedSurface, CommittedTopology,
    CommittedTransition, FeatureId, FeatureInstance, FeatureIntent, InstanceId, PlanOutcome,
    PortalId, QuantumVolume, RoomId, SupportRelation, SupportSurfaceKind, SurfaceId, SurfaceOwner,
    V3IdAllocator,
};
pub use intent::plan_composition;
pub use metadata::EnhancedV3Metadata;
pub use pipeline::{run_pipeline, V3PipelineOutput};
pub use reservation::{build_reservations, Reservation, ReservationSet};
pub use rng::{tags, CandidateSelector, V3Seed, V3StageSeed};
pub use topology::{build_topology, compute_reservations, compute_reservations_with_config};

// ── Public pipeline entry point ────────────────────────────────────────────

/// Generate a canonical .map string from a V3 configuration.
///
/// This is a compatibility wrapper around [`pipeline::run_pipeline`] that
/// returns only the map text. Use [`run_pipeline`] for the full output
/// including metadata.
///
/// ```text
/// config → seed → footprints → topology → reservations → assembly → .map
/// ```
///
/// # Determinism
///
/// Two calls with identical `config` produce byte-identical `.map` output.
pub fn generate_v3(config: &V3Config) -> Result<String, V3Error> {
    let output = pipeline::run_pipeline(config)?;
    Ok(output.map_text)
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generate_v3_sparse_produces_map() {
        let config = V3Config::nominal_sparse();
        let map = generate_v3(&config).unwrap();
        assert!(!map.is_empty());
        assert!(map.contains("worldspawn"));
        assert!(map.contains("info_player_start"));
        assert!(map.contains("light"));
        assert!(map.contains("cc0_dungeon_v2.wad"));
    }

    #[test]
    fn generate_v3_moderate_produces_map() {
        let config = V3Config::nominal_moderate();
        let map = generate_v3(&config).unwrap();
        assert!(!map.is_empty());
        assert!(map.contains("worldspawn"));
    }

    #[test]
    fn generate_v3_rich_produces_map() {
        let config = V3Config::nominal_rich();
        let map = generate_v3(&config).unwrap();
        assert!(!map.is_empty());
        assert!(map.contains("worldspawn"));
    }

    #[test]
    fn generate_v3_deterministic() {
        let config = V3Config::nominal_sparse();
        let map1 = generate_v3(&config).unwrap();
        let map2 = generate_v3(&config).unwrap();
        assert_eq!(map1, map2);
    }

    #[test]
    fn generate_v3_different_seeds_different_output() {
        let config_a = V3Config::new(0, V3Preset::Sparse, 2048).unwrap();
        let config_b = V3Config::new(42, V3Preset::Sparse, 2048).unwrap();
        let map_a = generate_v3(&config_a).unwrap();
        let map_b = generate_v3(&config_b).unwrap();
        assert!(!map_a.is_empty());
        assert!(!map_b.is_empty());
    }

    #[test]
    fn generated_map_has_valid_brush_syntax() {
        let config = V3Config::nominal_sparse();
        let map = generate_v3(&config).unwrap();
        let open_count = map.matches('{').count();
        let close_count = map.matches('}').count();
        assert_eq!(open_count, close_count, "mismatched braces in map");
        assert!(open_count > 2, "expected multiple brush blocks");
    }

    #[test]
    fn texture_roles_are_known() {
        assert_eq!(texture_for_role(BrushRole::WallShell), "bs_wall");
        assert_eq!(texture_for_role(BrushRole::FloorSlab), "bs_floor");
        assert_eq!(texture_for_role(BrushRole::CeilingSlab), "bs_ceil");
        assert_eq!(texture_for_role(BrushRole::Column), "bs_accent");
        assert_eq!(texture_for_role(BrushRole::Feature), "bs_accent");
    }
}
