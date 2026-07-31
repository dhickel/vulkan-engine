//! Schema-v3 semantic metadata for the Enhanced v3 proof.
//!
//! Derived from frozen topology + plan/assembly outcomes. Stable identities,
//! reservations, supports, and outcomes. Excludes planner internals.

use serde::{Deserialize, Serialize};

use super::contract::ProofConfig;
use super::ir::{CommittedTopology, PlanOutcome, SupportRelation};

/// Schema-v3 metadata record for an integrated proof run.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProofMetadata {
    /// Schema version tag.
    pub schema: String,
    /// The preset used.
    pub preset: String,
    /// XY extent of the proof volume.
    pub xy_extent: u32,
    /// Construction quantum.
    pub construction_quantum: i32,
    /// Number of rooms.
    pub room_count: u32,
    /// Number of portals.
    pub portal_count: u32,
    /// Number of routes.
    pub route_count: u32,
    /// Number of transitions.
    pub transition_count: u32,
    /// Number of committed support surfaces.
    pub surface_count: u32,
    /// Number of feature instances.
    pub instance_count: u32,
    /// Number of simplified (removed) features.
    pub simplified_count: u32,
    /// Number of rejected features.
    pub rejected_count: u32,
    /// Selected grammar families.
    pub grammar_families: Vec<String>,
    /// Estimated total faces.
    pub estimated_total_faces: u32,
    /// Estimated total entities.
    pub estimated_total_entities: u32,
    /// Identity constraints satisfied.
    pub identity_satisfied: bool,
    /// Spawn origin (in worldspawn coordinates).
    pub spawn_origin: [i32; 3],
    /// Spawn yaw angle.
    pub spawn_yaw: i32,
    /// Light origins (one per room).
    pub light_origins: Vec<[i32; 3]>,
    /// Room records.
    pub rooms: Vec<RoomRecord>,
    /// Portal records.
    pub portals: Vec<PortalRecord>,
    /// Feature instance records.
    pub instances: Vec<InstanceRecord>,
    /// Support edges: (dependent_id, parent_kind).
    pub support_edges: Vec<SupportEdgeRecord>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RoomRecord {
    /// Stable room ID.
    pub id: String,
    /// Layer index.
    pub layer: u8,
    /// Shell: (x0, y0, x1, y1).
    pub shell: [i32; 4],
    /// Floor Z.
    pub floor_z: i32,
    /// Dimensions: (width, depth, height).
    pub dims: [u32; 3],
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PortalRecord {
    /// Stable portal ID.
    pub id: String,
    /// Source room ID.
    pub source_room: String,
    /// Target room ID (optional for corridor terminations).
    pub target_room: Option<String>,
    /// Cardinal wall direction.
    pub wall: String,
    /// Portal width.
    pub width: u32,
    /// Portal height.
    pub height: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InstanceRecord {
    /// Stable instance ID.
    pub id: String,
    /// Feature ID.
    pub feature_id: String,
    /// Owning room ID.
    pub room_id: String,
    /// Volume: (x0, y0, z0, x1, y1, z1).
    pub volume: [i32; 6],
    /// Support kind (floor, wall, ceiling, or supported_by parent).
    pub support_kind: String,
    /// Estimated face count.
    pub estimated_faces: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SupportEdgeRecord {
    /// Dependent instance ID.
    pub dependent: String,
    /// Parent or support kind.
    pub parent: String,
}

/// Build metadata from the proof pipeline outcomes.
pub fn build_metadata(
    topology: &CommittedTopology,
    outcome: &PlanOutcome,
    config: &ProofConfig,
    spawn_origin: (i32, i32, i32),
    spawn_yaw: i32,
    light_origins: &[(i32, i32, i32)],
) -> ProofMetadata {
    let rooms: Vec<RoomRecord> = topology
        .rooms
        .iter()
        .map(|r| RoomRecord {
            id: r.id.stable_key(),
            layer: r.layer,
            shell: [r.shell.0, r.shell.1, r.shell.2, r.shell.3],
            floor_z: r.floor_z,
            dims: [r.dims.0, r.dims.1, r.dims.2],
        })
        .collect();

    let portals: Vec<PortalRecord> = topology
        .portals
        .iter()
        .map(|p| PortalRecord {
            id: p.id.stable_key(),
            source_room: p.source_room.stable_key(),
            target_room: p.target_room.map(|r| r.stable_key()),
            wall: p.wall.to_string(),
            width: p.width,
            height: p.height,
        })
        .collect();

    let instances: Vec<InstanceRecord> = outcome
        .instances
        .iter()
        .map(|fi| {
            let support_kind = match &fi.support {
                Some(SupportRelation::Floor(_)) => "floor".to_string(),
                Some(SupportRelation::Wall(_)) => "wall".to_string(),
                Some(SupportRelation::Ceiling(_)) => "ceiling".to_string(),
                Some(SupportRelation::SupportedBy(parent)) => {
                    format!("supported_by:{}", parent.stable_key())
                }
                None => "none".to_string(),
            };

            InstanceRecord {
                id: fi.id.stable_key(),
                feature_id: fi.feature_id.stable_key(),
                room_id: fi.room_id.stable_key(),
                volume: [
                    fi.volume.x0,
                    fi.volume.y0,
                    fi.volume.z0,
                    fi.volume.x1,
                    fi.volume.y1,
                    fi.volume.z1,
                ],
                support_kind,
                estimated_faces: fi.estimated_faces,
            }
        })
        .collect();

    let support_edges: Vec<SupportEdgeRecord> = outcome
        .support_edges
        .iter()
        .map(|(dependent, support)| {
            let parent = match support {
                SupportRelation::Floor(id) => format!("surface:{}", id.stable_key()),
                SupportRelation::Wall(id) => format!("surface:{}", id.stable_key()),
                SupportRelation::Ceiling(id) => format!("surface:{}", id.stable_key()),
                SupportRelation::SupportedBy(parent_id) => parent_id.stable_key(),
            };
            SupportEdgeRecord {
                dependent: dependent.stable_key(),
                parent,
            }
        })
        .collect();

    ProofMetadata {
        schema: "enhanced-v3-proof-metadata/v3".to_string(),
        preset: config.preset.tag().to_string(),
        xy_extent: config.xy_extent,
        construction_quantum: super::contract::CONSTRUCTION_QUANTUM,
        room_count: topology.rooms.len() as u32,
        portal_count: topology.portals.len() as u32,
        route_count: topology.routes.len() as u32,
        transition_count: topology.transitions.len() as u32,
        surface_count: topology.surfaces.len() as u32,
        instance_count: outcome.instances.len() as u32,
        simplified_count: outcome.simplified.len() as u32,
        rejected_count: outcome.rejected.len() as u32,
        grammar_families: outcome.grammar_families.iter().cloned().collect(),
        estimated_total_faces: outcome.estimated_total_faces,
        estimated_total_entities: outcome.estimated_total_entities,
        identity_satisfied: outcome.identity_satisfied,
        spawn_origin: [spawn_origin.0, spawn_origin.1, spawn_origin.2],
        spawn_yaw,
        light_origins: light_origins.iter().map(|l| [l.0, l.1, l.2]).collect(),
        rooms,
        portals,
        instances,
        support_edges,
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metadata_json_roundtrip() {
        let meta = ProofMetadata {
            schema: "enhanced-v3-proof-metadata/v3".into(),
            preset: "sparse".into(),
            xy_extent: 2048,
            construction_quantum: 16,
            room_count: 3,
            portal_count: 1,
            route_count: 1,
            transition_count: 1,
            surface_count: 3,
            instance_count: 1,
            simplified_count: 0,
            rejected_count: 0,
            grammar_families: vec!["portal_chamber".into()],
            estimated_total_faces: 120,
            estimated_total_entities: 3,
            identity_satisfied: true,
            spawn_origin: [64, 64, 24],
            spawn_yaw: 90,
            light_origins: vec![[64, 64, 160]],
            rooms: vec![],
            portals: vec![],
            instances: vec![],
            support_edges: vec![],
        };

        let json = serde_json::to_string_pretty(&meta).unwrap();
        let roundtripped: ProofMetadata = serde_json::from_str(&json).unwrap();
        assert_eq!(meta, roundtripped);
    }

    #[test]
    fn metadata_excludes_planner_internals() {
        // Metadata should not contain candidate enumeration, random draws,
        // or compiler provenance
        let meta = ProofMetadata {
            schema: "test".into(),
            preset: "sparse".into(),
            xy_extent: 2048,
            construction_quantum: 16,
            room_count: 0,
            portal_count: 0,
            route_count: 0,
            transition_count: 0,
            surface_count: 0,
            instance_count: 0,
            simplified_count: 0,
            rejected_count: 0,
            grammar_families: vec![],
            estimated_total_faces: 0,
            estimated_total_entities: 0,
            identity_satisfied: true,
            spawn_origin: [0, 0, 0],
            spawn_yaw: 0,
            light_origins: vec![],
            rooms: vec![],
            portals: vec![],
            instances: vec![],
            support_edges: vec![],
        };

        let json = serde_json::to_string(&meta).unwrap();
        // Must not contain transient fields
        assert!(!json.contains("random"));
        assert!(!json.contains("candidate_enum"));
        assert!(!json.contains("collection_order"));
        assert!(!json.contains("compiler"));
    }
}
