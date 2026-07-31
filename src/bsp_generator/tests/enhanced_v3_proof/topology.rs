//! Deterministic two-layer fixture topology for the Enhanced v3 proof.
//!
//! Produces a frozen [`CommittedTopology`] from footprint layout and
//! seed. Ordering is deterministic: primary room, portal neighbor,
//! lower transition host, upper landing/room, spawn+light reservations.
//! All typed volumes are frozen before composition begins.

use super::contract::{self, ContractError, ProofConfig};
use super::footprint::{Footprint, FootprintLayout};
use super::ir::{
    CommittedPortal, CommittedRoom, CommittedRoute, CommittedSurface, CommittedTopology,
    CommittedTransition, SupportSurfaceKind, SurfaceOwner, V3IdAllocator,
};
use super::seed::V3Seed;

/// Build a committed topology from footprints and seed.
///
/// The returned topology is immutable — once built, it is the canonical
/// input to the composition planner.
pub fn build_topology(
    config: &ProofConfig,
    footprints: &[Footprint],
    layouts: &[FootprintLayout],
    seed: V3Seed,
    alloc: &mut V3IdAllocator,
) -> Result<CommittedTopology, ContractError> {
    if footprints.is_empty() || layouts.is_empty() {
        return Err(ContractError::InvariantViolation {
            detail: "topology requires at least one footprint and layout".into(),
        });
    }

    let layout = &layouts[0];
    if layout.primary >= footprints.len()
        || layout.secondary >= footprints.len()
        || layout.transition_lower >= footprints.len()
        || layout.transition_upper >= footprints.len()
    {
        return Err(ContractError::InvariantViolation {
            detail: "layout indices out of bounds".into(),
        });
    }

    let q = contract::CONSTRUCTION_QUANTUM;

    // ── Build committed rooms ─────────────────────────────────────────
    let mut rooms: Vec<CommittedRoom> = Vec::new();
    let mut surfaces: Vec<CommittedSurface> = Vec::new();

    for fp in footprints {
        let (x0, y0, x1, y1) = fp.aabb;
        let width = (x1 - x0) as u32;
        let depth = (y1 - y0) as u32;
        let height = contract::ROOM_HEIGHT as u32;

        rooms.push(CommittedRoom {
            id: fp.room_id,
            layer: fp.layer,
            shell: (x0, y0, x1, y1),
            floor_z: fp.floor_z,
            dims: (width, depth, height),
        });

        // Create committed support surfaces for this room
        // Floor
        let floor_id = alloc
            .next_surface()
            .map_err(|e| ContractError::InvariantViolation { detail: e })?;
        surfaces.push(CommittedSurface {
            id: floor_id,
            room_id: fp.room_id,
            kind: SupportSurfaceKind::Floor,
            owner: SurfaceOwner {
                parent_kind: "room",
                parent_id: fp.room_id.raw(),
                face: "floor",
                direction: "up",
                qualifier: "primary",
            },
        });

        // TODO: could add wall and ceiling surfaces per direction, but
        // for the integrated thin slice we only need floor surfaces
        // for grounded assembly support.
    }

    // ── Build portals ─────────────────────────────────────────────────
    let primary = &footprints[layout.primary];
    let secondary = &footprints[layout.secondary];

    // Portal from primary to secondary (east wall of primary, west wall of secondary)
    let primary_aabb = primary.aabb;
    let secondary_aabb = secondary.aabb;

    let portal_width = contract::ROUTE_WIDTH;
    let portal_height = contract::HEADROOM;

    // Primary east wall → secondary west wall
    let anchor_x = primary_aabb.2; // east wall X
    let anchor_y = (primary_aabb.1 + primary_aabb.3) / 2; // midpoint Y
    let anchor_z = primary.floor_z + q + portal_height as i32 / 2;

    let portal_id = alloc
        .next_portal()
        .map_err(|e| ContractError::InvariantViolation { detail: e })?;

    let portal = CommittedPortal {
        id: portal_id,
        source_room: primary.room_id,
        target_room: Some(secondary.room_id),
        wall: "east",
        anchor: (anchor_x, anchor_y, anchor_z),
        width: portal_width as u32,
        height: portal_height as u32,
    };

    let portals = vec![portal];

    // ── Build routes ─────────────────────────────────────────────────
    // Route between primary and secondary rooms
    let envelope_x0 = primary_aabb.2 - portal_width as i32 / 2;
    let envelope_y0 = anchor_y - portal_width as i32 / 2;
    let envelope_x1 = secondary_aabb.0 + portal_width as i32 / 2;
    let envelope_y1 = anchor_y + portal_width as i32 / 2;

    let route = CommittedRoute {
        id: 0,
        source_room: primary.room_id,
        target_room: secondary.room_id,
        envelopes: vec![(envelope_x0, envelope_y0, envelope_x1, envelope_y1)],
    };

    let routes = vec![route];

    // ── Build transition ──────────────────────────────────────────────
    let lower_fp = &footprints[layout.transition_lower];
    let upper_fp = &footprints[layout.transition_upper];

    let lower_aabb = lower_fp.aabb;
    let upper_aabb = upper_fp.aabb;

    // Transition: protected volume between lower room ceiling and upper room floor.
    // Find overlapping X region, then place stair in that overlap.
    let q = contract::CONSTRUCTION_QUANTUM;
    let overlap_x0 = lower_aabb.0.max(upper_aabb.0);
    let overlap_x1 = lower_aabb.2.min(upper_aabb.2);
    // If no X overlap, use a narrow central column
    let pv_x0 = if overlap_x0 < overlap_x1 {
        ((overlap_x0 + overlap_x1) / 2 / q) * q - 2 * q
    } else {
        ((lower_aabb.0 + lower_aabb.2) / 2 / q) * q - 2 * q
    };
    let pv_x1 = pv_x0 + 4 * q;

    // Y: from lower room north edge to upper room south edge
    let pv_y0 = lower_aabb.3;
    let pv_y1 = upper_aabb.1;
    let pv_z0 = contract::LOWER_FLOOR_Z;
    let pv_z1 = contract::UPPER_FLOOR_Z + contract::ROOM_HEIGHT as i32;

    // Ensure dimensions are positive and quantum-aligned
    let pv_x0 = (pv_x0 / q) * q;
    let pv_x1 = (pv_x1 / q) * q;
    let pv_y0 = (pv_y0 / q) * q;
    let pv_y1 = (pv_y1 / q) * q;

    if pv_x0 >= pv_x1 || pv_y0 >= pv_y1 || pv_z0 >= pv_z1 {
        return Err(ContractError::InvariantViolation {
            detail: format!(
                "invalid transition protected volume: ({pv_x0},{pv_y0},{pv_z0})-({pv_x1},{pv_y1},{pv_z1})"
            ),
        });
    }

    let lower_landing_y0 = (lower_aabb.3 / q) * q;
    let lower_landing = (pv_x0, lower_landing_y0, pv_x1, lower_landing_y0 + 2 * q);
    let upper_landing_y1 = (upper_aabb.1 / q) * q;
    let upper_landing = (pv_x0, upper_landing_y1 - 2 * q, pv_x1, upper_landing_y1);

    let transition = CommittedTransition {
        id: 0,
        lower_room: lower_fp.room_id,
        upper_room: upper_fp.room_id,
        protected_volume: (pv_x0, pv_y0, pv_z0, pv_x1, pv_y1, pv_z1),
        lower_landing,
        upper_landing,
        headroom_volumes: vec![],
    };

    let transitions = vec![transition];

    // ── Assemble and validate ─────────────────────────────────────────
    let topology = CommittedTopology {
        rooms,
        surfaces,
        portals,
        routes,
        transitions,
    };

    topology.validate()?;

    // Validate XY bounds against config
    for room in &topology.rooms {
        let x1 = room.shell.2;
        let y1 = room.shell.3;
        if x1 > config.xy_extent as i32 || y1 > config.xy_extent as i32 {
            return Err(ContractError::InvariantViolation {
                detail: format!(
                    "room {:?} shell exceeds xy_extent {}",
                    room.id, config.xy_extent
                ),
            });
        }
    }

    Ok(topology)
}

/// Compute spawn and light reservation volumes from the frozen topology.
///
/// Returns (spawn_volume, light_volumes) where spawn is a single
/// QuantumVolume and lights is a list of QuantumVolumes.
pub fn compute_reservations(
    topology: &CommittedTopology,
    _alloc: &mut V3IdAllocator,
) -> Result<(super::ir::QuantumVolume, Vec<super::ir::QuantumVolume>), ContractError> {
    // Spawn goes in the primary room (first room)
    let spawn_room = topology
        .rooms
        .first()
        .ok_or_else(|| ContractError::InvariantViolation {
            detail: "no rooms in topology".into(),
        })?;

    let q = contract::CONSTRUCTION_QUANTUM;
    let cx = (spawn_room.shell.0 + spawn_room.shell.2) / 2;
    let cy = (spawn_room.shell.1 + spawn_room.shell.3) / 2;

    let spawn_volume = super::ir::QuantumVolume::new(
        cx - q,
        cy - q,
        spawn_room.floor_z + q,
        cx + q,
        cy + q,
        spawn_room.floor_z + q + contract::HEADROOM,
    )
    .ok_or_else(|| ContractError::InvariantViolation {
        detail: "invalid spawn volume".into(),
    })?;

    // Place a light in each room (single cell volume at ceiling level)
    let mut light_volumes = Vec::new();
    for room in &topology.rooms {
        let lx = ((room.shell.0 + room.shell.2) / 2 / q) * q;
        let ly = ((room.shell.1 + room.shell.3) / 2 / q) * q;
        let lz = room.floor_z + room.dims.2 as i32 - 2 * q;

        if let Some(vol) =
            super::ir::QuantumVolume::new(lx - q, ly - q, lz - q, lx + q, ly + q, lz + q)
        {
            light_volumes.push(vol);
        }
    }

    Ok((spawn_volume, light_volumes))
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::super::contract::Preset;
    use super::super::footprint::build_footprints;
    use super::*;

    #[test]
    fn build_sparse_topology() {
        let config = ProofConfig::new(Preset::Sparse, 2048).unwrap();
        let mut alloc = V3IdAllocator::new();
        let (footprints, layouts) = build_footprints(&config, V3Seed::new(0), &mut alloc).unwrap();
        let topology =
            build_topology(&config, &footprints, &layouts, V3Seed::new(0), &mut alloc).unwrap();

        assert_eq!(topology.rooms.len(), 3);
        assert_eq!(topology.portals.len(), 1);
        assert_eq!(topology.routes.len(), 1);
        assert_eq!(topology.transitions.len(), 1);

        // Must have surfaces
        assert_eq!(topology.surfaces.len(), footprints.len());
    }

    #[test]
    fn topology_validation_passes() {
        let config = ProofConfig::new(Preset::Moderate, 2048).unwrap();
        let mut alloc = V3IdAllocator::new();
        let (footprints, layouts) = build_footprints(&config, V3Seed::new(42), &mut alloc).unwrap();
        let topology =
            build_topology(&config, &footprints, &layouts, V3Seed::new(42), &mut alloc).unwrap();

        topology.validate().unwrap();
    }

    #[test]
    fn topology_deterministic() {
        let config = ProofConfig::new(Preset::Sparse, 2048).unwrap();
        let mut alloc1 = V3IdAllocator::new();
        let mut alloc2 = V3IdAllocator::new();

        let (fp1, lo1) = build_footprints(&config, V3Seed::new(0), &mut alloc1).unwrap();
        let (fp2, lo2) = build_footprints(&config, V3Seed::new(0), &mut alloc2).unwrap();

        let topo1 = build_topology(&config, &fp1, &lo1, V3Seed::new(0), &mut alloc1).unwrap();
        let topo2 = build_topology(&config, &fp2, &lo2, V3Seed::new(0), &mut alloc2).unwrap();

        assert_eq!(topo1.rooms.len(), topo2.rooms.len());
        assert_eq!(topo1.portals.len(), topo2.portals.len());
    }

    #[test]
    fn topology_bounds_within_config() {
        let config = ProofConfig::new(Preset::Sparse, 2048).unwrap();
        let mut alloc = V3IdAllocator::new();
        let (footprints, layouts) = build_footprints(&config, V3Seed::new(0), &mut alloc).unwrap();
        let topology =
            build_topology(&config, &footprints, &layouts, V3Seed::new(0), &mut alloc).unwrap();

        for room in &topology.rooms {
            assert!(room.shell.2 <= config.xy_extent as i32);
            assert!(room.shell.3 <= config.xy_extent as i32);
        }
    }

    #[test]
    fn spawn_and_light_reservations() {
        let config = ProofConfig::new(Preset::Sparse, 2048).unwrap();
        let mut alloc = V3IdAllocator::new();
        let (footprints, layouts) = build_footprints(&config, V3Seed::new(0), &mut alloc).unwrap();
        let topology =
            build_topology(&config, &footprints, &layouts, V3Seed::new(0), &mut alloc).unwrap();

        let (spawn, lights) = compute_reservations(&topology, &mut alloc).unwrap();
        assert!(spawn.width() > 0);
        assert_eq!(lights.len(), topology.rooms.len());
    }
}
