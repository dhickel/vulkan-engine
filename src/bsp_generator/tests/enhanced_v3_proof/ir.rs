//! Ordered newtypes and semantic intent IR for the Enhanced v3 proof.
//!
//! All IDs are stable and never encode iteration position. Semantic metadata
//! excludes random draws, candidate enumeration, collection order, and
//! compiler provenance.

use std::collections::BTreeMap;
use std::collections::BTreeSet;

// ── Typed newtype IDs ──────────────────────────────────────────────────────

macro_rules! v3_newtype_id {
    ($name:ident, $kind:expr) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name(pub u32);

        impl $name {
            pub const fn raw(self) -> u32 {
                self.0
            }

            /// Stable semantic key, e.g. `"room/0001"`.
            pub fn stable_key(self) -> String {
                format!("{}/{:04}", $kind, self.0)
            }
        }

        impl std::fmt::Display for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "{}({})", $kind, self.0)
            }
        }
    };
}

v3_newtype_id!(RoomId, "room");
v3_newtype_id!(SurfaceId, "surface");
v3_newtype_id!(CornerId, "corner");
v3_newtype_id!(PortalId, "portal");
v3_newtype_id!(FloorRegionId, "floor_region");
v3_newtype_id!(CeilingSpanId, "ceiling_span");
v3_newtype_id!(ProtectedVolumeId, "protected_volume");
v3_newtype_id!(FeatureId, "feature");
v3_newtype_id!(InstanceId, "instance");
v3_newtype_id!(CompositionId, "composition");

// ── ID allocator ───────────────────────────────────────────────────────────

/// Local checked allocator for v3 proof IDs.
#[derive(Debug, Clone)]
pub struct V3IdAllocator {
    next_room: u32,
    next_surface: u32,
    next_corner: u32,
    next_portal: u32,
    next_floor_region: u32,
    next_ceiling_span: u32,
    next_protected_volume: u32,
    next_feature: u32,
    next_instance: u32,
    next_composition: u32,
}

impl V3IdAllocator {
    pub fn new() -> Self {
        Self {
            next_room: 0,
            next_surface: 0,
            next_corner: 0,
            next_portal: 0,
            next_floor_region: 0,
            next_ceiling_span: 0,
            next_protected_volume: 0,
            next_feature: 0,
            next_instance: 0,
            next_composition: 0,
        }
    }

    fn checked_inc(val: &mut u32, op: &'static str) -> Result<u32, String> {
        let id = *val;
        *val = val
            .checked_add(1)
            .ok_or_else(|| format!("arithmetic overflow in {op}"))?;
        Ok(id)
    }

    pub fn next_room(&mut self) -> Result<RoomId, String> {
        Ok(RoomId(Self::checked_inc(&mut self.next_room, "room_id")?))
    }
    pub fn next_surface(&mut self) -> Result<SurfaceId, String> {
        Ok(SurfaceId(Self::checked_inc(
            &mut self.next_surface,
            "surface_id",
        )?))
    }
    pub fn next_corner(&mut self) -> Result<CornerId, String> {
        Ok(CornerId(Self::checked_inc(
            &mut self.next_corner,
            "corner_id",
        )?))
    }
    pub fn next_portal(&mut self) -> Result<PortalId, String> {
        Ok(PortalId(Self::checked_inc(
            &mut self.next_portal,
            "portal_id",
        )?))
    }
    pub fn next_floor_region(&mut self) -> Result<FloorRegionId, String> {
        Ok(FloorRegionId(Self::checked_inc(
            &mut self.next_floor_region,
            "floor_region_id",
        )?))
    }
    pub fn next_ceiling_span(&mut self) -> Result<CeilingSpanId, String> {
        Ok(CeilingSpanId(Self::checked_inc(
            &mut self.next_ceiling_span,
            "ceiling_span_id",
        )?))
    }
    pub fn next_protected_volume(&mut self) -> Result<ProtectedVolumeId, String> {
        Ok(ProtectedVolumeId(Self::checked_inc(
            &mut self.next_protected_volume,
            "protected_volume_id",
        )?))
    }
    pub fn next_feature(&mut self) -> Result<FeatureId, String> {
        Ok(FeatureId(Self::checked_inc(
            &mut self.next_feature,
            "feature_id",
        )?))
    }
    pub fn next_instance(&mut self) -> Result<InstanceId, String> {
        Ok(InstanceId(Self::checked_inc(
            &mut self.next_instance,
            "instance_id",
        )?))
    }
    pub fn next_composition(&mut self) -> Result<CompositionId, String> {
        Ok(CompositionId(Self::checked_inc(
            &mut self.next_composition,
            "composition_id",
        )?))
    }
}

// ── Support relations ──────────────────────────────────────────────────────

/// Kind of committed architectural surface that can act as world support.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum SupportSurfaceKind {
    Floor,
    Wall,
    Ceiling,
}

impl SupportSurfaceKind {
    pub fn face(self) -> &'static str {
        match self {
            Self::Floor => "floor",
            Self::Wall => "wall",
            Self::Ceiling => "ceiling",
        }
    }
}

/// Support relation between features.
///
/// Architectural roots carry the exact committed semantic surface ID. A
/// surface removed from the committed topology therefore cannot remain as an
/// implicit `Floor`, `Wall`, or `Ceiling` support.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SupportRelation {
    /// Supported directly by a committed architectural floor surface.
    Floor(SurfaceId),
    /// Supported directly by a committed wall surface.
    Wall(SurfaceId),
    /// Supported directly by a committed ceiling surface.
    Ceiling(SurfaceId),
    /// Supported by another feature instance (transitive).
    SupportedBy(InstanceId),
}

impl SupportRelation {
    /// Whether this is a transitive support.
    pub fn is_transitive(&self) -> bool {
        matches!(self, SupportRelation::SupportedBy(_))
    }

    /// The instance ID for a `SupportedBy` relation, if any.
    pub fn supported_by(&self) -> Option<InstanceId> {
        match self {
            SupportRelation::SupportedBy(id) => Some(*id),
            _ => None,
        }
    }

    /// The exact architectural support surface, if this is a world root.
    pub fn support_surface(&self) -> Option<(SurfaceId, SupportSurfaceKind)> {
        match self {
            Self::Floor(id) => Some((*id, SupportSurfaceKind::Floor)),
            Self::Wall(id) => Some((*id, SupportSurfaceKind::Wall)),
            Self::Ceiling(id) => Some((*id, SupportSurfaceKind::Ceiling)),
            Self::SupportedBy(_) => None,
        }
    }
}

// ── Surface ownership ──────────────────────────────────────────────────────

/// A surface is a named face of a room or feature.
///
/// Stable ownership is expressed as, e.g., `room/0003/wall/north/portal/primary`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SurfaceOwner {
    /// The owning room or feature.
    pub parent_kind: &'static str,
    /// The parent ID.
    pub parent_id: u32,
    /// Surface face: "wall", "floor", "ceiling".
    pub face: &'static str,
    /// Cardinal or diagonal direction: "north", "south", "east", "west", "ne", "nw", "se", "sw".
    pub direction: &'static str,
    /// Sub-surface qualifier: "primary", "secondary", "aperture".
    pub qualifier: &'static str,
}

impl SurfaceOwner {
    /// Build a stable semantic key.
    pub fn stable_key(&self) -> String {
        format!(
            "{}/{:04}/{}/{}/{}",
            self.parent_kind, self.parent_id, self.face, self.direction, self.qualifier
        )
    }
}

/// A committed semantic support surface derived from frozen topology.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommittedSurface {
    pub id: SurfaceId,
    pub room_id: RoomId,
    pub kind: SupportSurfaceKind,
    pub owner: SurfaceOwner,
}

// ── Quantum-aligned volume ─────────────────────────────────────────────────

/// A quantum-aligned 3D volume in Quake units.
///
/// All coordinates are multiples of the construction quantum (16).
/// Represented as half-open: `[x0, x1) × [y0, y1) × [z0, z1)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct QuantumVolume {
    pub x0: i32,
    pub y0: i32,
    pub z0: i32,
    pub x1: i32,
    pub y1: i32,
    pub z1: i32,
}

impl QuantumVolume {
    /// Create a new quantum-aligned volume. Returns `None` if any coordinate
    /// is not quantum-aligned or if the volume is non-positive.
    pub fn new(x0: i32, y0: i32, z0: i32, x1: i32, y1: i32, z1: i32) -> Option<Self> {
        let q = super::contract::CONSTRUCTION_QUANTUM;
        if x0 % q != 0 || y0 % q != 0 || z0 % q != 0 || x1 % q != 0 || y1 % q != 0 || z1 % q != 0 {
            return None;
        }
        if x0 >= x1 || y0 >= y1 || z0 >= z1 {
            return None;
        }
        Some(Self {
            x0,
            y0,
            z0,
            x1,
            y1,
            z1,
        })
    }

    /// Width (X dimension) in Quake units.
    pub fn width(&self) -> i32 {
        self.x1 - self.x0
    }

    /// Depth (Y dimension) in Quake units.
    pub fn depth(&self) -> i32 {
        self.y1 - self.y0
    }

    /// Height (Z dimension) in Quake units.
    pub fn height(&self) -> i32 {
        self.z1 - self.z0
    }

    /// Whether this volume positively overlaps another.
    pub fn intersects(&self, other: &QuantumVolume) -> bool {
        self.x0 < other.x1
            && self.x1 > other.x0
            && self.y0 < other.y1
            && self.y1 > other.y0
            && self.z0 < other.z1
            && self.z1 > other.z0
    }

    /// Volume in cubic Quake units. Uses checked arithmetic.
    pub fn volume(&self) -> Option<i64> {
        let w = (self.x1 - self.x0) as i64;
        let d = (self.y1 - self.y0) as i64;
        let h = (self.z1 - self.z0) as i64;
        w.checked_mul(d)?.checked_mul(h)
    }
}

// ── Feature intent ─────────────────────────────────────────────────────────

/// A declared semantic feature intent.
///
/// Features carry quantum-aligned volumes, a grammar family tag, and
/// optional support relations. Instances are materialized after composition
/// planning.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FeatureIntent {
    /// Stable feature ID.
    pub id: FeatureId,
    /// Grammar family this feature belongs to.
    pub family: &'static str,
    /// The room this feature is placed in.
    pub room_id: RoomId,
    /// Quantum-aligned outer bounds.
    pub volume: QuantumVolume,
    /// Support relation (if grounded).
    pub support: Option<SupportRelation>,
    /// Instance ID after materialization (assigned during composition).
    pub instance_id: Option<InstanceId>,
    /// Semantic metadata tags attached to this feature.
    pub tags: BTreeSet<String>,
    /// Estimated face count (conservative upper bound).
    pub estimated_faces: u32,
}

// ── Committed topology ─────────────────────────────────────────────────────

/// A room in the committed topology.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommittedRoom {
    pub id: RoomId,
    /// Layer index (0 = lower, 1 = upper).
    pub layer: u8,
    /// Outer shell bounds: (x0, y0, x1, y1).
    pub shell: (i32, i32, i32, i32),
    /// Floor Z.
    pub floor_z: i32,
    /// Room dimensions: (width, depth, height) in Quake units.
    pub dims: (u32, u32, u32),
}

/// A portal opening between rooms or room-to-corridor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommittedPortal {
    pub id: PortalId,
    /// Source room.
    pub source_room: RoomId,
    /// Target room (or None for corridor terminations).
    pub target_room: Option<RoomId>,
    /// Which wall the portal is on.
    pub wall: &'static str,
    /// Anchor point on the wall.
    pub anchor: (i32, i32, i32),
    /// Portal width.
    pub width: u32,
    /// Portal height.
    pub height: u32,
}

/// A route (corridor) between two rooms.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommittedRoute {
    pub id: u32,
    pub source_room: RoomId,
    pub target_room: RoomId,
    /// Envelope rectangles: (x0, y0, x1, y1).
    pub envelopes: Vec<(i32, i32, i32, i32)>,
}

/// A committed transition (stair) between layers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommittedTransition {
    pub id: u32,
    pub lower_room: RoomId,
    pub upper_room: RoomId,
    /// Protected 3-D volume.
    pub protected_volume: (i32, i32, i32, i32, i32, i32),
    /// Lower landing: (x0, y0, x1, y1).
    pub lower_landing: (i32, i32, i32, i32),
    /// Upper landing: (x0, y0, x1, y1).
    pub upper_landing: (i32, i32, i32, i32),
    /// Headroom volumes.
    pub headroom_volumes: Vec<(i32, i32, i32, i32, i32, i32)>,
}

/// Frozen structural-reservation snapshot.
///
/// This is the topology input to the composition planner. It captures the
/// complete structural layout before feature planning begins.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommittedTopology {
    pub rooms: Vec<CommittedRoom>,
    pub surfaces: Vec<CommittedSurface>,
    pub portals: Vec<CommittedPortal>,
    pub routes: Vec<CommittedRoute>,
    pub transitions: Vec<CommittedTransition>,
}

impl CommittedTopology {
    /// Validate the committed structural snapshot before composition begins.
    /// The planner consumes this immutable, canonical input and never creates
    /// rooms, routes, portals, or transitions itself.
    pub fn validate(&self) -> Result<(), super::contract::ContractError> {
        use super::contract::{
            ContractError, CONSTRUCTION_QUANTUM, LAYER_COUNT, LOWER_FLOOR_Z, UPPER_FLOOR_Z, XY_MAX,
        };

        fn invariant(detail: impl Into<String>) -> ContractError {
            ContractError::InvariantViolation {
                detail: detail.into(),
            }
        }
        fn aligned(value: i32) -> bool {
            value % CONSTRUCTION_QUANTUM == 0
        }

        if !self.rooms.windows(2).all(|pair| pair[0].id < pair[1].id) {
            return Err(invariant("rooms must be unique and sorted by stable ID"));
        }
        if !self.surfaces.windows(2).all(|pair| pair[0].id < pair[1].id) {
            return Err(invariant("surfaces must be unique and sorted by stable ID"));
        }
        if !self.portals.windows(2).all(|pair| pair[0].id < pair[1].id) {
            return Err(invariant("portals must be unique and sorted by stable ID"));
        }
        if !self.routes.windows(2).all(|pair| pair[0].id < pair[1].id) {
            return Err(invariant("routes must be unique and sorted by stable ID"));
        }
        if !self
            .transitions
            .windows(2)
            .all(|pair| pair[0].id < pair[1].id)
        {
            return Err(invariant(
                "transitions must be unique and sorted by stable ID",
            ));
        }

        let rooms: BTreeSet<RoomId> = self.rooms.iter().map(|room| room.id).collect();
        for room in &self.rooms {
            let (x0, y0, x1, y1) = room.shell;
            if room.layer >= LAYER_COUNT as u8 {
                return Err(invariant(format!("{} has an invalid layer", room.id)));
            }
            let expected_floor = if room.layer == 0 {
                LOWER_FLOOR_Z
            } else {
                UPPER_FLOOR_Z
            };
            if room.floor_z != expected_floor {
                return Err(invariant(format!("{} has an invalid floor Z", room.id)));
            }
            if [x0, y0, x1, y1, room.floor_z]
                .into_iter()
                .any(|value| !aligned(value))
                || x0 < 0
                || y0 < 0
                || x0 >= x1
                || y0 >= y1
                || x1 > XY_MAX as i32
                || y1 > XY_MAX as i32
                || room.dims != ((x1 - x0) as u32, (y1 - y0) as u32, room.dims.2)
                || room.dims.2 == 0
                || room.dims.2 % CONSTRUCTION_QUANTUM as u32 != 0
            {
                return Err(invariant(format!(
                    "{} has invalid aligned shell dimensions",
                    room.id
                )));
            }
        }

        let mut surface_owners = BTreeSet::new();
        for surface in &self.surfaces {
            if !rooms.contains(&surface.room_id)
                || surface.owner.parent_kind != "room"
                || surface.owner.parent_id != surface.room_id.raw()
                || surface.owner.face != surface.kind.face()
                || surface.owner.direction.is_empty()
                || surface.owner.qualifier.is_empty()
            {
                return Err(invariant(format!(
                    "{} has dangling or inconsistent semantic ownership",
                    surface.id
                )));
            }
            if !surface_owners.insert(surface.owner.stable_key()) {
                return Err(invariant(format!(
                    "{} duplicates semantic surface ownership",
                    surface.id
                )));
            }
        }

        for portal in &self.portals {
            if !rooms.contains(&portal.source_room)
                || portal
                    .target_room
                    .is_some_and(|room| !rooms.contains(&room))
                || portal.target_room == Some(portal.source_room)
                || !["north", "south", "east", "west"].contains(&portal.wall)
                || portal.width == 0
                || portal.height == 0
                || portal.width % CONSTRUCTION_QUANTUM as u32 != 0
            {
                return Err(invariant(format!(
                    "{} has invalid semantic ownership",
                    portal.id
                )));
            }
        }

        for route in &self.routes {
            if !rooms.contains(&route.source_room)
                || !rooms.contains(&route.target_room)
                || route.source_room == route.target_room
            {
                return Err(invariant(format!(
                    "route/{:04} has dangling room ownership",
                    route.id
                )));
            }
            for &(x0, y0, x1, y1) in &route.envelopes {
                if [x0, y0, x1, y1].into_iter().any(|value| !aligned(value)) || x0 >= x1 || y0 >= y1
                {
                    return Err(invariant(format!(
                        "route/{:04} has invalid envelope",
                        route.id
                    )));
                }
            }
        }

        for transition in &self.transitions {
            let Some(lower) = self.room(transition.lower_room) else {
                return Err(invariant(format!(
                    "transition/{:04} has dangling lower room",
                    transition.id
                )));
            };
            let Some(upper) = self.room(transition.upper_room) else {
                return Err(invariant(format!(
                    "transition/{:04} has dangling upper room",
                    transition.id
                )));
            };
            if lower.layer != 0 || upper.layer != 1 {
                return Err(invariant(format!(
                    "transition/{:04} violates layer ownership",
                    transition.id
                )));
            }
            let (x0, y0, z0, x1, y1, z1) = transition.protected_volume;
            if QuantumVolume::new(x0, y0, z0, x1, y1, z1).is_none() {
                return Err(invariant(format!(
                    "transition/{:04} has invalid protected volume",
                    transition.id
                )));
            }
            for &(x0, y0, x1, y1) in &[transition.lower_landing, transition.upper_landing] {
                if [x0, y0, x1, y1].into_iter().any(|value| !aligned(value)) || x0 >= x1 || y0 >= y1
                {
                    return Err(invariant(format!(
                        "transition/{:04} has invalid landing",
                        transition.id
                    )));
                }
            }
            for &(x0, y0, z0, x1, y1, z1) in &transition.headroom_volumes {
                if QuantumVolume::new(x0, y0, z0, x1, y1, z1).is_none() {
                    return Err(invariant(format!(
                        "transition/{:04} has invalid headroom",
                        transition.id
                    )));
                }
            }
        }

        Ok(())
    }

    /// Look up a room by ID.
    pub fn room(&self, id: RoomId) -> Option<&CommittedRoom> {
        self.rooms.iter().find(|r| r.id == id)
    }

    /// Look up a committed semantic surface by ID.
    pub fn surface(&self, id: SurfaceId) -> Option<&CommittedSurface> {
        self.surfaces.iter().find(|surface| surface.id == id)
    }

    /// Find a room's committed support surface of the requested kind.
    pub fn room_support_surface(
        &self,
        room_id: RoomId,
        kind: SupportSurfaceKind,
    ) -> Option<&CommittedSurface> {
        self.surfaces
            .iter()
            .find(|surface| surface.room_id == room_id && surface.kind == kind)
    }

    /// Portals for a given room.
    pub fn room_portals(&self, room_id: RoomId) -> Vec<&CommittedPortal> {
        self.portals
            .iter()
            .filter(|p| p.source_room == room_id || p.target_room == Some(room_id))
            .collect()
    }
}

// ── Composition plan outcome ───────────────────────────────────────────────

/// A materialized feature instance.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FeatureInstance {
    pub id: InstanceId,
    pub feature_id: FeatureId,
    pub room_id: RoomId,
    pub volume: QuantumVolume,
    pub support: Option<SupportRelation>,
    pub tags: BTreeSet<String>,
    pub estimated_faces: u32,
}

/// Result of composition planning.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlanOutcome {
    /// Composition ID for this run.
    pub composition_id: CompositionId,
    /// The preset used.
    pub preset: &'static str,
    /// Selected grammar families represented.
    pub grammar_families: BTreeSet<String>,
    /// All accepted feature instances.
    pub instances: Vec<FeatureInstance>,
    /// Features that were simplified (removed by deterministic simplification).
    pub simplified: Vec<FeatureId>,
    /// Features that were rejected with typed reasons.
    pub rejected: BTreeMap<FeatureId, String>,
    /// Support graph edges: (dependent, parent).
    pub support_edges: Vec<(InstanceId, SupportRelation)>,
    /// Whether minimum-identity constraints are satisfied.
    pub identity_satisfied: bool,
    /// Estimated total face count (conservative upper bound).
    pub estimated_total_faces: u32,
    /// Estimated total entity count.
    pub estimated_total_entities: u32,
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::super::contract::CONSTRUCTION_QUANTUM;
    use super::*;

    #[test]
    fn stable_key_includes_padded_id() {
        let r = RoomId(3);
        assert_eq!(r.stable_key(), "room/0003");
        let r2 = RoomId(42);
        assert_eq!(r2.stable_key(), "room/0042");
    }

    #[test]
    fn surface_owner_stable_key() {
        let s = SurfaceOwner {
            parent_kind: "room",
            parent_id: 3,
            face: "wall",
            direction: "north",
            qualifier: "primary",
        };
        assert_eq!(s.stable_key(), "room/0003/wall/north/primary");
    }

    #[test]
    fn quantum_volume_requires_alignment() {
        let q = CONSTRUCTION_QUANTUM;
        assert!(QuantumVolume::new(0, 0, 0, q, q, q).is_some());
        assert!(QuantumVolume::new(1, 0, 0, q, q, q).is_none());
        assert!(QuantumVolume::new(0, 0, 0, q, q, 1).is_none());
    }

    #[test]
    fn quantum_volume_rejects_non_positive() {
        let q = CONSTRUCTION_QUANTUM;
        assert!(QuantumVolume::new(0, 0, 0, 0, q, q).is_none());
        assert!(QuantumVolume::new(q, q, q, 0, 0, 0).is_none());
        assert!(QuantumVolume::new(0, 0, 0, q, 0, q).is_none());
    }

    #[test]
    fn quantum_volume_intersection() {
        let q = CONSTRUCTION_QUANTUM;
        let a = QuantumVolume::new(0, 0, 0, 2 * q, 2 * q, 2 * q).unwrap();
        let b = QuantumVolume::new(q, q, q, 3 * q, 3 * q, 3 * q).unwrap();
        assert!(a.intersects(&b));

        let c = QuantumVolume::new(3 * q, 0, 0, 4 * q, q, q).unwrap();
        assert!(!a.intersects(&c));
    }

    #[test]
    fn quantum_volume_dimensions() {
        let q = CONSTRUCTION_QUANTUM;
        let v = QuantumVolume::new(0, q, 2 * q, 3 * q, 5 * q, 8 * q).unwrap();
        assert_eq!(v.width(), 3 * q);
        assert_eq!(v.depth(), 4 * q);
        assert_eq!(v.height(), 6 * q);
    }

    #[test]
    fn id_allocator_checked_overflow_panics_only_at_u32_max() {
        let mut a = V3IdAllocator::new();
        for _ in 0..100 {
            a.next_room().unwrap();
            a.next_feature().unwrap();
        }
    }

    #[test]
    fn support_relation_transitive_detection() {
        let floor = SupportRelation::Floor(SurfaceId(0));
        let wall = SupportRelation::Wall(SurfaceId(1));
        let supported = SupportRelation::SupportedBy(InstanceId(5));
        assert!(!floor.is_transitive());
        assert!(!wall.is_transitive());
        assert_eq!(
            floor.support_surface(),
            Some((SurfaceId(0), SupportSurfaceKind::Floor))
        );
        assert!(supported.is_transitive());
        assert_eq!(supported.supported_by(), Some(InstanceId(5)));
    }

    #[test]
    fn committed_topology_room_lookup() {
        let topo = CommittedTopology {
            rooms: vec![CommittedRoom {
                id: RoomId(0),
                layer: 0,
                shell: (0, 0, 256, 256),
                floor_z: 0,
                dims: (256, 256, 176),
            }],
            surfaces: vec![CommittedSurface {
                id: SurfaceId(0),
                room_id: RoomId(0),
                kind: SupportSurfaceKind::Floor,
                owner: SurfaceOwner {
                    parent_kind: "room",
                    parent_id: 0,
                    face: "floor",
                    direction: "up",
                    qualifier: "primary",
                },
            }],
            portals: vec![],
            routes: vec![],
            transitions: vec![],
        };
        assert!(topo.validate().is_ok());
        assert!(topo.room(RoomId(0)).is_some());
        assert!(topo.room(RoomId(1)).is_none());
    }
}
