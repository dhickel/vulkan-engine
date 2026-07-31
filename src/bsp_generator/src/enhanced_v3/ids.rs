//! Stable identity types for the Enhanced V3 pipeline.
//!
//! All IDs are newtype wrappers with stable keys. They never encode
//! iteration position, collection order, or compiler provenance.

use std::collections::BTreeSet;
use std::fmt;

use super::error::V3Error;

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

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
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

/// Local checked allocator for Enhanced V3 IDs.
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

    fn checked_inc(val: &mut u32, kind: &'static str) -> Result<u32, V3Error> {
        let id = *val;
        *val = val.checked_add(1).ok_or(V3Error::IdOverflow { kind })?;
        Ok(id)
    }

    pub fn next_room(&mut self) -> Result<RoomId, V3Error> {
        Ok(RoomId(Self::checked_inc(&mut self.next_room, "room")?))
    }
    pub fn next_surface(&mut self) -> Result<SurfaceId, V3Error> {
        Ok(SurfaceId(Self::checked_inc(
            &mut self.next_surface,
            "surface",
        )?))
    }
    pub fn next_corner(&mut self) -> Result<CornerId, V3Error> {
        Ok(CornerId(Self::checked_inc(
            &mut self.next_corner,
            "corner",
        )?))
    }
    pub fn next_portal(&mut self) -> Result<PortalId, V3Error> {
        Ok(PortalId(Self::checked_inc(
            &mut self.next_portal,
            "portal",
        )?))
    }
    pub fn next_floor_region(&mut self) -> Result<FloorRegionId, V3Error> {
        Ok(FloorRegionId(Self::checked_inc(
            &mut self.next_floor_region,
            "floor_region",
        )?))
    }
    pub fn next_ceiling_span(&mut self) -> Result<CeilingSpanId, V3Error> {
        Ok(CeilingSpanId(Self::checked_inc(
            &mut self.next_ceiling_span,
            "ceiling_span",
        )?))
    }
    pub fn next_protected_volume(&mut self) -> Result<ProtectedVolumeId, V3Error> {
        Ok(ProtectedVolumeId(Self::checked_inc(
            &mut self.next_protected_volume,
            "protected_volume",
        )?))
    }
    pub fn next_feature(&mut self) -> Result<FeatureId, V3Error> {
        Ok(FeatureId(Self::checked_inc(
            &mut self.next_feature,
            "feature",
        )?))
    }
    pub fn next_instance(&mut self) -> Result<InstanceId, V3Error> {
        Ok(InstanceId(Self::checked_inc(
            &mut self.next_instance,
            "instance",
        )?))
    }
    pub fn next_composition(&mut self) -> Result<CompositionId, V3Error> {
        Ok(CompositionId(Self::checked_inc(
            &mut self.next_composition,
            "composition",
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
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SurfaceOwner {
    /// The owning room or feature kind.
    pub parent_kind: String,
    /// The parent ID.
    pub parent_id: u32,
    /// Surface face: "wall", "floor", "ceiling".
    pub face: String,
    /// Cardinal or diagonal direction.
    pub direction: String,
    /// Sub-surface qualifier: "primary", "secondary", "aperture".
    pub qualifier: String,
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
        let q = super::config::CONSTRUCTION_QUANTUM;
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
}

// ── Feature intent ─────────────────────────────────────────────────────────

/// A declared semantic feature intent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FeatureIntent {
    /// Stable feature ID.
    pub id: FeatureId,
    /// Grammar family this feature belongs to.
    pub family: String,
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
    /// CCW footprint polygon vertices (quantum-aligned).
    pub footprint_vertices: Vec<(i32, i32)>,
    /// Chamfer corner patterns present in this room.
    pub chamfer_corners: Vec<(i32, i32)>,
    /// Chamfer size in Quake units (0 if rectangular).
    pub chamfer_size: i32,
    /// Whether this room has 45° diagonal wall faces.
    pub is_chamfered: bool,
}

/// A portal opening between rooms.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommittedPortal {
    pub id: PortalId,
    /// Source room.
    pub source_room: RoomId,
    /// Target room (or None for corridor terminations).
    pub target_room: Option<RoomId>,
    /// Which wall the portal is on.
    pub wall: String,
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
    /// Supported lower connector outside the lower wall.
    pub lower_approach: (i32, i32, i32, i32),
    /// Exact 192-unit projected tread run.
    pub tread_run: (i32, i32, i32, i32),
    /// Authoritative positive-volume tread/riser columns in ascent order.
    pub tread_boxes: Vec<(i32, i32, i32, i32, i32, i32)>,
    /// Supported upper connector to the upper wall.
    pub upper_approach: (i32, i32, i32, i32),
    /// Lower landing: (x0, y0, x1, y1).
    pub lower_landing: (i32, i32, i32, i32),
    /// Upper landing: (x0, y0, x1, y1).
    pub upper_landing: (i32, i32, i32, i32),
    /// Exact clear volumes over both approaches, every tread, and the crest.
    pub headroom_volumes: Vec<(i32, i32, i32, i32, i32, i32)>,
}

/// Frozen structural-reservation snapshot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommittedTopology {
    pub rooms: Vec<CommittedRoom>,
    pub surfaces: Vec<CommittedSurface>,
    pub portals: Vec<CommittedPortal>,
    pub routes: Vec<CommittedRoute>,
    pub transitions: Vec<CommittedTransition>,
}

impl CommittedTopology {
    /// Look up a room by ID.
    pub fn room(&self, id: RoomId) -> Option<&CommittedRoom> {
        self.rooms.iter().find(|r| r.id == id)
    }

    /// Look up a committed semantic surface by ID.
    pub fn surface(&self, id: SurfaceId) -> Option<&CommittedSurface> {
        self.surfaces.iter().find(|surface| surface.id == id)
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
    pub preset: String,
    /// Selected grammar families represented.
    pub grammar_families: BTreeSet<String>,
    /// All accepted feature instances.
    pub instances: Vec<FeatureInstance>,
    /// Features that were simplified (removed by deterministic simplification).
    pub simplified: Vec<FeatureId>,
    /// Features that were rejected with typed reasons.
    pub rejected: Vec<(FeatureId, String)>,
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
    use super::super::config::CONSTRUCTION_QUANTUM;
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
            parent_kind: "room".into(),
            parent_id: 3,
            face: "wall".into(),
            direction: "north".into(),
            qualifier: "primary".into(),
        };
        assert_eq!(s.stable_key(), "room/0003/wall/north/primary");
    }

    #[test]
    fn quantum_volume_requires_alignment() {
        let q = CONSTRUCTION_QUANTUM;
        assert!(QuantumVolume::new(0, 0, 0, q, q, q).is_some());
        assert!(QuantumVolume::new(1, 0, 0, q, q, q).is_none());
    }

    #[test]
    fn quantum_volume_rejects_non_positive() {
        let q = CONSTRUCTION_QUANTUM;
        assert!(QuantumVolume::new(0, 0, 0, 0, q, q).is_none());
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
    fn id_allocator_produces_unique_ids() {
        let mut a = V3IdAllocator::new();
        let r1 = a.next_room().unwrap();
        let r2 = a.next_room().unwrap();
        assert_ne!(r1, r2);
        assert_eq!(r1.raw(), 0);
        assert_eq!(r2.raw(), 1);
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
}
