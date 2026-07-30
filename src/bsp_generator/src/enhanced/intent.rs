//! Enhanced v2 intent declarations — immutable, ordered, typed ownership.
//!
//! These records establish future geometry ownership without materializing
//! brushes, occupancy, connectivity, clearance, or compiler validity.
//! Every record carries a typed owner ID; canonical `Vec` order determines
//! selection and serialization order.

use super::error::EnhancedError;
use super::profile::StairType;

// ── Typed newtype IDs ──────────────────────────────────────────────────────

macro_rules! newtype_id {
    ($name:ident, $kind:expr) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name(pub u32);

        impl $name {
            pub const fn raw(self) -> u32 {
                self.0
            }
        }

        impl std::fmt::Display for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "{}({})", $kind, self.0)
            }
        }
    };
}

newtype_id!(LayerId, "LayerId");
newtype_id!(RoomId, "RoomId");
newtype_id!(SocketId, "SocketId");
newtype_id!(RouteId, "RouteId");
newtype_id!(TransitionId, "TransitionId");
newtype_id!(ReservationId, "ReservationId");
newtype_id!(ZoneId, "ZoneId");
newtype_id!(PaletteId, "PaletteId");

// ── ID allocator ───────────────────────────────────────────────────────────

/// Attempt-local checked allocator for typed Enhanced IDs.
#[derive(Debug, Clone)]
pub struct IdAllocator {
    next_layer: u32,
    next_room: u32,
    next_socket: u32,
    next_route: u32,
    next_transition: u32,
    next_reservation: u32,
    next_zone: u32,
    next_palette: u32,
}

impl IdAllocator {
    pub fn new() -> Self {
        Self {
            next_layer: 0,
            next_room: 0,
            next_socket: 0,
            next_route: 0,
            next_transition: 0,
            next_reservation: 0,
            next_zone: 0,
            next_palette: 0,
        }
    }

    fn checked_inc(val: &mut u32, op: &'static str) -> Result<u32, EnhancedError> {
        let id = *val;
        *val = val
            .checked_add(1)
            .ok_or(EnhancedError::ArithmeticOverflow { operation: op })?;
        Ok(id)
    }

    pub fn next_layer(&mut self) -> Result<LayerId, EnhancedError> {
        Ok(LayerId(Self::checked_inc(
            &mut self.next_layer,
            "layer_id",
        )?))
    }
    pub fn next_room(&mut self) -> Result<RoomId, EnhancedError> {
        Ok(RoomId(Self::checked_inc(&mut self.next_room, "room_id")?))
    }
    pub fn next_socket(&mut self) -> Result<SocketId, EnhancedError> {
        Ok(SocketId(Self::checked_inc(
            &mut self.next_socket,
            "socket_id",
        )?))
    }
    pub fn next_route(&mut self) -> Result<RouteId, EnhancedError> {
        Ok(RouteId(Self::checked_inc(
            &mut self.next_route,
            "route_id",
        )?))
    }
    pub fn next_transition(&mut self) -> Result<TransitionId, EnhancedError> {
        Ok(TransitionId(Self::checked_inc(
            &mut self.next_transition,
            "transition_id",
        )?))
    }
    pub fn next_reservation(&mut self) -> Result<ReservationId, EnhancedError> {
        Ok(ReservationId(Self::checked_inc(
            &mut self.next_reservation,
            "reservation_id",
        )?))
    }
    pub fn next_zone(&mut self) -> Result<ZoneId, EnhancedError> {
        Ok(ZoneId(Self::checked_inc(&mut self.next_zone, "zone_id")?))
    }
    pub fn next_palette(&mut self) -> Result<PaletteId, EnhancedError> {
        Ok(PaletteId(Self::checked_inc(
            &mut self.next_palette,
            "palette_id",
        )?))
    }
}

// ── Semantic intents ───────────────────────────────────────────────────────

/// A room in the Enhanced layout — purely semantic, not placed.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct RoomIntent {
    pub id: RoomId,
    /// Nominal layer (0 = lower, 1 = upper).
    pub layer: u8,
}

/// A socket (connection point) on a room boundary.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct SocketIntent {
    pub id: SocketId,
    pub room: RoomId,
}

/// A horizontal route between two sockets on the same layer.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct RouteIntent {
    pub id: RouteId,
    pub source_socket: SocketId,
    pub target_socket: SocketId,
    /// Room endpoints, retained so validation never infers graph ownership
    /// from allocation order.
    pub source_room: RoomId,
    pub target_room: RoomId,
    /// Canonically ordered, orthogonal centreline segments.
    pub path: Vec<((i32, i32), (i32, i32))>,
    /// Complete 64-unit projected reservations for `path`.
    pub envelopes: Vec<(i32, i32, i32, i32)>,
    /// Clear vertical extent for every corridor envelope.
    pub headroom: (i32, i32),
}

/// One materializable solid stair step. Bounds are half-open and include the
/// riser volume below the walkable tread surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct StairTread {
    pub bounds: (i32, i32, i32, i32, i32, i32),
}

/// One materialized-width upper-level approach segment. It is intentionally
/// separate from a horizontal `RouteIntent`: it belongs exclusively to its
/// transition and exists at the upper-floor clearance elevation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct TransitionApproachSegment {
    pub start: (i32, i32),
    pub end: (i32, i32),
    pub envelope: (i32, i32, i32, i32),
    pub z: (i32, i32),
}

/// A vertical transition (stair) between rooms on different layers.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct TransitionIntent {
    pub id: TransitionId,
    /// The concrete stair type selected for this transition.
    pub stair_type: StairType,
    pub lower_room: RoomId,
    pub upper_room: RoomId,
    pub lower_socket: SocketId,
    pub upper_socket: SocketId,
    /// Protected projected footprint `(x0, y0, x1, y1)`.
    pub footprint: (i32, i32, i32, i32),
    /// Protected 3-D volume `(x0, y0, z0, x1, y1, z1)`.
    pub protected_volume: (i32, i32, i32, i32, i32, i32),
    /// Direct room approaches and landings, each as an XY rectangle.
    pub lower_approach: (i32, i32, i32, i32),
    pub upper_approach: (i32, i32, i32, i32),
    pub lower_landing: (i32, i32, i32, i32),
    pub upper_landing: (i32, i32, i32, i32),
    /// Compatibility lower corners for the current emitter. `tread_boxes` is
    /// the authoritative materializable geometry.
    pub treads: Vec<(i32, i32, i32)>,
    /// Exact positive-volume tread/riser solids in ascent order.
    pub tread_boxes: Vec<StairTread>,
    /// Exact upper-level approach, from the ceiling exit to the selected upper
    /// room's wall opening. This prevents graph-only endpoint connections.
    pub upper_approach_segments: Vec<TransitionApproachSegment>,
    /// Every exact projected reservation owned by this transition. The union,
    /// not the bounding footprint, is the collision contract.
    pub reserved_projection: Vec<(i32, i32, i32, i32)>,
    /// Exact clear volumes above each tread and approach segment.
    pub headroom_volumes: Vec<(i32, i32, i32, i32, i32, i32)>,
    /// Compatibility bounding volume for consumers that need a conservative
    /// broad-phase query.
    pub headroom: (i32, i32, i32, i32, i32, i32),
    pub riser: i32,
    pub tread_depth: i32,
    pub sealed_shell: bool,
    /// Lower wall aperture through the lower host room.
    pub lower_wall_opening: WallOpening,
    /// Ceiling slab omission above the high treads; it is the actual vertical
    /// exit, not a strip derived from the upper room's wall.
    pub upper_ceiling_opening: CeilingOpening,
    /// Upper room wall aperture reached by `upper_approach_segments`.
    pub upper_wall_opening: WallOpening,
}

/// Describes an opening through a room wall for a stair entrance/exit.
///
/// Phase 2 emission uses this to split or omit wall brush solids at the
/// exact aperture location.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct WallOpening {
    /// Which wall face the opening is on.
    pub wall: super::placement::WallDirection,
    /// Minimum coordinate along the wall's tangent axis.
    pub tangent_min: i32,
    /// Maximum coordinate along the wall's tangent axis.
    pub tangent_max: i32,
    /// Bottom Z of the opening (floor-relative).
    pub bottom_z: i32,
    /// Top Z of the opening.
    pub top_z: i32,
}

/// Describes an opening through a ceiling for the upper stair exit.
///
/// Phase 2 emission uses this to omit ceiling slab solids where the stair
/// opens into the upper layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct CeilingOpening {
    /// The XY rectangle of the ceiling opening `(x0, y0, x1, y1)`.
    pub rect: (i32, i32, i32, i32),
    /// The Z level at which the ceiling opening exists.
    pub z: i32,
}

/// A zone of rooms sharing a theme palette.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct ZoneIntent {
    pub id: ZoneId,
    pub palette: PaletteId,
    pub rooms: Vec<RoomId>,
}

// ── Validated construction helpers ─────────────────────────────────────────

/// Validate that IDs in a sorted slice are unique and in increasing order.
pub fn validate_sorted_ids<T: Copy>(
    ids: &[T],
    kind: &'static str,
    to_u32: impl Fn(T) -> u32,
) -> Result<(), EnhancedError> {
    for w in ids.windows(2) {
        let a = to_u32(w[0]);
        let b = to_u32(w[1]);
        if a == b {
            return Err(EnhancedError::DuplicateId { kind, id: a });
        }
        if a > b {
            return Err(EnhancedError::IdOutOfOrder {
                kind,
                id: b,
                previous: a,
            });
        }
    }
    Ok(())
}
