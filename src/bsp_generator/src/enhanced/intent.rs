//! Enhanced v2 intent declarations — immutable, ordered, typed ownership.
//!
//! These records establish future geometry ownership without materializing
//! brushes, occupancy, connectivity, clearance, or compiler validity.
//! Every record carries a typed owner ID; canonical `Vec` order determines
//! selection and serialization order.

use super::error::EnhancedError;

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
}

/// A vertical transition (stair) between rooms on different layers.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct TransitionIntent {
    pub id: TransitionId,
    pub lower_room: RoomId,
    pub upper_room: RoomId,
    pub lower_socket: SocketId,
    pub upper_socket: SocketId,
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
