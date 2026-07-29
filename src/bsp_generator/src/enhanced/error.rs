//! Enhanced v2 error types — closed, typed, no string-matching dispatch.

use std::fmt;

/// Errors specific to the Enhanced v2 profile.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EnhancedError {
    /// A configuration field is out of its allowed range.
    ConfigOutOfRange {
        field: &'static str,
        value: u64,
        min: u64,
        max: u64,
    },
    /// The requested profile is not supported by this code path.
    WrongProfile { expected: &'static str },
    /// An arithmetic operation overflowed.
    ArithmeticOverflow { operation: &'static str },
    /// An ID is duplicated where uniqueness is required.
    DuplicateId { kind: &'static str, id: u32 },
    /// An ID is out of order (must be strictly increasing).
    IdOutOfOrder {
        kind: &'static str,
        id: u32,
        previous: u32,
    },
    /// A required contract value is absent or invalid.
    ContractViolation { detail: String },
    /// Room placement exhausted all attempts before placing every room.
    PlacementExhausted {
        rooms_placed: u32,
        total_attempts: u32,
    },
    /// A room cannot fit within the configured XY extent.
    RoomTooLarge {
        room_index: u32,
        width: u32,
        height: u32,
        xy_extent: u32,
    },
    /// Topology construction exhausted all backtracking alternatives.
    TopologyExhausted { detail: String },
    /// A* routing exhausted its expansion budget.
    RouteExhausted { expansions: u32 },
    /// A stair transition reservation failed (no compatible socket pair).
    TransitionReservationFailed { detail: String },
    /// Post-commit topology validation failed.
    TopologyValidationFailed { detail: String },
}

impl fmt::Display for EnhancedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ConfigOutOfRange {
                field,
                value,
                min,
                max,
            } => {
                write!(
                    f,
                    "config field '{}' value {} out of range [{}, {}]",
                    field, value, min, max
                )
            }
            Self::WrongProfile { expected } => {
                write!(f, "wrong profile: expected {}", expected)
            }
            Self::ArithmeticOverflow { operation } => {
                write!(f, "arithmetic overflow in {}", operation)
            }
            Self::DuplicateId { kind, id } => {
                write!(f, "duplicate {} ID: {}", kind, id)
            }
            Self::IdOutOfOrder { kind, id, previous } => {
                write!(
                    f,
                    "{} ID {} out of order (previous: {})",
                    kind, id, previous
                )
            }
            Self::ContractViolation { detail } => {
                write!(f, "contract violation: {}", detail)
            }
            Self::PlacementExhausted {
                rooms_placed,
                total_attempts,
            } => {
                write!(
                    f,
                    "placement exhausted after {} rooms placed, {} total attempts",
                    rooms_placed, total_attempts
                )
            }
            Self::RoomTooLarge {
                room_index,
                width,
                height,
                xy_extent,
            } => {
                write!(
                    f,
                    "room {} ({}×{}) cannot fit within xy_extent {}",
                    room_index, width, height, xy_extent
                )
            }
            Self::TopologyExhausted { detail } => {
                write!(f, "topology exhausted: {}", detail)
            }
            Self::RouteExhausted { expansions } => {
                write!(f, "A* routing exhausted after {} expansions", expansions)
            }
            Self::TransitionReservationFailed { detail } => {
                write!(f, "transition reservation failed: {}", detail)
            }
            Self::TopologyValidationFailed { detail } => {
                write!(f, "topology validation failed: {}", detail)
            }
        }
    }
}

impl std::error::Error for EnhancedError {}
