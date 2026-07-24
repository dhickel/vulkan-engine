use std::fmt;

/// All errors the dungeon generator can produce.
///
/// Every variant carries structured diagnostics; no error path panics or
/// silently falls back.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GeneratorError {
    /// Configuration validation failure — the supplied [`DungeonConfig`] is
    /// out of the frozen bounds or structurally invalid.
    InvalidConfig(String),

    /// Room placement exhausted the per-room attempt budget without
    /// successfully placing all rooms.
    PlacementExhausted {
        /// Total placement attempts consumed before exhaustion.
        attempts: u32,
    },

    /// Corridor routing exhausted the A* expansion budget without finding a
    /// path between required endpoints.
    RouteExhausted {
        /// Total A* node expansions consumed before exhaustion.
        expansions: u32,
    },

    /// An internal invariant was violated — indicates a generator bug, not
    /// user input error.
    InvariantViolation(String),

    /// The output .map text could not be serialized (e.g. formatting overflow
    /// or invalid state).
    SerializationFailed(String),

    /// Arithmetic overflow in a bounds or size computation.
    ArithmeticOverflow,
}

impl fmt::Display for GeneratorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GeneratorError::InvalidConfig(msg) => {
                write!(f, "invalid configuration: {}", msg)
            }
            GeneratorError::PlacementExhausted { attempts } => {
                write!(
                    f,
                    "room placement exhausted after {} attempts",
                    attempts
                )
            }
            GeneratorError::RouteExhausted { expansions } => {
                write!(
                    f,
                    "corridor routing exhausted after {} A* expansions",
                    expansions
                )
            }
            GeneratorError::InvariantViolation(msg) => {
                write!(f, "internal invariant violated: {}", msg)
            }
            GeneratorError::SerializationFailed(msg) => {
                write!(f, "serialization failed: {}", msg)
            }
            GeneratorError::ArithmeticOverflow => {
                write!(f, "arithmetic overflow in bounds computation")
            }
        }
    }
}

impl std::error::Error for GeneratorError {}
