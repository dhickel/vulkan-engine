//! Error types for the BSP runtime transaction coordinator.

use std::fmt;

/// Errors produced by the BSP runtime coordinator during prepare, validate,
/// commit, rollback, unload, and reload operations.
#[derive(Debug)]
pub enum BspRuntimeError {
    /// The generation counter has exhausted its representable range.
    GenerationExhausted,
    /// The generation token used for validation or commit does not match the
    /// coordinator's current generation — a newer prepare has superseded it.
    StaleGeneration { expected: u64, current: u64 },
    /// A BSP mount is already active; unload it first or use reload.
    OccupiedTarget,
    /// One or more app bridges failed during prepare, validate, or commit.
    BridgeFailure {
        bridge_name: String,
        phase: BridgePhase,
        message: String,
    },
    /// Rollback encountered errors. Individual sub-errors are aggregated,
    /// but the coordinator state may be inconsistent.
    RollbackFailure { failures: Vec<BridgeFailure> },
    /// The coordinator has been poisoned by a panic in a bridge hook during
    /// commit or rollback. No further BSP operations are accepted.
    CoordinatorPoisoned,
    /// The BSP source is unavailable (missing file, network error, etc.).
    SourceUnavailable { reason: String },
    /// Entity identity reconciliation on reload detected ambiguous matches.
    IdentityAmbiguous {
        entity_count: usize,
        context: String,
    },
}

/// Which phase of the two-step transaction a bridge failure occurred in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BridgePhase {
    Prepare,
    Validate,
    Commit,
    Rollback,
}

impl fmt::Display for BridgePhase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BridgePhase::Prepare => write!(f, "prepare"),
            BridgePhase::Validate => write!(f, "validate"),
            BridgePhase::Commit => write!(f, "commit"),
            BridgePhase::Rollback => write!(f, "rollback"),
        }
    }
}

/// A single bridge failure recorded during a multi-bridge operation.
#[derive(Debug)]
pub struct BridgeFailure {
    pub bridge_name: String,
    pub phase: BridgePhase,
    pub message: String,
}

impl BridgeFailure {
    pub fn new(
        bridge_name: impl Into<String>,
        phase: BridgePhase,
        message: impl Into<String>,
    ) -> Self {
        Self {
            bridge_name: bridge_name.into(),
            phase,
            message: message.into(),
        }
    }
}

impl fmt::Display for BspRuntimeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BspRuntimeError::GenerationExhausted => {
                write!(f, "BSP generation counter exhausted")
            }
            BspRuntimeError::StaleGeneration { expected, current } => {
                write!(
                    f,
                    "stale generation: expected {}, current {}",
                    expected, current
                )
            }
            BspRuntimeError::OccupiedTarget => {
                write!(f, "BSP mount target is already occupied")
            }
            BspRuntimeError::BridgeFailure {
                bridge_name,
                phase,
                message,
            } => {
                write!(
                    f,
                    "bridge '{}' failed during {}: {}",
                    bridge_name, phase, message
                )
            }
            BspRuntimeError::RollbackFailure { failures } => {
                write!(
                    f,
                    "rollback failed with {} sub-failure(s): {}",
                    failures.len(),
                    failures
                        .iter()
                        .map(|bf| format!(
                            "[{} during {}: {}]",
                            bf.bridge_name, bf.phase, bf.message
                        ))
                        .collect::<Vec<_>>()
                        .join("; ")
                )
            }
            BspRuntimeError::CoordinatorPoisoned => {
                write!(f, "BSP coordinator is poisoned")
            }
            BspRuntimeError::SourceUnavailable { reason } => {
                write!(f, "BSP source unavailable: {}", reason)
            }
            BspRuntimeError::IdentityAmbiguous {
                entity_count,
                context,
            } => {
                write!(
                    f,
                    "identity ambiguous: {} entities in context '{}'",
                    entity_count, context
                )
            }
        }
    }
}

impl std::error::Error for BspRuntimeError {}
