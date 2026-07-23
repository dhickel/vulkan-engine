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
    /// Source content hash mismatch between persistence record and loaded bytes.
    SourceMismatch {
        expected: String,
        actual: String,
    },
    /// Unsupported persistence schema version.
    UnsupportedSchema {
        version: u32,
        current: u32,
    },
    /// Migration from a prior schema version is not supported.
    InvalidMigration {
        from_version: u32,
        reason: String,
    },
    /// Companion file hash mismatch during restore validation.
    CompanionMismatch {
        kind: String,
        expected: String,
        actual: String,
    },
    /// Model-mapping identity mismatch during restore validation.
    MappingMismatch {
        reason: String,
    },
    /// Required state is missing from the restored persistence payload.
    MissingRequiredState {
        detail: String,
    },
    /// External asset reference in persistence does not match current package state.
    InvalidExternalAsset {
        asset_path: String,
        reason: String,
    },
    /// Mutable behavior state in persistence is invalid or corrupt.
    InvalidMutableBehavior {
        detail: String,
    },
    /// The restore candidate failed validation and the active generation is unchanged.
    RestoreCancelled {
        active_asset_id: String,
        reason: String,
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
            BspRuntimeError::SourceMismatch { expected, actual } => {
                write!(
                    f,
                    "source content hash mismatch: expected {expected}, got {actual}"
                )
            }
            BspRuntimeError::UnsupportedSchema { version, current } => {
                write!(
                    f,
                    "unsupported persistence schema version {version} (current: {current})"
                )
            }
            BspRuntimeError::InvalidMigration {
                from_version,
                reason,
            } => {
                write!(
                    f,
                    "invalid migration from schema version {from_version}: {reason}"
                )
            }
            BspRuntimeError::CompanionMismatch {
                kind,
                expected,
                actual,
            } => {
                write!(
                    f,
                    "companion {kind} hash mismatch: expected {expected}, got {actual}"
                )
            }
            BspRuntimeError::MappingMismatch { reason } => {
                write!(f, "model-mapping mismatch: {reason}")
            }
            BspRuntimeError::MissingRequiredState { detail } => {
                write!(f, "missing required state: {detail}")
            }
            BspRuntimeError::InvalidExternalAsset { asset_path, reason } => {
                write!(
                    f,
                    "invalid external asset '{asset_path}': {reason}"
                )
            }
            BspRuntimeError::InvalidMutableBehavior { detail } => {
                write!(f, "invalid mutable behavior state: {detail}")
            }
            BspRuntimeError::RestoreCancelled {
                active_asset_id,
                reason,
            } => {
                write!(
                    f,
                    "restore cancelled (active: '{active_asset_id}'): {reason}"
                )
            }
        }
    }
}

impl std::error::Error for BspRuntimeError {}
