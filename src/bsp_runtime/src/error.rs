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
    /// An illegal candidate state transition was attempted.
    InvalidCandidateTransition {
        current: CandidatePhase,
        attempted: CandidatePhase,
        detail: String,
    },
    /// A stale renderer completion was received — the lease was sent to
    /// cancellation/retirement and the current candidate was not mutated.
    StaleRendererCompletion {
        candidate_generation: u64,
        current_generation: u64,
    },
    /// A duplicate renderer-ready lease was received: the candidate already
    /// holds a ready lease and the new one was sent to retirement.
    DuplicateReadyLease {
        generation: u64,
    },
    /// The renderer could not accept an opaque retirement handoff.
    RetirementHandoffFailed {
        reason: String,
    },
    /// A supposedly prevalidated publication operation violated its contract.
    CommitContractViolated {
        detail: String,
    },
    /// Teardown of an active bridge receipt failed. The receipt is quarantined,
    /// not silently dropped.
    TeardownQuarantined {
        bridge_name: String,
        message: String,
        unattempted: usize,
    },
    /// A bridge invariant was violated (registration mismatch, double activation, etc.).
    BridgeInvariantViolated {
        bridge_name: String,
        phase: BridgePhase,
        detail: String,
    },
}

/// Which phase of the two-step transaction a bridge failure occurred in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BridgePhase {
    Prepare,
    Validate,
    /// Activation (Phase 05): non-fallible publication after validation.
    Activate,
    /// Teardown (Phase 05): exact-once active receipt cleanup.
    Teardown,
    #[doc(hidden)]
    Commit,
    Rollback,
}

// ── Invariant Violation String Constants ─────────────────────────────

pub(crate) const INVARIANT_REGISTRATION_MISMATCH: &str =
    "bridge registration mismatch";
pub(crate) const INVARIANT_DOUBLE_ACTIVATION: &str =
    "double activation of prepared token";
pub(crate) const INVARIANT_DUPLICATE_TEARDOWN: &str =
    "duplicate teardown of active receipt";
pub(crate) const INVARIANT_TEARDOWN_OF_PREPARED: &str =
    "teardown of a prepared token";

/// States in the candidate lifecycle for transition validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CandidatePhase {
    /// Candidate has been created with CPU-prepared DTOs (extraction done).
    CpuPrepared,
    /// Renderer upload has been started (async pending).
    RendererPending,
    /// Renderer upload is complete; the prepared mount is ready.
    RendererReady,
    /// All validations passed (bridges, scene preflight, publication checks).
    ValidatedForScene,
    /// Candidate has been consumed into an ActiveBspMount; state is transferred.
    Consumed,
    /// Preparation or validation failed; candidate may be rolled back.
    Failed,
    /// Candidate has been rolled back; all owned resources released.
    RolledBack,
}

impl fmt::Display for BridgePhase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BridgePhase::Prepare => write!(f, "prepare"),
            BridgePhase::Validate => write!(f, "validate"),
            BridgePhase::Activate => write!(f, "activate"),
            BridgePhase::Teardown => write!(f, "teardown"),
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
            BspRuntimeError::InvalidCandidateTransition {
                current,
                attempted,
                detail,
            } => {
                write!(
                    f,
                    "invalid candidate transition from {:?} to {:?}: {}",
                    current, attempted, detail
                )
            }
            BspRuntimeError::StaleRendererCompletion {
                candidate_generation,
                current_generation,
            } => {
                write!(
                    f,
                    "stale renderer completion: candidate generation {} does not match current {}",
                    candidate_generation, current_generation
                )
            }
            BspRuntimeError::DuplicateReadyLease { generation } => {
                write!(
                    f,
                    "duplicate renderer-ready lease for generation {}: sent to retirement",
                    generation
                )
            }
            BspRuntimeError::RetirementHandoffFailed { reason } => {
                write!(f, "renderer retirement handoff failed: {}", reason)
            }
            BspRuntimeError::CommitContractViolated { detail } => {
                write!(f, "commit contract violated: {}", detail)
            }
            BspRuntimeError::TeardownQuarantined {
                bridge_name,
                message,
                unattempted,
            } => {
                write!(
                    f,
                    "bridge '{}' teardown quarantined: {} ({} unattempted)",
                    bridge_name, message, unattempted
                )
            }
            BspRuntimeError::BridgeInvariantViolated {
                bridge_name,
                phase,
                detail,
            } => {
                write!(
                    f,
                    "bridge '{}' invariant violated during {}: {}",
                    bridge_name, phase, detail
                )
            }
        }
    }
}

impl std::error::Error for BspRuntimeError {}
