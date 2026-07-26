//! BSP candidate: self-contained preparation state for one generation.
//!
//! A candidate owns every staged resource for a single prepare cycle:
//! extracted DTOs, cache identity, renderer mount lease, bridge tokens,
//! source-link payload, external asset readiness, validation status,
//! and diagnostics. The coordinator holds at most one candidate at a time;
//! a new prepare atomically replaces the previous candidate.
//!
//! The candidate is the unit of idempotent rollback: releasing a candidate
//! frees all associated staged resources without touching the active world.
//!
//! # Phase 05: Typed State Machine
//!
//! Legal transitions:
//! ```text
//! CpuPrepared ──▶ RendererPending ──▶ RendererReady ──▶ ValidatedForScene ──▶ Consumed
//!      │                                        ▲
//!      └────────────────────────────────────────┘ (synchronous upload skip)
//!
//! Any non-terminal state ──▶ Failed ──▶ RolledBack
//! ```
//!
//! Duplicate, stale, and out-of-order transitions are rejected with typed
//! errors. A stale renderer completion is sent to retirement, not accepted.

use crate::bridge::BridgeToken;
use crate::cache::CacheIdentity;
use crate::error::{CandidatePhase, BspRuntimeError};
use crate::source_link::BspSourceLink;

use bsp::extract::ExtractedBsp;
use renderer::api::bsp::PreparedBspMount;
use renderer::api::PointLight;

// ── Renderer Lease ────────────────────────────────────────────────────

/// The state of a renderer mount lease within a candidate.
pub enum RendererLease {
    /// No upload has been started.
    NotStarted,
    /// Upload is in progress; the generation tag guards against cancellation.
    Pending { generation: u64 },
    /// Upload complete; the prepared mount is ready to attach to a scene.
    Ready(PreparedBspMount),
    /// Upload failed. The error message is preserved for diagnostics.
    Failed { reason: String },
}

impl std::fmt::Debug for RendererLease {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotStarted => write!(f, "NotStarted"),
            Self::Pending { generation } => f
                .debug_struct("Pending")
                .field("generation", generation)
                .finish(),
            Self::Ready(_) => write!(f, "Ready(PreparedBspMount)"),
            Self::Failed { reason } => f.debug_struct("Failed").field("reason", reason).finish(),
        }
    }
}

// ── Candidate State ───────────────────────────────────────────────────

/// The state of a candidate in its lifecycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CandidateState {
    /// Created with CPU-prepared DTOs; extraction complete.
    CpuPrepared,
    /// Renderer upload has been started (async path).
    RendererPending,
    /// Renderer upload complete; mount ready.
    RendererReady,
    /// All validations passed, ready for commit.
    ValidatedForScene,
    /// Consumed into an ActiveBspMount.
    Consumed,
    /// Preparation or validation failed.
    Failed,
    /// Rolled back; resources released.
    RolledBack,
}

// ── Prevalidated Point Light ──────────────────────────────────────────

/// Prevalidated scene point light payload for a BSP candidate.
#[derive(Debug, Clone)]
pub struct CandidatePointLight {
    /// Source entity index from the BSP entity lump.
    pub entity_index: u32,
    /// Renderer point-light payload prepared before commit.
    pub light: PointLight,
}

// ── Active BSP Mount ──────────────────────────────────────────────────

/// The published (committed) BSP mount state.
///
/// After a candidate is consumed by commit, its resources move into an
/// `ActiveBspMount`. The active mount owns the extracted DTOs, source-link
/// metadata, cache identity, point-light IDs, published source-link JSON,
/// and the opaque renderer mount lease.
///
/// When replaced or unloaded, the active mount's renderer lease is handed
/// off to the renderer for fence-based retirement. `bsp_runtime` records
/// the handoff identity but never reads raw GPU handles or computes
/// submission serials.
pub struct ActiveBspMount {
    /// Active extracted BSP DTOs.
    pub extracted: ExtractedBsp,
    /// Active source-link metadata.
    pub source_link: BspSourceLink,
    /// Published source-link JSON (already in the scene).
    pub source_link_json: serde_json::Value,
    /// Cache identity of the active mount.
    pub cache_identity: CacheIdentity,
    /// Point-light IDs created for this mount in the scene.
    pub light_ids: Vec<renderer::api::PointLightId>,
    /// Opaque renderer mount lease (held by the scene).
    pub renderer_lease: PreparedBspMount,
    /// Human-readable source identity for diagnostics.
    pub source_identity: String,
    /// Generation at which this mount was committed.
    pub committed_generation: u64,
}

impl std::fmt::Debug for ActiveBspMount {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ActiveBspMount")
            .field("source_identity", &self.source_identity)
            .field("committed_generation", &self.committed_generation)
            .field("light_ids", &self.light_ids)
            .field("renderer_lease", &"<PreparedBspMount>")
            .finish()
    }
}

impl ActiveBspMount {
    /// Create a new active mount from a consumed candidate.
    pub fn from_candidate(
        candidate: BspCandidate,
        light_ids: Vec<renderer::api::PointLightId>,
        committed_generation: u64,
    ) -> Self {
        let mount = match candidate.renderer_lease {
            RendererLease::Ready(m) => m,
            _ => {
                // This is a contract violation — caller must ensure the candidate
                // is in RendererReady state before consuming.
                panic!("ActiveBspMount::from_candidate requires a renderer-ready candidate");
            }
        };
        Self {
            extracted: candidate.extracted,
            source_link: candidate.source_link,
            source_link_json: candidate.source_link_json,
            cache_identity: candidate.cache_identity,
            light_ids,
            renderer_lease: mount,
            source_identity: candidate.source_identity,
            committed_generation,
        }
    }
}

// ── BspCandidate ──────────────────────────────────────────────────────

/// A self-contained BSP preparation candidate.
///
/// All preparation work — parsing, extraction, upload, bridge preparation,
/// validation — is performed on the candidate. The commit step is a pure
/// publish: it moves the candidate's ready resources into the active world
/// without performing any new work.
///
/// # Phase 05: State Machine
///
/// The candidate enforces a strict state machine. All transitions are typed
/// and consume-once. Once consumed, the candidate cannot be reused.
#[derive(Debug)]
pub struct BspCandidate {
    /// Current state in the candidate lifecycle.
    pub state: CandidateState,

    /// The generation this candidate was prepared for.
    pub generation: u64,
    /// Human-readable package identity.
    pub source_identity: String,
    /// Parsed and extracted neutral DTOs.
    pub extracted: ExtractedBsp,
    /// Cache identity computed from extraction inputs.
    pub cache_identity: CacheIdentity,
    /// Renderer mount lease state.
    pub renderer_lease: RendererLease,
    /// Bridge tokens from prepare, indexed by bridge position.
    pub bridge_tokens: Vec<Option<BridgeToken>>,
    /// Typed source-link payload.
    pub source_link: BspSourceLink,
    /// Scene-owned JSON source-link payload, serialized before commit.
    pub source_link_json: serde_json::Value,
    /// Prevalidated point lights to publish with the mount.
    pub point_lights: Vec<CandidatePointLight>,
    /// Diagnostics collected during preparation.
    pub diagnostics: Vec<String>,
    /// Whether the target was occupied when prepare started.
    pub was_occupied: bool,
    /// Face count.
    pub face_count: usize,
    /// Entity count.
    pub entity_count: usize,
    /// Light entity count.
    pub light_count: usize,
    /// Render batch count.
    pub batch_count: usize,
    /// Whether PVS data is available.
    pub has_pvs: bool,
    /// Phase 03 import provenance for diagnostics (not used for path resolution).
    pub import_provenance: Option<ImportProvenanceRecord>,
    /// Number of times a retirement handoff was recorded for this candidate's
    /// replaced active mount (diagnostics only).
    pub retirement_handoff_count: u64,
}

/// Phase 03 import provenance retained for diagnostics.
///
/// These labels provide traceability but must never be used to re-resolve
/// paths, substitute defaults, or alter strictness after extraction.
#[derive(Debug, Clone)]
pub struct ImportProvenanceRecord {
    /// Import route: `"package"` or `"direct"`.
    pub route: String,
    /// Whether strict mode was active.
    pub strict: bool,
    /// The logical asset identity from the resolver.
    pub asset_id: String,
}

impl BspCandidate {
    /// Create a new unvalidated candidate from extracted BSP data.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        generation: u64,
        source_identity: String,
        extracted: ExtractedBsp,
        cache_identity: CacheIdentity,
        source_link: BspSourceLink,
        source_link_json: serde_json::Value,
        point_lights: Vec<CandidatePointLight>,
        bridge_tokens: Vec<Option<BridgeToken>>,
        was_occupied: bool,
    ) -> Self {
        let face_count = extracted.face_geometries.len();
        let entity_count = extracted.entity_descriptors.len();
        let light_count = extracted.light_descriptors.len();
        let batch_count = extracted.render_batches.len();
        let has_pvs = extracted.has_pvs;

        Self {
            state: CandidateState::CpuPrepared,
            generation,
            source_identity,
            extracted,
            cache_identity,
            renderer_lease: RendererLease::NotStarted,
            bridge_tokens,
            source_link,
            source_link_json,
            point_lights,
            diagnostics: Vec::new(),
            was_occupied,
            face_count,
            entity_count,
            light_count,
            batch_count,
            has_pvs,
            import_provenance: None,
            retirement_handoff_count: 0,
        }
    }

    // ── State Queries ──────────────────────────────────────────────

    /// Returns true if the candidate is in a terminal state (Consumed or RolledBack).
    pub fn is_terminal(&self) -> bool {
        matches!(
            self.state,
            CandidateState::Consumed | CandidateState::RolledBack
        )
    }

    /// Returns true if the candidate is ready for commit.
    pub fn is_commit_ready(&self) -> bool {
        self.state == CandidateState::ValidatedForScene
    }

    /// Returns true if the renderer lease is ready.
    pub fn is_renderer_ready(&self) -> bool {
        matches!(self.renderer_lease, RendererLease::Ready(_))
    }

    /// Returns true if the candidate is in a state that can accept a
    /// renderer completion.
    pub fn can_accept_renderer_completion(&self) -> bool {
        matches!(
            self.state,
            CandidateState::CpuPrepared | CandidateState::RendererPending
        )
    }

    // ── Transition: Start Renderer Upload ──────────────────────────

    /// Begin a renderer upload. Transitions CpuPrepared → RendererPending.
    ///
    /// Idempotent if already pending. Rejected if already ready, failed, or consumed.
    pub fn transition_to_renderer_pending(
        &mut self,
        current_generation: u64,
    ) -> Result<(), BspRuntimeError> {
        // Validate generation
        if self.generation != current_generation {
            return Err(BspRuntimeError::StaleGeneration {
                expected: self.generation,
                current: current_generation,
            });
        }

        match self.state {
            CandidateState::CpuPrepared => {
                self.state = CandidateState::RendererPending;
                self.renderer_lease = RendererLease::Pending {
                    generation: self.generation,
                };
                Ok(())
            }
            CandidateState::RendererPending => {
                // Idempotent: already pending.
                Ok(())
            }
            CandidateState::RendererReady => {
                // Already ready — no-op (caller can use sync path).
                Ok(())
            }
            CandidateState::ValidatedForScene | CandidateState::Consumed => {
                Err(BspRuntimeError::InvalidCandidateTransition {
                    current: CandidatePhase::from(self.state),
                    attempted: CandidatePhase::RendererPending,
                    detail: "candidate already validated or consumed".to_string(),
                })
            }
            CandidateState::Failed => Err(BspRuntimeError::InvalidCandidateTransition {
                current: CandidatePhase::from(self.state),
                attempted: CandidatePhase::RendererPending,
                detail: "candidate has failed".to_string(),
            }),
            CandidateState::RolledBack => Err(BspRuntimeError::InvalidCandidateTransition {
                current: CandidatePhase::from(self.state),
                attempted: CandidatePhase::RendererPending,
                detail: "candidate has been rolled back".to_string(),
            }),
        }
    }

    // ── Transition: Mark Renderer Upload Failed ────────────────────

    /// Mark the renderer upload as failed. Transitions any non-terminal state → Failed.
    pub fn transition_to_renderer_failed(
        &mut self,
        current_generation: u64,
        reason: String,
    ) -> Result<(), BspRuntimeError> {
        if self.generation != current_generation {
            return Err(BspRuntimeError::StaleGeneration {
                expected: self.generation,
                current: current_generation,
            });
        }

        match self.state {
            CandidateState::CpuPrepared
            | CandidateState::RendererPending
            | CandidateState::RendererReady => {
                self.state = CandidateState::Failed;
                self.renderer_lease = RendererLease::Failed { reason };
                Ok(())
            }
            CandidateState::ValidatedForScene | CandidateState::Consumed => {
                Err(BspRuntimeError::InvalidCandidateTransition {
                    current: CandidatePhase::from(self.state),
                    attempted: CandidatePhase::Failed,
                    detail: "candidate already validated or consumed".to_string(),
                })
            }
            CandidateState::Failed => {
                // Idempotent: already failed.
                self.renderer_lease = RendererLease::Failed { reason };
                Ok(())
            }
            CandidateState::RolledBack => Err(BspRuntimeError::InvalidCandidateTransition {
                current: CandidatePhase::from(self.state),
                attempted: CandidatePhase::Failed,
                detail: "candidate has been rolled back".to_string(),
            }),
        }
    }

    // ── Transition: Set Renderer Ready ─────────────────────────────

    /// Transition the candidate to RendererReady.
    ///
    /// Accepts the completed [`PreparedBspMount`]. Transitions from:
    /// - `CpuPrepared` → `RendererReady` (synchronous path)
    /// - `RendererPending` → `RendererReady` (async completion)
    ///
    /// **Duplicate lease handling**: If the candidate is already `RendererReady`,
    /// the incoming lease is NOT accepted. The caller must route it to
    /// renderer cancellation/retirement. Returns `DuplicateReadyLease`.
    pub fn transition_to_renderer_ready(
        &mut self,
        current_generation: u64,
        mount: PreparedBspMount,
    ) -> Result<(), BspRuntimeError> {
        // Validate generation
        if self.generation != current_generation {
            // Stale completion: the lease must be sent to retirement by the caller.
            return Err(BspRuntimeError::StaleRendererCompletion {
                candidate_generation: self.generation,
                current_generation,
            });
        }

        match self.state {
            CandidateState::CpuPrepared | CandidateState::RendererPending => {
                self.state = CandidateState::RendererReady;
                self.renderer_lease = RendererLease::Ready(mount);
                Ok(())
            }
            CandidateState::RendererReady => {
                // Duplicate lease — refuse it. Caller must retire it.
                Err(BspRuntimeError::DuplicateReadyLease {
                    generation: self.generation,
                })
            }
            CandidateState::ValidatedForScene | CandidateState::Consumed => {
                Err(BspRuntimeError::InvalidCandidateTransition {
                    current: CandidatePhase::from(self.state),
                    attempted: CandidatePhase::RendererReady,
                    detail: "candidate already validated or consumed".to_string(),
                })
            }
            CandidateState::Failed => Err(BspRuntimeError::InvalidCandidateTransition {
                current: CandidatePhase::from(self.state),
                attempted: CandidatePhase::RendererReady,
                detail: "candidate has failed".to_string(),
            }),
            CandidateState::RolledBack => Err(BspRuntimeError::InvalidCandidateTransition {
                current: CandidatePhase::from(self.state),
                attempted: CandidatePhase::RendererReady,
                detail: "candidate has been rolled back".to_string(),
            }),
        }
    }

    // ── Transition: Validate For Scene ─────────────────────────────

    /// Transition the candidate to ValidatedForScene.
    ///
    /// Requires `RendererReady`. All validation (bridges, scene preflight,
    /// publication checks) must complete before calling this.
    pub fn transition_to_validated_for_scene(
        &mut self,
        current_generation: u64,
    ) -> Result<(), BspRuntimeError> {
        if self.generation != current_generation {
            return Err(BspRuntimeError::StaleGeneration {
                expected: self.generation,
                current: current_generation,
            });
        }

        match self.state {
            CandidateState::RendererReady => {
                self.state = CandidateState::ValidatedForScene;
                Ok(())
            }
            CandidateState::ValidatedForScene => {
                // Idempotent: already validated.
                Ok(())
            }
            CandidateState::CpuPrepared | CandidateState::RendererPending => {
                Err(BspRuntimeError::InvalidCandidateTransition {
                    current: CandidatePhase::from(self.state),
                    attempted: CandidatePhase::ValidatedForScene,
                    detail: "renderer mount must be ready before validation".to_string(),
                })
            }
            CandidateState::Consumed => Err(BspRuntimeError::InvalidCandidateTransition {
                current: CandidatePhase::from(self.state),
                attempted: CandidatePhase::ValidatedForScene,
                detail: "candidate already consumed".to_string(),
            }),
            CandidateState::Failed => Err(BspRuntimeError::InvalidCandidateTransition {
                current: CandidatePhase::from(self.state),
                attempted: CandidatePhase::ValidatedForScene,
                detail: "candidate has failed".to_string(),
            }),
            CandidateState::RolledBack => Err(BspRuntimeError::InvalidCandidateTransition {
                current: CandidatePhase::from(self.state),
                attempted: CandidatePhase::ValidatedForScene,
                detail: "candidate has been rolled back".to_string(),
            }),
        }
    }

    // ── Transition: Mark Failed ────────────────────────────────────

    /// Mark the candidate as failed (generic failure path).
    pub fn transition_to_failed(
        &mut self,
        current_generation: u64,
        reason: String,
    ) -> Result<(), BspRuntimeError> {
        if self.generation != current_generation {
            return Err(BspRuntimeError::StaleGeneration {
                expected: self.generation,
                current: current_generation,
            });
        }

        match self.state {
            CandidateState::CpuPrepared
            | CandidateState::RendererPending
            | CandidateState::RendererReady
            | CandidateState::ValidatedForScene => {
                self.state = CandidateState::Failed;
                self.diagnostics.push(reason);
                Ok(())
            }
            CandidateState::Consumed => Err(BspRuntimeError::InvalidCandidateTransition {
                current: CandidatePhase::from(self.state),
                attempted: CandidatePhase::Failed,
                detail: "candidate already consumed".to_string(),
            }),
            CandidateState::Failed => {
                // Idempotent: append diagnostics.
                self.diagnostics.push(reason);
                Ok(())
            }
            CandidateState::RolledBack => Err(BspRuntimeError::InvalidCandidateTransition {
                current: CandidatePhase::from(self.state),
                attempted: CandidatePhase::Failed,
                detail: "candidate has been rolled back".to_string(),
            }),
        }
    }

    // ── Transition: Consume (Commit) ───────────────────────────────

    /// Consume the candidate, transferring ownership into an `ActiveBspMount`.
    ///
    /// Requires `ValidatedForScene`. The candidate is consumed exactly once;
    /// after this call, the candidate is in `Consumed` state and all owned
    /// resources have been transferred out.
    pub fn consume_into_active(
        mut self,
        current_generation: u64,
        light_ids: Vec<renderer::api::PointLightId>,
    ) -> Result<ActiveBspMount, BspRuntimeError> {
        if self.generation != current_generation {
            return Err(BspRuntimeError::StaleGeneration {
                expected: self.generation,
                current: current_generation,
            });
        }

        match self.state {
            CandidateState::ValidatedForScene => {
                // Take the ready mount out of the candidate before it moves.
                let mount = match std::mem::replace(
                    &mut self.renderer_lease,
                    RendererLease::NotStarted,
                ) {
                    RendererLease::Ready(m) => m,
                    other => {
                        self.renderer_lease = other;
                        return Err(BspRuntimeError::CommitContractViolated {
                            detail: "renderer mount not ready at consume time".to_string(),
                        });
                    }
                };

                self.state = CandidateState::Consumed;

                Ok(ActiveBspMount {
                    extracted: self.extracted,
                    source_link: self.source_link,
                    source_link_json: self.source_link_json,
                    cache_identity: self.cache_identity,
                    light_ids,
                    renderer_lease: mount,
                    source_identity: self.source_identity,
                    committed_generation: self.generation,
                })
            }
            CandidateState::CpuPrepared | CandidateState::RendererPending => {
                Err(BspRuntimeError::InvalidCandidateTransition {
                    current: CandidatePhase::from(self.state),
                    attempted: CandidatePhase::Consumed,
                    detail: "candidate must be validated before commit".to_string(),
                })
            }
            CandidateState::RendererReady => Err(BspRuntimeError::InvalidCandidateTransition {
                current: CandidatePhase::from(self.state),
                attempted: CandidatePhase::Consumed,
                detail: "candidate must be validated for scene before commit".to_string(),
            }),
            CandidateState::Consumed => Err(BspRuntimeError::InvalidCandidateTransition {
                current: CandidatePhase::from(self.state),
                attempted: CandidatePhase::Consumed,
                detail: "candidate already consumed".to_string(),
            }),
            CandidateState::Failed | CandidateState::RolledBack => {
                Err(BspRuntimeError::InvalidCandidateTransition {
                    current: CandidatePhase::from(self.state),
                    attempted: CandidatePhase::Consumed,
                    detail: "cannot consume a failed or rolled-back candidate".to_string(),
                })
            }
        }
    }

    // ── Transition: Rollback ───────────────────────────────────────

    /// Roll back the candidate. Releases staged resources.
    ///
    /// Idempotent: subsequent calls are no-ops. The candidate's bridge tokens
    /// and renderer lease are yielded to the caller for cleanup (bridge rollback
    /// and renderer retirement, respectively).
    ///
    /// Returns the bridge tokens that need rollback and any ready renderer mount
    /// that needs retirement.
    pub fn rollback(
        &mut self,
    ) -> (
        Vec<Option<BridgeToken>>,
        Option<PreparedBspMount>,
    ) {
        match self.state {
            CandidateState::Consumed | CandidateState::RolledBack => {
                // Terminal states: nothing to do.
                (Vec::new(), None)
            }
            _ => {
                self.state = CandidateState::RolledBack;
                let tokens = std::mem::take(&mut self.bridge_tokens);
                let mount = match std::mem::replace(
                    &mut self.renderer_lease,
                    RendererLease::NotStarted,
                ) {
                    RendererLease::Ready(m) => Some(m),
                    _ => None,
                };
                (tokens, mount)
            }
        }
    }

    // ── Deprecated Compatibility Methods ───────────────────────────

    /// Begin a renderer upload. Legacy path — prefer transition_to_renderer_pending.
    #[doc(hidden)]
    pub fn start_renderer_upload(&mut self) -> Result<(), BspRuntimeError> {
        match self.renderer_lease {
            RendererLease::NotStarted => {
                self.renderer_lease = RendererLease::Pending {
                    generation: self.generation,
                };
                self.state = CandidateState::RendererPending;
                Ok(())
            }
            RendererLease::Pending { .. } => Ok(()),
            RendererLease::Ready(_) => Err(BspRuntimeError::BridgeFailure {
                bridge_name: "coordinator".to_string(),
                phase: crate::error::BridgePhase::Prepare,
                message: "renderer lease already ready".to_string(),
            }),
            RendererLease::Failed { ref reason } => Err(BspRuntimeError::SourceUnavailable {
                reason: format!("renderer upload previously failed: {reason}"),
            }),
        }
    }

    /// Mark the renderer upload as failed. Legacy path.
    #[doc(hidden)]
    pub fn fail_renderer_upload(&mut self, reason: String) {
        self.renderer_lease = RendererLease::Failed { reason };
        self.state = CandidateState::Failed;
    }

    /// Transition the renderer lease to Ready. Legacy path.
    #[doc(hidden)]
    pub fn set_renderer_ready(
        &mut self,
        mount: PreparedBspMount,
    ) -> Result<(), BspRuntimeError> {
        match &self.renderer_lease {
            RendererLease::Pending { generation } if *generation == self.generation => {
                self.renderer_lease = RendererLease::Ready(mount);
                self.state = CandidateState::RendererReady;
                Ok(())
            }
            RendererLease::Pending { generation } => {
                let old_gen = *generation;
                self.renderer_lease = RendererLease::NotStarted;
                Err(BspRuntimeError::StaleGeneration {
                    expected: old_gen,
                    current: self.generation,
                })
            }
            RendererLease::NotStarted => {
                self.renderer_lease = RendererLease::Ready(mount);
                self.state = CandidateState::RendererReady;
                Ok(())
            }
            RendererLease::Ready(_) => {
                self.renderer_lease = RendererLease::Ready(mount);
                Ok(())
            }
            RendererLease::Failed { ref reason } => Err(BspRuntimeError::SourceUnavailable {
                reason: format!("renderer upload failed: {reason}"),
            }),
        }
    }

    /// Take the ready mount, leaving the lease in NotStarted.
    #[doc(hidden)]
    pub fn take_ready_mount(&mut self) -> Result<PreparedBspMount, BspRuntimeError> {
        match std::mem::replace(&mut self.renderer_lease, RendererLease::NotStarted) {
            RendererLease::Ready(mount) => Ok(mount),
            other => {
                self.renderer_lease = other;
                Err(BspRuntimeError::BridgeFailure {
                    bridge_name: "coordinator".to_string(),
                    phase: crate::error::BridgePhase::Commit,
                    message: "renderer mount not ready".to_string(),
                })
            }
        }
    }

    /// Mark the candidate as validated. Legacy path.
    #[doc(hidden)]
    pub fn mark_validated(&mut self) {
        if self.state == CandidateState::RendererReady {
            self.state = CandidateState::ValidatedForScene;
        }
    }

    /// Mark scene publication preflight as complete. Legacy path.
    #[doc(hidden)]
    pub fn mark_publication_validated(&mut self) {
        // No separate flag — validation implies publication readiness.
        if self.state == CandidateState::RendererReady {
            self.state = CandidateState::ValidatedForScene;
        }
    }

    /// Returns true if the candidate has passed validation. Legacy path.
    #[doc(hidden)]
    pub fn is_validated(&self) -> bool {
        self.state == CandidateState::ValidatedForScene
    }

    /// Returns true if scene publication has been preflighted. Legacy path.
    #[doc(hidden)]
    pub fn is_publication_validated(&self) -> bool {
        self.state == CandidateState::ValidatedForScene
    }
}

// ── CandidatePhase From CandidateState ────────────────────────────────

impl From<CandidateState> for CandidatePhase {
    fn from(state: CandidateState) -> Self {
        match state {
            CandidateState::CpuPrepared => CandidatePhase::CpuPrepared,
            CandidateState::RendererPending => CandidatePhase::RendererPending,
            CandidateState::RendererReady => CandidatePhase::RendererReady,
            CandidateState::ValidatedForScene => CandidatePhase::ValidatedForScene,
            CandidateState::Consumed => CandidatePhase::Consumed,
            CandidateState::Failed => CandidatePhase::Failed,
            CandidateState::RolledBack => CandidatePhase::RolledBack,
        }
    }
}
