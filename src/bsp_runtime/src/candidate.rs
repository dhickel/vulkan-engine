//! BSP candidate: self-contained preparation state for one generation.
//!
//! A candidate owns every staged resource for a single prepare cycle:
//! extracted DTOs, cache identity, renderer mount lease (PreparedBspMount),
//! prepared bridge tokens, source-link payload, prevalidated light payload
//! and scene reservation, generation, and opaque renderer attach/replacement
//! permit. The coordinator holds at most one candidate at a time; a new
//! prepare atomically replaces the previous candidate.
//!
//! The candidate is the unit of idempotent rollback: releasing a candidate
//! frees all associated staged resources without touching the active world.
//!
//! # Phase 06: Move-Only Ownership
//!
//! The candidate owns B's ready lease (PreparedBspMount), prepared bridge
//! receipts, pre-serialized source-link payload, validated light payloads
//! and scene reservation, and an opaque single-use replacement/attach permit.
//! Scene/renderer owns the published lease; the renderer retirement queue
//! owns it only after typed acknowledgement.
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

use crate::bridge::{ActiveBridgeReceipts, PreparedBridgeToken};
use crate::cache::CacheIdentity;
use crate::error::{BspRuntimeError, CandidatePhase};
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

/// The published (committed) BSP mount state — coordinator metadata.
///
/// After a candidate is consumed by commit, its resources move into an
/// `ActiveBspMount`. The active mount owns the extracted DTOs, source-link
/// metadata, cache identity, point-light IDs, published source-link JSON,
/// and active bridge receipts. The opaque renderer lease (PublishedBspMount)
/// lives in `Scene`, not in this record.
///
/// On replacement or unload, `Scene::retire_bsp_mount` returns an opaque
/// scene-detachment receipt. The coordinator passes it to the renderer for
/// fence-aware GPU retirement and receives a typed acknowledgement.
/// `bsp_runtime` never retains a cloned mount or reads raw GPU handles.
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
    /// Human-readable source identity for diagnostics.
    pub source_identity: String,
    /// Generation at which this mount was committed.
    pub committed_generation: u64,
    /// Active bridge receipts for all committed bridges.
    pub active_bridge_receipts: ActiveBridgeReceipts,
}

impl std::fmt::Debug for ActiveBspMount {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ActiveBspMount")
            .field("source_identity", &self.source_identity)
            .field("committed_generation", &self.committed_generation)
            .field("light_ids", &self.light_ids)
            .finish()
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
    /// Prepared bridge tokens from prepare, indexed by bridge position.
    pub prepared_tokens: Vec<Option<PreparedBridgeToken>>,
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
    /// Reserved diagnostic count for scene-detachment handoffs associated
    /// with this candidate (not evidence of GPU queueing).
    pub retirement_handoff_count: u64,
    /// Opaque renderer attach/replacement permit, set during validation
    /// and consumed by commit. Proves preflighted publication readiness.
    pub(crate) attach_permit: Option<RendererAttachPermit>,
}

// ── Renderer Attach/Replacement Permit ────────────────────────────────

/// Opaque single-use permit binding a candidate's prepared lease to an
/// attach or replacement operation.
///
/// Created during validation; consumed during commit. The permit proves
/// the coordinator preflighted every fallible publication check and owns
/// the exact lease at the correct generation.
#[derive(Debug)]
pub struct RendererAttachPermit {
    pub(crate) generation: u64,
    pub(crate) is_replacement: bool,
}

// ── Unload Permit ─────────────────────────────────────────────────────

/// Opaque single-use permit authorizing one active-unload operation.
///
/// Created during unload preflight; consumed when finalizing the unload.
/// The permit binds the exact active lease and scene-clear to one
/// generation. Dropping without finalization retains custody.
#[derive(Debug)]
pub struct UnloadPermit {
    pub(crate) generation: u64,
    pub(crate) source_identity: String,
}

// ── Import Provenance ─────────────────────────────────────────────────

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
        prepared_tokens: Vec<Option<PreparedBridgeToken>>,
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
            prepared_tokens,
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
            attach_permit: None,
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
            CandidateState::CpuPrepared | CandidateState::RendererPending => {
                self.state = CandidateState::Failed;
                self.renderer_lease = RendererLease::Failed { reason };
                Ok(())
            }
            CandidateState::RendererReady => Err(BspRuntimeError::InvalidCandidateTransition {
                current: CandidatePhase::RendererReady,
                attempted: CandidatePhase::Failed,
                detail: "a ready renderer lease must be retired through rollback".to_string(),
            }),
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
    ) -> Result<(), (BspRuntimeError, PreparedBspMount)> {
        // Every rejected completion returns its lease to the caller so the
        // coordinator can detach it through the opaque renderer/Scene facade
        // exactly once instead of letting Rust drop it at the error boundary.
        if self.generation != current_generation {
            return Err((
                BspRuntimeError::StaleRendererCompletion {
                    candidate_generation: self.generation,
                    current_generation,
                },
                mount,
            ));
        }

        match self.state {
            CandidateState::CpuPrepared | CandidateState::RendererPending => {
                self.state = CandidateState::RendererReady;
                self.renderer_lease = RendererLease::Ready(mount);
                Ok(())
            }
            CandidateState::RendererReady => Err((
                BspRuntimeError::DuplicateReadyLease {
                    generation: self.generation,
                },
                mount,
            )),
            CandidateState::ValidatedForScene | CandidateState::Consumed => Err((
                BspRuntimeError::InvalidCandidateTransition {
                    current: CandidatePhase::from(self.state),
                    attempted: CandidatePhase::RendererReady,
                    detail: "candidate already validated or consumed".to_string(),
                },
                mount,
            )),
            CandidateState::Failed => Err((
                BspRuntimeError::InvalidCandidateTransition {
                    current: CandidatePhase::from(self.state),
                    attempted: CandidatePhase::RendererReady,
                    detail: "candidate has failed".to_string(),
                },
                mount,
            )),
            CandidateState::RolledBack => Err((
                BspRuntimeError::InvalidCandidateTransition {
                    current: CandidatePhase::from(self.state),
                    attempted: CandidatePhase::RendererReady,
                    detail: "candidate has been rolled back".to_string(),
                },
                mount,
            )),
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
        active_bridge_receipts: ActiveBridgeReceipts,
    ) -> Result<(ActiveBspMount, PreparedBspMount), (BspRuntimeError, BspCandidate)> {
        if self.generation != current_generation {
            let error = BspRuntimeError::StaleGeneration {
                expected: self.generation,
                current: current_generation,
            };
            return Err((error, self));
        }

        match self.state {
            CandidateState::ValidatedForScene => {
                // Take the ready mount out of the candidate before it moves.
                let mount =
                    match std::mem::replace(&mut self.renderer_lease, RendererLease::NotStarted) {
                        RendererLease::Ready(mount) => mount,
                        other => {
                            self.renderer_lease = other;
                            let error = BspRuntimeError::CommitContractViolated {
                                detail: "renderer mount not ready at consume time".to_string(),
                            };
                            return Err((error, self));
                        }
                    };

                self.state = CandidateState::Consumed;

                Ok((
                    ActiveBspMount {
                        extracted: self.extracted,
                        source_link: self.source_link,
                        source_link_json: self.source_link_json,
                        cache_identity: self.cache_identity,
                        light_ids,
                        source_identity: self.source_identity,
                        committed_generation: self.generation,
                        active_bridge_receipts,
                    },
                    mount,
                ))
            }
            CandidateState::CpuPrepared | CandidateState::RendererPending => {
                let error = BspRuntimeError::InvalidCandidateTransition {
                    current: CandidatePhase::from(self.state),
                    attempted: CandidatePhase::Consumed,
                    detail: "candidate must be validated before commit".to_string(),
                };
                Err((error, self))
            }
            CandidateState::RendererReady => {
                let error = BspRuntimeError::InvalidCandidateTransition {
                    current: CandidatePhase::from(self.state),
                    attempted: CandidatePhase::Consumed,
                    detail: "candidate must be validated for scene before commit".to_string(),
                };
                Err((error, self))
            }
            CandidateState::Consumed => {
                let error = BspRuntimeError::InvalidCandidateTransition {
                    current: CandidatePhase::from(self.state),
                    attempted: CandidatePhase::Consumed,
                    detail: "candidate already consumed".to_string(),
                };
                Err((error, self))
            }
            CandidateState::Failed | CandidateState::RolledBack => {
                let error = BspRuntimeError::InvalidCandidateTransition {
                    current: CandidatePhase::from(self.state),
                    attempted: CandidatePhase::Consumed,
                    detail: "cannot consume a failed or rolled-back candidate".to_string(),
                };
                Err((error, self))
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
    /// Returns the prepared bridge tokens that need rollback and any ready
    /// renderer mount that needs opaque scene detachment.
    pub fn rollback(&mut self) -> (Vec<Option<PreparedBridgeToken>>, Option<PreparedBspMount>) {
        match self.state {
            CandidateState::Consumed | CandidateState::RolledBack => {
                // Terminal states: nothing to do.
                (Vec::new(), None)
            }
            _ => {
                self.state = CandidateState::RolledBack;
                let tokens = std::mem::take(&mut self.prepared_tokens);
                let mount =
                    match std::mem::replace(&mut self.renderer_lease, RendererLease::NotStarted) {
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
        if let RendererLease::Ready(mount) =
            std::mem::replace(&mut self.renderer_lease, RendererLease::Failed { reason })
        {
            let _retired = mount.retire();
        }
        self.state = CandidateState::Failed;
    }

    /// Transition the renderer lease to Ready. Legacy path.
    #[doc(hidden)]
    pub fn set_renderer_ready(&mut self, mount: PreparedBspMount) -> Result<(), BspRuntimeError> {
        match self.transition_to_renderer_ready(self.generation, mount) {
            Ok(()) => Ok(()),
            Err((error, mount)) => {
                let _retired = mount.retire();
                Err(error)
            }
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
