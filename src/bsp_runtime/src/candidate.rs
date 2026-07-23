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

use crate::bridge::BridgeToken;
use crate::cache::CacheIdentity;
use crate::error::BspRuntimeError;
use crate::source_link::BspSourceLink;

use bsp::extract::ExtractedBsp;
use renderer::api::bsp::PreparedBspMount;
use renderer::api::PointLight;

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

/// The validation state of a candidate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CandidateValidation {
    /// Not yet validated.
    Unvalidated,
    /// Bridges have been validated and are ready for commit.
    Validated,
    /// Validation failed.
    Failed,
}

/// Prevalidated scene point light payload for a BSP candidate.
#[derive(Debug, Clone)]
pub struct CandidatePointLight {
    /// Source entity index from the BSP entity lump.
    pub entity_index: u32,
    /// Renderer point-light payload prepared before commit.
    pub light: PointLight,
}

/// A self-contained BSP preparation candidate.
///
/// All preparation work — parsing, extraction, upload, bridge preparation,
/// validation — is performed on the candidate. The commit step is a pure
/// publish: it moves the candidate's ready resources into the active world
/// without performing any new work.
#[derive(Debug)]
pub struct BspCandidate {
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
    /// Validation status.
    pub validation: CandidateValidation,
    /// Whether scene publication preflight has completed.
    pub publication_validated: bool,
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
}

impl BspCandidate {
    /// Create a new unvalidated candidate from extracted BSP data.
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
            generation,
            source_identity,
            extracted,
            cache_identity,
            renderer_lease: RendererLease::NotStarted,
            bridge_tokens,
            source_link,
            source_link_json,
            point_lights,
            validation: CandidateValidation::Unvalidated,
            publication_validated: false,
            diagnostics: Vec::new(),
            was_occupied,
            face_count,
            entity_count,
            light_count,
            batch_count,
            has_pvs,
        }
    }

    /// Begin a renderer upload. Transitions the lease from NotStarted to Pending.
    pub fn start_renderer_upload(&mut self) -> Result<(), BspRuntimeError> {
        match self.renderer_lease {
            RendererLease::NotStarted => {
                self.renderer_lease = RendererLease::Pending {
                    generation: self.generation,
                };
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

    /// Mark the renderer upload as failed.
    pub fn fail_renderer_upload(&mut self, reason: String) {
        self.renderer_lease = RendererLease::Failed { reason };
    }

    /// Transition the renderer lease from Pending (or NotStarted) to Ready.
    ///
    /// Accepts the mount directly even from `NotStarted` so callers that
    /// build the mount synchronously can skip the async Pending stage.
    pub fn set_renderer_ready(&mut self, mount: PreparedBspMount) -> Result<(), BspRuntimeError> {
        match &self.renderer_lease {
            RendererLease::Pending { generation } if *generation == self.generation => {
                self.renderer_lease = RendererLease::Ready(mount);
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
                Ok(())
            }
            RendererLease::Ready(_) => {
                // Idempotent: already ready, replace with new mount.
                self.renderer_lease = RendererLease::Ready(mount);
                Ok(())
            }
            RendererLease::Failed { ref reason } => Err(BspRuntimeError::SourceUnavailable {
                reason: format!("renderer upload failed: {reason}"),
            }),
        }
    }

    /// Take the ready mount, leaving the lease in NotStarted.
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

    /// Returns true if the renderer lease is ready.
    pub fn is_renderer_ready(&self) -> bool {
        matches!(self.renderer_lease, RendererLease::Ready(_))
    }

    /// Mark the candidate as validated.
    pub fn mark_validated(&mut self) {
        self.validation = CandidateValidation::Validated;
    }

    /// Mark scene publication preflight as complete.
    pub fn mark_publication_validated(&mut self) {
        self.publication_validated = true;
    }

    /// Returns true if the candidate has passed validation.
    pub fn is_validated(&self) -> bool {
        self.validation == CandidateValidation::Validated
    }

    /// Returns true if scene publication has been preflighted.
    pub fn is_publication_validated(&self) -> bool {
        self.publication_validated
    }

    /// Returns true if the candidate is ready for commit (validated + renderer ready).
    pub fn is_commit_ready(&self) -> bool {
        self.is_validated() && self.is_publication_validated() && self.is_renderer_ready()
    }
}
