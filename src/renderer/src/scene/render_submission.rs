//! # Per-Frame Render Submission
//!
//! Flat draw payload emitted by `SceneWorld` and consumed by the rendergraph.
//! Keeps only stable handles and transforms so render code can resolve cache data safely.

use crate::data::data_cache::MeshCache;
use crate::data::gpu_data::SceneDataUBO;
use crate::data::handles::{EnvironmentHandle, MeshHandle};
use crate::vulkan::vk_sprites::SpriteInstance;
use glam::{Mat4, Vec3};

/// Maximum number of directional lights that can be uploaded to GPU per frame.
pub const MAX_DIRECTIONAL_LIGHTS_GPU: usize = 4;
/// Maximum number of point lights that can be uploaded to GPU per frame.
pub const MAX_POINT_LIGHTS_GPU: usize = 16;
/// Maximum number of spot lights that can be uploaded to GPU per frame.
pub const MAX_SPOT_LIGHTS_GPU: usize = 16;

#[derive(Debug, Copy, Clone)]
pub struct SubmissionFlags {
    pub draw_skybox: bool,
    pub draw_geometry: bool,
    pub draw_imgui: bool,
}

impl Default for SubmissionFlags {
    fn default() -> Self {
        Self {
            draw_skybox: true,
            draw_geometry: true,
            draw_imgui: true,
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub struct FrameDrawItem {
    pub mesh_id: MeshHandle,
    pub transform: Mat4,
}

#[derive(Debug, Copy, Clone)]
pub struct FramePointLight {
    pub position: Vec3,
    pub color: Vec3,
    pub intensity: f32,
    pub range: f32,
}

#[derive(Debug, Copy, Clone)]
pub struct FrameDirectionalLight {
    pub direction: Vec3,
    pub color: Vec3,
    pub intensity: f32,
    pub enable_shadows: bool,
}

#[derive(Debug, Copy, Clone)]
pub struct FrameSpotLight {
    pub position: Vec3,
    pub direction: Vec3,
    pub color: Vec3,
    pub intensity: f32,
    pub range: f32,
    pub inner_cos: f32,
    pub outer_cos: f32,
}

/// Stable cull reason carried through PVS and frustum decisions.
#[cfg(feature = "bsp")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BspCullReason {
    PvsCulled,
    FrustumCulled,
}

/// A typed BSP submission failure carrying batch identity and context.
#[cfg(feature = "bsp")]
#[derive(Debug, Clone)]
pub struct BspSubmissionFailure {
    /// Mounted batch index in canonical order.
    pub batch_index: usize,
    /// First source-face index from the mounted record.
    pub source_face_first: u32,
    /// Number of source faces in this batch.
    pub source_face_count: u32,
    /// Expected pipeline class.
    pub pipeline_class: Option<crate::data::data_cache::VkPipelineType>,
    /// Model index (0 = worldspawn, 1+ = inline).
    pub model_index: u32,
    /// Human-readable failure reason.
    pub reason: String,
}

/// A typed BSP recording failure after acquisition.
#[cfg(feature = "bsp")]
#[derive(Debug, Clone)]
pub enum BspRecordingFailure {
    /// Submission-side failure carried into recording.
    Submission(BspSubmissionFailure),
    /// Cache lock poisoned.
    CacheLockPoisoned(String),
    /// Missing or stale mesh handle.
    MissingMesh {
        batch_index: usize,
        mesh: MeshHandle,
    },
    /// Missing or stale material handle.
    MissingMaterial {
        batch_index: usize,
        material: crate::data::handles::BspMaterialHandle,
    },
    /// Missing or stale albedo texture.
    MissingAlbedoTexture { batch_index: usize },
    /// Missing or stale lightmap texture.
    MissingLightmapTexture { batch_index: usize },
    /// Missing fullbright texture (only when expected by pipeline).
    MissingFullbrightTexture { batch_index: usize },
    /// Null scene descriptor (set 0).
    NullSceneDescriptor,
    /// Null material descriptor (set 1).
    NullMaterialDescriptor { batch_index: usize },
    /// Null frame-values descriptor (set 2).
    NullFrameValuesDescriptor,
    /// Mesh index/vertex buffer missing.
    MissingMeshBuffer { batch_index: usize },
    /// Null or incompatible BSP pipeline.
    NullOrIncompatiblePipeline {
        batch_index: usize,
        expected: String,
    },
    /// Invalid frame slot index.
    InvalidFrameSlot { slot: u32 },
    /// Pipeline class mismatch between draw item and resolved material.
    PipelineClassDrift {
        batch_index: usize,
        expected: String,
        actual: String,
    },
    /// Failed to mark mesh referenced (stale generation or cache error).
    FailedMeshReference {
        batch_index: usize,
        mesh: MeshHandle,
    },
    /// Failed to mark albedo texture referenced.
    FailedAlbedoReference { batch_index: usize },
    /// Failed to mark fullbright texture referenced.
    FailedFullbrightReference { batch_index: usize },
    /// Failed to mark lightmap texture referenced.
    FailedLightmapReference { batch_index: usize },
}

/// Per-draw outcome recorded in the BSP command diagnostic collector.
#[cfg(feature = "bsp")]
#[derive(Debug, Clone)]
pub enum BspDrawOutcome {
    Recorded,
    Culled(BspCullReason),
    Failed(String),
}

/// Fixed-capacity per-frame BSP command diagnostic entry.
#[cfg(feature = "bsp")]
#[derive(Debug, Clone)]
pub struct BspCommandDiag {
    pub frame_slot: u32,
    pub pipeline: Option<crate::data::data_cache::VkPipelineType>,
    pub set_0: u64,
    pub set_1: u64,
    pub set_2: u64,
    pub batch_index: usize,
    pub mesh_generation: u32,
    pub material_generation: u32,
    pub outcome: BspDrawOutcome,
}

/// By-value BSP submission diagnostics produced at collection time.
#[cfg(feature = "bsp")]
#[derive(Debug, Clone, Default)]
pub struct BspSubmissionDiagnostics {
    pub total_mounted: u32,
    pub pvs_eligible: u32,
    pub pvs_visible: u32,
    pub pvs_culled: u32,
    pub conservative_visible: u32,
    pub invalid_membership: u32,
    pub frustum_culled: u32,
    pub recorded: u32,
    pub failed: u32,
}

/// A single BSP frame draw item with immutable trace identity.
#[cfg(feature = "bsp")]
#[derive(Debug, Clone)]
pub struct BspFrameDrawItem {
    pub mesh_id: MeshHandle,
    pub bsp_material_id: crate::data::handles::BspMaterialHandle,
    pub transform: Mat4,
    /// Mounted batch index in canonical order.
    pub batch_index: usize,
    /// First source-face index.
    pub source_face_first: u32,
    /// Source-face count from the mounted record.
    pub source_face_count: u32,
    /// Expected pipeline class from the planned material.
    pub pipeline_class: Option<crate::data::data_cache::VkPipelineType>,
    /// Model index (0 = worldspawn).
    pub model_index: u32,
    /// Phase 07: Canonical digest of the batch's immutable identity for evidence.
    pub canonical_digest: u64,
}

// ── Phase 07: Evidence carrier types ────────────────────────────────────

/// By-value semantic identity of a BSP batch, computed from immutable batch data only.
/// Used for evidence boundary comparisons and recorded-outcome tracking.
#[cfg(feature = "bsp")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BspBatchSemanticIdentity {
    pub batch_index: usize,
    pub canonical_digest: u64,
    pub face_count: u32,
    pub source_faces: Vec<u32>,
    pub model_index: u32,
    pub is_static: bool,
}

/// Phase 07: In-flight BSP evidence collector, carried through submission building
/// and populated during command recording. Sealed into a [`BspFrameEvidence`] report.
#[cfg(feature = "bsp")]
#[derive(Debug, Clone)]
pub struct BspEvidenceCollector {
    pub corpus_identity: String,
    pub request_identity: String,
    pub visibility_mode: crate::api::bsp::BspEvidenceVisibility,
    pub arena_id: Option<u64>,
    pub frame_number: u32,
    /// Neutral static-world identities (from canonical mount records).
    pub neutral_identities: Vec<BspBatchSemanticIdentity>,
    /// Mounted static-world identities (must match neutral for eligibility).
    pub mounted_identities: Vec<BspBatchSemanticIdentity>,
    /// Cull decisions: batch_index -> reason. Only present for culled static batches.
    pub cull_decisions: Vec<(usize, BspCullReason)>,
    /// Submitted identities (batches admitted after PVS/visibility).
    pub submitted_identities: Vec<BspBatchSemanticIdentity>,
    /// Recorded outcomes: batch_index + digest + outcome tag.
    pub recorded_outcomes: Vec<BspRecordedOutcome>,
    /// Inline model summary counts.
    pub inline_batch_count: u32,
    pub inline_face_count: u32,
    /// PVS diagnostics.
    pub pvs_eligible: u32,
    pub pvs_culled: u32,
    /// Atlas bytes at mount time.
    pub atlas_bytes: u64,
    /// Frame CPU time in ms (populated at seal time).
    pub frame_time_ms: f32,
    /// Typed failures.
    pub failures: Vec<crate::api::bsp::BspEvidenceFailure>,
    /// Whether the collector has been sealed.
    pub sealed: bool,
}

/// A single recorded outcome for a BSP draw.
#[cfg(feature = "bsp")]
#[derive(Debug, Clone)]
pub enum BspRecordedOutcome {
    Recorded {
        batch_index: usize,
        digest: u64,
    },
    Culled {
        batch_index: usize,
        reason: BspCullReason,
    },
    Failed {
        batch_index: usize,
        digest: u64,
        failure: crate::api::bsp::BspEvidenceFailure,
    },
}

#[cfg(feature = "bsp")]
impl BspEvidenceCollector {
    /// Create a new evidence collector from a request.
    pub fn new(
        request: &crate::api::bsp::BspEvidenceRequest,
        arena_id: Option<u64>,
        frame_number: u32,
    ) -> Self {
        Self {
            corpus_identity: request.corpus_identity.clone(),
            request_identity: request.request_identity.clone(),
            visibility_mode: request.visibility,
            arena_id,
            frame_number,
            neutral_identities: Vec::new(),
            mounted_identities: Vec::new(),
            cull_decisions: Vec::new(),
            submitted_identities: Vec::new(),
            recorded_outcomes: Vec::new(),
            inline_batch_count: 0,
            inline_face_count: 0,
            pvs_eligible: 0,
            pvs_culled: 0,
            atlas_bytes: 0,
            frame_time_ms: 0.0,
            failures: Vec::new(),
            sealed: false,
        }
    }

    /// Check if neutral and mounted static-world identities match.
    pub fn neutral_mounted_match(&self) -> bool {
        if self.neutral_identities.len() != self.mounted_identities.len() {
            return false;
        }
        self.neutral_identities
            .iter()
            .zip(self.mounted_identities.iter())
            .all(|(n, m)| {
                n.canonical_digest == m.canonical_digest
                    && n.batch_index == m.batch_index
                    && n.face_count == m.face_count
            })
    }

    /// Seal the collector into a [`BspFrameEvidence`] report.
    pub fn seal(mut self) -> crate::api::bsp::BspFrameEvidence {
        use crate::api::bsp::{
            BspCanonicalDigest, BspEvidenceBatchEntry, BspEvidenceBoundary, BspEvidenceFailure,
            BspFrameEvidence, BSP_EVIDENCE_MAX_BATCH_ENTRIES, BSP_EVIDENCE_MAX_FAILURES,
            BSP_EVIDENCE_MAX_SOURCE_FACES,
        };

        self.sealed = true;
        let failures: Vec<BspEvidenceFailure> = self
            .failures
            .iter()
            .take(BSP_EVIDENCE_MAX_FAILURES)
            .cloned()
            .collect();
        let truncated = self.failures.len() > BSP_EVIDENCE_MAX_FAILURES;
        let eligible = failures.is_empty() && !truncated && self.neutral_mounted_match();

        let build_boundary = |identities: &[BspBatchSemanticIdentity]| -> BspEvidenceBoundary {
            let batch_count = identities.len() as u32;
            let mut aggregate = 0u64;
            let material_ids: std::collections::BTreeSet<u32> = std::collections::BTreeSet::new();
            let entries_truncated = identities.len() > BSP_EVIDENCE_MAX_BATCH_ENTRIES;
            let batch_entries: Vec<BspEvidenceBatchEntry> = identities
                .iter()
                .take(BSP_EVIDENCE_MAX_BATCH_ENTRIES)
                .map(|id| {
                    aggregate ^= id.canonical_digest;
                    BspEvidenceBatchEntry {
                        batch_index: id.batch_index,
                        digest: BspCanonicalDigest(id.canonical_digest),
                        face_count: id.face_count,
                        source_faces: id
                            .source_faces
                            .iter()
                            .take(BSP_EVIDENCE_MAX_SOURCE_FACES)
                            .copied()
                            .collect(),
                    }
                })
                .collect();
            BspEvidenceBoundary {
                batch_count,
                draw_call_count: batch_count,
                triangle_count: 0,
                material_count: material_ids.len() as u32,
                aggregate_digest: BspCanonicalDigest(aggregate),
                batch_entries,
                truncated: entries_truncated,
            }
        };

        let neutral = build_boundary(&self.neutral_identities);
        let mounted = build_boundary(&self.mounted_identities);
        let submitted = build_boundary(&self.submitted_identities);

        // Recorded gets additional draw/triangle counts from outcomes.
        let recorded_batch_count = self.recorded_outcomes.len() as u32;
        let recorded_draw_calls = self
            .recorded_outcomes
            .iter()
            .filter(|o| matches!(o, BspRecordedOutcome::Recorded { .. }))
            .count() as u32;
        let mut recorded_aggregate = 0u64;
        let recorded_truncated = self.recorded_outcomes.len() > BSP_EVIDENCE_MAX_BATCH_ENTRIES;
        let recorded_entries: Vec<BspEvidenceBatchEntry> = self
            .recorded_outcomes
            .iter()
            .take(BSP_EVIDENCE_MAX_BATCH_ENTRIES)
            .filter_map(|outcome| match outcome {
                BspRecordedOutcome::Recorded {
                    batch_index,
                    digest,
                } => {
                    recorded_aggregate ^= digest;
                    Some(BspEvidenceBatchEntry {
                        batch_index: *batch_index,
                        digest: BspCanonicalDigest(*digest),
                        face_count: 0,
                        source_faces: Vec::new(),
                    })
                }
                BspRecordedOutcome::Culled { .. } => None,
                BspRecordedOutcome::Failed { .. } => None,
            })
            .collect();

        let recorded = BspEvidenceBoundary {
            batch_count: recorded_batch_count,
            draw_call_count: recorded_draw_calls,
            triangle_count: 0,
            material_count: 0,
            aggregate_digest: BspCanonicalDigest(recorded_aggregate),
            batch_entries: recorded_entries,
            truncated: recorded_truncated,
        };

        BspFrameEvidence {
            corpus_identity: self.corpus_identity,
            request_identity: self.request_identity,
            arena_id: self.arena_id,
            frame_number: self.frame_number,
            visibility_mode: self.visibility_mode,
            neutral,
            mounted,
            submitted,
            recorded,
            inline_batch_count: self.inline_batch_count,
            inline_face_count: self.inline_face_count,
            pvs_eligible: self.pvs_eligible,
            pvs_culled: self.pvs_culled,
            atlas_bytes: self.atlas_bytes,
            frame_time_ms: self.frame_time_ms,
            failures,
            eligible,
        }
    }
}

/// BSP frame-varying values captured from the scene snapshot for this submission.
#[cfg(feature = "bsp")]
#[derive(Debug, Copy, Clone)]
pub struct BspFrameValuesState {
    pub style_intensities: [f32; 64],
    pub liquid_time: f32,
    /// Active BSP surface-cache arena identity for the published mount.
    pub arena_id: Option<u64>,
}

#[cfg(feature = "bsp")]
impl Default for BspFrameValuesState {
    fn default() -> Self {
        let mut style_intensities = [0.0; 64];
        style_intensities[0] = 1.0;
        Self {
            style_intensities,
            liquid_time: 0.0,
            arena_id: None,
        }
    }
}

pub struct RenderSubmission {
    pub camera: SceneDataUBO,
    pub draw_items: Vec<FrameDrawItem>,
    pub flags: SubmissionFlags,
    pub skybox_mesh_id: MeshHandle,
    pub skybox_env_id: EnvironmentHandle,
    pub point_lights: Vec<FramePointLight>,
    /// First directional light, retained for legacy consumers.
    pub directional_light: Option<FrameDirectionalLight>,
    /// Bounded collection used by multi-directional direct lighting.
    pub directional_lights: Vec<FrameDirectionalLight>,
    pub spot_lights: Vec<FrameSpotLight>,
    pub culling_stats: CullingStats,
    pub bounds_references: Vec<MeshHandle>,
    /// BSP draw items for BSP-specific pipeline dispatch.
    /// Present only when `feature = "bsp"` and a BSP mount is active.
    #[cfg(feature = "bsp")]
    pub bsp_draw_items: Vec<BspFrameDrawItem>,
    /// BSP light selection: PVS-filtered + scored lights for this frame.
    #[cfg(feature = "bsp")]
    pub bsp_selected_lights: Vec<FramePointLight>,
    /// BSP frame-varying shader values captured with this submission.
    #[cfg(feature = "bsp")]
    pub bsp_frame_values: BspFrameValuesState,
    /// First BSP submission failure (carried into recording as typed error).
    #[cfg(feature = "bsp")]
    pub bsp_failure: Option<BspSubmissionFailure>,
    /// By-value BSP submission diagnostics.
    #[cfg(feature = "bsp")]
    pub bsp_diagnostics: BspSubmissionDiagnostics,
    /// Per-frame BSP command diagnostic collector (populated during recording).
    /// Fixed capacity; truncation is explicitly logged.
    #[cfg(feature = "bsp")]
    pub bsp_command_diags: Vec<BspCommandDiag>,
    /// Phase 07: Evidence collector for runtime draw evidence (one-shot, opt-in).
    /// Wrapped in `RefCell` so recording can populate outcomes via shared reference.
    #[cfg(feature = "bsp")]
    pub bsp_evidence_collector: std::cell::RefCell<Option<BspEvidenceCollector>>,
    /// Debug line segments from [`FrameExtensions`]. Rendered by the debug-lines
    /// pass when the `debug-draw` feature is enabled and lines are non-empty.
    #[cfg(feature = "debug-draw")]
    pub debug_lines: Vec<(glam::Vec3, glam::Vec3, glam::Vec3)>,
    /// Sprite instances for the sprite batch pass.
    pub sprites: Vec<SpriteInstance>,
    /// Orthographic view matrix for the 2D sprite camera.
    pub sprite_camera_view: Mat4,
    /// Orthographic projection matrix for the 2D sprite camera.
    pub sprite_camera_projection: Mat4,
}

#[derive(Debug, Default, Copy, Clone, PartialEq, Eq)]
pub struct CullingStats {
    pub known_nodes_tested: u32,
    pub proxy_nodes_tested: u32,
    pub conservative_nodes_by_reason: [u32; 5],
    pub subtree_bounds_tested: u32,
    pub subtrees_pruned: u32,
    pub submitted_draw_items: u32,
}

impl RenderSubmission {
    pub fn new(camera: SceneDataUBO, draw_capacity: usize) -> Self {
        Self {
            camera,
            draw_items: Vec::with_capacity(draw_capacity),
            flags: SubmissionFlags::default(),
            skybox_mesh_id: MeshCache::SKYBOX_MESH,
            skybox_env_id: EnvironmentHandle::new(0, 0),
            point_lights: Vec::with_capacity(MAX_POINT_LIGHTS_GPU),
            directional_light: None,
            directional_lights: Vec::with_capacity(MAX_DIRECTIONAL_LIGHTS_GPU),
            spot_lights: Vec::with_capacity(MAX_SPOT_LIGHTS_GPU),
            culling_stats: CullingStats::default(),
            bounds_references: Vec::with_capacity(draw_capacity),
            #[cfg(feature = "bsp")]
            bsp_draw_items: Vec::new(),
            #[cfg(feature = "bsp")]
            bsp_selected_lights: Vec::new(),
            #[cfg(feature = "bsp")]
            bsp_frame_values: BspFrameValuesState::default(),
            #[cfg(feature = "bsp")]
            bsp_failure: None,
            #[cfg(feature = "bsp")]
            bsp_diagnostics: BspSubmissionDiagnostics::default(),
            #[cfg(feature = "bsp")]
            bsp_command_diags: Vec::new(),
            #[cfg(feature = "bsp")]
            bsp_evidence_collector: std::cell::RefCell::new(None),
            #[cfg(feature = "debug-draw")]
            debug_lines: Vec::new(),
            sprites: Vec::new(),
            sprite_camera_view: Mat4::IDENTITY,
            sprite_camera_projection: Mat4::IDENTITY,
        }
    }

    pub fn push_draw_item(&mut self, draw_item: FrameDrawItem) {
        self.draw_items.push(draw_item);
        self.culling_stats.submitted_draw_items += 1;
    }

    pub fn has_draw_targets(&self) -> bool {
        self.flags.draw_skybox || self.flags.draw_geometry
    }
}
