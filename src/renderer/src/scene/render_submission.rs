//! # Per-Frame Render Submission
//!
//! Flat draw payload emitted by `SceneWorld` and consumed by the rendergraph.
//! Keeps only stable handles and transforms so render code can resolve cache data safely.

use crate::data::data_cache::MeshCache;
use crate::data::gpu_data::SceneDataUBO;
use crate::data::handles::{EnvironmentHandle, MeshHandle};
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
    MissingMesh { batch_index: usize, mesh: MeshHandle },
    /// Missing or stale material handle.
    MissingMaterial { batch_index: usize, material: crate::data::handles::BspMaterialHandle },
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
    NullOrIncompatiblePipeline { batch_index: usize, expected: String },
    /// Invalid frame slot index.
    InvalidFrameSlot { slot: u32 },
    /// Pipeline class mismatch between draw item and resolved material.
    PipelineClassDrift { batch_index: usize, expected: String, actual: String },
    /// Failed to mark mesh referenced (stale generation or cache error).
    FailedMeshReference { batch_index: usize, mesh: MeshHandle },
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
}

/// BSP frame-varying values captured from the scene snapshot for this submission.
#[cfg(feature = "bsp")]
#[derive(Debug, Copy, Clone)]
pub struct BspFrameValuesState {
    pub style_intensities: [f32; 64],
    pub liquid_time: f32,
}

#[cfg(feature = "bsp")]
impl Default for BspFrameValuesState {
    fn default() -> Self {
        let mut style_intensities = [0.0; 64];
        style_intensities[0] = 1.0;
        Self {
            style_intensities,
            liquid_time: 0.0,
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
