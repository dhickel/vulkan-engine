//! # BSP Renderer API Surface
//!
//! Feature-gated (`renderer/bsp`) public API for registering BSP materials,
//! textures, and surface parameters with the renderer. All BSP-specific API
//! surface lives here and is only available when the `bsp` feature is active.

// Internal module-use imports.
#[cfg(feature = "bsp")]
use crate::data::handles::{BspMaterialHandle, MeshHandle, TextureHandle};

// Re-exports for external consumers (tests, extraction pipeline).
pub use crate::data::bsp_import::{
    build_bsp_material_descs, build_face_meshes, face_to_procedural_mesh, BspLightmapAtlasPage,
    BspMeshUploadResult, BspRenderSubmissionData,
};
pub use crate::data::bsp_material::{
    BspMaterialDesc, BspMaterialPipeline, BspSurfaceClass, BspTextureSet,
};
pub use crate::data::data_cache::{
    BspCachedSurface as BspCachedSurfaceRepr, BspSurfaceCache as BspSurfaceCacheRepr,
    VkPipelineType,
};
pub use crate::data::gpu_data::BspSurfaceUniform;
pub use crate::scene::bsp_visibility::{filter_batches_by_pvs, pvs_should_disable, BspMountState};

/// Resources prepared but not yet published to the BSP surface cache.
///
/// Created by the BSP extraction pipeline before commit. Call
/// [`publish`](BspRendererResources::publish) to atomically insert into the cache.
#[cfg(feature = "bsp")]
pub struct BspRendererResources {
    /// Prepared surface records keyed by arbitrary extraction-side keys.
    materials: Vec<BspMaterialDesc>,
}

#[cfg(feature = "bsp")]
impl BspRendererResources {
    /// Create an empty prepared-resources container.
    pub fn new() -> Self {
        Self {
            materials: Vec::new(),
        }
    }

    /// Stage a BSP material for later commit.
    pub fn add_material(&mut self, desc: BspMaterialDesc) {
        self.materials.push(desc);
    }

    /// Publish all staged materials to the BSP surface cache.
    ///
    /// Returns material handles in staging order, excluding nodraw surfaces.
    /// The cache lock must be held by the caller.
    pub fn publish(self, cache: &mut BspSurfaceCacheRepr) -> Vec<BspMaterialHandle> {
        self.materials
            .into_iter()
            .filter_map(|desc| {
                let pipeline = match desc.surface_class {
                    BspSurfaceClass::Lightmapped => VkPipelineType::BspOpaque,
                    BspSurfaceClass::Fullbright => VkPipelineType::BspFullbright,
                    BspSurfaceClass::AlphaMask => VkPipelineType::BspAlphaMask,
                    BspSurfaceClass::Sky => VkPipelineType::BspSky,
                    BspSurfaceClass::Liquid => VkPipelineType::BspLiquid,
                    BspSurfaceClass::Nodraw => return None,
                };

                let to_texture_handle = |handle: crate::data::handles::BspTextureHandle| {
                    TextureHandle::new(handle.slot, handle.generation)
                };

                Some(cache.add(BspCachedSurfaceRepr {
                    material_descriptor: ash::vk::DescriptorSet::null(),
                    surf_ubo_alloc: Default::default(),
                    pipeline,
                    albedo_tex: to_texture_handle(desc.textures.albedo),
                    fullbright_tex: desc.textures.fullbright_mask.map(to_texture_handle),
                    lightmap_tex: to_texture_handle(desc.textures.lightmap_atlas),
                }))
            })
            .collect()
    }
}

#[cfg(feature = "bsp")]
impl Default for BspRendererResources {
    fn default() -> Self {
        Self::new()
    }
}

// ── BSP Mount State ────────────────────────────────────────────────────

/// Prepared BSP mount ready to be attached to a scene.
///
/// Contains GPU-uploaded meshes, materials, lightmap atlas handles,
/// leaf membership data, and PVS state. Constructed by the asset
/// loading pipeline and consumed by [`crate::api::scene::Scene::set_bsp_mount`].
#[cfg(feature = "bsp")]
#[derive(Clone)]
pub struct PreparedBspMount {
    /// BSP visibility state (nodes, leaves, PVS data).
    pub mount_state: BspMountState,
    /// Mesh handles per face (0,0 for nodraw).
    pub face_meshes: Vec<MeshHandle>,
    /// BSP material handles per face.
    pub face_materials: Vec<Option<BspMaterialHandle>>,
    /// Leaf membership per face.
    pub leaf_membership: Vec<Vec<u32>>,
    /// Render batches from extraction (for PVS filtering).
    pub render_batches: Vec<bsp::geometry::RenderBatch>,
    /// BSP light descriptors extracted from light entities.
    pub light_descriptors: Vec<bsp::extract::LightDescriptor>,
}

#[cfg(feature = "bsp")]
impl PreparedBspMount {
    /// Create a new empty prepared mount.
    pub fn new() -> Self {
        Self {
            mount_state: BspMountState::new(),
            face_meshes: Vec::new(),
            face_materials: Vec::new(),
            leaf_membership: Vec::new(),
            render_batches: Vec::new(),
            light_descriptors: Vec::new(),
        }
    }

    /// Create a prepared mount from an extracted BSP and uploaded resources.
    pub fn from_extracted(
        mount_state: BspMountState,
        face_meshes: Vec<MeshHandle>,
        face_materials: Vec<Option<BspMaterialHandle>>,
        leaf_membership: Vec<Vec<u32>>,
        render_batches: Vec<bsp::geometry::RenderBatch>,
        light_descriptors: Vec<bsp::extract::LightDescriptor>,
    ) -> Self {
        Self {
            mount_state,
            face_meshes,
            face_materials,
            leaf_membership,
            render_batches,
            light_descriptors,
        }
    }

    /// Build a PreparedBspMount from an extracted BSP without GPU upload.
    ///
    /// This is a coordinator integration hook: the mount state is populated
    /// with PVS, leaf membership, render batches, and light descriptors from
    /// the extraction, but mesh and material arrays are empty. GPU resource
    /// upload is handled separately by the renderer's extraction pipeline.
    ///
    /// The resulting mount is suitable for `Scene::set_bsp_mount` and will
    /// correctly participate in PVS culling and light selection.
    pub fn from_extraction_stub(extracted: &bsp::extract::ExtractedBsp) -> Self {
        let mut mount_state = BspMountState::new();
        mount_state.activate();
        mount_state.set_leaf_membership(extracted.leaf_membership.clone());

        let face_count = extracted.face_geometries.len();
        let stub_meshes = vec![MeshHandle::new(0, 0); face_count];
        let stub_materials = vec![None; face_count];

        // Use the PVS camera set if available
        // (actual camera-dependent PVS is updated per-frame by the renderer)

        mount_state.set_render_assets(
            stub_meshes.clone(),
            stub_materials,
            extracted.render_batches.clone(),
            extracted.light_descriptors.clone(),
        );

        Self {
            mount_state,
            face_meshes: stub_meshes,
            face_materials: vec![None; face_count],
            leaf_membership: extracted.leaf_membership.clone(),
            render_batches: extracted.render_batches.clone(),
            light_descriptors: extracted.light_descriptors.clone(),
        }
    }
}

#[cfg(feature = "bsp")]
impl Default for PreparedBspMount {
    fn default() -> Self {
        Self::new()
    }
}
