//! # BSP Renderer API Surface
//!
//! Feature-gated (`renderer/bsp`) public API for registering BSP materials,
//! textures, and surface parameters with the renderer. All BSP-specific API
//! surface lives here and is only available when the `bsp` feature is active.

// Internal module-use imports.
use crate::data::handles::{BspMaterialHandle, TextureHandle};

// Re-exports for external consumers (tests, extraction pipeline).
pub use crate::data::bsp_material::{
    BspMaterialDesc, BspMaterialPipeline, BspSurfaceClass, BspTextureSet,
};
pub use crate::data::data_cache::{
    BspCachedSurface as BspCachedSurfaceRepr, BspSurfaceCache as BspSurfaceCacheRepr,
    VkPipelineType,
};
pub use crate::data::gpu_data::BspSurfaceUniform;

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
