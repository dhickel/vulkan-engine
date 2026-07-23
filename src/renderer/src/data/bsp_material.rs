//! # BSP Material Types
//!
//! Feature-gated (`renderer/bsp`) BSP material pipeline variants, texture set
//! descriptors, and surface classification types.

use crate::data::gpu_data::BspSurfaceUniform;
use crate::data::handles::BspTextureHandle;

/// BSP material pipeline variants matching the BSP shader ABI.
///
/// Each variant maps to a dedicated graphics pipeline with distinct depth/blend/cull
/// state and a shared BSP descriptor-set layout.
#[cfg(feature = "bsp")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BspMaterialPipeline {
    /// Opaque lightmapped surfaces; depth write, back-face cull.
    Opaque,
    /// Fullbright/emissive surfaces; same depth/blend as Opaque but skips lightmap.
    Fullbright,
    /// Alpha-masked surfaces (fences, grates); alpha test, depth write, two-sided.
    AlphaMask,
    /// Sky surfaces; depth write max, back-face cull, environment sampling.
    Sky,
    /// Liquid/warp surfaces; alpha blend, depth write off, two-sided.
    Liquid,
}

/// Descriptive classification for a BSP surface.
///
/// Used during extraction to determine which pipeline variant a surface uses
/// and what texture set it requires.
#[cfg(feature = "bsp")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BspSurfaceClass {
    /// Standard lightmapped surface.
    Lightmapped,
    /// Fullbright/emissive surface (skip lightmap sampling).
    Fullbright,
    /// Alpha-masked surface (fences, grates, etc.).
    AlphaMask,
    /// Sky surface (depth-preserving, environment behind).
    Sky,
    /// Liquid/warp surface (animated, two-sided, transparent).
    Liquid,
    /// Nodraw — excluded from render geometry.
    Nodraw,
}

/// Texture set descriptor for a BSP material.
///
/// BSP materials bind three samplers plus a UBO in their own descriptor set:
/// - Albedo (base color / diffuse)
/// - Fullbright emissive mask (optional; default white if not used)
/// - Lightmap atlas (sampler2DArray, style-indexed)
#[cfg(feature = "bsp")]
#[derive(Debug, Clone)]
pub struct BspTextureSet {
    /// Albedo texture handle (base color).
    pub albedo: BspTextureHandle,
    /// Fullbright emissive mask handle (optional).
    pub fullbright_mask: Option<BspTextureHandle>,
    /// Lightmap atlas texture handle (sampler2DArray).
    pub lightmap_atlas: BspTextureHandle,
}

/// CPU-side descriptor for a BSP material submitted through the public API.
#[cfg(feature = "bsp")]
#[derive(Debug, Clone)]
pub struct BspMaterialDesc {
    /// Surface class determines pipeline variant.
    pub surface_class: BspSurfaceClass,
    /// Texture handles for albedo, fullbright mask, lightmap atlas.
    pub textures: BspTextureSet,
    /// Surface UBO parameters (scale/bias, style, fullbright range, etc.).
    pub surface_params: BspSurfaceUniform,
}
