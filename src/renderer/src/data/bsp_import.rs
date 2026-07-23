//! BSP mesh and lightmap upload from neutral ExtractedBsp DTOs.
//!
//! Converts BSP render batches to GPU vertex/index buffers via the existing
//! `ProceduralMeshData` path and builds lightmap atlas texture arrays.

#[cfg(feature = "bsp")]
use crate::api::ProceduralMeshData;
#[cfg(feature = "bsp")]
use crate::api::ProceduralVertex;
#[cfg(feature = "bsp")]
use crate::data::bsp_material::{BspMaterialDesc, BspSurfaceClass, BspTextureSet};
#[cfg(feature = "bsp")]
use crate::data::gpu_data::BspSurfaceUniform;
#[cfg(feature = "bsp")]
use crate::data::handles::{BspTextureHandle, MaterialHandle, MeshHandle};
#[cfg(feature = "bsp")]
use bsp::extract::ExtractedBsp;
#[cfg(feature = "bsp")]
use bsp::geometry::{FaceGeometry, RenderBatch};
#[cfg(feature = "bsp")]
use bsp::lightmaps::FaceLightmapLayout;
#[cfg(feature = "bsp")]
use glam::{Vec2, Vec3, Vec4};

// ── Mesh upload ────────────────────────────────────────────────────────

#[cfg(feature = "bsp")]
/// Convert a single BSP face geometry into `ProceduralMeshData` suitable for
/// GPU upload through the existing `upload_procedural_mesh` path.
///
/// Returns `None` when the face has fewer than 3 valid vertices.
pub fn face_to_procedural_mesh(
    face_geo: &FaceGeometry,
    _texinfo_scale_s: f32,
    _texinfo_scale_t: f32,
    material: Option<MaterialHandle>,
) -> Option<ProceduralMeshData> {
    let n = face_geo.vertices.len();
    if n < 3 {
        return None;
    }

    if !face_geo.normal.is_finite() || face_geo.normal.length_squared() < 1e-8 {
        return None;
    }
    if face_geo.vertices.iter().any(|v| !v.is_finite()) {
        return None;
    }

    let normal = face_geo.normal.normalize();
    // Flat-shaded faces: all vertices share the face normal and a
    // computed tangent.
    let bitangent = compute_flat_bitangent(normal);
    let tangent_w = compute_tangent_w(&face_geo.vertices, &face_geo.uv0, normal, bitangent);

    let mut vertices: Vec<ProceduralVertex> = Vec::with_capacity(n);
    for i in 0..n {
        let pos = face_geo.vertices[i];
        let uv0 = face_geo.uv0.get(i).copied().unwrap_or(Vec2::ZERO);
        let uv1 = face_geo.uv1.get(i).copied().unwrap_or(Vec2::ZERO);
        vertices.push(ProceduralVertex {
            position: pos,
            normal,
            tangent: Vec4::new(bitangent.x, bitangent.y, bitangent.z, tangent_w),
            uv0,
            uv1,
            color: Vec4::ONE,
        });
    }

    // Fan triangulation for convex polygon (standard for BSP faces).
    let mut indices: Vec<u32> = Vec::with_capacity((n - 2) * 3);
    for k in 1..(n - 1) {
        indices.push(0);
        indices.push(k as u32);
        indices.push((k + 1) as u32);
    }

    let name = format!("bsp_face_{}", face_geo.face_index);
    Some(ProceduralMeshData {
        name,
        vertices,
        indices,
        material,
    })
}

#[cfg(feature = "bsp")]
/// Compute a bitangent perpendicular to `normal` for flat-shaded faces.
fn compute_flat_bitangent(normal: Vec3) -> Vec3 {
    let abs = normal.abs();
    // Pick a reference vector least aligned with the normal.
    let ref_vec = if abs.x <= abs.y && abs.x <= abs.z {
        Vec3::X
    } else if abs.y <= abs.z {
        Vec3::Y
    } else {
        Vec3::Z
    };
    ref_vec.cross(normal).normalize()
}

#[cfg(feature = "bsp")]
/// Compute tangent.w handedness sign: +1 or -1 based on UV winding orientation.
fn compute_tangent_w(vertices: &[Vec3], uv0: &[Vec2], normal: Vec3, bitangent: Vec3) -> f32 {
    if vertices.len() < 3 || uv0.len() < 3 {
        return 1.0;
    }
    // Use the first triangle to determine handedness.
    let (v0, v1, v2) = (vertices[0], vertices[1], vertices[2]);
    let (uv_a, uv_b, uv_c) = (uv0[0], uv0[1], uv0[2]);

    let edge1 = v1 - v0;
    let edge2 = v2 - v0;
    let delta_uv1 = uv_b - uv_a;
    let delta_uv2 = uv_c - uv_a;

    let f = 1.0 / (delta_uv1.x * delta_uv2.y - delta_uv2.x * delta_uv1.y);
    let tangent = Vec3::new(
        f * (delta_uv2.y * edge1.x - delta_uv1.y * edge2.x),
        f * (delta_uv2.y * edge1.y - delta_uv1.y * edge2.y),
        f * (delta_uv2.y * edge1.z - delta_uv1.y * edge2.z),
    );

    // handedness: sign of (tangent × bitangent) · normal
    let handedness = tangent.cross(bitangent).dot(normal);
    if handedness < 0.0 {
        -1.0
    } else {
        1.0
    }
}

// ── Batch mesh upload ───────────────────────────────────────────────────

#[cfg(feature = "bsp")]
/// Result of uploading BSP mesh batches to GPU.
#[derive(Debug, Clone)]
pub struct BspMeshUploadResult {
    /// Mesh handles for every uploaded face (in face index order).
    /// Faces classified as Nodraw have `MeshHandle::new(0, 0)`.
    pub mesh_handles: Vec<MeshHandle>,
    /// Total number of meshes actually uploaded (excluding nodraw).
    pub uploaded_count: usize,
    /// Total number of faces processed.
    pub face_count: usize,
}

#[cfg(feature = "bsp")]
/// Build `ProceduralMeshData` for every renderable face in the extracted BSP.
///
/// Returns a Vec mapping face index → Option<ProceduralMeshData>.
/// Nodraw faces return None.
pub fn build_face_meshes(extracted: &ExtractedBsp) -> Vec<Option<ProceduralMeshData>> {
    let num_faces = extracted.face_geometries.len();
    let mut meshes: Vec<Option<ProceduralMeshData>> = Vec::with_capacity(num_faces);

    for (fi, face_geo) in extracted.face_geometries.iter().enumerate() {
        let sc = extracted.face_materials.get(fi).map(|m| m.surface_class);
        let is_hidden = sc.map(|class| !class.is_visible()).unwrap_or(false);

        if is_hidden || !face_geo.is_valid {
            meshes.push(None);
            continue;
        }

        // No texinfo scale needed; UV0 is baked into the geometry.
        let mesh = face_to_procedural_mesh(face_geo, 1.0, 1.0, None);
        meshes.push(mesh);
    }

    meshes
}

// ── Lightmap atlas upload ───────────────────────────────────────────────

#[cfg(feature = "bsp")]
/// A prepared lightmap atlas page ready for GPU upload.
#[derive(Debug, Clone)]
pub struct BspLightmapAtlasPage {
    /// Width in texels.
    pub width: u32,
    /// Height in texels.
    pub height: u32,
    /// Number of style layers (array layers).
    pub layer_count: u32,
    /// RGBA8 pixel data per layer: `layer * (width * height * 4)`.
    pub pixels: Vec<u8>,
}

#[cfg(feature = "bsp")]
impl BspLightmapAtlasPage {
    /// Build atlas pages from the extracted BSP lightmap data.
    ///
    /// The extraction pipeline already populates atlas pages with RGB8 data.
    /// This function converts each `AtlasPage` into a `BspLightmapAtlasPage`
    /// suitable for GPU upload by expanding to RGBA8 and organizing by style layers.
    pub fn from_extracted(extracted: &ExtractedBsp) -> Vec<Self> {
        let atlas = &extracted.lightmap_atlas;

        if atlas.pages.is_empty() {
            return Vec::new();
        }

        // For now, produce one page per atlas page.
        // Style layers are packed within the page data by the extraction pipeline.
        atlas
            .pages
            .iter()
            .map(|page| {
                let pixel_count = (page.width * page.height) as usize;
                // Convert from RGB8 to RGBA8
                let mut rgba = vec![0u8; pixel_count * 4];
                for y in 0..page.height as usize {
                    for x in 0..page.width as usize {
                        let src_idx = (y * page.width as usize + x) * 3;
                        let dst_idx = (y * page.width as usize + x) * 4;
                        if src_idx + 2 < page.data.len() {
                            rgba[dst_idx] = page.data[src_idx];
                            rgba[dst_idx + 1] = page.data[src_idx + 1];
                            rgba[dst_idx + 2] = page.data[src_idx + 2];
                            rgba[dst_idx + 3] = 255;
                        }
                    }
                }

                BspLightmapAtlasPage {
                    width: page.width,
                    height: page.height,
                    // The current neutral atlas stores one packed RGB8 layer. Style-indexed
                    // arrays are represented by the BSP material ABI but are not duplicated
                    // in `AtlasPage::data` yet, so expose the uploaded layer count honestly.
                    layer_count: 1,
                    pixels: rgba,
                }
            })
            .collect()
    }

    /// Allocate a BspTextureHandle for this atlas page.
    ///
    /// The caller handles GPU upload; this provides a tracking handle.
    pub fn allocate_handle(&self, next_handle_slot: &mut u32) -> BspTextureHandle {
        let slot = *next_handle_slot;
        *next_handle_slot = slot.wrapping_add(1);
        BspTextureHandle::new(slot, 0)
    }
}

// ── Internal helpers ────────────────────────────────────────────────────

/// Write the luxels for a face to a specific atlas layer.
/// This operates on the packed atlas page data.
#[allow(unused)]
fn write_face_luxels_to_layer(
    pixels: &mut [u8],
    atlas_width: u32,
    atlas_height: u32,
    layer: u32,
    layout: &FaceLightmapLayout,
    _page_data: &[u8],
) {
    let x0 = layout.atlas_offset.0;
    let y0 = layout.atlas_offset.1;
    let lw = layout.luxel_extents.0;
    let lh = layout.luxel_extents.1;

    if lw == 0 || lh == 0 {
        return;
    }

    let layer_offset = layer as usize * (atlas_width as usize * atlas_height as usize * 4);

    for ly in 0..lh {
        for lx in 0..lw {
            let px = (x0 as usize + lx as usize) % atlas_width as usize;
            let py = (y0 as usize + ly as usize) % atlas_height as usize;

            if px < atlas_width as usize && py < atlas_height as usize {
                let dst_offset = layer_offset + (py * atlas_width as usize + px) * 4;
                if dst_offset + 3 < pixels.len() {
                    // Default to dark gray for unpopulated areas.
                    pixels[dst_offset] = 64;
                    pixels[dst_offset + 1] = 64;
                    pixels[dst_offset + 2] = 64;
                    pixels[dst_offset + 3] = 255;
                }
            }
        }
    }
}

// ── BSP surface material builder ────────────────────────────────────────

#[cfg(feature = "bsp")]
/// Build `BspMaterialDesc` entries for each unique material in the extracted BSP.
///
/// Returns a mapping from face index → `BspMaterialDesc` (excluding nodraw faces).
pub fn build_bsp_material_descs(
    extracted: &ExtractedBsp,
    albedo_handles: &[BspTextureHandle],
    lightmap_atlas_handle: BspTextureHandle,
) -> Vec<Option<BspMaterialDesc>> {
    let num_faces = extracted.face_geometries.len();
    let mut descs: Vec<Option<BspMaterialDesc>> = Vec::with_capacity(num_faces);

    for fi in 0..num_faces {
        let sc = extracted.face_materials.get(fi).map(|m| m.surface_class);
        let is_hidden = sc.map(|class| !class.is_visible()).unwrap_or(false);
        let face_geo = &extracted.face_geometries[fi];

        if is_hidden || !face_geo.is_valid {
            descs.push(None);
            continue;
        }

        let bsp_class = match sc {
            Some(bsp::materials::SurfaceClass::AlphaMask) => BspSurfaceClass::AlphaMask,
            Some(bsp::materials::SurfaceClass::Sky) => BspSurfaceClass::Sky,
            Some(bsp::materials::SurfaceClass::Liquid) => BspSurfaceClass::Liquid,
            // Opaque surfaces (standard lightmapped with fullbright mask)
            _ => BspSurfaceClass::Lightmapped,
        };

        let albedo = albedo_handles
            .get(fi)
            .copied()
            .unwrap_or_else(|| BspTextureHandle::new(0, 0));

        // For lightmapped, fullbright, alpha-mask, and liquid surfaces, attach the atlas.
        let lightmap = match bsp_class {
            BspSurfaceClass::Lightmapped
            | BspSurfaceClass::Fullbright
            | BspSurfaceClass::AlphaMask
            | BspSurfaceClass::Liquid => lightmap_atlas_handle,
            _ => BspTextureHandle::new(0, 0),
        };

        let layout = if let Some(layout) = extracted.face_lightmap_layouts.get(fi) {
            layout
        } else {
            descs.push(None);
            continue;
        };

        let luxel_w = layout.luxel_extents.0.max(1) as f32;
        let luxel_h = layout.luxel_extents.1.max(1) as f32;
        let atlas_w = extracted
            .lightmap_atlas
            .pages
            .get(layout.page_index as usize)
            .map(|p| p.width.max(1) as f32)
            .unwrap_or(4096.0);
        let atlas_h = extracted
            .lightmap_atlas
            .pages
            .get(layout.page_index as usize)
            .map(|p| p.height.max(1) as f32)
            .unwrap_or(4096.0);

        let surface_params = BspSurfaceUniform {
            lightmap_scale_bias: Vec4::new(
                luxel_w / atlas_w,
                luxel_h / atlas_h,
                layout.atlas_offset.0 as f32 / atlas_w,
                layout.atlas_offset.1 as f32 / atlas_h,
            ),
            style_index: 0,
            fullbright_base: 224,
            fullbright_count: 32,
            alpha_threshold: 0.5,
            animation_frame: 0,
            animation_time: 0.0,
            _pad0: 0,
            _pad1: 0,
        };

        let textures = BspTextureSet {
            albedo,
            fullbright_mask: None,
            lightmap_atlas: lightmap,
        };

        descs.push(Some(BspMaterialDesc {
            surface_class: bsp_class,
            textures,
            surface_params,
        }));
    }

    descs
}

// ── Integration helpers ─────────────────────────────────────────────────

#[cfg(feature = "bsp")]
/// Resolved BSP render batches ready for scene submission.
#[derive(Debug, Clone)]
pub struct BspRenderSubmissionData {
    /// Mesh handle per face (0,0 for nodraw).
    pub face_meshes: Vec<MeshHandle>,
    /// BSP material handle per face (None for nodraw).
    pub face_materials: Vec<Option<crate::data::handles::BspMaterialHandle>>,
    /// BSP material descriptors per face (None for nodraw).
    pub face_bsp_materials: Vec<Option<BspMaterialDesc>>,
    /// Lightmap atlas texture handle.
    pub lightmap_atlas_handle: BspTextureHandle,
    /// Render batches from extraction.
    pub render_batches: Vec<RenderBatch>,
    /// Leaf membership per face.
    pub leaf_membership: Vec<Vec<u32>>,
    /// Whether PVS data is available.
    pub has_pvs: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "bsp")]
    #[test]
    fn face_to_procedural_mesh_triangulation() {
        let face_geo = FaceGeometry {
            face_index: 0,
            vertices: vec![
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(1.0, 1.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            ],
            uv0: vec![
                Vec2::new(0.0, 0.0),
                Vec2::new(1.0, 0.0),
                Vec2::new(1.0, 1.0),
                Vec2::new(0.0, 1.0),
            ],
            uv1: vec![Vec2::ZERO; 4],
            normal: Vec3::Z,
            bounds: (Vec3::ZERO, Vec3::ONE),
            luxel_extents: (16, 16),
            is_valid: true,
        };

        let result = face_to_procedural_mesh(&face_geo, 1.0, 1.0, None);
        assert!(result.is_some());
        let mesh = result.unwrap();
        // Quad triangulated into 2 triangles = 6 indices
        assert_eq!(mesh.indices.len(), 6);
        assert_eq!(mesh.vertices.len(), 4);
        assert_eq!(mesh.indices, vec![0, 1, 2, 0, 2, 3]);
    }

    #[cfg(feature = "bsp")]
    #[test]
    fn face_to_procedural_mesh_degenerate() {
        let face_geo = FaceGeometry {
            face_index: 0,
            vertices: vec![Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0)],
            uv0: vec![Vec2::new(0.0, 0.0), Vec2::new(1.0, 0.0)],
            uv1: vec![Vec2::ZERO; 2],
            normal: Vec3::Z,
            bounds: (Vec3::ZERO, Vec3::ONE),
            luxel_extents: (16, 16),
            is_valid: true,
        };

        let result = face_to_procedural_mesh(&face_geo, 1.0, 1.0, None);
        assert!(result.is_none());
    }
}
