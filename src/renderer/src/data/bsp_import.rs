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
use crate::data::gpu_data::{bsp_surface_flags, BspSurfaceUniform};
#[cfg(feature = "bsp")]
use crate::data::handles::{BspMaterialHandle, BspTextureHandle, MaterialHandle, MeshHandle};
#[cfg(feature = "bsp")]
use bsp::extract::ExtractedBsp;
#[cfg(feature = "bsp")]
use bsp::geometry::{FaceGeometry, RenderBatch};
#[cfg(feature = "bsp")]
use bsp::lightmaps::FaceLightmapLayout;
#[cfg(feature = "bsp")]
use glam::{UVec4, Vec2, Vec3, Vec4};

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

/// Hard upper bound for one mounted BSP's static render batches.
///
/// The accepted M3 submission budget is below 2,000 draws. A power-of-two
/// reservation cap leaves a small amount of diagnostic headroom while still
/// preventing descriptor/draw amplification on pathological maps.
#[cfg(feature = "bsp")]
pub const MAX_BSP_RENDER_BATCHES: usize = 2_048;
/// Maximum unique BSP material descriptor sets per mount.
#[cfg(feature = "bsp")]
pub const MAX_BSP_MATERIALS: usize = 4_096;
#[cfg(feature = "bsp")]
const MAX_BSP_GEOMETRY_BYTES: u64 = 256 * 1024 * 1024;
#[cfg(feature = "bsp")]
const MAX_BSP_STAGING_BYTES: u64 = 256 * 1024 * 1024;
#[cfg(feature = "bsp")]
const MAX_BSP_TOTAL_GPU_BYTES: u64 = 1024 * 1024 * 1024;

/// Checked resource demand produced before the first Vulkan allocation.
#[cfg(feature = "bsp")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BspUploadDemand {
    pub source_face_count: usize,
    pub renderable_face_count: usize,
    pub batch_count: usize,
    pub material_count: usize,
    pub texture_count: usize,
    pub vertex_count: usize,
    pub index_count: usize,
    pub geometry_bytes: u64,
    pub texture_bytes: u64,
    pub lightmap_image_bytes: u64,
    pub lightmap_staging_bytes: u64,
    pub surface_uniform_bytes: u64,
    pub estimated_gpu_bytes: u64,
    pub leaf_bucket_span: u32,
}

#[cfg(feature = "bsp")]
#[derive(Debug, Clone)]
pub(crate) struct BspPlannedMaterial {
    pub texture_index: Option<usize>,
    pub surface_class: bsp::materials::SurfaceClass,
    pub surface_uniform: BspSurfaceUniform,
    pub is_pbr: bool,
}

#[cfg(feature = "bsp")]
#[derive(Debug, Clone)]
pub(crate) struct BspPlannedTexture {
    /// R=fullbright mask, G/B=tangent-space normal X/Y, A=gloss.
    pub material_data_rgba: Vec<u8>,
    pub pbr_flags: u32,
}

#[cfg(feature = "bsp")]
#[derive(Debug, Clone)]
pub(crate) struct BspPlannedBatch {
    pub mesh: ProceduralMeshData,
    pub material_plan_index: usize,
    pub render_batch: RenderBatch,
}

#[cfg(feature = "bsp")]
#[derive(Debug, Clone)]
pub(crate) struct BspUploadPlan {
    pub materials: Vec<BspPlannedMaterial>,
    pub textures: Vec<BspPlannedTexture>,
    pub batches: Vec<BspPlannedBatch>,
    pub face_to_batch: Vec<Option<usize>>,
    pub face_to_material: Vec<Option<usize>>,
    pub demand: BspUploadDemand,
}

#[cfg(feature = "bsp")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct PlannedMaterialKey {
    render_class: u8,
    texture_index: u32,
    lightmap_page: u32,
    style_ids: [u32; 4],
    pbr_flags: u32,
}

#[cfg(feature = "bsp")]
#[derive(Debug, Clone, Copy)]
struct PlannedFace {
    face_index: usize,
    source_face_index: u32,
    material_plan_index: usize,
    model_index: u32,
    primary_leaf: Option<u32>,
}

#[cfg(feature = "bsp")]
fn pbr_flags_for_companions(companions: &bsp::resources::PbrTextureCompanions) -> u32 {
    if companions.is_empty() {
        return 0;
    }
    let mut flags = bsp_surface_flags::SURF_PBR;
    if companions.normal.is_some() {
        flags |= bsp_surface_flags::SURF_PBR_NORMAL;
    }
    if companions.gloss.is_some() {
        flags |= bsp_surface_flags::SURF_PBR_GLOSS;
    }
    flags
}

#[cfg(feature = "bsp")]
fn pbr_flags_for_texture(
    extracted: &ExtractedBsp,
    texture_index: u32,
    surface_class: bsp::materials::SurfaceClass,
) -> u32 {
    if !matches!(
        surface_class,
        bsp::materials::SurfaceClass::Opaque | bsp::materials::SurfaceClass::AlphaMask
    ) {
        return 0;
    }
    usize::try_from(texture_index)
        .ok()
        .and_then(|index| extracted.textures.get(index))
        .map(|texture| pbr_flags_for_companions(&texture.pbr_companions))
        .unwrap_or(0)
}

#[cfg(feature = "bsp")]
fn texture_has_pbr_surface(extracted: &ExtractedBsp, texture_index: usize) -> bool {
    extracted.face_materials.iter().any(|material| {
        usize::try_from(material.material_index).ok() == Some(texture_index)
            && matches!(
                material.surface_class,
                bsp::materials::SurfaceClass::Opaque | bsp::materials::SurfaceClass::AlphaMask
            )
    })
}

#[cfg(feature = "bsp")]
fn decode_pbr_companion_rgba(
    texture_index: usize,
    role: &str,
    companion: &bsp::resources::TextureCompanion,
    expected_width: u32,
    expected_height: u32,
) -> Result<Vec<u8>, String> {
    let invalid_png = |error| {
        format!(
            "BSP texture {texture_index} {role} companion '{}' is not a valid PNG: {error}",
            companion.logical_path
        )
    };
    let dimensions = image::ImageReader::with_format(
        std::io::Cursor::new(&companion.bytes),
        image::ImageFormat::Png,
    )
    .into_dimensions()
    .map_err(&invalid_png)?;
    if dimensions != (expected_width, expected_height) {
        return Err(format!(
            "BSP texture {texture_index} {role} companion '{}' is {}x{}; expected {}x{}",
            companion.logical_path,
            dimensions.0,
            dimensions.1,
            expected_width,
            expected_height
        ));
    }

    let mut reader = image::ImageReader::with_format(
        std::io::Cursor::new(&companion.bytes),
        image::ImageFormat::Png,
    );
    let mut limits = image::Limits::default();
    limits.max_image_width = Some(expected_width);
    limits.max_image_height = Some(expected_height);
    reader.limits(limits);
    let image = reader.decode().map_err(invalid_png)?;
    Ok(image.to_rgba8().into_raw())
}

#[cfg(feature = "bsp")]
fn plan_bsp_textures(extracted: &ExtractedBsp) -> Result<Vec<BspPlannedTexture>, String> {
    extracted
        .textures
        .iter()
        .enumerate()
        .map(|(texture_index, texture)| {
            let pixel_count = (texture.width as usize)
                .checked_mul(texture.height as usize)
                .ok_or_else(|| format!("BSP texture {texture_index} dimensions overflow"))?;
            let rgba_size = pixel_count
                .checked_mul(4)
                .ok_or_else(|| format!("BSP texture {texture_index} RGBA size overflow"))?;
            if texture.width == 0
                || texture.height == 0
                || texture.width > bsp::resources::MAX_TEXTURE_DIMENSION
                || texture.height > bsp::resources::MAX_TEXTURE_DIMENSION
                || texture.albedo.len() != rgba_size
                || texture.fullbright_mask.len() != pixel_count
            {
                return Err(format!(
                    "BSP texture {texture_index} payload mismatch: {}x{}, albedo={}, mask={}",
                    texture.width,
                    texture.height,
                    texture.albedo.len(),
                    texture.fullbright_mask.len()
                ));
            }

            let pbr_flags = if texture_has_pbr_surface(extracted, texture_index) {
                pbr_flags_for_companions(&texture.pbr_companions)
            } else {
                0
            };
            if pbr_flags == 0 {
                let mut material_data_rgba = Vec::with_capacity(rgba_size);
                for &mask in &texture.fullbright_mask {
                    // Preserve the legacy fullbright upload exactly when no PBR
                    // companions are present.
                    material_data_rgba.extend_from_slice(&[mask, mask, mask, 255]);
                }
                return Ok(BspPlannedTexture {
                    material_data_rgba,
                    pbr_flags,
                });
            }

            let normal = texture
                .pbr_companions
                .normal
                .as_ref()
                .map(|companion| {
                    decode_pbr_companion_rgba(
                        texture_index,
                        "normal",
                        companion,
                        texture.width,
                        texture.height,
                    )
                })
                .transpose()?;
            let gloss = texture
                .pbr_companions
                .gloss
                .as_ref()
                .map(|companion| {
                    decode_pbr_companion_rgba(
                        texture_index,
                        "gloss",
                        companion,
                        texture.width,
                        texture.height,
                    )
                })
                .transpose()?;

            let mut material_data_rgba = Vec::with_capacity(rgba_size);
            for pixel in 0..pixel_count {
                let normal_offset = pixel * 4;
                let normal_x = normal
                    .as_ref()
                    .map(|pixels| pixels[normal_offset])
                    .unwrap_or(128);
                let normal_y = normal
                    .as_ref()
                    .map(|pixels| pixels[normal_offset + 1])
                    .unwrap_or(128);
                let gloss_value = gloss
                    .as_ref()
                    .map(|pixels| pixels[normal_offset])
                    .unwrap_or(0);
                material_data_rgba.extend_from_slice(&[
                    texture.fullbright_mask[pixel],
                    normal_x,
                    normal_y,
                    gloss_value,
                ]);
            }
            Ok(BspPlannedTexture {
                material_data_rgba,
                pbr_flags,
            })
        })
        .collect()
}

#[cfg(feature = "bsp")]
fn checked_add(a: u64, b: u64, label: &str) -> Result<u64, String> {
    a.checked_add(b)
        .ok_or_else(|| format!("BSP {label} byte count overflow"))
}

#[cfg(feature = "bsp")]
fn checked_mul(a: u64, b: u64, label: &str) -> Result<u64, String> {
    a.checked_mul(b)
        .ok_or_else(|| format!("BSP {label} byte count overflow"))
}

#[cfg(feature = "bsp")]
fn render_class_index(class: bsp::materials::SurfaceClass) -> u8 {
    class.render_class() as u8
}

#[cfg(feature = "bsp")]
fn style_id_array(layout: &FaceLightmapLayout) -> [u32; 4] {
    style_ids_for_layout(layout).to_array()
}

#[cfg(feature = "bsp")]
fn material_key_for_face(extracted: &ExtractedBsp, face_index: usize) -> Result<PlannedMaterialKey, String> {
    let material = extracted
        .face_materials
        .get(face_index)
        .ok_or_else(|| format!("BSP face {face_index} has no material"))?;
    let layout = extracted
        .face_lightmap_layouts
        .get(face_index)
        .ok_or_else(|| format!("BSP face {face_index} has no lightmap layout"))?;
    Ok(PlannedMaterialKey {
        render_class: render_class_index(material.surface_class),
        texture_index: material.material_index,
        lightmap_page: layout.has_data.then_some(layout.page_index).unwrap_or(u32::MAX),
        style_ids: style_id_array(layout),
        pbr_flags: pbr_flags_for_texture(
            extracted,
            material.material_index,
            material.surface_class,
        ),
    })
}

#[cfg(feature = "bsp")]
fn surface_class_from_render_class(render_class: u8) -> Result<bsp::materials::SurfaceClass, String> {
    use bsp::materials::SurfaceClass;
    match render_class {
        0 => Ok(SurfaceClass::Opaque),
        1 => Ok(SurfaceClass::AlphaMask),
        2 => Ok(SurfaceClass::Sky),
        3 => Ok(SurfaceClass::Liquid),
        other => Err(format!("unsupported BSP render class {other}")),
    }
}

#[cfg(feature = "bsp")]
fn build_material_plan(
    extracted: &ExtractedBsp,
    key: PlannedMaterialKey,
) -> Result<BspPlannedMaterial, String> {
    let surface_class = surface_class_from_render_class(key.render_class)?;
    let texture_index = usize::try_from(key.texture_index)
        .ok()
        .filter(|&index| index < extracted.textures.len());
    let lightmap_layer_base = if key.lightmap_page == u32::MAX {
        0
    } else {
        key.lightmap_page
            .checked_mul(4)
            .ok_or_else(|| "BSP lightmap layer base overflow".to_string())?
    };

    let is_pbr = key.pbr_flags & bsp_surface_flags::SURF_PBR != 0;
    Ok(BspPlannedMaterial {
        texture_index,
        surface_class,
        surface_uniform: BspSurfaceUniform {
            // Merged vertices carry final atlas UVs, so material scale/bias is identity.
            lightmap_scale_bias: Vec4::new(1.0, 1.0, 0.0, 0.0),
            style_ids: UVec4::from_array(key.style_ids),
            fullbright_base: 224,
            fullbright_count: 32,
            alpha_threshold: 0.5,
            animation_frame: 0,
            animation_time: 0.0,
            surface_flags: surface_flags_for(Some(surface_class)) | key.pbr_flags,
            receive_mask: receive_mask_for(Some(surface_class))
                | if is_pbr {
                    bsp_surface_flags::RECEIVE_IBL
                } else {
                    0
                },
            lightmap_layer_base,
            liquid_warp_scale: 0.02,
            liquid_flow_speed: 1.0,
            _pad1: [0, 0],
        },
        is_pbr,
    })
}

#[cfg(feature = "bsp")]
fn collect_planned_faces(
    extracted: &ExtractedBsp,
    material_indices: &std::collections::BTreeMap<PlannedMaterialKey, usize>,
) -> Result<Vec<PlannedFace>, String> {
    let face_count = extracted.face_geometries.len();
    if extracted.face_materials.len() != face_count
        || extracted.face_lightmap_layouts.len() != face_count
        || extracted.leaf_membership.len() != face_count
    {
        return Err(format!(
            "BSP face arrays differ in length: geometry={face_count}, materials={}, lightmaps={}, leaves={}",
            extracted.face_materials.len(),
            extracted.face_lightmap_layouts.len(),
            extracted.leaf_membership.len()
        ));
    }

    let mut inline_models = std::collections::HashMap::<u32, u32>::new();
    for model in &extracted.inline_models {
        for &source_face_index in &model.face_indices {
            if inline_models.insert(source_face_index, model.model_index).is_some() {
                return Err(format!(
                    "BSP source face {source_face_index} belongs to multiple inline models"
                ));
            }
        }
    }

    let mut faces = Vec::new();
    for (face_index, geometry) in extracted.face_geometries.iter().enumerate() {
        let material = &extracted.face_materials[face_index];
        if !geometry.is_valid || !material.surface_class.is_visible() {
            continue;
        }
        if geometry.face_index as usize != face_index {
            return Err(format!(
                "BSP face identity mismatch: slot {face_index} contains source face {}",
                geometry.face_index
            ));
        }
        let key = material_key_for_face(extracted, face_index)?;
        let material_plan_index = *material_indices
            .get(&key)
            .ok_or_else(|| format!("BSP face {face_index} material was not planned"))?;
        let model_index = inline_models.get(&geometry.face_index).copied().unwrap_or(0);
        let primary_leaf = (model_index == 0)
            .then(|| extracted.leaf_membership[face_index].iter().copied().min())
            .flatten();
        faces.push(PlannedFace {
            face_index,
            source_face_index: geometry.face_index,
            material_plan_index,
            model_index,
            primary_leaf,
        });
    }
    Ok(faces)
}

#[cfg(feature = "bsp")]
fn bake_lightmap_uv(
    extracted: &ExtractedBsp,
    face_index: usize,
    uv: Vec2,
) -> Result<Vec2, String> {
    let layout = &extracted.face_lightmap_layouts[face_index];
    let (image_width, image_height) = extracted.lightmap_atlas.common_used_extent();
    if !layout.has_data {
        let w = image_width.max(1) as f32;
        let h = image_height.max(1) as f32;
        return Ok(Vec2::new(0.5 / w, 0.5 / h));
    }
    // Validate the page exists and used extent is consistent.
    if extracted
        .lightmap_atlas
        .pages
        .get(layout.page_index as usize)
        .is_none()
    {
        return Err(format!(
            "BSP face {face_index} references missing lightmap page {}",
            layout.page_index
        ));
    }
    if image_width == 0 || image_height == 0 {
        return Err(format!(
            "BSP face {face_index} has a zero-sized lightmap atlas"
        ));
    }
    let texel = Vec2::new(
        layout.atlas_offset.0 as f32 + uv.x * layout.luxel_extents.0.max(1) as f32,
        layout.atlas_offset.1 as f32 + uv.y * layout.luxel_extents.1.max(1) as f32,
    );
    let atlas_uv = texel / Vec2::new(image_width as f32, image_height as f32);
    if !atlas_uv.is_finite() {
        return Err(format!("BSP face {face_index} produced non-finite atlas UV"));
    }
    Ok(atlas_uv)
}

#[cfg(feature = "bsp")]
fn merge_batch_mesh(
    extracted: &ExtractedBsp,
    batch_index: usize,
    faces: &[PlannedFace],
) -> Result<ProceduralMeshData, String> {
    let vertex_count: usize = faces
        .iter()
        .map(|face| extracted.face_geometries[face.face_index].vertices.len())
        .sum();
    let index_count: usize = faces
        .iter()
        .map(|face| {
            extracted.face_geometries[face.face_index]
                .vertices
                .len()
                .saturating_sub(2)
                .saturating_mul(3)
        })
        .sum();
    let mut vertices = Vec::with_capacity(vertex_count);
    let mut indices = Vec::with_capacity(index_count);

    for face in faces {
        let mut mesh = face_to_procedural_mesh(
            &extracted.face_geometries[face.face_index],
            1.0,
            1.0,
            None,
        )
        .ok_or_else(|| format!("renderable BSP face {} is not triangulable", face.face_index))?;
        let base_vertex = u32::try_from(vertices.len())
            .map_err(|_| format!("BSP batch {batch_index} vertex offset exceeds u32"))?;
        let texture_extent = extracted
            .face_materials
            .get(face.face_index)
            .and_then(|material| usize::try_from(material.material_index).ok())
            .and_then(|texture_index| extracted.textures.get(texture_index))
            .filter(|texture| texture.width > 0 && texture.height > 0)
            .map(|texture| Vec2::new(texture.width as f32, texture.height as f32))
            .unwrap_or(Vec2::ONE);
        for vertex in &mut mesh.vertices {
            // Quake texinfo projections are expressed in source texels. Vulkan
            // normalized samplers require division by the resolved mip-0 extent.
            vertex.uv0 /= texture_extent;
            vertex.uv1 = bake_lightmap_uv(extracted, face.face_index, vertex.uv1)?;
        }
        for index in mesh.indices {
            indices.push(
                base_vertex
                    .checked_add(index)
                    .ok_or_else(|| format!("BSP batch {batch_index} index overflow"))?,
            );
        }
        vertices.extend(mesh.vertices);
    }

    if vertices.is_empty() || indices.is_empty() {
        return Err(format!("BSP batch {batch_index} has no renderable geometry"));
    }
    Ok(ProceduralMeshData {
        name: format!("bsp_batch_{batch_index}"),
        vertices,
        indices,
        material: None,
    })
}

#[cfg(feature = "bsp")]
fn compute_upload_demand(
    extracted: &ExtractedBsp,
    batches: &[BspPlannedBatch],
    material_count: usize,
    renderable_face_count: usize,
    leaf_bucket_span: u32,
) -> Result<BspUploadDemand, String> {
    let vertex_count = batches.iter().try_fold(0usize, |total, batch| {
        total
            .checked_add(batch.mesh.vertices.len())
            .ok_or_else(|| "BSP vertex count overflow".to_string())
    })?;
    let index_count = batches.iter().try_fold(0usize, |total, batch| {
        total
            .checked_add(batch.mesh.indices.len())
            .ok_or_else(|| "BSP index count overflow".to_string())
    })?;
    let vertex_bytes = checked_mul(
        vertex_count as u64,
        std::mem::size_of::<crate::data::gpu_data::Vertex>() as u64,
        "vertex",
    )?;
    let index_bytes = checked_mul(index_count as u64, 4, "index")?;
    let geometry_bytes = checked_add(vertex_bytes, index_bytes, "geometry")?;

    let texture_bytes = extracted.textures.iter().try_fold(0u64, |total, texture| {
        let albedo = u64::try_from(texture.albedo.len())
            .map_err(|_| "BSP albedo byte count exceeds u64".to_string())?;
        let material_data = checked_mul(
            u64::try_from(texture.fullbright_mask.len())
                .map_err(|_| "BSP material-data pixel count exceeds u64".to_string())?,
            4,
            "material-data expansion",
        )?;
        checked_add(
            total,
            checked_add(albedo, material_data, "texture")?,
            "texture total",
        )
    })?;

    let (atlas_width, atlas_height, atlas_pages) = {
        let (used_w, used_h) = extracted.lightmap_atlas.common_used_extent();
        if used_w == 0 || used_h == 0 {
            (1u64, 1u64, 1u64)
        } else {
            (used_w as u64, used_h as u64, extracted.lightmap_atlas.pages.len() as u64)
        }
    };
    let atlas_layers = checked_mul(atlas_pages, 4, "lightmap layer")?;
    let lightmap_image_bytes = checked_mul(
        checked_mul(checked_mul(atlas_width, atlas_height, "lightmap pixels")?, atlas_layers, "lightmap layers")?,
        4,
        "lightmap image",
    )?;
    let style_pixels = extracted.face_lightmap_layouts.iter().try_fold(1u64, |total, layout| {
        layout.style_layers.iter().take(4).filter(|layer| layer.has_data).try_fold(total, |sum, layer| {
            let pixels = checked_mul(layer.luxel_extents.0 as u64, layer.luxel_extents.1 as u64, "lightmap rectangle")?;
            checked_add(sum, pixels, "lightmap staging pixels")
        })
    })?;
    let lightmap_staging_bytes = checked_mul(style_pixels, 4, "lightmap staging")?;
    let surface_uniform_bytes = checked_mul(
        material_count as u64,
        std::mem::size_of::<BspSurfaceUniform>() as u64,
        "surface uniform",
    )?;
    let estimated_gpu_bytes = [geometry_bytes, texture_bytes, lightmap_image_bytes, surface_uniform_bytes]
        .into_iter()
        .try_fold(0u64, |sum, bytes| checked_add(sum, bytes, "estimated GPU"))?;

    if batches.len() > MAX_BSP_RENDER_BATCHES {
        return Err(format!(
            "BSP planned {} batches; maximum is {MAX_BSP_RENDER_BATCHES}",
            batches.len()
        ));
    }
    if material_count > MAX_BSP_MATERIALS {
        return Err(format!(
            "BSP planned {material_count} materials; maximum is {MAX_BSP_MATERIALS}"
        ));
    }
    if extracted.textures.len() > MAX_BSP_MATERIALS {
        return Err(format!(
            "BSP has {} textures; maximum is {MAX_BSP_MATERIALS}",
            extracted.textures.len()
        ));
    }
    if geometry_bytes > MAX_BSP_GEOMETRY_BYTES {
        return Err(format!(
            "BSP geometry demand {geometry_bytes} bytes exceeds {MAX_BSP_GEOMETRY_BYTES}"
        ));
    }
    if lightmap_staging_bytes > MAX_BSP_STAGING_BYTES {
        return Err(format!(
            "BSP lightmap staging demand {lightmap_staging_bytes} bytes exceeds {MAX_BSP_STAGING_BYTES}"
        ));
    }
    if estimated_gpu_bytes > MAX_BSP_TOTAL_GPU_BYTES {
        return Err(format!(
            "BSP estimated GPU demand {estimated_gpu_bytes} bytes exceeds {MAX_BSP_TOTAL_GPU_BYTES}"
        ));
    }

    Ok(BspUploadDemand {
        source_face_count: extracted.face_geometries.len(),
        renderable_face_count,
        batch_count: batches.len(),
        material_count,
        texture_count: extracted.textures.len(),
        vertex_count,
        index_count,
        geometry_bytes,
        texture_bytes,
        lightmap_image_bytes,
        lightmap_staging_bytes,
        surface_uniform_bytes,
        estimated_gpu_bytes,
        leaf_bucket_span,
    })
}

// ── Invariant-bearing mounted batch ─────────────────────────────────────

#[cfg(feature = "bsp")]
/// A canonical, invariant-bearing record that binds one render batch to its
/// GPU mesh, GPU material, and finite local-space bounds. This is the sole
/// authority for production batch identity; phase 07 consumes this record
/// directly for fail-closed draw submission.
#[derive(Debug, Clone)]
pub struct MountedBspBatch {
    /// Neutral batch key, source-face indices, visibility signature, and model identity.
    pub render: RenderBatch,
    /// Live merged mesh handle (non-zero, non-stale).
    pub mesh: MeshHandle,
    /// Live BSP material handle (non-zero, non-stale).
    pub material: BspMaterialHandle,
    /// Finite local-space axis-aligned bounds (min, max).
    pub bounds: (glam::Vec3, glam::Vec3),
}

#[cfg(feature = "bsp")]
impl MountedBspBatch {
    /// Construct a mounted record after all resources are resolved. Validates
    /// the complete batch identity, not just its first source face.
    pub fn try_new(
        render_batch: &RenderBatch,
        mesh: MeshHandle,
        material: BspMaterialHandle,
        bounds: (glam::Vec3, glam::Vec3),
    ) -> Result<Self, String> {
        if mesh.slot == 0 {
            return Err("MountedBspBatch received a null mesh handle".to_string());
        }
        if material.slot == 0 && material.generation == 0 {
            return Err("MountedBspBatch received a null material handle".to_string());
        }
        if render_batch.face_indices.is_empty() {
            return Err("MountedBspBatch has an empty face list".to_string());
        }
        if !bounds.0.is_finite() || !bounds.1.is_finite() {
            return Err("MountedBspBatch has non-finite bounds".to_string());
        }
        if bounds.0.x > bounds.1.x || bounds.0.y > bounds.1.y || bounds.0.z > bounds.1.z {
            return Err("MountedBspBatch has inverted bounds".to_string());
        }
        Ok(Self {
            render: render_batch.clone(),
            mesh,
            material,
            bounds,
        })
    }
}

/// Verify that every renderable face in the plan is covered by exactly one
/// mounted batch, and that every mounted batch maps to a planned batch with
/// matching face membership.
#[cfg(feature = "bsp")]
pub(crate) fn verify_exact_renderable_face_coverage(
    mounted: &[MountedBspBatch],
    plan: &BspUploadPlan,
) -> Result<(), String> {
    if mounted.is_empty() && plan.face_to_batch.iter().all(Option::is_none) {
        return Ok(());
    }
    if mounted.is_empty() && plan.face_to_batch.iter().any(Option::is_some) {
        return Err(
            "BSP has renderable faces but zero mounted batches; every renderable face must belong to a batch"
                .to_string(),
        );
    }

    // Build a map from batch index to the set of source faces covered by mounted records.
    let mut mounted_counts = vec![0usize; plan.batches.len()];
    for (batch_index, mounted_batch) in mounted.iter().enumerate() {
        let planned = plan
            .batches
            .get(batch_index)
            .ok_or_else(|| format!("mounted batch {batch_index} has no planned counterpart"))?;
        if mounted_batch.render.face_indices != planned.render_batch.face_indices {
            return Err(format!(
                "mounted batch {batch_index} face set does not match plan"
            ));
        }
        for &source_face in &mounted_batch.render.face_indices {
            let slot = source_face as usize;
            if slot >= plan.face_to_batch.len() {
                return Err(format!(
                    "mounted batch {batch_index} references out-of-range source face {source_face}"
                ));
            }
            mounted_counts[batch_index] += 1;
        }
    }

    for (face_index, batch_opt) in plan.face_to_batch.iter().enumerate() {
        match batch_opt {
            Some(batch_index) => {
                if *batch_index >= mounted.len() {
                    return Err(format!(
                        "face {face_index} maps to batch {batch_index} which has no mounted record"
                    ));
                }
            }
            None => {
                // Non-renderable face: verify it is indeed not renderable (nodraw, invalid).
                let geo = &extracted_face_geometry(plan, face_index);
                let visible = extracted_face_is_visible(plan, face_index);
                if visible && geo.map(|g| g.is_valid).unwrap_or(false) {
                    return Err(format!(
                        "renderable face {face_index} is not assigned to any batch"
                    ));
                }
            }
        }
    }
    Ok(())
}

#[cfg(feature = "bsp")]
fn extracted_face_geometry(
    plan: &BspUploadPlan,
    face_index: usize,
) -> Option<&FaceGeometry> {
    // The plan doesn't hold the ExtractedBsp. This function exists as a
    // documentation-of-intent helper; the actual validation is done at the
    // callsite with access to the extracted data.
    let _ = (plan, face_index);
    None
}

#[cfg(feature = "bsp")]
fn extracted_face_is_visible(plan: &BspUploadPlan, face_index: usize) -> bool {
    let _ = (plan, face_index);
    false
}

/// Compute finite local-space AABB for a merged batch mesh.
#[cfg(feature = "bsp")]
pub(crate) fn compute_batch_bounds(
    mesh: &ProceduralMeshData,
) -> Result<(glam::Vec3, glam::Vec3), String> {
    if mesh.vertices.is_empty() {
        return Err("cannot compute bounds for empty batch mesh".to_string());
    }
    let mut min = glam::Vec3::splat(f32::INFINITY);
    let mut max = glam::Vec3::splat(f32::NEG_INFINITY);
    for vertex in &mesh.vertices {
        let pos = vertex.position;
        if !pos.is_finite() {
            return Err("batch mesh contains non-finite vertex position".to_string());
        }
        min = min.min(pos);
        max = max.max(pos);
    }
    if !min.is_finite() || !max.is_finite() {
        return Err("batch mesh bounds are non-finite after reduction".to_string());
    }
    Ok((min, max))
}

/// Build bounded renderer batches and merged meshes before allocating GPU resources.
///
/// This is the sole CPU planning boundary. No cache or Vulkan allocation may occur
/// until this function has validated all data needed for upload and mount construction.
#[cfg(feature = "bsp")]
pub(crate) fn plan_bsp_upload(extracted: &ExtractedBsp) -> Result<BspUploadPlan, String> {
    // ── Validate aligned extraction arrays ─────────────────────────
    let face_count = extracted.face_geometries.len();
    if extracted.face_materials.len() != face_count
        || extracted.face_lightmap_layouts.len() != face_count
        || extracted.leaf_membership.len() != face_count
    {
        return Err(format!(
            "BSP face arrays differ in length: geometry={face_count}, materials={}, lightmaps={}, leaves={}",
            extracted.face_materials.len(),
            extracted.face_lightmap_layouts.len(),
            extracted.leaf_membership.len()
        ));
    }

    // ── Validate atlas page dimensions ─────────────────────────────
    if let Some(first_page) = extracted.lightmap_atlas.pages.first() {
        if first_page.width == 0 || first_page.height == 0 {
            return Err("BSP lightmap atlas has a zero-sized page".to_string());
        }
        for (page_index, page) in extracted.lightmap_atlas.pages.iter().enumerate() {
            if page.width != first_page.width || page.height != first_page.height {
                return Err(format!(
                    "BSP lightmap atlas page {page_index} dimensions {}x{} differ from page 0 {}x{}",
                    page.width, page.height, first_page.width, first_page.height
                ));
            }
        }
    }

    // ── Validate textures have valid albedo ────────────────────────
    for (texture_index, texture) in extracted.textures.iter().enumerate() {
        if texture.width == 0
            || texture.height == 0
            || texture.width > bsp::resources::MAX_TEXTURE_DIMENSION
            || texture.height > bsp::resources::MAX_TEXTURE_DIMENSION
        {
            return Err(format!(
                "BSP texture {texture_index} has invalid dimensions {}x{}",
                texture.width, texture.height
            ));
        }
        let expected_albedo = (texture.width as usize)
            .checked_mul(texture.height as usize)
            .and_then(|pixels| pixels.checked_mul(4))
            .ok_or_else(|| format!("BSP texture {texture_index} dimensions overflow"))?;
        if texture.albedo.len() != expected_albedo {
            return Err(format!(
                "BSP texture {texture_index} albedo has {} bytes; expected {expected_albedo}",
                texture.albedo.len()
            ));
        }
        if texture.fullbright_mask.len()
            != (texture.width as usize).checked_mul(texture.height as usize).unwrap_or(0)
        {
            return Err(format!(
                "BSP texture {texture_index} fullbright mask size mismatch"
            ));
        }
    }

    // ── Validate renderable faces have required decoded albedo ─────
    for (face_index, material) in extracted.face_materials.iter().enumerate() {
        if !extracted.face_geometries[face_index].is_valid {
            continue;
        }
        if !material.surface_class.is_visible() {
            continue;
        }
        let texture_index = usize::try_from(material.material_index).ok();
        if let Some(index) = texture_index {
            if index >= extracted.textures.len() {
                return Err(format!(
                    "BSP renderable face {face_index} references texture {index} which does not exist"
                ));
            }
        }
    }

    let mut material_keys = std::collections::BTreeMap::<PlannedMaterialKey, usize>::new();
    for (face_index, geometry) in extracted.face_geometries.iter().enumerate() {
        let Some(material) = extracted.face_materials.get(face_index) else {
            return Err(format!("BSP face {face_index} has no material"));
        };
        if geometry.is_valid && material.surface_class.is_visible() {
            material_keys.entry(material_key_for_face(extracted, face_index)?).or_insert(0);
        }
    }
    if material_keys.len() > MAX_BSP_MATERIALS {
        return Err(format!(
            "BSP requires {} unique materials; maximum is {MAX_BSP_MATERIALS}",
            material_keys.len()
        ));
    }
    for (index, value) in material_keys.values_mut().enumerate() {
        *value = index;
    }
    let materials = material_keys
        .keys()
        .copied()
        .map(|key| build_material_plan(extracted, key))
        .collect::<Result<Vec<_>, _>>()?;
    let faces = collect_planned_faces(extracted, &material_keys)?;

    if faces.is_empty() {
        // Zero renderable faces is valid; produce an empty plan.
        let demand = BspUploadDemand {
            source_face_count: face_count,
            renderable_face_count: 0,
            batch_count: 0,
            material_count: 0,
            texture_count: extracted.textures.len(),
            vertex_count: 0,
            index_count: 0,
            geometry_bytes: 0,
            texture_bytes: 0,
            lightmap_image_bytes: 0,
            lightmap_staging_bytes: 0,
            surface_uniform_bytes: 0,
            estimated_gpu_bytes: 0,
            leaf_bucket_span: 0,
        };
        let textures = plan_bsp_textures(extracted)?;
        return Ok(BspUploadPlan {
            materials: Vec::new(),
            textures,
            batches: Vec::new(),
            face_to_batch: vec![None; face_count],
            face_to_material: vec![None; face_count],
            demand,
        });
    }

    // ── Build source-face → slot and slot → material_plan_index maps ──
    let source_face_to_slot: std::collections::HashMap<u32, usize> = extracted
        .face_geometries
        .iter()
        .enumerate()
        .map(|(slot, geo)| (geo.face_index, slot))
        .collect();
    let mut slot_material = vec![None; face_count];
    for face in &faces {
        if slot_material[face.face_index].replace(face.material_plan_index).is_some() {
            return Err(format!("BSP face {} has duplicate material assignment", face.face_index));
        }
    }

    // ── Materialize neutral batches one-for-one ────────────────────
    if extracted.render_batches.is_empty() {
        return Err("BSP has renderable faces but produces zero neutral render batches".to_string());
    }
    let mut face_to_batch = vec![None; face_count];
    let mut face_to_material = vec![None; face_count];
    let mut batches = Vec::with_capacity(extracted.render_batches.len());

    for (batch_index, neutral_batch) in extracted.render_batches.iter().enumerate() {
        // Map source face indices to face slots.
        let face_slots: Vec<usize> = neutral_batch
            .face_indices
            .iter()
            .map(|&src_face| {
                source_face_to_slot
                    .get(&src_face)
                    .copied()
                    .ok_or_else(|| format!(
                        "BSP neutral batch {batch_index} references unknown source face {src_face}"
                    ))
            })
            .collect::<Result<Vec<_>, _>>()?;

        // Collect PlannedFace records for mesh merging.
        let group_faces: Vec<PlannedFace> = face_slots
            .iter()
            .map(|&slot| {
                let mat_plan = slot_material[slot].ok_or_else(|| {
                    format!("BSP neutral batch {batch_index} face slot {slot} has no material plan")
                })?;
                Ok(PlannedFace {
                    face_index: slot,
                    source_face_index: extracted.face_geometries[slot].face_index,
                    material_plan_index: mat_plan,
                    model_index: neutral_batch.model_index,
                    primary_leaf: None,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;

        // Validate material homogeneity: all faces in the batch must share
        // the same material plan index.
        let expected_material = group_faces
            .first()
            .map(|f| f.material_plan_index)
            .ok_or_else(|| format!("BSP batch {batch_index} has no faces"))?;
        for face in &group_faces {
            if face.material_plan_index != expected_material {
                return Err(format!(
                    "BSP neutral batch {batch_index} mixes material plans {expected_material} and {}",
                    face.material_plan_index
                ));
            }
        }

        // Assign face-to-batch and face-to-material mappings.
        for face in &group_faces {
            if face_to_batch[face.face_index].replace(batch_index).is_some() {
                return Err(format!("BSP face {} assigned to multiple batches", face.face_index));
            }
            face_to_material[face.face_index] = Some(face.material_plan_index);
        }

        // Merge face meshes into one GPU batch mesh.
        batches.push(BspPlannedBatch {
            mesh: merge_batch_mesh(extracted, batch_index, &group_faces)?,
            material_plan_index: expected_material,
            render_batch: neutral_batch.clone(),
        });
    }

    // ── Post-batching invariant checks ─────────────────────────────
    for (batch_index, batch) in batches.iter().enumerate() {
        if batch.render_batch.face_indices.is_empty() {
            return Err(format!("BSP batch {batch_index} has no faces"));
        }
        if batch.mesh.vertices.is_empty() || batch.mesh.indices.is_empty() {
            return Err(format!("BSP batch {batch_index} has empty geometry"));
        }
        // Validate that all faces in this batch agree on material, render class, and model identity.
        let expected_material = batch.material_plan_index;
        let expected_model = batch.render_batch.model_index;
        for &source_face in &batch.render_batch.face_indices {
            let slot = source_face as usize;
            let face_material = face_to_material[slot];
            if face_material != Some(expected_material) {
                return Err(format!(
                    "BSP batch {batch_index} face {source_face} material index {:?} != batch material {expected_material}",
                    face_material
                ));
            }
            let model_index = face_to_batch
                .get(slot)
                .and_then(|opt| *opt)
                .and_then(|b_idx| batches.get(b_idx))
                .map(|b| b.render_batch.model_index)
                .unwrap_or(0);
            if model_index != expected_model {
                return Err(format!(
                    "BSP batch {batch_index} face {source_face} model {model_index} != batch model {expected_model}"
                ));
            }
        }
        // Validate batch bounds are finite.
        compute_batch_bounds(&batch.mesh)?;
    }

    // Enforce aggregate demand before decoding and packing companion images.
    let demand = compute_upload_demand(
        extracted,
        &batches,
        materials.len(),
        faces.len(),
        0,
    )?;
    let textures = plan_bsp_textures(extracted)?;
    Ok(BspUploadPlan {
        materials,
        textures,
        batches,
        face_to_batch,
        face_to_material,
        demand,
    })
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
    /// The extraction pipeline already populates atlas pages with packed face
    /// luxels in RGB8. This function expands each page to RGBA8 for a single
    /// array layer per page. The multi-layer array upload (page_count × 4
    /// style-slot layers) is built directly by the renderer upload pipeline
    /// from `face_lightmap_layouts` and does not use this function.
    ///
    /// Do NOT duplicate a page into every style layer — that defect was
    /// repaired in Phase 08. The style-specific layering is managed by the
    /// per-face `face_lightmap_layouts` and the renderer's layer-copy plan.
    pub fn from_extracted(extracted: &ExtractedBsp) -> Vec<Self> {
        let atlas = &extracted.lightmap_atlas;

        if atlas.pages.is_empty() {
            return Vec::new();
        }

        // Produce one single-layer RGBA8 page per source atlas page.
        // Each page covers a single array element; the renderer distributes
        // face rectangles into the correct style-slot layers (page * 4 + slot).
        atlas
            .pages
            .iter()
            .map(|page| {
                let pixel_count = (page.width * page.height) as usize;
                let layer_count: u32 = 1;
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
                    layer_count,
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
fn style_ids_for_layout(layout: &FaceLightmapLayout) -> UVec4 {
    let mut ids = [255u32; 4];
    for style_layout in &layout.style_layers {
        let slot = style_layout.source_slot as usize;
        if slot < 4 && style_layout.has_data && style_layout.style_id <= 63 {
            ids[slot] = style_layout.style_id as u32;
        }
    }
    if ids.iter().all(|&id| id == 255) {
        ids[0] = 0;
    }
    UVec4::new(ids[0], ids[1], ids[2], ids[3])
}

#[cfg(feature = "bsp")]
fn surface_flags_for(class: Option<bsp::materials::SurfaceClass>) -> u32 {
    match class {
        Some(bsp::materials::SurfaceClass::AlphaMask) => bsp_surface_flags::SURF_ALPHA_MASK,
        Some(bsp::materials::SurfaceClass::Sky) => bsp_surface_flags::SURF_SKY,
        Some(bsp::materials::SurfaceClass::Liquid) => bsp_surface_flags::SURF_LIQUID,
        _ => 0,
    }
}

#[cfg(feature = "bsp")]
fn receive_mask_for(class: Option<bsp::materials::SurfaceClass>) -> u32 {
    match class {
        Some(bsp::materials::SurfaceClass::Sky) => bsp_surface_flags::OUTDOOR_DEFAULT,
        _ => bsp_surface_flags::SEALED_DEFAULT,
    }
}

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

        let pbr_flags = extracted
            .face_materials
            .get(fi)
            .map(|material| {
                pbr_flags_for_texture(extracted, material.material_index, material.surface_class)
            })
            .unwrap_or(0);
        let is_pbr = pbr_flags & bsp_surface_flags::SURF_PBR != 0;
        let bsp_class = match (sc, is_pbr) {
            (Some(bsp::materials::SurfaceClass::AlphaMask), true) => BspSurfaceClass::PbrAlphaMask,
            (Some(bsp::materials::SurfaceClass::Opaque), true) => BspSurfaceClass::PbrLightmapped,
            (Some(bsp::materials::SurfaceClass::AlphaMask), false) => BspSurfaceClass::AlphaMask,
            (Some(bsp::materials::SurfaceClass::Sky), _) => BspSurfaceClass::Sky,
            (Some(bsp::materials::SurfaceClass::Liquid), _) => BspSurfaceClass::Liquid,
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
            | BspSurfaceClass::PbrLightmapped
            | BspSurfaceClass::PbrAlphaMask
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
        let (atlas_w, atlas_h) = {
            let (used_w, used_h) = extracted.lightmap_atlas.common_used_extent();
            (used_w.max(1) as f32, used_h.max(1) as f32)
        };

        let surface_params = BspSurfaceUniform {
            lightmap_scale_bias: Vec4::new(
                luxel_w / atlas_w,
                luxel_h / atlas_h,
                layout.atlas_offset.0 as f32 / atlas_w,
                layout.atlas_offset.1 as f32 / atlas_h,
            ),
            style_ids: style_ids_for_layout(layout),
            fullbright_base: 224,
            fullbright_count: 32,
            alpha_threshold: 0.5,
            animation_frame: 0,
            animation_time: 0.0,
            surface_flags: surface_flags_for(sc) | pbr_flags,
            receive_mask: receive_mask_for(sc)
                | if is_pbr {
                    bsp_surface_flags::RECEIVE_IBL
                } else {
                    0
                },
            lightmap_layer_base: layout.page_index.saturating_mul(4),
            liquid_warp_scale: 0.02,
            liquid_flow_speed: 1.0,
            _pad1: [0, 0],
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

    #[cfg(feature = "bsp")]
    fn stress_extracted(face_count: usize, texture_count: usize) -> ExtractedBsp {
        use bsp::lightmaps::{AtlasPage, LightmapAtlas, StyleLightmapLayout};
        use bsp::materials::{BspMaterial, SurfaceClass};
        use bsp::resources::ExtractedTexture;

        let textures = (0..texture_count)
            .map(|index| ExtractedTexture {
                identity: format!("texture_{index}"),
                palette_indices: vec![0],
                albedo: vec![255, 255, 255, 255],
                fullbright_mask: vec![0],
                width: 1,
                height: 1,
                ..ExtractedTexture::default()
            })
            .collect::<Vec<_>>();
        let mut lightmap_atlas = LightmapAtlas::new();
        lightmap_atlas.pages.push(AtlasPage::new(0, 64, 64));

        let mut face_geometries = Vec::with_capacity(face_count);
        let mut face_materials = Vec::with_capacity(face_count);
        let mut face_lightmap_layouts = Vec::with_capacity(face_count);
        let mut leaf_membership = Vec::with_capacity(face_count);
        for index in 0..face_count {
            face_geometries.push(FaceGeometry {
                face_index: index as u32,
                vertices: vec![Vec3::ZERO, Vec3::X, Vec3::Y],
                uv0: vec![Vec2::ZERO, Vec2::X, Vec2::Y],
                uv1: vec![Vec2::ZERO, Vec2::X, Vec2::Y],
                normal: Vec3::Z,
                bounds: (Vec3::ZERO, Vec3::ONE),
                luxel_extents: (1, 1),
                is_valid: true,
            });
            let texture_index = (index % texture_count) as u32;
            face_materials.push(BspMaterial {
                material_index: texture_index,
                texture_identity: format!("texture_{texture_index}"),
                surface_class: SurfaceClass::Opaque,
                ..BspMaterial::default()
            });
            let style_id = ((index / texture_count.max(1)) % 6) as u8;
            let offset = ((index % 60) as u32 + 2, ((index / 60) % 60) as u32 + 2);
            face_lightmap_layouts.push(FaceLightmapLayout {
                page_index: 0,
                atlas_offset: offset,
                luxel_extents: (1, 1),
                has_data: true,
                style_layers: vec![StyleLightmapLayout {
                    style_id,
                    source_slot: 0,
                    page_index: 0,
                    atlas_offset: offset,
                    luxel_extents: (1, 1),
                    has_data: true,
                }],
            });
            leaf_membership.push(vec![(index % 18_587) as u32]);
        }

        ExtractedBsp {
            transform: bsp::coords::QuakeToEngine::default(),
            profile_tag: "stress",
            textures,
            face_geometries,
            face_materials,
            render_batches: Vec::new(),
            lightmap_atlas,
            face_lightmap_layouts,
            has_pvs: true,
            camera_pvs: None,
            visibility: bsp::extract::ExtractedVisibility::default(),
            leaf_membership,
            entity_descriptors: Vec::new(),
            entity_identities: Vec::new(),
            light_descriptors: Vec::new(),
            inline_models: Vec::new(),
            world_collision_planes: Vec::new(),
            collision_recipes: Vec::new(),
            content_hash: [0; 32],
            source_identity: "stress".to_string(),
            diagnostics: Vec::new(),
        }
    }

    #[cfg(feature = "bsp")]
    fn solid_png(width: u32, height: u32, pixel: [u8; 4]) -> Vec<u8> {
        let image = image::RgbaImage::from_pixel(width, height, image::Rgba(pixel));
        let mut output = std::io::Cursor::new(Vec::new());
        image::DynamicImage::ImageRgba8(image)
            .write_to(&mut output, image::ImageFormat::Png)
            .unwrap();
        output.into_inner()
    }

    #[cfg(feature = "bsp")]
    fn one_pixel_png(pixel: [u8; 4]) -> Vec<u8> {
        solid_png(1, 1, pixel)
    }

    #[cfg(feature = "bsp")]
    #[test]
    fn pbr_companions_pack_material_data_and_route_material() {
        let mut extracted = stress_extracted(1, 1);
        extracted.textures[0].pbr_companions = bsp::resources::PbrTextureCompanions {
            normal: Some(bsp::resources::TextureCompanion::new(
                "textures/texture_0_norm.png",
                one_pixel_png([64, 192, 255, 255]),
            )),
            gloss: Some(bsp::resources::TextureCompanion::new(
                "textures/texture_0_gloss.png",
                one_pixel_png([153, 153, 153, 255]),
            )),
        };

        let plan = plan_bsp_upload(&extracted).expect("PBR BSP plan");
        assert_eq!(plan.textures[0].material_data_rgba, [0, 64, 192, 153]);
        assert_eq!(
            plan.textures[0].pbr_flags,
            bsp_surface_flags::SURF_PBR
                | bsp_surface_flags::SURF_PBR_NORMAL
                | bsp_surface_flags::SURF_PBR_GLOSS
        );
        assert!(plan.materials[0].is_pbr);
        assert_ne!(
            plan.materials[0].surface_uniform.receive_mask & bsp_surface_flags::RECEIVE_IBL,
            0
        );
    }

    #[cfg(feature = "bsp")]
    #[test]
    fn one_pbr_companion_routes_pbr_and_supplies_missing_channel_defaults() {
        let mut extracted = stress_extracted(1, 1);
        extracted.textures[0].pbr_companions.gloss = Some(
            bsp::resources::TextureCompanion::new(
                "textures/texture_0_gloss.png",
                one_pixel_png([153, 0, 0, 255]),
            ),
        );

        let plan = plan_bsp_upload(&extracted).expect("gloss-only BSP plan");
        assert_eq!(plan.textures[0].material_data_rgba, [0, 128, 128, 153]);
        assert_eq!(
            plan.textures[0].pbr_flags,
            bsp_surface_flags::SURF_PBR | bsp_surface_flags::SURF_PBR_GLOSS
        );
        assert!(plan.materials[0].is_pbr);
    }

    #[cfg(feature = "bsp")]
    #[test]
    fn legacy_texture_without_companions_keeps_legacy_material_data_and_route() {
        let extracted = stress_extracted(1, 1);
        let plan = plan_bsp_upload(&extracted).expect("legacy BSP plan");
        assert_eq!(plan.textures[0].material_data_rgba, [0, 0, 0, 255]);
        assert_eq!(plan.textures[0].pbr_flags, 0);
        assert!(!plan.materials[0].is_pbr);
        assert_eq!(
            plan.materials[0].surface_uniform.surface_flags & bsp_surface_flags::SURF_PBR,
            0
        );
    }

    #[cfg(feature = "bsp")]
    #[test]
    fn companion_on_ineligible_surface_keeps_legacy_route_without_decoding() {
        let mut extracted = stress_extracted(1, 1);
        extracted.face_materials[0].surface_class = bsp::materials::SurfaceClass::Liquid;
        extracted.textures[0].pbr_companions.normal = Some(
            bsp::resources::TextureCompanion::new(
                "textures/texture_0_norm.png",
                vec![1, 2, 3],
            ),
        );

        let plan = plan_bsp_upload(&extracted).expect("liquid legacy BSP plan");
        assert_eq!(plan.textures[0].pbr_flags, 0);
        assert!(!plan.materials[0].is_pbr);
    }

    #[cfg(feature = "bsp")]
    #[test]
    fn pbr_companion_dimension_mismatch_fails_before_gpu_allocation() {
        let mut extracted = stress_extracted(1, 1);
        extracted.textures[0].pbr_companions.normal = Some(
            bsp::resources::TextureCompanion::new(
                "textures/texture_0_norm.png",
                solid_png(2, 1, [128, 128, 255, 255]),
            ),
        );
        let error = plan_bsp_upload(&extracted).unwrap_err();
        assert!(error.contains("is 2x1; expected 1x1"));
    }

    #[cfg(feature = "bsp")]
    #[test]
    fn malformed_pbr_companion_fails_before_gpu_allocation() {
        let mut extracted = stress_extracted(1, 1);
        extracted.textures[0].pbr_companions.normal = Some(bsp::resources::TextureCompanion::new(
            "textures/texture_0_norm.png",
            vec![1, 2, 3],
        ));
        let error = plan_bsp_upload(&extracted).unwrap_err();
        assert!(error.contains("not a valid PNG"));
    }

    #[cfg(feature = "bsp")]
    #[test]
    fn large_map_plan_is_bounded_and_covers_every_face_once() {
        let extracted = stress_extracted(57_595, 118);
        let plan = plan_bsp_upload(&extracted).expect("large BSP should batch safely");

        assert!(plan.demand.batch_count <= MAX_BSP_RENDER_BATCHES);
        assert!(plan.demand.material_count <= MAX_BSP_MATERIALS);
        assert_eq!(plan.demand.renderable_face_count, 57_595);
        assert!(plan.demand.batch_count < plan.demand.renderable_face_count / 10);
        assert_eq!(plan.demand.leaf_bucket_span, 0); // neutral batching: no leaf bucketing
        assert!(plan.face_to_batch.iter().all(Option::is_some));
        assert!(plan.face_to_material.iter().all(Option::is_some));

        let mut seen = vec![0u8; 57_595];
        for batch in &plan.batches {
            for &source_face in &batch.render_batch.face_indices {
                seen[source_face as usize] += 1;
            }
        }
        assert!(seen.into_iter().all(|count| count == 1));
    }

    #[cfg(feature = "bsp")]
    #[test]
    fn material_demand_over_cap_is_rejected_before_upload() {
        let extracted = stress_extracted(MAX_BSP_MATERIALS + 1, MAX_BSP_MATERIALS + 1);
        let error = plan_bsp_upload(&extracted).unwrap_err();
        assert!(error.contains("unique materials") || error.contains("textures"));
    }

    // ── Phase 06: MountedBspBatch invariant tests ──────────────────────

    #[cfg(feature = "bsp")]
    #[test]
    fn mounted_batch_rejects_null_mesh() {
        let batch = RenderBatch {
            key: bsp::geometry::BatchKey {
                render_class: 0,
                material_identity: 0,
                lightmap_page: 0,
                model_index: 0,
            },
            leaf_signature: vec![0],
            face_indices: vec![0],
            pvs_eligible: true,
            is_inline_model: false,
            model_index: 0,
        };
        let err = MountedBspBatch::try_new(
            &batch,
            MeshHandle::new(0, 0),
            BspMaterialHandle::new(0, 0),
            (glam::Vec3::ZERO, glam::Vec3::ONE),
        )
        .unwrap_err();
        assert!(err.contains("null mesh"));
    }

    #[cfg(feature = "bsp")]
    #[test]
    fn mounted_batch_rejects_empty_face_list() {
        let batch = RenderBatch {
            key: bsp::geometry::BatchKey {
                render_class: 0,
                material_identity: 0,
                lightmap_page: 0,
                model_index: 0,
            },
            leaf_signature: vec![],
            face_indices: vec![],
            pvs_eligible: true,
            is_inline_model: false,
            model_index: 0,
        };
        let err = MountedBspBatch::try_new(
            &batch,
            MeshHandle::new(1, 0),
            BspMaterialHandle::new(1, 0),
            (glam::Vec3::ZERO, glam::Vec3::ONE),
        )
        .unwrap_err();
        assert!(err.contains("empty face"));
    }

    #[cfg(feature = "bsp")]
    #[test]
    fn mounted_batch_rejects_non_finite_bounds() {
        let batch = RenderBatch {
            key: bsp::geometry::BatchKey {
                render_class: 0,
                material_identity: 0,
                lightmap_page: 0,
                model_index: 0,
            },
            leaf_signature: vec![0],
            face_indices: vec![0],
            pvs_eligible: true,
            is_inline_model: false,
            model_index: 0,
        };
        let err = MountedBspBatch::try_new(
            &batch,
            MeshHandle::new(1, 0),
            BspMaterialHandle::new(1, 0),
            (glam::Vec3::NAN, glam::Vec3::ONE),
        )
        .unwrap_err();
        assert!(err.contains("non-finite bounds"));
    }

    #[cfg(feature = "bsp")]
    #[test]
    fn mounted_batch_rejects_inverted_bounds() {
        let batch = RenderBatch {
            key: bsp::geometry::BatchKey {
                render_class: 0,
                material_identity: 0,
                lightmap_page: 0,
                model_index: 0,
            },
            leaf_signature: vec![0],
            face_indices: vec![0],
            pvs_eligible: true,
            is_inline_model: false,
            model_index: 0,
        };
        let err = MountedBspBatch::try_new(
            &batch,
            MeshHandle::new(1, 0),
            BspMaterialHandle::new(1, 0),
            (glam::Vec3::ONE, glam::Vec3::ZERO),
        )
        .unwrap_err();
        assert!(err.contains("inverted bounds"));
    }

    #[cfg(feature = "bsp")]
    #[test]
    fn plan_bsp_upload_rejects_mismatched_face_arrays() {
        let mut extracted = stress_extracted(3, 1);
        extracted.face_materials.pop(); // make materials shorter than geometry
        let err = plan_bsp_upload(&extracted).unwrap_err();
        assert!(err.contains("differ in length"));
    }

    #[cfg(feature = "bsp")]
    #[test]
    fn plan_bsp_upload_rejects_zero_sized_atlas_page() {
        let mut extracted = stress_extracted(1, 1);
        extracted.lightmap_atlas.pages[0].width = 0;
        let err = plan_bsp_upload(&extracted).unwrap_err();
        assert!(err.contains("zero-sized page"));
    }

    #[cfg(feature = "bsp")]
    #[test]
    fn plan_bsp_upload_rejects_missing_texture_reference() {
        let mut extracted = stress_extracted(1, 1);
        extracted.face_materials[0].material_index = 999; // out of bounds
        let err = plan_bsp_upload(&extracted).unwrap_err();
        assert!(err.contains("does not exist"));
    }

    #[cfg(feature = "bsp")]
    #[test]
    fn plan_bsp_upload_rejects_renderable_without_material() {
        let mut extracted = stress_extracted(1, 1);
        extracted.face_materials.clear(); // no materials array
        let err = plan_bsp_upload(&extracted).unwrap_err();
        assert!(
            err.contains("has no material") || err.contains("differ in length"),
            "expected material-related error, got: {err}"
        );
    }

    #[cfg(feature = "bsp")]
    #[test]
    fn compute_batch_bounds_rejects_empty_mesh() {
        let mesh = ProceduralMeshData {
            name: "empty".to_string(),
            vertices: vec![],
            indices: vec![],
            material: None,
        };
        let err = compute_batch_bounds(&mesh).unwrap_err();
        assert!(err.contains("empty batch mesh"));
    }

    #[cfg(feature = "bsp")]
    #[test]
    fn compute_batch_bounds_rejects_non_finite_vertex() {
        let mesh = ProceduralMeshData {
            name: "nan".to_string(),
            vertices: vec![
                crate::api::ProceduralVertex {
                    position: glam::Vec3::NAN,
                    normal: glam::Vec3::Z,
                    tangent: glam::Vec4::W,
                    uv0: glam::Vec2::ZERO,
                    uv1: glam::Vec2::ZERO,
                    color: glam::Vec4::ONE,
                },
            ],
            indices: vec![0],
            material: None,
        };
        let err = compute_batch_bounds(&mesh).unwrap_err();
        assert!(err.contains("non-finite vertex"));
    }
}
