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
use crate::data::handles::{BspTextureHandle, MaterialHandle, MeshHandle};
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
}

#[cfg(feature = "bsp")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct PlannedBatchKey {
    material_plan_index: usize,
    model_index: u32,
    leaf_bucket: u32,
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
            surface_flags: surface_flags_for(Some(surface_class)),
            receive_mask: receive_mask_for(Some(surface_class)),
            lightmap_layer_base,
            liquid_warp_scale: 0.02,
            liquid_flow_speed: 1.0,
            _pad1: [0, 0],
        },
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
fn batch_key(face: PlannedFace, leaf_bucket_span: u32) -> PlannedBatchKey {
    let leaf_bucket = if face.model_index != 0 {
        u32::MAX
    } else if leaf_bucket_span == u32::MAX {
        0
    } else {
        face.primary_leaf
            .map(|leaf| leaf / leaf_bucket_span.max(1))
            .unwrap_or(u32::MAX - 1)
    };
    PlannedBatchKey {
        material_plan_index: face.material_plan_index,
        model_index: face.model_index,
        leaf_bucket,
    }
}

#[cfg(feature = "bsp")]
fn grouped_faces(
    faces: &[PlannedFace],
    leaf_bucket_span: u32,
) -> std::collections::BTreeMap<PlannedBatchKey, Vec<PlannedFace>> {
    let mut groups = std::collections::BTreeMap::new();
    for &face in faces {
        groups
            .entry(batch_key(face, leaf_bucket_span))
            .or_insert_with(Vec::new)
            .push(face);
    }
    groups
}

#[cfg(feature = "bsp")]
fn choose_leaf_bucket_span(faces: &[PlannedFace]) -> Result<u32, String> {
    let mut span = 1u32;
    loop {
        let count = grouped_faces(faces, span).len();
        if count <= MAX_BSP_RENDER_BATCHES {
            return Ok(span);
        }
        if span == u32::MAX {
            return Err(format!(
                "BSP requires {count} render batches even with global spatial grouping; maximum is {MAX_BSP_RENDER_BATCHES}"
            ));
        }
        span = span.checked_mul(2).unwrap_or(u32::MAX);
    }
}

#[cfg(feature = "bsp")]
fn bake_lightmap_uv(
    extracted: &ExtractedBsp,
    face_index: usize,
    uv: Vec2,
) -> Result<Vec2, String> {
    let layout = &extracted.face_lightmap_layouts[face_index];
    if !layout.has_data {
        let (width, height) = extracted
            .lightmap_atlas
            .pages
            .first()
            .map(|page| (page.width.max(1), page.height.max(1)))
            .unwrap_or((1, 1));
        return Ok(Vec2::new(0.5 / width as f32, 0.5 / height as f32));
    }
    let page = extracted
        .lightmap_atlas
        .pages
        .get(layout.page_index as usize)
        .ok_or_else(|| {
            format!(
                "BSP face {face_index} references missing lightmap page {}",
                layout.page_index
            )
        })?;
    if page.width == 0 || page.height == 0 {
        return Err(format!(
            "BSP face {face_index} references zero-sized lightmap page {}",
            layout.page_index
        ));
    }
    let texel = Vec2::new(
        layout.atlas_offset.0 as f32 + uv.x * layout.luxel_extents.0.max(1) as f32,
        layout.atlas_offset.1 as f32 + uv.y * layout.luxel_extents.1.max(1) as f32,
    );
    let atlas_uv = texel / Vec2::new(page.width as f32, page.height as f32);
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
        let mask = checked_mul(
            u64::try_from(texture.fullbright_mask.len())
                .map_err(|_| "BSP fullbright byte count exceeds u64".to_string())?,
            4,
            "fullbright expansion",
        )?;
        checked_add(total, checked_add(albedo, mask, "texture")?, "texture total")
    })?;

    let (atlas_width, atlas_height, atlas_pages) = if let Some(first) = extracted.lightmap_atlas.pages.first() {
        if first.width == 0 || first.height == 0 {
            return Err("BSP lightmap atlas has a zero-sized page".to_string());
        }
        for page in &extracted.lightmap_atlas.pages {
            if page.width != first.width || page.height != first.height {
                return Err("BSP lightmap atlas pages must have identical dimensions".to_string());
            }
        }
        (first.width as u64, first.height as u64, extracted.lightmap_atlas.pages.len() as u64)
    } else {
        (1, 1, 1)
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

/// Build bounded renderer batches and merged meshes before allocating GPU resources.
#[cfg(feature = "bsp")]
pub(crate) fn plan_bsp_upload(extracted: &ExtractedBsp) -> Result<BspUploadPlan, String> {
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
    let leaf_bucket_span = choose_leaf_bucket_span(&faces)?;
    let groups = grouped_faces(&faces, leaf_bucket_span);
    let mut face_to_batch = vec![None; extracted.face_geometries.len()];
    let mut face_to_material = vec![None; extracted.face_geometries.len()];
    let mut batches = Vec::with_capacity(groups.len());

    for (batch_index, (key, mut group_faces)) in groups.into_iter().enumerate() {
        group_faces.sort_by_key(|face| face.source_face_index);
        let mut leaf_signature = group_faces
            .iter()
            .flat_map(|face| extracted.leaf_membership[face.face_index].iter().copied())
            .collect::<Vec<_>>();
        leaf_signature.sort_unstable();
        leaf_signature.dedup();
        let source_faces = group_faces
            .iter()
            .map(|face| face.source_face_index)
            .collect::<Vec<_>>();
        for face in &group_faces {
            if face_to_batch[face.face_index].replace(batch_index).is_some() {
                return Err(format!("BSP face {} was assigned to multiple batches", face.face_index));
            }
            face_to_material[face.face_index] = Some(key.material_plan_index);
        }
        let material = &materials[key.material_plan_index];
        let lightmap_page = material.surface_uniform.lightmap_layer_base / 4;
        let render_batch = RenderBatch {
            key: bsp::geometry::BatchKey {
                leaf_signature,
                render_class: render_class_index(material.surface_class),
                material_identity: key.material_plan_index as u64,
                lightmap_page,
            },
            face_indices: source_faces,
            pvs_eligible: key.model_index == 0,
            is_inline_model: key.model_index != 0,
            model_index: key.model_index,
        };
        batches.push(BspPlannedBatch {
            mesh: merge_batch_mesh(extracted, batch_index, &group_faces)?,
            material_plan_index: key.material_plan_index,
            render_batch,
        });
    }

    let demand = compute_upload_demand(
        extracted,
        &batches,
        materials.len(),
        faces.len(),
        leaf_bucket_span,
    )?;
    Ok(BspUploadPlan {
        materials,
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
                let layer_count = atlas.styles.len().max(1) as u32;
                let mut rgba = vec![0u8; pixel_count * 4 * layer_count as usize];
                for layer in 0..layer_count as usize {
                    for y in 0..page.height as usize {
                        for x in 0..page.width as usize {
                            let src_idx = (y * page.width as usize + x) * 3;
                            let dst_idx =
                                layer * pixel_count * 4 + (y * page.width as usize + x) * 4;
                            if src_idx + 2 < page.data.len() {
                                rgba[dst_idx] = page.data[src_idx];
                                rgba[dst_idx + 1] = page.data[src_idx + 1];
                                rgba[dst_idx + 2] = page.data[src_idx + 2];
                                rgba[dst_idx + 3] = 255;
                            }
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
    for (slot, style_layout) in layout.style_layers.iter().take(4).enumerate() {
        if style_layout.has_data && style_layout.style_id <= 63 {
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
            style_ids: style_ids_for_layout(layout),
            fullbright_base: 224,
            fullbright_count: 32,
            alpha_threshold: 0.5,
            animation_frame: 0,
            animation_time: 0.0,
            surface_flags: surface_flags_for(sc),
            receive_mask: receive_mask_for(sc),
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
                    layer_index: 0,
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
    #[test]
    fn large_map_plan_is_bounded_and_covers_every_face_once() {
        let extracted = stress_extracted(57_595, 118);
        let plan = plan_bsp_upload(&extracted).expect("large BSP should batch safely");

        assert!(plan.demand.batch_count <= MAX_BSP_RENDER_BATCHES);
        assert!(plan.demand.material_count <= MAX_BSP_MATERIALS);
        assert_eq!(plan.demand.renderable_face_count, 57_595);
        assert!(plan.demand.batch_count < plan.demand.renderable_face_count / 10);
        assert!(plan.demand.leaf_bucket_span > 1);
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
}
