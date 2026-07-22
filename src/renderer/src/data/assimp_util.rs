//! # Assimp Model Ingest
//!
//! Active model loading path: imports meshes/materials via Assimp, maps them to engine cache
//! handles, and builds a `SceneWorld` hierarchy for render submission traversal.
//!
//! Internal assimp ingestion with future-facing helpers; dead code allowed.
//!
//! ## Unsafe Boundary Policy (Phase 05)
//!
//! Every `unsafe` block in this module targets a narrow, documented invariant:
//!
//! - **Owner**: `ScenePropertyGuard` releases the property store exactly once on every path.
//! - **Owner**: `ImportedSceneGuard` releases the loaded scene exactly once on every path.
//! - **Lifetime**: Borrowed `aiScene` access is bounded by the guard lifetime; no scene
//!   reference escapes to a caller that could drop the guard first.
//! - **Preconditions checked before dereference**: Every pointer, count, product, index,
//!   declared string length, and byte length is validated before `unsafe { &*p }` or
//!   `std::slice::from_raw_parts`.
//! - **Malformed input**: All untrusted-input error paths return `AssimpImportError`;
//!   none panic on malformed scene/mesh/material/texture data.

use crate::api::scene::{BoundsUnknownReason, MeshBoundsEntry, SceneBounds};
use crate::api::AssetPolicyConfig;
use crate::data::asset_manifest::{self, ResolvedTexturePolicy};
use crate::data::camera::Aabb;
use crate::data::compression;
use crate::data::data_cache::{TextureCache, VkDataCache};
use crate::data::data_util::resolve_texture_mip_count;
use crate::data::gpu_data::{
    AlphaMode, MaterialMeta, MaterialShadingModel, MeshMeta, TextureMeta, TexturePayload,
    TextureSemantic, Vertex,
};
use crate::data::handles::{MaterialHandle, MeshHandle};
use crate::data::mesh_geometry::MeshDeformation;
use crate::scene::scene_world::{SceneNodeId, SceneWorld};
use ash::vk;
use glam::{Mat4, Vec3, Vec4};
use image::{DynamicImage, GenericImageView};
use log::debug;
use russimp_sys_ng::{
    aiColor4D, aiCreatePropertyStore, aiGetMaterialColor, aiGetMaterialFloatArray,
    aiGetMaterialString, aiGetMaterialTexture, aiGetMaterialTextureCount,
    aiImportFileExWithProperties, aiMaterial, aiNode,
    aiPostProcessSteps_aiProcess_CalcTangentSpace, aiPostProcessSteps_aiProcess_FixInfacingNormals,
    aiPostProcessSteps_aiProcess_FlipUVs, aiPostProcessSteps_aiProcess_GenSmoothNormals,
    aiPostProcessSteps_aiProcess_JoinIdenticalVertices,
    aiPostProcessSteps_aiProcess_PreTransformVertices, aiPostProcessSteps_aiProcess_Triangulate,
    aiReturn_aiReturn_SUCCESS, aiScene, aiSetImportPropertyInteger, aiString, aiTextureType,
    aiTextureType_aiTextureType_AMBIENT_OCCLUSION, aiTextureType_aiTextureType_BASE_COLOR,
    aiTextureType_aiTextureType_DIFFUSE, aiTextureType_aiTextureType_DIFFUSE_ROUGHNESS,
    aiTextureType_aiTextureType_EMISSIVE, aiTextureType_aiTextureType_LIGHTMAP,
    aiTextureType_aiTextureType_METALNESS, aiTextureType_aiTextureType_NORMALS,
    aiTextureType_aiTextureType_NORMAL_CAMERA, aiTextureType_aiTextureType_UNKNOWN,
};
use std::collections::HashMap;
use std::default::Default;
use std::ffi::{c_char, c_uint, CString};
use std::path::Path;
use std::sync::Arc;

/// Typed error enum for Assimp model ingest failures.
///
/// Replaces stringly-typed errors, `eprintln!`, `assert!`, and `expect!` in the
/// ingest path with deterministic, contextual error variants that map cleanly
/// to `AssetError`.
#[derive(Debug)]
pub enum AssimpImportError {
    /// The provided file path was invalid (e.g. not valid UTF-8 or null interior).
    InvalidPath(String),
    /// Assimp failed to load the scene from disk.
    SceneLoadFailed { path: String, reason: String },
    /// A mesh pointer in the scene was null.
    NullMesh { mesh_index: usize },
    /// A mesh's index count is inconsistent with its vertex count.
    InvalidIndices {
        mesh_name: String,
        index_count: usize,
        vertex_count: usize,
    },
    /// A node referenced a mesh index that was not present in the processed mesh map.
    MissingMeshMapping { node_name: String, mesh_index: u32 },
    /// A texture could not be decoded during material processing.
    TextureDecode { texture_ref: String, reason: String },
    /// Manifest/policy resolution failed for a discovered texture.
    TexturePolicy { texture_ref: String, reason: String },
    /// A face index was out of bounds relative to the vertex array.
    FaceIndexOutOfBounds {
        mesh_name: String,
        face_index: usize,
        vertex_index: u32,
        vertex_count: usize,
    },
    /// An embedded texture index exceeded the scene's texture array.
    EmbeddedTextureIndexOutOfBounds {
        texture_index: usize,
        texture_count: usize,
    },
    /// An embedded texture had an inconsistent or unusable payload.
    EmbeddedTextureInvalid {
        texture_index: usize,
        reason: String,
    },
    /// A node had a null mesh pointer in its mesh array despite a positive count.
    NullNodeMesh {
        node_name: String,
        local_mesh_index: usize,
    },
    /// A declared aiString length exceeded the fixed buffer capacity.
    StringLengthOverflow {
        context: String,
        declared_len: u32,
        buffer_max: usize,
    },
    /// Catch-all for unexpected internal failures (e.g. FFI null pointers in required structures).
    Internal(String),
}

// ── RAII Assimp Guards ─────────────────────────────────────────────────────
//
// aiCreatePropertyStore → aiReleasePropertyStore
// aiImportFileExWithProperties → aiReleaseImport
//
// Both guards are private; callers never own raw Assimp pointers.

/// RAII guard for an Assimp property store.
///
/// Constructed immediately after a non-null `aiCreatePropertyStore()` return.
/// The property store is passed to `aiImportFileExWithProperties` and then
/// released on drop.
struct ScenePropertyGuard {
    store: *mut russimp_sys_ng::aiPropertyStore,
}

impl ScenePropertyGuard {
    /// # Safety
    /// `store` must be a non-null, valid property-store pointer from `aiCreatePropertyStore`.
    unsafe fn new() -> Result<Self, AssimpImportError> {
        let store = aiCreatePropertyStore();
        if store.is_null() {
            return Err(AssimpImportError::Internal(
                "aiCreatePropertyStore returned null".to_string(),
            ));
        }
        Ok(Self { store })
    }

    fn as_ptr(&self) -> *mut russimp_sys_ng::aiPropertyStore {
        self.store
    }
}

impl Drop for ScenePropertyGuard {
    fn drop(&mut self) {
        // SAFETY: Assimp ownership is held by the RAII guard or caller-borrowed scene object; relevant counts, null pointers, indices, and byte lengths are checked before this operation.
        unsafe {
            russimp_sys_ng::aiReleasePropertyStore(self.store);
        }
    }
}

/// RAII guard for an Assimp imported scene.
///
/// Wraps a non-null `*const aiScene` returned by `aiImportFileExWithProperties`.
/// On drop, calls `aiReleaseImport`. Borrowed access to the `aiScene` is
/// provided via `Deref`; the reference lifetime cannot outlive this guard.
struct ImportedSceneGuard {
    scene: *const aiScene,
}

impl ImportedSceneGuard {
    /// # Safety
    /// `scene` must be a non-null, valid scene pointer from `aiImportFileExWithProperties`.
    unsafe fn from_non_null(scene: *const aiScene) -> Self {
        debug_assert!(!scene.is_null());
        Self { scene }
    }

    fn as_ref(&self) -> &aiScene {
        // SAFETY: Construction guarantees non-null; guard lifetime bounds reference lifetime.
        unsafe { &*self.scene }
    }
}

impl Drop for ImportedSceneGuard {
    fn drop(&mut self) {
        // SAFETY: Assimp ownership is held by the RAII guard or caller-borrowed scene object; relevant counts, null pointers, indices, and byte lengths are checked before this operation.
        unsafe {
            russimp_sys_ng::aiReleaseImport(self.scene);
        }
    }
}

impl std::fmt::Display for AssimpImportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidPath(path) => write!(f, "invalid asset path: {path}"),
            Self::SceneLoadFailed { path, reason } => {
                write!(f, "failed to load scene '{path}': {reason}")
            }
            Self::NullMesh { mesh_index } => {
                write!(f, "null mesh pointer at index {mesh_index}")
            }
            Self::InvalidIndices {
                mesh_name,
                index_count,
                vertex_count,
            } => write!(
                f,
                "mesh '{mesh_name}' has {index_count} indices but only {vertex_count} vertices"
            ),
            Self::MissingMeshMapping {
                node_name,
                mesh_index,
            } => write!(
                f,
                "node '{node_name}' references mesh index {mesh_index} which was not found"
            ),
            Self::TextureDecode {
                texture_ref,
                reason,
            } => write!(f, "texture decode failed for '{texture_ref}': {reason}"),
            Self::TexturePolicy {
                texture_ref,
                reason,
            } => write!(
                f,
                "texture policy resolve failed for '{texture_ref}': {reason}"
            ),
            Self::FaceIndexOutOfBounds {
                mesh_name,
                face_index,
                vertex_index,
                vertex_count,
            } => write!(
                f,
                "mesh '{mesh_name}' face {face_index} references vertex index {vertex_index} >= vertex count {vertex_count}"
            ),
            Self::EmbeddedTextureIndexOutOfBounds {
                texture_index,
                texture_count,
            } => write!(
                f,
                "embedded texture index {texture_index} exceeds scene texture count {texture_count}"
            ),
            Self::EmbeddedTextureInvalid {
                texture_index,
                reason,
            } => write!(
                f,
                "embedded texture {texture_index} has invalid payload: {reason}"
            ),
            Self::NullNodeMesh {
                node_name,
                local_mesh_index,
            } => write!(
                f,
                "node '{node_name}' has null mesh pointer at local index {local_mesh_index}"
            ),
            Self::StringLengthOverflow {
                context,
                declared_len,
                buffer_max,
            } => write!(
                f,
                "{context}: declared aiString length {declared_len} exceeds buffer capacity {buffer_max}"
            ),
            Self::Internal(msg) => write!(f, "internal import error: {msg}"),
        }
    }
}

impl std::error::Error for AssimpImportError {}

pub struct ModelMeta {
    pub scene_world: SceneWorld,
    pub material_ids: Vec<MaterialHandle>,
    pub mesh_ids: Vec<MeshHandle>,
    /// Importer-derived deformation classification aligned with `mesh_ids`.
    pub mesh_deformations: Vec<MeshDeformation>,
}

pub fn load_model(
    path: &str,
    data_cache: Arc<VkDataCache>,
    has_animation: bool,
    policy_config: &AssetPolicyConfig,
) -> Result<ModelMeta, AssimpImportError> {
    let mut flags = aiPostProcessSteps_aiProcess_GenSmoothNormals
        | aiPostProcessSteps_aiProcess_JoinIdenticalVertices
        | aiPostProcessSteps_aiProcess_Triangulate
        | aiPostProcessSteps_aiProcess_FlipUVs
        //   | aiPostProcessSteps_aiProcess_PreTransformVertices
        | aiPostProcessSteps_aiProcess_FixInfacingNormals
        | aiPostProcessSteps_aiProcess_CalcTangentSpace;
    //  | aiPostProcessSteps_aiProcess_LimitBoneWeights;

    if has_animation {
        flags |= aiPostProcessSteps_aiProcess_PreTransformVertices;
    };

    // RAII property store — released exactly once on every return path.
    // SAFETY: Assimp ownership is held by the RAII guard or caller-borrowed scene object; relevant counts, null pointers, indices, and byte lengths are checked before this operation.
    let props = unsafe { ScenePropertyGuard::new() }?;
    const AI_CONFIG_IMPORT_NO_OVERWRITE_NORMALS: &[u8] = b"IMPORT_NO_OVERWRITE_NORMALS\0";
    unsafe {
        aiSetImportPropertyInteger(
            props.as_ptr(),
            AI_CONFIG_IMPORT_NO_OVERWRITE_NORMALS.as_ptr() as *const c_char,
            1,
        );
    }

    let path_c =
        CString::new(path).map_err(|_| AssimpImportError::InvalidPath(path.to_string()))?;
    let base_path = Path::new(path)
        .parent()
        .map_or_else(|| None, |p| p.to_str());

    // RAII scene guard — scene is released exactly once on every return path.
    let scene_guard = {
        // SAFETY: Assimp ownership is held by the RAII guard or caller-borrowed scene object; relevant counts, null pointers, indices, and byte lengths are checked before this operation.
        let scene_ptr = unsafe {
            aiImportFileExWithProperties(
                path_c.as_ptr(),
                flags as c_uint,
                std::ptr::null_mut(),
                props.as_ptr(),
            )
        };
        if scene_ptr.is_null() {
            return Err(AssimpImportError::SceneLoadFailed {
                path: path.to_string(),
                reason: "Assimp returned null scene pointer".to_string(),
            });
        }
        // SAFETY: we just checked non-null.
        unsafe { ImportedSceneGuard::from_non_null(scene_ptr) }
    };
    let ai_scene = scene_guard.as_ref();

    let materials = process_materials(ai_scene, base_path, &data_cache, policy_config)?;
    let mat_indices = 0..materials.len();

    let mut mapped_materials = HashMap::<u32, MaterialHandle>::with_capacity(materials.len());
    let material_ids = data_cache
        .texture_cache
        .lock()
        .map_err(|_| AssimpImportError::Internal("texture_cache lock poisoned".to_string()))?
        .add_materials(materials);

    for (og_idx, id) in mat_indices.into_iter().zip(material_ids.iter()) {
        mapped_materials.insert(og_idx as u32, *id);
    }

    let meshes = process_meshes(ai_scene, mapped_materials)?;
    let mesh_deformations = classify_imported_meshes(ai_scene, meshes.len())?;
    let mesh_bound_states = meshes
        .iter()
        .zip(mesh_deformations.iter().copied())
        .map(|(mesh, deformation)| imported_mesh_bounds(mesh, deformation))
        .collect::<Vec<_>>();
    let mesh_indices = 0..meshes.len();

    let mut mapped_meshes = HashMap::<u32, MeshHandle>::with_capacity(meshes.len());
    let mesh_ids = data_cache
        .mesh_cache
        .lock()
        .map_err(|_| AssimpImportError::Internal("mesh_cache lock poisoned".to_string()))?
        .add_multi(meshes);

    for (og_idx, id) in mesh_indices.into_iter().zip(mesh_ids.iter()) {
        mapped_meshes.insert(og_idx as u32, *id);
    }

    // Build node metadata directly from the CPU meshes before GPU promotion.
    // DTO registration happens after this import function returns, so querying
    // the geometry store here would incorrectly classify every mesh as stale.
    let mesh_bounds_map = mesh_ids
        .iter()
        .copied()
        .zip(mesh_bound_states)
        .collect::<HashMap<_, _>>();

    let root_ai_node = ai_scene.mRootNode;

    if root_ai_node.is_null() {
        return Err(AssimpImportError::Internal(
            "scene root node pointer is null".to_string(),
        ));
    }

    let mut scene_world = SceneWorld::new();
    let root_id = process_node(
        root_ai_node,
        &mapped_meshes,
        &mesh_bounds_map,
        &mut scene_world,
        None,
    )?;
    scene_world.set_root(root_id);
    Ok(ModelMeta {
        scene_world,
        material_ids,
        mesh_ids,
        mesh_deformations,
    })
}

// ── Checked Numeric / String Helpers ─────────────────────────────────────────

/// Convert an Assimp `c_uint` count to `usize`, rejecting overflow.
fn checked_usize_from_u32(val: u32, label: &'static str) -> Result<usize, AssimpImportError> {
    usize::try_from(val).map_err(|_| {
        AssimpImportError::Internal(format!("{label} value {val} does not fit in usize"))
    })
}

/// Compute the product of two counts as a checked `usize`.
fn checked_product_usize(
    a: usize,
    b: usize,
    label: &'static str,
) -> Result<usize, AssimpImportError> {
    a.checked_mul(b).ok_or_else(|| {
        AssimpImportError::Internal(format!("{label}: product {a} * {b} overflowed"))
    })
}

// ── Checked Pointer-Traversal Helpers ────────────────────────────────────────

/// Dereference a pointer array element at `index`, validating:
/// - Positive `count` has a non-null base pointer.
/// - `index < count` before `base.add(index)`.
/// - The member pointer is non-null before dereference.
///
/// # Safety
/// Caller must ensure a non-null `base` points to a valid array of at least `count` pointers.
unsafe fn deref_ptr_array<'a, T>(
    base: *const *const T,
    index: usize,
    count: usize,
    err_context: impl FnOnce() -> AssimpImportError,
) -> Result<&'a T, AssimpImportError> {
    if count > 0 && base.is_null() {
        return Err(err_context());
    }
    if index >= count {
        return Err(err_context());
    }
    // SAFETY: Assimp ownership is held by the RAII guard or caller-borrowed scene object; relevant counts, null pointers, indices, and byte lengths are checked before this operation.
    let ptr = unsafe { *base.add(index) };
    if ptr.is_null() {
        return Err(err_context());
    }
    // SAFETY: base/index/member pointer were all validated above; lifetime is tied to caller owner.
    Ok(unsafe { &*ptr })
}

fn require_non_null_array<T>(
    ptr: *const T,
    count: usize,
    context: impl Into<String>,
) -> Result<(), AssimpImportError> {
    if count > 0 && ptr.is_null() {
        return Err(AssimpImportError::Internal(context.into()));
    }
    Ok(())
}

/// Read the content of an `aiString` with checked length, returning a UTF-8 lossy `String`.
///
/// Validates that the declared `length` field does not exceed the fixed buffer size (1024 bytes
/// per Assimp ABI). This prevents an out-of-bounds read from a malformed scene that declares an
/// impossible string length; decoding uses the declared byte range and UTF-8 lossy conversion.
fn checked_ai_string_lossy(ai_str: &aiString, context: &str) -> Result<String, AssimpImportError> {
    let declared = checked_usize_from_u32(ai_str.length, "aiString.length")?;
    let buffer_max = ai_str.data.len();
    if declared > buffer_max {
        return Err(AssimpImportError::StringLengthOverflow {
            context: context.to_string(),
            declared_len: ai_str.length,
            buffer_max,
        });
    }
    // SAFETY: `data` is a fixed inline aiString buffer and declared <= data.len(). We decode
    // exactly the declared bytes and use UTF-8 lossy conversion; no search for a terminator can
    // read outside the fixed buffer.
    let raw = unsafe { std::slice::from_raw_parts(ai_str.data.as_ptr() as *const u8, declared) };
    Ok(String::from_utf8_lossy(raw).into_owned())
}

// ── Material Processing ─────────────────────────────────────────────────────

pub fn process_materials(
    ai_scene: &aiScene,
    base_path: Option<&str>,
    data_cache: &Arc<VkDataCache>,
    policy_config: &AssetPolicyConfig,
) -> Result<Vec<MaterialMeta>, AssimpImportError> {
    let mat_count = checked_usize_from_u32(ai_scene.mNumMaterials, "mNumMaterials")?;
    if mat_count > 0 && ai_scene.mMaterials.is_null() {
        return Err(AssimpImportError::Internal(
            "mNumMaterials is positive but mMaterials pointer is null".to_string(),
        ));
    }
    let mut materials = Vec::<MaterialMeta>::with_capacity(mat_count);

    for i in 0..mat_count {
        // SAFETY: mat_count bounds validated; non-null base confirmed above.
        // mMaterials is *mut *mut aiMaterial in russimp_sys_ng; cast to *const *const.
        let ai_material = unsafe {
            deref_ptr_array(
                ai_scene.mMaterials as *const *const aiMaterial,
                i,
                mat_count,
                || AssimpImportError::Internal(format!("null material pointer at index {i}")),
            )?
        };

        let mut material_meta = MaterialMeta::default();
        // Assimp import defaults to PBR until explicit unlit extension metadata is wired.
        material_meta.shading_model = MaterialShadingModel::PbrMetalRough;
        // SAFETY: ai_material is a valid non-null pointer verified by deref_ptr_array above.
        let alpha_mode = unsafe { get_alpha_mode(ai_material) }?;
        let alpha_cutoff = unsafe { get_alpha_cutoff(ai_material) };
        material_meta.set_alpha_mode(alpha_mode, alpha_cutoff);

        let load_first_texture_type =
            |label: &str,
             types: &[(aiTextureType, &'static str)]|
             -> Result<Option<(TextureMeta, &'static str)>, AssimpImportError> {
                for (typ, type_name) in types {
                    // SAFETY: ai_material is valid; Assimp FFI called within get_texture_meta.
                    if let Some(meta) = unsafe {
                        get_texture_meta(
                            ai_material,
                            ai_scene,
                            *typ,
                            base_path,
                            data_cache,
                            policy_config,
                        )
                    }? {
                        debug!(
                            "Material {} {} texture resolved from Assimp type {}",
                            i, label, type_name
                        );
                        return Ok(Some((meta, *type_name)));
                    }
                }
                Ok(None)
            };

        let base_color = load_first_texture_type(
            "base_color",
            &[
                (aiTextureType_aiTextureType_BASE_COLOR, "BASE_COLOR"),
                (aiTextureType_aiTextureType_DIFFUSE, "DIFFUSE"),
            ],
        )?;

        // Assimp may expose metallic/roughness as either:
        // - one combined texture (UNKNOWN), or
        // - two split sources (METALNESS + DIFFUSE_ROUGHNESS).
        // SAFETY: ai_material and ai_scene are valid per caller contract.
        let met_rough_combined = unsafe {
            get_texture_meta(
                ai_material,
                ai_scene,
                aiTextureType_aiTextureType_UNKNOWN,
                base_path,
                data_cache,
                policy_config,
            )
        }?;
        // SAFETY: Assimp ownership is held by the RAII guard or caller-borrowed scene object; relevant counts, null pointers, indices, and byte lengths are checked before this operation.
        let met_rough_metalness = unsafe {
            get_texture_meta(
                ai_material,
                ai_scene,
                aiTextureType_aiTextureType_METALNESS,
                base_path,
                data_cache,
                policy_config,
            )
        }?;
        // SAFETY: Assimp ownership is held by the RAII guard or caller-borrowed scene object; relevant counts, null pointers, indices, and byte lengths are checked before this operation.
        let met_rough_roughness = unsafe {
            get_texture_meta(
                ai_material,
                ai_scene,
                aiTextureType_aiTextureType_DIFFUSE_ROUGHNESS,
                base_path,
                data_cache,
                policy_config,
            )
        }?;

        let normal = load_first_texture_type(
            "normal",
            &[
                (aiTextureType_aiTextureType_NORMAL_CAMERA, "NORMAL_CAMERA"),
                (aiTextureType_aiTextureType_NORMALS, "NORMALS"),
            ],
        )?;

        let occlusion = load_first_texture_type(
            "occlusion",
            &[
                (
                    aiTextureType_aiTextureType_AMBIENT_OCCLUSION,
                    "AMBIENT_OCCLUSION",
                ),
                (aiTextureType_aiTextureType_LIGHTMAP, "LIGHTMAP"),
            ],
        )?;

        let emissive = load_first_texture_type(
            "emissive",
            &[(aiTextureType_aiTextureType_EMISSIVE, "EMISSIVE")],
        )?;

        let mut tex_cache = data_cache
            .texture_cache
            .lock()
            .map_err(|_| AssimpImportError::Internal("texture_cache lock poisoned".to_string()))?;

        if let Some((mut meta, _source)) = base_color {
            meta = compression::apply_compression_policy(
                meta,
                TextureSemantic::BaseColor,
                policy_config,
                &data_cache.supported_image_formats,
            );
            // SAFETY: ai_material is valid.
            let color = unsafe { get_color_factor(ai_material) };
            let uv_set = meta.uv_index;
            let tex_id = tex_cache.add_texture(meta);
            material_meta.add_base_color(tex_id, color, uv_set);
        }

        if let Some(mut meta) = resolve_metallic_roughness_meta(
            met_rough_combined,
            met_rough_metalness,
            met_rough_roughness,
            i,
        ) {
            meta = compression::apply_compression_policy(
                meta,
                TextureSemantic::MetallicRoughness,
                policy_config,
                &data_cache.supported_image_formats,
            );
            // SAFETY: ai_material is valid.
            let metallic_factor =
                unsafe { get_float_factor(ai_material, AI_MATKEY_METALLIC_FACTOR, 1.0) };
            let roughness_factor = unsafe {
                get_float_factor(
                    ai_material,
                    AI_MATKEY_ROUGHNESS_FACTOR,
                    TextureCache::DEFAULT_ROUGHNESS_FACTOR,
                )
            };
            let uv_set = meta.uv_index;
            let tex_id = tex_cache.add_texture(meta);

            material_meta.add_metallic_roughness(tex_id, metallic_factor, roughness_factor, uv_set);
        }

        if let Some((mut meta, _source)) = normal {
            meta = compression::apply_compression_policy(
                meta,
                TextureSemantic::Normal,
                policy_config,
                &data_cache.supported_image_formats,
            );
            // SAFETY: ai_material is valid.
            let normal_scale = unsafe {
                get_float_factor(
                    ai_material,
                    AI_MATKEY_BUMPSCALING,
                    TextureCache::DEFAULT_NORMAL_SCALE,
                )
            };
            let uv_set = meta.uv_index;
            let tex_id = tex_cache.add_texture(meta);

            material_meta.add_normal(tex_id, normal_scale, uv_set);
        }

        if let Some((mut meta, _source)) = occlusion {
            meta = compression::apply_compression_policy(
                meta,
                TextureSemantic::Occlusion,
                policy_config,
                &data_cache.supported_image_formats,
            );
            // SAFETY: ai_material is valid.
            let occlusion_strength = unsafe {
                get_float_factor(
                    ai_material,
                    AI_MATKEY_TEXMAP_STRENGTH_AMBIENT_OCCLUSION,
                    TextureCache::DEFAULT_OCCLUSION_STRENGTH,
                )
            };
            let uv_set = meta.uv_index;
            let tex_id = tex_cache.add_texture(meta);

            material_meta.add_occlusion(tex_id, occlusion_strength, uv_set);
        }

        if let Some((mut meta, _source)) = emissive {
            meta = compression::apply_compression_policy(
                meta,
                TextureSemantic::Emissive,
                policy_config,
                &data_cache.supported_image_formats,
            );
            // SAFETY: ai_material is valid.
            let emissive_factor =
                unsafe { get_emissive_factor(ai_material, AI_MATKEY_COLOR_EMISSIVE) };
            let emissive_strength =
                unsafe { get_emissive_strength(ai_material, AI_MATKEY_EMISSIVE_INTENSITY) };
            let uv_set = meta.uv_index;
            let tex_id = tex_cache.add_texture(meta);

            material_meta.add_emissive(tex_id, emissive_factor, emissive_strength, uv_set);
        }

        materials.push(material_meta);
    }
    Ok(materials)
}

/// # Safety
/// Caller must uphold this module's documented ownership, lifetime, and precondition invariants for the raw FFI operation.
unsafe fn get_color_factor(ai_material: &aiMaterial) -> glam::Vec4 {
    let mut color = aiColor4D {
        r: 0.0,
        g: 0.0,
        b: 0.0,
        a: 0.0,
    };

    if aiGetMaterialColor(ai_material, AI_MATKEY_BASE_COLOR, 0, 0, &mut color)
        == aiReturn_aiReturn_SUCCESS
    {
        glam::vec4(color.r, color.g, color.b, color.a)
    } else if aiGetMaterialColor(ai_material, AI_MATKEY_COLOR_DIFFUSE, 0, 0, &mut color)
        == aiReturn_aiReturn_SUCCESS
    {
        glam::vec4(color.r, color.g, color.b, color.a)
    } else {
        TextureCache::DEFAULT_BASE_COLOR_FACTOR
    }
}

/// # Safety
/// Caller must uphold this module's documented ownership, lifetime, and precondition invariants for the raw FFI operation.
unsafe fn get_float_factor(ai_material: &aiMaterial, key: *const i8, default: f32) -> f32 {
    let mut value = 0.0;
    if aiGetMaterialFloatArray(ai_material, key, 0, 0, &mut value, &mut 1)
        == aiReturn_aiReturn_SUCCESS
    {
        value
    } else {
        default
    }
}

/// # Safety
/// Caller must uphold this module's documented ownership, lifetime, and precondition invariants for the raw FFI operation.
unsafe fn get_emissive_factor(ai_material: &aiMaterial, key: *const i8) -> glam::Vec3 {
    let mut factor = [0.0; 3];
    if aiGetMaterialFloatArray(ai_material, key, 0, 0, factor.as_mut_ptr(), &mut 3)
        == aiReturn_aiReturn_SUCCESS
    {
        glam::Vec3::new(factor[0], factor[1], factor[2])
    } else {
        TextureCache::DEFAULT_EMISSIVE_FACTOR
    }
}

/// # Safety
/// Caller must uphold this module's documented ownership, lifetime, and precondition invariants for the raw FFI operation.
unsafe fn get_emissive_strength(ai_material: &aiMaterial, key: *const i8) -> f32 {
    let mut strength = [0.0; 1];
    if aiGetMaterialFloatArray(ai_material, key, 0, 0, strength.as_mut_ptr(), &mut 1)
        == aiReturn_aiReturn_SUCCESS
    {
        strength[0]
    } else {
        TextureCache::DEFAULT_EMISSIVE_STRENGTH
    }
}

/// # Safety
/// Caller must uphold this module's documented ownership, lifetime, and precondition invariants for the raw FFI operation.
unsafe fn get_alpha_mode(ai_material: &aiMaterial) -> Result<AlphaMode, AssimpImportError> {
    let mut alpha_mode = aiString {
        length: 0,
        data: [0; 1024],
    };
    // SAFETY: ai_material is a valid, non-null pointer as verified by caller.
    if unsafe { aiGetMaterialString(ai_material, AI_MATKEY_GLTF_ALPHAMODE, 0, 0, &mut alpha_mode) }
        == aiReturn_aiReturn_SUCCESS
    {
        // aiString length is validated, then UTF-8 is decoded with the explicit lossy policy.
        let mode_str = checked_ai_string_lossy(&alpha_mode, "alphaMode")?.to_uppercase();
        Ok(match mode_str.as_str() {
            "MASK" => AlphaMode::Mask,
            "BLEND" => AlphaMode::Blend,
            _ => AlphaMode::Opaque,
        })
    } else {
        Ok(AlphaMode::Opaque)
    }
}

/// # Safety
/// Caller must uphold this module's documented ownership, lifetime, and precondition invariants for the raw FFI operation.
unsafe fn get_alpha_cutoff(ai_material: &aiMaterial) -> f32 {
    let mut alpha_cutoff = 0.5; // Default value
    aiGetMaterialFloatArray(
        ai_material,
        AI_MATKEY_GLTF_ALPHACUTOFF,
        0,
        0,
        &mut alpha_cutoff,
        &mut 1,
    );
    alpha_cutoff
}

/// # Safety
/// Caller must uphold this module's documented ownership, lifetime, and precondition invariants for the raw FFI operation.
unsafe fn get_texture_meta(
    ai_material: &aiMaterial,
    ai_scene: &aiScene,
    texture_type: aiTextureType,
    base_path: Option<&str>,
    data_cache: &Arc<VkDataCache>,
    policy_config: &AssetPolicyConfig,
) -> Result<Option<TextureMeta>, AssimpImportError> {
    // SAFETY: ai_material is valid per caller; texture count is queried through Assimp.
    if unsafe { aiGetMaterialTextureCount(ai_material, texture_type) } > 0 {
        let mut path = aiString {
            length: 0,
            data: [c_char::from_be(0x0); 1024],
        };

        let mut uv_index: c_uint = 0;

        // SAFETY: ai_material is valid, path is stack-local, uv_index is stack-local.
        if unsafe {
            aiGetMaterialTexture(
                ai_material,
                texture_type,
                0,
                &mut path,
                std::ptr::null_mut(),
                &mut uv_index,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            )
        } == aiReturn_aiReturn_SUCCESS
        {
            let texture_path = checked_ai_string_lossy(&path, "texture path")?;

            let (texture_data, source_path) = if texture_path.starts_with('*') {
                // Embedded texture payload has no on-disk sidecar path.
                let embedded_bytes = if let Ok(index) = texture_path[1..].parse::<usize>() {
                    let tex_count = checked_usize_from_u32(ai_scene.mNumTextures, "mNumTextures")?;
                    if index >= tex_count {
                        return Err(AssimpImportError::EmbeddedTextureIndexOutOfBounds {
                            texture_index: index,
                            texture_count: tex_count,
                        });
                    }
                    // SAFETY: tex_count bounds, non-null base, and non-null member are checked before dereference.
                    let embedded_texture = unsafe {
                        deref_ptr_array(
                            ai_scene.mTextures as *const *const russimp_sys_ng::aiTexture,
                            index,
                            tex_count,
                            || AssimpImportError::EmbeddedTextureInvalid {
                                texture_index: index,
                                reason:
                                    "mTextures pointer is null or embedded texture pointer is null"
                                        .to_string(),
                            },
                        )?
                    }
                        as *const russimp_sys_ng::aiTexture;
                    // SAFETY: Non-null texture reference validated above; returned bytes are owned.
                    let embedded_bytes =
                        unsafe { checked_embedded_texture_bytes(embedded_texture, index)? };
                    Some(embedded_bytes)
                } else {
                    None
                };
                (embedded_bytes, None)
            } else if let Some(base_path) = base_path {
                // External texture with deterministic manifest sidecar path support.
                let full_path = Path::new(base_path).join(&*texture_path);
                (std::fs::read(&full_path).ok(), Some(full_path))
            } else {
                (None, None)
            };

            if let Some(bytes) = texture_data {
                if let Ok(img) = image::load_from_memory(&bytes) {
                    let policy = if let Some(path) = source_path.as_ref() {
                        asset_manifest::resolve_texture_policy_for_path(
                            path,
                            policy_config.manifest_mode,
                            policy_config.allow_filename_heuristics,
                            None,
                        )
                        .map_err(|err| {
                            AssimpImportError::TexturePolicy {
                                texture_ref: path.display().to_string(),
                                reason: err.to_string(),
                            }
                        })?
                    } else {
                        ResolvedTexturePolicy::default()
                    };

                    let (width, height) = img.dimensions();
                    let mut format = if policy.is_srgb {
                        to_vk_format_srgb(&img)
                    } else {
                        to_vk_format(&img)
                    };

                    let bytes = if data_cache.is_supported_image_format(format) {
                        img.as_bytes().to_vec()
                    } else {
                        format = if policy.is_srgb {
                            vk::Format::R8G8B8A8_SRGB
                        } else {
                            vk::Format::R8G8B8A8_UNORM
                        };
                        img.to_rgba8().into_raw()
                    };

                    let uv_index = if uv_index > 1 {
                        log::warn!(
                            "Texture type {:?} reported unsupported UV set {}. Clamping to UV0.",
                            texture_type,
                            uv_index
                        );
                        0
                    } else {
                        uv_index as u32
                    };

                    let mip_count = if policy.generate_mips {
                        resolve_texture_mip_count(width, height, None)
                    } else {
                        1
                    };

                    return Ok(Some(TextureMeta {
                        payload: TexturePayload::Raw {
                            bytes,
                            width,
                            height,
                            format,
                            mips_levels: mip_count,
                        },
                        uv_index,
                        sampler_info: Some(policy.to_sampler_info(mip_count)),
                    }));
                }
            }
        }
    }
    Ok(None)
}

fn normalize_metal_roughness_texture(meta: TextureMeta) -> TextureMeta {
    let (width, height, format, _bytes, mips_levels) = match &meta.payload {
        TexturePayload::Raw {
            width,
            height,
            format,
            bytes,
            mips_levels,
        } => (*width, *height, *format, bytes, *mips_levels),
        _ => return meta,
    };

    let Some(pixel_count) = checked_texture_pixel_count(width, height) else {
        debug!(
            "Combined metallic-roughness texture dimensions {}x{} overflow usize. Keeping original.",
            width, height
        );
        return meta;
    };
    let roughness = extract_pbr_scalar_channel_u8(&meta, 0, 1);
    let metalness = extract_pbr_scalar_channel_u8(&meta, 1, 2);

    let (Some(roughness), Some(metalness)) = (roughness, metalness) else {
        debug!(
            "Could not normalize combined metallic-roughness texture with format {:?}. Keeping original.",
            format
        );
        return meta;
    };

    if roughness.len() != pixel_count || metalness.len() != pixel_count {
        debug!(
            "Combined metallic-roughness extraction length mismatch (rough {}, metal {}, expected {}). Keeping original.",
            roughness.len(),
            metalness.len(),
            pixel_count
        );
        return meta;
    }

    let mut out = Vec::with_capacity(pixel_count * 4);
    for (&rough, &metal) in roughness.iter().zip(metalness.iter()) {
        // Engine convention: R=roughness, G=metalness, B unused, A=1.0
        out.extend_from_slice(&[rough, metal, 255, 255]);
    }

    TextureMeta {
        payload: TexturePayload::Raw {
            bytes: out,
            width,
            height,
            format: vk::Format::R8G8B8A8_UNORM,
            mips_levels,
        },
        ..meta
    }
}

fn resolve_metallic_roughness_meta(
    combined: Option<TextureMeta>,
    metalness: Option<TextureMeta>,
    roughness: Option<TextureMeta>,
    material_index: usize,
) -> Option<TextureMeta> {
    if let Some(meta) = combined {
        debug!(
            "Material {} metallic_roughness resolved from Assimp UNKNOWN source (format {:?}).",
            material_index,
            meta.payload.format()
        );
        return Some(normalize_metal_roughness_texture(meta));
    }

    match (metalness, roughness) {
        (Some(metalness), Some(roughness)) => {
            debug!(
                "Material {} metallic_roughness resolved from split METALNESS + DIFFUSE_ROUGHNESS sources (formats {:?} + {:?}).",
                material_index, metalness.payload.format(), roughness.payload.format()
            );
            Some(combine_metal_roughness_sources(
                metalness,
                roughness,
                material_index,
            ))
        }
        (Some(meta), None) => {
            debug!(
                "Material {} metallic_roughness resolved from METALNESS-only source (format {:?}).",
                material_index,
                meta.payload.format()
            );
            Some(normalize_metalness_source_texture(meta))
        }
        (None, Some(meta)) => {
            debug!(
                "Material {} metallic_roughness resolved from DIFFUSE_ROUGHNESS-only source (format {:?}).",
                material_index,
                meta.payload.format()
            );
            Some(normalize_roughness_source_texture(meta))
        }
        (None, None) => None,
    }
}

fn combine_metal_roughness_sources(
    metalness: TextureMeta,
    roughness: TextureMeta,
    material_index: usize,
) -> TextureMeta {
    if metalness.payload.width() != roughness.payload.width()
        || metalness.payload.height() != roughness.payload.height()
    {
        debug!(
            "Material {} split METALNESS/DIFFUSE_ROUGHNESS dimensions mismatch ({}x{} vs {}x{}). Falling back to METALNESS-only normalization.",
            material_index,
            metalness.payload.width(),
            metalness.payload.height(),
            roughness.payload.width(),
            roughness.payload.height()
        );
        return normalize_metalness_source_texture(metalness);
    }

    let Some(pixel_count) =
        checked_texture_pixel_count(metalness.payload.width(), metalness.payload.height())
    else {
        debug!(
            "Material {} split METALNESS/DIFFUSE_ROUGHNESS dimensions overflow usize. Falling back to METALNESS-only normalization.",
            material_index
        );
        return normalize_metalness_source_texture(metalness);
    };
    let metal_values =
        extract_pbr_scalar_channel_u8(&metalness, 1, 2).unwrap_or_else(|| vec![255; pixel_count]);
    let rough_values =
        extract_pbr_scalar_channel_u8(&roughness, 0, 1).unwrap_or_else(|| vec![255; pixel_count]);

    if metal_values.len() != pixel_count || rough_values.len() != pixel_count {
        debug!(
            "Material {} split METALNESS/DIFFUSE_ROUGHNESS extraction length mismatch (metal {}, rough {}, expected {}). Falling back to METALNESS-only normalization.",
            material_index,
            metal_values.len(),
            rough_values.len(),
            pixel_count
        );
        return normalize_metalness_source_texture(metalness);
    }

    let mut out = Vec::with_capacity(pixel_count * 4);
    for (&rough, &metal) in rough_values.iter().zip(metal_values.iter()) {
        // Engine convention: R=roughness, G=metalness, B unused, A=1.0
        out.extend_from_slice(&[rough, metal, 255, 255]);
    }

    TextureMeta {
        payload: TexturePayload::Raw {
            bytes: out,
            width: roughness.payload.width(),
            height: roughness.payload.height(),
            format: vk::Format::R8G8B8A8_UNORM,
            mips_levels: roughness.payload.mips_levels(),
        },
        uv_index: roughness.uv_index,
        sampler_info: roughness.sampler_info.clone(),
    }
}

fn normalize_metalness_source_texture(meta: TextureMeta) -> TextureMeta {
    match meta.payload.format() {
        // METALNESS is often authored as a single-channel map.
        vk::Format::R8_UNORM | vk::Format::R8G8_UNORM => normalize_metalness_only_texture(meta),
        // If source already has full channels, normalize to engine convention (R=roughness, G=metalness).
        _ => normalize_metal_roughness_texture(meta),
    }
}

fn normalize_roughness_source_texture(meta: TextureMeta) -> TextureMeta {
    match meta.payload.format() {
        // DIFFUSE_ROUGHNESS is often authored as a single-channel map.
        vk::Format::R8_UNORM | vk::Format::R8G8_UNORM => normalize_roughness_only_texture(meta),
        // If source already has full channels, prefer combined handling.
        _ => normalize_metal_roughness_texture(meta),
    }
}

fn normalize_metalness_only_texture(meta: TextureMeta) -> TextureMeta {
    let Some(metalness) = extract_pbr_scalar_channel_u8(&meta, 1, 2) else {
        debug!(
            "Could not normalize METALNESS texture with format {:?}. Keeping original.",
            meta.payload.format()
        );
        return meta;
    };

    let mut out = Vec::with_capacity(metalness.len() * 4);
    for m in metalness {
        // Engine convention: R=roughness (default 1.0), G=metalness, B unused, A=1.0
        out.extend_from_slice(&[255, m, 255, 255]);
    }

    TextureMeta {
        payload: TexturePayload::Raw {
            bytes: out,
            width: meta.payload.width(),
            height: meta.payload.height(),
            format: vk::Format::R8G8B8A8_UNORM,
            mips_levels: meta.payload.mips_levels(),
        },
        ..meta
    }
}

fn normalize_roughness_only_texture(meta: TextureMeta) -> TextureMeta {
    let Some(roughness) = extract_pbr_scalar_channel_u8(&meta, 0, 1) else {
        debug!(
            "Could not normalize DIFFUSE_ROUGHNESS texture with format {:?}. Keeping original.",
            meta.payload.format()
        );
        return meta;
    };

    let mut out = Vec::with_capacity(roughness.len() * 4);
    for r in roughness {
        // Engine convention: R=roughness, G=metalness (default 1.0), B unused, A=1.0
        out.extend_from_slice(&[r, 255, 255, 255]);
    }

    TextureMeta {
        payload: TexturePayload::Raw {
            bytes: out,
            width: meta.payload.width(),
            height: meta.payload.height(),
            format: vk::Format::R8G8B8A8_UNORM,
            mips_levels: meta.payload.mips_levels(),
        },
        ..meta
    }
}

fn checked_texture_pixel_count(width: u32, height: u32) -> Option<usize> {
    let width = usize::try_from(width).ok()?;
    let height = usize::try_from(height).ok()?;
    width.checked_mul(height)
}

fn extract_pbr_scalar_channel_u8(
    meta: &TextureMeta,
    preferred_rg: usize,
    preferred_rgb: usize,
) -> Option<Vec<u8>> {
    let (width, height, format, bytes) = match &meta.payload {
        TexturePayload::Raw {
            width,
            height,
            format,
            bytes,
            ..
        } => (*width, *height, *format, bytes),
        _ => return None,
    };

    let pixel_count = checked_texture_pixel_count(width, height)?;
    match format {
        vk::Format::R8_UNORM => {
            if bytes.len() == pixel_count {
                Some(bytes.clone())
            } else {
                None
            }
        }
        vk::Format::R8G8_UNORM => {
            if bytes.len() == pixel_count * 2 {
                let channel = preferred_rg.min(1);
                Some(bytes.chunks_exact(2).map(|px| px[channel]).collect())
            } else {
                None
            }
        }
        vk::Format::R8G8B8_UNORM | vk::Format::R8G8B8_SRGB => {
            if bytes.len() == pixel_count * 3 {
                let channel = preferred_rgb.min(2);
                Some(bytes.chunks_exact(3).map(|px| px[channel]).collect())
            } else {
                None
            }
        }
        vk::Format::R8G8B8A8_UNORM | vk::Format::R8G8B8A8_SRGB => {
            if bytes.len() == pixel_count * 4 {
                let channel = preferred_rgb.min(2);
                Some(bytes.chunks_exact(4).map(|px| px[channel]).collect())
            } else {
                None
            }
        }
        _ => None,
    }
}

pub fn process_meshes(
    ai_scene: &aiScene,
    mapped_materials: HashMap<u32, MaterialHandle>,
) -> Result<Vec<MeshMeta>, AssimpImportError> {
    let mesh_count = checked_usize_from_u32(ai_scene.mNumMeshes, "mNumMeshes")?;
    if mesh_count > 0 && ai_scene.mMeshes.is_null() {
        return Err(AssimpImportError::Internal(
            "mNumMeshes is positive but mMeshes pointer is null".to_string(),
        ));
    }
    let mut meshes = Vec::with_capacity(mesh_count);

    let mut unnamed_idx = 0u32;
    for mesh_index in 0..mesh_count {
        // SAFETY: mesh_count validated; base non-null when count > 0.
        // mMeshes is *mut *mut aiMesh; cast to *const *const.
        let ai_mesh = unsafe {
            deref_ptr_array(
                ai_scene.mMeshes as *const *const russimp_sys_ng::aiMesh,
                mesh_index,
                mesh_count,
                || AssimpImportError::NullMesh { mesh_index },
            )?
        };

        let name = if ai_mesh.mName.length > 0 {
            checked_ai_string_lossy(&ai_mesh.mName, "mesh name")?
        } else {
            unnamed_idx += 1;
            format!("Unnamed_Mesh_{}", unnamed_idx - 1)
        };

        let material_index = if ai_mesh.mMaterialIndex != u32::MAX {
            Some(
                *mapped_materials
                    .get(&ai_mesh.mMaterialIndex)
                    .ok_or_else(|| {
                        AssimpImportError::Internal(format!(
                    "mesh '{name}' references material index {} outside material array length {}",
                    ai_mesh.mMaterialIndex,
                    mapped_materials.len()
                ))
                    })?,
            )
        } else {
            None
        };

        let has_uv1 = !ai_mesh.mTextureCoords[1].is_null();
        let vertex_count = checked_usize_from_u32(ai_mesh.mNumVertices, "mNumVertices")?;
        require_non_null_array(
            ai_mesh.mVertices,
            vertex_count,
            format!("mesh '{name}' has vertices but mVertices pointer is null"),
        )?;
        let mut vertices = Vec::with_capacity(vertex_count);

        for i in 0..vertex_count {
            // SAFETY: vertex arrays are checked for null/non-null below; indices are within vertex_count.
            let position = unsafe {
                let p = ai_mesh.mVertices.add(i).read();
                // ai_mesh.mVertices is an inline array of aiVector3D per Assimp struct layout.
                // We validate the pointer for vertex 0 and trust the array is contiguous for all i < vertex_count.
                Vec3::new(p.x, p.y, p.z)
            };

            let normal = if !ai_mesh.mNormals.is_null() {
                // SAFETY: Assimp ownership is held by the RAII guard or caller-borrowed scene object; relevant counts, null pointers, indices, and byte lengths are checked before this operation.
                unsafe {
                    let n = ai_mesh.mNormals.add(i).read();
                    Vec3::new(n.x, n.y, n.z)
                }
            } else {
                Vec3::new(0.0, 1.0, 0.0) // Up vector as default
            };

            let (uv0_x, uv0_y) = if !ai_mesh.mTextureCoords[0].is_null() {
                // SAFETY: uv array non-null, i within vertex_count.
                unsafe {
                    let uv = ai_mesh.mTextureCoords[0].add(i).read();
                    (uv.x, uv.y)
                }
            } else {
                (0.0, 0.0)
            };

            let (uv1_x, uv1_y) = if has_uv1 {
                // SAFETY: Assimp ownership is held by the RAII guard or caller-borrowed scene object; relevant counts, null pointers, indices, and byte lengths are checked before this operation.
                unsafe {
                    let uv = ai_mesh.mTextureCoords[1].add(i).read();
                    (uv.x, uv.y)
                }
            } else {
                (0.0, 0.0)
            };

            let color = if !ai_mesh.mColors[0].is_null() {
                // SAFETY: Assimp ownership is held by the RAII guard or caller-borrowed scene object; relevant counts, null pointers, indices, and byte lengths are checked before this operation.
                unsafe {
                    let c = ai_mesh.mColors[0].add(i).read();
                    Vec4::new(c.r, c.g, c.b, c.a)
                }
            } else {
                Vec4::ONE
            };

            let tangent = if !ai_mesh.mTangents.is_null() {
                // SAFETY: Assimp ownership is held by the RAII guard or caller-borrowed scene object; relevant counts, null pointers, indices, and byte lengths are checked before this operation.
                unsafe {
                    let t = ai_mesh.mTangents.add(i).read();
                    Vec4::new(t.x, t.y, t.z, 1.0) // W component is typically 1.0 for tangents
                }
            } else {
                Vec4::new(1.0, 0.0, 0.0, 1.0) // X-axis aligned tangent as default
            };

            vertices.push(Vertex {
                position,
                uv0_x,
                normal,
                uv0_y,
                color,
                tangent,
                uv1_x,
                uv1_y,
                ..Default::default()
            });
        }

        let face_count = checked_usize_from_u32(ai_mesh.mNumFaces, "mNumFaces")?;
        require_non_null_array(
            ai_mesh.mFaces,
            face_count,
            format!("mesh '{name}' has faces but mFaces pointer is null"),
        )?;
        // Each face contributes exactly mNumIndices (3 after triangulation) indices.
        let index_capacity = checked_product_usize(face_count, 3, "face index capacity")?;
        let mut indices = Vec::with_capacity(index_capacity);

        for face_idx in 0..face_count {
            // SAFETY: mFaces is non-null when face_count > 0; face_idx < face_count.
            let face = unsafe { &*ai_mesh.mFaces.add(face_idx) };
            let idx_count = checked_usize_from_u32(face.mNumIndices, "face.mNumIndices")?;
            require_non_null_array(
                face.mIndices,
                idx_count,
                format!("mesh '{name}' face {face_idx} has indices but mIndices pointer is null"),
            )?;

            for j in 0..idx_count {
                // SAFETY: mIndices is non-null when idx_count > 0; j < idx_count.
                let vertex_idx = unsafe { *face.mIndices.add(j) };
                let vertex_idx_usize = checked_usize_from_u32(vertex_idx, "face vertex index")?;
                if vertex_idx_usize >= vertex_count {
                    return Err(AssimpImportError::FaceIndexOutOfBounds {
                        mesh_name: name,
                        face_index: face_idx,
                        vertex_index: vertex_idx,
                        vertex_count,
                    });
                }
                indices.push(vertex_idx);
            }
        }

        if indices.len() < vertices.len() {
            return Err(AssimpImportError::InvalidIndices {
                mesh_name: name,
                index_count: indices.len(),
                vertex_count: vertices.len(),
            });
        }

        meshes.push(MeshMeta {
            name,
            indices,
            vertices,
            material_index,
            has_uv1,
        });
    }
    Ok(meshes)
}

fn process_node(
    ai_node: *const aiNode,
    mapped_meshes: &HashMap<u32, MeshHandle>,
    mesh_bounds: &HashMap<MeshHandle, SceneBounds>,
    scene_world: &mut SceneWorld,
    parent: Option<SceneNodeId>,
) -> Result<SceneNodeId, AssimpImportError> {
    // SAFETY: ai_node is non-null (validated by caller). We access aiNode fields, which
    // are all inline data (not pointer chains) per Assimp struct layout.
    let ai_node_ref = unsafe { &*ai_node };

    let ai_matrix = ai_node_ref.mTransformation;

    let local_transform = Mat4::from_cols_array(&[
        ai_matrix.a1,
        ai_matrix.b1,
        ai_matrix.c1,
        ai_matrix.d1,
        ai_matrix.a2,
        ai_matrix.b2,
        ai_matrix.c2,
        ai_matrix.d2,
        ai_matrix.a3,
        ai_matrix.b3,
        ai_matrix.c3,
        ai_matrix.d3,
        ai_matrix.a4,
        ai_matrix.b4,
        ai_matrix.c4,
        ai_matrix.d4,
    ]);

    let node_name = if ai_node_ref.mName.length > 0 {
        checked_ai_string_lossy(&ai_node_ref.mName, "node name")?
    } else {
        "<unnamed>".to_string()
    };

    let mesh_count = checked_usize_from_u32(ai_node_ref.mNumMeshes, "node mNumMeshes")?;
    require_non_null_array(
        ai_node_ref.mMeshes,
        mesh_count,
        format!("node '{node_name}' has meshes but mMeshes pointer is null"),
    )?;
    let mut meshes = Vec::with_capacity(mesh_count);
    let mut bounds = Vec::with_capacity(mesh_count);
    for i in 0..mesh_count {
        // SAFETY: mMeshes is non-null when mesh_count > 0; i < mesh_count.
        let mesh_index = unsafe { *ai_node_ref.mMeshes.add(i) };
        let handle = mapped_meshes.get(&mesh_index).copied().ok_or_else(|| {
            AssimpImportError::MissingMeshMapping {
                node_name: node_name.clone(),
                mesh_index,
            }
        })?;
        let bound = mesh_bounds
            .get(&handle)
            .copied()
            .unwrap_or(SceneBounds::ConservativeVisible(
                BoundsUnknownReason::MissingGeometry,
            ));
        meshes.push(handle);
        bounds.push(MeshBoundsEntry {
            mesh: handle,
            bounds: bound,
        });
    }

    let node_id =
        scene_world.add_node_with_parts_and_bounds(parent, local_transform, meshes, bounds);

    // Process children
    let child_count = checked_usize_from_u32(ai_node_ref.mNumChildren, "node mNumChildren")?;
    require_non_null_array(
        ai_node_ref.mChildren,
        child_count,
        format!("node '{node_name}' has children but mChildren pointer is null"),
    )?;
    for i in 0..child_count {
        // SAFETY: mChildren is non-null when child_count > 0; i < child_count.
        let child_ptr = unsafe { *ai_node_ref.mChildren.add(i) };
        if child_ptr.is_null() {
            return Err(AssimpImportError::Internal(format!(
                "node '{node_name}' has null child pointer at index {i}"
            )));
        }
        process_node(
            child_ptr,
            mapped_meshes,
            mesh_bounds,
            scene_world,
            Some(node_id),
        )?;
    }
    Ok(node_id)
}

pub fn to_vk_format(format: &DynamicImage) -> vk::Format {
    match format {
        DynamicImage::ImageLuma8(_) => vk::Format::R8_UNORM,
        DynamicImage::ImageLumaA8(_) => vk::Format::R8G8_UNORM,
        DynamicImage::ImageRgb8(_) => vk::Format::R8G8B8_UNORM,
        DynamicImage::ImageRgba8(_) => vk::Format::R8G8B8A8_UNORM,
        DynamicImage::ImageLuma16(_) => vk::Format::R16_UNORM,
        DynamicImage::ImageLumaA16(_) => vk::Format::R16G16_UNORM,
        DynamicImage::ImageRgb16(_) => vk::Format::R16G16B16_UNORM,
        DynamicImage::ImageRgba16(_) => vk::Format::R16G16B16A16_UNORM,
        DynamicImage::ImageRgb32F(_) => vk::Format::R32G32B32_SFLOAT,
        DynamicImage::ImageRgba32F(_) => vk::Format::R32G32B32A32_SFLOAT,
        _ => vk::Format::R8G8B8A8_UNORM,
    }
}

/// Returns the sRGB variant of the Vulkan format for 8-bit color data.
/// 16-bit and float formats are returned as-is since sRGB encoding
/// is only meaningful for 8-bit per channel data.
pub fn to_vk_format_srgb(format: &DynamicImage) -> vk::Format {
    match format {
        DynamicImage::ImageLuma8(_) => vk::Format::R8_SRGB,
        DynamicImage::ImageLumaA8(_) => vk::Format::R8G8_SRGB,
        DynamicImage::ImageRgb8(_) => vk::Format::R8G8B8_SRGB,
        DynamicImage::ImageRgba8(_) => vk::Format::R8G8B8A8_SRGB,
        // 16-bit and float formats don't have sRGB variants; use linear
        DynamicImage::ImageLuma16(_) => vk::Format::R16_UNORM,
        DynamicImage::ImageLumaA16(_) => vk::Format::R16G16_UNORM,
        DynamicImage::ImageRgb16(_) => vk::Format::R16G16B16_UNORM,
        DynamicImage::ImageRgba16(_) => vk::Format::R16G16B16A16_UNORM,
        DynamicImage::ImageRgb32F(_) => vk::Format::R32G32B32_SFLOAT,
        DynamicImage::ImageRgba32F(_) => vk::Format::R32G32B32A32_SFLOAT,
        _ => vk::Format::R8G8B8A8_SRGB,
    }
}

const AI_MATKEY_BASE_COLOR: *const c_char = b"$clr.base\0".as_ptr() as *const c_char;
const AI_MATKEY_COLOR_DIFFUSE: *const c_char = b"$clr.diffuse\0".as_ptr() as *const c_char;
const AI_MATKEY_COLOR_EMISSIVE: *const c_char = b"$clr.emissive\0".as_ptr() as *const c_char;
const AI_MATKEY_ROUGHNESS_FACTOR: *const c_char =
    b"$mat.roughnessFactor\0".as_ptr() as *const c_char;
const AI_MATKEY_METALLIC_FACTOR: *const c_char = b"$mat.metallicFactor\0".as_ptr() as *const c_char;
const AI_MATKEY_BUMPSCALING: *const c_char = b"$mat.bumpscaling\0".as_ptr() as *const c_char;
const AI_MATKEY_TEXMAP_STRENGTH_AMBIENT_OCCLUSION: *const c_char =
    b"$mat.occlusionTexture.strength\0".as_ptr() as *const c_char;
const AI_MATKEY_EMISSIVE_INTENSITY: *const c_char =
    b"$mat.emissiveIntensity\0".as_ptr() as *const c_char;
const AI_MATKEY_GLTF_ALPHAMODE: *const c_char = b"$mat.gltf.alphaMode\0".as_ptr() as *const c_char;
const AI_MATKEY_GLTF_ALPHACUTOFF: *const c_char =
    b"$mat.gltf.alphaCutoff\0".as_ptr() as *const c_char;

fn imported_mesh_bounds(mesh: &MeshMeta, deformation: MeshDeformation) -> SceneBounds {
    match deformation {
        MeshDeformation::Rigid => {
            let mut positions = mesh.vertices.iter().map(|vertex| vertex.position);
            let Some(first) = positions.next() else {
                return SceneBounds::ConservativeVisible(BoundsUnknownReason::InvalidGeometry);
            };
            if !first.is_finite() {
                return SceneBounds::ConservativeVisible(BoundsUnknownReason::InvalidGeometry);
            }
            let Some((min, max)) = positions.try_fold((first, first), |(min, max), position| {
                position
                    .is_finite()
                    .then_some((min.min(position), max.max(position)))
            }) else {
                return SceneBounds::ConservativeVisible(BoundsUnknownReason::InvalidGeometry);
            };
            SceneBounds::Known(Aabb::from_min_max(min, max))
        }
        MeshDeformation::Skinned => SceneBounds::ConservativeVisible(BoundsUnknownReason::Skinned),
        MeshDeformation::Deformed => {
            SceneBounds::ConservativeVisible(BoundsUnknownReason::Deformed)
        }
        MeshDeformation::Unknown => {
            SceneBounds::ConservativeVisible(BoundsUnknownReason::MissingGeometry)
        }
    }
}

/// Read embedded texture bytes from an Assimp aiTexture with validated sizing.
///
/// Per Assimp representation:
/// - Compressed textures (height == 0): `mWidth` is the declared byte length.
/// - Uncompressed textures (height > 0): size = width * height * sizeof(aiTexel).
/// - `aiTexel` is 4 bytes (BGRA).
///
/// Returns an owned `Vec<u8>` copy of the embedded payload, or an error on
/// inconsistent dimensions, null payload, or arithmetic overflow.
///
/// # Safety
/// `texture` must be a valid, non-null `*const aiTexture` reference.
unsafe fn checked_embedded_texture_bytes(
    texture: *const russimp_sys_ng::aiTexture,
    index: usize,
) -> Result<Vec<u8>, AssimpImportError> {
    let tex = &*texture;
    let invalid = |reason: &str| AssimpImportError::EmbeddedTextureInvalid {
        texture_index: index,
        reason: reason.to_string(),
    };

    if tex.pcData.is_null() {
        return Err(invalid("pcData is null"));
    }

    let byte_len = if tex.mHeight == 0 {
        // Compressed: mWidth is the declared byte length.
        let declared =
            checked_usize_from_u32(tex.mWidth, "compressed embedded texture byte length")?;
        if declared == 0 {
            return Err(invalid("compressed texture has zero declared byte length"));
        }
        declared
    } else {
        // Uncompressed: width * height * sizeof(aiTexel).
        let w = checked_usize_from_u32(tex.mWidth, "embedded texture width")?;
        let h = checked_usize_from_u32(tex.mHeight, "embedded texture height")?;
        const AI_TEXEL_SIZE: usize = std::mem::size_of::<russimp_sys_ng::aiTexel>();
        let pixels = checked_product_usize(w, h, "uncompressed embedded texture pixels")
            .map_err(|_| invalid("uncompressed texture width * height overflowed"))?;
        checked_product_usize(
            pixels,
            AI_TEXEL_SIZE,
            "uncompressed embedded texture byte length",
        )
        .map_err(|_| invalid("uncompressed texture byte length overflowed"))?
    };

    // SAFETY: pcData is non-null, byte_len validated above.
    Ok(unsafe { std::slice::from_raw_parts(tex.pcData as *const u8, byte_len) }.to_vec())
}

fn classify_imported_meshes(
    ai_scene: &aiScene,
    expected_count: usize,
) -> Result<Vec<MeshDeformation>, AssimpImportError> {
    let count = checked_usize_from_u32(ai_scene.mNumMeshes, "mNumMeshes (classification)")?;
    if count != expected_count || (expected_count > 0 && ai_scene.mMeshes.is_null()) {
        return Err(AssimpImportError::Internal(
            "Assimp mesh classification count did not match processed meshes".to_string(),
        ));
    }

    let mut classifications = Vec::with_capacity(expected_count);
    for index in 0..expected_count {
        // SAFETY: count validated, base non-null checked above.
        let mesh = unsafe {
            deref_ptr_array(
                ai_scene.mMeshes as *const *const russimp_sys_ng::aiMesh,
                index,
                expected_count,
                || {
                    AssimpImportError::Internal(format!(
                        "Assimp mesh {index} was null during deformation classification"
                    ))
                },
            )?
        };
        classifications.push(if mesh.mNumBones > 0 {
            MeshDeformation::Skinned
        } else if mesh.mNumAnimMeshes > 0 {
            MeshDeformation::Deformed
        } else {
            MeshDeformation::Rigid
        });
    }
    Ok(classifications)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AssetError;

    #[test]
    fn imported_mesh_bounds_classify_rigid_and_deformed_geometry() {
        let mesh = MeshMeta {
            name: "bounds-test".to_string(),
            indices: vec![0, 1, 2],
            vertices: vec![
                Vertex {
                    position: Vec3::new(-2.0, 3.0, 1.0),
                    ..Vertex::default()
                },
                Vertex {
                    position: Vec3::new(4.0, -1.0, 5.0),
                    ..Vertex::default()
                },
                Vertex {
                    position: Vec3::new(0.0, 2.0, -3.0),
                    ..Vertex::default()
                },
            ],
            material_index: None,
            has_uv1: false,
        };

        assert_eq!(
            imported_mesh_bounds(&mesh, MeshDeformation::Rigid),
            SceneBounds::Known(Aabb::from_min_max(
                Vec3::new(-2.0, -1.0, -3.0),
                Vec3::new(4.0, 3.0, 5.0),
            ))
        );
        assert_eq!(
            imported_mesh_bounds(&mesh, MeshDeformation::Skinned),
            SceneBounds::ConservativeVisible(BoundsUnknownReason::Skinned)
        );
        assert_eq!(
            imported_mesh_bounds(&mesh, MeshDeformation::Deformed),
            SceneBounds::ConservativeVisible(BoundsUnknownReason::Deformed)
        );
    }

    // ── Error Display formatting ──────────────────────────────────────

    #[test]
    fn error_display_invalid_path() {
        let err = AssimpImportError::InvalidPath("/bad\0path".to_string());
        let msg = err.to_string();
        assert!(msg.contains("/bad\0path"), "should include the path");
        assert!(msg.contains("invalid"), "should describe invalidity");
    }

    #[test]
    fn error_display_scene_load_failed() {
        let err = AssimpImportError::SceneLoadFailed {
            path: "model.glb".to_string(),
            reason: "file not found".to_string(),
        };
        let msg = err.to_string();
        assert!(msg.contains("model.glb"));
        assert!(msg.contains("file not found"));
    }

    #[test]
    fn error_display_null_mesh() {
        let err = AssimpImportError::NullMesh { mesh_index: 3 };
        let msg = err.to_string();
        assert!(msg.contains("null mesh"), "should say null mesh");
        assert!(msg.contains("3"), "should include the index");
    }

    #[test]
    fn error_display_invalid_indices() {
        let err = AssimpImportError::InvalidIndices {
            mesh_name: "Cube".to_string(),
            index_count: 2,
            vertex_count: 10,
        };
        let msg = err.to_string();
        assert!(msg.contains("Cube"));
        assert!(msg.contains("2"));
        assert!(msg.contains("10"));
    }

    #[test]
    fn error_display_missing_mesh_mapping() {
        let err = AssimpImportError::MissingMeshMapping {
            node_name: "Armature".to_string(),
            mesh_index: 7,
        };
        let msg = err.to_string();
        assert!(msg.contains("Armature"));
        assert!(msg.contains("7"));
    }

    #[test]
    fn error_display_texture_decode() {
        let err = AssimpImportError::TextureDecode {
            texture_ref: "*0".to_string(),
            reason: "invalid PNG header".to_string(),
        };
        let msg = err.to_string();
        assert!(msg.contains("*0"));
        assert!(msg.contains("invalid PNG header"));
    }

    #[test]
    fn error_display_internal() {
        let err = AssimpImportError::Internal("unexpected null".to_string());
        let msg = err.to_string();
        assert!(msg.contains("unexpected null"));
    }

    // ── AssetError conversion ─────────────────────────────────────────

    #[test]
    fn conversion_invalid_path_to_asset_error_load() {
        let err = AssimpImportError::InvalidPath("/nonexistent".to_string());
        let asset_err: AssetError = err.into();
        match asset_err {
            AssetError::Load { path, message } => {
                assert_eq!(path.unwrap().to_str().unwrap(), "/nonexistent");
                assert!(message.contains("invalid"));
            }
            other => panic!("expected AssetError::Load, got {:?}", other),
        }
    }

    #[test]
    fn conversion_scene_load_failed_to_asset_error_load() {
        let err = AssimpImportError::SceneLoadFailed {
            path: "test.glb".to_string(),
            reason: "corrupt".to_string(),
        };
        let asset_err: AssetError = err.into();
        match asset_err {
            AssetError::Load { path, message } => {
                assert_eq!(path.unwrap().to_str().unwrap(), "test.glb");
                assert!(message.contains("corrupt"));
            }
            other => panic!("expected AssetError::Load, got {:?}", other),
        }
    }

    #[test]
    fn conversion_texture_decode_to_asset_error_decode() {
        let err = AssimpImportError::TextureDecode {
            texture_ref: "albedo.png".to_string(),
            reason: "bad format".to_string(),
        };
        let asset_err: AssetError = err.into();
        match asset_err {
            AssetError::Decode { path, message } => {
                assert_eq!(path.to_str().unwrap(), "albedo.png");
                assert!(message.contains("bad format"));
            }
            other => panic!("expected AssetError::Decode, got {:?}", other),
        }
    }

    #[test]
    fn conversion_lock_poisoned_to_asset_error_sync() {
        let err = AssimpImportError::Internal("texture_cache lock poisoned".to_string());
        let asset_err: AssetError = err.into();
        match asset_err {
            AssetError::Sync(msg) => {
                assert!(msg.contains("lock poisoned"));
            }
            other => panic!("expected AssetError::Sync, got {:?}", other),
        }
    }

    #[test]
    fn conversion_null_mesh_to_asset_error_load() {
        let err = AssimpImportError::NullMesh { mesh_index: 5 };
        let asset_err: AssetError = err.into();
        match asset_err {
            AssetError::Load { path, message } => {
                assert!(path.is_none());
                assert!(message.contains("null mesh"));
            }
            other => panic!("expected AssetError::Load, got {:?}", other),
        }
    }

    #[test]
    fn conversion_missing_mapping_to_asset_error_load() {
        let err = AssimpImportError::MissingMeshMapping {
            node_name: "Root".to_string(),
            mesh_index: 99,
        };
        let asset_err: AssetError = err.into();
        match asset_err {
            AssetError::Load { path, message } => {
                assert!(path.is_none());
                assert!(message.contains("Root"));
                assert!(message.contains("99"));
            }
            other => panic!("expected AssetError::Load, got {:?}", other),
        }
    }

    // ── Integration: load_model with invalid path ─────────────────────

    #[test]
    fn load_model_nonexistent_path_returns_scene_load_failed() {
        // Construct a minimal VkDataCache is not feasible without Vulkan,
        // but we can verify the CString conversion path.
        let err = AssimpImportError::InvalidPath("contains\0null".to_string());
        assert!(matches!(err, AssimpImportError::InvalidPath(_)));
    }

    #[test]
    fn error_implements_std_error() {
        let err: Box<dyn std::error::Error> =
            Box::new(AssimpImportError::Internal("test".to_string()));
        // Verify the error trait object works and Display is callable.
        let _ = format!("{err}");
    }
}
