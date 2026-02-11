use crate::data::data_cache::{MeshCache, TextureCache, VkDataCache};
use crate::data::gpu_data;
use crate::data::gpu_data::{
    AlphaMode, EmissiveMap, MaterialMeta, MeshMeta, Node, NormalMap, OcclusionMap, Sampler,
    SurfaceMeta, TextureMeta, Transform, Vertex,
};
use ash::vk;
use glam::{Mat4, Quat, Vec3, Vec4};
use image::{DynamicImage, GenericImageView};
use log::__private_api::loc;
use log::debug;
use russimp_sys::{
    aiColor4D, aiCreatePropertyStore, aiGetMaterialColor, aiGetMaterialFloatArray,
    aiGetMaterialString, aiGetMaterialTexture, aiGetMaterialTextureCount, aiImportFile,
    aiImportFileExWithProperties, aiMaterial, aiNode, aiPostProcessSteps,
    aiPostProcessSteps_aiProcess_CalcTangentSpace, aiPostProcessSteps_aiProcess_FixInfacingNormals,
    aiPostProcessSteps_aiProcess_FlipUVs, aiPostProcessSteps_aiProcess_GenSmoothNormals,
    aiPostProcessSteps_aiProcess_JoinIdenticalVertices,
    aiPostProcessSteps_aiProcess_LimitBoneWeights,
    aiPostProcessSteps_aiProcess_PreTransformVertices, aiPostProcessSteps_aiProcess_Triangulate,
    aiReturn_aiReturn_SUCCESS, aiScene, aiSetImportPropertyInteger,
    aiShadingMode_aiShadingMode_PBR_BRDF, aiString, aiTexture, aiTextureType,
    aiTextureType_aiTextureType_AMBIENT, aiTextureType_aiTextureType_AMBIENT_OCCLUSION,
    aiTextureType_aiTextureType_BASE_COLOR, aiTextureType_aiTextureType_DIFFUSE,
    aiTextureType_aiTextureType_DIFFUSE_ROUGHNESS, aiTextureType_aiTextureType_EMISSIVE,
    aiTextureType_aiTextureType_HEIGHT, aiTextureType_aiTextureType_LIGHTMAP,
    aiTextureType_aiTextureType_METALNESS, aiTextureType_aiTextureType_NORMALS,
    aiTextureType_aiTextureType_NORMAL_CAMERA, aiTextureType_aiTextureType_SPECULAR,
    aiTextureType_aiTextureType_UNKNOWN, ai_real, AI_DEFAULT_MATERIAL_NAME,
};
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::default::Default;
use std::ffi::{c_char, c_uint, CStr, CString};
use std::fs::File;
use std::hash::{DefaultHasher, Hasher};
use std::io::Read;
use std::os::raw;
use std::os::raw::c_int;
use std::path::Path;
use std::rc::Rc;
use std::sync::Arc;

pub struct ModelMeta {
    pub node: Rc<RefCell<gpu_data::Node>>,
    pub material_ids: Vec<u32>,
    pub mesh_ids: Vec<u32>,
}

pub fn load_model(
    path: &str,
    data_cache: Arc<VkDataCache>,
    has_animation: bool,
) -> Result<ModelMeta, String> {
    let mut flags = aiPostProcessSteps_aiProcess_GenSmoothNormals
        | aiPostProcessSteps_aiProcess_JoinIdenticalVertices
        | aiPostProcessSteps_aiProcess_Triangulate
        | aiPostProcessSteps_aiProcess_FlipUVs
        //   | aiPostProcessSteps_aiProcess_PreTransformVertices
        | aiPostProcessSteps_aiProcess_FixInfacingNormals
        | aiPostProcessSteps_aiProcess_CalcTangentSpace;
    //  | aiPostProcessSteps_aiProcess_LimitBoneWeights;

    if has_animation {
        flags = flags | aiPostProcessSteps_aiProcess_PreTransformVertices;
    };

    let props = unsafe { aiCreatePropertyStore() };
    const AI_CONFIG_IMPORT_NO_OVERWRITE_NORMALS: &[u8] = b"IMPORT_NO_OVERWRITE_NORMALS\0";
    unsafe {
        aiSetImportPropertyInteger(
            props,
            AI_CONFIG_IMPORT_NO_OVERWRITE_NORMALS.as_ptr() as *const c_char,
            1,
        );
    }

    let path_c = CString::new(path).map_err(|err| "Failed: &str -> CString".to_string())?;
    let base_path = Path::new(path)
        .parent()
        .map_or_else(|| None, |p| p.to_str());

    let ai_scene: &aiScene = unsafe {
        let scene_ptr = aiImportFileExWithProperties(
            path_c.as_ptr(),
            flags as c_uint,
            std::ptr::null_mut(),
            props,
        );

        if scene_ptr.is_null() {
            Err("Error loading scene file, invalid path?".to_string())
        } else {
            Ok(&*scene_ptr)
        }
    }?;

    let materials = process_materials(ai_scene, base_path, &data_cache)?;
    let mat_indices = 0..materials.len();

    let mut mapped_materials = HashMap::<u32, u32>::with_capacity(materials.len());
    let material_ids = data_cache
        .texture_cache
        .lock()
        .unwrap()
        .add_materials(materials);

    for (og_idx, id) in mat_indices.into_iter().zip(material_ids.iter()) {
        mapped_materials.insert(og_idx as u32, *id);
    }

    let meshes = process_meshes(ai_scene, mapped_materials);
    let mesh_indices = 0..meshes.len();

    let mut mapped_meshes = HashMap::<u32, u32>::with_capacity(meshes.len());
    let mesh_ids = data_cache.mesh_cache.lock().unwrap().add_multi(meshes);

    for (og_idx, id) in mesh_indices.into_iter().zip(mesh_ids.iter()) {
        mapped_meshes.insert(og_idx as u32, *id);
    }

    let root_ai_node = (*ai_scene).mRootNode;

    if root_ai_node.is_null() {
        Err("Failed to find root node in model".to_string())
    } else {
        let node = process_node(root_ai_node, &mapped_meshes, None);
        Ok(ModelMeta {
            node,
            material_ids,
            mesh_ids,
        })
    }
}

pub fn process_materials(
    ai_scene: &aiScene,
    base_path: Option<&str>,
    data_cache: &Arc<VkDataCache>,
) -> Result<Vec<MaterialMeta>, String> {
    let mat_count = ai_scene.mNumMaterials as usize;
    let mut materials = Vec::<MaterialMeta>::with_capacity(mat_count);

    unsafe {
        for i in 0..mat_count {
            let ai_material_ptr = *ai_scene.mMaterials.add(i);

            if ai_material_ptr.is_null() {
                return Err(format!("Failed to resolve pointer for material: {}", i));
            }

            let ai_material = &*ai_material_ptr;
            let mut material_meta = MaterialMeta::default();

            let load_first_texture_type = |label: &str, types: &[(aiTextureType, &'static str)]| {
                for (typ, type_name) in types {
                    if let Some(meta) =
                        get_texture_meta(ai_material, ai_scene, *typ, base_path, data_cache)
                    {
                        debug!(
                            "Material {} {} texture resolved from Assimp type {}",
                            i, label, type_name
                        );
                        return Some((meta, *type_name));
                    }
                }
                None
            };

            let base_color = load_first_texture_type(
                "base_color",
                &[
                    (aiTextureType_aiTextureType_BASE_COLOR, "BASE_COLOR"),
                    (aiTextureType_aiTextureType_DIFFUSE, "DIFFUSE"),
                ],
            );

            // Assimp may expose metallic/roughness as either:
            // - one combined texture (UNKNOWN), or
            // - two split sources (METALNESS + DIFFUSE_ROUGHNESS).
            let met_rough_combined = get_texture_meta(
                ai_material,
                ai_scene,
                aiTextureType_aiTextureType_UNKNOWN,
                base_path,
                data_cache,
            );
            let met_rough_metalness = get_texture_meta(
                ai_material,
                ai_scene,
                aiTextureType_aiTextureType_METALNESS,
                base_path,
                data_cache,
            );
            let met_rough_roughness = get_texture_meta(
                ai_material,
                ai_scene,
                aiTextureType_aiTextureType_DIFFUSE_ROUGHNESS,
                base_path,
                data_cache,
            );

            let normal = load_first_texture_type(
                "normal",
                &[
                    (aiTextureType_aiTextureType_NORMAL_CAMERA, "NORMAL_CAMERA"),
                    (aiTextureType_aiTextureType_NORMALS, "NORMALS"),
                ],
            );

            let occlusion = load_first_texture_type(
                "occlusion",
                &[
                    (
                        aiTextureType_aiTextureType_AMBIENT_OCCLUSION,
                        "AMBIENT_OCCLUSION",
                    ),
                    (aiTextureType_aiTextureType_LIGHTMAP, "LIGHTMAP"),
                ],
            );

            let emissive = load_first_texture_type(
                "emissive",
                &[(aiTextureType_aiTextureType_EMISSIVE, "EMISSIVE")],
            );

            let mut tex_cache = data_cache.texture_cache.lock().unwrap();

            if let Some((meta, _source)) = base_color {
                let color = get_color_factor(ai_material);
                let uv_set = meta.uv_index;
                let tex_id = tex_cache.add_texture(meta);
                material_meta.add_base_color(tex_id, color, uv_set);
            }

            if let Some(meta) = resolve_metallic_roughness_meta(
                met_rough_combined,
                met_rough_metalness,
                met_rough_roughness,
                i,
            ) {
                let metallic_factor = get_float_factor(ai_material, AI_MATKEY_METALLIC_FACTOR, 1.0);
                let roughness_factor = get_float_factor(
                    ai_material,
                    AI_MATKEY_ROUGHNESS_FACTOR,
                    TextureCache::DEFAULT_ROUGHNESS_FACTOR,
                );
                let uv_set = meta.uv_index;
                let tex_id = tex_cache.add_texture(meta);

                material_meta.add_metallic_roughness(
                    tex_id,
                    metallic_factor,
                    roughness_factor,
                    uv_set,
                );
            }

            if let Some((meta, _source)) = normal {
                let normal_scale = get_float_factor(
                    ai_material,
                    AI_MATKEY_BUMPSCALING,
                    TextureCache::DEFAULT_NORMAL_SCALE,
                );
                let uv_set = meta.uv_index;
                let tex_id = tex_cache.add_texture(meta);

                material_meta.add_normal(tex_id, normal_scale, uv_set);
            }

            if let Some((meta, _source)) = occlusion {
                let occlusion_strength = get_float_factor(
                    ai_material,
                    AI_MATKEY_TEXMAP_STRENGTH_AMBIENT_OCCLUSION,
                    TextureCache::DEFAULT_OCCLUSION_STRENGTH,
                );
                let uv_set = meta.uv_index;
                let tex_id = tex_cache.add_texture(meta);

                material_meta.add_occlusion(tex_id, occlusion_strength, uv_set);
            }

            if let Some((meta, _source)) = emissive {
                let emissive_factor = get_emissive_factor(ai_material, AI_MATKEY_COLOR_EMISSIVE);
                let emissive_strength =
                    get_emissive_strength(ai_material, AI_MATKEY_EMISSIVE_INTENSITY);
                let uv_set = meta.uv_index;
                let tex_id = tex_cache.add_texture(meta);

                material_meta.add_emissive(tex_id, emissive_factor, emissive_strength, uv_set);
            }

            materials.push(material_meta);
        }
    }
    Ok(materials)
}

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

unsafe fn get_alpha_mode(ai_material: &aiMaterial) -> AlphaMode {
    let mut alpha_mode = aiString {
        length: 0,
        data: [0; 1024],
    };
    if aiGetMaterialString(ai_material, AI_MATKEY_GLTF_ALPHAMODE, 0, 0, &mut alpha_mode)
        == aiReturn_aiReturn_SUCCESS
    {
        let mode_str = CStr::from_ptr(alpha_mode.data.as_ptr())
            .to_string_lossy()
            .to_uppercase();
        match mode_str.as_str() {
            "MASK" => AlphaMode::Mask,
            "BLEND" => AlphaMode::Blend,
            _ => AlphaMode::Opaque,
        }
    } else {
        AlphaMode::Opaque
    }
}

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

unsafe fn get_texture_meta(
    ai_material: &aiMaterial,
    ai_scene: &aiScene,
    texture_type: aiTextureType,
    base_path: Option<&str>,
    data_cache: &Arc<VkDataCache>,
) -> Option<TextureMeta> {
    if aiGetMaterialTextureCount(ai_material, texture_type) > 0 {
        let mut path = aiString {
            length: 0,
            data: [c_char::from_be(0x0); 1024],
        };

        let mut uv_index: c_uint = 0;

        if aiGetMaterialTexture(
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
        ) == aiReturn_aiReturn_SUCCESS
        {
            let texture_path = CStr::from_ptr(path.data.as_ptr()).to_string_lossy();

            let texture_data = if texture_path.starts_with("*") {
                // Embedded texture .....This is a doozy
                if let Ok(index) = texture_path[1..].parse::<usize>() {
                    if index < ai_scene.mNumTextures as usize {
                        let embedded_texture = *ai_scene.mTextures.add(index);
                        if !embedded_texture.is_null() {
                            let texture = &*embedded_texture;
                            Some(
                                std::slice::from_raw_parts(
                                    texture.pcData as *const u8,
                                    texture.mWidth as usize,
                                )
                                .to_vec(),
                            )
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                } else {
                    None
                }
            } else if let Some(base_path) = base_path {
                // External texture
                let full_path = Path::new(base_path).join(texture_path.as_ref());
                std::fs::read(full_path).ok()
            } else {
                None
            };

            if let Some(bytes) = texture_data {
                if let Ok(img) = image::load_from_memory(&bytes) {
                    let (width, height) = img.dimensions();
                    let mut format = to_vk_format(&img);

                    let bytes = if data_cache.is_supported_image_format(format) {
                        img.as_bytes().to_vec()
                    } else {
                        format = vk::Format::R8G8B8A8_UNORM;
                        img.to_rgba8().into_raw()
                    };

                    let uv_index = if uv_index > 1 {
                        debug!(
                            "Texture type {:?} reported unsupported UV set {}. Clamping to UV0.",
                            texture_type, uv_index
                        );
                        0
                    } else {
                        uv_index as u32
                    };

                    return Some(TextureMeta {
                        bytes,
                        width,
                        height,
                        format,
                        mips_levels: 1,
                        uv_index,
                    });
                }
            }
        }
    }
    None
}

fn normalize_metal_roughness_texture(meta: TextureMeta) -> TextureMeta {
    let pixel_count = (meta.width as usize) * (meta.height as usize);

    match meta.format {
        // Single channel METALNESS map: place metal in B and default roughness to 1.0 in G.
        vk::Format::R8_UNORM => {
            if meta.bytes.len() != pixel_count {
                debug!(
                    "Unexpected R8 metallic/roughness byte length (got {}, expected {}). Keeping original.",
                    meta.bytes.len(),
                    pixel_count
                );
                return meta;
            }

            let mut out = Vec::with_capacity(pixel_count * 4);
            for &m in &meta.bytes {
                out.extend_from_slice(&[255, 255, m, 255]);
            }

            TextureMeta {
                bytes: out,
                format: vk::Format::R8G8B8A8_UNORM,
                ..meta
            }
        }
        // Two channel map: interpret R as roughness, G as metalness.
        vk::Format::R8G8_UNORM => {
            if meta.bytes.len() != pixel_count * 2 {
                debug!(
                    "Unexpected R8G8 metallic/roughness byte length (got {}, expected {}). Keeping original.",
                    meta.bytes.len(),
                    pixel_count * 2
                );
                return meta;
            }

            let mut out = Vec::with_capacity(pixel_count * 4);
            for px in meta.bytes.chunks_exact(2) {
                let roughness = px[0];
                let metal = px[1];
                out.extend_from_slice(&[255, roughness, metal, 255]);
            }

            TextureMeta {
                bytes: out,
                format: vk::Format::R8G8B8A8_UNORM,
                ..meta
            }
        }
        _ => meta,
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
            material_index, meta.format
        );
        return Some(normalize_metal_roughness_texture(meta));
    }

    match (metalness, roughness) {
        (Some(metalness), Some(roughness)) => {
            debug!(
                "Material {} metallic_roughness resolved from split METALNESS + DIFFUSE_ROUGHNESS sources (formats {:?} + {:?}).",
                material_index, metalness.format, roughness.format
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
                material_index, meta.format
            );
            Some(normalize_metalness_source_texture(meta))
        }
        (None, Some(meta)) => {
            debug!(
                "Material {} metallic_roughness resolved from DIFFUSE_ROUGHNESS-only source (format {:?}).",
                material_index, meta.format
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
    if metalness.width != roughness.width || metalness.height != roughness.height {
        debug!(
            "Material {} split METALNESS/DIFFUSE_ROUGHNESS dimensions mismatch ({}x{} vs {}x{}). Falling back to METALNESS-only normalization.",
            material_index,
            metalness.width,
            metalness.height,
            roughness.width,
            roughness.height
        );
        return normalize_metalness_source_texture(metalness);
    }

    let pixel_count = (metalness.width as usize) * (metalness.height as usize);
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
        // R reserved, G=roughness, B=metalness, A=1.0
        out.extend_from_slice(&[255, rough, metal, 255]);
    }

    TextureMeta {
        bytes: out,
        width: roughness.width,
        height: roughness.height,
        format: vk::Format::R8G8B8A8_UNORM,
        mips_levels: roughness.mips_levels,
        uv_index: roughness.uv_index,
    }
}

fn normalize_metalness_source_texture(meta: TextureMeta) -> TextureMeta {
    match meta.format {
        // METALNESS is often authored as a single-channel map.
        vk::Format::R8_UNORM | vk::Format::R8G8_UNORM => normalize_metalness_only_texture(meta),
        // If source already has full channels, prefer combined handling (g=roughness, b=metalness).
        _ => normalize_metal_roughness_texture(meta),
    }
}

fn normalize_roughness_source_texture(meta: TextureMeta) -> TextureMeta {
    match meta.format {
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
            meta.format
        );
        return meta;
    };

    let mut out = Vec::with_capacity(metalness.len() * 4);
    for m in metalness {
        // R unused, G=roughness (default 1.0), B=metalness, A=1.0
        out.extend_from_slice(&[255, 255, m, 255]);
    }

    TextureMeta {
        bytes: out,
        format: vk::Format::R8G8B8A8_UNORM,
        ..meta
    }
}

fn normalize_roughness_only_texture(meta: TextureMeta) -> TextureMeta {
    let Some(roughness) = extract_pbr_scalar_channel_u8(&meta, 0, 1) else {
        debug!(
            "Could not normalize DIFFUSE_ROUGHNESS texture with format {:?}. Keeping original.",
            meta.format
        );
        return meta;
    };

    let mut out = Vec::with_capacity(roughness.len() * 4);
    for r in roughness {
        // R unused, G=roughness, B=metalness (default 1.0), A=1.0
        out.extend_from_slice(&[255, r, 255, 255]);
    }

    TextureMeta {
        bytes: out,
        format: vk::Format::R8G8B8A8_UNORM,
        ..meta
    }
}

fn extract_pbr_scalar_channel_u8(
    meta: &TextureMeta,
    preferred_rg: usize,
    preferred_rgb: usize,
) -> Option<Vec<u8>> {
    let pixel_count = (meta.width as usize) * (meta.height as usize);
    match meta.format {
        vk::Format::R8_UNORM => {
            if meta.bytes.len() == pixel_count {
                Some(meta.bytes.clone())
            } else {
                None
            }
        }
        vk::Format::R8G8_UNORM => {
            if meta.bytes.len() == pixel_count * 2 {
                let channel = preferred_rg.min(1);
                Some(meta.bytes.chunks_exact(2).map(|px| px[channel]).collect())
            } else {
                None
            }
        }
        vk::Format::R8G8B8_UNORM | vk::Format::R8G8B8_SRGB => {
            if meta.bytes.len() == pixel_count * 3 {
                let channel = preferred_rgb.min(2);
                Some(meta.bytes.chunks_exact(3).map(|px| px[channel]).collect())
            } else {
                None
            }
        }
        vk::Format::R8G8B8A8_UNORM | vk::Format::R8G8B8A8_SRGB => {
            if meta.bytes.len() == pixel_count * 4 {
                let channel = preferred_rgb.min(2);
                Some(meta.bytes.chunks_exact(4).map(|px| px[channel]).collect())
            } else {
                None
            }
        }
        _ => None,
    }
}

pub fn process_meshes(ai_scene: &aiScene, mapped_materials: HashMap<u32, u32>) -> Vec<MeshMeta> {
    let mesh_count = ai_scene.mNumMeshes as usize;
    let mut meshes = Vec::with_capacity(mesh_count);

    unsafe {
        let mut unnamed_idx = 0;
        for mesh_index in 0..mesh_count {
            let ai_mesh_ptr = *ai_scene.mMeshes.add(mesh_index);
            if ai_mesh_ptr.is_null() {
                eprintln!("Skipping null mesh at index {}", mesh_index);
                continue;
            }

            let ai_mesh = &*ai_mesh_ptr;

            let name = if ai_mesh.mName.length > 0 {
                CStr::from_ptr(ai_mesh.mName.data.as_ptr())
                    .to_string_lossy()
                    .into_owned()
            } else {
                unnamed_idx += 1;
                format!("Unnamed_Mesh_{}", unnamed_idx - 1)
            };

            let material_index = if ai_mesh.mMaterialIndex != u32::MAX {
                Some(
                    *mapped_materials
                        .get(&(ai_mesh.mMaterialIndex as u32))
                        .unwrap_or(&TextureCache::DEFAULT_ERROR_MAT),
                )
            } else {
                None
            };

            let vertex_count = ai_mesh.mNumVertices as usize;
            let mut vertices = Vec::with_capacity(vertex_count);

            for i in 0..vertex_count {
                let position = Vec3::new(
                    ai_mesh.mVertices.add(i).read().x,
                    ai_mesh.mVertices.add(i).read().y,
                    ai_mesh.mVertices.add(i).read().z,
                );

                let normal = if !ai_mesh.mNormals.is_null() {
                    Vec3::new(
                        ai_mesh.mNormals.add(i).read().x,
                        ai_mesh.mNormals.add(i).read().y,
                        ai_mesh.mNormals.add(i).read().z,
                    )
                } else {
                    Vec3::new(0.0, 1.0, 0.0) // Up vector as default
                };

                let (uv0_x, uv0_y) = if !ai_mesh.mTextureCoords[0].is_null() {
                    let uv = ai_mesh.mTextureCoords[0].add(i).read();
                    (uv.x, uv.y)
                } else {
                    (0.0, 0.0)
                };

                let (uv1_x, uv1_y) = if !ai_mesh.mTextureCoords[1].is_null() {
                    let uv = ai_mesh.mTextureCoords[1].add(i).read();
                    (uv.x, uv.y)
                } else {
                    (0.0, 0.0)
                };

                let color = if !ai_mesh.mColors[0].is_null() {
                    let color = ai_mesh.mColors[0].add(i).read();
                    Vec4::new(color.r, color.g, color.b, color.a)
                } else {
                    Vec4::ONE
                };

                let tangent = if !ai_mesh.mTangents.is_null() {
                    let t = ai_mesh.mTangents.add(i).read();
                    Vec4::new(t.x, t.y, t.z, 1.0) // W component is typically 1.0 for tangents
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

            let mut indices = Vec::with_capacity(ai_mesh.mNumFaces as usize * 3);

            for face_index in 0..ai_mesh.mNumFaces as usize {
                let face = ai_mesh.mFaces.add(face_index).read();
                let count = face.mNumIndices as usize;

                for j in 0..count {
                    indices.push(*face.mIndices.add(j) as u32);
                }
            }

            assert!(indices.len() >= vertices.iter().len());

            meshes.push(MeshMeta {
                name,
                indices,
                vertices,
                material_index,
            });
        }
    }
    meshes
}

fn process_node(
    ai_node: *const aiNode,
    mapped_meshes: &HashMap<u32, u32>,
    parent: Option<Rc<RefCell<Node>>>,
) -> Rc<RefCell<Node>> {
    unsafe {
        let ai_matrix = (*ai_node).mTransformation;

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

        let name = {
            let ai_string = &(*ai_node).mName;
            let c_str = CStr::from_bytes_with_nul_unchecked(std::slice::from_raw_parts(
                ai_string.data.as_ptr() as *const u8,
                ai_string.length as usize + 1,
            ));
            c_str.to_string_lossy().into_owned()
        };

        let mesh_count = (*ai_node).mNumMeshes as usize;
        let meshes: Vec<u32> = (0..mesh_count)
            .map(|i| {
                let mesh_index = *(*ai_node).mMeshes.add(i) as u32;
                *mapped_meshes
                    .get(&mesh_index)
                    .expect("Fatal: Mesh index not found")
            })
            .collect();

        let node = Rc::new(RefCell::new(Node {
            parent: if parent.is_some() {
                Some(Rc::downgrade(&parent.clone().unwrap()))
            } else {
                None
            },
            children: Vec::new(),
            meshes,
            world_transform: if let Some(parent) = parent {
                parent.borrow().world_transform.mul_mat4(&local_transform)
            } else {
                local_transform
            },
            local_transform,
            dirty: true,
        }));

        // Process children
        for i in 0..(*ai_node).mNumChildren {
            let child_ai_node = *(*ai_node).mChildren.add(i as usize);
            let child_node = process_node(child_ai_node, mapped_meshes, Some(node.clone()));
            node.borrow_mut().children.push(child_node);
        }
        node
    }
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

struct MeshTemp {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
}

const AI_MATKEY_BASE_COLOR: *const c_char = b"$clr.base\0".as_ptr() as *const c_char;
const AI_MATKEY_COLOR_DIFFUSE: *const c_char = b"$clr.diffuse\0".as_ptr() as *const c_char;
const AI_MATKEY_COLOR_SPECULAR: *const c_char = b"$clr.specular\0".as_ptr() as *const c_char;
const AI_MATKEY_COLOR_AMBIENT: *const c_char = b"$clr.ambient\0".as_ptr() as *const c_char;
const AI_MATKEY_COLOR_EMISSIVE: *const c_char = b"$clr.emissive\0".as_ptr() as *const c_char;
const AI_MATKEY_ROUGHNESS_FACTOR: *const c_char =
    b"$mat.roughnessFactor\0".as_ptr() as *const c_char;
const AI_MATKEY_METALLIC_FACTOR: *const c_char = b"$mat.metallicFactor\0".as_ptr() as *const c_char;
const AI_MATKEY_BUMPSCALING: *const c_char = b"$mat.bumpscaling\0".as_ptr() as *const c_char;
const AI_MATKEY_OPACITY: *const c_char = b"$mat.opacity\0".as_ptr() as *const c_char;
const AI_MATKEY_TEXMAP_STRENGTH_AMBIENT_OCCLUSION: *const c_char =
    b"$mat.occlusionTexture.strength\0".as_ptr() as *const c_char;
const AI_MATKEY_EMISSIVE_INTENSITY: *const c_char =
    b"$mat.emissiveIntensity\0".as_ptr() as *const c_char;
const AI_MATKEY_GLTF_ALPHAMODE: *const c_char = b"$mat.gltf.alphaMode\0".as_ptr() as *const c_char;
const AI_MATKEY_GLTF_ALPHACUTOFF: *const c_char =
    b"$mat.gltf.alphaCutoff\0".as_ptr() as *const c_char;
