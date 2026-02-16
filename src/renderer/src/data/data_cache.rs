//! # Asset Caching System (Textures, Materials, Meshes)
//!
//! ## Purpose
//! Manages loading, GPU upload, and lifetime of textures, materials, and meshes. Implements
//! lazy loading with Unloaded/Loaded state machines. Provides default resources (white texture,
//! error pink texture, default materials).
//!
//! ## Key Concepts
//! - **TextureCache**: Texture and material storage with lazy loading
//! - **MeshCache**: Mesh geometry storage (vertices, indices, sub-allocated from VkSubAllocator)
//! - **Loading States**: Unloaded (CPU data) → Loaded (GPU resources)
//! - **Default Resources**: Pre-loaded fallback textures/materials (indices 0-5)
//! - **Descriptor Management**: Dynamic allocation for material descriptors
//!
//! ## Architecture
//! ```
//! TextureCache
//!   ├─ cached_textures: Vec<CachedTexture>     // Indexed by texture ID
//!   │    └─ Unloaded(TextureMeta) | Loaded(VkLoadedTexture)
//!   ├─ cached_materials: Vec<CachedMaterial>   // Indexed by material ID
//!   │    └─ Unloaded(MaterialMeta) | Loaded(VkLoadedMaterial)
//!   ├─ material_meta_storage: VkSubAllocator   // SSBO for material parameters
//!   └─ desc_manager: DescriptorManager         // Allocates texture samplers descriptors
//!
//! MeshCache
//!   ├─ cached_meshes: Vec<CachedMesh>
//!   │    └─ Unloaded(MeshMeta) | Loaded(VkMeshBuffers)
//!   ├─ index_storage: VkSubAllocator           // Shared index buffer
//!   └─ vertex_storage: VkSubAllocator          // Shared vertex buffer
//! ```
//!
//! ## Default Resources (TextureCache)
//! - **Index 0**: 1x1 white (default base color)
//! - **Index 1**: 1x1 white (neutral metallic/roughness sample)
//! - **Index 2**: 1x1 [128,128,255] (default normal map, points up)
//! - **Index 3**: 1x1 white (default occlusion)
//! - **Index 4**: 1x1 black (default emissive)
//! - **Index 5**: 2x2 pink (error texture)
//!
//! ## Loading Flow
//! 1. Add unloaded resource: cache.add_texture(TextureMeta) → returns ID
//! 2. Background thread or lazy load: cache.load_texture(ID)
//!    a. Allocate GPU image (vk_util::create_image_from_data)
//!    b. Upload via VkHostBuffer async transfer
//!    c. Transition CachedTexture::Unloaded → Loaded
//! 3. Rendering: cache.get_loaded_texture(ID) → VkLoadedTexture
//!
//! ## Material Metadata Storage
//! Materials store parameters (base_color_factor, metallic_factor, etc.) in GPU SSBO.
//! VkSubAllocator packs multiple materials into large buffer. Material descriptor set
//! points to slice via offset+range.
//!
//! ## Why Caches
//! - De-duplication: Same texture used by multiple materials → loaded once
//! - Stable IDs: u32 indices survive cache resize (Vec never shrinks)
//! - Lazy loading: Only upload textures for visible objects
//! - Batch transfers: Multiple textures uploaded in single frame

use crate::data::data_util::PackUnorm;
use crate::data::environment_import::{self, EnvironmentSource, PendingSkyboxSource};
use crate::data::gpu_data::{
    AlphaMode, AsByteSlice, EmissiveMap, EnvironmentUBO, MaterialMeta, MaterialShadingModel,
    MaterialValues, MeshMeta, MetRoughUniform, MetRoughUniformExt, NormalMap, OcclusionMap,
    Sampler, SurfaceMeta, TextureIds, TextureMeta, TextureSamplers, Vertex, VkCubeMap,
    VkMeshBuffers,
};
use crate::data::handles::{
    CacheError, EnvironmentHandle, MaterialHandle, MeshHandle, TextureHandle,
};
use crate::data::{assimp_util, data_util, gpu_data};
use crate::vulkan::vk_descriptor::{
    PoolSizeRatio, VkDescriptorAllocator, VkDescriptorWriter, VkDynamicDescriptorAllocator,
};
use crate::vulkan::vk_storage::{BufferPlacement, VkAllocResult, VkSubAllocator};
use crate::vulkan::vk_types::{
    VkBuffer, VkBufferAndDescriptorLimits, VkCmdSubmitInfo, VkCommandPool, VkDestroyable,
    VkDeviceQueues, VkFenceQueue, VkHostBuffer, VkImageAlloc, VkImmediate, VkPipeline, VkQueueType,
    VkSubAlloc, VkSubmitParam,
};

use crate::vulkan::{vk_debug, vk_util};
use ash::prelude::VkResult;
use ash::vk::{Format, PFN_vkFreeDescriptorSets};
use ash::{vk, Device};
use glam::{vec3, vec4, Vec3, Vec4};
use gltf::json::Path;
use image::{
    EncodableLayout, GenericImageView, ImageBuffer, ImageResult, Rgb32FImage, Rgba, Rgba32FImage,
};
use log::{debug, error, info};
use once_cell::unsync::Lazy;
use std::collections::{HashMap, HashSet};
use std::hash::{DefaultHasher, Hasher};
use std::marker::PhantomData;
use std::path;
use std::rc::Rc;
use std::sync::mpsc::{Receiver, SendError};
use std::sync::{Arc, LazyLock, Mutex};
use std::time::{Duration, Instant, SystemTime};
use vk_mem::Allocator;

///////////////////
// TEXTURE CACHE //
///////////////////

pub enum LoadResult<T> {
    Success(Option<Vec<T>>),
    Failed(Option<Vec<T>>),
}

pub struct PendingTextureBatch {
    pub batch_id: u64,
    pub texture_ids: Vec<TextureHandle>,
    pub image_allocs: Vec<(VkImageAlloc, vk::Sampler)>,
    pub submitted_at: Instant,
    pub status: UploadBatchStatus,
}

pub enum UploadBatchStatus {
    WaitingFence,
    Completed,
    Failed(String),
}

/// Texture loading state: CPU data or GPU resource.
///
/// ## Purpose
/// Lazy loading pattern. Textures start as Unloaded (TextureMeta with CPU bytes),
/// transition to Loaded (VkLoadedTexture with GPU image) when needed.
///
/// ## State Transitions
/// - Unloaded → Loaded: load_texture() uploads to GPU
/// - Loaded → Unloaded: Never (resources stay loaded until cache destroyed)
/// - _NULL: Placeholder (unused, legacy)
#[derive(Debug)]
pub enum CachedTexture {
    Unloaded(TextureMeta),
    Loaded(VkLoadedTexture),
    _NULL,
}

/// Material loading state: CPU metadata or GPU resources.
///
/// ## Purpose
/// Materials reference textures (by ID) and store parameters (base_color_factor, etc.).
/// Unloaded state holds CPU data, Loaded state has GPU descriptor set + SSBO allocation.
///
/// ## VkLoadedMaterial
/// - texture_ids: Indices into TextureCache
/// - meta_alloc: VkSubAlloc into material_meta_storage SSBO
/// - image_descriptor: Descriptor set binding 5 texture samplers
/// - pipeline: Which pipeline to use (Opaque/Transparent)
#[derive(Debug)]
pub enum CachedMaterial {
    Unloaded(MaterialMeta),
    Loaded(VkLoadedMaterial),
    _NULL,
}

#[derive(Debug, Clone, Copy)]
pub struct VkLoadedMaterial {
    pub texture_ids: TextureIds,
    pub meta_alloc: VkSubAlloc,
    pub image_descriptor: vk::DescriptorSet,
    pub pipeline: VkPipelineType,
    pub alpha_mode: AlphaMode,
    pub requires_uv1: bool,
}

#[derive(Debug)]
pub struct VkLoadedTexture {
    pub alloc: VkImageAlloc,
    pub sampler: vk::Sampler,
}

pub struct DescriptorManager {
    image_desc_allocator: VkDynamicDescriptorAllocator,
    image_desc_layout: vk::DescriptorSetLayout,
}

impl DescriptorManager {
    pub fn alloc_image_desc(&mut self, device: &ash::Device) -> vk::DescriptorSet {
        self.image_desc_allocator
            .allocate(device, &[self.image_desc_layout])
            .unwrap()
    }
}

unsafe impl Send for MeshCache {}

unsafe impl Send for TextureCache {}

pub struct VkDataCache {
    pub mesh_cache: Mutex<MeshCache>,
    pub texture_cache: Mutex<TextureCache>,
    pub environment_cache: Mutex<EnvironmentCache>,
    pub supported_image_formats: HashSet<vk::Format>,
}

impl VkDataCache {
    pub fn is_supported_image_format(&self, format: vk::Format) -> bool {
        self.supported_image_formats.contains(&format)
    }
}

pub struct VkCache {
    pub shaders: VkShaderCache,
    pub desc_layouts: VkDescLayoutCache,
    pub pipelines: VkPipelineCache,
    pub queues: VkDeviceQueues,
}

impl VkDestroyable for VkCache {
    fn destroy(&mut self, device: &Device, allocator: &Allocator) {
        self.shaders.destroy(device, allocator);
        self.desc_layouts.destroy(device, allocator);
        self.pipelines.destroy(device, allocator);
    }
}

impl VkDataCache {
    pub fn destroy(&self, device: &Device, allocator: &Allocator) {
        self.mesh_cache.lock().unwrap().destroy(device, allocator);
        self.texture_cache
            .lock()
            .unwrap()
            .destroy(device, allocator);
    }
}

pub struct TextureCache {
    instance: ash::Instance,
    physical_device: vk::PhysicalDevice,
    device: ash::Device,
    allocator: Arc<Mutex<Allocator>>,
    cached_textures: Vec<CachedTexture>,
    cached_materials: Vec<CachedMaterial>,
    texture_generations: Vec<u32>,
    material_generations: Vec<u32>,
    free_texture_slots: Vec<u32>,
    free_material_slots: Vec<u32>,
    desc_manager: DescriptorManager,
    supported_formats: HashSet<vk::Format>,
    sampler_cache: VkSamplerCache,
    material_meta_storage: VkSubAllocator,
    host_buffer: Arc<Mutex<VkHostBuffer>>,
    host_alignment: u64,
    gfx_pool: VkCommandPool,
    gfx_queue: vk::Queue,
    linear_blit_support: Mutex<HashMap<vk::Format, bool>>,
    pending_batches: HashMap<u64, PendingTextureBatch>,
    pending_textures: HashMap<TextureHandle, u64>,
    next_batch_id: u64,
}

impl TextureCache {
    pub const DEFAULT_ERROR_TEX: TextureHandle = TextureHandle::new(5, 0);
    pub const DEFAULT_COLOR_TEX: TextureHandle = TextureHandle::new(0, 0);
    pub const DEFAULT_ROUGH_TEX: TextureHandle = TextureHandle::new(1, 0);
    pub const DEFAULT_NORMAL_TEX: TextureHandle = TextureHandle::new(2, 0);
    pub const DEFAULT_OCCLUSION_TEX: TextureHandle = TextureHandle::new(3, 0);
    pub const DEFAULT_EMISSIVE_TEX: TextureHandle = TextureHandle::new(4, 0);
    pub const DEFAULT_TEX_ITER_START: usize = 6;

    pub const DEFAULT_BASE_COLOR_FACTOR: Vec4 = vec4(1.0, 1.0, 1.0, 1.0);
    pub const DEFAULT_METALLIC_FACTOR: f32 = 0.0;
    pub const DEFAULT_ROUGHNESS_FACTOR: f32 = 1.0;
    pub const DEFAULT_NORMAL_SCALE: f32 = 1.0;
    pub const DEFAULT_OCCLUSION_STRENGTH: f32 = 1.0;
    pub const DEFAULT_EMISSIVE_STRENGTH: f32 = 1.0;
    pub const DEFAULT_EMISSIVE_FACTOR: Vec3 = Vec3::ZERO;

    pub const DEFAULT_MAT_ROUGH_MAT: MaterialHandle = MaterialHandle::new(0, 0);
    pub const DEFAULT_ERROR_MAT: MaterialHandle = MaterialHandle::new(1, 0);
    pub const DEFAULT_MAT_ITER_START: usize = 2;

    pub const DEFAULT_NORMAL_MAP: NormalMap = NormalMap {
        scale: Self::DEFAULT_NORMAL_SCALE,
        texture_id: Self::DEFAULT_NORMAL_TEX,
    };

    pub const DEFAULT_OCCLUSION_MAP: OcclusionMap = OcclusionMap {
        strength: Self::DEFAULT_OCCLUSION_STRENGTH,
        texture_id: Self::DEFAULT_OCCLUSION_TEX,
    };

    pub const DEFAULT_EMISSIVE_MAP: EmissiveMap = EmissiveMap {
        factor: Self::DEFAULT_EMISSIVE_FACTOR,
        texture_id: Self::DEFAULT_EMISSIVE_TEX,
    };

    pub fn new(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        device: &ash::Device,
        allocator: Arc<Mutex<Allocator>>,
        sampler_cache: VkSamplerCache,
        supported_formats: HashSet<vk::Format>,
        meta_desc_layout: vk::DescriptorSetLayout,
        image_desc_layout: vk::DescriptorSetLayout,
        host_buffer: Arc<Mutex<VkHostBuffer>>,
        meta_buffer_size: u64,
        limits: &VkBufferAndDescriptorLimits,
        gfx_pool: VkCommandPool,
        gfx_queue: vk::Queue,
    ) -> Result<Self, String> {
        let def_color = CachedTexture::Unloaded(TextureMeta {
            payload: gpu_data::TexturePayload::Raw {
                bytes: vec![255, 255, 255, 255],
                width: 1,
                height: 1,
                format: vk::Format::R8G8B8A8_UNORM,
                mips_levels: 1,
            },
            uv_index: 0,
            sampler_info: None,
        });

        let def_metallic_rough = CachedTexture::Unloaded(TextureMeta {
            // Sampled as .g (roughness) and .b (metallic) in the shader.
            // Keep these channels valid even in fallback texture.
            payload: gpu_data::TexturePayload::Raw {
                bytes: vec![255, 255, 255, 255],
                width: 1,
                height: 1,
                format: vk::Format::R8G8B8A8_UNORM,
                mips_levels: 1,
            },
            uv_index: 0,
            sampler_info: None,
        });

        let r8_support = supported_formats.contains(&vk::Format::R8_UNORM);

        let def_occlusion = CachedTexture::Unloaded(TextureMeta {
            payload: gpu_data::TexturePayload::Raw {
                bytes: if r8_support {
                    vec![255]
                } else {
                    vec![255, 255, 255, 255]
                },
                width: 1,
                height: 1,
                format: if r8_support {
                    vk::Format::R8_UNORM
                } else {
                    vk::Format::R8G8B8A8_UNORM
                },
                mips_levels: 1,
            },
            uv_index: 0,
            sampler_info: None,
        });

        let def_normal = CachedTexture::Unloaded(TextureMeta {
            payload: gpu_data::TexturePayload::Raw {
                bytes: vec![128, 128, 255, 255],
                width: 1,
                height: 1,
                format: vk::Format::R8G8B8A8_UNORM,
                mips_levels: 1,
            },
            uv_index: 0,
            sampler_info: None,
        });

        let def_emissive = CachedTexture::Unloaded(TextureMeta {
            payload: gpu_data::TexturePayload::Raw {
                bytes: vec![0, 0, 0, 255],
                width: 1,
                height: 1,
                format: vk::Format::R8G8B8A8_UNORM,
                mips_levels: 1,
            },
            uv_index: 0,
            sampler_info: None,
        });

        let def_error = CachedTexture::Unloaded(TextureMeta {
            payload: gpu_data::TexturePayload::Raw {
                bytes: vec![255, 20, 147, 255],
                width: 2,
                height: 2,
                format: vk::Format::R8G8B8A8_UNORM,
                mips_levels: 1,
            },
            uv_index: 0,
            sampler_info: None,
        });

        let err_mat = CachedMaterial::Unloaded(MaterialMeta {
            texture_ids: TextureIds {
                base_color: Self::DEFAULT_ERROR_TEX,
                ..Default::default()
            },
            alpha_mode: AlphaMode::Opaque,
            shading_model: MaterialShadingModel::PbrMetalRough,
            material_values: Default::default(),
        });

        let mut cached_textures = Vec::with_capacity(100);
        cached_textures.push(def_color);
        cached_textures.push(def_metallic_rough);
        cached_textures.push(def_normal);
        cached_textures.push(def_occlusion);
        cached_textures.push(def_emissive);
        cached_textures.push(def_error);

        let mut cached_materials = Vec::with_capacity(100);
        cached_materials.push(CachedMaterial::Unloaded(MaterialMeta::default()));
        cached_materials.push(err_mat);

        let image_desc_ratios = [PoolSizeRatio::new(
            vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            5.0,
        )];
        let meta_desc_ratios = [PoolSizeRatio::new(vk::DescriptorType::STORAGE_BUFFER, 1.0)];

        let image_desc_allocator =
            VkDynamicDescriptorAllocator::new(&device, 5_000, &image_desc_ratios).unwrap();
        let meta_desc_allocator =
            VkDynamicDescriptorAllocator::new(&device, 1_000, &meta_desc_ratios).unwrap();

        let material_meta_storage = VkSubAllocator::new_storage_buffer(
            &device,
            allocator.clone(),
            host_buffer.clone(),
            meta_buffer_size,
            limits.optimal_buffer_copy_offset_alignment,
            vk::BufferUsageFlags::empty(),
        )?;

        let desc_manager = DescriptorManager {
            image_desc_allocator,
            image_desc_layout,
        };

        Ok(Self {
            instance: instance.clone(),
            physical_device,
            device: device.clone(),
            allocator,
            texture_generations: vec![0; cached_textures.len()],
            material_generations: vec![0; cached_materials.len()],
            free_texture_slots: Vec::new(),
            free_material_slots: Vec::new(),
            cached_textures,
            cached_materials,
            supported_formats,
            desc_manager,
            material_meta_storage,
            host_buffer,
            sampler_cache,
            host_alignment: std::cmp::max(limits.optimal_buffer_copy_offset_alignment, 4),
            gfx_pool,
            gfx_queue,
            linear_blit_support: Mutex::new(HashMap::new()),
            pending_batches: HashMap::new(),
            pending_textures: HashMap::new(),
            next_batch_id: 1,
        })
    }

    pub fn is_supported_format(&self, format: vk::Format) -> bool {
        self.supported_formats.contains(&format)
    }

    fn supports_linear_mip_blit(&self, format: vk::Format) -> bool {
        let mut cache = self.linear_blit_support.lock().unwrap();
        if let Some(supported) = cache.get(&format) {
            return *supported;
        }

        let supported =
            vk_util::format_supports_linear_mip_blit(&self.instance, self.physical_device, format);
        cache.insert(format, supported);
        supported
    }

    fn texture_handle_for_slot(&self, slot: u32) -> TextureHandle {
        TextureHandle::new(slot, self.texture_generations[slot as usize])
    }

    fn material_handle_for_slot(&self, slot: u32) -> MaterialHandle {
        MaterialHandle::new(slot, self.material_generations[slot as usize])
    }

    fn validate_texture_slot(&self, handle: TextureHandle) -> Result<usize, CacheError> {
        let slot = handle.slot as usize;
        let Some(generation) = self.texture_generations.get(slot) else {
            return Err(CacheError::OutOfBounds);
        };
        if *generation != handle.generation {
            return Err(CacheError::StaleHandle);
        }
        Ok(slot)
    }

    fn validate_material_slot(&self, handle: MaterialHandle) -> Result<usize, CacheError> {
        let slot = handle.slot as usize;
        let Some(generation) = self.material_generations.get(slot) else {
            return Err(CacheError::OutOfBounds);
        };
        if *generation != handle.generation {
            return Err(CacheError::StaleHandle);
        }
        Ok(slot)
    }

    fn alloc_texture_slot(&mut self, data: CachedTexture) -> TextureHandle {
        if let Some(slot) = self.free_texture_slots.pop() {
            self.cached_textures[slot as usize] = data;
            self.texture_handle_for_slot(slot)
        } else {
            let slot = self.cached_textures.len() as u32;
            self.cached_textures.push(data);
            self.texture_generations.push(0);
            TextureHandle::new(slot, 0)
        }
    }

    fn alloc_material_slot(&mut self, data: CachedMaterial) -> MaterialHandle {
        if let Some(slot) = self.free_material_slots.pop() {
            self.cached_materials[slot as usize] = data;
            self.material_handle_for_slot(slot)
        } else {
            let slot = self.cached_materials.len() as u32;
            self.cached_materials.push(data);
            self.material_generations.push(0);
            MaterialHandle::new(slot, 0)
        }
    }

    pub fn add_texture(&mut self, mut data: TextureMeta) -> TextureHandle {
        if !self.supported_formats.contains(&data.payload.format()) {
            info!(
                "Unsupported Format: {:?}, converting to R8G8B8A8_UNORM",
                data.payload.format()
            );

            if let gpu_data::TexturePayload::Raw {
                bytes,
                width,
                height,
                format,
                mips_levels,
            } = &data.payload
            {
                let converted =
                    ImageBuffer::<image::Rgb<u8>, _>::from_raw(*width, *height, bytes.clone());

                if let Some(image) = converted {
                    let new_bytes = image::DynamicImage::ImageRgb8(image).to_rgba8();
                    data.payload = gpu_data::TexturePayload::Raw {
                        bytes: new_bytes.to_vec(),
                        width: *width,
                        height: *height,
                        format: vk::Format::R8G8B8A8_UNORM,
                        mips_levels: *mips_levels,
                    };
                } else {
                    log::info!(
                        "Error converting material of type: {:?} to RGBA. Using error texture.",
                        format
                    );
                    return Self::DEFAULT_ERROR_TEX;
                }
            } else {
                log::error!(
                    "Cannot convert unsupported compressed format {:?} to RGBA on the fly.",
                    data.payload.format()
                );
                return Self::DEFAULT_ERROR_TEX;
            }
        }

        self.alloc_texture_slot(CachedTexture::Unloaded(data))
    }

    pub(crate) fn save_debug_image(data: &TextureMeta, bytes: &[u8], filename: String) {
        let path = path::Path::new("debug_textures").join(filename);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();

        match data.payload.format() {
            vk::Format::R8G8B8A8_UNORM | vk::Format::R8G8B8A8_SRGB => {
                if let Some(img) = ImageBuffer::<Rgba<u8>, _>::from_raw(
                    data.payload.width(),
                    data.payload.height(),
                    bytes,
                ) {
                    img.save(&path).unwrap();
                }
            }
            vk::Format::R8G8B8_UNORM | vk::Format::R8G8B8_SRGB => {
                if let Some(img) = ImageBuffer::<image::Rgb<u8>, _>::from_raw(
                    data.payload.width(),
                    data.payload.height(),
                    bytes,
                ) {
                    img.save(&path).unwrap();
                }
            }
            _ => {
                println!(
                    "Unsupported format for debug save: {:?}",
                    data.payload.format()
                );
            }
        }

        println!("Saved debug image: {:?}", path);
    }

    pub fn add_textures(&mut self, data: Vec<TextureMeta>) -> Vec<TextureHandle> {
        data.into_iter()
            .map(|meta| self.add_texture(meta))
            .collect()
    }

    pub fn add_material(&mut self, data: MaterialMeta) -> MaterialHandle {
        self.alloc_material_slot(CachedMaterial::Unloaded(data))
    }

    pub fn add_materials(&mut self, data: Vec<MaterialMeta>) -> Vec<MaterialHandle> {
        data.into_iter()
            .map(|meta| self.add_material(meta))
            .collect()
    }

    pub fn set_unloaded_material_shading_model(
        &mut self,
        material_ids: &[MaterialHandle],
        shading_model: MaterialShadingModel,
    ) -> Result<(), String> {
        for id in material_ids.iter().copied() {
            let slot_idx = self.validate_material_slot(id).map_err(|err| {
                format!(
                    "Invalid material handle in debug override {:?}: {:?}",
                    id, err
                )
            })?;

            match self.cached_materials.get_mut(slot_idx) {
                Some(CachedMaterial::Unloaded(meta)) => {
                    meta.shading_model = shading_model;
                }
                Some(CachedMaterial::Loaded(_)) => {
                    return Err(format!(
                        "Material {:?} already loaded before debug shading override",
                        id
                    ));
                }
                Some(CachedMaterial::_NULL) | None => {
                    return Err(format!(
                        "Material {:?} is a tombstone and cannot be overridden",
                        id
                    ));
                }
            }
        }

        Ok(())
    }

    pub fn get_material(&self, id: MaterialHandle) -> Result<&CachedMaterial, CacheError> {
        let slot = self.validate_material_slot(id)?;
        self.cached_materials
            .get(slot)
            .ok_or(CacheError::OutOfBounds)
    }

    pub fn get_loaded_material(&self, id: MaterialHandle) -> Result<VkLoadedMaterial, CacheError> {
        let slot = self.validate_material_slot(id)?;
        match self.cached_materials.get(slot) {
            Some(CachedMaterial::Loaded(loaded)) => Ok(*loaded),
            Some(CachedMaterial::Unloaded(_)) => Err(CacheError::NotLoaded),
            Some(CachedMaterial::_NULL) => Err(CacheError::InvalidHandle),
            None => Err(CacheError::OutOfBounds),
        }
    }

    pub fn get_loaded_material_ptr(
        &self,
        id: MaterialHandle,
    ) -> Result<*const VkLoadedMaterial, CacheError> {
        let slot = self.validate_material_slot(id)?;
        match self.cached_materials.get(slot) {
            Some(CachedMaterial::Loaded(loaded)) => Ok(loaded as *const VkLoadedMaterial),
            Some(CachedMaterial::Unloaded(_)) => Err(CacheError::NotLoaded),
            Some(CachedMaterial::_NULL) => Err(CacheError::InvalidHandle),
            None => Err(CacheError::OutOfBounds),
        }
    }

    pub fn get_texture(&self, id: TextureHandle) -> Result<&CachedTexture, CacheError> {
        let slot = self.validate_texture_slot(id)?;
        self.cached_textures
            .get(slot)
            .ok_or(CacheError::OutOfBounds)
    }

    pub fn get_loaded_texture(&self, id: TextureHandle) -> Result<&VkLoadedTexture, CacheError> {
        let slot = self.validate_texture_slot(id)?;
        match self.cached_textures.get(slot) {
            Some(CachedTexture::Loaded(loaded)) => Ok(loaded),
            Some(CachedTexture::Unloaded(_)) => Err(CacheError::NotLoaded),
            Some(CachedTexture::_NULL) => Err(CacheError::InvalidHandle),
            None => Err(CacheError::OutOfBounds),
        }
    }

    pub fn is_texture_loaded(&self, id: TextureHandle) -> bool {
        let Ok(slot) = self.validate_texture_slot(id) else {
            return false;
        };
        if let Some(found) = self.cached_textures.get(slot) {
            matches!(found, CachedTexture::Loaded(_))
        } else {
            false
        }
    }

    fn destroy_uploaded_images(&self, image_allocs: Vec<(VkImageAlloc, vk::Sampler)>) {
        if image_allocs.is_empty() {
            return;
        }

        let allocator = self.allocator.lock().unwrap();
        for (image_alloc, _) in image_allocs.into_iter() {
            vk_util::destroy_image(&allocator, image_alloc);
        }
    }

    fn promote_uploaded_images(
        &mut self,
        texture_ids: &[TextureHandle],
        image_allocs: Vec<(VkImageAlloc, vk::Sampler)>,
    ) {
        if texture_ids.len() != image_allocs.len() {
            error!(
                "Texture upload finalize mismatch: {} texture ids, {} image allocs",
                texture_ids.len(),
                image_allocs.len()
            );
            for id in texture_ids.iter() {
                self.pending_textures.remove(id);
            }
            self.destroy_uploaded_images(image_allocs);
            return;
        }

        let mut stale_images = Vec::<(VkImageAlloc, vk::Sampler)>::new();
        for (id, image) in texture_ids.iter().zip(image_allocs.into_iter()) {
            self.pending_textures.remove(id);
            let Ok(slot) = self.validate_texture_slot(*id) else {
                error!("Stale texture handle {:?} during upload finalization", id);
                stale_images.push(image);
                continue;
            };

            self.cached_textures[slot] = CachedTexture::Loaded(VkLoadedTexture {
                alloc: image.0,
                sampler: image.1,
            });
        }

        self.destroy_uploaded_images(stale_images);
    }

    /// Synchronous texture upload: submits GPU transfers and blocks until complete.
    /// This is the backward-compatible wrapper used by startup and sync loading paths.
    pub fn allocate_textures(&mut self, texture_ids: Vec<TextureHandle>) -> bool {
        loop {
            let all_loaded = texture_ids.iter().all(|id| self.is_texture_loaded(*id));
            if all_loaded {
                return true;
            }

            if let Err(msg) = self.submit_texture_uploads(&texture_ids) {
                error!("allocate_textures failed: {}", msg);
                return false;
            }

            let finalized = self.poll_texture_uploads();
            if finalized == 0 {
                std::thread::sleep(Duration::from_millis(1));
            }
        }
    }

    /// Submit texture data to the GPU without blocking for completion.
    ///
    /// Returns `Ok(Some(batch_id))` if a batch was submitted and is now pending,
    /// `Ok(None)` if there were no unloaded textures to process, or `Err` on failure.
    pub fn submit_texture_uploads(
        &mut self,
        texture_ids: &[TextureHandle],
    ) -> Result<Option<u64>, String> {
        let host_buffer = self.host_buffer.lock().unwrap();
        let max_upload_bytes = host_buffer.buffer.size;

        // A staging upload is already in flight; poll/finalize first, then submit again.
        if host_buffer.countdown_latch.get_count() != 0 || !self.pending_batches.is_empty() {
            return Ok(None);
        }

        // Filter for Unloaded textures only while still validating all handles.
        let mut upload_ids = Vec::<TextureHandle>::with_capacity(texture_ids.len());
        for id in texture_ids.iter().copied() {
            let slot = self
                .validate_texture_slot(id)
                .map_err(|err| format!("invalid texture handle {:?}: {:?}", id, err))?;
            if matches!(
                self.cached_textures.get(slot),
                Some(CachedTexture::Unloaded(_))
            ) {
                upload_ids.push(id);
            }
        }

        if upload_ids.is_empty() {
            return Ok(None);
        }

        let mut curr_bytes = 0usize;
        let mut next_upload = Vec::<&TextureMeta>::with_capacity(upload_ids.len());
        let mut next_upload_blit_support = Vec::<bool>::with_capacity(upload_ids.len());
        let mut batch_texture_ids = Vec::<TextureHandle>::new();
        let mut ids = Vec::<u32>::new();

        for id in upload_ids.iter().copied() {
            let Ok(slot) = self.validate_texture_slot(id) else {
                return Err(format!("invalid texture handle {:?}", id));
            };

            match self.cached_textures.get(slot) {
                Some(CachedTexture::Unloaded(meta)) => {
                    let aligned_size = meta
                        .payload
                        .bytes()
                        .len()
                        .next_multiple_of(self.host_alignment as usize);

                    if aligned_size > max_upload_bytes as usize {
                        return Err(format!(
                            "texture {:?} requires {} bytes but staging buffer holds {} bytes",
                            id, aligned_size, max_upload_bytes
                        ));
                    }

                    // Submit one non-blocking batch per call; larger workloads are chunked
                    // by repeated submit/poll cycles via allocate_textures().
                    if curr_bytes + aligned_size > max_upload_bytes as usize {
                        break;
                    }

                    curr_bytes += aligned_size;
                    next_upload.push(meta);
                    next_upload_blit_support
                        .push(self.supports_linear_mip_blit(meta.payload.format()));
                    ids.push(id.slot);
                    batch_texture_ids.push(id);
                }
                _ => {
                    return Err(format!("texture {:?} not in Unloaded state", id));
                }
            }
        }

        if curr_bytes == 0 {
            return Ok(None);
        }

        let image_allocs = match vk_util::record_host_to_image_buffer(
            &self.device,
            &self.allocator.lock().unwrap(),
            &mut self.sampler_cache,
            &host_buffer,
            &next_upload,
            &next_upload_blit_support,
            self.host_alignment,
            &ids,
            self.gfx_queue,
        ) {
            Ok(images) => images,
            Err(err) => {
                return Err(format!("texture upload record failed: {:?}", err));
            }
        };

        debug!("Submitting texture upload batch (non-blocking)");
        if let Err(err) = host_buffer.submit_transfer_commands(VkSubmitParam::signaling(
            // Transfer submit contains staging copies, so signal once transfer-domain
            // commands are complete and ownership can move to graphics.
            vk_util::async_transfer_signal_stage_mask(),
        )) {
            host_buffer.reset_buffers(&self.device);
            self.destroy_uploaded_images(image_allocs);
            return Err(format!(
                "failed to submit transfer commands for texture upload batch: {}",
                err
            ));
        }

        if let Err(err) = host_buffer.submit_graphics_commands(VkSubmitParam::waiting(
            // Texture upload graphics work starts in transfer domain (ownership acquire +
            // mip blits), so waiting at TRANSFER is the earliest correct synchronization point.
            vk_util::async_texture_upload_wait_stage_mask(),
        )) {
            let err_msg = format!(
                "failed to submit graphics commands for texture upload batch: {}",
                err
            );
            error!("{}", err_msg);

            // Transfer submission may already be in flight; defer image cleanup through
            // normal pending-batch finalization once the latch reaches zero.
            let batch_id = self.next_batch_id;
            self.next_batch_id = self.next_batch_id.wrapping_add(1).max(1);
            for id in batch_texture_ids.iter() {
                self.pending_textures.insert(*id, batch_id);
            }
            self.pending_batches.insert(
                batch_id,
                PendingTextureBatch {
                    batch_id,
                    texture_ids: batch_texture_ids,
                    image_allocs,
                    submitted_at: Instant::now(),
                    status: UploadBatchStatus::Failed(err_msg.clone()),
                },
            );
            return Err(err_msg);
        }

        if host_buffer.countdown_latch.get_count() == 0 {
            host_buffer.reset_buffers(&self.device);
            drop(host_buffer);
            self.promote_uploaded_images(batch_texture_ids.as_slice(), image_allocs);
            return Ok(None);
        }

        // Store as pending batch for later poll_texture_uploads() finalization
        let batch_id = self.next_batch_id;
        self.next_batch_id = self.next_batch_id.wrapping_add(1).max(1);

        for id in batch_texture_ids.iter() {
            self.pending_textures.insert(*id, batch_id);
        }

        self.pending_batches.insert(
            batch_id,
            PendingTextureBatch {
                batch_id,
                texture_ids: batch_texture_ids,
                image_allocs,
                submitted_at: Instant::now(),
                status: UploadBatchStatus::WaitingFence,
            },
        );

        Ok(Some(batch_id))
    }

    /// Poll pending texture upload batches for completion.
    ///
    /// For each completed batch, promotes textures from Unloaded → Loaded and
    /// resets the staging buffer for reuse. Returns the number of finalized batches.
    pub fn poll_texture_uploads(&mut self) -> usize {
        if self.pending_batches.is_empty() {
            return 0;
        }

        let host_buffer = self.host_buffer.lock().unwrap();
        let latch_count = host_buffer.countdown_latch.get_count();

        if latch_count != 0 {
            // Check for timeout (30s safety net)
            let now = Instant::now();
            let timed_out: Vec<u64> = self
                .pending_batches
                .iter()
                .filter(|(_, batch)| {
                    matches!(batch.status, UploadBatchStatus::WaitingFence)
                        && now.duration_since(batch.submitted_at) > Duration::from_secs(30)
                })
                .map(|(id, _)| *id)
                .collect();

            for batch_id in timed_out {
                if let Some(batch) = self.pending_batches.get_mut(&batch_id) {
                    error!(
                        "Texture upload batch {} timed out after 30s",
                        batch.batch_id
                    );
                    batch.status =
                        UploadBatchStatus::Failed("upload timed out after 30s".to_string());
                }
            }

            return 0;
        }

        // All fences signaled — reset staging buffer and promote all pending batches
        host_buffer.reset_buffers(&self.device);
        drop(host_buffer);

        let batch_ids: Vec<u64> = self.pending_batches.keys().copied().collect();
        let mut finalized = 0usize;

        for batch_id in batch_ids {
            let Some(batch) = self.pending_batches.remove(&batch_id) else {
                continue;
            };

            match batch.status {
                UploadBatchStatus::WaitingFence => {
                    self.promote_uploaded_images(batch.texture_ids.as_slice(), batch.image_allocs);
                    finalized += 1;
                }
                UploadBatchStatus::Failed(ref msg) => {
                    error!("Dropping failed batch {}: {}", batch_id, msg);
                    for id in batch.texture_ids.iter() {
                        self.pending_textures.remove(id);
                    }
                    self.destroy_uploaded_images(batch.image_allocs);
                    finalized += 1;
                }
                UploadBatchStatus::Completed => {
                    for id in batch.texture_ids.iter() {
                        self.pending_textures.remove(id);
                    }
                    self.destroy_uploaded_images(batch.image_allocs);
                    finalized += 1;
                }
            }
        }

        finalized
    }

    fn allocate_materials(
        &mut self,
        material_ids: Vec<MaterialHandle>,
        buffer_placement: BufferPlacement,
        rtn_alloc: bool,
    ) -> LoadResult<VkLoadedMaterial> {
        let mut materials =
            Vec::<(MaterialHandle, MaterialMeta)>::with_capacity(material_ids.len());
        for id in material_ids {
            let Ok(slot) = self.validate_material_slot(id) else {
                error!("Failed to locate unloaded material id: {:?}", id);
                return LoadResult::Failed(None);
            };
            let Some(CachedMaterial::Unloaded(meta)) = self.cached_materials.get(slot) else {
                error!("Failed to locate unloaded material id: {:?}", id);
                return LoadResult::Failed(None);
            };

            materials.push((id, *meta));
        }

        let mut texture_ids: Vec<TextureHandle> = materials
            .iter()
            .flat_map(|(_, meta)| meta.texture_ids.to_vec())
            .collect();

        // Dedupe
        texture_ids.sort_unstable();
        texture_ids.dedup();

        if !self.allocate_textures(texture_ids) {
            return LoadResult::Failed(None);
        }

        let meta_bytes: Vec<&[u8]> = materials
            .iter()
            .map(|(_, material)| bytemuck::bytes_of(&material.material_values))
            .collect();

        let meta_allocs = match self
            .material_meta_storage
            .allocate_bytes(&meta_bytes, buffer_placement)
        {
            VkAllocResult::Success(allocs) => allocs,
            VkAllocResult::Failure {
                error_msg,
                successful_allocs,
            } => {
                error!("Error allocating material meta: {:?}", error_msg);
                successful_allocs
                    .into_iter()
                    .for_each(|alloc| self.material_meta_storage.deallocate(alloc));
                return LoadResult::Failed(None);
            }
        };

        let mut loaded_materials = if rtn_alloc {
            Some(Vec::<VkLoadedMaterial>::with_capacity(materials.len()))
        } else {
            None
        };

        for ((id, meta), alloc) in materials.into_iter().zip(meta_allocs.into_iter()) {
            let loaded_mat = match self.write_material_descriptors(&meta, alloc) {
                Ok(mat) => mat,
                Err(err) => {
                    error!(
                        "Failed to write material descriptors for {:?}: {:?}",
                        id, err
                    );
                    return LoadResult::Failed(None);
                }
            };
            let Ok(slot) = self.validate_material_slot(id) else {
                error!("Failed to update loaded material slot for {:?}", id);
                return LoadResult::Failed(None);
            };
            self.cached_materials[slot] = CachedMaterial::Loaded(loaded_mat);

            if let Some(rtn_vec) = &mut loaded_materials {
                rtn_vec.push(loaded_mat)
            }
        }
        LoadResult::Success(loaded_materials)
    }

    fn write_material_descriptors(
        &mut self,
        meta: &MaterialMeta,
        meta_alloc: VkSubAlloc,
    ) -> Result<VkLoadedMaterial, CacheError> {
        let pipeline = Self::pipeline_for_material(meta);

        let color_tex = self.get_loaded_texture(meta.texture_ids.base_color)?;
        let metallic_tex = self.get_loaded_texture(meta.texture_ids.metallic_roughness)?;
        let normal_tex = self.get_loaded_texture(meta.texture_ids.normal_map)?;
        let occlusion_tex = self.get_loaded_texture(meta.texture_ids.occlusion_map)?;
        let emissive_tex = self.get_loaded_texture(meta.texture_ids.emissive_map)?;

        debug!(" color id: {:?}", meta.texture_ids.base_color);
        debug!(" metal rough id: {:?}", meta.texture_ids.metallic_roughness);
        debug!(" normal id: {:?}", meta.texture_ids.normal_map);
        debug!(" occlusion id: {:?}", meta.texture_ids.occlusion_map);
        debug!(" emissive id: {:?}", meta.texture_ids.emissive_map);
        debug!(
            " base color uv set: {}",
            meta.material_values.base_color_uv_set
        );
        debug!(
            " metal rough uv set: {}",
            meta.material_values.met_rough_uv_set
        );
        debug!(" normal uv set: {}", meta.material_values.normal_uv_set);
        debug!(
            " occlusion uv set: {}",
            meta.material_values.occlusion_uv_set
        );
        debug!(" emissive uv set: {}", meta.material_values.emissive_uv_set);
        debug!(" metallic factor: {}", meta.material_values.metallic_factor);
        debug!(
            " roughness factor: {}",
            meta.material_values.roughness_factor
        );
        debug!(" normal scale: {}", meta.material_values.normal_scale);
        debug!(
            " occlusion strength: {}",
            meta.material_values.occlusion_strength
        );

        let mut writer = VkDescriptorWriter::default();
        writer.write_image(
            0,
            color_tex.alloc.image_view,
            color_tex.sampler,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        );

        writer.write_image(
            1,
            metallic_tex.alloc.image_view,
            metallic_tex.sampler,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        );

        writer.write_image(
            2,
            normal_tex.alloc.image_view,
            normal_tex.sampler,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        );

        writer.write_image(
            3,
            occlusion_tex.alloc.image_view,
            occlusion_tex.sampler,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        );

        writer.write_image(
            4,
            emissive_tex.alloc.image_view,
            emissive_tex.sampler,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        );

        let image_descriptor = self.desc_manager.alloc_image_desc(&self.device);

        writer.update_set(&self.device, image_descriptor);

        let requires_uv1 = meta.material_values.base_color_uv_set == 1
            || meta.material_values.met_rough_uv_set == 1
            || meta.material_values.normal_uv_set == 1
            || meta.material_values.occlusion_uv_set == 1
            || meta.material_values.emissive_uv_set == 1;

        let mat = VkLoadedMaterial {
            texture_ids: meta.texture_ids,
            meta_alloc,
            image_descriptor,
            pipeline,
            alpha_mode: meta.alpha_mode,
            requires_uv1,
        };
        debug!("Bound material to descriptorset: {:#?}", mat);
        Ok(mat)
    }

    fn pipeline_for_material(meta: &MaterialMeta) -> VkPipelineType {
        match (meta.shading_model, meta.alpha_mode) {
            (MaterialShadingModel::PbrMetalRough, AlphaMode::Blend) => {
                VkPipelineType::PbrMetRoughAlpha
            }
            (MaterialShadingModel::PbrMetalRough, AlphaMode::Opaque | AlphaMode::Mask) => {
                VkPipelineType::PbrMetRoughOpaque
            }
            (MaterialShadingModel::Unlit, AlphaMode::Blend) => VkPipelineType::UnlitAlpha,
            (MaterialShadingModel::Unlit, AlphaMode::Opaque | AlphaMode::Mask) => {
                VkPipelineType::UnlitOpaque
            }
        }
    }

    pub fn allocate_all(
        &mut self,
        buffer_placement: BufferPlacement,
        rtn_alloc: bool,
    ) -> LoadResult<VkLoadedMaterial> {
        let id_mats: Vec<MaterialHandle> = self
            .cached_materials
            .iter()
            .enumerate()
            .filter_map(|(id, mat)| {
                if let CachedMaterial::Unloaded(_) = mat {
                    Some(self.material_handle_for_slot(id as u32))
                } else {
                    None
                }
            })
            .collect();

        self.allocate_materials(id_mats, buffer_placement, rtn_alloc)
    }

    pub fn allocate_ids(
        &mut self,
        material_ids: &[MaterialHandle],
        buffer_placement: BufferPlacement,
        rtn_alloc: bool,
    ) -> LoadResult<VkLoadedMaterial> {
        let mut existing_loads = Vec::<VkLoadedMaterial>::new();
        let mut id_mats = Vec::<MaterialHandle>::with_capacity(material_ids.len());
        for id in material_ids.iter() {
            let Ok(slot) = self.validate_material_slot(*id) else {
                error!("Failed to locate material handle: {:?}", id);
                return LoadResult::Failed(None);
            };

            match self.cached_materials.get(slot) {
                Some(CachedMaterial::Unloaded(_)) => id_mats.push(*id),
                Some(CachedMaterial::Loaded(loaded)) if rtn_alloc => existing_loads.push(*loaded),
                _ => {
                    error!("Failed to locate material id: {:?}", id);
                    return LoadResult::Failed(None);
                }
            }
        }

        let mut alloc_result = self.allocate_materials(id_mats, buffer_placement, rtn_alloc);
        if !existing_loads.is_empty() {
            match alloc_result {
                LoadResult::Success(Some(allocs)) => {
                    existing_loads.extend(allocs);
                    alloc_result = LoadResult::Success(Some(existing_loads));
                }
                LoadResult::Failed(Some(allocs)) => {
                    existing_loads.extend(allocs);
                    alloc_result = LoadResult::Failed(Some(existing_loads));
                }
                LoadResult::Success(None) => {
                    alloc_result = LoadResult::Success(Some(existing_loads))
                }
                LoadResult::Failed(None) => alloc_result = LoadResult::Failed(Some(existing_loads)),
            }
        }

        alloc_result
    }

    pub fn allocate_id(
        &mut self,
        id: MaterialHandle,
        buffer_placement: BufferPlacement,
        rtn_alloc: bool,
    ) -> LoadResult<VkLoadedMaterial> {
        let Ok(slot) = self.validate_material_slot(id) else {
            return LoadResult::Failed(None);
        };
        match self.cached_materials.get(slot) {
            Some(CachedMaterial::Loaded(loaded)) => {
                if rtn_alloc {
                    LoadResult::Success(Some(vec![*loaded]))
                } else {
                    LoadResult::Success(None)
                }
            }
            Some(CachedMaterial::Unloaded(_)) => {
                self.allocate_materials(vec![id], buffer_placement, rtn_alloc)
            }
            _ => LoadResult::Failed(None),
        }
    }

    fn deallocate_textures_with_policy(
        &mut self,
        texture_ids: Vec<TextureHandle>,
        preserve_reserved: bool,
    ) {
        for id in texture_ids.into_iter() {
            if preserve_reserved && (id.slot as usize) < Self::DEFAULT_TEX_ITER_START {
                continue;
            }

            let Ok(slot_idx) = self.validate_texture_slot(id) else {
                continue;
            };

            if let Some(slot) = self.cached_textures.get_mut(slot_idx) {
                let old_tex = std::mem::replace(slot, CachedTexture::_NULL);
                if let CachedTexture::Loaded(tex) = old_tex {
                    let allocator = self.allocator.lock().unwrap();
                    vk_util::destroy_image(&allocator, tex.alloc)
                }
                self.texture_generations[slot_idx] =
                    self.texture_generations[slot_idx].wrapping_add(1);
                self.free_texture_slots.push(slot_idx as u32);
            }
        }
    }

    fn deallocate_textures(&mut self, texture_ids: Vec<TextureHandle>) {
        self.deallocate_textures_with_policy(texture_ids, true);
    }

    pub fn deallocate_texture(&mut self, texture_id: TextureHandle) {
        self.deallocate_textures(vec![texture_id]);
    }

    pub fn deallocate_textures_safe(&mut self, texture_ids: Vec<TextureHandle>) {
        self.deallocate_textures(texture_ids);
    }

    /// Unsafe because this can destroy reserved/default texture slots relied on by engine code.
    pub unsafe fn deallocate_textures_unchecked(&mut self, texture_ids: Vec<TextureHandle>) {
        self.deallocate_textures_with_policy(texture_ids, false);
    }

    fn deallocate_materials_with_policy(
        &mut self,
        material_ids: Vec<MaterialHandle>,
        preserve_reserved: bool,
    ) {
        let mut texture_ids = Vec::<TextureHandle>::with_capacity(material_ids.len() * 5);

        for id in material_ids {
            if preserve_reserved && (id.slot as usize) < Self::DEFAULT_MAT_ITER_START {
                continue;
            }

            let Ok(slot_idx) = self.validate_material_slot(id) else {
                continue;
            };

            if let Some(slot) = self.cached_materials.get_mut(slot_idx) {
                let old_mat = std::mem::replace(slot, CachedMaterial::_NULL);
                if let CachedMaterial::Loaded(mat) = old_mat {
                    texture_ids.extend(mat.texture_ids.to_vec());
                    self.material_meta_storage.deallocate(mat.meta_alloc)
                }
                self.material_generations[slot_idx] =
                    self.material_generations[slot_idx].wrapping_add(1);
                self.free_material_slots.push(slot_idx as u32);
            }
        }

        self.deallocate_textures_with_policy(texture_ids, preserve_reserved);
    }

    pub fn deallocate_materials(&mut self, material_ids: Vec<MaterialHandle>) {
        self.deallocate_materials_with_policy(material_ids, true);
    }

    /// Unsafe because this can destroy reserved/default material slots relied on by engine code.
    pub unsafe fn deallocate_materials_unchecked(&mut self, material_ids: Vec<MaterialHandle>) {
        self.deallocate_materials_with_policy(material_ids, false);
    }
}

impl VkDestroyable for TextureCache {
    fn destroy(&mut self, device: &Device, allocator: &Allocator) {
        let pending_images: Vec<(VkImageAlloc, vk::Sampler)> = self
            .pending_batches
            .drain()
            .flat_map(|(_, batch)| batch.image_allocs.into_iter())
            .collect();
        self.pending_textures.clear();
        self.destroy_uploaded_images(pending_images);

        for slot in self.cached_textures.iter_mut() {
            let old_tex = std::mem::replace(slot, CachedTexture::_NULL);
            if let CachedTexture::Loaded(tex) = old_tex {
                vk_util::destroy_image(allocator, tex.alloc);
            }
        }

        for slot in self.cached_materials.iter_mut() {
            let old_mat = std::mem::replace(slot, CachedMaterial::_NULL);
            if let CachedMaterial::Loaded(mat) = old_mat {
                self.material_meta_storage.deallocate(mat.meta_alloc);
            }
        }

        self.material_meta_storage.destroy(device, allocator);
        self.desc_manager
            .image_desc_allocator
            .destroy(device, allocator);
        self.sampler_cache.destroy(device);
    }
}

////////////////
// MESH CACHE //
////////////////

#[derive(Debug)]
pub enum CachedMesh {
    Unloaded(MeshMeta),
    Loaded(VkMeshBuffers),
    _NULL,
}

pub struct MeshCache {
    vertex_storage: VkSubAllocator,
    index_storage: VkSubAllocator,
    cached_meshes: Vec<CachedMesh>,
    mesh_generations: Vec<u32>,
    free_mesh_slots: Vec<u32>,
    joint_desc_pool: VkDynamicDescriptorAllocator,
    default_joint_desc: vk::DescriptorSet,
    default_joint_buffer: VkBuffer,
}

impl MeshCache {
    const DEFAULT_JOINTS: [glam::Mat4; 128] = [glam::Mat4::IDENTITY; 128];
    const DEFAULT_MESH_ITER_START: usize = 1;
    pub const SKYBOX_MESH: MeshHandle = MeshHandle::new(0, 0);

    pub fn new(
        device: &ash::Device,
        allocator: &Allocator,
        joint_desc_layout: vk::DescriptorSetLayout,
        vertex_storage: VkSubAllocator,
        index_storage: VkSubAllocator,
    ) -> Self {
        use glam::{Vec3, Vec4};

        let mut cached_meshes = Vec::<CachedMesh>::with_capacity(100);

        let (vertices, indices) = data_util::get_skybox_mesh();
        let skybox = MeshMeta {
            name: "Skybox Cube".to_string(),
            indices,
            vertices,
            material_index: None,
            has_uv1: false,
        };

        cached_meshes.push(CachedMesh::Unloaded(skybox));

        let default_joint_buffer = vk_util::allocate_and_write_buffer(
            allocator,
            Self::DEFAULT_JOINTS.as_byte_slice(),
            vk::BufferUsageFlags::UNIFORM_BUFFER,
        )
        .unwrap();

        let mut joint_desc_pool = VkDynamicDescriptorAllocator::new(
            device,
            1,
            &[PoolSizeRatio::new(vk::DescriptorType::UNIFORM_BUFFER, 1.0)],
        )
        .unwrap();

        let default_joint_desc = joint_desc_pool
            .allocate(device, &[joint_desc_layout])
            .unwrap();

        let mut writer = VkDescriptorWriter::default();
        writer.write_buffer(
            0,
            default_joint_buffer.buffer,
            (std::mem::size_of::<glam::Mat4>() * 128) as u64,
            0,
            vk::DescriptorType::UNIFORM_BUFFER,
        );

        writer.update_set(device, default_joint_desc);

        Self {
            cached_meshes,
            mesh_generations: vec![0],
            free_mesh_slots: Vec::new(),
            vertex_storage,
            index_storage,
            joint_desc_pool,
            default_joint_buffer,
            default_joint_desc,
        }
    }

    pub fn get_default_joint_desc(&self) -> vk::DescriptorSet {
        self.default_joint_desc
    }

    fn mesh_handle_for_slot(&self, slot: u32) -> MeshHandle {
        MeshHandle::new(slot, self.mesh_generations[slot as usize])
    }

    fn validate_mesh_slot(&self, handle: MeshHandle) -> Result<usize, CacheError> {
        let slot = handle.slot as usize;
        let Some(generation) = self.mesh_generations.get(slot) else {
            return Err(CacheError::OutOfBounds);
        };
        if *generation != handle.generation {
            return Err(CacheError::StaleHandle);
        }
        Ok(slot)
    }

    pub fn add(&mut self, data: MeshMeta) -> MeshHandle {
        if let Some(slot) = self.free_mesh_slots.pop() {
            self.cached_meshes[slot as usize] = CachedMesh::Unloaded(data);
            self.mesh_handle_for_slot(slot)
        } else {
            let slot = self.cached_meshes.len() as u32;
            self.cached_meshes.push(CachedMesh::Unloaded(data));
            self.mesh_generations.push(0);
            MeshHandle::new(slot, 0)
        }
    }

    pub fn add_multi(&mut self, data: Vec<MeshMeta>) -> Vec<MeshHandle> {
        data.into_iter().map(|mesh| self.add(mesh)).collect()
    }

    pub fn add_and_allocate(
        &mut self,
        data: MeshMeta,
        buffer_placement: BufferPlacement,
        return_buffers: bool,
    ) -> LoadResult<VkMeshBuffers> {
        let id = self.add(data);
        self.allocate_id(id, buffer_placement, return_buffers)
    }

    pub fn add_and_allocate_multi(
        &mut self,
        data: Vec<MeshMeta>,
        buffer_placement: BufferPlacement,
        return_buffers: bool,
    ) -> LoadResult<VkMeshBuffers> {
        let ids = self.add_multi(data);
        self.allocate_ids(&ids, buffer_placement, return_buffers)
    }

    pub fn get_id(&self, id: MeshHandle) -> Result<&CachedMesh, CacheError> {
        let slot = self.validate_mesh_slot(id)?;
        self.cached_meshes.get(slot).ok_or(CacheError::OutOfBounds)
    }

    pub fn get_loaded_id(&self, id: MeshHandle) -> Result<VkMeshBuffers, CacheError> {
        let slot = self.validate_mesh_slot(id)?;
        match self.cached_meshes.get(slot) {
            Some(CachedMesh::Loaded(buffers)) => Ok(*buffers),
            Some(CachedMesh::Unloaded(_)) => Err(CacheError::NotLoaded),
            Some(CachedMesh::_NULL) => Err(CacheError::InvalidHandle),
            None => Err(CacheError::OutOfBounds),
        }
    }

    pub fn get_ids(&self, ids: Vec<MeshHandle>) -> Vec<Result<&CachedMesh, CacheError>> {
        ids.into_iter().map(|id| self.get_id(id)).collect()
    }

    pub fn is_id_loaded(&self, id: MeshHandle) -> bool {
        let Ok(slot) = self.validate_mesh_slot(id) else {
            return false;
        };
        if let Some(found) = self.cached_meshes.get(slot) {
            matches!(found, CachedMesh::Loaded(_))
        } else {
            false
        }
    }

    unsafe fn allocate(
        &mut self,
        meshes: Vec<(MeshHandle, *const MeshMeta)>,
        buffer_placement: BufferPlacement,
        return_buffers: bool,
    ) -> LoadResult<VkMeshBuffers> {
        let mut vertex_data = Vec::<&[u8]>::with_capacity(meshes.len());
        let mut index_data = Vec::<&[u8]>::with_capacity(meshes.len());

        for (_, mesh_ptr) in &meshes {
            unsafe {
                let mesh = &**mesh_ptr;
                vertex_data.push(bytemuck::cast_slice(&mesh.vertices));
                index_data.push(bytemuck::cast_slice(&mesh.indices));
            }
        }

        let vertex_allocs = match self
            .vertex_storage
            .allocate_bytes(&mut vertex_data, buffer_placement)
        {
            VkAllocResult::Success(allocs) => allocs,
            VkAllocResult::Failure {
                error_msg,
                successful_allocs,
            } => {
                successful_allocs
                    .into_iter()
                    .for_each(|alloc| self.vertex_storage.deallocate(alloc));
                error!("Failed to allocate vertices: {:?}", error_msg);
                return LoadResult::Failed(None);
            }
        };

        let index_allocs = match self
            .index_storage
            .allocate_bytes(&mut index_data, buffer_placement)
        {
            VkAllocResult::Success(allocs) => allocs,
            VkAllocResult::Failure {
                error_msg,
                successful_allocs,
            } => {
                successful_allocs
                    .into_iter()
                    .for_each(|alloc| self.vertex_storage.deallocate(alloc));
                error!("Failed to allocate vertices: {:?}", error_msg);
                return LoadResult::Failed(None);
            }
        };

        let mut rtn_buffers = if return_buffers {
            Some(Vec::<VkMeshBuffers>::with_capacity(vertex_allocs.len()))
        } else {
            None
        };

        meshes
            .iter()
            .map(|(id, meta)| *id)
            .zip(vertex_allocs.into_iter())
            .zip(index_allocs.into_iter())
            .for_each(|((id, vert_alloc), index_alloc)| {
                if let CachedMesh::Unloaded(meta) =
                    unsafe { self.cached_meshes.get_unchecked(id.slot as usize) }
                {
                    let material_id = meta
                        .material_index
                        .unwrap_or(TextureCache::DEFAULT_MAT_ROUGH_MAT);
                    let buffer = VkMeshBuffers {
                        cache_id: id,
                        index_count: meta.indices.len() as u32,
                        vertex_count: meta.vertices.len() as u32,
                        index_buffer: index_alloc.clone(),
                        vertex_buffer: vert_alloc.clone(),
                        joint_desc: self.default_joint_desc,
                        material_id,
                        has_uv1: meta.has_uv1,
                    };

                    debug!(
                        "Loaded mesh '{}' (cache handle {:?}) uses material handle {:?}",
                        meta.name, id, material_id
                    );

                    if let Some(rtn_meshes) = &mut rtn_buffers {
                        rtn_meshes.push(buffer)
                    }

                    self.cached_meshes[id.slot as usize] = CachedMesh::Loaded(buffer);
                } else {
                    panic!("Unreachable")
                }
            });

        debug!("Allocated Meshes: {:?}", meshes);
        LoadResult::Success(rtn_buffers)
    }

    pub fn allocate_all(
        &mut self,
        buffer_placement: BufferPlacement,
        return_buffers: bool,
    ) -> LoadResult<VkMeshBuffers> {
        let id_meshes: Vec<(MeshHandle, *const MeshMeta)> = self
            .cached_meshes
            .iter()
            .enumerate()
            .filter_map(|(i, mesh)| {
                if let CachedMesh::Unloaded(meta) = mesh {
                    Some((self.mesh_handle_for_slot(i as u32), meta as *const MeshMeta))
                } else {
                    None
                }
            })
            .collect();

        unsafe { self.allocate(id_meshes, buffer_placement, return_buffers) }
    }

    pub fn allocate_ids(
        &mut self,
        mesh_ids: &[MeshHandle],
        buffer_placement: BufferPlacement,
        rtn_buffers: bool,
    ) -> LoadResult<VkMeshBuffers> {
        let mut existing_loads = Vec::<VkMeshBuffers>::new();
        let mut id_meshes = Vec::<(MeshHandle, *const MeshMeta)>::with_capacity(mesh_ids.len());
        for id in mesh_ids.iter() {
            let Ok(slot) = self.validate_mesh_slot(*id) else {
                error!("Failed to locate mesh handle: {:?}", id);
                return LoadResult::Failed(None);
            };

            match self.cached_meshes.get(slot) {
                Some(CachedMesh::Unloaded(meta)) => id_meshes.push((*id, meta as *const MeshMeta)),
                Some(CachedMesh::Loaded(loaded)) if rtn_buffers => existing_loads.push(*loaded),
                _ => {
                    error!("Failed to located material id: {:?}", id);
                    return LoadResult::Failed(None);
                }
            }
        }

        let mut alloc_result = unsafe { self.allocate(id_meshes, buffer_placement, rtn_buffers) };
        if !existing_loads.is_empty() {
            match alloc_result {
                LoadResult::Success(Some(allocs)) => {
                    existing_loads.extend(allocs);
                    alloc_result = LoadResult::Success(Some(existing_loads));
                }
                LoadResult::Failed(Some(allocs)) => {
                    existing_loads.extend(allocs);
                    alloc_result = LoadResult::Failed(Some(existing_loads));
                }
                LoadResult::Success(None) => {
                    alloc_result = LoadResult::Success(Some(existing_loads))
                }
                LoadResult::Failed(None) => alloc_result = LoadResult::Failed(Some(existing_loads)),
            }
        }

        alloc_result
    }

    pub fn allocate_id(
        &mut self,
        mesh_id: MeshHandle,
        buffer_placement: BufferPlacement,
        return_buffers: bool,
    ) -> LoadResult<VkMeshBuffers> {
        let Ok(slot) = self.validate_mesh_slot(mesh_id) else {
            return LoadResult::Failed(None);
        };

        if let Some(CachedMesh::Unloaded(meta)) = self.cached_meshes.get(slot) {
            unsafe {
                self.allocate(
                    vec![(mesh_id, meta as *const MeshMeta)],
                    buffer_placement,
                    return_buffers,
                )
            }
        } else {
            LoadResult::Failed(None)
        }
    }

    fn deallocate_id_with_policy(&mut self, mesh_id: MeshHandle, preserve_reserved: bool) {
        if preserve_reserved && (mesh_id.slot as usize) < Self::DEFAULT_MESH_ITER_START {
            return;
        }

        let Ok(slot_idx) = self.validate_mesh_slot(mesh_id) else {
            return;
        };

        if let Some(slot) = self.cached_meshes.get_mut(slot_idx) {
            let old_mesh = std::mem::replace(slot, CachedMesh::_NULL);
            if let CachedMesh::Loaded(loaded_mesh) = old_mesh {
                self.index_storage.deallocate(loaded_mesh.index_buffer);
                self.vertex_storage.deallocate(loaded_mesh.vertex_buffer);
            }
            self.mesh_generations[slot_idx] = self.mesh_generations[slot_idx].wrapping_add(1);
            self.free_mesh_slots.push(slot_idx as u32);
        }
    }

    pub fn deallocate_id(&mut self, mesh_id: MeshHandle) {
        self.deallocate_id_with_policy(mesh_id, true);
    }

    /// Unsafe because this can destroy reserved/default mesh slots relied on by engine code.
    pub unsafe fn deallocate_id_unchecked(&mut self, mesh_id: MeshHandle) {
        self.deallocate_id_with_policy(mesh_id, false);
    }

    pub fn deallocate_ids(&mut self, mesh_ids: &[MeshHandle]) {
        mesh_ids.iter().for_each(|&id| self.deallocate_id(id))
    }

    pub fn deallocate_all(&mut self, allocator: &vk_mem::Allocator) {
        (Self::DEFAULT_MESH_ITER_START..self.cached_meshes.len())
            .for_each(|i| self.deallocate_id(self.mesh_handle_for_slot(i as u32)))
    }
}

impl VkDestroyable for MeshCache {
    fn destroy(&mut self, device: &Device, allocator: &Allocator) {
        self.cached_meshes.clear();
        self.index_storage.destroy(device, allocator);
        self.vertex_storage.destroy(device, allocator)
    }
}

//////////////////
// SHADER CACHE //
//////////////////

#[repr(C)]
#[derive(Ord, Eq, PartialEq, PartialOrd, Debug, Clone, Copy, Hash)]
pub enum CoreShaderType {
    MetRoughVert,
    MetRoughFrag,
    MetRoughFragUnlit,
    BrtFlutVert,
    BrtFlutFrag,
    SkyBoxVert,
    SkyBoxFrag,
    CubeFilterVert,
    EnvIrradianceFrag,
    EnvPrefilterFrag,
    EnvEquirectToCubeFrag,
}

impl CoreShaderType {
    const COUNT: usize = 11;

    fn from_manifest_key(key: &str) -> Option<Self> {
        match key {
            "MetRoughVert" => Some(Self::MetRoughVert),
            "MetRoughFrag" => Some(Self::MetRoughFrag),
            "MetRoughFragUnlit" => Some(Self::MetRoughFragUnlit),
            "BrtFlutVert" => Some(Self::BrtFlutVert),
            "BrtFlutFrag" => Some(Self::BrtFlutFrag),
            "SkyBoxVert" => Some(Self::SkyBoxVert),
            "SkyBoxFrag" => Some(Self::SkyBoxFrag),
            "CubeFilterVert" => Some(Self::CubeFilterVert),
            "EnvIrradianceFrag" => Some(Self::EnvIrradianceFrag),
            "EnvPrefilterFrag" => Some(Self::EnvPrefilterFrag),
            "EnvEquirectToCubeFrag" => Some(Self::EnvEquirectToCubeFrag),
            _ => None,
        }
    }
}

const CORE_SHADER_MANIFEST: &str = include_str!("../shaders/core_shader_manifest.txt");

pub fn load_core_shader_manifest() -> Result<Vec<(CoreShaderType, &'static str)>, String> {
    let mut shader_paths =
        Vec::<(CoreShaderType, &'static str)>::with_capacity(CoreShaderType::COUNT);
    let mut seen = std::collections::HashSet::with_capacity(CoreShaderType::COUNT);

    for (line_index, line) in CORE_SHADER_MANIFEST.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        let Some((key, path)) = trimmed.split_once('=') else {
            return Err(format!(
                "Invalid shader manifest entry at line {}: '{}'",
                line_index + 1,
                line
            ));
        };

        let shader_type = CoreShaderType::from_manifest_key(key.trim()).ok_or_else(|| {
            format!(
                "Unknown shader key '{}' in manifest at line {}",
                key.trim(),
                line_index + 1
            )
        })?;

        if !seen.insert(shader_type) {
            return Err(format!(
                "Duplicate shader key '{}' in manifest at line {}",
                key.trim(),
                line_index + 1
            ));
        }

        let path = path.trim();
        if path.is_empty() {
            return Err(format!(
                "Empty shader path for key '{}' at line {}",
                key.trim(),
                line_index + 1
            ));
        }

        shader_paths.push((shader_type, path));
    }

    if shader_paths.len() != CoreShaderType::COUNT {
        return Err(format!(
            "Shader manifest size mismatch: expected {}, found {}",
            CoreShaderType::COUNT,
            shader_paths.len()
        ));
    }

    Ok(shader_paths)
}

pub struct VkShaderCache {
    pub core_shader_cache: [vk::ShaderModule; CoreShaderType::COUNT],
    pub user_shader_cache: Vec<vk::ShaderModule>,
}

impl VkShaderCache {
    pub fn new(
        device: &ash::Device,
        shader_paths: Vec<(CoreShaderType, &str)>,
    ) -> Result<Self, String> {
        let mut compiled_shaders = shader_paths
            .iter()
            .map(|(typ, path)| {
                vk_util::load_shader_module(&device, path).map(|shader| (*typ, shader))
            })
            .collect::<Result<Vec<(CoreShaderType, vk::ShaderModule)>, String>>()?;

        compiled_shaders.sort_by_key(|(typ, path)| *typ);

        let sorted_shaders: [vk::ShaderModule; CoreShaderType::COUNT] = compiled_shaders
            .into_iter()
            .map(|(_, shader)| shader)
            .collect::<Vec<_>>()
            .try_into()
            .map_err(|_| "Number of shaders did not match number of enum keys")?;

        Ok(Self {
            core_shader_cache: sorted_shaders,
            user_shader_cache: Vec::new(),
        })
    }

    pub fn get_core_shader(&self, typ: CoreShaderType) -> vk::ShaderModule {
        self.core_shader_cache[typ as usize]
    }

    pub fn destory_all(&mut self, device: &ash::Device) {}
}

impl VkDestroyable for VkShaderCache {
    fn destroy(&mut self, device: &Device, allocator: &Allocator) {
        self.core_shader_cache
            .iter()
            .for_each(|shader| unsafe { device.destroy_shader_module(*shader, None) });

        self.user_shader_cache
            .iter()
            .for_each(|shader| unsafe { device.destroy_shader_module(*shader, None) });
    }
}

///////////////////////
// VK PIPELINE CACHE //
///////////////////////

#[repr(u8)]
#[derive(Ord, Eq, PartialEq, PartialOrd, Debug, Clone, Copy, Hash)]
pub enum VkPipelineType {
    PbrMetRoughOpaque,
    PbrMetRoughAlpha,
    UnlitOpaque,
    UnlitAlpha,
    BrdfLut,
    Skybox,
    EnvPreFilter,
    EnvIrradiance,
    EnvEquirectToCube,
}

impl VkPipelineType {
    pub const COUNT: usize = 9;
}

//#[derive(Clone, Copy)]
pub struct VkPipelineCache {
    pipelines: [VkPipeline; VkPipelineType::COUNT],
}

impl VkPipelineCache {
    pub fn new(mut pipelines: Vec<(VkPipelineType, VkPipeline)>) -> Result<Self, String> {
        pipelines.sort_by_key(|(typ, _)| *typ);

        let sorted_pipelines: [VkPipeline; VkPipelineType::COUNT] = pipelines
            .into_iter()
            .map(|(_, pipeline)| pipeline)
            .collect::<Vec<_>>()
            .try_into()
            .map_err(|_| "Number of pipelines did not match number of enum keys".to_string())?;

        Ok(Self {
            pipelines: sorted_pipelines,
        })
    }

    pub fn get_pipeline(&self, typ: VkPipelineType) -> &VkPipeline {
        unsafe { self.pipelines.get_unchecked(typ as usize) }
    }
}

impl VkDestroyable for VkPipelineCache {
    fn destroy(&mut self, device: &Device, allocator: &Allocator) {
        self.pipelines
            .iter_mut()
            .for_each(|pipe| pipe.destroy(device, allocator));
    }
}

/////////////////////////////
// Descriptor Layout Cache //
/////////////////////////////

#[repr(u8)]
#[derive(Ord, Eq, PartialEq, PartialOrd, Debug, Clone, Copy)]
pub enum VkDescType {
    DrawImage,
    SceneData,
    PbrSamplers,
    PbrProperties,
    SkinData,
    Skybox,
    EnvIrradiance,
    EnvPreFilter,
    EnvEquirect,
    Empty,
}

impl VkDescType {
    const COUNT: usize = 10;
}

pub struct VkDescLayoutCache {
    layouts: [vk::DescriptorSetLayout; VkDescType::COUNT],
}

impl VkDescLayoutCache {
    pub fn new(mut layouts: Vec<(VkDescType, vk::DescriptorSetLayout)>) -> Self {
        layouts.sort();

        let sorted_layouts: [vk::DescriptorSetLayout; VkDescType::COUNT] = layouts
            .into_iter()
            .map(|(_, layout)| layout)
            .collect::<Vec<_>>()
            .try_into()
            .expect("Number of descriptor layouts did not match number of enum keys");

        Self {
            layouts: sorted_layouts,
        }
    }

    pub fn get(&self, typ: VkDescType) -> vk::DescriptorSetLayout {
        self.layouts[typ as usize]
    }

    pub fn debug(&self) {
        debug!("Descriptor Set Layouts:");
        for (i, set) in self.layouts.iter().enumerate() {
            let typ = match i {
                0 => VkDescType::DrawImage,
                1 => VkDescType::SceneData,
                2 => VkDescType::PbrSamplers,
                3 => VkDescType::PbrProperties,
                4 => VkDescType::SkinData,
                5 => VkDescType::Skybox,
                6 => VkDescType::EnvIrradiance,
                7 => VkDescType::EnvPreFilter,
                8 => VkDescType::EnvEquirect,
                9 => VkDescType::Empty,
                _ => panic!(),
            };
            debug!("\t{:?} : {:?}", typ, *set)
        }
    }
}

impl VkDestroyable for VkDescLayoutCache {
    fn destroy(&mut self, device: &Device, allocator: &Allocator) {
        self.layouts.iter().for_each(|layout| unsafe {
            device.destroy_descriptor_set_layout(*layout, None);
        })
    }
}

pub enum CachedEnvironment {
    Unloaded(PendingSkyboxSource),
    Loaded(VkCubeMap),
}

pub struct EnvMaps {
    pub environment_ubo: EnvironmentUBO,
    pub irradiance: VkCubeMap,
    pub pre_filter: VkCubeMap,
}

pub struct EnvironmentCache {
    skyboxes: Vec<CachedEnvironment>,
    env_maps: Vec<Option<EnvMaps>>,
    env_generations: Vec<u32>,
    supported_formats: HashSet<vk::Format>,
}

impl EnvironmentCache {
    pub fn new(supported_formats: HashSet<vk::Format>) -> Self {
        Self {
            skyboxes: Vec::with_capacity(10),
            env_maps: Vec::with_capacity(10),
            env_generations: Vec::with_capacity(10),
            supported_formats,
        }
    }

    fn env_handle_for_slot(&self, slot: u32) -> EnvironmentHandle {
        EnvironmentHandle::new(slot, self.env_generations[slot as usize])
    }

    fn validate_env_slot(&self, handle: EnvironmentHandle) -> Result<usize, CacheError> {
        let slot = handle.slot as usize;
        let Some(generation) = self.env_generations.get(slot) else {
            return Err(CacheError::OutOfBounds);
        };
        if *generation != handle.generation {
            return Err(CacheError::StaleHandle);
        }
        Ok(slot)
    }

    pub fn get_skybox(&self, env_id: EnvironmentHandle) -> Result<&CachedEnvironment, CacheError> {
        let slot = self.validate_env_slot(env_id)?;
        self.skyboxes.get(slot).ok_or(CacheError::OutOfBounds)
    }

    pub fn import_environment(
        &mut self,
        source: EnvironmentSource,
    ) -> Result<EnvironmentHandle, String> {
        let pending =
            environment_import::import_environment_source(&source, &self.supported_formats)?;
        let index = self.skyboxes.len() as u32;

        info!("Imported environment source as Unloaded: {:?}", source);

        self.skyboxes.push(CachedEnvironment::Unloaded(pending));
        self.env_maps.push(None);
        self.env_generations.push(0);
        Ok(self.env_handle_for_slot(index))
    }

    pub fn add_env_maps(
        &mut self,
        env_id: EnvironmentHandle,
        env_maps: EnvMaps,
    ) -> Result<(), CacheError> {
        let slot = self.validate_env_slot(env_id)?;
        if let Some(map_slot) = self.env_maps.get_mut(slot) {
            *map_slot = Some(env_maps);
            Ok(())
        } else {
            Err(CacheError::OutOfBounds)
        }
    }

    pub fn get_env_map(&self, env_id: EnvironmentHandle) -> Result<&Option<EnvMaps>, CacheError> {
        let slot = self.validate_env_slot(env_id)?;
        self.env_maps.get(slot).ok_or(CacheError::OutOfBounds)
    }

    pub fn take_unloaded_source(
        &mut self,
        env_id: EnvironmentHandle,
    ) -> Result<Option<PendingSkyboxSource>, CacheError> {
        let slot = self.validate_env_slot(env_id)?;
        match self.skyboxes.get(slot) {
            Some(CachedEnvironment::Loaded(_)) => Ok(None),
            Some(CachedEnvironment::Unloaded(_)) => {
                let old = std::mem::replace(
                    &mut self.skyboxes[slot],
                    CachedEnvironment::Unloaded(PendingSkyboxSource::CubemapFaces {
                        face_size: 0,
                        format: vk::Format::UNDEFINED,
                        bytes: vec![],
                    }),
                );
                match old {
                    CachedEnvironment::Unloaded(source) => Ok(Some(source)),
                    CachedEnvironment::Loaded(_) => unreachable!(),
                }
            }
            None => Err(CacheError::OutOfBounds),
        }
    }

    pub fn restore_unloaded_source(
        &mut self,
        env_id: EnvironmentHandle,
        source: PendingSkyboxSource,
    ) -> Result<(), CacheError> {
        let slot = self.validate_env_slot(env_id)?;
        self.skyboxes[slot] = CachedEnvironment::Unloaded(source);
        Ok(())
    }

    pub fn store_loaded_cube_map(
        &mut self,
        env_id: EnvironmentHandle,
        cube_map: VkCubeMap,
    ) -> Result<(), CacheError> {
        let slot = self.validate_env_slot(env_id)?;
        self.skyboxes[slot] = CachedEnvironment::Loaded(cube_map);
        Ok(())
    }

    pub fn get_loaded_cube_map_handles(
        &self,
        env_id: EnvironmentHandle,
    ) -> Result<Option<(vk::ImageView, vk::Sampler)>, CacheError> {
        let slot = self.validate_env_slot(env_id)?;
        match self.skyboxes.get(slot) {
            Some(CachedEnvironment::Loaded(map)) => Ok(Some((map.image_view, map.sampler))),
            Some(CachedEnvironment::Unloaded(_)) => Ok(None),
            None => Err(CacheError::OutOfBounds),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LodBias {
    Sharp,
    Normal,
    Soft,
}

impl LodBias {
    fn to_float(&self) -> f32 {
        match self {
            LodBias::Sharp => -0.5,
            LodBias::Normal => 0.0,
            LodBias::Soft => 0.5,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct VkSamplerInfo {
    pub mag_filter: vk::Filter,
    pub min_filter: vk::Filter,
    pub mipmap_mode: vk::SamplerMipmapMode,
    pub address_mode_u: vk::SamplerAddressMode,
    pub address_mode_v: vk::SamplerAddressMode,
    pub address_mode_w: vk::SamplerAddressMode,
    pub mip_lod_bias: LodBias,
    pub anisotropy_enable: bool,
    pub max_anisotropy: u32,
    pub compare_enable: bool,
    pub compare_op: vk::CompareOp,
    pub min_lod: u32,
    pub max_lod: u32,
    pub border_color: vk::BorderColor,
    pub unnormalized_coordinates: bool,
}

impl VkSamplerInfo {
    pub fn to_create_info(&self) -> vk::SamplerCreateInfo<'_> {
        vk::SamplerCreateInfo::default()
            .mag_filter(self.mag_filter)
            .min_filter(self.min_filter)
            .mipmap_mode(self.mipmap_mode)
            .address_mode_u(self.address_mode_u)
            .address_mode_v(self.address_mode_v)
            .address_mode_w(self.address_mode_w)
            .mip_lod_bias(self.mip_lod_bias.to_float())
            .anisotropy_enable(self.anisotropy_enable)
            .max_anisotropy(self.max_anisotropy as f32)
            .compare_enable(self.compare_enable)
            .compare_op(self.compare_op)
            .min_lod(self.min_lod as f32)
            .max_lod(self.max_lod as f32)
            .border_color(self.border_color)
            .unnormalized_coordinates(self.unnormalized_coordinates)
    }
}

pub struct VkSamplerCache {
    pub samplers: HashMap<VkSamplerInfo, vk::Sampler>,
}

impl Default for VkSamplerCache {
    fn default() -> Self {
        Self {
            samplers: HashMap::with_capacity(20),
        }
    }
}

impl VkSamplerCache {
    pub fn get_or_create_sampler(
        &mut self,
        device: &ash::Device,
        info: VkSamplerInfo,
    ) -> vk::Sampler {
        if let Some(sampler) = self.samplers.get(&info) {
            *sampler
        } else {
            let create_info = info.to_create_info();
            let sampler = unsafe { device.create_sampler(&create_info, None).unwrap() };
            self.samplers.insert(info, sampler);
            sampler
        }
    }

    pub fn destroy(&mut self, device: &ash::Device) {
        self.samplers.values().for_each(|sampler| unsafe {
            device.destroy_sampler(*sampler, None);
        });
        self.samplers.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn material_tombstones_keep_indices_stable() {
        let mut cached_materials = Vec::with_capacity(10);
        for i in 0..10 {
            let mut meta = MaterialMeta::default();
            meta.material_values.base_color_factor = vec4(i as f32, 0.0, 0.0, 1.0);
            cached_materials.push(CachedMaterial::Unloaded(meta));
        }

        for id in [3usize, 4, 5] {
            if let Some(slot) = cached_materials.get_mut(id) {
                let _ = std::mem::replace(slot, CachedMaterial::_NULL);
            }
        }

        for id in 0..10 {
            if (3..=5).contains(&id) {
                assert!(matches!(cached_materials[id], CachedMaterial::_NULL));
                continue;
            }

            let expected = id as f32;
            match cached_materials[id] {
                CachedMaterial::Unloaded(meta) => {
                    assert_eq!(meta.material_values.base_color_factor.x, expected);
                }
                _ => panic!("Material slot {} should remain populated", id),
            }
        }
    }

    #[test]
    fn material_pipeline_mapping_uses_shading_model_and_alpha_mode() {
        let mut pbr_opaque = MaterialMeta::default();
        pbr_opaque.alpha_mode = AlphaMode::Opaque;
        assert_eq!(
            TextureCache::pipeline_for_material(&pbr_opaque),
            VkPipelineType::PbrMetRoughOpaque
        );

        let mut pbr_blend = MaterialMeta::default();
        pbr_blend.alpha_mode = AlphaMode::Blend;
        assert_eq!(
            TextureCache::pipeline_for_material(&pbr_blend),
            VkPipelineType::PbrMetRoughAlpha
        );

        let mut unlit_opaque = MaterialMeta::unlit(vec4(1.0, 1.0, 1.0, 1.0), None);
        unlit_opaque.alpha_mode = AlphaMode::Opaque;
        assert_eq!(
            TextureCache::pipeline_for_material(&unlit_opaque),
            VkPipelineType::UnlitOpaque
        );

        let mut unlit_blend = MaterialMeta::unlit(vec4(1.0, 1.0, 1.0, 0.5), None);
        unlit_blend.alpha_mode = AlphaMode::Blend;
        assert_eq!(
            TextureCache::pipeline_for_material(&unlit_blend),
            VkPipelineType::UnlitAlpha
        );
    }

    #[test]
    fn pending_batch_tracking_records_and_clears() {
        // Test that PendingTextureBatch types and tracking tables work correctly
        // without requiring a GPU device.
        let mut pending_batches: HashMap<u64, PendingTextureBatch> = HashMap::new();
        let mut pending_textures: HashMap<TextureHandle, u64> = HashMap::new();

        let tex_a = TextureHandle::new(10, 0);
        let tex_b = TextureHandle::new(11, 0);

        let batch = PendingTextureBatch {
            batch_id: 1,
            texture_ids: vec![tex_a, tex_b],
            image_allocs: Vec::new(), // no GPU allocs in unit test
            submitted_at: Instant::now(),
            status: UploadBatchStatus::WaitingFence,
        };

        pending_textures.insert(tex_a, 1);
        pending_textures.insert(tex_b, 1);
        pending_batches.insert(1, batch);

        assert_eq!(pending_batches.len(), 1);
        assert_eq!(pending_textures.len(), 2);
        assert_eq!(*pending_textures.get(&tex_a).unwrap(), 1u64);
        assert_eq!(*pending_textures.get(&tex_b).unwrap(), 1u64);

        // Simulate finalization: remove batch and clear texture tracking
        let removed = pending_batches.remove(&1).unwrap();
        assert!(matches!(removed.status, UploadBatchStatus::WaitingFence));
        assert_eq!(removed.texture_ids.len(), 2);

        for id in removed.texture_ids.iter() {
            pending_textures.remove(id);
        }

        assert!(pending_batches.is_empty());
        assert!(pending_textures.is_empty());
    }
}
