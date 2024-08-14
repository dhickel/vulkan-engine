use crate::data::data_util::PackUnorm;
use crate::data::gpu_data::{AlphaMode, AsByteSlice, EmissiveMap, EnvironmentUBO, MaterialMeta, MaterialValues, MeshMeta, MetRoughUniform, MetRoughUniformExt, NormalMap, OcclusionMap, Sampler, SurfaceMeta, TextureIds, TextureMeta, TextureSamplers, Vertex, VkCubeMap, VkMeshBuffers};
use crate::data::{assimp_util, data_util, gpu_data};
use crate::vulkan::vk_descriptor::{
    PoolSizeRatio, VkDescriptorAllocator, VkDescriptorWriter, VkDynamicDescriptorAllocator,
};
use crate::vulkan::vk_storage::{BufferPlacement, VkAllocResult, VkSubAllocator};
use crate::vulkan::vk_types::{VkDeviceQueues, VkBuffer, VkBufferAndDescriptorLimits, VkCommandPool, VkDestroyable, VkHostBuffer, VkImageAlloc, VkImmediate, VkPipeline, VkSubAlloc, VkQueueType, VkCmdSubmitInfo, VkFenceQueue, VkSubmitParam};

use crate::vulkan::vk_util;
use ash::vk::{Format, PFN_vkFreeDescriptorSets};
use ash::{vk, Device};
use glam::{vec3, vec4, Vec3, Vec4};
use gltf::json::Path;
use image::{EncodableLayout, GenericImageView, ImageBuffer, ImageResult, Rgb32FImage, Rgba, Rgba32FImage};
use log::{debug, error, info};
use once_cell::unsync::Lazy;
use std::collections::{HashMap, HashSet};
use std::hash::{DefaultHasher, Hasher};
use std::marker::PhantomData;
use std::path;
use std::rc::Rc;
use std::sync::{Arc, LazyLock, Mutex};
use std::sync::mpsc::{Receiver, SendError};
use std::time::Duration;
use ash::prelude::VkResult;
use vk_mem::Allocator;

///////////////////
// TEXTURE CACHE //
///////////////////

pub enum LoadResult<T> {
    Success(Option<Vec<T>>),
    Failed(Option<Vec<T>>),
}


#[derive(Debug)]
pub enum CachedTexture {
    Unloaded(TextureMeta),
    Loaded(VkLoadedTexture),
    _NULL,
}


#[derive(Debug)]
pub enum CachedMaterial {
    Unloaded(MaterialMeta),
    Loaded(VkLoadedMaterial),
}


#[derive(Debug, Clone, Copy)]
pub struct VkLoadedMaterial {
    pub texture_ids: TextureIds,
    pub meta_alloc: VkSubAlloc,
    pub image_descriptor: vk::DescriptorSet,
    pub pipeline: VkPipelineType,
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
        self.texture_cache.lock().unwrap().destroy(device, allocator);
    }
}


pub struct TextureCache {
    device: ash::Device,
    allocator: Arc<Mutex<Allocator>>,
    cached_textures: Vec<CachedTexture>,
    cached_materials: Vec<CachedMaterial>,
    desc_manager: DescriptorManager,
    supported_formats: HashSet<vk::Format>,
    sampler_cache: VkSamplerCache,
    material_meta_storage: VkSubAllocator,
    host_buffer: Arc<Mutex<VkHostBuffer>>,
    host_alignment: u64,
}


impl TextureCache {
    pub const DEFAULT_ERROR_TEX: u32 = 5;
    pub const DEFAULT_COLOR_TEX: u32 = 0;
    pub const DEFAULT_ROUGH_TEX: u32 = 1;
    pub const DEFAULT_NORMAL_TEX: u32 = 2;
    pub const DEFAULT_OCCLUSION_TEX: u32 = 3;
    pub const DEFAULT_EMISSIVE_TEX: u32 = 4;
    pub const DEFAULT_TEX_ITER_START: usize = 6;

    pub const DEFAULT_BASE_COLOR_FACTOR: Vec4 = vec4(1.0, 1.0, 1.0, 1.0);
    pub const DEFAULT_METALLIC_FACTOR: f32 = 0.0;
    pub const DEFAULT_ROUGHNESS_FACTOR: f32 = 1.0;
    pub const DEFAULT_NORMAL_SCALE: f32 = 1.0;
    pub const DEFAULT_OCCLUSION_STRENGTH: f32 = 1.0;
    pub const DEFAULT_EMISSIVE_STRENGTH: f32 = 1.0;
    pub const DEFAULT_EMISSIVE_FACTOR: Vec3 = Vec3::ZERO;

    pub const DEFAULT_MAT_ROUGH_MAT: u32 = 0;
    pub const DEFAULT_ERROR_MAT: u32 = 1;
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
        device: &ash::Device,
        allocator: Arc<Mutex<Allocator>>,
        sampler_cache: VkSamplerCache,
        supported_formats: HashSet<vk::Format>,
        meta_desc_layout: vk::DescriptorSetLayout,
        image_desc_layout: vk::DescriptorSetLayout,
        host_buffer: Arc<Mutex<VkHostBuffer>>,
        meta_buffer_size: u64,
        limits: &VkBufferAndDescriptorLimits,
    ) -> Result<Self, String> {
        let def_color = CachedTexture::Unloaded(TextureMeta {
            bytes: vec![255, 255, 255, 255],
            width: 1,
            height: 1,
            format: vk::Format::R8G8B8A8_UNORM,
            mips_levels: 1,
            uv_index: 0,
        });

        let r8_support = supported_formats.contains(&vk::Format::R8_UNORM);

        let def_metallic_rough = CachedTexture::Unloaded(TextureMeta {
            bytes: if r8_support { vec![127] } else { vec![127, 127, 127, 255] },
            width: 1,
            height: 1,
            format: if r8_support { vk::Format::R8_UNORM } else { vk::Format::R8G8B8A8_UNORM },
            mips_levels: 1,
            uv_index: 0,
        });


        let def_occlusion = CachedTexture::Unloaded(TextureMeta {
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
            uv_index: 0,
        });

        let def_normal = CachedTexture::Unloaded(TextureMeta {
            bytes: vec![128, 128, 255, 255],
            width: 1,
            height: 1,
            format: vk::Format::R8G8B8A8_UNORM,
            mips_levels: 1,
            uv_index: 0,
        });

        let def_emissive = CachedTexture::Unloaded(TextureMeta {
            bytes: vec![0, 0, 0, 255],
            width: 1,
            height: 1,
            format: vk::Format::R8G8B8A8_UNORM,
            mips_levels: 1,
            uv_index: 0,
        });

        let error_tex: [u8; 16] = [
            255, 20, 147, 255, 255, 20, 147, 255, 255, 20, 147, 255, 255, 20, 147, 255,
        ];

        let def_error = CachedTexture::Unloaded(TextureMeta {
            bytes: error_tex.to_vec(),
            width: 2,
            height: 2,
            format: vk::Format::R8G8B8A8_UNORM,
            mips_levels: 1,
            uv_index: 0,
        });

        let err_mat = CachedMaterial::Unloaded(MaterialMeta {
            texture_ids: TextureIds {
                base_color: Self::DEFAULT_ERROR_TEX,
                ..Default::default()
            },
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


        let image_desc_ratios = [PoolSizeRatio::new(vk::DescriptorType::COMBINED_IMAGE_SAMPLER, 5.0)];
        let meta_desc_ratios = [PoolSizeRatio::new(vk::DescriptorType::STORAGE_BUFFER, 1.0)];


        let image_desc_allocator = VkDynamicDescriptorAllocator::new(&device, 5_000, &image_desc_ratios).unwrap();
        let meta_desc_allocator = VkDynamicDescriptorAllocator::new(&device, 1_000, &meta_desc_ratios).unwrap();

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
            device: device.clone(),
            allocator,
            cached_textures,
            cached_materials,
            supported_formats,
            desc_manager,
            material_meta_storage,
            host_buffer,
            sampler_cache,
            host_alignment: limits.optimal_buffer_copy_offset_alignment,
        })
    }

    pub fn is_supported_format(&self, format: vk::Format) -> bool {
        self.supported_formats.contains(&format)
    }

    pub fn add_texture(&mut self, mut data: TextureMeta) -> u32 {
        let index = self.cached_textures.len();


        if !self.supported_formats.contains(&data.format) {
            info!(
                "Unsupported Format: {:?}, converting to R8G8B8A8_UNORM",
                data.format
            );

            let converted =
                ImageBuffer::<image::Rgb<u8>, _>::from_raw(data.width, data.height, data.bytes);

            if let Some(image) = converted {
                let new_bytes = image::DynamicImage::ImageRgb8(image).to_rgba8();
                data.format = vk::Format::R8G8B8A8_UNORM;
                data.bytes = new_bytes.to_vec();
            } else {
                log::info!(
                    "Error converting material of type: {:?} to RGBA. Using error texture.",
                    data.format
                );
                return Self::DEFAULT_ERROR_MAT;
            }
        }

        self.cached_textures.push(CachedTexture::Unloaded(data));
        index as u32
    }


    pub(crate) fn save_debug_image(data: &TextureMeta, bytes: &[u8], filename: String) {
        let path = path::Path::new("debug_textures").join(filename);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();

        match data.format {
            vk::Format::R8G8B8A8_UNORM | vk::Format::R8G8B8A8_SRGB => {
                if let Some(img) = ImageBuffer::<Rgba<u8>, _>::from_raw(data.width, data.height, bytes.clone()) {
                    img.save(&path).unwrap();
                }
            }
            vk::Format::R8G8B8_UNORM | vk::Format::R8G8B8_SRGB => {
                if let Some(img) = ImageBuffer::<image::Rgb<u8>, _>::from_raw(data.width, data.height, bytes.clone()) {
                    img.save(&path).unwrap();
                }
            }
            _ => {
                println!("Unsupported format for debug save: {:?}", data.format);
            }
        }

        println!("Saved debug image: {:?}", path);
    }

    pub fn add_textures(&mut self, data: Vec<TextureMeta>) -> Vec<u32> {
        data.into_iter().map(|meta| self.add_texture(meta)).collect()
    }

    pub fn add_material(&mut self, data: MaterialMeta) -> u32 {
        let index = self.cached_materials.len();
        self.cached_materials.push(CachedMaterial::Unloaded(data));
        index as u32
    }

    pub fn add_materials(&mut self, data: Vec<MaterialMeta>) -> Vec<u32> {
        data.into_iter().map(|meta| self.add_material(meta)).collect()
    }

    pub fn get_material(&self, id: u32) -> Option<&CachedMaterial> {
        self.cached_materials.get(id as usize)
    }


    pub unsafe fn get_material_unchecked(&self, id: u32) -> &CachedMaterial {
        unsafe { self.cached_materials.get_unchecked(id as usize) }
    }


    pub unsafe fn get_loaded_material_unchecked(&self, id: u32) -> VkLoadedMaterial {
        unsafe {
            match self.cached_materials.get_unchecked(id as usize) {
                CachedMaterial::Loaded(loaded) => *loaded,
                _ => std::hint::unreachable_unchecked(),
            }
        }
    }

    pub unsafe fn get_loaded_material_unchecked_ptr(&self, id: u32) -> *const VkLoadedMaterial {
        unsafe {
            match self.cached_materials.get_unchecked(id as usize) {
                CachedMaterial::Loaded(loaded) => loaded,
                _ => std::hint::unreachable_unchecked(),
            }
        }
    }

    pub fn get_texture(&self, id: u32) -> Option<&CachedTexture> {
        self.cached_textures.get(id as usize)
    }

    pub unsafe fn get_texture_unchecked(&self, id: u32) -> &CachedTexture {
        unsafe { self.cached_textures.get_unchecked(id as usize) }
    }

    pub unsafe fn get_loaded_texture_unchecked(&self, id: u32) -> &VkLoadedTexture {
        unsafe {
            match self.cached_textures.get_unchecked(id as usize) {
                CachedTexture::Loaded(loaded) => loaded,
                _ => {
                    error!("Unloaded texture: {:?}", id);
                    std::hint::unreachable_unchecked()
                }
            }
        }
    }


    pub fn is_texture_loaded(&self, id: u32) -> bool {
        if let Some(found) = self.cached_textures.get(id as usize) {
            matches!(found, CachedTexture::Loaded(_))
        } else {
            false
        }
    }

    pub fn allocate_textures(
        &mut self,
        mut texture_ids: Vec<u32>,
    ) -> bool {
        let host_buffer = self.host_buffer.lock().unwrap();
        let max_upload_bytes = host_buffer.buffer.size;

        let mut curr_bytes = 0;
        let mut next_upload = Vec::<&TextureMeta>::with_capacity(texture_ids.len());
        let mut loaded = Vec::<CachedTexture>::with_capacity(texture_ids.len());

        texture_ids.retain(|id| {
            let tex = self.cached_textures.get(*id as usize);
            matches!( tex, Some(CachedTexture::Unloaded(_)))
        });

        for id in texture_ids.iter().copied() {
            match self.cached_textures.get(id as usize) {
                Some(CachedTexture::Unloaded(meta)) => {
                    if curr_bytes + meta.bytes.len()
                        .next_multiple_of(self.host_alignment as usize) > max_upload_bytes as usize
                    {
                        let image_allocs = vk_util::record_host_to_image_buffer(
                            &self.device,
                            &self.allocator.lock().unwrap(),
                            &mut self.sampler_cache,
                            &host_buffer,
                            &next_upload,
                            self.host_alignment,
                        );

                        // Submit texture uploads, requires graphics queue for mips

                        debug!("Submitting Material Commands");
                        host_buffer.submit_transfer_commands(VkSubmitParam::signaling(vk::PipelineStageFlags2::ALL_TRANSFER)).unwrap();
                        host_buffer.submit_graphics_commands(VkSubmitParam::waiting(vk::PipelineStageFlags2::VERTEX_SHADER)).unwrap();

                        if let Err(error) = host_buffer.await_done(10) {
                            error!("Error awaiting tx upload response for textures: {:?}", error);
                            return false;
                        } else {
                            host_buffer.reset_buffers(&self.device);
                            debug!("Storage upload latch passed")
                        }


                        match image_allocs {
                            Ok(images) => {
                                curr_bytes = 0;
                                next_upload.clear();
                                assert!(!images.is_empty());
                                for image in images {
                                    let loaded_tex = VkLoadedTexture {
                                        alloc: image.0,
                                        sampler: image.1,
                                    };
                                    loaded.push(CachedTexture::Loaded(loaded_tex));
                                }
                            }
                            Err(err) => {
                                error!("Error loading textures: {:?}", err);
                                return false;
                            }
                        }
                    }
                    curr_bytes += meta.bytes.len().next_multiple_of(self.host_alignment as usize);
                    next_upload.push(meta);
                }
                _ => {
                    error!("Error loading textures: Invalid texture index");
                    return false;
                }
            }
        }


        // Upload any remaining data
        if curr_bytes > 0 {
            let image_allocs = vk_util::record_host_to_image_buffer(
                &self.device,
                &self.allocator.lock().unwrap(),
                &mut self.sampler_cache,
                &host_buffer,
                &next_upload,
                self.host_alignment,
            );

            debug!("Submitting Material Commands");
            // Submit texture uploads, requires graphics queue for mips
            host_buffer.submit_transfer_commands(VkSubmitParam::signaling(vk::PipelineStageFlags2::ALL_TRANSFER)).unwrap();
            host_buffer.submit_graphics_commands(VkSubmitParam::waiting(vk::PipelineStageFlags2::VERTEX_SHADER)).unwrap();


            if let Err(error) = host_buffer.await_done(10) {
                error!("Error awaiting tx upload response for textures: {:?}", error);
                return false;
            } else {
                host_buffer.reset_buffers(&self.device);
                debug!("Storage upload latch passed")
            }


            match image_allocs {
                Ok(images) => {
                    curr_bytes = 0;
                    next_upload.clear();

                    assert!(!images.is_empty());
                    for image in images {
                        let loaded_tex = VkLoadedTexture {
                            alloc: image.0,
                            sampler: image.1,
                        };
                        loaded.push(CachedTexture::Loaded(loaded_tex));
                    }
                }
                Err(err) => {
                    error!("Error loading textures: {:?}", err);
                    return false;
                }
            }
        }


        assert_eq!(texture_ids.len(), loaded.len());

        for (id, tex) in texture_ids.iter().zip(loaded.into_iter()) {
            self.cached_textures[*id as usize] = tex;
        }

        true
    }

    unsafe fn allocate_materials(
        &mut self,
        materials: Vec<(u32, *const MaterialMeta)>,
        buffer_placement: BufferPlacement,
        rtn_alloc: bool,
    ) -> LoadResult<VkLoadedMaterial> {
        let mut texture_ids: Vec<u32> = materials.iter().flat_map(|(id, meta)| {
            (&**meta).texture_ids.to_vec()
        }).collect();

        // Dedupe
        texture_ids.sort_unstable();
        texture_ids.dedup();

        if !self.allocate_textures(texture_ids) {
            return LoadResult::Failed(None);
        }

        let meta_bytes: Vec<&[u8]> = materials.iter()
            .map(|(id, material)| bytemuck::bytes_of(&(**material).material_values))
            .collect();

        let meta_allocs = match self.material_meta_storage.allocate_bytes(&meta_bytes, buffer_placement) {
            VkAllocResult::Success(allocs) => allocs,
            VkAllocResult::Failure { error_msg, successful_allocs } => {
                error!("Error allocating material meta: {:?}", error_msg);
                successful_allocs.into_iter().for_each(|alloc| self.material_meta_storage.deallocate(alloc));
                return LoadResult::Failed(None);
            }
        };

        let mut loaded_materials = if rtn_alloc {
            Some(Vec::<VkLoadedMaterial>::with_capacity(materials.len()))
        } else { None };

        for ((id, meta), alloc) in materials.into_iter()
            .zip(meta_allocs.into_iter()) {
            let loaded_mat = self.write_material_descriptors(&*meta, alloc);
            self.cached_materials[id as usize] = CachedMaterial::Loaded(loaded_mat);
            
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
    ) -> VkLoadedMaterial {
        let pipeline = match meta.material_values.alpha_mask {
            0.0 => VkPipelineType::PbrMetRoughOpaque,
            1.0.. => VkPipelineType::PbrMetRoughAlpha, // TODO make sure both can use same pipeline
            _ => panic!(
                "Invalid alpha mask (Valid values: 0.0..=2.0, found: {:?}",
                meta.material_values.alpha_mask
            ),
        };


        let color_tex = unsafe { self.get_loaded_texture_unchecked(meta.texture_ids.base_color) };
        let metallic_tex = unsafe { self.get_loaded_texture_unchecked(meta.texture_ids.metallic_roughness) };
        let normal_tex = unsafe { self.get_loaded_texture_unchecked(meta.texture_ids.normal_map) };
        let occlusion_tex = unsafe { self.get_loaded_texture_unchecked(meta.texture_ids.occlusion_map) };
        let emissive_tex = unsafe { self.get_loaded_texture_unchecked(meta.texture_ids.emissive_map) };

        debug!(" color id: {}", meta.texture_ids.base_color);
        debug!(" metal rough id: {}", meta.texture_ids.metallic_roughness);
        debug!(" normal id: {}", meta.texture_ids.normal_map);
        debug!(" occlusion id: {}", meta.texture_ids.occlusion_map);
        debug!(" emissive id: {}", meta.texture_ids.emissive_map);

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

        VkLoadedMaterial {
            texture_ids: meta.texture_ids,
            meta_alloc,
            image_descriptor,
            pipeline,
        }
    }


    pub fn allocate_all(
        &mut self,
        buffer_placement: BufferPlacement,
        rtn_alloc: bool,
    ) -> LoadResult<VkLoadedMaterial> {
        let id_mats: Vec<(u32, *const MaterialMeta)> = self.cached_materials.iter().enumerate()
            .filter_map(|(id, mat)| {
                if let CachedMaterial::Unloaded(meta) = mat {
                    Some((id as u32, meta as *const MaterialMeta))
                } else { None }
            }).collect();

        unsafe { self.allocate_materials(id_mats, buffer_placement, rtn_alloc) }
    }

    pub fn allocate_ids(
        &mut self,
        material_ids: &[u32],
        buffer_placement: BufferPlacement,
        rtn_alloc: bool,
    ) -> LoadResult<VkLoadedMaterial> {
        let mut existing_loads = Vec::<VkLoadedMaterial>::new();
        let mut id_mats = Vec::<(u32, *const MaterialMeta)>::with_capacity(material_ids.len());
        for id in material_ids.iter() {
            match self.cached_materials.get(*id as usize) {
                Some(CachedMaterial::Unloaded(meta)) => id_mats.push((*id, meta as *const MaterialMeta)),
                Some(CachedMaterial::Loaded(loaded)) if rtn_alloc => existing_loads.push(*loaded),
                _ => {
                    error!("Failed to located material id: {:?}", id);
                    return LoadResult::Failed(None);
                }
            }
        };


        let mut alloc_result = unsafe { self.allocate_materials(id_mats, buffer_placement, rtn_alloc) };
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
                LoadResult::Success(None) => alloc_result = LoadResult::Success(Some(existing_loads)),
                LoadResult::Failed(None) => alloc_result = LoadResult::Failed(Some(existing_loads)),
            }
        }

        alloc_result
    }


    pub fn allocate_id(
        &mut self,
        id: u32,
        buffer_placement: BufferPlacement,
        rtn_alloc: bool,
    ) -> LoadResult<VkLoadedMaterial> {
        match self.cached_materials.get(id as usize) {
            Some(CachedMaterial::Loaded(loaded)) => {
                if rtn_alloc {
                    LoadResult::Success(Some(vec![*loaded]))
                } else { LoadResult::Success(None) }
            }
            Some(CachedMaterial::Unloaded(meta)) => {
                unsafe {
                    self.allocate_materials(vec![(id, meta as *const MaterialMeta)], buffer_placement, rtn_alloc)
                }
            }
            _ => LoadResult::Failed(None)
        }
    }

    fn deallocate_textures(&mut self, texture_ids: Vec<u32>) {
        let allocator = self.allocator.lock().unwrap();

        texture_ids.into_iter().for_each(|id| {
            if (id as usize) < self.cached_textures.len() {
                if let CachedTexture::Loaded(tex) = self.cached_textures.remove(id as usize) {
                    vk_util::destroy_image(&allocator, tex.alloc)
                }
            }
        })
    }

    pub fn deallocate_materials(&mut self, material_ids: Vec<u32>) {
        let mut texture_ids = Vec::<u32>::with_capacity(material_ids.len() * 5);

        for id in material_ids {
            if (id as usize) < self.cached_materials.len() {
                if let CachedMaterial::Loaded(mat) = self.cached_materials.remove(id as usize) {
                    texture_ids.extend(mat.texture_ids.to_vec());
                    self.material_meta_storage.deallocate(mat.meta_alloc)
                }
            }
        }
        self.deallocate_textures(texture_ids);
    }
}


impl VkDestroyable for TextureCache {
    fn destroy(&mut self, device: &Device, allocator: &Allocator) {
        todo!()
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
    joint_desc_pool: VkDynamicDescriptorAllocator,
    default_joint_desc: vk::DescriptorSet,
    default_joint_buffer: VkBuffer,
}


impl MeshCache {
    const DEFAULT_JOINTS: [glam::Mat4; 128] = [glam::Mat4::IDENTITY; 128];
    const DEFAULT_MESH_ITER_START: usize = 1;
    pub const SKYBOX_MESH: u32 = 0;

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
        };

        cached_meshes.push(CachedMesh::Unloaded(skybox));

        let default_joint_buffer = vk_util::allocate_and_write_buffer(
            allocator,
            Self::DEFAULT_JOINTS.as_byte_slice(),
            vk::BufferUsageFlags::UNIFORM_BUFFER,
        ).unwrap();

        let mut joint_desc_pool = VkDynamicDescriptorAllocator::new(
            device,
            1,
            &[PoolSizeRatio::new(vk::DescriptorType::UNIFORM_BUFFER,
                1.0)],
        ).unwrap();

        let default_joint_desc = joint_desc_pool.allocate(device, &[joint_desc_layout]).unwrap();


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


    pub fn add(&mut self, data: MeshMeta) -> u32 {
        let index = self.cached_meshes.len();
        self.cached_meshes.push(CachedMesh::Unloaded(data));
        index as u32
    }

    pub fn add_multi(&mut self, data: Vec<MeshMeta>) -> Vec<u32> {
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

    pub fn get_id(&self, id: u32) -> Option<&CachedMesh> {
        self.cached_meshes.get(id as usize)
    }

    pub fn get_loaded_id_unchecked(&self, id: u32) -> VkMeshBuffers {
        unsafe {
            match self.cached_meshes.get_unchecked(id as usize) {
                CachedMesh::Loaded(buffers) => *buffers,
                _ => {
                    error!("Error getting loaded mesh unchecked: {}", id);
                    std::hint::unreachable_unchecked()
                }
            }
        }
    }

    pub fn get_ids(&self, ids: Vec<u32>) -> Vec<Option<&CachedMesh>> {
        ids.into_iter().map(|id| self.get_id(id)).collect()
    }

    pub fn is_id_loaded(&self, id: u32) -> bool {
        if let Some(found) = self.cached_meshes.get(id as usize) {
            matches!(found, CachedMesh::Loaded(_))
        } else {
            false
        }
    }

    unsafe fn allocate(
        &mut self,
        meshes: Vec<(u32, *const MeshMeta)>,
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


        let vertex_allocs = match self.vertex_storage.allocate_bytes(&mut vertex_data, buffer_placement) {
            VkAllocResult::Success(allocs) => allocs,
            VkAllocResult::Failure { error_msg, successful_allocs } => {
                successful_allocs.into_iter().for_each(|alloc| self.vertex_storage.deallocate(alloc));
                error!("Failed to allocate vertices: {:?}", error_msg);
                return LoadResult::Failed(None);
            }
        };


        let index_allocs = match self.index_storage.allocate_bytes(&mut index_data, buffer_placement) {
            VkAllocResult::Success(allocs) => allocs,
            VkAllocResult::Failure { error_msg, successful_allocs } => {
                successful_allocs.into_iter().for_each(|alloc| self.vertex_storage.deallocate(alloc));
                error!("Failed to allocate vertices: {:?}", error_msg);
                return LoadResult::Failed(None);
            }
        };


        let mut rtn_buffers = if return_buffers {
            Some(Vec::<VkMeshBuffers>::with_capacity(vertex_allocs.len()))
        } else { None };


        meshes.iter()
            .map(|(id, meta)| *id)
            .zip(vertex_allocs.into_iter())
            .zip(index_allocs.into_iter())
            .for_each(|((id, vert_alloc), index_alloc)| {
                if let CachedMesh::Unloaded(meta) = unsafe { self.cached_meshes.get_unchecked(id as usize) } {
                    let buffer = VkMeshBuffers {
                        cache_id: id,
                        index_count: meta.indices.len() as u32,
                        vertex_count: meta.vertices.len() as u32,
                        index_buffer: index_alloc.clone(),
                        vertex_buffer: vert_alloc.clone(),
                        joint_desc: self.default_joint_desc,
                        material_id: meta.material_index.unwrap_or(TextureCache::DEFAULT_MAT_ROUGH_MAT),
                    };

                    if let Some(rtn_meshes) = &mut rtn_buffers {
                        rtn_meshes.push(buffer)
                    }

                    self.cached_meshes[id as usize] = CachedMesh::Loaded(buffer);
                } else { panic!("Unreachable") }
            });

        debug!("Allocated Meshes: {:?}", meshes);
        LoadResult::Success(rtn_buffers)
    }

    pub fn allocate_all(
        &mut self,
        buffer_placement: BufferPlacement,
        return_buffers: bool,
    ) -> LoadResult<VkMeshBuffers> {
        let id_meshes: Vec<(u32, *const MeshMeta)> = self.cached_meshes.iter()
            .enumerate()
            .filter_map(|(i, mesh)| {
                if let CachedMesh::Unloaded(meta) = mesh {
                    Some((i as u32, meta as *const MeshMeta))
                } else { None }
            }).collect();

        unsafe { self.allocate(id_meshes, buffer_placement, return_buffers) }
    }

    pub fn allocate_ids(
        &mut self,
        mesh_ids: &[u32],
        buffer_placement: BufferPlacement,
        rtn_buffers: bool,
    ) -> LoadResult<VkMeshBuffers> {
        let mut existing_loads = Vec::<VkMeshBuffers>::new();
        let mut id_meshes = Vec::<(u32, *const MeshMeta)>::with_capacity(mesh_ids.len());
        for id in mesh_ids.iter() {
            match self.cached_meshes.get(*id as usize) {
                Some(CachedMesh::Unloaded(meta)) => id_meshes.push((*id, meta as *const MeshMeta)),
                Some(CachedMesh::Loaded(loaded)) if rtn_buffers => existing_loads.push(*loaded),
                _ => {
                    error!("Failed to located material id: {:?}", id);
                    return LoadResult::Failed(None);
                }
            }
        };

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
                LoadResult::Success(None) => alloc_result = LoadResult::Success(Some(existing_loads)),
                LoadResult::Failed(None) => alloc_result = LoadResult::Failed(Some(existing_loads))
            }
        }

        alloc_result
    }

    pub fn allocate_id(
        &mut self,
        mesh_id: u32,
        buffer_placement: BufferPlacement,
        return_buffers: bool,
    ) -> LoadResult<VkMeshBuffers> {
        if let Some(CachedMesh::Unloaded(meta)) = self.cached_meshes.get(mesh_id as usize) {
            unsafe { self.allocate(vec![(mesh_id, meta as *const MeshMeta)], buffer_placement, return_buffers) }
        } else {
            LoadResult::Failed(None)
        }
    }


    pub fn deallocate_id(&mut self, mesh_id: u32) {
        if (mesh_id as usize) < self.cached_meshes.len() {
            if let CachedMesh::Loaded(loaded_mesh) = self.cached_meshes.remove(mesh_id as usize) {
                self.index_storage.deallocate(loaded_mesh.index_buffer);
                self.vertex_storage.deallocate(loaded_mesh.vertex_buffer);
            }
        }
    }


    pub fn deallocate_ids(&mut self, mesh_ids: &[u32]) {
        mesh_ids.iter().for_each(|&id| self.deallocate_id(id))
    }


    pub fn deallocate_all(&mut self, allocator: &vk_mem::Allocator) {
        (Self::DEFAULT_MESH_ITER_START..self.cached_meshes.len()).for_each(|i| self.deallocate_id(i as u32))
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
#[derive(Ord, Eq, PartialEq, PartialOrd, Debug, Clone, Copy)]
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
}


impl CoreShaderType {
    const COUNT: usize = 10;
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
    BrdfLut,
    Skybox,
    EnvPreFilter,
    EnvIrradiance,
}


impl VkPipelineType {
    const COUNT: usize = 6;
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
    Empty,
}


impl VkDescType {
    const COUNT: usize = 9;
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
                8 => VkDescType::Empty,
                _ => panic!()
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
    Unloaded(TextureMeta),
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
    supported_formats: HashSet<vk::Format>,
}


impl EnvironmentCache {
    pub fn new(supported_formats: HashSet<vk::Format>) -> Self {
        Self {
            skyboxes: Vec::with_capacity(10),
            env_maps: Vec::with_capacity(10),
            supported_formats,
        }
    }

    pub fn get_skybox(&self, env_id: u32) -> &CachedEnvironment {
        unsafe { self.skyboxes.get_unchecked(env_id as usize) }
    }

    pub fn load_cubemap_file(&mut self, path: &str) -> Result<u32, String> {
        let path = path::Path::new(path);
        match image::open(path) {
            Ok(mut image) => {
                let index = self.skyboxes.len() as u32;
                let mut format = assimp_util::to_vk_format(&image);

                let image_bytes = if !self.supported_formats.contains(&format) {
                    if self
                        .supported_formats
                        .contains(&vk::Format::R32G32B32A32_SFLOAT)
                    {
                        format = vk::Format::R32G32B32A32_SFLOAT;
                        data_util::convert_rgb32f_to_rgba32f(image.to_rgb32f())
                            .as_bytes()
                            .to_vec()
                    } else {
                        panic!("No Fallback format") // Not sure if falling back to rgba8 is acceptable
                    }
                } else {
                    image.as_bytes().to_vec()
                };

                info!(
                    "Added cube map file as Unloaded: {:?} \tformat: {:?}  \twidth: {:?}, height: {:?}",
                    path,
                    format,
                    image.width(),
                    image.height()
                );

                let meta = TextureMeta {
                    bytes: image_bytes,
                    width: image.width(),
                    height: image.height(),
                    format,
                    mips_levels: 1,
                    ..Default::default()
                };

                self.skyboxes.push(CachedEnvironment::Unloaded(meta));
                self.env_maps.push(None);
                Ok(index)
            }
            Err(err) => Err(format!("Failed to add cube map file: {:?}", err)),
        }
    }

    pub fn add_env_maps(&mut self, env_id: u32, env_maps: EnvMaps) {
        self.env_maps.insert(env_id as usize, Some(env_maps))
    }

    pub fn get_env_map(&self, env_id: u32) -> &Option<EnvMaps> {
        unsafe { self.env_maps.get_unchecked(env_id as usize) }
    }

    pub fn load_cubemap_dir(&mut self, dir: &str) -> Result<u32, String> {
        let face_files = ["px.hdr", "nx.hdr", "py.hdr", "ny.hdr", "pz.hdr", "nz.hdr"];
        let mut face_images: Vec<Rgba32FImage> = Vec::new();
        let mut width = 0;
        let mut height = 0;
        let format = vk::Format::R32G32B32A32_SFLOAT;

        for face_file in face_files.iter() {
            let path = path::Path::new(dir).join(face_file);

            match image::open(&path) {
                Ok(image) => {
                    // Convert to RGBA32F
                    let rgba32f = image.into_rgba32f();

                    if width == 0 {
                        width = rgba32f.width();
                        height = rgba32f.height();
                    } else if width != rgba32f.width() || height != rgba32f.height() {
                        return Err(format!("Inconsistent face dimensions in {}", face_file));
                    }

                    face_images.push(rgba32f);

                    info!(
                        "Loaded cubemap face: {:?} \tformat: {:?} \twidth: {:?}, height: {:?}",
                        path, format, width, height
                    );
                }
                Err(err) => return Err(format!("Failed to load face {}: {:?}", face_file, err)),
            }
        }

        // Combine all face data into a single vector of f32
        let combined_data: Vec<f32> = face_images
            .into_iter()
            .flat_map(|img| img.into_raw())
            .collect();

        // Convert f32 data to bytes
        let byte_data: Vec<u8> = bytemuck::cast_slice(&combined_data).to_vec();

        let index = self.skyboxes.len() as u32;
        width *= 6;
        info!(
            "Loaded cubemap meta: \tformat: {:?}, width: {:?}, height: {:?}, total bytes: {:?}",
            format,
            width,
            height,
            byte_data.len()
        );

        let meta = TextureMeta {
            bytes: byte_data,
            width,
            height,
            format,
            mips_levels: 1,
            ..Default::default()
        };

        self.skyboxes.push(CachedEnvironment::Unloaded(meta));
        Ok(index)
    }

    pub fn allocate_cube_map(
        &mut self,
        env_id: u32,
        device: &ash::Device,
        allocator: &Allocator,
        transfer_pool: &VkCommandPool,
        transfer_queue: vk::Queue,
    ) {
        let env_id = env_id as usize;
        let texture = std::mem::replace(
            &mut self.skyboxes[env_id],
            CachedEnvironment::Unloaded(TextureMeta {
                bytes: vec![],
                width: 0,
                height: 0,
                format: vk::Format::UNDEFINED,
                mips_levels: 0,
                ..Default::default()
            }),
        );

        if let CachedEnvironment::Unloaded(meta) = texture {
            let cube_map = vk_util::upload_skybox(device, allocator, meta, transfer_pool, transfer_queue);
            self.skyboxes[env_id] = CachedEnvironment::Loaded(cube_map);
        } else {
            self.skyboxes[env_id] = texture;
            log::info!(
                "Attempted to allocate, already allocated texture: {}",
                env_id
            );
        }
    }
}


#[derive(Clone, Copy, PartialEq, Eq, Hash)]
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


#[derive(Clone, PartialEq, Eq, Hash)]
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
    pub fn to_create_info(&self) -> vk::SamplerCreateInfo {
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
}
