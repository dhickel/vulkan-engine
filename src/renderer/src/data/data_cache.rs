use crate::data::data_util::PackUnorm;
use crate::data::gpu_data;
use crate::data::gpu_data::{
    AlphaMode, EmissiveMap, MaterialMeta, MeshMeta, MetRoughUniform, MetRoughUniformExt, NormalMap,
    OcclusionMap, Sampler, SurfaceMeta, TextureMeta, Vertex, VkGpuMeshBuffers,
};
use crate::vulkan::vk_descriptor::{
    DescriptorAllocator, DescriptorWriter, PoolSizeRatio, VkDynamicDescriptorAllocator,
};
use crate::vulkan::vk_types::{VkBuffer, VkDestroyable, VkImageAlloc, VkPipeline};
use crate::vulkan::vk_util;
use ash::vk::Format;
use ash::{vk, Device};
use glam::{vec3, vec4, Vec3, Vec4};
use image::ImageBuffer;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use vk_mem::Allocator;

///////////////////
// TEXTURE CACHE //
///////////////////

#[derive(Debug)]
pub enum CachedTexture {
    UnLoaded(TextureMeta),
    Loaded(VkLoadedTexture),
}

#[derive(Debug)]
pub enum CacheMaterial {
    Unloaded(MaterialMeta),
    Loaded(VkLoadedMaterial),
}

#[derive(Debug)]
pub struct VkLoadedMaterial {
    pub meta: MaterialMeta,
    pub descriptors: [vk::DescriptorSet; 2],
    pub pipeline: VkPipelineType,
    pub uniform_buffer: VkBuffer,
    pub buffer_offset: u32,
}

#[derive(Debug)]
pub struct VkLoadedTexture {
    pub meta: TextureMeta,
    pub alloc: VkImageAlloc,
}

#[derive(Debug)]
pub struct TextureCache {
    cached_textures: Vec<CachedTexture>,
    cached_materials: Vec<CacheMaterial>,
    image_descriptors: [VkDynamicDescriptorAllocator; 2],
    supported_formats: HashSet<vk::Format>,
    pub linear_sampler: vk::Sampler,
    pub nearest_sampler: vk::Sampler,
}

impl TextureCache {
    pub const DEFAULT_COLOR_TEX: u32 = 0;
    pub const DEFAULT_ROUGH_TEX: u32 = 1;
    pub const DEFAULT_ERROR_TEX: u32 = 2;
    pub const DEFAULT_NORMAL_TEX: u32 = 3;
    pub const DEFAULT_OCCLUSION_TEX: u32 = 4;
    pub const DEFAULT_EMISSIVE_TEX: u32 = 5;

    pub const DEFAULT_BASE_COLOR_FACTOR: Vec4 = vec4(1.0, 1.0, 1.0, 1.0);
    pub const DEFAULT_METALLIC_FACTOR: f32 = 0.0;
    pub const DEFAULT_ROUGHNESS_FACTOR: f32 = 1.0;
    pub const DEFAULT_NORMAL_SCALE: f32 = 1.0;
    pub const DEFAULT_OCCLUSION_STRENGTH: f32 = 1.0;
    pub const DEFAULT_EMISSIVE_FACTOR: Vec3 = Vec3::ZERO;

    pub const DEFAULT_MAT_ROUGH_MAT: u32 = 0;
    pub const DEFAULT_ERROR_MAT: u32 = 1;

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

    pub fn new(device: &ash::Device, supported_formats: HashSet<vk::Format>) -> Self {
        let def_color = CachedTexture::UnLoaded(TextureMeta {
            bytes: vec![255, 255, 255, 255],
            width: 1,
            height: 1,
            format: vk::Format::R8G8B8A8_UNORM,
            mips_levels: 1,
            sampler: Sampler::Linear,
        });

        let def_rough = CachedTexture::UnLoaded(TextureMeta {
            bytes: vec![128, 128, 128, 255],
            width: 1,
            height: 1,
            format: vk::Format::R8G8B8A8_UNORM,
            mips_levels: 1,
            sampler: Sampler::Linear,
        });

        let r8_support = supported_formats.contains(&vk::Format::R8_UNORM);

        let def_occlusion = CachedTexture::UnLoaded(TextureMeta {
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
            sampler: Sampler::Linear,
        });

        let def_normal = CachedTexture::UnLoaded(TextureMeta {
            bytes: vec![128, 128, 128, 255],
            width: 1,
            height: 1,
            format: vk::Format::R8G8B8A8_UNORM,
            mips_levels: 1,
            sampler: Sampler::Nearest,
        });

        let def_emissive = CachedTexture::UnLoaded(TextureMeta {
            bytes: vec![0, 0, 0, 0],
            width: 1,
            height: 1,
            format: vk::Format::R8G8B8A8_UNORM,
            mips_levels: 1,
            sampler: Sampler::Linear,
        });

        let def_mat = CacheMaterial::Unloaded(MaterialMeta {
            base_color_factor: Vec4::new(1.0, 1.0, 1.0, 1.0),
            base_color_tex_id: 0,
            metallic_factor: 0.0,
            roughness_factor: 1.0,
            metallic_roughness_tex_id: 1,
            alpha_mode: gpu_data::AlphaMode::Opaque,
            alpha_cutoff: 0.5,
            normal_map: None,
            occlusion_map: None,
            emissive_map: None,
        });

        let black_val = glam::vec4(0.0, 0.0, 0.0, 0.0).pack_unorm_4x8();
        let magenta_val = glam::vec4(1.0, 0.0, 1.0, 1.0).pack_unorm_4x8();
        let mut error_val = vec![0_u32; 16 * 16]; // 16x16 checkerboard texture
        for y in 0..16 {
            for x in 0..16 {
                error_val[y * 16 + x] = if ((x % 2) ^ (y % 2)) != 0 {
                    magenta_val
                } else {
                    black_val
                };
            }
        }

        let def_error = CachedTexture::UnLoaded(TextureMeta {
            bytes: error_val
                .iter()
                .flat_map(|&pixel| pixel.to_le_bytes())
                .collect(),
            width: 16,
            height: 16,
            format: vk::Format::R8G8B8A8_UNORM,
            mips_levels: 1,
            sampler: Sampler::Linear,
        });

        let err_mat = CacheMaterial::Unloaded(MaterialMeta {
            base_color_factor: Vec4::new(1.0, 1.0, 1.0, 1.0),
            base_color_tex_id: 2,
            metallic_factor: 0.0,
            roughness_factor: 1.0,
            metallic_roughness_tex_id: 1,
            alpha_mode: gpu_data::AlphaMode::Opaque,
            alpha_cutoff: 0.5,
            normal_map: None,
            occlusion_map: None,
            emissive_map: None,
        });

        let mut cached_textures = Vec::with_capacity(100);
        cached_textures.push(def_color);
        cached_textures.push(def_rough);
        cached_textures.push(def_error);
        cached_textures.push(def_normal);
        cached_textures.push(def_occlusion);
        cached_textures.push(def_emissive);

        let mut cached_materials = Vec::with_capacity(100);
        cached_materials.push(def_mat);
        cached_materials.push(err_mat);

        let mut sampler = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::NEAREST)
            .min_filter(vk::Filter::NEAREST);

        let nearest_sampler = unsafe { device.create_sampler(&sampler, None).unwrap() };

        sampler.mag_filter = vk::Filter::LINEAR;
        sampler.min_filter = vk::Filter::LINEAR;

        let linear_sampler = unsafe { device.create_sampler(&sampler, None).unwrap() };

        let pool_ratios = [
            PoolSizeRatio::new(vk::DescriptorType::STORAGE_IMAGE, 3.0),
            PoolSizeRatio::new(vk::DescriptorType::STORAGE_BUFFER, 3.0),
            PoolSizeRatio::new(vk::DescriptorType::COMBINED_IMAGE_SAMPLER, 7.0),
        ];

        let image_descriptors = [
            VkDynamicDescriptorAllocator::new(device, 10_000, &pool_ratios).unwrap(),
            VkDynamicDescriptorAllocator::new(device, 10_000, &pool_ratios).unwrap(),
        ];

        Self {
            cached_textures,
            cached_materials,
            supported_formats,
            linear_sampler,
            nearest_sampler,
            image_descriptors,
        }
    }

    pub fn is_supported_format(&self, format: vk::Format) -> bool {
        self.supported_formats.contains(&format)
    }

    pub fn get_sampler(&self, sampler: Sampler) -> vk::Sampler {
        match sampler {
            Sampler::Linear => self.linear_sampler,
            Sampler::Nearest => self.nearest_sampler,
        }
    }

    pub fn add_texture(&mut self, mut data: TextureMeta) -> u32 {
        let index = self.cached_textures.len();

        // TODO roughness textures can actually just be R8 UNORM to save space and bandwidth
        if !self.supported_formats.contains(&data.format) {
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

        self.cached_textures.push(CachedTexture::UnLoaded(data));
        index as u32
    }

    pub fn add_textures(&mut self, data: Vec<(u32, TextureMeta)>) -> HashMap<u32, u32> {
        let mut index = self.cached_textures.len() as u32;
        let mut index_pairs = HashMap::<u32, u32>::with_capacity(data.len());

        for (ext_idx, data) in data {
            index_pairs.insert(ext_idx, index);
            index += 1;
            self.cached_textures.push(CachedTexture::UnLoaded(data))
        }
        index_pairs
    }

    pub fn add_material(&mut self, mut data: MaterialMeta) -> u32 {
        println!("{:#?}", data);
        if data.is_ext() {
            if data.normal_map.is_none() {
                data.normal_map = Some(Self::DEFAULT_NORMAL_MAP);
            }
            if data.occlusion_map.is_none() {
                data.occlusion_map = Some(Self::DEFAULT_OCCLUSION_MAP)
            }
            if data.emissive_map.is_none() {
                data.emissive_map = Some(Self::DEFAULT_EMISSIVE_MAP)
            }
        }

        let index = self.cached_materials.len();
        self.cached_materials.push(CacheMaterial::Unloaded(data));
        index as u32
    }

    pub fn get_material(&self, id: u32) -> Option<&CacheMaterial> {
        self.cached_materials.get(id as usize)
    }

    pub fn get_material_unchecked(&self, id: u32) -> &CacheMaterial {
        unsafe { self.cached_materials.get_unchecked(id as usize) }
    }

    pub fn get_loaded_material_unchecked(&self, id: u32) -> &VkLoadedMaterial {
        unsafe {
            match self.cached_materials.get_unchecked(id as usize) {
                CacheMaterial::Loaded(loaded) => loaded,
                _ => std::hint::unreachable_unchecked(),
            }
        }
    }

    pub fn get_loaded_material_unchecked_ptr(&self, id: u32) -> *const VkLoadedMaterial {
        unsafe {
            match self.cached_materials.get_unchecked(id as usize) {
                CacheMaterial::Loaded(loaded) => loaded,
                _ => std::hint::unreachable_unchecked(),
            }
        }
    }

    pub fn get_texture(&self, id: u32) -> Option<&CachedTexture> {
        self.cached_textures.get(id as usize)
    }

    pub fn get_texture_unchecked(&self, id: u32) -> &CachedTexture {
        unsafe { self.cached_textures.get_unchecked(id as usize) }
    }

    pub fn get_loaded_texture_unchecked(&self, id: u32) -> &VkLoadedTexture {
        unsafe {
            match self.cached_textures.get_unchecked(id as usize) {
                CachedTexture::Loaded(loaded) => loaded,
                _ => std::hint::unreachable_unchecked(),
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

    pub fn get_descriptor_allocator(&self, index: u32) -> &VkDynamicDescriptorAllocator {
        unsafe { self.image_descriptors.get_unchecked(index as usize) }
    }

    pub fn allocate_texture<F>(&mut self, mut upload_fn: F, tex_id: u32)
    where
        F: Fn(&[u8], vk::Extent3D, vk::Format, vk::ImageUsageFlags, bool) -> VkImageAlloc,
    {
        let tex_id = tex_id as usize;
        let texture = std::mem::replace(
            &mut self.cached_textures[tex_id],
            CachedTexture::UnLoaded(TextureMeta {
                bytes: vec![],
                width: 0,
                height: 0,
                format: vk::Format::UNDEFINED,
                mips_levels: 0,
                sampler: Sampler::Linear,
            }),
        );

        if let CachedTexture::UnLoaded(meta) = texture {
            let size = vk::Extent3D {
                width: meta.width,
                height: meta.height,
                depth: 1,
            };

            let alloc = upload_fn(
                &meta.bytes,
                size,
                meta.format,
                vk::ImageUsageFlags::SAMPLED,
                meta.mips_levels > 1,
            );

            let loaded_texture = VkLoadedTexture { meta, alloc };
            self.cached_textures[tex_id] = CachedTexture::Loaded(loaded_texture);
        } else {
            self.cached_textures[tex_id] = texture;
            log::info!(
                "Attempted to allocate, already allocated texture: {}",
                tex_id
            );
        }
    }

    pub fn allocate_material(
        &mut self,
        device: &ash::Device,
        allocator: Arc<Mutex<vk_mem::Allocator>>,
        desc_layout_cache: &VkDescLayoutCache,
        mat_id: u32,
    ) {
        let mat_id = mat_id as usize;
        let material = std::mem::replace(
            &mut self.cached_materials[mat_id],
            CacheMaterial::Unloaded(MaterialMeta {
                base_color_factor: Default::default(),
                base_color_tex_id: 0,
                metallic_factor: 0.0,
                roughness_factor: 0.0,
                metallic_roughness_tex_id: 0,
                alpha_mode: gpu_data::AlphaMode::Opaque,
                alpha_cutoff: 0.0,
                normal_map: None,
                occlusion_map: None,
                emissive_map: None,
            }),
        );

        if let CacheMaterial::Unloaded(meta) = material {
            let loaded_material: VkLoadedMaterial;
            if meta.is_ext() {
                let shader_consts = MetRoughUniformExt {
                    color_factors: meta.base_color_factor,
                    metal_rough_factors: vec4(
                        meta.metallic_factor,
                        meta.roughness_factor,
                        0.0,
                        0.0,
                    ),
                    normal_scale: vec4(meta.normal_map.unwrap().scale, 0.0, 0.0, 0.0),
                    occlusion_strength: vec4(meta.occlusion_map.unwrap().strength, 0.0, 0.0, 0.0),
                    emissive_factor: meta.emissive_map.unwrap().factor.extend(0.0),
                    extra: [Vec4::ZERO; 11],
                };

                let const_bytes = bytemuck::bytes_of(&shader_consts);

                let uniform_buffer = vk_util::allocate_and_write_buffer(
                    &allocator.lock().unwrap(),
                    const_bytes,
                    vk::BufferUsageFlags::UNIFORM_BUFFER,
                )
                .unwrap();

                loaded_material =
                    self.write_material_ext(meta, uniform_buffer, 0, device, desc_layout_cache);
            } else {
                let shader_consts = MetRoughUniform {
                    color_factors: meta.base_color_factor,
                    metal_rough_factors: vec4(
                        meta.metallic_factor,
                        meta.roughness_factor,
                        0.0,
                        0.0,
                    ),
                    extra: [Vec4::ZERO; 14],
                };

                let const_bytes = bytemuck::bytes_of(&shader_consts);

                let uniform_buffer = vk_util::allocate_and_write_buffer(
                    &allocator.lock().unwrap(),
                    const_bytes,
                    vk::BufferUsageFlags::UNIFORM_BUFFER,
                )
                .unwrap();

                loaded_material =
                    self.write_material(meta, uniform_buffer, 0, device, desc_layout_cache);
            }

            self.cached_materials[mat_id] = CacheMaterial::Loaded(loaded_material);
        } else {
            self.cached_materials[mat_id] = material;
            log::info!(
                "Attempted to allocate already allocated material: {}",
                mat_id
            );
        }
    }

    fn write_material(
        &mut self,
        meta: MaterialMeta,
        uniform_buffer: VkBuffer,
        buffer_offset: u32,
        device: &ash::Device,
        desc_layout_cache: &VkDescLayoutCache,
    ) -> VkLoadedMaterial {
        let color_tex = self.get_loaded_texture_unchecked(meta.base_color_tex_id);
        let metallic_tex = self.get_loaded_texture_unchecked(meta.metallic_roughness_tex_id);

        let pipeline = match meta.alpha_mode {
            AlphaMode::Opaque => VkPipelineType::PbrMetRoughOpaque,
            AlphaMode::Mask | AlphaMode::Blend => VkPipelineType::PbrMetRoughAlpha,
        };
        let mut writer = DescriptorWriter::default();

        writer.write_buffer(
            0,
            uniform_buffer.buffer,
            std::mem::size_of::<MetRoughUniform>(),
            buffer_offset as usize,
            vk::DescriptorType::UNIFORM_BUFFER,
        );

        writer.write_image(
            1,
            color_tex.alloc.image_view,
            self.get_sampler(color_tex.meta.sampler),
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        );

        writer.write_image(
            2,
            metallic_tex.alloc.image_view,
            self.get_sampler(metallic_tex.meta.sampler),
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        );

        // TODO maybe store the size rations with the layouts?
        let layout = [desc_layout_cache.get(VkDescType::PbrMetRough)];

        let descriptors: [vk::DescriptorSet; 2] = [
            {
                let descriptor = &self.image_descriptors[0].allocate(device, &layout).unwrap();
                writer.update_set(device, *descriptor);
                *descriptor
            },
            {
                let descriptor = &self.image_descriptors[1].allocate(device, &layout).unwrap();
                writer.update_set(device, *descriptor);
                *descriptor
            },
        ];

        VkLoadedMaterial {
            meta,
            descriptors,
            pipeline,
            uniform_buffer,
            buffer_offset,
        }
    }

    fn write_material_ext(
        &mut self,
        meta: MaterialMeta,
        uniform_buffer: VkBuffer,
        buffer_offset: u32,
        device: &ash::Device,
        desc_layout_cache: &VkDescLayoutCache,
    ) -> VkLoadedMaterial {
        let color_tex = self.get_loaded_texture_unchecked(meta.base_color_tex_id);
        let metallic_tex = self.get_loaded_texture_unchecked(meta.metallic_roughness_tex_id);
        let normal_tex = self.get_loaded_texture_unchecked(meta.normal_map.unwrap().texture_id);
        let occlusion_tex =
            self.get_loaded_texture_unchecked(meta.occlusion_map.unwrap().texture_id);
        let emissive_tex = self.get_loaded_texture_unchecked(meta.emissive_map.unwrap().texture_id);

        let pipeline = match meta.alpha_mode {
            AlphaMode::Opaque => VkPipelineType::PbrMetRoughOpaqueExt,
            AlphaMode::Mask | AlphaMode::Blend => VkPipelineType::PbrMetRoughAlphaExt,
        };
        let mut writer = DescriptorWriter::default();

        writer.write_buffer(
            0,
            uniform_buffer.buffer,
            std::mem::size_of::<MetRoughUniform>(),
            buffer_offset as usize,
            vk::DescriptorType::UNIFORM_BUFFER,
        );

        writer.write_image(
            1,
            color_tex.alloc.image_view,
            self.get_sampler(color_tex.meta.sampler),
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        );

        writer.write_image(
            2,
            metallic_tex.alloc.image_view,
            self.get_sampler(metallic_tex.meta.sampler),
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        );

        writer.write_image(
            3,
            normal_tex.alloc.image_view,
            self.get_sampler(normal_tex.meta.sampler),
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        );

        writer.write_image(
            4,
            occlusion_tex.alloc.image_view,
            self.get_sampler(occlusion_tex.meta.sampler),
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        );

        writer.write_image(
            5,
            emissive_tex.alloc.image_view,
            self.get_sampler(emissive_tex.meta.sampler),
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        );

        // TODO maybe store the size rations with the layouts?
        let layout = [desc_layout_cache.get(VkDescType::PbrMetRoughExt)];

        let descriptors: [vk::DescriptorSet; 2] = [
            {
                let descriptor = &self.image_descriptors[0].allocate(device, &layout).unwrap();
                writer.update_set(device, *descriptor);
                *descriptor
            },
            {
                let descriptor = &self.image_descriptors[1].allocate(device, &layout).unwrap();
                writer.update_set(device, *descriptor);
                *descriptor
            },
        ];

        VkLoadedMaterial {
            meta,
            descriptors,
            pipeline,
            uniform_buffer,
            buffer_offset,
        }
    }

    pub fn deallocate_material(&mut self, allocator: &vk_mem::Allocator, mat_id: u32) {
        let mat_id = mat_id as usize;
        let material = std::mem::replace(
            &mut self.cached_materials[mat_id],
            CacheMaterial::Unloaded(MaterialMeta {
                base_color_factor: Default::default(),
                base_color_tex_id: 0,
                metallic_factor: 0.0,
                roughness_factor: 0.0,
                metallic_roughness_tex_id: 0,
                alpha_mode: gpu_data::AlphaMode::Opaque,
                alpha_cutoff: 0.0,
                normal_map: None,
                occlusion_map: None,
                emissive_map: None,
            }),
        );

        if let CacheMaterial::Loaded(loaded_material) = material {
            vk_util::destroy_buffer(allocator, loaded_material.uniform_buffer);
            let unloaded_material = CacheMaterial::Unloaded(loaded_material.meta);
            self.cached_materials[mat_id] = unloaded_material;
        } else {
            self.cached_materials[mat_id] = material;
        }
    }

    pub fn deallocate_texture(&mut self, allocator: &vk_mem::Allocator, tex_id: u32) {
        let tex_id = tex_id as usize;
        let texture = std::mem::replace(
            &mut self.cached_textures[tex_id],
            CachedTexture::UnLoaded(TextureMeta {
                bytes: vec![],
                width: 0,
                height: 0,
                format: vk::Format::UNDEFINED,
                mips_levels: 0,
                sampler: Sampler::Linear,
            }),
        );

        if let CachedTexture::Loaded(loaded_texture) = texture {
            vk_util::destroy_image(allocator, loaded_texture.alloc);
            let unloaded_texture = CachedTexture::UnLoaded(loaded_texture.meta);
            self.cached_textures[tex_id] = unloaded_texture;
        } else {
            self.cached_textures[tex_id] = texture;
        }
    }

    pub fn allocate_all<F>(
        &mut self,
        upload_fn: F,
        device: &ash::Device,
        allocator: Arc<Mutex<Allocator>>,
        desc_layout_cache: &VkDescLayoutCache,
    ) where
        F: Fn(&[u8], vk::Extent3D, vk::Format, vk::ImageUsageFlags, bool) -> VkImageAlloc,
    {
        for x in 0..self.cached_textures.len() {
            self.allocate_texture(&upload_fn, x as u32);
        }

        for x in 0..self.cached_materials.len() {
            self.allocate_material(device, allocator.clone(), desc_layout_cache, x as u32);
        }
    }

    pub fn deallocate_all(&mut self, allocator: &vk_mem::Allocator) {
        for x in 0..self.cached_textures.len() {
            self.deallocate_texture(allocator, x as u32)
        }

        for x in 0..self.cached_materials.len() {
            self.deallocate_material(allocator, x as u32)
        }
    }

    // pub fn get_loaded_texture_unchecked(&self, id : u32) -> &
}

impl VkDestroyable for TextureCache {
    fn destroy(&mut self, device: &Device, allocator: &Allocator) {
        for tex in self.cached_textures.drain(..) {
            if let CachedTexture::Loaded(mut loaded) = tex {
                loaded.alloc.destroy(device, allocator)
            }
        }

        for mat in self.cached_materials.drain(..) {
            if let CacheMaterial::Loaded(mut loaded) = mat {
                loaded.uniform_buffer.destroy(device, allocator);
            }
        }

        self.image_descriptors
            .iter_mut()
            .for_each(|desc| desc.destroy(device, allocator));

        unsafe {
            device.destroy_sampler(self.linear_sampler, None);
            device.destroy_sampler(self.nearest_sampler, None);
        }
    }
}

////////////////
// MESH CACHE //
////////////////

#[derive(Debug)]
pub enum CachedMesh {
    UnLoaded(MeshMeta),
    Loaded(VkLoadedMesh),
}

#[derive(Debug)]
pub struct VkLoadedMesh {
    pub meta: MeshMeta,
    pub buffer: VkGpuMeshBuffers,
}

#[derive(Default, Debug)]
pub struct MeshCache {
    cached_meshes: Vec<CachedMesh>,
    cached_surface: Vec<SurfaceMeta>,
}

impl MeshCache {
    pub fn add_mesh(&mut self, data: MeshMeta) -> u32 {
        let index = self.cached_meshes.len();

        self.cached_meshes.push(CachedMesh::UnLoaded(data));
        index as u32
    }

    pub fn add_meshes(&mut self, data: Vec<(u32, MeshMeta)>) -> HashMap<u32, u32> {
        let mut index = self.cached_meshes.len() as u32;
        let mut index_pairs = HashMap::<u32, u32>::with_capacity(data.len());

        for (ext_idx, data) in data {
            index_pairs.insert(ext_idx, index);
            index += 1;
            self.cached_meshes.push(CachedMesh::UnLoaded(data))
        }
        index_pairs
    }

    pub fn get_mesh(&self, id: u32) -> Option<&CachedMesh> {
        self.cached_meshes.get(id as usize)
    }

    pub fn get_mesh_unchecked(&self, id: u32) -> &CachedMesh {
        unsafe { self.cached_meshes.get_unchecked(id as usize) }
    }

    pub fn is_mesh_loaded(&self, id: u32) -> bool {
        if let Some(found) = self.cached_meshes.get(id as usize) {
            matches!(found, CachedMesh::Loaded(_))
        } else {
            false
        }
    }

    pub fn get_loaded_mesh_unchecked(&self, id: u32) -> &VkLoadedMesh {
        unsafe {
            match self.cached_meshes.get_unchecked(id as usize) {
                CachedMesh::Loaded(loaded) => loaded,
                _ => std::hint::unreachable_unchecked(),
            }
        }
    }

    pub fn allocate_mesh<F>(&mut self, mut upload_fn: F, mesh_id: usize)
    where
        F: Fn(&[u32], &[Vertex]) -> VkGpuMeshBuffers,
    {
        let mesh = std::mem::replace(
            &mut self.cached_meshes[mesh_id],
            CachedMesh::UnLoaded(MeshMeta {
                name: "".to_string(),
                indices: vec![],
                vertices: vec![],
                material_index: None,
            }),
        );

        if let CachedMesh::UnLoaded(meta) = mesh {
            let buffer = upload_fn(&meta.indices, &meta.vertices);

            let loaded_mesh = VkLoadedMesh { meta, buffer };
            self.cached_meshes[mesh_id] = CachedMesh::Loaded(loaded_mesh);
        } else {
            self.cached_meshes[mesh_id] = mesh;
            log::info!(
                "Attempted to allocate, already allocated texture: {}",
                mesh_id
            );
        }
    }

    pub fn deallocate_mesh(&mut self, allocator: &vk_mem::Allocator, mesh_id: usize) {
        let texture = std::mem::replace(
            &mut self.cached_meshes[mesh_id],
            CachedMesh::UnLoaded(MeshMeta {
                name: "".to_string(),
                indices: vec![],
                vertices: vec![],
                material_index: None,
            }),
        );

        if let CachedMesh::Loaded(loaded_mesh) = texture {
            vk_util::destroy_mesh_buffers(allocator, loaded_mesh.buffer);
            let unloaded_mesh = CachedMesh::UnLoaded(loaded_mesh.meta);
            self.cached_meshes[mesh_id] = unloaded_mesh;
        } else {
            self.cached_meshes[mesh_id] = texture;
        }
    }

    pub fn allocate_all<F>(&mut self, upload_fn: F)
    where
        F: Fn(&[u32], &[Vertex]) -> VkGpuMeshBuffers,
    {
        for x in 0..self.cached_meshes.len() {
            self.allocate_mesh(&upload_fn, x);
        }
    }

    pub fn deallocate_all(&mut self, allocator: &vk_mem::Allocator) {
        for x in 0..self.cached_meshes.len() {
            self.deallocate_mesh(allocator, x)
        }
    }
}

impl VkDestroyable for MeshCache {
    fn destroy(&mut self, device: &Device, allocator: &Allocator) {
        for mesh in self.cached_meshes.drain(..) {
            if let CachedMesh::Loaded(mut loaded) = mesh {
                loaded.buffer.index_buffer.destroy(device, allocator);
                loaded.buffer.vertex_buffer.destroy(device, allocator);
            }
        }
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
    MetRoughVertExt,
    MetRoughFragExt,
}

pub struct VkShaderCache {
    pub core_shader_cache: [vk::ShaderModule; 4],
    pub user_shader_cache: Vec<vk::ShaderModule>,
}

impl VkShaderCache {
    pub fn new(
        device: &ash::Device,
        shader_paths: Vec<(CoreShaderType, String)>,
    ) -> Result<Self, String> {
        let mut compiled_shaders = shader_paths
            .iter()
            .map(|(typ, path)| {
                vk_util::load_shader_module(&device, path).map(|shader| (*typ, shader))
            })
            .collect::<Result<Vec<(CoreShaderType, vk::ShaderModule)>, String>>()?;

        compiled_shaders.sort_by_key(|(typ, path)| *typ);

        let sorted_shaders: [vk::ShaderModule; 4] = compiled_shaders
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
    PbrMetRoughOpaqueExt,
    PbrMetRoughAlphaExt,
    // Mesh,
}

//#[derive(Clone, Copy)]
pub struct VkPipelineCache {
    pipelines: [VkPipeline; 4],
}

impl VkPipelineCache {
    pub fn new(mut pipelines: Vec<(VkPipelineType, VkPipeline)>) -> Result<Self, String> {
        pipelines.sort_by_key(|(typ, _)| *typ);

        // Convert sorted vectors to fixed-size arrays
        let sorted_pipelines: [VkPipeline; 4] = pipelines
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
    GpuScene,
    PbrMetRough,
    PbrMetRoughExt,
}
pub struct VkDescLayoutCache {
    layouts: [vk::DescriptorSetLayout; 4],
}

impl VkDescLayoutCache {
    pub fn new(mut layouts: Vec<(VkDescType, vk::DescriptorSetLayout)>) -> Self {
        layouts.sort();

        let sorted_layouts: [vk::DescriptorSetLayout; 4] = layouts
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
}

impl VkDestroyable for VkDescLayoutCache {
    fn destroy(&mut self, device: &Device, allocator: &Allocator) {
        self.layouts.iter().for_each(|layout| unsafe {
            device.destroy_descriptor_set_layout(*layout, None);
        })
    }
}
