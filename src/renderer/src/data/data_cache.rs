use crate::data::data_util::PackUnorm;
use crate::data::gpu_data::{
    AlphaMode, EmissiveMap, MaterialMeta, MeshMeta, MetRoughUniform, MetRoughUniformExt, NormalMap,
    OcclusionMap, Sampler, SurfaceMeta, TextureMeta, Vertex, VkCubeMap, VkGpuMeshBuffers,
};
use crate::data::{assimp_util, data_util, gpu_data};
use crate::vulkan::vk_descriptor::{
    PoolSizeRatio, VkDescriptorAllocator, VkDescriptorWriter, VkDynamicDescriptorAllocator,
};
use crate::vulkan::vk_types::{
    VkBuffer, VkCommandPool, VkDestroyable, VkImageAlloc, VkImmediate, VkPipeline,
};
use crate::vulkan::vk_util;
use ash::vk::Format;
use ash::{vk, Device};
use glam::{vec3, vec4, Vec3, Vec4};
use gltf::json::Path;
use image::{
    EncodableLayout, GenericImageView, ImageBuffer, ImageResult, Rgb32FImage, Rgba32FImage,
};
use log::info;
use once_cell::unsync::Lazy;
use std::collections::{HashMap, HashSet};
use std::hash::{DefaultHasher, Hasher};
use std::path;
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use vk_mem::Allocator;

///////////////////
// TEXTURE CACHE //
///////////////////

#[derive(Debug)]
pub enum CachedTexture {
    Unloaded(TextureMeta),
    Loaded(VkLoadedTexture),
}

#[derive(Debug)]
pub enum CachedMaterial {
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
    cached_materials: Vec<CachedMaterial>,
    image_descriptors: [VkDynamicDescriptorAllocator; 2],
    supported_formats: HashSet<vk::Format>,
    linear_sampler: vk::Sampler,
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

    const PLACEHOLDER_MAT: CachedMaterial = CachedMaterial::Unloaded(MaterialMeta {
        base_color_factor: Vec4::ZERO,
        base_color_tex_id: 0,
        metallic_factor: 0.0,
        roughness_factor: 0.0,
        metallic_roughness_tex_id: 0,
        alpha_mode: gpu_data::AlphaMode::Opaque,
        alpha_cutoff: 0.0,
        normal_map: Self::DEFAULT_NORMAL_MAP,
        occlusion_map: Self::DEFAULT_OCCLUSION_MAP,
        emissive_map: Self::DEFAULT_EMISSIVE_MAP,
    });

    pub fn new(device: &ash::Device, supported_formats: HashSet<vk::Format>) -> Self {
        let def_color = CachedTexture::Unloaded(TextureMeta {
            bytes: vec![255, 255, 255, 255],
            width: 1,
            height: 1,
            format: vk::Format::R8G8B8A8_UNORM,
            mips_levels: 1,
            uv_index: 0,
        });

        let def_rough = CachedTexture::Unloaded(TextureMeta {
            bytes: vec![0, 127, 0, 255],
            width: 1,
            height: 1,
            format: vk::Format::R8G8B8A8_UNORM,
            mips_levels: 1,
            uv_index: 0,
        });

        let r8_support = supported_formats.contains(&vk::Format::R8_UNORM);

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
            bytes: vec![128, 128, 128, 255],
            width: 1,
            height: 1,
            format: vk::Format::R8G8B8A8_UNORM,
            mips_levels: 1,
            uv_index: 0,
        });

        let def_emissive = CachedTexture::Unloaded(TextureMeta {
            bytes: vec![0, 0, 0, 0],
            width: 1,
            height: 1,
            format: vk::Format::R8G8B8A8_UNORM,
            mips_levels: 1,
            uv_index: 0,
        });

        let def_mat = CachedMaterial::Unloaded(MaterialMeta {
            base_color_factor: Vec4::new(1.0, 1.0, 1.0, 1.0),
            base_color_tex_id: 0,
            metallic_factor: 0.0,
            roughness_factor: 1.0,
            metallic_roughness_tex_id: 1,
            alpha_mode: gpu_data::AlphaMode::Opaque,
            alpha_cutoff: 0.5,
            normal_map: Self::DEFAULT_NORMAL_MAP,
            occlusion_map: Self::DEFAULT_OCCLUSION_MAP,
            emissive_map: Self::DEFAULT_EMISSIVE_MAP,
        });

        let error_tex: [u8; 16] = [
            255, 20, 147, 255, // Pixel 1: R, G, B, A
            255, 20, 147, 255, // Pixel 2: R, G, B, A
            255, 20, 147, 255, // Pixel 3: R, G, B, A
            255, 20, 147, 255, // Pixel 4: R, G, B, A
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
            base_color_factor: Vec4::new(1.0, 1.0, 1.0, 1.0),
            base_color_tex_id: 2,
            metallic_factor: 0.0,
            roughness_factor: 1.0,
            metallic_roughness_tex_id: 1,
            alpha_mode: gpu_data::AlphaMode::Opaque,
            alpha_cutoff: 0.5,
            normal_map: Self::DEFAULT_NORMAL_MAP,
            occlusion_map: Self::DEFAULT_OCCLUSION_MAP,
            emissive_map: Self::DEFAULT_EMISSIVE_MAP,
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
            image_descriptors,
            linear_sampler,
        }
    }

    pub fn is_supported_format(&self, format: vk::Format) -> bool {
        self.supported_formats.contains(&format)
    }

    pub fn add_texture(&mut self, mut data: TextureMeta) -> u32 {
        let index = self.cached_textures.len();

        // TODO roughness textures can actually just be R8 UNORM to save space and bandwidth
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

    pub fn add_textures(&mut self, data: Vec<(u32, TextureMeta)>) -> HashMap<u32, u32> {
        let mut index = self.cached_textures.len() as u32;
        let mut index_pairs = HashMap::<u32, u32>::with_capacity(data.len());

        for (ext_idx, data) in data {
            index_pairs.insert(ext_idx, index);
            index += 1;
            self.cached_textures.push(CachedTexture::Unloaded(data))
        }
        index_pairs
    }

    pub fn add_material(&mut self, mut data: MaterialMeta) -> u32 {
        let index = self.cached_materials.len();
        self.cached_materials.push(CachedMaterial::Unloaded(data));
        index as u32
    }

    pub fn get_material(&self, id: u32) -> Option<&CachedMaterial> {
        self.cached_materials.get(id as usize)
    }

    pub fn get_material_unchecked(&self, id: u32) -> &CachedMaterial {
        unsafe { self.cached_materials.get_unchecked(id as usize) }
    }

    pub fn get_loaded_material_unchecked(&self, id: u32) -> &VkLoadedMaterial {
        unsafe {
            match self.cached_materials.get_unchecked(id as usize) {
                CachedMaterial::Loaded(loaded) => loaded,
                _ => std::hint::unreachable_unchecked(),
            }
        }
    }

    pub fn get_loaded_material_unchecked_ptr(&self, id: u32) -> *const VkLoadedMaterial {
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

    pub fn allocate_texture(
        &mut self,
        device: &ash::Device,
        allocator: Arc<Mutex<Allocator>>,
        immediate: &VkImmediate,
        sampler_cache: &mut SamplerCache,
        tex_id: u32,
    ) {
        let tex_id = tex_id as usize;
        let texture = std::mem::replace(
            &mut self.cached_textures[tex_id],
            CachedTexture::Unloaded(TextureMeta::default()),
        );

        if let CachedTexture::Unloaded(meta) = texture {
            let size = vk::Extent3D {
                width: meta.width,
                height: meta.height,
                depth: 1,
            };

            // fixme, sampler needs to be stored for binding to material shader descriptor
            let (alloc, sampler) = vk_util::upload_image(
                device,
                allocator,
                immediate,
                sampler_cache,
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
            info!(
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
        let material = std::mem::replace(&mut self.cached_materials[mat_id], Self::PLACEHOLDER_MAT);

        if let CachedMaterial::Unloaded(meta) = material {
            let loaded_material: VkLoadedMaterial;
            // let shader_consts = MetRoughUniformExt {
            //     color_factors: meta.base_color_factor,
            //     metal_rough_factors: vec4(
            //         meta.metallic_factor,
            //         meta.roughness_factor,
            //         0.0,
            //         0.0,
            //     ),
            //     normal_scale: vec4(meta.normal_map.scale, 0.0, 0.0, 0.0),
            //     occlusion_strength: vec4(meta.occlusion_map.strength, 0.0, 0.0, 0.0),
            //     emissive_factor: meta.emissive_map.factor.extend(0.0),
            //     extra: [Vec4::ZERO; 11],
            // };
            //
            // let const_bytes = bytemuck::bytes_of(&shader_consts);
            //
            // let uniform_buffer = vk_util::allocate_and_write_buffer(
            //     &allocator.lock().unwrap(),
            //     const_bytes,
            //     vk::BufferUsageFlags::UNIFORM_BUFFER,
            // )
            //     .unwrap();
            //
            // loaded_material =
            //     self.write_material_ext(meta, uniform_buffer, 0, device, desc_layout_cache);

            let shader_consts = MetRoughUniform {
                color_factors: meta.base_color_factor,
                metal_rough_factors: vec4(meta.metallic_factor, meta.roughness_factor, 0.0, 0.0),
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

            self.cached_materials[mat_id] = CachedMaterial::Loaded(loaded_material);
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
        let mut writer = VkDescriptorWriter::default();

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
            self.linear_sampler,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        );

        writer.write_image(
            2,
            metallic_tex.alloc.image_view,
            self.linear_sampler,
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
        let normal_tex = self.get_loaded_texture_unchecked(meta.normal_map.texture_id);
        let occlusion_tex = self.get_loaded_texture_unchecked(meta.occlusion_map.texture_id);
        let emissive_tex = self.get_loaded_texture_unchecked(meta.emissive_map.texture_id);

        let pipeline = match meta.alpha_mode {
            AlphaMode::Opaque => VkPipelineType::PbrMetRoughOpaqueExt,
            AlphaMode::Mask | AlphaMode::Blend => VkPipelineType::PbrMetRoughAlphaExt,
        };
        let mut writer = VkDescriptorWriter::default();

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
            self.linear_sampler,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        );

        writer.write_image(
            2,
            metallic_tex.alloc.image_view,
            self.linear_sampler,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        );

        writer.write_image(
            3,
            normal_tex.alloc.image_view,
            self.linear_sampler,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        );

        writer.write_image(
            4,
            occlusion_tex.alloc.image_view,
            self.linear_sampler,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        );

        writer.write_image(
            5,
            emissive_tex.alloc.image_view,
            self.linear_sampler,
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
        let material = std::mem::replace(&mut self.cached_materials[mat_id], Self::PLACEHOLDER_MAT);

        if let CachedMaterial::Loaded(loaded_material) = material {
            vk_util::destroy_buffer(allocator, loaded_material.uniform_buffer);
            let unloaded_material = CachedMaterial::Unloaded(loaded_material.meta);
            self.cached_materials[mat_id] = unloaded_material;
        } else {
            self.cached_materials[mat_id] = material;
        }
    }

    pub fn deallocate_texture(&mut self, allocator: &vk_mem::Allocator, tex_id: u32) {
        let tex_id = tex_id as usize;
        let texture = std::mem::replace(
            &mut self.cached_textures[tex_id],
            CachedTexture::Unloaded(TextureMeta::default()),
        );

        if let CachedTexture::Loaded(loaded_texture) = texture {
            vk_util::destroy_image(allocator, loaded_texture.alloc);
            let unloaded_texture = CachedTexture::Unloaded(loaded_texture.meta);
            self.cached_textures[tex_id] = unloaded_texture;
        } else {
            self.cached_textures[tex_id] = texture;
        }
    }

    pub fn allocate_all(
        &mut self,
        device: &ash::Device,
        allocator: Arc<Mutex<Allocator>>,
        immediate: &VkImmediate,
        sampler_cache: &mut SamplerCache,
        desc_layout_cache: &VkDescLayoutCache,
    ) {
        for x in 0..self.cached_textures.len() {
            self.allocate_texture(device, allocator.clone(),immediate,sampler_cache, x as u32)
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
            if let CachedMaterial::Loaded(mut loaded) = mat {
                loaded.uniform_buffer.destroy(device, allocator);
            }
        }

        self.image_descriptors
            .iter_mut()
            .for_each(|desc| desc.destroy(device, allocator));
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

#[derive(Debug)]
pub struct MeshCache {
    cached_meshes: Vec<CachedMesh>,
    cached_surface: Vec<SurfaceMeta>,
}

impl Default for MeshCache {
    fn default() -> Self {
        use glam::{Vec3, Vec4};

        let vertices = vec![
            // Front face
            Vertex {
                position: Vec3::new(-1.0, -1.0, 1.0),
                uv0_x: 0.0,
                normal: Vec3::Z,
                uv0_y: 0.0,
                color: Vec4::ONE,
                tangent: Vec4::X,
                ..Default::default()
            },
            Vertex {
                position: Vec3::new(1.0, -1.0, 1.0),
                uv0_x: 1.0,
                normal: Vec3::Z,
                uv0_y: 0.0,
                color: Vec4::ONE,
                tangent: Vec4::X,
                ..Default::default()
            },
            Vertex {
                position: Vec3::new(1.0, 1.0, 1.0),
                uv0_x: 1.0,
                normal: Vec3::Z,
                uv0_y: 1.0,
                color: Vec4::ONE,
                tangent: Vec4::X,
                ..Default::default()
            },
            Vertex {
                position: Vec3::new(-1.0, 1.0, 1.0),
                uv0_x: 0.0,
                normal: Vec3::Z,
                uv0_y: 1.0,
                color: Vec4::ONE,
                tangent: Vec4::X,
                ..Default::default()
            },
            // Back face
            Vertex {
                position: Vec3::new(-1.0, -1.0, -1.0),
                uv0_x: 1.0,
                normal: -Vec3::Z,
                uv0_y: 0.0,
                color: Vec4::ONE,
                tangent: -Vec4::X,
                ..Default::default()
            },
            Vertex {
                position: Vec3::new(-1.0, 1.0, -1.0),
                uv0_x: 1.0,
                normal: -Vec3::Z,
                uv0_y: 1.0,
                color: Vec4::ONE,
                tangent: -Vec4::X,
                ..Default::default()
            },
            Vertex {
                position: Vec3::new(1.0, 1.0, -1.0),
                uv0_x: 0.0,
                normal: -Vec3::Z,
                uv0_y: 1.0,
                color: Vec4::ONE,
                tangent: -Vec4::X,
                ..Default::default()
            },
            Vertex {
                position: Vec3::new(1.0, -1.0, -1.0),
                uv0_x: 0.0,
                normal: -Vec3::Z,
                uv0_y: 0.0,
                color: Vec4::ONE,
                tangent: -Vec4::X,
                ..Default::default()
            },
            // Top face
            Vertex {
                position: Vec3::new(-1.0, 1.0, -1.0),
                uv0_x: 0.0,
                normal: Vec3::Y,
                uv0_y: 1.0,
                color: Vec4::ONE,
                tangent: Vec4::X,
                ..Default::default()
            },
            Vertex {
                position: Vec3::new(-1.0, 1.0, 1.0),
                uv0_x: 0.0,
                normal: Vec3::Y,
                uv0_y: 0.0,
                color: Vec4::ONE,
                tangent: Vec4::X,
                ..Default::default()
            },
            Vertex {
                position: Vec3::new(1.0, 1.0, 1.0),
                uv0_x: 1.0,
                normal: Vec3::Y,
                uv0_y: 0.0,
                color: Vec4::ONE,
                tangent: Vec4::X,
                ..Default::default()
            },
            Vertex {
                position: Vec3::new(1.0, 1.0, -1.0),
                uv0_x: 1.0,
                normal: Vec3::Y,
                uv0_y: 1.0,
                color: Vec4::ONE,
                tangent: Vec4::X,
                ..Default::default()
            },
            // Bottom face
            Vertex {
                position: Vec3::new(-1.0, -1.0, -1.0),
                uv0_x: 0.0,
                normal: -Vec3::Y,
                uv0_y: 0.0,
                color: Vec4::ONE,
                tangent: Vec4::X,
                ..Default::default()
            },
            Vertex {
                position: Vec3::new(1.0, -1.0, -1.0),
                uv0_x: 1.0,
                normal: -Vec3::Y,
                uv0_y: 0.0,
                color: Vec4::ONE,
                tangent: Vec4::X,
                ..Default::default()
            },
            Vertex {
                position: Vec3::new(1.0, -1.0, 1.0),
                uv0_x: 1.0,
                normal: -Vec3::Y,
                uv0_y: 1.0,
                color: Vec4::ONE,
                tangent: Vec4::X,
                ..Default::default()
            },
            Vertex {
                position: Vec3::new(-1.0, -1.0, 1.0),
                uv0_x: 0.0,
                normal: -Vec3::Y,
                uv0_y: 1.0,
                color: Vec4::ONE,
                tangent: Vec4::X,
                ..Default::default()
            },
            // Right face
            Vertex {
                position: Vec3::new(1.0, -1.0, -1.0),
                uv0_x: 1.0,
                normal: Vec3::X,
                uv0_y: 0.0,
                color: Vec4::ONE,
                tangent: -Vec4::Z,
                ..Default::default()
            },
            Vertex {
                position: Vec3::new(1.0, 1.0, -1.0),
                uv0_x: 1.0,
                normal: Vec3::X,
                uv0_y: 1.0,
                color: Vec4::ONE,
                tangent: -Vec4::Z,
                ..Default::default()
            },
            Vertex {
                position: Vec3::new(1.0, 1.0, 1.0),
                uv0_x: 0.0,
                normal: Vec3::X,
                uv0_y: 1.0,
                color: Vec4::ONE,
                tangent: -Vec4::Z,
                ..Default::default()
            },
            Vertex {
                position: Vec3::new(1.0, -1.0, 1.0),
                uv0_x: 0.0,
                normal: Vec3::X,
                uv0_y: 0.0,
                color: Vec4::ONE,
                tangent: -Vec4::Z,
                ..Default::default()
            },
            // Left face
            Vertex {
                position: Vec3::new(-1.0, -1.0, -1.0),
                uv0_x: 0.0,
                normal: -Vec3::X,
                uv0_y: 0.0,
                color: Vec4::ONE,
                tangent: Vec4::Z,
                ..Default::default()
            },
            Vertex {
                position: Vec3::new(-1.0, -1.0, 1.0),
                uv0_x: 1.0,
                normal: -Vec3::X,
                uv0_y: 0.0,
                color: Vec4::ONE,
                tangent: Vec4::Z,
                ..Default::default()
            },
            Vertex {
                position: Vec3::new(-1.0, 1.0, 1.0),
                uv0_x: 1.0,
                normal: -Vec3::X,
                uv0_y: 1.0,
                color: Vec4::ONE,
                tangent: Vec4::Z,
                ..Default::default()
            },
            Vertex {
                position: Vec3::new(-1.0, 1.0, -1.0),
                uv0_x: 0.0,
                normal: -Vec3::X,
                uv0_y: 1.0,
                color: Vec4::ONE,
                tangent: Vec4::Z,
                ..Default::default()
            },
        ];

        let indices = vec![
            0, 1, 2, 2, 3, 0, // front
            4, 5, 6, 6, 7, 4, // back
            8, 9, 10, 10, 11, 8, // top
            12, 13, 14, 14, 15, 12, // bottom
            16, 17, 18, 18, 19, 16, // right
            20, 21, 22, 22, 23, 20, // left
        ];
        let mut cached_meshes = Vec::<CachedMesh>::with_capacity(100);

        let skybox = MeshMeta {
            name: "Skybox Cube".to_string(),
            indices,
            vertices,
            material_index: None,
        };

        cached_meshes.push(CachedMesh::UnLoaded(skybox));

        Self {
            cached_meshes,
            cached_surface: vec![],
        }
    }
}

impl MeshCache {
    pub const SKYBOX_MESH: u32 = 0;

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

    pub fn allocate_mesh(
        &mut self,
        device: &ash::Device,
        allocator: Arc<Mutex<Allocator>>,
        immediate: &VkImmediate,
        mesh_id: usize,
    ) {
        let mesh = std::mem::replace(
            &mut self.cached_meshes[mesh_id],
            CachedMesh::UnLoaded(MeshMeta::default()),
        );

        if let CachedMesh::UnLoaded(meta) = mesh {
            let buffer =
                vk_util::upload_mesh(device, allocator, immediate, &meta.indices, &meta.vertices);

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

    pub fn allocate_all(
        &mut self,
        device: &ash::Device,
        allocator: Arc<Mutex<Allocator>>,
        immediate: &VkImmediate,
    ) {
        for x in 0..self.cached_meshes.len() {
            self.allocate_mesh(device, allocator.clone(), immediate, x);
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
    BrtFlutVert,
    BrtFlutFrag,
    SkyBoxVert,
    SkyBoxFrag,
    CubeFilterVert,
    EnvIrradianceFrag,
    EnvPrefilterFrag,
}

impl CoreShaderType {
    const COUNT: usize = 11;
}

pub struct VkShaderCache {
    pub core_shader_cache: [vk::ShaderModule; CoreShaderType::COUNT],
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
    PbrMetRoughOpaqueExt,
    PbrMetRoughAlphaExt,
    BrdfLut,
    Skybox,
    EnvPreFilter,
    EnvIrradiance,
}

impl VkPipelineType {
    const COUNT: usize = 8;
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
    GpuScene,
    PbrMetRough,
    PbrMetRoughExt,
    Skybox,
    Empty,
    EnvIrradiance,
    EnvPreFilter,
}

impl VkDescType {
    const COUNT: usize = 8;
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
        pipeline: vk::Pipeline,
        cmd_pool: &VkCommandPool,
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
            let cube_map = vk_util::upload_skybox(device, allocator, meta, pipeline, cmd_pool);
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

pub struct SamplerCache {
    pub samplers: HashMap<VkSamplerInfo, vk::Sampler>,
}

impl Default for SamplerCache {
    fn default() -> Self {
        Self {
            samplers: HashMap::with_capacity(20),
        }
    }
}

impl SamplerCache {
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
