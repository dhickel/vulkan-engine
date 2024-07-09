use crate::data::gpu_data;
use crate::data::gpu_data::{
    AlphaMode, MaterialMeta, MeshMeta, MetRoughShaderConsts, Sampler, SurfaceMeta, TextureMeta,
    Vertex, VkGpuMeshBuffers,
};
use crate::vulkan::vk_descriptor::{
    DescriptorAllocator, DescriptorWriter, VkDynamicDescriptorAllocator,
};
use crate::vulkan::vk_types::{VkBuffer, VkImageAlloc, VkPipeline};
use crate::vulkan::vk_util;
use ash::vk;
use glam::{vec4, Vec4};
use std::collections::HashMap;

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
    pub linear_sampler: vk::Sampler,
    pub nearest_sampler: vk::Sampler,
}

impl TextureCache {
    pub fn new(device: &ash::Device) -> Self {
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

        let mut cached_textures = Vec::with_capacity(100);
        cached_textures.push(def_color);
        cached_textures.push(def_rough);

        let mut cached_materials = Vec::with_capacity(100);
        cached_materials.push(def_mat);

        let mut sampler = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::NEAREST)
            .min_filter(vk::Filter::NEAREST);

        let nearest_sampler = unsafe { device.create_sampler(&sampler, None).unwrap() };

        sampler.mag_filter = vk::Filter::LINEAR;
        sampler.min_filter = vk::Filter::LINEAR;

        let linear_sampler = unsafe { device.create_sampler(&sampler, None).unwrap() };

        Self {
            cached_textures,
            cached_materials,
            linear_sampler,
            nearest_sampler,
        }
    }

    pub fn get_sampler(&self, sampler: Sampler) -> vk::Sampler {
        match sampler {
            Sampler::Linear => self.linear_sampler,
            Sampler::Nearest => self.nearest_sampler,
        }
    }

    pub fn add_texture(&mut self, data: TextureMeta) -> u32 {
        let index = self.cached_textures.len();
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

    pub fn add_material(&mut self, data: MaterialMeta) -> u32 {
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
        allocator: &vk_mem::Allocator,
        desc_allocators: &mut [VkDynamicDescriptorAllocator],
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
            let shader_consts = MetRoughShaderConsts {
                color_factors: meta.base_color_factor,
                metal_rough_factors: vec4(meta.metallic_factor, meta.roughness_factor, 0.0, 0.0),
                extra: [Vec4::ZERO; 14],
            };

            let const_bytes = bytemuck::bytes_of(&shader_consts);

            let uniform_buffer = vk_util::allocate_and_write_buffer(
                allocator,
                const_bytes,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
            )
            .unwrap();

            let loaded_material = self.write_material(
                meta,
                uniform_buffer,
                0,
                device,
                desc_allocators,
                desc_layout_cache,
            );

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
        &self,
        meta: MaterialMeta,
        uniform_buffer: VkBuffer,
        buffer_offset: u32,
        device: &ash::Device,
        desc_allocators: &mut [VkDynamicDescriptorAllocator],
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
            std::mem::size_of::<MetRoughShaderConsts>(),
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

        let layout = [desc_layout_cache.get(VkDescType::PbrMetRough)];

        let descriptors: [vk::DescriptorSet; 2] = desc_allocators
            .iter_mut()
            .map(|desc_alloc| {
                let descriptor = desc_alloc.allocate(device, &layout).unwrap();
                writer.update_set(device, descriptor);
                descriptor
            })
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

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
        mut upload_fn: F,
        device: &ash::Device,
        allocator: &vk_mem::Allocator,
        desc_allocators: &mut [VkDynamicDescriptorAllocator],
        desc_layout_cache: &VkDescLayoutCache,
    ) where
        F: Fn(&[u8], vk::Extent3D, vk::Format, vk::ImageUsageFlags, bool) -> VkImageAlloc,
    {
        for x in 0..self.cached_textures.len() {
            self.allocate_texture(&upload_fn, x as u32);
        }

        for x in 0..self.cached_materials.len() {
            self.allocate_material(
                device,
                allocator,
                desc_allocators,
                desc_layout_cache,
                x as u32,
            );
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
                surfaces: vec![],
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
                surfaces: vec![],
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

    pub fn allocate_all<F>(&mut self,  upload_fn: F)
    where
        F: Fn(&[u32], &[Vertex]) -> VkGpuMeshBuffers,
    {
        for x in 0..self.cached_meshes.len() {
            self.allocate_mesh(& upload_fn, x);
        }
    }

    pub fn deallocate_all(&mut self, allocator: &vk_mem::Allocator) {
        for x in 0..self.cached_meshes.len() {
            self.deallocate_mesh(allocator, x)
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
}

pub struct ShaderCache {
    pub core_shader_cache: [vk::ShaderModule; 2],
    pub user_shader_cache: Vec<vk::ShaderModule>,
}

impl ShaderCache {
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

        let sorted_shaders: [vk::ShaderModule; 2] = compiled_shaders
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

    pub fn destory_all(&mut self, device: &ash::Device) {
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
#[derive(Ord, Eq, PartialEq, PartialOrd, Debug, Clone, Copy)]
pub enum VkPipelineType {
    PbrMetRoughOpaque,
    PbrMetRoughAlpha,
}

//#[derive(Clone, Copy)]
pub struct VkPipelineCache {
    pipelines: Vec<[VkPipeline; 2]>,
}

impl VkPipelineCache {
    pub fn new(mut pipelines: Vec<Vec<(VkPipelineType, VkPipeline)>>) -> Result<Self, String> {
        pipelines.iter_mut().for_each(|p| {
            p.sort_by_key(|(typ, _)| *typ);
        });

        // Convert sorted vectors to fixed-size arrays
        let sorted_pipelines: Vec<[VkPipeline; 2]> = pipelines
            .into_iter()
            .map(|p| {
                p.into_iter()
                    .map(|(_, pipeline)| pipeline)
                    .collect::<Vec<_>>()
                    .try_into()
                    .map_err(|_| "Number of pipelines did not match number of enum keys")
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self {
            pipelines: sorted_pipelines,
        })
    }

    pub fn get_pipeline(&self, index: u32, typ: VkPipelineType) -> &VkPipeline {
        unsafe {
            self.pipelines
                .get_unchecked(index as usize)
                .get_unchecked(typ as usize)
        }
    }

    pub fn get_unchecked(&self, index: usize, typ: VkPipelineType) -> &VkPipeline {
        unsafe {
            self.pipelines
                .get_unchecked(index)
                .get_unchecked(typ as usize)
        }
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
}
pub struct VkDescLayoutCache {
    layouts: [vk::DescriptorSetLayout; 3],
}

impl VkDescLayoutCache {
    pub fn new(mut layouts: Vec<(VkDescType, vk::DescriptorSetLayout)>) -> Self {
        layouts.sort();

        let sorted_layouts: [vk::DescriptorSetLayout; 3] = layouts
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
