use crate::data::gpu_data;
use crate::data::gpu_data::{
    MaterialMeta, MeshMeta, SurfaceMeta, TextureMeta, Vertex, VkGpuMeshBuffers, VkGpuTextureBuffer,
};
use crate::vulkan::vk_types::{VkImageAlloc, VkPipeline};
use crate::vulkan::vk_util;
use ash::vk;
use glam::Vec4;
use std::collections::HashMap;

pub const EXTENT3D_ONE: vk::Extent3D = vk::Extent3D {
    width: 1,
    height: 1,
    depth: 1,
};

pub trait PackUnorm {
    fn pack_unorm_4x8(&self) -> u32;
}

impl PackUnorm for Vec4 {
    fn pack_unorm_4x8(&self) -> u32 {
        let x = (self.x.clamp(0.0, 1.0) * 255.0).round() as u32;
        let y = (self.y.clamp(0.0, 1.0) * 255.0).round() as u32;
        let z = (self.z.clamp(0.0, 1.0) * 255.0).round() as u32;
        let w = (self.w.clamp(0.0, 1.0) * 255.0).round() as u32;

        (x << 0) | (y << 8) | (z << 16) | (w << 24)
    }
}

#[derive(Debug)]
pub enum CachedTexture {
    UnLoaded(TextureMeta),
    Loaded(VkLoadedTexture),
}

#[derive(Debug)]
pub struct VkLoadedTexture {
    pub meta: TextureMeta,
    pub alloc: VkImageAlloc,
}

#[derive(Debug)]
pub struct TextureCache {
    cached_textures: Vec<CachedTexture>,
    cached_materials: Vec<MaterialMeta>,
}

impl Default for TextureCache {
    fn default() -> Self {
        let def_color = CachedTexture::UnLoaded(TextureMeta {
            bytes: vec![255, 255, 255, 255],
            width: 1,
            height: 1,
            format: vk::Format::R8G8B8A8_UNORM,
            mips_levels: 1,
        });

        let def_rough = CachedTexture::UnLoaded(TextureMeta {
            bytes: vec![128, 128, 128, 255],
            width: 1,
            height: 1,
            format: vk::Format::R8G8B8A8_UNORM,
            mips_levels: 1,
        });

        let def_mat = MaterialMeta {
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
        };

        let mut cached_textures = Vec::with_capacity(100);
        cached_textures.push(def_color);
        cached_textures.push(def_rough);

        let mut cached_materials = Vec::with_capacity(100);
        cached_materials.push(def_mat);

        Self {
            cached_textures,
            cached_materials,
        }
    }
}

impl TextureCache {
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
        self.cached_materials.push(data);
        index as u32
    }

    pub fn get_material(&self, id: usize) -> Option<&MaterialMeta> {
        self.cached_materials.get(id)
    }

    pub fn get_material_unchecked(&self, id: usize) -> &MaterialMeta {
        unsafe { self.cached_materials.get_unchecked(id) }
    }

    pub fn get_texture(&self, id: u32) -> Option<&CachedTexture> {
        self.cached_textures.get(id as usize)
    }

    pub fn get_texture_unchecked(&self, id: u32) -> &CachedTexture {
        unsafe { self.cached_textures.get_unchecked(id as usize) }
    }

    pub fn is_texture_loaded(&self, id: u32) -> bool {
        if let Some(found) = self.cached_textures.get(id as usize) {
            matches!(found, CachedTexture::Loaded(_))
        } else {
            false
        }
    }

    pub fn allocate_texture<F>(&mut self, mut upload_fn: F, tex_id: usize)
    where
        F: FnMut(&[u8], vk::Extent3D, vk::Format, vk::ImageUsageFlags, bool) -> VkImageAlloc,
    {
        let texture = std::mem::replace(
            &mut self.cached_textures[tex_id],
            CachedTexture::UnLoaded(TextureMeta {
                bytes: vec![],
                width: 0,
                height: 0,
                format: vk::Format::UNDEFINED,
                mips_levels: 0,
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

    pub fn deallocate_texture(&mut self, allocator: &vk_mem::Allocator, tex_id: usize) {
        let texture = std::mem::replace(
            &mut self.cached_textures[tex_id],
            CachedTexture::UnLoaded(TextureMeta {
                bytes: vec![],
                width: 0,
                height: 0,
                format: vk::Format::UNDEFINED,
                mips_levels: 0,
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

    pub fn allocate_all<F>(&mut self, mut upload_fn: F)
    where
        F: FnMut(&[u8], vk::Extent3D, vk::Format, vk::ImageUsageFlags, bool) -> VkImageAlloc,
    {
        for x in 0..self.cached_textures.len() {
            self.allocate_texture(&mut upload_fn, x);
        }
    }

    pub fn deallocate_all(&mut self, allocator: &vk_mem::Allocator) {
        for x in 0..self.cached_textures.len() {
            self.deallocate_texture(allocator, x)
        }
    }

    // pub fn get_loaded_texture_unchecked(&self, id : u32) -> &
}

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
        F: FnMut(&[u32], &[Vertex]) -> VkGpuMeshBuffers,
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

    pub fn allocate_all<F>(&mut self, mut upload_fn: F)
    where
        F: FnMut(&[u32], &[Vertex]) -> VkGpuMeshBuffers,
    {
        for x in 0..self.cached_meshes.len() {
            self.allocate_mesh(&mut upload_fn, x);
        }
    }

    pub fn deallocate_all(&mut self, allocator: &vk_mem::Allocator) {
        for x in 0..self.cached_meshes.len() {
            self.deallocate_mesh(allocator, x)
        }
    }
}
