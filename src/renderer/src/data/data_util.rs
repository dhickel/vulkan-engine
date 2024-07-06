use crate::data::gpu_data::{MaterialMeta, MeshMeta, SurfaceMeta, TextureMeta, VkGpuMeshBuffers};
use ash::vk;
use glam::Vec4;
use std::collections::HashMap;
use crate::vulkan::vk_types::VkPipeline;


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
    Loaded,
}

pub struct LoadedTexture {
    pub meta: TextureMeta,

}

#[derive(Default, Debug)]
pub struct TextureCache {
    cached_textures: Vec<CachedTexture>,
    cached_materials: Vec<MaterialMeta>
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

    pub fn get_material(&self, id : u32) -> Option<&MaterialMeta> {
        self.cached_materials.get(id as usize)
    }

    pub fn get_material_unchecked(&self, id: u32) -> &MaterialMeta {
        unsafe {self.cached_materials.get_unchecked(id as usize)}
    }

    pub fn get_texture(&self, id: u32) -> Option<&CachedTexture> {
        self.cached_textures.get(id as usize)
    }

    pub fn get_texture_unchecked(&self, id: u32) -> &CachedTexture {
        unsafe { self.cached_textures.get_unchecked(id as usize) }
    }

    pub fn is_texture_loaded(&self, id: u32) -> bool {
        if let Some(found) = self.cached_textures.get(id as usize) {
            matches!(found, CachedTexture::Loaded)
        } else {
            false
        }
    }

    pub fn get_loaded_texture_unchecked(&self, id : u32) -> &
}


#[derive(Debug)]
pub enum CachedMesh {
    UnLoaded(MeshMeta),
    Loaded(VkLoadedMesh),
}

#[derive( Debug)]
pub struct VkLoadedMesh {
    pub meta: MeshMeta,
    pub buffer: VkGpuMeshBuffers,
    pub pipeline: VkPipeline
}

#[derive(Default, Debug)]
pub struct MeshCache {
    cached_meshes: Vec<CachedMesh>,
    cached_surface: Vec<SurfaceMeta>
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

    pub fn get_loaded_mesh_unchecked(&self, id : u32) -> &VkLoadedMesh {
        unsafe {
            match self.cached_meshes.get_unchecked(id as usize) {
                CachedMesh::Loaded(loaded) => loaded,
                _ => std::hint::unreachable_unchecked(),
            }
        }
    }
}
