use crate::data::data_cache::{
    CoreShaderType, MeshCache, TextureCache, VkLoadedMaterial, VkPipelineType, VkShaderCache,
};
use crate::data::gltf_util;
use crate::vulkan::vk_descriptor::{
    DescriptorLayoutBuilder, DescriptorWriter, VkDescWriterType, VkDynamicDescriptorAllocator,
};
use crate::vulkan::vk_pipeline::PipelineBuilder;
use crate::vulkan::vk_render::VkRender;
use crate::vulkan::vk_types::{LogicalDevice, VkBuffer, VkDescriptors, VkImageAlloc, VkPipeline};
use crate::vulkan::vk_util;
use ash::vk;
use ash::vk::DescriptorSet;
use bytemuck::{Pod, Zeroable};
use glam::{vec4, Mat4, Quat, Vec2, Vec3, Vec4};
use imgui::sys::igSetClipboardText;
use std::cell::{Ref, RefCell};
use std::cmp::PartialEq;
use std::collections::HashSet;
use std::f32::consts::PI;
use std::ffi::{CStr, CString};
use std::rc::{Rc, Weak};

//////////////////////////
//  MESH & TEXTURE DATA //
//////////////////////////

// Used In shaders as well
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Vertex {
    pub position: Vec3,
    pub uv_x: f32, // pad with uv_x in between to optimize
    pub normal: Vec3,
    pub uv_y: f32,
    pub color: Vec4,
    pub tangent: Vec4,
}

impl Default for Vertex {
    fn default() -> Self {
        Self {
            position: Default::default(),
            normal: Default::default(),
            color: Default::default(),
            tangent: Default::default(),
            uv_x: 0.0,
            uv_y: 0.0,
        }
    }
}

#[derive(Copy, Clone, PartialEq, Debug)]
pub enum AlphaMode {
    Opaque,
    Mask,
    Blend,
}



#[derive(Copy, Clone, PartialEq)]
pub enum PbrTexture {
    MetallicRough(PbrMetallicRoughness),
    SpecularGloss(PbrSpecularGlossiness),
    Transmission(PbrTransmission),
}

#[derive(Copy, Clone, PartialEq)]
pub struct PbrMetallicRoughness {
    pub base_color_factor: Vec4,
    pub base_color_tex_id: u32,
    pub metallic_factor: f32,
    pub roughness_factor: f32,
    pub metallic_roughness_tex_id: u32,
}

#[derive(Copy, Clone, PartialEq)]
pub struct PbrSpecularGlossiness {
    pub diffuse_factor: Vec4,
    pub diffuse_tex_idx: u32,
    pub specular_factor: Vec3,
    pub glossiness_factor: f32,
    pub specular_glossiness_tex_id: u32,
}

#[derive(Copy, Clone, PartialEq)]
pub struct PbrTransmission {
    pub transmission_factor: f32,
    pub transmission_tex_id: u32,
}

#[derive(Copy, Clone, PartialEq)]
pub struct SpecularMap {
    pub specular_factor: f32,
    pub specular_tex_id: u32,
    pub specular_color_factor: Vec3,
    pub specular_color_tex_id: u32,
}

#[derive(Copy, Clone, PartialEq, Debug)]
pub struct EmissiveMap {
    pub factor: Vec3,
    pub texture_id: u32,
}

#[derive(Copy, Clone, PartialEq, Debug)]
pub struct NormalMap {
    pub scale: f32,
    pub texture_id: u32,
}

#[derive(Copy, Clone, PartialEq, Debug)]
pub struct OcclusionMap {
    pub strength: f32,
    pub texture_id: u32,
}

#[derive(Copy, Clone, PartialEq, Debug)]
pub struct VolumeMap {
    pub thickness_factor: f32,
    pub thickness_tex_id: u32,
    pub attenuation_distance: f32,
    pub attenuation_color: Vec3,
}

// MESH & TEXTURE METADATA

#[derive(Copy, Clone, PartialEq, Debug)]
pub struct MaterialMeta {
    pub base_color_factor: Vec4,
    pub base_color_tex_id: u32,
    pub metallic_factor: f32,
    pub roughness_factor: f32,
    pub metallic_roughness_tex_id: u32,
    pub alpha_mode: AlphaMode,
    pub alpha_cutoff: f32,
    pub normal_map: Option<NormalMap>,
    pub occlusion_map: Option<OcclusionMap>,
    pub emissive_map: Option<EmissiveMap>,
}

impl MaterialMeta {
    pub fn has_normal(&self) -> bool {
        self.normal_map.is_some()
    }

    pub fn has_occlusion(&self) -> bool {
        self.occlusion_map.is_some()
    }

    pub fn is_ext(&self) -> bool {
        self.normal_map.is_some() || self.occlusion_map.is_some() || self.emissive_map.is_some()
    }
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Sampler {
    Linear,
    Nearest,
}

#[derive(Clone, PartialEq, Debug)]
pub struct TextureMeta {
    pub bytes: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub format: vk::Format,
    pub mips_levels: u32,
    pub sampler: Sampler,
}

pub struct VkCubeMap {
    pub texture_meta: TextureMeta,
    pub full_extent: vk::Extent3D,
    pub face_extent: vk::Extent3D,
    pub allocation: vk_mem::Allocation,
    pub image : vk::Image,
    pub image_view : vk::ImageView,
    pub sampler : vk::Sampler
}

impl TextureMeta {
    // pub fn from_gltf_texture(data: &gltf::image::Data) -> Self {
    //     Self {
    //         bytes: data.pixels.clone(),
    //         width: data.width,
    //         height: data.height,
    //         format: gltf_util::gltf_format_to_vk_format(data.format),
    //         mips_levels: 1,
    //         sampler: Sampler::Linear,
    //     }
    // }
}

#[derive(Copy, Clone, PartialEq, Debug)]
pub struct SurfaceMeta {
    pub start_index: u32,
    pub count: u32,
    pub material_index: Option<u32>,
}

#[derive(Clone, PartialEq, Debug)]
pub struct MeshMeta {
    pub name: String,
    pub indices: Vec<u32>,
    pub vertices: Vec<Vertex>,
    pub material_index: Option<u32>,
}

#[derive(Clone, PartialEq)]
pub struct NodeMeta {
    pub name: String,
    pub mesh_indices: Vec<u32>,
    pub local_transform: Transform,
    pub og_matrix: Mat4,
    pub children: Vec<u32>,
}

/////////////////
// SHADER DATA //
/////////////////

// VERTEX - See top of file

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct MetRoughUniform {
    pub color_factors: Vec4,
    pub metal_rough_factors: Vec4,
    //padding
    pub extra: [Vec4; 14],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct MetRoughUniformExt {
    pub color_factors: Vec4,
    pub metal_rough_factors: Vec4,
    pub normal_scale: Vec4,
    pub occlusion_strength: Vec4,
    pub emissive_factor: Vec4,
    //padding
    pub extra: [Vec4; 11],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct VkGpuPushConsts {
    pub world_matrix: [[f32; 4]; 4],
    pub vertex_buffer_addr: vk::DeviceAddress,
}

impl VkGpuPushConsts {
    pub fn new(world_matrix: glam::Mat4, vertex_buffer_addr: vk::DeviceAddress) -> Self {
        Self {
            world_matrix: world_matrix.to_cols_array_2d(),
            vertex_buffer_addr,
        }
    }

    pub fn as_byte_slice(&self) -> &[u8] {
        bytemuck::bytes_of(self)
        // unsafe {
        //     let ptr = self as *const VkGpuPushConsts as *const u8;
        //     slice::from_raw_parts(ptr, mem::size_of::<VkGpuPushConsts>())
        // }
    }
}

#[repr(C)]
#[derive(Default, Copy, Clone, Pod, Zeroable)]
pub struct GPUSceneData {
    pub view: Mat4,
    pub projection: Mat4,
    pub view_projection: Mat4,
    pub ambient_color: Vec4,
    pub sunlight_direction: Vec4,
    pub sunlight_color: Vec4,
}

impl GPUSceneData {
    pub fn as_byte_slice(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
}


#[repr(C)]
#[derive(Default, Copy, Clone, Pod, Zeroable)]
pub struct UBOMatrices {
    pub projection: Mat4,
    pub model: Mat4,
    pub view: Mat4,
    pub cam_pos: Vec3,
    pad: f32
}


#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct SkyboxPushConstants {
    pub projection: Mat4,
    pub vertex_buffer_addr: vk::DeviceAddress,
    pub exposure: f32,
    pub gamma: f32,

}

impl Default for SkyboxPushConstants {
    fn default() -> Self {
        Self {
            projection: Default::default(),
            vertex_buffer_addr: vk::DeviceAddress::default(),
            exposure: 4.5,
            gamma: 2.2,
        }
    }
}


////////////////////////////
// VULKAN ALLOCATION DATA //
////////////////////////////

#[derive(Debug)]
pub struct VkGpuMeshBuffers {
    pub index_buffer: VkBuffer,
    pub vertex_buffer: VkBuffer,
    pub vertex_buffer_addr: vk::DeviceAddress,
}

#[derive(Debug)]
pub struct VkGpuTextureBuffer {
    pub image_alloc: VkImageAlloc,
    pub data_buffer: vk::Buffer,
    pub data_buffer_offset: u32,
}

#[derive(Debug)]
pub struct VkGpuMetRoughBuffer {
    pub color_image: VkImageAlloc,
    pub metal_rough_image: VkImageAlloc,
    pub data_buffer: vk::Buffer,
    pub data_buffer_offset: u32,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct VkMetRoughUniforms {
    pub color_factors: Vec4,
    pub metal_rough_factors: Vec4,
    pub extra: [Vec4; 14],
}

/////////////////////////////
// SCENE GRAPH & RENDERING //
/////////////////////////////

#[derive(Debug, Copy, Clone)]
pub struct RenderObject {
    pub index_count: u32,
    pub first_index: u32,
    pub index_buffer: vk::Buffer,
    pub material: *const VkLoadedMaterial,
    pub transform: Mat4,
    pub vertex_buffer_addr: vk::DeviceAddress,
}

// pub trait Renderable {
//     fn draw(&self, top_matrix: &Mat4, context: &mut DrawContext);
//     fn refresh_transform(&mut self);
//     fn get_children(&self) -> &Vec<Rc<RefCell<Node>>>;
// }
#[derive(Debug, Copy, Clone, Default, PartialEq)]
pub struct Transform {
    pub position: Vec3,
    pub scale: Vec3,
    pub rotation: Quat,
}

impl Transform {
    pub fn compose(&mut self) -> Mat4 {
        glam::Mat4::from_scale_rotation_translation(self.scale, self.rotation, self.position)
    }

    pub fn new_vulkan_adjusted(translation: [f32; 3], rotation: [f32; 4], scale: [f32; 3]) -> Self {
        Transform {
            position: glam::Vec3::from_array(translation),
            scale: glam::Vec3::from_array(scale),
            rotation: glam::Quat::from_array(rotation),
        }
    }
}

#[derive(Debug)]
pub struct Node {
    pub parent: Option<Weak<RefCell<Node>>>,
    pub children: Vec<Rc<RefCell<Node>>>,
    pub meshes: Vec<u32>,
    pub world_transform: Mat4,
    pub local_transform: Mat4,
    pub dirty: bool,
}

impl Default for Node {
    fn default() -> Self {
        Self {
            parent: None,
            children: vec![],
            meshes: vec![],
            world_transform: Mat4::IDENTITY,
            local_transform: Mat4::IDENTITY,
            dirty: true,
        }
    }
}

impl Node {
    pub(crate) fn draw(
        &mut self,
        top_matrix: &Mat4,
        ctx: &mut DrawContext,
        mesh_cache: &MeshCache,
        tex_cache: &TextureCache,
    ) {
        if self.dirty {
            self.refresh_transform(*top_matrix);
        }

        for mesh_id in &self.meshes {
            let mesh = mesh_cache.get_loaded_mesh_unchecked(*mesh_id);

            if let Some(id) = mesh.meta.material_index {
                let material = tex_cache.get_loaded_material_unchecked(id);
                let material_ptr = material as *const VkLoadedMaterial;
                let ro = RenderObject {
                    index_count: mesh.meta.indices.len() as u32,
                    first_index: 0,
                    index_buffer: mesh.buffer.index_buffer.buffer,
                    material: material_ptr,
                    transform: self.world_transform,
                    vertex_buffer_addr: mesh.buffer.vertex_buffer_addr,
                };

                // FIXME, should use something more performant than a hash set since all types are known
                ctx.active_pipelines.insert(material.pipeline);

                unsafe {
                    ctx.render_objects
                        .get_unchecked_mut(material.pipeline as usize)
                        .push(ro);
                }
            }
        }

        for child in &self.children {
            child
                .borrow_mut()
                .draw(top_matrix, ctx, mesh_cache, tex_cache);
        }
    }

    pub fn refresh_transform(&mut self, parent_transform: Mat4) {
        self.world_transform = parent_transform.mul_mat4(&self.local_transform);
        self.dirty = false;

        for child in &self.children {
            let mut child = child.borrow_mut();
            child.refresh_transform(self.world_transform);
        }
    }

    fn get_children(&self) -> &Vec<Rc<RefCell<Node>>> {
        &self.children
    }
}

pub struct DrawContext {
    pub active_pipelines: HashSet<VkPipelineType>,
    pub render_objects: [Vec<RenderObject>; 4],
}

impl DrawContext {
    pub fn new(vector_capacity: usize) -> Self {
        DrawContext {
            active_pipelines: HashSet::with_capacity(4),
            render_objects: Self::create_render_object_array(vector_capacity),
        }
    }

    fn create_render_object_array(capacity: usize) -> [Vec<RenderObject>; 4] {
        let vec_iter = std::iter::repeat_with(|| Vec::with_capacity(capacity));
        vec_iter.take(4).collect::<Vec<_>>().try_into().unwrap()
    }

    pub fn clear(&mut self) {
        self.active_pipelines
            .iter()
            .for_each(|pl| self.render_objects[*pl as usize].clear());

        self.active_pipelines.clear();
    }
}

impl Default for DrawContext {
    fn default() -> Self {
        Self::new(400)
    }
}

#[repr(C)]
#[derive(PartialEq, Debug, Copy, Clone)]
pub enum MaterialPass {
    MainColor,
    Transparent,
    Other,
    NULL,
}
