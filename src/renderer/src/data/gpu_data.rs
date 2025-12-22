use crate::data::data_cache::{
    CoreShaderType, MeshCache, TextureCache, VkLoadedMaterial, VkPipelineType, VkShaderCache,
};
use crate::data::gltf_util;
use crate::vulkan::vk_descriptor::{
    DescriptorLayoutBuilder, VkDescWriterType, VkDescriptorWriter, VkDynamicDescriptorAllocator,
};
use crate::vulkan::vk_pipeline::PipelineBuilder;
use crate::vulkan::vk_render::VkRender;
use crate::vulkan::vk_types::{LogicalDevice, VkBuffer, VkDescriptors, VkImageAlloc, VkPipeline, VkSubAlloc};
use crate::vulkan::vk_util;
use ash::vk;
use ash::vk::DescriptorSet;
use bytemuck::{Pod, Zeroable};
use glam::{vec4, Mat4, Quat, UVec4, Vec2, Vec3, Vec4};
use imgui::sys::igSetClipboardText;
use std::cell::{Ref, RefCell};
use std::cmp::PartialEq;
use std::collections::HashSet;
use std::f32::consts::PI;
use std::ffi::{CStr, CString};
use std::rc::{Rc, Weak};
use log::debug;
//////////////////////////
//  MESH & TEXTURE DATA //
//////////////////////////

// Used In shaders as well
/// Represents a single vertex in the mesh.
/// Maps directly to the vertex buffer layout expected by the shader.
#[repr(C)]
#[derive(Clone, Default, Copy, Debug, Pod, Zeroable)]
pub struct Vertex {
    pub position: Vec3,
    pub uv0_x: f32,
    pub normal: Vec3,
    pub uv0_y: f32,
    pub color: Vec4,
    pub tangent: Vec4,
    pub joints: UVec4,
    pub weights: Vec4,
    pub uv1_x: f32,
    pub uv1_y: f32,
    pub _pad: u64,
}

/// Alpha blending mode for a material.
#[repr(C)]
#[derive(Copy, Clone, PartialEq, Debug)]
pub enum AlphaMode {
    /// No transparency.
    Opaque = 0,
    /// Alpha blending.
    Blend = 1,
    /// Alpha masking (cutout).
    Mask = 2,
}


impl AlphaMode {
    pub fn to_float_value(&self) -> f32 {
        match self {
            AlphaMode::Opaque => 0.0,
            AlphaMode::Blend => 1.0,
            AlphaMode::Mask => 2.0,
        }
    }
}

// #[derive(Copy, Clone, PartialEq)]
// pub enum PbrTexture {
//     MetallicRough(PbrMetallicRoughness),
//     SpecularGloss(PbrSpecularGlossiness),
//     Transmission(PbrTransmission),
// }
//
// #[derive(Copy, Clone, PartialEq)]
// pub struct PbrSpecularGlossiness {
//     pub diffuse_factor: Vec4,
//     pub diffuse_tex_idx: u32,
//     pub specular_factor: Vec3,
//     pub glossiness_factor: f32,
//     pub specular_glossiness_tex_id: u32,
// }
//
// #[derive(Copy, Clone, PartialEq)]
// pub struct PbrTransmission {
//     pub transmission_factor: f32,
//     pub transmission_tex_id: u32,
// }

// #[derive(Copy, Clone, PartialEq, Debug)]
// pub struct VolumeMap {
//     pub thickness_factor: f32,
//     pub thickness_tex_id: u32,
//     pub attenuation_distance: f32,
//     pub attenuation_color: Vec3,
// }
//
// #[derive(Copy, Clone, PartialEq)]
// pub struct SpecularMap {
//     pub specular_factor: f32,
//     pub specular_tex_id: u32,
//     pub specular_color_factor: Vec3,
//     pub specular_color_tex_id: u32,
// }

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

/////////////////////////////
// MESH & TEXTURE METADATA //
/////////////////////////////

/// Metadata for a material, including texture IDs and scalar values.
/// This is used during asset loading to organize material data before uploading to the GPU.
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct MaterialMeta {
    pub texture_ids: TextureIds,
    pub material_values: MaterialValues,
}

/// IDs of textures used by a material.
/// References textures loaded in the `TextureCache`.
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct TextureIds {
    pub base_color: u32,
    pub metallic_roughness: u32,
    pub normal_map: u32,
    pub occlusion_map: u32,
    pub emissive_map: u32,
}


impl TextureIds {
    pub fn to_vec(self) -> Vec<u32> {
        vec![self.base_color,
            self.metallic_roughness,
            self.normal_map,
            self.occlusion_map,
            self.emissive_map]
    }
}


impl Default for TextureIds {
    fn default() -> Self {
        Self {
            base_color: TextureCache::DEFAULT_COLOR_TEX,
            metallic_roughness: TextureCache::DEFAULT_ROUGH_TEX,
            normal_map: TextureCache::DEFAULT_NORMAL_TEX,
            occlusion_map: TextureCache::DEFAULT_OCCLUSION_TEX,
            emissive_map: TextureCache::DEFAULT_EMISSIVE_TEX,
        }
    }
}


impl Default for MaterialMeta {
    fn default() -> Self {
        Self {
            texture_ids: TextureIds::default(),
            material_values: MaterialValues::default(),
        }
    }
}


impl MaterialMeta {
    pub fn add_base_color(&mut self, tex_id: u32, factor: Vec4, uv_set: u32) {
        self.texture_ids.base_color = tex_id;
        self.material_values.base_color_factor = factor;
        self.material_values.base_color_uv_set = uv_set;
    }

    pub fn add_metallic_roughness(
        &mut self,
        tex_id: u32,
        metallic_factor: f32,
        roughness_factor: f32,
        uv_set: u32,
    ) {
        self.texture_ids.metallic_roughness = tex_id;
        self.material_values.metallic_factor = metallic_factor;
        self.material_values.roughness_factor = roughness_factor;
        self.material_values.met_rough_uv_set = uv_set;
    }

    pub fn add_normal(&mut self, tex_id: u32, normal_scale: f32, uv_set: u32) {
        self.texture_ids.normal_map = tex_id;
        self.material_values.normal_scale = normal_scale;
        self.material_values.normal_uv_set = uv_set;
    }

    pub fn add_occlusion(&mut self, tex_id: u32, occlusion_strength: f32, uv_set: u32) {
        self.texture_ids.occlusion_map = tex_id;
        self.material_values.occlusion_strength = occlusion_strength;
        self.material_values.occlusion_uv_set = uv_set;
    }

    pub fn add_emissive(
        &mut self,
        tex_id: u32,
        emissive_factor: Vec3,
        emissive_strength: f32,
        uv_set: u32,
    ) {
        self.texture_ids.emissive_map = tex_id;
        self.material_values.emissive_factor = emissive_factor.extend(0.0);
        self.material_values.emissive_strength = emissive_strength;
        self.material_values.emissive_uv_set = uv_set;
    }
}


/// Scalar values defining material properties.
/// Uploaded to the GPU as part of the material buffer.
#[repr(C)]
#[derive(Copy, Clone, PartialEq, Debug, Pod, Zeroable)]
pub struct MaterialValues {
    pub base_color_factor: Vec4,
    pub emissive_factor: Vec4,
    pub base_color_uv_set: u32,
    pub met_rough_uv_set: u32,
    pub normal_uv_set: u32,
    pub occlusion_uv_set: u32,
    pub emissive_uv_set: u32,
    pub metallic_factor: f32,
    pub roughness_factor: f32,
    pub emissive_strength: f32,
    pub normal_scale: f32,
    pub occlusion_strength: f32,
    pub alpha_mask: f32,
    pub alpha_mask_cutoff: f32,
}


impl Default for MaterialValues {
    fn default() -> Self {
        Self {
            base_color_factor: TextureCache::DEFAULT_BASE_COLOR_FACTOR,
            emissive_factor: TextureCache::DEFAULT_EMISSIVE_FACTOR.extend(0.0),
            base_color_uv_set: 0,
            met_rough_uv_set: 0,
            normal_uv_set: 0,
            occlusion_uv_set: 0,
            emissive_uv_set: 0,
            metallic_factor: TextureCache::DEFAULT_METALLIC_FACTOR,
            roughness_factor: TextureCache::DEFAULT_ROUGHNESS_FACTOR,
            emissive_strength: TextureCache::DEFAULT_EMISSIVE_STRENGTH,
            normal_scale: TextureCache::DEFAULT_NORMAL_SCALE,
            occlusion_strength: TextureCache::DEFAULT_OCCLUSION_STRENGTH,
            alpha_mask: 0.0,
            alpha_mask_cutoff: 0.5,
        }
    }
}


#[derive(Copy, Clone, Default, PartialEq, Debug)]
pub struct TextureSamplers {
    base_color: vk::Sampler,
    met_rough: vk::Sampler,
    normal: vk::Sampler,
    occlusion: vk::Sampler,
    emissive: vk::Sampler,
}


#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Sampler {
    Linear,
    Nearest,
}


/// Metadata for a single texture image.
/// Contains the raw bytes (before upload) and image properties.
#[derive(Clone, Default, PartialEq, Debug)]
pub struct TextureMeta {
    pub bytes: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub format: vk::Format,
    pub mips_levels: u32,
    pub uv_index: u32,
}

/// Represents a Cubemap texture on the GPU.
pub struct VkCubeMap {
    pub texture_meta: Option<TextureMeta>,
    pub full_extent: vk::Extent3D,
    pub face_extent: vk::Extent3D,
    pub allocation: vk_mem::Allocation,
    pub image: vk::Image,
    pub image_view: vk::ImageView,
    pub sampler: vk::Sampler,
}


#[derive(Copy, Clone, PartialEq, Debug)]
pub struct SurfaceMeta {
    pub start_index: u32,
    pub count: u32,
    pub material_index: Option<u32>,
}


/// CPU-side representation of a mesh geometry.
#[derive(Clone, Default, Debug)]
pub struct MeshMeta {
    pub name: String,
    pub indices: Vec<u32>,
    pub vertices: Vec<Vertex>,
    pub material_index: Option<u32>,
}

/// Intermediate representation of a scene node during loading.
#[derive(Clone, PartialEq)]
pub struct NodeMeta {
    pub name: String,
    pub mesh_indices: Vec<u32>,
    pub local_transform: Transform,
    pub og_matrix: Mat4,
    pub children: Vec<u32>,
}

/////////////////////
// SHADER UNIFORMS //
/////////////////////

// VERTEX - See top of file

/// Trait to easily convert Pod types to byte slices for Vulkan uploads.
pub trait AsByteSlice: Pod {
    fn as_byte_slice(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
}


impl<T> AsByteSlice for T where T: Pod + bytemuck::Zeroable {}

/// Uniform buffer structure for Metallic-Roughness PBR workflow.
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

////////////////////////
// SHADER PUSH CONSTS //
////////////////////////

/// Push constants for model rendering.
/// Provides the model matrix and buffer addresses for vertex and material data.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct VkModelPushConsts {
    pub model_matrix: Mat4,
    pub vertex_buffer_addr: vk::DeviceAddress,
    pub mat_meta_buffer_addr: vk::DeviceAddress,
    pub joint_count: u32,
    _pad: [u32; 3],
}


impl VkModelPushConsts {
    pub fn new(
        model_matrix: Mat4,
        vertex_buffer_addr: vk::DeviceAddress,
        mat_meta_buffer_addr:
        vk::DeviceAddress) -> Self {
        Self {
            model_matrix,
            vertex_buffer_addr,
            mat_meta_buffer_addr,
            joint_count: 0,
            _pad: [0; 3],
        }
    }

    pub fn new_anim(
        model_matrix: Mat4,
        joint_count: u32,
        vertex_buffer_addr: vk::DeviceAddress,
        mat_meta_buffer_addr: vk::DeviceAddress,
    ) -> Self {
        Self {
            model_matrix,
            vertex_buffer_addr,
            mat_meta_buffer_addr,
            joint_count,
            _pad: [0; 3],
        }
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


/// Global scene data Uniform Buffer Object (UBO).
/// Contains camera matrices and position.
#[repr(C)]
#[derive(Default, Copy, Clone, Pod, Zeroable)]
pub struct SceneDataUBO {
    pub projection: Mat4,
    pub view: Mat4,
    pub cam_pos: Vec3,
    pad: f32,
}

/// Environment lighting settings UBO.
/// Controls exposure, gamma, and IBL settings.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct EnvironmentUBO {
    pub light_dir: Vec4,
    pub exposure: f32,
    pub gamma: f32,
    pub prefilter_mips_levels: f32,
    pub ibl_ambient_scale: f32,
    pub debug_view_inputs: f32,
    pub debug_view_equation: f32,
    _pad: u64,
}


impl Default for EnvironmentUBO {
    fn default() -> Self {
        Self {
            light_dir: Vec4::new(0.1, 0.7, 0.7, 0.0),
            exposure: 4.5,
            gamma: 2.2,
            prefilter_mips_levels: 5.0,
            ibl_ambient_scale: 3.0,
            debug_view_inputs: 0.0,
            debug_view_equation: 0.0,
            _pad: 0,
        }
    }
}


#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct PushConstSkyBox {
    pub projection: Mat4, // FIXME this need combined to to stay under 128byte push const
    pub model: Mat4,
    pub vertex_buffer_addr: vk::DeviceAddress,
    pub exposure: f32,
    pub gamma: f32,
}


impl Default for PushConstSkyBox {
    fn default() -> Self {
        Self {
            projection: Default::default(),
            model: Default::default(),
            vertex_buffer_addr: vk::DeviceAddress::default(),
            exposure: 4.5,
            gamma: 2.2,
        }
    }
}


#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct PushConstIrradiance {
    pub mvp: Mat4,
    pub vertex_buffer_addr: vk::DeviceAddress,
    delta_phi: f32,
    delta_theta: f32,
}


impl PushConstIrradiance {
    pub fn new(mvp: Mat4, vertex_buffer_addr: vk::DeviceAddress) -> Self {
        Self {
            mvp,
            vertex_buffer_addr,
            delta_phi: (2.0 * PI) / 180.0,
            delta_theta: (0.5 * PI) / 64.0,
        }
    }
}


#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct PushConstPrefilterEnv {
    pub mvp: Mat4,
    pub vertex_buffer_addr: vk::DeviceAddress,
    pub roughness: f32,
    pub num_samples: u32,
}


impl PushConstPrefilterEnv {
    pub fn new(mvp: Mat4, roughness: f32, vertex_buffer_addr: vk::DeviceAddress) -> Self {
        Self {
            mvp,
            roughness,
            vertex_buffer_addr,
            num_samples: 32,
        }
    }
}

////////////////////////////
// VULKAN ALLOCATION DATA //
////////////////////////////

/// Holds handles to the allocated buffers for a mesh on the GPU.
#[derive(Debug, Copy, Clone)]
pub struct VkMeshBuffers {
    pub cache_id: u32,
    pub index_count: u32,
    pub vertex_count: u32,
    pub material_id: u32,
    pub index_buffer: VkSubAlloc,
    pub vertex_buffer: VkSubAlloc,
    pub joint_desc: vk::DescriptorSet,
}


impl VkMeshBuffers {
    pub fn get_first_index(&self) -> u32 {
        // 4 bytes per u32
        (self.index_buffer.offset / 4) as u32
    }
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

/// A single object ready to be drawn in the render pass.
/// Contains all necessary references to buffers, descriptors, and transform data.
#[derive(Debug, Copy, Clone)]
pub struct RenderObject {
    pub index_count: u32,
    pub first_index: u32,
    pub index_buffer: vk::Buffer,
    pub joint_desc: vk::DescriptorSet,
    pub material: *const VkLoadedMaterial,
    pub transform: Mat4,
    pub vertex_buffer_addr: vk::DeviceAddress,
}


// pub trait Renderable {
//     fn draw(&self, top_matrix: &Mat4, context: &mut DrawContext);
//     fn refresh_transform(&mut self);
//     fn get_children(&self) -> &Vec<Rc<RefCell<Node>>>;
// }

/// Component representing 3D spatial properties.
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


/// A node in the scene graph.
/// Can contain meshes and children nodes.
/// Handles transform hierarchy and updates.
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
    /// Traverses the node hierarchy, updating transforms and adding render objects to the draw context.
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
            let mesh = mesh_cache.get_loaded_id_unchecked(*mesh_id);
            let material_ptr = unsafe { tex_cache.get_loaded_material_unchecked_ptr(mesh.material_id) };
            let material = unsafe { *material_ptr };

            //
            // debug!("Drawing Mesh: {}", mesh_id);
            // // // debug!("\t Mesh: {:#?}", mesh);
             //debug!("\t Material: {:#?}", material);
            //
            let ro = RenderObject {
                index_count: mesh.index_count,
                joint_desc: mesh.joint_desc,
                first_index: mesh.get_first_index(),
                index_buffer: mesh.index_buffer.buffer,
                material: material_ptr,
                transform: self.world_transform,
                vertex_buffer_addr: mesh.vertex_buffer.alloc_address,
            };

            // FIXME, should use something more performant than a hash set since all types are known
            ctx.active_pipelines.insert(material.pipeline);

            unsafe {
                ctx.render_objects
                    .get_unchecked_mut(material.pipeline as usize)
                    .push(ro);
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


/// Context collected during scene traversal.
/// Buckets render objects by pipeline type for efficient rendering.
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
