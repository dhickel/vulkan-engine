use crate::data::data_cache::{
    CoreShaderType, MeshCache, ShaderCache, TextureCache, VkLoadedMaterial, VkPipelineType,
};
use crate::data::gltf_util;
use crate::data::gltf_util::MeshAsset;
use crate::vulkan::vk_descriptor::{
    DescriptorLayoutBuilder, DescriptorWriter, VkDescWriterType, VkDynamicDescriptorAllocator,
};
use crate::vulkan::vk_pipeline::PipelineBuilder;
use crate::vulkan::vk_render::VkRender;
use crate::vulkan::vk_types::{
    LogicalDevice, VkBuffer, VkDescriptors, VkGpuPushConsts, VkImageAlloc, VkPipeline,
};
use crate::vulkan::vk_util;
use ash::vk;
use ash::vk::DescriptorSet;
use bytemuck::{Pod, Zeroable};
use glam::{vec4, Mat4, Quat, Vec2, Vec3, Vec4};
use imgui::sys::igSetClipboardText;
use std::cell::{Ref, RefCell};
use std::cmp::PartialEq;
use std::f32::consts::PI;
use std::ffi::{CStr, CString};
use std::rc::{Rc, Weak};

//////////////////////////
//  MESH & TEXTURE DATA //
//////////////////////////

#[repr(C)]
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Vertex {
    pub position: Vec3,
    pub uv_x: f32,
    pub normal: Vec3,
    pub uv_y: f32,
    pub color: Vec4,
}

impl Default for Vertex {
    fn default() -> Self {
        Self {
            position: Default::default(),
            normal: Default::default(),
            color: Default::default(),
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

impl TextureMeta {
    pub fn from_gltf_texture(data: &gltf::image::Data) -> Self {
        Self {
            bytes: data.pixels.clone(),
            width: data.width,
            height: data.height,
            format: gltf_util::gltf_format_to_vk_format(data.format),
            mips_levels: 1,
            sampler: Sampler::Linear,
        }
    }
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
    pub surfaces: Vec<SurfaceMeta>,
}

#[derive(Clone, PartialEq)]
pub struct NodeMeta {
    pub name: String,
    pub mesh_index: Option<u32>,
    pub local_transform: Transform,
    pub children: Vec<u32>,
}

/////////////////
// SHADER DATA //
/////////////////
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct MetRoughShaderConsts {
    pub color_factors: Vec4,
    pub metal_rough_factors: Vec4,
    //padding
    pub extra: [Vec4; 14],
}

//
// pub struct GLTFMetallicRoughness<'a> {
//     pub opaque_pipeline: VkPipeline,
//     pub transparent_pipeline: VkPipeline,
//     pub descriptor_layout: [vk::DescriptorSetLayout; 1],
//     pub writer: DescriptorWriter<'a>,
// }
//
// pub struct GLTFMetallicRoughnessResources {
//     pub color_image: VkImageAlloc,
//     pub color_sampler: vk::Sampler,
//     pub metal_rough_image: VkImageAlloc,
//     pub metal_rough_sampler: vk::Sampler,
//     pub data_buffer: vk::Buffer,
//     pub data_buffer_offset: u32,
// }

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

//////////////////////
// VULKAN PIPELINES //
//////////////////////

// pub struct VkGpuMetRoughPipeline<'a> {
//     pub opaque_pipeline: VkPipeline,
//     pub transparent_pipeline: VkPipeline,
//     pub descriptor_layout: [vk::DescriptorSetLayout; 1],
//     pub writer: DescriptorWriter<'a>,
// }
//
// impl VkGpuMetRoughPipeline<'_> {
//     pub fn build_pipelines(
//         device: &ash::Device,
//         descriptors: &VkDescLayoutMap,
//         shader_cache: &ShaderCache,
//     ) -> (VkPipeline, VkPipeline, vk::DescriptorSetLayout) {
//         let vert_shader = shader_cache.get_core_shader(CoreShaderType::MetRoughVert);
//
//         let frag_shader = shader_cache.get_core_shader(CoreShaderType::MetRoughFrag);
//
//         let matrix_range = [vk::PushConstantRange::default()
//             .offset(0)
//             .size(std::mem::size_of::<VkGpuPushConsts>() as u32)
//             .stage_flags(vk::ShaderStageFlags::VERTEX)];
//
//         let material_layout =
//
//         let layouts = [descriptors.get(VkDescType::GpuScene), material_layout];
//
//         let mesh_layout_info = vk_util::pipeline_layout_create_info()
//             .set_layouts(&layouts)
//             .push_constant_ranges(&matrix_range);
//
//         let layout = unsafe {
//             device
//                 .create_pipeline_layout(&mesh_layout_info, None)
//                 .unwrap()
//         };
//
//         let entry = CString::new("main").unwrap();
//
//         let mut pipeline_builder = PipelineBuilder::default()
//             .set_shaders(vert_shader, &entry, frag_shader, &entry)
//             .set_input_topology(vk::PrimitiveTopology::TRIANGLE_LIST)
//             .set_color_attachment_format(vk::Format::B8G8R8A8_UNORM)
//             .set_polygon_mode(vk::PolygonMode::FILL)
//             .set_cull_mode(vk::CullModeFlags::NONE, vk::FrontFace::CLOCKWISE)
//             .set_multisample_none()
//             .disable_blending()
//             .enable_depth_test(true, vk::CompareOp::GREATER_OR_EQUAL)
//             .set_pipeline_layout(layout);
//
//         let opaque_pipeline = pipeline_builder.build_pipeline(device).unwrap();
//
//         let mut pipeline_builder = pipeline_builder
//             .enable_blending_additive()
//             .enable_depth_test(false, vk::CompareOp::GREATER_OR_EQUAL);
//
//         let transparent_pipeline = pipeline_builder.build_pipeline(device).unwrap();
//
//         (
//             VkPipeline::new(opaque_pipeline, layout),
//             VkPipeline::new(transparent_pipeline, layout),
//             material_layout,
//         )
//     }
//
//     pub fn clear_resources(&mut self, device: &ash::Device) {
//         todo!()
//     }
//
//     pub fn write_material(
//         &mut self,
//         device: &ash::Device,
//         material_id: u32,
//         texture_cache: &TextureCache,
//         descriptor_allocator: &mut VkDynamicDescriptorAllocator,
//     )  {
//
//         let material = texture_cache.get_loaded_material_unchecked(material_id);
//         let color_tex = texture_cache.get_loaded_texture_unchecked(material.meta.base_color_tex_id);
//         let metallic_tex =
//             texture_cache.get_loaded_texture_unchecked(material.meta.metallic_roughness_tex_id);
//
//         let pipeline = match material.meta.alpha_mode {
//             AlphaMode::Opaque => self.opaque_pipeline
//             AlphaMode::Mask | AlphaMode::Blend => self.transparent_pipeline
//         };
//
//         let descriptor = descriptor_allocator
//             .allocate(device, &self.descriptor_layout)
//             .unwrap();
//
//         self.writer.write_buffer(
//             0,
//             material.uniform_buffer.buffer,
//             std::mem::size_of::<MetRoughShaderConsts>(),
//             material.buffer_offset as usize,
//             vk::DescriptorType::UNIFORM_BUFFER,
//         );
//
//         self.writer.write_image(
//             1,
//             color_tex.alloc.image_view,
//             texture_cache.get_sampler(color_tex.meta.sampler),
//             vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
//             vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
//         );
//
//         self.writer.write_image(
//             2,
//             metallic_tex.alloc.image_view,
//             texture_cache.get_sampler(metallic_tex.meta.sampler),
//             vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
//             vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
//         );
//
//         self.writer.update_set(device, descriptor);
//     }
// }

/////////////////////////////
// SCENE GRAPH & RENDERING //
/////////////////////////////

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
    pub fn compose(&self) -> Mat4 {
        glam::Mat4::from_scale_rotation_translation(-self.scale, self.rotation, self.position)
    }

    pub fn new_vulkan_adjusted(translation: [f32; 3], rotation: [f32; 4], scale: [f32; 3]) -> Self {
        Transform {
            position: glam::Vec3::from_array(translation),
            scale: glam::Vec3::from_array(scale),
            rotation: glam::quat(rotation[3], rotation[0], rotation[1],rotation[2]),
        }
    }
}

#[derive(Debug)]
pub struct Node {
    pub parent: Option<Weak<RefCell<Node>>>,
    pub children: Vec<Rc<RefCell<Node>>>,
    pub meshes: Option<u32>,
    pub world_transform: Mat4,
    pub local_transform: Transform,
    pub dirty: bool,
}

impl Default for Node {
    fn default() -> Self {
        Self {
            parent: None,
            children: vec![],
            meshes: None,
            world_transform: Mat4::IDENTITY,
            local_transform: Transform {
                position: glam::vec3(0.0, 0.0, 0.0),
                scale: glam::vec3(1.0, 1.0, 1.0),
                rotation: glam::Quat::IDENTITY,
            },
            dirty: false,
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
            self.refresh_transform(top_matrix);
        }

        if let Some(id) = self.meshes {
            let mesh = mesh_cache.get_loaded_mesh_unchecked(id);

            for surface in &mesh.meta.surfaces {
                if let Some(id) = surface.material_index {
                    let material = tex_cache.get_loaded_material_unchecked(id);
                    let material_ptr = material as *const VkLoadedMaterial;
                    let ro = RenderObject {
                        index_count: surface.count,
                        first_index: surface.start_index,
                        index_buffer: mesh.buffer.index_buffer.buffer,
                        material: material_ptr,
                        transform: self.world_transform,
                        vertex_buffer_addr: mesh.buffer.vertex_buffer_addr,
                    };

                    match material.pipeline {
                        VkPipelineType::PbrMetRoughOpaque => {
                            ctx.opaque_surfaces.push(ro)
                        }
                        VkPipelineType::PbrMetRoughAlpha => {
                            ctx.transparent_surfaces.push(ro)
                        }
                        VkPipelineType::Mesh => {panic!("Wrong pipeline")}
                    }
                }
            }
        }

        for child in &self.children {
            child
                .borrow_mut()
                .draw(top_matrix, ctx, mesh_cache, tex_cache);
        }
    }

    pub fn refresh_transform(&mut self, parent_transform: &Mat4) {
        // Compute new world transform based on parent's world transform if available

        self.world_transform = parent_transform.mul_mat4(&self.local_transform.compose());

        // Update the node's world transform
        self.dirty = false;

        println!("set transform to: {:?}", self.world_transform);

        // Pass the new world transform to the children
        for child in &self.children {
            let mut child = child.borrow_mut();
            child.refresh_transform(&self.world_transform);
        }
    }

    fn get_children(&self) -> &Vec<Rc<RefCell<Node>>> {
        &self.children
    }
}

pub struct DrawContext {
    pub opaque_surfaces: Vec<RenderObject>,
    pub transparent_surfaces: Vec<RenderObject>,
}

impl Default for DrawContext {
    fn default() -> Self {
        DrawContext {
            opaque_surfaces: Vec::new(),
            transparent_surfaces: Vec::new(),
        }
    }
}

pub struct GPUScene {
    pub data: GPUSceneData,
    pub descriptor: [vk::DescriptorSetLayout; 1],
}

impl GPUScene {
    pub fn new(descriptor: vk::DescriptorSetLayout) -> Self {
        Self {
            data: Default::default(),
            descriptor: [descriptor],
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

impl GPUSceneData {
    pub fn as_byte_slice(&self) -> &[u8] {
        bytemuck::bytes_of(self)
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
