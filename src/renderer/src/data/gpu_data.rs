use crate::data::gltf_util::MeshAsset;
use crate::vulkan::vk_descriptor::{
    DescriptorLayoutBuilder, DescriptorWriter, VkDescWriterType, VkDynamicDescriptorAllocator,
};
use crate::vulkan::vk_pipeline::PipelineBuilder;
use crate::vulkan::vk_render::VkRender;
use crate::vulkan::vk_types::{LogicalDevice, VkDescLayoutMap, VkDescType, VkDescriptors, VkGpuPushConsts, VkImageAlloc, VkPipeline, VkBuffer};
use crate::vulkan::vk_util;
use ash::vk;
use ash::vk::DescriptorSet;
use std::cell::RefCell;
use std::cmp::PartialEq;
use std::ffi::{CStr, CString};
use std::rc::{Rc, Weak};
use glam::{Mat4, Quat, Vec3, Vec4};
use crate::data::gltf_util;

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
            uv_x: 0.0,
            normal: Default::default(),
            uv_y: 0.0,
            color: Default::default(),
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
    pub emissive_map: Option<EmissiveMap>
}

impl MaterialMeta {
    pub fn has_normal(&self) -> bool {
        self.normal_map.is_some()
    }

    pub fn has_occlusion(&self) -> bool {
        self.occlusion_map.is_some()
    }
}


#[derive(Clone, PartialEq, Debug)]
pub struct TextureMeta {
    pub bytes: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub format: vk::Format,
    pub mips_levels: u32,
}

impl TextureMeta {
    pub fn from_gltf_texture(data : &gltf::image::Data) -> Self {
        Self {
            bytes: data.pixels.clone(),
            width: data.width,
            height: data.height,
            format: gltf_util::gltf_format_to_vk_format(data.format),
            mips_levels: 1,
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
pub struct GLTFMetallicRoughnessConstants {
    pub color_factors: Vec4,
    pub metal_rough_factors: Vec4,
    //padding
    pub extra: [Vec4; 14],
}


pub struct GLTFMetallicRoughness<'a> {
    pub opaque_pipeline: VkPipeline,
    pub transparent_pipeline: VkPipeline,
    pub descriptor_layout: [vk::DescriptorSetLayout; 1],
    pub writer: DescriptorWriter<'a>,
}

pub struct GLTFMetallicRoughnessResources {
    pub color_image: VkImageAlloc,
    pub color_sampler: vk::Sampler,
    pub metal_rough_image: VkImageAlloc,
    pub metal_rough_sampler: vk::Sampler,
    pub data_buffer: vk::Buffer,
    pub data_buffer_offset: u32,
}

impl VkGpuMetRoughPipeline<'_> {
    pub fn build_pipelines(
        device: &LogicalDevice,
        descriptors: &VkDescLayoutMap,
    ) -> (VkPipeline, VkPipeline, vk::DescriptorSetLayout) {
        let vert_shader = vk_util::load_shader_module(
            &device,
            "/home/mindspice/code/rust/engine/src/renderer/src/shaders/mesh.vert.spv",
        )
        .expect("Error loading shader");

        let frag_shader = vk_util::load_shader_module(
            &device,
            "/home/mindspice/code/rust/engine/src/renderer/src/shaders/mesh.frag.spv",
        )
        .expect("Error loading shader");

        let matrix_range = [vk::PushConstantRange::default()
            .offset(0)
            .size(std::mem::size_of::<VkGpuPushConsts>() as u32)
            .stage_flags(vk::ShaderStageFlags::VERTEX)];

        let material_layout = DescriptorLayoutBuilder::default()
            .add_binding(0, vk::DescriptorType::UNIFORM_BUFFER)
            .add_binding(1, vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .add_binding(2, vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .build(
                &device,
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                vk::DescriptorSetLayoutCreateFlags::empty(),
            )
            .unwrap();

        let layouts = [descriptors.get(VkDescType::GpuScene), material_layout];

        let mesh_layout_info = vk_util::pipeline_layout_create_info()
            .set_layouts(&layouts)
            .push_constant_ranges(&matrix_range);

        let layout = unsafe {
            device
                .device
                .create_pipeline_layout(&mesh_layout_info, None)
                .unwrap()
        };

        let entry = CString::new("main").unwrap();

        let mut pipeline_builder = PipelineBuilder::default()
            .set_shaders(vert_shader, &entry, frag_shader, &entry)
            .set_input_topology(vk::PrimitiveTopology::TRIANGLE_LIST)
            .set_color_attachment_format(vk::Format::B8G8R8A8_UNORM)
            .set_polygon_mode(vk::PolygonMode::FILL)
            .set_cull_mode(vk::CullModeFlags::NONE, vk::FrontFace::CLOCKWISE)
            .set_multisample_none()
            .disable_blending()
            .enable_depth_test(true, vk::CompareOp::GREATER_OR_EQUAL)
            .set_pipeline_layout(layout);

        let opaque_pipeline = pipeline_builder.build_pipeline(device).unwrap();

        let mut pipeline_builder = pipeline_builder
            .enable_blending_additive()
            .enable_depth_test(false, vk::CompareOp::GREATER_OR_EQUAL);

        let transparent_pipeline = pipeline_builder.build_pipeline(device).unwrap();

        unsafe {
            device.device.destroy_shader_module(vert_shader, None);
            device.device.destroy_shader_module(frag_shader, None);
        }

        (
            VkPipeline::new(opaque_pipeline, layout),
            VkPipeline::new(transparent_pipeline, layout),
            material_layout,
        )
    }

    pub fn clear_resources(&mut self, device: &ash::Device) {
        todo!()
    }

    pub fn write_material(
        &mut self,
        device: &LogicalDevice,
        pass: MaterialPass,
        resources: &GLTFMetallicRoughnessResources,
        descriptor_allocator: &mut VkDynamicDescriptorAllocator,
    ) -> VkGpuTextureBuffer {
        todo!();
        // let mat_data = VkGpuTextureBuffer {
        //     pipeline: if pass == MaterialPass::Transparent {
        //         self.transparent_pipeline
        //     } else {
        //         self.opaque_pipeline
        //     },
        //     descriptor: descriptor_allocator
        //         .allocate(device, &self.descriptor_layout)
        //         .unwrap(),
        //     pass_type: pass,
        // };
        //
        // self.writer.clear();
        // self.writer.write_buffer(
        //     0,
        //     resources.data_buffer,
        //     std::mem::size_of::<GLTFMetallicRoughnessConstants>(),
        //     resources.data_buffer_offset as usize,
        //     vk::DescriptorType::UNIFORM_BUFFER,
        // );
        //
        // self.writer.write_image(
        //     1,
        //     resources.color_image.image_view,
        //     resources.color_sampler,
        //     vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        //     vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        // );
        //
        // self.writer.write_image(
        //     2,
        //     resources.metal_rough_image.image_view,
        //     resources.metal_rough_sampler,
        //     vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        //     vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        // );
        //
        // self.writer.update_set(device, mat_data.descriptor);
        // mat_data
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


//////////////////////
// VULKAN PIPELINES //
//////////////////////

pub struct VkGpuMetRoughPipeline<'a> {
    pub opaque_pipeline: VkPipeline,
    pub transparent_pipeline: VkPipeline,
    pub descriptor_layout: [vk::DescriptorSetLayout; 1],
    pub writer: DescriptorWriter<'a>,
}

/////////////////////////////
// SCENE GRAPH & RENDERING //
/////////////////////////////

pub struct RenderObject {
    pub index_count: u32,
    pub first_index: u32,
    pub index_buffer: vk::Buffer,
    pub material: VkGpuTextureBuffer,
    pub transform: Mat4,
    pub vertex_buffer_addr: vk::DeviceAddress,
}

pub trait Renderable {
    fn draw(&self, top_matrix: &Mat4, context: &mut DrawContext);
    fn refresh_transform(&mut self, transform: &Mat4);
    fn get_children(&self) -> &Vec<Rc<RefCell<Node>>>;
}
#[derive(Debug, Copy, Clone, Default, PartialEq)]
pub struct Transform {
    pub position: Vec3,
    pub scale: Vec3,
    pub rotation: Quat,
}

impl Transform {
    pub fn compose(&self) -> Mat4 {
        let scale_matrix = Mat4::from_scale(self.scale);
        let rotation_matrix = Mat4::from_quat(self.rotation);
        let translation_matrix = Mat4::from_translation(self.position);
        translation_matrix * rotation_matrix * scale_matrix
    }
}

#[derive(Debug, Default)]
pub struct Node {
    pub parent: Option<Weak<RefCell<Node>>>,
    pub children: Vec<Rc<RefCell<Node>>>,
    pub meshes: Option<u32>,
    pub world_transform: Mat4,
    pub local_transform: Transform,
    pub dirty: bool,
}


impl Renderable for Node {
    fn draw(&self, top_matrix: &Mat4, ctx: &mut DrawContext) {
        for child in &self.children {
            child.borrow().draw(top_matrix, ctx);
        }
    }

    fn refresh_transform(&mut self, parent_matrix: &Mat4) {
        // self.world_transform = parent_matrix.mul_mat4(&self.local_transform);
        // 
        // for child in self.children.iter_mut() {
        //     child.borrow_mut().refresh_transform(&self.world_transform);
        // }
    }

    fn get_children(&self) -> &Vec<Rc<RefCell<Node>>> {
        &self.children
    }
}

pub struct DrawContext {
    pub opaque_surfaces: Vec<RenderObject>,
}

impl Default for DrawContext {
    fn default() -> Self {
        DrawContext {
            opaque_surfaces: Vec::new(),
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

#[derive(Default, Copy, Clone)]
pub struct GPUSceneData {
    pub view: Mat4,
    pub projection: Mat4,
    pub view_projection: Mat4,
    pub ambient_color: Vec4,
    pub sunlight_direction: Vec4,
    pub sunlight_color: Vec4,
}


#[repr(C)]
#[derive(PartialEq, Debug, Copy, Clone)]
pub enum MaterialPass {
    MainColor,
    Transparent,
    Other,
    NULL,
}
