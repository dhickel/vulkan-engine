use crate::data::gltf_util::MeshAsset;
use crate::vulkan::vk_descriptor::{
    DescriptorLayoutBuilder, DescriptorWriter, VkDescWriterType, VkDynamicDescriptorAllocator,
};
use crate::vulkan::vk_pipeline::PipelineBuilder;
use crate::vulkan::vk_render::VkRender;
use crate::vulkan::vk_types::{
    LogicalDevice, VkDescLayoutMap, VkDescType, VkDescriptors, VkGpuPushConsts, VkImageAlloc,
    VkPipeline,
};
use crate::vulkan::vk_util;
use ash::vk;
use ash::vk::DescriptorSet;
use glam::Mat4;
use std::cell::RefCell;
use std::cmp::PartialEq;
use std::ffi::{CStr, CString};
use std::rc::{Rc, Weak};

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Vertex {
    pub position: glam::Vec3,
    pub uv_x: f32,
    pub normal: glam::Vec3,
    pub uv_y: f32,
    pub color: glam::Vec4,
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
    pub view: glam::Mat4,
    pub projection: glam::Mat4,
    pub view_projection: glam::Mat4,
    pub ambient_color: glam::Vec4,
    pub sunlight_direction: glam::Vec4,
    pub sunlight_color: glam::Vec4,
}

#[repr(C)]
#[derive(PartialEq, Debug, Copy, Clone)]
pub enum MaterialPass {
    MainColor,
    Transparent,
    Other,
    NULL,
}
#[derive(Debug, Copy, Clone)]
pub struct MaterialInstance {
    pub pipeline: VkPipeline,
    pub descriptor: vk::DescriptorSet,
    pub pass_type: MaterialPass,
}

impl MaterialInstance {
    pub fn null() -> Self {
        Self {
            pipeline: VkPipeline::new(vk::Pipeline::null(), vk::PipelineLayout::null()),
            descriptor: vk::DescriptorSet::null(),
            pass_type: MaterialPass::NULL,
        }
    }
}

pub struct RenderObject {
    pub index_count: u32,
    pub first_index: u32,
    pub index_buffer: vk::Buffer,
    pub material: MaterialInstance,
    pub transform: glam::Mat4,
    pub vertex_buffer_addr: vk::DeviceAddress,
}

pub struct GLTFMetallicRoughness<'a> {
    pub opaque_pipeline: VkPipeline,
    pub transparent_pipeline: VkPipeline,
    pub descriptor_layout: [vk::DescriptorSetLayout; 1],
    pub writer: DescriptorWriter<'a>,
}

#[repr(C)]
pub struct GLTFMetallicRoughnessConstants {
    pub color_factors: glam::Vec4,
    pub metal_rough_factors: glam::Vec4,
    //padding
    pub extra: [glam::Vec4; 14],
}

pub struct GLTFMetallicRoughnessResources {
    pub color_image: VkImageAlloc,
    pub color_sampler: vk::Sampler,
    pub metal_rough_image: VkImageAlloc,
    pub metal_rough_sampler: vk::Sampler,
    pub data_buffer: vk::Buffer,
    pub data_buffer_offset: u32,
}

impl GLTFMetallicRoughness<'_> {
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
            material_layout
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
    ) -> MaterialInstance {
        let mat_data = MaterialInstance {
            pipeline: if pass == MaterialPass::Transparent {
                self.transparent_pipeline
            } else {
                self.opaque_pipeline
            },
            descriptor: descriptor_allocator
                .allocate(device, &self.descriptor_layout)
                .unwrap(),
            pass_type: pass,
        };

        self.writer.clear();
        self.writer.write_buffer(
            0,
            resources.data_buffer,
            std::mem::size_of::<GLTFMetallicRoughnessConstants>(),
            resources.data_buffer_offset as usize,
            vk::DescriptorType::UNIFORM_BUFFER,
        );

        self.writer.write_image(
            1,
            resources.color_image.image_view,
            resources.color_sampler,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        );

        self.writer.write_image(
            2,
            resources.metal_rough_image.image_view,
            resources.metal_rough_sampler,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        );

        self.writer.update_set(device, mat_data.descriptor);
        mat_data
    }
}

pub trait Renderable {
    fn draw(&self, top_matrix: &glam::Mat4, context: &mut DrawContext);
    fn refresh_transform(&mut self, transform: &glam::Mat4);
    fn get_children(&self) -> &Vec<Rc<RefCell<dyn Renderable>>>;
}


pub struct Node {
    pub parent: Option<Weak<Node>>,
    pub children: Vec<Rc<RefCell<dyn Renderable>>>,
    pub world_transform: glam::Mat4,
    pub local_transform: glam::Mat4,
}


impl Default for Node {
    fn default() -> Self {
        Self {
            parent: None,
            children: vec![],
            world_transform: Default::default(),
            local_transform: Default::default(),
        }
    }
}

impl Renderable for Node {
    fn draw(&self, top_matrix: &glam::Mat4, ctx: &mut DrawContext) {
        for child in &self.children {
            child.borrow().draw(top_matrix, ctx);
        }
    }

    fn refresh_transform(&mut self, parent_matrix: &glam::Mat4) {
        self.world_transform = parent_matrix.mul_mat4(&self.local_transform);

        for child in self.children.iter_mut() {
            child.borrow_mut().refresh_transform(&self.world_transform);
        }
    }

    fn get_children(&self) -> &Vec<Rc<RefCell<dyn Renderable>>> {
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


pub struct MeshNode {
    pub node: Node,
    pub mesh: Rc<MeshAsset>,
}

impl MeshNode {
    pub fn new(mesh: Rc<MeshAsset>) -> Self {
        Self {
            node: Default::default(),
            mesh,
        }
    }
}

impl Renderable for MeshNode {
    fn draw(&self, top_matrix: &Mat4, context: &mut DrawContext) {
        let node_matrix = top_matrix.mul_mat4(&self.node.world_transform);

        for surface in &self.mesh.surfaces {
            let ro = RenderObject {
                index_count: surface.count,
                first_index: surface.start_index,
                index_buffer: self.mesh.mesh_buffers.index_buffer.buffer,
                material: surface.material.data,
                transform: node_matrix,
                vertex_buffer_addr: self.mesh.mesh_buffers.vertex_buffer_addr,
            };

            context.opaque_surfaces.push(ro);
        }
        self.node.draw(top_matrix, context);
    }

    fn refresh_transform(&mut self, transform: &Mat4) {
        todo!()
    }

    fn get_children(&self) -> &Vec<Rc<RefCell<dyn Renderable>>> {
        todo!()
    }
}
