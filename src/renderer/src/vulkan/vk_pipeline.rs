use crate::data::data_cache::{
    CoreShaderType, VkDescLayoutCache, VkDescType, VkPipelineCache, VkPipelineType, VkShaderCache,
};
use crate::data::gpu_data::{SkyboxPushConstants, VkGpuPushConsts};
use crate::vulkan::vk_types::*;
use crate::vulkan::{vk_descriptor, vk_util};
use ash::vk;
use std::ffi::{CStr, CString};
use std::io::{Read, Seek, SeekFrom};

pub struct PipelineBuilder<'a> {
    pub shader_stages: Vec<vk::PipelineShaderStageCreateInfo<'a>>,
    pub input_assembly: vk::PipelineInputAssemblyStateCreateInfo<'a>,
    pub rasterizer: vk::PipelineRasterizationStateCreateInfo<'a>,
    pub color_blend_attachment: [vk::PipelineColorBlendAttachmentState; 1],
    pub multi_sampling: vk::PipelineMultisampleStateCreateInfo<'a>,
    pub pipeline_layout: vk::PipelineLayout,
    pub depth_stencil: vk::PipelineDepthStencilStateCreateInfo<'a>,
    pub render_info: vk::PipelineRenderingCreateInfo<'a>,
    pub color_attachment_format: [vk::Format; 1],
}

impl<'a> Default for PipelineBuilder<'a> {
    fn default() -> Self {
        Self {
            shader_stages: vec![],
            input_assembly: Default::default(),
            rasterizer: vk::PipelineRasterizationStateCreateInfo::default(),
            color_blend_attachment: [vk::PipelineColorBlendAttachmentState::default()],
            multi_sampling: Default::default(),
            pipeline_layout: Default::default(),
            depth_stencil: Default::default(),
            render_info: Default::default(),
            color_attachment_format: [vk::Format::UNDEFINED],
        }
    }
}

impl<'a> PipelineBuilder<'a> {
    pub fn clear(&mut self) {
        self.shader_stages.clear();
        self.input_assembly = Default::default();
        self.rasterizer = Default::default();
        self.color_blend_attachment = [vk::PipelineColorBlendAttachmentState::default()];
        self.multi_sampling = Default::default();
        self.pipeline_layout = Default::default();
        self.depth_stencil = Default::default();
        self.render_info = Default::default();
        self.color_attachment_format = [vk::Format::UNDEFINED];
    }

    pub fn build_pipeline(&mut self, device: &ash::Device) -> Result<vk::Pipeline, String> {
        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);

        let color_blending = vk::PipelineColorBlendStateCreateInfo::default()
            .logic_op_enable(false)
            .logic_op(vk::LogicOp::COPY)
            .attachments(&self.color_blend_attachment);

        let vertex_input_info = vk::PipelineVertexInputStateCreateInfo::default();

        let state = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_info = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&state);

        let mut render_info = self
            .render_info
            .color_attachment_formats(&self.color_attachment_format);

        let pipeline_info = [vk::GraphicsPipelineCreateInfo::default()
            .stages(&self.shader_stages)
            .vertex_input_state(&vertex_input_info)
            .input_assembly_state(&self.input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&self.rasterizer)
            .multisample_state(&self.multi_sampling)
            .color_blend_state(&color_blending)
            .depth_stencil_state(&self.depth_stencil)
            .layout(self.pipeline_layout)
            .dynamic_state(&dynamic_info)
            .push_next(&mut render_info)];

        unsafe {
            Ok(device
                .create_graphics_pipelines(vk::PipelineCache::null(), &pipeline_info, None)
                .map_err(|err| format!("Error creating pipeline: {:?}", err))?[0])
        }
    }

    pub fn set_shaders(
        mut self,
        vertex_shader: vk::ShaderModule,
        vertex_entry: &'a CStr,
        fragment_shader: vk::ShaderModule,
        fragment_entry: &'a CStr,
    ) -> Self {
        self.shader_stages.clear();

        let vertex_info = vk_util::pipeline_shader_stage_create_info(
            vk::ShaderStageFlags::VERTEX,
            vertex_shader,
            vertex_entry,
        );

        let fragment_info = vk_util::pipeline_shader_stage_create_info(
            vk::ShaderStageFlags::FRAGMENT,
            fragment_shader,
            fragment_entry,
        );

        self.shader_stages.push(vertex_info);
        self.shader_stages.push(fragment_info);
        self
    }

    pub fn set_input_topology(mut self, topology: vk::PrimitiveTopology) -> Self {
        self.input_assembly = self
            .input_assembly
            .topology(topology)
            .primitive_restart_enable(false);
        self
    }

    pub fn set_polygon_mode(mut self, mode: vk::PolygonMode) -> Self {
        self.rasterizer = self.rasterizer.polygon_mode(mode).line_width(1f32);
        self
    }

    pub fn set_cull_mode(
        mut self,
        cull_mode: vk::CullModeFlags,
        front_face: vk::FrontFace,
    ) -> Self {
        self.rasterizer = self.rasterizer.cull_mode(cull_mode);
        self.rasterizer = self.rasterizer.front_face(front_face);
        self
    }

    pub fn set_multisample_none(mut self) -> Self {
        self.multi_sampling = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1)
            .sample_shading_enable(false)
            .min_sample_shading(1.0)
            .sample_mask(&[])
            .alpha_to_coverage_enable(false)
            .alpha_to_one_enable(false);
        self
    }

    pub fn disable_blending(mut self) -> Self {
        self.color_blend_attachment[0] = self.color_blend_attachment[0]
            .color_write_mask(
                vk::ColorComponentFlags::R
                    | vk::ColorComponentFlags::G
                    | vk::ColorComponentFlags::B
                    | vk::ColorComponentFlags::A,
            )
            .blend_enable(false);
        self
    }

    pub fn enable_blending_additive(mut self) -> Self {
        self.color_blend_attachment[0] = self.color_blend_attachment[0]
            .color_write_mask(
                vk::ColorComponentFlags::R
                    | vk::ColorComponentFlags::G
                    | vk::ColorComponentFlags::B
                    | vk::ColorComponentFlags::A,
            )
            .blend_enable(true)
            .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
            .dst_color_blend_factor(vk::BlendFactor::ONE)
            .color_blend_op(vk::BlendOp::ADD)
            .src_alpha_blend_factor(vk::BlendFactor::ONE)
            .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
            .alpha_blend_op(vk::BlendOp::ADD);
        self
    }

    pub fn enable_blending_alpha_blend(mut self) -> Self {
        self.color_blend_attachment[0] = self.color_blend_attachment[0]
            .color_write_mask(
                vk::ColorComponentFlags::R
                    | vk::ColorComponentFlags::G
                    | vk::ColorComponentFlags::B
                    | vk::ColorComponentFlags::A,
            )
            .blend_enable(true)
            .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
            .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .color_blend_op(vk::BlendOp::ADD)
            .src_alpha_blend_factor(vk::BlendFactor::ONE)
            .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
            .alpha_blend_op(vk::BlendOp::ADD);
        self
    }

    pub fn set_color_attachment_format(mut self, format: vk::Format) -> Self {
        self.color_attachment_format = [format];

        // self.render_info = vk::PipelineRenderingCreateInfo::default()
        //     .color_attachment_formats(&self.color_attachment_format);
        //
        // println!(":? {:?}", self.color_attachment_format);
        // println!(":? {:?}", self.render_info);
        //
        self
    }

    pub fn set_depth_format(mut self, format: vk::Format) -> Self {
        self.render_info = self.render_info.depth_attachment_format(format);
        self
    }

    pub fn disable_depth_test(mut self) -> Self {
        self.depth_stencil = self
            .depth_stencil
            .depth_test_enable(false)
            .depth_write_enable(false)
            .depth_compare_op(vk::CompareOp::NEVER)
            .depth_bounds_test_enable(false)
            .stencil_test_enable(false)
            // .front(vk::StencilOpState::default())
            // .back(vk::StencilOpState::default()) // maybe skip these
            .min_depth_bounds(0.0)
            .max_depth_bounds(1.0);
        self
    }

    pub fn enable_depth_test(mut self, write_enable: bool, compare_op: vk::CompareOp) -> Self {
        self.depth_stencil = self
            .depth_stencil
            .depth_test_enable(true)
            .depth_write_enable(write_enable)
            .depth_compare_op(compare_op)
            .depth_bounds_test_enable(false)
            .stencil_test_enable(false)
            // .front(vk::StencilOpState::default())
            // .back(vk::StencilOpState::default()) // maybe skip these
            .min_depth_bounds(0.0)
            .max_depth_bounds(1.0);
        self
    }

    pub fn set_pipeline_layout(mut self, layout: vk::PipelineLayout) -> Self {
        self.pipeline_layout = layout;
        self
    }
}

pub fn init_pipeline_cache(
    device: &ash::Device,
    desc_layout_cache: &VkDescLayoutCache,
    shader_cache: &VkShaderCache,
    color_format: vk::Format,
    depth_format: vk::Format,
) -> VkPipelineCache {
    let (pbr_opaque, pbr_alpha) = init_met_rough_pipelines(
        device,
        desc_layout_cache,
        shader_cache,
        color_format,
        depth_format,
    );

    let (pbr_opaque_ext, pbr_alpha_ext) = init_met_rough_ext_pipelines(
        device,
        desc_layout_cache,
        shader_cache,
        color_format,
        depth_format,
    );

    let brd_flut_pipeline = init_brd_flut_pipeline(
        device,
        desc_layout_cache,
        shader_cache,
        color_format,
        depth_format,
    );
    
    let skybox_pipeline = init_skybox_pipeline(
        device,
        desc_layout_cache,
        shader_cache,
        color_format,
        depth_format
    );

    VkPipelineCache::new(vec![
        (VkPipelineType::PbrMetRoughOpaque, pbr_opaque),
        (VkPipelineType::PbrMetRoughAlpha, pbr_alpha),
        (VkPipelineType::PbrMetRoughOpaqueExt, pbr_opaque_ext),
        (VkPipelineType::PbrMetRoughAlphaExt, pbr_alpha_ext),
        (VkPipelineType::BrdfLut, brd_flut_pipeline),
        (VkPipelineType::Skybox, skybox_pipeline),
    ])
    .unwrap()
}

fn init_met_rough_pipelines(
    device: &ash::Device,
    desc_layout_cache: &VkDescLayoutCache,
    shader_cache: &VkShaderCache,
    color_format: vk::Format,
    depth_format: vk::Format,
) -> (VkPipeline, VkPipeline) {
    let vert_shader = shader_cache.get_core_shader(CoreShaderType::MetRoughVert);

    let frag_shader = shader_cache.get_core_shader(CoreShaderType::MetRoughFrag);

    let matrix_range = [vk::PushConstantRange::default()
        .offset(0)
        .size(std::mem::size_of::<VkGpuPushConsts>() as u32)
        .stage_flags(vk::ShaderStageFlags::VERTEX)];

    let layouts = [
        desc_layout_cache.get(VkDescType::GpuScene),
        desc_layout_cache.get(VkDescType::PbrMetRough),
    ];

    let mesh_layout_info = vk_util::pipeline_layout_create_info()
        .set_layouts(&layouts)
        .push_constant_ranges(&matrix_range);

    let layout = unsafe {
        device
            .create_pipeline_layout(&mesh_layout_info, None)
            .unwrap()
    };

    let entry = CString::new("main").unwrap();

    let mut pipeline_builder = PipelineBuilder::default()
        .set_shaders(vert_shader, &entry, frag_shader, &entry)
        .set_input_topology(vk::PrimitiveTopology::TRIANGLE_LIST)
        .set_color_attachment_format(color_format)
        .set_depth_format(depth_format)
        .set_polygon_mode(vk::PolygonMode::FILL)
        .set_cull_mode(vk::CullModeFlags::NONE, vk::FrontFace::CLOCKWISE)
        .set_multisample_none()
        .disable_blending()
        .enable_depth_test(true, vk::CompareOp::LESS_OR_EQUAL)
        .set_pipeline_layout(layout);

    let opaque_pipeline = pipeline_builder.build_pipeline(device).unwrap();

    let mut pipeline_builder = pipeline_builder
        .enable_blending_additive()
        .enable_depth_test(false, vk::CompareOp::LESS_OR_EQUAL);

    let transparent_pipeline = pipeline_builder.build_pipeline(device).unwrap();

    (
        VkPipeline::new(opaque_pipeline, layout),
        VkPipeline::new(transparent_pipeline, layout),
    )
}

fn init_met_rough_ext_pipelines(
    device: &ash::Device,
    desc_layout_cache: &VkDescLayoutCache,
    shader_cache: &VkShaderCache,
    color_format: vk::Format,
    depth_format: vk::Format,
) -> (VkPipeline, VkPipeline) {
    let vert_shader = shader_cache.get_core_shader(CoreShaderType::MetRoughVertExt);

    let frag_shader = shader_cache.get_core_shader(CoreShaderType::MetRoughFragExt);

    let matrix_range = [vk::PushConstantRange::default()
        .offset(0)
        .size(std::mem::size_of::<VkGpuPushConsts>() as u32)
        .stage_flags(vk::ShaderStageFlags::VERTEX)];

    let layouts = [
        desc_layout_cache.get(VkDescType::GpuScene),
        desc_layout_cache.get(VkDescType::PbrMetRoughExt),
    ];

    let mesh_layout_info = vk_util::pipeline_layout_create_info()
        .set_layouts(&layouts)
        .push_constant_ranges(&matrix_range);

    let layout = unsafe {
        device
            .create_pipeline_layout(&mesh_layout_info, None)
            .unwrap()
    };

    let entry = CString::new("main").unwrap();

    let mut pipeline_builder = PipelineBuilder::default()
        .set_shaders(vert_shader, &entry, frag_shader, &entry)
        .set_input_topology(vk::PrimitiveTopology::TRIANGLE_LIST)
        .set_color_attachment_format(color_format)
        .set_depth_format(depth_format)
        .set_polygon_mode(vk::PolygonMode::FILL)
        .set_cull_mode(vk::CullModeFlags::NONE, vk::FrontFace::CLOCKWISE)
        .set_multisample_none()
        .disable_blending()
        .enable_depth_test(true, vk::CompareOp::LESS_OR_EQUAL)
        .set_pipeline_layout(layout);

    let opaque_pipeline = pipeline_builder.build_pipeline(device).unwrap();

    let mut pipeline_builder = pipeline_builder
        .enable_blending_additive()
        .enable_depth_test(false, vk::CompareOp::LESS_OR_EQUAL);

    let transparent_pipeline = pipeline_builder.build_pipeline(device).unwrap();

    (
        VkPipeline::new(opaque_pipeline, layout),
        VkPipeline::new(transparent_pipeline, layout),
    )
}

fn init_brd_flut_pipeline(
    device: &ash::Device,
    desc_layout_cache: &VkDescLayoutCache,
    shader_cache: &VkShaderCache,
    color_format: vk::Format,
    depth_format: vk::Format,
) -> VkPipeline {
    let vert_shader = shader_cache.get_core_shader(CoreShaderType::BrtFlutVert);

    let frag_shader = shader_cache.get_core_shader(CoreShaderType::BrtFlutFrag);

    let layouts = [desc_layout_cache.get(VkDescType::Empty)];

    let mesh_layout_info = vk_util::pipeline_layout_create_info().set_layouts(&layouts);

    let layout = unsafe {
        device
            .create_pipeline_layout(&mesh_layout_info, None)
            .unwrap()
    };

    let entry = CString::new("main").unwrap();

    let mut pipeline_builder = PipelineBuilder::default()
        .set_shaders(vert_shader, &entry, frag_shader, &entry)
        .set_input_topology(vk::PrimitiveTopology::TRIANGLE_LIST)
        .set_color_attachment_format(color_format)
        .set_polygon_mode(vk::PolygonMode::FILL)
        .set_cull_mode(vk::CullModeFlags::NONE, vk::FrontFace::CLOCKWISE)
        .set_multisample_none()
        .disable_blending()
        .disable_depth_test()
        .enable_depth_test(true, vk::CompareOp::LESS_OR_EQUAL)
        .set_pipeline_layout(layout);

    let pipeline = pipeline_builder.build_pipeline(device).unwrap();

    VkPipeline::new(pipeline, layout)
}

fn init_skybox_pipeline(
    device: &ash::Device,
    desc_layout_cache: &VkDescLayoutCache,
    shader_cache: &VkShaderCache,
    color_format: vk::Format,
    depth_format: vk::Format,
) -> VkPipeline {
    let vert_shader = shader_cache.get_core_shader(CoreShaderType::SkyBoxVert);
    let frag_shader = shader_cache.get_core_shader(CoreShaderType::SkyBoxFrag);

    let push_constant_range = [vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
        .offset(0)
        .size(std::mem::size_of::<SkyboxPushConstants>() as u32)];
    

    let layouts = [desc_layout_cache.get(VkDescType::Skybox)];
    
    let layout_info = vk_util::pipeline_layout_create_info()
        .set_layouts(&layouts)
        .push_constant_ranges(&push_constant_range);

    let layout = unsafe {
        device
            .create_pipeline_layout(&layout_info, None)
            .unwrap()
    };

    let entry = CString::new("main").unwrap();

    let mut pipeline_builder = PipelineBuilder::default()
        .set_shaders(vert_shader, &entry, frag_shader, &entry)
        .set_input_topology(vk::PrimitiveTopology::TRIANGLE_LIST)
        .set_polygon_mode(vk::PolygonMode::FILL)
        .set_cull_mode(vk::CullModeFlags::NONE, vk::FrontFace::CLOCKWISE)
        .set_multisample_none()
        .disable_blending()
        .set_color_attachment_format(color_format)
        .set_depth_format(depth_format)
        .enable_depth_test(true, vk::CompareOp::LESS_OR_EQUAL)
        .set_pipeline_layout(layout);

    let pipeline = pipeline_builder.build_pipeline(device).unwrap();

    VkPipeline::new(pipeline, layout)
}
