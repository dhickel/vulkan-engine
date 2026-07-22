//! # Graphics Pipeline Builder & Initialization
//!
//! ## Purpose
//! Provides builder pattern for creating Vulkan graphics pipelines with dynamic rendering
//! (Vulkan 1.3). No VkRenderPass objects - uses VkPipelineRenderingCreateInfo instead.
//!
//! Internal Vulkan pipeline builder; dead code allowed.
//!
//! ## Key Concepts
//! - **PipelineBuilder**: Fluent interface for pipeline creation
//! - **Dynamic rendering**: VK_KHR_dynamic_rendering (core in Vulkan 1.3)
//! - **Dynamic state**: Viewport/scissor set via vkCmdSetViewport/Scissor (not baked into pipeline)
//! - **Vertex pulling**: No vertex input state (shaders fetch vertices via buffer device address)
//!
//! ## Why Dynamic Rendering
//! - Simpler API: No VkRenderPass/VkFramebuffer objects to manage
//! - More flexible: Can change attachments without recreating pipeline
//! - Vulkan 1.3 core: No extension required
//! - Matches modern rendering patterns (forward+ rendering, dynamic resolution)
//!
//! ## Pipeline Types
//! - **PbrMetRoughOpaque**: Opaque PBR materials (depth write enabled)
//! - **PbrMetRoughAlpha**: Transparent PBR (additive blending, depth write disabled)
//! - **Skybox**: Environment skybox (no depth test, rendered first)
//! - **EnvIrradiance/PreFilter**: IBL cubemap generation (offline, not used per-frame)
//!
//! ## Vertex Pulling Pattern
//! No vertex input state defined. Shaders fetch vertices manually:
//! ```glsl
//! layout(push_constant) uniform PushConsts { uint64_t vertex_buffer_addr; };
//! layout(buffer_reference, std430) readonly buffer VertexBuffer { Vertex vertices[]; };
//! Vertex v = VertexBuffer(vertex_buffer_addr).vertices[gl_VertexIndex];
//! ```
//! Why: Simpler pipeline creation, flexible vertex formats, better for GPU-driven rendering.
//!
//! ## PipelineSpec — centralised creation boundary
//!
//! Every pipeline constructor maps its parameters into a [`PipelineSpec`] value and calls
//! [`create_pipeline_from_spec`]. The spec is validated before any Vulkan call; contradictory
//! attachment/depth combos are rejected. Rollback destroys only successfully-created resources
//! and shared layouts are deduplicated in the [`VkPipelineCache`] destructor.

use crate::data::data_cache::{
    CoreShaderType, VkDescLayoutCache, VkDescType, VkPipelineCache, VkPipelineType, VkShaderCache,
};
use crate::data::gpu_data::{
    PushConstCubeCapture, PushConstIrradiance, PushConstPrefilterEnv, PushConstSkyBox,
    VkModelPushConsts,
};
use crate::vulkan::vk_types::*;
use crate::vulkan::vk_util;
use ash::vk;
use std::ffi::CStr;

/// Builder for Vulkan graphics pipelines with dynamic rendering.
///
/// ## Purpose
/// Fluent interface for configuring graphics pipeline state. Accumulates state,
/// then creates VkPipeline via build_pipeline().
///
/// ## Dynamic Rendering
/// Uses VkPipelineRenderingCreateInfo (Vulkan 1.3) instead of VkRenderPass.
/// Attachments specified at pipeline creation, can vary at render time.
///
/// ## Vertex Input State
/// Empty (no vertex bindings/attributes). Shaders use vertex pulling via
/// buffer device address from push constants.
///
/// ## Dynamic State
/// Viewport and scissor are dynamic (set via vkCmdSetViewport/Scissor).
/// Allows same pipeline for different resolutions without recreation.
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
    pub color_attachment_count: usize,
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
            color_attachment_count: 1,
        }
    }
}

impl<'a> PipelineBuilder<'a> {
    /// Create VkPipeline from accumulated state.
    ///
    /// ## Logic Flow
    /// 1. Configure viewport state (dynamic, so just counts)
    /// 2. Configure color blending from color_blend_attachment
    /// 3. Empty vertex input (vertex pulling pattern)
    /// 4. Set dynamic states (VIEWPORT, SCISSOR)
    /// 5. Assemble VkGraphicsPipelineCreateInfo with push_next for dynamic rendering
    /// 6. Call vkCreateGraphicsPipelines
    ///
    /// ## Dynamic Rendering Integration
    /// render_info (VkPipelineRenderingCreateInfo) linked via push_next.
    /// Specifies color/depth attachment formats without VkRenderPass.
    ///
    /// ## Why Empty Vertex Input
    /// Vertex pulling: shaders fetch from buffer via device address. No need to
    /// declare vertex attributes in pipeline.
    pub fn build_pipeline(&mut self, device: &ash::Device) -> Result<vk::Pipeline, String> {
        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);

        let color_blending = vk::PipelineColorBlendStateCreateInfo::default()
            .logic_op_enable(false)
            .logic_op(vk::LogicOp::COPY)
            .attachments(&self.color_blend_attachment[..self.color_attachment_count]);

        // Empty vertex input state (vertex pulling via buffer device address)
        let vertex_input_info = vk::PipelineVertexInputStateCreateInfo::default();

        let state = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_info = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&state);

        let mut render_info = self
            .render_info
            .color_attachment_formats(&self.color_attachment_format[..self.color_attachment_count]);

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
        self.color_attachment_count = 1;

        // self.render_info = vk::PipelineRenderingCreateInfo::default()
        //     .color_attachment_formats(&self.color_attachment_format);
        //
        // println!(":? {:?}", self.color_attachment_format);
        // println!(":? {:?}", self.render_info);
        //
        self
    }

    pub fn disable_color_attachments(mut self) -> Self {
        self.color_attachment_count = 0;
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

// ---------------------------------------------------------------------------
// PipelineSpec — centralised creation boundary
// ---------------------------------------------------------------------------

/// Immutable specification for a single graphics pipeline.
///
/// Every pipeline constructor maps its parameters into this value and calls
/// [`create_pipeline_from_spec`]. The spec is validated before any Vulkan call;
/// contradictory attachment/depth combos are rejected.
#[derive(Clone)]
pub(crate) struct PipelineSpec {
    pub vert_module: vk::ShaderModule,
    pub frag_module: vk::ShaderModule,
    pub topology: vk::PrimitiveTopology,
    pub polygon_mode: vk::PolygonMode,
    pub cull_mode: vk::CullModeFlags,
    pub front_face: vk::FrontFace,
    pub color_attachment_format: Option<vk::Format>,
    pub depth_format: Option<vk::Format>,
    /// (write_enable, compare_op). `None` means depth test disabled.
    pub depth_test: Option<(bool, vk::CompareOp)>,
    pub blend: BlendingMode,
    pub layout: vk::PipelineLayout,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum BlendingMode {
    Disabled,
    AlphaBlend,
}

impl PipelineSpec {
    fn validate(&self) -> Result<(), String> {
        if self.vert_module == vk::ShaderModule::null() {
            return Err("pipeline vertex shader module is null".to_string());
        }
        if self.frag_module == vk::ShaderModule::null() {
            return Err("pipeline fragment shader module is null".to_string());
        }
        if self.layout == vk::PipelineLayout::null() {
            return Err("pipeline layout is null".to_string());
        }
        if self.color_attachment_format == Some(vk::Format::UNDEFINED) {
            return Err("color attachment format must not be UNDEFINED".to_string());
        }
        if self.depth_format == Some(vk::Format::UNDEFINED) {
            return Err("depth attachment format must not be UNDEFINED".to_string());
        }
        if self.depth_format.is_some() && self.depth_test.is_none() {
            return Err(
                "depth attachment format is set but depth test is disabled; \
                 either remove the depth format or enable depth testing"
                    .to_string(),
            );
        }
        if self.depth_test.is_some() && self.depth_format.is_none() {
            return Err(
                "depth test is enabled but no depth attachment format was provided".to_string(),
            );
        }
        if self.color_attachment_format.is_none() && self.depth_format.is_none() {
            return Err(
                "pipeline must have at least one color or depth attachment format".to_string(),
            );
        }
        Ok(())
    }
}

/// Build a Vulkan pipeline from a validated [`PipelineSpec`].
///
/// The caller owns the pipeline layout and is responsible for destroying it on
/// rollback. On success, the returned `vk::Pipeline` is fully created.
pub(crate) fn create_pipeline_from_spec(
    device: &ash::Device,
    spec: &PipelineSpec,
) -> Result<vk::Pipeline, String> {
    spec.validate()?;

    let entry = c"main";

    let mut builder = PipelineBuilder::default()
        .set_shaders(spec.vert_module, entry, spec.frag_module, entry)
        .set_input_topology(spec.topology)
        .set_polygon_mode(spec.polygon_mode)
        .set_cull_mode(spec.cull_mode, spec.front_face)
        .set_multisample_none()
        .set_pipeline_layout(spec.layout);

    match spec.color_attachment_format {
        Some(fmt) => {
            builder = builder.set_color_attachment_format(fmt);
        }
        None => {
            builder = builder.disable_color_attachments();
        }
    }

    match spec.depth_format {
        Some(fmt) => {
            builder = builder.set_depth_format(fmt);
        }
        None => {}
    }

    match spec.depth_test {
        Some((write_enable, compare_op)) => {
            builder = builder.enable_depth_test(write_enable, compare_op);
        }
        None => {
            builder = builder.disable_depth_test();
        }
    }

    match spec.blend {
        BlendingMode::Disabled => {
            builder = builder.disable_blending();
        }
        BlendingMode::AlphaBlend => {
            builder = builder.enable_blending_alpha_blend();
        }
    }

    builder.build_pipeline(device)
}

/// Create two pipelines that share one layout, with atomic rollback.
///
/// If the second pipeline fails, the first pipeline *and* the shared layout
/// are destroyed before the error propagates.
pub(crate) fn create_pipeline_pair(
    device: &ash::Device,
    spec_a: &PipelineSpec,
    spec_b: &PipelineSpec,
) -> Result<(VkPipeline, VkPipeline), String> {
    if spec_a.layout != spec_b.layout {
        unsafe {
            device.destroy_pipeline_layout(spec_a.layout, None);
            if spec_b.layout != spec_a.layout {
                device.destroy_pipeline_layout(spec_b.layout, None);
            }
        }
        return Err("pipeline pair specs must share the same layout".to_string());
    }

    let pipeline_a = create_pipeline_from_spec(device, spec_a).map_err(|err| {
        // Layout was never consumed; destroy it.
        unsafe { device.destroy_pipeline_layout(spec_a.layout, None) };
        err
    })?;

    let pipeline_b = create_pipeline_from_spec(device, spec_b).map_err(|err| {
        unsafe {
            device.destroy_pipeline(pipeline_a, None);
            device.destroy_pipeline_layout(spec_a.layout, None);
        }
        err
    })?;

    Ok((
        VkPipeline::new(pipeline_a, spec_a.layout),
        VkPipeline::new(pipeline_b, spec_b.layout),
    ))
}

pub fn init_pipeline_cache(
    device: &ash::Device,
    desc_layout_cache: &VkDescLayoutCache,
    shader_cache: &VkShaderCache,
    draw_color_format: vk::Format,
    draw_depth_format: vk::Format,
) -> Result<VkPipelineCache, String> {
    let (pbr_opaque, pbr_alpha) = init_met_rough_pipelines(
        device,
        desc_layout_cache,
        shader_cache,
        draw_color_format,
        draw_depth_format,
    )?;
    let (unlit_opaque, unlit_alpha) = init_unlit_pipelines(
        device,
        desc_layout_cache,
        shader_cache,
        draw_color_format,
        draw_depth_format,
    )?;

    let brd_flut_pipeline = init_brd_flut_pipeline(
        device,
        desc_layout_cache,
        shader_cache,
        draw_color_format,
        draw_depth_format,
    )?;

    let skybox_pipeline = init_skybox_pipeline(
        device,
        desc_layout_cache,
        shader_cache,
        draw_color_format,
        draw_depth_format,
    )?;

    let env_irradiance_pipeline =
        init_irradiance_pipeline(device, desc_layout_cache, shader_cache)?;

    let env_prefilter_pipeline = init_pre_filter_pipeline(device, desc_layout_cache, shader_cache)?;

    let env_equirect_pipeline =
        init_equirect_to_cube_pipeline(device, desc_layout_cache, shader_cache)?;

    let shadow_depth_pipeline = init_shadow_depth_pipeline(device, shader_cache)?;

    #[allow(unused_mut)]
    let mut pipelines = vec![
        (VkPipelineType::PbrMetRoughOpaque, pbr_opaque),
        (VkPipelineType::PbrMetRoughAlpha, pbr_alpha),
        (VkPipelineType::UnlitOpaque, unlit_opaque),
        (VkPipelineType::UnlitAlpha, unlit_alpha),
        (VkPipelineType::BrdfLut, brd_flut_pipeline),
        (VkPipelineType::Skybox, skybox_pipeline),
        (VkPipelineType::EnvIrradiance, env_irradiance_pipeline),
        (VkPipelineType::EnvPreFilter, env_prefilter_pipeline),
        (VkPipelineType::EnvEquirectToCube, env_equirect_pipeline),
        (VkPipelineType::ShadowDepth, shadow_depth_pipeline),
    ];

    #[cfg(feature = "instancing")]
    {
        let (pbr_instanced, unlit_instanced) = init_instanced_pipelines(
            device,
            desc_layout_cache,
            shader_cache,
            draw_color_format,
            draw_depth_format,
        )?;
        pipelines.push((VkPipelineType::PbrMetRoughOpaqueInstanced, pbr_instanced));
        pipelines.push((VkPipelineType::UnlitOpaqueInstanced, unlit_instanced));
    }

    VkPipelineCache::new(pipelines)
}

fn init_met_rough_pipelines(
    device: &ash::Device,
    desc_layout_cache: &VkDescLayoutCache,
    shader_cache: &VkShaderCache,
    color_format: vk::Format,
    depth_format: vk::Format,
) -> Result<(VkPipeline, VkPipeline), String> {
    let vert_shader = shader_cache.get_core_shader(CoreShaderType::MetRoughVert);
    let frag_shader = shader_cache.get_core_shader(CoreShaderType::MetRoughFrag);

    let matrix_range = [vk::PushConstantRange::default()
        .offset(0)
        .size(std::mem::size_of::<VkModelPushConsts>() as u32)
        .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)];

    let layouts = [
        desc_layout_cache.get(VkDescType::SceneData),
        desc_layout_cache.get(VkDescType::SkinData),
        desc_layout_cache.get(VkDescType::PbrSamplers),
    ];

    let mesh_layout_info = vk_util::pipeline_layout_create_info()
        .set_layouts(&layouts)
        .push_constant_ranges(&matrix_range);

    let layout = unsafe { device.create_pipeline_layout(&mesh_layout_info, None) }
        .map_err(|err| format!("failed to create PBR pipeline layout: {err:?}"))?;

    let opaque_spec = PipelineSpec {
        vert_module: vert_shader,
        frag_module: frag_shader,
        topology: vk::PrimitiveTopology::TRIANGLE_LIST,
        polygon_mode: vk::PolygonMode::FILL,
        cull_mode: vk::CullModeFlags::NONE,
        front_face: vk::FrontFace::CLOCKWISE,
        color_attachment_format: Some(color_format),
        depth_format: Some(depth_format),
        depth_test: Some((true, vk::CompareOp::LESS_OR_EQUAL)),
        blend: BlendingMode::Disabled,
        layout,
    };

    let alpha_spec = PipelineSpec {
        blend: BlendingMode::AlphaBlend,
        depth_test: Some((false, vk::CompareOp::LESS_OR_EQUAL)),
        ..opaque_spec.clone()
    };

    create_pipeline_pair(device, &opaque_spec, &alpha_spec)
}

fn init_unlit_pipelines(
    device: &ash::Device,
    desc_layout_cache: &VkDescLayoutCache,
    shader_cache: &VkShaderCache,
    color_format: vk::Format,
    depth_format: vk::Format,
) -> Result<(VkPipeline, VkPipeline), String> {
    let vert_shader = shader_cache.get_core_shader(CoreShaderType::MetRoughVert);
    let frag_shader = shader_cache.get_core_shader(CoreShaderType::MetRoughFragUnlit);

    let matrix_range = [vk::PushConstantRange::default()
        .offset(0)
        .size(std::mem::size_of::<VkModelPushConsts>() as u32)
        .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)];

    let layouts = [
        desc_layout_cache.get(VkDescType::SceneData),
        desc_layout_cache.get(VkDescType::SkinData),
        desc_layout_cache.get(VkDescType::PbrSamplers),
    ];

    let mesh_layout_info = vk_util::pipeline_layout_create_info()
        .set_layouts(&layouts)
        .push_constant_ranges(&matrix_range);

    let layout = unsafe { device.create_pipeline_layout(&mesh_layout_info, None) }
        .map_err(|err| format!("failed to create unlit pipeline layout: {err:?}"))?;

    let opaque_spec = PipelineSpec {
        vert_module: vert_shader,
        frag_module: frag_shader,
        topology: vk::PrimitiveTopology::TRIANGLE_LIST,
        polygon_mode: vk::PolygonMode::FILL,
        cull_mode: vk::CullModeFlags::NONE,
        front_face: vk::FrontFace::CLOCKWISE,
        color_attachment_format: Some(color_format),
        depth_format: Some(depth_format),
        depth_test: Some((true, vk::CompareOp::LESS_OR_EQUAL)),
        blend: BlendingMode::Disabled,
        layout,
    };

    let alpha_spec = PipelineSpec {
        blend: BlendingMode::AlphaBlend,
        depth_test: Some((false, vk::CompareOp::LESS_OR_EQUAL)),
        ..opaque_spec.clone()
    };

    create_pipeline_pair(device, &opaque_spec, &alpha_spec)
}

fn init_brd_flut_pipeline(
    device: &ash::Device,
    desc_layout_cache: &VkDescLayoutCache,
    shader_cache: &VkShaderCache,
    color_format: vk::Format,
    _depth_format: vk::Format,
) -> Result<VkPipeline, String> {
    let vert_shader = shader_cache.get_core_shader(CoreShaderType::BrtFlutVert);
    let frag_shader = shader_cache.get_core_shader(CoreShaderType::BrtFlutFrag);

    let layouts = [desc_layout_cache.get(VkDescType::Empty)];
    let mesh_layout_info = vk_util::pipeline_layout_create_info().set_layouts(&layouts);
    let layout = unsafe { device.create_pipeline_layout(&mesh_layout_info, None) }
        .map_err(|err| format!("failed to create BRDF LUT pipeline layout: {err:?}"))?;

    let spec = PipelineSpec {
        vert_module: vert_shader,
        frag_module: frag_shader,
        topology: vk::PrimitiveTopology::TRIANGLE_LIST,
        polygon_mode: vk::PolygonMode::FILL,
        cull_mode: vk::CullModeFlags::NONE,
        front_face: vk::FrontFace::CLOCKWISE,
        color_attachment_format: Some(color_format),
        depth_format: None,
        depth_test: None,
        blend: BlendingMode::Disabled,
        layout,
    };

    let pipeline = create_pipeline_from_spec(device, &spec).map_err(|err| {
        unsafe { device.destroy_pipeline_layout(layout, None) };
        err
    })?;

    Ok(VkPipeline::new(pipeline, layout))
}

fn init_skybox_pipeline(
    device: &ash::Device,
    desc_layout_cache: &VkDescLayoutCache,
    shader_cache: &VkShaderCache,
    color_format: vk::Format,
    _depth_format: vk::Format,
) -> Result<VkPipeline, String> {
    let vert_shader = shader_cache.get_core_shader(CoreShaderType::SkyBoxVert);
    let frag_shader = shader_cache.get_core_shader(CoreShaderType::SkyBoxFrag);

    let push_const_size = std::mem::size_of::<PushConstSkyBox>() as u32;
    debug_assert!(
        push_const_size <= 256,
        "PushConstSkyBox size {} exceeds 256 bytes",
        push_const_size
    );

    let push_constant_range = [vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
        .offset(0)
        .size(push_const_size)];

    let layouts = [desc_layout_cache.get(VkDescType::Skybox)];
    let layout_info = vk_util::pipeline_layout_create_info()
        .set_layouts(&layouts)
        .push_constant_ranges(&push_constant_range);
    let layout = unsafe { device.create_pipeline_layout(&layout_info, None) }
        .map_err(|err| format!("failed to create skybox pipeline layout: {err:?}"))?;

    let spec = PipelineSpec {
        vert_module: vert_shader,
        frag_module: frag_shader,
        topology: vk::PrimitiveTopology::TRIANGLE_LIST,
        polygon_mode: vk::PolygonMode::FILL,
        cull_mode: vk::CullModeFlags::NONE,
        front_face: vk::FrontFace::CLOCKWISE,
        color_attachment_format: Some(color_format),
        depth_format: None,
        depth_test: None,
        blend: BlendingMode::Disabled,
        layout,
    };

    let pipeline = create_pipeline_from_spec(device, &spec).map_err(|err| {
        unsafe { device.destroy_pipeline_layout(layout, None) };
        err
    })?;

    Ok(VkPipeline::new(pipeline, layout))
}

fn init_irradiance_pipeline(
    device: &ash::Device,
    desc_layout_cache: &VkDescLayoutCache,
    shader_cache: &VkShaderCache,
) -> Result<VkPipeline, String> {
    let vert_shader = shader_cache.get_core_shader(CoreShaderType::CubeFilterVert);
    let frag_shader = shader_cache.get_core_shader(CoreShaderType::EnvIrradianceFrag);

    let push_constant_range = [vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
        .offset(0)
        .size(std::mem::size_of::<PushConstIrradiance>() as u32)];

    let layouts = [desc_layout_cache.get(VkDescType::EnvIrradiance)];
    let layout_info = vk_util::pipeline_layout_create_info()
        .set_layouts(&layouts)
        .push_constant_ranges(&push_constant_range);
    let layout = unsafe { device.create_pipeline_layout(&layout_info, None) }
        .map_err(|err| format!("failed to create irradiance pipeline layout: {err:?}"))?;

    let spec = PipelineSpec {
        vert_module: vert_shader,
        frag_module: frag_shader,
        topology: vk::PrimitiveTopology::TRIANGLE_LIST,
        polygon_mode: vk::PolygonMode::FILL,
        cull_mode: vk::CullModeFlags::NONE,
        front_face: vk::FrontFace::CLOCKWISE,
        color_attachment_format: Some(vk::Format::R32G32B32A32_SFLOAT),
        depth_format: None,
        depth_test: None,
        blend: BlendingMode::Disabled,
        layout,
    };

    let pipeline = create_pipeline_from_spec(device, &spec).map_err(|err| {
        unsafe { device.destroy_pipeline_layout(layout, None) };
        err
    })?;

    Ok(VkPipeline::new(pipeline, layout))
}

fn init_pre_filter_pipeline(
    device: &ash::Device,
    desc_layout_cache: &VkDescLayoutCache,
    shader_cache: &VkShaderCache,
) -> Result<VkPipeline, String> {
    let vert_shader = shader_cache.get_core_shader(CoreShaderType::CubeFilterVert);
    let frag_shader = shader_cache.get_core_shader(CoreShaderType::EnvPrefilterFrag);

    let push_constant_range = [vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
        .offset(0)
        .size(std::mem::size_of::<PushConstPrefilterEnv>() as u32)];

    let layouts = [desc_layout_cache.get(VkDescType::EnvPreFilter)];
    let layout_info = vk_util::pipeline_layout_create_info()
        .set_layouts(&layouts)
        .push_constant_ranges(&push_constant_range);
    let layout = unsafe { device.create_pipeline_layout(&layout_info, None) }
        .map_err(|err| format!("failed to create prefilter pipeline layout: {err:?}"))?;

    let spec = PipelineSpec {
        vert_module: vert_shader,
        frag_module: frag_shader,
        topology: vk::PrimitiveTopology::TRIANGLE_LIST,
        polygon_mode: vk::PolygonMode::FILL,
        cull_mode: vk::CullModeFlags::NONE,
        front_face: vk::FrontFace::CLOCKWISE,
        color_attachment_format: Some(vk::Format::R16G16B16A16_SFLOAT),
        depth_format: None,
        depth_test: None,
        blend: BlendingMode::Disabled,
        layout,
    };

    let pipeline = create_pipeline_from_spec(device, &spec).map_err(|err| {
        unsafe { device.destroy_pipeline_layout(layout, None) };
        err
    })?;

    Ok(VkPipeline::new(pipeline, layout))
}

fn init_equirect_to_cube_pipeline(
    device: &ash::Device,
    desc_layout_cache: &VkDescLayoutCache,
    shader_cache: &VkShaderCache,
) -> Result<VkPipeline, String> {
    let vert_shader = shader_cache.get_core_shader(CoreShaderType::CubeFilterVert);
    let frag_shader = shader_cache.get_core_shader(CoreShaderType::EnvEquirectToCubeFrag);

    let push_constant_range = [vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
        .offset(0)
        .size(std::mem::size_of::<PushConstCubeCapture>() as u32)];

    let layouts = [desc_layout_cache.get(VkDescType::EnvEquirect)];
    let layout_info = vk_util::pipeline_layout_create_info()
        .set_layouts(&layouts)
        .push_constant_ranges(&push_constant_range);
    let layout = unsafe { device.create_pipeline_layout(&layout_info, None) }
        .map_err(|err| format!("failed to create equirect pipeline layout: {err:?}"))?;

    let spec = PipelineSpec {
        vert_module: vert_shader,
        frag_module: frag_shader,
        topology: vk::PrimitiveTopology::TRIANGLE_LIST,
        polygon_mode: vk::PolygonMode::FILL,
        cull_mode: vk::CullModeFlags::NONE,
        front_face: vk::FrontFace::CLOCKWISE,
        color_attachment_format: Some(vk::Format::R32G32B32A32_SFLOAT),
        depth_format: None,
        depth_test: None,
        blend: BlendingMode::Disabled,
        layout,
    };

    let pipeline = create_pipeline_from_spec(device, &spec).map_err(|err| {
        unsafe { device.destroy_pipeline_layout(layout, None) };
        err
    })?;

    Ok(VkPipeline::new(pipeline, layout))
}

/// Push constants for shadow depth pass (per-draw data).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PushConstShadowDepth {
    pub light_model_view_projection: glam::Mat4,
    pub vertex_buffer_addr: vk::DeviceAddress,
    pub _pad: [u32; 2],
}

fn init_shadow_depth_pipeline(
    device: &ash::Device,
    shader_cache: &VkShaderCache,
) -> Result<VkPipeline, String> {
    let vert_shader = shader_cache.get_core_shader(CoreShaderType::ShadowDepthVert);
    let frag_shader = shader_cache.get_core_shader(CoreShaderType::ShadowDepthFrag);

    let push_const_size = std::mem::size_of::<PushConstShadowDepth>() as u32;
    let push_constant_range = [vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::VERTEX)
        .offset(0)
        .size(push_const_size)];

    let layout_info = vk_util::pipeline_layout_create_info()
        .set_layouts(&[])
        .push_constant_ranges(&push_constant_range);
    let layout = unsafe { device.create_pipeline_layout(&layout_info, None) }
        .map_err(|err| format!("failed to create shadow pipeline layout: {err:?}"))?;

    let spec = PipelineSpec {
        vert_module: vert_shader,
        frag_module: frag_shader,
        topology: vk::PrimitiveTopology::TRIANGLE_LIST,
        polygon_mode: vk::PolygonMode::FILL,
        cull_mode: vk::CullModeFlags::NONE,
        front_face: vk::FrontFace::CLOCKWISE,
        color_attachment_format: None,
        depth_format: Some(vk::Format::D32_SFLOAT),
        depth_test: Some((true, vk::CompareOp::LESS_OR_EQUAL)),
        blend: BlendingMode::Disabled,
        layout,
    };

    let pipeline = create_pipeline_from_spec(device, &spec).map_err(|err| {
        unsafe { device.destroy_pipeline_layout(layout, None) };
        err
    })?;

    Ok(VkPipeline::new(pipeline, layout))
}

// ---------------------------------------------------------------------------
// Instanced pipeline initialization (behind `instancing` feature flag)
// ---------------------------------------------------------------------------

#[cfg(feature = "instancing")]
fn init_instanced_pipelines(
    device: &ash::Device,
    desc_layout_cache: &VkDescLayoutCache,
    shader_cache: &VkShaderCache,
    color_format: vk::Format,
    depth_format: vk::Format,
) -> Result<(VkPipeline, VkPipeline), String> {
    let vert_shader = shader_cache.get_core_shader(CoreShaderType::MetRoughInstancedVert);
    let pbr_frag = shader_cache.get_core_shader(CoreShaderType::MetRoughFrag);
    let unlit_frag = shader_cache.get_core_shader(CoreShaderType::MetRoughFragUnlit);

    let push_const_size = 32u32;
    let push_constant_range = [vk::PushConstantRange::default()
        .offset(0)
        .size(push_const_size)
        .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)];

    let layouts = [
        desc_layout_cache.get(VkDescType::SceneDataInstanced),
        desc_layout_cache.get(VkDescType::SkinData),
        desc_layout_cache.get(VkDescType::PbrSamplers),
    ];

    let mesh_layout_info = vk_util::pipeline_layout_create_info()
        .set_layouts(&layouts)
        .push_constant_ranges(&push_constant_range);

    let layout = unsafe { device.create_pipeline_layout(&mesh_layout_info, None) }
        .map_err(|err| format!("failed to create instanced pipeline layout: {err:?}"))?;

    let pbr_spec = PipelineSpec {
        vert_module: vert_shader,
        frag_module: pbr_frag,
        topology: vk::PrimitiveTopology::TRIANGLE_LIST,
        polygon_mode: vk::PolygonMode::FILL,
        cull_mode: vk::CullModeFlags::NONE,
        front_face: vk::FrontFace::CLOCKWISE,
        color_attachment_format: Some(color_format),
        depth_format: Some(depth_format),
        depth_test: Some((true, vk::CompareOp::LESS_OR_EQUAL)),
        blend: BlendingMode::Disabled,
        layout,
    };

    let unlit_spec = PipelineSpec {
        frag_module: unlit_frag,
        ..pbr_spec.clone()
    };

    create_pipeline_pair(device, &pbr_spec, &unlit_spec)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ash::vk::Handle;

    fn valid_spec() -> PipelineSpec {
        PipelineSpec {
            vert_module: vk::ShaderModule::from_raw(0x101),
            frag_module: vk::ShaderModule::from_raw(0x102),
            topology: vk::PrimitiveTopology::TRIANGLE_LIST,
            polygon_mode: vk::PolygonMode::FILL,
            cull_mode: vk::CullModeFlags::NONE,
            front_face: vk::FrontFace::CLOCKWISE,
            color_attachment_format: Some(vk::Format::R8G8B8A8_UNORM),
            depth_format: None,
            depth_test: None,
            blend: BlendingMode::Disabled,
            layout: vk::PipelineLayout::from_raw(0x103),
        }
    }

    #[test]
    fn pipeline_spec_accepts_color_only_pipeline() {
        assert!(valid_spec().validate().is_ok());
    }

    #[test]
    fn pipeline_spec_rejects_depth_state_mismatches_before_vulkan() {
        let mut depth_without_test = valid_spec();
        depth_without_test.depth_format = Some(vk::Format::D32_SFLOAT);
        assert!(depth_without_test.validate().is_err());

        let mut test_without_depth = valid_spec();
        test_without_depth.depth_test = Some((true, vk::CompareOp::LESS_OR_EQUAL));
        assert!(test_without_depth.validate().is_err());
    }

    #[test]
    fn pipeline_spec_rejects_missing_attachments_and_invalid_handles() {
        let mut no_attachments = valid_spec();
        no_attachments.color_attachment_format = None;
        assert!(no_attachments.validate().is_err());

        let mut undefined_color = valid_spec();
        undefined_color.color_attachment_format = Some(vk::Format::UNDEFINED);
        assert!(undefined_color.validate().is_err());

        let mut null_vert = valid_spec();
        null_vert.vert_module = vk::ShaderModule::null();
        assert!(null_vert.validate().is_err());

        let mut null_layout = valid_spec();
        null_layout.layout = vk::PipelineLayout::null();
        assert!(null_layout.validate().is_err());
    }
}
