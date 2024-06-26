use crate::data::gltf_util::MeshAsset;
use crate::data::gpu_data::{
    GLTFMetallicRoughness, GLTFMetallicRoughnessConstants, GLTFMetallicRoughnessResources,
    GPUScene, GPUSceneData, MaterialInstance, MaterialPass, Vertex,
};
use crate::data::{data_util, gltf_util};
use crate::vulkan;
use ash::prelude::VkResult;
use ash::vk::{
    AllocationCallbacks, DescriptorSetLayoutCreateFlags, ExtendsPhysicalDeviceFeatures2, Extent2D,
    ImageLayout, PipelineCache, ShaderStageFlags,
};
use ash::{vk, Device};
use data_util::PackUnorm;
use glam::{vec3, Vec4};
use gltf::accessor::Dimensions::Mat4;
use imgui_winit_support::{HiDpiMode, WinitPlatform};
use std::ffi::{CStr, CString};
use std::mem::align_of;
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use std::thread::sleep;
use std::time::{Duration, SystemTime};
use vk_mem::{AllocationCreateFlags, Allocator, AllocatorCreateInfo};

use crate::vulkan::vk_descriptor::*;
use crate::vulkan::vk_types::*;
use crate::vulkan::{vk_descriptor, vk_init, vk_pipeline, vk_types, vk_util};

pub struct DefaultTextures {
    white_image: VkImageAlloc,
    black_image: VkImageAlloc,
    grey_image: VkImageAlloc,
    error_image: VkImageAlloc,
    linear_sampler: vk::Sampler,
    nearest_sampler: vk::Sampler,
    image_descriptor: vk::DescriptorSetLayout,
}

pub struct GltfStuffs<'a> {
    pub default_data: MaterialInstance,
    pub metal_roughness: GLTFMetallicRoughness<'a>,
}

pub struct VkRender<'a> {
    pub window_state: VkWindowState,
    pub allocator: Arc<Mutex<Allocator>>,
    pub entry: ash::Entry,
    pub instance: ash::Instance,
    pub debug: Option<VkDebug>,
    pub physical_device: PhyDevice,
    pub logical_device: LogicalDevice,
    pub surface: VkSurface,
    pub swapchain: VkSwapchain,
    //   pub present_images: Vec<vk::ImageView>,
    // pub command_pools: Vec<VkCommandPool>,
    pub presentation: VkPresent,
    pub pipeline_cache: VkPipelineCache,
    pub immediate: VkImmediate,
    pub imgui: VkImgui,
    pub compute_data: ComputeData,
    pub scene_data: GPUScene,
    pub mesh: Option<VkGpuMeshBuffers>,
    pub meshes: Option<Vec<Rc<MeshAsset>>>,
    pub default_textures: Option<DefaultTextures>,
    pub descriptors: VkDescLayoutMap,
    pub global_desc_allocator: VkDynamicDescriptorAllocator,
    pub gltf_stuffs: Option<GltfStuffs<'a>>,
    pub resize_requested: bool,
    pub main_deletion_queue: Vec<VkDeletable>,
}

pub fn init_desc_map(device: &LogicalDevice) -> VkDescLayoutMap {
    let draw_image_layout = DescriptorLayoutBuilder::default()
        .add_binding(0, vk::DescriptorType::STORAGE_IMAGE)
        .build(
            device,
            vk::ShaderStageFlags::COMPUTE,
            DescriptorSetLayoutCreateFlags::empty(),
        )
        .unwrap();

    let gpu_scene_desc = DescriptorLayoutBuilder::default()
        .add_binding(0, vk::DescriptorType::UNIFORM_BUFFER)
        .build(
            device,
            vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
            DescriptorSetLayoutCreateFlags::empty(),
        )
        .unwrap();

    VkDescLayoutMap::new(vec![
        (VkDescType::DrawImage, draw_image_layout),
        (VkDescType::GpuScene, gpu_scene_desc),
    ])
}

pub fn init_image_descriptor(device: &LogicalDevice) -> vk::DescriptorSetLayout {
    DescriptorLayoutBuilder::default()
        .add_binding(0, vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .build(
            device,
            vk::ShaderStageFlags::FRAGMENT,
            vk::DescriptorSetLayoutCreateFlags::empty(),
        )
        .unwrap()
}

pub fn init_descriptors(device: &LogicalDevice, image_views: &[vk::ImageView]) -> VkDescriptors {
    let sizes = [PoolSizeRatio::new(vk::DescriptorType::STORAGE_IMAGE, 1.0)];

    let alloc = DescriptorAllocator::new(&device, 10, &sizes).unwrap();

    let mut descriptors = VkDescriptors::new(alloc);
    for view in image_views {
        let render_layout = [DescriptorLayoutBuilder::default()
            .add_binding(0, vk::DescriptorType::STORAGE_IMAGE)
            .build(
                &device,
                vk::ShaderStageFlags::COMPUTE,
                vk::DescriptorSetLayoutCreateFlags::empty(),
            )
            .unwrap()];

        let render_desc = descriptors
            .allocator
            .allocate(&device, &render_layout)
            .unwrap();

        let image_info = [vk::DescriptorImageInfo::default()
            .image_layout(vk::ImageLayout::GENERAL)
            .image_view(*view)];

        let image_write_desc = [vk::WriteDescriptorSet::default()
            .dst_binding(0)
            .dst_set(render_desc)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .image_info(&image_info)];

        unsafe {
            device
                .device
                .update_descriptor_sets(&image_write_desc, &vec![])
        }
        descriptors.add_descriptor(render_desc, render_layout[0])
    }

    descriptors
}

pub fn init_mesh_pipeline(
    device: &LogicalDevice,
    draw_format: vk::Format,
    depth_format: vk::Format,
    image_desc: vk::DescriptorSetLayout,
) -> VkPipeline {
    let vert_shader = vk_util::load_shader_module(
        device,
        "/home/mindspice/code/rust/engine/src/renderer/src/shaders/colored_triangle.vert.spv",
    )
    .expect("Error loading shader");

    let frag_shader = vk_util::load_shader_module(
        device,
        "/home/mindspice/code/rust/engine/src/renderer/src/shaders/tex_image.frag.spv",
    )
    .expect("Error loading shader");

    let buffer_range = [vk::PushConstantRange::default()
        .offset(0)
        .size(std::mem::size_of::<VkGpuPushConsts>() as u32)
        .stage_flags(vk::ShaderStageFlags::VERTEX)];

    let binding = [image_desc];
    let pipeline_info = vk::PipelineLayoutCreateInfo::default()
        .push_constant_ranges(&buffer_range)
        .set_layouts(&binding);

    let pipeline_layout = unsafe {
        device
            .device
            .create_pipeline_layout(&pipeline_info, None)
            .unwrap()
    };

    println!("Created layout: {:?}", pipeline_layout);

    let entry = CString::new("main").unwrap();

    let pipeline = vk_pipeline::PipelineBuilder::default()
        .set_pipeline_layout(pipeline_layout)
        .set_shaders(vert_shader, &entry, frag_shader, &entry)
        .set_input_topology(vk::PrimitiveTopology::TRIANGLE_LIST)
        .set_polygon_mode(vk::PolygonMode::FILL)
        .set_cull_mode(vk::CullModeFlags::NONE, vk::FrontFace::CLOCKWISE)
        .set_multisample_none()
        .disable_blending()
        .enable_depth_test(true, vk::CompareOp::GREATER_OR_EQUAL)
        .set_color_attachment_format(draw_format)
        .set_depth_format(depth_format)
        .build_pipeline(device)
        .unwrap();

    unsafe {
        device.device.destroy_shader_module(vert_shader, None);
        device.device.destroy_shader_module(frag_shader, None);
    }

    VkPipeline::new(pipeline, pipeline_layout)
}

pub fn init_scene_data(device: &LogicalDevice, layout: &VkDescriptors) -> ComputeData {
    let desc_layout = &layout.descriptor_layouts;

    let push_constants = [vk::PushConstantRange::default()
        .offset(0)
        .size(std::mem::size_of::<Compute4x4PushConstants>() as u32)
        .stage_flags(vk::ShaderStageFlags::COMPUTE)];

    let compute_info = vk::PipelineLayoutCreateInfo::default()
        .set_layouts(desc_layout)
        .push_constant_ranges(&push_constants);

    let compute_layout = unsafe {
        device
            .device
            .create_pipeline_layout(&compute_info, None)
            .unwrap()
    };

    let gradient_shader = vk_util::load_shader_module(
        device,
        "/home/mindspice/code/rust/engine/src/renderer/src/shaders/gradient_color.comp.spv",
    )
    .expect("Error loading shader");

    let sky_shader = vk_util::load_shader_module(
        device,
        "/home/mindspice/code/rust/engine/src/renderer/src/shaders/sky.comp.spv",
    )
    .expect("Error loading shader");

    let name = CString::new("main").unwrap();
    let stage_info = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(gradient_shader)
        .name(&name);

    let pipeline_info = [vk::ComputePipelineCreateInfo::default()
        .layout(compute_layout)
        .stage(stage_info)];

    let gradient_data = Compute4x4PushConstants::default()
        .set_data_1(glam::vec4(1.0, 0.0, 0.0, 1.0))
        .set_data_2(glam::vec4(0.0, 0.0, 1.0, 1.0));

    let gradient_pipeline = unsafe {
        device
            .device
            .create_compute_pipelines(vk::PipelineCache::null(), &pipeline_info, None)
            .unwrap()[0]
    };

    let gradient = ComputeEffect {
        name: "gradient".to_string(),
        pipeline: gradient_pipeline,
        layout: compute_layout,
        descriptors: layout.clone(),
        data: gradient_data,
    };

    let stage_info = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(sky_shader)
        .name(&name);

    let pipeline_info = [vk::ComputePipelineCreateInfo::default()
        .layout(compute_layout)
        .stage(stage_info)];

    let sky_pipeline = unsafe {
        device
            .device
            .create_compute_pipelines(vk::PipelineCache::null(), &pipeline_info, None)
            .unwrap()[0]
    };

    let sky_data =
        Compute4x4PushConstants::default().set_data_1(glam::Vec4::new(0.1, 0.2, 0.4, 0.97));

    let sky = ComputeEffect {
        name: "sky".to_string(),
        pipeline: sky_pipeline,
        layout: compute_layout,
        descriptors: layout.clone(),
        data: sky_data,
    };

    unsafe {
        device.device.destroy_shader_module(gradient_shader, None);
        device.device.destroy_shader_module(sky_shader, None);
    }

    let mut scene_data = ComputeData::default();
    scene_data.effects.push(gradient);
    scene_data.effects.push(sky);

    scene_data
}

impl<'a> Drop for VkRender<'a> {
    fn drop(&mut self) {
        unsafe {
            self.logical_device
                .device
                .device_wait_idle()
                .expect("Render drop failed waiting for device idle");

            self.imgui.renderer.destroy();

            self.logical_device
                .device
                .destroy_command_pool(self.immediate.command_pool.pool, None);
            self.logical_device
                .device
                .destroy_fence(self.immediate.fence[0], None);

            // self.logical_device
            //     .device
            //     .destroy_pipeline_layout(self.pipeline.pipeline_layout, None);
            // self.logical_device
            //     .device
            //     .destroy_pipeline(self.pipeline.pipeline, None);

            self.presentation
                .destroy(&self.logical_device.device, &self.allocator.lock().unwrap());

            self.main_deletion_queue.iter_mut().for_each(|mut del| {
                del.delete(&self.logical_device.device, &self.allocator.lock().unwrap())
            });

            // todo need to do some work on destruction

            self.swapchain
                .swapchain_loader
                .destroy_swapchain(self.swapchain.swapchain, None);

            self.logical_device.device.destroy_device(None);

            self.surface
                .surface_instance
                .destroy_surface(self.surface.surface, None);

            if let Some(debug) = &self.debug {
                debug
                    .debug_utils
                    .destroy_debug_utils_messenger(debug.debug_callback, None); // None == custom allocator
            }
            self.instance.destroy_instance(None); // None == allocator callback
        }
    }
}

impl<'a> VkRender<'a> {
    fn destroy(&mut self) {
        unsafe { std::mem::drop(self) }
    }

    pub fn new(mut window_state: VkWindowState, with_validation: bool) -> Result<Self, String> {
        let entry = vk_init::init_entry();

        let mut instance_ext = vk_init::get_winit_extensions(&window_state.window);
        let (instance, debug) = vk_init::init_instance(
            &entry,
            "test".to_string(),
            &mut instance_ext,
            with_validation,
        )?;

        let surface = vk_init::get_window_surface(&entry, &instance, &window_state.window)?;

        let physical_device = vk_init::get_physical_devices(
            &instance,
            Some(&surface),
            &vk_init::simple_device_suitability,
        )?
        .remove(0);

        let queue_indices = vk_init::queue_indices_with_preferences(
            &instance,
            &physical_device.p_device,
            &surface,
            true,
            true,
        )?;

        let mut core_features =
            vk_init::get_general_core_features(&instance, &physical_device.p_device);
        let vk11_features = vk_init::get_general_v11_features(&instance, &physical_device.p_device);
        let vk12_features = vk_init::get_general_v12_features(&instance, &physical_device.p_device);
        let vk13_features = vk_init::get_general_v13_features(&instance, &physical_device.p_device);

        let mut ext_feats: Vec<Box<dyn ExtendsPhysicalDeviceFeatures2>> = vec![
            Box::new(vk11_features),
            Box::new(vk12_features),
            Box::new(vk13_features),
        ];

        // FIXME better extension init
        let mut surface_ext = vk_init::get_basic_device_ext_ptrs();
        // let ext =
        //     unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_swapchain_mutable_format\0") };
        // surface_ext.push(ext.as_ptr());

        let logical_device = vk_init::create_logical_device(
            &instance,
            &physical_device.p_device,
            &queue_indices,
            &mut core_features,
            Some(&mut ext_feats),
            Some(&surface_ext),
        )?;

        let swapchain_support =
            vk_init::get_swapchain_support(&physical_device.p_device, &surface)?;

        let swapchain = vk_init::create_swapchain(
            &instance,
            &physical_device,
            &logical_device,
            &surface,
            window_state.curr_extent,
            Some(2),
            None,
            Some(vk::PresentModeKHR::IMMEDIATE),
            None,
            true,
        )?;

        if swapchain.extent != window_state.curr_extent {
            window_state.curr_extent = swapchain.extent;
        }

        let command_pools = vk_init::create_command_pools(&logical_device, 2, 1)?;

        let frame_buffers: Vec<VkFrameSync> = (0..2)
            .map(|_| vk_init::create_frame_sync(&logical_device))
            .collect::<Result<Vec<_>, _>>()?;

        // TODO will have to separate compute pools when we get to that stage
        //  currently we share present and graphics and just have a simple pool returned
        //  it needs to be ironed out how to manage separate pools in relation, maybe just
        //  store them all in the frame buffer?

        let mut allocator_info =
            AllocatorCreateInfo::new(&instance, &logical_device.device, physical_device.p_device);

        allocator_info.vulkan_api_version = vk::API_VERSION_1_3;
        allocator_info.flags = vk_mem::AllocatorCreateFlags::BUFFER_DEVICE_ADDRESS;

        let allocator = unsafe {
            Arc::new(Mutex::new(
                Allocator::new(allocator_info).map_err(|err| "Failed to initialize allocator")?,
            ))
        };

        // Set images to max extent, so they can be reused on window resizing

        let draw_images =
            vk_init::allocate_draw_images(&allocator, &logical_device, window_state.max_extent, 2)?;
        let draw_format = draw_images[0].image_format;

        let draw_views: Vec<vk::ImageView> =
            draw_images.iter().map(|data| data.image_view).collect();

        let present_images = vk_init::create_basic_present_views(&logical_device, &swapchain)?;

        let descriptors = init_descriptors(&logical_device, &draw_views);
        let layout = [descriptors.descriptor_layouts[0]];

     
        let depth_images = vk_init::allocate_depth_images(
            &allocator,
            &logical_device,
            window_state.max_extent,
            2,
        )?;
        let depth_format = depth_images[0].image_format;

        // FIXME, this needs generalized
        let compute_data = init_scene_data(&logical_device, &descriptors);

        let pool_ratios = [
            PoolSizeRatio::new(vk::DescriptorType::STORAGE_IMAGE, 3.0),
            PoolSizeRatio::new(vk::DescriptorType::STORAGE_BUFFER, 3.0),
            PoolSizeRatio::new(vk::DescriptorType::UNIFORM_BUFFER, 3.0),
            PoolSizeRatio::new(vk::DescriptorType::COMBINED_IMAGE_SAMPLER, 4.0),
        ];
   
        let descriptor_allocators: Vec<VkDynamicDescriptorAllocator> = (0..2)
            .map(|_| VkDynamicDescriptorAllocator::new(&logical_device, 1000, &pool_ratios))
            .collect::<Result<Vec<_>, _>>()
            .unwrap();

        let presentation = VkPresent::new(
            frame_buffers,
            draw_images,
            depth_images,
            present_images,
            command_pools,
            descriptor_allocators,
        )
        .unwrap();

        //create command pool for immediate usage
        // TODO this selection needs cleaned up, or we need to properly include all the of
        //  flags a queue support not just the ones we are using it for, right now selecting graphics
        //  which also supports transfer, but we are also using this as a queue for gui rendering
        //  maybe give the gui its own graphics queue as well, since immediate is mainly for transfer
        //  and should be using its own transfer queue, as we dont need them in the per frame?
        let im_index = queue_indices
            .iter()
            .find(|q| q.queue_types.contains(&QueueType::Graphics))
            .unwrap();

        print!("Queue Types Immediate: {:?}", im_index.queue_types);

        let im_queue = logical_device
            .queues
            .get_queue_by_index(im_index.index)
            .unwrap();

        let im_command_pool = vk_init::create_immediate_command_pool(
            &logical_device.device,
            im_queue,
            im_index.queue_types.clone(),
        )
        .unwrap();

        let im_fence_create_info =
            vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
        let im_fence = unsafe {
            logical_device
                .device
                .create_fence(&im_fence_create_info, None)
        }
        .unwrap();

        // GUI

        let mut imgui_context = imgui::Context::create();

        let mut platform = WinitPlatform::init(&mut imgui_context);
        platform.attach_window(
            imgui_context.io_mut(),
            &window_state.window,
            HiDpiMode::Default,
        );

        let imgui_opts = imgui_rs_vulkan_renderer::Options {
            in_flight_frames: 2,
            ..Default::default()
        };

        let imgui_dynamic = imgui_rs_vulkan_renderer::DynamicRendering {
            color_attachment_format: swapchain.surface_format.format,
            depth_attachment_format: None,
        };
        let imgui_render = imgui_rs_vulkan_renderer::Renderer::with_vk_mem_allocator(
            allocator.clone(),
            logical_device.device.clone(),
            im_command_pool.queue,
            im_command_pool.pool,
            imgui_dynamic,
            &mut imgui_context,
            Some(imgui_opts),
        )
        .unwrap();

        let immediate = VkImmediate::new(im_command_pool, im_fence);

        let imgui = VkImgui::new(imgui_context, platform, imgui_render);

        let mut pipeline_cache = VkPipelineCache::default();

        let image_desc = init_image_descriptor(&logical_device);

        let mesh_pipeline =
            init_mesh_pipeline(&logical_device, draw_format, depth_format, image_desc);

        pipeline_cache.add_pipeline(VkPipelineType::MESH, mesh_pipeline);

        let gpu_descriptor = DescriptorLayoutBuilder::default()
            .add_binding(0, vk::DescriptorType::UNIFORM_BUFFER)
            .build(
                &logical_device,
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                vk::DescriptorSetLayoutCreateFlags::empty(),
            );
        let scene_data = GPUScene::new(gpu_descriptor.unwrap());

        let desc_map = init_desc_map(&logical_device);

        let pool_ratios = [
            PoolSizeRatio::new(vk::DescriptorType::STORAGE_IMAGE, 3.0),
            PoolSizeRatio::new(vk::DescriptorType::STORAGE_BUFFER, 3.0),
            PoolSizeRatio::new(vk::DescriptorType::UNIFORM_BUFFER, 3.0),
            PoolSizeRatio::new(vk::DescriptorType::COMBINED_IMAGE_SAMPLER, 4.0),
        ];

        let global_alloc = VkDynamicDescriptorAllocator::new(
            &logical_device,
            10,
            &pool_ratios,
        ).unwrap();
        let mut render = VkRender {
            window_state,
            allocator,
            entry,
            instance,
            debug,
            physical_device,
            logical_device,
            surface,
            swapchain,
            presentation,
            pipeline_cache,
            immediate,
            imgui,
            compute_data,
            main_deletion_queue: Vec::new(),
            mesh: None,
            meshes: None,
            default_textures: None,
            scene_data,
            descriptors: desc_map,
            gltf_stuffs: None,
            resize_requested: false,
            global_desc_allocator: global_alloc
        };
        render.init_default_data(image_desc);
      //  render.init_gltf_data();
        Ok(render)
    }

    pub fn rebuild_swapchain(&mut self, new_size: Extent2D) {
        println!("Resize Rebuild SwapChain");
        self.window_state.curr_extent = new_size;

        unsafe { self.logical_device.device.device_wait_idle().unwrap() }

        let swapchain = vk_init::create_swapchain(
            &self.instance,
            &self.physical_device,
            &self.logical_device,
            &self.surface,
            new_size,
            Some(2),
            None,
            Some(vk::PresentModeKHR::MAILBOX),
            Some(self.swapchain.swapchain),
            true,
        )
        .unwrap();

        // FIXME, I think we will need to destory the old images view when we reassign
        let present_images =
            vk_init::create_basic_present_views(&self.logical_device, &swapchain).unwrap();
        self.swapchain = swapchain;
        self.presentation.replace_present_images(present_images);
        //self.presentation = presentation;
        self.resize_requested = false;
        println!("Resize Completed")
    }
}

impl<'a> VkRender<'a> {
    pub fn render(&mut self, frame_number: u32) {
        let start = SystemTime::now();

        let mut frame_data = self.presentation.get_next_frame();
        let frame_sync = frame_data.sync;
        let cmd_pool = frame_data.cmd_pool.get(QueueType::Graphics);
        let draw_image = frame_data.draw.image;
        let draw_view = frame_data.draw.image_view;
        let depth_image = frame_data.depth.image;
        let depth_view = frame_data.depth.image_view;
        let present_image = frame_data.present_image;
        let present_view = frame_data.present_image_view;

        let queue = cmd_pool.queue;
        let cmd_buffer = cmd_pool.buffers[0];
        let fence = &[frame_sync.render_fence];

        let swapchain = [self.swapchain.swapchain];

        unsafe {
            self.logical_device
                .device
                .wait_for_fences(fence, true, u32::MAX as u64)
                .unwrap();
            self.logical_device.device.reset_fences(fence).unwrap();

            let curr_frame = self.presentation.get_curr_frame_mut();

            curr_frame
                .process_deletions(&self.logical_device.device, &self.allocator.lock().unwrap());

            curr_frame
                .descriptors
                .clear_pools(&self.logical_device.device)
                .unwrap();

            let acquire_info = vk::AcquireNextImageInfoKHR::default()
                .swapchain(self.swapchain.swapchain)
                .semaphore(frame_sync.swap_semaphore)
                .device_mask(1)
                .timeout(u32::MAX as u64);

            let image_index = match self
                .swapchain
                .swapchain_loader
                .acquire_next_image2(&acquire_info)
            {
                Ok((index, _)) => index,
                Err(_) => {
                    self.resize_requested = true;
                    return;
                }
            };

            self.logical_device
                .device
                .reset_command_buffer(cmd_buffer, vk::CommandBufferResetFlags::empty())
                .unwrap();

            let begin_info = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

            self.logical_device
                .device
                .begin_command_buffer(cmd_buffer, &begin_info)
                .unwrap();

            println!(
                "On frame: {:?}",
                self.swapchain.swapchain_images[image_index as usize]
            );
            println!("present Image: {:?}", present_image);
            println!("present view: {:?}", present_view);
            println!("render Image: {:?}", draw_image);
            println!("render View: {:?}", draw_view);

            let extent = self.window_state.curr_extent;

            vk_util::transition_image(
                &self.logical_device.device,
                cmd_buffer,
                draw_image,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::GENERAL,
            );

            //

            let ds = [self
                .compute_data
                .get_current_effect()
                .descriptors
                .descriptor_sets[image_index as usize]];
            &self.draw_background(draw_image, cmd_buffer, &ds);

            //transition render image for copy from
            vk_util::transition_image(
                &self.logical_device.device,
                cmd_buffer,
                draw_image,
                vk::ImageLayout::GENERAL,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            );

            vk_util::transition_image(
                &self.logical_device.device,
                cmd_buffer,
                depth_image,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL,
            );

            self.draw_geometry(cmd_buffer, draw_view, depth_view);

            // transition present image to copy to
            vk_util::transition_image(
                &self.logical_device.device,
                cmd_buffer,
                draw_image,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            );

            vk_util::transition_image(
                &self.logical_device.device,
                cmd_buffer,
                present_image,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            );

            //copy render image onto present image
            vk_util::blit_copy_image_to_image(
                &self.logical_device.device,
                cmd_buffer,
                draw_image,
                extent,
                present_image,
                extent,
            );

            vk_util::transition_image(
                &self.logical_device.device,
                cmd_buffer,
                present_image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            );

            self.draw_imgui(cmd_buffer, present_view);

            // transition swapchain to present
            vk_util::transition_image(
                &self.logical_device.device,
                cmd_buffer,
                present_image,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                vk::ImageLayout::PRESENT_SRC_KHR,
            );

            self
                .logical_device
                .device
                .end_command_buffer(cmd_buffer)
                .unwrap();

            let cmd_info = [vk_util::command_buffer_submit_info(cmd_buffer)];
            let wait_info = [vk_util::semaphore_submit_info(
                vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT_KHR,
                frame_sync.swap_semaphore,
            )];
            let signal_info = [vk_util::semaphore_submit_info(
                vk::PipelineStageFlags2::ALL_GRAPHICS,
                frame_sync.render_semaphore,
            )];
            let submit = [vk_util::submit_info_2(&cmd_info, &signal_info, &wait_info)];
            self
                .logical_device
                .device
                .queue_submit2(queue, &submit, frame_sync.render_fence)
                .unwrap();

            let r_sem = [frame_sync.render_semaphore];
            let imf_idex = [image_index];

            let present_info = vk::PresentInfoKHR::default()
                .swapchains(&swapchain)
                .wait_semaphores(&r_sem)
                .image_indices(&imf_idex);

            let present_result = self
                .swapchain
                .swapchain_loader
                .queue_present(queue, &present_info);

            if let Err(_) = present_result {
                self.resize_requested = true;
            }
        }
        // println!(
        //     "Render Took: {}ms",
        //     SystemTime::now().duration_since(start).unwrap().as_millis()
        // )
    }

    pub fn draw_imgui(&mut self, cmd_buffer: vk::CommandBuffer, image_view: vk::ImageView) {
        let attachment_info = [vk_util::rendering_attachment_info(
            image_view,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            None,
        )];

        let render_info =
            vk_util::rendering_info(self.window_state.curr_extent, &attachment_info, &[]);

        unsafe {
            self.logical_device
                .device
                .cmd_begin_rendering(cmd_buffer, &render_info);
        }

        let mut selected = self.compute_data.get_current_effect();

        // let frame = self.imgui.context.new_frame();
        // frame.text(&selected.name);

        // let data_1_arr = &mut selected.data.data_1.to_array();
        // let mut data_1 = frame
        //     .input_float4("data1", data_1_arr);
        //
        // if data_1.build() {
        //     selected.data.data_1 = Vec4::from_array(*data_1_arr);
        // }
        //
        //
        //
        // let data_2 = frame
        //     .input_float4("data2", &mut selected.data.data_2.to_array())
        //     .build();
        // let data_3 = frame
        //     .input_float4("data3", &mut selected.data.data_3.to_array())
        //     .build();
        // let data_4 = frame
        //     .input_float4("data4", &mut selected.data.data_4.to_array())
        //     .build();

        //
        // frame.slider(
        //     "Effect Index".to_string(),
        //     0,
        //     (self.scene_data.effects.len() - 1) as u32,
        //     &mut self.scene_data.current,
        // );
        //
        // self.imgui.platform.prepare_render(frame, &self.window);

        self.imgui
            .context
            .new_frame()
            .show_demo_window(&mut self.imgui.opened);

        let draw_data = self.imgui.context.render();

        self.imgui.renderer.cmd_draw(cmd_buffer, draw_data).unwrap();

        unsafe {
            self.logical_device.device.cmd_end_rendering(cmd_buffer);
        }
    }

    pub fn draw_geometry(
        &mut self,
        cmd_buffer: vk::CommandBuffer,
        draw_view: vk::ImageView,
        depth_view: vk::ImageView,
    ) {
        let color_attachment = [vk_util::rendering_attachment_info(
            draw_view,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            None,
        )];

        let depth_attachment = [vk_util::depth_attachment_info(
            depth_view,
            vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL,
        )];

        let extent = self.window_state.curr_extent;

        let rendering_info = vk_util::rendering_info(extent, &color_attachment, &depth_attachment);

        unsafe {
            self.logical_device
                .device
                .cmd_begin_rendering(cmd_buffer, &rendering_info);

            self.logical_device.device.cmd_bind_pipeline(
                cmd_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_cache
                    .get_unchecked(VkPipelineType::MESH)
                    .pipeline,
            );

            let def_text = if let Some(tex) = &self.default_textures {
                tex
            } else {
                panic!("No image desc")
            };

            let def_desc = [def_text.image_descriptor];
            let image_set = [self
                .presentation
                .get_curr_frame_mut()
                .descriptors
                .allocate(&self.logical_device, &def_desc)
                .unwrap()];

            let mut writer = DescriptorWriter::default();
            writer.write_image(
                0,
                def_text.error_image.image_view,
                def_text.nearest_sampler,
                vk::ImageLayout::READ_ONLY_OPTIMAL,
                vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            );

            writer.update_set(&self.logical_device, image_set[0]);

            self.logical_device.device.cmd_bind_descriptor_sets(
                cmd_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_cache
                    .get_unchecked(VkPipelineType::MESH)
                    .layout,
                0,
                &image_set,
                &[],
            );

            let viewport = [vk::Viewport::default()
                .x(0.0)
                .y(0.0)
                .width(extent.width as f32)
                .height(extent.height as f32)
                .min_depth(0.0)
                .max_depth(0.0)];

            self.logical_device
                .device
                .cmd_set_viewport(cmd_buffer, 0, &viewport);

            let scissor = [vk::Rect2D::default()
                .offset(vk::Offset2D::default().y(0).y(0))
                .extent(extent)];

            self.logical_device
                .device
                .cmd_set_scissor(cmd_buffer, 0, &scissor);

            let view = glam::Mat4::from_translation(vec3(0.0, 0.0, -5.0));
            let mut projection = glam::Mat4::perspective_rh(
                70.0_f32.to_radians(),
                extent.width as f32 / extent.height as f32,
                0.1,
                10_000.0,
            );

            projection.y_axis.y *= -1.0;

            let meshes = if let Some(meshes) = &self.meshes {
                meshes
            } else {
                panic!("no meshes")
            };

            let mut push_constants =
                VkGpuPushConsts::new(projection * view, meshes[2].mesh_buffers.vertex_buffer_addr);

            self.logical_device.device.cmd_push_constants(
                cmd_buffer,
                self.pipeline_cache
                    .get_unchecked(VkPipelineType::MESH)
                    .layout,
                vk::ShaderStageFlags::VERTEX,
                0,
                &push_constants.as_byte_slice(),
            );

            self.logical_device.device.cmd_bind_index_buffer(
                cmd_buffer,
                meshes[2].mesh_buffers.index_buffer.buffer,
                0,
                vk::IndexType::UINT32,
            );

            self.logical_device.device.cmd_draw_indexed(
                cmd_buffer,
                meshes[2].surfaces[0].count,
                1,
                meshes[2].surfaces[0].start_index,
                0,
                0,
            );

            let mut gpu_scene_buffer = vk_util::allocate_buffer(
                &self.allocator.lock().unwrap(),
                std::mem::size_of::<GPUSceneData>(),
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                vk_mem::MemoryUsage::AutoPreferDevice,
            )
            .unwrap();

            let curr_frame = self.presentation.get_curr_frame_mut();

            let scene_uniform_data = GPUSceneData::default();

            unsafe {
                let allocator = self.allocator.lock().unwrap();
                let data_ptr = allocator
                    .map_memory(&mut gpu_scene_buffer.allocation)
                    .unwrap() as *mut GPUSceneData;
                data_ptr.write(scene_uniform_data);

                allocator.unmap_memory(&mut gpu_scene_buffer.allocation);
            }

            let global_descriptor = curr_frame
                .descriptors
                .allocate(&self.logical_device, &self.scene_data.descriptor)
                .unwrap();

            let mut writer = DescriptorWriter::default();

            writer.write_buffer(
                0,
                gpu_scene_buffer.buffer,
                std::mem::size_of::<GPUSceneData>(),
                0,
                vk::DescriptorType::UNIFORM_BUFFER,
            );

            writer.update_set(&self.logical_device, global_descriptor);

            curr_frame.add_deletion(VkDeletable::AllocatedBuffer(gpu_scene_buffer));

            self.logical_device.device.cmd_end_rendering(cmd_buffer);
        }
    }

    // TODO decide if this is only used for transfers
    pub fn immediate_submit<F>(&self, function: F)
    where
        F: FnOnce(vk::CommandBuffer),
    {
        unsafe {
            let cmd_buffer = self.immediate.command_pool.buffers[0];
            let queue = self.immediate.command_pool.queue;

            // Reset the fence correctly
            self.logical_device
                .device
                .reset_fences(&self.immediate.fence)
                .unwrap();

            self.logical_device
                .device
                .reset_command_buffer(cmd_buffer, vk::CommandBufferResetFlags::empty())
                .unwrap();

            let begin_info =
                vk_util::command_buffer_begin_info(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

            self.logical_device
                .device
                .begin_command_buffer(cmd_buffer, &begin_info)
                .unwrap();

            function(cmd_buffer);

            self.logical_device
                .device
                .end_command_buffer(cmd_buffer)
                .unwrap();

            let cmd_info = [vk_util::command_buffer_submit_info(cmd_buffer)];
            let submit_info = [vk_util::submit_info_2(&cmd_info, &[], &[])];

            // Submit the command buffer and signal the fence correctly
            self.logical_device
                .device
                .queue_submit2(queue, &submit_info, self.immediate.fence[0])
                .unwrap();

            // Wait for the fence to be signaled
            self.logical_device
                .device
                .wait_for_fences(&self.immediate.fence, true, u64::MAX)
                .unwrap();
        }
    }

    pub fn draw_background(
        &self,
        image: vk::Image,
        cmd_buffer: vk::CommandBuffer,
        descriptor_set: &[vk::DescriptorSet],
    ) {
        let compute_effect = self.compute_data.get_current_effect();
        unsafe {
            self.logical_device.device.cmd_bind_pipeline(
                cmd_buffer,
                vk::PipelineBindPoint::COMPUTE,
                compute_effect.pipeline,
            );

            self.logical_device.device.cmd_bind_descriptor_sets(
                cmd_buffer,
                vk::PipelineBindPoint::COMPUTE,
                compute_effect.layout,
                0,
                &descriptor_set,
                &[],
            );

            self.logical_device.device.cmd_push_constants(
                cmd_buffer,
                compute_effect.layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                compute_effect.data.as_byte_slice(),
            );

            self.logical_device.device.cmd_dispatch(
                cmd_buffer,
                (self.window_state.curr_extent.width as f32 / 16.0).ceil() as u32,
                (self.window_state.curr_extent.height as f32 / 16.0).ceil() as u32,
                1,
            );
        }
    }

    pub fn upload_mesh(&mut self, indices: &[u32], vertices: &[Vertex]) -> VkGpuMeshBuffers {
        let i_buffer_size = indices.len() * std::mem::size_of::<u32>();
        let v_buffer_size = vertices.len() * std::mem::size_of::<Vertex>();
        let allocator = self.allocator.lock().unwrap();

        let index_buffer = vk_util::allocate_buffer(
            &allocator,
            i_buffer_size,
            vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            vk_mem::MemoryUsage::AutoPreferDevice,
        )
        .expect("Failed to allocate index buffer");

        let vertex_buffer = vk_util::allocate_buffer(
            &allocator,
            v_buffer_size,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            vk_mem::MemoryUsage::AutoPreferDevice,
        )
        .expect("Failed to allocate vertex buffer");

        let v_buffer_addr_info =
            vk::BufferDeviceAddressInfo::default().buffer(vertex_buffer.buffer);

        let v_buffer_addr = unsafe {
            self.logical_device
                .device
                .get_buffer_device_address(&v_buffer_addr_info)
        };

        let new_surface = VkGpuMeshBuffers::new(index_buffer, vertex_buffer, v_buffer_addr);

        /*
        With the buffers allocated, we need to write the data into them. For that, we will be using a
        staging buffer. This is a very common pattern with Vulkan. As GPU_ONLY memory can't be written
        on the CPU, we first write the memory on a temporal staging buffer that is CPU writable,
        and then execute a copy command to copy this buffer into the GPU buffers. It's not necessary
        for meshes to use GPU_ONLY vertex buffers, but it's highly recommended unless
        it's something like a CPU-side particle system or other dynamic effects.
         */

        let staging_buffer = vk_util::allocate_buffer(
            &allocator,
            v_buffer_size + i_buffer_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk_mem::MemoryUsage::AutoPreferDevice,
        )
        .expect("Failed to allocate staging buffer");

        let staging_data = staging_buffer.alloc_info.mapped_data as *mut u8;

        unsafe {
            // Copy vertex data to the staging buffer
            std::ptr::copy_nonoverlapping(
                vertices.as_ptr() as *const u8,
                staging_data,
                v_buffer_size,
            );

            // Copy index data to the staging buffer
            std::ptr::copy_nonoverlapping(
                indices.as_ptr() as *const u8,
                staging_data.add(v_buffer_size),
                i_buffer_size,
            );
        }

        self.immediate_submit(|cmd| {
            let vertex_copy = [vk::BufferCopy::default()
                .src_offset(0)
                .dst_offset(0)
                .size(v_buffer_size as vk::DeviceSize)];

            let index_copy = [vk::BufferCopy::default()
                .dst_offset(0)
                .src_offset(v_buffer_size as vk::DeviceSize)
                .size(i_buffer_size as vk::DeviceSize)];

            unsafe {
                self.logical_device.device.cmd_copy_buffer(
                    cmd,
                    staging_buffer.buffer,
                    new_surface.vertex_buffer.buffer,
                    &vertex_copy,
                );
            }

            unsafe {
                self.logical_device.device.cmd_copy_buffer(
                    cmd,
                    staging_buffer.buffer,
                    new_surface.index_buffer.buffer,
                    &index_copy,
                );
            }
        });

        vk_util::destroy_buffer(&allocator, staging_buffer);

        new_surface
    }

    pub fn upload_image(
        &self,
        data: &[u8],
        size: vk::Extent3D,
        format: vk::Format,
        usage_flags: vk::ImageUsageFlags,
        mip_mapped: bool,
    ) -> VkImageAlloc {
        //let data_size = data.len();
        let data_size = size.width * size.height * size.depth * 4;
        let allocator = self.allocator.lock().unwrap();

        let mut upload_buffer = vk_util::allocate_buffer(
            &allocator,
            data_size as usize,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk_mem::MemoryUsage::AutoPreferDevice,
        )
        .unwrap();

        unsafe {
            let data_ptr = allocator.map_memory(&mut upload_buffer.allocation).unwrap();
            std::ptr::copy_nonoverlapping(data.as_ptr(), data_ptr, data_size as usize);
            allocator.unmap_memory(&mut upload_buffer.allocation);
        }

        let new_image = vk_util::create_image(
            &self.logical_device.device,
            &allocator,
            size,
            format,
            usage_flags | vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::TRANSFER_SRC,
            mip_mapped,
        );

        self.immediate_submit(|cmd| {
            // let curr_frame = self.presentation.get_curr_frame();
            vk_util::transition_image(
                &self.logical_device.device,
                cmd,
                new_image.image,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            );

            let copy_region = [vk::BufferImageCopy::default()
                .buffer_offset(0)
                .buffer_row_length(0)
                .buffer_image_height(0)
                .image_subresource(
                    vk::ImageSubresourceLayers::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .mip_level(0)
                        .base_array_layer(0)
                        .layer_count(1),
                )
                .image_extent(size)];

            unsafe {
                self.logical_device.device.cmd_copy_buffer_to_image(
                    cmd,
                    upload_buffer.buffer,
                    new_image.image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &copy_region,
                );
            }

            vk_util::transition_image(
                &self.logical_device.device,
                cmd,
                new_image.image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            );
        });

        vk_util::destroy_buffer(&allocator, upload_buffer);
        new_image
    }

    pub fn init_default_data(&mut self, image_descriptor: vk::DescriptorSetLayout) {
        self.meshes = Some(
            gltf_util::load_meshes(
                "/home/mindspice/code/rust/engine/src/renderer/src/assets/basicmesh.glb",
                |indices, vertices| self.upload_mesh(indices, vertices),
            )
            .unwrap(),
        );

        let white_val = glam::vec4(1.0, 1.0, 1.0, 1.0).pack_unorm_4x8();
        let grey_val = glam::vec4(0.66, 0.66, 0.66, 1.0).pack_unorm_4x8();
        let black_val = glam::vec4(0.0, 0.0, 0.0, 0.0).pack_unorm_4x8();
        let magenta_val = glam::vec4(1.0, 0.0, 1.0, 1.0).pack_unorm_4x8();

        let mut error_val = vec![0_u32; 16 * 16]; // 16x16 checkerboard texture
        for y in 0..16 {
            for x in 0..16 {
                error_val[y * 16 + x] = if ((x % 2) ^ (y % 2)) != 0 {
                    magenta_val
                } else {
                    black_val
                };
            }
        }

        let mut err_bytes: Vec<u8> = error_val
            .iter()
            .flat_map(|&pixel| pixel.to_le_bytes())
            .collect();

        let white_image = self.upload_image(
            &white_val.to_ne_bytes(),
            data_util::EXTENT3D_ONE,
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageUsageFlags::SAMPLED,
            false,
        );

        let grey_image = self.upload_image(
            &grey_val.to_ne_bytes(),
            data_util::EXTENT3D_ONE,
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageUsageFlags::SAMPLED,
            false,
        );

        let black_image = self.upload_image(
            &black_val.to_ne_bytes(),
            data_util::EXTENT3D_ONE,
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageUsageFlags::SAMPLED,
            false,
        );

        let error_image = self.upload_image(
            &err_bytes,
            vk::Extent3D {
                width: 16,
                height: 16,
                depth: 1,
            },
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageUsageFlags::SAMPLED,
            false,
        );

        let mut sampler = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::NEAREST)
            .min_filter(vk::Filter::NEAREST);

        let nearest_sampler = unsafe {
            self.logical_device
                .device
                .create_sampler(&sampler, None)
                .unwrap()
        };

        sampler.mag_filter = vk::Filter::LINEAR;
        sampler.min_filter = vk::Filter::LINEAR;

        let linear_sampler = unsafe {
            self.logical_device
                .device
                .create_sampler(&sampler, None)
                .unwrap()
        };

        self.default_textures = Some(DefaultTextures {
            white_image,
            black_image,
            grey_image,
            error_image,
            linear_sampler,
            nearest_sampler,
            image_descriptor,
        })
    }

    pub fn init_gltf_data(&mut self) {
        let (opaque, transparent) =
            GLTFMetallicRoughness::build_pipelines(&self.logical_device, &self.descriptors);

        let mut roughness = GLTFMetallicRoughness {
            opaque_pipeline: opaque,
            transparent_pipeline: transparent,
            descriptor_layout: [self.descriptors.get(VkDescType::DrawImage)],
            writer: Default::default(),
        };

        let default = if let Some(def) = &self.default_textures {
            def
        } else {
            panic!("Default not inited")
        };

        let mut material_constants = vk_util::allocate_buffer(
            &self.allocator.lock().unwrap(),
            std::mem::size_of::<GLTFMetallicRoughnessConstants>(),
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            vk_mem::MemoryUsage::Auto,
        )
        .unwrap();



        let scene_uniform_data = unsafe {
            let scene_uniform_data = self
                .allocator
                .lock()
                .unwrap()
                .map_memory(&mut material_constants.allocation)
                .unwrap()
                .cast::<GLTFMetallicRoughnessConstants>();

            (*scene_uniform_data).color_factors = Vec4::new(1.0, 1.0, 1.0, 1.0);
            (*scene_uniform_data).metal_rough_factors = Vec4::new(1.0, 0.5, 0.0, 0.0);
            scene_uniform_data
        };

        let white_val = glam::vec4(1.0, 1.0, 1.0, 1.0).pack_unorm_4x8();

        let white_image_1 = self.upload_image(
            &white_val.to_ne_bytes(),
            data_util::EXTENT3D_ONE,
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageUsageFlags::SAMPLED,
            false,
        );

        let white_image_2 = self.upload_image(
            &white_val.to_ne_bytes(),
            data_util::EXTENT3D_ONE,
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageUsageFlags::SAMPLED,
            false,
        );



        let material_resources = GLTFMetallicRoughnessResources {
            color_image: white_image_1,
            color_sampler: default.linear_sampler,
            metal_rough_image: white_image_2,
            metal_rough_sampler: default.linear_sampler,
            data_buffer: material_constants.buffer,
            data_buffer_offset: 0,
        };

        let def_data = roughness.write_material(
            &self.logical_device,
            MaterialPass::MainColor,
            &material_resources,
            &mut self.global_desc_allocator,
        );

        let gltf_data = GltfStuffs {
            default_data: def_data,
            metal_roughness: roughness,
        };
        self.gltf_stuffs = Some(gltf_data);

        // self.main_deletion_queue.push(VkDeletable::AllocatedBuffer(material_constants));
    }
}
