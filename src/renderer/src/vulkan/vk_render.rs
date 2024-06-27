use crate::data::gltf_util;
use crate::data::gltf_util::MeshAsset;
use crate::data::primitives::Vertex;
use crate::vulkan;
use ash::vk::{
    AllocationCallbacks, ExtendsPhysicalDeviceFeatures2, ImageLayout, PipelineCache,
    ShaderStageFlags,
};
use ash::{vk, Device};
use glam::{vec3, Vec4};
use gltf::accessor::Dimensions::Mat4;
use imgui_winit_support::{HiDpiMode, WinitPlatform};
use std::ffi::{CStr, CString};
use std::mem::align_of;
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use std::time::SystemTime;
use vk_mem::{AllocationCreateFlags, Allocator, AllocatorCreateInfo};

use crate::vulkan::vk_descriptor::*;
use crate::vulkan::vk_types::*;
use crate::vulkan::{vk_descriptor, vk_init, vk_pipeline, vk_types, vk_util};

pub struct VkRender {
    pub window: winit::window::Window,
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
    pub scene_data: SceneData,
    pub mesh: Option<VkGpuMeshBuffers>,
    pub meshes: Option<Vec<Rc<MeshAsset>>>,
    pub main_deletion_queue: Vec<Box<dyn VkDestroyable>>,
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
) -> VkPipeline {
    let vert_shader = vk_util::load_shader_module(
        device,
        "/home/mindspice/code/rust/engine/src/renderer/src/shaders/colored_triangle.vert.spv",
    )
    .expect("Error loading shader");

    let frag_shader = vk_util::load_shader_module(
        device,
        "/home/mindspice/code/rust/engine/src/renderer/src/shaders/colored_triangle.frag.spv",
    )
    .expect("Error loading shader");

    let buffer_range = [vk::PushConstantRange::default()
        .offset(0)
        .size(std::mem::size_of::<VkGpuPushConsts>() as u32)
        .stage_flags(vk::ShaderStageFlags::VERTEX)];

    let pipeline_info = vk::PipelineLayoutCreateInfo::default().push_constant_ranges(&buffer_range);

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
        .enable_blending_additive()
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

pub fn init_scene_data(device: &LogicalDevice, layout: &VkDescriptors) -> SceneData {
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

    let mut scene_data = SceneData::default();
    scene_data.effects.push(gradient);
    scene_data.effects.push(sky);

    scene_data
}

impl Drop for VkRender {
    fn drop(&mut self) {}
    
       // unsafe {
        //     self.logical_device
        //         .device
        //         .device_wait_idle()
        //         .expect("Render drop failed waiting for device idle");
        // 
        //     self.imgui.renderer.destroy();
        // 
        //     self.logical_device
        //         .device
        //         .destroy_command_pool(self.immediate.command_pool.pool, None);
        //     self.logical_device
        //         .device
        //         .destroy_fence(self.immediate.fence[0], None);
        // 
        //     // self.logical_device
        //     //     .device
        //     //     .destroy_pipeline_layout(self.pipeline.pipeline_layout, None);
        //     // self.logical_device
        //     //     .device
        //     //     .destroy_pipeline(self.pipeline.pipeline, None);
        // 
        //     for x in 0..self.presentation.descriptors.descriptor_layouts.len() {
        //         self.logical_device.device.destroy_descriptor_set_layout(
        //             self.presentation.descriptors.descriptor_layouts[x],
        //             None,
        //         );
        //     }
        // 
        //     self.presentation
        //         .descriptors
        //         .allocator
        //         .destroy(&self.logical_device);
        // 
        //     self.presentation.draw_images.iter_mut().for_each(|item| {
        //         self.logical_device
        //             .device
        //             .destroy_image_view(item.image_view, None);
        // 
        //         self.allocator
        //             .lock()
        //             .unwrap()
        //             .destroy_image(item.image, &mut item.allocation);
        //     });
        // 
        //     self.presentation
        //         .present_images
        //         .iter()
        //         .for_each(|(image, view)| {
        //             self.logical_device.device.destroy_image_view(*view, None)
        //         });
        // 
        //     self.main_deletion_queue.iter_mut().for_each(|item| {
        //         item.destroy(&self.logical_device.device, &self.allocator.lock().unwrap())
        //     });
        // 
        //     self.presentation.frame_sync.iter().for_each(|b| {
        //         self.logical_device
        //             .device
        //             .destroy_semaphore(b.swap_semaphore, None);
        //         self.logical_device
        //             .device
        //             .destroy_semaphore(b.render_semaphore, None);
        //         self.logical_device
        //             .device
        //             .destroy_fence(b.render_fence, None);
        //     });
        // 
        //     // todo need to do some work on destruction
        //     self.logical_device
        //         .device
        //         .destroy_command_pool(self.presentation.command_pool.pool, None);
        // 
        //     self.swapchain
        //         .swapchain_loader
        //         .destroy_swapchain(self.swapchain.swapchain, None);
        // 
        //     self.logical_device.device.destroy_device(None);
        // 
        //     self.surface
        //         .surface_instance
        //         .destroy_surface(self.surface.surface, None);
        // 
        //     if let Some(debug) = &self.debug {
        //         debug
        //             .debug_utils
        //             .destroy_debug_utils_messenger(debug.debug_callback, None); // None == custom allocator
        //     }
        //     self.instance.destroy_instance(None); // None == allocator callback
        // }
  //  }
}
impl VkRender {
    fn destroy(&mut self) {
        unsafe { std::mem::drop(self) }
    }

    pub fn new(
        window: winit::window::Window,
        size: (u32, u32),
        with_validation: bool,
    ) -> Result<Self, String> {
        let entry = vk_init::init_entry();

        let mut instance_ext = vk_init::get_winit_extensions(&window);
        let (instance, debug) = vk_init::init_instance(
            &entry,
            "test".to_string(),
            &mut instance_ext,
            with_validation,
        )?;

        let surface = vk_init::get_window_surface(&entry, &instance, &window)?;

        let physical_device = vk_init::get_physical_devices(
            &instance,
            Some(&surface),
            &vk_init::simple_device_suitability,
        )?
        .remove(0);

        let queue_indices =
            vk_init::queue_indices_with_preferences(&instance, &physical_device.p_device, &surface, true, true)?;

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
            size,
            Some(2),
            None,
            Some(vk::PresentModeKHR::IMMEDIATE),
            None,
            true,
        )?;

        let  command_pools = vk_init::create_command_pools(&logical_device, 2, 1)?;

        let frame_buffers: Vec<VkFrameSync> = (0..2)
            .map(|_| vk_init::create_frame_buffer(&logical_device))
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

        let draw_images = vk_init::allocate_draw_images(&allocator, &logical_device, size, 2)?;
        let draw_format = draw_images[0].image_format;

        let draw_views: Vec<vk::ImageView> =
            draw_images.iter().map(|data| data.image_view).collect();

        let present_images = vk_init::create_basic_present_views(&logical_device, &swapchain)?;

        let descriptors = init_descriptors(&logical_device, &draw_views);
        let layout = [descriptors.descriptor_layouts[0]];

        let depth_images = vk_init::allocate_depth_images(&allocator, &logical_device, size, 2)?;
        let depth_format = depth_images[0].image_format;

        // FIXME, this needs generalized
        let scene_data = init_scene_data(&logical_device, &descriptors);

        let presentation = VkPresent::new(
            frame_buffers,
            draw_images,
            depth_images,
            present_images,
            command_pools,
            descriptors,
        ).unwrap();

        //create command pool for immediate usage 
        // TODO this selection needs cleaned up, or we need to properly include all the of
        // flags a queue support not just the ones we are using it for
        let im_index = queue_indices
            .iter()
            .find(|q| {
            q.queue_types.contains(&QueueType::Graphics)
            })
            .unwrap();

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
        platform.attach_window(imgui_context.io_mut(), &window, HiDpiMode::Default);

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

        let mesh_pipeline = init_mesh_pipeline(&logical_device, draw_format, depth_format);

        pipeline_cache.add_pipeline(VkPipelineType::MESH, mesh_pipeline);

        Ok(VkRender {
            window,
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
            scene_data,
            main_deletion_queue: Vec::new(),
            mesh: None,
            meshes: None,
        })
    }
}

impl VkRender {
    pub fn render(&mut self, frame_number: u32) {
        let start = SystemTime::now();
        let frame_data = self.presentation.get_next_frame();
        let frame_sync = frame_data.sync;
        let cmd_pool = frame_data.cmd_pool.get(QueueType::Graphics);
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

            let acquire = vk::AcquireNextImageInfoKHR::default()
                .swapchain(self.swapchain.swapchain)
                .semaphore(frame_sync.swap_semaphore)
                .device_mask(1)
                .timeout(u32::MAX as u64);

            let (image_index, _) = self
                .swapchain
                .swapchain_loader
                .acquire_next_image2(&acquire)
                .unwrap();

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

            let draw_image = frame_data.draw.image;
            let draw_view = frame_data.draw.image_view;
            let depth_image = frame_data.depth.image;
            let depth_view = frame_data.depth.image_view;
            let present_image = frame_data.present_image;
            let present_view = frame_data.present_image_view;

            println!(
                "On frame: {:?}",
                self.swapchain.swapchain_images[image_index as usize]
            );
            println!("present Image: {:?}", present_image);
            println!("present view: {:?}", present_view);
            println!("render Image: {:?}", draw_image);
            println!("render View: {:?}", draw_view);

            let extent = self.swapchain.extent;

            vk_util::transition_image(
                &self.logical_device.device,
                cmd_buffer,
                draw_image,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::GENERAL,
            );

            //

            let ds = [self
                .scene_data
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

            if self.meshes.is_none() {
                self.init_default_data();
            }

            let meshes = if let Some(meshes) = &self.meshes {
                meshes
            } else {
                panic!()
            };

            vk_util::transition_image(
                &self.logical_device.device,
                cmd_buffer,
                depth_image,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL,
            );

            self.draw_geometry(cmd_buffer, draw_view, depth_view, meshes);

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

            &self
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
            &self
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

            self.swapchain
                .swapchain_loader
                .queue_present(queue, &present_info)
                .unwrap();
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

        let render_info = vk_util::rendering_info(self.swapchain.extent, &attachment_info, &[]);

        unsafe {
            self.logical_device
                .device
                .cmd_begin_rendering(cmd_buffer, &render_info);
        }

        let mut selected = self.scene_data.get_current_effect();

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
        &self,
        cmd_buffer: vk::CommandBuffer,
        draw_view: vk::ImageView,
        depth_view: vk::ImageView,
        model_buffers: &[Rc<MeshAsset>],
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

        let extent = self.swapchain.extent;

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

            let mut push_constants = VkGpuPushConsts::new(
                projection * view,
                model_buffers[2].mesh_buffers.vertex_buffer_addr,
            );

            self.logical_device.device.cmd_push_constants(
                cmd_buffer,
                self.pipeline_cache
                    .get_unchecked(VkPipelineType::MESH)
                    .pipeline_layout,
                vk::ShaderStageFlags::VERTEX,
                0,
                &push_constants.as_byte_slice(),
            );

            self.logical_device.device.cmd_bind_index_buffer(
                cmd_buffer,
                model_buffers[2].mesh_buffers.index_buffer.buffer,
                0,
                vk::IndexType::UINT32,
            );

            println!("Surfaces{:?}", model_buffers[2].surfaces);

            self.logical_device.device.cmd_draw_indexed(
                cmd_buffer,
                model_buffers[2].surfaces[0].count,
                1,
                model_buffers[2].surfaces[0].start_index,
                0,
                0,
            );

            self.logical_device.device.cmd_end_rendering(cmd_buffer);
        }
    }

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

            // Check the fence status before submission
            let fence_status_before = self
                .logical_device
                .device
                .get_fence_status(self.immediate.fence[0]);
            match fence_status_before {
                Ok(status) => println!("Fence status before submit: {:?}", status),
                Err(e) => println!("Error getting fence status before submit: {:?}", e),
            }

            // Submit the command buffer and signal the fence correctly
            self.logical_device
                .device
                .queue_submit2(queue, &submit_info, self.immediate.fence[0])
                .unwrap();

            // Check the fence status after submission
            let fence_status_after = self
                .logical_device
                .device
                .get_fence_status(self.immediate.fence[0]);
            match fence_status_after {
                Ok(status) => println!("Fence status after submit: {:?}", status),
                Err(e) => println!("Error getting fence status after submit: {:?}", e),
            }

            // Wait for the fence to be signaled
            self.logical_device
                .device
                .wait_for_fences(&self.immediate.fence, true, u64::MAX)
                .unwrap();

            // Check the fence status after waiting
            let fence_status_final = self
                .logical_device
                .device
                .get_fence_status(self.immediate.fence[0]);
            match fence_status_final {
                Ok(status) => println!("Fence status after waiting: {:?}", status),
                Err(e) => println!("Error getting fence status after waiting: {:?}", e),
            }
        }
    }

    pub fn draw_background(
        &self,
        image: vk::Image,
        cmd_buffer: vk::CommandBuffer,
        descriptor_set: &[vk::DescriptorSet],
    ) {
        let compute_effect = self.scene_data.get_current_effect();
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
                (self.swapchain.extent.width as f32 / 16.0).ceil() as u32,
                (self.swapchain.extent.height as f32 / 16.0).ceil() as u32,
                1,
            );
        }
    }

    pub fn upload_mesh(&mut self, indices: &[u32], vertices: &[Vertex]) -> VkGpuMeshBuffers {
        let i_buffer_size = indices.len() * std::mem::size_of::<u32>();
        let v_buffer_size = vertices.len() * std::mem::size_of::<Vertex>();

        let index_buffer = vk_util::allocate_buffer(
            &self.allocator,
            i_buffer_size,
            vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            vk_mem::MemoryUsage::AutoPreferDevice,
        )
        .expect("Failed to allocate index buffer");

        let vertex_buffer = vk_util::allocate_buffer(
            &self.allocator,
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
            &self.allocator,
            v_buffer_size + i_buffer_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk_mem::MemoryUsage::AutoPreferHost,
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

        vk_util::destroy_buffer(&self.allocator, staging_buffer);

        new_surface
    }

    pub fn init_default_data(&mut self) {
        self.mesh = None;
        self.meshes = Some(
            gltf_util::load_meshes(
                "/home/mindspice/code/rust/engine/src/renderer/src/assets/basicmesh.glb",
                |indices, vertices| self.upload_mesh(indices, vertices),
            )
            .unwrap(),
        );
    }
}
