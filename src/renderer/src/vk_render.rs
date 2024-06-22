use crate::vk_descriptor::{DescriptorAllocator, DescriptorLayoutBuilder, PoolSizeRatio};
use crate::vk_init::*;
use crate::vk_types::*;
use crate::{vk_init, vk_util};
use ash::vk::{AllocationCallbacks, ExtendsPhysicalDeviceFeatures2, ImageLayout};
use ash::{vk, Device};
use imgui_winit_support::{HiDpiMode, WinitPlatform};
use std::ffi::{CStr, CString};
use std::sync::{Arc, Mutex};
use std::time::SystemTime;
use vk_mem::{AllocationCreateFlags, Allocator, AllocatorCreateInfo};

pub struct VkDescriptors {
    pub allocator: DescriptorAllocator,
    pub descriptor_sets: Vec<vk::DescriptorSet>,
    pub descriptor_layouts: Vec<vk::DescriptorSetLayout>,
}

impl VkDescriptors {
    pub fn new(allocator: DescriptorAllocator) -> Self {
        Self {
            allocator,
            descriptor_sets: vec![],
            descriptor_layouts: vec![],
        }
    }

    pub fn add_descriptor(&mut self, set: vk::DescriptorSet, layout: vk::DescriptorSetLayout) {
        self.descriptor_sets.push(set);
        self.descriptor_layouts.push(layout);
    }
}

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
    pub pipeline: VkPipeline,
    pub immediate: VkImmediate,
    pub imgui: VkImgui,

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

pub fn init_background_pipeline(
    device: &LogicalDevice,
    layout: &[vk::DescriptorSetLayout],
) -> VkPipeline {
    let compute_info = vk::PipelineLayoutCreateInfo::default().set_layouts(layout);

    let compute_layout = unsafe {
        device
            .device
            .create_pipeline_layout(&compute_info, None)
            .unwrap()
    };

    let shader_module = vk_util::load_shader_module(
        device,
        "/home/mindspice/code/rust/engine/src/renderer/src/shaders/gradient.comp.spv",
    )
    .unwrap();

    let name = CString::new("main").unwrap();
    let stage_info = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(shader_module)
        .name(&name);

    let pipeline_info = [vk::ComputePipelineCreateInfo::default()
        .layout(compute_layout)
        .stage(stage_info)];

    let compute_pipeline = unsafe {
        device
            .device
            .create_compute_pipelines(vk::PipelineCache::null(), &pipeline_info, None)
            .unwrap()[0]
    };

    unsafe {
        device.device.destroy_shader_module(shader_module, None);
    }
    VkPipeline::new(compute_pipeline, compute_layout)
}



impl Drop for VkRender {
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

            self.logical_device
                .device
                .destroy_pipeline_layout(self.pipeline.pipeline_layout, None);
            self.logical_device
                .device
                .destroy_pipeline(self.pipeline.pipeline, None);

            for x in 0..self.presentation.descriptors.descriptor_layouts.len() {
                self.logical_device.device.destroy_descriptor_set_layout(
                    self.presentation.descriptors.descriptor_layouts[x],
                    None,
                );
            }

            self.presentation
                .descriptors
                .allocator
                .destroy(&self.logical_device);

            self.presentation.image_alloc.iter_mut().for_each(|item| {
                self.logical_device
                    .device
                    .destroy_image_view(item.image_view, None);

                self.allocator
                    .lock()
                    .unwrap()
                    .destroy_image(item.image, &mut item.allocation);
            });

            self.presentation
                .present_images
                .iter()
                .for_each(|(image, view)| {
                    self.logical_device.device.destroy_image_view(*view, None)
                });

            self.main_deletion_queue.iter_mut().for_each(|item| {
                item.destroy(&self.logical_device.device, &self.allocator.lock().unwrap())
            });

            self.presentation.frame_sync.iter().for_each(|b| {
                self.logical_device
                    .device
                    .destroy_semaphore(b.swap_semaphore, None);
                self.logical_device
                    .device
                    .destroy_semaphore(b.render_semaphore, None);
                self.logical_device
                    .device
                    .destroy_fence(b.render_fence, None);
            });

            // todo need to do some work on destruction
            self.logical_device
                .device
                .destroy_command_pool(self.presentation.command_pool.pool, None);

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
            vk_init::graphics_only_queue_indices(&instance, &physical_device.p_device, &surface)?;

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
            Some(vk::PresentModeKHR::MAILBOX),
            None,
            true,
        )?;

        let mut command_pools = vk_init::create_command_pools(&logical_device, 2)?;

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

        let allocator = unsafe {
            Arc::new(Mutex::new(
                Allocator::new(allocator_info).map_err(|err| "Failed to initialize allocator")?,
            ))
        };

        let image_data = vk_init::allocate_basic_images(&allocator, &logical_device, size, 2)?;
        let present_images = vk_init::create_basic_present_views(&logical_device, &swapchain)?;

        let render_views: Vec<vk::ImageView> =
            image_data.iter().map(|data| data.image_view).collect();

        let descriptors = init_descriptors(&logical_device, &render_views);
        let layout = [descriptors.descriptor_layouts[0]];

        let presentation = VkPresent::new(
            frame_buffers,
            image_data,
            present_images,
            command_pools.remove(0),
            descriptors,
        );

        // FIXME, this needs generalized
        let pipeline = init_background_pipeline(&logical_device, &layout);

        //create command pool for immediate rendering, this may need moved to an init
        let im_command_pool = vk_init::create_command_pools(&logical_device, 1)
            .unwrap()
            .remove(0);
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
            enable_depth_test: false,
            enable_depth_write: false,
        };

        let imgui_dynamic = imgui_rs_vulkan_renderer::DynamicRendering {
            color_attachment_format: vk::Format::B8G8R8A8_UNORM,
            depth_attachment_format: None,
        };
        let imgui_render = imgui_rs_vulkan_renderer::Renderer::with_vk_mem_allocator(
            allocator.clone(),
            logical_device.device.clone(),
            presentation.command_pool.queue,
            presentation.command_pool.pool,
            // im_command_pool.queue,
            // im_command_pool.pool,
            imgui_dynamic,
            &mut imgui_context,
            Some(imgui_opts),
        )
        .unwrap();

        let immediate = VkImmediate::new(im_command_pool, im_fence);

        let imgui = VkImgui::new(imgui_context, platform, imgui_render);

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
            pipeline,
            immediate,
            imgui,
            main_deletion_queue: Vec::new(),
        })
    }
}

impl VkRender {
    pub fn render(&mut self, frame_number: u32) {
        let start = SystemTime::now();
        let frame_data = self.presentation.get_next_frame();
        let frame_sync = frame_data.sync;
        let cmd_buffer = frame_data.cmd_buffer;
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

            let r_image = frame_data.render_image;
            let r_image_view = frame_data.render_image_view;
            let p_image = frame_data.present_image;
            let p_image_view = frame_data.present_image_view;

            let extent = self.swapchain.extent;

            vk_util::transition_image(
                &self.logical_device.device,
                cmd_buffer,
                r_image,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::GENERAL,
            );

            // Draw to render image
            self.draw_background(
                r_image,
                cmd_buffer,
                &frame_data.desc_set,
                self.pipeline.pipeline_layout,
                self.pipeline.pipeline,
            );

            //transition render image for copy from
            vk_util::transition_image(
                &self.logical_device.device,
                cmd_buffer,
                r_image,
                vk::ImageLayout::GENERAL,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            );

            // transition present image to copy to
            vk_util::transition_image(
                &self.logical_device.device,
                cmd_buffer,
                p_image,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            );

            //copy render image onto present image
            vk_util::blit_copy_image_to_image(
                &self.logical_device.device,
                cmd_buffer,
                r_image,
                extent,
                p_image,
                extent,
            );

            vk_util::transition_image(
                &self.logical_device.device,
                cmd_buffer,
                p_image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            );

            self.draw_imgui(cmd_buffer, p_image_view);

            // transition swapchain to present
            vk_util::transition_image(
                &self.logical_device.device,
                cmd_buffer,
                p_image,
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
                .queue_submit2(frame_data.cmd_queue, &submit, frame_sync.render_fence)
                .unwrap();

            let r_sem = [frame_sync.render_semaphore];
            let imf_idex = [image_index];
            let present_info = vk::PresentInfoKHR::default()
                .swapchains(&swapchain)
                .wait_semaphores(&r_sem)
                .image_indices(&imf_idex);

            self.swapchain
                .swapchain_loader
                .queue_present(frame_data.cmd_queue, &present_info)
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

        self.imgui.context.new_frame().show_demo_window(&mut self.imgui.opened);
        let draw_data = self.imgui.context.render();

        self.imgui.renderer.cmd_draw(cmd_buffer, draw_data).unwrap();

        unsafe {
            self.logical_device.device.cmd_end_rendering(cmd_buffer);
        }
    }

    pub fn immediate_submit<F>(&mut self, function: F)
    where
        F: FnOnce(vk::CommandBuffer),
    {
        unsafe {
            let cmd_buffer = self.immediate.command_pool.buffers[0];
            let queue = self.immediate.command_pool.queue;
            let graphics_queue = self
                .logical_device
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

            self.logical_device
                .device
                .queue_submit2(queue, &submit_info, self.immediate.fence[0])
                .unwrap();

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
        pipeline_layout: vk::PipelineLayout,
        pipeline: vk::Pipeline,
    ) {
        unsafe {
            self.logical_device.device.cmd_bind_pipeline(
                cmd_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline.pipeline,
            );
            self.logical_device.device.cmd_bind_descriptor_sets(
                cmd_buffer,
                vk::PipelineBindPoint::COMPUTE,
                pipeline_layout,
                0,
                descriptor_set,
                &vec![], // only need if dynamic offsets are used, if none make empty
            );
            self.logical_device.device.cmd_dispatch(
                cmd_buffer,
                (self.swapchain.extent.width as f32 / 16.0).ceil() as u32,
                (self.swapchain.extent.height as f32 / 16.0).ceil() as u32,
                1,
            );
        }
    }
}
