use crate::egui_init::EGUIInstance;
use crate::vk_descriptor::{DescriptorAllocator, DescriptorLayoutBuilder, PoolSizeRatio};
use crate::vk_init::{
    LogicalDevice, PhyDevice, VkDebug, VkDestroyable, VkFrameSync, VkPresent, VkSurface,
    VkSwapchain,
};
use crate::{vk_init, vk_util};
use ash::vk;
use ash::vk::{AllocationCallbacks, ExtendsPhysicalDeviceFeatures2};
use egui_ash_renderer::DynamicRendering;
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

pub struct VkPipeline {
    pub pipeline: vk::Pipeline,
    pub pipeline_layout: vk::PipelineLayout,
}

impl VkPipeline {
    pub fn new(pipeline: vk::Pipeline, pipeline_layout: vk::PipelineLayout) -> Self {
        Self {
            pipeline,
            pipeline_layout,
        }
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
    pub gui: EGUIInstance,
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

        let surface_ext = vk_init::get_basic_device_ext_ptrs();
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

        let pipeline = init_background_pipeline(&logical_device, &layout);

        let dynamic_render = DynamicRendering {
            color_attachment_format: vk::Format::R16G16B16A16_SFLOAT,
            depth_attachment_format: None,
        };

        let egui_opts = egui_ash_renderer::Options {
            in_flight_frames: 2,
            enable_depth_test: false,
            enable_depth_write: false,
            srgb_framebuffer: false,
        };

        let gui = EGUIInstance::new(
            &window,
            allocator.clone(),
            &logical_device,
            dynamic_render,
            egui_opts,
        );

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
            gui,
            main_deletion_queue: Vec::new(),
        })
    }
}

impl VkRender {
    pub fn render(&mut self, frame_number: u32) {
        let start = SystemTime::now();
        let device = &self.logical_device.device;
        let frame_data = self.presentation.get_next_frame();
        let frame_sync = frame_data.sync;
        let cmd_buffer = frame_data.cmd_buffer;
        let fence = &[frame_sync.render_fence];

        let swapchain = [self.swapchain.swapchain];

        unsafe {
            device
                .wait_for_fences(fence, true, u32::MAX as u64)
                .unwrap();
            device.reset_fences(fence).unwrap();

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

            device
                .reset_command_buffer(cmd_buffer, vk::CommandBufferResetFlags::empty())
                .unwrap();

            let begin_info = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

            device
                .begin_command_buffer(cmd_buffer, &begin_info)
                .unwrap();

            let r_image = frame_data.render_image;
            let p_image = frame_data.present_image;

            let extent = self.swapchain.extent;

            vk_util::transition_image(
                device,
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
                device,
                cmd_buffer,
                r_image,
                vk::ImageLayout::GENERAL,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            );

            // transition present image to copy to
            vk_util::transition_image(
                device,
                cmd_buffer,
                p_image,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            );

            //copy render image onto present image
            vk_util::blit_copy_image_to_image(
                &self.logical_device,
                cmd_buffer,
                r_image,
                extent,
                p_image,
                extent,
            );

            // transition swapchain to present
            vk_util::transition_image(
                device,
                cmd_buffer,
                p_image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::ImageLayout::PRESENT_SRC_KHR,
            );

            device.end_command_buffer(cmd_buffer).unwrap();

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
            device
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

    pub fn draw_background(
        &self,
        image: vk::Image,
        cmd_buffer: vk::CommandBuffer,
        descriptor_set: &[vk::DescriptorSet],
        pipeline_layout: vk::PipelineLayout,
        pipeline: vk::Pipeline,
    ) {
        // let flash = f32::abs(f32::sin(frame_num as f32 / 50f32));
        // let clear_values = vk::ClearValue {
        //     color: vk::ClearColorValue {
        //         float32: [0.0, 0.0, flash, 1.0],
        //     },
        // };
        //
        // let clear_range = [vk_util::image_subresource_range(
        //     vk::ImageAspectFlags::COLOR,
        // )];
        //
        // unsafe {
        //     self.logical_device.device.cmd_clear_color_image(
        //         cmd_buffer,
        //         image,
        //         vk::ImageLayout::GENERAL,
        //         &clear_values.color,
        //         &clear_range,
        //     );
        // }

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
