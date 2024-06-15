use std::time::SystemTime;
use ash::vk;
use ash::vk::ExtendsPhysicalDeviceFeatures2;
use crate::vk_init::{LogicalDevice, PhyDevice, VkDebug, VkFrameSync, VkPresent, VkSurface, VkSwapchain};
use crate::{vk_init, vk_util};


pub struct VkRender {
    pub window: glfw::PWindow,
    pub entry: ash::Entry,
    pub instance: ash::Instance,
    pub debug: Option<VkDebug>,
    pub physical_device: PhyDevice,
    pub logical_device: LogicalDevice,
    pub surface: VkSurface,
    pub swapchain: VkSwapchain,
    pub present_images: Vec<vk::ImageView>,
    // pub command_pools: Vec<VkCommandPool>,
    pub presentation: VkPresent,
}

impl Drop for VkRender {
    fn drop(&mut self) {
        unsafe {
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

            self.present_images
                .iter()
                .for_each(|view| self.logical_device.device.destroy_image_view(*view, None));

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
        unsafe {
            self.instance.destroy_instance(None); // None == allocator callback
        }
    }

    pub fn new(window: glfw::PWindow, size: (u32, u32)) -> Result<Self, String> {
        let entry = vk_init::init_entry();

        let mut instance_ext = vk_init::get_glfw_extensions(&window);
        let (instance, debug) =
            vk_init::init_instance(&entry, "test".to_string(), &mut instance_ext, false)?;

        let surface = vk_init::get_glfw_surface(&entry, &instance, &window)?;

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

        let swapchain_support = vk_init::get_swapchain_support(&physical_device.p_device, &surface)?;

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

        let present_images = vk_init::create_basic_image_views(&logical_device, &swapchain)?;

        let mut command_pools = vk_init::create_command_pools(&logical_device, 2)?;

        let frame_buffers: Vec<VkFrameSync> = (0..2)
            .map(|_| vk_init::create_frame_buffer(&logical_device))
            .collect::<Result<Vec<_>, _>>()?;

        // TODO will have to separate compute pools when we get to that stage
        //  currently we share present and graphics and just have a simple pool returned
        //  it needs to be ironed out how to manage separate pools in relation, maybe just
        //  store them all in the frame buffer?

        let presentation = VkPresent::new(frame_buffers, &present_images, command_pools.remove(0));

        Ok(VkRender {
            window,
            entry,
            instance,
            debug,
            physical_device,
            logical_device,
            surface,
            swapchain,
            present_images,
            presentation,
        })
    }

}


impl VkRender {
    pub fn render(&mut self, frame_number: u32) {
        let start = SystemTime::now();
        let device = &self.logical_device.device;
        let (frame_sync, cmd_pool, cmd_buffer, cmd_queue) = self.presentation.get_next_frame();
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

            let image = self.swapchain.swapchain_images[image_index as usize];
            vk_util::transition_image(
                device,
                cmd_buffer,
                image,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::GENERAL,
            );

            let flash = f32::abs(f32::sin(frame_number as f32 / 50f32));
            let clear_values = vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.0, 0.0, flash, 1.0],
                },
            };

            let clear_range = [vk_util::image_subresource_range(
                vk::ImageAspectFlags::COLOR,
            )];

            device.cmd_clear_color_image(
                cmd_buffer,
                image,
                vk::ImageLayout::GENERAL,
                &clear_values.color,
                &clear_range,
            );

            vk_util::transition_image(
                device,
                cmd_buffer,
                image,
                vk::ImageLayout::GENERAL,
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
                .queue_submit2(cmd_queue, &submit, frame_sync.render_fence)
                .unwrap();

            let r_sem = [frame_sync.render_semaphore];
            let imf_idex = [image_index];
            let present_info = vk::PresentInfoKHR::default()
                .swapchains(&swapchain)
                .wait_semaphores(&r_sem)
                .image_indices(&imf_idex);

            self.swapchain
                .swapchain_loader
                .queue_present(cmd_queue, &present_info)
                .unwrap();
        }
        println!(
            "Render Took: {}ms",
            SystemTime::now().duration_since(start).unwrap().as_millis()
        )
    }
}

