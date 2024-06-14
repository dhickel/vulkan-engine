#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(unused_variables)]

pub mod renderer;
mod texture;
mod vk_init;
mod vk_util;

use crate::vk_init::{QueueType, VkFrameBuffer, VkPresent, VulkanApp};
use ash::vk;
use ash::vk::{
    CommandBuffer, CommandBufferResetFlags, CommandBufferUsageFlags,
    ExtendsPhysicalDeviceFeatures2, PhysicalDeviceFeatures, SubmitInfo2,
};
use glam::*;
use glfw::{Action, ClientApiHint, Context, Key, WindowHint};
use image::GenericImageView;
use input;
use input::{InputManager, KeyboardListener, ListenerType, MousePosListener};
use std::collections::HashSet;


use std::time::{Duration, Instant, SystemTime};
use std::{env, ptr, time};
use raw_window_handle::{HasRawWindowHandle, RawWindowHandle};


const NANO: f64 = 1000000000.0;

pub struct GameLogic {
    input_manager: InputManager,
}

struct Control {
    app: Option<VulkanApp>,
    input_manager: InputManager,
    wait_time: Duration,
    frame: u32,
    last_time: SystemTime,
    request_redraw: bool,
    wait_cancelled: bool,
    close_requested: bool,
    window_size: (u32, u32),
}

pub fn run2() {


    let mut glfw = glfw::init(glfw::log_errors).unwrap();

    glfw.window_hint(WindowHint::ClientApi(ClientApiHint::NoApi));
    let (mut window, events) = glfw
        .create_window(
            1920,
            1080,
            "Hello this is window",
            glfw::WindowMode::Windowed,
        )
        .expect("Failed to create GLFW window.");

    window.set_key_polling(false);
    window.set_raw_mouse_motion(true);


    let mut app = init_vulkan_app(&window, (1920, 1080)).unwrap();
    window.make_current();


    // window.set_key_callback(|_, key, _, action, _| println!("Input: {:?}", action));

    let logic_ups = 10000.0;
    let frame_ups = 1000.0;

    let time_u = NANO / logic_ups;
    let time_r = if frame_ups > 0.0 {
        NANO / frame_ups
    } else {
        0.0
    };
    let mut delta_update = 0.0;
    let mut delta_fps = 0.0;

    let init_time = SystemTime::now();
    let mut last_time = init_time;
    let mut frames = 0;

    let mut fps_timer = SystemTime::now();
    let mut frame = 0;
    let running = true;

    while !window.should_close() {
        let now = SystemTime::now();
        let elapsed = now.duration_since(last_time).unwrap().as_nanos() as f64;
        delta_update += elapsed / time_u;
        delta_fps += elapsed / time_r;

        while delta_update >= 1.0 {
            delta_update -= 1.0;
            glfw.poll_events();
            // update logic here
        }

        if delta_fps >= 1.0 {
            app.render(frame);
            delta_fps -= 1.0;
            frames += 1;
            frame +=1;
        }

        if now.duration_since(fps_timer).unwrap() > Duration::from_secs(1) {
            window.set_title(&format!("FPS: {}", frames));
            fps_timer = SystemTime::now();
            frames = 0;
        }

        last_time = now;
    }
}

impl VulkanApp {
    pub fn render(&mut self, frame_number: u32) {
        let start = SystemTime::now();
        let device = &self.logical_device.device;
        let frame_buffer = self.presentation.get_next_frame();
        let fence = &[frame_buffer.render_fence];
        let cmd_pool = self
            .command_pools
            .iter()
            .find(|p| p.queue_type.contains(&QueueType::Present))
            .unwrap();
        let cmd = cmd_pool.buffers[0];

        let swapchain = [self.swapchain.swapchain];

        unsafe {
            device
                .wait_for_fences(fence, true, u32::MAX as u64)
                .unwrap();
            device.reset_fences(fence).unwrap();

            let acquire = vk::AcquireNextImageInfoKHR::default()
                .swapchain(self.swapchain.swapchain)
                .semaphore(frame_buffer.swap_semaphore)
                .device_mask(1)
                .timeout(u32::MAX as u64);

            let (image_index, _) = self
                .swapchain
                .swapchain_loader
                .acquire_next_image2(&acquire)
                .unwrap();

            device
                .reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())
                .unwrap();

            let begin_info = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

            device.begin_command_buffer(cmd, &begin_info).unwrap();

            let image = self.swapchain.swapchain_images[image_index as usize];
            vk_util::transition_image(
                device,
                cmd,
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
                cmd,
                image,
                vk::ImageLayout::GENERAL,
                &clear_values.color,
                &clear_range,
            );

            vk_util::transition_image(
                device,
                cmd,
                image,
                vk::ImageLayout::GENERAL,
                vk::ImageLayout::PRESENT_SRC_KHR,
            );

            device.end_command_buffer(cmd).unwrap();

            let cmd_info = [vk_util::command_buffer_submit_info(cmd)];
            let wait_info = [vk_util::semaphore_submit_info(
                vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT_KHR,
                frame_buffer.swap_semaphore,
            )];
            let signal_info = [vk_util::semaphore_submit_info(
                vk::PipelineStageFlags2::ALL_GRAPHICS,
                frame_buffer.render_semaphore,
            )];
            let submit = [vk_util::submit_info_2(&cmd_info, &signal_info, &wait_info)];
            device
                .queue_submit2(cmd_pool.queue, &submit, frame_buffer.render_fence)
                .unwrap();

            let r_sem = [frame_buffer.render_semaphore];
            let imf_idex = [image_index];
            let present_info = vk::PresentInfoKHR::default()
                .swapchains(&swapchain)
                .wait_semaphores(&r_sem)
                .image_indices(&imf_idex);

            self.swapchain
                .swapchain_loader
                .queue_present(cmd_pool.queue, &present_info)
                .unwrap();
        }
        println!("Render Took: {}ms", SystemTime::now().duration_since(start).unwrap().as_millis())
    }
}

pub fn init_vulkan_app(window: &glfw::PWindow, size: (u32, u32)) -> Result<VulkanApp, String> {
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

    let command_pools = vk_init::create_command_pools(&logical_device, 2)?;

    let frame_buffers: Vec<VkFrameBuffer> = (0..2)
        .map(|_| vk_init::create_frame_buffer(&logical_device))
        .collect::<Result<Vec<_>, _>>()?;

    let presentation = VkPresent::new(frame_buffers, &present_images);

    Ok(VulkanApp {
        entry,
        instance,
        debug,
        physical_device,
        logical_device,
        surface,
        swapchain,
        present_images,
        command_pools,
        presentation,
    })
}
