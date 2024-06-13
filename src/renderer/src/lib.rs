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
use bytemuck;
use glam::*;
use image::GenericImageView;
use input;
use input::{InputManager, KeyboardListener, ListenerType, MousePosListener};
use std::collections::HashSet;
use std::fmt::{Debug, Pointer};
use std::process::exit;
use std::time::{Duration, Instant, SystemTime};
use std::{env, ptr, time};
use std::thread::Thread;
use winit::application::ApplicationHandler;

use winit::event_loop::{ActiveEventLoop, ControlFlow};
use winit::platform::modifier_supplement::KeyEventExtModifierSupplement;
use winit::window::{Window, WindowId};
use winit::{
    event::*,
    event_loop::EventLoop,
    keyboard::{KeyCode, PhysicalKey},
};

pub struct Empty {}

impl KeyboardListener for Empty {
    fn listener_type(&self) -> ListenerType {
        ListenerType::GameInput
    }

    fn listener_id(&self) -> u32 {
        1
    }

    fn listener_for(&self, key: KeyCode) -> bool {
        true
    }

    fn broadcast(&mut self, key: KeyCode, pressed: bool, modifiers: &HashSet<Modifiers>) {
        println!("Key: {:?} : {:?}", key, pressed)
    }
}

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

impl Control {
    pub fn new(input_manager: InputManager, window_size: (u32, u32), wait_time: Duration) -> Self {
        Self {
            app: None,
            input_manager,
            wait_time,
            frame: 0,
            last_time: SystemTime::now(),
            request_redraw: true,
            wait_cancelled: false,
            close_requested: false,
            window_size,
        }
    }
}

impl ApplicationHandler for Control {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window_att = Window::default_attributes()
            .with_title("Winit fucking sucks")
            .with_inner_size(winit::dpi::LogicalSize::new(
                self.window_size.0,
                self.window_size.1,
            ));

        let window = event_loop.create_window(window_att).unwrap();
        window.request_redraw();
        self.app = Some(init_vulkan_app(window, self.window_size).unwrap());

    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                self.close_requested = true;
            }
            WindowEvent::ActivationTokenDone { .. } => {}
            WindowEvent::Resized(_) => {}
            WindowEvent::Moved(_) => {}
            WindowEvent::KeyboardInput { event, .. } => {
              //  self.input_manager.update();
                if let PhysicalKey::Code(key) = event.physical_key {
                    // self.input_manager.add_keycode(key) // TODO need to take pressed state into account
                }
            }
            WindowEvent::ModifiersChanged(modd, ..) => {}
            WindowEvent::CursorMoved { position, .. } => {
                self.input_manager
                    .update_mouse_pos(position.x as f32, position.y as f32)
                //TODO Maybe track as f64?
            }
            WindowEvent::CursorEntered { .. } => {}
            WindowEvent::CursorLeft { .. } => {}
            WindowEvent::MouseWheel { delta, .. } => {
                if let MouseScrollDelta::LineDelta(delta, ..) = delta {
                    self.input_manager.update_scroll_state(delta)
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                self.input_manager.add_mouse_button(button)
            }
            WindowEvent::RedrawRequested => {
                if let Some(ref mut app) = &mut self.app {
                    app.render(self.frame);
                }

                let now = SystemTime::now();
                let frame_ms = now.duration_since(self.last_time).unwrap().as_millis();
                self.last_time = now;
                if let Some(app) = &self.app {
                    app.window
                        .set_title(format!("Frame-time: {}", frame_ms).as_str());
                    // event_loop.set_control_flow(ControlFlow::WaitUntil(
                    //     std::time::Instant::now() + self.wait_time,
                    // ));
                }

                self.frame += 1;

            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        if self.request_redraw && !self.wait_cancelled && !self.close_requested {
            if let Some(app) = &self.app {
                self.last_time = time::SystemTime::now();
                self.frame += 1;
                app.window.request_redraw();
            }
        }

        // event_loop.set_control_flow(ControlFlow::Poll);
    }
}

pub fn run() {
    env_logger::Builder::new()
        .target(env_logger::Target::Stdout)
        .parse_filters(&*env::var("RUST_LOG").unwrap_or_else(|_| "info".to_string()))
        .init();

    let input_manager = InputManager::new();
    let mut control = Control::new(input_manager, (1920, 1080), Duration::from_secs(2));
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop.run_app(&mut control).unwrap();

}


impl VulkanApp {
    pub fn render(&mut self, frame_number: u32) {
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

            let clear_range = [vk_util::image_subresource_range(vk::ImageAspectFlags::COLOR)];
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
    }
}

pub fn init_vulkan_app(
    window: winit::window::Window,
    size: (u32, u32),
) -> Result<VulkanApp, String> {
    let entry = vk_init::init_entry();

    let mut instance_ext = vk_init::get_winit_extensions(&window);
    let (instance, debug) =
        vk_init::init_instance(&entry, "test".to_string(), &mut instance_ext, true)?;

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
        None,
        None,
        Some(vk::PresentModeKHR::FIFO),
        None,
        true,
    )?;

    let present_images = vk_init::create_basic_image_views(&logical_device, &swapchain)?;

    let command_pools = vk_init::create_command_pools(&logical_device, 2)?;

    let frame_buffers: Vec<VkFrameBuffer> = (0..3)
        .map(|_| vk_init::create_frame_buffer(&logical_device))
        .collect::<Result<Vec<_>, _>>()?;

    let presentation = VkPresent::new(frame_buffers, &present_images);

    Ok(VulkanApp {
        window,
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
