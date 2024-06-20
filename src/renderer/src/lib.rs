#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(unused_variables)]

pub mod renderer;
mod texture;
mod vk_descriptor;
mod vk_init;
mod vk_pipeline;
mod vk_render;
mod vk_util;

use crate::vk_init::{QueueType, VkFrameSync, VkPresent};
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

use crate::vk_render::VkRender;
use raw_window_handle::RawWindowHandle;
use std::process::exit;
use std::time::{Duration, Instant, SystemTime};
use std::{env, ptr, time};
use winit::application::ApplicationHandler;
use winit::event::{MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::PhysicalKey;
use winit::window::{Window, WindowId};

const NANO: f64 = 1000000000.0;

pub struct GameLogic {
    input_manager: InputManager,
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

struct Control {
    app: Option<VkRender>,
    input_manager: InputManager,
    wait_time: Duration,
    frame: u32,
    last_time: SystemTime,
    request_redraw: bool,
    wait_cancelled: bool,
    close_requested: bool,
    window_size: (u32, u32),
    fps_timer: SystemTime,
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
            fps_timer: SystemTime::now(),
        }
    }
}

impl ApplicationHandler for Control {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window_att = Window::default_attributes().with_title("").with_inner_size(
            winit::dpi::LogicalSize::new(self.window_size.0, self.window_size.1),
        );

        let window = event_loop.create_window(window_att).unwrap();
        self.app = Some(vk_render::VkRender::new(window, self.window_size, true).unwrap());
        self.app.as_mut().unwrap().window.request_redraw();
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
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
                let now = SystemTime::now();
                let frame_ms = now.duration_since(self.last_time).unwrap().as_millis();
                self.last_time = now;

                let app = unsafe { self.app.as_mut().unwrap_unchecked() };
                app.render(self.frame);
                app.window.request_redraw();
                if now.duration_since(self.fps_timer).unwrap() > Duration::from_secs(1) {
                    app.window
                        .set_title(format!("Frame-time: {}", self.frame).as_str());
                    self.frame = 0;
                    self.fps_timer = now;

                    // event_loop.set_control_flow(ControlFlow::WaitUntil(
                    //     std::time::Instant::now() + self.wait_time,
                    // ));

                }

                self.frame += 1;
            }
            _ => {}
        }
    }

    // fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
    //     if self.request_redraw && !self.wait_cancelled && !self.close_requested {
    //         if let Some(app) = &self.app {
    //             self.last_time = time::SystemTime::now();
    //             self.frame += 1;
    //             app.window.request_redraw();
    //         }
    //     }
    //
    //     println!("Updated physics");
    //
    //     // event_loop.set_control_flow(ControlFlow::Poll);
    // }
}

// pub fn run2() {
//     let mut glfw = glfw::init(glfw::log_errors).unwrap();
//
//     glfw.window_hint(glfw::WindowHint::ClientApi(ClientApiHint::NoApi));
//     glfw.window_hint(glfw::WindowHint::DoubleBuffer(false));
//
//     let (mut window, events) = glfw
//         .create_window(
//             1920,
//             1080,
//             "Hello this is window",
//             glfw::WindowMode::Windowed,
//         )
//         .expect("Failed to create GLFW window.");
//
//     window.set_key_polling(false);
//     window.set_raw_mouse_motion(true);
//     window.make_current();
//
//     let mut app = VkRender::new(window, (1920, 1080), true).unwrap();
//
//     // window.set_key_callback(|_, key, _, action, _| println!("Input: {:?}", action));
//
//     let logic_ups = 10000.0;
//     let frame_ups = 10000.0;
//
//     let time_u = NANO / logic_ups;
//     let time_r = if frame_ups > 0.0 {
//         NANO / frame_ups
//     } else {
//         0.0
//     };
//     let mut delta_update = 0.0;
//     let mut delta_fps = 0.0;
//
//     let init_time = SystemTime::now();
//     let mut last_time = init_time;
//     let mut frames = 0;
//
//     let mut fps_timer = SystemTime::now();
//     let mut frame = 0;
//     let running = true;
//
//     while !app.window.should_close() {
//         let now = SystemTime::now();
//         let elapsed = now.duration_since(last_time).unwrap().as_nanos() as f64;
//         delta_update += elapsed / time_u;
//         delta_fps += elapsed / time_r;
//
//
//         while delta_update >= 1.0 {
//             delta_update -= 1.0;
//             glfw.poll_events();
//             // update logic here
//         }
//
//         if delta_fps >= 1.0 {
//             app.render(frame);
//             delta_fps -= 1.0;
//             frames += 1;
//             frame += 1;
//         }
//
//         if now.duration_since(fps_timer).unwrap() > Duration::from_secs(1) {
//             app.window.set_title(&format!("FPS: {}", frames));
//             fps_timer = SystemTime::now();
//             frames = 0;
//         }
//
//         last_time = now;
//     }
//}
