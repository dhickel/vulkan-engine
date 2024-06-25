#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(unused_variables)]

pub mod renderer;
mod texture;
mod vulkan;
mod data;


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

use raw_window_handle::RawWindowHandle;
use std::process::exit;
use std::time::{Duration, Instant, SystemTime};
use std::{env, ptr, time};
use winit::event::{Event, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::keyboard::PhysicalKey;
use winit::window::{Window, WindowId};
use crate::vulkan::vk_render;


const NANO: f64 = 1000000000.0;

pub struct GameLogic {
    input_manager: InputManager,
}

pub fn run() {
    env_logger::Builder::new()
        .target(env_logger::Target::Stdout)
        .parse_filters(&*env::var("RUST_LOG").unwrap_or_else(|_| "info".to_string()))
        .init();

    let mut input_manager = InputManager::new();

    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);

    let size = (1920u32, 1080u32);
    let window = winit::window::WindowBuilder::new()
        .with_inner_size(winit::dpi::PhysicalSize::new(size.0, size.1))
        .build(&event_loop)
        .unwrap();

    let mut last_time = SystemTime::now();
    let mut frame: u32 = 0;
    let mut fps_timer = SystemTime::now();

    let mut app = vk_render::VkRender::new(window, size, true).unwrap();
    let mut opened  = true;

    event_loop
        .run(move |event, control_flow| {
            input_manager.update();
            app.imgui.handle_event(&app.window, &event);
            match event {
                Event::NewEvents(..) => {
                    app.imgui.context.io_mut().update_delta_time(SystemTime::now().duration_since(last_time).unwrap());
                }

                Event::WindowEvent {
                    ref event,
                    window_id,
                } if window_id == window_id => {

                    match event {

                        WindowEvent::ActivationTokenDone { .. } => {}
                        WindowEvent::Resized(_) => {}
                        WindowEvent::Moved(_) => {}
                        WindowEvent::CloseRequested => {
                            control_flow.exit();
                        },
                        WindowEvent::Destroyed => {}
                        WindowEvent::DroppedFile(_) => {}
                        WindowEvent::HoveredFile(_) => {}
                        WindowEvent::HoveredFileCancelled => {}
                        WindowEvent::Focused(_) => {}
                        WindowEvent::KeyboardInput { event: key_event, .. } => {

                            input_manager.update();
                            if let PhysicalKey::Code(key) = key_event.physical_key {
                                input_manager.add_keycode(key) // TODO need to take pressed state into account
                            }
                        }
                        WindowEvent::ModifiersChanged(modd, ..) => {}
                        WindowEvent::Ime(_) => {}
                        WindowEvent::CursorMoved { position, .. } => {
                            input_manager.update_mouse_pos(position.x as f32, position.y as f32)
                            //TODO Maybe track as f64?
                        }
                        WindowEvent::CursorEntered { .. } => {}
                        WindowEvent::CursorLeft { .. } => {}
                        WindowEvent::MouseWheel { delta, .. } => {
                            if let MouseScrollDelta::LineDelta(delta, ..) = delta {
                                input_manager.update_scroll_state(*delta)
                            }
                        }
                        WindowEvent::MouseInput { state, button, .. } => {
                            input_manager.add_mouse_button(*button)
                        }
                        WindowEvent::TouchpadMagnify { .. } => {}
                        WindowEvent::SmartMagnify { .. } => {}
                        WindowEvent::TouchpadRotate { .. } => {}
                        WindowEvent::TouchpadPressure { .. } => {}
                        WindowEvent::AxisMotion { .. } => {}
                        WindowEvent::Touch(_) => {}
                        WindowEvent::ScaleFactorChanged { .. } => {} // Tutorial resizes here but api changed
                        WindowEvent::ThemeChanged(_) => {}
                        WindowEvent::Occluded(_) => {}
                        WindowEvent::RedrawRequested => {


                            let now = SystemTime::now();
                            app.imgui.context.io_mut().update_delta_time(now.duration_since(last_time).unwrap());
                            app.imgui.platform.prepare_frame(app.imgui.context.io_mut(), &app.window).unwrap();
                            let frame_ms = now.duration_since(last_time).unwrap().as_millis();
                            last_time = now;

                            app.render(frame);
                            app.window.request_redraw();
                            if now.duration_since(fps_timer).unwrap() > Duration::from_secs(1) {
                                app.window
                                    .set_title(format!("Frame-Rate: {}", frame).as_str());
                                frame = 0;
                                fps_timer = now;

                                // event_loop.set_control_flow(ControlFlow::WaitUntil(
                                //     std::time::Instant::now() + self.wait_time,
                                // ));
                            }
                            frame += 1;
                        }
                    }
                }
                _ => {}
            }
        })
        .expect("TODO: panic message");


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
//}

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
