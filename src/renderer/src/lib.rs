#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(unused_variables)]

mod data;
mod texture;
mod vulkan;

use ash::vk;
use ash::vk::{
    CommandBuffer, CommandBufferResetFlags, CommandBufferUsageFlags,
    ExtendsPhysicalDeviceFeatures2, Extent2D, PhysicalDeviceFeatures, SubmitInfo2,
};
use glam::*;
use image::GenericImageView;
use input;
use input::{InputManager, KeyboardListener, ListenerType, MousePosListener};
use std::collections::HashSet;

use crate::data::{camera, gltf_util};
use crate::data::camera::FPSController;
use crate::vulkan::vk_render;
use crate::vulkan::vk_types::VkWindowState;
use raw_window_handle::{RawWindowHandle};
use std::cell::RefCell;
use std::cmp::max;
use std::process::exit;
use std::rc::Rc;
use std::time::{Duration, Instant, SystemTime};
use std::{env, ptr, time};
use log::{info, log};
use winit::dpi::Position;
use winit::event::{DeviceEvent, Event, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ControlFlow, DeviceEvents, EventLoop};
use winit::keyboard::NamedKey::Camera;
use winit::keyboard::PhysicalKey;
use winit::window::{CursorGrabMode, Window, WindowId, WindowLevel};
use crate::data::data_cache::{MeshCache, TextureCache};


const NANO: f64 = 1000000000.0;

pub struct GameLogic {
    input_manager: InputManager,
}


pub fn gltf(str : String) {
    // let mut texture_cache = TextureCache::new();
    // // let mut mesh_cache = MeshCache::default();
    // // gltf_util::parse_gltf_to_raw(str.as_str(), &mut texture_cache, &mut mesh_cache).unwrap();
}

pub fn run() {
    env_logger::Builder::new()
        .target(env_logger::Target::Stdout)
        .parse_filters(&env::var("RUST_LOG").unwrap_or_else(|_| "info".to_string()))
        .init();

    let mut input_manager = InputManager::default();

    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);

    let size = Extent2D::default().width(1920).height(1080);

    let window = winit::window::WindowBuilder::new()
        .with_inner_size(winit::dpi::PhysicalSize::new(size.width, size.height))
        .build(&event_loop)
        .unwrap();

    let max_extent = if let Some(mon) = event_loop.available_monitors().max_by_key(|mon| mon.size())
    {
        Extent2D::default()
            .height(mon.size().height)
            .width(mon.size().width)
    } else {
        panic!("Failed to detect monitor")
    };


    let camera = camera::Camera::default();
    let fps_controller = FPSController::new(1, camera, 0.01, 2.0);

    let window_state = VkWindowState::new(window, size, max_extent, fps_controller);
    input_manager.register_key_listener(window_state.controller.clone());
    input_manager.register_m_pos_listener(window_state.controller.clone());

    let mut last_time = SystemTime::now();
    let mut frame: u32 = 0;
    let mut fps_timer = SystemTime::now();

    let mut app = vk_render::VkRender::new(window_state, true, true).unwrap();
    
    event_loop
        .run(move |event, control_flow| {
            app.imgui.handle_event(&app.window_state.window, &event);
            let delta = SystemTime::now().duration_since(last_time).unwrap();

            match event {
                Event::NewEvents(..) => {
                    app.imgui.context.io_mut().update_delta_time(delta);
                }
                Event::DeviceEvent { device_id, event } => match event {
                    DeviceEvent::MouseMotion { delta } => {
                        input_manager.update_mouse_pos(delta);
                    },
                    DeviceEvent::MouseWheel { delta: MouseScrollDelta::LineDelta(delta, ..) } => {
                        input_manager.update_scroll_state(delta)
                    }
                    DeviceEvent::Button { button, state} => {
                        //input_manager.add_mouse_button(*button)
                    }
                    DeviceEvent::Key(key_event) => {
                        if let PhysicalKey::Code(key) = key_event.physical_key {
                            input_manager.add_keycode(key, key_event.state.is_pressed())
                        }
                    }
                    _ => {}
                },

                Event::WindowEvent {
                    ref event,
                    window_id,
                } if window_id == window_id => {
                    match event {
                        WindowEvent::ActivationTokenDone { .. } => {}
                        WindowEvent::Moved(_) => {}
                        WindowEvent::CloseRequested => {
                            control_flow.exit();
                        }
                        WindowEvent::Destroyed => {}
                        WindowEvent::DroppedFile(_) => {}
                        WindowEvent::HoveredFile(_) => {}
                        WindowEvent::HoveredFileCancelled => {}
                        WindowEvent::Focused(focused) => {}
                        WindowEvent::KeyboardInput {
                            event: key_event, ..
                        } => {
                        }
                        WindowEvent::ModifiersChanged(modd, ..) => {}
                        WindowEvent::Ime(_) => {}
                        WindowEvent::CursorMoved { position, .. } => {}
                        WindowEvent::CursorEntered { .. } => {
                            // app.window_state
                            //     .window
                            //     .set_cursor_grab(CursorGrabMode::Confined)
                            //     .unwrap();
                            // app.window_state.window.set_cursor_visible(false);
                        }
                        WindowEvent::CursorLeft { .. } => {
                            app.window_state
                                .window
                                .set_cursor_grab(CursorGrabMode::None)
                                .unwrap();
                            app.window_state.window.set_cursor_visible(true);
                        }
                        WindowEvent::MouseWheel { delta, .. } => {
                          
                        }
                        WindowEvent::MouseInput { state, button, .. } => {
                           
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
                        WindowEvent::Resized(new_size) => {
                            info!("Resize requested");
                            
                            app.resize_requested = true;
                            let new_extent = Extent2D::default()
                                .height(new_size.height)
                                .width(new_size.width);

                            app.rebuild_swapchain(new_extent)
                        }
                        WindowEvent::RedrawRequested => {
                            input_manager.update();
                            app.window_state
                                .controller
                                .borrow_mut()
                                .update(delta.as_millis());

                          

                            if !app.resize_requested {
                                let now = SystemTime::now();
                                app.imgui.context.io_mut().update_delta_time(delta);
                                app.imgui
                                    .platform
                                    .prepare_frame(
                                        app.imgui.context.io_mut(),
                                        &app.window_state.window,
                                    )
                                    .unwrap();
                                let frame_ms = delta.as_millis();
                                last_time = now;

                                app.render(frame);
                                app.window_state.window.request_redraw();
                                if now.duration_since(fps_timer).unwrap() > Duration::from_secs(1) {
                                    app.window_state
                                        .window
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
