#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(unused_variables)]

mod data;
mod rendergraph;
mod scene;
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

use crate::data::camera;
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
use log::{error, info, log, warn};
use winit::dpi::Position;
use winit::event::{DeviceEvent, Event, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ControlFlow, DeviceEvents, EventLoop, EventLoopWindowTarget};
use winit::keyboard::NamedKey::Camera;
use winit::keyboard::PhysicalKey;
use winit::window::{CursorGrabMode, Window, WindowId, WindowLevel};
use crate::data::data_cache::{MeshCache, TextureCache};
use crate::scene::scene_world::SceneWorld;


const NANO: f64 = 1000000000.0;

pub struct GameLogic {
    input_manager: InputManager,
}


pub fn gltf(str : String) {
    // let mut texture_cache = TextureCache::new();
    // // let mut mesh_cache = MeshCache::default();
    // // gltf_util::parse_gltf_to_raw(str.as_str(), &mut texture_cache, &mut mesh_cache).unwrap();
}

fn parse_debug_runtime_mode(args: &[String]) -> vk_render::DebugRuntimeMode {
    let parse_value = |value: &str| -> Option<vk_render::DebugRuntimeMode> {
        vk_render::DebugRuntimeMode::from_label(value)
    };

    let mut i = 0usize;
    while i < args.len() {
        let arg = &args[i];

        if arg == "debug_runtime" || arg == "--debug-runtime" {
            let Some(value) = args.get(i + 1) else {
                warn!(
                    "Missing debug runtime mode after '{}'. Valid values: testpbr, testunlit",
                    arg
                );
                return vk_render::DebugRuntimeMode::Default;
            };

            if let Some(mode) = parse_value(value) {
                return mode;
            }

            warn!(
                "Unsupported debug runtime mode '{}'. Valid values: testpbr, testunlit",
                value
            );
            return vk_render::DebugRuntimeMode::Default;
        }

        if let Some(value) = arg.strip_prefix("--debug-runtime=") {
            if let Some(mode) = parse_value(value) {
                return mode;
            }

            warn!(
                "Unsupported debug runtime mode '{}'. Valid values: testpbr, testunlit",
                value
            );
            return vk_render::DebugRuntimeMode::Default;
        }

        i += 1;
    }

    vk_render::DebugRuntimeMode::Default
}

#[derive(Copy, Clone)]
struct RuntimeFlags {
    compile_shaders: bool,
    debug_runtime_mode: vk_render::DebugRuntimeMode,
}

struct RuntimeLoopState {
    input_manager: InputManager,
    app: vk_render::VkRender,
    scene_world: SceneWorld,
    last_time: SystemTime,
    frame: u32,
    fps_timer: SystemTime,
}

/// Initialize logger output once at startup for all runtime subsystems.
fn init_runtime_logging() {
    env_logger::Builder::new()
        .target(env_logger::Target::Stdout)
        .parse_filters(&env::var("RUST_LOG").unwrap_or_else(|_| "debug".to_string()))
        .init();
}

/// Build the event loop, main window and camera controller wiring used by the renderer.
fn init_window_state(event_loop: &EventLoop<()>, input_manager: &mut InputManager) -> VkWindowState {
    let size = Extent2D::default().width(1920).height(1080);

    let window = winit::window::WindowBuilder::new()
        .with_inner_size(winit::dpi::PhysicalSize::new(size.width, size.height))
        .build(event_loop)
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
    let fps_controller = FPSController::new(1, camera, 0.002, 1.0);

    let window_state = VkWindowState::new(window, size, max_extent, fps_controller);
    input_manager.register_key_listener(window_state.controller.clone());
    input_manager.register_m_pos_listener(window_state.controller.clone());
    window_state
}

/// Parse runtime flags that alter renderer startup behavior.
fn parse_runtime_flags(args: &[String]) -> RuntimeFlags {
    let rebuild_from_env = env::var("ENGINE_REBUILD_SHADERS")
        .map(|v| {
            let lowered = v.trim().to_ascii_lowercase();
            matches!(lowered.as_str(), "1" | "true" | "yes" | "on")
        })
        .unwrap_or(false);
    let rebuild_from_args = args.iter().any(|arg| arg == "--rebuild-shaders");
    let debug_runtime_mode = parse_debug_runtime_mode(args);
    let compile_shaders = rebuild_from_env || rebuild_from_args;

    if compile_shaders {
        info!("Shader rebuild requested (--rebuild-shaders or ENGINE_REBUILD_SHADERS=1).");
    }

    if debug_runtime_mode != vk_render::DebugRuntimeMode::Default {
        info!(
            "Debug runtime mode selected: {}",
            debug_runtime_mode.as_label()
        );
    }

    RuntimeFlags {
        compile_shaders,
        debug_runtime_mode,
    }
}

/// Create the Vulkan renderer and startup scene payload.
fn create_runtime_renderer(
    window_state: VkWindowState,
    runtime_flags: RuntimeFlags,
) -> Option<(vk_render::VkRender, SceneWorld)> {
    match vk_render::VkRender::new(
        window_state,
        false,
        runtime_flags.compile_shaders,
        runtime_flags.debug_runtime_mode,
    ) {
        Ok(runtime) => Some(runtime),
        Err(err) => {
            error!("Renderer initialization failed: {err}");
            if runtime_flags.compile_shaders {
                error!("Shader rebuild requires 'glslc' or 'glslangValidator' in PATH. Install Vulkan shader tools (e.g. package 'shaderc' or LunarG Vulkan SDK), then retry.");
            }
            None
        }
    }
}

/// Process raw device events and forward input deltas into the input manager.
fn handle_device_event(state: &mut RuntimeLoopState, event: DeviceEvent) {
    match event {
        DeviceEvent::MouseMotion { delta } => {
            state.input_manager.update_mouse_pos(delta);
        }
        DeviceEvent::MouseWheel {
            delta: MouseScrollDelta::LineDelta(delta, ..),
        } => {
            state.input_manager.update_scroll_state(delta);
        }
        DeviceEvent::Button { button: _, state: _ } => {
            // input_manager.add_mouse_button(*button)
        }
        DeviceEvent::Key(_key_event) => {
            // Keyboard input moved to WindowEvent for better reliability
        }
        _ => {}
    }
}

/// Handle key-driven window control and keyboard broadcast state updates.
fn handle_keyboard_input(
    state: &mut RuntimeLoopState,
    control_flow: &EventLoopWindowTarget<()>,
    key_event: &winit::event::KeyEvent,
) {
    if let PhysicalKey::Code(key) = key_event.physical_key {
        state
            .input_manager
            .add_keycode(key, key_event.state.is_pressed());
    }

    // Hardwired hack to close the renderer on Escape key press
    if key_event.physical_key == PhysicalKey::Code(winit::keyboard::KeyCode::Escape) {
        control_flow.exit();
    }
}

/// Configure cursor lock/visibility when entering or leaving the render window.
fn handle_cursor_focus(state: &mut RuntimeLoopState, entered_window: bool) {
    if entered_window {
        let _ = state
            .app
            .core
            .window_state
            .window
            .set_cursor_grab(CursorGrabMode::Confined);
        state.app.core.window_state.window.set_cursor_visible(false);
    } else {
        let _ = state
            .app
            .core
            .window_state
            .window
            .set_cursor_grab(CursorGrabMode::None);
        state.app.core.window_state.window.set_cursor_visible(true);
    }
}

/// Handle swapchain rebuild requests after a window resize event.
fn handle_resize(state: &mut RuntimeLoopState, new_size: winit::dpi::PhysicalSize<u32>) {
    info!("Resize requested");

    state.app.core.resize_requested = true;
    let new_extent = Extent2D::default()
        .height(new_size.height)
        .width(new_size.width);

    state.app.rebuild_swapchain(new_extent)
}

/// Update camera state, build scene submission, execute a frame, and keep FPS window title updated.
fn handle_redraw_requested(state: &mut RuntimeLoopState, delta: Duration) {
    state.input_manager.update();
    state
        .app
        .core
        .window_state
        .controller
        .borrow_mut()
        .update(delta.as_secs_f32());

    if state.app.core.resize_requested {
        return;
    }

    let now = SystemTime::now();
    state.app.core.imgui.context.io_mut().update_delta_time(delta);
    state
        .app
        .core
        .imgui
        .platform
        .prepare_frame(
            state.app.core.imgui.context.io_mut(),
            &state.app.core.window_state.window,
        )
        .unwrap();
    state.last_time = now;

    let (camera_view, camera_pos) = {
        let cont = state.app.core.window_state.controller.borrow();
        (
            cont.get_camera().get_view_matrix(),
            cont.get_camera().get_position(),
        )
    };

    let fovy = 70_f32.to_radians();
    let aspect_ratio = state.app.core.window_state.get_aspect_ratio();
    // reversed depth
    let far = 0.1;
    let near = 10_000.0;
    let proj = glam::Mat4::perspective_rh(fovy, aspect_ratio, far, near);

    state
        .scene_world
        .update_camera(camera_view, proj, camera_pos);
    let submission = state.scene_world.build_submission();

    state.app.render(state.frame, &submission);
    state.app.core.window_state.window.request_redraw();
    if now.duration_since(state.fps_timer).unwrap() > Duration::from_secs(1) {
        state
            .app
            .core
            .window_state
            .window
            .set_title(format!("Frame-Rate: {}", state.frame).as_str());
        state.frame = 0;
        state.fps_timer = now;
    }
    state.frame += 1;
}

/// Route window events that are specific to the active renderer window.
fn handle_window_event(
    state: &mut RuntimeLoopState,
    control_flow: &EventLoopWindowTarget<()>,
    window_id: WindowId,
    event: &WindowEvent,
    delta: Duration,
) {
    if window_id != state.app.core.window_state.window.id() {
        return;
    }

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
        WindowEvent::Focused(_focused) => {}
        WindowEvent::KeyboardInput { event: key_event, .. } => {
            handle_keyboard_input(state, control_flow, key_event);
        }
        WindowEvent::ModifiersChanged(_modd, ..) => {}
        WindowEvent::Ime(_) => {}
        WindowEvent::CursorMoved { position: _, .. } => {}
        WindowEvent::CursorEntered { .. } => {
            handle_cursor_focus(state, true);
        }
        WindowEvent::CursorLeft { .. } => {
            handle_cursor_focus(state, false);
        }
        WindowEvent::MouseWheel { delta: _, .. } => {}
        WindowEvent::MouseInput {
            state: _,
            button: _,
            ..
        } => {}
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
            handle_resize(state, *new_size);
        }
        WindowEvent::RedrawRequested => {
            handle_redraw_requested(state, delta);
        }
    }
}

/// Run winit event pump and dispatch device/window events into renderer runtime handlers.
fn run_event_loop(event_loop: EventLoop<()>, mut state: RuntimeLoopState) {
    event_loop
        .run(move |event, control_flow| {
            state
                .app
                .core
                .imgui
                .handle_event(&state.app.core.window_state.window, &event);
            let delta = SystemTime::now().duration_since(state.last_time).unwrap();

            match event {
                Event::NewEvents(..) => {
                    state.app.core.imgui.context.io_mut().update_delta_time(delta);
                }
                Event::DeviceEvent { device_id: _, event } => {
                    handle_device_event(&mut state, event);
                }
                Event::WindowEvent {
                    ref event,
                    window_id,
                } => {
                    handle_window_event(&mut state, control_flow, window_id, event, delta);
                }
                _ => {}
            }
        })
        .expect("TODO: panic message");
}

pub fn run() {
    init_runtime_logging();
    let mut input_manager = InputManager::default();

    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);
    let args: Vec<String> = env::args().collect();
    let runtime_flags = parse_runtime_flags(&args);
    let window_state = init_window_state(&event_loop, &mut input_manager);

    let (app, scene_world) = match create_runtime_renderer(window_state, runtime_flags) {
        Some(runtime) => runtime,
        None => return,
    };

    let state = RuntimeLoopState {
        input_manager,
        app,
        scene_world,
        last_time: SystemTime::now(),
        frame: 0,
        fps_timer: SystemTime::now(),
    };

    run_event_loop(event_loop, state);
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
