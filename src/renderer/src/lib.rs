#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(unused_variables)]

pub mod api;

mod data;
mod rendergraph;
mod scene;
mod texture;
mod vulkan;

pub use api::{
    AssetError, AssetManager, DebugRuntimeMode, EnvironmentHandle, FrameContext, HookError,
    MeshHandle, Renderer, RendererConfig, RendererError, RendererFrameError, RendererInitError,
    Scene, SceneError, SceneFragment, SceneFragmentMount, SceneFragmentNode, SceneFragmentNodeId,
    SceneNodeId,
};

use log::{error, info, warn};
use std::env;
use std::time::{Duration, Instant};
use winit::dpi::PhysicalSize;
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::WindowBuilder;

#[derive(Copy, Clone)]
struct RuntimeFlags {
    compile_shaders: bool,
    debug_runtime_mode: DebugRuntimeMode,
}

/// Initialize logger output once at startup for all runtime subsystems.
fn init_runtime_logging() {
    env_logger::Builder::new()
        .target(env_logger::Target::Stdout)
        .parse_filters(&env::var("RUST_LOG").unwrap_or_else(|_| "debug".to_string()))
        .init();
}

fn parse_debug_runtime_mode(args: &[String]) -> DebugRuntimeMode {
    let mut i = 0usize;
    while i < args.len() {
        let arg = &args[i];

        if arg == "debug_runtime" || arg == "--debug-runtime" {
            let Some(value) = args.get(i + 1) else {
                warn!(
                    "Missing debug runtime mode after '{}'. Valid values: testpbr, testunlit",
                    arg
                );
                return DebugRuntimeMode::Default;
            };

            if let Some(mode) = DebugRuntimeMode::from_label(value) {
                return mode;
            }

            warn!(
                "Unsupported debug runtime mode '{}'. Valid values: testpbr, testunlit",
                value
            );
            return DebugRuntimeMode::Default;
        }

        if let Some(value) = arg.strip_prefix("--debug-runtime=") {
            if let Some(mode) = DebugRuntimeMode::from_label(value) {
                return mode;
            }

            warn!(
                "Unsupported debug runtime mode '{}'. Valid values: testpbr, testunlit",
                value
            );
            return DebugRuntimeMode::Default;
        }

        i += 1;
    }

    DebugRuntimeMode::Default
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

    if debug_runtime_mode != DebugRuntimeMode::Default {
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

pub fn run() {
    init_runtime_logging();

    let event_loop = match EventLoop::new() {
        Ok(event_loop) => event_loop,
        Err(err) => {
            error!("Failed to create event loop: {err}");
            return;
        }
    };
    event_loop.set_control_flow(ControlFlow::Poll);

    let args: Vec<String> = env::args().collect();
    let runtime_flags = parse_runtime_flags(&args);

    let mut config = RendererConfig::default();
    config.compile_shaders = runtime_flags.compile_shaders;
    config.shader_debug_mode = runtime_flags.debug_runtime_mode;

    let app_name = config.app_name.clone();
    let window = match WindowBuilder::new()
        .with_title(app_name.clone())
        .with_inner_size(PhysicalSize::new(config.window_width, config.window_height))
        .build(&event_loop)
    {
        Ok(window) => window,
        Err(err) => {
            error!("Failed to create window: {err}");
            return;
        }
    };

    let mut renderer = match Renderer::new(config.clone(), &window) {
        Ok(renderer) => renderer,
        Err(err) => {
            error!("Renderer initialization failed: {err}");
            if config.compile_shaders {
                error!("Shader rebuild requires 'glslc' or 'glslangValidator' in PATH. Install Vulkan shader tools (e.g. package 'shaderc' or LunarG Vulkan SDK), then retry.");
            }
            return;
        }
    };

    let mut scene = renderer.take_startup_scene().unwrap_or_else(Scene::new);
    let mut fps_timer = Instant::now();
    let mut frame_counter: u32 = 0;

    window.request_redraw();

    event_loop
        .run(move |event, control_flow| {
            if let Err(err) = renderer.update_input(&window, &event) {
                error!("Input update failed: {err}");
                control_flow.exit();
                return;
            }

            match event {
                Event::WindowEvent { window_id, event } if window_id == window.id() => {
                    match event {
                        WindowEvent::CloseRequested => {
                            control_flow.exit();
                        }
                        WindowEvent::KeyboardInput {
                            event: key_event, ..
                        } => {
                            if key_event.physical_key == PhysicalKey::Code(KeyCode::Escape) {
                                control_flow.exit();
                            }
                        }
                        WindowEvent::Resized(new_size) => {
                            if let Err(err) = renderer.resize(new_size.width, new_size.height) {
                                error!("Resize failed: {err}");
                                control_flow.exit();
                            }
                        }
                        WindowEvent::RedrawRequested => {
                            if let Err(err) = renderer.render_scene(&window, &mut scene) {
                                error!("Render failed: {err}");
                                control_flow.exit();
                                return;
                            }

                            frame_counter = frame_counter.wrapping_add(1);
                            if fps_timer.elapsed() >= Duration::from_secs(1) {
                                window.set_title(
                                    format!("{} - FPS: {}", app_name, frame_counter).as_str(),
                                );
                                fps_timer = Instant::now();
                                frame_counter = 0;
                            }

                            window.request_redraw();
                        }
                        _ => {}
                    }
                }
                _ => {}
            }
        })
        .expect("failed to run renderer compatibility loop");
}
