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
    MaterialHandle, MeshHandle, Renderer, RendererConfig, RendererError, RendererFrameError,
    RendererInitError, Scene, SceneError, SceneFragment, SceneFragmentMount, SceneFragmentNode,
    SceneFragmentNodeId, SceneNodeId, TextureHandle,
};

use glam::{Mat4, Vec3};
use log::{error, info, warn};
use std::env;
use std::time::{Duration, Instant};
use winit::dpi::PhysicalSize;
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::WindowBuilder;

const FACADE_DEMO_MODEL_PATH: &str = "src/renderer/src/assets/DamagedHelmet.glb";
const VALID_DEBUG_RUNTIME_LABELS: &str =
    "default, testpbr, testunlit, facade_pbr, facade_unlit, facade_model_load";

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum RuntimeScenario {
    Legacy(DebugRuntimeMode),
    FacadePbr,
    FacadeUnlit,
    FacadeModelLoad,
}

impl RuntimeScenario {
    fn from_label(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "default" => Some(Self::Legacy(DebugRuntimeMode::Default)),
            "testpbr" => Some(Self::Legacy(DebugRuntimeMode::TestPbr)),
            "testunlit" => Some(Self::Legacy(DebugRuntimeMode::TestUnlit)),
            "facade_pbr" => Some(Self::FacadePbr),
            "facade_unlit" => Some(Self::FacadeUnlit),
            "facade_model_load" => Some(Self::FacadeModelLoad),
            _ => None,
        }
    }

    fn as_label(self) -> &'static str {
        match self {
            Self::Legacy(mode) => mode.as_label(),
            Self::FacadePbr => "facade_pbr",
            Self::FacadeUnlit => "facade_unlit",
            Self::FacadeModelLoad => "facade_model_load",
        }
    }

    fn startup_debug_mode(self) -> DebugRuntimeMode {
        match self {
            Self::Legacy(mode) => mode,
            // Keep unlit startup behavior for parity until explicit material-override API exists.
            Self::FacadeUnlit => DebugRuntimeMode::TestUnlit,
            Self::FacadePbr | Self::FacadeModelLoad => DebugRuntimeMode::Default,
        }
    }
}

#[derive(Copy, Clone)]
struct RuntimeFlags {
    compile_shaders: bool,
    runtime_scenario: RuntimeScenario,
}

/// Initialize logger output once at startup for all runtime subsystems.
fn init_runtime_logging() {
    env_logger::Builder::new()
        .target(env_logger::Target::Stdout)
        .parse_filters(&env::var("RUST_LOG").unwrap_or_else(|_| "debug".to_string()))
        .init();
}

fn parse_runtime_scenario(args: &[String]) -> RuntimeScenario {
    let mut i = 0usize;
    while i < args.len() {
        let arg = &args[i];

        if arg == "debug_runtime" || arg == "--debug-runtime" {
            let Some(value) = args.get(i + 1) else {
                warn!(
                    "Missing debug runtime mode after '{}'. Valid values: {}",
                    arg, VALID_DEBUG_RUNTIME_LABELS
                );
                return RuntimeScenario::Legacy(DebugRuntimeMode::Default);
            };

            if let Some(mode) = RuntimeScenario::from_label(value) {
                return mode;
            }

            warn!(
                "Unsupported debug runtime mode '{}'. Valid values: {}",
                value, VALID_DEBUG_RUNTIME_LABELS
            );
            return RuntimeScenario::Legacy(DebugRuntimeMode::Default);
        }

        if let Some(value) = arg.strip_prefix("--debug-runtime=") {
            if let Some(mode) = RuntimeScenario::from_label(value) {
                return mode;
            }

            warn!(
                "Unsupported debug runtime mode '{}'. Valid values: {}",
                value, VALID_DEBUG_RUNTIME_LABELS
            );
            return RuntimeScenario::Legacy(DebugRuntimeMode::Default);
        }

        i += 1;
    }

    RuntimeScenario::Legacy(DebugRuntimeMode::Default)
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
    let runtime_scenario = parse_runtime_scenario(args);
    let compile_shaders = rebuild_from_env || rebuild_from_args;

    if compile_shaders {
        info!("Shader rebuild requested (--rebuild-shaders or ENGINE_REBUILD_SHADERS=1).");
    }

    if runtime_scenario != RuntimeScenario::Legacy(DebugRuntimeMode::Default) {
        info!(
            "Debug runtime mode selected: {}",
            runtime_scenario.as_label()
        );
    }

    RuntimeFlags {
        compile_shaders,
        runtime_scenario,
    }
}

fn build_facade_model_scene(
    renderer: &mut Renderer,
    duplicate_instance: bool,
) -> Result<Scene, RendererError> {
    // Discard internal startup scene and rebuild through facade-only APIs for dogfooding.
    let _ = renderer.take_startup_scene();
    let mut scene = Scene::new();

    let first_fragment = {
        let mut assets = renderer.assets();
        assets.load_model(FACADE_DEMO_MODEL_PATH)?
    };
    let first_mount = scene.merge_fragment(None, first_fragment)?;

    if duplicate_instance {
        let second_fragment = {
            let mut assets = renderer.assets();
            assets.load_model(FACADE_DEMO_MODEL_PATH)?
        };
        let second_mount = scene.merge_fragment(None, second_fragment)?;
        scene.set_transform(
            first_mount.mounted_root,
            Mat4::from_translation(Vec3::new(-2.5, 0.0, 0.0)),
        )?;
        scene.set_transform(
            second_mount.mounted_root,
            Mat4::from_translation(Vec3::new(2.5, 0.0, 0.0)),
        )?;
    }

    Ok(scene)
}

fn initialize_scene_for_runtime(
    renderer: &mut Renderer,
    runtime_scenario: RuntimeScenario,
) -> Result<Scene, RendererError> {
    match runtime_scenario {
        RuntimeScenario::Legacy(_) => Ok(renderer.take_startup_scene().unwrap_or_else(Scene::new)),
        RuntimeScenario::FacadePbr => Ok(renderer.take_startup_scene().unwrap_or_else(Scene::new)),
        RuntimeScenario::FacadeUnlit => {
            Ok(renderer.take_startup_scene().unwrap_or_else(Scene::new))
        }
        RuntimeScenario::FacadeModelLoad => build_facade_model_scene(renderer, true),
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
    config.shader_debug_mode = runtime_flags.runtime_scenario.startup_debug_mode();

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

    let mut scene =
        match initialize_scene_for_runtime(&mut renderer, runtime_flags.runtime_scenario) {
            Ok(scene) => scene,
            Err(err) => {
                error!(
                    "Failed to initialize runtime scene for '{}': {err}",
                    runtime_flags.runtime_scenario.as_label()
                );
                return;
            }
        };
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
