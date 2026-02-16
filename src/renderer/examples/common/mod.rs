#![allow(dead_code)]

use glam::{Mat4, Vec3};
use log::{error, info};
use renderer::{
    DebugRuntimeMode, FrameRenderOutcome, Renderer, RendererConfig, RendererError, Scene,
};
use std::env;
use std::time::{Duration, Instant};
use winit::dpi::PhysicalSize;
use winit::event::{ElementState, Event, KeyEvent, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, ModifiersState, PhysicalKey};
use winit::window::{Fullscreen, Window, WindowBuilder};

const FACADE_DEMO_MODEL_PATH: &str = "src/renderer/src/assets/DamagedHelmet.glb";

#[derive(Copy, Clone)]
pub enum DemoScenario {
    Pbr,
    Unlit,
    ModelLoad,
}

impl DemoScenario {
    fn title(self) -> &'static str {
        match self {
            Self::Pbr => "renderer facade demo (pbr)",
            Self::Unlit => "renderer facade demo (unlit)",
            Self::ModelLoad => "renderer facade demo (model load)",
        }
    }

    fn debug_runtime_mode(self) -> DebugRuntimeMode {
        match self {
            Self::Unlit => DebugRuntimeMode::TestUnlit,
            Self::Pbr | Self::ModelLoad => DebugRuntimeMode::Default,
        }
    }
}

pub fn run_demo(scenario: DemoScenario) {
    init_logging();

    let event_loop = match EventLoop::new() {
        Ok(event_loop) => event_loop,
        Err(err) => {
            error!("Failed to create event loop: {err}");
            return;
        }
    };
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut config = RendererConfig::default();
    config.app_name = scenario.title().to_string();
    config.shader_debug_mode = scenario.debug_runtime_mode();

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
                error!("Shader rebuild requires 'glslc' or 'glslangValidator' in PATH.");
            }
            return;
        }
    };
    renderer.install_default_fps_input();

    let mut scene = match initialize_scene(&mut renderer, scenario) {
        Ok(scene) => scene,
        Err(err) => {
            error!("Failed to initialize demo scene: {err}");
            return;
        }
    };

    let mut fps_timer = Instant::now();
    let mut frame_counter: u32 = 0;
    let mut modifiers = ModifiersState::default();
    let mut last_window_size = window.inner_size();

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
                            if handle_fullscreen_toggle(&window, &key_event, modifiers) {
                                return;
                            }
                            if key_event.physical_key == PhysicalKey::Code(KeyCode::Escape) {
                                control_flow.exit();
                            }
                        }
                        WindowEvent::ModifiersChanged(next_modifiers) => {
                            modifiers = next_modifiers.state();
                        }
                        WindowEvent::Resized(new_size) => {
                            last_window_size = new_size;
                            if let Err(err) = renderer.resize(new_size.width, new_size.height) {
                                error!("Resize failed: {err}");
                                control_flow.exit();
                            }
                        }
                        WindowEvent::ScaleFactorChanged {
                            mut inner_size_writer,
                            ..
                        } => {
                            // Keep current inner size during DPI transitions and rebuild swapchain accordingly.
                            let new_size = window.inner_size();
                            if let Err(err) = inner_size_writer.request_inner_size(new_size) {
                                error!("Scale factor size request failed: {err}");
                                control_flow.exit();
                                return;
                            }
                            last_window_size = new_size;
                            if let Err(err) = renderer.resize(new_size.width, new_size.height) {
                                error!("Resize failed after scale change: {err}");
                                control_flow.exit();
                            }
                        }
                        WindowEvent::RedrawRequested => {
                            let current_size = window.inner_size();
                            if current_size != last_window_size {
                                last_window_size = current_size;
                                if let Err(err) =
                                    renderer.resize(current_size.width, current_size.height)
                                {
                                    error!("Resize failed while redrawing: {err}");
                                    control_flow.exit();
                                    return;
                                }
                            }
                            let outcome = match renderer.render_scene(&window, &mut scene) {
                                Ok(outcome) => outcome,
                                Err(err) => {
                                    error!("Render failed: {err}");
                                    control_flow.exit();
                                    return;
                                }
                            };

                            if outcome == FrameRenderOutcome::Rendered {
                                frame_counter = frame_counter.wrapping_add(1);
                                if fps_timer.elapsed() >= Duration::from_secs(1) {
                                    window.set_title(
                                        format!("{} - FPS: {}", app_name, frame_counter).as_str(),
                                    );
                                    fps_timer = Instant::now();
                                    frame_counter = 0;
                                }
                            }

                            if outcome == FrameRenderOutcome::SkippedResizePending {
                                window.set_title(format!("{} - resizing...", app_name).as_str());
                            }

                            window.request_redraw();
                        }
                        _ => {}
                    }
                }
                _ => {}
            }
        })
        .expect("failed to run renderer example loop");
}

fn handle_fullscreen_toggle(
    window: &Window,
    key_event: &KeyEvent,
    modifiers: ModifiersState,
) -> bool {
    if key_event.state != ElementState::Pressed || key_event.repeat {
        return false;
    }

    if key_event.physical_key != PhysicalKey::Code(KeyCode::KeyF) || !modifiers.control_key() {
        return false;
    }

    let next_mode = if window.fullscreen().is_some() {
        None
    } else {
        Some(Fullscreen::Borderless(window.current_monitor()))
    };
    window.set_fullscreen(next_mode);
    true
}

fn initialize_scene(
    renderer: &mut Renderer,
    scenario: DemoScenario,
) -> Result<Scene, RendererError> {
    match scenario {
        DemoScenario::Pbr => Ok(renderer.take_startup_scene().unwrap_or_else(Scene::new)),
        DemoScenario::Unlit => Ok(renderer.take_startup_scene().unwrap_or_else(Scene::new)),
        DemoScenario::ModelLoad => build_model_load_scene(renderer, true),
    }
}

fn build_model_load_scene(
    renderer: &mut Renderer,
    duplicate_instance: bool,
) -> Result<Scene, RendererError> {
    // Force the demo down the public facade asset + scene path.
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

fn init_logging() {
    let _ = env_logger::Builder::new()
        .target(env_logger::Target::Stdout)
        .parse_filters(&env::var("RUST_LOG").unwrap_or_else(|_| "info".to_string()))
        .try_init();

    info!("Starting facade example runtime");
}
