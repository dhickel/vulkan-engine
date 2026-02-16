use log::{error, info};
use renderer::{
    EnvironmentSource, FrameRenderOutcome, Renderer, RendererConfig, RendererError, Scene,
};
use std::env;
use std::path::PathBuf;
use std::time::{Duration, Instant};
use winit::dpi::PhysicalSize;
use winit::event::{ElementState, Event, KeyEvent, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, ModifiersState, PhysicalKey};
use winit::window::{Fullscreen, Window, WindowBuilder};

fn main() {
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
    config.app_name = "renderer facade api_test".to_string();

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

    let mut scene = renderer.take_startup_scene().unwrap_or_else(Scene::new);

    if let Some(env_path) = parse_env_arg() {
        info!("Loading custom environment from: {}", env_path.display());
        match renderer
            .assets()
            .load_environment(EnvironmentSource::Auto(env_path))
        {
            Ok(handle) => {
                scene.set_skybox(handle);
                info!("Custom environment set as skybox");
            }
            Err(err) => {
                error!("Failed to load custom environment: {err}");
            }
        }
    }

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
                            let outcome = match render_frame(&mut renderer, &window, &mut scene) {
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
        .expect("failed to run renderer api_test example loop");
}

fn render_frame(
    renderer: &mut Renderer,
    window: &Window,
    scene: &mut Scene,
) -> Result<FrameRenderOutcome, RendererError> {
    let mut frame = renderer.begin_frame(window)?;
    let outcome = renderer.render_scene_in_frame(&mut frame, scene)?;
    renderer.end_frame(frame)?;
    Ok(outcome)
}

fn parse_env_arg() -> Option<PathBuf> {
    let args: Vec<String> = env::args().collect();
    let mut i = 1;
    while i < args.len() {
        if args[i] == "--env" {
            if let Some(path) = args.get(i + 1) {
                return Some(PathBuf::from(path));
            } else {
                error!("--env requires a path argument");
                return None;
            }
        }
        i += 1;
    }
    None
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

fn init_logging() {
    let _ = env_logger::Builder::new()
        .target(env_logger::Target::Stdout)
        .parse_filters(&env::var("RUST_LOG").unwrap_or_else(|_| "info".to_string()))
        .try_init();

    info!("Starting renderer facade api_test example");
}
