use log::{error, info, warn};
use renderer::{
    FrameRenderOutcome, LoadStatus, LoadTicket, Renderer, RendererConfig, RendererError, Scene,
};
use std::env;
use std::time::{Duration, Instant};
use winit::dpi::PhysicalSize;
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::WindowBuilder;

const DEMO_MODEL_PATH: &str = "src/renderer/src/assets/DamagedHelmet.glb";

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
    config.app_name = "renderer facade demo (async loading)".to_string();

    let app_name = config.app_name.clone();
    let window = match WindowBuilder::new()
        .with_title(format!("{} - requesting model", app_name))
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

    // Force the demo down the public facade ticket-loading path.
    let _ = renderer.take_startup_scene();
    let mut scene = Scene::new();

    let load_ticket = {
        let mut assets = renderer.assets();
        match assets.request_model_load(DEMO_MODEL_PATH) {
            Ok(ticket) => ticket,
            Err(err) => {
                error!("Failed to queue model load: {err}");
                return;
            }
        }
    };

    let mut load_state = AsyncLoadState::new(load_ticket);
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
                            if let Err(err) = poll_async_load(
                                &mut renderer,
                                &mut scene,
                                &mut load_state,
                                &window,
                                app_name.as_str(),
                            ) {
                                error!("Asset polling failed: {err}");
                                control_flow.exit();
                                return;
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
                                    let state_label = if load_state.model_mounted {
                                        "ready"
                                    } else {
                                        "loading"
                                    };
                                    window.set_title(
                                        format!(
                                            "{} - {} - FPS: {}",
                                            app_name, state_label, frame_counter
                                        )
                                        .as_str(),
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
        .expect("failed to run renderer async loading example loop");
}

struct AsyncLoadState {
    ticket: LoadTicket,
    model_mounted: bool,
    last_pending_log: Instant,
}

impl AsyncLoadState {
    fn new(ticket: LoadTicket) -> Self {
        Self {
            ticket,
            model_mounted: false,
            last_pending_log: Instant::now(),
        }
    }
}

fn poll_async_load(
    renderer: &mut Renderer,
    scene: &mut Scene,
    state: &mut AsyncLoadState,
    window: &winit::window::Window,
    app_name: &str,
) -> Result<(), RendererError> {
    if state.model_mounted {
        return Ok(());
    }

    let status = {
        let mut assets = renderer.assets();
        assets.poll_model_load(state.ticket)
    };

    match status {
        LoadStatus::Pending { queued_at } => {
            if state.last_pending_log.elapsed() >= Duration::from_millis(500) {
                let queued_ms = queued_at.elapsed().as_millis();
                info!("Model load still pending after {queued_ms}ms");
                window.set_title(format!("{} - loading ({queued_ms}ms)", app_name).as_str());
                state.last_pending_log = Instant::now();
            }
            Ok(())
        }
        LoadStatus::Uploaded { value } => {
            let mount = scene.merge_fragment(None, value)?;
            scene.set_transform(mount.mounted_root, glam::Mat4::IDENTITY)?;
            state.model_mounted = true;
            window.set_title(format!("{} - model ready", app_name).as_str());
            info!("Async model load completed and scene fragment mounted");
            Ok(())
        }
        LoadStatus::Failed { error } => Err(RendererError::Asset(error)),
        LoadStatus::Cancelled => {
            warn!("Async model load ticket was cancelled");
            Err(RendererError::InvalidState(
                "async model load ticket cancelled before upload completed",
            ))
        }
    }
}

fn init_logging() {
    let _ = env_logger::Builder::new()
        .target(env_logger::Target::Stdout)
        .parse_filters(&env::var("RUST_LOG").unwrap_or_else(|_| "info".to_string()))
        .try_init();

    info!("Starting facade async loading demo");
}
