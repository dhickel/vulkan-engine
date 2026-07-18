mod common;

use log::{error, info, warn};
use renderer::prelude::{
    FrameCaptureStatus, FrameRenderOutcome, LoadStatus, LoadTicket, Renderer, RendererConfig,
    RendererError, Scene,
};
use std::collections::HashSet;
use std::env;
use std::time::{Duration, Instant};
use winit::dpi::PhysicalSize;
use winit::event::{ElementState, Event, KeyEvent, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, ModifiersState, PhysicalKey};
use winit::window::{Fullscreen, Window, WindowBuilder};

const DEFAULT_MODEL_PATH: &str = "src/renderer/src/assets/DamagedHelmet.glb";

fn main() {
    init_logging();
    let launch_options = match common::parse_launch_options() {
        Ok(options) => options,
        Err(err) => {
            error!("Failed to parse launch arguments: {err}");
            return;
        }
    };

    let config = RendererConfig {
        app_name: "renderer facade demo (async loading)".to_string(),
        headless: launch_options.headless,
        ..RendererConfig::default()
    };

    let app_name = config.app_name.clone();
    if launch_options.headless {
        run_headless_async_demo(config, launch_options, &app_name);
        return;
    }

    let event_loop = match EventLoop::new() {
        Ok(event_loop) => event_loop,
        Err(err) => {
            error!("Failed to create event loop: {err}");
            return;
        }
    };
    event_loop.set_control_flow(ControlFlow::Poll);

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
    renderer.install_default_fps_input();
    if let Err(err) =
        common::apply_frame_capture_launch_options(&mut renderer, &launch_options, &app_name)
    {
        error!("Failed to configure frame capture: {err}");
        return;
    }
    match common::apply_debug_record_launch_options(&mut renderer, &launch_options) {
        Ok(Some(path)) => info!("Debug timing recording active -> {}", path),
        Ok(None) => {
            if launch_options.record_debug_interval_ms.is_some()
                || launch_options.record_debug_path.is_some()
            {
                info!("Debug timing defaults configured (not started): use --record_debug=<seconds> to start on launch");
            }
        }
        Err(err) => {
            error!("Failed to configure debug timing recording: {err}");
            return;
        }
    }

    // Force the demo down the public facade ticket-loading path.
    let _ = renderer.take_startup_scene();
    let mut scene = Scene::new();

    let model_path = launch_options
        .model_path
        .as_deref()
        .unwrap_or(std::path::Path::new(DEFAULT_MODEL_PATH));
    let load_ticket = {
        let mut assets = renderer.assets();
        match assets.request_model_load(model_path) {
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
                            if let Err(err) = poll_async_load(
                                &mut renderer,
                                &mut scene,
                                &mut load_state,
                                Some(&window),
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

fn run_headless_async_demo(
    config: RendererConfig,
    launch_options: common::LaunchOptions,
    app_name: &str,
) {
    let mut renderer = match Renderer::new_headless(config.clone()) {
        Ok(renderer) => renderer,
        Err(err) => {
            error!("Headless renderer initialization failed: {err}");
            return;
        }
    };
    if let Err(err) =
        common::apply_frame_capture_launch_options(&mut renderer, &launch_options, app_name)
    {
        error!("Failed to configure frame capture: {err}");
        return;
    }
    if let Err(err) = common::apply_debug_record_launch_options(&mut renderer, &launch_options) {
        error!("Failed to configure debug timing: {err}");
        return;
    }

    let _ = renderer.take_startup_scene();
    let mut scene = Scene::new();
    let model_path = launch_options
        .model_path
        .as_deref()
        .unwrap_or(std::path::Path::new(DEFAULT_MODEL_PATH));
    let load_ticket = {
        let mut assets = renderer.assets();
        match assets.request_model_load(model_path) {
            Ok(ticket) => ticket,
            Err(err) => {
                error!("Failed to queue model load: {err}");
                return;
            }
        }
    };
    let mut load_state = AsyncLoadState::new(load_ticket);
    let load_deadline = Instant::now() + Duration::from_secs(30);
    while !load_state.model_mounted {
        if let Err(err) = renderer.pump_asset_tasks(64) {
            error!("Asset pump failed: {err}");
            return;
        }
        if let Err(err) =
            poll_async_load(&mut renderer, &mut scene, &mut load_state, None, app_name)
        {
            error!("Asset polling failed: {err}");
            return;
        }
        if Instant::now() >= load_deadline {
            error!("Headless async model load did not complete within 30 seconds");
            return;
        }
        std::thread::yield_now();
    }

    let expected_captures = usize::from(launch_options.capture_frame.is_some())
        + launch_options.capture_frames.unwrap_or(0) as usize;
    let mut succeeded_paths = HashSet::new();

    for _ in 0..240 {
        if let Err(err) = renderer.render_scene_headless(&mut scene) {
            error!("Headless render failed: {err}");
            return;
        }

        match renderer.last_frame_capture_status() {
            Some(FrameCaptureStatus::Succeeded { output_path, .. }) => {
                succeeded_paths.insert(output_path.clone());
                if succeeded_paths.len() >= expected_captures {
                    info!(
                        "Headless capture completed: {} capture(s) written",
                        succeeded_paths.len()
                    );
                    return;
                }
            }
            Some(FrameCaptureStatus::Failed { message, .. }) => {
                error!("Headless capture failed: {message}");
                return;
            }
            _ if expected_captures == 0 => {
                info!("Headless async model load completed");
                return;
            }
            _ => {}
        }
    }

    error!(
        "Headless async demo did not complete ({} of {} capture(s) written)",
        succeeded_paths.len(),
        expected_captures
    );
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
    window: Option<&winit::window::Window>,
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
                if let Some(window) = window {
                    window.set_title(format!("{} - loading ({queued_ms}ms)", app_name).as_str());
                }
                state.last_pending_log = Instant::now();
            }
            Ok(())
        }
        LoadStatus::Uploaded { value } => {
            let mount = scene.merge_fragment(None, value)?;
            scene.set_transform(mount.mounted_root, glam::Mat4::IDENTITY)?;
            state.model_mounted = true;
            if let Some(window) = window {
                window.set_title(format!("{} - model ready", app_name).as_str());
            }
            info!("Async model load completed and scene fragment mounted");
            Ok(())
        }
        LoadStatus::Failed { error } => Err(RendererError::Asset(error)),
        LoadStatus::Cancelled => {
            warn!("Async model load ticket was cancelled");
            Err(RendererError::InvalidState(
                "async model load ticket cancelled before upload completed".to_string(),
            ))
        }
    }
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

    info!("Starting facade async loading demo");
}
