mod common;

use log::{error, info};
use renderer::prelude::{
    default_capture_run_dir, CaptureTarget, EnvironmentSource, FrameCaptureStatus,
    FrameRenderOutcome, Renderer, RendererConfig, RendererError, Scene,
};
use std::collections::HashSet;
use std::env;
use std::time::{Duration, Instant};
use winit::dpi::PhysicalSize;
use winit::event::{ElementState, Event, KeyEvent, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, ModifiersState, PhysicalKey};
use winit::window::{Fullscreen, Window, WindowBuilder};

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
        app_name: "renderer facade api_test".to_string(),
        headless: launch_options.headless,
        ..RendererConfig::default()
    };

    let app_name = config.app_name.clone();
    let capture_run_dir = default_capture_run_dir(&app_name);
    if launch_options.headless {
        run_headless_api_test(config, launch_options, &app_name, &capture_run_dir);
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
    if let Err(err) = common::apply_frame_capture_launch_options_with_run_dir(
        &mut renderer,
        &launch_options,
        &app_name,
        &capture_run_dir,
    ) {
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

    let mut scene = renderer.take_startup_scene().unwrap_or_else(Scene::new);

    if let Some(env_path) = launch_options.env_path {
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
                            if handle_manual_capture_key(&mut renderer, &key_event) {
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

fn run_headless_api_test(
    config: RendererConfig,
    launch_options: common::LaunchOptions,
    app_name: &str,
    capture_run_dir: &std::path::Path,
) {
    let mut renderer = match Renderer::new_headless(config.clone()) {
        Ok(renderer) => renderer,
        Err(err) => {
            error!("Headless renderer initialization failed: {err}");
            if config.compile_shaders {
                error!("Shader rebuild requires 'glslc' or 'glslangValidator' in PATH.");
            }
            return;
        }
    };

    if let Err(err) = common::apply_frame_capture_launch_options_with_run_dir(
        &mut renderer,
        &launch_options,
        app_name,
        capture_run_dir,
    ) {
        error!("Failed to configure frame capture: {err}");
        return;
    }
    match common::apply_debug_record_launch_options(&mut renderer, &launch_options) {
        Ok(Some(path)) => info!("Debug timing recording active -> {}", path),
        Ok(None) => {}
        Err(err) => {
            error!("Failed to configure debug timing recording: {err}");
            return;
        }
    }

    let mut scene = renderer.take_startup_scene().unwrap_or_else(Scene::new);
    if let Some(env_path) = launch_options.env_path.as_ref() {
        info!("Loading custom environment from: {}", env_path.display());
        match renderer
            .assets()
            .load_environment(EnvironmentSource::Auto(env_path.clone()))
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

    let expected_captures = launch_options
        .capture_frames
        .or_else(|| launch_options.capture_frame.map(|_| 1))
        .unwrap_or(0) as usize;
    let frame_budget = launch_options
        .capture_frame
        .or(launch_options.capture_frame_start)
        .unwrap_or(0)
        .saturating_add(
            launch_options
                .capture_frame_interval
                .unwrap_or(1)
                .saturating_mul(launch_options.capture_frames.unwrap_or(0).saturating_add(2)),
        )
        .max(180);
    let mut succeeded_paths = HashSet::new();

    for _ in 0..frame_budget {
        match renderer.render_scene_headless(&mut scene) {
            Ok(FrameRenderOutcome::Rendered)
            | Ok(FrameRenderOutcome::SkippedAcquireUnavailable)
            | Ok(FrameRenderOutcome::SkippedResizePending)
            | Ok(FrameRenderOutcome::SubmittedNotPresented)
            | Ok(FrameRenderOutcome::PresentedSuboptimal) => {}
            Err(err) => {
                error!("Headless render failed: {err}");
                return;
            }
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
            _ => {}
        }
    }

    if expected_captures > 0 {
        error!(
            "Headless capture did not complete within {} frames ({} of {} capture(s) written)",
            frame_budget,
            succeeded_paths.len(),
            expected_captures
        );
    }
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

fn handle_manual_capture_key(renderer: &mut Renderer, key_event: &KeyEvent) -> bool {
    if key_event.state != ElementState::Pressed || key_event.repeat {
        return false;
    }

    if key_event.physical_key != PhysicalKey::Code(KeyCode::F10) {
        return false;
    }

    if let Err(err) = renderer.queue_manual_frame_capture(CaptureTarget::Present) {
        error!("Manual frame capture request rejected: {err}");
    }
    true
}

fn init_logging() {
    let _ = env_logger::Builder::new()
        .target(env_logger::Target::Stdout)
        .parse_filters(&env::var("RUST_LOG").unwrap_or_else(|_| "info".to_string()))
        .try_init();

    info!("Starting renderer facade api_test example");
}
