// Shared example infrastructure is compiled separately into each sibling example binary,
// so helpers used only by other examples carry narrow dead-code allowances below.

use glam::{Mat4, Vec3};
use log::{error, info};
use renderer::prelude::{
    default_capture_run_dir, single_capture_path, CaptureTarget, DebugRuntimeMode,
    FrameCaptureRequest, FrameCaptureSequence, FrameCaptureStatus, FrameRenderOutcome, Renderer,
    RendererConfig, RendererError, Scene,
};
use std::collections::HashSet;
use std::env;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use winit::dpi::PhysicalSize;
use winit::event::{ElementState, Event, KeyEvent, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, ModifiersState, PhysicalKey};
use winit::window::{Fullscreen, Window, WindowBuilder};

#[allow(
    dead_code,
    reason = "shared helper constant is not used by every example binary"
)]
const FACADE_DEMO_MODEL_PATH: &str = "src/renderer/src/assets/DamagedHelmet.glb";

#[allow(dead_code, reason = "each demo binary constructs one scenario variant")]
#[derive(Copy, Clone)]
pub enum DemoScenario {
    Pbr,
    Unlit,
    ModelLoad,
}

#[allow(
    dead_code,
    reason = "scenario helpers are unused in non-demo example binaries"
)]
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

#[derive(Clone, Debug, Default)]
pub struct LaunchOptions {
    pub env_path: Option<PathBuf>,
    pub model_path: Option<PathBuf>,
    pub record_debug_secs: Option<u64>,
    pub record_debug_interval_ms: Option<u64>,
    pub record_debug_path: Option<String>,
    pub capture_frame: Option<u32>,
    pub capture_frame_path: Option<PathBuf>,
    pub capture_frames: Option<u32>,
    pub capture_frame_start: Option<u32>,
    pub capture_frame_interval: Option<u32>,
    pub capture_dir: Option<PathBuf>,
    pub capture_target: CaptureTarget,
    pub headless: bool,
    pub manual_capture_dir: Option<PathBuf>,
}

pub fn parse_launch_options() -> Result<LaunchOptions, String> {
    parse_launch_options_from(env::args().skip(1))
}

pub fn parse_launch_options_from(
    args: impl IntoIterator<Item = impl Into<String>>,
) -> Result<LaunchOptions, String> {
    let args: Vec<String> = args.into_iter().map(Into::into).collect();
    let mut options = LaunchOptions::default();
    let mut i = 0;
    while i < args.len() {
        let arg = args[i].as_str();
        if arg == "--env" {
            let Some(value) = args.get(i + 1) else {
                return Err("--env requires a path argument".to_string());
            };
            options.env_path = Some(PathBuf::from(value));
            i += 2;
            continue;
        }

        if let Some(value) = arg.strip_prefix("--env=") {
            options.env_path = Some(PathBuf::from(value));
            i += 1;
            continue;
        }

        if arg == "--model" {
            let Some(value) = args.get(i + 1) else {
                return Err("--model requires a path argument".to_string());
            };
            options.model_path = Some(PathBuf::from(value));
            i += 2;
            continue;
        }

        if let Some(value) = arg.strip_prefix("--model=") {
            options.model_path = Some(PathBuf::from(value));
            i += 1;
            continue;
        }

        if arg == "--record_debug" {
            let Some(value) = args.get(i + 1) else {
                return Err("--record_debug requires seconds (e.g. --record_debug=10)".to_string());
            };
            options.record_debug_secs = Some(parse_positive_u64("--record_debug", value)?);
            i += 2;
            continue;
        }

        if let Some(value) = arg.strip_prefix("--record_debug=") {
            options.record_debug_secs = Some(parse_positive_u64("--record_debug", value)?);
            i += 1;
            continue;
        }

        if arg == "--record_debug_interval" {
            let Some(value) = args.get(i + 1) else {
                return Err(
                    "--record_debug_interval requires milliseconds (e.g. --record_debug_interval=100)"
                        .to_string(),
                );
            };
            options.record_debug_interval_ms =
                Some(parse_positive_u64("--record_debug_interval", value)?);
            i += 2;
            continue;
        }

        if let Some(value) = arg.strip_prefix("--record_debug_interval=") {
            options.record_debug_interval_ms =
                Some(parse_positive_u64("--record_debug_interval", value)?);
            i += 1;
            continue;
        }

        if arg == "--record_debug_path" {
            let Some(value) = args.get(i + 1) else {
                return Err(
                    "--record_debug_path requires a file path (e.g. --record_debug_path=timing.jsonl)"
                        .to_string(),
                );
            };
            options.record_debug_path = Some(value.to_string());
            i += 2;
            continue;
        }

        if let Some(value) = arg.strip_prefix("--record_debug_path=") {
            options.record_debug_path = Some(value.to_string());
            i += 1;
            continue;
        }

        if arg == "--capture_frame" {
            let Some(value) = args.get(i + 1) else {
                return Err("--capture_frame requires a frame number".to_string());
            };
            options.capture_frame = Some(parse_u32("--capture_frame", value)?);
            i += 2;
            continue;
        }

        if let Some(value) = arg.strip_prefix("--capture_frame=") {
            options.capture_frame = Some(parse_u32("--capture_frame", value)?);
            i += 1;
            continue;
        }

        if arg == "--capture_frame_path" {
            let Some(value) = args.get(i + 1) else {
                return Err("--capture_frame_path requires a file path".to_string());
            };
            options.capture_frame_path = Some(PathBuf::from(value));
            i += 2;
            continue;
        }

        if let Some(value) = arg.strip_prefix("--capture_frame_path=") {
            options.capture_frame_path = Some(PathBuf::from(value));
            i += 1;
            continue;
        }

        if arg == "--capture_frames" {
            let Some(value) = args.get(i + 1) else {
                return Err("--capture_frames requires a frame count".to_string());
            };
            options.capture_frames = Some(parse_positive_u32("--capture_frames", value)?);
            i += 2;
            continue;
        }

        if let Some(value) = arg.strip_prefix("--capture_frames=") {
            options.capture_frames = Some(parse_positive_u32("--capture_frames", value)?);
            i += 1;
            continue;
        }

        if arg == "--capture_frame_start" {
            let Some(value) = args.get(i + 1) else {
                return Err("--capture_frame_start requires a frame number".to_string());
            };
            options.capture_frame_start = Some(parse_u32("--capture_frame_start", value)?);
            i += 2;
            continue;
        }

        if let Some(value) = arg.strip_prefix("--capture_frame_start=") {
            options.capture_frame_start = Some(parse_u32("--capture_frame_start", value)?);
            i += 1;
            continue;
        }

        if arg == "--capture_frame_interval" {
            let Some(value) = args.get(i + 1) else {
                return Err("--capture_frame_interval requires a frame interval".to_string());
            };
            options.capture_frame_interval =
                Some(parse_positive_u32("--capture_frame_interval", value)?);
            i += 2;
            continue;
        }

        if let Some(value) = arg.strip_prefix("--capture_frame_interval=") {
            options.capture_frame_interval =
                Some(parse_positive_u32("--capture_frame_interval", value)?);
            i += 1;
            continue;
        }

        if arg == "--capture_dir" {
            let Some(value) = args.get(i + 1) else {
                return Err("--capture_dir requires a directory path".to_string());
            };
            options.capture_dir = Some(PathBuf::from(value));
            i += 2;
            continue;
        }

        if let Some(value) = arg.strip_prefix("--capture_dir=") {
            options.capture_dir = Some(PathBuf::from(value));
            i += 1;
            continue;
        }

        if arg == "--capture_target" {
            let Some(value) = args.get(i + 1) else {
                return Err("--capture_target requires present or draw".to_string());
            };
            options.capture_target = parse_capture_target(value)?;
            i += 2;
            continue;
        }

        if let Some(value) = arg.strip_prefix("--capture_target=") {
            options.capture_target = parse_capture_target(value)?;
            i += 1;
            continue;
        }

        if arg == "--headless" {
            options.headless = true;
            i += 1;
            continue;
        }

        if arg == "--manual_capture_dir" {
            let Some(value) = args.get(i + 1) else {
                return Err("--manual_capture_dir requires a directory path".to_string());
            };
            options.manual_capture_dir = Some(PathBuf::from(value));
            i += 2;
            continue;
        }

        if let Some(value) = arg.strip_prefix("--manual_capture_dir=") {
            options.manual_capture_dir = Some(PathBuf::from(value));
            i += 1;
            continue;
        }

        i += 1;
    }

    validate_capture_options(&options)?;
    Ok(options)
}

pub fn apply_debug_record_launch_options(
    renderer: &mut Renderer,
    options: &LaunchOptions,
) -> Result<Option<String>, RendererError> {
    if options.record_debug_secs.is_none()
        && options.record_debug_interval_ms.is_none()
        && options.record_debug_path.is_none()
    {
        return Ok(None);
    }

    renderer.configure_debug_timing_recording(
        options.record_debug_secs,
        options.record_debug_interval_ms,
        options.record_debug_path.clone(),
    )?;

    if options.record_debug_secs.is_some() {
        return renderer.start_debug_timing_recording().map(Some);
    }

    Ok(None)
}

#[allow(
    dead_code,
    reason = "shared helper is not used by every example binary"
)]
pub fn apply_frame_capture_launch_options(
    renderer: &mut Renderer,
    options: &LaunchOptions,
    app_name: &str,
) -> Result<(), RendererError> {
    let capture_run_dir = default_capture_run_dir(app_name);
    apply_frame_capture_launch_options_with_run_dir(renderer, options, app_name, &capture_run_dir)
}

pub fn apply_frame_capture_launch_options_with_run_dir(
    renderer: &mut Renderer,
    options: &LaunchOptions,
    app_name: &str,
    capture_run_dir: &Path,
) -> Result<(), RendererError> {
    renderer.configure_manual_frame_capture_dir(
        options
            .manual_capture_dir
            .clone()
            .or_else(|| Some(capture_run_dir.to_path_buf())),
    )?;

    if let Some(frame_number) = options.capture_frame {
        let output_path = options.capture_frame_path.clone().unwrap_or_else(|| {
            single_capture_path(
                capture_run_dir,
                app_name,
                frame_number,
                options.capture_target,
            )
        });
        renderer.request_frame_capture_at(
            frame_number,
            FrameCaptureRequest::new(options.capture_target, output_path),
        )?;
    }

    if let Some(count) = options.capture_frames {
        let output_dir = options
            .capture_dir
            .clone()
            .unwrap_or_else(|| capture_run_dir.to_path_buf());
        let sequence = FrameCaptureSequence::new(
            options.capture_target,
            output_dir,
            options.capture_frame_start.unwrap_or(0),
            options.capture_frame_interval.unwrap_or(1),
            count,
        )?;
        renderer.configure_frame_capture_sequence(sequence)?;
    }

    Ok(())
}

fn parse_positive_u64(flag: &str, value: &str) -> Result<u64, String> {
    let parsed = value
        .parse::<u64>()
        .map_err(|_| format!("{flag} expects a positive integer, got '{value}'"))?;
    if parsed == 0 {
        return Err(format!("{flag} expects a value >= 1, got '{value}'"));
    }
    Ok(parsed)
}

fn parse_u32(flag: &str, value: &str) -> Result<u32, String> {
    value
        .parse::<u32>()
        .map_err(|_| format!("{flag} expects an integer, got '{value}'"))
}

fn parse_positive_u32(flag: &str, value: &str) -> Result<u32, String> {
    let parsed = parse_u32(flag, value)?;
    if parsed == 0 {
        return Err(format!("{flag} expects a value >= 1, got '{value}'"));
    }
    Ok(parsed)
}

fn parse_capture_target(value: &str) -> Result<CaptureTarget, String> {
    CaptureTarget::parse(value)
        .ok_or_else(|| format!("--capture_target expects present or draw, got '{value}'"))
}

fn validate_capture_options(options: &LaunchOptions) -> Result<(), String> {
    if options.capture_frame.is_some() && options.capture_frames.is_some() {
        return Err(
            "--capture_frame and --capture_frames cannot be used in the same launch".to_string(),
        );
    }
    if options.capture_frame_path.is_some() && options.capture_frame.is_none() {
        return Err("--capture_frame_path requires --capture_frame".to_string());
    }
    if options.capture_dir.is_some() && options.capture_frames.is_none() {
        return Err("--capture_dir requires --capture_frames".to_string());
    }
    if options.capture_frame_start.is_some() && options.capture_frames.is_none() {
        return Err("--capture_frame_start requires --capture_frames".to_string());
    }
    if options.capture_frame_interval.is_some() && options.capture_frames.is_none() {
        return Err("--capture_frame_interval requires --capture_frames".to_string());
    }
    Ok(())
}

#[allow(
    dead_code,
    reason = "demo runner is unused in non-demo example binaries"
)]
pub fn run_demo(scenario: DemoScenario) {
    init_logging();
    let launch_options = match parse_launch_options() {
        Ok(options) => options,
        Err(err) => {
            error!("Failed to parse launch arguments: {err}");
            return;
        }
    };

    let config = RendererConfig {
        app_name: scenario.title().to_string(),
        shader_debug_mode: scenario.debug_runtime_mode(),
        headless: launch_options.headless,
        ..RendererConfig::default()
    };

    let app_name = config.app_name.clone();
    let capture_run_dir = default_capture_run_dir(&app_name);
    if launch_options.headless {
        run_headless_demo(
            config,
            launch_options,
            scenario,
            &app_name,
            &capture_run_dir,
        );
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
    if let Err(err) = apply_frame_capture_launch_options_with_run_dir(
        &mut renderer,
        &launch_options,
        &app_name,
        &capture_run_dir,
    ) {
        error!("Failed to configure frame capture: {err}");
        return;
    }
    match apply_debug_record_launch_options(&mut renderer, &launch_options) {
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

    let model_path = launch_options
        .model_path
        .as_deref()
        .unwrap_or(Path::new(FACADE_DEMO_MODEL_PATH));
    let mut scene = match initialize_scene(&mut renderer, scenario, model_path) {
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

#[allow(
    dead_code,
    reason = "demo helper is unused in non-demo example binaries"
)]
fn run_headless_demo(
    config: RendererConfig,
    launch_options: LaunchOptions,
    scenario: DemoScenario,
    app_name: &str,
    capture_run_dir: &Path,
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

    if let Err(err) = apply_frame_capture_launch_options_with_run_dir(
        &mut renderer,
        &launch_options,
        app_name,
        capture_run_dir,
    ) {
        error!("Failed to configure frame capture: {err}");
        return;
    }
    match apply_debug_record_launch_options(&mut renderer, &launch_options) {
        Ok(Some(path)) => info!("Debug timing recording active -> {}", path),
        Ok(None) => {}
        Err(err) => {
            error!("Failed to configure debug timing recording: {err}");
            return;
        }
    }

    let model_path = launch_options
        .model_path
        .as_deref()
        .unwrap_or(Path::new(FACADE_DEMO_MODEL_PATH));
    let mut scene = match initialize_scene(&mut renderer, scenario, model_path) {
        Ok(scene) => scene,
        Err(err) => {
            error!("Failed to initialize headless demo scene: {err}");
            return;
        }
    };

    let expected_captures = expected_launch_captures(&launch_options);
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

#[allow(
    dead_code,
    reason = "demo helper is unused in non-demo example binaries"
)]
fn expected_launch_captures(options: &LaunchOptions) -> usize {
    options
        .capture_frames
        .or_else(|| options.capture_frame.map(|_| 1))
        .unwrap_or(0) as usize
}

#[allow(
    dead_code,
    reason = "demo helper is unused in non-demo example binaries"
)]
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

#[allow(
    dead_code,
    reason = "demo helper is unused in non-demo example binaries"
)]
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

#[allow(
    dead_code,
    reason = "demo helper is unused in non-demo example binaries"
)]
fn initialize_scene(
    renderer: &mut Renderer,
    scenario: DemoScenario,
    model_path: &Path,
) -> Result<Scene, RendererError> {
    match scenario {
        DemoScenario::Pbr => Ok(renderer.take_startup_scene().unwrap_or_default()),
        DemoScenario::Unlit => Ok(renderer.take_startup_scene().unwrap_or_default()),
        DemoScenario::ModelLoad => build_model_load_scene(renderer, model_path, true),
    }
}

#[allow(
    dead_code,
    reason = "demo helper is unused in non-demo example binaries"
)]
fn build_model_load_scene(
    renderer: &mut Renderer,
    model_path: &Path,
    duplicate_instance: bool,
) -> Result<Scene, RendererError> {
    // Force the demo down the public facade asset + scene path.
    let _ = renderer.take_startup_scene();

    let mut scene = Scene::new();
    let first_fragment = {
        let mut assets = renderer.assets();
        assets.load_model(model_path)?
    };
    let first_mount = scene.merge_fragment(None, first_fragment)?;

    if duplicate_instance {
        let second_fragment = {
            let mut assets = renderer.assets();
            assets.load_model(model_path)?
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

#[allow(
    dead_code,
    reason = "demo helper is unused in non-demo example binaries"
)]
fn init_logging() {
    let _ = env_logger::Builder::new()
        .target(env_logger::Target::Stdout)
        .parse_filters(&env::var("RUST_LOG").unwrap_or_else(|_| "info".to_string()))
        .try_init();

    info!("Starting facade example runtime");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_capture_single_flags() {
        let options = parse_launch_options_from([
            "--capture_frame",
            "30",
            "--capture_frame_path=.internal-dev/debug_reports/api_test-frame.png",
            "--capture_target=draw",
            "--headless",
            "--manual_capture_dir",
            ".internal-dev/debug_reports/manual",
        ])
        .expect("capture args should parse");

        assert_eq!(options.capture_frame, Some(30));
        assert_eq!(
            options.capture_frame_path,
            Some(PathBuf::from(
                ".internal-dev/debug_reports/api_test-frame.png"
            ))
        );
        assert_eq!(options.capture_target, CaptureTarget::Draw);
        assert!(options.headless);
        assert_eq!(
            options.manual_capture_dir,
            Some(PathBuf::from(".internal-dev/debug_reports/manual"))
        );
    }

    #[test]
    fn parse_capture_sequence_flags() {
        let options = parse_launch_options_from([
            "--capture_frames=5",
            "--capture_frame_start",
            "30",
            "--capture_frame_interval=10",
            "--capture_dir=.internal-dev/debug_reports/api_test-captures",
        ])
        .expect("sequence args should parse");

        assert_eq!(options.capture_frames, Some(5));
        assert_eq!(options.capture_frame_start, Some(30));
        assert_eq!(options.capture_frame_interval, Some(10));
        assert_eq!(
            options.capture_dir,
            Some(PathBuf::from(
                ".internal-dev/debug_reports/api_test-captures"
            ))
        );
        assert_eq!(options.capture_target, CaptureTarget::Present);
    }

    #[test]
    fn capture_parser_preserves_existing_debug_env_model_flags() {
        let options = parse_launch_options_from([
            "--env",
            "env.exr",
            "--model=model.glb",
            "--record_debug=10",
            "--record_debug_interval",
            "50",
            "--record_debug_path=timing.jsonl",
        ])
        .expect("existing args should parse");

        assert_eq!(options.env_path, Some(PathBuf::from("env.exr")));
        assert_eq!(options.model_path, Some(PathBuf::from("model.glb")));
        assert_eq!(options.record_debug_secs, Some(10));
        assert_eq!(options.record_debug_interval_ms, Some(50));
        assert_eq!(options.record_debug_path.as_deref(), Some("timing.jsonl"));
    }

    #[test]
    fn reject_invalid_capture_values() {
        assert!(parse_launch_options_from(["--capture_frames=0"])
            .unwrap_err()
            .contains("value >= 1"));
        assert!(
            parse_launch_options_from(["--capture_frames=2", "--capture_frame_interval=0"])
                .unwrap_err()
                .contains("value >= 1")
        );
        assert!(parse_launch_options_from(["--capture_target=swapchain"])
            .unwrap_err()
            .contains("present or draw"));
    }

    #[test]
    fn reject_ambiguous_capture_modes() {
        assert!(
            parse_launch_options_from(["--capture_frame=1", "--capture_frames=2"])
                .unwrap_err()
                .contains("cannot be used")
        );
        assert!(parse_launch_options_from(["--capture_frame_path=one.png"])
            .unwrap_err()
            .contains("requires --capture_frame"));
        assert!(parse_launch_options_from(["--capture_dir=captures"])
            .unwrap_err()
            .contains("requires --capture_frames"));
    }
}
