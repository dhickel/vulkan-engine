mod audio_bridge;
mod collision;
mod content;
mod events;
mod geometry;
mod layout;
mod player;
mod scene_seed;

use std::collections::HashSet;
use std::path::PathBuf;
use std::time::Duration;

use collision::CollisionWorld;
use content::{load_content_pack, resolve_content_path};
use engine::camera::{Camera, FPSController};
use engine::events::{runtime_event_bus, DispatchReport, EventBus};
use engine::frame::{FixedStepClock, FixedStepConfig, FrameClock};
use engine::input::{
    ActionMap, InputActionEventEmitter, InputSystem, LayerDescriptor, LayerPriority,
};
use layout::{load_level_file, tile_to_world, ParsedLevel};
use player::{CameraIntentGuard, PlayerState, PLAYER_EYE_HEIGHT};
use renderer::api::config::{CompressionConfig, TextureCompressionMode};
use renderer::prelude::{
    AssetManifestMode, AssetPolicyConfig, CaptureTarget, FrameCaptureSequence, FrameCaptureStatus,
};
use renderer::{FrameRenderOutcome, RendererConfig, RendererError};
use scene_seed::{renderer_visual_tuning, LevelScene};
use thiserror::Error;
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::keyboard::KeyCode;
use winit::window::WindowBuilder;

const APP_WINDOW_TITLE: &str = "Dungeon Dogfood - Phase 07";
const DEFAULT_LEVEL_ID: &str = LEVEL_01_PATH;
const LEVEL_SELECT_ENV: &str = "DUNGEON_DOGFOOD_LEVEL";
const LEVEL_01_PATH: &str = "apps/dungeon_dogfood/assets/levels/level_01.txt";
const LEVEL_02_PATH: &str = "apps/dungeon_dogfood/assets/levels/level_02_ramps.txt";
const LEVEL_03_PATH: &str = "apps/dungeon_dogfood/assets/levels/level_03_lighting.txt";
const CONTENT_PACK_PATH: &str = "apps/dungeon_dogfood/assets/content_pack.toml";

/// Fixed simulation timestep for physics and gameplay logic (60 Hz).
const FIXED_DT: f32 = 1.0 / 60.0;

#[derive(Debug, Default)]
struct HeadlessOptions {
    enabled: bool,
    capture_target: CaptureTarget,
    capture_frames: Option<u32>,
    capture_frame_start: Option<u32>,
    capture_frame_interval: Option<u32>,
    capture_dir: Option<PathBuf>,
}

impl HeadlessOptions {
    fn from_args() -> Self {
        let args: Vec<String> = std::env::args().collect();
        let mut opts = HeadlessOptions::default();
        let mut i = 1;

        while i < args.len() {
            match args[i].as_str() {
                "--headless" => {
                    opts.enabled = true;
                    i += 1;
                }
                "--capture_target" => {
                    if let Some(value) = args.get(i + 1) {
                        opts.capture_target = CaptureTarget::parse(value).unwrap_or_else(|| {
                            eprintln!(
                                "invalid --capture_target '{}'; expected 'present' or 'draw'",
                                value
                            );
                            std::process::exit(1);
                        });
                        i += 2;
                    } else {
                        eprintln!("--capture_target requires a value");
                        std::process::exit(1);
                    }
                }
                "--capture_frames" => {
                    if let Some(value) = args.get(i + 1) {
                        opts.capture_frames = Some(value.parse().unwrap_or_else(|_| {
                            eprintln!("--capture_frames expects a positive integer");
                            std::process::exit(1);
                        }));
                        i += 2;
                    } else {
                        eprintln!("--capture_frames requires a value");
                        std::process::exit(1);
                    }
                }
                "--capture_frame_start" => {
                    if let Some(value) = args.get(i + 1) {
                        opts.capture_frame_start = Some(value.parse().unwrap_or_else(|_| {
                            eprintln!("--capture_frame_start expects a non-negative integer");
                            std::process::exit(1);
                        }));
                        i += 2;
                    } else {
                        eprintln!("--capture_frame_start requires a value");
                        std::process::exit(1);
                    }
                }
                "--capture_frame_interval" => {
                    if let Some(value) = args.get(i + 1) {
                        opts.capture_frame_interval = Some(value.parse().unwrap_or_else(|_| {
                            eprintln!("--capture_frame_interval expects a positive integer");
                            std::process::exit(1);
                        }));
                        i += 2;
                    } else {
                        eprintln!("--capture_frame_interval requires a value");
                        std::process::exit(1);
                    }
                }
                "--capture_dir" => {
                    if let Some(value) = args.get(i + 1) {
                        opts.capture_dir = Some(PathBuf::from(value));
                        i += 2;
                    } else {
                        eprintln!("--capture_dir requires a value");
                        std::process::exit(1);
                    }
                }
                // Skip --level and its value (handled by parse_level_arg)
                "--level" => {
                    i += 2;
                }
                _ => {
                    i += 1;
                }
            }
        }

        opts
    }
}

fn main() {
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Info)
        .init();

    if let Err(err) = run() {
        eprintln!("{err}");
        if matches!(err, AppError::LevelLoad { .. }) {
            print_level_load_help();
        }
        std::process::exit(1);
    }
}

#[derive(Debug, Error)]
enum AppError {
    #[error("failed to load required content pack '{CONTENT_PACK_PATH}': {0}")]
    ContentPack(#[from] content::ContentError),
    #[error(
        "failed to load selected level '{label}' from '{path}': {source}",
        path = .selection.path.display(),
        label = .selection.label
    )]
    LevelLoad {
        selection: LevelSelection,
        source: layout::LayoutError,
    },
    #[error("failed to create event loop: {0}")]
    EventLoop(#[from] winit::error::EventLoopError),
    #[error("failed to create window: {0}")]
    Window(#[from] winit::error::OsError),
    #[error("failed to initialize renderer: {0}")]
    RendererInit(#[source] RendererError),
    #[error("failed to seed level scene before entering event loop: {0}")]
    SceneSeed(#[from] scene_seed::SceneSeedError),
}

fn run() -> Result<(), AppError> {
    let headless_opts = HeadlessOptions::from_args();
    let content_pack = load_content_pack(CONTENT_PACK_PATH)?;

    log::info!(
        "Loaded content pack: {} props ({} enabled), {} materials, {} environments, {} audio clips, {} light presets",
        content_pack.props.len(),
        content_pack.enabled_props().len(),
        content_pack.materials.len(),
        content_pack.environments.len(),
        content_pack.audio_clips.len(),
        content_pack.light_presets.len()
    );

    let level_selection = selected_level();
    let loaded_level = load_selected_level(&level_selection)?;
    let level = &loaded_level.level;

    log::info!(
        "Selected level '{}': {}",
        level_selection.label,
        loaded_level.source_description
    );
    log::info!(
        "Loaded level: {}x{} tiles across {} layers",
        level.width,
        level.height,
        level.layer_count()
    );
    log::info!(
        "Spawn position: layer={}, x={}, y={}",
        level.spawn.layer,
        level.spawn.x,
        level.spawn.y
    );
    log::info!("Light markers: {}", level.light_markers.len());
    log::info!("Model markers: {}", level.model_markers.len());

    let collision_world = CollisionWorld::from_level(level);

    if headless_opts.enabled {
        return run_headless(
            level,
            &content_pack,
            &collision_world,
            &headless_opts,
        );
    }

    let event_loop = EventLoop::new()?;
    let window = WindowBuilder::new()
        .with_title(APP_WINDOW_TITLE)
        .with_inner_size(winit::dpi::LogicalSize::new(1280, 720))
        .build(&event_loop)?;

    let config = RendererConfig {
        app_name: "dungeon_dogfood".to_string(),
        window_width: 1280,
        window_height: 720,
        validation_layer: env_flag("DUNGEON_DOGFOOD_VALIDATION"),
        compile_shaders: false,
        shader_debug_mode: renderer::DebugRuntimeMode::Default,
        preload_startup_scene: false,
        visual_tuning: renderer_visual_tuning(),
        headless: false,
        asset_policy: AssetPolicyConfig {
            manifest_mode: AssetManifestMode::BestEffort,
            allow_filename_heuristics: true,
            compression: CompressionConfig {
                mode: TextureCompressionMode::Disabled,
                quality: 50,
            },
        },
    };

    let mut renderer = renderer::Renderer::new(config, &window).map_err(AppError::RendererInit)?;
    let mut app_events = runtime_event_bus();
    events::install_dogfood_event_logger(&mut app_events);
    let audio_report = audio_bridge::run_startup_audio_probe(
        &mut app_events,
        &content_pack,
        audio_bridge::audio_smoke_requested(),
    );
    log::info!(
        "Dogfood audio bridge report: clip={:?} status={:?}",
        audio_report.clip_id,
        audio_report.device_smoke_status
    );

    let mut scene = renderer::Scene::new();
    {
        let assets = renderer.assets();
        scene.set_skybox(assets.default_environment());
    }

    let _level_scene = {
        let mut assets = renderer.assets();
        LevelScene::from_level(&level, &content_pack, &mut scene, &mut assets)?
    };

    let spawn_world = tile_to_world(level.spawn.x, level.spawn.y);
    let spawn_position = spawn_world
        + glam::Vec3::new(
            0.5,
            level.spawn.layer as f32 * collision::WALL_HEIGHT + PLAYER_EYE_HEIGHT,
            -0.5,
        );
    let mut player = PlayerState::new(spawn_position);
    let mut previous_player_position = player.position;
    let mut app_camera = Camera::new(spawn_position);
    let mut app_input = InputSystem::new();
    let mut fps_controller = install_app_fps_input(&mut app_input);
    let mut action_events = InputActionEventEmitter::new();
    let mut frame_clock = FrameClock::new();
    let mut fixed_clock = FixedStepClock::new(FixedStepConfig {
        step: Duration::from_secs_f32(FIXED_DT),
        max_steps_per_frame: 10,
    });

    log::info!("Dungeon dogfood initialized, starting event loop");

    let mut last_window_size = window.inner_size();
    let mut resize_pending = false;
    window.request_redraw();

    event_loop
        .run(move |event, elwt| {
            elwt.set_control_flow(ControlFlow::Poll);

            let _routing = match engine::input::route_platform_input_to_app(
                &mut renderer,
                &window,
                &mut app_input,
                &event,
            ) {
                Ok(routing) => routing,
                Err(e) => {
                    log::error!("Platform input routing failed: {}", e);
                    elwt.exit();
                    return;
                }
            };

            match event {
                Event::WindowEvent { event, window_id } if window_id == window.id() => {
                    match event {
                        WindowEvent::CloseRequested => {
                            log::info!("Close requested, exiting");
                            elwt.exit();
                        }
                        WindowEvent::Resized(new_size) => {
                            last_window_size = new_size;
                            if let Err(e) = renderer.resize(new_size.width, new_size.height) {
                                log::error!("Resize failed: {}", e);
                                elwt.exit();
                            }
                        }
                        WindowEvent::ScaleFactorChanged {
                            mut inner_size_writer,
                            ..
                        } => {
                            let new_size = window.inner_size();
                            if let Err(e) = inner_size_writer.request_inner_size(new_size) {
                                log::error!("Scale factor size request failed: {}", e);
                                elwt.exit();
                                return;
                            }
                            last_window_size = new_size;
                            if let Err(e) = renderer.resize(new_size.width, new_size.height) {
                                log::error!("Resize failed after scale change: {}", e);
                                elwt.exit();
                                return;
                            }
                        }
                        WindowEvent::RedrawRequested => {
                            let current_size = window.inner_size();
                            if current_size != last_window_size {
                                last_window_size = current_size;
                                if let Err(e) =
                                    renderer.resize(current_size.width, current_size.height)
                                {
                                    log::error!("Resize failed while redrawing: {}", e);
                                    elwt.exit();
                                    return;
                                }
                            }

                            match render_frame(
                                &mut renderer,
                                &mut scene,
                                &collision_world,
                                &mut player,
                                &mut previous_player_position,
                                &mut app_camera,
                                &mut fps_controller,
                                &mut app_input,
                                &mut action_events,
                                &mut app_events,
                                &mut frame_clock,
                                &mut fixed_clock,
                                current_size.width,
                                current_size.height,
                                false,
                            ) {
                                Ok(FrameRenderOutcome::Rendered) => {
                                    if resize_pending {
                                        resize_pending = false;
                                        window.set_title(APP_WINDOW_TITLE);
                                    }
                                }
                                Ok(FrameRenderOutcome::SkippedResizePending) => {
                                    if !resize_pending {
                                        resize_pending = true;
                                        window.set_title("Dungeon Dogfood - Phase 06 (resizing...)");
                                        log::info!(
                                            "Render skipped while swapchain resize is pending; waiting for a stable window size."
                                        );
                                    }
                                }
                                Ok(FrameRenderOutcome::SubmittedNotPresented)
                                | Ok(FrameRenderOutcome::PresentedSuboptimal) => {
                                    // Continue rendering; presentation suboptimality is not a fatal state.
                                }
                                Err(e) => {
                                    log::error!("Render failed: {}", e);
                                    elwt.exit();
                                }
                            }

                            window.request_redraw();
                        }
                        _ => {}
                    }
                }
                _ => {}
            }
        })
        .map_err(AppError::EventLoop)?;

    Ok(())
}

fn install_app_fps_input(input: &mut InputSystem) -> FPSController {
    let mut map = ActionMap::new();
    map.bind_key("move.forward", KeyCode::KeyW);
    map.bind_key("move.backward", KeyCode::KeyS);
    map.bind_key("move.left", KeyCode::KeyA);
    map.bind_key("move.right", KeyCode::KeyD);
    map.bind_key("move.up", KeyCode::Space);
    map.bind_key("move.down", KeyCode::ShiftLeft);

    input.add_layer(
        LayerDescriptor::new("dogfood-fps-actions", LayerPriority(10)),
        map.into_layer(),
    );

    FPSController::new(0.002, 1.0)
}

fn log_dispatch_failures(report: DispatchReport, context: &str) {
    for failure in report.failures {
        log::warn!(
            "{context} listener {:?} failed for event {:?}: {}",
            failure.listener,
            failure.sequence,
            failure.message
        );
    }
}

fn render_frame(
    renderer: &mut renderer::Renderer,
    scene: &mut renderer::Scene,
    collision_world: &CollisionWorld,
    player: &mut PlayerState,
    previous_player_position: &mut glam::Vec3,
    camera: &mut Camera,
    fps_controller: &mut FPSController,
    input: &mut InputSystem,
    action_events: &mut InputActionEventEmitter,
    events: &mut EventBus,
    frame_clock: &mut FrameClock,
    fixed_clock: &mut FixedStepClock,
    viewport_width: u32,
    viewport_height: u32,
    headless: bool,
) -> Result<renderer::FrameRenderOutcome, RendererError> {
    let begin_report = engine::frame::begin_app_frame(input, action_events, events, frame_clock);
    log_dispatch_failures(begin_report.input_dispatch, "dogfood input");
    log_dispatch_failures(begin_report.frame_started, "dogfood lifecycle");

    // Accumulate real time, sample display-frame input once, then advance the
    // authoritative player state in fixed simulation steps. Applying the FPS
    // controller once avoids multiplying this frame's mouse delta when a
    // catch-up frame runs more than one simulation step.
    let fixed_update = fixed_clock.update(begin_report.frame.delta);
    let simulated_seconds = FIXED_DT * fixed_update.steps as f32;
    fps_controller.update_from_snapshot(input.snapshot(), simulated_seconds, camera);

    if fixed_update.steps > 0 {
        match player.ingest_camera_intent(camera.get_position(), simulated_seconds) {
            CameraIntentGuard::Accepted => {}
            CameraIntentGuard::Clamped {
                attempted_displacement,
                applied_displacement,
            } => {
                log::warn!(
                    "Clamped player movement from {:.3}m to {:.3}m for this simulation update.",
                    attempted_displacement,
                    applied_displacement
                );
            }
            CameraIntentGuard::RejectedNonFinite => {
                log::error!(
                    "Rejected non-finite camera intent before collision resolution; keeping previous player position."
                );
            }
        }

        for _ in 0..fixed_update.steps {
            *previous_player_position = player.position;
            collision::resolve_player_step(player, collision_world, FIXED_DT);
            if !player.has_finite_position() {
                log::error!(
                    "Player position became non-finite after collision resolution: {:?}",
                    player.position
                );
                return Err(RendererError::InvalidState(
                    "player position must remain finite before camera view construction".to_string(),
                ));
            }
        }
    }
    camera.set_position(player.position);

    if !fixed_update.dropped_time.is_zero() {
        log::warn!(
            "Dropped {:.3}ms of accumulated simulation time after reaching the fixed-step catch-up limit.",
            fixed_update.dropped_time.as_secs_f64() * 1_000.0
        );
    }

    // Render one simulation step behind and interpolate toward the current
    // authoritative state using the accumulator remainder. The previous state
    // persists across display frames, including frames that run zero steps.
    let authoritative_position = player.position;
    camera.set_position(interpolated_player_position(
        *previous_player_position,
        authoritative_position,
        fixed_update.alpha,
    ));

    renderer.pump_asset_tasks(32)?;
    let view = engine::render::camera_view_for_size(camera, viewport_width, viewport_height);

    // Restore camera to authoritative simulation position.
    camera.set_position(authoritative_position);

    let outcome = if headless {
        renderer.render_scene_headless_with_view(scene, view)?
    } else {
        renderer.render_scene_with_view(scene, view)?
    };
    let end_report = engine::frame::end_app_frame(events, begin_report.frame.index);
    log_dispatch_failures(end_report.frame_ended, "dogfood lifecycle");

    Ok(outcome)
}

fn interpolated_player_position(
    previous: glam::Vec3,
    current: glam::Vec3,
    alpha: f32,
) -> glam::Vec3 {
    previous.lerp(current, alpha.clamp(0.0, 1.0))
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct LevelSelection {
    label: String,
    path: PathBuf,
}

struct LoadedLevel {
    level: ParsedLevel,
    source_description: String,
}

fn selected_level() -> LevelSelection {
    if let Some(selector) = parse_level_arg() {
        return resolve_level_selector(selector);
    }

    if let Ok(selector) = std::env::var(LEVEL_SELECT_ENV) {
        if !selector.trim().is_empty() {
            return resolve_level_selector(selector);
        }
    }

    resolve_level_selector(DEFAULT_LEVEL_ID)
}

fn resolve_level_selector(selector: impl AsRef<str>) -> LevelSelection {
    let selector = selector.as_ref().trim();

    let (label, path) = match selector {
        "level_01" | "level_01.txt" | LEVEL_01_PATH => ("level_01", LEVEL_01_PATH),
        "level_02_ramps" | "level_02_ramps.txt" | LEVEL_02_PATH => {
            ("level_02_ramps", LEVEL_02_PATH)
        }
        "level_03_lighting" | "level_03_lighting.txt" | LEVEL_03_PATH => {
            ("level_03_lighting", LEVEL_03_PATH)
        }
        _ => (selector, selector),
    };

    LevelSelection {
        label: label.to_string(),
        path: PathBuf::from(path),
    }
}

fn load_selected_level(selection: &LevelSelection) -> Result<LoadedLevel, AppError> {
    let resolved_level_path = resolve_content_path(&selection.path);
    let level = load_level_file(&resolved_level_path).map_err(|source| AppError::LevelLoad {
        selection: selection.clone(),
        source,
    })?;

    Ok(LoadedLevel {
        level,
        source_description: resolved_level_path.display().to_string(),
    })
}

fn parse_level_arg() -> Option<String> {
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;

    while i < args.len() {
        if args[i] == "--level" {
            if let Some(path) = args.get(i + 1) {
                return Some(path.clone());
            }
            eprintln!("--level requires a path argument");
            std::process::exit(1);
        }

        i += 1;
    }

    None
}

fn print_level_load_help() {
    eprintln!();
    eprintln!("Built-in level selectors:");
    eprintln!("  {}", DEFAULT_LEVEL_ID);
    eprintln!("  level_01");
    eprintln!("  level_02_ramps");
    eprintln!("  level_03_lighting");
    eprintln!();
    eprintln!(
        "Use --level <selector-or-path> or {}=<selector-or-path>",
        LEVEL_SELECT_ENV
    );
    eprintln!();
    eprintln!("Expected ASCII level file with tokens:");
    eprintln!("  # = wall");
    eprintln!("  . = floor");
    eprintln!("  _ = open shaft / void");
    eprintln!("  S = spawn marker (exactly 1 required)");
    eprintln!("  M = model marker");
    eprintln!("  L = point light marker");
    eprintln!("  R^ R> Rv R< = ramp tiles");
    eprintln!("  --- = next layer separator");
    eprintln!();
    eprintln!("Headless capture options:");
    eprintln!("  --headless                     Use headless renderer (no window)");
    eprintln!("  --capture_target <present|draw>");
    eprintln!("  --capture_frames <n>           Number of frames to capture");
    eprintln!("  --capture_frame_start <n>      Frame to start capturing (default: 0)");
    eprintln!("  --capture_frame_interval <n>   Frames between captures (default: 1)");
    eprintln!("  --capture_dir <dir>            Output directory for captures");
}

fn run_headless(
    level: &ParsedLevel,
    content_pack: &content::ContentPack,
    collision_world: &CollisionWorld,
    headless_opts: &HeadlessOptions,
) -> Result<(), AppError> {
    log::info!("Starting dogfood headless capture run");

    let config = RendererConfig {
        app_name: "dungeon_dogfood".to_string(),
        window_width: 1280,
        window_height: 720,
        validation_layer: env_flag("DUNGEON_DOGFOOD_VALIDATION"),
        compile_shaders: false,
        shader_debug_mode: renderer::DebugRuntimeMode::Default,
        preload_startup_scene: false,
        visual_tuning: renderer_visual_tuning(),
        headless: true,
        asset_policy: AssetPolicyConfig {
            manifest_mode: AssetManifestMode::BestEffort,
            allow_filename_heuristics: true,
            compression: CompressionConfig {
                mode: TextureCompressionMode::Disabled,
                quality: 50,
            },
        },
    };

    let mut renderer = renderer::Renderer::new_headless(config).map_err(AppError::RendererInit)?;
    let mut app_events = runtime_event_bus();
    events::install_dogfood_event_logger(&mut app_events);
    let audio_report = audio_bridge::run_startup_audio_probe(
        &mut app_events,
        content_pack,
        audio_bridge::audio_smoke_requested(),
    );
    log::info!(
        "Dogfood headless audio bridge report: clip={:?} status={:?}",
        audio_report.clip_id,
        audio_report.device_smoke_status
    );

    let mut scene = renderer::Scene::new();
    {
        let assets = renderer.assets();
        scene.set_skybox(assets.default_environment());
    }

    let _level_scene = {
        let mut assets = renderer.assets();
        LevelScene::from_level(level, content_pack, &mut scene, &mut assets)?
    };

    let spawn_world = tile_to_world(level.spawn.x, level.spawn.y);
    let spawn_position = spawn_world
        + glam::Vec3::new(
            0.5,
            level.spawn.layer as f32 * collision::WALL_HEIGHT + PLAYER_EYE_HEIGHT,
            -0.5,
        );
    let mut player = PlayerState::new(spawn_position);
    let mut previous_player_position = player.position;
    let mut app_camera = Camera::new(spawn_position);
    let mut app_input = InputSystem::new();
    let mut fps_controller = install_app_fps_input(&mut app_input);
    let mut action_events = InputActionEventEmitter::new();
    let mut frame_clock = FrameClock::new();
    let mut fixed_clock = FixedStepClock::new(FixedStepConfig {
        step: Duration::from_secs_f32(FIXED_DT),
        max_steps_per_frame: 10,
    });

    // Configure frame capture if requested
    let capture_target = headless_opts.capture_target;
    let capture_run_dir = headless_opts.capture_dir.clone().unwrap_or_else(|| {
        PathBuf::from(".internal-dev/captures/sprint-11-dogfood-vertical-slice/dogfood-baseline")
    });

    if let Some(count) = headless_opts.capture_frames {
        let sequence = FrameCaptureSequence::new(
            capture_target,
            &capture_run_dir,
            headless_opts.capture_frame_start.unwrap_or(0),
            headless_opts.capture_frame_interval.unwrap_or(1),
            count,
        )
        .map_err(|e| AppError::RendererInit(RendererError::CaptureConfig(e)))?;

        let _ = std::fs::create_dir_all(&capture_run_dir);
        renderer
            .configure_frame_capture_sequence(sequence)
            .map_err(AppError::RendererInit)?;

        log::info!(
            "Headless frame capture configured: target={} frames={} start={} interval={} dir={}",
            capture_target.as_label(),
            count,
            headless_opts.capture_frame_start.unwrap_or(0),
            headless_opts.capture_frame_interval.unwrap_or(1),
            capture_run_dir.display()
        );
    } else {
        log::info!(
            "Headless render mode without capture; rendering {} default smoke frames",
            3
        );
    }

    let expected_captures = headless_opts.capture_frames.unwrap_or(0) as usize;
    let frame_budget = if expected_captures == 0 {
        3
    } else {
        let last_frame = headless_opts.capture_frame_start.unwrap_or(0)
            + headless_opts
                .capture_frame_interval
                .unwrap_or(1)
                .saturating_mul(headless_opts.capture_frames.unwrap_or(1).saturating_sub(1));
        last_frame.saturating_add(120).max(180)
    };
    let mut succeeded_paths = HashSet::new();

    for frame_num in 0..frame_budget {
        match render_frame(
            &mut renderer,
            &mut scene,
            collision_world,
            &mut player,
            &mut previous_player_position,
            &mut app_camera,
            &mut fps_controller,
            &mut app_input,
            &mut action_events,
            &mut app_events,
            &mut frame_clock,
            &mut fixed_clock,
            1280,
            720,
            true,
        ) {
            Ok(FrameRenderOutcome::Rendered)
            | Ok(FrameRenderOutcome::SkippedResizePending)
            | Ok(FrameRenderOutcome::SubmittedNotPresented)
            | Ok(FrameRenderOutcome::PresentedSuboptimal) => {}
            Err(err) => {
                log::error!("Headless render failed at frame {frame_num}: {err}");
                return Err(AppError::RendererInit(err));
            }
        }

        match renderer.last_frame_capture_status() {
            Some(FrameCaptureStatus::Succeeded {
                output_path,
                sidecar_path,
                target,
                width,
                height,
                ..
            }) => {
                if !record_unique_capture_success(&mut succeeded_paths, output_path) {
                    continue;
                }
                let succeeded_count = succeeded_paths.len();
                log::info!(
                    "Capture #{succeeded_count} at frame {frame_num}: target={} size={}x{} path={} sidecar={:?}",
                    target.as_label(),
                    width,
                    height,
                    output_path.display(),
                    sidecar_path.as_ref().map(|p| p.display().to_string())
                );
                if succeeded_count >= expected_captures && expected_captures > 0 {
                    log::info!(
                        "Headless capture complete: {succeeded_count}/{expected_captures} captures written"
                    );
                    return Ok(());
                }
            }
            Some(FrameCaptureStatus::Failed {
                message,
                frame_number,
                ..
            }) => {
                log::error!("Headless capture failed at frame {frame_number}: {message}");
                return Err(AppError::RendererInit(RendererError::InvalidState(
                    "headless capture failed".to_string(),
                )));
            }
            Some(FrameCaptureStatus::BackendNotImplemented {
                target,
                frame_number,
                ..
            }) => {
                log::error!(
                    "Headless capture target '{}' is not implemented at frame {frame_number}",
                    target.as_label()
                );
                return Err(AppError::RendererInit(RendererError::InvalidState(
                    "headless capture target not implemented".to_string(),
                )));
            }
            _ => {}
        }
    }

    if expected_captures == 0 {
        log::info!("Headless smoke render completed ({frame_budget} frames)");
        Ok(())
    } else {
        Err(AppError::RendererInit(RendererError::InvalidState(
            "headless capture incomplete".to_string(),
        )))
    }
}

fn record_unique_capture_success(
    succeeded_paths: &mut HashSet<PathBuf>,
    output_path: &std::path::Path,
) -> bool {
    succeeded_paths.insert(output_path.to_path_buf())
}

fn env_flag(var_name: &str) -> bool {
    std::env::var(var_name)
        .ok()
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capture_success_counter_ignores_repeated_output_path() {
        let mut succeeded_paths = HashSet::new();
        let output_path =
            PathBuf::from(".internal-dev/captures/example/dungeon-dogfood-frame-60.png");

        assert!(record_unique_capture_success(
            &mut succeeded_paths,
            &output_path
        ));
        assert!(!record_unique_capture_success(
            &mut succeeded_paths,
            &output_path
        ));
        assert_eq!(succeeded_paths.len(), 1);
    }

    #[test]
    fn interpolates_between_previous_and_current_simulation_positions() {
        let previous = glam::Vec3::new(1.0, 2.0, 3.0);
        let current = glam::Vec3::new(5.0, 6.0, 7.0);

        assert_eq!(
            interpolated_player_position(previous, current, 0.25),
            glam::Vec3::new(2.0, 3.0, 4.0)
        );
        assert_eq!(
            interpolated_player_position(previous, current, 0.0),
            previous
        );
        assert_eq!(
            interpolated_player_position(previous, current, 1.0),
            current
        );
    }

    #[test]
    fn resolve_builtin_level_id() {
        let selection = resolve_level_selector("level_02_ramps");
        assert_eq!(selection.label, "level_02_ramps");
        assert_eq!(selection.path, PathBuf::from(LEVEL_02_PATH));
    }

    #[test]
    fn resolve_builtin_level_filename() {
        let selection = resolve_level_selector("level_03_lighting.txt");
        assert_eq!(selection.label, "level_03_lighting");
        assert_eq!(selection.path, PathBuf::from(LEVEL_03_PATH));
    }

    #[test]
    fn resolve_level_01_path_as_default() {
        let selection = resolve_level_selector(DEFAULT_LEVEL_ID);
        assert_eq!(selection.label, "level_01");
    }

    #[test]
    fn preserve_custom_level_paths() {
        let selection = resolve_level_selector("custom/level.txt");
        assert_eq!(selection.label, "custom/level.txt");
        assert_eq!(selection.path, PathBuf::from("custom/level.txt"));
    }
}
