mod audio_bridge;
mod behavior_bridge;
mod collision;
mod component_compatibility;
mod components;
mod content;
mod events;
mod generator;
mod geometry;
#[cfg(test)]
mod geometry_fixtures;
mod layout;
mod mesh_collider_bridge;
mod physics_bridge;
mod player;
mod scene_seed;

use std::collections::HashSet;
use std::path::PathBuf;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use content::{load_content_pack, resolve_content_path};
use engine::camera::{Camera, FPSController};
use engine::events::{runtime_event_bus, DispatchReport, EventBus};
use engine::input::{
    ActionId, ActionMap, Axis2D, AxisContributor, CompoundAxis, InputActionEventEmitter,
    InputSystem, LayerDescriptor, LayerPriority,
};
use engine::time::{Time, TimeConfig};
use generator::{generate, GeneratorConfig, GeneratorError};
use layout::{load_level_file, tile_to_world, ParsedLevel};
use mesh_collider_bridge::MeshColliderBridge;
use physics::BodyKind;
use physics_bridge::PhysicsBridge;
use glam::{Quat, Vec2, Vec3};
use player::{CameraIntentGuard, PlayerState, PLAYER_CAPSULE_HALF_HEIGHT, PLAYER_CAPSULE_RADIUS, PLAYER_EYE_HEIGHT};
use renderer::api::config::{CompressionConfig, TextureCompressionMode};
use renderer::api::FrameSerial;
use renderer::prelude::{
    AssetManifestMode, AssetPolicyConfig, CaptureTarget, FrameCaptureSequence, FrameCaptureSource,
    FrameCaptureStatus,
};
use renderer::{FrameRenderOutcome, RendererConfig, RendererError};
use scene_seed::{renderer_visual_tuning, LevelScene};
use thiserror::Error;
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::keyboard::KeyCode;
use winit::window::WindowBuilder;

const APP_WINDOW_TITLE: &str = "Dungeon Dogfood - Phase 07";
const GENERATED_SELECTOR: &str = "generated_sprawl";
const DEFAULT_LEVEL_ID: &str = GENERATED_SELECTOR;
const LEVEL_SELECT_ENV: &str = "DUNGEON_DOGFOOD_LEVEL";
const GENERATOR_SEED_ENV: &str = "DUNGEON_DOGFOOD_GENERATOR_SEED";
const GENERATOR_WIDTH_ENV: &str = "DUNGEON_DOGFOOD_GENERATOR_WIDTH";
const GENERATOR_HEIGHT_ENV: &str = "DUNGEON_DOGFOOD_GENERATOR_HEIGHT";
const GENERATOR_LAYERS_ENV: &str = "DUNGEON_DOGFOOD_GENERATOR_LAYERS";
const LEVEL_01_PATH: &str = "apps/dungeon_dogfood/assets/levels/level_01.txt";
const LEVEL_02_PATH: &str = "apps/dungeon_dogfood/assets/levels/level_02_ramps.txt";
const LEVEL_03_PATH: &str = "apps/dungeon_dogfood/assets/levels/level_03_lighting.txt";
const CONTENT_PACK_PATH: &str = "apps/dungeon_dogfood/assets/content_pack.toml";
const NOCLIP_TOGGLE_ACTION: &str = "noclip.toggle";
const CAPTURE_SCREENSHOT_ACTION: &str = "capture.screenshot";

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
    validate_colliders: bool,
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
                // Skip --seed and its value.
                "--seed" => {
                    i += 2;
                }
                "--validate-colliders" => {
                    opts.validate_colliders = true;
                    i += 1;
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
        } else if matches!(err, AppError::GeneratedLevel { .. }) {
            print_generated_level_help();
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
    #[error("failed to create capture directory '{}': {source}", path.display())]
    CaptureDirectory {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to seed level scene before entering event loop: {0}")]
    SceneSeed(#[from] scene_seed::SceneSeedError),
    #[error("failed to generate level (seed={seed}): {source}")]
    GeneratedLevel {
        seed: u64,
        #[source]
        source: GeneratorError,
    },
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

    let level_selection = selected_level()?;
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

    if headless_opts.enabled {
        return run_headless(level, &content_pack, &headless_opts);
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
        startup_model_path: None,
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
    let manual_capture_dir = manual_capture_run_dir();
    std::fs::create_dir_all(&manual_capture_dir).map_err(|source| AppError::CaptureDirectory {
        path: manual_capture_dir.clone(),
        source,
    })?;
    renderer
        .configure_manual_frame_capture_dir(Some(manual_capture_dir.clone()))
        .map_err(AppError::RendererInit)?;
    log::info!(
        "Manual draw captures will be saved under {}",
        manual_capture_dir.display()
    );
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

    // Seed collider recipes from explicit policy assignments.
    let mut bridge = seed_collider_bridge(&mut renderer, &_level_scene, level);

    // Create component-driven physics bridge sharing the mesh bridge's world.
    let mut physics_bridge = PhysicsBridge::new();
    // Register mesh-collider body-node mappings for unified writeback.
    bridge.export_body_node_mappings_to_physics_bridge(&mut physics_bridge);
    log::info!(
        "Physics bridge ready: {} registered nodes",
        physics_bridge.registered_count()
    );

    let spawn_world = tile_to_world(level.spawn.x, level.spawn.y);
    let spawn_position = spawn_world
        + glam::Vec3::new(
            0.5,
            level.spawn.layer as f32 * collision::WALL_HEIGHT + PLAYER_EYE_HEIGHT,
            -0.5,
        );

    // Create player character in the shared physics world.
    physics_bridge
        .create_player_character(
            &mut bridge.world,
            spawn_position.to_array(),
            PLAYER_CAPSULE_RADIUS,
            PLAYER_CAPSULE_HALF_HEIGHT,
        )
        .map_err(|e| AppError::RendererInit(RendererError::InvalidState(e.to_string())))?;
    let mut player = PlayerState::new(spawn_position);
    let mut previous_player_position = player.position;
    let mut app_camera = Camera::new(spawn_position);
    let mut app_input = InputSystem::new();
    let fps_controller = install_app_fps_input(&mut app_input);
    let fps_sensitivity = fps_controller.sensitivity();
    let fps_move_speed = fps_controller.move_speed();

    // Compound digital movement axes: stable intent computed once per app frame.
    let movement_axis = movement_axis();
    let mut camera_yaw: f32 = 0.0;

    let mut action_events = InputActionEventEmitter::new();
    let mut time = Time::new(TimeConfig {
        step: Duration::from_secs_f32(FIXED_DT),
        max_steps_per_frame: 10,
        time_scale: 1.0,
    })
    .expect("valid default TimeConfig");

    log::info!("Dungeon dogfood initialized, starting event loop");

    let mut last_window_size = window.inner_size();
    let mut resize_pending = false;
    let mut reported_manual_captures = HashSet::new();
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
                                &mut bridge,
                                &mut physics_bridge,
                                &mut player,
                                &mut previous_player_position,
                                &mut app_camera,
                                fps_sensitivity,
                                fps_move_speed,
                                &mut camera_yaw,
                                &movement_axis,
                                &mut app_input,
                                &mut action_events,
                                &mut app_events,
                                &mut time,
                                &mut reported_manual_captures,
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
                                Ok(FrameRenderOutcome::SkippedAcquireUnavailable)
                                | Ok(FrameRenderOutcome::SubmittedNotPresented)
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
    map.bind_key(NOCLIP_TOGGLE_ACTION, KeyCode::KeyF);
    map.bind_key(CAPTURE_SCREENSHOT_ACTION, KeyCode::KeyC);

    input.add_layer(
        LayerDescriptor::new("dogfood-fps-actions", LayerPriority(10)),
        map.into_layer(),
    );

    FPSController::new(0.002, 1.0)
}

fn movement_axis() -> Axis2D {
    let contributor = |action, weight| {
        AxisContributor::new(ActionId::new(action), weight)
            .expect("dogfood movement axis uses finite contributor weights")
    };
    let horizontal = CompoundAxis::new(vec![
        contributor("move.right", 1.0),
        contributor("move.left", -1.0),
    ])
    .expect("dogfood movement axis uses the valid default range");
    let forward = CompoundAxis::new(vec![
        contributor("move.backward", 1.0),
        contributor("move.forward", -1.0),
    ])
    .expect("dogfood movement axis uses the valid default range");
    Axis2D::new(horizontal, forward, 0.1).expect("dogfood movement axis uses a valid dead zone")
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
    bridge: &mut MeshColliderBridge,
    physics_bridge: &mut PhysicsBridge,
    player: &mut PlayerState,
    previous_player_position: &mut glam::Vec3,
    camera: &mut Camera,
    fps_sensitivity: f64,
    move_speed: f32,
    camera_yaw: &mut f32,
    movement_axis: &Axis2D,
    input: &mut InputSystem,
    action_events: &mut InputActionEventEmitter,
    events: &mut EventBus,
    time: &mut Time,
    reported_manual_captures: &mut HashSet<PathBuf>,
    viewport_width: u32,
    viewport_height: u32,
    headless: bool,
) -> Result<renderer::FrameRenderOutcome, RendererError> {
    let begin_report =
        engine::frame::begin_app_frame_with_time(input, action_events, events, time);
    log_dispatch_failures(begin_report.input_dispatch, "dogfood input");
    log_dispatch_failures(begin_report.frame_started, "dogfood lifecycle");

    let time_update = *time.update();

    let noclip_toggle = input
        .snapshot()
        .action_just_pressed(&ActionId::from(NOCLIP_TOGGLE_ACTION));
    let capture_screenshot = input
        .snapshot()
        .action_just_pressed(&ActionId::from(CAPTURE_SCREENSHOT_ACTION));

    if noclip_toggle {
        player.noclip = !player.noclip;
        log::info!(
            "Noclip {}",
            if player.noclip { "enabled" } else { "disabled" }
        );
    }

    if capture_screenshot {
        renderer.queue_manual_frame_capture(CaptureTarget::Draw)?;
        log::info!("Manual draw capture triggered");
    }

    // Build stable movement intent from compound digital axes — computed once
    // per app frame so pointer delta and axis values do not replay per fixed
    // simulation step.
    let (move_x, move_z) = movement_axis.evaluate(input.snapshot());
    let mouse_delta = input.snapshot().mouse_delta();
    let up = input.snapshot().action_value(&ActionId::new("move.up"));
    let down = input
        .snapshot()
        .action_value(&ActionId::new("move.down"));

    // Apply camera rotation from captured pointer delta (once).
    let delta_yaw = -mouse_delta.0 as f32 * fps_sensitivity as f32;
    *camera_yaw += delta_yaw;
    camera.update_rotation(
        delta_yaw,
        -mouse_delta.1 as f32 * fps_sensitivity as f32,
    );

    // Build normalized camera-local movement direction (FPSController convention:
    // x = right, y = up, z = forward-in-neg-z).
    let raw_move = Vec3::new(move_x, up - down, move_z);
    let local_dir = if raw_move.length_squared() > 0.0 {
        raw_move.normalize()
    } else {
        Vec3::ZERO
    };

    // Rotate direction to world space (Y-only rotation preserves vertical).
    let yaw_quat = Quat::from_rotation_y(*camera_yaw);
    let world_dir = yaw_quat * local_dir;
    let world_horizontal_dir = {
        let v = Vec2::new(world_dir.x, world_dir.z);
        if v.length_squared() > 0.0 {
            v.normalize()
        } else {
            Vec2::ZERO
        }
    };
    let vertical_input = local_dir.y;

    // Keep camera position in sync with the authoritative player position
    // (position is already resolved from the previous frame's physics step).
    camera.set_position(player.position);

    if time_update.fixed_step_count > 0 {
        for _ in 0..time_update.fixed_step_count {
            *previous_player_position = player.position;

            // Compute per-step desired translation from input direction.
            let guard = player.ingest_movement_intent(
                world_horizontal_dir,
                vertical_input,
                FIXED_DT,
                move_speed,
            );
            match guard {
                CameraIntentGuard::Accepted => {}
                CameraIntentGuard::Clamped {
                    attempted_displacement,
                    applied_displacement,
                } => {
                    log::warn!(
                        "Clamped player movement from {:.3}m to {:.3}m for this step.",
                        attempted_displacement,
                        applied_displacement
                    );
                }
                CameraIntentGuard::RejectedNonFinite => {
                    log::error!(
                        "Rejected non-finite camera intent; keeping previous player position."
                    );
                }
            }

            if player.noclip {
                // Noclip bypasses physics: apply desired translation directly
                // and teleport the kinematic body so it stays out of the way.
                player.position += player.desired_translation;
                if let Some(body_id) = physics_bridge.character_body_id() {
                    let body_y = player.position.y - PLAYER_EYE_HEIGHT
                        + PLAYER_CAPSULE_HALF_HEIGHT
                        + PLAYER_CAPSULE_RADIUS;
                    let _ = bridge.world.set_body_position_by_id(
                        body_id,
                        [player.position.x, body_y, player.position.z],
                    );
                }
            } else {
                // Character-controller-driven movement through the shared physics world.
                let desired = player.desired_translation.to_array();
                match physics_bridge.move_character(&mut bridge.world, desired, FIXED_DT) {
                    Ok(_actual) => {
                        if let Some(eye_pos) =
                            physics_bridge.character_eye_position(&bridge.world)
                        {
                            player.position = glam::Vec3::from_array(eye_pos);
                        }
                    }
                    Err(e) => {
                        log::warn!("Character movement failed: {e}");
                    }
                }
            }

            if !player.has_finite_position() {
                log::error!(
                    "Player position became non-finite after character movement: {:?}",
                    player.position
                );
                return Err(RendererError::InvalidState(
                    "player position must remain finite before camera view construction"
                        .to_string(),
                ));
            }

            // Step the shared physics world (single collision authority).
            if let Err(err) = bridge.world.step(FIXED_DT) {
                log::warn!("Physics step failed: {}", err);
            }
        }
        // Record contacts for observability.
        let contacts = bridge.world.last_contact_records();
        if !contacts.is_empty() {
            log::debug!("Physics contacts this frame: {}", contacts.len());
        }
    }

    if !time_update.dropped_time.is_zero() {
        log::warn!(
            "Dropped {:.3}ms of accumulated simulation time after reaching the fixed-step catch-up limit.",
            time_update.dropped_time.as_secs_f64() * 1_000.0
        );
    }

    // Render one simulation step behind and interpolate toward the current
    // authoritative state using the accumulator remainder. The previous state
    // persists across display frames, including frames that run zero steps.
    let authoritative_position = player.position;
    camera.set_position(interpolated_player_position(
        *previous_player_position,
        authoritative_position,
        time_update.alpha,
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
    log_manual_capture_status(renderer, reported_manual_captures);
    let end_report = engine::frame::end_app_frame(events, begin_report.frame.index);
    log_dispatch_failures(end_report.frame_ended, "dogfood lifecycle");

    let serials = renderer.retirement_serials();
    if let Err(err) = bridge.reap_retired(FrameSerial::new(serials.latest_completed)) {
        log::warn!("Collider recipe reaping failed: {}", err);
    }

    // Write back dynamic/kinematic physics body poses to scene nodes.
    if let Err(err) = bridge.writeback_dynamic_transforms(scene) {
        log::warn!("Transform writeback failed: {}", err);
    }
    if let Err(err) = physics_bridge.sync_transforms(&bridge.world, scene) {
        log::warn!("Physics bridge sync transforms failed: {}", err);
    }

    Ok(outcome)
}

fn log_manual_capture_status(
    renderer: &renderer::Renderer,
    reported_paths: &mut HashSet<PathBuf>,
) {
    match renderer.last_frame_capture_status() {
        Some(FrameCaptureStatus::Succeeded {
            output_path,
            source: FrameCaptureSource::Manual,
            ..
        }) if reported_paths.insert(output_path.clone()) => {
            log::info!("Manual draw capture completed: {}", output_path.display());
        }
        Some(FrameCaptureStatus::Failed {
            output_path,
            message,
            source: FrameCaptureSource::Manual,
            ..
        }) if reported_paths.insert(output_path.clone()) => {
            log::error!(
                "Manual draw capture failed for {}: {}",
                output_path.display(),
                message
            );
        }
        Some(FrameCaptureStatus::BackendNotImplemented {
            output_path,
            target,
            source: FrameCaptureSource::Manual,
            ..
        }) if reported_paths.insert(output_path.clone()) => {
            log::error!(
                "Manual capture target '{}' is not implemented for {}",
                target.as_label(),
                output_path.display()
            );
        }
        _ => {}
    }
}

fn manual_capture_run_dir() -> PathBuf {
    let timestamp_millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    PathBuf::from("captures").join(format!(
        "dungeon-dogfood-{timestamp_millis}-pid{}",
        std::process::id()
    ))
}

fn interpolated_player_position(
    previous: glam::Vec3,
    current: glam::Vec3,
    alpha: f32,
) -> glam::Vec3 {
    previous.lerp(current, alpha.clamp(0.0, 1.0))
}

fn seed_collider_bridge(
    renderer: &mut renderer::Renderer,
    level_scene: &LevelScene,
    level: &ParsedLevel,
) -> MeshColliderBridge {
    let mut bridge = MeshColliderBridge::new();
    let assets = renderer.assets();

    for assignment in &level_scene.collider_policies {
        match assets.mesh_geometry(assignment.mesh) {
            Ok(dto) => {
                match bridge.register_policy(&dto, assignment.policy) {
                    Ok(Some(handle)) => log::info!(
                        "Collider recipe registered: mesh_slot={} policy={:?} recipe_slot={}",
                        assignment.mesh.slot,
                        assignment.policy,
                        handle.slot,
                    ),
                    Ok(None) => log::info!(
                        "Collider policy recorded without recipe: mesh_slot={} policy=None",
                        assignment.mesh.slot,
                    ),
                    Err(err) => log::warn!(
                        "Failed to register collider recipe for mesh slot {}: {}",
                        assignment.mesh.slot,
                        err,
                    ),
                }
            }
            Err(err) => {
                log::warn!(
                    "Skipping collider recipe for mesh slot {} (no DTO): {}",
                    assignment.mesh.slot,
                    err,
                );
            }
        }
    }

    // Instantiate static trimesh bodies for dungeon chunks.
    for (idx, &mesh) in level_scene.chunk_meshes.iter().enumerate() {
        if let Ok(recipe) = bridge.recipe_for_mesh(mesh) {
            let body_id_str = format!("body.chunk_{idx}");
            let collider_id_str = format!("collider.chunk_{idx}");
            let node_id = level_scene.chunk_nodes.get(idx).copied();
            if let Err(err) = bridge.instantiate_collider(
                recipe.handle,
                BodyKind::Static,
                &body_id_str,
                &collider_id_str,
                level_scene.chunk_transforms[idx],
                node_id,
            ) {
                log::warn!("Failed to instantiate chunk collider {idx}: {}", err);
            }
        }
    }

    // Dynamic convex-hull proof body.
    if let Some((proof_mesh, proof_node)) = level_scene.dynamic_proof_mesh {
        if let Ok(recipe) = bridge.recipe_for_mesh(proof_mesh) {
            let recipe_handle = recipe.handle;
            let proof_idx = bridge.next_body_index();
            let body_id_str = format!("body.dynamic_proof_{proof_idx}");
            let collider_id_str = format!("collider.dynamic_proof_{proof_idx}");
            match bridge.instantiate_collider(
                recipe_handle,
                BodyKind::Dynamic,
                &body_id_str,
                &collider_id_str,
                glam::Mat4::from_translation(glam::Vec3::new(
                    tile_to_world(level.spawn.x, level.spawn.y).x + 1.5,
                    2.5,
                    tile_to_world(level.spawn.x, level.spawn.y).z,
                )),
                Some(proof_node),
            ) {
                Ok((body_id, collider_id)) => {
                    log::info!(
                        "Dynamic proof body instantiated: body={body_id} collider={collider_id}"
                    );
                }
                Err(err) => log::warn!("Failed to instantiate dynamic proof collider: {}", err),
            }
        }
    }

    log::info!(
        "Mesh collider bridge ready: {} recipes",
        bridge.recipe_count(),
    );

    bridge
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct LevelSelection {
    label: String,
    path: PathBuf,
    source: LevelSource,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum LevelSource {
    Generated(u64),
    Authored(PathBuf),
}

struct LoadedLevel {
    level: ParsedLevel,
    source_description: String,
}

fn selected_level() -> Result<LevelSelection, AppError> {
    if let Some(selector) = parse_level_arg() {
        return resolve_level_selector(selector);
    }

    if let Ok(selector) = std::env::var(LEVEL_SELECT_ENV) {
        if !selector.trim().is_empty() {
            return resolve_level_selector(selector);
        }
    }

    // Default: generated with seed 0.
    resolve_level_selector(DEFAULT_LEVEL_ID)
}

fn resolve_level_selector(selector: impl AsRef<str>) -> Result<LevelSelection, AppError> {
    let selector = selector.as_ref().trim();

    // Exact generated_sprawl → generated with CLI seed or env seed or 0.
    if selector == GENERATED_SELECTOR {
        let seed = match parse_seed_arg() {
            Some(seed) => seed,
            None => read_generator_env_u64(GENERATOR_SEED_ENV, "invalid_seed_override")?
                .unwrap_or(0),
        };
        return Ok(LevelSelection {
            label: GENERATED_SELECTOR.to_string(),
            path: PathBuf::from(GENERATED_SELECTOR),
            source: LevelSource::Generated(seed),
        });
    }

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

    Ok(LevelSelection {
        label: label.to_string(),
        path: PathBuf::from(path),
        source: LevelSource::Authored(PathBuf::from(path)),
    })
}

fn load_selected_level(selection: &LevelSelection) -> Result<LoadedLevel, AppError> {
    match &selection.source {
        LevelSource::Generated(seed) => {
            let config = build_generator_config().map_err(|source| AppError::GeneratedLevel {
                seed: *seed,
                source,
            })?;
            let catalog = generator::prefab::PrefabCatalog::load(
                &std::path::PathBuf::from("apps/dungeon_dogfood/assets/prefabs"),
            )
            .map_err(|source| AppError::GeneratedLevel {
                seed: *seed,
                source,
            })?;

            let result = generate(config, &catalog, *seed).map_err(|source| {
                AppError::GeneratedLevel {
                    seed: *seed,
                    source,
                }
            })?;

            let source_description = format!(
                "generated sprawl (seed={} attempt={} config_hash={} lights={} models={})",
                result.seed,
                result.attempt_index,
                &crate::generator::determinism::lowercase_hex(&catalog.identity_bytes())[..16],
                result.level.light_markers.len(),
                result.level.model_markers.len(),
            );

            Ok(LoadedLevel {
                level: result.level,
                source_description,
            })
        }
        LevelSource::Authored(path) => {
            let resolved_level_path = resolve_content_path(path);
            let level =
                load_level_file(&resolved_level_path).map_err(|source| AppError::LevelLoad {
                    selection: selection.clone(),
                    source,
                })?;

            Ok(LoadedLevel {
                level,
                source_description: resolved_level_path.display().to_string(),
            })
        }
    }
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
        // Skip --seed and its value.
        if args[i] == "--seed" {
            i += 2;
            continue;
        }
        i += 1;
    }

    None
}

fn parse_seed_arg() -> Option<u64> {
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;

    while i < args.len() {
        if args[i] == "--seed" {
            if let Some(value) = args.get(i + 1) {
                return Some(value.parse().unwrap_or_else(|_| {
                    eprintln!("--seed expects a non-negative integer");
                    std::process::exit(1);
                }));
            }
            eprintln!("--seed requires a value");
            std::process::exit(1);
        }
        i += 1;
    }

    None
}

fn read_generator_env_u64(
    name: &str,
    reason: &'static str,
) -> Result<Option<u64>, AppError> {
    generator_env_u64(name, reason).map_err(|source| AppError::GeneratedLevel { seed: 0, source })
}

fn generator_env_u64(name: &str, reason: &'static str) -> Result<Option<u64>, GeneratorError> {
    let Some(value) = std::env::var_os(name) else {
        return Ok(None);
    };
    value
        .to_string_lossy()
        .parse::<u64>()
        .map(Some)
        .map_err(|_| GeneratorError::UnsupportedConfiguration {
            stage: generator::ErrorStage::Configuration,
            reason,
            value: 0,
        })
}

fn build_generator_config() -> Result<GeneratorConfig, GeneratorError> {
    let mut config = GeneratorConfig::default();
    // Use single-bottleneck mode by default for reliable generation across all seeds.
    // Qualified multi-transition mode requires topology search coverage improvements.
    config.single_bottleneck = true;
    config.relax_transition_redundancy = true;
    config.width = generator_env_u64(GENERATOR_WIDTH_ENV, "invalid_width_override")?;
    config.height = generator_env_u64(GENERATOR_HEIGHT_ENV, "invalid_height_override")?;
    config.layers = generator_env_u64(GENERATOR_LAYERS_ENV, "invalid_layers_override")?;
    Ok(config)
}

fn print_generated_level_help() {
    eprintln!();
    eprintln!("Generated level selection:");
    eprintln!("  --level {}  (or no argument — default)", GENERATED_SELECTOR);
    eprintln!("  --seed <N>            Override generation seed");
    eprintln!();
    eprintln!("Generator environment overrides (generated only):");
    eprintln!("  {}=<N>   Seed (default: 0)", GENERATOR_SEED_ENV);
    eprintln!("  {}=<N>   Width (default: 96)", GENERATOR_WIDTH_ENV);
    eprintln!("  {}=<N>   Height (default: 96)", GENERATOR_HEIGHT_ENV);
    eprintln!("  {}=<N>   Layers (default: 3)", GENERATOR_LAYERS_ENV);
}

fn print_level_load_help() {
    eprintln!();
    eprintln!("Built-in level selectors:");
    eprintln!("  {}", GENERATED_SELECTOR);
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
    eprintln!("  --validate-colliders           Enable deterministic collider validation logging");
}

fn run_headless(
    level: &ParsedLevel,
    content_pack: &content::ContentPack,
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
        startup_model_path: None,
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

    if headless_opts.validate_colliders {
        log::info!(
            "[Collider Validation] Level seeded: {} chunk meshes, {} collider policies",
            _level_scene.chunk_meshes.len(),
            _level_scene.collider_policies.len(),
        );
    }

    // Seed collider bridge (headless path).
    let mut bridge = seed_collider_bridge(&mut renderer, &_level_scene, level);

    // Create component-driven physics bridge sharing the mesh bridge's world.
    let mut physics_bridge = PhysicsBridge::new();
    bridge.export_body_node_mappings_to_physics_bridge(&mut physics_bridge);

    if headless_opts.validate_colliders {
        log::info!(
            "[Collider Validation] Bridge seeded: {} recipes, bridge ready",
            bridge.recipe_count(),
        );
        // Log all body-node mappings.
        for (body_id, node_id) in bridge.body_node_map().iter() {
            if let Some(pos) = bridge.world.body_position_by_id(body_id) {
                log::info!(
                    "[Collider Validation] Body mapping: body={body_id} node_slot={} position=[{:.3}, {:.3}, {:.3}]",
                    node_id.slot,
                    pos[0], pos[1], pos[2],
                );
            }
        }

        // Deterministic proof independent of wall-clock frame pacing.
        for _ in 0..180 {
            bridge
                .world
                .step(FIXED_DT)
                .map_err(|error| AppError::RendererInit(RendererError::InvalidState(error.to_string())))?;
        }
        let contacts = bridge.world.last_contact_records();
        log::info!(
            "[Collider Validation] Contact proof: count={} records={:?}",
            contacts.len(),
            contacts,
        );
        if contacts.is_empty() {
            return Err(AppError::RendererInit(RendererError::InvalidState(
                "collider validation produced no contact".to_string(),
            )));
        }
        let writes = bridge
            .writeback_dynamic_transforms(&mut scene)
            .map_err(|error| AppError::RendererInit(RendererError::InvalidState(error.to_string())))?;
        log::info!(
            "[Collider Validation] Transform writeback proof: updated_nodes={writes}"
        );
        if writes == 0 {
            return Err(AppError::RendererInit(RendererError::InvalidState(
                "collider validation produced no transform writeback".to_string(),
            )));
        }
    }

    let spawn_world = tile_to_world(level.spawn.x, level.spawn.y);
    let spawn_position = spawn_world
        + glam::Vec3::new(
            0.5,
            level.spawn.layer as f32 * collision::WALL_HEIGHT + PLAYER_EYE_HEIGHT,
            -0.5,
        );

    // Create player character in the shared physics world.
    physics_bridge
        .create_player_character(
            &mut bridge.world,
            spawn_position.to_array(),
            PLAYER_CAPSULE_RADIUS,
            PLAYER_CAPSULE_HALF_HEIGHT,
        )
        .map_err(|e| AppError::RendererInit(RendererError::InvalidState(e.to_string())))?;

    let mut player = PlayerState::new(spawn_position);
    let mut previous_player_position = player.position;
    let mut app_camera = Camera::new(spawn_position);
    let mut app_input = InputSystem::new();
    let fps_controller = install_app_fps_input(&mut app_input);
    let fps_sensitivity = fps_controller.sensitivity();
    let fps_move_speed = fps_controller.move_speed();
    let movement_axis = movement_axis();
    let mut camera_yaw: f32 = 0.0;
    let mut action_events = InputActionEventEmitter::new();
    let mut time = Time::new(TimeConfig {
        step: Duration::from_secs_f32(FIXED_DT),
        max_steps_per_frame: 10,
        time_scale: 1.0,
    })
    .expect("valid default TimeConfig");

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
    let mut reported_manual_captures = HashSet::new();

    for frame_num in 0..frame_budget {
        match render_frame(
            &mut renderer,
            &mut scene,
            &mut bridge,
            &mut physics_bridge,
            &mut player,
            &mut previous_player_position,
            &mut app_camera,
            fps_sensitivity,
            fps_move_speed,
            &mut camera_yaw,
            &movement_axis,
            &mut app_input,
            &mut action_events,
            &mut app_events,
            &mut time,
            &mut reported_manual_captures,
            1280,
            720,
            true,
        ) {
            Ok(FrameRenderOutcome::Rendered)
            | Ok(FrameRenderOutcome::SkippedAcquireUnavailable)
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
        let selection = resolve_level_selector("level_02_ramps").unwrap();
        assert_eq!(selection.label, "level_02_ramps");
        assert_eq!(selection.path, PathBuf::from(LEVEL_02_PATH));
    }

    #[test]
    fn resolve_builtin_level_filename() {
        let selection = resolve_level_selector("level_03_lighting.txt").unwrap();
        assert_eq!(selection.label, "level_03_lighting");
        assert_eq!(selection.path, PathBuf::from(LEVEL_03_PATH));
    }

    #[test]
    fn resolve_level_01_path_as_default() {
        let selection = resolve_level_selector(DEFAULT_LEVEL_ID).unwrap();
        assert_eq!(selection.label, "generated_sprawl");
        assert!(matches!(selection.source, LevelSource::Generated(_)));
    }

    #[test]
    fn preserve_custom_level_paths() {
        let selection = resolve_level_selector("custom/level.txt").unwrap();
        assert_eq!(selection.label, "custom/level.txt");
        assert_eq!(selection.path, PathBuf::from("custom/level.txt"));
    }
}
