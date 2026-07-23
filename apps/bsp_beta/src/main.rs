//! BSP Beta — Atomic Runtime Publication (Phase 05)
//!
//! App-owned winit event loop with renderer integration, BSP mount lifecycle
//! via the bsp_runtime coordinator, physics bridge, behavior bridge, camera
//! controller, and headless capture support.
//!
//! # Usage
//!
//! Windowed: `cargo run -p bsp_beta -- --bsp maps/e1m1.bsp`
//! Headless: `cargo run -p bsp_beta -- --headless --capture-frames 5 --bsp maps/e1m1.bsp`

mod cli;

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use bsp_beta::physics_bridge::PhysicsBridge;
use bsp_beta::runtime_bridge::RuntimeBridge;
use bsp_beta::scene_sync::{sync_snapshot_to_scene, EntityNodeMap};
use bsp_beta::snapshot::{InlineModelInfo, ModelMappings, SnapshotProducer};
use bsp_runtime::bridge::{
    AppBridge, BehaviorEntityRecipe, EntityCollisionRecipe, LightEntityRecipe, WorldCollisionRecipe,
};
use bsp_runtime::coordinator::BspCoordinator;
use engine::camera::{Camera, FPSController};
use engine::events::{runtime_event_bus, EventBus};
use engine::frame::{FixedStepClock, FixedStepConfig, FrameClock};
use engine::input::{
    ActionMap, InputActionEventEmitter, InputSystem, LayerDescriptor, LayerPriority,
};
use engine::render::camera_view_for_size;
use engine_events::DispatchReport;
use glam::Vec3;
use renderer::api::config::RendererConfig;
use renderer::api::{CaptureTarget, FrameCaptureRequest, FrameCaptureStatus, FrameRenderOutcome};
use renderer::{Renderer, Scene};
use thiserror::Error;
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::keyboard::KeyCode;
use winit::window::WindowBuilder;

const APP_WINDOW_TITLE: &str = "BSP Beta — Phase 05";
const FIXED_DT: f32 = 1.0 / 60.0;

// ─── Error ────────────────────────────────────────────────────────────

#[derive(Debug, Error)]
enum AppError {
    #[error("no BSP path provided; use --bsp <path>")]
    NoBspPath,
    #[error("failed to read BSP file '{path}': {source}")]
    BspRead {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("BSP load/query proof failed: {code:?}: {message}")]
    BspLoadProof {
        code: bsp::diagnostic::DiagnosticCode,
        message: String,
    },
    #[error("BSP app-owned bridge proof failed: {0}")]
    BridgeProof(String),
    #[error("renderer init failed: {0}")]
    RendererInit(#[source] renderer::RendererError),
    #[error("renderer error: {0}")]
    Renderer(#[from] renderer::RendererError),
    #[error("BSP runtime error: {0}")]
    BspRuntime(#[from] bsp_runtime::BspRuntimeError),
    #[error("event loop error: {0}")]
    EventLoop(#[from] winit::error::EventLoopError),
    #[error("window error: {0}")]
    Window(#[from] winit::error::OsError),
}

// ─── Main ─────────────────────────────────────────────────────────────

fn main() {
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Info)
        .init();

    if let Err(err) = run() {
        eprintln!("{err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), AppError> {
    let args = cli::CliArgs::parse();

    let bsp_path = args.bsp_path.as_ref().ok_or(AppError::NoBspPath)?;
    let bsp_bytes = std::fs::read(bsp_path).map_err(|source| AppError::BspRead {
        path: bsp_path.clone(),
        source,
    })?;

    log::info!(
        "Loaded BSP: {} ({} bytes)",
        bsp_path.display(),
        bsp_bytes.len()
    );

    // ── Load companions (palette + .lit) ──────────────────────────────
    let palette_bytes = load_companion_bytes(
        &args.resolve_palette_path().map_err(|e| AppError::BridgeProof(e.to_string()))?,
        "palette",
    )?;
    log::info!("Loaded palette: {} bytes", palette_bytes.len());

    let lit_data: Option<Vec<u8>> = if let Some(lit_path) = args.resolve_lit_path() {
        let data = load_companion_bytes(&lit_path, ".lit")?;
        log::info!("Loaded .lit: {} bytes", data.len());
        Some(data)
    } else {
        log::info!("No .lit companion found (auto-discovered)");
        None
    };

    // ── Load BSP with companions ──────────────────────────────────────
    let t_build = Instant::now();
    let load_options = bsp::LoadOptions {
        strict: false,
        palette: Some(palette_bytes),
        lit_data: lit_data.clone(),
        source_identity: bsp_path.display().to_string(),
        ..bsp::LoadOptions::default()
    };
    let world = bsp::BspLoader::load(&bsp_bytes, &load_options).map_err(|report| {
        AppError::BspLoadProof {
            code: report.code,
            message: report.message,
        }
    })?;
    log::info!(
        "BSP parsed: {} faces, {} nodes, {} leaves ({}ms)",
        world.faces.len(),
        world.nodes.len(),
        world.leaves.len(),
        t_build.elapsed().as_millis(),
    );

    // ── Coordinator-based prepare ─────────────────────────────────────
    let mut coordinator = BspCoordinator::new();

    // Register app-owned bridges
    let physics_bridge = PhysicsBridge::new();
    let runtime_bridge = RuntimeBridge::new();
    coordinator.register_bridge("physics", Box::new(physics_bridge));
    coordinator.register_bridge("runtime", Box::new(runtime_bridge));

    // Prepare through coordinator using the pre-loaded world
    let prepare = coordinator.prepare_from_world(
        world,
        Some(args.scale),
        bsp_path.display().to_string(),
    )?;

    log::info!(
        "BSP extraction: {} faces, {} entities, {} lights, {} batches, PVS={} ({}ms total)",
        prepare.face_count,
        prepare.entity_count,
        prepare.light_count,
        prepare.batch_count,
        prepare.has_pvs,
        t_build.elapsed().as_millis(),
    );

    if args.show_lights {
        if let Some(extracted) = coordinator.staged_extraction() {
            for light in &extracted.light_descriptors {
                log::info!(
                    "  BSP light entity {}: pos=({:.1},{:.1},{:.1}) color=[{:.3},{:.3},{:.3}] intensity={:.1} radius={:.1} style={:?}",
                    light.entity_index,
                    light.origin.x, light.origin.y, light.origin.z,
                    light.color[0], light.color[1], light.color[2],
                    light.intensity, light.radius,
                    light.style,
                );
            }
        }
    }

    // ── Run startup proof ─────────────────────────────────────────────
    run_load_query_physics_behavior_proof(&args, &bsp_bytes, lit_data, bsp_path)?;

    // ── Run ────────────────────────────────────────────────────────────
    if args.headless {
        run_headless(&args, &mut coordinator)
    } else {
        run_windowed(&mut coordinator)
    }
}

/// Load companion file bytes.
fn load_companion_bytes(path: &PathBuf, _label: &str) -> Result<Vec<u8>, AppError> {
    std::fs::read(path).map_err(|source| AppError::BspRead {
        path: path.clone(),
        source,
    })
}

// ── Startup proof ─────────────────────────────────────────────────────

fn run_load_query_physics_behavior_proof(
    args: &cli::CliArgs,
    bsp_bytes: &[u8],
    lit_data: Option<Vec<u8>>,
    bsp_path: &PathBuf,
) -> Result<(), AppError> {
    // Direct extraction for physics/behavior proof (independent of coordinator
    // to exercise the raw bridge prepare/validate/commit path).
    let palette_bytes = match args.resolve_palette_path() {
        Ok(p) => std::fs::read(&p).map_err(|source| AppError::BspRead {
            path: p.clone(),
            source,
        })?,
        Err(e) => return Err(AppError::BridgeProof(format!("palette: {e}"))),
    };
    let load_options = bsp::LoadOptions {
        strict: false,
        palette: Some(palette_bytes),
        lit_data,
        source_identity: bsp_path.display().to_string(),
        ..bsp::LoadOptions::default()
    };
    let world = bsp::BspLoader::load(bsp_bytes, &load_options).map_err(|report| {
        AppError::BspLoadProof {
            code: report.code,
            message: report.message,
        }
    })?;
    let qte = bsp::coords::QuakeToEngine::new(args.scale);
    let contents = bsp::point_contents_with_transform(
        Vec3::ZERO,
        &world.nodes,
        &world.leaves,
        &world.planes,
        &qte,
    );

    // Direct extraction for bridge proof (world already has palette from load)
    let extracted = bsp::extract::extract(bsp::BspExtractionRequest {
        world,
        palette: None, // world.palette from load is used
        scale: args.scale,
        strict: false,
        ..Default::default()
    })
    .map_err(|report| AppError::BspLoadProof {
        code: report.code,
        message: report.message,
    })?;

    let bridge_inputs = bridge_inputs_from_extraction(&extracted);
    let mut physics_bridge = PhysicsBridge::new();
    let mut runtime_bridge = RuntimeBridge::new();

    let physics_token = physics_bridge
        .prepare(
            &bridge_inputs.world_collision,
            &bridge_inputs.entity_colliders,
            &bridge_inputs.lights,
            &bridge_inputs.behaviors,
        )
        .map_err(|err| AppError::BridgeProof(format!("physics prepare: {err}")))?;
    let runtime_token = runtime_bridge
        .prepare(
            &bridge_inputs.world_collision,
            &bridge_inputs.entity_colliders,
            &bridge_inputs.lights,
            &bridge_inputs.behaviors,
        )
        .map_err(|err| AppError::BridgeProof(format!("runtime prepare: {err}")))?;
    physics_bridge
        .validate(&physics_token)
        .map_err(|err| AppError::BridgeProof(format!("physics validate: {err}")))?;
    runtime_bridge
        .validate(&runtime_token)
        .map_err(|err| AppError::BridgeProof(format!("runtime validate: {err}")))?;

    let staged_body_count = physics_bridge
        .staged()
        .map_or(0, |staged| staged.bodies.len());
    let staged_collider_count = physics_bridge
        .staged()
        .map_or(0, |staged| staged.colliders.len());

    physics_bridge
        .commit(physics_token)
        .map_err(|err| AppError::BridgeProof(format!("physics commit: {err}")))?;
    runtime_bridge
        .commit(runtime_token)
        .map_err(|err| AppError::BridgeProof(format!("runtime commit: {err}")))?;

    let mut physics_world = physics::PhysicsWorld::new();
    physics_world.set_gravity(0.0, 0.0, 0.0);
    if let Err(err) = physics_bridge.commit_to_world(&mut physics_world) {
        log::warn!("BSP proof: physics world publish failed (non-fatal): {err}");
    }

    for (entity_index, position) in runtime_bridge.update(FIXED_DT) {
        let _ = physics_bridge.sync_body_transform(entity_index, position, &mut physics_world);
    }
    physics_world
        .step(FIXED_DT)
        .map_err(|err| AppError::BridgeProof(format!("physics step: {err}")))?;

    log::info!(
        "BSP proof: query@origin={:?}, physics={} bodies/{} colliders, behaviors={} entities",
        contents,
        staged_body_count,
        staged_collider_count,
        bridge_inputs.behaviors.len(),
    );

    Ok(())
}

struct BridgeInputs {
    world_collision: WorldCollisionRecipe,
    entity_colliders: Vec<EntityCollisionRecipe>,
    lights: Vec<LightEntityRecipe>,
    behaviors: Vec<BehaviorEntityRecipe>,
}

fn bridge_inputs_from_extraction(extracted: &bsp::extract::ExtractedBsp) -> BridgeInputs {
    let mut recipes_by_entity: HashMap<u32, Vec<bsp::collision::CollisionRecipe>> = HashMap::new();
    for recipe in &extracted.collision_recipes {
        recipes_by_entity
            .entry(recipe.entity_index)
            .or_default()
            .push(recipe.clone());
    }

    let entity_colliders = extracted
        .inline_models
        .iter()
        .map(|model| EntityCollisionRecipe {
            entity_index: model.entity_index,
            classname: model.classname.clone(),
            origin: model.origin,
            is_trigger: model.classname.starts_with("trigger_"),
            recipes: recipes_by_entity
                .remove(&model.entity_index)
                .unwrap_or_default(),
        })
        .collect();

    let lights = extracted
        .light_descriptors
        .iter()
        .map(|light| LightEntityRecipe {
            entity_index: light.entity_index,
            origin: light.origin,
            intensity: light.intensity,
            color: light.color,
            radius: light.radius,
            style: light.style.clone(),
        })
        .collect();

    let behaviors = extracted
        .entity_descriptors
        .iter()
        .filter(|entity| {
            entity.classname.starts_with("func_") || entity.classname.starts_with("trigger_")
        })
        .map(|entity| BehaviorEntityRecipe {
            entity_index: entity.entity_index,
            classname: entity.classname.clone(),
            origin: entity.origin.unwrap_or(Vec3::ZERO),
            targetname: None,
            target: None,
        })
        .collect();

    BridgeInputs {
        world_collision: WorldCollisionRecipe {
            planes: extracted.world_collision_planes.clone(),
        },
        entity_colliders,
        lights,
        behaviors,
    }
}

// ─── Windowed mode ─────────────────────────────────────────────────────

fn run_windowed(coordinator: &mut BspCoordinator) -> Result<(), AppError> {
    let event_loop = EventLoop::new()?;
    let window = WindowBuilder::new()
        .with_title(APP_WINDOW_TITLE)
        .with_inner_size(winit::dpi::LogicalSize::new(1280, 720))
        .build(&event_loop)?;

    let config = RendererConfig {
        app_name: "bsp_beta".to_string(),
        window_width: 1280,
        window_height: 720,
        ..RendererConfig::default()
    };

    let mut renderer = Renderer::new(config, &window).map_err(AppError::RendererInit)?;
    let mut scene = Scene::new();

    // Upload renderer resources from the staged extraction
    let extracted = coordinator
        .staged_extraction()
        .ok_or_else(|| AppError::BridgeProof("no staged extraction".to_string()))?;

    // ── Capture inline model info before commit consumes extraction ────
    let inline_model_infos: Vec<InlineModelInfo> = extracted
        .inline_models
        .iter()
        .map(|im| InlineModelInfo {
            entity_index: im.entity_index,
            model_index: im.model_index,
            origin: [im.origin.x, im.origin.y, im.origin.z],
            angles: im.angle.map(|a| [0.0_f32, a, 0.0_f32]),
            scale: None,
            local_mins: [
                im.local_bounds.0.x,
                im.local_bounds.0.y,
                im.local_bounds.0.z,
            ],
            local_maxs: [
                im.local_bounds.1.x,
                im.local_bounds.1.y,
                im.local_bounds.1.z,
            ],
        })
        .collect();

    let entity_classnames: HashMap<u32, String> = extracted
        .entity_descriptors
        .iter()
        .map(|ed| (ed.entity_index, ed.classname.clone()))
        .collect();

    let entity_source_models: HashMap<u32, String> = extracted
        .entity_descriptors
        .iter()
        .filter_map(|ed| ed.model_ref.map(|m| (ed.entity_index, format!("*{}", m))))
        .collect();

    let mount = renderer.prepare_bsp_mount(extracted)?;
    let token = bsp_runtime::BspGenerationToken {
        generation: coordinator.current_generation(),
    };
    coordinator.set_renderer_mount_ready(token, mount)?;

    // Validate all fallible publication checks, then commit.
    coordinator.validate_for_scene(token, &mut scene)?;
    coordinator.commit(token, &mut scene)?;

    log::info!("BSP mount uploaded and attached via coordinator");

    // ── Load model mappings ───────────────────────────────────────────
    let model_mappings = ModelMappings::default();

    // ── Camera + Input + Snapshot ──────────────────────────────────────
    let start_pos = Vec3::new(0.0, 2.0, 5.0);
    let mut loop_state = AppLoopState::new(start_pos, model_mappings);
    loop_state.inline_model_infos = inline_model_infos;
    loop_state.entity_classnames = entity_classnames;
    loop_state.entity_source_models = entity_source_models;
    install_app_fps_input(&mut loop_state.input);

    log::info!("BSP beta windowed mode initialized, starting event loop");
    window.request_redraw();

    event_loop
        .run(move |event, elwt| {
            elwt.set_control_flow(ControlFlow::Poll);

            let _routing = match engine::input::route_platform_input_to_app(
                &mut renderer,
                &window,
                &mut loop_state.input,
                &event,
            ) {
                Ok(r) => r,
                Err(e) => {
                    log::error!("Platform input routing failed: {e}");
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
                        WindowEvent::Resized(size) => {
                            if let Err(e) = renderer.resize(size.width, size.height) {
                                log::error!("Resize failed: {e}");
                                elwt.exit();
                            }
                        }
                        WindowEvent::RedrawRequested => {
                            let current_size = window.inner_size();

                            match render_app_frame(
                                &mut renderer,
                                &mut scene,
                                &mut loop_state,
                                current_size.width,
                                current_size.height,
                                false,
                            ) {
                                Ok(FrameRenderOutcome::Rendered)
                                | Ok(FrameRenderOutcome::SkippedAcquireUnavailable)
                                | Ok(FrameRenderOutcome::SkippedResizePending)
                                | Ok(FrameRenderOutcome::SubmittedNotPresented)
                                | Ok(FrameRenderOutcome::PresentedSuboptimal) => {}
                                Err(e) => {
                                    log::error!("Render failed: {e}");
                                    elwt.exit();
                                    return;
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

// ─── Headless mode ─────────────────────────────────────────────────────

fn run_headless(args: &cli::CliArgs, coordinator: &mut BspCoordinator) -> Result<(), AppError> {
    log::info!("Starting BSP beta headless mode");

    let config = RendererConfig {
        app_name: "bsp_beta".to_string(),
        window_width: 1920,
        window_height: 1080,
        headless: true,
        ..RendererConfig::default()
    };

    let mut renderer = Renderer::new_headless(config).map_err(AppError::RendererInit)?;
    let mut scene = Scene::new();

    // Upload renderer resources from the staged extraction
    let extracted = coordinator
        .staged_extraction()
        .ok_or_else(|| AppError::BridgeProof("no staged extraction".to_string()))?;

    // ── Capture inline model info before commit consumes extraction ────
    let inline_model_infos: Vec<InlineModelInfo> = extracted
        .inline_models
        .iter()
        .map(|im| InlineModelInfo {
            entity_index: im.entity_index,
            model_index: im.model_index,
            origin: [im.origin.x, im.origin.y, im.origin.z],
            angles: im.angle.map(|a| [0.0_f32, a, 0.0_f32]),
            scale: None,
            local_mins: [
                im.local_bounds.0.x,
                im.local_bounds.0.y,
                im.local_bounds.0.z,
            ],
            local_maxs: [
                im.local_bounds.1.x,
                im.local_bounds.1.y,
                im.local_bounds.1.z,
            ],
        })
        .collect();

    let entity_classnames: HashMap<u32, String> = extracted
        .entity_descriptors
        .iter()
        .map(|ed| (ed.entity_index, ed.classname.clone()))
        .collect();

    let entity_source_models: HashMap<u32, String> = extracted
        .entity_descriptors
        .iter()
        .filter_map(|ed| ed.model_ref.map(|m| (ed.entity_index, format!("*{}", m))))
        .collect();

    let mount = renderer.prepare_bsp_mount(extracted)?;
    let token = bsp_runtime::BspGenerationToken {
        generation: coordinator.current_generation(),
    };
    coordinator.set_renderer_mount_ready(token, mount)?;

    // Validate all fallible publication checks, then commit.
    coordinator.validate_for_scene(token, &mut scene)?;
    coordinator.commit(token, &mut scene)?;

    // ── Camera + app-owned loop state ─────────────────────────────────
    let mut loop_state = AppLoopState::new(Vec3::new(0.0, 3.0, 10.0), ModelMappings::default());
    loop_state.inline_model_infos = inline_model_infos;
    loop_state.entity_classnames = entity_classnames;
    loop_state.entity_source_models = entity_source_models;

    // ── Warmup ─────────────────────────────────────────────────────────
    for _ in 0..5 {
        render_app_frame(&mut renderer, &mut scene, &mut loop_state, 1920, 1080, true)
            .map_err(AppError::Renderer)?;
    }

    // ── Capture ────────────────────────────────────────────────────────
    if args.capture_frames > 0 {
        let capture_dir = PathBuf::from(format!(
            ".internal-dev/captures/bsp-beta/headless-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&capture_dir).map_err(|e| {
            AppError::RendererInit(renderer::RendererError::InvalidState(format!(
                "create capture dir: {e}"
            )))
        })?;

        for frame_num in 0..args.capture_frames {
            let png_path = capture_dir.join(format!("bsp_beta_frame_{frame_num:04}.png"));
            let sidecar_path =
                capture_dir.join(format!("bsp_beta_frame_{frame_num:04}_sidecar.json"));

            renderer
                .request_frame_capture(FrameCaptureRequest {
                    target: CaptureTarget::Draw,
                    output_path: png_path.clone(),
                    sidecar_path: Some(sidecar_path),
                })
                .map_err(|e| {
                    AppError::RendererInit(renderer::RendererError::InvalidState(format!(
                        "capture request: {e}"
                    )))
                })?;

            render_app_frame(&mut renderer, &mut scene, &mut loop_state, 1920, 1080, true)
                .map_err(AppError::Renderer)?;

            match renderer.last_frame_capture_status() {
                Some(FrameCaptureStatus::Succeeded {
                    output_path,
                    width,
                    height,
                    ..
                }) => {
                    log::info!(
                        "✓ Frame {frame_num}: {} ({}×{})",
                        output_path.display(),
                        width,
                        height
                    );
                }
                Some(FrameCaptureStatus::Failed { message, .. }) => {
                    log::error!("✗ Frame {frame_num}: {message}");
                }
                _ => {
                    log::warn!("Frame {frame_num}: capture status not reported");
                }
            }
        }
    } else {
        // Smoke render: just a few frames
        let smoke_frames = 5u32;
        for frame_num in 0..smoke_frames {
            render_app_frame(&mut renderer, &mut scene, &mut loop_state, 1920, 1080, true)
                .map_err(AppError::Renderer)?;
            log::info!("Smoke frame {frame_num}/{smoke_frames} rendered");
        }
    }

    log::info!("BSP beta headless complete");
    Ok(())
}

// ─── App-owned frame loop helpers ──────────────────────────────────────

struct AppLoopState {
    camera: Camera,
    fps_controller: FPSController,
    input: InputSystem,
    action_events: InputActionEventEmitter,
    events: EventBus,
    frame_clock: FrameClock,
    fixed_clock: FixedStepClock,
    /// Runtime bridge (owned for snapshot production).
    runtime_bridge: Option<RuntimeBridge>,
    /// Physics bridge (owned for snapshot sync).
    #[allow(dead_code)]
    physics_bridge: Option<PhysicsBridge>,
    /// Snapshot producer.
    snapshot_producer: SnapshotProducer,
    /// Inline model info for pose computation.
    inline_model_infos: Vec<InlineModelInfo>,
    /// Entity classname lookup.
    entity_classnames: HashMap<u32, String>,
    /// Entity source model lookup.
    entity_source_models: HashMap<u32, String>,
    /// Scene node map for snapshot-driven external/inline nodes.
    entity_node_map: EntityNodeMap,
}

impl AppLoopState {
    fn new(camera_position: Vec3, model_mappings: ModelMappings) -> Self {
        Self {
            camera: Camera::new(camera_position),
            fps_controller: FPSController::new(0.002, 1.0),
            input: InputSystem::new(),
            action_events: InputActionEventEmitter::new(),
            events: runtime_event_bus(),
            frame_clock: FrameClock::new(),
            fixed_clock: FixedStepClock::new(FixedStepConfig {
                step: Duration::from_secs_f32(FIXED_DT),
                max_steps_per_frame: 4,
            }),
            runtime_bridge: None,
            physics_bridge: None,
            snapshot_producer: SnapshotProducer::new(model_mappings),
            inline_model_infos: Vec::new(),
            entity_classnames: HashMap::new(),
            entity_source_models: HashMap::new(),
            entity_node_map: EntityNodeMap::default(),
        }
    }
}

fn render_app_frame(
    renderer: &mut Renderer,
    scene: &mut Scene,
    state: &mut AppLoopState,
    viewport_width: u32,
    viewport_height: u32,
    headless: bool,
) -> Result<FrameRenderOutcome, renderer::RendererError> {
    let begin_report = engine::frame::begin_app_frame(
        &mut state.input,
        &mut state.action_events,
        &mut state.events,
        &mut state.frame_clock,
    );
    log_dispatch_failures(&begin_report.input_dispatch, "bsp_beta input");
    log_dispatch_failures(&begin_report.frame_started, "bsp_beta lifecycle");

    let fixed_update = state.fixed_clock.update(begin_report.frame.delta);
    let simulated_dt = if fixed_update.steps > 0 {
        FIXED_DT * fixed_update.steps as f32
    } else {
        begin_report.frame.delta_seconds.min(FIXED_DT)
    };
    state.fps_controller.update_from_snapshot(
        state.input.snapshot(),
        simulated_dt,
        &mut state.camera,
    );

    if !fixed_update.dropped_time.is_zero() {
        log::warn!(
            "Dropped {:.3}ms of accumulated BSP beta simulation time.",
            fixed_update.dropped_time.as_secs_f64() * 1_000.0
        );
    }

    // ── Snapshot production at each fixed step ───────────────────────
    for _step in 0..fixed_update.steps {
        if let Some(ref mut runtime) = state.runtime_bridge {
            let snapshot = state.snapshot_producer.produce(
                FIXED_DT,
                runtime,
                &state.inline_model_infos,
                &state.entity_classnames,
                &state.entity_source_models,
            );

            sync_snapshot_to_scene(snapshot.as_ref(), &state.entity_node_map, scene);
        }
    }

    renderer.pump_asset_tasks(32)?;
    let view = camera_view_for_size(&state.camera, viewport_width, viewport_height);

    let outcome = if headless {
        renderer.render_scene_headless_with_view(scene, view)?
    } else {
        renderer.render_scene_with_view(scene, view)?
    };

    let end_report = engine::frame::end_app_frame(&mut state.events, begin_report.frame.index);
    log_dispatch_failures(&end_report.frame_ended, "bsp_beta lifecycle");

    Ok(outcome)
}

fn log_dispatch_failures(report: &DispatchReport, label: &str) {
    for failure in &report.failures {
        log::warn!(
            "{} dispatch listener {:?} failed at event {:?}: {}",
            label,
            failure.listener,
            failure.sequence,
            failure.message
        );
    }
}

// ─── Input helpers ─────────────────────────────────────────────────────

fn install_app_fps_input(input: &mut InputSystem) {
    let mut map = ActionMap::new();
    map.bind_key("move.forward", KeyCode::KeyW);
    map.bind_key("move.backward", KeyCode::KeyS);
    map.bind_key("move.left", KeyCode::KeyA);
    map.bind_key("move.right", KeyCode::KeyD);
    map.bind_key("move.up", KeyCode::Space);
    map.bind_key("move.down", KeyCode::ShiftLeft);

    input.add_layer(
        LayerDescriptor::new("bsp-beta-fps-actions", LayerPriority(10)),
        map.into_layer(),
    );
}
