//! BSP Beta — Atomic Runtime Publication (Phase 05)
//!
//! App-owned winit event loop with renderer integration, BSP mount lifecycle
//! via the bsp_runtime coordinator, physics bridge, behavior bridge, camera
//! controller, and headless capture support.
//!
//! # Usage
//!
//! Windowed: `cargo run -p bsp_beta -- --strict --bsp maps/e1m1.bsp --palette gfx/palette.lmp --wad maps/dungeon.wad --textures textures/`
//! Headless: `cargo run -p bsp_beta -- --strict --headless --capture-frames 5 --bsp maps/e1m1.bsp --palette gfx/palette.lmp`
//! MCP: `cargo run -p bsp_beta -- --strict --mcp --bsp maps/e1m1.bsp --palette gfx/palette.lmp`

mod cli;
mod mcp;

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
use bsp_runtime::package::{self, effective_import_summary};
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
    #[error("no import mode selected; use --strict or --development")]
    NoImportMode,
    #[error("BSP app-owned bridge proof failed: {0}")]
    BridgeProof(String),
    #[error("renderer init failed: {0}")]
    RendererInit(#[source] renderer::RendererError),
    #[error("renderer error: {0}")]
    Renderer(#[from] renderer::RendererError),
    #[error("BSP runtime error: {0}")]
    BspRuntime(#[from] bsp_runtime::BspRuntimeError),
    #[error("package load error: {0}")]
    PackageLoad(#[from] package::PackageLoadError),
    #[error("event loop error: {0}")]
    EventLoop(#[from] winit::error::EventLoopError),
    #[error("window error: {0}")]
    Window(#[from] winit::error::OsError),
    #[error("MCP stdio error: {0}")]
    McpIo(#[from] std::io::Error),
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
    let palette_path = args
        .resolve_palette_path()
        .map_err(|e| AppError::BridgeProof(e.to_string()))?;

    let lit_path = args.resolve_lit_path();
    let wad_path = args
        .resolve_wad_path()
        .map_err(|e| AppError::BridgeProof(e.to_string()))?;

    let import_mode = args
        .require_import_mode()
        .map_err(|_| AppError::NoImportMode)?;

    let bsp_runtime_mode = match import_mode {
        cli::ImportMode::Strict => package::ImportMode::Strict,
        cli::ImportMode::Development => package::ImportMode::Development,
    };

    let textures_dir = args.textures_dir.as_deref();

    // ── Authorize direct import through the runtime package boundary ────
    let t_build = Instant::now();
    let wad_paths: Vec<PathBuf> = wad_path.into_iter().collect();
    let import = package::authorize_direct_import(
        bsp_path,
        &palette_path,
        lit_path.as_deref(),
        &wad_paths,
        textures_dir,
        bsp_runtime_mode,
        args.scale,
    )?;

    log::info!(
        "BSP authorized: {} ({} bytes, {}ms)",
        bsp_path.display(),
        import.bsp.bytes.len(),
        t_build.elapsed().as_millis(),
    );

    // Emit effective-import summary.
    let summary = effective_import_summary(&import);
    log::info!("Effective import:\n{summary}");

    // ── Coordinator-based prepare ─────────────────────────────────────
    let mut coordinator = BspCoordinator::new();

    // Register app-owned bridges
    let physics_bridge = PhysicsBridge::new();
    let runtime_bridge = RuntimeBridge::new();
    coordinator.register_bridge("physics", Box::new(physics_bridge));
    coordinator.register_bridge("runtime", Box::new(runtime_bridge));

    // The proof below consumes this already-authorized parsed world and the
    // coordinator's staged extraction; it never reauthorizes, reparses, or
    // reextracts the launch inputs.
    let proof_world = import.world.clone();
    let prepare = coordinator.prepare_authorized_import(import)?;

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
    let extracted = coordinator.staged_extraction().ok_or_else(|| {
        AppError::BridgeProof("authorized import did not stage extraction".into())
    })?;
    run_load_query_physics_behavior_proof(&proof_world, args.scale, extracted)?;

    // ── Run ────────────────────────────────────────────────────────────
    if args.mcp {
        run_mcp(&mut coordinator)
    } else if args.headless {
        run_headless(&args, &mut coordinator)
    } else {
        run_windowed(&mut coordinator)
    }
}

// ─── Startup proof ─────────────────────────────────────────────────────

fn run_load_query_physics_behavior_proof(
    world: &bsp::world::BspWorld,
    scale: f32,
    extracted: &bsp::extract::ExtractedBsp,
) -> Result<(), AppError> {
    let qte = bsp::coords::QuakeToEngine::new(scale);
    let contents = bsp::point_contents_with_transform(
        Vec3::ZERO,
        &world.nodes,
        &world.leaves,
        &world.planes,
        &qte,
    );

    let bridge_inputs = bridge_inputs_from_extraction(extracted);
    let mut physics_bridge = PhysicsBridge::new();
    let mut runtime_bridge = RuntimeBridge::new();

    let mut physics_prepared = physics_bridge
        .prepare(
            &bridge_inputs.world_collision,
            &bridge_inputs.entity_colliders,
            &bridge_inputs.lights,
            &bridge_inputs.behaviors,
        )
        .map_err(|err| AppError::BridgeProof(format!("physics prepare: {err}")))?;
    let mut runtime_prepared = runtime_bridge
        .prepare(
            &bridge_inputs.world_collision,
            &bridge_inputs.entity_colliders,
            &bridge_inputs.lights,
            &bridge_inputs.behaviors,
        )
        .map_err(|err| AppError::BridgeProof(format!("runtime prepare: {err}")))?;

    physics_bridge
        .validate(&*physics_prepared)
        .map_err(|err| AppError::BridgeProof(format!("physics validate: {err}")))?;
    runtime_bridge
        .validate(&*runtime_prepared)
        .map_err(|err| AppError::BridgeProof(format!("runtime validate: {err}")))?;

    let staged_body_count = physics_prepared
        .as_any()
        .downcast_ref::<bsp_beta::physics_bridge::PhysicsPreparedState>()
        .map_or(0, |s| s.all_body_ids.len());
    let staged_collider_count = physics_prepared
        .as_any()
        .downcast_ref::<bsp_beta::physics_bridge::PhysicsPreparedState>()
        .map_or(0, |s| s.all_collider_ids.len());

    let mut physics_active = physics_bridge.activate(&mut *physics_prepared);
    let mut runtime_active = runtime_bridge.activate(&mut *runtime_prepared);

    let physics_active_state: &mut bsp_beta::physics_bridge::PhysicsActiveState = physics_active
        .as_any_mut()
        .downcast_mut::<bsp_beta::physics_bridge::PhysicsActiveState>()
        .expect("physics active state type mismatch");

    for (entity_index, position) in runtime_bridge.update(FIXED_DT) {
        let _ = physics_bridge.sync_body_transform(
            entity_index, position, &mut physics_active_state.world,
        );
    }
    physics_active_state
        .world
        .step(FIXED_DT)
        .map_err(|err| AppError::BridgeProof(format!("physics step: {err}")))?;

    log::info!(
        "BSP proof: query@origin={:?}, physics={} bodies/{} colliders, behaviors={} entities",
        contents,
        staged_body_count,
        staged_collider_count,
        bridge_inputs.behaviors.len(),
    );

    let _ = physics_bridge.teardown(&mut *physics_active);
    let _ = runtime_bridge.teardown(&mut *runtime_active);

    Ok(())
}

struct BridgeInputs {
    world_collision: WorldCollisionRecipe,
    entity_colliders: Vec<EntityCollisionRecipe>,
    lights: Vec<LightEntityRecipe>,
    behaviors: Vec<BehaviorEntityRecipe>,
}

fn bridge_inputs_from_extraction(extracted: &bsp::extract::ExtractedBsp) -> BridgeInputs {
    use std::collections::HashMap;

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
            killtarget: None,
            movedir: None,
            speed: None,
            wait: None,
            lip: None,
            height: None,
            light_style: None,
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

fn bsp_player_start(extracted: &bsp::extract::ExtractedBsp, fallback: Vec3) -> Vec3 {
    let start = extracted
        .entity_descriptors
        .iter()
        .find(|entity| {
            matches!(
                entity.classname.as_str(),
                "info_player_start" | "info_player_deathmatch"
            )
        })
        .and_then(|entity| entity.origin)
        .filter(|origin| origin.is_finite())
        .unwrap_or(fallback);
    log::info!("BSP camera start: {start:?}");
    start
}

/// Start headless and MCP views at the authored player spawn.
fn bsp_headless_camera(start_pos: Vec3, _extracted: &bsp::extract::ExtractedBsp) -> Camera {
    let camera = Camera::new(start_pos);
    log::info!("BSP headless camera: pos={start_pos:?} (info_player_start)");
    camera
}

const ACCEPTANCE_CLEARANCE_STEP: f32 = 0.1;
const ACCEPTANCE_CLEARANCE_LIMIT: f32 = 128.0;

fn acceptance_point_contents(
    point: Vec3,
    extracted: &bsp::extract::ExtractedBsp,
) -> bsp::queries::PointContents {
    bsp::queries::point_contents_with_transform(
        point,
        &extracted.visibility.nodes,
        &extracted.visibility.leaves,
        &extracted.visibility.planes,
        &extracted.transform,
    )
}

fn acceptance_clear_distance(
    eye: Vec3,
    direction: Vec3,
    extracted: &bsp::extract::ExtractedBsp,
) -> f32 {
    let mut distance = ACCEPTANCE_CLEARANCE_STEP;
    while distance <= ACCEPTANCE_CLEARANCE_LIMIT {
        if acceptance_point_contents(eye + direction * distance, extracted).is_solid() {
            return distance - ACCEPTANCE_CLEARANCE_STEP;
        }
        distance += ACCEPTANCE_CLEARANCE_STEP;
    }
    ACCEPTANCE_CLEARANCE_LIMIT
}

fn longest_acceptance_direction(eye: Vec3, extracted: &bsp::extract::ExtractedBsp) -> (Vec3, f32) {
    // Prefer the engine's default forward direction for deterministic ties.
    [Vec3::NEG_Z, Vec3::X, Vec3::Z, Vec3::NEG_X]
        .into_iter()
        .map(|direction| {
            (
                direction,
                acceptance_clear_distance(eye, direction, extracted),
            )
        })
        .reduce(|best, candidate| {
            if candidate.1 > best.1 {
                candidate
            } else {
                best
            }
        })
        .unwrap_or((Vec3::NEG_Z, 0.0))
}

fn camera_angles_for_direction(direction: Vec3) -> (f32, f32) {
    let direction = direction.normalize_or_zero();
    let horizontal = (direction.x * direction.x + direction.z * direction.z).sqrt();
    let yaw = (-direction.x).atan2(-direction.z);
    let pitch = direction.y.atan2(horizontal);
    (yaw, pitch)
}

/// Fixed acceptance camera for the given semantic label.
///
/// Generated spawn origins already represent eye height. The camera therefore
/// starts at an authored non-solid origin and chooses the longest clear
/// cardinal view instead of adding a second eye-height offset or averaging an
/// arbitrary point that may lie inside a wall or ceiling.
fn bsp_acceptance_camera(
    label: &str,
    extracted: &bsp::extract::ExtractedBsp,
) -> Result<Camera, AppError> {
    let player_start = extracted.entity_descriptors.iter().find(|entity| {
        matches!(
            entity.classname.as_str(),
            "info_player_start" | "info_player_deathmatch"
        )
    });

    let origins: Vec<Vec3> = extracted
        .entity_descriptors
        .iter()
        .filter_map(|entity| entity.origin)
        .filter(|origin| origin.is_finite())
        .collect();
    let map_center = if origins.is_empty() {
        Vec3::ZERO
    } else {
        origins.iter().copied().sum::<Vec3>() / origins.len() as f32
    };
    let spawn_pos = player_start
        .and_then(|entity| entity.origin)
        .filter(|origin| origin.is_finite())
        .unwrap_or(map_center);

    let eye = match label {
        "spawn" | "corridor" => spawn_pos,
        "junction" => origins
            .iter()
            .copied()
            .filter(|origin| !acceptance_point_contents(*origin, extracted).is_solid())
            .min_by(|left, right| {
                left.distance_squared(map_center)
                    .total_cmp(&right.distance_squared(map_center))
            })
            .unwrap_or(spawn_pos),
        _ => {
            return Err(AppError::BridgeProof(format!(
                "unknown acceptance camera label: {label}"
            )));
        }
    };

    if !eye.is_finite() {
        return Err(AppError::BridgeProof(format!(
            "acceptance camera '{label}' has non-finite eye data: {eye:?}"
        )));
    }
    let contents = acceptance_point_contents(eye, extracted);
    if contents.is_solid() {
        return Err(AppError::BridgeProof(format!(
            "acceptance camera '{label}' eye lies in solid space: {eye:?}"
        )));
    }

    let (direction, clear_distance) = longest_acceptance_direction(eye, extracted);
    let look_at = eye + direction * clear_distance.max(5.0);
    let (yaw, pitch) = camera_angles_for_direction(direction);

    let mut camera = Camera::new(eye);
    camera.update_rotation(yaw, pitch);

    log::info!(
        "Acceptance camera '{label}': eye={eye:?}, look_at={look_at:?}, contents={contents:?}, clear_distance={clear_distance:.2}, yaw={yaw:.3}, pitch={pitch:.3}",
    );

    Ok(camera)
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
    let player_start = bsp_player_start(extracted, Vec3::new(0.0, 2.0, 5.0));

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

    let entity_classnames: std::collections::HashMap<u32, String> = extracted
        .entity_descriptors
        .iter()
        .map(|ed| (ed.entity_index, ed.classname.clone()))
        .collect();

    let entity_source_models: std::collections::HashMap<u32, String> = extracted
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
    let mut loop_state = AppLoopState::new(Camera::new(player_start), model_mappings);
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

struct HeadlessRuntime {
    renderer: Renderer,
    scene: Scene,
    loop_state: AppLoopState,
    mcp_map: Option<mcp::McpMap>,
}

fn prepare_headless_runtime(
    coordinator: &mut BspCoordinator,
    mcp_map_data: Option<&bsp::extract::ExtractedBsp>,
    args: &cli::CliArgs,
) -> Result<HeadlessRuntime, AppError> {
    let is_acceptance = args.acceptance_camera.is_some();
    let config = RendererConfig {
        app_name: "bsp_beta".to_string(),
        window_width: if is_acceptance { 1280 } else { 1920 },
        window_height: if is_acceptance { 720 } else { 1080 },
        headless: true,
        ..RendererConfig::default()
    };

    let mut renderer = Renderer::new_headless(config).map_err(AppError::RendererInit)?;
    let mut scene = Scene::new();

    // Upload renderer resources from the staged extraction.
    let extracted = coordinator
        .staged_extraction()
        .ok_or_else(|| AppError::BridgeProof("no staged extraction".to_string()))?;
    let player_start = bsp_player_start(extracted, Vec3::new(0.0, 3.0, 10.0));

    // Capture app-owned state before coordinator commit consumes extraction.
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
    let entity_classnames: std::collections::HashMap<u32, String> = extracted
        .entity_descriptors
        .iter()
        .map(|ed| (ed.entity_index, ed.classname.clone()))
        .collect();
    let entity_source_models: std::collections::HashMap<u32, String> = extracted
        .entity_descriptors
        .iter()
        .filter_map(|ed| ed.model_ref.map(|m| (ed.entity_index, format!("*{}", m))))
        .collect();
    let headless_camera = if let Some(ref cam_label) = args.acceptance_camera {
        bsp_acceptance_camera(cam_label, extracted)?
    } else {
        bsp_headless_camera(player_start, extracted)
    };

    let mount = renderer.prepare_bsp_mount(extracted)?;
    let mcp_map = mcp_map_data.map(|ext| mcp::McpMap::from_mount(ext, &mount, 0));
    let token = bsp_runtime::BspGenerationToken {
        generation: coordinator.current_generation(),
    };
    coordinator.set_renderer_mount_ready(token, mount)?;
    coordinator.validate_for_scene(token, &mut scene)?;
    coordinator.commit(token, &mut scene)?;

    let mut loop_state = AppLoopState::new(headless_camera, ModelMappings::default());
    loop_state.inline_model_infos = inline_model_infos;
    loop_state.entity_classnames = entity_classnames;
    loop_state.entity_source_models = entity_source_models;

    Ok(HeadlessRuntime {
        renderer,
        scene,
        loop_state,
        mcp_map,
    })
}

fn run_headless(args: &cli::CliArgs, coordinator: &mut BspCoordinator) -> Result<(), AppError> {
    let is_acceptance = args.acceptance_camera.is_some();
    let (vp_width, vp_height) = if is_acceptance {
        log::info!(
            "Acceptance mode: camera={}, frozen 1280×720, exposure=1.0, overbright=2.0, style=0, anim_time=0.0",
            args.acceptance_camera.as_deref().unwrap_or("spawn")
        );
        (1280, 720)
    } else {
        (1920, 1080)
    };

    log::info!("Starting BSP beta headless mode ({}×{})", vp_width, vp_height);
    let HeadlessRuntime {
        mut renderer,
        mut scene,
        mut loop_state,
        ..
    } = prepare_headless_runtime(coordinator, None, args)?;

    // ── Warmup ─────────────────────────────────────────────────────────
    for _ in 0..5 {
        render_app_frame(
            &mut renderer,
            &mut scene,
            &mut loop_state,
            vp_width,
            vp_height,
            true,
        )
        .map_err(AppError::Renderer)?;
    }

    // ── Phase 07: Stats evidence request (before capture/smoke) ────────
    if args.stats {
        use renderer::api::bsp::BspEvidenceVisibility;
        let visibility = if args.all_visible {
            BspEvidenceVisibility::AllVisible
        } else {
            BspEvidenceVisibility::NormalPvs
        };
        let corpus = args.corpus_identity.clone()
            .unwrap_or_else(|| "bsp-beta-headless".to_string());
        let stats_key = renderer
            .request_bsp_frame_evidence(corpus, "headless-stats".to_string(), visibility)
            .map_err(|e| AppError::RendererInit(renderer::RendererError::InvalidState(
                format!("evidence request: {e}")
            )))?;

        // Render until evidence is sealed.
        for _attempt in 0..16 {
            render_app_frame(&mut renderer, &mut scene, &mut loop_state, vp_width, vp_height, true)
                .map_err(AppError::Renderer)?;
            match renderer.take_bsp_frame_evidence(stats_key) {
                renderer::api::bsp::BspEvidenceStatus::Sealed(report) => {
                    let serialized = serde_json::to_string_pretty(&report)
                        .map_err(|e| AppError::RendererInit(
                            renderer::RendererError::InvalidState(format!("serialize report: {e}"))
                        ))?;
                    println!("{serialized}");
                    if !report.eligible {
                        log::warn!("Stats report is not eligible for acceptance");
                        std::process::exit(1);
                    }
                    return Ok(());
                }
                renderer::api::bsp::BspEvidenceStatus::RejectedNoMount => {
                    log::error!("Stats request rejected: no active BSP mount");
                    std::process::exit(1);
                }
                renderer::api::bsp::BspEvidenceStatus::MissingReport => {
                    log::error!("Stats report missing");
                    std::process::exit(1);
                }
                renderer::api::bsp::BspEvidenceStatus::Pending => {
                    // Continue rendering
                }
            }
        }
        log::error!("Stats report not ready after bounded frame attempts");
        std::process::exit(1);
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

            for _ in 0..8 {
                render_app_frame(&mut renderer, &mut scene, &mut loop_state, vp_width, vp_height, true)
                    .map_err(AppError::Renderer)?;
                if png_path.is_file() {
                    break;
                }
            }

            if !png_path.is_file() {
                let message = match renderer.last_frame_capture_status() {
                    Some(FrameCaptureStatus::Failed { message, .. }) => message.clone(),
                    Some(FrameCaptureStatus::BackendNotImplemented { .. }) => {
                        "capture backend not implemented".to_string()
                    }
                    Some(FrameCaptureStatus::Pending { .. }) => {
                        "capture remained pending after bounded frame pumping".to_string()
                    }
                    _ => "capture status not reported".to_string(),
                };
                return Err(AppError::RendererInit(
                    renderer::RendererError::InvalidState(format!(
                        "frame {frame_num} capture failed: {message}"
                    )),
                ));
            }

            match renderer.last_frame_capture_status() {
                Some(FrameCaptureStatus::Succeeded {
                    output_path,
                    width,
                    height,
                    ..
                }) if *output_path == png_path => {
                    log::info!(
                        "✓ Frame {frame_num}: {} ({}×{})",
                        output_path.display(),
                        width,
                        height
                    );
                }
                _ => log::info!("✓ Frame {frame_num}: {}", png_path.display()),
            }
        }
    } else {
        let smoke_frames = 5u32;
        for frame_num in 0..smoke_frames {
            render_app_frame(&mut renderer, &mut scene, &mut loop_state, vp_width, vp_height, true)
                .map_err(AppError::Renderer)?;
            log::info!("Smoke frame {frame_num}/{smoke_frames} rendered");
        }
    }

    log::info!("BSP beta headless complete");
    Ok(())
}

fn run_mcp(coordinator: &mut BspCoordinator) -> Result<(), AppError> {
    log::info!("Starting BSP beta MCP mode (headless 1920×1080)");
    // Capture extraction before it gets consumed.
    let extracted_clone = coordinator
        .staged_extraction()
        .ok_or_else(|| AppError::BridgeProof("no staged extraction".to_string()))?
        .clone();
    let _bsp_size = extracted_clone.face_geometries.len();

    let HeadlessRuntime {
        mut renderer,
        mut scene,
        mut loop_state,
        mcp_map,
    } = prepare_headless_runtime(coordinator, Some(&extracted_clone), &cli::CliArgs::default())?;
    let mcp_map = mcp_map
        .ok_or_else(|| AppError::BridgeProof("MCP map data was not retained".to_string()))?;

    // Prime GPU resources before accepting the first capture request.
    for _ in 0..5 {
        render_app_frame(&mut renderer, &mut scene, &mut loop_state, 1920, 1080, true)?;
    }

    log::info!("BSP mount published; serving MCP JSON-RPC on stdio");
    mcp::serve(&mut renderer, &mut scene, &mut loop_state, mcp_map)?;
    log::info!("MCP stdin closed; shutting down");
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
    entity_classnames: std::collections::HashMap<u32, String>,
    /// Entity source model lookup.
    entity_source_models: std::collections::HashMap<u32, String>,
    /// Scene node map for snapshot-driven external/inline nodes.
    entity_node_map: EntityNodeMap,
}

impl AppLoopState {
    fn new(camera: Camera, model_mappings: ModelMappings) -> Self {
        Self {
            camera,
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
            entity_classnames: std::collections::HashMap::new(),
            entity_source_models: std::collections::HashMap::new(),
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

#[cfg(test)]
mod acceptance_camera_tests {
    use super::camera_angles_for_direction;
    use glam::Vec3;

    fn assert_near(actual: f32, expected: f32) {
        assert!((actual - expected).abs() < 1.0e-5, "{actual} != {expected}");
    }

    #[test]
    fn camera_angles_match_camera_forward_convention() {
        let (yaw, pitch) = camera_angles_for_direction(Vec3::NEG_Z);
        assert_near(yaw, 0.0);
        assert_near(pitch, 0.0);

        let (yaw, pitch) = camera_angles_for_direction(Vec3::X);
        assert_near(yaw, -std::f32::consts::FRAC_PI_2);
        assert_near(pitch, 0.0);

        let (_, pitch) = camera_angles_for_direction(Vec3::Y);
        assert_near(pitch, std::f32::consts::FRAC_PI_2);
    }
}
