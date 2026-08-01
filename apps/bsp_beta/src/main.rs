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
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use bsp_beta::generation::{self, GenConfig, GenWorker};
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

struct GeneratedLaunch {
    tools_dir: PathBuf,
    package_root: PathBuf,
}

fn cleanup_generated_launch(generated: Option<&GeneratedLaunch>) {
    if let Some(generated) = generated {
        let _ = std::fs::remove_dir_all(&generated.package_root);
    }
}

fn app_import_mode(args: &cli::CliArgs) -> Result<package::ImportMode, AppError> {
    match args
        .require_import_mode()
        .map_err(|_| AppError::NoImportMode)?
    {
        cli::ImportMode::Strict => Ok(package::ImportMode::Strict),
        cli::ImportMode::Development => Ok(package::ImportMode::Development),
    }
}

/// Authorize a complete engine_pack V3 closure. Generated mode always names
/// the LIT companion explicitly; its absence is a package failure, not an
/// implicit fallback.
fn authorize_generated_package(
    package_dir: &std::path::Path,
    scale: f32,
) -> Result<package::AuthorizedBspImport, AppError> {
    let bsp = package_dir.join("bsp_beta_gen.bsp");
    let lit = package_dir.join("bsp_beta_gen.lit");
    let palette = package_dir.join("palette.lmp");
    let wad = package_dir.join("cc0_dungeon_v2.wad");
    let textures = package_dir.join("textures");
    for (label, path) in [
        ("BSP", &bsp),
        ("LIT", &lit),
        ("palette", &palette),
        ("WAD", &wad),
    ] {
        if !path.is_file() {
            return Err(AppError::BridgeProof(format!(
                "generated package is missing {label} at {}",
                path.display()
            )));
        }
    }
    if !textures.is_dir() {
        return Err(AppError::BridgeProof(format!(
            "generated package is missing texture closure at {}",
            textures.display()
        )));
    }
    package::authorize_direct_import(
        &bsp,
        &palette,
        Some(&lit),
        &[wad],
        Some(&textures),
        package::ImportMode::Strict,
        scale,
    )
    .map_err(AppError::PackageLoad)
}

fn build_initial_generated_import(
    args: &cli::CliArgs,
) -> Result<(package::AuthorizedBspImport, GeneratedLaunch), AppError> {
    let tools_dir = generation::discover_ericw_tools(args.ericw_tools_dir.as_deref())
        .map_err(|error| AppError::BridgeProof(error.to_string()))?
        .ok_or_else(|| {
            AppError::BridgeProof(
                "ericw-tools not found (use --ericw-tools, ERICW_TOOLS_DIR, HOME default, or PATH)"
                    .into(),
            )
        })?;
    let package_root = generation::create_unique_package_root().map_err(|error| {
        AppError::BridgeProof(format!("reserve generated package root: {error}"))
    })?;
    let startup = generation::startup_package_dir(&package_root);
    let config = GenConfig::default_config()
        .to_v3_config()
        .map_err(|error| AppError::BridgeProof(format!("default V3 config: {error}")))?;
    if let Err(error) = engine_pack::enhanced_dungeon_v3::build_v3_package_from_config(
        &config,
        &startup,
        Some(&tools_dir),
        "bsp_beta_gen",
        None,
    ) {
        let _ = std::fs::remove_dir_all(&package_root);
        return Err(AppError::BridgeProof(format!(
            "initial M3 package build: {error}"
        )));
    }
    match authorize_generated_package(&startup, args.scale) {
        Ok(import) => Ok((
            import,
            GeneratedLaunch {
                tools_dir,
                package_root,
            },
        )),
        Err(error) => {
            let _ = std::fs::remove_dir_all(&package_root);
            Err(error)
        }
    }
}

fn run() -> Result<(), AppError> {
    let args = cli::CliArgs::parse();
    let import_mode = if args.m3_generate {
        package::ImportMode::Strict
    } else {
        // Preserve the direct-launch error order: a missing source is reported
        // before the direct import-mode requirement.
        args.bsp_path.as_ref().ok_or(AppError::NoBspPath)?;
        app_import_mode(&args)?
    };
    let t_build = Instant::now();
    let (import, generated) = if args.m3_generate {
        let (import, launch) = build_initial_generated_import(&args)?;
        (import, Some(launch))
    } else {
        let bsp_path = args.bsp_path.as_ref().ok_or(AppError::NoBspPath)?;
        let palette = args
            .resolve_palette_path()
            .map_err(|error| AppError::BridgeProof(error.to_string()))?;
        let wad_paths: Vec<PathBuf> = args
            .resolve_wad_path()
            .map_err(|error| AppError::BridgeProof(error.to_string()))?
            .into_iter()
            .collect();
        let import = package::authorize_direct_import(
            bsp_path,
            &palette,
            args.resolve_lit_path().as_deref(),
            &wad_paths,
            args.textures_dir.as_deref(),
            import_mode,
            args.scale,
        )?;
        log::info!(
            "BSP authorized: {} ({} bytes)",
            bsp_path.display(),
            import.bsp.bytes.len()
        );
        (import, None)
    };
    log::info!("Effective import:\n{}", effective_import_summary(&import));

    let proof_world = import.world.clone();
    let mut coordinator = BspCoordinator::new();
    coordinator.register_bridge("physics", Box::new(PhysicsBridge::new()));
    coordinator.register_bridge("runtime", Box::new(RuntimeBridge::new()));
    let prepare = match coordinator.prepare_authorized_import(import) {
        Ok(prepare) => prepare,
        Err(error) => {
            cleanup_generated_launch(generated.as_ref());
            return Err(AppError::BspRuntime(error));
        }
    };
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
    let extracted = match coordinator.staged_extraction() {
        Some(extracted) => extracted,
        None => {
            rollback_staged_without_renderer(&mut coordinator);
            cleanup_generated_launch(generated.as_ref());
            return Err(AppError::BridgeProof(
                "authorized import did not stage extraction".into(),
            ));
        }
    };
    if let Err(error) = run_load_query_physics_behavior_proof(&proof_world, args.scale, extracted) {
        rollback_staged_without_renderer(&mut coordinator);
        cleanup_generated_launch(generated.as_ref());
        return Err(error);
    }

    let result = if args.mcp {
        run_mcp(&mut coordinator)
    } else if args.headless {
        run_headless(&args, &mut coordinator)
    } else if let Some(ref generated) = generated {
        run_m3_generate_windowed(&mut coordinator, generated, args.scale)
    } else {
        run_windowed(&mut coordinator)
    };
    cleanup_generated_launch(generated.as_ref());
    result
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
            entity_index,
            position,
            &mut physics_active_state.world,
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
    let event_loop = match EventLoop::new() {
        Ok(event_loop) => event_loop,
        Err(error) => {
            rollback_staged_without_renderer(coordinator);
            return Err(AppError::EventLoop(error));
        }
    };
    let window = match WindowBuilder::new()
        .with_title(APP_WINDOW_TITLE)
        .with_inner_size(winit::dpi::LogicalSize::new(1280, 720))
        .build(&event_loop)
    {
        Ok(window) => window,
        Err(error) => {
            rollback_staged_without_renderer(coordinator);
            return Err(AppError::Window(error));
        }
    };

    let config = RendererConfig {
        app_name: "bsp_beta".to_string(),
        window_width: 1280,
        window_height: 720,
        ..RendererConfig::default()
    };

    let mut renderer = match Renderer::new(config, &window) {
        Ok(renderer) => renderer,
        Err(error) => {
            rollback_staged_without_renderer(coordinator);
            return Err(AppError::RendererInit(error));
        }
    };
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

    let mount = match renderer.prepare_bsp_mount(extracted) {
        Ok(mount) => mount,
        Err(error) => {
            rollback_and_retire(coordinator, &mut renderer);
            return Err(AppError::Renderer(error));
        }
    };
    let token = bsp_runtime::BspGenerationToken {
        generation: coordinator.current_generation(),
    };
    if let Err(error) = coordinator.set_renderer_mount_ready(token, mount) {
        rollback_and_retire(coordinator, &mut renderer);
        return Err(AppError::BspRuntime(error));
    }
    if let Err(error) = coordinator.validate_for_scene(token, &mut scene) {
        rollback_and_retire(coordinator, &mut renderer);
        return Err(AppError::BspRuntime(error));
    }
    if let Err(error) = coordinator.commit(token, &mut scene) {
        rollback_and_retire(coordinator, &mut renderer);
        return Err(AppError::BspRuntime(error));
    }

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
                    if let Err(error) =
                        teardown_retire_and_reap(coordinator, &mut renderer, &mut scene)
                    {
                        log::error!("BSP teardown handoff failed: {error}");
                    }
                    elwt.exit();
                    return;
                }
            };

            match event {
                Event::WindowEvent { event, window_id } if window_id == window.id() => {
                    match event {
                        WindowEvent::CloseRequested => {
                            log::info!("Close requested, exiting");
                            if let Err(error) =
                                teardown_retire_and_reap(coordinator, &mut renderer, &mut scene)
                            {
                                log::error!("BSP teardown handoff failed: {error}");
                            }
                            elwt.exit();
                        }
                        WindowEvent::Resized(size) => {
                            if let Err(e) = renderer.resize(size.width, size.height) {
                                log::error!("Resize failed: {e}");
                                if let Err(error) =
                                    teardown_retire_and_reap(coordinator, &mut renderer, &mut scene)
                                {
                                    log::error!("BSP teardown handoff failed: {error}");
                                }
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
                                    if let Err(error) = teardown_retire_and_reap(
                                        coordinator,
                                        &mut renderer,
                                        &mut scene,
                                    ) {
                                        log::error!("BSP teardown handoff failed: {error}");
                                    }
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

// ─── m3-generate mode ────────────────────────────────────────────────────

/// App-owned data that must be captured while a new extraction is still staged.
/// `BspCoordinator::commit` consumes the candidate, so reading it afterward
/// would silently retain state from the old world (or nothing at all).
struct MountedAppState {
    spawn: Vec3,
    inline_model_infos: Vec<InlineModelInfo>,
    entity_classnames: std::collections::HashMap<u32, String>,
    entity_source_models: std::collections::HashMap<u32, String>,
}

fn capture_staged_app_state(
    extracted: &bsp::extract::ExtractedBsp,
    fallback: Vec3,
) -> MountedAppState {
    MountedAppState {
        spawn: bsp_player_start(extracted, fallback),
        inline_model_infos: extracted
            .inline_models
            .iter()
            .map(|model| InlineModelInfo {
                entity_index: model.entity_index,
                model_index: model.model_index,
                origin: [model.origin.x, model.origin.y, model.origin.z],
                angles: model.angle.map(|angle| [0.0, angle, 0.0]),
                scale: None,
                local_mins: [
                    model.local_bounds.0.x,
                    model.local_bounds.0.y,
                    model.local_bounds.0.z,
                ],
                local_maxs: [
                    model.local_bounds.1.x,
                    model.local_bounds.1.y,
                    model.local_bounds.1.z,
                ],
            })
            .collect(),
        entity_classnames: extracted
            .entity_descriptors
            .iter()
            .map(|entity| (entity.entity_index, entity.classname.clone()))
            .collect(),
        entity_source_models: extracted
            .entity_descriptors
            .iter()
            .filter_map(|entity| {
                entity
                    .model_ref
                    .map(|model| (entity.entity_index, format!("*{model}")))
            })
            .collect(),
    }
}

fn apply_mounted_app_state(loop_state: &mut AppLoopState, mounted: MountedAppState) {
    loop_state.camera = Camera::new(mounted.spawn);
    loop_state.fps_controller = FPSController::new(0.002, 1.0);
    loop_state.inline_model_infos = mounted.inline_model_infos;
    loop_state.entity_classnames = mounted.entity_classnames;
    loop_state.entity_source_models = mounted.entity_source_models;
    // Entity indices are generation-local. Do not let an old map's node IDs
    // receive snapshot updates after a successful replacement.
    loop_state.entity_node_map = EntityNodeMap::default();
}

/// Resolve a generated package into an authorized import, stage it, and publish
/// it atomically. Any failure before publication rolls back the candidate while
/// leaving the old active scene mount untouched.
fn commit_generated_package(
    package_dir: &std::path::Path,
    coordinator: &mut BspCoordinator,
    renderer: &mut Renderer,
    scene: &mut Scene,
    scale: f32,
) -> Result<MountedAppState, String> {
    let import =
        authorize_generated_package(package_dir, scale).map_err(|error| error.to_string())?;
    if let Err(error) = coordinator.prepare_authorized_import(import) {
        rollback_and_retire(coordinator, renderer);
        return Err(format!("prepare: {error}"));
    }
    mount_staged_candidate(coordinator, renderer, scene, Vec3::new(0.0, 2.0, 5.0))
}

fn mount_staged_candidate(
    coordinator: &mut BspCoordinator,
    renderer: &mut Renderer,
    scene: &mut Scene,
    fallback_spawn: Vec3,
) -> Result<MountedAppState, String> {
    let extracted = coordinator
        .staged_extraction()
        .ok_or_else(|| "no staged extraction".to_string())?;
    let app_state = capture_staged_app_state(extracted, fallback_spawn);
    let mount = match renderer.prepare_bsp_mount(extracted) {
        Ok(mount) => mount,
        Err(error) => {
            rollback_and_retire(coordinator, renderer);
            return Err(format!("renderer prepare_bsp_mount: {error}"));
        }
    };
    let token = bsp_runtime::BspGenerationToken {
        generation: coordinator.current_generation(),
    };
    if let Err(error) = coordinator.set_renderer_mount_ready(token, mount) {
        rollback_and_retire(coordinator, renderer);
        return Err(format!("set_renderer_mount_ready: {error}"));
    }
    if let Err(error) = coordinator.validate_for_scene(token, scene) {
        rollback_and_retire(coordinator, renderer);
        return Err(format!("validate_for_scene: {error}"));
    }
    if let Err(error) = coordinator.commit(token, scene) {
        rollback_and_retire(coordinator, renderer);
        return Err(format!("commit: {error}"));
    }
    Ok(app_state)
}

fn rollback_staged_without_renderer(coordinator: &mut BspCoordinator) {
    if let Err(error) = coordinator.rollback() {
        log::error!("generated mount rollback failed: {error}");
    }
}

fn rollback_and_retire(coordinator: &mut BspCoordinator, renderer: &mut Renderer) {
    if let Err(error) = coordinator.rollback() {
        log::error!("generated mount rollback failed: {error}");
    }
    if let Err(error) = drain_and_retire(coordinator, renderer) {
        log::error!("generated mount retirement handoff failed: {error}");
    }
}

/// Drain all pending retirements from the coordinator and submit each to the
/// renderer. On rejection, reconstruct and requeue.
fn drain_and_retire(
    coordinator: &mut BspCoordinator,
    renderer: &mut Renderer,
) -> Result<(), String> {
    let mut pending = coordinator.drain_pending_retirements().into_iter();
    while let Some(detached) = pending.next() {
        match renderer.retire_bsp_mount(detached) {
            Ok(_) => log::debug!("BSP mount retired"),
            Err(rejection) => {
                let reason = rejection.reason.clone();
                coordinator.requeue_retirement(rejection.into_detached());
                // The caller transferred custody of the remaining vector when
                // draining it, so return every unprocessed receipt as well.
                for remaining in pending {
                    coordinator.requeue_retirement(remaining);
                }
                return Err(format!("renderer rejected BSP retirement: {reason}"));
            }
        }
    }
    Ok(())
}

/// Teardown while Scene and Renderer are alive. A rejected receipt remains in
/// coordinator custody, and no follow-up drain is allowed to consume/drop it.
fn teardown_and_retire_all(
    coordinator: &mut BspCoordinator,
    renderer: &mut Renderer,
    scene: &mut Scene,
) -> Result<(), String> {
    coordinator.teardown(scene);
    drain_and_retire(coordinator, renderer)
}

/// Submit every detached receipt while Renderer and Scene remain live.
/// Normal frame acquisition reaps fence-complete records, and Renderer drop
/// performs the terminal reap after device idle.
fn teardown_retire_and_reap(
    coordinator: &mut BspCoordinator,
    renderer: &mut Renderer,
    scene: &mut Scene,
) -> Result<(), AppError> {
    teardown_and_retire_all(coordinator, renderer, scene).map_err(AppError::BridgeProof)
}

/// Windowed generated explorer. The startup package was already built,
/// strictly authorized, and staged by `run`; headless and MCP use the same
/// established runners rather than shadow copies of their proof behavior.
fn run_m3_generate_windowed(
    coordinator: &mut BspCoordinator,
    generated: &GeneratedLaunch,
    scale: f32,
) -> Result<(), AppError> {
    let mut gen_config = GenConfig::default_config();
    let event_loop = match EventLoop::new() {
        Ok(event_loop) => event_loop,
        Err(error) => {
            rollback_staged_without_renderer(coordinator);
            return Err(AppError::EventLoop(error));
        }
    };
    let window = match WindowBuilder::new()
        .with_title(format!(
            "BSP Beta — m3-generate | {}",
            gen_config.describe()
        ))
        .with_inner_size(winit::dpi::LogicalSize::new(1280, 720))
        .build(&event_loop)
    {
        Ok(window) => window,
        Err(error) => {
            rollback_staged_without_renderer(coordinator);
            return Err(AppError::Window(error));
        }
    };
    let mut renderer = match Renderer::new(
        RendererConfig {
            app_name: "bsp_beta".to_string(),
            window_width: 1280,
            window_height: 720,
            ..RendererConfig::default()
        },
        &window,
    ) {
        Ok(renderer) => renderer,
        Err(error) => {
            rollback_staged_without_renderer(coordinator);
            return Err(AppError::RendererInit(error));
        }
    };
    let mut scene = Scene::new();
    let initial = mount_staged_candidate(
        coordinator,
        &mut renderer,
        &mut scene,
        Vec3::new(0.0, 2.0, 5.0),
    )
    .map_err(AppError::BridgeProof)?;
    let mut loop_state = AppLoopState::new(Camera::new(initial.spawn), ModelMappings::default());
    apply_mounted_app_state(&mut loop_state, initial);
    install_app_fps_input(&mut loop_state.input);

    let worker = GenWorker::spawn();
    let tools_dir = generated.tools_dir.clone();
    let package_root = generated.package_root.clone();
    let last_request = std::sync::atomic::AtomicU64::new(0);
    let mut ctrl_held = false;
    let mut torn_down = false;
    window.request_redraw();

    event_loop.run(move |event, elwt| {
        elwt.set_control_flow(ControlFlow::Poll);
        let mut shutdown = |coordinator: &mut BspCoordinator,
                            renderer: &mut Renderer,
                            scene: &mut Scene| {
            if !torn_down {
                torn_down = true;
                if let Err(error) = teardown_retire_and_reap(coordinator, renderer, scene) {
                    log::error!("BSP teardown handoff failed: {error}");
                }
            }
        };
        if let Err(error) = engine::input::route_platform_input_to_app(
            &mut renderer, &window, &mut loop_state.input, &event,
        ) {
            log::error!("Platform input routing failed: {error}");
            shutdown(coordinator, &mut renderer, &mut scene);
            elwt.exit();
            return;
        }
        if let Event::WindowEvent { event: WindowEvent::ModifiersChanged(modifiers), .. } = &event {
            ctrl_held = modifiers.state().control_key();
        }
        let regen = match &event {
            Event::WindowEvent { event: WindowEvent::KeyboardInput { event, .. }, .. }
                if event.state.is_pressed() && !event.repeat => match event.physical_key {
                    winit::keyboard::PhysicalKey::Code(key) => handle_gen_hotkey(key, ctrl_held, &mut gen_config),
                    _ => false,
                },
            _ => false,
        };
        if regen {
            window.set_title(&format!("BSP Beta — m3-generate [pending] | {}", gen_config.describe()));
            let id = worker.enqueue(gen_config.clone(), tools_dir.clone(), &package_root);
            last_request.store(id, Ordering::Relaxed);
        }
        while let Some(result) = worker.poll_result() {
            let latest = last_request.load(Ordering::Relaxed);
            if result.id != latest {
                log::info!("discarding stale generation result {} (latest {})", result.id, latest);
                let _ = std::fs::remove_dir_all(&result.package_dir);
            } else if result.success {
                match commit_generated_package(&result.package_dir, coordinator, &mut renderer, &mut scene, scale) {
                    Ok(mounted) => {
                        apply_mounted_app_state(&mut loop_state, mounted);
                        window.set_title(&format!("BSP Beta — m3-generate | {}", result.config.describe()));
                    }
                    Err(error) => {
                        log::error!("generated replacement failed; previous world remains active: {error}");
                        window.set_title(&format!("BSP Beta — m3-generate [failed] | {}", result.config.describe()));
                        let _ = std::fs::remove_dir_all(&result.package_dir);
                    }
                }
            } else {
                log::error!("generation {} failed: {}", result.id, result.error.unwrap_or_else(|| "unknown failure".into()));
                window.set_title(&format!("BSP Beta — m3-generate [failed] | {}", result.config.describe()));
                let _ = std::fs::remove_dir_all(&result.package_dir);
            }
        }
        if let Err(error) = drain_and_retire(coordinator, &mut renderer) {
            log::error!("BSP retirement handoff failed: {error}");
        }
        match event {
            Event::WindowEvent { event, window_id } if window_id == window.id() => match event {
                WindowEvent::CloseRequested => {
                    shutdown(coordinator, &mut renderer, &mut scene);
                    elwt.exit();
                }
                WindowEvent::Resized(size) => if let Err(error) = renderer.resize(size.width, size.height) {
                    log::error!("Resize failed: {error}");
                    shutdown(coordinator, &mut renderer, &mut scene);
                    elwt.exit();
                },
                WindowEvent::RedrawRequested => {
                    let size = window.inner_size();
                    match render_app_frame(&mut renderer, &mut scene, &mut loop_state, size.width, size.height, false) {
                        Ok(FrameRenderOutcome::Rendered | FrameRenderOutcome::SkippedAcquireUnavailable |
                           FrameRenderOutcome::SkippedResizePending | FrameRenderOutcome::SubmittedNotPresented |
                           FrameRenderOutcome::PresentedSuboptimal) => window.request_redraw(),
                        Err(error) => {
                            log::error!("Render failed: {error}");
                            shutdown(coordinator, &mut renderer, &mut scene);
                            elwt.exit();
                        }
                    }
                }
                _ => {}
            },
            _ => {}
        }
    }).map_err(AppError::EventLoop)?;
    Ok(())
}

/// Handle a generation hotkey press. Returns true if regeneration is needed.
fn handle_gen_hotkey(
    physical_key: winit::keyboard::KeyCode,
    ctrl_held: bool,
    config: &mut GenConfig,
) -> bool {
    use winit::keyboard::KeyCode;

    match physical_key {
        KeyCode::F5 => {
            config.increment_seed();
            log::info!("F5: seed incremented to {} — regenerating", config.seed);
            true
        }
        KeyCode::F6 => {
            config.cycle_preset();
            log::info!(
                "F6: preset cycled to {} (extent={}) — regenerating",
                config.preset.tag(),
                config.extent
            );
            true
        }
        KeyCode::F7 => {
            config.chamfer = !config.chamfer;
            log::info!("F7: chamfer toggled to {} — regenerating", config.chamfer);
            true
        }
        KeyCode::F8 => {
            config.cycle_arch_type();
            log::info!(
                "F8: arch cycled to {} — regenerating",
                config.arch_type.tag()
            );
            true
        }
        KeyCode::F9 => {
            config.stairs = !config.stairs;
            log::info!("F9: stairs toggled to {} — regenerating", config.stairs);
            true
        }
        KeyCode::KeyR if ctrl_held => {
            log::info!(
                "Ctrl+R: regenerating with unchanged config: {}",
                config.describe()
            );
            true
        }
        _ => false,
    }
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

    let mount = match renderer.prepare_bsp_mount(extracted) {
        Ok(mount) => mount,
        Err(error) => {
            rollback_and_retire(coordinator, &mut renderer);
            return Err(AppError::Renderer(error));
        }
    };
    let mcp_map = mcp_map_data.map(|ext| mcp::McpMap::from_mount(ext, &mount, 0));
    let token = bsp_runtime::BspGenerationToken {
        generation: coordinator.current_generation(),
    };
    if let Err(error) = coordinator.set_renderer_mount_ready(token, mount) {
        rollback_and_retire(coordinator, &mut renderer);
        return Err(AppError::BspRuntime(error));
    }
    if let Err(error) = coordinator.validate_for_scene(token, &mut scene) {
        rollback_and_retire(coordinator, &mut renderer);
        return Err(AppError::BspRuntime(error));
    }
    if let Err(error) = coordinator.commit(token, &mut scene) {
        rollback_and_retire(coordinator, &mut renderer);
        return Err(AppError::BspRuntime(error));
    }

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

    log::info!(
        "Starting BSP beta headless mode ({}×{})",
        vp_width,
        vp_height
    );
    let HeadlessRuntime {
        mut renderer,
        mut scene,
        mut loop_state,
        ..
    } = match prepare_headless_runtime(coordinator, None, args) {
        Ok(runtime) => runtime,
        Err(error) => {
            if let Err(rollback) = coordinator.rollback() {
                log::error!("headless setup rollback failed: {rollback}");
            }
            return Err(error);
        }
    };

    let run_result = (|| -> Result<(), AppError> {
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
            let corpus = args
                .corpus_identity
                .clone()
                .unwrap_or_else(|| "bsp-beta-headless".to_string());
            let stats_key = renderer
                .request_bsp_frame_evidence(corpus, "headless-stats".to_string(), visibility)
                .map_err(|e| {
                    AppError::RendererInit(renderer::RendererError::InvalidState(format!(
                        "evidence request: {e}"
                    )))
                })?;

            // Render until evidence is sealed.
            for _attempt in 0..16 {
                render_app_frame(
                    &mut renderer,
                    &mut scene,
                    &mut loop_state,
                    vp_width,
                    vp_height,
                    true,
                )
                .map_err(AppError::Renderer)?;
                match renderer.take_bsp_frame_evidence(stats_key) {
                    renderer::api::bsp::BspEvidenceStatus::Sealed(report) => {
                        let serialized = serde_json::to_string_pretty(&report).map_err(|e| {
                            AppError::RendererInit(renderer::RendererError::InvalidState(format!(
                                "serialize report: {e}"
                            )))
                        })?;
                        println!("{serialized}");
                        if !report.eligible {
                            log::warn!("Stats report is not eligible for acceptance");
                            return Err(AppError::BridgeProof(
                                "stats report is not eligible for acceptance".into(),
                            ));
                        }
                        return Ok(());
                    }
                    renderer::api::bsp::BspEvidenceStatus::RejectedNoMount => {
                        log::error!("Stats request rejected: no active BSP mount");
                        return Err(AppError::BridgeProof(
                            "stats request rejected: no active BSP mount".into(),
                        ));
                    }
                    renderer::api::bsp::BspEvidenceStatus::MissingReport => {
                        log::error!("Stats report missing");
                        return Err(AppError::BridgeProof("stats report missing".into()));
                    }
                    renderer::api::bsp::BspEvidenceStatus::Pending => {
                        // Continue rendering
                    }
                }
            }
            log::error!("Stats report not ready after bounded frame attempts");
            return Err(AppError::BridgeProof(
                "stats report not ready after bounded frame attempts".into(),
            ));
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
                    render_app_frame(
                        &mut renderer,
                        &mut scene,
                        &mut loop_state,
                        vp_width,
                        vp_height,
                        true,
                    )
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
                render_app_frame(
                    &mut renderer,
                    &mut scene,
                    &mut loop_state,
                    vp_width,
                    vp_height,
                    true,
                )
                .map_err(AppError::Renderer)?;
                log::info!("Smoke frame {frame_num}/{smoke_frames} rendered");
            }
        }

        Ok(())
    })();
    let teardown = teardown_retire_and_reap(coordinator, &mut renderer, &mut scene);
    match (run_result, teardown) {
        (Err(error), Err(teardown_error)) => {
            log::error!("headless teardown after failure also failed: {teardown_error}");
            Err(error)
        }
        (Err(error), Ok(())) => Err(error),
        (Ok(()), Err(error)) => Err(error),
        (Ok(()), Ok(())) => {
            log::info!("BSP beta headless complete");
            Ok(())
        }
    }
}

fn run_mcp(coordinator: &mut BspCoordinator) -> Result<(), AppError> {
    log::info!("Starting BSP beta MCP mode (headless 1920×1080)");
    // Capture extraction before it gets consumed.
    let extracted_clone = match coordinator.staged_extraction() {
        Some(extracted) => extracted.clone(),
        None => {
            if let Err(rollback) = coordinator.rollback() {
                log::error!("MCP setup rollback failed: {rollback}");
            }
            return Err(AppError::BridgeProof("no staged extraction".to_string()));
        }
    };
    let _bsp_size = extracted_clone.face_geometries.len();

    let HeadlessRuntime {
        mut renderer,
        mut scene,
        mut loop_state,
        mcp_map,
    } = match prepare_headless_runtime(
        coordinator,
        Some(&extracted_clone),
        &cli::CliArgs::default(),
    ) {
        Ok(runtime) => runtime,
        Err(error) => {
            if let Err(rollback) = coordinator.rollback() {
                log::error!("MCP setup rollback failed: {rollback}");
            }
            return Err(error);
        }
    };

    let run_result = (|| -> Result<(), AppError> {
        let mcp_map = mcp_map
            .ok_or_else(|| AppError::BridgeProof("MCP map data was not retained".to_string()))?;

        // Prime GPU resources before accepting the first capture request.
        for _ in 0..5 {
            render_app_frame(&mut renderer, &mut scene, &mut loop_state, 1920, 1080, true)?;
        }

        log::info!("BSP mount published; serving MCP JSON-RPC on stdio");
        mcp::serve(&mut renderer, &mut scene, &mut loop_state, mcp_map)?;
        Ok(())
    })();
    let teardown = teardown_retire_and_reap(coordinator, &mut renderer, &mut scene);
    log::info!("MCP stdin closed; shutting down");
    match (run_result, teardown) {
        (Err(error), Err(teardown_error)) => {
            log::error!("MCP teardown after failure also failed: {teardown_error}");
            Err(error)
        }
        (Err(error), Ok(())) => Err(error),
        (Ok(()), Err(error)) => Err(error),
        (Ok(()), Ok(())) => Ok(()),
    }
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
    use super::{camera_angles_for_direction, handle_gen_hotkey, GenConfig};
    use glam::Vec3;

    fn assert_near(actual: f32, expected: f32) {
        assert!((actual - expected).abs() < 1.0e-5, "{actual} != {expected}");
    }

    #[test]
    fn generation_hotkeys_change_only_their_declared_setting() {
        use winit::keyboard::KeyCode;

        let mut config = GenConfig::default_config();
        let original = config.clone();
        assert!(handle_gen_hotkey(KeyCode::F5, false, &mut config));
        assert_eq!(config.seed, original.seed.wrapping_add(1));
        assert_eq!(config.preset, original.preset);

        assert!(handle_gen_hotkey(KeyCode::F6, false, &mut config));
        assert_ne!(config.preset, original.preset);
        assert!(handle_gen_hotkey(KeyCode::F7, false, &mut config));
        assert_ne!(config.chamfer, original.chamfer);
        assert!(handle_gen_hotkey(KeyCode::F8, false, &mut config));
        assert!(handle_gen_hotkey(KeyCode::F9, false, &mut config));

        let before_ctrl_r = config.clone();
        assert!(handle_gen_hotkey(KeyCode::KeyR, true, &mut config));
        assert_eq!(config, before_ctrl_r);
        assert!(!handle_gen_hotkey(KeyCode::KeyR, false, &mut config));
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
