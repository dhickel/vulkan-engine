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

use std::cell::RefCell;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use bsp_beta::generation::{self, GenConfig, GenWorker};
use bsp_beta::m3_gui::{Action as M3Action, GuiAction, GuiMode, M3Gui};
use bsp_beta::physics_bridge::PhysicsBridge;
use bsp_beta::player_navigation::{
    BspMovementWorld, BspPlayerMovementController, MovementInput, BSP_FIXED_DT,
};
use bsp_beta::richness_generation::{
    self, production_richness_executor, ExecutorOutcome, RichnessGenerationController,
};
use bsp_beta::richness_gui::{
    InheritedOr, RichnessCaveMode, RichnessDraft, RichnessFieldId, RichnessGui, RichnessGuiAction,
    RichnessGuiMode, RichnessInputAction, RichnessPreset, RichnessTheme,
};
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
    ActionId, ActionMap, InputActionEventEmitter, InputSystem, LayerDescriptor, LayerPriority,
};
use engine::render::camera_view_for_size;
use engine_events::DispatchReport;
use glam::Vec3;
use renderer::api::config::RendererConfig;
use renderer::api::{CaptureTarget, FrameCaptureRequest, FrameCaptureStatus, FrameRenderOutcome};
use renderer::{Renderer, Scene};
use thiserror::Error;
use winit::dpi::PhysicalPosition;
use winit::event::{ElementState, Event, MouseButton, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, ModifiersState, PhysicalKey};
use winit::window::{Window, WindowBuilder};

const APP_WINDOW_TITLE: &str = "BSP Beta — Phase 05";
const FIXED_DT: f32 = BSP_FIXED_DT;

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
    initial_config: GenConfig,
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

fn initial_m3_config(args: &cli::CliArgs) -> Result<GenConfig, AppError> {
    let mut config = GenConfig::default_config();
    config.seed = args.m3_seed;
    config.preset = args.m3_preset;
    config.extent = if args.m3_preset == bsp_generator::enhanced_v3::V3Preset::Rich {
        3072
    } else {
        2048
    };
    config.rooms = args.m3_rooms;
    config.corridors = args.m3_corridors;
    config.loops = args.m3_loops;
    config.chamfer = args.m3_chamfer;
    config.arch_type = args.m3_arch_type;
    config.grammar_families = args.m3_grammar_families.clone();
    config
        .to_v3_config()
        .map_err(|error| AppError::BridgeProof(format!("invalid M3 CLI config: {error}")))?;
    Ok(config)
}

/// Remove stale explorer package roots left behind by killed or crashed
/// processes. Roots are process-scoped (`bsp-beta-m3-<pid>-<nonce>-<seq>`),
/// so any root older than `max_age` whose owning PID is not alive is safe to
/// reap. This keeps the system temp dir from accumulating compiled packages
/// across runs, which can exhaust tmpfs quotas on small mounts.
pub fn sweep_stale_package_roots(max_age: std::time::Duration) {
    let Ok(entries) = std::fs::read_dir(std::env::temp_dir()) else {
        return;
    };
    let now = std::time::SystemTime::now();
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(name) = name.to_str() else {
            continue;
        };
        let Some(rest) = name.strip_prefix("bsp-beta-m3-") else {
            continue;
        };
        let Some(pid_str) = rest.split('-').next() else {
            continue;
        };
        // Skip roots whose owning process is still alive; they may be mid-build.
        if let Ok(pid) = pid_str.parse::<u32>() {
            if pid != 0 && std::path::Path::new(&format!("/proc/{pid}")).exists() {
                continue;
            }
        }
        if let Ok(metadata) = entry.metadata() {
            if let Ok(modified) = metadata.modified() {
                if now.duration_since(modified).unwrap_or_default() >= max_age {
                    let _ = std::fs::remove_dir_all(entry.path());
                }
            }
        }
    }
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
    let initial_config = initial_m3_config(args)?;
    // Reap stale package roots from killed/crashed runs before reserving a
    // fresh one; keeps the temp mount from accumulating compiled packages.
    sweep_stale_package_roots(std::time::Duration::from_secs(300));
    let package_root = generation::create_unique_package_root().map_err(|error| {
        AppError::BridgeProof(format!("reserve generated package root: {error}"))
    })?;
    let startup = generation::startup_package_dir(&package_root);
    let config = initial_config
        .to_v3_config()
        .map_err(|error| AppError::BridgeProof(format!("invalid M3 CLI config: {error}")))?;
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
                initial_config,
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
    // Richness V1 launches build their own package through the real
    // pipeline; they bypass the direct-BSP import machinery entirely.
    if args.m3_richness {
        let richness_args = parse_richness_cli_overrides();
        return run_richness_launch(&args, &richness_args, &tools_dir_for_richness(&args)?);
    }
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
    let movement_world = BspMovementWorld::from_bsp(&proof_world, args.scale)
        .map_err(|error| AppError::BridgeProof(format!("movement descriptors: {error}")))?;
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
        run_mcp(&mut coordinator, movement_world)
    } else if args.headless {
        run_headless(&args, &mut coordinator, movement_world)
    } else if let Some(ref generated) = generated {
        run_m3_generate_windowed(&mut coordinator, generated, args.scale, movement_world)
    } else {
        run_windowed(&mut coordinator, movement_world, &args)
    };
    cleanup_generated_launch(generated.as_ref());
    result
}

// ─── Richness V1 launch ──────────────────────────────────────────────────

/// Parse Richness CLI overrides from the original process args.
/// Only `--richness-*` flags are recognized; unknown args are ignored.
fn parse_richness_cli_overrides() -> cli::RichnessLaunchToken {
    let args: Vec<String> = std::env::args().collect();
    cli::parse_richness_launch_token(&args).unwrap_or_default()
}

/// Resolve ericw-tools directory for Richness generation.
fn tools_dir_for_richness(args: &cli::CliArgs) -> Result<std::path::PathBuf, AppError> {
    generation::discover_ericw_tools(args.ericw_tools_dir.as_deref())
        .map_err(|error| AppError::BridgeProof(error.to_string()))?
        .ok_or_else(|| {
            AppError::BridgeProof(
                "ericw-tools not found (use --ericw-tools, ERICW_TOOLS_DIR, HOME default, or PATH)"
                    .into(),
            )
        })
}

/// Apply CLI richness overrides to a fresh draft.
fn apply_richness_overrides(
    draft: &mut RichnessDraft,
    token: &cli::RichnessLaunchToken,
) -> Result<(), String> {
    if let Some(ref tag) = token.richness_preset {
        let p = RichnessPreset::from_tag(tag)
            .ok_or_else(|| format!("unknown richness preset '{tag}'"))?;
        draft.set_preset(p);
    }
    if let Some(ref tag) = token.richness_theme {
        let t = RichnessTheme::from_tag(tag)
            .ok_or_else(|| format!("unknown richness theme '{tag}'"))?;
        draft.set_theme(t);
    }
    if let Some(v) = token.richness_extent {
        draft.try_set_extent(v)?;
    }
    if let Some(v) = token.richness_seed {
        draft.set_seed(v);
    }
    // Apply inherited-or-explicit fields (inherited flags win over explicit values)
    if token.richness_landmarks_inherited {
        draft.try_set_inherited_u32(RichnessFieldId::Landmarks, InheritedOr::Inherited)?;
    } else if let Some(v) = token.richness_landmarks {
        draft.try_set_explicit_u32(RichnessFieldId::Landmarks, v)?;
    }
    if token.richness_zones_inherited {
        draft.try_set_inherited_u32(RichnessFieldId::Zones, InheritedOr::Inherited)?;
    } else if let Some(v) = token.richness_zones {
        draft.try_set_explicit_u32(RichnessFieldId::Zones, v)?;
    }
    if token.richness_cave_mode_inherited {
        draft.try_set_cave_mode(InheritedOr::Inherited)?;
    } else if let Some(ref tag) = token.richness_cave_mode {
        let m =
            RichnessCaveMode::from_tag(tag).ok_or_else(|| format!("unknown cave mode '{tag}'"))?;
        draft.try_set_cave_mode(InheritedOr::Explicit(m))?;
    }
    if token.richness_vertical_openings_inherited {
        draft.try_set_inherited_u32(RichnessFieldId::VerticalOpenings, InheritedOr::Inherited)?;
    } else if let Some(v) = token.richness_vertical_openings {
        draft.try_set_explicit_u32(RichnessFieldId::VerticalOpenings, v)?;
    }
    if token.richness_budget_ceiling_inherited {
        draft.try_set_inherited_u32(RichnessFieldId::BudgetCeiling, InheritedOr::Inherited)?;
    } else if let Some(v) = token.richness_budget_ceiling {
        draft.try_set_explicit_u32(RichnessFieldId::BudgetCeiling, v)?;
    }
    Ok(())
}

/// Build a richness package from a draft, returning the package directory.
fn build_richness_startup_package(
    draft: &RichnessDraft,
    tools_dir: &std::path::Path,
    package_dir: &std::path::Path,
) -> Result<(), String> {
    let doc = richness_generation::draft_to_richness_document(draft)?;
    engine_pack::enhanced_dungeon_v3_richness_v1::build_richness_v1_package(
        &doc,
        package_dir,
        Some(tools_dir),
        "bsp_beta_richness_startup",
        None,
    )
    .map_err(|e| format!("richness package build failed: {e}"))?;
    Ok(())
}

/// Authorize a richness package closure.
fn authorize_richness_package(
    package_dir: &std::path::Path,
    scale: f32,
) -> Result<package::AuthorizedBspImport, AppError> {
    let bsp = package_dir.join("bsp_beta_richness_startup.bsp");
    let lit = package_dir.join("bsp_beta_richness_startup.lit");
    let palette = package_dir.join("palette.lmp");
    // Find the WAD file in the package directory
    let wad = find_wad_in_dir(package_dir)
        .map_err(|e| AppError::BridgeProof(format!("no WAD in richness package: {e}")))?;
    let textures = package_dir.join("textures");
    for (label, path) in [
        ("BSP", &bsp),
        ("LIT", &lit),
        ("palette", &palette),
        ("WAD", &wad),
    ] {
        if !path.is_file() {
            return Err(AppError::BridgeProof(format!(
                "richness package is missing {label} at {}",
                path.display()
            )));
        }
    }
    package::authorize_direct_import(
        &bsp,
        &palette,
        if lit.is_file() { Some(&lit) } else { None },
        &[wad],
        if textures.is_dir() {
            Some(&textures)
        } else {
            None
        },
        package::ImportMode::Strict,
        scale,
    )
    .map_err(AppError::PackageLoad)
}

/// Locate a .wad file in a directory.
fn find_wad_in_dir(dir: &std::path::Path) -> Result<std::path::PathBuf, String> {
    for entry in std::fs::read_dir(dir).map_err(|e| e.to_string())? {
        let entry = entry.map_err(|e| e.to_string())?;
        let name = entry.file_name();
        if name.to_string_lossy().ends_with(".wad") {
            return Ok(entry.path());
        }
    }
    Err("no .wad file found".to_string())
}

/// Mount a staged richness candidate via the coordinator.
fn mount_richness_candidate(
    coordinator: &mut BspCoordinator,
    renderer: &mut Renderer,
    scene: &mut Scene,
    fallback_spawn: Vec3,
    movement_world: BspMovementWorld,
) -> Result<MountedAppState, String> {
    mount_staged_candidate(coordinator, renderer, scene, fallback_spawn, movement_world)
}

/// Top-level dispatch for Richness V1 mode.
fn run_richness_launch(
    args: &cli::CliArgs,
    richness_args: &cli::RichnessLaunchToken,
    tools_dir: &std::path::Path,
) -> Result<(), AppError> {
    // Build a startup draft from defaults + CLI overrides
    let mut startup_draft = RichnessDraft::new();
    if let Err(e) = apply_richness_overrides(&mut startup_draft, richness_args) {
        return Err(AppError::BridgeProof(format!(
            "invalid richness CLI overrides: {e}"
        )));
    }

    // Create package root
    sweep_stale_richness_package_roots(std::time::Duration::from_secs(300));
    let package_root = generation::create_unique_package_root().map_err(|error| {
        AppError::BridgeProof(format!("reserve richness package root: {error}"))
    })?;
    let startup_dir = package_root.join("startup");

    // Build the startup package
    if let Err(e) = build_richness_startup_package(&startup_draft, tools_dir, &startup_dir) {
        let _ = std::fs::remove_dir_all(&package_root);
        return Err(AppError::BridgeProof(e));
    }

    // Authorize and load
    let import = match authorize_richness_package(&startup_dir, args.scale) {
        Ok(import) => import,
        Err(e) => {
            let _ = std::fs::remove_dir_all(&package_root);
            return Err(e);
        }
    };

    let proof_world = import.world.clone();
    let movement_world = BspMovementWorld::from_bsp(&proof_world, args.scale)
        .map_err(|error| AppError::BridgeProof(format!("movement descriptors: {error}")))?;
    let mut coordinator = BspCoordinator::new();
    coordinator.register_bridge("physics", Box::new(PhysicsBridge::new()));
    coordinator.register_bridge("runtime", Box::new(RuntimeBridge::new()));

    if let Err(error) = coordinator.prepare_authorized_import(import) {
        let _ = std::fs::remove_dir_all(&package_root);
        return Err(AppError::BspRuntime(error));
    }

    log::info!("Richness package built and authorized");

    if args.headless {
        run_richness_headless(args, &mut coordinator, movement_world, &package_root)
    } else {
        run_richness_windowed(
            &mut coordinator,
            &package_root,
            tools_dir.to_path_buf(),
            args.scale,
            movement_world,
            startup_draft,
        )
    }
}

/// Sweep stale richness package roots.
fn sweep_stale_richness_package_roots(max_age: std::time::Duration) {
    let Ok(entries) = std::fs::read_dir(std::env::temp_dir()) else {
        return;
    };
    let now = std::time::SystemTime::now();
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(name) = name.to_str() else {
            continue;
        };
        let Some(rest) = name.strip_prefix("bsp-beta-richness-") else {
            continue;
        };
        let Some(pid_str) = rest.split('-').next() else {
            continue;
        };
        if let Ok(pid) = pid_str.parse::<u32>() {
            if pid != 0 && std::path::Path::new(&format!("/proc/{pid}")).exists() {
                continue;
            }
        }
        if let Ok(metadata) = entry.metadata() {
            if let Ok(modified) = metadata.modified() {
                if now.duration_since(modified).unwrap_or_default() >= max_age {
                    let _ = std::fs::remove_dir_all(entry.path());
                }
            }
        }
    }
}

/// Headless richness: mount the initial package and capture frames.
fn run_richness_headless(
    args: &cli::CliArgs,
    coordinator: &mut BspCoordinator,
    movement_world: BspMovementWorld,
    package_root: &std::path::Path,
) -> Result<(), AppError> {
    let is_acceptance = args.acceptance_camera.is_some();
    let (vp_width, vp_height) = if is_acceptance {
        (1280, 720)
    } else {
        (1920, 1080)
    };

    let config = RendererConfig {
        app_name: "bsp_beta_richness".to_string(),
        window_width: vp_width,
        window_height: vp_height,
        headless: true,
        ..RendererConfig::default()
    };
    let mut renderer = Renderer::new_headless(config).map_err(AppError::RendererInit)?;
    let mut scene = Scene::new();

    let extracted = coordinator
        .staged_extraction()
        .ok_or_else(|| AppError::BridgeProof("no staged extraction".to_string()))?;
    let player_start = bsp_player_start(extracted, Vec3::new(0.0, 2.0, 5.0));
    let headless_camera = bsp_headless_camera(player_start, extracted);

    let mount = match renderer.prepare_bsp_mount(extracted) {
        Ok(mount) => mount,
        Err(error) => {
            rollback_and_retire(coordinator, &mut renderer);
            let _ = std::fs::remove_dir_all(package_root);
            return Err(AppError::Renderer(error));
        }
    };
    let token = bsp_runtime::BspGenerationToken {
        generation: coordinator.current_generation(),
    };
    if let Err(error) = coordinator.set_renderer_mount_ready(token, mount) {
        rollback_and_retire(coordinator, &mut renderer);
        let _ = std::fs::remove_dir_all(package_root);
        return Err(AppError::BspRuntime(error));
    }
    if let Err(error) = coordinator.validate_for_scene(token, &mut scene) {
        rollback_and_retire(coordinator, &mut renderer);
        let _ = std::fs::remove_dir_all(package_root);
        return Err(AppError::BspRuntime(error));
    }
    if let Err(error) = coordinator.commit(token, &mut scene) {
        rollback_and_retire(coordinator, &mut renderer);
        let _ = std::fs::remove_dir_all(package_root);
        return Err(AppError::BspRuntime(error));
    }

    let mut loop_state =
        AppLoopState::new(headless_camera, ModelMappings::default(), movement_world);

    let run_result = (|| -> Result<(), AppError> {
        // Warmup
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

        if args.capture_frames > 0 {
            let capture_dir = std::path::PathBuf::from(format!(
                ".internal-dev/captures/bsp-beta/richness-headless-{}",
                std::process::id()
            ));
            std::fs::create_dir_all(&capture_dir).map_err(|e| {
                AppError::RendererInit(renderer::RendererError::InvalidState(format!(
                    "create capture dir: {e}"
                )))
            })?;

            for frame_num in 0..args.capture_frames {
                let png_path =
                    capture_dir.join(format!("bsp_beta_richness_frame_{frame_num:04}.png"));
                let sidecar_path = capture_dir.join(format!(
                    "bsp_beta_richness_frame_{frame_num:04}_sidecar.json"
                ));

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

                if png_path.is_file() {
                    log::info!("✓ Richness frame {frame_num}: {}", png_path.display());
                } else {
                    return Err(AppError::RendererInit(
                        renderer::RendererError::InvalidState(format!(
                            "frame {frame_num} capture failed"
                        )),
                    ));
                }
            }
        } else {
            for frame_num in 0..5u32 {
                render_app_frame(
                    &mut renderer,
                    &mut scene,
                    &mut loop_state,
                    vp_width,
                    vp_height,
                    true,
                )
                .map_err(AppError::Renderer)?;
                log::info!("Richness smoke frame {frame_num}/5 rendered");
            }
        }
        Ok(())
    })();

    let teardown = teardown_retire_and_reap(coordinator, &mut renderer, &mut scene);
    let _ = std::fs::remove_dir_all(package_root);
    match (run_result, teardown) {
        (Err(error), _) => Err(error),
        (Ok(()), Err(error)) => Err(error),
        (Ok(()), Ok(())) => {
            log::info!("Richness headless complete");
            Ok(())
        }
    }
}

/// Windowed Richness explorer with full RichnessGui integration.
fn run_richness_windowed(
    coordinator: &mut BspCoordinator,
    package_root: &std::path::Path,
    tools_dir: std::path::PathBuf,
    scale: f32,
    movement_world: BspMovementWorld,
    initial_draft: RichnessDraft,
) -> Result<(), AppError> {
    let event_loop = match EventLoop::new() {
        Ok(event_loop) => event_loop,
        Err(error) => {
            rollback_staged_without_renderer(coordinator);
            return Err(AppError::EventLoop(error));
        }
    };

    let window = match WindowBuilder::new()
        .with_title("BSP Beta — Richness V1 Explorer")
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
            app_name: "bsp_beta_richness".to_string(),
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

    let initial = mount_richness_candidate(
        coordinator,
        &mut renderer,
        &mut scene,
        Vec3::new(0.0, 2.0, 5.0),
        movement_world.clone(),
    )
    .map_err(AppError::BridgeProof)?;

    let mut loop_state = AppLoopState::new(
        Camera::new(initial.spawn),
        ModelMappings::default(),
        movement_world,
    );
    apply_mounted_app_state(&mut loop_state, initial);
    install_app_fps_input(&mut loop_state.input);

    // ── Richness GUI setup ──────────────────────────────────────────────
    let gui: Rc<RefCell<RichnessGui>> = Rc::new(RefCell::new({
        let mut g = RichnessGui::new();
        g.mode = RichnessGuiMode::None;
        g
    }));
    loop_state.gameplay_input_enabled = true;

    let controller = RichnessGenerationController::spawn_at_root(package_root.to_path_buf());
    let mut close_intent: Option<u64> = None;
    let mut registration = M3UiRegistration::default();
    let mut cursor_position: Option<(f32, f32)> = None;
    let mut torn_down = false;

    window.request_redraw();

    event_loop
        .run(move |event, elwt| {
            elwt.set_control_flow(ControlFlow::Poll);

            let mut shutdown =
                |coordinator: &mut BspCoordinator,
                 renderer: &mut Renderer,
                 scene: &mut Scene| {
                    if !torn_down {
                        torn_down = true;
                        if let Err(error) =
                            teardown_retire_and_reap(coordinator, renderer, scene)
                        {
                            log::error!("BSP teardown handoff failed: {error}");
                        }
                    }
                };

            // F3/F4 mode hotkeys (Richness GUI uses F3/F4 to avoid collision
            // with M3 GUI's F1/F2)
            let mode_control = match &event {
                Event::WindowEvent { window_id, .. } if *window_id == window.id() => {
                    richness_mode_hotkey(&event)
                }
                _ => None,
            };
            if let Some(target) = mode_control {
                cursor_position = None;
                if let Err(error) = apply_richness_mode_transition(
                    &mut renderer,
                    &window,
                    &gui,
                    &mut loop_state,
                    &mut registration,
                    target,
                ) {
                    log::error!("Richness mode transition failed: {error}");
                    set_richness_gui_status(&gui, error);
                }
            } else {
                let mode = match gui.try_borrow() {
                    Ok(g) => g.mode,
                    Err(_) => RichnessGuiMode::Keyboard,
                };
                let blocks_gameplay = richness_blocks_gameplay_input(
                    mode,
                    m3_input_class(&event),
                );

                if mode != RichnessGuiMode::None {
                    if let Event::WindowEvent {
                        event: window_event,
                        window_id,
                    } = &event
                    {
                        if *window_id == window.id()
                            && is_m3_cursor_focus_event(window_event)
                        {
                            if let Err(error) =
                                renderer.route_platform_input(&window, &event)
                            {
                                log::error!("Platform cursor routing failed: {error}");
                                shutdown(coordinator, &mut renderer, &mut scene);
                                elwt.exit();
                                return;
                            }
                            cursor_position = None;
                        }
                    }
                }

                match mode {
                    RichnessGuiMode::None if !blocks_gameplay => {
                        if let Err(error) =
                            engine::input::route_platform_input_to_app(
                                &mut renderer,
                                &window,
                                &mut loop_state.input,
                                &event,
                            )
                        {
                            log::error!("Platform input routing failed: {error}");
                            shutdown(coordinator, &mut renderer, &mut scene);
                            elwt.exit();
                            return;
                        }
                    }
                    RichnessGuiMode::None => {}
                    RichnessGuiMode::Keyboard => {
                        if let Event::WindowEvent {
                            event:
                                WindowEvent::KeyboardInput {
                                    event: key_event, ..
                                },
                            window_id,
                        } = &event
                        {
                            if *window_id == window.id() {
                                if let PhysicalKey::Code(key) = key_event.physical_key {
                                    let action = match key_event.state {
                                        ElementState::Pressed if !key_event.repeat => {
                                            RichnessInputAction::Press
                                        }
                                        ElementState::Pressed => {
                                            RichnessInputAction::Repeat
                                        }
                                        ElementState::Released => {
                                            RichnessInputAction::Release
                                        }
                                    };
                                    let result = gui
                                        .try_borrow_mut()
                                        .map(|mut g| {
                                            g.handle_keyboard_input(key, action)
                                        })
                                        .unwrap_or(RichnessGuiAction::None);
                                    handle_richness_gui_action(
                                        result,
                                        RichnessGuiMode::Keyboard,
                                        &mut renderer,
                                        &window,
                                        &gui,
                                        &mut loop_state,
                                        &mut registration,
                                        &controller,
                                        &tools_dir,
                                        &mut close_intent,
                                    );
                                }
                            }
                        }
                    }
                    RichnessGuiMode::Mouse => match &event {
                        Event::WindowEvent {
                            event: WindowEvent::CursorMoved { position, .. },
                            window_id,
                        } if *window_id == window.id() => {
                            cursor_position = Some(logical_cursor_from_physical(
                                *position,
                                window.scale_factor(),
                            ));
                        }
                        Event::WindowEvent {
                            event:
                                WindowEvent::CursorEntered { .. }
                                | WindowEvent::CursorLeft { .. },
                            window_id,
                        } if *window_id == window.id() => {
                            cursor_position = None;
                        }
                        Event::WindowEvent {
                            event:
                                WindowEvent::MouseInput {
                                    state, button, ..
                                },
                            window_id,
                        } if *window_id == window.id() => {
                            if let Some((x, y)) = cursor_position {
                                let action = if *state == ElementState::Pressed {
                                    RichnessInputAction::Press
                                } else {
                                    RichnessInputAction::Release
                                };
                                let result = gui
                                    .try_borrow_mut()
                                    .map(|mut g| {
                                        g.handle_mouse_input(
                                            x as i32, y as i32, *button, action,
                                        )
                                    })
                                    .unwrap_or(RichnessGuiAction::None);
                                handle_richness_gui_action(
                                    result,
                                    RichnessGuiMode::Mouse,
                                    &mut renderer,
                                    &window,
                                    &gui,
                                    &mut loop_state,
                                    &mut registration,
                                    &controller,
                                    &tools_dir,
                                    &mut close_intent,
                                );
                            }
                        }
                        Event::WindowEvent {
                            event: WindowEvent::MouseWheel { delta, .. },
                            window_id,
                        } if *window_id == window.id() => {
                            if let Ok(mut g) = gui.try_borrow_mut() {
                                g.scroll_by(scroll_delta_to_gui_lines(delta) as i32);
                            }
                        }
                        Event::WindowEvent {
                            event:
                                WindowEvent::KeyboardInput {
                                    event: key_event, ..
                                },
                            window_id,
                        } if *window_id == window.id()
                            && key_event.state == ElementState::Pressed
                            && !key_event.repeat
                            && matches!(
                                key_event.physical_key,
                                PhysicalKey::Code(KeyCode::Escape)
                            ) =>
                        {
                            cursor_position = None;
                            if let Err(error) = apply_richness_mode_transition(
                                &mut renderer,
                                &window,
                                &gui,
                                &mut loop_state,
                                &mut registration,
                                RichnessGuiMode::Mouse,
                            ) {
                                set_richness_gui_status(&gui, error);
                            }
                        }
                        _ => {}
                    },
                }
            }

            // Process completed controller results
            while let Some(outcome) = controller.poll_result() {
                let latest = controller.latest_submitted_id();
                if outcome.request_id() != latest {
                    log::info!(
                        "discarding stale richness result {} (latest {})",
                        outcome.request_id(),
                        latest
                    );
                    if let Some(dir) = outcome.package_dir() {
                        let _ = std::fs::remove_dir_all(dir);
                    }
                    if close_intent == Some(outcome.request_id()) {
                        close_intent = None;
                    }
                    continue;
                }

                match outcome {
                    ExecutorOutcome::PackageReady {
                        request_id,
                        package_dir,
                    } => {
                        match commit_generated_package(
                            &package_dir,
                            coordinator,
                            &mut renderer,
                            &mut scene,
                            scale,
                        ) {
                            Ok(mounted) => {
                                apply_mounted_app_state(&mut loop_state, mounted);
                                set_richness_gui_status(&gui, "Generated.");
                                let apply_close = close_intent == Some(request_id);
                                close_intent = None;
                                if apply_close {
                                    let current_mode = match gui.try_borrow() {
                                        Ok(g) => Some(g.mode),
                                        Err(_) => None,
                                    };
                                    if let Some(current_mode) = current_mode {
                                        if let Err(error) =
                                            apply_richness_mode_transition(
                                                &mut renderer,
                                                &window,
                                                &gui,
                                                &mut loop_state,
                                                &mut registration,
                                                current_mode,
                                            )
                                        {
                                            set_richness_gui_status(
                                                &gui,
                                                format!(
                                                    "Generated, but could not close menu: {error}"
                                                ),
                                            );
                                        }
                                    }
                                }
                            }
                            Err(error) => {
                                log::error!(
                                    "richness replacement failed; previous world remains active: {error}"
                                );
                                set_richness_gui_status(
                                    &gui,
                                    format!(
                                        "Generation failed; previous world remains active: {error}"
                                    ),
                                );
                                let _ = std::fs::remove_dir_all(&package_dir);
                                if close_intent == Some(request_id) {
                                    close_intent = None;
                                }
                            }
                        }
                    }
                    ExecutorOutcome::Failed {
                        request_id,
                        error_message,
                    } => {
                        log::error!("richness generation {} failed: {error_message}", request_id);
                        set_richness_gui_status(
                            &gui,
                            format!("Generation failed: {error_message}"),
                        );
                        if close_intent == Some(request_id) {
                            close_intent = None;
                        }
                    }
                }
            }

            // Drain retirements
            if let Err(error) = drain_and_retire(coordinator, &mut renderer) {
                log::error!("BSP retirement handoff failed: {error}");
            }

            // Window events
            match event {
                Event::WindowEvent { event, window_id } if window_id == window.id() => {
                    match event {
                        WindowEvent::CloseRequested => {
                            shutdown(coordinator, &mut renderer, &mut scene);
                            elwt.exit();
                        }
                        WindowEvent::Resized(size) => {
                            if let Err(error) =
                                renderer.resize(size.width, size.height)
                            {
                                log::error!("Resize failed: {error}");
                                shutdown(coordinator, &mut renderer, &mut scene);
                                elwt.exit();
                            }
                        }
                        WindowEvent::RedrawRequested => {
                            let size = window.inner_size();
                            match render_app_frame(
                                &mut renderer,
                                &mut scene,
                                &mut loop_state,
                                size.width,
                                size.height,
                                false,
                            ) {
                                Ok(
                                    FrameRenderOutcome::Rendered
                                    | FrameRenderOutcome::SkippedAcquireUnavailable
                                    | FrameRenderOutcome::SkippedResizePending
                                    | FrameRenderOutcome::SubmittedNotPresented
                                    | FrameRenderOutcome::PresentedSuboptimal,
                                ) => window.request_redraw(),
                                Err(error) => {
                                    log::error!("Render failed: {error}");
                                    shutdown(coordinator, &mut renderer, &mut scene);
                                    elwt.exit();
                                }
                            }
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

// ─── Richness GUI helpers ─────────────────────────────────────────────────

const RICHNESS_APP_UI_ID: &str = "bsp-beta-richness-gui";

fn richness_mode_hotkey(event: &Event<()>) -> Option<RichnessGuiMode> {
    let Event::WindowEvent {
        event: WindowEvent::KeyboardInput { event, .. },
        ..
    } = event
    else {
        return None;
    };
    match event.physical_key {
        PhysicalKey::Code(key) => (event.state == ElementState::Pressed && !event.repeat)
            .then(|| match key {
                KeyCode::F3 => Some(RichnessGuiMode::Keyboard),
                KeyCode::F4 => Some(RichnessGuiMode::Mouse),
                _ => None,
            })
            .flatten(),
        _ => None,
    }
}

fn richness_blocks_gameplay_input(mode: RichnessGuiMode, class: M3InputClass) -> bool {
    mode != RichnessGuiMode::None && class != M3InputClass::Lifecycle
}

fn set_richness_gui_status(gui: &Rc<RefCell<RichnessGui>>, status: impl Into<String>) {
    if let Ok(mut g) = gui.try_borrow_mut() {
        g.status = Some(status.into());
    }
}

fn apply_richness_mode_transition(
    renderer: &mut Renderer,
    window: &Window,
    gui: &Rc<RefCell<RichnessGui>>,
    loop_state: &mut AppLoopState,
    registration: &mut M3UiRegistration,
    target_mode: RichnessGuiMode,
) -> Result<(), String> {
    let mut gui_state = gui
        .try_borrow_mut()
        .map_err(|_| "Richness GUI is busy; retry the mode hotkey".to_string())?;

    let current = gui_state.mode;
    let next = if current == target_mode {
        RichnessGuiMode::None
    } else {
        target_mode
    };

    let registration_owned = registration.id.is_some();
    let renderer_has_app_ui = renderer.has_app_ui();

    if next != RichnessGuiMode::None && !registration_owned && renderer_has_app_ui {
        return Err("cannot open Richness GUI while another app UI is registered".into());
    }

    // Handle registration
    if next != RichnessGuiMode::None && !registration_owned {
        let gui_clone = gui.clone();
        let callback = Box::new(
            move |ui: &imgui::Ui, ctx: &renderer::prelude::DebugUiFrameContext| {
                if let Ok(mut g) = gui_clone.try_borrow_mut() {
                    g.render_imgui(ui, ctx);
                }
            },
        ) as renderer::api::AppUiCallback;
        let id = renderer
            .register_app_ui(RICHNESS_APP_UI_ID, callback)
            .map_err(|error| format!("could not register Richness GUI: {error}"))?;
        registration.id = Some(id);
    } else if next == RichnessGuiMode::None && registration_owned {
        if let Some(id) = registration.id.take() {
            if !renderer.unregister_app_ui(&id) {
                log::warn!("Richness GUI callback {id} was already absent while closing");
            }
        }
    }

    gui_state.mode = next;
    let was_none = current == RichnessGuiMode::None;
    let now_open = next != RichnessGuiMode::None;
    if was_none && now_open {
        queue_gameplay_releases(&mut loop_state.input);
    }
    loop_state.gameplay_input_enabled = next == RichnessGuiMode::None;

    drop(gui_state);
    if let Err(error) = renderer.refresh_cursor_capture(window) {
        log::error!("Richness cursor capture refresh failed: {error}");
    }
    Ok(())
}

fn handle_richness_gui_action(
    action: RichnessGuiAction,
    mode: RichnessGuiMode,
    renderer: &mut Renderer,
    window: &Window,
    gui: &Rc<RefCell<RichnessGui>>,
    loop_state: &mut AppLoopState,
    registration: &mut M3UiRegistration,
    controller: &RichnessGenerationController,
    tools_dir: &std::path::Path,
    close_intent: &mut Option<u64>,
) {
    match action {
        RichnessGuiAction::None => {}
        RichnessGuiAction::Close => {
            if let Err(error) = apply_richness_mode_transition(
                renderer,
                window,
                gui,
                loop_state,
                registration,
                mode,
            ) {
                set_richness_gui_status(gui, error);
            }
        }
        RichnessGuiAction::Generate(draft) => {
            let executor = production_richness_executor(tools_dir.to_path_buf());
            let id = controller.enqueue(draft, executor);
            controller.clear_close_intent();
            set_richness_gui_status(gui, "Generation pending.");
            log::info!("Enqueued richness generation request {id}");
        }
        RichnessGuiAction::ApplyAndClose(draft) => {
            let executor = production_richness_executor(tools_dir.to_path_buf());
            let id = controller.enqueue(draft, executor);
            controller.set_close_intent(id);
            *close_intent = Some(id);
            set_richness_gui_status(gui, "Generation pending (Apply & Close).");
            log::info!("Enqueued richness generation request {id} (Apply & Close)");
        }
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
fn bsp_explicit_camera(
    origin: (f32, f32, f32),
    look_at: (f32, f32, f32),
    extracted: &bsp::extract::ExtractedBsp,
) -> Result<Camera, AppError> {
    // CLI coordinates are Quake map units; the camera and the acceptance
    // contents check live in engine space (extracted entity origins are
    // already engine-space, so convert explicitly here).
    let eye = extracted.transform.position(origin.0, origin.1, origin.2);
    let target = extracted
        .transform
        .position(look_at.0, look_at.1, look_at.2);
    if !eye.is_finite() || !target.is_finite() {
        return Err(AppError::BridgeProof(format!(
            "explicit camera has non-finite data: origin={origin:?} look_at={look_at:?}"
        )));
    }
    let contents = acceptance_point_contents(eye, extracted);
    if contents.is_solid() {
        return Err(AppError::BridgeProof(format!(
            "explicit camera eye lies in solid space: {origin:?}"
        )));
    }
    let direction = target - eye;
    let distance = direction.length();
    if distance <= f32::EPSILON {
        return Err(AppError::BridgeProof(format!(
            "explicit camera origin equals look-at: {origin:?}"
        )));
    }
    let (yaw, pitch) = camera_angles_for_direction(direction / distance);
    let mut camera = Camera::new(eye);
    camera.update_rotation(yaw, pitch);
    log::info!(
        "Explicit acceptance camera: origin={origin:?} look_at={look_at:?}, eye={eye:?}, contents={contents:?}, yaw={yaw:.3}, pitch={pitch:.3}",
    );
    Ok(camera)
}

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

fn run_windowed(
    coordinator: &mut BspCoordinator,
    movement_world: BspMovementWorld,
    args: &cli::CliArgs,
) -> Result<(), AppError> {
    let lifecycle_test = args.wsi_lifecycle_test;
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
    let mut loop_state =
        AppLoopState::new(Camera::new(player_start), model_mappings, movement_world);
    loop_state.inline_model_infos = inline_model_infos;
    loop_state.entity_classnames = entity_classnames;
    loop_state.entity_source_models = entity_source_models;
    install_app_fps_input(&mut loop_state.input);

    log::info!("BSP beta windowed mode initialized, starting event loop");
    window.request_redraw();

    // ── Phase 17 WSI lifecycle test ────────────────────────────────────
    // Resize -> minimize -> restore, driven from inside the app because
    // scriptable WM control (xdotool/wmctrl) is absent in the validation
    // environment. Exercises WindowEvent::Resized handling, swapchain
    // recreation, and minimize/restore without panic or ERROR logs.
    let mut lifecycle_renders: u32 = 0;
    let mut lifecycle_phase: u8 = 0;

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
                            if lifecycle_test {
                                log::info!(
                                    "WSI lifecycle: Resized event {}x{}",
                                    size.width,
                                    size.height
                                );
                                if lifecycle_phase == 1 {
                                    log::info!(
                                        "WSI lifecycle: resize applied {}x{}",
                                        size.width,
                                        size.height
                                    );
                                    window.set_minimized(true);
                                    lifecycle_phase = 2;
                                }
                            }
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

                            // Phase 17 WSI lifecycle sequencing (only in
                            // --wsi-lifecycle-test mode).
                            if lifecycle_test {
                                lifecycle_renders += 1;
                                let current = window.inner_size();
                                match lifecycle_phase {
                                    0 if lifecycle_renders >= 3 => {
                                        log::info!(
                                            "WSI lifecycle: requesting resize to 800x600 (current {}x{})",
                                            current.width,
                                            current.height
                                        );
                                        window.request_inner_size(winit::dpi::LogicalSize::new(
                                            800.0, 600.0,
                                        ));
                                        lifecycle_phase = 1;
                                    }
                                    1 if (current.width, current.height) != (1280, 720) => {
                                        log::info!(
                                            "WSI lifecycle: resize observed {}x{}",
                                            current.width,
                                            current.height
                                        );
                                        window.set_minimized(true);
                                        lifecycle_phase = 2;
                                    }
                                    2 if lifecycle_renders >= 6 => {
                                        log::info!("WSI lifecycle: restoring window");
                                        window.set_minimized(false);
                                        lifecycle_phase = 3;
                                    }
                                    3 if lifecycle_renders >= 9 => {
                                        log::info!(
                                            "WSI lifecycle test PASS: resize + minimize + restore rendered"
                                        );
                                        if let Err(error) = teardown_retire_and_reap(
                                            coordinator,
                                            &mut renderer,
                                            &mut scene,
                                        ) {
                                            log::error!("BSP teardown handoff failed: {error}");
                                        }
                                        elwt.exit();
                                    }
                                    _ => {}
                                }
                            }
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
    movement_world: BspMovementWorld,
    inline_model_infos: Vec<InlineModelInfo>,
    entity_classnames: std::collections::HashMap<u32, String>,
    entity_source_models: std::collections::HashMap<u32, String>,
}

fn capture_staged_app_state(
    extracted: &bsp::extract::ExtractedBsp,
    fallback: Vec3,
    movement_world: BspMovementWorld,
) -> MountedAppState {
    MountedAppState {
        spawn: bsp_player_start(extracted, fallback),
        movement_world,
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
    loop_state
        .movement_controller
        .reset_for_regeneration(mounted.spawn, mounted.movement_world);
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
    let movement_world = BspMovementWorld::from_bsp(&import.world, scale)
        .map_err(|error| format!("movement descriptors: {error}"))?;
    if let Err(error) = coordinator.prepare_authorized_import(import) {
        rollback_and_retire(coordinator, renderer);
        return Err(format!("prepare: {error}"));
    }
    mount_staged_candidate(
        coordinator,
        renderer,
        scene,
        Vec3::new(0.0, 2.0, 5.0),
        movement_world,
    )
}

fn mount_staged_candidate(
    coordinator: &mut BspCoordinator,
    renderer: &mut Renderer,
    scene: &mut Scene,
    fallback_spawn: Vec3,
    movement_world: BspMovementWorld,
) -> Result<MountedAppState, String> {
    let extracted = coordinator
        .staged_extraction()
        .ok_or_else(|| "no staged extraction".to_string())?;
    let app_state = capture_staged_app_state(extracted, fallback_spawn, movement_world);
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

// ─── m3-generate input routing & helpers ───────────────────────────────

const M3_APP_UI_ID: &str = "bsp-beta-m3-gui";
const GENERATED_TOAST_SECS: f32 = 2.0;

/// Queue synthetic release events for gameplay bindings so held keys and
/// accumulated mouse-look do not leak into gameplay when a menu opens.
fn queue_gameplay_releases(input: &mut InputSystem) {
    for code in [
        KeyCode::KeyW,
        KeyCode::KeyS,
        KeyCode::KeyA,
        KeyCode::KeyD,
        KeyCode::Space,
        KeyCode::ShiftLeft,
    ] {
        input.queue_event(engine::input::InputEvent::Key {
            code,
            state: ElementState::Released,
            repeat: false,
            modifiers: ModifiersState::empty(),
        });
    }
    // Release common mouse buttons too
    for button in [MouseButton::Left, MouseButton::Right, MouseButton::Middle] {
        input.queue_event(engine::input::InputEvent::MouseButton {
            button,
            state: ElementState::Released,
            modifiers: ModifiersState::empty(),
        });
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum M3RegistrationChange {
    None,
    Register,
    UnregisterOwned,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct M3Transition {
    next: GuiMode,
    registration: M3RegistrationChange,
    queue_releases: bool,
}

/// Production transition planner. Opening refuses a foreign app UI, mode
/// switches retain the owned callback, and close repairs local state even if
/// the callback was unexpectedly removed.
fn plan_m3_transition(
    current: GuiMode,
    target: GuiMode,
    registration_owned: bool,
    renderer_has_app_ui: bool,
) -> Result<M3Transition, String> {
    let next = if current == target {
        GuiMode::None
    } else {
        target
    };
    let registration = match (
        next != GuiMode::None,
        registration_owned,
        renderer_has_app_ui,
    ) {
        (false, true, _) => M3RegistrationChange::UnregisterOwned,
        (false, false, _) => M3RegistrationChange::None,
        (true, true, true) => M3RegistrationChange::None,
        (true, _, false) => M3RegistrationChange::Register,
        (true, false, true) => {
            return Err("cannot open M3 GUI while another app UI is registered".into())
        }
    };
    Ok(M3Transition {
        next,
        registration,
        queue_releases: current == GuiMode::None && next != GuiMode::None,
    })
}

fn commit_m3_transition_state(
    transition: M3Transition,
    mode: &mut GuiMode,
    input: &mut InputSystem,
    gameplay_input_enabled: &mut bool,
) {
    *mode = transition.next;
    if transition.queue_releases {
        queue_gameplay_releases(input);
    }
    *gameplay_input_enabled = transition.next == GuiMode::None;
}

#[derive(Default)]
struct M3UiRegistration {
    id: Option<renderer::api::DebugViewId>,
}

fn set_gui_status(gui: &Rc<RefCell<M3Gui>>, status: impl Into<String>) {
    if let Ok(mut gui) = gui.try_borrow_mut() {
        gui.status = Some(status.into());
    }
}

/// Apply a planned menu transition atomically. A failed open changes neither
/// mode nor gameplay gate and never removes another app's UI callback.
fn apply_m3_mode_transition(
    renderer: &mut Renderer,
    window: &Window,
    gui: &Rc<RefCell<M3Gui>>,
    loop_state: &mut AppLoopState,
    registration: &mut M3UiRegistration,
    target_mode: GuiMode,
) -> Result<(), String> {
    // Keep this guard through the renderer mutation. Once registration changes,
    // the remaining GUI/input state commit is infallible and cannot be blocked
    // by a RefCell borrow that would leave the two sides inconsistent.
    let mut gui_state = gui
        .try_borrow_mut()
        .map_err(|_| "M3 GUI is busy rendering; retry the mode hotkey".to_string())?;
    let transition = plan_m3_transition(
        gui_state.mode,
        target_mode,
        registration.id.is_some(),
        renderer.has_app_ui(),
    )?;

    enum PreparedRegistrationMutation {
        None,
        Register(renderer::api::AppUiCallback),
        UnregisterOwned(renderer::api::DebugViewId),
    }
    let registration_mutation = match transition.registration {
        M3RegistrationChange::None => PreparedRegistrationMutation::None,
        M3RegistrationChange::Register => {
            let gui_clone = gui.clone();
            let callback = Box::new(
                move |ui: &imgui::Ui, ctx: &renderer::prelude::DebugUiFrameContext| {
                    if let Ok(mut gui) = gui_clone.try_borrow_mut() {
                        gui.render_imgui(ui, ctx);
                    }
                },
            ) as renderer::api::AppUiCallback;
            PreparedRegistrationMutation::Register(callback)
        }
        M3RegistrationChange::UnregisterOwned => PreparedRegistrationMutation::UnregisterOwned(
            registration
                .id
                .clone()
                .ok_or_else(|| "M3 GUI registration ownership was lost".to_string())?,
        ),
    };

    match registration_mutation {
        PreparedRegistrationMutation::None => {}
        PreparedRegistrationMutation::Register(callback) => {
            let id = renderer
                .register_app_ui(M3_APP_UI_ID, callback)
                .map_err(|error| format!("could not register M3 GUI: {error}"))?;
            registration.id = Some(id);
        }
        PreparedRegistrationMutation::UnregisterOwned(id) => {
            if !renderer.unregister_app_ui(&id) {
                log::warn!("M3 GUI callback {id} was already absent while closing");
            }
            registration.id = None;
        }
    }

    commit_m3_transition_state(
        transition,
        &mut gui_state.mode,
        &mut loop_state.input,
        &mut loop_state.gameplay_input_enabled,
    );
    drop(gui_state);
    if let Err(error) = renderer.refresh_cursor_capture(window) {
        log::error!("M3 cursor capture refresh failed: {error}");
    }
    Ok(())
}

/// Normalize and validate the exact immutable snapshot sent to the worker.
fn normalized_generation_snapshot(mut config: GenConfig) -> Result<GenConfig, String> {
    config.normalize();
    config
        .to_v3_config()
        .map_err(|error| format!("invalid generation request: {error}"))?;
    Ok(config)
}

fn update_close_intent(intent: &mut Option<u64>, request_id: u64, apply_and_close: bool) {
    *intent = apply_and_close.then_some(request_id);
}

/// Enqueue a validated full-config snapshot and update latest-wins state.
fn enqueue_m3_generation(
    config: GenConfig,
    is_apply_close: bool,
    worker: &GenWorker,
    tools_dir: &std::path::Path,
    package_root: &std::path::Path,
    last_request: &std::sync::atomic::AtomicU64,
    close_intent: &mut Option<u64>,
    window: &Window,
) -> Result<GenConfig, String> {
    let config = normalized_generation_snapshot(config)?;
    window.set_title(&format!(
        "BSP Beta — m3-generate [pending] | {}",
        config.describe()
    ));
    let id = worker.enqueue(config.clone(), tools_dir.to_path_buf(), package_root);
    last_request.store(id, Ordering::Relaxed);
    update_close_intent(close_intent, id, is_apply_close);
    Ok(config)
}

/// Handle a generation hotkey press (F5–F9, Ctrl+R). Mutates the shared GUI
/// config. Returns true if regeneration should be enqueued.
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
            // An explicit previous vertical-edge override must agree with the
            // stairs toggle before the worker receives its snapshot.
            if !config.stairs {
                config.vertical_edges = Some(0);
            }
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

/// Process a [`GuiAction`] into a validated asynchronous request.
fn process_m3_gui_action(
    action: GuiAction,
    worker: &GenWorker,
    tools_dir: &std::path::Path,
    package_root: &std::path::Path,
    last_request: &std::sync::atomic::AtomicU64,
    close_intent: &mut Option<u64>,
    window: &Window,
) -> Result<Option<GenConfig>, String> {
    match action {
        GuiAction::Generate(config) => enqueue_m3_generation(
            config,
            false,
            worker,
            tools_dir,
            package_root,
            last_request,
            close_intent,
            window,
        )
        .map(Some),
        GuiAction::ApplyAndClose(config) => enqueue_m3_generation(
            config,
            true,
            worker,
            tools_dir,
            package_root,
            last_request,
            close_intent,
            window,
        )
        .map(Some),
        GuiAction::Close | GuiAction::None => Ok(None),
    }
}

/// Convert physical winit coordinates into the logical imgui coordinate space.
/// Winit scale factors may be below one, so only reject invalid values.
fn logical_scale_factor(scale_factor: f64) -> f64 {
    if scale_factor.is_finite() && scale_factor > 0.0 {
        scale_factor
    } else {
        1.0
    }
}

fn logical_cursor_from_physical(pos: PhysicalPosition<f64>, scale_factor: f64) -> (f32, f32) {
    let scale = logical_scale_factor(scale_factor) as f32;
    (pos.x as f32 / scale, pos.y as f32 / scale)
}

fn logical_viewport_from_physical(width: u32, height: u32, scale_factor: f64) -> (u32, u32) {
    let scale = logical_scale_factor(scale_factor);
    (
        (width as f64 / scale).round() as u32,
        (height as f64 / scale).round() as u32,
    )
}

/// Winit reports wheel-up as positive Y, while increasing GUI scroll moves
/// toward later content. Negate it so wheel-up moves toward the panel top.
fn scroll_delta_to_gui_lines(delta: &winit::event::MouseScrollDelta) -> f32 {
    match delta {
        winit::event::MouseScrollDelta::LineDelta(_, y) => -*y,
        winit::event::MouseScrollDelta::PixelDelta(pos) => -(pos.y as f32 / 120.0),
    }
}

fn m3_mode_hotkey(event: &Event<()>) -> Option<GuiMode> {
    let Event::WindowEvent {
        event: WindowEvent::KeyboardInput { event, .. },
        ..
    } = event
    else {
        return None;
    };
    match event.physical_key {
        PhysicalKey::Code(key) => mode_hotkey_from_key(key, event.state, event.repeat),
        _ => None,
    }
}

fn mode_hotkey_from_key(key: KeyCode, state: ElementState, repeat: bool) -> Option<GuiMode> {
    (state == ElementState::Pressed && !repeat)
        .then(|| match key {
            KeyCode::F1 => Some(GuiMode::Keyboard),
            KeyCode::F2 => Some(GuiMode::Mouse),
            _ => None,
        })
        .flatten()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum M3InputClass {
    Keyboard,
    Mouse,
    RawMouse,
    Lifecycle,
}

fn m3_input_class(event: &Event<()>) -> M3InputClass {
    match event {
        Event::WindowEvent {
            event: WindowEvent::KeyboardInput { .. },
            ..
        } => M3InputClass::Keyboard,
        Event::WindowEvent {
            event:
                WindowEvent::CursorMoved { .. }
                | WindowEvent::MouseInput { .. }
                | WindowEvent::MouseWheel { .. }
                | WindowEvent::CursorEntered { .. }
                | WindowEvent::CursorLeft { .. },
            ..
        } => M3InputClass::Mouse,
        Event::DeviceEvent { .. } => M3InputClass::RawMouse,
        _ => M3InputClass::Lifecycle,
    }
}

/// Open M3 modes never permit keyboard, pointer, or raw mouse input through
/// the gameplay route. Lifecycle events intentionally remain available.
fn m3_blocks_gameplay_input(mode: GuiMode, class: M3InputClass) -> bool {
    mode != GuiMode::None && class != M3InputClass::Lifecycle
}

/// Cursor enter/leave are not gameplay input, but renderer cursor policy must
/// see them (notably to preserve Wayland pointer-constraint ownership).
fn is_m3_cursor_focus_event(event: &WindowEvent) -> bool {
    matches!(
        event,
        WindowEvent::CursorEntered { .. } | WindowEvent::CursorLeft { .. }
    )
}

fn initial_physical_key(event: &Event<()>) -> Option<KeyCode> {
    let Event::WindowEvent {
        event: WindowEvent::KeyboardInput { event, .. },
        ..
    } = event
    else {
        return None;
    };
    (event.state == ElementState::Pressed && !event.repeat)
        .then(|| match event.physical_key {
            PhysicalKey::Code(key) => Some(key),
            _ => None,
        })
        .flatten()
}

/// Build a validated hotkey snapshot without mutating the shared GUI draft.
/// The draft is replaced only after the request was accepted by the worker.
fn hotkey_generation_snapshot(
    key: KeyCode,
    ctrl_held: bool,
    draft: &GenConfig,
) -> Result<Option<GenConfig>, String> {
    let mut snapshot = draft.clone();
    if !handle_gen_hotkey(key, ctrl_held, &mut snapshot) {
        return Ok(None);
    }
    normalized_generation_snapshot(snapshot).map(Some)
}

fn handle_m3_action(
    action: GuiAction,
    mode: GuiMode,
    renderer: &mut Renderer,
    window: &Window,
    gui: &Rc<RefCell<M3Gui>>,
    loop_state: &mut AppLoopState,
    registration: &mut M3UiRegistration,
    worker: &GenWorker,
    tools_dir: &std::path::Path,
    package_root: &std::path::Path,
    last_request: &std::sync::atomic::AtomicU64,
    close_intent: &mut Option<u64>,
    last_title_desc: &mut String,
) {
    if action == GuiAction::Close {
        if let Err(error) =
            apply_m3_mode_transition(renderer, window, gui, loop_state, registration, mode)
        {
            set_gui_status(gui, error);
        }
    } else if action != GuiAction::None {
        match process_m3_gui_action(
            action,
            worker,
            tools_dir,
            package_root,
            last_request,
            close_intent,
            window,
        ) {
            Ok(Some(config)) => {
                *last_title_desc = config.describe();
                set_gui_status(gui, "Generation pending.");
            }
            Ok(None) => {}
            Err(error) => set_gui_status(gui, error),
        }
    }
}

/// Windowed generated explorer with full M3Gui integration.
/// The startup package was already built, strictly authorized, and staged by
/// `run`; headless and MCP use the same established runners.
fn run_m3_generate_windowed(
    coordinator: &mut BspCoordinator,
    generated: &GeneratedLaunch,
    scale: f32,
    movement_world: BspMovementWorld,
) -> Result<(), AppError> {
    let event_loop = match EventLoop::new() {
        Ok(event_loop) => event_loop,
        Err(error) => {
            rollback_staged_without_renderer(coordinator);
            return Err(AppError::EventLoop(error));
        }
    };
    let initial_cfg = generated.initial_config.clone();
    let window = match WindowBuilder::new()
        .with_title(format!(
            "BSP Beta — m3-generate | {}",
            initial_cfg.describe()
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
        movement_world.clone(),
    )
    .map_err(AppError::BridgeProof)?;
    let mut loop_state = AppLoopState::new(
        Camera::new(initial.spawn),
        ModelMappings::default(),
        movement_world,
    );
    apply_mounted_app_state(&mut loop_state, initial);
    install_app_fps_input(&mut loop_state.input);

    // ── M3 GUI setup ──────────────────────────────────────────────────
    let mut gui_state = M3Gui::new();
    gui_state.config = initial_cfg.clone();
    let gui: Rc<RefCell<M3Gui>> = Rc::new(RefCell::new(gui_state));
    loop_state.gameplay_input_enabled = true;

    let worker = GenWorker::spawn();
    let tools_dir = generated.tools_dir.clone();
    let package_root = generated.package_root.clone();
    let last_request = std::sync::atomic::AtomicU64::new(0);
    let mut close_intent = None;
    let mut registration = M3UiRegistration::default();
    let mut cursor_position = None;
    let mut ctrl_held = false;
    let mut torn_down = false;
    let mut title_toast_deadline = None;
    let mut last_title_desc = initial_cfg.describe();

    window.request_redraw();

    event_loop
        .run(move |event, elwt| {
            elwt.set_control_flow(ControlFlow::Poll);

            let mut shutdown =
                |coordinator: &mut BspCoordinator, renderer: &mut Renderer, scene: &mut Scene| {
                    if !torn_down {
                        torn_down = true;
                        if let Err(error) = teardown_retire_and_reap(coordinator, renderer, scene) {
                            log::error!("BSP teardown handoff failed: {error}");
                        }
                    }
                };

            // F1/F2 are globally intercepted before renderer routing. Do not
            // return here: close/resize/redraw still need their lifecycle pass.
            let mode_control = match &event {
                Event::WindowEvent { window_id, .. } if *window_id == window.id() => {
                    m3_mode_hotkey(&event)
                }
                _ => None,
            };
            if let Some(target) = mode_control {
                // A position from before a mode transition cannot authorize a
                // click in the newly opened/switching mouse menu.
                cursor_position = None;
                if let Err(error) = apply_m3_mode_transition(
                    &mut renderer,
                    &window,
                    &gui,
                    &mut loop_state,
                    &mut registration,
                    target,
                ) {
                    log::error!("M3 mode transition failed: {error}");
                    set_gui_status(&gui, error);
                }
            } else {
                if let Event::WindowEvent {
                    event: WindowEvent::ModifiersChanged(modifiers),
                    window_id,
                } = &event
                {
                    if *window_id == window.id() {
                        ctrl_held = modifiers.state().control_key();
                    }
                }

                let mode = match gui.try_borrow() {
                    Ok(gui) => gui.mode,
                    Err(_) => {
                        // RefCell contention must fail closed: treating an
                        // unknown mode as None would leak this event to gameplay.
                        log::warn!("M3 GUI mode was busy; suppressing input for this event");
                        GuiMode::Keyboard
                    }
                };
                let blocks_gameplay = m3_blocks_gameplay_input(mode, m3_input_class(&event));
                if mode != GuiMode::None {
                    if let Event::WindowEvent {
                        event: window_event,
                        window_id,
                    } = &event
                    {
                        if *window_id == window.id() && is_m3_cursor_focus_event(window_event) {
                            // Preserve renderer-owned cursor policy without using
                            // route_platform_input_to_app, which would queue this
                            // focus event to gameplay.
                            if let Err(error) = renderer.route_platform_input(&window, &event) {
                                log::error!("Platform cursor routing failed: {error}");
                                shutdown(coordinator, &mut renderer, &mut scene);
                                elwt.exit();
                                return;
                            }
                            cursor_position = None;
                        }
                    }
                }
                match mode {
                    GuiMode::None if !blocks_gameplay => {
                        if let Err(error) = engine::input::route_platform_input_to_app(
                            &mut renderer,
                            &window,
                            &mut loop_state.input,
                            &event,
                        ) {
                            log::error!("Platform input routing failed: {error}");
                            shutdown(coordinator, &mut renderer, &mut scene);
                            elwt.exit();
                            return;
                        }
                        if let Some(key) = initial_physical_key(&event) {
                            let snapshot = match gui.try_borrow() {
                                Ok(gui) => hotkey_generation_snapshot(key, ctrl_held, &gui.config),
                                Err(_) => Err("M3 GUI is busy; hotkey was not applied".into()),
                            };
                            match snapshot {
                                Ok(Some(snapshot)) => match enqueue_m3_generation(
                                    snapshot,
                                    false,
                                    &worker,
                                    &tools_dir,
                                    &package_root,
                                    &last_request,
                                    &mut close_intent,
                                    &window,
                                ) {
                                    Ok(config) => {
                                        if let Ok(mut gui) = gui.try_borrow_mut() {
                                            gui.config = config.clone();
                                        }
                                        last_title_desc = config.describe();
                                        set_gui_status(&gui, "Generation pending.");
                                    }
                                    Err(error) => set_gui_status(&gui, error),
                                },
                                Ok(None) => {}
                                Err(error) => {
                                    set_gui_status(&gui, format!("Cannot generate: {error}"))
                                }
                            }
                        }
                    }
                    GuiMode::None => {}
                    GuiMode::Keyboard => {
                        // Every keyboard event belongs to M3Gui; all pointer and raw
                        // mouse events are consumed without renderer/gameplay routing.
                        if let Event::WindowEvent {
                            event:
                                WindowEvent::KeyboardInput {
                                    event: key_event, ..
                                },
                            window_id,
                        } = &event
                        {
                            if *window_id == window.id() {
                                if let PhysicalKey::Code(key) = key_event.physical_key {
                                    let action = match key_event.state {
                                        ElementState::Pressed if !key_event.repeat => {
                                            M3Action::Press
                                        }
                                        ElementState::Pressed => M3Action::Repeat,
                                        ElementState::Released => M3Action::Release,
                                    };
                                    let result = gui
                                        .try_borrow_mut()
                                        .map(|mut gui| gui.handle_keyboard_input(key, action))
                                        .unwrap_or(GuiAction::None);
                                    handle_m3_action(
                                        result,
                                        GuiMode::Keyboard,
                                        &mut renderer,
                                        &window,
                                        &gui,
                                        &mut loop_state,
                                        &mut registration,
                                        &worker,
                                        &tools_dir,
                                        &package_root,
                                        &last_request,
                                        &mut close_intent,
                                        &mut last_title_desc,
                                    );
                                }
                            }
                        }
                    }
                    GuiMode::Mouse => match &event {
                        Event::WindowEvent {
                            event: WindowEvent::CursorMoved { position, .. },
                            window_id,
                        } if *window_id == window.id() => {
                            cursor_position = Some(logical_cursor_from_physical(
                                *position,
                                window.scale_factor(),
                            ));
                        }
                        Event::WindowEvent {
                            event:
                                WindowEvent::CursorEntered { .. } | WindowEvent::CursorLeft { .. },
                            window_id,
                        } if *window_id == window.id() => {
                            // Wait for a fresh viewport-relative CursorMoved event
                            // before accepting another click.
                            cursor_position = None;
                        }
                        Event::WindowEvent {
                            event: WindowEvent::MouseInput { state, button, .. },
                            window_id,
                        } if *window_id == window.id() => {
                            if let Some((x, y)) = cursor_position {
                                let action = if *state == ElementState::Pressed {
                                    M3Action::Press
                                } else {
                                    M3Action::Release
                                };
                                let result = gui
                                    .try_borrow_mut()
                                    .map(|mut gui| gui.handle_mouse_input(x, y, *button, action))
                                    .unwrap_or(GuiAction::None);
                                handle_m3_action(
                                    result,
                                    GuiMode::Mouse,
                                    &mut renderer,
                                    &window,
                                    &gui,
                                    &mut loop_state,
                                    &mut registration,
                                    &worker,
                                    &tools_dir,
                                    &package_root,
                                    &last_request,
                                    &mut close_intent,
                                    &mut last_title_desc,
                                );
                            }
                        }
                        Event::WindowEvent {
                            event: WindowEvent::MouseWheel { delta, .. },
                            window_id,
                        } if *window_id == window.id() => {
                            if let Ok(mut gui) = gui.try_borrow_mut() {
                                gui.scroll_by(scroll_delta_to_gui_lines(delta));
                            }
                        }
                        Event::WindowEvent {
                            event:
                                WindowEvent::KeyboardInput {
                                    event: key_event, ..
                                },
                            window_id,
                        } if *window_id == window.id()
                            && key_event.state == ElementState::Pressed
                            && !key_event.repeat
                            && matches!(
                                key_event.physical_key,
                                PhysicalKey::Code(KeyCode::Escape)
                            ) =>
                        {
                            cursor_position = None;
                            if let Err(error) = apply_m3_mode_transition(
                                &mut renderer,
                                &window,
                                &gui,
                                &mut loop_state,
                                &mut registration,
                                GuiMode::Mouse,
                            ) {
                                set_gui_status(&gui, error);
                            }
                        }
                        _ => {}
                    },
                }
            }

            // Process every completed result. Only the latest request may
            // publish; failures retain both the old mount and an open menu.
            while let Some(result) = worker.poll_result() {
                let latest = last_request.load(Ordering::Relaxed);
                if result.id != latest {
                    log::info!(
                        "discarding stale generation result {} (latest {})",
                        result.id,
                        latest
                    );
                    let _ = std::fs::remove_dir_all(&result.package_dir);
                    if close_intent == Some(result.id) {
                        close_intent = None;
                    }
                    continue;
                }

                if !result.success {
                    let error = result
                        .error
                        .unwrap_or_else(|| "unknown worker failure".into());
                    log::error!("generation {} failed: {error}", result.id);
                    window.set_title(&format!(
                        "BSP Beta — m3-generate [failed] | {}",
                        result.config.describe()
                    ));
                    set_gui_status(&gui, format!("Generation failed: {error}"));
                    let _ = std::fs::remove_dir_all(&result.package_dir);
                    if close_intent == Some(result.id) {
                        close_intent = None;
                    }
                    continue;
                }

                match commit_generated_package(
                    &result.package_dir,
                    coordinator,
                    &mut renderer,
                    &mut scene,
                    scale,
                ) {
                    Ok(mounted) => {
                        apply_mounted_app_state(&mut loop_state, mounted);
                        if let Ok(mut gui) = gui.try_borrow_mut() {
                            gui.flash_generated();
                            gui.status = Some("Generated.".into());
                        }
                        let apply_close = close_intent == Some(result.id);
                        close_intent = None;
                        let desc = result.config.describe();
                        last_title_desc = desc.clone();
                        window.set_title(&format!("BSP Beta — m3-generate | {desc}"));
                        if apply_close {
                            // Copy the mode out of RefCell before the transition
                            // call so the call site has an explicit guard-drop
                            // boundary and cannot regress to passing a live guard.
                            let current_mode = match gui.try_borrow() {
                                Ok(gui) => Some(gui.mode),
                                Err(_) => {
                                    set_gui_status(
                                        &gui,
                                        "Generated, but the busy M3 GUI could not be closed.",
                                    );
                                    None
                                }
                            };
                            if let Some(current_mode) = current_mode {
                                if let Err(error) = apply_m3_mode_transition(
                                    &mut renderer,
                                    &window,
                                    &gui,
                                    &mut loop_state,
                                    &mut registration,
                                    current_mode,
                                ) {
                                    // A successful mount must not leave a stale close intent;
                                    // retain the menu and surface the ownership error instead.
                                    set_gui_status(
                                        &gui,
                                        format!("Generated, but could not close menu: {error}"),
                                    );
                                } else {
                                    title_toast_deadline = Some(
                                        Instant::now()
                                            + Duration::from_secs_f32(GENERATED_TOAST_SECS),
                                    );
                                }
                            }
                        }
                    }
                    Err(error) => {
                        log::error!(
                            "generated replacement failed; previous world remains active: {error}"
                        );
                        window.set_title(&format!(
                            "BSP Beta — m3-generate [failed] | {}",
                            result.config.describe()
                        ));
                        set_gui_status(
                            &gui,
                            format!("Generation failed; previous world remains active: {error}"),
                        );
                        let _ = std::fs::remove_dir_all(&result.package_dir);
                        if close_intent == Some(result.id) {
                            close_intent = None;
                        }
                    }
                }
            }

            // ── 5. Drain retirements ──────────────────────────────────────
            if let Err(error) = drain_and_retire(coordinator, &mut renderer) {
                log::error!("BSP retirement handoff failed: {error}");
            }

            if let Some(deadline) = title_toast_deadline {
                if Instant::now() < deadline {
                    window.set_title("BSP Beta — m3-generate | Generated!");
                } else {
                    title_toast_deadline = None;
                    window.set_title(&format!("BSP Beta — m3-generate | {last_title_desc}"));
                }
            }

            // ── 7. Window events ──────────────────────────────────────────
            match event {
                Event::WindowEvent { event, window_id } if window_id == window.id() => {
                    match event {
                        WindowEvent::CloseRequested => {
                            shutdown(coordinator, &mut renderer, &mut scene);
                            elwt.exit();
                        }
                        WindowEvent::Resized(size) => {
                            let (width, height) = logical_viewport_from_physical(
                                size.width,
                                size.height,
                                window.scale_factor(),
                            );
                            if let Ok(mut gui) = gui.try_borrow_mut() {
                                gui.set_viewport(width, height);
                            }
                            if let Err(error) = renderer.resize(size.width, size.height) {
                                log::error!("Resize failed: {error}");
                                shutdown(coordinator, &mut renderer, &mut scene);
                                elwt.exit();
                            }
                        }
                        WindowEvent::ScaleFactorChanged { .. } => {
                            let size = window.inner_size();
                            let (width, height) = logical_viewport_from_physical(
                                size.width,
                                size.height,
                                window.scale_factor(),
                            );
                            if let Ok(mut gui) = gui.try_borrow_mut() {
                                gui.set_viewport(width, height);
                            }
                        }
                        WindowEvent::RedrawRequested => {
                            let size = window.inner_size();
                            match render_app_frame(
                                &mut renderer,
                                &mut scene,
                                &mut loop_state,
                                size.width,
                                size.height,
                                false,
                            ) {
                                Ok(
                                    FrameRenderOutcome::Rendered
                                    | FrameRenderOutcome::SkippedAcquireUnavailable
                                    | FrameRenderOutcome::SkippedResizePending
                                    | FrameRenderOutcome::SubmittedNotPresented
                                    | FrameRenderOutcome::PresentedSuboptimal,
                                ) => window.request_redraw(),
                                Err(error) => {
                                    log::error!("Render failed: {error}");
                                    shutdown(coordinator, &mut renderer, &mut scene);
                                    elwt.exit();
                                }
                            }
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
    movement_world: BspMovementWorld,
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
    } else if let (Some(origin), Some(look_at)) = (
        args.acceptance_camera_origin,
        args.acceptance_camera_look_at,
    ) {
        bsp_explicit_camera(origin, look_at, extracted)?
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

    let mut loop_state =
        AppLoopState::new(headless_camera, ModelMappings::default(), movement_world);
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

fn run_headless(
    args: &cli::CliArgs,
    coordinator: &mut BspCoordinator,
    movement_world: BspMovementWorld,
) -> Result<(), AppError> {
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
    } = match prepare_headless_runtime(coordinator, None, args, movement_world) {
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

fn run_mcp(
    coordinator: &mut BspCoordinator,
    movement_world: BspMovementWorld,
) -> Result<(), AppError> {
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
        movement_world,
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
    movement_controller: BspPlayerMovementController,
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
    /// When false, FPS controller update and gameplay input are skipped.
    pub gameplay_input_enabled: bool,
}

impl AppLoopState {
    fn new(
        camera: Camera,
        model_mappings: ModelMappings,
        movement_world: BspMovementWorld,
    ) -> Self {
        let movement_controller =
            BspPlayerMovementController::new(camera.get_position(), movement_world);
        Self {
            camera,
            fps_controller: FPSController::new(0.002, 1.0),
            movement_controller,
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
            gameplay_input_enabled: true,
        }
    }
}

fn active_movement_input(
    snapshot: &engine::input::InputSnapshot,
    camera: &Camera,
) -> MovementInput {
    let forward_axis = snapshot.action_value(&ActionId::new("move.forward"))
        - snapshot.action_value(&ActionId::new("move.backward"));
    let right_axis = snapshot.action_value(&ActionId::new("move.right"))
        - snapshot.action_value(&ActionId::new("move.left"));
    let camera_to_world = camera.get_view_matrix().inverse();
    let mut forward = camera_to_world.transform_vector3(Vec3::NEG_Z);
    forward.y = 0.0;
    forward = forward.normalize_or_zero();
    let right = forward.cross(Vec3::Y).normalize_or_zero();
    let wish_direction = forward * forward_axis + right * right_axis;
    MovementInput::new(
        wish_direction,
        forward_axis,
        snapshot.action_just_pressed(&ActionId::new("move.up")),
    )
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
    let mut fixed_movement_input = None;
    if state.gameplay_input_enabled {
        if state.movement_controller.is_active() {
            state
                .movement_controller
                .synchronize_external_position(state.camera.get_position());
            // Rotation is sampled once per display frame. Translation remains
            // owned by the fixed-step Richness boundary below.
            state.fps_controller.update_from_snapshot(
                state.input.snapshot(),
                0.0,
                &mut state.camera,
            );
            fixed_movement_input =
                Some(active_movement_input(state.input.snapshot(), &state.camera));
        } else {
            // Direct/baseline maps without qualified Richness descriptors keep
            // the existing free-camera behavior unchanged.
            state.fps_controller.update_from_snapshot(
                state.input.snapshot(),
                simulated_dt,
                &mut state.camera,
            );
        }
    }

    if !fixed_update.dropped_time.is_zero() {
        log::warn!(
            "Dropped {:.3}ms of accumulated BSP beta simulation time.",
            fixed_update.dropped_time.as_secs_f64() * 1_000.0
        );
    }

    // ── Snapshot production at each fixed step ───────────────────────
    for _step in 0..fixed_update.steps {
        if let Some(input) = fixed_movement_input {
            state.movement_controller.fixed_step(input, FIXED_DT);
            state
                .camera
                .set_position(state.movement_controller.position());
        }
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

    for diagnostic in state.movement_controller.take_diagnostics() {
        log::warn!(
            "{} volume={:?}: {}",
            diagnostic.code,
            diagnostic.volume_id,
            diagnostic.detail
        );
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
mod sweep_stale_roots_tests {
    use super::sweep_stale_package_roots;
    use std::path::PathBuf;
    use std::time::Duration;

    fn root(name: &str) -> PathBuf {
        std::env::temp_dir().join(name)
    }

    fn age(path: &std::path::Path, spec: &str) {
        let _ = std::process::Command::new("touch")
            .args(["-d", spec])
            .arg(path)
            .status();
    }

    #[test]
    fn sweep_removes_stale_roots_and_keeps_fresh_and_live() {
        let stale = root(&format!("bsp-beta-m3-999999-100-0"));
        let fresh = root(&format!(
            "bsp-beta-m3-{}-{}-0",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let unrelated = root("unrelated-dir");
        std::fs::create_dir_all(&stale).unwrap();
        std::fs::create_dir_all(&fresh).unwrap();
        std::fs::create_dir_all(&unrelated).unwrap();

        age(&stale, "1 hour ago");
        age(&unrelated, "1 hour ago");

        sweep_stale_package_roots(Duration::from_secs(300));

        assert!(!stale.exists(), "stale root should be reaped");
        assert!(
            fresh.exists(),
            "fresh root owned by this live PID must survive"
        );
        assert!(unrelated.exists(), "non-package dirs must survive");

        let _ = std::fs::remove_dir_all(&stale);
        let _ = std::fs::remove_dir_all(&fresh);
        let _ = std::fs::remove_dir_all(&unrelated);
    }
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

#[cfg(test)]
mod m3_integration_tests {
    use super::*;
    use bsp_generator::enhanced_v3::{ArchType, V3Preset};
    use winit::event::MouseScrollDelta;

    #[test]
    fn cli_generation_overrides_seed_the_initial_gui_config() {
        let args = cli::parse_from([
            "--development",
            "--m3-generate",
            "--seed",
            "42",
            "--preset",
            "rich",
            "--rooms",
            "28",
            "--corridors",
            "30",
            "--loops",
            "4",
            "--no-chamfer",
            "--arch-type",
            "segmented",
            "--grammar-families",
            "portal-chamber,column-grove,terraced-shrine",
        ])
        .unwrap();
        let config = initial_m3_config(&args).unwrap();
        assert_eq!(config.seed, 42);
        assert_eq!(config.preset, V3Preset::Rich);
        assert_eq!(config.extent, 3072);
        assert_eq!(config.rooms, Some(28));
        assert_eq!(config.corridors, Some(30));
        assert_eq!(config.loops, Some(4));
        assert!(!config.chamfer);
        assert_eq!(config.arch_type, ArchType::Segmented);
        assert_eq!(config.grammar_families.len(), 3);
    }

    #[test]
    fn mode_planner_keeps_registration_on_switch_and_gates_gameplay() {
        let open = plan_m3_transition(GuiMode::None, GuiMode::Keyboard, false, false).unwrap();
        assert_eq!(open.next, GuiMode::Keyboard);
        assert_eq!(open.registration, M3RegistrationChange::Register);
        assert!(open.queue_releases);
        let switch = plan_m3_transition(GuiMode::Keyboard, GuiMode::Mouse, true, true).unwrap();
        assert_eq!(switch.next, GuiMode::Mouse);
        assert_eq!(switch.registration, M3RegistrationChange::None);
        assert!(!switch.queue_releases);
        let close = plan_m3_transition(GuiMode::Mouse, GuiMode::Mouse, true, true).unwrap();
        assert_eq!(close.next, GuiMode::None);
        assert_eq!(close.registration, M3RegistrationChange::UnregisterOwned);
        assert!(!close.queue_releases);
    }

    #[test]
    fn transition_planner_rejects_foreign_ui_before_opening() {
        let error = plan_m3_transition(GuiMode::None, GuiMode::Keyboard, false, true)
            .expect_err("a foreign app UI must block M3 registration");
        assert!(error.contains("another app UI"));
    }

    #[test]
    fn close_repairs_mode_and_gate_when_owned_callback_is_already_absent() {
        let transition = plan_m3_transition(GuiMode::Mouse, GuiMode::Mouse, false, true).unwrap();
        assert_eq!(transition.registration, M3RegistrationChange::None);

        let mut mode = GuiMode::Mouse;
        let mut input = InputSystem::new();
        let mut gameplay_input_enabled = false;
        commit_m3_transition_state(
            transition,
            &mut mode,
            &mut input,
            &mut gameplay_input_enabled,
        );

        assert_eq!(mode, GuiMode::None);
        assert!(gameplay_input_enabled);
        assert_eq!(input.debug_snapshot().queued_events, 0);
    }

    #[test]
    fn global_mode_hotkey_accepts_only_initial_f1_and_f2_presses() {
        assert_eq!(
            mode_hotkey_from_key(KeyCode::F1, ElementState::Pressed, false),
            Some(GuiMode::Keyboard)
        );
        assert_eq!(
            mode_hotkey_from_key(KeyCode::F2, ElementState::Pressed, false),
            Some(GuiMode::Mouse)
        );
        assert_eq!(
            mode_hotkey_from_key(KeyCode::F1, ElementState::Released, false),
            None
        );
        assert_eq!(
            mode_hotkey_from_key(KeyCode::F2, ElementState::Pressed, true),
            None
        );
    }

    #[test]
    fn open_modes_block_every_gameplay_input_class() {
        for mode in [GuiMode::Keyboard, GuiMode::Mouse] {
            for class in [
                M3InputClass::Keyboard,
                M3InputClass::Mouse,
                M3InputClass::RawMouse,
            ] {
                assert!(m3_blocks_gameplay_input(mode, class));
            }
            assert!(!m3_blocks_gameplay_input(mode, M3InputClass::Lifecycle));
        }
        assert!(!m3_blocks_gameplay_input(
            GuiMode::None,
            M3InputClass::Keyboard
        ));
    }

    #[test]
    fn cursor_conversion_and_scroll_direction_match_logical_gui_space() {
        assert_eq!(
            logical_cursor_from_physical(PhysicalPosition::new(300.0, 150.0), 1.5),
            (200.0, 100.0)
        );
        assert_eq!(logical_viewport_from_physical(1920, 1080, 1.5), (1280, 720));
        assert_eq!(
            logical_cursor_from_physical(PhysicalPosition::new(300.0, 150.0), 0.75),
            (400.0, 200.0)
        );
        assert_eq!(
            logical_viewport_from_physical(1440, 810, 0.75),
            (1920, 1080)
        );
        assert_eq!(
            scroll_delta_to_gui_lines(&MouseScrollDelta::LineDelta(0.0, 2.0)),
            -2.0
        );
        assert_eq!(
            scroll_delta_to_gui_lines(&MouseScrollDelta::LineDelta(0.0, -2.0)),
            2.0
        );
        let mut gui = M3Gui::new();
        gui.set_viewport(640, 180);
        gui.scroll_by(100.0);
        let before = gui.scroll_offset;
        gui.scroll_by(scroll_delta_to_gui_lines(&MouseScrollDelta::LineDelta(
            0.0, 1.0,
        )));
        assert!(
            gui.scroll_offset < before,
            "wheel-up must move toward the top"
        );
    }

    #[test]
    fn mouse_button_release_is_harmless_and_uses_real_cursor_coordinates() {
        let mut gui = M3Gui::new();
        gui.set_viewport(1280, 720);
        let before = gui.config.chamfer;
        // This is the viewport-relative cursor coordinate passed by CursorMoved,
        // not the window's screen position.
        let layout = gui.render();
        assert!(!layout.is_empty());
        assert_eq!(
            gui.handle_mouse_input(0.0, 0.0, MouseButton::Left, M3Action::Release),
            GuiAction::None
        );
        assert_eq!(gui.config.chamfer, before);
    }

    #[test]
    fn hotkey_snapshots_normalize_stairs_and_reject_invalid_values_transactionally() {
        let mut config = GenConfig::default_config();
        config.vertical_edges = Some(3);
        let snapshot = hotkey_generation_snapshot(KeyCode::F9, false, &config)
            .unwrap()
            .expect("F9 requests generation");
        assert!(!snapshot.stairs);
        assert_eq!(snapshot.vertical_edges, Some(0));
        assert_eq!(
            config.vertical_edges,
            Some(3),
            "draft changes only after enqueue"
        );

        let mut invalid = GenConfig::default_config();
        invalid.extent = 1;
        let before = invalid.clone();
        assert!(hotkey_generation_snapshot(KeyCode::F5, false, &invalid).is_err());
        assert_eq!(invalid, before, "rejected hotkey must not mutate the draft");
    }

    #[test]
    fn close_intent_is_latest_action_only_and_matching_success_only() {
        let mut intent = Some(4);
        update_close_intent(&mut intent, 5, false);
        assert_eq!(intent, None, "ordinary Generate cancels close intent");
        update_close_intent(&mut intent, 7, true);
        assert_eq!(intent, Some(7));
        assert_ne!(Some(6), intent, "stale completion cannot close");
        assert_eq!(
            Some(7),
            intent,
            "only the matching successful commit closes"
        );
    }

    #[test]
    fn opening_transition_releases_active_gameplay_and_closes_the_gate() {
        let mut input = InputSystem::new();
        install_app_fps_input(&mut input);
        input.queue_event(engine::input::InputEvent::Key {
            code: KeyCode::KeyW,
            state: ElementState::Pressed,
            repeat: false,
            modifiers: ModifiersState::empty(),
        });
        input.dispatch_frame();
        let forward = engine::input::ActionId::from("move.forward");
        assert!(input.snapshot().action_pressed(&forward));

        let transition =
            plan_m3_transition(GuiMode::None, GuiMode::Keyboard, false, false).unwrap();
        let mut mode = GuiMode::None;
        let mut gameplay_input_enabled = true;
        commit_m3_transition_state(
            transition,
            &mut mode,
            &mut input,
            &mut gameplay_input_enabled,
        );
        input.dispatch_frame();

        assert_eq!(mode, GuiMode::Keyboard);
        assert!(!gameplay_input_enabled);
        assert!(!input.snapshot().action_pressed(&forward));
        assert!(input.snapshot().action_just_released(&forward));
    }
}
