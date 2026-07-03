mod collision;
mod content;
mod generator;
mod geometry;
mod layout;
mod player;
mod scene_seed;

use std::path::PathBuf;
use std::time::Instant;

use collision::CollisionWorld;
use content::{load_content_pack, resolve_content_path};
use generator::{generate_dungeon, GeneratedDungeon, ProceduralLevelConfig, GENERATED_LEVEL_ID};
use layout::{load_level_file, tile_to_world, ParsedLevel};
use player::{CameraIntentGuard, PlayerState, PLAYER_EYE_HEIGHT};
use renderer::api::config::{CompressionConfig, TextureCompressionMode};
use renderer::{
    AssetManifestMode, AssetPolicyConfig, FrameRenderOutcome, RendererConfig, RendererError,
};
use scene_seed::{renderer_visual_tuning, LevelScene};
use thiserror::Error;
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::WindowBuilder;

const APP_WINDOW_TITLE: &str = "Dungeon Dogfood - Phase 07";
const DEFAULT_LEVEL_ID: &str = GENERATED_LEVEL_ID;
const LEVEL_SELECT_ENV: &str = "DUNGEON_DOGFOOD_LEVEL";
const LEVEL_01_PATH: &str = "apps/dungeon_dogfood/assets/levels/level_01.txt";
const LEVEL_02_PATH: &str = "apps/dungeon_dogfood/assets/levels/level_02_ramps.txt";
const LEVEL_03_PATH: &str = "apps/dungeon_dogfood/assets/levels/level_03_lighting.txt";
const CONTENT_PACK_PATH: &str = "apps/dungeon_dogfood/assets/content_pack.toml";
const GENERATOR_SEED_ENV: &str = "DUNGEON_DOGFOOD_GENERATOR_SEED";
const GENERATOR_WIDTH_ENV: &str = "DUNGEON_DOGFOOD_GENERATOR_WIDTH";
const GENERATOR_HEIGHT_ENV: &str = "DUNGEON_DOGFOOD_GENERATOR_HEIGHT";
const GENERATOR_LAYERS_ENV: &str = "DUNGEON_DOGFOOD_GENERATOR_LAYERS";

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
    let content_pack = load_content_pack(CONTENT_PACK_PATH)?;

    log::info!(
        "Loaded content pack: {} props ({} enabled), {} materials, {} environments, {} light presets",
        content_pack.props.len(),
        content_pack.enabled_props().len(),
        content_pack.materials.len(),
        content_pack.environments.len(),
        content_pack.light_presets.len()
    );

    let level_selection = selected_level();
    let loaded_level = load_selected_level(&level_selection)?;
    let level = loaded_level.level;

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
    if let Some(seed) = loaded_level.seed {
        log::info!(
            "Generated dungeon seed={} walkable_tiles={} connectors={}",
            seed,
            loaded_level.walkable_tiles,
            loaded_level.connector_count
        );
    }
    if let Some(map_overview) = loaded_level.map_overview.as_ref() {
        log::info!("Dungeon map overview:\n{}", map_overview);
    }

    let collision_world = CollisionWorld::from_level(&level);

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
    renderer.install_default_fps_input();

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
    renderer.set_camera_position(spawn_position);

    log::info!("Dungeon dogfood initialized, starting event loop");

    let mut last_frame = Instant::now();
    let mut last_window_size = window.inner_size();
    let mut resize_pending = false;
    window.request_redraw();

    event_loop
        .run(move |event, elwt| {
            elwt.set_control_flow(ControlFlow::Poll);

            if let Err(e) = renderer.update_input(&window, &event) {
                log::error!("Input update failed: {}", e);
                elwt.exit();
                return;
            }

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

                            let now = Instant::now();
                            let delta_seconds = now.duration_since(last_frame).as_secs_f32();
                            last_frame = now;

                            match render_frame(
                                &mut renderer,
                                &window,
                                &mut scene,
                                &collision_world,
                                &mut player,
                                delta_seconds,
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

#[derive(Debug, Clone, PartialEq, Eq)]
struct LevelSelection {
    label: String,
    path: PathBuf,
    procedural: bool,
}

struct LoadedLevel {
    level: ParsedLevel,
    source_description: String,
    map_overview: Option<String>,
    seed: Option<u64>,
    walkable_tiles: usize,
    connector_count: usize,
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

    let (label, path, procedural) = match selector {
        DEFAULT_LEVEL_ID | "generated" => (GENERATED_LEVEL_ID, GENERATED_LEVEL_ID, true),
        "level_01" | "level_01.txt" | LEVEL_01_PATH => ("level_01", LEVEL_01_PATH, false),
        "level_02_ramps" | "level_02_ramps.txt" | LEVEL_02_PATH => {
            ("level_02_ramps", LEVEL_02_PATH, false)
        }
        "level_03_lighting" | "level_03_lighting.txt" | LEVEL_03_PATH => {
            ("level_03_lighting", LEVEL_03_PATH, false)
        }
        _ => (selector, selector, false),
    };

    LevelSelection {
        label: label.to_string(),
        path: PathBuf::from(path),
        procedural,
    }
}

fn load_selected_level(selection: &LevelSelection) -> Result<LoadedLevel, AppError> {
    if selection.procedural {
        let procedural_config = procedural_level_config();
        let GeneratedDungeon {
            level,
            seed,
            map_overview,
            walkable_tiles,
            connector_count,
        } = generate_dungeon(procedural_config);

        return Ok(LoadedLevel {
            level,
            source_description: format!(
                "procedural generator (seed={}, {}x{}x{})",
                seed, procedural_config.width, procedural_config.height, procedural_config.layers
            ),
            map_overview: Some(map_overview),
            seed: Some(seed),
            walkable_tiles,
            connector_count,
        });
    }

    let resolved_level_path = resolve_content_path(&selection.path);
    let level = load_level_file(&resolved_level_path).map_err(|source| AppError::LevelLoad {
        selection: selection.clone(),
        source,
    })?;

    Ok(LoadedLevel {
        level,
        source_description: resolved_level_path.display().to_string(),
        map_overview: None,
        seed: None,
        walkable_tiles: 0,
        connector_count: 0,
    })
}

fn procedural_level_config() -> ProceduralLevelConfig {
    let default = ProceduralLevelConfig::default();
    ProceduralLevelConfig {
        width: env_usize(GENERATOR_WIDTH_ENV).unwrap_or(default.width),
        height: env_usize(GENERATOR_HEIGHT_ENV).unwrap_or(default.height),
        layers: env_usize(GENERATOR_LAYERS_ENV).unwrap_or(default.layers),
        seed: env_u64(GENERATOR_SEED_ENV).unwrap_or(default.seed),
    }
    .sanitized()
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

fn render_frame(
    renderer: &mut renderer::Renderer,
    window: &winit::window::Window,
    scene: &mut renderer::Scene,
    collision_world: &CollisionWorld,
    player: &mut PlayerState,
    delta_seconds: f32,
) -> Result<renderer::FrameRenderOutcome, RendererError> {
    renderer.pump_asset_tasks(32)?;

    let mut frame = renderer.begin_frame(window)?;

    // begin_frame() advances the internal FPS controller from input state.
    // Read that intended movement, resolve collisions, and push corrected eye position back.
    let camera_pos = renderer.camera_position();
    match player.ingest_camera_intent(camera_pos, delta_seconds) {
        CameraIntentGuard::Accepted => {}
        CameraIntentGuard::Clamped {
            attempted_displacement,
            applied_displacement,
        } => {
            log::warn!(
                "Clamped player movement from {:.3}m to {:.3}m for this frame.",
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
    collision::resolve_player_step(player, collision_world, delta_seconds);
    if !player.has_finite_position() {
        log::error!(
            "Player position became non-finite after collision resolution: {:?}",
            player.position
        );
        return Err(RendererError::InvalidState(
            "player position must remain finite before camera write-back",
        ));
    }
    renderer.set_camera_position(player.position);

    let outcome = renderer.render_scene_in_frame(&mut frame, scene)?;
    renderer.end_frame(frame)?;

    Ok(outcome)
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
    eprintln!(
        "Procedural generator env: {} {} {} {}",
        GENERATOR_SEED_ENV, GENERATOR_WIDTH_ENV, GENERATOR_HEIGHT_ENV, GENERATOR_LAYERS_ENV
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

fn env_usize(var_name: &str) -> Option<usize> {
    std::env::var(var_name).ok().and_then(|value| {
        value
            .trim()
            .parse::<usize>()
            .map_err(|err| {
                log::warn!("Ignoring invalid {}='{}': {}", var_name, value, err);
                err
            })
            .ok()
    })
}

fn env_u64(var_name: &str) -> Option<u64> {
    std::env::var(var_name).ok().and_then(|value| {
        value
            .trim()
            .parse::<u64>()
            .map_err(|err| {
                log::warn!("Ignoring invalid {}='{}': {}", var_name, value, err);
                err
            })
            .ok()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_builtin_level_id() {
        let selection = resolve_level_selector("level_02_ramps");
        assert_eq!(selection.label, "level_02_ramps");
        assert_eq!(selection.path, PathBuf::from(LEVEL_02_PATH));
        assert!(!selection.procedural);
    }

    #[test]
    fn resolve_builtin_level_filename() {
        let selection = resolve_level_selector("level_03_lighting.txt");
        assert_eq!(selection.label, "level_03_lighting");
        assert_eq!(selection.path, PathBuf::from(LEVEL_03_PATH));
        assert!(!selection.procedural);
    }

    #[test]
    fn resolve_generated_level_id() {
        let selection = resolve_level_selector("generated");
        assert_eq!(selection.label, GENERATED_LEVEL_ID);
        assert!(selection.procedural);
    }

    #[test]
    fn preserve_custom_level_paths() {
        let selection = resolve_level_selector("custom/level.txt");
        assert_eq!(selection.label, "custom/level.txt");
        assert_eq!(selection.path, PathBuf::from("custom/level.txt"));
        assert!(!selection.procedural);
    }
}
