mod game_state;
mod layout;
mod scene_seed;

use std::time::Instant;

use game_state::GameState;
use layout::{load_level_file, tile_to_world};
use renderer::{RendererConfig, RendererError};
use scene_seed::LevelScene;
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::WindowBuilder;

const LEVEL_PATH: &str = "apps/dungeon_dogfood/assets/levels/level_01.txt";

fn main() {
    // Initialize logging
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Info)
        .init();

    // Parse level file
    let level = load_level_file(LEVEL_PATH).unwrap_or_else(|e| {
        eprintln!("Failed to load level file '{}': {}", LEVEL_PATH, e);
        eprintln!("\nExpected ASCII level file with tokens:");
        eprintln!("  # = wall");
        eprintln!("  . = floor");
        eprintln!("  S = spawn marker (exactly 1 required)");
        eprintln!("  M = model marker");
        eprintln!("  L = point light marker");
        eprintln!("  R^ R> Rv R< = ramp tiles");
        std::process::exit(1);
    });

    log::info!("Loaded level: {}x{} tiles", level.width, level.height);
    log::info!("Spawn position: {:?}", level.spawn);
    log::info!("Light markers: {}", level.light_markers.len());
    log::info!("Model markers: {}", level.model_markers.len());

    // Create window and event loop
    let event_loop = EventLoop::new().expect("failed to create event loop");
    let window = WindowBuilder::new()
        .with_title("Dungeon Dogfood - Phase 03")
        .with_inner_size(winit::dpi::LogicalSize::new(1280, 720))
        .build(&event_loop)
        .expect("failed to create window");

    // Initialize renderer
    let config = RendererConfig {
        app_name: "dungeon_dogfood".to_string(),
        window_width: 1280,
        window_height: 720,
        validation_layer: cfg!(debug_assertions),
        compile_shaders: false,
        shader_debug_mode: renderer::DebugRuntimeMode::Default,
        headless: false,
    };

    let mut renderer = renderer::Renderer::new(config, &window)
        .expect("failed to initialize renderer");

    // Extract startup scene
    let mut scene = renderer
        .take_startup_scene()
        .expect("startup scene already taken");

    // Seed scene from level
    let _level_scene = {
        let mut assets = renderer.assets();
        LevelScene::from_level(&level, &mut scene, &mut assets)
            .expect("failed to seed scene from level")
    };

    // Initialize game state
    let spawn_world = tile_to_world(level.spawn.0, level.spawn.1);
    let spawn_position = spawn_world + glam::Vec3::new(0.5, 0.5, -0.5); // Center of spawn tile, raised
    let mut game_state = GameState::new(spawn_position);

    // Set initial camera position to spawn
    renderer.set_camera_position(spawn_position);

    log::info!("Dungeon dogfood initialized, starting event loop");

    // Frame timing
    let mut last_frame = Instant::now();

    // Run event loop
    event_loop
        .run(move |event, elwt| {
            elwt.set_control_flow(ControlFlow::Poll);

            // Forward input events to renderer
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
                            if let Err(e) = renderer.resize(new_size.width, new_size.height) {
                                log::error!("Resize failed: {}", e);
                                elwt.exit();
                            }
                        }
                        WindowEvent::RedrawRequested => {
                            // Calculate delta time
                            let now = Instant::now();
                            let delta_seconds = now.duration_since(last_frame).as_secs_f32();
                            last_frame = now;

                            // Update game state (Phase 04 will add collision here)
                            game_state.update(delta_seconds);

                            // Render frame using explicit frame API
                            match render_frame(&mut renderer, &window, &mut scene) {
                                Ok(_) => {}
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
                Event::AboutToWait => {
                    window.request_redraw();
                }
                _ => {}
            }
        })
        .expect("event loop failed");
}

/// Render a single frame using explicit frame API
///
/// Phase 03: Basic render path
/// Phase 04: Will add collision update between begin_frame and render_scene_in_frame
fn render_frame(
    renderer: &mut renderer::Renderer,
    window: &winit::window::Window,
    scene: &mut renderer::Scene,
) -> Result<renderer::FrameRenderOutcome, RendererError> {
    // Pump async asset loads
    renderer.pump_asset_tasks(32)?;

    // Begin frame
    let mut frame = renderer.begin_frame(window)?;

    // Phase 04 will add collision update here:
    // - Read movement intent from input
    // - Run collision solver
    // - Update camera position via renderer.set_camera_position()

    // Render scene
    let outcome = renderer.render_scene_in_frame(&mut frame, scene)?;

    // End frame
    renderer.end_frame(frame)?;

    Ok(outcome)
}
