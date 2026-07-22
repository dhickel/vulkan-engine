//! Guide App — Maintained checkpoint for `docs/guide/`.
//!
//! This is the minimal complete app-owned rendering loop. It owns winit window
//! lifecycle, input dispatch, event bus, frame clock, fixed-step simulation
//! timing, camera/controller state, and `CameraView` construction. The renderer
//! owns Vulkan frame submission, asset pumping, swapchain lifecycle, and
//! platform/UI side effects.
//!
//! Run from the repository root:
//! ```sh
//! cargo run --manifest-path examples/guide_app/Cargo.toml
//! ```

use std::time::Duration;

use engine::input;
use engine::prelude::*;
use engine::render::RendererError;
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::keyboard::KeyCode;
use winit::window::WindowBuilder;

const APP_NAME: &str = "Guide App";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Info)
        .init();

    // --- Platform window ---
    let event_loop = EventLoop::new()?;
    let window = WindowBuilder::new()
        .with_title(APP_NAME)
        .with_inner_size(winit::dpi::LogicalSize::new(1280, 720))
        .build(&event_loop)?;

    // --- Renderer ---
    let config = RendererConfig {
        app_name: "guide_app".to_string(),
        window_width: 1280,
        window_height: 720,
        preload_startup_scene: true,
        ..Default::default()
    };
    let mut renderer = Renderer::new(config, &window)?;
    let mut scene = renderer.take_startup_scene().unwrap_or_else(Scene::new);

    // --- App-owned runtime state ---
    let mut events = runtime_event_bus();
    let mut input = InputSystem::new();
    let mut frame_clock = FrameClock::new();
    let mut action_events = InputActionEventEmitter::new();
    let mut fixed_clock = FixedStepClock::new(FixedStepConfig {
        step: Duration::from_secs_f32(1.0 / 60.0),
        max_steps_per_frame: 10,
    });

    let mut camera = Camera::default();
    let mut fps_controller = FPSController::new(0.002, 1.0);

    // Install FPS action bindings in a named layer.
    {
        let mut map = input::ActionMap::new();
        map.bind_key("move.forward", KeyCode::KeyW);
        map.bind_key("move.backward", KeyCode::KeyS);
        map.bind_key("move.left", KeyCode::KeyA);
        map.bind_key("move.right", KeyCode::KeyD);
        map.bind_key("move.up", KeyCode::Space);
        map.bind_key("move.down", KeyCode::ShiftLeft);

        input.add_layer(
            input::LayerDescriptor::new("guide-fps", input::LayerPriority(10)),
            map.into_layer(),
        );
    }

    let mut last_window_size = window.inner_size();
    window.request_redraw();

    // --- Event loop ---
    event_loop.run(move |event, elwt| {
        elwt.set_control_flow(ControlFlow::Poll);

        // Route platform input through renderer, queue uncaptured app input.
        match engine::input::route_platform_input_to_app(&mut renderer, &window, &mut input, &event)
        {
            Ok(_) => {}
            Err(e) => {
                eprintln!("input routing failed: {e}");
                elwt.exit();
                return;
            }
        }

        match event {
            Event::WindowEvent { event, window_id } if window_id == window.id() => {
                match event {
                    WindowEvent::CloseRequested => elwt.exit(),

                    WindowEvent::Resized(new_size) => {
                        last_window_size = new_size;
                        if let Err(e) = renderer.resize(new_size.width, new_size.height) {
                            eprintln!("resize failed: {e}");
                            elwt.exit();
                        }
                    }

                    WindowEvent::ScaleFactorChanged {
                        mut inner_size_writer,
                        ..
                    } => {
                        let new_size = window.inner_size();
                        if let Err(e) = inner_size_writer.request_inner_size(new_size) {
                            eprintln!("scale-factor size request failed: {e}");
                            elwt.exit();
                            return;
                        }
                        last_window_size = new_size;
                        if let Err(e) = renderer.resize(new_size.width, new_size.height) {
                            eprintln!("resize failed after scale change: {e}");
                            elwt.exit();
                        }
                    }

                    WindowEvent::RedrawRequested => {
                        // Catch up to the current window size before rendering.
                        let current_size = window.inner_size();
                        if current_size != last_window_size {
                            last_window_size = current_size;
                            if let Err(e) = renderer.resize(current_size.width, current_size.height)
                            {
                                eprintln!("resize failed while redrawing: {e}");
                                elwt.exit();
                                return;
                            }
                        }

                        // --- Begin app frame ---
                        let begin = begin_app_frame(
                            &mut input,
                            &mut action_events,
                            &mut events,
                            &mut frame_clock,
                        );

                        // --- Fixed-step update ---
                        let fixed_update = fixed_clock.update(begin.frame.delta);
                        if fixed_update.dropped_time > Duration::ZERO {
                            eprintln!(
                                "dropped {:.3}ms of simulation time",
                                fixed_update.dropped_time.as_secs_f64() * 1000.0
                            );
                        }
                        let simulated_seconds = (1.0 / 60.0) * fixed_update.steps as f32;
                        fps_controller.update_from_snapshot(
                            input.snapshot(),
                            simulated_seconds,
                            &mut camera,
                        );

                        // --- Build CameraView from app-owned camera ---
                        let view =
                            camera_view_for_size(&camera, current_size.width, current_size.height);

                        // --- Pump assets and render ---
                        if let Err(e) = renderer.pump_asset_tasks(32) {
                            eprintln!("asset pump failed: {e}");
                            elwt.exit();
                            return;
                        }

                        match renderer.render_scene_with_view(&mut scene, view) {
                            Ok(FrameRenderOutcome::Rendered) => {
                                // Normal frame; continue.
                            }
                            Ok(FrameRenderOutcome::SkippedResizePending) => {
                                eprintln!("render skipped while swapchain resize is pending");
                            }
                            Ok(FrameRenderOutcome::SkippedAcquireUnavailable)
                            | Ok(FrameRenderOutcome::SubmittedNotPresented)
                            | Ok(FrameRenderOutcome::PresentedSuboptimal) => {
                                // Transient; continue.
                            }
                            Err(RendererError::DeviceLost) => {
                                eprintln!("Vulkan device lost; exiting");
                                elwt.exit();
                                return;
                            }
                            Err(RendererError::BackendPoisoned(msg)) => {
                                eprintln!("renderer backend poisoned: {msg}");
                                elwt.exit();
                                return;
                            }
                            Err(e) => {
                                eprintln!("render failed: {e}");
                                elwt.exit();
                                return;
                            }
                        }

                        // --- End app frame ---
                        end_app_frame(&mut events, begin.frame.index);

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
    })?;

    Ok(())
}
