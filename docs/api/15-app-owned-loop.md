# App-Owned Loop Primitives

App-owned loops keep application state in the app: input dispatch, frame timing, lifecycle/event bus, camera/controller state, gameplay simulation, and scene mutation. The renderer still owns renderer-only work: Vulkan submission, asset pumping, swapchain/headless targets, resize handling, debug UI/platform side effects, and capture output.

Renderer-owned loops remain available for compatibility and examples. In those loops, apps call `Renderer::update_input(...)` and `Renderer::render_scene(...)`; see [Renderer Lifecycle](02-renderer.md) and [Events and Lifecycle](12-events-and-lifecycle.md).

## Minimal pattern

Snippet Type: Pseudocode / compact Rust

```rust
use engine::camera::Camera;
use engine::events::runtime_event_bus;
use engine::frame::{begin_app_frame, end_app_frame, FrameClock};
use engine::input::{route_platform_input_to_app, InputActionEventEmitter, InputSystem};
use engine::render::{camera_view_for_size, Renderer, RendererConfig, Scene};
use winit::event::{Event, WindowEvent};
use winit::event_loop::EventLoop;
use winit::window::WindowBuilder;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let event_loop = EventLoop::new()?;
    let window = WindowBuilder::new().build(&event_loop)?;

    let mut renderer = Renderer::new(RendererConfig::default(), &window)?;
    let mut scene = renderer.take_startup_scene().unwrap_or_else(Scene::new);
    let mut camera = Camera::default();

    // App-owned runtime state.
    let mut events = runtime_event_bus();
    let mut input = InputSystem::new();
    let mut frame_clock = FrameClock::new();
    let mut action_events = InputActionEventEmitter::new();

    event_loop.run(move |event, elwt| {
        if let Err(error) = route_platform_input_to_app(
            &mut renderer,
            &window,
            &mut input,
            &event,
        ) {
            log::error!("input routing failed: {error}");
            elwt.exit();
            return;
        }

        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => elwt.exit(),

            Event::WindowEvent {
                event: WindowEvent::RedrawRequested,
                ..
            } => {
                let begin = begin_app_frame(
                    &mut input,
                    &mut action_events,
                    &mut events,
                    &mut frame_clock,
                );

                // App update: read input.snapshot(), mutate gameplay state, update scene,
                // and place the app-owned camera after simulation/collision correction.
                update_game(&input, begin.frame.delta_seconds, &mut scene, &mut camera);

                let size = window.inner_size();
                let view = camera_view_for_size(&camera, size.width, size.height);
                if let Err(error) = renderer.render_scene_with_view(&mut scene, view) {
                    log::error!("render failed: {error}");
                    elwt.exit();
                    return;
                }

                let _end = end_app_frame(&mut events, begin.frame.index);
                window.request_redraw();
            }

            Event::AboutToWait => window.request_redraw(),
            _ => {}
        }
    })?;

    Ok(())
}

# fn update_game(
#     _input: &InputSystem,
#     _dt: f32,
#     _scene: &mut Scene,
#     _camera: &mut Camera,
# ) {}
```

## Notes

- `route_platform_input_to_app` preserves renderer platform/UI side effects and queues only uncaptured app input into the caller-owned `InputSystem`.
- `begin_app_frame` ticks the caller-owned `FrameClock`, dispatches input, emits input action events, drains the input stage, and emits/drains `FrameStarted`.
- `end_app_frame` emits/drains `FrameEnded` for the same frame index.
- `camera_view_for_size` sanitizes zero viewport dimensions before constructing the renderer-owned `CameraView` DTO.
- `apps/dungeon_dogfood` is the full real-app proof for this path: it uses the root `engine` helpers while keeping gameplay, collision, camera, audio telemetry, and events app-owned.
