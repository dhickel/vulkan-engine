# Target Design

Date: 2026-07-07
Status: execution target

## Design Summary

Create a thin root `engine` library facade and app-loop helper layer. Keep support crates raw and independent. Split renderer runtime APIs so new app/facade flows can:

1. route platform/UI/debug side effects through renderer without queuing app input into renderer-owned `InputSystem`;
2. dispatch app-owned `InputSystem` exactly once per frame;
3. emit input action events into app-owned `EventBus`;
4. update app-owned camera/player state;
5. pass a renderer-owned/lower `CameraView` or `RenderView` DTO into renderer for submission.

Legacy renderer-owned lifecycle paths remain temporarily for examples and compatibility.

## Crate And Module Shape

Root `engine` package:

- Add `src/lib.rs`.
- Add or expose modules:
  - `engine::prelude`
  - `engine::input`
  - `engine::events`
  - `engine::render`
  - `engine::camera`
  - `engine::runtime` or equivalent frame-clock/runtime helper module
- Preserve existing `src/main.rs`, `src/launch.rs`, and existing launcher behavior.
- Re-export raw primitives; avoid sealing support crates behind wrapper-only APIs.

Renderer crate:

- Add renderer-facing view DTO in `renderer` API, e.g. `CameraView` or `RenderView`.
- Export it through `renderer::api`, `renderer::prelude`, and `renderer` crate root as appropriate.
- Add no-dispatch/no-camera-ownership frame/render APIs.
- Retain legacy renderer-owned methods as compatibility.

Input crate:

- Keep `InputSystem::dispatch_frame()` semantics unchanged.
- Add tests if root runtime helpers expose new action-event bridge behavior and transient guarantees.

Events crate:

- Keep dependency independence.
- Add tests only if root runtime helper code needs event ordering assertions in root or events crate.

Dogfood app:

- Own `InputSystem`, `EventBus`, frame clock, and camera/player state in app runtime.
- Continue using renderer for Vulkan rendering, assets, debug UI/capture, and resize.

## Suggested Public Types

Renderer/lower DTO:

```rust
pub struct CameraView {
    pub view: glam::Mat4,
    pub projection: glam::Mat4,
    pub position: glam::Vec3,
}
```

Root runtime helpers:

```rust
pub struct FrameClock {
    // tracks frame index, last instant, and delta seconds
}

pub struct FrameInfo {
    pub index: u64,
    pub delta_seconds: f32,
}

pub struct InputActionEventEmitter {
    // owns observed action values for one app input stream
}
```

Convenience bundle is allowed only if it stays small and optional:

```rust
pub struct RuntimeParts {
    pub input: input::InputSystem,
    pub events: engine_events::EventBus,
    pub frame_clock: FrameClock,
    pub action_events: InputActionEventEmitter,
}
```

Do not make such a bundle the only path to use the engine.

## Renderer New Path

Required new capability:

- prepare/pump renderer state without dispatching app input;
- render a scene with caller-provided `CameraView`/`RenderView`;
- preserve capture/debug UI/context updates;
- maintain resize skip behavior;
- support headless rendering without window/UI/cursor requirements.

Acceptable API shape can be refined by worker, but the plan expects functions equivalent to:

```rust
impl Renderer {
    pub fn prepare_render_only_frame(&mut self, window: Option<&winit::window::Window>) -> Result<FramePrepareOutcome, RendererError>;

    pub fn render_scene_with_view(
        &mut self,
        scene: &mut Scene,
        view: CameraView,
    ) -> Result<FrameRenderOutcome, RendererError>;
}
```

If `FramePrepareOutcome` remains private, expose a public equivalent or wrap it in a public frame context. The important contract is mechanical separation from legacy `prepare_frame()` and `prepare_frame_headless()` input/event/camera behavior.

## Platform Event Routing

Split old `Renderer::update_input()` into:

- renderer-owned platform/UI/debug/capture handling;
- app-owned input routing decision/result.

The new path should let the app know whether to queue keyboard/mouse events into app-owned `InputSystem`. It must preserve:

- ImGui platform event forwarding;
- F1/F2 debug UI toggles;
- F12 manual capture;
- cursor entered/left policy;
- `DeviceEvent::MouseMotion`;
- mouse wheel;
- keyboard repeat filtering as currently intended;
- window-id filtering for window events.

Legacy `update_input()` remains a wrapper that uses the split helper and queues into renderer-owned input for old examples.

## Frame Loop Contract

New windowed app frame:

```text
winit event:
  renderer handles platform UI/debug/capture side effects
  if route permits, app queues winit/device input into app-owned InputSystem

redraw:
  frame = frame_clock.tick()
  input.dispatch_frame()
  action_event_emitter.emit(input.snapshot(), events, frame.index)
  events.drain_stage(EventStage::Input)
  update app camera/player/gameplay
  emit/drain lifecycle stages from app-owned bus as appropriate
  view = app_camera.to_camera_view(renderer aspect or viewport size)
  renderer render-only frame/render_scene_with_view
```

New headless app frame:

```text
frame = frame_clock.tick()
input.dispatch_frame() if the app queues scripted/headless input
emit input events once
update camera/gameplay
render_scene_with_view_headless or no-window render-only path
```

## Lifecycle/Event Ownership

- One app-owned `EventBus` owns app lifecycle/input/audio events for a migrated app path.
- Renderer-owned bus remains only for compatibility renderer methods.
- `FrameStarted`/`FrameEnded` producer must be defined for new path to avoid duplicates. Preferred: root/app runtime emits lifecycle events; renderer does not emit app lifecycle events on no-dispatch path.
- Renderer may expose render/capture status through return values or renderer-specific callbacks, not by requiring app mutation through renderer-owned bus.

## Compatibility Contract

Legacy renderer APIs must keep working during this refactor, even if docs mark them legacy:

- `Renderer::update_input(...)`
- `Renderer::begin_frame(...)`
- `Renderer::render_scene(...)`
- `Renderer::render_scene_headless(...)`
- `Renderer::render_scene_in_frame(...)`
- `Renderer::end_frame(...)`
- `Renderer::with_frame(...)`
- `Renderer::camera_position(...)`
- `Renderer::set_camera_position(...)`
- `Renderer::set_camera_look_at(...)`
- `Renderer::install_default_fps_input(...)`
- `Renderer::events()` and `events_mut()`

New app/facade examples and dogfood must not call compatibility APIs that dispatch renderer-owned input, update renderer-owned camera state, or emit/drain renderer-owned app events.

## Documentation And Spec Target

Closeout must update:

- `.internal-dev/specifications/architecture.md`
- `.internal-dev/specifications/service-graph.md`
- `.internal-dev/specifications/services.md`
- `.internal-dev/specifications/api.md`
- `.internal-dev/specifications/decisions.md`
- `.internal-dev/knowledge/renderer-camera-override-behavior.md`
- relevant `docs/api/` and `docs/internal/` files
- `.internal-dev/changelogs/<date>-engine-runtime-abstractions-issues-35-37.md`

Spec update should explicitly distinguish:

- new app-owned runtime path as intended truth;
- legacy renderer-owned lifecycle helpers as temporary compatibility;
- raw primitive support as a supported contract.
