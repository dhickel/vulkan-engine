# Brainstorm Synthesis: Engine Runtime Abstractions for Issues #35-#37

Date: 2026-07-07
Repository: `/home/hickelpickle/Code/Rust/engine`
GitHub issues: #35, #36, #37

## Objective

Refactor the current renderer-owned `InputSystem`, `Camera`, and `EventBus` responsibilities into lightweight engine-level runtime abstractions. The result should make the app/game-loop layer the owner of input dispatch, camera state, event stage dispatch, and frame timing while keeping the renderer focused on Vulkan-backed rendering, asset/render submission, debug UI rendering, capture, and backend orchestration.

The solution should be approachable for a mid-level hobbyist engine: a small facade and direct raw primitives, not an ECS, scheduler, plugin framework, or broad engine rewrite.

## First-Pass Agent Consensus

Six first-pass reports converged on these points:

- The existing root package is already named `engine`; prefer adding `src/lib.rs` as the lightweight facade while keeping `src/main.rs` as the launcher unless planning proves a separate `engine_core` crate is worth the naming and dependency overhead.
- `renderer` should not be the lifecycle owner for `InputSystem`, one global `Camera`, or `EventBus`.
- The `input` and `engine_events` crates are already shaped as raw standalone primitives and should stay directly usable.
- Camera math/controllers can remain where they are initially or be re-exported through the root facade; the first refactor should solve ownership and render-time camera injection before moving every camera type.
- The facade should be a pleasant common path and a re-export surface, not a sealed abstraction that hides raw crates.
- Do not introduce ECS, scheduler/resources, plugin registry, hot reload, broad Vulkan rewrite, or a required monolithic engine object.

## Current Coupling

Observed live code anchors:

- `Renderer` owns `InputSystem`, `EventBus`, `Camera`, FPS plugin, frame lifecycle, assets, capture scheduler, debug UI, cursor policy, and Vulkan runtime in `src/renderer/src/api/renderer.rs`.
- `Renderer::update_input()` owns the `winit -> input` bridge and ImGui/debug UI capture filtering.
- `Renderer::prepare_frame()` and `prepare_frame_headless()` call `InputSystem::dispatch_frame()`, emit input action events, drain the input event stage, and update FPS camera state.
- `render_scene_internal()` derives view/projection/position from the renderer-owned camera and writes that into `Scene`, overriding scene-level camera state.
- `apps/dungeon_dogfood` lets the renderer advance camera state, pulls `renderer.camera_position()`, applies collision/player guards, then writes back with `renderer.set_camera_position()`.
- `apps/dungeon_dogfood/src/audio_bridge.rs` accesses events through `renderer.events_mut()`.

## Recommended Target Shape

Use the root `engine` package as a bin+lib package:

- `src/lib.rs`
- `src/runtime.rs` or `src/runtime/mod.rs`
- `src/camera.rs` or `src/camera/mod.rs` for re-export/wrapper only if needed
- `engine::prelude`
- `engine::input`, `engine::events`, `engine::render`, `engine::camera` re-export modules

Recommended primitive ownership:

- App/game-loop layer owns `InputSystem`.
- App/game-loop layer owns `EventBus`.
- App/game-loop layer owns one or more cameras.
- Renderer accepts render-time camera/view data and emits or returns renderer events through caller-owned event plumbing.
- Renderer keeps compatibility helpers temporarily, but new examples and apps should use app-owned input/events/camera.

Recommended minimal runtime types:

```rust
pub struct EngineRuntime {
    pub input: input::InputSystem,
    pub events: engine_events::EventBus,
    pub frame: FrameClock,
    pub default_camera: Camera,
}

pub struct FrameInfo {
    pub index: u64,
    pub delta_seconds: f32,
}

pub struct RenderCamera {
    pub view: glam::Mat4,
    pub projection: glam::Mat4,
    pub position: glam::Vec3,
}
```

Recommended renderer API direction:

```rust
impl Renderer {
    pub fn render_scene_with_camera(
        &mut self,
        scene: &mut Scene,
        camera: RenderCamera,
    ) -> Result<FrameRenderOutcome, RendererError>;
}
```

Exact naming is not locked. `RenderCamera`, `CameraView`, or `RenderView` are all acceptable if used consistently. The important contract is that renderer consumes camera data at render time rather than owning the app camera lifecycle.

## Loop Model

Recommended simple frame loop:

```text
winit event:
  renderer handles UI/debug/capture side effects that truly belong to renderer
  engine runtime queues input events if not captured by UI policy

redraw/frame:
  begin frame clock
  input.dispatch_frame exactly once
  emit input action events into caller-owned EventBus
  drain EventStage::Input
  variable update: camera look, UI, non-physics gameplay
  fixed update loop: capped physics/gameplay steps
  post-update/collision correction
  renderer.render_scene_with_camera(scene, camera view)
  emit/drain render and post-update lifecycle events at explicit boundaries
```

Camera look/mouse rotation should stay variable-rate unless a later design chooses otherwise. Physics/game rules can use a fixed-step accumulator with a catch-up cap. This keeps latency low and avoids making the renderer the loop owner.

## Raw Primitive Policy

Keep these directly importable:

- `input::InputSystem`, `InputSnapshot`, layers, priorities, action maps.
- `engine_events::EventBus`, `EventStage`, `EngineEvent`, IDs, recorder.
- `renderer::Renderer`, `RendererConfig`, `Scene`, assets/capture types.
- Camera math/controller types, either from their current renderer module or through `engine::camera`.
- Renderer procedural primitive data such as `ProceduralVertex` and `ProceduralMeshData`.

Do not widen normal facade access to Vulkan internals or unstable advanced interop.

## Key Red-Team Risks

- Double-dispatching input can erase `just_*`, mouse delta, scroll delta, and action transients if both facade and renderer call `dispatch_frame()`.
- Event ordering can drift if input events remain pending until after gameplay/render, if both facade and renderer emit duplicates, or if multiple buses split sequence ordering.
- Renderer-owned camera may keep overwriting scene camera state after app camera ownership moves outward.
- ImGui/debug UI capture semantics can regress if winit event ingestion moves out while renderer still owns UI capture state.
- Winit event handling must preserve `DeviceEvent::MouseMotion`, resize/scale events, redraw ordering, F1/F2/F12 overlays, manual capture, cursor grab, and headless behavior.
- A monolithic `Engine { renderer, input, events, camera }` can create borrow conflicts and simply move the god object up one layer.
- Tests can falsely pass if they only check held keys, compile status, camera math, or static headless smoke.

## Known Validation Drift Before Implementation

First-pass architecture agent reported:

- `cargo test -p input --lib` passed.
- `cargo test -p engine_events --lib` passed.
- `cargo check -p renderer --examples` failed on missing `Renderer::set_camera_look_at`.
- `cargo check -p dungeon_dogfood` failed on missing `audio` crate plus follow-on inference errors.

Local specs currently conflict with desired target:

- `.internal-dev/specifications/services.md` states renderer owns camera state used for rendering.
- `.internal-dev/specifications/api.md` documents renderer-owned `set_camera_look_at` behavior.
- `.internal-dev/knowledge/renderer-camera-override-behavior.md` records that renderer camera overrides scene camera in headless mode.

The plan must treat these as intentional drift to resolve, not hidden implementation details.

## Recommended Phase Direction

1. Preflight validation/drift cleanup:
   - Confirm current compile failures and decide whether `set_camera_look_at` must be restored before or during the camera render-view phase.
   - Record pre-existing validation blockers separately from new refactor regressions.
2. Add root `engine` library facade:
   - Re-export raw primitives and add lightweight runtime/frame-clock types without changing behavior.
3. Add renderer render-time camera input:
   - Introduce `RenderCamera`/`CameraView` and `render_scene_with_camera`.
   - Keep old renderer camera methods as compatibility wrappers temporarily.
4. Move input and event stage driving outward:
   - Runtime owns `InputSystem`, emits action events, drains event stages.
   - Renderer stops dispatching caller-owned input for new paths.
5. Migrate `dungeon_dogfood`:
   - Own input/events/camera through the facade.
   - Remove camera push/pull and `renderer.events_mut()` usage.
6. Migrate or wrap examples/root runtime:
   - Preserve renderer examples as smoke paths while adding at least one facade-first example or launcher path.
7. Specs/docs/changelog closeout:
   - Update architecture, service graph, services, API specs, docs, changelog, and bug/issue references.

## Validation Themes

Required compile/test gates:

```sh
cargo check
cargo check -p input
cargo test -p input
cargo check -p engine_events
cargo test -p engine_events
cargo check -p renderer --examples
cargo test -p renderer
cargo check -p dungeon_dogfood
cargo check -p marching_terrain
```

Required behavioral checks:

- One `InputSystem::dispatch_frame()` per facade frame path.
- Input transients survive exactly one frame and are not erased by legacy renderer calls.
- Input action event emission is single-source and ordered.
- Event sequence ordering remains monotonic through one caller-owned `EventBus`.
- Renderer uses caller-provided camera data for new render paths.
- Dogfood player/camera collision correction writes to app-owned camera state before render.
- UI capture prevents gameplay/FPS input where it did before.
- Headless capture camera sidecars distinguish old compatibility path from new caller-provided camera path.

Runtime smoke after compile gates:

```sh
RUST_LOG=debug timeout --signal=INT 60s cargo run -p renderer --example api_test
RUST_LOG=debug timeout --signal=INT 60s cargo run -p dungeon_dogfood
```

Use headless capture validation when visible renderer/camera behavior is claimed.

## Open Decisions For Feedback Pass

- Should implementation add `src/lib.rs` to the root `engine` package or add a separate `engine_core` workspace member?
- Should `Camera` move out of `renderer` now, or remain there and be re-exported while ownership changes?
- Should `Renderer::update_input()` remain as a legacy helper, or should it split into UI/capture handling plus input routing helper immediately?
- Should `Renderer::begin_frame()` continue to dispatch internal input for compatibility, or should a new no-dispatch frame path be introduced first?
- What is the minimum dogfood migration needed to close #35-#37 without rewriting gameplay?
- How should pre-existing compile drift (`set_camera_look_at`, dungeon audio dependency) be handled in the advanced plan gates?
