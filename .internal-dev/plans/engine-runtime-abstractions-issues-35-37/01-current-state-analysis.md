# Current State Analysis

Date: 2026-07-07
Status: planning analysis

## Verified Inputs Read

- `AGENTS.md`
- `.internal-dev/specifications/AGENTS.md`
- `.internal-dev/specifications/architecture.md`
- `.internal-dev/specifications/service-graph.md`
- `.internal-dev/specifications/services.md`
- `.internal-dev/specifications/api.md`
- `.internal-dev/specifications/decisions.md`
- `.internal-dev/specifications/workflow.md`
- `.internal-dev/knowledge/renderer-camera-override-behavior.md`
- `src/renderer/AGENTS.md`
- `src/input/AGENTS.md`
- `src/events/AGENTS.md`
- Plan inputs in this directory:
  - `00-ironed-out-brief.md`
  - `research/01-brainstorm-synthesis.md`
  - `feedback/01-feedback-synthesis.md`
  - `research/02-preflight-validation-drift.md`

## Specification Drift To Resolve

- `architecture.md` currently says root `engine` is a migration stub. The target is root bin+lib facade while preserving launcher behavior.
- `services.md` currently says renderer owns camera state used for rendering. The target is app-owned camera with renderer-owned camera retained only for compatibility paths.
- `api.md` already documents `Renderer::set_camera_look_at`, but code does not expose it. This is pre-existing drift and blocks `cargo check -p renderer --examples`.
- `renderer-camera-override-behavior.md` says scene camera is overwritten by renderer internal camera. The target is to update this as legacy behavior and document the new caller-provided view path.

## Current Crate Graph

- Root `engine` depends on `renderer`, `engine_events`, `launch_shared`, `winit`, logging crates.
- `renderer` depends on `input` and `engine_events`.
- `input` re-exports `engine_events::ActionId`.
- `engine_events` is independent by design.
- `dungeon_dogfood` depends on renderer and audio and currently uses renderer as a lifecycle owner.

Target graph must preserve one-way dependency from app/root facade code to support crates. App and example crates may depend on root `engine` when they consume the public facade, and may still use raw support crates directly. `renderer`, `input`, `engine_events`, `audio`, `physics`, `scripting`, `launch_shared`, tools, and pack tooling must not depend on root `engine`.

## Renderer Runtime Coupling

`src/renderer/src/api/renderer.rs` currently owns these coupled responsibilities:

- `input_system: InputSystem`
- `event_bus: EventBus`
- `observed_action_values` for input action event emission
- `camera: Camera`
- `fps_plugin: Option<FpsInputPlugin>`
- frame timing fields
- UI/debug/capture event handling
- cursor policy
- Vulkan runtime and render submission

Current combined lifecycle:

- `Renderer::update_input(window, event)` forwards ImGui events, handles F1/F2/F12 renderer hotkeys, applies cursor focus policy, checks UI capture, and queues uncaptured events into renderer-owned `InputSystem`.
- `prepare_frame(window)` and `prepare_frame_headless()` compute delta time, call `input_system.dispatch_frame()`, emit input action events into renderer-owned bus, drain `EventStage::Input`, advance the internal FPS camera, and perform window/imgui preparation.
- `begin_frame`, `render_scene`, `render_scene_headless`, and one-shot frame lifecycle all depend on the legacy renderer-owned input/event/camera model.
- `render_scene_internal` computes view/projection from renderer-owned `Camera`, calls `scene.update_camera(...)`, then builds the render submission.

## Camera State

- `src/renderer/src/data/camera.rs` already has `Camera::look_at(...)` and tests.
- `Renderer` exposes `camera_position()` and `set_camera_position(...)`.
- `Renderer::set_camera_look_at(...)` is documented and used by capture examples but missing from `renderer.rs`.
- Knowledge note confirms renderer currently overwrites scene camera every frame from internal camera state.

Implementation risk: simply adding root facade re-exports will not close #35-#37 unless the new render path avoids renderer-owned camera overwrite.

## Input State

- `src/input/src/lib.rs` owns frame-buffered event ingest and `InputSystem::dispatch_frame()` as the frame boundary.
- Snapshot transients include `just_*`, mouse delta, scroll delta, and action press/release.
- Double dispatch can erase transients and is the central migration risk.
- Existing renderer unit tests cover `emit_input_action_events_from_snapshot`, but that helper is private and still belongs to renderer in current code.

Implementation risk: tests that only inspect held keys will miss double-dispatch regressions.

## Event State

- `src/events/src/lib.rs` owns independent typed event vocabulary, `EventBus`, staged dispatch, monotonic sequence, and recorder.
- Renderer owns one internal bus today and emits/drains lifecycle and input events internally.
- Root runtime currently has a separate private `RuntimeEvents` wrapper for launcher project/scene lifecycle events.
- Dogfood audio bridge emits audio events through `renderer.events_mut()`.

Implementation risk: app runtime can accidentally split one frame across two buses or duplicate `FrameStarted`/`FrameEnded`.

## Dogfood Coupling

Active dogfood path currently:

- installs renderer default FPS input;
- calls `renderer.update_input(&window, &event)`;
- calls `renderer.begin_frame(window)`;
- reads intended movement from `renderer.camera_position()`;
- resolves collision into `PlayerState`;
- writes back with `renderer.set_camera_position(player.position)`;
- renders with `renderer.render_scene_in_frame(...)`;
- uses `renderer.events_mut()` in audio startup probe.

Minimum migration proof is to move input, events, camera, and frame clock ownership into dogfood/app runtime without changing content, collision, audio semantics, level loading, or asset setup beyond ownership plumbing.

## Baseline Drift

From `research/02-preflight-validation-drift.md`:

- `cargo check -p renderer --examples` fails because capture test code calls missing `Renderer::set_camera_look_at`.
- `cargo check -p dungeon_dogfood` fails with `can't find crate for audio` despite the path dependency and independent `cargo check -p audio` passing.
- These must be handled before clean regression claims.

## Architecture Fit

The intended target fits the existing crates if it stays thin:

- Root `engine` facade is acceptable as bin+lib because the package is already named `engine`.
- Renderer/lower DTO placement avoids reverse dependency.
- Input and events are already standalone primitives suitable for app-owned runtime.
- Camera math can remain in renderer initially because the immediate issue is lifecycle ownership, not moving every camera type.

The target does not fit if implementation creates a root-owned framework that support crates must know about, or if the new facade path merely wraps the existing renderer-owned lifecycle.
