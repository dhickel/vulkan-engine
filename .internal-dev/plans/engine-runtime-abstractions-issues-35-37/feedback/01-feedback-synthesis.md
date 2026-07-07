# Feedback Synthesis

Date: 2026-07-07

## Feedback Pass Summary

The six second-pass roles broadly approved the brainstorm synthesis but tightened the implementation constraints. The most important correction is that the new facade path must be mechanically separate from legacy renderer frame/input paths that still dispatch input or own camera state.

## Locked Direction

- Use the existing root `engine` package as a bin+lib package by adding `src/lib.rs`.
- Keep `src/main.rs`, `src/launch.rs`, and existing root runtime launcher behavior intact.
- Do not add a separate `engine_core`/`engine_runtime` crate unless implementation discovers a real dependency cycle or packaging need.
- Support crates (`input`, `engine_events`, `renderer`, `audio`, `physics`, `scripting`) must not depend on root `engine`.
- `renderer` must not depend on root `engine`.
- Root `engine` may depend on support crates and re-export their raw primitives.

One architecture role preferred a separate workspace crate to preserve the root launcher-stub contract. The plan rejects that for now because five roles preferred root bin+lib and the package is already named `engine`. The architecture concern is handled by preserving root launcher behavior and keeping root `engine` as a facade/re-export/runtime-helper layer only.

## API Placement Decisions

- The renderer-facing view DTO must live in `renderer` or a lower shared crate, not root `engine`, because `renderer` cannot depend on root `engine`.
- Prefer the name `CameraView` or `RenderView` over `RenderCamera` so the type reads as a per-frame view, not a renderer-owned camera.
- The DTO should include at least view matrix, projection matrix, and camera/eye position.
- App-owned camera/controller helpers may be re-exported through `engine::camera`; moving camera files is deferred.

## Legacy Compatibility Rules

These renderer APIs may remain temporarily for compatibility:

- `Renderer::update_input(...)`
- `Renderer::begin_frame(...)`
- `Renderer::render_scene(...)`
- `Renderer::camera_position(...)`
- `Renderer::set_camera_position(...)`
- `Renderer::install_default_fps_input(...)`
- `Renderer::events()` / `events_mut()`

After the new facade path exists, these are legacy renderer-owned lifecycle helpers. New app/facade paths must not call them if they dispatch input, update internal FPS camera state, or rely on renderer-owned camera/event ownership.

The plan must add new no-dispatch/no-camera-ownership frame/render APIs before migrating `dungeon_dogfood`.

## Input Routing Decision

`Renderer::update_input()` currently combines:

- ImGui platform event forwarding.
- Debug UI/manual capture hotkeys.
- UI capture checks.
- Cursor policy side effects.
- App/game input queueing into renderer-owned `InputSystem`.

The new path must split this:

- Renderer may handle platform/UI/debug/capture side effects and report capture/routing intent.
- The app/root runtime queues uncaptured events into an app-owned `InputSystem`.
- `DeviceEvent::MouseMotion` must remain supported.
- `Renderer::update_input()` remains as a legacy wrapper for old examples only.

## Frame Lifecycle Decision

Existing `Renderer::begin_frame()` may keep legacy behavior for compatibility.

New facade paths require a frame/render path that does not:

- Call `InputSystem::dispatch_frame()`.
- Emit input action events from a renderer-owned snapshot.
- Drain a renderer-owned app event bus.
- Update an internal FPS camera from renderer-owned input.
- Overwrite scene camera state from renderer-owned camera data.

Input migration and event migration should be separate phases so input transient correctness can be validated before event ownership changes.

## Runtime/Facade Shape

The root `engine` facade should be thin:

- `engine::prelude`
- `engine::input` re-exports
- `engine::events` re-exports
- `engine::render` re-exports
- `engine::camera` re-exports
- Small frame clock/runtime helper types
- Helpers for input action event emission currently buried in renderer

Avoid a long-term public god object. If an `EngineRuntime`, `RuntimeParts`, or similar convenience bundle exists, it must not be the only path and must not expose APIs that require overlapping mutable borrows of renderer, scene, input, events, and camera.

## Dogfood Migration Scope

Minimum migration to prove issues #35-#37:

- `dungeon_dogfood` owns or obtains app-owned `InputSystem`, `EventBus`, frame clock, and camera/player state through the facade/root runtime.
- Remove active-path `renderer.events_mut()` usage.
- Remove active-path camera pull/write-back via `renderer.camera_position()` and `renderer.set_camera_position()`.
- Preserve level loading, collision world, content pack loading, asset setup, audio semantics, and renderer draw setup where possible.
- Pass caller-owned `CameraView`/`RenderView` into renderer before rendering.

## Required Tests And Validation Details

Input correctness:

- Exactly one `InputSystem::dispatch_frame()` per new app frame path.
- `just_pressed`, `just_released`, mouse delta, scroll delta, action press/release, and same-frame press/release behavior are tested.
- New facade path does not call legacy renderer paths that dispatch input again.
- Resize-skipped frames have an explicit input/event policy and test coverage or documented residual risk.

Event correctness:

- Single caller-owned `EventBus` for one app runtime path.
- Input action events emitted once from one snapshot.
- Event sequence ordering is monotonic.
- `FrameStarted` / `FrameEnded` ownership is defined to avoid duplicates.

Camera/render correctness:

- New renderer path uses caller-provided `CameraView`/`RenderView`.
- Renderer-owned camera no longer overwrites app-owned camera on the new path.
- Projection ownership is explicit.
- Multi-view future is not blocked by a type named or shaped as a single default camera.

UI/window/headless correctness:

- ImGui/debug UI capture still suppresses gameplay/FPS input.
- F1/F2/F12 overlays/manual capture and cursor policy are preserved.
- `DeviceEvent::MouseMotion` is not dropped.
- Headless paths do not require `winit::Window`, UI, or cursor state.

Raw primitive policy:

- Raw primitive access means owned CPU primitives and stable renderer handles.
- Do not expose Vulkan backend pointers/cache refs through the normal facade.

## Stop Conditions

Stop and repair the plan before implementation if:

- The plan lacks an explicit no-dispatch renderer/facade path.
- The plan does not define crate graph rules.
- The plan does not state exact legacy APIs to preserve.
- Input migration and event migration are collapsed into one broad phase.
- UI capture routing is unspecified.
- Pre-existing compile blockers are not classified before regression claims.

Stop during implementation if:

- Any support crate depends on root `engine`.
- Renderer depends on root `engine`.
- A new facade path dispatches input in both runtime and renderer.
- Event ordering requires two unrelated event buses for one app frame.
- Headless runtime requires window/UI state.
- Dogfood still pulls camera state from renderer after the dogfood migration phase.
- Raw primitives become hidden behind a sealed facade.
