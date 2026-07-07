# 01 Current State Analysis

## Verified Repo State

- Workspace members currently include `src/input`, `src/renderer`, `src/audio`, `src/physics`, `src/scripting`, `apps/dungeon_dogfood`, `apps/editor`, and `tools/engine_pack`.
- No event crate exists yet.
- Root binary `engine` has launcher/runtime files in `src/main.rs`, `src/launch.rs`, and `src/runtime.rs`.
- Sprint tracker marks Sprint 05 as the event system and application lifecycle sprint.
- Relevant roadmap Track D states the problem: input, scene commands, assets, physics, audio, scripts, editor UI, and dogfood gameplay need a common event contract.

## Input Contract

- `src/input/src/lib.rs` owns `InputSystem`, `ActionMap`, `ActionId`, `InputSnapshot`, layered dispatch, priority bands, and frame-scoped transient state.
- `InputSystem::dispatch_frame()` is the explicit frame boundary.
- Same-priority input layers all run; consumption blocks only lower-priority layers.
- `InputSnapshot` exposes `action_value`, `action_pressed`, `action_just_pressed`, and `action_just_released`.
- Sprint 05 must bridge input/action events after dispatch/snapshot refresh, not during raw winit event ingest.

## Renderer Contract

- `src/renderer/src/api/renderer.rs` owns `Renderer`, `Renderer::update_input`, `input()`, `input_mut()`, `render_scene`, `render_scene_headless`, explicit frame APIs, asset pumping, and render hooks.
- Renderer facade exports are in `src/renderer/src/api/mod.rs` and `src/renderer/src/lib.rs`.
- Renderer already reexports input types.
- Render hooks exist, but events must not be implemented as render hook side effects or mid-render mutation points.
- Asset work flows through `Renderer::assets()` and `AssetManager`; deferred loading exposes `LoadStatus`.

## Runtime Contract

- `src/runtime.rs` validates project files, loads enabled package manifests, validates startup scenes, loads startup scenes, and runs either windowed or headless loops.
- Headless loop renders via `render_scene_headless` and handles capture status.
- Windowed loop calls `renderer.update_input(&window, &event)` on every event and renders on `RedrawRequested`.
- Runtime has no lifecycle event emission yet.

## Scene Command Contract

- `src/renderer/src/scene/command.rs` exposes `CommandHistory` and `CommandResult` with `description`, `node_remap`, and `created_node`.
- `PlaceAssetCommand` is the obvious scene asset placement boundary, but Sprint 05 should avoid broad command redesign.
- Scene event emission should be at command/facade/application boundaries where the mutation result is known.

## App/Tool State

- `apps/editor` uses renderer facade, command history, project/package loading, and a headless editor path.
- `apps/dungeon_dogfood` has custom collision/gameplay and direct renderer usage; Sprint 05 should demonstrate subscription/recording without migrating dogfood gameplay.
- `tools/engine_pack` validates packages/projects/scenes and should not become event-runtime dependent unless a tiny docs/sample hook is clearly useful.

## Docs State

- API docs cover renderer lifecycle, scene workflows, assets, hooks, input, engine arguments, packaging CLI, and runtime launcher.
- Internal docs cover architecture, asset lifecycle, API/backend handoff, scene flattening, and input/winit integration.
- No unified event API docs exist.

## Architecture Gaps

- No shared typed event vocabulary exists for engine systems.
- No frame-stage metadata exists to explain when app subscribers may react.
- No event recorder/debug stream exists for diagnosing ordering.
- Root runtime load/shutdown boundaries are not observable by app systems except logs.
- Input action state is pollable but not yet evented.

## Risk Areas

- Borrow/lifetime hazards if event callbacks are invoked while `Renderer` is mutably borrowed.
- Ordering ambiguity between package loaded, scene validated, scene loaded, first frame, and shutdown.
- Double emission if both renderer and app layers emit the same scene/asset event.
- Accidental Vulkan dependency in the core event crate.
- Over-implementation of placeholder physics/audio/scripting events.
- Docs overstating support for systems that only have event type placeholders.
