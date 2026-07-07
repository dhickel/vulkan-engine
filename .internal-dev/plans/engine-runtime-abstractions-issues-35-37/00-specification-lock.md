# Specification Lock: Engine Runtime Abstractions for Issues #35-#37

Date: 2026-07-07
Status: execution-ready planning lock
Work classification: large
Source issues: GitHub #35, #36, #37

## Objective

Move renderer-owned `InputSystem`, app camera, and app `EventBus` lifecycle responsibilities into lightweight root engine/app-loop runtime abstractions while keeping raw primitives directly available.

The root package should become a bin+lib package with a thin `engine` facade unless implementation proves a hard dependency cycle. The renderer remains a support crate and must not depend on root `engine`.

## Locked User-Visible Outcome

- `engine::prelude::*` is the common beginner path.
- `engine::input`, `engine::events`, `engine::camera`, and `engine::render` expose raw support-crate primitives without sealing them.
- Apps can own input, events, frame timing, and camera state in their loop.
- Renderer accepts caller-provided per-frame view data for the new path.
- Legacy renderer-owned lifecycle helpers continue to compile temporarily and are labeled compatibility.

## Acceptance Criteria

- Root `engine` library exists via `src/lib.rs` and preserves existing root launcher behavior in `src/main.rs`, `src/launch.rs`, and `src/runtime.rs`.
- Root facade re-exports raw support crates and renderer primitives without forcing a monolithic `Engine` object.
- Renderer-facing view DTO lives in `renderer` or a lower crate, includes at least `view`, `projection`, and `position`, and is re-exported by root `engine`.
- New renderer path can render using caller-provided view data without dispatching input, emitting input action events, draining app events, advancing an internal FPS camera, or overwriting view data from renderer-owned camera state.
- Input migration happens before event migration and proves exactly one `InputSystem::dispatch_frame()` per new app frame path.
- Input action event emission has one source of truth per new frame path and emits into caller-owned `EventBus`.
- Dogfood active path no longer uses `renderer.events_mut()`, `renderer.camera_position()`, or `renderer.set_camera_position()` after the dogfood migration phase.
- Raw primitives remain directly importable from their crates and through root facade modules.
- Compatibility renderer methods remain available during this refactor: `Renderer::update_input`, `begin_frame`, `render_scene`, `render_scene_headless`, `render_scene_in_frame`, `end_frame`, `with_frame`, `camera_position`, `set_camera_position`, `install_default_fps_input`, `events`, and `events_mut`.
- A changelog and affected specs/docs are updated during closeout.

## Validation Criteria

- Preflight records current baseline drift before regression claims:
  - `cargo check -p renderer --examples`
  - `cargo check -p dungeon_dogfood`
  - `cargo check -p audio`
- Focused checks pass as phases make their gates clean:
  - `cargo check -p input`
  - `cargo test -p input`
  - `cargo check -p engine_events`
  - `cargo test -p engine_events`
  - `cargo check -p renderer`
  - `cargo test -p renderer`
  - `cargo check -p renderer --examples`
  - `cargo check -p dungeon_dogfood`
  - `cargo check -p marching_terrain`
  - `cargo check`
- Runtime smoke runs after compile gates are clean:
  - `RUST_LOG=debug timeout --signal=INT 60s cargo run -p renderer --example api_test`
  - `RUST_LOG=debug timeout --signal=INT 60s cargo run -p dungeon_dogfood`
- Headless capture proof is required only if implementation claims visual/camera output correctness or materially changes visible renderer/camera behavior.
- Phase validators write reports under `.internal-dev/plans/engine-runtime-abstractions-issues-35-37/validation/`.
- Final implementation evidence index is `artifacts/engine-runtime-abstractions-issues-35-37/validation-summary.json`.

## Negative Criteria

- Do not introduce an ECS, scheduler, plugin registry, resource world, hot reload framework, or broad Vulkan backend rewrite.
- Do not add a separate `engine_core`, `engine_runtime`, or other workspace crate unless a real dependency cycle blocks root bin+lib.
- Do not make `renderer`, `input`, `engine_events`, `audio`, `physics`, `scripting`, `launch_shared`, `engine_pack`, tools, or other lower/support crates depend on root `engine`. App and example crates may consume the root `engine` facade while retaining direct raw-crate access.
- Do not place renderer-consumed DTOs only in root `engine`.
- Do not migrate dogfood before a no-dispatch/no-camera-ownership renderer path exists.
- Do not collapse input ownership migration and event bus migration into one broad phase.
- Do not hide `input::InputSystem`, `engine_events::EventBus`, `renderer::Renderer`, `renderer::Scene`, or camera primitives behind facade-only wrappers.
- Do not expose Vulkan backend pointers/cache references through the normal root facade.
- Do not use a facade wrapper that still relies on renderer as the real input/camera/event owner for #35-#37.
- Do not treat pre-existing `set_camera_look_at` or dogfood audio compile drift as regressions from this refactor unless a clean baseline is first established.

## Constraints

- Root `engine` may depend on support crates and `renderer`; app/example crates may depend on root `engine` as facade consumers; support crates, renderer, `launch_shared`, tools, and pack tooling must remain independent of root `engine`.
- Renderer UI/debug/capture platform handling may remain renderer-owned, but app/game input queueing must be app-owned on the new path.
- `DeviceEvent::MouseMotion`, F1/F2/F12 debug/manual-capture hotkeys, cursor policy, resize handling, and headless rendering must remain explicit in the new path.
- Exactly one app-owned `EventBus` owns app lifecycle/input/audio events for one app runtime path.
- Renderer-owned event bus and camera may remain only as legacy compatibility state.
- Headless paths must not require `winit::Window`, ImGui UI state, or cursor state.
- `.internal-dev` closeout must update affected specifications, docs, knowledge, changelogs, and plan validation artifacts.

## Assumptions To Verify

- Adding `src/lib.rs` to the existing `engine` package will not create a dependency cycle.
- `Camera`, `FPSController`, and camera math can remain in `renderer::data::camera` for this refactor and be re-exported.
- Restoring `Renderer::set_camera_look_at` is a narrow preflight compatibility repair, not the new ownership model.
- Dogfood audio compile failure is pre-existing drift until Phase 00 classifies and fixes/quarantines it.
- Runtime smoke may be environment-sensitive; compile and unit validation are the primary gates.

## User-Decision Gates

- Stop for user/main-thread approval if a real dependency cycle prevents root bin+lib and requires a new workspace crate.
- Stop for user/main-thread approval if dogfood audio drift requires a broad audio/package redesign rather than a narrow dependency/build repair.
- Stop for user/main-thread approval if preserving legacy renderer methods conflicts with making the new path correct.
- Stop for user/main-thread approval if validators require model/tool fallback from the requested/default validation route.

## Stop Rules

- Stop implementation and return to planning if a phase directive lacks a concrete no-dispatch path, crate graph rules, validation gate, or legacy compatibility boundary.
- Stop implementation if renderer or a support crate depends on root `engine`.
- Stop implementation if new facade flow dispatches input in both runtime and renderer.
- Stop implementation if one app frame requires two unrelated event buses for lifecycle/input/audio ordering.
- Stop implementation if dogfood migration still pulls or writes active camera state through renderer.
- Stop validation if evidence status claims final pass while any phase validator, runtime smoke, stale-reference sweep, or final quality review is missing.
