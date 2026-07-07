# Ironed-Out Brief: Engine Runtime Abstractions for Issues #35-#37

Date: 2026-07-07
Status: planning-ready
Source issues: GitHub #35, #36, #37

## Objective

Move input, event bus, and app camera lifecycle ownership out of `renderer` and into lightweight engine/app-loop runtime abstractions while preserving direct raw primitive access.

The intended outcome is a small root `engine` library facade and runtime helper layer that makes the common path easy without turning the project into an ECS, scheduler, plugin framework, or sealed engine monolith.

## User-Visible Outcome

A mid-level hobbyist user should be able to:

- Import `engine::prelude::*` for the common path.
- Use `engine::input`, `engine::events`, `engine::camera`, and `engine::render` for clear raw primitives.
- Own input, event bus, camera, and frame clock in the app/game-loop layer.
- Pass a per-frame camera/view object into renderer.
- Continue using raw crates (`input`, `engine_events`, `renderer`, `audio`, `physics`, `scripting`) directly when desired.

## Problem Type

Large mixed architecture refactor with API, examples/app migration, docs/spec, and validation work.

## Recommended Approach

Use the existing root package named `engine` as a bin+lib package:

- Add `src/lib.rs`.
- Keep existing root launcher behavior in `src/main.rs`, `src/launch.rs`, and `src/runtime.rs`.
- Add thin modules/re-exports, not a framework.
- Do not create a separate `engine_core` crate unless implementation discovers a real dependency cycle.

Add a renderer-facing `CameraView` or `RenderView` type in the `renderer` API, then re-export it from root `engine`. The type must live in `renderer` or a lower shared crate, not root `engine`, because `renderer` must not depend on root `engine`.

Add a no-dispatch/no-camera-ownership renderer frame/render path before migrating apps. New facade paths must not call legacy renderer methods that still dispatch input or update renderer-owned camera state.

## In Scope

- Root `engine` library facade and prelude.
- Runtime/frame-clock helpers.
- Input action event emission helper extracted from renderer behavior.
- Renderer `CameraView`/`RenderView` API and render path using caller-provided view/projection/position.
- Renderer platform event split for UI/debug/capture side effects versus app-owned input routing.
- No-dispatch frame/render APIs for facade-owned input/camera/event paths.
- Dogfood migration enough to prove ownership changes.
- Compatibility wrappers for legacy renderer examples/root runtime where needed.
- Specs, docs, changelog, plan validation artifacts.
- Preflight handling for current compile drift.

## Out Of Scope

- ECS/world/resources/scheduler.
- Plugin framework or hot reload.
- Broad Vulkan backend rewrite.
- Moving all camera math to a new crate in the first pass.
- Rewriting dogfood gameplay, level loading, collision, asset setup, or audio semantics beyond ownership integration needs.
- Hiding raw support crates behind a sealed facade.
- Widening stable facade access to raw Vulkan internals or cache references.

## Non-Goals And Deferred Ideas

- Multi-camera rendering beyond making the new view type future-compatible.
- Removing all legacy renderer-owned APIs immediately.
- Optimizing event listener sorting or event bus internals unless validation shows it is needed.
- Moving launch/capture shared enums out of renderer unless the implementation hits a direct dependency problem.
- Publishing/versioning crate boundaries.

## Target Surfaces

Primary:

- `Cargo.toml`
- `src/lib.rs`
- `src/main.rs`
- `src/launch.rs`
- `src/runtime.rs`
- `src/renderer/src/api/renderer.rs`
- `src/renderer/src/api/scene.rs`
- `src/renderer/src/api/prelude.rs`
- `src/renderer/src/api/mod.rs`
- `src/renderer/src/lib.rs`
- `src/renderer/src/data/camera.rs`
- `src/input/src/lib.rs`
- `src/events/src/lib.rs`
- `apps/dungeon_dogfood/src/main.rs`
- `apps/dungeon_dogfood/src/audio_bridge.rs`
- `apps/dungeon_dogfood/src/events.rs`
- `apps/dungeon_dogfood/src/player.rs`

Likely secondary:

- `src/renderer/examples/common/mod.rs`
- renderer examples using camera/input/runtime helpers
- `apps/marching_terrain/src/main.rs`
- `apps/marching_terrain/src/capture.rs`
- `src/launch_shared/src/lib.rs`
- `tools/engine_pack/src/main.rs`
- docs under `docs/api/` and `docs/internal/`
- `.internal-dev/specifications/architecture.md`
- `.internal-dev/specifications/service-graph.md`
- `.internal-dev/specifications/services.md`
- `.internal-dev/specifications/api.md`
- `.internal-dev/specifications/decisions.md`
- `.internal-dev/knowledge/renderer-camera-override-behavior.md`

## Source Context Read

- Root `AGENTS.md` instructions supplied in the user message.
- `.internal-dev/AGENTS.md`
- `.internal-dev/specifications/AGENTS.md`
- `.internal-dev/specifications/architecture.md`
- `.internal-dev/specifications/service-graph.md`
- `.internal-dev/specifications/services.md`
- `.internal-dev/specifications/api.md`
- `.internal-dev/specifications/decisions.md`
- `.internal-dev/knowledge/renderer-camera-override-behavior.md`
- First-pass brainstorm reports from six GPT-5.5 agents.
- Second-pass feedback reports from the same six roles.
- Local preflight compile checks and targeted source inspection.

## Constraints

- Preserve unrelated user/local edits.
- Root `engine` may depend on support crates and `renderer`; support crates and `renderer` must not depend on root `engine`.
- New renderer-facing DTOs used by renderer must live in `renderer` or a lower shared crate, then be re-exported by root `engine`.
- New facade path must not call legacy renderer input/camera/event ownership paths.
- Exactly one `InputSystem::dispatch_frame()` per new app frame path.
- Event action emission must have one source of truth per frame.
- One app runtime path should use one caller-owned app `EventBus`.
- Headless paths must not require `winit::Window`, UI, or cursor state.
- UI/debug capture routing must be explicit before input ownership moves.
- Camera projection ownership must be explicit.
- Raw primitive access remains supported and documented.
- Compatibility wrappers can remain temporarily but must be labeled as legacy renderer-owned lifecycle helpers after the new path exists.

## Assumptions

- Adding `src/lib.rs` to the existing package is acceptable despite current architecture spec calling root `engine` a migration stub, because the user asked for a new engine crate/facade and the package is already named `engine`.
- `Camera` math can remain in `renderer` for the first pass and be re-exported.
- `Renderer::set_camera_look_at` drift should be resolved in Phase 0 or in the camera-view phase before renderer examples become clean gates.
- The dogfood `audio` compile failure is pre-existing drift and should be diagnosed separately before dogfood is treated as a clean regression gate.
- Runtime smoke may be environment-sensitive; compile and unit gates come first.

## Gotchas And Risks

- Double input dispatch erases transient input state.
- Leaving renderer camera overwrite active breaks app-owned camera rendering.
- Splitting `Renderer::update_input()` can regress ImGui capture, debug overlay keys, manual capture keys, cursor grab, or `DeviceEvent::MouseMotion`.
- A public `EngineRuntime` bag can become a new god object or create borrow conflicts.
- Event duplication can happen if both renderer and runtime emit lifecycle/input events.
- Tests that only check held keys or compile state can miss transient and ordering bugs.
- Current docs/specs explicitly describe old renderer-owned behavior and must be updated.

## Acceptance Criteria

- Root `engine` library exists and exposes a thin facade/prelude while preserving root launcher behavior.
- Raw primitives remain directly usable.
- New renderer API accepts caller-provided view/projection/position for rendering.
- New facade/app path owns input, events, frame timing, and camera state.
- New facade/app path dispatches input exactly once per frame.
- New facade/app path emits input action events once into caller-owned `EventBus`.
- Dogfood no longer uses `renderer.events_mut()` or renderer camera pull/write-back on its active path after its migration phase.
- Legacy renderer examples remain either compiling or explicitly documented/baselined if blocked by pre-existing drift.
- Specs/docs/changelog reflect the new ownership model.

## Negative Criteria

- No support crate depends on root `engine`.
- Renderer does not depend on root `engine`.
- No new ECS/scheduler/plugin/resource registry.
- No facade-only wrapper that leaves renderer as the real owner for #35-#37.
- No new path requiring users to call both `renderer.update_input()` and app-owned input queueing for the same event.
- No new path requiring users to call renderer-owned camera methods for app camera state.
- No hidden raw backend pointer/cache access in the normal facade.

## Validation Expectations

Baseline/preflight:

```sh
cargo check -p audio
cargo check -p renderer --examples
cargo check -p dungeon_dogfood
```

Focused gates:

```sh
cargo check -p input
cargo test -p input
cargo check -p engine_events
cargo test -p engine_events
cargo check -p renderer
cargo test -p renderer
cargo check -p renderer --examples
cargo check -p dungeon_dogfood
cargo check -p marching_terrain
cargo check
```

Runtime smoke after compile gates:

```sh
RUST_LOG=debug timeout --signal=INT 60s cargo run -p renderer --example api_test
RUST_LOG=debug timeout --signal=INT 60s cargo run -p dungeon_dogfood
```

Headless capture validation is required only when the implementation claims visual/camera output proof or changes visible renderer/camera behavior.

Expected test themes:

- Input transient preservation/reset and no double dispatch.
- Single action-event emission and monotonic event sequence.
- Renderer uses caller-provided camera view on new path.
- UI capture blocks gameplay/FPS input.
- Dogfood camera/collision order remains correct after migration.
- Headless path stays independent of `winit::Window`.

## Open Decisions

No blocking product decisions remain. The advanced plan may refine names (`CameraView` versus `RenderView`) and exact phase decomposition, but it must preserve the constraints above.

## Advanced-Planner Handoff Report

Produce a large phased plan suite under `.internal-dev/plans/engine-runtime-abstractions-issues-35-37/`.

The plan must include:

- Phase 0: baseline drift repair or quarantine for `set_camera_look_at` and dogfood audio compile failure.
- Phase 1: root `engine` lib facade/re-exports/frame helpers with no behavior change.
- Phase 2: renderer-facing `CameraView`/`RenderView` and no-dispatch/no-camera-ownership render/frame path.
- Phase 3: app-owned input dispatch and action event emission helper, with transient tests.
- Phase 4: app-owned event bus/stage dispatch migration for new path.
- Phase 5: dogfood migration to app-owned input/events/camera as proof.
- Phase 6: compatibility cleanup, docs/spec/changelog closeout, and final validation evidence.

Each phase needs:

- Direct editable targets.
- Compatibility expectations.
- Acceptance and negative criteria.
- Validation commands.
- Validation report path under `.internal-dev/plans/engine-runtime-abstractions-issues-35-37/validation/`.
- Stop conditions.

Final validation must include an xhigh senior engineer review/signoff pass after planner output and again after implementation/validation if the plan materially changes.
