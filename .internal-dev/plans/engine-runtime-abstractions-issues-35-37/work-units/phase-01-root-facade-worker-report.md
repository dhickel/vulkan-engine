# Phase 01 Worker Report: Root Engine Library Facade

Date: 2026-07-07
Worker scope: `.internal-dev/plans/engine-runtime-abstractions-issues-35-37/worker-directives/phase-01-root-facade.md`

## Changed Files

- `Cargo.toml` and `Cargo.lock`: added the root package's direct `input` dependency so `engine::input` and raw import proof can compile without relying on a transitive renderer dependency.
- `src/lib.rs`: added root library facade modules and `engine::prelude`.
- `src/input.rs`: added explicit raw input re-exports from the `input` support crate.
- `src/events.rs`: added explicit raw event re-exports from the `engine_events` support crate.
- `src/render.rs`: added explicit renderer facade re-exports, including renderer/runtime, scene, handle, capture, config, validation, asset, and hook types.
- `src/camera.rs`: added renderer camera/math/controller re-export module.
- `src/frame.rs`: added `FrameClock` and `FrameInfo` helper scaffolding plus deterministic unit tests.
- `src/main.rs`: switched the binary to use the root library's `engine::launch` and `engine::runtime` modules so launcher behavior uses one coherent module path.
- `src/runtime.rs`: removed imports that became unused when the launcher runtime is compiled through the root library.
- `tests/facade_imports.rs`: added mandatory compile/runtime import proof for `engine::prelude::*`, `engine::input`, `engine::events`, `engine::camera`, `engine::render`, and direct raw crates.

## Exported Modules And Types

- `engine::prelude`: `Camera`, `FPSController`, `OrbitCamera`, `OrbitController`, `EventBus`, `EventStage`, `FrameId`, `EngineEvent`, `FrameClock`, `FrameInfo`, `InputSystem`, `InputSnapshot`, `InputEvent`, `ActionId`, `Renderer`, `RendererConfig`, `Scene`, `SceneNodeId`, `FrameContext`, `FrameRenderOutcome`.
- `engine::input`: `InputSystem`, `InputRuntime`, `InputSnapshot`, `FrameInputSnapshot`, `InputEvent`, `InputDevice`, `InputLayer`, `InputConsume`, `ActionMap`, `ActionMapLayer`, `ActionBinding`, `ActionId`, `InputChord`, `BindingModifiers`, `BindingTrigger`, `CaptureLayer`, `LayerDescriptor`, `LayerSpec`, `LayerHandle`, `LayerId`, `LayerPriority`, `InputDebugFrame`, `InputDebugSnapshot`, `priority_bands`, `editor_ui_capture_layer`.
- `engine::events`: `EventBus`, `EventStage`, `EngineEvent`, lifecycle/input/scene/asset/physics/audio/scripting event families, IDs, listener/sequence/frame IDs, recorder, envelope, dispatch report, and listener failure/error types.
- `engine::render`: `Renderer`, `RendererConfig`, `Scene`, `FrameContext`, `FrameRenderOutcome`, scene node and render handles, capture/config/scheduler/status types, validation/project/package/asset types, hooks, debug UI context types, asset manager and loading types.
- `engine::camera`: `Camera`, `FPSController`, `OrbitCamera`, `OrbitController`, `Ray`, `Aabb`, `Frustum`.
- `engine::frame`: `FrameClock`, `FrameInfo`.
- `engine::launch` and `engine::runtime`: existing launcher/runtime modules remain available through the root library; binary behavior was not intentionally changed.

## Import Proof

- Facade and raw import proof: `tests/facade_imports.rs`.
- The proof imports and uses `engine::prelude::*`, `engine::{input, events, camera, render}`, and direct raw crates `input`, `engine_events`, and `renderer`.

## Validation Commands

- `cargo check -p engine`: pass. Renderer dependency emitted existing dead-code warnings.
- `cargo test -p engine`: pass. Includes `tests/facade_imports.rs` and `src/frame.rs` unit tests.
- `cargo check`: pass. Renderer dependency emitted existing dead-code warnings.
- `cargo tree -p renderer`: pass, command completed successfully.
- `cargo tree -p engine`: pass, command completed successfully.
- `cargo tree -p renderer | rg "^engine v| engine v|\\bengine v0\\.1\\.0"`: no matches; exit code 1 is expected for no forbidden root-engine dependency in renderer tree.

## Criteria Status

- Satisfied: root `engine` compiles as library and binary package.
- Satisfied: existing launcher path is preserved through shared library modules.
- Satisfied: raw primitives are accessible through root facade modules and original crates.
- Satisfied: facade/raw import proof exists and passes.
- Satisfied: no new `engine_core` crate, no dogfood migration, no renderer frame/render behavior changes by this worker.
- Satisfied: dependency-tree check found no renderer dependency on root `engine`.

## Safe Adjacent Hygiene

- Updated `Cargo.lock` for the new direct root `input` dependency.
- Removed unused imports from `src/runtime.rs` after exposing it through `src/lib.rs`.
- Formatted only files touched by this phase. Full `cargo fmt --check` was not used as a final gate because it reports unrelated formatting drift in the pre-existing modified `src/renderer/src/api/renderer.rs`.

## Residuals And Blockers

- No blockers encountered.
- Pre-existing/unrelated worktree change remains in `src/renderer/src/api/renderer.rs`; this worker did not modify or revert it.
- Phase 06 remains responsible for broader documentation/spec/changelog closeout per the plan suite.
