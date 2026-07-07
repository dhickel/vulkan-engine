# Phase 01 Worker Directive: Root Engine Library Facade

Status: ready after Phase 00 validation
Validation report: `.internal-dev/plans/engine-runtime-abstractions-issues-35-37/validation/phase-01-validation-report.md`

## Objective

Add a thin root `engine` library facade, raw primitive re-export modules, and frame/helper scaffolding without changing renderer, dogfood, or launcher behavior.

## User-Visible Outcome

Users can import common primitives from `engine::prelude::*`, `engine::input`, `engine::events`, `engine::camera`, and `engine::render` while existing launcher commands still work.

## Direct Editable Targets

- `Cargo.toml`
- `src/lib.rs`
- `src/main.rs`
- `src/launch.rs`
- `src/runtime.rs`
- new root modules as needed:
  - `src/camera.rs`
  - `src/events.rs`
  - `src/input.rs`
  - `src/render.rs`
  - `src/frame.rs` or `src/runtime/mod.rs`
- root crate tests if added

## Forbidden Scope

- Do not modify renderer frame/render behavior.
- Do not move dogfood to the new facade.
- Do not move camera files out of renderer.
- Do not create a new workspace crate unless a real dependency cycle is proven and escalated.
- Do not add a public god object as the only supported path.

## Supporting Docs To Read

- `00-specification-lock.md`
- `02-target-design.md`
- `shared/senior-engineer-guidance.md`
- `.internal-dev/specifications/architecture.md`
- `.internal-dev/specifications/service-graph.md`
- root `AGENTS.md`

## Ordered Steps

1. Add root `src/lib.rs` exposing thin modules and `prelude`.
2. Re-export raw primitives:
   - `input::InputSystem`, snapshots, events, action maps, layers;
   - `engine_events::EventBus`, stages, events, IDs, recorder;
   - `renderer::Renderer`, `RendererConfig`, `Scene`, handles, capture/config types;
   - renderer camera math/controller types through `engine::camera`.
3. Add small frame-clock helper types such as `FrameClock` and `FrameInfo`.
4. If adding optional convenience runtime parts, keep it a simple bundle of public raw primitives and avoid borrow-heavy orchestration methods.
5. Preserve binary launcher behavior. If sharing `launch` or root runtime modules between bin and lib, ensure there is one coherent module path.
6. Add mandatory facade import proof:
   - either root integration tests or a small compile-only example/test that imports `engine::prelude::*`, `engine::input`, `engine::events`, `engine::camera`, and `engine::render`;
   - prove original raw crates remain directly importable in the same test or a paired compile-only test.
7. Add focused tests for frame clock monotonicity if useful.

## Senior-Engineer Guidance

- This phase is about surfacing stable paths, not changing ownership behavior yet.
- Root `engine` can re-export renderer camera types now; that does not imply renderer owns app camera in the new path.
- Avoid `pub use renderer::prelude::*` as the entire root facade if it hides module ownership. Prefer explicit modules.
- Check crate graph using `cargo tree -p renderer` and `cargo tree -p engine` if dependency direction is unclear.

## Acceptance Criteria

- `engine` compiles as both library and binary package.
- Existing root launcher usage remains intact.
- Raw primitives are accessible from root facade modules and their original crates.
- Facade import proof and raw-crate direct import proof exist and pass.
- No support crate depends on root `engine`.
- No runtime behavior changes outside facade availability.

## Negative Checks

- No new `engine_core` crate.
- No renderer dependency on root `engine`.
- No new facade API requiring renderer-owned input/camera/event state.
- No dogfood migration.

## Validation Commands

```sh
cargo check -p engine
cargo test -p engine
cargo check
cargo tree -p renderer
cargo tree -p engine
```

`cargo test -p engine` must include or exercise the facade/raw import proof. If a compile-only example is used instead, run and record the exact command that proves it.

## Evidence Expectations

- Worker notes list exported modules/types.
- Worker notes identify the facade/raw import proof location.
- Validator checks dependency tree output for forbidden reverse edges.
- Validator report records launcher behavior was not intentionally changed.

## Stop Conditions

- Stop if root lib creates duplicate module/type conflicts with `src/main.rs` that require launcher redesign.
- Stop if support crates would need to depend on root `engine`.

## Do Not Close Unless

- Facade compiles.
- Facade/raw import proof exists and passes.
- Crate graph rule is verified.
- Phase 01 validation report is written.
