# Phase 02 Worker Directive: Renderer And Root Runtime Integration

## Objective

Expose the Phase 01 event contract through the renderer facade and root runtime, emit lifecycle/project/scene/asset-ish events at safe boundaries, and bridge input/action snapshots after dispatch without changing input semantics.

## User-Visible Outcome

Apps can subscribe to engine lifecycle and input/action events through supported API surfaces while the root runtime emits observable project/scene/frame/shutdown lifecycle events.

## Editable Targets

- `src/renderer/Cargo.toml`
- `src/renderer/src/api/mod.rs`
- `src/renderer/src/lib.rs`
- `src/renderer/src/api/renderer.rs`
- `src/runtime.rs`
- `src/launch.rs` only if a narrowly scoped event recording option is needed and fully tested
- Tests in the touched crates
- Phase evidence notes under this sprint directory if needed

## Forbidden Scope

- Do not rewrite input dispatch, action mapping, or priority/consumption rules.
- Do not dispatch app callbacks while a mutable renderer/asset manager borrow is active.
- Do not rewrite render hooks or low-level Vulkan modules.
- Do not deeply implement physics/audio/scripting emission.
- Do not touch unrelated `.idea/engine.iml` or `.reasonix/`.

## Supporting Docs To Read

- Phase 01 implementation and tests.
- `src/input/AGENTS.md`
- `src/renderer/AGENTS.md`
- `src/runtime.rs`
- `docs/api/06-input-polling-and-listeners.md`
- `docs/api/11-runtime-project-launcher.md`

## Implementation Steps

1. Add `engine_events` dependency to `renderer` and root `engine` as needed.
2. Reexport selected event types from `renderer::api` and `renderer`.
3. Add renderer facade bus access or event recorder hooks that do not expose raw internals.
4. Implement input/action bridge from post-dispatch `InputSnapshot`:
   - emit pressed/released from transient fields;
   - emit changed/axis for value changes when practical;
   - do not consume input or install new input layers unless clearly required.
5. Emit renderer/runtime asset/package events at known safe boundaries such as package manifest load success/failure and scene load success/failure.
6. Emit root runtime lifecycle events for app starting, project loaded, packages loaded, scene loading/loaded, frame boundaries where practical, shutdown requested/completed where control flow permits.
7. Add tests:
   - root runtime helper lifecycle ordering without Vulkan where possible;
   - input bridge emits after dispatch and preserves input tests;
   - facade reexports compile through renderer tests.
8. Run validation commands and prepare phase report details.

## Senior Guidance

- Assign event ownership per boundary. Runtime owns project/scene launcher lifecycle. Renderer owns renderer facade/input/asset observations.
- Prefer collecting events then dispatching after short mutable borrows end.
- Windowed shutdown completion may be hard to observe after `EventLoop::run`; emit `ShutdownRequested` and document any limitation.
- If CLI event recording is added, keep it simple and backwards compatible.

## Acceptance Criteria

- Apps can obtain/subscribe to events through supported renderer/root runtime surfaces.
- Root runtime lifecycle ordering is tested without Vulkan where possible.
- Existing input tests pass unchanged or with justified additive tests.
- Renderer examples still compile.
- No contradictory project/scene/package events are emitted from multiple owners.

## Negative Checks

- `InputSystem::dispatch_frame()` behavior remains intact.
- No subscriber callback receives `&mut Renderer`.
- No Vulkan module imports `engine_events` unless there is a tightly justified facade-only need, which should normally not happen.
- No event claims real physics/audio/scripting emission.

## Validation Commands

```bash
cargo test -p input
cargo test -p renderer
cargo test -p engine
cargo check -p renderer --examples
```

## Evidence Expectations

- Validator report path: `.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-05-event-system-lifecycle/validation/phase-02-validation-report.md`
- Summarize exact event order tested and any limitations.

## Stop Conditions

- Stop if borrow/lifetime pressure requires broad renderer redesign.
- Stop if input bridge cannot be implemented post-dispatch.
- Stop if root runtime event recording requires CLI/API churn beyond the sprint objective.

## Do Not Close Unless

- Phase 01 remains green.
- Renderer/root runtime tests validate lifecycle and input bridge behavior.
- Renderer examples compile.
- Event ownership boundaries are documented for Phase 03 docs.
