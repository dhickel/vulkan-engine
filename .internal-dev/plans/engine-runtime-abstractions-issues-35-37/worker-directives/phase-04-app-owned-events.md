# Phase 04 Worker Directive: App-Owned Event Bus And Lifecycle Stages

Status: ready after Phase 03 validation
Validation report: `.internal-dev/plans/engine-runtime-abstractions-issues-35-37/validation/phase-04-validation-report.md`

## Objective

Move the new app/facade lifecycle and staged event dispatch path to caller-owned `EventBus` while leaving renderer-owned event bus as legacy compatibility state.

## User-Visible Outcome

Apps can run startup/input/update/render lifecycle events through one app-owned `EventBus` with monotonic ordering and no renderer-owned app event dependency.

## Direct Editable Targets

- root runtime/helper modules
- `src/runtime.rs`
- `src/events.rs`
- `src/renderer/src/api/renderer.rs` only for no-dispatch path/lifecycle separation adjustments
- `src/events/src/lib.rs` tests only if needed
- root crate tests

## Forbidden Scope

- Do not migrate dogfood yet.
- Do not remove `Renderer::events()` or `events_mut()`.
- Do not add renderer/windowing dependencies to `engine_events`.
- Do not duplicate lifecycle events between root runtime and renderer new path.

## Supporting Docs To Read

- `src/events/AGENTS.md`
- `.internal-dev/specifications/services.md`
- `.internal-dev/specifications/service-graph.md`
- `02-target-design.md`

## Ordered Steps

1. Add root runtime event helper for emitting/draining lifecycle stages on a caller-owned `EventBus`.
2. Define new path lifecycle ownership:
   - root/app emits `FrameStarted` and `FrameEnded`;
   - input stage drained after app-owned input dispatch/action emission;
   - renderer no-dispatch path does not emit/drain app lifecycle events.
3. Add tests for:
   - monotonic sequence in one bus across startup/input/pre/post/render events;
   - no duplicate frame lifecycle events from one new frame helper;
   - listener failure behavior remains collected and non-aborting.
4. Adjust root runtime launcher only if it can use the new helper without behavior drift; otherwise leave launcher migration to later and document residual.
5. Keep renderer legacy bus and event methods functioning.

## Senior-Engineer Guidance

- One bus per app runtime path is the goal. Avoid helper APIs that make users create a second bus for audio/input/lifecycle.
- `engine_events` is already independent; do not change that to make root helpers easier.
- Renderer render status can be returned by functions; do not force app event ordering through renderer-owned bus.

## Acceptance Criteria

- New runtime helper emits/drains lifecycle/input stages on caller-owned `EventBus`.
- Event sequence ordering is monotonic in one bus.
- New renderer no-dispatch path does not produce duplicate app lifecycle events.
- Renderer `events()` / `events_mut()` remain for legacy compatibility.

## Negative Checks

- No `engine_events` dependency on renderer, winit, or root `engine`.
- No dogfood active-path migration yet.
- No two-bus requirement for one app frame.
- No duplicate `FrameStarted`/`FrameEnded` on new path.

## Validation Commands

```sh
cargo check -p engine_events
cargo test -p engine_events
cargo test -p engine
cargo check -p renderer
cargo test -p renderer
rg -n "EventBus|FrameStarted|FrameEnded|events_mut\\(|drain_stage|dispatch_pending" src apps
```

## Evidence Expectations

- Worker notes identify the new lifecycle producer for the new path.
- Validator confirms renderer legacy bus is compatibility-only for new path.
- Validator records event ordering test coverage.

## Stop Conditions

- Stop if lifecycle ownership conflicts with Phase 02 renderer API shape.
- Stop if helper design requires support crates to depend on root `engine`.
- Stop if event ordering criteria are ambiguous and need planning revision.

## Do Not Close Unless

- Event bus ownership is separate from input migration and ready for dogfood.
- Phase 04 validation report is written.
