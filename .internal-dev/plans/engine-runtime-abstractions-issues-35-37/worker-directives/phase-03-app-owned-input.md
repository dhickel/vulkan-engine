# Phase 03 Worker Directive: App-Owned Input Dispatch And Action Events

Status: ready after Phase 02 validation
Validation report: `.internal-dev/plans/engine-runtime-abstractions-issues-35-37/validation/phase-03-validation-report.md`

## Objective

Move the new app/facade input path to app-owned `InputSystem` dispatch and app-owned input action event emission while preserving renderer UI/debug/capture platform handling.

## User-Visible Outcome

Apps can route winit input into their own `InputSystem`, dispatch once per frame, and emit action events once without relying on renderer-owned input.

## Direct Editable Targets

- `src/runtime.rs` or new root runtime helper modules
- `src/input.rs` / `src/events.rs` / `src/frame.rs` from root facade
- `src/renderer/src/api/renderer.rs`
- `src/renderer/src/api/mod.rs`
- `src/renderer/src/api/prelude.rs`
- `src/input/src/lib.rs` tests only if needed
- root crate tests
- renderer tests for split `update_input` compatibility

## Forbidden Scope

- Do not migrate dogfood event bus ownership yet.
- Do not remove renderer legacy `update_input`.
- Do not change input crate dispatch semantics.
- Do not merge event-bus lifecycle migration into this phase.

## Supporting Docs To Read

- `src/input/AGENTS.md`
- `src/renderer/AGENTS.md`
- `02-target-design.md`
- `shared/senior-engineer-guidance.md`

## Ordered Steps

1. Extract action-event emission into a reusable root/runtime helper such as `InputActionEventEmitter`.
2. Add tests for:
   - press emits once;
   - release emits once;
   - changed value emits once;
   - same-frame press/release behavior;
   - transients survive exactly one dispatch;
   - no duplicate emission across a frame.
3. Split renderer platform handling from renderer-owned input queueing:
   - new helper forwards ImGui events, toggles debug UI, handles manual capture/cursor side effects, and returns input routing/capture decision;
   - legacy `Renderer::update_input` wraps the helper and queues into renderer-owned `InputSystem`.
4. Add root/app helper to queue uncaptured winit/device events into app-owned `InputSystem`.
5. Define resize-skipped frame input policy:
   - preferred: input dispatch still occurs once at the app frame boundary even if render skips;
   - document/test policy if different.
6. Preserve `DeviceEvent::MouseMotion`, mouse wheel, keyboard repeat, modifiers, cursor focus, UI capture suppression, and window-id filtering.

## Senior-Engineer Guidance

- The action-event emitter needs its own observed-value map per app input stream.
- UI capture should suppress gameplay/FPS input queueing, not all renderer platform side effects.
- Keep legacy renderer input behavior intact by composing the new split helper.
- This phase proves input correctness before event lifecycle migration; keep event bus changes limited to emitting input events into a caller-supplied bus.

## Acceptance Criteria

- New app-owned input path dispatches exactly once per app frame.
- Action events are emitted once from app-owned snapshot into caller-owned `EventBus`.
- Renderer exposes split platform/input routing helper or equivalent.
- Legacy `Renderer::update_input` still compiles and preserves old examples.
- Tests cover transient and duplicate-emission risks.

## Negative Checks

- No dogfood active-path migration yet except optional compile adjustments.
- No event lifecycle ownership migration beyond input action emission helper.
- No second dispatch in renderer for new path.
- No dropped `DeviceEvent::MouseMotion`.

## Validation Commands

```sh
cargo check -p input
cargo test -p input
cargo test -p engine
cargo check -p renderer
cargo test -p renderer
rg -n "dispatch_frame\\(|emit_input_action|InputActionEventEmitter|update_input\\(|DeviceEvent::MouseMotion" src apps
```

## Evidence Expectations

- Worker notes show where exactly the app-owned dispatch happens.
- Validator confirms no new path calls both app dispatch and renderer legacy dispatch.
- Validator report includes grep/code-inspection findings for `dispatch_frame`.

## Stop Conditions

- Stop if split platform routing cannot preserve UI/debug/capture side effects without redesign.
- Stop if tests reveal input crate semantic ambiguity requiring planning revision.
- Stop if implementation requires renderer to depend on root `engine`.

## Do Not Close Unless

- Transient tests exist and pass.
- New input route is separate from event migration.
- Phase 03 validation report is written.
