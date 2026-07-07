# Phase 03 Validation Report: App-Owned Input Dispatch And Renderer Routing

Date: 2026-07-07
Validator: Codex validation agent
Original Result: FAIL
Latest Revalidation Result: PASS

## Findings

1. Superseded by revalidation: `DeviceEvent::MouseWheel::PixelDelta` was previously dropped by both the new app-owned route and the legacy renderer-owned wrapper. Revalidation on 2026-07-07 confirms this specific issue is closed: `src/input.rs:133-138`, `src/renderer/src/api/renderer.rs:377-388`, and `src/renderer/src/api/renderer.rs:485-491` now handle all `DeviceEvent::MouseWheel { delta }` variants, including `PixelDelta`.
2. Superseded by final revalidation: `cargo test -p engine --quiet` previously failed because `EventEnvelope` was imported twice in `src/runtime.rs:5-7`. Final revalidation confirms the duplicate import is gone and `cargo test -p engine` passes.
3. No remaining Phase 03 blockers found in the current workspace.

## Original Criteria Results (Superseded)

| Criterion | Result | Evidence |
| --- | --- | --- |
| `InputActionEventEmitter` has a per-emitter observed-value map and emits from caller-provided `InputSnapshot` into caller-owned `EventBus` with frame index. | PASS | `src/input.rs:25-43` owns `observed_action_values`; `src/input.rs:46-91` emits snapshot-derived events into the supplied `EventBus`; `src/input.rs:102-107` attaches `FrameId(frame_index)`. |
| Tests cover press once, release once, changed value once, same-frame press/release behavior, transients exactly one dispatch, and no duplicate emission across frames. | PASS | `tests/input_action_events.rs:50-183` covers the action-emission cases; repeated `emit_from_snapshot` after another `dispatch_frame()` asserts no duplicate emission. |
| `Renderer::route_platform_input` separates ImGui/debug/capture/cursor/window filtering side effects from queueing. | PASS | `src/renderer/src/api/renderer.rs:342-469` handles ImGui forwarding, F1/F2/F12, cursor focus, UI capture, and window filtering, returning `RendererInputRouting`; queueing is in `queue_renderer_owned_input` at `src/renderer/src/api/renderer.rs:471-513`. |
| `Renderer::update_input` preserves legacy behavior by composing routing plus renderer-owned queueing. | FAIL | Composition exists at `src/renderer/src/api/renderer.rs:324-331`, but legacy wheel behavior is not fully preserved for `DeviceEvent::MouseWheel::PixelDelta` because queueing only accepts line-delta device wheels. |
| Root `queue_routed_input_event` queues only uncaptured events into caller-owned `InputSystem` and preserves MouseMotion, wheel, modifiers, cursor focus, keyboard/mouse events. | FAIL | It respects `routing.queue_input` and covers MouseMotion plus window keyboard/mouse/modifiers/cursor/wheel at `src/input.rs:121-155`, but it drops `DeviceEvent::MouseWheel::PixelDelta` at `src/input.rs:133-142`. |
| No app-owned input dispatch happens inside renderer. No new path double-dispatches both app and renderer input. | PASS | Renderer dispatch remains in legacy `prepare_frame`/`prepare_frame_headless` at `src/renderer/src/api/renderer.rs:1030` and `src/renderer/src/api/renderer.rs:1074`; root app-owned helper has no renderer call path that invokes `dispatch_frame`. Grep found no active path combining `queue_routed_input_event` with `renderer.update_input`. |
| Dogfood active path and event lifecycle ownership were not migrated in this phase. | PASS | Dogfood still calls `renderer.update_input` at `apps/dungeon_dogfood/src/main.rs:309` and still uses renderer-owned event bus in `apps/dungeon_dogfood/src/audio_bridge.rs:75`; no app-owned dogfood input/event lifecycle migration was observed. |
| Renderer does not depend on root `engine`. | PASS | `cargo tree -p renderer | rg '(^engine v| engine v|\bengine v0\.1\.0)'` produced no matches; `rg -n 'engine\s*=|package = "engine"|\bengine::' src/renderer src/renderer/Cargo.toml` produced no matches. |
| Docs/spec/changelog updates are accurate enough and not overclaiming dogfood/lifecycle migration. | PASS with residual note | Specs/changelog distinguish app-owned input as intended truth and dogfood/lifecycle as deferred; see `.internal-dev/specifications/api.md:20`, `.internal-dev/specifications/architecture.md:17`, `.internal-dev/specifications/decisions.md:16`, and `.internal-dev/changelogs/2026-07-07-engine-runtime-abstractions-phase-03-app-owned-input.md:29-33`. The docs inherit the wheel-preservation implementation gap but do not overclaim dogfood/lifecycle migration. |

## Original Commands And Checks Run (Superseded)

- `cargo fmt --check` -> pass.
- `cargo check -p input` -> pass.
- `cargo test -p input` -> pass, 10 tests.
- `cargo test -p engine` -> pass, 22 lib tests, 0 bin tests, 2 facade tests, 6 input-action integration tests.
- `cargo check -p renderer` -> pass with existing renderer dead-code warnings.
- `cargo test -p renderer` -> pass, 167 unit tests, 21 integration tests, 5 ignored doctests, with existing renderer dead-code warnings.
- `cargo check` -> pass with existing renderer dead-code warnings.
- `cargo tree -p renderer | rg '(^engine v| engine v|\bengine v0\.1\.0)'` -> no matches.
- `rg -n 'engine\s*=|package = "engine"|\bengine::' src/renderer src/renderer/Cargo.toml` -> no matches.
- `rg -n "dispatch_frame\(|emit_input_action|InputActionEventEmitter|update_input\(|DeviceEvent::MouseMotion|DeviceEvent::MouseWheel|MouseScrollDelta::PixelDelta|queue_routed_input_event" src apps tests` -> inspected dispatch/routing surface and identified the device pixel-wheel gap.
- Read required governance and phase artifacts: `AGENTS.md`, `.internal-dev/AGENTS.md`, `.internal-dev/specifications/AGENTS.md`, `src/input/AGENTS.md`, `src/renderer/AGENTS.md`, phase directive, target design, senior guidance, and worker report.
- Listed `.internal-dev/knowledge/` filenames and read the relevant renderer camera behavior knowledge note.

## Original Remediation Handoff (Superseded)

Recommended repair worker model: gpt-5.3 high reasoning.

Scope: Fix Phase 03 wheel preservation without broad redesign.

Required changes:
- Update renderer routing and both queue helpers so `DeviceEvent::MouseWheel` preserves both `LineDelta` and `PixelDelta`, matching the conversion behavior already used by `InputSystem::queue_winit_window_event`.
- Add focused tests for root `queue_routed_input_event` and renderer-owned helper behavior or a nearby pure helper so both device wheel delta variants are covered.
- Re-run `cargo fmt --check`, `cargo test -p engine`, `cargo test -p renderer`, and the dispatch/routing grep.

Do not migrate dogfood or event lifecycle ownership as part of this repair.

## Residual Risk

No browser or visual proof is required for this non-visual input-routing phase. Runtime smoke was not run because the assigned acceptance checks are compile/unit/code-inspection focused and the main thread already supplied broader local pass evidence.

## Intermediate Revalidation Update 2026-07-07 (Superseded)

### Revalidation Findings

1. Original wheel-routing failure is closed. The root app-owned helper now queues both `LineDelta` and `PixelDelta` device-wheel events through `scroll_delta_to_lines` at `src/input.rs:133-167`, and the focused pixel-delta regression test at `tests/input_action_events.rs:233-250` passes.
2. Renderer-owned compatibility now routes and queues both `LineDelta` and `PixelDelta` device-wheel events: `route_platform_input` matches all `DeviceEvent::MouseWheel` variants at `src/renderer/src/api/renderer.rs:377-388`, and `queue_renderer_owned_input` queues them through the same conversion at `src/renderer/src/api/renderer.rs:485-491`.
3. Full root crate test validation is blocked by a separate current-workspace compile defect: `src/runtime.rs:5-7` imports `EventEnvelope` twice for test builds, causing `cargo test -p engine --quiet` to fail with `E0252`.

### Revalidation Criteria Results

| Criterion | Result | Evidence |
| --- | --- | --- |
| Original `DeviceEvent::MouseWheel::PixelDelta` failure is closed. | PASS | `src/input.rs:133-167`, `src/renderer/src/api/renderer.rs:377-388`, `src/renderer/src/api/renderer.rs:485-491`; `cargo test -p engine --test input_action_events --quiet` passes 7 tests. |
| App-owned input queueing parity for MouseMotion, wheel, modifiers, cursor focus, keyboard/mouse events. | PASS | `src/input.rs:125-153` queues MouseMotion, all DeviceEvent mouse-wheel variants, and supported WindowEvent input/focus/modifier branches only when `routing.queue_input` is true. |
| Renderer-owned `Renderer::update_input` compatibility. | PASS | `src/renderer/src/api/renderer.rs:324-331` composes `route_platform_input` and `queue_renderer_owned_input`; `cargo check -p renderer --quiet` and `cargo test -p renderer --quiet` pass. |
| Same-frame press/release preservation and no duplicate action emission across frames. | PASS | `tests/input_action_events.rs:51-183`; `cargo test -p engine --test input_action_events --quiet` passes 7 tests. |
| Raw primitive availability through facade/prelude remains intact. | PASS | `tests/facade_imports.rs` still compiles under the targeted input-action test build; full `cargo test -p engine` is blocked before completion by the unrelated duplicate import. |
| No renderer dependency on root `engine` facade. | PASS | `cargo tree -p renderer | rg '(^engine v| engine v|\bengine v0\.1\.0)'` produced no matches; `rg -n 'engine\s*=|package = "engine"|\bengine::' src/renderer src/renderer/Cargo.toml` produced no matches. |
| Full Phase 03 validation suite passes. | FAIL | `cargo test -p engine --quiet` fails with duplicate `EventEnvelope` import in `src/runtime.rs:5-7`. |

### Revalidation Commands

- `cargo fmt --check` -> pass.
- `cargo test -p engine --test input_action_events --quiet` -> pass, 7 tests; existing renderer dead-code warnings remain.
- `cargo check -p renderer --quiet` -> pass with existing renderer dead-code warnings.
- `cargo test -p renderer --quiet` -> pass, 167 unit tests, 21 integration tests, 5 ignored doctests; existing renderer dead-code warnings remain.
- `cargo tree -p renderer | rg '(^engine v| engine v|\bengine v0\.1\.0)'` -> no matches.
- `rg -n 'engine\s*=|package = "engine"|\bengine::' src/renderer src/renderer/Cargo.toml` -> no matches.
- `rg -n "dispatch_frame\(|emit_input_action|InputActionEventEmitter|update_input\(|route_platform_input|queue_routed_input_event|DeviceEvent::MouseMotion|DeviceEvent::MouseWheel|MouseScrollDelta::PixelDelta" src apps tests` -> inspected.
- `git diff --check` -> pass.
- `cargo check -p input` -> pass.
- `cargo test -p input` -> pass, 10 tests.
- `cargo check -p engine --quiet` -> pass with existing renderer dead-code warnings.
- `cargo test -p engine --quiet` -> fail with `E0252` duplicate `EventEnvelope` import in `src/runtime.rs:5-7`.

### Required Remediation (Completed)

Repair the duplicate `EventEnvelope` import in `src/runtime.rs` without changing Phase 03 input-routing behavior, then rerun:

- `cargo fmt --check`
- `cargo test -p engine --test input_action_events --quiet`
- `cargo test -p engine --quiet`
- `cargo check -p renderer --quiet`
- `cargo test -p renderer --quiet`
- renderer dependency tree/grep checks

## Final Revalidation Update 2026-07-07

### Final Result

PASS. No Phase 03 blocker remains.

### Final Findings

No active findings.

Previously failed issues are closed:

- `DeviceEvent::MouseWheel::PixelDelta` is preserved in root app-owned input routing and renderer-owned compatibility routing.
- The duplicate test-only `EventEnvelope` import in `src/runtime.rs` is fixed; `src/runtime.rs:5-7` now imports `EventEnvelope` only under `#[cfg(test)]` and no longer duplicates it in the general `engine_events` import.

### Final Criteria Results

| Criterion | Result | Evidence |
| --- | --- | --- |
| `InputActionEventEmitter` has a per-emitter observed-value map and emits from caller-provided `InputSnapshot` into caller-owned `EventBus` with frame index. | PASS | `src/input.rs:25-43`, `src/input.rs:46-91`, and `src/input.rs:102-107`. |
| Tests cover press once, release once, changed value once, same-frame press/release behavior, transients exactly one dispatch, and no duplicate emission across frames. | PASS | `tests/input_action_events.rs:51-183`; `cargo test -p engine` passes 7 input-action tests. |
| Root `queue_routed_input_event` queues only uncaptured events into caller-owned `InputSystem` and preserves MouseMotion, wheel, modifiers, cursor focus, keyboard/mouse events. | PASS | `src/input.rs:116-168`; wheel coverage includes `MouseScrollDelta::PixelDelta` at `src/input.rs:166`; regression test at `tests/input_action_events.rs:233-250`. |
| `Renderer::route_platform_input` separates ImGui/debug/capture/cursor/window filtering side effects from queueing. | PASS | `src/renderer/src/api/renderer.rs:342-388` routes side effects and queue decisions; renderer-owned queueing is separate at `src/renderer/src/api/renderer.rs:468-507`. |
| `Renderer::update_input` preserves legacy behavior by composing routing plus renderer-owned queueing. | PASS | `src/renderer/src/api/renderer.rs:324-331`; `cargo check -p renderer` and `cargo test -p renderer` pass. |
| No app-owned input dispatch happens inside renderer. No new path double-dispatches both app and renderer input. | PASS | Grep inspection found root app-owned queue helper usage only in root tests/facade exports; renderer dispatch remains in legacy renderer-owned frame prep paths. |
| Dogfood active path and event lifecycle ownership were not migrated in this phase. | PASS | Dogfood still uses legacy `renderer.update_input` at `apps/dungeon_dogfood/src/main.rs:309`; no app-owned dogfood lifecycle migration found in Phase 03 inspection. |
| Renderer does not depend on root `engine`. | PASS | `cargo tree -p renderer | rg '(^engine v| engine v|\bengine v0\.1\.0)'` produced no matches; `rg -n 'engine\s*=|package = "engine"|\bengine::' src/renderer src/renderer/Cargo.toml` produced no matches. |
| Full final validation command set passes. | PASS | Commands below. |

### Final Commands

- `cargo fmt --check` -> pass.
- `cargo test -p engine` -> pass: 22 lib tests, 0 bin tests, 2 facade tests, 7 input-action tests, 3 runtime event dispatcher tests, 0 doctests.
- `cargo check -p renderer` -> pass with existing renderer dead-code warnings.
- `cargo test -p renderer` -> pass: 167 lib tests, 21 integration tests, 5 ignored doctests, with existing renderer dead-code warnings.
- `cargo test -p engine_events` -> pass: 18 lib tests, 1 ignored doctest.
- `cargo tree -p renderer | rg '(^engine v| engine v|\bengine v0\.1\.0)'` -> no matches.
- `rg -n 'engine\s*=|package = "engine"|\bengine::' src/renderer src/renderer/Cargo.toml` -> no matches.
- `rg -n "dispatch_frame\(|emit_input_action|InputActionEventEmitter|update_input\(|route_platform_input|queue_routed_input_event|DeviceEvent::MouseMotion|DeviceEvent::MouseWheel|MouseScrollDelta::PixelDelta" src apps tests` -> inspected; no Phase 03 blocker found.
- `git diff --check` -> pass.

### Browser/Visual Proof

Not applicable. Phase 03 is non-visual input/event routing work; no renderer visual behavior changed that requires headless capture validation.
