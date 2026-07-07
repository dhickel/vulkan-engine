# Phase 02 Validator Review: Renderer And Root Runtime Integration

Date: 2026-07-03

Validator: Codex validation agent

Status: FAIL

## Findings

1. code_defect: Windowed shutdown-requested events are not emitted where the directive required them. `run_windowed` exits directly on close/Escape at `src/runtime.rs:369` and `src/runtime.rs:374`, but the phase directive says to emit `ShutdownRequested` where `EventLoop::run` prevents reliable `ShutdownCompleted` observation. The implementation summary and validation summary only call out windowed `ShutdownCompleted` as limited, which leaves the missing `ShutdownRequested` event hidden.

2. code_defect: Package load failures are not emitted as asset events. `load_enabled_project_packages` emits `PackageLoading` at `src/runtime.rs:464` and `PackageLoaded` at `src/runtime.rs:480`, but the error branch at `src/runtime.rs:471` returns a formatted error without emitting `AssetEvent::PackageFailed`. The worker directive explicitly called for package manifest load success/failure events at safe boundaries.

3. docs_or_evidence_defect: Phase evidence is too optimistic for the runtime residuals above. `phase-02-validation-report.md` reports root runtime package/scene/headless-shutdown event recording and lists only windowed `ShutdownCompleted` as a limitation. `validation-summary.json` similarly says windowed completion must be documented "if only shutdown-requested is observable", but shutdown-requested is not currently observable from the root runtime.

4. plan_defect: Forbidden local paths remain dirty in the phase worktree. `.idea/engine.iml` is modified and `.reasonix/` exists untracked, while the user and directive both said not to touch them. I did not edit either path during validation. They must be excluded from phase closeout unless the main thread confirms they are unrelated pre-existing local state.

## Criterion Results

| Criterion | Result | Evidence |
|---|---:|---|
| Renderer and root engine depend on `engine_events` only at facade/runtime boundaries. | PASS | `rg engine_events` found root manifest/runtime and renderer facade/API hits only; no low-level renderer internals. |
| Event types are reexported through renderer facade and crate root. | PASS | `src/renderer/src/api/mod.rs` and `src/renderer/src/lib.rs` reexport event types. |
| Renderer exposes event bus access without subscriber callbacks receiving `&mut Renderer`. | PASS | `Renderer::events`, `events_mut`, and `set_event_recorder` expose `EventBus`; `EventBus::subscribe` callbacks receive `&EventEnvelope`. |
| Input/action bridge observes `InputSnapshot` after `InputSystem::dispatch_frame` and does not install layers or change input consumption semantics. | PASS | Renderer bridge runs after `dispatch_frame`; input changes only add `InputSnapshot::action_values()` and observer tests. Existing input tests pass. |
| Root runtime emits/records app/project/package/scene/headless-shutdown lifecycle/asset events at safe boundaries. | FAIL | Success-path lifecycle/package/scene/headless shutdown events exist, but package failure events and windowed shutdown-requested events are missing. |
| Ordering is tested without Vulkan. | PASS | `runtime_lifecycle_helpers_record_project_and_scene_order_without_vulkan` passed. |
| Phase 01 remains green. | PASS | `cargo test -p engine_events` passed 7 tests. |
| Renderer examples compile. | PASS | `cargo check -p renderer --examples` passed with existing warnings. |
| No low-level Vulkan modules import `engine_events`. | PASS | Ownership scan had no `engine_events` hits under `src/renderer/src/vulkan`. |
| Validation report and validation-summary JSON are conservative and internally consistent. | FAIL | Evidence omits the missing package-failure and shutdown-requested behavior while claiming the phase event coverage is complete enough for validator handoff. |
| Do not touch `.idea/engine.iml` or `.reasonix`. | FAIL / PRESERVE | Worktree currently has modified `.idea/engine.iml` and untracked `.reasonix/`; validator made no changes there. |

## Commands Run

| Command | Result |
|---|---:|
| `cargo fmt --check` | PASS |
| `python3 -m json.tool .internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-05-event-system-lifecycle/artifacts/validation-summary.json` | PASS |
| `git diff --check` | PASS |
| `rg -n "engine_events" src/renderer/src/vulkan src/renderer/src/data src/renderer/src/scene src/renderer/src/shaders src/input/src/lib.rs src/renderer/src/api src/runtime.rs Cargo.toml src/renderer/Cargo.toml` | PASS: facade/runtime hits only. |
| `cargo test -p input` | PASS: 10 tests passed. |
| `cargo test -p engine` | PASS: 18 tests passed with existing renderer warnings. |
| `cargo test -p renderer` | PASS: 152 lib tests, 17 integration tests, 5 ignored doctests; existing warnings. |
| `cargo check -p renderer --examples` | PASS with existing warnings. |
| `cargo test -p engine_events` | PASS: 7 tests passed. |
| `git status --short` | Reviewed: phase files dirty plus `.idea/engine.iml` modified and `.reasonix/` untracked. |

## Files Changed By Validator

- `.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-05-event-system-lifecycle/validation/phase-02-validator-review.md`

## Required Remediation

Use a scoped repair worker for `src/runtime.rs` and phase evidence only:

- Emit and test `LifecycleEvent::ShutdownRequested` on window close/Escape before `control_flow.exit()` without claiming windowed `ShutdownCompleted`.
- Emit and test `AssetEvent::PackageFailed` before returning package manifest load errors.
- Update `phase-02-validation-report.md` and `validation-summary.json` conservatively after the fixes and rerun the focused plus requested validation commands.
- Preserve `.idea/engine.iml` and `.reasonix/`; do not include them in the phase commit/closeout.

No browser or headless capture proof is required for this phase because the changes are event/runtime API behavior and compile/test validation, not visual renderer output.
