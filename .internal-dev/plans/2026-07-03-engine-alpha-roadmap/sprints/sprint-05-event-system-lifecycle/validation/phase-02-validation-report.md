# Phase 02 Validation Report: Renderer And Root Runtime Integration

Date: 2026-07-03

Branch: `sprint/alpha-05-event-system-lifecycle`

Status: validator passed after remediation, commit/push/report pending

## Scope Validated

- Added `engine_events` dependencies to root `engine` and `renderer`.
- Reexported event contracts through `renderer::api` and crate-level `renderer` facade.
- Added renderer-owned event bus access through `Renderer::events()`, `Renderer::events_mut()`, and `Renderer::set_event_recorder(...)`.
- Added renderer input/action bridge after `InputSystem::dispatch_frame()` using `InputSnapshot::action_values()`.
- Added renderer frame lifecycle events at pre-update and post-update boundaries.
- Added root runtime event recorder for app/project/package/scene/headless-shutdown lifecycle evidence.
- Remediated initial validator findings by emitting `ShutdownRequested` for window close/Escape and `PackageFailed` before package manifest load errors return.
- Added unit tests for input snapshot observation, renderer input bridge press/release order, root runtime lifecycle ordering, windowed shutdown-requested emission, and package failure emission without Vulkan.

## Files Created Or Changed

| File | Change |
|---|---|
| `Cargo.toml` | Added root `engine_events` dependency. |
| `Cargo.lock` | Updated dependency graph for root/renderer event dependency. |
| `src/input/src/lib.rs` | Added `InputSnapshot::action_values()` and observer test. |
| `src/renderer/Cargo.toml` | Added `engine_events` dependency. |
| `src/renderer/src/api/mod.rs` | Reexported event facade types. |
| `src/renderer/src/lib.rs` | Reexported event facade types from crate root. |
| `src/renderer/src/api/renderer.rs` | Added renderer event bus, frame lifecycle events, post-dispatch action bridge, and tests. |
| `src/runtime.rs` | Added root runtime lifecycle/package/scene event recording, package failure emission, shutdown-requested emission, and tests. |

## Commands

| Command | Result | Notes |
|---|---:|---|
| `cargo fmt --check` | Passed | Clean after formatting. |
| `cargo test -p input` | Passed | 10 tests passed. |
| `cargo test -p engine` | Passed | 20 tests passed after remediation. |
| `cargo test -p renderer` | Passed | 152 lib tests, 17 integration tests, 5 ignored doctests. Existing renderer warnings remain. |
| `cargo check -p renderer --examples` | Passed | Existing renderer warnings remain. |
| `cargo test -p engine_events` | Passed | 7 tests passed; Phase 01 remains green. |
| `rg -n "engine_events" src/renderer/src/vulkan src/renderer/src/data src/renderer/src/scene src/renderer/src/api src/runtime.rs src/input/src/lib.rs Cargo.toml src/renderer/Cargo.toml` | Passed | Hits are root/runtime/facade/API dependency and reexport points only; no Vulkan/internal import. |
| `git diff --check` | Passed | No whitespace errors. |
| Phase 02 validator rerun | Passed | No blocking findings after remediation. |

## Event Order Tested

- Runtime helper test records:
  - `AppStarting`
  - `ProjectLoading`
  - `ProjectLoaded`
  - `SceneLoading`
  - `SceneLoaded`
- Runtime remediation tests record:
  - `ShutdownRequested`
  - `PackageFailed`
- Renderer bridge test records input action phases:
  - `Pressed`
  - `Released`

## Validator Remediation

The first Phase 02 validator review failed on two real runtime contract gaps:

- Window close/Escape exited the windowed event loop without emitting `LifecycleEvent::ShutdownRequested`.
- Package manifest load failures returned an error without emitting `AssetEvent::PackageFailed`.

Both gaps are now remediated in `src/runtime.rs`. Direct no-Vulkan tests cover the helper events, the full requested validation set was rerun after the fixes, and the validator rerun passed with no blocking findings.

## Limitations

- Windowed `ShutdownCompleted` remains limited by `winit::EventLoop::run`; this phase emits `ShutdownRequested` for observable close/Escape intent and deterministic shutdown completion for the headless path.
- Physics, audio, and scripting event families remain typed contracts only; real emission belongs to later sprints.
- Existing renderer dead-code warnings remain outside this phase.
- `.idea/engine.iml` and `.reasonix/` remain unrelated local state and must stay out of Phase 02 commits.

## Validator Handoff

Validate that event access does not expose `&mut Renderer` to subscribers, the input bridge observes post-dispatch snapshots without changing dispatch semantics, root runtime ordering/failure/shutdown events are tested without Vulkan, and `engine_events` is not imported by low-level Vulkan modules.
