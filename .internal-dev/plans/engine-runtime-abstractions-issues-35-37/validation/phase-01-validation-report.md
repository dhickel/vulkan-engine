# Phase 01 Validation Report

Status: passed
Date: 2026-07-07
Phase directive: `.internal-dev/plans/engine-runtime-abstractions-issues-35-37/worker-directives/phase-01-root-facade.md`
Worker report: `.internal-dev/plans/engine-runtime-abstractions-issues-35-37/work-units/phase-01-root-facade-worker-report.md`
Validator route: current validation agent; model identity is not exposed in-session. No fallback validation tooling was used.

## Scope

Validated Phase 01 root `engine` facade only. No implementation repairs were made. The existing modified `src/renderer/src/api/renderer.rs` was inspected as out-of-scope for this phase except for its effect on compile and dependency checks.

## Findings

- None blocking.
- Evidence hygiene: the user-provided read list names `.internal-dev/plans/engine-runtime-abstractions-issues-35-37/work-units/phase-01-root-facade.md`, but that file does not exist. The executable directive exists at `.internal-dev/plans/engine-runtime-abstractions-issues-35-37/worker-directives/phase-01-root-facade.md`, which is also the path referenced by the orchestration plan and worker report.

## Criteria

| Criterion | Result | Evidence |
| --- | --- | --- |
| Root `engine` compiles as library and binary package | Pass | `src/lib.rs` exists and exposes root modules; `src/main.rs` imports `engine::launch` and `engine::runtime`; `cargo check -p engine` passed. |
| Simple facade/prelude around renderer/input/events/camera/frame/runtime/launch | Pass | `src/lib.rs` exposes `camera`, `events`, `frame`, `input`, `launch`, `render`, `runtime`, and a narrow `prelude`; module files use explicit `pub use` re-exports. |
| Raw primitives directly importable from original crates | Pass | `tests/facade_imports.rs` imports `input as raw_input`, `engine_events as raw_events`, and `renderer as raw_renderer`; `cargo test -p engine` passed. |
| Raw primitives accessible through root facade modules | Pass | `tests/facade_imports.rs` imports and uses `engine::{camera, events, input, render}` and `engine::prelude::*`; `cargo test -p engine` passed. |
| Renderer does not depend on root `engine` | Pass | `cargo tree -p renderer > /tmp/engine-phase01-renderer-tree.txt` passed; `rg '(^engine v| engine v|\bengine v0\.1\.0)' /tmp/engine-phase01-renderer-tree.txt` returned no matches. |
| No support crate depends on root `engine` | Pass | Source dependency scan `rg -n '^engine\s*=|package\s*=\s*"engine"|path\s*=\s*"\.\./\.\."|path\s*=\s*"\.\."' src apps tools Cargo.toml` returned no matches. |
| No over-abstracted app framework or hidden ownership inversion | Pass | Root additions are re-export modules plus `FrameClock`/`FrameInfo`; scan for `Engine`, `RuntimeParts`, `App`, `World`, plugin registry, scheduler, resource world, `engine_core`, and `engine_runtime` in Phase 01 root sources returned no matches. |
| Existing launcher behavior remains intact | Pass | `src/main.rs` delegates to the same `launch` parser and `runtime::run` through the library path; `cargo test -p engine launch::tests -- --nocapture` passed 10 launch tests. |
| Frame clock helper has focused tests | Pass | `src/frame.rs` tests cover advancing indices/deltas and backwards tick clamping; included in `cargo test -p engine`. |
| Negative scope: no new `engine_core` crate | Pass | Workspace `Cargo.toml` only adds direct root `input` dependency; no new workspace crate added. |
| Negative scope: no dogfood migration | Pass | Phase 01 diff did not touch `apps/dungeon_dogfood`. |
| Negative scope: no renderer frame/render behavior change in Phase 01 facade work | Pass with residual | Root Phase 01 files do not change renderer behavior. `src/renderer/src/api/renderer.rs` is modified in the worktree, but it is out of the Phase 01 root facade scope and was not attributed to this phase. Compile gates still passed. |
| Validation matrix Phase 01 commands | Pass | `cargo fmt --check`, `cargo check -p engine`, `cargo test -p engine`, `cargo check`, `cargo tree -p renderer`, and `cargo tree -p engine` ran successfully; renderer reverse-edge grep returned no matches. |

## Commands Run

```sh
cargo fmt --check
cargo check -p engine
cargo test -p engine
cargo check
cargo tree -p engine
cargo tree -p renderer > /tmp/engine-phase01-renderer-tree.txt
rg '(^engine v| engine v|\bengine v0\.1\.0)' /tmp/engine-phase01-renderer-tree.txt
cargo test -p engine launch::tests -- --nocapture
rg -n '^engine\s*=|package\s*=\s*"engine"|path\s*=\s*"\.\./\.\."|path\s*=\s*"\.\."' src apps tools Cargo.toml
rg -n 'struct\s+Engine\b|pub\s+struct\s+RuntimeParts\b|pub\s+struct\s+App|pub\s+struct\s+World|plugin registry|scheduler|resource world|engine_core|engine_runtime' src/lib.rs src/*.rs tests
```

Results:

- `cargo fmt --check`: pass.
- `cargo check -p engine`: pass with existing renderer dead-code warnings.
- `cargo test -p engine`: pass, 22 lib tests, 2 integration import tests, and doc tests passed.
- `cargo check`: pass with existing renderer dead-code warnings.
- `cargo tree -p engine`: pass; root `engine` depends downward on `engine_events`, `input`, `launch_shared`, `renderer`, and external runtime/logging crates.
- Renderer reverse-edge grep: no matches, expected exit code 1.
- Launch parser focused tests: pass, 10 tests.
- Dependency declaration scan: no matches.
- Over-abstraction scan: no matches.

## Evidence Inspected

- Required repo/workflow guides: `AGENTS.md`, `.internal-dev/AGENTS.md`, `.internal-dev/specifications/AGENTS.md`.
- Relevant specs: `.internal-dev/specifications/architecture.md`, `.internal-dev/specifications/service-graph.md`, `.internal-dev/specifications/api.md`, `.internal-dev/specifications/decisions.md`.
- Plan/support docs: `final-orchestration-plan.md`, `00-specification-lock.md`, `02-target-design.md`, `shared/senior-engineer-guidance.md`, `shared/validation-matrix.md`, `shared/implementation-notes.md`, validation `README.md`.
- Worker report: `work-units/phase-01-root-facade-worker-report.md`.
- Implementation files: `Cargo.toml`, `src/lib.rs`, `src/input.rs`, `src/events.rs`, `src/render.rs`, `src/camera.rs`, `src/frame.rs`, `src/main.rs`, `src/runtime.rs`, `src/launch.rs`, `tests/facade_imports.rs`.

## Residual Risks

- The worktree contains an out-of-scope modified renderer file, `src/renderer/src/api/renderer.rs`. This validation did not certify that renderer change beyond confirming the Phase 01 command gates still pass.
- Runtime smoke and headless capture were not required for Phase 01 because this phase did not claim visual/camera output proof or materially change renderer/camera behavior.
- Phase 06 remains responsible for broad specs/docs/changelog/evidence-index closeout.

## Remediation Routing

No code remediation required. The stale user-listed directive path is a non-blocking docs/evidence hygiene issue for the orchestrator or plan owner to clean up if desired.

## Conclusion

Phase 01 may proceed to the next phase.
