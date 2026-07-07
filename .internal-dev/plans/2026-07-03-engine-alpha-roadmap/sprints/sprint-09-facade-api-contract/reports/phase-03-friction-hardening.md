# Phase 03 Friction Hardening Report

## Scope

Hardened beginner-facing friction docs around current error, input, camera, material, and capture behavior. The implementation stayed docs/test focused and did not change renderer runtime behavior.

## Friction Item Matrix

| Item | Status | Change | Evidence |
|---|---:|---|---|
| Error diagnostics for scene/package/asset failures | Fixed | Added docs that classify `RendererError::Scene`, `RendererError::Asset`, manifest parse errors, stale handles, missing durable asset IDs, and deferred ticket failures. Added a GPU-free integration test for beginner-readable error display. | `docs/api/03-scene-graph-and-fragment-workflows.md`, `docs/api/04-assets.md`, `docs/api/04-assets-sync-deferred-and-handles.md`, `src/renderer/tests/integration.rs` |
| Input-profile TOML expectations | Fixed | Documented the actual `ActionMap::from_toml_str`, `load_toml_file`, and `save_toml_file` app-owned setup flow. Clarified that `RendererConfig` does not auto-load profile files. | `docs/api/06-input.md`, `docs/api/06-input-polling-and-listeners.md`, `cargo test -p input` |
| Camera beginner path vs compatibility math helpers | Fixed | Clarified that `Renderer::install_default_fps_input()` is the beginner moving-camera path, while root-level camera/frustum/ray/orbit helpers are compatibility helpers outside `renderer::prelude`. | `docs/api/06-input-polling-and-listeners.md` |
| Material override limits | Fixed | Clarified that node material override entries are durable string metadata preserved by scene save/load and summaries, not live GPU material edits or first-class material authoring. | `docs/api/03-scene.md`, `docs/api/03-scene-graph-and-fragment-workflows.md`, `docs/api/04-assets-sync-deferred-and-handles.md` |
| Capture evidence path | Fixed | Repeated that visual validation must use root launcher `--headless --capture_target draw`; desktop screenshot evidence does not count. Added a GPU-free integration test for capture target/config parsing. | `docs/api/07-config.md`, `docs/api/07-engine-arguments.md`, `docs/api/08-debug.md`, `src/renderer/tests/integration.rs` |

## Changes

- Updated scene docs to explain material override metadata limits and scene/asset error routing.
- Updated asset docs to distinguish manifest/schema failures, loader failures, handle lifecycle failures, and deferred load failures.
- Updated input docs to document actual TOML profile loading and renderer-owned FPS camera behavior.
- Updated config/debug/argument docs to name headless draw capture as the validation path and reject desktop screenshots as proof.
- Added integration tests for current error display and capture target/config contracts.

## Validation

| Command | Result | Notes |
|---|---:|---|
| `cargo fmt --check` | Pass | Passed after running `cargo fmt` for the new integration test formatting. |
| `cargo check` | Pass | Existing renderer dead-code warnings only. |
| `cargo test -p renderer` | Pass | 160 unit tests, 20 integration tests, and 5 ignored doctests passed. |
| `cargo check -p renderer --examples` | Pass | Existing renderer dead-code warnings only. |
| `rg -n "TODO|pending|planned|not implemented|material override|input profile|capture_target|desktop screenshot|advanced-interop" docs/api src/renderer/src src/renderer/tests` | Pass for phase intent | Expected hits include new material/input/capture wording, existing internal `pending` names, existing `TODO` comments in renderer internals, and existing `advanced-interop` docs/gates. |
| `cargo test -p input` | Pass | Conditional input-profile validation. 10 unit tests and 0 doctests passed. |
| `cargo doc -p renderer --no-deps` | Not run | No rustdoc or public re-export organization changed. |

Runtime smoke and headless capture were not run because this phase made no visible renderer behavior changes.

## Residuals

- Existing renderer dead-code warning volume remains out of scope.
- Existing internal `TODO` and `pending` scan hits remain out of scope for this docs hardening phase.
- Material override strings remain metadata only until later material tooling resolves them into live renderer behavior.
- Camera helper types remain root-level compatibility exports outside the beginner prelude.

## Safe Adjacent Hygiene

- Ran `cargo fmt` to apply standard formatting for the new integration test imports/assertions.

## Artifacts Touched

- `.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-09-facade-api-contract/reports/phase-03-friction-hardening.md`
- `.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-09-facade-api-contract/artifacts/validation-summary.json`
