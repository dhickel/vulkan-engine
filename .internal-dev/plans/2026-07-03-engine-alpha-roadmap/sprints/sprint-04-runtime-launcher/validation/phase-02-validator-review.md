# Phase 02 Validator Review: Runtime Loading Loop

Date: 2026-07-03
Branch: `sprint/alpha-04-runtime-launcher`
Validator scope: Sprint 04 Phase 02 after implementation and narrow renderer doctest-fence repair.

## Findings

No blocking findings.

Non-blocking residuals:

- Existing renderer dead-code warnings remain visible during `cargo check`, `cargo test -p engine`, `cargo test -p renderer`, and runtime CLI commands. These warnings predate the Phase 02 runtime-loading behavior and do not block the phase.
- Phase 03 documentation is still pending, so sprint docs may still contain pre-runtime-launcher wording until the docs phase runs.
- The user-facing shorthand `artifacts/validation-summary.json` resolves in this plan to `.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-04-runtime-launcher/artifacts/validation-summary.json`; no root-level `artifacts/validation-summary.json` file exists.

## Scope And Code Review Evidence

The implementation stays inside the Phase 02 boundary:

- Root launcher reuses `renderer::CaptureTarget` through `src/launch.rs` rather than maintaining a duplicate enum.
- `src/main.rs` calls `runtime::run(options)` and initializes logging, replacing the migration-stub runtime error.
- `src/runtime.rs` resolves and validates the project, resolves the startup scene, builds `RendererConfig` from project name/window settings, and loads enabled package manifests before validating/loading the scene.
- Headless runtime uses `Renderer::new_headless` and `render_scene_headless`.
- Windowed runtime uses its own winit `EventLoop`, `WindowBuilder`, `Renderer::new`, input update, resize handling, and `render_scene`; no editor binary or renderer example shellout was found.
- Renderer doctest changes in `data_cache.rs`, `vk_storage.rs`, `vk_descriptor.rs`, `vk_util.rs`, and `vk_types.rs` are limited to `text` and `ignore` fence annotations for prose diagrams and illustrative Vulkan snippets.

## Criteria Results

| Criterion | Result | Evidence |
| --- | --- | --- |
| Root launcher loads sample project, enabled package manifests before startup scene, and validates scene/project boundaries | Pass | `src/runtime.rs` calls project validation, package manifest loading via `load_package_manifest_with_expected_id`, then scene validation with known package asset IDs before `Scene::load`; `cargo run -p engine_pack` project and scene validations passed. |
| Windowed path exists and is not editor/examples shellout | Pass | `run_windowed` constructs a winit event loop/window and calls `Renderer::new`/`render_scene`; static search found no `cargo run -p editor` shellout. |
| Headless path uses true headless renderer | Pass | `run_headless` uses `Renderer::new_headless` and `render_scene_headless`; capture sidecars report draw target and `R16G16B16A16_SFLOAT`. |
| Root CLI uses `renderer::CaptureTarget` cleanly | Pass | `src/launch.rs` exports `renderer::CaptureTarget`; invalid `swapchain` target exits 2 with controlled usage text. |
| Tests cover non-Vulkan helpers and error cases | Pass | `cargo test -p engine` passed 17 tests covering parse forms, invalid capture options, project resolution, missing project/scene, package ID mismatch, and headless budget calculation. |
| Doctest-fence repair is narrow and appropriate | Pass | Renderer diffs only changed doc fence tags to `text`/`ignore`; `cargo test -p renderer` passed with 5 doctests ignored. |
| Capture proof is true headless draw-target sidecar evidence | Pass | Both `phase-02-smoke` and `phase-02-smoke-rerun` sidecars passed status/target/format/extent/png existence predicates. No present/desktop proof was used. |
| Validation summary is conservative and consistent | Pass | Summary remains `fully_validated: false`; Phase 02 is now marked validator-passed while Phase 03 and Phase 04 remain pending. |

## Commands And Evidence

| Command/check | Result |
| --- | --- |
| `cargo fmt --check` | Pass, exit 0 |
| `cargo check` | Pass, exit 0; renderer dead-code warnings observed |
| `cargo test -p engine` | Pass, exit 0; 17 tests passed |
| `cargo test -p renderer` | Pass, exit 0; 150 lib tests, 17 integration tests, 5 ignored doctests |
| `cargo run -p engine_pack -- validate-project apps/editor/sample_project/engine.project.toml` | Pass, exit 0; `valid[project]: apps/editor/sample_project/engine.project.toml (project.editor_sample)` |
| `cargo run -p engine_pack -- validate-scene apps/editor/sample_project/scenes/start.engine.scene.json --project apps/editor/sample_project/engine.project.toml` | Pass, exit 0; `valid[scene]: apps/editor/sample_project/scenes/start.engine.scene.json` |
| `cargo run -- --project .internal-dev/does-not-exist/engine.project.toml` | Expected controlled failure, exit 1; runtime error reports missing project file |
| `cargo run -- --project apps/editor/sample_project/engine.project.toml --capture_target swapchain` | Expected controlled failure, exit 2; usage error lists accepted `present` and `draw` targets |
| Sidecar predicate check for `phase-02-smoke/*.json` and `phase-02-smoke-rerun/*.json` | Pass, exit 0; both sidecars have `status=succeeded`, `capture_target=draw`, `format=R16G16B16A16_SFLOAT`, positive extent, and existing non-empty PNGs |
| `git diff --check` | Pass, exit 0 |
| `jq empty .internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-04-runtime-launcher/artifacts/validation-summary.json` | Pass, exit 0 |

Capture evidence accepted:

- `.internal-dev/captures/sprint-04-runtime-launcher/phase-02-smoke/editor-sample-project-frame-5-draw-seq-0000.json`
- `.internal-dev/captures/sprint-04-runtime-launcher/phase-02-smoke/editor-sample-project-frame-5-draw-seq-0000.png`
- `.internal-dev/captures/sprint-04-runtime-launcher/phase-02-smoke-rerun/editor-sample-project-frame-5-draw-seq-0000.json`
- `.internal-dev/captures/sprint-04-runtime-launcher/phase-02-smoke-rerun/editor-sample-project-frame-5-draw-seq-0000.png`

## Phase Decision

Phase 02 passes validation. Phase 03 may proceed.

Do not promote the sprint to `fully_validated`: Phase 03 documentation, Phase 04 capture closeout, and final reconciliation remain pending.
