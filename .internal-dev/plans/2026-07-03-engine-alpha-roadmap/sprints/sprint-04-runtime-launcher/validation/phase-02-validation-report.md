# Phase 02 Validation Report: Runtime Loading Loop

Date: 2026-07-03
Branch: `sprint/alpha-04-runtime-launcher`
Worker scope: Sprint 04 Phase 02 only

## Summary

Phase 02 implementation connected the root `engine` launcher to project/package/startup-scene loading and added simple windowed/headless runtime loops. Headless validation used `Renderer::new_headless` and `render_scene_headless` through the root binary and produced accepted draw-target capture proof.

Status: implementation checks passed after a narrow doctest-fence repair for pre-existing renderer prose snippets.

## Commands

| Command | Result | Evidence |
| --- | --- | --- |
| `cargo fmt --check` | Passed | Exit 0 |
| `cargo check` | Passed | Exit 0; renderer dead-code warnings remain |
| `cargo test -p engine` | Passed | Exit 0; 17 tests passed |
| `cargo test -p renderer` | Passed | Exit 0 after marking pre-existing prose diagrams as `text` and illustrative Vulkan snippets as `ignore`; 150 lib tests, 17 integration tests, and doctests passed/ignored as expected |
| `cargo run -p engine_pack -- validate-project apps/editor/sample_project/engine.project.toml` | Passed | `valid[project]: apps/editor/sample_project/engine.project.toml (project.editor_sample)` |
| `cargo run -p engine_pack -- validate-scene apps/editor/sample_project/scenes/start.engine.scene.json --project apps/editor/sample_project/engine.project.toml` | Passed | `valid[scene]: apps/editor/sample_project/scenes/start.engine.scene.json` |
| `RUST_LOG=info timeout --signal=INT 60s cargo run -- --project apps/editor/sample_project/engine.project.toml --headless --capture_target draw --capture_frames 1 --capture_frame_start 5 --capture_dir .internal-dev/captures/sprint-04-runtime-launcher/phase-02-smoke` | Passed | Exit 0; true headless draw capture completed |
| `RUST_LOG=info timeout --signal=INT 60s cargo run -- --project apps/editor/sample_project/engine.project.toml --headless --capture_target draw --capture_frames 1 --capture_frame_start 5 --capture_dir .internal-dev/captures/sprint-04-runtime-launcher/phase-02-smoke-rerun` | Passed | Exit 0; true headless draw capture rerun completed after doctest-fence repair |
| `cargo run -- --project .internal-dev/does-not-exist/engine.project.toml` | Passed expected failure | Exit 1; controlled `runtime error: project file ... does not exist` |
| `cargo run -- --project apps/editor/sample_project/engine.project.toml --capture_target swapchain` | Passed expected failure | Exit 2; controlled usage error listing accepted `present` and `draw` targets |
| `jq -e 'select(.status == "succeeded" and .capture_target == "draw" and .extent.width > 0 and .extent.height > 0 and (.png_path \| length > 0))' .internal-dev/captures/sprint-04-runtime-launcher/phase-02-smoke/*.json` | Passed | Sidecar predicates matched |
| `git diff --check` | Passed | Exit 0 |
| `jq empty .internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-04-runtime-launcher/artifacts/validation-summary.json` | Passed | Exit 0 |

## Capture Proof

Capture directory:

- `.internal-dev/captures/sprint-04-runtime-launcher/phase-02-smoke/`
- `.internal-dev/captures/sprint-04-runtime-launcher/phase-02-smoke-rerun/`

Sidecar:

- `.internal-dev/captures/sprint-04-runtime-launcher/phase-02-smoke/editor-sample-project-frame-5-draw-seq-0000.json`

PNG:

- `.internal-dev/captures/sprint-04-runtime-launcher/phase-02-smoke/editor-sample-project-frame-5-draw-seq-0000.png`
- `.internal-dev/captures/sprint-04-runtime-launcher/phase-02-smoke-rerun/editor-sample-project-frame-5-draw-seq-0000.png`

Sidecar predicates:

- `status`: `succeeded`
- `capture_target`: `draw`
- `format`: `R16G16B16A16_SFLOAT`
- `extent`: `1440x900`
- `png_path`: `.internal-dev/captures/sprint-04-runtime-launcher/phase-02-smoke/editor-sample-project-frame-5-draw-seq-0000.png`
- PNG exists and is non-empty; `file` reports `PNG image data, 1440 x 900, 8-bit/color RGBA, non-interlaced`

This is true headless draw-target evidence. Desktop screenshots and present-target captures were not used.

## Residuals

- Existing renderer dead-code warnings remain visible during check/test runs and are outside Phase 02.
- Phase 03 documentation remains pending, so docs may still contain pre-Sprint-04 runtime wording.
- Debug timing flags are wired through the root runtime path, but a dedicated debug-record smoke was not part of the required Phase 02 command list and was not run.

## Conclusion

Phase 02 implementation criteria are satisfied from the worker side: the root launcher loads the sample project, validates project/package/scene boundaries before rendering, loads enabled package manifests before the startup scene, uses project settings in `RendererConfig`, supports controlled negative CLI errors, and produces accepted true headless draw-target capture proof.
