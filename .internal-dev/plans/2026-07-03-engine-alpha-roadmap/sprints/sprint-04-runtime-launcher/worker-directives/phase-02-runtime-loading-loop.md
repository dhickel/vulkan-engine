# Phase 02 Worker Directive: Runtime Loading Loop

## Objective

Connect the root launcher CLI to project/package/startup-scene loading and implement interactive/headless runtime render loops for the sample project.

## User-Visible Outcome

This works from the workspace root:

```bash
cargo run -- --project apps/editor/sample_project/engine.project.toml
```

And validation can run the same project headlessly through the root binary:

```bash
RUST_LOG=info timeout --signal=INT 60s cargo run -- \
  --project apps/editor/sample_project/engine.project.toml \
  --headless \
  --capture_target draw \
  --capture_frames 1 \
  --capture_frame_start 5 \
  --capture_dir .internal-dev/captures/sprint-04-runtime-launcher/phase-02-smoke
```

## Direct Editable Targets

Primary:

- `src/main.rs`
- Root launcher modules added in Phase 01, likely:
  - `src/launch.rs`
  - `src/runtime.rs`

Possible narrow renderer facade targets only if proven necessary:

- `src/renderer/src/api/mod.rs`
- `src/renderer/src/lib.rs`
- `src/renderer/src/api/config.rs`

Tests:

- Root package tests for loading/path/capture option behavior.
- Renderer validation tests only if a facade addition is required.

Evidence:

- `validation/phase-02-validation-report.md`
- `artifacts/validation-summary.json`
- `.internal-dev/captures/sprint-04-runtime-launcher/phase-02-smoke/`
- `.internal-dev/debug_reports/sprint-04-runtime-launcher/` if debug smoke is run.

Forbidden:

- Broad `src/renderer/src/vulkan/*` changes.
- Dynamic Rust hot reload, scripting, event system, physics, audio.
- Dogfood migration.
- Editor UI changes.

## Supporting Docs To Read

- `00-specification-lock.md`
- `02-target-design.md`
- `shared/implementation-notes.md`
- `apps/editor/src/main.rs`:
  - `run_headless_editor`;
  - `load_project_context`;
  - `load_enabled_project_packages`;
  - `load_startup_scene`;
  - capture/debug option application helpers.
- `src/renderer/src/api/renderer.rs`
- `src/renderer/src/data/asset_registry.rs`
- `src/renderer/src/api/scene.rs`
- `.internal-dev/skills/engine-headless-capture-validation/SKILL.md`

## Senior-Engineer Guidance

- Use existing renderer validators/facade APIs. Do not write a second TOML/JSON schema validator.
- Initialize package manifests before scene load so durable asset IDs resolve.
- Prefer a simple runtime context struct over global state.
- Headless mode should not create a winit window. Use `Renderer::new_headless`.
- Windowed mode can mirror the editor/example event loop, minus editor UI and command history.
- Use a bounded headless frame budget based on requested capture frame/count/interval, as the editor does.
- Return explicit errors for missing project, missing scene, missing package, scene load failure, and capture failure.
- If `default_environment` is easy to load through existing APIs, support it; if not, document it as deferred/residual instead of adding broad renderer changes.

## Implementation Steps

1. Build a runtime project-load helper:
   - validate project with file checks;
   - load `Project`;
   - compute `project_root`;
   - resolve startup scene from `--scene` or `project.startup_scene`;
   - fail if no scene is available;
   - validate scene against project registry where current APIs support it.
2. Build `RendererConfig` from project settings:
   - app name from project name;
   - width/height from settings;
   - headless from CLI;
   - asset policy aligned with editor's package path;
   - preload startup scene disabled for project-driven launch unless impossible.
3. Implement enabled-package loading:
   - iterate enabled project packages;
   - resolve manifest path under project root;
   - call `load_package_manifest_with_expected_id`;
   - collect useful counts/logs.
4. Implement scene loading:
   - call `Scene::load(scene_path, &mut renderer.assets())`;
   - if scene root is absent, either render as-is or add a non-editor-specific root only if required. Do not call it editor-specific.
5. Implement windowed loop:
   - create event loop/window with project dimensions/title;
   - initialize renderer;
   - install default FPS input if appropriate;
   - handle close/escape/resize/redraw;
   - call `renderer.update_input` and `renderer.render_scene`;
   - keep the loop simple and app-neutral.
6. Implement headless loop:
   - initialize `Renderer::new_headless`;
   - apply capture/debug launch options;
   - render via `render_scene_headless`;
   - count successful captures using `last_frame_capture_status`;
   - fail on capture failure or backend-not-implemented;
   - exit success when expected captures complete;
   - if no captures requested, run a small bounded smoke frame count and exit success if frames render.
7. Add/extend tests for:
   - project path resolution;
   - missing project file;
   - missing startup scene fixture;
   - package expected ID mismatch if feasible using fixtures;
   - headless frame-budget calculation;
   - no runtime handles in authored scene data if touched.
8. Run validation commands and update evidence summary conservatively.

## Acceptance Criteria

- Root launcher starts sample project outside the editor.
- Root headless command creates draw-target capture under the phase evidence path.
- Missing/invalid project and capture inputs fail cleanly.
- Enabled package manifests load before startup scene.
- Root windowed loop does not depend on editor UI modules.
- No broad renderer internals changes are required.

## Negative Criteria

- Do not shell out to `cargo run -p editor`.
- Do not use renderer examples as the runtime implementation.
- Do not accept present capture as proof.
- Do not migrate dogfood.
- Do not introduce runtime handle serialization into project/package/scene data.

## Validation Commands

```bash
cargo fmt --check
cargo check
cargo test -p engine
cargo test -p renderer
cargo run -p engine_pack -- validate-project apps/editor/sample_project/engine.project.toml
cargo run -p engine_pack -- validate-scene apps/editor/sample_project/scenes/start.engine.scene.json --project apps/editor/sample_project/engine.project.toml
RUST_LOG=info timeout --signal=INT 60s cargo run -- --project apps/editor/sample_project/engine.project.toml --headless --capture_target draw --capture_frames 1 --capture_frame_start 5 --capture_dir .internal-dev/captures/sprint-04-runtime-launcher/phase-02-smoke
cargo run -- --project .internal-dev/does-not-exist/engine.project.toml
cargo run -- --project apps/editor/sample_project/engine.project.toml --capture_target swapchain
git diff --check
```

The two negative `cargo run` commands must return non-zero controlled errors.

## Capture Evidence Requirements

For the phase smoke capture:

- inspect at least one sidecar JSON;
- require `status == "succeeded"`;
- require `capture_target == "draw"`;
- require positive extent;
- require PNG path exists.

Record paths in `validation/phase-02-validation-report.md`.

## Commit/Push/Report Gates

- Commit only after Phase 02 validator passes.
- Commit scope should include runtime launcher code/tests and phase evidence.
- Do not push unless the orchestrator opens the push gate.
- Do not send reports/email from this worker.

## Stop Conditions

- Stop if true headless draw capture fails due to environment/tooling; record `TOOLING_CONSTRAINT`.
- Stop if runtime loading requires broad renderer/Vulkan redesign.
- Stop if sample project cannot load without changing package/scene schema.
- Stop if dogfood migration appears necessary to pass Phase 02; it is not in scope.

## Do Not Close Unless

- Root launcher can load sample project in headless mode.
- Draw-target sidecar proof exists for root `cargo run`.
- Negative CLI/runtime checks are documented.
- Phase validation report exists.
- Evidence summary remains conservative.
