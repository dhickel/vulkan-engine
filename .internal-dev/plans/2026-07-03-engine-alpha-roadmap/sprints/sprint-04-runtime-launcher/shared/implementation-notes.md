# Shared Implementation Notes

## Read First

- `AGENTS.md`
- `.internal-dev/AGENTS.md`
- `src/renderer/AGENTS.md`
- `.internal-dev/skills/engine-headless-capture-validation/SKILL.md`
- `00-specification-lock.md`
- `01-current-state-analysis.md`
- `02-target-design.md`
- The worker directive for the active phase.

## Source Truth

- Code is logical source of truth.
- Docs are intended truth.
- If docs disagree with live code, fix in-scope docs or record the mismatch in the phase validation report.

## Existing Patterns To Reuse

- `apps/editor/src/launch.rs`: launch option parsing, capture option validation, parser tests.
- `apps/editor/src/main.rs`: debug timing/capture application, project context loading, enabled package manifest loading, scene loading, headless render/capture loop.
- `src/renderer/src/api/renderer.rs`: `Renderer::new`, `Renderer::new_headless`, `render_scene`, `render_scene_headless`, capture/debug APIs.
- `src/renderer/src/data/asset_registry.rs`: `Project`, `ProjectSettings`, validation options and diagnostics.
- `src/renderer/src/api/scene.rs`: `Scene::load`.
- `tools/engine_pack/src/main.rs`: project/package/scene validation CLI behavior and stable diagnostics.

## Error Handling Guidance

- Do not panic for CLI/user errors.
- Return controlled stderr with stable wording useful to tests.
- Keep exit codes conventional:
  - `0` for help/success;
  - `2` for usage/argument errors;
  - non-zero for validation/runtime failures.
- Unknown root launcher flags should fail. The editor parser currently ignores unknowns; do not copy that behavior into the root launcher.

## Runtime Loading Guidance

- Validate before rendering. `Project::load` uses `check_files(false)`, so workers must call existing validation functions or equivalent file checks where needed.
- Load enabled package manifests before loading scenes with package-backed durable asset IDs.
- Use `load_package_manifest_with_expected_id` for package ID enforcement.
- Resolve project-relative paths from the `engine.project.toml` parent directory.
- Do not use path hints as identity. Durable asset IDs are identity.
- Do not serialize runtime handles.

## Renderer Guidance

- Avoid `src/renderer/src/vulkan/*` changes. Current facade APIs are enough unless implementation proves a hard blocker.
- Prefer `RendererConfig::default()` plus explicit overrides.
- Set `preload_startup_scene = false` for project-driven launch unless a worker proves the startup default is needed.
- Use `Renderer::new_headless` for headless validation. Do not create a hidden desktop window and call it headless.
- Use `render_scene_headless` in headless mode.

## Capture Guidance

- Required proof is true headless draw-target capture:

```bash
RUST_LOG=info timeout --signal=INT 60s cargo run -- \
  --project apps/editor/sample_project/engine.project.toml \
  --headless \
  --capture_target draw \
  --capture_frames 3 \
  --capture_frame_start 5 \
  --capture_frame_interval 5 \
  --capture_dir .internal-dev/captures/sprint-04-runtime-launcher/headless-draw
```

- Validate sidecar JSON, not only PNG existence.
- Present-target capture, desktop screenshots, and compositor screenshots fail the proof gate.
- Capture evidence belongs under `.internal-dev/captures/sprint-04-runtime-launcher/`.

## Documentation Guidance

- Replace stale root `cargo run` migration-stub wording after implementation.
- Keep renderer examples documented as examples/diagnostics.
- Add or update root runtime launcher docs with exact command examples.
- Document `apps/<name>` app crates as the default custom Rust dev loop.
- Explicitly defer dynamic Rust hot reload, scripting, event system, physics, audio, and dogfood migration.

## Evidence Guidance

- Phase reports go under:
  - `validation/phase-01-validation-report.md`
  - `validation/phase-02-validation-report.md`
  - `validation/phase-03-validation-report.md`
  - `validation/phase-04-validation-report.md`
- Keep `artifacts/validation-summary.json` conservative.
- Do not write `fully_validated` until all phase validation, capture proof, stale-reference sweep, and closeout gates pass.

## Commit/Push/Report Gates

- Each phase may produce a scoped commit only after its phase validator passes and evidence is recorded.
- Do not push unless the main-thread orchestrator says the phase commit/push gate is open.
- Do not send report email from implementation workers. If closeout/report is required, main-thread orchestration handles it through the appropriate email workflow.
