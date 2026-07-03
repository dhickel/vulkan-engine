# Implementation Notes

## Files To Read First

- `AGENTS.md`
- `.internal-dev/AGENTS.md`
- `.internal-dev/skills/engine-headless-capture-validation/SKILL.md`
- `docs/api/09-editor-asset-browser-and-wall-chunks.md`
- `apps/editor/src/app_state.rs`
- `apps/editor/src/main.rs`
- `apps/editor/src/panels.rs`
- `apps/editor/src/launch.rs`
- `src/renderer/src/scene/command.rs`
- `src/renderer/src/api/scene.rs`
- `src/renderer/src/data/asset_registry.rs`
- `tools/engine_pack/src/main.rs`
- `tools/engine_pack/tests/cli_validation.rs`

## Useful Commands

```bash
cargo fmt --check
git diff --check
cargo check
cargo check -p editor
cargo check -p renderer
cargo check -p renderer --examples
cargo check -p engine_pack --locked
cargo test -p editor
cargo test -p renderer scene
cargo test -p renderer asset_registry
cargo test -p engine_pack --locked
cargo run -p engine_pack -- validate-project apps/editor/sample_project/engine.project.toml
cargo run -p engine_pack -- validate-scene apps/editor/sample_project/scenes/start.engine.scene.json --project apps/editor/sample_project/engine.project.toml
```

Headless capture command template:

```bash
RUST_LOG=info timeout --signal=INT 60s cargo run -p editor -- --project apps/editor/sample_project/engine.project.toml --scene <saved-scene-copy> --headless --capture_frames 3 --capture_frame_start 5 --capture_frame_interval 5 --capture_dir .internal-dev/captures/sprint-03-editor-packaged-placement
```

Fallback renderer capture command template:

```bash
RUST_LOG=info timeout --signal=INT 60s cargo run -p renderer --example <capture-example> -- --headless --capture_frames 3 --capture_frame_start 5 --capture_frame_interval 5 --capture_dir .internal-dev/captures/sprint-03-editor-packaged-placement
```

## Evidence Paths

- Phase validation reports: `validation/phase-XX-validation-report.md`
- Final quality review: `validation/final-quality-review.md`
- Canonical summary: `artifacts/validation-summary.json`
- Headless capture setup/evidence: `.internal-dev/headless_capture_tests/` and `.internal-dev/captures/`

## Commit Push Email Gate Notes

Each phase should leave enough evidence for the main thread to commit, push, and send a report:

- changed files summary;
- commands run and results;
- validator report path;
- capture paths when applicable;
- residual risks;
- unrelated dirty files explicitly excluded.

The plan does not add email wait/send mechanics; those belong to the main thread.
