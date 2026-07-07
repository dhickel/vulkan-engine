# Sprint 04 Validation Matrix

## Status Rules

`artifacts/validation-summary.json` must stay conservative:

- `planned`: plan exists, implementation not started.
- `implementation_in_progress`: implementation active.
- `phase_validation_failed`: one or more phase validators failed.
- `code_validation_passed_capture_pending`: compile/test checks passed but draw-target capture proof is missing.
- `capture_failed`: root launcher capture command failed or sidecar predicates failed.
- `final_quality_review_pending`: phase validators passed but final reconciliation is not done.
- `fully_validated`: only after final reconciliation passes with no unresolved blocking residuals.
- `blocked_tooling_constraint`: required model/tool/runtime support was unavailable and no approved fallback exists.

## Phase 01: Runtime CLI

Required checks:

```bash
cargo fmt --check
cargo check -p engine
cargo test -p engine
cargo run -- --help
cargo run -- --project apps/editor/sample_project/engine.project.toml --capture_target swapchain
git diff --check
```

Pass criteria:

- Root binary no longer prints migration-stub guidance for normal help.
- `--help` exits `0`.
- Missing required values and invalid capture targets fail with controlled errors.
- Parser tests cover accepted forms and negative forms.

Fail criteria:

- Unknown flags are silently ignored.
- Invalid capture options panic.
- Root CLI depends on editor UI internals.

## Phase 02: Runtime Loading Loop

Required checks:

```bash
cargo fmt --check
cargo check
cargo test -p engine
cargo test -p renderer
cargo run -p engine_pack -- validate-project apps/editor/sample_project/engine.project.toml
cargo run -p engine_pack -- validate-scene apps/editor/sample_project/scenes/start.engine.scene.json --project apps/editor/sample_project/engine.project.toml
RUST_LOG=info timeout --signal=INT 60s cargo run -- --project apps/editor/sample_project/engine.project.toml --headless --capture_target draw --capture_frames 1 --capture_frame_start 5 --capture_dir .internal-dev/captures/sprint-04-runtime-launcher/phase-02-smoke
git diff --check
```

Pass criteria:

- Root launcher loads the sample project, enabled package, and startup scene outside the editor.
- Headless path creates at least one draw-target capture through root `cargo run`.
- Missing project/startup scene/package paths produce controlled errors.
- Windowed path is implemented without editor UI dependencies.

Fail criteria:

- Uses editor binary as the runtime path.
- Loads scene before package manifests.
- Accepts present-target capture as visual proof.
- Requires renderer internals redesign.

## Phase 03: Dev Loop Docs

Required checks:

```bash
cargo fmt --check
cargo check -p engine
cargo check -p editor
cargo check -p engine_pack --locked
git diff --check
rg -n "migration stub|runtime project launcher.*deferred|dynamic Rust hot reload|renderer examples.*only runtime|cargo run\\` prints" README.md docs apps/dungeon_dogfood
```

Pass criteria:

- README and API docs point project users to root `engine` launcher.
- Renderer examples remain documented as examples/diagnostics.
- App crates under `apps/<name>` are documented as the custom Rust loop.
- Dogfood status is documented as custom app path for now.
- Deferred systems are explicit and not overclaimed.

Fail criteria:

- Docs still tell users root `cargo run` only prints migration guidance.
- Docs claim dynamic Rust hot reload, scripting, event system, physics, audio, or dogfood migration is implemented.
- Docs remove useful renderer diagnostic commands.

## Phase 04: Capture Closeout

Required checks:

```bash
cargo fmt --check
cargo check
cargo check -p renderer --examples
cargo check -p editor
cargo check -p engine_pack --locked
cargo test -p engine
cargo test -p renderer
cargo test -p engine_pack --locked
RUST_LOG=info timeout --signal=INT 60s cargo run -- --project apps/editor/sample_project/engine.project.toml --headless --capture_target draw --capture_frames 3 --capture_frame_start 5 --capture_frame_interval 5 --capture_dir .internal-dev/captures/sprint-04-runtime-launcher/headless-draw
RUST_LOG=debug timeout --signal=INT 60s cargo run -- --project apps/editor/sample_project/engine.project.toml --headless --capture_target draw --capture_frames 1 --capture_frame_start 5 --capture_dir .internal-dev/captures/sprint-04-runtime-launcher/debug-smoke --record_debug 10 --record_debug_interval 50 --record_debug_path .internal-dev/debug_reports/sprint-04-runtime-launcher/root-runtime-timing.jsonl
git diff --check
python -m json.tool .internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-04-runtime-launcher/artifacts/validation-summary.json >/dev/null
```

Sidecar predicate checks:

- At least one JSON sidecar under `.internal-dev/captures/sprint-04-runtime-launcher/headless-draw`.
- Each accepted sidecar has:
  - `status == "succeeded"`;
  - `capture_target == "draw"`;
  - non-present draw format, expected `R16G16B16A16_SFLOAT` unless code documents another draw image format;
  - positive extent;
  - existing PNG path.

Stale-reference sweep:

```bash
rg -n "migration stub|runtime project launcher.*deferred|present-target proof|desktop screenshot|dynamic Rust hot reload implemented|scripting implemented|physics implemented|audio implemented|TODO|not implemented" README.md docs .internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-04-runtime-launcher
```

Pass criteria:

- Final validation report reconciles compile/test/runtime/doc/capture evidence.
- Sprint tracker can be updated to the correct post-validation state by the orchestrator.
- Changelog timing is handled according to repo guidance.

Fail criteria:

- Evidence summary status contradicts validation reports.
- Capture proof lacks draw-target sidecars.
- Required reports are missing.
- Stale docs still point users away from the root launcher.
