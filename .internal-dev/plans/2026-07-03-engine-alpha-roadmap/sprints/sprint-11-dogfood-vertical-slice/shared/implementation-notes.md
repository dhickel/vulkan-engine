# Implementation Notes

## Planning Boundary

This suite is planning only. Product code, tests, schemas, runtime config, docs outside this plan directory, and tracker updates are for later execution.

## Branch And Worktree

- Intended branch: `sprint/alpha-11-dogfood-vertical-slice`.
- Before implementation, run:

```sh
git status --short
git worktree list
```

- Do not overwrite unrelated local changes.
- Do not touch `.idea/engine.iml` or `.reasonix/`.
- Do not update `SPRINT-TRACKER.md`; main thread reconciles after review.

## Sprint 09/Sprint 10 Coordination

Current planning observed active Sprint 09 edits in renderer API/example files. Workers must refresh these files before editing:

- `src/renderer/examples/api_test.rs`
- `src/renderer/examples/common/mod.rs`
- `src/renderer/examples/demo_async_loading.rs`
- `src/renderer/src/api/mod.rs`
- `src/renderer/src/lib.rs`
- `src/renderer/tests/integration.rs`
- `src/renderer/src/api/prelude.rs`

If these files are still dirty or owned by Sprint 09 during execution, stop and ask the main thread for sequencing. Do not merge by guesswork.

## Evidence Paths

- Validation summary: `.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-11-dogfood-vertical-slice/artifacts/validation-summary.json`
- Phase reports: `.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-11-dogfood-vertical-slice/validation/phase-XX-validation-report.md`
- Runtime debug reports: `.internal-dev/debug_reports/sprint-11-dogfood-vertical-slice/`
- Captures: `.internal-dev/captures/sprint-11-dogfood-vertical-slice/`
- Temporary capture specs: `.internal-dev/headless_capture_tests/sprint-11-dogfood-vertical-slice/`
- Final closeout draft: `.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-11-dogfood-vertical-slice/reports/final-report.md`

## Canonical Commands To Preserve

General checks:

```sh
cargo check
cargo check -p renderer
cargo check -p renderer --examples
cargo check -p input
cargo test -p input
cargo check -p engine_pack
cargo test -p engine_pack
cargo check -p dungeon_dogfood
```

Data validation target shape:

```sh
cargo run -p engine_pack -- validate-package apps/dungeon_dogfood/assets/dogfood_dungeon.package.toml --expected-package-id dogfood_dungeon
cargo run -p engine_pack -- validate-project apps/dungeon_dogfood/engine.project.toml
cargo run -p engine_pack -- validate-scene apps/dungeon_dogfood/scenes/start.engine.scene.json --project apps/dungeon_dogfood/engine.project.toml
```

Debug timing target shape:

```sh
mkdir -p .internal-dev/debug_reports/sprint-11-dogfood-vertical-slice
DUNGEON_DOGFOOD_FAST_STARTUP=0 \
DUNGEON_DOGFOOD_LOAD_PROPS=1 \
DUNGEON_DOGFOOD_LOAD_CUSTOM_ENV=1 \
RUST_LOG=debug timeout --signal=INT 60s cargo run -p dungeon_dogfood -- \
  --project apps/dungeon_dogfood/engine.project.toml \
  --level generated_sprawl \
  --record_debug=10 \
  --record_debug_interval=50 \
  --record_debug_path=.internal-dev/debug_reports/sprint-11-dogfood-vertical-slice/dogfood-generated-sprawl-timing.jsonl
```

Headless capture target shape:

```sh
mkdir -p .internal-dev/captures/sprint-11-dogfood-vertical-slice
DUNGEON_DOGFOOD_FAST_STARTUP=0 \
DUNGEON_DOGFOOD_LOAD_PROPS=1 \
DUNGEON_DOGFOOD_LOAD_CUSTOM_ENV=1 \
RUST_LOG=info timeout --signal=INT 60s cargo run -p dungeon_dogfood -- \
  --project apps/dungeon_dogfood/engine.project.toml \
  --level generated_sprawl \
  --headless \
  --capture_target draw \
  --capture_frames=3 \
  --capture_frame_start=5 \
  --capture_frame_interval=5 \
  --capture_dir .internal-dev/captures/sprint-11-dogfood-vertical-slice/dogfood-generated-sprawl
```

If implementation changes command shape, update this file, worker reports, docs, and `artifacts/validation-summary.json` together.
