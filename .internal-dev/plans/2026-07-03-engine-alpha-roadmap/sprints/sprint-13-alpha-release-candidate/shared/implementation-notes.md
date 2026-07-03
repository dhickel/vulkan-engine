# Implementation Notes

## Protected State

Do not edit:

- `.idea/engine.iml`
- `.reasonix/`
- `.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/SPRINT-TRACKER.md`
- active Sprint 09 files while Sprint 09 remains active, except read-only inspection

Current Sprint 09 active-looking files observed during planning:

- `src/renderer/examples/api_test.rs`
- `src/renderer/examples/common/mod.rs`
- `src/renderer/examples/demo_async_loading.rs`
- `src/renderer/src/api/mod.rs`
- `src/renderer/src/lib.rs`
- `src/renderer/tests/integration.rs`
- `src/renderer/src/api/prelude.rs`

If Sprint 13 executes after Sprint 09 is merged and these files are no longer active local work, the main thread should state that explicitly before workers edit them.

## Evidence Paths

- Canonical evidence index: `.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-13-alpha-release-candidate/artifacts/validation-summary.json`
- Phase reports: `.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-13-alpha-release-candidate/reports/phase-XX-*.md`
- Phase validation reports: `.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-13-alpha-release-candidate/validation/phase-XX-validation-report.md`
- Capture root: `.internal-dev/captures/sprint-13-alpha-release-candidate/`
- Debug report root: `.internal-dev/debug_reports/sprint-13-alpha-release-candidate/`
- Fresh validation root: `.internal-dev/fresh-clone-validation/sprint-13/`

## Standard Commands

```sh
cargo fmt --check
cargo check
cargo check -p renderer
cargo check -p renderer --examples
cargo check -p input
cargo check -p engine
cargo check -p editor
cargo check -p dungeon_dogfood
cargo check -p engine_pack --locked
cargo test -p input
cargo test -p engine
cargo test -p engine_pack --locked
```

Sample project validation:

```sh
cargo run -p engine_pack -- validate-project apps/editor/sample_project/engine.project.toml
cargo run -p engine_pack -- validate-scene apps/editor/sample_project/scenes/start.engine.scene.json --project apps/editor/sample_project/engine.project.toml
cargo run -p engine_pack -- pack apps/editor/sample_project/engine.project.toml --out .internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-13-alpha-release-candidate/artifacts/sample-pack
```

Root runtime capture:

```sh
RUST_LOG=info timeout --signal=INT 60s cargo run -- \
  --project apps/editor/sample_project/engine.project.toml \
  --headless \
  --capture_target draw \
  --capture_frames 3 \
  --capture_frame_start 5 \
  --capture_frame_interval 5 \
  --capture_dir .internal-dev/captures/sprint-13-alpha-release-candidate/sample-runtime-draw
```

Editor capture:

```sh
RUST_LOG=info timeout --signal=INT 60s cargo run -p editor -- \
  --project apps/editor/sample_project/engine.project.toml \
  --headless \
  --capture_target draw \
  --capture_frames 3 \
  --capture_frame_start 5 \
  --capture_frame_interval 5 \
  --capture_dir .internal-dev/captures/sprint-13-alpha-release-candidate/editor-sample-draw
```

Dogfood full-content windowed smoke:

```sh
DUNGEON_DOGFOOD_FAST_STARTUP=0 \
DUNGEON_DOGFOOD_LOAD_PROPS=1 \
DUNGEON_DOGFOOD_LOAD_CUSTOM_ENV=1 \
RUST_LOG=info timeout --signal=INT 60s cargo run -p dungeon_dogfood -- --level generated_sprawl
```

Dogfood visual capture target, to be implemented or verified:

```sh
DUNGEON_DOGFOOD_FAST_STARTUP=0 \
DUNGEON_DOGFOOD_LOAD_PROPS=1 \
DUNGEON_DOGFOOD_LOAD_CUSTOM_ENV=1 \
RUST_LOG=info timeout --signal=INT 60s cargo run -p dungeon_dogfood -- \
  --level generated_sprawl \
  --headless \
  --capture_target draw \
  --capture_frames 3 \
  --capture_frame_start 5 \
  --capture_frame_interval 5 \
  --capture_dir .internal-dev/captures/sprint-13-alpha-release-candidate/dogfood-draw
```

## Stale Reference Sweep

Use targeted scans over changed docs and this sprint directory:

```sh
rg -n "/tmp|desktop screenshot|present-target|pending|planned|not implemented|TODO|migration stub|fresh clone TBD|release TBD" README.md docs apps/dungeon_dogfood .internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-13-alpha-release-candidate
```

Expected matches are allowed only when they are explicit known issues, historical notes, or command examples with a justified context.

