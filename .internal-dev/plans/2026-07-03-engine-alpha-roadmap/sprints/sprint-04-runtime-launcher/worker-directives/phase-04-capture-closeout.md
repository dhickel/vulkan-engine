# Phase 04 Worker Directive: Capture Closeout

## Objective

Produce final Sprint 04 runtime/capture evidence, reconcile validation, update the evidence index, and prepare repo closeout gates.

## User-Visible Outcome

The sprint has durable proof that the root `engine` launcher runs the sample project outside the editor and produces true headless draw-target capture evidence.

## Direct Editable Targets

Primary:

- `artifacts/validation-summary.json`
- `validation/phase-04-validation-report.md`
- `.internal-dev/captures/sprint-04-runtime-launcher/`
- `.internal-dev/debug_reports/sprint-04-runtime-launcher/`

Possible closeout docs/artifacts if orchestrator opens the gate:

- `.internal-dev/changelogs/<date>-sprint-04-runtime-launcher.md`
- `.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/SPRINT-TRACKER.md`

Product code/docs:

- Only narrowly scoped remediation for findings from final validation. New features are forbidden.

Forbidden:

- New runtime features.
- Present-target or desktop screenshot proof substitution.
- Dogfood migration.
- Hot reload/scripting/event/physics/audio implementation.

## Supporting Docs To Read

- All prior phase validation reports.
- `shared/validation-matrix.md`
- `.internal-dev/skills/engine-headless-capture-validation/SKILL.md`
- Current `artifacts/validation-summary.json`
- Changed docs from Phase 03.

## Senior-Engineer Guidance

- This phase is evidence and reconciliation first. Do not add product scope.
- Treat capture sidecars as authoritative proof for target/source predicates.
- Use root `cargo run`, not `cargo run -p editor` or renderer examples, for final runtime proof.
- Keep final status conservative if any required gate is skipped or blocked.
- If Vulkan/headless support fails on the host, record `TOOLING_CONSTRAINT` and stop; do not use desktop proof as a substitute.

## Implementation Steps

1. Run full compile/test command set from the validation matrix.
2. Run final root launcher draw-target capture:

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

3. Run debug-record smoke:

```bash
RUST_LOG=debug timeout --signal=INT 60s cargo run -- \
  --project apps/editor/sample_project/engine.project.toml \
  --headless \
  --capture_target draw \
  --capture_frames 1 \
  --capture_frame_start 5 \
  --capture_dir .internal-dev/captures/sprint-04-runtime-launcher/debug-smoke \
  --record_debug 10 \
  --record_debug_interval 50 \
  --record_debug_path .internal-dev/debug_reports/sprint-04-runtime-launcher/root-runtime-timing.jsonl
```

4. Inspect sidecar JSON predicates:
   - `status == "succeeded"`;
   - `capture_target == "draw"`;
   - draw-target format, expected `R16G16B16A16_SFLOAT` unless current code documents a different draw format;
   - positive extent;
   - PNG exists and is non-empty.
5. Run stale-reference sweep over docs and this sprint plan directory.
6. Reconcile all phase reports and residuals.
7. Update `artifacts/validation-summary.json`:
   - command results;
   - capture artifact directory;
   - sidecar paths;
   - validation reports;
   - residual risks;
   - final status.
8. If all gates pass and orchestrator confirms closeout timing, prepare changelog and tracker update. If timing is not confirmed, leave these as closeout gates for the main thread.

## Acceptance Criteria

- Full required command set passes or accepted residuals are explicitly documented.
- Root launcher final capture produces draw-target sidecars.
- Debug timing JSONL is created by root launcher when requested.
- Validation summary is parseable and internally consistent.
- Stale docs are fixed or classified.
- Phase 04 validation report exists.

## Negative Criteria

- Do not promote status to `fully_validated` if any validator failed or is missing.
- Do not promote status to `fully_validated` if capture proof is missing or present-target only.
- Do not hide residual failures behind a successful screenshot/capture.
- Do not create changelog/tracker closeout if orchestrator has not opened the repo closeout gate.

## Validation Commands

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
rg -n "migration stub|runtime project launcher.*deferred|present-target proof|desktop screenshot|dynamic Rust hot reload implemented|scripting implemented|physics implemented|audio implemented|TODO|not implemented" README.md docs .internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-04-runtime-launcher
```

## Evidence Expectations

Write `validation/phase-04-validation-report.md` with:

- commands and outcomes;
- capture directories;
- sidecar JSON paths and predicate results;
- debug JSONL path;
- stale-reference sweep classifications;
- final validation-summary status;
- residual risks and closeout gates.

## Commit/Push/Report Gates

- Commit closeout evidence only after Phase 04 validator passes.
- Changelog creation requires the repo closeout timing gate.
- Sprint tracker update requires all required gates pass or a clearly marked blocked/validating state.
- Push/report/email are orchestrator responsibilities, not worker responsibilities.

## Stop Conditions

- Stop on missing true headless draw-target proof.
- Stop on contradictory validation-summary status.
- Stop if final command failures cannot be attributed to accepted pre-existing residuals.
- Stop if closeout requires user timing confirmation.

## Do Not Close Unless

- Root `cargo run` proof exists.
- Draw-target sidecars pass predicates.
- Debug record smoke evidence exists or a blocker is recorded.
- All validation reports exist.
- Evidence summary is conservative and parseable.
