# Phase 04 Worker Directive: Validation And Closeout Evidence

## Objective

Run the full Sprint 05 validation suite, collect runtime/debug/headless draw evidence, reconcile the validation summary, and prepare closeout artifacts for the main thread.

## User-Visible Outcome

The sprint is either ready for final acceptance with evidence or clearly blocked with exact failed gates and residual handling.

## Editable Targets

- `.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-05-event-system-lifecycle/artifacts/validation-summary.json`
- `.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-05-event-system-lifecycle/validation/phase-04-validation-report.md`
- `.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-05-event-system-lifecycle/validation/final-quality-review.md` only if acting as final validator; otherwise leave for final quality validator.
- `.internal-dev/debug_reports/sprint-05-event-system-lifecycle/`
- `.internal-dev/captures/sprint-05-event-system-lifecycle-headless-draw/`
- Changelog draft only if main thread/user confirms timing.
- Bug/residual docs only for discovered out-of-scope bugs.

## Forbidden Scope

- Do not implement feature fixes beyond trivial evidence/doc corrections allowed by validator policy.
- Do not use desktop screenshots as visual proof.
- Do not mark `fully_validated` unless all required validators and draw capture proof pass.
- Do not commit, push, or email; main thread owns those.
- Do not touch unrelated `.idea/engine.iml` or `.reasonix/`.

## Supporting Docs To Read

- `validation/README.md`
- `shared/validation-matrix.md`
- `.internal-dev/skills/engine-headless-capture-validation/SKILL.md`
- All phase validation reports.
- Current `artifacts/validation-summary.json`.

## Implementation Steps

1. Confirm phase 01-03 validation reports exist and passed.
2. Run the full compile/test suite.
3. Run root runtime debug timing smoke and record JSONL path.
4. Run true headless draw capture with `--capture_target=draw`.
5. Inspect capture PNG/JSON sidecars and record expected/actual visual result.
6. Run stale-reference sweep over docs and this sprint directory.
7. Update validation summary with command results, evidence paths, model/tooling constraints, residual risks, and final status.
8. If any gate fails, stop and write a remediation handoff instead of claiming closeout.
9. If all gates pass, prepare final quality review handoff for a fresh validator.

## Senior Guidance

- Treat capture success as evidence only after inspecting the output.
- If capture harness fails, repair the harness/evidence path first. Change product code only after evidence proves a real product bug.
- Keep statuses conservative. `final_quality_pending` is correct until the final validator reconciles all evidence.
- Main-thread email/report should summarize phase commits, pushed branch, validation evidence, and residuals.

## Acceptance Criteria

- All required commands have recorded pass/fail status.
- Runtime smoke has a debug JSONL evidence path.
- Draw capture has PNG and JSON sidecar evidence under the required directory.
- Stale sweep is clean or residuals are documented.
- Validation summary is internally consistent.
- Final quality review is completed or explicitly pending with no false completion claim.

## Negative Checks

- No `/tmp` evidence is treated as canonical.
- No `fully_validated` if final quality review or capture proof is missing.
- No untracked/dirty unrelated files are included in evidence claims.
- No changelog is created unless main thread/user confirms timing.

## Validation Commands

```bash
cargo check
cargo test -p engine_events
cargo test -p input
cargo test -p renderer
cargo test -p engine
cargo check -p renderer --examples
cargo check -p editor
cargo check -p dungeon_dogfood
cargo check -p engine_pack
RUST_LOG=debug timeout --signal=INT 60s cargo run -- --project apps/editor/sample_project/engine.project.toml --headless --record_debug=10 --record_debug_interval=50 --record_debug_path=.internal-dev/debug_reports/sprint-05-event-system-lifecycle/root-runtime-events-timing.jsonl
RUST_LOG=info timeout --signal=INT 60s cargo run -- --project apps/editor/sample_project/engine.project.toml --headless --capture_frames=3 --capture_frame_start=5 --capture_frame_interval=5 --capture_target=draw --capture_dir=.internal-dev/captures/sprint-05-event-system-lifecycle-headless-draw
rg -n "/tmp|pending|planned|not implemented|TODO|agent id|desktop screenshot|playwright" docs .internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-05-event-system-lifecycle
```

## Evidence Expectations

- Phase report path: `.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-05-event-system-lifecycle/validation/phase-04-validation-report.md`
- Evidence index path: `.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-05-event-system-lifecycle/artifacts/validation-summary.json`
- Capture directory: `.internal-dev/captures/sprint-05-event-system-lifecycle-headless-draw/`
- Debug report: `.internal-dev/debug_reports/sprint-05-event-system-lifecycle/root-runtime-events-timing.jsonl`

## Stop Conditions

- Stop on missing phase reports.
- Stop on failed compile/test unless failure is clearly unrelated and main thread accepts residual handling.
- Stop if draw capture target is unavailable; record `TOOLING_CONSTRAINT`.
- Stop if validation summary contradicts evidence.

## Do Not Close Unless

- Full validation has run.
- Evidence paths are durable and inspected.
- Validation summary status matches the real state.
- Final quality validator has passed or the sprint remains explicitly pending.
