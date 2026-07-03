# Validation README

## Phase Reports

Each phase validator must write:

- Phase 01: `.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-05-event-system-lifecycle/validation/phase-01-validation-report.md`
- Phase 02: `.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-05-event-system-lifecycle/validation/phase-02-validation-report.md`
- Phase 03: `.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-05-event-system-lifecycle/validation/phase-03-validation-report.md`
- Phase 04: `.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-05-event-system-lifecycle/validation/phase-04-validation-report.md`
- Final quality: `.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-05-event-system-lifecycle/validation/final-quality-review.md`

## Validator Checklist

- Compare implementation against `00-specification-lock.md`, `02-target-design.md`, and the phase directive.
- Confirm application contract fit: input frame boundary, renderer facade ownership, root runtime lifecycle, docs truthfulness.
- Inspect tests for meaningful coverage, not only compile smoke.
- Confirm no unrelated `.idea/engine.iml` or `.reasonix/` changes were touched.
- Confirm event crate is Vulkan-free.
- Confirm physics/audio/scripting support is not overstated.
- Reconcile `artifacts/validation-summary.json` with actual evidence.

## Final Stale Sweep

Before final quality review, search docs and this sprint directory for stale references:

```bash
rg -n "/tmp|pending|planned|not implemented|TODO|agent id|superseded|playwright|desktop screenshot" docs .internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-05-event-system-lifecycle
```

Finding `planned` in the sprint status may be acceptable before final closeout, but final reports must not claim future work as complete.

## Headless Capture Requirement

Use true engine-owned draw-target capture:

```bash
RUST_LOG=info timeout --signal=INT 60s cargo run -- --project apps/editor/sample_project/engine.project.toml --headless --capture_frames=3 --capture_frame_start=5 --capture_frame_interval=5 --capture_target=draw --capture_dir=.internal-dev/captures/sprint-05-event-system-lifecycle-headless-draw
```

Evidence must include command, capture directory, PNG path(s), JSON sidecar path(s), expected visual result, actual observed result, and pass/fail/inconclusive status.

## Validation Summary Updates

Update `artifacts/validation-summary.json` after each phase validation and final quality review. Use conservative statuses:

- `planned`
- `phase_01_validated`
- `phase_02_validated`
- `phase_03_validated`
- `implementation_checks_passed`
- `draw_capture_pending`
- `final_quality_pending`
- `fully_validated`
- `validator_failed`
- `blocked_tooling_constraint`
