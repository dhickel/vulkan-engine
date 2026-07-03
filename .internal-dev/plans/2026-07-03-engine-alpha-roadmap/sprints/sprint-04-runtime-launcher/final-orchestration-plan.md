# Sprint 04 Final Orchestration Plan

## Dispatch Order

1. Dispatch Phase 01 implementation worker:
   - directive: `worker-directives/phase-01-runtime-cli.md`
   - validation report: `validation/phase-01-validation-report.md`
2. After Phase 01 validator passes, dispatch Phase 02 implementation worker:
   - directive: `worker-directives/phase-02-runtime-loading-loop.md`
   - validation report: `validation/phase-02-validation-report.md`
3. After Phase 02 validator passes, dispatch Phase 03 implementation worker:
   - directive: `worker-directives/phase-03-dev-loop-docs.md`
   - validation report: `validation/phase-03-validation-report.md`
4. After Phase 03 validator passes, dispatch Phase 04 closeout worker:
   - directive: `worker-directives/phase-04-capture-closeout.md`
   - validation report: `validation/phase-04-validation-report.md`

## Model Defaults

Use the current advanced-planner/orchestration defaults unless the user overrides:

- implementation workers: `gpt-5.3`, high reasoning per active developer instruction;
- phase validators: `gpt-5.5`, high reasoning;
- second-failure targeted repair escalation: `gpt-5.5`, high reasoning.

If a requested model/tool is unavailable, record `TOOLING_CONSTRAINT` and stop for user approval before substituting.

## Phase Gates

Every phase requires:

- worker completion summary;
- validation report;
- command evidence;
- updated conservative `artifacts/validation-summary.json` when evidence changes;
- no unresolved blocking findings before dependent phase dispatch.

## Remediation Routing

- Code defect: fresh scoped repair worker unless the validator identifies a trivial mechanical fix.
- Docs/evidence defect: fresh scoped repair worker unless it is a one-place validator/report typo.
- Capture harness defect: repair command/evidence parsing first; change product code only if capture evidence proves a product bug.
- Plan defect: return to planning and revise artifacts before more coding.
- Validator error: correct checklist or use fresh validator.

If the same targeted issue fails validation twice after repairs, escalate to a fresh high-reasoning repair worker.

## Required Final Evidence

Canonical evidence index:

- `artifacts/validation-summary.json`

Required capture directory:

- `.internal-dev/captures/sprint-04-runtime-launcher/headless-draw`

Required debug report path:

- `.internal-dev/debug_reports/sprint-04-runtime-launcher/root-runtime-timing.jsonl`

Required final root command:

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

Sidecar predicates:

- `status == "succeeded"`;
- `capture_target == "draw"`;
- draw-target format such as `R16G16B16A16_SFLOAT`;
- positive extent;
- PNG path exists and is non-empty.

Present-target captures and desktop screenshots are not valid proof.

## Final Quality Review

This is medium work, so a full large-suite xhigh final validator is not mandatory. The Phase 04 validator must still reconcile:

- all phase directives;
- all validation reports;
- changed code/docs;
- command evidence;
- headless draw capture evidence;
- debug timing evidence;
- stale-reference sweep;
- validation-summary consistency.

Use a fresh final validator if:

- remediation touched multiple domains after Phase 04 began;
- any phase validator missed an obvious issue;
- final criteria changed;
- capture evidence was repaired after an initial final failure.

## Stale-Reference Sweep

Before final status promotion, run and classify:

```bash
rg -n "migration stub|runtime project launcher.*deferred|present-target proof|desktop screenshot|dynamic Rust hot reload implemented|scripting implemented|physics implemented|audio implemented|TODO|not implemented" README.md docs .internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-04-runtime-launcher
```

Hits are not automatically failures, but unresolved stale claims are failures.

## Closeout Gates

Do not mark Sprint 04 closed until:

- phase validators pass;
- final capture proof passes;
- `artifacts/validation-summary.json` is consistent;
- known residuals are fixed or tracked;
- changelog timing is confirmed and changelog is created if required;
- sprint tracker is updated to `closed`, `validating`, or `blocked` as evidence supports;
- commit/push/report gates are handled by the main-thread orchestrator.

Report/email sending is out of band and belongs to the main thread through the appropriate email workflow if requested.

## Stop Rules

- Stop if true headless draw-target capture cannot be produced.
- Stop if required validation tooling/model is unavailable.
- Stop if implementation requires broad renderer redesign.
- Stop if work expands into hot reload, scripting, event system, physics, audio, or dogfood migration.
- Stop if final evidence status would overclaim relative to actual validators and residuals.
