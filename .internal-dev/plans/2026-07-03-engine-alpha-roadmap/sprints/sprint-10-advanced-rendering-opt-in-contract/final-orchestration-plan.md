# Final Orchestration Plan

## Dispatch Order

1. Phase 01 worker: `worker-directives/phase-01-contract-audit.md`
2. Phase 01 validator writes `validation/phase-01-validation-report.md`
3. Phase 02 worker: `worker-directives/phase-02-feature-gate-docs.md`
4. Phase 02 validator writes `validation/phase-02-validation-report.md`
5. Phase 03 worker: `worker-directives/phase-03-named-advanced-surface.md`
6. Phase 03 validator writes `validation/phase-03-validation-report.md`
7. Phase 04 closeout worker: `worker-directives/phase-04-final-validation-closeout.md`
8. Phase 04 validator writes `validation/phase-04-validation-report.md`
9. Final quality validator reviews plan suite, changed code/docs/tests, all validation reports, and `artifacts/validation-summary.json`.

## Model Defaults

Use the session/main-thread selected defaults unless the user overrides:

- implementation worker: `gpt-5.3`, high reasoning;
- targeted repair worker: fresh `gpt-5.3`, high reasoning;
- second failure escalation repair: fresh `gpt-5.5`, high reasoning;
- phase validation/red-team: `gpt-5.5`, high reasoning;
- final quality validation: `gpt-5.5`, xhigh reasoning;
- browser proof: not applicable.

If a requested model/tool is unavailable, record `TOOLING_CONSTRAINT` and stop for main-thread/user approval before substituting.

## Validation Gates

- Do not start Phase 02 until Phase 01 audit validation passes or plan defects are remediated.
- Do not start Phase 03 until Phase 02 docs/feature-gate validation passes.
- Do not start Phase 04 until Phase 03 either implements and validates a minimal surface or records a validated deliberate defer.
- Do not run final quality review until Phase 04 evidence reconciliation is complete.

## Remediation Routing

- `plan_defect`: return to planning for revised criteria/directives.
- `code_defect`: fresh scoped repair worker unless trivial.
- `docs_or_evidence_defect`: fresh scoped repair worker unless the validator can make an obvious one-place correction.
- `browser_harness_defect`: not applicable.
- `validator_error`: correct checklist or use a fresh validator.

If the same targeted issue fails twice after repair, escalate to a fresh high-reasoning repair pass.

## Required Final Checks

At minimum, after the last code/doc change:

```sh
cargo check
cargo check -p renderer
cargo check -p renderer --examples
cargo check -p renderer --features advanced-interop
cargo check -p renderer --examples --features advanced-interop
```

Add focused tests and runtime/capture checks required by actual implementation.

## Capture Gate

Run headless draw capture only if Sprint 10 changes visible renderer behavior or capture/readback behavior:

```sh
RUST_LOG=info timeout --signal=INT 60s cargo run -p renderer --example api_test -- --headless --capture_target draw --capture_frames=3 --capture_frame_start=5 --capture_frame_interval=5 --capture_dir .internal-dev/captures/sprint-10-advanced-rendering-opt-in-contract/headless-draw
```

Desktop screenshots and present-target captures do not satisfy this gate.

## Closeout Gates

- Final evidence summary status is conservative and internally consistent.
- Known residuals are listed in `reports/final-closeout-notes.md` and `artifacts/validation-summary.json`.
- Changelog timing is confirmed with the user/main thread before creating changelog entries.
- Main thread owns `SPRINT-TRACKER.md` reconciliation after review.
- Do not claim `fully_validated` unless all required validation passed and no accepted residuals remain.
