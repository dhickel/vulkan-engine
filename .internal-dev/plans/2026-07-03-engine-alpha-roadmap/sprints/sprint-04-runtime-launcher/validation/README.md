# Sprint 04 Validation Guide

## Purpose

Validators must verify that the root `engine` binary is a real alpha project launcher, not just a compile-passing stub, and that visual proof uses true headless draw-target capture.

## Reports

Each phase validator writes:

- `validation/phase-01-validation-report.md`
- `validation/phase-02-validation-report.md`
- `validation/phase-03-validation-report.md`
- `validation/phase-04-validation-report.md`

Each report must include:

- criteria pass/fail;
- commands run and exit status;
- evidence inspected;
- capture proof status when applicable;
- findings ordered by severity;
- remediation handoff if failed;
- residual risks;
- whether `artifacts/validation-summary.json` is consistent.

## Validator Scope

Validators are non-mutating by default. They may only self-edit simple validator/report defects if the issue is a typo, stale report reference, missing link, or small obvious comparison mistake. They must not self-edit production code, schemas, runtime config, renderer internals, broad docs, security/concurrency behavior, or multi-file logic.

## Required Capture Standard

Visual proof must use:

```bash
--headless --capture_target draw
```

Evidence must include:

- root launcher command;
- capture directory under `.internal-dev/captures/sprint-04-runtime-launcher/`;
- PNG path(s);
- sidecar JSON path(s);
- sidecar predicates:
  - `status == "succeeded"`;
  - `capture_target == "draw"`;
  - draw-target format such as `R16G16B16A16_SFLOAT`;
  - positive extent;
  - existing PNG path.

Desktop screenshots and present-target captures are invalid for Sprint 04 proof.

## Validation Order

1. Validate Phase 01 before Phase 02 begins.
2. Validate Phase 02 before Phase 03 docs are considered final.
3. Validate Phase 03 before Phase 04 closeout.
4. Phase 04 performs final compile/test/runtime/capture/doc/evidence reconciliation.

## Remediation Routing

- `code_defect`: fresh scoped repair worker with the selected worker model unless the issue is trivial and mechanical.
- `docs_or_evidence_defect`: fresh scoped repair worker unless it is a simple validator/report typo.
- `browser_harness_defect`: not applicable; this is not browser UI work.
- `capture_harness_defect`: repair the capture command/evidence parser first; change product code only after evidence proves a product bug.
- `plan_defect`: return to planning before more coding.
- `validator_error`: correct checklist or use a fresh validator.

If the same targeted issue fails validation twice after repairs, stop and escalate to a fresh high-reasoning repair worker.

## Final Reconciliation

Before `fully_validated` can be recorded:

- all phase reports exist and pass;
- required compile/test commands pass or accepted residuals are explicitly documented;
- root launcher sample project run/capture passes;
- headless draw sidecars pass predicates;
- docs no longer contain stale Sprint 04 claims;
- `artifacts/validation-summary.json` is parseable and internally consistent;
- unresolved critical residuals are fixed or tracked.
