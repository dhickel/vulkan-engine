# Work Units

## Phase 01: Current Dogfood And Contract Audit

Directive: `worker-directives/phase-01-current-dogfood-project-audit.md`

Purpose: lock live state, active Sprint 09/Sprint 10 interactions, dogfood content contract gaps, and exact implementation path before mutations.

Validation report: `validation/phase-01-validation-report.md`

## Phase 02: Packaged Content And Sample Scene

Directive: `worker-directives/phase-02-packaged-content-sample-scene.md`

Purpose: create or normalize dogfood project/package/scene data and validation fixtures using canonical contracts.

Validation report: `validation/phase-02-validation-report.md`

## Phase 03: Runtime Gameplay Loop, Input, And Camera

Directive: `worker-directives/phase-03-runtime-gameplay-input-camera.md`

Purpose: wire dogfood runtime to the contract path while preserving custom Rust exploration gameplay.

Validation report: `validation/phase-03-validation-report.md`

## Phase 04: True Headless Visual Baseline

Directive: `worker-directives/phase-04-true-headless-visual-baseline.md`

Purpose: produce engine-owned draw-target capture proof for the dogfood vertical slice.

Validation report: `validation/phase-04-validation-report.md`

## Phase 05: Docs, Evidence, And Final Validation Prep

Directive: `worker-directives/phase-05-docs-final-validation-prep.md`

Purpose: align docs, evidence index, reports, residuals, and final review inputs.

Validation report: `validation/phase-05-validation-report.md`

## Dependency Order

Run phases sequentially. Phase 02 depends on Phase 01 contract findings. Phase 03 depends on Phase 02 data shape. Phase 04 depends on Phase 03 launch/capture support. Phase 05 depends on all implementation and validation evidence.
