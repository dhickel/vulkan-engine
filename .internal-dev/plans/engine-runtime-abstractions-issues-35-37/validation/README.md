# Validation Plan

Date: 2026-07-07
Status: validation routing

## Validator Model Defaults

- Phase validation/red-team agent: `gpt-5.5`, high reasoning.
- Final large-suite quality validator: `gpt-5.5`, xhigh reasoning after phase validations pass.
- Repair worker default: fresh scoped `gpt-5.5`, high reasoning unless main-thread dispatch policy overrides.
- Second failure on the same targeted issue: fresh scoped `gpt-5.5`, high reasoning.

If the required model/tool is unavailable, record `TOOLING_CONSTRAINT` and stop for main-thread/user approval before fallback.

## Reports

Required phase reports:

- `phase-00-validation-report.md`
- `phase-01-validation-report.md`
- `phase-02-validation-report.md`
- `phase-03-validation-report.md`
- `phase-04-validation-report.md`
- `phase-05-validation-report.md`
- `phase-06-validation-report.md`
- `final-quality-review.md`

Reports live in:

`.internal-dev/plans/engine-runtime-abstractions-issues-35-37/validation/`

## Validator Responsibilities

Each phase validator checks:

- plan criteria and phase directive compliance;
- crate graph and architecture fit;
- raw primitive access;
- Phase 01 facade import proof from outside the defining module, including `engine::prelude::*`, `engine::input`, `engine::events`, `engine::camera`, and `engine::render`;
- Phase 01 direct raw-crate import proof for the original support crates;
- legacy compatibility boundaries;
- behavioral tests and command evidence;
- docs/spec drift caused by the phase;
- `.internal-dev` closeout expectations for that phase;
- evidence path correctness.

## Browser/UI Validation

No Playwright/browser validation applies. This is a Rust desktop/runtime refactor. If a future phase adds a browser/editor UI surface, return to planning for a browser checklist.

## Runtime And Visual Validation

Runtime smoke is required after compile gates are clean. Headless capture validation is conditional and required only when camera output proof is claimed or visible camera/render behavior materially changes.

## Final Quality Review

Final quality validator must compare:

- this plan suite;
- phase directives;
- phase validation reports;
- implementation changes;
- tests;
- docs/spec/knowledge/changelog closeout;
- `artifacts/engine-runtime-abstractions-issues-35-37/validation-summary.json`;
- stale-reference sweep results.

Final status cannot pass if:

- any required phase report is missing or failed;
- evidence index status contradicts reports;
- stale specs still state renderer-owned camera/event/input as the only intended contract;
- dogfood active path still uses renderer-owned camera/event APIs;
- raw primitive support is obscured;
- required runtime smoke is skipped without an approved environmental/tooling constraint.
