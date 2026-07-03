# Validation Plan

## Report Paths

Each phase validator must write:

- `.internal-dev/plans/2026-07-03-frame-capture-plan/validation/phase-01-validation-report.md`
- `.internal-dev/plans/2026-07-03-frame-capture-plan/validation/phase-02-validation-report.md`
- `.internal-dev/plans/2026-07-03-frame-capture-plan/validation/phase-03-validation-report.md`
- `.internal-dev/plans/2026-07-03-frame-capture-plan/validation/phase-04-validation-report.md`
- `.internal-dev/plans/2026-07-03-frame-capture-plan/validation/phase-05-validation-report.md`

Final quality validator must write:

- `.internal-dev/plans/2026-07-03-frame-capture-plan/validation/final-quality-review.md`

Canonical evidence index:

- `.internal-dev/plans/2026-07-03-frame-capture-plan/artifacts/validation-summary.json`

## Validator Responsibilities

Validators must check:

- implementation matches this plan and the locked preplanning handoff;
- public API remains safe and does not expose raw Vulkan handles;
- image layout, synchronization, allocation cleanup, and error handling are coherent;
- parser behavior preserves existing flags;
- tests are meaningful and not only snapshots of implementation details;
- capture artifacts are real PNGs with nonzero dimensions and nonuniform pixels;
- documentation reflects actual final flags and behavior;
- evidence index status is conservative and internally consistent.

## Browser Validation

Browser/Playwright validation does not apply. Any desktop screenshot is supplementary only and cannot satisfy required proof.

## Stale-Reference Sweep

Before final quality review, sweep docs and `.internal-dev/plans/2026-07-03-frame-capture-plan/` for:

- stale flag names;
- stale `/tmp` artifact paths;
- pending/planned/not implemented wording that no longer matches reality;
- TODO markers introduced by the implementation;
- references to compositor screenshot proof as authoritative;
- evidence paths outside `.internal-dev/debug_reports/` unless explicitly justified.

## Final Status Rules

The evidence index top-level status must not be `fully_validated` unless:

- every compile gate passed;
- every required runtime capture matrix row passed;
- N-frame validation passed;
- headless validation passed or an approved user gate changed scope;
- manual capture validation passed or automation blocker is recorded with direct API proof;
- every phase validator passed;
- final quality review passed.

