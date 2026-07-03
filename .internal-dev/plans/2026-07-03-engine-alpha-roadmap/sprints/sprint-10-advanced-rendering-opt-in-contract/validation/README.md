# Validation Plan

## Validator Responsibilities

Validators must check the plan criteria, application contract fit, architecture fit, regression risk, docs drift, test quality, and evidence consistency. They must not mark the sprint fully validated unless every required phase report and command/capture result is reconciled.

## Phase Report Paths

- Phase 01: `validation/phase-01-validation-report.md`
- Phase 02: `validation/phase-02-validation-report.md`
- Phase 03: `validation/phase-03-validation-report.md`
- Phase 04: `validation/phase-04-validation-report.md`

## Required Checks By Phase

- Phase 01: audit completeness, no product mutations, clear feature/default boundary findings.
- Phase 02: default and feature-gated compile checks, docs consistency, no advanced API in beginner examples/prelude.
- Phase 03: focused tests for any new advanced surface, runtime/capture checks if behavior changed, explicit defer if unsafe.
- Phase 04: full matrix reconciliation, stale-reference sweep, conservative status in `artifacts/validation-summary.json`.

## Stale-Reference Sweep

Before final validation, search changed docs and this sprint directory for:

- stale artifact paths;
- `/tmp` evidence paths presented as durable;
- unresolved `TODO`;
- `pending`, `planned`, or `not implemented` claims that are no longer true;
- claims that desktop screenshots or present-target captures are acceptable renderer proof;
- claims that `advanced-interop` is stable or default.

## Browser Proof

Not applicable. This sprint has no web UI surface. Use engine headless capture only when renderer/capture behavior requires visual proof.
