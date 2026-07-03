# Sprint 09 Facade API Contract Plan

Status: planning_locked
Branch: sprint/alpha-09-facade-api-contract
Created: 2026-07-03

## Objective

Lock the renderer beginner facade contract before community alpha. The sprint should make the supported alpha API discoverable, keep the beginner path small, make examples compile against the same APIs users are expected to use, and classify legacy or advanced public exports without abruptly breaking existing consumers.

## Work Classification

Large. The work crosses public Rust exports, renderer examples, API docs, package/template validation, and final evidence governance. The implementation should stay conservative, but the plan needs multiple phased workers and validators because each phase changes a different contract surface.

## Required Files

- `00-specification-lock.md`
- `01-current-state-analysis.md`
- `02-target-design.md`
- `shared/senior-engineer-guidance.md`
- `shared/implementation-notes.md`
- `shared/validation-matrix.md`
- `work-units/README.md`
- `worker-directives/phase-01-facade-surface-audit.md`
- `worker-directives/phase-02-alpha-prelude-and-example-contract.md`
- `worker-directives/phase-03-error-input-camera-material-docs-hardening.md`
- `worker-directives/phase-04-docs-final-validation.md`
- `validation/README.md`
- `reports/README.md`
- `artifacts/validation-summary.json`
- `final-orchestration-plan.md`

## Phase Order

1. Phase 01: audit and classify public facade/root exports.
2. Phase 02: define alpha prelude/export contract and align compile-checked examples.
3. Phase 03: harden targeted beginner friction areas in docs/tests and add only small wrappers when justified.
4. Phase 04: reconcile docs, evidence, stale references, and final validation.

## Evidence Policy

Use `artifacts/validation-summary.json` as the canonical evidence index. It must remain conservative until all phase validators and final quality review pass. Desktop screenshots do not count. Use true engine-owned headless capture with `--headless --capture_target draw` only if a phase changes visible renderer/editor behavior.
