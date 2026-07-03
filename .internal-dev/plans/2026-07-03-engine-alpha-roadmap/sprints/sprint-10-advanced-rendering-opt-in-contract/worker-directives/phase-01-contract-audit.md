# Phase 01 Worker Directive: Contract Audit And Sprint Spec Confirmation

## Objective

Audit live source, docs, examples, and Sprint 09 boundary assumptions so Sprint 10 implementation can proceed without inventing the advanced rendering contract mid-edit.

## User-Visible Outcome

The sprint has a current, source-backed advanced rendering API audit that identifies safe default extension points, feature-gated unstable surfaces, docs drift, and deferred advanced rendering work.

## Editable Targets

- `.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-10-advanced-rendering-opt-in-contract/reports/phase-01-advanced-api-audit.md`
- `.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-10-advanced-rendering-opt-in-contract/artifacts/validation-summary.json` for phase status only.

## Read-Only Targets

- `src/renderer/src/api/advanced.rs`
- `src/renderer/src/api/mod.rs`
- `src/renderer/src/api/hooks.rs`
- `src/renderer/src/api/renderer.rs`
- `src/renderer/src/lib.rs`
- `src/renderer/src/rendergraph/mod.rs`
- `src/renderer/examples/`
- `docs/api/00-index.md`
- `docs/api/05-render-hooks-and-extension-points.md`
- `docs/api/05-hooks.md`
- `docs/api/08-debug.md`
- `docs/internal/07-rendergraph-dependencies-and-aliasing.md`
- Sprint 09 target design, read-only only if needed.

## Forbidden Scope

- Do not edit product code, docs outside this sprint directory, tests, schemas, `SPRINT-TRACKER.md`, `.idea/engine.iml`, or `.reasonix/`.
- Do not resolve Sprint 09 dirty files.

## Supporting Docs To Read

- `AGENTS.md`
- `.internal-dev/AGENTS.md`
- `src/renderer/AGENTS.md`
- This sprint `00-specification-lock.md`, `01-current-state-analysis.md`, and `02-target-design.md`.

## Senior Engineer Guidance

- Verify code before trusting docs because docs drift is already suspected.
- Classify public exposure by default vs `advanced-interop`, not just by whether a symbol is `pub`.
- Look for examples importing advanced/rendergraph APIs; beginner examples must not need them.
- Treat existing `renderer_core_mut` as an unsafe escape hatch, not a design endorsement.
- Rendergraph pass registration should be considered deferred unless resource/order contracts are explicit.

## Ordered Steps

1. Check current git status and note any Sprint 09/protected dirty files in the audit report without modifying them.
2. Inspect advanced exports, rendergraph visibility, hook context, debug view APIs, and capture/readback APIs.
3. Inspect renderer examples for advanced/rendergraph imports or feature requirements.
4. Compare both hook docs against live code and record mismatches.
5. Produce a table classifying current surfaces: beginner, safe extension, advanced interop, raw backend escape hatch, internal/deferred.
6. Update `artifacts/validation-summary.json` phase 01 entry to `audit_ready_for_validation`.

## Acceptance Criteria

- Audit report exists and cites exact files/lines where practical.
- Audit lists docs drift and feature-gate risks.
- Audit identifies whether Phase 03 should implement a minimal named surface or defer advanced custom passes.
- No product code/docs/tests outside the sprint directory are changed.

## Negative Checks

- No advanced API is proposed as beginner facade by assumption.
- No raw backend handle exposure is recommended without a stop/user decision.
- No capture requirement is added for audit-only work.

## Validation Commands

No compile commands are required for this audit-only phase unless the worker chooses to verify current state. If commands are run, record them in the phase report.

## Stop Conditions

- Stop if Sprint 09 state is too unresolved to identify the current facade boundary.
- Stop if required files are missing or contradictory enough that implementation criteria cannot be locked.

## Evidence Expectations

- Report: `reports/phase-01-advanced-api-audit.md`
- Validation report: `validation/phase-01-validation-report.md`
- Summary update: `artifacts/validation-summary.json`

## Do Not Close Unless

- The audit report exists.
- The validator can tell exactly what Phase 02 and Phase 03 should use as source truth.
- Protected paths remain untouched.
