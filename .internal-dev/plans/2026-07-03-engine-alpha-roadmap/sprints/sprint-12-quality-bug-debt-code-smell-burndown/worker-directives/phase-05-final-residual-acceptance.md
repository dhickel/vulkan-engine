# Phase 05 Worker Directive: Final Validation Matrix And Residual Acceptance

## Objective

Reconcile all Sprint 12 evidence, residuals, validation commands, bug artifacts, and closeout status into a conservative final package ready for final quality review.

## User-Visible Outcome

The sprint closes only if the quality story is honest: critical bugs are fixed or explicitly accepted with mitigation, evidence is complete, and the summary does not overclaim.

## Editable Targets

- `reports/phase-05-final-residual-acceptance.md`
- `artifacts/validation-summary.json`
- `validation/README.md` only if validation instructions need correction.
- `.internal-dev/bugs/<bug-id>/report.md` only to update status/evidence for bugs handled in Sprint 12.
- Optional docs touch-ups only for stale references found during final sweep, with no product behavior changes.

## Forbidden Scope

- Do not implement product code except trivial docs/evidence corrections.
- Do not update `SPRINT-TRACKER.md`.
- Do not create changelog entries unless the main thread/user confirms timing.
- Do not edit `.idea/engine.iml` or `.reasonix/`.
- Do not claim `fully_validated` while accepted residuals remain.

## Supporting Docs To Read

- All Sprint 12 plan files.
- All phase worker reports.
- All phase validation reports.
- `artifacts/validation-summary.json`.
- Bug artifacts created/updated during this sprint.
- Relevant changed docs/tests/source diff.

## Senior Engineer Guidance

- Direct target: evidence integrity and residual honesty.
- Approach: compare plan criteria to actual evidence, not just worker summaries.
- Gotcha: `final_quality_review_passed_with_residuals` is acceptable when residuals are explicit and mitigated; `fully_validated` is not.
- Gotcha: stale scans will find this plan's intentional words. Distinguish instructional text from stale completion claims.
- Best practice: make the residual ledger actionable enough for Sprint 13 release-candidate planning.
- Likely failure mode: closing with missing runtime smoke or validator reports.

## Implementation Steps

1. Verify every phase has a worker report and validator report.
2. Verify every command required by phase scope is present in evidence, or a blocker/omission rationale is recorded.
3. Verify runtime debug reports and captures exist for touched behavior that required them.
4. Run final stale-reference sweep:

```sh
rg -n "pending|planned|not implemented|/tmp|desktop screenshot|TOOLING_CONSTRAINT|fully_validated|final_quality_review_passed" docs .internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-12-quality-bug-debt-code-smell-burndown
rg -n "gap-report|old image views|destroy paths|VkSubAllocator::destroy|fence\\[0\\]|double free" docs/api docs/internal .internal-dev/bugs .internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-12-quality-bug-debt-code-smell-burndown
```

5. Run or verify the full validation set from `shared/validation-matrix.md`.
6. Build a final residual ledger in `reports/phase-05-final-residual-acceptance.md`.
7. Update `artifacts/validation-summary.json` with phase statuses, command results, evidence paths, residual risks, and final quality review pending status.
8. Hand off to final quality validator.

## Acceptance Criteria

- Evidence index matches all reports and does not overclaim.
- Every residual has a class, acceptance state, mitigation, and follow-up path.
- Critical residuals are fixed or explicitly user-accepted; otherwise status is blocked.
- Required runtime/capture evidence is present or a `TOOLING_CONSTRAINT`/blocker is recorded.
- Final quality review can run without reconstructing the sprint from scratch.

## Negative Checks

- No missing phase validation report.
- No unqualified `fully_validated`.
- No stale `/tmp` evidence paths.
- No unresolved critical defect marked as accepted by default.
- No protected path edits.

## Validation Commands

Full validation set:

```sh
cargo fmt --check
cargo check
cargo check -p renderer
cargo check -p renderer --examples
cargo check -p input
cargo test -p input
cargo test -p renderer
```

Run runtime smoke set from `shared/validation-matrix.md` when Phases 02-03 touched renderer runtime behavior.

Run capture validation only when visible renderer/editor behavior changed.

## Stop Conditions

- Stop if any phase validator is missing or failed.
- Stop if evidence index contradicts reports.
- Stop if a critical residual lacks explicit acceptance or mitigation.
- Stop if final stale sweep reveals stale status claims that need remediation beyond evidence edits.

## Evidence Expectations

- Worker report: `reports/phase-05-final-residual-acceptance.md`
- Validator report: `validation/phase-05-validation-report.md`
- Final quality review: `validation/final-quality-review.md`
- Evidence index complete and conservative.

## Do Not Close Unless

- Final quality validator has a complete handoff.
- The top-level status is conservative.
- Residuals/blockers are visible.
- Changelog timing remains a main-thread/user gate.
