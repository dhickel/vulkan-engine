# Phase 03 Worker Directive: Residual Register And Validation Matrix

## Objective

Convert the baseline audit into durable residual tracking and reusable alpha validation rules for future sprints.

## User-Visible Outcome

Known alpha gaps, stale historical claims, and validation expectations are organized into current, reviewable artifacts instead of scattered docs.

## Editable Targets

- `.internal-dev/reviews/2026-07-03-alpha-baseline-register.md` or equivalent focused register under `.internal-dev/bugs/`
- `.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-01-alpha-baseline-audit/shared/validation-matrix.md`
- `.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-01-alpha-baseline-audit/artifacts/validation-summary.json`
- `.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-01-alpha-baseline-audit/validation/phase-03-validation-report.md`
- Optional docs-facing pointer if phase 02 left one explicitly for this phase

## Read-Only Supporting Inputs

- Phase 01 baseline inventory
- Phase 02 docs drift audit
- Current docs modified in phase 02
- Root and package `Cargo.toml` files
- Alpha roadmap and deep tracks

## Forbidden Scope

- Do not fix product defects.
- Do not expand future sprint scope.
- Do not create one bug file per historical claim unless the claim is verified current and needs standalone tracking.
- Do not log future considerations under `.internal-dev/notes/` without user confirmation.

## Senior-Engineer Guidance

- A register is useful only if it separates current verified issues from stale history.
- Use the classification contract from `02-target-design.md`: `verified_current`, `stale_resolved`, `unknown_needs_audit`, `accepted_alpha_debt`, `blocked_validation`.
- Assign likely future sprint ownership when obvious from the roadmap, but avoid overcommitting exact fixes.
- Keep validation matrix action-oriented: which command/evidence is required for which kind of sprint.
- Update the existing plan validation matrix if phase 03 learns a stricter rule; do not fork competing matrices.

## Ordered Steps

1. Confirm phase 02 is validated, committed, pushed, and emailed.
2. Review phase 01 and phase 02 artifacts for unresolved drift and stale claims.
3. Create or update the consolidated alpha baseline register.
4. Ensure each register item has status, evidence, impact, next action, and likely sprint/track when known.
5. Refine `shared/validation-matrix.md` if needed so it covers future alpha sprint gate categories.
6. Update `artifacts/validation-summary.json`.
7. Run validation commands.
8. Write `validation/phase-03-validation-report.md`.
9. Commit, push, and send Dwight the post-phase HTML AgentMail report.

## Acceptance Criteria

- Consolidated register exists under `.internal-dev/reviews/` or `.internal-dev/bugs/`.
- Register distinguishes stale historical claims from verified current issues.
- Validation matrix is coherent and points to exact evidence locations.
- No out-of-scope bugs are silently discarded; if not repaired, they are registered.

## Negative Checks

- No product code changes.
- No unverified old gap-report claims listed as current.
- No future sprint claims marked complete.
- No notes/future-consideration artifact created without user approval.

## Validation Commands

```bash
git status --short --branch
test -f .internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-01-alpha-baseline-audit/shared/validation-matrix.md
test -f .internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-01-alpha-baseline-audit/artifacts/validation-summary.json
rg -n "verified_current|stale_resolved|unknown_needs_audit|accepted_alpha_debt|blocked_validation" .internal-dev/reviews .internal-dev/bugs .internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-01-alpha-baseline-audit
```

Run targeted Markdown link/path checks if local tooling exists; otherwise manually inspect modified Markdown links and record method.

## Stop Conditions

- A residual requires a user decision before it can be classified.
- A product defect blocks docs/process baseline but is outside Sprint 01 scope.
- User confirmation is needed for `.internal-dev/notes/`.
- Push or AgentMail send fails.

## Evidence Expectations

- Validation report: `validation/phase-03-validation-report.md`
- Register path and line count.
- Validation matrix path and line count.
- Evidence index update.
- Commit hash, pushed branch/ref, GitHub links, email evidence.

## Do Not Close Unless

- Phase validator passes.
- Commit is pushed.
- HTML email report is sent.
- Register and validation matrix are internally consistent.
