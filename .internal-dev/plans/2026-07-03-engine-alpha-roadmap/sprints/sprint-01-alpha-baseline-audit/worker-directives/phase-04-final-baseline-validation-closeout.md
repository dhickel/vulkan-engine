# Phase 04 Worker Directive: Final Baseline Validation And Closeout

## Objective

Run final baseline checks, reconcile evidence, update tracker status appropriately, and prepare Sprint 01 for closeout without overclaiming.

## User-Visible Outcome

Sprint 01 ends with a current validation record, pushed commits, email reports, and a tracker state that accurately reflects whether the baseline is fully validated, passed with residuals, or blocked.

## Editable Targets

- `.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/SPRINT-TRACKER.md`
- `.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-01-alpha-baseline-audit/artifacts/validation-summary.json`
- `.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-01-alpha-baseline-audit/validation/phase-04-validation-report.md`
- `.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-01-alpha-baseline-audit/validation/final-quality-review.md`
- `.internal-dev/changelogs/<date>-alpha-baseline-audit.md` only after required user confirmation if the repo guidance requires asking first
- Optional debug/capture evidence under `.internal-dev/debug_reports/` or `.internal-dev/captures/`

## Read-Only Supporting Inputs

- All Sprint 01 phase reports and artifacts
- Root and package `Cargo.toml` files
- Current docs touched by phases 01-03
- Engine alpha sprint skill
- Headless capture validation skill if capture becomes required

## Forbidden Scope

- Do not repair product code failures.
- Do not make broad docs rewrites; route docs defects through scoped remediation.
- Do not mark tracker `closed` unless closeout/changelog requirements are satisfied.
- Do not claim `fully_validated` with unresolved residuals or skipped required checks.

## Senior-Engineer Guidance

- This is a reconciliation phase, not a feature phase.
- Compile failures in a baseline sprint may be valid evidence; they become residuals/blockers unless caused by Sprint 01 docs/process edits.
- Runtime smoke is optional unless docs claim runtime readiness. If run, use debug-record output under `.internal-dev/debug_reports/`.
- Capture proof is not required for docs/process-only changes. If a visual surface changed unexpectedly, stop and require the headless capture skill.
- Tracker status should usually move to `validating` during this phase and `closed` only after final quality pass, email, push, and changelog timing are resolved.

## Ordered Steps

1. Confirm phases 01-03 are validated, committed, pushed, and emailed.
2. Confirm branch and dirty state.
3. Run compile/test validation commands:
   ```bash
   cargo check
   cargo check -p renderer
   cargo check -p renderer --examples
   cargo check -p input
   cargo test -p input
   ```
4. Add package-level checks for live workspace crates if not covered by `cargo check`, especially:
   ```bash
   cargo check -p audio
   cargo check -p physics
   cargo check -p scripting
   cargo check -p editor
   cargo check -p dungeon_dogfood
   ```
5. Decide whether runtime debug smoke is needed. If yes, use:
   ```bash
   RUST_LOG=debug timeout --signal=INT 60s cargo run -p renderer --example demo_pbr -- --record_debug=10 --record_debug_interval=50 --record_debug_path=.internal-dev/debug_reports/demo_pbr-sprint-01-timing.jsonl
   ```
6. Decide whether capture is required. If yes, follow `.internal-dev/skills/engine-headless-capture-validation/SKILL.md`.
7. Run stale-reference sweep from `validation/README.md`.
8. Reconcile `artifacts/validation-summary.json` with all reports, command results, pushes, and emails.
9. Update sprint tracker to `validating`, `closed`, or `blocked` according to actual evidence.
10. If changelog creation requires user confirmation, stop and ask before writing it. If already authorized by orchestrator/user, write the changelog.
11. Write `validation/phase-04-validation-report.md`.
12. Run final quality validator and write `validation/final-quality-review.md`.
13. Commit, push, and send Dwight the final post-phase HTML AgentMail report.

## Acceptance Criteria

- Required cargo commands pass or failures are recorded as baseline blockers/residuals.
- Stale-reference sweep is documented.
- Validation summary status matches the true final state.
- Tracker status is updated accurately.
- Final quality review reconciles all phase evidence.
- Final phase commit is pushed and final email sent.

## Negative Checks

- No product code changes.
- No hidden skipped validation.
- No `fully_validated` if residuals remain.
- No tracker `closed` with missing changelog if changelog is required for closeout.
- No unresolved stale gap-report current-truth references.

## Validation Commands

Use the commands in ordered steps. Record exact command, exit status, and summary in the phase report. Store any debug-record output under `.internal-dev/debug_reports/`.

## Stop Conditions

- Required cargo command cannot run and the blocker cannot be recorded clearly.
- Capture becomes required but cannot produce inspectable output.
- Validation summary contradicts phase reports.
- Tracker closeout requires user confirmation for changelog timing.
- Push or AgentMail send fails.

## Evidence Expectations

- Validation report: `validation/phase-04-validation-report.md`
- Final quality review: `validation/final-quality-review.md`
- Evidence index: `artifacts/validation-summary.json`
- Debug/capture paths if generated.
- Commit hash, pushed branch/ref, GitHub links, email evidence.

## Do Not Close Unless

- Final quality validator passes or records accepted residuals.
- Tracker status matches actual state.
- Commit is pushed.
- HTML email report is sent.
- Changelog requirement is satisfied or explicitly blocked for user confirmation.
