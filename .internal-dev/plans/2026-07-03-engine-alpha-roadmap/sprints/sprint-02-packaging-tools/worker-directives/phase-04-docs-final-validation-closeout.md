# Phase 04 Worker Directive: Docs, Final Validation, And Closeout Evidence

## Objective

Document the CLI path, run final validation, reconcile capture decision, and produce conservative closeout evidence.

## User-Visible Outcome

The sprint has docs and evidence showing how to use the packaging CLI and exactly what validation passed, failed, or remains residual.

## Editable Targets

- `docs/api/04-assets-sync-deferred-and-handles.md`
- `docs/api/03-scene-graph-and-fragment-workflows.md`
- optional CLI usage doc under `docs/api/` if useful
- `.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/SPRINT-TRACKER.md`
- this sprint's `artifacts/validation-summary.json`
- this sprint's validation reports
- no product code unless a tiny docs-test fix is required and validated

## Forbidden Scope

- Do not create Sprint 01 changelog or close Sprint 01.
- Do not implement new CLI behavior except trivial documentation-driven fixes validated as remediation.
- Do not claim visual validation unless capture evidence exists.
- Do not stage `.idea/engine.iml` or `.reasonix/`.

## Supporting Docs To Read

- All Sprint 02 plan files.
- All previous phase validation reports.
- `email-report-template.html`
- `.internal-dev/skills/engine-headless-capture-validation/SKILL.md` only if capture became required.

## Senior Guidance

- Docs should teach the supported CLI path without overstating alpha readiness.
- Evidence must be internally consistent. If a command failed but was accepted as residual, the top-level status cannot be `fully_validated`.
- Perform a stale-reference sweep before final validation.
- Keep Sprint 01 dependency visible as residual/dependency, not as closed work.

## Implementation Steps

1. Update API docs with CLI usage, validation scope, durable identity rules, and pack output shape.
2. Run the final command set from `shared/validation-matrix.md`.
3. Run CLI smokes against the sample editor project.
4. If any visual claim was introduced, run headless capture validation and record evidence. Otherwise record `not_required_cli_schema_only`.
5. Update `artifacts/validation-summary.json` with phases, commands, commits, pushes, AgentMail reports, residuals, and final conservative status.
6. Update the sprint tracker to the correct post-validation status. Use `validating` or `closed` only when gates justify it.
7. Write `validation/phase-04-validation-report.md`.
8. Prepare final quality review handoff.

## Acceptance Criteria

- Docs describe `engine_pack` commands and durable identity constraints.
- Final command set ran or blockers are recorded.
- Evidence summary is consistent with validation reports.
- Tracker reflects actual status.
- Final validation is ready for a validator without replanning.

## Negative Checks

- No stale `fully_validated` claim.
- No docs that tell users Python is canonical.
- No docs that say binary archives or thumbnails exist.
- No unresolved visual claim without capture proof.

## Validation Commands

```bash
cargo check
cargo check -p renderer
cargo check -p renderer --examples
cargo check -p input
cargo check -p engine_pack
cargo test -p engine_pack
cargo run -p engine_pack -- validate-package apps/editor/sample_project/assets/editor_sample.package.toml --expected-package-id editor_sample
cargo run -p engine_pack -- validate-project apps/editor/sample_project/engine.project.toml
cargo run -p engine_pack -- validate-scene apps/editor/sample_project/scenes/start.engine.scene.json --project apps/editor/sample_project/engine.project.toml
```

## Evidence Expectations

- Validation report path: `validation/phase-04-validation-report.md`
- Final quality review path: `validation/final-quality-review.md`
- Canonical summary path: `artifacts/validation-summary.json`
- Include files/line counts/git links matrix.
- Include capture decision or capture evidence.

## Commit/Push/AgentMail Gate

After phase validation passes, orchestrator must commit scoped changes, push `sprint/alpha-02-packaging-tools`, and send an AgentMail HTML progress report. After final quality review, send final AgentMail HTML report.

## Stop Conditions

- Stop if required final commands cannot run and no acceptable blocker/residual is documented.
- Stop if evidence summary contradicts validation reports.
- Stop if capture is required but cannot be produced.

## Do Not Close Unless

- All phase reports exist.
- Final quality review report exists.
- Evidence summary is conservative and consistent.
- Commit hash, pushed ref, GitHub links, and AgentMail evidence are recorded.
