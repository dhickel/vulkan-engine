# Advanced-Planner Handoff Seed: Engine Alpha Roadmap

Date: 2026-07-03

This is not an execution plan. Use it as the seed when a specific sprint is selected for advanced planning.

## Objective

Create a phased, validation-gated plan for the selected sprint from `.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/SPRINT-TRACKER.md`.

## Planning Inputs

- Roadmap: `.internal-dev/plans/2026-07-03-engine-alpha-roadmap/README.md`
- Sprint tracker: `.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/SPRINT-TRACKER.md`
- Sprint template: `.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/SPRINT-TEMPLATE.md`
- Repo guide: `AGENTS.md`
- Renderer guide: `src/renderer/AGENTS.md`
- Input guide: `src/input/AGENTS.md`
- Headless capture skill: `.internal-dev/skills/engine-headless-capture-validation/SKILL.md`
- Sprint skill: `.internal-dev/skills/engine-alpha-sprint/SKILL.md`

## Required Planner Behavior

- Read the selected sprint brief first.
- Verify live source before trusting stale docs.
- Keep product code changes out of planning unless the user explicitly asks to execute.
- Produce phase files under the selected sprint directory.
- Include validation criteria, evidence paths, stop conditions, and residual tracking.
- Preserve dirty worktree changes that are not part of the task.

## Standard Validation Menu

Choose only the checks relevant to the sprint, but justify omissions:

- `cargo check`
- `cargo check -p renderer`
- `cargo check -p renderer --examples`
- `cargo check -p input`
- `cargo test -p input`
- targeted package/scene/editor/runtime tests
- debug-record smoke under `.internal-dev/debug_reports/`
- headless capture evidence under `.internal-dev/captures/`

## Standard Phase Gates

1. Current-state audit.
2. Contract/spec lock.
3. Minimal implementation or remediation.
4. Focused tests.
5. Runtime/capture validation when applicable.
6. Docs and `.internal-dev` closeout.
7. Final conservative review.

## Conservative Closeout Language

Use `fully_validated` only when all required validation ran and no accepted residuals remain.

Use `final_quality_pending` when compile/tests pass but runtime/capture/docs evidence is incomplete.

Use `final_quality_review_passed_with_residuals` when the sprint is acceptable but known tracked issues remain.
