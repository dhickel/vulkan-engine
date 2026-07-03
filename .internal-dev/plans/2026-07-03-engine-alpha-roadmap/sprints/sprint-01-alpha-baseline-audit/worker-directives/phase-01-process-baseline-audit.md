# Phase 01 Worker Directive: Process Baseline Audit

## Objective

Restore the active `.internal-dev` process guide and produce a verified baseline inventory that later Sprint 01 phases can trust.

## User-Visible Outcome

Future agents can find the active internal-dev operating guide and a current baseline inventory without reading archived process files or stale gap reports.

## Editable Targets

- `.internal-dev/AGENTS.md`
- `.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-01-alpha-baseline-audit/artifacts/baseline-inventory.md`
- `.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-01-alpha-baseline-audit/artifacts/validation-summary.json`
- `.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-01-alpha-baseline-audit/validation/phase-01-validation-report.md`

## Read-Only Supporting Inputs

- `AGENTS.md`
- `.internal-dev/.archive/AGENTS.md`
- `Cargo.toml`
- package `Cargo.toml` files
- `.internal-dev/skills/engine-alpha-sprint/SKILL.md`
- `.internal-dev/skills/engine-headless-capture-validation/SKILL.md`

## Forbidden Scope

- Do not edit product code.
- Do not edit public docs beyond `.internal-dev/AGENTS.md` in this phase.
- Do not alter `.idea/engine.iml` or `.reasonix/`.
- Do not scan unrelated `.internal-dev` directories.

## Senior-Engineer Guidance

- Restore active process guidance from `.internal-dev/.archive/AGENTS.md` because root `AGENTS.md` references an active file that is currently missing.
- Keep the restored guide concise and current; do not add speculative sprint details there.
- Baseline inventory should be evidence, not recommendations: branch, HEAD, dirty state, remotes, workspace members, docs/process drift, and validation capabilities.
- Record capture as readiness-only unless a visual surface changes later.
- Treat memory and archived files as historical; verify live repo state for current claims.

## Ordered Steps

1. Confirm branch and dirty state with `git status --short --branch`.
2. Confirm remote URL and current HEAD.
3. Restore or recreate `.internal-dev/AGENTS.md` from `.internal-dev/.archive/AGENTS.md`, updating wording only where necessary for current repo policy.
4. Write `artifacts/baseline-inventory.md` with:
   - branch, HEAD, remote;
   - preserved unrelated dirty files;
   - root workspace members from `Cargo.toml`;
   - available package manifests;
   - active/archived `.internal-dev` guide status;
   - docs likely needing repair in later phases;
   - capture skill readiness status.
5. Initialize or update `artifacts/validation-summary.json` with phase 01 status.
6. Run validation commands listed below.
7. Write `validation/phase-01-validation-report.md`.
8. Commit only phase 01 files.
9. Push `sprint/alpha-01-baseline-audit`.
10. Send Dwight the post-phase HTML AgentMail report using `email-report-template.html`.

## Acceptance Criteria

- `.internal-dev/AGENTS.md` exists and covers purpose, source-of-truth, access discipline, directory contract, workflow rules, templates, and related guides.
- Baseline inventory names all live workspace members and preserved unrelated dirty files.
- Validation summary exists with conservative phase 01 status.
- Phase validation report records commands, evidence, commit hash, pushed ref, and email evidence.

## Negative Checks

- No product code changed.
- No unrelated dirty files staged.
- No claims about stale gap-report defects as current without verification.
- No capture validation claimed.

## Validation Commands

```bash
git status --short --branch
git diff -- .internal-dev/AGENTS.md .internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-01-alpha-baseline-audit
test -f .internal-dev/AGENTS.md
rg -n "Source-of-Truth|Access Discipline|Directory Contract|Workflow Rules" .internal-dev/AGENTS.md
```

Compile checks are not required in Phase 01 unless the worker touched files outside the allowed targets.

## Stop Conditions

- Current branch is not `sprint/alpha-01-baseline-audit`.
- Unexpected dirty files appear outside preserved unrelated dirt and phase 01 artifacts.
- `.internal-dev/.archive/AGENTS.md` is unavailable and the process contract cannot be reconstructed from root instructions.
- Push or AgentMail send fails.

## Evidence Expectations

- Validation report: `.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-01-alpha-baseline-audit/validation/phase-01-validation-report.md`
- Evidence index: `.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-01-alpha-baseline-audit/artifacts/validation-summary.json`
- Commit hash and pushed branch/ref.
- AgentMail sent evidence/thread identifier.

## Do Not Close Unless

- Phase validator passes.
- Commit is pushed.
- HTML email report is sent.
- Validation summary records phase 01 accurately.
