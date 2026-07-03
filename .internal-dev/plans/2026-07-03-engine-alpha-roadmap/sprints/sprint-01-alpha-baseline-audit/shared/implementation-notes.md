# Shared Implementation Notes

## Branch And Dirty State

- Required branch: `sprint/alpha-01-baseline-audit`.
- Preserve existing unrelated dirty changes:
  - `.idea/engine.iml`
  - `.reasonix/`
- Before every phase:
  - run `git status --short --branch`;
  - confirm only expected sprint files changed in addition to preserved unrelated dirt;
  - stop if unexpected product code changes appear.

## Commit And Push Gate

Each phase must:

1. Validate.
2. Write the phase validation report.
3. Update `artifacts/validation-summary.json`.
4. Commit only in-scope files.
5. Push `sprint/alpha-01-baseline-audit`.
6. Record commit hash, pushed ref, and GitHub links in the phase report.
7. Send the post-phase HTML email report to Dwight via AgentMail.

Use remote URL `https://github.com/dhickel/vulkan-engine.git` to form:

- commit link: `https://github.com/dhickel/vulkan-engine/commit/<hash>`
- branch compare link: `https://github.com/dhickel/vulkan-engine/compare/master...sprint/alpha-01-baseline-audit`

If the default base branch is not `master`, verify with `git remote show origin` or GitHub metadata before forming compare links.

## Validation Evidence

- Phase reports live under `validation/phase-XX-validation-report.md`.
- Command logs can be summarized in reports; full generated JSONL/capture files should live under `.internal-dev/debug_reports/` or `.internal-dev/captures/`.
- The canonical evidence index is `artifacts/validation-summary.json`.
- Never mark the sprint `fully_validated` until every required phase validator, push gate, and email report has passed.

## Docs Repair Principles

- Code is logical truth.
- Docs are intended truth.
- Replace stale claims with current facts or explicitly label them historical.
- Avoid broad aspirational language; state what exists, what is verified, and what remains alpha debt.
- Prefer links to current docs and sprint artifacts over duplicating long lists.

## Capture Guidance

Sprint 01 is process/docs-heavy. Capture is not required unless a phase changes visual rendering behavior or claims visual readiness.

If capture is required:

- read `.internal-dev/skills/engine-headless-capture-validation/SKILL.md`;
- define expected image behavior before running;
- use timeout-bound headless commands;
- record PNG and sidecar JSON paths;
- stop if capture output is missing or inconclusive.

## AgentMail Guidance

Planning artifacts specify report content; sending is an orchestrator responsibility through AgentMail. Do not block a worker on designing a new email workflow. Use `email-report-template.html` as the content template.
