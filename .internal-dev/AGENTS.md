# Internal Development Document Store Guide (`.internal-dev`)

This guide defines how agents use persistent engineering records in `.internal-dev/`.

## Purpose

`.internal-dev/` is the development document store for planning, bug capture, reviews, changelogs, notes, reusable knowledge, validation evidence, and sprint artifacts.

`.internal-dev/` is intentionally untracked in this repo so this document workflow can remain stable across repositories.

## Source-of-Truth Policy

- Code is the logical source of truth.
- Documentation is intended truth.
- If code and docs diverge, record the mismatch in task output and create or update a tracking artifact in `.internal-dev/`.
- Treat archived files as historical evidence unless a current task explicitly names them as a restoration source.

## Access Discipline

- Do not read `.internal-dev` directories or files randomly.
- Use controlled access: read only what the active task needs.
- Prefer targeted lookups over broad scans.
- Preserve unrelated local edits and untracked files unless the user explicitly asks to change them.

## Directory Contract

- `bugs/`: bug reports discovered during implementation or review.
- `plans/`: active implementation plans in nested plan directories.
- `reviews/`: completed review write-ups.
- `notes/`: future considerations and deferred ideas.
- `knowledge/`: reusable domain research and learner-facing summaries.
- `changelogs/`: dated change records that summarize completed work.
- `captures/`: renderer capture output used as visual validation evidence.
- `debug_reports/`: timeout-bound runtime/debug records used for diagnosis.
- `.archive/`: finalized or superseded artifacts in the same parent scope as the active content.

## Workflow Rules

- Out-of-scope bugs discovered in passing must be logged immediately.
- If a future consideration is identified and not implemented now, ask whether it should be recorded in `notes/`.
- Completed reviews are written to `reviews/`.
- Plans in progress should live in their own plan directories and include phase implementation files.
- When a bug or plan is finalized, move it to a sibling `.archive/` directory in the same parent path.
- Existing `plans/.completed/` content is legacy/read-only; use `.archive/` going forward.
- Finalized code or documentation changes should have a changelog entry in `changelogs/` when the user confirms it is time.
- Headless capture validation is required only when a task changes visible renderer behavior or claims visual proof; record capture paths and sidecar evidence when it is run.

## Minimum Templates

### Bug (`bugs/<bug-id>/report.md`)

Required headings:

- `Summary`
- `Scope`
- `Reproduction`
- `Expected`
- `Actual`
- `Evidence`
- `Impact`
- `Status`
- `Next Action`

### Plan phase (`plans/<plan-id>/phase-XX-<name>.md`)

Required headings:

- `Context`
- `Goal`
- `In Scope`
- `Out of Scope`
- `Implementation Steps`
- `Validation`
- `Exit Criteria`

### Review (`reviews/<date>-<topic>-review.md`)

Required headings:

- `Scope`
- `Findings`
- `Risk Assessment`
- `Recommendations`
- `Follow-ups`

### Changelog (`changelogs/<date>-<topic>.md`)

Required headings:

- `Date`
- `Change Summary`
- `Files`
- `Behavioral Impact`
- `Risks`
- `Follow-up Items`

### Knowledge (`knowledge/<topic>.md`)

Required headings:

- `Topic`
- `Source References`
- `Key Takeaways`
- `Engine Relevance`
- `Open Questions`

## Related Guides

- Top-level orientation: `AGENTS.md`
- API docs index: `docs/api/00-index.md`
- Internal docs index: `docs/internal/00-index.md`
- Headless capture validation skill: `.internal-dev/skills/engine-headless-capture-validation/SKILL.md`
