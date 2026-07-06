# Internal Development Document Store Guide (`.internal-dev`)

This guide defines how agents use persistent engineering records in `.internal-dev/`.

## Purpose

`.internal-dev/` is the development document store for specifications, planning, bug capture, reviews, changelogs, reusable knowledge, validation evidence, renderer captures, and sprint artifacts.

`.internal-dev/` is intentionally untracked in this repo so this document workflow can remain stable across repositories.

## Source-of-Truth Policy

- Code is the logical source of truth.
- Specifications are intended truth.
- Documentation and changelogs are historical or explanatory truth.
- If code, specifications, and docs diverge, record the mismatch in task output and create or update a tracking artifact in `.internal-dev/`.
- Treat archived files as historical evidence unless a current task explicitly names them as a restoration source.

## Access Discipline

- Do not read `.internal-dev` directories or files randomly.
- Use controlled access: read only what the active task needs.
- Prefer targeted lookups over broad scans.
- Preserve unrelated local edits and untracked files unless the user explicitly asks to change them.

## Directory Contract

- `specifications/`: living intended contracts, durable decisions, deferred capabilities, and horizon ideas.
- `bugs/`: bug reports discovered during implementation or review.
- `plans/`: active implementation plans in nested plan directories.
- `reviews/`: completed review write-ups.
- `knowledge/`: reusable domain research, implementation gotchas, validation patterns, and learner-facing summaries.
- `changelogs/`: dated change records that summarize completed work.
- `captures/`: renderer capture output used as visual validation evidence.
- `headless_capture_tests/`: temporary investigation specs and evidence for focused capture validation.
- `debug_reports/`: timeout-bound runtime/debug records used for diagnosis.
- `skills/`: repo-local skills used for engine-specific workflows.
- `.archive/`: finalized or superseded artifacts in the same parent scope as the active content.

Retired stores such as repo-local `focus/`, catch-all `notes/`, broad `research/`, and AgentMail inbox ledgers are not active workflow destinations. Use the specification, knowledge, changelog, bug, plan, and review stores above.

## Beginning Workflow

Before non-trivial work:

- Read `.internal-dev/specifications/AGENTS.md`.
- Read relevant files in `.internal-dev/specifications/` before changing renderer/runtime architecture, crate APIs, examples, tools, persistence-like artifact layout, workflow behavior, validation contracts, or user-facing engine behavior.
- List or search `.internal-dev/knowledge/` filenames and read only files whose filename or domain matches the task.
- If no knowledge filename looks relevant, proceed without broad reads.

When lost, confused, blocked by project context, or correcting a false assumption:

- Search `.internal-dev/knowledge/` filenames again.
- Run a deeper grep across `.internal-dev/knowledge/`.
- Use web or official documentation when the missing information is external framework, library, tool, protocol, or platform behavior and local knowledge is absent or stale.
- After resolving the learning, update a domain-named knowledge file when another agent is likely to need the same context.

## Mid-Workflow Routing

- Use specifications for intended contracts.
- Use `specifications/decisions.md` for durable architecture, design, product, and workflow tradeoffs.
- Use knowledge for reusable learning, framework techniques, implementation gotchas, validation patterns, corrections, and recurring failure modes.
- Use changelogs for prior edit context.
- Use bugs for defects.
- Use plans and reviews for scoped handoffs, implementation suites, validation campaigns, and completed review write-ups.
- Do not route active workflow material to retired catch-all stores. Classify the information into one of the active stores above.

## Specification Workflow

- Update existing living specification files by default.
- Create a new specification file only for a genuinely new specification class and update `specifications/index.md` in the same change with its ownership boundary.
- Future engine direction goes to `specifications/horizon-ideas.md`.
- Accepted deferred engine capability goes to `specifications/deferred-features.md`.
- Durable decisions go to `specifications/decisions.md` with justification, alternatives or tradeoffs when known, caveats, affected specs, source, and review timing.
- If an implementation has no impact on specifications, the changelog must say `Specification Impact: none` with one sentence explaining why.

## Knowledge Workflow

- When a false assumption, repeated mistake, major correction, important user correction, or repeated reverification reveals reusable context, update a domain-named knowledge file.
- Link the affected specification or changelog when useful.
- Name knowledge files after the domain they cover, not after a random incident title.
- If the reusable context is an intended contract, update the relevant specification instead or in addition.
- If the reusable context is a durable decision, record it in `specifications/decisions.md`.

## Workflow Rules

- Out-of-scope bugs discovered in passing must be logged immediately.
- If the project has a GitHub repository, every bug report created under `.internal-dev/bugs/` must be mirrored directly to that repository as a GitHub Issue when it is created or compiled.
- When adding or updating a local bug report in a project with a GitHub repository, check for related closed GitHub Issues before finishing; if the corresponding issue is already closed, move the local bug report to `.internal-dev/bugs/.archive/` instead of leaving it active.
- User hints like "future", "eventually", "later", or "this will become" go to `specifications/horizon-ideas.md` unless accepted as deferred engine capability.
- Any completed review is written to `reviews/`.
- Plans in progress should live in their own plan directories and include phase implementation files.
- When a bug or plan is finalized, move it to a sibling `.archive/` directory in the same parent path.
- Existing `plans/.completed/` content is legacy/read-only; use `.archive/` going forward.
- Finalized code or documentation changes should have a changelog entry in `changelogs/`.
- Headless capture validation is required only when a task changes visible renderer behavior or claims visual proof; record capture paths and sidecar evidence when it is run.
- Inbound AgentMail or remote-work coordination uses the global `mailctl status`, `mailctl next`, and `mailctl wait` workflow. Do not create a repo-local `.internal-dev/inbox` ledger.

## Closeout Workflow

- Update affected specifications, knowledge, bugs, changelogs, plans, and reviews.
- Record specification impact in the changelog, or state `Specification Impact: none` with one sentence explaining why.
- Record reusable lessons from false assumptions, repeated mistakes, large corrections, important user corrections, repeated reverification, and missing context in domain-named knowledge files.
- Report stale or conflicting specifications in the final response instead of silently rewriting broad project direction.
- Do not use retired catch-all workflow stores for closeout material.

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
- `Specification Impact`
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
- Specification guide: `.internal-dev/specifications/AGENTS.md`
- Specification index: `.internal-dev/specifications/index.md`
- API docs index: `docs/api/00-index.md`
- Internal docs index: `docs/internal/00-index.md`
- Headless capture validation skill: `.internal-dev/skills/engine-headless-capture-validation/SKILL.md`
