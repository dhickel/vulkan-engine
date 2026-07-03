# Target Design

## Sprint Artifact Shape

Sprint 01 produces a durable baseline package:

- active process contract: `.internal-dev/AGENTS.md`;
- current alpha readiness report: preferably `docs/alpha-readiness-report.md` or a rewritten `docs/gap-report.md` that clearly says it supersedes the stale report;
- consolidated residual register: `.internal-dev/reviews/2026-07-03-alpha-baseline-register.md` or a focused bug directory under `.internal-dev/bugs/`;
- alpha validation matrix: `.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-01-alpha-baseline-audit/shared/validation-matrix.md` plus any docs-facing reference the worker chooses;
- canonical evidence index: `.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-01-alpha-baseline-audit/artifacts/validation-summary.json`.

## Process Contract

The active `.internal-dev/AGENTS.md` should restore the archived contract with any needed Sprint 01 refinements:

- targeted `.internal-dev` reads only;
- plans under `.internal-dev/plans/`;
- reviews under `.internal-dev/reviews/`;
- bugs under `.internal-dev/bugs/`;
- changelogs under `.internal-dev/changelogs/`;
- notes/future considerations require user confirmation when out of scope;
- finalized plan/bug artifacts move to sibling `.archive/` directories;
- code is logical truth and docs are intended truth.

## Documentation Contract

Root and docs indexes should make current facts easy to find:

- workspace members match root `Cargo.toml`;
- canonical runtime entrypoints distinguish root migration stub, renderer examples, editor app, dogfood app, and workspace crates;
- `docs/gap-report.md` should not be described as current known limitations unless replaced with a current report;
- stale historical claims should be labeled historical or removed from current docs.

## Residual Register Contract

The register should classify findings:

- `verified_current`: live source/docs confirm it still exists;
- `stale_resolved`: old claim contradicted by live source;
- `unknown_needs_audit`: cannot verify cheaply during Sprint 01;
- `accepted_alpha_debt`: known issue accepted for later sprint;
- `blocked_validation`: check could not run and why.

Each item should include evidence path, owner sprint if obvious, and next action.

## Validation Summary Contract

`artifacts/validation-summary.json` should include:

- top-level status using conservative values: `planned`, `phase_checks_passed`, `validator_failed`, `blocked_tooling_constraint`, `final_quality_pending`, `final_quality_review_passed_with_residuals`, or `fully_validated`;
- branch, commit hashes, pushed refs, and GitHub links when available;
- command results and evidence paths;
- phase validation report paths;
- post-phase email report evidence;
- capture status with a clear reason when capture is not required;
- residual risks and unresolved blockers.

## Email Report Contract

Every phase must send Dwight an HTML report via AgentMail after validation and push. Each report must include:

- phase name/status;
- files created/changed;
- line counts where practical;
- commands run;
- validation/capture evidence paths;
- commit hash;
- pushed branch/ref;
- GitHub compare/commit links when remote URL can be formed;
- residuals, blockers, or explicit "none found".
