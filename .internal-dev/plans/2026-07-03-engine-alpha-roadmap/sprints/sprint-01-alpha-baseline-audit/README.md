# Sprint 01: Alpha Baseline Audit And Process Repair

Status: closed

## Objective

Make the repository's alpha planning inputs trustworthy before future feature sprints build on them.

## User-Visible Outcome

After this sprint, a contributor can identify the live workspace members, current docs/process contracts, known alpha residuals, and required validation gates without relying on stale gap-report claims.

## In Scope

- Restore or recreate active `.internal-dev/AGENTS.md` from the archived operating guide.
- Repair root/process documentation drift around workspace members, runtime entrypoints, and `.internal-dev` usage.
- Retire or replace stale `docs/gap-report.md` as current truth.
- Create a current alpha readiness report and consolidated bug/code-smell register.
- Create an alpha validation matrix usable by later sprints.
- Run compile/package checks and record blockers honestly.
- Establish headless capture readiness rules; run capture only if a changed visual surface requires it or if a readiness check is explicitly chosen during execution.

## Out Of Scope

- Renderer, shader, scene, asset, physics, audio, scripting, editor, or dogfood feature implementation.
- Broad refactors or dependency changes.
- New runtime behavior, new public APIs, new manifests, or schema migrations.
- Fixing discovered product defects unless a validator identifies a trivial docs/process-only correction.

## Target Surfaces

- Code: only Cargo manifests for read-only audit; no product code edits planned.
- Docs: `AGENTS.md`, `README.md`, `docs/api/00-index.md`, `docs/internal/00-index.md`, `docs/gap-report.md`.
- `.internal-dev` artifacts: `.internal-dev/AGENTS.md`, sprint reports, reviews/registers, validation evidence, changelog after user-confirmed closeout timing if required.
- Validation artifacts: `.internal-dev/debug_reports/`, `.internal-dev/captures/` if capture runs, and this sprint's `artifacts/validation-summary.json`.

## Assumptions

- Current branch is `sprint/alpha-01-baseline-audit`; execution must stop if it is not.
- Existing dirty changes `.idea/engine.iml` and `.reasonix/` are unrelated and must be preserved.
- Code is logical truth and docs are intended truth.
- AgentMail is available to the orchestrator for post-phase HTML email reports to Dwight.
- Remote origin is `https://github.com/dhickel/vulkan-engine.git`; compare and commit links can be formed from pushed refs when push succeeds.

## Acceptance Criteria

- Sprint tracker status is `planned` before orchestration and progresses only through validated gates.
- Active `.internal-dev/AGENTS.md` exists and matches the documented process contract well enough for future agents.
- Root `AGENTS.md`, `README.md`, and docs indexes no longer present stale workspace/runtime facts as current.
- `docs/gap-report.md` is no longer cited as current truth; a current alpha readiness report replaces or clearly supersedes it.
- Consolidated residual bug/code-smell register exists under `.internal-dev/bugs/` or `.internal-dev/reviews/`.
- Alpha validation matrix exists and distinguishes compile, test, runtime smoke, capture, docs/process, and closeout gates.
- Required cargo checks ran or blockers are recorded with exact commands and failure summaries.
- Each phase has a validation report, commit hash, pushed branch/ref, and post-phase AgentMail HTML report evidence.

## Negative Criteria

- Do not use stale gap-report claims as source of truth without live verification.
- Do not edit product code or schemas to make docs true.
- Do not overwrite or stage unrelated dirty changes.
- Do not claim `fully_validated` if any required check, push, email report, or validator reconciliation is missing.
- Do not run broad `.internal-dev` scans beyond files needed for this sprint.

## Validation Plan

- Compile/test: `cargo check`, `cargo check -p renderer`, `cargo check -p renderer --examples`, `cargo check -p input`, `cargo test -p input`; add package-level checks for audio, physics, scripting, editor, and dogfood when phase audit requires them.
- Runtime smoke: use debug-record smoke only if docs/process changes need runtime evidence or final validation chooses a representative startup check.
- Visual/capture proof: use `.internal-dev/skills/engine-headless-capture-validation/SKILL.md` only for changed rendering/scene/shader/material/asset/Vulkan visual behavior or explicit capture readiness proof.
- Docs/process checks: stale-link sweep, stale gap-report references, workspace-member consistency, `.internal-dev` process consistency, validation-summary consistency.

## Advanced-Planner Handoff

Use `final-orchestration-plan.md` and the phase directives under `worker-directives/`. The plan is execution-ready for `orchestrate-plan`; no replanning is expected unless a stop condition fires.

## Closeout Checklist

- All phase validation reports are present and reconciled.
- Validation summary status is conservative and internally consistent.
- Known residuals are tracked.
- Post-phase email reports were sent and evidence recorded.
- Branch was pushed after every phase.
- Sprint tracker moved to the correct execution/validation/closed status by the orchestrator.
- Changelog timing confirmed with user if required by repo guidance.
