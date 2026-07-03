# Specification Lock

## Locked Objective

Sprint 01 audits the live alpha baseline and repairs documentation/process drift so future alpha sprints can rely on current facts, explicit residuals, and repeatable validation gates.

## Work Classification

Medium/full planning suite. The sprint spans docs, process artifacts, residual tracking, validation evidence, git push gates, and email reporting, but it should not mutate product behavior.

## Source-Of-Truth Hierarchy

1. Live code and manifests are logical truth.
2. Current docs are intended truth.
3. Archived docs, stale gap reports, and old memories are historical context only.
4. When docs and code diverge, record the divergence and either repair the docs/process in scope or file a residual/follow-up.

## Acceptance Criteria

- Active `.internal-dev/AGENTS.md` exists and documents the internal-dev process contract.
- Workspace membership in root guidance and public docs matches `Cargo.toml`: `src/input`, `src/renderer`, `src/audio`, `src/physics`, `src/scripting`, `apps/dungeon_dogfood`, `apps/editor`.
- Root runtime guidance reflects the current root binary migration-stub role and renderer/app entrypoints.
- Stale `docs/gap-report.md` is replaced, archived, or clearly superseded by a current alpha readiness report.
- A consolidated bug/code-smell register exists and cites whether each item is verified, stale, accepted, or follow-up.
- Alpha validation matrix exists and can be reused by later sprints.
- Each phase has:
  - validation report at `.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-01-alpha-baseline-audit/validation/phase-XX-validation-report.md`;
  - commit hash;
  - pushed branch/ref;
  - post-phase HTML email report to Dwight via AgentMail;
  - evidence paths for commands and captures when applicable.

## Validation Criteria

- Required checks are executed unless blocked with exact blocker evidence:
  - `cargo check`
  - `cargo check -p renderer`
  - `cargo check -p renderer --examples`
  - `cargo check -p input`
  - `cargo test -p input`
- Docs/process validation checks:
  - no current doc tells agents to treat stale `docs/gap-report.md` as current truth;
  - no current docs omit live workspace crates when describing workspace membership;
  - `.internal-dev/AGENTS.md` is active, not only archived;
  - validation-summary status does not overclaim.
- Capture validation:
  - not required for docs/process-only changes;
  - required if a phase changes rendering, scene, shader, material, asset, camera, or Vulkan visual behavior;
  - if required, use `.internal-dev/skills/engine-headless-capture-validation/SKILL.md` and store evidence under `.internal-dev/captures/`.

## Negative Criteria

- Product code changes are out of scope unless the phase directive explicitly scopes a tiny docs/process helper and the validator agrees it is not behavior-changing.
- Do not stage or revert `.idea/engine.iml` or `.reasonix/`.
- Do not claim sprint completion if any phase has missing validation, missing push, missing email report, or unresolved branch/worktree dirt.
- Do not use stale report claims as current defects without verifying against live source or marking them historical.

## Non-Goals

- No alpha feature delivery.
- No API redesign.
- No renderer runtime repair.
- No editor feature hardening.
- No packaging CLI implementation.
- No physics/audio/scripting integration.

## Constraints

- Work on branch `sprint/alpha-01-baseline-audit`.
- Push after each phase.
- Preserve unrelated dirty changes.
- Use targeted `.internal-dev` reads only.
- Use conservative closeout language from the engine alpha sprint skill.

## User-Decision Gates

- If changelog creation timing requires user confirmation under repo guidance, stop before writing the changelog and ask.
- If phase execution discovers a product defect whose repair is not docs/process-only, file/register it and continue only if the sprint can proceed without fixing it.
- If branch push fails because credentials, network, or remote permissions are unavailable, stop for user/main-thread decision.

## Stop Rules

- Stop if current branch is not `sprint/alpha-01-baseline-audit` and branch creation/switching would risk unrelated dirty changes.
- Stop if unrelated dirty changes expand beyond `.idea/engine.iml`, `.reasonix/`, and current sprint artifacts.
- Stop if required evidence is missing and cannot be regenerated.
- Stop if capture readiness is required but headless renderer initialization or capture output fails.
- Stop if a stale doc conflict cannot be classified as either repaired docs/process or tracked residual.
- Stop if AgentMail cannot send required post-phase HTML report.
