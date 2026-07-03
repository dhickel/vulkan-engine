# Specification Lock

## Release Candidate Contract

Sprint 13 does not declare the engine finished. It locks the first alpha release-candidate contract:

- a clean user can follow the public docs from clone to validation;
- the sample project demonstrates project/package/scene/runtime/editor round-trip behavior;
- dogfood demonstrates a custom Rust app path with documented content settings;
- limitations and known issues are explicit;
- release validation is repeatable from clean source state.

## Acceptance Criteria

- Alpha release notes draft exists under this sprint's `reports/` or a selected release-doc path and includes:
  - release scope;
  - supported quickstart workflows;
  - supported platform/driver/toolchain expectations;
  - known issues and accepted alpha limitations;
  - validation evidence summary;
  - no-release criteria.
- Public quickstarts are re-audited against live code and updated only where needed.
- Fresh-clone style validation is performed using either:
  - a true clone into `.internal-dev/fresh-clone-validation/sprint-13/engine`, or
  - a clean `git worktree` under `.internal-dev/fresh-clone-validation/sprint-13/worktree`.
- Sample project proof includes:
  - `engine_pack validate-project`;
  - `engine_pack validate-scene`;
  - folder `pack` output and `PACK_REPORT.json`;
  - editor launch/capture from the sample project;
  - a deterministic edit/save artifact separate from the canonical sample unless the release explicitly chooses to update the canonical sample;
  - root launcher run/capture of the saved scene.
- Dogfood proof includes:
  - documented content settings;
  - runtime smoke for `generated_sprawl`, `level_02_ramps`, and `level_03_lighting` or a recorded release-blocking reason;
  - true headless draw capture proof for at least the default full-content dogfood presentation.
- Final quality review reconciles implementation reports, validation reports, commands, captures, release docs, and `artifacts/validation-summary.json`.

## Validation Criteria

- Required command baseline:
  - `cargo fmt --check`
  - `cargo check`
  - `cargo check -p renderer`
  - `cargo check -p renderer --examples`
  - `cargo check -p input`
  - `cargo check -p engine`
  - `cargo check -p editor`
  - `cargo check -p dungeon_dogfood`
  - `cargo check -p engine_pack --locked`
  - `cargo test -p input`
  - `cargo test -p engine`
  - `cargo test -p engine_pack --locked`
  - focused tests for changed crates
- Required runtime/capture baseline:
  - root launcher sample project headless draw capture;
  - editor sample project headless draw capture;
  - dogfood full-content headless draw capture;
  - debug timing records where runtime diagnosis is needed.
- Required docs/evidence checks:
  - JSON validity for `artifacts/validation-summary.json`;
  - stale-reference sweep over changed docs and this sprint directory;
  - overclaim scan for unsupported alpha promises;
  - protected-path check for `.idea/engine.iml` and `.reasonix/`;
  - no tracker mutation by workers.

## Negative Criteria

- Release is blocked if fresh-clone validation fails from missing files, unstated local paths, unstated environment variables, or uncommitted local state.
- Release is blocked if sample project edit/save/run cannot be proven without corrupting canonical fixtures.
- Release is blocked if dogfood cannot provide true engine-owned headless draw capture and no user-approved exception exists.
- Release is blocked if release notes omit critical accepted residuals or inherited blockers.
- Release is blocked if visual validation uses desktop screenshots or present-target captures as replacement evidence.

## Non-Goals

- Publishing a GitHub release, tag, binary archive, or package registry artifact.
- Migrating all dogfood content into data-driven project manifests.
- Solving all renderer hygiene warnings.
- Production-ready editor, physics, audio, scripting, hot reload, or packaging archives.

## Constraints

- Planning-only artifact creation in this turn.
- Do not modify `SPRINT-TRACKER.md`.
- Do not modify `.idea/engine.iml` or `.reasonix/`.
- Do not modify active Sprint 09 files while Sprint 09 remains active; current reads are for planning context only.
- Use `.internal-dev` controlled access and do not read it broadly during execution.

## User-Decision Gates

- Before execution: confirm Sprint 13 should proceed despite current tracker showing Sprint 09 active and Sprint 10-12 proposed, or wait for predecessor sprint closeout.
- Before release candidate pass: user or main thread must decide whether accepted residuals are acceptable for alpha.
- Before changelog/knowledge/notes creation: ask the user per repo guidance.
- Before a fallback model/tool substitution: record `TOOLING_CONSTRAINT` and get user approval.

## Stop Rules

- Stop for planning revision if a phase discovers that release criteria conflict with actual Sprint 10-12 contracts.
- Stop implementation if a worker needs to touch protected paths.
- Stop release work if clean validation depends on current uncommitted local state.
- Stop visual validation if capture sidecars are missing, not draw-target, or cannot be inspected.
- Stop closeout if `artifacts/validation-summary.json` contradicts validation reports.

