# Sprint 13: Alpha Release Candidate

Status: planning_locked
Branch: sprint/alpha-13-alpha-release-candidate
Created: 2026-07-03

## Objective

Prepare the repository for the first community-facing alpha release candidate with public docs, release notes, known issues, workflow guidance, and repeatable release validation evidence.

## User-Visible Outcome

A project author can start from the documented quickstart, validate/package the sample project, open it in the editor, make a small durable scene edit, save it, run it through the root launcher, and run the dogfood app with documented content settings. A contributor can see supported platform/driver expectations, known limitations, and the exact validation evidence behind the release-candidate decision.

## Work Classification

Large. This sprint crosses public docs, release artifacts, packaging/editor/runtime/dogfood workflows, clean-checkout validation, visual proof, and final release/no-release governance. It is not primarily a feature sprint, but it may require small release-blocker fixes when validation proves existing release-critical commands do not work.

## In Scope

- Alpha release notes draft and known issues.
- Supported platform, runtime, Vulkan driver, and toolchain expectations.
- Quickstart project, package-tool, editor, runtime, and dogfood instructions.
- Contributor and agent workflow notes for release validation and residual tracking.
- Fresh-clone style validation through a clean worktree or fresh clone equivalent.
- Alpha sample project open/edit/save/run proof.
- Dogfood app run proof with documented content settings.
- True engine-owned headless draw capture proof for renderer/editor/dogfood visuals.
- Conservative release/no-release decision criteria.

## Out Of Scope

- New engine feature promises beyond alpha release-readiness blockers.
- Broad renderer cleanup or warning eradication unless a warning blocks release validation.
- Migrating dogfood to project manifests unless Sprint 10-12 already made that the release contract.
- Production binary/package archives, installers, or publishing automation.
- Updating `SPRINT-TRACKER.md`; the main thread owns tracker reconciliation.
- Touching `.idea/engine.iml` or `.reasonix/`.
- Touching active Sprint 09 files while Sprint 09 remains active, except read-only inspection.

## Target Surfaces

- Code: `src/main.rs`, `src/runtime.rs`, `src/launch.rs`, `apps/editor/`, `apps/dungeon_dogfood/`, `tools/engine_pack/`, `src/renderer/examples/` only when release validation proves a release-critical defect.
- Docs: `README.md`, `docs/api/00-index.md`, `docs/api/01-student-quickstart.md`, `docs/api/09-editor-asset-browser-and-wall-chunks.md`, `docs/api/10-packaging-cli.md`, `docs/api/11-runtime-project-launcher.md`, `apps/dungeon_dogfood/README.md`, new or existing release/known-issues docs as selected by Phase 01.
- `.internal-dev` artifacts: this sprint directory, validation reports, evidence summaries, release notes draft, known issues draft, and capture/debug evidence indexes.

## Assumptions

- Sprint 13 is intended to execute after Sprints 09-12 are closed or explicitly skipped/accepted by the user.
- Current local active Sprint 09 files are planning context only; Sprint 13 execution should start from an integrated clean base.
- If a true fresh clone is impractical, a clean `git worktree` from the intended release branch is an acceptable fresh-clone equivalent when the report records the exact command and isolation.
- Desktop screenshots do not satisfy visual validation. Required visual proof must use `--headless --capture_target draw`.

## Risks And Gotchas

- Current tracker marks Sprint 09 active/planned and Sprint 10-12 proposed; Sprint 13 cannot make a release decision until predecessor status is reconciled.
- Current worktree has pre-existing protected local state: `.idea/engine.iml`, `.reasonix/`, and Sprint 09 renderer files.
- Dogfood currently documents windowed runtime commands. If it lacks a true headless capture path, release validation must either add a small app-owned headless path or mark release blocked.
- Prior evidence records inherited renderer warning noise and a `cargo test -p dungeon_dogfood` test-profile risk around `russimp_sys`; Sprint 13 must re-verify current behavior rather than inherit old pass/fail claims blindly.

## Acceptance Criteria

- Release docs list supported and unsupported alpha workflows without overclaiming.
- Release notes draft includes features, limitations, known issues, validation evidence, and no-release conditions.
- Fresh-clone or clean-worktree validation runs with exact commands and evidence paths.
- Sample project validates, can be editor-opened, edited, saved, and launched through the root runtime.
- Dogfood runs with documented full-content settings and has true headless draw capture visual proof.
- `artifacts/validation-summary.json` records all phase statuses, commands, capture directories, tooling constraints, superseded artifacts, and residual risks conservatively.
- Final quality review either passes the release candidate or records explicit no-release blockers.

## Negative Criteria

- No final status may claim `fully_validated` while a required validator, command, capture proof, or residual decision is missing.
- No release doc may imply production stability, binary archives, production scripting, broad hot Rust reload, full physics/audio/editor integration, or dogfood project-manifest migration unless live code and validation prove it.
- No desktop/compositor screenshot may be used as renderer/editor/dogfood visual proof.
- No undocumented local-only state may be required to run the quickstart.
- No runtime handles may be serialized into sample project/package/scene files.

## Validation Plan

- Compile/test: `cargo fmt --check`, `cargo check`, package checks for `renderer`, `renderer --examples`, `input`, `engine`, `editor`, `dungeon_dogfood`, `engine_pack`, and focused package tests selected by changed surfaces.
- Runtime smoke: root launcher help/error cases, sample project runtime, editor sample project, dogfood levels.
- Visual/capture proof: `--headless --capture_target draw` for root runtime, editor, and dogfood release visuals; sidecars must report `status = "succeeded"` and `capture_target = "draw"`.
- Docs/process checks: stale-reference sweep, release-doc overclaim scan, `.internal-dev` evidence consistency, clean-worktree/fresh-clone isolation report.

## Phase Order

1. Phase 01: release inventory and docs lock.
2. Phase 02: fresh-clone/clean-worktree validation.
3. Phase 03: alpha sample project, package tool, editor edit/save/run proof.
4. Phase 04: dogfood run and headless draw visual proof.
5. Phase 05: release notes, known issues, workflow notes, and final quality review.

## Closeout Checklist

- Validation evidence recorded.
- Known residuals tracked and classified as release-blocking or accepted alpha debt.
- Changelog timing confirmed with user if required by repo guidance.
- Sprint tracker update left to the main thread.

