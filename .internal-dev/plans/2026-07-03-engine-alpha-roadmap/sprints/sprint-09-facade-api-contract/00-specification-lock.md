# Specification Lock

## Acceptance Criteria

- The alpha-supported beginner facade surface is documented as a small, explicit API path for: renderer creation, project/package asset loading, scene creation/load/save, input update, render loop, asset load request/poll, and debug/capture controls.
- Public exports are audited and classified as one of: supported alpha beginner facade, legacy public compatibility, advanced interop, internal implementation detail, or deferred.
- Existing public exports are not removed unless the worker proves no in-repo consumer depends on them and the phase validator accepts the compatibility risk. The expected path is documentation/deprecation/classification, not breaking removal.
- A small alpha prelude or equivalent supported import path exists only if it materially reduces beginner friction and can be compile-checked.
- Renderer examples compile and use the same supported facade APIs that docs tell users to use.
- API docs identify duplicate/legacy chapters and point readers to the supported alpha path without attempting a full documentation rewrite.
- Tests or compile checks cover the supported public surface and examples.
- The evidence index records commands, validation reports, residual risks, capture status, and model/tooling constraints without claiming final validation early.

## Validation Criteria

- Required static checks:
  - `cargo fmt --check`
  - `cargo check`
  - `cargo test -p renderer`
  - `cargo check -p renderer --examples`
- Conditional checks:
  - `cargo test -p engine_pack` if engine_pack templates, docs, fixtures, or generated app guidance are touched.
  - `cargo test -p input` if input-profile schema or input crate behavior is touched.
  - `cargo doc -p renderer --no-deps` when public rustdoc or re-export structure changes, unless blocked by pre-existing doc failures recorded in evidence.
- Documentation stale scans:
  - `rg -n "stable public surface|Everything below api|legacy|TODO|pending|planned|not implemented|/tmp|sprint-08|Sprint 08|sprint-04|headless-draw" docs .internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-09-facade-api-contract`
  - `rg -n "AnimationPlayer|SceneWorld|CommandHistory|advanced-interop|prelude" docs/api src/renderer/src src/renderer/examples src/renderer/tests`
- Runtime smoke is optional and should be used only when implementation changes runtime behavior:
  - `RUST_LOG=debug timeout --signal=INT 60s cargo run -p renderer --example api_test -- --record_debug=10 --record_debug_interval=50 --record_debug_path=.internal-dev/debug_reports/sprint-09-api_test-timing.jsonl`
- Headless capture is required only if visible renderer/editor behavior changes:
  - Use `.internal-dev/skills/engine-headless-capture-validation/SKILL.md`.
  - Use engine-owned capture with `--headless --capture_target draw`.
  - Do not use desktop screenshots as proof.

## Negative Criteria

- Do not redesign the whole renderer API.
- Do not implement Sprint 10 advanced rendering opt-in beyond classifying or preserving `advanced-interop` boundaries.
- Do not remove root-level exports such as `AnimationPlayer`, camera helpers, command history commands, or `SceneWorld` without a compatibility plan and explicit validation.
- Do not promise internals as beginner APIs just because they are public today.
- Do not rewrite all old API docs; clarify the supported alpha path and mark duplicates/legacy status.
- Do not add package-level script assets, runtime Rust reload, renderer-window generated templates, or advanced rendering extension work.
- Do not count compositor/desktop screenshots as renderer proof.
- Do not touch `.idea/engine.iml` or `.reasonix/`.

## Non-Goals

- Production scripting runtime scheduling.
- Dynamic/runtime Rust reload.
- Package-level script asset implementation.
- Broad dogfood migration to project manifests.
- Renderer-window app template generation.
- Full public API semver stabilization.
- Full rustdoc cleanup outside the touched facade contract.

## Constraints

- Rust 2021 workspace.
- Code is the logical source of truth; docs are intended truth.
- `.internal-dev/` is ignored and the parent will force-add plan files.
- Current branch is `sprint/alpha-09-facade-api-contract`.
- Preserve unrelated local state: `.idea/engine.iml` modified and `.reasonix/` untracked.
- Phase execution must branch/push after every phase and send HTML email after every phase as an out-of-band main-thread responsibility.

## Assumptions To Verify

- `renderer::api` is intended to be the beginner facade namespace.
- Root-level `renderer::*` exports include compatibility leaks that may remain public but should be classified.
- Existing examples are the practical compile harness for beginner usage.
- Sprint 09 can add small wrappers or tests, but only where they remove real beginner friction and do not become a full redesign.

## User Decision Gates

- If a worker believes a public export must be removed, stop and ask before implementing.
- If alpha prelude naming conflicts arise, default to `renderer::prelude` only after confirming it does not collide with existing conventions; otherwise keep docs-only import guidance.
- If `cargo test -p renderer` fails for pre-existing doctest/prose issues unrelated to touched files, record residuals and ask whether to expand scope before broad doc cleanup.
- If visible behavior changes require capture but headless capture is blocked by tooling/runtime, record `TOOLING_CONSTRAINT` and stop before substituting desktop screenshots.

## Stop Rules

- Stop for plan revision if workers find the requested beginner surface conflicts with live architecture in a way that would require a larger API redesign.
- Stop for user approval before breaking public exports, changing workspace membership, or adding major dependencies.
- Stop phase progression if phase validation fails and remediation criteria are ambiguous.
- Stop final closeout if `artifacts/validation-summary.json` contradicts validation reports or claims final success while any required report is missing.
