# Current State Analysis

## Verified Planning Inputs

- Root guide says code is logical truth and docs are intended truth.
- `.internal-dev/AGENTS.md` requires controlled `.internal-dev` access, validation evidence, and changelog timing confirmation after finalized work.
- Renderer guide requires true headless capture for visual proof and points to `.internal-dev/skills/engine-headless-capture-validation/SKILL.md`.
- Roadmap defines Sprint 13 as the alpha release candidate sprint, with release notes, supported platforms/drivers, quickstarts, dogfood instructions, known issues, and contributor/agent workflow notes.
- Sprint tracker currently marks Sprint 09 as active/planned and Sprint 10-12 as proposed; Sprint 13 is proposed.

## Current Public Surface Snapshot

- `README.md` identifies the root `engine` binary as the alpha runtime launcher and points to:
  - `cargo run -- --project apps/editor/sample_project/engine.project.toml`
  - true headless draw capture for root runtime;
  - `cargo run -p dungeon_dogfood` for custom Rust app behavior.
- `docs/api/10-packaging-cli.md` documents `engine_pack` commands for validation, authoring, `new-app`, and folder `pack`.
- `docs/api/09-editor-asset-browser-and-wall-chunks.md` documents sample project editor placement, save/load, and editor headless draw capture.
- `docs/api/11-runtime-project-launcher.md` documents root launcher arguments and headless draw capture sidecar requirements.
- `apps/dungeon_dogfood/README.md` documents the app, audio smoke, full-content environment settings, and level selectors.

## Live Source Observations

- Root workspace includes `engine`, `renderer`, `input`, `audio`, `physics`, `scripting`, `engine_events`, `editor`, `dungeon_dogfood`, and `engine_pack`.
- Sample project files exist under `apps/editor/sample_project/`.
- Dogfood has documented full-content environment knobs:
  - `DUNGEON_DOGFOOD_FAST_STARTUP=0`
  - `DUNGEON_DOGFOOD_LOAD_PROPS=1`
  - `DUNGEON_DOGFOOD_LOAD_CUSTOM_ENV=1`
- Dogfood source scan found no obvious `--headless`/`--capture_target` release path. Phase 04 must verify this and either add a small app-owned true headless path or block release.

## Existing Evidence And Residuals To Re-Verify

- Sprint 03 accepted editor headless draw capture for a saved-scene copy.
- Sprint 04 accepted root launcher headless draw capture for the sample project.
- Sprint 07 records dogfood audio proof but also records a historical `cargo test -p dungeon_dogfood` blocker before dogfood tests due to renderer test-profile `russimp_sys` binding behavior.
- Sprint 08 records accepted residuals around protected local state, deferred renderer-window templates, deferred package-level script assets, dynamic/runtime Rust reload, and conditional dogfood tests.
- Sprint 09 evidence is partially started; Phase 01 validated, later phases are not started in the inspected evidence.

## Current Worktree Risk

`git status --short` during planning showed:

- `M .idea/engine.iml`
- `M src/renderer/examples/api_test.rs`
- `M src/renderer/examples/common/mod.rs`
- `M src/renderer/examples/demo_async_loading.rs`
- `M src/renderer/src/api/mod.rs`
- `M src/renderer/src/lib.rs`
- `M src/renderer/tests/integration.rs`
- `?? .reasonix/`
- `?? src/renderer/src/api/prelude.rs`

Treat these as active/unrelated local state for this planning turn. Sprint 13 execution must start from an integrated branch or isolated clean worktree, not from uncommitted local state.

## Architecture Fit

- Release docs should describe current contracts, not invent new APIs.
- `engine_pack` is the canonical project/package/scene validator and packer; release validation should use it instead of ad hoc schema parsing.
- Root `engine` is the data-driven runtime; dogfood remains a custom Rust app unless predecessor sprints change that contract.
- Visual validation belongs to engine-owned headless capture paths, not desktop tooling.

## Gaps For Phase Workers To Resolve

- Confirm final Sprint 10-12 contracts before release docs are locked.
- Confirm current `cargo test -p dungeon_dogfood` status; do not inherit old blocker without rerun.
- Confirm whether dogfood headless draw capture exists after Sprint 10-12. If absent, implement minimal release-proof support or block.
- Confirm public release doc destination. Phase 01 may choose a new `docs/alpha-release-candidate.md`, `docs/known-issues.md`, or sprint-local draft with public-doc links, but must avoid duplicate stale truth.

