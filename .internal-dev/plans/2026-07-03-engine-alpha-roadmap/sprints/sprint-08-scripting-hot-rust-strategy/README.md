# Sprint 08: Scripting And Hot Rust Development Strategy

Status: planned

## Objective

Lock the alpha extension strategy for custom engine users: Rust app crates remain the primary path, generated/minimal app scaffolding becomes buildable and documented, scripting becomes an explicitly experimental event/log automation layer if implemented narrowly, and hot Rust reload is scoped as future dev-loop/tooling research rather than a runtime promise.

## User-Visible Outcome

An engine user can tell which extension path to choose:

- use `apps/<name>` Rust app crates for custom behavior;
- use `engine_pack` app-template support, if implemented, to create a minimal app that builds without renderer internals;
- package script assets only through a narrow durable-ID contract, if enabled;
- treat scripts and hot Rust reload as experimental/deferred unless Sprint 08 tests prove a specific claim.

## In Scope

- Current-state contract audit for docs, code, tests, and Sprint residuals.
- Minimal Rust app template path, likely in `tools/engine_pack`, that builds without modifying renderer internals.
- Conservative scripting crate hardening around log/event bindings, script errors, and explicit experimental status.
- Package/scene validation support for script assets only if narrow and testable.
- Documentation updates that remove stale "deferred" claims only when implementation exists.
- Validation reports, phase email drafts, final quality review, and conservative evidence index.

## Out Of Scope

- Full runtime scripting framework.
- Direct script access to renderer, Vulkan, scene mutation, physics mutation, or audio mutation.
- Dynamic Rust plugin ABI or dylib hot reload.
- Editor visual UI for script status unless specifically required by a validated narrow implementation.
- Broad dogfood migration to generated templates.
- Any desktop screenshot evidence.

## Target Surfaces

- Code: `src/scripting`, `src/events`, `src/renderer/src/data/asset_registry.rs`, `src/renderer/src/api/scene.rs`, `tools/engine_pack`, optional focused template fixture under `apps/` or `tools/engine_pack/tests/fixtures/`.
- Docs: `docs/api/00-index.md`, `docs/api/01-student-quickstart.md`, `docs/api/07-engine-arguments.md`, `docs/api/10-packaging-cli.md`, `docs/api/11-runtime-project-launcher.md`, `docs/api/12-events-and-lifecycle.md`, optional new scripting/app-template chapter, matching internal docs.
- `.internal-dev` artifacts: validation reports, phase email drafts, final quality review, and `artifacts/validation-summary.json`.

## Assumptions

- `src/scripting` currently wraps Rhai and exposes logging plus raw `engine_mut`; it does not have safe app/runtime bindings.
- `engine_events` already defines `ScriptId` and `ScriptingEvent`.
- `engine_pack` is the likely shipped Rust CLI home for app/project/package tooling.
- Sprint 07 residuals remain accepted unless Sprint 08 directly touches the blocked surface.
- Capture is not required unless visible renderer/editor behavior changes.

## Risks And Gotchas

- Docs currently mention app crates as the custom Rust path while marking generated app templates, scripting runtime, and hot Rust reload as deferred.
- `src/scripting` crate docs overclaim scene/asset bindings compared with live code.
- App-template generation can accidentally create workspace churn or require manual root `Cargo.toml` edits.
- Script binding scope can expand quickly into unsafe borrow/order problems.
- Package validation must reject runtime handles and path-only identity, consistent with earlier asset/audio/collision work.

## Acceptance Criteria

- Extension decision matrix is explicit: Rust app crates primary, scripts experimental, hot Rust reload deferred/research.
- Generated/minimal app template path builds or is explicitly not implemented with documented reason and revised gate.
- Script assets and script events are either narrow/tested or documented as unsupported/experimental.
- Script errors can be surfaced through stable errors/events/status artifacts when script evaluation is enabled.
- No script API exposes unsandboxed renderer internals or broad mutable engine state.
- Docs and tests prove only implemented claims.

## Negative Criteria

- No direct Rhai access to `Renderer`, Vulkan, data caches, or raw mutable scene internals.
- No dynamic library/plugin ABI in Sprint 08 implementation.
- No runtime guarantee that Rust code hot reloads.
- No desktop screenshots as validation evidence.
- No edits to `.idea/engine.iml` or `.reasonix/`.
- No false final status in `artifacts/validation-summary.json` while validators, residuals, or capture gates remain pending.

## Validation Plan

Use `shared/validation-matrix.md`. Required checks include formatting, relevant crate tests, workspace checks, engine_pack CLI/template tests, docs/stale-reference review, and final quality review. True engine-owned headless capture with `--headless --capture_target draw` is required only if visible renderer/editor behavior changes.

## Phase Handoff

- Phase 01: current-state contract audit.
- Phase 02: Rust app template path.
- Phase 03: script asset/event boundary.
- Phase 04: docs, final validation, and closeout preparation.

## Closeout Checklist

- All phase validation reports exist.
- Final quality review passes or records conservative residuals.
- `artifacts/validation-summary.json` is internally consistent.
- Phase email drafts exist for each completed phase.
- Main thread handles branch/push/email gates after each validated phase.
- Changelog timing and sprint tracker updates remain main-thread responsibilities.
