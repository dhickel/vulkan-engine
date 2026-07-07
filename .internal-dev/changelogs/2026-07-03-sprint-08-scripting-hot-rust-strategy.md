# Sprint 08: Scripting And Hot Rust Development Strategy

Date: 2026-07-03

Branch: `sprint/alpha-08-scripting-hot-rust-strategy`

Status: closed with accepted residuals

## Summary

Sprint 08 locked the alpha extension strategy:

- Rust app crates remain the primary custom behavior path.
- `engine_pack new-app` now generates a standalone support-crate Rust app scaffold.
- The `scripting` crate now has an experimental `ScriptId`-aware eval/log/event/error boundary.
- Package-level script assets remain deferred.
- Dynamic Rust hot reload, plugin ABI loading, runtime hot reload, and renderer-window generated app templates remain deferred/tooling-only.

## Implementation

- Added `engine_pack new-app` with deterministic `Cargo.toml`, `src/main.rs`, and `README.md` generation.
- Added CLI tests for usage errors, existing-path protection, generated content, private renderer negative checks, and generated app `cargo check`.
- Added `ScriptEvalReport`, `ScriptError`, script-aware logging, and `emit_event(name[, payload])` collection in `src/scripting`.
- Kept script event dispatch app/runtime-owned; the scripting crate returns `ScriptingEvent` values and does not dispatch internally.
- Updated README and API docs to distinguish app crates, support scaffolding, experimental scripting, deferred script assets, and deferred hot Rust/runtime reload.

## Validation

- `cargo fmt --check`: pass
- `cargo check`: pass
- `cargo test -p scripting`: pass, 9 tests
- `cargo test -p engine_events`: pass, 7 tests
- `cargo test -p renderer`: pass, 160 unit tests, 17 integration tests, 5 ignored doctests
- `cargo test -p engine_pack`: pass, 20 CLI tests
- `cargo check -p renderer --examples`: pass
- `cargo check -p editor`: pass
- `cargo check -p dungeon_dogfood`: pass
- `cargo test -p dungeon_dogfood`: not applicable; Sprint 08 did not change dogfood expectations
- Capture: not applicable; Sprint 08 did not change visible renderer/editor behavior

## Residuals

- `fully_validated=false` in the evidence index by design because accepted residuals remain.
- Protected local state remains out of scope: `.idea/engine.iml` and `.reasonix/`.
- Renderer-window generated app templates are deferred.
- Package-level script assets are deferred.
- Dynamic/runtime Rust reload and plugin ABI loading are deferred.
- `cargo test -p dungeon_dogfood` remains conditional and was not run for Sprint 08.

## Evidence

- Plan suite: `.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-08-scripting-hot-rust-strategy/`
- Validation summary: `.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-08-scripting-hot-rust-strategy/artifacts/validation-summary.json`
- Final quality review: `.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-08-scripting-hot-rust-strategy/validation/final-quality-review.md`
