# Phase 03 Validation Report: Script Asset And Event Boundary

Date: 2026-07-03
Validator: Codex validation agent

## Verdict

PASS. No blocking findings.

Phase 03 satisfies the directive's narrow scripting boundary: `src/scripting` now exposes durable `ScriptId`-aware evaluation reports, typed script errors, script-aware log binding, and collected `ScriptingEvent` values without hidden `EventBus` dispatch or default mutable access to renderer, scene, Vulkan, physics, audio, editor, dogfood, or app state. Script package assets are explicitly deferred and docs do not claim `script` is accepted as a package asset kind.

## Findings

No blocking findings.

## Criterion Results

| Criterion | Result | Evidence |
|---|---:|---|
| Read directive, specification lock, target design, senior guidance, phase audit, implementation report, and applicable governance docs | PASS | Reviewed `AGENTS.md`, `.internal-dev/AGENTS.md`, phase directive, `00-specification-lock.md`, `02-target-design.md`, `shared/senior-engineer-guidance.md`, Phase 01 audit artifact, and `reports/phase-03-email.md`. |
| Narrow experimental scripting API with durable `ScriptId`, `ScriptEvalReport`, `ScriptError`, `emit_event`, and script-aware logging | PASS | `src/scripting/src/lib.rs` defines `ScriptEvalReport` with `script/value/events`, `ScriptError` with `script/message/to_event`, `eval_for_script`, `eval_with_scope_for_script`, `eval_file_for_script`, log bindings, and `emit_event` bindings. |
| No hidden EventBus dispatch; returned `ScriptingEvent` values are app/runtime-owned | PASS | `src/scripting/src/lib.rs` imports only `ScriptId` and `ScriptingEvent` from `engine_events`; no `EventBus`, `EngineEvent`, `emit`, `drain_stage`, or `dispatch_pending` use exists in `src/scripting`. Docs state app/runtime code emits returned events at safe boundaries. |
| No default mutable access to renderer/Vulkan/scene/physics/audio/editor/dogfood/app state | PASS | Dependency scan found only the expected comment-only `renderer` match in `src/scripting/src/lib.rs:5`. `cargo tree -p scripting` shows only `engine_events`, `log`, and `rhai` plus Rhai transitive dependencies. |
| `engine_events` remains dependency-free from scripting/rendering/app crates; scripting may depend on `engine_events` | PASS | `src/events/Cargo.toml` has no dependencies. `src/scripting/Cargo.toml` adds `engine_events = { path = "../events" }`; `Cargo.lock` records only the `scripting -> engine_events` dependency edge. |
| `engine_mut` remains but is not promoted as normal path | PASS | `engine_mut` remains for advanced use and its doc comment tells callers to prefer script-ID-aware eval APIs for normal script execution. Only existing custom-function test uses it. |
| Script assets deferred; docs do not claim `script` asset kind is accepted | PASS | `AssetKind` has no `Script` variant; `docs/api/10-packaging-cli.md` says package-level script assets are deferred and `script` is not currently accepted. |
| Hot reload, dynamic plugin, production runtime scheduling not implemented or overclaimed | PASS | No new watcher/hot reload/plugin/runtime scheduler implementation in touched code. Docs say dynamic Rust hot reload, package-level script scanning/validation, and production scripting runtime are deferred. Existing renderer file-watcher references are unrelated pre-existing renderer internals. |
| Tests cover basic eval, log binding, emitted events, script error context, file eval error/success, and existing custom function path | PASS | `cargo test -p scripting` ran 9 tests: basic eval, log binding, variables, custom function, emitted event with payload, emitted event without payload, script error context, file eval error, and file eval success. |
| Capture not applicable | PASS | Phase changed non-visual scripting/docs behavior only. No renderer/editor visible behavior changed. |
| Protected `.idea/engine.iml` and `.reasonix` remain out of scope | PASS | `git status --short -- .idea/engine.iml .reasonix ...` showed `.idea/engine.iml` modified and `.reasonix/` untracked, but these were not part of the phase diff and were not touched by validation. |

## Commands Run

| Command | Result | Notes |
|---|---:|---|
| `cargo fmt --check` | PASS | No output. |
| `cargo test -p scripting` | PASS | 9 passed; 0 failed. |
| `cargo test -p engine_events` | PASS | 7 passed; 0 failed. |
| `cargo test -p renderer` | PASS | 160 unit tests passed; 17 integration tests passed; 5 doctests ignored. Existing renderer warning noise observed. |
| `cargo test -p engine_pack` | PASS | 20 CLI tests passed; existing renderer warning noise observed. |
| `cargo check` | PASS | Workspace check completed; existing renderer warning noise observed. |
| `cargo tree -p scripting` | PASS | Direct dependencies: `engine_events`, `log`, `rhai`. No renderer/app/audio/physics dependency edge. |
| `rg -n "renderer\|vulkan\|ash\|winit\|imgui\|physics\|audio\|dungeon_dogfood\|editor" src/scripting` | PASS | One expected comment-only match: `src/scripting/src/lib.rs:5`, saying renderer internals are not exposed. |
| Broader rg scan for dispatch/hot reload/script asset claims | PASS | No hidden scripting dispatch or script asset enablement found. Pre-existing renderer/doc references were reviewed as unrelated or deferred-language references. |
| `git diff --check` | PASS | No whitespace errors. |

## Implementation Surface Review

- `src/scripting/Cargo.toml`: Adds only local `engine_events` dependency.
- `src/scripting/src/lib.rs`: Adds the narrow event/log/error API and tests; no renderer, app, Vulkan, physics, audio, editor, or dogfood imports.
- `Cargo.lock`: Lockfile reflects `scripting -> engine_events`.
- `docs/api/10-packaging-cli.md`: Correctly says script package assets are deferred and `script` is not accepted.
- `docs/api/12-events-and-lifecycle.md`: Correctly states script events/errors are returned for app/runtime safe-boundary emission and production scheduling/hot reload are deferred.
- `reports/phase-03-email.md`: Matches observed source and command evidence.

## Browser/Capture Checklist

Not applicable. No visible renderer/editor/UI behavior changed, and the directive explicitly marks capture as not applicable.

## Residual Risk

No phase-blocking residual risk. `engine_mut` intentionally remains an escape hatch for advanced Rhai customization, so future examples/docs should continue avoiding any suggestion that direct renderer/scene/physics/audio bindings are the normal scripting path.
