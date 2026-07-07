# Phase 03 Report: Script Asset And Event Boundary

Date: 2026-07-03

## Summary

Phase 03 hardened the experimental Rhai boundary in `src/scripting` around durable script identity, safe logging, script-emitted event collection, and typed error surfacing. Script events are returned to app/runtime code for explicit safe-boundary dispatch; the scripting crate does not dispatch events internally and does not expose renderer, scene, Vulkan, physics, audio, editor, dogfood, or mutable app state bindings by default.

Package-level script assets were deferred. Adding `.rhai` scan/add/manifest validation would require coordinated renderer registry and `engine_pack` schema work beyond the smallest complete scripting boundary change. Docs now state that `script` is not currently an accepted package asset kind.

## File Matrix

| File | Status | Reason |
|---|---|---|
| `src/scripting/Cargo.toml` | changed | Added local `engine_events` dependency so `ScriptEngine` can return `ScriptId`/`ScriptingEvent` without making `engine_events` depend on scripting. |
| `Cargo.lock` | changed | Necessary lockfile hygiene for the new local `scripting -> engine_events` dependency edge. |
| `src/scripting/src/lib.rs` | changed | Replaced overbroad crate docs, added `ScriptEvalReport`, `ScriptError`, ID-aware eval APIs, `emit_event` bindings, script-ID-aware log prefixes, compatibility wrappers, and focused unit tests. |
| `docs/api/10-packaging-cli.md` | changed | Clarified package-level script asset scanning/validation is deferred and `script` is not an accepted asset kind. |
| `docs/api/12-events-and-lifecycle.md` | changed | Documented the experimental scripting event boundary and explicit safe-boundary dispatch responsibility. |
| `src/events/src/lib.rs` | unchanged | Existing `ScriptId` and `ScriptingEvent` vocabulary was sufficient. |
| `src/renderer/src/data/asset_registry.rs` | unchanged | Script assets deferred. |
| `src/renderer/src/api/scene.rs` | unchanged | No scene script references added. |
| `tools/engine_pack/src/main.rs` / `tools/engine_pack/tests/cli_validation.rs` | unchanged | Script scan/add support deferred. |

## Validation

- `cargo fmt --check` passed.
- `cargo test -p scripting` passed: 9 tests.
- `cargo test -p engine_events` passed: 7 tests.
- `cargo test -p renderer` passed: 160 unit tests, 17 integration tests, 5 ignored doctests; existing renderer warning noise remains.
- `cargo test -p engine_pack` passed: 20 CLI tests; existing renderer warning noise remains.
- `cargo check` passed; existing renderer warning noise remains.
- `cargo tree -p scripting` shows dependencies limited to `engine_events`, `log`, and `rhai` plus Rhai transitive dependencies.
- `rg -n "renderer|vulkan|ash|winit|imgui|physics|audio|dungeon_dogfood|editor" src/scripting` returned only the reviewed crate-level doc sentence saying renderer internals are not exposed.

## Script Asset Status

Deferred. No `AssetKind::Script`, script package metadata schema, scene script references, or `engine_pack` `.rhai` scan/add support were added in this phase.

## Capture Status

Not applicable. This phase changed non-visual scripting/docs behavior only and did not affect renderer/editor layout or visible runtime behavior.

## Residuals

- `engine_mut` remains available for advanced use, but docs and comments now promote the script-ID-aware APIs as the normal path.
- Script event collection is intentionally narrow and returned to callers; production runtime scheduling, hot reload, and package-level script asset validation remain future work.
