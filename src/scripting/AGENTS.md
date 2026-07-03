# Scripting Crate Agent Guide (`src/scripting`)

Use this guide for Rhai scripting integration, script identity, and safe evaluation boundaries.

## Crate Role

`scripting` provides an experimental alpha scripting layer using the Rhai language:

- `ScriptEngine` for loading and evaluating scripts
- Safe logging builtins (no renderer, scene, or asset access)
- Script-emitted event collection through `engine_events`
- `ScriptEvalReport` with typed results and emitted events
- `ScriptError` with durable script identity

## Public API

- `ScriptEngine` -- evaluates scripts through a narrow boundary
- `ScriptEvalReport` -- result value + emitted events
- `ScriptError` -- typed error with script identity

## Architecture

- Uses `rhai` v1.x engine with minimal scope
- Scripts access only `log_info`, `log_warn`, `emit_event` builtins
- No mutable app state, renderer internals, asset caches, or scene mutation
- Event emission uses `engine_events::ScriptingEvent` and `engine_events::ScriptId`

## Current Alpha Status

- Core evaluation and event system work
- Builtins are limited to logging + event emission
- No hot-reloading or script module system
- No trait-based host function registration
- Track for full scripting feature set: Track G (future sprint)

## Safety Boundary

Scripts are sandboxed from:
- Renderer/Vulkan internals
- Scene mutation
- File system access
- Network I/O
- Thread spawning

This is enforced by the limited scope, not by OS-level sandboxing.

## Working Rules

- Do not expand script scope without explicit user decision
- Keep script identity durable through `ScriptId`
- Emit events through the shared `engine_events` crate
- If docs and code diverge, treat code as logical truth

## Validation

- `cargo check -p scripting`
- `cargo test -p scripting`
