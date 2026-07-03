# Target Design

## Extension Strategy

Sprint 08 should leave the repo with a simple decision model:

| Need | Alpha Path | Status Language |
|---|---|---|
| Custom game/tool behavior | Rust app crate under `apps/<name>` | Primary supported alpha path |
| Data-driven project launch | Root `engine --project <path>` | Supported launcher path |
| New app skeleton | `engine_pack` generated/minimal app template if implemented | Supported only if generated output builds |
| Scripted automation/gameplay snippets | Rhai scripts with log/event/error boundary | Experimental |
| Asset/data/script reload | Future scoped hot reload target | Deferred unless implemented narrowly |
| Rust code hot reload | Incremental rebuild/dev-loop tooling | Deferred; not runtime guarantee |
| Dynamic plugin ABI | Future research | Out of scope |

## App Template Shape

Preferred implementation:

- Add a command under `tools/engine_pack`, for example `new-app`, only if it can be tested cleanly.
- Emit a minimal Rust app crate with:
  - `Cargo.toml`;
  - `src/main.rs`;
  - optional README;
  - dependency on public facade/support crates only.
- Generated code should mirror current documented app-loop shape and avoid private renderer modules.
- Test generation in a temporary directory and run a compile check against the generated manifest when practical.

Do not modify root workspace membership automatically unless the command and docs make that behavior explicit and tests cover it. If workspace integration is needed, prefer a clear opt-in flag or documented manual step over hidden mutation.

## Script Asset Boundary

If script assets are enabled:

- Add a durable `script` asset kind with relative file path validation.
- Define optional `[assets.metadata.script]` fields conservatively, such as:
  - `id` or derived asset ID;
  - `language = "rhai"`;
  - `entry = "main"` or event name only if implemented;
  - no runtime handles.
- Add scene references only if there is an actual consumer or validator path; otherwise keep package-level script assets only.
- Package/scene diagnostics must use stable codes and reject blank IDs, runtime handle shapes, absolute paths, parent traversal, unsupported languages, and invalid metadata types.

If script assets are not enabled, docs must say they remain deferred and Phase 03 should harden only the scripting crate/event boundary.

## Scripting Runtime Boundary

Allowed:

- `log_info`, `log_warn`, `log_error`.
- Emitting `ScriptingEvent::ScriptEmitted` from a narrow helper API.
- Returning `ScriptingEvent::ScriptError` or a typed/stable error with `ScriptId` on failure.
- Read-only/app-provided scope values, such as action names or selected scene metadata, only if represented as copies/snapshots.

Forbidden:

- Passing `Renderer`, `Scene`, Vulkan/data caches, physics world, audio engine, or mutable app state directly into Rhai.
- Scene/physics/audio mutation from script callbacks.
- Recursive event dispatch hidden inside script evaluation.
- Claims that scripts are a supported gameplay runtime beyond tested helpers.

## Hot Rust Scope

Sprint 08 should document:

- Alpha default: `cargo run -p <app>` with incremental Rust builds.
- Hot Rust reload is not a runtime feature.
- Dynamic plugin ABI is deferred.
- Future hot-loop candidates should prioritize asset/data/scripts before Rust code.

## Experience Contract

This sprint is non-visual by default. If implementation changes editor/runtime visible placement/status UI:

- Keep the current dense operational renderer/editor style.
- Status text must be readable and non-overlapping in desktop and narrow windows.
- Capture proof must use engine-owned true headless draw capture with `--headless --capture_target draw`.
- Desktop screenshots do not count.
