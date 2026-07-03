# Current State Analysis

## Verified Current State

- Root workspace includes `src/scripting`, `src/events`, app crates, editor, renderer, and `tools/engine_pack`.
- `src/scripting/src/lib.rs` currently wraps Rhai with `log_info`, `log_warn`, `log_error`, `eval`, `eval_with_scope`, `eval_file`, `new_scope`, and raw `engine_mut`.
- `src/scripting` crate docs claim scene manipulation and asset access bindings, but live code does not provide them.
- `src/events/src/lib.rs` defines `ScriptId` and `ScriptingEvent::{ScriptEmitted, ScriptError}` while keeping `engine_events` independent of scripting/rendering crates.
- `tools/engine_pack` already owns `validate-package`, `validate-project`, `validate-scene`, `new-project`, `new-package`, `scan-assets`, `add-asset`, and `pack`.
- `AssetKind` currently includes `Audio` but not `Script`.
- Package/scene validators already reject runtime handle shapes and have durable-ID patterns for collision/audio.
- Docs currently say custom Rust behavior belongs in app crates under `apps/<name>`.
- Docs repeatedly mark dynamic Rust hot reload, scripting runtime, and generated app templates as deferred.
- Sprint 07 accepted residual: `cargo test -p dungeon_dogfood` is blocked before dogfood tests by renderer test-profile `russimp_sys` binding behavior in `src/renderer/src/data/assimp_util.rs`.

## Architecture Fit

- Primary extension path should reuse app crates because they already own custom control flow and can depend on facade/support crates directly.
- App-template tooling fits `engine_pack` only if it emits ordinary Rust crate files and avoids rewriting engine internals.
- Script package validation fits existing durable asset metadata patterns if `AssetKind::Script` and minimal metadata are added.
- Script runtime helpers should live in `src/scripting`; conversion to `EngineEvent::Scripting` can use `engine_events` types without making `engine_events` depend on scripting.

## Contract Conflicts

- Scripting crate docs overstate implemented bindings.
- User-facing docs say templates are deferred, but Sprint 08 may implement a template path. Docs must be updated only after code lands.
- Hot reload wording must distinguish data/script reload from Rust code recompilation.
- Any script "engine binding" language must avoid implying mutable renderer access.

## Security And Safety Concerns

- Rhai scripts must not receive raw renderer, Vulkan, cache, or broad mutable scene references.
- `engine_mut` is useful for advanced crate consumers but should be documented as unstable/low-level, not the beginner path.
- Script file loading must report errors without panics and without swallowing script ID context.
- Event emission from scripts must be staged/app-owned and must not recursively dispatch from inside unsafe callbacks.

## Validation Blind Spots

- `cargo check` does not prove generated app template output builds unless tests create/check the generated crate.
- Docs can easily overclaim hot reload because "hot" can mean assets, scripts, or Rust code.
- Script events can compile but still be useless if errors are not surfaced with durable script IDs.
- Capture evidence is irrelevant for non-visual work and should not be substituted with desktop screenshots.

## Known Residual Handling

- If `cargo test -p dungeon_dogfood` is run and blocked before dogfood tests by the existing `russimp_sys` issue, record it as an accepted inherited residual unless Sprint 08 changed that surface.
- Device audio smoke remains optional and should not block Sprint 08 unless a phase touches device audio behavior.
- No capture is required for non-visual script/tooling/docs changes.
