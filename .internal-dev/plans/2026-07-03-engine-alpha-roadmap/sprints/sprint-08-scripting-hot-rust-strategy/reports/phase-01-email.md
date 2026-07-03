# Sprint 08 Phase 01 Report: Current-State Contract Audit

Phase 01 is complete. The audit verified the live app-crate, app-template, scripting, script-asset, script-event, hot Rust, and `engine_pack` contracts before implementation begins.

Key results:

- App crates under `apps/<name>` are the current custom Rust behavior path.
- Generated app-template tooling does not exist yet in `engine_pack`.
- No `script` asset kind exists in the renderer asset registry or `engine_pack` parser/scanner.
- The current scripting crate is a thin Rhai wrapper with log functions, generic eval helpers, scope creation, and raw `engine_mut`.
- `engine_events` already defines `ScriptId`, `EngineEvent::Scripting`, `ScriptingEvent::ScriptEmitted`, and `ScriptingEvent::ScriptError` while remaining dependency-free.
- API docs mostly keep scripting runtime, generated app templates, dynamic Rust hot reload, and hot reload/reimport deferred.
- Drift found: `src/scripting/src/lib.rs` crate docs overclaim scene manipulation and asset access bindings, and `docs/api/10-packaging-cli.md` is misleading when it says script records are manually authored even though `script` is not an accepted asset kind.
- Inherited Sprint 07 residual handling is preserved: `cargo test -p dungeon_dogfood` may still be blocked before dogfood tests by renderer test-profile `russimp_sys` behavior in `src/renderer/src/data/assimp_util.rs`.

Phase 02 readiness:

Phase 02 is ready to proceed. No user decision is required before app-template work as long as the implementation stays within `tools/engine_pack`, emits a minimal Rust app crate using public facade/support crates only, and tests that generated output builds. Docs should not claim generated app templates are supported until that command and validation exist.

Files produced:

- `.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-08-scripting-hot-rust-strategy/artifacts/phase-01-current-state-contract-audit.md`
- `.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-08-scripting-hot-rust-strategy/reports/phase-01-email.md`

Validation:

- `git status --short`
- `rg -n "scripting runtime|generated app templates|hot reload|dynamic Rust|app crates|script" docs src tools apps -g '*.md' -g '*.rs' -g '*.toml'`

No product code, tests, Cargo files, docs outside the sprint plan directory, `.idea/engine.iml`, `.reasonix/`, or unrelated local changes were edited.
