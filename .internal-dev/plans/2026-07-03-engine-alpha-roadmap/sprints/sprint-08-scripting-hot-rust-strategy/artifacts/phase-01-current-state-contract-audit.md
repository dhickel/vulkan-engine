# Phase 01 Current-State Contract Audit

Date: 2026-07-03

## Scope

This audit verifies live source and API documentation before Sprint 08 implementation. It covers app crates, app-template tooling, scripting, script assets, script events, hot Rust claims, `engine_pack`, inherited Sprint 07 residual handling, and Phase 02 readiness.

## Verified Facts

- App crates are the current custom Rust behavior path. `AGENTS.md` lists `apps/dungeon_dogfood` and `apps/editor` as workspace members, and `docs/api/00-index.md:11` says custom Rust behavior belongs in app crates under `apps/<name>`. `docs/api/11-runtime-project-launcher.md:91-97` says app crates own Rust control flow and may depend on `renderer`, `input`, `physics`, and `audio` directly.
- Generated Rust app-template tooling does not exist yet. `tools/engine_pack/src/main.rs:39-48` exposes `validate-package`, `validate-project`, `validate-scene`, `new-project`, `new-package`, `scan-assets`, `add-asset`, and `pack`; there is no `new-app` or template command. `tools/engine_pack/tests/cli_validation.rs:285-353` tests project/package/asset authoring only, not generated app crates.
- `engine_pack` command docs match the live command surface. `docs/api/10-packaging-cli.md:23-71` documents validation, project/package authoring, asset scanning/addition, and packing, and `docs/api/10-packaging-cli.md:106-121` lists generated Rust app templates, scripting runtime, dynamic Rust hot reload, and hot reload/reimport as deferred.
- No `script` asset kind exists in the renderer package registry. `src/renderer/src/data/asset_registry.rs:122-131` defines `Model`, `Texture`, `Material`, `Environment`, `Prefab`, `WallChunk`, `SceneFragment`, and `Audio`; `src/renderer/src/data/asset_registry.rs:148-157` deserializes those strings only; `src/renderer/src/data/asset_registry.rs:165-175` serializes those strings only.
- `engine_pack` cannot scan or add script assets today. `tools/engine_pack/src/main.rs:484-492` classifies model, texture, environment, and audio extensions only. `tools/engine_pack/src/main.rs:495-510` parses authored asset kinds but omits `script`, so `--kind script` would produce `asset.unsupported_kind`.
- Existing package validation has durable asset patterns and runtime-handle rejection. `tools/engine_pack/tests/cli_validation.rs:50-73` asserts stable validation codes including `asset.runtime_handle_identity`, and `docs/api/10-packaging-cli.md:75-81` documents durable identity and runtime-handle exclusion.
- Current scripting API is a thin Rhai wrapper. `src/scripting/src/lib.rs:15-29` builds a `rhai::Engine` and registers only `log_info`, `log_warn`, and `log_error`. `src/scripting/src/lib.rs:32-65` exposes `eval`, `eval_with_scope`, `eval_file`, `engine_mut`, and `new_scope`. Tests at `src/scripting/src/lib.rs:78-109` cover basic eval, log binding, scoped variables, and custom function registration through `engine_mut`.
- Current scripting code does not emit `engine_events` values. `src/scripting/src/lib.rs` has no dependency on `engine_events`; script errors are returned as plain `String` values from `eval`, `eval_with_scope`, and `eval_file` at lines 33-55.
- The event boundary already has script vocabulary while remaining dependency-free. `src/events/src/lib.rs:1-5` states the crate is independent from renderer, editor, dogfood, physics, audio, and scripting crates. `src/events/src/lib.rs:64-74` defines `ScriptId`, `src/events/src/lib.rs:113-120` includes `EngineEvent::Scripting`, and `src/events/src/lib.rs:222-232` defines `ScriptingEvent::ScriptEmitted` and `ScriptingEvent::ScriptError`.
- Event dispatch is staged and app-boundary oriented. `src/events/src/lib.rs:76-100` defines lifecycle stages, `src/events/src/lib.rs:307-347` exposes subscription and emit, and `docs/api/12-events-and-lifecycle.md:25-27` says listeners receive `&EventEnvelope`, not mutable renderer access.
- API docs currently keep scripting runtime and hot Rust reload deferred. `docs/api/00-index.md:75`, `docs/api/01-student-quickstart.md:101`, `docs/api/10-packaging-cli.md:106-121`, `docs/api/11-runtime-project-launcher.md:7`, and `docs/api/11-runtime-project-launcher.md:97` all treat dynamic Rust hot reload, scripting runtime execution, and generated app templates as not currently supported.
- `docs/api/12-events-and-lifecycle.md:37-45` correctly says `ScriptingEvent` exists but scripting runtime and Rust hot reload are later roadmap work.
- Inherited Sprint 07 residual is recorded in the Sprint 08 plan. `01-current-state-analysis.md:14` and `01-current-state-analysis.md:46` say `cargo test -p dungeon_dogfood` may be blocked before dogfood tests by the renderer test-profile `russimp_sys` behavior in `src/renderer/src/data/assimp_util.rs`, and should be recorded as an accepted inherited residual if encountered.

## Drift And Claims To Update

- `src/scripting/src/lib.rs:1-4` overclaims live bindings. The crate comment says `ScriptEngine` provides engine API bindings for scene manipulation, logging, and asset access, but live code exposes only logging, generic Rhai evaluation, scope creation, and raw `engine_mut`. Phase 03 or Phase 04 should correct this comment unless the code gains tested bindings in scope.
- `docs/api/10-packaging-cli.md:92` says "Script, collision, and material records are still manually authored." That is misleading for scripts because `AssetKind` does not deserialize `script` and `engine_pack add-asset --kind script` is unsupported. It should become "script records are deferred" if Phase 03 does not add `AssetKind::Script`.
- The target design remains feasible, but Phase 02 must not update user-facing docs to say generated app templates are supported until a command exists and generated output is build-checked.

## Implementation Opportunities

- Phase 02 can add app-template tooling under `tools/engine_pack` without changing renderer internals if it emits a standalone app crate using public facade/support crates only.
- Phase 02 tests should extend `tools/engine_pack/tests/cli_validation.rs` with generated-app coverage and, where practical, `cargo check --manifest-path <generated>/Cargo.toml`.
- If Phase 03 enables script assets, it should update `AssetKind`, package parsing, `engine_pack` parsing/scanning as explicitly scoped, and stable validation tests together.
- If Phase 03 keeps script assets deferred, docs should explicitly say no package-level `script` asset kind exists yet.
- Script runtime hardening should add typed `ScriptId` error/event helpers in `src/scripting` without making `engine_events` depend on `scripting`.

## Blockers And User Decisions

- No required user decision is blocking Phase 02.
- No material contradiction was found between live code and the Sprint 08 target design. The target design says app-template tooling may be added if buildable; live code simply has not implemented it yet.
- No product behavior changes were made in this audit.

## Phase 02 Readiness

Phase 02 is ready to proceed if it stays inside `tools/engine_pack` and tests generated app output. The current source truth is:

- app crates are supported as the alpha custom Rust path;
- app-template tooling is absent;
- `script` asset kind is absent;
- scripting crate is eval/log/scope plus raw `engine_mut`;
- script event vocabulary exists in `engine_events`;
- hot Rust reload is deferred, not runtime-supported;
- inherited Sprint 07 `russimp_sys` residual remains an accepted conditional blocker for dogfood tests.

## Validation Evidence

- `git status --short` before edits showed unrelated local changes: `M .idea/engine.iml` and `?? .reasonix/`. These were not touched.
- Required search command was run:
  `rg -n "scripting runtime|generated app templates|hot reload|dynamic Rust|app crates|script" docs src tools apps -g '*.md' -g '*.rs' -g '*.toml'`
- No compile commands were needed because this phase changed only sprint plan artifacts.

## Repair Notes

- Corrected the `docs/api/01-student-quickstart.md` citation from `86-88` to `101` after validator review. The original cited lines were custom app pseudocode; line 101 is the deferred-feature statement.
