# Phase 03 Validation Report: Dev Loop Docs

Date: 2026-07-03
Branch: `sprint/alpha-04-runtime-launcher`
Scope: Sprint 04 Phase 03 docs only

## Summary

Phase 03 updated public docs to match the implemented root runtime launcher. The docs now describe:

- the root `engine` binary as the alpha data-driven project launcher;
- the exact sample project launch command;
- the exact true headless draw-target capture command;
- renderer examples as diagnostics/API references;
- custom Rust app crates under `apps/<name>` as the supported custom-code loop;
- `apps/dungeon_dogfood` as a custom app crate, not a migrated project manifest;
- dynamic Rust hot reload, scripting, event system integration, physics/collision gameplay, audio gameplay integration, and dogfood manifest migration as deferred work.

Status: passed local validation, pending independent validator review.

## Docs Changed

| File | Change |
| --- | --- |
| `README.md` | Replaced root migration-stub wording with root launcher, sample project launch, headless draw capture, renderer diagnostics, and app-crate loop. |
| `docs/api/00-index.md` | Added runtime launcher navigation and root launcher/headless draw capture examples. |
| `docs/api/01-student-quickstart.md` | Updated quickstart paths so project launcher, renderer examples, and app crates are distinct. |
| `docs/api/07-engine-arguments.md` | Kept renderer-example argument scope and linked to the root runtime launcher page. |
| `docs/api/09-editor-asset-browser-and-wall-chunks.md` | Removed stale runtime-launcher limitation and pointed editor users to the root launcher. |
| `docs/api/10-packaging-cli.md` | Clarified that `engine_pack` validates/authors data consumed by the root launcher and no longer lists runtime launcher as deferred. |
| `docs/api/11-runtime-project-launcher.md` | Added root launcher command contract, headless draw capture path, argument reference, data flow, and app-crate loop. |
| `docs/gap-report.md` | Updated alpha readiness orientation from root stub/examples-only to root launcher plus renderer diagnostics/app crates. |
| `apps/dungeon_dogfood/README.md` | Documented dogfood as a custom Rust app crate and explicitly deferred migration/hot reload/scripting/event/physics/audio work. |

## Live Help Source

Checked with:

```bash
cargo run -- --help
```

The live help reports:

```text
Usage: engine --project <path> [options]
--project <path>
--scene <path>
--headless
--capture_target <present|draw>
--capture_frames <n>
--capture_frame_start <n>
--capture_frame_interval <n>
--capture_dir <dir>
--record_debug <seconds>
--record_debug_interval <ms>
--record_debug_path <path>
```

Docs were written against that contract.

## Stale-Reference Sweep

Command:

```bash
rg -n 'migration stub|runtime project launcher.*deferred|renderer examples.*only runtime|cargo run` prints|dynamic Rust hot reload|scripting implemented|physics implemented|audio implemented|prints migration guidance' README.md docs apps/dungeon_dogfood
```

Results:

| Hit | Classification |
| --- | --- |
| `docs/api/01-student-quickstart.md`: dynamic Rust hot reload, scripting, event-system integration, physics/audio integration, broad dogfood migration, and generated app templates are deferred | Accepted deferred-feature wording |
| `docs/api/11-runtime-project-launcher.md`: root launcher is not dynamic Rust hot reload, scripting, an event system, physics/audio integration, generated app templates, or gameplay lifecycle system | Accepted deferred-feature wording |
| `docs/api/10-packaging-cli.md`: dynamic Rust hot reload remains deferred | Accepted packaging limitation wording |
| `docs/api/09-editor-asset-browser-and-wall-chunks.md`: editor does not include dynamic Rust hot reload or other advanced authoring/runtime features | Accepted editor limitation wording |

No remaining current-tense hits claim that the root binary is a migration stub, that root `cargo run` only prints guidance, that renderer examples are the only runtime path, or that the runtime project launcher is deferred.

Independent validator initially found stale wording in `docs/api/02-renderer-lifecycle-and-frame-api.md`. That page now distinguishes:

- project manifests: `cargo run -- --project apps/editor/sample_project/engine.project.toml`;
- renderer facade diagnostics: `cargo run -p renderer --example api_test`;
- custom Rust apps: `cargo run -p <app>`.

The broader stale sweep still reports accepted context wording for schema migrations, legacy internal stubs, dogfood asset manifest paths, and the explicit launcher/app/deferred-feature docs. It no longer reports stale current-tense claims that root `cargo run` only prints guidance or that renderer examples are the primary runtime path.

## Commands

| Command | Result | Notes |
| --- | --- | --- |
| `cargo fmt --check` | Passed | Exit 0 |
| `cargo check -p engine` | Passed | Exit 0; existing renderer warnings remain |
| `cargo check -p editor` | Passed | Exit 0; existing renderer warnings and editor dead-code warning remain |
| `cargo check -p engine_pack --locked` | Passed | Exit 0; existing renderer warnings remain |
| `cargo run -- --help` | Passed | Exit 0; live root launcher help checked |
| `git diff --check` | Passed | Exit 0; rerun after validator remediation |
| stale-reference `rg` sweep | Passed with classified residuals | Only accepted deferred-feature wording remains |
| `python -m json.tool .../artifacts/validation-summary.json >/dev/null` | Passed | Exit 0 |

## Residuals

- Existing renderer dead-code warnings remain visible during compile/help checks.
- Existing editor dead-code warning for `set_active_scene_text` remains visible during `cargo check -p editor`.
- Phase 04 final capture/debug closeout remains pending.

## Conclusion

Phase 03 documentation criteria are satisfied from the main-thread implementation side. The docs now match the Sprint 04 launcher behavior and keep non-implemented advanced systems explicit.
