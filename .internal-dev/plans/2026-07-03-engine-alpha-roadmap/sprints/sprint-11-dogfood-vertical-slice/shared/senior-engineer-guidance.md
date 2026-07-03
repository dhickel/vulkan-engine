# Senior Engineer Guidance

## Core Guidance

- Treat dogfood as a contract test for the engine, not as a place to hide missing engine features.
- Keep custom Rust gameplay in `apps/dungeon_dogfood`; move content identity and validation toward package/project/scene contracts.
- Prefer small bridges into existing validators and launch config over new schema copies.
- Validate live source before trusting roadmap wording, especially after Sprint 09/Sprint 10.
- Use conservative closeout language. `fully_validated` requires complete command, validator, and capture evidence with no accepted residual.

## Direct Targets

- Primary dogfood targets: `apps/dungeon_dogfood/src/*`, `apps/dungeon_dogfood/assets/*`, `apps/dungeon_dogfood/README.md`.
- Canonical validators: `tools/engine_pack/src/main.rs`, `tools/engine_pack/tests/cli_validation.rs`, renderer project/package/scene validation modules.
- Runtime/capture targets: `src/launch.rs`, `src/runtime.rs`, dogfood launch parsing, renderer capture API only if narrow and required.
- Docs: `docs/api/10-packaging-cli.md`, `docs/api/11-runtime-project-launcher.md`, `docs/api/00-index.md`, optional new dogfood chapter.
- Forbidden direct edits during execution unless main thread clears them: active Sprint 09 files currently modified in git status.

## Gotchas

- `content_pack.toml` is app-owned and uses absolute-looking repo-relative paths. Do not promote it as a canonical package contract without making the debt explicit.
- `engine_pack` must remain backed by renderer validators. Do not create a second validator in dogfood.
- Scene/project data cannot contain runtime handles.
- `--headless --capture_target draw` is mandatory for visual proof. Windowed smoke is useful but insufficient.
- The known `cargo test -p dungeon_dogfood` blocker may persist. Record it instead of pretending tests ran.
- Engine startup can take 20-30 seconds; use `timeout --signal=INT 60s`.

## Likely Failure Modes

- Package/project files validate, but dogfood still loads the old content pack silently.
- A capture command writes PNGs but sidecar JSON reports `capture_target = "present"`.
- The visual baseline uses fast-startup mode and misses props/environment, giving false confidence.
- API friction is buried in one-off dogfood code instead of filed.
- Docs list commands that do not exist after implementation.

## Reasoning Cues

- If a choice affects content identity, choose the package/project contract first.
- If a choice affects frame-to-frame gameplay behavior, keep it app-owned unless an engine API already exists.
- If a change touches renderer internals, ask whether the same outcome can be achieved through facade/runtime helpers.
- If a validation command cannot run, record the exact blocker and downgrade status.
