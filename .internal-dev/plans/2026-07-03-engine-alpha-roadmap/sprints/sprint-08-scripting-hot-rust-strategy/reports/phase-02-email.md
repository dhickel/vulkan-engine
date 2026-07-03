# Sprint 08 Phase 02 Report: Rust App Template Path

Date: 2026-07-03

## Summary

Implemented `engine_pack new-app` as a deterministic standalone Rust app scaffold. The command writes `Cargo.toml`, `src/main.rs`, and `README.md`, uses public support crates by absolute path (`engine_events`, `input`, `physics`), refuses existing target paths, and does not mutate the root workspace.

The generated app is compile-first and avoids renderer internals, dynamic Rust reload, plugin ABI loading, file watchers, dylib loading, and runtime hot reload. Renderer-window app generation remains deferred because off-workspace fresh-target checks that depend on `renderer` currently trip the existing `russimp_sys`/Assimp binding behavior outside this phase's allowed renderer scope.

## Files Changed

- `tools/engine_pack/src/main.rs`: added `new-app` command parsing, usage text, path protection, deterministic app file generation, app crate-name sanitization, and public support-crate path dependency rendering.
- `tools/engine_pack/tests/cli_validation.rs`: added tests for `new-app` usage failure, existing directory/file protection, deterministic generated content, private-renderer negative strings, and generated app `cargo check`.
- `docs/api/10-packaging-cli.md`: documented `new-app`, supported compile/check invocation, and the distinction between support-crate scaffolds and deferred renderer-window templates/hot reload.
- `docs/api/01-student-quickstart.md`: added quickstart command examples and updated deferred-feature language.
- `docs/api/11-runtime-project-launcher.md`: clarified that the root launcher is not app generation or hot reload and linked the `new-app` support scaffold path.
- `.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-08-scripting-hot-rust-strategy/reports/phase-02-email.md`: this report.

## Criteria Status

- Satisfied: `engine_pack` exposes a documented app-template path.
- Satisfied: generated app builds without renderer internals.
- Satisfied: tests cover generated output and failure behavior.
- Satisfied: docs no longer say generated app templates are wholly deferred; renderer-window generated templates remain deferred.
- Satisfied: hot Rust remains scoped as deferred runtime reload/plugin ABI work, not implemented.

## Validation

- `cargo fmt --check`: pass.
- `cargo test -p engine_pack`: pass, 20 integration tests passed.
- `cargo check -p engine_pack`: pass.
- `rm -rf /tmp/engine-sprint08-template && cargo run -p engine_pack -- new-app /tmp/engine-sprint08-template --id sprint08.template --name "Sprint 08 Template"`: pass, printed `created[app]: /tmp/engine-sprint08-template/Cargo.toml`.
- `rg -n "crate::vulkan|renderer::vulkan|renderer::data::|src/renderer/src" /tmp/engine-sprint08-template`: no matches.
- `cargo check --manifest-path /tmp/engine-sprint08-template/Cargo.toml`: pass.

## Residuals

- A renderer-dependent off-workspace app template was not implemented. Temporary probes showed fresh standalone `renderer` dependency checks fail in existing renderer `assimp_util.rs`/`russimp_sys` binding code, while `cargo check -p renderer` in the workspace passes from the current target state. Fixing that requires renderer dependency/build work outside Phase 02's allowed scope.
- No headless capture was applicable; this phase changed CLI/docs only.
