# Sprint 02 Packaging Tools Changelog

## Date

2026-07-03

## Change Summary

Sprint 02 added a Rust `engine_pack` tooling path for alpha project/package/scene validation, package/project authoring, deterministic asset scanning, asset manifest mutation, and folder-based pack output. The sprint also documented the CLI in the API docs and closed with final validation evidence.

## Files

- `Cargo.toml`
- `Cargo.lock`
- `tools/engine_pack/Cargo.toml`
- `tools/engine_pack/src/main.rs`
- `tools/engine_pack/tests/cli_validation.rs`
- `tools/engine_pack/fixtures/**`
- `docs/api/00-index.md`
- `docs/api/03-scene-graph-and-fragment-workflows.md`
- `docs/api/04-assets-sync-deferred-and-handles.md`
- `docs/api/10-packaging-cli.md`
- `.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-02-packaging-tools/**`

## Behavioral Impact

- Contributors can run `engine_pack validate-package`, `validate-project`, and `validate-scene` against package-backed project data.
- Contributors can create starter project/package manifests with `new-project` and `new-package`.
- Contributors can scan recognized asset files into deterministic manifest snippets with `scan-assets`.
- Contributors can append one manifest asset entry with `add-asset`.
- Contributors can folder-pack a validated project with `pack`, which writes copied files and `PACK_REPORT.json`.
- The CLI rejects unsafe project/package paths and avoids stale pack reports after failed repacks.

## Risks

- `pack` is intentionally folder-based and does not create binary archives, thumbnails, import databases, runtime launchers, or editor placement transactions.
- Existing renderer warnings remain outside this sprint's scope.
- Visual capture was not required for this CLI/schema sprint; Sprint 03 must use capture validation for editor placement.

## Follow-up Items

- Sprint 03: harden packaged-asset placement in the editor with save/reload and headless capture proof.
- Sprint 04: define the runtime project launcher and application development loop.
- Sprint 01: resolve its independent changelog timing and closeout gate.

## Amendment (2026-07-03 — Gate Review Remediation, AGR-030)

The `new-app` subcommand in `engine_pack` (with `--id` and `--name` flags) was pre-built during Sprint 02 as part of the packaging tool infrastructure, alongside `new-project` and `new-package`. This was not recorded in the original changelog. The feature-level changelog entry was deferred to Sprint 08. Test coverage exists in `tools/engine_pack/tests/cli_validation.rs`.

## Amendment (2026-07-03 — Gate Review Remediation, AGR-037)

The following files were identified by the gate review as within sprint 02 scope but not listed in the original changelog Files section:

- `src/renderer/src/data/asset_registry.rs` — shared asset validation logic used by engine_pack
