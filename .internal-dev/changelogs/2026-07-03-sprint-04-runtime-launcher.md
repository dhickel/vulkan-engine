# Sprint 04 Runtime Launcher

Date: 2026-07-03
Branch: `sprint/alpha-04-runtime-launcher`

## Summary

Closed Sprint 04 by turning the root `engine` binary into the alpha data-driven project launcher and documenting the supported app development loops.

## Changes

- Added root launcher CLI parsing and controlled usage/runtime errors.
- Added project/package/startup-scene loading through the root runtime path.
- Added windowed and true headless root runtime loops.
- Wired capture and debug timing options through the root launcher.
- Documented:
  - root launcher for data-driven projects;
  - renderer examples as diagnostics/API references;
  - app crates under `apps/<name>` for custom Rust behavior;
  - dogfood as a custom app crate for now;
  - hot reload, scripting, event system integration, physics/collision gameplay, audio gameplay integration, generated app templates, and dogfood project-manifest migration as deferred.
- Repaired renderer doctest fences for illustrative internal snippets so renderer tests pass.

## Validation

- `cargo fmt --check`
- `cargo check`
- `cargo check -p renderer --examples`
- `cargo check -p editor`
- `cargo check -p engine_pack --locked`
- `cargo test -p engine`
- `cargo test -p renderer`
- `cargo test -p engine_pack --locked`
- Root runtime final capture:
  `RUST_LOG=info timeout --signal=INT 60s cargo run -- --project apps/editor/sample_project/engine.project.toml --headless --capture_target draw --capture_frames 3 --capture_frame_start 5 --capture_frame_interval 5 --capture_dir .internal-dev/captures/sprint-04-runtime-launcher/headless-draw`
- Root runtime debug smoke:
  `RUST_LOG=debug timeout --signal=INT 60s cargo run -- --project apps/editor/sample_project/engine.project.toml --headless --capture_target draw --capture_frames 1 --capture_frame_start 5 --capture_dir .internal-dev/captures/sprint-04-runtime-launcher/debug-smoke --record_debug 10 --record_debug_interval 50 --record_debug_path .internal-dev/debug_reports/sprint-04-runtime-launcher/root-runtime-timing.jsonl`

## Evidence

- Final draw capture sidecars under `.internal-dev/captures/sprint-04-runtime-launcher/headless-draw/`
- Debug smoke sidecar under `.internal-dev/captures/sprint-04-runtime-launcher/debug-smoke/`
- Debug timing JSONL at `.internal-dev/debug_reports/sprint-04-runtime-launcher/root-runtime-timing.jsonl`
- All final capture sidecars report:
  - `status = "succeeded"`
  - `capture_target = "draw"`
  - `format = "R16G16B16A16_SFLOAT"`
  - `extent = 1440x900`

## Residuals

- Existing renderer dead-code warnings remain visible.
- Existing editor `set_active_scene_text` dead-code warning remains visible.
- Event system, physics/collision, audio gameplay integration, scripting/hot reload, generated app templates, and dogfood project-manifest migration remain future sprints.

## Amendment (2026-07-03 — Gate Review Remediation, AGR-037)

The following files were identified by the gate review as within sprint 04 scope but not listed in the original changelog:

- Renderer API doctest files repaired during this sprint (see Changes section: "Repaired renderer doctest fences")
- `src/launch.rs` — root runtime launcher CLI parsing (listed in sprint 02 changelog but primarily developed in sprint 04)
