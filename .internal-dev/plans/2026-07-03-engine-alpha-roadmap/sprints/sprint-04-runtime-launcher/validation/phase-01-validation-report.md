# Phase 01 Validation Report: Runtime CLI

Date: 2026-07-03
Branch: `sprint/alpha-04-runtime-launcher`
Scope: Phase 01 parser/options/help/error behavior only.

## Implementation Summary

- Replaced the root `engine` binary migration-stub argument behavior with a thin parse/run/exit-code flow.
- Added `src/launch.rs` with root launcher options for `--project`, `--scene`, `--headless`, capture flags, and debug timing flags.
- Added controlled usage errors for missing `--project`, unknown flags, missing values, invalid positive integer fields, invalid capture target values, capture dependency errors, and single-vs-sequence capture conflicts.
- Added a Phase 02 placeholder runtime error for otherwise valid launches. It does not claim project runtime loading is implemented.

## Tests Added

- `--help` command and help text coverage.
- `--project path` and `--project=path`.
- Scene, headless, debug timing, capture sequence, capture target, manual capture directory accepted forms.
- Missing project.
- Unknown flags and unexpected positional arguments.
- Missing flag values and empty inline values.
- Invalid capture target including accepted `present` and `draw` values.
- Zero count, interval, and debug duration.
- Capture dependency errors and mutually exclusive capture modes.

## Validation Commands

| Command | Exit | Result |
| --- | ---: | --- |
| `cargo fmt --check` | 0 | Pass |
| `cargo check -p engine` | 0 | Pass |
| `cargo test -p engine` | 0 | Pass, 10 parser/unit tests |
| `cargo run -- --help` | 0 | Pass, prints root runtime launcher usage |
| `cargo run -- --project apps/editor/sample_project/engine.project.toml --capture_target swapchain` | 2 | Pass, controlled usage error with accepted capture targets |
| `cargo run --` | 2 | Pass, controlled missing-project usage error without old migration guidance |
| `git diff --check` | 0 | Pass |
| `jq empty .internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-04-runtime-launcher/artifacts/validation-summary.json` | 0 | Pass |

## CLI Behavior Summary

- `--help` and `-h` are supported as standalone commands and exit 0.
- `--project` is required for runtime launch commands.
- Both separated and inline value forms are accepted for key options.
- Unknown flags fail instead of being ignored.
- Capture target values are limited to `present` and `draw`.
- Capture sequence options require `--capture_frames`; single capture and sequence capture cannot be combined.

## Residuals

- Phase 02 still owns project validation, package loading, scene loading, renderer creation, and headless/windowed rendering.
- Valid project launch commands currently fail with a controlled runtime-pending error until Phase 02 wires runtime loading.
- Headless draw-target capture proof is still pending later Sprint 04 phases.
- Documentation remains stale until Phase 03.

## Validation Summary Status

`artifacts/validation-summary.json` is updated to `phase_01_code_checks_passed_phase_02_pending` and keeps `fully_validated` as `false`.
