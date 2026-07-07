# Phase 01 Validator Review: Runtime CLI

Date: 2026-07-03
Branch: `sprint/alpha-04-runtime-launcher`
Validator: Codex validation agent
Scope: Sprint 04 Phase 01 root runtime CLI parser, help text, usage errors, tests, and conservative evidence summary.

## Findings

No blocking findings.

## Evidence Reviewed

- Governance: `AGENTS.md`, `.internal-dev/AGENTS.md`.
- Phase directive: `worker-directives/phase-01-runtime-cli.md`.
- Sprint criteria: `00-specification-lock.md`, `shared/validation-matrix.md`.
- Supporting context: `01-current-state-analysis.md`, `02-target-design.md`, `shared/implementation-notes.md`.
- Worker evidence: `validation/phase-01-validation-report.md`.
- Changed files: `src/main.rs`, `src/launch.rs`, `artifacts/validation-summary.json`.

## Code Review Notes

- `src/main.rs:8` keeps the root entrypoint thin: parse command, run, and map errors to exit codes.
- `src/main.rs:21` returns a controlled runtime error for valid launch syntax: `runtime project loading is not wired until Sprint 04 Phase 02`. This does not overclaim Phase 02 runtime loading.
- `src/launch.rs:91` treats `--help` as an explicit command and rejects combinations with runtime options.
- `src/launch.rs:116` and `src/launch.rs:196` support both `--project path` and `--project=path`.
- `src/launch.rs:270` rejects unknown flags instead of ignoring them.
- `src/launch.rs:285` requires `--project`.
- `src/launch.rs:311`, `src/launch.rs:321`, `src/launch.rs:332`, and `src/launch.rs:344` provide missing-value, empty inline-value, and positive-integer validation.
- `src/launch.rs:356` limits capture targets to `present` and `draw`; `swapchain` fails with accepted values.
- `src/launch.rs:364` validates capture dependency and mutual exclusion rules.
- `src/launch.rs:431` contains 10 parser/unit tests covering accepted forms and negative forms.
- `rg -n "editor|dungeon|renderer::|vulkan|migration stub|canonical renderer|cargo run" src/main.rs src/launch.rs` found only sample-path strings in tests. No editor UI, dogfood, renderer internals, or Vulkan dependency was introduced.

## Command Evidence

| Command | Exit | Result |
| --- | ---: | --- |
| `cargo fmt --check` | 0 | Pass |
| `cargo check -p engine` | 0 | Pass |
| `cargo test -p engine` | 0 | Pass, 10 tests passed |
| `cargo run -- --help` | 0 | Pass, prints root runtime launcher usage and no old migration-stub guidance |
| `cargo run --` | 2 | Pass, controlled missing-project usage error |
| `cargo run -- --project apps/editor/sample_project/engine.project.toml --capture_target swapchain` | 2 | Pass, controlled usage error naming accepted `present` and `draw` targets |
| `git diff --check` | 0 | Pass |
| `jq empty .internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-04-runtime-launcher/artifacts/validation-summary.json` | 0 | Pass |

Additional focused probes:

| Command | Exit | Result |
| --- | ---: | --- |
| `cargo run -- --project apps/editor/sample_project/engine.project.toml` | 1 | Pass, valid syntax reaches controlled Phase 02 runtime-pending error |
| `cargo run -- --project=apps/editor/sample_project/engine.project.toml` | 1 | Pass, inline project form reaches controlled Phase 02 runtime-pending error |
| `cargo run -- --project apps/editor/sample_project/engine.project.toml --bogus` | 2 | Pass, unknown flag rejected |
| `cargo run -- --project apps/editor/sample_project/engine.project.toml --capture_frames 2 --capture_dir .internal-dev/captures/tmp --capture_frame 1` | 2 | Pass, single and sequence capture modes rejected together |

## Criteria Results

| Criterion | Status | Evidence |
| --- | --- | --- |
| CLI parser/help/error behavior satisfies Phase 01 | Pass | Help, missing project, invalid capture target, unknown flag, and unit tests passed |
| Unknown args fail | Pass | `--bogus` returned exit 2 with `usage error: unknown flag '--bogus'` |
| No old migration-stub behavior remains in root CLI output | Pass | Help and missing-project output contain runtime launcher usage only |
| `--project path` and `--project=path` work | Pass | Unit tests plus runtime probes for both forms |
| Capture target/dependency/mutual exclusion/positive integer validation is covered | Pass | Unit tests cover invalid target, zero values, dependencies, and mutually exclusive capture forms; runtime probes confirmed invalid target and mutual exclusion |
| Valid runtime command does not falsely claim Phase 02 runtime loading is implemented | Pass | Valid project commands return exit 1 with explicit Phase 02 runtime-pending error |
| No editor UI/dogfood/renderer internals dependency introduced | Pass | Root files import only std/local module; text search found no forbidden dependency references outside test fixture paths |
| validation-summary JSON is conservative and consistent | Pass | `fully_validated` remains false, capture/debug evidence remains pending, Phase 02-04 remain pending; summary updated to mark Phase 01 validator passed |

## Phase 02 Gate

Phase 02 may proceed.

The remaining risks are expected Sprint 04 residuals, not Phase 01 failures: project/package/scene loading, renderer creation, headless/windowed runtime loops, draw-target capture proof, debug-record evidence, and docs updates are still pending later phases.
