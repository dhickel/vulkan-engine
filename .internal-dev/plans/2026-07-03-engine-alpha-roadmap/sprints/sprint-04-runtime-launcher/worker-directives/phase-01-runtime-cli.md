# Phase 01 Worker Directive: Runtime CLI

## Objective

Replace the root `engine` binary migration-stub argument behavior with an alpha runtime launcher CLI contract.

## User-Visible Outcome

Users can run `cargo run -- --help` and see root runtime launcher usage for `--project`, headless mode, capture flags, and debug timing flags. Invalid CLI input fails with controlled errors.

## Direct Editable Targets

Primary:

- `src/main.rs`
- Optional new root binary modules under `src/`, such as:
  - `src/launch.rs`
  - `src/runtime.rs` only as a stub/shape if needed for Phase 02 handoff

Tests:

- Unit tests inside new root modules.
- If integration tests are more appropriate, add under root package test targets according to Cargo conventions.

Evidence:

- `validation/phase-01-validation-report.md`
- update `artifacts/validation-summary.json` only to a conservative in-progress/pass-for-phase state.

Forbidden:

- `apps/editor/src/*` except reading for reference.
- `src/renderer/src/vulkan/*`
- `apps/dungeon_dogfood/*`
- docs, except no docs are required in Phase 01.

## Supporting Docs To Read

- `00-specification-lock.md`
- `01-current-state-analysis.md`
- `02-target-design.md`
- `shared/implementation-notes.md`
- `apps/editor/src/launch.rs`
- `apps/editor/src/main.rs` capture/debug helper sections

## Senior-Engineer Guidance

- The root CLI should fail unknown flags. Editor launch parsing currently ignores unknowns; that is not suitable for a user-facing root launcher.
- Keep `main` thin: parse args, call a `run` function, map errors to exit codes.
- Model usage errors separately from runtime/validation errors so tests can assert stable behavior.
- Accept both `--flag value` and `--flag=value` for key options because editor launch already supports that style.
- Do not silently default `--project` to the editor sample. The root launcher should require an explicit project.
- Keep Phase 01 focused. It can define runtime structs/enums and stubs, but Phase 02 owns actual project rendering.

## Implementation Steps

1. Replace migration-stub logic in `src/main.rs` with:
   - logging initialization if needed;
   - parse/run/exit-code flow;
   - help text.
2. Add a CLI options type with fields for:
   - project path;
   - optional scene override;
   - headless;
   - capture frame/sequence/manual options;
   - capture target;
   - debug timing options.
3. Add validation for:
   - required `--project` unless `--help`;
   - missing flag values;
   - positive integer fields;
   - capture option dependencies;
   - mutual exclusion between single capture and capture sequence;
   - accepted capture targets.
4. Add tests for:
   - `--help`;
   - `--project path` and `--project=path`;
   - scene/debug/capture accepted forms;
   - missing project;
   - unknown flag;
   - invalid capture target;
   - zero count/interval;
   - incompatible capture options.
5. Wire parsed options to a placeholder runtime call if Phase 02 behavior is not yet implemented. The placeholder must not claim success for real project launch; it can return a clear "runtime loading not wired" only if tests avoid treating it as final runtime behavior.
6. Update `artifacts/validation-summary.json` conservatively, for example `implementation_in_progress` or `phase_01_code_checks_passed_phase_02_pending`.

## Acceptance Criteria

- `cargo run -- --help` exits `0` and prints root runtime launcher usage.
- Running without `--project` exits with a usage error and does not print old migration-stub example guidance.
- Invalid `--capture_target swapchain` fails with accepted values.
- Parser tests cover accepted and rejected forms.
- `cargo test -p engine` passes.
- No editor UI dependency is introduced.

## Negative Criteria

- Do not implement full runtime rendering in this phase unless it is trivially required by structure; Phase 02 owns that.
- Do not ignore unknown arguments.
- Do not keep old migration-stub behavior as the normal root command.
- Do not make root launcher call `cargo run -p editor` or renderer examples.

## Validation Commands

```bash
cargo fmt --check
cargo check -p engine
cargo test -p engine
cargo run -- --help
cargo run -- --project apps/editor/sample_project/engine.project.toml --capture_target swapchain
git diff --check
python -m json.tool .internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-04-runtime-launcher/artifacts/validation-summary.json >/dev/null
```

Expected behavior:

- invalid capture command returns non-zero;
- help returns zero;
- JSON summary parses.

## Evidence Expectations

Write `validation/phase-01-validation-report.md` with:

- command list and exit status;
- tests added;
- CLI behavior summary;
- residuals for Phase 02;
- validation-summary status.

## Commit/Push/Report Gates

- Commit only after Phase 01 validator passes.
- Commit scope should include root CLI code/tests and validation evidence only.
- Do not push unless main-thread orchestration opens the push gate.
- Do not send reports/email from this worker.

## Stop Conditions

- Stop if root package cannot have unit/integration tests without build-system changes.
- Stop if implementing CLI requires changing renderer internals.
- Stop if user-facing CLI behavior conflicts with the locked spec.

## Do Not Close Unless

- CLI parser tests exist and pass.
- `cargo run -- --help` works.
- invalid capture target fails.
- phase validation report exists.
- evidence summary remains conservative.
