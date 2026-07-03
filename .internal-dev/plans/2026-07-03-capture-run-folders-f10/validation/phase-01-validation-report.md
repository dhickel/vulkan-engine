# Phase 01 Validation Report: Capture Run Folders and F10

Date: 2026-07-03
Validator: Codex validation agent

## Verdict

No blocking findings for the renderer example capture-default/F10 contract covered by the validation matrix.

Local closeout evidence is sufficient for Linux renderer example behavior: static review, capture unit tests, parser tests, compile evidence supplied by orchestrator, and existing headless single/N-frame PNG artifacts under `.internal-dev/captures`.

## Findings

### Medium - Editor `--headless` is parsed but still opens a window

- Files: `apps/editor/src/main.rs:47`, `apps/editor/src/main.rs:68`, `apps/editor/src/main.rs:74`, `src/renderer/src/api/renderer.rs:128`
- Evidence: editor `run()` builds a `winit` event loop/window and calls `Renderer::new(config, &window)` even when `launch_options.headless` is true. The actual headless path is `Renderer::new_headless(...)`, used by renderer examples but not by the editor.
- Impact: If the locked requirement "Headless mode remains available for single-frame and N-frame captures" is intended to include the editor binary, editor headless capture is incomplete and may still hijack the screen. If the plan intended headless validation only for renderer examples, this is a documented residual risk rather than a blocker.
- Classification: code_defect if editor headless was in scope; otherwise residual scope risk.

### Low - Public default path helpers still create fresh run folders per call

- Files: `src/renderer/src/api/config.rs:506`, `src/renderer/src/api/config.rs:519`, `src/renderer/src/api/config.rs:510`
- Evidence: `default_manual_capture_dir()` and `default_single_capture_path(...)` call `default_capture_run_dir(...)`, which includes current timestamp and PID. Repeated public calls can produce different folders in the same process.
- Impact: Examples/editor avoid this by computing one `capture_run_dir` per launch and using `single_capture_path(...)`. Existing external API consumers using the old default helpers repeatedly may see path shape and stability changes.
- Classification: API compatibility risk, not a local closeout blocker for the specified example/editor wiring.

### Governance Caveat - `.internal-dev/AGENTS.md` is referenced but absent

- Files: `AGENTS.md` repository guidance references `.internal-dev/AGENTS.md`; attempted read failed because the file does not exist.
- Impact: no additional internal-dev operating guide could be validated. The named plan artifacts were read directly.
- Classification: docs_or_evidence_defect / repo governance caveat.

## Criteria Results

| Criterion | Result | Evidence |
| --- | --- | --- |
| Default root is `.internal-dev/captures` | Pass | `src/renderer/src/api/config.rs:5`; unit test `default_capture_root_uses_internal_captures` passed |
| Per-run folders include sanitized app name, timestamp, PID | Pass | `src/renderer/src/api/config.rs:510`; unit test `run_dir_includes_sanitized_app_timestamp_and_pid` passed; runtime folders include `renderer-facade-api-test-20260703-...-pid...` |
| Single, sequence, and manual captures share one default run folder in examples/editor | Pass | examples compute `capture_run_dir` once at `src/renderer/examples/common/mod.rs:443` and `src/renderer/examples/api_test.rs:34`; editor computes once at `apps/editor/src/main.rs:66`; wiring uses passed run dir |
| Explicit `--capture_frame_path` override preserved | Pass | `src/renderer/examples/common/mod.rs:342`; `apps/editor/src/main.rs:830`; parser tests passed |
| Explicit `--capture_dir` override preserved | Pass | `src/renderer/examples/common/mod.rs:357`; `apps/editor/src/main.rs:845`; parser tests passed |
| Explicit `--manual_capture_dir` override preserved | Pass | `src/renderer/examples/common/mod.rs:335`; `apps/editor/src/main.rs:823`; parser tests passed |
| F10 queues one manual capture in windowed examples/editor | Pass by static review | `src/renderer/examples/common/mod.rs:760`, `src/renderer/examples/api_test.rs:361`, `apps/editor/src/main.rs:885` |
| F10 ignores key repeat | Pass by static review | all F10 handlers require `ElementState::Pressed` and `!key_event.repeat` |
| Renderer headless single/N-frame behavior preserved | Pass | orchestrator runtime evidence has one PNG in one run folder and three PNGs in one run folder; local capture tests passed |
| Editor headless behavior | Caveat | editor parses `--headless` but still creates a window and calls `Renderer::new` |

## Commands And Evidence

- `sed -n ...` / `nl -ba ...` over plan artifacts and implementation files: inspected requirements and line-level implementation.
- `git status --short`: confirmed dirty files include scoped implementation files plus unrelated `.idea/engine.iml` and `.reasonix/`; unrelated entries were not touched.
- `cargo test -p renderer capture -- --nocapture`: passed, 10 capture tests, existing warnings only.
- `cargo test -p editor parse_capture -- --nocapture`: passed, 2 parser tests, existing warnings only.
- `cargo test -p renderer parse_capture --examples -- --nocapture`: passed across renderer examples, existing warnings only.
- `find .internal-dev/captures -maxdepth 2 -type f -name '*.png'`: confirmed orchestrator evidence paths:
  - `.internal-dev/captures/renderer-facade-api-test-20260703-034337-713-pid93378/renderer-facade-api-test-frame-5-present.png`
  - `.internal-dev/captures/renderer-facade-api-test-20260703-034402-334-pid93693/renderer-facade-api-test-frame-5-present-seq-0000.png`
  - `.internal-dev/captures/renderer-facade-api-test-20260703-034402-334-pid93693/renderer-facade-api-test-frame-10-present-seq-0001.png`
  - `.internal-dev/captures/renderer-facade-api-test-20260703-034402-334-pid93693/renderer-facade-api-test-frame-15-present-seq-0002.png`

## Validation Notes

- Manual F10 was not executed interactively in this validation pass. The matrix marks it as a manual windowed check; current evidence is static review plus compile/test coverage.
- The full orchestrator commands listed in the prompt were not rerun except focused tests; they were treated as supplied evidence and reconciled with local code/file inspection.
- Two attempted targeted test commands used invalid `cargo test` syntax with multiple filters. They failed before running tests and were rerun correctly with a single `parse_capture` filter.
