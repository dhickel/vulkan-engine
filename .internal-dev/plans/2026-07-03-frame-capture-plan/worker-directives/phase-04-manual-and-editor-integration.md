# Phase 04 Worker Directive: Manual Input And Editor Integration

## Objective

Wire manual input-triggered single capture and editor launch/runtime support into the capture system, while preserving existing input semantics and debug UI behavior.

## User-Visible Outcome

Pressing `F12` during local renderer/editor use queues one PNG capture under `.internal-dev/debug_reports/manual-captures/`. Editor launch accepts the same capture flags where applicable.

## Editable Targets

- `src/renderer/src/api/renderer.rs`
- `src/renderer/src/api/config.rs`
- `src/input/src/lib.rs` only if shared binding support is truly needed
- `src/renderer/examples/common/mod.rs`
- `apps/editor/src/launch.rs`
- `apps/editor/src/main.rs`
- `apps/editor/src/app_state.rs`
- `docs/api/06-input-polling-and-listeners.md` only if input contract changes
- `.internal-dev/plans/2026-07-03-frame-capture-plan/validation/phase-04-validation-report.md`

## Forbidden Scope

- Do not redesign the input system.
- Do not break F1/F2 debug/console behavior.
- Do not make manual capture depend on ImGui consuming keyboard input in a hidden way.
- Do not require editor full automation for this phase.
- Do not add email/remote coordination instructions.

## Supporting Docs To Read

- `src/input/AGENTS.md`
- `docs/api/06-input-polling-and-listeners.md`
- `docs/internal/09-input-winit-integration.md`
- `docs/api/08-debug.md`
- `shared/implementation-notes.md`

## Senior Engineer Guidance

- Put the manual trigger at the renderer input boundary where F1/F2 are already handled unless the input crate has an obvious action-binding fit.
- Prefer a small configurable/manual-capture config over hard-coding paths in multiple loops.
- If ImGui wants keyboard capture, decide deliberately whether F12 remains global debug infrastructure or is ignored while UI captures keyboard; document the behavior.
- For editor support, parse and forward launch flags; avoid full editor automation unless already present.

## Implementation Steps

1. Add manual capture config/defaults to renderer config or capture config.
2. In `Renderer::update_input`, detect non-repeat `F12` press and queue one capture.
3. Ensure manual capture path defaults to `.internal-dev/debug_reports/manual-captures/`.
4. Ensure filenames include app name, frame index, and timestamp or monotonic sequence.
5. Add optional `--manual_capture_dir` support to examples/editor if not completed in phase 01.
6. Wire editor launch capture flags to renderer config/request methods.
7. Add tests for manual output naming/config where possible.
8. Validate manual path through input automation or direct request API fallback.

## Acceptance Criteria

- Manual trigger queues one capture and writes one PNG on the next renderable frame.
- Default manual output directory is correct.
- Existing F1/F2 behavior still works or existing docs drift is recorded without worsening behavior.
- Editor launch accepts capture flags and configures renderer capture.
- Input automation blocker, if any, is recorded with direct API proof.

## Negative Checks

- No repeated captures from a held key due to key repeat.
- No conflict with existing Escape/fullscreen/example controls.
- No input crate semantic changes unless tests cover them.
- No hidden click/key behavior without docs.

## Validation Commands

- `cargo check`
- `cargo check -p renderer`
- `cargo check -p renderer --examples`
- `cargo check -p input`
- focused input/parser tests
- manual/direct request capture proof as feasible.

## Evidence Expectations

Write `.internal-dev/plans/2026-07-03-frame-capture-plan/validation/phase-04-validation-report.md` with:

- manual trigger behavior;
- PNG proof path or direct API fallback proof;
- editor flag wiring status;
- input/debug behavior risk assessment.

## Stop Conditions

- Stop if `F12` conflicts with a required editor/input behavior and cannot be made configurable in a small change.
- Stop if editor launch wiring requires unrelated editor architecture work.

## Do Not Close Unless

- manual capture default path is implemented and validated;
- editor launch behavior is implemented or explicitly scoped out by user gate;
- compile gates pass or blockers are recorded;
- phase validation report is written.

