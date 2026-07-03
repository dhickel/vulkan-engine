# Phase 02 Worker Directive: Windowed Vulkan Capture Path

## Objective

Implement robust engine-owned PNG capture for the current windowed renderer path. Capture must occur from a named engine target, preferably final present image after UI and before terminal present transition.

## User-Visible Outcome

At least `api_test` can produce a valid PNG through capture flags in the normal windowed runtime. N-frame capture can write exactly N PNGs.

## Editable Targets

- `src/renderer/src/vulkan/vk_debug.rs`
- `src/renderer/src/vulkan/vk_render.rs`
- `src/renderer/src/vulkan/vk_types.rs`
- `src/renderer/src/vulkan/vk_init.rs`
- `src/renderer/src/vulkan/vk_util.rs`
- `src/renderer/src/rendergraph/mod.rs`
- `src/renderer/src/rendergraph/passes/imgui_pass.rs`
- `src/renderer/src/rendergraph/passes/present_copy_pass.rs`
- new rendergraph pass file if needed, for example `debug_capture_pass.rs` or `terminal_present_pass.rs`
- `src/renderer/src/api/renderer.rs`
- `src/renderer/src/api/errors.rs`
- capture-related tests/helpers
- `.internal-dev/plans/2026-07-03-frame-capture-plan/validation/phase-02-validation-report.md`

## Forbidden Scope

- Do not implement true headless/offscreen in this phase.
- Do not expose raw Vulkan handles in public API.
- Do not rely on desktop screenshots for proof.
- Do not rewrite the renderer outside capture, terminal transition, and required swapchain usage changes.
- Do not leave images in transfer layouts after capture.

## Supporting Docs To Read

- `01-current-state-analysis.md`
- `02-target-design.md`
- `shared/senior-engineer-guidance.md`
- `docs/internal/05-vulkan-sync-and-frame-lifecycle.md`
- `src/renderer/src/vulkan/AGENTS.md`
- `.internal-dev/plans/2026-07-03-debug-capture-hooks/brainstorm-and-brief.md`

## Senior Engineer Guidance

- First make the capture point legal. Move final `PRESENT_SRC_KHR` transition out of `ImguiPass` or add an equivalent named terminal step.
- Prefer active frame command-buffer recording over a separate queue wait. If an immediate path is chosen, justify it and preserve layout/fence correctness.
- Update swapchain usage only after checking surface support for `TRANSFER_SRC`. If unsupported, capture a renderer-owned offscreen-present target instead and document the windowed limitation.
- Convert formats explicitly. Present images are often BGRA; draw images are currently half-float.
- Treat `vk_debug::capture_and_save_image_view` as rough lineage only; replacing it is acceptable.

## Implementation Steps

1. Split final present transition:
   - `ImguiPass` leaves the current present target in `COLOR_ATTACHMENT_OPTIMAL`;
   - add `TerminalPresentPass` or explicit terminal transition after capture;
   - preserve no-Imgui behavior.
2. Add `DebugCapturePass` or backend capture stage after UI and before terminal present.
3. Implement structured capture reports/errors.
4. Replace/harden `vk_debug` readback:
   - explicit source layout and restored layout;
   - row pitch handling;
   - format conversion for present target;
   - vk_mem allocation cleanup;
   - non-panicking errors.
5. Ensure swapchain present images are capture-capable if using present-image capture. If not supported, record the constraint and capture from draw/offscreen-compatible target with clear status.
6. Wire due captures from the phase 01 scheduler into backend execution.
7. Write PNG and sidecar JSON.
8. Add sequence output naming and completion handling for exactly N captures.
9. Add focused tests for pure capture output naming/status where possible.
10. Run a windowed runtime smoke that writes at least one PNG.

## Acceptance Criteria

- Single windowed capture writes exactly one valid PNG.
- N-frame capture writes exactly N valid PNGs.
- Capture from `Present` includes UI when UI is drawn.
- Capture failure logs structured errors without panicking expected paths.
- Existing render loop continues after a successful capture.
- Existing `--record_debug` timing capture still works.
- Terminal present layout is correct when windowed presentation proceeds.

## Negative Checks

- No `PRESENT_SRC_KHR` image is copied without legal transition.
- No final image is left in `TRANSFER_SRC_OPTIMAL`.
- No vk_mem allocation leak in capture helper.
- No `unwrap()`/`expect()` in expected capture failure paths.
- No validation report claims full matrix proof yet.

## Validation Commands

- `cargo check`
- `cargo check -p renderer`
- `cargo check -p renderer --examples`
- `cargo check -p input`
- focused tests added/affected by phase 02
- one runtime smoke producing PNG, for example:

```sh
RUST_LOG=debug timeout --signal=INT 60s cargo run -p renderer --example api_test -- \
  --capture_frame=30 \
  --capture_frame_path=.internal-dev/debug_reports/frame-capture/phase02-api_test.png
```

Use the final implemented flag names if different.

## Evidence Expectations

Write `.internal-dev/plans/2026-07-03-frame-capture-plan/validation/phase-02-validation-report.md` with:

- command results;
- PNG path, `file` output, dimensions, nonblank/nonuniform check result;
- sidecar JSON excerpt/field checklist if present;
- layout/sync design summary;
- remaining risks for headless phase.

## Stop Conditions

- Stop if swapchain images cannot legally support transfer-source capture and no renderer-owned present-equivalent target can be added within this phase.
- Stop if capture requires public raw Vulkan handles.
- Stop if layout changes introduce validation-layer errors that are not understood.

## Do Not Close Unless

- a real engine-owned PNG was produced by a renderer example;
- N-frame scheduling writes exactly N files or a concrete blocker is recorded for repair;
- compile gates pass or blockers are recorded;
- phase validation report is written.

