# Phase 03 Worker Directive: Headless/Offscreen Capture

## Objective

Implement headless/offscreen frame capture as a core engine target, using renderer-owned images and a no-present path. If true headless is blocked by current architecture, stop at the user-decision gate with concrete evidence and options.

## User-Visible Outcome

Users and agents can run `--headless` with capture flags and receive PNG captures without relying on desktop screenshot access.

Visible windowed validation is not reliable as the long-term agent workloop: the runtime window appears on the user's active desktop and may be manually closed while the user is working. Treat manual close events as an expected operator interruption, not as proof that capture failed, and make true headless/offscreen capture the preferred validation path.

## Editable Targets

- `src/renderer/src/api/config.rs`
- `src/renderer/src/api/renderer.rs`
- `src/renderer/src/api/errors.rs`
- `src/renderer/src/vulkan/vk_init.rs`
- `src/renderer/src/vulkan/vk_render.rs`
- `src/renderer/src/vulkan/vk_types.rs`
- `src/renderer/src/rendergraph/mod.rs`
- rendergraph pass files touched in phase 02
- `src/renderer/examples/common/mod.rs`
- example files if loop/control-flow changes are needed for headless runs
- docs only if needed to record exact headless behavior in this phase
- `.internal-dev/plans/2026-07-03-frame-capture-plan/validation/phase-03-validation-report.md`

## Forbidden Scope

- Do not accept desktop screenshot fallback as headless proof.
- Do not implement a broad renderer rewrite beyond offscreen target/no-present needs.
- Do not remove the windowed path unless user explicitly approves headless-only architecture.
- Do not require editor UI capture in true headless unless robustly supported.

## Supporting Docs To Read

- `00-specification-lock.md`
- `02-target-design.md`
- `shared/validation-matrix.md`
- `docs/internal/05-vulkan-sync-and-frame-lifecycle.md`
- `src/renderer/src/vulkan/AGENTS.md`

## Senior Engineer Guidance

- The clean target is a present-target abstraction: windowed uses acquired swapchain images; headless uses owned offscreen present images.
- Keep no-present submission explicit: acquire and queue-present are windowed-only; command recording and graphics submit still run.
- Device selection without a surface may differ from current suitability checks. Keep changes minimal and document any surface-present assumptions that remain.
- If headless still creates a hidden window/surface, that is not true headless. Stop for user approval before treating it as acceptable.
- Validate scene examples first; editor/headless UI can be deferred unless easy.

## Implementation Steps

1. Introduce an internal target mode, for example `RenderSurfaceMode::Windowed` and `RenderSurfaceMode::HeadlessOffscreen`.
2. Add construction path for headless/offscreen rendering:
   - no required winit surface if feasible;
   - device/queue selection without present support if feasible;
   - renderer-owned offscreen present-color images with capture-compatible usage;
   - frame sync and command pool reuse compatible with existing frame ring.
3. Adjust frame acquisition:
   - windowed path acquires swapchain image and binds present target;
   - headless path chooses the current frame's offscreen present target without swapchain acquire.
4. Adjust submit/present:
   - both modes submit graphics work;
   - only windowed mode calls `queue_present`.
5. Ensure rendergraph passes target the current present-equivalent image in both modes.
6. Ensure capture from `CaptureTarget::Present` maps to offscreen-present in headless.
7. Wire `--headless` from launch options into `RendererConfig`.
8. Run at least one headless capture smoke.
9. Attempt full canonical headless matrix. If blocked, stop and write the gate report before substituting a fallback.

## Acceptance Criteria

- `--headless` capture writes a valid PNG through engine-owned capture.
- Headless mode does not require desktop screenshot access.
- Headless capture can run under the normal `timeout --signal=INT 60s` smoke pattern.
- Windowed capture from phase 02 still works.
- Full canonical headless matrix is either passing or blocked with explicit user gate evidence.

## Negative Checks

- No queue-present call in true headless mode.
- No reliance on winit window/swapchain in the true headless path unless the user approved a fallback.
- No claim that hidden-window capture is true headless.
- No broad unrelated renderer initialization rewrite.

## Validation Commands

- `cargo check`
- `cargo check -p renderer`
- `cargo check -p renderer --examples`
- `cargo check -p input`
- headless smoke:

```sh
RUST_LOG=debug timeout --signal=INT 60s cargo run -p renderer --example api_test -- \
  --headless \
  --capture_frames=3 \
  --capture_dir=.internal-dev/debug_reports/frame-capture/headless/api_test
```

- if feasible, full headless matrix from `shared/validation-matrix.md`.

## Evidence Expectations

Write `.internal-dev/plans/2026-07-03-frame-capture-plan/validation/phase-03-validation-report.md` with:

- true headless architecture summary;
- whether winit surface/swapchain are avoided;
- PNG proof metadata;
- full headless matrix status or gate blocker;
- any approved fallback status.

## Stop Conditions

- Stop if true headless requires a broad rework of instance/device/surface/swapchain ownership beyond this phase.
- Stop if the only feasible path is hidden-window/windowed capture and ask for user approval.
- Stop if a platform/GPU limitation prevents offscreen rendering and no bounded workaround exists.

## Do Not Close Unless

- true headless/offscreen capture works, or the user-decision gate report is complete and handed to the main thread;
- compile gates pass or blockers are recorded;
- phase validation report is written.
