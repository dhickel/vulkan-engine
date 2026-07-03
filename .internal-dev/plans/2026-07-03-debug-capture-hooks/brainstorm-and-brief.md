# Debug Capture Hooks Brainstorm And Ironed-Out Brief

Date: 2026-07-03

## Source Context Read

- `AGENTS.md`
- `src/renderer/AGENTS.md`
- `src/renderer/src/vulkan/AGENTS.md`
- `src/renderer/src/api/hooks.rs`
- `src/renderer/src/api/renderer.rs`
- `src/renderer/src/vulkan/vk_debug.rs`
- `src/renderer/src/vulkan/vk_render.rs`
- `src/renderer/src/vulkan/vk_types.rs`
- `src/renderer/src/rendergraph/mod.rs`
- `src/renderer/src/rendergraph/passes/{prepare_targets_pass,present_copy_pass,imgui_pass}.rs`
- `docs/api/05-hooks.md`

## Brainstorm

### Current Shape

We need an agent-owned visual proof loop for native Vulkan output under Wayland, where external screenshot tools are unreliable. The engine already has:

- public facade hooks (`RenderHook`) with frame index, viewport size, and optional depth handle;
- lower-level closures around `VkRender::render_with_hooks`;
- a `vk_debug::capture_and_save_image_view(...)` helper that copies an image to CPU and writes PNG;
- a rendergraph order of `PrepareTargetsPass -> SkyboxPass -> GeometryPass -> PresentCopyPass -> ImguiPass`;
- final present-image transition to `PRESENT_SRC_KHR` inside `ImguiPass` or the no-imgui branch.

The public hook context is intentionally narrow and does not expose command buffers or Vulkan images. That is good for app safety, but insufficient for frame capture.

### Promising Directions

1. One-shot capture request API on `Renderer`
   - Add a facade method such as `request_debug_frame_capture(path, options)`.
   - The renderer stores a pending capture request and fulfills it during the render path.
   - Best for validation scripts and agents because app code does not need raw Vulkan access.
   - Main risk: careful synchronization and layout restoration.

2. Debug capture hook service below public `RenderHook`
   - Add a renderer-owned internal hook stage after rendergraph execution but before command buffer end/submit.
   - The service records copy commands into the active frame command buffer.
   - Best fit for current architecture because post-render closure already fires at the right time.
   - Main risk: post-render currently happens after `transition_present_for_present`, so capture either needs a second transition or a new capture stage just before final present transition.

3. Rendergraph capture pass
   - Add an optional `DebugCapturePass` after `ImguiPass` or split final-present transition into its own pass.
   - Best long-term architecture because capture becomes a named pass with timing and ordering.
   - More invasive because `ImguiPass` currently owns final present transition.

4. External capture fallback
   - Use compositor screenshot tools when available.
   - Weak fit: current Wayland session and installed tools do not give a reliable noninteractive capture path.

5. Test-only readback harness
   - Build a temporary or feature-gated capture path used only by validation.
   - Useful for narrow proof, but risks repeatedly rebuilding throwaway infrastructure.

### Holes And Risks

- `vk_debug::capture_and_save_image_view(...)` currently assumes `old_layout = UNDEFINED`, does not restore image layout, submits a separate command buffer and waits idle, and destroys only the buffer, not the allocation. It is useful as lineage but not production-ready frame proof.
- Capturing the present image after `ImguiPass` means the image may already be in `PRESENT_SRC_KHR`; capture needs a legal transition to `TRANSFER_SRC_OPTIMAL` and then back to `PRESENT_SRC_KHR`.
- Capturing before `ImguiPass` would miss editor panels, which defeats the visual proof goal for editor validation.
- Capturing the offscreen draw image is useful for render-only proof but misses present-copy and UI overlay behavior.
- Blocking queue waits inside the main render path are acceptable for one-shot debug proof but should not become continuous capture behavior.
- Readback memory must be mapped/coherent correctly. A PNG can be written only after GPU completion.
- Color channel ordering and row pitch must be handled explicitly. A naive tightly packed RGBA assumption can produce corrupt output on some formats.
- Swapchain image usage must support transfer source. If not guaranteed, capture should target a renderer-owned image or ensure swapchain creation includes `TRANSFER_SRC`.

### Recommendation

Build a first-class one-shot debug capture service with a facade request API and an internal Vulkan implementation. Do not expose raw Vulkan handles through public `RenderHookContext` for this use case.

Preferred target for the first sprint: final present image after ImGui, because this proves exactly what the user sees: scene, present copy, and editor/debug UI.

The clean architecture is to split final present transition out of `ImguiPass` into a small terminal step or pass:

1. `ImguiPass` leaves the present image in `COLOR_ATTACHMENT_OPTIMAL`.
2. Optional debug capture records a copy from present image to a readback buffer.
3. Terminal present transition moves present image to `PRESENT_SRC_KHR`.

That avoids transitioning from `PRESENT_SRC_KHR` back to transfer source in a post hook and makes ordering easier to reason about.

### Worth Considering Later

- Capture both `present` and `draw` targets for diagnosing UI/compositor vs render-path issues.
- Add `CaptureTrigger::AfterNFrames(n)` so startup can warm up before the proof frame.
- Emit sidecar JSON with frame index, viewport, render outcome, scene counts, timing snapshot, target, and file hash.
- Add small image checks: nonblank, nonuniform, expected dimensions, and optional marker-color probes.
- Add a capture comparison mode for golden smoke tests.

## Ironed-Out Brief

### Objective

Add a native renderer-side debug frame capture path that lets agents request a one-shot PNG proof from the engine itself, without relying on desktop screenshots or browser tooling.

### User-Visible Outcome

A validation command can run the editor or examples with a capture flag and produce files under `.internal-dev/debug_reports/`, for example:

```sh
RUST_LOG=debug timeout --signal=INT 60s cargo run -p renderer --example api_test -- \
  --capture_frame=30 \
  --capture_frame_path=.internal-dev/debug_reports/api_test-frame.png \
  --record_debug=10 \
  --record_debug_interval=50 \
  --record_debug_path=.internal-dev/debug_reports/api_test-timing.jsonl
```

The agent can then inspect the PNG directly and pair it with the existing timing JSONL evidence.

### Problem Type

Mixed feature and validation infrastructure.

### Recommended Approach

Implement a renderer-owned debug capture service:

1. Public facade request/configuration
   - Add `Renderer::request_debug_frame_capture(...)`.
   - Add launch flags for examples and `apps/editor`:
     - `--capture_frame=<n>` or `--capture_frame_after=<n>`;
     - `--capture_frame_path=<path>`;
     - optional `--capture_target=present|draw`.
   - Default output path should live under `.internal-dev/debug_reports/`.

2. Internal pending request state
   - Store pending one-shot capture config in `Renderer` or `VkRenderCore`.
   - Capture should complete once and report success/failure through logs and optional status API.

3. Rendergraph/frame ordering cleanup
   - Stop having `ImguiPass` perform the final `PRESENT_SRC_KHR` transition directly, or add a terminal post-imgui capture stage before final present transition.
   - Preferred order:
     - render scene to draw image;
     - copy draw to present;
     - draw ImGui/editor UI to present;
     - capture present image if requested;
     - transition present image to `PRESENT_SRC_KHR`;
     - end command buffer, submit, present.

4. Robust Vulkan readback helper
   - Replace or harden `vk_debug::capture_and_save_image_view(...)`.
   - Record copy commands into the active frame command buffer or use a controlled immediate-submit path after preserving layout.
   - Allocate a CPU-readable buffer with correct size and row handling.
   - Wait for completion before reading the buffer.
   - Save PNG and clean up both buffer and allocation.
   - Return `Result<DebugCaptureReport, DebugCaptureError>` instead of panicking.

5. Evidence sidecar
   - Write optional `.json` sidecar with:
     - capture target;
     - frame index;
     - viewport;
     - path;
     - bytes written;
     - timing snapshot if available;
     - source image format/layout assumptions;
     - success/failure reason.

### In Scope

- One-shot PNG capture from the final present image.
- Optional capture from offscreen draw image if it is cheap after present capture exists.
- CLI flags for renderer examples and `apps/editor`.
- Local output under `.internal-dev/debug_reports/`.
- Basic image sanity check helper for validation scripts or tests.
- Documentation updates for debug capture usage.

### Out Of Scope

- Continuous video capture.
- Golden image CI across drivers.
- Browser or desktop screenshot integration.
- Public raw Vulkan command buffer access in `RenderHookContext`.
- Full offscreen/headless renderer.
- General frame debugger UI.

### Non-Goals And Deferred Ideas

- Do not turn public hooks into unsafe Vulkan interop.
- Do not require Wayland compositor screenshot permissions.
- Do not make capture mandatory during normal rendering.
- Defer multi-frame capture, image diffing, and golden baselines until the one-shot path is stable.

### Target Surfaces

- `src/renderer/src/api/renderer.rs`
- `src/renderer/src/api/hooks.rs` only if context/status docs need adjustment
- `src/renderer/src/api/errors.rs`
- `src/renderer/src/vulkan/vk_debug.rs`
- `src/renderer/src/vulkan/vk_render.rs`
- `src/renderer/src/vulkan/vk_types.rs`
- `src/renderer/src/rendergraph/mod.rs`
- `src/renderer/src/rendergraph/passes/imgui_pass.rs`
- `src/renderer/src/rendergraph/passes/present_copy_pass.rs`
- `src/renderer/examples/common/mod.rs`
- `src/renderer/examples/api_test.rs`
- `apps/editor/src/launch.rs`
- `apps/editor/src/main.rs`
- `docs/api/07-engine-arguments.md`
- `docs/api/08-debug.md`
- `docs/internal/05-vulkan-sync-and-frame-lifecycle.md`

### Constraints

- Keep public facade safe and simple.
- Keep rendergraph/frame image layout transitions explicit.
- Preserve existing `--record_debug` behavior.
- Write agent evidence under `.internal-dev/debug_reports/`.
- Validation commands should stay bounded with `timeout --signal=INT 60s`.
- Do not depend on desktop screenshot tools.
- Treat code as logical truth when docs diverge.

### Assumptions

- First useful target is final present image after ImGui/editor UI.
- One-shot capture may stall the GPU briefly; acceptable for debug proof.
- PNG output is enough for visual validation.
- The capture request can trigger after a fixed frame count so startup can settle.
- A sidecar JSON report is worth the small extra work because it makes evidence auditable.

### Gotchas And Risks

- Swapchain present images may not currently be created with `TRANSFER_SRC` usage. Confirm and update swapchain creation if needed.
- Present image format may not be plain RGBA. Handle BGRA/RGBA swizzle explicitly.
- Row pitch may differ from `width * 4`; use buffer/image copy layout carefully.
- Current helper assumes `UNDEFINED` old layout and would break presentation if reused directly.
- Capturing after final present transition requires extra layout churn; better to capture before that transition.
- If rendergraph errors before ImGui, capture should not silently produce stale or blank output.
- If the window is minimized or resize-pending, capture should fail clearly or defer.
- Do not leak `vk_mem` allocations in the debug helper.

### Acceptance Criteria

- A user can request one PNG capture from an example or the editor through launch flags.
- The capture includes ImGui/editor UI when UI is enabled.
- The output path is deterministic when supplied.
- The engine logs whether capture succeeded and where it wrote the file.
- The render loop continues to present after capture.
- Failed capture returns/logs a clear error without panicking.
- Existing timing JSONL capture still works.
- Docs explain the capture flags and recommended agent command.

### Negative Criteria

- No raw Vulkan handles exposed through public `RenderHookContext`.
- No dependency on external desktop screenshot tools.
- No continuous capture or unbounded disk writes.
- No `unwrap()`/panic path in the capture helper for expected runtime failures.
- No final image layout left in `TRANSFER_SRC_OPTIMAL`.
- No claim of full visual validation from timing JSONL alone.

### Validation Expectations

- `cargo check`
- `cargo check -p renderer`
- `cargo check -p renderer --examples`
- Focused unit tests for launch flag parsing.
- Focused tests for capture request state transitions if implemented in pure Rust structs.
- Runtime smoke:
  - `RUST_LOG=debug timeout --signal=INT 60s cargo run -p renderer --example api_test -- --capture_frame=30 --capture_frame_path=.internal-dev/debug_reports/api_test-frame.png --record_debug=10 --record_debug_interval=50 --record_debug_path=.internal-dev/debug_reports/api_test-timing.jsonl`
  - equivalent editor command once editor launch supports the flags.
- Inspect generated PNG with local image viewer tooling.
- Optional scripted sanity check:
  - file exists;
  - dimensions match viewport;
  - image is not all black/transparent;
  - sidecar JSON exists and frame index is plausible.

### Open Decisions

- Capture trigger name: `--capture_frame`, `--capture_after_frames`, or reuse a `--record_frame_image` naming family.
- Whether first implementation supports only present target or both present and draw.
- Whether sidecar JSON is required for MVP or a follow-up.
- Whether final-present transition should become its own rendergraph pass now or a core post-render step.

### Advanced-Planner Handoff Report

Objective: implement native renderer-side one-shot debug frame capture for visual proof of examples and editor output.

Work type: feature plus validation infrastructure.

Likely targets: facade renderer API, launch parsers, Vulkan debug helper, rendergraph/pass transition ordering, docs, and validation scripts/artifacts.

Architecture constraints:

- public hooks remain safe and do not expose raw Vulkan handles;
- capture is renderer-owned internal debug infrastructure;
- final present capture should include ImGui/editor UI;
- image layout transitions must remain explicit and restored before present;
- `.internal-dev/debug_reports/` is the default evidence location.

Expected changes:

- add capture request/config types;
- add facade request/status API;
- add CLI flags to examples and editor;
- harden/replace `vk_debug` readback helper;
- adjust terminal present transition ordering;
- add docs and focused tests.

Validation gates:

- compile checks for workspace/renderer/examples;
- launch-parser tests;
- bounded runtime smoke producing PNG plus timing JSONL;
- visual inspection of generated PNG;
- negative check that resize-pending/minimized capture fails or defers clearly.

Review gates:

- verify no memory leaks in Vulkan allocation cleanup;
- verify no layout is left transfer-readable before present;
- verify capture failure cannot panic the render loop;
- verify docs distinguish visual proof PNG from timing JSONL.

