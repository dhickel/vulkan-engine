# Current State Analysis

## Sources Read

- `AGENTS.md` from the user-provided repo instructions.
- `src/renderer/AGENTS.md`
- `src/renderer/src/vulkan/AGENTS.md`
- `src/input/AGENTS.md`
- `docs/api/07-engine-arguments.md`
- `docs/api/08-debug.md`
- `docs/internal/05-vulkan-sync-and-frame-lifecycle.md`
- `.internal-dev/plans/2026-07-03-debug-capture-hooks/brainstorm-and-brief.md`
- `.internal-dev/plans/2026-07-03-debug-capture-hooks/gnome-screenshot-probe.md`
- Key code surfaces in `src/renderer/src/api/`, `src/renderer/src/vulkan/`, `src/renderer/src/rendergraph/`, `src/renderer/examples/common/mod.rs`, and `apps/editor/`.

## Verified Current Facts

- Public runtime construction is window-first: `Renderer::new(config, window)` always receives a `winit::window::Window`.
- `RendererConfig.headless` exists, but current behavior only logs that full offscreen rendering is not implemented while still creating window/swapchain resources.
- Shared example launch parsing lives in `src/renderer/examples/common/mod.rs` and currently handles `--env`, `--model`, and timing capture flags.
- Existing timing capture is JSONL-only and must remain compatible.
- Rendergraph default order is `PrepareTargetsPass -> SkyboxPass -> GeometryPass -> PresentCopyPass -> ImguiPass`.
- `PresentCopyPass` moves the offscreen draw image to the present image.
- `ImguiPass` currently owns final present transition:
  - if ImGui is not drawn, it calls `transition_present_for_present`;
  - if ImGui is drawn, `draw_imgui_to_present` draws and then calls `transition_present_for_present`.
- Current swapchain image usage is `COLOR_ATTACHMENT | TRANSFER_DST`; it does not include `TRANSFER_SRC`.
- Current draw images include `TRANSFER_SRC | STORAGE | COLOR_ATTACHMENT`.
- `vk_debug::capture_and_save_image_view(...)` is lineage only: it assumes `UNDEFINED` old layout, uses unwrap/expect, does not restore layout, submits/waits separately, assumes tightly packed RGBA, and destroys the buffer without releasing the vk_mem allocation correctly.
- `gnome-screenshot` is not viable in the current Wayland session and must not be authoritative proof.

## Architecture Fit

The capture feature belongs below public render hooks. The public hook context intentionally avoids raw Vulkan handles, and that safety property should remain intact. The right ownership boundary is:

- public API: typed capture request/config/status methods;
- example/editor launch: parse capture flags and call the public API;
- renderer backend: schedule capture requests and fulfill them during the frame;
- Vulkan layer: own image layout transitions, copies, readback buffers, PNG writing, and sidecar reporting.

## Current Gaps

- No capture request model exists.
- No multi-frame capture scheduler exists.
- No robust PNG readback helper exists.
- No final-present capture point exists before the terminal present transition.
- No terminal rendergraph pass exists to separate capture from presentation.
- No true headless/offscreen render target path exists.
- No validation helper currently proves PNG dimensions/nonblank/nonuniform output.
- No shared capture parser exists for examples/editor.

## Risks

- Headless/offscreen may be larger than windowed capture because `init_vulkan_core` currently creates a winit surface and swapchain before normal rendering can proceed.
- Present-image readback is risky unless swapchain creation includes `TRANSFER_SRC` and surface capabilities allow it.
- Capturing after `PRESENT_SRC_KHR` adds layout churn and can easily leave incorrect final layout if error handling is wrong.
- Capturing the draw image proves scene rendering but omits present copy and UI.
- Writing PNGs on the render thread can stall; acceptable for debug capture, but continuous capture must be bounded.
- Pixel data may be BGRA, RGBA, or high-precision draw format; readback conversion must be explicit.
- Non-coherent memory requires invalidation before CPU reads if memory is not coherent.
- Manual `F12` capture must not break ImGui/debug UI keyboard capture or existing F1/F2 behavior.

## Documentation Drift To Watch

- `docs/api/08-debug.md` says F1 toggles Debug UI and F2 toggles Console, while current `Renderer::update_input()` toggles console on F1 and debug overlay on F2. This task should avoid expanding that drift, and validators should record it if touched.
- Existing debug docs cover timing capture only, not image capture.

## Recommended Direction

Use a hybrid target architecture:

1. Build the capture scheduler/API and robust Vulkan readback service.
2. Move final present transition into a terminal step/pass so capture can occur after scene/UI rendering and before present.
3. Implement windowed capture as the first executable proof path because it reuses the current render loop.
4. Implement true headless/offscreen by introducing a renderer-owned offscreen present target and a no-present submission path. This is the core target, not an optional stretch.
5. If true headless requires a broad renderer initialization rewrite beyond the bounded target abstraction, stop at the user-decision gate with proof and options.

