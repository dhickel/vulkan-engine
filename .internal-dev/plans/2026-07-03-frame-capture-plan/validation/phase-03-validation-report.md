# Phase 03 Validation Report: Headless/Offscreen Capture Gate

Date: 2026-07-03
Branch: `codex/frame-capture-plan`
Worker: Foxtrot

## Verdict

Phase 03 is blocked at the required user-decision gate. I did not implement or bless the existing `--headless` path as true headless/offscreen capture.

The current renderer architecture is still window/surface/swapchain centered. A true no-surface, no-swapchain, no-present path requires a broader ownership split than this phase permits: public renderer construction, example control flow, Vulkan instance/device/surface/swapchain setup, queue-family selection, frame-ring present targets, ImGui setup, acquire/submit/present synchronization, resize handling, and rendergraph terminal present behavior are all coupled to the windowed path.

## Evidence

- `src/renderer/src/api/renderer.rs:87-94`: `Renderer::new` requires `&Window` and builds `VkWindowState` from that window even when `config.headless` is true. The current warning says a window and swapchain will be created.
- `src/renderer/examples/api_test.rs:25-51` and `src/renderer/examples/common/mod.rs:413-442`: examples always create a winit `EventLoop` and `WindowBuilder` before constructing the renderer. `--headless` is only copied into `RendererConfig`; it does not select a non-windowed runner.
- `src/renderer/src/vulkan/vk_render.rs:559-628`: `init_vulkan_core` always gets winit-required instance extensions, creates a window surface, selects a physical device with that surface, requires graphics+present queue support, enables swapchain extensions, and creates a swapchain.
- `src/renderer/src/vulkan/vk_render.rs:727-808`: presentation resources are sized from the swapchain image count and present targets are created from swapchain image views through `create_basic_present_views`.
- `src/renderer/src/vulkan/vk_render.rs:811-823`: ImGui platform setup attaches to a winit window and uses the swapchain format/count.
- `src/renderer/src/vulkan/vk_render.rs:2038-2148`: frame acquisition always calls `acquire_next_image2` on the swapchain and binds the acquired swapchain image as the current present target.
- `src/renderer/src/vulkan/vk_render.rs:2191-2229`: submit always waits on the acquire semaphore and `present_frame` always calls `queue_present`.
- `src/renderer/src/rendergraph/mod.rs:66-75` and `src/renderer/src/rendergraph/passes/terminal_present_pass.rs:12-14`: the default graph always includes `TerminalPresentPass`, which transitions the present image for swapchain presentation.

## Gate Decision Needed

Choose one option before Phase 03 implementation continues:

1. Approve a true headless architecture phase.
   - Add an explicit renderer surface mode, for example `Windowed` vs `HeadlessOffscreen`.
   - Add a renderer construction path that does not require `&Window`.
   - Add no-surface instance/device initialization and queue selection.
   - Allocate renderer-owned offscreen present-equivalent images and views.
   - Make frame acquire bind the current offscreen target instead of acquiring a swapchain image.
   - Submit without acquire/present semaphores in headless mode.
   - Skip `queue_present` and skip/replace `TerminalPresentPass` in headless mode.
   - Decide whether headless ImGui/UI is unsupported, disabled, or initialized through a separate offscreen-safe path.

2. Approve a labeled hidden/window-backed fallback.
   - Keep a window/surface/swapchain but minimize visible disruption where the platform allows it.
   - Continue using engine-owned PNG readback from the render target.
   - Label all validation as fallback/window-backed, not true headless.
   - This does not satisfy the locked true headless/offscreen target without explicit user approval.

3. Split the work.
   - First phase: make `--headless` fail fast with a clear unsupported error instead of opening a windowed/swapchain run.
   - Second phase: implement the true no-surface/offscreen architecture above.

## Validation Commands

No runtime headless smoke was run. Running the requested command with the current code would create a visible winit window and swapchain, which the directive explicitly forbids treating as true headless proof.

No compile gates were run because this pass made no product-code changes. The only changed artifact is this gate report.

## Criteria Status

| Criterion | Status | Notes |
|---|---:|---|
| `--headless` capture writes a valid PNG through engine-owned capture | Blocked | Current `--headless` still uses a window/surface/swapchain path. |
| Headless mode does not require desktop screenshot access | Blocked | Windowed PNG readback exists from Phase 02, but true headless/offscreen ownership is not present. |
| Headless capture can run under `timeout --signal=INT 60s` | Blocked | Would currently be a visible windowed run. |
| Windowed capture from Phase 02 still works | Preserved | No product code changed in this phase. |
| Full canonical headless matrix passing or blocked with explicit gate evidence | Gate report complete | This report records the blocker and concrete options. |

## PNG Evidence

No Phase 03 PNG evidence was produced. Phase 02 already validated windowed PNG capture; reusing that path here would violate the Phase 03 negative check against hidden/windowed fallback as true headless.

## Residual Risks

- The current `--headless` flag is misleading: it is parsed and passed into `RendererConfig`, but the renderer still constructs a window, surface, swapchain, acquires swapchain images, and presents. Until the user chooses an option above, workers should not use `--headless` as validation proof.
- Implementing true headless will likely affect public API shape, example loop structure, Vulkan core initialization, frame sync semantics, rendergraph pass selection, and documentation. It should be planned as a bounded architecture phase rather than patched opportunistically.
- `.internal-dev/AGENTS.md` remains a previously recorded process gap from Phase 02; I did not touch it in this scoped worker pass.
