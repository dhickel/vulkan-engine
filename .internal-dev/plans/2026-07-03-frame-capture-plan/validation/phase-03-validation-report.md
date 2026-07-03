# Phase 03 Validation Report: True Headless/Offscreen Capture

Date: 2026-07-03
Branch: `codex/frame-capture-plan`
Worker: Golf

## Verdict

Phase 03 is implemented and validated.

The renderer now has a true headless/offscreen path that does not create a winit `Window`, Vulkan surface, swapchain, or presentation operation. It creates renderer-owned offscreen present-equivalent images and maps `CaptureTarget::Present` to those images. The required smoke command writes three valid engine-owned PNGs and exits cleanly.

## Implementation Evidence

- `src/renderer/src/vulkan/vk_types.rs`: added `RenderSurfaceMode::{Windowed, HeadlessOffscreen}`, frame-owned offscreen present image storage, shared command-pool destroy deduplication, headless-safe present-view ownership, and transfer host-buffer registration.
- `src/renderer/src/vulkan/vk_init.rs`: added no-surface queue-family selection and offscreen present image allocation using `COLOR_ATTACHMENT | TRANSFER_SRC | TRANSFER_DST`.
- `src/renderer/src/vulkan/vk_render.rs`: added `VkRender::new_headless` and `VkRenderCore::new_headless`, no-surface/no-swapchain initialization, mode-aware acquire/submit/present, optional ImGui/surface/swapchain ownership, capture readback using the configured present format, and explicit data-cache/allocator teardown before device destruction.
- `src/renderer/src/api/renderer.rs`: added `Renderer::new_headless` and `render_scene_headless`.
- `src/renderer/examples/api_test.rs` and `src/renderer/examples/common/mod.rs`: branch to direct headless runners before creating any `EventLoop` or `Window`.
- `src/renderer/src/rendergraph/passes/terminal_present_pass.rs`: terminal present pass is a no-op for headless/offscreen mode.
- `src/renderer/src/data/data_cache.rs`: added cleanup for environment cache resources, scene-adjacent cached resources, joint descriptor pool, and default joint buffer.

## Required Headless Smoke

Command:

```bash
RUST_LOG=info timeout --signal=INT 60s cargo run -p renderer --example api_test -- --headless --capture_frames=3 --capture_dir=.internal-dev/debug_reports/frame-capture/headless/api_test
```

Result: pass, exit code 0.

The run recorded and saved all three requested frame captures, emitted `Headless capture completed: 3 capture(s) written`, and exited without the prior VMA allocation assertion or teardown segfault.

PNG evidence:

```text
.internal-dev/debug_reports/frame-capture/headless/api_test/renderer-facade-api-test-frame-0-present-seq-0000.png
.internal-dev/debug_reports/frame-capture/headless/api_test/renderer-facade-api-test-frame-1-present-seq-0001.png
.internal-dev/debug_reports/frame-capture/headless/api_test/renderer-facade-api-test-frame-2-present-seq-0002.png
```

`identify -format '%f %wx%h %[channels] colors=%k\n'` reported:

```text
renderer-facade-api-test-frame-0-present-seq-0000.png 1920x1080 srgba 4.0 colors=225576
renderer-facade-api-test-frame-1-present-seq-0001.png 1920x1080 srgba 4.0 colors=225576
renderer-facade-api-test-frame-2-present-seq-0002.png 1920x1080 srgba 4.0 colors=225576
```

## Teardown Root Cause Resolved

The first true-headless implementation wrote valid PNGs but failed during shutdown. The final fix set included:

- Register mesh and texture staging host buffers with `VkTransfer`, so the advertised transfer teardown path actually releases both VMA-backed host buffers.
- Destroy the mesh cache default joint buffer and joint descriptor pool.
- Destroy environment cache cubemaps and generated irradiance/prefilter maps.
- Avoid double-destroying headless present image views that are owned by the offscreen `VkImageAlloc`.
- Drop the data cache and VMA allocator before `vkDestroyDevice`, preventing allocator destruction after the logical device is gone.

Temporary VMA allocation counters showed the final teardown reached `allocations=0 bytes=0` after BRDF cleanup; those counters were removed before final validation.

## Criteria Status

| Criterion | Status | Notes |
|---|---:|---|
| Explicit renderer surface mode | Pass | `RenderSurfaceMode` selects windowed vs headless/offscreen behavior. |
| Headless construction without winit `Window` | Pass | Headless examples construct the renderer before any event-loop/window path. |
| No-surface/no-swapchain initialization | Pass | Headless initialization omits surface creation, swapchain extension setup, and swapchain creation. |
| Offscreen present-equivalent images | Pass | Frame-owned images are created with color attachment and transfer usage. |
| Headless acquire skips `acquire_next_image2` | Pass | Headless frame acquisition selects the current offscreen frame slot. |
| Headless submit skips swapchain acquire wait | Pass | Headless submit uses the graphics command buffer without acquire/present semaphore synchronization. |
| Headless present skips `queue_present` | Pass | `present_frame` returns immediately in headless mode. |
| Terminal present pass mode-aware | Pass | No-op in headless mode. |
| `CaptureTarget::Present` maps to headless offscreen target | Pass | Captures read the headless present target. |
| Requested headless smoke cleanly passes | Pass | Three PNGs written and process exited 0. |
| Full headless matrix | Partial | `api_test` headless path validated; other examples remain follow-up matrix expansion. |
| Windowed Phase 02 runtime behavior | Compile-preserved | Windowed path compiles; runtime smoke was not rerun in this continuation. |

## Validation Commands

```bash
cargo fmt
```

Result: pass.

```bash
git diff --check
```

Result: pass.

```bash
cargo test -p renderer capture_tests
```

Result: pass, 4 tests.

```bash
cargo test -p renderer --example api_test parse_capture
```

Result: pass, 2 tests.

```bash
cargo check
```

Result: pass.

```bash
cargo check -p renderer
```

Result: pass with existing renderer warnings.

```bash
cargo check -p renderer --examples
```

Result: pass with existing renderer warnings.

```bash
cargo check -p input
```

Result: pass.

```bash
RUST_LOG=info timeout --signal=INT 60s cargo run -p renderer --example api_test -- --headless --capture_frames=3 --capture_dir=.internal-dev/debug_reports/frame-capture/headless/api_test
```

Result: pass, three captures written.

## Residual Risks

- Headless ImGui is disabled in this path; headless captures are scene/present-equivalent captures without UI overlays.
- The full canonical example matrix is not yet automated for headless capture; `api_test` is the validated environment in this phase.
- Windowed Phase 02 capture was compile-preserved but not runtime revalidated in this continuation.
- `.idea/engine.iml` and `.reasonix/` remain unrelated local changes and were not touched.
