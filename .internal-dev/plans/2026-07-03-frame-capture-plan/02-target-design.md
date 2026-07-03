# Target Design

## Design Summary

Frame capture becomes renderer-owned debug infrastructure with four layers:

- `CaptureConfig` and `CaptureRequest` in the public renderer API.
- A scheduler that turns CLI/manual requests into exact frame captures.
- A Vulkan capture service that copies a named engine target to CPU and writes PNG/sidecar output.
- A presentation-target abstraction that supports both windowed swapchain presentation and headless/offscreen capture.

The preferred final frame order is:

```text
PrepareTargets
Skybox
Geometry
PresentCopy or OffscreenPresentCopy
Imgui when available
DebugCapturePass when any capture is due
TerminalPresentPass when windowed presentation is active
Submit
Present only when windowed presentation is active
```

## Public API Contract

Add typed capture APIs to the renderer facade. Names may vary if local naming patterns demand it, but the contract must remain equivalent:

```rust
pub enum CaptureTarget {
    Present,
    Draw,
}

pub struct FrameCaptureRequest {
    pub target: CaptureTarget,
    pub output_path: PathBuf,
    pub sidecar_path: Option<PathBuf>,
}

pub struct FrameCaptureSequence {
    pub target: CaptureTarget,
    pub output_dir: PathBuf,
    pub start_frame: u32,
    pub interval: u32,
    pub remaining: u32,
}

impl Renderer {
    pub fn request_frame_capture(&mut self, request: FrameCaptureRequest) -> Result<(), RendererError>;
    pub fn configure_frame_capture_sequence(&mut self, sequence: FrameCaptureSequence) -> Result<(), RendererError>;
    pub fn last_frame_capture_status(&self) -> Option<&FrameCaptureStatus>;
}
```

Public API must not expose raw `vk::Image`, command buffers, image views, queue handles, allocators, or Vulkan layouts.

## CLI Contract

Shared example parser and editor launch parser should support:

- `--capture_frame=<n>`: schedule one capture at frame `n`.
- `--capture_frame_path=<path>`: exact single-capture PNG path.
- `--capture_frames=<n>`: capture exactly `n` frames.
- `--capture_frame_start=<n>`: first frame for sequence, default `0` or a documented warmup default.
- `--capture_frame_interval=<n>`: sequence interval, default `1`.
- `--capture_dir=<path>`: sequence output directory.
- `--capture_target=present|draw`: default `present` for windowed, `present`/offscreen-present for headless.
- `--headless`: request headless/offscreen renderer mode.
- `--manual_capture_dir=<path>`: optional override for F12 captures.
- `--manual_capture_key=<key>` only if a local parser/key mapping already makes this cheap; otherwise keep `F12` documented and configurable in code later.

Validation must use the exact final names implemented by workers. If names change, docs and validation matrix must be updated in the same phase.

## Output Contract

Single capture:

- supplied `--capture_frame_path` is written exactly once;
- if omitted, default path is `.internal-dev/debug_reports/<app>-frame-<frame>.png`.

N-frame capture:

- exactly N files are written under `--capture_dir`;
- names include app name, captured frame index, target, and sequence number;
- no unbounded loop continues after N captures.

Manual capture:

- default directory is `.internal-dev/debug_reports/manual-captures/`;
- filename includes app/example name, frame index, and timestamp or monotonic sequence.

Sidecar JSON:

- required unless implementation records that JSON support blocks core PNG delivery;
- fields: app/example name, frame index, capture target, PNG path, extent, format, color conversion, result status, error message when failed, and capture timestamp.

## Vulkan Capture Contract

The capture helper must:

- accept an explicit source target descriptor: image, format, extent, current layout, target kind, and expected final layout;
- record layout transitions into the active frame command buffer when possible;
- copy image data to a CPU-visible readback buffer;
- honor row pitch and format swizzle;
- wait for GPU completion through the existing frame/fence model or a clearly bounded immediate path;
- invalidate non-coherent memory when needed;
- write PNG after the GPU has completed the copy;
- restore source image layout for downstream rendering/presentation;
- clean up buffer and allocation through vk_mem-safe paths;
- return structured `Result<FrameCaptureReport, FrameCaptureError>`.

Expected source targets:

- `Present`: final visible color target after ImGui/editor UI, before `PRESENT_SRC_KHR` in windowed mode, and the equivalent offscreen-present image in headless mode.
- `Draw`: offscreen scene color image before present copy/UI, useful as fallback diagnostics but not sufficient as full visual proof when UI is expected.

## Headless/Offscreen Contract

Core target: `--headless` should run without relying on compositor screenshot access. The preferred implementation is:

- create Vulkan instance/device without requiring a winit surface when possible;
- allocate renderer-owned offscreen present-color images with `COLOR_ATTACHMENT | TRANSFER_SRC | TRANSFER_DST`;
- reuse existing draw/depth images and rendergraph passes as much as possible;
- skip swapchain acquire and `queue_present`;
- submit graphics work and use fences for completion;
- allow capture from the offscreen-present target.

Acceptable scoped fallback during implementation:

- if no-surface initialization is too large, workers may implement a hidden/window-backed offscreen-compatible mode only after stopping at the user-decision gate. This fallback must still produce engine-owned PNGs and must be labeled as not true headless.

Headless UI:

- renderer examples do not require ImGui UI in headless proof unless their scene path depends on it;
- editor UI capture in true headless is not required unless implementation finds a robust offscreen ImGui path;
- docs must state whether headless capture includes UI overlays.

## Rendergraph Changes

Split terminal presentation responsibility out of `ImguiPass`.

Planned pass ownership:

- `ImguiPass`: draws UI to the current present/offscreen-present target and leaves it in `COLOR_ATTACHMENT_OPTIMAL`.
- `DebugCapturePass`: if a capture is due, transitions source to transfer layout, records copy, and restores source to the layout required by the next pass.
- `TerminalPresentPass`: windowed only; transitions present image from `COLOR_ATTACHMENT_OPTIMAL` to `PRESENT_SRC_KHR`.

If the codebase resists a new pass, the same ordering may be implemented as an explicit post-render step in `VkRenderCore::render_with_hooks`, but it must be named, documented, and timed clearly.

## Validation Design

Validation is engine-native:

- compile gates;
- focused parser/state tests;
- runtime PNG capture matrix;
- local image metadata and pixel sanity checks;
- evidence index reconciliation.

Desktop screenshots and timing JSONL can be supplementary only.

