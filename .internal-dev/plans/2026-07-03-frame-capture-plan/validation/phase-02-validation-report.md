# Phase 02 Validation Report: Windowed Vulkan PNG Capture

Date: 2026-07-03
Branch: `codex/frame-capture-plan`

## Verdict

Phase 02 is acceptable to commit and push after remediation of the validator findings.

The implementation now records engine-owned PNG captures from the windowed Vulkan present target after ImGui and before the terminal `PRESENT_SRC_KHR` transition. Single capture and finite N-frame capture both write valid PNG files plus JSON sidecars.

This phase intentionally does not implement true headless/offscreen capture. Windowed examples may be interrupted if the user closes the visible window while it is on their active desktop; Phase 03 should remove that operational dependency.

## Remediated Validator Findings

| Finding | Status | Remediation |
|---|---:|---|
| ImGui transitioned the present image before `DebugCapturePass` | Fixed | `draw_imgui_to_present` no longer calls `transition_present_for_present`; `TerminalPresentPass` is now the single terminal present transition. |
| Readback allocation cleanup could leak on finalize errors | Fixed | `finalize_frame_capture` now destroys the readback `VkBuffer` after every success or error result. |
| `.internal-dev/AGENTS.md` is referenced but missing | Residual process gap | Non-blocking for Phase 02 product code; keep as follow-up governance cleanup. |

## Layout And Sync Design

- Default rendergraph order is `PrepareTargetsPass -> SkyboxPass -> GeometryPass -> PresentCopyPass -> ImguiPass -> DebugCapturePass -> TerminalPresentPass`.
- `ImguiPass` leaves the present image in `COLOR_ATTACHMENT_OPTIMAL`.
- `DebugCapturePass` records capture copies into the active frame graphics command buffer.
- Present target capture transitions `COLOR_ATTACHMENT_OPTIMAL -> TRANSFER_SRC_OPTIMAL -> COLOR_ATTACHMENT_OPTIMAL`.
- `TerminalPresentPass` performs the only `COLOR_ATTACHMENT_OPTIMAL -> PRESENT_SRC_KHR` transition after capture.
- Capture readback is finalized after frame submission/present by waiting the submitted frame fence only when pending captures exist.
- Swapchain creation now requires `COLOR_ATTACHMENT | TRANSFER_DST | TRANSFER_SRC`; unsupported surfaces fail initialization instead of attempting illegal present-image readback.

## Command Results

| Command | Result |
|---|---:|
| `cargo fmt` | Pass |
| `git diff --check` | Pass |
| `cargo test -p renderer capture_tests` | Pass, 4 tests |
| `cargo test -p renderer --example api_test parse_capture` | Pass, 2 tests |
| `cargo check` | Pass |
| `cargo check -p renderer` | Pass |
| `cargo check -p renderer --examples` | Pass |
| `cargo check -p input` | Pass |

All compile/test commands emitted the repository's existing warning set.

## Runtime Smoke: Repaired Single Capture

Command:

```sh
RUST_LOG=debug timeout --signal=INT 60s cargo run -p renderer --example api_test -- \
  --capture_frame=30 \
  --capture_frame_path=.internal-dev/debug_reports/frame-capture/phase02-fixed-api_test.png
```

Result:

- Capture was recorded and saved before timeout.
- Process exited by timeout with code 124.
- The earlier double-free was not observed in this repaired run.

Evidence:

```text
.internal-dev/debug_reports/frame-capture/phase02-fixed-api_test.png:
PNG image data, 1920 x 1080, 8-bit/color RGBA, non-interlaced

identify:
phase02-fixed-api_test.png 1920x1080 srgba 4.0 colors=225576
```

Sidecar:

```json
{
  "capture_target": "present",
  "color_conversion": "bgra8-to-rgba8",
  "extent": {
    "height": 1080,
    "width": 1920
  },
  "format": "B8G8R8A8_UNORM",
  "frame_number": 30,
  "png_path": ".internal-dev/debug_reports/frame-capture/phase02-fixed-api_test.png",
  "row_layout": "vkCmdCopyImageToBuffer tightly packed (buffer_row_length=0)",
  "sequence_index": null,
  "source": "Single",
  "status": "succeeded"
}
```

## Runtime Smoke: Repaired Finite Sequence

Command:

```sh
RUST_LOG=info timeout --signal=INT 60s cargo run -p renderer --example api_test -- \
  --capture_frames=3 \
  --capture_frame_start=30 \
  --capture_frame_interval=1 \
  --capture_dir=.internal-dev/debug_reports/frame-capture/phase02-fixed-sequence
```

Result:

- Captures for frames 30, 31, and 32 were recorded and saved before timeout.
- Process exited by timeout with code 124.
- Output directory contains exactly 3 PNG files and 3 JSON sidecars.

Evidence:

```text
png_count=3
json_count=3

renderer-facade-api-test-frame-30-present-seq-0000.png 1920x1080 srgba 4.0 colors=225576
renderer-facade-api-test-frame-31-present-seq-0001.png 1920x1080 srgba 4.0 colors=225576
renderer-facade-api-test-frame-32-present-seq-0002.png 1920x1080 srgba 4.0 colors=225576
```

## Criteria Status

| Criterion | Status | Notes |
|---|---:|---|
| Single windowed capture writes exactly one valid PNG | Pass | Repaired single capture wrote one valid PNG and sidecar. |
| N-frame capture writes exactly N valid PNGs | Pass | Repaired N=3 sequence wrote exactly 3 PNGs and 3 sidecars. |
| Capture from `Present` includes the final post-UI present path | Pass by implementation | Capture pass is ordered after `ImguiPass`; `ImguiPass` no longer performs the terminal present transition. |
| Capture failure logs structured errors without panicking expected paths | Pass by implementation | Capture errors map to `FrameCaptureStatus::Failed`; readback cleanup is now unconditional after finalize. |
| Existing render loop continues after successful capture | Pass with timeout caveat | Captures were saved and the loop continued until the external timeout interrupted the process. |
| Existing `--record_debug` timing capture still works | Pass | Earlier Phase 02 timing evidence wrote 99 rows with `DebugCapturePass`, `TerminalPresentPass`, and `reason=end`. |
| Terminal present layout is correct when windowed presentation proceeds | Pass by implementation | `TerminalPresentPass` owns the final present transition. |

## Residual Risks

- True headless/offscreen capture remains Phase 03+ work and is important because visible windowed runs can interrupt the user's active desktop and may be manually closed.
- Runtime smokes still rely on visible window behavior and external `timeout`; user-driven close events should be treated as environmental interruption unless logs show capture ownership failure.
- `vk_util::transition_image` uses broad synchronization. Acceptable for debug capture, but future video-style capture should use narrower barriers and avoid synchronous readback stalls.
- `.internal-dev/AGENTS.md` is referenced by repo governance but missing.
