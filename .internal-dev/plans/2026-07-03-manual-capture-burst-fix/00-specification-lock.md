# Manual Capture Burst Fix Specification Lock

Date: 2026-07-03

## Objective

Fix repeated manual capture requests so rapid F10/button presses schedule one capture per successive future frame instead of piling all requests onto the same next frame.

## Acceptance Criteria

- Three manual captures queued at current frame `0` become due on frames `1`, `2`, and `3`.
- Manual capture filenames remain unique.
- Repeated manual captures share the same manual run directory.
- Single capture, sequence capture, and explicit manual output directory behavior remain unchanged.
- No Vulkan/backend refactor.
- No broad public API change.
- Focused scheduler tests cover the burst behavior.

## Negative Criteria

- Do not change F10/F12 key handling.
- Do not change capture path naming except the frame number reflecting the scheduled frame.
- Do not change `due_captures` backend semantics unless tests prove it is required.
- Do not solve `last_frame_capture_status()` history in this bug fix unless validation proves it blocks the fix.

## Root Cause

`Renderer::queue_manual_frame_capture()` passes the current renderer frame to `FrameCaptureScheduler::queue_manual_capture()`. The scheduler currently schedules every manual request at `current_frame + 1`, so multiple requests before the next render all become due on one frame.

## Residual Risk

`last_frame_capture_status()` still exposes only one status. The scheduler fix reduces same-frame bursts, but status history remains a possible follow-up if UI/status reporting needs to show every capture completion.
