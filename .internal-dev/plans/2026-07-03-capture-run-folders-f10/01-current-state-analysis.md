# Current State Analysis

## Existing Capture Support

- CLI parsing already supports `--headless`, `--capture_frame`, `--capture_frame_path`, `--capture_frames`, `--capture_frame_start`, `--capture_frame_interval`, `--capture_dir`, `--capture_target`, and `--manual_capture_dir`.
- `Renderer` already exposes manual capture methods:
  - `configure_manual_frame_capture_dir(...)`
  - `queue_manual_frame_capture(CaptureTarget)`
- `FrameCaptureScheduler` already queues single, sequence, and manual capture requests.

## Gaps

- Default capture paths still point at `.internal-dev/debug_reports`.
- Default single, sequence, and manual capture paths do not consistently share a run folder.
- No windowed event loop currently binds `F10` to manual capture.
- The default helper `default_single_capture_path(...)` cannot be the only run-folder mechanism because repeated calls could produce distinct timestamped folders.

## Likely Code Surfaces

- `src/renderer/src/api/config.rs`
- `src/renderer/src/api/mod.rs`
- `src/renderer/examples/common/mod.rs`
- `src/renderer/examples/api_test.rs`
- `apps/editor/src/main.rs`
- `apps/editor/src/launch.rs`, only if tests or docs need alignment
