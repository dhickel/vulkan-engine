# Capture Run Folders and F10 Specification Lock

Date: 2026-07-03

## User Requirements

- Default frame captures write under `.internal-dev/captures/`.
- Each renderer/editor run gets its own default capture folder.
- The run folder name includes date/time and differentiates concurrent runs.
- Users should not need to pass output paths for normal single-frame or N-frame capture.
- Existing explicit overrides remain honored:
  - `--capture_frame_path`
  - `--capture_dir`
  - `--manual_capture_dir`
- Windowed mode binds `F10` to queue one manual screenshot capture.
- `F10` must ignore key-repeat spam.
- Headless mode remains available for single-frame and N-frame captures so validation does not hijack the user's screen.

## Locked Defaults

- Default root: `.internal-dev/captures`
- Default run folder format: `<sanitized-app-name>-<YYYYMMDD-HHMMSS-mmm>-pid<PID>`
- Single, sequence, and manual captures share the same run folder by default.
- Explicit paths/dirs override only the mode they configure.

## Non-Goals

- Do not move debug timing reports out of `.internal-dev/debug_reports`.
- Do not refactor Vulkan capture internals unless parent directory creation is missing.
- Do not touch unrelated dirty files such as `.idea/engine.iml` or `.reasonix/`.
