# 07 - Error Model and Recovery Patterns

This chapter maps facade errors to practical recovery actions.

## Top-level Error Envelope

`RendererError` variants:
- `Init(RendererInitError)`
- `Frame(RendererFrameError)`
- `Scene(SceneError)`
- `Asset(AssetError)`
- `Hook(HookError)`
- `Unsupported(&'static str)`
- `InvalidState(&'static str)`

Pattern:
- Most app code should match on `RendererError` at subsystem boundaries, then downcast behavior per variant.

## Init Errors

`RendererInitError`:
- `Vulkan(String)`
- `Window(String)`
- `ShaderCompile(String)`
- `StartupScene(String)`

Recovery ideas:
- Vulkan: show requirements, disable startup path, exit gracefully.
- Shader compile: advise installing `glslc`/`glslangValidator` if compile mode enabled.

## Frame Errors

`RendererFrameError`:
- `Input(String)`
- `Resize(String)`
- `Render(String)`
- `FrameContext(String)`

Recovery ideas:
- `Input`: log and continue unless persistent.
- `Resize`: retry next resize event.
- `Render`: usually fatal for session; consider shutdown.
- `FrameContext`: integration bug; fix event/render ordering.

## Scene Errors

`SceneError`:
- `InvalidNode`
- `StaleNode`
- `CycleDetected`
- `InvalidParent`
- `MergeFailed(String)`

Recovery ideas:
- Invalid/stale: rebuild references from authoritative scene state.
- Merge/cycle: reject content and keep prior scene intact.

## Asset Errors

High-frequency variants:
- Path/data: `Load`, `Io`, `Decode`
- Handles: `InvalidHandle`, `StaleHandle`, `NotLoaded`, `OutOfBounds`, `ReservedHandle`
- Deferred: `UnknownTicket`, `CancelRejected`
- Infra: `Cache`, `Sync`, `Unsupported`, `Internal`

Recovery ideas:
- Path/data: fallback asset/material and continue.
- Handles: refresh handle source; do not retry blindly.
- Deferred ticket errors: reset request and issue new ticket.

## Hook Errors

`HookError`:
- `Unsupported`
- `Registration`
- `Invocation`

Current behavior:
- Hook failures are wrapped/logged and do not fail frame by default.

## Learn More

- Error definitions: `src/renderer/src/api/errors.rs`
- Hook semantics: `06_render_hooks_and_extension_points.md`
