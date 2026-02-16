# Input Package Agent Guide (`src/input`)

Use this guide for input-system maintenance in `src/input`.

## Package Role

`input` provides a frame-buffered layered input stack with:
- priority-group dispatch
- per-event consumption
- action mapping
- snapshot polling
- strict profile serialization/parsing (`version = 1`)

Primary implementation: `src/input/src/lib.rs`.

## Current Contract

- Events are queued during the event pump.
- `InputSystem::dispatch_frame()` is the frame boundary.
- Layers run by descending priority.
- All handlers in the same priority level always run.
- Consumption blocks lower-priority levels only.
- Snapshot transient fields (`just_*`, mouse delta, wheel delta) are frame-scoped.

## Documentation Routing

- API index: `docs/api/00-index.md`
- Internal index: `docs/internal/00-index.md`
- API input model: `docs/api/06-input-polling-and-listeners.md`
- Internal winit integration: `docs/internal/09-input-winit-integration.md`
- Renderer integration points:
  - `src/renderer/src/api/renderer.rs`
  - `src/renderer/src/data/camera.rs`

## Working Rules

- Preserve `dispatch_frame()` as the frame boundary unless intentionally redesigning semantics.
- Preserve priority-group consumption behavior.
- Keep hot path allocation-minimal.
- Add/adjust tests for layer ordering, consumption, and per-frame reset behavior.
- Prefer explicit layer priority bands (engine/UI/gameplay/debug) when introducing new layers.
- If docs and code diverge, treat code as logical truth and record divergence.

## Validation

- `cargo check -p input`
- `cargo test -p input`
- `cargo check -p renderer` (integration sanity)
