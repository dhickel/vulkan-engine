# Implementation Notes

Date: 2026-07-07
Status: shared execution notes

## Phase Order

1. Phase 00: preflight drift repair/quarantine.
2. Phase 01: root `engine` lib facade and frame helpers, no behavior change.
3. Phase 02: renderer `CameraView`/`RenderView` and no-dispatch/no-camera-ownership render path.
4. Phase 03: app-owned input dispatch and action-event emission helper.
5. Phase 04: app-owned event bus/lifecycle stage migration for new path.
6. Phase 05: dogfood migration as integration proof.
7. Phase 06: compatibility labeling, docs/spec/changelog, final evidence.

Do not move to a dependent phase until the prior phase validator passes or records an approved, bounded residual.

## Naming Preference

Prefer `CameraView` unless implementation finds an existing renderer convention that makes `RenderView` clearer. Avoid `RenderCamera` because it can read as renderer-owned camera state.

## Root Facade Modules

Suggested `src/lib.rs` structure:

```rust
pub mod camera;
pub mod events;
pub mod input;
pub mod render;
pub mod runtime;

pub mod prelude {
    pub use crate::camera::*;
    pub use crate::events::*;
    pub use crate::input::*;
    pub use crate::render::*;
    pub use crate::runtime::*;
}
```

Keep `src/main.rs` module declarations compatible with the new lib layout. If `launch`/`runtime` are shared between bin and lib, avoid duplicate module definitions that cause divergent types.

## Renderer API Notes

- Consider extracting the view/projection application from `render_scene_internal` into a helper that accepts `CameraView`.
- Keep legacy `render_scene_internal` behavior by creating a `CameraView` from `self.camera`.
- Add new render-only path that does not call `prepare_frame()` or `prepare_frame_headless()`.
- If the new path must update debug UI frame context, pass input debug data explicitly or use an empty/default debug snapshot for no-dispatch path. Do not read renderer-owned `InputSystem` as if it is app-owned state.

## Input/Event Helper Notes

- Move or duplicate logic from `emit_input_action_events_from_snapshot` into root runtime helper only after adding tests that prove phases.
- The emitter should own `observed_action_values` per app input stream.
- Preserve source label `"input_snapshot"` unless docs intentionally change it.
- Retain changed-value behavior for analog/action values and release pruning.

## Dogfood Notes

- Migrate dogfood in the minimum viable way:
  - construct app-owned input/events/frame clock/camera;
  - use renderer split platform routing for UI/debug/capture;
  - run FPS controller or equivalent against app-owned input snapshot and camera;
  - collision-correct `PlayerState`;
  - build `CameraView` from app camera;
  - call new renderer view path.
- Preserve level loading, collision world, content pack loading, asset setup, render scene construction, and audio semantics where possible.
- Convert `audio_bridge::run_startup_audio_probe` to accept `&mut EventBus` instead of `&mut Renderer` on the migrated path.

## Validation Report Template

Each phase validator should write:

```markdown
# Phase XX Validation Report

Status: passed | failed | blocked
Commit/working tree reference:
Plan criteria checked:
Commands run:
Evidence inspected:
Findings:
Remediation routing:
Residual risks:
```

## Remediation Routing

- `code_defect`: fresh scoped repair worker for the phase unless the miss is trivial and validator-safe.
- `docs_or_evidence_defect`: fresh scoped repair worker unless one-place typo/stale path.
- `browser_harness_defect`: not applicable unless future UI/browser work is added.
- `plan_defect`: return to advanced planning before more implementation.
- `validator_error`: fix checklist or use a fresh validator.
- Same targeted issue failing twice requires fresh escalation repair worker.
