# Senior Engineer Guidance

Date: 2026-07-07
Status: shared guidance for all phases

## Ground Rules

- Treat this as an ownership refactor, not a framework build.
- Make the new correct path mechanically impossible to confuse with legacy renderer-owned lifecycle.
- Keep dependency direction obvious: root/app uses support crates; support crates never use root/app.
- Prefer narrow public types over large context objects.
- Preserve legacy renderer methods until closeout explicitly labels them compatibility.

## Facts And Reasoning Cues

- `InputSystem::dispatch_frame()` is the frame boundary. Calling it twice in one new app frame erases or corrupts transient semantics.
- The current action-event bridge uses a private `observed_action_values` map. Moving/duplicating this carelessly can produce duplicate or missing `Changed` events.
- Renderer platform handling and app input routing are currently intertwined. Split by responsibility, not by event enum branch.
- ImGui/debug UI capture affects whether gameplay input should be queued; it does not mean renderer should own gameplay input forever.
- `CameraView`/`RenderView` must live in renderer/lower because renderer consumes it and cannot depend on root `engine`.
- App camera ownership is not proven until the render submission uses caller-provided matrices/position and dogfood stops pulling/writing renderer camera state.
- Event bus ownership is not proven until dogfood audio and app lifecycle events use caller-owned `EventBus`.
- Root `engine` facade should make the common path easy, but raw crates are still first-class.

## Likely Failure Modes

- New root facade calls old `Renderer::render_scene` or `begin_frame`, silently keeping renderer-owned input/camera/event behavior.
- Renderer helper returns a capture decision but legacy `update_input` and new app path both queue the same event.
- Input tests miss same-frame press/release or mouse delta reset behavior.
- Dogfood compiles by leaving `renderer.events_mut()` or camera pull/write-back in a helper that remains on the active path.
- Headless path accidentally takes `&Window` because the split started from windowed helper signatures.
- Docs update describes the desired target but leaves active specs saying renderer owns camera state.

## Implementation Style

- Keep edits close to crate boundaries: renderer API changes in `src/renderer/src/api/`, root facade helpers in root `src/`, dogfood migration in `apps/dungeon_dogfood/`.
- Add compatibility doc comments where legacy methods remain.
- Add tests near the code that owns behavior. Root runtime helpers should have root crate tests; input semantics stay in `src/input`; renderer view-path tests stay in `renderer`.
- Avoid naming that implies a monolith. Prefer `FrameClock`, `FrameInfo`, `InputActionEventEmitter`, `RuntimeParts`, `CameraView`.
- Avoid exposing backend internals in facade modules. `advanced-interop` remains the existing explicit unsafe/unstable route.

## Evidence Discipline

- Every phase produces a validation report under this plan directory.
- Runtime/debug artifacts go under task-scoped paths:
  - `.internal-dev/debug_reports/engine-runtime-abstractions-issues-35-37/`
  - `.internal-dev/captures/engine-runtime-abstractions-issues-35-37/`
  - `artifacts/engine-runtime-abstractions-issues-35-37/`
- Evidence index status must not say `fully_validated` until all phase validators, final quality review, stale-reference sweep, and required runtime smokes pass.
