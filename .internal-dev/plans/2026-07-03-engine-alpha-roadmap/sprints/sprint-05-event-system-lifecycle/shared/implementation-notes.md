# Implementation Notes

## Recommended API Sketch

This is illustrative, not a required exact signature:

```rust
pub enum EngineEvent {
    Lifecycle(LifecycleEvent),
    Input(InputEvent),
    Scene(SceneEvent),
    Asset(AssetEvent),
    Physics(PhysicsEvent),
    Audio(AudioEvent),
    Scripting(ScriptingEvent),
}

pub struct EventEnvelope {
    pub sequence: u64,
    pub stage: EventStage,
    pub frame: Option<u64>,
    pub event: EngineEvent,
}

pub struct EventBus {
    // append events, subscribe listeners, drain stage, attach recorder
}
```

## Identity Rules

- Project, package, scene, asset, action, node, script, audio, and physics ids should be durable string/newtype identifiers where possible.
- Runtime handles may appear only in renderer-facing helper payloads when unavoidable and must be documented as non-durable.
- Event ids/sequences should be monotonic within one bus instance.

## Dispatch Rules

- Prefer `emit(...)` to append and `drain_stage(stage)` or `dispatch_pending()` to notify.
- If the bus dispatches immediately, tests must cover recursive emit behavior and listener removal.
- Subscribers should receive `&EventEnvelope` and not `&mut Renderer`.
- Failed listeners should have a defined policy: return an error list, log and continue, or panic-propagate. The policy must be tested and documented.

## Input Bridge Notes

- Use `InputSnapshot` after `dispatch_frame`.
- Emit action `pressed/released/changed` based on snapshot transient fields and current values.
- Preserve action map semantics; the bridge observes snapshots and must not add input layers or consume events unless explicitly needed.

## Runtime Notes

- Root lifecycle events should be emitted in both headless and windowed paths where control flow permits.
- `EventLoop::run` shutdown completion is hard to observe after control flow exit; emit `ShutdownRequested` at `CloseRequested`/Escape and document any limitation.
- Headless path can emit a cleaner start/project/scene/frame/shutdown sequence and should be unit-testable around load helper boundaries where possible.

## Docs Notes

- Add a new public event page instead of burying the contract in hooks/input docs.
- Cross-link event docs from runtime launcher and input docs.
- Internal docs should include an ordering table and mutation safety rules.

## Artifact Notes

- Keep validation reports under this sprint directory.
- Keep debug timing output under `.internal-dev/debug_reports/sprint-05-event-system-lifecycle/`.
- Keep capture proof under `.internal-dev/captures/sprint-05-event-system-lifecycle-headless-draw/`.
