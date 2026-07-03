# Senior Engineer Guidance

## Core Principles

- The event crate is a contract crate, not an implementation dumping ground. Keep it small, typed, and renderer-free so tests are cheap and reliable.
- Ordering must be explicit because events are only useful if apps can reason about when they are safe to react.
- Prefer app systems queuing later commands over callbacks mutating renderer/scene state during emission.
- Emit events after successful state changes, not before. For failures, emit failure events only where the failure boundary is caught and meaningful.
- Keep placeholder event families honest. Physics/audio/scripting can have types and unit tests, but real emission belongs to later sprints.

## Direct Targets

- `src/events/`: owns event bus, families, stage/order semantics, recorder, tests.
- `src/renderer/src/api/mod.rs` and `src/renderer/src/lib.rs`: public reexports only.
- `src/renderer/src/api/renderer.rs`: facade bus access and input/action bridge helpers where appropriate.
- `src/runtime.rs`: root lifecycle/project/scene/package/frame/shutdown emission.
- `apps/editor/` and `apps/dungeon_dogfood/`: minimal subscription/recording examples.
- `docs/api/` and `docs/internal/`: event contract and internal ordering docs.

## Gotchas

- `InputSystem::dispatch_frame()` is the frame boundary. Emitting action events from raw queued winit input will be wrong.
- `Renderer::assets()` returns an asset manager tied to mutable renderer internals. Avoid holding it while dispatching arbitrary subscriber callbacks.
- Headless runtime capture must use `--capture_target draw`; present-target proof is not sufficient for this sprint.
- Event callbacks that can panic should either be isolated by the bus contract or explicitly documented as propagating. Pick one and test it.
- Recursive dispatch can make ordering hard to reason about. Avoid it for alpha unless there is a clear queue/drain rule.
- Avoid event payloads that force app crates to depend on renderer-specific handle types unless the event is explicitly renderer-facing.

## Best Practices

- Make event structs easy to construct in tests.
- Use typed enums/newtypes for family/stage/kind fields instead of stringly typed event names where practical.
- Keep subscription handles removable and test listener removal.
- Add tests for deterministic ordering across stages.
- If event recorder has a limit, test truncation/eviction behavior.
- Document every deferred system family so docs do not imply runtime support that does not exist.

## Likely Failure Modes

- Compile-cycle dependency: `renderer -> engine_events -> renderer`. Avoid by keeping `engine_events` independent.
- Borrow checker pressure: dispatching callbacks while a mutable renderer borrow is active. Solve by recording events then draining after the borrow ends.
- Double lifecycle events: root runtime and renderer both emitting the same event. Assign one owner per boundary.
- Docs drift: public docs claiming physics/audio/script events emit today. Say "event types reserved; real emission deferred."
- Validation false positive: compile passes but draw capture fails. Treat capture failure as blocking unless tool/environment is proven unavailable and recorded.
