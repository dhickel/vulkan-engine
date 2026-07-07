# Date
2026-07-07

# Change Summary
Implemented Phase 04 app-owned event lifecycle support for the engine runtime abstractions plan. Added root `engine::events` helpers for constructing a recorded caller-owned `EventBus` and emitting/draining lifecycle stages through `RuntimeEventDispatcher`.

# Files
- `src/events.rs`: added `runtime_event_bus`, configurable recorder-capacity constructor, and `RuntimeEventDispatcher` helpers for lifecycle/input stage dispatch.
- `src/lib.rs`: re-exported app-owned event helpers through the root prelude.
- `src/runtime.rs`: routed the private runtime event shim through the new root event helper without changing launcher behavior.
- `tests/runtime_event_dispatcher.rs`: covered one-bus monotonic sequence, single frame-start/frame-end helper emission, and listener failure collection/continuation.
- `docs/api/12-events-and-lifecycle.md`, `docs/internal/10-event-system-and-lifecycle.md`, `docs/api/02-renderer-lifecycle-and-frame-api.md`, `docs/api/06-input-polling-and-listeners.md`: documented app-owned lifecycle ownership and renderer legacy compatibility.
- `.internal-dev/specifications/architecture.md`, `.internal-dev/specifications/service-graph.md`, `.internal-dev/specifications/services.md`, `.internal-dev/specifications/api.md`, `.internal-dev/specifications/decisions.md`: recorded app-owned event lifecycle contracts and decision.

# Behavioral Impact
Apps can now keep one caller-owned `EventBus` for lifecycle, input, and subsystem events while still using raw `engine_events` primitives directly. Renderer-owned `events()` and `events_mut()` remain available for legacy renderer frame/input paths.

# Specification Impact
Updated architecture, service graph, service, API, and decision specifications to record root app-owned lifecycle dispatch as intended truth while preserving renderer-owned lifecycle events as compatibility behavior.

# Risks
Dogfood still uses the legacy renderer-owned event path by design for this phase. The new app-owned event helper is covered by root tests but is not yet wired into a migrated active app loop.

# Follow-up Items
- Phase 05 should migrate dogfood to the app-owned input/event/camera/render path without duplicating frame lifecycle events.
- Final cleanup should revisit docs once renderer legacy lifecycle APIs are no longer the primary beginner path.
