# Phase 04 Worker Report: App-Owned Event Bus And Lifecycle Stages

Date: 2026-07-07
Status: implemented, awaiting validator review

## Summary

Added lightweight root event helpers over caller-owned `EventBus`:

- `engine::events::runtime_event_bus()`
- `engine::events::runtime_event_bus_with_recorder_capacity(...)`
- `engine::events::RuntimeEventDispatcher`

The dispatcher emits/drains lifecycle stages without owning or hiding the event bus. This keeps one app-owned bus available for lifecycle, input, audio, physics, scripting, and diagnostics while preserving raw `engine_events` access.

## Changed Files

- `src/events.rs`
- `src/lib.rs`
- `src/runtime.rs`
- `tests/runtime_event_dispatcher.rs`
- `docs/api/12-events-and-lifecycle.md`
- `docs/internal/10-event-system-and-lifecycle.md`
- `docs/api/02-renderer-lifecycle-and-frame-api.md`
- `docs/api/06-input-polling-and-listeners.md`
- `.internal-dev/specifications/architecture.md`
- `.internal-dev/specifications/service-graph.md`
- `.internal-dev/specifications/services.md`
- `.internal-dev/specifications/api.md`
- `.internal-dev/specifications/decisions.md`
- `.internal-dev/changelogs/2026-07-07-engine-runtime-abstractions-phase-04-app-owned-events.md`

## Contract Notes

- New app-owned path: root/app code emits `FrameStarted` and `FrameEnded` through `RuntimeEventDispatcher` on its own `EventBus`.
- Renderer no-dispatch/view rendering remains lifecycle-silent for app events.
- Legacy renderer-owned bus and `Renderer::events()` / `events_mut()` remain unchanged compatibility APIs.
- `engine_events` remains independent; no renderer/windowing dependency was added.
- Dogfood was not migrated in this phase.

## Validation

Passed:

- `cargo fmt --check`
- `cargo check -p engine_events`
- `cargo test -p engine_events`
- `cargo test -p engine`
- `cargo check -p renderer`
- `cargo test -p renderer`
- `rg -n "EventBus|FrameStarted|FrameEnded|events_mut\\(|drain_stage|dispatch_pending|RuntimeEventDispatcher|runtime_event_bus" src apps tests`

Observed existing warning noise:

- Renderer dead-code warnings remain present and unrelated to this phase.

## Residuals

- Dogfood still consumes renderer-owned event compatibility paths until Phase 05.
- Renderer legacy frame APIs still emit renderer-owned lifecycle events by design.
