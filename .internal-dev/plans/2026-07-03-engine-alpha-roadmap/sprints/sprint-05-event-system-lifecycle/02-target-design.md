# 02 Target Design

## Design Summary

Add a small `engine_events` crate that owns the typed event vocabulary and dispatch/recording mechanics. Renderer/root runtime integrate it at frame-safe boundaries. Apps/tools consume it through facade exports and examples. The core crate remains Vulkan-free and testable with ordinary unit tests.

## Crate Shape

Target path:

- `src/events/Cargo.toml`
- `src/events/src/lib.rs`

Likely package name:

- `engine_events`

Recommended modules:

- `bus`: subscription, dispatch, listener handles, staged drain API.
- `event`: top-level `EngineEvent`, event ids, timestamps/frame metadata if used.
- `families`: lifecycle, input/action, scene, asset, physics, audio, scripting event types.
- `stage`: `EventStage`/`FrameStage` and ordering rules.
- `recording`: bounded `EventRecorder` or `EventLog`.

Keep dependencies light:

- Prefer `std` only for Phase 01.
- Add optional `serde` only if workers need structured event recording output and can justify it.
- Do not depend on `renderer`, `winit`, `imgui`, Vulkan, or app crates.

## Event Vocabulary

Initial typed families:

- Lifecycle: app starting/started, project loaded, scene loading/loaded/saved, frame starting/ended, shutdown requested/completed.
- Input/action: action pressed, released, changed/axis, sourced from post-dispatch snapshot.
- Scene: node created/removed/renamed/transformed, asset placed, material changed.
- Asset: package loading/loaded/failed, asset loading/ready/failed/invalidated.
- Physics: collision enter/stay/exit, trigger enter/exit, query hit.
- Audio: clip started/stopped/finished/failed.
- Scripting: script event emitted, script error.

Use durable string IDs or small newtypes for cross-crate identity where possible. Do not put renderer runtime handles in event payloads as durable identity.

## Dispatch And Ordering

Recommended stage model:

- `Startup`: app/runtime initialization before first project load.
- `ProjectLoad`: project/package validation and load.
- `SceneLoad`: startup scene validation/load.
- `Input`: after input dispatch/snapshot refresh.
- `PreUpdate`: app systems may inspect events before scene mutation.
- `PostUpdate`: app systems have completed frame mutations.
- `Render`: render lifecycle markers only; no app mutation callback.
- `Shutdown`: controlled exit.

Core ordering contract:

- Events are appended in emission order.
- Subscribers receive events in append order within a stage.
- Stage draining is explicit; workers should avoid immediate recursive dispatch unless explicitly tested and documented.
- Event callbacks should not receive `&mut Renderer`; app-level systems can observe and queue commands for later safe execution.

## Renderer Integration

Renderer should:

- Depend on `engine_events`.
- Reexport event types through `renderer::api` and `renderer`.
- Own or expose an `EngineEventBus`/`EventBus` where callers can subscribe without raw internals.
- Provide helpers to bridge input action snapshot to `InputActionEvent` after `InputSystem::dispatch_frame()`.
- Emit asset/load events only at boundaries that are already known, such as package manifest load result and deferred load status transitions if accessible without invasive changes.

Avoid:

- Rewriting render hooks as event hooks.
- Dispatching during Vulkan command submission.
- Adding event dependencies to low-level Vulkan modules.

## Root Runtime Integration

Root runtime should:

- Create/use an event bus for `run_headless` and `run_windowed`.
- Emit app/project/scene/package/frame/shutdown events at documented points.
- Provide enough event recording/debug output to validate lifecycle ordering in tests or debug JSONL.
- Keep current CLI behavior stable unless a narrowly useful `--record_events` option is added and tested.

Windowed input bridge:

- `renderer.update_input` queues raw input.
- `Renderer::render_scene` or explicit runtime frame boundary currently calls input dispatch internally where applicable.
- Emit action events only after the relevant dispatch has refreshed `InputSnapshot`.

## App/Docs Integration

Editor and dogfood should receive minimal consumption examples:

- A subscription that records or logs lifecycle/action events.
- No broad UI or gameplay rewrite.
- Keep app examples demonstrative and low-risk.

Docs should add:

- Public API event lifecycle page.
- Internal event ordering and integration page.
- Cross-links from runtime launcher, input, assets, scene, and hooks docs.

## Validation Evidence Design

Validation summary path:

- `.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-05-event-system-lifecycle/artifacts/validation-summary.json`

Required final evidence:

- Phase validation reports under `validation/`.
- Command output summaries in validation reports.
- Runtime debug timing JSONL under `.internal-dev/debug_reports/sprint-05-event-system-lifecycle/`.
- True headless draw capture PNG/JSON under `.internal-dev/captures/sprint-05-event-system-lifecycle-headless-draw/`.
- Final quality review reconciling the plan, code, tests, docs, and evidence.
