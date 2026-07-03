# Phase 03 Validation Report: Apps, Samples, And Docs Consumption

Date: 2026-07-03

Branch: `sprint/alpha-05-event-system-lifecycle`

Status: validator passed, commit/push/report pending

## Scope Validated

- Added app-side event consumption through public `renderer` facade APIs.
- Added a bounded event recorder and selected event logging in the editor startup path.
- Added a bounded event recorder and selected event logging in the dungeon dogfood startup path.
- Added public API docs for event subscriptions, recorders, stages, ordering, mutation safety, and deferred event families.
- Added internal docs for event ownership boundaries, runtime/renderer integration points, ordering, and validation guidance.
- Cross-linked event docs from API/internal indexes, input, runtime launcher, assets, scene, hooks, and API-to-backend handoff docs.

## Files Created Or Changed

| File | Change |
|---|---|
| `apps/editor/src/events.rs` | Added public-facade event recorder/logger helper. |
| `apps/editor/src/main.rs` | Installs editor event logger for windowed and headless renderer startup. |
| `apps/dungeon_dogfood/src/events.rs` | Added public-facade event recorder/logger helper. |
| `apps/dungeon_dogfood/src/main.rs` | Installs dogfood event logger after renderer creation. |
| `docs/api/12-events-and-lifecycle.md` | New public event-system usage and ordering guide. |
| `docs/internal/10-event-system-and-lifecycle.md` | New internal ownership, integration, and validation guide. |
| `docs/api/00-index.md` | Linked events page and top-level reexports. |
| `docs/internal/00-index.md` | Linked internal events page and event crate source. |
| `docs/api/03-scene-graph-and-fragment-workflows.md` | Clarified scene event contract vs deferred broad mutation emission. |
| `docs/api/04-assets-sync-deferred-and-handles.md` | Linked package lifecycle events and deferred per-asset events. |
| `docs/api/05-hooks.md` | Clarified events vs render hooks. |
| `docs/api/06-input-polling-and-listeners.md` | Documented input action event bridge timing. |
| `docs/api/11-runtime-project-launcher.md` | Replaced stale event-system wording and documented runtime lifecycle recording. |
| `docs/internal/04-api-to-backend-handoff.md` | Added event envelope boundary note. |
| `docs/internal/09-input-winit-integration.md` | Added input event bridge timing and event-doc links. |

## Commands

| Command | Result | Notes |
|---|---:|---|
| `cargo fmt --check` | Passed | App event modules formatted. |
| `cargo check -p editor` | Passed | Existing renderer/editor warnings remain. |
| `cargo check -p dungeon_dogfood` | Passed | Existing renderer/dogfood warnings remain. |
| `cargo check -p engine_pack` | Passed | Existing renderer warnings remain. |
| `rg -n "events|lifecycle|EventBus|engine_events" docs/api docs/internal` | Passed | New event docs and cross-links visible. |
| `rg -n "/tmp|TODO|desktop screenshot|playwright|not implemented" docs .internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-05-event-system-lifecycle` | Reviewed | Hits are existing packaging CLI `/tmp` examples, unrelated rendergraph future-direction wording, and sprint directive text; no Phase 03 event-doc stale claim. |
| `rg -n "engine_events" apps docs src/renderer/src/vulkan src/renderer/src/data src/renderer/src/scene src/renderer/src/shaders src/renderer/src/api src/runtime.rs Cargo.toml src/renderer/Cargo.toml` | Passed | App code consumes facade reexports; `engine_events` imports remain at runtime/facade/docs only. |
| `git diff --check` | Passed | No whitespace errors. |
| Phase 03 validator review | Passed | No blocking findings. |

## App Consumption Path

- Editor installs `events::install_editor_event_logger(&mut renderer)` after renderer creation in both windowed and headless paths.
- Dungeon dogfood installs `events::install_dogfood_event_logger(&mut renderer)` after renderer creation.
- Both helpers call `Renderer::set_event_recorder(Some(EventRecorder::bounded(128)))` and subscribe through `Renderer::events_mut()`.
- Listener callbacks receive immutable `EventEnvelope` values and only log selected lifecycle/input observations.

## Deferred Scope Stated In Docs

- Broad scene mutation event emission is deferred.
- Broad per-asset load/ready/failure emission is deferred except root runtime package lifecycle events.
- Live physics, audio, and scripting emission are deferred to later roadmap sprints.
- Dynamic Rust hot reload and scripting runtime execution remain deferred.

## Limitations

- Phase 03 does not add an in-editor event browser or dogfood gameplay event handling.
- No visual capture was required because this phase changes app consumption and docs, not renderer output.
- Existing warning noise remains outside this phase.
- `.idea/engine.iml` and `.reasonix/` remain unrelated local state and must stay out of Phase 03 commits.
- `apps/dungeon_dogfood/src/events.rs` is ignored by current git rules and must be force-added in the Phase 03 commit.

## Validator Handoff

Validate that app code consumes events only through the public renderer facade, editor/dogfood compile, docs accurately distinguish emitted events from typed deferred contracts, and no docs claim desktop screenshot or Playwright validation for this phase.
