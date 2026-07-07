# Phase 04 Validation Report: App-Owned Event Bus And Lifecycle Stages

Date: 2026-07-07
Validator: Codex validation agent
Result: PASS

## Findings

No blocking findings.

Non-blocking residual: dogfood remains on the legacy renderer-owned event/input/frame path, which is explicitly in scope for Phase 05 rather than Phase 04. Evidence: `apps/dungeon_dogfood/src/main.rs:309` still calls `renderer.update_input(...)`, `apps/dungeon_dogfood/src/main.rs:537` / `apps/dungeon_dogfood/src/main.rs:573` still use `begin_frame(...)` / `end_frame(...)`, and `apps/dungeon_dogfood/src/audio_bridge.rs:75` still passes `renderer.events_mut()` into the audio probe.

## Criteria Results

| Criterion | Result | Evidence |
| --- | --- | --- |
| `RuntimeEventDispatcher` and `runtime_event_bus` operate on caller-owned `EventBus` and do not hide raw primitives. | PASS | `src/events.rs:19` returns a raw `EventBus`; `src/events.rs:24` uses `EventBus::with_recorder(...)`; dispatcher methods take `&mut EventBus` at `src/events.rs:37`, `src/events.rs:48`, `src/events.rs:58`, `src/events.rs:68`, and `src/events.rs:73`. Raw event primitives remain re-exported at `src/events.rs:3`. |
| Event sequence ordering is monotonic in one bus. | PASS | `EventBus::emit` increments one per-bus `next_sequence` at `src/events/src/lib.rs:523` through `src/events/src/lib.rs:530`. Root coverage asserts sequence `[0, 1, 2, 3]` across one bus at `tests/runtime_event_dispatcher.rs:27` through `tests/runtime_event_dispatcher.rs:75`; engine_events unit coverage also includes monotonic emission. |
| Helper does not duplicate `FrameStarted`/`FrameEnded` for one app frame. | PASS | `RuntimeEventDispatcher::frame_started` emits one `FrameStarted` at `src/events.rs:58`; `frame_ended` emits one `FrameEnded` at `src/events.rs:73`. Test `runtime_frame_helpers_emit_one_start_and_end_for_frame` asserts exactly one start/end pair and no pending events at `tests/runtime_event_dispatcher.rs:78` through `tests/runtime_event_dispatcher.rs:104`. |
| Listener failures remain collected and non-aborting. | PASS | Event bus dispatch tests pass in `engine_events`; root helper coverage at `tests/runtime_event_dispatcher.rs:107` through `tests/runtime_event_dispatcher.rs:133` confirms a failing listener is collected while a later listener still observes the event. |
| Renderer no-dispatch/view path remains app-lifecycle-silent; legacy renderer-owned `events()`/`events_mut()` remain available. | PASS | `Renderer::events()` / `events_mut()` remain public at `src/renderer/src/api/renderer.rs:302` and `src/renderer/src/api/renderer.rs:306`. `render_scene_with_view` at `src/renderer/src/api/renderer.rs:576` through `src/renderer/src/api/renderer.rs:596` renders without lifecycle emission/drain calls. Legacy lifecycle emission remains in `execute_frame_lifecycle` at `src/renderer/src/api/renderer.rs:620` through `src/renderer/src/api/renderer.rs:654` and explicit `begin_frame` / `end_frame` at `src/renderer/src/api/renderer.rs:658` through `src/renderer/src/api/renderer.rs:725`. |
| `engine_events` remains independent from renderer/windowing/root engine. | PASS | `src/events/Cargo.toml` has only `serde_json` as a normal dependency; `cargo check -p engine_events` and `cargo test -p engine_events` passed. Spec boundary also states this at `.internal-dev/specifications/service-graph.md:18`. |
| Dogfood is not migrated in this phase. | PASS | Active dogfood still uses legacy renderer path: `renderer.update_input(...)` at `apps/dungeon_dogfood/src/main.rs:309`, `begin_frame(...)` / `end_frame(...)` at `apps/dungeon_dogfood/src/main.rs:537` and `apps/dungeon_dogfood/src/main.rs:573`, and `renderer.events_mut()` at `apps/dungeon_dogfood/src/audio_bridge.rs:75`. |
| Docs/spec/changelog accurately reflect behavior without overstating dogfood migration. | PASS | Public docs identify app-owned helper behavior and legacy renderer compatibility at `docs/api/12-events-and-lifecycle.md:29` through `docs/api/12-events-and-lifecycle.md:41`; app-owned ordering states `render_scene_with_view` is lifecycle-silent at `docs/api/12-events-and-lifecycle.md:117` through `docs/api/12-events-and-lifecycle.md:127`. Specs record dogfood as not migrated in Phase 04 at `.internal-dev/specifications/architecture.md:18` and `.internal-dev/specifications/decisions.md:17`. Changelog records the same residual at `.internal-dev/changelogs/2026-07-07-engine-runtime-abstractions-phase-04-app-owned-events.md:21` through `.internal-dev/changelogs/2026-07-07-engine-runtime-abstractions-phase-04-app-owned-events.md:25`. |

## Commands And Results

| Command | Result | Notes |
| --- | --- | --- |
| `cargo fmt --check` | PASS | No formatting diffs reported. |
| `cargo check -p engine_events` | PASS | Finished successfully. |
| `cargo test -p engine_events` | PASS | 18 unit tests passed; 1 doctest ignored by design. |
| `cargo test -p engine` | PASS | Root unit/integration tests passed, including `tests/runtime_event_dispatcher.rs`. Existing renderer dead-code warning noise was present. |
| `cargo check -p renderer` | PASS | Finished successfully with existing renderer dead-code warnings. |
| `cargo test -p renderer` | PASS | 167 unit tests and 21 integration tests passed; 5 renderer doctests ignored by design. Existing renderer dead-code warning noise was present. |
| `rg -n "EventBus|FrameStarted|FrameEnded|events_mut\\(|drain_stage|dispatch_pending" src apps tests` | PASS | Inspected call sites. Results match expected root helper, raw event bus, legacy renderer bus, and dogfood legacy-path usage. |

## Governance And Scope Review

- Read repo-level instructions from `AGENTS.md` as supplied in the prompt.
- Read `.internal-dev/specifications/AGENTS.md`, `src/events/AGENTS.md`, `.internal-dev/specifications/services.md`, `.internal-dev/specifications/service-graph.md`, and `02-target-design.md`.
- Read the Phase 04 directive and worker report.
- Listed `.internal-dev/knowledge/`; no domain-specific knowledge file was required for this event-helper validation.
- Ownership boundaries were respected: root helper code depends on `engine_events`; `engine_events` does not depend on renderer/windowing/root engine; renderer keeps legacy event APIs and does not depend on root `engine`.

## Browser / Visual Proof

Not applicable. This phase changes Rust event/lifecycle ownership contracts and tests, not browser UI or visible rendering behavior.

## Required Remediation

None.
