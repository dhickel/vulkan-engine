# Phase 03 Validator Review: Apps, Samples, And Docs Consumption

Date: 2026-07-03

Branch: `sprint/alpha-05-event-system-lifecycle`

Status: PASS

## Findings

No blocking findings.

Closeout note: `apps/dungeon_dogfood/src/events.rs` is ignored by the current git rules (`git status --short --ignored ...` reports `!! apps/dungeon_dogfood/src/events.rs`). The working tree implementation validates, but the main thread must force-add that file if this phase is accepted. `.idea/engine.iml` and `.reasonix/` are unrelated local state and remain excluded from this phase.

## Criterion Results

| Criterion | Result | Evidence |
|---|---:|---|
| 1. At least one app path, preferably both editor and dogfood, demonstrates event subscription/recording through public renderer facade APIs only. | PASS | Both `apps/editor/src/events.rs` and `apps/dungeon_dogfood/src/events.rs` import from `renderer::{...}` and call `Renderer::set_event_recorder(...)` plus `Renderer::events_mut().subscribe(...)`. Editor installs the helper in both windowed and headless startup paths; dogfood installs it after renderer creation. |
| 2. App code does not reach into renderer private modules or depend directly on `engine_events`. | PASS | App ownership scan found no app `engine_events` import. The only direct `engine_events` hits are workspace dependencies, root runtime, renderer facade/API implementation, and docs. |
| 3. Listener callbacks receive immutable `EventEnvelope` and do not mutate renderer/scene or change gameplay/editor behavior. | PASS | App callbacks call logging helpers only; helpers accept `&EventEnvelope`, match lifecycle/input events, and emit `log::debug!` messages. No renderer, scene, UI, collision, or gameplay mutation is performed from callbacks. |
| 4. `editor`, `dungeon_dogfood`, and `engine_pack` compile. | PASS | `cargo check -p editor`, `cargo check -p dungeon_dogfood`, and `cargo check -p engine_pack` all passed with existing warning noise only. |
| 5. Public docs explain facade imports, event families, subscription/recorder examples, frame stage/order table, input action bridge timing, mutation safety rules, and deferred physics/audio/scripting behavior. | PASS | `docs/api/12-events-and-lifecycle.md` covers facade reexports, event families, recorder/subscription example, ordering tables, input bridge timing, mutation safety, and deferred family status. |
| 6. Internal docs explain ownership boundaries, runtime/renderer integration points, ordering rules, and validation guidance. | PASS | `docs/internal/10-event-system-and-lifecycle.md` documents the boundary map, facade/runtime integration points, ordering, ownership rules, and validation command guidance. |
| 7. Docs do not claim broad scene/per-asset/physics/audio/scripting emission works today. | PASS | Public docs state broad scene mutation emission, broad per-asset async emission, live physics, live audio, and scripting runtime emission are deferred. |
| 8. Docs do not claim desktop screenshot or Playwright validation for this phase; true headless draw-target capture only when visual proof is required. | PASS | Phase docs state image proof is not needed for app/doc/event-consumption changes; the only screenshot-related hits are prior/future plan constraints and existing runtime-launcher guidance rejecting desktop screenshots as evidence. |
| 9. Stale scan hits, if any, are understood and not stale event-system claims. | PASS | Stale scan hits are packaging CLI `/tmp` examples, rendergraph future-direction text, sprint directive/checklist text, and validator/report references. None are stale event-system implementation claims. |
| 10. `.idea/engine.iml` and `.reasonix` remain protected and must not be included in phase closeout. | PASS | `git status --short --ignored ...` shows `.idea/engine.iml` modified and `.reasonix/` untracked before/after validation; this review did not modify or stage them. |

## Commands Run

| Command | Result |
|---|---:|
| `git status --short --branch` | Reviewed dirty state; protected `.idea/engine.iml` and `.reasonix/` present. |
| `cargo fmt --check` | PASS |
| `cargo check -p editor` | PASS with existing renderer/editor warnings |
| `cargo check -p dungeon_dogfood` | PASS with existing renderer/dogfood warnings |
| `cargo check -p engine_pack` | PASS with existing renderer warnings |
| `rg -n "events|lifecycle|EventBus|engine_events" docs/api docs/internal` | PASS/reviewed event docs and cross-links |
| `rg -n "/tmp|TODO|desktop screenshot|playwright|Playwright|not implemented" docs .internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-05-event-system-lifecycle` | PASS/reviewed non-blocking hits |
| `rg -n "engine_events" apps docs src/renderer/src/vulkan src/renderer/src/data src/renderer/src/scene src/renderer/src/shaders src/renderer/src/api src/runtime.rs Cargo.toml src/renderer/Cargo.toml` | PASS; no app or low-level renderer leakage |
| `git diff --check` | PASS |
| `git status --short --ignored .idea/engine.iml .reasonix apps/dungeon_dogfood/src/events.rs apps/editor/src/events.rs .internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-05-event-system-lifecycle/validation/phase-03-validator-review.md` | Reviewed closeout state; dogfood events file is ignored |

## Evidence Notes

- Canonical evidence index was later updated by the main thread after this review to report Phase 03 as validator-passed, committed, pushed, and reported while keeping `fully_validated: false` until final quality review.
- No browser or visual proof is required for this phase. The work changes app-side event observation and documentation, not visible renderer output.
- Existing renderer dead-code warnings remain outside this phase.

## Required Closeout

- Force-add `apps/dungeon_dogfood/src/events.rs` if accepting the phase, because git currently ignores it.
- Do not include `.idea/engine.iml` or `.reasonix/` in phase closeout.
