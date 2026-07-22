---
schema_version: 1
document_type: service-graph-specification
status: active
owner: architecture
created: 2026-07-03
---

# Service Graph Specification

## Allowed Interactions

| id | from | to | status | allowed_interaction | boundary_rule | validation | related_decisions |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SVC-20260703-01 | Apps/examples | Engine crates | active | Apps and examples consume renderer, input, audio, physics, scripting, events, and pack crates through their public crate APIs. | Avoid reaching into private implementation modules when a facade API exists. | `cargo check` for touched packages and examples. | none |
| SVC-20260703-02 | Renderer API facade | Renderer internals | active | Public renderer API delegates runtime work to data and Vulkan internals while keeping callers on facade-level types where practical. | Preserve module AGENTS ownership boundaries before editing internals. | `cargo check -p renderer --examples` plus targeted tests/captures. | none |
| SVC-20260707-01 | Root facade | Renderer/input/events crates | active | Root app-owned input helpers may consume renderer routing results, queue uncaptured winit/device events into caller-owned `input::InputSystem`, and emit input action events into caller-owned `engine_events::EventBus`. | Renderer must not depend on the root `engine` crate; support crates remain independent. | `cargo test -p engine`, `cargo check -p renderer`. | `DECISION-20260707-01` |
| SVC-20260707-02 | Root facade | Events crate | active | Root event helpers may construct a recorded app-owned `EventBus` and emit/drain lifecycle stages through `RuntimeEventDispatcher` without hiding the raw `engine_events` primitives. | `engine_events` must not depend on root `engine`, renderer, Vulkan, or windowing crates. Renderer must not depend on the root `engine` facade. | `cargo test -p engine`, `cargo test -p engine_events`, `cargo check -p renderer`. | `DECISION-20260707-02` |
| SVC-20260707-03 | Dungeon dogfood app | Root facade and renderer facade | active | Dogfood may depend on the root `engine` crate for app-owned input/event/camera/frame helpers and on `renderer` for scene/assets/rendering. | This app dependency must not create a reverse dependency from renderer/support crates to root `engine`; dogfood remains an app proof, not a framework layer. | `cargo check -p dungeon_dogfood`, `cargo test -p dungeon_dogfood`, `cargo check`. | `DECISION-20260707-01`, `DECISION-20260707-02` |
| SVC-20260707-04 | Root engine binary | Root engine library and support crates | active | The binary launcher may use the root library helpers and renderer facade to run project/package/scene data. | The binary is a launcher, not a required app runtime object. Support crates must not depend on it. | `cargo check`, root runtime smoke when required. | `DECISION-20260707-03` |
| SVC-20260707-05 | Renderer facade | Caller app/root facade | active | Renderer consumes caller-provided `CameraView` values but never calls back into root `engine` for app-owned state. | The render DTO dependency direction is app/root -> renderer API type -> renderer internals; no renderer -> root edge. | `cargo check -p renderer`, `cargo test -p renderer`, `cargo test -p engine`. | `DECISION-20260707-04` |
| SVC-20260722-02 | Stabilization phases | Workspace crates | active | During the safety-refactor stabilization sprint, no new dependency edges are created between workspace crates. Internal renderer module extraction does not change crate-level dependency direction. Phase implementers must verify `cargo tree` edges before and after each phase to confirm no dependency inversion. | Does not block internal module extraction within the renderer crate. App-level bridge modules (e.g., mesh_collider_bridge) remain app-local. | `cargo check`, `cargo tree` diff. | `DECISION-20260722-07` |

## Phase 03 Dependency Audit Note

Phase 01 audits for the core app-owned workflow primitives confirmed no new dependency direction is required beyond the existing root-facade-to-support-crates edges above. Renderer and support crates must not depend on the root `engine` facade.

## Drift Records

| id | spec | status | observed_drift | impact | routing | source | review_after |
| --- | --- | --- | --- | --- | --- | --- | --- |
