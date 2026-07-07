# 00 Specification Lock

## Locked Objective

Sprint 05 adds a common event and application lifecycle contract for alpha engine users. The core contract must be usable and testable without Vulkan. Renderer, root runtime, editor, dogfood, and later systems should consume the same typed event vocabulary.

## Acceptance Criteria

- Workspace contains a new Vulkan-free event crate, likely `src/events` with package name `engine_events`.
- The event crate provides:
  - `EventBus` or equivalent dispatch surface;
  - typed `EngineEvent` envelope or enum;
  - event families for lifecycle, input/action, scene, asset, physics, audio, and scripting;
  - frame stage/order metadata;
  - subscription/listener API;
  - bounded or inspectable event recorder/debug stream;
  - deterministic tests for order, filtering/subscription, recorder behavior, and family construction.
- Renderer and root runtime expose events without requiring subscribers to own raw renderer internals.
- Lifecycle/project/scene/asset-ish runtime events are emitted at safe boundaries.
- Input/action events are bridged after input dispatch and snapshot refresh.
- Docs explain event families, ordering, mutation rules, and deferred areas.
- Final validation includes true engine-owned headless draw capture, not desktop screenshots.

## Validation Criteria

- Required commands:
  - `cargo check`
  - `cargo test -p engine_events`
  - `cargo test -p input`
  - `cargo test -p renderer`
  - `cargo test -p engine`
  - `cargo check -p renderer --examples`
  - `cargo check -p editor`
  - `cargo check -p dungeon_dogfood`
  - `cargo check -p engine_pack`
- Runtime smoke:
  - `RUST_LOG=debug timeout --signal=INT 60s cargo run -- --project apps/editor/sample_project/engine.project.toml --headless --record_debug=10 --record_debug_interval=50 --record_debug_path=.internal-dev/debug_reports/sprint-05-event-system-lifecycle/root-runtime-events-timing.jsonl`
- Visual proof:
  - `RUST_LOG=info timeout --signal=INT 60s cargo run -- --project apps/editor/sample_project/engine.project.toml --headless --capture_frames=3 --capture_frame_start=5 --capture_frame_interval=5 --capture_target=draw --capture_dir=.internal-dev/captures/sprint-05-event-system-lifecycle-headless-draw`
- Evidence index:
  - `.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-05-event-system-lifecycle/artifacts/validation-summary.json`

## Negative Criteria

- No Vulkan, winit, imgui, renderer-internal, editor, dogfood, physics, audio, or scripting dependency may enter `engine_events`.
- Do not change `InputSystem::dispatch_frame()` semantics, priority grouping, per-event consumption, or transient snapshot reset behavior.
- Do not emit events before the state they describe has actually occurred.
- Do not let subscribers mutate scene/renderer state during unsafe mid-render windows.
- Do not use log messages as the only event contract.
- Do not serialize runtime handles as durable event identity.
- Do not mark final validation complete with compile checks alone.

## Non-Goals

- Full editor event browser.
- Full physics collision pipeline.
- Real audio playback lifecycle.
- Real scripting bindings.
- Hot reload/event replay persistence.
- Public stable semver guarantee beyond alpha documentation.

## Constraints

- Current branch: `sprint/alpha-05-event-system-lifecycle`.
- Main thread owns commits, pushes, tracker closeout, changelog timing, and email/report.
- Each sprint phase should be committed and pushed after validation by the main thread.
- Preserve unrelated `.idea/engine.iml` and `.reasonix/`.
- `.internal-dev` is the durable planning/evidence store.
- Headless capture validation must use engine-owned `--headless --capture_target draw` output.

## Assumptions To Verify

- `src/events` can be added as a workspace member without feature unification surprises.
- `engine_events` can use only `std` initially; `serde` remains optional unless docs/recorder output requires it.
- Renderer can depend on `engine_events` and reexport selected types through `renderer` and `renderer::api`.
- Root `engine` can depend on `engine_events` directly or through renderer reexports, whichever keeps API cleaner.

## User Decision Gates

- Stop if package naming `engine_events` conflicts with user preference or crates.io expectations.
- Stop if implementing app-level subscription requires a major renderer lifetime redesign.
- Stop if true headless draw capture is unavailable due to environment/tooling; record `TOOLING_CONSTRAINT` and ask the main thread before fallback.
- Ask before recording out-of-scope future considerations in `.internal-dev/notes/`.

## Stop Rules

- Stop phase work if event integration requires raw renderer internals in app code.
- Stop if event ordering cannot be documented and tested without Vulkan.
- Stop if a worker needs to deeply implement physics/audio/scripting behavior.
- Stop final closeout if validation summary status contradicts missing/failed evidence.
