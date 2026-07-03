# Sprint 05: Event System And Application Lifecycle

Status: planned

## Objective

Create the alpha event contract that lets runtime apps, editor tooling, dogfood gameplay, and later physics/audio/scripting systems observe engine lifecycle, input/action, scene, asset, physics, audio, and scripting events without owning renderer internals or requiring Vulkan to test event behavior.

## User-Visible Outcome

After this sprint, an app or tool can subscribe to a stable event bus, receive lifecycle/project/scene/asset/input-action events at documented frame boundaries, and inspect a recorded event stream for diagnosis. The root runtime continues to launch projects and render headless/windowed, with lifecycle events available to non-render systems.

## Work Classification

Large-ish roadmap sprint. The work spans a new core crate, renderer facade integration, root runtime lifecycle, app/tool examples, docs, and final visual/runtime validation.

## In Scope

- Add a Vulkan-free workspace crate, likely `src/events` with package name `engine_events`.
- Define typed event families: lifecycle, input/action, scene, asset, physics, audio, scripting.
- Define event ordering, frame stage, delivery, subscription, and recorder/debug contracts.
- Bridge input action snapshots to app events after `InputSystem::dispatch_frame()` without changing input consumption semantics.
- Expose event types and bus access through renderer facade and root runtime where appropriate.
- Emit lifecycle/project/scene/asset-ish events at safe load/frame/shutdown boundaries.
- Provide placeholder physics/audio/scripting event types and construction/dispatch tests only.
- Update API/internal docs and add a small non-render sample/test demonstrating subscription and recording.
- Validate with compile/tests plus true engine-owned headless draw capture through the root runtime.

## Out Of Scope

- Deep physics, audio, or scripting implementation.
- Replacing the input crate's dispatch model or consumption behavior.
- Raw renderer ownership from app event subscribers.
- Event-driven mutation during renderer mid-frame or render hooks unless explicitly staged.
- Networked, async, cross-thread, or persisted event replay.
- Editor UI redesign or a full event debug panel unless a minimal recorder display already fits safely.

## Target Surfaces

- Code: `Cargo.toml`, new `src/events/`, `src/renderer/Cargo.toml`, `src/renderer/src/lib.rs`, `src/renderer/src/api/mod.rs`, `src/renderer/src/api/renderer.rs`, `src/runtime.rs`, `src/launch.rs` if needed, `apps/editor/`, `apps/dungeon_dogfood/`, and optional `tools/engine_pack/` sample/help only if needed.
- Docs: `docs/api/00-index.md`, new `docs/api/12-events-and-lifecycle.md`, `docs/internal/00-index.md`, new `docs/internal/10-event-system-and-lifecycle.md`, and targeted references from runtime/input/asset docs.
- Validation artifacts: this sprint directory, `.internal-dev/debug_reports/sprint-05-event-system-lifecycle/`, `.internal-dev/captures/sprint-05-event-system-lifecycle-headless-draw/`.

## Assumptions

- `engine_events` is the preferred package name because it is explicit and avoids colliding with generic crates.
- The event crate should depend on `input` only if needed for action identifiers; otherwise use string/newtype equivalents and provide bridge conversions in renderer/root runtime.
- Serialization is optional and should not be required for the hot path unless the recorder/debug output needs it.
- Root runtime remains the canonical project launcher from Sprint 04.
- Sprint closeout commits, pushes, tracker finalization, changelog timing, and email are main-thread responsibilities.

## Acceptance Criteria

- A Vulkan-free event crate exists in the workspace and passes `cargo test -p engine_events`.
- Event bus semantics are typed, ordered, frame-staged, documented, and tested without renderer/Vulkan.
- Initial event families are present with stable constructors or structs for lifecycle, input/action, scene, asset, physics, audio, and scripting.
- Renderer facade and root runtime expose event access without requiring raw renderer ownership.
- Root runtime emits lifecycle/project/scene/load/shutdown events at safe boundaries for both headless and windowed paths where observable.
- Input/action event bridge runs only after input dispatch/frame snapshot and does not change action mapping, priority, or consumption behavior.
- Editor/dogfood/sample code demonstrates event subscription or recording without broad gameplay/editor rewrites.
- Docs describe ordering, supported event families, and known deferred behavior.
- Compile/test validation and true headless draw capture evidence are recorded in `artifacts/validation-summary.json`.

## Negative Criteria

- No event code may depend on Vulkan, window handles, imgui, or renderer internals.
- Do not emit contradictory events for the same scene/action lifecycle boundary.
- Do not dispatch app callbacks while the renderer is mid-rendering or while a mutable renderer borrow is exposed.
- Do not implement real physics/audio/scripting runtime behavior beyond event type definitions and tests.
- Do not mark the sprint `fully_validated` until phase validators, final quality review, and draw-target capture proof pass.
- Do not touch unrelated `.idea/engine.iml` or `.reasonix/` changes.

## Validation Plan

- Compile/test: `cargo check`, `cargo test -p engine_events`, `cargo test -p input`, `cargo test -p renderer`, `cargo test -p engine`, `cargo check -p renderer --examples`, `cargo check -p editor`, `cargo check -p dungeon_dogfood`, `cargo check -p engine_pack`.
- Runtime smoke: `RUST_LOG=debug timeout --signal=INT 60s cargo run -- --project apps/editor/sample_project/engine.project.toml --headless --record_debug=10 --record_debug_interval=50 --record_debug_path=.internal-dev/debug_reports/sprint-05-event-system-lifecycle/root-runtime-events-timing.jsonl`.
- Visual/capture proof: `RUST_LOG=info timeout --signal=INT 60s cargo run -- --project apps/editor/sample_project/engine.project.toml --headless --capture_frames=3 --capture_frame_start=5 --capture_frame_interval=5 --capture_target=draw --capture_dir=.internal-dev/captures/sprint-05-event-system-lifecycle-headless-draw`.
- Docs/process checks: stale-reference sweep over docs and this sprint directory; verify no `/tmp` evidence, stale agent ids, or pending claims in closeout artifacts.

## Advanced-Planner Handoff

Phases are locked in `work-units/README.md` and `worker-directives/`. Orchestrate phases in order. Validate each phase before dispatching dependent work. Main thread commits/pushes after each validated phase and owns final email/report responsibilities.

## Closeout Checklist

- Phase validation reports are present under `validation/`.
- `artifacts/validation-summary.json` is reconciled with phase validators, final quality review, and capture evidence.
- Known residuals are fixed or tracked in `.internal-dev/bugs/` or `.internal-dev/notes/` after user confirmation where required.
- Changelog timing is confirmed with the user per repo guidance.
- Sprint tracker is updated by the main thread at closeout.
