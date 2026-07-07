# Phase 05 Worker Report: Dungeon Dogfood Runtime Migration Proof

Date: 2026-07-07
Status: implemented, awaiting validator review

## Summary

Migrated `dungeon_dogfood` active runtime paths to app-owned runtime primitives:

- app-owned `InputSystem` with routed renderer platform input;
- app-owned `EventBus` with input, frame lifecycle, and audio telemetry;
- app-owned `FrameClock`;
- app-owned `Camera` and `FPSController`;
- renderer caller-view rendering through `CameraView`.

This keeps renderer ownership focused on platform side effects, assets, resize/capture, and Vulkan submission while preserving raw primitive access through the root facade and underlying crates.

## Changed Files

- `apps/dungeon_dogfood/Cargo.toml`
- `apps/dungeon_dogfood/src/main.rs`
- `apps/dungeon_dogfood/src/events.rs`
- `apps/dungeon_dogfood/src/audio_bridge.rs`
- `apps/dungeon_dogfood/src/game_state.rs`
- `docs/api/14-dogfood-vertical-slice.md`
- `docs/api/02-renderer-lifecycle-and-frame-api.md`
- `docs/api/06-input-polling-and-listeners.md`
- `docs/api/12-events-and-lifecycle.md`
- `docs/internal/09-input-winit-integration.md`
- `docs/internal/10-event-system-and-lifecycle.md`
- `.internal-dev/specifications/api.md`
- `.internal-dev/specifications/architecture.md`
- `.internal-dev/specifications/service-graph.md`
- `.internal-dev/specifications/services.md`
- `.internal-dev/specifications/decisions.md`
- `.internal-dev/changelogs/2026-07-07-engine-runtime-abstractions-phase-05-dogfood-migration.md`

## Frame Sequence

Windowed dogfood now runs this sequence:

1. `Renderer::route_platform_input(...)` handles renderer platform/UI/debug/capture side effects.
2. `engine::input::queue_routed_input_event(...)` queues uncaptured events into the app `InputSystem`.
3. On redraw, `FrameClock::tick()` produces the app frame index and delta.
4. `InputSystem::dispatch_frame()` runs exactly once for that app frame.
5. `InputActionEventEmitter::emit_from_snapshot(...)` emits app input telemetry into the app `EventBus`.
6. `RuntimeEventDispatcher::drain_input(...)` drains input listeners.
7. `RuntimeEventDispatcher::frame_started(...)` emits/drains app frame start.
8. `FPSController` reads the app input snapshot and mutates the app camera intent.
9. `PlayerState` ingests camera intent, collision resolves movement, and the app camera is corrected to the player position.
10. Dogfood pumps renderer asset tasks, builds `CameraView`, and calls `render_scene_with_view(...)`.
11. `RuntimeEventDispatcher::frame_ended(...)` emits/drains app frame end.

The headless capture path uses the same app-owned input/event/camera/frame path and calls `render_scene_headless_with_view(...)`.

## Contract Notes

- Active dogfood code no longer calls `renderer.events_mut()`, `renderer.camera_position()`, `renderer.set_camera_position()`, `renderer.install_default_fps_input()`, `renderer.begin_frame()`, `renderer.render_scene_in_frame()`, `renderer.render_scene(...)`, `renderer.render_scene_headless(...)`, or `renderer.update_input(...)`.
- Startup audio telemetry emits into the same caller-owned app `EventBus`.
- The app camera is collision-corrected before the `CameraView` is constructed.
- Renderer-owned legacy APIs remain in place for compatibility and examples.
- No reverse dependency from renderer/support crates to the root `engine` crate was introduced.

## Validation

Passed:

- `cargo fmt --check`
- `cargo check -p dungeon_dogfood`
- `cargo test -p dungeon_dogfood`
- `cargo check`
- `rg -n "events_mut\\(|camera_position\\(|set_camera_position\\(|install_default_fps_input\\(|begin_frame\\(|render_scene_in_frame\\(|render_scene\\(|render_scene_headless\\(|update_input\\(" apps/dungeon_dogfood/src`
- `RUST_LOG=debug timeout --signal=INT 60s cargo run -p dungeon_dogfood`
- `RUST_LOG=debug timeout --signal=INT 60s cargo run -p dungeon_dogfood -- --headless --capture_target draw --capture_frames 1 --capture_dir .internal-dev/captures/engine-runtime-abstractions-issues-35-37/phase-05-dogfood`

Runtime proof:

- Windowed smoke reached `Dungeon dogfood initialized, starting event loop` and ran until the scripted timeout.
- Headless capture completed with `Headless capture complete: 1/1 captures written`.
- Capture sidecar: `.internal-dev/captures/engine-runtime-abstractions-issues-35-37/phase-05-dogfood/dungeon-dogfood-frame-0-draw-seq-0000.json`
- Capture PNG: `.internal-dev/captures/engine-runtime-abstractions-issues-35-37/phase-05-dogfood/dungeon-dogfood-frame-0-draw-seq-0000.png`

Observed existing warning noise:

- Renderer dead-code warnings remain present and unrelated to this phase.
- Windowed smoke logged repeated swapchain acquire retry warnings near the end of the bounded run. Headless draw capture succeeded.

## Residuals

- Renderer examples and compatibility paths still use renderer-owned input/event/camera APIs by design.
- Phase 06 should perform final compatibility labeling, documentation closeout, and whole-plan validation.
