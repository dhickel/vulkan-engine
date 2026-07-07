# Date
2026-07-07

# Change Summary
Implemented Phase 05 dogfood migration for the engine runtime abstractions plan. `dungeon_dogfood` now proves the app-owned input, event bus, frame clock, camera/controller, and renderer caller-view path in a real runtime app.

# Files
- `apps/dungeon_dogfood/Cargo.toml`: added the root `engine` facade dependency for app-owned runtime helpers.
- `apps/dungeon_dogfood/src/main.rs`: replaced renderer-owned input/camera/frame lifecycle calls with app-owned `InputSystem`, `InputActionEventEmitter`, `RuntimeEventDispatcher`, `FrameClock`, `Camera`, `FPSController`, and caller-provided `CameraView` rendering.
- `apps/dungeon_dogfood/src/events.rs`: installed the dogfood event logger against a caller-owned `EventBus`.
- `apps/dungeon_dogfood/src/audio_bridge.rs`: changed startup audio telemetry to emit into a caller-owned `EventBus`.
- `apps/dungeon_dogfood/src/game_state.rs`: updated stale camera ownership wording.
- `docs/api/14-dogfood-vertical-slice.md`, `docs/api/02-renderer-lifecycle-and-frame-api.md`, `docs/api/06-input-polling-and-listeners.md`, `docs/api/12-events-and-lifecycle.md`, `docs/internal/09-input-winit-integration.md`, `docs/internal/10-event-system-and-lifecycle.md`: documented dogfood as the app-owned runtime proof and clarified legacy renderer compatibility.
- `.internal-dev/specifications/architecture.md`, `.internal-dev/specifications/service-graph.md`, `.internal-dev/specifications/services.md`, `.internal-dev/specifications/api.md`, `.internal-dev/specifications/decisions.md`: recorded dogfood integration proof and updated the Phase 03/04 residuals.

# Behavioral Impact
Dogfood active windowed and headless paths now own gameplay input dispatch, input action events, frame lifecycle events, startup audio telemetry, and camera/player state. The renderer still handles platform/UI/debug/capture side effects, assets, resize, and Vulkan rendering. Legacy renderer-owned APIs remain available for existing examples and compatibility paths.

# Specification Impact
Updated API, architecture, service graph, service, and decision specifications because dogfood moved from a deferred migration target to the real-app proof for the app-owned runtime abstraction contract.

# Validation
- `cargo fmt --check`
- `cargo check -p dungeon_dogfood`
- `cargo test -p dungeon_dogfood`
- `cargo check`
- Legacy-call grep over `apps/dungeon_dogfood/src`
- `RUST_LOG=debug timeout --signal=INT 60s cargo run -p dungeon_dogfood`
- `RUST_LOG=debug timeout --signal=INT 60s cargo run -p dungeon_dogfood -- --headless --capture_target draw --capture_frames 1 --capture_dir .internal-dev/captures/engine-runtime-abstractions-issues-35-37/phase-05-dogfood`

# Risks
The windowed smoke reached the event loop and ran until timeout, but this environment logged repeated swapchain acquire retry warnings near the end. The headless draw capture completed successfully and produced nonblank visual proof for the migrated app-owned camera/view path.

# Follow-up Items
- Phase 06 should finish compatibility labeling, final documentation cleanup, and whole-plan validation.
