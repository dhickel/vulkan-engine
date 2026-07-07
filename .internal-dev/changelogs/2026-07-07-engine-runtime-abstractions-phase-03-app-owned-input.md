# Date
2026-07-07

# Change Summary
Implemented Phase 03 app-owned input dispatch support for the engine runtime abstractions plan. Added a root `InputActionEventEmitter`, a root helper for queueing renderer-routed uncaptured input into caller-owned `InputSystem`, and split renderer platform handling from renderer-owned input queueing through `Renderer::route_platform_input`.

# Files
- `src/input.rs`: added app-owned action event emitter, routed input queue helper, and `InputContext` facade re-export for app/custom layers.
- `src/lib.rs`: re-exported app-owned input helpers and renderer routing types through the root prelude.
- `src/render.rs`: re-exported renderer input routing types through `engine::render`.
- `src/renderer/src/api/renderer.rs`: added `RendererInputRouting`, `RendererInputSuppression`, `route_platform_input`, and legacy queue wrapper; preserved `Renderer::update_input` by composing the split helper.
- `src/renderer/src/api/mod.rs`, `src/renderer/src/api/prelude.rs`, `src/renderer/src/lib.rs`: exported renderer routing types.
- `tests/input_action_events.rs`: covered press, release, changed value, same-frame press/release, transient lifetime/no duplicate emission, and routed device input queueing.
- `tests/facade_imports.rs`: covered root facade imports for app-owned input helper and renderer routing result.
- `src/renderer/tests/integration.rs`: covered renderer routing type public API.
- `docs/api/06-input-polling-and-listeners.md`, `docs/internal/09-input-winit-integration.md`, `docs/api/02-renderer-lifecycle-and-frame-api.md`: documented app-owned input routing and resize-skip input policy.
- `.internal-dev/specifications/architecture.md`, `.internal-dev/specifications/service-graph.md`, `.internal-dev/specifications/services.md`, `.internal-dev/specifications/api.md`, `.internal-dev/specifications/decisions.md`: recorded intended contracts and decision.
- `.internal-dev/plans/engine-runtime-abstractions-issues-35-37/work-units/phase-03-app-owned-input-worker-report.md`: phase work report.

# Behavioral Impact
Apps can now call renderer platform routing for ImGui/debug/capture/cursor side effects, queue uncaptured winit/device input into their own `InputSystem`, dispatch exactly once at the app frame boundary, and emit snapshot-derived input action events into their own `EventBus`.

Legacy renderer-owned `Renderer::update_input` still compiles and queues input into the renderer-owned `InputSystem` by composing the same routing decision.

# Specification Impact
Updated architecture, service graph, service, API, and decision specifications to record app-owned input dispatch/action-event emission as intended truth while preserving renderer-owned compatibility paths.

# Risks
Dogfood still uses the legacy renderer-owned input path by design for this phase. The new route is covered by unit/API tests but is not yet wired into a migrated active app path.

# Follow-up Items
- Phase 04 should continue event lifecycle ownership migration without duplicating input dispatch.
- Phase 05 should migrate dogfood only after app-owned event lifecycle ownership is defined.
