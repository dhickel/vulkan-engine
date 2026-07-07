# Phase 03 App-Owned Input Worker Report

Date: 2026-07-07

## Scope
Implemented only the app-owned input dispatch/action-event helper and renderer platform/input routing split. Did not migrate dogfood active path, renderer event lifecycle ownership, or app lifecycle event ownership.

## Changed Files
- `src/input.rs`: added `InputActionEventEmitter` with a per-emitter observed action-value map; added `queue_routed_input_event` for caller-owned input queueing from renderer routing results; documented that app dispatch should occur once at app frame boundary independent of resize-skip render outcomes.
- `src/lib.rs`: exported app-owned input helper and renderer routing types in root prelude.
- `src/render.rs`: exported renderer routing types through `engine::render`.
- `src/renderer/src/api/renderer.rs`: added `RendererInputRouting`, `RendererInputSuppression`, `Renderer::route_platform_input`, and `queue_renderer_owned_input`; updated `Renderer::update_input` to compose the split helper; aligned legacy action-event bridge with same-frame press/release transients.
- `src/renderer/src/api/mod.rs`, `src/renderer/src/api/prelude.rs`, `src/renderer/src/lib.rs`: exported renderer routing types.
- `tests/input_action_events.rs`: added root app-owned input/action-event and queue helper coverage.
- `tests/facade_imports.rs`: added facade import checks for `InputActionEventEmitter` and renderer routing result.
- `src/renderer/tests/integration.rs`: added renderer routing result public API check.
- `docs/api/06-input-polling-and-listeners.md`, `docs/internal/09-input-winit-integration.md`, `docs/api/02-renderer-lifecycle-and-frame-api.md`: documented app-owned route and input dispatch policy.
- `.internal-dev/specifications/architecture.md`, `.internal-dev/specifications/service-graph.md`, `.internal-dev/specifications/services.md`, `.internal-dev/specifications/api.md`, `.internal-dev/specifications/decisions.md`: recorded intended contracts and decision.
- `.internal-dev/changelogs/2026-07-07-engine-runtime-abstractions-phase-03-app-owned-input.md`: closeout changelog.

## Criteria Satisfied
- Added root `InputActionEventEmitter` with its own observed value map per emitter/app input stream.
- Emits `EngineEvent::Input(InputActionEvent)` into caller-owned `EventBus` from caller-provided `InputSnapshot` and frame index.
- Tests cover press once, release once, changed value once, same-frame press/release, transients surviving exactly one dispatch, and no duplicate emission across frames.
- Split renderer platform handling from renderer-owned input queueing via `Renderer::route_platform_input`.
- Preserved ImGui forwarding, F1/F2/F12 platform handling, cursor focus side effects, UI capture suppression, window filtering, mouse motion, mouse wheel, modifiers, and keyboard repeat behavior consistent with the legacy path.
- Kept `Renderer::update_input` behavior-preserving by composing route result plus renderer-owned queueing.
- Added app/root helper to queue uncaptured routed events into caller-owned `InputSystem`.
- Did not change `InputSystem::dispatch_frame` semantics.
- Did not dispatch app-owned input inside renderer.
- Did not migrate dogfood active path or event lifecycle ownership.
- Did not introduce renderer dependency on root `engine`.

## Criteria Not Satisfied
None known within Phase 03 scope.

## Validation
- `cargo fmt --check`: pass.
- `cargo check -p input`: pass.
- `cargo test -p input`: pass, 10 tests.
- `cargo test -p engine`: pass, root/unit/facade/app-owned input tests.
- `cargo check -p renderer`: pass with existing renderer dead-code warnings.
- `cargo test -p renderer`: pass, 167 unit tests, 21 integration tests, 5 ignored doctests; existing renderer dead-code warnings remain.
- `rg -n "dispatch_frame\\(|emit_input_action|InputActionEventEmitter|update_input\\(|DeviceEvent::MouseMotion" src apps tests`: inspected. New app-owned helper/tests are visible; legacy renderer dispatch remains only in renderer-owned prepare paths; `Renderer::update_input` call sites remain in apps/examples by design; `DeviceEvent::MouseMotion` remains handled in renderer routing and root queue helper.

## Safe Adjacent Hygiene
- Re-exported `InputContext` from the root input facade because root app-owned tests and custom app layers need the public support-crate callback context type.
- Updated docs/specs/changelog for the new public behavior and resize-skip dispatch policy.

## Internal-Dev Artifacts Touched
- `.internal-dev/specifications/architecture.md`
- `.internal-dev/specifications/service-graph.md`
- `.internal-dev/specifications/services.md`
- `.internal-dev/specifications/api.md`
- `.internal-dev/specifications/decisions.md`
- `.internal-dev/changelogs/2026-07-07-engine-runtime-abstractions-phase-03-app-owned-input.md`
- `.internal-dev/plans/engine-runtime-abstractions-issues-35-37/work-units/phase-03-app-owned-input-worker-report.md`

## Blockers Or Plan Flaws
No blockers found. The remaining cross-domain risk is intentional: dogfood and lifecycle event ownership still use the legacy renderer-owned path until later phases migrate them.
