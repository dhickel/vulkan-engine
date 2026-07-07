---
schema_version: 1
document_type: inventory-matrix
status: phase_00_complete
owner: deprecated-renderer-api-migration
created: 2026-07-07
related_issue:
  - https://github.com/dhickel/vulkan-engine/issues/39
---

# Deprecated Renderer API Inventory Matrix — Phase 00

> Generated 2026-07-07 from `rg -n "update_input\(|\.input\(\)|\.input_mut\(\)|\.events\(\)|\.events_mut\(\)|render_scene\(|set_camera_position|set_camera_look_at|install_default_fps_input|uninstall_default_fps_input|FPSController|route_platform_input|queue_routed_input_event|render_scene_with_view|render_scene_headless_with_view|CameraView" src apps tests docs .internal-dev/specifications`

## Seed Row Classification

| candidate_api_or_symbol | classification | eligibility | senior_decision |
| --- | --- | --- | --- |
| `Renderer::update_input(...)` | mixed | keep_compatibility | Root launcher + marching_terrain + examples keep compatibility; dogfood already migrated. See senior decision table for launcher/marching_terrain migration scope. |
| `Renderer::input()` / `Renderer::input_mut()` | internal_compatibility_implementation | keep_compatibility | Used only in marching_terrain (compatibility path) and renderer internals. No active app code depends on it. |
| `Renderer::events()` / `Renderer::events_mut()` | internal_compatibility_implementation | keep_compatibility | Only used in `event_logging.rs` (internal renderer logging) and docs. No external app code uses it. |
| renderer-owned `render_scene(...)` | mixed | keep_compatibility | Root launcher + examples + marching_terrain keep compatibility. Dogfood already migrated to `render_scene_with_view`. See senior decision table. |
| `set_camera_position` / `set_camera_look_at` | mixed | keep_compatibility | Used by marching_terrain (compatibility), capture_tests (renderer tooling), and internal renderer camera mgmt. Still active for capture/demo tooling. |
| `install_default_fps_input` / `uninstall_default_fps_input` | mixed | keep_compatibility | Used by renderer examples (compatibility coverage), root launcher (compatibility), and marching_terrain (compatibility). Dogfood installs own app-owned FPS layer. |
| `FPSController` type | active_current_path | keep_current | Used both by renderer internally (FpsInputPlugin) and by dogfood on app-owned path. The type itself is current — renderer-owned *installation* is compatibility. |
| docs presenting renderer-owned loops as current app path | stale_documentation | migrate_call_site | Multiple docs still present `update_input`/`render_scene`/`install_default_fps_input` as the primary or default quickstart. Phase 01 will label compatibility. |
| root launcher compatibility path (`src/runtime.rs`) | intentional_compatibility_coverage | defer | Default classification per rules. Migration scope needs senior decision. |
| renderer examples compatibility coverage | intentional_compatibility_coverage | keep_compatibility | All renderer examples use compatibility path through common module. Intentionally kept for smoke/demo coverage. |
| `route_platform_input` / `queue_routed_input_event` | active_current_path | keep_current | App-owned path APIs. Already active in dogfood, root helpers, tests. |
| `render_scene_with_view` / `render_scene_headless_with_view` | active_current_path | keep_current | Active replacement APIs. Used by dogfood. |
| `CameraView` struct + `camera_view_for_size` | active_current_path | keep_current | Active current path. Dogfood, docs, specs, tests all use/prove this path. |

## Per-Use Detail Rows

### Renderer API Definitions — Renderer Crate (`src/renderer/src/api/renderer.rs`)

| file_line | candidate_api_or_symbol | observed_usage | classification | replacement_path | retained_compatibility_reason | validation_command | docs_spec_update | follow_up |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| renderer.rs:20 | `FPSController` | Import for internal `FpsInputPlugin` struct | internal_compatibility_implementation | N/A | Renderer uses `FPSController` in its internal default FPS plugin. | `cargo check -p renderer` | none | none |
| renderer.rs:88 | `CameraView` | Struct definition | active_current_path | N/A | Active API. Renderer DTO consumed at submission boundary. | `cargo test -p renderer` | none | none |
| renderer.rs:94 | `CameraView` | `impl CameraView` block | active_current_path | N/A | Active API implementation. | `cargo test -p renderer` | none | none |
| renderer.rs:147 | `FPSController` | Field in `FpsInputPlugin` struct | internal_compatibility_implementation | N/A | Internal renderer plugin state. | `cargo check -p renderer` | none | none |
| renderer.rs:260 | `install_default_fps_input` | Public API definition — installs WASD+mouse layer into renderer-owned `InputSystem` | internal_compatibility_implementation | App-owned: `InputSystem::add_layer` + caller-owned `FPSController` | Compatibility API definition for examples/demos. | `cargo check -p renderer --examples` | Mark as compatibility in API docs if Phase 03 deprecates. | Phase 03 decision |
| renderer.rs:279 | `FPSController::new` | Internal call within `install_default_fps_input` | internal_compatibility_implementation | N/A | Part of compatibility implementation. | `cargo check -p renderer` | none | none |
| renderer.rs:288 | `uninstall_default_fps_input` | Public API definition | internal_compatibility_implementation | N/A | Companion to `install_default_fps_input`. | `cargo check -p renderer` | Mark compatibility in API docs if needed. | Phase 03 |
| renderer.rs:324 | `update_input` | Public API definition — composes `route_platform_input` + renderer-owned queueing | internal_compatibility_implementation | `Renderer::route_platform_input` + `engine::input::queue_routed_input_event` | Compatibility API. Exists for backward compat; `route_platform_input` is the current primitive. | `cargo check -p renderer --examples` | Mark as compatibility in API docs. | Phase 03 |
| renderer.rs:329 | `route_platform_input` | Called internally by `update_input` | internal_compatibility_implementation | N/A | `update_input` composes current primitive internally. | `cargo test -p renderer` | none | none |
| renderer.rs:342 | `route_platform_input` | Public API definition — current primitive | active_current_path | N/A | Active API. Used by dogfood through `engine::input::route_platform_input_to_app`. | `cargo test -p renderer`, `cargo check -p dungeon_dogfood` | none | none |
| renderer.rs:539 | `render_scene` | Public API definition — one-shot compatibility render | internal_compatibility_implementation | `render_scene_with_view` / `render_scene_headless_with_view` with caller-provided `CameraView` | Compatibility API. Internally calls `execute_frame_lifecycle` + `prepare_frame`. | `cargo check -p renderer --examples` | Mark as compatibility in API docs. | Phase 03 |
| renderer.rs:576 | `render_scene_with_view` | Public API definition — caller-view render | active_current_path | N/A | Active API. Used by dogfood. | `cargo test -p renderer`, `cargo check -p dungeon_dogfood` | none | none |
| renderer.rs:601 | `render_scene_headless_with_view` | Public API definition — caller-view headless render | active_current_path | N/A | Active API. Used by dogfood capture path. | `cargo test -p renderer`, `cargo check -p dungeon_dogfood` | none | none |
| renderer.rs:1092 | `CameraView::from_camera` | Internal call: builds `CameraView` from renderer-owned camera for compatibility render path | internal_compatibility_implementation | N/A | Used in `render_scene` compatibility path. | `cargo test -p renderer` | none | none |
| renderer.rs:1101 | `CameraView` | Parameter in internal `render_scene_internal_with_view` | active_current_path | N/A | Active internal API for view-based rendering. | `cargo test -p renderer` | none | none |
| renderer.rs:1265 | `set_camera_position` | Public API definition — sets renderer-owned camera position | internal_compatibility_implementation | App-owned: build `Camera` in app loop, position it before `camera_view_for_size` | Compatibility API. Still needed for capture_tests, marching_terrain compatibility. | `cargo check -p renderer` | Mark as capture/compatibility tooling in API docs. | requires_senior_decision |
| renderer.rs:1273 | `set_camera_look_at` | Public API definition — sets renderer-owned camera from look-at | internal_compatibility_implementation | App-owned: `Camera::look_at` before `camera_view_for_size` | Compatibility API. Used by capture_tests/common.rs. | `cargo check -p renderer` | Mark as capture/compatibility tooling in API docs. | requires_senior_decision |
| renderer.rs:1310 | `CameraView` | Parameter in `build_submission_with_camera_view` | active_current_path | N/A | Internal helper for view-based submission. | `cargo test -p renderer` | none | none |
| renderer.rs:1435 | `CameraView`, `build_submission_with_camera_view` | Internal imports | active_current_path | N/A | Active internal code. | `cargo test -p renderer` | none | none |
| renderer.rs:1562 | `CameraView::from_matrices` | Test code within renderer | active_current_path | N/A | Test of active API. | `cargo test -p renderer` | none | none |
| renderer.rs:294-298 | `input_mut()` / `input()` | Public API definitions — access renderer-owned `InputSystem` | internal_compatibility_implementation | App-owned `InputSystem` | Compatibility accessors. Only marching_terrain uses externally. | `cargo check -p renderer` | Mark as compatibility. | Phase 03 |
| renderer.rs:302-306 | `events()` / `events_mut()` | Public API definitions — access renderer-owned `EventBus` | internal_compatibility_implementation | App-owned `EventBus` + `RuntimeEventDispatcher` | Compatibility accessors. Only event_logging.rs uses internally. | `cargo check -p renderer` | Mark as compatibility/internal tooling. | Phase 03 |

### Renderer API Re-exports & Prelude

| file_line | candidate_api_or_symbol | observed_usage | classification | replacement_path | retained_compatibility_reason | validation_command | docs_spec_update | follow_up |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| renderer/src/api/mod.rs:61 | `CameraView` | Public re-export in renderer facade | active_current_path | N/A | Active API re-export. | `cargo check -p renderer` | none | none |
| renderer/src/api/prelude.rs:17 | `CameraView` | Prelude re-export | active_current_path | N/A | Active API in prelude. | `cargo check -p renderer` | none | none |
| renderer/src/lib.rs:49 | `CameraView` | Crate root re-export | active_current_path | N/A | Active API re-export. | `cargo check -p renderer` | none | none |
| renderer/src/lib.rs:83 | `FPSController` | Crate root re-export | active_current_path | N/A | `FPSController` type is current; used in both compatibility and app-owned paths. | `cargo check -p renderer` | none | none |
| renderer/src/data/camera.rs:411 | `FPSController` | Struct definition | active_current_path | N/A | Core data type definition; used by both renderer plugin and dogfood. | `cargo test -p renderer camera` | none | none |
| renderer/src/data/camera.rs:417 | `FPSController` | `impl FPSController` block | active_current_path | N/A | Core data type implementation. | `cargo test -p renderer camera` | none | none |

### Renderer Internal: Event Logging

| file_line | candidate_api_or_symbol | observed_usage | classification | replacement_path | retained_compatibility_reason | validation_command | docs_spec_update | follow_up |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| event_logging.rs:15 | `renderer.events_mut()` | Subscribes to renderer-owned event bus for debug logging | internal_compatibility_implementation | N/A | Internal renderer debugging/logging tool. Not a compatibility burden for apps. | `cargo check -p renderer` | Internal docs note: logging helper uses renderer-owned bus. | None unless Phase 03 removes event bus. |

### Renderer Internal: Scene

| file_line | candidate_api_or_symbol | observed_usage | classification | replacement_path | retained_compatibility_reason | validation_command | docs_spec_update | follow_up |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| api/scene.rs:667 | `Scene::set_camera` | Scene-level camera metadata setter | active_current_path | N/A | Different from `Renderer::set_camera_*`. Scene camera for submission. Not covered by this migration. | `cargo test -p renderer` | none | none |

### Renderer Examples — Compatibility Coverage

| file_line | candidate_api_or_symbol | observed_usage | classification | replacement_path | retained_compatibility_reason | validation_command | docs_spec_update | follow_up |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| examples/common/mod.rs:486 | `renderer.install_default_fps_input()` | Shared example init — installs FPS input layer | intentional_compatibility_coverage | N/A | All renderer examples (`demo_pbr`, `demo_unlit`, `demo_model_load`, `api_test`, `demo_async_loading`) use this shared module. Intentionally kept for smoke/demo coverage. | `cargo check -p renderer --examples` | Label in examples README as compatibility path. | Phase 03 warning policy |
| examples/common/mod.rs:532 | `renderer.update_input(&window, &event)` | Shared example event loop — feeds winit events to renderer | intentional_compatibility_coverage | `route_platform_input` + `queue_routed_input_event` (not required for examples) | Examples intentionally use compatibility path per spec lock. | `cargo check -p renderer --examples` | Label as compatibility. | Phase 03 warning policy |
| examples/common/mod.rs:596 | `renderer.render_scene(&window, &mut scene)` | Shared example render — one-shot compatibility render | intentional_compatibility_coverage | `render_scene_with_view` (not required for examples) | Examples intentionally use compatibility path. | `cargo check -p renderer --examples` | Label as compatibility. | Phase 03 warning policy |
| examples/demo_async_loading.rs:65 | `renderer.install_default_fps_input()` | Example-specific init | intentional_compatibility_coverage | N/A | Example intentionally covers compatibility. | `cargo check -p renderer --examples` | none | Phase 03 |
| examples/demo_async_loading.rs:116 | `renderer.update_input(&window, &event)` | Example event loop | intentional_compatibility_coverage | N/A | Example intentionally covers compatibility. | `cargo check -p renderer --examples` | none | Phase 03 |
| examples/demo_async_loading.rs:188 | `renderer.render_scene(&window, &mut scene)` | Example render call | intentional_compatibility_coverage | N/A | Example intentionally covers compatibility. | `cargo check -p renderer --examples` | none | Phase 03 |
| examples/api_test.rs:71 | `renderer.install_default_fps_input()` | Example-specific init | intentional_compatibility_coverage | N/A | Example intentionally covers compatibility. | `cargo check -p renderer --examples` | none | Phase 03 |
| examples/api_test.rs:123 | `renderer.update_input(&window, &event)` | Example event loop | intentional_compatibility_coverage | N/A | Example intentionally covers compatibility. | `cargo check -p renderer --examples` | none | Phase 03 |

### Renderer Capture Tests

| file_line | candidate_api_or_symbol | observed_usage | classification | replacement_path | retained_compatibility_reason | validation_command | docs_spec_update | follow_up |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| examples/capture_tests/common.rs:283 | `renderer.set_camera_look_at(eye, target, Vec3::Y)` | Capture test camera setup — configures renderer-owned camera for headless captures | intentional_compatibility_coverage | N/A | Capture test tooling needs renderer camera setters for headless validation. Active capture tool, not stale app usage. | `cargo check -p renderer --examples` | Document as active capture tooling (not deprecated app path). | requires_senior_decision |

### Renderer Integration Tests

| file_line | candidate_api_or_symbol | observed_usage | classification | replacement_path | retained_compatibility_reason | validation_command | docs_spec_update | follow_up |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tests/integration.rs:66 | `CameraView` | Type existence compile-time contract | active_current_path | N/A | Tests active API surface contract. | `cargo test -p renderer` | none | none |

### Root Launcher (`src/runtime.rs`)

| file_line | candidate_api_or_symbol | observed_usage | classification | replacement_path | retained_compatibility_reason | validation_command | docs_spec_update | follow_up |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| runtime.rs:326 | `renderer.install_default_fps_input()` | Root launcher init — installs renderer-owned FPS input | intentional_compatibility_coverage | `engine::input::InputSystem` with app action layer + `FPSController` | Root launcher is generic project runner. Migration scope requires senior decision. | `cargo check` | If classified as compatibility, document in launcher docs. | requires_senior_decision |
| runtime.rs:360 | `renderer.update_input(&window, &event)` | Root launcher event loop — renderer-owned input dispatch | intentional_compatibility_coverage | `engine::input::route_platform_input_to_app` + app-owned `InputSystem` | Launcher uses compatibility path. Migration may affect project-manifest behavior. | `cargo check` | Document as compatibility launcher path. | requires_senior_decision |
| runtime.rs:423 | `renderer.render_scene(&window, &mut scene)` | Root launcher render — one-shot compatibility render | intentional_compatibility_coverage | `render_scene_with_view` with launcher-managed `CameraView` | Launcher uses compatibility path. | `cargo check` | Document as compatibility. | requires_senior_decision |

### Marching Terrain (`apps/marching_terrain/`)

| file_line | candidate_api_or_symbol | observed_usage | classification | replacement_path | retained_compatibility_reason | validation_command | docs_spec_update | follow_up |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| src/main.rs:88 | `renderer.install_default_fps_input()` | Marching terrain windowed — installs renderer-owned FPS input | intentional_compatibility_coverage | App-owned `InputSystem` + `FPSController` | Marching terrain is app code with its own validation risk. Default classification per rules; migration needs senior decision. | `cargo check -p marching_terrain` | Update marching_terrain docs if migrated. | requires_senior_decision |
| src/main.rs:140 | `renderer.set_camera_position(camera_pos)` | Marching terrain windowed — sets initial camera position | intentional_compatibility_coverage | App-owned `Camera` positioned before `CameraView` construction | Camera setter used for initial placement. | `cargo check -p marching_terrain` | Update docs if migrated. | requires_senior_decision |
| src/main.rs:172 | `renderer.update_input(&window, &event)` | Marching terrain windowed event loop — renderer-owned input dispatch | intentional_compatibility_coverage | `engine::input::route_platform_input_to_app` + app-owned `InputSystem` | Uses compatibility path. | `cargo check -p marching_terrain` | Update docs if migrated. | requires_senior_decision |
| src/main.rs:249 | `renderer.input()` | Marching terrain windowed — reads renderer-owned `InputSystem` snapshot for walk-mode movement | intentional_compatibility_coverage | App-owned `InputSystem::snapshot()` | Compatibility input access. | `cargo check -p marching_terrain` | Update docs if migrated. | requires_senior_decision |
| src/main.rs:287 | `renderer.set_camera_position(...)` | Marching terrain windowed — syncs camera from player state each frame | intentional_compatibility_coverage | App-owned `Camera::set_position` before `CameraView` | Compatibility camera management. | `cargo check -p marching_terrain` | Update docs if migrated. | requires_senior_decision |
| src/main.rs:294 | `renderer.render_scene(&window, &mut scene)` | Marching terrain windowed — one-shot compatibility render | intentional_compatibility_coverage | `render_scene_with_view` with caller-provided `CameraView` | Compatibility render path. | `cargo check -p marching_terrain` | Update docs if migrated. | requires_senior_decision |
| src/capture.rs:90 | `renderer.install_default_fps_input()` | Marching terrain headless capture — installs renderer-owned FPS input | intentional_compatibility_coverage | App-owned path for capture (low priority) | Headless capture init. | `cargo check -p marching_terrain` | Update docs if migrated. | requires_senior_decision |
| src/capture.rs:119 | `renderer.set_camera_position(preset.camera_position)` | Marching terrain headless capture — sets capture camera from preset | intentional_compatibility_coverage | `CameraView` construction for headless capture | Camera setter for headless validation. | `cargo check -p marching_terrain` | Update docs if migrated. | requires_senior_decision |
| src/capture.rs:200 | `"applied_camera_api": "renderer.set_camera_position"` | Sidecar JSON documenting camera API used | intentional_compatibility_coverage | Update sidecar label if migrated | Documentation of current method. | N/A (JSON) | Update if migrated. | requires_senior_decision |

### Dungeon Dogfood — App-Owned Path (`apps/dungeon_dogfood/`)

| file_line | candidate_api_or_symbol | observed_usage | classification | replacement_path | retained_compatibility_reason | validation_command | docs_spec_update | follow_up |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| src/main.rs:16 | `FPSController`, `Camera` | Import from `engine::camera` (root facade re-export) | active_current_path | N/A | Dogfood uses `FPSController` app-owned. Active proof path. | `cargo check -p dungeon_dogfood` | none | none |
| src/main.rs:318 | `engine::input::route_platform_input_to_app(...)` | Dogfood event loop — app-owned input routing | active_current_path | N/A | Active app-owned path. Dogfood does NOT use `renderer.update_input`. | `cargo check -p dungeon_dogfood`, `cargo test -p dungeon_dogfood` | none | none |
| src/main.rs:425 | `install_app_fps_input` → returns `FPSController` | Dogfood installs FPS action layer into app-owned `InputSystem`, returns app-owned `FPSController` | active_current_path | N/A | App-owned input installation. Does NOT use `renderer.install_default_fps_input`. | `cargo check -p dungeon_dogfood` | none | none |
| src/main.rs:439 | `FPSController::new(0.002, 1.0)` | Creates app-owned FPS controller | active_current_path | N/A | App-owned controller instance. | `cargo check -p dungeon_dogfood` | none | none |
| src/main.rs:459 | `fps_controller: &mut FPSController` | Parameter in `render_frame` | active_current_path | N/A | App-owned controller passed to render frame. | `cargo check -p dungeon_dogfood` | none | none |
| src/main.rs:506 | `renderer.render_scene_headless_with_view(scene, view)?` | Dogfood headless capture render — caller-provided `CameraView` | active_current_path | N/A | Active app-owned render path. | `cargo check -p dungeon_dogfood`, dogfood headless draw capture | none | none |
| src/main.rs:508 | `renderer.render_scene_with_view(scene, view)?` | Dogfood windowed render — caller-provided `CameraView` | active_current_path | N/A | Active app-owned render path. | `cargo check -p dungeon_dogfood`, windowed smoke | none | none |
| src/game_state.rs:27 | `CameraView` | Comment: "Update app-owned camera before building a renderer CameraView" | active_current_path | N/A | Code comment documenting app-owned pattern. | `cargo check -p dungeon_dogfood` | none | none |

### Root Facade Helpers (`src/` — engine crate)

| file_line | candidate_api_or_symbol | observed_usage | classification | replacement_path | retained_compatibility_reason | validation_command | docs_spec_update | follow_up |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| input.rs:117 | `route_platform_input_to_app` | Public helper definition — routes platform input to app-owned `InputSystem` | active_current_path | N/A | Active app-owned helper. | `cargo test -p engine` | none | none |
| input.rs:123 | `renderer.route_platform_input(window, event)?` | Internal: delegates platform side effects to renderer | active_current_path | N/A | Uses current renderer primitive. | `cargo test -p engine` | none | none |
| input.rs:124 | `queue_routed_input_event(...)` | Internal: queues uncaptured input into app-owned system | active_current_path | N/A | Active path. | `cargo test -p engine` | none | none |
| input.rs:133 | `queue_routed_input_event` | Public helper definition — low-level queueing | active_current_path | N/A | Active low-level helper. | `cargo test -p engine` | none | none |
| render.rs:11 | `CameraView` | Import from renderer facade | active_current_path | N/A | Root helper re-exports `CameraView`. | `cargo test -p engine` | none | none |
| render.rs:35 | `camera_view_for_size` | Public helper definition | active_current_path | N/A | Active app-owned helper. Used by dogfood. | `cargo test -p engine` | none | none |
| render.rs:42 | `CameraView::from_camera(camera, aspect)` | Internal: constructs `CameraView` from app camera | active_current_path | N/A | Active path. | `cargo test -p engine` | none | none |
| render.rs:55,65,75 | `CameraView::from_camera` | Test code in `camera_view_for_size` tests | active_current_path | N/A | Test of active API. | `cargo test -p engine` | none | none |
| camera.rs:3 | `FPSController` | Re-export from renderer crate | active_current_path | N/A | Root facade re-exports active type. | `cargo check -p engine` | none | none |
| lib.rs:16 | `FPSController` | Prelude re-export | active_current_path | N/A | Active type in root prelude. | `cargo check -p engine` | none | none |
| lib.rs:25 | `queue_routed_input_event`, `route_platform_input_to_app`, `InputActionEventEmitter` | Public re-exports | active_current_path | N/A | Active app-owned helpers. | `cargo test -p engine` | none | none |
| lib.rs:29 | `camera_view_for_size`, `CameraView` | Public re-exports | active_current_path | N/A | Active app-owned helpers. | `cargo test -p engine` | none | none |

### Engine Tests

| file_line | candidate_api_or_symbol | observed_usage | classification | replacement_path | retained_compatibility_reason | validation_command | docs_spec_update | follow_up |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tests/facade_imports.rs:23 | `route_platform_input_to_app` | Compile-time import check | active_current_path | N/A | Tests active API surface. | `cargo test -p engine` | none | none |
| tests/facade_imports.rs:37 | `FPSController` | Compile-time import check | active_current_path | N/A | Tests active type. | `cargo test -p engine` | none | none |
| tests/facade_imports.rs:44 | `CameraView` | Compile-time import check | active_current_path | N/A | Tests active type. | `cargo test -p engine` | none | none |
| tests/facade_imports.rs:45 | `camera_view_for_size` | Compile-time import check | active_current_path | N/A | Tests active helper. | `cargo test -p engine` | none | none |
| tests/facade_imports.rs:61 | `raw_renderer::CameraView` | Compile-time import check (direct renderer import) | active_current_path | N/A | Tests re-export integrity. | `cargo test -p engine` | none | none |
| tests/facade_imports.rs:101 | `CameraView`, `camera_view_for_size` | Compile-time import check (prelude path) | active_current_path | N/A | Tests prelude path. | `cargo test -p engine` | none | none |
| tests/facade_imports.rs:107 | `route_platform_input_to_app` | Compile-time import check (prelude path) | active_current_path | N/A | Tests prelude path. | `cargo test -p engine` | none | none |
| tests/facade_imports.rs:124 | `CameraView` | Type usage in test helper | active_current_path | N/A | Tests active type. | `cargo test -p engine` | none | none |
| tests/facade_imports.rs:128 | `FPSController` | Type usage in test helper | active_current_path | N/A | Tests active type. | `cargo test -p engine` | none | none |
| tests/input_action_events.rs:5 | `queue_routed_input_event` | Import for tests | active_current_path | N/A | Tests active helper. | `cargo test -p engine` | none | none |
| tests/input_action_events.rs:187-244 | `queue_routed_input_event(...)` | Test functions (6 occurrences) | active_current_path | N/A | Tests active API. | `cargo test -p engine` | none | none |

### Specifications (`.internal-dev/specifications/`)

| file_line | candidate_api_or_symbol | observed_usage | classification | replacement_path | retained_compatibility_reason | validation_command | docs_spec_update | follow_up |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| decisions.md:19 | `CameraView` | Decision record — keep in renderer facade, re-export through root | active_current_path | N/A | Spec is correct; describes desired state. | N/A (spec) | none | none |
| api.md:19 | `Renderer::set_camera_look_at` | API spec — marks as active facade | active_current_path | N/A | Spec labels this as active. Per current-state-analysis, this is a narrow facade over existing behavior. | N/A (spec) | none | Verify classification against code: `set_camera_look_at` is compatibility per this matrix; spec may need update. |
| api.md:20 | `Renderer::update_input`, `route_platform_input`, `queue_routed_input_event` | API spec — marks app-owned bridge as active, `update_input` as compatibility | active_current_path | N/A | Spec correctly distinguishes. | N/A (spec) | none | none |
| api.md:22 | `CameraView` APIs | API spec — dogfood proof path | active_current_path | N/A | Spec correctly describes app-owned path. | N/A (spec) | none | none |
| api.md:24 | `CameraView` | API spec — renderer-owned DTO | active_current_path | N/A | Spec correct. | N/A (spec) | none | none |
| api.md:25 | `route_platform_input_to_app`, `queue_routed_input_event`, `camera_view_for_size` | API spec — core app-owned loop primitives | active_current_path | N/A | Spec correct. | N/A (spec) | none | none |
| services.md:21 | `CameraView` | Service spec — renderer caller-view service | active_current_path | N/A | Spec correct. | N/A (spec) | none | none |
| services.md:22 | `route_platform_input_to_app`, `queue_routed_input_event`, `camera_view_for_size` | Service spec — root app workflow helpers | active_current_path | N/A | Spec correct. | N/A (spec) | none | none |
| service-graph.md:21 | `CameraView` | Service graph — dependency direction | active_current_path | N/A | Spec correct. | N/A (spec) | none | none |
| architecture.md:21 | `CameraView` | Architecture spec — render view boundary | active_current_path | N/A | Spec correct. | N/A (spec) | none | none |

### Documentation — Presenting Renderer-Owned Loops as Current or Default

| file_line | candidate_api_or_symbol | observed_usage | classification | replacement_path | retained_compatibility_reason | validation_command | docs_spec_update | follow_up |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| docs/api/00-index.md:75-76 | `FPSController`, `CameraView` | API index — correctly lists types in current/extension points | active_current_path | N/A | Correct index entry. | N/A (docs) | none | none |
| docs/api/00-index.md:112 | `renderer.update_input()`, `renderer.render_scene()` | Quickstart steps presenting renderer-owned loop as primary | stale_documentation | App-owned path via `route_platform_input_to_app` + `render_scene_with_view` | This doc section presents compatibility path as default. Should be labeled compatibility or point to app-owned quickstart. | `cargo check` (doc examples) | Phase 01: Add compatibility label, link to app-owned loop doc. | Phase 01 |
| docs/api/00-index.md:120 | `renderer.update_input(&window, &event)?` | Quickstart example code | stale_documentation | `route_platform_input_to_app(renderer, window, input, event)` | Same as line 112. | `cargo check` | Phase 01 update. | Phase 01 |
| docs/api/00-index.md:123 | `renderer.render_scene(&window, &mut scene)?` | Quickstart example code | stale_documentation | `render_scene_with_view(scene, view)?` | Same as line 112. | `cargo check` | Phase 01 update. | Phase 01 |
| docs/api/01-quickstart.md:61 | `renderer.install_default_fps_input()` | Quickstart code example | stale_documentation | App-owned `InputSystem::add_layer` + `FPSController` | Old quickstart presents compatibility path as primary. | `cargo check` | Phase 01: Label as compatibility quickstart; add app-owned quickstart link. | Phase 01 |
| docs/api/01-quickstart.md:64 | `renderer.update_input(&window, &event)` | Quickstart code example | stale_documentation | `route_platform_input_to_app` | Same. | `cargo check` | Phase 01 update. | Phase 01 |
| docs/api/01-quickstart.md:82 | `renderer.render_scene(&window, &mut scene)?` | Quickstart code example | stale_documentation | `render_scene_with_view` | Same. | `cargo check` | Phase 01 update. | Phase 01 |
| docs/api/01-quickstart.md:93-95 | `update_input()`, `install_default_fps_input()`, `render_scene()` | Quickstart explanation text | stale_documentation | App-owned equivalents | Explanation presents compatibility APIs as primary. | N/A (docs) | Phase 01: add compatibility/app-owned distinction. | Phase 01 |
| docs/api/01-quickstart.md:108 | `update_input()` | Debug toggle note | stale_documentation | Note that `route_platform_input` also handles debug toggles | Doc mentions bypass caveat. Needs app-owned path note. | N/A (docs) | Phase 01 update. | Phase 01 |
| docs/api/01-student-quickstart.md:8 | `update_input(...)`, `render_scene(...)` | Student quickstart overview | stale_documentation | App-owned path | Compatibility path presented as primary. | N/A (docs) | Phase 01: add compatibility label + app-owned track. | Phase 01 |
| docs/api/01-student-quickstart.md:11 | `route_platform_input`, `CameraView` | Student quickstart — app-owned track overview | active_current_path | N/A | Correctly describes app-owned path. | N/A (docs) | none | none |
| docs/api/01-student-quickstart.md:26 | `render_scene(...)` | Quickstart comparison table | stale_documentation | `render_scene_with_view` | Table entry presents compatibility shortcut. | N/A (docs) | Phase 01: label compatibility. | Phase 01 |
| docs/api/01-student-quickstart.md:98 | `renderer.update_input(...)` | Quickstart code example | stale_documentation | `route_platform_input_to_app` | Old quickstart code. | `cargo check` | Phase 01 update. | Phase 01 |
| docs/api/01-student-quickstart.md:103 | `renderer.route_platform_input(...)` | Quickstart code example | active_current_path | N/A | Correct current path. | `cargo check` | none | none |
| docs/api/01-student-quickstart.md:107 | `CameraView` | Quickstart code example — render with CameraView | active_current_path | N/A | Correct current path. | `cargo check` | none | none |
| docs/api/02-renderer.md:24-27 | `render_scene()` | API reference — documents convenience method | stale_documentation | Add compatibility label | API reference is accurate but needs compatibility labeling. | N/A (docs) | Phase 01: add "Compatibility path" label. | Phase 01 |
| docs/api/02-renderer.md:60 | `render_scene()` | API reference — InvalidState note | stale_documentation | Keep but label compatibility | Reference docs. | N/A (docs) | Phase 01: label compatibility. | Phase 01 |
| docs/api/02-renderer.md:75-77 | `update_input`, `install_default_fps_input`, `uninstall_default_fps_input` | API reference — documents input methods | stale_documentation | Add compatibility labels | Reference docs for compatibility APIs. | N/A (docs) | Phase 01: add compatibility label. | Phase 01 |
| docs/api/02-renderer.md:82 | `update_input()` | API reference — describes debug toggle behavior | stale_documentation | Add note about route_platform_input handling same toggles | Reference docs. | N/A (docs) | Phase 01: add compatibility label + current-primitive note. | Phase 01 |
| docs/api/02-renderer.md:97 | `set_camera_position` | API reference | stale_documentation | Add compatibility label; note capture tooling use | Reference docs. | N/A (docs) | Phase 01: label as capture/compatibility tooling. | Phase 01 |
| docs/api/02-renderer-lifecycle-and-frame-api.md:8 | `update_input(...)`, `render_scene(...)` | Lifecycle doc — compatibility path overview | stale_documentation | Already describes both paths; needs clearer labeling | Doc describes both but doesn't clearly mark compatibility. | N/A (docs) | Phase 01: add explicit compatibility label. | Phase 01 |
| docs/api/02-renderer-lifecycle-and-frame-api.md:11 | `route_platform_input(...)`, `render_scene_with_view(...)` | Lifecycle doc — app-owned path overview | active_current_path | N/A | Correctly describes app-owned path. | N/A (docs) | none | none |
| docs/api/02-renderer-lifecycle-and-frame-api.md:19 | `render_scene(...)` | Lifecycle doc — compatibility shortcut | stale_documentation | Needs compatibility label | Describes shortcut without labeling. | N/A (docs) | Phase 01: label compatibility. | Phase 01 |
| docs/api/02-renderer-lifecycle-and-frame-api.md:21-22 | `render_scene_with_view(...)`, `render_scene_headless_with_view(...)`, `CameraView` | Lifecycle doc — app-owned path | active_current_path | N/A | Correct current path. | N/A (docs) | none | none |
| docs/api/02-renderer-lifecycle-and-frame-api.md:33-35 | `update_input(...)`, `route_platform_input(...)`, `queue_routed_input_event(...)` | Lifecycle doc — correctly distinguishes compatibility from app-owned | active_current_path | N/A | Already correctly labeled. | N/A (docs) | none | none |
| docs/api/02-renderer-lifecycle-and-frame-api.md:37-39 | `render_scene_with_view(...)`, `begin_app_frame(...)`, `update_input(...)` | Lifecycle doc — app-owned path recommendations | active_current_path | N/A | Correctly recommends app-owned path. | N/A (docs) | none | none |
| docs/api/02-renderer-lifecycle-and-frame-api.md:70 | `renderer.render_scene(&window, &mut scene)?` | Lifecycle doc — compatibility code example | stale_documentation | `render_scene_with_view` | Example code for compatibility path. Needs label. | `cargo check` | Phase 01: label compatibility. | Phase 01 |
| docs/api/02-renderer-lifecycle-and-frame-api.md:105 | `route_platform_input + queue_routed_input_event` | Lifecycle doc — app-owned input path | active_current_path | N/A | Correct current path. | N/A (docs) | none | none |
| docs/api/02-renderer-lifecycle-and-frame-api.md:122 | `render_scene(...)` | Lifecycle doc — InvalidState note | stale_documentation | Keep but label | Reference note. | N/A (docs) | Phase 01: label compatibility. | Phase 01 |
| docs/api/06-input.md:41 | `renderer.update_input()`, `renderer.input()`, `renderer.input_mut()` | Input doc — presents renderer-owned input as primary | stale_documentation | App-owned `InputSystem` + `route_platform_input_to_app` | Doc section presents compatibility path as primary. | N/A (docs) | Phase 01: add compatibility label + app-owned section. | Phase 01 |
| docs/api/06-input.md:167 | `renderer.input_mut().add_layer(...)` | Input doc — code example | stale_documentation | App-owned `input.add_layer(...)` | Code example for compatibility path. | `cargo check` | Phase 01: label compatibility. | Phase 01 |
| docs/api/06-input-polling-and-listeners.md:8 | `Renderer::update_input(...)`, `render_scene(...)` | Input doc — compatibility path overview | stale_documentation | Needs compatibility label | Doc section header for compatibility track. | N/A (docs) | Phase 01: add explicit compatibility label. | Phase 01 |
| docs/api/06-input-polling-and-listeners.md:14 | `Renderer::route_platform_input(...)`, `queue_routed_input_event(...)` | Input doc — app-owned path overview | active_current_path | N/A | Correctly describes app-owned path. | N/A (docs) | none | none |
| docs/api/06-input-polling-and-listeners.md:29 | `Renderer::install_default_fps_input()`, `FPSController`, `CameraView` | Input doc — camera controls section. Partially correct: distinguishes compatibility FPS from app-owned, but needs labeling | stale_documentation | App-owned `InputSystem::add_layer` + `FPSController` | Section describes both but defaults to compatibility. | N/A (docs) | Phase 01: improve labeling. | Phase 01 |
| docs/api/06-input-polling-and-listeners.md:42 | `renderer.install_default_fps_input()` | Input doc — compatibility code example | stale_documentation | App-owned layer installation | Compatibility example code. | `cargo check` | Phase 01: label compatibility. | Phase 01 |
| docs/api/06-input-polling-and-listeners.md:49 | `renderer.input_mut().add_layer(...)` | Input doc — compatibility code example | stale_documentation | App-owned `input.add_layer(...)` | Compatibility example code. | `cargo check` | Phase 01: label compatibility. | Phase 01 |
| docs/api/06-input-polling-and-listeners.md:60 | `renderer.update_input(&window, &event)?` | Input doc — compatibility code example | stale_documentation | `route_platform_input_to_app` | Compatibility example code. | `cargo check` | Phase 01: label compatibility. | Phase 01 |
| docs/api/06-input-polling-and-listeners.md:66 | `renderer.input().snapshot()` | Input doc — compatibility code example | stale_documentation | App-owned `input.snapshot()` | Compatibility example code. | `cargo check` | Phase 01: label compatibility. | Phase 01 |
| docs/api/06-input-polling-and-listeners.md:100-104 | `update_input(...)`, `route_platform_input(...)`, `queue_routed_input_event(...)`, `CameraView` | Input doc — recommendations section | active_current_path | N/A | Correctly recommends both paths with preference. | N/A (docs) | Minor: could strengthen app-owned preference. | Phase 01 |
| docs/api/06-input-polling-and-listeners.md:134-135 | `renderer.update_input(...)`, `renderer.input().debug_snapshot()` | Input doc — validation steps | stale_documentation | App-owned equivalents | Validation steps reference compatibility path. | N/A (docs) | Phase 01: add app-owned validation steps. | Phase 01 |
| docs/api/12-events-and-lifecycle.md:12 | `Renderer::update_input(...)`, `Renderer::render_scene(...)` | Events doc — compatibility path diagram | stale_documentation | Needs compatibility label | Doc presents both paths; this line shows compatibility. | N/A (docs) | Phase 01: add explicit compatibility label. | Phase 01 |
| docs/api/12-events-and-lifecycle.md:15 | `route_platform_input_to_app(...)`, `begin_app_frame(...)`, `render_scene_with_view(...)`, `end_app_frame(...)` | Events doc — app-owned path diagram | active_current_path | N/A | Correctly describes app-owned path. | N/A (docs) | none | none |
| docs/api/12-events-and-lifecycle.md:62 | `renderer.events_mut().subscribe(...)` | Events doc — code example using renderer-owned event bus | stale_documentation | App-owned `EventBus` + `RuntimeEventDispatcher` | Example code uses compatibility event bus. | `cargo check` | Phase 01: label compatibility; add app-owned example. | Phase 01 |
| docs/api/12-events-and-lifecycle.md:87 | `CameraView`, `begin_app_frame`, `end_app_frame` | Events doc — dogfood app-owned reference | active_current_path | N/A | Correctly describes dogfood path. | N/A (docs) | none | none |
| docs/api/12-events-and-lifecycle.md:131 | `render_scene_with_view` | Events doc — renderer no-dispatch render | active_current_path | N/A | Correct current path. | N/A (docs) | none | none |
| docs/api/12-events-and-lifecycle.md:159-160 | `renderer.update_input(...)`, `route_platform_input_to_app(...)`, `render_scene`, `render_scene_with_view` | Events doc — validation steps (both paths) | active_current_path | N/A | Validation steps cover both correct paths. | N/A (docs) | none | none |
| docs/api/14-dogfood-vertical-slice.md:73 | `FPSController` | Dogfood docs — architecture diagram showing app-owned FPSController | active_current_path | N/A | Correctly documents dogfood app-owned path. | N/A (docs) | none | none |
| docs/api/14-dogfood-vertical-slice.md:81 | `route_platform_input_to_app` | Dogfood docs — input routing diagram | active_current_path | N/A | Correctly documents app-owned path. | N/A (docs) | none | none |
| docs/api/14-dogfood-vertical-slice.md:85 | `render_scene_with_view`, `render_scene_headless_with_view` | Dogfood docs — render path diagram | active_current_path | N/A | Correctly documents app-owned path. | N/A (docs) | none | none |
| docs/api/14-dogfood-vertical-slice.md:90 | `route_platform_input_to_app`, `begin_app_frame`, `end_app_frame`, `camera_view_for_size`, `renderer.events_mut()` | Dogfood docs — notes dogfood does NOT use renderer-owned paths | active_current_path | N/A | Correctly documents dogfood's active path. The mention of `renderer.events_mut()` is in a negative ("does not use") context. | N/A (docs) | none | none |
| docs/api/15-app-owned-loop.md:5 | `Renderer::update_input(...)`, `Renderer::render_scene(...)` | App-owned loop doc — mentions compatibility path for context | active_current_path | N/A | Mentions compatibility for context; correctly focuses on app-owned path. | N/A (docs) | none | none |
| docs/api/15-app-owned-loop.md:15 | `route_platform_input_to_app`, `InputActionEventEmitter`, `InputSystem` | App-owned loop doc — imports | active_current_path | N/A | Correct app-owned path imports. | N/A (docs) | none | none |
| docs/api/15-app-owned-loop.md:36 | `route_platform_input_to_app(...)` | App-owned loop doc — code example | active_current_path | N/A | Correct app-owned code. | `cargo check` | none | none |
| docs/api/15-app-owned-loop.md:70 | `renderer.render_scene_with_view(&mut scene, view)` | App-owned loop doc — code example | active_current_path | N/A | Correct app-owned render call. | `cargo check` | none | none |
| docs/api/15-app-owned-loop.md:98 | `route_platform_input_to_app` | App-owned loop doc — explanation | active_current_path | N/A | Correct explanation. | N/A (docs) | none | none |
| docs/api/15-app-owned-loop.md:101 | `camera_view_for_size`, `CameraView` | App-owned loop doc — explanation | active_current_path | N/A | Correct explanation. | N/A (docs) | none | none |
| docs/api/05-render-hooks-and-extension-points.md:8 | `Renderer::render_scene(...)` | Render hooks doc — pipeline diagram | stale_documentation | Add compatibility label | Documentation of render pipeline using compatibility API name. | N/A (docs) | Phase 01: label compatibility in diagram; note that same pipeline serves `render_scene_with_view`. | Phase 01 |
| docs/api/08-debug.md:17 | `Renderer::update_input()` | Debug doc — mentions update_input handles debug toggles | stale_documentation | Add note that `route_platform_input` also handles debug toggles | Doc accurately describes toggle behavior but references old API. | N/A (docs) | Phase 01: add note about `route_platform_input` also handling toggles. | Phase 01 |

### Documentation — Internal Implementation Docs (Correctly Labeled or Archival)

| file_line | candidate_api_or_symbol | observed_usage | classification | replacement_path | retained_compatibility_reason | validation_command | docs_spec_update | follow_up |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| docs/internal/01-architecture.md:36 | `update_input()` | Internal arch doc — describes compatibility path | active_current_path | N/A | Already labeled "Compatibility renderer-owned path". Correct. | N/A (docs) | none | none |
| docs/internal/01-architecture.md:37 | `route_platform_input()` | Internal arch doc — describes app-owned path | active_current_path | N/A | Already labeled "App-owned path". Correct. | N/A (docs) | none | none |
| docs/internal/01-architecture.md:38 | `render_scene()` | Internal arch doc — describes compatibility render | active_current_path | N/A | Documented as part of legacy path alongside explicit frame trio. | N/A (docs) | none | none |
| docs/internal/01-architecture.md:51-52 | `CameraView`, `render_scene_with_view(...)`, `render_scene_headless_with_view(...)` | Internal arch doc — app-owned camera path | active_current_path | N/A | Correctly documents app-owned path. | N/A (docs) | none | none |
| docs/internal/01-rendering-pipeline-mental-model.md:8 | `Renderer::render_scene(...)` | Internal pipeline doc | stale_documentation | Could cross-reference `render_scene_with_view` but acceptable as internal pipeline reference | Internal doc describing pipeline. Uses compatibility API name as entry point. | N/A (docs) | Phase 01: add note about view-based path. | Phase 01 |
| docs/internal/03-asset-pipeline.md:96 | `render_scene()` | Internal asset doc | stale_documentation | Cross-reference with view-based render | Describes automatic polling by `render_scene()`. | N/A (docs) | Phase 01: add note that `render_scene_with_view` also polls. | Phase 01 |
| docs/internal/04-api-to-backend-handoff.md:8 | `Renderer::render_scene(...)` | Internal handoff doc — compatibility path | active_current_path | N/A | Already shows both paths. This line is compatibility diagram. | N/A (docs) | none | none |
| docs/internal/04-api-to-backend-handoff.md:11 | `CameraView`, `Renderer::render_scene_with_view(...)` | Internal handoff doc — app-owned path | active_current_path | N/A | Correctly shows app-owned path. | N/A (docs) | none | none |
| docs/internal/04-api-to-backend-handoff.md:15 | `CameraView` | Internal handoff doc — DTO description | active_current_path | N/A | Correct description. | N/A (docs) | none | none |
| docs/internal/04-api-to-backend-handoff.md:41-42 | `CameraView::from_camera`, `renderer.render_scene_with_view(&mut scene, view)` | Internal handoff doc — code example | active_current_path | N/A | Correct app-owned example. | `cargo check` | none | none |
| docs/internal/04-api-to-backend-handoff.md:90 | `CameraView` | Internal handoff doc — boundary table | active_current_path | N/A | Correctly defines boundary. | N/A (docs) | none | none |
| docs/internal/05-vulkan-sync-and-frame-lifecycle.md:8 | `Renderer::render_scene(...)` | Internal Vulkan doc | stale_documentation | Could cross-reference view-based path | Internal pipeline reference. | N/A (docs) | Phase 01: cross-reference. | Phase 01 |
| docs/internal/07-rendergraph-dependencies-and-aliasing.md:8 | `Renderer::render_scene(...)` | Internal rendergraph doc | stale_documentation | Could cross-reference view-based path | Internal pipeline reference. | N/A (docs) | Phase 01: cross-reference. | Phase 01 |
| docs/internal/08-scene-flattening-and-culling.md:8 | `Renderer::render_scene(...)` | Internal scene doc | stale_documentation | Could cross-reference view-based path | Internal pipeline reference. | N/A (docs) | Phase 01: cross-reference. | Phase 01 |
| docs/internal/09-input-winit-integration.md:8 | `Renderer::update_input(...)` | Internal input doc — compatibility path | active_current_path | N/A | Already shows both compatibility and app-owned paths. | N/A (docs) | none | none |
| docs/internal/09-input-winit-integration.md:11 | `route_platform_input_to_app(...)`, `route_platform_input(...)`, `queue_routed_input_event(...)`, `begin_app_frame(...)` | Internal input doc — app-owned path | active_current_path | N/A | Correctly documents app-owned path. | N/A (docs) | none | none |
| docs/internal/09-input-winit-integration.md:23-24 | `route_platform_input_to_app`, `queue_routed_input_event` | Internal input doc — guidance | active_current_path | N/A | Correct guidance. | N/A (docs) | none | none |
| docs/internal/09-input-winit-integration.md:33 | `route_platform_input` | Internal input doc — API signature documentation | active_current_path | N/A | Documents current API. | N/A (docs) | none | none |
| docs/internal/09-input-winit-integration.md:78-79 | `route_platform_input_to_app(renderer, window, app_input, event)`, `renderer.route_platform_input + queue_routed_input_event` | Internal input doc — usage examples | active_current_path | N/A | Correct current path examples. | N/A (docs) | none | none |
| docs/internal/09-input-winit-integration.md:86 | `CameraView` | Internal input doc — app builds CameraView | active_current_path | N/A | Correct app-owned pattern. | N/A (docs) | none | none |
| docs/internal/09-input-winit-integration.md:97-98 | `Renderer::route_platform_input`, `route_platform_input_to_app`, `queue_routed_input_event` | Internal input doc — recommendations | active_current_path | N/A | Correct recommendations. | N/A (docs) | none | none |
| docs/internal/09-input-winit-integration.md:113 | `update_input`, `route_platform_input_to_app`, `route_platform_input`, `queue_routed_input_event` | Internal input doc — validation steps | active_current_path | N/A | Correct validation for both paths. | N/A (docs) | none | none |
| docs/internal/10-event-system-and-lifecycle.md:81 | `route_platform_input_to_app` | Internal events doc | active_current_path | N/A | Correct current path. | N/A (docs) | none | none |
| docs/internal/12-audio-foundation.md:63 | `renderer.events_mut()` | Internal audio doc — says dogfood bridge emits through `renderer.events_mut()` | stale_documentation | App-owned `EventBus` (dogfood already uses this) | Doc is stale: dogfood no longer uses `renderer.events_mut()` for audio bridge. Dogfood uses app-owned `EventBus`. | N/A (docs) | Phase 01: update to reflect dogfood's app-owned EventBus. | Phase 01 |

### Renderer Internal Developer Docs

| file_line | candidate_api_or_symbol | observed_usage | classification | replacement_path | retained_compatibility_reason | validation_command | docs_spec_update | follow_up |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| api/.developer-documentation.md:32 | `self.update_input()` | Pseudo-code in internal developer doc | stale_documentation | `self.route_platform_input(...)` + queueing | Internal dev doc uses old API name in illustrative pseudo-code. | N/A (internal dev doc) | Update pseudo-code to reference current primitives. | Phase 01 |

### Dogfood README

| file_line | candidate_api_or_symbol | observed_usage | classification | replacement_path | retained_compatibility_reason | validation_command | docs_spec_update | follow_up |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| apps/dungeon_dogfood/README.md:17 | `route_platform_input_to_app`, `begin_app_frame`, `end_app_frame`, `camera_view_for_size` | Dogfood README — correctly describes app-owned path | active_current_path | N/A | Correctly describes dogfood's active path. | N/A (docs) | none | none |

### Captured but Not Mission-Critical (scene-level API, not renderer input/camera)

| file_line | candidate_api_or_symbol | observed_usage | classification | replacement_path | retained_compatibility_reason | validation_command | docs_spec_update | follow_up |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| capture_tests/common.rs:288 | `scene.set_camera(view, projection, eye)` | Scene-level camera metadata for submission | active_current_path | N/A | Scene camera is a different concern — sets submission metadata, not renderer-owned camera state. Not covered by this migration plan. | N/A | none | none |

---

## Senior Decision Table

| # | item | context | affected surfaces | risk if deferred | recommendation |
| --- | --- | --- | --- | --- | --- |
| SD-01 | Root launcher (`src/runtime.rs`) migration scope | Launcher uses `install_default_fps_input`, `update_input`, and `render_scene`. It is the generic project runner. Migration would mean launcher owns `InputSystem`, `EventBus`, `Camera`, `FrameClock`, and uses app-owned helpers. | `src/runtime.rs`, any project-manifest-driven launch paths | Low. Launcher can remain compatibility indefinitely; migration is a product decision. | Keep as `intentional_compatibility_coverage` unless app-owned launcher is a product requirement. |
| SD-02 | Marching terrain migration scope | Marching terrain uses renderer-owned input, camera setters, and `render_scene` in both windowed and capture modes. Migration could blend API cleanup with terrain/capture correctness risks. | `apps/marching_terrain/src/main.rs`, `apps/marching_terrain/src/capture.rs` | Medium. Terrain has known bugs; migration may compound. | Defer to separate workflow unless senior explicitly includes in this plan. |
| SD-03 | Renderer example deprecation warning policy | If `update_input` / `install_default_fps_input` / `render_scene` are marked `#[deprecated]`, all renderer examples will emit warnings, creating noise in `cargo check -p renderer --examples`. | All renderer examples (`demo_pbr`, `demo_unlit`, `demo_model_load`, `api_test`, `demo_async_loading`), common module | High. Warning noise could mask real issues. | Do NOT deprecate yet. Keep as `intentional_compatibility_coverage` with clear docs labeling. Only deprecate after warning suppression policy is accepted. |
| SD-04 | `set_camera_position` / `set_camera_look_at` status | These APIs are used by marching_terrain (compatibility), capture_tests (active renderer tooling), and internally for compatibility render path. Are they active capture tools or compatibility? | `renderer.rs:1265-1273`, `capture_tests/common.rs:283`, `marching_terrain` | Medium. Removing would break capture tests and marching_terrain. | Keep as active capture/demo tooling. Label in docs as "capture/compatibility camera setters". |
| SD-05 | `CameraView::from_camera` internal usage at renderer.rs:1092 | Used in renderer's compatibility `render_scene` path to build a `CameraView` from renderer-owned camera. If renderer-owned camera is removed, this call goes away. | `renderer.rs:1092` | Low. Internal implementation detail. | Remove only if renderer-owned camera path is removed. No action needed now. |
| SD-06 | `events_mut()` in `event_logging.rs` | Internal renderer debugging helper subscribes to renderer-owned event bus. Is this active tooling or compatibility? | `event_logging.rs:15` | Low. Internal-only. | Keep as active internal renderer tooling. Not a migration concern. |
| SD-07 | Spec vs code mismatch: `api.md` labels `set_camera_look_at` as active facade | The spec says `set_camera_look_at` is active; this matrix classifies it as `internal_compatibility_implementation`. Which is correct? | `api.md:19`, `renderer.rs:1273` | Low. Spec may need update. | Align spec with matrix: `set_camera_look_at` is active capture/compatibility tooling, not primary app path. |
| SD-08 | Doc split: which quickstart stays compatibility vs gets app-owned update? | `01-quickstart.md` and `01-student-quickstart.md` both present compatibility path as primary. Phase 01 should add compatibility labels and app-owned quickstart references. | `docs/api/01-quickstart.md`, `docs/api/01-student-quickstart.md`, `docs/api/00-index.md` | Medium. New users may follow stale guidance. | Phase 01 should: (1) label existing quickstarts as "compatibility quickstart", (2) ensure app-owned loop doc (`15-app-owned-loop.md`) is prominently linked, (3) update index to lead with app-owned path. |

---

## Summary

### Classification Counts

| classification | count | notes |
| --- | --- | --- |
| `active_current_path` | 72 | Dogfood (8), root facade helpers (12), tests (11), specs (10), internal arch docs correctly labeled (16), CameraView/FPSController type definitions (6), renderer route/render_with_view APIs (4), renderer prelude/re-exports (5) |
| `intentional_compatibility_coverage` | 19 | Renderer examples (9: common/mod.rs ×3, demo_async_loading ×3, api_test ×2, capture_tests ×1), root launcher (3), marching_terrain (9) |
| `internal_compatibility_implementation` | 14 | Renderer API definitions for old APIs (update_input, install_default_fps_input, uninstall_default_fps_input, render_scene, set_camera_position, set_camera_look_at, input, input_mut, events, events_mut, internal calls, event_logging) |
| `stale_documentation` | 33 | Old quickstart docs (13), API reference needing labeling (7), input docs compatibility sections (10), internal pipeline docs (5), audio foundation stale reference (1), internal dev doc (1) |
| `not_yet_migrated_app_code` | 0 | No app code found that uses old APIs unintentionally. Marching terrain is classified as compatibility per rules. |
| `requires_senior_decision` | 8 (table rows) | Launcher, marching_terrain, warning policy, camera setter status, CameraView internal, events_mut, spec mismatch, doc split |

### Dogfood App-Owned Path (Separate Call-Out)

`apps/dungeon_dogfood/` is the only app-owned proof path. All 8 matches are `active_current_path`:

- Uses `engine::input::route_platform_input_to_app` (not `renderer.update_input`)
- Installs FPS input into app-owned `InputSystem` (not `renderer.install_default_fps_input`)
- Creates app-owned `FPSController` (type is current)
- Renders with `render_scene_with_view` / `render_scene_headless_with_view` + `CameraView`
- Uses `engine::frame::begin_app_frame` / `end_app_frame`
- Uses `engine::render::camera_view_for_size`
- Does NOT use `renderer.input()`, `renderer.events()`, `renderer.events_mut()`, `renderer.set_camera_*`, or `renderer.render_scene()`

**Validation gate:** `cargo check -p dungeon_dogfood` and `cargo test -p dungeon_dogfood` pass.

### Dependency Boundary Verification

No forbidden edges found:
- Renderer does not depend on root `engine` — confirmed by design (renderer.rs imports only from its own crate and support crates)
- Input/events/audio/physics/scripting crates do not depend on root `engine`
- All app-owned helpers in `src/input.rs`, `src/render.rs`, `src/camera.rs` consume renderer types but renderer never calls back into root

---

## Review Gate

### Inventory Completeness

- ✅ All grep matches from discovery command have been classified (124+ detail rows)
- ✅ Every candidate API family from template has a seed row classification
- ✅ Dogfood app-owned path is separately verified and confirmed active
- ✅ Renderer examples, root launcher, marching_terrain default to `intentional_compatibility_coverage`
- ✅ Dependency boundary verified: no renderer/support → root `engine`
- ✅ No code was edited outside the plan directory
- ✅ 8 senior decision items documented with context and recommendations

### Open Items Requiring Phase 01+

- 33 stale documentation rows need Phase 01 compatibility labeling
- 8 senior decisions need resolution before Phase 02/03
- `not_yet_migrated_app_code` count is 0 — no forced migration targets; all migration is by senior choice
- Phase 03 deprecation/removal cannot proceed without SD-03 (warning policy) resolution

### Ready for Senior Review

The inventory matrix is complete and ready for senior review. All concrete grep matches are classified. The matrix correctly distinguishes:
1. App-owned active path (dogfood + root helpers) — green
2. Intentional compatibility coverage (examples, launcher, marching_terrain) — yellow
3. Internal compatibility implementation (renderer API definitions) — blue
4. Stale documentation needing labeling — orange
5. Senior decisions requiring explicit approval — red
