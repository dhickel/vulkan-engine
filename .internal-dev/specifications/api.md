---
schema_version: 1
document_type: api-specification
status: active
owner: api
created: 2026-07-03
---

# API Specification

## Active Contracts

| id | route_or_surface | status | intended_contract | payload_or_status | compatibility_rule | validation | related_decisions | related_knowledge |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| API-20260703-01 | Renderer examples | active | Renderer examples are canonical runtime entrypoints for smoke validation and visual capture validation. | `src/renderer/examples/*.rs`, root `AGENTS.md` | Do not break documented example commands without updating root guidance and relevant docs. | `cargo check -p renderer --examples` and timeout-bound smoke/capture commands. | none | none |
| API-20260703-02 | Headless capture flags | active | Capture-enabled examples write durable PNG/JSON sidecar evidence under task-scoped `.internal-dev/captures/` paths when visual proof is required. | capture examples, capture skill | Sidecar metadata must identify status, target, frame, source, format, and extent when available. | Headless capture skill validation. | none | none |
| API-20260703-03 | Runtime launch registry | active | This file keeps the current list of app/example launch commands, runtime arguments, and scene/project entrypoints used for manual testing and regression validation. | `src/launch.rs`, `apps/editor/src/launch.rs`, `apps/dungeon_dogfood/src/main.rs`, `src/renderer/examples/common/mod.rs`, `tools/engine_pack/src/main.rs` | When adding, removing, or changing a runtime flag, app binary, scene/project manifest, or reusable launch path, update the Runtime Launch Registry below in the same change. | `cargo check` for touched packages plus focused runtime smoke/headless capture where practical. | `DECISION-20260703-01` | none |
| API-20260706-01 | Marching terrain capture presets | active | The marching terrain dogfood app exposes deterministic headless draw-capture presets for mesh-gap validation. `multi-chunk-seams`, `close-up-terrain`, and `local-seam-inspection` use the same full local 3x3x3 chunk neighborhood around `(0,0,0)` as windowed seam validation, ordered by Y, then Z, then X. `close-up-terrain` frames the origin chunk at inspection scale. `local-seam-inspection` frames a local seam/chunk-edge inspection view at inspection scale. | `apps/marching_terrain/src/args.rs`, `apps/marching_terrain/src/capture.rs`, `apps/marching_terrain/src/chunk.rs` | Preserve both `--capture_preset <preset>` and `--capture_preset=<preset>` forms for `single-chunk`, `multi-chunk-seams`, `close-up-terrain`, and `local-seam-inspection`; preserve the shared neighborhood builder contract unless this registry is updated. | `cargo test -p marching_terrain` plus the close-up and local seam draw-capture commands. | none | `marching-terrain-research-baseline.md`, `renderer-camera-override-behavior.md` |
| API-20260706-02 | Renderer camera look-at facade | active | `Renderer::set_camera_look_at(eye, target, up)` applies the renderer-owned camera position and orientation used by frame rendering and headless captures. Invalid look-at vectors return `RendererError::InvalidState` through the existing renderer API error surface and do not silently apply false camera metadata. | `src/renderer/src/api/renderer.rs`, `src/renderer/src/data/camera.rs` | Keep this as a narrow facade over existing camera behavior; do not change renderer/Vulkan frame submission or scene-camera precedence without updating the renderer camera behavior knowledge note. | `cargo test -p renderer camera -- --nocapture`, `cargo check -p renderer --examples`, and headless draw captures when used for visual proof. | none | `renderer-camera-override-behavior.md` |
| API-20260707-01 | App-owned input action bridge and renderer routing | active | `engine::input::InputActionEventEmitter` emits action events from a caller-provided `InputSnapshot` and frame index into a caller-owned `EventBus`; `Renderer::route_platform_input` handles renderer platform side effects and returns `RendererInputRouting`; `engine::input::queue_routed_input_event` queues uncaptured input into caller-owned `InputSystem`. | `src/input.rs`, `src/renderer/src/api/renderer.rs`, `src/renderer/src/api/mod.rs`, `src/renderer/src/api/prelude.rs` | `Renderer::update_input` remains compatibility behavior that composes routing with renderer-owned input queueing. Do not make renderer depend on root `engine`. | `cargo test -p engine`, `cargo test -p renderer`, `rg -n "dispatch_frame\\(|emit_input_action|InputActionEventEmitter|update_input\\(|DeviceEvent::MouseMotion" src apps tests`. | `DECISION-20260707-01` | none |
| API-20260707-02 | App-owned event lifecycle helper | active | `engine::events::runtime_event_bus()` constructs a recorded caller-owned `EventBus`; `RuntimeEventDispatcher` emits/drains staged lifecycle events against a caller-owned bus, including `frame_started`, `drain_input`, and `frame_ended`. | `src/events.rs`, `src/lib.rs`, `src/runtime.rs`, `tests/runtime_event_dispatcher.rs` | Raw `engine_events` primitives remain re-exported and usable directly. `Renderer::events()` and `Renderer::events_mut()` remain compatibility APIs for renderer-owned legacy frame/input paths. | `cargo test -p engine`, `cargo test -p engine_events`, `cargo test -p renderer`, `rg -n "EventBus|FrameStarted|FrameEnded|events_mut\\(|drain_stage|dispatch_pending" src apps tests`. | `DECISION-20260707-02` | none |
| API-20260707-03 | Dogfood app-owned runtime proof | active | `apps/dungeon_dogfood` owns its input system, event bus, frame clock, camera/controller, and audio telemetry while rendering through caller-provided `CameraView` APIs. | `apps/dungeon_dogfood/src/main.rs`, `apps/dungeon_dogfood/src/events.rs`, `apps/dungeon_dogfood/src/audio_bridge.rs`, `docs/api/14-dogfood-vertical-slice.md` | Dogfood must not use renderer-owned input dispatch, `Renderer::events_mut()`, or renderer-owned gameplay camera state on the active path. Renderer still owns Vulkan frame submission, assets, resize, capture output, and platform/UI side effects. | `cargo check -p dungeon_dogfood`, `cargo test -p dungeon_dogfood`, `cargo check`, legacy-call grep, windowed smoke, headless draw capture. | `DECISION-20260707-01`, `DECISION-20260707-02` | none |
| API-20260707-04 | Root engine bin plus lib facade | active | The root `engine` crate provides the data-driven launcher binary and a thin library facade with `camera`, `events`, `frame`, `input`, `render`, and `prelude` modules. | `src/main.rs`, `src/lib.rs`, `src/camera.rs`, `src/events.rs`, `src/frame.rs`, `src/input.rs`, `src/render.rs` | The root facade must not hide raw support crates or become a required runtime object. Apps may use root facade modules or raw crates directly. | `cargo test -p engine`, `tests/facade_imports.rs`, `cargo check`. | `DECISION-20260707-03` | none |
| API-20260707-05 | Renderer-owned camera view DTO | active | `CameraView` lives in the renderer facade because the renderer consumes it at the scene-to-render-submission boundary; the root facade re-exports it for app-owned loops. | `src/renderer/src/api/renderer.rs`, `src/renderer/src/api/mod.rs`, `src/render.rs`, `tests/facade_imports.rs` | Keep the DTO Vulkan-opaque and caller-owned. Do not move renderer internals into root `engine`. | `cargo test -p renderer`, `cargo test -p engine`, `cargo check -p renderer --examples`. | `DECISION-20260707-04` | none |

## Runtime Launch Registry

Keep this registry current with runtime launch surfaces. Use paths relative to the repository root.

### Shared Runtime Flags

| scope | flags | notes |
| --- | --- | --- |
| Root `engine` launcher | `--project <path>` or `--project=<path>` | Required. Points at an `engine.project.toml`. |
| Root `engine` launcher | `--scene <path>` or `--scene=<path>` | Optional scene override. If absent, uses the project's `startup_scene`. |
| Root `engine` launcher | `--headless` | Uses the headless runtime path. |
| Root `engine` launcher | `-h`, `--help` | Prints usage. Cannot be combined with runtime options. |
| Root `engine`, editor, renderer examples | `--record_debug <seconds>`, `--record_debug=<seconds>` | Starts debug timing capture. Value must be >= 1. |
| Root `engine`, editor, renderer examples | `--record_debug_interval <ms>`, `--record_debug_interval=<ms>` | Debug timing sample interval. Value must be >= 1. |
| Root `engine`, editor, renderer examples | `--record_debug_path <path>`, `--record_debug_path=<path>` | Writes debug timing JSONL to an explicit path. |
| Root `engine`, editor, renderer examples | `--capture_target <present\|draw>`, `--capture_target=<present\|draw>` | `present` is default. Use `draw` for headless draw-target validation. |
| Root `engine`, editor, renderer examples | `--capture_frame <n>`, `--capture_frame=<n>` | Single frame capture. Root/examples require positive `n`; editor currently accepts non-negative `n`. |
| Root `engine`, editor, renderer examples | `--capture_frame_path <path>`, `--capture_frame_path=<path>` | Requires `--capture_frame`. |
| Root `engine`, editor, renderer examples | `--capture_frames <n>`, `--capture_frames=<n>` | Sequence capture count. Requires `n >= 1`. |
| Dogfood | `--capture_frames <n>` | Sequence capture count. Current dogfood parser does not accept equals form; see local bug `dogfood-capture-equals-flags-documented-but-not-parsed`. |
| Root `engine`, editor, renderer examples | `--capture_frame_start <n>`, `--capture_frame_start=<n>` | Requires `--capture_frames`. Root/examples require positive `n`; editor currently accepts non-negative `n`. |
| Dogfood | `--capture_frame_start <n>` | Requires `--capture_frames`. Current dogfood parser accepts non-negative `n` and does not accept equals form. |
| Root `engine`, editor, renderer examples | `--capture_frame_interval <n>`, `--capture_frame_interval=<n>` | Requires `--capture_frames`; `n >= 1`. |
| Dogfood | `--capture_frame_interval <n>` | Requires `--capture_frames`; `n >= 1`. Current dogfood parser does not accept equals form. |
| Root `engine`, editor, renderer examples | `--capture_dir <dir>`, `--capture_dir=<dir>` | Requires `--capture_frames`. |
| Dogfood | `--capture_dir <dir>` | Requires `--capture_frames`. Current dogfood parser does not accept equals form. |
| Root `engine`, editor, renderer examples | `--manual_capture_dir <dir>`, `--manual_capture_dir=<dir>` | Directory used for manual capture requests. |

### Application Entrypoints

| surface | command | supported arguments | current project/scene defaults | notes |
| --- | --- | --- | --- | --- |
| Root runtime launcher | `cargo run -- --project apps/editor/sample_project/engine.project.toml` | Shared root launcher flags above. | Project startup scene is `apps/editor/sample_project/scenes/start.engine.scene.json`. | Uses `src/main.rs` and `src/runtime.rs`; best generic project-manifest launch path. |
| Editor app | `cargo run -p editor -- --project apps/editor/sample_project/engine.project.toml` | `--project`, `--scene`, `--headless`, debug flags, capture flags, manual capture dir. | If no project is passed, editor falls back to `apps/editor/sample_project/engine.project.toml`; if no scene is passed, uses the project startup scene. | Current parser ignores unknown flags and has no `--help` command. |
| Dungeon dogfood app | `cargo run -p dungeon_dogfood` | `--level <selector-or-path>`, `--headless`, `--capture_target`, `--capture_frames`, `--capture_frame_start`, `--capture_frame_interval`, `--capture_dir`. Environment variables: `DUNGEON_DOGFOOD_LEVEL`, `DUNGEON_DOGFOOD_GENERATOR_SEED`, `DUNGEON_DOGFOOD_GENERATOR_WIDTH`, `DUNGEON_DOGFOOD_GENERATOR_HEIGHT`, `DUNGEON_DOGFOOD_GENERATOR_LAYERS`, `DUNGEON_DOGFOOD_VALIDATION`. | Built-in selectors: generated default, `level_01`, `level_02_ramps`, `level_03_lighting`; project manifest exists at `apps/dungeon_dogfood/engine.project.toml`. | Runtime app parses its own level/capture args rather than the shared root launcher. |
| Marching terrain app | `cargo run -p marching_terrain -- --headless --capture_target draw --capture_preset single-chunk` | `--headless`, `--capture_target <present\|draw>`, `--capture_preset <single-chunk\|multi-chunk-seams\|close-up-terrain\|local-seam-inspection>`, `--capture_frames <n>`, `--capture_frame_start <n>`, `--capture_frame_interval <n>`, `--capture_dir <dir>`, `--record_debug <seconds>`, `--record_debug_interval <ms>`, `--record_debug_path <path>`. All options accept both space and equals forms. | Default capture preset is `single-chunk`; `multi-chunk-seams`, `close-up-terrain`, and `local-seam-inspection` use the shared full local 3x3x3 validation neighborhood; `close-up-terrain` and `local-seam-inspection` use renderer look-at camera control for close inspection proof. Deterministic preset sidecars include target, preset, chunk list, requested camera, applied camera/API path, seed/config, expected visual context, terrain diagnostic summary, superseded symptom captures, and residuals. | App-local mesh-gap validation must use headless draw-target captures; present/manual captures are symptom evidence only. |
| Engine pack tool | `cargo run -p engine_pack -- <command>` | `validate-package`, `validate-project`, `validate-scene`, `new-app`, `new-project`, `new-package`, `scan-assets`, `add-asset`, `pack`. | Uses explicit paths per command. | Packaging/validation CLI, not a renderer runtime. |

### Renderer Example Entrypoints

| surface | command | supported arguments | notes |
| --- | --- | --- | --- |
| PBR demo | `cargo run -p renderer --example demo_pbr` | Renderer example flags: `--env`, `--model`, debug flags, capture flags, `--headless`, manual capture dir. | General PBR/material smoke path. |
| Unlit demo | `cargo run -p renderer --example demo_unlit` | Renderer example flags. | Unlit shader/runtime smoke path. |
| Model-load demo | `cargo run -p renderer --example demo_model_load` | Renderer example flags. | glTF/model loading smoke path. |
| Async loading demo | `cargo run -p renderer --example demo_async_loading` | Renderer example flags. | Asset task and async loading smoke path. |
| API test demo | `cargo run -p renderer --example api_test` | Renderer example flags. `--env src/renderer/src/assets/sky_maps/indoor_4k.exr` is a useful environment override. | Renderer facade and environment runtime smoke path. |
| Capture geometry | `cargo run -p renderer --example capture_geometry -- --headless` | Capture-test flags: `--headless`, `--capture_target`, `--capture_frames`, `--capture_frame_start`, `--capture_frame_interval`, `--capture_dir`. | Baseline visual capture scene. |
| Capture lighting | `cargo run -p renderer --example capture_lighting -- --headless` | Capture-test flags. | Baseline visual capture scene. |
| Capture transform | `cargo run -p renderer --example capture_transform -- --headless` | Capture-test flags. | Baseline visual capture scene; current validated baseline has a framing caveat. |
| Capture material PBR | `cargo run -p renderer --example capture_material_pbr -- --headless` | Capture-test flags. | Baseline visual capture scene. |
| Capture environment | `cargo run -p renderer --example capture_environment -- --headless` | Capture-test flags. | Baseline visual capture scene. |
| Capture model load | `cargo run -p renderer --example capture_model_load -- --headless` | Capture-test flags. | Baseline visual capture scene. |

## Drift Records

| id | spec | status | observed_drift | impact | routing | source | review_after |
| --- | --- | --- | --- | --- | --- | --- | --- |
