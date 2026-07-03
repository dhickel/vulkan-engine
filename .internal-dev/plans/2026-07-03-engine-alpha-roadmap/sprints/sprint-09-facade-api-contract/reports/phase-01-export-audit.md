# Phase 01 Export Audit

## Scope

Audited the current renderer facade exports in `src/renderer/src/api/mod.rs` and
the extra crate-root exports in `src/renderer/src/lib.rs`. This phase preserved
all public exports and updated `docs/api/00-index.md` so public reachability is
not presented as beginner-stable alpha support.

## Export Classification

| Symbol group | Source path | Current exposure | Intended classification | Action |
|--------------|-------------|------------------|-------------------------|--------|
| Renderer lifecycle and frame results: `Renderer`, `RendererConfig`, `RendererError`, `RendererInitError`, `RendererFrameError`, `FrameContext`, `FrameRenderOutcome`, `EnvironmentRuntimeStatus` | `src/renderer/src/api/mod.rs`, re-exported by `src/renderer/src/lib.rs` | `renderer::api::*` and crate root | Alpha beginner facade | Preserved. Documented as part of the small supported alpha path. |
| Scene facade and validation: `Scene`, `SceneNodeId`, `PointLight`, `SceneAssetReference`, `SceneFragment*`, `SceneNodeSummary`, `SceneValidationOptions`, `SceneError`, `validate_scene_*` | `src/renderer/src/api/mod.rs`, re-exported by `src/renderer/src/lib.rs` | `renderer::api::*` and crate root | Alpha beginner facade, with editor-style command workflows noted separately | Preserved. Documented as beginner-facing for scene creation/load/save and validation. |
| Asset/project/package facade: `AssetManager`, `AssetError`, `AssetKind`, `AssetRegistry`, `AssetRegistryError`, durable/package/project records, `PackageManifest`, validation options, `parse_package_manifest`, `validate_package_manifest_*`, `validate_project_*` | `src/renderer/src/api/mod.rs`, re-exported by `src/renderer/src/lib.rs` | `renderer::api::*` and crate root | Alpha beginner facade for project/package asset loading and validation | Preserved. Documented as part of the supported alpha path. |
| Asset handles and material/texture descriptors: `MeshHandle`, `TextureHandle`, `MaterialHandle`, `EnvironmentHandle`, `PbrMaterialDesc`, `ProceduralMeshData`, `ProceduralVertex`, `TextureLoadOptions`, `SamplerOverride`, `FilterMode`, `WrapMode`, `ResolvedTexturePolicy` | `src/renderer/src/api/mod.rs`, re-exported by `src/renderer/src/lib.rs` | `renderer::api::*` and crate root | Mixed alpha beginner facade and advanced-adjacent data descriptors | Preserved. Beginner docs should use handles and simple descriptors; deeper material/texture policy work remains alpha/advanced-adjacent. |
| Loading tickets/status: `LoadTicket`, `LoadStatus` | `src/renderer/src/api/mod.rs`, re-exported by `src/renderer/src/lib.rs` | `renderer::api::*` and crate root | Alpha beginner facade | Preserved. Documented as supported for request/poll asset loading. |
| Input facade re-exports: `InputSystem`, `InputSnapshot`, `FrameInputSnapshot`, `ActionMap`, `ActionId`, `ActionBinding`, layers, bindings, chords, debug snapshots, `editor_ui_capture_layer`, `priority_bands` | `src/renderer/src/api/mod.rs`, re-exported by `src/renderer/src/lib.rs` | `renderer::api::*` and crate root | Alpha beginner facade for event/input update, with lower-level layer/debug types advanced-adjacent | Preserved. Documented as beginner path for input update without narrowing exports. |
| Debug, timing, capture, and config controls: `CaptureTarget`, `FrameCaptureRequest`, `FrameCaptureSequence`, `FrameCaptureStatus`, `FrameCaptureScheduler`, `DueFrameCapture`, capture path helpers, `DebugRuntimeMode`, `DebugTiming*`, debug view callbacks/descriptors, `VisualTuning`, asset policy config | `src/renderer/src/api/mod.rs`, re-exported by `src/renderer/src/lib.rs` | `renderer::api::*` and crate root | Alpha beginner facade for debug/capture controls; callbacks are advanced-adjacent extension points | Preserved. Documented as part of alpha debug/capture surface. |
| Engine event facade: `EngineEvent`, `EventBus`, `EventRecorder`, `EventEnvelope`, `EventSequence`, event IDs/stages and domain event enums | `src/renderer/src/api/mod.rs`, re-exported by `src/renderer/src/lib.rs` | `renderer::api::*` and crate root | Alpha beginner facade where docs/examples use events; some domain IDs are compatibility public | Preserved. Documented as part of the alpha event contract. |
| Render hooks: `RenderHook`, `RenderHookContext`, `HookError` | `src/renderer/src/api/mod.rs`, re-exported by `src/renderer/src/lib.rs` | `renderer::api::*` and crate root | Advanced-adjacent public facade | Preserved. Kept out of the beginner promise except where hook docs explicitly discuss extension points. |
| `renderer::api::advanced` | `src/renderer/src/api/mod.rs` | Only exposed with `advanced-interop` feature | Advanced interop | Preserved. Index now calls out the feature-gated advanced tier. |
| Animation: `AnimationPlayer` root export plus animation module types | `src/renderer/src/lib.rs`, `src/renderer/src/animation/` | Root export for `AnimationPlayer`; `renderer::animation::*` module is public | Compatibility public | Preserved. Classified as not beginner-stable; integration tests currently use it. |
| Camera/frustum/ray helpers: `Aabb`, `Camera`, `FPSController`, `Frustum`, `OrbitCamera`, `OrbitController`, `Ray` | `src/renderer/src/lib.rs`, `src/renderer/src/data/camera.rs` | Root-only re-exports | Compatibility public, advanced-adjacent for editor/diagnostic workflows | Preserved. Classified as root-only helper group outside the beginner contract. |
| Scene command history: `AddNodeCommand`, `Command`, `CommandHistory`, `CommandResult`, `PlaceAssetCommand`, `RemoveNodeCommand`, `SceneNodeRemap`, `SetTransformCommand` | `src/renderer/src/lib.rs`, `src/renderer/src/scene/command.rs` | Root-only re-exports | Compatibility public for editor-style workflows | Preserved. Classified outside the beginner contract; current docs and tests use `CommandHistory`. |
| `SceneWorld` | `src/renderer/src/lib.rs`, `src/renderer/src/scene/scene_world.rs` | Root-only re-export | Internal implementation detail exposed through legacy compatibility path | Preserved. Classified as compatibility public, not beginner facade. |
| Private renderer modules: `data`, `debug_ui`, `scene`, `vulkan`, default `rendergraph` | `src/renderer/src/lib.rs` | Private modules, except `rendergraph` when `advanced-interop` is enabled | Internal implementation detail | No change. Do not newly document as beginner API. |
| Larger project runtime, material override, generated app-template, and advanced rendering extension shape | Plan target docs | Not fully implemented as a simple beginner API | Deferred | Documented as deferred in the API index rather than promised. |

## Root Export Mismatch

The previous API index said the full re-export list lived in
`src/renderer/src/api/mod.rs` and that everything below `api::*` in `lib.rs` was
the stable public surface. That was inaccurate because `src/renderer/src/lib.rs`
also exposes root-only groups for animation, camera/frustum/ray helpers, scene
command history, and `SceneWorld`. Those exports are still public and compile in
`src/renderer/tests/integration.rs`, but they are compatibility public rather
than the beginner alpha contract.

## Changes Made

- Updated `docs/api/00-index.md` to replace the overbroad top-level re-export
  promise with a tiered alpha API contract.
- Preserved all current public exports.
- Did not add compile tests because the existing GPU-free integration test
  already imports and exercises the key compatibility root exports.
- Did not change rustdoc or re-export structure, so `cargo doc -p renderer
  --no-deps` was not required for this phase.

## Validation

| Command | Result | Notes |
|---------|--------|-------|
| `cargo fmt --check` | Pass | No formatting changes required. |
| `cargo check -p renderer` | Pass | Completed with existing dead-code warnings in renderer internals. |
| `rg -n "stable public surface|Everything below api|advanced-interop|AnimationPlayer|SceneWorld|CommandHistory" docs/api src/renderer/src src/renderer/tests` | Pass for intent | Removed the stale overpromise phrases. Remaining hits are expected references to advanced interop and compatibility symbols in docs/code/tests. |
| `cargo doc -p renderer --no-deps` | Not run | Markdown-only docs change; no rustdoc or re-export organization changed materially. |

## Residuals

- Public root compatibility exports remain intentionally broad until later phases
  decide whether a curated prelude or import guide is worth adding.
- `docs/api/03-scene-graph-and-fragment-workflows.md` still documents
  `CommandHistory` for editor transaction workflows; that is compatible with
  this phase as long as it is not described as the beginner facade.
- `docs/api/05-hooks.md` and
  `docs/api/05-render-hooks-and-extension-points.md` both mention
  `advanced-interop`; later documentation cleanup may decide which hook chapter
  is canonical.
- `cargo check -p renderer` warning volume remains outside this phase.
