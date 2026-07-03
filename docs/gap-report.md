# Engine Gap & Readiness Report

> Auto-generated from a codebase audit. All findings cite source files directly — no old docs were consulted.

## 🔴 Critical (blocks engine/editor viability)

### 1. No scene serialization (save/load to disk)

`Scene`, `SceneWorld`, `SceneNode`, and `PointLight` have **no `Serialize`/`Deserialize` derives**. The only serde usage in the renderer crate is for `.meta` sidecar parsing. Without serialization, an editor cannot save or restore scene state — the most fundamental editor requirement.

- **Evidence**: `SceneWorld` fields at `src/renderer/src/scene/scene_world.rs:87-95`, `SceneNode` at `:35-45`. Zero serde derives in any scene file.
- **Resolution**: Derive `Serialize`/`Deserialize` on `SceneNode`, `SceneNodeId`, `PointLight`, `SceneWorld`. Add `Scene::save(path)` / `Scene::load(path)`. Handle types need path-based (not slot-index) serialization.

### 2. No ray-picking or object selection

Zero ray-casting infrastructure exists: no `screen_to_world` ray, no mouse-to-world intersection, no hit-testing against scene geometry. An editor can't select objects it can't identify.

- **Evidence**: Search for `ray|picking|select|mouse.*world|screen.*ray` across the codebase returned zero relevant matches.
- **Resolution**: Add `screen_to_ray(view, proj, screen_pos)` utility. Add CPU-side AABB intersection or GPU ray query. Expose as `Renderer::pick(&Scene, x, y) -> Option<SceneNodeId>`.

### 3. No undo/redo infrastructure

Zero command pattern, history stack, or undo/redo system anywhere in the codebase.

- **Evidence**: Search for `undo|redo|command.*pattern|history.*stack` returned only a `KeyCode::Undo` keyboard constant in `src/input/src/lib.rs:1265` — not a system.
- **Resolution**: Implement `Command` trait with `execute`/`undo`, `CommandHistory` stack. Wrap scene mutations in command objects.

### 4. No audio subsystem

Zero audio support.

- **Evidence**: Search for `audio|sound` returned zero matches in `src/`.
- **Resolution**: Add an audio crate to the workspace with spatial audio and basic playback.

### 5. No physics integration

Zero physics engine integration.

- **Evidence**: Search for `physics|rapier|bevy_xpbd` returned zero matches.
- **Resolution**: Integrate a physics crate (e.g., `rapier3d`) and expose collision/rigid-body APIs.

### 6. No scripting bindings

Zero scripting language support (Lua, Rhai, Python, etc.).

- **Evidence**: Search for `script|rhai|lua|python|rune` returned only GPU descriptor-type false positives.
- **Resolution**: Integrate a scripting runtime (e.g., Rhai for native Rust interop) and expose the engine API via script bindings.

### 7. Headless/offscreen mode explicitly unsupported

The renderer refuses to start in headless configuration.

- **Evidence**: `src/renderer/src/api/renderer.rs:82-83` — `if config.headless { return Err(RendererError::Unsupported("headless mode not implemented")); }`
- **Resolution**: Implement headless rendering via Vulkan without a surface/present — or provide a software rasterization fallback.

### 8. Swapchain rebuild has known GPU memory leak

Swapchain recreation does not destroy old image views before reassigning, risking gradual VRAM exhaustion on window resize.

- **Evidence**: `src/renderer/src/vulkan/vk_render.rs:1154` — `// FIXME, I think we will need to destroy the old images view when we reassign`. Confirmed in `src/renderer/AGENTS.md`: "Swapchain rebuild still has explicit cleanup FIXME areas."
- **Resolution**: Call `vkDestroyImageView` on old swapchain image views before reassigning during resize.

---

## 🟡 Important (significant limitation)

### 9. Animation/skinning runtime is incomplete

GPU skinning infrastructure exists (joint buffers, descriptors at `src/renderer/src/data/gpu_data.rs:61`, `joint_count` at `:519`), and assimp detects `has_animation`. But there is **no CPU-side animation playback**: no bone interpolation, no clip playback, no time-based updates. Skinned models render in bind pose only.

- **Evidence**: `src/renderer/src/data/data_cache.rs:1466` allocates 128 identity matrices as fallback joints. No animation update code anywhere.
- **Resolution**: Implement `AnimationPlayer` component that reads glTF animation data, interpolates bone matrices per-frame, and uploads joint transforms.

### 10. Only one camera model (FPS, no editor camera)

Only `Camera` + `FPSController` exist. No orbiting/arcball camera, no pan/zoom/dolly, no `look_at` target.

- **Evidence**: `src/renderer/src/data/camera.rs:82-103` — `FPSController::update_from_snapshot` directly maps FPS input only.
- **Resolution**: Add `OrbitCamera` with target point, spherical coordinates (theta, phi, radius), smooth zoom/dolly.

### 11. No project system or asset browser

No `Project` type, no workspace format, no asset registry beyond the in-memory `AssetManager`.

- **Evidence**: `apps/dungeon_dogfood/` has a hardcoded `content_pack.toml` — not general-purpose.
- **Resolution**: Design a project manifest format. Build an asset browser UI. Add `AssetRegistry` with path-based lookup.

### 12. No hot-reloading of assets

No file watcher. Shaders, textures, and models cannot be hot-reloaded at runtime.

- **Evidence**: Search for `hot.reload|watch|file_watch|notify` returned only `condvar.notify_all()` in sync primitives.
- **Resolution**: Add a `notify`-based file watcher that invalidates asset caches and triggers reloads.

### 13. Many `unwrap()` calls in production code

Critical paths use `unwrap()` where errors should propagate. Key examples:
- `src/renderer/src/data/data_cache.rs:172` — descriptor allocation unwraps
- `src/renderer/src/data/data_cache.rs:407-409` — descriptor allocator creation unwraps
- `src/renderer/src/vulkan/vk_debug.rs:102-108` — command buffer end/submit unwraps
- `src/renderer/src/vulkan/vk_init.rs:80` — window handle unwrap
- `src/renderer/src/vulkan/vk_descriptor.rs:512-574` — multiple allocate unwraps
- `src/renderer/src/vulkan/vk_util.rs:1486-1708` — command buffer begin/end/submit unwraps throughout

- **Resolution**: Replace with `?` propagation using the `map_init_err`/`map_frame_render_err` pattern from `errors.rs`.

### 14. No depth buffer access exposed to users

The depth buffer exists internally (`prepare_targets_pass.rs`, `vk_pipeline.rs:286-307`) but cannot be read by users for picking, post-processing, or effects.

- **Resolution**: Expose depth as a bindable texture in `RenderHookContext` or add a depth-read debug view.

### 15. Limited public access to rendering internals

Users cannot add custom rendergraph passes. `RenderPassNode` trait exists at `src/renderer/src/rendergraph/mod.rs:31` but the `rendergraph` module is private (`mod rendergraph;` in `lib.rs`).

- **Evidence**: The `advanced-interop` feature gate at `src/renderer/src/api/advanced.rs` exposes `raw_core_mut()` but warns it's unsafe.
- **Resolution**: Make `rendergraph` public. Expose `RenderGraph::new(passes)` via `Renderer` under the `advanced-interop` feature.

### 16. Missing test coverage in critical areas

Tests exist for data utilities but **none** for: the Vulkan backend (`vk_render.rs`, `vk_pipeline.rs`, `vk_descriptor.rs`), the `Renderer`+`Scene` integration, the debug UI, or the rendergraph.

- **Resolution**: Add integration tests for the public API (smoke: create Renderer, load model, render one frame). Unit tests for rendergraph pass ordering.

### 17. Push constant size may exceed Vulkan minimum

The skybox push constant struct may exceed the 128-byte minimum guarantee.

- **Evidence**: `src/renderer/src/data/gpu_data.rs:628` — `// FIXME this need combined to to stay under 128byte push const`
- **Resolution**: Verify push constant size against `maxPushConstantsSize` at device creation. Split into multiple push ranges or move to UBO if needed.

---

## 🟢 Nice-to-have (quality-of-life, polish)

### 18. Hardcoded asset paths in examples

- **Evidence**: `src/renderer/examples/common/mod.rs:14` — `FACADE_DEMO_MODEL_PATH: &str = "src/renderer/src/assets/DamagedHelmet.glb"`. Also in `demo_async_loading.rs`.
- **Resolution**: Accept model path as a CLI argument or environment variable.

### 19. `texture.rs` is a legacy stub

- **Evidence**: `src/renderer/src/texture.rs` — single line: `// Legacy texture scratch module retained for experimentation.`
- **Resolution**: Remove or repurpose.

### 20. `#[allow(dead_code)]` on the example common module

- **Evidence**: `src/renderer/examples/common/mod.rs:1` — `#![allow(dead_code)]`
- **Resolution**: Clean up unused code or remove the blanket allowance.

### 21. TOML input profile format is undocumented for end users

The `ActionMap::from_toml_str` parser works but no reference doc explains the TOML schema to users.

- **Evidence**: `src/input/src/lib.rs:1172-1225` — parser expects `version = 1`, `[[bindings]]` with `action`, `trigger.key`/`trigger.mouse_button`, `modifiers`, `scale`, `consume`, `context`. No user-facing doc.
- **Resolution**: Add a TOML schema reference to input documentation.

### 22. No frustum or occlusion culling

The scene flattener builds a flat `RenderSubmission` but does not perform frustum culling or occlusion culling on the CPU side.

- **Evidence**: `src/renderer/src/scene/scene_world.rs` — `build_submission()` iterates all nodes unconditionally.
- **Resolution**: Add frustum culling against camera AABB before building draw commands.

---

## Current Readiness Assessment

This engine is **not ready for editor development**. The single biggest blocker is the **absence of scene serialization** (gap #1) — an editor that can't save or load scenes is not an editor. The runner-up blockers are ray-picking (#2) and undo/redo (#3), both essential for any editing workflow. Beyond those, the engine has a solid rendering core (Vulkan 1.3, dynamic rendering, PBR/IBL, rendergraph) but is currently a **rendering demo framework**, not a general-purpose engine. Addressing the 🔴 critical items in order would constitute a viable editor-foundation milestone.
