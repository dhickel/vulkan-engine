# Renderer Package Agent Guide (`src/renderer`)

This is the package-level guide for the renderer crate.
Use this for architecture and maintenance strategy across `data/` and `vulkan/`.

## Package Role

`renderer` owns:
- window + event loop integration (`winit`)
- Vulkan lifecycle and frame rendering
- scene ownership (`SceneWorld`) and per-frame submission build
- rendergraph pass orchestration over Vulkan core
- model/material/texture/mesh caching and upload
- ImGui rendering integration

Entrypoints: `src/renderer/examples/*.rs` (facade-first runtime examples)

## Package Structure

- `src/renderer/src/lib.rs`: public facade exports and internal module wiring.
- `src/renderer/src/data/`: scene, camera, caches, model loaders.
- `src/renderer/src/vulkan/`: Vulkan initialization, wrappers, sync, render loop.
- `src/renderer/src/shaders/`: GLSL + SPIR-V artifacts.
- `src/renderer/src/assets/`: default model/cubemap assets.

Deep dives:
- `src/renderer/src/data/AGENTS.md`
- `src/renderer/src/vulkan/AGENTS.md`
- `src/renderer/src/shaders/AGENTS.md`

## End-to-End Runtime Flow

1. A renderer example creates event loop, window, FPS controller, and `Renderer`.
2. `Renderer::new(...)` initializes `VkRender` and core caches.
3. Startup scene load path currently uses Assimp:
- `assimp_util::load_model("src/renderer/src/assets/DamagedHelmet.glb", ...)`
- Then concurrent mesh/material allocation threads run.
4. Each frame (example loop + `Renderer` facade + `VkRender::render`):
- poll async transfer completions
- update camera and build `RenderSubmission` from `SceneWorld`
- acquire swapchain image
- execute rendergraph passes (skybox, geometry, present copy, imgui)
- submit and present

## Architectural Style

- Explicit Vulkan wrappers over `ash`.
- Traditional descriptor sets (not bindless indexing architecture).
- Scene ownership via `SceneWorld` stable node handles (slot + generation, not ECS).
- Mixed memory management:
- `vk_mem` for images/buffers
- custom sub-allocator for packed geometry/material data

## Current Operational Assumptions

- Vulkan 1.3-capable device is expected.
- Graphics and present queues are currently treated as effectively shared in render path submission.
- Shader compile-at-runtime path is disabled by default (uses precompiled `.spv` artifacts).
- Startup loads a hardcoded asset and skybox resources.
- Runtime example entrypoints:
  - `cargo run -p renderer --example api_test`
  - `cargo run -p renderer --example demo_pbr`
  - `cargo run -p renderer --example demo_unlit`
  - `cargo run -p renderer --example demo_model_load`
  - `cargo run -p renderer --example demo_async_loading`

## PBR and Radiance Reference Map

Use these files first for rendering feature work:

- Frame orchestration and IBL generation:
  - `src/renderer/src/vulkan/vk_render.rs`
  - `generate_environment(...)` handles irradiance + prefilter cubemap generation.
  - `draw_geometry_from_submission(...)` is the main PBR draw path.
  - `draw_skybox_from_submission(...)` renders environment background.
  - rendergraph pass wiring is in `src/renderer/src/rendergraph/`.
- Pipeline creation:
  - `src/renderer/src/vulkan/vk_pipeline.rs`
  - `init_met_rough_pipelines(...)` for core PBR pipelines.
  - `init_irradiance_pipeline(...)` for diffuse irradiance cube generation.
  - `init_pre_filter_pipeline(...)` for specular prefilter cube generation.
  - `init_brd_flut_pipeline(...)` for BRDF LUT generation.
- Descriptor layout contract:
  - `src/renderer/src/vulkan/vk_descriptor.rs`
  - `init_descriptor_cache(...)` defines scene/material/env descriptor layouts.
- Environment cache and cubemap loading:
  - `src/renderer/src/data/data_cache.rs`
  - `EnvironmentCache` and `allocate_cube_map(...)`.
- Local shader sources:
  - `src/renderer/src/shaders/pbr_base.vert`
  - `src/renderer/src/shaders/material_pbr.frag`
  - `src/renderer/src/shaders/env_irradiance_cube.frag`
  - `src/renderer/src/shaders/env_prefilter_cube.frag`
  - `src/renderer/src/shaders/gen_brd_flut.frag`

External lineage reference:
- `https://github.com/SaschaWillems/Vulkan-glTF-PBR`

## High-Value Maintenance Hotspots

- `src/renderer/src/vulkan/vk_render.rs` (~2k LOC): central orchestration and risk concentration.
- `src/renderer/src/data/data_cache.rs` (~1.7k LOC): typed stable handles, allocation lifecycles, pointer stability.
- `src/renderer/src/vulkan/vk_types.rs` + `vk_storage.rs`: frame ownership and allocator semantics.

## Known Risks You Should Internalize

1. Partial cleanup coverage.
- Some destroy paths are unimplemented (`todo!()` in dependent subsystems).

2. Handle validity assumptions can break.
- Handle users must treat `slot + generation` as the API contract and handle `CacheError` outcomes.

3. Render path is sensitive to descriptor/pipeline binding order.
- Small refactors can silently break draw correctness.

4. Heavy `unwrap`/`panic` footprint.
- Error handling policy is inconsistent; be deliberate when hardening.

5. Legacy/disabled code remains in tree.
- `gltf_util.rs` is commented legacy path.

## Suggested Working Style for Agents

- Prefer scoped, testable edits in one subsystem at a time.
- When touching render frame logic, verify pipeline/descriptor bindings with validation layers.
- Keep docs updated when changing data ownership, sync, or cache semantics.
- Treat typed handle caches as API contracts; avoid changing slot/generation semantics casually.

## Build and Verification

- Compile renderer crate: `cargo check -p renderer`
- Compile renderer examples: `cargo check -p renderer --examples`
- Full project: `cargo check`
- Runtime validation requires a Vulkan environment.

## Related Docs

- Top-level: `AGENTS.md`
- Data details: `src/renderer/src/data/AGENTS.md`
- Vulkan details: `src/renderer/src/vulkan/AGENTS.md`
- Shader details: `src/renderer/src/shaders/AGENTS.md`
- Architecture notes: `.internal-dev/`
