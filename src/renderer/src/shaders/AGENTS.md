# Shader Agent Guide (`src/renderer/src/shaders`)

This guide covers shader ownership, local file mapping, and external reference lineage.

## Directory Role

This directory holds the renderer's active GLSL sources plus precompiled SPIR-V artifacts (`.spv`).

Key point:
- Runtime currently consumes `.spv` files from disk during cache init.
- Compile-at-startup path in `VkRender::new(..., compile_shaders)` is effectively disabled.

## Local Shader Map

### Core PBR

- Vertex: `src/renderer/src/shaders/pbr_base.vert`
- Fragment: `src/renderer/src/shaders/material_pbr.frag`
- Optional unlit variant: `src/renderer/src/shaders/material_unlit.frag.spv` (source in `ignored/`)

### Environment / IBL

- Irradiance convolution: `src/renderer/src/shaders/env_irradiance_cube.frag`
- Specular prefilter: `src/renderer/src/shaders/env_prefilter_cube.frag`
- Shared cube vertex stage: `src/renderer/src/shaders/filtered_cube.vert`
- BRDF LUT generation: `src/renderer/src/shaders/gen_brd_flut.vert`, `src/renderer/src/shaders/gen_brd_flut.frag`
- Skybox draw: `src/renderer/src/shaders/skybox.vert`, `src/renderer/src/shaders/skybox.frag`

### Shared Include-Style Sources

- Material structs/utilities: `src/renderer/src/shaders/shader_material.glsl`
- Vertex struct definitions: `src/renderer/src/shaders/vertex_struct.glsl`
- Color conversion: `src/renderer/src/shaders/srgbtolinear.glsl`
- Tonemapping: `src/renderer/src/shaders/tonemapping.glsl`

### Compute

- Sky compute: `src/renderer/src/shaders/sky.comp`

## Integration Points in Rust

- Pipeline creation and shader module loading:
  - `src/renderer/src/vulkan/vk_pipeline.rs`
  - `src/renderer/src/data/data_cache.rs` (`CoreShaderType`, `VkShaderCache`)
- Environment map generation pass wiring:
  - `src/renderer/src/vulkan/vk_render.rs` (`generate_environment`)
- Descriptor layout binding contracts:
  - `src/renderer/src/vulkan/vk_descriptor.rs` (`init_descriptor_cache`)

## Binding and Layout Discipline

When editing shaders, preserve compatibility with:
- Push constant structs in Rust (`VkModelPushConsts`, `PushConstIrradiance`, `PushConstPrefilterEnv`, `PushConstSkyBox`).
- Descriptor set ordering expected by draw path:
  - set 0: scene/environment
  - set 1: skin/joint
  - set 2: material samplers
- Vertex pulling convention (buffer device address path) used by pipeline setup.

If you change shader interfaces, update corresponding Rust structs and pipeline layouts in the same change.

## External Lineage Reference

PBR/IBL design guidance reference:
- `https://github.com/SaschaWillems/Vulkan-glTF-PBR`

Relevant shader examples in that repository include:
- `data/shaders/pbr.vert`
- `data/shaders/pbr_khr.frag`
- `data/shaders/irradiancecube.frag`

Use that repository as conceptual baseline, not as a drop-in source. Descriptor layouts and push constants differ between engines.

## Maintenance Gotchas

1. Keep `.vert/.frag/.comp` and `.spv` in sync.
- If GLSL changes without refreshed SPIR-V, runtime behavior will not match source.

2. Preserve format assumptions for IBL passes.
- Irradiance and prefilter targets use different formats and mip behavior.

3. Validate pipeline compatibility after edits.
- A shader edit can require pipeline layout or descriptor cache changes.

4. Avoid silent include drift.
- Shared GLSL helpers must remain consistent with fragment expectations and Rust-side packing.
