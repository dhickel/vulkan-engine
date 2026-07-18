# Shader System

> Source: [`src/renderer/src/shaders/`](../../src/renderer/src/shaders/), [`src/renderer/src/shaders/AGENTS.md`](../../src/renderer/src/shaders/AGENTS.md) — no legacy docs consulted.

## Shader Inventory

All shaders live under `src/renderer/src/shaders/`:

| Shader File | Type | Purpose | SPIR-V |
|------------|------|---------|--------|
| `pbr_base.vert` | Vertex | Main vertex shader — transforms, passes UV/normal/tangent to fragment | `pbr_base.vert.spv` |
| `material_pbr.frag` | Fragment | PBR material evaluation (metallic-roughness, IBL, point + directional light, 3×3 PCF shadow sampling) | `material_pbr.frag.spv` |
| `material_unlit.frag` | Fragment | Unlit material (lighting-independent) | `material_unlit.frag.spv` |
| `shadow_depth.vert` | Vertex | Depth-only directional shadow caster transform via push constants | `shadow_depth.vert.spv` |
| `shadow_depth.frag` | Fragment | Empty depth-only fragment stage | `shadow_depth.frag.spv` |
| `skybox.vert` | Vertex | Full-screen triangle for skybox pass | `skybox.vert.spv` |
| `skybox.frag` | Fragment | Skybox sampling from environment cubemap | `skybox.frag.spv` |
| `env_equirect_to_cube.frag` | Fragment (compute via cube render) | Equirectangular → cubemap projection | `env_equirect_to_cube.frag.spv` |
| `env_irradiance_cube.frag` | Fragment (compute via cube render) | Diffuse IBL irradiance convolution | `env_irradiance_cube.frag.spv` |
| `env_prefilter_cube.frag` | Fragment (compute via cube render) | Specular IBL prefilter (split-sum approx) | `env_prefilter_cube.frag.spv` |
| `filtered_cube.vert` | Vertex | Cube rendering for environment processing | `filtered_cube.vert.spv` |
| `gen_brd_flut.vert` | Vertex | Full-screen triangle for BRDF LUT | `gen_brd_flut.vert.spv` |
| `gen_brd_flut.frag` | Fragment | BRDF integration LUT generation | `gen_brd_flut.frag.spv` |
| `sky.comp` | Compute | Sky/atmosphere compute shader | `sky.comp.spv` |

### Shader Libraries (GLSL includes)

| File | Purpose |
|------|---------|
| `shader_material.glsl` | Material struct definition, shared with Rust via `VkModelPushConsts` |
| `vertex_struct.glsl` | `Vertex` struct matching `gpu_data.rs:36` |
| `tonemapping.glsl` | Tonemapping functions (ACES, Reinhard, etc.) |
| `srgbtolinear.glsl` | sRGB ↔ linear color space conversions |

### Ignored

`ignored/material_unlit.frag` — an alternative/old unlit shader not currently in use.

## Compilation

Shaders are written in **GLSL** and compiled to **SPIR-V** via `glslc` (shaderc). Pre-compiled `.spv` files are checked in alongside the `.glsl`/`.frag`/`.vert`/`.comp` sources.

To recompile shaders at engine startup:
```rust
RendererConfig { compile_shaders: true, ..Default::default() }
```

This requires `glslc` (or `glslangValidator`) in `PATH`. The engine invokes the compiler as an external process during `Renderer::new()`.

## Shader Manifest

`core_shader_manifest.txt` tracks which SPIR-V files correspond to which GLSL sources, enabling the engine to validate that pre-compiled artifacts match the source versions.

## Vertex Format Contract

The `Vertex` struct in Rust ([`gpu_data.rs`](../../src/renderer/src/data/gpu_data.rs)) and `pbr_base.vert` must agree:

```glsl
// vertex_struct.glsl
layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec4 inTangent;    // w = handedness
layout(location = 3) in vec2 inUV0;
layout(location = 4) in vec2 inUV1;
layout(location = 5) in vec4 inColor;
layout(location = 6) in uvec4 inJoints;    // skinning
layout(location = 7) in vec4 inWeights;    // skinning
```

Total: 80 bytes per vertex (interleaved).

## PBR Material Pipeline

`material_pbr.frag` implements the metallic-roughness PBR model:

1. **Base color**: sampled from base color texture × `baseColorFactor`
2. **Metallic/Roughness**: sampled from metallic-roughness texture (B=metallic, G=roughness) × factors
3. **Normal**: tangent-space normal from normal map, transformed to world space
4. **Occlusion**: sampled from occlusion texture
5. **Emissive**: sampled from emissive texture × emissive factor

Lighting:
- **Direct lighting**: point lights plus one optional directional light via Cook-Torrance BRDF (GGX distribution, Smith-GGX geometry, Fresnel-Schlick)
- **Directional shadowing**: scene binding 5 comparison sampler, Vulkan `[0, 1]` depth, slope-aware receiver bias, and fixed 3×3 PCF
- **Indirect diffuse**: sampled from irradiance cubemap
- **Indirect specular**: sampled from prefiltered environment cubemap × BRDF LUT (split-sum approximation)
- **Tonemapping**: applied at the end of the fragment shader

## Push Constants

```glsl
// shader_material.glsl
layout(push_constant) uniform PushConsts {
    mat4 model;
    uint64_t vertexBufferAddress;    // VK_EXT_buffer_device_address
    uint64_t materialMetaBufferAddress;
    uint jointCount;
    uint hasUV1;
} pushConsts;
```

Total: 96 bytes. The shadow depth pass uses a separate 80-byte block containing the light-model-view-projection matrix and vertex buffer address. The skybox pass has a larger block that exceeds Vulkan's 128-byte minimum guarantee and remains a desktop-target portability caveat.

## Descriptor Sets Referenced by Shaders

| Set | Binding | Type | Content |
|-----|---------|------|---------|
| 0 | 0 | Uniform Buffer | Camera projection, view, and position |
| 0 | 1 | Uniform Buffer | `EnvironmentUBO`: directional light/matrix, tuning, point lights |
| 0 | 2 | Combined Image Sampler | Irradiance cubemap |
| 0 | 3 | Combined Image Sampler | Prefiltered environment cubemap |
| 0 | 4 | Combined Image Sampler | BRDF integration LUT |
| 0 | 5 | Combined Image Sampler (comparison) | Frame-local directional shadow map |
| 1 | 0 | Uniform Buffer | Joint transforms (128 × Mat4, skinning) |
| 2 | 0 | Combined Image Sampler | Base color texture |
| 2 | 1 | Combined Image Sampler | Metallic-roughness texture |
| 2 | 2 | Combined Image Sampler | Normal map |
| 2 | 3 | Combined Image Sampler | Occlusion map |
| 2 | 4 | Combined Image Sampler | Emissive map |

Material factors and GPU addresses are carried in the geometry push-constant record rather than a set-2 uniform buffer.

## See Also

- [07-rendergraph.md](07-rendergraph.md) — which passes use which shaders
- [04-vulkan-subsystem.md](04-vulkan-subsystem.md) — descriptor layout and pipeline creation
- [src/renderer/src/shaders/AGENTS.md](../../src/renderer/src/shaders/AGENTS.md) — shader contributor guide
