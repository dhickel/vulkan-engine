# Renderer Descriptor ABI Manifest — Phase 08 CSM Baseline

Sources of truth: `vk_descriptor.rs` (layout cache), `vk_pipeline.rs` (pipeline layouts),
`vk_render.rs`/`data_cache.rs` (writes and bind order), GLSL, and
`core_shader_manifest.txt` (source/SPIR-V runtime pairs). Counts are one for every binding and
all dynamic-offset rules are **none**.

## Cached Rust descriptor layouts

| `VkDescType` | Set/binding(s), Vulkan type, stages | Ownership and image/buffer contract | Live pipeline consumers and shader pair |
|---|---|---|---|
| `DrawImage` | set 0 b0 `STORAGE_IMAGE`, compute | Cached layout; 2D storage image, `GENERAL`; no frame allocation found. | No live pipeline consumer. `sky.comp` / `sky.comp.spv` exist but are absent from `core_shader_manifest.txt`. |
| `SceneData` | set 0: b0/b1 `UNIFORM_BUFFER`; b2–b5 `COMBINED_IMAGE_SAMPLER`; vertex+fragment | Per-frame scene descriptors. b0 `SceneDataUBO` 144 B (projection 0, view 64, cam_pos 128, pad 140). b1 `EnvironmentUBO` 2048 B (table below). b2/b3 cube, b4 2D, b5 2D-array comparison depth. | PBR/unlit: `pbr_base.vert(.spv)`, `material_pbr.frag(.spv)`, `material_unlit.frag(.spv)`. |
| `PbrSamplers` | set 2 b0–b4 `COMBINED_IMAGE_SAMPLER`, fragment | Per-material, persistent across frames. Five 2D samplers: base color, metallic/roughness, normal, occlusion, emissive. | PBR/unlit fragment pairs above. |
| `PbrProperties` | cached set layout b0 `STORAGE_BUFFER`, fragment | **Not in a live pipeline layout and never bound.** `MaterialValues` CPU storage is 80 B, but live shaders access it through the buffer-device-address `MaterialMeta` reference in push constants, not this descriptor. | No live consumer. The only set-3 declaration is under `shaders/ignored/` and is non-runtime. |
| `SkinData` | set 1 b0 `UNIFORM_BUFFER`, vertex | Persistent/default or mesh skin descriptor. `mat4 jointMatrix[128]`, 8192 B. | `pbr_base.vert(.spv)` in both PBR and unlit pipelines. |
| `Skybox` | set 0 b0 `COMBINED_IMAGE_SAMPLER`, fragment | Per-environment persistent cube sampler. | `skybox.frag(.spv)` with `skybox.vert(.spv)`. |
| `EnvIrradiance` | set 0 b0 `COMBINED_IMAGE_SAMPLER`, fragment | Environment-generation cube sampler. | `env_irradiance_cube.frag(.spv)` + `filtered_cube.vert(.spv)`. |
| `EnvPreFilter` | set 0 b0 `COMBINED_IMAGE_SAMPLER`, fragment | Environment-generation cube sampler. | `env_prefilter_cube.frag(.spv)` + `filtered_cube.vert(.spv)`. |
| `EnvEquirect` | set 0 b0 `COMBINED_IMAGE_SAMPLER`, fragment | Environment-generation 2D equirectangular sampler. | `env_equirect_to_cube.frag(.spv)` + `filtered_cube.vert(.spv)`. |
| `Empty` | no bindings | Empty set layout; no descriptor payload. | BRDF LUT: `gen_brd_flut.vert(.spv)` + `gen_brd_flut.frag(.spv)`. |

The shared Rust scene layout has vertex+fragment stage visibility for every binding even where a
particular GLSL stage does not declare every member. That broader Rust visibility is part of the
current ABI.

## Live scene set 0 bindings

| Binding | GLSL declaration | Dimensionality / format | Pipeline use |
|---:|---|---|---|
| 0 | `UBO { mat4 projection; mat4 view; vec3 camPos; }` | 144-byte UBO | PBR/unlit vertex and fragment |
| 1 | `UBOParams` / `EnvironmentUBO` | 2048-byte UBO (CSM and multi-directional extended) | PBR/unlit fragment |
| 2 | `samplerCube samplerIrradiance` | cube sampled image | PBR IBL |
| 3 | `samplerCube prefilteredMap` | cube sampled image | PBR IBL |
| 4 | `sampler2D samplerBRDFLUT` | 2D RG LUT | PBR IBL |
| 5 | `sampler2DArrayShadow shadowMap` | 2D-array D32 comparison image, 3×1024² | PBR directional CSM shadow |

### Binding 5 CSM migration (applied Phase 08)

Binding 5 is now `sampler2DArrayShadow`. The descriptor type remains `COMBINED_IMAGE_SAMPLER`.
When CSM is active: a three-layer 1024² D32 2D-array image is bound with one comparison sampler.
Coordinates are `vec4(uv, cascadeLayer, compareDepth)`. Cascade matrices/splits are in a
synchronized Rust/GLSL `EnvironmentUBO` contract. When CSM is inactive (legacy path), layer 0
is sampled with the same array view.

## `EnvironmentUBO` std140-compatible offsets (2048 B, CSM/multi-light extended)

| Field | Offset | Size |
|---|---:|---:|
| `light_dir` | 0 | 16 |
| `light_color` | 16 | 16 |
| `light_view_proj` | 32 | 64 |
| `exposure`, `gamma`, `prefilter_mips_levels`, `ibl_ambient_scale` | 96, 100, 104, 108 | 4 each |
| `debug_view_inputs`, `debug_view_equation` | 112, 116 | 4 each |
| `cascade_count`, `directional_light_count` | 120, 124 | 4 each |
| `cascade_splits` | 128 | 16 |
| `point_light_count`, `_pad1` | 144, 148 | 4 + 12 |
| `spot_light_count`, `_pad_spot` | 160, 164 | 4 + 12 |
| `cascade_view_proj[3]` | 176 | 192 |
| `blend_fraction`, `_pad_blend` | 368, 372 | 4 + 12 |
| `point_lights[16]` | 384 | 512 |
| `spot_lights[16]` | 896 | 1024 |
| `directional_lights[4]` | 1920 | 128 |

## Pipeline set order and push constants

| Pipeline | Descriptor set order | Push constants / source pair |
|---|---|---|
| PBR and unlit mesh | set 0 `SceneData`, set 1 `SkinData`, set 2 `PbrSamplers` | `VkModelPushConsts`, 96 B, vertex+fragment. `MaterialMeta` is an 80-byte std430 buffer-reference payload reached by the address at offset 72. |
| Shadow depth | none | `PushConstShadowDepth`, 80 B, vertex; `shadow_depth.vert(.spv)` + `shadow_depth.frag(.spv)`. |
| Skybox | set 0 `Skybox` | `PushConstSkyBox`, 144 B, vertex+fragment. |
| Irradiance | set 0 `EnvIrradiance` | `PushConstIrradiance`, 80 B, vertex+fragment. |
| Prefilter | set 0 `EnvPreFilter` | `PushConstPrefilterEnv`, 80 B, vertex+fragment. |
| Equirect-to-cube | set 0 `EnvEquirect` | `PushConstCubeCapture`, 80 B, vertex+fragment. |
| BRDF LUT | set 0 `Empty` | none. |

Mesh draw bind order in `vk_render.rs` is scene set 0, joint set 1, and material image set 2.
There is no set 3 bind and no dynamic descriptor offset.

## Executable drift guard

`src/renderer/tests/descriptor_abi.rs` checks the live Rust layout declarations, pipeline set
order, GLSL declarations, shader manifest pairs, critical Rust sizes, and this document's CSM
compatibility marker. Any ABI change updates all of those as one compatibility unit.

## BSP descriptor ABI (feature `renderer/bsp`, Phase 01 frozen baseline)

**Frozen**: 2026-07-23. Set 0 remains compatible with shared scene bindings. In-flight descriptors are never mutated for frame-varying state. Static textures use one array layer.

BSP lightmapped surfaces use a dedicated descriptor path **separate** from PBR. BSP set 0
reuses the `SceneData` six-binding layout so scene data and shadow map are shared. BSP set 1
is a material-specific layout with four bindings. BSP set 2 carries frame-local values.

| `VkDescType` | Set/binding(s), Vulkan type, stages | Ownership and image/buffer contract | Live pipeline consumers and shader pair |
|---|---|---|---|
| `BspScene` | set 0: b0/b1 `UNIFORM_BUFFER`; b2–b5 `COMBINED_IMAGE_SAMPLER`; vertex+fragment | Same binding structure as `SceneData`. b0 `SceneDataUBO`, b1 `EnvironmentUBO`, b2-b3 cube env, b4 BRDF LUT, b5 shadow array. Vertex + fragment visibility. Compatible with shared scene bindings — BSP and PBR can share the same set 0 descriptor set at bind time. | BSP opaque, fullbright, alpha-mask, sky, liquid pipelines. |
| `BspMaterial` | set 1: b0/b1/b2 `COMBINED_IMAGE_SAMPLER`, b3 `UNIFORM_BUFFER`; fragment | b0 albedo 2D (one array layer for static textures), b1 fullbright mask 2D, b2 lightmap atlas `sampler2DArray` (one layer per populated style), b3 `BspSurfaceUniform` UBO (48 bytes). The vertex shader must not read set 1. | All BSP fragment shaders in `bsp_shader_manifest.txt`. |
| `BspFrameValues` | set 2: b0 `UNIFORM_BUFFER`; fragment | Frame-local BSP values: style intensities (vec4 × 64), animation frame indices, liquid simulation parameters, deterministic simulation values. Written once per frame max. In-flight descriptors are never mutated. | All BSP fragment shaders. |

### BSP surface UBO (set 1, binding 3) — `BspSurfaceUniform`

Rust and GLSL size: 48 bytes (std140 scalar fields packed into two 16-byte groups after `lightmapScaleBias`).

| Field | GLSL type | Rust type | Notes |
|---|---|---|---|
| `lightmapScaleBias` | `vec4` | `Vec4` | xy = atlas UV scale, zw = atlas UV offset |
| `styleIndex` | `uint` | `u32` | active light style layer index (0–63) |
| `fullbrightBase` | `uint` | `u32` | first palette index in fullbright range |
| `fullbrightCount` | `uint` | `u32` | number of fullbright entries |
| `alphaThreshold` | `float` | `f32` | alpha test threshold (default 0.5) |
| `animationFrame` | `uint` | `u32` | current animation frame layer |
| `animationTime` | `float` | `f32` | engine time ticks (0.1s resolution) |

### BSP pipeline set order and push constants

| Pipeline | Descriptor set order | Push constants / source pair |
|---|---|---|
| BSP opaque | set 0 `BspScene`, set 1 `BspMaterial`, set 2 `BspFrameValues` | `mat4 model` + `vertex_buffer_addr`, 80 B, vertex. `bsp_lightmapped.vert(.spv)` + `bsp_lightmapped.frag(.spv)`. |
| BSP fullbright | same as opaque | Same shader pair; fullbright path is inside fragment. |
| BSP alpha mask | set 0 `BspScene`, set 1 `BspMaterial`, set 2 `BspFrameValues` | Same push constants. Two-sided (no cull). `bsp_lightmapped.vert(.spv)` + `bsp_lightmapped.frag(.spv)`. |
| BSP sky | set 0 `BspScene`, set 1 `BspMaterial`, set 2 `BspFrameValues` in the shared BSP pipeline layout; shader reads set 0 only. Set 1/2 remain layout-compatible and may be bound for uniform bind order. | `mat4 model` + `vertex_buffer_addr`, 80 B, vertex. `bsp_lightmapped.vert(.spv)` + `bsp_sky.frag(.spv)`. |
| BSP liquid | set 0 `BspScene`, set 1 `BspMaterial`, set 2 `BspFrameValues` | Same push constants. Two-sided, alpha blend. `bsp_lightmapped.vert(.spv)` + `bsp_liquid.frag(.spv)`. |

**Frame-varying update rule (frozen)**: In-flight descriptors are never mutated. Frame-local BSP values (style intensities, animation indices, liquid parameters) are written through fresh or frame-rotated set 2 descriptors each frame, not by mutating descriptors that may still be in flight. Static textures (albedo, fullbright mask) use one array layer — per-frame animation changes are communicated via the animationFrame/animationTime uniforms, not by rewriting texture bindings.

### BSP ABI exec guard

BSP descriptor bindings are tested in `descriptor_abi_bsp_bindings_registered` (feature-gated behind `#[cfg(feature = "bsp")]`). BSP shader manifest is `src/renderer/src/shaders/bsp_shader_manifest.txt`.
