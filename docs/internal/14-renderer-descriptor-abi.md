# Renderer Descriptor ABI Manifest — Phase 01 Baseline

Sources of truth: `vk_descriptor.rs` (layout cache), `vk_pipeline.rs` (pipeline layouts),
`vk_render.rs`/`data_cache.rs` (writes and bind order), GLSL, and
`core_shader_manifest.txt` (source/SPIR-V runtime pairs). Counts are one for every binding and
all dynamic-offset rules are **none**.

## Cached Rust descriptor layouts

| `VkDescType` | Set/binding(s), Vulkan type, stages | Ownership and image/buffer contract | Live pipeline consumers and shader pair |
|---|---|---|---|
| `DrawImage` | set 0 b0 `STORAGE_IMAGE`, compute | Cached layout; 2D storage image, `GENERAL`; no frame allocation found. | No live pipeline consumer. `sky.comp` / `sky.comp.spv` exist but are absent from `core_shader_manifest.txt`. |
| `SceneData` | set 0: b0/b1 `UNIFORM_BUFFER`; b2–b5 `COMBINED_IMAGE_SAMPLER`; vertex+fragment | Per-frame scene descriptors. b0 `SceneDataUBO` 144 B (projection 0, view 64, cam_pos 128, pad 140). b1 `EnvironmentUBO` 656 B (table below). b2/b3 cube, b4 2D, b5 2D comparison depth. | PBR/unlit: `pbr_base.vert(.spv)`, `material_pbr.frag(.spv)`, `material_unlit.frag(.spv)`. |
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
| 1 | `UBOParams` / `EnvironmentUBO` | 656-byte UBO | PBR/unlit fragment |
| 2 | `samplerCube samplerIrradiance` | cube sampled image | PBR IBL |
| 3 | `samplerCube prefilteredMap` | cube sampled image | PBR IBL |
| 4 | `sampler2D samplerBRDFLUT` | 2D RG LUT | PBR IBL |
| 5 | `sampler2DShadow shadowMap` | 2D D32 comparison image, currently 2048² | PBR directional shadow |

### Binding 5 CSM migration delta (not applied)

Keep set 0 binding 5 and descriptor type `COMBINED_IMAGE_SAMPLER`, but replace the 2D image/view
with a three-layer 1024² D32 2D-array image/view and change GLSL to
`sampler2DArrayShadow`. Coordinates change from `vec3(uv, compareDepth)` to
`vec4(uv, cascadeLayer, compareDepth)`. Add cascade matrices/splits to a synchronized Rust/GLSL
uniform contract, update descriptor writes, shader source and SPIR-V together, and preserve a
comparison-enabled sampler.

## `EnvironmentUBO` std140-compatible offsets (656 B)

| Field | Offset | Size |
|---|---:|---:|
| `light_dir` | 0 | 16 |
| `light_color` | 16 | 16 |
| `light_view_proj` | 32 | 64 |
| `exposure`, `gamma`, `prefilter_mips_levels`, `ibl_ambient_scale` | 96, 100, 104, 108 | 4 each |
| `debug_view_inputs`, `debug_view_equation` | 112, 116 | 4 each |
| `_pad0` | 120 | 8 |
| `point_light_count` | 128 | 4 |
| `_pad1` | 132 | 12 |
| `point_lights[16]` | 144 | 512 |

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
order, GLSL declarations, shader manifest pairs, critical Rust sizes, and this document’s CSM
compatibility marker. Any ABI change updates all of those as one compatibility unit.
