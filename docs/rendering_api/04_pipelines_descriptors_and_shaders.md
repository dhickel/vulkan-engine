# 04 - Pipelines, Descriptors, and Shaders

## Rendering style contract

Current renderer uses:
- Dynamic rendering (`vkCmdBeginRendering`/`vkCmdEndRendering`).
- Traditional descriptor sets (non-bindless).
- Vertex pulling (buffer device address + push constants).

Best practice:
- Keep pipeline layout, descriptor layout, and shader bindings aligned as one contract.

Learn more:
- Dynamic rendering: https://github.khronos.org/Vulkan-Site/guide/latest/dynamic_rendering.html
- Buffer device address: https://github.khronos.org/Vulkan-Site/guide/latest/buffer_device_address.html

## Pipeline inventory

Current `VkPipelineType` values:
- `PbrMetRoughOpaque`
- `PbrMetRoughAlpha`
- `UnlitOpaque`
- `UnlitAlpha`
- `BrdfLut`
- `Skybox`
- `EnvPreFilter`
- `EnvIrradiance`

Code example (in-tree/internal):
```rust
let pipeline = self.vulkan_cache.pipelines.get_pipeline(VkPipelineType::PbrMetRoughOpaque);
self.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline.pipeline);
```

Best practice:
- Keep enum ordering and cache array ordering synchronized whenever adding new variants.

Learn more:
- Pipeline cache type: `src/renderer/src/data/data_cache.rs` (`VkPipelineCache`)

## Descriptor set layout contract

Geometry draw layout expects:
- set 0: scene/environment data
- set 1: skin/joints
- set 2: material samplers

Code example (in-tree/internal, from draw path):
```rust
self.device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::GRAPHICS, layout, 0, &[scene_desc], &[]);
self.device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::GRAPHICS, layout, 1, &[joint_desc], &[]);
self.device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::GRAPHICS, layout, 2, &[material.image_descriptor], &[]);
```

Best practice:
- Treat descriptor set number and binding index as ABI between Rust and shaders.

Learn more:
- Descriptor layout creation: `src/renderer/src/vulkan/vk_descriptor.rs` (`init_descriptor_cache`)
- Descriptor model overview: https://github.khronos.org/Vulkan-Site/guide/latest/descriptorsets.html

## Push constant contract

Per-draw geometry push constants include:
- model transform
- vertex buffer address
- material metadata address

Code example (in-tree/internal):
```rust
let push_consts = VkModelPushConsts::new(model, vertex_addr, material_meta_addr);
self.device.cmd_push_constants(
    cmd,
    layout,
    vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
    0,
    push_consts.as_byte_slice(),
);
```

Best practice:
- When updating push constant structs, update both Rust and GLSL in the same change.

Learn more:
- Push constants: https://github.khronos.org/Vulkan-Site/guide/latest/push_constants.html

## Shader manifest contract

Core shaders are loaded by manifest:
- `src/renderer/src/shaders/core_shader_manifest.txt`

Code example (in-tree/internal):
```rust
let shader_paths = data_cache::load_core_shader_manifest()?;
let shader_cache = VkShaderCache::new(device, shader_paths)?;
```

Best practice:
- Keep `.vert/.frag/.comp` and `.spv` in sync; stale SPIR-V is a common confusion source.

Learn more:
- Manifest loader: `src/renderer/src/data/data_cache.rs` (`load_core_shader_manifest`)
- glslang tools: https://github.com/KhronosGroup/glslang

## Using your own shaders today

Current supported path:
1. Edit GLSL in `src/renderer/src/shaders`.
2. Rebuild SPIR-V (`--rebuild-shaders` or env var).
3. Keep manifest mapping valid.
4. If shader interface changed, update descriptor/pipeline contracts.

Best practice:
- Validate changes incrementally: shader compile -> pipeline creation -> draw path -> validation-layer pass.

Learn more:
- Shader compile entry: `src/renderer/src/vulkan/vk_util.rs` (`compile_shaders`)
