# 07 - Gotchas, Best Practices, and Debugging

## Known alpha gotchas (current)

- Some destroy paths are incomplete (`todo!()` still present in parts of resource cleanup).
- Swapchain rebuild has explicit FIXME about old present image view lifecycle.
- `VkWindowState::update_window_scale` is marked broken.
- Environment switching can stall due to synchronous preparation.
- Several hot paths still use `unwrap`-style failure handling.

Best practice:
- Treat these as active constraints and design test coverage around resize, environment switch, and startup loading.

Learn more:
- Vulkan pitfalls: https://github.khronos.org/Vulkan-Site/guide/latest/common_pitfalls.html

## Descriptor/pipeline mismatch failures

Symptoms:
- black materials,
- wrong textures bound,
- validation layout errors.

Quick checklist:
1. set order unchanged,
2. binding indices unchanged,
3. push constant size/layout unchanged,
4. shader + descriptor + pipeline updated together.

Code example (in-tree/internal pattern):
```rust
// when changing shader bindings, update descriptor layout and write side together
let pbr_samplers = DescriptorLayoutBuilder::default()
    .add_binding(0, vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
    .add_binding(1, vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
    .add_binding(2, vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
    .add_binding(3, vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
    .add_binding(4, vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
    .build(device, vk::ShaderStageFlags::FRAGMENT, vk::DescriptorSetLayoutCreateFlags::empty())?;
```

Best practice:
- Make shader-interface changes in one commit and validate immediately.

Learn more:
- Descriptor abstractions: https://vkguide.dev/docs/new_chapter_4/descriptor_abstractions/

## Transparency artifacts

Most alpha artifacts are ordering/classification issues.

Verify:
- imported `alpha_mode` is correct,
- object lands in expected pipeline bucket,
- blend list is sorted back-to-front.

Best practice:
- Keep a dedicated transparency test scene in CI/manual smoke runs.

Learn more:
- Transparency basics: https://learnopengl.com/Advanced-OpenGL/Blending

## Resize/present regressions

If you see flicker or stale frames, validate:
1. acquired image index binding,
2. present target replacement on rebuild,
3. final `PRESENT_SRC_KHR` transition.

Best practice:
- Re-test these scenarios after presentation-flow edits:
1. skybox off / geometry on
2. skybox off / geometry off / imgui on
3. environment switch while rendering

Learn more:
- Swapchain recreation: https://vulkan-tutorial.com/Drawing_a_triangle/Swap_chain_recreation

## Validation/logging workflow (recommended)

Code example (in-tree/internal):
```bash
RUST_LOG=info cargo run -- debug_runtime testpbr
```

Then temporarily set validation on in `create_runtime_renderer` call.

Best practice:
- Fix the first validation error before handling later ones; later errors are often cascading artifacts.

Learn more:
- Validation overview: https://github.khronos.org/Vulkan-Site/guide/latest/validation_overview.html
