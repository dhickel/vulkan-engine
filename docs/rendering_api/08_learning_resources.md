# 08 - Learning Resources (Mapped to This Renderer)

## Core Vulkan references

Use these for exact API semantics:
- Vulkan 1.3 spec: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/index.html
- Vulkan Guide: https://github.khronos.org/Vulkan-Site/guide/latest/
- Vulkan Tutorial: https://vulkan-tutorial.com/

Best practice:
- Learn concepts in Guide/Tutorial, then validate edge-case behavior in the spec.

Learn more:
- Khronos Vulkan portal: https://www.khronos.org/vulkan/

## Resources aligned to this codebase

Dynamic rendering + pass flow:
- https://github.khronos.org/Vulkan-Site/guide/latest/dynamic_rendering.html

Synchronization (fences/semaphores/barriers):
- https://github.khronos.org/Vulkan-Site/guide/latest/synchronization.html
- https://github.khronos.org/Vulkan-Site/guide/latest/extensions/VK_KHR_synchronization2.html

Descriptor sets + bindings:
- https://github.khronos.org/Vulkan-Site/guide/latest/descriptorsets.html

Best practice:
- For any pass-order or layout change, revisit dynamic rendering and synchronization docs together.

Learn more:
- Sync examples: https://github.khronos.org/Vulkan-Site/guide/latest/synchronization_examples.html

## PBR and glTF references

- Sascha Willems Vulkan glTF PBR: https://github.com/SaschaWillems/Vulkan-glTF-PBR
- LearnOpenGL PBR theory: https://learnopengl.com/PBR/Theory
- glTF 2.0 spec: https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html

Best practice:
- Use Sascha Willems as conceptual baseline, not as a drop-in API contract for this engine.

Learn more:
- Khronos glTF repo: https://github.com/KhronosGroup/glTF

## Engine architecture references

- vkguide: https://vkguide.dev/
- Vulkan Memory Allocator docs: https://gpuopen-librariesandsdks.github.io/VulkanMemoryAllocator/html/

Best practice:
- Read architecture material after you can already debug one frame end-to-end.

Learn more:
- GPUOpen portal: https://gpuopen.com/
