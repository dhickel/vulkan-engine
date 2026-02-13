# 09 - Internal Render Pipeline Mental Model

This chapter explains internals you should understand when extending or debugging the renderer.

## Runtime Flow (Facade to Vulkan)

1. App updates input through facade.
2. Facade updates camera/controller state.
3. Facade asks scene to build immutable `RenderSubmission`.
4. Vulkan core resolves handles and executes rendergraph passes.
5. Submit and present.

Relevant files:
- `src/renderer/src/api/renderer.rs`
- `src/renderer/src/scene/scene_world.rs`
- `src/renderer/src/vulkan/vk_render.rs`
- `src/renderer/src/rendergraph/passes/*`

## Submission Boundary

`SceneWorld` emits handle-based draw items:
- Mesh handles
- World transforms
- Submission flags
- Skybox environment handle

Renderer internals then resolve handles into GPU draw objects.

This boundary is a major safety contract:
- Scene code remains Vulkan-opaque.
- Vulkan/cache code stays internal.

## Rendergraph Pass Order

Current order:
1. `PrepareTargetsPass`
2. `SkyboxPass`
3. `GeometryPass`
4. `PresentCopyPass`
5. `ImguiPass`

Do not reorder blindly. Pass order carries image layout assumptions.

## Pipeline and Descriptor ABI

Geometry pipeline contract:
- Descriptor set 0: scene/environment data
- Descriptor set 1: skin/joints
- Descriptor set 2: material samplers
- Push constants: model matrix + vertex address + material metadata address

If you change descriptor layout, pipeline layout, or shader interface, update all three together.

## PBR/IBL Model Notes

Engine tracks a PBR+Unlit mixed path with skybox and generated IBL maps.

Reference baseline used by this repository:
- https://github.com/SaschaWillems/Vulkan-glTF-PBR

Use baseline for behavior expectations, not for direct API compatibility.

## Learn More

- Dynamic rendering guide:
  - https://github.khronos.org/Vulkan-Site/guide/latest/dynamic_rendering.html
- Descriptor sets guide:
  - https://github.khronos.org/Vulkan-Site/guide/latest/descriptorsets.html
- Push constants guide:
  - https://github.khronos.org/Vulkan-Site/guide/latest/push_constants.html
