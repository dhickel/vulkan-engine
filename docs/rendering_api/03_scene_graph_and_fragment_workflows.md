# 03 - Scene Graph and Fragment Workflows

This chapter explains how to build and mutate scenes using the public API.

## Core Scene Types

- `Scene`
- `SceneNodeId` (slot + generation stable handle)
- `SceneFragment`
- `SceneFragmentNodeId`
- `SceneFragmentMount`

The scene API is renderer-facing and Vulkan-opaque.

## Node Lifecycle

Create nodes:
- `create_node(parent, transform)`
- `create_node_default(parent)`

Mutate nodes:
- `set_transform(node, transform)`
- `add_mesh(node, mesh_handle)`
- `clear_meshes(node)`

Remove nodes:
- `remove_node(node)` (recursive)

Read state:
- `root()`
- `transform(node)`

## Handle Safety Model

`SceneNodeId` uses slot+generation semantics.

Errors:
- `SceneError::InvalidNode` for out-of-bounds/vacant handles.
- `SceneError::StaleNode` for generation mismatch (old handle reused slot).
- `SceneError::InvalidParent` for invalid parent handles.

Rule:
- Store whole handles, never raw slot indices.

## Fragment Merge Workflow

Use fragments when importing model hierarchies or reusable scene chunks.

High-level:
1. Load/create `SceneFragment`.
2. `scene.merge_fragment(parent, fragment)`.
3. Use returned `SceneFragmentMount.node_mapping` to post-adjust transforms.

Example:
```rust
let fragment = {
    let mut assets = renderer.assets();
    assets.load_model("src/renderer/src/assets/DamagedHelmet.glb")?
};

let mount = scene.merge_fragment(None, fragment)?;
scene.set_transform(mount.mounted_root, glam::Mat4::IDENTITY)?;
```

Merge validation catches:
- Cycles (`SceneError::CycleDetected`)
- Ambiguous/missing root
- Disconnected nodes
- Out-of-bounds references
- Multi-parent violations

## Camera and Skybox from Scene

- `set_camera(view, projection, position)`
- `set_skybox(environment_handle)`

Most apps can let renderer camera controller drive view/projection and use these for explicit overrides only.

## Learn More

- Asset-to-fragment import path: `04_assets_sync_deferred_and_handles.md`
- Runtime environment control: `05_environment_and_skybox_runtime.md`
- Scene implementation: `src/renderer/src/api/scene.rs`
