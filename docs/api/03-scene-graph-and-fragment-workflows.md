# Scene Graph and Fragment Workflows

## 1. Purpose & Audience
This chapter explains how facade users build, mutate, and extend scene content with `Scene` and `SceneFragment`. It is aimed at Rust developers new to scene graph workflows in engines.

## 2. Where This Fits in Engine Flow
Scene authoring path:
asset load (`AssetManager`) -> scene creation/mutation (`Scene`) -> optional fragment merge (`Scene::merge_fragment`) -> renderer submission (`Renderer::render_scene*`).

## 3. Key Concepts
- `Scene` is the runtime-owned mutable graph used for rendering.
- `SceneNodeId` and `PointLightId` are slot+generation handles; they can become stale.
- `SceneFragment` is detached content meant for merge/mount workflows.
- `SceneFragmentMount` returns:
  - `mounted_root`: root node in destination scene
  - `node_mapping`: fragment node IDs mapped to scene node IDs
- `Scene::set_skybox(env)` sets scene skybox environment handle.

## 4. Code Walkthrough
Snippet Type: Real
```rust
// src/renderer/src/api/scene.rs (typical scene-node workflow)
let root = scene.create_node_default(None)?;
let child = scene.create_node_default(Some(root))?;
scene.set_transform(child, glam::Mat4::IDENTITY)?;
scene.add_mesh(child, mesh_handle)?;
```

Snippet Type: Real
```rust
// Scene fragment assembly + merge
let mut fragment = SceneFragment::new();
let frag_root = fragment.add_node_default(None)?;
let frag_child = fragment.add_node_default(Some(frag_root))?;
fragment.set_root(frag_root)?;

let mount = scene.merge_fragment(None, fragment)?;
let mounted_child = mount.node_mapping[&frag_child];
```

Snippet Type: Real
```rust
// Point-light lifecycle
let light_id = scene.create_point_light(PointLight {
    position: glam::Vec3::new(0.0, 2.0, 0.0),
    color: glam::Vec3::new(1.0, 0.95, 0.9),
    intensity: 8.0,
    range: 15.0,
})?;

scene.update_point_light(light_id, PointLight {
    position: glam::Vec3::new(0.0, 2.5, 0.0),
    color: glam::Vec3::new(1.0, 0.9, 0.8),
    intensity: 10.0,
    range: 18.0,
})?;
```

Snippet Type: Pseudocode
```text
Keep app-level scene edits in one stage of your frame loop:
  1) apply gameplay/editor mutations
  2) mount any newly completed fragments
  3) render with renderer facade
Avoid mutating and removing the same node handle in unrelated systems without ownership rules.
```

## 5. Best Practices
- Validate graph ownership in your app: decide which system can create/remove nodes.
- Keep fragment merge boundaries clean: build fragments detached, then mount once.
- Store returned handle IDs from `SceneFragmentMount` rather than re-querying by index assumptions.
- Treat scene handles as opaque IDs; never encode assumptions about slot values.

## 6. Gotchas & Failure Modes
- Passing an invalid parent to `create_node` returns `SceneError::InvalidParent`.
- Using a node handle after removal can return `SceneError::StaleNode`.
- Merging an empty fragment fails (`MergeFailed`).
- Fragment roots can be ambiguous if not set and multiple parentless nodes exist.
- Invalid point light values (non-finite color, non-positive range, negative intensity) are rejected.

## 7. Debugging Playbook
- Step 1: print node IDs (`slot`, `generation`) for failing operations.
- Step 2: if merge fails, inspect fragment root choice and parent links.
- Step 3: verify mutation ordering: remove operations should invalidate downstream users of that handle.
- Step 4: isolate by building a minimal fragment with one root and one child.
- Step 5: if rendering looks wrong, confirm merged fragment nodes actually received meshes/transforms.

## 8. Cross-Module Links
- Scene facade implementation: `src/renderer/src/api/scene.rs`
- Scene node IDs and world internals: `src/renderer/src/scene/scene_world.rs`
- API usage entrypoint: `src/renderer/examples/api_test.rs`
- Internal scene flattening notes: `docs/internal/01-rendering-pipeline-mental-model.md`

## 9. Standard References
- glTF node transform model: https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#nodes-and-hierarchy
- glam crate docs (`Mat4`, `Vec3`): https://docs.rs/glam/latest/glam/
- Engine baseline reference: https://github.com/SaschaWillems/Vulkan-glTF-PBR

## 10. See Also
- `docs/api/02-renderer-lifecycle-and-frame-api.md`
- `docs/api/04-assets-sync-deferred-and-handles.md`
- `docs/internal/01-rendering-pipeline-mental-model.md`
