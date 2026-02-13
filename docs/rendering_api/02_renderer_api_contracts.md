# 02 - Renderer API Contracts

## Contract boundary: `SceneWorld -> RenderSubmission`

Renderer-facing frame payload is `RenderSubmission`.

Core types:
- `SceneWorld` (`src/renderer/src/scene/scene_world.rs`)
- `RenderSubmission` (`src/renderer/src/scene/render_submission.rs`)
- `FrameDrawItem { mesh_id, transform }`

Important submission fields:
- `camera: SceneDataUBO`
- `draw_items: Vec<FrameDrawItem>`
- `flags: SubmissionFlags`
- `skybox_env_id: EnvironmentHandle`

Code example (in-tree/internal):
```rust
scene_world.update_camera(view, proj, cam_pos);
let mut submission = scene_world.build_submission();

submission.flags.draw_skybox = true;
submission.flags.draw_geometry = true;
submission.flags.draw_imgui = true;

app.render(frame_number, &submission);
```

Best practice:
- Keep Vulkan handles and raw pointers out of scene/gameplay boundary types.

Learn more:
- Render submission source: `src/renderer/src/scene/render_submission.rs`

## Stable handle contract (slot + generation)

Resources are referenced by stable handles:
- `MeshHandle`
- `MaterialHandle`
- `TextureHandle`
- `EnvironmentHandle`

Invalid/stale handles return cache errors (`StaleHandle`, `InvalidHandle`, `NotLoaded`, `OutOfBounds`).

Code example (in-tree/internal):
```rust
let mesh = mesh_cache.get_loaded_id(mesh_handle)?; // fails if stale/not loaded
let mat_ptr = tex_cache.get_loaded_material_ptr(mesh.material_id)?;
```

Best practice:
- Never persist raw slot indices alone; always carry full handles.

Learn more:
- Handle definitions: `src/renderer/src/data/handles.rs`
- Slot-map concept: https://docs.rs/slotmap/latest/slotmap/

## Reserved fallback resources

Reserved IDs currently used by engine internals:
- textures: `0..=5`
- materials: `0..=1`
- mesh: `0` (`SKYBOX_MESH`)

Code example (in-tree/internal):
```rust
let default_color = TextureCache::DEFAULT_COLOR_TEX;
let default_error = TextureCache::DEFAULT_ERROR_TEX;
let skybox_mesh = MeshCache::SKYBOX_MESH;
```

Best practice:
- Do not deallocate reserved entries through unchecked APIs unless you are intentionally changing core engine contracts.

Learn more:
- Defaults and deallocation policy: `src/renderer/src/data/data_cache.rs`

## Scene graph contract

`SceneWorld` owns node hierarchy and emits draw items each frame.

Code example (in-tree/internal):
```rust
use crate::scene::scene_world::{SceneNode, SceneWorld};

let mut world = SceneWorld::new();
let root = world.add_node(None, SceneNode::default());
world.set_root(root);

// attach mesh handles to nodes via get_node_mut(...)
if let Some(root_node) = world.get_node_mut(root) {
    root_node.meshes.push(mesh_id);
}
```

Best practice:
- Validate handles when mutating scene nodes and avoid holding stale IDs across destructive scene edits.

Learn more:
- Scene traversal and submission build: `src/renderer/src/scene/scene_world.rs`

## Public API limitation (current alpha)

The examples above are in-tree patterns. External crate consumers currently have only `renderer::run()` as stable API.

Best practice:
- If you need external scene/asset submission APIs, treat this as an alpha gap and track required façade design before release-hardening.

Learn more:
- Gap review: `.internal-dev/reviews/2026-02-13-rendering-api-dogfood-gaps.md`
