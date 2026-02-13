# 09 - Examples and Recipes

This file is a practical cookbook. Each recipe marks whether it is currently external/public or in-tree/internal.

## Recipe 1: Run the renderer (external/public)

```rust
fn main() {
    renderer::run();
}
```

Best practice:
- Use this as a baseline smoke test before changing internals.

Learn more:
- `src/main.rs`

## Recipe 2: Use your own startup asset (in-tree/internal)

Edit startup model constant:

```rust
// src/renderer/src/scene/debug_scenarios.rs
pub const DEFAULT_STARTUP_MODEL_PATH: &str = "src/renderer/src/assets/MyGameScene.glb";
```

Run:
```bash
cargo run
```

Best practice:
- Use relative in-repo paths first; path handling is currently not a polished public API.

Learn more:
- Startup loader: `src/renderer/src/scene/debug_scenarios.rs`

## Recipe 3: Load model + allocate caches manually (in-tree/internal)

```rust
use crate::data::assimp_util;
use crate::data::data_cache::LoadResult;
use crate::vulkan::vk_storage::BufferPlacement;

fn load_scene_into_caches(path: &str, data_cache: std::sync::Arc<crate::data::data_cache::VkDataCache>)
    -> Result<crate::scene::scene_world::SceneWorld, String>
{
    let loaded = assimp_util::load_model(path, data_cache.clone(), false)?;

    {
        let mut mesh_cache = data_cache.mesh_cache.lock().unwrap();
        let result = mesh_cache.allocate_ids(&loaded.mesh_ids, BufferPlacement::ContiguousPreferred, false);
        if matches!(result, LoadResult::Failed(_)) {
            return Err("Mesh allocation failed".to_string());
        }
    }

    {
        let mut tex_cache = data_cache.texture_cache.lock().unwrap();
        let result = tex_cache.allocate_ids(&loaded.material_ids, BufferPlacement::ContiguousPreferred, false);
        if matches!(result, LoadResult::Failed(_)) {
            return Err("Material allocation failed".to_string());
        }
    }

    Ok(loaded.scene_world)
}
```

Best practice:
- Keep allocation error handling explicit; silent partial load is difficult to debug.

Learn more:
- `src/renderer/src/data/assimp_util.rs`
- `src/renderer/src/data/data_cache.rs`

## Recipe 4: Build a tiny procedural scene (in-tree/internal)

```rust
use crate::data::gpu_data::{MaterialMeta, MeshMeta, Vertex};
use crate::scene::scene_world::{SceneNode, SceneWorld};
use crate::vulkan::vk_storage::BufferPlacement;

fn build_procedural_scene(data_cache: std::sync::Arc<crate::data::data_cache::VkDataCache>)
    -> Result<SceneWorld, String>
{
    let mut tex_cache = data_cache.texture_cache.lock().unwrap();
    let mat_id = tex_cache.add_material(MaterialMeta::pbr_simple(
        glam::vec4(0.2, 0.8, 1.0, 1.0),
        0.2,
        0.7,
    ));

    let mut mesh_cache = data_cache.mesh_cache.lock().unwrap();
    let mesh_id = mesh_cache.add(MeshMeta {
        name: "ProcTri".to_string(),
        vertices: vec![Vertex::default(), Vertex::default(), Vertex::default()],
        indices: vec![0, 1, 2],
        material_index: Some(mat_id),
    });

    let _ = mesh_cache.allocate_id(mesh_id, BufferPlacement::ContiguousPreferred, false);
    let _ = tex_cache.allocate_id(mat_id, BufferPlacement::ContiguousPreferred, false);

    let mut world = SceneWorld::new();
    let root = world.add_node(None, SceneNode::default());
    world.set_root(root);

    if let Some(root_node) = world.get_node_mut(root) {
        root_node.meshes.push(mesh_id);
    }

    Ok(world)
}
```

Best practice:
- Start procedural content with a single mesh/material, then expand to multi-node scene composition.

Learn more:
- `src/renderer/src/scene/scene_world.rs`
- `src/renderer/src/data/gpu_data.rs`

## Recipe 5: Submit frame with custom flags (in-tree/internal)

```rust
let mut submission = scene_world.build_submission();
submission.flags.draw_skybox = false;
submission.flags.draw_geometry = true;
submission.flags.draw_imgui = true;

app.render(frame_number, &submission);
```

Best practice:
- Validate all flag combinations when changing rendergraph pass sequencing.

Learn more:
- Rendergraph passes: `src/renderer/src/rendergraph/passes/*.rs`

## Recipe 6: Request environment switch via submission (in-tree/internal)

```rust
let mut submission = scene_world.build_submission();
submission.skybox_env_id = my_environment_handle;
app.render(frame_number, &submission);
```

Best practice:
- Do not toggle environment every frame; switching can trigger expensive preparation.

Learn more:
- Environment switch logic: `src/renderer/src/vulkan/vk_render.rs` (`prepare_submission_environment`)

## Recipe 7: Internal deferred-loading pipe submit (in-tree/internal)

```rust
host_buffer.submit_transfer_commands(
    crate::vulkan::vk_types::VkSubmitParam::signaling(vk::PipelineStageFlags2::ALL_TRANSFER)
)?;
host_buffer.submit_graphics_commands(
    crate::vulkan::vk_types::VkSubmitParam::waiting(vk::PipelineStageFlags2::VERTEX_SHADER)
)?;

// blocks background thread until fence queue retires submitted work
host_buffer.await_done(10)?;
```

Can external users call this today?
- Not as stable public API; this is internal engine plumbing.

Best practice:
- Keep queue submission ownership centralized on render thread unless redesigning synchronization architecture.

Learn more:
- `src/renderer/src/vulkan/vk_types.rs` (`VkHostBuffer`, `VkTransfer`, `VkFenceQueue`)

## Recipe 8: Add a new debug runtime scenario (in-tree/internal)

Typical path:
1. Add new mode label in `DebugRuntimeMode`.
2. Extend arg parsing in `parse_debug_runtime_mode`.
3. Add scene/material behavior in `debug_scenarios` or startup flow.

Best practice:
- Keep debug scenarios deterministic and narrowly scoped to one rendering behavior under test.

Learn more:
- `src/renderer/src/lib.rs`
- `src/renderer/src/vulkan/vk_render.rs`
- `src/renderer/src/scene/debug_scenarios.rs`
