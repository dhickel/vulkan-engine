# 06 - Assets, Environments, and Uploads

## Asset loader path (current)

Active loader path is Assimp-based:
- `assimp_util::load_model(path, data_cache, has_animation)`
- returns `ModelMeta { scene_world, material_ids, mesh_ids }`

Code example (in-tree/internal):
```rust
use crate::data::assimp_util;

let loaded = assimp_util::load_model("assets/my_scene.glb", data_cache.clone(), false)?;
let scene_world = loaded.scene_world;
```

Best practice:
- Keep model ingest and GPU allocation separate so failures are easier to isolate.

Learn more:
- Loader implementation: `src/renderer/src/data/assimp_util.rs`
- Assimp project: https://github.com/assimp/assimp

## Using your own startup asset today

Simplest path:
- Change `DEFAULT_STARTUP_MODEL_PATH` in `src/renderer/src/scene/debug_scenarios.rs`.

Code example:
```rust
pub const DEFAULT_STARTUP_MODEL_PATH: &str = "src/renderer/src/assets/MyScene.glb";
```

Best practice:
- Keep one tiny known-good asset and one large stress asset; run both regularly.

Learn more:
- Startup debug scenarios: `src/renderer/src/scene/debug_scenarios.rs`

## Explicit cache allocation (deferred loading flow)

After ingest, mesh/material resources are promoted by cache allocation functions.

Code example (in-tree/internal):
```rust
use crate::vulkan::vk_storage::BufferPlacement;
use crate::data::data_cache::LoadResult;

{
    let mut mesh_cache = data_cache.mesh_cache.lock().unwrap();
    let mesh_result = mesh_cache.allocate_ids(
        &loaded.mesh_ids,
        BufferPlacement::ContiguousPreferred,
        false,
    );
    if matches!(mesh_result, LoadResult::Failed(_)) {
        return Err("Mesh allocation failed".to_string());
    }
}

{
    let mut tex_cache = data_cache.texture_cache.lock().unwrap();
    let mat_result = tex_cache.allocate_ids(
        &loaded.material_ids,
        BufferPlacement::ContiguousPreferred,
        false,
    );
    if matches!(mat_result, LoadResult::Failed(_)) {
        return Err("Material allocation failed".to_string());
    }
}
```

Best practice:
- Check `LoadResult` explicitly and fail fast on partial/failed allocation.

Learn more:
- Cache APIs: `src/renderer/src/data/data_cache.rs`

## Procedural asset path (manual mesh/material creation)

You can create mesh and material metadata directly.

Code example (in-tree/internal, simplified):
```rust
use crate::data::gpu_data::{MaterialMeta, MeshMeta, Vertex};
use crate::vulkan::vk_storage::BufferPlacement;

let mut tex_cache = data_cache.texture_cache.lock().unwrap();
let mat_id = tex_cache.add_material(MaterialMeta::pbr_simple(
    glam::vec4(1.0, 0.7, 0.2, 1.0),
    0.1,
    0.8,
));

let mut mesh_cache = data_cache.mesh_cache.lock().unwrap();
let mesh_id = mesh_cache.add(MeshMeta {
    name: "ProceduralTriangle".to_string(),
    vertices: vec![Vertex::default(), Vertex::default(), Vertex::default()],
    indices: vec![0, 1, 2],
    material_index: Some(mat_id),
});

let _ = mesh_cache.allocate_id(mesh_id, BufferPlacement::ContiguousPreferred, false);
let _ = tex_cache.allocate_id(mat_id, BufferPlacement::ContiguousPreferred, false);
```

Best practice:
- Validate vertex data conventions (normals/tangents/UVs) before blaming shader logic.

Learn more:
- Vertex/material types: `src/renderer/src/data/gpu_data.rs`

## Environment and skybox usage

`RenderSubmission.skybox_env_id` requests the active environment.
Renderer ensures environment resources before switching.

Code example (in-tree/internal):
```rust
let mut submission = scene_world.build_submission();
submission.skybox_env_id = desired_env;
app.render(frame, &submission);
```

Best practice:
- Treat environment switching as potentially expensive; avoid toggling every frame.

Learn more:
- Env prep/switch path: `src/renderer/src/vulkan/vk_render.rs` (`ensure_environment_ready`, `prepare_submission_environment`)
- PBR reference: https://github.com/SaschaWillems/Vulkan-glTF-PBR

## What external users can control today

External crate users (without engine-internal edits) can currently:
- Run renderer (`renderer::run()`).

External users cannot yet directly:
- load arbitrary asset paths through stable public API,
- build/submitting custom `SceneWorld` payloads,
- manage deferred loading pipeline directly.

Best practice:
- Treat this as alpha API gap feedback and prioritize a public façade if dogfooding confirms these needs.

Learn more:
- Gap review: `.internal-dev/reviews/2026-02-13-rendering-api-dogfood-gaps.md`
