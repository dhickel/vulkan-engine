# Scene Flattening and Culling

## 1. Purpose & Audience
This chapter is for contributors editing scene traversal, world-transform propagation, and renderer-facing submission payloads. It is aimed at Rust-proficient intermediate programmers who are new to engine frame-prep design.

## 2. Where This Fits in Engine Flow
Current flow:
`Renderer::render_scene(...)` -> `Scene::build_submission()` -> `SceneWorld::build_submission()` -> `VkRender::render_with_hooks(...)`.

`SceneWorld` is the boundary between hierarchical scene data (parent/child nodes) and flat per-frame draw payloads (`RenderSubmission`).

## 3. Key Concepts
- Dirty propagation:
  - A node recomputes `world_transform` when its own `dirty` flag is true or when an ancestor changed this frame.
  - Parent world transforms must be finalized before child evaluation.
- Flattening:
  - Hierarchical nodes are traversed recursively and converted into `FrameDrawItem { mesh_id, transform }` entries.
  - Flattened draw items carry stable handles and transforms only, not Vulkan objects.
- Draw item extraction:
  - Each mesh handle in a node emits one draw item with the node's `world_transform`.
- Light extraction constraints:
  - `RenderSubmission` stores `Vec<FramePointLight>` but clamps to `MAX_POINT_LIGHTS_GPU` (16) for GPU upload contract compatibility.
  - It also carries at most one `FrameDirectionalLight` for PBR direct lighting and the directional shadow pass.
  - Collection uses active entries, not raw slot order assumptions, so sparse slot churn still yields active lights.
- Culling status:
  - Frustum culling is enabled by default in `SceneWorld::build_submission`; occlusion culling is not implemented.
  - Mesh-backed nodes carry authoritative `SceneBounds` (`Known`, `Proxy`, `ConservativeVisible`) with local AABBs from registered `MeshGeometryDto`.
  - World AABBs are computed by transforming all eight corners through `node.world_transform` (not min/max only); `Aabb::transformed` handles rotation, shear, and negative scale.
  - `compute_node_world_bounds` uses explicit local proxy bounds when allowed, otherwise unions all known/proxy mesh AABBs in node-local space, then transforms to world. Any conservative-visible mesh makes the entire node conservative-visible.
  - `compute_subtree_bounds_post_order` aggregates node world bounds with all child subtree bounds. Known subtrees may prune the branch; conservative-visible subtrees always traverse children.
  - `collect_draw_items_culled` tests known/proxy node world bounds against the frustum; conservative-visible nodes submit unconditionally. Children are always traversed.
  - `CullingStats` records known/proxy tests, exact conservative-visible reasons, subtree tests/prunes, and submitted draws.
  - `RenderSubmission::bounds_references` carries exact-generation mesh handles whose bounds participated in aggregation. Command resolution marks them against the prospective submit serial, and unloaded DTO metadata retires as `RetirementClass::BoundsEntry`.
  - Descendants are tested independently when subtree pruning is unavailable (conservative-visible parent or empty group nodes).

## 4. Code Walkthrough
Snippet Type: Real
```rust
// src/renderer/src/scene/scene_world.rs
pub(crate) fn build_submission(&mut self) -> RenderSubmission {
    let mut submission = RenderSubmission::new(self.camera, 400);
    submission.skybox_env_id = self.skybox_env_id;

    for entry in self.point_lights.iter() {
        if submission.point_lights.len() >= MAX_POINT_LIGHTS_GPU {
            break;
        }
        if let Some(light) = entry.light {
            submission.point_lights.push(FramePointLight {
                position: light.position,
                color: light.color,
                intensity: light.intensity,
                range: light.range,
            });
        }
    }

    let Some(root_id) = self.root else {
        return submission;
    };
    if !self.is_valid_node_id(root_id) {
        self.root = None;
        return submission;
    }

    self.refresh_world_recursive(root_id, Mat4::IDENTITY, false);
    self.compute_subtree_bounds_post_order(root_id);
    self.collect_bounds_references(root_id, &mut submission.bounds_references);
    submission.bounds_references.sort_unstable_by_key(|mesh| (mesh.slot, mesh.generation));
    submission.bounds_references.dedup();
    let frustum = if self.enable_frustum_culling {
        Some(Frustum::from_view_projection(
            &(self.camera.projection * self.camera.view),
        ))
    } else {
        None
    };
    self.collect_draw_items_culled(root_id, &mut submission, frustum.as_ref());
    submission
}
```

Snippet Type: Real
```rust
// src/renderer/src/scene/scene_world.rs
fn refresh_world_recursive(
    &mut self,
    node_id: SceneNodeId,
    parent_world: Mat4,
    parent_dirty: bool,
) {
    let Some((world, children, propagate_dirty)) = ({
        let Some(node) = self.get_node_mut(node_id) else {
            return;
        };
        let effective_dirty = node.dirty || parent_dirty;
        if effective_dirty {
            node.world_transform = parent_world.mul_mat4(&node.local_transform);
            node.dirty = false;
        }
        Some((node.world_transform, node.children.clone(), effective_dirty))
    }) else {
        return;
    };

    for child in children {
        if self.is_valid_node_id(child) {
            self.refresh_world_recursive(child, world, propagate_dirty);
        }
    }
}
```

Snippet Type: Real
```rust
// src/renderer/src/scene/scene_world.rs
fn collect_draw_items_culled(
    &self,
    node_id: SceneNodeId,
    submission: &mut RenderSubmission,
    frustum: Option<&Frustum>,
) {
    let Some(node) = self.get_node(node_id) else {
        return;
    };

    // If subtree is known and completely outside, prune the branch.
    if let Some(f) = frustum {
        if let Some(ref subtree) = node.subtree_world_bounds {
            if subtree.is_trusted_for_pruning() {
                if let Some(aabb) = subtree.aabb() {
                    if !f.intersects_aabb(aabb) {
                        return;
                    }
                }
            }
        }
    }

    // Determine if this node's meshes should submit.
    let node_visible = match &node.node_world_bounds {
        Some(SceneBounds::Known(aabb)) | Some(SceneBounds::Proxy(aabb)) => {
            frustum.is_none_or(|f| f.intersects_aabb(aabb))
        }
        _ => true, // conservative-visible or no bounds: always submit
    };

    if node_visible {
        for mesh_id in node.meshes.iter().copied() {
            submission.push_draw_item(FrameDrawItem {
                mesh_id,
                transform: node.world_transform,
            });
        }
    }

    for child in node.children.iter().copied() {
        if self.is_valid_node_id(child) {
            self.collect_draw_items_culled(child, submission, frustum);
        }
    }
}
```

Snippet Type: Real
```rust
// src/renderer/src/scene/render_submission.rs
pub struct FrameDrawItem {
    pub mesh_id: MeshHandle,
    pub transform: Mat4,
}

pub struct RenderSubmission {
    pub camera: SceneDataUBO,
    pub draw_items: Vec<FrameDrawItem>,
    pub bounds_references: Vec<MeshHandle>,
    pub culling_stats: CullingStats,
    pub flags: SubmissionFlags,
    pub skybox_mesh_id: MeshHandle,
    pub skybox_env_id: EnvironmentHandle,
    pub point_lights: Vec<FramePointLight>,
    pub directional_light: Option<FrameDirectionalLight>,
}
```

Snippet Type: Pseudocode
```text
future_bvh_submission():
  refresh_world_recursive(root)
  candidates = query_bvh(frustum)
  submission.draw_items = pack_visible_items(candidates)
```

Snippet Type: Pseudocode
```text
simd_friendly_candidate_layout:
  centers_x[], centers_y[], centers_z[], radii[]
  mesh_slot[], mesh_generation[]
  transform_index[]

reason:
  contiguous per-field arrays reduce gather/scatter cost for frustum tests
  and can batch plane-distance math more efficiently than pointer-heavy AoS layouts.
```

## 5. Best Practices
- Keep traversal order explicit: parent transform resolution must happen before child recursion.
- Preserve handle-validation checks (`is_valid_node_id`) when touching recursive traversal.
- Keep `RenderSubmission` payload renderer-agnostic; resolve Vulkan resources later in backend code.
- Do not prune descendants from a conservative-visible node; branch pruning requires a `Known` or `Proxy` subtree bound (Phase 06).
- Keep current behavior and roadmap behavior separated in docs and code comments.
- Keep GPU payload limits explicit and synchronized (`MAX_POINT_LIGHTS_GPU` between scene submission and GPU UBO definitions).

## 6. Gotchas & Failure Modes
- Invalid root handle:
  - If `root` points to a stale/vacant slot, `build_submission` clears `root` and returns an empty draw list for the frame.
- Dirty flags not propagating:
  - If parent updates are not propagated, child world transforms become stale even when local child transforms are unchanged.
- Sparse light slots:
  - Light extraction iterates active entries and clamps by count; assumptions based on dense slot indexing can cause confusing debugging expectations.
- Proxy-bound limitations:
  - As of Phase 06, the scene graph owns CPU mesh bounds via `SceneNode.mesh_bounds` and `node_world_bounds`/`subtree_world_bounds`. `SceneBounds::ConservativeVisible` marks skinned, deformed, missing, or stale geometry. Explicit local proxy bounds are stored separately so transform/cache invalidation cannot erase the proxy input. `dungeon_dogfood` re-enables culling with authoritative chunk bounds.
  - A node proxy is not a subtree bound. Skipping recursive traversal when a parent proxy is outside can hide an in-frustum child.
- Constant drift risk:
  - `MAX_POINT_LIGHTS_GPU` exists in both scene and GPU-data modules (`src/renderer/src/scene/scene_world.rs` and `src/renderer/src/data/gpu_data.rs`); mismatches would silently corrupt or truncate light upload behavior.
- Derived invalidation (Phase 08):
  - `SceneWorld::invalidate_derived_state` must be called after any authoritative change (local transform, parent, meshes, mesh_bounds, proxy bounds). This clears `node_world_bounds`, `subtree_world_bounds` and sets `dirty = true` without clearing local proxy input. Bounds are fully rebuilt during the next `build_submission` call and before picking queries regardless of dirty state.

## 7. Debugging Playbook
- Step 1: verify root validity first (`root_id()` and `is_valid_node_id`) when submissions unexpectedly contain zero draw items.
- Step 2: verify transform propagation by forcing a root move and checking emitted child transforms (tests in `src/renderer/src/scene/scene_world.rs` already cover this pattern).
- Step 3: inspect stale child handles in `children` vectors; traversal intentionally skips invalid handles.
- Step 4: when lighting seems missing, compare active light count with `MAX_POINT_LIGHTS_GPU` and confirm sparse-slot behavior (`submission_collects_active_lights_from_sparse_slots` test case).
- Step 5: if backend lighting mismatches remain, verify packing into `EnvironmentUBO.point_lights` in `src/renderer/src/vulkan/vk_render.rs` against `GpuPointLight` layout in `src/renderer/src/data/gpu_data.rs`.

## 8. Cross-Module Links
- Scene graph and traversal: `src/renderer/src/scene/scene_world.rs`
- Submission payload types: `src/renderer/src/scene/render_submission.rs`
- GPU light layout contract: `src/renderer/src/data/gpu_data.rs`
- Backend submission consumption: `src/renderer/src/vulkan/vk_render.rs`
- Facade-level scene workflow: `docs/api/03-scene-graph-and-fragment-workflows.md`

## 9. Standard References
- Real-Time Rendering (frustum culling concepts): https://www.realtimerendering.com/
- Real-Time Collision Detection (Christer Ericson): https://realtimecollisiondetection.net/
- Vulkan Guide common pitfalls: https://github.khronos.org/Vulkan-Site/guide/latest/common_pitfalls.html
- Vulkan Guide synchronization overview: https://github.khronos.org/Vulkan-Site/guide/latest/synchronization.html
- glTF 2.0 spec: https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html

## 10. See Also
- `docs/internal/04-api-to-backend-handoff.md`
- `docs/internal/06-data-suballocation-and-transfer.md`
- `docs/internal/07-rendergraph-dependencies-and-aliasing.md`
- `src/renderer/src/data/AGENTS.md`
- `docs/internal/18-bsp-runtime-and-lifetime.md` — BSP render batches carry leaf-membership signatures for PVS filtering; BSP draw items submit through the same `build_submission` path with additional PVS-aware batch culling
