# Scene Construction

This legacy chapter is retained as a stable link target. The current scene API
reference lives in [03-scene-graph-and-fragment-workflows.md](03-scene-graph-and-fragment-workflows.md).

Use that chapter for the implemented contracts around:

- `Scene`, `SceneFragment`, and `Scene::merge_fragment`;
- default-on frustum culling through `set_frustum_culling` and `frustum_culling_enabled`;
- one scene `DirectionalLight` with the fixed frame-local directional shadow path;
- `SceneFragmentMount::mounted_root` and `SceneFragmentMount::node_mapping`;
- command-backed editor placement and transform workflows;
- versioned scene persistence, durable asset references, tags, and material override metadata.

Material override metadata is durable scene metadata only. The implemented API is
`Scene::set_node_material_override(node, slot, override_id)`, which records a
slot string such as `"0"` and a stable override ID such as
`"mat_override.damp_stone"`. It does not mutate GPU material cache state, edit
PBR factors, assign textures, or provide a first-class material authoring API.

Current model loading and fragment creation are documented in
[04-assets-sync-deferred-and-handles.md](04-assets-sync-deferred-and-handles.md).
