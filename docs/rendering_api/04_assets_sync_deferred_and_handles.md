# 04 - Assets: Sync, Deferred, and Handles

This chapter covers practical asset loading with the public `AssetManager` facade.

## Access Pattern

`AssetManager` is borrowed from renderer:
```rust
let mut assets = renderer.assets();
```

Borrowing rule:
- Keep asset manager scopes short so you can return to render calls quickly.

## Synchronous Loads

Methods:
- `load_mesh(path) -> Result<MeshHandle, AssetError>`
- `load_texture(path) -> Result<TextureHandle, AssetError>`
- `load_texture_with_options(path, options) -> Result<TextureHandle, AssetError>`
- `load_model(path) -> Result<SceneFragment, AssetError>`
- `load_environment(source) -> Result<EnvironmentHandle, AssetError>`

Use sync loads for:
- Startup bootstrap.
- Editor tools.
- Explicit loading screens where frame stall is acceptable.

Note:
- `load_mesh` currently imports model path and returns first mesh found.

## Texture Policy and `.meta` Sidecars

Renderer config includes asset policy:
- `RendererConfig.asset_policy.manifest_mode`: `Disabled | BestEffort | Strict`
- `RendererConfig.asset_policy.allow_filename_heuristics`: enable `_normal`, `_mr`, etc. hints
- `RendererConfig.asset_policy.compression.mode`: `Disabled | Auto | Force`

Texture policy precedence:
1. Per-call `TextureLoadOptions`
2. `<texture_path>.meta` TOML sidecar
3. Filename heuristics
4. Engine defaults

Minimal sidecar example:
```toml
is_srgb = false
generate_mips = true

[sampler]
wrap_u = "repeat"
wrap_v = "repeat"
min_filter = "linear_mipmap_linear"
mag_filter = "linear"
max_anisotropy = 8
```

## Deferred Loads (Ticket Pipeline)

Methods:
- `request_model_load(path) -> Result<LoadTicket, AssetError>`
- `request_texture_load(path) -> Result<LoadTicket, AssetError>`
- `request_texture_load_with_options(path, options) -> Result<LoadTicket, AssetError>`
- `poll_model_load(ticket) -> LoadStatus<SceneFragment>`
- `poll_texture_load(ticket) -> LoadStatus<TextureHandle>`
- `cancel_load(ticket) -> Result<(), AssetError>`

Status model:
- `Pending { queued_at }`
- `Uploaded { value }`
- `Failed { error }`
- `Cancelled`

Key behavior (current alpha implementation):
- Deferred scheduler supports bounded concurrency (default max 4 in-flight jobs).
- Texture upload finalization is polled during asset pumping.
- Terminal ticket states are retained in bounded history.

## Pumping Rules

Deferred progress requires pumping.

Automatic pump points:
- `render_scene`
- `begin_frame`

Manual pumping:
- `renderer.pump_asset_tasks(max_steps)`

Use manual pumping if rendering is paused but you still want background loads to move.

## Unload APIs and Reserved Handles

Unload methods:
- `unload_mesh(mesh_handle)`
- `unload_material(material_handle)`
- `unload_texture(texture_handle)`

Reserved handles are protected and return `AssetError::ReservedHandle`.

Treat this as contract enforcement, not transient failure.

## Error Types You Will See Most

- `AssetError::Load { path, message }`
- `AssetError::Io { path, message }`
- `AssetError::Decode { path, message }`
- `AssetError::ManifestParse { path, message }`
- `AssetError::{InvalidHandle, StaleHandle, NotLoaded, OutOfBounds}`
- `AssetError::UnknownTicket`
- `AssetError::CancelRejected`

## Learn More

- Async example: `src/renderer/examples/demo_async_loading.rs`
- Handle model: `src/renderer/src/data/handles.rs`
- Scene merge flow: `03_scene_graph_and_fragment_workflows.md`
- Environment loading: `05_environment_and_skybox_runtime.md`
