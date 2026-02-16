# Assets: Sync, Deferred, and Handle Semantics

## 1. Purpose & Audience
This chapter documents facade-level asset loading for `AssetManager`, including synchronous loads, deferred loads with `LoadTicket`, and practical handle lifecycle behavior.

## 2. Where This Fits in Engine Flow
Asset flow at API level:
`renderer.assets()` -> sync load (`load_*`) or deferred request (`request_*_load`) -> polling (`poll_*_load`) -> scene integration -> frame rendering.

## 3. Key Concepts
- `AssetManager` is borrowed from `Renderer` (`renderer.assets()`) and scoped to mutable renderer access.
- Sync loading APIs block until upload/prepare is complete:
  - `load_model`, `load_mesh`, `load_texture`, `load_environment`
- Deferred loading APIs return `LoadTicket` immediately:
  - `request_model_load`, `request_texture_load`, `request_texture_load_with_options`
- Deferred completion is observed through `LoadStatus<T>`:
  - `Pending { queued_at }`
  - `Uploaded { value }`
  - `Failed { error }`
  - `Cancelled`
- Deferred progress requires pumping (`Renderer::pump_asset_tasks`) and/or regular render loop activity.
- Handle types (`MeshHandle`, `TextureHandle`, `EnvironmentHandle`) use slot+generation; stale handles are expected after unload/reuse.

## 4. Code Walkthrough
Snippet Type: Real
```rust
// Sync model load and mount
let mut assets = renderer.assets();
let fragment = assets.load_model("src/renderer/src/assets/DamagedHelmet.glb")?;
let mount = scene.merge_fragment(None, fragment)?;
let root = mount.mounted_root;
```

Snippet Type: Real
```rust
// Deferred model load ticket polling
let ticket = renderer
    .assets()
    .request_model_load("src/renderer/src/assets/BoomBox.glb")?;

loop {
    let status = renderer.assets().poll_model_load(ticket);
    match status {
        renderer::LoadStatus::Pending { .. } => {
            let _ = renderer.pump_asset_tasks(32)?;
        }
        renderer::LoadStatus::Uploaded { value: fragment } => {
            scene.merge_fragment(None, fragment)?;
            break;
        }
        renderer::LoadStatus::Failed { error } => return Err(error.into()),
        renderer::LoadStatus::Cancelled => break,
    }
}
```

Snippet Type: Real
```rust
// Environment load + scene skybox
let env = renderer.assets().load_environment(EnvironmentSource::Auto(
    "src/renderer/src/assets/sky_maps/indoor_4k.exr".into(),
))?;
scene.set_skybox(env);
```

Snippet Type: Pseudocode
```text
Use two loops/concepts:
  render loop: draws every frame
  load service: polls tickets and mounts ready fragments
The load service can run each tick but should remain logically separate from draw submission code.
```

## 5. Best Practices
- Start with sync loads for first integrations, then adopt deferred tickets for larger content.
- Handle `LoadStatus` exhaustively and log terminal failures.
- Keep loading orchestration and rendering orchestration decoupled for simpler debugging.
- Document current alpha constraints in your app docs (ticket retention limits, cancellation behavior, pending requirements).

## 6. Gotchas & Failure Modes
- Forgetting to pump deferred work can leave tickets stuck in `Pending`.
- Polling a wrong/expired ticket returns `UnknownTicket` in `LoadStatus::Failed`.
- `cancel_load` rejects tickets that are already running or completed.
- Unloading reserved/default resources can fail (`ReservedHandle`).
- Using stale handles after resource lifecycle changes produces stale/invalid errors.

## 7. Debugging Playbook
- Step 1: print ticket IDs and status transitions with timestamps.
- Step 2: confirm that `pump_asset_tasks` is called regularly during deferred workflows.
- Step 3: distinguish `Load`, `Decode`, `Io`, `Sync`, and handle errors in logs.
- Step 4: if content is loaded but not visible, verify fragment merge and scene skybox assignment.
- Step 5: if environment appears inactive, inspect `renderer.environment_runtime_status()` and `assets.environment_state(env)`.

## 8. Cross-Module Links
- Asset facade implementation: `src/renderer/src/api/assets.rs`
- Loading types (`LoadTicket`, `LoadStatus`): `src/renderer/src/api/loading.rs`
- Renderer pump integration: `src/renderer/src/api/renderer.rs`
- Asset internals and cache transitions: `docs/internal/03-asset-lifecycle-and-io.md`

## 9. Standard References
- Rust `Result` and error handling: https://doc.rust-lang.org/book/ch09-00-error-handling.html
- Vulkan Guide memory allocation: https://github.khronos.org/Vulkan-Site/guide/latest/memory_allocation.html
- Vulkan Guide synchronization: https://github.khronos.org/Vulkan-Site/guide/latest/synchronization.html
- Engine baseline reference: https://github.com/SaschaWillems/Vulkan-glTF-PBR

## 10. See Also
- `docs/api/02-renderer-lifecycle-and-frame-api.md`
- `docs/api/03-scene-graph-and-fragment-workflows.md`
- `docs/internal/03-asset-lifecycle-and-io.md`
