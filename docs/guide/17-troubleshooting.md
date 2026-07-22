# 17 — Troubleshooting

> Provenance: `G-17` — compiled from `src/renderer/src/api/errors.rs`, renderer lifecycle knowledge, dogfood/voxel error paths, and recorded knowledge files

This chapter covers common errors, diagnostic workflows, and resolution strategies. It is organized by error category: compile-time, initialization, runtime, terminal, subsystem-specific, and validation.

## Compile Errors

### Wrong Cargo target

**Symptom**: `cargo run --manifest-path examples/guide_app/Cargo.toml` works, but you see errors about missing crates.

**Diagnosis**: The guide app is a nested workspace, not a root member. Use `--manifest-path`:

```sh
# Correct
cargo run --manifest-path examples/guide_app/Cargo.toml

# Incorrect — guide_app is not in the root workspace
cargo run -p guide_app
```

> Provenance: `G-17-COMPILE-1` — from `docs/guide/03-building-your-first-app.md`

### Missing project flag

**Symptom**: `cargo run -p dungeon_dogfood` succeeds but the app cannot find assets.

**Diagnosis**: The working directory must be the repository root. The renderer locates shaders, assets, and startup scenes relative to `std::env::current_dir()`. Running from `apps/dungeon_dogfood/` fails.

```sh
# Correct — run from repository root
cd /path/to/vulkan-engine
cargo run -p dungeon_dogfood
```

> Provenance: `G-17-COMPILE-2` — from `AGENTS.md` working directory constraint

### libclang / russimp-sys incompatibility

**Symptom**: Build fails on `russimp-sys` or `clang-sys` with linker errors on systems with libclang ≥ 22.

**Diagnosis**: The `russimp-sys` crate (pulled in via assimp for glTF loading) has a version ceiling on `clang-sys`. This is a known limitation on bleeding-edge Linux distributions. No workaround is currently available — the build is blocked.

> Provenance: `G-17-COMPILE-3` — from `docs/api/14-dogfood-vertical-slice.md` known limitations

## Initialization Errors

### Vulkan: No capable device

**Symptom**: `Renderer::new()` returns `RendererError::Init(RendererInitError::Vulkan(...))` with a message about physical device selection.

**Diagnosis steps**:

1. Verify your GPU and driver:
   ```sh
   vulkaninfo --summary
   ```
   Look for a discrete or integrated GPU. If only `llvmpipe` appears, you do not have a usable Vulkan device.

2. Check your driver version:
   - **NVIDIA**: proprietary driver 535+ required
   - **AMD**: RADV (Mesa) or AMDGPU-PRO
   - **Intel**: ANV (Mesa) — integrated GPUs are not systematically validated

3. On Wayland, verify `XDG_RUNTIME_DIR` is set:
   ```sh
   echo $XDG_RUNTIME_DIR
   ```

4. On headless servers, install a software rasterizer or use a GPU with display offload:
   ```sh
   # Headless smoke (no window required)
   RUST_LOG=debug timeout --signal=INT 60s cargo run -p renderer --example api_test -- --headless
   ```

> Provenance: `G-17-INIT-1` — from `docs/guide/01-getting-started.md`

### Shader compilation failure

**Symptom**: `Renderer::new()` fails with `RendererInitError::ShaderCompile`.

**Diagnosis**: The renderer ships pre-compiled `.spv` files. If `compile_shaders` is `true` in `RendererConfig` and `glslangValidator` is missing from your `PATH`, shader compilation fails.

**Fix**: Set `compile_shaders: false` (the default). If you need runtime compilation, install the Vulkan SDK's `glslangValidator`.

> Provenance: `G-17-INIT-2` — from `RendererConfig` compile_shaders field

### Missing shader assets

**Symptom**: Asset load failures at startup mentioning `.spv` files.

**Diagnosis**: The working directory must be the repository root. Shaders are loaded from `src/renderer/src/shaders/spv/` relative to the current directory.

```sh
# Verify
ls src/renderer/src/shaders/spv/*.spv | head
```

> Provenance: `G-17-INIT-3` — from renderer asset path convention

### Async upload stall at startup

**Symptom**: The app starts but renders a blank or incomplete scene for the first few seconds.

**Diagnosis**: Async asset uploads (textures, meshes) run on background threads. The renderer's render calls perform a default bounded asset pump, and app-owned loops may also call `pump_asset_tasks()` explicitly to surface pump errors and drain a known amount of work before rendering. If deferred tickets remain `Pending`, the app may not be pumping often enough, or the source asset may still be queued/running/failing in the background loader.

**Fix**: In app-owned loops, call `renderer.pump_asset_tasks(32)?` before rendering and continue polling deferred load tickets until they leave `Pending`.

> Provenance: `G-17-INIT-4` — from `docs/guide/04-app-owned-loop.md` asset pumping section and `Renderer::render_scene_with_view`

## Runtime Errors

### Input queue without dispatch

**Symptom**: WASD keys do nothing; `InputSnapshot` reports no active actions.

**Diagnosis**: `begin_app_frame()` calls `input.dispatch_frame()` internally. If you are bypassing `begin_app_frame` and calling `input.queue_event()` directly, you must call `input.dispatch_frame()` yourself before reading the snapshot.

**Fix**: Use `begin_app_frame()` as your frame boundary — it handles dispatch automatically.

> Provenance: `G-17-RUNTIME-1` — from `docs/guide/06-input.md` dispatch boundary

### Cursor / UI capture prevents movement

**Symptom**: Mouse look stops working when ImGui panels are visible.

**Diagnosis**: The renderer's ImGui integration captures the mouse cursor while the debug UI is active. This is by design — UI interaction takes priority over camera look. The voxel demo's editor explicitly suppresses camera input while visible.

**Fix**: Close debug panels (F1 toggles the main overlay) or hide the voxel editor. The renderer does not currently support viewport click-through.

> Provenance: `G-17-RUNTIME-2` — from `docs/api/16-voxel-demo.md` editor behavior

### Pending async loads never complete

**Symptom**: `renderer.pump_asset_tasks(32)` returns `Ok(n)`, but the scene still shows placeholder or missing assets.

**Diagnosis**: `pump_asset_tasks` runs a bounded amount of queued/completed deferred work. If an asset remains pending, it may still be queued/running, the source file may be malformed or missing, or the ticket may have transitioned to a terminal failure that the app has not checked.

**Steps**:

1. Check `RUST_LOG=debug` for asset loading errors:
   ```sh
   RUST_LOG=debug cargo run -p renderer --example demo_model_load
   ```

2. Verify the asset file exists and is valid:
   ```sh
   ls src/renderer/src/assets/DamagedHelmet.glb
   ```

3. Check `LoadTicket` status programmatically with the asset-specific poller:
   ```rust
   match assets.poll_model_load(ticket) {
       LoadStatus::Pending { queued_at } => { /* still queued or running */ }
       LoadStatus::Uploaded { value: fragment } => { /* ready */ }
       LoadStatus::Failed { error } => return Err(error.into()),
       LoadStatus::Cancelled => { /* cancelled */ }
   }
   ```

> Provenance: `G-17-RUNTIME-3` — from `docs/api/04-assets-sync-deferred-and-handles.md` and `AssetError` variants

### Resize / acquire outcomes

**Symptom**: Logs show `SkippedResizePending`, `SkippedAcquireUnavailable`, or `PresentedSuboptimal` repeatedly.

**Diagnosis**: These are **normal transient WSI outcomes**, not errors:

| Outcome | Meaning | Action |
|---------|---------|--------|
| `SkippedResizePending` | Swapchain is mid-rebuild after a resize | The next frame will retry. If this persists, check that resize requests are arriving from the platform. |
| `SkippedAcquireUnavailable` | `vkAcquireNextImageKHR` returned `VK_NOT_READY` or `VK_TIMEOUT` | Continue. This is a transient WSI state, common during compositor transitions. |
| `SubmittedNotPresented` | Frame submitted but presentation was skipped | Normal in headless mode; unusual in windowed mode — may indicate surface loss. |
| `PresentedSuboptimal` | Swapchain no longer matches the surface (e.g., window was moved between monitors with different capabilities) | Continue; the renderer sets a resize flag internally and will rebuild the swapchain on the next opportunity. |

**When to act**: If `SkippedResizePending` persists for more than a second, the resize may have been lost. Force a resize:

```rust
renderer.resize(width, height)?;
```

> Provenance: `G-17-RUNTIME-4` — from `FrameRenderOutcome` variants in `docs/guide/04-app-owned-loop.md`

### Stale handle errors

**Symptom**: `SceneError::StaleNode`, `AssetError::StaleHandle`, or `AssetError::InvalidHandle`.

**Diagnosis**: Handles use slot+generation pairs. A stale handle means the slot was reused with a new generation between when you obtained the handle and when you used it. Common causes:

- Holding a `SceneNodeId` across a scene mutation that removes and recreates nodes.
- Holding a `MeshHandle` after calling `assets.unload_mesh()`.
- Using a handle from a previous `Scene` instance after loading a new scene.

**Fix**: Never cache handles across frames where scene mutations occur. Re-query handles each frame or store them only for the duration of the current operation.

> Provenance: `G-17-RUNTIME-5` — from `SceneError` and `AssetError` handle variants in `src/renderer/src/api/errors.rs`

### Player position becomes NaN

**Symptom**: Dogfood logs `Rejected non-finite camera intent` or the camera freezes.

**Diagnosis**: Physics collision resolution can produce NaN positions if the player is wedged in degenerate geometry (zero-thickness walls, overlapping colliders).

**Fix**: The `CameraIntentGuard` system rejects non-finite positions and keeps the previous valid position. If this repeats, check your level data for degenerate tiles (walls with zero thickness, ramps with NaN normals).

> Provenance: `G-17-RUNTIME-6` — from `player.rs` `CameraIntentGuard::RejectedNonFinite`

## Terminal Errors

### DeviceLost

**Symptom**: `RendererError::DeviceLost` — `"Vulkan device lost"`.

**Causes**:
- GPU hang (long-running shader, invalid descriptor access)
- Driver crash or timeout (Linux: GPU reset by kernel)
- Physical GPU removal (eGPU disconnect, VM migration)
- System suspend/resume on some driver/GPU combinations

**Required action**: The `DeviceLost` error is **terminal**. You must:
1. Exit the event loop.
2. Destroy the `Renderer`.
3. Recreate `Renderer::new()` from scratch.

Do not attempt to continue rendering after `DeviceLost`. The Vulkan logical device is in an undefined state.

> Provenance: `G-17-TERMINAL-1` — from `RendererError::DeviceLost` in `src/renderer/src/api/errors.rs`

### BackendPoisoned

**Symptom**: `RendererError::BackendPoisoned(msg)` — `"renderer backend poisoned: ..."`.

**Causes**: A previous renderer operation panicked (unwound through FFI) or a prior terminal error was not handled, and a subsequent operation was attempted.

**Required action**: Same as `DeviceLost` — destroy and recreate the `Renderer`. A poisoned backend is irrecoverable.

**Prevention**: Always match `DeviceLost` and `BackendPoisoned` explicitly in your error handling. Never use a catch-all `_ =>` arm that discards terminal errors and continues the loop.

```rust
match renderer.render_scene_with_view(&mut scene, view) {
    Ok(outcome) => { /* handle transient outcomes */ }
    Err(RendererError::DeviceLost) => {
        eprintln!("Vulkan device lost; exiting");
        elwt.exit();
        return;
    }
    Err(RendererError::BackendPoisoned(msg)) => {
        eprintln!("renderer backend poisoned: {msg}");
        elwt.exit();
        return;
    }
    Err(e) => {
        eprintln!("render failed: {e}");
        elwt.exit();
        return;
    }
}
```

> Provenance: `G-17-TERMINAL-2` — from `RendererError::BackendPoisoned` and `docs/guide/04-app-owned-loop.md`

## Subsystem Errors

### Alpha audio device failure

**Symptom**: Audio smoke test logs `DeviceSmokeStatus::Failed` or no audio plays.

**Diagnosis**: Audio is alpha. Device creation may fail on systems without PulseAudio, PipeWire, or ALSA. The dogfood app treats audio failures as non-fatal.

```sh
# Enable device-backed smoke test (dogfood)
DUNGEON_DOGFOOD_AUDIO_SMOKE=1 cargo run -p dungeon_dogfood
```

**Expected behavior**: The app starts, logs a clip probe report, and continues regardless of audio outcome.

> Provenance: `G-17-SUBSYS-1` — from `audio_bridge.rs` and `docs/guide/11-audio.md`

### Physics step failures

**Symptom**: `"Physics step failed: ..."` in dogfood logs.

**Diagnosis**: The `physics` crate is alpha. Step failures can occur with degenerate collider shapes (zero-area triangles, NaN vertices) or when the solver fails to converge.

**Fix**: Check the collider mesh for validity. The `--validate-colliders` flag in dogfood headless mode checks recipes for shape validity before instantiation.

> Provenance: `G-17-SUBSYS-2` — from `physics` crate API and dogfood `--validate-colliders`

## Validation and Evidence

### Headless capture for visual evidence

When debugging visual issues (incorrect materials, missing geometry, lighting artifacts), use deterministic headless captures rather than describing what you see on screen:

```sh
# Single frame capture
RUST_LOG=info timeout --signal=INT 60s cargo run -p renderer --example api_test -- \
  --headless --capture_target draw --capture_frame=5

# Sequence capture
RUST_LOG=info timeout --signal=INT 60s cargo run -p renderer --example api_test -- \
  --headless --capture_target draw \
  --capture_frames=3 --capture_frame_start=5 --capture_frame_interval=5 \
  --env src/renderer/src/assets/sky_maps/indoor_4k.exr
```

Captures land under `.internal-dev/captures/` with PNG frames and JSON sidecars.

> **Constraint**: Headless captures are **deterministic rendering evidence**. They do **not** exercise swapchain, presentation, or WSI paths. Window-specific issues (tearing, VSync timing, resize artifacts) cannot be reproduced headless.

For the full capture validation workflow (scene setup, camera positioning, pass/fail criteria), use the project skill:

```
.skill:engine-headless-capture-validation
```

> Provenance: `G-17-VAL-1` — from `.internal-dev/skills/engine-headless-capture-validation/SKILL.md`

### WSI limitations

**Symptom**: The app renders correctly in headless mode but has visual artifacts in windowed mode.

**Diagnosis**: WSI (Window System Integration) behavior depends on your compositor, GPU driver, and display configuration. Known limitations:

- **Wayland**: Some compositors report resize events after `RedrawRequested`, causing one-frame size mismatches. The guide checkpoint handles this by re-checking `window.inner_size()` at the top of the redraw handler.
- **VSync timing**: Presentation timing is platform-dependent. Frame pacing issues (stutter, uneven frame times) are WSI artifacts, not renderer bugs.
- **Suboptimal presentation**: Moving a window between monitors with different capabilities (refresh rate, color depth) produces `PresentedSuboptimal` — the renderer handles this internally.
- **Minimized windows**: A minimized window may stop receiving `RedrawRequested` events. The checkpoint's `AboutToWait` handler with `window.request_redraw()` ensures the loop continues when the window is restored.

**Evidence distinction**: Always distinguish headless rendering evidence from WSI observation:

| Evidence type | Proves | Does not prove |
|---------------|--------|----------------|
| Headless capture PNG | Renderer output correctness for a given scene and camera | Swapchain/presentation behavior, frame timing, VSync |
| WSI observation (screenshot, visual report) | End-to-end frame delivery | Deterministic reproducibility |
| Compile check (`cargo check`) | API shape correctness | Any visual behavior |

> Provenance: `G-17-VAL-2` — from headless capture skill and WSI knowledge

### Debug timing capture

When investigating frame-time issues (GPU bubbles, asset-load stalls, frame drops):

```sh
# Record 10 seconds at 50ms intervals
RUST_LOG=debug timeout --signal=INT 60s cargo run -p renderer --example demo_pbr -- \
  --record_debug=10 --record_debug_interval=50 \
  --record_debug_path=.internal-dev/debug_reports/demo_pbr-timing.jsonl
```

Each JSONL line contains per-pass timing data. Analyze with `jq` or any JSONL-capable tool.

> Provenance: `G-17-VAL-3` — from `AGENTS.md` debug-record smoke pattern

## Quick Diagnostic Commands

```sh
# ── Compile checks ──
cargo check                                          # entire workspace
cargo check -p renderer --examples                   # renderer + examples
cargo check -p dungeon_dogfood                       # dogfood app
cargo check -p voxel_demo                            # voxel demo

# ── Tests ──
cargo test -p engine                                 # root engine tests
cargo test -p renderer                               # renderer tests
cargo test -p dungeon_dogfood                        # dogfood tests
cargo test -p voxel_demo                             # voxel demo tests

# ── Renderer smokes (windowed) ──
cargo run -p renderer --example demo_pbr
cargo run -p renderer --example demo_unlit
cargo run -p renderer --example demo_model_load
cargo run -p renderer --example demo_async_loading
cargo run -p renderer --example api_test

# ── Renderer smokes (headless) ──
RUST_LOG=debug timeout --signal=INT 60s cargo run -p renderer --example api_test -- --headless

# ── Validation layers ──
# Set validation_layer: true in RendererConfig, or:
VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation cargo run -p renderer --example demo_pbr

# ── Vulkan info ──
vulkaninfo --summary
```

> Provenance: `G-17-DIAG` — compiled from `AGENTS.md` runtime validation commands

## Next

Return to the [Guide Index](00-index.md) for the full chapter list, or see the [API Reference](../api/00-index.md) for function signatures and error variants.
