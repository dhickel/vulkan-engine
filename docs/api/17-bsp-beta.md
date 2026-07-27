# BSP Beta — API Reference

> Feature-gated public API surface for the BSP Map Support beta. All BSP-specific types, functions, and feature gates are documented here.

## Audience

Rust developers reading runtime contracts for BSP map loading, rendering, and lifecycle management. Assumes familiarity with the [Renderer API](00-index.md) and the [App-Owned Loop](15-app-owned-loop.md) chapter.

## Feature Gate

All BSP API surface is behind the `renderer/bsp` Cargo feature:

```toml
[dependencies]
renderer = { path = "src/renderer", features = ["bsp"] }
bsp_runtime = { path = "src/bsp_runtime" }
bsp = { path = "src/bsp" }                        # for direct extraction
```

- `bsp` crate: pure parser, zero engine dependencies. Only depends on `glam`.
- `bsp_runtime` crate: integration coordinator. Depends on `bsp`, `renderer` (with `bsp` feature), `engine_events`.
- `renderer` crate with `bsp` feature: GPU upload, BSP materials, lightmap atlas, legacy and external-companion PBR BSP pipelines, PVS-aware submission.

Feature-gated builds:
```bash
cargo check -p renderer --features bsp
cargo check -p renderer --all-features   # bsp + csm + all others
cargo check                               # default: no bsp linked
```

## Crate Entrypoints

### `bsp` — Parser Crate

```rust
use bsp::{BspLoader, LoadOptions, BspWorld};

let data = std::fs::read("maps/mylevel.bsp")?;
let options = LoadOptions {
    strict: false,
    palette: Some(std::fs::read("palettes/my.lmp")?),
    lit_data: None,
    wad_archives: vec![],
    texture_overrides: vec![],
    source_identity: "maps/mylevel.bsp".into(),
};
let world: BspWorld = BspLoader::load(&data, &options)?;
for diagnostic in &world.diagnostics {
    eprintln!("{} {:?}: {}", diagnostic.code, diagnostic.severity, diagnostic.message);
}

// Query the world through the public free functions.
let leaf = bsp::camera_leaf_index(&glam::Vec3::ZERO, &world.nodes, &world.leaves, &world.planes);
let pvs = bsp::camera_pvs(&glam::Vec3::ZERO, &world.vis_data, &world.nodes, &world.leaves, &world.planes);
let contents = bsp::point_contents(glam::Vec3::ZERO, &world.nodes, &world.leaves, &world.planes);
```

### `bsp` — Extraction

```rust
use bsp::extract::{BspExtractionRequest, extract};

let request = BspExtractionRequest {
    world,
    scale: 0.0254,
    palette: Some(bsp::resources::decode_palette(&palette_bytes)),
    texture_companions: vec![
        bsp::TextureCompanion::new("textures/brick_norm.png", normal_png),
        bsp::TextureCompanion::new("textures/brick_gloss.png", gloss_png),
    ],
    ..Default::default()
};
let extracted = extract(request)?;
// extracted.geometries, extracted.materials, extracted.light_descriptors, etc.
```

`ExtractedVisibility::visleaf_count` is world model 0's authoritative PVS row width. It excludes reserved raw BSP leaf 0; PVS bit `i` corresponds to raw leaf `i + 1`. Renderer batching and light selection consume PVS-bit indices, not raw leaf-lump indices.

`TextureCompanion` carries caller-authorized encoded bytes. Extraction matches the basename conventions `<texture>_norm.png` and `<texture>_gloss.png` (exact case first, then ASCII case-insensitive) and stores matches on `ExtractedTexture::pbr_companions`. It performs no filesystem I/O. Renderer preflight decodes PNGs, requires base-texture dimensions, and treats either companion as PBR opt-in. No match preserves the legacy BSP material route.

### `bsp_runtime` — Coordinator

```rust
use bsp_runtime::{
    BspCoordinator, BspGenerationToken, AppBridge,
    BspCandidate, BspRuntimeError, BspPersistenceEnvelope,
};

let mut coordinator = BspCoordinator::new();

// Package and direct launch paths first create one complete authorized import.
// Every declared file is read through PackageResolver before parsing/extraction.
let import = bsp_runtime::package::authorize_package_import(
    &mut resolver,
    "maps/mylevel.bsp",
    "palettes/my.lmp",
    None,             // optional declared .lit
    &[],              // ordered declared WAD paths
    Some("textures"), // one confined companion root
    bsp_runtime::PackageImportMode::Strict,
    0.0254,
)?;
let prepare = coordinator.prepare_authorized_import(import)?;

// ... build `mount` with renderer.prepare_bsp_mount(coordinator.staged_extraction().unwrap()) ...
coordinator.set_renderer_mount_ready(prepare.token, mount)?;
coordinator.validate_for_scene(prepare.token, &mut scene)?;
coordinator.commit(prepare.token, &mut scene)?;

// Calling `prepare_authorized_import` with a replacement authorized import
// cancels the previous prepare and makes its token stale.

// Reimport a different package/direct source by authorizing a new import and
// running the normal prepare → upload → validate → commit transaction.

// Persist
let mutable_behavior = coordinator.capture_mutable_behavior();
let link: BspPersistenceEnvelope = coordinator.capture_source_link(mutable_behavior).unwrap();
scene.set_bsp_source_link(serde_json::to_value(&link)?);

// Unload
coordinator.unload(&mut scene)?;
```

### `renderer` — Mount and GPU Upload

```rust
use renderer::api::bsp::{
    PreparedBspMount, BspMountState,
    BspMaterialDesc, BspSurfaceClass, BspTextureSet,
    BspMeshUploadResult, BspRenderSubmissionData,
    build_bsp_material_descs, build_face_meshes,
};

// Prepare mount from extraction output
let mount: PreparedBspMount = renderer.prepare_bsp_mount(&extracted)?;

// Attach to scene
scene.set_bsp_mount(mount);

// Access BSP-specific render data for frame submission
scene.bsp_mount_state(); // → Option<&BspMountState>
```

## Renderer Pending-Mount Lease States

The `RendererLease` state machine governs when a BSP mount can be committed:

```
NotStarted → Pending → Ready
NotStarted → Ready          (synchronous upload)
Ready → Ready               (idempotent replace)
```

```rust
// Coordinator-side lease setup
coordinator.set_renderer_mount_ready(prepare.token, mount)?;
// coordinator.validate_for_scene(...) and coordinator.commit(...) can now proceed
```

## App Bridge Responsibilities

The `AppBridge` trait is implemented by the app to own physics and behavior state:

```rust
use bsp_runtime::bridge::{
    AppBridge, BehaviorEntityRecipe, BridgeToken, EntityCollisionRecipe,
    LightEntityRecipe, WorldCollisionRecipe,
};

impl AppBridge for MyAppBridge {
    fn name(&self) -> &str { "my-app-bridge" }

    fn prepare(
        &mut self,
        world_collider: &WorldCollisionRecipe,
        entity_colliders: &[EntityCollisionRecipe],
        lights: &[LightEntityRecipe],
        behaviors: &[BehaviorEntityRecipe],
    ) -> Result<BridgeToken, String> {
        // Create hidden app-owned resources and return an opaque token.
        Ok(BridgeToken::new(vec![1]))
    }

    fn validate(&self, token: &BridgeToken) -> Result<(), String> {
        // Confirm hidden resources are valid and capacity is reserved.
        Ok(())
    }

    fn commit(&mut self, token: BridgeToken) -> Result<(), String> {
        // Publish app-owned resources to simulation.
        Ok(())
    }

    fn rollback(&mut self, token: BridgeToken) {
        // Remove created resources (idempotent).
    }
}

coordinator.register_bridge("my-app-bridge", Box::new(my_bridge));
```

## Frame Snapshot Handoff

Each frame, the app captures a BSP simulation snapshot and synchronizes it to the scene:

```rust
use bsp_runtime::snapshot::{BspSimulationSnapshot, SnapshotBuilder};

let mut builder = SnapshotBuilder::new(generation, tick, dt, elapsed);
// Defaults are static-only: style 0 = 1.0 and styles 1..63 = 0.0.
// Set animated styles explicitly from app-owned simulation state.
for (style_id, intensity) in current_style_intensities.iter().copied().enumerate() {
    builder.set_light_style(style_id as u8, intensity);
}
builder.set_liquid_time(elapsed);
let snapshot: BspSimulationSnapshot = builder.build();

// Write frame-varying uniforms to the renderer.
scene.set_bsp_frame_values(snapshot.light_styles.intensities, snapshot.liquid_time);

// Apps that map BSP entities to scene nodes can additionally sync transforms;
// see `apps/bsp_beta/src/scene_sync.rs` for the workspace reference adapter.
```

## Source-Link Schema

The BSP persistence envelope stored in scene files:

```json
{
  "schema_version": 1,
  "bsp_source": {
    "asset_id": "maps/e1m1",
    "content_hash": "sha256:abcd1234...",
    "compiler_provenance": {
      "compiler": "ericw-tools",
      "version": "2.0.0-alpha3",
      "qbsp_hash": "...",
      "vis_hash": "...",
      "light_hash": "..."
    },
    "companion_hashes": {
      "palette": "sha256:...",
      "lit": null
    },
    "import_policy": {
      "scale": 0.0254,
      "texture_roots": [],
      "wad_roots": []
    },
    "entity_identity_records": {},
    "model_mapping_identity": {},
    "overrides": {
      "entity_overrides": [],
      "light_overrides": []
    },
    "mutable_behavior": {
      "doors": [],
      "buttons": [],
      "platforms": [],
      "triggers": [],
      "light_styles": {},
      "timers": [],
      "external_model_overrides": []
    }
  }
}
```

### Banned from Serialization

GPU handles (`VkImage`, `VkBuffer`, `VkDescriptorSet`), descriptor pool allocations, GPU cache slot indices, transient generation handles (`SceneNodeId`, `MeshHandle`, `MaterialHandle`), and expanded generated geometry must **never** appear in persisted data.

## Thread and Frame-Boundary Rules

- **Prepare** acquires no locks across I/O. BSP parsing and DTO extraction run on the calling thread.
- **Commit** holds `SceneWorld` lock briefly for publication-only operations. No parsing, I/O, or GPU allocation during commit.
- **GPU upload** runs through the renderer's transfer queue. The caller must pump transfer submissions (done automatically by the render loop).
- **Frame-varying uniforms** (style intensities, liquid time) are written once per frame after the frame-slot fence wait. In-flight descriptors are never mutated.

## Cancellation and Retirement

- Calling `prepare()` with new bytes increments the generation token, canceling any in-flight prepare.
- CPU-side staged allocations from a cancelled prepare are freed immediately.
- GPU-used payloads (meshes, materials, atlas textures) retire through the existing fence-observed `GpuRetirementQueue`.
- `commit()` for a stale generation returns `BspRuntimeError::StaleGeneration`.

## Diagnostics

All BSP diagnostics pass through `BspReport` with stable `DiagnosticCode` values:

| code | category | severity (dev) | severity (strict) |
|------|----------|---------------|-------------------|
| `BSP-UNSUPPORTED-DIALECT` | unsupported compatibility | error | error |
| `BSP-STRUCT-CORRUPT-LUMP` | structural corruption | error | error |
| `BSP-SECURITY-PATH-TRAVERSAL` | security | error | error |
| `BSP-MISSING-REQUIRED-PALETTE` | missing required | error | error |
| `BSP-FALLBACK-DEFAULT-PALETTE` | optional fallback | warning | warning |
| `BSP-FALLBACK-EMBEDDED-MIPTEX` | optional fallback | warning | warning |
| `BSP-ENTITY-UNKNOWN-CLASS` | unknown app entity | info | info |
| (see [bsp-compatibility](../../.internal-dev/specifications/bsp-compatibility.md) §7 for the full table) | | | |

## Example — Direct Runtime Workflow

```rust
use std::path::Path;

use bsp_runtime::{BspCoordinator, PackageImportMode};
use bsp_runtime::package::authorize_direct_import;
use renderer::prelude::*;

fn load_bsp_map(
    renderer: &mut Renderer,
    scene: &mut Scene,
    bsp_path: &Path,
    palette_path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    // The runtime boundary reads and authorizes every declared path once.
    let import = authorize_direct_import(
        bsp_path,
        palette_path,
        None,
        &[],
        None,
        PackageImportMode::Strict,
        0.0254,
    )?;

    let mut coordinator = BspCoordinator::new();
    let prepare = coordinator.prepare_authorized_import(import)?;
    let mount = renderer.prepare_bsp_mount(coordinator.staged_extraction().unwrap())?;
    coordinator.set_renderer_mount_ready(prepare.token, mount)?;
    coordinator.validate_for_scene(prepare.token, scene)?;
    coordinator.commit(prepare.token, scene)?;

    Ok(())
}
```

## Beta Status (2026-07-26)

The BSP beta gate is **NO-GO**. The following open issues block beta sign-off:

| issue | summary |
|-------|--------|
| [#57](https://github.com/dhickel/vulkan-engine/issues/57) | Static batch ceiling not enforced across frozen corpus |
| [#58](https://github.com/dhickel/vulkan-engine/issues/58) | Strict extraction fails on generated faces (missing lightmap) |
| [#59](https://github.com/dhickel/vulkan-engine/issues/59) | Renderer mount retirement missing fence-aware queue |
| [#60](https://github.com/dhickel/vulkan-engine/issues/60) | Committed bridge has no active teardown receipt |
| [#61](https://github.com/dhickel/vulkan-engine/issues/61) | GPU upload rollback crashes (SIGSEGV) |
| [#62](https://github.com/dhickel/vulkan-engine/issues/62) | Planned mesh bounds lost after GPU transfer |
| [#63](https://github.com/dhickel/vulkan-engine/issues/63) | First material slot (0,0) rejected as null sentinel |

All crate-level tests pass (`bsp`, `bsp_runtime`, `bsp_generator`, `bsp_beta` unit tests). Development-mode authorization can reach upload preflight. Visual acceptance, reference calibration, and live WSI evidence require a GPU/WSI environment and have not yet been produced.

## See Also

- [Guide: BSP Beta](../guide/18-bsp-beta.md) — app-builder how-to
- [Internal: BSP Runtime and Lifetime](../internal/18-bsp-runtime-and-lifetime.md) — ownership and protocol details
- [Dungeon Generation Spec](../../.internal-dev/specifications/bsp-dungeon-generation.md) — frozen M1/M2 bounds, construction parameters, generator authorization gate
- [BSP Acceptance Spec](../../.internal-dev/specifications/bsp-acceptance.md)
- [BSP Compatibility Spec](../../.internal-dev/specifications/bsp-compatibility.md)
- [BSP Transaction/Ownership Spec](../../.internal-dev/specifications/bsp-transaction-ownership.md)
- [Renderer BSP Feature — Cargo.toml](../../src/renderer/Cargo.toml)
