# BSP Map Support — Beta

> App-builder guide for loading, rendering, and interacting with Quake 1 BSP maps in the engine.

## Audience

Rust developers building applications that include compiled Quake-format BSP maps as scene content. You should be comfortable with the [App-Owned Loop](04-app-owned-loop.md) and [Asset Pipeline](09-asset-pipeline.md) before reading this chapter. This chapter describes a **beta** capability: the API is stable within the sprint but broader BSP features remain in development.

## Prerequisites

- **Feature flag**: The `bsp` Cargo feature must be enabled on the `renderer` crate. It is excluded from `default`.
  ```toml
  [dependencies]
  renderer = { path = "src/renderer", features = ["bsp"] }
  ```
- **Compiler toolchain**: The engine does NOT bundle a BSP compiler. You must supply `ericw-tools 2.0.0-alpha3` (or a pinned compatible version) on your build host. The approved executables are `qbsp`, `vis`, and `light`.
- **Fixture rights**: All `.map` sources, palettes, and compiled `.bsp` files must be your own work or licensed for redistribution (CC0 or project license). The engine includes **zero** copyrighted id Software content.
- **Dungeon generator**: A procedural dungeon generator is implemented in `src/bsp_generator/` and targets the frozen M1/M2 bounds. See the [BSP Generator Guide](19-bsp-generator.md) for usage and the [dungeon generation specification](../../.internal-dev/specifications/bsp-dungeon-generation.md) for the frozen contract. **Beta gate status: NO-GO** — the exact generated dungeon now has a strict visual fallback mount, but compiler/release evidence and lifecycle work remain blocked by open GitHub issues #57–#62.

## Package Layout

A BSP-enabled package contains:

```
my_package/
  maps/
    my_level.map          ← TrenchBroom source
    my_level.bsp          ← compiled BSP (output of qbsp + vis + light)
    my_level.lit          ← optional colored light companion
  palettes/
    my_palette.lmp        ← 768-byte raw RGB palette
  wads/
    my_textures.wad       ← optional WAD2 texture archive
  textures/
    brick_norm.png         ← optional tangent-space normal companion
    brick_gloss.png        ← optional gloss companion
    replacement/*.png      ← optional loose replacement textures
  manifest.toml           ← package manifest referencing the BSP asset
```

### Package Manifest

```toml
# manifest.toml
format_version = 1
package_id = "com.example.my_package"
[package]
name = "My Package"
version = "1.0.0"

[[assets]]
id = "maps/my_level"
kind = "bsp"
path = "maps/my_level.bsp"
[assets.companions]
palette = "palettes/my_palette.lmp"
lit = "maps/my_level.lit"
```

## Loading a BSP Map

### Quick Start — App-Owned Loop with BSP

```rust
use bsp_runtime::{BspCoordinator, PackageImportMode};
use bsp_runtime::package::authorize_package_import;
use package_io::resolver::PackageResolver;
use renderer::prelude::*;

fn load_map(
    renderer: &mut Renderer,
    scene: &mut Scene,
    resolver: &mut PackageResolver,
) -> Result<(), Box<dyn std::error::Error>> {
    // 1. Authorize every declared resource and parse exactly once.
    let import = authorize_package_import(
        resolver,
        "maps/my_level.bsp",
        "palettes/my_palette.lmp",
        None,
        &[],
        Some("textures"),
        PackageImportMode::Strict,
        0.0254,
    )?;

    // 2. Register app bridges before preparation so their hidden resources
    // are included in the candidate.
    let mut coordinator = BspCoordinator::new();

    // 3. Extract from the one authorized import, upload, validate, publish.
    let prepare = coordinator.prepare_authorized_import(import)?;
    let mount = renderer.prepare_bsp_mount(coordinator.staged_extraction().unwrap())?;
    coordinator.set_renderer_mount_ready(prepare.token, mount)?;
    coordinator.validate_for_scene(prepare.token, scene)?;
    coordinator.commit(prepare.token, scene)?;

    Ok(())
}
```

### Maintained `bsp_beta` Entrypoint and Authorized Inputs

The workspace app requires an explicit import policy and explicit declared
resources. It does not probe the BSP parent, game root, or any fallback
location for companions:

```bash
cargo run -p bsp_beta -- \
  --strict \
  --bsp /path/to/game/maps/start.bsp \
  --palette /path/to/game/gfx/palette.lmp \
  --wad /path/to/game/id1/pak0-textures.wad \
  --lit /path/to/game/maps/start.lit \
  --textures /path/to/game
```

`--strict` and `--development` are mutually exclusive; one is required for a
BSP launch. `--palette` is required, while `--wad` and `--lit` are optional
only when omitted (a declared missing resource is an authorization error).
`--textures` is the sole PBR companion root: a path ending exactly in
`textures` is used directly; every other path is confined to its `textures/`
child. No `--textures` argument means no PBR discovery. The runtime authorizes
all accepted bytes through one `PackageResolver` boundary before parse or
extraction.

### EnhancedV3 live explorer

`--m3-generate` builds an EnhancedV3 full-config package through `engine_pack`,
then strictly authorizes its BSP, explicit `.lit`, WAD, palette, and texture
closure before using the same runtime path as direct imports. It always uses
strict mode and rejects direct BSP, palette, LIT, WAD, and texture companion
flags. Tool discovery
uses `--ericw-tools`, `ERICW_TOOLS_DIR`, the documented HOME installation, then
one PATH directory containing all of `qbsp`, `vis`, and `light`.

```bash
cargo run -p bsp_beta -- --m3-generate --headless
cargo run -p bsp_beta -- --m3-generate --ericw-tools /path/to/ericw/bin
```

#### In-Game GUI (F1 / F2)

In a live window (`--m3-generate` without `--headless` or `--mcp`), press
**F1** to open the in-game EnhancedV3 GUI in Keyboard mode or **F2** to open
it in Mouse mode. Press the same key again to close the menu. The two modes
are mutually exclusive — pressing the other hotkey while a menu is open
switches modes without ever allowing both simultaneously. F1 and F2 are
globally consumed by the GUI and never forwarded to built-in renderer panels
or gameplay.

**Keyboard mode** (F1):
- Arrow keys / Tab navigate fields; Enter / Space activate; digits begin
  numeric editing; Backspace deletes the current digit (and leaves an empty edit); Escape closes the menu.
- All mouse input (buttons, wheel, cursor motion) is discarded. Device-level
  mouse motion and wheel are consumed.
- No keyboard or mouse events reach gameplay.

**Mouse mode** (F2):
- Left-click fields, +/- steppers, checkboxes, dropdowns, and action buttons.
  Mouse wheel scrolls the panel when content overflows.
- Other keyboard input is discarded except Escape (which closes the menu); the global F1/F2 controls still switch or close menu modes.
- Device-level mouse motion and wheel are consumed.

**Input guard**: On any transition into a menu, synthetic release events are
queued for all gameplay keyboard bindings (W/S/A/D/Space/ShiftLeft) and
common mouse buttons. The FPS camera controller is gated while a menu is
open. On close, the gameplay input gate is restored without synthesizing
press events, sealing held-key and accumulated-look leakage.

**Cursor coordinates**: Mouse hitboxes use the latest `WindowEvent::CursorMoved`
position, converted from physical to logical coordinates by the window scale
factor (including factors below 1). They never use the window's screen position;
viewport state is refreshed on resize and scale-factor changes. Enter/leave clears
the cached hit-test position until a new move arrives; those events still update
renderer cursor policy but are never queued to gameplay.

#### Menu Actions

| action | behavior |
|--------|----------|
| **Generate** | Snapshots the current full `GenConfig` and enqueues an asynchronous EnhancedV3 generation through `engine_pack`. The menu stays open. If a prior apply-close intent was pending, it is cancelled (latest action owns close intent). |
| **Apply & Close** | Same as Generate, but records the enqueued request ID as a close intent. The menu closes only after that specific request completes successfully and the live BSP mount is atomically replaced. On failure the menu stays open with an actionable status. |
| **Escape** | Closes the menu immediately without generating. |

#### Regeneration Lifecycle

- Generation runs off the render thread via `GenWorker`. Every request owns
  a distinct package directory. Queued requests coalesce to latest-wins.
- On successful commit: the old BSP mount is detached for fence-aware
  retirement, the new mount is published atomically, the GUI flashes a
  visible "Generated" indicator for two seconds, and the window title shows
  the active config. The old world remains active on any failure.
- After Apply & Close succeeds: the menu closes, input returns to gameplay
  immediately, and a 2-second "Generated!" title-bar toast appears before
  restoring the normal title.

#### Explorer Hotkeys (No Menu Open)

When no menu is open, the following hotkeys mutate the shared GUI config
and immediately enqueue regeneration:

- **F5**: increment seed
- **F6**: cycle preset (Sparse → Moderate → Rich) and extent
- **F7**: toggle chamfer
- **F8**: cycle arch type (Pointed → Segmented → None)
- **F9**: toggle stairs
- **Ctrl+R**: regenerate with unchanged config

These hotkeys are discarded while either menu is open; they mutate the same
shared config, never a divergent second config.

For deterministic renderer-owned captures:

```bash
cargo run -p bsp_beta -- \
  --strict --headless --capture-frames 5 \
  --bsp /path/to/game/maps/start.bsp \
  --palette /path/to/game/gfx/palette.lmp \
  --wad /path/to/game/id1/pak0-textures.wad \
  --textures /path/to/game
```

### MCP Headless Server

`--mcp` mounts the BSP in a 1920×1080 headless renderer and then serves newline-delimited MCP JSON-RPC 2.0 over stdin/stdout. It implies headless mode and never creates a window or WSI surface. The initial camera uses the authored `info_player_start` origin rather than the BSP bounds center, which may be exterior void. Stdout contains only JSON-RPC responses; engine diagnostics remain on stderr.

```bash
cargo run -p bsp_beta -- \
  --strict --mcp \
  --bsp /path/to/game/maps/start.bsp \
  --palette /path/to/game/gfx/palette.lmp \
  --wad /path/to/game/id1/pak0-textures.wad \
  --textures /path/to/game
```

The server supports the MCP `initialize`, `tools/list`, and `tools/call` methods and exposes:

- `set_camera`: engine-space `x`, `y`, `z` plus `yaw` and `pitch` in radians.
- `capture`: optional PNG `path`; returns the completed path and 1920×1080 dimensions.
- `get_info`: face, batch, material, texture, PBR-texture, BSP-byte-size, and camera-position data.
- `point_contents`: engine-space `x`, `y`, `z`; returns `solid` and the raw BSP leaf index.

Each request and response is one JSON object followed by a newline. Close stdin to stop the server.

### External PBR Texture Companions

Companions are named from the BSP texture identity, not the BSP filename:

- `<texture>_norm.png`: tangent-space normal map. Red and green encode X/Y; Z is reconstructed positive in the shader.
- `<texture>_gloss.png`: gloss in the red channel. The shader uses `roughness = 1 - gloss` (clamped to a minimum roughness of 0.04).

Either file is sufficient to opt an opaque or alpha-mask surface into the BSP PBR pipeline. A missing normal map defaults to a flat normal; a missing gloss map defaults to fully rough. Companion dimensions must exactly match the resolved base texture, and malformed or mismatched PNGs reject the renderer upload rather than silently changing material behavior.

PBR surfaces remain BSP lightmapped surfaces: normalized baked-light bytes modulate diffuse albedo with classic 2× overbright, while the prefiltered scene environment and BRDF LUT provide dielectric specular. The bundled CC0 Stone Beta companions keep stone gloss at 0.30–0.50 (roughness 0.50–0.70), and the renderer defaults to the project-authored CC0 Dungeon HDR cubemap rather than a flat LDR color source. Palette fullbright pixels remain additive, hue-preserving emission. BSP albedo/material images use complete mip chains for stable minification. Sky and liquid surfaces keep their dedicated legacy shaders. If neither companion exists, the original packed fullbright upload, pipeline selection, and `bsp_lightmapped.frag` path are unchanged.

Package loading derives these files through `PackageResolver`, records both
present and absent PBR closure entries, and carries their verified identities
into the BSP cache identity:

```rust
let import = bsp_runtime::package::authorize_package_import(
    &mut resolver,
    "maps/my_level.bsp",
    "palettes/my_palette.lmp",
    None,
    &[],
    Some("textures"),
    bsp_runtime::PackageImportMode::Strict,
    0.0254,
)?;
let prepare = coordinator.prepare_authorized_import(import)?;
```

The raw-world and raw-byte coordinator helpers are development/test
compatibility paths; package and direct production launch must use an
`AuthorizedBspImport`.

### Crate Dependencies

The root `engine` crate does not currently re-export BSP APIs. BSP applications should depend directly on the support crates they use:

```toml
[dependencies]
bsp = { path = "src/bsp" }
bsp_runtime = { path = "src/bsp_runtime" }
renderer = { path = "src/renderer", features = ["bsp"] }
```

## Compiling and Validating Maps

### Compiler Invocation

Use the `engine_pack compile-bsp` command:

```bash
engine_pack compile-bsp maps/my_level.map \
  --profile profiles/ericw-tools.toml \
  --out maps \
  --palette palettes/my_palette.lmp
```

The CLI enforces:
- Shell-free subprocess invocation (no shell interpolation)
- Compiler identity verification (SHA-256 of executables)
- Warning-free compiler stages: any `qbsp`, `vis`, or `light` warning, including missing textures or skipped fill, returns `CompilerWarning` and blocks publication
- Output size budget (default 128 MiB)
- Post-compile re-validation through the `bsp` parser in strict mode
- Timeout per stage (default 120 seconds)

### Manual Compilation

If you compile maps outside `engine_pack`, verify the compiler identity:

```bash
qbsp --version   # must report "ericw-tools 2.0.0-alpha3"
vis --version    # must report "ericw-tools 2.0.0-alpha3"
light --version  # must report "ericw-tools 2.0.0-alpha3"
```

Compile with:

```bash
qbsp my_level.map              # produces my_level.bsp
vis my_level.bsp               # computes visibility
light my_level.bsp             # bakes lightmaps
```

For BSP2 output with colored lights:

```bash
qbsp -bsp2 my_level.map
vis my_level.bsp
light -bsp2 -lit -colored my_level.bsp
```

## Development vs Strict Policy

The BSP coordinator supports two diagnostic severity modes:

| mode | behavior |
|------|----------|
| **Development** (`--development`) | Explicit diagnostic policy for development/test imports. Unsupported compatibility features are diagnosed but may remain usable. |
| **Strict / Release** (`--strict`) | Explicit release policy. Required-resource, compatibility, and structural failures reject the import before candidate/GPU work, except for the temporary generated-dungeon visual fallbacks below. |

Set strict mode when validating release-ready package/direct content. For generated dungeons, a structurally valid but unresolved texture slot receives `BSP-FALLBACK-DIAGNOSTIC-TEXTURE`, and a visible face with no lightmap offset receives `BSP-FALLBACK-MISSING-LIGHTMAP` and renders unlit. These owner-authorized warnings keep the dungeon drawable; they are not release or compiler-publication evidence.



```rust
let import = bsp_runtime::package::authorize_direct_import(
    bsp_path,
    palette_path,
    lit_path,
    wad_paths,
    textures_root,
    bsp_runtime::PackageImportMode::Strict,
    0.0254,
)?;
let prepare = coordinator.prepare_authorized_import(import)?;
```

## App-Owned Loop Responsibilities

The app owns the full BSP lifecycle within its event loop:

1. **Load**: Parse BSP bytes and prepare the coordinator.
2. **Query**: Use public helpers such as `bsp::point_contents()`, `bsp::camera_leaf_index()`, and `bsp::camera_pvs()` before commit when app logic needs BSP spatial data.
3. **Bridge**: Implement `AppBridge` for physics colliders and behavior state machines.
4. **Simulate**: Run app-owned physics stepping and behavior updates each frame.
5. **Snapshot**: Build an app-owned `BspSimulationSnapshot` with `SnapshotBuilder` or an app adapter such as `apps/bsp_beta/src/snapshot.rs`.
6. **Sync to Scene**: Call `Scene::set_bsp_frame_values()` and update app-owned entity scene nodes from the snapshot.
7. **Render**: Submit the scene normally — BSP geometry renders automatically when a mount is attached.
8. **Reload**: Call `coordinator.reload()` to atomically swap a new map version.
9. **Persist**: Call `coordinator.capture_mutable_behavior()`, then `coordinator.capture_source_link(mutable_behavior)` and `Scene::set_bsp_source_link()` to save state.
10. **Unload**: Call `coordinator.unload(&mut scene)` to clean up all BSP resources.

## Model Mappings

Inline brush models (doors, platforms, buttons) and external models referenced by entities are placed as scene nodes by the coordinator. The `scene_sync` module maps snapshot entity transforms to scene-node world transforms each frame.

```rust
// In your app's frame loop (the workspace `apps/bsp_beta` crate contains
// one reference `scene_sync` implementation):
let snapshot = snapshot_producer.capture(dt);
scene.set_bsp_frame_values(snapshot.light_styles.intensities, snapshot.liquid_time);
scene_sync::sync_snapshot_to_scene(&snapshot, &entity_node_map, &mut scene);
// Snapshot defaults are deterministic: style 0 = 1.0; styles 1..63 = 0.0
// until the app explicitly activates an animated light style.
```

## Deterministic Captures vs Live WSI

- **Headless captures** (`--headless --capture_target draw`) are valid for pixel-correctness validation of BSP lighting and geometry. Use frozen settings: exposure 1.0, static style 0, animation time 0.0.
- **Live WSI** (real GPU + windowing system) is required to validate swapchain lifecycle, resize, minimize/restore, and surface-loss recovery. Headless captures cannot substitute for WSI lifecycle checks.
- If no live GPU/WSI is available, record the gap explicitly — do not substitute a headless claim.

## Diagnostics

All BSP diagnostics carry stable machine-readable codes (e.g., `BSP-STRUCT-CORRUPT-LUMP`, `BSP-MISSING-REQUIRED-PALETTE`). Diagnostic severity depends on the policy mode. Inspect diagnostics via:

```rust
match BspLoader::load(&data, &options) {
    Ok(world) => {
        for diag in &world.diagnostics {
            println!("{} {:?} {}", diag.code, diag.severity, diag.message);
        }
    }
    Err(report) => {
        println!("{} {:?} {}", report.code, report.severity, report.message);
    }
}
```

## Supported Profiles

| profile | magic | status |
|---------|-------|--------|
| `q1-portable-ericw` BSP29 | `29` (LE i32) | **beta** — supported |
| `q1-portable-ericw` BSP2 | `BSP2` (4-byte ASCII) | **beta** — supported |

## Excluded Formats

The following BSP formats produce `BSP-UNSUPPORTED-DIALECT` errors and will never load:

| dialect | magic | reason |
|---------|-------|--------|
| Half-Life BSP30 | version 30 | distinct product |
| Quake 2 BSP38 | `"2PSB"` | distinct product |
| Quake 3 / IBSP BSP46 | `"IBSP"` | distinct product |
| Valve/Source VBSP | varied | distinct product |

## Limitations (Beta) and Known Issues

### Blocked Paths (Open GitHub Issues)

These issues block the beta gate and must be resolved before sign-off:

| issue | summary | effect |
|-------|---------|--------|
| [#57](https://github.com/dhickel/vulkan-engine/issues/57) | Static batch ceiling not enforced across frozen corpus | Batch/draw budget compliance unproven |
| [#58](https://github.com/dhickel/vulkan-engine/issues/58) | Generated faces can lack baked lightmap data | Runtime uses the unlit fallback; compiler/release evidence remains blocked |
| [#59](https://github.com/dhickel/vulkan-engine/issues/59) | Renderer mount retirement missing fence-aware queue | Stale handle invalidation incomplete |
| [#60](https://github.com/dhickel/vulkan-engine/issues/60) | Committed bridge has no active teardown receipt | Generic bridge unload/replacement not atomic |
| [#61](https://github.com/dhickel/vulkan-engine/issues/61) | GPU upload rollback crashes with SIGSEGV (descriptor pool double-free) | Development GPU mount blocked |
| [#62](https://github.com/dhickel/vulkan-engine/issues/62) | Planned mesh bounds lost after GPU transfer | Canonical batch record construction fails |

**Current generated-dungeon status**: The exact baseline mounts in strict headless mode with 411 renderable faces, 6 batches, and three PBR textures. Missing WAD/lightmap data uses the documented visual fallbacks. This proves the draw path only; it does not close compiler/release evidence or the remaining lifecycle issues.

### Unavailable Evidence (Requires GPU/WSI Environment)

- **Visual acceptance**: Project-owned 1280×720 headless captures with frozen settings — not yet produced.
- **Reference renderer calibration**: SSIM comparison against vkQuake — requires reference renderer installation.
- **Live WSI matrix**: Resize, minimize/restore, surface-loss recovery — requires live GPU + windowing system.
- **M1/M2 runtime performance budgets**: Timed parse/extract/upload/reload measurements on face-visible fixtures — not yet measured.

### Beta Design Limitations

- **No redistributable visible-face fixture**: Third-party `start.bsp` headless captures exist but are not checked in. The `dungeon-evidence-bsp2.bsp` fixture (41 faces, BSP2) proves face-visible BSP2 compilation but is a small technical proof, not a visual-calibration fixture.
- **Render-to-texture deferred**: BSP surfaces render into the main color/depth targets. Off-screen render targets are not yet supported.
- **Dynamic entity transforms**: Inline brush model transforms update through `BspMountState` per-batch transform maps. Full entity-to-scene-node mapping for external models is partially implemented.
- **Physics world colliders**: World trimesh collision from clipnodes is pending; point-contents and hull traces are functional.
- **Open arches only**: Doors (`func_door`, `func_button`, `func_plat`) are excluded from generator output. All generated room connections use open arches with no moving geometry or trigger wiring.
- **Single-layer Cartesian only**: No ramps, stairs, multi-floor, diagonal rooms, or curved corridors.

## See Also

- [API: BSP Beta](../api/17-bsp-beta.md) — exact function signatures, feature flags, error types
- [Internal: BSP Runtime and Lifetime](../internal/18-bsp-runtime-and-lifetime.md) — ownership graphs, protocol details
- [BSP Acceptance Spec](../../.internal-dev/specifications/bsp-acceptance.md) — acceptance criteria and evidence
- [Renderer AGENTS.md](../../src/renderer/AGENTS.md) — renderer contributor guide
