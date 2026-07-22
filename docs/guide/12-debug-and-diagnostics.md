# 12 — Debug & Diagnostics

> Provenance: `G-12`

This chapter covers the engine's diagnostic tooling: logging, timing capture (JSONL), headless frame capture, debug UI panels, validation layers, asset validation, and collider validation. Use these tools during development to profile performance, validate visual output, and diagnose engine behavior.

For the API reference, see [Debug UI & Timing Capture](../api/08-debug.md). For the headless capture validation skill, see `.internal-dev/skills/engine-headless-capture-validation/SKILL.md`.

## Logging

The engine uses `env_logger` for structured logging via the `log` crate. Set the `RUST_LOG` environment variable to control verbosity:

```sh
# Info-level: startup sequence, scene loading, frame timing
RUST_LOG=info cargo run -p renderer --example demo_pbr

# Debug-level: asset loading detail, material binding, pipeline state
RUST_LOG=debug cargo run -p renderer --example demo_pbr

# Per-module filtering:
RUST_LOG=renderer=debug,winit=warn cargo run -p renderer --example demo_pbr
```

Your app should initialize logging early in `main()`:

```rust
env_logger::Builder::from_default_env()
    .filter_level(log::LevelFilter::Info)
    .init();
```

## Timing Capture (JSONL)

> Provenance: `G-12-TIMING` — Excerpt

The renderer can record frame timing data to a JSONL file for offline profiling. This is the **recommended default diagnostic** for investigating frame time spikes, GPU bubbles, and asset load stalls.

### Launch Flags (CLI)

All renderer examples support timing-capture launch flags. Run from the repository root:

```sh
# Record 10 seconds at 50ms intervals
RUST_LOG=debug timeout --signal=INT 60s cargo run -p renderer --example demo_pbr -- \
  --record_debug=10 --record_debug_interval=50 \
  --record_debug_path=.internal-dev/debug_reports/demo_pbr-timing.jsonl

# With custom environment map
RUST_LOG=debug timeout --signal=INT 60s cargo run -p renderer --example api_test -- \
  --env src/renderer/src/assets/sky_maps/indoor_4k.exr \
  --record_debug=10 --record_debug_interval=50 \
  --record_debug_path=.internal-dev/debug_reports/api_test-timing.jsonl
```

| Flag | Default | Purpose |
|------|---------|---------|
| `--record_debug` | (disabled) | Recording duration in seconds |
| `--record_debug_interval` | 50 | Sample interval in milliseconds |
| `--record_debug_path` | auto-generated | Output file path |

Output is newline-delimited JSON. Each line is a timing record with frame number, elapsed time, and per-pass duration data.

### Programmatic API

```rust
// Configure before the render loop
renderer.configure_debug_timing_recording(
    Some(10),                            // record for 10 seconds
    Some(50),                            // sample every 50ms
    Some("timing.jsonl".to_string()),    // output path
)?;

// Start recording
let output_path = renderer.start_debug_timing_recording()?;
// ... run the loop ...
// Recording stops automatically after duration_secs
```

## Headless Frame Capture

> Provenance: `G-12-HEADLESS` — Excerpt

The engine supports headless rendering with deterministic frame capture. This is the preferred validation method for renderer changes when WSI (windowing) is unavailable or when visual evidence must be reproducible without a desktop.

### Headless Examples

```sh
# Culling validation
cargo run -p renderer --example capture_culling -- --headless --culling=on

# Shadow map validation
cargo run -p renderer --example capture_shadows -- --headless
```

Headless captures write PNG frames plus sidecar JSON metadata to the capture output directory. The output path is printed on startup.

> **Important**: Headless frame captures are deterministic evidence of renderer output, not WSI (swapchain presentation) evidence. Do not describe headless captures as "windowed" or "real-WSI" results.

### Capture Validation Skill

For comprehensive headless capture workflows (scene setup, camera positioning, capture comparison), use the project skill:

```
.skill:engine-headless-capture-validation
```

This skill provides timeout-bound capture commands, output directory conventions, and validation patterns.

## Debug UI

> Provenance: `G-12-DEBUG-UI` — Excerpt

The engine includes an ImGui-based debug UI with built-in panels.

### Built-in Toggles

| Key | Panel |
|-----|-------|
| F1 | Main debug overlay: performance graphs, frame timing, GPU stats |
| F2 | In-engine console window |

These toggles work through `route_platform_input_to_app` (app-owned path) or `update_input` (compatibility path). On the app-owned path, they are processed as renderer platform side effects and do not require direct `update_input` calls.

### Programmatic Control

```rust
// Toggle main debug UI
renderer.toggle_debug_ui();
renderer.set_debug_ui_visible(true);
let visible = renderer.is_debug_ui_visible();

// Toggle console
renderer.toggle_console_ui();
renderer.set_console_ui_visible(true);

// Toggle FPS overlay
renderer.toggle_debug_overlay_ui();
```

### Custom Debug Views

Register ImGui windows as debug panels:

```rust
use renderer::prelude::{DebugViewDescriptor, DebugUiFrameContext};

let view_id = renderer.register_debug_view(
    DebugViewDescriptor {
        name: "My App Stats".into(),
        default_visible: true,
    },
    Box::new(|ctx: &DebugUiFrameContext| {
        ctx.ui.text("Hello from custom debug view!");
    }),
);

// Toggle visibility
renderer.set_debug_view_enabled(view_id, false);

// Remove
renderer.unregister_debug_view(view_id);
```

## Validation Layers

> Provenance: `G-12-VALIDATION` — Excerpt

Vulkan validation layers can be enabled via `RendererConfig`:

```rust
let config = RendererConfig {
    validation_layer: true,
    ..Default::default()
};
let mut renderer = Renderer::new(config, &window)?;
```

When enabled, `VK_LAYER_KHRONOS_validation` is activated. Validation messages appear in stderr via the debug callback. **This significantly reduces performance** — use only for debugging Vulkan issues, not for normal development or production runs.

### Validation Layer Diagnostics

Common issues surfaced by validation layers:

| Message Pattern | Likely Cause |
|-----------------|--------------|
| `VUID-vkCmdDraw-None-*` | Pipeline state mismatch, missing descriptor binding |
| `VUID-vkAcquireNextImageKHR-*` | Swapchain out of date, surface lost |
| `VUID-vkQueueSubmit-*` | Semaphore/fence lifecycle error |
| `UNASSIGNED-CoreValidation-DrawState-*` | Descriptor set not updated before draw |
| Object leak messages at shutdown | Missing `vkDestroy*` or `vkFree*` calls |

## Asset Validation

### Command-Line Validation

Validate package manifests, projects, and scenes without starting the renderer:

```sh
# Create a temp package and validate:
TMPDIR=$(mktemp -d)
cargo run -p engine_pack -- new-package "$TMPDIR/package.toml" --id pkg --name "Pkg"
cargo run -p engine_pack -- validate-package "$TMPDIR/package.toml" --expected-package-id pkg

# Or validate a complete project scaffold:
cargo run -p engine_pack -- new-project "$TMPDIR/project" --id test.valid --name "Valid Test"
cargo run -p engine_pack -- validate-project "$TMPDIR/project/engine.project.toml"
cargo run -p engine_pack -- validate-scene "$TMPDIR/project/scenes/start.engine.scene.json" \
  --project "$TMPDIR/project/engine.project.toml"
rm -rf "$TMPDIR"
```

### Programmatic Validation

```rust
use renderer::prelude::{
    validate_scene_file, validate_scene_str,
    validate_package_manifest_file, validate_package_manifest_str,
    SceneValidationOptions, PackageValidationOptions, ValidationError,
};

// Validate a scene with known asset IDs
let options = SceneValidationOptions::default()
    .with_known_asset_ids(vec!["package.model.crate", "package.tex.brick"]);
validate_scene_file_with_options("scenes/my_scene.json", &options)?;

// Validate a package with source file checks
let options = PackageValidationOptions::default().check_source_files(true);
let records = validate_package_manifest_file("assets/package.toml", &options)?;
```

## Collider Validation

> Provenance: `G-12-COLLIDER` — Excerpt

Validate physics collider geometry without creating a world:

```rust
use physics::{validate_collider_shape, BodyKind, ColliderShape};

// Check a TriMeshStatic before attempting world creation
validate_collider_shape(
    &ColliderShape::TriMeshStatic {
        vertices: my_vertices,
        indices: my_indices,
    },
    BodyKind::Static,
)?;

// Check a ConvexHull
validate_collider_shape(
    &ColliderShape::ConvexHull { points: my_points },
    BodyKind::Dynamic,
)?;
```

This catches geometry errors early without mutating `PhysicsWorld` state.

## Asset Pumping Diagnostics

Monitor the asset pipeline's health:

```rust
let pending = renderer.pending_load_count();
if pending > 0 {
    eprintln!("{pending} asset loads pending");
}

if renderer.has_pending_loads() {
    // Consider showing a loading indicator
}
```

The `pump_asset_tasks` return value indicates how many task steps were processed:

```rust
let pumped = renderer.pump_asset_tasks(32)?;
if pumped > 0 {
    log::debug!("asset pump processed {pumped} steps");
}
```

## Environment State Diagnostics

Track environment map loading:

```rust
if let Some(env_handle) = current_env {
    match asset_manager.environment_state(env_handle)? {
        EnvironmentState::Unloaded => { /* pre-load */ }
        EnvironmentState::Loading => { /* IBL maps processing */ }
        EnvironmentState::Ready => { /* fully available */ }
        EnvironmentState::Failed(e) => {
            eprintln!("environment load failed: {e}");
        }
    }
}
```

## Resize Diagnostics

The `SkippedResizePending` outcome is rate-limited in log output (logged once per resize event, not every skipped frame):

```rust
Ok(FrameRenderOutcome::SkippedResizePending) => {
    // Printed at most once per resize transition
    eprintln!("render skipped while swapchain resize is pending");
}
```

## Runnable Verification

### Logging and Basic Diagnostics

```sh
# Info-level smoke — verify startup logging, frame count, shutdown
RUST_LOG=info timeout --signal=INT 30s cargo run -p renderer --example demo_pbr
```

### Timing Capture

```sh
# Record 10 seconds of frame timing
RUST_LOG=debug timeout --signal=INT 60s cargo run -p renderer --example demo_pbr -- \
  --record_debug=10 --record_debug_interval=50 \
  --record_debug_path=.internal-dev/debug_reports/demo_pbr-timing.jsonl

# Verify output
ls -la .internal-dev/debug_reports/demo_pbr-timing.jsonl
head -n 3 .internal-dev/debug_reports/demo_pbr-timing.jsonl
```

### Asset Validation (no GPU required)

```sh
# Create a temp package and validate:
TMPDIR=$(mktemp -d)
cargo run -p engine_pack -- new-package "$TMPDIR/package.toml" --id pkg --name "Pkg"
cargo run -p engine_pack -- validate-package "$TMPDIR/package.toml" --expected-package-id pkg
rm -rf "$TMPDIR"
```

### Physics Collider Validation

```sh
cargo test -p physics
```

### Build Check

```sh
cargo check -p renderer --examples
cargo check -p engine_pack
```

## Distinguishing Diagnostic Evidence

| Evidence Type | When to Use | Example |
|---------------|-------------|---------|
| **Log output** | Startup, asset loading, error tracing | `RUST_LOG=debug` |
| **Timing JSONL** | Frame time profiling, GPU bubbles | `--record_debug=10` |
| **Headless capture** | Visual validation without desktop | `--headless` examples |
| **WSI observation** | Visual validation with display | Run windowed examples |
| **Validation layers** | Vulkan API misuse debugging | `validation_layer: true` |
| **CLI validators** | Asset/manifest schema checking | `engine_pack validate-*` |
| **Unit tests** | Crate-level contract validation | `cargo test -p <crate>` |

Do not describe headless captures as WSI evidence. Do not describe log output as visual proof. Each diagnostic tool has a specific evidence class.

## Next

Continue to [13 — Packaging & Distribution](13-packaging-and-distribution.md) to learn about `engine_pack`: scaffolding apps and projects, authoring packages, validating assets, and producing packed output.
