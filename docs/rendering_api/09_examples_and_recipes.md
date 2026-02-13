# 09 - Examples and Recipes

This cookbook now tracks both compatibility-runtime scenarios and facade-first dogfooding paths.

## Recipe 1: Run compatibility runtime (`renderer::run`) (external/public)

```rust
fn main() {
    renderer::run();
}
```

Run:
```bash
cargo run
```

Best practice:
- Keep this path healthy through Unit 08; it is the parity baseline.

## Recipe 2: Run legacy debug-runtime scenarios (external/public)

```bash
cargo run -- debug_runtime testpbr
cargo run -- debug_runtime testunlit
cargo run -- --debug-runtime=testunlit
```

What this validates:
- Legacy startup scene behavior used as side-by-side parity reference.

## Recipe 3: Run facade-routed debug-runtime scenarios (external/public)

```bash
cargo run -- debug_runtime facade_pbr
cargo run -- debug_runtime facade_unlit
cargo run -- debug_runtime facade_model_load
```

What this validates:
- `facade_pbr`: facade startup-scene pbr path (parity-oriented).
- `facade_unlit`: unlit parity path while unlit material override is still startup-driven.
- `facade_model_load`: explicit facade `AssetManager::load_model + Scene::merge_fragment` workflow.

## Recipe 4: Run facade examples directly (external/public)

```bash
cargo run -p renderer --example demo_pbr
cargo run -p renderer --example demo_unlit
cargo run -p renderer --example demo_model_load
```

Best practice:
- Prefer these examples for day-to-day API dogfooding.

## Recipe 5: Model load with public facade only (external/public)

```rust
use renderer::{Renderer, Scene};

fn build_scene(renderer: &mut Renderer) -> Result<Scene, renderer::RendererError> {
    let mut scene = Scene::new();
    let fragment = {
        let mut assets = renderer.assets();
        assets.load_model("src/renderer/src/assets/DamagedHelmet.glb")?
    };
    scene.merge_fragment(None, fragment)?;
    Ok(scene)
}
```

Best practice:
- Keep model loading and scene mounting in facade APIs; avoid internal cache imports.

## 04.5 Core Parity Checklist

Use this checklist after each Unit 05-08 change.

| Scenario | Legacy Baseline | Facade Runtime | Facade Example | Status | Notes |
|---|---|---|---|---|---|
| PBR startup/material path | `cargo run -- debug_runtime testpbr` | `cargo run -- debug_runtime facade_pbr` | `cargo run -p renderer --example demo_pbr` | Pending | Compare camera motion, skybox, material shading.
| Unlit startup/material path | `cargo run -- debug_runtime testunlit` | `cargo run -- debug_runtime facade_unlit` | `cargo run -p renderer --example demo_unlit` | Pending | Validate unlit pipeline routing and visual parity.
| Model load + scene mount | N/A (legacy startup-only) | `cargo run -- debug_runtime facade_model_load` | `cargo run -p renderer --example demo_model_load` | Pending | Validate repeated load+merge and transform edits (two mounted instances).

## Staged Parity Boundaries

- `04.5 core parity`: pbr/unlit/model-load facade paths and baseline side-by-side checks.
- `08 full parity`: includes Unit 05 deferred loading and Unit 06 environment switching parity.

## Internal-only advanced recipes (in-tree/internal)

- Submission flags and rendergraph internals remain implementation details.
- Deferred transfer submission plumbing (`VkHostBuffer`, `VkTransfer`, `VkFenceQueue`) remains internal.
- Debug scenario internals in `src/renderer/src/scene/debug_scenarios.rs` remain compatibility scaffolding.
