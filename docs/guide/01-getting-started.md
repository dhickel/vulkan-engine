# 01 — Getting Started

> Provenance: `CP-01`

## Prerequisites

You need a Rust toolchain and a Vulkan-capable GPU with an up-to-date driver.

### Rust Toolchain

Install Rust via [rustup](https://rustup.rs/). The engine targets Rust 2021 edition and builds on stable:

```sh
rustup update stable
rustc --version  # should be 1.70 or later
```

### Vulkan SDK

Install the [Vulkan SDK](https://vulkan.lunarg.com/) for your platform. After installation, verify that `vulkaninfo` runs and reports a capable device:

```sh
vulkaninfo --summary
```

Look for a discrete or integrated GPU in the output. If `vulkaninfo` only reports CPU-based implementations (e.g. `llvmpipe`), you do not have a usable Vulkan device and the renderer will fail at swapchain creation.

### GPU Support

The engine has been validated on:

- **Linux**: AMD RDNA2/RDNA3 (RADV), NVIDIA Turing/Ampere/Ada (proprietary driver 535+)
- **Windows**: not yet validated (contributions welcome)
- **macOS**: Vulkan is not natively supported; MoltenVK via `VK_ICD_FILENAMES` may work but is untested

Integrated GPUs (Intel UHD, AMD Radeon integrated) may work for simple scenes but have not been systematically validated. Do not claim generic GPU support beyond these observed working configurations.

## Clone and Build

```sh
git clone <repo-url> vulkan-engine
cd vulkan-engine
```

Verify the workspace builds:

```sh
cargo check
cargo test -p engine
```

This confirms the Rust toolchain and engine support crates (`engine_events`, `input`, etc.) are healthy. It does **not** verify Vulkan is usable.

## Compatibility Smoke: Run a Renderer Example

> **Label**: This is a **renderer-owned compatibility/diagnostic** example. It uses the renderer's own input and camera state (`Renderer::update_input`, `Renderer::render_scene`) — the pattern used by renderer-internal tests. Custom apps use the **app-owned loop** instead (see [Chapter 04](04-app-owned-loop.md)).

Run a renderer example to confirm Vulkan, GPU driver, and WSI are working end-to-end:

```sh
cargo run -p renderer --example demo_pbr
```

**Expected**: A window opens showing a PBR-rendered scene with a skybox, directional light, and several material samples. You can rotate the camera by moving the mouse and holding left-click.

If this fails with a Vulkan error, check:

- Your GPU driver is up to date
- `vulkaninfo --summary` shows a capable device
- You are not running over SSH without `DISPLAY`/`WAYLAND_DISPLAY` set
- (Wayland) `XDG_RUNTIME_DIR` is set

### Headless Smoke (No Window Required)

If you do not have a display or want to check GPU compute/rendering without a window:

```sh
RUST_LOG=debug timeout --signal=INT 60s cargo run -p renderer --example api_test -- --headless
```

A successful headless smoke prints startup logs and exits cleanly before the timeout fires. This validates Vulkan device creation, shader compilation, and offscreen rendering but does **not** exercise swapchain/WSI paths.

> **Constraint**: Headless smoke proves Vulkan device and rendering pipeline are functional. It does **not** prove windowed rendering will work — the swapchain, present, and resize paths are only exercised by the windowed examples.

## Next

Read [Chapter 02 — Architecture Overview](02-architecture-overview.md) to understand the crate layout and which parts you own vs which the renderer owns.
