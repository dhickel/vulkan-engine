# Debug and Diagnostics

## Overview

The engine ships with two layers of diagnostics:

1. **Frame extensions** (`FrameExtensions`): A general-purpose, immutable per-frame DTO
   for overriding renderer behaviour without scene mutation. Always available.

2. **Debug draw** (`debug-draw` feature): World-space line rendering for editor gizmos,
   physics wireframes, AI navmesh, and other development overlays. Gated behind an
   opt-in Cargo feature.

## Quick Start

### Frame Extensions

```rust
use renderer::{FrameExtensions, Renderer, SceneNodeId};
use glam::Mat4;

let mut ext = FrameExtensions::new();
ext.transform_overrides.insert(node_id, Mat4::IDENTITY);
renderer.set_frame_extensions(ext);
```

### Debug Draw

Enable the feature in `Cargo.toml`:

```toml
[dependencies]
renderer = { path = "src/renderer", features = ["debug-draw"] }
```

Then draw debug lines each frame:

```rust
use renderer::debug_draw::DebugDrawState;

let mut debug = DebugDrawState::new();
debug.push_line(Vec3::ZERO, Vec3::X, Vec3::new(1.0, 0.0, 0.0));

let mut ext = FrameExtensions::new();
ext.debug_lines = debug.take_lines();
renderer.set_frame_extensions(ext);

// Clear for next frame
debug.clear();
```

## Available Primitives

| Method | Description |
|--------|-------------|
| `push_line(from, to, color)` | Single world-space line segment |
| `push_aabb(min, max, color)` | Axis-aligned bounding box (12 edges) |
| `push_sphere(center, radius, color)` | Sphere as 3 orthogonal rings (96 segments) |
| `push_cross(position, size, cx, cy, cz)` | RGB axis cross/gizmo |

## Configuration

- Default capacity: 64K lines (128K vertices).
- Custom capacity: `DebugDrawState::with_capacity(lines)`.
- Overflow: silent discard (no runtime reallocation).

## Rendering Order

Debug lines render:
1. After the main geometry pass (depth-tested against scene geometry)
2. Before the UI (Imgui) pass

This ensures gizmos appear correctly occluded but never overlay editor UI.

## See Also

- [Frame Extensions API](../api/08-debug.md)
- [RenderExtensions Architecture](../internal/07-rendergraph-dependencies-and-aliasing.md)
- [Engine Integration Contracts](../internal/13-engine-integration-contracts.md)
