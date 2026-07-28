# Debug and Diagnostics

## Frame Extensions

The `FrameExtensions` DTO provides a safe, immutable per-frame extension mechanism for
overriding renderer behaviour without mutating the scene graph.

```rust
use renderer::{FrameExtensions, SceneNodeId};
use glam::Mat4;

let mut ext = FrameExtensions::new();

// Override a node's world transform (hierarchy-consistent, app-owned)
ext.transform_overrides.insert(node_id, Mat4::from_translation(...));

renderer.set_frame_extensions(ext);
// Extensions are consumed during the next render submission and cleared.
```

### Properties
- **Immutable per frame**: Extensions are set before a frame and cannot be changed mid-frame.
- **No scene mutation**: Transform overrides propagate to subtrees but never mutate the
  scene-graph's local transforms.
- **Fence-safe**: Extensions are consumed by value during submission build and replaced
  with an empty default set.

## Debug Draw (feature-gated)

When the `debug-draw` Cargo feature is enabled, `FrameExtensions` can carry debug-line
segments for editor gizmos, physics wireframes, AI navmesh visualisation, and other
development overlays.

```rust
#[cfg(feature = "debug-draw")]
{
    use renderer::debug_draw::DebugDrawState;
    use glam::Vec3;

    let mut debug = DebugDrawState::new();

    // Push individual line segments
    debug.push_line(from, to, color);

    // Push common primitives
    debug.push_aabb(min, max, color);
    debug.push_sphere(center, radius, color);
    debug.push_cross(position, size, color_x, color_y, color_z);

    // Transfer to frame extensions
    let mut ext = FrameExtensions::new();
    ext.debug_lines = debug.take_lines();
    renderer.set_frame_extensions(ext);
}
```

### Rendering Pipeline
- **Pass**: DebugLinesPass runs after geometry, before UI.
- **Depth-tested**: Lines are occluded by scene geometry.
- **World-space**: Line coordinates are in world space, transformed by the active camera's
  view-projection matrix.
- **Ring-buffer**: Lines are uploaded to a host-visible GPU ring buffer each frame and
  cleared automatically.

### Capacity
- Default capacity is 64K lines (128K vertices).
- Lines beyond capacity are silently discarded (no reallocation at runtime).
- Use `DebugDrawState::with_capacity(n)` to configure a larger buffer.

### Shaders
- `debug_line.vert`: Transforms world-space position by view-projection push constant,
  reads vertices via buffer device address, and passes color to the fragment shader.
- `debug_line.frag`: Passes through the interpolated color as the output.

### Feature Gate
`debug-draw` is **not** in default features. Enable with:
```bash
cargo run --features debug-draw
```
