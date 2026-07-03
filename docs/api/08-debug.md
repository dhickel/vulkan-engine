# Debug UI & Timing Capture

> Source: [`src/renderer/src/debug_ui/mod.rs`](../src/renderer/src/debug_ui/mod.rs), [`src/renderer/src/api/renderer.rs`](../src/renderer/src/api/renderer.rs) — no legacy docs consulted.

## Overview

The engine includes an imgui-based debug UI with built-in panels for performance monitoring and the ability to register custom debug views. It also supports JSONL timing capture for offline analysis.

## Built-in Panels

| Toggle | Key | Panel |
|--------|-----|-------|
| Debug UI | F1 | Main debug overlay with performance graphs, frame timing, GPU stats |
| Console | F2 | In-engine console window |
| Debug overlay | (API only) | Performance overlay (FPS counter, frame time) |

These toggles are handled by `Renderer::update_input()` — if you bypass it for custom input handling, they won't work.

## Global Visibility Control

```rust
// Toggle main debug UI
renderer.toggle_debug_ui();
renderer.set_debug_ui_visible(true);
let visible = renderer.is_debug_ui_visible();

// Toggle console
renderer.toggle_console_ui();
renderer.set_console_ui_visible(true);

// Toggle performance overlay
renderer.toggle_debug_overlay_ui();
```

Defined at [`renderer.rs:262-280`](../src/renderer/src/api/renderer.rs:262).

## Custom Debug Views

Register your own imgui windows as debug panels:

```rust
pub struct DebugViewDescriptor {
    pub name: String,
    pub default_visible: bool,
}

pub type DebugViewCallback = Box<dyn FnMut(&DebugUiFrameContext<'_>) + Send>;

pub struct DebugUiFrameContext<'a> {
    pub ui: &'a imgui::Ui,
    // ...
}
```

```rust
// Register
let view_id = renderer.register_debug_view(
    DebugViewDescriptor { name: "My Stats".into(), default_visible: true },
    Box::new(|ctx: &DebugUiFrameContext| {
        ctx.ui.text("Hello from custom debug view!");
    }),
);

// Control
renderer.set_debug_view_enabled(view_id, false);
renderer.unregister_debug_view(view_id);
```

Defined at [`renderer.rs:244-259`](../src/renderer/src/api/renderer.rs:244) and [`debug_ui/mod.rs`](../src/renderer/src/debug_ui/mod.rs).

## Timing Capture (JSONL)

Record frame timing data to a JSONL file for offline profiling:

```rust
// Configure before starting
renderer.configure_debug_timing_recording(
    Some(10),                         // record for 10 seconds
    Some(50),                         // sample every 50ms
    Some("timing.jsonl".to_string()), // output path
)?;

// Start recording
let output_path = renderer.start_debug_timing_recording()?;
// ... run your app for 10 seconds ...
// Recording stops automatically after duration_secs
```

### CLI Integration

Examples support launch flags (see [`examples/common/mod.rs`](../src/renderer/examples/common/mod.rs)):

```sh
# Record 10 seconds at 50ms intervals
cargo run -p renderer --example demo_pbr -- \
  --record_debug=10 --record_debug_interval=50 \
  --record_debug_path=.internal-dev/debug_reports/demo_pbr-timing.jsonl

# With custom environment map
cargo run -p renderer --example api_test -- \
  --env src/renderer/src/assets/sky_maps/indoor_4k.exr \
  --record_debug=10 --record_debug_interval=50
```

### Timing Data Format

Each JSONL line is a `DebugTimingSnapshot`:

```rust
pub struct DebugTimingSnapshot {
    // frame index, timestamp, per-pass durations, GPU timestamps
}
pub struct DebugTimingRow {
    // individual pass timing
}
```

## See Also

- [02-renderer.md](02-renderer.md) — debug API on the Renderer
- [src/renderer/src/debug_ui/mod.rs](../src/renderer/src/debug_ui/mod.rs) — debug UI implementation
- [src/renderer/examples/common/mod.rs](../src/renderer/examples/common/mod.rs) — CLI argument parsing
