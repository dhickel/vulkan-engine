# Engine Arguments

## 1. Purpose & Audience
This page is the launch-argument reference for renderer runtime examples. Use it when running diagnostics, automation smoke runs, or reproducing rendering issues from the command line.

For the root project launcher, see [11-runtime-project-launcher.md](11-runtime-project-launcher.md). The root launcher runs data-driven projects with `cargo run -- --project <path>` and owns the alpha headless draw capture path. Renderer examples remain examples/diagnostics, not the primary alpha app runtime path.

## 2. Where This Fits in Engine Flow
Arguments on this page are parsed at renderer example startup before `Renderer::new(...)` enters the per-frame loop. Debug-record arguments configure/trigger the same timing recorder used by the in-engine debug UI.

## 3. Key Concepts
- Pass example arguments after `--` when using Cargo.
- The current shared parser lives in `src/renderer/examples/common/mod.rs`.
- The same argument set applies across runtime examples (`demo_pbr`, `demo_unlit`, `demo_model_load`, `demo_async_loading`, `api_test`).
- Custom Rust behavior belongs in app crates under `apps/<name>` and runs with `cargo run -p <app>`.
- Dynamic Rust hot reload, scripting runtime, runtime physics scene loading, audio integration, broad dogfood migration to project manifests, and generated app templates are deferred.
- `--record_debug=<seconds>` starts capture immediately at launch.
- `--record_debug_interval` and `--record_debug_path` can be supplied with or without `--record_debug`.
- If interval/path are supplied without `--record_debug`, values are configured but capture is not auto-started.

## 4. Code Walkthrough
Snippet Type: Real
```rust
// src/renderer/examples/common/mod.rs
if let Some(value) = arg.strip_prefix("--record_debug=") {
    options.record_debug_secs = Some(parse_positive_u64("--record_debug", value)?);
}
if let Some(value) = arg.strip_prefix("--record_debug_interval=") {
    options.record_debug_interval_ms =
        Some(parse_positive_u64("--record_debug_interval", value)?);
}
if let Some(value) = arg.strip_prefix("--record_debug_path=") {
    options.record_debug_path = Some(value.to_string());
}
```

Argument reference:

| Argument | Value | Meaning | Notes |
|---|---|---|---|
| `--env` / `--env=<path>` | file path | Environment map path for skybox/environment loading | Typically used with `api_test` |
| `--record_debug` / `--record_debug=<seconds>` | integer `>= 1` | Starts timing JSONL capture immediately | Example: `10` |
| `--record_debug_interval` / `--record_debug_interval=<ms>` | integer `>= 1` | Snapshot interval in milliseconds | Example: `50` |
| `--record_debug_path` / `--record_debug_path=<path>` | file path | Output JSONL path override | If omitted, default timestamped filename is used |

## 5. Best Practices
- For automated diagnosis, use:
- `--record_debug=10 --record_debug_interval=50`
- Keep run timeout at `60s` because startup commonly takes ~20-30 seconds.
- Use an explicit `--record_debug_path` in scripts so downstream parsing can target a known file.
- Contributor/agent default path: `.internal-dev/debug_reports/<example>-timing.jsonl`.

## 6. Gotchas & Failure Modes
- Missing values (for example bare `--record_debug`) are treated as argument errors and example startup exits early.
- Zero values are invalid for record duration/interval.
- Passing `--record_debug_interval` without `--record_debug` does not start capture.

## 7. Debugging Playbook
- Basic capture on startup:
- `RUST_LOG=debug timeout --signal=INT 60s cargo run -p renderer --example demo_pbr -- --record_debug=10 --record_debug_interval=50`
- Capture with known output path:
- `RUST_LOG=debug timeout --signal=INT 60s cargo run -p renderer --example demo_pbr -- --record_debug=10 --record_debug_interval=50 --record_debug_path=.internal-dev/debug_reports/demo_pbr-timing.jsonl`
- `api_test` with custom environment:
- `RUST_LOG=debug timeout --signal=INT 60s cargo run -p renderer --example api_test -- --env src/renderer/src/assets/sky_maps/indoor_4k.exr --record_debug=10 --record_debug_interval=50`

Root project launcher visual validation:

```sh
RUST_LOG=info timeout --signal=INT 60s cargo run -- \
  --project apps/editor/sample_project/engine.project.toml \
  --headless \
  --capture_target draw \
  --capture_frames 3 \
  --capture_frame_start 5 \
  --capture_frame_interval 5 \
  --capture_dir .internal-dev/captures/sprint-04-runtime-launcher/headless-draw
```

This command is intentionally rooted at `cargo run --`, not `cargo run -p renderer --example ...`, because the root binary is the data-driven project launcher.

## 8. Cross-Module Links
- Shared argument parser: `src/renderer/examples/common/mod.rs`
- Example with environment arg usage: `src/renderer/examples/api_test.rs`
- Debug recorder internals: `src/renderer/src/debug_ui/mod.rs`
- Renderer launch hooks for debug recording: `src/renderer/src/api/renderer.rs`

## 9. Standard References
- Cargo run argument forwarding: https://doc.rust-lang.org/cargo/commands/cargo-run.html

## 10. See Also
- [`docs/api/00-index.md`](00-index.md)
- [`docs/api/01-student-quickstart.md`](01-student-quickstart.md)
- [`src/renderer/AGENTS.md`](../../src/renderer/AGENTS.md)
- [`AGENTS.md`](../../AGENTS.md)
