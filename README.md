# vulkan-engine

Rust 2021 Vulkan engine workspace.

Current workspace members:
- `engine` root binary: alpha runtime project launcher for data-driven projects.
- `src/renderer`: renderer runtime/API facade and diagnostic examples.
- `src/input`: frame-buffered input system.
- `src/audio`: alpha audio crate.
- `src/physics`: alpha physics crate.
- `src/scripting`: alpha scripting crate.
- `apps/dungeon_dogfood`: custom Rust dogfood application.
- `apps/editor`: alpha editor application.

Documentation entrypoints:
- API path: `docs/api/00-index.md`
- Internal contributor path: `docs/internal/00-index.md`
- Current alpha readiness baseline: `docs/gap-report.md`

Runtime entrypoints:
- Run a project through the root launcher:
  `cargo run -- --project apps/editor/sample_project/engine.project.toml`
- Capture validation must use the true headless draw-target path:
  `RUST_LOG=info timeout --signal=INT 60s cargo run -- --project apps/editor/sample_project/engine.project.toml --headless --capture_target draw --capture_frames 3 --capture_frame_start 5 --capture_frame_interval 5 --capture_dir .internal-dev/captures/sprint-04-runtime-launcher/headless-draw`
- Use renderer examples under `src/renderer/examples/` for renderer diagnostics and facade examples, for example `cargo run -p renderer --example demo_pbr`.
- Custom Rust applications live under `apps/<name>` and run with `cargo run -p <app>`, for example `cargo run -p dungeon_dogfood`.
- Dynamic Rust hot reload, scripting runtime, event-system integration, physics integration, audio integration, broad dogfood migration to project manifests, and generated app templates are deferred.
- Treat crate/app existence as workspace presence, not alpha readiness. Current residuals are tracked through the alpha readiness baseline and Sprint 01 follow-up artifacts.
