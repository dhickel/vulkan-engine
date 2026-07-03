# vulkan-engine

Rust 2021 Vulkan engine workspace.

Current workspace members:
- `engine` root binary: migration stub that prints example commands and exits.
- `src/renderer`: renderer runtime/API facade and canonical runtime examples.
- `src/input`: frame-buffered input system.
- `src/audio`: alpha audio crate.
- `src/physics`: alpha physics crate.
- `src/scripting`: alpha scripting crate.
- `apps/dungeon_dogfood`: dogfood application.
- `apps/editor`: alpha editor application.

Documentation entrypoints:
- API path: `docs/api/00-index.md`
- Internal contributor path: `docs/internal/00-index.md`
- Current alpha readiness baseline: `docs/gap-report.md`

Runtime entrypoints:
- `cargo run` prints migration guidance and exits.
- Use renderer examples under `src/renderer/examples/` for renderer behavior, for example `cargo run -p renderer --example demo_pbr`.
- Treat crate/app existence as workspace presence, not alpha readiness. Current residuals are tracked through the alpha readiness baseline and Sprint 01 follow-up artifacts.
