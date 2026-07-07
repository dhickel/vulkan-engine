# Date
2026-07-07

# Change Summary
Closed out the engine runtime abstractions work for GitHub issues #35-#37. The root `engine` crate is now documented as a thin app-facing facade, the renderer caller-view path is the intended camera handoff for app-owned loops, and `dungeon_dogfood` is the real-app proof for app-owned input, events, frame clock, camera/controller, and raw primitive access.

# Files
- `artifacts/engine-runtime-abstractions-issues-35-37/validation-summary.json`: added the canonical validation index for all phases, command gates, runtime smokes, headless capture evidence, residual risks, and final review.
- `.internal-dev/specifications/api.md`, `.internal-dev/specifications/architecture.md`, `.internal-dev/specifications/service-graph.md`, `.internal-dev/specifications/services.md`, `.internal-dev/specifications/decisions.md`: recorded the root bin+lib facade, support-crate ownership boundaries, renderer `CameraView` DTO placement, and caller-view render contract.
- `.internal-dev/knowledge/renderer-camera-override-behavior.md`: clarified legacy renderer-owned camera behavior versus the caller-provided `CameraView` path.
- `docs/api/00-index.md`, `docs/api/01-student-quickstart.md`, `docs/api/11-runtime-project-launcher.md`, `docs/internal/01-architecture.md`, `docs/internal/04-api-to-backend-handoff.md`: updated the beginner and internal contracts to present the root facade and app-owned runtime model as intended truth.
- Existing Phase 03, Phase 04, and Phase 05 docs remain as supporting implementation history and compatibility labeling.

# Behavioral Impact
No new runtime behavior was added in this closeout phase. The behavioral contract is now documented consistently: apps may own input, events, frame timing, and camera state through the root `engine` facade, pass a `CameraView` into renderer submission, and still drop down to raw renderer/input/events primitives when needed.

# Specification Impact
Updated API, architecture, service graph, service, and decision specifications because this closeout finalized the intended ownership model for the root `engine` facade, support crates, renderer caller-view handoff, compatibility exports, and dogfood proof.

# Validation
- `cargo fmt --check`
- `cargo check -p input --quiet`
- `cargo test -p input --quiet`
- `cargo check -p engine_events --quiet`
- `cargo test -p engine_events --quiet`
- `cargo check -p renderer --quiet`
- `cargo test -p renderer --quiet`
- `cargo check -p renderer --examples --quiet`
- `cargo check -p dungeon_dogfood --quiet`
- `cargo check -p marching_terrain --quiet`
- `cargo check --quiet`
- `RUST_LOG=debug timeout --signal=INT 60s cargo run -p renderer --example api_test`
- `RUST_LOG=debug timeout --signal=INT 60s cargo run -p dungeon_dogfood`
- Stale active-contract sweep over `docs`, `.internal-dev`, `src`, and `apps`
- Beginner-path/facade-language classification sweep over `docs`, `.internal-dev`, `src`, and `apps`
- Capture sidecar metadata reconciliation for requested app-owned `CameraView`, submitted caller-view render path, unused legacy renderer camera path, and residual risks

# Risks
The compatibility renderer-owned lifecycle/input/camera APIs remain available and must not be mistaken for the new intended app-owned path in future docs. Windowed runtime smokes reached startup/event-loop milestones and ran until timeout, but this environment still logged repeated swapchain acquire retry warnings near timeout. Known dead-code warning noise remains outside the scope of this refactor.

# Follow-up Items
- Investigate swapchain acquire retry warnings separately if they reproduce in non-timeout runtime validation.
- Keep future engine orchestration additions thin until more dogfood usage proves the abstraction needs.
