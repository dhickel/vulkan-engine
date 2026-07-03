# Validation Matrix

| Gate | Phase | Required Evidence | Pass Criteria | Failure Routing |
|---|---:|---|---|---|
| Core crate compiles/tests | 01 | `cargo test -p engine_events` | Event bus, ordering, recorder, and family tests pass without Vulkan | Fresh scoped code repair worker |
| Workspace dependency sanity | 01 | `cargo check` | New crate member does not create cycles or feature drift | Fresh scoped code repair worker |
| Input contract preserved | 02 | `cargo test -p input`, targeted tests | Existing layer ordering/consumption tests still pass; bridge observes post-dispatch state | Fresh scoped code repair worker |
| Renderer integration | 02 | `cargo test -p renderer`, `cargo check -p renderer --examples` | Facade exports compile and event bridge/runtime hooks do not break examples | Fresh scoped code repair worker |
| Root runtime integration | 02 | `cargo test -p engine` | Lifecycle/load helper tests verify event order without Vulkan where possible | Fresh scoped code repair worker |
| App/sample consumption | 03 | `cargo check -p editor`, `cargo check -p dungeon_dogfood` | Apps compile with minimal subscription/recording examples | Fresh scoped code repair worker |
| Docs alignment | 03 | Docs review plus stale sweep | Public/internal docs match implemented behavior and deferred families | Docs/evidence repair worker |
| Full compile/test | 04 | Commands listed in validation README | Required checks pass or blockers recorded | Repair by failed target |
| Runtime smoke | 04 | Debug JSONL path | Root runtime starts headless with no fatal errors before timeout | Runtime repair worker |
| True visual proof | 04 | Draw-target capture dir with PNG/JSON | Engine-owned headless draw capture succeeds and image is nonblank/expected | Browser/headless harness repair first, product repair only if real bug |
| Final quality | 04 | Final quality report and validation summary | Evidence index is internally consistent and no unresolved critical residuals remain | Planning revision or scoped repair depending on defect |

## Required Final Commands

```bash
cargo check
cargo test -p engine_events
cargo test -p input
cargo test -p renderer
cargo test -p engine
cargo check -p renderer --examples
cargo check -p editor
cargo check -p dungeon_dogfood
cargo check -p engine_pack
RUST_LOG=debug timeout --signal=INT 60s cargo run -- --project apps/editor/sample_project/engine.project.toml --headless --record_debug=10 --record_debug_interval=50 --record_debug_path=.internal-dev/debug_reports/sprint-05-event-system-lifecycle/root-runtime-events-timing.jsonl
RUST_LOG=info timeout --signal=INT 60s cargo run -- --project apps/editor/sample_project/engine.project.toml --headless --capture_frames=3 --capture_frame_start=5 --capture_frame_interval=5 --capture_target=draw --capture_dir=.internal-dev/captures/sprint-05-event-system-lifecycle-headless-draw
```

## Evidence Consistency Rules

- If any phase validator is missing or failed, top-level status must not be `fully_validated`.
- If draw capture is pending, top-level status must not be `fully_validated`.
- If a tooling fallback is used, record `TOOLING_CONSTRAINT` and main-thread approval.
- Superseded artifacts must be listed rather than silently ignored.
