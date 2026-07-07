# Validation Matrix

| Gate | Phase | Required Evidence | Pass Criteria | Failure Routing |
|---|---:|---|---|---|
| Contract audit | 01 | audit artifact and validator report | Live code/docs claims are inventoried, stale claims identified, phase scope confirmed | Plan revision or docs/evidence repair |
| App-template command | 02 | `cargo test -p engine_pack`, generated app check | Template generation is deterministic and generated app builds without renderer internals | Fresh scoped code repair |
| Template docs | 02 | docs diff and validator review | Docs explain app crates, generated template status, and workspace/manual steps honestly | Docs/evidence repair |
| Scripting crate boundary | 03 | `cargo test -p scripting` | Log/event/error helpers work, errors carry script context, raw mutable access is not the promoted path | Fresh scoped code repair |
| Script event contract | 03 | `cargo test -p engine_events` plus producer tests | `ScriptingEvent` use is dependency-safe and event ordering claims are documented | Fresh scoped code repair |
| Script asset validation | 03 | `cargo test -p renderer`, `cargo test -p engine_pack` if implemented | Script asset metadata accepts valid durable IDs and rejects invalid schema/handles | Fresh scoped code repair |
| Hot Rust docs | 04 | docs review | Rust app loop, asset/script reload, and Rust code reload are distinguished | Docs/evidence repair |
| Full compile/test | 04 | final command log | Required checks pass or inherited blockers are recorded conservatively | Repair by failed target |
| Headless capture | 04 only if visible behavior changed | capture directory and reconciled validator notes | Engine-owned `--headless --capture_target draw` proof exists; no desktop screenshots | Harness repair first, product repair only after proven bug |
| Evidence index | 04 | `artifacts/validation-summary.json` | Status is conservative and cross-fields do not contradict validators/residuals | Docs/evidence repair |
| Final quality | 04 | `validation/final-quality-review.md` | Plan criteria, reports, code/docs/tests/evidence are consistent | Scoped repair or planning revision |

## Required Final Commands

```bash
cargo fmt --check
cargo check
cargo test -p scripting
cargo test -p engine_events
cargo test -p renderer
cargo test -p engine_pack
cargo check -p renderer --examples
cargo check -p editor
cargo check -p dungeon_dogfood
```

Conditional:

```bash
cargo test -p dungeon_dogfood
```

Run only if Sprint 08 changes dogfood tests/runtime expectations. If it remains blocked before dogfood tests by the inherited renderer test-profile `russimp_sys` issue in `src/renderer/src/data/assimp_util.rs`, record that exact blocker as inherited.

Template-specific if Phase 02 implements `new-app`:

```bash
cargo run -p engine_pack -- new-app /tmp/engine-sprint08-template --id sprint08.template --name "Sprint 08 Template"
cargo check --manifest-path /tmp/engine-sprint08-template/Cargo.toml
```

Capture-specific if visible behavior changes:

```bash
RUST_LOG=info timeout --signal=INT 60s cargo run -- --project apps/editor/sample_project/engine.project.toml --headless --capture_target draw --capture_frames 3 --capture_frame_start 5 --capture_frame_interval 5 --capture_dir .internal-dev/captures/sprint-08-scripting-hot-rust-strategy/headless-draw
```

## Evidence Consistency Rules

- If any required validator is missing or failed, top-level status must not be `fully_validated`.
- If capture is applicable and pending, top-level status must not be `fully_validated`.
- If `cargo test -p dungeon_dogfood` is blocked by inherited renderer test-profile behavior, list it under accepted residuals.
- If script support remains experimental, docs and summary must not call it supported gameplay scripting.
- If a tooling/model fallback occurs, record `TOOLING_CONSTRAINT` and main-thread approval.
