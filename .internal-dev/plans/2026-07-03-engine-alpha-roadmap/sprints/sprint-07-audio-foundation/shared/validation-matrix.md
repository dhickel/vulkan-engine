# Validation Matrix

| Gate | Phase | Required Evidence | Pass Criteria | Failure Routing |
|---|---:|---|---|---|
| Audio core API | 01 | `cargo test -p audio`, code review | Clip/probe/error/control tests pass without opening default device | Fresh scoped code repair worker |
| Audio dependency hygiene | 01 | dependency/import scan, `cargo tree -p audio` if useful | No renderer/Vulkan/window/editor/dogfood dependency in audio | Fresh scoped code repair worker |
| Device smoke gating | 01/03 | ignored/manual test or runtime command status | Device-backed playback is optional and reported as pass/skipped/blocked | Repair code/docs depending on defect |
| Package audio metadata | 02 | `cargo test -p renderer` targeted tests | Audio asset kind/metadata accepts valid records and rejects invalid IDs/formats/handle shapes | Fresh scoped code repair worker |
| Scene/app audio refs | 02 | `cargo test -p renderer` targeted tests | Audio references round-trip or validate and reject unknown/blank/runtime-handle IDs | Fresh scoped code repair worker |
| CLI validation | 02 | `cargo test -p engine_pack`, CLI fixture tests | `engine_pack` reports audio metadata failures through existing validation APIs | Fresh scoped code repair worker |
| Event bridge | 03 | `cargo test -p engine_events`, `cargo test -p audio`, consumer tests | Playback status maps to `AudioEvent` with durable `AudioClipId` and no dependency inversion | Fresh scoped code repair worker |
| Sample/dogfood proof | 03 | `cargo check -p dungeon_dogfood` or sample check, optional smoke/debt report | Packaged clip reference is demonstrated or debt artifact is explicit | Fresh scoped repair or docs/evidence repair |
| Docs | 04 | docs review and stale-reference sweep | Docs match implemented alpha contract and device-dependent limits | Docs/evidence repair worker |
| Full compile/test | 04 | required final commands | All required checks pass or blockers are recorded conservatively | Repair by failed target |
| Runtime smoke | 04 | optional timeout-bound log/debug report if attempted | No fatal errors before timeout; device result honestly recorded | Runtime repair worker |
| Headless draw capture | 04 | capture dir only if visible behavior changed | True engine-owned capture succeeds and is reconciled by validator | Harness repair first; product repair only after proven bug |
| Final quality | 04 | `validation/final-quality-review.md`, validation summary | Plan criteria, reports, code/docs/tests/evidence are consistent | Planning revision or scoped repair depending on defect |

## Required Final Commands

```bash
cargo fmt --check
cargo check
cargo test -p audio
cargo check -p audio
cargo test -p engine_events
cargo test -p renderer
cargo test -p engine_pack
cargo check -p renderer --examples
cargo check -p editor
cargo check -p dungeon_dogfood
```

If Phase 03 changes dogfood runtime behavior:

```bash
cargo test -p dungeon_dogfood
```

Optional device smoke, only when explicitly enabled and a device is available:

```bash
RUST_LOG=debug timeout --signal=INT 60s cargo run -p dungeon_dogfood -- --audio-smoke
```

Workers may replace the optional command with the actual implemented flag/example path, but the final report must record the exact command and whether it was skipped, blocked, or passed.

If visible renderer/editor behavior changes, additionally run true headless draw capture through `.internal-dev/skills/engine-headless-capture-validation/SKILL.md` with `--headless --capture_target draw`; no desktop screenshots.

## Evidence Consistency Rules

- If any required phase validator is missing or failed, top-level status must not be `fully_validated`.
- If device smoke is skipped or blocked, `fully_validated` may still be true only if the summary clearly states core support is validated and device playback remains host-dependent.
- If visible behavior changed and capture proof is pending, top-level status must not be `fully_validated`.
- If dogfood proof is deferred, the debt artifact must be listed as an accepted residual.
- If a tooling fallback is used, record `TOOLING_CONSTRAINT` and main-thread approval.
- Superseded artifacts must be listed in `artifacts/validation-summary.json`.
