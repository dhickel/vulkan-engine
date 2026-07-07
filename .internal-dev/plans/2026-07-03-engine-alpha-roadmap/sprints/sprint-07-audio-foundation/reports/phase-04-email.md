# Sprint 07 Phase 04 Docs And Final Validation Report Draft

## Summary

Updated public and internal docs for the implemented alpha audio foundation, reconciled stale audio wording, ran the final validation matrix, and kept the canonical summary conservative pending Phase 04 validator and final quality review.

## Changed Docs

- `docs/api/13-audio-foundation.md`: new public audio contract covering package metadata, scene references, device-independent load/probe, explicit playback, dogfood proof, audio events, and limits.
- `docs/internal/12-audio-foundation.md`: new internal boundary guide for audio crate, validators, event bridge, device gate, and deferred work.
- `docs/api/00-index.md`: added audio foundation navigation and corrected top-level deferred wording.
- `docs/internal/00-index.md`: added audio foundation navigation and source pointer.
- `docs/api/10-packaging-cli.md`: documented audio extension scanning and audio metadata validation.
- `docs/api/12-events-and-lifecycle.md`: documented dogfood audio event bridge and preserved `engine_events` dependency boundary.
- `docs/api/11-runtime-project-launcher.md`: clarified root runtime does not perform audio playback while app crates may own audio bridges.
- `docs/api/07-engine-arguments.md`, `docs/api/01-student-quickstart.md`, and `apps/dungeon_dogfood/README.md`: replaced stale broad audio-deferred wording with the current alpha limits and dogfood smoke command.

## Validation Commands

- `cargo fmt --check`: pass.
- `cargo check`: pass with existing renderer warning noise.
- `cargo test -p audio`: pass, 7 passed, 1 ignored manual device smoke.
- `cargo check -p audio`: pass.
- `cargo test -p engine_events`: pass, 7 passed.
- `cargo test -p renderer`: pass, 160 unit tests, 17 integration tests, 5 ignored doctests.
- `cargo test -p engine_pack`: pass, 17 CLI tests.
- `cargo check -p renderer --examples`: pass with existing renderer warning noise.
- `cargo check -p editor`: pass with existing renderer/editor warning noise.
- `cargo check -p dungeon_dogfood`: pass with existing renderer/dogfood warning noise.
- `cargo test -p dungeon_dogfood`: blocked before dogfood tests run by existing renderer test-profile compile errors in `src/renderer/src/data/assimp_util.rs` against current `russimp_sys` `aiScene`/`aiNode` bindings.

## Stale-Reference Sweep

Command:

```bash
rg -n "/tmp|desktop screenshot|screenshot|TODO|pending|planned|not implemented|agent id|fully_validated|TOOLING_CONSTRAINT|audio integration|does not infer audio|live audio emission is later roadmap" docs .internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-07-audio-foundation
```

Triage:

- No stale public docs remain claiming `scan-assets` does not infer audio or that all audio integration is deferred.
- `/tmp` hits in `docs/api/10-packaging-cli.md` are command examples, not authoritative evidence paths.
- `desktop screenshot` and `screenshot` hits are negative validation rules or historical validator evidence saying screenshots were not used.
- `pending`/`planned` hits in sprint planning files, validation reports, and `validation-summary.json` are phase history or intentionally conservative state pending Phase 04 validation/final quality review.
- `not implemented` hits in rendergraph docs are unrelated rendergraph future-direction wording.
- `TOOLING_CONSTRAINT` appears only in planning instructions and no actual tooling constraint was used.
- `audio integration` hits in sprint phase history are historical context; current public docs now use narrower implemented/unsupported wording.

## Device And Capture Status

Device smoke remains skipped. The opt-in dogfood command is:

```bash
RUST_LOG=debug timeout --signal=INT 60s cargo run -p dungeon_dogfood -- --audio-smoke
```

No default audio output device was opened during Phase 04 validation.

No capture was required. Sprint 07 changed non-visual audio, metadata, event, dogfood, and docs surfaces only. No desktop screenshots were used.

## Residuals

- `cargo test -p dungeon_dogfood` remains blocked by the existing renderer test-profile `russimp_sys` binding issue before dogfood tests execute.
- Device-backed playback depends on host output availability and remains explicitly opt-in.
- Root runtime/editor audio playback and editor audio placement are not implemented.
