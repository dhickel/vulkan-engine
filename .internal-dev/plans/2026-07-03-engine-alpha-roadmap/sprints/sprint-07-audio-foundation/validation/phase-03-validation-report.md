# Sprint 07 Phase 03 Validation Report

Verdict: PASS

## Files Reviewed

- `AGENTS.md`
- `.internal-dev/AGENTS.md`
- `.gitignore`
- `Cargo.lock`
- `Cargo.toml`
- `src/events/Cargo.toml`
- `src/events/src/lib.rs`
- `src/renderer/Cargo.toml`
- `src/renderer/src/lib.rs`
- `apps/dungeon_dogfood/Cargo.toml`
- `apps/dungeon_dogfood/assets/content_pack.toml`
- `apps/dungeon_dogfood/assets/content_manifest.md`
- `apps/dungeon_dogfood/assets/audio/startup_ping.wav`
- `apps/dungeon_dogfood/src/audio_bridge.rs`
- `apps/dungeon_dogfood/src/content.rs`
- `apps/dungeon_dogfood/src/events.rs`
- `apps/dungeon_dogfood/src/main.rs`
- `.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-07-audio-foundation/00-specification-lock.md`
- `.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-07-audio-foundation/02-target-design.md`
- `.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-07-audio-foundation/shared/senior-engineer-guidance.md`
- `.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-07-audio-foundation/shared/implementation-notes.md`
- `.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-07-audio-foundation/shared/validation-matrix.md`
- `.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-07-audio-foundation/worker-directives/phase-03-event-bridge-dogfood-proof.md`
- `.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-07-audio-foundation/validation/phase-01-validation-report.md`
- `.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-07-audio-foundation/validation/phase-02-validation-report.md`
- `.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-07-audio-foundation/reports/phase-03-email.md`

## Commands Run

```bash
git status --short
```

Result: showed protected unrelated local state in `.idea/engine.iml` and `.reasonix/`, plus in-scope tracked changes. Did not show `audio_bridge.rs` or the WAV fixture because they are ignored.

```bash
cargo tree -p engine_events -e normal
```

Result: pass. `engine_events` has no normal dependencies, including no `audio` dependency.

```bash
cargo tree -p dungeon_dogfood -e normal | rg -n "^(dungeon_dogfood|├|└|│).*?(audio|engine_events|renderer|rodio)|engine_events|audio v|rodio"
```

Result: pass for dependency direction. `dungeon_dogfood` directly depends on `audio`; `engine_events` appears through `renderer`, not through `audio`.

```bash
cargo test -p engine_events
```

Result: pass. 7 passed.

```bash
cargo test -p audio
```

Result: pass. 7 passed, 1 ignored manual device smoke. No audio device was opened.

```bash
cargo check -p dungeon_dogfood
```

Result: pass with existing renderer and dogfood dead-code warning noise.

```bash
cargo test -p dungeon_dogfood
```

Result: blocked before dogfood tests ran by existing renderer test-profile compile errors in `src/renderer/src/data/assimp_util.rs` against current `russimp_sys` `aiScene`/`aiNode` bindings. Observed missing fields include `mRootNode`, `mNumMaterials`, `mMaterials`, `mNumTextures`, `mTextures`, `mNumMeshes`, `mMeshes`, `mTransformation`, `mName`, `mNumChildren`, and `mChildren`. This is not a Phase 03 failure by itself.

```bash
rg -n "AudioEngine::new|play_with_options|OutputStream|Sink|AudioClip::load|\.probe\(" apps/dungeon_dogfood/src/main.rs apps/dungeon_dogfood/src/content.rs apps/dungeon_dogfood/src/events.rs apps/dungeon_dogfood/src/audio_bridge.rs
```

Result: pass for gate placement. Matches were only in `apps/dungeon_dogfood/src/audio_bridge.rs`, with `AudioClip::load`, `probe`, `AudioEngine::new`, and `play_with_options` behind the smoke gate.

```bash
file apps/dungeon_dogfood/assets/audio/startup_ping.wav
od -An -tx1 -N80 apps/dungeon_dogfood/assets/audio/startup_ping.wav
```

Result: fixture exists locally and is a RIFF/WAVE PCM file: 16-bit mono, 8000 Hz, 116 bytes. `xxd` was unavailable, so `od` was used for header inspection.

```bash
git diff --check -- Cargo.lock apps/dungeon_dogfood/Cargo.toml apps/dungeon_dogfood/assets/content_pack.toml apps/dungeon_dogfood/assets/content_manifest.md apps/dungeon_dogfood/src/audio_bridge.rs apps/dungeon_dogfood/src/content.rs apps/dungeon_dogfood/src/events.rs apps/dungeon_dogfood/src/main.rs .internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-07-audio-foundation/reports/phase-03-email.md
```

Result: pass.

```bash
git ls-files --stage -- apps/dungeon_dogfood/src/audio_bridge.rs apps/dungeon_dogfood/assets/audio/startup_ping.wav
git check-ignore -v apps/dungeon_dogfood/src/audio_bridge.rs apps/dungeon_dogfood/assets/audio/startup_ping.wav
git show HEAD:apps/dungeon_dogfood/src/audio_bridge.rs
git show HEAD:apps/dungeon_dogfood/assets/audio/startup_ping.wav
git status --short --ignored -- apps/dungeon_dogfood/src/audio_bridge.rs apps/dungeon_dogfood/assets/audio/startup_ping.wav
```

Result: fail. Both files exist locally but are not tracked in `HEAD`, are ignored by `.gitignore:5` (`apps/*`), and show as ignored (`!!`). `git show HEAD:...` reports both paths are not in `HEAD`.

Remediation revalidation:

```bash
git status --short --ignored -- Cargo.lock apps/dungeon_dogfood/Cargo.toml apps/dungeon_dogfood/assets/content_pack.toml apps/dungeon_dogfood/assets/content_manifest.md apps/dungeon_dogfood/assets/audio/startup_ping.wav apps/dungeon_dogfood/src/audio_bridge.rs apps/dungeon_dogfood/src/content.rs apps/dungeon_dogfood/src/events.rs apps/dungeon_dogfood/src/main.rs .internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-07-audio-foundation/reports/phase-03-email.md .internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-07-audio-foundation/validation/phase-03-validation-report.md
```

Result: pass. Expected Phase 03 artifact set is staged, including `A apps/dungeon_dogfood/src/audio_bridge.rs` and `A apps/dungeon_dogfood/assets/audio/startup_ping.wav`.

```bash
git diff --cached --name-status -- Cargo.lock apps/dungeon_dogfood/Cargo.toml apps/dungeon_dogfood/assets/content_pack.toml apps/dungeon_dogfood/assets/content_manifest.md apps/dungeon_dogfood/assets/audio/startup_ping.wav apps/dungeon_dogfood/src/audio_bridge.rs apps/dungeon_dogfood/src/content.rs apps/dungeon_dogfood/src/events.rs apps/dungeon_dogfood/src/main.rs .internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-07-audio-foundation/reports/phase-03-email.md .internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-07-audio-foundation/validation/phase-03-validation-report.md
```

Result: pass. Cached diff includes `A apps/dungeon_dogfood/src/audio_bridge.rs`, `A apps/dungeon_dogfood/assets/audio/startup_ping.wav`, the expected Cargo/content/source/report changes, and this validation report.

```bash
git ls-files --stage -- apps/dungeon_dogfood/src/audio_bridge.rs apps/dungeon_dogfood/assets/audio/startup_ping.wav .internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-07-audio-foundation/validation/phase-03-validation-report.md
```

Result: pass. The staged index contains blob entries for both force-added ignored files.

```bash
git diff --check --cached
```

Result: pass.

```bash
cargo check -p dungeon_dogfood
```

Result: pass with existing renderer and dogfood dead-code warning noise. No audio device smoke was run.

## Findings

### Resolved: new required Phase 03 files are force-added in the staged index

`apps/dungeon_dogfood/src/main.rs:1` declares `mod audio_bridge;`, and `apps/dungeon_dogfood/src/main.rs:156` calls `audio_bridge::run_startup_audio_probe(...)`. The required implementation file exists locally at `apps/dungeon_dogfood/src/audio_bridge.rs`, but it is not tracked and is ignored by `.gitignore:5` (`apps/*`). The audio fixture referenced by `apps/dungeon_dogfood/assets/content_pack.toml:42` and documented in `apps/dungeon_dogfood/assets/content_manifest.md:25` has the same problem: `apps/dungeon_dogfood/assets/audio/startup_ping.wav` exists locally, is not in `HEAD`, and is ignored by `.gitignore:5`.

This means the local worktree can pass `cargo check -p dungeon_dogfood`, but the Phase 03 artifact set is not reproducible from tracked changes. A commit containing only the visible tracked diff would leave `main.rs` pointing at a missing module and the content pack pointing at a missing WAV. This is a Phase 03 Git/artifact completeness failure.

Classification: `code_defect` / Git workflow gap.

Remediation status: fixed. The main thread force-added the ignored required files into the staged index rather than changing `.gitignore`. Revalidation confirmed both files are staged as additions, `git diff --check --cached` passes, and `cargo check -p dungeon_dogfood` still passes. This finding is no longer blocking.

## Criteria Results

| Criterion | Result | Evidence |
|---|---:|---|
| Existing engine `AudioEvent` vocabulary is used | PASS | `src/events/src/lib.rs` already defines `AudioEvent::{ClipStarted, ClipStopped, ClipFinished, ClipFailed}`. `apps/dungeon_dogfood/src/audio_bridge.rs:172` maps runtime outcomes to those variants. |
| `engine_events` must not depend on `audio` | PASS | `src/events/Cargo.toml` has no dependencies; `cargo tree -p engine_events -e normal` shows only `engine_events`. |
| Dogfood app may depend on `audio` at consumer boundary | PASS | `apps/dungeon_dogfood/Cargo.toml` adds `audio`; dependency scan shows this is the app boundary. |
| Packaged/dogfood audio reference exists with clear internal/generated fixture provenance | PASS | Metadata exists, provenance text in `content_manifest.md` is clear, and `startup_ping.wav` is now staged as a force-added ignored file. |
| Default dogfood startup must not open an audio device | PASS | Static scan finds `AudioEngine::new` only in `audio_bridge.rs`, after `if !device_smoke_enabled { return report; }`. No runtime default device smoke was run. |
| Default startup should not load/probe/play the clip unless gated | PASS | `AudioClip::load`, `.probe()`, and `play_with_options` occur only behind the smoke gate in `audio_bridge.rs`. Default startup parses content metadata and validates path existence through the existing content loader. |
| Opt-in path loads/probes clip and attempts playback only behind `--audio-smoke` or `DUNGEON_DOGFOOD_AUDIO_SMOKE=1` | PASS | `audio_smoke_requested_from` accepts the flag/env gate; load/probe/device playback code is after that gate. |
| Audio outcomes map to `EngineEvent::Audio(AudioEvent::*)` with durable clip IDs | PASS | `audio_event_for_outcome` wraps the configured durable string in `AudioClipId` and emits the existing variants; tests cover all four outcomes but are blocked from dogfood execution by the renderer test-profile issue. |
| Normal validation must not require a physical audio device | PASS | `cargo test -p audio` leaves manual device smoke ignored; `cargo check -p dungeon_dogfood` and static checks do not open devices. |
| No desktop screenshots/capture | PASS | No screenshots, captures, or visual validation were run. |
| Phase artifact set is complete and committable | PASS | `audio_bridge.rs` and `startup_ping.wav` remain ignored by `.gitignore`, but both are force-added and present in the staged index for the Phase 03 commit. |

## Residual Risks

- `cargo test -p dungeon_dogfood` remains blocked before dogfood tests run by pre-existing renderer test-profile compile errors in `src/renderer/src/data/assimp_util.rs` against current `russimp_sys` bindings. I classify this as a residual/blocker for dogfood tests, not a Phase 03 product failure, because `cargo check -p dungeon_dogfood`, `cargo test -p audio`, and `cargo test -p engine_events` pass and the errors are outside the Phase 03 editable target set.
- Device-backed playback was intentionally not attempted. Device smoke remains host-dependent and should only be run through the documented opt-in command.
- The default dogfood content loader validates that the configured audio file path exists. It does not load/decode/play the clip by default, but default startup is no longer fully independent of the fixture file's presence.

## Browser/Capture

Not applicable. This phase is non-visual. No desktop screenshots, no headless captures, and no audio-device smoke were used.
