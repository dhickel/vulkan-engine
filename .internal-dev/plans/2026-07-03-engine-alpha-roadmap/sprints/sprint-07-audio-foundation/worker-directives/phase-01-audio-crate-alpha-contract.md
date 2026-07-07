# Phase 01 Worker Directive: Audio Crate Alpha Contract

## Objective

Turn `src/audio` into a renderer-independent alpha audio facade with durable clip identity support, device-independent clip/decode validation, typed errors, and optional device-backed playback.

## User-Visible Outcome

Engine users can load or construct an audio clip, validate/probe clip data without speakers, and explicitly create a device-backed engine only when they want to attempt playback.

## Editable Targets

- `src/audio/Cargo.toml`
- `src/audio/src/lib.rs`
- Optional new modules under `src/audio/src/`
- Optional fixtures under `src/audio/tests/fixtures/` or generated test bytes if local pattern is cleaner
- `reports/phase-01-email.md`

## Forbidden Scope

- Do not edit package/scene validation, renderer, engine_pack, events, editor, or dogfood in this phase.
- Do not add dependencies from `audio` to renderer, Vulkan, windowing, editor, dogfood, or app crates.
- Do not make normal tests require `OutputStream::try_default()` or any physical audio device.
- Do not serialize or expose rodio runtime handles as durable authored identity.
- Do not touch `.idea/engine.iml` or `.reasonix/`.

## Supporting Docs To Read

- `00-specification-lock.md`
- `01-current-state-analysis.md`
- `02-target-design.md`
- `shared/senior-engineer-guidance.md`
- Top-level `AGENTS.md`

## Senior-Engineer Guidance

- Keep the facade small. A tested decode/probe path is more valuable than speculative mixer abstractions.
- Replace the current required `engine_initialization` behavior with ignored/gated/manual device smoke or a clearly optional API test.
- Use typed errors or stable error variants where practical so validation can assert decode/device/read failures without string-matching everything.
- Generate a tiny valid WAV in tests if adding binary fixtures would be noisy.
- Stop/drop playback handles in device smoke paths to avoid hangs or background-thread surprises.

## Ordered Implementation Steps

1. Audit `src/audio/src/lib.rs` and current tests.
2. Define the minimal alpha API shape for durable clip identity, clip bytes, optional metadata/probe, playback options, and playback errors.
3. Add or adjust clip constructors so bytes/path loading can be tested without a device.
4. Add decode/probe validation using rodio decoder or a standard parser path that does not open an output stream.
5. Keep `AudioEngine::new` or equivalent as explicit device-backed construction.
6. Convert the current device-dependent test into ignored/gated/manual smoke, or replace it with a no-device test and add a separate ignored device test.
7. Add tests for valid clip bytes, invalid bytes/decode failure, load error, volume clamping/control behavior if testable without a device, and device-smoke gating.
8. Run validation commands and record exact results in the implementation handoff.
9. Draft `reports/phase-01-email.md` with a short summary and device-smoke status.

## Acceptance Criteria

- `cargo test -p audio` passes on a host with no audio output device.
- `cargo check -p audio` passes.
- Audio crate remains renderer/window/editor/dogfood independent.
- Device-backed playback is still possible through an explicit API or ignored/manual smoke path.
- Errors distinguish at least read/decode/device/playback failures well enough for docs/tests.
- No normal test opens a default audio device.

## Negative Checks

- `rg -n "renderer|ash|vulkan|winit|imgui|dungeon_dogfood|editor" src/audio` must not show imports/dependencies except explanatory comments if unavoidable.
- No `Serialize`/`Deserialize` of rodio streams, sinks, or playback handles.
- No "spatial audio" claim in crate docs unless real API/tests implement it.

## Validation Commands

```bash
cargo fmt --check
cargo test -p audio
cargo check -p audio
cargo check
```

Optional dependency scan:

```bash
cargo tree -p audio
```

## Evidence Expectations

- Validator report path: `.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-07-audio-foundation/validation/phase-01-validation-report.md`
- Phase report draft path: `.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-07-audio-foundation/reports/phase-01-email.md`
- Implementation handoff must state whether any device smoke was skipped, blocked, or run.
- Do not mark `artifacts/validation-summary.json` as passed from implementation alone.

## Stop Conditions

- Stop if device-independent decoding cannot be tested without opening a device.
- Stop if a dependency cycle or renderer/window dependency is needed.
- Stop if rodio behavior forces broad architectural work beyond a facade hardening phase.

## Do Not Close Unless

- Audio tests are device-independent.
- Dependency hygiene has been checked.
- Device behavior is clearly optional and documented in the phase report.
- The phase is ready for validator review.
