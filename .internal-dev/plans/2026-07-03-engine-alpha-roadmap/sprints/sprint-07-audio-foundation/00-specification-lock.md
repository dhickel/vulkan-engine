# Specification Lock

## Locked Objective

Build the Sprint 07 alpha audio foundation so packaged audio clips have durable package/scene/app contracts, can be loaded and validated without a physical output device in tests, and can be played through an opt-in sample or dogfood path when an output device is available.

## Acceptance Criteria

- `src/audio` remains renderer-independent and does not depend on Vulkan, windowing, editor, dogfood, or renderer internals.
- Core audio API tests are device-independent: clip identity, supported format metadata, decode/probe paths, error handling, and playback facade behavior that can be tested without opening the default audio device.
- Device-backed playback exists behind an explicit opt-in runtime path, feature/config gate, ignored test, or manual smoke command; validation reports must say whether it was run and whether a device was present.
- Audio clip identity uses stable durable IDs such as `AudioClipId`/string IDs in package, scene, and event contracts; runtime sinks/handles are not serialized or treated as authored identity.
- Package manifests can represent audio assets with validation for supported kind/format/path/metadata and can reject invalid durable IDs, unknown formats when format is declared, missing source files when file checks are enabled, and runtime handle-shaped metadata.
- Scene or app-level audio references can round-trip and validate as durable clip references without requiring editor placement UI.
- `engine_pack` validation surfaces audio package/scene/project failures through existing CLI validation flows.
- `engine_events::AudioEvent` can represent audio playback status from the audio subsystem or dogfood/sample bridge without making `engine_events` depend on `audio`.
- A sample or dogfood path demonstrates how an app references a packaged audio clip and attempts playback only when explicitly requested and safe for the host.
- Public and internal docs describe the implemented alpha contract and clearly state device-dependent limits, unsupported mixing/spatialization guarantees, and deferred editor authoring.

## Validation Criteria

- Required phase reports:
  - `validation/phase-01-validation-report.md`
  - `validation/phase-02-validation-report.md`
  - `validation/phase-03-validation-report.md`
  - `validation/phase-04-validation-report.md`
  - `validation/final-quality-review.md`
- Canonical evidence index: `artifacts/validation-summary.json`.
- `artifacts/validation-summary.json` must remain conservative until every required validator passes and any optional device smoke status is honestly recorded.
- Required core checks include `cargo test -p audio`, `cargo check -p audio`, `cargo test -p renderer`, `cargo test -p engine_pack`, and workspace checks listed in `shared/validation-matrix.md`.
- Device playback smoke is optional/gated and must not be required for CI-like validation.
- True headless draw capture using `--headless --capture_target draw` is required only if visible renderer/editor behavior changes; desktop screenshots are not acceptable evidence and should not be used for this audio sprint.

## Negative Criteria

- No mandatory default-device open in normal unit tests.
- No hidden dependency from `audio` to renderer, Vulkan, windowing, editor, or dogfood.
- No serialization of `rodio::Sink`, stream handles, runtime playback handles, slot/generation handles, or path-only identity as the durable audio contract.
- No claim of production mixer, spatialization, streaming, DSP, occlusion, distance attenuation, or platform-complete device support unless actually implemented and validated.
- No claim of editor audio placement unless editor placement UI is implemented and validated.
- No broad gameplay rewrite of dungeon dogfood.
- No desktop screenshot capture for audio validation.
- No stale plan/evidence language claiming implementation is complete during planning.

## Non-Goals

- Full audio mixer, bus graph, spatial audio, environmental audio, or streaming asset pipeline.
- Editor UI for placing audio emitters.
- Cross-platform audio device matrix.
- Runtime hot-reload of audio devices or assets.
- Timeline/cutscene/audio sequencing system.
- Production-grade error recovery for device loss.

## Constraints

- Branch: `sprint/alpha-07-audio-foundation`.
- Protected local state: do not touch `.idea/engine.iml` or `.reasonix/`.
- Main thread commits, pushes, and sends emails/reports after each validated phase; workers and validators only write report drafts/paths.
- `.internal-dev` is the durable planning/evidence store. Do not read it broadly outside named artifacts.
- Preserve Sprint 05 event ownership: `engine_events` owns event vocabulary and must not depend on `audio`.
- Preserve Sprint 06 metadata pattern: durable IDs and typed validation; no runtime handle identity in authored data.

## Assumptions

- The current `audio` crate is a thin `rodio` wrapper and currently has a device-dependent initialization test that should be made optional or replaced with device-independent coverage.
- `engine_events` already defines `AudioClipId` and `AudioEvent` variants that can be reused or extended conservatively.
- Package and scene validation already live primarily in renderer data/API modules and are surfaced by `tools/engine_pack`; Sprint 07 should extend those paths instead of duplicating validation in the CLI.
- Dogfood proof may be a narrow opt-in path or a debt artifact if adding audio risks destabilizing the app.
- Existing renderer warning noise may appear during validation; reports must distinguish existing noise from new warnings.

## User-Decision Gates

- If adding `audio` as a dependency to dogfood or the root runtime causes broad runtime ownership or threading changes, stop and ask whether to defer dogfood proof to a sample-only path.
- If audio metadata requires a package/scene format-version bump with backward compatibility impact, stop for approval before locking the format change.
- If device-backed playback cannot be attempted safely on the current host, record `device_smoke_status` as blocked/skipped with reason rather than failing core validation.
- If visible renderer/editor behavior changes unexpectedly, add headless draw capture validation before closeout.

## Stop Rules

- Stop on dependency cycles involving `audio`, `renderer`, `engine_events`, `dungeon_dogfood`, or `engine_pack`.
- Stop if any normal test requires a physical audio output device.
- Stop if packaging/scene audio identity drifts to path-only identity or runtime handles.
- Stop if a required validator model/tool is unavailable and record `TOOLING_CONSTRAINT`.
