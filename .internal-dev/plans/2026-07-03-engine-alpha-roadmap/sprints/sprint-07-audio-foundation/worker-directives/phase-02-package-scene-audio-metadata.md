# Phase 02 Worker Directive: Package/Scene Audio Metadata

## Objective

Add durable audio clip metadata to package and scene/app contracts, and surface validation through existing renderer validation APIs and `engine_pack`.

## User-Visible Outcome

Authors can declare packaged audio clips and validate durable audio references before runtime, without requiring an audio device.

## Editable Targets

- `src/renderer/src/data/asset_registry.rs`
- `src/renderer/src/api/scene.rs`
- `tools/engine_pack/src/main.rs` only if command help, scanning, or add-asset behavior needs adjustment
- `tools/engine_pack/tests/cli_validation.rs`
- Existing CLI fixtures under `tools/engine_pack/fixtures/` if needed
- Existing docs only if schema comments are colocated; broader docs belong to Phase 04
- `reports/phase-02-email.md`

## Forbidden Scope

- Do not change audio playback behavior in this phase.
- Do not make renderer depend on `audio` unless there is a narrow type-only reason and no cycle; prefer local schema validation.
- Do not implement editor UI placement or claim editor audio authoring.
- Do not make package/scene validation open an audio output device.
- Do not touch `.idea/engine.iml` or `.reasonix/`.

## Supporting Docs To Read

- `00-specification-lock.md`
- `02-target-design.md`
- `shared/senior-engineer-guidance.md`
- `src/renderer/AGENTS.md`
- `src/renderer/src/data/AGENTS.md`
- Existing package/scene validation tests in `src/renderer/src/data/asset_registry.rs` and `src/renderer/src/api/scene.rs`
- Existing CLI tests in `tools/engine_pack/tests/cli_validation.rs`

## Experience Contract

This phase has no UI surface. The authoring experience is file/schema validation:

- valid package files without audio metadata still validate;
- valid package files with audio clips validate through `engine_pack`;
- invalid audio IDs/formats/runtime-handle fields fail with stable diagnostics;
- scene/app audio references validate against known package asset IDs when project context is supplied.

## Senior-Engineer Guidance

- Extend existing validation paths instead of adding CLI-only checks.
- Add an `Audio` asset kind or equivalent only if it does not break existing manifests.
- Keep metadata minimal: clip kind/format/path/usage/volume are acceptable; avoid unverified duration/sample-rate claims unless parsed.
- Runtime-handle rejection must recurse into audio metadata/references.
- If scene persistence cannot preserve audio references yet, validate a clearly named scene-level metadata shape and document that runtime loading is Phase 03.

## Ordered Implementation Steps

1. Inspect current package and scene validation tests for collision metadata patterns.
2. Add an audio package asset kind and metadata validation shape.
3. Validate supported declared formats, positive finite optional volume/default gain if present, non-empty durable IDs, and runtime handle-shaped metadata.
4. Ensure package source-file checks work for audio asset paths.
5. Add scene/app audio reference validation with durable clip IDs and known-asset checks where project context is available.
6. Ensure old package/scene files without audio metadata remain valid.
7. Add renderer tests for valid package audio, invalid package audio, valid scene audio reference, unknown clip reference, and runtime-handle rejection.
8. Add `engine_pack` CLI tests proving audio metadata failures are reported.
9. If `scan-assets` or `add-asset` supports kind lists, update it to handle audio files consistently.
10. Run validation commands and draft `reports/phase-02-email.md`.

## Acceptance Criteria

- Audio assets can be represented in package manifests with durable IDs.
- Package validation rejects invalid audio metadata and handle-shaped identities.
- Scene/app audio references validate and reject blank/unknown/runtime-handle IDs.
- `engine_pack` reports audio validation failures through existing command flows.
- Backward-compatible files without audio data still validate.

## Negative Checks

- No rodio/device use in package or scene validation.
- No runtime playback handles in serialized examples or tests.
- No editor UI placement claim.
- No duplicate validation fork between renderer and engine_pack.

## Validation Commands

```bash
cargo fmt --check
cargo test -p renderer
cargo test -p engine_pack
cargo check -p engine_pack
cargo check
```

## Evidence Expectations

- Validator report path: `.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-07-audio-foundation/validation/phase-02-validation-report.md`
- Phase report draft path: `.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-07-audio-foundation/reports/phase-02-email.md`
- Implementation handoff should name the exact audio schema accepted and diagnostic codes added/changed.
- `artifacts/validation-summary.json` remains non-final until validator reconciliation.

## Stop Conditions

- Stop if schema changes require a format-version bump with compatibility consequences.
- Stop if renderer must depend on `audio` in a way that creates a cycle or heavy runtime dependency.
- Stop if scene audio persistence requires editor UI work to avoid misleading users; return for plan revision.

## Do Not Close Unless

- Positive and negative schema tests exist.
- CLI validation has at least one audio-specific failure test.
- Old files remain accepted.
- The phase is ready for validator review.
