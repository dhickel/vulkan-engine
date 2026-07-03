# Sprint 07: Audio Foundation

Status: closed

## Objective

Build the alpha audio foundation so packaged audio clips can be represented with durable IDs, validated through project/package/scene flows, loaded through a small audio facade, and played through an opt-in sample or dogfood path when a device is available.

## User-Visible Outcome

An engine user can add an audio clip to a package, validate the project, reference the clip from a sample/app contract, and run a documented optional playback proof without confusing device availability for guaranteed platform support.

## In Scope

- Device-independent audio crate API and tests.
- Package/scene/app audio metadata validation using durable IDs.
- `engine_pack` validation coverage for audio metadata.
- Audio event bridge using existing or minimally extended `engine_events` vocabulary.
- Narrow sample/dogfood proof or explicit debt artifact.
- Public/internal docs and conservative validation evidence.

## Out Of Scope

- Production mixer, spatialization, streaming, DSP, occlusion, device matrix, editor placement UI, or broad dogfood gameplay rewrite.

## Target Surfaces

- Code: `src/audio`, `src/events` only if needed, `src/renderer/src/data/asset_registry.rs`, `src/renderer/src/api/scene.rs`, `tools/engine_pack`, `apps/dungeon_dogfood` or a sample path.
- Docs: `docs/api`, `docs/internal`, package/scene docs touched by prior sprints.
- `.internal-dev` artifacts: validation reports, reports/email drafts, final quality review, `artifacts/validation-summary.json`.

## Assumptions

- Current required audio test is device-dependent and should be gated or replaced.
- Existing event vocabulary already has audio event variants.
- Existing validation source of truth is renderer validation surfaced by `engine_pack`.

## Risks And Gotchas

- Default audio devices are not available on many validation hosts.
- Docs may overclaim spatial audio because the current audio crate comment mentions it.
- Adding dogfood audio should not change normal launch behavior or require speakers.
- Package/scene contracts must not serialize runtime playback handles.

## Acceptance Criteria

- Device-independent audio API tests pass.
- Audio metadata validation exists in package/project/scene flow.
- Optional runtime smoke or dogfood/sample proof is recorded honestly.
- Audio events are bridged where appropriate.
- Docs and evidence report device-dependent limits clearly.

## Negative Criteria

- No mandatory device in CI-like tests.
- No hidden thread leaks or hanging playback tests.
- No broad gameplay rewrite.
- No false editor placement claim.
- No desktop screenshot capture.

## Validation Plan

- Compile/test: see `shared/validation-matrix.md`.
- Runtime smoke: optional and gated; record skipped/blocked/pass status.
- Visual/capture proof: not required unless visible renderer/editor behavior changes.
- Docs/process checks: stale-reference sweep, final quality review, conservative evidence index.

## Advanced-Planner Handoff

Use the worker directives in `worker-directives/`:

- Phase 01: audio crate alpha contract.
- Phase 02: package/scene audio metadata.
- Phase 03: event bridge and sample/dogfood proof.
- Phase 04: docs, validation, closeout preparation.

## Closeout Checklist

- Validation evidence recorded.
- Known residuals tracked.
- Changelog recorded.
- Sprint tracker updated by main thread after validation/closeout.

## Closeout Status

Sprint 07 is fully validated with recorded residuals in `artifacts/validation-summary.json`.

Accepted residuals:

- Device smoke was skipped and remains optional/host-dependent.
- `cargo test -p dungeon_dogfood` remains blocked before dogfood tests execute by the existing renderer test-profile `russimp_sys` binding issue in `src/renderer/src/data/assimp_util.rs`.
- Root-runtime/editor audio playback, editor audio placement, production mixer/spatialization/streaming, and platform/device support matrix remain deferred.
