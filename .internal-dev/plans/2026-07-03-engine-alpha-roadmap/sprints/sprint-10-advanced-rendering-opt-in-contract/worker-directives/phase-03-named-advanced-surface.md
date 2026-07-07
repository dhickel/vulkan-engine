# Phase 03 Worker Directive: Minimal Named Advanced Surface Or Deliberate Defer

## Objective

Either add one minimal, named, feature-gated advanced rendering extension surface that avoids raw backend handles, or record a deliberate defer explaining why the current synchronization/resource contracts are not ready.

## User-Visible Outcome

Advanced users get either a small documented opt-in surface they can compile against or an honest alpha limitation explaining which advanced rendering controls remain unavailable and why.

## Editable Targets

Only edit targets justified by Phase 01 and Phase 02:

- `src/renderer/src/api/advanced.rs`
- `src/renderer/src/api/hooks.rs`
- `src/renderer/src/api/renderer.rs`
- `src/renderer/src/api/mod.rs`
- `src/renderer/src/lib.rs`
- `src/renderer/src/rendergraph/mod.rs` only if adding constrained feature-gated pass registration and Phase 01 explicitly approves.
- Focused renderer tests/examples needed for the new surface.
- Relevant docs from Phase 02.
- Sprint 10 reports/evidence files.

## Forbidden Scope

- No raw command buffers, raw `ash` handles, descriptor sets, allocator handles, queues, fences, or swapchain image handles as a new user-facing API.
- No default-feature dependency on advanced APIs.
- No rendergraph scheduler rewrite.
- No broad material/shader pipeline overhaul.
- No desktop screenshot validation.

## Supporting Docs To Read

- Phase 01 audit report.
- Phase 02 validation report.
- `02-target-design.md`
- `shared/senior-engineer-guidance.md`
- `docs/internal/07-rendergraph-dependencies-and-aliasing.md`
- `.internal-dev/skills/engine-headless-capture-validation/SKILL.md` if capture/readback behavior changes.

## Senior Engineer Guidance

- The best Phase 03 result may be a deliberate defer. Do not force a new API if it would normalize unsafe backend ownership.
- If adding a surface, prefer read-only descriptors or validated registration metadata.
- Keep all advanced additions behind `advanced-interop` unless they are truly safe default observation APIs.
- If touching capture/readback, require headless draw capture proof.
- If touching rendergraph pass order, assume high blast radius and stop unless the contract is narrower than a full custom pass API.

## Ordered Steps

1. Read Phase 01/02 outputs and decide: implement minimal named surface or defer.
2. If deferring, write `reports/phase-03-advanced-surface-defer.md` with exact blockers and future contract requirements, update docs/evidence, and run compile checks.
3. If implementing, define the minimal public type(s) and docs first. Examples of acceptable shape:
   - read-only frame/debug resource descriptor with no raw handles;
   - advanced diagnostic snapshot that reports pass names/timings/resource availability;
   - validated named extension registration that cannot record GPU commands.
4. Add focused tests for construction, export, feature-gate behavior, and error/validation cases.
5. Run default and feature-enabled compile checks.
6. Run runtime smoke if API/runtime behavior changed.
7. Run headless draw capture only if visible renderer or capture/readback behavior changed.

## Acceptance Criteria

- Either a minimal named advanced surface is implemented and validated, or a defer report explains why no safe surface should ship in Sprint 10.
- Any implemented advanced surface is feature-gated or demonstrably safe as default observation.
- Tests cover changed behavior.
- Runtime/capture evidence exists if required by behavior changes.
- Docs match the implemented/deferred state.

## Negative Checks

- No `renderer::prelude` advanced export.
- No new raw backend handle fields.
- No claim that custom rendergraph pass registration is stable unless resource/order validation exists.
- No false visual proof from compile checks only.

## Validation Commands

Minimum:

```sh
cargo check -p renderer
cargo check -p renderer --examples
cargo check -p renderer --features advanced-interop
cargo check -p renderer --examples --features advanced-interop
```

Focused tests as appropriate:

```sh
cargo test -p renderer <focused_filter>
```

Runtime smoke if behavior changed:

```sh
RUST_LOG=debug timeout --signal=INT 60s cargo run -p renderer --example api_test -- --record_debug=10 --record_debug_interval=50 --record_debug_path=.internal-dev/debug_reports/sprint-10-api-test-timing.jsonl
```

Conditional capture:

```sh
RUST_LOG=info timeout --signal=INT 60s cargo run -p renderer --example api_test -- --headless --capture_target draw --capture_frames=3 --capture_frame_start=5 --capture_frame_interval=5 --capture_dir .internal-dev/captures/sprint-10-advanced-rendering-opt-in-contract/headless-draw
```

## Stop Conditions

- Stop and ask the main thread if implementation requires exposing raw handles.
- Stop if feature-gated code cannot compile without changing broad renderer architecture.
- Stop if capture/readback validation is required but headless capture cannot initialize.

## Evidence Expectations

- Implemented path: tests/commands/capture evidence in `validation/phase-03-validation-report.md`.
- Deferred path: `reports/phase-03-advanced-surface-defer.md` plus compile/docs validation.
- Summary update in `artifacts/validation-summary.json`.

## Do Not Close Unless

- The implement/defer decision is explicit.
- Docs and evidence agree with the decision.
- Required default and feature-gated checks are run or blockers are exact.
