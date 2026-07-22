# Safety Refactor Remediation Ledger

## 1. Purpose & Audience

This chapter is the human-readable companion to the authoritative finding ledger at
`tests/remediation_ledger.rs`. It is for contributors and reviewers who need a
concise summary of every finding addressed, deferred, or left partially resolved
by the 2026-07-22 engine safety-refactor stabilization sprint.

## 2. Where This Fits in Engine Flow

The ledger is a closeout artifact. It does not participate in runtime behavior.
It serves as the canonical cross-reference between:

- the original code-review synthesis (`.internal-dev/reviews/2026-07-22-comprehensive-code-review-refactor-target-synthesis.md`),
- the phased finding matrix (`.internal-dev/plans/engine-safety-refactor-stabilization/finding-matrix.md`),
- implementation commits across 10 phases, and
- the living specifications and decisions in `.internal-dev/specifications/`.

## 3. Key Concepts

- **21 entries**: 10 high-severity addressed (H-A1–H-A10), 10 medium-severity
  addressed (M-A1–M-A10), and 1 deferred (D-01).
- **Status taxonomy**: `Resolved` (code + tests + docs align), `PartiallyResolved`
  (core invariant fixed; remainder deferred with justification), `Deferred`
  (acknowledged but left for a future milestone).
- **Evidence requirements**: every `Resolved` entry must cite at least one fix
  commit and at least one focused test. `PartiallyResolved` entries must cite a
  `deferred_reason`. `Deferred` entries must cite a `deferred_reason`.

## 4. Finding Summary Table

| ID | Summary | Phase | Status |
|----|---------|-------|--------|
| H-A1 | Animation atomicity: fallible try_* methods, typed AnimationError, validation before mutation | 08 | Resolved |
| H-A2 | Input multi-binding: per-instance BindingInstanceId, aggregate action state | 09 | Resolved |
| H-A3 | Scripting isolation: per-evaluation state, thread-safe identity | 09 | Resolved |
| H-A4 | Retirement atomicity: texture/material fence-aware retirement, RetirementClass taxonomy | 04 | Resolved |
| H-A5 | Scene v2 format: directional/spot lights, collision/audio/prefab metadata, staged save | 08 | Resolved |
| H-A6 | Shadow persistence: shadow owner survives round-trip, CSM cascade config | 08 | Resolved |
| H-A7 | Transactional Assimp: RAII scene ownership, null guards, typed errors | 05 | Resolved |
| H-A8 | Image lifetime: view-before-image teardown, sampler ownership, prefilter batching | 06 | Resolved |
| H-A9 | Pipeline ownership: staged construction, failure-atomic rollback, dedup handle teardown | 07 | Resolved |
| H-A10 | Capture truthfulness: serde_json output, NaN→null, recording init atomicity | 09 | Resolved |
| M-A1 | CSM fallbacks: feature-gated compilation, dead-code cleanup, legacy preservation | 08 | Resolved |
| M-A2 | Race-free scene save: StagedSceneFile, write→fsync→rename, partial-write rejection | 10 | Resolved |
| M-A3 | O_NOFOLLOW staging: symlink_metadata, root containment, cycle detection | 10 | Resolved |
| M-A4 | UTF-8 rejection: serializer-backed output, ad-hoc escape removal, boundary validation | 10 | Resolved |
| M-A5 | CLI schema unification: single declarative parser, equals-form, duplicate rejection | 10 | PartiallyResolved |
| M-A6 | Asset identity normalization: canonical project-relative keys, version consolidation | 10 | Resolved |
| M-A7 | Fail-closed pack publication: staging→rename, ExistingTarget error, rollback journal | 10 | Resolved |
| M-A8 | Hook failure observation: HookReport per-frame, typed registration errors | 09 | Resolved |
| M-A9 | Descriptor reset safety: CompletedFrameSlot token, epoch monotonicity, quarantine | 02 | Resolved |
| M-A10 | Frame indexing safety: checked_add counters, NoActiveReservation sentinel, safe indexing | 02 | Resolved |
| D-01 | God-module split: deferred past safety-refactor sprint | Deferred | Deferred |

## 5. Cross-Reference Map

| Finding ID | Original Matrix Finding | Related Decisions | Related Specs |
|------------|------------------------|-------------------|---------------|
| H-A1 | H7 (animation), H8 (command) | — | API-20260722-08 |
| H-A2 | H12 (input actions) | DECISION-20260707-01 | API-20260707-01 |
| H-A3 | H11 (scripting) | — | — |
| H-A4 | C4 (fence-aware retirement) | DECISION-20260725-15 | API-20260722-06, ARCH-20260725-10 |
| H-A5 | C6 (scene persistence) | — | API-20260722-08 |
| H-A6 | C6 subclaim | DECISION-20260718-01 | API-20260718-03 |
| H-A7 | C5, H9 (Assimp safety) | DECISION-20260725-10 | ARCH-20260725-07 |
| H-A8 | H1, H2 (image/sync) | DECISION-20260718-03 | ARCH-20260718-02 |
| H-A9 | H3 (pipeline), H4 (cache) | — | ARCH-20260725-04 |
| H-A10 | M5 (capture/debug) | — | API-20260703-02 |
| M-A1 | review-identified | DECISION-20260718-01, DECISION-20260725-03 | ARCH-20260718-01 |
| M-A2 | C6 subclaim | — | — |
| M-A3 | C8 (symlink escape) | — | — |
| M-A4 | M6 subclaim | — | — |
| M-A5 | M1 (CLI parsing) | — | API-20260722-01 |
| M-A6 | M6 (asset identity) | — | — |
| M-A7 | C7 (engine_pack) | DECISION-20260722-06 | — |
| M-A8 | M4 (structured errors) | DECISION-20260722-07 | API-20260722-07 |
| M-A9 | reviewed-original | DECISION-20260725-13 | ARCH-20260725-08 |
| M-A10 | C3 (frame indexing) | — | ARCH-20260718-02 |
| D-01 | M2 (god modules) | DECISION-20260722-09 | — |

## 6. Gotchas & Failure Modes

- Do not use this ledger's status as a substitute for reading the actual
  implementation. A `Resolved` status means the invariant is enforced in code
  and verified by tests, not that the implementation cannot regress.
- The `fix_commits` field records the commit that introduced the remediation.
  Later commits may have refined or extended the fix. The ledger is a snapshot
  at Phase 10 closeout.
- `test_evidence` entries are descriptive labels that match test function names
  or module paths. They are not automated links — use `rg` to locate the
  corresponding source.

## 7. Debugging Playbook

- To verify a finding's status: read the cited fix commit(s), run the cited
  tests, and confirm the behavior described in the summary.
- To audit a deferred finding: check whether the `deferred_reason` still holds
  and whether any dependent work has been completed since closeout.
- To add a new finding: create a new row following the H-A/M-A/D- prefix
  convention, add it to `tests/remediation_ledger.rs`, update this chapter's
  summary table and cross-reference map, and record the change in a changelog.

## 8. Cross-Module Links

- Canonical ledger: `tests/remediation_ledger.rs`
- Finding matrix: `.internal-dev/plans/engine-safety-refactor-stabilization/finding-matrix.md`
- Review synthesis: `.internal-dev/reviews/2026-07-22-comprehensive-code-review-refactor-target-synthesis.md`
- Phase evidence packets: `.internal-dev/plans/engine-safety-refactor-stabilization/evidence/`

## 9. Standard References

- `.internal-dev/specifications/decisions.md` — all active decisions
- `.internal-dev/specifications/architecture.md` — architecture contracts
- `.internal-dev/specifications/api.md` — API contracts

## 10. See Also

- `docs/internal/00-index.md` — internal docs index (includes this chapter)
- `docs/internal/03-asset-lifecycle-and-io.md` — retirement and handle lifecycle
- `docs/internal/05-vulkan-sync-and-frame-lifecycle.md` — frame safety invariants
- `docs/internal/07-rendergraph-dependencies-and-aliasing.md` — pass ordering contracts
