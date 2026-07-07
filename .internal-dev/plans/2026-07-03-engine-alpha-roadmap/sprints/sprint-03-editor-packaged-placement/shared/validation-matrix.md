# Validation Matrix

| Gate | Phase | Evidence | Pass Criteria |
|---|---:|---|---|
| Editor placement state | 01 | Unit/focused tests and phase report | Selecting an asset, starting placement, canceling placement, and confirming placement produce expected state/status without stale selection. |
| Command-created node identity | 01 | Renderer/editor tests | Placed root has stable node ID, asset reference, display name, tags, and transform. |
| Undo/redo selection | 01 | Focused tests | Undo clears or remaps invalid selection; redo reselects created node when command result exposes it. |
| Save/reload durability | 02 | Temp scene fixture/test | Saved scene copy preserves durable asset IDs, stable node IDs, tags, transforms, and material override metadata where covered. |
| No runtime handle persistence | 02 | JSON assertions and `engine_pack validate-scene` | Saved scene JSON contains stable strings, not runtime slot/generation handles. |
| Project/scene validation | 02 | CLI command logs | `validate-project` and `validate-scene` pass for sample project and saved scene copy. |
| Visual package placement proof | 03 | Capture PNG/JSON and validation report | Capture visibly shows placed package-backed model/wall assets from the approved data path. |
| Docs and evidence consistency | 04 | Docs diff, stale sweep, validation summary | Docs match implementation and final evidence index status is conservative and internally consistent. |
| Final quality review | 04 | `validation/final-quality-review.md` | Validator reconciles code, tests, docs, capture evidence, phase reports, and residual risks. |

## Required Validator Checks

- Architecture fit: editor composes renderer scene/asset APIs and does not fork persistence.
- Contract fit: durable IDs stay durable; runtime handles stay runtime only.
- Regression risk: sample project remains valid and canonical scene is not accidentally mutated.
- Test quality: tests assert behavior, not merely current implementation details.
- Evidence quality: visual proof includes actual capture artifact inspection.
- Closeout quality: `artifacts/validation-summary.json` does not claim final status while any required gate is pending.

## Remediation Routing

- `code_defect`: fresh scoped repair worker for the failed phase, unless the miss is a trivial one-place typo/import.
- `docs_or_evidence_defect`: fresh scoped repair worker for docs/evidence, unless the validator can safely fix an obvious stale reference.
- `browser_harness_defect` or capture harness defect: repair the harness/evidence path first; change product code only after evidence shows a product bug.
- `plan_defect`: return to planning before more implementation.
- `validator_error`: fix checklist or use a fresh validator.
