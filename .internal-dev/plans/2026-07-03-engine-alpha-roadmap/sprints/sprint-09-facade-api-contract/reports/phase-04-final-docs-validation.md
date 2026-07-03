# Phase 04 Final Docs Validation Report

Date: 2026-07-03
Branch: `sprint/alpha-09-facade-api-contract`

## Objective

Reconcile Sprint 09 public API docs, phase evidence, stale-reference scans, and validation summary before final quality review.

## Files Changed

| Path | Change |
| --- | --- |
| `docs/api/00-index.md` | Replaced prior sprint-specific headless capture output path with a neutral runtime-launcher capture path. |
| `docs/api/01-student-quickstart.md` | Replaced prior sprint-specific headless capture output path with a neutral runtime-launcher capture path. |
| `docs/api/07-engine-arguments.md` | Replaced prior sprint-specific headless capture output path with a neutral runtime-launcher capture path. |
| `docs/api/09-editor-asset-browser-and-wall-chunks.md` | Replaced prior sprint-specific editor capture output path with a neutral current-use editor capture path. |
| `docs/api/11-runtime-project-launcher.md` | Replaced prior sprint-specific headless capture output path with a neutral runtime-launcher capture path. |
| `.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-09-facade-api-contract/artifacts/validation-summary.json` | Updated after this report with Phase 04 command evidence and closeout state. |

## Evidence Reconciliation

The Phase 04 stale scan initially found public API examples that wrote new capture output into old sprint-specific directories. Those paths were changed to neutral validation-output paths:

- `.internal-dev/captures/runtime-launcher/headless-draw`
- `.internal-dev/captures/editor-packaged-placement/headless-draw`

Remaining prior-sprint references in `docs/api/09-editor-asset-browser-and-wall-chunks.md` are provenance links to accepted Sprint 03 scene evidence, not new output directories. Existing `Sprint 08` references in the sprint plan are residual-risk records and remain intentional.

The facade API story is now consistent across the docs checked by this sprint:

- `renderer::prelude` is the beginner-supported alpha import path.
- Root-level compatibility exports remain public for existing users, editor workflows, tests, and diagnostics.
- Lower-level renderer integration remains explicitly gated behind `advanced-interop`.
- Larger advanced rendering opt-in work is deferred to Sprint 10.

## Validation Commands

| Command | Result | Notes |
| --- | --- | --- |
| `cargo fmt --check` | Pass | Re-run after docs-only patch; no formatting changes required. |
| `cargo check` | Pass | Completed with existing renderer dead-code warnings. |
| `cargo test -p renderer` | Pass | 160 unit tests, 20 integration tests, and 5 ignored doctests passed. |
| `cargo check -p renderer --examples` | Pass | Completed with existing renderer dead-code warnings. |
| `cargo test -p input` | Pass | Conditional input-profile validation: 10 unit tests and 0 doctests passed. |
| `rg -n "sprint-04\|sprint-03" docs/api` | Pass for intent | No prior-sprint capture output paths remain. Remaining Sprint 03 references are evidence provenance links in the editor placement doc. |
| `rg -n "TODO\|pending\|planned\|not implemented\|/tmp\|sprint-08\|Sprint 08\|sprint-04\|headless-draw" docs .internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-09-facade-api-contract` | Pass for intent | Remaining hits are accepted residual records, scan command text, legitimate `/tmp` CLI examples, current `headless-draw` paths, and existing internal-doc future-state prose. |
| `rg -n "stable public surface\|Everything below api\|advanced-interop\|prelude\|SceneWorld\|CommandHistory\|AnimationPlayer" docs/api src/renderer/src src/renderer/examples src/renderer/tests` | Pass for intent | No stale `stable public surface` or `Everything below api` claims. Remaining hits are intended prelude usage, compatibility exports, tests, and advanced-interop gates. |

Detailed command logs were written under `.internal-dev/debug_reports/sprint-09-phase04/`.

## Capture Status

Not required. Phase 04 changed docs and evidence only; it did not change renderer, scene, shader, camera, material, asset, Vulkan, or editor behavior. If later remediation changes visible output, it must use true engine-owned headless draw capture with `--headless --capture_target draw`.

## Residual Risks

- Existing renderer dead-code warnings remain outside Sprint 09.
- Accepted Sprint 08 residuals remain out of scope unless explicitly reopened.
- Existing public compatibility exports remain available at the crate root, but they are not beginner-supported facade surface.
- Existing `cargo doc -p renderer --no-deps` unresolved intra-doc link warning from Phase 02 remains a pre-existing rustdoc/prose cleanup item.

## Recommended Email Summary Bullets

- Phase 04 reconciled Sprint 09 docs and evidence, including public API capture examples.
- Final validation gates passed: format, workspace check, renderer tests, renderer examples, input tests, and stale scans.
- No headless capture was required because the phase was docs/evidence-only.
- Final quality review can proceed with conservative residuals still visible.
