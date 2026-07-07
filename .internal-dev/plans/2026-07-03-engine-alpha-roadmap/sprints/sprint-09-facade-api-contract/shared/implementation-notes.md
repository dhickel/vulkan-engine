# Implementation Notes

## Files Workers Should Inspect

- `AGENTS.md`
- `.internal-dev/AGENTS.md`
- `src/renderer/AGENTS.md`
- `src/renderer/src/lib.rs`
- `src/renderer/src/api/mod.rs`
- `src/renderer/src/api/*.rs`
- `src/renderer/examples/*.rs`
- `src/renderer/examples/common/mod.rs`
- `src/renderer/tests/integration.rs`
- `docs/api/00-index.md`
- `docs/api/01-quickstart.md`
- `docs/api/02-renderer.md`
- `docs/api/02-renderer-lifecycle-and-frame-api.md`
- `docs/api/03-scene.md`
- `docs/api/03-scene-graph-and-fragment-workflows.md`
- `docs/api/04-assets.md`
- `docs/api/04-assets-sync-deferred-and-handles.md`
- `docs/api/06-input.md`
- `docs/api/06-input-polling-and-listeners.md`
- `docs/api/08-debug.md`
- `tools/engine_pack/**` only if generated app docs/templates are touched.

## Suggested Artifacts

- Export audit: `reports/phase-01-export-audit.md`
- Example contract notes: `reports/phase-02-example-contract.md`
- Targeted friction notes: `reports/phase-03-friction-hardening.md`
- Final docs/evidence report: `reports/phase-04-final-docs-validation.md`

## Compatibility Notes

`src/renderer/tests/integration.rs` currently imports non-facade root exports. This is evidence that legacy root exports are still part of current compatibility. Workers should not treat those tests as wrong by default.

## Capture Notes

Most Sprint 09 work should be static docs/API/examples/testing. If a worker changes runtime rendering behavior, they must use the headless capture skill and true engine-owned capture with `--headless --capture_target draw`. Desktop screenshots do not satisfy validation.

## Email/Branch Notes

The user requested branch/push and HTML email after every phase. Planning artifacts should require phase reports for that email, but the main thread handles actual git push and email through the proper orchestration/email tools.
