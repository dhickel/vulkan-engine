# Validation Matrix

| Area | Required Evidence | Phase | Pass Criteria | Capture |
|---|---|---:|---|---|
| Branch hygiene | `git status --short --branch` before/after commit | all | On `sprint/alpha-02-packaging-tools`; `.idea/engine.iml` and `.reasonix/` not staged | no |
| Shared validation | focused renderer tests | 01 | package/project/scene validators reject locked invalid cases | no |
| Workspace build | `cargo check -p renderer`, `cargo check -p renderer --examples` | 01, 04 | no new compile failures | no |
| CLI build | `cargo check -p engine_pack` | 02, 03, 04 | CLI compiles as workspace package | no |
| CLI validation | `cargo test -p engine_pack` and command smokes | 02, 04 | valid fixtures pass, invalid fixtures fail with stable diagnostics | no |
| Authoring commands | CLI tests/tempdir smokes | 03 | generated project/package files revalidate | no |
| Pack command | temp output fixture/report | 03, 04 | folder output contains expected manifests/assets/report and rejects unsafe paths | no |
| Sample project | `engine_pack validate-project apps/editor/sample_project/engine.project.toml` | 02, 04 | exits 0 and reports valid | no |
| Runtime/editor agreement | compatibility test or shared API proof | 02, 04 | CLI and editor use same validation or tests prove same validity result | no |
| Visual/render behavior | headless capture skill | conditional | required only if rendered asset readiness is changed or claimed | conditional |
| Evidence summary | `artifacts/validation-summary.json` | 04 | statuses match reports and no overclaiming | no |

## Final Required Command Set

```bash
cargo check
cargo check -p renderer
cargo check -p renderer --examples
cargo check -p input
cargo check -p engine_pack
cargo test -p engine_pack
```

Add focused commands for any renderer/shared-validator tests created by implementation.
