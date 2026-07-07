# Phase 03 Worker Directive: Error, Input, Camera, Material Docs Hardening

## Objective

Harden the most likely beginner friction points with targeted docs, tests, and small wrappers only where they are already supported by the architecture.

## User-Visible Outcome

Beginners can understand project/package/scene load errors, input-profile schema expectations, camera mode choices, material override limits, and debug/capture controls without being directed into internals or deferred Sprint 10 work.

## Editable Targets

- `src/renderer/src/api/errors.rs`
- `src/renderer/src/api/renderer.rs`
- `src/renderer/src/api/assets.rs`
- `src/renderer/src/api/scene.rs`
- `src/renderer/src/api/config.rs`
- `src/renderer/tests/integration.rs`
- `docs/api/03-scene*.md`
- `docs/api/04-assets*.md`
- `docs/api/06-input*.md`
- `docs/api/07-config.md`
- `docs/api/07-engine-arguments.md`
- `docs/api/08-debug.md`
- Optional `docs/api/*facade*` chapter created in earlier phases.
- `tools/engine_pack/**` only if generated app guidance or docs templates are touched.
- `reports/phase-03-friction-hardening.md`
- `artifacts/validation-summary.json`

## Forbidden Scope

- Do not implement a full project runtime API if it does not already exist.
- Do not build a new material system or renderer interop layer.
- Do not implement dynamic/runtime Rust reload or package-level script assets.
- Do not add production camera mode architecture.
- Do not change visible rendering behavior unless necessary; capture is required if this happens.

## Supporting Docs To Read

- Phase 01 and Phase 02 reports/validator reports.
- `00-specification-lock.md`
- `02-target-design.md`
- `shared/senior-engineer-guidance.md`
- `src/renderer/src/api/*.rs`
- Relevant docs under `docs/api/03*`, `04*`, `06*`, `07*`, `08*`
- `tools/engine_pack/` only if touched.

## Senior Engineer Guidance

- Start with docs/tests. Add code only for small wrappers around already-existing functionality.
- Error hardening should improve names, docs, examples, or tests without replacing the hierarchy.
- For input-profile TOML, document the actual schema if one exists; if not, explicitly record the gap instead of inventing one.
- For material overrides, do not promise first-class APIs unless implemented and compile-tested.
- For camera helpers, distinguish beginner usage from compatibility helper math.
- For debug/capture, repeat the repo rule: headless draw capture is the proof path; desktop screenshots do not count.

## Implementation Steps

1. Review the supported beginner path from Phase 02.
2. Identify up to four small friction fixes across errors, input, camera, material, and capture docs.
3. For each fix, decide: docs-only, test-only, small wrapper, or deferred.
4. Implement only the fixes that are bounded and directly support the sprint gate.
5. Add/adjust tests for any changed behavior.
6. Update docs to avoid aspirational claims about deferred features.
7. If visible behavior changed, run headless capture validation with `--headless --capture_target draw` and record artifacts.
8. Write `reports/phase-03-friction-hardening.md`.
9. Update `artifacts/validation-summary.json`.

## Acceptance Criteria

- Targeted docs explain current error/input/camera/material/capture behavior accurately.
- Any new wrapper/helper is small, public only where intended, and covered by tests or compile checks.
- Unsupported features are called deferred rather than silently promised.
- Conditional package/input checks run when those domains are touched.

## Negative Checks

- No full API redesign.
- No advanced rendering interop implementation.
- No package scripting/runtime reload/template generation.
- No hidden visual behavior changes without capture.

## Validation Commands

```sh
cargo fmt --check
cargo check
cargo test -p renderer
cargo check -p renderer --examples
rg -n "TODO|pending|planned|not implemented|material override|input profile|capture_target|desktop screenshot|advanced-interop" docs/api src/renderer/src src/renderer/tests
```

Run if touched:

```sh
cargo test -p input
cargo test -p engine_pack
cargo doc -p renderer --no-deps
```

Runtime smoke only if runtime behavior changes:

```sh
RUST_LOG=debug timeout --signal=INT 60s cargo run -p renderer --example api_test -- --record_debug=10 --record_debug_interval=50 --record_debug_path=.internal-dev/debug_reports/sprint-09-api_test-timing.jsonl
```

Headless capture only if visible behavior changes:

```sh
# Follow .internal-dev/skills/engine-headless-capture-validation/SKILL.md
# Must use --headless --capture_target draw
```

## Stop Conditions

- Stop if a desired friction fix requires broad renderer architecture.
- Stop if material/camera/input work would create unsupported API commitments.
- Stop if capture is required but headless capture is blocked; record `TOOLING_CONSTRAINT`.

## Evidence Expectations

- Worker report: `reports/phase-03-friction-hardening.md`
- Validator report path: `validation/phase-03-validation-report.md`
- Capture artifacts under `.internal-dev/captures/` only if required.
- Debug reports under `.internal-dev/debug_reports/` only if runtime smoke runs.

## Do Not Close Unless

- Each friction item is marked fixed, deferred, or out of scope.
- All conditional validation for touched domains is recorded.
- Evidence index remains conservative.
