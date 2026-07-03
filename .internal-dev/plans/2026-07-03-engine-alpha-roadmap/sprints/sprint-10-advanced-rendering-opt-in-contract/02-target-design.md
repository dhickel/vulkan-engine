# Target Design

## Contract Tiers

Sprint 10 should leave the renderer with four clearly named tiers:

| Tier | Feature gate | Intended use | Stability |
|---|---|---|---|
| Beginner facade | default | normal renderer/project/scene/input/debug/capture workflow | alpha supported |
| Safe extension points | default | app logic, telemetry, debug UI, lightweight frame observation | alpha supported with constraints |
| Advanced named extensions | `advanced-interop` unless proven safe for default | read-only frame/debug/capture descriptors or narrow validated extension points | alpha unstable |
| Raw backend interop | `advanced-interop`, unsafe | engine-internal experiments and expert diagnostics | unstable escape hatch |

## Safe Default Extensions

Default-safe APIs should remain API-level:

- pre/post render hooks;
- custom debug views;
- debug timing JSONL capture;
- frame capture requests/statuses through existing facade controls;
- event observation and app-owned state updates outside backend internals.

These docs must not promise command buffer access, descriptor mutation, queue submission, or custom GPU pass recording.

## Advanced Interop Boundary

`advanced-interop` can continue to expose:

- `renderer::api::advanced`;
- `renderer::rendergraph` if needed for expert experiments;
- unsafe raw core access only with prominent safety docs.

The target contract is that `advanced-interop` is not beginner-stable and does not imply API compatibility across alpha sprints.

## Named Extension Preference

If Sprint 10 adds code beyond documentation/tests, prefer one narrow named surface:

- read-only frame resource descriptors;
- debug texture/capture/readback descriptors;
- constrained advanced diagnostics reporting pass names/timings/resource availability;
- registration types that validate names/order before touching backend state.

Do not add raw Vulkan handles as new fields. If a feature cannot be expressed without raw handles, defer it and document the required synchronization/resource contract first.

## Rendergraph Position

Rendergraph custom pass registration is not stable enough to expose broadly in Sprint 10 unless a worker first adds:

- declared pass name and fixed insertion point;
- declared reads/writes or explicit "no GPU resource access" constraint;
- validation for duplicate names and unsupported insertion points;
- docs for incoming and outgoing layout/state assumptions;
- feature-gated compile checks and runtime smoke.

If those conditions cannot be met narrowly, the correct Sprint 10 outcome is to document rendergraph pass registration as deferred.

## Documentation Shape

- Make `docs/api/05-render-hooks-and-extension-points.md` the canonical hook/extension chapter or clearly redirect duplicates to it.
- Correct `docs/api/05-hooks.md` so it no longer overclaims raw GPU command extension.
- Update `docs/api/00-index.md` with the four-tier contract.
- Update `docs/internal/07-rendergraph-dependencies-and-aliasing.md` only for current truth and Sprint 10 residuals, not future marketing.
- Add source-level docs to `advanced.rs` if implementation changes the advanced module.

## Evidence Shape

`artifacts/validation-summary.json` is the canonical evidence index and starts conservatively. It should not claim final validation until phase reports, command results, and residuals are reconciled.

Required phase reports:

- `validation/phase-01-validation-report.md`
- `validation/phase-02-validation-report.md`
- `validation/phase-03-validation-report.md`
- `validation/phase-04-validation-report.md`

Capture evidence path, only if needed:

- `.internal-dev/captures/sprint-10-advanced-rendering-opt-in-contract/headless-draw/`
