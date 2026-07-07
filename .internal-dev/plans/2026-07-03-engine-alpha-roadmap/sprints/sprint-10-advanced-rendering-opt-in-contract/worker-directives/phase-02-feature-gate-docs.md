# Phase 02 Worker Directive: Feature-Gate And Documentation Hardening

## Objective

Harden the default vs advanced contract in exports, docs, and focused checks so beginner users see safe extension points while advanced interop remains explicit, feature-gated, and unstable.

## User-Visible Outcome

Renderer API docs clearly state which extension points are safe by default, which advanced APIs require `advanced-interop`, and what risks apply. Beginner examples continue compiling without advanced APIs.

## Editable Targets

- `docs/api/00-index.md`
- `docs/api/05-render-hooks-and-extension-points.md`
- `docs/api/05-hooks.md`
- `docs/api/08-debug.md` if debug view classification needs clarification.
- `src/renderer/src/api/advanced.rs` for source docs only unless Phase 01 recommends tiny compile-gate code checks.
- `src/renderer/src/api/prelude.rs`, `src/renderer/src/api/mod.rs`, `src/renderer/src/lib.rs` only if Phase 01 finds accidental advanced leakage.
- Focused tests only if required to prove feature-gate behavior.
- Sprint 10 reports/evidence files.

## Forbidden Scope

- Do not add raw Vulkan handle access.
- Do not add advanced APIs to `renderer::prelude`.
- Do not change rendergraph pass order.
- Do not edit `.idea/engine.iml`, `.reasonix/`, or `SPRINT-TRACKER.md`.

## Supporting Docs To Read

- Phase 01 audit report.
- `00-specification-lock.md`
- `02-target-design.md`
- `shared/senior-engineer-guidance.md`
- `src/renderer/AGENTS.md`

## Senior Engineer Guidance

- Make one docs page canonical for render hooks and extension points.
- Correct stale hook claims; safe hooks do not expose command buffers or raw rendergraph state.
- Use alpha/unstable language for `advanced-interop` and state that compatibility may change across alpha sprints.
- Keep docs grounded in live code and explicitly mark deferred custom rendergraph pass registration.
- If a doc example shows advanced use, wrap it in feature-gate context and label it advanced.

## Ordered Steps

1. Read the Phase 01 audit and confirm there are no plan defects.
2. Update docs to define contract tiers: beginner facade, safe extensions, advanced named extensions, raw backend interop.
3. Fix `docs/api/05-hooks.md` or make it a concise redirect/compatibility page so it no longer contradicts live code.
4. Clarify `advanced-interop` source docs if needed.
5. Add or adjust focused tests only if Phase 01 found accidental export/prelude leakage.
6. Run the required compile checks listed below.
7. Update phase evidence and validation summary conservatively.

## Acceptance Criteria

- Docs clearly state safe hooks/debug views do not expose raw backend resources.
- Advanced interop is documented as feature-gated alpha/unstable.
- Default build and examples compile without `advanced-interop`.
- Feature-enabled renderer and examples compile.
- No beginner facade/prelude expansion includes advanced modules.

## Negative Checks

- Search docs for obsolete claims that hooks are for custom Vulkan command recording.
- Search examples for `api::advanced`, `rendergraph`, or `advanced-interop` requirements.
- Verify no default Cargo feature enables advanced interop.

## Validation Commands

```sh
cargo check -p renderer
cargo check -p renderer --examples
cargo check -p renderer --features advanced-interop
cargo check -p renderer --examples --features advanced-interop
```

If docs-only changes are made and commands are blocked by unrelated Sprint 09 state, record exact blocker output and do not claim pass.

## Stop Conditions

- Stop if default examples cannot compile without advanced APIs due to unresolved Sprint 09 work.
- Stop if fixing docs requires deciding a new advanced API shape beyond this phase.

## Evidence Expectations

- Validation report: `validation/phase-02-validation-report.md`
- Command results recorded in `artifacts/validation-summary.json`
- Any docs drift residuals recorded in `reports/`

## Do Not Close Unless

- Duplicate hook docs no longer conflict.
- Feature-gate compile checks are run or blockers are explicit.
- The validator can verify no advanced API was added to the beginner prelude.
