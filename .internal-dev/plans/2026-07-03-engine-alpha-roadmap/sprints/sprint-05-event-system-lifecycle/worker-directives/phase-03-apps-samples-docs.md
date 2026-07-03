# Phase 03 Worker Directive: Editor/Dogfood/Sample/Docs Consumption

## Objective

Show that apps/tools can consume the event contract, and update API/internal docs so alpha users understand lifecycle, ordering, subscription, and deferred event families.

## User-Visible Outcome

Editor/dogfood/sample code demonstrates event subscription or recording without raw renderer ownership, and docs teach how to use and reason about the event system.

## Editable Targets

- `apps/editor/Cargo.toml` and `apps/editor/src/*` as needed for a minimal subscription/recording example.
- `apps/dungeon_dogfood/Cargo.toml` and `apps/dungeon_dogfood/src/*` as needed for a minimal subscription/recording example.
- Optional non-render sample/test in `src/events/examples/`, `src/renderer/examples/`, or root tests if it better fits the implemented API.
- `docs/api/00-index.md`
- New `docs/api/12-events-and-lifecycle.md`
- `docs/internal/00-index.md`
- New `docs/internal/10-event-system-and-lifecycle.md`
- Targeted cross-links in input/runtime/assets/scene/hooks docs.
- This sprint validation/evidence files if needed.

## Forbidden Scope

- Do not redesign editor panels.
- Do not migrate dogfood gameplay/collision.
- Do not add a full debug UI event browser unless it is already trivial and low risk.
- Do not make docs claim physics/audio/scripting emission works today.
- Do not touch unrelated `.idea/engine.iml` or `.reasonix/`.

## Supporting Docs To Read

- Phase 01 and Phase 02 implementation.
- `docs/api/00-index.md`
- `docs/internal/00-index.md`
- `docs/api/06-input-polling-and-listeners.md`
- `docs/api/11-runtime-project-launcher.md`
- `docs/internal/09-input-winit-integration.md`

## Implementation Steps

1. Add minimal event subscription/recording to editor and/or dogfood startup paths.
2. Keep consumption observable through logs, a recorder, or tests; avoid UI churn.
3. Add a simple non-render example or test that records lifecycle/action events if app changes alone are not easy to validate.
4. Add public event docs:
   - crate/facade imports;
   - event families;
   - subscription and recorder examples;
   - frame stage/order table;
   - input action bridge timing;
   - mutation safety rules;
   - deferred physics/audio/scripting behavior.
5. Add internal event docs:
   - ownership boundaries;
   - runtime/renderer integration points;
   - ordering rules;
   - validation guidance.
6. Cross-link docs from relevant index/runtime/input/assets/scene/hooks pages.
7. Run app/doc validation commands and prepare phase report.

## Senior Guidance

- A small event recorder log is enough. Avoid product UI unless needed for compile visibility.
- Docs should be exact: "event type exists" is different from "system emits this event today."
- Prefer examples that compile over prose-only API claims.
- Keep app code using public facade imports, not renderer internals.

## Acceptance Criteria

- At least one app path demonstrates subscription/recording through public APIs.
- Editor and dogfood still compile.
- Docs have a public and internal event page linked from indexes.
- Docs state event ordering and mutation safety clearly.
- Deferred physics/audio/scripting scope is explicit.

## Negative Checks

- No app code reaches into renderer private modules.
- No broad UI/gameplay refactor.
- No docs claim desktop screenshot proof or Playwright validation.
- No stale "not implemented" wording for in-scope event crate features.

## Validation Commands

```bash
cargo check -p editor
cargo check -p dungeon_dogfood
cargo check -p engine_pack
rg -n "events|lifecycle|EventBus|engine_events" docs/api docs/internal
```

Also run a docs stale sweep focused on this sprint:

```bash
rg -n "/tmp|TODO|desktop screenshot|playwright|not implemented" docs .internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-05-event-system-lifecycle
```

## Evidence Expectations

- Validator report path: `.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-05-event-system-lifecycle/validation/phase-03-validation-report.md`
- Report docs files changed, app consumption path, and any deferred residuals.

## Stop Conditions

- Stop if app consumption requires broad lifetime/API redesign.
- Stop if docs cannot accurately describe behavior because Phase 02 criteria are ambiguous.
- Stop and return to planning if public API names changed enough to invalidate earlier directives.

## Do Not Close Unless

- App/sample consumption compiles.
- Docs match implemented behavior.
- Deferred system families are honestly described.
- Phase 03 validation report is ready.
