# Senior Engineer Guidance

## Principles

- Preserve compatibility first. This is an alpha contract sprint, not a breaking cleanup sprint.
- Classify public surface before editing exports. The risk is user trust and compile stability, not only test green.
- Make examples the executable truth for beginner docs. If docs recommend an import or loop shape, at least one checked example or test should use it.
- Keep `advanced-interop` explicit. Do not smuggle advanced rendering internals into beginner docs.
- Prefer small wrappers only where they remove real beginner friction and can be tested in isolation.
- Treat docs as intended truth. If code and docs diverge, either reconcile them in scope or record a residual.

## Likely Failure Modes

- Over-curating `renderer::prelude` until it becomes a second broad root export list.
- Removing root exports that current tests or downstream users still rely on.
- Updating docs without making examples compile against the stated API.
- Writing aspirational docs for project runtime or material override APIs that do not exist.
- Running `cargo test -p renderer` and treating unrelated doctest failures as sprint regressions without triage.
- Using screenshots from the desktop/compositor as visual proof.

## Decision Logic

- If a symbol is needed by a beginner example, include it in the supported beginner facade or change the example.
- If a symbol is public and used by tests but too low-level for beginners, classify it as compatibility public.
- If a symbol needs `advanced-interop`, keep it out of beginner docs and mention the feature gate.
- If a desired API requires deeper renderer architecture work, document it as deferred to Sprint 10 or later.
- If a doc claim cannot be backed by code or examples, soften it or remove it.

## Protected State

Do not edit:

- `.idea/engine.iml`
- `.reasonix/`

Do not perform destructive git cleanup. The main thread owns branch/push, email, and closeout coordination.
