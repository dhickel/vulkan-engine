# Work Units

## Phase 01: Current-State Contract Audit

Audit live code/docs/tests for app crates, app templates, scripting, script assets, script events, and hot reload claims. Produce an audit artifact and phase report. No product behavior changes.

## Phase 02: Rust App Template Path

Implement or deliberately defer the buildable minimal app template path. Preferred home is `tools/engine_pack`. Generated output must build without renderer internals.

## Phase 03: Script Asset And Event Boundary

Harden `src/scripting` around safe log/event/error helpers and, if narrow and testable, add script asset validation support. Keep scripts experimental.

## Phase 04: Docs And Final Validation

Align public/internal docs with implemented claims, run final validation, reconcile evidence, and prepare closeout artifacts.

## Dispatch Rule

Run phases sequentially. Each mutating phase must pass validation before the next phase starts. Main thread handles branch/push and HTML email after each validated phase.
