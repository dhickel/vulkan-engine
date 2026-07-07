# Sprint 09: Facade API Alpha Contract

Date: 2026-07-03

Branch: `sprint/alpha-09-facade-api-contract`

Status: closed with accepted residuals

## Summary

Sprint 09 locked the alpha facade API contract:

- `renderer::prelude` is the supported beginner import path.
- Root-level exports remain compatibility public for existing users, tests, diagnostics, and editor-style workflows.
- `renderer::api::advanced` remains feature-gated behind `advanced-interop`.
- Advanced rendering extension work remains deferred to Sprint 10.

## Implementation

- Added `renderer::prelude` as a curated beginner-facing facade.
- Re-exported the prelude from the crate root for simple `use renderer::prelude::*` style imports.
- Updated renderer examples to use the beginner prelude where appropriate.
- Added integration tests for the prelude import contract, beginner error display contract, and strict capture target/sequence parsing.
- Reworked API docs to separate beginner facade, compatibility exports, advanced interop, input/camera/material/capture boundaries, and deferred gaps.
- Reconciled public capture examples so current docs write new output to neutral capture directories instead of prior sprint-specific evidence paths.

## Validation

- `cargo fmt --check`: pass
- `cargo check`: pass with existing renderer dead-code warnings
- `cargo check -p renderer`: pass with existing renderer dead-code warnings
- `cargo check -p renderer --examples`: pass with existing renderer dead-code warnings
- `cargo test -p renderer`: pass, 160 unit tests, 20 integration tests, 5 ignored doctests
- `cargo test -p input`: pass, 10 tests
- `cargo doc -p renderer --no-deps`: pass in Phase 02 with an existing unresolved intra-doc link warning recorded
- Stale API/facade scans: pass for intent
- Capture: not applicable; Sprint 09 did not change visible renderer/editor behavior

## Residuals

- `fully_validated=false` by design because accepted residuals remain.
- Protected local state remains out of scope: `.idea/engine.iml` and `.reasonix/`.
- Existing renderer dead-code warnings remain outside Sprint 09.
- The existing rustdoc unresolved intra-doc link warning remains a future cleanup item.
- Root compatibility exports remain public but are not beginner-supported facade surface.
- Sprint 08 accepted residuals remain out of scope unless explicitly reopened.

## Evidence

- Plan suite: `.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-09-facade-api-contract/`
- Validation summary: `.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-09-facade-api-contract/artifacts/validation-summary.json`
- Final quality review: `.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-09-facade-api-contract/validation/final-quality-review.md`
