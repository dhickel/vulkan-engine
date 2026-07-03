# Specification Lock

## Classification

Medium-to-large planning suite. The sprint spans API contracts, renderer feature gates, docs, tests, and optional runtime/capture evidence, but it should remain a focused Sprint 10 unit rather than a broad renderer refactor.

## Locked Objective

Create an explicit advanced rendering opt-in contract that protects the Sprint 09 beginner facade while giving advanced users a documented, feature-gated, alpha/unstable path.

## Acceptance Criteria

- The default beginner path remains small and does not require `advanced-interop`.
- `advanced-interop` is opt-in, documented as alpha/unstable, and validated in both default and feature-enabled compile modes.
- Safe facade hooks/debug views are documented as default-safe API-level extension points.
- Unsafe/raw backend interop remains isolated and documented with misuse risks.
- Custom rendergraph pass registration, material/shader override registration, and readback/debug texture features are not advertised as stable unless implementation validates their resource/order contracts.
- Any new product surface added during Sprint 10 is named, scoped, feature-gated where advanced, and covered by focused tests.
- Runtime/capture evidence is engine-owned headless draw capture only when visible renderer or capture/readback behavior changes.

## Validation Criteria

- Compile:
  - `cargo check`
  - `cargo check -p renderer`
  - `cargo check -p renderer --examples`
  - `cargo check -p renderer --features advanced-interop`
  - `cargo check -p renderer --examples --features advanced-interop`
- Focused tests:
  - Existing hook tests if hooks change.
  - Added API/export tests or feature-gate checks where practical.
  - Docs/examples compile checks where examples are touched.
- Runtime:
  - Debug-record smoke for `api_test` when runtime/API behavior changes.
  - Advanced runtime smoke only if an advanced runnable path is added.
- Capture:
  - Required only for visible renderer or capture/readback behavior changes.
  - Must use `--headless --capture_target draw`.
- Evidence:
  - Phase reports in `validation/phase-XX-validation-report.md`.
  - Canonical status in `artifacts/validation-summary.json`.

## Negative Criteria

- No raw command buffers, raw `ash` handles, descriptor sets, allocator handles, swapchain images, queues, or fences as a new normal user-facing API.
- No advanced API in `renderer::prelude`.
- No beginner examples that require `advanced-interop`.
- No rendergraph pass registration without explicit order/resource/synchronization constraints.
- No desktop screenshots as renderer proof.
- No `fully_validated` status unless all required validation and residual reconciliation are complete.

## Non-Goals

- Full rendergraph redesign.
- Production plugin ABI.
- Shader/material graph system.
- Broad dogfood/editor migration.
- Sprint 09 code changes.
- Sprint tracker update.

## Constraints

- Planning task is non-mutating for product code.
- Do not touch Sprint 09 active files except read-only during plan creation.
- Do not edit `.idea/engine.iml`, `.reasonix/`, or `SPRINT-TRACKER.md`.
- Later execution should start from the intended branch `sprint/alpha-10-advanced-rendering-opt-in-contract` after the main thread reconciles Sprint 09.
- Code remains the logical source of truth; docs are intended truth.

## Assumptions To Verify Before Execution

- Sprint 09 has landed or the execution branch includes its accepted facade boundary.
- Current dirty files are either Sprint 09-owned or reconciled before Sprint 10 workers mutate overlapping files.
- Local Vulkan/headless environment can initialize when runtime/capture validation is required.
- Existing renderer doctest/prose issues, if present, are not silently treated as Sprint 10 regressions unless Sprint 10 touches them.

## User Decision Gates

- If implementing a named advanced surface requires exposing raw backend handles, stop and ask whether to defer the feature instead.
- If the audit finds Sprint 09 facade state unresolved, stop before mutating API exports.
- If headless capture cannot initialize and capture proof is required, stop with blocker evidence.
- If validators find a plan defect or ambiguous acceptance criteria, return to planning before more coding.

## Stop Rules

- Stop if `advanced-interop` must become default to satisfy an implementation path.
- Stop if default examples require advanced imports.
- Stop if a proposed API bypasses synchronization or resource ownership contracts without staying inside explicitly unsafe interop.
- Stop if validation status and evidence disagree.
