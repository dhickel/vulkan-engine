# Validation Matrix

| Gate | Phase | Required Evidence | Pass Criteria | Block Criteria |
|---|---:|---|---|---|
| Release inventory | 01 | `reports/phase-01-release-inventory.md` | Public docs and live contracts reconciled; predecessor sprint status recorded | Missing Sprint 10-12 contracts or docs/code conflict without decision |
| Public docs lock | 01 | Changed docs plus validation report | Docs state alpha limits, platform/driver expectations, quickstarts, and no overclaims | Docs imply unsupported production features |
| Clean source validation | 02 | `reports/phase-02-fresh-clone-validation.md` | Clone/worktree commands and required checks run from isolated clean path | Validation depends on uncommitted local state |
| Package validation | 03 | command results and pack output | Sample project/package/scene validate and pack output has `PACK_REPORT.json` | Runtime handles in scene/package data or pack fails |
| Editor edit/save | 03 | edited scene artifact, editor capture | Sample can be opened, edited, saved, and visually captured via draw target | Edit/save corrupts canonical sample or capture missing |
| Runtime sample run | 03 | root runtime capture | Headless draw capture succeeds for sample/edited scene path | Capture sidecars not draw target or runtime fails |
| Dogfood run | 04 | runtime command reports | Full-content documented dogfood settings run for required selectors or accepted residuals | Default dogfood path fails from clean source |
| Dogfood visual proof | 04 | draw capture directory | Dogfood supports true headless draw capture and visual result is inspectable | Only desktop/windowed/present proof exists |
| Release notes | 05 | release notes draft | Scope, features, limitations, validation, known issues complete | Critical residual omitted |
| Evidence consistency | 05 | `artifacts/validation-summary.json` | Status fields match reports and no unearned final status | JSON contradicts reports or claims full validation too early |
| Final quality review | 05 | `validation/final-quality-review.md` | Independent validator passes or accepts residuals | No-release blocker remains |

## Required Capture Sidecar Predicate

Each accepted visual proof sidecar must satisfy:

```text
status == "succeeded"
capture_target == "draw"
png path exists
extent.width > 0
extent.height > 0
```

Where format is available, prefer a draw-target format such as `R16G16B16A16_SFLOAT`; do not fail solely on format name if the renderer legitimately changes format and the validator confirms it is the draw target.

## Evidence Summary Status Rules

Allowed initial/intermediate statuses:

- `not_started`
- `in_progress`
- `implementation_checks_passed`
- `validator_passed`
- `validator_failed`
- `capture_pending`
- `capture_failed`
- `blocked_tooling_constraint`
- `final_quality_pending`
- `final_quality_review_passed_with_residuals`
- `fully_validated`
- `release_blocked`

Do not use `fully_validated` unless all required validators, capture proof, clean validation, and final quality review pass with no unaccepted release blockers.

