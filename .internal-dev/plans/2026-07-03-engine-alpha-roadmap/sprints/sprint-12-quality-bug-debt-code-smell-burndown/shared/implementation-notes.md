# Implementation Notes

## Required Reading For Every Worker

- `AGENTS.md`
- `.internal-dev/AGENTS.md`
- `src/renderer/AGENTS.md`
- `src/renderer/src/vulkan/AGENTS.md` when touching Vulkan
- `00-specification-lock.md`
- `01-current-state-analysis.md`
- `02-target-design.md`
- `shared/senior-engineer-guidance.md`
- The assigned phase directive

## Branch And Local State

Intended execution branch:

```sh
git switch -c sprint/alpha-12-quality-bug-debt-code-smell-burndown
```

Only do this when the main thread says Sprint 09 state is reconciled. If the branch already exists, switch to it after verifying no unrelated user work would be overwritten.

Protected paths:

- `.idea/engine.iml`
- `.reasonix/`
- `SPRINT-TRACKER.md`

## Report Paths

Workers write reports under:

`reports/phase-XX-*.md`

Validators write reports under:

`validation/phase-XX-validation-report.md`

The final quality review writes:

`validation/final-quality-review.md`

## Evidence Index Rules

Update:

`artifacts/validation-summary.json`

Do not use final status language until validators have passed. If any required validator is missing, any runtime/capture proof is pending, or any residual is unaccepted, the top-level status must not be `fully_validated`.

## Bug Artifact Rules

If execution finds an out-of-scope bug, create or update a focused artifact under `.internal-dev/bugs/<bug-id>/report.md` with the required `.internal-dev/AGENTS.md` headings.

Do not create `.internal-dev/notes/` future-consideration entries without asking the user.

## Runtime Evidence Paths

Debug reports:

`.internal-dev/debug_reports/sprint-12-*.jsonl`

Headless captures:

`.internal-dev/captures/sprint-12-quality-burndown/<scenario>/`

Temporary investigation notes:

`.internal-dev/headless_capture_tests/sprint-12-quality-burndown/`

## Validation Command Notes

Use timeout-bound runtime commands. Engine startup can take 20-30 seconds; default to 60 seconds unless a focused scenario needs a different bound.

If `cargo test -p renderer` fails due to pre-existing doctest/prose failures, record exact failing targets and determine whether Sprint 12 should fix them in Phase 04 or accept them with mitigation.

## Source Scans

Useful targeted scans:

```sh
rg -n "TODO|FIXME|todo!\\(|unimplemented!\\(|panic!\\(|unwrap\\(|expect\\(" src/renderer/src src/renderer/examples src/runtime.rs src/launch.rs apps docs/api docs/internal
rg -n "destroy path|double free|swapchain|old image view|VkSubAllocator|VkHostBuffer|fence\\[0\\]" src/renderer/src/vulkan src/renderer/src/data docs/internal .internal-dev/bugs
rg -n "pub use|pub mod|advanced-interop|prelude|SceneWorld|CommandHistory|AnimationPlayer" src/renderer/src docs/api
```

## Stop And Escalate

Escalate to the main thread instead of improvising if:

- a fix requires public API removal;
- a Vulkan ownership repair touches many resource owners;
- runtime/capture tooling is unavailable;
- the same targeted issue fails validation twice;
- residual acceptance would hide a critical crash, data loss, or invalid Vulkan lifetime bug.
