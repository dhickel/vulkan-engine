# Phase 01 Validation Report: Manual Capture Burst Fix

Date: 2026-07-03
Validator: Codex validation agent

## Scope Reviewed

- Specification: `.internal-dev/plans/2026-07-03-manual-capture-burst-fix/00-specification-lock.md`
- Worker directive: `.internal-dev/plans/2026-07-03-manual-capture-burst-fix/worker-directives/phase-01-scheduler-burst-fix.md`
- Patch under review: `src/renderer/src/api/config.rs`
- Repo governance read: root `AGENTS.md` supplied in task context and `src/renderer/AGENTS.md`

## Findings

1. No blocking product-code findings.

2. Low / process: root governance references `.internal-dev/AGENTS.md`, but that file is absent in this checkout. Validation used the explicit plan path and report requirement from the task instead. This is not a blocker for the scheduler fix, but it is a repo/process documentation gap.

3. Residual risk, nonblocking: `last_frame_capture_status()` still exposes only one status. The scheduler fix makes rapid manual captures due on successive frames, so it reduces the original same-frame overwrite case for manual bursts. The single-status API can still only report the latest status when multiple capture sources produce statuses before the UI reads them. This matches the specification's residual-risk section and does not block this phase.

## Criterion Results

| Criterion | Result | Evidence |
| --- | --- | --- |
| Three rapid manual captures at current frame `0` become due on frames `1`, `2`, and `3` | Pass | `FrameCaptureScheduler::queue_manual_capture` stores and advances `next_manual_capture_frame`; `rapid_manual_captures_are_staggered_across_future_frames` asserts one due capture each at frames 1, 2, and 3. |
| No duplicate manual due captures occur on one frame for the burst case | Pass | The focused test drains frames 1, 2, and 3 separately and asserts `len() == 1` for each due result. |
| Manual filenames remain unique | Pass | `manual_sequence` remains the filename uniqueness source and the focused test asserts three distinct paths ending in `manual-0000.png`, `manual-0001.png`, and `manual-0002.png`. |
| Repeated manual captures share the same manual run directory | Pass | `scheduler_reuses_one_default_manual_run_dir` now expects successive due frames and asserts the same parent directory; focused burst test also asserts all parents match. |
| Single capture behavior unchanged | Pass | `schedule_single_capture` implementation is unchanged; `single_capture_fires_once_at_requested_frame` passes. |
| Sequence capture behavior unchanged | Pass | Sequence scheduling implementation is unchanged except shared `due_captures` pending-manual cleanup; `sequence_capture_fires_exact_count_at_interval` passes. |
| Explicit manual output directory behavior unchanged | Pass | `configure_manual_output_dir` is unchanged; `manual_capture_uses_default_dir_and_next_frame` still passes and confirms the explicit output dir path. |
| No Vulkan/backend refactor | Pass | `git diff --name-status` shows product-code changes only in `src/renderer/src/api/config.rs`; `rg` inspection found no changed Vulkan/backend path. |
| No F10/F12/public API broad changes | Pass | Diff inspection shows no changes to key handling or public renderer API. Existing F10/F12 handling remains in `src/renderer/examples/*` and `src/renderer/src/api/renderer.rs`; the patch only adds private scheduler state and test changes in `config.rs`. |
| Focused scheduler tests cover burst behavior | Pass | Added `rapid_manual_captures_are_staggered_across_future_frames`, covering due frame numbers, one due capture per frame, unique filenames, and shared parent directory. |

## Commands Run

```bash
cargo fmt --check
```

Result: Pass.

```bash
cargo test -p renderer capture
```

Result: Pass. 11 tests passed, 0 failed, 125 filtered out. Existing renderer warnings were emitted.

```bash
cargo check -p renderer --examples
```

Result: Pass. Existing renderer warnings were emitted.

```bash
git diff --check
```

Result: Pass.

Additional inspection:

```bash
git diff --name-status
git diff --stat
rg -n "queue_manual_capture|FrameCaptureScheduler|due_captures|last_frame_capture_status|FrameCaptureSource|Manual|F10|F12|record_status|last_status" src/renderer/src src/renderer/examples
```

Evidence: product-code patch remains constrained to `src/renderer/src/api/config.rs`; unrelated dirty paths `.idea/engine.iml` and `.reasonix/` were preserved.

## Final Status

Phase 01 validation passes with no blockers. No remediation handoff is required.
