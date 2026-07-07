# Worker Directive: Scheduler Burst Fix

## Scope

Implement the scheduler-side fix for rapid manual capture bursts.

## Editable Targets

- `src/renderer/src/api/config.rs`
- `src/renderer/src/api/renderer.rs`, only for a clarifying doc comment if needed

## Implementation Guidance

1. Add scheduler state such as `next_manual_capture_frame: Option<u32>` to `FrameCaptureScheduler`.
2. Initialize it in `FrameCaptureScheduler::new`.
3. In `queue_manual_capture(current_frame, target)`, compute the scheduled frame as:
   - `current_frame.wrapping_add(1)` when there is no pending manual slot;
   - otherwise the stored next manual frame slot.
4. After scheduling, advance the next manual slot by one frame.
5. Use the scheduled frame for both `manual_capture_path(...)` and `ScheduledSingleCapture.frame_number`.
6. Keep `manual_sequence` as the filename uniqueness source.
7. Update `scheduler_reuses_one_default_manual_run_dir` so it expects successive frames instead of same-frame due captures.
8. Add a focused test proving three rapid queues at frame `0` are due on frames `1`, `2`, and `3`, with one due capture per frame.

## Stop Conditions

- Stop if the fix requires Vulkan capture finalization changes.
- Stop if wraparound-safe scheduling requires a broader frame-index policy decision.
- Stop if existing tests reveal manual sequence captures or explicit output directory behavior intentionally relies on same-frame bursts.

## Validation

Run:

```bash
cargo fmt --check
cargo test -p renderer capture
cargo check -p renderer --examples
git diff --check
```

Record results for the validator.
