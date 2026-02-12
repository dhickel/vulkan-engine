## Summary

- Describe the change set and intent.

## Validation

### Required Command Checklist

```bash
cargo check
cargo check -p renderer
cargo check -p input
```

- [ ] `cargo check`
- [ ] `cargo check -p renderer`
- [ ] `cargo check -p input`

### Required Runtime Smoke (Vulkan Environment)

Reference: `.internal-dev/plans/baseline-smoke-checklist.md`

- [ ] Launch app normally.
- [ ] Rotate camera 360 degrees (no NaN jumps).
- [ ] Resize window 3 times (no crash/swapchain rebuild stable).
- [ ] Confirm ImGui renders after resize.
- [ ] Exit with `Escape`.

## Risk / Rollback

- Primary risk:
- Rollback plan:

## Notes For Reviewers

- Areas to review closely:
- Known limitations:
