# EnhancedV3 In-Game GUI Closeout

**Date:** 2026-08-01
**Status:** Complete

## Summary

Completed the production windowed GUI for every public EnhancedV3 generator setting in `bsp_beta --m3-generate`.

## Changes

- Added the seven-section app-owned ImGui overlay with complete `V3Config` draft conversion, presets, exact grammar and enum selection, feature toggles, numeric editing and steppers, randomization, reset, bounded scrolling, validation status, and the disabled fixed wall-thickness invariant.
- Added mutually exclusive F1 keyboard and F2 mouse modes with pre-renderer hotkey interception, complete gameplay-input suppression, synthetic gameplay releases on open, FPS controller gating, cursor-policy routing, HiDPI coordinate conversion, and Escape-always-close behavior.
- Connected Generate and Apply & Close to the existing asynchronous EnhancedV3, ericw, atomic package publication, strict authorization, coordinator prepare/validate/commit, and renderer retirement lifecycle. Close intent is keyed to the exact request and a successful publication reports `Generated!` for two seconds.
- Preserved direct BSP, headless generation, MCP, and no-menu gameplay behavior.
- Added focused GUI, routing, request-identity, error-path, and CPU-only ImGui draw-data tests.
- Corrected legacy-only generator wording in the BSP guide and specifications.

## Validation

- `cargo check -p bsp_beta`
- `cargo test -p bsp_beta --lib` — 51 passed
- `cargo test -p bsp_beta --bin bsp_beta` — 42 passed
- `cargo test -p bsp_beta`
- `cargo fmt --check -p bsp_beta`
- `git diff --check`
- Timeout-bound live `bsp_beta --m3-generate` startup reached swapchain creation, BSP upload, and repeated successful frame diagnostics without panic, ERROR-level engine output, or Vulkan validation errors.

Automated live F1/F2/click interaction and formal resize/minimize/surface-loss WSI lifecycle testing were not run. No headless GUI screenshot is claimed because the renderer headless path does not own the app ImGui context.

## Specification Impact

Updated `.internal-dev/specifications/bsp-dungeon-generation.md`, `bsp-acceptance.md`, `bsp-transaction-ownership.md`, `architecture.md`, and `decisions.md` to record the all-knob GUI, input-isolation boundary, transactional regeneration ownership, evidence, and remaining validation gaps.
