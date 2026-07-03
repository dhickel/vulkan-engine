# Phase 01 Validation Report: Core Event Crate/API

Date: 2026-07-03

Branch: `sprint/alpha-05-event-system-lifecycle`

Status: implementation checks passed, validator pending

## Scope Validated

- Added workspace crate `src/events` with package name `engine_events`.
- Added typed event families for lifecycle, input/action, scene, asset, physics, audio, and scripting.
- Added staged event metadata with monotonic sequence IDs and optional frame IDs.
- Added `EventBus` subscription, unsubscription, stage drain, full pending dispatch, listener failure collection, and bounded `EventRecorder`.
- Kept the crate independent from renderer/windowing/Vulkan/app crates.

## Files Created Or Changed

| File | Change |
|---|---|
| `Cargo.toml` | Added `src/events` workspace member. |
| `Cargo.lock` | Added `engine_events` package lock entry. |
| `src/events/Cargo.toml` | New std-only event crate manifest. |
| `src/events/src/lib.rs` | New event vocabulary, bus, recorder, and tests. |

## Commands

| Command | Result | Notes |
|---|---:|---|
| `cargo fmt --check` | Passed | Clean after formatting. |
| `cargo test -p engine_events` | Passed | 7 unit tests passed; 0 doctests. |
| `cargo check` | Passed | Existing renderer dead-code warnings only. |
| `rg -n "renderer\|ash\|vulkan\|winit\|imgui" src/events` | Reviewed | Matches were rustdoc independence text and `Hash` substrings; no dependency/import violations found. |
| `rg -n "^(use\|extern crate).*\\b(renderer\|ash\|vulkan\|winit\|imgui)\\b\|\\b(renderer\|ash\|vulkan\|winit\|imgui)\\s*=" src/events/Cargo.toml src/events/src/lib.rs` | Passed | No output; `rg` exit code 1 means no matches. |

## Test Coverage

- Event family construction.
- Monotonic sequence assignment.
- Stage-specific dispatch preserving emission order.
- Listener unsubscribe behavior.
- Listener failure collection while later listeners/events continue.
- Bounded recorder retention order.
- Zero-capacity recorder behavior.

## Residuals

- No Phase 01 product residuals accepted.
- Existing renderer dead-code warnings remain outside this phase.
- Renderer/runtime integration is intentionally deferred to Phase 02.

## Validator Handoff

Validate that `engine_events` is Vulkan-free, explicit enough for Phase 02 integration, and satisfies the Phase 01 directive. Confirm the evidence index remains conservative until validator pass and commit/push/report gates are complete.
