# BSP Beta M3 GUI Input and Regeneration

## Topic
Reusable input-routing, GUI-state, cursor-coordinate, and live-regeneration rules for the windowed EnhancedV3 BSP beta explorer.

## Source References
- `apps/bsp_beta/src/m3_gui.rs`
- `apps/bsp_beta/src/generation.rs`
- `apps/bsp_beta/src/main.rs`
- `.internal-dev/specifications/bsp-dungeon-generation.md` §20.17
- `.internal-dev/specifications/bsp-transaction-ownership.md` §15
- `.internal-dev/specifications/architecture.md` (`ARCH-20260801-01`)
- `.internal-dev/specifications/decisions.md` (`DECISION-20260801-01`)
- `.internal-dev/knowledge/app-owned-input-frame-dispatch.md`

## Key Takeaways

### Device Exclusivity Must Precede Renderer/App Queueing
- F1/F2 must be intercepted before `Renderer::route_platform_input`, because that method owns built-in F1/F2 panels.
- An open menu must classify and suppress input before `route_platform_input_to_app`; ImGui `want_capture_*` is not a sufficient dual-input guard.
- Keyboard mode sends only keyboard events to `M3Gui` and discards pointer/button/wheel/raw motion. Mouse mode sends only pointer/button/window-wheel to `M3Gui` and discards keyboard except Escape and the global mode keys.
- Cursor enter/leave is a platform-policy exception, not gameplay input. Route it through `Renderer::route_platform_input`, discard the routing result, and never mirror it into the app input queue. This preserves Wayland cursor-constraint ownership.

### Opening and Closing Need Both State and Edge Repair
- Queue synthetic releases for every gameplay binding and common mouse buttons on `None → menu`; otherwise a key pressed before menu registration can stay held after capture begins.
- Continue the single app-frame `InputSystem::dispatch_frame()` boundary so those releases become authoritative.
- Pause FPS controller updates while the menu is open. On close, restore the gate without synthesizing presses; the player must release/repress normally.
- Register app-owned ImGui only while a menu is open and call `refresh_cursor_capture` after registration changes. A two-second post-close message must not retain capturing app UI; use a noncapturing title indication.

### RefCell and Renderer Registration Form One Transition
- Acquire the mutable GUI guard before mutating renderer registration. After register/unregister succeeds, the remaining GUI/input state commit must be infallible.
- Never compute a mode with an inline `gui.try_borrow()` argument to a function that later needs `try_borrow_mut`; copy the `GuiMode` into a local and let the guard drop first.
- Store the exact `DebugViewId` returned by registration. Remove only that owned callback; never use a broad `has_app_ui` check as authority to unregister another view.
- Registration failure must preserve `GuiMode::None` and gameplay input. Closing should repair local mode/gate even if the owned callback is already absent.

### Mouse Coordinates Are Viewport-Relative
- `WindowEvent::CursorMoved.position` is the click authority. `Window::inner_position()` is the window's screen position and is not a cursor coordinate.
- Convert physical cursor and viewport values by the current positive finite scale factor to match ImGui logical display coordinates. Scale factors below 1.0 are valid.
- Clear cached cursor coordinates on mode changes and cursor enter/leave; require a fresh `CursorMoved` before accepting a click.
- Winit wheel-up is positive Y, while increasing GUI scroll moves to later content, so negate the Y delta before applying the GUI's bounded scroll.

### V3 Optional Fields and Grammar Semantics
- `V3Config::grammar_families.is_empty()` means all six families are eligible. The default GUI must therefore show all six checked. Unchecking one first materializes the other five as an explicit canonical allowlist.
- Disabling a feature flag must remove any explicit family that requires it, or reduce density to zero if no eligible family remains.
- Optional numeric fields keep `None` until explicitly edited so preset/per-room/default semantics and compatibility identity are preserved. A keyboard numeric edit that is left empty and committed must restore `None`; it must not silently retain a stale override.
- Feature-family check state includes both the grammar allowlist and the feature flag. A disabled feature makes its dependent family unavailable; the UI must not display it as selected or create an invalid allowlist when clicked.
- Decimal density editing accepts both `Period` and `NumpadDecimal`; integer fields accept top-row and numpad digits.
- Wall thickness is frozen at 16 and is not a V3Config field; show it disabled rather than fabricating a knob.

### Regeneration Reuses Existing Transactions
- GUI actions snapshot and validate a complete `GenConfig`; the event-loop thread never runs the generator or compiler.
- Domain B remains engine_pack generation, compiler execution, and atomic package publication in a request-unique directory.
- Domain A remains strict authorization, hidden prepare/upload, scene preflight, coordinator commit, and detached-mount retirement handoff.
- Apply-and-Close is request-ID presentation intent. It closes only after the matching latest request commits; ordinary Generate clears prior close intent, and stale/failing results never close or replace the active world.

## Engine Relevance
These rules generalize to any app-owned in-game editor that must coexist with renderer platform side effects and frame-buffered gameplay input. The key boundary is explicit device routing plus transactional UI registration, not passive ImGui capture flags.

## Open Questions
- Automated live F1/F2, dropdown, click, and Apply-and-Close interaction evidence is not yet recorded.
- Formal resize, minimize/restore, and surface-loss WSI lifecycle validation remains separate from the task-local startup smoke.
