# EnhancedV3 GUI Domain and Input

## Date
2026-08-01

## Change Summary
Added the complete EnhancedV3 GUI draft model, keyboard and mouse interaction logic, deterministic hitboxes, text representation, and ImGui overlay renderer. Expanded the existing BSP beta generation request model to carry every public `V3Config` knob without losing optional default semantics.

## Files
- `apps/bsp_beta/src/m3_gui.rs`
- `apps/bsp_beta/src/generation.rs`
- `apps/bsp_beta/src/lib.rs`
- `apps/bsp_beta/Cargo.toml`
- `Cargo.lock`

## Behavioral Impact
The BSP beta library now exposes a validated seven-section M3 menu model with mutually exclusive mode state, complete keyboard navigation/editing, raw mouse hit testing, exact dropdown selection, bounded scrolling, presets, randomization, reset actions, generation commands, status reporting, and a two-second completion flash. Wall thickness remains a displayed disabled structural invariant rather than a fabricated generator setting.

## Specification Impact
The implementation realizes the already authorized EnhancedV3 explorer inventory in `bsp-dungeon-generation.md` §20 and does not change any generator bound, preset, geometry, compiler, RNG, or compatibility contract. Runtime input-routing and publication behavior remain pending the dependent integration phase.

## Risks
- Runtime event routing and application callback registration are not part of this phase and must preserve the GUI model's raw coordinate convention.
- Existing unrelated workspace warnings remain unchanged.

## Follow-up Items
- Integrate F1/F2 mode ownership, input suppression, worker actions, BSP replacement, and UI callback registration in the BSP beta windowed M3 loop.
- Validate the integrated menu and regeneration lifecycle through focused tests and live startup evidence.
