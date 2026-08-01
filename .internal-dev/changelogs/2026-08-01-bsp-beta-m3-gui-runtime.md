# BSP Beta EnhancedV3 GUI Runtime Integration

## Date
2026-08-01

## Change Summary
Integrated the EnhancedV3 menu into the windowed `bsp_beta --m3-generate` runtime. F1 and F2 now own mutually exclusive keyboard and mouse modes, menu input is isolated before gameplay routing, and complete GUI snapshots drive the existing asynchronous generation, package compilation, atomic BSP replacement, and fence-aware retirement path.

## Files
- `apps/bsp_beta/src/main.rs`
- `docs/guide/18-bsp-beta.md`
- `docs/internal/18-bsp-runtime-and-lifetime.md`

## Behavioral Impact
- F1 opens or toggles keyboard mode; F2 opens or toggles mouse mode; switching never opens both.
- Keyboard mode consumes keyboard and discards mouse input. Mouse mode consumes pointer, button, and wheel input and discards keyboard except Escape and the global mode controls.
- Opening a menu queues gameplay releases and pauses FPS updates; closing restores gameplay immediately.
- App-owned ImGui registration and cursor policy transitions are ownership-checked and transactional.
- Generate keeps the menu open. Apply & Close closes only after its matching latest request commits successfully.
- Worker, compiler, authorization, coordinator publication, previous-world preservation, and detached-mount retirement behavior remain asynchronous and transactional.
- Successful generation displays a two-second overlay or noncapturing title indication.

## Specification Impact
The change implements a new windowed control surface for the already authorized EnhancedV3 explorer inventory. It does not alter any generator geometry, preset, RNG, compiler, package, or runtime mount ownership contract. The living generation specification requires a focused GUI/input integration note during final closeout.

## Risks
- Timeout-bound live startup proved the generated map, swapchain, BSP upload, and frame loop, but did not automate F1/F2 interaction or Apply & Close clicks.
- Formal resize, minimize/restore, and surface-loss WSI lifecycle evidence remains outside this feature validation.

## Follow-up Items
- Run final integration review across both GUI phases and the complete requested validation matrix.
- Record reusable dual-input and cursor-routing rules in project knowledge and reconcile the living EnhancedV3 specification.
