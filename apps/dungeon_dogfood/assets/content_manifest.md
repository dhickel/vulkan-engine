# Dungeon Dogfood Content Manifest

**Project:** `apps/dungeon_dogfood`  
**Purpose:** Track runtime content inventory, provenance, and readiness for alpha dogfood.

## 1. Usage Rules

- Every runtime-loaded asset used by dungeon dogfood must be listed here.
- Keep paths repo-relative.
- Record license/source before merging external content.
- `alpha_blocker=yes` means missing/invalid asset blocks alpha readiness.

## 2. Current Asset Inventory

| Asset ID | Category | Path | Source | License | Status | alpha_blocker | Notes |
|---|---|---|---|---|---|---|---|
| lvl_01 | level | `apps/dungeon_dogfood/assets/levels/level_01.txt` | internal | internal | present | no | Existing phase 03/04 level scaffold. |
| prop_torch_sconce_a | model | `apps/dungeon_dogfood/assets/models/props/torch_sconce.glb` | TBD | TBD | missing | yes | Required for wall-mounted prop validation via `M` markers. |
| prop_crate_a | model | `apps/dungeon_dogfood/assets/models/props/crate_a.glb` | TBD | TBD | missing | yes | Required for floor prop coverage. |
| prop_altar_a | model | `apps/dungeon_dogfood/assets/models/props/altar_a.glb` | TBD | TBD | missing | yes | Required landmark/focal prop. |
| pbr_stone_rough | texture_set | `apps/dungeon_dogfood/assets/textures/pbr/stone_rough/` | TBD | TBD | missing | yes | Base wall/floor family. |
| pbr_stone_wet | texture_set | `apps/dungeon_dogfood/assets/textures/pbr/stone_wet/` | TBD | TBD | missing | yes | Accent material family. |
| pbr_metal_worn | texture_set | `apps/dungeon_dogfood/assets/textures/pbr/metal_worn/` | TBD | TBD | missing | yes | Prop/light fixture family. |
| env_dungeon_neutral | environment | `apps/dungeon_dogfood/assets/environments/dungeon_neutral.hdr` | TBD | TBD | missing | yes | Primary alpha skybox/IBL candidate. |
| env_dungeon_cool | environment | `apps/dungeon_dogfood/assets/environments/dungeon_cool.hdr` | TBD | TBD | missing | no | A/B tuning variant. |
| lvl_02_ramps | level | `apps/dungeon_dogfood/assets/levels/level_02_ramps.txt` | internal | internal | missing | yes | Must cover all 4 ramp tokens. |
| lvl_03_lighting | level | `apps/dungeon_dogfood/assets/levels/level_03_lighting.txt` | internal | internal | missing | yes | Dense lights for tuning matrix. |

## 3. Required Folder Layout (Alpha)

- `apps/dungeon_dogfood/assets/levels/`
- `apps/dungeon_dogfood/assets/models/props/`
- `apps/dungeon_dogfood/assets/textures/pbr/stone_rough/`
- `apps/dungeon_dogfood/assets/textures/pbr/stone_wet/`
- `apps/dungeon_dogfood/assets/textures/pbr/metal_worn/`
- `apps/dungeon_dogfood/assets/environments/`

## 4. Intake Checklist for New Assets

1. Confirm import format compatibility (prefer `.glb` for models, `.hdr` for environments).
2. Normalize world scale and orientation for consistent placement.
3. Run compile + bounded runtime smoke after adding assets.
4. Add/update manifest row with license and source URL/reference.
5. Mark `Status` as `present` only after runtime validation.
