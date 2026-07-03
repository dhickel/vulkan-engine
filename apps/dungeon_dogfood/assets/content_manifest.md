# Dungeon Dogfood Content Manifest

**Project:** `apps/dungeon_dogfood`  
**Purpose:** Track runtime content inventory, provenance, and readiness for alpha dogfood.
**Updated:** 2026-03-25

## 1. Usage Rules

- Every runtime-loaded asset used by dungeon dogfood must be listed here.
- Keep paths repo-relative.
- Canonical runtime paths are under `apps/dungeon_dogfood/assets/*`; legacy import mirrors are staging-only.
- Record license/source before merging external content.
- `alpha_blocker=yes` means missing/invalid asset blocks alpha readiness.

## 2. Current Asset Inventory

| Asset ID | Category | Path | Source | License | Status | alpha_blocker | Notes |
|---|---|---|---|---|---|---|---|
| lvl_01 | level | `apps/dungeon_dogfood/assets/levels/level_01.txt` | internal | internal | present | no | Existing level scaffold. |
| lvl_02_ramps | level | `apps/dungeon_dogfood/assets/levels/level_02_ramps.txt` | internal | internal | present | no | Covers all 4 ramp tokens; parser-valid in tests. |
| lvl_03_lighting | level | `apps/dungeon_dogfood/assets/levels/level_03_lighting.txt` | internal | internal | present | no | Dense light markers; parser-valid in tests. |
| prop_torch_sconce_a | model | `apps/dungeon_dogfood/assets/models/props/torch_sconce/scene.gltf` | dogfood_dungeon import | needs_verification | present | no | Non-PBR fallback prop for now. |
| prop_crate_a | model | `apps/dungeon_dogfood/assets/models/props/crate_a/wooden_crate_01_4k.gltf` | dogfood_dungeon import | needs_verification | present | no | Primary PBR prop. |
| prop_landmark_a | model | `apps/dungeon_dogfood/assets/models/props/landmark_a/` | needs_verification | needs_verification | missing | no | Optional follow-up; not required to start phase 2. |
| dogfood.audio.startup_ping | audio | `apps/dungeon_dogfood/assets/audio/startup_ping.wav` | internal/generated | internal | present | no | Tiny generated WAV used for opt-in audio bridge proof; startup only probes it unless audio smoke is explicitly enabled. |
| pbr_stone_wall | texture_set | `apps/dungeon_dogfood/assets/textures/pbr/stone_rough/` | dogfood_dungeon import | needs_verification | present | yes | Wall material family (PBR). |
| pbr_stone_floor | texture_set | `apps/dungeon_dogfood/assets/textures/pbr/stone_floor/` | dogfood_dungeon import | needs_verification | present | yes | Floor material family (PBR). |
| pbr_metal_worn | texture_set | `apps/dungeon_dogfood/assets/textures/pbr/metal_worn/` | needs_verification | needs_verification | missing | no | Deferred for later phase if needed. |
| env_dungeon_neutral | environment | `apps/dungeon_dogfood/assets/environments/dungeon_neutral.exr` | dogfood_dungeon import | needs_verification | present | yes | Primary EXR skybox/IBL candidate. |
| env_dungeon_alt | environment | `apps/dungeon_dogfood/assets/environments/dungeon_alt.*` | needs_verification | needs_verification | missing | no | Optional A/B environment. |

## 3. Required Folder Layout (Current Alpha)

- `apps/dungeon_dogfood/assets/levels/`
- `apps/dungeon_dogfood/assets/models/props/`
- `apps/dungeon_dogfood/assets/audio/`
- `apps/dungeon_dogfood/assets/textures/pbr/stone_rough/`
- `apps/dungeon_dogfood/assets/textures/pbr/stone_floor/`
- `apps/dungeon_dogfood/assets/environments/`

## 4. Intake Checklist for New Assets

1. Confirm import format compatibility (prefer `.gltf/.glb` for models, `.exr/.hdr` for environments).
2. Normalize world scale/orientation and record corrections in machine-readable content pack manifest.
3. Run compile + bounded runtime smoke after adding assets.
4. Add/update manifest row with license and source URL/reference.
5. Mark `Status` as `present` only after runtime validation.

## 5. Manifest Split

- Human-readable inventory/provenance: this file (`content_manifest.md`).
- Runtime machine-readable content selection: `apps/dungeon_dogfood/assets/content_pack.toml` (schema locked in phase 01; parser/mapping expansion continues in phase 02).
