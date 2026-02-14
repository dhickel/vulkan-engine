# Model Marker Mapping (`M`) - Dogfood Dungeon

**Purpose:** Define deterministic runtime mapping from ASCII `M` markers to prop assets.

## 1. Current Parser Contract

- Level parser records `M` marker positions in row-major discovery order.
- Token does not yet encode model subtype (`M` is generic).
- Determinism requirement: identical level text must produce identical model selection.

## 2. Alpha Mapping Policy (v1)

Use marker index modulo asset list length:

- `asset_index = marker_index % N`
- `marker_index` is index in `ParsedLevel.model_markers` (row-major parse order)
- `N` is size of configured prop list

This gives deterministic variation without changing level token format.

## 3. Planned Asset Table

| Map Key | Path | Intended Placement | Scale | Orientation Policy |
|---|---|---|---|---|
| `prop_wall_torch` | `apps/dungeon_dogfood/assets/models/props/torch_sconce.glb` | wall-adjacent tiles | `1.0` | rotate to nearest wall normal |
| `prop_floor_crate` | `apps/dungeon_dogfood/assets/models/props/crate_a.glb` | floor center tiles | `1.0` | keep default yaw |
| `prop_landmark_altar` | `apps/dungeon_dogfood/assets/models/props/altar_a.glb` | room focal tiles | `1.0` | face spawn or corridor axis |

## 4. Placement Rules

- Base position: tile center (`tile_to_world + (0.5, 0.0, -0.5)`).
- Apply Y offset only if model pivot requires grounding correction.
- Wall-mounted props may add small wall offset (`~0.1m`) to avoid clipping.
- If model load fails, log and continue (no panic, no hard abort).

## 5. Future Extension

If v2 needs explicit authoring, expand token grammar to typed markers (example: `M1`, `M2`, `M3`) and retain fallback to v1 modulo behavior for backwards compatibility.
