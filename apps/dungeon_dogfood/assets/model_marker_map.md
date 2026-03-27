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
- `N` is size of enabled prop list from `content_pack.toml`

This gives deterministic variation without changing level token format.

## 3. Current Prop Table

| Map Key | Path | Intended Placement | Scale | Orientation Policy | Shading Note |
|---|---|---|---|---|---|
| `prop_wall_torch` | `apps/dungeon_dogfood/assets/models/props/torch_sconce/scene.gltf` | marker tile center | `1.0` | use `yaw_degrees` from `content_pack.toml` (currently `0.0`) | non-PBR fallback |
| `prop_floor_crate` | `apps/dungeon_dogfood/assets/models/props/crate_a/wooden_crate_01_4k.gltf` | marker tile center | `1.0` | use `yaw_degrees` from `content_pack.toml` (currently `0.0`) | full PBR |

## 4. Placement Rules

- Base position: tile center (`tile_to_world + (0.5, 0.0, -0.5)`).
- Per-prop transform comes from `content_pack.toml` (`scale`, `yaw_degrees`, `y_offset`).
- v1 does not apply automatic wall-normal alignment or wall offset.
- If model load fails, log and continue (no panic, no hard abort).

## 5. Next Extension

- Add optional wall-normal placement policy fields in `content_pack.toml` for mounted props.
- If v2 needs explicit authoring, expand token grammar to typed markers (example: `M1`, `M2`) and retain fallback to v1 modulo behavior for backwards compatibility.
