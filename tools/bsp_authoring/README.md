# BSP Authoring Tools

Assets, profiles, and editor configuration for authoring Q1-family BSP maps
targeting the vulkan-engine BSP beta.

## Directory Layout

```
tools/bsp_authoring/
├── README.md                    ← this file
├── ericw-q1-profile.toml        ← pinned BSP29 compiler profile for engine_pack compile-bsp
├── ericw-q1-bsp2-generated-profile.toml ← pinned deterministic BSP2 compiler profile
└── TrenchBroom/
    ├── GameConfig.cfg           ← TrenchBroom game configuration
    └── Engine.fgd               ← entity definitions (FGD)
```

## Prerequisites

- **ericw-tools** 2.0.0-alpha3 (minimum)
  - Download: https://github.com/ericwa/ericw-tools/releases
  - Required executables: `qbsp`, `vis`, `light`
  - Pinned executable hashes (SHA-256):
    - qbsp: `4a05974acf9e59f73a9c8f4e8236f3d1e0961be477dae002837e166278882f17`
    - vis:  `f7f429e0ad9bebbb0ebdefea8d6cd5e13a6ad6ff9f6893126772e00b66e364ea`
    - light: `1210ee9bed8990f67e3be7e28fbfd8d210329b052ac57dba017200d2da1ca5e5`
- **TrenchBroom** 2024.1+ (recommended)
  - Download: https://trenchbroom.github.io/
- **vulkan-engine** with BSP beta support

## Setup

### 1. Install ericw-tools

Build or download ericw-tools and place `qbsp`, `vis`, `light` on your PATH
or in a known directory.

### 2. Install TrenchBroom Game Configuration

Copy or symlink the `TrenchBroom/` directory into the TrenchBroom games folder:

**Linux:**
```sh
mkdir -p ~/.TrenchBroom/games/vulkan-engine-q1
cp -r tools/bsp_authoring/TrenchBroom/* ~/.TrenchBroom/games/vulkan-engine-q1/
```

**macOS:**
```sh
mkdir -p ~/Library/Application\ Support/TrenchBroom/games/vulkan-engine-q1
cp -r tools/bsp_authoring/TrenchBroom/* ~/Library/Application\ Support/TrenchBroom/games/vulkan-engine-q1/
```

**Windows:**
```sh
mkdir %APPDATA%\TrenchBroom\games\vulkan-engine-q1
xcopy tools\bsp_authoring\TrenchBroom\* %APPDATA%\TrenchBroom\games\vulkan-engine-q1\
```

### 3. Create a Project Palette

Copy `src/bsp/tests/fixtures/palettes/project_palette.lmp` into your project
directory or create your own 768-byte palette (256 RGB triples).

## Workflow

### Authoring (TrenchBroom)

1. Open TrenchBroom and select **vulkan-engine Q1 (ericw-tools BSP29)**.
2. Create a new map with the game configuration.
3. Use only engine-recognized entities:
   - `worldspawn` — world geometry and settings
   - `light`, `light_fluoro`, `light_flame_large_yellow`, `light_torch_small_walltorch` — lights
   - `func_door`, `func_button`, `func_plat` — structural brush models
   - `trigger_once`, `trigger_multiple`, `trigger_push` — trigger volumes
   - `target` — trigger chain targets
   - `info_player_start`, `info_player_deathmatch` — spawn markers
4. Save the `.map` file.

### Compilation (engine_pack)

```sh
# Compile with pinned ericw-tools BSP29 profile
engine_pack compile-bsp \
    path/to/map.map \
    --profile tools/bsp_authoring/ericw-q1-profile.toml \
    --out output/ \
    --palette path/to/palette.lmp \
    --tool-path /path/to/ericw-tools/bin

# Compile deterministic BSP2 with an authored WAD staged by basename
engine_pack compile-bsp \
    path/to/map.map \
    --profile tools/bsp_authoring/ericw-q1-bsp2-generated-profile.toml \
    --out output/ \
    --palette path/to/palette.lmp \
    --tool-path /path/to/ericw-tools/bin \
    --wad path/to/textures.wad

# Validate the compiled BSP
engine_pack validate-bsp output/map.bsp --palette path/to/palette.lmp
```

### Compiler Invocation Notes

- `compile-bsp` invokes `qbsp`, `vis`, and `light` as direct subprocesses
  (no shell). Each executable is located via `--tool-path` or PATH.
- The compiler version is verified against the required version in the
  profile before execution.
- Output BSP is re-validated through the engine's `bsp` parser after
  compilation (fail-closed: if validation fails, compilation is rejected).
- Compiler provenance (identity, version, arguments, staged WAD hashes, and
  output hashes) is recorded alongside the output.
- The BSP2 generated profile runs `light -threads 1 -lit` so `.bsp` and `.lit`
  bytes are reproducible across duplicate builds.

## Entity Reference

### Structural Entities (Engine-Recognized)

| Classname                     | Type        | Description                          |
|-------------------------------|-------------|--------------------------------------|
| `worldspawn`                  | solid       | World geometry, BSP tree root        |
| `light`                       | point       | Point light source                   |
| `light_fluoro`                | point       | Fluorescent light variant            |
| `light_flame_large_yellow`    | point       | Large yellow flame light             |
| `light_torch_small_walltorch` | point       | Small wall torch light               |
| `func_door`                   | solid       | Sliding door                         |
| `func_button`                 | solid       | Push button                          |
| `func_plat`                   | solid       | Lift / platform                      |
| `trigger_once`                | solid       | One-shot trigger volume              |
| `trigger_multiple`            | solid       | Reusable trigger volume              |
| `trigger_push`                | solid       | Push/launch trigger volume           |
| `target`                      | point       | Trigger chain relay                  |
| `info_player_start`           | point       | Player spawn marker                  |
| `info_player_deathmatch`      | point       | Deathmatch spawn marker              |
| `info_teleport_destination`   | point       | Teleport destination marker          |

Unrecognized entities are preserved as generic tagged nodes with all
key/value data intact.

### Entity Key Conventions

- `targetname` — durable entity name for targeting
- `target` — target entity name to trigger
- `angle` — Quake angle (yaw in degrees; -1 = up, -2 = down)
- `light` — brightness (Quake light units)
- `_color` — RGB light color (space-separated floats 0.0-1.0)
- `_tb_id` — TrenchBroom per-entity UUID (candidate identity source)

## Limits

- BSP29 limits: 65,535 vertices, 65,535 edges, 32,767 clipnodes
- BSP2 limits: ~2 billion vertices/edges/faces (use `-bsp2` qbsp arg)
- Maximum entity string length: 1 MiB
- Maximum total lump allocation: 256 MiB

For maps exceeding BSP29 limits, add `-bsp2` to `default_qbsp_args` and
`default_light_args` in the compiler profile.

## Dungeon Generator

A procedural dungeon generator is specified but not yet implemented. The frozen
generation contract is in `.internal-dev/specifications/bsp-dungeon-generation.md`.
When the generator is implemented:
- Generated `.map` output is BSP2 only (use `ericw-q1-bsp2-generated-profile.toml`).
- Room connections use open arches (no doors for beta).
- Compile with `light -threads 1 -lit` for reproducible colored-light output.
- PBR texture companions follow `<texture>_norm.png` / `<texture>_gloss.png` naming.

## License

These authoring tools and configurations are provided under the same license
as the vulkan-engine project. They contain no copyrighted Quake content.
