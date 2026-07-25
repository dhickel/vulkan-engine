#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: ./tools/dungeon_explore.sh [seed] [m1|m2]" >&2
}

if [[ $# -gt 2 ]]; then
  usage
  exit 2
fi

seed="${1:-0}"
class="${2:-m1}"

if [[ ! "$seed" =~ ^[0-9]+$ ]]; then
  echo "error: seed must be an unsigned integer, got '$seed'" >&2
  usage
  exit 2
fi

if [[ "$class" != "m1" && "$class" != "m2" ]]; then
  echo "error: class must be 'm1' or 'm2', got '$class'" >&2
  usage
  exit 2
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
cd "$repo_root"

cache_dir=".internal-dev/captures/bsp-dungeon-generator"
cache_bsp="$cache_dir/${class}-seed-${seed}.bsp"
cache_lit="$cache_dir/${class}-seed-${seed}.lit"

wad_path="src/bsp_generator/themes/cc0_stone_beta/cc0_stone_beta.wad"
palette_src="src/bsp_generator/themes/cc0_stone_beta/palette.lmp"
palette_tmp="/tmp/palette.lmp"
companion_dir="src/bsp_generator/themes/cc0_stone_beta/textures"
profile_path="tools/bsp_authoring/ericw-q1-bsp2-generated-profile.toml"
default_tool_path="$HOME/.local/ericw-tools/ericw-tools-2.0.0-alpha3-Linux/bin"

mkdir -p "$cache_dir"
if [[ ! -f "$palette_tmp" ]]; then
  cp "$palette_src" "$palette_tmp"
fi

if [[ -f "$cache_bsp" ]]; then
  echo "Using cached BSP: $cache_bsp"
else
  tmp_dir="$(mktemp -d -t dungeon-explore-${class}-${seed}-XXXXXX)"
  trap 'rm -rf "$tmp_dir"' EXIT

  map_path="$tmp_dir/${class}-seed-${seed}.map"
  out_dir="$tmp_dir/compiled"

  echo "Generating $class dungeon seed $seed..."
  cargo run -p dungeon_gen -- --seed "$seed" --class "$class" --out "$map_path"

  compile_args=(
    run -p engine_pack -- compile-bsp "$map_path"
    --profile "$profile_path"
    --out "$out_dir"
    --palette "$palette_tmp"
    --wad "$wad_path"
  )
  if [[ -x "$default_tool_path/qbsp" && -x "$default_tool_path/vis" && -x "$default_tool_path/light" ]]; then
    compile_args+=(--tool-path "$default_tool_path")
  fi

  echo "Compiling BSP2 with pinned profile..."
  cargo "${compile_args[@]}"

  compiled_bsp="$out_dir/${class}-seed-${seed}.bsp"
  compiled_lit="$out_dir/${class}-seed-${seed}.lit"
  cp "$compiled_bsp" "$cache_bsp"
  if [[ -f "$compiled_lit" ]]; then
    cp "$compiled_lit" "$cache_lit"
  fi
  echo "Cached BSP: $cache_bsp"
fi

echo "Launching bsp_beta for $class seed $seed..."
exec cargo run -p bsp_beta -- \
  --bsp "$cache_bsp" \
  --wad "$wad_path" \
  --palette "$palette_tmp" \
  --companion-dir "$companion_dir"
