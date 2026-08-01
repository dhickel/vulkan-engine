#!/usr/bin/env bash
set -euo pipefail

# ── paths ──────────────────────────────────────────────────────────────────
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
cd "$repo_root"

CACHE_ROOT=".internal-dev/captures/bsp-dungeon-generator"
M1_WAD="src/bsp_generator/themes/cc0_stone_beta/cc0_stone_beta.wad"
M1_PALETTE="src/bsp_generator/themes/cc0_stone_beta/palette.lmp"
M1_TEXTURES="src/bsp_generator/themes/cc0_stone_beta/textures"
M2_WAD="src/bsp_generator/themes/cc0_dungeon_v2/cc0_dungeon_v2.wad"
M2_PALETTE="src/bsp_generator/themes/cc0_dungeon_v2/palette.lmp"
M2_TEXTURES="src/bsp_generator/themes/cc0_dungeon_v2/textures"
DEFAULT_TOOL_PATH="${DUNGEON_TOOL_PATH:-$HOME/.local/ericw-tools/ericw-tools-2.0.0-alpha3-Linux/bin}"
PROFILE_PATH="${DUNGEON_PROFILE_PATH:-tools/bsp_authoring/ericw-q1-bsp2-generated-profile.toml}"

# ── defaults ───────────────────────────────────────────────────────────────
MODE="architectural"
SEED=""
PRESET="moderate"
ROOMS=""
CORRIDORS=""
LOOPS=""
CHAMFER_FLAG="--chamfer"
ARCH_TYPE=""
GRAMMAR_FAMILIES=""
DEVELOPMENT=""
BUST=""
CACHE_ONLY=""

# ── helpers ────────────────────────────────────────────────────────────────
bold()  { printf '\033[1m%s\033[0m' "$*"; }
green() { printf '\033[32m%s\033[0m' "$*"; }
red()   { printf '\033[31m%s\033[0m' "$*"; }
dim()   { printf '\033[2m%s\033[0m' "$*"; }
die()   { printf '%s\n' "$(red "error"): $*" >&2; exit 1; }
calc_sha256() { sha256sum "$1" | awk '{print $1}'; }

texture_tree_sha256() {
  local dir="$1"
  [[ -d "$dir" ]] || return 1
  [[ -n "$(find "$dir" -type f -print -quit 2>/dev/null)" ]] || return 1
  {
    local relative digest
    while IFS= read -r -d '' relative; do
      digest="$(calc_sha256 "$dir/$relative")" || return 1
      printf '%s\0%s\0' "$relative" "$digest"
    done < <(find "$dir" -type f -printf '%P\0' | LC_ALL=C sort -z)
  } | sha256sum | awk '{print $1}'
}

generator_version() {
  local files=("$repo_root/Cargo.lock" "$PROFILE_PATH" "$M2_WAD" "$M2_PALETTE")
  while IFS= read -r -d '' f; do
    files+=("$f")
  done < <(find "$repo_root/src/bsp" "$repo_root/src/bsp_generator" \
    "$repo_root/src/launch_shared" "$repo_root/tools/dungeon_gen" \
    "$repo_root/tools/engine_pack" \
    -type f \( -name '*.rs' -o -name 'Cargo.toml' \) -print0)
  local texture_tree
  texture_tree="$(texture_tree_sha256 "$M2_TEXTURES")" || return 1
  local tool tool_path
  for tool in qbsp vis light; do
    tool_path="$DEFAULT_TOOL_PATH/$tool"
    [[ -x "$tool_path" ]] || tool_path="$(command -v "$tool" 2>/dev/null || true)"
    [[ -n "$tool_path" ]] && files+=("$tool_path")
  done
  local hash_input=""
  while IFS= read -r -d '' f; do
    [[ -f "$f" ]] && hash_input+="$(sha256sum "$f" | awk '{print $1}')"
  done < <(printf '%s\0' "${files[@]}" | sort -zu)
  hash_input+="textures_tree:$texture_tree"
  echo "$hash_input" | sha256sum | awk '{print $1}' | head -c 16
}

verify_cache() {
  local bsp="$1" lit="$2" manifest="$3" wad="$4" palette="$5" textures="$6" gen_label="$7"
  [[ -f "$bsp" && -f "$lit" && -f "$manifest" ]] || return 1

  local bsp_header; bsp_header="$(head -c 4 "$bsp" 2>/dev/null || true)"
  [[ "$bsp_header" == "BSP2" ]] || { echo "  $(dim "BSP2 magic missing, invalid cache")"; return 1; }
  local lit_header; lit_header="$(head -c 4 "$lit" 2>/dev/null || true)"
  [[ "$lit_header" == "QLIT" ]] || { echo "  $(dim "QLIT magic missing, invalid cache")"; return 1; }

  local stored_version; stored_version="$(grep '^generator_version' "$manifest" 2>/dev/null | awk -F'"' '{print $2}')"
  local current_version; current_version="$(generator_version)"
  [[ "$stored_version" == "$current_version" ]] || { echo "  $(dim "generator changed ($stored_version → $current_version), rebuilding")"; return 1; }

  local stored_gen; stored_gen="$(grep '^generator[[:space:]]*=' "$manifest" 2>/dev/null | awk -F'"' '{print $2}')"
  [[ "$stored_gen" == "$gen_label" ]] || return 1

  local stored_bsp; stored_bsp="$(grep '^bsp\.sha256' "$manifest" 2>/dev/null | awk -F'"' '{print $2}')"
  local stored_lit; stored_lit="$(grep '^lit\.sha256' "$manifest" 2>/dev/null | awk -F'"' '{print $2}')"
  [[ -n "$stored_bsp" && -n "$stored_lit" ]] || return 1
  [[ "$stored_bsp" == "$(calc_sha256 "$bsp")" ]] || return 1
  [[ "$stored_lit" == "$(calc_sha256 "$lit")" ]] || return 1

  local stored_pal; stored_pal="$(grep '^palette\.sha256' "$manifest" 2>/dev/null | awk -F'"' '{print $2}')"
  local stored_wad; stored_wad="$(grep '^wad\.sha256' "$manifest" 2>/dev/null | awk -F'"' '{print $2}')"
  [[ "$stored_pal" == "$(calc_sha256 "$palette")" ]] || return 1
  [[ "$stored_wad" == "$(calc_sha256 "$wad")" ]] || return 1

  local stored_tx; stored_tx="$(grep '^textures_tree\.sha256' "$manifest" 2>/dev/null | awk -F'"' '{print $2}')"
  [[ -n "$stored_tx" ]] || return 1
  local current_tx; current_tx="$(texture_tree_sha256 "$textures")" || return 1
  [[ "$stored_tx" == "$current_tx" ]] || return 1
  return 0
}

bust_cache() {
  local seed="$1" label="$2"
  rm -f "$CACHE_ROOT/${label}-seed-${seed}.bsp" "$CACHE_ROOT/${label}-seed-${seed}.lit" "$CACHE_ROOT/${label}-seed-${seed}.manifest.toml"
  echo "$(green "✓") Cache busted for $(bold "$label seed $seed")"
}

build_cache() {
  local seed="$1" label="$2" wad="$3" palette="$4" textures="$5" gen_label="$6"
  local cache_bsp="$CACHE_ROOT/${label}-seed-${seed}.bsp"
  local cache_lit="$CACHE_ROOT/${label}-seed-${seed}.lit"
  local cache_manifest="$CACHE_ROOT/${label}-seed-${seed}.manifest.toml"
  local tmp_dir; tmp_dir="$(mktemp -d -t dungeon-explore-${label}-${seed}-XXXXXX)"
  trap 'rm -rf "$tmp_dir"' RETURN
  mkdir -p "$CACHE_ROOT"

  local candidate_bsp candidate_lit
  if [[ "$label" == "enhanced" ]]; then
    echo "  $(dim "generating+compiling") $(bold "$label") seed $(bold "$seed") via engine_pack..."
    local out_dir="$tmp_dir/out"
    local compile_args=(run -q -p engine_pack -- enhanced-dungeon --seed "$seed" --out "$out_dir" --name "${label}-seed-${seed}")
    [[ -x "$DEFAULT_TOOL_PATH/qbsp" && -x "$DEFAULT_TOOL_PATH/vis" && -x "$DEFAULT_TOOL_PATH/light" ]] && compile_args+=(--tool-path "$DEFAULT_TOOL_PATH")
    cargo "${compile_args[@]}" || { echo "  $(red "✗") engine_pack enhanced-dungeon failed" >&2; return 1; }
    candidate_bsp="$out_dir/${label}-seed-${seed}.bsp"
    candidate_lit="$out_dir/${label}-seed-${seed}.lit"
  else
    echo "  $(dim "generating") $(bold "$label") seed $(bold "$seed")..."
    local map_path="$tmp_dir/${label}-seed-${seed}.map"
    cargo run -q -p dungeon_gen -- --seed "$seed" --class "m1" --out "$map_path" || { echo "  $(red "✗") generation failed" >&2; return 1; }

    local out_dir="$tmp_dir/compiled"
    local compile_args=(run -q -p engine_pack -- compile-bsp "$map_path" --profile "$PROFILE_PATH" --out "$out_dir" --palette "$palette" --wad "$wad")
    [[ -x "$DEFAULT_TOOL_PATH/qbsp" && -x "$DEFAULT_TOOL_PATH/vis" && -x "$DEFAULT_TOOL_PATH/light" ]] && compile_args+=(--tool-path "$DEFAULT_TOOL_PATH")
    echo "  $(dim "compiling") $(bold "BSP2")..."
    cargo "${compile_args[@]}" || { echo "  $(red "✗") compilation failed" >&2; return 1; }
    candidate_bsp="$out_dir/${label}-seed-${seed}.bsp"
    candidate_lit="$out_dir/${label}-seed-${seed}.lit"
  fi

  [[ -f "$candidate_bsp" && -f "$candidate_lit" ]] || { echo "  $(red "✗") compiler did not produce BSP/LIT pair" >&2; return 1; }

  local palette_sha256; palette_sha256="$(calc_sha256 "$palette")"
  local wad_sha256; wad_sha256="$(calc_sha256 "$wad")"
  local bsp_sha256; bsp_sha256="$(calc_sha256 "$candidate_bsp")"
  local lit_sha256; lit_sha256="$(calc_sha256 "$candidate_lit")"
  local tx_tree; tx_tree="$(texture_tree_sha256 "$textures")" || { echo "  $(red "✗") texture tree empty: $textures" >&2; return 1; }

  local candidate_manifest="$tmp_dir/cache.manifest.toml"
  cat > "$candidate_manifest" <<MANIFEST
# Auto-generated dungeon manifest — do not edit
[generator]
generator = "$gen_label"
seed = $seed
class = "$label"
generator_version = "$(generator_version)"

[profile]
profile = "ericw-q1-bsp2-generated"

[resources]
palette.sha256 = "$palette_sha256"
wad.path = "$wad"
wad.sha256 = "$wad_sha256"
textures_tree.sha256 = "$tx_tree"

[compiled]
bsp.sha256 = "$bsp_sha256"
lit.sha256 = "$lit_sha256"
MANIFEST

  local stage; stage="$(mktemp -d "$CACHE_ROOT/.${label}-seed-${seed}.XXXXXX")"
  cp "$candidate_bsp" "$stage/bsp"; cp "$candidate_lit" "$stage/lit"; cp "$candidate_manifest" "$stage/manifest.toml"
  mv -f "$stage/bsp" "$cache_bsp"; mv -f "$stage/lit" "$cache_lit"; mv -f "$stage/manifest.toml" "$cache_manifest"
  rmdir "$stage"; rm -rf "$tmp_dir"; trap - RETURN
  echo "  $(green "✓") cached $(bold "$(basename "$cache_bsp")") ($(du -h "$cache_bsp" | awk '{print $1}'))"
}

# ── help ───────────────────────────────────────────────────────────────────
show_help() {
  echo "Usage: ./tools/dungeon_explore.sh [MODE] [OPTIONS]"
  echo ""
  echo "Dungeon exploration launcher for the BSP engine."
  echo ""
  echo "Modes (default: architectural):"
  echo "  --classic (m1)       Legacy v1 single-layer, theme cc0_stone_beta"
  echo "  --enhanced (m2)      Enhanced v2 two-layer, theme cc0_dungeon_v2"
  echo "  --architectural (m3) Enhanced v3 grammar dungeons, theme cc0_dungeon_v2"
  echo ""
  echo "Options:"
  echo "  --seed N             Generation seed (default: 0 classic/enhanced,"
  echo "                       random architectural)"
  echo "  --preset sparse|moderate|rich  Architectural preset (default: moderate)"
  echo "  --rooms N            Room-count override (architectural only)"
  echo "  --corridors N        Corridor-count override (architectural only)"
  echo "  --loops N            Loop-count override (architectural only)"
  echo "  --chamfer / --no-chamfer       Toggle chamfer (architectural only)"
  echo "  --arch-type none|pointed|segmented  Portal arch type (architectural)"
  echo "  --grammar-families a,b,c       Grammar allowlist (architectural only)"
  echo "  --strict             Pass --strict to bsp_beta (default classic/enhanced)"
  echo "  --development        Pass --development to bsp_beta"
  echo "  --bust               Force cache rebuild (classic/enhanced only)"
  echo "  --cache-only         Only ensure cache exists, don't launch"
  echo "  --help, -h           Show this help"
  exit 0
}

# ── parse args ─────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --classic|--m1)       MODE="classic"; shift ;;
    --enhanced|--m2)      MODE="enhanced"; shift ;;
    --architectural|--m3) MODE="architectural"; shift ;;
    --seed)               SEED="$2"; shift 2 ;;
    --preset)             PRESET="$2"; shift 2 ;;
    --rooms)              ROOMS="$2"; shift 2 ;;
    --corridors)          CORRIDORS="$2"; shift 2 ;;
    --loops)              LOOPS="$2"; shift 2 ;;
    --chamfer)            CHAMFER_FLAG="--chamfer"; shift ;;
    --no-chamfer)         CHAMFER_FLAG="--no-chamfer"; shift ;;
    --arch-type)          ARCH_TYPE="$2"; shift 2 ;;
    --grammar-families)   GRAMMAR_FAMILIES="$2"; shift 2 ;;
    --development)        DEVELOPMENT="--development"; shift ;;
    --strict)             DEVELOPMENT="--strict"; shift ;;
    --bust)               BUST="1"; shift ;;
    --cache-only)         CACHE_ONLY="1"; shift ;;
    --help|-h)            show_help ;;
    *)                    die "unknown argument: $1 (use --help)" ;;
  esac
done

# ── classic / enhanced: cache-backed launch ─────────────────────────────────
if [[ "$MODE" == "classic" || "$MODE" == "enhanced" ]]; then
  SEED="${SEED:-0}"
  WAD="$M2_WAD"; PALETTE="$M2_PALETTE"; TEXTURES="$M2_TEXTURES"; GEN_LABEL="engine_pack:enhanced-dungeon"
  if [[ "$MODE" == "classic" ]]; then
    WAD="$M1_WAD"; PALETTE="$M1_PALETTE"; TEXTURES="$M1_TEXTURES"; GEN_LABEL="dungeon_gen"
  fi

  BSP="$CACHE_ROOT/${MODE}-seed-${SEED}.bsp"
  LIT="$CACHE_ROOT/${MODE}-seed-${SEED}.lit"
  MANIFEST="$CACHE_ROOT/${MODE}-seed-${SEED}.manifest.toml"

  [[ -n "$BUST" ]] && bust_cache "$SEED" "$MODE"

  if ! verify_cache "$BSP" "$LIT" "$MANIFEST" "$WAD" "$PALETTE" "$TEXTURES" "$GEN_LABEL"; then
    echo ""
    echo "$(bold "Building cache...")"
    build_cache "$SEED" "$MODE" "$WAD" "$PALETTE" "$TEXTURES" "$GEN_LABEL" || die "build failed"
    echo ""
  else
    echo "$(green "✓") using cached $(basename "$BSP")"
  fi

  [[ -n "$CACHE_ONLY" ]] && { echo "$(green "✓") cache ready: $BSP"; exit 0; }

  IMPORT_MODE="--strict"
  [[ -n "$DEVELOPMENT" ]] && IMPORT_MODE="$DEVELOPMENT"
  LAUNCH_ARGS=("$IMPORT_MODE" "--bsp" "$BSP" "--palette" "$PALETTE" "--wad" "$WAD" "--textures" "$TEXTURES")
  [[ -f "$LIT" ]] && LAUNCH_ARGS+=(--lit "$LIT")
  echo ""
  echo "  $(bold "Launching") $(green "$MODE") seed $(bold "$SEED")..."
  exec cargo run -p bsp_beta -- "${LAUNCH_ARGS[@]}"
fi

# ── architectural: direct GUI launch (default) ─────────────────────────────
M3_MODE="--development"
[[ -n "$DEVELOPMENT" ]] && M3_MODE="$DEVELOPMENT"
M3_ARGS=("$M3_MODE" "--m3-generate" "--preset" "$PRESET")
[[ -n "$SEED" ]] && M3_ARGS+=(--seed "$SEED")
[[ -n "$ROOMS" ]] && M3_ARGS+=(--rooms "$ROOMS")
[[ -n "$CORRIDORS" ]] && M3_ARGS+=(--corridors "$CORRIDORS")
[[ -n "$LOOPS" ]] && M3_ARGS+=(--loops "$LOOPS")
[[ "$CHAMFER_FLAG" == "--no-chamfer" ]] && M3_ARGS+=(--no-chamfer)
[[ -n "$ARCH_TYPE" ]] && M3_ARGS+=(--arch-type "$ARCH_TYPE")
[[ -n "$GRAMMAR_FAMILIES" ]] && M3_ARGS+=(--grammar-families "$GRAMMAR_FAMILIES")

echo "$(green "✓") Launching $(bold "architectural") (m3) with GUI editor..."
exec cargo run -p bsp_beta -- "${M3_ARGS[@]}"
