#!/usr/bin/env bash
set -euo pipefail

# ─── defaults ───────────────────────────────────────────────────────────────
SEED="${DUNGEON_SEED:-0}"
CLASS="${DUNGEON_CLASS:-m1}"
MODE="${DUNGEON_MODE:-strict}"
CAMERA="${DUNGEON_CAMERA:-}"
STATS="${DUNGEON_STATS:-}"
ALL_VISIBLE="${DUNGEON_ALL_VISIBLE:-}"
CACHE_ONLY="${DUNGEON_CACHE_ONLY:-}"
BUST="${DUNGEON_BUST:-}"

# ─── paths ──────────────────────────────────────────────────────────────────
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
cd "$repo_root"

cache_root=".internal-dev/captures/bsp-dungeon-generator"

WAD_PATH="${DUNGEON_WAD_PATH:-src/bsp_generator/themes/cc0_stone_beta/cc0_stone_beta.wad}"
PALETTE_PATH="${DUNGEON_PALETTE_PATH:-src/bsp_generator/themes/cc0_stone_beta/palette.lmp}"
TEXTURES_DIR="${DUNGEON_TEXTURES_DIR:-src/bsp_generator/themes/cc0_stone_beta/textures}"
PROFILE_PATH="${DUNGEON_PROFILE_PATH:-tools/bsp_authoring/ericw-q1-bsp2-generated-profile.toml}"
DEFAULT_TOOL_PATH="${DUNGEON_TOOL_PATH:-$HOME/.local/ericw-tools/ericw-tools-2.0.0-alpha3-Linux/bin}"

# ─── helpers ────────────────────────────────────────────────────────────────
bold()  { printf '\033[1m%s\033[0m' "$*"; }
green() { printf '\033[32m%s\033[0m' "$*"; }
red()   { printf '\033[31m%s\033[0m' "$*"; }
dim()   { printf '\033[2m%s\033[0m' "$*"; }
clear_screen() { printf '\033[2J\033[H'; }

die() { echo "$(red "error"): $*" >&2; exit 1; }

cache_paths() {
  local seed="$1" class="$2"
  mkdir -p "$cache_root"
  CACHE_BSP="$cache_root/${class}-seed-${seed}.bsp"
  CACHE_LIT="$cache_root/${class}-seed-${seed}.lit"
  CACHE_MANIFEST="$cache_root/${class}-seed-${seed}.manifest.toml"
}

calc_sha256() { sha256sum "$1" | awk '{print $1}'; }

verify_cache() {
  local bsp="$1" lit="$2" manifest="$3"
  [[ -f "$bsp" ]] || return 1
  [[ -f "$manifest" ]] || return 1

  local stored
  stored="$(grep '^bsp\.sha256' "$manifest" 2>/dev/null | awk -F'"' '{print $2}')"
  [[ -n "$stored" ]] || return 1
  local actual; actual="$(calc_sha256 "$bsp")"
  [[ "$stored" == "$actual" ]] || return 1

  local stored_pal; stored_pal="$(grep '^palette\.sha256' "$manifest" 2>/dev/null | awk -F'"' '{print $2}')"
  local actual_pal; actual_pal="$(calc_sha256 "$PALETTE_PATH")"
  [[ "$stored_pal" == "$actual_pal" ]] || return 1

  local stored_gen; stored_gen="$(grep '^generator' "$manifest" 2>/dev/null | awk -F'"' '{print $2}')"
  [[ "$stored_gen" == "dungeon_gen" ]] || return 1

  return 0
}

bust_cache() {
  local seed="$1" class="$2"
  cache_paths "$seed" "$class"
  rm -f "$CACHE_BSP" "$CACHE_LIT" "$CACHE_MANIFEST"
  echo "$(green "✓") Cache busted for $(bold "$class seed $seed")"
}

build_cache() {
  local seed="$1" class="$2"
  cache_paths "$seed" "$class"
  local bsp="$CACHE_BSP" lit="$CACHE_LIT" manifest="$CACHE_MANIFEST"

  local tmp_dir; tmp_dir="$(mktemp -d -t dungeon-explore-${class}-${seed}-XXXXXX)"
  trap 'rm -rf "$tmp_dir"' RETURN

  local map_path="$tmp_dir/${class}-seed-${seed}.map"
  local out_dir="$tmp_dir/compiled"

  echo "  $(dim "generating") $(bold "$class") seed $(bold "$seed")..."
  cargo run -q -p dungeon_gen -- --seed "$seed" --class "$class" --out "$map_path" || {
    echo "  $(red "✗") generation failed" >&2; return 1
  }

  local compile_args=(
    run -q -p engine_pack -- compile-bsp "$map_path"
    --profile "$PROFILE_PATH"
    --out "$out_dir"
    --palette "$PALETTE_PATH"
    --wad "$WAD_PATH"
  )
  if [[ -x "$DEFAULT_TOOL_PATH/qbsp" && -x "$DEFAULT_TOOL_PATH/vis" && -x "$DEFAULT_TOOL_PATH/light" ]]; then
    compile_args+=(--tool-path "$DEFAULT_TOOL_PATH")
  fi

  echo "  $(dim "compiling") $(bold "BSP2")..."
  cargo "${compile_args[@]}" || {
    echo "  $(red "✗") compilation failed" >&2; return 1
  }

  local compiled_bsp="$out_dir/${class}-seed-${seed}.bsp"
  local compiled_lit="$out_dir/${class}-seed-${seed}.lit"
  cp "$compiled_bsp" "$bsp"
  [[ -f "$compiled_lit" ]] && cp "$compiled_lit" "$lit"
  rm -rf "$tmp_dir"
  trap - RETURN

  local palette_sha256; palette_sha256="$(calc_sha256 "$PALETTE_PATH")"
  local bsp_sha256; bsp_sha256="$(calc_sha256 "$bsp")"
  cat > "$manifest" <<MANIFEST
# Auto-generated dungeon manifest — do not edit manually
[generator]
generator = "dungeon_gen"
seed = $seed
class = "$class"

[profile]
profile = "ericw-q1-bsp2-generated"

[resources]
palette.sha256 = "$palette_sha256"
wad.path = "$WAD_PATH"

[compiled]
bsp.sha256 = "$bsp_sha256"
MANIFEST

  echo "  $(green "✓") cached $(bold "$(basename "$bsp")") ($(du -h "$bsp" | awk '{print $1}'))"
}

status_badge() {
  local mode="$1"
  if [[ "$mode" == "strict" ]]; then
    echo "$(red "●") strict"
  else
    echo "$(dim "○") development"
  fi
}

# ─── menu UI ────────────────────────────────────────────────────────────────
draw_menu() {
  clear_screen
  echo "  $(bold "Dungeon Explorer")"
  echo "  $(dim "─────────────────────────────────────────────")"
  echo ""
  printf "  $(bold "1.") Seed:       $(green "%s" "$SEED")\n" "$SEED"
  printf "  $(bold "2.") Class:      $(green "%s" "$CLASS")  $(dim "(m1 / m2)")\n"
  printf "  $(bold "3.") Mode:       %s\n" "$(status_badge "$MODE")"
  printf "  $(bold "4.") Camera:     %s  $(dim "('' / spawn / corridor / junction)")\n" "${CAMERA:-(default)}"
  printf "  $(bold "5.") Stats:      %s  $(dim "(set to '1' or leave empty)")\n" "${STATS:-(off)}"
  printf "  $(bold "6.") All-Visible:%s  $(dim "(set to '1' or leave empty)")\n" "${ALL_VISIBLE:-(off)}"
  echo ""
  printf "  $(bold "c.") Cache:      %s\n" "$(cache_status_line)"
  printf "  $(bold "x.") Bust cache\n"
  echo ""
  printf "  $(bold "R.") $(green "Run")  $(bold "Q.") Quit\n"
  echo ""
  printf "  $(dim "─────────────────────────────────────────────")\n"
  printf "  Choice: "
}

cache_status_line() {
  cache_paths "$SEED" "$CLASS"
  if verify_cache "$CACHE_BSP" "$CACHE_LIT" "$CACHE_MANIFEST"; then
    local sz; sz="$(du -h "$CACHE_BSP" | awk '{print $1}')"
    echo "$(green "valid") $(dim "($sz)")"
  elif [[ -f "$bsp" ]]; then
    echo "$(red "stale") $(dim "(will rebuild)")"
  else
    echo "$(dim "none") $(dim "(needs build)")"
  fi
}

run_explorer() {
  cache_paths "$SEED" "$CLASS"
  local bsp="$CACHE_BSP" lit="$CACHE_LIT" manifest="$CACHE_MANIFEST"

  # Build or verify cache
  if [[ -n "$BUST" ]] || ! verify_cache "$bsp" "$lit" "$manifest"; then
    if [[ -n "$BUST" ]]; then
      bust_cache "$SEED" "$CLASS"
    fi
    echo ""
    echo "$(bold "Building cache...")"
    build_cache "$SEED" "$CLASS" || die "build failed"
    echo ""
  else
    echo "$(green "✓") using cached $(basename "$bsp")"
  fi

  if [[ -n "$CACHE_ONLY" ]]; then
    echo "$(green "✓") cache ready: $bsp"
    return 0
  fi

  local args=(--bsp "$bsp" --palette "$PALETTE_PATH" --wad "$WAD_PATH")
  [[ -f "$lit" ]] && args+=(--lit "$lit")
  args+=(--textures "$TEXTURES_DIR")
  [[ "$MODE" == "development" ]] && args+=(--development)
  [[ -n "$CAMERA" ]] && args+=(--acceptance-camera "$CAMERA")
  [[ -n "$STATS" ]] && args+=(--stats)
  [[ -n "$ALL_VISIBLE" ]] && args+=(--all-visible)

  echo ""
  echo "  $(bold "Launching") $(green "$MODE") mode — $(bold "$CLASS") seed $(bold "$SEED")..."
  echo "  $(dim "─────────────────────────────────────────────")"
  echo ""

  exec cargo run -p bsp_beta -- "${args[@]}"
}

# ─── main loop ──────────────────────────────────────────────────────────────
interactive() {
  while true; do
    draw_menu
    read -r choice
    choice="${choice,,}"
    case "$choice" in
      1)
        printf "  Enter seed (0-255): "
        read -r val
        [[ "$val" =~ ^[0-9]+$ ]] || { echo "  $(red "✗") invalid seed"; sleep 1; continue; }
        [[ "$val" -le 255 ]] || { echo "  $(red "✗") seed must be ≤ 255"; sleep 1; continue; }
        SEED="$val"
        BUST=""  # don't auto-bust; user can bust manually or run to rebuild if stale
        ;;
      2)
        printf "  Enter class (m1/m2): "
        read -r val
        if [[ "$val" == "m1" || "$val" == "m2" ]]; then
          CLASS="$val"
        else
          echo "  $(red "✗") class must be m1 or m2"; sleep 1
        fi
        ;;
      3)
        if [[ "$MODE" == "strict" ]]; then MODE="development"; else MODE="strict"; fi
        ;;
      4)
        printf "  Camera (empty / spawn / corridor / junction): "
        read -r val
        CAMERA="$val"
        ;;
      5)
        if [[ -z "$STATS" ]]; then STATS="1"; else STATS=""; fi
        ;;
      6)
        if [[ -z "$ALL_VISIBLE" ]]; then ALL_VISIBLE="1"; else ALL_VISIBLE=""; fi
        ;;
      c)
        cache_paths "$SEED" "$CLASS"
        if [[ -f "$CACHE_BSP" ]]; then
          echo "  BSP:   $CACHE_BSP ($(du -h "$CACHE_BSP" | awk '{print $1}'))"
          echo "  LIT:   $CACHE_LIT ($([[ -f "$CACHE_LIT" ]] && du -h "$CACHE_LIT" | awk '{print $1}' || echo "none"))"
          echo "  SHA-256: $(calc_sha256 "$CACHE_BSP")"
        else
          echo "  $(dim "no cache for $CLASS seed $SEED")"
        fi
        echo ""
        printf "  Press Enter..."
        read -r
        ;;
      x)
        printf "  Bust cache for $(bold "$CLASS seed $SEED")? (y/N) "
        read -r yn
        if [[ "${yn,,}" == "y" ]]; then
          bust_cache "$SEED" "$CLASS"
          BUST=""
        fi
        ;;
      r)  BUST=""; run_explorer; return ;;
      q)  echo ""; exit 0 ;;
      *)  ;;
    esac
  done
}

# ─── dispatch ───────────────────────────────────────────────────────────────
if [[ $# -eq 0 ]]; then
  interactive
else
  # CLI mode — parse args for seed, class, flags
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --seed)       SEED="$2"; shift 2 ;;
      --class)      CLASS="$2"; shift 2 ;;
      --strict)     MODE="strict"; shift ;;
      --development) MODE="development"; shift ;;
      --camera)     CAMERA="$2"; shift 2 ;;
      --stats)      STATS="1"; shift ;;
      --all-visible) ALL_VISIBLE="1"; shift ;;
      --bust)       BUST="1"; shift ;;
      --cache-only) CACHE_ONLY="1"; shift ;;
      -h|--help)
        echo "Usage: ./tools/dungeon_explore.sh [options]"
        echo ""
        echo "Interactive mode (default):"
        echo "  ./tools/dungeon_explore.sh"
        echo ""
        echo "CLI options:"
        echo "  --seed <0-255>     Seed value (default: 0)"
        echo "  --class <m1|m2>    Dungeon class (default: m1)"
        echo "  --strict            Strict mode (default)"
        echo "  --development       Development mode"
        echo "  --camera <label>    Acceptance camera (spawn/corridor/junction)"
        echo "  --stats             Request runtime draw evidence"
        echo "  --all-visible       All-visible mode"
        echo "  --bust              Force cache rebuild"
        echo "  --cache-only        Only ensure cache exists (don't launch)"
        echo ""
        echo "Environment overrides: DUNGEON_SEED, DUNGEON_CLASS, DUNGEON_MODE,"
        echo "  DUNGEON_CAMERA, DUNGEON_STATS, DUNGEON_ALL_VISIBLE, DUNGEON_BUST,"
        echo "  DUNGEON_WAD_PATH, DUNGEON_PALETTE_PATH, DUNGEON_TEXTURES_DIR,"
        echo "  DUNGEON_PROFILE_PATH, DUNGEON_TOOL_PATH"
        exit 0
        ;;
      *) die "unknown argument: $1 (use -h for help)" ;;
    esac
  done
  run_explorer
fi
