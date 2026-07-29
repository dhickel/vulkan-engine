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

configure_theme_paths() {
  # Recompute defaults whenever the class changes so CLI and interactive M2
  # runs cannot accidentally retain the Legacy v1 resource roots.
  if [[ "$CLASS" == "m2" ]]; then
    DEFAULT_WAD_PATH="src/bsp_generator/themes/cc0_dungeon_v2/cc0_dungeon_v2.wad"
    DEFAULT_PALETTE_PATH="src/bsp_generator/themes/cc0_dungeon_v2/palette.lmp"
    DEFAULT_TEXTURES_DIR="src/bsp_generator/themes/cc0_dungeon_v2/textures"
  else
    DEFAULT_WAD_PATH="src/bsp_generator/themes/cc0_stone_beta/cc0_stone_beta.wad"
    DEFAULT_PALETTE_PATH="src/bsp_generator/themes/cc0_stone_beta/palette.lmp"
    DEFAULT_TEXTURES_DIR="src/bsp_generator/themes/cc0_stone_beta/textures"
  fi

  WAD_PATH="${DUNGEON_WAD_PATH:-$DEFAULT_WAD_PATH}"
  PALETTE_PATH="${DUNGEON_PALETTE_PATH:-$DEFAULT_PALETTE_PATH}"
  TEXTURES_DIR="${DUNGEON_TEXTURES_DIR:-$DEFAULT_TEXTURES_DIR}"
}

configure_theme_paths
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

# Hash a non-empty texture tree by stable relative name and file bytes. The
# per-file digest framing prevents ambiguity between adjacent file contents.
texture_tree_sha256() {
  local dir="$1"
  [[ -d "$dir" ]] || return 1
  [[ -n "$(find "$dir" -type f -print -quit)" ]] || return 1
  {
    local relative digest
    while IFS= read -r -d '' relative; do
      digest="$(calc_sha256 "$dir/$relative")" || return 1
      printf '%s\0%s\0' "$relative" "$digest"
    done < <(find "$dir" -type f -printf '%P\0' | LC_ALL=C sort -z)
  } | sha256sum | awk '{print $1}'
}

verify_cache() {
  local bsp="$1" lit="$2" manifest="$3"
  [[ -f "$bsp" ]] || return 1
  [[ -f "$manifest" ]] || return 1

  # Check manifest declares a version and it matches current code.
  local stored_version; stored_version="$(grep '^generator_version' "$manifest" 2>/dev/null | awk -F'"' '{print $2}')"
  local current_version; current_version="$(generator_version)"
  [[ "$stored_version" == "$current_version" ]] || { echo "  $(dim "generator changed ($stored_version → $current_version), rebuilding")"; return 1; }

  [[ -f "$lit" ]] || return 1
  local stored_bsp; stored_bsp="$(grep '^bsp\.sha256' "$manifest" 2>/dev/null | awk -F'"' '{print $2}')"
  local stored_lit; stored_lit="$(grep '^lit\.sha256' "$manifest" 2>/dev/null | awk -F'"' '{print $2}')"
  [[ -n "$stored_bsp" && -n "$stored_lit" ]] || return 1
  [[ "$stored_bsp" == "$(calc_sha256 "$bsp")" ]] || return 1
  [[ "$stored_lit" == "$(calc_sha256 "$lit")" ]] || return 1

  local stored_pal; stored_pal="$(grep '^palette\.sha256' "$manifest" 2>/dev/null | awk -F'"' '{print $2}')"
  local stored_wad; stored_wad="$(grep '^wad\.sha256' "$manifest" 2>/dev/null | awk -F'"' '{print $2}')"
  [[ "$stored_pal" == "$(calc_sha256 "$PALETTE_PATH")" ]] || return 1
  [[ "$stored_wad" == "$(calc_sha256 "$WAD_PATH")" ]] || return 1

  local stored_gen; stored_gen="$(grep '^generator[[:space:]]*=' "$manifest" 2>/dev/null | awk -F'"' '{print $2}')"
  if [[ "$CLASS" == "m2" ]]; then
    # M2 Enhanced v2 provenance and the configured texture closure are both
    # mandatory cache inputs. A missing directory or hash is never a cache hit.
    [[ "$stored_gen" == "engine_pack:enhanced-dungeon" ]] || return 1
    local stored_tx; stored_tx="$(grep '^textures_tree\.sha256' "$manifest" 2>/dev/null | awk -F'"' '{print $2}')"
    [[ -n "$stored_tx" ]] || return 1
    local current_tx; current_tx="$(texture_tree_sha256 "$TEXTURES_DIR")" || return 1
    [[ "$stored_tx" == "$current_tx" ]] || return 1
  else
    [[ "$stored_gen" == "dungeon_gen" ]] || return 1
  fi

  return 0
}

generator_version() {
  # Fingerprint every generator source plus all compiler inputs. A curated
  # source list previously omitted serialize.rs, so texture-scale changes kept
  # reusing a scale-1 BSP even after the generator had moved to scale 0.25.
  # For M2 (Enhanced v2), also fingerprint the theme textures/ closure.
  local files=(
    "$repo_root/Cargo.lock"
    "$repo_root/tools/dungeon_explore.sh"
    "$PROFILE_PATH"
    "$WAD_PATH"
    "$PALETTE_PATH"
  )
  while IFS= read -r -d '' f; do
    files+=("$f")
  done < <(
    find \
      "$repo_root/src/bsp" \
      "$repo_root/src/bsp_generator" \
      "$repo_root/src/launch_shared" \
      "$repo_root/tools/dungeon_gen" \
      "$repo_root/tools/engine_pack" \
      -type f \( -name '*.rs' -o -name 'Cargo.toml' \) -print0
  )
  # For M2, fingerprint the effective texture closure with stable relative
  # names and bytes. This catches addition, removal, rename, and replacement.
  if [[ "$CLASS" == "m2" ]]; then
    local texture_tree
    texture_tree="$(texture_tree_sha256 "$TEXTURES_DIR")" || return 1
    files+=("$TEXTURES_DIR")
  fi
  local tool tool_path
  for tool in qbsp vis light; do
    tool_path="$DEFAULT_TOOL_PATH/$tool"
    if [[ ! -x "$tool_path" ]]; then
      tool_path="$(command -v "$tool" 2>/dev/null || true)"
    fi
    [[ -n "$tool_path" ]] && files+=("$tool_path")
  done

  local hash_input=""
  while IFS= read -r -d '' f; do
    [[ -f "$f" ]] && hash_input+="$(sha256sum "$f" | awk '{print $1}')"
  done < <(printf '%s\0' "${files[@]}" | sort -zu)
  if [[ "$CLASS" == "m2" ]]; then
    hash_input+="textures_tree:$texture_tree"
  fi
  echo "$hash_input" | sha256sum | awk '{print $1}' | head -c 16
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

  if [[ "$class" == "m2" ]]; then
    # Enhanced v2: use engine_pack enhanced-dungeon (generate + compile + publish atomically)
    echo "  $(dim "generating+compiling") $(bold "$class") seed $(bold "$seed") via engine_pack..."
    local out_dir="$tmp_dir/out"
    local compile_args=(
      run -q -p engine_pack -- enhanced-dungeon
      --seed "$seed"
      --out "$out_dir"
      --name "${class}-seed-${seed}"
    )
    if [[ -x "$DEFAULT_TOOL_PATH/qbsp" && -x "$DEFAULT_TOOL_PATH/vis" && -x "$DEFAULT_TOOL_PATH/light" ]]; then
      compile_args+=(--tool-path "$DEFAULT_TOOL_PATH")
    fi
    cargo "${compile_args[@]}" || {
      echo "  $(red "✗") engine_pack enhanced-dungeon failed" >&2; return 1
    }

    local published_name="${class}-seed-${seed}"
    local compiled_bsp="$out_dir/${published_name}.bsp"
    local compiled_lit="$out_dir/${published_name}.lit"
    [[ -f "$compiled_bsp" && -f "$compiled_lit" ]] || {
      echo "  $(red "✗") compiler did not produce the required BSP/LIT pair" >&2
      return 1
    }
    rm -f "$bsp" "$lit" "$manifest"
    cp "$compiled_bsp" "$bsp"
    cp "$compiled_lit" "$lit"
    rm -rf "$tmp_dir"
  else
    # Legacy v1: use dungeon_gen + engine_pack compile-bsp
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
    [[ -f "$compiled_bsp" && -f "$compiled_lit" ]] || {
      echo "  $(red "✗") compiler did not produce the required BSP/LIT pair" >&2
      return 1
    }
    rm -f "$bsp" "$lit" "$manifest"
    cp "$compiled_bsp" "$bsp"
    cp "$compiled_lit" "$lit"
    rm -rf "$tmp_dir"
  fi
  trap - RETURN

  local palette_sha256; palette_sha256="$(calc_sha256 "$PALETTE_PATH")"
  local wad_sha256; wad_sha256="$(calc_sha256 "$WAD_PATH")"
  local bsp_sha256; bsp_sha256="$(calc_sha256 "$bsp")"
  local lit_sha256; lit_sha256="$(calc_sha256 "$lit")"

  local gen_id; gen_id="dungeon_gen"
  local textures_tree_line=""
  if [[ "$class" == "m2" ]]; then
    gen_id="engine_pack:enhanced-dungeon"
    local tx_tree; tx_tree="$(texture_tree_sha256 "$TEXTURES_DIR")" || {
      echo "  $(red "✗") Enhanced v2 requires a non-empty texture directory: $TEXTURES_DIR" >&2
      return 1
    }
    textures_tree_line="textures_tree.sha256 = \"$tx_tree\""
  fi

  cat > "$manifest" <<MANIFEST
# Auto-generated dungeon manifest — do not edit manually
[generator]
generator = "$gen_id"
seed = $seed
class = "$class"
generator_version = "$(generator_version)"

[profile]
profile = "ericw-q1-bsp2-generated"

[resources]
palette.sha256 = "$palette_sha256"
wad.path = "$WAD_PATH"
wad.sha256 = "$wad_sha256"
$textures_tree_line
[compiled]
bsp.sha256 = "$bsp_sha256"
lit.sha256 = "$lit_sha256"
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
  printf "  %s Seed:       %s\n" "$(bold "1.")" "$(green "$SEED")"
  printf "  %s Class:      %s  %s\n" "$(bold "2.")" "$(green "$CLASS")" "$(dim "(m1 / m2)")"
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
  configure_theme_paths
  cache_paths "$SEED" "$CLASS"
  if verify_cache "$CACHE_BSP" "$CACHE_LIT" "$CACHE_MANIFEST"; then
    local sz; sz="$(du -h "$CACHE_BSP" | awk '{print $1}')"
    echo "$(green "valid") $(dim "($sz)")"
  elif [[ -f "$CACHE_BSP" ]]; then
    echo "$(red "stale") $(dim "(will rebuild)")"
  else
    echo "$(dim "none") $(dim "(needs build)")"
  fi
}

run_explorer() {
  configure_theme_paths
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

  local args=("$([[ "$MODE" == "development" ]] && echo "--development" || echo "--strict")")
  args+=(--bsp "$bsp" --palette "$PALETTE_PATH" --wad "$WAD_PATH")
  [[ -f "$lit" ]] && args+=(--lit "$lit")
  args+=(--textures "$TEXTURES_DIR")
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
