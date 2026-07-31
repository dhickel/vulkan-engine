#!/usr/bin/env bash
set -euo pipefail
# ── dungeon_explore_v3_cache.sh ────────────────────────────────────────────
# Validate M3 cache integrity: hit, bust, regeneration, invalid rejection.
#
# Usage:
#   bash tools/tests/dungeon_explore_v3_cache.sh
#
# Requires: engine_pack enhanced-dungeon-v3, cargo, sha256sum, bash 4+

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../.." && pwd)"
cd "$repo_root"

# ── helpers ────────────────────────────────────────────────────────────────
bold()  { printf '\033[1m%s\033[0m' "$*"; }
green() { printf '\033[32m%s\033[0m' "$*"; }
red()   { printf '\033[31m%s\033[0m' "$*"; }
dim()   { printf '\033[2m%s\033[0m' "$*"; }

pass() { echo "  $(green "✓") PASS: $*"; }
fail() { echo "  $(red "✗") FAIL: $*" >&2; FAILURES=$((FAILURES + 1)); }
fail_count() { FAILURES=0; }

cache_root=".internal-dev/captures/bsp-dungeon-generator"
explore_script="tools/dungeon_explore.sh"
TEST_SEED="42"
TEST_CLASS="m3"

calc_sha256() { sha256sum "$1" | awk '{print $1}'; }

# ── pre-flight ─────────────────────────────────────────────────────────────
echo "$(bold "M3 Cache Validation")"
echo "$(dim "─────────────────────────────────────────────")"
echo ""

# Ensure the cache root exists
mkdir -p "$cache_root"

cache_paths() {
  CACHE_BSP="$cache_root/${TEST_CLASS}-seed-${TEST_SEED}.bsp"
  CACHE_LIT="$cache_root/${TEST_CLASS}-seed-${TEST_SEED}.lit"
  CACHE_MANIFEST="$cache_root/${TEST_CLASS}-seed-${TEST_SEED}.manifest.toml"
}

# Save any pre-existing cache so we can restore it after tests
cache_paths
RESTORE_BSP=""
RESTORE_LIT=""
RESTORE_MANIFEST=""
if [[ -f "$CACHE_BSP" ]]; then
  RESTORE_BSP="$(mktemp)"
  cp "$CACHE_BSP" "$RESTORE_BSP"
fi
if [[ -f "$CACHE_LIT" ]]; then
  RESTORE_LIT="$(mktemp)"
  cp "$CACHE_LIT" "$RESTORE_LIT"
fi
if [[ -f "$CACHE_MANIFEST" ]]; then
  RESTORE_MANIFEST="$(mktemp)"
  cp "$CACHE_MANIFEST" "$RESTORE_MANIFEST"
fi

cleanup() {
  rm -f "$RESTORE_BSP" "$RESTORE_LIT" "$RESTORE_MANIFEST"
}
trap cleanup EXIT

# ── Test 1: Cache hit ──────────────────────────────────────────────────────
fail_count
echo "$(bold "Test 1:") Cache hit"

# Ensure clean slate
rm -f "$CACHE_BSP" "$CACHE_LIT" "$CACHE_MANIFEST"

# First run builds the cache
echo "  $(dim "Building initial cache...")"
DUNGEON_CLASS=m3 DUNGEON_SEED="$TEST_SEED" DUNGEON_CACHE_ONLY=1 \
  bash "$explore_script" --class m3 --seed "$TEST_SEED" --cache-only 2>&1 || {
  fail "initial cache build failed"; exit 1
}

[[ -f "$CACHE_BSP" ]] || { fail "cache BSP not created"; exit 1; }
[[ -f "$CACHE_LIT" ]] || { fail "cache LIT not created"; exit 1; }
[[ -f "$CACHE_MANIFEST" ]] || { fail "cache manifest not created"; exit 1; }

# Verify BSP2 and QLIT magic
bsp_header="$(head -c 4 "$CACHE_BSP")"
lit_header="$(head -c 4 "$CACHE_LIT")"
[[ "$bsp_header" == "BSP2" ]] || { fail "BSP2 magic missing (got: $(echo "$bsp_header" | od -A x -t x1z))"; }
[[ "$lit_header" == "QLIT" ]] || { fail "QLIT magic missing (got: $(echo "$lit_header" | od -A x -t x1z))"; }

# Verify manifest integrity
generator="$(grep '^generator[[:space:]]*=' "$CACHE_MANIFEST" | awk -F'"' '{print $2}')"
[[ "$generator" == "engine_pack:enhanced-dungeon-v3" ]] || {
  fail "manifest generator mismatch: $generator"
}

preset="$(grep '^preset' "$CACHE_MANIFEST" | awk -F'"' '{print $2}')"
[[ "$preset" == "moderate" ]] || { fail "manifest preset mismatch: $preset"; }

# Record hashes for tamper test
ORIG_BSP_SHA="$(calc_sha256 "$CACHE_BSP")"
ORIG_LIT_SHA="$(calc_sha256 "$CACHE_LIT")"

# Second run should use cache (no rebuild)
echo "  $(dim "Checking cache hit...")"
output="$(DUNGEON_CLASS=m3 DUNGEON_SEED="$TEST_SEED" DUNGEON_CACHE_ONLY=1 \
  bash "$explore_script" --class m3 --seed "$TEST_SEED" --cache-only 2>&1)" || {
  fail "cache hit run failed"; exit 1
}

# Cache should not have been rebuilt (sha unchanged)
[[ "$(calc_sha256 "$CACHE_BSP")" == "$ORIG_BSP_SHA" ]] || {
  fail "cache was rebuilt on hit (BSP sha changed)"
}
[[ "$(calc_sha256 "$CACHE_LIT")" == "$ORIG_LIT_SHA" ]] || {
  fail "cache was rebuilt on hit (LIT sha changed)"
}

if [[ $FAILURES -eq 0 ]]; then
  pass "cache hit"
else
  echo "  $(red "$FAILURES failure(s)")"
fi
echo ""

# ── Test 2: Cache bust ─────────────────────────────────────────────────────
fail_count
echo "$(bold "Test 2:") Cache bust"

DUNGEON_CLASS=m3 DUNGEON_SEED="$TEST_SEED" DUNGEON_BUST=1 DUNGEON_CACHE_ONLY=1 \
  bash "$explore_script" --class m3 --seed "$TEST_SEED" --bust --cache-only 2>&1 || {
  fail "cache bust command failed"; exit 1
}

# After bust, cache files should be removed
if [[ -f "$CACHE_BSP" ]]; then
  fail "BSP still present after bust"
fi
if [[ -f "$CACHE_LIT" ]]; then
  fail "LIT still present after bust"
fi
if [[ -f "$CACHE_MANIFEST" ]]; then
  fail "manifest still present after bust"
fi

if [[ $FAILURES -eq 0 ]]; then
  pass "cache bust"
else
  echo "  $(red "$FAILURES failure(s)")"
fi
echo ""

# ── Test 3: Cache regeneration ─────────────────────────────────────────────
fail_count
echo "$(bold "Test 3:") Cache regeneration after bust"

# After bust, running again should regenerate
DUNGEON_CLASS=m3 DUNGEON_SEED="$TEST_SEED" DUNGEON_CACHE_ONLY=1 \
  bash "$explore_script" --class m3 --seed "$TEST_SEED" --cache-only 2>&1 || {
  fail "cache regeneration failed"; exit 1
}

[[ -f "$CACHE_BSP" ]] || { fail "BSP not regenerated"; }
[[ -f "$CACHE_LIT" ]] || { fail "LIT not regenerated"; }
[[ -f "$CACHE_MANIFEST" ]] || { fail "manifest not regenerated"; }

# Verify BSP2/QLIT magic on regenerated files
bsp_header="$(head -c 4 "$CACHE_BSP")"
lit_header="$(head -c 4 "$CACHE_LIT")"
[[ "$bsp_header" == "BSP2" ]] || { fail "regenerated BSP2 magic missing"; }
[[ "$lit_header" == "QLIT" ]] || { fail "regenerated QLIT magic missing"; }

if [[ $FAILURES -eq 0 ]]; then
  pass "cache regeneration"
else
  echo "  $(red "$FAILURES failure(s)")"
fi
echo ""

# ── Test 4: Invalid cache rejection (tampered BSP) ─────────────────────────
fail_count
echo "$(bold "Test 4:") Invalid cache rejection (tampered BSP)"

# Corrupt the BSP by truncating it
BSP_SIZE="$(stat -c%s "$CACHE_BSP" 2>/dev/null || echo 0)"
TRUNC_SIZE=$((BSP_SIZE / 2))
truncate -s "$TRUNC_SIZE" "$CACHE_BSP" 2>/dev/null || {
  fail "could not truncate BSP (maybe truncate not available)"
}

if [[ $FAILURES -eq 0 ]]; then
  # Running should detect the corruption and rebuild
  # The sha256 won't match the manifest → cache miss → rebuild
  DUNGEON_CLASS=m3 DUNGEON_SEED="$TEST_SEED" DUNGEON_CACHE_ONLY=1 \
    bash "$explore_script" --class m3 --seed "$TEST_SEED" --cache-only 2>&1 || {
    fail "cache rebuild after tamper failed"; exit 1
  }

  # After rebuild, BSP magic should be valid
  bsp_header="$(head -c 4 "$CACHE_BSP")"
  [[ "$bsp_header" == "BSP2" ]] || {
    fail "post-tamper BSP magic is not BSP2"
  }
fi

if [[ $FAILURES -eq 0 ]]; then
  pass "invalid cache rejection (tampered BSP)"
else
  echo "  $(red "$FAILURES failure(s)")"
fi
echo ""

# ── Test 5: Invalid cache rejection (BSP2 magic stripped) ──────────────────
fail_count
echo "$(bold "Test 5:") Invalid cache rejection (bad BSP magic)"

# Rebuild clean first
rm -f "$CACHE_BSP" "$CACHE_LIT" "$CACHE_MANIFEST"
DUNGEON_CLASS=m3 DUNGEON_SEED="$TEST_SEED" DUNGEON_CACHE_ONLY=1 \
  bash "$explore_script" --class m3 --seed "$TEST_SEED" --cache-only 2>&1 || {
  fail "pre-test cache build failed"; exit 1
}

# Corrupt the BSP magic bytes
printf '\x00\x00\x00\x00' | dd of="$CACHE_BSP" bs=1 count=4 conv=notrunc 2>/dev/null || {
  fail "could not overwrite BSP magic"
}

if [[ $FAILURES -eq 0 ]]; then
  # Running should detect bad magic → cache miss → rebuild
  DUNGEON_CLASS=m3 DUNGEON_SEED="$TEST_SEED" DUNGEON_CACHE_ONLY=1 \
    bash "$explore_script" --class m3 --seed "$TEST_SEED" --cache-only 2>&1 || {
    fail "cache rebuild after magic corruption failed"; exit 1
  }

  # After rebuild, BSP magic should be valid
  bsp_header="$(head -c 4 "$CACHE_BSP")"
  [[ "$bsp_header" == "BSP2" ]] || {
    fail "post-corruption BSP magic not restored"
  }
fi

if [[ $FAILURES -eq 0 ]]; then
  pass "invalid cache rejection (bad BSP magic)"
else
  echo "  $(red "$FAILURES failure(s)")"
fi
echo ""

# ── Test 6: Invalid cache rejection (bad QLIT magic) ───────────────────────
fail_count
echo "$(bold "Test 6:") Invalid cache rejection (bad LIT magic)"

# Rebuild clean again
rm -f "$CACHE_BSP" "$CACHE_LIT" "$CACHE_MANIFEST"
DUNGEON_CLASS=m3 DUNGEON_SEED="$TEST_SEED" DUNGEON_CACHE_ONLY=1 \
  bash "$explore_script" --class m3 --seed "$TEST_SEED" --cache-only 2>&1 || {
  fail "pre-test cache build failed"; exit 1
}

# Corrupt the LIT magic bytes
printf '\x00\x00\x00\x00' | dd of="$CACHE_LIT" bs=1 count=4 conv=notrunc 2>/dev/null || {
  fail "could not overwrite LIT magic"
}

if [[ $FAILURES -eq 0 ]]; then
  DUNGEON_CLASS=m3 DUNGEON_SEED="$TEST_SEED" DUNGEON_CACHE_ONLY=1 \
    bash "$explore_script" --class m3 --seed "$TEST_SEED" --cache-only 2>&1 || {
    fail "cache rebuild after LIT magic corruption failed"; exit 1
  }

  lit_header="$(head -c 4 "$CACHE_LIT")"
  [[ "$lit_header" == "QLIT" ]] || {
    fail "post-corruption LIT magic not restored"
  }
fi

if [[ $FAILURES -eq 0 ]]; then
  pass "invalid cache rejection (bad LIT magic)"
else
  echo "  $(red "$FAILURES failure(s)")"
fi
echo ""

# ── Test 7: Manifest sha mismatch triggers rebuild ─────────────────────────
fail_count
echo "$(bold "Test 7:") Manifest BSP sha mismatch triggers rebuild"

rm -f "$CACHE_BSP" "$CACHE_LIT" "$CACHE_MANIFEST"
DUNGEON_CLASS=m3 DUNGEON_SEED="$TEST_SEED" DUNGEON_CACHE_ONLY=1 \
  bash "$explore_script" --class m3 --seed "$TEST_SEED" --cache-only 2>&1 || {
  fail "pre-test cache build failed"; exit 1
}

# Tamper the manifest to change the stored BSP sha
TAMPERED_MANIFEST="$(mktemp)"
sed 's/^bsp\.sha256 = .*/bsp.sha256 = "0000000000000000000000000000000000000000000000000000000000000000"/' \
  "$CACHE_MANIFEST" > "$TAMPERED_MANIFEST"
mv "$TAMPERED_MANIFEST" "$CACHE_MANIFEST"

DUNGEON_CLASS=m3 DUNGEON_SEED="$TEST_SEED" DUNGEON_CACHE_ONLY=1 \
  bash "$explore_script" --class m3 --seed "$TEST_SEED" --cache-only 2>&1 || {
  fail "cache rebuild after manifest tamper failed"; exit 1
}

# Manifest should have been rebuilt with correct sha
stored_sha="$(grep '^bsp\.sha256' "$CACHE_MANIFEST" | awk -F'"' '{print $2}')"
actual_sha="$(calc_sha256 "$CACHE_BSP")"
[[ "$stored_sha" == "$actual_sha" ]] || {
  fail "manifest BSP sha not corrected: stored=$stored_sha actual=$actual_sha"
}

if [[ $FAILURES -eq 0 ]]; then
  pass "manifest sha mismatch triggers rebuild"
else
  echo "  $(red "$FAILURES failure(s)")"
fi
echo ""

# ── Test 8: Manifest version mismatch triggers rebuild ─────────────────────
fail_count
echo "$(bold "Test 8:") Generator version mismatch triggers rebuild"

rm -f "$CACHE_BSP" "$CACHE_LIT" "$CACHE_MANIFEST"
DUNGEON_CLASS=m3 DUNGEON_SEED="$TEST_SEED" DUNGEON_CACHE_ONLY=1 \
  bash "$explore_script" --class m3 --seed "$TEST_SEED" --cache-only 2>&1 || {
  fail "pre-test cache build failed"; exit 1
}

# Tamper the manifest generator_version
TAMPERED_MANIFEST="$(mktemp)"
sed 's/^generator_version = .*/generator_version = "0000000000000000"/' \
  "$CACHE_MANIFEST" > "$TAMPERED_MANIFEST"
mv "$TAMPERED_MANIFEST" "$CACHE_MANIFEST"

DUNGEON_CLASS=m3 DUNGEON_SEED="$TEST_SEED" DUNGEON_CACHE_ONLY=1 \
  bash "$explore_script" --class m3 --seed "$TEST_SEED" --cache-only 2>&1 || {
  fail "cache rebuild after version tamper failed"; exit 1
}

# Generator version should have been updated
stored_ver="$(grep '^generator_version' "$CACHE_MANIFEST" | awk -F'"' '{print $2}')"
[[ "$stored_ver" != "0000000000000000" ]] || {
  fail "generator_version was not updated after version mismatch"
}

if [[ $FAILURES -eq 0 ]]; then
  pass "generator version mismatch triggers rebuild"
else
  echo "  $(red "$FAILURES failure(s)")"
fi
echo ""

# ── restore pre-existing cache ──────────────────────────────────────────────
cache_paths
if [[ -n "$RESTORE_BSP" && -f "$RESTORE_BSP" ]]; then
  cp "$RESTORE_BSP" "$CACHE_BSP"
fi
if [[ -n "$RESTORE_LIT" && -f "$RESTORE_LIT" ]]; then
  cp "$RESTORE_LIT" "$CACHE_LIT"
fi
if [[ -n "$RESTORE_MANIFEST" && -f "$RESTORE_MANIFEST" ]]; then
  cp "$RESTORE_MANIFEST" "$CACHE_MANIFEST"
fi

echo "$(dim "─────────────────────────────────────────────")"
echo "$(green "✓") M3 cache validation complete"
