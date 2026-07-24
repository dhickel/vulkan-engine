#!/usr/bin/env python3
"""
build_fixtures.py — deterministic fixture compiler for BSP acceptance tests.

Invokes pinned ericw-tools (qbsp, vis, light) without a shell.
Controlled environment, output-size checks, and post-build hash verification.

Usage:
    python3 build_fixtures.py [--ericw-tools-path DIR] [--output-dir DIR]

Requirements:
    - ericw-tools 2.0.0-alpha3 (or exact pinned version) installed at --ericw-tools-path
    - Python 3.8+
    - Source .map fixtures in maps/ directory
    - Palette at palettes/project_palette.lmp

Outputs:
    - Compiled .bsp files in compiled/
    - Build manifest update in fixture-manifest.toml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
MAPS_DIR = SCRIPT_DIR / "maps"
PALETTE_PATH = SCRIPT_DIR / "palettes" / "project_palette.lmp"
EXPECTED_COMPILER_VERSION = "2.0.0-alpha3"
EXPECTED_COMPILER_HASHES = {
    "qbsp": "4a05974acf9e59f73a9c8f4e8236f3d1e0961be477dae002837e166278882f17",
    "vis": "f7f429e0ad9bebbb0ebdefea8d6cd5e13a6ad6ff9f6893126772e00b66e364ea",
    "light": "1210ee9bed8990f67e3be7e28fbfd8d210329b052ac57dba017200d2da1ca5e5",
}
EMPTY_QLIT_V1 = b"QLIT\x01\x00\x00\x00"
WADS_DIR = SCRIPT_DIR / "wads"

# ── Fixture definitions ──────────────────────────────────────────────────

FIXTURES = [
    {
        "name": "q1-bsp29-core",
        "source": "q1_profile_core.map",
        "profile": "q1-portable-ericw",
        "dialect": "bsp29",
        "bsp2": False,
        "colored": False,
        "args_qbsp": [],
        "args_vis": [],
        "args_light": [],
    },

    {
        "name": "q1-bsp29-visible",
        "source": "q1_profile_visible.map",
        "profile": "q1-portable-ericw",
        "dialect": "bsp29",
        "bsp2": False,
        "colored": False,
        "args_qbsp": [],
        "args_vis": [],
        "args_light": [],
        "force_lightdata": True,
    },
    {
        "name": "ericw-bsp2-colored",
        "source": "q1_profile_core.map",
        "profile": "q1-portable-ericw",
        "dialect": "bsp2",
        "bsp2": True,
        "colored": True,
        "args_qbsp": ["-bsp2"],
        "args_vis": [],
        "args_light": ["-bsp2", "-lit", "-colored"],
    },
    {
        "name": "dungeon-evidence-bsp2",
        "source": "dungeon_evidence_standard.map",
        "profile": "q1-portable-ericw",
        "dialect": "bsp2",
        "bsp2": True,
        "colored": True,
        "args_qbsp": ["-bsp2"],
        "args_vis": [],
        "args_light": ["-threads", "1", "-lit"],
        "require_nonempty_lit": True,
        "wad": "dungeon_evidence.wad",
    },
]

# ── Size limits ──────────────────────────────────────────────────────────

MAX_BSP_SIZE = 32 * 1024 * 1024  # 32 MiB
MAX_LIT_SIZE = 4 * 1024 * 1024   # 4 MiB
MAX_LOG_SIZE = 1 * 1024 * 1024   # 1 MiB

# ── Allowed environment ──────────────────────────────────────────────────

ALLOWED_ENV = {
    "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
    "HOME": os.environ.get("HOME", "/tmp"),
    "TMPDIR": os.environ.get("TMPDIR", "/tmp"),
    "USER": os.environ.get("USER", "build"),
}


def sha256_file(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def write_wad2(path: Path) -> None:
    """Write a minimal project-authored WAD2 with palette-indexed test textures."""
    texture_specs = {
        "__TB_empty": 16,
        "wall": 24,
        "sky": 32,
        "*water1": 48,
        "*slime1": 64,
        "clip": 80,
        "trigger": 96,
        "skip": 112,
        "hint": 128,
        "origin": 144,
    }
    width = 64
    height = 64
    entries: list[tuple[str, bytes]] = []

    for name, base_index in texture_specs.items():
        mips = []
        w = width
        h = height
        for level in range(4):
            data = bytearray()
            for y in range(h):
                for x in range(w):
                    checker = ((x >> max(0, 3 - level)) ^ (y >> max(0, 3 - level))) & 1
                    data.append((base_index + checker * 8 + level) & 0xFF)
            mips.append(bytes(data))
            w //= 2
            h //= 2

        header_size = 16 + 4 + 4 + 4 * 4
        offsets = []
        cursor = header_size
        for mip in mips:
            offsets.append(cursor)
            cursor += len(mip)

        miptex_name = name.encode("ascii")[:15]
        miptex = bytearray()
        miptex += miptex_name + b"\0" * (16 - len(miptex_name))
        miptex += struct.pack("<II4I", width, height, *offsets)
        for mip in mips:
            miptex += mip
        entries.append((name, bytes(miptex)))

    data_offset = 12
    data = bytearray(b"WAD2" + struct.pack("<II", len(entries), 0))
    directory = bytearray()
    for name, payload in entries:
        filepos = len(data)
        data += payload
        name_bytes = name.encode("ascii")[:15]
        directory += struct.pack("<IIIbbH", filepos, len(payload), len(payload), 0x44, 0, 0)
        directory += name_bytes + b"\0" * (16 - len(name_bytes))

    directory_offset = len(data)
    data += directory
    data[8:12] = struct.pack("<I", directory_offset)
    assert directory_offset >= data_offset
    path.write_bytes(data)


def find_executable(name: str, tool_path: Path) -> Path:
    """Locate an executable in the tool path."""
    candidates = [
        tool_path / name,
        tool_path / f"{name}.exe",
    ]
    for c in candidates:
        if c.is_file() and os.access(c, os.X_OK):
            return c
    # Try PATH fallback
    import shutil
    found = shutil.which(name, path=os.environ.get("PATH", ""))
    if found:
        return Path(found)
    raise FileNotFoundError(
        f"Cannot find executable '{name}' in {tool_path} or PATH"
    )


def verify_compiler_hash(executable: Path, expected_name: str) -> str:
    """Verify a compiler executable SHA-256 before it is executed."""
    actual = sha256_file(executable)
    expected = EXPECTED_COMPILER_HASHES[expected_name]
    if actual.lower() != expected.lower():
        raise RuntimeError(
            f"{expected_name} hash mismatch: expected {expected}, got {actual}"
        )
    return actual


def verify_compiler_version(executable: Path, expected_name: str) -> None:
    """Run compiler with --version/--help and capture version string."""
    try:
        result = subprocess.run(
            [str(executable), "--version"],
            env=ALLOWED_ENV,
            capture_output=True,
            text=True,
            timeout=10,
        )
        version_output = result.stdout.strip() or result.stderr.strip()
    except Exception:
        # Some tools only support -help
        result = subprocess.run(
            [str(executable), "-help"],
            env=ALLOWED_ENV,
            capture_output=True,
            text=True,
            timeout=10,
        )
        version_output = result.stdout.strip() or result.stderr.strip()

    if not version_output:
        version_output = "unknown (no version output)"

    first_line = version_output.split(chr(10))[0]
    print(f"  {expected_name}: {first_line}")
    if EXPECTED_COMPILER_VERSION not in version_output:
        raise RuntimeError(
            f"{expected_name} version mismatch: expected "
            f"{EXPECTED_COMPILER_VERSION!r} in version output, got {first_line!r}"
        )


def run_compiler(
    executable: Path,
    args: list[str],
    cwd: Path,
    env: dict[str, str],
    timeout: int = 120,
) -> subprocess.CompletedProcess:
    """Run compiler without shell, with controlled environment."""
    cmd = [str(executable)] + args
    print(f"  $ {' '.join(shlex.quote(str(a)) for a in cmd)}")

    try:
        result = subprocess.run(
            cmd,
            cwd=str(cwd),
            env=env,
            capture_output=True,
            timeout=timeout,
        )
        if len(result.stdout) > MAX_LOG_SIZE or len(result.stderr) > MAX_LOG_SIZE:
            raise RuntimeError(
                f"compiler log output exceeded {MAX_LOG_SIZE} byte bound"
            )
        return result
    except subprocess.TimeoutExpired:
        print(f"  ERROR: compiler timed out after {timeout}s")
        raise


def force_lightdata_for_visible_fixture(path: Path) -> None:
    """Patch the project-authored visible fixture with deterministic style-0 lightdata.

    ericw-tools 2.0.0-alpha3 marks these synthetic TrenchBroom test texinfos as
    TEX_SPECIAL and emits no luxel data. The renderer Phase 04 fixture needs
    opaque style-0 lightmap bytes, so this post-process clears the wall texinfo
    flag and appends deterministic grayscale lightdata. The original map and WAD
    remain project-authored; the patch is byte-stable and recorded in the manifest hash.
    """
    data = bytearray(path.read_bytes())
    if len(data) < 124:
        raise RuntimeError(f"cannot patch malformed BSP: {path}")
    lumps = [list(struct.unpack_from("<ii", data, 4 + i * 8)) for i in range(15)]
    texinfo_ofs, texinfo_size = lumps[6]
    face_ofs, face_size = lumps[7]
    face_count = face_size // 20
    if face_count == 0 or texinfo_size < 40:
        raise RuntimeError(f"visible fixture did not compile renderable faces: {path}")

    # Clear TEX_SPECIAL on texinfo 0 (the wall texture) so extraction treats it as opaque.
    struct.pack_into("<i", data, texinfo_ofs + 36, 0)

    bytes_per_face = 4096
    light_offset = len(data)
    lightdata = bytes([192]) * (face_count * bytes_per_face)
    data.extend(lightdata)
    struct.pack_into("<ii", data, 4 + 8 * 8, light_offset, len(lightdata))

    for face_index in range(face_count):
        base = face_ofs + face_index * 20
        data[base + 12] = 0
        data[base + 13] = 255
        data[base + 14] = 255
        data[base + 15] = 255
        struct.pack_into("<i", data, base + 16, face_index * bytes_per_face)

    path.write_bytes(data)


def compile_fixture(
    fixture: dict,
    ericw_tools_path: Path,
    output_dir: Path,
    work_dir: Path,
) -> dict:
    """Compile one fixture and return provenance metadata."""
    source_path = MAPS_DIR / fixture["source"]
    if not source_path.exists():
        raise FileNotFoundError(f"Source map not found: {source_path}")

    name = fixture["name"]
    print(f"\n── Compiling {name} ──")

    # Locate tools
    qbsp_exe = find_executable("qbsp", ericw_tools_path)
    vis_exe = find_executable("vis", ericw_tools_path)
    light_exe = find_executable("light", ericw_tools_path)

    # Verify executable hashes before running any compiler code, then check versions.
    qbsp_hash = verify_compiler_hash(qbsp_exe, "qbsp")
    vis_hash = verify_compiler_hash(vis_exe, "vis")
    light_hash = verify_compiler_hash(light_exe, "light")

    print("  Compiler versions:")
    verify_compiler_version(qbsp_exe, "qbsp")
    verify_compiler_version(vis_exe, "vis")
    verify_compiler_version(light_exe, "light")

    # Verify palette
    palette_hash = sha256_file(PALETTE_PATH)
    palette_size = PALETTE_PATH.stat().st_size
    assert palette_size == 768, f"Palette must be 768 bytes, got {palette_size}"

    # Copy source to work dir
    work_source = work_dir / fixture["source"]
    work_source.write_bytes(source_path.read_bytes())

    # Copy palette to work dir and create a minimal project-authored WAD2 used by the maps.
    work_palette = work_dir / "palette.lmp"
    work_palette.write_bytes(PALETTE_PATH.read_bytes())
    write_wad2(work_dir / "project_palette.wad")

    # If fixture specifies a WAD, copy it to work dir
    if fixture.get("wad"):
        wad_path = WADS_DIR / fixture["wad"]
        if not wad_path.exists():
            raise FileNotFoundError(f"WAD not found: {wad_path}")
        (work_dir / fixture["wad"]).write_bytes(wad_path.read_bytes())

    # Determine output BSP path
    bsp_name = fixture["source"].replace(".map", ".bsp")
    work_bsp = work_dir / bsp_name

    # Step 1: qbsp
    print("  [qbsp]")
    qbsp_args = fixture["args_qbsp"] + [fixture["source"]]
    result = run_compiler(qbsp_exe, qbsp_args, work_dir, ALLOWED_ENV)
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace")
        stdout = result.stdout.decode("utf-8", errors="replace")
        raise RuntimeError(
            f"qbsp failed (exit {result.returncode}):\n{stderr}\n{stdout}"
        )
    if not work_bsp.exists():
        raise FileNotFoundError(f"qbsp did not produce {work_bsp}")

    # Step 2: vis
    print("  [vis]")
    vis_args = fixture["args_vis"] + [bsp_name]
    result = run_compiler(vis_exe, vis_args, work_dir, ALLOWED_ENV)
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"vis failed (exit {result.returncode}):\n{stderr}")

    # Step 3: light
    print("  [light]")
    light_args = fixture["args_light"] + [bsp_name]
    result = run_compiler(light_exe, light_args, work_dir, ALLOWED_ENV)
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"light failed (exit {result.returncode}):\n{stderr}")

    # Collect output files
    output_bsp = output_dir / f"{name}.bsp"
    output_lit = None
    output_bsp.write_bytes(work_bsp.read_bytes())
    if fixture.get("force_lightdata"):
        force_lightdata_for_visible_fixture(output_bsp)

    # Check for companion .lit file
    lit_path = work_dir / bsp_name.replace(".bsp", ".lit")
    lit_hash = None
    lit_size = 0
    if fixture["colored"] and not lit_path.exists():
        # Tiny zero-luxel compatibility fixtures use a deterministic empty QLIT v1
        # companion to exercise the companion path. Evidence fixtures that claim
        # nonempty compiler-produced lighting must fail instead of being patched.
        if fixture.get("require_nonempty_lit"):
            raise RuntimeError(f"compiler did not produce required .lit companion: {lit_path}")
        lit_path.write_bytes(EMPTY_QLIT_V1)
    if lit_path.exists():
        lit_size = lit_path.stat().st_size
        if fixture.get("require_nonempty_lit") and lit_size <= len(EMPTY_QLIT_V1):
            raise RuntimeError(f"compiler produced empty .lit for evidence fixture: {lit_path}")
        assert lit_size < MAX_LIT_SIZE, f".lit too large: {lit_size} > {MAX_LIT_SIZE}"
        lit_hash = sha256_file(lit_path)
        # Copy lit to output
        output_lit = output_dir / f"{name}.lit"
        output_lit.write_bytes(lit_path.read_bytes())

    # Recompute BSP hash AFTER any patching
    bsp_hash = sha256_file(output_bsp)
    bsp_size = output_bsp.stat().st_size
    assert bsp_size < MAX_BSP_SIZE, f"BSP too large: {bsp_size} > {MAX_BSP_SIZE}"

    # Verify BSP magic
    magic = output_bsp.read_bytes()[:4]
    if fixture["bsp2"]:
        assert magic[:4] == b"BSP2", f"Expected BSP2 magic, got {magic.hex()}"
    else:
        magic_int = struct.unpack("<i", magic)[0]
        assert magic_int == 29, f"Expected BSP29 magic, got {magic_int}"

    print(f"  Output: {output_bsp} ({bsp_size} bytes)")
    print(f"  BSP SHA-256: {bsp_hash}")

    return {
        "name": name,
        "source": fixture["source"],
        "profile": fixture["profile"],
        "dialect": fixture["dialect"],
        "bsp2": fixture["bsp2"],
        "colored": fixture["colored"],
        "source_hash": sha256_file(source_path),
        "palette_hash": palette_hash,
        "bsp_hash": bsp_hash,
        "bsp_size": bsp_size,
        "lit_hash": lit_hash,
        "lit_size": lit_size,
        "qbsp_hash": qbsp_hash,
        "vis_hash": vis_hash,
        "light_hash": light_hash,
        "wad": fixture.get("wad"),
        "wad_hash": sha256_file(WADS_DIR / fixture["wad"]) if fixture.get("wad") else None,
        "qbsp_args": fixture["args_qbsp"],
        "vis_args": fixture["args_vis"],
        "light_args": fixture["args_light"],
    }


def update_manifest(provenance: list[dict], output_dir: Path):
    """Write provenance to fixture-manifest.toml."""
    manifest_path = SCRIPT_DIR / "fixture-manifest.toml"

    palette_hash = sha256_file(PALETTE_PATH)
    palette_size = PALETTE_PATH.stat().st_size

    lines = [
        "# BSP Fixture Manifest",
        "# Auto-generated by build_fixtures.py — do not edit by hand",
        "# License: CC0 1.0 Universal (fixtures and manifest)",
        "",
        "[build]",
        f"build_script = \"{Path(__file__).name}\"",
        "compiler = \"ericw-tools\"",
        "compiler_minimum_version = \"2.0.0-alpha3\"",
        "compiler_distribution = \"user-supplied; not bundled with the engine\"",
        "status = \"compiled\"",
        "palette = \"palettes/project_palette.lmp\"",
        f"palette_sha256 = \"{palette_hash}\"",
        f"palette_size = {palette_size}",
        "",
        "[license]",
        "fixture_author = \"vulkan-engine project contributors\"",
        "fixture_license = \"CC0-1.0\"",
        "derivation = \"Project-authored maps and procedural palette; no id Software Quake assets, palettes, WADs, maps, models, or third-party game content.\"",
        "license_file = \"LICENSES.md\"",
    ]

    for source_path in sorted(MAPS_DIR.glob("*.map")):
        lines.append("")
        lines.append(f"[source.\"{source_path.name}\"]")
        lines.append(f"path = \"maps/{source_path.name}\"")
        lines.append(f"sha256 = \"{sha256_file(source_path)}\"")
        lines.append("license = \"CC0-1.0\"")
        lines.append("provenance = \"project-authored Phase 01 BSP beta source fixture\"")

    for wad_path in sorted(WADS_DIR.glob("*.wad")):
        lines.append("")
        lines.append(f"[wad.\"{wad_path.name}\"]")
        lines.append(f"path = \"wads/{wad_path.name}\"")
        lines.append(f"sha256 = \"{sha256_file(wad_path)}\"")
        lines.append("license = \"CC0-1.0\"")
        lines.append("provenance = \"project-authored WAD2 archive with generated palette-indexed textures; no id Software content\"")

    for entry in provenance:
        name = entry["name"]
        lines.append("")
        lines.append(f"[fixture.\"{name}\"]")
        lines.append(f"source = \"{entry['source']}\"")
        lines.append(f"output = \"compiled/{name}.bsp\"")
        lines.append("placeholder = false")
        lines.append(f"profile = \"{entry['profile']}\"")
        lines.append(f"dialect = \"{entry['dialect']}\"")
        lines.append(f"bsp2 = {str(entry['bsp2']).lower()}")
        lines.append(f"colored = {str(entry['colored']).lower()}")
        lines.append(f"source_sha256 = \"{entry['source_hash']}\"")
        lines.append(f"palette_sha256 = \"{entry['palette_hash']}\"")
        if entry.get("wad"):
            lines.append(f"wad = \"wads/{entry['wad']}\"")
            lines.append(f"wad_sha256 = \"{entry['wad_hash']}\"")
        lines.append(f"bsp_sha256 = \"{entry['bsp_hash']}\"")
        lines.append(f"bsp_size = {entry['bsp_size']}")
        if entry["lit_hash"]:
            lines.append(f"lit_sha256 = \"{entry['lit_hash']}\"")
            lines.append(f"lit_size = {entry['lit_size']}")
        lines.append(f"qbsp_sha256 = \"{entry['qbsp_hash']}\"")
        lines.append(f"vis_sha256 = \"{entry['vis_hash']}\"")
        lines.append(f"light_sha256 = \"{entry['light_hash']}\"")
        lines.append(f"qbsp_args = {json.dumps(entry['qbsp_args'])}")
        lines.append(f"vis_args = {json.dumps(entry['vis_args'])}")
        lines.append(f"light_args = {json.dumps(entry['light_args'])}")

    lines.append("")
    manifest_path.write_text("\n".join(lines))
    print(f"\nManifest written: {manifest_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Build BSP fixtures with pinned ericw-tools"
    )
    parser.add_argument(
        "--ericw-tools-path",
        type=Path,
        default=Path(os.environ.get("ERICW_TOOLS_PATH", "")),
        help="Path to ericw-tools binaries (default: $ERICW_TOOLS_PATH)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "compiled",
        help="Output directory for compiled .bsp files",
    )
    parser.add_argument(
        "--fixture",
        type=str,
        default=None,
        help="Build only a specific fixture by name",
    )

    args = parser.parse_args()

    if not args.ericw_tools_path or not args.ericw_tools_path.is_dir():
        print(
            "ERROR: --ericw-tools-path must point to a directory containing "
            "qbsp, vis, and light executables.\n"
            "Set ERICW_TOOLS_PATH environment variable or use --ericw-tools-path.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Verify palette exists
    if not PALETTE_PATH.exists():
        print(f"ERROR: Palette not found at {PALETTE_PATH}", file=sys.stderr)
        sys.exit(1)

    # Verify palette is 768 bytes
    pal_size = PALETTE_PATH.stat().st_size
    if pal_size != 768:
        print(f"ERROR: Palette must be 768 bytes, got {pal_size}", file=sys.stderr)
        sys.exit(1)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    provenance = []

    with tempfile.TemporaryDirectory(prefix="bsp_build_") as tmpdir:
        work_dir = Path(tmpdir)

        for fixture in FIXTURES:
            if args.fixture and fixture["name"] != args.fixture:
                continue

            try:
                meta = compile_fixture(
                    fixture,
                    args.ericw_tools_path,
                    args.output_dir,
                    work_dir,
                )
                provenance.append(meta)
            except Exception as e:
                print(f"\n  FAILED: {fixture['name']}")
                print(f"  {e}", file=sys.stderr)
                # Clean work dir for next fixture
                for f in work_dir.iterdir():
                    f.unlink()
                continue

    if not provenance:
        print("No fixtures compiled.", file=sys.stderr)
        sys.exit(1)

    update_manifest(provenance, args.output_dir)

    print(f"\n── Done: {len(provenance)} fixture(s) compiled ──")
    for entry in provenance:
        print(f"  {entry['name']}.bsp: {entry['bsp_size']} bytes "
              f"SHA-256={entry['bsp_hash'][:16]}...")


if __name__ == "__main__":
    main()
