//! Theme asset validation — verify the CC0 Stone Beta theme is
//! deterministically reproducible, format-correct, and correctly licensed.
//!
//! Tests invoke `build.py` in a temporary directory, inspect every output,
//! and confirm byte-identical regeneration.

use std::path::{Path, PathBuf};
use std::process::Command;

// ── Helpers ──────────────────────────────────────────────────────────────

/// Return the absolute path to the `themes/cc0_stone_beta` directory.
fn theme_dir() -> PathBuf {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    manifest.join("themes").join("cc0_stone_beta")
}

/// Run build.py in `out_dir`, returning the command exit status.
fn run_build(out_dir: &Path) -> std::process::ExitStatus {
    let build_py = theme_dir().join("build.py");
    assert!(
        build_py.is_file(),
        "build.py missing at {}",
        build_py.display()
    );

    Command::new("python3")
        .arg(&build_py)
        .arg(out_dir)
        .status()
        .expect("failed to execute build.py")
}

/// Assert that `path` exists and is a regular file.
fn assert_file(path: &Path) {
    assert!(path.is_file(), "missing file: {}", path.display());
}

/// Read the first 8 bytes of a PNG and parse width/height from IHDR.
fn png_dimensions(path: &Path) -> (u32, u32) {
    let data = std::fs::read(path).expect("cannot read PNG");
    // PNG signature: 8 bytes, then IHDR chunk: 4B length, 4B type, data
    assert!(
        &data[0..8] == b"\x89PNG\r\n\x1a\n",
        "not a valid PNG: {}",
        path.display()
    );
    assert!(
        &data[12..16] == b"IHDR",
        "IHDR chunk missing: {}",
        path.display()
    );
    let width = u32::from_be_bytes(data[16..20].try_into().unwrap());
    let height = u32::from_be_bytes(data[20..24].try_into().unwrap());
    (width, height)
}

// ── Existence & executability ────────────────────────────────────────────

#[test]
fn build_py_exists_and_is_readable() {
    let bp = theme_dir().join("build.py");
    assert!(bp.is_file(), "build.py not found at {}", bp.display());
    // Verify it starts with a shebang or is otherwise Python
    let first_line = std::fs::read_to_string(&bp)
        .expect("cannot read build.py")
        .lines()
        .next()
        .unwrap_or("")
        .to_string();
    assert!(
        first_line.starts_with("#!/") || first_line.contains("python"),
        "build.py should be a Python script"
    );
}

// ── Output existence ─────────────────────────────────────────────────────

#[test]
fn build_produces_all_outputs() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let status = run_build(tmp.path());
    assert!(status.success(), "build.py failed");

    // Top-level outputs
    assert_file(&tmp.path().join("palette.lmp"));
    assert_file(&tmp.path().join("cc0_stone_beta.wad"));
    assert_file(&tmp.path().join("theme.toml"));
    assert_file(&tmp.path().join("LICENSE"));

    // Texture companions
    let tex_dir = tmp.path().join("textures");
    for name in &["stone_floor", "stone_wall", "stone_ceiling", "stone_accent"] {
        assert_file(&tex_dir.join(format!("{name}_basecolor.png")));
        assert_file(&tex_dir.join(format!("{name}_norm.png")));
        assert_file(&tex_dir.join(format!("{name}_gloss.png")));
    }
}

// ── Palette validation ───────────────────────────────────────────────────

#[test]
fn palette_is_768_bytes() {
    let tmp = tempfile::tempdir().expect("tempdir");
    assert!(run_build(tmp.path()).success());
    let pal = std::fs::read(tmp.path().join("palette.lmp")).expect("read palette");
    assert_eq!(pal.len(), 768, "palette.lmp must be exactly 768 bytes");
}

#[test]
fn palette_reserves_224_to_255_for_fullbrights() {
    let tmp = tempfile::tempdir().expect("tempdir");
    assert!(run_build(tmp.path()).success());
    let pal = std::fs::read(tmp.path().join("palette.lmp")).expect("read palette");
    assert_eq!(pal.len(), 768, "palette.lmp must be exactly 768 bytes");

    for idx in 224usize..=255 {
        let rgb = &pal[idx * 3..idx * 3 + 3];
        assert!(
            rgb.iter().any(|&channel| channel == 255),
            "fullbright palette index {idx} should contain at least one saturated channel; got {rgb:?}"
        );
    }
}

// ── WAD2 header and directory validation ────────────────────────────────

#[test]
fn wad2_has_valid_header_and_four_miptex_entries() {
    let tmp = tempfile::tempdir().expect("tempdir");
    assert!(run_build(tmp.path()).success());
    let wad = std::fs::read(tmp.path().join("cc0_stone_beta.wad")).expect("read wad");

    // Header: magic, numlumps (i32 LE), infotableofs (i32 LE)
    assert!(&wad[0..4] == b"WAD2", "WAD2 magic missing");
    let numlumps = i32::from_le_bytes(wad[4..8].try_into().unwrap());
    assert_eq!(numlumps, 4, "expected 4 lumps");
    let infotableofs = i32::from_le_bytes(wad[8..12].try_into().unwrap()) as usize;
    assert!(infotableofs >= 12, "infotableofs must be after header");

    // Parse each directory entry
    let expected_names = ["stone_floor", "stone_wall", "stone_ceiling", "stone_accent"];
    for i in 0..4usize {
        let off = infotableofs + i * 32;
        let filepos = i32::from_le_bytes(wad[off..off + 4].try_into().unwrap());
        let disksize = i32::from_le_bytes(wad[off + 4..off + 8].try_into().unwrap());
        let size = i32::from_le_bytes(wad[off + 8..off + 12].try_into().unwrap());
        let typ = wad[off + 12];
        let comp = wad[off + 13];
        let name_bytes = &wad[off + 16..off + 32];
        let nul_pos = name_bytes.iter().position(|&b| b == 0).unwrap_or(16);
        let name = std::str::from_utf8(&name_bytes[..nul_pos]).expect("non-UTF8 lump name");
        assert_eq!(name, expected_names[i], "lump {i} name mismatch");

        assert_eq!(
            typ, 0x44,
            "lump {i} type should be miptex (0x44), got 0x{typ:02x}"
        );
        assert_eq!(comp, 0, "lump {i} compression should be 0");
        assert_eq!(disksize, size, "lump {i} disksize != size");
        assert!(disksize > 0, "lump {i} has zero size");
        assert!(filepos >= 12, "lump {i} filepos {filepos} overlaps header");

        // Read miptex header at filepos
        let fp = filepos as usize;
        let mip_name_bytes = &wad[fp..fp + 16];
        let mip_nul = mip_name_bytes.iter().position(|&b| b == 0).unwrap_or(16);
        let mip_name = std::str::from_utf8(&mip_name_bytes[..mip_nul]).unwrap();
        assert_eq!(mip_name, expected_names[i]);

        let mip_w = u32::from_le_bytes(wad[fp + 16..fp + 20].try_into().unwrap());
        let mip_h = u32::from_le_bytes(wad[fp + 20..fp + 24].try_into().unwrap());
        assert_eq!(mip_w, 64, "lump {i} width");
        assert_eq!(mip_h, 64, "lump {i} height");

        // All 4 mip offsets should be ≥ 40 (header size) and within lump
        for m in 0..4usize {
            let mip_off =
                u32::from_le_bytes(wad[fp + 24 + m * 4..fp + 28 + m * 4].try_into().unwrap());
            assert!(mip_off >= 40, "lump {i} mip {m} offset {mip_off} < header");
            assert!(
                (mip_off as usize) < disksize as usize,
                "lump {i} mip {m} offset {mip_off} beyond lump size {disksize}"
            );
        }
    }
}

// ── WAD lump bounds ──────────────────────────────────────────────────────

#[test]
fn wad_lumps_do_not_overlap() {
    let tmp = tempfile::tempdir().expect("tempdir");
    assert!(run_build(tmp.path()).success());
    let wad = std::fs::read(tmp.path().join("cc0_stone_beta.wad")).expect("read wad");

    let infotableofs = i32::from_le_bytes(wad[8..12].try_into().unwrap()) as usize;
    let mut ranges: Vec<(usize, usize)> = Vec::new();

    for i in 0..4usize {
        let off = infotableofs + i * 32;
        let filepos = i32::from_le_bytes(wad[off..off + 4].try_into().unwrap()) as usize;
        let disksize = i32::from_le_bytes(wad[off + 4..off + 8].try_into().unwrap()) as usize;
        ranges.push((filepos, filepos + disksize));
    }

    // Sort by start, check no overlap
    ranges.sort_by_key(|r| r.0);
    for w in ranges.windows(2) {
        assert!(w[0].1 <= w[1].0, "lump overlap: {:?} and {:?}", w[0], w[1]);
    }

    // Last lump must not extend into info table
    let last_end = ranges.last().unwrap().1;
    assert!(
        last_end <= infotableofs,
        "last lump ends at {last_end}, info table starts at {infotableofs}"
    );
}

// ── Companion texture validation ─────────────────────────────────────────

#[test]
fn each_base_texture_has_norm_and_gloss_at_matching_dimensions() {
    let tmp = tempfile::tempdir().expect("tempdir");
    assert!(run_build(tmp.path()).success());
    let tex_dir = tmp.path().join("textures");

    for name in &["stone_floor", "stone_wall", "stone_ceiling", "stone_accent"] {
        let base = tex_dir.join(format!("{name}_basecolor.png"));
        let norm = tex_dir.join(format!("{name}_norm.png"));
        let gloss = tex_dir.join(format!("{name}_gloss.png"));

        let (bw, bh) = png_dimensions(&base);
        let (nw, nh) = png_dimensions(&norm);
        let (gw, gh) = png_dimensions(&gloss);

        assert_eq!(bw, 64, "{name} basecolor width");
        assert_eq!(bh, 64, "{name} basecolor height");
        assert_eq!((nw, nh), (bw, bh), "{name} norm dimensions mismatch");
        assert_eq!((gw, gh), (bw, bh), "{name} gloss dimensions mismatch");
    }
}

// ── Determinism ──────────────────────────────────────────────────────────

#[test]
fn two_runs_produce_byte_identical_outputs() {
    let tmp_a = tempfile::tempdir().expect("tempdir A");
    let tmp_b = tempfile::tempdir().expect("tempdir B");

    assert!(run_build(tmp_a.path()).success());
    assert!(run_build(tmp_b.path()).success());

    // Compare every generated file
    for entry in walkdir::WalkDir::new(tmp_a.path())
        .sort_by_file_name()
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
    {
        let rel = entry.path().strip_prefix(tmp_a.path()).unwrap();
        let path_b = tmp_b.path().join(rel);
        let bytes_a = std::fs::read(entry.path()).expect("read A");
        let bytes_b = std::fs::read(&path_b).expect("read B");
        assert_eq!(
            bytes_a,
            bytes_b,
            "byte mismatch in {} (len {} vs {})",
            rel.display(),
            bytes_a.len(),
            bytes_b.len()
        );
    }
}

#[test]
fn two_runs_produce_byte_identical_wad() {
    let tmp_a = tempfile::tempdir().expect("tempdir A");
    let tmp_b = tempfile::tempdir().expect("tempdir B");
    assert!(run_build(tmp_a.path()).success());
    assert!(run_build(tmp_b.path()).success());

    let wad_a = std::fs::read(tmp_a.path().join("cc0_stone_beta.wad")).unwrap();
    let wad_b = std::fs::read(tmp_b.path().join("cc0_stone_beta.wad")).unwrap();
    assert_eq!(wad_a.len(), wad_b.len());
    assert_eq!(wad_a, wad_b);
}

#[test]
fn two_runs_produce_byte_identical_pngs() {
    let tmp_a = tempfile::tempdir().expect("tempdir A");
    let tmp_b = tempfile::tempdir().expect("tempdir B");
    assert!(run_build(tmp_a.path()).success());
    assert!(run_build(tmp_b.path()).success());

    let tex_a = tmp_a.path().join("textures");
    let tex_b = tmp_b.path().join("textures");
    for name in &[
        "stone_floor_basecolor.png",
        "stone_floor_norm.png",
        "stone_floor_gloss.png",
        "stone_wall_basecolor.png",
        "stone_wall_norm.png",
        "stone_wall_gloss.png",
        "stone_ceiling_basecolor.png",
        "stone_ceiling_norm.png",
        "stone_ceiling_gloss.png",
        "stone_accent_basecolor.png",
        "stone_accent_norm.png",
        "stone_accent_gloss.png",
    ] {
        let a = std::fs::read(tex_a.join(name)).unwrap();
        let b = std::fs::read(tex_b.join(name)).unwrap();
        assert_eq!(a.len(), b.len());
        assert_eq!(a, b, "PNG {name} not byte-identical between runs");
    }
}

// ── theme.toml validation ────────────────────────────────────────────────

#[test]
fn theme_toml_has_correct_role_bindings() {
    let tmp = tempfile::tempdir().expect("tempdir");
    assert!(run_build(tmp.path()).success());

    let toml_path = tmp.path().join("theme.toml");
    assert_file(&toml_path);
    let content = std::fs::read_to_string(&toml_path).expect("read theme.toml");

    // Parse as TOML (no external dep — simple line-based check + structural
    // assertions)
    assert!(content.contains("[roles]"), "missing [roles] section");

    // Each expected binding
    let expected = [
        ("floor", "stone_floor"),
        ("wall", "stone_wall"),
        ("ceiling", "stone_ceiling"),
        ("accent", "stone_accent"),
    ];
    for (role, tex) in &expected {
        let line = format!("{role} = \"{tex}\"");
        assert!(
            content.contains(&line),
            "theme.toml missing binding: {line}"
        );
    }

    // No extra roles
    let role_count = content.lines().filter(|l| l.contains('=')).count();
    assert_eq!(role_count, 4, "expected exactly 4 role bindings");
}

// ── LICENSE validation ───────────────────────────────────────────────────

#[test]
fn license_file_exists_with_cc0_text() {
    let tmp = tempfile::tempdir().expect("tempdir");
    assert!(run_build(tmp.path()).success());

    let lic_path = tmp.path().join("LICENSE");
    assert_file(&lic_path);
    let content = std::fs::read_to_string(&lic_path).expect("read LICENSE");

    assert!(content.contains("CC0"), "LICENSE must mention CC0");
    assert!(
        content.to_lowercase().contains("public domain"),
        "LICENSE must reference public domain"
    );
    assert!(
        content.len() > 100,
        "LICENSE too short ({} bytes), expected substantive text",
        content.len()
    );
}
