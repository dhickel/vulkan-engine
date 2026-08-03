//! Phase 12-D: Shared validation and package rules — test suite.
//!
//! Tests every failure mode for Richness V1 theme asset validation and
//! the deterministic fresh-build comparator. All tests are fail-closed:
//! any mismatch, corruption, ambiguity, or missing asset must produce
//! an error.

use std::collections::BTreeSet;
use std::path::PathBuf;

use engine_pack::richness_assets::{
    self, test_theme_def, THEME_ANCIENT, THEME_BRUTALIST, THEME_EGYPTIAN,
};

// ── Helpers ───────────────────────────────────────────────────────────────

fn theme_root() -> PathBuf {
    let manifest_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    manifest_dir
        .parent()
        .and_then(|p| p.parent())
        .expect("engine_pack not under workspace tools/")
        .join("src/bsp_generator/themes")
}

fn ancient_dir() -> PathBuf {
    theme_root().join("richness_ancient_v1")
}

// ── Valid closure tests ───────────────────────────────────────────────────

#[test]
fn ancient_closure_validates() {
    richness_assets::validate_theme_closure(&ancient_dir(), &THEME_ANCIENT)
        .expect("ancient closure must validate");
}

#[test]
fn egyptian_closure_validates() {
    let dir = theme_root().join("richness_egyptian_v1");
    richness_assets::validate_theme_closure(&dir, &THEME_EGYPTIAN)
        .expect("egyptian closure must validate");
}

#[test]
fn brutalist_closure_validates() {
    let dir = theme_root().join("richness_brutalist_v1");
    richness_assets::validate_theme_closure(&dir, &THEME_BRUTALIST)
        .expect("brutalist closure must validate");
}

// ── Package table / authored metadata agreement ──────────────────────────

#[test]
fn package_role_table_matches_every_theme_toml() {
    for (dir, definition) in [
        (ancient_dir(), &THEME_ANCIENT),
        (theme_root().join("richness_egyptian_v1"), &THEME_EGYPTIAN),
        (theme_root().join("richness_brutalist_v1"), &THEME_BRUTALIST),
    ] {
        richness_assets::validate_theme_toml(&dir, definition)
            .unwrap_or_else(|error| panic!("{dir:?}: {error}"));
    }
}

// ── PNG validation ────────────────────────────────────────────────────────

#[test]
fn all_pngs_have_correct_dimensions_and_crc() {
    for (dir, def) in [
        (&ancient_dir(), &THEME_ANCIENT),
        (&theme_root().join("richness_egyptian_v1"), &THEME_EGYPTIAN),
        (
            &theme_root().join("richness_brutalist_v1"),
            &THEME_BRUTALIST,
        ),
    ] {
        let textures = dir.join("textures");
        for filename in def.all_png_filenames() {
            let path = textures.join(filename);
            let bytes = std::fs::read(&path).unwrap();
            let (w, h) = richness_assets::validate_png_crc_and_dimensions_bytes(&bytes, 256)
                .unwrap_or_else(|e| panic!("{path:?}: {e}"));
            assert_eq!((w, h), (256, 256), "{path:?}: wrong dimensions");
        }
    }
}

// ── Missing file test ─────────────────────────────────────────────────────

#[test]
fn reject_missing_texture() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let textures = tmp.path().join("textures");
    std::fs::create_dir(&textures).unwrap();
    // Stage a theme.toml so the directory looks semi-valid
    std::fs::write(tmp.path().join("theme.toml"), "x").unwrap();
    std::fs::write(tmp.path().join("LICENSE"), "CC0").unwrap();
    std::fs::write(tmp.path().join("palette.lmp"), &[0u8; 768]).unwrap();
    std::fs::write(
        tmp.path().join("provenance.toml"),
        "[theme]\nname = \"test\"",
    )
    .unwrap();

    // No WAD, no PNGs — validation must fail
    let def = test_theme_def("test");
    let result = richness_assets::validate_theme_closure(tmp.path(), &def);
    assert!(result.is_err(), "missing WAD should fail");
}

// ── Corrupt PNG test ──────────────────────────────────────────────────────

#[test]
fn reject_corrupt_png_crc() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let textures = tmp.path().join("textures");
    std::fs::create_dir(&textures).unwrap();

    // Write a valid PNG, then corrupt a CRC byte
    let valid_png = make_valid_rgb_png(256, 256);
    assert!(valid_png.len() > 40);
    let mut corrupt = valid_png.clone();
    // Change a byte in the IDAT data
    corrupt[50] ^= 0xFF;

    let path = textures.join("wall_basecolor.png");
    std::fs::write(&path, &corrupt).unwrap();

    let result = richness_assets::validate_png_crc_and_dimensions_bytes(&corrupt, 256);
    assert!(result.is_err(), "corrupt PNG CRC should fail");
    let err = format!("{result:?}");
    assert!(
        err.to_ascii_lowercase().contains("crc"),
        "error should mention CRC: {err}"
    );
}

// ── Wrong dimensions test ─────────────────────────────────────────────────

#[test]
fn reject_wrong_dimensions() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let textures = tmp.path().join("textures");
    std::fs::create_dir(&textures).unwrap();

    let png_128 = make_valid_rgb_png(128, 128);
    let path = textures.join("wall_basecolor.png");
    std::fs::write(&path, &png_128).unwrap();

    let result = richness_assets::validate_png_crc_and_dimensions_bytes(&png_128, 256);
    assert!(result.is_err(), "wrong dimensions should fail");
    let err = format!("{result:?}");
    assert!(
        err.contains("dimension") || err.contains("256"),
        "error should mention dimensions: {err}"
    );
}

// ── Palette size test ─────────────────────────────────────────────────────

#[test]
fn reject_wrong_palette_size() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let pal = tmp.path().join("palette.lmp");
    std::fs::write(&pal, &[0u8; 100]).unwrap();

    let result = richness_assets::validate_palette(&pal, 768);
    assert!(result.is_err(), "wrong palette size should fail");
}

// ── WAD identity tests ────────────────────────────────────────────────────

#[test]
fn reject_wad_missing_identity() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let wad_path = tmp.path().join("test.wad");

    // Create a minimal WAD with only "skip" (missing the 9 role identities)
    let wad_bytes = make_minimal_wad(&["skip"]);
    std::fs::write(&wad_path, &wad_bytes).unwrap();

    let expected: Vec<&str> = vec![
        "wall", "floor", "ceiling", "accent", "portal", "vertical", "cave", "prop", "emissive",
        "skip",
    ];
    let result = richness_assets::validate_wad(&wad_path, &expected);
    assert!(result.is_err(), "WAD missing identities should fail");
}

#[test]
fn reject_wad_extra_identity() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let wad_path = tmp.path().join("test.wad");

    let wad_bytes = make_minimal_wad(&["wall", "hint"]);
    std::fs::write(&wad_path, &wad_bytes).unwrap();

    let expected: Vec<&str> = vec!["wall"];
    let result = richness_assets::validate_wad(&wad_path, &expected);
    assert!(result.is_err(), "WAD extra identity should fail");
}

#[test]
fn reject_wad_uppercase_identity() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let wad_path = tmp.path().join("test.wad");

    let wad_bytes = make_minimal_wad(&["WALL"]);
    std::fs::write(&wad_path, &wad_bytes).unwrap();

    let expected: Vec<&str> = vec!["WALL"];
    let result = richness_assets::validate_wad(&wad_path, &expected);
    assert!(result.is_err(), "uppercase WAD identity should fail");
}

// ── License test ──────────────────────────────────────────────────────────

#[test]
fn reject_non_cc0_license() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let license = tmp.path().join("LICENSE");
    std::fs::write(&license, "All Rights Reserved").unwrap();

    let result = richness_assets::validate_license_cc0(&license);
    assert!(result.is_err(), "non-CC0 license should fail");
}

// ── Case stability test ───────────────────────────────────────────────────

#[test]
fn reject_case_mismatch() {
    let tmp = tempfile::tempdir().expect("tempdir");
    std::fs::write(tmp.path().join("THEME.TOML"), "x").unwrap();

    let mut expected = BTreeSet::new();
    expected.insert("theme.toml".to_string());
    let result = richness_assets::validate_case_stability(tmp.path(), &expected);
    assert!(result.is_err(), "case mismatch should fail");
}

// ── Extra file test ───────────────────────────────────────────────────────

#[test]
fn closure_rejects_extra_file() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let textures = tmp.path().join("textures");
    std::fs::create_dir(&textures).unwrap();

    // Write enough to look valid-ish
    std::fs::write(tmp.path().join("theme.toml"), "x").unwrap();
    std::fs::write(tmp.path().join("LICENSE"), "CC0").unwrap();
    std::fs::write(tmp.path().join("palette.lmp"), &[0u8; 768]).unwrap();
    std::fs::write(
        tmp.path().join("provenance.toml"),
        "[theme]\nname = \"test\"",
    )
    .unwrap();
    // Extra file
    std::fs::write(tmp.path().join("README.md"), "hi").unwrap();

    let def = test_theme_def("test");
    let result = richness_assets::validate_theme_closure(tmp.path(), &def);
    assert!(result.is_err(), "extra file should fail validation");
    let err = format!("{result:?}");
    assert!(
        err.to_ascii_lowercase().contains("extra"),
        "error should mention extra file: {err}"
    );
}

// ── Symlink test ──────────────────────────────────────────────────────────

#[test]
fn reject_symlinked_asset() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let real = tmp.path().join("real.png");
    std::fs::write(&real, b"\x89PNG\r\n\x1a\nnotreally").unwrap();
    let link = tmp.path().join("wall_basecolor.png");

    #[cfg(unix)]
    {
        std::os::unix::fs::symlink(&real, &link).unwrap();
        let result = richness_assets::validate_png_asset(
            &link,
            256,
            engine_pack::richness_assets::PngCompanionKind::Basecolor,
        );
        assert!(result.is_err(), "symlinked PNG should be rejected");
    }
    let _ = (tmp, real, link); // silence unused warnings on non-unix
}

// ── Fresh-build comparator ────────────────────────────────────────────────

#[test]
fn fresh_build_ancient() {
    let dir = ancient_dir();
    match richness_assets::fresh_build_compare(&dir, &THEME_ANCIENT) {
        Ok(()) => {} // success
        Err(engine_pack::richness_assets::RichnessAssetError::FreshBuildUnavailable { .. }) => {
            eprintln!("SKIP: fresh-build comparison requires python3 + Pillow");
        }
        Err(e) => panic!("fresh-build comparison failed: {e}"),
    }
}

#[test]
fn fresh_build_egyptian() {
    let dir = theme_root().join("richness_egyptian_v1");
    match richness_assets::fresh_build_compare(&dir, &THEME_EGYPTIAN) {
        Ok(()) => {}
        Err(engine_pack::richness_assets::RichnessAssetError::FreshBuildUnavailable { .. }) => {
            eprintln!("SKIP: fresh-build comparison requires python3 + Pillow");
        }
        Err(e) => panic!("fresh-build comparison failed: {e}"),
    }
}

#[test]
fn fresh_build_brutalist() {
    let dir = theme_root().join("richness_brutalist_v1");
    match richness_assets::fresh_build_compare(&dir, &THEME_BRUTALIST) {
        Ok(()) => {}
        Err(engine_pack::richness_assets::RichnessAssetError::FreshBuildUnavailable { .. }) => {
            eprintln!("SKIP: fresh-build comparison requires python3 + Pillow");
        }
        Err(e) => panic!("fresh-build comparison failed: {e}"),
    }
}

// ── Staging test ──────────────────────────────────────────────────────────

#[test]
fn stage_ancient_is_well_formed() {
    let staging = tempfile::tempdir().expect("tempdir");
    let dir = ancient_dir();
    let staged = richness_assets::stage_richness_package(&dir, staging.path(), &THEME_ANCIENT)
        .expect("stage should succeed");
    assert_eq!(staged.len(), 32, "5 static + 27 PNGs = 32 staged files");
    assert!(staging.path().join(THEME_ANCIENT.wad_filename).exists());
    assert!(staging.path().join("textures/wall_basecolor.png").exists());
    assert!(staging.path().join("textures/wall_norm.png").exists());
    assert!(staging.path().join("textures/wall_gloss.png").exists());
    assert!(staging.path().join("LICENSE").exists());
}

// ── Hash computation test ─────────────────────────────────────────────────

#[test]
fn hashes_are_complete() {
    for (dir, def) in [
        (&ancient_dir(), &THEME_ANCIENT),
        (&theme_root().join("richness_egyptian_v1"), &THEME_EGYPTIAN),
        (
            &theme_root().join("richness_brutalist_v1"),
            &THEME_BRUTALIST,
        ),
    ] {
        let hashes = richness_assets::compute_richness_hashes(dir, def)
            .expect("hash computation should succeed");
        assert_eq!(hashes.len(), 32, "32 assets per theme");
        for (name, hash) in &hashes {
            assert_eq!(hash.len(), 64, "SHA-256 must be 64 hex chars for {name}");
            assert!(
                hash.chars().all(|c| c.is_ascii_hexdigit()),
                "non-hex hash for {name}"
            );
        }
    }
}

// ── Helpers for test WAD construction ─────────────────────────────────────

/// Create a valid minimal RGB PNG image (8-bit, no alpha).
fn make_valid_rgb_png(width: u32, height: u32) -> Vec<u8> {
    let mut buf = Vec::new();
    // Minimal PNG encoder for test purposes
    // Signature
    buf.extend_from_slice(b"\x89PNG\r\n\x1a\n");

    // IHDR
    let mut ihdr_data = Vec::new();
    ihdr_data.extend_from_slice(&width.to_be_bytes());
    ihdr_data.extend_from_slice(&height.to_be_bytes());
    ihdr_data.push(8); // bit depth
    ihdr_data.push(2); // color type: RGB
    ihdr_data.push(0); // compression
    ihdr_data.push(0); // filter
    ihdr_data.push(0); // interlace
    write_png_chunk(&mut buf, b"IHDR", &ihdr_data);

    // Single IDAT with one row of pixels (uncompressed, stored)
    let row_size = (width as usize) * 3 + 1; // filter byte + RGB
    let mut raw_data = Vec::with_capacity(row_size * height as usize);
    for _ in 0..height {
        raw_data.push(0); // filter: None
        for _ in 0..width {
            raw_data.push(128); // R
            raw_data.push(128); // G
            raw_data.push(128); // B
        }
    }

    // Compress with zlib/deflate
    let compressed = deflate(&raw_data);
    write_png_chunk(&mut buf, b"IDAT", &compressed);

    // IEND
    write_png_chunk(&mut buf, b"IEND", &[]);

    buf
}

fn write_png_chunk(buf: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) {
    buf.extend_from_slice(&(data.len() as u32).to_be_bytes());
    let crc_start = buf.len();
    buf.extend_from_slice(kind);
    let crc_data_start = buf.len();
    buf.extend_from_slice(data);
    let crc_data = &buf[crc_start..];
    let crc = png_crc32(crc_data);
    buf.extend_from_slice(&crc.to_be_bytes());
    let _ = crc_data_start;
}

fn png_crc32(bytes: &[u8]) -> u32 {
    let mut crc = !0u32;
    for &byte in bytes {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            crc = if crc & 1 == 1 {
                (crc >> 1) ^ 0xedb8_8320
            } else {
                crc >> 1
            };
        }
    }
    !crc
}

/// Minimal DEFLATE compressor producing a single stored block (no compression).
fn deflate(data: &[u8]) -> Vec<u8> {
    // zlib header: CMF=0x78 (deflate, window=32K), FLG=0x01 (no dict, level 0)
    let mut out = vec![0x78, 0x01];

    // Stored block for each chunk
    let mut offset = 0usize;
    while offset < data.len() {
        let remaining = data.len() - offset;
        let chunk_size = remaining.min(65535);
        let is_final = offset + chunk_size >= data.len();

        // Block header: BFINAL + BTYPE=00 (stored)
        out.push(if is_final { 1 } else { 0 });
        // Length and one's complement
        let len = chunk_size as u16;
        let nlen = !len;
        out.extend_from_slice(&len.to_le_bytes());
        out.extend_from_slice(&nlen.to_le_bytes());
        out.extend_from_slice(&data[offset..offset + chunk_size]);
        offset += chunk_size;
    }

    // Adler-32 checksum
    let adler = adler32(data);
    out.extend_from_slice(&adler.to_be_bytes());

    out
}

fn adler32(data: &[u8]) -> u32 {
    let mut a: u32 = 1;
    let mut b: u32 = 0;
    for &byte in data {
        a = (a + byte as u32) % 65521;
        b = (b + a) % 65521;
    }
    (b << 16) | a
}

/// Create a minimal valid WAD2 archive with the given identities.
/// Each miptex is 64×64 with 4 mip levels.
fn make_minimal_wad(identities: &[&str]) -> Vec<u8> {
    let mut out = Vec::new();
    let mut directory = Vec::new();
    let mut filepos: i32 = 12; // after header

    // Header placeholder
    let num_entries = identities.len() as i32;

    for name in identities {
        let miptex = make_minimal_miptex(name, 64);
        // WAD directory entry
        let mut entry = Vec::new();
        entry.extend_from_slice(&filepos.to_le_bytes());
        let size = miptex.len() as i32;
        entry.extend_from_slice(&size.to_le_bytes()); // disksize
        entry.extend_from_slice(&size.to_le_bytes()); // size
        entry.push(0x44); // type
        entry.push(0); // compression
        entry.extend_from_slice(&0u16.to_le_bytes()); // pad
        let mut name_bytes = [0u8; 16];
        let name_ascii = name.as_bytes();
        let copy_len = name_ascii.len().min(16);
        name_bytes[..copy_len].copy_from_slice(&name_ascii[..copy_len]);
        entry.extend_from_slice(&name_bytes);
        directory.push(entry);

        out.extend_from_slice(&miptex);
        filepos += size;
    }

    let dir_offset = filepos;

    // Write header
    let mut header = Vec::new();
    header.extend_from_slice(b"WAD2");
    header.extend_from_slice(&num_entries.to_le_bytes());
    header.extend_from_slice(&dir_offset.to_le_bytes());
    // Prepend header
    let mut full = header;
    full.append(&mut out);

    // Append directory
    for entry in directory {
        full.extend_from_slice(&entry);
    }

    full
}

fn make_minimal_miptex(name: &str, size: u32) -> Vec<u8> {
    let mut out = Vec::new();

    // Miptex header
    let mut name_bytes = [0u8; 16];
    let name_ascii = name.as_bytes();
    let copy_len = name_ascii.len().min(16);
    name_bytes[..copy_len].copy_from_slice(&name_ascii[..copy_len]);
    out.extend_from_slice(&name_bytes);

    out.extend_from_slice(&size.to_le_bytes()); // width
    out.extend_from_slice(&size.to_le_bytes()); // height

    // Mip offsets placeholder (4 offsets)
    let offsets_pos = out.len();
    for _ in 0..4 {
        out.extend_from_slice(&0u32.to_le_bytes());
    }

    // Mip data
    let mip_sizes = [
        (size as usize) * (size as usize),
        (size as usize / 2) * (size as usize / 2),
        (size as usize / 4) * (size as usize / 4),
        (size as usize / 8) * (size as usize / 8),
    ];

    let header_size = (16 + 4 + 4 + 4 * 4) as u32; // name + width + height + 4 offsets
    let mut current_offset = header_size;

    for (i, &mip_size) in mip_sizes.iter().enumerate() {
        // Write offset
        let off_bytes = current_offset.to_le_bytes();
        let off_pos = offsets_pos + i * 4;
        out[off_pos..off_pos + 4].copy_from_slice(&off_bytes);

        // Write mip pixels (all zeros = black)
        for _ in 0..mip_size {
            out.push(0);
        }
        current_offset += mip_size as u32;
    }

    out
}

// ── Material-role fixture compilation tests ───────────────────────────────
//
// Each test compiles a minimal Quake .map referencing a single textured face
// from the theme WAD, using ericw-tools qbsp+light, then strict-loads the
// resulting BSP to verify the material role is warned-free.
//
// These tests follow the skip-when-tools-unavailable pattern: if ericw-tools
// is not installed, the test is skipped (not failed).

/// Check if the ericw-tools compiler toolchain is available.
fn ericw_tools_available() -> bool {
    std::process::Command::new("qbsp")
        .arg("--version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// Build a minimal .map string for a single wall-textured cube.
fn minimal_map_text(wad_name: &str) -> String {
    format!(
        r#"{{
"classname" "worldspawn"
"wad" "{wad_name}"
{{
( 64 64 64 ) ( 64 -64 64 ) ( 64 64 -64 ) wall 0 0 0 1.0 1.0
( -64 -64 64 ) ( -64 64 64 ) ( -64 -64 -64 ) wall 0 0 0 1.0 1.0
( -64 64 64 ) ( 64 64 64 ) ( -64 64 -64 ) wall 0 0 0 1.0 1.0
( 64 -64 64 ) ( -64 -64 64 ) ( 64 -64 -64 ) wall 0 0 0 1.0 1.0
( -64 -64 -64 ) ( 64 -64 -64 ) ( -64 64 -64 ) wall 0 0 0 1.0 1.0
( 64 64 -64 ) ( -64 64 -64 ) ( 64 -64 -64 ) wall 0 0 0 1.0 1.0
}}
}}
"#,
        wad_name = wad_name
    )
}

/// Run a material-role fixture compilation for a single theme.
fn compile_fixture(theme_dir: &std::path::Path, wad_name: &str) -> Result<(), String> {
    if !ericw_tools_available() {
        return Err("ericw-tools not available".into());
    }

    let temp = tempfile::tempdir().map_err(|e| format!("tempdir: {e}"))?;
    let work = temp.path();

    // Copy WAD and palette
    let wad_src = theme_dir.join(wad_name);
    let palette_src = theme_dir.join("palette.lmp");
    let wad_dst = work.join(wad_name);
    let palette_dst = work.join("palette.lmp");
    std::fs::copy(&wad_src, &wad_dst).map_err(|e| format!("copy wad: {e}"))?;
    std::fs::copy(&palette_src, &palette_dst).map_err(|e| format!("copy palette: {e}"))?;

    // Write .map
    let map_text = minimal_map_text(wad_name);
    let map_path = work.join("fixture.map");
    std::fs::write(&map_path, &map_text).map_err(|e| format!("write map: {e}"))?;

    // Run qbsp
    let qbsp = std::process::Command::new("qbsp")
        .arg("-bsp2")
        .arg(&map_path)
        .current_dir(work)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .output()
        .map_err(|e| format!("qbsp spawn: {e}"))?;

    if !qbsp.status.success() {
        let stderr = String::from_utf8_lossy(&qbsp.stderr);
        let stdout = String::from_utf8_lossy(&qbsp.stdout);
        return Err(format!("qbsp failed: {stdout}\n{stderr}"));
    }

    // Check for warnings in qbsp output
    let qbsp_out = format!(
        "{}{}",
        String::from_utf8_lossy(&qbsp.stdout),
        String::from_utf8_lossy(&qbsp.stderr)
    );
    if qbsp_out.to_ascii_lowercase().contains("warning")
        || qbsp_out
            .to_ascii_lowercase()
            .contains("unable to find texture")
    {
        return Err(format!("qbsp produced warnings: {qbsp_out}"));
    }

    // Run light
    let bsp_path = work.join("fixture.bsp");
    if !bsp_path.exists() {
        return Err("qbsp did not produce fixture.bsp".into());
    }

    let light = std::process::Command::new("light")
        .arg("-bsp2")
        .arg(&bsp_path)
        .current_dir(work)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .output()
        .map_err(|e| format!("light spawn: {e}"))?;

    if !light.status.success() {
        let stderr = String::from_utf8_lossy(&light.stderr);
        return Err(format!("light failed: {stderr}"));
    }

    let light_out = format!(
        "{}{}",
        String::from_utf8_lossy(&light.stdout),
        String::from_utf8_lossy(&light.stderr)
    );
    if light_out.to_ascii_lowercase().contains("warning") {
        return Err(format!("light produced warnings: {light_out}"));
    }

    // Strict-load the BSP
    let bsp_bytes = std::fs::read(&bsp_path).map_err(|e| format!("read bsp: {e}"))?;
    let palette_bytes = std::fs::read(&palette_dst).map_err(|e| format!("read palette: {e}"))?;

    let wad_bytes = std::fs::read(&wad_dst).map_err(|e| format!("read wad: {e}"))?;
    let wad_archives = vec![(wad_name.to_string(), wad_bytes)];

    let load_options = bsp::LoadOptions {
        strict: true,
        palette: Some(palette_bytes),
        lit_data: None,
        wad_archives,
        texture_overrides: Vec::new(),
        source_identity: "richness-fixture-test".to_string(),
    };

    let world =
        bsp::BspLoader::load(&bsp_bytes, &load_options).map_err(|report| format!("{report}"))?;

    if !world.diagnostics.is_empty() {
        let messages: Vec<_> = world
            .diagnostics
            .iter()
            .map(|d| d.message.as_str())
            .collect();
        return Err(format!("strict-load diagnostics: {messages:?}"));
    }

    Ok(())
}

#[test]
fn material_role_fixture_ancient() {
    let dir = theme_root().join("richness_ancient_v1");
    match compile_fixture(&dir, "richness_ancient_v1.wad") {
        Ok(()) => {}
        Err(e) if e.contains("not available") => {
            eprintln!("SKIP: material-role fixture requires ericw-tools");
        }
        Err(e) => panic!("ancient fixture compilation failed: {e}"),
    }
}

#[test]
fn material_role_fixture_egyptian() {
    let dir = theme_root().join("richness_egyptian_v1");
    match compile_fixture(&dir, "richness_egyptian_v1.wad") {
        Ok(()) => {}
        Err(e) if e.contains("not available") => {
            eprintln!("SKIP: material-role fixture requires ericw-tools");
        }
        Err(e) => panic!("egyptian fixture compilation failed: {e}"),
    }
}

#[test]
fn material_role_fixture_brutalist() {
    let dir = theme_root().join("richness_brutalist_v1");
    match compile_fixture(&dir, "richness_brutalist_v1.wad") {
        Ok(()) => {}
        Err(e) if e.contains("not available") => {
            eprintln!("SKIP: material-role fixture requires ericw-tools");
        }
        Err(e) => panic!("brutalist fixture compilation failed: {e}"),
    }
}
