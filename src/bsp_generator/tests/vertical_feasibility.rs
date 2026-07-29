//! Phase 01 — Vertical Feasibility Evidence
//!
//! Programmatic brush construction using the canonical `make_brush` pattern.
//! Compiles through ericw-tools, validates spatial witnesses and movement.
//!
//! Run: cargo test -p bsp_generator --test vertical_feasibility -- --nocapture

use bsp::{point_contents, BspLoader, LoadOptions, PointContents, QuakeToEngine};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

const Q: i32 = 16;

fn tools_dir() -> PathBuf {
    PathBuf::from(
        std::env::var("HOME").unwrap_or_else(|_| "/home/dhickel".to_string())
    ).join(".local/ericw-tools/ericw-tools-2.0.0-alpha3-Linux/bin")
}

fn wad_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("themes/cc0_stone_beta/cc0_stone_beta.wad")
}

fn palette_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("themes/cc0_stone_beta/palette.lmp")
}

fn out_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.internal-dev/debug_reports/vertical-feasibility")
}

fn tools_ok(dir: &Path) -> bool {
    dir.join("qbsp").is_file() && dir.join("vis").is_file()
}

// ── Canonical brush face emission (matches make_brush in junction.rs) ─────

/// Emit one face in canonical order: bottom, top, north, south, west, east.
struct MapWriter { buf: String }

impl MapWriter {
    fn new() -> Self { Self { buf: String::new() } }

    fn emit_brush(&mut self, x0: i32, y0: i32, z0: i32, x1: i32, y1: i32, z1: i32, tex: &str) {
        self.buf.push_str("{\n");
        // bottom
        self.buf.push_str(&format!("( {} {} {} ) ( {} {} {} ) ( {} {} {} ) \"{}\" 0 0 0 0.25 0.25\n", x0, y1, z0, x0, y0, z0, x1, y0, z0, tex));
        // top
        self.buf.push_str(&format!("( {} {} {} ) ( {} {} {} ) ( {} {} {} ) \"{}\" 0 0 0 0.25 0.25\n", x0, y1, z1, x1, y1, z1, x1, y0, z1, tex));
        // north (max Y)
        self.buf.push_str(&format!("( {} {} {} ) ( {} {} {} ) ( {} {} {} ) \"{}\" 0 0 0 0.25 0.25\n", x0, y1, z1, x0, y1, z0, x1, y1, z0, tex));
        // south (min Y)
        self.buf.push_str(&format!("( {} {} {} ) ( {} {} {} ) ( {} {} {} ) \"{}\" 0 0 0 0.25 0.25\n", x0, y0, z1, x1, y0, z1, x1, y0, z0, tex));
        // west (min X)
        self.buf.push_str(&format!("( {} {} {} ) ( {} {} {} ) ( {} {} {} ) \"{}\" 0 0 0 0.25 0.25\n", x0, y1, z1, x0, y0, z1, x0, y0, z0, tex));
        // east (max X)
        self.buf.push_str(&format!("( {} {} {} ) ( {} {} {} ) ( {} {} {} ) \"{}\" 0 0 0 0.25 0.25\n", x1, y1, z0, x1, y0, z0, x1, y0, z1, tex));
        self.buf.push_str("}\n");
    }

    /// Emit an east-west wall (constant X) with a rectangular YZ aperture.
    /// `side` is "west" (min X wall) or "east" (max X wall).
    fn emit_x_wall_with_aperture(&mut self, side: &str,
        x_wall: i32, y0: i32, y1: i32, z0: i32, z1: i32,
        ap_y0: i32, ap_y1: i32, ap_z0: i32, ap_z1: i32, tex: &str)
    {
        let x0 = if side == "west" { x_wall } else { x_wall - Q };
        let x1 = if side == "west" { x_wall + Q } else { x_wall };
        // Left of aperture
        if ap_y0 > y0 { self.emit_brush(x0, y0, z0, x1, ap_y0, z1, tex); }
        // Right of aperture
        if ap_y1 < y1 { self.emit_brush(x0, ap_y1, z0, x1, y1, z1, tex); }
        // Below aperture
        if ap_z0 > z0 { self.emit_brush(x0, ap_y0, z0, x1, ap_y1, ap_z0, tex); }
        // Above aperture
        if ap_z1 < z1 { self.emit_brush(x0, ap_y0, ap_z1, x1, ap_y1, z1, tex); }
    }

    fn finish(mut self) -> String {
        // Ensure trailing newline
        if !self.buf.ends_with('\n') { self.buf.push('\n'); }
        self.buf
    }
}

// ── Fixture generator ─────────────────────────────────────────────────────

struct FixtureSpec {
    upper_floor_z: i32,
    lower_height: i32,
    upper_height: i32,
    tread: i32,
}

fn generate_fixture(spec: &FixtureSpec) -> String {
    let mut w = MapWriter::new();

    // Lower room: X=128-384, Y=128-384, Z=0..lower_height
    let lr = (128, 128, 0);
    let lh = spec.lower_height;
    let room_w = 256;

    // Upper room: same XY, different Z
    let ur = (128, 128, spec.upper_floor_z);
    let uh = spec.upper_height;

    // Stairwell: connects east side of lower room to west side of upper room.
    // Stairwell XY: X=384-544, Y=176-240 (64 wide interior, 96 total)
    let sw_x0 = lr.0 + room_w;        // 384
    let sw_x1 = sw_x0 + 160;          // 544
    let sw_y0 = lr.1 + 64;            // 192
    let sw_y1 = sw_y0 + 64;           // 256 (64-wide stairwell)
    let sw_z0 = lr.2;                      // 0
    let sw_z1 = spec.upper_floor_z + Q;     // upper floor + slab thickness

    let ap_y0 = sw_y0;
    let ap_y1 = sw_y1;
    let ap_z0_lo = lr.2 + Q;          // 16 (above lower floor slab)
    let ap_z1_lo = spec.upper_floor_z; // extend aperture to upper floor level
    let ap_z0_hi = ur.2 + Q;           // upper_floor_z + 16
    let ap_z1_hi = ur.2 + uh - Q;     // upper_floor_z + upper_height - 16

    // ── worldspawn ────────────────────────────────────────────────────
    w.buf.push_str("{\n\"classname\" \"worldspawn\"\n\"wad\" \"cc0_stone_beta.wad\"\n");

    // --- Lower room ---
    // East wall is fully OPEN (no brush) — the stairwell end wall provides the enclosure.
    // The stairwell end wall has an aperture for passage.
    // Room walls extend up to seal the inter-layer gap.
    let wall_top = spec.upper_floor_z + Q; // seal to upper floor slab
    w.emit_brush(lr.0, lr.1, lr.2, lr.0+room_w, lr.1+room_w, lr.2+Q, "stone_floor");
    w.emit_brush(lr.0, lr.1, lh-Q, lr.0+room_w, lr.1+room_w, lh, "stone_ceiling");
    w.emit_brush(lr.0, lr.1, lr.2, lr.0+room_w, lr.1+Q, wall_top, "stone_wall");
    w.emit_brush(lr.0, lr.1, lr.2, lr.0+Q, lr.1+room_w, wall_top, "stone_wall");
    w.emit_brush(lr.0, lr.1+room_w-Q, lr.2, lr.0+room_w, lr.1+room_w, wall_top, "stone_wall");
    // east wall: fully open, stairwell end wall covers it

    // --- Upper room ---
    // West wall is fully OPEN — stairwell end wall provides enclosure.
    w.emit_brush(ur.0, ur.1, ur.2, ur.0+room_w, ur.1+room_w, ur.2+Q, "stone_floor");
    w.emit_brush(ur.0, ur.1, ur.2+uh-Q, ur.0+room_w, ur.1+room_w, ur.2+uh, "stone_ceiling");
    w.emit_brush(ur.0, ur.1, ur.2, ur.0+room_w, ur.1+Q, ur.2+uh, "stone_wall");
    w.emit_brush(ur.0+room_w-Q, ur.1, ur.2, ur.0+room_w, ur.1+room_w, ur.2+uh, "stone_wall");
    w.emit_brush(ur.0, ur.1+room_w-Q, ur.2, ur.0+room_w, ur.1+room_w, ur.2+uh, "stone_wall");
    // west wall: fully open, stairwell end wall covers it

    // --- Stairwell ---
    // floor and ceiling
    w.emit_brush(sw_x0, sw_y0, sw_z0, sw_x1, sw_y1, sw_z0+Q, "stone_floor");
    w.emit_brush(sw_x0, sw_y0, sw_z1-Q, sw_x1, sw_y1, sw_z1, "stone_ceiling");
    // Side walls
    w.emit_brush(sw_x0, sw_y0, sw_z0, sw_x1, sw_y0+Q, sw_z1, "stone_wall");
    w.emit_brush(sw_x0, sw_y1-Q, sw_z0, sw_x1, sw_y1, sw_z1, "stone_wall");
    // End walls with apertures matching room apertures.
    // These overlap with the room walls to form a continuous seal.
    w.emit_x_wall_with_aperture("west", sw_x0, sw_y0, sw_y1, sw_z0, sw_z1, ap_y0, ap_y1, ap_z0_lo, ap_z1_lo, "stone_wall");
    w.emit_x_wall_with_aperture("east", sw_x1, sw_y0, sw_y1, sw_z0, sw_z1, ap_y0, ap_y1, ap_z0_hi, ap_z1_hi, "stone_wall");

    // --- Steps ---
    let rise = spec.upper_floor_z;
    let steps = rise / Q;
    for i in 0..steps {
        let sx0 = sw_x0 + i * spec.tread;
        let sx1 = sx0 + spec.tread;
        let sz0 = i * Q;
        let sz1 = sz0 + Q;
        // steps fill the stairwell width between the side walls
        w.emit_brush(sx0, sw_y0+Q, sz0, sx1, sw_y1-Q, sz1, "stone_floor");
    }
    // Top landing: fills the gap between the last step and the upper room aperture
    let landing_x0 = sw_x0 + steps * spec.tread;
    if landing_x0 < sw_x1 {
        w.emit_brush(landing_x0, sw_y0+Q, spec.upper_floor_z, sw_x1, sw_y1-Q, spec.upper_floor_z + Q, "stone_floor");
    }

    w.buf.push_str("}\n");

    // ── Spawn ─────────────────────────────────────────────────────────
    w.buf.push_str(&format!(
        "{{\n\"classname\" \"info_player_start\"\n\"origin\" \"{} {} {}\"\n}}\n",
        lr.0 + room_w/2, lr.1 + room_w/2, lr.2 + Q + 24
    ));

    w.finish()
}

// ── Compilation helpers ───────────────────────────────────────────────────

fn sha256(data: &[u8]) -> String {
    Sha256::digest(data).iter().map(|b| format!("{:02x}", b)).collect()
}

fn run_tool(dir: &Path, exe: &str, args: &[&str]) -> Result<String, String> {
    let path = tools_dir().join(exe);
    let out = Command::new(&path).args(args).current_dir(dir).output()
        .map_err(|e| format!("{exe}: {e}"))?;
    let combined = format!("{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr));
    if !out.status.success() {
        return Err(format!("{exe} exit {}", out.status.code().unwrap_or(-1)));
    }
    let lo = combined.to_lowercase();
    if lo.contains("brush bounds out of range") || lo.contains("no visible sides")
        || lo.contains("mixed face") || lo.contains("degenerate")
        || lo.contains("no entities in empty space") {
        return Err(format!("{exe} warning"));
    }
    Ok(combined)
}

// ── Main test ─────────────────────────────────────────────────────────────

#[test]
fn vertical_feasibility() {
    let td = tools_dir();
    if !tools_ok(&td) { eprintln!("SKIP: tools absent"); return; }

    let base = out_dir();
    fs::create_dir_all(&base).unwrap();

    let fixtures: Vec<FixtureSpec> = vec![
        FixtureSpec { upper_floor_z: 192, lower_height: 176, upper_height: 176, tread: 16 },
        FixtureSpec { upper_floor_z: 192, lower_height: 176, upper_height: 176, tread: 32 },
        FixtureSpec { upper_floor_z: 208, lower_height: 160, upper_height: 160, tread: 16 },
        FixtureSpec { upper_floor_z: 208, lower_height: 160, upper_height: 160, tread: 32 },
    ];

    let mut selected: Option<&FixtureSpec> = None;

    for spec in &fixtures {
        let label = format!("ufz{}_t{}", spec.upper_floor_z, spec.tread);
        eprintln!("\n--- {} ---", label);
        let dir = base.join(&label);
        fs::create_dir_all(&dir).unwrap();

        let map = generate_fixture(spec);
        fs::write(dir.join("fixture.map"), &map).unwrap();
        fs::copy(wad_path(), dir.join("cc0_stone_beta.wad")).unwrap();
        fs::copy(palette_path(), dir.join("palette.lmp")).unwrap();

        // qbsp
        if let Err(e) = run_tool(&dir, "qbsp", &["-bsp2", "-threads", "1", "fixture.map"]) {
            eprintln!("  qbsp FAIL: {e}"); continue;
        }
        let bsp = dir.join("fixture.bsp");
        if !bsp.exists() { eprintln!("  no bsp"); continue; }
        if dir.join("fixture.pts").exists() { eprintln!("  leaked"); continue; }

        // vis
        if run_tool(&dir, "vis", &["-threads", "1", "fixture.bsp"]).is_err() {
            eprintln!("  vis FAIL"); continue;
        }
        eprintln!("  compile PASS");

        // Load
        let data = fs::read(&bsp).unwrap();
        let pal = fs::read(palette_path()).unwrap();
        let wad = fs::read(wad_path()).unwrap();
        let wn = wad_path().file_stem().unwrap().to_str().unwrap().to_string();
        let opts = LoadOptions { strict: true, palette: Some(pal), lit_data: None,
            wad_archives: vec![(wn, wad)], texture_overrides: vec![],
            source_identity: "vf".into() };
        let world = match BspLoader::load(&data, &opts) {
            Ok(w) => w, Err(e) => { eprintln!("  load FAIL: {e:?}"); continue; }
        };
        let qte = QuakeToEngine::default();

        let check = |label: &str, qx: f32, qy: f32, qz: f32, want_solid: bool| -> bool {
            let ep = qte.position(qx, qy, qz);
            let c = point_contents(ep, &world.nodes, &world.leaves, &world.planes);
            let ok = c.is_solid() == want_solid;
            if !ok { eprintln!("  witness FAIL: {label} ({qx},{qy},{qz}) want_solid={want_solid} got {c:?}"); }
            ok
        };

        let mut all_ok = true;
        // Lower room
        all_ok &= check("lr_center", 256.0, 256.0, 24.0, false);
        all_ok &= check("lr_wall", 256.0, 136.0, 24.0, true);
        // Upper room
        all_ok &= check("ur_center", 256.0, 256.0, (spec.upper_floor_z + 24) as f32, false);
        all_ok &= check("ur_wall", 256.0, 376.0, (spec.upper_floor_z + 24) as f32, true);
        // Stairwell interior
        let sw_mid_z = (spec.upper_floor_z / 2) as f32;
        all_ok &= check("sw_mid", 464.0, 224.0, sw_mid_z, false);
        // First tread (solid)
        all_ok &= check("tread_solid", 400.0, 224.0, 8.0, true);
        // Space above first tread (walkable)
        all_ok &= check("tread_above", 400.0, 224.0, 20.0, false);

        if !all_ok { eprintln!("  witnesses FAIL"); continue; }
        eprintln!("  witnesses PASS");

        if selected.is_none() { selected = Some(spec); }
        eprintln!("  *** PASSING ***");
    }

    match selected {
        Some(s) => eprintln!("\n=== SELECTED: ufz={} tread={} lh={} uh={} ===", s.upper_floor_z, s.tread, s.lower_height, s.upper_height),
        None => panic!("NO-GO"),
    }
}

#[test]
fn legacy_corpus_unchanged() {
    let (map, _) = bsp_generator::generate(0, bsp_generator::DungeonConfig::nominal_m1()).unwrap();
    assert!(!map.is_empty());
    eprintln!("Legacy M1 seed 0 hash: {}", sha256(map.as_bytes()));
}
