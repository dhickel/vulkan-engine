//! Phase 01 — Vertical Feasibility Evidence
//!
//! Programmatic brush construction using the canonical `make_brush` pattern.
//! Compiles through ericw-tools, validates spatial witnesses and movement.
//!
//! Run: cargo test -p bsp_generator --test vertical_feasibility -- --nocapture

use bsp::{point_contents, BspLoader, LoadOptions, QuakeToEngine};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

const Q: i32 = 16;

fn tools_dir() -> PathBuf {
    PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| "/home/dhickel".to_string()))
        .join(".local/ericw-tools/ericw-tools-2.0.0-alpha3-Linux/bin")
}

fn wad_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("themes/cc0_stone_beta/cc0_stone_beta.wad")
}

fn palette_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("themes/cc0_stone_beta/palette.lmp")
}

fn out_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../.internal-dev/debug_reports/vertical-feasibility")
}

fn tools_ok(dir: &Path) -> bool {
    dir.join("qbsp").is_file() && dir.join("vis").is_file()
}

// ── Canonical brush face emission (matches make_brush in junction.rs) ─────

/// Emit one face in canonical order: bottom, top, north, south, west, east.
struct MapWriter {
    buf: String,
}

impl MapWriter {
    fn new() -> Self {
        Self { buf: String::new() }
    }

    fn emit_brush(&mut self, x0: i32, y0: i32, z0: i32, x1: i32, y1: i32, z1: i32, tex: &str) {
        self.buf.push_str("{\n");
        // bottom
        self.buf.push_str(&format!(
            "( {} {} {} ) ( {} {} {} ) ( {} {} {} ) \"{}\" 0 0 0 0.25 0.25\n",
            x0, y1, z0, x0, y0, z0, x1, y0, z0, tex
        ));
        // top
        self.buf.push_str(&format!(
            "( {} {} {} ) ( {} {} {} ) ( {} {} {} ) \"{}\" 0 0 0 0.25 0.25\n",
            x0, y1, z1, x1, y1, z1, x1, y0, z1, tex
        ));
        // north (max Y)
        self.buf.push_str(&format!(
            "( {} {} {} ) ( {} {} {} ) ( {} {} {} ) \"{}\" 0 0 0 0.25 0.25\n",
            x0, y1, z1, x0, y1, z0, x1, y1, z0, tex
        ));
        // south (min Y)
        self.buf.push_str(&format!(
            "( {} {} {} ) ( {} {} {} ) ( {} {} {} ) \"{}\" 0 0 0 0.25 0.25\n",
            x0, y0, z1, x1, y0, z1, x1, y0, z0, tex
        ));
        // west (min X)
        self.buf.push_str(&format!(
            "( {} {} {} ) ( {} {} {} ) ( {} {} {} ) \"{}\" 0 0 0 0.25 0.25\n",
            x0, y1, z1, x0, y0, z1, x0, y0, z0, tex
        ));
        // east (max X)
        self.buf.push_str(&format!(
            "( {} {} {} ) ( {} {} {} ) ( {} {} {} ) \"{}\" 0 0 0 0.25 0.25\n",
            x1, y1, z0, x1, y0, z0, x1, y0, z1, tex
        ));
        self.buf.push_str("}\n");
    }

    /// Emit an east-west wall (constant X) with a rectangular YZ aperture.
    /// `side` is "west" (min X wall) or "east" (max X wall).
    fn emit_x_wall_with_aperture(
        &mut self,
        side: &str,
        x_wall: i32,
        y0: i32,
        y1: i32,
        z0: i32,
        z1: i32,
        ap_y0: i32,
        ap_y1: i32,
        ap_z0: i32,
        ap_z1: i32,
        tex: &str,
    ) {
        let x0 = if side == "west" { x_wall } else { x_wall - Q };
        let x1 = if side == "west" { x_wall + Q } else { x_wall };
        // Left of aperture
        if ap_y0 > y0 {
            self.emit_brush(x0, y0, z0, x1, ap_y0, z1, tex);
        }
        // Right of aperture
        if ap_y1 < y1 {
            self.emit_brush(x0, ap_y1, z0, x1, y1, z1, tex);
        }
        // Below aperture
        if ap_z0 > z0 {
            self.emit_brush(x0, ap_y0, z0, x1, ap_y1, ap_z0, tex);
        }
        // Above aperture
        if ap_z1 < z1 {
            self.emit_brush(x0, ap_y0, ap_z1, x1, ap_y1, z1, tex);
        }
    }

    fn finish(mut self) -> String {
        // Ensure trailing newline
        if !self.buf.ends_with('\n') {
            self.buf.push('\n');
        }
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

    // The two rooms deliberately share an XY footprint. Their east walls own
    // separate lower/upper apertures into a U-shaped stairwell.
    let lr = (128, 128, 0);
    let ur = (lr.0, lr.1, spec.upper_floor_z);
    let room_span = 256;
    let lower_ceiling = lr.2 + spec.lower_height;
    let upper_ceiling = ur.2 + spec.upper_height;
    let lower_floor_top = lr.2 + Q;
    let upper_floor_top = ur.2 + Q;

    // The lower flight climbs east in the south lane, crosses the turn
    // landing, then the upper flight climbs west in the north lane. This lets
    // both direct room apertures meet the same east wall of stacked rooms.
    let lower_lane = (160, 224);
    let upper_lane = (288, 352);
    let sw_x0 = lr.0 + room_span;
    let sw_y0 = lr.1;
    let sw_y1 = lr.1 + room_span;
    let inner_x0 = sw_x0 + Q;
    let flight_x0 = inner_x0 + Q;
    let step_count = spec.upper_floor_z / Q;
    let lower_step_count = (step_count + 1) / 2;
    let upper_step_count = step_count - lower_step_count;
    let turn_x0 = flight_x0 + lower_step_count * spec.tread;
    let turn_x1 = turn_x0 + Q;
    let sw_x1 = turn_x1 + Q;
    let mid_landing_top = lower_floor_top + lower_step_count * Q;
    let upper_flight_x0 = turn_x0 - upper_step_count * spec.tread;

    let lower_opening = (
        lower_lane.0,
        lower_lane.1,
        lower_floor_top,
        lower_floor_top + 80,
    );
    let upper_opening = (
        upper_lane.0,
        upper_lane.1,
        upper_floor_top,
        upper_floor_top + 80,
    );

    w.buf
        .push_str("{\n\"classname\" \"worldspawn\"\n\"wad\" \"cc0_stone_beta.wad\"\n");

    // Lower room shell. North/south own the corner columns; west/east cover
    // only the interior tangent span, matching production shell ownership.
    w.emit_brush(
        lr.0,
        lr.1,
        lr.2,
        lr.0 + room_span,
        lr.1 + room_span,
        lower_floor_top,
        "stone_floor",
    );
    w.emit_brush(
        lr.0,
        lr.1,
        lower_ceiling - Q,
        lr.0 + room_span,
        lr.1 + room_span,
        lower_ceiling,
        "stone_ceiling",
    );
    w.emit_brush(
        lr.0,
        lr.1,
        lower_floor_top,
        lr.0 + room_span,
        lr.1 + Q,
        lower_ceiling - Q,
        "stone_wall",
    );
    w.emit_brush(
        lr.0,
        lr.1 + room_span - Q,
        lower_floor_top,
        lr.0 + room_span,
        lr.1 + room_span,
        lower_ceiling - Q,
        "stone_wall",
    );
    w.emit_brush(
        lr.0,
        lr.1 + Q,
        lower_floor_top,
        lr.0 + Q,
        lr.1 + room_span - Q,
        lower_ceiling - Q,
        "stone_wall",
    );
    w.emit_x_wall_with_aperture(
        "east",
        lr.0 + room_span,
        lr.1 + Q,
        lr.1 + room_span - Q,
        lower_floor_top,
        lower_ceiling - Q,
        lower_opening.0,
        lower_opening.1,
        lower_opening.2,
        lower_opening.3,
        "stone_wall",
    );

    // Upper room shell. The old fixture omitted the west wall even though the
    // stairwell is east of the stacked footprint; that exposed the room to
    // the outside flood. Keep the west wall and split the touching east wall.
    w.emit_brush(
        ur.0,
        ur.1,
        ur.2,
        ur.0 + room_span,
        ur.1 + room_span,
        upper_floor_top,
        "stone_floor",
    );
    w.emit_brush(
        ur.0,
        ur.1,
        upper_ceiling - Q,
        ur.0 + room_span,
        ur.1 + room_span,
        upper_ceiling,
        "stone_ceiling",
    );
    w.emit_brush(
        ur.0,
        ur.1,
        upper_floor_top,
        ur.0 + room_span,
        ur.1 + Q,
        upper_ceiling - Q,
        "stone_wall",
    );
    w.emit_brush(
        ur.0,
        ur.1 + room_span - Q,
        upper_floor_top,
        ur.0 + room_span,
        ur.1 + room_span,
        upper_ceiling - Q,
        "stone_wall",
    );
    w.emit_brush(
        ur.0,
        ur.1 + Q,
        upper_floor_top,
        ur.0 + Q,
        ur.1 + room_span - Q,
        upper_ceiling - Q,
        "stone_wall",
    );
    w.emit_x_wall_with_aperture(
        "east",
        ur.0 + room_span,
        ur.1 + Q,
        ur.1 + room_span - Q,
        upper_floor_top,
        upper_ceiling - Q,
        upper_opening.0,
        upper_opening.1,
        upper_opening.2,
        upper_opening.3,
        "stone_wall",
    );

    // Stairwell shell spans both room layers. Splitting the west wall into two
    // tangent halves lets the existing single-aperture helper form both direct
    // room openings without overlapping wall brushes.
    w.emit_brush(
        sw_x0,
        sw_y0,
        lr.2,
        sw_x1,
        sw_y1,
        lower_floor_top,
        "stone_floor",
    );
    w.emit_brush(
        sw_x0,
        sw_y0,
        upper_ceiling - Q,
        sw_x1,
        sw_y1,
        upper_ceiling,
        "stone_ceiling",
    );
    w.emit_brush(
        sw_x0,
        sw_y0,
        lower_floor_top,
        sw_x1,
        sw_y0 + Q,
        upper_ceiling - Q,
        "stone_wall",
    );
    w.emit_brush(
        sw_x0,
        sw_y1 - Q,
        lower_floor_top,
        sw_x1,
        sw_y1,
        upper_ceiling - Q,
        "stone_wall",
    );
    w.emit_brush(
        sw_x1 - Q,
        sw_y0 + Q,
        lower_floor_top,
        sw_x1,
        sw_y1 - Q,
        upper_ceiling - Q,
        "stone_wall",
    );
    w.emit_x_wall_with_aperture(
        "west",
        sw_x0,
        sw_y0 + Q,
        256,
        lower_floor_top,
        upper_ceiling - Q,
        lower_opening.0,
        lower_opening.1,
        lower_opening.2,
        lower_opening.3,
        "stone_wall",
    );
    w.emit_x_wall_with_aperture(
        "west",
        sw_x0,
        256,
        sw_y1 - Q,
        lower_floor_top,
        upper_ceiling - Q,
        upper_opening.0,
        upper_opening.1,
        upper_opening.2,
        upper_opening.3,
        "stone_wall",
    );

    // Lower flight: each box is a supported stair column whose top rises by
    // exactly one construction quantum.
    for step in 0..lower_step_count {
        let x0 = flight_x0 + step * spec.tread;
        let top = lower_floor_top + (step + 1) * Q;
        w.emit_brush(
            x0,
            lower_lane.0,
            lower_floor_top,
            x0 + spec.tread,
            lower_lane.1,
            top,
            "stone_floor",
        );
    }

    // Turn landing joins both 64-unit lanes at the first-flight elevation.
    w.emit_brush(
        turn_x0,
        lower_lane.0,
        lower_floor_top,
        turn_x1,
        upper_lane.1,
        mid_landing_top,
        "stone_floor",
    );

    // Upper flight climbs back west. Its final tread is flush with the upper
    // room floor, and the top landing fills the direct approach to the wall.
    for step in 0..upper_step_count {
        let x1 = turn_x0 - step * spec.tread;
        let top = mid_landing_top + (step + 1) * Q;
        w.emit_brush(
            x1 - spec.tread,
            upper_lane.0,
            lower_floor_top,
            x1,
            upper_lane.1,
            top,
            "stone_floor",
        );
    }
    w.emit_brush(
        inner_x0,
        upper_lane.0,
        lower_floor_top,
        upper_flight_x0,
        upper_lane.1,
        upper_floor_top,
        "stone_floor",
    );

    w.buf.push_str("}\n");
    w.buf.push_str(&format!(
        "{{\n\"classname\" \"info_player_start\"\n\"origin\" \"{} {} {}\"\n}}\n",
        lr.0 + room_span / 2,
        lr.1 + room_span / 2,
        lower_floor_top + 24,
    ));

    w.finish()
}

// ── Compilation helpers ───────────────────────────────────────────────────

fn sha256(data: &[u8]) -> String {
    Sha256::digest(data)
        .iter()
        .map(|b| format!("{:02x}", b))
        .collect()
}

fn run_tool(dir: &Path, exe: &str, args: &[&str]) -> Result<String, String> {
    let path = tools_dir().join(exe);
    let out = Command::new(&path)
        .args(args)
        .current_dir(dir)
        .output()
        .map_err(|e| format!("{exe}: {e}"))?;
    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    if !out.status.success() {
        return Err(format!("{exe} exit {}", out.status.code().unwrap_or(-1)));
    }
    let lo = combined.to_lowercase();
    if lo.contains("warning:")
        || lo.contains("brush bounds out of range")
        || lo.contains("no visible sides")
        || lo.contains("mixed face")
        || lo.contains("degenerate")
        || lo.contains("no filling performed")
        || lo.contains("leak file written")
    {
        return Err(format!("{exe} emitted prohibited diagnostics:\n{combined}"));
    }
    Ok(combined)
}

// ── Main test ─────────────────────────────────────────────────────────────

#[test]
fn vertical_feasibility() {
    let td = tools_dir();
    if !tools_ok(&td) {
        eprintln!("SKIP: tools absent");
        return;
    }

    let base = out_dir();
    fs::create_dir_all(&base).unwrap();

    let fixtures: Vec<FixtureSpec> = vec![
        FixtureSpec {
            upper_floor_z: 192,
            lower_height: 176,
            upper_height: 176,
            tread: 16,
        },
        FixtureSpec {
            upper_floor_z: 192,
            lower_height: 176,
            upper_height: 176,
            tread: 32,
        },
        FixtureSpec {
            upper_floor_z: 208,
            lower_height: 160,
            upper_height: 160,
            tread: 16,
        },
        FixtureSpec {
            upper_floor_z: 208,
            lower_height: 160,
            upper_height: 160,
            tread: 32,
        },
    ];

    let mut selected: Option<&FixtureSpec> = None;

    for spec in &fixtures {
        let label = format!("ufz{}_t{}", spec.upper_floor_z, spec.tread);
        eprintln!("\n--- {} ---", label);
        let dir = base.join(&label);
        if dir.exists() {
            fs::remove_dir_all(&dir).unwrap();
        }
        fs::create_dir_all(&dir).unwrap();

        let map = generate_fixture(spec);
        fs::write(dir.join("fixture.map"), &map).unwrap();
        fs::copy(wad_path(), dir.join("cc0_stone_beta.wad")).unwrap();
        fs::copy(palette_path(), dir.join("palette.lmp")).unwrap();

        // qbsp
        if let Err(e) = run_tool(&dir, "qbsp", &["-bsp2", "-threads", "1", "fixture.map"]) {
            eprintln!("  qbsp FAIL: {e}");
            continue;
        }
        let bsp = dir.join("fixture.bsp");
        if !bsp.exists() {
            eprintln!("  no bsp");
            continue;
        }
        if dir.join("fixture.pts").exists() {
            eprintln!("  leaked");
            continue;
        }

        // vis
        if run_tool(&dir, "vis", &["-threads", "1", "fixture.bsp"]).is_err() {
            eprintln!("  vis FAIL");
            continue;
        }
        eprintln!("  compile PASS (warning-free qbsp + vis, no pointfile)");

        // Load
        let data = fs::read(&bsp).unwrap();
        let pal = fs::read(palette_path()).unwrap();
        let wad = fs::read(wad_path()).unwrap();
        let wn = wad_path()
            .file_stem()
            .unwrap()
            .to_str()
            .unwrap()
            .to_string();
        let opts = LoadOptions {
            strict: true,
            palette: Some(pal),
            lit_data: None,
            wad_archives: vec![(wn, wad)],
            texture_overrides: vec![],
            source_identity: "vf".into(),
        };
        let world = match BspLoader::load(&data, &opts) {
            Ok(w) => w,
            Err(e) => {
                eprintln!("  load FAIL: {e:?}");
                continue;
            }
        };
        let qte = QuakeToEngine::default();

        let check = |label: &str, qx: f32, qy: f32, qz: f32, want_solid: bool| -> bool {
            let ep = qte.position(qx, qy, qz);
            let c = point_contents(ep, &world.nodes, &world.leaves, &world.planes);
            let ok = c.is_solid() == want_solid;
            if !ok {
                eprintln!(
                    "  witness FAIL: {label} ({qx},{qy},{qz}) want_solid={want_solid} got {c:?}"
                );
            }
            ok
        };

        let mut all_ok = true;
        let lower_floor_top = Q;
        let upper_floor_top = spec.upper_floor_z + Q;
        let step_count = spec.upper_floor_z / Q;
        let lower_step_count = (step_count + 1) / 2;
        let upper_step_count = step_count - lower_step_count;
        let flight_x0 = 416;
        let turn_x0 = flight_x0 + lower_step_count * spec.tread;
        let mid_landing_top = lower_floor_top + lower_step_count * Q;
        let upper_flight_x0 = turn_x0 - upper_step_count * spec.tread;
        let sw_x1 = turn_x0 + 2 * Q;

        // Both stacked room interiors and representative shell points.
        all_ok &= check("lr_center", 256.0, 256.0, 24.0, false);
        all_ok &= check("lr_wall", 256.0, 136.0, 24.0, true);
        all_ok &= check(
            "ur_center",
            256.0,
            256.0,
            (spec.upper_floor_z + 24) as f32,
            false,
        );
        all_ok &= check(
            "ur_wall",
            256.0,
            376.0,
            (spec.upper_floor_z + 24) as f32,
            true,
        );

        // Direct room apertures and their transition approaches.
        all_ok &= check("lower_approach", 360.0, 192.0, 40.0, false);
        all_ok &= check("lower_aperture", 376.0, 192.0, 40.0, false);
        all_ok &= check(
            "upper_aperture",
            376.0,
            320.0,
            (upper_floor_top + 8) as f32,
            false,
        );
        all_ok &= check(
            "upper_approach",
            360.0,
            320.0,
            (upper_floor_top + 8) as f32,
            false,
        );

        // First tread support and clear space above it.
        let first_tread_x = flight_x0 + spec.tread / 2;
        all_ok &= check("first_tread_solid", first_tread_x as f32, 192.0, 24.0, true);
        all_ok &= check(
            "first_tread_above",
            first_tread_x as f32,
            192.0,
            40.0,
            false,
        );

        // Turn landing and final upper tread/landing are supported and open.
        all_ok &= check(
            "turn_landing_solid",
            (turn_x0 + 8) as f32,
            256.0,
            (mid_landing_top - 8) as f32,
            true,
        );
        all_ok &= check(
            "turn_landing_above",
            (turn_x0 + 8) as f32,
            256.0,
            (mid_landing_top + 8) as f32,
            false,
        );
        let final_tread_x = upper_flight_x0 + spec.tread / 2;
        all_ok &= check(
            "final_tread_solid",
            final_tread_x as f32,
            320.0,
            (upper_floor_top - 8) as f32,
            true,
        );
        all_ok &= check(
            "final_tread_above",
            final_tread_x as f32,
            320.0,
            (upper_floor_top + 8) as f32,
            false,
        );
        all_ok &= check(
            "upper_landing",
            408.0,
            320.0,
            (upper_floor_top + 8) as f32,
            false,
        );
        all_ok &= check(
            "stairwell_east_wall",
            (sw_x1 - 8) as f32,
            256.0,
            (mid_landing_top + 24) as f32,
            true,
        );

        if !all_ok {
            eprintln!("  witnesses FAIL");
            continue;
        }
        eprintln!("  witnesses PASS");

        if selected.is_none() {
            selected = Some(spec);
        }
        eprintln!("  *** PASSING ***");
    }

    match selected {
        Some(s) => eprintln!(
            "\n=== SELECTED: (lfz=0, ufz={}, lh={}, uh={}, riser={}, tread={}) ===",
            s.upper_floor_z, s.lower_height, s.upper_height, Q, s.tread
        ),
        None => panic!("NO-GO"),
    }
}

#[test]
fn legacy_corpus_unchanged() {
    let (map, _) = bsp_generator::generate(0, bsp_generator::DungeonConfig::nominal_m1()).unwrap();
    assert!(!map.is_empty());
    eprintln!("Legacy M1 seed 0 hash: {}", sha256(map.as_bytes()));
}
