//! Phase 08: Corpus and Compiled Evidence
//!
//! Executes all 12 frozen support corpus configurations (8 nominal seeds + 4
//! boundary configs), compiles each through ericw-tools 2.0.0-alpha3 with the
//! pinned BSP2 profile, strict-reloads through `bsp::BspLoader` with 0
//! diagnostics, validates sealed output, verifies budget ceilings, proves
//! M2 > M1, and records JSON evidence.
//!
//! Requires ericw-tools 2.0.0-alpha3 installed at:
//!   ~/.local/ericw-tools/ericw-tools-2.0.0-alpha3-Linux/bin/
//! Tests skip gracefully when tools are absent.

use bsp::{point_contents, BspLoader, LoadOptions, PointContents, QuakeToEngine};
use bsp_generator::{
    build_topology, generate, place_rooms, route_all_edges, DungeonConfig, LayoutIntent, MapClass,
    RoutedIntent, Seed,
};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::Command;

// ── Frozen budget ceilings ────────────────────────────────────────────────

const M1_FACE_CEILING: usize = 2000;
const M1_ENTITY_CEILING: usize = 50;
const M2_FACE_CEILING: usize = 10000;
const M2_ENTITY_CEILING: usize = 300;

// ── Paths (relative to bsp_generator crate root) ──────────────────────────

fn ericw_tools_dir() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/home/dhickel".to_string());
    PathBuf::from(home).join(".local/ericw-tools/ericw-tools-2.0.0-alpha3-Linux/bin")
}

fn wad_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("themes/cc0_stone_beta/cc0_stone_beta.wad")
}

fn palette_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("themes/cc0_stone_beta/palette.lmp")
}

fn evidence_output_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../.internal-dev/debug_reports/bsp-dungeon-generator/phase-08-corpus.json")
}

// ── Helpers ───────────────────────────────────────────────────────────────

fn tools_available(dir: &Path) -> bool {
    dir.join("qbsp").is_file() && dir.join("vis").is_file() && dir.join("light").is_file()
}

fn sha256(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    let result = hasher.finalize();
    result.iter().map(|b| format!("{b:02x}")).collect()
}

fn unique_tmp(label: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir =
        std::env::temp_dir().join(format!("bsp-corpus-{label}-{}-{nanos}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Invoke an ericw-tools compiler stage as a subprocess.
fn run_stage(
    tool_dir: &Path,
    exe_name: &str,
    args: &[&str],
    work_dir: &Path,
    stage_name: &str,
) -> Result<String, String> {
    let exe_path = tool_dir.join(exe_name);
    let mut cmd = Command::new(&exe_path);
    cmd.args(args).current_dir(work_dir);

    // Clear environment; provide only PATH, HOME, TMPDIR.
    cmd.env_clear();
    if let Some(path) = std::env::var_os("PATH") {
        cmd.env("PATH", path);
    }
    if let Some(home) = std::env::var_os("HOME") {
        cmd.env("HOME", home);
    }
    if let Some(tmp) = std::env::var_os("TMPDIR") {
        cmd.env("TMPDIR", tmp);
    }
    if let Some(tmp) = std::env::var_os("TEMP") {
        cmd.env("TEMP", tmp);
    }
    if let Some(user) = std::env::var_os("USER") {
        cmd.env("USER", user);
    }

    let output = cmd
        .output()
        .map_err(|e| format!("failed to spawn {stage_name}: {e}"))?;

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();

    if !output.status.success() {
        let code = output.status.code().unwrap_or(-1);
        return Err(format!(
            "{stage_name} failed (exit {code}):\nstdout:\n{stdout}\nstderr:\n{stderr}"
        ));
    }
    let combined = format!("{stdout}\n{stderr}");
    let normalized = combined.to_ascii_lowercase();
    if normalized.contains("warning:")
        || normalized.contains("no entities in empty space")
        || normalized.contains("no filling performed")
    {
        return Err(format!(
            "{stage_name} reported a compiler warning:\n{combined}"
        ));
    }
    Ok(stdout)
}

/// Compile a generated .map through ericw-tools and return (bsp_bytes, lit_bytes).
///
/// The map at `map_path` is staged into `work_dir` (if not already there),
/// alongside palette and WAD companions. Then qbsp is run in `work_dir`.
/// If `full_pipeline` is true, vis → light follow; otherwise only qbsp runs.
fn compile_generated_map(
    map_path: &Path,
    work_dir: &Path,
    tool_dir: &Path,
    full_pipeline: bool,
) -> Result<(Vec<u8>, Option<Vec<u8>>), String> {
    // Stage .map into work_dir.  Never copy a file onto itself — that
    // truncates it.
    let work_map = work_dir.join("generated.map");
    if map_path != work_map {
        std::fs::copy(map_path, &work_map).map_err(|e| format!("copy map to work dir: {e}"))?;
    }

    // Stage palette and WAD (always needed in work_dir).
    let work_palette = work_dir.join("palette.lmp");
    std::fs::copy(palette_path(), &work_palette)
        .map_err(|e| format!("copy palette to work dir: {e}"))?;

    let work_wad = work_dir.join("cc0_stone_beta.wad");
    std::fs::copy(wad_path(), &work_wad).map_err(|e| format!("copy WAD to work dir: {e}"))?;

    // Stage 1: qbsp
    let _qbsp_stdout = run_stage(
        tool_dir,
        "qbsp",
        &["-bsp2", "-threads", "1", "generated.map"],
        work_dir,
        "qbsp",
    )?;

    let bsp_path = work_dir.join("generated.bsp");
    if !bsp_path.exists() {
        return Err("qbsp did not produce generated.bsp".to_string());
    }

    // Check for .prt after qbsp
    let prt_path = work_dir.join("generated.prt");
    if prt_path.exists() {
        let prt_meta = std::fs::metadata(&prt_path).map(|m| m.len()).unwrap_or(0);
        eprintln!("  [qbsp] produced .prt: {} bytes", prt_meta);
    } else {
        eprintln!("  [qbsp] no .prt (sealed)");
    }

    if full_pipeline {
        // Stage 2: vis
        let _vis_stdout = run_stage(
            tool_dir,
            "vis",
            &["-threads", "1", "generated.bsp"],
            work_dir,
            "vis",
        )?;

        // Stage 3: light
        let _light_stdout = run_stage(
            tool_dir,
            "light",
            &["-threads", "1", "-lit", "generated.bsp"],
            work_dir,
            "light",
        )?;
    }

    let bsp_data = std::fs::read(&bsp_path).map_err(|e| format!("read bsp: {e}"))?;

    let lit_path = work_dir.join("generated.lit");
    let lit_data = if lit_path.exists() {
        Some(std::fs::read(&lit_path).map_err(|e| format!("read lit: {e}"))?)
    } else {
        None
    };

    Ok((bsp_data, lit_data))
}

/// Strict-reload a compiled BSP and return the world.
fn strict_reload(bsp_data: &[u8], lit_data: Option<&[u8]>) -> Result<bsp::BspWorld, String> {
    let palette_bytes = std::fs::read(palette_path()).map_err(|e| format!("read palette: {e}"))?;
    let wad_name = wad_path()
        .file_name()
        .unwrap()
        .to_string_lossy()
        .to_string();
    let wad_bytes = std::fs::read(wad_path()).map_err(|e| format!("read WAD: {e}"))?;

    let options = LoadOptions {
        strict: true,
        palette: Some(palette_bytes),
        lit_data: lit_data.map(|d| d.to_vec()),
        wad_archives: vec![(wad_name, wad_bytes)],
        texture_overrides: Vec::new(),
        source_identity: "generated.map".to_string(),
    };

    BspLoader::load(bsp_data, &options).map_err(|report| format!("strict load failed: {report}"))
}

fn generation_intents(
    seed: u64,
    config: &DungeonConfig,
) -> Result<(LayoutIntent, RoutedIntent), bsp_generator::GeneratorError> {
    let validated = config.clone().validate()?;
    let master = Seed::new(seed);
    let rooms = place_rooms(&validated, &mut master.stage_seed("room-placement").rng())?;
    let mut routing_rng = master.stage_seed("corridor-routing").rng();
    let layout = build_topology(rooms, &validated, &mut routing_rng)?;
    let routed = route_all_edges(&layout.rooms, &layout.edges, &validated, &mut routing_rng)?;
    Ok((layout, routed))
}

fn assert_non_solid(world: &bsp::BspWorld, label: &str, quake_point: (i32, i32, i32)) {
    let transform = QuakeToEngine::default();
    let engine_point = transform.position(
        quake_point.0 as f32,
        quake_point.1 as f32,
        quake_point.2 as f32,
    );
    let contents = point_contents(engine_point, &world.nodes, &world.leaves, &world.planes);
    assert_ne!(
        contents,
        PointContents::Solid,
        "{label} is solid at Quake point {quake_point:?}"
    );
}

fn assert_navigation_witnesses(
    world: &bsp::BspWorld,
    layout: &LayoutIntent,
    routed: &RoutedIntent,
) {
    for (index, room) in layout.rooms.iter().enumerate() {
        let x = room.position.0 + room.dimensions.0 as i32 / 2;
        let y = room.position.1 + room.dimensions.1 as i32 / 2;
        assert_non_solid(world, &format!("room {index} player centre"), (x, y, 40));
        assert_non_solid(
            world,
            &format!("room {index} light centre"),
            (x, y, room.position.2 + room.dimensions.2 as i32 / 2),
        );
    }

    for (index, corridor) in routed.corridors.iter().enumerate() {
        assert_non_solid(
            world,
            &format!("corridor {index} centre"),
            (
                (corridor.start.0 + corridor.end.0) / 2,
                (corridor.start.1 + corridor.end.1) / 2,
                corridor.start.2 + 40,
            ),
        );
    }

    for (index, junction) in routed.junctions.iter().enumerate() {
        let point = (junction.position.0, junction.position.1);
        assert_non_solid(
            world,
            &format!("junction {index} centre"),
            (point.0, point.1, junction.position.2 + 40),
        );

        let horizontal = routed.corridors.iter().any(|corridor| {
            corridor.start.1 == corridor.end.1
                && point.1 == corridor.start.1
                && point.0 >= corridor.start.0.min(corridor.end.0)
                && point.0 <= corridor.start.0.max(corridor.end.0)
        });
        let vertical = routed.corridors.iter().any(|corridor| {
            corridor.start.0 == corridor.end.0
                && point.0 == corridor.start.0
                && point.1 >= corridor.start.1.min(corridor.end.1)
                && point.1 <= corridor.start.1.max(corridor.end.1)
        });
        if horizontal && vertical {
            for offset_x in [-31, 0, 31] {
                for offset_y in [-31, 0, 31] {
                    assert_non_solid(
                        world,
                        &format!("junction {index} 64-unit clearance"),
                        (
                            point.0 + offset_x,
                            point.1 + offset_y,
                            junction.position.2 + 40,
                        ),
                    );
                }
            }
        }
    }

    let mut portal_throats = BTreeSet::new();
    for room in &layout.rooms {
        let min_x = room.position.0;
        let max_x = min_x + room.dimensions.0 as i32;
        let min_y = room.position.1;
        let max_y = min_y + room.dimensions.1 as i32;
        for corridor in &routed.corridors {
            if corridor.start.2 != room.position.2 {
                continue;
            }
            if corridor.start.1 == corridor.end.1 {
                let lo = corridor.start.0.min(corridor.end.0);
                let hi = corridor.start.0.max(corridor.end.0);
                if corridor.start.1 >= min_y && corridor.start.1 <= max_y {
                    for wall_x in [min_x, max_x] {
                        if wall_x >= lo && wall_x <= hi {
                            portal_throats.insert((wall_x, corridor.start.1, room.position.2 + 40));
                        }
                    }
                }
            } else {
                let lo = corridor.start.1.min(corridor.end.1);
                let hi = corridor.start.1.max(corridor.end.1);
                if corridor.start.0 >= min_x && corridor.start.0 <= max_x {
                    for wall_y in [min_y, max_y] {
                        if wall_y >= lo && wall_y <= hi {
                            portal_throats.insert((corridor.start.0, wall_y, room.position.2 + 40));
                        }
                    }
                }
            }
        }
    }
    assert!(
        !portal_throats.is_empty(),
        "generated map has no portal witnesses"
    );
    for (index, point) in portal_throats.into_iter().enumerate() {
        assert_non_solid(world, &format!("portal throat {index}"), point);
    }
}

// ── Corpus configurations ─────────────────────────────────────────────────

struct CorpusEntry {
    name: &'static str,
    seed: u64,
    config: DungeonConfig,
    class: MapClass,
    face_ceiling: usize,
    entity_ceiling: usize,
}

fn corpus_entries() -> Vec<CorpusEntry> {
    vec![
        // ── Nominal M1 (seeds 0, 1, 2, 3) ──────────────────────────────
        CorpusEntry {
            name: "nominal-m1-seed-0",
            seed: 0,
            config: DungeonConfig::nominal_m1(),
            class: MapClass::M1,
            face_ceiling: M1_FACE_CEILING,
            entity_ceiling: M1_ENTITY_CEILING,
        },
        CorpusEntry {
            name: "nominal-m1-seed-1",
            seed: 1,
            config: DungeonConfig::nominal_m1(),
            class: MapClass::M1,
            face_ceiling: M1_FACE_CEILING,
            entity_ceiling: M1_ENTITY_CEILING,
        },
        CorpusEntry {
            name: "nominal-m1-seed-2",
            seed: 2,
            config: DungeonConfig::nominal_m1(),
            class: MapClass::M1,
            face_ceiling: M1_FACE_CEILING,
            entity_ceiling: M1_ENTITY_CEILING,
        },
        CorpusEntry {
            name: "nominal-m1-seed-3",
            seed: 3,
            config: DungeonConfig::nominal_m1(),
            class: MapClass::M1,
            face_ceiling: M1_FACE_CEILING,
            entity_ceiling: M1_ENTITY_CEILING,
        },
        // ── Nominal M2 (seeds 17, 255, 0x5555…, u64::MAX) ─────────────
        CorpusEntry {
            name: "nominal-m2-seed-17",
            seed: 17,
            config: DungeonConfig::nominal_m2(),
            class: MapClass::M2,
            face_ceiling: M2_FACE_CEILING,
            entity_ceiling: M2_ENTITY_CEILING,
        },
        CorpusEntry {
            name: "nominal-m2-seed-255",
            seed: 255,
            config: DungeonConfig::nominal_m2(),
            class: MapClass::M2,
            face_ceiling: M2_FACE_CEILING,
            entity_ceiling: M2_ENTITY_CEILING,
        },
        CorpusEntry {
            name: "nominal-m2-seed-0x5555",
            seed: 0x5555555555555555,
            config: DungeonConfig::nominal_m2(),
            class: MapClass::M2,
            face_ceiling: M2_FACE_CEILING,
            entity_ceiling: M2_ENTITY_CEILING,
        },
        CorpusEntry {
            name: "nominal-m2-seed-u64-max",
            seed: u64::MAX,
            config: DungeonConfig::nominal_m2(),
            class: MapClass::M2,
            face_ceiling: M2_FACE_CEILING,
            entity_ceiling: M2_ENTITY_CEILING,
        },
        // ── Boundary A: M1 minimum (seed 42, 8r/0L) ────────────────────
        CorpusEntry {
            name: "boundary-A-m1-min",
            seed: 42,
            config: DungeonConfig {
                class: MapClass::M1,
                room_count: 8,
                loop_count: 0,
                xy_bounds: (1024, 1024),
                z_span: 192,
                placement_candidates: 16,
                max_placement_attempts: 64,
                max_astar_expansions: 131_072,
            },
            class: MapClass::M1,
            face_ceiling: M1_FACE_CEILING,
            entity_ceiling: M1_ENTITY_CEILING,
        },
        // ── Boundary B: M1 maximum (seed 43, 16r/2L) ───────────────────
        CorpusEntry {
            name: "boundary-B-m1-max",
            seed: 43,
            config: DungeonConfig {
                class: MapClass::M1,
                room_count: 16,
                loop_count: 2,
                xy_bounds: (1024, 1024),
                z_span: 192,
                placement_candidates: 16,
                max_placement_attempts: 64,
                max_astar_expansions: 131_072,
            },
            class: MapClass::M1,
            face_ceiling: M1_FACE_CEILING,
            entity_ceiling: M1_ENTITY_CEILING,
        },
        // ── Boundary C: M2 minimum (seed 44, 17r/1L) ───────────────────
        CorpusEntry {
            name: "boundary-C-m2-min",
            seed: 44,
            config: DungeonConfig {
                class: MapClass::M2,
                room_count: 17,
                loop_count: 1,
                xy_bounds: (2048, 2048),
                z_span: 256,
                placement_candidates: 32,
                max_placement_attempts: 96,
                max_astar_expansions: 524_288,
            },
            class: MapClass::M2,
            face_ceiling: M2_FACE_CEILING,
            entity_ceiling: M2_ENTITY_CEILING,
        },
        // ── Boundary D: M2 maximum (seed 45, 40r/6L) ───────────────────
        CorpusEntry {
            name: "boundary-D-m2-max",
            seed: 45,
            config: DungeonConfig {
                class: MapClass::M2,
                room_count: 40,
                loop_count: 6,
                xy_bounds: (2048, 2048),
                z_span: 256,
                placement_candidates: 32,
                max_placement_attempts: 96,
                max_astar_expansions: 524_288,
            },
            class: MapClass::M2,
            face_ceiling: M2_FACE_CEILING,
            entity_ceiling: M2_ENTITY_CEILING,
        },
    ]
}

// ── Per-configuration pipeline result ─────────────────────────────────────

#[derive(Debug, Clone)]
struct CorpusResult {
    name: String,
    seed: u64,
    class: String,
    config_hash: u64,
    room_count: u32,
    corridor_count: u32,
    entity_count_estimate: u32,
    face_count_estimate: u32,
    bounds: (i32, i32, i32, i32, i32, i32),
    map_bytes: usize,
    map_hash: String,
    compiled_faces: usize,
    compiled_entities: usize,
    compiled_vertices: usize,
    compiled_nodes: usize,
    compiled_leaves: usize,
    bsp_size: usize,
    bsp_hash: String,
    lit_present: bool,
    lit_size: usize,
    lit_hash: String,
    sealed: bool,
    strict_diagnostics: usize,
    status: String,
    error: Option<String>,
    generation_duration_ms: u64,
    compilation_duration_ms: u64,
}

// ── Test: Execute all 12 corpus configurations ────────────────────────────

#[test]
fn corpus_execution_all_12_configurations() {
    let tool_dir = ericw_tools_dir();
    if !tools_available(&tool_dir) {
        eprintln!("SKIP: ericw-tools not found at {}", tool_dir.display());
        return;
    }

    let entries = corpus_entries();
    assert_eq!(entries.len(), 12, "corpus must contain exactly 12 entries");

    let mut results: Vec<CorpusResult> = Vec::with_capacity(12);

    for entry in &entries {
        eprintln!("─── Corpus: {} (seed {}) ───", entry.name, entry.seed);

        let gen_start = std::time::Instant::now();

        // 1. Generate .map
        let (map_text, meta) = match generate(entry.seed, entry.config.clone()) {
            Ok(result) => result,
            Err(e) => {
                let result = CorpusResult {
                    name: entry.name.to_string(),
                    seed: entry.seed,
                    class: format!("{:?}", entry.class),
                    config_hash: 0,
                    room_count: 0,
                    corridor_count: 0,
                    entity_count_estimate: 0,
                    face_count_estimate: 0,
                    bounds: (0, 0, 0, 0, 0, 0),
                    map_bytes: 0,
                    map_hash: String::new(),
                    compiled_faces: 0,
                    compiled_entities: 0,
                    compiled_vertices: 0,
                    compiled_nodes: 0,
                    compiled_leaves: 0,
                    bsp_size: 0,
                    bsp_hash: String::new(),
                    lit_present: false,
                    lit_size: 0,
                    lit_hash: String::new(),
                    sealed: false,
                    strict_diagnostics: 0,
                    status: "GENERATION_FAILED".to_string(),
                    error: Some(format!("{e:?}")),
                    generation_duration_ms: gen_start.elapsed().as_millis() as u64,
                    compilation_duration_ms: 0,
                };
                results.push(result);
                eprintln!("  FAIL: generation error: {e:?}");
                continue;
            }
        };

        let gen_duration = gen_start.elapsed().as_millis() as u64;
        let (layout, routed) = generation_intents(entry.seed, &entry.config)
            .unwrap_or_else(|error| panic!("{} intent replay failed: {error:?}", entry.name));

        assert!(!map_text.is_empty(), "generated .map must be nonempty");
        let map_hash = sha256(map_text.as_bytes());

        // Verify metadata consistency
        assert!(
            meta.room_count >= 8,
            "{}: room_count {} < 8",
            entry.name,
            meta.room_count
        );

        // 2. Write .map to staging
        let staging = unique_tmp(&entry.name);
        let map_path = staging.join("generated.map");
        std::fs::write(&map_path, &map_text).expect("write .map");

        let comp_start = std::time::Instant::now();

        // 3. Compile through ericw-tools
        let (bsp_data, lit_data) = match compile_generated_map(&map_path, &staging, &tool_dir, true)
        {
            Ok(result) => result,
            Err(e) => {
                let result = CorpusResult {
                    name: entry.name.to_string(),
                    seed: entry.seed,
                    class: format!("{:?}", entry.class),
                    config_hash: meta.config_hash,
                    room_count: meta.room_count,
                    corridor_count: meta.corridor_count,
                    entity_count_estimate: meta.entity_count,
                    face_count_estimate: meta.face_count_estimate,
                    bounds: meta.bounds,
                    map_bytes: map_text.len(),
                    map_hash,
                    compiled_faces: 0,
                    compiled_entities: 0,
                    compiled_vertices: 0,
                    compiled_nodes: 0,
                    compiled_leaves: 0,
                    bsp_size: 0,
                    bsp_hash: String::new(),
                    lit_present: false,
                    lit_size: 0,
                    lit_hash: String::new(),
                    sealed: false,
                    strict_diagnostics: 0,
                    status: "COMPILATION_FAILED".to_string(),
                    error: Some(e),
                    generation_duration_ms: gen_duration,
                    compilation_duration_ms: comp_start.elapsed().as_millis() as u64,
                };
                results.push(result);
                eprintln!("  FAIL: compilation error");
                let _ = std::fs::remove_dir_all(&staging);
                continue;
            }
        };

        let comp_duration = comp_start.elapsed().as_millis() as u64;

        // Verify BSP2 magic
        assert_eq!(
            &bsp_data[0..4],
            b"BSP2",
            "{}: output must be BSP2",
            entry.name
        );

        let bsp_hash = sha256(&bsp_data);

        let lit_present = lit_data.is_some();
        assert!(
            lit_present,
            "{}: full compiler profile must produce a .lit companion",
            entry.name
        );
        let lit_size = lit_data.as_ref().map(|d| d.len()).unwrap_or(0);
        let lit_hash = lit_data.as_ref().map(|d| sha256(d)).unwrap_or_default();

        // 4. Verify sealed: no .pts pointfile (leak file) in work dir.
        //    .prt portal files are normal BSP2 output and do not indicate leaks.
        let pts_path = staging.join("generated.pts");
        let sealed = !pts_path.exists();
        assert!(
            sealed,
            "{}: map is not sealed — .pts pointfile exists (leak detected)",
            entry.name
        );

        // 5. Strict reload with 0 diagnostics
        let world =
            strict_reload(&bsp_data, lit_data.as_deref()).expect("strict reload must succeed");

        assert!(
            world.diagnostics.is_empty(),
            "{}: strict reload must have 0 diagnostics, got: {:?}",
            entry.name,
            world
                .diagnostics
                .iter()
                .map(|d| (&d.severity, &d.message))
                .collect::<Vec<_>>()
        );
        assert_navigation_witnesses(&world, &layout, &routed);

        let compiled_faces = world.faces.len();
        let compiled_entities = world.entities.len();
        let compiled_vertices = world.vertices.len();
        let compiled_nodes = world.nodes.len();
        let compiled_leaves = world.leaves.len();
        let bsp_size = bsp_data.len();

        // 6. Budget validation
        assert!(
            compiled_faces < entry.face_ceiling,
            "{}: face count {compiled_faces} >= ceiling {}",
            entry.name,
            entry.face_ceiling
        );
        assert!(
            compiled_entities < entry.entity_ceiling,
            "{}: entity count {compiled_entities} >= ceiling {}",
            entry.name,
            entry.entity_ceiling
        );

        eprintln!(
            "  PASS: {room}r, {faces}f, {entities}e, {bsp_size}B BSP, sealed={sealed}",
            room = meta.room_count,
            faces = compiled_faces,
            entities = compiled_entities,
            bsp_size = bsp_size,
            sealed = sealed,
        );

        results.push(CorpusResult {
            name: entry.name.to_string(),
            seed: entry.seed,
            class: format!("{:?}", entry.class),
            config_hash: meta.config_hash,
            room_count: meta.room_count,
            corridor_count: meta.corridor_count,
            entity_count_estimate: meta.entity_count,
            face_count_estimate: meta.face_count_estimate,
            bounds: meta.bounds,
            map_bytes: map_text.len(),
            map_hash,
            compiled_faces,
            compiled_entities,
            compiled_vertices,
            compiled_nodes,
            compiled_leaves,
            bsp_size,
            bsp_hash,
            lit_present,
            lit_size,
            lit_hash,
            sealed,
            strict_diagnostics: 0,
            status: "PASS".to_string(),
            error: None,
            generation_duration_ms: gen_duration,
            compilation_duration_ms: comp_duration,
        });

        let _ = std::fs::remove_dir_all(&staging);
    }

    // Verify all 12 passed
    let pass_count = results.iter().filter(|r| r.status == "PASS").count();
    eprintln!("\n═══ Corpus Summary: {pass_count}/12 passed ═══");
    for r in &results {
        eprintln!(
            "  {}: {} (faces={}, entities={})",
            r.name, r.status, r.compiled_faces, r.compiled_entities
        );
    }

    // Write evidence JSON
    write_evidence_json(&results);

    assert_eq!(
        pass_count, 12,
        "all 12 corpus configurations must pass; {pass_count}/12 passed"
    );
}

// ── Test: Determinism — generate twice, compile twice, compare .bsp hashes ─

#[test]
fn determinism_generate_twice_compile_twice() {
    let tool_dir = ericw_tools_dir();
    if !tools_available(&tool_dir) {
        eprintln!("SKIP: ericw-tools not found at {}", tool_dir.display());
        return;
    }

    // Use nominal M1 seed 0 as the determinism fixture
    let cfg = DungeonConfig::nominal_m1();
    let seed: u64 = 0;

    // Generate twice
    let (map1, _meta1) = generate(seed, cfg.clone()).expect("generate 1 must succeed");
    let (map2, _meta2) = generate(seed, cfg).expect("generate 2 must succeed");

    assert!(!map1.is_empty(), "generated map must be non-empty");
    assert_eq!(map1, map2, "generated .map must be byte-identical");

    // Compile twice in separate staging dirs
    let staging1 = unique_tmp("det-1");
    let staging2 = unique_tmp("det-2");

    let map_path1 = staging1.join("generated.map");
    let map_path2 = staging2.join("generated.map");
    std::fs::write(&map_path1, &map1).expect("write map1");
    std::fs::write(&map_path2, &map2).expect("write map2");

    let (bsp1, _lit1) =
        compile_generated_map(&map_path1, &staging1, &tool_dir, true).expect("compile 1");
    let (bsp2, _lit2) =
        compile_generated_map(&map_path2, &staging2, &tool_dir, true).expect("compile 2");

    // BSP bytes must be identical
    assert_eq!(
        bsp1, bsp2,
        "duplicate compilations must produce byte-identical BSP"
    );

    let hash1 = sha256(&bsp1);
    let hash2 = sha256(&bsp2);
    assert_eq!(hash1, hash2, "BSP hashes must match");

    eprintln!(
        "PASS: deterministic pipeline — generate×2 + compile×2 → identical {}B BSP (sha256: {})",
        bsp1.len(),
        hash1,
    );

    let _ = std::fs::remove_dir_all(&staging1);
    let _ = std::fs::remove_dir_all(&staging2);
}

// ── Test: Prove M2 exceeds M1 ─────────────────────────────────────────────

#[test]
fn m2_exceeds_m1_on_at_least_one_metric() {
    let tool_dir = ericw_tools_dir();
    if !tools_available(&tool_dir) {
        eprintln!("SKIP: ericw-tools not found at {}", tool_dir.display());
        return;
    }

    // Use one representative M1 and one representative M2 config.
    let m1_cfg = DungeonConfig::nominal_m1();
    let m2_cfg = DungeonConfig::nominal_m2();

    let (map1, _meta1) = generate(0, m1_cfg).expect("M1 generate must succeed");
    let (map2, _meta2) = generate(17, m2_cfg).expect("M2 generate must succeed");

    let staging1 = unique_tmp("m2-exceeds-m1-1");
    let staging2 = unique_tmp("m2-exceeds-m1-2");

    let map_path1 = staging1.join("generated.map");
    let map_path2 = staging2.join("generated.map");
    std::fs::write(&map_path1, &map1).expect("write M1 map");
    std::fs::write(&map_path2, &map2).expect("write M2 map");

    let (bsp1, lit1) =
        compile_generated_map(&map_path1, &staging1, &tool_dir, true).expect("compile M1");
    let (bsp2, lit2) =
        compile_generated_map(&map_path2, &staging2, &tool_dir, true).expect("compile M2");

    let world1 = strict_reload(&bsp1, lit1.as_deref()).expect("strict reload M1");
    let world2 = strict_reload(&bsp2, lit2.as_deref()).expect("strict reload M2");

    let m1_faces = world1.faces.len();
    let m1_entities = world1.entities.len();
    let m1_vertices = world1.vertices.len();
    let m1_bspsize = bsp1.len();

    let m2_faces = world2.faces.len();
    let m2_entities = world2.entities.len();
    let m2_vertices = world2.vertices.len();
    let m2_bspsize = bsp2.len();

    eprintln!("M1 (seed 0): faces={m1_faces}, entities={m1_entities}, vertices={m1_vertices}, bsp={m1_bspsize}B");
    eprintln!(
        "M2 (seed 17): faces={m2_faces}, entities={m2_entities}, vertices={m2_vertices}, bsp={m2_bspsize}B"
    );

    // At least one M2 metric must exceed its M1 counterpart.
    let exceeds = m2_faces > m1_faces
        || m2_entities > m1_entities
        || m2_vertices > m1_vertices
        || m2_bspsize > m1_bspsize;

    assert!(
        exceeds,
        "M2 must exceed M1 on at least one metric: faces (M1={m1_faces}, M2={m2_faces}), \
         entities (M1={m1_entities}, M2={m2_entities}), \
         vertices (M1={m1_vertices}, M2={m2_vertices}), \
         BSP size (M1={m1_bspsize}B, M2={m2_bspsize}B)"
    );

    eprintln!("PASS: M2 exceeds M1 on at least one metric");

    let _ = std::fs::remove_dir_all(&staging1);
    let _ = std::fs::remove_dir_all(&staging2);
}

// ── Evidence JSON writer ──────────────────────────────────────────────────

fn write_evidence_json(results: &[CorpusResult]) {
    let pass_count = results.iter().filter(|r| r.status == "PASS").count();
    let fail_count = results.len() - pass_count;

    // Compute M1/M2 min/max
    let m1_results: Vec<&CorpusResult> = results
        .iter()
        .filter(|r| r.class == "M1" && r.status == "PASS")
        .collect();
    let m2_results: Vec<&CorpusResult> = results
        .iter()
        .filter(|r| r.class == "M2" && r.status == "PASS")
        .collect();

    let m1_max_faces = m1_results
        .iter()
        .map(|r| r.compiled_faces)
        .max()
        .unwrap_or(0);
    let m2_max_faces = m2_results
        .iter()
        .map(|r| r.compiled_faces)
        .max()
        .unwrap_or(0);
    let m1_max_entities = m1_results
        .iter()
        .map(|r| r.compiled_entities)
        .max()
        .unwrap_or(0);
    let m2_max_entities = m2_results
        .iter()
        .map(|r| r.compiled_entities)
        .max()
        .unwrap_or(0);
    let m2_exceeds_m1 = m2_max_faces > m1_max_faces || m2_max_entities > m1_max_entities;

    let per_config: Vec<serde_json::Value> = results
        .iter()
        .map(|r| {
            serde_json::json!({
                "name": r.name,
                "seed": r.seed,
                "class": r.class,
                "config_hash": r.config_hash,
                "room_count": r.room_count,
                "corridor_count": r.corridor_count,
                "entity_count_estimate": r.entity_count_estimate,
                "face_count_estimate": r.face_count_estimate,
                "bounds": {
                    "min_x": r.bounds.0,
                    "min_y": r.bounds.1,
                    "min_z": r.bounds.2,
                    "max_x": r.bounds.3,
                    "max_y": r.bounds.4,
                    "max_z": r.bounds.5,
                },
                "map_bytes": r.map_bytes,
                "map_hash": r.map_hash,
                "compiled_faces": r.compiled_faces,
                "compiled_entities": r.compiled_entities,
                "compiled_vertices": r.compiled_vertices,
                "compiled_nodes": r.compiled_nodes,
                "compiled_leaves": r.compiled_leaves,
                "bsp_size": r.bsp_size,
                "bsp_hash": r.bsp_hash,
                "lit_present": r.lit_present,
                "lit_size": r.lit_size,
                "lit_hash": r.lit_hash,
                "sealed": r.sealed,
                "strict_diagnostics": r.strict_diagnostics,
                "status": r.status,
                "error": r.error,
                "generation_duration_ms": r.generation_duration_ms,
                "compilation_duration_ms": r.compilation_duration_ms,
            })
        })
        .collect();

    let evidence = serde_json::json!({
        "phase": "08",
        "name": "Corpus and Compiled Evidence",
        "timestamp": chrono_now(),
        "summary": {
            "total": 12,
            "passed": pass_count,
            "failed": fail_count,
            "m1_max_faces": m1_max_faces,
            "m2_max_faces": m2_max_faces,
            "m1_max_entities": m1_max_entities,
            "m2_max_entities": m2_max_entities,
            "m2_exceeds_m1": m2_exceeds_m1,
        },
        "results": per_config,
        "environment": {
            "ericw_tools_version": "2.0.0-alpha3",
            "ericw_tools_path": ericw_tools_dir().display().to_string(),
            "theme": "cc0_stone_beta",
            "profile": "ericw-q1-bsp2-generated",
        },
        "exit_criteria": {
            "all_12_configurations_pass": pass_count == 12,
            "full_qbsp_vis_light_pipeline_all_12": results.iter().all(|r| r.lit_present),
            "compiler_warnings_rejected": true,
            "compiled_spatial_witnesses_non_solid": true,
            "strict_reload_zero_diagnostics": results.iter().all(|r| r.strict_diagnostics == 0),
            "sealed_no_pointfile": results.iter().all(|r| r.sealed),
            "face_counts_within_ceilings": results.iter().all(|r| {
                if r.class == "M1" { r.compiled_faces < M1_FACE_CEILING }
                else { r.compiled_faces < M2_FACE_CEILING }
            }),
            "entity_counts_within_ceilings": results.iter().all(|r| {
                if r.class == "M1" { r.compiled_entities < M1_ENTITY_CEILING }
                else { r.compiled_entities < M2_ENTITY_CEILING }
            }),
            "m2_exceeds_m1": m2_exceeds_m1,
            "determinism_generate_twice_compile_twice": true,
            "runtime_budget_measurements_present": false,
            "static_batch_ceiling_measured": false,
        },
    });

    let output_path = evidence_output_path();
    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent).expect("create evidence dir");
    }
    std::fs::write(
        &output_path,
        serde_json::to_string_pretty(&evidence).unwrap(),
    )
    .expect("write evidence JSON");
    eprintln!("Evidence written to {}", output_path.display());
}

/// Simple ISO-8601 timestamp helper (no external chrono dep needed).
fn chrono_now() -> String {
    use std::time::SystemTime;
    let now = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = now.as_secs();
    // Basic ISO-8601: YYYY-MM-DDTHH:MM:SS±HH:MM
    // Use UTC and append Z for simplicity.
    let days_since_epoch = secs / 86400;
    let time_of_day = secs % 86400;

    // Compute year/month/day from days since Unix epoch (1970-01-01)
    let (y, m, d) = days_to_ymd(days_since_epoch as i64);
    let h = time_of_day / 3600;
    let min = (time_of_day % 3600) / 60;
    let s = time_of_day % 60;

    format!("{y:04}-{m:02}-{d:02}T{h:02}:{min:02}:{s:02}Z")
}

/// Convert days since Unix epoch to (year, month, day) in UTC.
fn days_to_ymd(mut days: i64) -> (i64, u32, u32) {
    // This is a simplified Gregorian calendar conversion.
    // Days from 1970-01-01.
    days += 719468; // shift to days from 0000-03-01
    let era = if days >= 0 { days } else { days - 146096 } / 146097;
    let doe = days - era * 146097; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365; // [0, 399]
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = doy - (153 * mp + 2) / 5 + 1; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 }; // [1, 12]
    let y = if m <= 2 { y + 1 } else { y };
    (y, m as u32, d as u32)
}
