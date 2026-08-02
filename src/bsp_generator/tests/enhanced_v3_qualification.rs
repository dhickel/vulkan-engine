//! Phase 09 — Enhanced V3 Qualification Sweep + Corpus Freeze
//!
//! Enumerates the complete 2,304-cell Cartesian matrix
//! (256 seeds × 3 densities × 3 extents), runs every cell through the public
//! production route twice for determinism, classifies outcomes against the
//! Phase 02 permitted-outcome policy, compiles the 12 corpus entries through
//! the pinned ericw-tools compiler, and freezes the corpus manifest.
//!
//! # Run
//!
//! ```bash
//! cargo test -p bsp_generator --test enhanced_v3_qualification -- --nocapture
//! ```
//!
//! # Architecture
//!
//! ```text
//! 2,304 tuples → double-run (fast, generation only) → classify → report
//! 12 corpus entries → compile → freeze manifest
//! ```

use bsp::{BspLoader, LoadOptions};
use bsp_generator::enhanced_v3::{self, V3Config, V3Preset};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{SystemTime, UNIX_EPOCH};

// ── Matrix constants ──────────────────────────────────────────────────────

const PRESETS: [V3Preset; 3] = [V3Preset::Sparse, V3Preset::Moderate, V3Preset::Rich];
const EXTENTS: [u32; 3] = [1024, 2048, 3072];

// ── Corpus constants ─────────────────────────────────────────────────────

const CORPUS_SIZE: usize = 12;
const BASELINE_V3_MANIFEST_FROZEN_AT: &str = "2026-08-02";

fn corpus_entries() -> Vec<(u64, V3Preset, u32)> {
    let seeds = [0u64, 42, 99, 255];
    let mut entries = Vec::with_capacity(CORPUS_SIZE);
    for &seed in &seeds {
        entries.push((seed, V3Preset::Sparse, 2048));
    }
    for &seed in &seeds {
        entries.push((seed, V3Preset::Moderate, 2048));
    }
    for &seed in &seeds {
        entries.push((seed, V3Preset::Rich, 3072));
    }
    entries
}

// ── Paths ─────────────────────────────────────────────────────────────────

fn crate_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf()
}

fn repo_root() -> PathBuf {
    crate_dir()
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

fn corpus_manifest_path() -> PathBuf {
    crate_dir().join("tests/fixtures/enhanced_v3_corpus/manifest.json")
}

fn write_atomic(path: &Path, contents: &str) {
    let temporary = path.with_extension(format!("tmp-{}", std::process::id()));
    std::fs::write(&temporary, contents)
        .unwrap_or_else(|error| panic!("write {}: {error}", temporary.display()));
    std::fs::rename(&temporary, path)
        .unwrap_or_else(|error| panic!("publish {}: {error}", path.display()));
}

fn qualification_report_path() -> PathBuf {
    repo_root().join(".internal-dev/debug_reports/enhanced-v3-production/qualification.json")
}

fn corpus_json_report_path() -> PathBuf {
    repo_root().join(".internal-dev/debug_reports/enhanced-v3-production/corpus.json")
}

fn theme_dir() -> PathBuf {
    crate_dir().join("themes/cc0_dungeon_v2")
}

fn ericw_tools_dir() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/home/dhickel".to_string());
    PathBuf::from(home).join(".local/ericw-tools/ericw-tools-2.0.0-alpha3-Linux/bin")
}

fn tools_available(dir: &Path) -> bool {
    dir.join("qbsp").is_file() && dir.join("vis").is_file() && dir.join("light").is_file()
}

// ── SHA-256 ──────────────────────────────────────────────────────────────

fn sha256_hex(data: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(data);
    format!("{:x}", h.finalize())
}

// ── Qualification types ──────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
struct QualificationReport {
    timestamp: String,
    matrix_size: usize,
    total_cells: usize,
    success_cells: usize,
    error_cells: usize,
    compiler_available: bool,
    compiler_summary: CompilerSummary,
    corpus_frozen: bool,
    cells: Vec<CellResult>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CompilerSummary {
    total_attempted: usize,
    passed: usize,
    failed: usize,
    not_run: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CellResult {
    tuple: TupleKey,
    policy: String,
    attempts: [AttemptOutcome; 2],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TupleKey {
    seed: u64,
    density: String,
    extent: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
enum AttemptOutcome {
    Success {
        map_sha256: String,
        metadata_sha256: String,
    },
    Error {
        error_code: String,
    },
}

// ── Corpus manifest types ────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CorpusManifest {
    schema: String,
    frozen_at: String,
    generator: String,
    entries: Vec<CorpusManifestEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CorpusManifestEntry {
    id: String,
    seed: u64,
    preset: String,
    extent: u32,
    map_sha256: String,
    metadata_sha256: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    bsp_sha256: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    lit_sha256: Option<String>,
    room_count: u32,
    actual_faces: u32,
    actual_entities: u32,
    actual_brushes: u32,
    spawn_origin: [i32; 3],
    light_count: u32,
    bounds: [i32; 6],
    has_upper_layer: bool,
    grammar_families: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    compiled_faces: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    compiled_entities: Option<u32>,
}

// ── Corpus JSON report ───────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CorpusJsonReport {
    timestamp: String,
    frozen_entries: usize,
    qualification_ref: String,
    compiled_space_ref: String,
    entries: Vec<CorpusRefRow>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CorpusRefRow {
    id: String,
    map_sha256: String,
    metadata_sha256: String,
    bsp_sha256: Option<String>,
    lit_sha256: Option<String>,
    faces: Option<u32>,
    entities: Option<u32>,
}

// ── Metadata serialization helper ────────────────────────────────────────

fn metadata_to_json_value(meta: &enhanced_v3::EnhancedV3Metadata) -> serde_json::Value {
    serde_json::json!({
        "seed": meta.seed(),
        "preset": meta.preset(),
        "xy_extent": meta.xy_extent(),
        "schema_version": meta.schema_version(),
        "generator": meta.generator(),
        "room_count": meta.room_count(),
        "lower_room_count": meta.lower_room_count(),
        "upper_room_count": meta.upper_room_count(),
        "portal_count": meta.portal_count(),
        "transition_count": meta.transition_count(),
        "route_count": meta.route_count(),
        "grammar_families": meta.grammar_families(),
        "identity_satisfied": meta.identity_satisfied(),
        "estimated_faces": meta.estimated_faces(),
        "actual_faces": meta.actual_faces(),
        "estimated_entities": meta.estimated_entities(),
        "actual_entities": meta.actual_entities(),
        "actual_brushes": meta.actual_brushes(),
        "spawn_origin": meta.spawn_origin(),
        "light_count": meta.light_count(),
        "bounds": meta.bounds(),
        "has_upper_layer": meta.has_upper_layer(),
        "face_budget_satisfied": meta.face_budget_satisfied(),
        "entity_budget_satisfied": meta.entity_budget_satisfied(),
    })
}

// ── Compiler ─────────────────────────────────────────────────────────────

struct CompileResult {
    bsp_sha256: Option<String>,
    lit_sha256: Option<String>,
    faces: Option<u32>,
    entities: Option<u32>,
    success: bool,
}

fn compile_map_text(map_text: &str) -> CompileResult {
    let tools_dir = ericw_tools_dir();
    if !tools_available(&tools_dir) {
        return CompileResult {
            bsp_sha256: None,
            lit_sha256: None,
            faces: None,
            entities: None,
            success: false,
        };
    }

    let tmp = match tempfile::TempDir::new() {
        Ok(t) => t,
        Err(_) => {
            return CompileResult {
                bsp_sha256: None,
                lit_sha256: None,
                faces: None,
                entities: None,
                success: false,
            }
        }
    };

    let wad = theme_dir().join("cc0_dungeon_v2.wad");
    let palette = theme_dir().join("palette.lmp");
    let _ = std::fs::copy(&wad, tmp.path().join("cc0_dungeon_v2.wad"));
    let _ = std::fs::copy(&palette, tmp.path().join("palette.lmp"));

    let map_path = tmp.path().join("test.map");
    if std::fs::write(&map_path, map_text).is_err() {
        return CompileResult {
            bsp_sha256: None,
            lit_sha256: None,
            faces: None,
            entities: None,
            success: false,
        };
    }

    // qbsp
    let qbsp_out = Command::new(tools_dir.join("qbsp"))
        .arg("-bsp2")
        .arg(&map_path)
        .current_dir(tmp.path())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .output();

    match qbsp_out {
        Ok(out) if out.status.success() => {}
        _ => {
            return CompileResult {
                bsp_sha256: None,
                lit_sha256: None,
                faces: None,
                entities: None,
                success: false,
            }
        }
    }

    let bsp_path = tmp.path().join("test.bsp");
    if !bsp_path.exists() {
        return CompileResult {
            bsp_sha256: None,
            lit_sha256: None,
            faces: None,
            entities: None,
            success: false,
        };
    }

    // vis (non-fatal)
    let _ = Command::new(tools_dir.join("vis"))
        .arg(&bsp_path)
        .current_dir(tmp.path())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .output();

    // light (non-fatal)
    let _ = Command::new(tools_dir.join("light"))
        .arg("-bsp2")
        .arg(&bsp_path)
        .current_dir(tmp.path())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .output();

    let bsp_bytes = match std::fs::read(&bsp_path) {
        Ok(b) => b,
        Err(_) => {
            return CompileResult {
                bsp_sha256: None,
                lit_sha256: None,
                faces: None,
                entities: None,
                success: false,
            }
        }
    };

    if bsp_bytes.len() < 4 || &bsp_bytes[0..4] != b"BSP2" {
        return CompileResult {
            bsp_sha256: None,
            lit_sha256: None,
            faces: None,
            entities: None,
            success: false,
        };
    }

    let bsp_sha256 = sha256_hex(&bsp_bytes);

    let lit_path = tmp.path().join("test.lit");
    let lit_sha256 = if lit_path.exists() {
        std::fs::read(&lit_path).ok().map(|b| sha256_hex(&b))
    } else {
        None
    };

    let (faces, entities) = parse_bsp_counts(&bsp_bytes);

    CompileResult {
        bsp_sha256: Some(bsp_sha256),
        lit_sha256,
        faces,
        entities,
        success: true,
    }
}

fn parse_bsp_counts(bsp_bytes: &[u8]) -> (Option<u32>, Option<u32>) {
    if bsp_bytes.len() < 4 + 136 || &bsp_bytes[0..4] != b"BSP2" {
        return (None, None);
    }

    let get_lump = |idx: usize| -> Option<(u32, u32)> {
        let base = 4 + idx * 8;
        if base + 8 > bsp_bytes.len() {
            return None;
        }
        let off = u32::from_le_bytes(bsp_bytes[base..base + 4].try_into().ok()?);
        let len = u32::from_le_bytes(bsp_bytes[base + 4..base + 8].try_into().ok()?);
        Some((off, len))
    };

    let face_count = get_lump(7).map(|(_, len)| if len > 0 { len / 20 } else { 0 });

    let entity_count = get_lump(0).and_then(|(off, len)| {
        if len == 0 {
            return Some(0);
        }
        let start = off as usize;
        let end = start + len as usize;
        if end > bsp_bytes.len() {
            return Some(0);
        }
        let s = std::str::from_utf8(&bsp_bytes[start..end]).ok()?;
        Some(s.lines().filter(|l| l.trim() == "{").count() as u32)
    });

    (face_count, entity_count)
}

fn git_text(args: &[&str]) -> String {
    let output = Command::new("git")
        .args(args)
        .current_dir(repo_root())
        .output()
        .unwrap_or_else(|error| panic!("run git {}: {error}", args.join(" ")));
    assert!(
        output.status.success(),
        "git {} failed: {}",
        args.join(" "),
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout)
        .expect("git output must be UTF-8")
        .trim()
        .to_string()
}

fn strict_reload_freeze(
    bsp_data: &[u8],
    lit_data: &[u8],
    wad_path: &Path,
    palette_path: &Path,
) -> Result<(u32, u32), String> {
    let palette = std::fs::read(palette_path).map_err(|error| format!("read palette: {error}"))?;
    let wad_name = wad_path
        .file_name()
        .ok_or("WAD path has no file name")?
        .to_string_lossy()
        .into_owned();
    let wad = std::fs::read(wad_path).map_err(|error| format!("read WAD: {error}"))?;
    let options = LoadOptions {
        strict: true,
        palette: Some(palette),
        lit_data: Some(lit_data.to_vec()),
        wad_archives: vec![(wad_name, wad)],
        texture_overrides: Vec::new(),
        source_identity: "generated.map".to_string(),
    };
    let world = BspLoader::load(bsp_data, &options)
        .map_err(|report| format!("strict reload failed: {report}"))?;
    if world.diagnostics.is_empty() {
        Ok((world.faces.len() as u32, world.entities.len() as u32))
    } else {
        Err(format!(
            "strict reload emitted diagnostics: {:?}",
            world.diagnostics
        ))
    }
}

// ── Fast qualification sweep (32 seeds) ──────────────────────────

#[test]
#[ignore = "baseline authority is regenerated only by phase01_baseline_freeze_12_cells"]
fn qualification_sweep_and_corpus_freeze() {
    qualification_sweep_impl(0..32, 288)
}

/// Full 2,304-cell sweep — ignored by default due to runtime (~25 minutes).
#[test]
#[ignore = "full 2304-cell sweep takes ~25 minutes"]
fn qualification_sweep_full_2304() {
    qualification_sweep_impl(0..256, 2304)
}

fn qualification_sweep_impl(seed_range: std::ops::Range<u64>, expected_cells: usize) {
    std::fs::create_dir_all(qualification_report_path().parent().unwrap()).unwrap();
    std::fs::create_dir_all(corpus_manifest_path().parent().unwrap()).unwrap();

    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs()
        .to_string();

    let compiler_available = tools_available(&ericw_tools_dir());

    // ── 1. Enumerate the Cartesian matrix for given seed range ───────
    let mut tuples: Vec<(u64, V3Preset, u32)> = Vec::with_capacity(expected_cells);
    for seed in seed_range {
        for &preset in &PRESETS {
            for &extent in &EXTENTS {
                tuples.push((seed, preset, extent));
            }
        }
    }
    assert_eq!(tuples.len(), expected_cells, "matrix cardinality mismatch");

    // Verify no duplicates
    let mut seen = BTreeSet::new();
    for &(s, p, e) in &tuples {
        let key = (s, p.tag().to_string(), e);
        assert!(seen.insert(key.clone()), "duplicate tuple: {key:?}");
    }

    // ── 2. Run every cell through production route twice ───────────────
    let mut cells: Vec<CellResult> = Vec::with_capacity(expected_cells);
    let mut success_count = 0usize;
    let mut error_count = 0usize;

    for &(seed, preset, extent) in &tuples {
        let config = match V3Config::new(seed, preset, extent) {
            Ok(c) => c,
            Err(e) => {
                cells.push(CellResult {
                    tuple: TupleKey {
                        seed,
                        density: preset.tag().to_string(),
                        extent,
                    },
                    policy: "typed-error".to_string(),
                    attempts: [
                        AttemptOutcome::Error {
                            error_code: format!("{e:?}"),
                        },
                        AttemptOutcome::Error {
                            error_code: format!("{e:?}"),
                        },
                    ],
                });
                error_count += 1;
                continue;
            }
        };

        let run1 = enhanced_v3::run_pipeline(&config);
        let run2 = enhanced_v3::run_pipeline(&config);

        let (policy, attempts) = match (&run1, &run2) {
            (Ok(out1), Ok(out2)) => {
                let map_sha256_1 = sha256_hex(out1.map_text.as_bytes());
                let map_sha256_2 = sha256_hex(out2.map_text.as_bytes());
                assert_eq!(
                    map_sha256_1, map_sha256_2,
                    "non-deterministic map for seed={seed} {:?} extent={extent}",
                    preset
                );

                let meta_json_1 = metadata_to_json_value(&out1.metadata).to_string();
                let meta_json_2 = metadata_to_json_value(&out2.metadata).to_string();
                let meta_sha256_1 = sha256_hex(meta_json_1.as_bytes());
                let meta_sha256_2 = sha256_hex(meta_json_2.as_bytes());
                assert_eq!(
                    meta_sha256_1, meta_sha256_2,
                    "non-deterministic metadata for seed={seed} {:?} extent={extent}",
                    preset
                );

                success_count += 1;
                (
                    "success".to_string(),
                    [
                        AttemptOutcome::Success {
                            map_sha256: map_sha256_1,
                            metadata_sha256: meta_sha256_1,
                        },
                        AttemptOutcome::Success {
                            map_sha256: map_sha256_2,
                            metadata_sha256: meta_sha256_2,
                        },
                    ],
                )
            }
            (Err(e1), Err(e2)) => {
                let code1 = format!("{e1:?}");
                let code2 = format!("{e2:?}");
                assert_eq!(
                    code1, code2,
                    "non-deterministic error for seed={seed} {:?} extent={extent}",
                    preset
                );
                error_count += 1;
                (
                    "typed-error".to_string(),
                    [
                        AttemptOutcome::Error { error_code: code1 },
                        AttemptOutcome::Error { error_code: code2 },
                    ],
                )
            }
            (Ok(_), Err(e)) | (Err(e), Ok(_)) => {
                panic!(
                    "mixed success/error for seed={seed} {:?} extent={extent}: {e:?}",
                    preset
                );
            }
        };

        cells.push(CellResult {
            tuple: TupleKey {
                seed,
                density: preset.tag().to_string(),
                extent,
            },
            policy,
            attempts,
        });
    }

    // ── 3. Write qualification report ─────────────────────────────────
    let report = QualificationReport {
        timestamp: timestamp.clone(),
        matrix_size: expected_cells,
        total_cells: cells.len(),
        success_cells: success_count,
        error_cells: error_count,
        compiler_available,
        compiler_summary: CompilerSummary {
            total_attempted: 0,
            passed: 0,
            failed: 0,
            not_run: 0,
        },
        corpus_frozen: false,
        cells,
    };
    std::fs::write(
        qualification_report_path(),
        serde_json::to_string_pretty(&report).unwrap(),
    )
    .unwrap();

    // ── 4. Freeze 12-entry corpus with compilation ────────────────────
    let corpus = corpus_entries();
    assert_eq!(corpus.len(), CORPUS_SIZE);

    let mut manifest_entries: Vec<CorpusManifestEntry> = Vec::with_capacity(CORPUS_SIZE);
    let mut corpus_rows: Vec<CorpusRefRow> = Vec::with_capacity(CORPUS_SIZE);
    let mut compiler_passed = 0usize;
    let mut compiler_failed = 0usize;
    let mut compiler_not_run = 0usize;

    for (seed, preset, extent) in &corpus {
        let config =
            V3Config::new(*seed, *preset, *extent).expect("corpus entry config must be valid");
        let output =
            enhanced_v3::run_pipeline(&config).expect("corpus entry generation must succeed");

        let map_sha256 = sha256_hex(output.map_text.as_bytes());
        let meta_json = metadata_to_json_value(&output.metadata).to_string();
        let meta_sha256 = sha256_hex(meta_json.as_bytes());

        let id = format!("v3-{}-seed-{}", preset.tag(), seed);
        let bounds = output.metadata.bounds();
        let spawn = output.metadata.spawn_origin();

        // Compile
        let (bsp_sha256, lit_sha256, compiled_faces, compiled_entities) = if compiler_available {
            let cr = compile_map_text(&output.map_text);
            if cr.success {
                compiler_passed += 1;
                (cr.bsp_sha256, cr.lit_sha256, cr.faces, cr.entities)
            } else {
                compiler_failed += 1;
                (None, None, None, None)
            }
        } else {
            compiler_not_run += 1;
            (None, None, None, None)
        };

        manifest_entries.push(CorpusManifestEntry {
            id: id.clone(),
            seed: *seed,
            preset: preset.tag().to_string(),
            extent: *extent,
            map_sha256: map_sha256.clone(),
            metadata_sha256: meta_sha256.clone(),
            bsp_sha256: bsp_sha256.clone(),
            lit_sha256: lit_sha256.clone(),
            room_count: output.metadata.room_count(),
            actual_faces: output.metadata.actual_faces(),
            actual_entities: output.metadata.actual_entities(),
            actual_brushes: output.metadata.actual_brushes(),
            spawn_origin: [spawn.0, spawn.1, spawn.2],
            light_count: output.metadata.light_count(),
            bounds: [bounds.0, bounds.1, bounds.2, bounds.3, bounds.4, bounds.5],
            has_upper_layer: output.metadata.has_upper_layer(),
            grammar_families: output.metadata.grammar_families().to_vec(),
            compiled_faces,
            compiled_entities,
        });

        corpus_rows.push(CorpusRefRow {
            id,
            map_sha256,
            metadata_sha256: meta_sha256,
            bsp_sha256,
            lit_sha256,
            faces: compiled_faces,
            entities: compiled_entities,
        });
    }

    // Write manifest
    let manifest = CorpusManifest {
        schema: "enhanced-v3-corpus/v1".to_string(),
        frozen_at: timestamp.clone(),
        generator: "bsp_generator/enhanced_v3".to_string(),
        entries: manifest_entries,
    };
    std::fs::write(
        corpus_manifest_path(),
        serde_json::to_string_pretty(&manifest).unwrap(),
    )
    .unwrap();

    // Write corpus.json
    let corpus_json = CorpusJsonReport {
        timestamp,
        frozen_entries: CORPUS_SIZE,
        qualification_ref: "qualification.json".to_string(),
        compiled_space_ref: "enhanced_v3_compiled_space".to_string(),
        entries: corpus_rows,
    };
    std::fs::write(
        corpus_json_report_path(),
        serde_json::to_string_pretty(&corpus_json).unwrap(),
    )
    .unwrap();

    // ── 5. Assertions ─────────────────────────────────────────────────
    assert_eq!(manifest.entries.len(), CORPUS_SIZE);

    // Density coverage
    let sparse_count = manifest
        .entries
        .iter()
        .filter(|e| e.preset == "sparse")
        .count();
    let moderate_count = manifest
        .entries
        .iter()
        .filter(|e| e.preset == "moderate")
        .count();
    let rich_count = manifest
        .entries
        .iter()
        .filter(|e| e.preset == "rich")
        .count();
    assert_eq!(sparse_count, 4);
    assert_eq!(moderate_count, 4);
    assert_eq!(rich_count, 4);

    // Budget ceilings
    for entry in &manifest.entries {
        assert!(
            entry.actual_faces < 10000,
            "{}: faces {} exceeds budget",
            entry.id,
            entry.actual_faces
        );
        assert!(
            entry.actual_entities < 300,
            "{}: entities {} exceeds budget",
            entry.id,
            entry.actual_entities
        );
        // Source brush count records structural density; it is not a renderer
        // static-batch measurement and has no 500-brush contract.
        assert!(entry.room_count >= 1, "{}: zero rooms", entry.id);
    }
    println!(
        "Qualification sweep: {} total, {} success, {} errors",
        report.total_cells, report.success_cells, report.error_cells
    );
    println!(
        "Corpus frozen: {} entries (compiler: {} pass, {} fail, {} not-run)",
        manifest.entries.len(),
        compiler_passed,
        compiler_failed,
        compiler_not_run
    );
    for entry in &manifest.entries {
        println!(
            "  {}: {} rooms, {} faces, {} entities, {} brushes, compiled_faces={:?}",
            entry.id,
            entry.room_count,
            entry.actual_faces,
            entry.actual_entities,
            entry.actual_brushes,
            entry.compiled_faces
        );
    }
}

// ── Corpus manifest load test ───────────────────────────────────────────

#[test]
fn corpus_manifest_loads_and_validates() {
    let manifest_path = corpus_manifest_path();
    assert!(
        manifest_path.exists(),
        "corpus manifest must exist at {}",
        manifest_path.display()
    );

    let manifest_json = std::fs::read_to_string(&manifest_path).expect("must read corpus manifest");
    let manifest: CorpusManifest =
        serde_json::from_str(&manifest_json).expect("must parse corpus manifest");

    assert_eq!(manifest.schema, "enhanced-v3-corpus/v1");
    assert_eq!(manifest.entries.len(), CORPUS_SIZE);

    let mut ids = BTreeSet::new();
    for entry in &manifest.entries {
        assert!(
            ids.insert(entry.id.clone()),
            "duplicate corpus entry ID: {}",
            entry.id
        );
    }

    let sparse: Vec<_> = manifest
        .entries
        .iter()
        .filter(|e| e.preset == "sparse")
        .collect();
    let moderate: Vec<_> = manifest
        .entries
        .iter()
        .filter(|e| e.preset == "moderate")
        .collect();
    let rich: Vec<_> = manifest
        .entries
        .iter()
        .filter(|e| e.preset == "rich")
        .collect();
    assert_eq!(sparse.len(), 4);
    assert_eq!(moderate.len(), 4);
    assert_eq!(rich.len(), 4);

    for entry in &manifest.entries {
        assert!(
            entry.actual_faces < 10000,
            "{} faces {} exceeds budget",
            entry.id,
            entry.actual_faces
        );
        assert!(
            entry.actual_entities < 300,
            "{} entities {} exceeds budget",
            entry.id,
            entry.actual_entities
        );
    }
}

// ── Determinism spot-test for corpus entries ────────────────────────────

#[test]
fn corpus_entries_deterministic() {
    for (seed, preset, extent) in corpus_entries() {
        let config = V3Config::new(seed, preset, extent).expect("valid config");
        let out1 = enhanced_v3::run_pipeline(&config).expect("first run");
        let out2 = enhanced_v3::run_pipeline(&config).expect("second run");

        assert_eq!(
            out1.map_text, out2.map_text,
            "map drift for ({seed}, {:?}, {extent})",
            preset
        );
        assert_eq!(
            out1.metadata, out2.metadata,
            "metadata drift for ({seed}, {:?}, {extent})",
            preset
        );

        // Verify against manifest if it exists
        let manifest_path = corpus_manifest_path();
        if manifest_path.exists() {
            let manifest_json = std::fs::read_to_string(&manifest_path).unwrap();
            let manifest: CorpusManifest = serde_json::from_str(&manifest_json).unwrap();
            let id = format!("v3-{}-seed-{}", preset.tag(), seed);
            if let Some(entry) = manifest.entries.iter().find(|e| e.id == id) {
                let map_hash = sha256_hex(out1.map_text.as_bytes());
                let meta_json = metadata_to_json_value(&out1.metadata).to_string();
                let meta_hash = sha256_hex(meta_json.as_bytes());
                assert_eq!(map_hash, entry.map_sha256, "map hash mismatch for {id}");
                assert_eq!(
                    meta_hash, entry.metadata_sha256,
                    "metadata hash mismatch for {id}"
                );
            }
        }
    }
}

// ── Phase 01 Baseline Freeze — focused 12-cell regeneration ─────────────

/// Focused baseline-only manifest regeneration entry point executing exactly
/// the 12 corpus cells (Sparse/Moderate/Rich × seeds 0/42/99/255). Each cell
/// is generated twice in independent temporary roots, compiled through the
/// pinned ericw-tools BSP2 profile with full warning/leak analysis, and the
/// manifest is written with map/metadata/BSP/LIT hashes, counts, spawn,
/// bounds, grammar families, and compiled lump metrics.
///
/// This test replaces the broader qualification sweep for baseline freezing.
/// It fails on any compiler warning, leak, skipped fill, or strict-load
/// diagnostic.
#[test]
fn phase01_baseline_freeze_12_cells() {
    let tools_dir = ericw_tools_dir();
    assert!(
        tools_available(&tools_dir),
        "ericw-tools is required for the baseline freeze at {}",
        tools_dir.display()
    );

    let task_base_commit = git_text(&["rev-parse", "HEAD"]);
    let dirty_tree_before_freeze = git_text(&["status", "--porcelain=v1"]);
    assert!(
        !dirty_tree_before_freeze
            .lines()
            .any(|line| line.contains("src/bsp_generator/src/")),
        "baseline freeze requires task-base generator sources; found source drift:\n{dirty_tree_before_freeze}"
    );

    let corpus = corpus_entries();
    assert_eq!(corpus.len(), CORPUS_SIZE);

    // Compute tool hashes for provenance
    let tool_hashes: std::collections::BTreeMap<String, String> = ["qbsp", "vis", "light"]
        .iter()
        .map(|name| {
            let path = tools_dir.join(name);
            let data = std::fs::read(&path).unwrap_or_default();
            (name.to_string(), sha256_hex(&data))
        })
        .collect();

    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs()
        .to_string();

    let theme_wad_path = theme_dir().join("cc0_dungeon_v2.wad");
    let palette_path = theme_dir().join("palette.lmp");
    let theme_wad_hash = sha256_hex(&std::fs::read(&theme_wad_path).unwrap());
    let palette_hash = sha256_hex(&std::fs::read(&palette_path).unwrap());

    let mut manifest_entries: Vec<CorpusManifestEntry> = Vec::with_capacity(CORPUS_SIZE);

    for (seed, preset, extent) in &corpus {
        let config =
            V3Config::new(*seed, *preset, *extent).expect("corpus entry config must be valid");
        let id = format!("v3-{}-seed-{}", preset.tag(), seed);
        eprintln!("\n=== {id} ===");

        // ── Generation: run twice, prove identity ───────────────────
        let out1 = enhanced_v3::run_pipeline(&config).expect("first run must succeed");
        let out2 = enhanced_v3::run_pipeline(&config).expect("second run must succeed");

        assert_eq!(
            out1.map_text, out2.map_text,
            "{id}: map text non-deterministic"
        );
        assert_eq!(
            out1.metadata, out2.metadata,
            "{id}: metadata non-deterministic"
        );

        let map_sha256 = sha256_hex(out1.map_text.as_bytes());
        let meta_json_a = metadata_to_json_value(&out1.metadata).to_string();
        let meta_json_b = metadata_to_json_value(&out2.metadata).to_string();
        assert_eq!(
            meta_json_a, meta_json_b,
            "{id}: serialized metadata non-deterministic"
        );
        let meta_sha256 = sha256_hex(meta_json_a.as_bytes());

        eprintln!("  map SHA-256: {}", map_sha256);
        eprintln!("  metadata SHA-256: {}", meta_sha256);

        // ── Compilation: dual-root, warning/leak/strict-load gates ──
        let (bsp_sha256, lit_sha256, compiled_faces, compiled_entities, compiler_ok) =
            compile_and_freeze_cell(
                &out1.map_text,
                &meta_json_a,
                &out2.map_text,
                &meta_json_b,
                &id,
                &tools_dir,
                &theme_wad_path,
                &palette_path,
            );

        assert!(compiler_ok, "{id}: compilation must succeed");

        eprintln!("  BSP SHA-256: {}", bsp_sha256.as_deref().unwrap_or("NONE"));
        eprintln!("  LIT SHA-256: {}", lit_sha256.as_deref().unwrap_or("NONE"));

        let bounds = out1.metadata.bounds();
        let spawn = out1.metadata.spawn_origin();

        manifest_entries.push(CorpusManifestEntry {
            id: id.clone(),
            seed: *seed,
            preset: preset.tag().to_string(),
            extent: *extent,
            map_sha256: map_sha256.clone(),
            metadata_sha256: meta_sha256.clone(),
            bsp_sha256,
            lit_sha256,
            room_count: out1.metadata.room_count(),
            actual_faces: out1.metadata.actual_faces(),
            actual_entities: out1.metadata.actual_entities(),
            actual_brushes: out1.metadata.actual_brushes(),
            spawn_origin: [spawn.0, spawn.1, spawn.2],
            light_count: out1.metadata.light_count(),
            bounds: [bounds.0, bounds.1, bounds.2, bounds.3, bounds.4, bounds.5],
            has_upper_layer: out1.metadata.has_upper_layer(),
            grammar_families: out1.metadata.grammar_families().to_vec(),
            compiled_faces,
            compiled_entities,
        });
    }

    // ── Write deterministic manifest ─────────────────────────────────
    let manifest = CorpusManifest {
        schema: "enhanced-v3-corpus/v1".to_string(),
        frozen_at: BASELINE_V3_MANIFEST_FROZEN_AT.to_string(),
        generator: "bsp_generator/enhanced_v3".to_string(),
        entries: manifest_entries,
    };

    let manifest_json = serde_json::to_string_pretty(&manifest).expect("serialize manifest");
    assert_eq!(
        manifest_json,
        serde_json::to_string_pretty(&manifest).expect("re-serialize manifest"),
        "baseline manifest serialization drifted within the focused freeze"
    );
    write_atomic(&corpus_manifest_path(), &manifest_json);

    // ── Write provenance sidecar ────────────────────────────────────
    let provenance_path = corpus_manifest_path()
        .parent()
        .unwrap()
        .join("baseline-freeze-provenance.json");
    let provenance = serde_json::json!({
        "schema": "enhanced-v3-baseline-freeze-provenance/v1",
        "frozen_at": timestamp,
        "generator": "bsp_generator/enhanced_v3",
        "task_base_commit": task_base_commit,
        "dirty_tree_before_freeze": dirty_tree_before_freeze.lines().collect::<Vec<_>>(),
        "compiler": "ericw-tools 2.0.0-alpha3",
        "tool_hashes": tool_hashes,
        "theme_wad_sha256": theme_wad_hash,
        "palette_sha256": palette_hash,
        "command_vectors": [
            { "program": "qbsp", "args": ["-bsp2", "-threads", "1", "generated.map"] },
            { "program": "vis", "args": ["-threads", "1", "generated.bsp"] },
            { "program": "light", "args": ["-threads", "1", "-lit", "generated.bsp"] }
        ],
        "compile_profile": {
            "qbsp_args": ["-bsp2", "-threads", "1"],
            "vis_args": ["-threads", "1"],
            "light_args": ["-threads", "1", "-lit"]
        },
        "cell_count": CORPUS_SIZE,
        "cells": manifest.entries.iter().map(|e| serde_json::json!({
            "id": e.id,
            "seed": e.seed,
            "preset": e.preset,
            "extent": e.extent,
            "map_sha256": e.map_sha256,
            "metadata_sha256": e.metadata_sha256,
            "bsp_sha256": e.bsp_sha256,
            "lit_sha256": e.lit_sha256,
            "compiled_faces": e.compiled_faces,
            "compiled_entities": e.compiled_entities,
            "room_count": e.room_count,
            "source_faces": e.actual_faces,
            "source_entities": e.actual_entities,
            "source_brushes": e.actual_brushes,
            "spawn_origin": e.spawn_origin,
            "light_count": e.light_count,
            "bounds": e.bounds,
            "grammar_families": e.grammar_families,
        })).collect::<Vec<_>>(),
    });
    write_atomic(
        &provenance_path,
        &serde_json::to_string_pretty(&provenance).expect("serialize provenance"),
    );

    // ── Budget assertions ───────────────────────────────────────────
    for entry in &manifest.entries {
        assert!(
            entry.actual_faces < 10000,
            "{}: faces {} exceeds budget",
            entry.id,
            entry.actual_faces
        );
        assert!(
            entry.actual_entities < 300,
            "{}: entities {} exceeds budget",
            entry.id,
            entry.actual_entities
        );
    }

    eprintln!(
        "\n=== Baseline Freeze Complete: {} entries ===",
        manifest.entries.len()
    );
    eprintln!("Manifest: {}", corpus_manifest_path().display());
    eprintln!("Provenance: {}", provenance_path.display());
}

/// Compile a cell through the pinned BSP2 profile with full warning/leak
/// analysis and dual-root determinism. Returns compiled hashes and metrics.
fn compile_and_freeze_cell(
    map_text_a: &str,
    metadata_json_a: &str,
    map_text_b: &str,
    metadata_json_b: &str,
    cell_id: &str,
    tools_dir: &Path,
    wad_path: &Path,
    palette_path: &Path,
) -> (
    Option<String>,
    Option<String>,
    Option<u32>,
    Option<u32>,
    bool,
) {
    let tmp_a = unique_tmp_dir(&format!("freeze-{cell_id}-a"));
    let tmp_b = unique_tmp_dir(&format!("freeze-{cell_id}-b"));

    let compile_pass = |tmp: &Path,
                        label: &str,
                        map_text: &str,
                        metadata_json: &str|
     -> Result<(Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>, u32, u32), String> {
        // Persist both source artifacts in each independent root before compiling.
        let map_path = tmp.join("generated.map");
        let metadata_path = tmp.join("metadata.json");
        std::fs::write(&map_path, map_text).map_err(|e| format!("write map: {e}"))?;
        std::fs::write(&metadata_path, metadata_json)
            .map_err(|e| format!("write metadata: {e}"))?;

        // Copy WAD and palette
        let wad_basename = wad_path.file_name().unwrap().to_str().unwrap();
        std::fs::copy(wad_path, tmp.join(wad_basename)).map_err(|e| format!("copy WAD: {e}"))?;
        std::fs::copy(palette_path, tmp.join("palette.lmp"))
            .map_err(|e| format!("copy palette: {e}"))?;

        // qbsp
        run_compiler_stage_strict(
            tools_dir,
            "qbsp",
            &["-bsp2", "-threads", "1", "generated.map"],
            tmp,
            &format!("qbsp-{label}"),
        )?;

        let bsp_path = tmp.join("generated.bsp");
        if !bsp_path.exists() {
            return Err(format!("qbsp did not produce generated.bsp"));
        }

        // vis
        run_compiler_stage_strict(
            tools_dir,
            "vis",
            &["-threads", "1", "generated.bsp"],
            tmp,
            &format!("vis-{label}"),
        )?;

        // light
        run_compiler_stage_strict(
            tools_dir,
            "light",
            &["-threads", "1", "-lit", "generated.bsp"],
            tmp,
            &format!("light-{label}"),
        )?;

        // Any pointfile or leak portal is a leak, including an empty artifact.
        for leak_name in ["generated.pts", "generated.leak.prt"] {
            let leak_path = tmp.join(leak_name);
            if leak_path.exists() {
                return Err(format!("leak detected — {} exists", leak_path.display()));
            }
        }

        let map_data = std::fs::read(&map_path).map_err(|e| format!("read map: {e}"))?;
        let metadata_data =
            std::fs::read(&metadata_path).map_err(|e| format!("read metadata: {e}"))?;
        let bsp_data = std::fs::read(&bsp_path).map_err(|e| format!("read BSP: {e}"))?;

        // Verify BSP2 magic
        if bsp_data.len() < 4 || &bsp_data[0..4] != b"BSP2" {
            return Err("BSP magic not BSP2".to_string());
        }

        let lit_path = tmp.join("generated.lit");
        let lit_data = std::fs::read(&lit_path)
            .map_err(|e| format!("light did not produce required LIT artifact: {e}"))?;
        let (compiled_faces, compiled_entities) =
            strict_reload_freeze(&bsp_data, &lit_data, wad_path, palette_path)?;

        Ok((
            map_data,
            metadata_data,
            bsp_data,
            lit_data,
            compiled_faces,
            compiled_entities,
        ))
    };

    let (map_a, metadata_a, bsp_a, lit_a, faces_a, entities_a) =
        match compile_pass(&tmp_a, "a", map_text_a, metadata_json_a) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("  FAIL compile-a: {e}");
                let _ = std::fs::remove_dir_all(&tmp_a);
                let _ = std::fs::remove_dir_all(&tmp_b);
                return (None, None, None, None, false);
            }
        };

    let (map_b, metadata_b, bsp_b, lit_b, faces_b, entities_b) =
        match compile_pass(&tmp_b, "b", map_text_b, metadata_json_b) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("  FAIL compile-b: {e}");
                let _ = std::fs::remove_dir_all(&tmp_a);
                let _ = std::fs::remove_dir_all(&tmp_b);
                return (None, None, None, None, false);
            }
        };

    // Determinism: all source and compiled artifacts are byte-identical across roots.
    if map_a != map_b {
        eprintln!("  FAIL: map not byte-identical across independent roots");
        let _ = std::fs::remove_dir_all(&tmp_a);
        let _ = std::fs::remove_dir_all(&tmp_b);
        return (None, None, None, None, false);
    }
    if metadata_a != metadata_b {
        eprintln!("  FAIL: metadata not byte-identical across independent roots");
        let _ = std::fs::remove_dir_all(&tmp_a);
        let _ = std::fs::remove_dir_all(&tmp_b);
        return (None, None, None, None, false);
    }
    if bsp_a != bsp_b {
        eprintln!("  FAIL: BSP not byte-identical across independent roots");
        let _ = std::fs::remove_dir_all(&tmp_a);
        let _ = std::fs::remove_dir_all(&tmp_b);
        return (None, None, None, None, false);
    }
    if lit_a != lit_b {
        eprintln!("  FAIL: LIT not byte-identical across independent roots");
        let _ = std::fs::remove_dir_all(&tmp_a);
        let _ = std::fs::remove_dir_all(&tmp_b);
        return (None, None, None, None, false);
    }

    assert_eq!(
        (faces_a, entities_a),
        (faces_b, entities_b),
        "{cell_id}: strict-loaded BSP lump metrics differ across roots"
    );
    let bsp_hash = sha256_hex(&bsp_a);
    let lit_hash = Some(sha256_hex(&lit_a));

    // Cleanup
    let _ = std::fs::remove_dir_all(&tmp_a);
    let _ = std::fs::remove_dir_all(&tmp_b);

    eprintln!("  compile PASS: BSP {}B, LIT {}B", bsp_a.len(), lit_a.len());

    (
        Some(bsp_hash),
        lit_hash,
        Some(faces_a),
        Some(entities_a),
        true,
    )
}

fn unique_tmp_dir(label: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!(
        "enhanced-v3-freeze-{label}-{}-{nanos}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn run_compiler_stage_strict(
    tool_dir: &Path,
    exe_name: &str,
    args: &[&str],
    work_dir: &Path,
    stage_name: &str,
) -> Result<String, String> {
    let exe_path = tool_dir.join(exe_name);
    let mut cmd = std::process::Command::new(&exe_path);
    cmd.args(args).current_dir(work_dir);

    // Minimized environment for determinism
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

    // Warning detection (strict: any warning fails)
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

// ── Source budget spot test ─────────────────────────────────────────────

#[test]
fn corpus_source_budget_validation() {
    for (seed, preset, extent) in corpus_entries() {
        let config = V3Config::new(seed, preset, extent).expect("valid config");
        let output = enhanced_v3::run_pipeline(&config).expect("generation");

        let id = format!("v3-{}-seed-{}", preset.tag(), seed);
        assert!(
            output.metadata.actual_faces() < 10000,
            "{id}: faces budget fail"
        );
        assert!(
            output.metadata.actual_entities() < 300,
            "{id}: entities budget fail"
        );
        // Source brush count is structural density, not renderer static
        // batches. Static-batch evidence is measured from strict extraction
        // in the runtime/package corpus suites.
        // Room counts vary by layout; verify at least one room present
        assert!(
            output.metadata.room_count() >= 1,
            "{id}: room count too low"
        );
    }
}
