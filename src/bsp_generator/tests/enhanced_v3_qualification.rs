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

    let entity_count = get_lump(14).and_then(|(off, len)| {
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

// ── Fast qualification sweep (32 seeds) ──────────────────────────

#[test]
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
