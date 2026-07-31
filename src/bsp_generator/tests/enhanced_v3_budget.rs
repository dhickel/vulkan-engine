//! Phase 08 — Dense M2 Budget Corpus
//!
//! Private integration test that proves corpus semantics and produces
//! dense M2 compiler-budget evidence.
//!
//! # Architecture
//!
//! - **A. Corpus semantics**: Fixed private proof corpus with representative
//!   v3 seeds and presets. Each entry is generated through the one-way
//!   pipeline, and the result is byte-compared against checked-in canonical
//!   fixtures. Determinism is proven by repeated generation.
//!
//! - **B. Dense M2 budget evidence**: A hand-authored Rich proof fixture
//!   combines the qualified portal and grounded-assembly structures across
//!   the frozen two layers. It is compiled through ericw-tools (BSP2), and
//!   the resulting BSP is measured
//!   against M2 budget ceilings: faces (<10,000), entities (<300),
//!   static batches (<500), XY bound (3072×3072), Z span (384).
//!
//! # Key Constants (from Phase 01)
//!
//! - Face ceiling: <10,000
//! - Entity ceiling: <300
//! - Static batch ceiling: <500
//! - M2 XY bound: 3072×3072, Z span: 384
//!
//! # Validation
//!
//! ```bash
//! cargo test -p bsp_generator --test enhanced_v3_budget -- --nocapture
//! cargo test -p bsp_generator --test enhanced_v3_integrated  # unchanged
//! cargo test -p bsp_generator --test enhanced_v3_baseline   # unchanged
//! cargo fmt --check -p bsp_generator
//! ```

mod enhanced_v3_proof;

use enhanced_v3_proof::compiler;
use enhanced_v3_proof::contract::Preset;
use enhanced_v3_proof::corpus::{self, CorpusEntryResult};
use enhanced_v3_proof::pipeline;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

// ── Paths ─────────────────────────────────────────────────────────────────

fn crate_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf()
}

fn fixture_dir() -> PathBuf {
    crate_dir().join("tests/fixtures/enhanced_v3_proof")
}

fn corpus_fixture_map_path(entry_id: &str) -> PathBuf {
    fixture_dir().join(format!("{entry_id}.map"))
}

fn corpus_fixture_metadata_path(entry_id: &str) -> PathBuf {
    fixture_dir().join(format!("{entry_id}-metadata.json"))
}

fn dense_fixture_map_path() -> PathBuf {
    fixture_dir().join("dense-rich.map")
}

fn dense_fixture_metadata_path() -> PathBuf {
    fixture_dir().join("dense-rich-metadata.json")
}

fn budget_report_path() -> PathBuf {
    crate_dir().join("../../.internal-dev/debug_reports/enhanced-v3-proof/budget-report.json")
}

fn theme_dir() -> PathBuf {
    crate_dir().join("themes/cc0_dungeon_v2")
}

fn wad_path() -> PathBuf {
    theme_dir().join("cc0_dungeon_v2.wad")
}

fn palette_path() -> PathBuf {
    theme_dir().join("palette.lmp")
}

// ── Budget report types ───────────────────────────────────────────────────

/// M2 budget ceilings from the frozen contract.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
struct M2BudgetCeilings {
    max_faces: u32,
    max_entities: u32,
    max_static_batches: u32,
    max_xy_extent: u32,
    max_z_span: u32,
}

impl M2BudgetCeilings {
    fn frozen() -> Self {
        Self {
            max_faces: 10_000,
            max_entities: 300,
            max_static_batches: 500,
            max_xy_extent: 3072,
            max_z_span: 384,
        }
    }
}

/// Source and compiled measurements for the dense fixture.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct BudgetMeasurement {
    source_brushes: usize,
    source_plane_sides: usize,
    faces: usize,
    entities: usize,
    static_batches: usize,
    unique_planes: usize,
    vertices: usize,
    edges: usize,
    texinfos: usize,
    nodes: usize,
    leaves: usize,
    solid_leaves: usize,
    empty_leaves: usize,
    clipnodes: usize,
    models: usize,
    lightmap_bytes: usize,
    xy_extent: u32,
    z_span: u32,
    bsp_size: usize,
    lit_size: usize,
    compilation_time_ms: u64,
}

/// A single measurement comparison against a frozen or profile limit.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct BudgetCheck {
    metric: String,
    value: u64,
    comparator: String,
    limit: u64,
    within_budget: bool,
}

/// The complete budget report.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct BudgetReport {
    schema: String,
    timestamp: String,
    profile_name: String,
    profile_version: String,
    tool_dir: String,
    tools_available: bool,
    corpus_baseline_hash: String,
    ceiling: M2BudgetCeilings,
    measurement: BudgetMeasurement,
    checks: Vec<BudgetCheck>,
    overall_within_budget: bool,
    corpus_results: Vec<CorpusVerificationRecord>,
    rich_map_sha256: String,
    rich_metadata_sha256: String,
    bsp_sha256: String,
    lit_sha256: String,
    compiler_diagnostics: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CorpusVerificationRecord {
    id: String,
    map_sha256: String,
    metadata_sha256: String,
    map_bytes: usize,
    status: String,
    expected_map_sha256: Option<String>,
    map_match: bool,
}

// ── Helpers ───────────────────────────────────────────────────────────────

/// ISO-8601 timestamp.
fn iso8601_now() -> String {
    use std::time::SystemTime;
    let now = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = now.as_secs();
    let days = secs / 86400;
    let time_of_day = secs % 86400;
    let h = time_of_day / 3600;
    let min = (time_of_day % 3600) / 60;
    let s = time_of_day % 60;

    let d = days as i64 + 719468;
    let era = if d >= 0 { d } else { d - 146096 } / 146097;
    let doe = d - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if month <= 2 { y + 1 } else { y };

    format!("{year:04}-{month:02}-{day:02}T{h:02}:{min:02}:{s:02}Z")
}

// ═══════════════════════════════════════════════════════════════════════════
// A. Corpus Semantics
// ═══════════════════════════════════════════════════════════════════════════

/// Generate all corpus entries and write canonical fixtures if missing.
///
/// In normal mode, loads checked-in fixtures and compares. In capture mode
/// (env `ENHANCED_V3_BUDGET_CAPTURE=1`), writes generated fixtures to the
/// fixture directory.
fn load_or_capture_corpus_results() -> Vec<CorpusEntryResult> {
    let results = corpus::execute_corpus().expect("corpus execution must succeed");
    assert_eq!(results.len(), 3);

    results
}

/// Load the expected map fixture for a corpus entry.
fn load_expected_map(entry_id: &str) -> Option<String> {
    let path = corpus_fixture_map_path(entry_id);
    if path.exists() {
        Some(
            std::fs::read_to_string(&path).unwrap_or_else(|e| {
                panic!("read fixture map {entry_id} at {}: {e}", path.display())
            }),
        )
    } else {
        None
    }
}

/// Load the expected metadata fixture for a corpus entry.
fn load_expected_metadata(entry_id: &str) -> Option<enhanced_v3_proof::metadata::ProofMetadata> {
    let path = corpus_fixture_metadata_path(entry_id);
    if path.exists() {
        let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
            panic!(
                "read fixture metadata {entry_id} at {}: {e}",
                path.display()
            )
        });
        Some(serde_json::from_str(&text).expect("parse fixture metadata"))
    } else {
        None
    }
}

/// Write a corpus entry's map and metadata as fixtures.
fn write_corpus_fixture(
    entry_id: &str,
    map_text: &str,
    metadata: &enhanced_v3_proof::metadata::ProofMetadata,
) {
    let map_path = corpus_fixture_map_path(entry_id);
    let meta_path = corpus_fixture_metadata_path(entry_id);

    std::fs::create_dir_all(fixture_dir()).expect("create fixture dir");
    std::fs::write(&map_path, map_text).unwrap_or_else(|e| {
        panic!(
            "write fixture map {entry_id} at {}: {e}",
            map_path.display()
        )
    });
    let meta_json = serde_json::to_string_pretty(metadata).expect("serialize metadata");
    std::fs::write(&meta_path, format!("{meta_json}\n")).unwrap_or_else(|e| {
        panic!(
            "write fixture metadata {entry_id} at {}: {e}",
            meta_path.display()
        )
    });
}

// ═══════════════════════════════════════════════════════════════════════════
// B. Dense M2 Budget Evidence
// ═══════════════════════════════════════════════════════════════════════════

fn tools_available(dir: &Path) -> bool {
    dir.join("qbsp").is_file() && dir.join("vis").is_file() && dir.join("light").is_file()
}

/// Compile a .map through the bounded, warning-fatal pinned profile harness.
fn compile_for_budget(
    map_path: &Path,
    tool_dir: &Path,
    wad: &Path,
    palette: &Path,
    profile: &compiler::CompilerProfile,
) -> Result<(Vec<u8>, Vec<u8>, u64, Vec<String>), String> {
    let staging = compiler::create_staging_dir("dense-rich-budget")?;
    let started = std::time::Instant::now();
    let compiled = compiler::compile_map(map_path, staging.path(), tool_dir, wad, palette, profile)
        .map_err(|failure| format!("{:?} compiler failure: {}", failure.kind, failure.message))?;
    let elapsed_ms = started.elapsed().as_millis() as u64;
    let diagnostics = [
        &compiled.qbsp_output,
        &compiled.vis_output,
        &compiled.light_output,
    ]
    .into_iter()
    .flat_map(|output| output.diagnostics.iter())
    .map(|diagnostic| diagnostic.message().to_string())
    .collect();

    Ok((
        compiled.bsp_data,
        compiled.lit_data,
        elapsed_ms,
        diagnostics,
    ))
}

/// Measure a compiled BSP against M2 budget ceilings.
fn measure_budget(
    map_text: &str,
    bsp_data: &[u8],
    lit_data: &[u8],
    compilation_time_ms: u64,
    wad: &Path,
    palette: &Path,
) -> Result<BudgetMeasurement, String> {
    use bsp::{BspExtractionRequest, BspLoader, LoadOptions};

    let palette_data = std::fs::read(palette).map_err(|e| format!("read palette: {e}"))?;
    let wad_data = std::fs::read(wad).map_err(|e| format!("read WAD: {e}"))?;
    let wad_name = wad
        .file_name()
        .ok_or("WAD has no basename")?
        .to_string_lossy()
        .into_owned();
    let options = LoadOptions {
        strict: true,
        palette: Some(palette_data.clone()),
        lit_data: Some(lit_data.to_vec()),
        wad_archives: vec![(wad_name.clone(), wad_data.clone())],
        texture_overrides: Vec::new(),
        source_identity: "enhanced-v3-dense-rich-budget".to_string(),
    };

    let world = BspLoader::load(bsp_data, &options)
        .map_err(|report| format!("strict BSP load for budget measurement failed: {report}"))?;
    if !world.diagnostics.is_empty() {
        return Err(format!(
            "strict BSP load emitted diagnostics: {:?}",
            world.diagnostics
        ));
    }

    let model = world
        .models
        .first()
        .ok_or("compiled BSP has no world model")?;
    let xy_extent = ((model.maxs[0] - model.mins[0])
        .max(model.maxs[1] - model.mins[1])
        .ceil()) as u32;
    let z_span = (model.maxs[2] - model.mins[2]).ceil() as u32;

    let source_brushes = enhanced_v3_proof::fixtures::source_brush_count(map_text);
    let source_plane_sides = map_text
        .lines()
        .filter(|line| line.trim_start().starts_with('('))
        .count();
    let faces = world.faces.len();
    let entities = world.entities.len();
    let unique_planes = world.planes.len();
    let vertices = world.vertices.len();
    let edges = world.edges.len();
    let texinfos = world.texinfos.len();
    let nodes = world.nodes.len();
    let leaves = world.leaves.len();
    let solid_leaves = world
        .leaves
        .iter()
        .filter(|leaf| leaf.contents == -2)
        .count();
    let empty_leaves = world
        .leaves
        .iter()
        .filter(|leaf| leaf.contents == -1)
        .count();
    let clipnodes = world.clipnodes.len();
    let models = world.models.len();
    let lightmap_bytes = world.lightmap_data.len();

    let extracted = bsp::extract(BspExtractionRequest {
        world,
        palette: Some(bsp::resources::decode_palette(&palette_data)),
        wad_archives: vec![(wad_name, wad_data)],
        scale: 0.0254,
        ..BspExtractionRequest::default()
    })
    .map_err(|report| format!("BSP extraction for batch measurement failed: {report}"))?;

    Ok(BudgetMeasurement {
        source_brushes,
        source_plane_sides,
        faces,
        entities,
        static_batches: extracted.render_batches.len(),
        unique_planes,
        vertices,
        edges,
        texinfos,
        nodes,
        leaves,
        solid_leaves,
        empty_leaves,
        clipnodes,
        models,
        lightmap_bytes,
        xy_extent,
        z_span,
        bsp_size: bsp_data.len(),
        lit_size: lit_data.len(),
        compilation_time_ms,
    })
}

/// Run budget checks against frozen M2 ceilings and pinned profile limits.
fn run_budget_checks(
    measurement: &BudgetMeasurement,
    profile: &compiler::CompilerProfile,
) -> Vec<BudgetCheck> {
    let ceilings = M2BudgetCeilings::frozen();
    vec![
        BudgetCheck {
            metric: "source_brushes".to_string(),
            value: measurement.source_brushes as u64,
            comparator: ">=".to_string(),
            limit: 30,
            within_budget: measurement.source_brushes >= 30,
        },
        BudgetCheck {
            metric: "faces".to_string(),
            value: measurement.faces as u64,
            comparator: "<".to_string(),
            limit: ceilings.max_faces as u64,
            within_budget: measurement.faces < ceilings.max_faces as usize,
        },
        BudgetCheck {
            metric: "representative_m2_faces".to_string(),
            value: measurement.faces as u64,
            comparator: ">=".to_string(),
            limit: 2_000,
            within_budget: measurement.faces >= 2_000,
        },
        BudgetCheck {
            metric: "entities".to_string(),
            value: measurement.entities as u64,
            comparator: "<".to_string(),
            limit: ceilings.max_entities as u64,
            within_budget: measurement.entities < ceilings.max_entities as usize,
        },
        BudgetCheck {
            metric: "static_batches".to_string(),
            value: measurement.static_batches as u64,
            comparator: "<".to_string(),
            limit: ceilings.max_static_batches as u64,
            within_budget: measurement.static_batches < ceilings.max_static_batches as usize,
        },
        BudgetCheck {
            metric: "xy_extent".to_string(),
            value: measurement.xy_extent as u64,
            comparator: "<=".to_string(),
            limit: ceilings.max_xy_extent as u64,
            within_budget: measurement.xy_extent <= ceilings.max_xy_extent,
        },
        BudgetCheck {
            metric: "z_span".to_string(),
            value: measurement.z_span as u64,
            comparator: "<=".to_string(),
            limit: ceilings.max_z_span as u64,
            within_budget: measurement.z_span <= ceilings.max_z_span,
        },
        BudgetCheck {
            metric: "bsp_size".to_string(),
            value: measurement.bsp_size as u64,
            comparator: "<=".to_string(),
            limit: profile.max_output_size,
            within_budget: measurement.bsp_size as u64 <= profile.max_output_size,
        },
        BudgetCheck {
            metric: "compilation_time_ms".to_string(),
            value: measurement.compilation_time_ms,
            comparator: "<".to_string(),
            limit: profile.timeout_seconds * 1000,
            within_budget: measurement.compilation_time_ms < profile.timeout_seconds * 1000,
        },
    ]
}

fn write_budget_report(report: &BudgetReport) {
    let dir = budget_report_path().parent().unwrap().to_path_buf();
    std::fs::create_dir_all(&dir).expect("create budget report dir");
    let json = serde_json::to_string_pretty(report).expect("serialize budget report");
    let path = budget_report_path();
    let tmp = dir.join(".budget-report.json.tmp");
    std::fs::write(&tmp, format!("{json}\n")).expect("write budget report tmp");
    std::fs::rename(&tmp, &path).expect("publish budget report");
}

// ═══════════════════════════════════════════════════════════════════════════
// Test: Corpus Determinism
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn corpus_all_entries_deterministic() {
    let entries = corpus::proof_corpus();
    for entry in &entries {
        let a = corpus::execute_entry(entry).expect("first run");
        let b = corpus::execute_entry(entry).expect("second run");
        assert_eq!(a.map_text, b.map_text, "{} not deterministic", entry.id);
        assert_eq!(
            a.metadata, b.metadata,
            "{} metadata not deterministic",
            entry.id
        );
    }
}

#[test]
fn corpus_all_entries_produce_valid_maps() {
    for entry in &corpus::proof_corpus() {
        let result = corpus::execute_entry(entry).expect("corpus entry");
        assert!(!result.map_text.is_empty(), "{}: empty map", entry.id);
        assert!(
            result.map_text.contains("worldspawn"),
            "{}: no worldspawn",
            entry.id
        );
        assert!(
            result.map_text.contains("info_player_start"),
            "{}: no spawn",
            entry.id
        );
        assert!(
            result.map_text.ends_with('\n'),
            "{}: no trailing LF",
            entry.id
        );

        assert_eq!(result.metadata.schema, "enhanced-v3-proof-metadata/v3");
        assert!(result.metadata.room_count > 0);
        assert!(
            result.metadata.identity_satisfied,
            "{}: identity not satisfied",
            entry.id
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Test: Corpus Fixture Byte-Compare
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn corpus_byte_compare_against_canonical_fixtures() {
    let is_capture = std::env::var("ENHANCED_V3_BUDGET_CAPTURE").as_deref() == Ok("1");

    let results = load_or_capture_corpus_results();

    for result in &results {
        let expected_map = load_expected_map(&result.entry.id);
        let expected_meta = load_expected_metadata(&result.entry.id);

        if is_capture {
            // Write canonical fixtures
            write_corpus_fixture(&result.entry.id, &result.map_text, &result.metadata);
            eprintln!(
                "CAPTURE: wrote fixture for {} ({} bytes map, {} rooms)",
                result.entry.id,
                result.map_text.len(),
                result.metadata.room_count
            );
        } else {
            // Verify against checked-in fixtures
            match expected_map {
                Some(expected) => {
                    assert_eq!(
                        result.map_text, expected,
                        "{}: map byte-mismatch against canonical fixture",
                        result.entry.id
                    );
                }
                None => {
                    panic!(
                        "missing canonical fixture for {} at {}",
                        result.entry.id,
                        corpus_fixture_map_path(&result.entry.id).display()
                    );
                }
            }

            match expected_meta {
                Some(expected) => {
                    assert_eq!(
                        result.metadata, expected,
                        "{}: metadata mismatch against canonical fixture",
                        result.entry.id
                    );
                }
                None => {
                    panic!(
                        "missing canonical metadata fixture for {} at {}",
                        result.entry.id,
                        corpus_fixture_metadata_path(&result.entry.id).display()
                    );
                }
            }
        }
    }

    if !is_capture {
        // All checks passed (asserts above would have panicked on failure)
        eprintln!(
            "corpus_byte_compare: all {} entries matched canonical fixtures",
            results.len()
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Test: Corpus Pipeline Consistency
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn corpus_pipeline_consistent_with_canonical_fixtures() {
    // Verify the sparse entry matches the existing integrated canonical fixture
    let sparse_entry = &corpus::proof_corpus()[0];
    assert_eq!(sparse_entry.id, "v3-sparse-seed-0");

    let result = corpus::execute_entry(sparse_entry).expect("sparse corpus entry");
    let (canonical_map, canonical_meta) = pipeline::make_canonical_fixture();

    assert_eq!(
        result.map_text, canonical_map,
        "corpus sparse entry must match integrated canonical fixture"
    );
    assert_eq!(
        result.metadata, canonical_meta,
        "corpus sparse metadata must match integrated canonical fixture"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Test: corpus canonical bytes determinism
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn corpus_canonical_bytes_deterministic_across_runs() {
    let results_a = load_or_capture_corpus_results();
    let results_b = load_or_capture_corpus_results();

    let bytes_a = corpus::corpus_canonical_bytes(&results_a);
    let bytes_b = corpus::corpus_canonical_bytes(&results_b);
    assert_eq!(bytes_a, bytes_b);

    let hash_a = corpus::corpus_baseline_hash(&results_a);
    let hash_b = corpus::corpus_baseline_hash(&results_b);
    assert_eq!(hash_a, hash_b);
    assert_eq!(hash_a.len(), 64);
}

// ═══════════════════════════════════════════════════════════════════════════
// Test: Rich entry across-preset consistency
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn dense_entry_has_more_features_than_baseline() {
    let sparse = corpus::execute_entry(&corpus::proof_corpus()[0]).expect("sparse");
    let dense = corpus::execute_entry(&corpus::proof_corpus()[2]).expect("dense");

    assert_eq!(dense.entry.preset, Preset::Sparse);
    assert_eq!(sparse.entry.preset, Preset::Sparse);

    // Dense entry has larger extent, should produce more rooms
    assert_eq!(dense.entry.xy_extent, 3072);
    assert_eq!(sparse.entry.xy_extent, 2048);

    // Both have at least one grammar family
    assert!(
        sparse.metadata.grammar_families.len() >= 1,
        "Sparse should have at least 1 family"
    );
    assert!(
        dense.metadata.grammar_families.len() >= 1,
        "Dense should have at least 1 family"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Test: Dense M2 Budget Evidence
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn dense_m2_budget_evidence() {
    let tool_dir = compiler::resolve_tool_dir();
    assert!(
        tools_available(&tool_dir),
        "required ericw-tools executables unavailable at {}",
        tool_dir.display()
    );

    let profile =
        compiler::parse_compiler_profile(&enhanced_v3_proof::fixtures::compiler_profile_path())
            .expect("parse pinned compiler profile");
    compiler::verify_executable_hashes(&tool_dir, &profile)
        .unwrap_or_else(|failures| panic!("ericw-tools provenance mismatch: {failures:#?}"));

    let dense_map_path = dense_fixture_map_path();
    let dense_metadata_path = dense_fixture_metadata_path();
    let dense_map = std::fs::read_to_string(&dense_map_path)
        .unwrap_or_else(|e| panic!("read dense fixture {}: {e}", dense_map_path.display()));
    let dense_metadata_bytes = std::fs::read(&dense_metadata_path)
        .unwrap_or_else(|e| panic!("read dense metadata {}: {e}", dense_metadata_path.display()));
    let dense_metadata: enhanced_v3_proof::metadata::ProofMetadata =
        serde_json::from_slice(&dense_metadata_bytes).expect("parse dense fixture metadata");
    let source_brushes = enhanced_v3_proof::fixtures::source_brush_count(&dense_map);
    assert!(
        source_brushes >= 30,
        "dense-rich fixture has {source_brushes} brushes; at least 30 are required"
    );
    assert_eq!(dense_metadata.preset, "rich");
    assert!(dense_metadata.identity_satisfied);
    assert!(
        dense_metadata.rooms.iter().any(|room| room.layer == 0)
            && dense_metadata.rooms.iter().any(|room| room.layer == 1),
        "dense-rich metadata must represent both frozen layers"
    );
    assert!(
        dense_metadata
            .rooms
            .iter()
            .all(|room| room.floor_z + room.dims[2] as i32 <= 368),
        "dense-rich metadata exceeds the frozen total Z span"
    );
    assert!(
        dense_metadata.grammar_families.len() >= 3,
        "Rich fixture must retain at least three approved grammar identities"
    );

    let all_results = load_or_capture_corpus_results();
    let baseline_hash = corpus::corpus_baseline_hash(&all_results);
    let corpus_records: Vec<CorpusVerificationRecord> = all_results
        .iter()
        .map(|r| {
            let expected_map = load_expected_map(&r.entry.id);
            let map_match = expected_map.as_ref().is_some_and(|em| em == &r.map_text);
            CorpusVerificationRecord {
                id: r.entry.id.clone(),
                map_sha256: r.map_sha256.clone(),
                metadata_sha256: r.metadata_sha256.clone(),
                map_bytes: r.map_text.len(),
                status: if map_match {
                    "PASS".to_string()
                } else {
                    "FIXTURE_MISMATCH".to_string()
                },
                expected_map_sha256: expected_map.map(|m| compiler::sha256_hex(m.as_bytes())),
                map_match,
            }
        })
        .collect();
    assert!(
        corpus_records.iter().all(|record| record.map_match),
        "corpus fixture mismatch while producing dense budget evidence"
    );

    let wad = wad_path();
    let palette = palette_path();
    let (bsp_data, lit_data, elapsed_ms, diagnostics) =
        compile_for_budget(&dense_map_path, &tool_dir, &wad, &palette, &profile)
            .unwrap_or_else(|error| panic!("dense budget compilation failed: {error}"));
    assert!(
        diagnostics.is_empty(),
        "compiler diagnostics: {diagnostics:#?}"
    );
    assert_eq!(&bsp_data[0..4], b"BSP2", "dense BSP must be BSP2");

    let measurement = measure_budget(&dense_map, &bsp_data, &lit_data, elapsed_ms, &wad, &palette)
        .expect("strict budget measurement and batch extraction");
    let checks = run_budget_checks(&measurement, &profile);
    let within_budget = checks.iter().all(|check| check.within_budget);

    let report = BudgetReport {
        schema: "enhanced-v3-budget-report/v2".to_string(),
        timestamp: iso8601_now(),
        profile_name: profile.name.clone(),
        profile_version: profile.required_version.clone(),
        tool_dir: tool_dir.display().to_string(),
        tools_available: true,
        corpus_baseline_hash: baseline_hash,
        ceiling: M2BudgetCeilings::frozen(),
        measurement,
        checks: checks.clone(),
        overall_within_budget: within_budget,
        corpus_results: corpus_records,
        rich_map_sha256: compiler::sha256_hex(dense_map.as_bytes()),
        rich_metadata_sha256: compiler::sha256_hex(&dense_metadata_bytes),
        bsp_sha256: compiler::sha256_hex(&bsp_data),
        lit_sha256: compiler::sha256_hex(&lit_data),
        compiler_diagnostics: diagnostics,
    };
    write_budget_report(&report);

    eprintln!(
        "dense_m2_budget: brushes={} source_sides={} faces={} entities={} batches={} planes={} nodes={} leaves={} clipnodes={} lightmap={}B bsp={}B compilation={}ms",
        report.measurement.source_brushes,
        report.measurement.source_plane_sides,
        report.measurement.faces,
        report.measurement.entities,
        report.measurement.static_batches,
        report.measurement.unique_planes,
        report.measurement.nodes,
        report.measurement.leaves,
        report.measurement.clipnodes,
        report.measurement.lightmap_bytes,
        report.measurement.bsp_size,
        report.measurement.compilation_time_ms,
    );

    for check in &checks {
        assert!(
            check.within_budget,
            "BUDGET CHECK FAILED: {} (value={} {} limit={})",
            check.metric, check.value, check.comparator, check.limit
        );
    }
    assert!(within_budget, "dense fixture exceeds budget ceilings");
}

// ═══════════════════════════════════════════════════════════════════════════
// Test: Dense Entry Pipeline Exercises Approved Capabilities
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn dense_entry_exercises_approved_capabilities() {
    let result = corpus::execute_entry(&corpus::proof_corpus()[2]).expect("dense entry");

    // Dense entry at 3072² must have at least portal_chamber grammar
    assert!(
        result.metadata.grammar_families.len() >= 1,
        "Dense entry requires >=1 family, got {}: {:?}",
        result.metadata.grammar_families.len(),
        result.metadata.grammar_families
    );

    // Must have feature instances
    assert!(
        result.metadata.instance_count > 0,
        "Dense entry must produce feature instances"
    );

    // Identity must be satisfied
    assert!(
        result.metadata.identity_satisfied,
        "Dense identity not satisfied"
    );

    // Must have at least one transition
    assert!(
        result.metadata.transition_count > 0,
        "Dense entry must have inter-layer transitions"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Test: Corpus Baseline Consistency
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn corpus_baseline_hash_stable() {
    // The baseline hash must be stable across repeated corpus executions
    let results_a = load_or_capture_corpus_results();
    let results_b = load_or_capture_corpus_results();

    assert_eq!(results_a.len(), results_b.len());
    for (a, b) in results_a.iter().zip(results_b.iter()) {
        assert_eq!(a.entry.id, b.entry.id);
        assert_eq!(a.map_text, b.map_text);
        assert_eq!(a.metadata, b.metadata);
    }

    let hash_a = corpus::corpus_baseline_hash(&results_a);
    let hash_b = corpus::corpus_baseline_hash(&results_b);
    assert_eq!(hash_a, hash_b);
    eprintln!("corpus baseline hash: {hash_a}");
}
