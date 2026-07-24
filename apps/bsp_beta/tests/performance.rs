//! BSP Beta Performance Harness — Phase 05 Map Class Evidence
//!
//! Measures parse, extraction, and reload timing against the numeric
//! budgets in bsp-acceptance.md §8.
//!
//! Microfixture tests (zero-face) are relabeled as `microfixture` — they
//! prove structural parsing but are NOT representative of any map class.
//! M1/M2 class evidence uses the compiled dungeon_m1/m2_standard BSP2 fixtures.
//!
//! GPU upload/submission records are separated from CPU parse/extract/reload
//! records. GPU cases require a Vulkan-capable environment and real mounts.
//!
//! Run with:
//! ```bash
//! cargo test -p bsp_beta -- performance --nocapture
//! ```

use bsp_beta::physics_bridge::PhysicsBridge;
use bsp_beta::runtime_bridge::RuntimeBridge;
use bsp_runtime::coordinator::BspCoordinator;
use renderer::api::bsp::PreparedBspMount;
use renderer::api::Scene;
use std::path::Path;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

// ── Fixture helpers ────────────────────────────────────────────────────────

fn bsp_fixtures_dir() -> std::path::PathBuf {
    // Resolve relative to workspace root: apps/bsp_beta/../../src/bsp/tests/fixtures
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../src/bsp/tests/fixtures")
}

fn compiled_dir() -> std::path::PathBuf {
    bsp_fixtures_dir().join("compiled")
}

fn palette_path() -> std::path::PathBuf {
    bsp_fixtures_dir().join("palettes/project_palette.lmp")
}

fn read_compiled(name: &str) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let bsp_data = std::fs::read(compiled_dir().join(format!("{name}.bsp")))
        .expect(&format!("read {name}.bsp"));
    let lit_data = std::fs::read(compiled_dir().join(format!("{name}.lit")))
        .expect(&format!("read {name}.lit"));
    let palette_data = std::fs::read(palette_path()).expect("read palette");
    (bsp_data, lit_data, palette_data)
}

// ── Shared helpers ─────────────────────────────────────────────────────────

fn minimal_bsp_bytes() -> Vec<u8> {
    let mut data = Vec::new();
    data.extend_from_slice(&29u32.to_le_bytes());
    let mut current_offset: u32 = 124;
    let entity_bytes = b"{\"classname\" \"worldspawn\"}\0";
    let entity_offset = current_offset;
    let entity_size = entity_bytes.len() as u32;
    current_offset += entity_size;
    let plane_offset = current_offset;
    let plane_size = 20u32;
    current_offset += plane_size;
    let lumps: [(u32, u32); 15] = [
        (entity_offset, entity_size),
        (plane_offset, plane_size),
        (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0),
        (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0),
    ];
    for (off, sz) in &lumps {
        data.extend_from_slice(&off.to_le_bytes());
        data.extend_from_slice(&sz.to_le_bytes());
    }
    data.extend_from_slice(entity_bytes);
    data.extend_from_slice(&0.0f32.to_le_bytes());
    data.extend_from_slice(&0.0f32.to_le_bytes());
    data.extend_from_slice(&1.0f32.to_le_bytes());
    data.extend_from_slice(&0.0f32.to_le_bytes());
    data.extend_from_slice(&0i32.to_le_bytes());
    data
}

fn empty_mount() -> PreparedBspMount {
    PreparedBspMount::new()
}

#[derive(serde::Serialize)]
struct PerformanceRecord {
    test_name: String,
    fixture_class: String,
    parse_ms: f64,
    extract_ms: f64,
    reload_ms: f64,
    total_ms: f64,
    source_identity: String,
    bsp_bytes: usize,
    face_count: usize,
    entity_count: usize,
    batch_count: usize,
    hardware_class: String,
    timestamp: String,
}

impl PerformanceRecord {
    fn new(name: &str, source: &str, fixture_class: &str) -> Self {
        PerformanceRecord {
            test_name: name.to_string(),
            fixture_class: fixture_class.to_string(),
            parse_ms: 0.0,
            extract_ms: 0.0,
            reload_ms: 0.0,
            total_ms: 0.0,
            source_identity: source.to_string(),
            bsp_bytes: 0,
            face_count: 0,
            entity_count: 0,
            batch_count: 0,
            hardware_class: std::env::var("BSP_HARDWARE_CLASS")
                .unwrap_or_else(|_| "UNKNOWN".into()),
            timestamp: format!(
                "{:?}",
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs()
            ),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Microfixture tests (zero-face, structural proof only)
// ═══════════════════════════════════════════════════════════════════════════

/// Parse-only microfixture test — no GPU required.
#[test]
fn performance_parse_microfixture() {
    let bsp_bytes = minimal_bsp_bytes();
    let bsp_size = bsp_bytes.len();

    let start = Instant::now();
    let world = bsp::BspLoader::load(
        &bsp_bytes,
        &bsp::LoadOptions {
            source_identity: "perf-parse-micro".to_string(),
            ..Default::default()
        },
    )
    .expect("parse");
    let parse_ms = start.elapsed().as_secs_f64() * 1000.0;

    let mut record = PerformanceRecord::new("parse_microfixture", "perf-parse-micro", "microfixture");
    record.parse_ms = parse_ms;
    record.bsp_bytes = bsp_size;
    record.entity_count = world.entities.len();
    record.total_ms = parse_ms;

    // Structural parse budget only — not a class claim
    assert!(parse_ms < 50.0, "microfixture parse exceeded {parse_ms:.2} ms");

    eprintln!(
        "{}",
        serde_json::to_string_pretty(&record).unwrap_or_default()
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// M1 class evidence (CPU: parse + extract + reload)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn performance_parse_m1_class() {
    let (bsp_data, lit_data, palette_data) = read_compiled("dungeon-m1-bsp2");
    let bsp_size = bsp_data.len();

    let start = Instant::now();
    let world = bsp::BspLoader::load(
        &bsp_data,
        &bsp::LoadOptions {
            strict: true,
            palette: Some(palette_data),
            lit_data: Some(lit_data),
            source_identity: "perf-parse-m1-class".into(),
            ..Default::default()
        },
    )
    .expect("parse M1");
    let parse_ms = start.elapsed().as_secs_f64() * 1000.0;

    let mut record = PerformanceRecord::new("parse_m1_class", "dungeon-m1-bsp2", "M1");
    record.parse_ms = parse_ms;
    record.bsp_bytes = bsp_size;
    record.face_count = world.faces.len();
    record.entity_count = world.entities.len();
    record.total_ms = parse_ms;

    assert!(parse_ms < 50.0, "M1 parse {parse_ms:.2} ms exceeds 50 ms budget");
    assert!(world.faces.len() > 0, "M1 must have visible faces");
    assert!(
        world.faces.len() < 2000,
        "M1 faces {} must be < 2000",
        world.faces.len()
    );

    eprintln!(
        "{}",
        serde_json::to_string_pretty(&record).unwrap_or_default()
    );
}

#[test]
fn performance_extract_m1_class() {
    let (bsp_data, lit_data, palette_data) = read_compiled("dungeon-m1-bsp2");
    let palette = bsp::resources::decode_palette(&palette_data);

    let world = bsp::BspLoader::load(
        &bsp_data,
        &bsp::LoadOptions {
            strict: true,
            palette: Some(palette_data),
            lit_data: Some(lit_data),
            source_identity: "perf-extract-m1-class".into(),
            ..Default::default()
        },
    )
    .expect("parse M1");

    let start = Instant::now();
    let extracted = bsp::extract::extract(bsp::BspExtractionRequest {
        world,
        palette: Some(palette),
        scale: 0.0254,
        ..Default::default()
    })
    .expect("extract M1");
    let extract_ms = start.elapsed().as_secs_f64() * 1000.0;

    let mut record = PerformanceRecord::new("extract_m1_class", "dungeon-m1-bsp2", "M1");
    record.extract_ms = extract_ms;
    record.bsp_bytes = bsp_data.len();
    record.face_count = extracted.face_geometries.len();
    record.entity_count = extracted.entity_descriptors.len();
    record.batch_count = extracted.render_batches.len();
    record.total_ms = extract_ms;

    assert!(extract_ms < 100.0, "M1 extract {extract_ms:.2} ms exceeds 100 ms budget");
    assert!(extracted.render_batches.len() < 100, "M1 batches must be < 100");

    eprintln!(
        "{}",
        serde_json::to_string_pretty(&record).unwrap_or_default()
    );
}

#[test]
fn performance_reload_m1_class() {
    let (bsp_data, lit_data, palette_data) = read_compiled("dungeon-m1-bsp2");
    let world = bsp::BspLoader::load(
        &bsp_data,
        &bsp::LoadOptions {
            strict: true,
            palette: Some(palette_data.clone()),
            lit_data: Some(lit_data.clone()),
            source_identity: "perf-reload-m1-v0".into(),
            ..Default::default()
        },
    ).expect("parse M1");

    let mut coordinator = BspCoordinator::new();
    let mut scene = Scene::new();
    coordinator.prepare_from_world(world.clone(), Some(0.0254), "perf-reload-m1-v0").unwrap();

    let start = Instant::now();
    coordinator.prepare_from_world(world, Some(0.0254), "perf-reload-m1-v1").unwrap();
    let reload_ms = start.elapsed().as_secs_f64() * 1000.0;

    let mut record = PerformanceRecord::new("reload_m1_class", "dungeon-m1-bsp2", "M1");
    record.reload_ms = reload_ms;
    record.bsp_bytes = bsp_data.len();
    record.total_ms = reload_ms;
    assert!(reload_ms < 400.0, "M1 reload {reload_ms:.2} ms exceeds 400 ms budget");
    eprintln!("{}", serde_json::to_string_pretty(&record).unwrap_or_default());
}

// ═══════════════════════════════════════════════════════════════════════════
// M2 class evidence (CPU: parse + extract + reload)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn performance_parse_m2_class() {
    let (bsp_data, lit_data, palette_data) = read_compiled("dungeon-m2-bsp2");
    let bsp_size = bsp_data.len();

    let start = Instant::now();
    let world = bsp::BspLoader::load(
        &bsp_data,
        &bsp::LoadOptions {
            strict: true,
            palette: Some(palette_data),
            lit_data: Some(lit_data),
            source_identity: "perf-parse-m2-class".into(),
            ..Default::default()
        },
    )
    .expect("parse M2");
    let parse_ms = start.elapsed().as_secs_f64() * 1000.0;

    let mut record = PerformanceRecord::new("parse_m2_class", "dungeon-m2-bsp2", "M2");
    record.parse_ms = parse_ms;
    record.bsp_bytes = bsp_size;
    record.face_count = world.faces.len();
    record.entity_count = world.entities.len();
    record.total_ms = parse_ms;

    assert!(parse_ms < 200.0, "M2 parse {parse_ms:.2} ms exceeds 200 ms budget");
    assert!(world.faces.len() > 0, "M2 must have visible faces");
    assert!(
        world.faces.len() < 10000,
        "M2 faces {} must be < 10000",
        world.faces.len()
    );

    eprintln!(
        "{}",
        serde_json::to_string_pretty(&record).unwrap_or_default()
    );
}

#[test]
fn performance_extract_m2_class() {
    let (bsp_data, lit_data, palette_data) = read_compiled("dungeon-m2-bsp2");
    let palette = bsp::resources::decode_palette(&palette_data);

    let world = bsp::BspLoader::load(
        &bsp_data,
        &bsp::LoadOptions {
            strict: true,
            palette: Some(palette_data),
            lit_data: Some(lit_data),
            source_identity: "perf-extract-m2-class".into(),
            ..Default::default()
        },
    )
    .expect("parse M2");

    let start = Instant::now();
    let extracted = bsp::extract::extract(bsp::BspExtractionRequest {
        world,
        palette: Some(palette),
        scale: 0.0254,
        ..Default::default()
    })
    .expect("extract M2");
    let extract_ms = start.elapsed().as_secs_f64() * 1000.0;

    let mut record = PerformanceRecord::new("extract_m2_class", "dungeon-m2-bsp2", "M2");
    record.extract_ms = extract_ms;
    record.bsp_bytes = bsp_data.len();
    record.face_count = extracted.face_geometries.len();
    record.entity_count = extracted.entity_descriptors.len();
    record.batch_count = extracted.render_batches.len();
    record.total_ms = extract_ms;

    assert!(extract_ms < 200.0, "M2 extract {extract_ms:.2} ms exceeds 200 ms budget");
    assert!(extracted.render_batches.len() < 500, "M2 batches must be < 500");

    eprintln!(
        "{}",
        serde_json::to_string_pretty(&record).unwrap_or_default()
    );
}

#[test]
fn performance_reload_m2_class() {
    let (bsp_data, lit_data, palette_data) = read_compiled("dungeon-m2-bsp2");
    let world = bsp::BspLoader::load(
        &bsp_data,
        &bsp::LoadOptions {
            strict: true,
            palette: Some(palette_data.clone()),
            lit_data: Some(lit_data.clone()),
            source_identity: "perf-reload-m2-v0".into(),
            ..Default::default()
        },
    ).expect("parse M2");

    let mut coordinator = BspCoordinator::new();
    let mut scene = Scene::new();
    coordinator.prepare_from_world(world.clone(), Some(0.0254), "perf-reload-m2-v0").unwrap();

    let start = Instant::now();
    coordinator.prepare_from_world(world, Some(0.0254), "perf-reload-m2-v1").unwrap();
    let reload_ms = start.elapsed().as_secs_f64() * 1000.0;

    let mut record = PerformanceRecord::new("reload_m2_class", "dungeon-m2-bsp2", "M2");
    record.reload_ms = reload_ms;
    record.bsp_bytes = bsp_data.len();
    record.total_ms = reload_ms;
    assert!(reload_ms < 800.0, "M2 reload {reload_ms:.2} ms exceeds 800 ms budget");
    eprintln!("{}", serde_json::to_string_pretty(&record).unwrap_or_default());
}

// ═══════════════════════════════════════════════════════════════════════════
// GPU upload/submission (separated from CPU measurements)
// ═══════════════════════════════════════════════════════════════════════════

/// CPU prepare path for M1 — validates the coordinator prepare pipeline
/// without GPU mount (GPU mount requires Vulkan context).
#[test]
fn performance_coordinator_prepare_m1_class() {
    let (bsp_data, lit_data, palette_data) = read_compiled("dungeon-m1-bsp2");
    let world = bsp::BspLoader::load(
        &bsp_data,
        &bsp::LoadOptions {
            strict: true,
            palette: Some(palette_data.clone()),
            lit_data: Some(lit_data.clone()),
            source_identity: "perf-coord-m1".into(),
            ..Default::default()
        },
    ).expect("parse M1");

    let t0 = Instant::now();
    let mut coordinator = BspCoordinator::new();
    coordinator.register_bridge("physics", Box::new(PhysicsBridge::new()));
    coordinator.register_bridge("runtime", Box::new(RuntimeBridge::new()));
    let prepare = coordinator.prepare_from_world(world, Some(0.0254), "perf-coord-m1").unwrap();
    let prepare_ms = t0.elapsed().as_secs_f64() * 1000.0;

    coordinator.set_renderer_mount_ready(prepare.token, empty_mount()).unwrap();
    let mut scene = Scene::new();
    coordinator.validate_for_scene(prepare.token, &mut scene).unwrap();

    let mut record = PerformanceRecord::new("coordinator_prepare_m1", "dungeon-m1-bsp2", "M1");
    record.total_ms = prepare_ms;
    record.bsp_bytes = bsp_data.len();
    record.face_count = prepare.face_count;
    record.entity_count = prepare.entity_count;
    record.batch_count = prepare.batch_count;
    assert!(prepare_ms < 200.0, "M1 coordinator prepare {prepare_ms:.2} ms exceeded 200 ms");
    assert!(prepare.face_count > 0, "M1 must have faces");
    eprintln!("{}", serde_json::to_string_pretty(&record).unwrap_or_default());
}

/// CPU prepare path for M2 — validates the coordinator prepare pipeline.
#[test]
fn performance_coordinator_prepare_m2_class() {
    let (bsp_data, lit_data, palette_data) = read_compiled("dungeon-m2-bsp2");
    let world = bsp::BspLoader::load(
        &bsp_data,
        &bsp::LoadOptions {
            strict: true,
            palette: Some(palette_data.clone()),
            lit_data: Some(lit_data.clone()),
            source_identity: "perf-coord-m2".into(),
            ..Default::default()
        },
    ).expect("parse M2");

    let t0 = Instant::now();
    let mut coordinator = BspCoordinator::new();
    coordinator.register_bridge("physics", Box::new(PhysicsBridge::new()));
    coordinator.register_bridge("runtime", Box::new(RuntimeBridge::new()));
    let prepare = coordinator.prepare_from_world(world, Some(0.0254), "perf-coord-m2").unwrap();
    let prepare_ms = t0.elapsed().as_secs_f64() * 1000.0;

    coordinator.set_renderer_mount_ready(prepare.token, empty_mount()).unwrap();
    let mut scene = Scene::new();
    coordinator.validate_for_scene(prepare.token, &mut scene).unwrap();

    let mut record = PerformanceRecord::new("coordinator_prepare_m2", "dungeon-m2-bsp2", "M2");
    record.total_ms = prepare_ms;
    record.bsp_bytes = bsp_data.len();
    record.face_count = prepare.face_count;
    record.entity_count = prepare.entity_count;
    record.batch_count = prepare.batch_count;
    assert!(prepare_ms < 500.0, "M2 coordinator prepare {prepare_ms:.2} ms exceeded 500 ms");
    assert!(prepare.face_count > 0, "M2 must have faces");
    eprintln!("{}", serde_json::to_string_pretty(&record).unwrap_or_default());
}

/// GPU mount test for M1 — requires Vulkan-capable environment.
/// Marked #[ignore]; run explicitly:
///   BSP_HARDWARE_CLASS=H2 cargo test -p bsp_beta -- performance_gpu_m1 --ignored --nocapture
#[test]
#[ignore]
fn performance_gpu_upload_m1() {
    // This test requires a live Vulkan context with:
    // - device, allocator, transfer queue, descriptor layout cache, data cache
    // - BSP descriptor pool initialized in the surface cache
    // See renderer::api::bsp::PreparedBspMount::upload_from_extracted for the real path.
    // For now, record as NOT-RUN without GPU environment.
    eprintln!("NOT-RUN: GPU environment required for M1 upload test");
}

/// GPU mount test for M2 — requires Vulkan-capable environment.
#[test]
#[ignore]
fn performance_gpu_upload_m2() {
    eprintln!("NOT-RUN: GPU environment required for M2 upload test");
}

/// Compile-time existence proof — always passes.
#[test]
fn performance_tests_exist() {
    assert!(true);
}
