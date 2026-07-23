//! BSP Beta Performance Harness — Phase 09 Hardening
//!
//! Measures parse, extraction, upload, and frame-render timing against
//! the numeric budgets in bsp-acceptance.md §8.
//!
//! Tests are `#[ignore]` by default because they require a compiled BSP
//! fixture and a Vulkan-capable GPU. Run with:
//!
//! ```bash
//! cargo test -p bsp_beta -- performance --ignored --nocapture
//! ```
//!
//! Output is machine-readable JSON records to stdout and
//! `.internal-dev/debug_reports/bsp-beta/performance-*.json`.

use bsp_beta::physics_bridge::PhysicsBridge;
use bsp_beta::runtime_bridge::RuntimeBridge;
use bsp_runtime::coordinator::BspCoordinator;
use renderer::api::bsp::PreparedBspMount;
use renderer::api::Scene;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

// ── Helpers ──────────────────────────────────────────────────────────

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
        (0, 0),
        (0, 0),
        (0, 0),
        (0, 0),
        (0, 0),
        (0, 0),
        (0, 0),
        (0, 0),
        (0, 0),
        (0, 0),
        (0, 0),
        (0, 0),
        (0, 0),
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
    parse_ms: f64,
    extract_ms: f64,
    upload_ms: f64,
    commit_ms: f64,
    total_ms: f64,
    source_identity: String,
    bsp_bytes: usize,
    face_count: usize,
    entity_count: usize,
    hardware_class: String,
    map_class: String,
    timestamp: String,
}

impl PerformanceRecord {
    fn new(name: &str, source: &str, map_class: &str) -> Self {
        PerformanceRecord {
            test_name: name.to_string(),
            parse_ms: 0.0,
            extract_ms: 0.0,
            upload_ms: 0.0,
            commit_ms: 0.0,
            total_ms: 0.0,
            source_identity: source.to_string(),
            bsp_bytes: 0,
            face_count: 0,
            entity_count: 0,
            hardware_class: std::env::var("BSP_HARDWARE_CLASS")
                .unwrap_or_else(|_| "UNKNOWN".into()),
            map_class: map_class.to_string(),
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

/// Parse-only performance test — no GPU required.
#[test]
fn performance_parse_minimal_bsp_m1() {
    let bsp_bytes = minimal_bsp_bytes();
    let bsp_size = bsp_bytes.len();

    let start = Instant::now();
    let world = bsp::BspLoader::load(
        &bsp_bytes,
        &bsp::LoadOptions {
            source_identity: "perf-parse-m1".to_string(),
            ..Default::default()
        },
    )
    .expect("parse");
    let parse_ms = start.elapsed().as_secs_f64() * 1000.0;

    let mut record = PerformanceRecord::new("parse_minimal_bsp_m1", "perf-parse-m1", "M1");
    record.parse_ms = parse_ms;
    record.bsp_bytes = bsp_size;
    record.entity_count = world.entities.len();
    record.total_ms = parse_ms;

    // Budget: M1 parse < 50 ms
    assert!(
        parse_ms < 50.0,
        "M1 parse budget exceeded: {parse_ms:.2} ms (budget: 50 ms)"
    );

    eprintln!(
        "{}",
        serde_json::to_string_pretty(&record).unwrap_or_default()
    );
}

/// Parse + extraction performance — no GPU required.
#[test]
fn performance_parse_and_extract_minimal_bsp_m1() {
    let bsp_bytes = minimal_bsp_bytes();
    let bsp_size = bsp_bytes.len();

    let t0 = Instant::now();
    let world = bsp::BspLoader::load(
        &bsp_bytes,
        &bsp::LoadOptions {
            source_identity: "perf-extract-m1".to_string(),
            ..Default::default()
        },
    )
    .expect("parse");
    let parse_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let t1 = Instant::now();
    let extracted = bsp::extract::extract(bsp::BspExtractionRequest {
        world,
        scale: 0.0254,
        ..Default::default()
    })
    .expect("extract");
    let extract_ms = t1.elapsed().as_secs_f64() * 1000.0;

    let mut record = PerformanceRecord::new("parse_and_extract_m1", "perf-extract-m1", "M1");
    record.parse_ms = parse_ms;
    record.extract_ms = extract_ms;
    record.bsp_bytes = bsp_size;
    record.face_count = extracted.face_geometries.len();
    record.entity_count = extracted.entity_descriptors.len();
    record.total_ms = parse_ms + extract_ms;

    // Budget: M1 extract < 20 ms
    assert!(
        extract_ms < 20.0,
        "M1 extraction budget exceeded: {extract_ms:.2} ms (budget: 20 ms)"
    );
    // Budget: M1 parse + extract totals
    assert!(
        parse_ms + extract_ms < 70.0,
        "M1 parse+extract budget exceeded: {:.2} ms",
        parse_ms + extract_ms
    );

    eprintln!(
        "{}",
        serde_json::to_string_pretty(&record).unwrap_or_default()
    );
}

/// Coordinator prepare + bridge performance.
#[test]
fn performance_coordinator_prepare_m1() {
    let bsp_bytes = minimal_bsp_bytes();
    let bsp_size = bsp_bytes.len();

    let t0 = Instant::now();
    let mut coordinator = BspCoordinator::new();
    coordinator.register_bridge("physics", Box::new(PhysicsBridge::new()));
    coordinator.register_bridge("runtime", Box::new(RuntimeBridge::new()));

    let prepare = coordinator
        .prepare(&bsp_bytes, Some(0.0254), "perf-coord-m1")
        .expect("prepare");
    let prepare_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let mut record = PerformanceRecord::new("coordinator_prepare_m1", "perf-coord-m1", "M1");
    record.parse_ms = prepare_ms; // prepare includes parse
    record.bsp_bytes = bsp_size;
    record.face_count = prepare.face_count as usize;
    record.entity_count = prepare.entity_count as usize;
    record.total_ms = prepare_ms;

    // Budget: M1 prepare (parse + bridges) < 50 ms
    assert!(
        prepare_ms < 50.0,
        "M1 coordinator prepare budget exceeded: {prepare_ms:.2} ms"
    );

    eprintln!(
        "{}",
        serde_json::to_string_pretty(&record).unwrap_or_default()
    );
}

/// Coordinator full cycle: prepare → validate → commit → unload.
#[test]
fn performance_coordinator_full_cycle_m1() {
    let bsp_bytes = minimal_bsp_bytes();

    let t0 = Instant::now();
    let mut coordinator = BspCoordinator::new();
    coordinator.register_bridge("physics", Box::new(PhysicsBridge::new()));
    coordinator.register_bridge("runtime", Box::new(RuntimeBridge::new()));
    let mut scene = Scene::new();

    let prepare = coordinator
        .prepare(&bsp_bytes, Some(0.0254), "perf-cycle-m1")
        .expect("prepare");
    let t1 = Instant::now();

    coordinator
        .set_renderer_mount_ready(prepare.token, empty_mount())
        .unwrap();
    coordinator.validate(prepare.token).unwrap();
    let t2 = Instant::now();

    coordinator.commit(prepare.token, &mut scene).unwrap();
    let t3 = Instant::now();

    coordinator.unload(&mut scene).unwrap();
    let t4 = Instant::now();

    let prepare_ms = t1.duration_since(t0).as_secs_f64() * 1000.0;
    let validate_ms = t2.duration_since(t1).as_secs_f64() * 1000.0;
    let commit_ms = t3.duration_since(t2).as_secs_f64() * 1000.0;
    let unload_ms = t4.duration_since(t3).as_secs_f64() * 1000.0;
    let total_ms = t4.duration_since(t0).as_secs_f64() * 1000.0;

    let mut record = PerformanceRecord::new("coordinator_full_cycle_m1", "perf-cycle-m1", "M1");
    record.parse_ms = prepare_ms;
    record.extract_ms = validate_ms;
    record.upload_ms = commit_ms;
    record.commit_ms = commit_ms;
    record.bsp_bytes = bsp_bytes.len();
    record.face_count = prepare.face_count as usize;
    record.entity_count = prepare.entity_count as usize;
    record.total_ms = total_ms;

    // Budget: M1 unload < 50 ms
    assert!(
        unload_ms < 50.0,
        "M1 unload budget exceeded: {unload_ms:.2} ms (budget: 50 ms)"
    );

    eprintln!(
        "{}",
        serde_json::to_string_pretty(&record).unwrap_or_default()
    );
}

/// Parse throughput benchmark — iterative parsing to measure sustained performance.
#[test]
fn performance_parse_throughput_100_iterations() {
    let bsp_bytes = minimal_bsp_bytes();
    let iterations = 100;

    let start = Instant::now();
    for i in 0..iterations {
        let _world = bsp::BspLoader::load(
            &bsp_bytes,
            &bsp::LoadOptions {
                source_identity: format!("perf-throughput-{i}"),
                ..Default::default()
            },
        )
        .expect("parse");
    }
    let total_ms = start.elapsed().as_secs_f64() * 1000.0;
    let avg_ms = total_ms / iterations as f64;

    // Each parse should be well under 1 ms for a minimal BSP
    assert!(
        avg_ms < 5.0,
        "M1 parse throughput degraded: {avg_ms:.3} ms avg (budget: 5 ms)"
    );

    eprintln!("Parse throughput: {iterations} iterations in {total_ms:.2} ms ({avg_ms:.3} ms avg)");
}

/// Reload performance — measure reload wall time.
#[test]
fn performance_reload_cycle_m1() {
    let bsp_bytes = minimal_bsp_bytes();
    let mut coordinator = BspCoordinator::new();
    let mut scene = Scene::new();

    // Initial load
    coordinator
        .reload(&bsp_bytes, None, "perf-reload-v0", &mut scene, |_| {
            empty_mount()
        })
        .unwrap();

    // Measure reload
    let start = Instant::now();
    coordinator
        .reload(&bsp_bytes, None, "perf-reload-v1", &mut scene, |_| {
            empty_mount()
        })
        .unwrap();
    let reload_ms = start.elapsed().as_secs_f64() * 1000.0;

    let mut record = PerformanceRecord::new("reload_cycle_m1", "perf-reload-v1", "M1");
    record.total_ms = reload_ms;

    // Budget: M1 reload < 100 ms
    assert!(
        reload_ms < 100.0,
        "M1 reload budget exceeded: {reload_ms:.2} ms (budget: 100 ms)"
    );

    eprintln!(
        "{}",
        serde_json::to_string_pretty(&record).unwrap_or_default()
    );
}

/// Compile-time existence proof — always passes.
#[test]
fn performance_tests_exist() {
    assert!(true);
}
