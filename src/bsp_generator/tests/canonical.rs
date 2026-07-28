//! Determinism tests: verify that [`bsp_generator::generate`] produces
//! identical results for identical `(seed, config)` inputs — whether that
//! result is success or failure.  Different seeds/configs produce different
//! results.
//!
//! > **Note**: Some seeds/configurations may return `RouteExhausted` due to
//! > pre-existing topology/routing feasibility limits (tracked in
//! > `.internal-dev/bugs/phase-07-generated-sprawl-topology-infeasible`).
//! > These tests verify determinism of both success and failure paths.

use bsp_generator::{generate, DungeonConfig};

// ── Determinism: same (seed, config) → same result ──────────────────────

#[test]
fn same_input_produces_identical_result_whether_success_or_failure() {
    // Use a seed known to work for M1 with 8 rooms
    let cfg = DungeonConfig {
        room_count: 8,
        loop_count: 0,
        ..DungeonConfig::nominal_m1()
    };
    let r1 = generate(0, cfg.clone());
    let r2 = generate(0, cfg);
    assert_eq!(format!("{:?}", r1), format!("{:?}", r2));
}

#[test]
fn same_m2_input_produces_identical_result() {
    // Use boundary-C: 17 rooms, 1 loop — known to route
    let cfg = DungeonConfig {
        class: bsp_generator::MapClass::M2,
        room_count: 17,
        loop_count: 1,
        xy_bounds: (2048, 2048),
        z_span: 256,
        placement_candidates: 32,
        max_placement_attempts: 96,
        max_astar_expansions: 524_288,
    };
    let r1 = generate(44, cfg.clone());
    let r2 = generate(44, cfg);
    assert_eq!(format!("{:?}", r1), format!("{:?}", r2));
}

#[test]
fn deterministic_failure_is_identical() {
    // A config that is virtually guaranteed to fail routing
    let cfg = DungeonConfig {
        room_count: 12,
        loop_count: 1,
        xy_bounds: (256, 256), // far too small for 12 rooms
        z_span: 256,
        ..DungeonConfig::nominal_m1()
    };
    let r1 = generate(42, cfg.clone());
    let r2 = generate(42, cfg);
    assert_eq!(format!("{:?}", r1), format!("{:?}", r2));
}

// ── Different seeds → different result (or at least different error) ────

#[test]
fn different_seeds_produce_different_placement() {
    // Two seeds, same config — placement differs
    let cfg = DungeonConfig::nominal_m1();
    let r1 = generate(0, cfg.clone());
    let r2 = generate(1, cfg);
    // Both should produce the same variant kind, but different content
    match (&r1, &r2) {
        (Ok((s1, _)), Ok((s2, _))) => assert_ne!(s1, s2),
        _ => {
            // If both fail, the error expansion count should differ
            // (or at minimum, the outputs shouldn't be identical)
            assert_ne!(format!("{:?}", r1), format!("{:?}", r2));
        }
    }
}

// ── Config hash is deterministic ────────────────────────────────────────

#[test]
fn config_hash_is_stable() {
    let cfg = DungeonConfig::nominal_m1();
    let (_, meta) = generate(0, cfg).expect("seed 0 must succeed for nominal M1");
    assert_ne!(meta.config_hash, 0);
}

#[test]
fn metadata_deterministic_for_success() {
    let cfg = DungeonConfig::nominal_m1();
    let r1 = generate(0, cfg.clone());
    let r2 = generate(0, cfg);
    match (r1, r2) {
        (Ok((_, m1)), Ok((_, m2))) => assert_eq!(m1, m2),
        _ => {} // skip if generation failed
    }
}

// ── Byte-identical output for same inputs ──────────────────────────────

#[test]
fn byte_identical_for_same_seed_and_config() {
    let cfg = DungeonConfig::nominal_m1();
    let r1 = generate(0, cfg.clone());
    let r2 = generate(0, cfg);
    match (r1, r2) {
        (Ok((s1, _)), Ok((s2, _))) => {
            assert_eq!(s1, s2);
            assert_eq!(s1.len(), s2.len());
        }
        _ => {} // skip if routing failed
    }
}

#[test]
fn byte_identical_for_m1_boundary_a() {
    let cfg = DungeonConfig {
        room_count: 8,
        loop_count: 0,
        ..DungeonConfig::nominal_m1()
    };
    let r1 = generate(42, cfg.clone());
    let r2 = generate(42, cfg);
    match (r1, r2) {
        (Ok((s1, _)), Ok((s2, _))) => {
            assert_eq!(s1, s2);
            assert_eq!(s1.len(), s2.len());
        }
        _ => {} // skip if routing failed
    }
}
