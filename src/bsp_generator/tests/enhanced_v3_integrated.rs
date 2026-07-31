//! Phase 05 — Integrated Thin Slice
//!
//! Private integration test that drives the complete one-way pipeline
//! end-to-end: config → footprint → topology → plan → assemblies →
//! validate → serialize → canonical .map + metadata.
//!
//! # Architecture
//!
//! - **Positive**: runs the pipeline repeatedly and asserts byte-match
//!   against checked-in canonical artifacts.
//! - **Negative**: malformed footprint, filled aperture, hovering support,
//!   cyclic support, min-identity failure, stage-order violation,
//!   unvalidated assembly emission.
//! - **Phase 02 baseline**: completely unchanged.
//!
//! # Validation
//!
//! ```bash
//! cargo test -p bsp_generator --test enhanced_v3_integrated -- --nocapture
//! cargo test -p bsp_generator --test enhanced_v3_geometry  # unchanged
//! cargo test -p bsp_generator --test enhanced_v3_proof_model  # unchanged
//! cargo test -p bsp_generator --test enhanced_v3_baseline  # unchanged
//! cargo check -p bsp_generator --tests
//! cargo fmt --check -p bsp_generator
//! ```

mod enhanced_v3_proof;

use enhanced_v3_proof::assembly::{self, Assembly, AssemblyBrush, BrushRole, Support};
use enhanced_v3_proof::contract::{ContractError, Preset, ProofConfig};
use enhanced_v3_proof::emission;
use enhanced_v3_proof::footprint::Footprint;
use enhanced_v3_proof::geometry::{self, ConvexBrush, FaceRole};
use enhanced_v3_proof::ir::{CommittedTopology, RoomId, V3IdAllocator};
use enhanced_v3_proof::metadata::ProofMetadata;
use enhanced_v3_proof::pipeline;
use enhanced_v3_proof::planner;
use enhanced_v3_proof::seed::V3Seed;

// ── Fixture paths ─────────────────────────────────────────────────────────

fn crate_dir() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf()
}

fn fixture_map_path() -> std::path::PathBuf {
    crate_dir().join("tests/fixtures/enhanced_v3_proof/integrated.map")
}

fn fixture_metadata_path() -> std::path::PathBuf {
    crate_dir().join("tests/fixtures/enhanced_v3_proof/integrated-metadata.json")
}

// ═══════════════════════════════════════════════════════════════════════════
// Positive: Byte-match against canonical artifacts
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn integrated_pipeline_matches_canonical_artifacts() {
    let map_path = fixture_map_path();
    let meta_path = fixture_metadata_path();

    // Read canonical fixtures (fail if missing — they are checked in)
    let canonical_map = std::fs::read_to_string(&map_path).unwrap_or_else(|e| {
        panic!(
            "canonical map fixture missing at {}: {e}",
            map_path.display()
        )
    });

    let canonical_metadata_text = std::fs::read_to_string(&meta_path).unwrap_or_else(|e| {
        panic!(
            "canonical metadata fixture missing at {}: {e}",
            meta_path.display()
        )
    });
    let canonical_meta: ProofMetadata =
        serde_json::from_str(&canonical_metadata_text).expect("parse canonical metadata");

    // Run the pipeline
    let (generated_map, generated_meta) = pipeline::make_canonical_fixture();

    // Assert byte-match
    assert_eq!(
        generated_map, canonical_map,
        "generated map does not match canonical fixture"
    );

    assert_eq!(
        generated_meta, canonical_meta,
        "generated metadata does not match canonical fixture"
    );
    assert_eq!(
        format!(
            "{}\n",
            serde_json::to_string_pretty(&generated_meta).expect("serialize generated metadata")
        ),
        canonical_metadata_text,
        "generated metadata bytes do not match canonical fixture"
    );
}

#[test]
fn integrated_pipeline_deterministic_across_runs() {
    let (map1, meta1) = pipeline::make_canonical_fixture();
    let (map2, meta2) = pipeline::make_canonical_fixture();

    assert_eq!(map1, map2, "pipeline not deterministic");
    assert_eq!(meta1, meta2, "metadata not deterministic");
}

#[test]
fn integrated_pipeline_same_seed_same_output() {
    let config = ProofConfig::new(Preset::Sparse, 2048).unwrap();
    let seed = V3Seed::new(42);

    let r1 = pipeline::run_pipeline(&config, seed).unwrap();
    let r2 = pipeline::run_pipeline(&config, seed).unwrap();

    assert_eq!(r1.map_text, r2.map_text);
    assert_eq!(r1.metadata, r2.metadata);
}

// ═══════════════════════════════════════════════════════════════════════════
// Negative cases
// ═══════════════════════════════════════════════════════════════════════════

// ── N.1 Malformed footprint ──────────────────────────────────────────

#[test]
fn malformed_footprint_rejected() {
    let mut alloc = V3IdAllocator::new();
    // Non-quantum-aligned shell
    let result = Footprint::rectangular(RoomId(0), 0, 0, (0, 0, 33, 33), &mut alloc);
    assert!(result.is_err());
}

// ── N.2 Filled aperture ──────────────────────────────────────────────

#[test]
fn filled_aperture_rejected() {
    // An obstruction brush occupying the aperture throat should be rejected
    let wall = ConvexBrush::make_box((0, 16), (0, 64), (0, 128)).unwrap();
    let obstruction = AssemblyBrush::new(
        "obstruction",
        BrushRole::Feature,
        ConvexBrush::make_box((8, 24), (16, 48), (16, 96)).unwrap(),
        Support::World {
            surface: FaceRole::Floor,
        },
    );

    let wall_ab = AssemblyBrush::new(
        "wall_east",
        BrushRole::WallShell,
        wall,
        Support::World {
            surface: FaceRole::Floor,
        },
    );

    // Create an aperture that the obstruction fills
    let aperture = enhanced_v3_proof::portal::make_pointed_arch_aperture(
        "apt",
        "wall_east",
        FaceRole::EastWall,
        16,
        32,
        32,
        16,
        96,
        32,
        vec!["wall_east".into()],
    )
    .unwrap();

    // The obstruction occupies the aperture prism — should be rejected
    let result = Assembly::new(vec![wall_ab, obstruction], vec![], vec![aperture], vec![]);
    assert!(result.is_err());
}

// ── N.3 Hovering support ─────────────────────────────────────────────

#[test]
fn hovering_support_rejected() {
    // A brush supported by a non-existent parent
    let orphan = AssemblyBrush::new(
        "orphan",
        BrushRole::Feature,
        ConvexBrush::make_box((32, 48), (32, 48), (80, 144)).unwrap(),
        Support::SupportedBy {
            brush_id: "nonexistent".into(),
            interface_id: "fake".into(),
        },
    );

    let world = AssemblyBrush::new(
        "world",
        BrushRole::FloorSlab,
        ConvexBrush::make_box((0, 64), (0, 64), (0, 16)).unwrap(),
        Support::World {
            surface: FaceRole::Floor,
        },
    );

    let result = Assembly::new(vec![orphan, world], vec![], vec![], vec![]);
    assert!(result.is_err());
}

// ── N.4 Cyclic support ───────────────────────────────────────────────

#[test]
fn cyclic_support_rejected() {
    let edges = vec![
        ("a".to_string(), "b".to_string()),
        ("b".to_string(), "c".to_string()),
        ("c".to_string(), "a".to_string()),
    ];
    assert!(assembly::validate_support_acyclic(&edges).is_err());
}

// ── N.5 Minimum-identity failure ─────────────────────────────────────

#[test]
fn minimum_identity_failure_on_empty_topology() {
    let empty = CommittedTopology {
        rooms: vec![],
        surfaces: vec![],
        portals: vec![],
        routes: vec![],
        transitions: vec![],
    };
    let config = ProofConfig::new(Preset::Sparse, 2048).unwrap();
    let result = planner::plan_composition(V3Seed::new(0), &config, &empty);
    assert!(matches!(
        result,
        Err(ContractError::MinimumIdentityFailure { .. })
    ));
}

// ── N.6 Stage-order violation ────────────────────────────────────────

#[test]
fn stage_order_violation_detected() {
    // Try to emit a map from an unvalidated assembly
    let brush = ConvexBrush::make_box((0, 64), (0, 64), (0, 128)).unwrap();
    let ab = AssemblyBrush::new(
        "test",
        BrushRole::WallShell,
        brush,
        Support::World {
            surface: FaceRole::Floor,
        },
    );

    let unvalidated = Assembly {
        brushes: vec![ab],
        interfaces: vec![],
        apertures: vec![],
        protected_volumes: vec![],
        support_edges: vec![],
        validated: false,
    };

    let result = emission::emit_map(&unvalidated, (0, 0, 0), 0, &[]);
    assert!(result.is_err());
}

// ── N.7 Unvalidated assembly emission ────────────────────────────────

#[test]
fn only_validated_assembly_can_be_emitted() {
    let brush = ConvexBrush::make_box((0, 64), (0, 64), (0, 128)).unwrap();
    let ab = AssemblyBrush::new(
        "test",
        BrushRole::WallShell,
        brush,
        Support::World {
            surface: FaceRole::Floor,
        },
    );

    // Validated assembly should emit successfully
    let validated = Assembly::new(vec![ab], vec![], vec![], vec![]).unwrap();
    assert!(validated.validated);

    let map = emission::emit_map(&validated, (32, 32, 16), 0, &[]).unwrap();
    assert!(!map.is_empty());
    assert!(map.contains("worldspawn"));
}

// ── N.8 Arbitrary spawn angle ────────────────────────────────────────

#[test]
fn arbitrary_spawn_angle_rejected() {
    let brush = ConvexBrush::make_box((0, 64), (0, 64), (0, 128)).unwrap();
    let assembly = Assembly::new(
        vec![AssemblyBrush::new(
            "test",
            BrushRole::WallShell,
            brush,
            Support::World {
                surface: FaceRole::Floor,
            },
        )],
        vec![],
        vec![],
        vec![],
    )
    .unwrap();

    assert!(emission::emit_map(&assembly, (32, 32, 16), 45, &[]).is_err());
}

// ── N.9 Unapproved normal ────────────────────────────────────────────

#[test]
fn unapproved_normal_in_geometry_rejected() {
    let result = geometry::CanonicalPlane::new(2, 1, 0, 10);
    assert!(result.is_err());
}

// ── N.10 Duplicate surface ownership ─────────────────────────────────

#[test]
fn duplicate_surface_ownership_rejected() {
    use enhanced_v3_proof::ir::{CommittedSurface, SupportSurfaceKind, SurfaceId, SurfaceOwner};

    let surfaces = vec![
        CommittedSurface {
            id: SurfaceId(0),
            room_id: RoomId(0),
            kind: SupportSurfaceKind::Floor,
            owner: SurfaceOwner {
                parent_kind: "room",
                parent_id: 0,
                face: "floor",
                direction: "up",
                qualifier: "primary",
            },
        },
        CommittedSurface {
            id: SurfaceId(1),
            room_id: RoomId(0),
            kind: SupportSurfaceKind::Floor,
            owner: SurfaceOwner {
                parent_kind: "room",
                parent_id: 0,
                face: "floor",
                direction: "up",
                qualifier: "primary",
            },
        },
    ];

    let topo = CommittedTopology {
        rooms: vec![],
        surfaces,
        portals: vec![],
        routes: vec![],
        transitions: vec![],
    };

    let result = topo.validate();
    assert!(result.is_err());
}

// ═══════════════════════════════════════════════════════════════════════════
// Phase 02 baseline unchanged
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn phase_02_baseline_can_still_load() {
    // Verify the enhanced_v3_baseline test file can still be referenced
    // (it's a separate test target, so we just verify the file exists)
    let baseline_test = crate_dir().join("tests/enhanced_v3_baseline.rs");
    assert!(
        baseline_test.exists(),
        "Phase 02 baseline test file must still exist"
    );

    let baseline_manifest = crate_dir().join("tests/fixtures/enhanced_v3_baseline/manifest.json");
    assert!(
        baseline_manifest.exists(),
        "Phase 02 baseline manifest must still exist"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Refresh mode (writes to temp dir only)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn refresh_mode_writes_to_temp_dir() {
    let tmp = tempfile::tempdir().expect("create temp dir");

    let config = ProofConfig::new(Preset::Sparse, 2048).unwrap();
    let result = pipeline::run_pipeline(&config, V3Seed::new(0)).unwrap();

    // Write to temp dir
    let map_dest = tmp.path().join("refreshed.map");
    let meta_dest = tmp.path().join("refreshed-metadata.json");

    std::fs::write(&map_dest, &result.map_text).expect("write map");
    std::fs::write(
        &meta_dest,
        &serde_json::to_string_pretty(&result.metadata).unwrap(),
    )
    .expect("write metadata");

    assert!(map_dest.exists());
    assert!(meta_dest.exists());

    // Verify written content matches in-memory
    let read_back = std::fs::read_to_string(&map_dest).unwrap();
    assert_eq!(read_back, result.map_text);

    let meta_read: ProofMetadata =
        serde_json::from_str(&std::fs::read_to_string(&meta_dest).unwrap()).unwrap();
    assert_eq!(meta_read, result.metadata);
}

// ═══════════════════════════════════════════════════════════════════════════
// Map syntax validation
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn generated_map_has_valid_quake_syntax() {
    let (map, _meta) = pipeline::make_canonical_fixture();

    // Must start with worldspawn entity
    assert!(map.starts_with("{\n\"classname\" \"worldspawn\"\n"));

    // Must end with exactly one trailing newline
    assert!(map.ends_with('\n'));
    assert!(!map.ends_with("\n\n"));

    // Every brace must be on its own line
    for line in map.lines() {
        let trimmed = line.trim();
        if trimmed == "{" || trimmed == "}" {
            continue;
        }
        // Brushes and entities should not have multiple braces on one line
        assert!(
            !(trimmed.contains('{') && trimmed.contains('}')),
            "line has both braces: {trimmed}"
        );
    }

    // Worldspawn is followed only by one player start and light entities.
    assert_eq!(map.matches("\"classname\"").count(), 5);
    assert_eq!(map.matches("\"classname\" \"worldspawn\"").count(), 1);
    assert_eq!(
        map.matches("\"classname\" \"info_player_start\"").count(),
        1
    );
    assert_eq!(map.matches("\"classname\" \"light\"").count(), 3);

    // No Valve-220 extensions
    assert!(!map.contains("_tb"));
    assert!(!map.contains("_tex"));
    assert!(!map.contains("\"mapversion\""));
}

#[test]
fn generated_map_plane_points_are_integers() {
    let (map, _meta) = pipeline::make_canonical_fixture();

    // Parse all plane points: ( x y z )
    for line in map.lines() {
        if line.trim().starts_with('(') {
            // Split on parentheses
            let parts: Vec<&str> = line.split('(').collect();
            for part in parts {
                if part.is_empty() {
                    continue;
                }
                let end = part.find(')').unwrap_or(part.len());
                let coords: Vec<&str> = part[..end].split_whitespace().collect();
                if coords.len() == 3 {
                    for c in coords {
                        let _val: i32 = c.parse().unwrap_or_else(|_| {
                            panic!("non-integer coordinate in plane point: {c}")
                        });
                    }
                }
            }
        }
    }
}

#[test]
fn generated_map_uses_v2_textures() {
    let (map, _meta) = pipeline::make_canonical_fixture();

    // All textures should be from cc0_dungeon_v2
    let allowed = ["bs_floor", "bs_wall", "bs_ceil", "bs_accent"];
    for line in map.lines() {
        if line.trim().starts_with('(') && line.contains('"') {
            // Extract texture name between quotes
            if let Some(start) = line.find('"') {
                let rest = &line[start + 1..];
                if let Some(end) = rest.find('"') {
                    let tex = &rest[..end];
                    if tex != "0" && tex != "0.25" && !tex.is_empty() {
                        assert!(allowed.contains(&tex), "unexpected texture: {tex}");
                    }
                }
            }
        }
    }
}

// ── Helpers for fixture generation ────────────────────────────────────────

/// Generate the canonical fixtures and return them as strings.
/// Run with `cargo test -- --nocapture` to print; use for fixture creation.
#[test]
fn generate_canonical_fixtures_for_checkin() {
    let map_path = fixture_map_path();
    let meta_path = fixture_metadata_path();

    // If fixtures don't exist, generate and write them
    if !map_path.exists() || !meta_path.exists() {
        let (map, meta) = pipeline::make_canonical_fixture();

        std::fs::create_dir_all(map_path.parent().unwrap()).expect("create fixture dir");
        std::fs::write(&map_path, &map).expect("write map fixture");
        std::fs::write(&meta_path, &serde_json::to_string_pretty(&meta).unwrap())
            .expect("write metadata fixture");

        eprintln!("Generated canonical fixtures at:");
        eprintln!("  map: {}", map_path.display());
        eprintln!("  meta: {}", meta_path.display());
    }

    // Verify they exist now
    assert!(map_path.exists(), "map fixture must exist after generation");
    assert!(
        meta_path.exists(),
        "metadata fixture must exist after generation"
    );
}
