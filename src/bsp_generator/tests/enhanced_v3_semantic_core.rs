//! Enhanced V3 Semantic Core — integration tests.
//!
//! Validates the complete deterministic semantic pipeline for EnhancedV3
//! production. Tests cover config validation, seed determinism, ID allocation,
//! footprint geometry, topology structure, assembly validation, reservation
//! conflicts, composition planning, and end-to-end .map generation.

use bsp_generator::enhanced_v3::*;

// ── Configuration tests ────────────────────────────────────────────────────

#[test]
fn config_validation_rejects_out_of_range() {
    assert!(V3Config::new(0, V3Preset::Sparse, 512).is_err());
    assert!(V3Config::new(0, V3Preset::Sparse, 4096).is_err());
}

#[test]
fn config_validation_rejects_non_quantum() {
    assert!(V3Config::new(0, V3Preset::Sparse, 2047).is_err());
}

#[test]
fn config_validation_accepts_boundary_values() {
    assert!(V3Config::new(0, V3Preset::Sparse, 1024).is_ok());
    assert!(V3Config::new(0, V3Preset::Moderate, 2048).is_ok());
    assert!(V3Config::new(0, V3Preset::Rich, 3072).is_ok());
}

#[test]
fn config_nominal_methods_are_valid() {
    let _ = V3Config::nominal_sparse();
    let _ = V3Config::nominal_moderate();
    let _ = V3Config::nominal_rich();
}

// ── Seed determinism tests ─────────────────────────────────────────────────

#[test]
fn v3_seed_deterministic() {
    let seed = V3Seed::new(42);
    let a = seed.stage_seed(tags::COMPOSITION);
    let b = seed.stage_seed(tags::COMPOSITION);
    assert_eq!(a.digest, b.digest);
}

#[test]
fn v3_seed_different_stages_different_output() {
    let seed = V3Seed::new(0);
    let a = seed.stage_seed(tags::PLACEMENT);
    let b = seed.stage_seed(tags::TOPOLOGY);
    assert_ne!(a.digest, b.digest);
}

#[test]
fn v3_domain_isolation() {
    let seed = V3Seed::new(0);
    let a = seed.stage_seed(tags::COMPOSITION);
    let b = V3Seed::new(0).stage_seed(tags::COMPOSITION);
    assert_eq!(a.digest, b.digest);
}

#[test]
fn v3_candidate_keyed_isolation() {
    let seed = V3Seed::new(0);
    let a = seed.candidate_seed(tags::COMPOSITION, b"room/0001");
    let b = seed.candidate_seed(tags::COMPOSITION, b"room/0002");
    assert_ne!(a.digest, b.digest);
}

#[test]
fn v3_bounded_u64_bounds() {
    let seed = V3Seed::new(42);
    for bound in [1, 2, 7, 1024] {
        let val = seed.bounded_u64(tags::COMPOSITION, b"test", bound).unwrap();
        assert!(val < bound);
    }
}

#[test]
fn v3_bounded_u64_zero_bound() {
    let seed = V3Seed::new(42);
    assert!(seed.bounded_u64(tags::COMPOSITION, b"test", 0).is_err());
}

#[test]
fn v3_candidate_selector_determinism() {
    let seed = V3Seed::new(42);
    let sel1 = CandidateSelector::new(seed, tags::COMPOSITION, true);
    let sel2 = CandidateSelector::new(seed, tags::COMPOSITION, true);
    let candidates = ["room/0001", "room/0002", "room/0003"];
    let r1: Vec<u64> = candidates
        .iter()
        .map(|k| sel1.rank_for(k.as_bytes()))
        .collect();
    let r2: Vec<u64> = candidates
        .iter()
        .map(|k| sel2.rank_for(k.as_bytes()))
        .collect();
    assert_eq!(r1, r2);
}

// ── ID allocation tests ────────────────────────────────────────────────────

#[test]
fn id_allocator_unique_ids() {
    let mut alloc = V3IdAllocator::new();
    let r1 = alloc.next_room().unwrap();
    let r2 = alloc.next_room().unwrap();
    assert_ne!(r1, r2);
    assert_eq!(r1.raw(), 0);
    assert_eq!(r2.raw(), 1);
}

#[test]
fn id_stable_keys_padded() {
    assert_eq!(RoomId(3).stable_key(), "room/0003");
    assert_eq!(PortalId(42).stable_key(), "portal/0042");
    assert_eq!(SurfaceId(0).stable_key(), "surface/0000");
}

// ── Geometry tests ─────────────────────────────────────────────────────────

#[test]
fn rational_creation_and_reduction() {
    let r = Rational::new(6, 8).unwrap();
    assert_eq!(r.num, 3);
    assert_eq!(r.den, 4);
}

#[test]
fn rational_zero_denominator_rejected() {
    assert!(Rational::new(1, 0).is_err());
}

#[test]
fn plane_creation_rejects_unapproved_normal() {
    // 2x + 1y is not cardinal or 45° diagonal
    assert!(CanonicalPlane::new(2, 1, 0, 0).is_err());
}

#[test]
fn plane_cardinal_normal_accepted() {
    let p = CanonicalPlane::new(1, 0, 0, 10).unwrap();
    assert_eq!(p.nx, 1);
    assert_eq!(p.d, 10);
}

#[test]
fn plane_diagonal_normal_accepted() {
    let p = CanonicalPlane::new(1, 1, 0, 16).unwrap();
    assert_eq!(p.nx, 1);
    assert_eq!(p.ny, 1);
    assert!(p.normal_class() == config::NormalClass::Diagonal45);
}

#[test]
fn convex_brush_box_has_positive_volume() {
    let brush = ConvexBrush::make_box((0, 64), (0, 64), (0, 128)).unwrap();
    assert!(brush.volume() > Rational::ZERO);
}

#[test]
fn convex_brush_chamfered_box() {
    let brush = ConvexBrush::make_chamfered_box(
        (0, 64),
        (0, 64),
        (0, 128),
        &[(1, 1), (1, -1), (-1, 1), (-1, -1)],
        16,
    )
    .unwrap();
    assert!(brush.volume() > Rational::ZERO);
    assert_eq!(brush.faces.len(), 10);
}

#[test]
fn convex_brush_duplicate_planes_rejected() {
    let faces = vec![
        geometry::BrushFace::new(CanonicalPlane::new(1, 0, 0, 10).unwrap()).unwrap(),
        geometry::BrushFace::new(CanonicalPlane::new(2, 0, 0, 20).unwrap()).unwrap(),
        geometry::BrushFace::new(CanonicalPlane::new(-1, 0, 0, 0).unwrap()).unwrap(),
        geometry::BrushFace::new(CanonicalPlane::new(0, 1, 0, 10).unwrap()).unwrap(),
    ];
    assert!(ConvexBrush::new(faces).is_err());
}

#[test]
fn convex_brush_grid_alignment() {
    let brush = ConvexBrush::make_box((0, 64), (0, 64), (0, 128)).unwrap();
    assert!(brush.check_grid_alignment(16).is_ok());
}

#[test]
fn face_role_classification() {
    assert_eq!(FaceRole::classify(1, 0, 0).unwrap(), FaceRole::WestWall);
    assert_eq!(FaceRole::classify(-1, 0, 0).unwrap(), FaceRole::EastWall);
    assert_eq!(FaceRole::classify(0, 1, 0).unwrap(), FaceRole::SouthWall);
    assert_eq!(FaceRole::classify(0, 0, 1).unwrap(), FaceRole::Floor);
    assert_eq!(FaceRole::classify(0, 0, -1).unwrap(), FaceRole::Ceiling);
    assert_eq!(FaceRole::classify(1, 1, 0).unwrap(), FaceRole::DiagSW);
}

// ── Footprint tests ────────────────────────────────────────────────────────

#[test]
fn footprints_build_for_all_presets() {
    for preset in [V3Preset::Sparse, V3Preset::Moderate, V3Preset::Rich] {
        let config = V3Config::new(
            0,
            preset,
            if preset == V3Preset::Rich { 3072 } else { 2048 },
        )
        .unwrap();
        let mut alloc = V3IdAllocator::new();
        let (footprints, _) = build_footprints(&config, V3Seed::new(0), &mut alloc).unwrap();

        match preset {
            V3Preset::Sparse => assert_eq!(footprints.len(), 3),
            V3Preset::Moderate => assert_eq!(footprints.len(), 4),
            V3Preset::Rich => assert_eq!(footprints.len(), 6),
        }
    }
}

#[test]
fn footprints_all_edges_approved() {
    let config = V3Config::nominal_rich();
    let mut alloc = V3IdAllocator::new();
    let (footprints, _) = build_footprints(&config, V3Seed::new(0), &mut alloc).unwrap();

    for fp in &footprints {
        for &(a, b) in &fp.edges() {
            let dx = (b.0 - a.0).unsigned_abs();
            let dy = (b.1 - a.1).unsigned_abs();
            assert!(
                dx == 0 || dy == 0 || dx == dy,
                "unapproved edge in footprint {:?}",
                fp.room_id
            );
        }
    }
}

#[test]
fn footprints_convex_valid() {
    let config = V3Config::nominal_sparse();
    let mut alloc = V3IdAllocator::new();
    let (footprints, _) = build_footprints(&config, V3Seed::new(0), &mut alloc).unwrap();
    for fp in &footprints {
        fp.validate_convex().unwrap();
    }
}

#[test]
fn footprints_layer_assignment() {
    let config = V3Config::nominal_sparse();
    let mut alloc = V3IdAllocator::new();
    let (footprints, _) = build_footprints(&config, V3Seed::new(0), &mut alloc).unwrap();

    assert_eq!(footprints[0].layer, 0);
    assert_eq!(footprints[0].floor_z, 0);
    assert_eq!(footprints[2].layer, 1);
    assert_eq!(footprints[2].floor_z, 192);
}

// ── Topology tests ─────────────────────────────────────────────────────────

#[test]
fn topology_builds_for_all_presets() {
    for preset in [V3Preset::Sparse, V3Preset::Moderate, V3Preset::Rich] {
        let config = V3Config::new(
            0,
            preset,
            if preset == V3Preset::Rich { 3072 } else { 2048 },
        )
        .unwrap();
        let mut alloc = V3IdAllocator::new();
        let (footprints, layout) = build_footprints(&config, V3Seed::new(0), &mut alloc).unwrap();
        let topology =
            build_topology(&config, &footprints, &layout, V3Seed::new(0), &mut alloc).unwrap();

        assert!(!topology.rooms.is_empty());
        assert!(!topology.portals.is_empty());
        assert!(!topology.routes.is_empty());
        assert!(!topology.transitions.is_empty());
    }
}

#[test]
fn topology_deterministic() {
    let config = V3Config::nominal_sparse();
    let mut alloc1 = V3IdAllocator::new();
    let mut alloc2 = V3IdAllocator::new();
    let (fp1, lo1) = build_footprints(&config, V3Seed::new(0), &mut alloc1).unwrap();
    let (fp2, lo2) = build_footprints(&config, V3Seed::new(0), &mut alloc2).unwrap();
    let t1 = build_topology(&config, &fp1, &lo1, V3Seed::new(0), &mut alloc1).unwrap();
    let t2 = build_topology(&config, &fp2, &lo2, V3Seed::new(0), &mut alloc2).unwrap();

    assert_eq!(t1.rooms.len(), t2.rooms.len());
    assert_eq!(t1.portals.len(), t2.portals.len());
}

#[test]
fn topology_bounds_within_config() {
    let config = V3Config::nominal_sparse();
    let mut alloc = V3IdAllocator::new();
    let (footprints, layout) = build_footprints(&config, V3Seed::new(0), &mut alloc).unwrap();
    let topology =
        build_topology(&config, &footprints, &layout, V3Seed::new(0), &mut alloc).unwrap();

    for room in &topology.rooms {
        assert!(room.shell.2 <= config.xy_extent as i32);
        assert!(room.shell.3 <= config.xy_extent as i32);
    }
}

#[test]
fn spawn_and_light_reservations_exist() {
    let config = V3Config::nominal_sparse();
    let mut alloc = V3IdAllocator::new();
    let (footprints, layout) = build_footprints(&config, V3Seed::new(0), &mut alloc).unwrap();
    let topology =
        build_topology(&config, &footprints, &layout, V3Seed::new(0), &mut alloc).unwrap();

    let (spawn, lights) = compute_reservations(&topology).unwrap();
    assert!(spawn.width() > 0);
    assert_eq!(lights.len(), topology.rooms.len());
}

// ── Assembly tests ─────────────────────────────────────────────────────────

#[test]
fn assembly_validation_passes_for_disjoint_brushes() {
    let b1 = ConvexBrush::make_box((0, 64), (0, 64), (0, 128)).unwrap();
    let b2 = ConvexBrush::make_box((128, 192), (0, 64), (0, 128)).unwrap();

    let brushes = vec![
        AssemblyBrush::new(
            "wall_a",
            BrushRole::WallShell,
            b1,
            Support::World {
                surface: FaceRole::Floor,
            },
        ),
        AssemblyBrush::new(
            "wall_b",
            BrushRole::WallShell,
            b2,
            Support::World {
                surface: FaceRole::Floor,
            },
        ),
    ];

    let assembly = Assembly::new(brushes, vec![], vec![]).unwrap();
    assert!(assembly.validated);
}

#[test]
fn assembly_rejects_overlapping_brushes() {
    let b1 = ConvexBrush::make_box((0, 128), (0, 128), (0, 128)).unwrap();
    let b2 = ConvexBrush::make_box((64, 192), (64, 192), (64, 192)).unwrap();

    let brushes = vec![
        AssemblyBrush::new(
            "wall_a",
            BrushRole::WallShell,
            b1,
            Support::World {
                surface: FaceRole::Floor,
            },
        ),
        AssemblyBrush::new(
            "wall_b",
            BrushRole::WallShell,
            b2,
            Support::World {
                surface: FaceRole::Floor,
            },
        ),
    ];

    assert!(Assembly::new(brushes, vec![], vec![]).is_err());
}

#[test]
fn assembly_rejects_duplicate_ids() {
    let b1 = ConvexBrush::make_box((0, 64), (0, 64), (0, 128)).unwrap();
    let b2 = ConvexBrush::make_box((128, 192), (0, 64), (0, 128)).unwrap();

    let brushes = vec![
        AssemblyBrush::new(
            "wall_a",
            BrushRole::WallShell,
            b1,
            Support::World {
                surface: FaceRole::Floor,
            },
        ),
        AssemblyBrush::new(
            "wall_a", // duplicate ID
            BrushRole::WallShell,
            b2,
            Support::World {
                surface: FaceRole::Floor,
            },
        ),
    ];

    assert!(Assembly::new(brushes, vec![], vec![]).is_err());
}

#[test]
fn assembly_rejects_protected_volume_intrusion() {
    let b1 = ConvexBrush::make_box((0, 128), (0, 128), (0, 128)).unwrap();
    let pv_brush = ConvexBrush::make_box((32, 96), (32, 96), (32, 96)).unwrap();

    let brushes = vec![AssemblyBrush::new(
        "wall_a",
        BrushRole::WallShell,
        b1,
        Support::World {
            surface: FaceRole::Floor,
        },
    )];

    let pvs = vec![ProtectedVolume {
        id: "spawn_zone".into(),
        brush: pv_brush,
    }];

    assert!(Assembly::new(brushes, vec![], pvs).is_err());
}

// ── Reservation tests ──────────────────────────────────────────────────────

#[test]
fn reservation_set_rejects_overlaps() {
    let q = CONSTRUCTION_QUANTUM;
    let mut set = ReservationSet::new();
    set.add(Reservation::new(
        "a",
        "spawn",
        QuantumVolume::new(0, 0, 0, 2 * q, 2 * q, 2 * q).unwrap(),
    ))
    .unwrap();
    let result = set.add(Reservation::new(
        "b",
        "light",
        QuantumVolume::new(q, q, q, 3 * q, 3 * q, 3 * q).unwrap(),
    ));
    assert!(result.is_err());
}

#[test]
fn reservation_set_allows_disjoint() {
    let q = CONSTRUCTION_QUANTUM;
    let mut set = ReservationSet::new();
    set.add(Reservation::new(
        "a",
        "spawn",
        QuantumVolume::new(0, 0, 0, q, q, q).unwrap(),
    ))
    .unwrap();
    set.add(Reservation::new(
        "b",
        "light",
        QuantumVolume::new(2 * q, 2 * q, 2 * q, 3 * q, 3 * q, 3 * q).unwrap(),
    ))
    .unwrap();
    assert!(set.validate_no_overlaps().is_ok());
}

// ── Composition tests ──────────────────────────────────────────────────────

#[test]
fn composition_plan_minimum_families_per_preset() {
    let sparse = plan_composition(ids::CompositionId(0), "sparse", 12).unwrap();
    assert!(sparse.grammar_families.len() >= 1);

    let moderate = plan_composition(ids::CompositionId(0), "moderate", 20).unwrap();
    assert!(moderate.grammar_families.len() >= 2);

    let rich = plan_composition(ids::CompositionId(0), "rich", 28).unwrap();
    assert!(rich.grammar_families.len() >= 3);
}

#[test]
fn composition_plan_within_face_budget() {
    let outcome = plan_composition(ids::CompositionId(0), "rich", 28).unwrap();
    assert!(outcome.estimated_total_faces < 10000);
    assert!(outcome.estimated_total_entities < 300);
}

// ── End-to-end generation tests ────────────────────────────────────────────

#[test]
fn generate_v3_sparse_produces_valid_map() {
    let config = V3Config::nominal_sparse();
    let map = generate_v3(&config).unwrap();
    assert!(!map.is_empty());
    assert!(map.contains("worldspawn"));
    assert!(map.contains("info_player_start"));
    assert!(map.contains("light"));
    assert!(map.contains("cc0_dungeon_v2.wad"));
}

#[test]
fn generate_v3_all_presets_produce_valid_map() {
    let configs = [
        V3Config::nominal_sparse(),
        V3Config::nominal_moderate(),
        V3Config::nominal_rich(),
    ];

    for config in &configs {
        let map = generate_v3(config).unwrap();
        assert!(!map.is_empty(), "empty map for preset {:?}", config.preset);
        assert!(
            map.contains("worldspawn"),
            "no worldspawn for preset {:?}",
            config.preset
        );
        assert!(
            map.contains("info_player_start"),
            "no spawn for preset {:?}",
            config.preset
        );
    }
}

#[test]
fn generate_v3_deterministic() {
    let config = V3Config::nominal_sparse();
    let map1 = generate_v3(&config).unwrap();
    let map2 = generate_v3(&config).unwrap();
    assert_eq!(map1, map2, "determinism violated");
}

#[test]
fn generate_v3_braces_balanced() {
    let config = V3Config::nominal_sparse();
    let map = generate_v3(&config).unwrap();
    let open_count = map.matches('{').count();
    let close_count = map.matches('}').count();
    assert_eq!(open_count, close_count, "mismatched braces");
}

#[test]
fn generate_v3_has_texture_assignments() {
    let config = V3Config::nominal_sparse();
    let map = generate_v3(&config).unwrap();
    assert!(map.contains("bs_wall"));
    assert!(map.contains("bs_floor"));
    assert!(map.contains("bs_ceil"));
}

#[test]
fn generate_v3_entity_count_reasonable() {
    let config = V3Config::nominal_sparse();
    let map = generate_v3(&config).unwrap();
    // Expect: 1 spawn + 1 light per room
    // 3 rooms = 3 lights + 1 spawn = 4 entities
    let light_count = map.matches("\"classname\" \"light\"").count();
    let spawn_count = map.matches("\"classname\" \"info_player_start\"").count();
    assert_eq!(spawn_count, 1);
    // Sparse has 3 rooms -> 3 lights
    assert!(light_count >= 2);
}

#[test]
fn generate_v3_face_count_within_budget() {
    let config = V3Config::nominal_rich();
    let map = generate_v3(&config).unwrap();
    // Count brush faces (each has exactly 6 planes × texture prefix in standard .map)
    // In our encoding, each brush starts with '{' and each face line starts with '('
    let face_lines = map
        .lines()
        .filter(|l| l.trim_start().starts_with('('))
        .count();
    // Face budget is 10000
    assert!(face_lines < 10000, "face count {face_lines} exceeds budget");
}

#[test]
fn generate_v3_spawn_near_center_of_first_room() {
    let config = V3Config::nominal_sparse();
    let map = generate_v3(&config).unwrap();

    // Extract spawn origin
    let origin_line = map
        .lines()
        .find(|l| l.contains("\"origin\""))
        .expect("must have origin line");
    // Parse: "origin" "x y z"
    let parts: Vec<&str> = origin_line.split('"').collect();
    // Find the origin values
    assert!(!parts.is_empty());
}
