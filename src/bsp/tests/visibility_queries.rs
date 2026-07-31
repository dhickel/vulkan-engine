//! Tests for PVS decompression, camera leaf lookup, point_contents, and
//! stored-hull trace queries.
//!
//! Tests assert deterministic behavior, corrupt-fallback policy, and correct
//! query semantics against specifications.

use bsp::*;
use glam::Vec3;

// ── PVS Decompression ──

#[test]
fn golden_pvs_rle_all_visible() {
    let vis_data = vec![0xFFu8, 0xFFu8]; // 2 raw bytes → 16 leaves all visible
    let state = PvsState::new(16, &vis_data);
    assert!(!state.corrupt);

    let leaf = lumps::Leaf {
        contents: 0,
        visofs: 0,
        mins: [0; 3],
        maxs: [0; 3],
        mark_id: 0,
        mark_num: 0,
        ambient: [0; 4],
    };

    let pvs = state.decompress_for_leaf(0, &leaf, &vis_data);
    assert!(pvs.valid, "PVS should decompress successfully");
    assert_eq!(pvs.leaf_index, 0);

    // All 16 leaves should be visible
    for i in 0..16 {
        assert!(pvs.is_visible(i), "leaf {} should be visible", i);
    }
}

#[test]
fn golden_pvs_rle_all_invisible() {
    // Zero command: 0x00 0x02 → 2*8 = 16 zero bits
    let vis_data = vec![0x00u8, 0x02u8];
    let state = PvsState::new(16, &vis_data);

    let leaf = lumps::Leaf {
        contents: 0,
        visofs: 0,
        mins: [0; 3],
        maxs: [0; 3],
        mark_id: 0,
        mark_num: 0,
        ambient: [0; 4],
    };

    let pvs = state.decompress_for_leaf(0, &leaf, &vis_data);
    assert!(pvs.valid);

    for i in 0..16 {
        assert!(!pvs.is_visible(i), "leaf {} should NOT be visible", i);
    }
}

#[test]
fn golden_pvs_mixed_visibility() {
    // 0xFF (leaves 0-7 visible) then 0x00 0x01 (leaves 8-15 invisible)
    let vis_data = vec![0xFFu8, 0x00u8, 0x01u8];
    let state = PvsState::new(16, &vis_data);

    let leaf = lumps::Leaf {
        contents: 0,
        visofs: 0,
        mins: [0; 3],
        maxs: [0; 3],
        mark_id: 0,
        mark_num: 0,
        ambient: [0; 4],
    };

    let pvs = state.decompress_for_leaf(0, &leaf, &vis_data);
    assert!(pvs.valid);

    // Leaves 0-7 visible
    for i in 0..8 {
        assert!(pvs.is_visible(i));
    }
    // Leaves 8-15 invisible
    for i in 8..16 {
        assert!(!pvs.is_visible(i));
    }
}

#[test]
fn golden_pvs_corrupt_stream_fallback() {
    // Truncated RLE: 0x00 (zero command) but no count byte follows
    let vis_data = vec![0x00u8];
    let state = PvsState::new(16, &vis_data);

    let leaf = lumps::Leaf {
        contents: 0,
        visofs: 0,
        mins: [0; 3],
        maxs: [0; 3],
        mark_id: 0,
        mark_num: 0,
        ambient: [0; 4],
    };

    let pvs = state.decompress_for_leaf(0, &leaf, &vis_data);
    assert!(!pvs.valid, "corrupt PVS should produce invalid result");
    assert_eq!(
        pvs.bits,
        vec![0xFF, 0xFF],
        "corrupt PVS falls back to all-visible"
    );
}

#[test]
fn golden_pvs_partial_decode_discarded() {
    // One raw byte for a 16-leaf PVS is a truncated stream. The raw partial
    // result must be discarded rather than interpreted as leaves 8-15 hidden.
    let vis_data = vec![0x01u8];
    let state = PvsState::new(16, &vis_data);
    let leaf = lumps::Leaf {
        contents: 0,
        visofs: 0,
        mins: [0; 3],
        maxs: [0; 3],
        mark_id: 0,
        mark_num: 0,
        ambient: [0; 4],
    };

    let pvs = state.decompress_for_leaf(0, &leaf, &vis_data);
    assert!(!pvs.valid);
    assert_eq!(pvs.bits, vec![0xFF, 0xFF]);
}

#[test]
fn golden_pvs_empty_vis_data_is_corrupt() {
    let state = PvsState::new(32, &[]);
    assert!(state.corrupt, "empty VIS should be marked corrupt");

    let leaf = lumps::Leaf {
        contents: 0,
        visofs: 0,
        mins: [0; 3],
        maxs: [0; 3],
        mark_id: 0,
        mark_num: 0,
        ambient: [0; 4],
    };

    let pvs = state.decompress_for_leaf(0, &leaf, &[]);
    assert!(!pvs.valid, "should return invalid conservative fallback");
    assert_eq!(pvs.bits, vec![0xFF, 0xFF, 0xFF, 0xFF]);
}

#[test]
fn golden_pvs_conservative_all_visible() {
    let state = PvsState::new(24, &[]);
    let fallback = state.conservative_fallback(5);
    assert!(!fallback.valid);
    assert_eq!(fallback.leaf_index, 5);

    // 24 leaves → 3 bytes. All bits should be set.
    assert_eq!(fallback.bits.len(), 3);
    assert_eq!(fallback.bits[0], 0xFF);
    assert_eq!(fallback.bits[1], 0xFF);
    assert_eq!(fallback.bits[2], 0xFF);
}

#[test]
fn golden_pvs_odd_leaf_count_trims_last_byte() {
    let state = PvsState::new(10, &[]); // 10 leaves → 2 bytes, last 6 bits unused
    let fallback = state.conservative_fallback(0);
    assert_eq!(fallback.bits.len(), 2);
    // First byte fully set
    assert_eq!(fallback.bits[0], 0xFF);
    // Second byte: only lower 2 bits should be set (leaves 8 and 9)
    assert_eq!(fallback.bits[1] & 0x03, 0x03);
}

// ── Camera Leaf Lookup ──

#[test]
fn golden_camera_leaf_simple_node() {
    let planes = vec![lumps::Plane {
        normal: Vec3::X,
        dist: 0.0,
        plane_type: 0,
    }];
    let nodes = vec![lumps::Node {
        plane_id: 0,
        children: [-1, -2], // front = leaf 0, back = leaf 1
        mins: [0; 3],
        maxs: [0; 3],
        face_id: 0,
        face_num: 0,
    }];
    let leaves = vec![
        lumps::Leaf {
            contents: 0,
            visofs: 0,
            mins: [0; 3],
            maxs: [0; 3],
            mark_id: 0,
            mark_num: 0,
            ambient: [0; 4],
        },
        lumps::Leaf {
            contents: -2, // SOLID
            visofs: -1,
            mins: [0; 3],
            maxs: [0; 3],
            mark_id: 0,
            mark_num: 0,
            ambient: [0; 4],
        },
    ];

    let cam = visibility::camera_leaf_index(&Vec3::new(10.0, 0.0, 0.0), &nodes, &leaves, &planes);
    assert_eq!(cam.leaf_index, 0);
    assert!(!cam.in_solid);

    let cam = visibility::camera_leaf_index(&Vec3::new(-10.0, 0.0, 0.0), &nodes, &leaves, &planes);
    assert_eq!(cam.leaf_index, 1);
    assert!(cam.in_solid);
}

#[test]
fn golden_camera_on_plane_defaults_front() {
    let planes = vec![lumps::Plane {
        normal: Vec3::X,
        dist: 0.0,
        plane_type: 0,
    }];
    let nodes = vec![lumps::Node {
        plane_id: 0,
        children: [-1, -2],
        mins: [0; 3],
        maxs: [0; 3],
        face_id: 0,
        face_num: 0,
    }];
    let leaves = vec![
        lumps::Leaf {
            contents: 0,
            visofs: 0,
            mins: [0; 3],
            maxs: [0; 3],
            mark_id: 0,
            mark_num: 0,
            ambient: [0; 4],
        },
        lumps::Leaf {
            contents: 0,
            visofs: 0,
            mins: [0; 3],
            maxs: [0; 3],
            mark_id: 0,
            mark_num: 0,
            ambient: [0; 4],
        },
    ];

    // Exactly on plane → front child (default)
    let cam = visibility::camera_leaf_index(&Vec3::new(0.0, 0.0, 0.0), &nodes, &leaves, &planes);
    assert_eq!(cam.leaf_index, 0);
}

#[test]
fn golden_camera_empty_tree() {
    let cam = visibility::camera_leaf_index(&Vec3::new(1.0, 0.0, 0.0), &[], &[], &[]);
    assert!(cam.outside);
    assert!(cam.in_solid);
}

// ── Point Contents Queries ──

#[test]
fn golden_point_contents_empty() {
    let result = queries::point_contents(Vec3::ZERO, &[], &[], &[]);
    assert_eq!(result, PointContents::Empty);
}

#[test]
fn golden_point_contents_solid_and_empty() {
    let planes = vec![lumps::Plane {
        normal: Vec3::X,
        dist: 0.0,
        plane_type: 0,
    }];
    let nodes = vec![lumps::Node {
        plane_id: 0,
        children: [-1, -2],
        mins: [0; 3],
        maxs: [0; 3],
        face_id: 0,
        face_num: 0,
    }];
    let leaves = vec![
        lumps::Leaf {
            contents: -2, // SOLID
            visofs: 0,
            mins: [0; 3],
            maxs: [0; 3],
            mark_id: 0,
            mark_num: 0,
            ambient: [0; 4],
        },
        lumps::Leaf {
            contents: -1, // EMPTY
            visofs: 0,
            mins: [0; 3],
            maxs: [0; 3],
            mark_id: 0,
            mark_num: 0,
            ambient: [0; 4],
        },
    ];

    let result = queries::point_contents(Vec3::new(10.0, 0.0, 0.0), &nodes, &leaves, &planes);
    assert_eq!(result, PointContents::Solid);

    let result = queries::point_contents(Vec3::new(-10.0, 0.0, 0.0), &nodes, &leaves, &planes);
    assert_eq!(result, PointContents::Empty);
}

#[test]
fn golden_point_contents_all_liquid_types() {
    let planes = vec![lumps::Plane {
        normal: Vec3::X,
        dist: 0.0,
        plane_type: 0,
    }];
    let nodes = vec![lumps::Node {
        plane_id: 0,
        children: [-1, -2],
        mins: [0; 3],
        maxs: [0; 3],
        face_id: 0,
        face_num: 0,
    }];
    let leaves = vec![
        lumps::Leaf {
            contents: -3, // WATER
            visofs: 0,
            mins: [0; 3],
            maxs: [0; 3],
            mark_id: 0,
            mark_num: 0,
            ambient: [0; 4],
        },
        lumps::Leaf {
            contents: -5, // LAVA
            visofs: 0,
            mins: [0; 3],
            maxs: [0; 3],
            mark_id: 0,
            mark_num: 0,
            ambient: [0; 4],
        },
    ];

    let result = queries::point_contents(Vec3::new(1.0, 0.0, 0.0), &nodes, &leaves, &planes);
    assert_eq!(result, PointContents::Water);
    assert!(result.is_liquid());

    let result = queries::point_contents(Vec3::new(-1.0, 0.0, 0.0), &nodes, &leaves, &planes);
    assert_eq!(result, PointContents::Lava);
    assert!(result.is_liquid());
}

// ── Stored-Hull Trace ──

#[test]
fn golden_trace_empty_clipnodes_no_hit() {
    let result = queries::trace_stored_hull(
        Vec3::ZERO,
        Vec3::X,
        StoredHull::Point,
        &[],
        &[],
        -1,
        &QuakeToEngine::default(),
    );
    assert!(result.no_hit);
    assert_eq!(result.hit_fraction, 1.0);
}

#[test]
fn golden_trace_start_solid() {
    // Point starts inside a solid clip leaf
    let clipnodes = vec![lumps::Clipnode {
        plane: 0,
        children: [-2, -2], // both SOLID
    }];
    let planes = vec![lumps::Plane {
        normal: Vec3::X,
        dist: 0.0,
        plane_type: 0,
    }];

    let result = queries::trace_stored_hull(
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(10.0, 0.0, 0.0),
        StoredHull::Point,
        &clipnodes,
        &planes,
        0,
        &QuakeToEngine::default(),
    );
    assert!(result.starts_solid);
    assert_eq!(result.hit_fraction, 0.0);
}

#[test]
fn golden_trace_hit_plane() {
    // Clipnode tree: root splits at d=100, front=empty, back=solid
    let clipnodes = vec![lumps::Clipnode {
        plane: 0,
        children: [-1, -2], // front EMPTY, back SOLID
    }];
    let planes = vec![lumps::Plane {
        normal: Vec3::X,
        dist: 100.0,
        plane_type: 0,
    }];

    // Trace from x=0 to x=200, should hit at x=100
    let result = queries::trace_stored_hull(
        Vec3::ZERO,
        Vec3::new(200.0, 0.0, 0.0),
        StoredHull::Point,
        &clipnodes,
        &planes,
        0,
        &QuakeToEngine::default(),
    );
    // The trace implementation traces in quake space, so the hit fraction should be ~0.5
    assert!(result.hit_fraction < 1.0, "should hit before end");
}

#[test]
fn golden_hull_extents_different() {
    let qte = QuakeToEngine::default();

    let point_ext = StoredHull::Point.extents_engine(&qte);
    assert!(point_ext.length_squared() < 1e-6);

    let player_ext = StoredHull::Player.extents_engine(&qte);
    assert!(player_ext.x > 0.0);
    assert!(player_ext.z > 0.0);
    assert!((player_ext.y - 24.0 * qte.scale).abs() < 1e-6);
    assert!((player_ext.z - 16.0 * qte.scale).abs() < 1e-6);

    let monster_ext = StoredHull::LargeMonster.extents_engine(&qte);
    assert!(
        monster_ext.x > player_ext.x,
        "large monster wider than player"
    );
}

// ── Leaf Membership Maps ──

#[test]
fn golden_leaf_membership_deduplicated() {
    let leaves = vec![
        // Raw leaf 0 is the reserved solid leaf and has no PVS bit.
        lumps::Leaf {
            contents: -2,
            visofs: -1,
            mins: [0; 3],
            maxs: [0; 3],
            mark_id: 0,
            mark_num: 0,
            ambient: [0; 4],
        },
        // Raw leaf 1 / PVS bit 0 references faces [0, 1].
        lumps::Leaf {
            contents: 0,
            visofs: 0,
            mins: [0; 3],
            maxs: [0; 3],
            mark_id: 0,
            mark_num: 2,
            ambient: [0; 4],
        },
        // Raw leaf 2 / PVS bit 1 also references face 0.
        lumps::Leaf {
            contents: 0,
            visofs: 0,
            mins: [0; 3],
            maxs: [0; 3],
            mark_id: 2,
            mark_num: 1,
            ambient: [0; 4],
        },
    ];
    let markfaces = vec![0u32, 1, 0];

    let members = visibility::build_leaf_membership(&leaves, &markfaces);

    assert_eq!(members[0], vec![0, 1]);
    assert_eq!(members[1], vec![0]);
}

// ═══════════════════════════════════════════════════════════════════════
// Phase 06: Navigation Evidence Tests
// ═══════════════════════════════════════════════════════════════════════

mod navigation {
    use bsp::coords::QuakeToEngine;
    use bsp::*;
    use glam::Vec3;
    use std::path::Path;

    fn fixtures_dir() -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
    }

    fn compiled_dir() -> std::path::PathBuf {
        fixtures_dir().join("compiled")
    }

    fn read(path: &std::path::Path) -> Vec<u8> {
        std::fs::read(path).expect(&format!("failed to read {}", path.display()))
    }

    fn load_navigation_fixture() -> BspWorld {
        let bsp_data = read(&compiled_dir().join("dungeon-navigation-bsp2.bsp"));
        let options = LoadOptions {
            strict: true,
            source_identity: "dungeon-navigation-bsp2".into(),
            ..LoadOptions::default()
        };
        BspLoader::load(&bsp_data, &options).expect("strict load of dungeon-navigation-bsp2")
    }

    fn load_straight_junction_fixture() -> BspWorld {
        let bsp_data = read(&compiled_dir().join("dungeon-junction-straight-bsp2.bsp"));
        let options = LoadOptions {
            strict: true,
            source_identity: "dungeon-junction-straight-bsp2".into(),
            ..LoadOptions::default()
        };
        BspLoader::load(&bsp_data, &options).expect("strict load of dungeon-junction-straight-bsp2")
    }

    // ── EVIDENCE FINDING: Hull 0 and Hull 1 share headnodes ──────────

    #[test]
    fn nav_hull_dispute_resolved_headnodes_equal() {
        for (name, load) in [
            ("navigation", load_navigation_fixture as fn() -> BspWorld),
            ("straight-junction", load_straight_junction_fixture),
        ] {
            let world = load();
            let m0 = &world.models[0];
            assert_eq!(
                m0.headnode[0], m0.headnode[1],
                "{name}: hull 0 and hull 1 headnodes must be equal (compiler merges them)"
            );
        }
    }

    // ── NAV-SPAWN-VALID ───────────────────────────────────────────────

    #[test]
    fn nav_spawn_point_non_solid() {
        let world = load_navigation_fixture();
        let qte = QuakeToEngine::default();

        // info_player_start is at Quake (-128, 0, 0) in the navigation fixture
        let spawn_q = Vec3::new(-128.0, 0.0, 0.0);
        let spawn_eng = qte.position_vec3(spawn_q);
        let contents =
            queries::point_contents(spawn_eng, &world.nodes, &world.leaves, &world.planes);
        assert!(
            !contents.is_solid(),
            "spawn point must be non-solid, got {:?}",
            contents
        );
        assert!(
            contents.is_empty(),
            "spawn point must be in empty space, got {:?}",
            contents
        );
    }

    #[test]
    fn nav_spawn_valid_point_trace() {
        let world = load_navigation_fixture();
        let qte = QuakeToEngine::default();
        let spawn_q = Vec3::new(-128.0, 0.0, 0.0);
        let spawn_eng = qte.position_vec3(spawn_q);

        let result = queries::trace_line(
            spawn_eng,
            spawn_eng,
            StoredHull::Point,
            &world.clipnodes,
            &world.planes,
            &world.models,
            &qte,
        );
        assert!(
            !result.starts_solid,
            "spawn must not be start-solid for point hull"
        );
    }

    // ── NAV-STRAIGHT-LINE-TRAVERSAL ───────────────────────────────────

    #[test]
    fn nav_straight_line_move_east_across_room() {
        let world = load_navigation_fixture();
        let qte = QuakeToEngine::default();

        // Move from west side to east side, north of the central pillar
        let start_q = Vec3::new(-128.0, 100.0, 0.0);
        let end_q = Vec3::new(128.0, 100.0, 0.0);
        let start_eng = qte.position_vec3(start_q);
        let end_eng = qte.position_vec3(end_q);

        let result = queries::trace_line(
            start_eng,
            end_eng,
            StoredHull::Point,
            &world.clipnodes,
            &world.planes,
            &world.models,
            &qte,
        );
        assert!(!result.starts_solid);
        assert!(
            result.no_hit,
            "straight line north of pillar must complete without hitting, got fraction={}",
            result.hit_fraction
        );
    }

    #[test]
    fn nav_straight_line_blocked_by_pillar() {
        let world = load_navigation_fixture();
        let qte = QuakeToEngine::default();

        // Trace through the central pillar at origin
        let start_q = Vec3::new(-128.0, 0.0, 0.0);
        let end_q = Vec3::new(128.0, 0.0, 0.0);
        let start_eng = qte.position_vec3(start_q);
        let end_eng = qte.position_vec3(end_q);

        let result = queries::trace_line(
            start_eng,
            end_eng,
            StoredHull::Point,
            &world.clipnodes,
            &world.planes,
            &world.models,
            &qte,
        );
        assert!(!result.starts_solid);
        assert!(!result.no_hit, "trace through pillar must hit something");
        assert!(result.hit_fraction > 0.0 && result.hit_fraction < 1.0);
    }

    // ── NAV-STRAIGHT-JUNCTION-TRAVERSAL ───────────────────────────────

    #[test]
    fn nav_straight_junction_traverse_corridor() {
        let world = load_straight_junction_fixture();
        let qte = QuakeToEngine::default();

        // info_player_start at Quake (-192, 0, 0). Two rooms connected.
        let start_q = Vec3::new(-192.0, 0.0, 0.0);
        let end_q = Vec3::new(192.0, 0.0, 0.0);
        let start_eng = qte.position_vec3(start_q);
        let end_eng = qte.position_vec3(end_q);

        let result = queries::trace_line(
            start_eng,
            end_eng,
            StoredHull::Point,
            &world.clipnodes,
            &world.planes,
            &world.models,
            &qte,
        );
        assert!(!result.starts_solid);
        assert!(
            result.no_hit,
            "point trace from Room A to Room B must complete without hitting, got fraction={}",
            result.hit_fraction
        );
    }

    // ── NAV-CORNER-SLIDING ────────────────────────────────────────────

    #[test]
    fn nav_corner_sliding_around_pillar() {
        let world = load_navigation_fixture();
        let qte = QuakeToEngine::default();

        // Move south of pillar (clear path)
        let start_q = Vec3::new(-64.0, -64.0, 0.0);
        let end_q = Vec3::new(64.0, -64.0, 0.0);
        let start_eng = qte.position_vec3(start_q);
        let end_eng = qte.position_vec3(end_q);

        let result = queries::trace_line(
            start_eng,
            end_eng,
            StoredHull::Point,
            &world.clipnodes,
            &world.planes,
            &world.models,
            &qte,
        );
        assert!(!result.starts_solid);
        assert!(
            result.no_hit,
            "trace south of pillar must complete without hitting, got fraction={}",
            result.hit_fraction
        );
    }

    // ── NAV-WALL-HIT ──────────────────────────────────────────────────

    #[test]
    fn nav_wall_hit_west_wall() {
        let world = load_navigation_fixture();
        let qte = QuakeToEngine::default();

        let start_q = Vec3::new(-128.0, 0.0, 0.0);
        let end_q = Vec3::new(-300.0, 0.0, 0.0);
        let start_eng = qte.position_vec3(start_q);
        let end_eng = qte.position_vec3(end_q);

        let result = queries::trace_line(
            start_eng,
            end_eng,
            StoredHull::Point,
            &world.clipnodes,
            &world.planes,
            &world.models,
            &qte,
        );
        assert!(!result.starts_solid);
        assert!(!result.no_hit, "westward trace must hit wall");
        assert!(result.hit_fraction > 0.0 && result.hit_fraction < 1.0);
    }

    // ── NAV-ROUTE-REACHABILITY ────────────────────────────────────────

    #[test]
    fn nav_route_reachability_all_quadrants_reachable() {
        let world = load_navigation_fixture();
        let qte = QuakeToEngine::default();

        let start_q = Vec3::new(-128.0, 0.0, 0.0);
        let targets = [
            ("NE", Vec3::new(128.0, 128.0, 0.0)),
            ("NW", Vec3::new(-128.0, 128.0, 0.0)),
            ("SE", Vec3::new(128.0, -128.0, 0.0)),
            ("SW", Vec3::new(-128.0, -128.0, 0.0)),
        ];
        for (label, target_q) in &targets {
            let start_eng = qte.position_vec3(start_q);
            let end_eng = qte.position_vec3(*target_q);
            let result = queries::trace_line(
                start_eng,
                end_eng,
                StoredHull::Point,
                &world.clipnodes,
                &world.planes,
                &world.models,
                &qte,
            );
            assert!(
                result.no_hit,
                "spawn to {} quadrant must be reachable, got fraction={}",
                label, result.hit_fraction
            );
        }
    }

    // ── NAV-HULL2 ─────────────────────────────────────────────────────

    #[test]
    fn nav_hull2_navigable_around_pillar() {
        let world = load_navigation_fixture();
        let qte = QuakeToEngine::default();

        // Hull 2 around pillar north side: plenty of space
        let start_q = Vec3::new(-128.0, 100.0, 0.0);
        let end_q = Vec3::new(128.0, 100.0, 0.0);
        let start_eng = qte.position_vec3(start_q);
        let end_eng = qte.position_vec3(end_q);

        let result = queries::trace_line(
            start_eng,
            end_eng,
            StoredHull::LargeMonster,
            &world.clipnodes,
            &world.planes,
            &world.models,
            &qte,
        );
        assert!(!result.starts_solid);
        assert!(
            result.no_hit,
            "hull 2 trace north of pillar must complete, got fraction={}",
            result.hit_fraction
        );
    }

    // ── NAV-FIXTURE-INTEGRITY ─────────────────────────────────────────

    #[test]
    fn nav_fixture_strict_reload() {
        let world = load_navigation_fixture();
        assert!(world.num_models() > 0);
        assert!(world.num_leaves() > 0);
        assert!(!world.entities.is_empty());
        assert!(world.worldspawn().is_some());
        assert!(
            !world.clipnodes.is_empty(),
            "navigation fixture must have clipnodes"
        );
    }

    #[test]
    fn nav_fixture_has_info_player_start() {
        let world = load_navigation_fixture();
        let has_player_start = world
            .entities
            .iter()
            .any(|e| matches!(e.class, bsp::entities::EntityClass::SpawnMarker));
        assert!(
            has_player_start,
            "navigation fixture must have info_player_start"
        );
    }

    #[test]
    fn nav_fixture_hull_headnodes_valid() {
        let world = load_navigation_fixture();
        let model0 = &world.models[0];
        assert!(
            (model0.headnode[0] as usize) < world.clipnodes.len(),
            "hull 0 headnode must be within clipnodes"
        );
        assert!(
            (model0.headnode[2] as usize) < world.clipnodes.len() || model0.headnode[2] == 0,
            "hull 2 headnode must be within clipnodes"
        );
    }
}
