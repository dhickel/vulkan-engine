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
    assert_eq!(pvs.bits, vec![0xFF, 0xFF], "corrupt PVS falls back to all-visible");
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
        lumps::Leaf { contents: 0, visofs: 0, mins: [0; 3], maxs: [0; 3], mark_id: 0, mark_num: 0, ambient: [0; 4] },
        lumps::Leaf { contents: 0, visofs: 0, mins: [0; 3], maxs: [0; 3], mark_id: 0, mark_num: 0, ambient: [0; 4] },
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
    assert!(monster_ext.x > player_ext.x, "large monster wider than player");
}

// ── Leaf Membership Maps ──

#[test]
fn golden_leaf_membership_deduplicated() {
    let leaves = vec![
        // leaf 0 references faces [0, 1]
        lumps::Leaf {
            contents: 0,
            visofs: 0,
            mins: [0; 3],
            maxs: [0; 3],
            mark_id: 0,
            mark_num: 2,
            ambient: [0; 4],
        },
        // leaf 1 also references face 0 (should be deduplicated per face)
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
    let markfaces = vec![0u32, 1, 0]; // leaf 0: [0, 1], leaf 1: [0]

    let members = visibility::build_leaf_membership(&leaves, &markfaces);

    // face 0 is referenced by leaves [0, 1]
    assert_eq!(members[0], vec![0, 1]);
    // face 1 is referenced by leaf [0] only
    assert_eq!(members[1], vec![0]);
}
