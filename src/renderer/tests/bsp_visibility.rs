//! Tests for BSP PVS-aware culling, camera leaf lookup, and corrupt-VIS fallback.

#![cfg(feature = "bsp")]

use bsp::geometry::{BatchKey, RenderBatch};
use bsp::lumps;
use bsp::visibility::{camera_leaf_index, PvsSet, PvsState};
use glam::Vec3;
use renderer::api::bsp::{filter_batches_by_pvs, pvs_should_disable, BspMountState};

// ── Camera leaf lookup tests ────────────────────────────────────────────

fn make_test_leaf(contents: i32, visofs: i32) -> lumps::Leaf {
    lumps::Leaf {
        contents,
        visofs,
        mins: [0i32; 3],
        maxs: [0i32; 3],
        mark_id: 0,
        mark_num: 0,
        ambient: [0u8; 4],
    }
}

fn make_test_node(plane_id: u32, front: i32, back: i32) -> lumps::Node {
    lumps::Node {
        plane_id,
        children: [front, back],
        mins: [0i32; 3],
        maxs: [0i32; 3],
        face_id: 0,
        face_num: 0,
    }
}

#[test]
fn camera_leaf_basic_traversal() {
    // Root splits at x=0. front (x>=0) → leaf 0, back (x<0) → leaf 1.
    let planes = vec![lumps::Plane {
        normal: Vec3::X,
        dist: 0.0,
        plane_type: 0,
    }];
    let nodes = vec![make_test_node(0, -1, -2)];
    let leaves = vec![
        make_test_leaf(0, 0), // leaf 0 (x >= 0)
        make_test_leaf(0, 0), // leaf 1 (x < 0)
    ];

    let cam = camera_leaf_index(&Vec3::new(10.0, 0.0, 0.0), &nodes, &leaves, &planes);
    assert_eq!(cam.leaf_index, 0);
    assert!(!cam.in_solid);

    let cam = camera_leaf_index(&Vec3::new(-10.0, 0.0, 0.0), &nodes, &leaves, &planes);
    assert_eq!(cam.leaf_index, 1);
    assert!(!cam.in_solid);
}

#[test]
fn camera_on_plane_defaults_front() {
    let planes = vec![lumps::Plane {
        normal: Vec3::X,
        dist: 0.0,
        plane_type: 0,
    }];
    let nodes = vec![make_test_node(0, -1, -2)];
    let leaves = vec![make_test_leaf(0, 0), make_test_leaf(0, 0)];

    let cam = camera_leaf_index(&Vec3::new(0.0, 0.0, 0.0), &nodes, &leaves, &planes);
    assert_eq!(cam.leaf_index, 0);
}

#[test]
fn camera_in_solid_leaf() {
    let planes = vec![lumps::Plane {
        normal: Vec3::X,
        dist: 0.0,
        plane_type: 0,
    }];
    let nodes = vec![make_test_node(0, -1, -2)];
    let leaves = vec![
        make_test_leaf(-2, 0), // solid
        make_test_leaf(0, 0),
    ];

    let cam = camera_leaf_index(&Vec3::new(10.0, 0.0, 0.0), &nodes, &leaves, &planes);
    assert!(cam.in_solid);
}

// ── PVS tests ───────────────────────────────────────────────────────────

#[test]
fn pvs_rle_decompression_all_visible() {
    let leaves = vec![make_test_leaf(0, 0); 16];
    let num_leaves = leaves.len() as u32;
    // Two 0xFF commands → all 16 leaves visible
    let vis_data = vec![0xFFu8, 0xFFu8];
    let state = PvsState::new(num_leaves, &vis_data);
    let leaf = make_test_leaf(0, 0);
    let pvs = state.decompress_for_leaf(0, &leaf, &vis_data);
    assert!(pvs.valid);
    for i in 0..16 {
        assert!(pvs.is_visible(i));
    }
}

#[test]
fn pvs_rle_zero_fill_all_invisible() {
    let leaves = vec![make_test_leaf(0, 0); 32];
    let num_leaves = leaves.len() as u32;
    // RLE: 0x00 (zero command) + 0x04 (4 × 8 = 32 bits) → all zero
    let vis_data = vec![0x00u8, 0x04u8];
    let state = PvsState::new(num_leaves, &vis_data);
    let leaf = make_test_leaf(0, 0);
    let pvs = state.decompress_for_leaf(0, &leaf, &vis_data);
    assert!(pvs.valid);
    for i in 0..32 {
        assert!(!pvs.is_visible(i));
    }
}

#[test]
fn pvs_empty_vis_disables_globally() {
    assert!(pvs_should_disable(&[]));
}

#[test]
fn pvs_nonempty_vis_enabled() {
    assert!(!pvs_should_disable(&[0xFF]));
}

#[test]
fn pvs_negative_visofs_is_conservative() {
    let state = PvsState::new(16, &[0xFF]);
    let leaf = make_test_leaf(0, -1);
    let pvs = state.decompress_for_leaf(0, &leaf, &[0xFF]);
    assert!(!pvs.valid); // conservative fallback is invalid
}

#[test]
fn pvs_truncated_stream_discards_partial_decode() {
    let state = PvsState::new(16, &[0x01]);
    let leaf = make_test_leaf(0, 0);
    let pvs = state.decompress_for_leaf(0, &leaf, &[0x01]);
    assert!(!pvs.valid);
    for leaf in 0..16 {
        assert!(
            pvs.is_visible(leaf),
            "leaf {leaf} should be conservative-visible"
        );
    }
}

#[test]
fn pvs_zero_run_overflow_discards_partial_decode() {
    let state = PvsState::new(8, &[0x00, 0x02]);
    let leaf = make_test_leaf(0, 0);
    let pvs = state.decompress_for_leaf(0, &leaf, &[0x00, 0x02]);
    assert!(!pvs.valid);
    for leaf in 0..8 {
        assert!(
            pvs.is_visible(leaf),
            "leaf {leaf} should be conservative-visible"
        );
    }
}

#[test]
fn pvs_conservative_all_visible() {
    let state = PvsState::new(8, &[]);
    let fallback = state.conservative_fallback(0);
    assert!(!fallback.valid);
    assert_eq!(fallback.bits.len(), 1);
    assert_eq!(fallback.bits[0], 0xFF);
}

// ── Batch filtering tests ───────────────────────────────────────────────

fn make_batch(leaf_signature: Vec<u32>, pvs_eligible: bool, is_inline: bool) -> RenderBatch {
    let model_index = if is_inline { 1 } else { 0 };
    RenderBatch {
        key: BatchKey {
            render_class: 0,
            material_identity: 0,
            lightmap_page: 0,
            style_ids: [0, 255, 255, 255],
            model_index,
        },
        leaf_signature,
        face_indices: vec![0],
        pvs_eligible,
        is_inline_model: is_inline,
        model_index,
    }
}

#[test]
fn filter_no_pvs_returns_all() {
    let batches = vec![
        make_batch(vec![0, 1], true, false),
        make_batch(vec![2, 3], true, false),
    ];
    let mount = BspMountState::new();
    let members = vec![vec![0u32, 1], vec![2, 3]];

    let result = filter_batches_by_pvs(&batches, &members, &mount);
    assert_eq!(result.len(), 2);
}

#[test]
fn filter_with_pvs_filters_correctly() {
    let mut mount = BspMountState::new();
    mount.active = true;
    // Create a PVS where only leaf 0 is visible.
    let mut bits = vec![0u8; 1]; // 8 leaves
    bits[0] = 0x01; // leaf 0 visible, rest not
    mount.current_pvs = Some(PvsSet {
        leaf_index: 0,
        bits,
        valid: true,
    });

    let batches = vec![
        make_batch(vec![0], true, false),    // visible (leaf 0 in PVS)
        make_batch(vec![1], true, false),    // NOT visible (leaf 1 not in PVS)
        make_batch(vec![0, 1], true, false), // visible (leaf 0 intersection)
    ];
    let members = vec![vec![0u32], vec![1], vec![0, 1]];

    let result = filter_batches_by_pvs(&batches, &members, &mount);
    assert_eq!(result.len(), 2); // batches 0 and 2 survive
}

#[test]
fn filter_inline_models_always_pass() {
    let mut mount = BspMountState::new();
    mount.active = true;
    // All leaves invisible → nothing should pass... except inline models.
    let bits = vec![0u8; 1];
    mount.current_pvs = Some(PvsSet {
        leaf_index: 0,
        bits,
        valid: true,
    });

    let batches = vec![
        make_batch(vec![0], true, false), // PVS-eligible → filtered out
        make_batch(vec![], false, true),  // inline model → always passes
    ];
    let members = vec![vec![0u32], vec![]];

    let result = filter_batches_by_pvs(&batches, &members, &mount);
    assert_eq!(result.len(), 1); // only the inline model survives
}

#[test]
fn filter_invalid_pvs_returns_all() {
    let mut mount = BspMountState::new();
    mount.active = true;
    mount.current_pvs = Some(PvsSet {
        leaf_index: 0,
        bits: vec![0xFF],
        valid: false, // invalid PVS
    });

    let batches = vec![
        make_batch(vec![0], true, false),
        make_batch(vec![1], true, false),
    ];
    let members = vec![vec![0u32], vec![1]];

    let result = filter_batches_by_pvs(&batches, &members, &mount);
    assert_eq!(result.len(), 2); // all pass (conservative)
}

#[test]
fn filter_empty_leaf_signature_is_conservative_visible() {
    let mut mount = BspMountState::new();
    mount.active = true;
    mount.current_pvs = Some(PvsSet {
        leaf_index: 0,
        bits: vec![0x00],
        valid: true,
    });
    let batches = vec![make_batch(vec![], true, false)];
    let members = vec![vec![]];

    let result = filter_batches_by_pvs(&batches, &members, &mount);
    assert_eq!(result.len(), 1);
}

// ── BspMountState tests ─────────────────────────────────────────────────

#[test]
fn mount_state_default_inactive() {
    let state = BspMountState::new();
    assert!(!state.active);
    assert!(!state.has_pvs);
    assert!(state.current_pvs.is_none());
    assert!(state.camera_leaf().is_none());
}

#[test]
fn mount_state_activate_deactivate() {
    let mut state = BspMountState::new();
    state.activate();
    assert!(state.active);

    state.deactivate();
    assert!(!state.active);
    assert!(state.current_pvs.is_none());
}

#[test]
fn update_pvs_with_empty_nodes_returns_none() {
    let mut state = BspMountState::new();
    state.has_pvs = true;
    state.nodes = vec![];
    state.activate();
    let result = state.update_pvs(Vec3::ZERO);
    assert!(result.is_none());
}

#[test]
fn update_pvs_solid_camera_returns_none() {
    // Build a BSP where the camera ends up in a solid leaf.
    let planes = vec![lumps::Plane {
        normal: Vec3::X,
        dist: 0.0,
        plane_type: 0,
    }];
    let nodes = vec![make_test_node(0, -1, -2)];
    let leaves = vec![
        make_test_leaf(-2, 0), // solid
        make_test_leaf(0, 0),
    ];

    let mut state = BspMountState::new();
    state.nodes = nodes;
    state.leaves = leaves;
    state.planes = planes;
    state.vis_data = vec![0xFF, 0xFF];
    state.has_pvs = true;
    state.num_leaves = 2;
    state.activate();

    // Camera at x=10 → leaf 0 (solid)
    let result = state.update_pvs(Vec3::new(10.0 * 0.0254, 0.0, 0.0));
    assert!(result.is_none());
}

// ── Leaf membership intersection tests ──────────────────────────────────

#[test]
fn batch_intersects_pvs_no_pvs_true() {
    let state = BspMountState::new();
    assert!(state.batch_intersects_pvs(&[0, 1]));
}

#[test]
fn batch_intersects_pvs_with_match() {
    let mut state = BspMountState::new();
    state.active = true;
    let mut bits = vec![0u8; 1];
    bits[0] = 0x01; // only leaf 0 visible
    state.current_pvs = Some(PvsSet {
        leaf_index: 0,
        bits,
        valid: true,
    });

    assert!(state.batch_intersects_pvs(&[0])); // leaf 0 in PVS
    assert!(!state.batch_intersects_pvs(&[1])); // leaf 1 not in PVS
    assert!(state.batch_intersects_pvs(&[0, 1])); // at least one in PVS
}

#[test]
fn leaf_in_pvs_no_pvs_true() {
    let state = BspMountState::new();
    assert!(state.leaf_in_pvs(42));
}
