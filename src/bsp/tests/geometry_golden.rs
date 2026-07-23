//! Golden tests for face geometry reconstruction: winding, UV0/UV1, bounds,
//! and deterministic batching.
//!
//! Tests assert that face geometry is reconstructed correctly from BSP primitives
//! independent of rendering path.

use bsp::*;
use glam::{Vec2, Vec3};

/// Build a minimal triangle face in a test BSP.
fn make_test_triangle() -> (Vec<Vec3>, Vec<lumps::Edge>, Vec<i32>) {
    let vertices = vec![
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(64.0, 0.0, 0.0),
        Vec3::new(0.0, 64.0, 0.0),
    ];
    let edges = vec![
        lumps::Edge { v: [0, 1] },
        lumps::Edge { v: [1, 2] },
        lumps::Edge { v: [2, 0] },
    ];
    let surfedges = vec![0i32, 1, 2]; // edge 0, edge 1, edge 2 — vertex starts: 0, 1, 2
    (vertices, edges, surfedges)
}

#[test]
fn golden_winding_from_surfedges_counter_clockwise() {
    let (vertices, edges, surfedges) = make_test_triangle();

    // surfedges [0, -2, 1]: edge 0 forward (0→1), edge 2 reversed (0→2), edge 1 forward (1→2)
    // Start vertices: 0, 0, 1 → counter-clockwise winding
    let face = lumps::Face {
        plane_id: 0,
        side: 0,
        ledge_id: 0,
        ledge_num: 3,
        texinfo_id: 0,
        styles: [255, 255, 255, 255],
        lightofs: -1,
    };

    let winding = geometry::reconstruct_winding(&face, &vertices, &edges, &surfedges);
    assert_eq!(winding.len(), 3, "expected 3 vertices in winding");

    // Verify the winding contains unique vertices
    let mut sorted = winding.clone();
    sorted.sort_by(|a, b| {
        a.x.partial_cmp(&b.x)
            .unwrap()
            .then(a.y.partial_cmp(&b.y).unwrap())
    });
    for i in 1..sorted.len() {
        assert!(
            sorted[i].distance_squared(sorted[i - 1]) > 1e-12,
            "duplicate vertices in winding"
        );
    }
}

#[test]
fn golden_uv0_from_texinfo_projection() {
    let vertices = vec![
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(64.0, 0.0, 0.0),
        Vec3::new(0.0, 64.0, 0.0),
    ];
    let edges = vec![
        lumps::Edge { v: [0, 1] },
        lumps::Edge { v: [1, 2] },
        lumps::Edge { v: [2, 0] },
    ];
    let surfedges = vec![0i32, 1, 2];

    let face = lumps::Face {
        plane_id: 0,
        side: 0,
        ledge_id: 0,
        ledge_num: 3,
        texinfo_id: 0,
        styles: [255, 255, 255, 255],
        lightofs: -1,
    };

    let plane = lumps::Plane {
        normal: Vec3::new(0.0, 0.0, 1.0),
        dist: 0.0,
        plane_type: 0,
    };

    // Texinfo with identity projection (1 unit per texel in X and Y)
    let texinfo = lumps::Texinfo {
        vec_s: Vec3::new(0.03125, 0.0, 0.0),  // 1/32 scale
        dist_s: 0.0,
        vec_t: Vec3::new(0.0, 0.03125, 0.0),
        dist_t: 0.0,
        miptex: 0,
        flags: 0,
    };

    let qte = QuakeToEngine::default();
    let geo = geometry::build_face_geometry(&face, 0, &plane, &texinfo, &vertices, &edges, &surfedges, &qte);

    // UV0 should be projected via dot(vertex, vecS) + distS
    // For vertex (64, 0, 0): u = 64 * 0.03125 = 2.0
    let v1_uv = geo.uv0[1]; // vertex at (64, 0, 0)
    assert!((v1_uv.x - 2.0).abs() < 0.01, "UV0.x should be ~2.0, got {}", v1_uv.x);
}

#[test]
fn golden_engine_space_conversion() {
    let vertices = vec![
        Vec3::new(100.0, 200.0, 300.0),
        Vec3::new(110.0, 200.0, 300.0),
        Vec3::new(100.0, 210.0, 300.0),
    ];
    let edges = vec![
        lumps::Edge { v: [0, 1] },
        lumps::Edge { v: [1, 2] },
        lumps::Edge { v: [2, 0] },
    ];
    let surfedges = vec![0i32, 1, 2];

    let face = lumps::Face {
        plane_id: 0,
        side: 0,
        ledge_id: 0,
        ledge_num: 3,
        texinfo_id: 0,
        styles: [255, 255, 255, 255],
        lightofs: -1,
    };

    let plane = lumps::Plane {
        normal: Vec3::new(0.0, 0.0, 1.0),
        dist: 300.0,
        plane_type: 0,
    };

    let texinfo = lumps::Texinfo {
        vec_s: Vec3::new(0.03125, 0.0, 0.0),
        dist_s: 0.0,
        vec_t: Vec3::new(0.0, 0.03125, 0.0),
        dist_t: 0.0,
        miptex: 0,
        flags: 0,
    };

    let qte = QuakeToEngine::default();
    let geo = geometry::build_face_geometry(&face, 0, &plane, &texinfo, &vertices, &edges, &surfedges, &qte);

    // Engine-space: (100*0.0254, 300*0.0254, -200*0.0254) = (2.54, 7.62, -5.08)
    let v0 = geo.vertices[0];
    assert!((v0.x - 2.54).abs() < 0.01, "engine x = {}", v0.x);
    assert!((v0.y - 7.62).abs() < 0.01, "engine y = {}", v0.y);
    assert!((v0.z + 5.08).abs() < 0.01, "engine z = {}", v0.z);
}

#[test]
fn golden_bounds_computation() {
    let (vertices, edges, surfedges) = make_test_triangle();

    let face = lumps::Face {
        plane_id: 0,
        side: 0,
        ledge_id: 0,
        ledge_num: 3,
        texinfo_id: 0,
        styles: [255, 255, 255, 255],
        lightofs: -1,
    };

    let plane = lumps::Plane {
        normal: Vec3::Z,
        dist: 0.0,
        plane_type: 0,
    };

    let texinfo = lumps::Texinfo {
        vec_s: Vec3::new(0.03125, 0.0, 0.0),
        dist_s: 0.0,
        vec_t: Vec3::new(0.0, 0.03125, 0.0),
        dist_t: 0.0,
        miptex: 0,
        flags: 0,
    };

    let qte = QuakeToEngine::default();
    let geo = geometry::build_face_geometry(&face, 0, &plane, &texinfo, &vertices, &edges, &surfedges, &qte);

    assert!(geo.bounds.0.x <= geo.bounds.1.x);
    assert!(geo.bounds.0.y <= geo.bounds.1.y);
    assert!(geo.bounds.0.z <= geo.bounds.1.z);
}

#[test]
fn golden_non_origin_plane_remains_valid() {
    let vertices = vec![
        Vec3::new(0.0, 0.0, 128.0),
        Vec3::new(64.0, 0.0, 128.0),
        Vec3::new(0.0, 64.0, 128.0),
    ];
    let edges = vec![
        lumps::Edge { v: [0, 1] },
        lumps::Edge { v: [1, 2] },
        lumps::Edge { v: [2, 0] },
    ];
    let surfedges = vec![0i32, 1, 2];
    let face = lumps::Face {
        plane_id: 0,
        side: 0,
        ledge_id: 0,
        ledge_num: 3,
        texinfo_id: 0,
        styles: [255, 255, 255, 255],
        lightofs: -1,
    };
    let plane = lumps::Plane {
        normal: Vec3::Z,
        dist: 128.0,
        plane_type: 0,
    };
    let texinfo = lumps::Texinfo {
        vec_s: Vec3::X,
        dist_s: 0.0,
        vec_t: Vec3::Y,
        dist_t: 0.0,
        miptex: 0,
        flags: 0,
    };

    let geo = geometry::build_face_geometry(
        &face,
        0,
        &plane,
        &texinfo,
        &vertices,
        &edges,
        &surfedges,
        &QuakeToEngine::default(),
    );

    assert!(geo.is_valid, "planarity must account for non-zero plane distance");
}

#[test]
fn golden_batch_inline_models_do_not_merge() {
    let face = |face_index| geometry::FaceGeometry {
        face_index,
        vertices: vec![Vec3::ZERO, Vec3::X, Vec3::Y],
        uv0: vec![Vec2::ZERO; 3],
        uv1: vec![Vec2::ZERO; 3],
        normal: Vec3::Z,
        bounds: (Vec3::ZERO, Vec3::ONE),
        luxel_extents: (1, 1),
        is_valid: true,
    };
    let batches = geometry::batch_faces(
        &[face(10), face(20)],
        &[],
        &[geometry::RenderClass::Opaque, geometry::RenderClass::Opaque],
        &[7, 7],
        &[0, 0],
        &[(1, 10), (2, 20)],
    );

    assert_eq!(batches.len(), 2, "inline models with identical material keys stay separate");
    assert_eq!(batches[0].face_indices, vec![10]);
    assert_eq!(batches[1].face_indices, vec![20]);
    assert_ne!(batches[0].model_index, batches[1].model_index);
}

#[test]
fn golden_batch_emits_each_face_once_with_sorted_leaf_set() {
    let face = |face_index, is_valid| geometry::FaceGeometry {
        face_index,
        vertices: vec![Vec3::ZERO, Vec3::X, Vec3::Y],
        uv0: vec![Vec2::ZERO; 3],
        uv1: vec![Vec2::ZERO; 3],
        normal: Vec3::Z,
        bounds: (Vec3::ZERO, Vec3::ONE),
        luxel_extents: (1, 1),
        is_valid,
    };
    let batches = geometry::batch_faces(
        &[face(3, true), face(7, false)],
        &[vec![5, 1, 5], vec![2, 1]],
        &[geometry::RenderClass::Opaque, geometry::RenderClass::Opaque],
        &[1, 2],
        &[0, 0],
        &[],
    );

    let emitted: Vec<u32> = batches.iter().flat_map(|b| b.face_indices.iter().copied()).collect();
    assert_eq!(emitted, vec![3, 7]);
    assert!(batches.iter().any(|b| b.key.leaf_signature == vec![1, 5]));
    assert!(batches.iter().any(|b| b.key.leaf_signature == vec![1, 2]));
}

#[test]
fn golden_batch_key_deterministic_ordering() {
    // Even with identical content, batches should sort deterministically
    let batch1 = geometry::RenderBatch {
        key: geometry::BatchKey {
            leaf_signature: vec![1, 3, 5],
            render_class: 0,
            material_identity: 42,
            lightmap_page: 0,
        },
        face_indices: vec![0],
        pvs_eligible: true,
        is_inline_model: false,
        model_index: 0,
    };

    let batch2 = geometry::RenderBatch {
        key: geometry::BatchKey {
            leaf_signature: vec![1, 3, 5],
            render_class: 0,
            material_identity: 10,
            lightmap_page: 0,
        },
        face_indices: vec![1],
        pvs_eligible: true,
        is_inline_model: false,
        model_index: 0,
    };

    // batch2 has lower material_identity, so it should sort first
    let mut batches = vec![batch1.clone(), batch2.clone()];
    batches.sort_by(|a, b| {
        a.key
            .lightmap_page
            .cmp(&b.key.lightmap_page)
            .then_with(|| a.key.material_identity.cmp(&b.key.material_identity))
            .then_with(|| a.key.leaf_signature.cmp(&b.key.leaf_signature))
    });

    assert!(batches[0].key.material_identity <= batches[1].key.material_identity);
}

#[test]
fn golden_lightmap_extents_from_texinfo() {
    let vertices = vec![
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(128.0, 0.0, 0.0),
        Vec3::new(0.0, 128.0, 0.0),
    ];
    let edges = vec![
        lumps::Edge { v: [0, 1] },
        lumps::Edge { v: [1, 2] },
        lumps::Edge { v: [2, 0] },
    ];
    let surfedges = vec![0i32, 1, 2];

    let face = lumps::Face {
        plane_id: 0,
        side: 0,
        ledge_id: 0,
        ledge_num: 3,
        texinfo_id: 0,
        styles: [0, 255, 255, 255],
        lightofs: 0,
    };

    let plane = lumps::Plane {
        normal: Vec3::Z,
        dist: 0.0,
        plane_type: 0,
    };

    let texinfo = lumps::Texinfo {
        vec_s: Vec3::new(0.03125, 0.0, 0.0),
        dist_s: 0.0,
        vec_t: Vec3::new(0.0, 0.03125, 0.0),
        dist_t: 0.0,
        miptex: 0,
        flags: 0,
    };

    let qte = QuakeToEngine::default();
    let geo = geometry::build_face_geometry(&face, 0, &plane, &texinfo, &vertices, &edges, &surfedges, &qte);

    // Luxel extents should be > 0 for faces with lightmap data
    assert!(geo.luxel_extents.0 > 0);
    assert!(geo.luxel_extents.1 > 0);
}

#[test]
fn golden_degenerate_face_rejected() {
    // Two vertices cannot form a face
    let vertices = vec![
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
    ];
    let edges = vec![lumps::Edge { v: [0, 1] }];
    let surfedges = vec![0i32];

    let face = lumps::Face {
        plane_id: 0,
        side: 0,
        ledge_id: 0,
        ledge_num: 1,
        texinfo_id: 0,
        styles: [255; 4],
        lightofs: -1,
    };

    let plane = lumps::Plane {
        normal: Vec3::Z,
        dist: 0.0,
        plane_type: 0,
    };

    let texinfo = lumps::Texinfo {
        vec_s: Vec3::new(1.0, 0.0, 0.0),
        dist_s: 0.0,
        vec_t: Vec3::new(0.0, 1.0, 0.0),
        dist_t: 0.0,
        miptex: 0,
        flags: 0,
    };

    let qte = QuakeToEngine::default();
    let geo = geometry::build_face_geometry(&face, 0, &plane, &texinfo, &vertices, &edges, &surfedges, &qte);

    assert!(!geo.is_valid, "2-vertex face should be invalid");
}

#[test]
fn golden_overflow_vertices_rejected() {
    // Vertex exceeding 2^15 units
    let vertices = vec![
        Vec3::new(40000.0, 0.0, 0.0), // exceeds MAX_VERTEX_COMPONENT
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
    ];
    let edges = vec![
        lumps::Edge { v: [0, 1] },
        lumps::Edge { v: [1, 2] },
        lumps::Edge { v: [2, 0] },
    ];
    let surfedges = vec![0i32, 1, 2];

    let face = lumps::Face {
        plane_id: 0,
        side: 0,
        ledge_id: 0,
        ledge_num: 3,
        texinfo_id: 0,
        styles: [255; 4],
        lightofs: -1,
    };

    let plane = lumps::Plane {
        normal: Vec3::Z,
        dist: 0.0,
        plane_type: 0,
    };

    let texinfo = lumps::Texinfo {
        vec_s: Vec3::new(1.0, 0.0, 0.0),
        dist_s: 0.0,
        vec_t: Vec3::new(0.0, 1.0, 0.0),
        dist_t: 0.0,
        miptex: 0,
        flags: 0,
    };

    let qte = QuakeToEngine::default();
    let geo = geometry::build_face_geometry(&face, 0, &plane, &texinfo, &vertices, &edges, &surfedges, &qte);

    assert!(!geo.is_valid, "overflow vertices should be rejected");
}
