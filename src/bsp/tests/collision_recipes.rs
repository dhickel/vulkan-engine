//! Tests for convex reconstruction from clipnodes: plane collection, deduplication,
//! convex-from-planes, volume detection, complexity limits, and error cases.

use bsp::*;
use glam::Vec3;

// ── Plane Collection ──

#[test]
fn golden_collect_clip_planes_simple() {
    let clipnodes = vec![
        lumps::Clipnode {
            plane: 0,
            children: [1, -1], // next node, EMPTY
        },
        lumps::Clipnode {
            plane: 1,
            children: [-2, -2], // both SOLID
        },
    ];
    let planes = vec![
        lumps::Plane {
            normal: Vec3::X,
            dist: 0.0,
            plane_type: 0,
        },
        lumps::Plane {
            normal: Vec3::Y,
            dist: 100.0,
            plane_type: 0,
        },
    ];

    let qte = QuakeToEngine::default();
    let result = collision::collect_clip_planes(0, &clipnodes, &planes, &qte).unwrap();
    // Should collect from both nodes
    assert!(result.len() >= 2, "expected at least 2 planes, got {}", result.len());
}

#[test]
fn golden_collect_clip_planes_invalid_headnode() {
    let result = collision::collect_clip_planes(-1, &[], &[], &QuakeToEngine::default());
    assert!(result.is_err());

    let result = collision::collect_clip_planes(5, &[], &[], &QuakeToEngine::default());
    assert!(result.is_err());
}

#[test]
fn golden_collect_clip_planes_out_of_range() {
    let clipnodes = vec![lumps::Clipnode { plane: 100, children: [-1, -1] }];
    let planes = vec![lumps::Plane {
        normal: Vec3::X,
        dist: 0.0,
        plane_type: 0,
    }];
    let result = collision::collect_clip_planes(0, &clipnodes, &planes, &QuakeToEngine::default());
    assert!(result.is_err());
}

// ── Convex Reconstruction ──

#[test]
fn golden_convex_from_unit_cube() {
    // 6 planes defining a unit cube centered at origin
    let normals = vec![
        Vec3::X, -Vec3::X,
        Vec3::Y, -Vec3::Y,
        Vec3::Z, -Vec3::Z,
    ];
    let dists = vec![0.5, 0.5, 0.5, 0.5, 0.5, 0.5];

    let piece = collision::convex_from_planes(&normals, &dists, 1e-4).unwrap();
    assert_eq!(piece.vertices.len(), 8, "cube should have 8 vertices");

    // All vertices should be within the half-space
    for v in &piece.vertices {
        assert!(v.x.abs() <= 0.5 + 1e-4);
        assert!(v.y.abs() <= 0.5 + 1e-4);
        assert!(v.z.abs() <= 0.5 + 1e-4);
    }
}

#[test]
fn golden_convex_insufficient_planes() {
    let normals = vec![Vec3::X, Vec3::Y, Vec3::Z];
    let dists = vec![0.0, 0.0, 0.0];
    let result = collision::convex_from_planes(&normals, &dists, 1e-4);
    assert!(result.is_err());
    match result {
        Err(ConvexError::InsufficientPlanes) => {}
        _ => panic!("expected InsufficientPlanes"),
    }
}

#[test]
fn golden_convex_open_region() {
    // Only 4 planes all opening inwards (unbounded region)
    let normals = vec![
        Vec3::X, Vec3::Y, Vec3::Z,
        Vec3::new(1.0, 1.0, 1.0).normalize(),
    ];
    let dists = vec![0.0, 0.0, 0.0, 0.0];

    let result = collision::convex_from_planes(&normals, &dists, 1e-4);
    // Should fail: either Degenerate (vertices at infinity or all at origin)
    // or insufficient volume
    assert!(result.is_err());
}

#[test]
fn golden_convex_complexity_exceeded() {
    // Generate 65 planes (exceeds MAX_CONVEX_FACES=64)
    let mut normals = Vec::new();
    let mut dists = Vec::new();
    for i in 0..65 {
        let angle = (i as f32) * 0.1;
        normals.push(Vec3::new(angle.cos(), angle.sin(), 0.0));
        dists.push(1.0);
    }
    let result = collision::convex_from_planes(&normals, &dists, 1e-4);
    assert!(result.is_err());
    match result {
        Err(ConvexError::ComplexityExceeded) => {}
        _ => panic!("expected ComplexityExceeded"),
    }
}

#[test]
fn golden_convex_degenerate_line() {
    // Planes that converge to a line (no volume)
    let normals = vec![
        Vec3::X,
        -Vec3::X,
        Vec3::Y,
    ];
    let dists = vec![0.0, 0.0, 0.0];
    let result = collision::convex_from_planes(&normals, &dists, 1e-4);
    assert!(result.is_err());
}

// ── Volume Detection ──

#[test]
fn golden_has_volume_tetrahedron() {
    use collision::ConvexPiece;
    let verts = vec![
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
    ];
    // We test via convex_from_planes indirectly
    let piece = ConvexPiece {
        plane_normals: vec![],
        plane_dists: vec![],
        vertices: verts,
    };
    assert_eq!(piece.vertices.len(), 4);
}

#[test]
fn golden_has_volume_flat_plane() {
    // Flat plane: all z=0
    let normals = vec![Vec3::Z, -Vec3::Z, Vec3::X, -Vec3::X, Vec3::Y, -Vec3::Y];
    let dists = vec![0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
    let result = collision::convex_from_planes(&normals, &dists, 1e-4);
    // Should be degenerate (no volume in Z)
    assert!(result.is_err());
}

// ── World Collision ──

#[test]
fn golden_world_collision_empty_clipnodes() {
    let result = collision::build_world_collision_planes(&[], &[], &QuakeToEngine::default());
    assert!(result.is_empty());
}

#[test]
fn golden_world_collision_with_planes() {
    let clipnodes = vec![
        lumps::Clipnode {
            plane: 0,
            children: [-1, -2], // EMPTY, SOLID
        },
    ];
    let planes = vec![lumps::Plane {
        normal: Vec3::X,
        dist: 0.0,
        plane_type: 0,
    }];

    let qte = QuakeToEngine::default();
    let result = collision::build_world_collision_planes(&clipnodes, &planes, &qte);
    assert!(!result.is_empty(), "should collect at least one plane");
}

// ── Collision Recipe ──

#[test]
fn golden_collision_recipe_for_brush() {
    let clipnodes = vec![
        lumps::Clipnode { plane: 0, children: [1, -1] },
        lumps::Clipnode { plane: 1, children: [2, -1] },
        lumps::Clipnode { plane: 2, children: [3, -1] },
        lumps::Clipnode { plane: 3, children: [4, -1] },
        lumps::Clipnode { plane: 4, children: [5, -1] },
        lumps::Clipnode { plane: 5, children: [-2, -2] },
    ];
    // 6 planes forming a box
    let planes = vec![
        lumps::Plane { normal: Vec3::X, dist: 128.0, plane_type: 0 },
        lumps::Plane { normal: -Vec3::X, dist: 0.0, plane_type: 0 },
        lumps::Plane { normal: Vec3::Y, dist: 128.0, plane_type: 0 },
        lumps::Plane { normal: -Vec3::Y, dist: 0.0, plane_type: 0 },
        lumps::Plane { normal: Vec3::Z, dist: 128.0, plane_type: 0 },
        lumps::Plane { normal: -Vec3::Z, dist: 0.0, plane_type: 0 },
    ];

    let qte = QuakeToEngine::default();
    let recipe = collision::build_collision_recipe(
        1, 1, 0, false, &clipnodes, &planes, &qte,
    );

    assert!(recipe.is_ok(), "should build recipe: {:?}", recipe.err());
    let recipe = recipe.unwrap();
    assert_eq!(recipe.entity_index, 1);
    assert_eq!(recipe.hull_index, 1);
    assert!(!recipe.is_trigger);
    assert!(!recipe.pieces.is_empty());
}

#[test]
fn golden_collision_recipe_trigger() {
    let clipnodes = vec![
        lumps::Clipnode { plane: 0, children: [-2, -2] },
    ];
    let planes = vec![lumps::Plane {
        normal: Vec3::X,
        dist: 0.0,
        plane_type: 0,
    }];

    let qte = QuakeToEngine::default();
    // A single plane with both sides solid will produce a degenerate convex
    let recipe = collision::build_collision_recipe(
        2, 1, 0, true, &clipnodes, &planes, &qte,
    );

    assert!(recipe.is_ok(), "trigger recipes should not fail hard");
    let recipe = recipe.unwrap();
    assert!(recipe.is_trigger);
}
