//! Brush-construction and release-validation tests.
//!
//! Legacy L/T/X/portal helpers are intentionally absent from this integration
//! test's imports: production uses the corridor open-cell union and those
//! helpers are not public API.

use bsp_generator::junction::make_brush;
use bsp_generator::{validate_all_brushes, validate_brush, Brush, BrushFace, CONSTRUCTION_QUANTUM};

#[test]
fn make_brush_produces_canonical_six_face_box() {
    let brush = make_brush((0, 0, 0), (64, 64, 128), "stone_wall");
    assert_eq!(brush.faces.len(), 6);
    assert!(brush.faces.iter().all(|face| face.texture == "stone_wall"));
    assert!(validate_brush(&brush, 0).is_ok());
}

#[test]
fn make_brush_plane_points_are_quantum_aligned() {
    let brush = make_brush((16, 16, 0), (80, 80, 96), "stone_wall");
    let quantum = CONSTRUCTION_QUANTUM as i32;
    for face in &brush.faces {
        for &(x, y, z) in &face.plane_points {
            assert_eq!(x.rem_euclid(quantum), 0);
            assert_eq!(y.rem_euclid(quantum), 0);
            assert_eq!(z.rem_euclid(quantum), 0);
        }
    }
}

#[test]
fn validate_brush_rejects_empty_or_unknown_texture_identity() {
    for texture in ["", "unapproved_texture"] {
        let brush = make_brush((0, 0, 0), (64, 64, 128), texture);
        let error = validate_brush(&brush, 0).unwrap_err();
        assert!(
            error.to_string().contains("unsupported texture identity"),
            "{error}"
        );
    }
}

#[test]
fn validate_brush_rejects_fewer_than_six_faces() {
    let mut brush = make_brush((0, 0, 0), (64, 64, 128), "stone_wall");
    brush.faces.truncate(5);
    let error = validate_brush(&brush, 0).unwrap_err();
    assert!(error.to_string().contains("has 5 faces"), "{error}");
}

#[test]
fn validate_brush_rejects_collinear_face_points() {
    let mut brush = make_brush((0, 0, 0), (64, 64, 128), "stone_wall");
    brush.faces[0] = BrushFace {
        plane_points: [(0, 0, 0), (32, 0, 0), (64, 0, 0)],
        texture: "stone_wall".to_string(),
    };
    let error = validate_brush(&brush, 0).unwrap_err();
    assert!(error.to_string().contains("collinear"), "{error}");
}

#[test]
fn validate_brush_rejects_non_positive_volume() {
    let flat_face = BrushFace {
        plane_points: [(0, 0, 0), (0, 64, 0), (0, 0, 128)],
        texture: "stone_wall".to_string(),
    };
    let brush = Brush {
        faces: vec![flat_face; 6],
    };
    let error = validate_brush(&brush, 0).unwrap_err();
    assert!(
        error.to_string().contains("non-positive volume")
            || error.to_string().contains("canonical box"),
        "{error}"
    );
}

#[test]
fn validate_brush_rejects_non_quantum_aligned_plane() {
    let mut brush = make_brush((0, 0, 0), (64, 64, 128), "stone_wall");
    brush.faces[5].plane_points = [(63, 64, 0), (63, 0, 0), (63, 0, 128)];
    let error = validate_brush(&brush, 0).unwrap_err();
    assert!(error.to_string().contains("not quantum-aligned"), "{error}");
}

#[test]
fn validate_brush_rejects_reversed_face_orientation() {
    let mut brush = make_brush((0, 0, 0), (64, 64, 128), "stone_wall");
    brush.faces[0].plane_points.swap(1, 2);
    let error = validate_brush(&brush, 0).unwrap_err();
    assert!(error
        .to_string()
        .contains("canonical box plane/order/orientation"));
}

#[test]
fn validate_brush_rejects_duplicate_plane_even_with_positive_aabb() {
    let mut brush = make_brush((0, 0, 0), (64, 64, 128), "stone_wall");
    brush.faces[5] = brush.faces[0].clone();
    let error = validate_brush(&brush, 0).unwrap_err();
    assert!(error
        .to_string()
        .contains("canonical box plane/order/orientation"));
}

#[test]
fn validate_brush_rejects_mixed_face_textures() {
    let mut brush = make_brush((0, 0, 0), (64, 64, 128), "stone_wall");
    brush.faces[1].texture = "stone_floor".to_string();
    let error = validate_brush(&brush, 0).unwrap_err();
    assert!(error.to_string().contains("differs from box texture"));
}

#[test]
fn validate_all_brushes_returns_the_first_invalid_index() {
    let good = make_brush((0, 0, 0), (64, 64, 128), "stone_wall");
    let bad = make_brush((80, 0, 0), (144, 64, 128), "");
    let error = validate_all_brushes(&[good, bad]).unwrap_err();
    assert!(error.to_string().contains("brush 1"), "{error}");
}

#[test]
fn validate_all_brushes_accepts_empty_slice() {
    assert!(validate_all_brushes(&[]).is_ok());
}
