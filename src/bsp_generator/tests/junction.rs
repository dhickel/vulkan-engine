//! Integration tests for junction geometry.
//!
//! Validates that the release brush validator catches malformed brushes,
//! and that `make_brush` produces valid, well-formed output.
//! Legacy L/T/X/portal diagnostics are tested through their `pub(crate)`
//! path via `bsp_generator::junction`.

use bsp_generator::junction::{
    build_junction_closures, build_l_junction, build_room_portal, build_t_junction,
    build_x_junction, make_brush,
};
use bsp_generator::{validate_all_brushes, validate_brush, Brush, Corridor, RoomIntent, CONSTRUCTION_QUANTUM};

fn corridor_h(x: i32, y: i32, z: i32, len: i32) -> Corridor {
    Corridor {
        start: (x, y, z),
        end: (x + len, y, z),
        width: 64,
        height: 80,
    }
}

fn corridor_v(x: i32, y: i32, z: i32, len: i32) -> Corridor {
    Corridor {
        start: (x, y, z),
        end: (x, y + len, z),
        width: 64,
        height: 80,
    }
}

fn room(x: i32, y: i32, z: i32, dx: u32, dy: u32, dz: u32) -> RoomIntent {
    RoomIntent {
        position: (x, y, z),
        dimensions: (dx, dy, dz),
    }
}

fn assert_brush_faces_are_non_degenerate(brush: &Brush) {
    assert_eq!(brush.faces.len(), 6, "brush must have exactly 6 faces");
    for face in &brush.faces {
        let [p0, p1, p2] = face.plane_points;
        let v1 = (p1.0 - p0.0, p1.1 - p0.1, p1.2 - p0.2);
        let v2 = (p2.0 - p0.0, p2.1 - p0.1, p2.2 - p0.2);
        let cross = (
            v1.1 * v2.2 - v1.2 * v2.1,
            v1.2 * v2.0 - v1.0 * v2.2,
            v1.0 * v2.1 - v1.1 * v2.0,
        );
        assert_ne!(
            cross,
            (0, 0, 0),
            "face has degenerate plane points: {:?}",
            face.plane_points
        );
    }
}

fn brush_z_range(brush: &Brush) -> (i32, i32) {
    let mut min_z = i32::MAX;
    let mut max_z = i32::MIN;
    for face in &brush.faces {
        for &(_, _, z) in &face.plane_points {
            min_z = min_z.min(z);
            max_z = max_z.max(z);
        }
    }
    (min_z, max_z)
}

// ── make_brush ────────────────────────────────────────────────────────────

#[test]
fn make_brush_produces_six_faces() {
    let brush = make_brush((0, 0, 0), (64, 64, 128), "wall_tex");
    assert_eq!(brush.faces.len(), 6, "brush must have exactly 6 faces");
}

#[test]
fn make_brush_faces_use_correct_texture() {
    let brush = make_brush((0, 0, 0), (64, 64, 128), "brick");
    for face in &brush.faces {
        assert_eq!(face.texture, "brick");
    }
}

#[test]
fn make_brush_planes_are_non_collinear() {
    let brush = make_brush((16, 16, 0), (80, 80, 96), "t");
    for face in &brush.faces {
        let [p0, p1, p2] = face.plane_points;
        let all_same = p0 == p1 && p1 == p2;
        assert!(
            !all_same,
            "face has three identical points: {:?}",
            face.plane_points
        );
        let v1 = (p1.0 - p0.0, p1.1 - p0.1, p1.2 - p0.2);
        let v2 = (p2.0 - p0.0, p2.1 - p0.1, p2.2 - p0.2);
        let cross = (
            v1.1 * v2.2 - v1.2 * v2.1,
            v1.2 * v2.0 - v1.0 * v2.2,
            v1.0 * v2.1 - v1.1 * v2.0,
        );
        assert!(
            cross != (0, 0, 0),
            "face has collinear points: {:?}",
            face.plane_points
        );
    }
}

#[test]
fn make_brush_dimensions_are_quantum_aligned() {
    let brush = make_brush((0, 0, 0), (64, 80, 96), "t");
    let q = CONSTRUCTION_QUANTUM as i32;
    for face in &brush.faces {
        for &(x, y, z) in &face.plane_points {
            assert_eq!(x % q, 0, "plane point x {} not quantum-aligned", x);
            assert_eq!(y % q, 0, "plane point y {} not quantum-aligned", y);
            assert_eq!(z % q, 0, "plane point z {} not quantum-aligned", z);
        }
    }
}

// ── Release brush validator (G5) ──────────────────────────────────────────

#[test]
fn validate_brush_accepts_valid_box() {
    let brush = make_brush((0, 0, 0), (64, 64, 128), "stone_wall");
    assert!(validate_brush(&brush, 0).is_ok());
}

#[test]
fn validate_brush_rejects_empty_texture() {
    let brush = make_brush((0, 0, 0), (64, 64, 128), "");
    let err = validate_brush(&brush, 0).unwrap_err();
    assert!(err.to_string().contains("empty texture"), "{}", err);
}

#[test]
fn validate_brush_rejects_fewer_than_six_faces() {
    let mut brush = make_brush((0, 0, 0), (64, 64, 128), "t");
    brush.faces.truncate(5);
    let err = validate_brush(&brush, 0).unwrap_err();
    assert!(err.to_string().contains("has 5 faces"), "{}", err);
}

#[test]
fn validate_brush_rejects_zero_volume() {
    // Build a flat brush where all points share the same X coordinate,
    // producing min.x == max.x (zero volume). We craft all 6 faces with
    // distinct non-collinear triples by varying Y and Z only.
    //
    // If the collinear check fires first, the test still proves the
    // validator rejects the brush.
    use bsp_generator::intent::BrushFace;
    let brush = Brush {
        faces: vec![
            // bottom z=0: use different Y values
            BrushFace { plane_points: [(16,64,0),(16,0,0),(16,32,32)], texture: "t".into() },
            // top z=128
            BrushFace { plane_points: [(16,32,128),(16,0,128),(16,64,96)], texture: "t".into() },
            // north y=64
            BrushFace { plane_points: [(16,64,32),(16,64,0),(16,64,128)], texture: "t".into() },
            // south y=0
            BrushFace { plane_points: [(16,0,0),(16,0,128),(16,0,64)], texture: "t".into() },
            // west x=16 (variant A)
            BrushFace { plane_points: [(16,16,96),(16,48,32),(16,32,64)], texture: "t".into() },
            // east x=16 (variant B — same plane, different triples)
            BrushFace { plane_points: [(16,0,128),(16,64,0),(16,32,64)], texture: "t".into() },
        ],
    };
    // The validator should reject this brush.
    let err = validate_brush(&brush, 0).unwrap_err();
    assert!(
        err.to_string().contains("collinear")
            || err.to_string().contains("non-positive volume")
            || err.to_string().contains("inconsistent half-space"),
        "unexpected ok for zero-volume brush"
    );
}

#[test]
fn validate_brush_rejects_non_quantum_aligned() {
    // Build a valid brush then shift an entire face's plane off-quantum.
    // Move the east face from x=64 to x=63 (63 % 16 != 0), updating all
    // three plane points to keep the face planar.
    let mut brush = make_brush((0, 0, 0), (64, 64, 128), "t");
    // Face 5 is east: (64,64,0), (64,0,0), (64,0,128)
    brush.faces[5].plane_points = [(63, 64, 0), (63, 0, 0), (63, 0, 128)];
    let err = validate_brush(&brush, 0).unwrap_err();
    assert!(err.to_string().contains("not quantum-aligned"), "{}", err);
}

#[test]
fn validate_brush_rejects_collinear_face_points() {
    use bsp_generator::intent::BrushFace;
    let mut brush = make_brush((0, 0, 0), (64, 64, 128), "t");
    // Make face 0 points collinear: p0, p1, and p2 all on same line
    brush.faces[0] = BrushFace {
        plane_points: [(0, 0, 0), (32, 0, 0), (64, 0, 0)],
        texture: "t".to_string(),
    };
    let err = validate_brush(&brush, 0).unwrap_err();
    assert!(err.to_string().contains("collinear"), "{}", err);
}

#[test]
fn validate_all_brushes_returns_first_error() {
    let good = make_brush((0, 0, 0), (64, 64, 128), "good");
    let bad = make_brush((0, 0, 0), (64, 64, 128), "");
    let brushes = vec![good, bad];
    let err = validate_all_brushes(&brushes).unwrap_err();
    assert!(err.to_string().contains("brush 1"), "{}", err);
    assert!(err.to_string().contains("empty texture"), "{}", err);
}

#[test]
fn validate_all_brushes_empty_slice_ok() {
    assert!(validate_all_brushes(&[]).is_ok());
}

// ── L-junction (diagnostic — pub(crate) access) ───────────────────────────

#[test]
fn l_junction_produces_closure_brushes() {
    let h = corridor_h(0, 0, 0, 128);
    let v = corridor_v(128, 0, 0, 128);
    let brushes = build_l_junction(&h, &v);
    assert!(
        !brushes.is_empty(),
        "L-junction must produce closure brushes"
    );
    for b in &brushes {
        assert_eq!(b.faces.len(), 6);
    }
}

#[test]
fn l_junction_reverse_order_same_result() {
    let h = corridor_h(0, 0, 0, 128);
    let v = corridor_v(128, 0, 0, 128);
    let b1 = build_l_junction(&h, &v);
    let b2 = build_l_junction(&v, &h);
    assert_eq!(b1.len(), b2.len());
    for b in b1.iter().chain(b2.iter()) {
        assert_eq!(b.faces.len(), 6);
    }
}

#[test]
fn l_junction_outer_corner_brush_is_quantum_aligned() {
    let h = corridor_h(16, 32, 0, 96);
    let v = corridor_v(112, 32, 0, 96);
    let brushes = build_l_junction(&h, &v);
    let q = CONSTRUCTION_QUANTUM as i32;
    for b in &brushes {
        for face in &b.faces {
            for &(x, y, z) in &face.plane_points {
                assert_eq!(x % q, 0);
                assert_eq!(y % q, 0);
                assert_eq!(z % q, 0);
            }
        }
    }
}

// ── T-junction (diagnostic — pub(crate) access) ───────────────────────────

#[test]
fn t_junction_produces_closure_brushes() {
    let through = corridor_v(64, 0, 0, 192);
    let term = corridor_h(0, 64, 0, 64);
    let brushes = build_t_junction(&term, &through);
    assert!(
        !brushes.is_empty(),
        "T-junction must produce closure brushes"
    );
    for b in &brushes {
        assert_eq!(b.faces.len(), 6);
    }
}

#[test]
fn t_junction_terminating_from_north() {
    let through = corridor_h(0, 64, 0, 192);
    let term = corridor_v(64, 64, 0, 96);
    let brushes = build_t_junction(&term, &through);
    assert!(!brushes.is_empty());
    for b in &brushes {
        assert_eq!(b.faces.len(), 6);
    }
}

#[test]
fn t_junction_no_gaps_all_faces_valid() {
    let through = corridor_v(64, 0, 0, 192);
    let term = corridor_h(0, 64, 0, 64);
    let brushes = build_t_junction(&term, &through);
    let q = CONSTRUCTION_QUANTUM as i32;
    for b in &brushes {
        for face in &b.faces {
            for &(x, y, z) in &face.plane_points {
                assert_eq!(x % q, 0, "x not quantum-aligned");
                assert_eq!(y % q, 0, "y not quantum-aligned");
                assert_eq!(z % q, 0, "z not quantum-aligned");
            }
        }
    }
}

// ── X-junction (diagnostic — pub(crate) access) ───────────────────────────

#[test]
fn x_junction_produces_four_corner_brushes() {
    let h = corridor_h(0, 64, 0, 192);
    let v = corridor_v(64, 0, 0, 192);
    let brushes = build_x_junction(&h, &v);
    assert_eq!(
        brushes.len(),
        4,
        "X-junction must produce exactly 4 corner closure brushes, got {}",
        brushes.len()
    );
    for b in &brushes {
        assert_brush_faces_are_non_degenerate(b);
        assert_eq!(brush_z_range(b), (0, 112));
    }
}

#[test]
fn x_junction_brushes_are_full_height_and_non_degenerate() {
    let h = corridor_h(0, 64, 0, 192);
    let v = corridor_v(64, 0, 0, 192);
    let brushes = build_x_junction(&h, &v);

    assert_eq!(brushes.len(), 4);
    for brush in &brushes {
        assert_brush_faces_are_non_degenerate(brush);
        assert_eq!(brush_z_range(brush), (0, 112));
    }
}

#[test]
fn x_junction_parallel_corridors_produces_nothing() {
    let h1 = corridor_h(0, 0, 0, 128);
    let h2 = corridor_h(0, 64, 0, 128);
    let brushes = build_x_junction(&h1, &h2);
    assert!(
        brushes.is_empty(),
        "parallel corridors should not produce X-junction"
    );
}

#[test]
fn x_junction_corners_are_symmetric() {
    let h = corridor_h(0, 64, 0, 192);
    let v = corridor_v(64, 0, 0, 192);
    let brushes = build_x_junction(&h, &v);

    assert_eq!(brushes.len(), 4);
    let face_counts: Vec<usize> = brushes.iter().map(|b| b.faces.len()).collect();
    assert_eq!(&face_counts, &[6, 6, 6, 6]);
}

// ── Room portal (diagnostic — pub(crate) access) ──────────────────────────

#[test]
fn room_portal_produces_wall_pieces_around_opening() {
    let room = room(0, 0, 0, 112, 112, 128);
    let corr = Corridor {
        start: (112, 56, 0),
        end: (176, 56, 0),
        width: 64,
        height: 80,
    };
    let brushes = build_room_portal(&corr, &room);
    assert_eq!(brushes.len(), 3, "two columns and one lintel expected");
    for brush in &brushes {
        assert_eq!(brush.faces.len(), 6);
        assert!(brush.faces.iter().all(|face| face.texture == "stone_wall"));
    }
}

#[test]
fn room_portal_north_wall() {
    let room = room(32, 0, 0, 112, 112, 128);
    let corr = Corridor {
        start: (88, 112, 0),
        end: (88, 176, 0),
        width: 64,
        height: 80,
    };
    let brushes = build_room_portal(&corr, &room);
    assert_eq!(brushes.len(), 3);
}

#[test]
fn room_portal_west_wall() {
    let room = room(64, 32, 0, 112, 112, 128);
    let corr = Corridor {
        start: (64, 88, 0),
        end: (0, 88, 0),
        width: 64,
        height: 80,
    };
    let brushes = build_room_portal(&corr, &room);
    assert_eq!(brushes.len(), 3);
}

#[test]
fn room_portal_south_wall() {
    let room = room(32, 64, 0, 112, 112, 128);
    let corr = Corridor {
        start: (88, 64, 0),
        end: (88, 0, 0),
        width: 64,
        height: 80,
    };
    let brushes = build_room_portal(&corr, &room);
    assert_eq!(brushes.len(), 3);
}

// ── build_junction_closures (diagnostic — pub(crate) access) ─────────────

#[test]
fn build_junction_closures_for_single_corridor_returns_empty() {
    let corridors = vec![corridor_h(0, 0, 0, 64)];
    let brushes = build_junction_closures(&corridors);
    assert!(brushes.is_empty());
}

#[test]
fn build_junction_closures_for_two_parallel_corridors() {
    let corridors = vec![
        corridor_h(0, 0, 0, 64),
        corridor_h(64, 0, 0, 64),
    ];
    let brushes = build_junction_closures(&corridors);
    for b in &brushes {
        assert_eq!(b.faces.len(), 6);
    }
}

#[test]
fn build_junction_closures_for_l_shape() {
    let corridors = vec![corridor_h(0, 0, 0, 64), corridor_v(64, 0, 0, 64)];
    let brushes = build_junction_closures(&corridors);
    assert!(!brushes.is_empty());
    for b in &brushes {
        assert_eq!(b.faces.len(), 6);
    }
}

#[test]
fn build_junction_closures_for_star_topology() {
    let corridors = vec![
        corridor_h(0, 0, 0, 64),
        corridor_v(64, 0, 0, 64),
        corridor_v(64, 64, 0, 64),
    ];
    let brushes = build_junction_closures(&corridors);
    for b in &brushes {
        assert_eq!(b.faces.len(), 6);
    }
}

#[test]
fn junction_closure_brushes_are_small_and_local() {
    let corridors = vec![
        corridor_h(100, 100, 0, 200),
        corridor_v(200, 100, 0, 200),
        corridor_v(200, 200, 0, 200),
    ];
    let brushes = build_junction_closures(&corridors);
    for b in &brushes {
        for face in &b.faces {
            for &(x, y, z) in &face.plane_points {
                assert!(x >= 0);
                assert!(y >= 0);
                assert!(z >= 0);
                assert!(x <= 500, "brush extends too far east: x={}", x);
                assert!(y <= 500, "brush extends too far north: y={}", y);
            }
        }
    }
}
