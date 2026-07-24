//! Integration tests for [`bsp_generator::geometry`] validation utilities.
//!
//! Covers quantum snapping, overlap detection, wall-thickness separation,
//! bounds validation, cycle count validation, and connectedness checks.

use bsp_generator::geometry::{
    rooms_overlap, snap_to_quantum, validate_bounds, validate_connectedness, validate_cycle_count,
    validate_no_overlap, validate_quantum_alignment,
};
use bsp_generator::RoomIntent;

fn room_at(x: i32, y: i32, z: i32, dx: u32, dy: u32, dz: u32) -> RoomIntent {
    RoomIntent {
        position: (x, y, z),
        dimensions: (dx, dy, dz),
    }
}

// ── snap_to_quantum ────────────────────────────────────────────────────────

#[test]
fn quantum_snapping_rounds_correctly() {
    // Positive
    assert_eq!(snap_to_quantum(0, 16), 0);
    assert_eq!(snap_to_quantum(1, 16), 0);
    assert_eq!(snap_to_quantum(7, 16), 0);
    assert_eq!(snap_to_quantum(8, 16), 16); // half-up
    assert_eq!(snap_to_quantum(15, 16), 16);
    assert_eq!(snap_to_quantum(16, 16), 16);
    assert_eq!(snap_to_quantum(23, 16), 16);
    assert_eq!(snap_to_quantum(24, 16), 32); // half-up
    assert_eq!(snap_to_quantum(25, 16), 32);

    // Negative
    assert_eq!(snap_to_quantum(-1, 16), 0);
    assert_eq!(snap_to_quantum(-7, 16), 0);
    assert_eq!(snap_to_quantum(-8, 16), 0); // half-up → 0
    assert_eq!(snap_to_quantum(-9, 16), -16);
    assert_eq!(snap_to_quantum(-24, 16), -16); // half-up
    assert_eq!(snap_to_quantum(-25, 16), -32);
    assert_eq!(snap_to_quantum(-40, 16), -32); // exactly halfway → up
    assert_eq!(snap_to_quantum(-41, 16), -48);
}

#[test]
fn snap_to_quantum_preserves_already_aligned() {
    for v in (-1536..=1536).step_by(16) {
        assert_eq!(snap_to_quantum(v, 16), v);
    }
}

#[test]
fn snap_to_quantum_large_bounds() {
    assert_eq!(snap_to_quantum(3072, 16), 3072);
    assert_eq!(snap_to_quantum(3071, 16), 3072);
    assert_eq!(snap_to_quantum(-3072, 16), -3072);
}

// ── rooms_overlap ──────────────────────────────────────────────────────────

#[test]
fn overlapping_rooms_detected() {
    let a = room_at(0, 0, 0, 64, 64, 128);
    let b = room_at(32, 32, 0, 64, 64, 128); // clearly overlaps
    assert!(rooms_overlap(&a, &b, 16));
}

#[test]
fn adjacent_rooms_with_wall_thickness_do_not_overlap() {
    // 16-unit wall gap required: room ends at 64, next starts at 80
    let a = room_at(0, 0, 0, 64, 64, 128);
    let b = room_at(80, 0, 0, 64, 64, 128);
    assert!(!rooms_overlap(&a, &b, 16));
}

#[test]
fn wall_thickness_separation_enforced() {
    // Rooms separated by exactly 15 units (less than wall thickness 16)
    // Room A: x=0..64, Room B: x=79..143, gap=15 < 16
    let a = room_at(0, 0, 0, 64, 64, 128);
    let b = room_at(79, 0, 0, 64, 64, 128);
    assert!(rooms_overlap(&a, &b, 16));
}

#[test]
fn rooms_with_exact_wall_gap_passes() {
    let a = room_at(0, 0, 0, 64, 64, 128);
    let b = room_at(80, 0, 0, 64, 64, 128); // exact 16 gap
    assert!(!rooms_overlap(&a, &b, 16));
}

#[test]
fn rooms_separated_in_y_only() {
    let a = room_at(0, 0, 0, 64, 64, 128);
    let b = room_at(0, 80, 0, 64, 64, 128); // exact 16 gap in Y
    assert!(!rooms_overlap(&a, &b, 16));
}

#[test]
fn rooms_identical_overlap() {
    let a = room_at(0, 0, 0, 64, 64, 128);
    assert!(rooms_overlap(&a, &a, 16));
}

// ── validate_bounds ────────────────────────────────────────────────────────

#[test]
fn bounds_validation_catches_out_of_bounds_room_x() {
    let rooms = vec![room_at(1000, 0, 0, 64, 64, 128)];
    let err = validate_bounds(&rooms, (1024, 1024), 256).unwrap_err();
    assert!(err.to_string().contains("exceeds max_x"));
}

#[test]
fn bounds_validation_catches_out_of_bounds_room_y() {
    let rooms = vec![room_at(0, 1000, 0, 64, 64, 128)];
    let err = validate_bounds(&rooms, (1024, 1024), 256).unwrap_err();
    assert!(err.to_string().contains("exceeds max_y"));
}

#[test]
fn bounds_validation_catches_out_of_bounds_room_z() {
    let rooms = vec![room_at(0, 0, 200, 64, 64, 128)];
    let err = validate_bounds(&rooms, (1024, 1024), 256).unwrap_err();
    assert!(err.to_string().contains("exceeds max_z"));
}

#[test]
fn bounds_validation_catches_negative_position() {
    let rooms = vec![room_at(-16, 0, 0, 64, 64, 128)];
    let err = validate_bounds(&rooms, (1024, 1024), 256).unwrap_err();
    assert!(err.to_string().contains("negative"));
}

#[test]
fn bounds_validation_passes_for_in_bounds_rooms() {
    let rooms = vec![
        room_at(0, 0, 0, 64, 64, 128),
        room_at(80, 80, 0, 64, 64, 128),
        room_at(500, 500, 0, 96, 96, 128),
    ];
    assert!(validate_bounds(&rooms, (1024, 1024), 256).is_ok());
}

// ── validate_cycle_count ───────────────────────────────────────────────────

#[test]
fn cycle_count_exact_match_passes() {
    let edges = vec![(0, 1), (1, 2), (0, 2)]; // 3 rooms, MST=2, loops=1, total=3
    assert!(validate_cycle_count(&edges, 3, 1).is_ok());
}

#[test]
fn cycle_count_mismatch_fails() {
    let edges = vec![(0, 1)]; // 1 edge, 3 rooms needs at least 2
    let err = validate_cycle_count(&edges, 3, 0).unwrap_err();
    assert!(err.to_string().contains("edge count"));
}

#[test]
fn cycle_count_with_extra_loops() {
    let edges = vec![(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]; // 4 rooms, MST=3, loops=2
    assert!(validate_cycle_count(&edges, 4, 2).is_ok());
}

#[test]
fn cycle_count_zero_rooms() {
    assert!(validate_cycle_count(&[], 0, 0).is_ok());
}

// ── validate_connectedness ─────────────────────────────────────────────────

#[test]
fn connectedness_single_component() {
    let edges = vec![(0, 1), (1, 2), (2, 3), (3, 4)];
    assert!(validate_connectedness(&edges, 5));
}

#[test]
fn connectedness_two_components_fails() {
    let edges = vec![(0, 1), (2, 3)]; // two separate pairs
    assert!(!validate_connectedness(&edges, 4));
}

#[test]
fn connectedness_disjoint_with_extra_edges() {
    // Component 1: 0-1-2, Component 2: 3-4
    let edges = vec![(0, 1), (1, 2), (3, 4)];
    assert!(!validate_connectedness(&edges, 5));
}

#[test]
fn connectedness_single_room() {
    assert!(validate_connectedness(&[], 1));
}

#[test]
fn connectedness_empty() {
    assert!(validate_connectedness(&[], 0));
}

// ── validate_no_overlap ────────────────────────────────────────────────────

#[test]
fn no_overlap_passes_for_separated_rooms() {
    let rooms = vec![
        room_at(0, 0, 0, 64, 64, 128),
        room_at(80, 0, 0, 64, 64, 128),
        room_at(0, 80, 0, 64, 64, 128),
        room_at(80, 80, 0, 64, 64, 128),
    ];
    assert!(validate_no_overlap(&rooms, 16).is_ok());
}

#[test]
fn no_overlap_detects_overlap() {
    let rooms = vec![
        room_at(0, 0, 0, 64, 64, 128),
        room_at(48, 48, 0, 64, 64, 128), // overlaps first
    ];
    let err = validate_no_overlap(&rooms, 16).unwrap_err();
    assert!(err.to_string().contains("overlap"));
}

#[test]
fn no_overlap_empty_list() {
    assert!(validate_no_overlap(&[], 16).is_ok());
}

// ── validate_quantum_alignment ─────────────────────────────────────────────

#[test]
fn quantum_alignment_all_valid() {
    let rooms = vec![
        room_at(0, 0, 0, 64, 64, 128),
        room_at(80, 80, 0, 96, 96, 128),
    ];
    assert!(validate_quantum_alignment(&rooms).is_ok());
}

#[test]
fn quantum_alignment_detects_unaligned_position() {
    let rooms = vec![room_at(8, 0, 0, 64, 64, 128)];
    let err = validate_quantum_alignment(&rooms).unwrap_err();
    assert!(err.to_string().contains("aligned"));
}

#[test]
fn quantum_alignment_detects_unaligned_dimension() {
    let rooms = vec![room_at(0, 0, 0, 63, 64, 128)];
    let err = validate_quantum_alignment(&rooms).unwrap_err();
    assert!(err.to_string().contains("multiples"));
}
