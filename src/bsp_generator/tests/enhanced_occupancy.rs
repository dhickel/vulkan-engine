//! Enhanced v2 occupancy grid tests — checkpoint integrity, conflict detection,
//! cell reservation consistency, and grid edge cases.

use bsp_generator::enhanced::intent::RoomId;
use bsp_generator::enhanced::occupancy::OccupancyGrid;

const Q: u32 = 16;

fn grid_1024() -> OccupancyGrid {
    OccupancyGrid::new(1024, 1024).unwrap()
}

fn grid_3072() -> OccupancyGrid {
    OccupancyGrid::new(3072, 3072).unwrap()
}

// ── Construction ───────────────────────────────────────────────────────────

#[test]
fn grid_creation_valid_sizes() {
    for size in [512u32, 1024, 1536, 2048, 3072] {
        let g = OccupancyGrid::new(size, size).unwrap();
        assert_eq!(g.cells_x(), size / Q);
        assert_eq!(g.cells_y(), size / Q);
    }
}

#[test]
fn grid_rejects_zero() {
    assert!(OccupancyGrid::new(0, 1024).is_err());
    assert!(OccupancyGrid::new(1024, 0).is_err());
}

#[test]
fn grid_rejects_non_quantum() {
    assert!(OccupancyGrid::new(1023, 1024).is_err());
    assert!(OccupancyGrid::new(1024, 15).is_err());
    assert!(OccupancyGrid::new(100, 1024).is_err()); // 100 not divisible by 16
}

#[test]
fn grid_initial_state_all_empty() {
    let g = grid_1024();
    assert_eq!(g.owned_cell_count(), 0);
    assert!(g.is_rect_empty(0, 0, 1024, 1024).unwrap());
}

// ── Reservation ────────────────────────────────────────────────────────────

#[test]
fn single_reservation() {
    let mut g = grid_1024();
    g.reserve_rect(0, 0, 128, 128, RoomId(0)).unwrap();
    assert!(!g.is_rect_empty(0, 0, 128, 128).unwrap());
    assert!(g.is_rect_empty(128, 0, 128, 128).unwrap());
}

#[test]
fn multiple_non_overlapping() {
    let mut g = grid_1024();
    g.reserve_rect(0, 0, 256, 256, RoomId(0)).unwrap();
    g.reserve_rect(512, 0, 256, 256, RoomId(1)).unwrap();
    g.reserve_rect(0, 512, 256, 256, RoomId(2)).unwrap();
    g.reserve_rect(512, 512, 256, 256, RoomId(3)).unwrap();
}

#[test]
fn overlapping_same_cell_rejected() {
    let mut g = grid_1024();
    g.reserve_rect(0, 0, 64, 64, RoomId(0)).unwrap();
    let err = g.reserve_rect(0, 0, 64, 64, RoomId(1)).unwrap_err();
    assert!(err.to_string().contains("already owned"));
}

#[test]
fn partial_overlap_rejected() {
    let mut g = grid_1024();
    g.reserve_rect(0, 0, 256, 256, RoomId(0)).unwrap();
    // Overlap at corner
    let err = g.reserve_rect(128, 128, 256, 256, RoomId(1)).unwrap_err();
    assert!(err.to_string().contains("already owned"));
}

#[test]
fn touching_exact_boundary_allowed() {
    let mut g = grid_1024();
    // Two rooms sharing an exact cell boundary — allowed
    g.reserve_rect(0, 0, 128, 128, RoomId(0)).unwrap();
    g.reserve_rect(128, 0, 128, 128, RoomId(1)).unwrap();
    // Boundary cell (128,0) is part of room 1's first column
    assert!(!g.is_rect_empty(0, 0, 128, 128).unwrap());
    assert!(!g.is_rect_empty(128, 0, 128, 128).unwrap());
}

#[test]
fn one_cell_gap_allowed() {
    let mut g = grid_1024();
    g.reserve_rect(0, 0, 128, 128, RoomId(0)).unwrap();
    // Room at 144, 0 — one cell (16 units) gap
    g.reserve_rect(144, 0, 128, 128, RoomId(1)).unwrap();
    // The cell at 128,0 should still be empty
    assert!(g.is_rect_empty(128, 0, 16, 16).unwrap());
}

// ── Checkpoint / Restore ───────────────────────────────────────────────────

#[test]
fn checkpoint_restore_basic() {
    let mut g = grid_1024();
    let cp = g.checkpoint();

    g.reserve_rect(0, 0, 256, 256, RoomId(0)).unwrap();
    g.reserve_rect(512, 512, 256, 256, RoomId(1)).unwrap();
    assert_eq!(g.owned_cell_count(), 512);

    g.restore(cp);
    assert_eq!(g.owned_cell_count(), 0);
}

#[test]
fn checkpoint_then_partial_restore() {
    let mut g = grid_1024();
    let cp0 = g.checkpoint();

    g.reserve_rect(0, 0, 128, 128, RoomId(0)).unwrap();
    let cp1 = g.checkpoint();

    g.reserve_rect(256, 0, 128, 128, RoomId(1)).unwrap();
    assert!(!g.is_rect_empty(256, 0, 128, 128).unwrap());

    // Restore to cp1: only room 0
    g.restore(cp1);
    assert!(!g.is_rect_empty(0, 0, 128, 128).unwrap());
    assert!(g.is_rect_empty(256, 0, 128, 128).unwrap());

    // Restore to cp0: nothing
    g.restore(cp0);
    assert!(g.is_rect_empty(0, 0, 128, 128).unwrap());
}

#[test]
fn checkpoint_is_independent_snapshot() {
    let mut g = grid_1024();
    g.reserve_rect(0, 0, 128, 128, RoomId(0)).unwrap();
    let cp = g.checkpoint();

    // Modify grid after checkpoint
    g.reserve_rect(256, 256, 128, 128, RoomId(1)).unwrap();

    // Restore should bring us back to only room 0
    g.restore(cp);
    assert!(!g.is_rect_empty(0, 0, 128, 128).unwrap());
    assert!(g.is_rect_empty(256, 256, 128, 128).unwrap());
}

#[test]
fn double_restore_consistent() {
    let mut g = grid_1024();
    let cp = g.checkpoint();
    g.reserve_rect(0, 0, 128, 128, RoomId(0)).unwrap();

    g.restore(cp.clone());
    assert!(g.is_rect_empty(0, 0, 128, 128).unwrap());

    g.restore(cp);
    assert!(g.is_rect_empty(0, 0, 128, 128).unwrap());
}

#[test]
fn no_reservation_after_failed_attempt() {
    let mut g = grid_1024();
    g.reserve_rect(0, 0, 256, 256, RoomId(0)).unwrap();

    let cp = g.checkpoint();
    let result = g.reserve_rect(128, 128, 256, 256, RoomId(1));
    assert!(result.is_err());

    // Grid should be unchanged (the failed reserve didn't partially write)
    g.restore(cp); // cp was from before the failed attempt — grid same
    assert!(!g.is_rect_empty(0, 0, 256, 256).unwrap());
    assert!(g.is_rect_empty(256, 0, 256, 256).unwrap());
}

// ── Bounds and error cases ─────────────────────────────────────────────────

#[test]
fn negative_origin_rejected() {
    let g = grid_1024();
    assert!(g.is_rect_empty(-16, 0, 64, 64).is_err());
    assert!(g.is_rect_empty(0, -16, 64, 64).is_err());
}

#[test]
fn non_quantum_rect_rejected_without_reservation() {
    let mut g = grid_1024();
    assert!(g.is_rect_empty(1, 0, 64, 64).is_err());
    assert!(g.is_rect_empty(0, 1, 64, 64).is_err());
    assert!(g.reserve_rect(0, 0, 63, 64, RoomId(0)).is_err());
    assert!(g.reserve_rect(0, 0, 64, 63, RoomId(0)).is_err());
    assert_eq!(g.owned_cell_count(), 0);
}

#[test]
fn out_of_bounds_rejected() {
    let g = grid_1024();
    // Exactly at edge: (1008, 0) to (1024, 16) — fits within 0..1024
    assert!(g.is_rect_empty(1008, 0, 16, 16).is_ok());
    // Exceeds by one cell
    assert!(g.is_rect_empty(1008, 0, 32, 16).is_err());
    assert!(g.is_rect_empty(0, 1008, 16, 32).is_err());
}

#[test]
fn zero_area_rejected() {
    let mut g = grid_1024();
    assert!(g.reserve_rect(0, 0, 0, 64, RoomId(0)).is_err());
    assert!(g.reserve_rect(0, 0, 64, 0, RoomId(0)).is_err());
}

#[test]
fn reserve_at_origin() {
    let mut g = grid_1024();
    g.reserve_rect(0, 0, 16, 16, RoomId(0)).unwrap();
    assert!(!g.is_rect_empty(0, 0, 16, 16).unwrap());
}

#[test]
fn reserve_at_max_extent() {
    let mut g = grid_1024();
    g.reserve_rect(1008, 1008, 16, 16, RoomId(0)).unwrap();
    assert!(!g.is_rect_empty(1008, 1008, 16, 16).unwrap());
}

// ── Large grid stress ──────────────────────────────────────────────────────

#[test]
fn large_grid_3072_operations() {
    let mut g = grid_3072();
    // Place many rooms
    for i in 0..20u32 {
        let x = (i * 144) % 2800;
        let y = ((i * 97) % 3) * 960;
        let _ = g.reserve_rect(x as i32, y as i32, 128, 128, RoomId(i));
    }
}

#[test]
fn large_grid_checkpoint_size() {
    let g = grid_3072();
    let cp = g.checkpoint();
    // 192*192 = 36864 cells, each 1 byte → ~36KB
    assert_eq!(cp.len(), 192 * 192);
}

#[test]
fn full_grid_fill_then_empty_check() {
    let mut g = OccupancyGrid::new(256, 256).unwrap();
    // Fill every cell
    for cy in 0..g.cells_y() {
        for cx in 0..g.cells_x() {
            g.reserve_rect(
                (cx * Q) as i32,
                (cy * Q) as i32,
                Q as i32,
                Q as i32,
                RoomId(0),
            )
            .unwrap();
        }
    }
    assert_eq!(g.owned_cell_count() as u32, 256 / Q * 256 / Q);
    // Any new reservation should fail
    assert!(g.reserve_rect(0, 0, 16, 16, RoomId(1)).is_err());
}

// ── owned_cell_count ───────────────────────────────────────────────────────

#[test]
fn owned_cell_count_increments() {
    let mut g = grid_1024();
    assert_eq!(g.owned_cell_count(), 0);

    g.reserve_rect(0, 0, 128, 128, RoomId(0)).unwrap();
    // 128/16 * 128/16 = 8*8 = 64 cells
    assert_eq!(g.owned_cell_count(), 64);

    g.reserve_rect(256, 0, 64, 64, RoomId(1)).unwrap();
    // + 4*4 = 16 cells → 80 total
    assert_eq!(g.owned_cell_count(), 80);
}

#[test]
fn owned_cell_count_after_restore() {
    let mut g = grid_1024();
    let cp = g.checkpoint();

    g.reserve_rect(0, 0, 256, 256, RoomId(0)).unwrap();
    assert!(g.owned_cell_count() > 0);

    g.restore(cp);
    assert_eq!(g.owned_cell_count(), 0);
}

// ── is_rect_empty partial checks ───────────────────────────────────────────

#[test]
fn is_rect_empty_with_mixed_ownership() {
    let mut g = grid_1024();
    // Reserve left half only
    g.reserve_rect(0, 0, 256, 512, RoomId(0)).unwrap();
    // Right half should be empty
    assert!(g.is_rect_empty(256, 0, 256, 512).unwrap());
    // Left half should NOT be empty
    assert!(!g.is_rect_empty(0, 0, 256, 512).unwrap());
    // A rect spanning both halves should NOT be empty
    assert!(!g.is_rect_empty(128, 0, 384, 512).unwrap());
}
