//! Enhanced v2 projected occupancy — owner-bearing XY grid with journal snapshots.
//!
//! Tracks per-cell ownership for room shell footprints across both layers.
//! Supports full journal-based checkpoint/rollback for transactional placement.

use crate::config::CONSTRUCTION_QUANTUM;

use super::error::EnhancedError;
use super::intent::{ReservationId, RoomId, RouteId, TransitionId};

const Q: u32 = CONSTRUCTION_QUANTUM;

/// Who owns a projected grid cell.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Owner {
    /// Unclaimed.
    Empty,
    /// Owned by a room shell.
    Room(RoomId),
    /// Owned by a horizontal route.
    Route(RouteId),
    /// Owned by a vertical transition (stair).
    Transition(TransitionId),
    /// A generic reservation (used for partial staging).
    Reservation(ReservationId),
}

/// A snapshot of the grid for checkpoint/rollback.
#[derive(Debug, Clone)]
pub struct GridCheckpoint {
    pub(crate) cells: Vec<Owner>,
}

impl GridCheckpoint {
    /// Number of cells in this checkpoint.
    pub fn len(&self) -> usize {
        self.cells.len()
    }

    /// Returns true if the checkpoint has no cells.
    pub fn is_empty(&self) -> bool {
        self.cells.is_empty()
    }
}

/// Projected XY occupancy grid with owner-bearing conflict detection.
///
/// The grid covers `[0, width)` × `[0, height)` in Quake units at
/// [`CONSTRUCTION_QUANTUM`] cell resolution. All public coordinates must
/// be quantum-aligned.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OccupancyGrid {
    /// Grid dimensions in cells.
    cells_x: u32,
    cells_y: u32,
    /// Flat cell array: `cells[cells_x * y + x]`.
    cells: Vec<Owner>,
}

impl OccupancyGrid {
    /// Create a new empty grid covering `width`×`height` Quake units.
    ///
    /// Both `width` and `height` must be positive multiples of
    /// [`CONSTRUCTION_QUANTUM`].
    pub fn new(width: u32, height: u32) -> Result<Self, EnhancedError> {
        if width == 0 || height == 0 {
            return Err(EnhancedError::ContractViolation {
                detail: format!("grid dimensions must be non-zero: {}x{}", width, height),
            });
        }
        if width % Q != 0 || height % Q != 0 {
            return Err(EnhancedError::ContractViolation {
                detail: format!(
                    "grid dimensions must be quantum-aligned: {}x{} (quantum {})",
                    width, height, Q,
                ),
            });
        }
        let cx = width / Q;
        let cy = height / Q;
        let cap =
            (cx as usize)
                .checked_mul(cy as usize)
                .ok_or(EnhancedError::ArithmeticOverflow {
                    operation: "grid_capacity",
                })?;
        Ok(Self {
            cells_x: cx,
            cells_y: cy,
            cells: vec![Owner::Empty; cap],
        })
    }

    /// Number of cells along the X axis.
    pub fn cells_x(&self) -> u32 {
        self.cells_x
    }

    /// Number of cells along the Y axis.
    pub fn cells_y(&self) -> u32 {
        self.cells_y
    }

    /// Read-only access to the cell array (for manual checks).
    pub fn cells(&self) -> &[Owner] {
        &self.cells
    }

    /// Mutable access to the cell array (for manual writes).
    pub fn cells_mut(&mut self) -> &mut [Owner] {
        &mut self.cells
    }

    /// Capture a full checkpoint of the grid.
    pub fn checkpoint(&self) -> GridCheckpoint {
        GridCheckpoint {
            cells: self.cells.clone(),
        }
    }

    /// Restore the grid to a previously captured checkpoint.
    pub fn restore(&mut self, cp: GridCheckpoint) {
        self.cells = cp.cells;
    }

    /// Compute the linear cell index. Coordinates are in cell space.
    fn cell_index(&self, cx: u32, cy: u32) -> Result<usize, EnhancedError> {
        if cx >= self.cells_x || cy >= self.cells_y {
            return Err(EnhancedError::ContractViolation {
                detail: format!(
                    "cell index out of bounds: ({}, {}) in grid {}×{}",
                    cx, cy, self.cells_x, self.cells_y,
                ),
            });
        }
        Ok((self.cells_x as usize) * (cy as usize) + (cx as usize))
    }

    /// Convert and validate a Quake-space rectangle into grid-cell space.
    fn rect_cells(
        &self,
        x0: i32,
        y0: i32,
        w: i32,
        h: i32,
    ) -> Result<(u32, u32, u32, u32), EnhancedError> {
        if x0 < 0 || y0 < 0 {
            return Err(EnhancedError::ContractViolation {
                detail: format!("negative rect origin: ({}, {})", x0, y0),
            });
        }
        if w <= 0 || h <= 0 {
            return Err(EnhancedError::ContractViolation {
                detail: format!("non-positive rect: {}×{} at ({}, {})", w, h, x0, y0),
            });
        }

        let quantum = Q as i32;
        if x0 % quantum != 0 || y0 % quantum != 0 || w % quantum != 0 || h % quantum != 0 {
            return Err(EnhancedError::ContractViolation {
                detail: format!(
                    "rect must be quantum-aligned: ({}, {}) {}×{} (quantum {})",
                    x0, y0, w, h, Q,
                ),
            });
        }

        let qx0 = (x0 as u32) / Q;
        let qy0 = (y0 as u32) / Q;
        let qw = (w as u32) / Q;
        let qh = (h as u32) / Q;
        let qx1 = qx0
            .checked_add(qw)
            .ok_or(EnhancedError::ArithmeticOverflow {
                operation: "rect_x_extent",
            })?;
        let qy1 = qy0
            .checked_add(qh)
            .ok_or(EnhancedError::ArithmeticOverflow {
                operation: "rect_y_extent",
            })?;

        if qx1 > self.cells_x || qy1 > self.cells_y {
            return Err(EnhancedError::ContractViolation {
                detail: format!(
                    "rect ({}, {}) {}×{} exceeds grid {}×{}",
                    x0,
                    y0,
                    w,
                    h,
                    self.cells_x * Q,
                    self.cells_y * Q,
                ),
            });
        }
        Ok((qx0, qy0, qw, qh))
    }

    /// Check whether a rectangular region is entirely unclaimed.
    ///
    /// All coordinates are in Quake units and must be quantum-aligned.
    /// `x0`, `y0` are the minimum-corner position; `w`, `h` are positive
    /// multiples of the quantum.
    pub fn is_rect_empty(&self, x0: i32, y0: i32, w: i32, h: i32) -> Result<bool, EnhancedError> {
        let (qx0, qy0, qw, qh) = self.rect_cells(x0, y0, w, h)?;

        for dy in 0..qh {
            for dx in 0..qw {
                let idx = self.cell_index(qx0 + dx, qy0 + dy)?;
                if !matches!(self.cells[idx], Owner::Empty) {
                    return Ok(false);
                }
            }
        }
        Ok(true)
    }

    /// Reserve a rectangular region for a generic owner.
    ///
    /// All coordinates are in Quake units and must be quantum-aligned.
    /// V2: lenient — overwrites any existing owner without error
    /// (used for routes and transitions which can share cells).
    pub fn reserve_rect_owner(
        &mut self,
        x0: i32,
        y0: i32,
        w: i32,
        h: i32,
        owner: Owner,
    ) -> Result<(), EnhancedError> {
        let (qx0, qy0, qw, qh) = self.rect_cells(x0, y0, w, h)?;

        for dy in 0..qh {
            for dx in 0..qw {
                let idx = self.cell_index(qx0 + dx, qy0 + dy)?;
                self.cells[idx] = owner;
            }
        }
        Ok(())
    }

    /// Reserve a rectangular region for a room.
    ///
    /// All coordinates are in Quake units and must be quantum-aligned.
    /// Returns an error if any cell in the region is already owned by
    /// another room (rooms must not overlap).
    pub fn reserve_rect(
        &mut self,
        x0: i32,
        y0: i32,
        w: i32,
        h: i32,
        owner: RoomId,
    ) -> Result<(), EnhancedError> {
        let (qx0, qy0, qw, qh) = self.rect_cells(x0, y0, w, h)?;

        // Two-pass: first check all cells for room-room conflicts only.
        for dy in 0..qh {
            for dx in 0..qw {
                let idx = self.cell_index(qx0 + dx, qy0 + dy)?;
                match self.cells[idx] {
                    Owner::Empty => {}
                    Owner::Room(_) => {
                        return Err(EnhancedError::ContractViolation {
                            detail: format!(
                                "cell ({}, {}) already owned by another room",
                                (qx0 + dx) * Q,
                                (qy0 + dy) * Q,
                            ),
                        });
                    }
                    _ => {} // Routes/transitions can coexist with rooms
                }
            }
        }

        for dy in 0..qh {
            for dx in 0..qw {
                let idx = self.cell_index(qx0 + dx, qy0 + dy)?;
                self.cells[idx] = Owner::Room(owner);
            }
        }
        Ok(())
    }

    /// Check whether a single cell at the given Quake coordinates is empty
    /// or owned by any of the given room IDs.
    ///
    /// Coordinates must be quantum-aligned.
    pub fn is_cell_empty_or_owned_by(
        &self,
        qx: i32,
        qy: i32,
        allowed_rooms: &[RoomId],
    ) -> Result<bool, EnhancedError> {
        let cx = (qx as u32) / Q;
        let cy = (qy as u32) / Q;
        let idx = self.cell_index(cx, cy)?;
        match self.cells[idx] {
            Owner::Empty => Ok(true),
            Owner::Room(rid) => Ok(allowed_rooms.contains(&rid)),
            Owner::Route(_) | Owner::Transition(_) | Owner::Reservation(_) => Ok(false),
        }
    }

    /// Reserve a rectangular region, allowing cells already owned by the
    /// given `allowed_rooms` to be overwritten.
    pub fn reserve_rect_allow_rooms(
        &mut self,
        x0: i32,
        y0: i32,
        w: i32,
        h: i32,
        owner: Owner,
        allowed_rooms: &[RoomId],
    ) -> Result<(), EnhancedError> {
        let (qx0, qy0, qw, qh) = self.rect_cells(x0, y0, w, h)?;

        // Two-pass: first check all cells
        for dy in 0..qh {
            for dx in 0..qw {
                let idx = self.cell_index(qx0 + dx, qy0 + dy)?;
                match self.cells[idx] {
                    Owner::Empty => {}
                    Owner::Room(rid) if allowed_rooms.contains(&rid) => {}
                    other => {
                        return Err(EnhancedError::ContractViolation {
                            detail: format!(
                                "cell ({}, {}) already owned by {:?}",
                                (qx0 + dx) * Q,
                                (qy0 + dy) * Q,
                                other,
                            ),
                        });
                    }
                }
            }
        }

        for dy in 0..qh {
            for dx in 0..qw {
                let idx = self.cell_index(qx0 + dx, qy0 + dy)?;
                self.cells[idx] = owner;
            }
        }
        Ok(())
    }

    /// Count how many cells are owned (non-empty).
    pub fn owned_cell_count(&self) -> usize {
        self.cells
            .iter()
            .filter(|c| !matches!(c, Owner::Empty))
            .count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GRID_1024: u32 = 1024;

    fn make_grid() -> OccupancyGrid {
        OccupancyGrid::new(GRID_1024, GRID_1024).unwrap()
    }

    #[test]
    fn new_grid_is_all_empty() {
        let g = make_grid();
        assert!(g.is_rect_empty(0, 0, 256, 256).unwrap());
        assert_eq!(g.owned_cell_count(), 0);
    }

    #[test]
    fn reserve_and_check() {
        let mut g = make_grid();
        g.reserve_rect(0, 0, 256, 256, RoomId(0)).unwrap();
        assert!(!g.is_rect_empty(0, 0, 256, 256).unwrap());
        assert!(g.is_rect_empty(256, 0, 256, 256).unwrap());
    }

    #[test]
    fn overlapping_reservation_rejected() {
        let mut g = make_grid();
        g.reserve_rect(0, 0, 256, 256, RoomId(0)).unwrap();
        let err = g.reserve_rect(128, 128, 256, 256, RoomId(1)).unwrap_err();
        assert!(err.to_string().contains("already owned"));
    }

    #[test]
    fn checkpoint_restore_full_cycle() {
        let mut g = make_grid();
        let cp_before = g.checkpoint();

        g.reserve_rect(0, 0, 256, 256, RoomId(0)).unwrap();
        assert!(!g.is_rect_empty(0, 0, 256, 256).unwrap());

        g.restore(cp_before);
        assert!(g.is_rect_empty(0, 0, 256, 256).unwrap());
        assert_eq!(g.owned_cell_count(), 0);
    }

    #[test]
    fn non_overlapping_regions_accepted() {
        let mut g = make_grid();
        g.reserve_rect(0, 0, 256, 256, RoomId(0)).unwrap();
        g.reserve_rect(512, 512, 256, 256, RoomId(1)).unwrap();
        g.reserve_rect(256, 768, 128, 128, RoomId(2)).unwrap();
        assert_eq!(
            g.owned_cell_count(),
            256 / 16 * 256 / 16 * 2 + 128 / 16 * 128 / 16
        );
    }

    #[test]
    fn partial_overlap_rejected() {
        let mut g = make_grid();
        g.reserve_rect(0, 0, 256, 256, RoomId(0)).unwrap();
        // Overlaps by one cell row
        assert!(g.reserve_rect(0, 240, 256, 256, RoomId(1)).is_err());
    }

    #[test]
    fn negative_origin_rejected() {
        let g = make_grid();
        assert!(g.is_rect_empty(-16, 0, 64, 64).is_err());
        assert!(g.is_rect_empty(0, -16, 64, 64).is_err());
    }

    #[test]
    fn non_quantum_rect_rejected() {
        let mut g = make_grid();
        assert!(g.is_rect_empty(1, 0, 64, 64).is_err());
        assert!(g.is_rect_empty(0, 1, 64, 64).is_err());
        assert!(g.reserve_rect(0, 0, 63, 64, RoomId(0)).is_err());
        assert!(g.reserve_rect(0, 0, 64, 63, RoomId(0)).is_err());
    }

    #[test]
    fn out_of_bounds_rejected() {
        let g = make_grid();
        // 1008 + 16 = 1024 — exactly fits at the edge (last cell)
        assert!(g.is_rect_empty(1008, 0, 16, 16).is_ok());
        // 1008 + 32 = 1040 — exceeds grid
        assert!(g.is_rect_empty(1008, 0, 32, 16).is_err());
    }

    #[test]
    fn zero_area_rect_rejected() {
        let mut g = make_grid();
        assert!(g.reserve_rect(0, 0, 0, 64, RoomId(0)).is_err());
        assert!(g.reserve_rect(0, 0, 64, 0, RoomId(0)).is_err());
    }

    #[test]
    fn checkpoint_does_not_affect_live_grid() {
        let mut g = make_grid();
        g.reserve_rect(0, 0, 256, 256, RoomId(0)).unwrap();
        let _cp = g.checkpoint();
        g.reserve_rect(512, 512, 256, 256, RoomId(1)).unwrap();
        assert!(!g.is_rect_empty(512, 512, 256, 256).unwrap());
        // cp is discarded; grid still has both rooms
    }

    #[test]
    fn multiple_checkpoints_independent() {
        let mut g = make_grid();
        let cp0 = g.checkpoint();

        g.reserve_rect(0, 0, 256, 256, RoomId(0)).unwrap();
        let cp1 = g.checkpoint();

        g.reserve_rect(512, 512, 256, 256, RoomId(1)).unwrap();

        // Restore to cp1: only room 0
        g.restore(cp1);
        assert!(!g.is_rect_empty(0, 0, 256, 256).unwrap());
        assert!(g.is_rect_empty(512, 512, 256, 256).unwrap());

        // Restore to cp0: nothing
        g.restore(cp0);
        assert!(g.is_rect_empty(0, 0, 256, 256).unwrap());
    }

    #[test]
    fn large_grid_3072() {
        let g = OccupancyGrid::new(3072, 3072).unwrap();
        assert_eq!(g.cells_x(), 192);
        assert_eq!(g.cells_y(), 192);
        assert_eq!(g.owned_cell_count(), 0);
    }

    #[test]
    fn new_rejects_non_quantum_dimensions() {
        assert!(OccupancyGrid::new(1023, 1024).is_err());
        assert!(OccupancyGrid::new(1024, 1023).is_err());
    }

    #[test]
    fn new_rejects_zero_dimensions() {
        assert!(OccupancyGrid::new(0, 1024).is_err());
        assert!(OccupancyGrid::new(1024, 0).is_err());
    }

    #[test]
    fn sequential_reservations_adjacent() {
        let mut g = make_grid();
        // Two rooms exactly adjacent (touching shells) — allowed
        g.reserve_rect(0, 0, 128, 128, RoomId(0)).unwrap();
        g.reserve_rect(128, 0, 128, 128, RoomId(1)).unwrap();
        // They share a boundary at x=128 but cells are distinct
        assert!(!g.is_rect_empty(0, 0, 128, 128).unwrap());
        assert!(!g.is_rect_empty(128, 0, 128, 128).unwrap());
    }
}
