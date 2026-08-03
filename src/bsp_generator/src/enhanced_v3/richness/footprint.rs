//! 3D occupancy grid with owner-bearing cells and same-XY enforcement.
//!
//! Every occupied cell carries its owning `ReservationId`. Anaonymous cells
//! are forbidden — occupancy is always attributed. Same-XY projected
//! exclusivity is enforced for ordinary reservations; multi-storey composites
//! are the sole exception.
//!
//! # Contract
//!
//! - Construction quantum: 16 Quake units (frozen).
//! - Z bands: lower = 0..192, upper = 192..368 (frozen).
//! - No floats. All coordinates and spans are i32.
//! - Cells are quantum-aligned grid positions.
//! - Occupancy checks are O(1) via dense array (extent / quantum)² × 2 layers.
//! - Transactional: mark / rollback / commit restores byte-identical state.

// Richness remains intentionally crate-private and pipeline-unwired until the
// atomic sealing phase; unit and matrix tests are its current callers.
#![allow(dead_code)]

use std::collections::BTreeMap;

use super::ids::ReservationId;

// ── Frozen constants ───────────────────────────────────────────────────────

/// Construction quantum in Quake units.
pub(crate) const CONSTRUCTION_QUANTUM: i32 = 16;

/// Wall thickness in Quake units.
pub(crate) const WALL_THICKNESS: i32 = 16;

/// Lower floor Z.
pub(crate) const LOWER_FLOOR_Z: i32 = 0;

/// Upper floor Z.
pub(crate) const UPPER_FLOOR_Z: i32 = 192;

/// Room height (both layers).
pub(crate) const ROOM_HEIGHT: i32 = 176;

/// Total Z span.
pub(crate) const TOTAL_Z_SPAN: i32 = 368;

/// Layer count (frozen at 2).
pub(crate) const LAYER_COUNT: u8 = 2;

/// Headroom for routes.
pub(crate) const HEADROOM: i32 = 80;

/// Route width.
pub(crate) const ROUTE_WIDTH: i32 = 64;

/// 64×80 protected route witness.
pub(crate) const ROUTE_WITNESS_WIDTH: i32 = 64;
pub(crate) const ROUTE_WITNESS_HEIGHT: i32 = 80;

// ── Cell coordinate ────────────────────────────────────────────────────────

/// A quantum-aligned grid cell coordinate.
///
/// `x` and `y` are in grid units (each unit = 16 Quake units).
/// `layer` is 0 (lower) or 1 (upper).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct CellCoord {
    /// Grid X (in construction-quantum units).
    pub x: u32,
    /// Grid Y (in construction-quantum units).
    pub y: u32,
    /// Layer: 0 = lower (Z 0..192), 1 = upper (Z 192..368).
    pub layer: u8,
}

impl CellCoord {
    /// Create a new cell coordinate.
    pub const fn new(x: u32, y: u32, layer: u8) -> Self {
        Self { x, y, layer }
    }

    /// Convert from Quake XY to grid coordinates (floor division by quantum).
    pub fn from_quake(x: i32, y: i32, layer: u8) -> Self {
        debug_assert!(x >= 0 && y >= 0);
        Self {
            x: (x / CONSTRUCTION_QUANTUM) as u32,
            y: (y / CONSTRUCTION_QUANTUM) as u32,
            layer,
        }
    }

    /// Quake X at the minimum edge of this cell.
    pub fn quake_x_min(self) -> i32 {
        (self.x as i32) * CONSTRUCTION_QUANTUM
    }

    /// Quake Y at the minimum edge of this cell.
    pub fn quake_y_min(self) -> i32 {
        (self.y as i32) * CONSTRUCTION_QUANTUM
    }

    /// Quake Z at the minimum edge of this cell.
    pub fn quake_z_min(self) -> i32 {
        if self.layer == 0 {
            LOWER_FLOOR_Z
        } else {
            UPPER_FLOOR_Z
        }
    }
}

// ── Occupancy owner ────────────────────────────────────────────────────────

/// The type of reservation that owns a cell.
///
/// Every occupied cell carries one of these variants. There are no anonymous
/// cells in the grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) enum OccupancyOwnerKind {
    /// Standard single-storey room.
    StandardRoom,
    /// Multi-storey room spanning both layers.
    MultiStoreyRoom,
    /// Vertical host (stairwell, ladder shaft, spiral stair, drop hole).
    VerticalHost,
    /// Pit omission (empty volume in upper floor paired with committed lower
    /// room floor as pit bottom).
    PitOmission,
    /// Cave host (protected shell around a cave volume).
    CaveHost,
    /// Route corridor.
    Route,
    /// Portal throat.
    PortalThroat,
    /// Turn / junction.
    Turn,
    /// Spawn point.
    Spawn,
    /// Light entity cell.
    Light,
    /// Structural support.
    Support,
    /// Protected negative space.
    NegativeSpace,
    /// Composite reservation owning same-XY bands.
    Composite,
}

// ── Occupancy cell ─────────────────────────────────────────────────────────

/// A single occupied grid cell.
///
/// Carries its coordinate, owning reservation ID, and owner kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct OccupancyCell {
    /// Grid coordinate.
    pub coord: CellCoord,
    /// The reservation that owns this cell.
    pub owner: ReservationId,
    /// The kind of owner.
    pub owner_kind: OccupancyOwnerKind,
}

// ── Footprint bounds ───────────────────────────────────────────────────────

/// A quantum-aligned 3D bounding box in grid coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct Footprint3D {
    /// Grid X minimum (inclusive).
    pub x0: u32,
    /// Grid Y minimum (inclusive).
    pub y0: u32,
    /// Grid X maximum (exclusive).
    pub x1: u32,
    /// Grid Y maximum (exclusive).
    pub y1: u32,
    /// Lower layer occupancy flag.
    pub occupies_lower: bool,
    /// Upper layer occupancy flag.
    pub occupies_upper: bool,
}

impl Footprint3D {
    /// Create a single-layer footprint.
    pub fn single_layer(qx0: i32, qy0: i32, qx1: i32, qy1: i32, layer: u8) -> Self {
        debug_assert!(qx0 >= 0 && qy0 >= 0 && qx1 > qx0 && qy1 > qy0);
        debug_assert!(qx0 % CONSTRUCTION_QUANTUM == 0);
        debug_assert!(qy0 % CONSTRUCTION_QUANTUM == 0);
        debug_assert!(qx1 % CONSTRUCTION_QUANTUM == 0);
        debug_assert!(qy1 % CONSTRUCTION_QUANTUM == 0);
        Self {
            x0: (qx0 / CONSTRUCTION_QUANTUM) as u32,
            y0: (qy0 / CONSTRUCTION_QUANTUM) as u32,
            x1: (qx1 / CONSTRUCTION_QUANTUM) as u32,
            y1: (qy1 / CONSTRUCTION_QUANTUM) as u32,
            occupies_lower: layer == 0,
            occupies_upper: layer == 1,
        }
    }

    /// Create a dual-layer footprint.
    pub fn dual_layer(qx0: i32, qy0: i32, qx1: i32, qy1: i32) -> Self {
        debug_assert!(qx0 >= 0 && qy0 >= 0 && qx1 > qx0 && qy1 > qy0);
        debug_assert!(qx0 % CONSTRUCTION_QUANTUM == 0);
        debug_assert!(qy0 % CONSTRUCTION_QUANTUM == 0);
        debug_assert!(qx1 % CONSTRUCTION_QUANTUM == 0);
        debug_assert!(qy1 % CONSTRUCTION_QUANTUM == 0);
        Self {
            x0: (qx0 / CONSTRUCTION_QUANTUM) as u32,
            y0: (qy0 / CONSTRUCTION_QUANTUM) as u32,
            x1: (qx1 / CONSTRUCTION_QUANTUM) as u32,
            y1: (qy1 / CONSTRUCTION_QUANTUM) as u32,
            occupies_lower: true,
            occupies_upper: true,
        }
    }

    /// Iterate all cell coordinates in this footprint.
    pub fn cells(&self) -> Vec<CellCoord> {
        let mut result = Vec::with_capacity(self.cell_count());
        for x in self.x0..self.x1 {
            for y in self.y0..self.y1 {
                if self.occupies_lower {
                    result.push(CellCoord::new(x, y, 0));
                }
                if self.occupies_upper {
                    result.push(CellCoord::new(x, y, 1));
                }
            }
        }
        result
    }

    /// Cell count (both layers).
    pub fn cell_count(&self) -> usize {
        let layers = if self.occupies_lower && self.occupies_upper {
            2usize
        } else if self.occupies_lower || self.occupies_upper {
            1
        } else {
            0
        };
        let w = (self.x1 - self.x0) as usize;
        let h = (self.y1 - self.y0) as usize;
        w * h * layers
    }

    /// Width in grid cells.
    pub fn grid_width(&self) -> u32 {
        self.x1 - self.x0
    }

    /// Depth in grid cells.
    pub fn grid_depth(&self) -> u32 {
        self.y1 - self.y0
    }

    /// Quake-unit span.
    pub fn quake_span(&self) -> (i32, i32) {
        (
            (self.x1 - self.x0) as i32 * CONSTRUCTION_QUANTUM,
            (self.y1 - self.y0) as i32 * CONSTRUCTION_QUANTUM,
        )
    }

    /// Whether this footprint overlaps another in XY projection.
    pub fn overlaps_xy(&self, other: &Self) -> bool {
        self.x0 < other.x1 && self.x1 > other.x0 && self.y0 < other.y1 && self.y1 > other.y0
    }

    /// Whether this footprint contains the given XY grid cell.
    pub fn contains_xy(&self, x: u32, y: u32) -> bool {
        x >= self.x0 && x < self.x1 && y >= self.y0 && y < self.y1
    }

    /// Expand by `margin` grid cells on all sides.
    pub fn expanded(&self, margin: u32) -> Self {
        Self {
            x0: self.x0.saturating_sub(margin),
            y0: self.y0.saturating_sub(margin),
            x1: self.x1.saturating_add(margin),
            y1: self.y1.saturating_add(margin),
            occupies_lower: self.occupies_lower,
            occupies_upper: self.occupies_upper,
        }
    }
}

// ── Occupancy grid ─────────────────────────────────────────────────────────

/// The 3D occupancy grid.
///
/// Stores a dense array of `Option<OccupancyCell>` indexed by grid
/// coordinate. Supports transactional mark/rollback/commit. Enforces
/// same-XY exclusivity for non-composite reservations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct OccupancyGrid {
    /// Grid width in cells (extent / quantum).
    grid_w: u32,
    /// Grid depth in cells (extent / quantum).
    grid_h: u32,
    /// Dense cell array: [layer][y][x].
    cells: Vec<Vec<Vec<Option<OccupancyCell>>>>,
    /// Number of occupied cells.
    occupied_count: usize,
    /// Composite reservation footprints (for same-XY exemption).
    composite_footprints: BTreeMap<ReservationId, Footprint3D>,
    /// Checkpoint storage for rollback.
    checkpoints: Vec<OccupancyGridCheckpoint>,
}

/// A stored snapshot for rollback.
#[derive(Debug, Clone, PartialEq, Eq)]
struct OccupancyGridCheckpoint {
    /// Serialized cell state at checkpoint time.
    cells: Vec<Vec<Vec<Option<OccupancyCell>>>>,
    /// Serialized occupied count.
    occupied_count: usize,
    /// Serialized composite footprints.
    composite_footprints: BTreeMap<ReservationId, Footprint3D>,
}

impl OccupancyGrid {
    /// Create a new empty grid for the given Quake extent.
    pub fn new(extent: u32) -> Self {
        debug_assert!(extent.is_multiple_of(CONSTRUCTION_QUANTUM as u32));
        let grid_w = extent / CONSTRUCTION_QUANTUM as u32;
        let grid_h = extent / CONSTRUCTION_QUANTUM as u32;
        let cells = vec![vec![vec![None; grid_w as usize]; grid_h as usize]; LAYER_COUNT as usize];
        Self {
            grid_w,
            grid_h,
            cells,
            occupied_count: 0,
            composite_footprints: BTreeMap::new(),
            checkpoints: Vec::new(),
        }
    }

    /// Grid width in cells.
    pub fn grid_width(&self) -> u32 {
        self.grid_w
    }

    /// Grid depth in cells.
    pub fn grid_height(&self) -> u32 {
        self.grid_h
    }

    /// Total number of occupied cells.
    pub fn occupied_count(&self) -> usize {
        self.occupied_count
    }

    /// Check if a cell is occupied.
    pub fn is_occupied(&self, coord: CellCoord) -> bool {
        self.in_bounds(coord)
            && self.cells[coord.layer as usize][coord.y as usize][coord.x as usize].is_some()
    }

    /// Get the occupancy cell at a coordinate, if any.
    pub fn get(&self, coord: CellCoord) -> Option<&OccupancyCell> {
        if !self.in_bounds(coord) {
            return None;
        }
        self.cells[coord.layer as usize][coord.y as usize][coord.x as usize].as_ref()
    }

    /// Check if a footprint can be reserved (no same-XY overlap outside an
    /// explicit composite container).
    pub fn can_reserve(
        &self,
        footprint: &Footprint3D,
        owner: ReservationId,
        owner_kind: OccupancyOwnerKind,
    ) -> Result<(), OccupancyError> {
        let is_composite = owner_kind == OccupancyOwnerKind::Composite;

        for coord in footprint.cells() {
            if !self.in_bounds(coord) {
                return Err(OccupancyError::OutOfBounds {
                    coord,
                    grid_w: self.grid_w,
                    grid_h: self.grid_h,
                });
            }

            if let Some(existing) = self.get(coord) {
                // Same cell, same owner — allowed (can happen during composite child placement)
                if existing.owner == owner {
                    continue;
                }

                // A composite container is the sole authorization for an
                // overlapping child. A newly placed composite never overlaps
                // an existing owner: it must reserve both bands atomically.
                let composite_child = matches!(
                    owner_kind,
                    OccupancyOwnerKind::StandardRoom
                        | OccupancyOwnerKind::MultiStoreyRoom
                        | OccupancyOwnerKind::VerticalHost
                        | OccupancyOwnerKind::PitOmission
                        | OccupancyOwnerKind::PortalThroat
                        | OccupancyOwnerKind::Turn
                        | OccupancyOwnerKind::Spawn
                        | OccupancyOwnerKind::Light
                        | OccupancyOwnerKind::Support
                );
                let ordinary_room_child = matches!(
                    existing.owner_kind,
                    OccupancyOwnerKind::StandardRoom
                        | OccupancyOwnerKind::CaveHost
                        | OccupancyOwnerKind::NegativeSpace
                ) && matches!(
                    owner_kind,
                    OccupancyOwnerKind::PortalThroat
                        | OccupancyOwnerKind::Turn
                        | OccupancyOwnerKind::Spawn
                        | OccupancyOwnerKind::Light
                        | OccupancyOwnerKind::Support
                );
                if (!is_composite
                    && existing.owner_kind == OccupancyOwnerKind::Composite
                    && composite_child)
                    || ordinary_room_child
                {
                    continue;
                }

                // Same-XY check for different layers
                if coord.layer != existing.coord.layer {
                    return Err(OccupancyError::SameXYConflict {
                        coord,
                        existing_owner: existing.owner,
                        new_owner: owner,
                    });
                }

                // Same cell occupied by a different owner — conflict
                return Err(OccupancyError::CellAlreadyOccupied {
                    coord,
                    existing_owner: existing.owner,
                    new_owner: owner,
                });
            }

            // Cross-layer same-XY check for non-composite. Pit omissions
            // receive no special exemption: their paired room must be a child
            // of the same explicit composite container.
            if !is_composite {
                let other_layer = if coord.layer == 0 { 1u8 } else { 0u8 };
                let other_coord = CellCoord::new(coord.x, coord.y, other_layer);
                if let Some(other) = self.get(other_coord) {
                    if other.owner != owner && other.owner_kind != OccupancyOwnerKind::Composite {
                        return Err(OccupancyError::SameXYConflict {
                            coord: other_coord,
                            existing_owner: other.owner,
                            new_owner: owner,
                        });
                    }
                }
            }
        }
        Ok(())
    }

    /// Reserve a footprint for an owner. The caller must have already
    /// validated with `can_reserve`.
    pub fn reserve(
        &mut self,
        footprint: &Footprint3D,
        owner: ReservationId,
        owner_kind: OccupancyOwnerKind,
    ) {
        if owner_kind == OccupancyOwnerKind::Composite {
            self.composite_footprints.insert(owner, *footprint);
        }

        for coord in footprint.cells() {
            if self.in_bounds(coord) {
                let cell = OccupancyCell {
                    coord,
                    owner,
                    owner_kind,
                };
                let slot =
                    &mut self.cells[coord.layer as usize][coord.y as usize][coord.x as usize];
                if slot.is_none() {
                    self.occupied_count += 1;
                    *slot = Some(cell);
                } else if slot
                    .as_ref()
                    .is_some_and(|existing| existing.owner_kind != OccupancyOwnerKind::Composite)
                {
                    *slot = Some(cell);
                }
            }
        }
    }

    /// Release all cells owned by a reservation.
    pub fn release(&mut self, owner: ReservationId) {
        self.composite_footprints.remove(&owner);
        for layer in 0..LAYER_COUNT as usize {
            for y in 0..self.grid_h as usize {
                for x in 0..self.grid_w as usize {
                    if let Some(cell) = &self.cells[layer][y][x] {
                        if cell.owner == owner {
                            self.cells[layer][y][x] = None;
                            self.occupied_count = self.occupied_count.saturating_sub(1);
                        }
                    }
                }
            }
        }
    }

    /// Mark a checkpoint for rollback.
    pub fn mark(&mut self) {
        self.checkpoints.push(OccupancyGridCheckpoint {
            cells: self.cells.clone(),
            occupied_count: self.occupied_count,
            composite_footprints: self.composite_footprints.clone(),
        });
    }

    /// Rollback to the most recent checkpoint. Returns `true` if a
    /// checkpoint was available.
    pub fn rollback(&mut self) -> bool {
        if let Some(cp) = self.checkpoints.pop() {
            self.cells = cp.cells;
            self.occupied_count = cp.occupied_count;
            self.composite_footprints = cp.composite_footprints;
            true
        } else {
            false
        }
    }

    /// Commit the most recent checkpoint (discard it without rollback).
    /// Returns `true` if a checkpoint was available.
    pub fn commit(&mut self) -> bool {
        self.checkpoints.pop().is_some()
    }

    /// Discard all checkpoints.
    pub fn clear_checkpoints(&mut self) {
        self.checkpoints.clear();
    }

    /// Snapshot the full grid state for byte-identity comparisons.
    pub fn state_snapshot(&self) -> OccupancyGridSnapshot {
        OccupancyGridSnapshot {
            cells: self.cells.clone(),
            occupied_count: self.occupied_count,
            composite_footprints: self.composite_footprints.clone(),
            checkpoints: self.checkpoints.clone(),
        }
    }

    /// Restore from a snapshot (for byte-identity tests).
    pub fn restore_snapshot(&mut self, snapshot: &OccupancyGridSnapshot) {
        self.cells = snapshot.cells.clone();
        self.occupied_count = snapshot.occupied_count;
        self.composite_footprints = snapshot.composite_footprints.clone();
        self.checkpoints = snapshot.checkpoints.clone();
    }

    /// Whether the coordinate is in bounds.
    fn in_bounds(&self, coord: CellCoord) -> bool {
        coord.x < self.grid_w && coord.y < self.grid_h && coord.layer < LAYER_COUNT
    }
}

// ── Occupancy grid snapshot ────────────────────────────────────────────────

/// A complete snapshot of the occupancy grid state for byte-identity testing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct OccupancyGridSnapshot {
    cells: Vec<Vec<Vec<Option<OccupancyCell>>>>,
    occupied_count: usize,
    composite_footprints: BTreeMap<ReservationId, Footprint3D>,
    checkpoints: Vec<OccupancyGridCheckpoint>,
}

impl OccupancyGridSnapshot {
    /// Returns the number of occupied cells.
    pub fn occupied_count(&self) -> usize {
        self.occupied_count
    }

    /// Returns `true` if this snapshot has the same occupied count as another.
    pub fn same_occupied_count(&self, other: &Self) -> bool {
        self.occupied_count == other.occupied_count
    }
}

// ── Occupancy error ────────────────────────────────────────────────────────

/// Errors from the occupancy grid.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum OccupancyError {
    /// Coordinate out of grid bounds.
    OutOfBounds {
        coord: CellCoord,
        grid_w: u32,
        grid_h: u32,
    },
    /// Cell already occupied by a different reservation.
    CellAlreadyOccupied {
        coord: CellCoord,
        existing_owner: ReservationId,
        new_owner: ReservationId,
    },
    /// Same-XY conflict: two non-composite owners at the same XY.
    SameXYConflict {
        coord: CellCoord,
        existing_owner: ReservationId,
        new_owner: ReservationId,
    },
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_grid(extent: u32) -> OccupancyGrid {
        OccupancyGrid::new(extent)
    }

    #[test]
    fn grid_creation() {
        let grid = make_grid(1024);
        assert_eq!(grid.grid_width(), 64); // 1024/16
        assert_eq!(grid.grid_height(), 64);
        assert_eq!(grid.occupied_count(), 0);
    }

    #[test]
    fn cell_coord_from_quake() {
        let c = CellCoord::from_quake(64, 80, 0);
        assert_eq!(c.x, 4);
        assert_eq!(c.y, 5);
        assert_eq!(c.layer, 0);
        assert_eq!(c.quake_x_min(), 64);
        assert_eq!(c.quake_y_min(), 80);
        assert_eq!(c.quake_z_min(), LOWER_FLOOR_Z);
    }

    #[test]
    fn cell_coord_upper_layer_z() {
        let c = CellCoord::new(0, 0, 1);
        assert_eq!(c.quake_z_min(), UPPER_FLOOR_Z);
    }

    #[test]
    fn footprint_single_layer() {
        let fp = Footprint3D::single_layer(0, 0, 64, 64, 0);
        assert!(fp.occupies_lower);
        assert!(!fp.occupies_upper);
        assert_eq!(fp.grid_width(), 4);
        assert_eq!(fp.grid_depth(), 4);
        assert_eq!(fp.cell_count(), 16); // 4*4*1
    }

    #[test]
    fn footprint_dual_layer() {
        let fp = Footprint3D::dual_layer(0, 0, 64, 64);
        assert!(fp.occupies_lower);
        assert!(fp.occupies_upper);
        assert_eq!(fp.cell_count(), 32); // 4*4*2
    }

    #[test]
    fn footprint_overlaps_xy() {
        let a = Footprint3D::single_layer(0, 0, 64, 64, 0);
        let b = Footprint3D::single_layer(32, 32, 96, 96, 0);
        let c = Footprint3D::single_layer(64, 0, 128, 64, 0);

        assert!(a.overlaps_xy(&b));
        assert!(!a.overlaps_xy(&c));
        assert!(b.overlaps_xy(&c));
    }

    #[test]
    fn reserve_and_query() {
        let mut grid = make_grid(1024);
        let owner = ReservationId::new(0);
        let fp = Footprint3D::single_layer(0, 0, 64, 64, 0);

        assert!(grid
            .can_reserve(&fp, owner, OccupancyOwnerKind::StandardRoom)
            .is_ok());
        grid.reserve(&fp, owner, OccupancyOwnerKind::StandardRoom);
        assert_eq!(grid.occupied_count(), 16);

        // Query a cell
        let coord = CellCoord::from_quake(16, 16, 0);
        let cell = grid.get(coord);
        assert!(cell.is_some());
        assert_eq!(cell.unwrap().owner, owner);
        assert_eq!(cell.unwrap().owner_kind, OccupancyOwnerKind::StandardRoom);
    }

    #[test]
    fn cannot_double_occupy() {
        let mut grid = make_grid(1024);
        let a = ReservationId::new(0);
        let b = ReservationId::new(1);
        let fp = Footprint3D::single_layer(0, 0, 64, 64, 0);

        grid.reserve(&fp, a, OccupancyOwnerKind::StandardRoom);
        let result = grid.can_reserve(&fp, b, OccupancyOwnerKind::StandardRoom);
        assert!(result.is_err());
        match result {
            Err(OccupancyError::CellAlreadyOccupied { .. }) => {}
            _ => panic!("expected CellAlreadyOccupied"),
        }
    }

    #[test]
    fn same_xy_conflict_outside_composite() {
        let mut grid = make_grid(1024);
        let a = ReservationId::new(0);
        let b = ReservationId::new(1);

        let lower = Footprint3D::single_layer(0, 0, 64, 64, 0);
        let upper = Footprint3D::single_layer(0, 0, 64, 64, 1);

        grid.reserve(&lower, a, OccupancyOwnerKind::StandardRoom);
        let result = grid.can_reserve(&upper, b, OccupancyOwnerKind::StandardRoom);
        assert!(
            result.is_err(),
            "same-XY should be rejected for different non-composite owners"
        );
    }

    #[test]
    fn same_xy_allowed_inside_composite() {
        let mut grid = make_grid(1024);
        let composite = ReservationId::new(0);
        let fp = Footprint3D::dual_layer(0, 0, 64, 64);

        // Composite reservation owns both layers (container for multi-storey)
        grid.reserve(&fp, composite, OccupancyOwnerKind::Composite);

        // A standard room placed within a composite container should be allowed.
        // The composite authorizes same-XY multi-storey occupancy.
        let room = ReservationId::new(1);
        let lower = Footprint3D::single_layer(0, 0, 32, 32, 0);
        let result = grid.can_reserve(&lower, room, OccupancyOwnerKind::StandardRoom);
        assert!(
            result.is_ok(),
            "composite container should allow child placement: {:?}",
            result.err()
        );
    }

    #[test]
    fn pit_pair_without_composite_is_rejected() {
        let mut grid = make_grid(1024);
        let room = ReservationId::new(0);
        let pit = ReservationId::new(1);

        let lower = Footprint3D::single_layer(0, 0, 64, 64, 0);
        let upper = Footprint3D::single_layer(0, 0, 64, 64, 1);

        grid.reserve(&lower, room, OccupancyOwnerKind::StandardRoom);
        let result = grid.can_reserve(&upper, pit, OccupancyOwnerKind::PitOmission);
        assert!(
            result.is_err(),
            "a pit pair without an explicit composite owner must be rejected"
        );
    }

    #[test]
    fn transactional_mark_rollback() {
        let mut grid = make_grid(1024);
        let before = grid.state_snapshot();

        grid.mark();
        let owner = ReservationId::new(0);
        let fp = Footprint3D::single_layer(0, 0, 64, 64, 0);
        grid.reserve(&fp, owner, OccupancyOwnerKind::StandardRoom);
        assert!(grid.occupied_count() > 0);

        // Rollback restores byte-identical state
        let rolled = grid.rollback();
        assert!(rolled);
        assert_eq!(grid.occupied_count(), 0);
        let after = grid.state_snapshot();
        assert_eq!(before, after, "rollback must restore byte-identical state");
    }

    #[test]
    fn transactional_mark_commit() {
        let mut grid = make_grid(1024);

        grid.mark();
        let owner = ReservationId::new(0);
        let fp = Footprint3D::single_layer(0, 0, 64, 64, 0);
        grid.reserve(&fp, owner, OccupancyOwnerKind::StandardRoom);
        let committed = grid.commit();
        assert!(committed);
        let after_commit = grid.state_snapshot();

        // Rollback should now fail (no checkpoint)
        let rolled = grid.rollback();
        assert!(!rolled);

        // State should still be the committed reserved state.  The mark is
        // intentionally absent from this snapshot because commit consumes it.
        assert_eq!(grid.state_snapshot(), after_commit);
    }

    #[test]
    fn nested_checkpoints() {
        let mut grid = make_grid(1024);

        grid.mark(); // checkpoint 0
        let r0 = ReservationId::new(0);
        grid.reserve(
            &Footprint3D::single_layer(0, 0, 32, 32, 0),
            r0,
            OccupancyOwnerKind::StandardRoom,
        );

        grid.mark(); // checkpoint 1
        let r1 = ReservationId::new(1);
        grid.reserve(
            &Footprint3D::single_layer(64, 64, 96, 96, 0),
            r1,
            OccupancyOwnerKind::StandardRoom,
        );

        // Rollback to checkpoint 1 (undo r1)
        assert!(grid.rollback());
        assert!(grid.get(CellCoord::from_quake(16, 16, 0)).is_some()); // r0 still there
        assert!(grid.get(CellCoord::from_quake(80, 80, 0)).is_none()); // r1 gone

        // Rollback to checkpoint 0 (undo r0)
        assert!(grid.rollback());
        assert_eq!(grid.occupied_count(), 0);
    }

    #[test]
    fn release_reservation() {
        let mut grid = make_grid(1024);
        let owner = ReservationId::new(0);
        let fp = Footprint3D::single_layer(0, 0, 64, 64, 0);
        grid.reserve(&fp, owner, OccupancyOwnerKind::StandardRoom);
        assert_eq!(grid.occupied_count(), 16);

        grid.release(owner);
        assert_eq!(grid.occupied_count(), 0);
    }

    #[test]
    fn out_of_bounds_footprint() {
        let grid = make_grid(1024); // grid_w = 64
                                    // Try to reserve beyond extent
        let fp = Footprint3D::single_layer(1008, 1008, 1040, 1040, 0); // 1008/16=63, 1040/16=65
        let result = grid.can_reserve(&fp, ReservationId::new(0), OccupancyOwnerKind::StandardRoom);
        assert!(result.is_err());
        match result {
            Err(OccupancyError::OutOfBounds { .. }) => {}
            _ => panic!("expected OutOfBounds"),
        }
    }

    #[test]
    fn worst_case_envelope_fits_max_span() {
        // Verify the largest archetype (grand_arena, 448×448) fits within
        // the minimum extent boundary. grid = 448/16 = 28 cells.
        let extent = 1024u32;
        let grid = make_grid(extent);
        let fp = Footprint3D::single_layer(0, 0, 448, 448, 0);
        assert!(grid
            .can_reserve(&fp, ReservationId::new(0), OccupancyOwnerKind::StandardRoom)
            .is_ok());
    }

    #[test]
    fn route_witness_envelope() {
        let mut grid = make_grid(1024);
        let owner = ReservationId::new(0);
        // Route witness: 64 units wide, 80 units tall in a cell footprint
        // 64/16=4 cells wide, 80/16=5 cells tall -> 20 cells
        let fp = Footprint3D::single_layer(0, 0, ROUTE_WITNESS_WIDTH, ROUTE_WITNESS_HEIGHT, 0);
        assert_eq!(fp.grid_width(), 4);
        assert_eq!(fp.grid_depth(), 5);
        assert!(grid
            .can_reserve(&fp, owner, OccupancyOwnerKind::Route)
            .is_ok());
        grid.reserve(&fp, owner, OccupancyOwnerKind::Route);
        assert_eq!(grid.occupied_count(), 20);
    }

    #[test]
    fn footprint_cells_iteration() {
        let fp = Footprint3D::single_layer(0, 0, 32, 48, 0);
        let cells = fp.cells();
        assert_eq!(cells.len(), 6); // 2 grid x * 3 grid y * 1 layer = 6

        let dual = Footprint3D::dual_layer(0, 0, 32, 48);
        let dual_cells = dual.cells();
        assert_eq!(dual_cells.len(), 12); // 2*3*2 = 12
    }
}
