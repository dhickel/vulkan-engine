use std::collections::BTreeMap;
use std::fmt;

use super::config::NormalizedGeneratorConfig;
use super::error::{ErrorStage, GeneratorError};

// ─── Checked grid coordinate ───────────────────────────────────────────────

/// Checked three-dimensional integer coordinate with layer, x, y components
/// and a reversible index within the configured grid dimensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) struct GridCoord {
    pub(super) layer: u16,
    pub(super) x: u16,
    pub(super) y: u16,
}

impl GridCoord {
    /// Construct a coordinate, rejecting out-of-bounds components.
    pub(super) fn new(
        layer: u16,
        x: u16,
        y: u16,
        width: u16,
        height: u16,
        layers: u16,
    ) -> Result<Self, GeneratorError> {
        if layer >= layers {
            return Err(GeneratorError::IrInvariant {
                stage: ErrorStage::Ir,
                detail: format!(
                    "coord_layer_out_of_bounds layer={} max={}",
                    layer,
                    layers.saturating_sub(1)
                ),
            });
        }
        if x >= width {
            return Err(GeneratorError::IrInvariant {
                stage: ErrorStage::Ir,
                detail: format!("coord_x_out_of_bounds x={} max={}", x, width.saturating_sub(1)),
            });
        }
        if y >= height {
            return Err(GeneratorError::IrInvariant {
                stage: ErrorStage::Ir,
                detail: format!(
                    "coord_y_out_of_bounds y={} max={}",
                    y,
                    height.saturating_sub(1)
                ),
            });
        }
        Ok(Self { layer, x, y })
    }

    /// Convert to a flat index: `layer * width * height + y * width + x`.
    /// Checks for overflow at each multiplication and addition step.
    pub(super) fn to_flat_index(self, width: u16, height: u16) -> Result<usize, GeneratorError> {
        let w = u64::from(width);
        let h = u64::from(height);
        let layer_dim = w.checked_mul(h).ok_or_else(|| GeneratorError::ArithmeticOverflow {
            stage: ErrorStage::Ir,
            operation: "grid_index_layer_dim",
        })?;
        let layer_offset = u64::from(self.layer)
            .checked_mul(layer_dim)
            .ok_or_else(|| GeneratorError::ArithmeticOverflow {
                stage: ErrorStage::Ir,
                operation: "grid_index_layer_offset",
            })?;
        let row_offset = u64::from(self.y)
            .checked_mul(w)
            .ok_or_else(|| GeneratorError::ArithmeticOverflow {
                stage: ErrorStage::Ir,
                operation: "grid_index_row_offset",
            })?;
        let idx = layer_offset
            .checked_add(row_offset)
            .and_then(|v| v.checked_add(u64::from(self.x)))
            .ok_or_else(|| GeneratorError::ArithmeticOverflow {
                stage: ErrorStage::Ir,
                operation: "grid_index_final",
            })?;
        usize::try_from(idx).map_err(|_| GeneratorError::ArithmeticOverflow {
            stage: ErrorStage::Ir,
            operation: "grid_index_usize_convert",
        })
    }

    /// Reconstruct from flat index: `layer = idx / (w*h)`, `remainder = idx % (w*h)`,
    /// `y = remainder / w`, `x = remainder % w`.
    /// Returns error on out-of-range or zero-dimension.
    pub(super) fn from_flat_index(
        idx: usize,
        width: u16,
        height: u16,
        layers: u16,
    ) -> Result<Self, GeneratorError> {
        // Guard against zero dimensions which would cause division by zero.
        if width == 0 || height == 0 || layers == 0 {
            return Err(GeneratorError::IrInvariant {
                stage: ErrorStage::Ir,
                detail: format!(
                    "from_flat_index_zero_dimensions w={} h={} l={}",
                    width, height, layers
                ),
            });
        }
        let idx = u64::try_from(idx).map_err(|_| GeneratorError::ArithmeticOverflow {
            stage: ErrorStage::Ir,
            operation: "grid_index_u64_convert",
        })?;
        let w = u64::from(width);
        let h = u64::from(height);
        let layer_dim = w.checked_mul(h).ok_or_else(|| GeneratorError::ArithmeticOverflow {
            stage: ErrorStage::Ir,
            operation: "grid_index_layer_dim_inverse",
        })?;
        // The division by layer_dim is safe because layer_dim > 0 (width>0 && height>0).
        let max_idx = u64::from(layers)
            .checked_mul(layer_dim)
            .ok_or_else(|| GeneratorError::ArithmeticOverflow {
                stage: ErrorStage::Ir,
                operation: "grid_index_max_idx",
            })?;
        if idx >= max_idx {
            return Err(GeneratorError::IrInvariant {
                stage: ErrorStage::Ir,
                detail: format!("flat_index_out_of_range idx={} max={}", idx, max_idx),
            });
        }
        let layer = (idx / layer_dim) as u16;
        let remainder = idx % layer_dim;
        // Division by w is safe because w > 0 (width > 0 guarded above)
        let y = (remainder / w) as u16;
        let x = (remainder % w) as u16;
        Ok(Self { layer, x, y })
    }
}

impl fmt::Display for GridCoord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {}, {})", self.layer, self.x, self.y)
    }
}

// ─── Attempt-local ID allocator ─────────────────────────────────────────────

/// An attempt-local checked allocator for typed IDs.
/// No global atomics, no panics. Returns errors on overflow,
/// which can never happen in practice with real profiles.
#[derive(Debug, Clone)]
pub(super) struct IdAllocator {
    next_region: u32,
    next_socket: u32,
    next_edge: u32,
    next_transition: u32,
}

impl IdAllocator {
    /// Create a fresh allocator for a generation attempt.
    pub(super) fn new() -> Self {
        Self {
            next_region: 0,
            next_socket: 0,
            next_edge: 0,
            next_transition: 0,
        }
    }

    pub(super) fn next_region(&mut self) -> Result<RegionId, GeneratorError> {
        let id = self.next_region;
        // Allow allocating u32::MAX; subsequent allocation returns error.
        self.next_region = self.next_region.checked_add(1).ok_or_else(|| {
            GeneratorError::ArithmeticOverflow {
                stage: ErrorStage::Ir,
                operation: "region_id_overflow",
            }
        })?;
        Ok(RegionId(id))
    }

    pub(super) fn next_socket(&mut self) -> Result<SocketId, GeneratorError> {
        let id = self.next_socket;
        self.next_socket = self.next_socket.checked_add(1).ok_or_else(|| {
            GeneratorError::ArithmeticOverflow {
                stage: ErrorStage::Ir,
                operation: "socket_id_overflow",
            }
        })?;
        Ok(SocketId(id))
    }

    pub(super) fn next_edge(&mut self) -> Result<EdgeId, GeneratorError> {
        let id = self.next_edge;
        self.next_edge = self.next_edge.checked_add(1).ok_or_else(|| {
            GeneratorError::ArithmeticOverflow {
                stage: ErrorStage::Ir,
                operation: "edge_id_overflow",
            }
        })?;
        Ok(EdgeId(id))
    }

    pub(super) fn next_transition(&mut self) -> Result<TransitionId, GeneratorError> {
        let id = self.next_transition;
        self.next_transition = self.next_transition.checked_add(1).ok_or_else(|| {
            GeneratorError::ArithmeticOverflow {
                stage: ErrorStage::Ir,
                operation: "transition_id_overflow",
            }
        })?;
        Ok(TransitionId(id))
    }
}

// ─── Newtype IDs ────────────────────────────────────────────────────────────

/// Stable typed newtype ID for regions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) struct RegionId(pub(super) u32);

impl RegionId {
    pub(super) const fn raw(self) -> u32 {
        self.0
    }
}

impl fmt::Display for RegionId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RegionId({})", self.0)
    }
}

/// Stable typed newtype ID for sockets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) struct SocketId(pub(super) u32);

impl SocketId {
    pub(super) const fn raw(self) -> u32 {
        self.0
    }
}

impl fmt::Display for SocketId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SocketId({})", self.0)
    }
}

/// Stable typed newtype ID for edges.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) struct EdgeId(pub(super) u32);

impl EdgeId {
    pub(super) const fn raw(self) -> u32 {
        self.0
    }
}

impl fmt::Display for EdgeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "EdgeId({})", self.0)
    }
}

/// Stable typed newtype ID for transitions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) struct TransitionId(pub(super) u32);

impl TransitionId {
    pub(super) const fn raw(self) -> u32 {
        self.0
    }
}

impl fmt::Display for TransitionId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TransitionId({})", self.0)
    }
}

// ─── Region role ────────────────────────────────────────────────────────────

/// The topological function of a dungeon region in the generated level graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) enum RegionRole {
    /// Player entry region — exactly one per level.
    Spawn,
    /// A visually/thematically prominent distant goal region — exactly one.
    DistantLandmark,
    /// A major spatial anchor or view-chamber — configurable count.
    MajorLandmark,
    /// A region that joins multiple corridor/hall branches.
    Junction,
    /// A region intentionally placed at a branch terminus.
    DeadEnd,
    /// A region containing a vertical ramp between adjacent layers.
    VerticalHub,
    /// A region that must lie on at least one edge-disjoint route.
    RequiredRoute,
    /// A region connected to the graph as an optional detour branch.
    OptionalBranch,
    /// A plain room with no special topological obligation.
    OrdinaryRoom,
}

impl RegionRole {
    /// Canonical ordinal for stable sorting.
    pub(super) const fn ordinal(self) -> u8 {
        match self {
            Self::Spawn => 0,
            Self::DistantLandmark => 1,
            Self::MajorLandmark => 2,
            Self::Junction => 3,
            Self::DeadEnd => 4,
            Self::VerticalHub => 5,
            Self::RequiredRoute => 6,
            Self::OptionalBranch => 7,
            Self::OrdinaryRoom => 8,
        }
    }

    pub(super) const fn label(self) -> &'static str {
        match self {
            Self::Spawn => "spawn",
            Self::DistantLandmark => "distant_landmark",
            Self::MajorLandmark => "major_landmark",
            Self::Junction => "junction",
            Self::DeadEnd => "dead_end",
            Self::VerticalHub => "vertical_hub",
            Self::RequiredRoute => "required_route",
            Self::OptionalBranch => "optional_branch",
            Self::OrdinaryRoom => "ordinary_room",
        }
    }
}

// ─── Occupancy grid ─────────────────────────────────────────────────────────

/// What owns a reserved cell in the occupancy grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum OccupancyClass {
    /// Not yet occupied.
    Empty,
    /// Projection of a transition prefab's hub footprint. This is not ordinary
    /// floor/wall occupancy: only endpoint regions owned by this transition may
    /// be placed over it.
    TransitionHub(u32),
    /// Reserved ramp run, upper opening, landing, or headroom cell.
    Transition(u32),
    /// Reserved by a placed region footprint.
    Region(u32),
    /// Reserved as a socket funnel / approach corridor approach.
    Socket(u32),
    /// Spacing cushion around a placed footprint.
    Spacing(u32),
    /// Border wall cell (outer perimeter of each layer).
    Border,
}

/// Occupancy grid tracking per-cell ownership across all layers.
///
/// Backed by a flat vector per layer for fast probing; coordinates are
/// clamped to grid bounds through the `GridCoord` constructor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct OccupancyGrid {
    width: u16,
    height: u16,
    layers: u16,
    pub(super) cells: Vec<OccupancyClass>,
}

impl OccupancyGrid {
    /// Create a new occupancy grid. Returns error on overflow or zero dimensions.
    pub(super) fn new(
        width: u16,
        height: u16,
        layers: u16,
    ) -> Result<Self, GeneratorError> {
        if width == 0 || height == 0 || layers == 0 {
            return Err(GeneratorError::IrInvariant {
                stage: ErrorStage::Ir,
                detail: format!(
                    "occupancy_grid_zero_dimensions w={} h={} l={}",
                    width, height, layers
                ),
            });
        }
        let capacity = (width as usize)
            .checked_mul(height as usize)
            .and_then(|v| v.checked_mul(layers as usize))
            .ok_or_else(|| GeneratorError::ArithmeticOverflow {
                stage: ErrorStage::Ir,
                operation: "occupancy_grid_capacity",
            })?;
        Ok(Self {
            width,
            height,
            layers,
            cells: vec![OccupancyClass::Empty; capacity],
        })
    }

    #[allow(dead_code)]
    pub(super) const fn dimensions(&self) -> (u16, u16, u16) {
        (self.width, self.height, self.layers)
    }

    /// Reserve border cells on every layer as Border occupancy.
    /// These are never walkable and corridors may not route through them.
    pub(super) fn reserve_borders(&mut self) {
        let w = self.width as usize;
        let h = self.height as usize;
        for layer in 0..self.layers as usize {
            let base = layer * w * h;
            for x in 0..w {
                self.cells[base + x] = OccupancyClass::Border; // y=0
                self.cells[base + (h - 1) * w + x] = OccupancyClass::Border; // y=last
            }
            for y in 0..h {
                self.cells[base + y * w] = OccupancyClass::Border; // x=0
                self.cells[base + y * w + (w - 1)] = OccupancyClass::Border; // x=last
            }
        }
    }

    fn flat_index(&self, coord: GridCoord) -> Result<usize, GeneratorError> {
        coord.to_flat_index(self.width, self.height)
    }

    /// Read the occupancy class at a coordinate. Returns None for out-of-bounds.
    pub(super) fn get(&self, coord: GridCoord) -> Option<OccupancyClass> {
        let idx = self.flat_index(coord).ok()?;
        if idx >= self.cells.len() {
            return None;
        }
        Some(self.cells[idx])
    }

    /// Write an occupancy class at a coordinate, returning the previous value.
    /// Returns an error for out-of-bounds coordinates.
    pub(super) fn set(
        &mut self,
        coord: GridCoord,
        class: OccupancyClass,
    ) -> Result<OccupancyClass, GeneratorError> {
        let idx = self.flat_index(coord)?;
        if idx >= self.cells.len() {
            return Err(GeneratorError::IrInvariant {
                stage: ErrorStage::Ir,
                detail: format!("occupancy_set_out_of_bounds {}", coord),
            });
        }
        let prev = self.cells[idx];
        self.cells[idx] = class;
        Ok(prev)
    }

    /// Test whether a rectangle of cells is entirely `Empty`.
    /// Uses checked arithmetic for coordinate additions.
    pub(super) fn is_rect_empty(
        &self,
        layer: u16,
        x: u16,
        y: u16,
        w: u16,
        h: u16,
    ) -> Result<bool, GeneratorError> {
        for dy in 0..h {
            for dx in 0..w {
                let cx = x.checked_add(dx).ok_or_else(|| {
                    GeneratorError::ArithmeticOverflow {
                        stage: ErrorStage::Ir,
                        operation: "is_rect_empty_x_add",
                    }
                })?;
                let cy = y.checked_add(dy).ok_or_else(|| {
                    GeneratorError::ArithmeticOverflow {
                        stage: ErrorStage::Ir,
                        operation: "is_rect_empty_y_add",
                    }
                })?;
                let coord =
                    GridCoord::new(layer, cx, cy, self.width, self.height, self.layers)?;
                if self.get(coord) != Some(OccupancyClass::Empty) {
                    return Ok(false);
                }
            }
        }
        Ok(true)
    }

    /// Mark a rectangle as occupied, returning an error if any cell is
    /// already non-Empty. Uses checked arithmetic.
    pub(super) fn reserve_rect(
        &mut self,
        layer: u16,
        x: u16,
        y: u16,
        w: u16,
        h: u16,
        class: OccupancyClass,
    ) -> Result<(), GeneratorError> {
        for dy in 0..h {
            for dx in 0..w {
                let cx = x.checked_add(dx).ok_or_else(|| {
                    GeneratorError::ArithmeticOverflow {
                        stage: ErrorStage::Ir,
                        operation: "reserve_rect_x_add",
                    }
                })?;
                let cy = y.checked_add(dy).ok_or_else(|| {
                    GeneratorError::ArithmeticOverflow {
                        stage: ErrorStage::Ir,
                        operation: "reserve_rect_y_add",
                    }
                })?;
                let coord =
                    GridCoord::new(layer, cx, cy, self.width, self.height, self.layers)?;
                let prev = self.get(coord);
                if prev != Some(OccupancyClass::Empty) {
                    return Err(GeneratorError::OccupancyConflict {
                        stage: ErrorStage::Placement,
                        detail: format!(
                            "cell_already_occupied {} was={:?} wanted={:?}",
                            coord, prev, class
                        ),
                    });
                }
            }
        }
        for dy in 0..h {
            for dx in 0..w {
                let cx = x.checked_add(dx).ok_or_else(|| {
                    GeneratorError::ArithmeticOverflow {
                        stage: ErrorStage::Ir,
                        operation: "reserve_rect_x_add2",
                    }
                })?;
                let cy = y.checked_add(dy).ok_or_else(|| {
                    GeneratorError::ArithmeticOverflow {
                        stage: ErrorStage::Ir,
                        operation: "reserve_rect_y_add2",
                    }
                })?;
                let coord =
                    GridCoord::new(layer, cx, cy, self.width, self.height, self.layers)?;
                self.set(coord, class)?;
            }
        }
        Ok(())
    }

    /// Reserve a set of cells; returns conflict error if any cell already occupied.
    pub(super) fn reserve_cells(
        &mut self,
        cells: &[GridCoord],
        class: OccupancyClass,
    ) -> Result<(), GeneratorError> {
        for coord in cells {
            let prev = self.get(*coord);
            if prev != Some(OccupancyClass::Empty) {
                return Err(GeneratorError::OccupancyConflict {
                    stage: ErrorStage::Placement,
                    detail: format!(
                        "cell_already_occupied {} was={:?} wanted={:?}",
                        coord, prev, class
                    ),
                });
            }
        }
        for coord in cells {
            self.set(*coord, class)?;
        }
        Ok(())
    }
}

// ─── Transition reservation ─────────────────────────────────────────────────

/// A validated transition between two adjacent layers, materialized from a
/// prefab variant's reservation and transition definitions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct TransitionReservation {
    /// Unique transition ID.
    pub(super) id: TransitionId,
    /// Prefab variant that supplied the transition contract.
    pub(super) variant_index: u16,
    /// The lower layer in the adjacent-layer pair.
    pub(super) lower_layer: u16,
    /// Hub footprint projection (x, y, w, h) on both endpoint layers.
    pub(super) hub_footprint: (u16, u16, u16, u16),
    /// Explicit lower and upper endpoint regions allocated before ordinary rooms.
    pub(super) lower_region: RegionId,
    pub(super) upper_region: RegionId,
    /// Concrete placed sockets that form the vertical candidate edge.
    pub(super) lower_socket: SocketId,
    pub(super) upper_socket: SocketId,
    /// The three ramp-run cells on the lower layer [(x0,y0),(x1,y1),(x2,y2)].
    pub(super) ramp_run_cells: Vec<GridCoord>,
    /// Upper-layer opening cells (must be void — layer+1).
    pub(super) upper_opening_cells: Vec<GridCoord>,
    /// Upper-layer landing cells (must be floor — layer+1).
    pub(super) landing_cells: Vec<GridCoord>,
    /// Upper-layer headroom cells that must remain void.
    pub(super) headroom_cells: Vec<GridCoord>,
    /// Lower-layer funnel cells approaching the ramp base.
    pub(super) lower_funnel_cells: Vec<GridCoord>,
    /// Lower-layer approach corridor cells.
    pub(super) lower_approach_cells: Vec<GridCoord>,
}

// ─── Placed socket ──────────────────────────────────────────────────────────

/// A socket as anchored within its host region in the global grid.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct PlacedSocket {
    pub(super) id: SocketId,
    /// Stable reference back to the prefab-variant socket.
    pub(super) variant_socket_index: u16,
    /// Global anchor position after placement.
    pub(super) global_anchor: GridCoord,
    /// Direction in global space (rotated from variant space).
    pub(super) direction: Direction,
    pub(super) width: u16,
    pub(super) role: SocketRole,
    /// Sockets that must connect to this one (by variant design).
    pub(super) paired_socket_id: Option<SocketId>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum Direction {
    North,
    East,
    South,
    West,
}

impl Direction {
    pub(super) fn delta(self) -> (i32, i32) {
        match self {
            Self::North => (0, -1),
            Self::East => (1, 0),
            Self::South => (0, 1),
            Self::West => (-1, 0),
        }
    }

    pub(super) fn opposite(self) -> Self {
        match self {
            Self::North => Self::South,
            Self::East => Self::West,
            Self::South => Self::North,
            Self::West => Self::East,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum SocketRole {
    Corridor,
    Hall,
    Doorway,
    Junction,
    DeadEnd,
    LandmarkApproach,
    LowerRampApproach,
    UpperLanding,
}

// ─── Placed region ──────────────────────────────────────────────────────────

/// A region after placement in the global grid.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct PlacedRegion {
    pub(super) id: RegionId,
    pub(super) role: RegionRole,
    /// Prefab variant index into the catalog variants list.
    pub(super) variant_index: u16,
    /// Layer this region occupies.
    pub(super) layer: u16,
    /// Global bounding box: (x, y, width, height) within the layer.
    pub(super) footprint: (u16, u16, u16, u16),
    /// Placed sockets belonging to this region.
    pub(super) sockets: Vec<PlacedSocket>,
    /// Transition IDs that this region supports (vertical hubs).
    pub(super) transitions: Vec<TransitionId>,
    /// Prefab marker intents (variant indices into the variant's marker list).
    pub(super) marker_variant_indices: Vec<u16>,
}

impl PlacedRegion {
    pub(super) fn origin(&self) -> GridCoord {
        GridCoord {
            layer: self.layer,
            x: self.footprint.0,
            y: self.footprint.1,
        }
    }
}

// ─── Candidate and intended edges ──────────────────────────────────────────

/// A geometrically realizable candidate connection between two placed sockets.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct CandidateEdge {
    pub(super) id: EdgeId,
    pub(super) source_socket: SocketId,
    pub(super) target_socket: SocketId,
    pub(super) source_region: RegionId,
    pub(super) target_region: RegionId,
    pub(super) path_witness: Vec<GridCoord>,
    pub(super) allowed_envelope_cells: Vec<GridCoord>,
    pub(super) cost: u64,
    pub(super) width: u16,
    /// Present only for the reservation-bound vertical edge of this transition.
    pub(super) transition: Option<TransitionId>,
}

/// Canonically ordered candidate graph built from the committed occupancy grid.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct CandidateGraph {
    pub(super) edges: Vec<CandidateEdge>,
    /// Exact committed placement occupancy used to reroute a selected witness
    /// around already-selected corridors during bounded topology search.
    pub(super) occupancy: OccupancyGrid,
}

/// A selected connection between two placed sockets.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct IntendedEdge {
    pub(super) id: EdgeId,
    pub(super) source_socket: SocketId,
    pub(super) target_socket: SocketId,
    pub(super) source_region: RegionId,
    pub(super) target_region: RegionId,
    /// Whether this edge is required or optional in the selected topology.
    pub(super) required: bool,
    /// A deterministic A* path witness: sequence of GridCoord from source socket
    /// inward cell to target socket inward cell.
    pub(super) path_witness: Vec<GridCoord>,
    /// Cell-level allowed envelope: the set of cells the corridor covers
    /// (path cells plus width clearance on each side).
    pub(super) allowed_envelope_cells: Vec<GridCoord>,
    /// Path cost (Manhattan distance).
    pub(super) cost: u64,
    /// Width of the corridor/hall this edge represents.
    pub(super) width: u16,
    /// Reservation bound to a vertical edge; absent for ordinary corridors.
    pub(super) transition: Option<TransitionId>,
}

// ─── Intended topology ──────────────────────────────────────────────────────

/// The complete structural intent for a single generation attempt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct IntendedTopology {
    /// All placed regions, in stable order by RegionId.
    pub(super) regions: Vec<PlacedRegion>,
    /// All selected edges forming the connectivity graph, in stable order by EdgeId.
    pub(super) edges: Vec<IntendedEdge>,
    /// Reserved transitions (ramp hubs) across layer pairs.
    pub(super) transitions: Vec<TransitionReservation>,
    /// Route distance metric: shortest-path distance from spawn to distant-landmark.
    pub(super) route_distance: u64,
    /// Number of cycles detected per layer.
    pub(super) per_layer_cycles: Vec<u32>,
    /// Maximum branch depth from spawn.
    pub(super) max_branch_depth: u32,
    /// Number of intentional dead-end regions.
    pub(super) dead_end_count: u32,
    /// Number of articulation points (proper detection).
    pub(super) articulation_count: u32,
    /// Number of edge-crossings that do not share a region.
    pub(super) crossing_count: u32,
    /// Configuration snapshot used during generation.
    pub(super) config: NormalizedGeneratorConfig,
}

// ─── ID-space validators ────────────────────────────────────────────────────

impl IntendedTopology {
    /// Verify that every region ID is unique.
    pub(super) fn validate_unique_region_ids(&self) -> Result<(), GeneratorError> {
        let mut seen = BTreeMap::new();
        for r in &self.regions {
            if let Some(prev) = seen.insert(r.id, r.role) {
                return Err(GeneratorError::IrInvariant {
                    stage: ErrorStage::Ir,
                    detail: format!(
                        "duplicate_region_id {} roles={:?} and {:?}",
                        r.id.raw(),
                        prev,
                        r.role
                    ),
                });
            }
        }
        Ok(())
    }

    /// Verify that every edge ID referenced is unique across the topology.
    pub(super) fn validate_unique_edge_ids(&self) -> Result<(), GeneratorError> {
        let mut seen = BTreeMap::new();
        for e in &self.edges {
            if let Some(prev) = seen.insert(e.id, (e.source_socket, e.target_socket)) {
                return Err(GeneratorError::IrInvariant {
                    stage: ErrorStage::Ir,
                    detail: format!(
                        "duplicate_edge_id {} prev={}->{}",
                        e.id.raw(),
                        prev.0.raw(),
                        prev.1.raw()
                    ),
                });
            }
        }
        Ok(())
    }

    /// Verify that socket references in edges map to real placed sockets.
    pub(super) fn validate_socket_references(&self) -> Result<(), GeneratorError> {
        let socket_map: BTreeMap<SocketId, RegionId> = self
            .regions
            .iter()
            .flat_map(|r| r.sockets.iter().map(move |s| (s.id, r.id)))
            .collect();
        for e in &self.edges {
            if socket_map.get(&e.source_socket) != Some(&e.source_region) {
                return Err(GeneratorError::IrInvariant {
                    stage: ErrorStage::Ir,
                    detail: format!("dangling_or_misowned_source_socket {}", e.source_socket.raw()),
                });
            }
            if socket_map.get(&e.target_socket) != Some(&e.target_region) {
                return Err(GeneratorError::IrInvariant {
                    stage: ErrorStage::Ir,
                    detail: format!("dangling_or_misowned_target_socket {}", e.target_socket.raw()),
                });
            }
        }
        Ok(())
    }

    /// Verify explicit transition endpoint region/socket bindings.
    pub(super) fn validate_transition_bindings(&self) -> Result<(), GeneratorError> {
        let regions: BTreeMap<RegionId, &PlacedRegion> =
            self.regions.iter().map(|region| (region.id, region)).collect();
        for transition in &self.transitions {
            let lower = regions.get(&transition.lower_region).ok_or_else(|| {
                GeneratorError::TransitionBinding {
                    stage: ErrorStage::Ir,
                    transition: transition.id.raw(),
                    reason: "missing_lower_endpoint_region",
                }
            })?;
            let upper = regions.get(&transition.upper_region).ok_or_else(|| {
                GeneratorError::TransitionBinding {
                    stage: ErrorStage::Ir,
                    transition: transition.id.raw(),
                    reason: "missing_upper_endpoint_region",
                }
            })?;
            let expected_upper = transition.lower_layer.checked_add(1).ok_or(
                GeneratorError::ArithmeticOverflow {
                    stage: ErrorStage::Ir,
                    operation: "transition_binding_upper_layer",
                },
            )?;
            if lower.layer != transition.lower_layer || upper.layer != expected_upper {
                return Err(GeneratorError::TransitionBinding {
                    stage: ErrorStage::Ir,
                    transition: transition.id.raw(),
                    reason: "endpoint_layer_mismatch",
                });
            }
            if !lower.transitions.contains(&transition.id)
                || !upper.transitions.contains(&transition.id)
            {
                return Err(GeneratorError::TransitionBinding {
                    stage: ErrorStage::Ir,
                    transition: transition.id.raw(),
                    reason: "endpoint_region_missing_transition",
                });
            }
            let lower_socket = lower
                .sockets
                .iter()
                .find(|socket| socket.id == transition.lower_socket);
            let upper_socket = upper
                .sockets
                .iter()
                .find(|socket| socket.id == transition.upper_socket);
            if !matches!(lower_socket, Some(socket) if socket.role == SocketRole::LowerRampApproach)
                || !matches!(upper_socket, Some(socket) if socket.role == SocketRole::UpperLanding)
            {
                return Err(GeneratorError::TransitionBinding {
                    stage: ErrorStage::Ir,
                    transition: transition.id.raw(),
                    reason: "endpoint_socket_mismatch",
                });
            }
        }
        Ok(())
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── IdAllocator ────────────────────────────────────────────────────

    #[test]
    fn id_allocator_sequential_u32_ids() {
        let mut alloc = IdAllocator::new();
        let r0 = alloc.next_region().unwrap();
        let r1 = alloc.next_region().unwrap();
        assert_eq!(r0.raw(), 0);
        assert_eq!(r1.raw(), 1);
        assert!(r0 < r1);

        let s0 = alloc.next_socket().unwrap();
        let s1 = alloc.next_socket().unwrap();
        assert_eq!(s0.raw(), 0);
        assert_eq!(s1.raw(), 1);

        // Counters are independent
        let r2 = alloc.next_region().unwrap();
        assert_eq!(r2.raw(), 2);
    }

    #[test]
    fn id_allocator_wraps_to_error() {
        let mut alloc = IdAllocator {
            next_region: u32::MAX,
            next_socket: 0,
            next_edge: 0,
            next_transition: 0,
        };
        // next_region at u32::MAX means we've exhausted the namespace
        assert!(alloc.next_region().is_err());
    }

    #[test]
    fn id_allocator_last_valid_id() {
        let mut alloc = IdAllocator {
            next_region: u32::MAX - 1,
            next_socket: 0,
            next_edge: 0,
            next_transition: 0,
        };
        // Can allocate u32::MAX-1
        let r = alloc.next_region().unwrap();
        assert_eq!(r.raw(), u32::MAX - 1);
        // Next allocation overflows
        assert!(alloc.next_region().is_err());
    }

    // ── GridCoord ───────────────────────────────────────────────────────

    #[test]
    fn coord_new_rejects_out_of_bounds() {
        assert!(GridCoord::new(0, 0, 0, 10, 10, 2).is_ok());
        assert!(GridCoord::new(2, 0, 0, 10, 10, 2).is_err());
        assert!(GridCoord::new(0, 10, 0, 10, 10, 2).is_err());
        assert!(GridCoord::new(0, 0, 10, 10, 10, 2).is_err());
    }

    #[test]
    fn coord_flat_index_roundtrip() {
        for (layer, x, y) in [(0, 0, 0), (0, 9, 9), (1, 5, 5), (1, 0, 9), (0, 9, 0)] {
            let coord = GridCoord::new(layer, x, y, 10, 10, 2).unwrap();
            let idx = coord.to_flat_index(10, 10).unwrap();
            let back = GridCoord::from_flat_index(idx, 10, 10, 2).unwrap();
            assert_eq!(coord, back, "roundtrip failed for {:?}", coord);
        }
    }

    #[test]
    fn coord_from_flat_index_rejects_out_of_range() {
        assert!(GridCoord::from_flat_index(200, 10, 10, 2).is_err());
        assert!(GridCoord::from_flat_index(0, 10, 10, 2).is_ok());
    }

    #[test]
    fn coord_from_flat_index_rejects_zero_dimensions() {
        assert!(GridCoord::from_flat_index(0, 0, 10, 2).is_err());
        assert!(GridCoord::from_flat_index(0, 10, 0, 2).is_err());
        assert!(GridCoord::from_flat_index(0, 10, 10, 0).is_err());
    }

    #[test]
    fn coord_to_flat_index_handles_large_grids() {
        let coord = GridCoord::new(0, 0, 0, 64, 64, 4).unwrap();
        assert!(coord.to_flat_index(64, 64).is_ok());
    }

    // ── Occupancy grid ──────────────────────────────────────────────────

    #[test]
    fn occupancy_grid_zero_dimensions_rejected() {
        assert!(OccupancyGrid::new(0, 10, 2).is_err());
        assert!(OccupancyGrid::new(10, 0, 2).is_err());
        assert!(OccupancyGrid::new(10, 10, 0).is_err());
    }

    #[test]
    fn occupancy_grid_rect_basics() {
        let mut grid = OccupancyGrid::new(10, 10, 2).unwrap();
        assert!(grid.is_rect_empty(0, 0, 0, 3, 3).unwrap());
        let region_id: u32 = 42;
        grid.reserve_rect(0, 0, 0, 2, 2, OccupancyClass::Region(region_id))
            .unwrap();
        assert!(!grid.is_rect_empty(0, 0, 0, 3, 3).unwrap());
        assert!(grid.is_rect_empty(0, 2, 0, 1, 2).unwrap());
    }

    #[test]
    fn occupancy_grid_rejects_overlap() {
        let mut grid = OccupancyGrid::new(10, 10, 2).unwrap();
        grid.reserve_rect(0, 1, 1, 3, 3, OccupancyClass::Region(100))
            .unwrap();
        let result = grid.reserve_rect(0, 2, 2, 2, 2, OccupancyClass::Region(200));
        assert!(result.is_err());
        match result {
            Err(GeneratorError::OccupancyConflict { .. }) => {}
            _ => panic!("expected OccupancyConflict"),
        }
    }

    #[test]
    fn occupancy_grid_out_of_bounds_rejected() {
        let mut grid = OccupancyGrid::new(5, 5, 1).unwrap();
        assert!(grid
            .reserve_rect(0, 3, 0, 3, 5, OccupancyClass::Region(1))
            .is_err());
        assert!(grid.reserve_rect(1, 0, 0, 1, 1, OccupancyClass::Empty).is_err());
    }

    #[test]
    fn occupancy_grid_coord_overflow_rejected() {
        let mut grid = OccupancyGrid::new(64, 64, 2).unwrap();
        // x + w overflow
        let result = grid.reserve_rect(0, u16::MAX, 0, 1, 1, OccupancyClass::Region(1));
        assert!(result.is_err());
    }

    // ── IntendedTopology validators ─────────────────────────────────────

    #[test]
    fn validate_unique_region_ids() {
        let mut alloc = IdAllocator::new();
        let id0 = alloc.next_region().unwrap();
        let id1 = alloc.next_region().unwrap();

        let mut topology = IntendedTopology {
            regions: vec![
                PlacedRegion {
                    id: id0,
                    role: RegionRole::Spawn,
                    variant_index: 0,
                    layer: 0,
                    footprint: (0, 0, 5, 5),
                    sockets: vec![],
                    transitions: vec![],
                    marker_variant_indices: vec![],
                },
                PlacedRegion {
                    id: id1,
                    role: RegionRole::DistantLandmark,
                    variant_index: 0,
                    layer: 0,
                    footprint: (10, 0, 5, 5),
                    sockets: vec![],
                    transitions: vec![],
                    marker_variant_indices: vec![],
                },
            ],
            edges: vec![],
            transitions: vec![],
            route_distance: 0,
            per_layer_cycles: vec![0],
            max_branch_depth: 0,
            dead_end_count: 0,
            articulation_count: 0,
            crossing_count: 0,
            config: crate::generator::config::GeneratorConfig::custom(64, 64, 2)
                .normalize()
                .unwrap(),
        };
        assert!(topology.validate_unique_region_ids().is_ok());

        // Break uniqueness
        topology.regions[1].id = topology.regions[0].id;
        assert!(topology.validate_unique_region_ids().is_err());
    }
}
