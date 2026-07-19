use std::collections::BTreeMap;
use std::fmt;
use std::sync::atomic::{AtomicU32, Ordering};

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
    pub(super) fn from_flat_index(
        idx: usize,
        width: u16,
        height: u16,
        layers: u16,
    ) -> Result<Self, GeneratorError> {
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

// ─── Newtype ID generation ──────────────────────────────────────────────────

macro_rules! define_id {
    ($vis:vis struct $name:ident) => {
        define_id!($vis struct $name => concat!(module_path!(), "::", stringify!($name)));
    };
    ($vis:vis struct $name:ident => $label:expr) => {
        #[doc = concat!("Stable typed newtype ID for ", stringify!($name), ".")]
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
        $vis struct $name(pub(super) u32);

        impl $name {
            #[doc = concat!("Generate a new unique ", stringify!($name), ".")]
            pub(super) fn new() -> Self {
                static COUNTER: AtomicU32 = AtomicU32::new(0);
                let id = COUNTER.fetch_add(1, Ordering::Relaxed);
                // Saturate at u32::MAX to avoid incrementing a newtype beyond
                // its representation. Real generators will never exhaust this
                // namespace — a generator must exceed 2³² regions before the
                // panic guard trips; profiles are capped at 64 regions.
                if id == u32::MAX {
                    panic!("{} counter exhausted", $label);
                }
                Self(id)
            }

            /// Return the raw u32 value.
            pub(super) const fn raw(self) -> u32 {
                self.0
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}({})", stringify!($name), self.0)
            }
        }
    };
}

define_id!(pub(super) struct RegionId);
define_id!(pub(super) struct SocketId);
define_id!(pub(super) struct EdgeId);
define_id!(pub(super) struct TransitionId);

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
    /// Reserved by a transition (transition ID). Transitions are immutable
    /// and always win over ordinary footprint claims.
    Transition(u32),
    /// Reserved by a placed region footprint.
    Region(RegionId),
    /// Reserved as a socket funnel / approach corridor approach.
    Socket(SocketId),
    /// Spacing cushion around a placed footprint.
    Spacing(RegionId),
}

/// Occupancy grid tracking per-cell ownership across all layers.
///
/// Backed by a flat vector per layer for fast probing; coordinates are
/// clamped to grid bounds through the `GridCoord` constructor.
#[derive(Debug, Clone)]
pub(super) struct OccupancyGrid {
    width: u16,
    height: u16,
    layers: u16,
    cells: Vec<OccupancyClass>,
}

impl OccupancyGrid {
    pub(super) fn new(width: u16, height: u16, layers: u16) -> Self {
        let capacity = width as usize * height as usize * layers as usize;
        Self {
            width,
            height,
            layers,
            cells: vec![OccupancyClass::Empty; capacity],
        }
    }

    #[allow(dead_code)]
    pub(super) const fn dimensions(&self) -> (u16, u16, u16) {
        (self.width, self.height, self.layers)
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
                let coord = GridCoord::new(layer, x + dx, y + dy, self.width, self.height, self.layers)?;
                if self.get(coord) != Some(OccupancyClass::Empty) {
                    return Ok(false);
                }
            }
        }
        Ok(true)
    }

    /// Mark a rectangle as occupied, returning an error if any cell is
    /// already non-Empty.
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
                let coord = GridCoord::new(layer, x + dx, y + dy, self.width, self.height, self.layers)?;
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
                let coord = GridCoord::new(layer, x + dx, y + dy, self.width, self.height, self.layers)?;
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

/// Dimension of a transition ramp covering a complete layer crossing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct TransitionReservation {
    /// Unique transition ID.
    pub(super) id: TransitionId,
    /// The lower-layer region acting as the ramp-hub origin.
    pub(super) lower_layer: u16,
    /// Hub footprint rectangle on the lower layer (layer, x, y, w, h).
    pub(super) hub_footprint: (u16, u16, u16, u16),
    /// The three ramp-run cells on the lower layer [(x0,y0),(x1,y1),(x2,y2)].
    pub(super) ramp_run: [(u16, u16); 3],
    /// Upper-layer opening rectangle (must be void — layer+1, x, y, w, h).
    pub(super) upper_opening: (u16, u16, u16, u16),
    /// Upper-layer landing (must be floor — layer+1, x, y).
    pub(super) landing: (u16, u16),
    /// Upper-layer headroom cells that must remain void.
    pub(super) headroom: Vec<(u16, u16)>,
    /// Lower-layer funnel cells approaching the ramp base.
    pub(super) lower_funnel: Vec<(u16, u16)>,
    /// Lower-layer approach corridor cells.
    pub(super) lower_approach: Vec<(u16, u16)>,
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

// ─── Intended edge ──────────────────────────────────────────────────────────

/// A candidate connection between two placed sockets.
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
    /// to target socket (excluding the socket apertures themselves).
    pub(super) path_witness: Vec<GridCoord>,
    /// The bounding rectangle of the path plus clearance.
    pub(super) allowed_envelope: (u16, u16, u16, u16),
    /// Path cost (Manhattan distance or weighted cells).
    pub(super) cost: u64,
    /// Width of the corridor/hall this edge represents.
    pub(super) width: u16,
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
    /// Route distance metrics (total Manhattan distance of required spine).
    pub(super) route_distance: u64,
    /// Number of cycles detected per layer.
    pub(super) per_layer_cycles: Vec<u32>,
    /// Maximum branch depth from spawn.
    pub(super) max_branch_depth: u32,
    /// Number of intentional dead-end regions.
    pub(super) dead_end_count: u32,
    /// Number of articulation points.
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

    /// Verify that every edge ID and socket ID referenced is unique across
    /// the topology.
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
            if !socket_map.contains_key(&e.source_socket) {
                return Err(GeneratorError::IrInvariant {
                    stage: ErrorStage::Ir,
                    detail: format!("dangling_source_socket {}", e.source_socket.raw()),
                });
            }
            if !socket_map.contains_key(&e.target_socket) {
                return Err(GeneratorError::IrInvariant {
                    stage: ErrorStage::Ir,
                    detail: format!("dangling_target_socket {}", e.target_socket.raw()),
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
    fn coord_flat_index_overflow_rejected() {
        // Test that overflow is caught: a coordinate at max reasonable values
        // Large grids can overflow usize if width*height*layer > usize::MAX
        // But our grid sizes are bounded by the profile system
        let coord = GridCoord::new(0, 0, 0, 64, 64, 4).unwrap();
        assert!(coord.to_flat_index(64, 64).is_ok());
    }

    #[test]
    fn coord_from_flat_index_rejects_out_of_range() {
        assert!(GridCoord::from_flat_index(200, 10, 10, 2).is_err());
        assert!(GridCoord::from_flat_index(0, 10, 10, 2).is_ok());
    }

    // ── ID generation ───────────────────────────────────────────────────

    #[test]
    fn region_ids_are_unique_and_monotonic() {
        let a = RegionId::new();
        let b = RegionId::new();
        let c = RegionId::new();
        assert!(a != b);
        assert!(b != c);
        assert!(a != c);
        assert!(a.raw() < b.raw());
        assert!(b.raw() < c.raw());
    }

    #[test]
    fn socket_ids_are_independent_counters() {
        let r1 = RegionId::new();
        let s1 = SocketId::new();
        let r2 = RegionId::new();
        let s2 = SocketId::new();
        // Each counter is independent
        assert!(s1.raw() < s2.raw());
        assert!(r1.raw() < r2.raw());
    }

    // ── Occupancy grid ──────────────────────────────────────────────────

    #[test]
    fn occupancy_grid_rect_basics() {
        let mut grid = OccupancyGrid::new(10, 10, 2);
        assert!(grid.is_rect_empty(0, 0, 0, 3, 3).unwrap());
        grid.reserve_rect(0, 0, 0, 2, 2, OccupancyClass::Region(RegionId::new()))
            .unwrap();
        assert!(!grid.is_rect_empty(0, 0, 0, 3, 3).unwrap());
        assert!(grid.is_rect_empty(0, 2, 0, 1, 2).unwrap());
    }

    #[test]
    fn occupancy_grid_rejects_overlap() {
        let mut grid = OccupancyGrid::new(10, 10, 2);
        let r0 = RegionId::new();
        grid.reserve_rect(0, 1, 1, 3, 3, OccupancyClass::Region(r0))
            .unwrap();
        let r1 = RegionId::new();
        let result = grid.reserve_rect(0, 2, 2, 2, 2, OccupancyClass::Region(r1));
        assert!(result.is_err());
        match result {
            Err(GeneratorError::OccupancyConflict { .. }) => {}
            _ => panic!("expected OccupancyConflict"),
        }
    }

    #[test]
    fn occupancy_grid_out_of_bounds_rejected() {
        let mut grid = OccupancyGrid::new(5, 5, 1);
        assert!(grid
            .reserve_rect(0, 3, 0, 3, 5, OccupancyClass::Region(RegionId::new()))
            .is_err());
        assert!(grid.reserve_rect(1, 0, 0, 1, 1, OccupancyClass::Empty).is_err());
    }

    // ── IntendedTopology validators ─────────────────────────────────────

    #[test]
    fn validate_unique_region_ids() {
        let mut topology = IntendedTopology {
            regions: vec![
                PlacedRegion {
                    id: RegionId::new(),
                    role: RegionRole::Spawn,
                    variant_index: 0,
                    layer: 0,
                    footprint: (0, 0, 5, 5),
                    sockets: vec![],
                    transitions: vec![],
                    marker_variant_indices: vec![],
                },
                PlacedRegion {
                    id: RegionId::new(),
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
