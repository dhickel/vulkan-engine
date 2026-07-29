//! Enhanced v2 horizontal routing — width-aware A* corridor pathfinding.
//!
//! Computes orthogonal corridor paths between two room sockets on the
//! same layer. The corridor is 64 Quake units wide (4 quantum cells).
//! A* operates on a cell-resolution grid derived from the occupancy grid;
//! each expansion checks the full corridor envelope for clearance.

use crate::config::CONSTRUCTION_QUANTUM;

use super::error::EnhancedError;
use super::intent::RoomId;
use super::occupancy::OccupancyGrid;

const Q: i32 = CONSTRUCTION_QUANTUM as i32;
const Q_U: u32 = CONSTRUCTION_QUANTUM;

/// Corridor width in Quake units.
pub const CORRIDOR_WIDTH: i32 = 64;
/// Half-width in Quake units.
const CORRIDOR_HALF_WIDTH: i32 = CORRIDOR_WIDTH / 2;
// ── Route segment ──────────────────────────────────────────────────────────

/// A straight, axis-aligned corridor segment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RouteSegment {
    /// Start position in Quake units: (x, y) on the corridor centerline.
    pub start: (i32, i32),
    /// End position in Quake units: (x, y) on the corridor centerline.
    pub end: (i32, i32),
    /// Corridor envelope in Quake units: (x0, y0, x1, y1).
    pub envelope: (i32, i32, i32, i32),
}

// ── A* routing result ──────────────────────────────────────────────────────

/// The result of routing a single horizontal edge.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RouteResult {
    pub segments: Vec<RouteSegment>,
    /// Actual A* node expansions, including failed detours.
    pub expansions: u32,
}

// ── Entry point ────────────────────────────────────────────────────────────

/// Find an orthogonal corridor path between two socket anchors on the same layer.
///
/// `from_anchor` and `to_anchor` are in Quake units. The corridor is
/// `CORRIDOR_WIDTH` wide (64 units). Returns a list of axis-aligned
/// segments forming the centerline path.
///
/// `source_room` and `target_room` allow the corridor envelope to overlap
/// cells owned by these rooms (for portal clearance at the socket wall).
///
/// `max_expansions` bounds the A* search; when exceeded, returns
/// `RouteExhausted`.
pub fn route_sockets(
    from_anchor: (i32, i32),
    to_anchor: (i32, i32),
    grid: &OccupancyGrid,
    xy_extent: u32,
    max_expansions: u32,
    source_room: RoomId,
    target_room: RoomId,
) -> Result<RouteResult, EnhancedError> {
    let cells_x = xy_extent / Q_U;
    let cells_y = xy_extent / Q_U;

    // Convert anchors to cell coordinates.
    // For a socket on a wall, the corridor centerline starts at the cell
    // just outside the wall face. We snap to the nearest cell center
    // quantized to the grid.
    let start_cell = world_to_cell(from_anchor.0, from_anchor.1);
    let end_cell = world_to_cell(to_anchor.0, to_anchor.1);

    // Ensure start and end are within bounds
    if start_cell.0 >= cells_x
        || start_cell.1 >= cells_y
        || end_cell.0 >= cells_x
        || end_cell.1 >= cells_y
    {
        return Err(EnhancedError::ContractViolation {
            detail: format!(
                "route anchors out of bounds: start={:?} end={:?} grid={}x{}",
                from_anchor, to_anchor, xy_extent, xy_extent,
            ),
        });
    }

    // Try the primary start cell; if blocked, try neighbors
    let start_candidates = [
        start_cell,
        (
            start_cell.0.saturating_add(1).min(cells_x - 1),
            start_cell.1,
        ),
        (start_cell.0.saturating_sub(1), start_cell.1),
        (
            start_cell.0,
            start_cell.1.saturating_add(1).min(cells_y - 1),
        ),
        (start_cell.0, start_cell.1.saturating_sub(1)),
    ];

    let mut last_err: Option<EnhancedError> = None;
    for &sc in &start_candidates {
        let path = a_star_search(
            sc,
            end_cell,
            grid,
            cells_x,
            cells_y,
            max_expansions,
            source_room,
            target_room,
        );
        match path {
            Ok((p, expansions)) => {
                let segments = simplify_path(&p, from_anchor, to_anchor, grid)?;
                // A* probes each step's complete 64-unit envelope. Recheck
                // the final merged segments so no simplification can weaken
                // that guarantee.
                for segment in &segments {
                    if !envelope_is_clear(segment.envelope, grid, source_room, target_room)? {
                        return Err(EnhancedError::RouteExhausted { expansions });
                    }
                }
                return Ok(RouteResult {
                    segments,
                    expansions,
                });
            }
            Err(e) => {
                last_err = Some(e);
                continue;
            }
        }
    }

    // All start candidates failed
    Err(last_err.unwrap_or(EnhancedError::RouteExhausted { expansions: 0 }))
}

// ── World/cell coordinate helpers ──────────────────────────────────────────

fn world_to_cell(x: i32, y: i32) -> (u32, u32) {
    // Snap to nearest quantum-aligned position
    let cx = ((x + Q / 2) / Q).max(0) as u32;
    let cy = ((y + Q / 2) / Q).max(0) as u32;
    (cx, cy)
}

fn cell_to_world(cx: u32, cy: u32) -> (i32, i32) {
    ((cx * Q_U) as i32, (cy * Q_U) as i32)
}

// ── A* search ──────────────────────────────────────────────────────────────

fn a_star_search(
    start: (u32, u32),
    goal: (u32, u32),
    grid: &OccupancyGrid,
    cells_x: u32,
    cells_y: u32,
    max_expansions: u32,
    source_room: RoomId,
    target_room: RoomId,
) -> Result<(Vec<(u32, u32)>, u32), EnhancedError> {
    use std::collections::BinaryHeap;

    // For deterministic tie-breaking, we use a custom ordering.
    // Nodes with lower f-cost come first; ties broken by (x, y).
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct HeapEntry {
        f: u32,
        g: u32,
        x: u32,
        y: u32,
    }

    // Reverse ordering for BinaryHeap (min-heap)
    impl Ord for HeapEntry {
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            other
                .f
                .cmp(&self.f)
                .then_with(|| other.x.cmp(&self.x))
                .then_with(|| other.y.cmp(&self.y))
        }
    }
    impl PartialOrd for HeapEntry {
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(other))
        }
    }

    let total_cells = (cells_x as usize) * (cells_y as usize);

    // g_score: cost from start to each cell
    let mut g_score: Vec<u32> = vec![u32::MAX; total_cells];
    // parent: for path reconstruction
    let mut parent: Vec<Option<(u32, u32)>> = vec![None; total_cells];

    let start_idx = (cells_x as usize) * (start.1 as usize) + (start.0 as usize);
    g_score[start_idx] = 0;

    let mut open = BinaryHeap::new();
    let start_h = manhattan(start.0, start.1, goal.0, goal.1);
    open.push(HeapEntry {
        f: start_h,
        g: 0,
        x: start.0,
        y: start.1,
    });

    let mut expansions: u32 = 0;

    while let Some(entry) = open.pop() {
        let (cx, cy) = (entry.x, entry.y);
        let idx = (cells_x as usize) * (cy as usize) + (cx as usize);

        // Skip if we already found a better path
        if entry.g > g_score[idx] {
            continue;
        }

        // Check if we reached the goal
        if cx == goal.0 && cy == goal.1 {
            // Reconstruct path
            let mut path = Vec::new();
            let mut cur = (cx, cy);
            path.push(cur);
            while let Some(p) = parent[(cells_x as usize) * (cur.1 as usize) + (cur.0 as usize)] {
                path.push(p);
                cur = p;
            }
            path.reverse();
            return Ok((path, expansions));
        }

        expansions += 1;
        if expansions > max_expansions {
            return Err(EnhancedError::RouteExhausted { expansions });
        }

        // Expand in canonical order: East, North, South, West
        let neighbors: [(i32, i32); 4] = [(1, 0), (0, 1), (0, -1), (-1, 0)];

        for (dx, dy) in &neighbors {
            let nx = cx as i32 + dx;
            let ny = cy as i32 + dy;

            if nx < 0 || ny < 0 || nx >= cells_x as i32 || ny >= cells_y as i32 {
                continue;
            }

            let nx = nx as u32;
            let ny = ny as u32;
            let n_idx = (cells_x as usize) * (ny as usize) + (nx as usize);

            // Check corridor envelope for this step
            if !corridor_step_clear(
                cx,
                cy,
                nx,
                ny,
                grid,
                cells_x,
                cells_y,
                source_room,
                target_room,
            )? {
                continue;
            }

            let tentative_g = entry.g + 1;
            if tentative_g < g_score[n_idx] {
                g_score[n_idx] = tentative_g;
                parent[n_idx] = Some((cx, cy));
                let h = manhattan(nx, ny, goal.0, goal.1);
                open.push(HeapEntry {
                    f: tentative_g + h,
                    g: tentative_g,
                    x: nx,
                    y: ny,
                });
            }
        }
    }

    Err(EnhancedError::RouteExhausted { expansions })
}

// ── Corridor clearance check ───────────────────────────────────────────────

/// Check that the corridor envelope for a single-cell step from (cx,cy) to
/// (nx,ny) does not overlap any occupied cells, except cells owned by
/// `source_room` or `target_room` (for portal clearance).
///
/// The full 64-unit envelope is checked for every expansion. This is
/// deliberately conservative: a centreline path is not a materializable
/// route unless its entire corridor footprint is clear.
fn corridor_step_clear(
    _cx: u32,
    _cy: u32,
    nx: u32,
    ny: u32,
    grid: &OccupancyGrid,
    _cells_x: u32,
    _cells_y: u32,
    _source_room: RoomId,
    _target_room: RoomId,
) -> Result<bool, EnhancedError> {
    let center = ((nx * Q_U) as i32, (ny * Q_U) as i32);
    let envelope = compute_envelope(center.0, center.1, center.0, center.1);
    envelope_is_clear(envelope, grid, _source_room, _target_room)
}

fn envelope_is_clear(
    (x0, y0, x1, y1): (i32, i32, i32, i32),
    grid: &OccupancyGrid,
    source_room: RoomId,
    target_room: RoomId,
) -> Result<bool, EnhancedError> {
    use super::occupancy::Owner;
    if x0 < 0 || y0 < 0 || x1 <= x0 || y1 <= y0 {
        return Ok(false);
    }
    let qx0 = x0 as u32 / Q_U;
    let qy0 = y0 as u32 / Q_U;
    let qx1 = x1 as u32 / Q_U;
    let qy1 = y1 as u32 / Q_U;
    if qx1 > grid.cells_x() || qy1 > grid.cells_y() {
        return Ok(false);
    }
    for cy in qy0..qy1 {
        for cx in qx0..qx1 {
            let idx = grid.cells_x() as usize * cy as usize + cx as usize;
            match grid.cells()[idx] {
                Owner::Empty => {}
                Owner::Room(room) if room == source_room || room == target_room => {}
                // Existing horizontal reservations form explicit junctions;
                // transition footprints are still exclusive blockers.
                Owner::Route(_) => {}
                _ => return Ok(false),
            }
        }
    }
    Ok(true)
}

// ── Manhattan distance heuristic ───────────────────────────────────────────

fn manhattan(x1: u32, y1: u32, x2: u32, y2: u32) -> u32 {
    (if x1 > x2 { x1 - x2 } else { x2 - x1 }) + (if y1 > y2 { y1 - y2 } else { y2 - y1 })
}

// ── Path simplification ────────────────────────────────────────────────────

/// Convert a raw cell path to simplified orthogonal segments with envelopes.
fn simplify_path(
    path: &[(u32, u32)],
    from_anchor: (i32, i32),
    to_anchor: (i32, i32),
    _grid: &OccupancyGrid,
) -> Result<Vec<RouteSegment>, EnhancedError> {
    if path.is_empty() {
        return Ok(Vec::new());
    }

    let mut segments = Vec::new();

    // Start from the socket anchor, not the cell center
    // We need to emit segments connecting the socket anchors through the path.

    // For simplicity, we emit segments between consecutive cells,
    // converting each to its envelope.
    // Then merge consecutive same-direction segments.

    if path.len() == 1 {
        // Direct connection — single segment between anchors
        let envelope = compute_envelope(from_anchor.0, from_anchor.1, to_anchor.0, to_anchor.1);
        return Ok(vec![RouteSegment {
            start: (from_anchor.0, from_anchor.1),
            end: (to_anchor.0, to_anchor.1),
            envelope,
        }]);
    }

    // Multi-cell path: emit segments along the path
    // First segment from socket anchor to first cell center
    let first_cell_center = cell_to_world(path[0].0, path[0].1);

    // Build raw segments from cell centers
    let mut raw_segs: Vec<((i32, i32), (i32, i32))> = Vec::new();

    // Start segment: from socket anchor to first waypoint (skip if same)
    if from_anchor != first_cell_center {
        raw_segs.push((from_anchor, first_cell_center));
    }

    for i in 1..path.len() {
        let prev = cell_to_world(path[i - 1].0, path[i - 1].1);
        let curr = cell_to_world(path[i].0, path[i].1);
        raw_segs.push((prev, curr));
    }

    // Final segment: from last waypoint to target socket anchor (skip if same)
    let last_cell_center = cell_to_world(path[path.len() - 1].0, path[path.len() - 1].1);
    if last_cell_center != to_anchor {
        raw_segs.push((last_cell_center, to_anchor));
    }

    // Merge collinear consecutive segments
    let mut merged: Vec<((i32, i32), (i32, i32))> = Vec::new();
    for (start, end) in raw_segs {
        if start == end {
            continue; // zero-length, skip
        }
        if let Some(last) = merged.last_mut() {
            // Same direction and collinear
            let last_dir_x = (last.1 .0 - last.0 .0).signum();
            let last_dir_y = (last.1 .1 - last.0 .1).signum();
            let this_dir_x = (end.0 - start.0).signum();
            let this_dir_y = (end.1 - start.1).signum();

            if last_dir_x == this_dir_x && last_dir_y == this_dir_y {
                // Check collinearity: the line continues
                if (last_dir_x != 0 && last.1 .1 == start.1 && last.1 .0 == start.0)
                    || (last_dir_y != 0 && last.1 .0 == start.0 && last.1 .1 == start.1)
                {
                    last.1 = end;
                    continue;
                }
            }
        }
        merged.push((start, end));
    }

    // Convert to RouteSegments with envelopes
    for (start, end) in merged {
        let envelope = compute_envelope(start.0, start.1, end.0, end.1);
        segments.push(RouteSegment {
            start,
            end,
            envelope,
        });
    }

    Ok(segments)
}

/// Compute the corridor envelope for a centerline segment from (x0,y0) to (x1,y1).
/// Snaps to quantum (16-unit) boundaries.
fn compute_envelope(x0: i32, y0: i32, x1: i32, y1: i32) -> (i32, i32, i32, i32) {
    let q = CONSTRUCTION_QUANTUM as i32;
    let hw = CORRIDOR_HALF_WIDTH;
    // Snap utility: round down to quantum boundary
    let snap_down = |v: i32| -> i32 { (v / q) * q };
    let snap_up = |v: i32| -> i32 { ((v + q - 1) / q) * q };

    if x0 == x1 && y0 == y1 {
        // Zero-length: produce a minimal 64x64 square around the point
        (
            snap_down(x0 - hw),
            snap_down(y0 - hw),
            snap_up(x0 + hw),
            snap_up(y0 + hw),
        )
    } else if y0 == y1 {
        // Horizontal
        let min_x = std::cmp::min(x0, x1);
        let max_x = std::cmp::max(x0, x1);
        (
            snap_down(min_x),
            snap_down(y0 - hw),
            snap_up(max_x),
            snap_up(y0 + hw),
        )
    } else {
        // Vertical (x0 == x1)
        let min_y = std::cmp::min(y0, y1);
        let max_y = std::cmp::max(y0, y1);
        (
            snap_down(x0 - hw),
            snap_down(min_y),
            snap_up(x0 + hw),
            snap_up(max_y),
        )
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::super::intent::RoomId;
    use super::super::occupancy::OccupancyGrid;
    use super::*;

    fn make_grid() -> OccupancyGrid {
        OccupancyGrid::new(1024, 1024).unwrap()
    }

    #[test]
    fn straight_horizontal_route() {
        let grid = make_grid();
        let result = route_sockets(
            (16, 48),
            (208, 48),
            &grid,
            1024,
            10000,
            RoomId(0),
            RoomId(0),
        )
        .unwrap();
        assert!(!result.segments.is_empty());
        // Should be a straight horizontal corridor
        for seg in &result.segments {
            assert_eq!(seg.start.1, seg.end.1); // same y
            assert!(seg.envelope.3 - seg.envelope.1 >= CORRIDOR_WIDTH);
        }
    }

    #[test]
    fn straight_vertical_route() {
        let grid = make_grid();
        let result = route_sockets(
            (48, 16),
            (48, 208),
            &grid,
            1024,
            10000,
            RoomId(0),
            RoomId(0),
        )
        .unwrap();
        assert!(!result.segments.is_empty());
        for seg in &result.segments {
            assert_eq!(seg.start.0, seg.end.0); // same x
            assert!(seg.envelope.2 - seg.envelope.0 >= CORRIDOR_WIDTH);
        }
    }

    #[test]
    fn l_shaped_route() {
        let grid = make_grid();
        let result = route_sockets(
            (16, 48),
            (208, 160),
            &grid,
            1024,
            10000,
            RoomId(0),
            RoomId(0),
        )
        .unwrap();
        assert!(!result.segments.is_empty());
        // Should have at least 2 segments (L-shape)
        for seg in &result.segments {
            let dx = (seg.end.0 - seg.start.0).abs();
            let dy = (seg.end.1 - seg.start.1).abs();
            assert!(dx == 0 || dy == 0, "segment must be axis-aligned");
        }
        // Segments should connect end-to-end
        for i in 1..result.segments.len() {
            assert_eq!(
                result.segments[i - 1].end,
                result.segments[i].start,
                "segments must be connected"
            );
        }
    }

    #[test]
    fn route_avoids_obstacle() {
        let mut grid = make_grid();
        // Place a blocking room between start and end
        grid.reserve_rect(80, 16, 64, 64, RoomId(0)).unwrap();

        let result = route_sockets(
            (16, 48),
            (208, 48),
            &grid,
            1024,
            10000,
            RoomId(0),
            RoomId(0),
        );
        assert!(result.is_ok(), "should find path around obstacle");
    }

    #[test]
    fn route_exhaustion_small_budget() {
        let grid = make_grid();
        let result = route_sockets((16, 48), (900, 900), &grid, 1024, 5, RoomId(0), RoomId(0));
        assert!(result.is_err());
        match result.unwrap_err() {
            EnhancedError::RouteExhausted { .. } => {}
            e => panic!("expected RouteExhausted, got {:?}", e),
        }
    }

    #[test]
    fn route_envelope_clearance() {
        let mut grid = make_grid();
        // Place a room that partially blocks the corridor envelope
        grid.reserve_rect(16, 80, 64, 64, RoomId(0)).unwrap();

        // Route should go around, not through
        let result = route_sockets(
            (48, 16),
            (48, 208),
            &grid,
            1024,
            10000,
            RoomId(0),
            RoomId(0),
        )
        .unwrap();
        for seg in &result.segments {
            let (ex0, ey0, ex1, ey1) = seg.envelope;
            for rx in [ex0, ex1 - 16] {
                for ry in [ey0, ey1 - 16] {
                    // Envelope cells should not overlap room cells
                    // (room is at 16..80, 80..144)
                    let room_x0 = 16;
                    let room_y0 = 80;
                    let room_x1 = 80;
                    let room_y1 = 144;
                    let overlaps =
                        rx < room_x1 && rx + 16 > room_x0 && ry < room_y1 && ry + 16 > room_y0;
                    assert!(
                        !overlaps,
                        "segment envelope {:?} overlaps room at ({},{},{},{})",
                        seg.envelope, room_x0, room_y0, room_x1, room_y1
                    );
                }
            }
        }
    }

    #[test]
    fn deterministic_same_input_same_path() {
        let grid = make_grid();
        let r1 = route_sockets(
            (16, 48),
            (208, 160),
            &grid,
            1024,
            10000,
            RoomId(0),
            RoomId(0),
        )
        .unwrap();
        let r2 = route_sockets(
            (16, 48),
            (208, 160),
            &grid,
            1024,
            10000,
            RoomId(0),
            RoomId(0),
        )
        .unwrap();
        assert_eq!(r1.segments, r2.segments);
    }

    #[test]
    fn corridor_width_enforced() {
        let grid = make_grid();
        let result = route_sockets(
            (16, 48),
            (208, 48),
            &grid,
            1024,
            10000,
            RoomId(0),
            RoomId(0),
        )
        .unwrap();
        for seg in &result.segments {
            let (ex0, ey0, ex1, ey1) = seg.envelope;
            let ew = ex1 - ex0;
            let eh = ey1 - ey0;
            // At least one dimension must equal corridor width
            assert!(
                ew == CORRIDOR_WIDTH || eh == CORRIDOR_WIDTH,
                "envelope {}x{} must have corridor width {}",
                ew,
                eh,
                CORRIDOR_WIDTH
            );
        }
    }
}
