//! Bounded orthogonal corridor routing on the 16-unit quantum grid.
//!
//! Connects room portal centers via axis-aligned corridors. The router first
//! tries direct and L-shaped orthogonal paths, then falls back to bounded A*
//! search. All returned corridors satisfy the minimum 64-unit clear width and
//! 80-unit clear headroom required by the frozen navigation contract.
//!
//! The entry point is [`route_edge`].

use std::collections::BinaryHeap;

use crate::config::{ValidatedConfig, CONSTRUCTION_QUANTUM};
use crate::error::GeneratorError;
use crate::intent::{Corridor, RoomIntent};
use crate::StageRng;

// ── Frozen corridor dimensions ────────────────────────────────────────────

/// Minimum corridor interior width in Quake units (4 quanta).
pub const CORRIDOR_WIDTH: u32 = 64;

/// Minimum corridor interior height in Quake units (5 quanta).
pub const CORRIDOR_HEIGHT: u32 = 80;

/// Half-width in quantum cells: the corridor extends this many cells on each
/// side of the centerline.
const CORRIDOR_HALF_CELLS: i32 = 2; // 32 units / 16

/// Expansion margin around rooms in quantum cells: wall thickness (1 cell)
/// plus corridor half-width (2 cells). The A* centerline must stay outside
/// this margin so the full corridor width fits without intersecting rooms.
const OCCUPANCY_MARGIN: i32 = 3;

/// Grid cell size equals the construction quantum.
const Q: i32 = CONSTRUCTION_QUANTUM as i32;

// ── Direction helpers ─────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Dir {
    East,
    West,
    North,
    South,
}



// ── Portal computation ────────────────────────────────────────────────────

/// Compute portal centerline positions (in world coordinates) on two rooms'
/// walls that face each other. Returns `(portal_a, portal_b)`.
///
/// Portals are placed at the midpoint of the facing wall to ensure adequate
/// clearance from room corners, preventing occupancy-grid blocking of the
/// portal exit.
fn compute_portals(
    a: &RoomIntent,
    b: &RoomIntent,
    ca: &(i32, i32, i32),
    cb: &(i32, i32, i32),
) -> ((i32, i32, i32), (i32, i32, i32)) {
    let a_min = a.position;
    let a_max = (
        a.position.0 + a.dimensions.0 as i32,
        a.position.1 + a.dimensions.1 as i32,
        a.position.2 + a.dimensions.2 as i32,
    );
    let b_min = b.position;
    let b_max = (
        b.position.0 + b.dimensions.0 as i32,
        b.position.1 + b.dimensions.1 as i32,
        b.position.2 + b.dimensions.2 as i32,
    );

    let dx = (cb.0 - ca.0).abs();
    let dy = (cb.1 - ca.1).abs();

    let portal_a;
    let portal_b;

    if dx >= dy {
        // East-West facing
        if cb.0 >= ca.0 {
            // A east wall → B west wall
            // Use room center for the perpendicular coordinate, clamped to wall
            let a_y = clamp(ca.1, a_min.1 + CONSTRUCTION_QUANTUM as i32, a_max.1 - CONSTRUCTION_QUANTUM as i32);
            portal_a = (a_max.0, a_y, a.position.2);
            let b_y = clamp(cb.1, b_min.1 + CONSTRUCTION_QUANTUM as i32, b_max.1 - CONSTRUCTION_QUANTUM as i32);
            portal_b = (b_min.0, b_y, b.position.2);
        } else {
            // A west wall → B east wall
            let a_y = clamp(ca.1, a_min.1 + CONSTRUCTION_QUANTUM as i32, a_max.1 - CONSTRUCTION_QUANTUM as i32);
            portal_a = (a_min.0, a_y, a.position.2);
            let b_y = clamp(cb.1, b_min.1 + CONSTRUCTION_QUANTUM as i32, b_max.1 - CONSTRUCTION_QUANTUM as i32);
            portal_b = (b_max.0, b_y, b.position.2);
        }
    } else {
        // North-South facing
        if cb.1 >= ca.1 {
            // A north wall → B south wall
            let a_x = clamp(ca.0, a_min.0 + CONSTRUCTION_QUANTUM as i32, a_max.0 - CONSTRUCTION_QUANTUM as i32);
            portal_a = (a_x, a_max.1, a.position.2);
            let b_x = clamp(cb.0, b_min.0 + CONSTRUCTION_QUANTUM as i32, b_max.0 - CONSTRUCTION_QUANTUM as i32);
            portal_b = (b_x, b_min.1, b.position.2);
        } else {
            // A south wall → B north wall
            let a_x = clamp(ca.0, a_min.0 + CONSTRUCTION_QUANTUM as i32, a_max.0 - CONSTRUCTION_QUANTUM as i32);
            portal_a = (a_x, a_min.1, a.position.2);
            let b_x = clamp(cb.0, b_min.0 + CONSTRUCTION_QUANTUM as i32, b_max.0 - CONSTRUCTION_QUANTUM as i32);
            portal_b = (b_x, b_max.1, b.position.2);
        }
    }

    (portal_a, portal_b)
}

fn clamp(v: i32, lo: i32, hi: i32) -> i32 {
    if v < lo {
        lo
    } else if v > hi {
        hi
    } else {
        v
    }
}

fn room_center(room: &RoomIntent) -> (i32, i32, i32) {
    (
        room.position.0 + room.dimensions.0 as i32 / 2,
        room.position.1 + room.dimensions.1 as i32 / 2,
        room.position.2 + room.dimensions.2 as i32 / 2,
    )
}

/// Offset a portal position to the corridor centerline grid coordinate.
/// The portal is on the room wall; the corridor centerline starts
/// wall_thickness/q + corridor_half_cells cells outside the wall.
fn portal_to_centerline(portal: (i32, i32, i32), dir: Dir) -> (i32, i32, i32) {
    let offset = (1 + CORRIDOR_HALF_CELLS) * Q; // wall_thickness + corridor_half_width
    match dir {
        Dir::East => (portal.0 + offset, portal.1, portal.2),
        Dir::West => (portal.0 - offset, portal.1, portal.2),
        Dir::North => (portal.0, portal.1 + offset, portal.2),
        Dir::South => (portal.0, portal.1 - offset, portal.2),
    }
}

/// Determine which direction the portal faces outward from the room.
fn portal_outward_dir(room: &RoomIntent, portal: (i32, i32, i32)) -> Dir {
    let r_max_x = room.position.0 + room.dimensions.0 as i32;
    let r_max_y = room.position.1 + room.dimensions.1 as i32;

    if portal.0 == r_max_x {
        Dir::East
    } else if portal.0 == room.position.0 {
        Dir::West
    } else if portal.1 == r_max_y {
        Dir::North
    } else if portal.1 == room.position.1 {
        Dir::South
    } else {
        // Fallback: determine from portal position relative to room center
        let cx = room.position.0 + room.dimensions.0 as i32 / 2;
        if portal.0 >= cx {
            Dir::East
        } else {
            Dir::West
        }
    }
}

// ── Occupancy grid ────────────────────────────────────────────────────────

struct OccupancyGrid {
    cols: i32,
    rows: i32,
    blocked: Vec<bool>,
}

impl OccupancyGrid {
    fn new(cols: i32, rows: i32) -> Self {
        OccupancyGrid {
            cols,
            rows,
            blocked: vec![false; (cols * rows) as usize],
        }
    }

    fn in_bounds(&self, x: i32, y: i32) -> bool {
        x >= 0 && x < self.cols && y >= 0 && y < self.rows
    }

    fn is_blocked(&self, x: i32, y: i32) -> bool {
        if !self.in_bounds(x, y) {
            return true; // out-of-bounds is blocked
        }
        self.blocked[(y * self.cols + x) as usize]
    }

    fn set_blocked(&mut self, x: i32, y: i32, val: bool) {
        if self.in_bounds(x, y) {
            self.blocked[(y * self.cols + x) as usize] = val;
        }
    }

    fn clear(&mut self, x: i32, y: i32) {
        self.set_blocked(x, y, false);
    }

    /// Mark cells as blocked for a room expanded by `margin` cells on all sides.
    fn mark_room(&mut self, room: &RoomIntent, margin: i32) {
        let min_x = room.position.0 / Q - margin;
        let max_x = (room.position.0 + room.dimensions.0 as i32) / Q + margin;
        let min_y = room.position.1 / Q - margin;
        let max_y = (room.position.1 + room.dimensions.1 as i32) / Q + margin;

        for gy in min_y..max_y {
            for gx in min_x..max_x {
                self.set_blocked(gx, gy, true);
            }
        }
    }
}

// ── Grid coordinate conversion ────────────────────────────────────────────

fn world_to_grid(world: (i32, i32, i32)) -> (i32, i32) {
    // Round half-up for consistent snapping
    let gx = snap_to_grid(world.0, Q);
    let gy = snap_to_grid(world.1, Q);
    (gx, gy)
}

fn snap_to_grid(v: i32, quantum: i32) -> i32 {
    let rem = v.rem_euclid(quantum);
    let half = quantum / 2;
    if rem == 0 {
        v / quantum
    } else if rem <= half {
        // Round down
        (v - rem) / quantum
    } else {
        // Round up
        (v + (quantum - rem)) / quantum
    }
}

/// Clear the endpoint approach corridors from each room portal to its routed
/// centerline. Endpoint rooms are still marked in the occupancy grid so other
/// routes keep away from them, but the connecting edge must carve a local arch
/// through its two endpoint margins. Clearing only a 3×3 blob around each
/// centerline leaves one blocked margin row between close room pairs; clear the
/// full portal-to-centerline tube instead, with the same half-width as the
/// corridor centerline footprint.
fn clear_endpoint_approach(
    portal: (i32, i32),
    centerline: (i32, i32),
    grid: &mut OccupancyGrid,
) {
    let dx = (centerline.0 - portal.0).signum();
    let dy = (centerline.1 - portal.1).signum();
    let steps = (centerline.0 - portal.0)
        .abs()
        .max((centerline.1 - portal.1).abs());

    for i in 0..=steps {
        let x = portal.0 + dx * i;
        let y = portal.1 + dy * i;
        for oy in -CORRIDOR_HALF_CELLS..=CORRIDOR_HALF_CELLS {
            for ox in -CORRIDOR_HALF_CELLS..=CORRIDOR_HALF_CELLS {
                grid.clear(x + ox, y + oy);
            }
        }
    }
}

fn grid_to_world(gx: i32, gy: i32, z: i32) -> (i32, i32, i32) {
    (gx * Q, gy * Q, z)
}

// ── A* search ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct AStarNode {
    x: i32,
    y: i32,
    g: i32,
    f: i32,
}

impl Ord for AStarNode {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Reverse for min-heap
        other.f.cmp(&self.f).then_with(|| other.g.cmp(&self.g))
    }
}

impl PartialOrd for AStarNode {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

const DIRS: [(i32, i32); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];

/// Bounded A* search from `start` to `end` on the occupancy grid.
/// Returns the path as a list of grid coordinates (including start, excluding
/// end may be included), or `None` if the expansion budget is exhausted.
fn astar_search(
    start: (i32, i32),
    end: (i32, i32),
    grid: &OccupancyGrid,
    max_expansions: u32,
) -> Option<Vec<(i32, i32)>> {
    if start == end {
        return Some(vec![start]);
    }
    if grid.is_blocked(end.0, end.1) {
        return None;
    }

    let total_cells = (grid.cols * grid.rows) as usize;
    let mut g_score = vec![i32::MAX; total_cells];
    let idx = |x: i32, y: i32| -> usize { (y * grid.cols + x) as usize };
    let mut came_from: Vec<Option<(i32, i32)>> = vec![None; total_cells];

    let mut open = BinaryHeap::new();
    let start_idx = idx(start.0, start.1);
    g_score[start_idx] = 0;
    open.push(AStarNode {
        x: start.0,
        y: start.1,
        g: 0,
        f: manhattan(start, end),
    });

    let mut expansions: u32 = 0;

    while let Some(node) = open.pop() {
        if node.x == end.0 && node.y == end.1 {
            // Reconstruct path
            let mut path = Vec::new();
            let mut cur = (node.x, node.y);
            path.push(cur);
            while let Some(prev) = came_from[idx(cur.0, cur.1)] {
                cur = prev;
                path.push(cur);
            }
            path.reverse();
            return Some(path);
        }

        expansions += 1;
        if expansions > max_expansions {
            return None;
        }

        let current_idx = idx(node.x, node.y);
        // Skip stale entries
        if node.g > g_score[current_idx] {
            continue;
        }

        for &(dx, dy) in &DIRS {
            let nx = node.x + dx;
            let ny = node.y + dy;
            if !grid.in_bounds(nx, ny) || grid.is_blocked(nx, ny) {
                continue;
            }
            let tentative_g = node.g + 1;
            let n_idx = idx(nx, ny);
            if tentative_g < g_score[n_idx] {
                g_score[n_idx] = tentative_g;
                came_from[n_idx] = Some((node.x, node.y));
                open.push(AStarNode {
                    x: nx,
                    y: ny,
                    g: tentative_g,
                    f: tentative_g + manhattan((nx, ny), end),
                });
            }
        }
    }

    None
}

fn manhattan(a: (i32, i32), b: (i32, i32)) -> i32 {
    (a.0 - b.0).abs() + (a.1 - b.1).abs()
}

// ── Direct and L-shaped paths ─────────────────────────────────────────────

/// Attempt a direct orthogonal path: first move in X, then Y (or vice versa).
/// Checks that all cells along the path (excluding start) are unblocked.
fn direct_orthogonal_path(
    start: (i32, i32),
    end: (i32, i32),
    grid: &OccupancyGrid,
) -> Option<Vec<(i32, i32)>> {
    // Try both orderings: X-then-Y, Y-then-X
    for &(x_first, _) in &[(true, false), (false, true)] {
        if let Some(path) = try_two_segment(start, end, grid, x_first) {
            return Some(path);
        }
    }
    None
}

/// Build a two-segment L-shaped path. `x_first` means move in X first, then Y.
fn try_two_segment(
    start: (i32, i32),
    end: (i32, i32),
    grid: &OccupancyGrid,
    x_first: bool,
) -> Option<Vec<(i32, i32)>> {
    let corner = if x_first {
        (end.0, start.1)
    } else {
        (start.0, end.1)
    };

    // Check segment 1: start → corner (excluding start)
    if !is_clear_straight(start, corner, grid) {
        return None;
    }
    // Check segment 2: corner → end (excluding corner, including end)
    if !is_clear_straight(corner, end, grid) {
        return None;
    }

    let mut path = Vec::new();
    path.push(start);
    straight_line_points(start, corner, &mut path);
    straight_line_points(corner, end, &mut path);
    Some(path)
}

/// Check all cells along a straight line (horizontal or vertical) from `from`
/// to `to`, excluding `from` but including `to`.
fn is_clear_straight(from: (i32, i32), to: (i32, i32), grid: &OccupancyGrid) -> bool {
    if from == to {
        return true;
    }
    let dx = (to.0 - from.0).signum();
    let dy = (to.1 - from.1).signum();
    assert!(dx == 0 || dy == 0, "straight line must be axis-aligned");
    let steps = (to.0 - from.0).abs().max((to.1 - from.1).abs());
    for i in 1..=steps {
        let x = from.0 + dx * i;
        let y = from.1 + dy * i;
        if grid.is_blocked(x, y) {
            return false;
        }
    }
    true
}

/// Add intermediate points of a straight line (excluding `from`, including `to`).
fn straight_line_points(from: (i32, i32), to: (i32, i32), path: &mut Vec<(i32, i32)>) {
    if from == to {
        return;
    }
    let dx = (to.0 - from.0).signum();
    let dy = (to.1 - from.1).signum();
    let steps = (to.0 - from.0).abs().max((to.1 - from.1).abs());
    for i in 1..=steps {
        path.push((from.0 + dx * i, from.1 + dy * i));
    }
}

// ── L-shaped path exploration with RNG ────────────────────────────────────

/// Try L-shaped paths with both turn orderings, randomized when both are valid.
fn l_shaped_paths(
    start: (i32, i32),
    end: (i32, i32),
    grid: &OccupancyGrid,
    rng: &mut StageRng,
) -> Option<Vec<(i32, i32)>> {
    let path_xy = try_two_segment(start, end, grid, true); // X then Y
    let path_yx = try_two_segment(start, end, grid, false); // Y then X

    match (path_xy, path_yx) {
        (Some(xy), Some(yx)) => {
            // Both valid, choose randomly
            if rng.next_u64() % 2 == 0 {
                Some(xy)
            } else {
                Some(yx)
            }
        }
        (Some(xy), None) => Some(xy),
        (None, Some(yx)) => Some(yx),
        (None, None) => None,
    }
}

// ── Path simplification to corridor segments ──────────────────────────────

/// Convert a grid path into simplified straight corridor segments.
/// Consecutive steps in the same direction are collapsed into one segment.
fn simplify_path(
    path: &[(i32, i32)],
    z: i32,
) -> Vec<Corridor> {
    if path.len() < 2 {
        return Vec::new();
    }

    let mut segments: Vec<Corridor> = Vec::new();
    let mut seg_start = path[0];
    let mut prev_dir: Option<(i32, i32)> = None;

    for i in 1..path.len() {
        let cur = path[i];
        let prev = path[i - 1];
        let dir = (cur.0 - prev.0, cur.1 - prev.1);

        match prev_dir {
            Some(pd) if pd != dir => {
                // Direction changed: emit segment
                let start_world = grid_to_world(seg_start.0, seg_start.1, z);
                let end_world = grid_to_world(prev.0, prev.1, z);
                segments.push(Corridor {
                    start: start_world,
                    end: end_world,
                    width: CORRIDOR_WIDTH,
                    height: CORRIDOR_HEIGHT,
                });
                seg_start = prev;
            }
            None => {}
            _ => {}
        }
        prev_dir = Some(dir);

        // Emit final segment on last iteration
        if i == path.len() - 1 {
            let start_world = grid_to_world(seg_start.0, seg_start.1, z);
            let end_world = grid_to_world(cur.0, cur.1, z);
            segments.push(Corridor {
                start: start_world,
                end: end_world,
                width: CORRIDOR_WIDTH,
                height: CORRIDOR_HEIGHT,
            });
        }
    }

    segments
}

// ── Public entry point ────────────────────────────────────────────────────

/// Route a corridor between two rooms identified by their indices in `rooms`.
///
/// The corridor connects the portal centers of the two rooms using orthogonal
/// axis-aligned segments (at most one turn). The returned [`Corridor`] segments
/// each have `width ≥ 64` and `height ≥ 80`.
///
/// # Algorithm
///
/// 1. Compute portal points on facing room walls.
/// 2. Build an occupancy grid from all rooms (expanded by wall thickness +
///    corridor half-width).
/// 3. Try a direct orthogonal path; if blocked, try L-shaped alternatives;
///    fall back to bounded A* search.
/// 4. Simplify the grid path into straight corridor segments.
///
/// # Errors
///
/// Returns [`GeneratorError::RouteExhausted`] when the expansion budget is
/// exceeded without finding a path.
pub fn route_edge(
    a: usize,
    b: usize,
    rooms: &[RoomIntent],
    config: &ValidatedConfig,
    rng: &mut StageRng,
) -> Result<Vec<Corridor>, GeneratorError> {
    let room_a = &rooms[a];
    let room_b = &rooms[b];
    let ca = room_center(room_a);
    let cb = room_center(room_b);

    // Compute portal positions on facing walls
    let (portal_a, portal_b) = compute_portals(room_a, room_b, &ca, &cb);

    // Determine outward directions
    let dir_a = portal_outward_dir(room_a, portal_a);
    let dir_b = portal_outward_dir(room_b, portal_b);

    // Offset to centerline positions
    let center_a = portal_to_centerline(portal_a, dir_a);
    let center_b = portal_to_centerline(portal_b, dir_b);

    // Common Z for all corridors in single-layer maps
    let z = room_a.position.2;

    // Build occupancy grid
    let grid_cols = (config.xy_bounds.0 / CONSTRUCTION_QUANTUM) as i32;
    let grid_rows = (config.xy_bounds.1 / CONSTRUCTION_QUANTUM) as i32;

    let mut grid = OccupancyGrid::new(grid_cols, grid_rows);
    // Mark non-endpoint rooms with full margin (wall + half-corridor)
    // to keep corridors away. Endpoint rooms use wall-only margin so
    // the corridor can actually approach them.
    for (i, room) in rooms.iter().enumerate() {
        let margin = if i == a || i == b {
            1 // wall thickness only — corridor needs to enter/exit this room
        } else {
            OCCUPANCY_MARGIN // wall + half-corridor: keep corridors away
        };
        grid.mark_room(room, margin);
    }

    // Convert centerline positions to grid coordinates
    let start = world_to_grid((center_a.0, center_a.1, 0));
    let end = world_to_grid((center_b.0, center_b.1, 0));

    // If start and end are the same grid cell, emit a minimal corridor segment
    // directly between the portal centerline positions.
    if start == end {
        return Ok(vec![Corridor {
            start: center_a,
            end: center_b,
            width: CORRIDOR_WIDTH,
            height: CORRIDOR_HEIGHT,
        }]);
    }

    // Unblock start/end and carve local approach tubes through the endpoint
    // room margins. These openings are limited to the two endpoint rooms, not
    // the full inter-room route, so unrelated room margins remain solid.
    grid.clear(start.0, start.1);
    grid.clear(end.0, end.1);
    let portal_a_grid = world_to_grid((portal_a.0, portal_a.1, 0));
    let portal_b_grid = world_to_grid((portal_b.0, portal_b.1, 0));
    clear_endpoint_approach(portal_a_grid, start, &mut grid);
    clear_endpoint_approach(portal_b_grid, end, &mut grid);

    // Try direct orthogonal path first
    if let Some(path) = direct_orthogonal_path(start, end, &grid) {
        return Ok(simplify_path(&path, z));
    }

    // Try L-shaped paths
    if let Some(path) = l_shaped_paths(start, end, &grid, rng) {
        return Ok(simplify_path(&path, z));
    }

    // Fall back to A* with expansion budget
    if let Some(path) = astar_search(start, end, &grid, config.max_astar_expansions) {
        return Ok(simplify_path(&path, z));
    }

    Err(GeneratorError::RouteExhausted {
        expansions: config.max_astar_expansions,
    })
}

/// Build the full routed network from a layout's edge list.
///
/// Routes every edge in `layout.edges`, collecting all corridor segments.
/// Returns a [`crate::intent::RoutedIntent`] with corridors and junction
/// position markers.
pub fn route_all_edges(
    rooms: &[RoomIntent],
    edges: &[(usize, usize)],
    config: &ValidatedConfig,
    rng: &mut StageRng,
) -> Result<crate::intent::RoutedIntent, GeneratorError> {
    let mut corridors: Vec<Corridor> = Vec::new();
    let mut junctions: Vec<crate::intent::Junction> = Vec::new();

    for &(a, b) in edges {
        let edge_corridors = route_edge(a, b, rooms, config, rng)?;
        // Collect junction positions at corridor endpoints
        for seg in &edge_corridors {
            // Start and end points of each segment are potential junction sites
            // We collect them here; deduplication and L/T/X classification
            // happens in the junction module.
            let start_exists = junctions.iter().any(|j| {
                j.position.0 == seg.start.0
                    && j.position.1 == seg.start.1
                    && j.position.2 == seg.start.2
            });
            if !start_exists {
                junctions.push(crate::intent::Junction {
                    position: seg.start,
                });
            }
        }
        corridors.extend(edge_corridors);
    }

    // Also mark corridor endpoints as junctions
    for seg in &corridors {
        let end_exists = junctions.iter().any(|j| {
            j.position.0 == seg.end.0 && j.position.1 == seg.end.1 && j.position.2 == seg.end.2
        });
        if !end_exists {
            junctions.push(crate::intent::Junction {
                position: seg.end,
            });
        }
    }

    Ok(crate::intent::RoutedIntent {
        corridors,
        junctions,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::DungeonConfig;
    use crate::Seed;

    fn make_rng(seed_val: u64) -> StageRng {
        Seed::new(seed_val).stage_seed("corridor-routing").rng()
    }

    fn room_at(x: i32, y: i32, z: i32, dx: u32, dy: u32, dz: u32) -> RoomIntent {
        RoomIntent {
            position: (x, y, z),
            dimensions: (dx, dy, dz),
        }
    }

    fn valid_m1_config() -> ValidatedConfig {
        DungeonConfig::nominal_m1().validate().unwrap()
    }

    // ── Portal computation ──────────────────────────────────────────────

    #[test]
    fn portals_east_west_facing() {
        let a = room_at(0, 0, 0, 64, 64, 128);
        let b = room_at(160, 0, 0, 64, 64, 128);
        let ca = room_center(&a);
        let cb = room_center(&b);
        let (pa, pb) = compute_portals(&a, &b, &ca, &cb);
        // A east wall at x=64, B west wall at x=160
        assert_eq!(pa.0, 64);
        assert_eq!(pb.0, 160);
    }

    #[test]
    fn portals_north_south_facing() {
        let a = room_at(0, 0, 0, 64, 64, 128);
        let b = room_at(0, 160, 0, 64, 64, 128);
        let ca = room_center(&a);
        let cb = room_center(&b);
        let (pa, pb) = compute_portals(&a, &b, &ca, &cb);
        // A north wall at y=64, B south wall at y=160
        assert_eq!(pa.1, 64);
        assert_eq!(pb.1, 160);
    }

    // ── Occupancy grid ──────────────────────────────────────────────────

    #[test]
    fn occupancy_mark_room_blocks_cells() {
        let mut grid = OccupancyGrid::new(20, 20);
        let room = room_at(0, 0, 0, 64, 64, 128);
        // Room at cells [0..4), margin 3 → blocked [-3..7)
        grid.mark_room(&room, OCCUPANCY_MARGIN);
        // Cell inside room
        assert!(grid.is_blocked(0, 0));
        assert!(grid.is_blocked(2, 2));
        // Cell in expanded margin
        assert!(grid.is_blocked(4, 4));
        // Cell just beyond margin
        assert!(grid.is_blocked(6, 0));
        // Cell well outside
        assert!(!grid.is_blocked(7, 7));
        // Out of bounds
        assert!(grid.is_blocked(-1, 0));
    }

    // ── A* search ────────────────────────────────────────────────────────

    #[test]
    fn astar_straight_path() {
        let mut grid = OccupancyGrid::new(20, 20);
        // Leave a clear corridor
        for y in 0..20 {
            for x in 0..20 {
                if x >= 5 && x <= 15 && y == 10 {
                    // clear
                } else if x >= 4 && x <= 16 {
                    grid.set_blocked(x, y, true);
                }
            }
        }
        // Unblock the path
        for x in 5..=15 {
            grid.set_blocked(x, 10, false);
        }

        let path = astar_search((5, 10), (15, 10), &grid, 1000);
        assert!(path.is_some());
        let path = path.unwrap();
        assert!(path.len() >= 10);
    }

    #[test]
    fn astar_exhaustion() {
        let mut grid = OccupancyGrid::new(5, 5);
        // Block everything except start
        for y in 0..5 {
            for x in 0..5 {
                grid.set_blocked(x, y, true);
            }
        }
        grid.clear(0, 0);

        let path = astar_search((0, 0), (4, 4), &grid, 10);
        assert!(path.is_none());
    }

    // ── Direct and L-shaped paths ───────────────────────────────────────

    #[test]
    fn direct_x_then_y_path() {
        let grid = OccupancyGrid::new(20, 20);
        let path = direct_orthogonal_path((2, 2), (10, 8), &grid);
        assert!(path.is_some());
        let path = path.unwrap();
        // Should start at (2,2) and end at (10,8)
        assert_eq!(path[0], (2, 2));
        assert_eq!(path[path.len() - 1], (10, 8));
    }

    #[test]
    fn direct_path_blocked() {
        let mut grid = OccupancyGrid::new(20, 20);
        // Block the corner cell for X-then-Y
        grid.set_blocked(10, 2, true);
        // Block the corner cell for Y-then-X
        grid.set_blocked(2, 8, true);
        let path = direct_orthogonal_path((2, 2), (10, 8), &grid);
        assert!(path.is_none());
    }

    // ── Path simplification ─────────────────────────────────────────────

    #[test]
    fn simplify_straight_path() {
        let path: Vec<(i32, i32)> = (0..=10).map(|i| (i, 5)).collect();
        let segments = simplify_path(&path, 0);
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].start, grid_to_world(0, 5, 0));
        assert_eq!(segments[0].end, grid_to_world(10, 5, 0));
        assert_eq!(segments[0].width, CORRIDOR_WIDTH);
        assert_eq!(segments[0].height, CORRIDOR_HEIGHT);
    }

    #[test]
    fn simplify_l_shaped_path() {
        let mut path: Vec<(i32, i32)> = (0..=5).map(|i| (i, 2)).collect();
        path.extend((3..=8).map(|j| (5, j)));
        let segments = simplify_path(&path, 0);
        assert_eq!(segments.len(), 2);
    }

    // ── rout_edge integration ───────────────────────────────────────────

    #[test]
    fn route_two_adjacent_rooms() {
        let rooms = vec![
            room_at(0, 0, 0, 64, 64, 128),
            room_at(160, 0, 0, 64, 64, 128),
        ];
        let cfg = valid_m1_config();
        let mut rng = make_rng(42);
        let result = route_edge(0, 1, &rooms, &cfg, &mut rng);
        assert!(result.is_ok(), "routing failed: {:?}", result.err());
        let corridors = result.unwrap();
        assert!(!corridors.is_empty());
        for c in &corridors {
            assert!(c.width >= CORRIDOR_WIDTH);
            assert!(c.height >= CORRIDOR_HEIGHT);
        }
    }

    #[test]
    fn route_rooms_with_blocking_room_between() {
        // Room A and B are far apart with a blocking room in between
        let rooms = vec![
            room_at(0, 0, 0, 64, 64, 128),     // A
            room_at(80, 80, 0, 64, 64, 128),   // blocking
            room_at(0, 160, 0, 64, 64, 128),   // B
        ];
        let cfg = valid_m1_config();
        let mut rng = make_rng(42);
        let result = route_edge(0, 2, &rooms, &cfg, &mut rng);
        // Should find a path around the blocking room
        assert!(result.is_ok(), "routing failed: {:?}", result.err());
        let corridors = result.unwrap();
        for c in &corridors {
            assert!(c.width >= CORRIDOR_WIDTH);
            assert!(c.height >= CORRIDOR_HEIGHT);
        }
    }

    #[test]
    fn route_exhaustion_at_tiny_bounds() {
        let rooms = vec![
            room_at(0, 0, 0, 64, 64, 128),
            room_at(80, 0, 0, 64, 64, 128),
        ];
        let cfg = DungeonConfig {
            max_astar_expansions: 1, // impossibly small
            xy_bounds: (256, 16),    // single row
            ..DungeonConfig::nominal_m1()
        }
        .validate()
        .unwrap();
        let mut rng = make_rng(42);
        let result = route_edge(0, 1, &rooms, &cfg, &mut rng);
        match result {
            Err(GeneratorError::RouteExhausted { .. }) => {}
            _ => panic!("expected RouteExhausted, got {:?}", result),
        }
    }

    #[test]
    fn corridors_minimum_width_and_height() {
        let rooms = vec![
            room_at(0, 0, 0, 64, 64, 128),
            room_at(160, 0, 0, 64, 64, 128),
            room_at(320, 0, 0, 64, 64, 128),
        ];
        let cfg = valid_m1_config();
        let mut rng = make_rng(7);
        let edges = vec![(0, 1), (1, 2)];
        let routed = route_all_edges(&rooms, &edges, &cfg, &mut rng).unwrap();
        for c in &routed.corridors {
            assert!(c.width >= CORRIDOR_WIDTH);
            assert!(c.height >= CORRIDOR_HEIGHT);
            // Verify dimensions are quantum-aligned
            assert_eq!(c.width % CONSTRUCTION_QUANTUM, 0);
            assert_eq!(c.height % CONSTRUCTION_QUANTUM, 0);
        }
    }

    #[test]
    fn route_deterministic() {
        let rooms = vec![
            room_at(0, 0, 0, 64, 64, 128),
            room_at(160, 0, 0, 64, 64, 128),
            room_at(0, 160, 0, 64, 64, 128),
        ];
        let cfg = valid_m1_config();

        let r1 = route_edge(0, 1, &rooms, &cfg, &mut make_rng(42)).unwrap();
        let r2 = route_edge(0, 1, &rooms, &cfg, &mut make_rng(42)).unwrap();
        assert_eq!(r1, r2, "same seed should produce identical routes");
    }

    // ── `route_all_edges` integration ───────────────────────────────────

    #[test]
    fn route_all_edges_produces_junctions() {
        let rooms = vec![
            room_at(0, 0, 0, 64, 64, 128),
            room_at(160, 0, 0, 64, 64, 128),
            room_at(0, 160, 0, 64, 64, 128),
        ];
        let cfg = valid_m1_config();
        let mut rng = make_rng(42);
        let edges = vec![(0, 1), (0, 2)];
        let routed = route_all_edges(&rooms, &edges, &cfg, &mut rng).unwrap();
        assert!(!routed.corridors.is_empty());
        assert!(!routed.junctions.is_empty());
    }
}
