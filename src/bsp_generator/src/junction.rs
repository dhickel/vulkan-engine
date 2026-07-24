//! Explicit junction geometry for corridor-to-corridor and corridor-to-room
//! connections.
//!
//! Every junction produces explicit closure brushes (6 faces per brush) rather
//! than relying on CSG ambiguity from overlapping brushes. This guarantees
//! sealed maps with no gaps at corridor intersections.
//!
//! Supported junction types:
//! - **L-junction**: two perpendicular corridors meet at a corner turn.
//! - **T-junction**: one corridor terminates into a through corridor.
//! - **X-junction**: two corridors cross at right angles.
//! - **Room portal**: corridor meets a room wall (produces opening marker).
//!
//! Each closure brush is an axis-aligned rectangular prism whose faces use the
//! canonical face ordering: bottom, top, north, south, west, east.

use crate::config::CONSTRUCTION_QUANTUM;
use crate::intent::{Brush, BrushFace, Corridor, RoomIntent};

/// Default wall texture name for generated closure brushes.
const DEFAULT_WALL_TEXTURE: &str = "generator_brick";

/// Create an axis-aligned solid brush from `(min_x, min_y, min_z)` to
/// `(max_x, max_y, max_z)` with the given texture.
///
/// All coordinates must be multiples of [`CONSTRUCTION_QUANTUM`].
/// The 6 faces are ordered: bottom, top, north, south, west, east.
pub fn make_brush(
    min: (i32, i32, i32),
    max: (i32, i32, i32),
    texture: &str,
) -> Brush {
    let tex = texture.to_string();
    let default_axis = [1, 0, 0, 0];

    Brush {
        faces: vec![
            // Bottom face (z = min.2)
            BrushFace {
                plane_points: [
                    (min.0, max.1, min.2),
                    (min.0, min.1, min.2),
                    (max.0, min.1, min.2),
                ],
                texture: tex.clone(),
                u_axis: default_axis,
                v_axis: default_axis,
            },
            // Top face (z = max.2)
            BrushFace {
                plane_points: [
                    (min.0, max.1, max.2),
                    (max.0, max.1, max.2),
                    (max.0, min.1, max.2),
                ],
                texture: tex.clone(),
                u_axis: default_axis,
                v_axis: default_axis,
            },
            // North face (y = max.1)
            BrushFace {
                plane_points: [
                    (min.0, max.1, max.2),
                    (min.0, max.1, min.2),
                    (max.0, max.1, min.2),
                ],
                texture: tex.clone(),
                u_axis: default_axis,
                v_axis: default_axis,
            },
            // South face (y = min.1)
            BrushFace {
                plane_points: [
                    (min.0, min.1, max.2),
                    (max.0, min.1, max.2),
                    (max.0, min.1, min.2),
                ],
                texture: tex.clone(),
                u_axis: default_axis,
                v_axis: default_axis,
            },
            // West face (x = min.0)
            BrushFace {
                plane_points: [
                    (min.0, max.1, max.2),
                    (min.0, min.1, max.2),
                    (min.0, min.1, min.2),
                ],
                texture: tex.clone(),
                u_axis: default_axis,
                v_axis: default_axis,
            },
            // East face (x = max.0)
            BrushFace {
                plane_points: [
                    (max.0, max.1, min.2),
                    (max.0, min.1, min.2),
                    (max.0, min.1, max.2),
                ],
                texture: tex,
                u_axis: default_axis,
                v_axis: default_axis,
            },
        ],
    }
}

// ── Corridor extent helpers ───────────────────────────────────────────────

/// Compute the bounding box of a corridor's solid wall region.
/// Returns `(min, max)` in world coordinates.
///
/// The corridor centerline runs from `c.start` to `c.end`. The corridor
/// occupies `width` units across the centerline and `height` units vertically
/// from floor to ceiling. The solid wall surrounds the corridor's clear
/// interior.
fn corridor_extents(c: &Corridor) -> ((i32, i32, i32), (i32, i32, i32)) {
    let hw = (c.width / 2) as i32; // half-width
    let hh = c.height as i32;

    let (x0, x1) = if c.start.0 <= c.end.0 {
        (c.start.0 - hw, c.end.0 + hw)
    } else {
        (c.end.0 - hw, c.start.0 + hw)
    };
    let (y0, y1) = if c.start.1 <= c.end.1 {
        (c.start.1 - hw, c.end.1 + hw)
    } else {
        (c.end.1 - hw, c.start.1 + hw)
    };

    let z0 = c.start.2;
    let z1 = z0 + hh;

    ((x0, y0, z0), (x1, y1, z1))
}

/// Determine if a corridor is horizontal (east-west) or vertical (north-south).
fn corridor_orientation(c: &Corridor) -> Orientation {
    let dx = (c.end.0 - c.start.0).abs();
    let dy = (c.end.1 - c.start.1).abs();
    if dx >= dy {
        Orientation::Horizontal
    } else {
        Orientation::Vertical
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Orientation {
    Horizontal,
    Vertical,
}

/// Get the wall thickness in world units.
const WALL: i32 = CONSTRUCTION_QUANTUM as i32; // 16

/// The outer wall half-thickness used to extend closure brushes beyond the
/// corridor extents to prevent gaps.
const CLOSURE_MARGIN: i32 = WALL;

// ── L-junction ────────────────────────────────────────────────────────────

/// Build closure brushes for an L-junction where two perpendicular corridors
/// meet at a corner turn.
///
/// An L-junction occurs when corridor A connects to corridor B at a shared
/// endpoint, and the two are perpendicular. The **outer corner** of the turn
/// requires a solid brush to fill the gap between the two corridor outer
/// walls. The **inner corner** is open (both corridors share walkable space).
///
/// Returns 1 closure brush for the outer corner.
pub fn build_l_junction(a: &Corridor, b: &Corridor) -> Vec<Brush> {
    let ((ax0, ay0, az0), (ax1, ay1, az1)) = corridor_extents(a);
    let ((bx0, by0, bz0), (bx1, by1, bz1)) = corridor_extents(b);

    let z0 = az0.min(bz0);
    let z1 = az1.max(bz1);

    let a_orient = corridor_orientation(a);
    let b_orient = corridor_orientation(b);

    // Determine the outer corner region based on corridor orientations
    // and which side they approach from.

    // The shared endpoint (junction center)
    let (jx, jy) = find_shared_endpoint(a, b);

    // Compute the closure brush for the outer corner.
    // The outer corner is the quadrant that is NOT covered by either corridor.
    let brush = match (a_orient, b_orient) {
        (Orientation::Horizontal, Orientation::Vertical) => {
            // Determine which quadrant needs filling based on corridor directions
            l_closure_hv(a, b, jx, jy, z0, z1, ax0, ax1, ay0, ay1, bx0, bx1, by0, by1)
        }
        (Orientation::Vertical, Orientation::Horizontal) => {
            l_closure_hv(b, a, jx, jy, z0, z1, bx0, bx1, by0, by1, ax0, ax1, ay0, ay1)
        }
        _ => {
            // Parallel corridors at L-junction — shouldn't happen; return empty
            return Vec::new();
        }
    };

    brush.into_iter().collect()
}

/// Compute the L-junction outer corner closure when `h` is horizontal (E-W)
/// and `v` is vertical (N-S).
fn l_closure_hv(
    h: &Corridor,
    v: &Corridor,
    jx: i32,
    jy: i32,
    z0: i32,
    z1: i32,
    _hx0: i32,
    _hx1: i32,
    _hy0: i32,
    _hy1: i32,
    _vx0: i32,
    _vx1: i32,
    _vy0: i32,
    _vy1: i32,
) -> Vec<Brush> {
    let hw = WALL; // use full wall thickness for closure brush size

    // Horizontal corridor: comes from one side of jx, extends to the other
    let h_from_west = h.start.0 < jx || h.end.0 < jx;
    let h_from_east = h.start.0 > jx || h.end.0 > jx;

    // Vertical corridor: comes from one side of jy
    let v_from_south = v.start.1 < jy || v.end.1 < jy;
    let v_from_north = v.start.1 > jy || v.end.1 > jy;

    // The outer corner is the quadrant opposite both "from" directions.
    // Snap closure brush to quantum grid.
    let (ox_min, ox_max, oy_min, oy_max) = match (h_from_west, h_from_east, v_from_south, v_from_north) {
        (true, false, true, false) => {
            // H from west, V from south → outer NE
            (jx - hw, jx + hw, jy - hw, jy + hw)
        }
        (true, false, false, true) => {
            // H from west, V from north → outer SE
            (jx - hw, jx + hw, jy - hw, jy + hw)
        }
        (false, true, true, false) => {
            // H from east, V from south → outer NW
            (jx - hw, jx + hw, jy - hw, jy + hw)
        }
        (false, true, false, true) => {
            // H from east, V from north → outer SW
            (jx - hw, jx + hw, jy - hw, jy + hw)
        }
        _ => {
            // Can't determine: build a minimal closure at junction
            (jx - CLOSURE_MARGIN, jx + CLOSURE_MARGIN, jy - CLOSURE_MARGIN, jy + CLOSURE_MARGIN)
        }
    };

    // Snap to quantum grid
    let q = CONSTRUCTION_QUANTUM as i32;
    let ox_min = snap_to_quantum_grid(ox_min, q);
    let ox_max = snap_to_quantum_grid(ox_max, q);
    let oy_min = snap_to_quantum_grid(oy_min, q);
    let oy_max = snap_to_quantum_grid(oy_max, q);
    // Ensure minimum size of at least one quantum
    let ox_min = ox_min.min(ox_max - q);
    let oy_min = oy_min.min(oy_max - q);

    // Build the outer corner closure brush
    let brush = make_brush(
        (ox_min, oy_min, z0),
        (ox_max, oy_max, z1),
        DEFAULT_WALL_TEXTURE,
    );

    vec![brush]
}

// ── T-junction ────────────────────────────────────────────────────────────

/// Build closure brushes for a T-junction where one corridor terminates
/// into another through corridor.
///
/// The terminating corridor's dead-end wall needs a closure brush at the
/// point where it meets the through corridor (covering the wall gap).
/// The through corridor's side walls also need closure at the junction
/// edges.
///
/// Returns 1–2 closure brushes.
pub fn build_t_junction(terminating: &Corridor, through: &Corridor) -> Vec<Brush> {
    let (_tx0, _ty0, _tz0) = corridor_extents(terminating).0;
    let (_, (_thx1, _thy1, thz1)) = corridor_extents(through);

    let z0 = terminating.start.2;
    let z1 = terminating.start.2 + terminating.height as i32;
    let z1 = z1.max(thz1);

    let q = CONSTRUCTION_QUANTUM as i32;

    let t_orient = corridor_orientation(terminating);
    let th_orient = corridor_orientation(through);

    let mut brushes = Vec::new();

    match (t_orient, th_orient) {
        (Orientation::Horizontal, Orientation::Vertical) => {
            // Terminating E-W corridor meets through N-S corridor
            // The terminating corridor's east/west end meets the through corridor
            let (jx, jy) = find_shared_endpoint(terminating, through);

            // Closure brushes: fill wall gaps at the junction
            let hw = (terminating.width / 2) as i32;
            let th_hw = (through.width / 2) as i32;

            // Two closure brushes at the T-junction: one on each side of the
            // terminating corridor where it meets the through corridor wall.
            let t_comes_from_west = terminating.start.0 < jx || terminating.end.0 < jx;

            let (cx_min, cx_max) = if t_comes_from_west {
                // Terminating from west: closure at east end
                (jx - WALL, jx)
            } else {
                // Terminating from east: closure at west end
                (jx, jx + WALL)
            };

            // Closure on north side of terminating corridor
            brushes.push(make_brush(
                (snap_to_quantum_grid(cx_min, q), snap_to_quantum_grid(jy + hw - WALL, q), z0),
                (snap_to_quantum_grid(cx_max, q), snap_to_quantum_grid(jy + th_hw, q), z1),
                DEFAULT_WALL_TEXTURE,
            ));
            // Closure on south side of terminating corridor
            brushes.push(make_brush(
                (snap_to_quantum_grid(cx_min, q), snap_to_quantum_grid(jy - th_hw, q), z0),
                (snap_to_quantum_grid(cx_max, q), snap_to_quantum_grid(jy - hw + WALL, q), z1),
                DEFAULT_WALL_TEXTURE,
            ));
        }
        (Orientation::Vertical, Orientation::Horizontal) => {
            // Terminating N-S corridor meets through E-W corridor
            let (jx, jy) = find_shared_endpoint(terminating, through);

            let hw = (terminating.width / 2) as i32;
            let th_hw = (through.width / 2) as i32;

            let t_comes_from_south = terminating.start.1 < jy || terminating.end.1 < jy;

            let (cy_min, cy_max) = if t_comes_from_south {
                (jy - WALL, jy)
            } else {
                (jy, jy + WALL)
            };

            // Closure on east side of terminating corridor
            brushes.push(make_brush(
                (snap_to_quantum_grid(jx + hw - WALL, q), snap_to_quantum_grid(cy_min, q), z0),
                (snap_to_quantum_grid(jx + th_hw, q), snap_to_quantum_grid(cy_max, q), z1),
                DEFAULT_WALL_TEXTURE,
            ));
            // Closure on west side of terminating corridor
            brushes.push(make_brush(
                (snap_to_quantum_grid(jx - th_hw, q), snap_to_quantum_grid(cy_min, q), z0),
                (snap_to_quantum_grid(jx - hw + WALL, q), snap_to_quantum_grid(cy_max, q), z1),
                DEFAULT_WALL_TEXTURE,
            ));
        }
        _ => {
            // Parallel corridors: treat as simple wall closure
            let (jx, jy) = find_shared_endpoint(terminating, through);
            brushes.push(make_brush(
                (jx - WALL, jy - WALL, z0),
                (jx + WALL, jy + WALL, z1),
                DEFAULT_WALL_TEXTURE,
            ));
        }
    }

    brushes
}

// ── X-junction ────────────────────────────────────────────────────────────

/// Build closure brushes for an X-junction where two corridors cross at
/// right angles.
///
/// All 4 outer corners of the crossing need closure brushes to seal gaps
/// between the corridor outer walls. The center region (where both corridors
/// overlap) is open walkable space.
///
/// Returns 4 closure brushes (one per outer corner).
pub fn build_x_junction(a: &Corridor, b: &Corridor) -> Vec<Brush> {
    let ((_ax0, _ay0, az0), (_ax1, _ay1, az1)) = corridor_extents(a);
    let ((_bx0, _by0, bz0), (_bx1, _by1, bz1)) = corridor_extents(b);

    let z0 = az0.min(bz0);
    let z1 = az1.max(bz1);

    let a_orient = corridor_orientation(a);
    let b_orient = corridor_orientation(b);

    if a_orient == b_orient {
        // Parallel corridors crossing — not a true X-junction
        return Vec::new();
    }

    // Determine which corridor is horizontal and which is vertical
    let (hc, vc) = if a_orient == Orientation::Horizontal {
        (a, b)
    } else {
        (b, a)
    };

    let ((hx0, hy0, _), (hx1, hy1, _)) = corridor_extents(hc);
    let ((vx0, vy0, _), (vx1, vy1, _)) = corridor_extents(vc);

    // Intersection region
    let ix0 = hx0.max(vx0);
    let ix1 = hx1.min(vx1);
    let iy0 = hy0.max(vy0);
    let iy1 = hy1.min(vy1);

    // Four outer corner closures:
    // NE: x > ix1, y > iy1
    // NW: x < ix0, y > iy1
    // SE: x > ix1, y < iy0
    // SW: x < ix0, y < iy0

    let m = CLOSURE_MARGIN;
    let q = CONSTRUCTION_QUANTUM as i32;

    vec![
        // NE corner
        make_brush(
            (snap_to_quantum_grid(ix1 - m, q), snap_to_quantum_grid(iy1 - m, q), z0),
            (snap_to_quantum_grid(ix1 + m, q), snap_to_quantum_grid(iy1 + m, q), z1),
            DEFAULT_WALL_TEXTURE,
        ),
        // NW corner
        make_brush(
            (snap_to_quantum_grid(ix0 - m, q), snap_to_quantum_grid(iy1 - m, q), z0),
            (snap_to_quantum_grid(ix0 + m, q), snap_to_quantum_grid(iy1 + m, q), z1),
            DEFAULT_WALL_TEXTURE,
        ),
        // SE corner
        make_brush(
            (snap_to_quantum_grid(ix1 - m, q), snap_to_quantum_grid(iy0 - m, q), z0),
            (snap_to_quantum_grid(ix1 + m, q), snap_to_quantum_grid(iy0 + m, q), z1),
            DEFAULT_WALL_TEXTURE,
        ),
        // SW corner
        make_brush(
            (snap_to_quantum_grid(ix0 - m, q), snap_to_quantum_grid(iy0 - m, q), z0),
            (snap_to_quantum_grid(ix0 + m, q), snap_to_quantum_grid(iy0 + m, q), z1),
            DEFAULT_WALL_TEXTURE,
        ),
    ]
}

// ── Room portal ───────────────────────────────────────────────────────────

/// Build the portal opening brush for a corridor entering a room.
///
/// In Quake .map terms, this returns the **opening marker** — the rectangular
/// region on the room wall where the corridor passes through. The wall brush
/// for the room should omit this region (or be split around it), creating an
/// open arch.
///
/// Returns a single brush representing the portal opening (used as a
/// subtraction hint for room wall generation).
pub fn build_room_portal(corridor: &Corridor, room: &RoomIntent) -> Vec<Brush> {
    let hw = (corridor.width / 2) as i32;
    let hh = corridor.height as i32;
    let z0 = room.position.2;
    let z1 = z0 + hh;

    let r_min_x = room.position.0;
    let r_max_x = room.position.0 + room.dimensions.0 as i32;
    let r_min_y = room.position.1;
    let r_max_y = room.position.1 + room.dimensions.1 as i32;

    // Find which wall the corridor connects to
    let (cx, cy) = find_corridor_room_contact(corridor, room);

    // Determine portal position on the wall
    let portal = if cx <= r_min_x + WALL {
        // West wall
        Some(make_brush(
            (r_min_x - WALL, cy - hw, z0),
            (r_min_x + WALL, cy + hw, z1),
            DEFAULT_WALL_TEXTURE,
        ))
    } else if cx >= r_max_x - WALL {
        // East wall
        Some(make_brush(
            (r_max_x - WALL, cy - hw, z0),
            (r_max_x + WALL, cy + hw, z1),
            DEFAULT_WALL_TEXTURE,
        ))
    } else if cy <= r_min_y + WALL {
        // South wall
        Some(make_brush(
            (cx - hw, r_min_y - WALL, z0),
            (cx + hw, r_min_y + WALL, z1),
            DEFAULT_WALL_TEXTURE,
        ))
    } else if cy >= r_max_y - WALL {
        // North wall
        Some(make_brush(
            (cx - hw, r_max_y - WALL, z0),
            (cx + hw, r_max_y + WALL, z1),
            DEFAULT_WALL_TEXTURE,
        ))
    } else {
        None
    };

    portal.into_iter().collect()
}

/// Find where a corridor contacts a room wall.
/// Returns the contact point (center of portal) on the room wall.
fn find_corridor_room_contact(corridor: &Corridor, room: &RoomIntent) -> (i32, i32) {
    let r_min_x = room.position.0;
    let r_max_x = room.position.0 + room.dimensions.0 as i32;
    let r_min_y = room.position.1;
    let r_max_y = room.position.1 + room.dimensions.1 as i32;

    // The corridor start is at the room portal
    let sx = corridor.start.0;
    let sy = corridor.start.1;

    // Clamp to room wall
    let cx = clamp_to_range(sx, r_min_x, r_max_x);
    let cy = clamp_to_range(sy, r_min_y, r_max_y);

    // Determine which wall the corridor contacts
    let dist_left = (sx - r_min_x).abs();
    let dist_right = (sx - r_max_x).abs();
    let dist_bottom = (sy - r_min_y).abs();
    let dist_top = (sy - r_max_y).abs();
    let min_dist = dist_left.min(dist_right).min(dist_bottom).min(dist_top);

    if min_dist == dist_left {
        (r_min_x, cy)
    } else if min_dist == dist_right {
        (r_max_x, cy)
    } else if min_dist == dist_top {
        (cx, r_max_y)
    } else {
        (cx, r_min_y)
    }
}

// ── Utility ────────────────────────────────────────────────────────────────

/// Find the shared endpoint between two corridors (the junction center).
fn find_shared_endpoint(a: &Corridor, b: &Corridor) -> (i32, i32) {
    // Check if a.start matches b.start or b.end, etc.
    let eps = CONSTRUCTION_QUANTUM as i32;
    let points = [
        (a.start.0, a.start.1),
        (a.end.0, a.end.1),
        (b.start.0, b.start.1),
        (b.end.0, b.end.1),
    ];

    // Find the midpoint of the two closest endpoints
    for &(ax, ay) in &[(a.start.0, a.start.1), (a.end.0, a.end.1)] {
        for &(bx, by) in &[(b.start.0, b.start.1), (b.end.0, b.end.1)] {
            if (ax - bx).abs() <= eps && (ay - by).abs() <= eps {
                return ((ax + bx) / 2, (ay + by) / 2);
            }
        }
    }

    // Fallback: use midpoint of all endpoints
    let sum_x: i32 = points.iter().map(|p| p.0).sum();
    let sum_y: i32 = points.iter().map(|p| p.1).sum();
    (sum_x / 4, sum_y / 4)
}

fn clamp_to_range(v: i32, lo: i32, hi: i32) -> i32 {
    if v < lo {
        lo
    } else if v > hi {
        hi
    } else {
        v
    }
}

/// Snap a value to the nearest multiple of `quantum` (rounds toward zero for
/// half-way values to ensure conservative snapping).
fn snap_to_quantum_grid(v: i32, quantum: i32) -> i32 {
    let rem = v.rem_euclid(quantum);
    let half = quantum / 2;
    if rem == 0 {
        v
    } else if rem <= half {
        v - rem
    } else {
        v + (quantum - rem)
    }
}

// ── Junction classification ───────────────────────────────────────────────

/// Classify the type of junction formed by a set of corridors meeting at a
/// common point.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JunctionKind {
    /// Two corridors meet at a corner turn (L-shape).
    L,
    /// One corridor terminates into another (T-shape).
    T,
    /// Two corridors cross (X-shape).
    X,
    /// A corridor endpoint at a room wall (portal).
    Portal,
    /// Straight pass-through: corridors align end-to-end.
    Straight,
}

/// Build all closure brushes for explicit endpoint junctions in a routed
/// corridor set.
///
/// This function classifies corridors that share endpoints and delegates to
/// the appropriate builder. Incidental mid-span crossings are left to normal
/// CSG overlap instead of receiving pairwise X-closure brushes; otherwise dense
/// generated maps emit O(n²) source brushes and can exceed the M1 face budget.
pub fn build_junction_closures(corridors: &[Corridor]) -> Vec<Brush> {
    if corridors.len() < 2 {
        return Vec::new();
    }

    // Classify pairs of corridors
    let mut brushes = Vec::new();

    for i in 0..corridors.len() {
        for j in (i + 1)..corridors.len() {
            let a = &corridors[i];
            let b = &corridors[j];

            if !corridors_meet(a, b) {
                continue;
            }

            let a_orient = corridor_orientation(a);
            let b_orient = corridor_orientation(b);

            if a_orient == b_orient {
                // Parallel: straight pass-through, no closure needed
                continue;
            }

            // Determine if share endpoint (L/T) or crossing (X)
            if share_endpoint(a, b) {
                // Could be L or T — build both and let L-junction handle it
                // For T-junction, one corridor terminates at the other's midpoint
                let a_terminates = terminates_at(a, b);
                let b_terminates = terminates_at(b, a);

                if a_terminates && !b_terminates {
                    brushes.extend(build_t_junction(a, b));
                } else if b_terminates && !a_terminates {
                    brushes.extend(build_t_junction(b, a));
                } else {
                    brushes.extend(build_l_junction(a, b));
                }
            } else {
                // Incidental crossing, not an explicit routed endpoint
                // junction. Do not emit pairwise X closures in generated maps;
                // explicit X fixtures can still call `build_x_junction`.
            }
        }
    }

    brushes
}

/// Check if two corridors share an endpoint (within 1 quantum tolerance).
fn share_endpoint(a: &Corridor, b: &Corridor) -> bool {
    let eps = CONSTRUCTION_QUANTUM as i32;
    let a_pts = [(a.start.0, a.start.1), (a.end.0, a.end.1)];
    let b_pts = [(b.start.0, b.start.1), (b.end.0, b.end.1)];

    for &(ax, ay) in &a_pts {
        for &(bx, by) in &b_pts {
            if (ax - bx).abs() <= eps && (ay - by).abs() <= eps {
                return true;
            }
        }
    }
    false
}

/// Check if corridor `a` terminates into corridor `b` (instead of meeting at
/// an endpoint). Termination means one endpoint of `a` lies along the length
/// of `b`, not at `b`'s endpoints.
fn terminates_at(a: &Corridor, b: &Corridor) -> bool {
    let eps = CONSTRUCTION_QUANTUM as i32;
    let a_pts = [(a.start.0, a.start.1), (a.end.0, a.end.1)];
    let b_min_x = b.start.0.min(b.end.0);
    let b_max_x = b.start.0.max(b.end.0);
    let b_min_y = b.start.1.min(b.end.1);
    let b_max_y = b.start.1.max(b.end.1);

    for &(ax, ay) in &a_pts {
        // Check if this point lies along b's interior
        let on_b = if b_min_x == b_max_x {
            // b is vertical
            (ax - b_min_x).abs() <= eps && ay > b_min_y + eps && ay < b_max_y - eps
        } else if b_min_y == b_max_y {
            // b is horizontal
            (ay - b_min_y).abs() <= eps && ax > b_min_x + eps && ax < b_max_x - eps
        } else {
            false
        };

        if on_b {
            return true;
        }
    }
    false
}

/// Check if two corridors meet (share an endpoint or one terminates into the
/// other).
fn corridors_meet(a: &Corridor, b: &Corridor) -> bool {
    share_endpoint(a, b) || terminates_at(a, b) || terminates_at(b, a) || corridors_cross(a, b)
}

/// Check if two perpendicular corridors cross each other.
fn corridors_cross(a: &Corridor, b: &Corridor) -> bool {
    let a_orient = corridor_orientation(a);
    let b_orient = corridor_orientation(b);
    if a_orient == b_orient {
        return false;
    }

    let ((ax0, ay0, _), (ax1, ay1, _)) = corridor_extents(a);
    let ((bx0, by0, _), (bx1, by1, _)) = corridor_extents(b);

    // Check if the extents overlap in both axes
    let overlap_x = ax0 < bx1 && bx0 < ax1;
    let overlap_y = ay0 < by1 && by0 < ay1;

    overlap_x && overlap_y
}

#[cfg(test)]
mod tests {
    use super::*;

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

    // ── make_brush ─────────────────────────────────────────────────────

    #[test]
    fn make_brush_has_six_faces() {
        let brush = make_brush((0, 0, 0), (64, 64, 128), "test_tex");
        assert_eq!(brush.faces.len(), 6);
        for face in &brush.faces {
            assert_eq!(face.texture, "test_tex");
        }
    }

    #[test]
    fn make_brush_planes_are_valid() {
        let brush = make_brush((0, 0, 0), (64, 64, 128), "t");
        // Each face should have 3 non-collinear points
        for face in &brush.faces {
            let (p0, p1, p2) = (face.plane_points[0], face.plane_points[1], face.plane_points[2]);
            // Points should not all be identical
            assert!(
                p0 != p1 || p1 != p2,
                "face has collinear/identical points: {:?}",
                face.plane_points
            );
        }
    }

    // ── L-junction ──────────────────────────────────────────────────────

    #[test]
    fn l_junction_produces_closure_brush() {
        // Horizontal corridor from (0,0) to (128,0), vertical from (128,0) to (128,128)
        let h = corridor_h(0, 0, 0, 128);
        let v = corridor_v(128, 0, 0, 128);
        let brushes = build_l_junction(&h, &v);
        assert!(!brushes.is_empty(), "L-junction should produce closure brushes");
        for b in &brushes {
            assert_eq!(b.faces.len(), 6);
        }
    }

    // ── T-junction ──────────────────────────────────────────────────────

    #[test]
    fn t_junction_produces_closure_brushes() {
        // Through: vertical corridor from (64,0) to (64,192)
        let through = corridor_v(64, 0, 0, 192);
        // Terminating: horizontal from (0,64) to (64,64)
        let term = corridor_h(0, 64, 0, 64);
        let brushes = build_t_junction(&term, &through);
        assert!(!brushes.is_empty(), "T-junction should produce closure brushes");
        for b in &brushes {
            assert_eq!(b.faces.len(), 6);
        }
    }

    // ── X-junction ──────────────────────────────────────────────────────

    #[test]
    fn x_junction_produces_four_corner_brushes() {
        let h = corridor_h(0, 64, 0, 192);
        let v = corridor_v(64, 0, 0, 192);
        let brushes = build_x_junction(&h, &v);
        // An X-junction with crossing corridors should produce 4 corner closures
        assert!(brushes.len() == 4, "expected 4 corner brushes, got {}", brushes.len());
        for b in &brushes {
            assert_eq!(b.faces.len(), 6);
        }
    }

    #[test]
    fn x_junction_parallel_corridors_no_closure() {
        let h1 = corridor_h(0, 0, 0, 128);
        let h2 = corridor_h(0, 64, 0, 128); // parallel, offset
        let brushes = build_x_junction(&h1, &h2);
        assert!(brushes.is_empty());
    }

    // ── Room portal ────────────────────────────────────────────────────

    #[test]
    fn room_portal_produces_opening_brush() {
        // Room at (0,0) 64x64, corridor enters from east
        let room = room(0, 0, 0, 64, 64, 128);
        let corr = Corridor {
            start: (64, 32, 0), // on east wall
            end: (128, 32, 0),
            width: 64,
            height: 80,
        };
        let brushes = build_room_portal(&corr, &room);
        assert!(!brushes.is_empty(), "room portal should produce portal brush");
        for b in &brushes {
            assert_eq!(b.faces.len(), 6);
        }
    }

    // ── Utility ─────────────────────────────────────────────────────────

    #[test]
    fn find_shared_endpoint_at_common_point() {
        let a = corridor_h(0, 0, 0, 64);
        let b = corridor_v(64, 0, 0, 64);
        let (x, y) = find_shared_endpoint(&a, &b);
        assert!((x - 64).abs() <= 16);
        assert!((y - 0).abs() <= 16);
    }

    #[test]
    fn share_endpoint_detects_common_endpoint() {
        let a = corridor_h(0, 0, 0, 64);
        let b = corridor_v(64, 0, 0, 64);
        assert!(share_endpoint(&a, &b));
    }

    #[test]
    fn share_endpoint_rejects_separate_corridors() {
        let a = corridor_h(0, 0, 0, 64);
        let b = corridor_v(128, 128, 0, 64);
        assert!(!share_endpoint(&a, &b));
    }

    // ── build_junction_closures ────────────────────────────────────────

    #[test]
    fn build_junction_closures_for_l_shape() {
        let corridors = vec![
            corridor_h(0, 0, 0, 64),
            corridor_v(64, 0, 0, 64),
        ];
        let brushes = build_junction_closures(&corridors);
        assert!(!brushes.is_empty());
    }

    #[test]
    fn build_junction_closures_empty_for_single_corridor() {
        let corridors = vec![corridor_h(0, 0, 0, 64)];
        let brushes = build_junction_closures(&corridors);
        assert!(brushes.is_empty());
    }

    #[test]
    fn all_junction_brushes_have_six_faces() {
        let corridors = vec![
            corridor_h(0, 0, 0, 64),
            corridor_v(64, 0, 0, 64),
            corridor_v(64, 64, 0, 64),
            corridor_h(64, 128, 0, 64),
        ];
        let brushes = build_junction_closures(&corridors);
        for b in &brushes {
            assert_eq!(b.faces.len(), 6, "every closure brush must have exactly 6 faces");
        }
    }
}
