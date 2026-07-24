//! Geometry utilities — pure functions for quantum snapping, overlap
//! detection, bounds enforcement, and topology validation.
//!
//! All functions are deterministic and operate on [`RoomIntent`] and
//! [`LayoutIntent`] records. None mutate state or perform I/O.

use crate::config::CONSTRUCTION_QUANTUM;
use crate::error::GeneratorError;
use crate::intent::RoomIntent;

/// Snap a signed integer value to the nearest multiple of `quantum`.
///
/// Rounding is half-up: values exactly halfway between two multiples round
/// toward positive infinity. This matches the canonical placement contract:
/// candidate positions are derived from snapped random coordinates.
///
/// # Examples
///
/// ```
/// # use bsp_generator::geometry::snap_to_quantum;
/// assert_eq!(snap_to_quantum(47, 16), 48);
/// assert_eq!(snap_to_quantum(40, 16), 48); // exactly halfway → up
/// assert_eq!(snap_to_quantum(39, 16), 32);
/// assert_eq!(snap_to_quantum(-47, 16), -48);
/// ```
pub fn snap_to_quantum(value: i32, quantum: i32) -> i32 {
    debug_assert!(quantum > 0, "quantum must be positive");
    let q = quantum;
    // Half-up rounding: values exactly halfway between two multiples
    // round toward positive infinity.
    //
    // Use Euclidean division: remainder is always non-negative.
    let rem = value.rem_euclid(q);
    let half = q / 2;
    if rem == 0 {
        value
    } else if rem < half {
        // Closer to the lower multiple
        value - rem
    } else if rem > half {
        // Closer to the higher multiple
        value + (q - rem)
    } else {
        // Exactly halfway: round up (toward +∞)
        value + half
    }
}

/// Return `true` if two rooms overlap each other when `wall_thickness` units
/// of separation are required between their bounding boxes.
///
/// Two rooms overlap if their axis-aligned bounding boxes, each expanded by
/// `wall_thickness / 2` on all sides, intersect in any dimension (3D
/// overlap). Since all rooms in beta share a common floor and ceiling Z,
/// the Z check is degenerate but included for future stacking.
pub fn rooms_overlap(a: &RoomIntent, b: &RoomIntent, wall_thickness: i32) -> bool {
    let hw = wall_thickness / 2; // half-wall margin on each side

    let a_x0 = a.position.0 - hw;
    let a_x1 = a.position.0 + a.dimensions.0 as i32 + hw;
    let b_x0 = b.position.0 - hw;
    let b_x1 = b.position.0 + b.dimensions.0 as i32 + hw;

    if a_x1 <= b_x0 || b_x1 <= a_x0 {
        return false;
    }

    let a_y0 = a.position.1 - hw;
    let a_y1 = a.position.1 + a.dimensions.1 as i32 + hw;
    let b_y0 = b.position.1 - hw;
    let b_y1 = b.position.1 + b.dimensions.1 as i32 + hw;

    if a_y1 <= b_y0 || b_y1 <= a_y0 {
        return false;
    }

    let a_z0 = a.position.2 - hw;
    let a_z1 = a.position.2 + a.dimensions.2 as i32 + hw;
    let b_z0 = b.position.2 - hw;
    let b_z1 = b.position.2 + b.dimensions.2 as i32 + hw;

    if a_z1 <= b_z0 || b_z1 <= a_z0 {
        return false;
    }

    true
}

/// Validate that every room is fully contained within `(0, 0, 0)` to
/// `(xy_bounds.0, xy_bounds.1, z_span)`.
///
/// Returns `Ok(())` if all rooms are in bounds, or
/// `Err(GeneratorError::InvariantViolation)` with a diagnostic describing
/// the first out-of-bounds room.
pub fn validate_bounds(
    rooms: &[RoomIntent],
    xy_bounds: (u32, u32),
    z_span: u32,
) -> Result<(), GeneratorError> {
    let max_x = xy_bounds.0 as i32;
    let max_y = xy_bounds.1 as i32;
    let max_z = z_span as i32;

    for (i, room) in rooms.iter().enumerate() {
        let (x, y, z) = room.position;
        let (dx, dy, dz) = room.dimensions;

        if x < 0 || y < 0 || z < 0 {
            return Err(GeneratorError::InvariantViolation(format!(
                "room {} position ({}, {}, {}) has negative component; bounds ((0..{}, 0..{}, 0..{}))",
                i, x, y, z, max_x, max_y, max_z,
            )));
        }
        if x + dx as i32 > max_x {
            return Err(GeneratorError::InvariantViolation(format!(
                "room {} X extent ({}+{}={}) exceeds max_x {}",
                i,
                x,
                dx,
                x + dx as i32,
                max_x,
            )));
        }
        if y + dy as i32 > max_y {
            return Err(GeneratorError::InvariantViolation(format!(
                "room {} Y extent ({}+{}={}) exceeds max_y {}",
                i,
                y,
                dy,
                y + dy as i32,
                max_y,
            )));
        }
        if z + dz as i32 > max_z {
            return Err(GeneratorError::InvariantViolation(format!(
                "room {} Z extent ({}+{}={}) exceeds max_z {}",
                i,
                z,
                dz,
                z + dz as i32,
                max_z,
            )));
        }
    }
    Ok(())
}

/// Validate that the graph formed by `edges` on `room_count` vertices has
/// exactly `mst_edges + loop_count` edges, where `mst_edges = room_count - 1`
/// is the minimum spanning tree edge count.
///
/// Returns `Err(GeneratorError::InvariantViolation)` if the edge count does
/// not match.
pub fn validate_cycle_count(
    edges: &[(usize, usize)],
    room_count: usize,
    loop_count: u32,
) -> Result<(), GeneratorError> {
    let expected_mst = if room_count == 0 { 0 } else { room_count - 1 };
    let expected_total = expected_mst + loop_count as usize;
    let actual = edges.len();

    if actual != expected_total {
        return Err(GeneratorError::InvariantViolation(format!(
            "edge count {} != expected {} (mst {} + loops {}) for {} rooms",
            actual, expected_total, expected_mst, loop_count, room_count,
        )));
    }
    Ok(())
}

/// Union-Find for connectedness checks.
struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        UnionFind {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]);
        }
        self.parent[x]
    }

    fn union(&mut self, a: usize, b: usize) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return;
        }
        match self.rank[ra].cmp(&self.rank[rb]) {
            std::cmp::Ordering::Less => self.parent[ra] = rb,
            std::cmp::Ordering::Greater => self.parent[rb] = ra,
            std::cmp::Ordering::Equal => {
                self.parent[rb] = ra;
                self.rank[ra] += 1;
            }
        }
    }
}

/// Return `true` if the given edges connect all `room_count` vertices into a
/// single connected component. An empty room set is trivially connected.
///
/// This is a structural check, not a geometric one — it operates purely on
/// the abstract graph.
pub fn validate_connectedness(edges: &[(usize, usize)], room_count: usize) -> bool {
    if room_count == 0 {
        return true;
    }
    let mut uf = UnionFind::new(room_count);
    for &(a, b) in edges {
        uf.union(a, b);
    }
    let root = uf.find(0);
    for v in 1..room_count {
        if uf.find(v) != root {
            return false;
        }
    }
    true
}

/// Check that every room's position and dimensions are multiples of the
/// construction quantum.
pub fn validate_quantum_alignment(rooms: &[RoomIntent]) -> Result<(), GeneratorError> {
    let q = CONSTRUCTION_QUANTUM as i32;
    for (i, room) in rooms.iter().enumerate() {
        let (x, y, z) = room.position;
        let (dx, dy, dz) = room.dimensions;

        if x % q != 0 || y % q != 0 || z % q != 0 {
            return Err(GeneratorError::InvariantViolation(format!(
                "room {} position ({}, {}, {}) not aligned to quantum {}",
                i, x, y, z, q,
            )));
        }
        if dx % CONSTRUCTION_QUANTUM != 0
            || dy % CONSTRUCTION_QUANTUM != 0
            || dz % CONSTRUCTION_QUANTUM != 0
        {
            return Err(GeneratorError::InvariantViolation(format!(
                "room {} dimensions ({}, {}, {}) not multiples of quantum {}",
                i, dx, dy, dz, CONSTRUCTION_QUANTUM,
            )));
        }
    }
    Ok(())
}

/// Check that no two rooms overlap (including wall-thickness separation).
pub fn validate_no_overlap(
    rooms: &[RoomIntent],
    wall_thickness: i32,
) -> Result<(), GeneratorError> {
    for i in 0..rooms.len() {
        for j in (i + 1)..rooms.len() {
            if rooms_overlap(&rooms[i], &rooms[j], wall_thickness) {
                return Err(GeneratorError::InvariantViolation(format!(
                    "rooms {} and {} overlap with wall_thickness {}",
                    i, j, wall_thickness,
                )));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── snap_to_quantum ────────────────────────────────────────────────

    #[test]
    fn snap_positive_rounds_correctly() {
        assert_eq!(snap_to_quantum(0, 16), 0);
        assert_eq!(snap_to_quantum(7, 16), 0);
        assert_eq!(snap_to_quantum(8, 16), 16); // half-up
        assert_eq!(snap_to_quantum(15, 16), 16);
        assert_eq!(snap_to_quantum(16, 16), 16);
        assert_eq!(snap_to_quantum(24, 16), 32); // half-up
        assert_eq!(snap_to_quantum(23, 16), 16);
    }

    #[test]
    fn snap_negative_rounds_correctly() {
        assert_eq!(snap_to_quantum(-7, 16), 0);
        assert_eq!(snap_to_quantum(-8, 16), 0); // half-up → 0
        assert_eq!(snap_to_quantum(-9, 16), -16);
        assert_eq!(snap_to_quantum(-16, 16), -16);
        assert_eq!(snap_to_quantum(-24, 16), -16); // half-up
        assert_eq!(snap_to_quantum(-25, 16), -32);
        assert_eq!(snap_to_quantum(-40, 16), -32); // exactly halfway → up
        assert_eq!(snap_to_quantum(-41, 16), -48);
        assert_eq!(snap_to_quantum(-47, 16), -48);
    }

    #[test]
    fn snap_large_values() {
        assert_eq!(snap_to_quantum(1536, 16), 1536);
        assert_eq!(snap_to_quantum(1535, 16), 1536);
        assert_eq!(snap_to_quantum(3072, 16), 3072);
    }

    // ── rooms_overlap ──────────────────────────────────────────────────

    fn room_at(x: i32, y: i32, z: i32, dx: u32, dy: u32, dz: u32) -> RoomIntent {
        RoomIntent {
            position: (x, y, z),
            dimensions: (dx, dy, dz),
        }
    }

    #[test]
    fn identical_rooms_overlap() {
        let a = room_at(0, 0, 0, 64, 64, 128);
        assert!(rooms_overlap(&a, &a, 16));
    }

    #[test]
    fn adjacent_rooms_with_wall_gap_do_not_overlap() {
        // Room A: (0,0)-(64,64), Room B at x=80 (64+16 wall)
        let a = room_at(0, 0, 0, 64, 64, 128);
        let b = room_at(80, 0, 0, 64, 64, 128);
        assert!(!rooms_overlap(&a, &b, 16));
    }

    #[test]
    fn rooms_too_close_overlap() {
        let a = room_at(0, 0, 0, 64, 64, 128);
        let b = room_at(64, 0, 0, 64, 64, 128); // just touching without wall
        assert!(rooms_overlap(&a, &b, 16));
    }

    #[test]
    fn rooms_with_wall_thickness_separated() {
        // 64 + 16 wall = 80 apart minimum
        let a = room_at(0, 0, 0, 64, 64, 128);
        let b = room_at(80, 0, 0, 64, 64, 128); // exact 16 wall gap
        assert!(!rooms_overlap(&a, &b, 16));
    }

    #[test]
    fn diagonal_rooms_no_overlap() {
        let a = room_at(0, 0, 0, 64, 64, 128);
        let b = room_at(80, 80, 0, 64, 64, 128);
        assert!(!rooms_overlap(&a, &b, 16));
    }

    // ── validate_bounds ────────────────────────────────────────────────

    #[test]
    fn rooms_within_bounds_pass() {
        let rooms = vec![
            room_at(0, 0, 0, 64, 64, 128),
            room_at(80, 80, 0, 64, 64, 128),
        ];
        assert!(validate_bounds(&rooms, (1024, 1024), 256).is_ok());
    }

    #[test]
    fn room_outside_x_bound_fails() {
        let rooms = vec![room_at(1000, 0, 0, 64, 64, 128)];
        let err = validate_bounds(&rooms, (1024, 1024), 256).unwrap_err();
        assert!(err.to_string().contains("exceeds max_x"), "{}", err);
    }

    #[test]
    fn room_outside_y_bound_fails() {
        let rooms = vec![room_at(0, 1000, 0, 64, 64, 128)];
        let err = validate_bounds(&rooms, (1024, 1024), 256).unwrap_err();
        assert!(err.to_string().contains("exceeds max_y"), "{}", err);
    }

    #[test]
    fn room_outside_z_bound_fails() {
        let rooms = vec![room_at(0, 0, 200, 64, 64, 128)];
        let err = validate_bounds(&rooms, (1024, 1024), 256).unwrap_err();
        assert!(err.to_string().contains("exceeds max_z"), "{}", err);
    }

    #[test]
    fn room_negative_position_fails() {
        let rooms = vec![room_at(-16, 0, 0, 64, 64, 128)];
        let err = validate_bounds(&rooms, (1024, 1024), 256).unwrap_err();
        assert!(err.to_string().contains("negative"), "{}", err);
    }

    // ── validate_cycle_count ───────────────────────────────────────────

    #[test]
    fn exact_mst_plus_loops_passes() {
        // 3 rooms: MST = 2 edges, +1 loop = 3 edges
        let edges = vec![(0, 1), (1, 2), (0, 2)];
        assert!(validate_cycle_count(&edges, 3, 1).is_ok());
    }

    #[test]
    fn wrong_loop_count_fails() {
        let edges = vec![(0, 1), (1, 2)]; // only MST, 0 loops
        let err = validate_cycle_count(&edges, 3, 1).unwrap_err();
        assert!(err.to_string().contains("edge count"), "{}", err);
    }

    #[test]
    fn zero_rooms_zero_edges() {
        assert!(validate_cycle_count(&[], 0, 0).is_ok());
    }

    // ── validate_connectedness ─────────────────────────────────────────

    #[test]
    fn fully_connected_graph() {
        let edges = vec![(0, 1), (1, 2), (2, 3)];
        assert!(validate_connectedness(&edges, 4));
    }

    #[test]
    fn disconnected_graph_fails() {
        let edges = vec![(0, 1), (2, 3)]; // two components
        assert!(!validate_connectedness(&edges, 4));
    }

    #[test]
    fn single_room_is_connected() {
        assert!(validate_connectedness(&[], 1));
    }

    #[test]
    fn empty_rooms_is_connected() {
        assert!(validate_connectedness(&[], 0));
    }

    #[test]
    fn line_graph_is_connected() {
        let edges: Vec<(usize, usize)> = (0..9).map(|i| (i, i + 1)).collect();
        assert!(validate_connectedness(&edges, 10));
    }

    // ── validate_quantum_alignment ─────────────────────────────────────

    #[test]
    fn aligned_rooms_pass() {
        let rooms = vec![room_at(0, 0, 0, 64, 64, 128)];
        assert!(validate_quantum_alignment(&rooms).is_ok());
    }

    #[test]
    fn unaligned_position_fails() {
        let rooms = vec![room_at(8, 0, 0, 64, 64, 128)];
        let err = validate_quantum_alignment(&rooms).unwrap_err();
        assert!(err.to_string().contains("aligned"), "{}", err);
    }

    #[test]
    fn unaligned_dimension_fails() {
        let rooms = vec![room_at(0, 0, 0, 63, 64, 128)];
        let err = validate_quantum_alignment(&rooms).unwrap_err();
        assert!(err.to_string().contains("multiples"), "{}", err);
    }

    // ── validate_no_overlap ────────────────────────────────────────────

    #[test]
    fn non_overlapping_rooms_pass() {
        let rooms = vec![
            room_at(0, 0, 0, 64, 64, 128),
            room_at(80, 0, 0, 64, 64, 128),
            room_at(0, 80, 0, 64, 64, 128),
        ];
        assert!(validate_no_overlap(&rooms, 16).is_ok());
    }

    #[test]
    fn overlapping_rooms_fail() {
        let rooms = vec![
            room_at(0, 0, 0, 64, 64, 128),
            room_at(32, 0, 0, 64, 64, 128), // overlaps first
        ];
        let err = validate_no_overlap(&rooms, 16).unwrap_err();
        assert!(err.to_string().contains("overlap"), "{}", err);
    }
}
