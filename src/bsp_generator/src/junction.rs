//! Explicit wall and junction geometry for generated BSP corridors.
//!
//! Quake brushes are additive: an overlapping corridor never subtracts a
//! doorway from a room wall. This module therefore emits only the solid pieces
//! around room portals and keeps junction closure solids in outer corner
//! quadrants, outside the 64-unit clear route through each junction centre.
//!
//! # Production junction architecture
//!
//! Per `DECISION-20260725-16`, production emission uses the corridor open-cell
//! union exclusively. The L/T/X/portal functions below are retained as
//! `pub(crate)` for diagnostic and test use but are never called by
//! [`crate::emission::build_emission`].

#![allow(dead_code)] // diagnostic L/T/X/portal functions retained for test access

use crate::config::CONSTRUCTION_QUANTUM;
use crate::error::GeneratorError;
use crate::intent::{Brush, BrushFace, Corridor, RoomIntent};

/// Wall texture role from the CC0 Stone Beta theme.
const WALL_TEXTURE: &str = "stone_wall";
/// Frozen wall/floor/ceiling slab thickness.
const WALL: i32 = CONSTRUCTION_QUANTUM as i32;

/// Create an axis-aligned solid brush from `min` to `max`.
///
/// Faces are emitted in canonical order: bottom, top, north, south, west,
/// east. Callers are responsible for passing a non-empty box. The approved
/// Standard Quake serializer (`DECISION-20260726-01`) uses `"texture" 0 0 0
/// 1.0 1.0`; this function stores only the plane points and texture identity.
pub fn make_brush(min: (i32, i32, i32), max: (i32, i32, i32), texture: &str) -> Brush {
    debug_assert!(min.0 < max.0 && min.1 < max.1 && min.2 < max.2);

    let tex = texture.to_string();

    Brush {
        faces: vec![
            BrushFace {
                plane_points: [
                    (min.0, max.1, min.2),
                    (min.0, min.1, min.2),
                    (max.0, min.1, min.2),
                ],
                texture: tex.clone(),
            },
            BrushFace {
                plane_points: [
                    (min.0, max.1, max.2),
                    (max.0, max.1, max.2),
                    (max.0, min.1, max.2),
                ],
                texture: tex.clone(),
            },
            BrushFace {
                plane_points: [
                    (min.0, max.1, max.2),
                    (min.0, max.1, min.2),
                    (max.0, max.1, min.2),
                ],
                texture: tex.clone(),
            },
            BrushFace {
                plane_points: [
                    (min.0, min.1, max.2),
                    (max.0, min.1, max.2),
                    (max.0, min.1, min.2),
                ],
                texture: tex.clone(),
            },
            BrushFace {
                plane_points: [
                    (min.0, max.1, max.2),
                    (min.0, min.1, max.2),
                    (min.0, min.1, min.2),
                ],
                texture: tex.clone(),
            },
            BrushFace {
                plane_points: [
                    (max.0, max.1, min.2),
                    (max.0, min.1, min.2),
                    (max.0, min.1, max.2),
                ],
                texture: tex,
            },
        ],
    }
}

// ── Release brush validation (G5) ─────────────────────────────────────────

/// Validate that a [`Brush`] satisfies every mandatory construction contract
/// required for safe Quake `.map` serialization.
///
/// Returns `Ok(())` or a typed [`GeneratorError`]. This is a release-mode
/// gate; the generator must never rely on `debug_assert` alone for brush
/// correctness.
pub fn validate_brush(brush: &Brush, index: usize) -> Result<(), GeneratorError> {
    let tag = || format!("brush {index}");

    // 1. Exactly six faces
    if brush.faces.len() != 6 {
        return Err(GeneratorError::InvariantViolation(format!(
            "{} has {} faces (must be 6)",
            tag(),
            brush.faces.len(),
        )));
    }

    // 2. Every face has non-collinear plane points
    for (fi, face) in brush.faces.iter().enumerate() {
        let [p0, p1, p2] = face.plane_points;
        let v1 = (p1.0 - p0.0, p1.1 - p0.1, p1.2 - p0.2);
        let v2 = (p2.0 - p0.0, p2.1 - p0.1, p2.2 - p0.2);
        let cross = (
            v1.1 * v2.2 - v1.2 * v2.1,
            v1.2 * v2.0 - v1.0 * v2.2,
            v1.0 * v2.1 - v1.1 * v2.0,
        );
        if cross == (0, 0, 0) {
            return Err(GeneratorError::InvariantViolation(format!(
                "{} face {fi} has collinear plane points",
                tag(),
            )));
        }
    }

    // 3. Positive volume: compute AABB from all plane points
    let mut min = (i32::MAX, i32::MAX, i32::MAX);
    let mut max = (i32::MIN, i32::MIN, i32::MIN);
    for face in &brush.faces {
        for &(x, y, z) in &face.plane_points {
            min.0 = min.0.min(x);
            min.1 = min.1.min(y);
            min.2 = min.2.min(z);
            max.0 = max.0.max(x);
            max.1 = max.1.max(y);
            max.2 = max.2.max(z);
        }
    }
    if min.0 >= max.0 || min.1 >= max.1 || min.2 >= max.2 {
        return Err(GeneratorError::InvariantViolation(format!(
            "{} has non-positive volume: AABB {:?} -> {:?}",
            tag(),
            min,
            max,
        )));
    }

    // 4. Consistent half-space: for an axis-aligned box, each face's plane
    //    must contain its three defining points and the opposite corner must
    //    lie on the negative side of the plane.
    for (fi, face) in brush.faces.iter().enumerate() {
        let [p0, p1, p2] = face.plane_points;
        let normal = plane_normal(p0, p1, p2);
        if normal == (0.0, 0.0, 0.0) {
            return Err(GeneratorError::InvariantViolation(format!(
                "{} face {fi} has zero-area plane",
                tag(),
            )));
        }
        // Pick a point known to be on the opposite side: the AABB corner
        // farthest from the plane along the normal.
        let test_point = if normal.0 > 0.0 {
            (min.0, min.1, min.2)
        } else if normal.0 < 0.0 {
            (max.0, max.1, max.2)
        } else if normal.1 > 0.0 {
            (min.0, min.1, min.2)
        } else if normal.1 < 0.0 {
            (max.0, max.1, max.2)
        } else if normal.2 > 0.0 {
            (min.0, min.1, min.2)
        } else {
            (max.0, max.1, max.2)
        };
        let d = signed_distance(p0, normal, test_point);
        if d > 0.0 {
            return Err(GeneratorError::InvariantViolation(format!(
                "{} face {fi} has inconsistent half-space (d={d:.2})",
                tag(),
            )));
        }
    }

    // 5. Quantum alignment: every coordinate must be a multiple of
    //    CONSTRUCTION_QUANTUM.
    let q = CONSTRUCTION_QUANTUM as i32;
    for (fi, face) in brush.faces.iter().enumerate() {
        for (pi, &(x, y, z)) in face.plane_points.iter().enumerate() {
            if x % q != 0 || y % q != 0 || z % q != 0 {
                return Err(GeneratorError::InvariantViolation(format!(
                    "{} face {fi} point {pi} ({x}, {y}, {z}) not quantum-aligned",
                    tag(),
                )));
            }
        }
    }

    // 6. Finite coordinates (i32 is always finite, but guard for future)
    //    Already satisfied by i32 type.

    // 7. Non-empty texture identity
    for (fi, face) in brush.faces.iter().enumerate() {
        if face.texture.is_empty() {
            return Err(GeneratorError::InvariantViolation(format!(
                "{} face {fi} has empty texture identity",
                tag(),
            )));
        }
    }

    Ok(())
}

/// Validate every brush in a slice, returning the first error.
pub fn validate_all_brushes(brushes: &[Brush]) -> Result<(), GeneratorError> {
    for (i, brush) in brushes.iter().enumerate() {
        validate_brush(brush, i)?;
    }
    Ok(())
}

/// Compute an outward-facing normal from three non-collinear plane points.
fn plane_normal(
    p0: (i32, i32, i32),
    p1: (i32, i32, i32),
    p2: (i32, i32, i32),
) -> (f64, f64, f64) {
    let v1 = (
        (p1.0 - p0.0) as f64,
        (p1.1 - p0.1) as f64,
        (p1.2 - p0.2) as f64,
    );
    let v2 = (
        (p2.0 - p0.0) as f64,
        (p2.1 - p0.1) as f64,
        (p2.2 - p0.2) as f64,
    );
    (
        v1.1 * v2.2 - v1.2 * v2.1,
        v1.2 * v2.0 - v1.0 * v2.2,
        v1.0 * v2.1 - v1.1 * v2.0,
    )
}

/// Signed distance from `point` to the plane defined by `origin` and
/// `normal`. Positive means the point is on the side the normal points to.
fn signed_distance(origin: (i32, i32, i32), normal: (f64, f64, f64), point: (i32, i32, i32)) -> f64 {
    let dx = point.0 as f64 - origin.0 as f64;
    let dy = point.1 as f64 - origin.1 as f64;
    let dz = point.2 as f64 - origin.2 as f64;
    let len = (normal.0 * normal.0 + normal.1 * normal.1 + normal.2 * normal.2).sqrt();
    if len == 0.0 {
        return 0.0;
    }
    (normal.0 * dx + normal.1 * dy + normal.2 * dz) / len
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Orientation {
    Horizontal,
    Vertical,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RoomWall {
    West,
    East,
    South,
    North,
}

fn orientation(corridor: &Corridor) -> Orientation {
    if (corridor.end.0 - corridor.start.0).abs() >= (corridor.end.1 - corridor.start.1).abs() {
        Orientation::Horizontal
    } else {
        Orientation::Vertical
    }
}

fn clear_half(corridor: &Corridor) -> i32 {
    corridor.width as i32 / 2
}

fn outer_half(corridor: &Corridor) -> i32 {
    clear_half(corridor) + WALL
}

fn shell_z(corridor: &Corridor) -> (i32, i32) {
    let z0 = corridor.start.2.min(corridor.end.2);
    (z0, z0 + WALL + corridor.height as i32 + WALL)
}

fn push_box(brushes: &mut Vec<Brush>, min: (i32, i32, i32), max: (i32, i32, i32)) {
    if min.0 < max.0 && min.1 < max.1 && min.2 < max.2 {
        brushes.push(make_brush(min, max, WALL_TEXTURE));
    }
}

fn signed_ring(center: i32, sign: i32, inner: i32, outer: i32) -> (i32, i32) {
    if sign > 0 {
        (center + inner, center + outer)
    } else {
        (center - outer, center - inner)
    }
}

fn shared_endpoint(a: &Corridor, b: &Corridor) -> Option<(i32, i32)> {
    for ap in [(a.start.0, a.start.1), (a.end.0, a.end.1)] {
        for bp in [(b.start.0, b.start.1), (b.end.0, b.end.1)] {
            if ap == bp {
                return Some(ap);
            }
        }
    }
    None
}

fn other_endpoint(corridor: &Corridor, junction: (i32, i32)) -> (i32, i32) {
    let start = (corridor.start.0, corridor.start.1);
    if start == junction {
        (corridor.end.0, corridor.end.1)
    } else {
        start
    }
}

/// Build the one outer-corner post needed by a perpendicular L turn.
///
/// Corridor shells extend one clear half-width past their centerline endpoint,
/// so the central 64×64 square remains open. The only uncovered solid is the
/// 16×16 post outside that clear square.
///
/// **Diagnostic only.** Production emission uses the corridor open-cell union
/// and never calls this function.
pub fn build_l_junction(a: &Corridor, b: &Corridor) -> Vec<Brush> {
    let (horizontal, vertical) = match (orientation(a), orientation(b)) {
        (Orientation::Horizontal, Orientation::Vertical) => (a, b),
        (Orientation::Vertical, Orientation::Horizontal) => (b, a),
        _ => return Vec::new(),
    };
    let Some((jx, jy)) = shared_endpoint(horizontal, vertical) else {
        return Vec::new();
    };

    let h_other = other_endpoint(horizontal, (jx, jy));
    let v_other = other_endpoint(vertical, (jx, jy));
    let missing_x_sign = if h_other.0 < jx { 1 } else { -1 };
    let missing_y_sign = if v_other.1 < jy { 1 } else { -1 };

    let clear = clear_half(horizontal).max(clear_half(vertical));
    let outer = outer_half(horizontal).max(outer_half(vertical));
    let (x0, x1) = signed_ring(jx, missing_x_sign, clear, outer);
    let (y0, y1) = signed_ring(jy, missing_y_sign, clear, outer);
    let (az0, az1) = shell_z(a);
    let (bz0, bz1) = shell_z(b);

    vec![make_brush(
        (x0, y0, az0.min(bz0)),
        (x1, y1, az1.max(bz1)),
        WALL_TEXTURE,
    )]
}

fn point_on_corridor_interior(point: (i32, i32), corridor: &Corridor) -> bool {
    let min_x = corridor.start.0.min(corridor.end.0);
    let max_x = corridor.start.0.max(corridor.end.0);
    let min_y = corridor.start.1.min(corridor.end.1);
    let max_y = corridor.start.1.max(corridor.end.1);

    match orientation(corridor) {
        Orientation::Horizontal => {
            point.1 == corridor.start.1 && point.0 > min_x && point.0 < max_x
        }
        Orientation::Vertical => point.0 == corridor.start.0 && point.1 > min_y && point.1 < max_y,
    }
}

fn termination_point(terminating: &Corridor, through: &Corridor) -> Option<(i32, i32)> {
    [
        (terminating.start.0, terminating.start.1),
        (terminating.end.0, terminating.end.1),
    ]
    .into_iter()
    .find(|point| point_on_corridor_interior(*point, through))
}

/// Build two outer corner posts where a terminating branch meets a through
/// corridor. No post enters the central clear square.
///
/// **Diagnostic only.** Production emission uses the corridor open-cell union
/// and never calls this function.
pub fn build_t_junction(terminating: &Corridor, through: &Corridor) -> Vec<Brush> {
    if orientation(terminating) == orientation(through) {
        return Vec::new();
    }
    let Some((jx, jy)) = termination_point(terminating, through) else {
        return Vec::new();
    };

    let other = other_endpoint(terminating, (jx, jy));
    let clear = clear_half(terminating).max(clear_half(through));
    let outer = outer_half(terminating).max(outer_half(through));
    let (tz0, tz1) = shell_z(terminating);
    let (hz0, hz1) = shell_z(through);
    let z0 = tz0.min(hz0);
    let z1 = tz1.max(hz1);
    let mut brushes = Vec::with_capacity(2);

    match orientation(terminating) {
        Orientation::Horizontal => {
            let branch_sign = if other.0 < jx { -1 } else { 1 };
            let (x0, x1) = signed_ring(jx, branch_sign, clear, outer);
            for y_sign in [-1, 1] {
                let (y0, y1) = signed_ring(jy, y_sign, clear, outer);
                push_box(&mut brushes, (x0, y0, z0), (x1, y1, z1));
            }
        }
        Orientation::Vertical => {
            let branch_sign = if other.1 < jy { -1 } else { 1 };
            let (y0, y1) = signed_ring(jy, branch_sign, clear, outer);
            for x_sign in [-1, 1] {
                let (x0, x1) = signed_ring(jx, x_sign, clear, outer);
                push_box(&mut brushes, (x0, y0, z0), (x1, y1, z1));
            }
        }
    }

    brushes
}

fn crossing_point(a: &Corridor, b: &Corridor) -> Option<(i32, i32)> {
    let (horizontal, vertical) = match (orientation(a), orientation(b)) {
        (Orientation::Horizontal, Orientation::Vertical) => (a, b),
        (Orientation::Vertical, Orientation::Horizontal) => (b, a),
        _ => return None,
    };
    let point = (vertical.start.0, horizontal.start.1);
    let hx0 = horizontal.start.0.min(horizontal.end.0);
    let hx1 = horizontal.start.0.max(horizontal.end.0);
    let vy0 = vertical.start.1.min(vertical.end.1);
    let vy1 = vertical.start.1.max(vertical.end.1);
    (point.0 >= hx0 && point.0 <= hx1 && point.1 >= vy0 && point.1 <= vy1).then_some(point)
}

/// Build four wall-thickness corner posts around an X crossing while leaving
/// the complete central 64×64 clear square untouched.
///
/// **Diagnostic only.** Production emission uses the corridor open-cell union
/// and never calls this function.
pub fn build_x_junction(a: &Corridor, b: &Corridor) -> Vec<Brush> {
    let Some((jx, jy)) = crossing_point(a, b) else {
        return Vec::new();
    };
    let clear = clear_half(a).max(clear_half(b));
    let outer = outer_half(a).max(outer_half(b));
    let (az0, az1) = shell_z(a);
    let (bz0, bz1) = shell_z(b);
    let z0 = az0.min(bz0);
    let z1 = az1.max(bz1);
    let mut brushes = Vec::with_capacity(4);

    for x_sign in [-1, 1] {
        let (x0, x1) = signed_ring(jx, x_sign, clear, outer);
        for y_sign in [-1, 1] {
            let (y0, y1) = signed_ring(jy, y_sign, clear, outer);
            push_box(&mut brushes, (x0, y0, z0), (x1, y1, z1));
        }
    }
    brushes
}

fn nearest_room_wall(corridor: &Corridor, room: &RoomIntent) -> (RoomWall, (i32, i32)) {
    let min_x = room.position.0;
    let max_x = min_x + room.dimensions.0 as i32;
    let min_y = room.position.1;
    let max_y = min_y + room.dimensions.1 as i32;
    let mut best: Option<(i32, usize, RoomWall, (i32, i32))> = None;

    for point in [
        (corridor.start.0, corridor.start.1),
        (corridor.end.0, corridor.end.1),
    ] {
        let cy = point.1.clamp(min_y, max_y);
        let cx = point.0.clamp(min_x, max_x);
        let candidates = [
            (
                (point.0 - min_x).abs() + (point.1 - cy).abs(),
                0,
                RoomWall::West,
                (min_x, cy),
            ),
            (
                (point.0 - max_x).abs() + (point.1 - cy).abs(),
                1,
                RoomWall::East,
                (max_x, cy),
            ),
            (
                (point.1 - min_y).abs() + (point.0 - cx).abs(),
                2,
                RoomWall::South,
                (cx, min_y),
            ),
            (
                (point.1 - max_y).abs() + (point.0 - cx).abs(),
                3,
                RoomWall::North,
                (cx, max_y),
            ),
        ];
        for candidate in candidates {
            if best
                .as_ref()
                .is_none_or(|current| (candidate.0, candidate.1) < (current.0, current.1))
            {
                best = Some(candidate);
            }
        }
    }

    best.map(|(_, _, wall, contact)| (wall, contact))
        .unwrap_or((RoomWall::West, (min_x, min_y)))
}

/// Build only the solid target-wall pieces around a corridor portal.
///
/// The aperture begins at the top of the floor slab and has the corridor's
/// full clear width and clear height. No brush is emitted in that aperture;
/// returned brushes are the low/high side columns and the lintel above it.
///
/// **Diagnostic only.** Production emission uses [`crate::emission::build_split_wall`]
/// and never calls this function.
pub fn build_room_portal(corridor: &Corridor, room: &RoomIntent) -> Vec<Brush> {
    let (wall, (cx, cy)) = nearest_room_wall(corridor, room);
    let min_x = room.position.0;
    let max_x = min_x + room.dimensions.0 as i32;
    let min_y = room.position.1;
    let max_y = min_y + room.dimensions.1 as i32;
    let z0 = room.position.2;
    let z1 = z0 + room.dimensions.2 as i32;
    let opening_bottom = z0 + WALL;
    let opening_top = (opening_bottom + corridor.height as i32).min(z1);
    let half = clear_half(corridor);
    let mut brushes = Vec::with_capacity(3);

    match wall {
        RoomWall::West | RoomWall::East => {
            let open_min = (cy - half).max(min_y + WALL);
            let open_max = (cy + half).min(max_y - WALL);
            let (wx0, wx1) = if wall == RoomWall::West {
                (min_x, min_x + WALL)
            } else {
                (max_x - WALL, max_x)
            };
            if open_min >= open_max {
                push_box(&mut brushes, (wx0, min_y, z0), (wx1, max_y, z1));
                return brushes;
            }
            push_box(&mut brushes, (wx0, min_y, z0), (wx1, open_min, z1));
            push_box(&mut brushes, (wx0, open_max, z0), (wx1, max_y, z1));
            push_box(
                &mut brushes,
                (wx0, open_min, opening_top),
                (wx1, open_max, z1),
            );
        }
        RoomWall::South | RoomWall::North => {
            let open_min = (cx - half).max(min_x + WALL);
            let open_max = (cx + half).min(max_x - WALL);
            let (wy0, wy1) = if wall == RoomWall::South {
                (min_y, min_y + WALL)
            } else {
                (max_y - WALL, max_y)
            };
            if open_min >= open_max {
                push_box(&mut brushes, (min_x, wy0, z0), (max_x, wy1, z1));
                return brushes;
            }
            push_box(&mut brushes, (min_x, wy0, z0), (open_min, wy1, z1));
            push_box(&mut brushes, (open_max, wy0, z0), (max_x, wy1, z1));
            push_box(
                &mut brushes,
                (open_min, wy0, opening_top),
                (open_max, wy1, z1),
            );
        }
    }

    brushes
}

fn terminates_at(a: &Corridor, b: &Corridor) -> bool {
    termination_point(a, b).is_some()
}

/// Build all unique L/T/X outer-corner closures in a corridor network.
///
/// **Diagnostic only.** Production emission uses the corridor open-cell union
/// and never calls this function.
pub fn build_junction_closures(corridors: &[Corridor]) -> Vec<Brush> {
    let mut brushes = Vec::new();

    for i in 0..corridors.len() {
        for j in (i + 1)..corridors.len() {
            let a = &corridors[i];
            let b = &corridors[j];
            if orientation(a) == orientation(b) {
                continue;
            }

            let candidates = if terminates_at(a, b) {
                build_t_junction(a, b)
            } else if terminates_at(b, a) {
                build_t_junction(b, a)
            } else if shared_endpoint(a, b).is_some() {
                build_l_junction(a, b)
            } else {
                build_x_junction(a, b)
            };

            for brush in candidates {
                if !brushes.contains(&brush) {
                    brushes.push(brush);
                }
            }
        }
    }

    brushes
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

    fn bounds(brush: &Brush) -> ((i32, i32, i32), (i32, i32, i32)) {
        let mut min = (i32::MAX, i32::MAX, i32::MAX);
        let mut max = (i32::MIN, i32::MIN, i32::MIN);
        for face in &brush.faces {
            for &(x, y, z) in &face.plane_points {
                min = (min.0.min(x), min.1.min(y), min.2.min(z));
                max = (max.0.max(x), max.1.max(y), max.2.max(z));
            }
        }
        (min, max)
    }

    fn contains_open(brush: &Brush, point: (i32, i32, i32)) -> bool {
        let (min, max) = bounds(brush);
        point.0 > min.0
            && point.0 < max.0
            && point.1 > min.1
            && point.1 < max.1
            && point.2 > min.2
            && point.2 < max.2
    }

    #[test]
    fn make_brush_has_canonical_faces() {
        let brush = make_brush((0, 0, 0), (64, 64, 128), "test");
        assert_eq!(brush.faces.len(), 6);
        assert!(brush.faces.iter().all(|face| face.texture == "test"));
    }

    #[test]
    fn l_closure_does_not_occupy_junction_centre() {
        let h = corridor_h(0, 0, 0, 128);
        let v = corridor_v(128, 0, 0, 128);
        let brushes = build_l_junction(&h, &v);
        assert_eq!(brushes.len(), 1);
        assert!(!contains_open(&brushes[0], (128, 0, 40)));
        assert_eq!(bounds(&brushes[0]), ((160, -48, 0), (176, -32, 112)));
    }

    #[test]
    fn t_and_x_closures_leave_clear_centres() {
        let through = corridor_v(64, 0, 0, 192);
        let terminating = corridor_h(0, 64, 0, 64);
        let t = build_t_junction(&terminating, &through);
        assert_eq!(t.len(), 2);
        assert!(t.iter().all(|brush| !contains_open(brush, (64, 64, 40))));

        let h = corridor_h(0, 64, 0, 192);
        let x = build_x_junction(&h, &through);
        assert_eq!(x.len(), 4);
        assert!(x.iter().all(|brush| !contains_open(brush, (64, 64, 40))));
    }

    #[test]
    fn portal_returns_wall_pieces_not_an_opening_solid() {
        let room = room(0, 0, 0, 96, 96, 192);
        let corridor = Corridor {
            start: (96, 48, 0),
            end: (192, 48, 0),
            width: 64,
            height: 80,
        };
        let pieces = build_room_portal(&corridor, &room);
        assert_eq!(pieces.len(), 3);
        assert!(pieces
            .iter()
            .all(|brush| { brush.faces.iter().all(|face| face.texture == WALL_TEXTURE) }));
        assert!(pieces
            .iter()
            .all(|brush| !contains_open(brush, (88, 48, 40))));
    }
}
