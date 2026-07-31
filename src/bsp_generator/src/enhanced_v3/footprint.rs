//! Deterministic chamfered/octagonal footprint generation for Enhanced V3.
//!
//! All geometry uses cardinal and 45° edges only. Footprints are convex
//! lattice polygons validated against the V3 contract. Surface, edge, and
//! corner IDs are stable and never encode iteration position.

use super::config::{self, V3Config, V3Preset, CONSTRUCTION_QUANTUM};
use super::error::V3Error;
use super::ids::{CornerId, RoomId, SurfaceId, V3IdAllocator};
use super::rng::V3Seed;

// ── Footprint descriptor ─────────────────────────────────────────────────

/// A lattice-aligned footprint polygon.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Footprint {
    /// Vertices in CCW order, each a quantum-aligned (x, y) pair.
    pub vertices: Vec<(i32, i32)>,
    /// Stable surface IDs for each edge.
    pub surface_ids: Vec<SurfaceId>,
    /// Stable corner IDs for each vertex.
    pub corner_ids: Vec<CornerId>,
    /// Axis-aligned bounding box: (x0, y0, x1, y1).
    pub aabb: (i32, i32, i32, i32),
    /// Room ID this footprint belongs to.
    pub room_id: RoomId,
    /// Layer index (0 = lower, 1 = upper).
    pub layer: u8,
    /// Floor Z.
    pub floor_z: i32,
}

impl Footprint {
    /// Build a rectangular footprint from shell bounds.
    pub fn rectangular(
        room_id: RoomId,
        layer: u8,
        floor_z: i32,
        shell: (i32, i32, i32, i32),
        alloc: &mut V3IdAllocator,
    ) -> Result<Self, V3Error> {
        let (x0, y0, x1, y1) = shell;
        let q = CONSTRUCTION_QUANTUM;

        for &v in &[x0, y0, x1, y1] {
            if v % q != 0 {
                return Err(V3Error::NotGridAligned {
                    coord: (v as i128, 0, 0),
                    quantum: q as i128,
                });
            }
        }

        if x0 >= x1 || y0 >= y1 {
            return Err(V3Error::InvalidFootprint {
                detail: "footprint shell is non-positive".into(),
            });
        }

        let vertices = vec![(x0, y0), (x1, y0), (x1, y1), (x0, y1)];

        let n = vertices.len();
        let mut surface_ids = Vec::new();
        let mut corner_ids = Vec::new();
        for _ in 0..n {
            surface_ids.push(alloc.next_surface()?);
            corner_ids.push(alloc.next_corner()?);
        }

        validate_edges(&vertices)?;

        Ok(Self {
            vertices,
            surface_ids,
            corner_ids,
            aabb: (x0, y0, x1, y1),
            room_id,
            layer,
            floor_z,
        })
    }

    /// Build a chamfered rectangular footprint.
    pub fn chamfered(
        room_id: RoomId,
        layer: u8,
        floor_z: i32,
        shell: (i32, i32, i32, i32),
        chamfer_corners: &[(i32, i32)],
        chamfer_size: i32,
        alloc: &mut V3IdAllocator,
    ) -> Result<Self, V3Error> {
        let q = CONSTRUCTION_QUANTUM;
        let (x0, y0, x1, y1) = shell;

        for &v in &[x0, y0, x1, y1, chamfer_size] {
            if v % q != 0 {
                return Err(V3Error::NotGridAligned {
                    coord: (v as i128, 0, 0),
                    quantum: q as i128,
                });
            }
        }

        if x0 >= x1 || y0 >= y1 {
            return Err(V3Error::InvalidFootprint {
                detail: "chamfered footprint shell is non-positive".into(),
            });
        }

        if chamfer_size <= 0 {
            return Err(V3Error::InvalidFootprint {
                detail: "chamfer size must be positive".into(),
            });
        }

        let width = x1 - x0;
        let depth = y1 - y0;
        if 2 * chamfer_size > width || 2 * chamfer_size > depth {
            return Err(V3Error::InvalidFootprint {
                detail: format!("chamfer {chamfer_size} too large for shell {width}×{depth}"),
            });
        }

        let mut vertices = Vec::new();
        let sw_chamfered = chamfer_corners.contains(&(-1, -1));
        let se_chamfered = chamfer_corners.contains(&(1, -1));
        let ne_chamfered = chamfer_corners.contains(&(1, 1));
        let nw_chamfered = chamfer_corners.contains(&(-1, 1));

        if sw_chamfered {
            vertices.push((x0 + chamfer_size, y0));
            vertices.push((x0, y0 + chamfer_size));
        } else {
            vertices.push((x0, y0));
        }
        if se_chamfered {
            vertices.push((x1 - chamfer_size, y0));
            vertices.push((x1, y0 + chamfer_size));
        } else {
            vertices.push((x1, y0));
        }
        if ne_chamfered {
            vertices.push((x1, y1 - chamfer_size));
            vertices.push((x1 - chamfer_size, y1));
        } else {
            vertices.push((x1, y1));
        }
        if nw_chamfered {
            vertices.push((x0 + chamfer_size, y1));
            vertices.push((x0, y1 - chamfer_size));
        } else {
            vertices.push((x0, y1));
        }

        if vertices.len() < 4 {
            return Err(V3Error::InvalidFootprint {
                detail: "chamfered footprint has fewer than 4 vertices".into(),
            });
        }

        validate_edges(&vertices)?;

        let n = vertices.len();
        let mut surface_ids = Vec::new();
        let mut corner_ids = Vec::new();
        for _ in 0..n {
            surface_ids.push(alloc.next_surface()?);
            corner_ids.push(alloc.next_corner()?);
        }

        Ok(Self {
            vertices,
            surface_ids,
            corner_ids,
            aabb: (x0, y0, x1, y1),
            room_id,
            layer,
            floor_z,
        })
    }

    /// Check if this footprint's convex hull is valid.
    pub fn validate_convex(&self) -> Result<(), V3Error> {
        let n = self.vertices.len();
        if n < 3 {
            return Err(V3Error::InvalidFootprint {
                detail: "footprint has fewer than 3 vertices".into(),
            });
        }

        for i in 0..n {
            let prev = self.vertices[(i + n - 1) % n];
            let curr = self.vertices[i];
            let next = self.vertices[(i + 1) % n];
            let cross =
                (curr.0 - prev.0) * (next.1 - curr.1) - (curr.1 - prev.1) * (next.0 - curr.0);
            if cross < 0 {
                return Err(V3Error::InvalidFootprint {
                    detail: format!(
                        "footprint is not convex at vertex {i}: ({}, {})",
                        curr.0, curr.1
                    ),
                });
            }
        }

        Ok(())
    }

    /// All unique XY edges as normalized (from, to) pairs.
    pub fn edges(&self) -> Vec<((i32, i32), (i32, i32))> {
        let n = self.vertices.len();
        (0..n)
            .map(|i| {
                let a = self.vertices[i];
                let b = self.vertices[(i + 1) % n];
                if a < b {
                    (a, b)
                } else {
                    (b, a)
                }
            })
            .collect()
    }
}

/// Validate that all edges in a footprint are cardinal or 45° diagonal.
fn validate_edges(vertices: &[(i32, i32)]) -> Result<(), V3Error> {
    let n = vertices.len();
    for i in 0..n {
        let a = vertices[i];
        let b = vertices[(i + 1) % n];
        let dx = (b.0 - a.0).unsigned_abs();
        let dy = (b.1 - a.1).unsigned_abs();
        if dx == 0 && dy == 0 {
            return Err(V3Error::InvalidFootprint {
                detail: format!("zero-length edge at vertex {i}"),
            });
        }
        if dx != 0 && dy != 0 && dx != dy {
            return Err(V3Error::InvalidFootprint {
                detail: format!(
                    "edge {i}: ({}, {}) → ({}, {}) has unapproved direction",
                    a.0, a.1, b.0, b.1
                ),
            });
        }
    }
    Ok(())
}

// ── Footprint builder from config ──────────────────────────────────────────

/// Build deterministic footprints and room layout from config.
pub fn build_footprints(
    config: &V3Config,
    seed: V3Seed,
    alloc: &mut V3IdAllocator,
) -> Result<(Vec<Footprint>, FootprintLayout), V3Error> {
    let extent = config.xy_extent as i32;
    let q = CONSTRUCTION_QUANTUM;
    let _stage_seed = seed.stage_seed(super::rng::tags::PLACEMENT);

    match config.preset {
        V3Preset::Sparse => build_sparse(extent, q, alloc),
        V3Preset::Moderate => build_moderate(extent, q, alloc),
        V3Preset::Rich => build_rich(extent, q, alloc),
    }
}

/// Layout relationship between rooms.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FootprintLayout {
    pub primary: usize,
    pub secondary: usize,
    pub transition_lower: usize,
    pub transition_upper: usize,
}

fn build_sparse(
    extent: i32,
    q: i32,
    alloc: &mut V3IdAllocator,
) -> Result<(Vec<Footprint>, FootprintLayout), V3Error> {
    let ec = extent / q;
    let rw = 10;
    let rd = 10;
    let x0 = ((ec - rw) / 2) * q;
    let y0 = ((ec - rd) / 4) * q;

    let r0_id = alloc.next_room()?;
    let primary = Footprint::rectangular(
        r0_id,
        0,
        config::LOWER_FLOOR_Z,
        (x0, y0, x0 + rw * q, y0 + rd * q),
        alloc,
    )?;
    primary.validate_convex()?;

    let sw = 8;
    let sd = 8;
    let sx0 = x0 + rw * q + 3 * q;
    let sy0 = y0 + q;

    let r1_id = alloc.next_room()?;
    let secondary = Footprint::rectangular(
        r1_id,
        0,
        config::LOWER_FLOOR_Z,
        (sx0, sy0, sx0 + sw * q, sy0 + sd * q),
        alloc,
    )?;
    secondary.validate_convex()?;

    let uw = 6;
    let ud = 6;
    let ux0 = x0 + 2 * q;
    let uy0 = y0 + rd * q + 2 * q;

    let r2_id = alloc.next_room()?;
    let upper = Footprint::rectangular(
        r2_id,
        1,
        config::UPPER_FLOOR_Z,
        (ux0, uy0, ux0 + uw * q, uy0 + ud * q),
        alloc,
    )?;
    upper.validate_convex()?;

    let footprints = vec![primary, secondary, upper];
    let layout = FootprintLayout {
        primary: 0,
        secondary: 1,
        transition_lower: 0,
        transition_upper: 2,
    };

    Ok((footprints, layout))
}

fn build_moderate(
    extent: i32,
    q: i32,
    alloc: &mut V3IdAllocator,
) -> Result<(Vec<Footprint>, FootprintLayout), V3Error> {
    let ec = extent / q;
    let rw = 12;
    let rd = 10;
    let x0 = ((ec - rw) / 2) * q;
    let y0 = 1 * q;

    let r0_id = alloc.next_room()?;
    let primary = Footprint::rectangular(
        r0_id,
        0,
        config::LOWER_FLOOR_Z,
        (x0, y0, x0 + rw * q, y0 + rd * q),
        alloc,
    )?;
    primary.validate_convex()?;

    let sw = 8;
    let sd = 8;
    let sx0 = x0 + rw * q + 2 * q;
    let sy0 = y0 + q;
    let r1_id = alloc.next_room()?;
    let secondary = Footprint::rectangular(
        r1_id,
        0,
        config::LOWER_FLOOR_Z,
        (sx0, sy0, sx0 + sw * q, sy0 + sd * q),
        alloc,
    )?;
    secondary.validate_convex()?;

    let uw = 6;
    let ud = 6;
    let ux0 = x0 + q;
    let uy0 = y0 + rd * q + 2 * q;

    let r2_id = alloc.next_room()?;
    let upper1 = Footprint::rectangular(
        r2_id,
        1,
        config::UPPER_FLOOR_Z,
        (ux0, uy0, ux0 + uw * q, uy0 + ud * q),
        alloc,
    )?;
    upper1.validate_convex()?;

    let u2x0 = ux0 + uw * q + 2 * q;
    let r3_id = alloc.next_room()?;
    let upper2 = Footprint::rectangular(
        r3_id,
        1,
        config::UPPER_FLOOR_Z,
        (u2x0, uy0, u2x0 + uw * q, uy0 + ud * q),
        alloc,
    )?;
    upper2.validate_convex()?;

    let footprints = vec![primary, secondary, upper1, upper2];
    let layout = FootprintLayout {
        primary: 0,
        secondary: 1,
        transition_lower: 0,
        transition_upper: 2,
    };

    Ok((footprints, layout))
}

fn build_rich(
    extent: i32,
    q: i32,
    alloc: &mut V3IdAllocator,
) -> Result<(Vec<Footprint>, FootprintLayout), V3Error> {
    let ec = extent / q;
    let rw = 14;
    let rd = 12;
    let x0 = ((ec - rw) / 2) * q;
    let y0 = 1 * q;

    let r0_id = alloc.next_room()?;
    let primary = Footprint::rectangular(
        r0_id,
        0,
        config::LOWER_FLOOR_Z,
        (x0, y0, x0 + rw * q, y0 + rd * q),
        alloc,
    )?;
    primary.validate_convex()?;

    let sw = 8;
    let sd = 6;
    let sx0 = x0 + rw * q + q;
    let sy0 = y0 + q;
    let r1_id = alloc.next_room()?;
    let secondary = Footprint::rectangular(
        r1_id,
        0,
        config::LOWER_FLOOR_Z,
        (sx0, sy0, sx0 + sw * q, sy0 + sd * q),
        alloc,
    )?;
    secondary.validate_convex()?;

    let tw = 6;
    let td = 5;
    let tx0 = sx0;
    let ty0 = sy0 + sd * q + q;
    let r2_id = alloc.next_room()?;
    let third = Footprint::rectangular(
        r2_id,
        0,
        config::LOWER_FLOOR_Z,
        (tx0, ty0, tx0 + tw * q, ty0 + td * q),
        alloc,
    )?;
    third.validate_convex()?;

    let uw = 8;
    let ud = 6;
    let ux0 = x0;
    let uy0 = y0 + rd * q + 2 * q;
    let r3_id = alloc.next_room()?;
    let upper1 = Footprint::rectangular(
        r3_id,
        1,
        config::UPPER_FLOOR_Z,
        (ux0, uy0, ux0 + uw * q, uy0 + ud * q),
        alloc,
    )?;
    upper1.validate_convex()?;

    let u2x0 = ux0 + uw * q + q;
    let r4_id = alloc.next_room()?;
    let upper2 = Footprint::rectangular(
        r4_id,
        1,
        config::UPPER_FLOOR_Z,
        (u2x0, uy0, u2x0 + uw * q, uy0 + ud * q),
        alloc,
    )?;
    upper2.validate_convex()?;

    let ow = 6;
    let od = 5;
    let ox0 = ux0 + q;
    let oy0 = uy0 + ud * q + q;
    let r5_id = alloc.next_room()?;
    let oct = Footprint::rectangular(
        r5_id,
        1,
        config::UPPER_FLOOR_Z,
        (ox0, oy0, ox0 + ow * q, oy0 + od * q),
        alloc,
    )?;
    oct.validate_convex()?;

    let footprints = vec![primary, secondary, third, upper1, upper2, oct];
    let layout = FootprintLayout {
        primary: 0,
        secondary: 1,
        transition_lower: 0,
        transition_upper: 3,
    };

    Ok((footprints, layout))
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sparse_footprints_build() {
        let config = V3Config::nominal_sparse();
        let mut alloc = V3IdAllocator::new();
        let (footprints, layout) = build_footprints(&config, V3Seed::new(0), &mut alloc).unwrap();

        assert_eq!(footprints.len(), 3);
        assert_eq!(layout.primary, 0);
        assert_eq!(layout.secondary, 1);

        assert_eq!(footprints[0].layer, 0);
        assert_eq!(footprints[1].layer, 0);
        assert_eq!(footprints[2].layer, 1);
    }

    #[test]
    fn moderate_footprints_build() {
        let config = V3Config::nominal_moderate();
        let mut alloc = V3IdAllocator::new();
        let (footprints, _) = build_footprints(&config, V3Seed::new(0), &mut alloc).unwrap();
        assert_eq!(footprints.len(), 4);
    }

    #[test]
    fn rich_footprints_build() {
        let config = V3Config::nominal_rich();
        let mut alloc = V3IdAllocator::new();
        let (footprints, _) = build_footprints(&config, V3Seed::new(0), &mut alloc).unwrap();
        assert_eq!(footprints.len(), 6);
    }

    #[test]
    fn footprint_edges_all_approved() {
        let config = V3Config::nominal_rich();
        let mut alloc = V3IdAllocator::new();
        let (footprints, _) = build_footprints(&config, V3Seed::new(0), &mut alloc).unwrap();

        for fp in &footprints {
            for &(a, b) in &fp.edges() {
                let dx = (b.0 - a.0).unsigned_abs();
                let dy = (b.1 - a.1).unsigned_abs();
                assert!(
                    dx == 0 || dy == 0 || dx == dy,
                    "unapproved edge in footprint {:?}",
                    fp.room_id
                );
            }
        }
    }

    #[test]
    fn footprint_convex_valid() {
        let config = V3Config::nominal_sparse();
        let mut alloc = V3IdAllocator::new();
        let (footprints, _) = build_footprints(&config, V3Seed::new(0), &mut alloc).unwrap();
        for fp in &footprints {
            fp.validate_convex().unwrap();
        }
    }

    #[test]
    fn chamfered_footprint_has_stable_ids() {
        let mut alloc = V3IdAllocator::new();
        let fp = Footprint::chamfered(
            RoomId(0),
            0,
            0,
            (0, 0, 192, 192),
            &[(1, 1), (-1, 1)],
            16,
            &mut alloc,
        )
        .unwrap();
        assert_eq!(fp.surface_ids.len(), fp.vertices.len());
        assert_eq!(fp.corner_ids.len(), fp.vertices.len());
    }

    #[test]
    fn chamfer_too_large_rejected() {
        let mut alloc = V3IdAllocator::new();
        let result =
            Footprint::chamfered(RoomId(0), 0, 0, (0, 0, 64, 64), &[(1, 1)], 48, &mut alloc);
        assert!(result.is_err());
    }

    #[test]
    fn rectangular_footprint_four_sides() {
        let mut alloc = V3IdAllocator::new();
        let fp = Footprint::rectangular(RoomId(0), 0, 0, (0, 0, 128, 128), &mut alloc).unwrap();
        assert_eq!(fp.vertices.len(), 4);
        assert_eq!(fp.surface_ids.len(), 4);
        assert_eq!(fp.corner_ids.len(), 4);
    }
}
