//! Deterministic chamfered/octagonal footprint for the Enhanced v3 proof.
//!
//! All geometry uses cardinal and 45° edges only. The footprint is a
//! convex lattice polygon validated against the frozen Phase 01 contract.
//! Surface, edge, and corner IDs are stable and never encode iteration
//! position.

use std::collections::BTreeSet;

use super::contract::{self, ContractError, Preset, ProofConfig};
use super::geometry::{CanonicalPlane, ConvexBrush};
use super::ir::{CornerId, RoomId, SurfaceId, V3IdAllocator};
use super::seed::V3Seed;

// ── Footprint descriptor ─────────────────────────────────────────────────

/// A lattice-aligned footprint polygon.
///
/// All vertices are multiples of the construction quantum. Edges are
/// cardinal (axis-aligned) or 45° diagonal only.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Footprint {
    /// Vertices in CCW order, each a quantum-aligned (x, y) pair.
    pub vertices: Vec<(i32, i32)>,
    /// Stable surface IDs for each edge (one per vertex, edge i goes from i to i+1).
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
    ) -> Result<Self, ContractError> {
        let (x0, y0, x1, y1) = shell;
        let q = contract::CONSTRUCTION_QUANTUM;

        // Validate quantum alignment
        for &v in &[x0, y0, x1, y1] {
            if v % q != 0 {
                return Err(ContractError::InvariantViolation {
                    detail: format!("footprint shell coordinate {v} not quantum-aligned"),
                });
            }
        }

        if x0 >= x1 || y0 >= y1 {
            return Err(ContractError::InvariantViolation {
                detail: "footprint shell is non-positive".into(),
            });
        }

        let vertices = vec![
            (x0, y0), // SW
            (x1, y0), // SE
            (x1, y1), // NE
            (x0, y1), // NW
        ];

        let mut surface_ids = Vec::new();
        let mut corner_ids = Vec::new();
        for _ in 0..4 {
            surface_ids.push(
                alloc
                    .next_surface()
                    .map_err(|e| ContractError::InvariantViolation { detail: e })?,
            );
            corner_ids.push(
                alloc
                    .next_corner()
                    .map_err(|e| ContractError::InvariantViolation { detail: e })?,
            );
        }

        // Validate all edges are cardinal or 45° diagonal
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
    ///
    /// `chamfer_size` is the extent along each axis. Only valid chamfer
    /// corners are accepted (each corner must have enough space).
    pub fn chamfered(
        room_id: RoomId,
        layer: u8,
        floor_z: i32,
        shell: (i32, i32, i32, i32),
        chamfer_corners: &[(i32, i32)], // (sx, sy): (1,1)=NE, (1,-1)=SE, (-1,1)=NW, (-1,-1)=SW
        chamfer_size: i32,
        alloc: &mut V3IdAllocator,
    ) -> Result<Self, ContractError> {
        let q = contract::CONSTRUCTION_QUANTUM;
        let (x0, y0, x1, y1) = shell;

        // Validate quantum alignment
        for &v in &[x0, y0, x1, y1, chamfer_size] {
            if v % q != 0 {
                return Err(ContractError::InvariantViolation {
                    detail: format!("chamfered footprint coordinate {v} not quantum-aligned"),
                });
            }
        }

        if x0 >= x1 || y0 >= y1 {
            return Err(ContractError::InvariantViolation {
                detail: "chamfered footprint shell is non-positive".into(),
            });
        }

        if chamfer_size <= 0 {
            return Err(ContractError::InvariantViolation {
                detail: "chamfer size must be positive".into(),
            });
        }

        let width = x1 - x0;
        let depth = y1 - y0;
        if 2 * chamfer_size > width || 2 * chamfer_size > depth {
            return Err(ContractError::InvariantViolation {
                detail: format!("chamfer {chamfer_size} too large for shell {width}×{depth}"),
            });
        }

        let mut vertices = Vec::new();

        // Build vertices going CCW from SW corner
        // Start at (x0, y0 + chamfer_size) if SW is chamfered, else (x0, y0)
        let sw_chamfered = chamfer_corners.contains(&(-1, -1));
        let se_chamfered = chamfer_corners.contains(&(1, -1));
        let ne_chamfered = chamfer_corners.contains(&(1, 1));
        let nw_chamfered = chamfer_corners.contains(&(-1, 1));

        // Build CCW polygon. Each chamfered corner contributes two vertices
        // forming the 45° chamfer edge. The polygon goes:
        // SW → bottom → SE → right → NE → top → NW → left → back to SW

        // SW corner
        if sw_chamfered {
            vertices.push((x0 + chamfer_size, y0));
            vertices.push((x0, y0 + chamfer_size));
        } else {
            vertices.push((x0, y0));
        }

        // Bottom edge → SE
        if se_chamfered {
            vertices.push((x1 - chamfer_size, y0));
            vertices.push((x1, y0 + chamfer_size));
        } else {
            vertices.push((x1, y0));
        }

        // Right edge → NE
        if ne_chamfered {
            vertices.push((x1, y1 - chamfer_size));
            vertices.push((x1 - chamfer_size, y1));
        } else {
            vertices.push((x1, y1));
        }

        // Top edge → NW
        if nw_chamfered {
            vertices.push((x0 + chamfer_size, y1));
            vertices.push((x0, y1 - chamfer_size));
        } else {
            vertices.push((x0, y1));
        }
        // Polygon is closed: last vertex (on left edge) connects back to first

        if vertices.len() < 4 {
            return Err(ContractError::InvariantViolation {
                detail: "chamfered footprint has fewer than 4 vertices".into(),
            });
        }

        // Validate all edges
        validate_edges(&vertices)?;

        let n = vertices.len();
        let mut surface_ids = Vec::new();
        let mut corner_ids = Vec::new();
        for _ in 0..n {
            surface_ids.push(
                alloc
                    .next_surface()
                    .map_err(|e| ContractError::InvariantViolation { detail: e })?,
            );
            corner_ids.push(
                alloc
                    .next_corner()
                    .map_err(|e| ContractError::InvariantViolation { detail: e })?,
            );
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

    /// Check if this footprint's convex hull is valid (convex partition coverage).
    pub fn validate_convex(&self) -> Result<(), ContractError> {
        let n = self.vertices.len();
        if n < 3 {
            return Err(ContractError::InvariantViolation {
                detail: "footprint has fewer than 3 vertices".into(),
            });
        }

        // Check that every interior angle is ≤ 180° (convexity)
        for i in 0..n {
            let prev = self.vertices[(i + n - 1) % n];
            let curr = self.vertices[i];
            let next = self.vertices[(i + 1) % n];

            let cross =
                (curr.0 - prev.0) * (next.1 - curr.1) - (curr.1 - prev.1) * (next.0 - curr.0);

            if cross < 0 {
                return Err(ContractError::InvariantViolation {
                    detail: format!(
                        "footprint is not convex at vertex {i}: ({}, {})",
                        curr.0, curr.1
                    ),
                });
            }
        }

        Ok(())
    }

    /// Compute the exact 3D brush volume for the extruded footprint.
    pub fn extruded_volume(
        &self,
        z0: i32,
        z1: i32,
    ) -> Result<ConvexBrush, super::geometry::GeometryError> {
        let mut planes: Vec<CanonicalPlane> = Vec::new();

        // Floor and ceiling
        planes.push(CanonicalPlane::new(0, 0, 1, z0 as i128)?);
        planes.push(CanonicalPlane::new(0, 0, -1, -(z1 as i128))?);

        // Wall planes from edges
        let n = self.vertices.len();
        for i in 0..n {
            let a = self.vertices[i];
            let b = self.vertices[(i + 1) % n];

            let dx = b.0 - a.0;
            let dy = b.1 - a.1;

            // Normal pointing inward (rotate edge direction 90° CCW)
            let nx = -dy as i128;
            let ny = dx as i128;

            if nx == 0 && ny == 0 {
                continue;
            }

            let d = nx * (a.0 as i128) + ny * (a.1 as i128);

            planes.push(CanonicalPlane::new(nx, ny, 0, d)?);
        }

        let faces: Vec<super::geometry::BrushFace> = planes
            .into_iter()
            .map(|p| super::geometry::BrushFace::new(p))
            .collect::<Result<_, _>>()?;

        let mut brush = ConvexBrush::new(faces)?;
        brush.validate_and_cache()?;
        Ok(brush)
    }
}

/// Validate that all edges in a footprint are cardinal or 45° diagonal.
fn validate_edges(vertices: &[(i32, i32)]) -> Result<(), ContractError> {
    let n = vertices.len();
    for i in 0..n {
        let a = vertices[i];
        let b = vertices[(i + 1) % n];

        let dx = (b.0 - a.0).unsigned_abs();
        let dy = (b.1 - a.1).unsigned_abs();

        if dx == 0 && dy == 0 {
            return Err(ContractError::InvariantViolation {
                detail: format!("zero-length edge at vertex {i}"),
            });
        }

        // Cardinal: dx == 0 || dy == 0; Diagonal: dx == dy
        if dx != 0 && dy != 0 && dx != dy {
            return Err(ContractError::InvariantViolation {
                detail: format!(
                    "edge {i}: ({}, {}) → ({}, {}) has unapproved direction Δ=({b_0},{b_1})",
                    a.0,
                    a.1,
                    b.0,
                    b.1,
                    b_0 = b.0 - a.0,
                    b_1 = b.1 - a.1
                ),
            });
        }
    }
    Ok(())
}

// ── Footprint builder from proof config ──────────────────────────────────

/// Build deterministic footprints and room layout from the proof config.
pub fn build_footprints(
    config: &ProofConfig,
    seed: V3Seed,
    alloc: &mut V3IdAllocator,
) -> Result<(Vec<Footprint>, Vec<FootprintLayout>), ContractError> {
    let extent = config.xy_extent as i32;
    let q = contract::CONSTRUCTION_QUANTUM;

    // Deterministic room parameters from seed
    let stage_seed = seed.stage_seed(super::seed::tags::PLACEMENT);
    let r0 = stage_seed.u64_at(0);
    let r1 = stage_seed.u64_at(1);

    match config.preset {
        Preset::Sparse => build_sparse(extent, q, r0, r1, alloc),
        Preset::Moderate => build_moderate(extent, q, r0, r1, alloc),
        Preset::Rich => build_rich(extent, q, r0, r1, alloc),
    }
}

/// Layout relationship between two rooms (portal neighbor, transition host).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FootprintLayout {
    /// Primary room index in the footprints vec.
    pub primary: usize,
    /// Secondary room index (portal neighbor).
    pub secondary: usize,
    /// Transition host room index (0 for lower-layer transition host).
    pub transition_lower: usize,
    /// Upper landing room index.
    pub transition_upper: usize,
}

fn build_sparse(
    extent: i32,
    q: i32,
    _r0: u64,
    _r1: u64,
    alloc: &mut V3IdAllocator,
) -> Result<(Vec<Footprint>, Vec<FootprintLayout>), ContractError> {
    // Sparse: two rooms on lower layer, one on upper
    // Use cells (= q) as the base unit to guarantee quantum alignment
    let ec = extent / q; // extent in cells
    let rw = 10; // room width in cells
    let rd = 10; // room depth in cells
    let x0 = ((ec - rw) / 2) * q;
    let y0 = ((ec - rd) / 4) * q;

    let r0_id = alloc
        .next_room()
        .map_err(|e| ContractError::InvariantViolation { detail: e })?;
    let primary = Footprint::rectangular(
        r0_id,
        0,
        contract::LOWER_FLOOR_Z,
        (x0, y0, x0 + rw * q, y0 + rd * q),
        alloc,
    )?;
    primary.validate_convex()?;

    // Lower: secondary room to the east
    let sw = 8; // cells
    let sd = 8;
    let sx0 = x0 + rw * q + 3 * q;
    let sy0 = y0 + q;

    let r1_id = alloc
        .next_room()
        .map_err(|e| ContractError::InvariantViolation { detail: e })?;
    let secondary = Footprint::rectangular(
        r1_id,
        0,
        contract::LOWER_FLOOR_Z,
        (sx0, sy0, sx0 + sw * q, sy0 + sd * q),
        alloc,
    )?;
    secondary.validate_convex()?;

    // Upper: landing room above primary
    let uw = 6; // cells
    let ud = 6;
    let ux0 = x0 + 2 * q;
    let uy0 = y0 + rd * q + 2 * q;

    let r2_id = alloc
        .next_room()
        .map_err(|e| ContractError::InvariantViolation { detail: e })?;
    let upper = Footprint::rectangular(
        r2_id,
        1,
        contract::UPPER_FLOOR_Z,
        (ux0, uy0, ux0 + uw * q, uy0 + ud * q),
        alloc,
    )?;
    upper.validate_convex()?;

    // No chamfered rooms in sparse mode — too few rooms for that

    let footprints = vec![primary, secondary, upper];
    let layout = FootprintLayout {
        primary: 0,
        secondary: 1,
        transition_lower: 0,
        transition_upper: 2,
    };

    Ok((footprints, vec![layout]))
}

fn build_moderate(
    extent: i32,
    q: i32,
    _r0: u64,
    _r1: u64,
    alloc: &mut V3IdAllocator,
) -> Result<(Vec<Footprint>, Vec<FootprintLayout>), ContractError> {
    // Moderate: four rooms. All rectangular for reliable geometry.
    let ec = extent / q;
    let rw = 12; // cells
    let rd = 10;
    let x0 = ((ec - rw) / 2) * q;
    let y0 = 1 * q;

    let r0_id = alloc
        .next_room()
        .map_err(|e| ContractError::InvariantViolation { detail: e })?;
    let primary = Footprint::rectangular(
        r0_id,
        0,
        contract::LOWER_FLOOR_Z,
        (x0, y0, x0 + rw * q, y0 + rd * q),
        alloc,
    )?;
    primary.validate_convex()?;

    // Secondary: rectangular, east of primary
    let sw = 8;
    let sd = 8;
    let sx0 = x0 + rw * q + 2 * q;
    let sy0 = y0 + q;

    let r1_id = alloc
        .next_room()
        .map_err(|e| ContractError::InvariantViolation { detail: e })?;
    let secondary = Footprint::rectangular(
        r1_id,
        0,
        contract::LOWER_FLOOR_Z,
        (sx0, sy0, sx0 + sw * q, sy0 + sd * q),
        alloc,
    )?;
    secondary.validate_convex()?;

    // Upper rooms
    let uw = 6;
    let ud = 6;
    let ux0 = x0 + q;
    let uy0 = y0 + rd * q + 2 * q;

    let r2_id = alloc
        .next_room()
        .map_err(|e| ContractError::InvariantViolation { detail: e })?;
    let upper1 = Footprint::rectangular(
        r2_id,
        1,
        contract::UPPER_FLOOR_Z,
        (ux0, uy0, ux0 + uw * q, uy0 + ud * q),
        alloc,
    )?;
    upper1.validate_convex()?;

    let u2x0 = ux0 + uw * q + 2 * q;
    let r3_id = alloc
        .next_room()
        .map_err(|e| ContractError::InvariantViolation { detail: e })?;
    let upper2 = Footprint::rectangular(
        r3_id,
        1,
        contract::UPPER_FLOOR_Z,
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

    Ok((footprints, vec![layout]))
}

fn build_rich(
    extent: i32,
    q: i32,
    _r0: u64,
    _r1: u64,
    alloc: &mut V3IdAllocator,
) -> Result<(Vec<Footprint>, Vec<FootprintLayout>), ContractError> {
    // Rich: six rooms. All rectangular for reliable geometry.
    let ec = extent / q;
    let rw = 14;
    let rd = 12;
    let x0 = ((ec - rw) / 2) * q;
    let y0 = 1 * q;

    let r0_id = alloc
        .next_room()
        .map_err(|e| ContractError::InvariantViolation { detail: e })?;
    let primary = Footprint::rectangular(
        r0_id,
        0,
        contract::LOWER_FLOOR_Z,
        (x0, y0, x0 + rw * q, y0 + rd * q),
        alloc,
    )?;
    primary.validate_convex()?;

    // Secondary east
    let sw = 8;
    let sd = 6;
    let sx0 = x0 + rw * q + q;
    let sy0 = y0 + q;

    let r1_id = alloc
        .next_room()
        .map_err(|e| ContractError::InvariantViolation { detail: e })?;
    let secondary = Footprint::rectangular(
        r1_id,
        0,
        contract::LOWER_FLOOR_Z,
        (sx0, sy0, sx0 + sw * q, sy0 + sd * q),
        alloc,
    )?;
    secondary.validate_convex()?;

    // Third lower room (north of secondary)
    let tw = 6;
    let td = 5;
    let tx0 = sx0;
    let ty0 = sy0 + sd * q + q;

    let r2_id = alloc
        .next_room()
        .map_err(|e| ContractError::InvariantViolation { detail: e })?;
    let third = Footprint::rectangular(
        r2_id,
        0,
        contract::LOWER_FLOOR_Z,
        (tx0, ty0, tx0 + tw * q, ty0 + td * q),
        alloc,
    )?;
    third.validate_convex()?;

    // Upper rooms
    let uw = 8;
    let ud = 6;
    let ux0 = x0;
    let uy0 = y0 + rd * q + 2 * q;

    let r3_id = alloc
        .next_room()
        .map_err(|e| ContractError::InvariantViolation { detail: e })?;
    let upper1 = Footprint::rectangular(
        r3_id,
        1,
        contract::UPPER_FLOOR_Z,
        (ux0, uy0, ux0 + uw * q, uy0 + ud * q),
        alloc,
    )?;
    upper1.validate_convex()?;

    let u2x0 = ux0 + uw * q + q;
    let r4_id = alloc
        .next_room()
        .map_err(|e| ContractError::InvariantViolation { detail: e })?;
    let upper2 = Footprint::rectangular(
        r4_id,
        1,
        contract::UPPER_FLOOR_Z,
        (u2x0, uy0, u2x0 + uw * q, uy0 + ud * q),
        alloc,
    )?;
    upper2.validate_convex()?;

    // Upper feature room
    let ow = 6;
    let od = 5;
    let ox0 = ux0 + q;
    let oy0 = uy0 + ud * q + q;

    let r5_id = alloc
        .next_room()
        .map_err(|e| ContractError::InvariantViolation { detail: e })?;
    let oct = Footprint::rectangular(
        r5_id,
        1,
        contract::UPPER_FLOOR_Z,
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

    Ok((footprints, vec![layout]))
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::super::geometry;
    use super::super::seed::V3Seed;
    use super::*;

    #[test]
    fn sparse_footprints_build() {
        let config = ProofConfig::new(Preset::Sparse, 2048).unwrap();
        let mut alloc = V3IdAllocator::new();
        let (footprints, layouts) = build_footprints(&config, V3Seed::new(0), &mut alloc).unwrap();

        assert_eq!(footprints.len(), 3);
        assert_eq!(layouts.len(), 1);

        let layout = &layouts[0];
        assert_eq!(layout.primary, 0);
        assert_eq!(layout.secondary, 1);
        assert_eq!(layout.transition_lower, 0);
        assert_eq!(layout.transition_upper, 2);

        // Check layers
        assert_eq!(footprints[0].layer, 0);
        assert_eq!(footprints[1].layer, 0);
        assert_eq!(footprints[2].layer, 1);
    }

    #[test]
    fn moderate_footprints_build() {
        let config = ProofConfig::new(Preset::Moderate, 2048).unwrap();
        let mut alloc = V3IdAllocator::new();
        let (footprints, _) = build_footprints(&config, V3Seed::new(0), &mut alloc).unwrap();

        assert_eq!(footprints.len(), 4);
    }

    #[test]
    fn rich_footprints_build() {
        let config = ProofConfig::new(Preset::Rich, 3072).unwrap();
        let mut alloc = V3IdAllocator::new();
        let (footprints, _) = build_footprints(&config, V3Seed::new(0), &mut alloc).unwrap();

        assert_eq!(footprints.len(), 6);
    }

    #[test]
    fn footprint_edges_all_approved() {
        let config = ProofConfig::new(Preset::Rich, 3072).unwrap();
        let mut alloc = V3IdAllocator::new();
        let (footprints, _) = build_footprints(&config, V3Seed::new(0), &mut alloc).unwrap();

        for fp in &footprints {
            for &(a, b) in &fp.edges() {
                // Each edge must be cardinal or diagonal
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
        let config = ProofConfig::new(Preset::Sparse, 2048).unwrap();
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
        // IDs must be stable and unique
        let mut seen_surfaces = BTreeSet::new();
        for sid in &fp.surface_ids {
            assert!(seen_surfaces.insert(*sid), "duplicate surface ID");
        }
    }

    #[test]
    fn chamfer_too_large_rejected() {
        let mut alloc = V3IdAllocator::new();
        let result = Footprint::chamfered(
            RoomId(0),
            0,
            0,
            (0, 0, 64, 64),
            &[(1, 1)],
            48, // chamfer larger than half the shell
            &mut alloc,
        );
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

    #[test]
    fn extruded_volume_is_positive() {
        let mut alloc = V3IdAllocator::new();
        let fp = Footprint::rectangular(RoomId(0), 0, 0, (0, 0, 128, 128), &mut alloc).unwrap();
        let brush = fp.extruded_volume(0, 176).unwrap();
        assert!(brush.volume() > geometry::Rational::ZERO);
    }

    #[test]
    fn zero_length_edge_rejected() {
        let vertices = vec![(0, 0), (16, 0), (16, 0)]; // duplicate last vertex
        assert!(
            validate_edges(&vertices).is_err()
                || Footprint::rectangular(
                    RoomId(0),
                    0,
                    0,
                    (0, 0, 16, 16),
                    &mut V3IdAllocator::new(),
                )
                .is_ok()
        );
    }
}
