//! Richness-owned geometry adapters and primitives.
//!
//! Reuses exact baseline geometry primitives (ConvexBrush, CanonicalPlane,
//! BrushFace, FaceRole) through Richness-owned adapters. The baseline
//! execution path never calls Richness code and is never modified.
//!
//! # Contract
//!
//! - All coordinates are i128 (Quake units, integer only).
//! - All dimensions are quantum-aligned (multiples of 16).
//! - Approved normals only: cardinal XY/Z + 45-degree XY diagonals.
//! - Every brush is validated for positive volume and approved normals.
//! - No floats; exact rational arithmetic through the baseline kernel.
//! - Crate-private; canonical ordering.

use crate::enhanced_v3::geometry::{
    self, BrushFace, CanonicalPlane, ConvexBrush, Point3, Rational,
};
use crate::enhanced_v3::richness::error::{
    RichnessError, RichnessErrorCategory, RichnessErrorCode,
};

use super::footprint::Footprint3D;

// ── Frozen constants ───────────────────────────────────────────────────────

/// Construction quantum (16 Quake units).
pub(crate) const QUANTUM: i128 = 16;

/// Wall thickness (16 Quake units).
pub(crate) const WALL_THICKNESS: i128 = 16;

/// Floor Z minimum.
pub(crate) const FLOOR_Z_MIN: i128 = 0;

/// Floor Z maximum (top of floor slab).
pub(crate) const FLOOR_Z_MAX: i128 = 16;

/// Wall Z minimum (bottom of wall, on top of floor).
pub(crate) const WALL_Z_MIN: i128 = 16;

/// Wall Z maximum (top of wall, under ceiling).
pub(crate) const WALL_Z_MAX: i128 = 160;

/// Ceiling Z minimum (bottom of ceiling, on top of walls).
pub(crate) const CEILING_Z_MIN: i128 = 160;

/// Ceiling Z maximum (top of ceiling slab).
pub(crate) const CEILING_Z_MAX: i128 = 176;

/// Default chamfer size in Quake units (64 = 4 × quantum).
pub(crate) const DEFAULT_CHAMFER_SIZE: i128 = 64;

// ── Coordinate conversion ─────────────────────────────────────────────────

/// Convert grid footprint bounds to Quake i128 ranges.
#[inline]
pub(crate) fn footprint_quake_bounds(fp: &Footprint3D) -> (i128, i128, i128, i128) {
    (
        fp.x0 as i128 * QUANTUM,
        fp.y0 as i128 * QUANTUM,
        fp.x1 as i128 * QUANTUM,
        fp.y1 as i128 * QUANTUM,
    )
}

/// Exact slab/wall elevations for a reservation footprint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct VerticalBounds {
    pub floor_min: i128,
    pub floor_max: i128,
    pub wall_min: i128,
    pub wall_max: i128,
    pub ceiling_min: i128,
    pub ceiling_max: i128,
}

pub(crate) fn footprint_vertical_bounds(fp: &Footprint3D) -> Result<VerticalBounds, RichnessError> {
    use super::footprint::{ROOM_HEIGHT, UPPER_FLOOR_Z};

    let lower = fp.occupies_lower;
    let upper = fp.occupies_upper;
    let (base, top) = match (lower, upper) {
        (true, false) => (0, ROOM_HEIGHT as i128),
        (false, true) => (UPPER_FLOOR_Z as i128, (UPPER_FLOOR_Z + ROOM_HEIGHT) as i128),
        (true, true) => (0, (UPPER_FLOOR_Z + ROOM_HEIGHT) as i128),
        (false, false) => {
            return Err(geometry_error(
                RichnessErrorCode::SemanticInfeasible,
                "vertical_bounds",
                "room footprint occupies no vertical layer",
            ));
        }
    };
    Ok(VerticalBounds {
        floor_min: base,
        floor_max: base + QUANTUM,
        wall_min: base + QUANTUM,
        wall_max: top - QUANTUM,
        ceiling_min: top - QUANTUM,
        ceiling_max: top,
    })
}

/// Convert grid X coordinate to Quake i128.
#[inline]
pub(crate) fn grid_to_quake_x(gx: u32) -> i128 {
    gx as i128 * QUANTUM
}

/// Convert grid Y coordinate to Quake i128.
#[inline]
pub(crate) fn grid_to_quake_y(gy: u32) -> i128 {
    gy as i128 * QUANTUM
}

/// Convert from cell coord to quake i128 origin.
#[inline]
pub(crate) fn cell_to_quake_origin(cell: super::footprint::CellCoord) -> (i128, i128, i128) {
    let z = if cell.layer == 0 {
        FLOOR_Z_MIN
    } else {
        crate::enhanced_v3::richness::footprint::UPPER_FLOOR_Z as i128
    };
    (cell.x as i128 * QUANTUM, cell.y as i128 * QUANTUM, z)
}

// ── Validation helpers ────────────────────────────────────────────────────

/// Validate that a ConvexBrush has positive volume.
pub(crate) fn validate_positive_volume(brush: &ConvexBrush) -> Result<(), RichnessError> {
    let vol = brush.volume();
    if vol <= geometry::Rational::ZERO {
        return Err(geometry_error(
            RichnessErrorCode::ValueOutOfRange,
            "positive_volume",
            "brush has zero or negative volume",
        ));
    }
    Ok(())
}

/// Validate that all face normals are approved (cardinal XY/Z or 45-degree XY diagonal).
pub(crate) fn validate_approved_normals(brush: &ConvexBrush) -> Result<(), RichnessError> {
    for face in &brush.faces {
        let cls = crate::enhanced_v3::config::classify_normal(
            face.plane.nx,
            face.plane.ny,
            face.plane.nz,
        );
        if !cls.is_approved() {
            return Err(geometry_error(
                RichnessErrorCode::ValueOutOfRange,
                "approved_normals",
                format!(
                    "unapproved normal ({}, {}, {}) in brush face {}",
                    face.plane.nx, face.plane.ny, face.plane.nz, face.role
                ),
            ));
        }
    }
    Ok(())
}

/// Validate that all face plane d values are grid-aligned to the quantum.
pub(crate) fn validate_grid_alignment(brush: &ConvexBrush) -> Result<(), RichnessError> {
    for face in &brush.faces {
        if face.plane.d.rem_euclid(QUANTUM) != 0 {
            return Err(geometry_error(
                RichnessErrorCode::NotQuantumAligned,
                "grid_alignment",
                format!(
                    "plane d={} not quantum-aligned (quantum={})",
                    face.plane.d, QUANTUM
                ),
            ));
        }
    }
    Ok(())
}

/// Full validation: positive volume + approved normals + grid alignment.
pub(crate) fn validate_brush(brush: &ConvexBrush) -> Result<(), RichnessError> {
    validate_positive_volume(brush)?;
    validate_approved_normals(brush)?;
    validate_grid_alignment(brush)?;
    Ok(())
}

/// Check that a brush is fully contained within a footprint.
pub(crate) fn validate_containment(
    brush: &ConvexBrush,
    fp: &Footprint3D,
) -> Result<(), RichnessError> {
    let ((min_x, min_y, min_z), (max_x, max_y, max_z)) = brush
        .aabb()
        .map_err(|e| geometry_error(RichnessErrorCode::ValueOutOfRange, "aabb", format!("{e}")))?;

    let qx0 = fp.x0 as i128 * QUANTUM;
    let qy0 = fp.y0 as i128 * QUANTUM;
    let qx1 = fp.x1 as i128 * QUANTUM;
    let qy1 = fp.y1 as i128 * QUANTUM;
    let vertical = footprint_vertical_bounds(fp)?;

    if min_x < qx0
        || max_x > qx1
        || min_y < qy0
        || max_y > qy1
        || min_z < vertical.floor_min
        || max_z > vertical.ceiling_max
    {
        return Err(geometry_error(
            RichnessErrorCode::ValueOutOfRange,
            "containment",
            format!(
                "brush ({},{},{})-({},{},{}) not contained in footprint ({},{})-({},{}) z=[{},{}]",
                min_x,
                min_y,
                min_z,
                max_x,
                max_y,
                max_z,
                qx0,
                qy0,
                qx1,
                qy1,
                vertical.floor_min,
                vertical.ceiling_max
            ),
        ));
    }
    Ok(())
}

fn geometry_error(
    code: RichnessErrorCode,
    path: &str,
    context: impl Into<String>,
) -> RichnessError {
    RichnessError::new(
        code,
        0,
        "?",
        "?",
        "?",
        "?",
        "?",
        "?",
        "?",
        path,
        RichnessErrorCategory::PlacementTopologyExhaustion,
        context,
    )
}

// ── Slab builders ──────────────────────────────────────────────────────────

/// Build a floor slab covering the full footprint.
///
/// Spans x0..x1, y0..y1, z=0..16. Owns the full partition beneath walls.
pub(crate) fn make_floor_slab(fp: &Footprint3D) -> Result<ConvexBrush, RichnessError> {
    let (qx0, qy0, qx1, qy1) = footprint_quake_bounds(fp);
    let vertical = footprint_vertical_bounds(fp)?;
    let brush = ConvexBrush::make_box(
        (qx0, qx1),
        (qy0, qy1),
        (vertical.floor_min, vertical.floor_max),
    )
    .map_err(|e| {
        geometry_error(
            RichnessErrorCode::ValueOutOfRange,
            "floor_slab",
            format!("{e}"),
        )
    })?;
    validate_brush(&brush)?;
    Ok(brush)
}

/// Build a ceiling slab covering the full footprint.
///
/// Spans x0..x1, y0..y1, z=160..176. Owns the full partition above walls.
pub(crate) fn make_ceiling_slab(fp: &Footprint3D) -> Result<ConvexBrush, RichnessError> {
    let (qx0, qy0, qx1, qy1) = footprint_quake_bounds(fp);
    let vertical = footprint_vertical_bounds(fp)?;
    let brush = ConvexBrush::make_box(
        (qx0, qx1),
        (qy0, qy1),
        (vertical.ceiling_min, vertical.ceiling_max),
    )
    .map_err(|e| {
        geometry_error(
            RichnessErrorCode::ValueOutOfRange,
            "ceiling_slab",
            format!("{e}"),
        )
    })?;
    validate_brush(&brush)?;
    Ok(brush)
}

// ── Cardinal wall builders ────────────────────────────────────────────────

/// Build a north wall (y0..y0+16, full x span, z=16..160).
pub(crate) fn make_north_wall(fp: &Footprint3D) -> Result<ConvexBrush, RichnessError> {
    let (qx0, qy0, qx1, _qy1) = footprint_quake_bounds(fp);
    let vertical = footprint_vertical_bounds(fp)?;
    let brush = ConvexBrush::make_box(
        (qx0, qx1),
        (qy0, qy0 + WALL_THICKNESS),
        (vertical.wall_min, vertical.wall_max),
    )
    .map_err(|e| {
        geometry_error(
            RichnessErrorCode::ValueOutOfRange,
            "north_wall",
            format!("{e}"),
        )
    })?;
    validate_brush(&brush)?;
    Ok(brush)
}

/// Build a south wall (y1-16..y1, full x span, z=16..160).
pub(crate) fn make_south_wall(fp: &Footprint3D) -> Result<ConvexBrush, RichnessError> {
    let (qx0, _qy0, qx1, qy1) = footprint_quake_bounds(fp);
    let vertical = footprint_vertical_bounds(fp)?;
    let brush = ConvexBrush::make_box(
        (qx0, qx1),
        (qy1 - WALL_THICKNESS, qy1),
        (vertical.wall_min, vertical.wall_max),
    )
    .map_err(|e| {
        geometry_error(
            RichnessErrorCode::ValueOutOfRange,
            "south_wall",
            format!("{e}"),
        )
    })?;
    validate_brush(&brush)?;
    Ok(brush)
}

/// Build a west wall (x0..x0+16, y0+16..y1-16 span, z=16..160).
///
/// The y span is shortened to avoid corner overlap with N/S walls.
pub(crate) fn make_west_wall(fp: &Footprint3D) -> Result<ConvexBrush, RichnessError> {
    let (qx0, qy0, _qx1, qy1) = footprint_quake_bounds(fp);
    let vertical = footprint_vertical_bounds(fp)?;
    let brush = ConvexBrush::make_box(
        (qx0, qx0 + WALL_THICKNESS),
        (qy0 + WALL_THICKNESS, qy1 - WALL_THICKNESS),
        (vertical.wall_min, vertical.wall_max),
    )
    .map_err(|e| {
        geometry_error(
            RichnessErrorCode::ValueOutOfRange,
            "west_wall",
            format!("{e}"),
        )
    })?;
    validate_brush(&brush)?;
    Ok(brush)
}

/// Build an east wall (x1-16..x1, y0+16..y1-16 span, z=16..160).
///
/// The y span is shortened to avoid corner overlap with N/S walls.
pub(crate) fn make_east_wall(fp: &Footprint3D) -> Result<ConvexBrush, RichnessError> {
    let (_qx0, qy0, qx1, qy1) = footprint_quake_bounds(fp);
    let vertical = footprint_vertical_bounds(fp)?;
    let brush = ConvexBrush::make_box(
        (qx1 - WALL_THICKNESS, qx1),
        (qy0 + WALL_THICKNESS, qy1 - WALL_THICKNESS),
        (vertical.wall_min, vertical.wall_max),
    )
    .map_err(|e| {
        geometry_error(
            RichnessErrorCode::ValueOutOfRange,
            "east_wall",
            format!("{e}"),
        )
    })?;
    validate_brush(&brush)?;
    Ok(brush)
}

// ── Diagonal wall builders ────────────────────────────────────────────────
//
// Diagonal walls are constructed via the baseline geometry::make_diagonal_wall
// function called from composition.rs. No separate constructors needed here.

// ── Column / interior mass builders ────────────────────────────────────────

/// Build an interior column at the given grid-aligned minimum corner.
///
/// Column is 16×16 (one quantum), from floor top to ceiling bottom.
pub(crate) fn make_column(x0: i128, y0: i128) -> Result<ConvexBrush, RichnessError> {
    let brush = ConvexBrush::make_box(
        (x0, x0 + QUANTUM),
        (y0, y0 + QUANTUM),
        (WALL_Z_MIN, WALL_Z_MAX),
    )
    .map_err(|e| geometry_error(RichnessErrorCode::ValueOutOfRange, "column", format!("{e}")))?;
    validate_brush(&brush)?;
    Ok(brush)
}

/// Build a 32×32 pillar at the given grid-aligned minimum corner.
pub(crate) fn make_pillar(x0: i128, y0: i128) -> Result<ConvexBrush, RichnessError> {
    let w = QUANTUM * 2; // 32 units
    let brush = ConvexBrush::make_box((x0, x0 + w), (y0, y0 + w), (WALL_Z_MIN, WALL_Z_MAX))
        .map_err(|e| {
            geometry_error(RichnessErrorCode::ValueOutOfRange, "pillar", format!("{e}"))
        })?;
    validate_brush(&brush)?;
    Ok(brush)
}

// ── Two-point intersection check ───────────────────────────────────────────

/// Check whether two ConvexBrushes have positive-volume overlap.
///
/// Uses the baseline's exact intersection test via half-space enumeration.
pub(crate) fn brushes_overlap(a: &ConvexBrush, b: &ConvexBrush) -> Result<bool, RichnessError> {
    let mut all_faces: Vec<BrushFace> = Vec::with_capacity(a.faces.len() + b.faces.len());
    for face in &a.faces {
        all_faces.push(face.clone());
    }
    for face in &b.faces {
        all_faces.push(face.clone());
    }

    if all_faces.len() < 4 {
        return Ok(false);
    }

    let vertices = geometry::half_space_vertices(&all_faces).map_err(|e| {
        geometry_error(
            RichnessErrorCode::ValueOutOfRange,
            "overlap_verts",
            format!("{e}"),
        )
    })?;

    if vertices.len() < 4 {
        return Ok(false);
    }

    // Check for positive volume via tetrahedron test
    // Use a simple test: pick the first point as reference, look for 3 more
    // that form a non-degenerate tetrahedron.
    let p0 = vertices[0];
    for i in 1..vertices.len() {
        let v1 = p0.checked_sub(vertices[i]).map_err(|e| {
            geometry_error(
                RichnessErrorCode::ValueOutOfRange,
                "overlap_v1",
                format!("{e}"),
            )
        })?;
        for j in (i + 1)..vertices.len() {
            let v2 = p0.checked_sub(vertices[j]).map_err(|e| {
                geometry_error(
                    RichnessErrorCode::ValueOutOfRange,
                    "overlap_v2",
                    format!("{e}"),
                )
            })?;
            for k in (j + 1)..vertices.len() {
                let v3 = p0.checked_sub(vertices[k]).map_err(|e| {
                    geometry_error(
                        RichnessErrorCode::ValueOutOfRange,
                        "overlap_v3",
                        format!("{e}"),
                    )
                })?;
                // Scalar triple product: v1 · (v2 × v3)
                let cx =
                    v1.y.checked_mul(v3.z)
                        .map_err(|e| {
                            geometry_error(
                                RichnessErrorCode::ValueOutOfRange,
                                "overlap_cx",
                                format!("{e}"),
                            )
                        })?
                        .checked_sub(v1.z.checked_mul(v3.y).map_err(|e| {
                            geometry_error(
                                RichnessErrorCode::ValueOutOfRange,
                                "overlap_cx_sub",
                                format!("{e}"),
                            )
                        })?)
                        .map_err(|e| {
                            geometry_error(
                                RichnessErrorCode::ValueOutOfRange,
                                "overlap_cx_sub2",
                                format!("{e}"),
                            )
                        })?;
                let cy =
                    v1.z.checked_mul(v3.x)
                        .map_err(|e| {
                            geometry_error(
                                RichnessErrorCode::ValueOutOfRange,
                                "overlap_cy",
                                format!("{e}"),
                            )
                        })?
                        .checked_sub(v1.x.checked_mul(v3.z).map_err(|e| {
                            geometry_error(
                                RichnessErrorCode::ValueOutOfRange,
                                "overlap_cy_sub",
                                format!("{e}"),
                            )
                        })?)
                        .map_err(|e| {
                            geometry_error(
                                RichnessErrorCode::ValueOutOfRange,
                                "overlap_cy_sub2",
                                format!("{e}"),
                            )
                        })?;
                let cz =
                    v1.x.checked_mul(v3.y)
                        .map_err(|e| {
                            geometry_error(
                                RichnessErrorCode::ValueOutOfRange,
                                "overlap_cz",
                                format!("{e}"),
                            )
                        })?
                        .checked_sub(v1.y.checked_mul(v3.x).map_err(|e| {
                            geometry_error(
                                RichnessErrorCode::ValueOutOfRange,
                                "overlap_cz_sub",
                                format!("{e}"),
                            )
                        })?)
                        .map_err(|e| {
                            geometry_error(
                                RichnessErrorCode::ValueOutOfRange,
                                "overlap_cz_sub2",
                                format!("{e}"),
                            )
                        })?;
                let det =
                    v2.x.checked_mul(cx)
                        .map_err(|e| {
                            geometry_error(
                                RichnessErrorCode::ValueOutOfRange,
                                "overlap_det_x",
                                format!("{e}"),
                            )
                        })?
                        .checked_add(v2.y.checked_mul(cy).map_err(|e| {
                            geometry_error(
                                RichnessErrorCode::ValueOutOfRange,
                                "overlap_det_y",
                                format!("{e}"),
                            )
                        })?)
                        .map_err(|e| {
                            geometry_error(
                                RichnessErrorCode::ValueOutOfRange,
                                "overlap_det_xy",
                                format!("{e}"),
                            )
                        })?
                        .checked_add(v2.z.checked_mul(cz).map_err(|e| {
                            geometry_error(
                                RichnessErrorCode::ValueOutOfRange,
                                "overlap_det_z",
                                format!("{e}"),
                            )
                        })?)
                        .map_err(|e| {
                            geometry_error(
                                RichnessErrorCode::ValueOutOfRange,
                                "overlap_det_xyz",
                                format!("{e}"),
                            )
                        })?;
                if det != geometry::Rational::ZERO {
                    return Ok(true);
                }
            }
        }
    }
    Ok(false)
}

// ── Portal / lintel / surround / liner / pilaster / recess / buttress ────

/// Portal post: vertical framing column.
///
/// A post is a rectangular column, typically 16×16 (one quantum) in plan,
/// spanning from floor-top to ceiling-bottom.
pub(crate) fn make_portal_post(
    x0: i128,
    y0: i128,
    x1: i128,
    y1: i128,
) -> Result<ConvexBrush, RichnessError> {
    let brush =
        ConvexBrush::make_box((x0, x1), (y0, y1), (WALL_Z_MIN, WALL_Z_MAX)).map_err(|e| {
            geometry_error(
                RichnessErrorCode::ValueOutOfRange,
                "portal_post",
                format!("{e}"),
            )
        })?;
    validate_brush(&brush)?;
    Ok(brush)
}

/// Portal lintel: horizontal beam above the throat.
///
/// The lintel spans the full width of the throat plus posts, from post-top to
/// ceiling-bottom. Thickness is 16u.
pub(crate) fn make_portal_lintel(
    x0: i128,
    y0: i128,
    x1: i128,
    y1: i128,
) -> Result<ConvexBrush, RichnessError> {
    let lintel_z0 = WALL_Z_MIN + 80; // throat height = 80, starting from wall bottom (16)
    let lintel_z1 = lintel_z0 + WALL_THICKNESS; // 16u thick
    let brush = ConvexBrush::make_box((x0, x1), (y0, y1), (lintel_z0, lintel_z1)).map_err(|e| {
        geometry_error(
            RichnessErrorCode::ValueOutOfRange,
            "portal_lintel",
            format!("{e}"),
        )
    })?;
    validate_brush(&brush)?;
    Ok(brush)
}

/// Portal surround: decorative frame element around the throat.
///
/// A surround is a thin (8u or 16u) rectangular frame element placed adjacent
/// to the throat opening.
pub(crate) fn make_surround_frame(
    x0: i128,
    y0: i128,
    x1: i128,
    y1: i128,
    z0: i128,
    z1: i128,
) -> Result<ConvexBrush, RichnessError> {
    let brush = ConvexBrush::make_box((x0, x1), (y0, y1), (z0, z1)).map_err(|e| {
        geometry_error(
            RichnessErrorCode::ValueOutOfRange,
            "surround_frame",
            format!("{e}"),
        )
    })?;
    validate_brush(&brush)?;
    Ok(brush)
}

/// Stepped surround layer: a trapezoidal or stepped rectangular frame.
///
/// For Egyptian stepped surround, each layer is a rectangular course that
/// steps back from the throat.
pub(crate) fn make_stepped_course(
    x0: i128,
    y0: i128,
    x1: i128,
    y1: i128,
    z0: i128,
    z1: i128,
) -> Result<ConvexBrush, RichnessError> {
    let brush = ConvexBrush::make_box((x0, x1), (y0, y1), (z0, z1)).map_err(|e| {
        geometry_error(
            RichnessErrorCode::ValueOutOfRange,
            "stepped_course",
            format!("{e}"),
        )
    })?;
    validate_brush(&brush)?;
    Ok(brush)
}

/// Wall liner: an inward mass course applied to the interior face of a wall.
///
/// The liner is a rectangular slab of thickness t, applied to the wall interior
/// face, spanning from floor-top to ceiling-bottom.
pub(crate) fn make_liner(
    x0: i128,
    y0: i128,
    x1: i128,
    y1: i128,
    z_min: i128,
    z_max: i128,
) -> Result<ConvexBrush, RichnessError> {
    let brush = ConvexBrush::make_box((x0, x1), (y0, y1), (z_min, z_max))
        .map_err(|e| geometry_error(RichnessErrorCode::ValueOutOfRange, "liner", format!("{e}")))?;
    validate_brush(&brush)?;
    Ok(brush)
}

/// Pilaster: an engaged rectangular column attached to a wall face.
///
/// Typically 16×16 or 16×32 in plan, from floor-top to ceiling-bottom.
pub(crate) fn make_pilaster(
    x0: i128,
    y0: i128,
    x1: i128,
    y1: i128,
) -> Result<ConvexBrush, RichnessError> {
    let brush =
        ConvexBrush::make_box((x0, x1), (y0, y1), (WALL_Z_MIN, WALL_Z_MAX)).map_err(|e| {
            geometry_error(
                RichnessErrorCode::ValueOutOfRange,
                "pilaster",
                format!("{e}"),
            )
        })?;
    validate_brush(&brush)?;
    Ok(brush)
}

/// Wall recess: a carved-out negative volume in a wall.
///
/// A recess is NOT an additive brush — it is an omission from the wall brush.
/// This function exists to compute the recess bounds for opening records.
pub(crate) fn make_recess_volume(
    x0: i128,
    y0: i128,
    x1: i128,
    y1: i128,
    z0: i128,
    z1: i128,
) -> Result<ConvexBrush, RichnessError> {
    let brush = ConvexBrush::make_box((x0, x1), (y0, y1), (z0, z1)).map_err(|e| {
        geometry_error(
            RichnessErrorCode::ValueOutOfRange,
            "recess_volume",
            format!("{e}"),
        )
    })?;
    validate_brush(&brush)?;
    Ok(brush)
}

/// Buttress: an external mass attached to the outside face of a wall.
///
/// Typically 16×16 or 32×16 in plan, from floor-top to ceiling-bottom.
pub(crate) fn make_buttress(
    x0: i128,
    y0: i128,
    x1: i128,
    y1: i128,
) -> Result<ConvexBrush, RichnessError> {
    let brush =
        ConvexBrush::make_box((x0, x1), (y0, y1), (WALL_Z_MIN, WALL_Z_MAX)).map_err(|e| {
            geometry_error(
                RichnessErrorCode::ValueOutOfRange,
                "buttress",
                format!("{e}"),
            )
        })?;
    validate_brush(&brush)?;
    Ok(brush)
}

/// Sill: a low wall or ledge at the bottom of an opening.
///
/// Sills are 48-64u tall for overlooks, placed at floor-top.
pub(crate) fn make_sill(
    x0: i128,
    y0: i128,
    x1: i128,
    y1: i128,
    sill_height: i128,
) -> Result<ConvexBrush, RichnessError> {
    if !(48..=64).contains(&sill_height) {
        return Err(geometry_error(
            RichnessErrorCode::ValueOutOfRange,
            "sill",
            format!("sill height {sill_height} is outside 48..=64"),
        ));
    }
    let brush = ConvexBrush::make_box((x0, x1), (y0, y1), (WALL_Z_MIN, WALL_Z_MIN + sill_height))
        .map_err(|e| {
        geometry_error(RichnessErrorCode::ValueOutOfRange, "sill", format!("{e}"))
    })?;
    validate_brush(&brush)?;
    Ok(brush)
}

/// Partial wall: a wall segment that does not span the full room dimension.
///
/// Used for bent approaches and offset shafts.
pub(crate) fn make_partial_wall(
    x0: i128,
    y0: i128,
    x1: i128,
    y1: i128,
    z_min: i128,
    z_max: i128,
) -> Result<ConvexBrush, RichnessError> {
    let brush = ConvexBrush::make_box((x0, x1), (y0, y1), (z_min, z_max)).map_err(|e| {
        geometry_error(
            RichnessErrorCode::ValueOutOfRange,
            "partial_wall",
            format!("{e}"),
        )
    })?;
    validate_brush(&brush)?;
    Ok(brush)
}

// ── Contact detection ─────────────────────────────────────────────────────

/// Exact positive-area face contact shared by two convex brushes.
#[derive(Debug, Clone)]
pub(crate) struct ExactFaceContact {
    pub plane: CanonicalPlane,
    pub vertices: Vec<Point3>,
    pub area_squared: Rational,
}

fn opposite_oriented(a: &CanonicalPlane, b: &CanonicalPlane) -> bool {
    a.nx == -b.nx && a.ny == -b.ny && a.nz == -b.nz && a.d == -b.d
}

/// Find an exact shared face. AABB intersection is only a broad phase; the
/// accepted contact must be bounded by coincident, oppositely oriented brush
/// planes and have positive exact polygon area.
pub(crate) fn exact_face_contact(a: &ConvexBrush, b: &ConvexBrush) -> Option<ExactFaceContact> {
    let ((amin_x, amin_y, amin_z), (amax_x, amax_y, amax_z)) = a.aabb().ok()?;
    let ((bmin_x, bmin_y, bmin_z), (bmax_x, bmax_y, bmax_z)) = b.aabb().ok()?;
    if amin_x > bmax_x
        || bmin_x > amax_x
        || amin_y > bmax_y
        || bmin_y > amax_y
        || amin_z > bmax_z
        || bmin_z > amax_z
    {
        return None;
    }

    let mut faces = a.faces.clone();
    faces.extend(b.faces.iter().cloned());
    let intersection = geometry::half_space_vertices(&faces).ok()?;

    for a_face in &a.faces {
        for b_face in &b.faces {
            if !opposite_oriented(&a_face.plane, &b_face.plane) {
                continue;
            }
            let vertices: Vec<_> = intersection
                .iter()
                .copied()
                .filter(|vertex| {
                    a_face
                        .plane
                        .signed_distance_rational(vertex)
                        .is_ok_and(|distance| distance == Rational::ZERO)
                })
                .collect();
            if vertices.len() < 3 {
                continue;
            }
            let area_squared = geometry::polygon_area_squared(&vertices, &a_face.plane).ok()?;
            if area_squared > Rational::ZERO {
                return Some(ExactFaceContact {
                    plane: a_face.plane.clone(),
                    vertices,
                    area_squared,
                });
            }
        }
    }
    None
}

fn rational_floor(value: Rational) -> i128 {
    value.num.div_euclid(value.den)
}

fn rational_ceil(value: Rational) -> Option<i128> {
    let floor = rational_floor(value);
    floor.checked_add(i128::from(value.num.rem_euclid(value.den) != 0))
}

/// Integer envelope of the exact positive-area contact polygon.
pub(crate) fn face_contact_bounds(
    a: &ConvexBrush,
    b: &ConvexBrush,
) -> Option<(i128, i128, i128, i128, i128, i128)> {
    let contact = exact_face_contact(a, b)?;
    let min_x = contact.vertices.iter().map(|p| rational_floor(p.x)).min()?;
    let min_y = contact.vertices.iter().map(|p| rational_floor(p.y)).min()?;
    let min_z = contact.vertices.iter().map(|p| rational_floor(p.z)).min()?;
    let max_x = contact
        .vertices
        .iter()
        .map(|p| rational_ceil(p.x))
        .collect::<Option<Vec<_>>>()?
        .into_iter()
        .max()?;
    let max_y = contact
        .vertices
        .iter()
        .map(|p| rational_ceil(p.y))
        .collect::<Option<Vec<_>>>()?
        .into_iter()
        .max()?;
    let max_z = contact
        .vertices
        .iter()
        .map(|p| rational_ceil(p.z))
        .collect::<Option<Vec<_>>>()?
        .into_iter()
        .max()?;
    Some((min_x, min_y, min_z, max_x, max_y, max_z))
}

/// Whether two brushes share an exact positive-area face contact, including
/// approved XY-45 diagonal planes.
pub(crate) fn has_positive_area_contact(a: &ConvexBrush, b: &ConvexBrush) -> bool {
    exact_face_contact(a, b).is_some()
}

/// Derive the interface kind from two brush roles.
pub(crate) fn derive_interface_kind(
    a_role: super::assembly::BrushAssemblyRole,
    b_role: super::assembly::BrushAssemblyRole,
) -> Option<super::assembly::InterfaceKind> {
    use super::assembly::{BrushAssemblyRole, InterfaceKind};
    match (a_role, b_role) {
        // Cave complement members form a real, explicitly declared shell
        // interface with one another and with their containing host shell.
        // This preserves every positive-area contact while keeping cave
        // material roles distinct from ordinary room partitions.
        (r1, r2)
            if matches!(
                r1,
                BrushAssemblyRole::CaveFloor
                    | BrushAssemblyRole::CaveWall
                    | BrushAssemblyRole::CaveCeiling
            ) || matches!(
                r2,
                BrushAssemblyRole::CaveFloor
                    | BrushAssemblyRole::CaveWall
                    | BrushAssemblyRole::CaveCeiling
            ) =>
        {
            Some(InterfaceKind::CaveShellContact)
        }
        // Floor/wall
        (BrushAssemblyRole::FloorSlab, r) if r.is_wall() => Some(InterfaceKind::WallToFloor),
        (r, BrushAssemblyRole::FloorSlab) if r.is_wall() => Some(InterfaceKind::WallToFloor),
        // Ceiling/wall
        (BrushAssemblyRole::CeilingSlab, r) if r.is_wall() => Some(InterfaceKind::WallToCeiling),
        (r, BrushAssemblyRole::CeilingSlab) if r.is_wall() => Some(InterfaceKind::WallToCeiling),
        // Wall/wall corner (cardinal meets cardinal) — includes UpperShellWall
        (
            BrushAssemblyRole::NorthWall | BrushAssemblyRole::UpperShellWall,
            BrushAssemblyRole::WestWall
            | BrushAssemblyRole::EastWall
            | BrushAssemblyRole::UpperShellWall,
        ) => Some(InterfaceKind::WallToWallCorner),
        (
            BrushAssemblyRole::SouthWall,
            BrushAssemblyRole::WestWall
            | BrushAssemblyRole::EastWall
            | BrushAssemblyRole::UpperShellWall,
        ) => Some(InterfaceKind::WallToWallCorner),
        (
            BrushAssemblyRole::WestWall | BrushAssemblyRole::UpperShellWall,
            BrushAssemblyRole::NorthWall
            | BrushAssemblyRole::SouthWall
            | BrushAssemblyRole::UpperShellWall,
        ) => Some(InterfaceKind::WallToWallCorner),
        (
            BrushAssemblyRole::EastWall,
            BrushAssemblyRole::NorthWall
            | BrushAssemblyRole::SouthWall
            | BrushAssemblyRole::UpperShellWall,
        ) => Some(InterfaceKind::WallToWallCorner),
        // Split runs on one cardinal wall retain an explicit structural joint.
        (r1, r2) if r1 == r2 && is_cardinal_wall(r1) => Some(InterfaceKind::WallSegmentJoint),
        // Parallel opposite walls meet face-to-face at corridor junctions and
        // shared-wall chains (e.g. a corridor side wall abutting a room wall).
        (r1, r2) if is_cardinal_wall(r1) && is_cardinal_wall(r2) => {
            Some(InterfaceKind::WallToWallCorner)
        }
        // Adjacent room slab partitions meet without overlap.
        (r1, r2) if r1 == r2 && r1.is_slab() => Some(InterfaceKind::SlabRunJoint),
        // A diagonal joint is valid only between one diagonal wall and one
        // cardinal wall. Other diagonal contacts remain undeclared.
        (r1, r2) if is_cardinal_wall(r1) && names_diag(r2) => Some(InterfaceKind::WallToDiagJoint),
        (r1, r2) if names_diag(r1) && is_cardinal_wall(r2) => Some(InterfaceKind::WallToDiagJoint),
        // Two diagonal walls of the same chamfered corner may meet with
        // positive-area contact when the chamfer is small; that contact is a
        // declared structural corner joint, not an error.
        (r1, r2) if names_diag(r1) && names_diag(r2) => Some(InterfaceKind::WallToDiagCorner),
        // Column/floor
        (BrushAssemblyRole::InteriorColumn, BrushAssemblyRole::FloorSlab) => {
            Some(InterfaceKind::ColumnToFloor)
        }
        (BrushAssemblyRole::FloorSlab, BrushAssemblyRole::InteriorColumn) => {
            Some(InterfaceKind::ColumnToFloor)
        }
        // Mass/wall
        (BrushAssemblyRole::InteriorMass, r) if r.is_wall() => Some(InterfaceKind::MassToWall),
        (r, BrushAssemblyRole::InteriorMass) if r.is_wall() => Some(InterfaceKind::MassToWall),
        // Mass/floor
        (BrushAssemblyRole::InteriorMass, r)
            if matches!(
                r,
                BrushAssemblyRole::InteriorMass | BrushAssemblyRole::InteriorColumn
            ) =>
        {
            Some(InterfaceKind::PropToProp)
        }
        (BrushAssemblyRole::InteriorColumn, r)
            if matches!(
                r,
                BrushAssemblyRole::InteriorMass | BrushAssemblyRole::InteriorColumn
            ) =>
        {
            Some(InterfaceKind::PropToProp)
        }
        // Props attach to any structural role: a declared detail-to-structure
        // interface (shelf on wall, altar on floor, etc.).
        (BrushAssemblyRole::InteriorColumn | BrushAssemblyRole::InteriorMass, r) => {
            Some(InterfaceKind::MassToWall)
        }
        (r, BrushAssemblyRole::InteriorColumn | BrushAssemblyRole::InteriorMass) => {
            Some(InterfaceKind::MassToWall)
        }
        (BrushAssemblyRole::InteriorMass, BrushAssemblyRole::FloorSlab) => {
            Some(InterfaceKind::MassToFloor)
        }
        (BrushAssemblyRole::FloorSlab, BrushAssemblyRole::InteriorMass) => {
            Some(InterfaceKind::MassToFloor)
        }
        // Portal post/floor
        (BrushAssemblyRole::PortalPost, BrushAssemblyRole::FloorSlab) => {
            Some(InterfaceKind::PostToFloor)
        }
        (BrushAssemblyRole::FloorSlab, BrushAssemblyRole::PortalPost) => {
            Some(InterfaceKind::PostToFloor)
        }
        // Portal post/wall
        (BrushAssemblyRole::PortalPost, r) if r.is_wall() => Some(InterfaceKind::PostToWall),
        (r, BrushAssemblyRole::PortalPost) if r.is_wall() => Some(InterfaceKind::PostToWall),
        // Portal lintel/post
        (BrushAssemblyRole::PortalLintel, BrushAssemblyRole::PortalPost) => {
            Some(InterfaceKind::LintelToPost)
        }
        (BrushAssemblyRole::PortalPost, BrushAssemblyRole::PortalLintel) => {
            Some(InterfaceKind::LintelToPost)
        }
        // Portal lintel/wall
        (BrushAssemblyRole::PortalLintel, r) if r.is_wall() => Some(InterfaceKind::LintelToWall),
        (r, BrushAssemblyRole::PortalLintel) if r.is_wall() => Some(InterfaceKind::LintelToWall),
        // Pieces of one portal frame.
        (BrushAssemblyRole::PortalSurround, BrushAssemblyRole::PortalSurround) => {
            Some(InterfaceKind::PortalFrameJoint)
        }
        (r1, r2) if r1 != r2 && is_portal_frame(r1) && is_portal_frame(r2) => {
            Some(InterfaceKind::PortalFrameJoint)
        }
        // Portal surround/floor
        (BrushAssemblyRole::PortalSurround, BrushAssemblyRole::FloorSlab)
        | (BrushAssemblyRole::FloorSlab, BrushAssemblyRole::PortalSurround) => {
            Some(InterfaceKind::SurroundToFloor)
        }
        // Portal surround/wall
        (BrushAssemblyRole::PortalSurround, r) if r.is_wall() => {
            Some(InterfaceKind::SurroundToWall)
        }
        (r, BrushAssemblyRole::PortalSurround) if r.is_wall() => {
            Some(InterfaceKind::SurroundToWall)
        }
        // Liner/wall
        (BrushAssemblyRole::WallLiner, r) if r.is_wall() => Some(InterfaceKind::LinerToWall),
        (r, BrushAssemblyRole::WallLiner) if r.is_wall() => Some(InterfaceKind::LinerToWall),
        // Pilaster/wall
        (BrushAssemblyRole::Pilaster, r) if r.is_wall() => Some(InterfaceKind::PilasterToWall),
        (r, BrushAssemblyRole::Pilaster) if r.is_wall() => Some(InterfaceKind::PilasterToWall),
        // Buttress/wall
        (BrushAssemblyRole::Buttress, r) if r.is_wall() => Some(InterfaceKind::ButtressToWall),
        (r, BrushAssemblyRole::Buttress) if r.is_wall() => Some(InterfaceKind::ButtressToWall),
        // Sill/wall
        (BrushAssemblyRole::Sill, r) if r.is_wall() => Some(InterfaceKind::SillToWall),
        (r, BrushAssemblyRole::Sill) if r.is_wall() => Some(InterfaceKind::SillToWall),
        // Sill/floor
        (BrushAssemblyRole::Sill, BrushAssemblyRole::FloorSlab) => Some(InterfaceKind::SillToFloor),
        (BrushAssemblyRole::FloorSlab, BrushAssemblyRole::Sill) => Some(InterfaceKind::SillToFloor),
        // Bent approach/wall
        (BrushAssemblyRole::BentApproach, r) if r.is_wall() => {
            Some(InterfaceKind::BentApproachToWall)
        }
        (r, BrushAssemblyRole::BentApproach) if r.is_wall() => {
            Some(InterfaceKind::BentApproachToWall)
        }
        // Partial wall/wall
        (BrushAssemblyRole::PartialWall, r) if r.is_wall() => {
            Some(InterfaceKind::PartialWallToWall)
        }
        (r, BrushAssemblyRole::PartialWall) if r.is_wall() => {
            Some(InterfaceKind::PartialWallToWall)
        }
        // Offset shaft/floor
        (BrushAssemblyRole::OffsetShaft, BrushAssemblyRole::FloorSlab) => {
            Some(InterfaceKind::ShaftToFloor)
        }
        (BrushAssemblyRole::FloorSlab, BrushAssemblyRole::OffsetShaft) => {
            Some(InterfaceKind::ShaftToFloor)
        }
        // Offset shaft/wall
        (BrushAssemblyRole::OffsetShaft, r) if r.is_wall() => Some(InterfaceKind::ShaftToWall),
        (r, BrushAssemblyRole::OffsetShaft) if r.is_wall() => Some(InterfaceKind::ShaftToWall),
        // Balcony/wall
        (BrushAssemblyRole::BalconySlab, r) if r.is_wall() => Some(InterfaceKind::BalconyToWall),
        (r, BrushAssemblyRole::BalconySlab) if r.is_wall() => Some(InterfaceKind::BalconyToWall),
        // Guard rail/balcony
        (BrushAssemblyRole::GuardRail, BrushAssemblyRole::BalconySlab)
        | (BrushAssemblyRole::BalconySlab, BrushAssemblyRole::GuardRail) => {
            Some(InterfaceKind::RailToSlab)
        }
        // Guard rail/catwalk
        (BrushAssemblyRole::GuardRail, BrushAssemblyRole::CatwalkDeck)
        | (BrushAssemblyRole::CatwalkDeck, BrushAssemblyRole::GuardRail) => {
            Some(InterfaceKind::RailToSlab)
        }
        // Corbel/wall
        (BrushAssemblyRole::Corbel, r) if r.is_wall() => Some(InterfaceKind::CorbelToWall),
        (r, BrushAssemblyRole::Corbel) if r.is_wall() => Some(InterfaceKind::CorbelToWall),
        // Corbel/balcony
        (BrushAssemblyRole::Corbel, BrushAssemblyRole::BalconySlab)
        | (BrushAssemblyRole::BalconySlab, BrushAssemblyRole::Corbel) => {
            Some(InterfaceKind::CorbelToSlab)
        }
        // Pit perimeter/upper shell wall
        (BrushAssemblyRole::PitPerimeterSlab, BrushAssemblyRole::UpperShellWall)
        | (BrushAssemblyRole::UpperShellWall, BrushAssemblyRole::PitPerimeterSlab) => {
            Some(InterfaceKind::SlabToWall)
        }
        // Upper shell wall/floor
        (BrushAssemblyRole::UpperShellWall, BrushAssemblyRole::FloorSlab)
        | (BrushAssemblyRole::FloorSlab, BrushAssemblyRole::UpperShellWall) => {
            Some(InterfaceKind::WallToFloor)
        }
        // Upper shell wall/ceiling
        (BrushAssemblyRole::UpperShellWall, BrushAssemblyRole::CeilingSlab)
        | (BrushAssemblyRole::CeilingSlab, BrushAssemblyRole::UpperShellWall) => {
            Some(InterfaceKind::WallToCeiling)
        }
        // Grand-arena monoliths are structural floor-supported massing even
        // though ordinary monoliths are not classified as vertical roles.
        (BrushAssemblyRole::MonolithSolid, BrushAssemblyRole::FloorSlab)
        | (BrushAssemblyRole::FloorSlab, BrushAssemblyRole::MonolithSolid) => {
            Some(InterfaceKind::MassToFloor)
        }
        // Vertical architecture is assembled from exact, non-overlapping
        // gravity contacts. Keep those contacts explicit without pretending
        // every tread/landing/support combination is a baseline room joint.
        (r1, r2) if r1.is_vertical_architecture() || r2.is_vertical_architecture() => {
            Some(InterfaceKind::VerticalMemberContact)
        }
        _ => None,
    }
}

pub(crate) fn role_is_diag_wall(r: super::assembly::BrushAssemblyRole) -> bool {
    names_diag(r)
}

fn names_diag(r: super::assembly::BrushAssemblyRole) -> bool {
    use super::assembly::BrushAssemblyRole;
    matches!(
        r,
        BrushAssemblyRole::DiagNEWall
            | BrushAssemblyRole::DiagNWWall
            | BrushAssemblyRole::DiagSEWall
            | BrushAssemblyRole::DiagSWWall
    )
}

fn is_cardinal_wall(r: super::assembly::BrushAssemblyRole) -> bool {
    use super::assembly::BrushAssemblyRole;
    matches!(
        r,
        BrushAssemblyRole::NorthWall
            | BrushAssemblyRole::SouthWall
            | BrushAssemblyRole::EastWall
            | BrushAssemblyRole::WestWall
    )
}

fn is_portal_frame(r: super::assembly::BrushAssemblyRole) -> bool {
    use super::assembly::BrushAssemblyRole;
    matches!(
        r,
        BrushAssemblyRole::PortalPost
            | BrushAssemblyRole::PortalLintel
            | BrushAssemblyRole::PortalSurround
    )
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::super::footprint::Footprint3D;
    use super::*;

    fn make_test_fp() -> Footprint3D {
        Footprint3D::single_layer(0, 0, 256, 256, 0)
    }

    #[test]
    fn footprint_quake_bounds_conversion() {
        // Footprint3D::single_layer takes Quake coordinates
        let fp = Footprint3D::single_layer(0, 0, 256, 256, 0); // 16x16 grid cells
        let (x0, y0, x1, y1) = footprint_quake_bounds(&fp);
        assert_eq!(x0, 0);
        assert_eq!(y0, 0);
        assert_eq!(x1, 256);
        assert_eq!(y1, 256);
    }

    #[test]
    fn floor_slab_positive_volume() {
        let fp = make_test_fp();
        let slab = make_floor_slab(&fp).unwrap();
        assert!(slab.volume() > geometry::Rational::ZERO);
    }

    #[test]
    fn ceiling_slab_positive_volume() {
        let fp = make_test_fp();
        let slab = make_ceiling_slab(&fp).unwrap();
        assert!(slab.volume() > geometry::Rational::ZERO);
    }

    #[test]
    fn north_wall_positive_volume() {
        let fp = make_test_fp();
        let wall = make_north_wall(&fp).unwrap();
        assert!(wall.volume() > geometry::Rational::ZERO);
    }

    #[test]
    fn south_wall_positive_volume() {
        let fp = make_test_fp();
        let wall = make_south_wall(&fp).unwrap();
        assert!(wall.volume() > geometry::Rational::ZERO);
    }

    #[test]
    fn west_wall_positive_volume() {
        let fp = make_test_fp();
        let wall = make_west_wall(&fp).unwrap();
        assert!(wall.volume() > geometry::Rational::ZERO);
    }

    #[test]
    fn east_wall_positive_volume() {
        let fp = make_test_fp();
        let wall = make_east_wall(&fp).unwrap();
        assert!(wall.volume() > geometry::Rational::ZERO);
    }

    #[test]
    fn walls_are_disjoint() {
        let fp = make_test_fp();
        let north = make_north_wall(&fp).unwrap();
        let south = make_south_wall(&fp).unwrap();
        let west = make_west_wall(&fp).unwrap();
        let east = make_east_wall(&fp).unwrap();

        let walls = [&north, &south, &west, &east];
        for i in 0..walls.len() {
            for j in (i + 1)..walls.len() {
                assert!(
                    !brushes_overlap(walls[i], walls[j]).unwrap(),
                    "walls {} and {} overlap",
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn walls_do_not_overlap_floor_and_ceiling() {
        let fp = make_test_fp();
        let floor = make_floor_slab(&fp).unwrap();
        let ceiling = make_ceiling_slab(&fp).unwrap();
        let north = make_north_wall(&fp).unwrap();

        // Walls sit ON floor, UNDER ceiling — should be non-overlapping
        // (contact is zero-volume, not positive-volume overlap)
        assert!(!brushes_overlap(&north, &floor).unwrap());
        assert!(!brushes_overlap(&north, &ceiling).unwrap());
    }

    #[test]
    fn column_positive_volume() {
        let col = make_column(128, 128).unwrap();
        assert!(col.volume() > geometry::Rational::ZERO);
    }

    #[test]
    fn pillar_positive_volume() {
        let pil = make_pillar(128, 128).unwrap();
        assert!(pil.volume() > geometry::Rational::ZERO);
    }

    #[test]
    fn chamfered_walls_have_approved_normals() {
        // Test that make_diagonal_wall from baseline produces approved normals
        let diag = geometry::make_diagonal_wall((0, 320), (0, 320), 16, 160, -1, -1, 64).unwrap();
        validate_approved_normals(&diag).unwrap();
    }

    #[test]
    fn validate_brush_rejects_unapproved_normals() {
        // Create a brush with a diagonal normal that is NOT 45-degree
        let bad_plane = CanonicalPlane::new(2, 1, 0, 0); // Normal (2,1,0) is not 45-degree
        if let Ok(bad) = bad_plane {
            let face = BrushFace::new(bad);
            // This should fail at brush construction due to unapproved normal
            if let Ok(f) = face {
                let result = ConvexBrush::new(vec![f]);
                // Either construction fails or validation catches it
                if let Ok(brush) = result {
                    assert!(validate_approved_normals(&brush).is_err());
                }
            }
        }
    }
}
