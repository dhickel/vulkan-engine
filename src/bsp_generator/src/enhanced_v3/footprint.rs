//! Deterministic chamfered/octagonal footprint generation for Enhanced V3.
//!
//! All geometry uses cardinal and 45° edges only. Footprints are convex
//! lattice polygons validated against the V3 contract. Surface, edge, and
//! corner IDs are stable and never encode iteration position.

use super::config::{self, V3Config, CONSTRUCTION_QUANTUM};
use super::error::V3Error;
use super::ids::{CornerId, RoomId, SurfaceId, V3IdAllocator};
use super::rng::{self, V3Seed};

// ── Placement constants ────────────────────────────────────────────────────

/// Compatibility room-span bounds. Explorer overrides are resolved from
/// `V3Config` inside `build_footprints`.
#[cfg(test)]
const MIN_OUTER_SPAN: i32 = config::DEFAULT_ROOM_SPAN_MIN as i32;
#[cfg(test)]
const MAX_OUTER_SPAN: i32 = config::DEFAULT_ROOM_SPAN_MAX as i32;

/// Every non-empty subset of the four corners.  The constructor accepts all
/// patterns, including an all-corner octagon.
const CHAMFER_PATTERNS: &[&[(i32, i32)]] = &[
    &[(-1, -1)],
    &[(1, -1)],
    &[(1, 1)],
    &[(-1, 1)],
    &[(-1, -1), (1, -1)],
    &[(-1, -1), (1, 1)],
    &[(-1, -1), (-1, 1)],
    &[(1, -1), (1, 1)],
    &[(1, -1), (-1, 1)],
    &[(1, 1), (-1, 1)],
    &[(-1, -1), (1, -1), (1, 1)],
    &[(-1, -1), (1, -1), (-1, 1)],
    &[(-1, -1), (1, 1), (-1, 1)],
    &[(1, -1), (1, 1), (-1, 1)],
    &[(-1, -1), (1, -1), (1, 1), (-1, 1)],
];

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

        let vertices = chamfered_vertices(shell, chamfer_corners, chamfer_size)?;
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

/// Layout relationship between rooms.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FootprintLayout {
    pub primary: usize,
    pub secondary: usize,
    pub transition_lower: usize,
    pub transition_upper: usize,
}

/// Build deterministic footprints and room layout from config.
///
/// Uses the frozen v3 placement seed stream to generate a seeded,
/// deterministic, varied, non-overlapping set of room footprints
/// satisfying the preset room minimums. Every preset includes
/// genuinely chamfered footprints.
pub fn build_footprints(
    config: &V3Config,
    seed: V3Seed,
    alloc: &mut V3IdAllocator,
) -> Result<(Vec<Footprint>, FootprintLayout), V3Error> {
    config.validate()?;
    let extent = config.xy_extent as i32;
    let q = CONSTRUCTION_QUANTUM;
    let target = config.effective_rooms() as usize;
    let min_outer_span = config.effective_room_span_min() as i32;
    let max_outer_span = config.effective_room_span_max() as i32;
    let min_span_q = min_outer_span / q;
    let max_span_q = max_outer_span / q;
    let extent_q = extent / q;
    let lower_target = (target + 1) / 2;
    let upper_target = target - lower_target;
    let layer_target = lower_target.max(upper_target);
    let (columns, layer_rows) = placement_grid(layer_target);

    // The stair is a real 12×16-unit structure, not a graph-only edge. Keep
    // the projected room bands apart by a 256-unit structural lane: 64 units
    // are reserved for the lower approach before the exact 192-unit run. Slot
    // slack supplies the upper crest approach. Any odd remainder stays above
    // the upper band.
    const TRANSITION_LANE_Q: i32 = 16;
    let transition_lane_q = if config.effective_vertical_edges() > 0 {
        TRANSITION_LANE_Q
    } else {
        0
    };
    let room_band_q = (extent_q - transition_lane_q) / 2;
    let upper_band_origin_q = room_band_q + transition_lane_q;
    let slot_width_q = extent_q / columns;
    let slot_depth_q = room_band_q / layer_rows;

    // A full rectangular slot is reserved for each room, rather than only the
    // cells whose centers are inside a chamfered polygon. This deliberately
    // leaves no occupancy holes at clipped corners and makes overlap proof a
    // simple interval proof independent of a hash-table traversal.
    if slot_width_q < min_span_q + 1 || slot_depth_q < min_span_q + 1 {
        return Err(V3Error::InvalidFootprint {
            detail: format!(
                "extent {extent} cannot reserve {target} rooms with span {min_outer_span}..={max_outer_span} and a {}-unit transition lane",
                transition_lane_q * q
            ),
        });
    }

    // ── Pass 1: compute per-room dimensions from seed ──────────────
    struct RoomDim {
        room_index: usize,
        w_q: i32,
        d_q: i32,
        column: i32,
        row: i32,
        layer: u8,
        floor_z: i32,
        u3_chamfer_bits: u64,
    }
    let mut room_dims: Vec<RoomDim> = Vec::with_capacity(target);
    for room_index in 0..target {
        let layer = u8::from(room_index >= lower_target);
        let floor_z = if layer == 0 {
            config::LOWER_FLOOR_Z
        } else {
            config::UPPER_FLOOR_Z
        };
        let key = format!("room/{room_index:04}");
        let [u0, u1, _u2, u3] = seed
            .candidate_seed(rng::tags::PLACEMENT, key.as_bytes())
            .u64s();
        let layer_index = if layer == 0 {
            room_index
        } else {
            room_index - lower_target
        };
        let column = (layer_index % columns as usize) as i32;
        let row = (layer_index / columns as usize) as i32;
        // Horizontal slots retain two quanta of slack. The tight Rich/1024
        // vertical band retains one full quantum, which is sufficient for a
        // positive route gap while preserving the frozen 112-unit minimum.
        let max_width_q = (slot_width_q - 2).min(max_span_q);
        let max_depth_q = (slot_depth_q - 1).min(max_span_q);
        if max_width_q < min_span_q || max_depth_q < min_span_q {
            return Err(V3Error::InvalidFootprint {
                detail: format!(
                    "room {room_index} slot cannot fit configured span {min_outer_span}..={max_outer_span}"
                ),
            });
        }
        let w_q = if room_index == 0 {
            seeded_even_span(u0, min_span_q, max_width_q)
        } else {
            min_span_q + (u0 % (max_width_q - min_span_q + 1) as u64) as i32
        };
        let d_q = if room_index == 0 {
            (min_span_q + 1).min(max_depth_q)
        } else {
            min_span_q + (u1 % (max_depth_q - min_span_q + 1) as u64) as i32
        };
        room_dims.push(RoomDim {
            room_index,
            w_q,
            d_q,
            column,
            row,
            layer,
            floor_z,
            u3_chamfer_bits: u3,
        });
    }

    // ── Pass 2: reserve fixed slot origins ───────────────────────
    use std::collections::BTreeMap;
    let mut col_min_gap: BTreeMap<i32, i32> = BTreeMap::new();
    let mut row_min_gap: BTreeMap<i32, i32> = BTreeMap::new();
    for d in &room_dims {
        let x_gap = slot_width_q - d.w_q;
        let y_gap = slot_depth_q - d.d_q;
        col_min_gap
            .entry(d.column)
            .and_modify(|v| *v = (*v).min(x_gap))
            .or_insert(x_gap);
        row_min_gap
            .entry(d.row)
            .and_modify(|v| *v = (*v).min(y_gap))
            .or_insert(y_gap);
    }

    let mut col_offsets: BTreeMap<i32, i32> = BTreeMap::new();
    let mut row_offsets: BTreeMap<i32, i32> = BTreeMap::new();
    for (&col, _) in &col_min_gap {
        col_offsets.insert(col, 0);
    }
    for (&row, _) in &row_min_gap {
        row_offsets.insert(row, 0);
    }

    // ── Pass 3: place rooms with aligned offsets ──────────────────
    let mut footprints = Vec::with_capacity(target);
    for d in &room_dims {
        let column = d.column;
        let row = d.row;
        let w_q = d.w_q;
        let d_q = d.d_q;
        let u3 = d.u3_chamfer_bits;
        let layer = d.layer;
        let floor_z = d.floor_z;

        let x_offset_q = col_offsets[&column];
        let y_offset_q = row_offsets[&row];

        let x0 = (column * slot_width_q + x_offset_q) * q;
        let layer_origin_q = if layer == 0 { 0 } else { upper_band_origin_q };
        let y0 = (layer_origin_q + row * slot_depth_q + y_offset_q) * q;
        let shell = (x0, y0, x0 + w_q * q, y0 + d_q * q);

        // A diagonal shell needs a 32-unit diagonal-plane offset: its
        // perpendicular thickness is 32 / sqrt(2), safely above the frozen
        // 16-unit minimum.  A 16-unit chamfer is only 11.31 units thick and
        // is therefore never committed.  Retain rectangles where the room
        // cannot also retain a 64-unit cardinal portal edge after chamfering.
        let chamfer_size = 2 * q;
        let room_index = d.room_index;
        let vertices = if !config.chamfer || room_index % 3 == 0 {
            rect_vertices(shell)
        } else {
            // A 112-unit side still retains an 80-unit cardinal edge with one
            // 32-unit chamfer. Restrict tight slots to one-corner patterns so
            // every wall remains eligible for a real 64-unit portal.
            let patterns = if w_q < 8 || d_q < 8 {
                &CHAMFER_PATTERNS[..4]
            } else {
                CHAMFER_PATTERNS
            };
            let pattern = patterns[(u3 >> 16) as usize % patterns.len()];
            chamfered_vertices(shell, pattern, chamfer_size)?
        };
        validate_edges(&vertices)?;

        let room_id = alloc.next_room()?;
        let mut surface_ids = Vec::with_capacity(vertices.len());
        let mut corner_ids = Vec::with_capacity(vertices.len());
        for _ in 0..vertices.len() {
            surface_ids.push(alloc.next_surface()?);
            corner_ids.push(alloc.next_corner()?);
        }
        let footprint = Footprint {
            vertices,
            surface_ids,
            corner_ids,
            aabb: shell,
            room_id,
            layer,
            floor_z,
        };
        footprint.validate_convex()?;
        footprints.push(footprint);
    }

    let lower_count = lower_target;
    let upper_count = target - lower_target;

    // Slots reserve complete shell AABBs, so this exact positive-area test
    // also proves that chamfered corners cannot admit an occupancy hole.
    for (index, footprint) in footprints.iter().enumerate() {
        for other in &footprints[..index] {
            if aabbs_overlap(footprint.aabb, other.aabb) {
                return Err(V3Error::TopologyInvariant {
                    detail: format!(
                        "projected room overlap between {} and {}",
                        other.room_id, footprint.room_id
                    ),
                });
            }
        }
    }

    // ── Verify layer balance ──────────────────────────────────────
    let diff = if lower_count > upper_count {
        lower_count - upper_count
    } else {
        upper_count - lower_count
    };
    if diff > 1 {
        return Err(V3Error::TopologyInvariant {
            detail: format!(
                "layer membership not balanced: lower={lower_count}, upper={upper_count}, diff={diff}"
            ),
        });
    }

    // ── Verify every room is within bounds ────────────────────────
    for fp in &footprints {
        let (x0, y0, x1, y1) = fp.aabb;
        if x0 < 0 || y0 < 0 || x1 > extent || y1 > extent {
            return Err(V3Error::RoomOutOfBounds {
                room_id: fp.room_id.raw(),
                extent: extent as u32,
            });
        }
        let w = x1 - x0;
        let d = y1 - y0;
        if w < min_outer_span || w > max_outer_span || d < min_outer_span || d > max_outer_span {
            return Err(V3Error::InvalidFootprint {
                detail: format!(
                    "room {} span {w}×{d} outside [{min_outer_span}..{max_outer_span}]",
                    fp.room_id
                ),
            });
        }
    }

    // ── Build FootprintLayout for downstream compatibility ─────────
    let layout = build_layout_from_footprints(&footprints, lower_count, upper_count)?;

    Ok((footprints, layout))
}

// ── Polygon and reservation helpers ────────────────────────────────────────

/// Compute rectangular footprint vertices from shell.
fn rect_vertices(shell: (i32, i32, i32, i32)) -> Vec<(i32, i32)> {
    let (x0, y0, x1, y1) = shell;
    vec![(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
}

/// Compute a CCW chamfered polygon. The southwest corner's second vertex is
/// emitted last so its diagonal closes the loop; this preserves valid edges
/// for every corner combination, including the all-corner octagon.
fn chamfered_vertices(
    shell: (i32, i32, i32, i32),
    chamfer_corners: &[(i32, i32)],
    chamfer_size: i32,
) -> Result<Vec<(i32, i32)>, V3Error> {
    let (x0, y0, x1, y1) = shell;
    let width = x1 - x0;
    let depth = y1 - y0;
    if chamfer_size <= 0 || 2 * chamfer_size > width || 2 * chamfer_size > depth {
        return Err(V3Error::InvalidFootprint {
            detail: format!("chamfer {chamfer_size} too large for shell {width}×{depth}"),
        });
    }

    let sw = chamfer_corners.contains(&(-1, -1));
    let se = chamfer_corners.contains(&(1, -1));
    let ne = chamfer_corners.contains(&(1, 1));
    let nw = chamfer_corners.contains(&(-1, 1));
    let mut vertices = Vec::with_capacity(8);
    vertices.push(if sw {
        (x0 + chamfer_size, y0)
    } else {
        (x0, y0)
    });
    if se {
        vertices.extend([(x1 - chamfer_size, y0), (x1, y0 + chamfer_size)]);
    } else {
        vertices.push((x1, y0));
    }
    if ne {
        vertices.extend([(x1, y1 - chamfer_size), (x1 - chamfer_size, y1)]);
    } else {
        vertices.push((x1, y1));
    }
    if nw {
        vertices.extend([(x0 + chamfer_size, y1), (x0, y1 - chamfer_size)]);
    } else {
        vertices.push((x0, y1));
    }
    if sw {
        vertices.push((x0, y0 + chamfer_size));
    }
    Ok(vertices)
}

/// Pick an even quantum span in the configured range for the spawn host.
/// A one-value odd range remains valid and returns its sole value.
fn seeded_even_span(random: u64, minimum: i32, maximum: i32) -> i32 {
    let first = if minimum % 2 == 0 {
        minimum
    } else {
        minimum + 1
    };
    if first > maximum {
        return minimum;
    }
    let count = (maximum - first) / 2 + 1;
    first + 2 * (random % count as u64) as i32
}

/// Return a compact per-layer grid while leaving room for the stair band.
fn placement_grid(layer_target: usize) -> (i32, i32) {
    let rows = if layer_target <= 10 { 2usize } else { 3usize };
    let columns = layer_target.div_ceil(rows);
    (columns as i32, rows as i32)
}

fn aabbs_overlap(a: (i32, i32, i32, i32), b: (i32, i32, i32, i32)) -> bool {
    a.0 < b.2 && a.2 > b.0 && a.1 < b.3 && a.3 > b.1
}

// ── FootprintLayout construction ───────────────────────────────────────────

/// Build a `FootprintLayout` from the placed footprints for downstream
/// transitional compatibility.
///
/// Selects a large lower primary, another lower secondary, and valid
/// transition indices for compatibility with the Phase 02 topology
/// replacement.
fn build_layout_from_footprints(
    footprints: &[Footprint],
    lower_count: usize,
    upper_count: usize,
) -> Result<FootprintLayout, V3Error> {
    if lower_count < 2 {
        return Err(V3Error::TopologyInvariant {
            detail: format!("need at least 2 lower rooms for layout, got {lower_count}"),
        });
    }
    if upper_count == 0 {
        return Err(V3Error::TopologyInvariant {
            detail: "need at least 1 upper room for layout".into(),
        });
    }

    // Order by position, then stable ID. This gives the fixed Phase 01
    // topology shim an eastward lower-layer route and a non-zero northward
    // transition span without depending on random placement order.
    let mut lower_indices: Vec<usize> = footprints
        .iter()
        .enumerate()
        .filter(|(_, footprint)| footprint.layer == 0)
        .map(|(index, _)| index)
        .collect();
    lower_indices.sort_by_key(|&index| {
        let footprint = &footprints[index];
        (footprint.aabb.1, footprint.aabb.0, footprint.room_id)
    });
    let primary = lower_indices[0];
    let secondary = lower_indices
        .iter()
        .copied()
        .find(|&index| footprints[index].aabb.0 >= footprints[primary].aabb.2)
        .unwrap_or(lower_indices[1]);
    let transition_upper = footprints
        .iter()
        .enumerate()
        .filter(|(_, footprint)| {
            footprint.layer == 1 && footprint.aabb.1 > footprints[primary].aabb.3
        })
        .min_by_key(|(_, footprint)| (footprint.aabb.1, footprint.aabb.0, footprint.room_id))
        .map(|(index, _)| index)
        .ok_or_else(|| V3Error::TopologyInvariant {
            detail: "no upper room lies north of the transition host".into(),
        })?;

    Ok(FootprintLayout {
        primary,
        secondary,
        transition_lower: primary,
        transition_upper,
    })
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::config::V3Preset;
    use super::*;

    // ── Basic placement tests ──────────────────────────────────────

    #[test]
    fn sparse_placement_minimum_12_rooms() {
        for &extent in &[1024, 2048, 3072] {
            let config = V3Config::new(0, V3Preset::Sparse, extent).unwrap();
            let mut alloc = V3IdAllocator::new();
            let (footprints, _) = build_footprints(&config, V3Seed::new(0), &mut alloc).unwrap();
            assert_eq!(footprints.len(), 12, "Sparse at {extent}");
        }
    }

    #[test]
    fn moderate_placement_minimum_20_rooms() {
        for &extent in &[1024, 2048, 3072] {
            let config = V3Config::new(0, V3Preset::Moderate, extent).unwrap();
            let mut alloc = V3IdAllocator::new();
            let (footprints, _) = build_footprints(&config, V3Seed::new(0), &mut alloc).unwrap();
            assert_eq!(footprints.len(), 20, "Moderate at {extent}");
        }
    }

    #[test]
    fn rich_placement_minimum_28_rooms() {
        for &extent in &[1024, 2048, 3072] {
            let config = V3Config::new(0, V3Preset::Rich, extent).unwrap();
            let mut alloc = V3IdAllocator::new();
            let (footprints, _) = build_footprints(&config, V3Seed::new(0), &mut alloc).unwrap();
            assert_eq!(footprints.len(), 28, "Rich at {extent}");
        }
    }

    // ── Exact room count tests ─────────────────────────────────────

    #[test]
    fn sparse_exact_12_rooms() {
        let config = V3Config::new(0, V3Preset::Sparse, 2048).unwrap();
        let mut alloc = V3IdAllocator::new();
        let (footprints, _) = build_footprints(&config, V3Seed::new(0), &mut alloc).unwrap();
        assert_eq!(footprints.len(), 12);
    }

    #[test]
    fn moderate_exact_20_rooms() {
        let config = V3Config::new(0, V3Preset::Moderate, 2048).unwrap();
        let mut alloc = V3IdAllocator::new();
        let (footprints, _) = build_footprints(&config, V3Seed::new(0), &mut alloc).unwrap();
        assert_eq!(footprints.len(), 20);
    }

    #[test]
    fn rich_exact_28_rooms() {
        let config = V3Config::new(0, V3Preset::Rich, 3072).unwrap();
        let mut alloc = V3IdAllocator::new();
        let (footprints, _) = build_footprints(&config, V3Seed::new(0), &mut alloc).unwrap();
        assert_eq!(footprints.len(), 28);
    }

    // ── Span verification ──────────────────────────────────────────

    #[test]
    fn all_room_spans_within_range() {
        for (preset, extent) in &[
            (V3Preset::Sparse, 2048),
            (V3Preset::Moderate, 2048),
            (V3Preset::Rich, 3072),
        ] {
            let config = V3Config::new(0, *preset, *extent).unwrap();
            let mut alloc = V3IdAllocator::new();
            let (footprints, _) = build_footprints(&config, V3Seed::new(0), &mut alloc).unwrap();
            for fp in &footprints {
                let w = fp.aabb.2 - fp.aabb.0;
                let d = fp.aabb.3 - fp.aabb.1;
                assert!(
                    w >= MIN_OUTER_SPAN && w <= MAX_OUTER_SPAN,
                    "room {} width {w} outside [{MIN_OUTER_SPAN}..{MAX_OUTER_SPAN}]",
                    fp.room_id
                );
                assert!(
                    d >= MIN_OUTER_SPAN && d <= MAX_OUTER_SPAN,
                    "room {} depth {d} outside [{MIN_OUTER_SPAN}..{MAX_OUTER_SPAN}]",
                    fp.room_id
                );
            }
        }
    }

    // ── Bounds test ────────────────────────────────────────────────

    #[test]
    fn all_rooms_within_xy_extent() {
        for (preset, extent) in &[
            (V3Preset::Sparse, 1024),
            (V3Preset::Moderate, 2048),
            (V3Preset::Rich, 3072),
        ] {
            let config = V3Config::new(0, *preset, *extent).unwrap();
            let mut alloc = V3IdAllocator::new();
            let (footprints, _) = build_footprints(&config, V3Seed::new(0), &mut alloc).unwrap();
            let ext = *extent as i32;
            for fp in &footprints {
                assert!(fp.aabb.0 >= 0, "room {} x0={} < 0", fp.room_id, fp.aabb.0);
                assert!(fp.aabb.1 >= 0, "room {} y0={} < 0", fp.room_id, fp.aabb.1);
                assert!(
                    fp.aabb.2 <= ext,
                    "room {} x1={} > {ext}",
                    fp.room_id,
                    fp.aabb.2
                );
                assert!(
                    fp.aabb.3 <= ext,
                    "room {} y1={} > {ext}",
                    fp.room_id,
                    fp.aabb.3
                );
            }
        }
    }

    // ── Non-overlap test ───────────────────────────────────────────

    #[test]
    fn no_room_overlap() {
        for (preset, extent) in &[
            (V3Preset::Sparse, 1024),
            (V3Preset::Moderate, 2048),
            (V3Preset::Rich, 3072),
        ] {
            let config = V3Config::new(0, *preset, *extent).unwrap();
            let mut alloc = V3IdAllocator::new();
            let (footprints, _) = build_footprints(&config, V3Seed::new(0), &mut alloc).unwrap();

            for i in 0..footprints.len() {
                for j in (i + 1)..footprints.len() {
                    let a = &footprints[i];
                    let b = &footprints[j];

                    assert!(
                        !aabbs_overlap(a.aabb, b.aabb),
                        "overlap between room {} and room {} at {extent} preset {preset:?}",
                        a.room_id,
                        b.room_id
                    );
                }
            }
        }
    }

    // ── Layer balance test ─────────────────────────────────────────

    #[test]
    fn layers_balanced() {
        for (preset, extent) in &[
            (V3Preset::Sparse, 2048),
            (V3Preset::Moderate, 2048),
            (V3Preset::Rich, 3072),
        ] {
            let config = V3Config::new(0, *preset, *extent).unwrap();
            let mut alloc = V3IdAllocator::new();
            let (footprints, _) = build_footprints(&config, V3Seed::new(0), &mut alloc).unwrap();

            let lower: Vec<_> = footprints.iter().filter(|fp| fp.layer == 0).collect();
            let upper: Vec<_> = footprints.iter().filter(|fp| fp.layer == 1).collect();
            let diff = if lower.len() > upper.len() {
                lower.len() - upper.len()
            } else {
                upper.len() - lower.len()
            };
            assert!(
                diff <= 1,
                "preset {preset:?} at {extent}: lower={}, upper={}, diff={diff}",
                lower.len(),
                upper.len()
            );
        }
    }

    // ── Chamfer presence test ──────────────────────────────────────

    #[test]
    fn chamfered_rooms_present() {
        for (preset, extent) in &[
            (V3Preset::Sparse, 2048),
            (V3Preset::Moderate, 2048),
            (V3Preset::Rich, 3072),
        ] {
            let config = V3Config::new(0, *preset, *extent).unwrap();
            let mut alloc = V3IdAllocator::new();
            let (footprints, _) = build_footprints(&config, V3Seed::new(0), &mut alloc).unwrap();

            let chamfered_count = footprints.iter().filter(|fp| fp.vertices.len() > 4).count();
            let total = footprints.len();

            assert!(
                chamfered_count > 0,
                "preset {preset:?} at {extent}: no chamfered rooms in {total} rooms"
            );

            // Meaningful fraction: at least 25% chamfered
            let fraction = chamfered_count as f64 / total as f64;
            assert!(
                fraction >= 0.25,
                "preset {preset:?} at {extent}: only {chamfered_count}/{total} chamfered ({fraction:.1})"
            );
        }
    }

    // ── Edge direction test ────────────────────────────────────────

    #[test]
    fn all_edges_cardinal_or_45() {
        for (preset, extent) in &[
            (V3Preset::Sparse, 2048),
            (V3Preset::Moderate, 2048),
            (V3Preset::Rich, 3072),
        ] {
            let config = V3Config::new(0, *preset, *extent).unwrap();
            let mut alloc = V3IdAllocator::new();
            let (footprints, _) = build_footprints(&config, V3Seed::new(0), &mut alloc).unwrap();

            for fp in &footprints {
                for &(a, b) in &fp.edges() {
                    let dx = (b.0 - a.0).unsigned_abs();
                    let dy = (b.1 - a.1).unsigned_abs();
                    assert!(
                        dx == 0 || dy == 0 || dx == dy,
                        "unapproved edge ({a:?}→{b:?}) in room {}",
                        fp.room_id
                    );
                }
            }
        }
    }

    // ── Convexity test ─────────────────────────────────────────────

    #[test]
    fn all_footprints_convex() {
        for (preset, extent) in &[
            (V3Preset::Sparse, 2048),
            (V3Preset::Moderate, 2048),
            (V3Preset::Rich, 3072),
        ] {
            let config = V3Config::new(0, *preset, *extent).unwrap();
            let mut alloc = V3IdAllocator::new();
            let (footprints, _) = build_footprints(&config, V3Seed::new(0), &mut alloc).unwrap();
            for fp in &footprints {
                fp.validate_convex().unwrap();
            }
        }
    }

    // ── Determinism test ───────────────────────────────────────────

    #[test]
    fn deterministic_replay() {
        for (preset, extent) in &[
            (V3Preset::Sparse, 2048),
            (V3Preset::Moderate, 2048),
            (V3Preset::Rich, 3072),
        ] {
            let config = V3Config::new(99, *preset, *extent).unwrap();

            let mut alloc1 = V3IdAllocator::new();
            let (fp1, layout1) = build_footprints(&config, V3Seed::new(99), &mut alloc1).unwrap();

            let mut alloc2 = V3IdAllocator::new();
            let (fp2, layout2) = build_footprints(&config, V3Seed::new(99), &mut alloc2).unwrap();

            assert_eq!(fp1, fp2);
            assert_eq!(layout1, layout2);
        }
    }

    // ── Seed difference test ───────────────────────────────────────

    #[test]
    fn different_seeds_produce_different_placement() {
        for (preset, extent) in &[
            (V3Preset::Sparse, 2048),
            (V3Preset::Moderate, 2048),
            (V3Preset::Rich, 3072),
        ] {
            let seed_a = 0u64;
            let seed_b = 42u64;

            let cfg_a = V3Config::new(seed_a, *preset, *extent).unwrap();
            let cfg_b = V3Config::new(seed_b, *preset, *extent).unwrap();

            let mut alloc_a = V3IdAllocator::new();
            let (fp_a, _) = build_footprints(&cfg_a, V3Seed::new(seed_a), &mut alloc_a).unwrap();

            let mut alloc_b = V3IdAllocator::new();
            let (fp_b, _) = build_footprints(&cfg_b, V3Seed::new(seed_b), &mut alloc_b).unwrap();

            // Build a deterministic signature of the placement
            let sig_a: Vec<(i32, i32, i32, i32, u8)> = fp_a
                .iter()
                .map(|fp| (fp.aabb.0, fp.aabb.1, fp.aabb.2, fp.aabb.3, fp.layer))
                .collect();
            let sig_b: Vec<(i32, i32, i32, i32, u8)> = fp_b
                .iter()
                .map(|fp| (fp.aabb.0, fp.aabb.1, fp.aabb.2, fp.aabb.3, fp.layer))
                .collect();

            assert_ne!(
                sig_a, sig_b,
                "seeds {seed_a} and {seed_b} produced identical placement for {preset:?} at {extent}"
            );
        }
    }

    // ── All requested seeds succeed ────────────────────────────────

    #[test]
    fn seeds_0_42_99_255_all_presets_and_valid_extents_succeed() {
        for &seed_val in &[0u64, 42, 99, 255] {
            for preset in [V3Preset::Sparse, V3Preset::Moderate, V3Preset::Rich] {
                for &extent in &[1024, 2048, 3072] {
                    let config = V3Config::new(seed_val, preset, extent).unwrap();
                    let mut alloc = V3IdAllocator::new();
                    let (footprints, layout) =
                        build_footprints(&config, V3Seed::new(seed_val), &mut alloc).unwrap();
                    let mut replay_alloc = V3IdAllocator::new();
                    let (replay, replay_layout) =
                        build_footprints(&config, V3Seed::new(seed_val), &mut replay_alloc)
                            .unwrap();
                    assert_eq!(footprints, replay);
                    assert_eq!(layout, replay_layout);
                    assert_eq!(footprints.len(), preset.min_rooms() as usize);
                    assert!(layout.primary < footprints.len());
                    assert!(layout.secondary < footprints.len());
                    assert!(layout.transition_lower < footprints.len());
                    assert!(layout.transition_upper < footprints.len());
                    let primary = &footprints[layout.primary];
                    let secondary = &footprints[layout.secondary];
                    let transition_lower = &footprints[layout.transition_lower];
                    let transition_upper = &footprints[layout.transition_upper];
                    assert_eq!(primary.layer, 0);
                    assert_eq!(secondary.layer, 0);
                    assert_eq!(transition_lower.layer, 0);
                    assert_eq!(transition_upper.layer, 1);
                    assert!(primary.aabb.2 < secondary.aabb.0);
                    let portal_y = (primary.aabb.1 + primary.aabb.3) / 2;
                    assert!(secondary.aabb.1 <= portal_y && portal_y <= secondary.aabb.3);
                    assert!(transition_lower.aabb.3 < transition_upper.aabb.1);
                    for (index, footprint) in footprints.iter().enumerate() {
                        footprint.validate_convex().unwrap();
                        let (x0, y0, x1, y1) = footprint.aabb;
                        assert!(x0 >= 0 && y0 >= 0 && x1 <= extent as i32 && y1 <= extent as i32);
                        assert!((MIN_OUTER_SPAN..=MAX_OUTER_SPAN).contains(&(x1 - x0)));
                        assert!((MIN_OUTER_SPAN..=MAX_OUTER_SPAN).contains(&(y1 - y0)));
                        for (from, to) in footprint.edges() {
                            let dx = (to.0 - from.0).unsigned_abs();
                            let dy = (to.1 - from.1).unsigned_abs();
                            assert!(dx == 0 || dy == 0 || dx == dy);
                        }
                        for &(x, y) in &footprint.vertices {
                            assert_eq!(x % CONSTRUCTION_QUANTUM, 0);
                            assert_eq!(y % CONSTRUCTION_QUANTUM, 0);
                        }
                        for other in &footprints[..index] {
                            assert!(!aabbs_overlap(footprint.aabb, other.aabb));
                        }
                    }
                    let mut sizes: Vec<_> = footprints
                        .iter()
                        .map(|footprint| {
                            (
                                footprint.aabb.2 - footprint.aabb.0,
                                footprint.aabb.3 - footprint.aabb.1,
                            )
                        })
                        .collect();
                    sizes.sort_unstable();
                    sizes.dedup();
                    assert!(
                        sizes.len() >= 3,
                        "insufficient room-size variation: {sizes:?}"
                    );
                    assert!(footprints
                        .iter()
                        .any(|footprint| footprint.vertices.len() > 4));
                    let lower = footprints
                        .iter()
                        .filter(|footprint| footprint.layer == 0)
                        .count();
                    let upper = footprints.len() - lower;
                    assert!(lower.abs_diff(upper) <= 1);
                }
            }
        }
    }

    // ── Boundary extent test ───────────────────────────────────────

    #[test]
    fn sparse_at_1024_succeeds() {
        let config = V3Config::new(0, V3Preset::Sparse, 1024).unwrap();
        let mut alloc = V3IdAllocator::new();
        let (footprints, _) = build_footprints(&config, V3Seed::new(0), &mut alloc).unwrap();
        assert_eq!(footprints.len(), 12);
        for fp in &footprints {
            assert!(fp.aabb.2 <= 1024);
            assert!(fp.aabb.3 <= 1024);
        }
    }

    #[test]
    fn sparse_at_3072_succeeds() {
        let config = V3Config::new(0, V3Preset::Sparse, 3072).unwrap();
        let mut alloc = V3IdAllocator::new();
        let (footprints, _) = build_footprints(&config, V3Seed::new(0), &mut alloc).unwrap();
        assert_eq!(footprints.len(), 12);
        for fp in &footprints {
            assert!(fp.aabb.2 <= 3072);
            assert!(fp.aabb.3 <= 3072);
        }
    }

    #[test]
    fn rich_at_3072_succeeds() {
        let config = V3Config::new(255, V3Preset::Rich, 3072).unwrap();
        let mut alloc = V3IdAllocator::new();
        let (footprints, _) = build_footprints(&config, V3Seed::new(255), &mut alloc).unwrap();
        assert_eq!(footprints.len(), 28);
    }

    // ── FootprintLayout validity ───────────────────────────────────

    #[test]
    fn layout_indices_are_valid() {
        for (preset, extent) in &[
            (V3Preset::Sparse, 2048),
            (V3Preset::Moderate, 2048),
            (V3Preset::Rich, 3072),
        ] {
            let config = V3Config::new(0, *preset, *extent).unwrap();
            let mut alloc = V3IdAllocator::new();
            let (footprints, layout) =
                build_footprints(&config, V3Seed::new(0), &mut alloc).unwrap();

            let n = footprints.len();
            assert!(layout.primary < n, "primary index out of bounds");
            assert!(layout.secondary < n, "secondary index out of bounds");
            assert!(
                layout.transition_lower < n,
                "transition_lower index out of bounds"
            );
            assert!(
                layout.transition_upper < n,
                "transition_upper index out of bounds"
            );

            // Primary and secondary must be on lower layer
            assert_eq!(
                footprints[layout.primary].layer, 0,
                "primary must be lower layer"
            );
            assert_eq!(
                footprints[layout.secondary].layer, 0,
                "secondary must be lower layer"
            );
            // Transition lower must be on lower layer
            assert_eq!(
                footprints[layout.transition_lower].layer, 0,
                "transition_lower must be lower layer"
            );
            // Transition upper must be on upper layer
            assert_eq!(
                footprints[layout.transition_upper].layer, 1,
                "transition_upper must be upper layer"
            );
        }
    }

    // ── Quantum alignment test ─────────────────────────────────────

    #[test]
    fn all_positions_quantum_aligned() {
        let config = V3Config::new(0, V3Preset::Rich, 3072).unwrap();
        let mut alloc = V3IdAllocator::new();
        let (footprints, _) = build_footprints(&config, V3Seed::new(0), &mut alloc).unwrap();

        let q = CONSTRUCTION_QUANTUM;
        for fp in &footprints {
            for &(vx, vy) in &fp.vertices {
                assert_eq!(
                    vx % q,
                    0,
                    "vertex x={vx} not quantum-aligned in room {}",
                    fp.room_id
                );
                assert_eq!(
                    vy % q,
                    0,
                    "vertex y={vy} not quantum-aligned in room {}",
                    fp.room_id
                );
            }
        }
    }

    // ── Multiple room sizes test ───────────────────────────────────

    #[test]
    fn multiple_room_sizes_present() {
        for (preset, extent) in &[
            (V3Preset::Sparse, 2048),
            (V3Preset::Moderate, 2048),
            (V3Preset::Rich, 3072),
        ] {
            let config = V3Config::new(42, *preset, *extent).unwrap();
            let mut alloc = V3IdAllocator::new();
            let (footprints, _) = build_footprints(&config, V3Seed::new(42), &mut alloc).unwrap();

            let mut sizes: Vec<(i32, i32)> = footprints
                .iter()
                .map(|fp| (fp.aabb.2 - fp.aabb.0, fp.aabb.3 - fp.aabb.1))
                .collect();
            sizes.sort_unstable();
            sizes.dedup();
            assert!(
                sizes.len() >= 3,
                "preset {preset:?} at {extent}: only {} distinct room sizes ({sizes:?})",
                sizes.len()
            );
        }
    }

    // ── Legacy Footprint constructor tests (preserved) ─────────────

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

    #[test]
    fn footprint_convex_valid() {
        // Test with a known rectangular footprint
        let mut alloc = V3IdAllocator::new();
        let fp = Footprint::rectangular(RoomId(0), 0, 0, (0, 0, 128, 128), &mut alloc).unwrap();
        fp.validate_convex().unwrap();
    }

    #[test]
    fn every_chamfer_pattern_is_ccw_convex_with_approved_edges() {
        for &corners in CHAMFER_PATTERNS {
            let mut alloc = V3IdAllocator::new();
            let footprint =
                Footprint::chamfered(RoomId(0), 0, 0, (0, 0, 192, 192), corners, 16, &mut alloc)
                    .unwrap();
            assert_eq!(footprint.vertices.len(), 4 + corners.len());
            footprint.validate_convex().unwrap();
            for (from, to) in footprint.edges() {
                let dx = (to.0 - from.0).unsigned_abs();
                let dy = (to.1 - from.1).unsigned_abs();
                assert!(dx == 0 || dy == 0 || dx == dy);
            }
        }
    }
}
