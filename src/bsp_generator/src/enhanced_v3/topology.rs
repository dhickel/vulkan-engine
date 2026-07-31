//! Deterministic two-layer topology for Enhanced V3.
//!
//! Builds a connected per-layer spanning tree, adds exactly `target_loops`
//! non-tree edges where adjacency permits, creates one cardinal portal per
//! route, a straight route envelope, one valid inter-layer transition, and
//! selects the largest lower room for spawn.

use std::collections::BTreeMap;

use super::config::{self, V3Config, CONSTRUCTION_QUANTUM, HEADROOM, ROUTE_WIDTH};
use super::error::V3Error;
use super::footprint::{Footprint, FootprintLayout};
use super::ids::{
    CommittedPortal, CommittedRoom, CommittedRoute, CommittedSurface, CommittedTopology,
    CommittedTransition, QuantumVolume, SupportSurfaceKind, SurfaceOwner, V3IdAllocator,
};
use super::rng::{self, V3Seed};

// ── Cardinal direction ─────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Dir {
    North,
    South,
    West,
    East,
}

impl Dir {
    fn tag(self) -> &'static str {
        match self {
            Dir::North => "north",
            Dir::South => "south",
            Dir::West => "west",
            Dir::East => "east",
        }
    }

    fn opposite(self) -> Dir {
        match self {
            Dir::North => Dir::South,
            Dir::South => Dir::North,
            Dir::West => Dir::East,
            Dir::East => Dir::West,
        }
    }
}

// ── Candidate edge ─────────────────────────────────────────────────────────

/// A candidate same-layer edge between two rooms with a cardinal adjacency.
#[derive(Debug, Clone)]
struct CandidateEdge {
    /// The source room direction (portal is on this wall of source).
    source_dir: Dir,
    /// Source room footprint index.
    source_idx: usize,
    /// Target room footprint index.
    target_idx: usize,
    /// Cross-axis overlap interval: (lo, hi) in Quake units.
    overlap: (i32, i32),
    /// Deterministic rank for ordering.
    rank: u64,
}

/// Return the actual cardinal-wall interior span for `direction`.
///
/// A footprint AABB is not sufficient for a chamfered room: an AABB corner
/// can be outside the convex room. Routes and apertures therefore derive their
/// cross-axis spans from the matching cardinal polygon edge, never from the
/// AABB alone.
fn cardinal_wall_span(footprint: &Footprint, direction: Dir) -> Option<(i32, i32)> {
    let mut span: Option<(i32, i32)> = None;
    for index in 0..footprint.vertices.len() {
        let a = footprint.vertices[index];
        let b = footprint.vertices[(index + 1) % footprint.vertices.len()];
        let candidate = match direction {
            Dir::East if a.0 == footprint.aabb.2 && b.0 == footprint.aabb.2 => {
                Some((a.1.min(b.1), a.1.max(b.1)))
            }
            Dir::West if a.0 == footprint.aabb.0 && b.0 == footprint.aabb.0 => {
                Some((a.1.min(b.1), a.1.max(b.1)))
            }
            Dir::North if a.1 == footprint.aabb.1 && b.1 == footprint.aabb.1 => {
                Some((a.0.min(b.0), a.0.max(b.0)))
            }
            Dir::South if a.1 == footprint.aabb.3 && b.1 == footprint.aabb.3 => {
                Some((a.0.min(b.0), a.0.max(b.0)))
            }
            _ => None,
        };
        if let Some(candidate) = candidate {
            span = match span {
                Some(existing) => Some((existing.0.min(candidate.0), existing.1.max(candidate.1))),
                None => Some(candidate),
            };
        }
    }
    span.filter(|(lo, hi)| lo < hi)
}

/// Select a centered, quantum-aligned 64-unit aperture span from an available
/// common cardinal-wall interval.
fn centered_route_span(lo: i32, hi: i32) -> Option<(i32, i32)> {
    if hi - lo < ROUTE_WIDTH {
        return None;
    }
    let q = CONSTRUCTION_QUANTUM;
    let center = ((lo + hi) / 2 / q) * q;
    let half = ROUTE_WIDTH / 2;
    let span = (center - half, center + half);
    if span.0 >= lo && span.1 <= hi {
        Some(span)
    } else {
        // All footprint coordinates are quantum-aligned. This fallback covers
        // a midpoint that rounded down beyond an odd-width common interval.
        let start = lo;
        let span = (start, start + ROUTE_WIDTH);
        (span.1 <= hi).then_some(span)
    }
}

/// Compute the exact 64-unit cross-axis span shared by the two real cardinal
/// wall interiors. The result is both the route clear span and the aperture
/// span at both endpoints.
fn cross_overlap(source: &Footprint, target: &Footprint, source_dir: Dir) -> Option<(i32, i32)> {
    let source_span = cardinal_wall_span(source, source_dir)?;
    let target_span = cardinal_wall_span(target, source_dir.opposite())?;
    centered_route_span(
        source_span.0.max(target_span.0),
        source_span.1.min(target_span.1),
    )
}

/// Test whether the straight corridor between source and target intersects
/// any third room's AABB.
fn corridor_clear_of_third_rooms(
    footprints: &[Footprint],
    source_idx: usize,
    target_idx: usize,
    source_dir: Dir,
    overlap: (i32, i32),
) -> bool {
    let src = footprints[source_idx].aabb;
    let tgt = footprints[target_idx].aabb;

    // Build the corridor bounds in XY.
    // Engine convention: North=y0(minY), South=y3(maxY).
    let (cor_x0, cor_x1, cor_y0, cor_y1) = match source_dir {
        Dir::East => {
            // Source is west of target. Corridor from source east to target west.
            (src.2, tgt.0, overlap.0, overlap.1)
        }
        Dir::West => {
            // Source is east of target. Corridor from target east to source west.
            (tgt.2, src.0, overlap.0, overlap.1)
        }
        Dir::North => {
            // Source is north of target (higher Y). Corridor from target south to source north.
            (overlap.0, overlap.1, tgt.3, src.1)
        }
        Dir::South => {
            // Source is south of target (lower Y). Corridor from source south to target north.
            (overlap.0, overlap.1, src.3, tgt.1)
        }
    };

    // The corridor must be non-degenerate.
    if cor_x0 >= cor_x1 || cor_y0 >= cor_y1 {
        return false;
    }

    // Check against every other room.
    for (idx, fp) in footprints.iter().enumerate() {
        if idx == source_idx || idx == target_idx {
            continue;
        }
        // Skip rooms on different layers — they don't block same-layer
        // corridors.
        if fp.layer != footprints[source_idx].layer {
            continue;
        }
        let (rx0, ry0, rx1, ry1) = fp.aabb;
        // Positive overlap check.
        if cor_x0 < rx1 && cor_x1 > rx0 && cor_y0 < ry1 && cor_y1 > ry0 {
            return false;
        }
    }
    true
}

/// Build all eligible candidate edges for a set of footprints on the same
/// layer. Returns edges sorted by deterministic rank.
fn build_candidate_edges(footprints: &[Footprint], seed: V3Seed) -> Vec<CandidateEdge> {
    let mut edges = Vec::new();

    for i in 0..footprints.len() {
        for j in (i + 1)..footprints.len() {
            if footprints[i].layer != footprints[j].layer {
                continue;
            }
            let src = &footprints[i];
            let tgt = &footprints[j];
            let sa = src.aabb;
            let ta = tgt.aabb;

            // Check all four cardinal adjacency possibilities.
            let tests: [(Dir, bool); 4] = [
                // source_dir, is_horizontal
                (Dir::East, true),
                (Dir::West, true),
                (Dir::North, false),
                (Dir::South, false),
            ];

            for &(source_dir, _horizontal) in &tests {
                // For the edge to be cardinal-adjacent, the source wall must
                // face the target room.
                // Engine convention: North=y0(minY), South=y3(maxY), West=x0(minX), East=x2(maxX).
                let facing = match source_dir {
                    Dir::East => sa.2 <= ta.0,
                    Dir::West => sa.0 >= ta.2,
                    Dir::North => sa.1 >= ta.3,
                    Dir::South => sa.3 <= ta.1,
                };
                if !facing {
                    continue;
                }

                if let Some(overlap) = cross_overlap(src, tgt, source_dir) {
                    // Build deterministic rank.
                    let key = if i < j {
                        format!(
                            "edge/{:04}/{:04}/{}",
                            footprints[i].room_id.raw(),
                            footprints[j].room_id.raw(),
                            source_dir.tag()
                        )
                    } else {
                        format!(
                            "edge/{:04}/{:04}/{}",
                            footprints[j].room_id.raw(),
                            footprints[i].room_id.raw(),
                            source_dir.opposite().tag()
                        )
                    };
                    let rank = seed
                        .candidate_seed(rng::tags::TOPOLOGY, key.as_bytes())
                        .u64_at(0);

                    // Check corridor clearance.
                    if corridor_clear_of_third_rooms(footprints, i, j, source_dir, overlap) {
                        edges.push(CandidateEdge {
                            source_dir,
                            source_idx: i,
                            target_idx: j,
                            overlap,
                            rank,
                        });
                    }
                }
            }
        }
    }

    // Sort by rank ascending, then by complete semantic edge identity. No
    // hash-table traversal can influence topology or serialized route order.
    edges.sort_by(|a, b| {
        a.rank
            .cmp(&b.rank)
            .then_with(|| {
                footprints[a.source_idx]
                    .room_id
                    .cmp(&footprints[b.source_idx].room_id)
            })
            .then_with(|| {
                footprints[a.target_idx]
                    .room_id
                    .cmp(&footprints[b.target_idx].room_id)
            })
            .then_with(|| a.source_dir.cmp(&b.source_dir))
    });

    edges
}

// ── Union-Find ─────────────────────────────────────────────────────────────

struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
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

    fn union(&mut self, x: usize, y: usize) -> bool {
        let rx = self.find(x);
        let ry = self.find(y);
        if rx == ry {
            return false;
        }
        match self.rank[rx].cmp(&self.rank[ry]) {
            std::cmp::Ordering::Less => self.parent[rx] = ry,
            std::cmp::Ordering::Greater => self.parent[ry] = rx,
            std::cmp::Ordering::Equal => {
                self.parent[ry] = rx;
                self.rank[rx] += 1;
            }
        }
        true
    }
}

// ── Topology construction ──────────────────────────────────────────────────

/// Build a committed topology from footprints and seed.
///
/// Constructs a spanning tree on each layer, adds exactly `target_loops`
/// extra non-tree edges, creates one `CommittedPortal` and one
/// `CommittedRoute` per edge, one valid `CommittedTransition`, and selects
/// the largest lower room for spawn.
pub fn build_topology(
    config: &V3Config,
    footprints: &[Footprint],
    _layout: &FootprintLayout,
    seed: V3Seed,
    alloc: &mut V3IdAllocator,
) -> Result<CommittedTopology, V3Error> {
    if footprints.is_empty() {
        return Err(V3Error::TopologyInvariant {
            detail: "topology requires at least one footprint".into(),
        });
    }

    let q = CONSTRUCTION_QUANTUM;
    let target_loops = config.preset.target_loops() as usize;

    // ── Build committed rooms ─────────────────────────────────────────
    let mut rooms: Vec<CommittedRoom> = Vec::new();
    let mut surfaces: Vec<CommittedSurface> = Vec::new();

    for fp in footprints {
        let (x0, y0, x1, y1) = fp.aabb;
        let width = (x1 - x0) as u32;
        let depth = (y1 - y0) as u32;
        let height = config::ROOM_HEIGHT as u32;

        rooms.push(CommittedRoom {
            id: fp.room_id,
            layer: fp.layer,
            shell: (x0, y0, x1, y1),
            floor_z: fp.floor_z,
            dims: (width, depth, height),
        });

        let floor_id = alloc.next_surface()?;
        surfaces.push(CommittedSurface {
            id: floor_id,
            room_id: fp.room_id,
            kind: SupportSurfaceKind::Floor,
            owner: SurfaceOwner {
                parent_kind: "room".into(),
                parent_id: fp.room_id.raw(),
                face: "floor".into(),
                direction: "up".into(),
                qualifier: "primary".into(),
            },
        });
    }

    // ── Split footprints by layer ─────────────────────────────────────
    let lower_indices: Vec<usize> = footprints
        .iter()
        .enumerate()
        .filter(|(_, fp)| fp.layer == 0)
        .map(|(i, _)| i)
        .collect();
    let upper_indices: Vec<usize> = footprints
        .iter()
        .enumerate()
        .filter(|(_, fp)| fp.layer == 1)
        .map(|(i, _)| i)
        .collect();

    if lower_indices.len() < 2 || upper_indices.is_empty() {
        return Err(V3Error::TopologyInvariant {
            detail: format!(
                "need at least 2 lower and 1 upper rooms, got {} / {}",
                lower_indices.len(),
                upper_indices.len()
            ),
        });
    }

    // ── Build candidate edges and split by layer ────────────────────
    let all_edges = build_candidate_edges(footprints, seed);
    let lower_edges: Vec<&CandidateEdge> = all_edges
        .iter()
        .filter(|e| footprints[e.source_idx].layer == 0)
        .collect();
    let upper_edges: Vec<&CandidateEdge> = all_edges
        .iter()
        .filter(|e| footprints[e.source_idx].layer == 1)
        .collect();

    // ── Kruskal spanning tree on each layer ───────────────────────────
    let mut selected_edges: Vec<&CandidateEdge> = Vec::new();
    let mut selected_lower_edges: Vec<&CandidateEdge> = Vec::new();
    let mut selected_upper_edges: Vec<&CandidateEdge> = Vec::new();

    // --- Lower layer ---
    {
        // Map global footprint indices to local union-find indices.
        let idx_map: BTreeMap<usize, usize> = lower_indices
            .iter()
            .enumerate()
            .map(|(local, &global)| (global, local))
            .collect();
        let mut uf = UnionFind::new(lower_indices.len());
        for edge in &lower_edges {
            let li = idx_map[&edge.source_idx];
            let lj = idx_map[&edge.target_idx];
            if uf.union(li, lj) {
                selected_lower_edges.push(edge);
                selected_edges.push(edge);
                if selected_lower_edges.len() == lower_indices.len() - 1 {
                    break;
                }
            }
        }
        if selected_lower_edges.len() != lower_indices.len() - 1 {
            return Err(V3Error::TopologyInvariant {
                detail: format!(
                    "lower layer spanning tree incomplete: {}/{} edges from {} candidates",
                    selected_lower_edges.len(),
                    lower_indices.len() - 1,
                    lower_edges.len()
                ),
            });
        }
    }

    // --- Upper layer ---
    {
        let idx_map: BTreeMap<usize, usize> = upper_indices
            .iter()
            .enumerate()
            .map(|(local, &global)| (global, local))
            .collect();
        let mut uf = UnionFind::new(upper_indices.len());
        for edge in &upper_edges {
            let li = idx_map[&edge.source_idx];
            let lj = idx_map[&edge.target_idx];
            if uf.union(li, lj) {
                selected_upper_edges.push(edge);
                selected_edges.push(edge);
                if selected_upper_edges.len() == upper_indices.len() - 1 {
                    break;
                }
            }
        }
        if selected_upper_edges.len() != upper_indices.len() - 1 {
            return Err(V3Error::TopologyInvariant {
                detail: format!(
                    "upper layer spanning tree incomplete: {}/{} edges from {} candidates",
                    selected_upper_edges.len(),
                    upper_indices.len() - 1,
                    upper_edges.len()
                ),
            });
        }
    }

    // ── Add loop edges ────────────────────────────────────────────────
    if target_loops > 0 {
        let mut loops_added = 0usize;

        // Track every accepted unordered pair. Inserting each loop candidate
        // immediately prevents a second directed representation of the same
        // edge from being counted as a loop.
        let mut selected_keys: std::collections::BTreeSet<(usize, usize)> = selected_edges
            .iter()
            .map(|edge| {
                let a = edge.source_idx.min(edge.target_idx);
                let b = edge.source_idx.max(edge.target_idx);
                (a, b)
            })
            .collect();

        // Candidates are already rank-ordered; lower then upper is a stable
        // layer partition, not an unordered collection traversal.
        for edge in lower_edges.iter().chain(upper_edges.iter()) {
            if loops_added == target_loops {
                break;
            }
            let key = (
                edge.source_idx.min(edge.target_idx),
                edge.source_idx.max(edge.target_idx),
            );
            if !selected_keys.insert(key) {
                continue;
            }
            selected_edges.push(edge);
            loops_added += 1;
        }

        if loops_added < target_loops {
            return Err(V3Error::TopologyInvariant {
                detail: format!("could only add {loops_added}/{target_loops} loop edges"),
            });
        }
    }

    // Canonicalize committed route/portal allocation independently from the
    // per-layer Kruskal passes. This preserves deterministic serialization
    // even when candidate ranks interleave between layers.
    selected_edges.sort_by(|a, b| {
        a.rank
            .cmp(&b.rank)
            .then_with(|| {
                footprints[a.source_idx]
                    .room_id
                    .cmp(&footprints[b.source_idx].room_id)
            })
            .then_with(|| {
                footprints[a.target_idx]
                    .room_id
                    .cmp(&footprints[b.target_idx].room_id)
            })
            .then_with(|| a.source_dir.cmp(&b.source_dir))
    });

    // ── Build portals and routes ──────────────────────────────────────
    let portal_width = ROUTE_WIDTH;
    let portal_height = HEADROOM;

    let mut portals: Vec<CommittedPortal> = Vec::new();
    let mut routes: Vec<CommittedRoute> = Vec::new();

    for (route_index, edge) in selected_edges.iter().enumerate() {
        let source_fp = &footprints[edge.source_idx];
        let target_fp = &footprints[edge.target_idx];
        let source_room = &rooms[edge.source_idx];
        let target_room = &rooms[edge.target_idx];

        let (overlap_lo, overlap_hi) = edge.overlap;
        let cross_center = ((overlap_lo + overlap_hi) / 2 / q) * q;

        // Anchor on the source room's wall.
        // Engine convention: North=y0(minY), South=y3(maxY).
        let (anchor_x, anchor_y) = match edge.source_dir {
            Dir::East => (source_fp.aabb.2, cross_center),
            Dir::West => (source_fp.aabb.0, cross_center),
            Dir::North => (cross_center, source_fp.aabb.1),
            Dir::South => (cross_center, source_fp.aabb.3),
        };
        let anchor_z = source_fp.floor_z + q + portal_height as i32 / 2;

        let portal_id = alloc.next_portal()?;
        portals.push(CommittedPortal {
            id: portal_id,
            source_room: source_room.id,
            target_room: Some(target_room.id),
            wall: edge.source_dir.tag().into(),
            anchor: (anchor_x, anchor_y, anchor_z),
            width: portal_width as u32,
            height: portal_height as u32,
        });

        // Build route envelope: a straight clear corridor from source wall
        // to target wall, centered on the cross-axis overlap.
        let envelope_half = portal_width as i32 / 2;
        let envelope = match edge.source_dir {
            Dir::East => (
                source_fp.aabb.2 - envelope_half,
                cross_center - envelope_half,
                target_fp.aabb.0 + envelope_half,
                cross_center + envelope_half,
            ),
            Dir::West => (
                target_fp.aabb.2 - envelope_half,
                cross_center - envelope_half,
                source_fp.aabb.0 + envelope_half,
                cross_center + envelope_half,
            ),
            Dir::North => (
                cross_center - envelope_half,
                target_fp.aabb.3 - envelope_half,
                cross_center + envelope_half,
                source_fp.aabb.1 + envelope_half,
            ),
            Dir::South => (
                cross_center - envelope_half,
                source_fp.aabb.3 - envelope_half,
                cross_center + envelope_half,
                target_fp.aabb.1 + envelope_half,
            ),
        };

        routes.push(CommittedRoute {
            id: route_index as u32,
            source_room: source_room.id,
            target_room: target_room.id,
            envelopes: vec![envelope],
        });
    }

    // ── Build transition ──────────────────────────────────────────────
    // Validate the selected spawn host against the actual convex footprint
    // before topology becomes immutable. `compute_reservations` repeats this
    // deterministic room selection from committed room data.
    validate_spawn_host(footprints)?;

    let transition = build_transition(footprints, &rooms, &lower_indices, &upper_indices, seed)?;
    let transitions = vec![transition];

    // ── Assemble ─────────────────────────────────────────────────────
    let topology = CommittedTopology {
        rooms,
        surfaces,
        portals,
        routes,
        transitions,
    };

    // Validate XY bounds against config.
    for room in &topology.rooms {
        let x1 = room.shell.2;
        let y1 = room.shell.3;
        if x1 > config.xy_extent as i32 || y1 > config.xy_extent as i32 {
            return Err(V3Error::RoomOutOfBounds {
                room_id: room.id.raw(),
                extent: config.xy_extent,
            });
        }
    }

    Ok(topology)
}

// ── Transition construction ────────────────────────────────────────────────

fn build_transition(
    footprints: &[Footprint],
    rooms: &[CommittedRoom],
    lower_indices: &[usize],
    upper_indices: &[usize],
    seed: V3Seed,
) -> Result<CommittedTransition, V3Error> {
    let q = CONSTRUCTION_QUANTUM;

    // Collect candidate (lower, upper) pairs with XY projection overlap
    // where the lower room is south of the upper room (lower.aabb.3 < upper.aabb.1).
    struct TransitionCandidate {
        lower_idx: usize,
        upper_idx: usize,
        x_overlap: (i32, i32),
        rank: u64,
    }

    let mut candidates: Vec<TransitionCandidate> = Vec::new();

    for &li in lower_indices {
        let lower = &footprints[li];
        for &ui in upper_indices {
            let upper = &footprints[ui];
            // Lower must be south of upper.
            if lower.aabb.3 >= upper.aabb.1 {
                continue;
            }
            // Transition approaches use the lower south and upper north
            // cardinal wall interiors. AABB overlap alone could land in a
            // chamfer cutout, so reserve an actual 64-unit host span.
            let Some(x_span) = cross_overlap(lower, upper, Dir::South) else {
                continue;
            };
            // A positive gap is required between the real host walls. The
            // placement grid reserves at least one quantum for this envelope.
            if upper.aabb.1 - lower.aabb.3 < q {
                continue;
            }
            let key = format!(
                "transition/{:04}/{:04}",
                lower.room_id.raw(),
                upper.room_id.raw()
            );
            let rank = seed
                .candidate_seed(rng::tags::TOPOLOGY, key.as_bytes())
                .u64_at(0);
            candidates.push(TransitionCandidate {
                lower_idx: li,
                upper_idx: ui,
                x_overlap: x_span,
                rank,
            });
        }
    }

    if candidates.is_empty() {
        return Err(V3Error::TopologyInvariant {
            detail: "no valid transition pair: lower room south of upper room with X overlap"
                .into(),
        });
    }

    // Select the best candidate by deterministic rank.
    candidates.sort_by(|a, b| {
        a.rank
            .cmp(&b.rank)
            .then_with(|| {
                footprints[a.lower_idx]
                    .room_id
                    .cmp(&footprints[b.lower_idx].room_id)
            })
            .then_with(|| {
                footprints[a.upper_idx]
                    .room_id
                    .cmp(&footprints[b.upper_idx].room_id)
            })
    });

    let chosen = &candidates[0];
    let lower = &footprints[chosen.lower_idx];
    let upper = &footprints[chosen.upper_idx];
    let (pv_x0, pv_x1) = chosen.x_overlap;

    // Build a protected volume spanning between actual host walls. Its
    // 64-unit X span is the same cardinal interior used by both landings.
    let pv_y0 = lower.aabb.3;
    let pv_y1 = upper.aabb.1;
    let pv_z0 = config::LOWER_FLOOR_Z;
    let pv_z1 = config::UPPER_FLOOR_Z + config::ROOM_HEIGHT as i32;

    if pv_x0 >= pv_x1 || pv_y0 >= pv_y1 || pv_z0 >= pv_z1 {
        return Err(V3Error::TopologyInvariant {
            detail: format!(
                "invalid transition protected volume: ({pv_x0},{pv_y0},{pv_z0})-({pv_x1},{pv_y1},{pv_z1})"
            ),
        });
    }

    // Lower landing: inside the lower room, at its south wall (y3, max Y).
    let lower_landing_y1 = (lower.aabb.3 / q) * q;
    let lower_landing = (pv_x0, lower_landing_y1 - 2 * q, pv_x1, lower_landing_y1);

    // Upper landing: inside the upper room, at its north wall (y1, min Y).
    let upper_landing = (
        pv_x0,
        (upper.aabb.1 / q) * q,
        pv_x1,
        (upper.aabb.1 / q) * q + 2 * q,
    );

    Ok(CommittedTransition {
        id: 0,
        lower_room: rooms[chosen.lower_idx].id,
        upper_room: rooms[chosen.upper_idx].id,
        protected_volume: (pv_x0, pv_y0, pv_z0, pv_x1, pv_y1, pv_z1),
        lower_landing,
        upper_landing,
        // This is a structural reservation for later stair emission, not an
        // emitted stair claim. Recording it gives both landing approaches a
        // bounded, aligned headroom witness in the committed topology.
        headroom_volumes: vec![(pv_x0, pv_y0, pv_z0, pv_x1, pv_y1, pv_z1)],
    })
}

// ── Spawn and light reservations ───────────────────────────────────────────

fn footprint_area(footprint: &Footprint) -> i64 {
    i64::from(footprint.aabb.2 - footprint.aabb.0) * i64::from(footprint.aabb.3 - footprint.aabb.1)
}

fn room_area(room: &CommittedRoom) -> i64 {
    i64::from(room.shell.2 - room.shell.0) * i64::from(room.shell.3 - room.shell.1)
}

fn footprint_contains_strictly(footprint: &Footprint, point: (i32, i32)) -> bool {
    (0..footprint.vertices.len()).all(|index| {
        let a = footprint.vertices[index];
        let b = footprint.vertices[(index + 1) % footprint.vertices.len()];
        let cross = i64::from(b.0 - a.0) * i64::from(point.1 - a.1)
            - i64::from(b.1 - a.1) * i64::from(point.0 - a.0);
        cross > 0
    })
}

fn select_spawn_footprint(footprints: &[Footprint]) -> Result<&Footprint, V3Error> {
    footprints
        .iter()
        .filter(|footprint| footprint.layer == 0)
        .max_by(|left, right| {
            footprint_area(left)
                .cmp(&footprint_area(right))
                // Prefer the stable lower RoomId when areas tie.
                .then_with(|| right.room_id.cmp(&left.room_id))
        })
        .ok_or_else(|| V3Error::TopologyInvariant {
            detail: "no lower rooms in topology".into(),
        })
}

fn validate_spawn_host(footprints: &[Footprint]) -> Result<(), V3Error> {
    let footprint = select_spawn_footprint(footprints)?;
    let q = CONSTRUCTION_QUANTUM;
    let center = (
        ((footprint.aabb.0 + footprint.aabb.2) / 2 / q) * q,
        ((footprint.aabb.1 + footprint.aabb.3) / 2 / q) * q,
    );
    let reservation_corners = [
        (center.0 - q, center.1 - q),
        (center.0 - q, center.1 + q),
        (center.0 + q, center.1 - q),
        (center.0 + q, center.1 + q),
    ];
    if footprint_area(footprint) < i64::from(7 * q) * i64::from(7 * q)
        || !reservation_corners
            .iter()
            .all(|&point| footprint_contains_strictly(footprint, point))
    {
        return Err(V3Error::TopologyInvariant {
            detail: format!(
                "lower spawn host {} lacks a clear quantum-aligned 32×32 convex interior",
                footprint.room_id
            ),
        });
    }
    Ok(())
}

/// Compute spawn and light reservation volumes from the frozen topology.
///
/// Spawn is placed in the largest lower room. Its 32×32 reservation is
/// quantum-aligned and has already been proven inside that room's actual
/// convex footprint by `build_topology`; it is not merely AABB-derived.
pub fn compute_reservations(
    topology: &CommittedTopology,
) -> Result<(QuantumVolume, Vec<QuantumVolume>), V3Error> {
    let q = CONSTRUCTION_QUANTUM;

    // Select the largest lower room for spawn.
    let spawn_room = topology
        .rooms
        .iter()
        .filter(|r| r.layer == 0)
        .max_by(|left, right| {
            room_area(left)
                .cmp(&room_area(right))
                // Match `select_spawn_footprint` exactly for area ties.
                .then_with(|| right.id.cmp(&left.id))
        })
        .ok_or_else(|| V3Error::TopologyInvariant {
            detail: "no lower rooms in topology".into(),
        })?;

    let cx = ((spawn_room.shell.0 + spawn_room.shell.2) / 2 / q) * q;
    let cy = ((spawn_room.shell.1 + spawn_room.shell.3) / 2 / q) * q;

    let spawn_volume = QuantumVolume::new(
        cx - q,
        cy - q,
        spawn_room.floor_z + q,
        cx + q,
        cy + q,
        spawn_room.floor_z + q + HEADROOM,
    )
    .ok_or_else(|| V3Error::InvalidReservation {
        detail: "invalid spawn volume".into(),
    })?;

    let mut light_volumes = Vec::new();
    for room in &topology.rooms {
        let lx = ((room.shell.0 + room.shell.2) / 2 / q) * q;
        let ly = ((room.shell.1 + room.shell.3) / 2 / q) * q;
        let lz = room.floor_z + room.dims.2 as i32 - 2 * q;

        if let Some(vol) = QuantumVolume::new(lx - q, ly - q, lz - q, lx + q, ly + q, lz + q) {
            light_volumes.push(vol);
        }
    }

    Ok((spawn_volume, light_volumes))
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::super::footprint::build_footprints;
    use super::*;

    // ── Helper ────────────────────────────────────────────────────────

    fn build_topology_for(
        preset: super::super::config::V3Preset,
        seed_val: u64,
        extent: u32,
    ) -> (CommittedTopology, Vec<Footprint>, V3Config) {
        let config = V3Config::new(seed_val, preset, extent).unwrap();
        let mut alloc = V3IdAllocator::new();
        let (footprints, layout) =
            build_footprints(&config, V3Seed::new(seed_val), &mut alloc).unwrap();
        let topology = build_topology(
            &config,
            &footprints,
            &layout,
            V3Seed::new(seed_val),
            &mut alloc,
        )
        .unwrap();
        (topology, footprints, config)
    }

    fn dir_from_tag(tag: &str) -> Dir {
        match tag {
            "north" => Dir::North,
            "south" => Dir::South,
            "west" => Dir::West,
            "east" => Dir::East,
            _ => panic!("non-cardinal portal wall {tag}"),
        }
    }

    fn assert_matrix_topology_contract(
        config: &V3Config,
        footprints: &[Footprint],
        topology: &CommittedTopology,
    ) {
        let room_count = config.preset.min_rooms() as usize;
        let target_loops = config.preset.target_loops() as usize;
        assert_eq!(topology.rooms.len(), room_count);
        assert_eq!(topology.routes.len(), room_count - 2 + target_loops);
        assert_eq!(topology.portals.len(), topology.routes.len());
        assert_eq!(topology.transitions.len(), 1);

        let footprints_by_id: BTreeMap<_, _> = footprints
            .iter()
            .map(|footprint| (footprint.room_id, footprint))
            .collect();
        let room_index: BTreeMap<_, _> = topology
            .rooms
            .iter()
            .enumerate()
            .map(|(index, room)| (room.id, index))
            .collect();
        let mut seen_edges = std::collections::BTreeSet::new();
        let mut route_components = UnionFind::new(topology.rooms.len());
        let mut non_tree_edges = 0usize;

        for route in &topology.routes {
            assert_ne!(
                route.source_room, route.target_room,
                "route {} is a self-edge",
                route.id
            );
            assert_eq!(
                route.envelopes.len(),
                1,
                "route {} is not straight",
                route.id
            );
            let source = footprints_by_id[&route.source_room];
            let target = footprints_by_id[&route.target_room];
            assert_eq!(
                source.layer, target.layer,
                "route {} crosses layers",
                route.id
            );
            let edge = (
                route.source_room.min(route.target_room),
                route.source_room.max(route.target_room),
            );
            assert!(
                seen_edges.insert(edge),
                "route {} duplicates {edge:?}",
                route.id
            );
            if !route_components.union(
                room_index[&route.source_room],
                room_index[&route.target_room],
            ) {
                non_tree_edges += 1;
            }

            let portal_matches: Vec<_> = topology
                .portals
                .iter()
                .filter(|portal| {
                    portal.source_room == route.source_room
                        && portal.target_room == Some(route.target_room)
                })
                .collect();
            assert_eq!(
                portal_matches.len(),
                1,
                "route {} lacks exactly one portal",
                route.id
            );
            let portal = portal_matches[0];
            assert_eq!((portal.width, portal.height), (64, 80));
            let direction = dir_from_tag(&portal.wall);
            let source_span = cardinal_wall_span(source, direction).unwrap();
            let target_span = cardinal_wall_span(target, direction.opposite()).unwrap();
            let center = if matches!(direction, Dir::West | Dir::East) {
                portal.anchor.1
            } else {
                portal.anchor.0
            };
            let aperture = (center - ROUTE_WIDTH / 2, center + ROUTE_WIDTH / 2);
            assert!(aperture.0 >= source_span.0 && aperture.1 <= source_span.1);
            assert!(aperture.0 >= target_span.0 && aperture.1 <= target_span.1);
            assert_eq!(
                portal.anchor.2 - HEADROOM / 2,
                source.floor_z + CONSTRUCTION_QUANTUM
            );
            assert_eq!(
                portal.anchor.2 + HEADROOM / 2,
                source.floor_z + CONSTRUCTION_QUANTUM + HEADROOM
            );
            match direction {
                Dir::East => assert_eq!(portal.anchor.0, source.aabb.2),
                Dir::West => assert_eq!(portal.anchor.0, source.aabb.0),
                Dir::North => assert_eq!(portal.anchor.1, source.aabb.1),
                Dir::South => assert_eq!(portal.anchor.1, source.aabb.3),
            }

            let (x0, y0, x1, y1) = route.envelopes[0];
            assert!(x0 < x1 && y0 < y1);
            match direction {
                Dir::East => {
                    assert!(x0 <= source.aabb.2 && x1 >= target.aabb.0);
                    assert_eq!((y0, y1), aperture);
                }
                Dir::West => {
                    assert!(x0 <= target.aabb.2 && x1 >= source.aabb.0);
                    assert_eq!((y0, y1), aperture);
                }
                Dir::North => {
                    assert!(y0 <= target.aabb.3 && y1 >= source.aabb.1);
                    assert_eq!((x0, x1), aperture);
                }
                Dir::South => {
                    assert!(y0 <= source.aabb.3 && y1 >= target.aabb.1);
                    assert_eq!((x0, x1), aperture);
                }
            }
            for third in footprints {
                if third.room_id == route.source_room
                    || third.room_id == route.target_room
                    || third.layer != source.layer
                {
                    continue;
                }
                assert!(
                    !(x0 < third.aabb.2
                        && x1 > third.aabb.0
                        && y0 < third.aabb.3
                        && y1 > third.aabb.1),
                    "route {} intersects unrelated room {}",
                    route.id,
                    third.room_id
                );
            }
        }
        assert_eq!(
            non_tree_edges, target_loops,
            "loop surplus must be true non-tree edges"
        );

        let transition = &topology.transitions[0];
        let lower = footprints_by_id[&transition.lower_room];
        let upper = footprints_by_id[&transition.upper_room];
        assert_eq!(lower.layer, 0);
        assert_eq!(upper.layer, 1);
        let (x0, y0, z0, x1, y1, z1) = transition.protected_volume;
        assert!(x0 < x1 && y0 < y1 && z0 < z1);
        assert_eq!(x1 - x0, ROUTE_WIDTH);
        for coordinate in [x0, y0, z0, x1, y1, z1] {
            assert_eq!(coordinate % CONSTRUCTION_QUANTUM, 0);
        }
        assert!(y0 >= lower.aabb.3 && y1 <= upper.aabb.1);
        assert!(transition
            .headroom_volumes
            .iter()
            .all(|volume| *volume == transition.protected_volume));
        assert!(!transition.headroom_volumes.is_empty());
        let (llx0, lly0, llx1, lly1) = transition.lower_landing;
        let (ulx0, uly0, ulx1, uly1) = transition.upper_landing;
        assert!(
            llx0 >= lower.aabb.0
                && llx1 <= lower.aabb.2
                && lly0 >= lower.aabb.1
                && lly1 <= lower.aabb.3
        );
        assert!(
            ulx0 >= upper.aabb.0
                && ulx1 <= upper.aabb.2
                && uly0 >= upper.aabb.1
                && uly1 <= upper.aabb.3
        );

        let mut global_components = route_components;
        assert!(global_components.union(
            room_index[&transition.lower_room],
            room_index[&transition.upper_room]
        ));
        let root = global_components.find(0);
        for index in 1..topology.rooms.len() {
            assert_eq!(
                global_components.find(index),
                root,
                "room {} is disconnected",
                topology.rooms[index].id
            );
        }

        let (spawn, _) = compute_reservations(topology).unwrap();
        let spawn_host = select_spawn_footprint(footprints).unwrap();
        for corner in [
            (spawn.x0, spawn.y0),
            (spawn.x0, spawn.y1),
            (spawn.x1, spawn.y0),
            (spawn.x1, spawn.y1),
        ] {
            assert!(
                footprint_contains_strictly(spawn_host, corner),
                "spawn corner {corner:?} leaves convex host"
            );
        }
        assert_eq!(spawn.z0, spawn_host.floor_z + CONSTRUCTION_QUANTUM);
        assert_eq!(spawn.height(), HEADROOM);
    }

    // ── Basic structure ───────────────────────────────────────────────

    #[test]
    fn build_sparse_topology() {
        let (topology, _, config) =
            build_topology_for(super::super::config::V3Preset::Sparse, 0, 2048);
        let n = config.preset.min_rooms() as usize;
        assert_eq!(topology.rooms.len(), n);
        assert_eq!(topology.surfaces.len(), n);
        assert!(!topology.portals.is_empty());
        assert!(!topology.routes.is_empty());
        assert_eq!(topology.transitions.len(), 1);
    }

    #[test]
    fn build_moderate_topology() {
        let (topology, _, config) =
            build_topology_for(super::super::config::V3Preset::Moderate, 0, 2048);
        assert_eq!(topology.rooms.len(), config.preset.min_rooms() as usize);
        assert_eq!(topology.transitions.len(), 1);
    }

    #[test]
    fn build_rich_topology() {
        let (topology, _, config) =
            build_topology_for(super::super::config::V3Preset::Rich, 0, 3072);
        assert_eq!(topology.rooms.len(), config.preset.min_rooms() as usize);
        assert_eq!(topology.transitions.len(), 1);
    }

    // ── Edge counts ───────────────────────────────────────────────────

    #[test]
    fn route_count_matches_formula() {
        for (preset, extent) in &[
            (super::super::config::V3Preset::Sparse, 2048u32),
            (super::super::config::V3Preset::Moderate, 2048),
            (super::super::config::V3Preset::Rich, 3072),
        ] {
            let config = V3Config::new(0, *preset, *extent).unwrap();
            let rooms = config.preset.min_rooms() as usize;
            let target_loops = config.preset.target_loops() as usize;
            let expected_routes = (rooms - 2) + target_loops;

            let (topology, _, _) = build_topology_for(*preset, 0, *extent);
            assert_eq!(
                topology.routes.len(),
                expected_routes,
                "route count mismatch for {:?} at {}: expected {}, got {}",
                preset,
                extent,
                expected_routes,
                topology.routes.len()
            );
        }
    }

    #[test]
    fn portals_equal_routes() {
        for (preset, extent) in &[
            (super::super::config::V3Preset::Sparse, 2048u32),
            (super::super::config::V3Preset::Moderate, 2048),
            (super::super::config::V3Preset::Rich, 3072),
        ] {
            let (topology, _, _) = build_topology_for(*preset, 0, *extent);
            assert_eq!(
                topology.portals.len(),
                topology.routes.len(),
                "portals != routes for {:?} at {}",
                preset,
                extent
            );
        }
    }

    #[test]
    fn transition_count_is_one() {
        for (preset, extent) in &[
            (super::super::config::V3Preset::Sparse, 2048u32),
            (super::super::config::V3Preset::Moderate, 2048),
            (super::super::config::V3Preset::Rich, 3072),
        ] {
            let (topology, _, _) = build_topology_for(*preset, 0, *extent);
            assert_eq!(topology.transitions.len(), 1);
        }
    }

    // ── Graph reachability ────────────────────────────────────────────

    #[test]
    fn all_rooms_reachable() {
        for (preset, extent) in &[
            (super::super::config::V3Preset::Sparse, 2048u32),
            (super::super::config::V3Preset::Moderate, 2048),
            (super::super::config::V3Preset::Rich, 3072),
        ] {
            let (topology, _, _) = build_topology_for(*preset, 0, *extent);

            let n = topology.rooms.len();
            let mut uf = UnionFind::new(n);

            // Map room IDs to indices.
            let id_to_idx: BTreeMap<_, usize> = topology
                .rooms
                .iter()
                .enumerate()
                .map(|(i, r)| (r.id, i))
                .collect();

            // Connect via routes (same-layer edges).
            for route in &topology.routes {
                let si = id_to_idx[&route.source_room];
                let ti = id_to_idx[&route.target_room];
                uf.union(si, ti);
            }

            // Connect via transition (inter-layer edge).
            for trans in &topology.transitions {
                let li = id_to_idx[&trans.lower_room];
                let ui = id_to_idx[&trans.upper_room];
                uf.union(li, ui);
            }

            // All rooms should be in the same component.
            let root = uf.find(0);
            for i in 1..n {
                assert_eq!(
                    uf.find(i),
                    root,
                    "room {} ({}) not reachable in {:?} at {}",
                    i,
                    topology.rooms[i].id,
                    preset,
                    extent
                );
            }
        }
    }

    #[test]
    fn no_isolated_room() {
        for (preset, extent) in &[
            (super::super::config::V3Preset::Sparse, 2048u32),
            (super::super::config::V3Preset::Moderate, 2048),
            (super::super::config::V3Preset::Rich, 3072),
        ] {
            let (topology, _, _) = build_topology_for(*preset, 0, *extent);

            let mut connected: std::collections::BTreeSet<_> = std::collections::BTreeSet::new();
            for route in &topology.routes {
                connected.insert(route.source_room);
                connected.insert(route.target_room);
            }
            for trans in &topology.transitions {
                connected.insert(trans.lower_room);
                connected.insert(trans.upper_room);
            }
            for room in &topology.rooms {
                assert!(
                    connected.contains(&room.id),
                    "room {} is isolated in {:?} at {}",
                    room.id,
                    preset,
                    extent
                );
            }
        }
    }

    // ── Loop surplus ──────────────────────────────────────────────────

    #[test]
    fn loop_surplus_matches_target() {
        for (preset, extent) in &[
            (super::super::config::V3Preset::Sparse, 2048u32),
            (super::super::config::V3Preset::Moderate, 2048),
            (super::super::config::V3Preset::Rich, 3072),
        ] {
            let config = V3Config::new(0, *preset, *extent).unwrap();
            let (topology, _, _) = build_topology_for(*preset, 0, *extent);

            let target_loops = config.preset.target_loops() as usize;

            // Total edges = (rooms - 2 + loops) + 1 transition — but
            // transition isn't a same-layer edge. The loop surplus is measured
            // against the spanning tree baseline per layer.
            let lower_count = topology.rooms.iter().filter(|r| r.layer == 0).count();
            let upper_count = topology.rooms.len() - lower_count;
            let tree_edges = (lower_count - 1) + (upper_count - 1);
            let loop_surplus = topology.routes.len() - tree_edges;
            assert_eq!(
                loop_surplus, target_loops,
                "loop surplus mismatch for {:?} at {}: expected {}, got {}",
                preset, extent, target_loops, loop_surplus
            );
        }
    }

    // ── No duplicate edges ────────────────────────────────────────────

    #[test]
    fn no_duplicate_edges() {
        for (preset, extent) in &[
            (super::super::config::V3Preset::Sparse, 2048u32),
            (super::super::config::V3Preset::Moderate, 2048),
            (super::super::config::V3Preset::Rich, 3072),
        ] {
            let (topology, _, _) = build_topology_for(*preset, 0, *extent);

            let mut seen: std::collections::BTreeSet<(u32, u32)> =
                std::collections::BTreeSet::new();
            for route in &topology.routes {
                let a = route.source_room.raw();
                let b = route.target_room.raw();
                let key = if a < b { (a, b) } else { (b, a) };
                assert!(
                    seen.insert(key),
                    "duplicate edge ({a}, {b}) in {:?} at {}",
                    preset,
                    extent
                );
            }
        }
    }

    // ── Portal dimensions and cardinality ─────────────────────────────

    #[test]
    fn portal_dimensions_are_64_by_80() {
        for (preset, extent) in &[
            (super::super::config::V3Preset::Sparse, 2048u32),
            (super::super::config::V3Preset::Moderate, 2048),
            (super::super::config::V3Preset::Rich, 3072),
        ] {
            let (topology, _, _) = build_topology_for(*preset, 0, *extent);
            for portal in &topology.portals {
                assert_eq!(portal.width, 64, "portal width not 64 for {:?}", preset);
                assert_eq!(portal.height, 80, "portal height not 80 for {:?}", preset);
            }
        }
    }

    #[test]
    fn portal_wall_is_cardinal() {
        for (preset, extent) in &[
            (super::super::config::V3Preset::Sparse, 2048u32),
            (super::super::config::V3Preset::Moderate, 2048),
            (super::super::config::V3Preset::Rich, 3072),
        ] {
            let (topology, _, _) = build_topology_for(*preset, 0, *extent);
            for portal in &topology.portals {
                assert!(
                    matches!(portal.wall.as_str(), "north" | "south" | "west" | "east"),
                    "portal wall '{}' not cardinal for {:?}",
                    portal.wall,
                    preset
                );
            }
        }
    }

    #[test]
    fn portal_interior_span_at_least_64() {
        for (preset, extent) in &[
            (super::super::config::V3Preset::Sparse, 2048u32),
            (super::super::config::V3Preset::Moderate, 2048),
            (super::super::config::V3Preset::Rich, 3072),
        ] {
            let (topology, footprints, _) = build_topology_for(*preset, 0, *extent);

            // Build a lookup of footprint by room ID.
            let fp_by_id: BTreeMap<_, &Footprint> =
                footprints.iter().map(|fp| (fp.room_id, fp)).collect();

            for portal in &topology.portals {
                let src_fp = fp_by_id[&portal.source_room];
                if let Some(tgt_id) = portal.target_room {
                    let tgt_fp = fp_by_id[&tgt_id];

                    // The cross-axis span should be at least 64.
                    let cross_span = match portal.wall.as_str() {
                        "east" | "west" => {
                            // Cross-axis is Y.
                            let lo = src_fp.aabb.1.max(tgt_fp.aabb.1);
                            let hi = src_fp.aabb.3.min(tgt_fp.aabb.3);
                            hi - lo
                        }
                        "north" | "south" => {
                            // Cross-axis is X.
                            let lo = src_fp.aabb.0.max(tgt_fp.aabb.0);
                            let hi = src_fp.aabb.2.min(tgt_fp.aabb.2);
                            hi - lo
                        }
                        _ => 0,
                    };
                    assert!(
                        cross_span >= 64,
                        "portal {} has cross-span {cross_span} < 64 for {:?}",
                        portal.id,
                        preset
                    );
                }
            }
        }
    }

    // ── Route no-third-room overlap ───────────────────────────────────

    #[test]
    fn route_envelope_does_not_intersect_third_room() {
        for (preset, extent) in &[
            (super::super::config::V3Preset::Sparse, 2048u32),
            (super::super::config::V3Preset::Moderate, 2048),
            (super::super::config::V3Preset::Rich, 3072),
        ] {
            let (topology, footprints, _) = build_topology_for(*preset, 0, *extent);

            let fp_by_id: BTreeMap<_, &Footprint> =
                footprints.iter().map(|fp| (fp.room_id, fp)).collect();

            for route in &topology.routes {
                for &(ex0, ey0, ex1, ey1) in &route.envelopes {
                    for fp in &footprints {
                        if fp.room_id == route.source_room || fp.room_id == route.target_room {
                            continue;
                        }
                        if fp.layer != fp_by_id[&route.source_room].layer {
                            continue;
                        }
                        let (rx0, ry0, rx1, ry1) = fp.aabb;
                        // The envelope may overlap the source/target room's
                        // own walls (it's trimmed later in assembly). We
                        // only check that it doesn't overlap a third room's
                        // AABB.
                        assert!(
                            !(ex0 < rx1 && ex1 > rx0 && ey0 < ry1 && ey1 > ry0),
                            "route {} envelope {ex0},{ey0},{ex1},{ey1} overlaps room {} for {:?}",
                            route.id,
                            fp.room_id,
                            preset
                        );
                    }
                }
            }
        }
    }

    // ── Deterministic replay ──────────────────────────────────────────

    #[test]
    fn topology_deterministic() {
        for seed_val in [0u64, 42, 99, 255] {
            for (preset, extent) in &[
                (super::super::config::V3Preset::Sparse, 2048u32),
                (super::super::config::V3Preset::Moderate, 2048),
                (super::super::config::V3Preset::Rich, 3072),
            ] {
                let config = V3Config::new(seed_val, *preset, *extent).unwrap();
                let mut alloc1 = V3IdAllocator::new();
                let mut alloc2 = V3IdAllocator::new();

                let (fp1, lo1) =
                    build_footprints(&config, V3Seed::new(seed_val), &mut alloc1).unwrap();
                let (fp2, lo2) =
                    build_footprints(&config, V3Seed::new(seed_val), &mut alloc2).unwrap();

                let t1 = build_topology(&config, &fp1, &lo1, V3Seed::new(seed_val), &mut alloc1)
                    .unwrap();
                let t2 = build_topology(&config, &fp2, &lo2, V3Seed::new(seed_val), &mut alloc2)
                    .unwrap();

                assert_eq!(t1.rooms.len(), t2.rooms.len());
                assert_eq!(t1.portals.len(), t2.portals.len());
                assert_eq!(t1.routes.len(), t2.routes.len());
                assert_eq!(t1.transitions.len(), t2.transitions.len());
                // Full structural equality.
                assert_eq!(
                    t1, t2,
                    "determinism violated for seed {seed_val} {:?} at {extent}",
                    preset
                );
            }
        }
    }

    // ── Seed topology variation ───────────────────────────────────────

    #[test]
    fn different_seeds_produce_different_topology() {
        for (preset, extent) in &[
            (super::super::config::V3Preset::Sparse, 2048u32),
            (super::super::config::V3Preset::Moderate, 2048),
            (super::super::config::V3Preset::Rich, 3072),
        ] {
            let (t0, _, _) = build_topology_for(*preset, 0, *extent);
            let (t42, _, _) = build_topology_for(*preset, 42, *extent);

            assert_ne!(
                t0, t42,
                "topology seed stream had no influence for {preset:?} at {extent}"
            );
        }
    }

    // ── All presets, all seeds, all extents ───────────────────────────

    #[test]
    fn all_presets_seeds_0_42_99_255_extents_1024_2048_3072() {
        for preset in [
            super::super::config::V3Preset::Sparse,
            super::super::config::V3Preset::Moderate,
            super::super::config::V3Preset::Rich,
        ] {
            for &extent in &[1024u32, 2048, 3072] {
                let mut seed_signatures = std::collections::BTreeSet::new();
                for seed_val in [0u64, 42, 99, 255] {
                    let config = V3Config::new(seed_val, preset, extent).unwrap();
                    let mut alloc = V3IdAllocator::new();
                    let (footprints, layout) =
                        build_footprints(&config, V3Seed::new(seed_val), &mut alloc).unwrap();
                    let topology = build_topology(
                        &config,
                        &footprints,
                        &layout,
                        V3Seed::new(seed_val),
                        &mut alloc,
                    )
                    .unwrap_or_else(|error| {
                        panic!(
                            "topology failed for seed {seed_val} {preset:?} at {extent}: {error:?}"
                        )
                    });
                    assert_matrix_topology_contract(&config, &footprints, &topology);
                    seed_signatures
                        .insert(format!("{:?}{:?}", topology.routes, topology.transitions));
                }
                assert!(
                    seed_signatures.len() > 1,
                    "topology seed stream had no influence for {preset:?} at {extent}"
                );
            }
        }
    }

    // ── Safe large spawn selection ────────────────────────────────────

    #[test]
    fn spawn_is_largest_lower_room() {
        for (preset, extent) in &[
            (super::super::config::V3Preset::Sparse, 2048u32),
            (super::super::config::V3Preset::Moderate, 2048),
            (super::super::config::V3Preset::Rich, 3072),
        ] {
            let (topology, _, _) = build_topology_for(*preset, 0, *extent);

            let (spawn_volume, _) = compute_reservations(&topology).unwrap();

            let lower_rooms: Vec<&CommittedRoom> =
                topology.rooms.iter().filter(|r| r.layer == 0).collect();

            let largest = lower_rooms
                .iter()
                .max_by(|left, right| {
                    room_area(left)
                        .cmp(&room_area(right))
                        .then_with(|| right.id.cmp(&left.id))
                })
                .unwrap();

            // Spawn center should be inside the largest lower room's AABB.
            let scx = (spawn_volume.x0 + spawn_volume.x1) / 2;
            let scy = (spawn_volume.y0 + spawn_volume.y1) / 2;
            assert!(
                scx >= largest.shell.0 && scx <= largest.shell.2,
                "spawn X {scx} outside largest room [{}, {}]",
                largest.shell.0,
                largest.shell.2
            );
            assert!(
                scy >= largest.shell.1 && scy <= largest.shell.3,
                "spawn Y {scy} outside largest room [{}, {}]",
                largest.shell.1,
                largest.shell.3
            );
        }
    }

    #[test]
    fn spawn_not_in_first_tiny_room() {
        // The first room by ID is often small. Spawn should be in the
        // largest lower room, which may or may not be the first room.
        for (preset, extent) in &[
            (super::super::config::V3Preset::Sparse, 2048u32),
            (super::super::config::V3Preset::Moderate, 2048),
            (super::super::config::V3Preset::Rich, 3072),
        ] {
            let (topology, _, _) = build_topology_for(*preset, 42, *extent);

            let (spawn_volume, _) = compute_reservations(&topology).unwrap();

            let largest_lower = topology
                .rooms
                .iter()
                .filter(|room| room.layer == 0)
                .max_by(|left, right| {
                    room_area(left)
                        .cmp(&room_area(right))
                        .then_with(|| right.id.cmp(&left.id))
                })
                .unwrap();

            let scx = (spawn_volume.x0 + spawn_volume.x1) / 2;
            let scy = (spawn_volume.y0 + spawn_volume.y1) / 2;

            // Spawn center is inside the largest lower room.
            let margin = CONSTRUCTION_QUANTUM;
            assert!(
                scx > largest_lower.shell.0 + margin && scx < largest_lower.shell.2 - margin,
                "spawn too close to wall in X"
            );
            assert!(
                scy > largest_lower.shell.1 + margin && scy < largest_lower.shell.3 - margin,
                "spawn too close to wall in Y"
            );
        }
    }

    // ── Bounds within config ──────────────────────────────────────────

    #[test]
    fn topology_bounds_within_config() {
        for (preset, extent) in &[
            (super::super::config::V3Preset::Sparse, 2048u32),
            (super::super::config::V3Preset::Moderate, 2048),
            (super::super::config::V3Preset::Rich, 3072),
        ] {
            let (topology, _, config) = build_topology_for(*preset, 0, *extent);
            for room in &topology.rooms {
                assert!(room.shell.2 <= config.xy_extent as i32);
                assert!(room.shell.3 <= config.xy_extent as i32);
            }
        }
    }

    // ── Reservations ──────────────────────────────────────────────────

    #[test]
    fn spawn_and_light_reservations() {
        for (preset, extent) in &[
            (super::super::config::V3Preset::Sparse, 2048u32),
            (super::super::config::V3Preset::Moderate, 2048),
            (super::super::config::V3Preset::Rich, 3072),
        ] {
            let (topology, _, _) = build_topology_for(*preset, 0, *extent);
            let (spawn, lights) = compute_reservations(&topology).unwrap();
            assert!(spawn.width() > 0);
            assert!(spawn.depth() > 0);
            assert!(spawn.height() > 0);
            assert_eq!(lights.len(), topology.rooms.len());
            for light in &lights {
                assert!(light.width() > 0);
                assert!(light.height() > 0);
            }
        }
    }

    // ── Transition validity ───────────────────────────────────────────

    #[test]
    fn transition_has_valid_landings() {
        for (preset, extent) in &[
            (super::super::config::V3Preset::Sparse, 2048u32),
            (super::super::config::V3Preset::Moderate, 2048),
            (super::super::config::V3Preset::Rich, 3072),
        ] {
            let (topology, _, _) = build_topology_for(*preset, 0, *extent);

            let trans = &topology.transitions[0];
            let lower = topology.room(trans.lower_room).unwrap();
            let upper = topology.room(trans.upper_room).unwrap();

            assert_eq!(lower.layer, 0);
            assert_eq!(upper.layer, 1);

            // Protected volume must be non-degenerate.
            let (pvx0, pvy0, pvz0, pvx1, pvy1, pvz1) = trans.protected_volume;
            assert!(pvx0 < pvx1);
            assert!(pvy0 < pvy1);
            assert!(pvz0 < pvz1);

            // Lower landing is inside lower room.
            let (llx0, lly0, llx1, lly1) = trans.lower_landing;
            assert!(llx0 >= lower.shell.0 && llx1 <= lower.shell.2);
            assert!(lly0 >= lower.shell.1 && lly1 <= lower.shell.3);

            // Upper landing is inside upper room.
            let (ulx0, uly0, ulx1, uly1) = trans.upper_landing;
            assert!(ulx0 >= upper.shell.0 && ulx1 <= upper.shell.2);
            assert!(uly0 >= upper.shell.1 && uly1 <= upper.shell.3);
        }
    }

    // ── Every route has a matching portal ─────────────────────────────

    #[test]
    fn every_route_has_matching_portal() {
        for (preset, extent) in &[
            (super::super::config::V3Preset::Sparse, 2048u32),
            (super::super::config::V3Preset::Moderate, 2048),
            (super::super::config::V3Preset::Rich, 3072),
        ] {
            let (topology, _, _) = build_topology_for(*preset, 0, *extent);

            for route in &topology.routes {
                let found = topology.portals.iter().any(|p| {
                    p.source_room == route.source_room && p.target_room == Some(route.target_room)
                });
                assert!(
                    found,
                    "route {} has no matching portal for {:?}",
                    route.id, preset
                );
            }
        }
    }
}
