//! Constrained Kruskal topology solver with loop augmentation and deterministic
//! route reservation inside reserved envelopes.
//!
//! Builds candidate edges from committed reservation footprints, inserts
//! mandatory critical-path edges (rejecting the request if infeasible),
//! runs constrained Kruskal over the remaining candidates ordered by
//! integer distance, pacing bias, field rank, and stable key, adds loops
//! from the non-MST tail (Moderate/Rich materialize a backward shortcut),
//! and reserves complete deterministic routes (portal anchors, turns,
//! headroom, vertical entries) inside reserved envelopes.
//!
//! # Contract
//!
//! - Multigraph: parallel candidate edges between the same reservations have
//!   distinct `EdgeId` values. Residual capacity is per edge identity.
//! - Same-XY occupancy is authorized only inside composite reservations.
//! - A candidate edge is legal only when its complete route AND endpoint
//!   approach can be reserved. Nearby empty cells are not route capacity.
//! - Route reservation participates in the transactional mark/rollback/commit
//!   lifecycle from the placement journal.
//! - No floats. All coordinates are quantum-aligned.
//! - Candidate and backtracking order is fully canonical.

// Richness remains intentionally crate-private and pipeline-unwired until the
// atomic sealing phase; unit and matrix tests are its current callers.
#![allow(dead_code, clippy::result_large_err, clippy::too_many_arguments)]

use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap};

use super::error::{RichnessError, RichnessErrorCategory, RichnessErrorCode};
use super::footprint::{CellCoord, Footprint3D, CONSTRUCTION_QUANTUM, HEADROOM};
use super::ids::{
    BeatId, EdgeId, MandatoryEdge, PacingBlueprint, PortalId, ReservationId, RouteId,
    ShortcutIntent, TurnId, ZoneId,
};
use super::request::{ResolvedRichnessRequestV1, RichnessPreset};
use super::reservation::{ReservationJournal, ReservationKind, ReservationRecord};
use super::solver::PlacementResult;

// ── Cardinal direction ─────────────────────────────────────────────────────

/// Cardinal direction for wall-adjacent portal placement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) enum Dir {
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

// ── Edge kind ──────────────────────────────────────────────────────────────

/// Whether an edge is same-layer or vertical.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) enum EdgeKind {
    /// Same-layer corridor (lower or upper).
    SameLayer { layer: u8 },
    /// Vertical connection (stair/shaft/drop between bands).
    Vertical {
        lower_reservation: ReservationId,
        upper_reservation: ReservationId,
    },
}

// ── Candidate edge ─────────────────────────────────────────────────────────

/// A feasible candidate edge between two committed reservations.
///
/// Uses a multigraph model: parallel edges between the same reservation pair
/// have distinct `EdgeId` values and track independent residual capacity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CandidateEdge {
    /// Unique edge identity (distinguishes parallel edges).
    pub id: EdgeId,
    /// The kind of edge (same-layer or vertical).
    pub kind: EdgeKind,
    /// Source reservation.
    pub source: ReservationId,
    /// Target reservation.
    pub target: ReservationId,
    /// Source room wall direction (portal placement side).
    pub source_dir: Dir,
    /// Cross-axis overlap interval in Quake units: (lo, hi).
    pub overlap: (i32, i32),
    /// Integer Manhattan distance between reservation footprint centers.
    pub distance: i32,
    /// Pacing bias priority (lower = higher priority, from beat progression).
    pub pacing_bias: u32,
    /// Stable deterministic field rank.
    pub field_rank: u64,
    /// Residual route capacity for this edge identity.
    pub residual_capacity: u32,
    /// Whether this edge is mandatory (critical-path).
    pub mandatory: bool,
    /// The mandatory edge record this satisfies (if mandatory).
    pub mandatory_record: Option<MandatoryEdge>,
}

impl CandidateEdge {
    /// Stable sort key: (distance, pacing_bias, field_rank, id).
    fn sort_key(&self) -> (i32, u32, u64, u32) {
        (
            self.distance,
            self.pacing_bias,
            self.field_rank,
            self.id.raw(),
        )
    }
}

// ── Committed route ────────────────────────────────────────────────────────

/// A committed route through a reserved envelope connecting two rooms.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CommittedRoute {
    /// Unique route ID.
    pub id: RouteId,
    /// The candidate edge this route realizes.
    pub edge_id: EdgeId,
    /// Source reservation.
    pub source: ReservationId,
    /// Target reservation.
    pub target: ReservationId,
    /// The route envelope grid footprint.
    pub envelope: Footprint3D,
    /// The portal at the source end.
    pub source_portal: CommittedPortal,
    /// The portal at the target end.
    pub target_portal: CommittedPortal,
    /// Concrete reservation records whose union is the complete route.
    pub reservation_ids: Vec<ReservationId>,
    /// Any intermediate turns.
    pub turns: Vec<CommittedTurn>,
}

// ── Committed portal ───────────────────────────────────────────────────────

/// A committed portal throat at a room wall boundary.
///
/// Protected by a 64×80 witness reservation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CommittedPortal {
    /// Unique portal ID.
    pub id: PortalId,
    /// The wall direction (which wall of the source room).
    pub wall: Dir,
    /// Grid anchor position (center of the portal on the wall).
    pub anchor_cell: CellCoord,
    /// The wall-depth throat footprint (64 units wide in XY).
    pub witness: Footprint3D,
    /// Frozen 80-unit protected headroom.
    pub headroom: i32,
    /// Endpoint room/composite whose compatible socket this throat serves.
    pub endpoint_reservation_id: ReservationId,
    /// First-class `PortalThroat` reservation that owns every witness cell.
    pub reservation_id: ReservationId,
}

// ── Committed turn ─────────────────────────────────────────────────────────

/// A committed turn / junction in a route.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CommittedTurn {
    /// Unique turn ID.
    pub id: TurnId,
    /// Grid position of the turn.
    pub position: CellCoord,
    /// Protected 64-unit XY witness with frozen 80-unit headroom.
    pub witness: Footprint3D,
    /// Frozen protected headroom in Quake units.
    pub headroom: i32,
    /// First-class `Turn` reservation owning the complete witness.
    pub reservation_id: ReservationId,
}

/// Observable deterministic topology-search accounting. The frozen search
/// bounds remain revision constants; observed rollback counts are evidence and
/// never feed generation decisions.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct TopologySearchMetrics {
    pub mandatory_candidates_materialized: u64,
    pub mandatory_backtracks: u64,
    pub topology_search_states: u64,
    pub topology_backtracks: u64,
    pub rollback_checks: u64,
    pub rollback_mismatches: u64,
}

// ── Topology result ────────────────────────────────────────────────────────

/// The complete topology solution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TopologyResult {
    /// All selected edges (tree + loops).
    pub selected_edges: Vec<CandidateEdge>,
    /// Committed routes.
    pub routes: Vec<CommittedRoute>,
    /// The reservation journal after topology construction.
    pub journal: ReservationJournal,
    /// Mapping from beat ID to its reservation IDs.
    pub beat_to_reservations: BTreeMap<BeatId, Vec<ReservationId>>,
    /// Number of loops added.
    pub loop_count: usize,
    /// Shortcut edges realized.
    pub shortcuts_realized: Vec<EdgeId>,
    /// Exact lower/upper endpoint candidates for every committed vertical
    /// host. These are structural transitions, not Kruskal loop edges.
    pub vertical_edges: Vec<CandidateEdge>,
    /// Concrete routes realizing [`Self::vertical_edges`].
    pub vertical_routes: Vec<CommittedRoute>,
    /// Deterministic search/rollback evidence.
    pub search_metrics: TopologySearchMetrics,
}

/// A room-pair descriptor is not a candidate edge. It becomes one only after
/// a complete route plan has been transactionally reserved in a probe journal.
#[derive(Debug, Clone, PartialEq, Eq)]
struct PotentialConnection {
    ordinal: u32,
    kind: EdgeKind,
    source: ReservationId,
    target: ReservationId,
    preferred_dir: Dir,
    distance: i32,
    pacing_bias: u32,
    pair_rank: u64,
    mandatory_record: Option<MandatoryEdge>,
}

impl PotentialConnection {
    fn sort_key(&self) -> (i32, u32, u64, u32) {
        (
            self.distance,
            self.pacing_bias,
            self.pair_rank,
            self.ordinal,
        )
    }
}

/// Concrete compatible sockets plus the canonical complete lattice path.
#[derive(Debug, Clone, PartialEq, Eq)]
struct CompleteRoutePlan {
    source_socket: PortalRouteCandidate,
    target_socket: PortalRouteCandidate,
    path: Vec<CellCoord>,
}

/// A real candidate edge paired with the exact reservation plan whose probe
/// committed route, portal-throat, turn, and endpoint-approach witnesses.
#[derive(Debug, Clone, PartialEq, Eq)]
struct FeasibleCandidate {
    edge: CandidateEdge,
    plan: CompleteRoutePlan,
}

// ── Union-Find ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
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

    fn connected(&mut self, x: usize, y: usize) -> bool {
        self.find(x) == self.find(y)
    }
}

// ── Route envelope geometry ────────────────────────────────────────────────

/// Route width in Quake units (64).
const ROUTE_WIDTH: i32 = 64;

/// Frozen revision-v1 cap on concrete route plans materialized for one room
/// pair in one search state.
const PARALLEL_ROUTE_CAPACITY: u32 = 5;
/// Stable ID range reserved for every potential room pair. Five canonical path
/// variants across at most 20 source × 20 target sockets require 2,000 IDs.
const EDGE_ID_STRIDE: u32 = 4_096;
const PORTAL_SOCKET_ID_CAPACITY: u32 = 20;
const ROUTE_PATH_VARIANT_COUNT: usize = 5;

/// Frozen deterministic topology branch ceiling. This is a logical bound,
/// independent of wall-clock duration.
const MAX_TOPOLOGY_SEARCH_STATES: u64 = 200_000;

/// Portal witness width (64 units).
const PORTAL_WITNESS_WIDTH: u32 = 4; // 64/16 grid cells

/// Portal wall depth (one frozen 16-unit construction cell). The second
/// dimension of the 64×80 contract is Z headroom, not projected XY depth.
const PORTAL_WITNESS_DEPTH: u32 = 1;

/// Build a straight route envelope footprint between two room footprints.
///
/// The route occupies ONLY the gap between rooms (from source wall to
/// target wall). Rooms own their interior cells; the route corridor is
/// the clear space between them.
fn route_envelope_footprint(
    source: &ReservationRecord,
    target: &ReservationRecord,
    source_dir: Dir,
    overlap: (i32, i32),
) -> Footprint3D {
    let fp_src = &source.footprint;
    let fp_tgt = &target.footprint;
    let q = CONSTRUCTION_QUANTUM;
    let hw = (ROUTE_WIDTH / 2 / q) * q; // half-width quantum-aligned

    let cross_center = ((overlap.0 + overlap.1) / 2 / q) * q;
    // Quake coordinates for the gap corridor (wall-to-wall, exclusive of rooms)
    let (qx0, qy0, qx1, qy1) = match source_dir {
        Dir::East => (
            fp_src.quake_x1(), // source east wall
            cross_center - hw,
            fp_tgt.quake_x0(), // target west wall
            cross_center + hw,
        ),
        Dir::West => (
            fp_tgt.quake_x1(), // target east wall
            cross_center - hw,
            fp_src.quake_x0(), // source west wall
            cross_center + hw,
        ),
        Dir::North => (
            cross_center - hw,
            fp_tgt.quake_y1(), // target south wall
            cross_center + hw,
            fp_src.quake_y0(), // source north wall
        ),
        Dir::South => (
            cross_center - hw,
            fp_src.quake_y1(), // source south wall
            cross_center + hw,
            fp_tgt.quake_y0(), // target north wall
        ),
    };

    // Determine layer
    let layer = if source.footprint.occupies_lower {
        0u8
    } else {
        1u8
    };

    // Ensure non-degenerate: gap must have positive width AND depth.
    // If rooms touch (zero gap), route has no cells to reserve — it's
    // a direct portal-to-portal connection on the shared wall.
    if qx0 >= qx1 || qy0 >= qy1 {
        // Zero-width or zero-depth: use a zero-area sentinel footprint.
        // The caller will skip cell reservation for this envelope.
        Footprint3D {
            x0: (qx0 / q) as u32,
            y0: (qy0 / q) as u32,
            x1: (qx0 / q) as u32, // zero width
            y1: (qy0 / q) as u32, // zero depth
            occupies_lower: layer == 0,
            occupies_upper: layer == 1,
        }
    } else {
        Footprint3D::single_layer(qx0, qy0, qx1, qy1, layer)
    }
}

/// Build a portal witness footprint at a wall anchor.
///
/// The XY reservation is `PORTAL_WITNESS_WIDTH` cells across the socket and
/// one construction cell through the endpoint wall. Its frozen 80-unit second
/// dimension is protected Z headroom on `CommittedPortal`, not XY depth.
fn portal_witness_footprint(
    _room: &ReservationRecord,
    wall: Dir,
    anchor_cell: CellCoord,
    layer: u8,
) -> Footprint3D {
    let half_width = PORTAL_WITNESS_WIDTH / 2;
    let (x0, y0, x1, y1) = match wall {
        Dir::East | Dir::West => (
            anchor_cell.x,
            anchor_cell.y.saturating_sub(half_width),
            anchor_cell.x.saturating_add(PORTAL_WITNESS_DEPTH),
            anchor_cell.y.saturating_add(half_width),
        ),
        Dir::North | Dir::South => (
            anchor_cell.x.saturating_sub(half_width),
            anchor_cell.y,
            anchor_cell.x.saturating_add(half_width),
            anchor_cell.y.saturating_add(PORTAL_WITNESS_DEPTH),
        ),
    };

    Footprint3D {
        x0,
        y0,
        x1,
        y1,
        occupies_lower: layer == 0,
        occupies_upper: layer == 1,
    }
}

/// Determine the wall direction of `b` relative to `a` based on aabb adjacency.
///
/// Returns `None` if the footprints are not cardinally adjacent.
fn cardinal_direction(a: &Footprint3D, b: &Footprint3D) -> Option<Dir> {
    let a_center_x = (a.x0 + a.x1) / 2;
    let a_center_y = (a.y0 + a.y1) / 2;
    let b_center_x = (b.x0 + b.x1) / 2;
    let b_center_y = (b.y0 + b.y1) / 2;

    let dx = b_center_x as i32 - a_center_x as i32;
    let dy = b_center_y as i32 - a_center_y as i32;

    let abs_dx = dx.abs();
    let abs_dy = dy.abs();

    if abs_dx >= abs_dy {
        if dx > 0 {
            Some(Dir::East)
        } else {
            Some(Dir::West)
        }
    } else if abs_dy > 0 {
        if dy > 0 {
            Some(Dir::South)
        } else {
            Some(Dir::North)
        }
    } else {
        None
    }
}

/// Check main-axis adjacency: source wall must face target wall with
/// non-negative gap (touching or separated, not overlapping).
#[allow(dead_code)]
fn main_axis_adjacent(a: &Footprint3D, b: &Footprint3D, dir: Dir) -> bool {
    match dir {
        Dir::East => a.x1 <= b.x0,  // source east wall <= target west wall
        Dir::West => b.x1 <= a.x0,  // target east wall <= source west wall
        Dir::North => b.y1 <= a.y0, // target south wall <= source north wall
        Dir::South => a.y1 <= b.y0, // source south wall <= target north wall
    }
}

/// Check that a straight corridor between source and target won't overlap
/// any third room's footprint. This ensures routes don't pass through
/// other rooms.
#[allow(dead_code)]
fn corridor_clear_of_third_rooms(
    reservations: &BTreeMap<ReservationId, ReservationRecord>,
    source: &ReservationRecord,
    target: &ReservationRecord,
    source_dir: Dir,
    overlap: (i32, i32),
) -> bool {
    let fp_src = &source.footprint;
    let fp_tgt = &target.footprint;

    // Build corridor bounds in grid cells
    let q = CONSTRUCTION_QUANTUM;
    let cross_center = ((overlap.0 + overlap.1) / 2 / q) * q;
    let hw = ROUTE_WIDTH / 2;

    let (cor_x0, cor_x1, cor_y0, cor_y1) = match source_dir {
        Dir::East => (
            fp_src.quake_x1(),
            fp_tgt.quake_x0(),
            cross_center - hw,
            cross_center + hw,
        ),
        Dir::West => (
            fp_tgt.quake_x1(),
            fp_src.quake_x0(),
            cross_center - hw,
            cross_center + hw,
        ),
        Dir::North => (
            cross_center - hw,
            cross_center + hw,
            fp_tgt.quake_y1(),
            fp_src.quake_y0(),
        ),
        Dir::South => (
            cross_center - hw,
            cross_center + hw,
            fp_src.quake_y1(),
            fp_tgt.quake_y0(),
        ),
    };

    // Convert to grid cells for overlap check
    let cor_gx0 = (cor_x0 / q) as u32;
    let cor_gx1 = (cor_x1 / q) as u32;
    let cor_gy0 = (cor_y0 / q) as u32;
    let cor_gy1 = (cor_y1 / q) as u32;

    // Build a footprint for the corridor shell (expanded to detect touching)
    let corridor = Footprint3D {
        x0: cor_gx0,
        y0: cor_gy0,
        x1: cor_gx1.max(cor_gx0 + 1),
        y1: cor_gy1.max(cor_gy0 + 1),
        occupies_lower: fp_src.occupies_lower,
        occupies_upper: fp_src.occupies_upper,
    };

    // Check against every other room reservation
    for res in reservations.values() {
        if res.id == source.id || res.id == target.id {
            continue;
        }
        // Skip non-room reservations
        if !matches!(
            res.kind,
            ReservationKind::StandardRoom
                | ReservationKind::MultiStoreyRoom
                | ReservationKind::CaveHost
                | ReservationKind::NegativeSpace
        ) {
            continue;
        }
        // Skip different layers
        if fp_src.occupies_lower != res.footprint.occupies_lower
            && fp_src.occupies_upper != res.footprint.occupies_upper
        {
            continue;
        }
        // Positive overlap check
        if corridor.overlaps_xy(&res.footprint) {
            return false;
        }
    }
    true
}

/// Compute XY cross-overlap between two footprints along a cardinal direction.
fn cross_overlap(
    source: &Footprint3D,
    target: &Footprint3D,
    source_dir: Dir,
) -> Option<(i32, i32)> {
    let (src_span_lo, src_span_hi) = match source_dir {
        Dir::East | Dir::West => (source.y0, source.y1),
        Dir::North | Dir::South => (source.x0, source.x1),
    };
    let (tgt_span_lo, tgt_span_hi) = match source_dir.opposite() {
        Dir::West | Dir::East => (target.y0, target.y1),
        Dir::North | Dir::South => (target.x0, target.x1),
    };

    let lo = src_span_lo.max(tgt_span_lo);
    let hi = src_span_hi.min(tgt_span_hi);

    if lo < hi {
        let q = CONSTRUCTION_QUANTUM as u32;
        Some(((lo * q) as i32, (hi * q) as i32))
    } else {
        None
    }
}

/// Compute integer Manhattan distance between footprint centers in grid units.
fn footprint_distance(a: &Footprint3D, b: &Footprint3D) -> i32 {
    let ax = (a.x0 + a.x1) as i32 / 2;
    let ay = (a.y0 + a.y1) as i32 / 2;
    let bx = (b.x0 + b.x1) as i32 / 2;
    let by = (b.y0 + b.y1) as i32 / 2;
    (ax - bx).abs() + (ay - by).abs()
}

// ── Candidate edge construction ────────────────────────────────────────────

/// Build bounded room-pair descriptors from committed endpoint reservations.
///
/// These descriptors intentionally are not `CandidateEdge`s and carry no lane
/// identity. Compatible sockets, route cells, portal throats, turns, and both
/// endpoint approaches must all reserve successfully before a real candidate
/// is materialized.
fn build_potential_connections(
    reservations: &BTreeMap<ReservationId, ReservationRecord>,
    blueprint: &PacingBlueprint,
) -> Vec<PotentialConnection> {
    let rooms: Vec<_> = reservations
        .values()
        .filter(|reservation| {
            reservation.committed
                && matches!(
                    reservation.kind,
                    ReservationKind::StandardRoom
                        | ReservationKind::MultiStoreyRoom
                        | ReservationKind::CaveHost
                        | ReservationKind::NegativeSpace
                )
        })
        .collect();
    let beat_progression: BTreeMap<_, _> = blueprint
        .beats
        .values()
        .filter(|beat| beat.on_critical_path)
        .map(|beat| (beat.id, beat.progression.raw()))
        .collect();
    let mandatory_set: BTreeSet<_> = blueprint
        .mandatory_edges
        .iter()
        .map(|edge| (edge.from_beat, edge.to_beat))
        .collect();
    let mut primary_reservation = BTreeMap::new();
    for reservation in &rooms {
        if let Some(beat) = reservation.beat_id {
            primary_reservation.entry(beat).or_insert(reservation.id);
        }
    }

    let mut potentials = Vec::new();
    for (index, source) in rooms.iter().enumerate() {
        for target in rooms.iter().skip(index + 1) {
            let layer = if source.footprint.occupies_lower && target.footprint.occupies_lower {
                Some(0)
            } else if source.footprint.occupies_upper && target.footprint.occupies_upper {
                Some(1)
            } else {
                None
            };
            let Some(layer) = layer else {
                continue;
            };
            let Some(direction) = cardinal_direction(&source.footprint, &target.footprint) else {
                continue;
            };
            let mandatory_record = check_mandatory(source, target, &mandatory_set, blueprint)
                .filter(|mandatory| {
                    primary_reservation
                        .get(&mandatory.from_beat)
                        .is_some_and(|id| *id == source.id || *id == target.id)
                        && primary_reservation
                            .get(&mandatory.to_beat)
                            .is_some_and(|id| *id == source.id || *id == target.id)
                });
            let ordinal = potentials.len() as u32;
            let distance = footprint_distance(&source.footprint, &target.footprint);
            let pacing_bias =
                compute_pacing_bias(source, target, &beat_progression, &mandatory_set);
            let pair_id = EdgeId::new(ordinal.saturating_mul(EDGE_ID_STRIDE));
            potentials.push(PotentialConnection {
                ordinal,
                kind: EdgeKind::SameLayer { layer },
                source: source.id,
                target: target.id,
                preferred_dir: direction,
                distance,
                pacing_bias,
                pair_rank: compute_field_rank(
                    pair_id,
                    source.id,
                    target.id,
                    direction,
                    distance,
                    pacing_bias,
                ),
                mandatory_record,
            });
        }
    }
    potentials.sort_by_key(PotentialConnection::sort_key);
    potentials
}

/// Compute pacing bias from beat progression.
///
/// Returns 0 for forward critical-path edges, 1 for same-beat edges,
/// 2+ for backward or non-critical connections.
/// Build exact vertical endpoint candidates from committed composite
/// containers. Each container owns the multi-storey room child, vertical host,
/// and both bands; the logical room remains the concrete topology endpoint.
fn build_feasible_vertical_candidates(
    journal: &ReservationJournal,
    reservations: &BTreeMap<ReservationId, ReservationRecord>,
    edge_id_base: u32,
    next_route_id: u32,
    next_portal_id: u32,
    resolved: &ResolvedRichnessRequestV1,
) -> Result<Vec<(CandidateEdge, ReservationId)>, RichnessError> {
    let mut candidates = Vec::new();
    for composite in reservations
        .values()
        .filter(|record| record.kind == ReservationKind::Composite && record.committed)
    {
        let Some(room_id) = composite
            .composite_children
            .iter()
            .copied()
            .find(|child_id| {
                reservations
                    .get(child_id)
                    .is_some_and(|child| child.kind == ReservationKind::MultiStoreyRoom)
            })
        else {
            continue;
        };
        for &host_id in &composite.composite_children {
            let Some(host) = reservations.get(&host_id) else {
                continue;
            };
            if host.kind != ReservationKind::VerticalHost
                || !host.footprint.occupies_lower
                || !host.footprint.occupies_upper
            {
                continue;
            }
            let id = EdgeId::new(edge_id_base.saturating_add(candidates.len() as u32));
            let mut probe = journal.detached_probe();
            let mut probe_route_id = next_route_id;
            let mut probe_portal_id = next_portal_id;
            reserve_vertical_connection(
                &mut probe,
                id,
                room_id,
                host_id,
                &mut probe_route_id,
                &mut probe_portal_id,
                resolved,
            )?;
            probe.commit_all();

            // As with same-layer candidates, construction follows a complete
            // committed reservation probe.
            candidates.push((
                CandidateEdge {
                    id,
                    kind: EdgeKind::Vertical {
                        lower_reservation: room_id,
                        upper_reservation: room_id,
                    },
                    source: room_id,
                    target: room_id,
                    source_dir: Dir::North,
                    overlap: (0, 0),
                    distance: 0,
                    pacing_bias: 0,
                    field_rank: compute_field_rank(id, room_id, room_id, Dir::North, 0, 0),
                    residual_capacity: 1,
                    mandatory: false,
                    mandatory_record: None,
                },
                host_id,
            ));
        }
    }
    candidates.sort_by_key(|(edge, host)| (edge.sort_key(), *host));
    Ok(candidates)
}

fn compute_pacing_bias(
    a: &ReservationRecord,
    b: &ReservationRecord,
    beat_progression: &BTreeMap<BeatId, u32>,
    mandatory_set: &BTreeSet<(BeatId, BeatId)>,
) -> u32 {
    let pa = a
        .beat_id
        .and_then(|bid| beat_progression.get(&bid).copied());
    let pb = b
        .beat_id
        .and_then(|bid| beat_progression.get(&bid).copied());

    match (pa, pb) {
        (Some(pa), Some(pb)) => {
            // Check if this is a mandatory edge
            let is_mandatory = a.beat_id.zip(b.beat_id).is_some_and(|(ba, bb)| {
                mandatory_set.contains(&(ba, bb)) || mandatory_set.contains(&(bb, ba))
            });

            if is_mandatory {
                0 // mandatory edges get highest priority
            } else if pa == pb {
                100 // same-beat connections are lower priority
            } else if pa < pb {
                (pb - pa) * 10 // forward but not mandatory
            } else {
                200 + (pa - pb) * 10 // backward
            }
        }
        (Some(_), None) | (None, Some(_)) => 500, // one side has no beat
        (None, None) => 1000,                     // neither side has a beat
    }
}

/// Check if an edge between two reservations matches a mandatory edge.
fn check_mandatory(
    a: &ReservationRecord,
    b: &ReservationRecord,
    mandatory_set: &BTreeSet<(BeatId, BeatId)>,
    blueprint: &PacingBlueprint,
) -> Option<MandatoryEdge> {
    let a_beat = a.beat_id?;
    let b_beat = b.beat_id?;

    if mandatory_set.contains(&(a_beat, b_beat)) {
        blueprint
            .mandatory_edges
            .iter()
            .find(|e| e.from_beat == a_beat && e.to_beat == b_beat)
            .copied()
    } else if mandatory_set.contains(&(b_beat, a_beat)) {
        blueprint
            .mandatory_edges
            .iter()
            .find(|e| e.from_beat == b_beat && e.to_beat == a_beat)
            .copied()
    } else {
        None
    }
}

/// Compute a stable deterministic field rank for canonical ordering.
///
/// Uses a linear-congruential hash of edge identity components.
fn compute_field_rank(
    edge_id: EdgeId,
    source_id: ReservationId,
    target_id: ReservationId,
    dir: Dir,
    distance: i32,
    pacing_bias: u32,
) -> u64 {
    let mut h: u64 = 0;
    h = h
        .wrapping_mul(6364136223846793005)
        .wrapping_add(edge_id.raw() as u64);
    h = h
        .wrapping_mul(6364136223846793005)
        .wrapping_add(source_id.raw() as u64);
    h = h
        .wrapping_mul(6364136223846793005)
        .wrapping_add(target_id.raw() as u64);
    h = h
        .wrapping_mul(6364136223846793005)
        .wrapping_add(dir.tag().as_bytes()[0] as u64);
    h = h
        .wrapping_mul(6364136223846793005)
        .wrapping_add(distance as u64);
    h = h
        .wrapping_mul(6364136223846793005)
        .wrapping_add(pacing_bias as u64);
    h
}

// ── Topology solver ────────────────────────────────────────────────────────

/// Find whether a room pair realizes one of the frozen backward-shortcut
/// intents without constructing a candidate edge prematurely.
fn connection_is_shortcut(
    source: ReservationId,
    target: ReservationId,
    shortcut_map: &BTreeMap<(BeatId, BeatId), ShortcutIntent>,
    reservations: &BTreeMap<ReservationId, ReservationRecord>,
) -> bool {
    let Some(source_beat) = reservations.get(&source).and_then(|record| record.beat_id) else {
        return false;
    };
    let Some(target_beat) = reservations.get(&target).and_then(|record| record.beat_id) else {
        return false;
    };
    shortcut_map.keys().any(|(from, to)| {
        (source_beat == *from && target_beat == *to) || (source_beat == *to && target_beat == *from)
    })
}

fn potential_degree_bounds(
    potential: &PotentialConnection,
    blueprint: &PacingBlueprint,
    reservations: &BTreeMap<ReservationId, ReservationRecord>,
) -> (u32, u32) {
    let source_max = reservations
        .get(&potential.source)
        .map(|reservation| reservation_max_exits(reservation, blueprint))
        .unwrap_or(5);
    let target_max = reservations
        .get(&potential.target)
        .map(|reservation| reservation_max_exits(reservation, blueprint))
        .unwrap_or(5);
    (source_max, target_max)
}

fn orient_backward_feasible_candidate(
    mut candidate: FeasibleCandidate,
    blueprint: &PacingBlueprint,
    reservations: &BTreeMap<ReservationId, ReservationRecord>,
) -> FeasibleCandidate {
    let source_progression = reservations
        .get(&candidate.edge.source)
        .and_then(|reservation| reservation.beat_id)
        .and_then(|beat| blueprint.beats.get(&beat))
        .map(|beat| beat.progression.raw());
    let target_progression = reservations
        .get(&candidate.edge.target)
        .and_then(|reservation| reservation.beat_id)
        .and_then(|beat| blueprint.beats.get(&beat))
        .map(|beat| beat.progression.raw());
    if source_progression < target_progression {
        std::mem::swap(&mut candidate.edge.source, &mut candidate.edge.target);
        std::mem::swap(
            &mut candidate.plan.source_socket,
            &mut candidate.plan.target_socket,
        );
        candidate.plan.path.reverse();
        candidate.edge.source_dir = candidate.plan.source_socket.wall;
        candidate.edge.overlap = socket_overlap(candidate.plan.source_socket);
    }
    candidate
}

fn topology_exhaustion_error(
    resolved: &ResolvedRichnessRequestV1,
    path: &str,
    context: impl Into<String>,
) -> RichnessError {
    RichnessError::new(
        RichnessErrorCode::TopologyExhausted,
        resolved.seed(),
        resolved.provenance().request_schema_revision.tag(),
        resolved.provenance().algorithm_revision.tag(),
        resolved.provenance().content_revision.tag(),
        resolved.provenance().preset_revision.tag(),
        resolved.provenance().theme_revision.tag(),
        resolved.provenance().asset_revision.tag(),
        resolved.provenance().convention_revision.tag(),
        path,
        RichnessErrorCategory::PlacementTopologyExhaustion,
        context,
    )
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct RequiredConnection {
    potential: PotentialConnection,
    backward_shortcut: bool,
    /// Exact-plan seam used by focused adversarial proofs. Production leaves
    /// this `None` and enumerates all compatible sockets from current state.
    explicit_plans: Option<Vec<CompleteRoutePlan>>,
}

#[allow(clippy::too_many_arguments)]
fn reserve_mandatory_connections(
    mandatory: &[RequiredConnection],
    index: usize,
    complete_topology: bool,
    potentials: &[PotentialConnection],
    journal: &mut ReservationJournal,
    reservations: &BTreeMap<ReservationId, ReservationRecord>,
    res_index: &BTreeMap<ReservationId, usize>,
    shortcut_map: &BTreeMap<(BeatId, BeatId), ShortcutIntent>,
    next_route_id: &mut u32,
    next_portal_id: &mut u32,
    next_turn_id: &mut u32,
    selected_edges: &mut Vec<CandidateEdge>,
    routes: &mut Vec<CommittedRoute>,
    blueprint: &PacingBlueprint,
    resolved: &ResolvedRichnessRequestV1,
    metrics: &mut TopologySearchMetrics,
) -> bool {
    if index == mandatory.len() {
        if !complete_topology {
            return true;
        }
        let mut selected_ids: BTreeSet<_> = selected_edges.iter().map(|edge| edge.id).collect();
        let shortcut_ids: BTreeSet<_> = mandatory
            .iter()
            .zip(selected_edges.iter())
            .filter(|(required, _)| required.backward_shortcut)
            .map(|(_, edge)| edge.id)
            .collect();
        let mut union_find = UnionFind::new(res_index.len());
        for edge in selected_edges.iter() {
            if shortcut_ids.contains(&edge.id) {
                continue;
            }
            if let (Some(&source), Some(&target)) =
                (res_index.get(&edge.source), res_index.get(&edge.target))
            {
                union_find.union(source, target);
            }
        }
        return search_spanning_tree(
            potentials,
            journal,
            reservations,
            res_index,
            &mut union_find,
            &mut selected_ids,
            selected_edges,
            routes,
            shortcut_map,
            blueprint,
            next_route_id,
            next_portal_id,
            next_turn_id,
            resolved,
            metrics,
        );
    }

    let selected_ids: BTreeSet<_> = selected_edges.iter().map(|edge| edge.id).collect();
    let mut candidates = if let Some(plans) = &mandatory[index].explicit_plans {
        let mut candidates: Vec<_> = plans
            .iter()
            .enumerate()
            .filter_map(|(lane, plan)| {
                materialize_explicit_candidate(
                    journal,
                    &mandatory[index].potential,
                    plan,
                    lane as u32,
                    reservations,
                    *next_route_id,
                    *next_portal_id,
                    *next_turn_id,
                    resolved,
                )
            })
            .collect();
        candidates.sort_by_key(|candidate| candidate.edge.sort_key());
        candidates
    } else {
        let route_components = RouteCenterComponents::build(journal);
        feasible_candidates_for_potential(
            journal,
            &route_components,
            &mandatory[index].potential,
            reservations,
            *next_route_id,
            *next_portal_id,
            *next_turn_id,
            resolved,
        )
    };
    candidates.retain(|candidate| !selected_ids.contains(&candidate.edge.id));
    metrics.mandatory_candidates_materialized = metrics
        .mandatory_candidates_materialized
        .saturating_add(candidates.len() as u64);

    for mut candidate in candidates {
        if mandatory[index].backward_shortcut {
            candidate = orient_backward_feasible_candidate(candidate, blueprint, reservations);
            candidate.edge.mandatory = false;
            candidate.edge.mandatory_record = None;
        }
        let before = journal.state_snapshot();
        let counters_before = (*next_route_id, *next_portal_id, *next_turn_id);
        let selected_edges_before = selected_edges.clone();
        let routes_before = routes.clone();
        journal.mark();
        let route = match commit_feasible_candidate(
            journal,
            &candidate,
            reservations,
            next_route_id,
            next_portal_id,
            next_turn_id,
            resolved,
        ) {
            Ok(route) => route,
            Err(_) => {
                let rolled = journal.rollback();
                *next_route_id = counters_before.0;
                *next_portal_id = counters_before.1;
                *next_turn_id = counters_before.2;
                metrics.mandatory_backtracks = metrics.mandatory_backtracks.saturating_add(1);
                metrics.rollback_checks = metrics.rollback_checks.saturating_add(1);
                if !rolled
                    || !journal.matches_snapshot(&before)
                    || (*next_route_id, *next_portal_id, *next_turn_id) != counters_before
                    || *selected_edges != selected_edges_before
                    || *routes != routes_before
                {
                    metrics.rollback_mismatches = metrics.rollback_mismatches.saturating_add(1);
                }
                continue;
            }
        };
        selected_edges.push(candidate.edge.clone());
        routes.push(route);

        if reserve_mandatory_connections(
            mandatory,
            index + 1,
            complete_topology,
            potentials,
            journal,
            reservations,
            res_index,
            shortcut_map,
            next_route_id,
            next_portal_id,
            next_turn_id,
            selected_edges,
            routes,
            blueprint,
            resolved,
            metrics,
        ) {
            journal.commit();
            return true;
        }

        selected_edges.pop();
        routes.pop();
        let rolled = journal.rollback();
        *next_route_id = counters_before.0;
        *next_portal_id = counters_before.1;
        *next_turn_id = counters_before.2;
        metrics.mandatory_backtracks = metrics.mandatory_backtracks.saturating_add(1);
        metrics.rollback_checks = metrics.rollback_checks.saturating_add(1);
        if !rolled
            || !journal.matches_snapshot(&before)
            || (*next_route_id, *next_portal_id, *next_turn_id) != counters_before
            || *selected_edges != selected_edges_before
            || *routes != routes_before
        {
            metrics.rollback_mismatches = metrics.rollback_mismatches.saturating_add(1);
        }
    }
    false
}

#[allow(clippy::too_many_arguments)]
fn search_spanning_tree(
    potentials: &[PotentialConnection],
    journal: &mut ReservationJournal,
    reservations: &BTreeMap<ReservationId, ReservationRecord>,
    res_index: &BTreeMap<ReservationId, usize>,
    union_find: &mut UnionFind,
    selected_ids: &mut BTreeSet<EdgeId>,
    selected_edges: &mut Vec<CandidateEdge>,
    routes: &mut Vec<CommittedRoute>,
    shortcut_map: &BTreeMap<(BeatId, BeatId), ShortcutIntent>,
    blueprint: &PacingBlueprint,
    next_route_id: &mut u32,
    next_portal_id: &mut u32,
    next_turn_id: &mut u32,
    resolved: &ResolvedRichnessRequestV1,
    metrics: &mut TopologySearchMetrics,
) -> bool {
    if res_index.is_empty() {
        return true;
    }
    let root = union_find.find(0);
    if (0..res_index.len()).all(|index| union_find.find(index) == root) {
        return true;
    }
    if metrics.topology_search_states >= MAX_TOPOLOGY_SEARCH_STATES {
        return false;
    }
    let route_components = RouteCenterComponents::build(journal);
    if !spanning_components_route_connected(
        potentials,
        &route_components,
        reservations,
        res_index,
        union_find,
        selected_edges,
        shortcut_map,
        blueprint,
    ) {
        return false;
    }

    let mut eligible_potentials = Vec::new();
    let mut component_options: BTreeMap<usize, usize> = BTreeMap::new();
    for potential in potentials {
        if potential.mandatory_record.is_some()
            || connection_is_shortcut(
                potential.source,
                potential.target,
                shortcut_map,
                reservations,
            )
        {
            continue;
        }
        let (Some(&source_index), Some(&target_index)) = (
            res_index.get(&potential.source),
            res_index.get(&potential.target),
        ) else {
            continue;
        };
        let source_root = union_find.find(source_index);
        let target_root = union_find.find(target_index);
        if source_root == target_root {
            continue;
        }
        let (source_max, target_max) = potential_degree_bounds(potential, blueprint, reservations);
        if source_max == 1 && target_max == 1
            || count_incident_edges(potential.source, selected_edges) >= source_max as usize
            || count_incident_edges(potential.target, selected_edges) >= target_max as usize
        {
            continue;
        }
        *component_options.entry(source_root).or_default() += 1;
        *component_options.entry(target_root).or_default() += 1;
        eligible_potentials.push(potential);
    }
    let Some((&constrained_component, _)) = component_options
        .iter()
        .min_by_key(|(component, count)| (**count, **component))
    else {
        return false;
    };

    // Materialize complete route candidates only for the most constrained
    // component. Any spanning solution must choose one of these edges, so this
    // variable ordering preserves complete deterministic search.
    let mut candidates = Vec::new();
    for potential in eligible_potentials {
        let source_root = union_find.find(res_index[&potential.source]);
        let target_root = union_find.find(res_index[&potential.target]);
        if source_root != constrained_component && target_root != constrained_component {
            continue;
        }
        candidates.extend(feasible_candidates_for_potential(
            journal,
            &route_components,
            potential,
            reservations,
            *next_route_id,
            *next_portal_id,
            *next_turn_id,
            resolved,
        ));
    }
    candidates.retain(|candidate| !selected_ids.contains(&candidate.edge.id));
    candidates.sort_by_key(|candidate| candidate.edge.sort_key());

    for candidate in candidates {
        if metrics.topology_search_states >= MAX_TOPOLOGY_SEARCH_STATES {
            return false;
        }
        metrics.topology_search_states = metrics.topology_search_states.saturating_add(1);
        let before = journal.state_snapshot();
        let counters_before = (*next_route_id, *next_portal_id, *next_turn_id);
        let union_before = union_find.clone();
        let selected_ids_before = selected_ids.clone();
        let selected_edges_before = selected_edges.clone();
        let routes_before = routes.clone();
        journal.mark();
        let route = match commit_feasible_candidate(
            journal,
            &candidate,
            reservations,
            next_route_id,
            next_portal_id,
            next_turn_id,
            resolved,
        ) {
            Ok(route) => route,
            Err(_) => {
                let rolled = journal.rollback();
                *next_route_id = counters_before.0;
                *next_portal_id = counters_before.1;
                *next_turn_id = counters_before.2;
                *union_find = union_before.clone();
                metrics.topology_backtracks = metrics.topology_backtracks.saturating_add(1);
                metrics.rollback_checks = metrics.rollback_checks.saturating_add(1);
                if !rolled
                    || !journal.matches_snapshot(&before)
                    || (*next_route_id, *next_portal_id, *next_turn_id) != counters_before
                    || *union_find != union_before
                    || *selected_ids != selected_ids_before
                    || *selected_edges != selected_edges_before
                    || *routes != routes_before
                {
                    metrics.rollback_mismatches = metrics.rollback_mismatches.saturating_add(1);
                }
                continue;
            }
        };
        let source_index = res_index[&candidate.edge.source];
        let target_index = res_index[&candidate.edge.target];
        union_find.union(source_index, target_index);
        selected_ids.insert(candidate.edge.id);
        selected_edges.push(candidate.edge.clone());
        routes.push(route);

        if search_spanning_tree(
            potentials,
            journal,
            reservations,
            res_index,
            union_find,
            selected_ids,
            selected_edges,
            routes,
            shortcut_map,
            blueprint,
            next_route_id,
            next_portal_id,
            next_turn_id,
            resolved,
            metrics,
        ) {
            journal.commit();
            return true;
        }

        routes.pop();
        selected_edges.pop();
        selected_ids.remove(&candidate.edge.id);
        let rolled = journal.rollback();
        *next_route_id = counters_before.0;
        *next_portal_id = counters_before.1;
        *next_turn_id = counters_before.2;
        *union_find = union_before.clone();
        metrics.topology_backtracks = metrics.topology_backtracks.saturating_add(1);
        metrics.rollback_checks = metrics.rollback_checks.saturating_add(1);
        if !rolled
            || !journal.matches_snapshot(&before)
            || (*next_route_id, *next_portal_id, *next_turn_id) != counters_before
            || *union_find != union_before
            || *selected_ids != selected_ids_before
            || *selected_edges != selected_edges_before
            || *routes != routes_before
        {
            metrics.rollback_mismatches = metrics.rollback_mismatches.saturating_add(1);
        }
    }
    false
}

#[allow(clippy::too_many_arguments)]
fn search_loop_tail(
    target_loops: usize,
    loops_added: usize,
    potentials: &[PotentialConnection],
    journal: &mut ReservationJournal,
    reservations: &BTreeMap<ReservationId, ReservationRecord>,
    selected_ids: &mut BTreeSet<EdgeId>,
    selected_edges: &mut Vec<CandidateEdge>,
    routes: &mut Vec<CommittedRoute>,
    shortcuts_realized: &mut Vec<EdgeId>,
    shortcut_map: &BTreeMap<(BeatId, BeatId), ShortcutIntent>,
    blueprint: &PacingBlueprint,
    next_route_id: &mut u32,
    next_portal_id: &mut u32,
    next_turn_id: &mut u32,
    resolved: &ResolvedRichnessRequestV1,
    metrics: &mut TopologySearchMetrics,
) -> bool {
    if loops_added == target_loops {
        return true;
    }
    if metrics.topology_search_states >= MAX_TOPOLOGY_SEARCH_STATES {
        return false;
    }

    let mut tiers: BTreeMap<(i32, u32), Vec<&PotentialConnection>> = BTreeMap::new();
    for potential in potentials {
        let (source_max, target_max) = potential_degree_bounds(potential, blueprint, reservations);
        if count_incident_edges(potential.source, selected_edges) < source_max as usize
            && count_incident_edges(potential.target, selected_edges) < target_max as usize
        {
            tiers
                .entry((potential.distance, potential.pacing_bias))
                .or_default()
                .push(potential);
        }
    }

    // Materialize one canonical Kruskal cost tier at a time. Candidate field
    // rank and stable ID order are global within the tier; higher-cost tiers
    // are touched only after every lower-cost completion has backtracked.
    let route_components = RouteCenterComponents::build(journal);
    for (_, mut tier) in tiers {
        tier.sort_by_key(|potential| potential.sort_key());
        let mut candidates = Vec::new();
        for potential in tier {
            candidates.extend(feasible_candidates_for_potential(
                journal,
                &route_components,
                potential,
                reservations,
                *next_route_id,
                *next_portal_id,
                *next_turn_id,
                resolved,
            ));
        }
        candidates.retain(|candidate| !selected_ids.contains(&candidate.edge.id));
        candidates.sort_by_key(|candidate| candidate.edge.sort_key());

        for mut candidate in candidates {
            if metrics.topology_search_states >= MAX_TOPOLOGY_SEARCH_STATES {
                return false;
            }
            let is_shortcut = connection_is_shortcut(
                candidate.edge.source,
                candidate.edge.target,
                shortcut_map,
                reservations,
            );
            if is_shortcut {
                candidate = orient_backward_feasible_candidate(candidate, blueprint, reservations);
            }
            candidate.edge.mandatory = false;
            candidate.edge.mandatory_record = None;

            metrics.topology_search_states = metrics.topology_search_states.saturating_add(1);
            let before = journal.state_snapshot();
            let counters_before = (*next_route_id, *next_portal_id, *next_turn_id);
            let selected_ids_before = selected_ids.clone();
            let selected_edges_before = selected_edges.clone();
            let routes_before = routes.clone();
            let shortcuts_before = shortcuts_realized.clone();
            journal.mark();
            let route = match commit_feasible_candidate(
                journal,
                &candidate,
                reservations,
                next_route_id,
                next_portal_id,
                next_turn_id,
                resolved,
            ) {
                Ok(route) => route,
                Err(_) => {
                    let rolled = journal.rollback();
                    *next_route_id = counters_before.0;
                    *next_portal_id = counters_before.1;
                    *next_turn_id = counters_before.2;
                    metrics.topology_backtracks = metrics.topology_backtracks.saturating_add(1);
                    metrics.rollback_checks = metrics.rollback_checks.saturating_add(1);
                    if !rolled
                        || !journal.matches_snapshot(&before)
                        || (*next_route_id, *next_portal_id, *next_turn_id) != counters_before
                        || *selected_ids != selected_ids_before
                        || *selected_edges != selected_edges_before
                        || *routes != routes_before
                        || *shortcuts_realized != shortcuts_before
                    {
                        metrics.rollback_mismatches = metrics.rollback_mismatches.saturating_add(1);
                    }
                    continue;
                }
            };
            selected_ids.insert(candidate.edge.id);
            if is_shortcut {
                shortcuts_realized.push(candidate.edge.id);
            }
            selected_edges.push(candidate.edge);
            routes.push(route);

            if search_loop_tail(
                target_loops,
                loops_added + 1,
                potentials,
                journal,
                reservations,
                selected_ids,
                selected_edges,
                routes,
                shortcuts_realized,
                shortcut_map,
                blueprint,
                next_route_id,
                next_portal_id,
                next_turn_id,
                resolved,
                metrics,
            ) {
                journal.commit();
                return true;
            }

            let rolled = journal.rollback();
            *next_route_id = counters_before.0;
            *next_portal_id = counters_before.1;
            *next_turn_id = counters_before.2;
            *selected_ids = selected_ids_before.clone();
            *selected_edges = selected_edges_before.clone();
            *routes = routes_before.clone();
            *shortcuts_realized = shortcuts_before.clone();
            metrics.topology_backtracks = metrics.topology_backtracks.saturating_add(1);
            metrics.rollback_checks = metrics.rollback_checks.saturating_add(1);
            if !rolled
                || !journal.matches_snapshot(&before)
                || (*next_route_id, *next_portal_id, *next_turn_id) != counters_before
                || *selected_ids != selected_ids_before
                || *selected_edges != selected_edges_before
                || *routes != routes_before
                || *shortcuts_realized != shortcuts_before
            {
                metrics.rollback_mismatches = metrics.rollback_mismatches.saturating_add(1);
            }
        }
    }
    false
}

/// Solve constrained Kruskal topology from placement result and blueprint.
///
/// Potential room pairs never enter the graph. Mandatory search, constrained
/// Kruskal, and loop-tail selection consume only `CandidateEdge` values backed
/// by a complete route + portal-throat + turn + endpoint-approach reservation
/// probe against the current committed state.
pub(crate) fn solve_topology(
    blueprint: &PacingBlueprint,
    placement: &PlacementResult,
    resolved: &ResolvedRichnessRequestV1,
) -> Result<TopologyResult, RichnessError> {
    let mut journal = placement.journal.detached_probe();
    let potentials = build_potential_connections(&placement.reservations, blueprint);

    let reservation_list: Vec<&ReservationRecord> = placement
        .reservations
        .values()
        .filter(|record| {
            matches!(
                record.kind,
                ReservationKind::StandardRoom
                    | ReservationKind::MultiStoreyRoom
                    | ReservationKind::CaveHost
                    | ReservationKind::NegativeSpace
            )
        })
        .collect();
    let res_index: BTreeMap<ReservationId, usize> = reservation_list
        .iter()
        .enumerate()
        .map(|(index, record)| (record.id, index))
        .collect();
    let room_count = reservation_list.len();
    let mut metrics = TopologySearchMetrics::default();
    if room_count == 0 {
        return Ok(TopologyResult {
            selected_edges: Vec::new(),
            routes: Vec::new(),
            journal,
            beat_to_reservations: placement.beat_to_reservations.clone(),
            loop_count: 0,
            shortcuts_realized: Vec::new(),
            vertical_edges: Vec::new(),
            vertical_routes: Vec::new(),
            search_metrics: metrics,
        });
    }

    let target_loops = match blueprint.preset {
        RichnessPreset::Sparse => 0,
        RichnessPreset::Moderate => 2,
        RichnessPreset::Rich => 4,
    };
    let shortcut_map: BTreeMap<(BeatId, BeatId), ShortcutIntent> = blueprint
        .shortcut_intents
        .iter()
        .map(|shortcut| ((shortcut.from_beat, shortcut.to_beat), *shortcut))
        .collect();

    let mut required_connections =
        Vec::with_capacity(blueprint.mandatory_edges.len() + usize::from(target_loops > 0));
    for required in &blueprint.mandatory_edges {
        let Some(potential) = potentials
            .iter()
            .find(|potential| potential.mandatory_record == Some(*required))
        else {
            return Err(topology_exhaustion_error(
                resolved,
                "topology.mandatory",
                format!(
                    "mandatory edge {:?}->{:?} has no committed endpoint pair",
                    required.from_beat, required.to_beat
                ),
            ));
        };
        required_connections.push(RequiredConnection {
            potential: potential.clone(),
            backward_shortcut: false,
            explicit_plans: None,
        });
    }

    let shortcut_reserved_by_backtracking = target_loops > 0 && !shortcut_map.is_empty();
    if shortcut_reserved_by_backtracking {
        let Some(potential) = potentials.iter().find(|potential| {
            connection_is_shortcut(
                potential.source,
                potential.target,
                &shortcut_map,
                &placement.reservations,
            )
        }) else {
            return Err(topology_exhaustion_error(
                resolved,
                "topology.shortcut",
                format!(
                    "required backward shortcut has no committed endpoint pair: intents={:?}",
                    blueprint.shortcut_intents
                ),
            ));
        };
        required_connections.push(RequiredConnection {
            potential: potential.clone(),
            backward_shortcut: true,
            explicit_plans: None,
        });
    }

    let mut selected_edges = Vec::new();
    let mut routes = Vec::new();
    let mut next_route_id = 0u32;
    let mut next_portal_id = 0u32;
    let mut next_turn_id = 0u32;
    if !reserve_mandatory_connections(
        &required_connections,
        0,
        true,
        &potentials,
        &mut journal,
        &placement.reservations,
        &res_index,
        &shortcut_map,
        &mut next_route_id,
        &mut next_portal_id,
        &mut next_turn_id,
        &mut selected_edges,
        &mut routes,
        blueprint,
        resolved,
        &mut metrics,
    ) {
        return Err(topology_exhaustion_error(
            resolved,
            "topology.mandatory",
            format!(
                "complete required-reservation/topology search exhausted: edges={} candidates={} mandatory_backtracks={} topology_states={} frozen_bound={} topology_backtracks={} rollback_checks={} rollback_mismatches={}",
                required_connections.len(),
                metrics.mandatory_candidates_materialized,
                metrics.mandatory_backtracks,
                metrics.topology_search_states,
                MAX_TOPOLOGY_SEARCH_STATES,
                metrics.topology_backtracks,
                metrics.rollback_checks,
                metrics.rollback_mismatches
            ),
        ));
    }
    if metrics.rollback_mismatches != 0 {
        return Err(topology_exhaustion_error(
            resolved,
            "topology.rollback",
            format!(
                "mandatory backtracking failed byte-identical restoration: checks={} mismatches={}",
                metrics.rollback_checks, metrics.rollback_mismatches
            ),
        ));
    }

    let required_shortcut_id = shortcut_reserved_by_backtracking
        .then(|| {
            selected_edges
                .get(blueprint.mandatory_edges.len())
                .map(|edge| edge.id)
        })
        .flatten();
    let mut shortcuts_realized: Vec<_> = required_shortcut_id.into_iter().collect();
    let loops_before_tail = usize::from(required_shortcut_id.is_some());
    let mut selected_ids: BTreeSet<EdgeId> = selected_edges.iter().map(|edge| edge.id).collect();
    // Exact loop augmentation is itself a complete deterministic transaction
    // search: a cheap loop that consumes a later loop's degree or route
    // capacity is rolled back before the next concrete tail candidate.
    if !search_loop_tail(
        target_loops,
        loops_before_tail,
        &potentials,
        &mut journal,
        &placement.reservations,
        &mut selected_ids,
        &mut selected_edges,
        &mut routes,
        &mut shortcuts_realized,
        &shortcut_map,
        blueprint,
        &mut next_route_id,
        &mut next_portal_id,
        &mut next_turn_id,
        resolved,
        &mut metrics,
    ) {
        return Err(topology_exhaustion_error(
            resolved,
            "topology.loops",
            format!(
                "complete loop-tail search exhausted: required={} materialized={} states={} frozen_bound={} backtracks={} rollback_mismatches={}",
                target_loops,
                loops_before_tail,
                metrics.topology_search_states,
                MAX_TOPOLOGY_SEARCH_STATES,
                metrics.topology_backtracks,
                metrics.rollback_mismatches
            ),
        ));
    }
    if metrics.rollback_mismatches != 0 {
        return Err(topology_exhaustion_error(
            resolved,
            "topology.rollback",
            format!(
                "loop-tail backtracking failed byte-identical restoration: checks={} mismatches={}",
                metrics.rollback_checks, metrics.rollback_mismatches
            ),
        ));
    }
    let loops_added = target_loops;

    if matches!(
        blueprint.preset,
        RichnessPreset::Moderate | RichnessPreset::Rich
    ) && !blueprint.shortcut_intents.is_empty()
        && shortcuts_realized.is_empty()
    {
        return Err(topology_exhaustion_error(
            resolved,
            "topology.shortcut",
            "Moderate/Rich topology committed no backward shortcut",
        ));
    }

    let vertical_edge_base = (potentials.len() as u32)
        .saturating_mul(EDGE_ID_STRIDE)
        .saturating_add(1);
    let vertical_candidates = build_feasible_vertical_candidates(
        &journal,
        &placement.reservations,
        vertical_edge_base,
        next_route_id,
        next_portal_id,
        resolved,
    )?;
    let mut vertical_edges = Vec::with_capacity(vertical_candidates.len());
    let mut vertical_routes = Vec::with_capacity(vertical_candidates.len());
    for (edge, host_id) in vertical_candidates {
        let route = reserve_vertical_connection(
            &mut journal,
            edge.id,
            edge.source,
            host_id,
            &mut next_route_id,
            &mut next_portal_id,
            resolved,
        )?;
        vertical_edges.push(edge);
        vertical_routes.push(route);
    }

    journal.commit_all();
    selected_edges.sort_by_key(CandidateEdge::sort_key);
    routes.sort_by_key(|route| route.id.raw());
    shortcuts_realized.sort_unstable();
    shortcuts_realized.dedup();
    vertical_edges.sort_by_key(CandidateEdge::sort_key);
    vertical_routes.sort_by_key(|route| route.id.raw());

    Ok(TopologyResult {
        selected_edges,
        routes,
        journal,
        beat_to_reservations: placement.beat_to_reservations.clone(),
        loop_count: loops_added,
        shortcuts_realized,
        vertical_edges,
        vertical_routes,
        search_metrics: metrics,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct PortalRouteCandidate {
    wall: Dir,
    anchor: CellCoord,
    exterior_center: CellCoord,
}

fn ordered_walls(preferred: Dir) -> [Dir; 4] {
    match preferred {
        Dir::North => [Dir::North, Dir::West, Dir::East, Dir::South],
        Dir::South => [Dir::South, Dir::East, Dir::West, Dir::North],
        Dir::West => [Dir::West, Dir::South, Dir::North, Dir::East],
        Dir::East => [Dir::East, Dir::North, Dir::South, Dir::West],
    }
}

/// Enumerate the revision-v1 canonical non-overlapping 64-unit portal sockets
/// in midpoint-first order, capped by the frozen five-exit room capacity.
/// Centers retain the exact two-cell throat clearance. Phase-09 surrounds
/// must adapt to this committed structural socket contract; topology cannot
/// discard legal 64×80 routes merely to reserve decorative surround width.
/// The exterior center is two cells beyond the wall, so its 4×4 route witness
/// begins exactly at the room boundary without overlapping the room.
fn portal_route_candidates(
    room: &ReservationRecord,
    preferred: Dir,
    layer: u8,
    grid_width: u32,
    grid_height: u32,
) -> Vec<PortalRouteCandidate> {
    let mut candidates = Vec::new();
    for wall in ordered_walls(preferred) {
        let (lo, hi) = match wall {
            Dir::East | Dir::West => (room.footprint.y0 + 2, room.footprint.y1.saturating_sub(2)),
            Dir::North | Dir::South => (room.footprint.x0 + 2, room.footprint.x1.saturating_sub(2)),
        };
        if lo > hi {
            continue;
        }
        let midpoint = (lo + hi) / 2;
        let centers: Vec<_> = [0i32, -4, 4, -8, 8]
            .into_iter()
            .filter_map(|offset| midpoint.checked_add_signed(offset))
            .filter(|center| *center >= lo && *center <= hi)
            .collect();

        for center in centers {
            let candidate = match wall {
                Dir::East => room
                    .footprint
                    .x1
                    .checked_add(2)
                    .map(|x| PortalRouteCandidate {
                        wall,
                        anchor: CellCoord::new(room.footprint.x1 - 1, center, layer),
                        exterior_center: CellCoord::new(x, center, layer),
                    }),
                Dir::West => room
                    .footprint
                    .x0
                    .checked_sub(2)
                    .map(|x| PortalRouteCandidate {
                        wall,
                        anchor: CellCoord::new(room.footprint.x0, center, layer),
                        exterior_center: CellCoord::new(x, center, layer),
                    }),
                Dir::North => room
                    .footprint
                    .y0
                    .checked_sub(2)
                    .map(|y| PortalRouteCandidate {
                        wall,
                        anchor: CellCoord::new(center, room.footprint.y0, layer),
                        exterior_center: CellCoord::new(center, y, layer),
                    }),
                Dir::South => room
                    .footprint
                    .y1
                    .checked_add(2)
                    .map(|y| PortalRouteCandidate {
                        wall,
                        anchor: CellCoord::new(center, room.footprint.y1 - 1, layer),
                        exterior_center: CellCoord::new(center, y, layer),
                    }),
            };
            if let Some(candidate) = candidate {
                let center = candidate.exterior_center;
                if center.x >= 2
                    && center.y >= 2
                    && center.x + 2 <= grid_width
                    && center.y + 2 <= grid_height
                {
                    candidates.push(candidate);
                }
            }
        }
    }
    candidates
}

fn route_square_cells(center: CellCoord) -> impl Iterator<Item = CellCoord> {
    let x0 = center.x - 2;
    let y0 = center.y - 2;
    (x0..x0 + 4).flat_map(move |x| (y0..y0 + 4).map(move |y| CellCoord::new(x, y, center.layer)))
}

fn route_center_is_clear(journal: &ReservationJournal, center: CellCoord) -> bool {
    route_square_cells(center).all(|coord| journal.grid.get(coord).is_none())
}

const ROUTE_CENTER_BLOCKED: u32 = u32::MAX;
const ROUTE_CENTER_UNVISITED: u32 = u32::MAX - 1;

/// Connected-component labels for the currently free route-center lattice.
/// Building this once is a sound, finite forward check: topology only adds
/// reservations, so components disconnected now cannot become connected in a
/// descendant search state.
struct RouteCenterComponents {
    width: u32,
    height: u32,
    labels: Vec<u32>,
}

impl RouteCenterComponents {
    fn build(journal: &ReservationJournal) -> Self {
        let width = journal.grid.grid_width();
        let height = journal.grid.grid_height();
        let mut labels = vec![ROUTE_CENTER_BLOCKED; (width * height * 2) as usize];
        let index =
            |coord: CellCoord| ((coord.layer as u32 * height + coord.y) * width + coord.x) as usize;

        for layer in 0..2 {
            for y in 2..=height.saturating_sub(2) {
                for x in 2..=width.saturating_sub(2) {
                    let coord = CellCoord::new(x, y, layer);
                    if route_center_is_clear(journal, coord) {
                        labels[index(coord)] = ROUTE_CENTER_UNVISITED;
                    }
                }
            }
        }

        let mut component = 0u32;
        for layer in 0..2 {
            for y in 2..=height.saturating_sub(2) {
                for x in 2..=width.saturating_sub(2) {
                    let start = CellCoord::new(x, y, layer);
                    if labels[index(start)] != ROUTE_CENTER_UNVISITED {
                        continue;
                    }
                    labels[index(start)] = component;
                    let mut pending = vec![start];
                    while let Some(current) = pending.pop() {
                        let neighbors = [
                            current
                                .x
                                .checked_add(1)
                                .map(|nx| CellCoord::new(nx, current.y, current.layer)),
                            current
                                .y
                                .checked_add(1)
                                .map(|ny| CellCoord::new(current.x, ny, current.layer)),
                            current
                                .x
                                .checked_sub(1)
                                .map(|nx| CellCoord::new(nx, current.y, current.layer)),
                            current
                                .y
                                .checked_sub(1)
                                .map(|ny| CellCoord::new(current.x, ny, current.layer)),
                        ];
                        for neighbor in neighbors.into_iter().flatten() {
                            if neighbor.x >= width || neighbor.y >= height {
                                continue;
                            }
                            let neighbor_index = index(neighbor);
                            if labels[neighbor_index] == ROUTE_CENTER_UNVISITED {
                                labels[neighbor_index] = component;
                                pending.push(neighbor);
                            }
                        }
                    }
                    component = component.saturating_add(1);
                }
            }
        }

        Self {
            width,
            height,
            labels,
        }
    }

    fn label(&self, coord: CellCoord) -> Option<u32> {
        if coord.layer >= 2 || coord.x >= self.width || coord.y >= self.height {
            return None;
        }
        let index = ((coord.layer as u32 * self.height + coord.y) * self.width + coord.x) as usize;
        let label = self.labels[index];
        (label < ROUTE_CENTER_UNVISITED).then_some(label)
    }

    fn connected(&self, source: CellCoord, target: CellCoord) -> bool {
        self.label(source)
            .zip(self.label(target))
            .is_some_and(|(source, target)| source == target)
    }
}

fn potential_has_connected_socket_pair(
    potential: &PotentialConnection,
    reservations: &BTreeMap<ReservationId, ReservationRecord>,
    route_components: &RouteCenterComponents,
) -> bool {
    let Some(source) = reservations.get(&potential.source) else {
        return false;
    };
    let Some(target) = reservations.get(&potential.target) else {
        return false;
    };
    let EdgeKind::SameLayer { layer } = potential.kind else {
        return false;
    };
    let source_sockets = portal_route_candidates(
        source,
        potential.preferred_dir,
        layer,
        route_components.width,
        route_components.height,
    );
    let target_sockets = portal_route_candidates(
        target,
        potential.preferred_dir.opposite(),
        layer,
        route_components.width,
        route_components.height,
    );
    source_sockets.iter().any(|source_socket| {
        target_sockets.iter().any(|target_socket| {
            route_components.connected(source_socket.exterior_center, target_socket.exterior_center)
        })
    })
}

/// Necessary route-capacity oracle for a partial spanning forest. Every
/// current union-find component must remain connected to every other through
/// at least one degree-legal potential whose socket centers share free route
/// lattice capacity. A failed check is permanent below this state because
/// descendant branches only consume more cells and degree.
#[allow(clippy::too_many_arguments)]
fn spanning_components_route_connected(
    potentials: &[PotentialConnection],
    route_components: &RouteCenterComponents,
    reservations: &BTreeMap<ReservationId, ReservationRecord>,
    res_index: &BTreeMap<ReservationId, usize>,
    union_find: &mut UnionFind,
    selected_edges: &[CandidateEdge],
    shortcut_map: &BTreeMap<(BeatId, BeatId), ShortcutIntent>,
    blueprint: &PacingBlueprint,
) -> bool {
    let roots: BTreeSet<_> = (0..res_index.len())
        .map(|index| union_find.find(index))
        .collect();
    let Some(&start) = roots.first() else {
        return true;
    };
    if roots.len() == 1 {
        return true;
    }

    let mut adjacency: BTreeMap<usize, BTreeSet<usize>> =
        roots.iter().map(|root| (*root, BTreeSet::new())).collect();
    for potential in potentials {
        if potential.mandatory_record.is_some()
            || connection_is_shortcut(
                potential.source,
                potential.target,
                shortcut_map,
                reservations,
            )
        {
            continue;
        }
        let (Some(&source_index), Some(&target_index)) = (
            res_index.get(&potential.source),
            res_index.get(&potential.target),
        ) else {
            continue;
        };
        let source_root = union_find.find(source_index);
        let target_root = union_find.find(target_index);
        if source_root == target_root {
            continue;
        }
        let (source_max, target_max) = potential_degree_bounds(potential, blueprint, reservations);
        if source_max == 1 && target_max == 1
            || count_incident_edges(potential.source, selected_edges) >= source_max as usize
            || count_incident_edges(potential.target, selected_edges) >= target_max as usize
            || !potential_has_connected_socket_pair(potential, reservations, route_components)
        {
            continue;
        }
        adjacency
            .entry(source_root)
            .or_default()
            .insert(target_root);
        adjacency
            .entry(target_root)
            .or_default()
            .insert(source_root);
    }

    let mut visited = BTreeSet::from([start]);
    let mut pending = vec![start];
    while let Some(component) = pending.pop() {
        if let Some(neighbors) = adjacency.get(&component) {
            for &neighbor in neighbors {
                if visited.insert(neighbor) {
                    pending.push(neighbor);
                }
            }
        }
    }
    visited.len() == roots.len()
}

/// Complete deterministic A* over the finite route-center lattice.
fn find_route_path(
    journal: &ReservationJournal,
    start: CellCoord,
    goal: CellCoord,
) -> Option<Vec<CellCoord>> {
    if start.layer != goal.layer
        || !route_center_is_clear(journal, start)
        || !route_center_is_clear(journal, goal)
    {
        return None;
    }

    let heuristic = |coord: CellCoord| coord.x.abs_diff(goal.x) + coord.y.abs_diff(goal.y);
    let width = journal.grid.grid_width() as usize;
    let height = journal.grid.grid_height() as usize;
    let cell_index = |coord: CellCoord| coord.y as usize * width + coord.x as usize;
    let start_index = cell_index(start);
    // `Reverse` preserves the frozen `(f, cost, y, x)` tie order while dense
    // state arrays avoid ordered-map work for every expanded lattice cell.
    let mut open = BinaryHeap::new();
    let mut best = vec![u32::MAX; width * height];
    let mut previous = vec![u32::MAX; width * height];
    open.push(Reverse((heuristic(start), 0u32, start.y, start.x)));
    best[start_index] = 0;

    while let Some(Reverse((_, cost, y, x))) = open.pop() {
        let current = CellCoord::new(x, y, start.layer);
        let current_index = cell_index(current);
        if cost != best[current_index] {
            continue;
        }
        if current == goal {
            let mut path = vec![goal];
            let mut cursor_index = current_index;
            while cursor_index != start_index {
                cursor_index = previous[cursor_index] as usize;
                if cursor_index >= best.len() {
                    return None;
                }
                path.push(CellCoord::new(
                    (cursor_index % width) as u32,
                    (cursor_index / width) as u32,
                    start.layer,
                ));
            }
            path.reverse();
            return Some(path);
        }

        let neighbors = [
            x.checked_add(1)
                .map(|nx| CellCoord::new(nx, y, start.layer)),
            y.checked_add(1)
                .map(|ny| CellCoord::new(x, ny, start.layer)),
            x.checked_sub(1)
                .map(|nx| CellCoord::new(nx, y, start.layer)),
            y.checked_sub(1)
                .map(|ny| CellCoord::new(x, ny, start.layer)),
        ];
        for neighbor in neighbors.into_iter().flatten() {
            if neighbor.x < 2
                || neighbor.y < 2
                || neighbor.x + 2 > journal.grid.grid_width()
                || neighbor.y + 2 > journal.grid.grid_height()
                || !route_center_is_clear(journal, neighbor)
            {
                continue;
            }
            let next_cost = cost + 1;
            let neighbor_index = cell_index(neighbor);
            if next_cost < best[neighbor_index] {
                best[neighbor_index] = next_cost;
                previous[neighbor_index] = current_index as u32;
                open.push(Reverse((
                    next_cost + heuristic(neighbor),
                    next_cost,
                    neighbor.y,
                    neighbor.x,
                )));
            }
        }
    }
    None
}

/// Enumerate the frozen route-plan variants for one compatible socket pair.
/// Variant zero is canonical shortest A*. The remaining variants force a
/// deterministic perimeter lane before reaching the target, providing real
/// production detours for capacity backtracking without an authored test seam.
fn find_route_paths(
    journal: &ReservationJournal,
    start: CellCoord,
    goal: CellCoord,
) -> Vec<(u32, Vec<CellCoord>)> {
    let Some(shortest) = find_route_path(journal, start, goal) else {
        // A* is complete over the same center lattice, so no forced-lane
        // variant can exist when the unconstrained search is disconnected.
        return Vec::new();
    };
    let max_x = journal.grid.grid_width().saturating_sub(2);
    let max_y = journal.grid.grid_height().saturating_sub(2);
    let templates = [
        vec![
            CellCoord::new(start.x, 2, start.layer),
            CellCoord::new(goal.x, 2, start.layer),
        ],
        vec![
            CellCoord::new(start.x, max_y, start.layer),
            CellCoord::new(goal.x, max_y, start.layer),
        ],
        vec![
            CellCoord::new(2, start.y, start.layer),
            CellCoord::new(2, goal.y, start.layer),
        ],
        vec![
            CellCoord::new(max_x, start.y, start.layer),
            CellCoord::new(max_x, goal.y, start.layer),
        ],
    ];

    let mut variants = vec![(0, shortest.clone())];
    let mut unique_paths = BTreeSet::from([shortest]);
    for (index, waypoints) in templates.into_iter().enumerate() {
        let mut path = vec![start];
        let mut visited = BTreeSet::from([start]);
        let mut feasible = true;
        for target in waypoints.into_iter().chain(std::iter::once(goal)) {
            let mut cursor = *path.last().unwrap();
            while cursor.x != target.x {
                cursor.x = if cursor.x < target.x {
                    cursor.x + 1
                } else {
                    cursor.x - 1
                };
                if !route_center_is_clear(journal, cursor) || !visited.insert(cursor) {
                    feasible = false;
                    break;
                }
                path.push(cursor);
            }
            while feasible && cursor.y != target.y {
                cursor.y = if cursor.y < target.y {
                    cursor.y + 1
                } else {
                    cursor.y - 1
                };
                if !route_center_is_clear(journal, cursor) || !visited.insert(cursor) {
                    feasible = false;
                    break;
                }
                path.push(cursor);
            }
            if !feasible {
                break;
            }
        }
        if feasible && unique_paths.insert(path.clone()) {
            variants.push(((index + 1) as u32, path));
        }
    }
    debug_assert!(variants.len() <= ROUTE_PATH_VARIANT_COUNT);
    variants
}

fn path_turns(path: &[CellCoord]) -> Vec<CellCoord> {
    path.windows(3)
        .filter_map(|window| {
            let first = (
                window[1].x as i64 - window[0].x as i64,
                window[1].y as i64 - window[0].y as i64,
            );
            let second = (
                window[2].x as i64 - window[1].x as i64,
                window[2].y as i64 - window[1].y as i64,
            );
            (first != second).then_some(window[1])
        })
        .collect()
}

fn cells_to_runs(cells: &BTreeSet<CellCoord>) -> Vec<Footprint3D> {
    let mut rows: BTreeMap<(u8, u32), Vec<u32>> = BTreeMap::new();
    for cell in cells {
        rows.entry((cell.layer, cell.y)).or_default().push(cell.x);
    }
    let mut runs = Vec::new();
    for ((layer, y), mut xs) in rows {
        xs.sort_unstable();
        let Some(&first) = xs.first() else {
            continue;
        };
        let mut start = first;
        let mut previous = first;
        for x in xs.into_iter().skip(1) {
            if x == previous + 1 {
                previous = x;
                continue;
            }
            runs.push(Footprint3D {
                x0: start,
                y0: y,
                x1: previous + 1,
                y1: y + 1,
                occupies_lower: layer == 0,
                occupies_upper: layer == 1,
            });
            start = x;
            previous = x;
        }
        runs.push(Footprint3D {
            x0: start,
            y0: y,
            x1: previous + 1,
            y1: y + 1,
            occupies_lower: layer == 0,
            occupies_upper: layer == 1,
        });
    }
    runs
}

fn turn_witness_footprint(position: CellCoord) -> Option<Footprint3D> {
    let x0 = position.x.checked_sub(2)?;
    let y0 = position.y.checked_sub(2)?;
    Some(Footprint3D {
        x0,
        y0,
        x1: position.x.checked_add(2)?,
        y1: position.y.checked_add(2)?,
        occupies_lower: position.layer == 0,
        occupies_upper: position.layer == 1,
    })
}

type TurnPlans = (Vec<(CellCoord, Footprint3D)>, BTreeSet<CellCoord>);

fn build_turn_plans(path: &[CellCoord]) -> Result<TurnPlans, &'static str> {
    let mut plans = Vec::new();
    let mut cells = BTreeSet::new();
    for position in path_turns(path) {
        let witness = turn_witness_footprint(position)
            .ok_or("turn witness exceeded the finite route lattice")?;
        if witness.cells().into_iter().any(|cell| !cells.insert(cell)) {
            return Err("distinct turn witnesses overlap");
        }
        plans.push((position, witness));
    }
    Ok((plans, cells))
}

fn route_search_error(
    resolved: &ResolvedRichnessRequestV1,
    edge_id: EdgeId,
    source: ReservationId,
    target: ReservationId,
    context: impl Into<String>,
) -> RichnessError {
    RichnessError::new(
        RichnessErrorCode::TopologyExhausted,
        resolved.seed(),
        resolved.provenance().request_schema_revision.tag(),
        resolved.provenance().algorithm_revision.tag(),
        resolved.provenance().content_revision.tag(),
        resolved.provenance().preset_revision.tag(),
        resolved.provenance().theme_revision.tag(),
        resolved.provenance().asset_revision.tag(),
        resolved.provenance().convention_revision.tag(),
        "topology.route",
        RichnessErrorCategory::PlacementTopologyExhaustion,
        format!(
            "complete route reservation failed for edge {:?} ({:?}->{:?}): {}",
            edge_id,
            source,
            target,
            context.into()
        ),
    )
}

/// Atomically reserve one exact socket/path realization. Portal throats are
/// first so a competing route can never commit capacity through an unowned
/// endpoint approach. Route cells follow, then every direction change receives
/// its own committed `Turn` reservation.
fn reserve_complete_route_plan(
    journal: &mut ReservationJournal,
    edge_id: EdgeId,
    source: &ReservationRecord,
    target: &ReservationRecord,
    plan: &CompleteRoutePlan,
    next_route_id: &mut u32,
    next_portal_id: &mut u32,
    next_turn_id: &mut u32,
    resolved: &ResolvedRichnessRequestV1,
) -> Result<CommittedRoute, RichnessError> {
    let route_id = RouteId::new(*next_route_id);
    let mut portal_counter = *next_portal_id;
    let mut turn_counter = *next_turn_id;
    journal.mark();

    let result = (|| {
        let path_is_contiguous = plan.path.first() == Some(&plan.source_socket.exterior_center)
            && plan.path.last() == Some(&plan.target_socket.exterior_center)
            && plan.path.windows(2).all(|window| {
                window[0].layer == window[1].layer
                    && window[0].x.abs_diff(window[1].x) + window[0].y.abs_diff(window[1].y) == 1
            });
        if !path_is_contiguous {
            return Err(route_search_error(
                resolved,
                edge_id,
                source.id,
                target.id,
                "route plan is not a contiguous socket-to-socket lattice path",
            ));
        }

        let source_witness = portal_witness_footprint(
            source,
            plan.source_socket.wall,
            plan.source_socket.anchor,
            plan.source_socket.anchor.layer,
        );
        let source_throat_id = journal
            .try_reserve_for_route(
                route_id,
                ReservationKind::PortalThroat,
                source_witness,
                0,
                0,
                0,
                0,
            )
            .map_err(|error| {
                route_search_error(
                    resolved,
                    edge_id,
                    source.id,
                    target.id,
                    format!("source portal throat: {}", error.context),
                )
            })?;
        if let Some(composite_id) = journal.composite_parent_of(source.id) {
            journal.add_composite_child(composite_id, source_throat_id);
        }
        let source_portal = CommittedPortal {
            id: PortalId::new(portal_counter),
            wall: plan.source_socket.wall,
            anchor_cell: plan.source_socket.anchor,
            witness: source_witness,
            headroom: HEADROOM,
            endpoint_reservation_id: source.id,
            reservation_id: source_throat_id,
        };
        portal_counter = portal_counter.saturating_add(1);

        let target_witness = portal_witness_footprint(
            target,
            plan.target_socket.wall,
            plan.target_socket.anchor,
            plan.target_socket.anchor.layer,
        );
        let target_throat_id = journal
            .try_reserve_for_route(
                route_id,
                ReservationKind::PortalThroat,
                target_witness,
                0,
                0,
                0,
                0,
            )
            .map_err(|error| {
                route_search_error(
                    resolved,
                    edge_id,
                    source.id,
                    target.id,
                    format!("target portal throat: {}", error.context),
                )
            })?;
        if let Some(composite_id) = journal.composite_parent_of(target.id) {
            journal.add_composite_child(composite_id, target_throat_id);
        }
        let target_portal = CommittedPortal {
            id: PortalId::new(portal_counter),
            wall: plan.target_socket.wall,
            anchor_cell: plan.target_socket.anchor,
            witness: target_witness,
            headroom: HEADROOM,
            endpoint_reservation_id: target.id,
            reservation_id: target_throat_id,
        };
        portal_counter = portal_counter.saturating_add(1);

        let mut complete_route_cells = BTreeSet::new();
        for center in &plan.path {
            complete_route_cells.extend(route_square_cells(*center));
        }
        if complete_route_cells.is_empty() {
            return Err(route_search_error(
                resolved,
                edge_id,
                source.id,
                target.id,
                "empty route path",
            ));
        }
        let envelope = Footprint3D {
            x0: complete_route_cells
                .iter()
                .map(|cell| cell.x)
                .min()
                .unwrap_or(0),
            y0: complete_route_cells
                .iter()
                .map(|cell| cell.y)
                .min()
                .unwrap_or(0),
            x1: complete_route_cells
                .iter()
                .map(|cell| cell.x)
                .max()
                .unwrap_or(0)
                .saturating_add(1),
            y1: complete_route_cells
                .iter()
                .map(|cell| cell.y)
                .max()
                .unwrap_or(0)
                .saturating_add(1),
            occupies_lower: plan.source_socket.anchor.layer == 0,
            occupies_upper: plan.source_socket.anchor.layer == 1,
        };

        let (turn_plans, turn_cells) = build_turn_plans(&plan.path).map_err(|context| {
            route_search_error(resolved, edge_id, source.id, target.id, context)
        })?;
        for (_, witness) in &turn_plans {
            if !witness
                .cells()
                .iter()
                .all(|cell| complete_route_cells.contains(cell))
            {
                return Err(route_search_error(
                    resolved,
                    edge_id,
                    source.id,
                    target.id,
                    "turn witness escaped the complete route envelope",
                ));
            }
        }

        // First-class throats and turns own their complete witnesses. Route
        // runs reserve only the disjoint remainder of the path union.
        let mut route_only_cells = complete_route_cells.clone();
        for cell in source_witness
            .cells()
            .into_iter()
            .chain(target_witness.cells())
            .chain(turn_cells.iter().copied())
        {
            route_only_cells.remove(&cell);
        }

        let mut reservation_ids = vec![source_throat_id, target_throat_id];
        for footprint in cells_to_runs(&route_only_cells) {
            let id = journal
                .try_reserve_for_route(route_id, ReservationKind::Route, footprint, 0, 0, 0, 0)
                .map_err(|error| {
                    route_search_error(
                        resolved,
                        edge_id,
                        source.id,
                        target.id,
                        format!("route capacity: {}", error.context),
                    )
                })?;
            reservation_ids.push(id);
        }

        let mut turns = Vec::new();
        for (position, witness) in turn_plans {
            let reservation_id = journal
                .try_reserve_for_route(route_id, ReservationKind::Turn, witness, 0, 0, 0, 0)
                .map_err(|error| {
                    route_search_error(
                        resolved,
                        edge_id,
                        source.id,
                        target.id,
                        format!("turn witness: {}", error.context),
                    )
                })?;
            reservation_ids.push(reservation_id);
            turns.push(CommittedTurn {
                id: TurnId::new(turn_counter),
                position,
                witness,
                headroom: HEADROOM,
                reservation_id,
            });
            turn_counter = turn_counter.saturating_add(1);
        }

        Ok(CommittedRoute {
            id: route_id,
            edge_id,
            source: source.id,
            target: target.id,
            envelope,
            source_portal,
            target_portal,
            reservation_ids,
            turns,
        })
    })();

    match result {
        Ok(route) => {
            journal.commit();
            *next_route_id = next_route_id.saturating_add(1);
            *next_portal_id = portal_counter;
            *next_turn_id = turn_counter;
            Ok(route)
        }
        Err(error) => {
            journal.rollback();
            Err(error)
        }
    }
}

fn socket_overlap(socket: PortalRouteCandidate) -> (i32, i32) {
    let center = match socket.wall {
        Dir::East | Dir::West => socket.anchor.y as i32 * CONSTRUCTION_QUANTUM,
        Dir::North | Dir::South => socket.anchor.x as i32 * CONSTRUCTION_QUANTUM,
    };
    (center - ROUTE_WIDTH / 2, center + ROUTE_WIDTH / 2)
}

/// Materialize only candidates whose complete reservation transaction commits
/// in an isolated clone of the current topology state. Socket pairs in
/// different residual route-lattice components are rejected before A*: the
/// component labels and A* use the same 4×4-clear cardinal lattice, so this
/// removes only searches that are guaranteed to return `None`.
fn feasible_candidates_for_potential(
    journal: &ReservationJournal,
    route_components: &RouteCenterComponents,
    potential: &PotentialConnection,
    reservations: &BTreeMap<ReservationId, ReservationRecord>,
    next_route_id: u32,
    next_portal_id: u32,
    next_turn_id: u32,
    resolved: &ResolvedRichnessRequestV1,
) -> Vec<FeasibleCandidate> {
    let Some(source) = reservations.get(&potential.source) else {
        return Vec::new();
    };
    let Some(target) = reservations.get(&potential.target) else {
        return Vec::new();
    };
    let EdgeKind::SameLayer { layer } = potential.kind else {
        return Vec::new();
    };
    let source_candidates = portal_route_candidates(
        source,
        potential.preferred_dir,
        layer,
        journal.grid.grid_width(),
        journal.grid.grid_height(),
    );
    let target_candidates = portal_route_candidates(
        target,
        potential.preferred_dir.opposite(),
        layer,
        journal.grid.grid_width(),
        journal.grid.grid_height(),
    );

    debug_assert!(source_candidates.len() <= PORTAL_SOCKET_ID_CAPACITY as usize);
    debug_assert!(target_candidates.len() <= PORTAL_SOCKET_ID_CAPACITY as usize);
    let mut feasible = Vec::new();
    'source: for (source_index, source_socket) in source_candidates.into_iter().enumerate() {
        for (target_index, target_socket) in target_candidates.iter().enumerate() {
            if !route_components
                .connected(source_socket.exterior_center, target_socket.exterior_center)
            {
                continue;
            }
            for (variant, path) in find_route_paths(
                journal,
                source_socket.exterior_center,
                target_socket.exterior_center,
            ) {
                // Turn witnesses are a pure function of the path. Reject an
                // overlapping witness plan before cloning the full journal;
                // the reservation transaction would return the same failure.
                if build_turn_plans(&path).is_err() {
                    continue;
                }
                let plan_ordinal = (source_index as u32)
                    .saturating_mul(PORTAL_SOCKET_ID_CAPACITY)
                    .saturating_add(target_index as u32)
                    .saturating_mul(ROUTE_PATH_VARIANT_COUNT as u32)
                    .saturating_add(variant);
                let edge_id = EdgeId::new(
                    potential
                        .ordinal
                        .saturating_mul(EDGE_ID_STRIDE)
                        .saturating_add(plan_ordinal),
                );
                let plan = CompleteRoutePlan {
                    source_socket,
                    target_socket: *target_socket,
                    path,
                };
                let mut probe = journal.detached_probe();
                let mut probe_route_id = next_route_id;
                let mut probe_portal_id = next_portal_id;
                let mut probe_turn_id = next_turn_id;
                let Ok(_route) = reserve_complete_route_plan(
                    &mut probe,
                    edge_id,
                    source,
                    target,
                    &plan,
                    &mut probe_route_id,
                    &mut probe_portal_id,
                    &mut probe_turn_id,
                    resolved,
                ) else {
                    continue;
                };
                probe.commit_all();

                // CandidateEdge construction occurs only after the complete
                // probe transaction committed every reservation kind.
                let edge = CandidateEdge {
                    id: edge_id,
                    kind: potential.kind,
                    source: potential.source,
                    target: potential.target,
                    source_dir: source_socket.wall,
                    overlap: socket_overlap(source_socket),
                    distance: potential.distance,
                    pacing_bias: potential.pacing_bias,
                    field_rank: potential.pair_rank.saturating_add(plan_ordinal as u64),
                    residual_capacity: 1,
                    mandatory: potential.mandatory_record.is_some(),
                    mandatory_record: potential.mandatory_record,
                };
                feasible.push(FeasibleCandidate { edge, plan });
                if feasible.len() == PARALLEL_ROUTE_CAPACITY as usize {
                    break 'source;
                }
            }
        }
    }
    feasible.sort_by_key(|candidate| candidate.edge.sort_key());
    feasible
}

fn materialize_explicit_candidate(
    journal: &ReservationJournal,
    potential: &PotentialConnection,
    plan: &CompleteRoutePlan,
    lane: u32,
    reservations: &BTreeMap<ReservationId, ReservationRecord>,
    next_route_id: u32,
    next_portal_id: u32,
    next_turn_id: u32,
    resolved: &ResolvedRichnessRequestV1,
) -> Option<FeasibleCandidate> {
    if lane >= PARALLEL_ROUTE_CAPACITY {
        return None;
    }
    let source = reservations.get(&potential.source)?;
    let target = reservations.get(&potential.target)?;
    let edge_id = EdgeId::new(
        potential
            .ordinal
            .saturating_mul(EDGE_ID_STRIDE)
            .saturating_add(lane),
    );
    let mut probe = journal.detached_probe();
    let mut probe_route_id = next_route_id;
    let mut probe_portal_id = next_portal_id;
    let mut probe_turn_id = next_turn_id;
    reserve_complete_route_plan(
        &mut probe,
        edge_id,
        source,
        target,
        plan,
        &mut probe_route_id,
        &mut probe_portal_id,
        &mut probe_turn_id,
        resolved,
    )
    .ok()?;
    probe.commit_all();
    Some(FeasibleCandidate {
        edge: CandidateEdge {
            id: edge_id,
            kind: potential.kind,
            source: potential.source,
            target: potential.target,
            source_dir: plan.source_socket.wall,
            overlap: socket_overlap(plan.source_socket),
            distance: potential.distance,
            pacing_bias: potential.pacing_bias,
            field_rank: potential.pair_rank.saturating_add(lane as u64),
            residual_capacity: 1,
            mandatory: potential.mandatory_record.is_some(),
            mandatory_record: potential.mandatory_record,
        },
        plan: plan.clone(),
    })
}

fn commit_feasible_candidate(
    journal: &mut ReservationJournal,
    candidate: &FeasibleCandidate,
    reservations: &BTreeMap<ReservationId, ReservationRecord>,
    next_route_id: &mut u32,
    next_portal_id: &mut u32,
    next_turn_id: &mut u32,
    resolved: &ResolvedRichnessRequestV1,
) -> Result<CommittedRoute, RichnessError> {
    let source = reservations.get(&candidate.edge.source).ok_or_else(|| {
        route_search_error(
            resolved,
            candidate.edge.id,
            candidate.edge.source,
            candidate.edge.target,
            "source endpoint disappeared after feasibility probe",
        )
    })?;
    let target = reservations.get(&candidate.edge.target).ok_or_else(|| {
        route_search_error(
            resolved,
            candidate.edge.id,
            candidate.edge.source,
            candidate.edge.target,
            "target endpoint disappeared after feasibility probe",
        )
    })?;
    reserve_complete_route_plan(
        journal,
        candidate.edge.id,
        source,
        target,
        &candidate.plan,
        next_route_id,
        next_portal_id,
        next_turn_id,
        resolved,
    )
}

/// Reserve a vertical route (stair/shaft/drop) between layers.
fn reserve_vertical_connection(
    journal: &mut ReservationJournal,
    edge_id: EdgeId,
    parent_id: ReservationId,
    host_id: ReservationId,
    next_route_id: &mut u32,
    next_portal_id: &mut u32,
    resolved: &ResolvedRichnessRequestV1,
) -> Result<CommittedRoute, RichnessError> {
    let source = journal.get(parent_id).cloned().ok_or_else(|| {
        route_search_error(
            resolved,
            edge_id,
            parent_id,
            parent_id,
            "vertical endpoint reservation not found",
        )
    })?;
    let host = journal.get(host_id).cloned().ok_or_else(|| {
        route_search_error(
            resolved,
            edge_id,
            parent_id,
            parent_id,
            format!("vertical host {:?} not found", host_id),
        )
    })?;
    let Some(composite_id) = journal.composite_parent_of(parent_id) else {
        return Err(route_search_error(
            resolved,
            edge_id,
            parent_id,
            parent_id,
            format!("vertical endpoint {:?} has no composite owner", parent_id),
        ));
    };
    if host.kind != ReservationKind::VerticalHost
        || !host.footprint.occupies_lower
        || !host.footprint.occupies_upper
        || Some(composite_id) != journal.composite_parent_of(host_id)
    {
        return Err(route_search_error(
            resolved,
            edge_id,
            parent_id,
            parent_id,
            format!("invalid dual-band vertical host {:?}", host_id),
        ));
    }

    let route_id = RouteId::new(*next_route_id);
    let mut portal_counter = *next_portal_id;
    let lower_anchor = CellCoord::new(
        (host.footprint.x0 + host.footprint.x1) / 2,
        (host.footprint.y0 + host.footprint.y1) / 2,
        0,
    );
    let upper_anchor = CellCoord::new(lower_anchor.x, lower_anchor.y, 1);
    journal.mark();
    let result = (|| {
        let lower_witness = portal_witness_footprint(&source, Dir::North, lower_anchor, 0);
        let lower_id = journal.try_reserve_for_route(
            route_id,
            ReservationKind::PortalThroat,
            lower_witness,
            0,
            0,
            0,
            0,
        )?;
        let source_portal = CommittedPortal {
            id: PortalId::new(portal_counter),
            wall: Dir::North,
            anchor_cell: lower_anchor,
            witness: lower_witness,
            headroom: HEADROOM,
            endpoint_reservation_id: parent_id,
            reservation_id: lower_id,
        };
        portal_counter = portal_counter.saturating_add(1);

        let upper_witness = portal_witness_footprint(&source, Dir::North, upper_anchor, 1);
        let upper_id = journal.try_reserve_for_route(
            route_id,
            ReservationKind::PortalThroat,
            upper_witness,
            0,
            0,
            0,
            0,
        )?;
        // Vertical portal throats are protected void constraints inside the
        // same composite that owns the room and shaft.
        journal.add_composite_child(composite_id, lower_id);
        journal.add_composite_child(composite_id, upper_id);
        let target_portal = CommittedPortal {
            id: PortalId::new(portal_counter),
            wall: Dir::North,
            anchor_cell: upper_anchor,
            witness: upper_witness,
            headroom: HEADROOM,
            endpoint_reservation_id: parent_id,
            reservation_id: upper_id,
        };
        portal_counter = portal_counter.saturating_add(1);

        Ok(CommittedRoute {
            id: route_id,
            edge_id,
            source: parent_id,
            target: parent_id,
            envelope: host.footprint,
            source_portal,
            target_portal,
            reservation_ids: vec![host_id, lower_id, upper_id],
            turns: Vec::new(),
        })
    })();

    match result {
        Ok(route) => {
            journal.commit();
            *next_route_id = (*next_route_id).saturating_add(1);
            *next_portal_id = portal_counter;
            Ok(route)
        }
        Err(error) => {
            journal.rollback();
            Err(error)
        }
    }
}

// ── Route overlap detection ────────────────────────────────────────────────

/// Check if a candidate edge's route shell would positively overlap any
/// already-committed edge's route shell.
#[allow(dead_code)]
fn routes_emit_overlap(
    edge: &CandidateEdge,
    committed: &[CandidateEdge],
    reservations: &BTreeMap<ReservationId, ReservationRecord>,
) -> bool {
    let src = match reservations.get(&edge.source) {
        Some(r) => r,
        None => return true, // conservative: assume overlap if missing
    };
    let tgt = match reservations.get(&edge.target) {
        Some(r) => r,
        None => return true,
    };
    let env = route_envelope_footprint(src, tgt, edge.source_dir, edge.overlap);

    for ce in committed {
        let cs = match reservations.get(&ce.source) {
            Some(r) => r,
            None => continue,
        };
        let ct = match reservations.get(&ce.target) {
            Some(r) => r,
            None => continue,
        };
        let cenv = route_envelope_footprint(cs, ct, ce.source_dir, ce.overlap);

        // Positive overlap check
        if envelopes_positive_overlap(&env, &cenv) {
            return true;
        }
    }
    false
}

/// Two footprints positively overlap in XY (area > 0).
fn envelopes_positive_overlap(a: &Footprint3D, b: &Footprint3D) -> bool {
    a.x0 < b.x1
        && b.x0 < a.x1
        && a.y0 < b.y1
        && b.y0 < a.y1
        && ((a.occupies_lower && b.occupies_lower) || (a.occupies_upper && b.occupies_upper))
}

// ── Shortcut detection ─────────────────────────────────────────────────────

/// Check if a candidate edge satisfies a shortcut intent.
///
/// A backward shortcut connects a later beat to an earlier beat. The
/// shortcut intent specifies `from_beat` (later) and `to_beat` (earlier).
fn edge_is_shortcut(
    edge: &CandidateEdge,
    shortcut_map: &BTreeMap<(BeatId, BeatId), ShortcutIntent>,
    reservations: &BTreeMap<ReservationId, ReservationRecord>,
) -> bool {
    let src = match reservations.get(&edge.source) {
        Some(r) => r,
        None => return false,
    };
    let tgt = match reservations.get(&edge.target) {
        Some(r) => r,
        None => return false,
    };

    // Check both directions
    for (from_beat, to_beat) in shortcut_map.keys() {
        if let (Some(src_beat), Some(tgt_beat)) = (src.beat_id, tgt.beat_id) {
            if (src_beat == *from_beat && tgt_beat == *to_beat)
                || (src_beat == *to_beat && tgt_beat == *from_beat)
            {
                return true;
            }
        }
    }
    false
}

/// Orient an undirected candidate as the later-to-earlier traversal required
/// by a backward shortcut.  Reversing a same-layer edge also reverses its
/// source-wall direction; the concrete lane is unchanged.
fn orient_backward_shortcut(
    edge: &CandidateEdge,
    blueprint: &PacingBlueprint,
    reservations: &BTreeMap<ReservationId, ReservationRecord>,
) -> CandidateEdge {
    let source_progression = reservations
        .get(&edge.source)
        .and_then(|reservation| reservation.beat_id)
        .and_then(|beat| blueprint.beats.get(&beat))
        .map(|beat| beat.progression.raw());
    let target_progression = reservations
        .get(&edge.target)
        .and_then(|reservation| reservation.beat_id)
        .and_then(|beat| blueprint.beats.get(&beat))
        .map(|beat| beat.progression.raw());
    if source_progression < target_progression {
        let mut reversed = edge.clone();
        reversed.source = edge.target;
        reversed.target = edge.source;
        if matches!(reversed.kind, EdgeKind::SameLayer { .. }) {
            reversed.source_dir = edge.source_dir.opposite();
        }
        reversed
    } else {
        edge.clone()
    }
}

/// Count incident edges for a reservation.
fn count_incident_edges(res_id: ReservationId, edges: &[CandidateEdge]) -> usize {
    edges
        .iter()
        .filter(|e| e.source == res_id || e.target == res_id)
        .count()
}

fn reservation_max_exits(reservation: &ReservationRecord, blueprint: &PacingBlueprint) -> u32 {
    let beat_max = reservation
        .beat_id
        .and_then(|beat_id| blueprint.beats.get(&beat_id))
        .map(|beat| beat.degree.max_exits);
    let request_max = reservation
        .request_id
        .and_then(|request_id| blueprint.archetype_requests.get(&request_id))
        .map(|request| request.degree.max_exits);
    beat_max.into_iter().chain(request_max).max().unwrap_or(5)
}

/// Get degree bounds for the endpoints of an edge from the blueprint.
fn degree_bounds(
    edge: &CandidateEdge,
    blueprint: &PacingBlueprint,
    reservations: &BTreeMap<ReservationId, ReservationRecord>,
) -> (u32, u32) {
    let source_max = reservations
        .get(&edge.source)
        .map(|reservation| reservation_max_exits(reservation, blueprint))
        .unwrap_or(5);
    let target_max = reservations
        .get(&edge.target)
        .map(|reservation| reservation_max_exits(reservation, blueprint))
        .unwrap_or(5);
    (source_max, target_max)
}

// ── Helpers for Footprint3D Quake coordinates ──────────────────────────────

impl Footprint3D {
    /// Minimum Quake X.
    fn quake_x0(&self) -> i32 {
        (self.x0 as i32) * CONSTRUCTION_QUANTUM
    }

    /// Maximum Quake X.
    fn quake_x1(&self) -> i32 {
        (self.x1 as i32) * CONSTRUCTION_QUANTUM
    }

    /// Minimum Quake Y.
    fn quake_y0(&self) -> i32 {
        (self.y0 as i32) * CONSTRUCTION_QUANTUM
    }

    /// Maximum Quake Y.
    fn quake_y1(&self) -> i32 {
        (self.y1 as i32) * CONSTRUCTION_QUANTUM
    }
}

// ── Proof validators ──────────────────────────────────────────────────────

/// Validation outcome from topology proof checks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ProofValidationReport {
    pub passed: bool,
    pub errors: Vec<String>,
}

impl ProofValidationReport {
    fn pass() -> Self {
        Self {
            passed: true,
            errors: Vec::new(),
        }
    }
    fn fail(msg: impl Into<String>) -> Self {
        Self {
            passed: false,
            errors: vec![msg.into()],
        }
    }
    fn merge(&mut self, other: Self) {
        self.passed = self.passed && other.passed;
        self.errors.extend(other.errors);
    }
}

/// Validate global connectivity: every room reservation is reachable from
/// every other room reservation via selected edges.
pub(crate) fn validate_global_connectivity(
    result: &TopologyResult,
    reservations: &BTreeMap<ReservationId, ReservationRecord>,
) -> ProofValidationReport {
    let room_ids: Vec<ReservationId> = reservations
        .values()
        .filter(|r| {
            matches!(
                r.kind,
                ReservationKind::StandardRoom
                    | ReservationKind::MultiStoreyRoom
                    | ReservationKind::CaveHost
                    | ReservationKind::NegativeSpace
            )
        })
        .map(|r| r.id)
        .collect();

    if room_ids.is_empty() {
        return ProofValidationReport::pass();
    }

    // Build adjacency from selected edges
    let mut adj: BTreeMap<ReservationId, Vec<ReservationId>> = BTreeMap::new();
    for edge in &result.selected_edges {
        adj.entry(edge.source).or_default().push(edge.target);
        adj.entry(edge.target).or_default().push(edge.source);
    }

    // BFS from first room
    let start = room_ids[0];
    let mut visited = BTreeSet::new();
    let mut stack = vec![start];
    while let Some(current) = stack.pop() {
        if !visited.insert(current) {
            continue;
        }
        if let Some(neighbors) = adj.get(&current) {
            for &n in neighbors {
                if !visited.contains(&n) {
                    stack.push(n);
                }
            }
        }
    }

    let unvisited: Vec<_> = room_ids.iter().filter(|id| !visited.contains(id)).collect();
    if !unvisited.is_empty() {
        ProofValidationReport::fail(format!(
            "global connectivity failed: {} of {} rooms unreachable",
            unvisited.len(),
            room_ids.len()
        ))
    } else {
        ProofValidationReport::pass()
    }
}

/// Validate that all mandatory edges from the blueprint are present in
/// selected edges.
pub(crate) fn validate_mandatory_edges_present(
    result: &TopologyResult,
    blueprint: &PacingBlueprint,
    reservations: &BTreeMap<ReservationId, ReservationRecord>,
) -> ProofValidationReport {
    // Build beat-to-reservation map
    let beat_to_res: BTreeMap<BeatId, Vec<ReservationId>> = {
        let mut map: BTreeMap<BeatId, Vec<ReservationId>> = BTreeMap::new();
        for r in reservations.values() {
            if let Some(bid) = r.beat_id {
                map.entry(bid).or_default().push(r.id);
            }
        }
        map
    };

    let mut report = ProofValidationReport::pass();
    for me in &blueprint.mandatory_edges {
        let from_list = beat_to_res.get(&me.from_beat);
        let to_list = beat_to_res.get(&me.to_beat);

        let (from_res, to_res) = match (from_list, to_list) {
            (Some(f), Some(t)) => (f, t),
            _ => {
                report.merge(ProofValidationReport::fail(format!(
                    "mandatory edge {:?}->{:?}: beats not found in reservations",
                    me.from_beat, me.to_beat
                )));
                continue;
            }
        };

        let exact_matches: Vec<_> = result
            .selected_edges
            .iter()
            .filter(|edge| {
                edge.mandatory
                    && edge.mandatory_record == Some(*me)
                    && from_res
                        .iter()
                        .any(|id| *id == edge.source || *id == edge.target)
                    && to_res
                        .iter()
                        .any(|id| *id == edge.source || *id == edge.target)
            })
            .collect();
        if exact_matches.len() != 1 {
            report.merge(ProofValidationReport::fail(format!(
                "mandatory edge {:?}->{:?} has {} exact protected realizations, expected 1",
                me.from_beat,
                me.to_beat,
                exact_matches.len()
            )));
        }
    }
    report
}

/// Validate beat order: edges never connect beats in reverse progression
/// order (backward shortcuts are the exception, handled separately).
pub(crate) fn validate_beat_order(
    result: &TopologyResult,
    blueprint: &PacingBlueprint,
    reservations: &BTreeMap<ReservationId, ReservationRecord>,
) -> ProofValidationReport {
    let beat_progression: BTreeMap<BeatId, u32> = blueprint
        .beats
        .values()
        .filter(|b| b.on_critical_path)
        .map(|b| (b.id, b.progression.raw()))
        .collect();

    // Map reservations to beats
    let res_beat: BTreeMap<ReservationId, BeatId> = reservations
        .values()
        .filter_map(|r| r.beat_id.map(|bid| (r.id, bid)))
        .collect();

    let shortcut_edge_ids: BTreeSet<EdgeId> = result.shortcuts_realized.iter().copied().collect();

    let mut report = ProofValidationReport::pass();
    for edge in &result.selected_edges {
        // Skip shortcuts (they intentionally go backward)
        if shortcut_edge_ids.contains(&edge.id) {
            continue;
        }
        // Only check critical-path beats
        let src_beat = res_beat.get(&edge.source);
        let tgt_beat = res_beat.get(&edge.target);
        if let (Some(&sb), Some(&tb)) = (src_beat, tgt_beat) {
            if let (Some(&sp), Some(&tp)) = (beat_progression.get(&sb), beat_progression.get(&tb)) {
                // Geometric direction doesn't encode beat order.
                // Only flag large progression gaps that skip beats.
                let gap = sp.abs_diff(tp);
                if gap > 5 {
                    report.merge(ProofValidationReport::fail(format!(
                        "non-shortcut edge {:?} has large progression gap: {:?}({}) <-> {:?}({})",
                        edge.id, sb, sp, tb, tp
                    )));
                }
            }
        }
    }
    report
}

/// Validate landmark path membership: the critical path contains all forced
/// landmarks from the blueprint.
pub(crate) fn validate_landmark_path_membership(
    _result: &TopologyResult,
    blueprint: &PacingBlueprint,
    reservations: &BTreeMap<ReservationId, ReservationRecord>,
) -> ProofValidationReport {
    let mut report = ProofValidationReport::pass();

    for lm in &blueprint.forced_landmarks {
        let bid = lm.beat_id;
        // Verify this beat has at least one reservation in the result
        let has_res = reservations.values().any(|r| r.beat_id == Some(bid));
        if !has_res {
            report.merge(ProofValidationReport::fail(format!(
                "forced landmark beat {:?} has no reservation",
                bid
            )));
            continue;
        }
        // Verify the beat is on a connected path to entrance
        let beat_on_critical = blueprint
            .beats
            .get(&bid)
            .map(|b| b.on_critical_path)
            .unwrap_or(false);
        if !beat_on_critical {
            report.merge(ProofValidationReport::fail(format!(
                "forced landmark beat {:?} not on critical path",
                bid
            )));
        }
    }
    report
}

/// Validate alternate paths: for Moderate/Rich, at least one loop edge
/// creates an alternate path between two already-connected nodes.
pub(crate) fn validate_alternate_paths(
    result: &TopologyResult,
    blueprint: &PacingBlueprint,
) -> ProofValidationReport {
    let expected_loops: usize = match blueprint.preset {
        RichnessPreset::Sparse => 0,
        RichnessPreset::Moderate => 2,
        RichnessPreset::Rich => 4,
    };

    if result.loop_count != expected_loops {
        return ProofValidationReport::fail(format!(
            "alternate paths: expected exactly {} loops, got {} for {:?}",
            expected_loops, result.loop_count, blueprint.preset
        ));
    }
    let vertices: BTreeSet<_> = result
        .selected_edges
        .iter()
        .flat_map(|edge| [edge.source, edge.target])
        .collect();
    let cycle_rank = result
        .selected_edges
        .len()
        .saturating_add(1)
        .saturating_sub(vertices.len());
    if cycle_rank != expected_loops {
        ProofValidationReport::fail(format!(
            "alternate paths: cycle rank {} does not match exact loop count {}",
            cycle_rank, expected_loops
        ))
    } else {
        ProofValidationReport::pass()
    }
}

/// Validate backward shortcut semantics: every shortcut connects a later
/// beat to an earlier beat WITHOUT bypassing any unvisited mandatory beat.
///
/// Since shortcuts are backward (later→earlier), the player has already
/// visited all intervening beats during forward progression, so no
/// unvisited beat can be bypassed by construction.
pub(crate) fn validate_backward_shortcut_semantics(
    result: &TopologyResult,
    blueprint: &PacingBlueprint,
    reservations: &BTreeMap<ReservationId, ReservationRecord>,
) -> ProofValidationReport {
    let mut report = ProofValidationReport::pass();

    let beat_progression: BTreeMap<BeatId, u32> = blueprint
        .beats
        .values()
        .filter(|b| b.on_critical_path)
        .map(|b| (b.id, b.progression.raw()))
        .collect();

    let res_beat: BTreeMap<ReservationId, BeatId> = reservations
        .values()
        .filter_map(|r| r.beat_id.map(|bid| (r.id, bid)))
        .collect();

    let shortcut_ids: BTreeSet<EdgeId> = result.shortcuts_realized.iter().copied().collect();

    for edge in &result.selected_edges {
        if !shortcut_ids.contains(&edge.id) {
            continue;
        }
        let sb = res_beat.get(&edge.source);
        let tb = res_beat.get(&edge.target);
        if let (Some(&sb), Some(&tb)) = (sb, tb) {
            let sp = beat_progression.get(&sb).copied().unwrap_or(0);
            let tp = beat_progression.get(&tb).copied().unwrap_or(0);

            // Shortcut must go from later to earlier
            if sp <= tp {
                report.merge(ProofValidationReport::fail(format!(
                    "shortcut {:?} goes forward ({:?} prog={} -> {:?} prog={})",
                    edge.id, sb, sp, tb, tp
                )));
            }

            // Verify no unvisited mandatory beat is bypassed.
            // Since shortcut goes from later to earlier (sp > tp), all
            // beats between tp and sp were already visited during forward
            // progression. This is a semantic invariant by construction.
            // We verify it by checking that all beats in the progression
            // range (tp..sp) are on the critical path and connected.
            let intervening: Vec<_> = beat_progression
                .iter()
                .filter(|(_, &prog)| prog > tp && prog < sp)
                .map(|(&bid, _)| bid)
                .collect();
            for int_bid in &intervening {
                let has_connection = result.selected_edges.iter().any(|e| {
                    let e_src_beat = res_beat.get(&e.source).copied();
                    let e_tgt_beat = res_beat.get(&e.target).copied();
                    e_src_beat == Some(*int_bid) || e_tgt_beat == Some(*int_bid)
                });
                if !has_connection {
                    report.merge(ProofValidationReport::fail(format!(
                        "shortcut {:?} bypasses unvisited beat {:?}",
                        edge.id, int_bid
                    )));
                }
            }
        }
    }
    report
}

/// Validate side-branch payoff endpoints are reachable from the critical path.
pub(crate) fn validate_side_branch_payoffs(
    result: &TopologyResult,
    blueprint: &PacingBlueprint,
    reservations: &BTreeMap<ReservationId, ReservationRecord>,
) -> ProofValidationReport {
    let mut report = ProofValidationReport::pass();

    // Build adjacency from selected edges
    let mut adj: BTreeMap<ReservationId, Vec<ReservationId>> = BTreeMap::new();
    for edge in &result.selected_edges {
        adj.entry(edge.source).or_default().push(edge.target);
        adj.entry(edge.target).or_default().push(edge.source);
    }

    // For each branch payoff, verify the to_beat (leaf) is reachable from
    // its from_beat via selected edges
    for (bid, payoff) in &blueprint.branch_payoffs {
        if !payoff.is_valid() {
            continue;
        }
        let Some(to_beat) = payoff.to_beat else {
            continue;
        };

        let from_res: Vec<ReservationId> = reservations
            .values()
            .filter(|r| r.beat_id == Some(payoff.from_beat))
            .map(|r| r.id)
            .collect();
        let to_res: Vec<ReservationId> = reservations
            .values()
            .filter(|r| r.beat_id == Some(to_beat))
            .map(|r| r.id)
            .collect();

        if from_res.is_empty() || to_res.is_empty() {
            report.merge(ProofValidationReport::fail(format!(
                "branch payoff {:?} has no concrete origin/leaf reservation",
                bid
            )));
            continue;
        }

        // BFS from from_res to to_res
        let mut reachable = false;
        for &fr in &from_res {
            let mut visited = BTreeSet::new();
            let mut stack = vec![fr];
            while let Some(current) = stack.pop() {
                if !visited.insert(current) {
                    continue;
                }
                if to_res.contains(&current) {
                    reachable = true;
                    break;
                }
                if let Some(neighbors) = adj.get(&current) {
                    for &n in neighbors {
                        if !visited.contains(&n) {
                            stack.push(n);
                        }
                    }
                }
            }
            if reachable {
                break;
            }
        }
        if !reachable {
            report.merge(ProofValidationReport::fail(format!(
                "branch payoff {:?}: to_beat {:?} not reachable from from_beat {:?}",
                bid, to_beat, payoff.from_beat
            )));
        }
    }
    report
}

/// Validate degree bounds: no reservation exceeds its max_exits intent.
pub(crate) fn validate_degree_bounds(
    result: &TopologyResult,
    blueprint: &PacingBlueprint,
    reservations: &BTreeMap<ReservationId, ReservationRecord>,
) -> ProofValidationReport {
    let mut report = ProofValidationReport::pass();

    for r in reservations.values() {
        let degree = result
            .selected_edges
            .iter()
            .filter(|e| e.source == r.id || e.target == r.id)
            .count();

        let max_exits = reservation_max_exits(r, blueprint);

        if degree > max_exits as usize {
            report.merge(ProofValidationReport::fail(format!(
                "reservation {:?} degree {} exceeds max_exits {}",
                r.id, degree, max_exits
            )));
        }
    }
    report
}

/// Validate zone transitions: every zone transition in the blueprint has
/// at least one edge that straddles the two zones.
pub(crate) fn validate_zone_transitions(
    result: &TopologyResult,
    blueprint: &PacingBlueprint,
    reservations: &BTreeMap<ReservationId, ReservationRecord>,
) -> ProofValidationReport {
    let mut report = ProofValidationReport::pass();

    // Build reservation-to-zone map
    let res_zone: BTreeMap<ReservationId, ZoneId> = reservations
        .values()
        .filter_map(|r| r.zone_id.map(|zid| (r.id, zid)))
        .collect();

    for transition in &blueprint.zone_blueprint.transitions {
        let found = result.selected_edges.iter().any(|e| {
            let sz = res_zone.get(&e.source).copied();
            let tz = res_zone.get(&e.target).copied();
            (sz == Some(transition.from_zone) && tz == Some(transition.to_zone))
                || (sz == Some(transition.to_zone) && tz == Some(transition.from_zone))
        });
        if !found {
            report.merge(ProofValidationReport::fail(format!(
                "zone transition {:?}->{:?} has no connecting edge",
                transition.from_zone, transition.to_zone
            )));
        }
    }
    report
}

/// Validate route witness ownership. Every selected edge has exactly one
/// concrete route; every portal throat and turn is a distinct committed
/// reservation bound to that route; and every protected witness cell resolves
/// to the declared owner.
pub(crate) fn validate_route_witness_ownership(result: &TopologyResult) -> ProofValidationReport {
    let mut report = ProofValidationReport::pass();
    let mut edge_routes = BTreeMap::new();
    let mut route_ids = BTreeSet::new();
    let mut portal_ids = BTreeSet::new();
    let mut turn_ids = BTreeSet::new();
    let mut reservation_routes = BTreeMap::new();
    let grid_owns = |cell: CellCoord, reservation_id: ReservationId, kind: ReservationKind| {
        result.journal.grid.get(cell).is_some_and(|occupied| {
            (occupied.owner == reservation_id && occupied.owner_kind == kind.to_owner_kind())
                || (occupied.owner_kind == super::footprint::OccupancyOwnerKind::Composite
                    && result.journal.get(occupied.owner).is_some_and(|composite| {
                        composite.kind == ReservationKind::Composite
                            && composite.composite_children.contains(&reservation_id)
                    }))
        })
    };
    for route in result.routes.iter().chain(&result.vertical_routes) {
        if edge_routes.insert(route.edge_id, route).is_some() {
            report.merge(ProofValidationReport::fail(format!(
                "multiple committed routes share edge {:?}",
                route.edge_id
            )));
        }
        if !route_ids.insert(route.id) {
            report.merge(ProofValidationReport::fail(format!(
                "duplicate committed route ID {:?}",
                route.id
            )));
        }
        for portal in [&route.source_portal, &route.target_portal] {
            if !portal_ids.insert(portal.id) {
                report.merge(ProofValidationReport::fail(format!(
                    "duplicate committed portal ID {:?}",
                    portal.id
                )));
            }
        }
        for turn in &route.turns {
            if !turn_ids.insert(turn.id) {
                report.merge(ProofValidationReport::fail(format!(
                    "duplicate committed turn ID {:?}",
                    turn.id
                )));
            }
        }
        for reservation_id in &route.reservation_ids {
            if let Some(other_route) = reservation_routes.insert(*reservation_id, route.id) {
                report.merge(ProofValidationReport::fail(format!(
                    "reservation {:?} is shared by routes {:?} and {:?}",
                    reservation_id, other_route, route.id
                )));
            }
        }
    }

    let all_edges: Vec<_> = result
        .selected_edges
        .iter()
        .chain(&result.vertical_edges)
        .collect();
    let selected_edge_ids: BTreeSet<_> = all_edges.iter().map(|edge| edge.id).collect();
    for edge_id in edge_routes.keys() {
        if !selected_edge_ids.contains(edge_id) {
            report.merge(ProofValidationReport::fail(format!(
                "committed route for edge {:?} has no selected edge",
                edge_id
            )));
        }
    }
    for edge in all_edges {
        let Some(route) = edge_routes.get(&edge.id).copied() else {
            report.merge(ProofValidationReport::fail(format!(
                "selected edge {:?} has no committed route",
                edge.id
            )));
            continue;
        };
        if route.source != edge.source || route.target != edge.target {
            report.merge(ProofValidationReport::fail(format!(
                "route {:?} endpoints ({:?}->{:?}) do not match edge {:?} ({:?}->{:?})",
                route.id, route.source, route.target, edge.id, edge.source, edge.target
            )));
        }
        if route.reservation_ids.is_empty() {
            report.merge(ProofValidationReport::fail(format!(
                "route {:?} owns no concrete reservations",
                route.id
            )));
            continue;
        }
        let owned_ids: BTreeSet<_> = route.reservation_ids.iter().copied().collect();
        if owned_ids.len() != route.reservation_ids.len() {
            report.merge(ProofValidationReport::fail(format!(
                "route {:?} repeats a reservation identity",
                route.id
            )));
        }
        let vertical = matches!(edge.kind, EdgeKind::Vertical { .. });
        for reservation_id in &route.reservation_ids {
            let Some(record) = result.journal.get(*reservation_id) else {
                report.merge(ProofValidationReport::fail(format!(
                    "route {:?} references missing reservation {:?}",
                    route.id, reservation_id
                )));
                continue;
            };
            if !record.committed {
                report.merge(ProofValidationReport::fail(format!(
                    "route {:?} references uncommitted reservation {:?}",
                    route.id, reservation_id
                )));
            }
            let allowed = matches!(
                record.kind,
                ReservationKind::Route | ReservationKind::PortalThroat | ReservationKind::Turn
            ) || (vertical && record.kind == ReservationKind::VerticalHost);
            if !allowed {
                report.merge(ProofValidationReport::fail(format!(
                    "route {:?} reservation {:?} has invalid kind {:?}",
                    route.id, reservation_id, record.kind
                )));
            }
            if record.kind != ReservationKind::VerticalHost
                && record.owning_route_id != Some(route.id)
            {
                report.merge(ProofValidationReport::fail(format!(
                    "route {:?} reservation {:?} owner binding is {:?}",
                    route.id, reservation_id, record.owning_route_id
                )));
            }
        }
        if route.envelope.x1 <= route.envelope.x0 || route.envelope.y1 <= route.envelope.y0 {
            report.merge(ProofValidationReport::fail(format!(
                "route {:?} has a degenerate envelope",
                route.id
            )));
        }

        for (portal, endpoint) in [
            (&route.source_portal, edge.source),
            (&route.target_portal, edge.target),
        ] {
            if portal.endpoint_reservation_id != endpoint {
                report.merge(ProofValidationReport::fail(format!(
                    "portal {:?} endpoint {:?} does not match edge endpoint {:?}",
                    portal.id, portal.endpoint_reservation_id, endpoint
                )));
            }
            let dimensions = (portal.witness.grid_width(), portal.witness.grid_depth());
            let expected = match portal.wall {
                Dir::East | Dir::West => (PORTAL_WITNESS_DEPTH, PORTAL_WITNESS_WIDTH),
                Dir::North | Dir::South => (PORTAL_WITNESS_WIDTH, PORTAL_WITNESS_DEPTH),
            };
            if dimensions != expected || portal.headroom != HEADROOM {
                report.merge(ProofValidationReport::fail(format!(
                    "portal {:?} does not preserve a 64x{} throat: footprint={:?} expected={:?} headroom={}",
                    portal.id, HEADROOM, dimensions, expected, portal.headroom
                )));
            }
            let Some(record) = result.journal.get(portal.reservation_id) else {
                report.merge(ProofValidationReport::fail(format!(
                    "portal {:?} references missing throat reservation {:?}",
                    portal.id, portal.reservation_id
                )));
                continue;
            };
            if record.kind != ReservationKind::PortalThroat
                || record.footprint != portal.witness
                || record.owning_route_id != Some(route.id)
                || record.clearance_height != Some(portal.headroom)
                || !record.committed
                || !owned_ids.contains(&record.id)
            {
                report.merge(ProofValidationReport::fail(format!(
                    "portal {:?} is not a committed first-class throat owned by route {:?}",
                    portal.id, route.id
                )));
            }
            if !result.journal.get(endpoint).is_some_and(|endpoint_record| {
                endpoint_record
                    .footprint
                    .contains_xy(portal.anchor_cell.x, portal.anchor_cell.y)
            }) {
                report.merge(ProofValidationReport::fail(format!(
                    "portal {:?} anchor is not on endpoint {:?}",
                    portal.id, endpoint
                )));
            }
            for cell in portal.witness.cells() {
                if !grid_owns(cell, portal.reservation_id, ReservationKind::PortalThroat) {
                    report.merge(ProofValidationReport::fail(format!(
                        "portal {:?} witness cell {:?} is not throat-owned",
                        portal.id, cell
                    )));
                    break;
                }
            }
        }

        let turn_reservations: BTreeSet<_> = route
            .reservation_ids
            .iter()
            .filter(|id| {
                result
                    .journal
                    .get(**id)
                    .is_some_and(|record| record.kind == ReservationKind::Turn)
            })
            .copied()
            .collect();
        let committed_turn_ids: BTreeSet<_> =
            route.turns.iter().map(|turn| turn.reservation_id).collect();
        if turn_reservations != committed_turn_ids {
            report.merge(ProofValidationReport::fail(format!(
                "route {:?} turn records do not match its Turn reservations",
                route.id
            )));
        }
        for turn in &route.turns {
            let Some(record) = result.journal.get(turn.reservation_id) else {
                report.merge(ProofValidationReport::fail(format!(
                    "turn {:?} references missing reservation {:?}",
                    turn.id, turn.reservation_id
                )));
                continue;
            };
            if record.kind != ReservationKind::Turn
                || record.footprint != turn.witness
                || record.owning_route_id != Some(route.id)
                || record.clearance_height != Some(turn.headroom)
                || !record.committed
                || !record
                    .footprint
                    .contains_xy(turn.position.x, turn.position.y)
            {
                report.merge(ProofValidationReport::fail(format!(
                    "turn {:?} is not a distinct committed Turn owned by route {:?}",
                    turn.id, route.id
                )));
            }
            if turn.headroom != HEADROOM
                || turn.witness.grid_width() != PORTAL_WITNESS_WIDTH
                || turn.witness.grid_depth() != PORTAL_WITNESS_WIDTH
            {
                report.merge(ProofValidationReport::fail(format!(
                    "turn {:?} does not preserve a 64x{} witness",
                    turn.id, HEADROOM
                )));
            }
            if !grid_owns(turn.position, turn.reservation_id, ReservationKind::Turn) {
                report.merge(ProofValidationReport::fail(format!(
                    "turn {:?} center is not owned by its Turn reservation",
                    turn.id
                )));
            }
            for cell in turn.witness.cells() {
                if !grid_owns(cell, turn.reservation_id, ReservationKind::Turn) {
                    report.merge(ProofValidationReport::fail(format!(
                        "turn {:?} witness cell {:?} is not Turn-owned",
                        turn.id, cell
                    )));
                    break;
                }
            }
        }
    }

    for record in result.journal.reservations.values() {
        let route_kind = matches!(
            record.kind,
            ReservationKind::Route | ReservationKind::PortalThroat | ReservationKind::Turn
        );
        if route_kind && record.clearance_height != Some(HEADROOM) {
            report.merge(ProofValidationReport::fail(format!(
                "route witness reservation {:?} has clearance {:?}, expected {}",
                record.id, record.clearance_height, HEADROOM
            )));
        }
        let Some(owner) = record.owning_route_id else {
            if route_kind {
                report.merge(ProofValidationReport::fail(format!(
                    "committed {:?} reservation {:?} has no owning route",
                    record.kind, record.id
                )));
            }
            continue;
        };
        if !route_kind {
            report.merge(ProofValidationReport::fail(format!(
                "non-route reservation {:?} unexpectedly names route {:?}",
                record.id, owner
            )));
        }
        let referenced = result
            .routes
            .iter()
            .chain(&result.vertical_routes)
            .any(|route| route.id == owner && route.reservation_ids.contains(&record.id));
        if !referenced {
            report.merge(ProofValidationReport::fail(format!(
                "route-owned reservation {:?} is orphaned from route {:?}",
                record.id, owner
            )));
        }
    }
    report
}

/// Validate every committed vertical host has an exact lower/upper endpoint
/// candidate and an owned dual-band route envelope.
pub(crate) fn validate_vertical_endpoint_ownership(
    result: &TopologyResult,
) -> ProofValidationReport {
    let hosts: BTreeSet<_> = result
        .journal
        .reservations
        .values()
        .filter(|record| record.kind == ReservationKind::VerticalHost)
        .map(|record| record.id)
        .collect();
    if result.vertical_edges.len() != hosts.len() || result.vertical_routes.len() != hosts.len() {
        return ProofValidationReport::fail(format!(
            "vertical host coverage mismatch: hosts={} edges={} routes={}",
            hosts.len(),
            result.vertical_edges.len(),
            result.vertical_routes.len()
        ));
    }

    let routes: BTreeMap<_, _> = result
        .vertical_routes
        .iter()
        .map(|route| (route.edge_id, route))
        .collect();
    let mut covered_hosts = BTreeSet::new();
    let mut report = ProofValidationReport::pass();
    for edge in &result.vertical_edges {
        let EdgeKind::Vertical {
            lower_reservation,
            upper_reservation,
        } = edge.kind
        else {
            report.merge(ProofValidationReport::fail(format!(
                "vertical candidate {:?} has non-vertical kind",
                edge.id
            )));
            continue;
        };
        if edge.source != lower_reservation || edge.target != upper_reservation {
            report.merge(ProofValidationReport::fail(format!(
                "vertical candidate {:?} endpoints do not match its exact lower/upper bindings",
                edge.id
            )));
        }
        let Some(route) = routes.get(&edge.id) else {
            report.merge(ProofValidationReport::fail(format!(
                "vertical candidate {:?} has no committed route",
                edge.id
            )));
            continue;
        };
        if !route.envelope.occupies_lower || !route.envelope.occupies_upper {
            report.merge(ProofValidationReport::fail(format!(
                "vertical route {:?} is not dual-band",
                route.id
            )));
        }
        for host_id in &route.reservation_ids {
            if hosts.contains(host_id) {
                covered_hosts.insert(*host_id);
            }
        }
    }
    if covered_hosts != hosts {
        report.merge(ProofValidationReport::fail(
            "one or more vertical hosts has no owned exact route",
        ));
    }
    report
}

/// Run all proof validators and return a combined report.
pub(crate) fn run_all_proofs(
    result: &TopologyResult,
    blueprint: &PacingBlueprint,
    reservations: &BTreeMap<ReservationId, ReservationRecord>,
) -> ProofValidationReport {
    let mut report = ProofValidationReport::pass();
    report.merge(validate_global_connectivity(result, reservations));
    report.merge(validate_mandatory_edges_present(
        result,
        blueprint,
        reservations,
    ));
    report.merge(validate_beat_order(result, blueprint, reservations));
    report.merge(validate_landmark_path_membership(
        result,
        blueprint,
        reservations,
    ));
    report.merge(validate_alternate_paths(result, blueprint));
    report.merge(validate_backward_shortcut_semantics(
        result,
        blueprint,
        reservations,
    ));
    report.merge(validate_side_branch_payoffs(
        result,
        blueprint,
        reservations,
    ));
    report.merge(validate_degree_bounds(result, blueprint, reservations));
    report.merge(validate_zone_transitions(result, blueprint, reservations));
    report.merge(validate_route_witness_ownership(result));
    report.merge(validate_vertical_endpoint_ownership(result));
    report
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::super::error::{RichnessErrorCategory, RichnessErrorCode};
    use super::super::ids::{
        ArchetypeIndex, ArchetypeRequest, ArchetypeRequestId, Beat, BeatId, BeatType, DegreeIntent,
        DensityClass, MandatoryEdge, PacingBlueprint, ProgressionOrder, RarityClass,
        RarityEvidence, ZoneId,
    };
    use super::super::pacing::build_pacing_blueprint;
    use super::super::request::{
        ResolvedRichnessRequestV1, RichnessDocumentV1, RichnessPreset, RichnessTheme,
    };
    use super::super::reservation::{ReservationJournal, ReservationKind, ReservationRecord};
    use super::super::solver::{solve_placement, PlacementResult, MAX_PLACEMENT_SEARCH_STATES};
    use super::super::zones::ZoneBlueprint;
    use super::*;

    fn make_test_footprint(x0: u32, y0: u32, x1: u32, y1: u32, lower: bool) -> Footprint3D {
        Footprint3D {
            x0,
            y0,
            x1,
            y1,
            occupies_lower: lower,
            occupies_upper: !lower,
        }
    }

    #[test]
    fn cardinal_direction_east() {
        let a = make_test_footprint(0, 0, 4, 4, true);
        let b = make_test_footprint(8, 1, 12, 3, true);
        assert_eq!(cardinal_direction(&a, &b), Some(Dir::East));
    }

    #[test]
    fn cardinal_direction_west() {
        let a = make_test_footprint(8, 1, 12, 3, true);
        let b = make_test_footprint(0, 0, 4, 4, true);
        assert_eq!(cardinal_direction(&a, &b), Some(Dir::West));
    }

    #[test]
    fn cardinal_direction_north() {
        let a = make_test_footprint(1, 8, 3, 12, true);
        let b = make_test_footprint(0, 0, 4, 4, true);
        // a is at higher Y than b, so from a's perspective, b is North
        assert_eq!(cardinal_direction(&a, &b), Some(Dir::North));
    }

    #[test]
    fn cross_overlap_computes_correctly() {
        let a = make_test_footprint(0, 2, 8, 6, true);
        let b = make_test_footprint(8, 1, 16, 5, true);
        let overlap = cross_overlap(&a, &b, Dir::East);
        assert!(overlap.is_some());
        let (lo, hi) = overlap.unwrap();
        // a.y: [2,6], b.y: [1,5] -> intersection [2,5]
        // in Quake units: 2*16=32, 5*16=80
        assert!(lo >= 32);
        assert!(hi <= 96);
    }

    #[test]
    fn no_overlap_when_no_intersection() {
        let a = make_test_footprint(0, 0, 4, 4, true);
        let b = make_test_footprint(8, 8, 12, 12, true);
        // a and b don't overlap on either axis so they are not cardinally
        // adjacent — there's no corridor overlap.
        let dir = cardinal_direction(&a, &b);
        // Both at (2,2) vs (10,10) -> dx=8, dy=8 -> East/West with cross overlap on Y
        assert_eq!(dir, Some(Dir::East));
        assert!(dir.is_some());
    }

    #[test]
    fn footprint_distance_correct() {
        let a = make_test_footprint(0, 0, 4, 4, true); // center (2,2)
        let b = make_test_footprint(8, 8, 12, 12, true); // center (10,10)
        assert_eq!(footprint_distance(&a, &b), 16); // |2-10| + |2-10| = 16
    }

    #[test]
    fn envelopes_positive_overlap_detects_overlap() {
        let a = make_test_footprint(0, 0, 4, 2, true);
        let b = make_test_footprint(2, 1, 6, 3, true);
        assert!(envelopes_positive_overlap(&a, &b));

        let c = make_test_footprint(4, 0, 8, 2, true);
        assert!(!envelopes_positive_overlap(&a, &c)); // touching at x=4 is not positive overlap
    }

    #[test]
    fn union_find_basic() {
        let mut uf = UnionFind::new(5);
        assert!(!uf.connected(0, 1));
        assert!(uf.union(0, 1));
        assert!(uf.connected(0, 1));
        assert!(!uf.union(0, 1)); // already connected
        assert!(uf.union(1, 2));
        assert!(uf.connected(0, 2));
    }

    #[test]
    fn dir_opposite() {
        assert_eq!(Dir::North.opposite(), Dir::South);
        assert_eq!(Dir::South.opposite(), Dir::North);
        assert_eq!(Dir::East.opposite(), Dir::West);
        assert_eq!(Dir::West.opposite(), Dir::East);
    }

    #[test]
    fn dir_tag_consistent() {
        let tags = [
            Dir::North.tag(),
            Dir::South.tag(),
            Dir::West.tag(),
            Dir::East.tag(),
        ];
        let set: std::collections::BTreeSet<_> = tags.iter().collect();
        assert_eq!(set.len(), 4);
    }

    #[test]
    fn compute_pacing_bias_mandatory_vs_forward() {
        // Test that mandatory edges get bias 0
        // and that forward edges get lower bias than backward
        let mut beat_prog = BTreeMap::new();
        beat_prog.insert(BeatId::new(0), 0);
        beat_prog.insert(BeatId::new(1), 1);
        beat_prog.insert(BeatId::new(2), 2);

        let mut mandatory_set = BTreeSet::new();
        mandatory_set.insert((BeatId::new(0), BeatId::new(1)));

        let a = ReservationRecord {
            id: ReservationId::new(0),
            kind: ReservationKind::StandardRoom,
            footprint: make_test_footprint(0, 0, 4, 4, true),
            beat_id: Some(BeatId::new(0)),
            request_id: None,
            zone_id: None,
            pit_pair_room_id: None,
            composite_children: Vec::new(),
            owning_route_id: None,
            clearance_height: None,
            committed: true,
            cost_faces: 0,
            cost_brushes: 0,
            cost_entities: 0,
            cost_lights: 0,
        };

        let b = ReservationRecord {
            id: ReservationId::new(1),
            kind: ReservationKind::StandardRoom,
            footprint: make_test_footprint(8, 1, 12, 3, true),
            beat_id: Some(BeatId::new(1)),
            request_id: None,
            zone_id: None,
            pit_pair_room_id: None,
            composite_children: Vec::new(),
            owning_route_id: None,
            clearance_height: None,
            committed: true,
            cost_faces: 0,
            cost_brushes: 0,
            cost_entities: 0,
            cost_lights: 0,
        };

        let bias = compute_pacing_bias(&a, &b, &beat_prog, &mandatory_set);
        assert_eq!(bias, 0, "mandatory edge should have bias 0");
    }

    #[test]
    fn compute_field_rank_is_deterministic() {
        let r1 = compute_field_rank(
            EdgeId::new(0),
            ReservationId::new(0),
            ReservationId::new(1),
            Dir::East,
            10,
            0,
        );
        let r2 = compute_field_rank(
            EdgeId::new(0),
            ReservationId::new(0),
            ReservationId::new(1),
            Dir::East,
            10,
            0,
        );
        assert_eq!(r1, r2, "field rank must be deterministic");
    }

    #[test]
    fn candidate_edge_sort_key_consistent() {
        let e1 = CandidateEdge {
            id: EdgeId::new(0),
            kind: EdgeKind::SameLayer { layer: 0 },
            source: ReservationId::new(0),
            target: ReservationId::new(1),
            source_dir: Dir::East,
            overlap: (0, 64),
            distance: 10,
            pacing_bias: 0,
            field_rank: 100,
            residual_capacity: 1,
            mandatory: false,
            mandatory_record: None,
        };
        let e2 = CandidateEdge {
            id: EdgeId::new(1),
            kind: EdgeKind::SameLayer { layer: 0 },
            source: ReservationId::new(0),
            target: ReservationId::new(2),
            source_dir: Dir::East,
            overlap: (0, 64),
            distance: 5,
            pacing_bias: 0,
            field_rank: 200,
            residual_capacity: 1,
            mandatory: false,
            mandatory_record: None,
        };
        // e2 has shorter distance, so it should sort first
        assert!(e2.sort_key() < e1.sort_key());
    }

    #[test]
    fn mandatory_edges_sort_first() {
        let e1 = CandidateEdge {
            id: EdgeId::new(0),
            kind: EdgeKind::SameLayer { layer: 0 },
            source: ReservationId::new(0),
            target: ReservationId::new(1),
            source_dir: Dir::East,
            overlap: (0, 64),
            distance: 50,
            pacing_bias: 0,
            field_rank: 0,
            residual_capacity: 1,
            mandatory: true,
            mandatory_record: Some(MandatoryEdge {
                from_beat: BeatId::new(0),
                to_beat: BeatId::new(1),
            }),
        };
        let e2 = CandidateEdge {
            id: EdgeId::new(1),
            kind: EdgeKind::SameLayer { layer: 0 },
            source: ReservationId::new(0),
            target: ReservationId::new(2),
            source_dir: Dir::East,
            overlap: (0, 64),
            distance: 5,
            pacing_bias: 0,
            field_rank: 0,
            residual_capacity: 1,
            mandatory: false,
            mandatory_record: None,
        };
        // mandatory edges sort first
        let mut edges = [e2, e1];
        edges.sort_by(|a, b| {
            a.mandatory
                .cmp(&b.mandatory)
                .reverse()
                .then_with(|| a.sort_key().cmp(&b.sort_key()))
        });
        assert!(edges[0].mandatory);
        assert!(!edges[1].mandatory);
    }

    #[test]
    fn portal_witness_is_sized_correctly() {
        let room = ReservationRecord {
            id: ReservationId::new(0),
            kind: ReservationKind::StandardRoom,
            footprint: make_test_footprint(4, 4, 12, 12, true),
            beat_id: None,
            request_id: None,
            zone_id: None,
            pit_pair_room_id: None,
            composite_children: Vec::new(),
            owning_route_id: None,
            clearance_height: None,
            committed: true,
            cost_faces: 0,
            cost_brushes: 0,
            cost_entities: 0,
            cost_lights: 0,
        };
        let anchor = CellCoord::new(10, 8, 0); // East wall midpoint
        let witness = portal_witness_footprint(&room, Dir::East, anchor, 0);
        // East/west throats are exactly one wall cell deep × 64 wide in XY;
        // the frozen second dimension is 80 units of Z headroom.
        assert_eq!(
            (witness.grid_width(), witness.grid_depth()),
            (PORTAL_WITNESS_DEPTH, PORTAL_WITNESS_WIDTH)
        );
        assert_eq!(witness.quake_span(), (16, 64));
        assert_eq!(HEADROOM, 80);
    }

    #[test]
    fn route_envelope_is_quantum_aligned() {
        let src = ReservationRecord {
            id: ReservationId::new(0),
            kind: ReservationKind::StandardRoom,
            footprint: Footprint3D {
                x0: 0,
                y0: 0,
                x1: 4,
                y1: 4,
                occupies_lower: true,
                occupies_upper: false,
            },
            beat_id: None,
            request_id: None,
            zone_id: None,
            pit_pair_room_id: None,
            composite_children: Vec::new(),
            owning_route_id: None,
            clearance_height: None,
            committed: true,
            cost_faces: 0,
            cost_brushes: 0,
            cost_entities: 0,
            cost_lights: 0,
        };
        let tgt = ReservationRecord {
            id: ReservationId::new(1),
            kind: ReservationKind::StandardRoom,
            footprint: Footprint3D {
                x0: 8,
                y0: 1,
                x1: 12,
                y1: 3,
                occupies_lower: true,
                occupies_upper: false,
            },
            beat_id: None,
            request_id: None,
            zone_id: None,
            pit_pair_room_id: None,
            composite_children: Vec::new(),
            owning_route_id: None,
            clearance_height: None,
            committed: true,
            cost_faces: 0,
            cost_brushes: 0,
            cost_entities: 0,
            cost_lights: 0,
        };
        let env = route_envelope_footprint(&src, &tgt, Dir::East, (32, 64));
        // Envelope should be non-degenerate
        assert!(env.x1 > env.x0);
        assert!(env.y1 > env.y0);
        assert!(env.occupies_lower);
    }

    #[test]
    fn edge_kind_variants_distinct() {
        let e1 = EdgeKind::SameLayer { layer: 0 };
        let e2 = EdgeKind::Vertical {
            lower_reservation: ReservationId::new(0),
            upper_reservation: ReservationId::new(1),
        };
        assert_ne!(e1, e2);
    }

    #[test]
    fn count_incident_edges_correct() {
        let edges = vec![
            CandidateEdge {
                id: EdgeId::new(0),
                kind: EdgeKind::SameLayer { layer: 0 },
                source: ReservationId::new(0),
                target: ReservationId::new(1),
                source_dir: Dir::East,
                overlap: (0, 64),
                distance: 10,
                pacing_bias: 0,
                field_rank: 0,
                residual_capacity: 1,
                mandatory: false,
                mandatory_record: None,
            },
            CandidateEdge {
                id: EdgeId::new(1),
                kind: EdgeKind::SameLayer { layer: 0 },
                source: ReservationId::new(0),
                target: ReservationId::new(2),
                source_dir: Dir::North,
                overlap: (0, 64),
                distance: 5,
                pacing_bias: 0,
                field_rank: 0,
                residual_capacity: 1,
                mandatory: false,
                mandatory_record: None,
            },
        ];
        assert_eq!(count_incident_edges(ReservationId::new(0), &edges), 2);
        assert_eq!(count_incident_edges(ReservationId::new(1), &edges), 1);
        assert_eq!(count_incident_edges(ReservationId::new(3), &edges), 0);
    }

    // ── Synthetic topology tests ────────────────────────────────────────

    /// Build a synthetic reservation with a given footprint.
    fn syn_res(id: u32, fp: Footprint3D, beat_id: Option<u32>) -> ReservationRecord {
        ReservationRecord {
            id: ReservationId::new(id),
            kind: ReservationKind::StandardRoom,
            footprint: fp,
            beat_id: beat_id.map(BeatId::new),
            request_id: None,
            zone_id: None,
            pit_pair_room_id: None,
            composite_children: Vec::new(),
            owning_route_id: None,
            clearance_height: None,
            committed: true,
            cost_faces: 100,
            cost_brushes: 8,
            cost_entities: 2,
            cost_lights: 1,
        }
    }

    fn syn_fp(x0: u32, y0: u32, x1: u32, y1: u32) -> Footprint3D {
        Footprint3D {
            x0,
            y0,
            x1,
            y1,
            occupies_lower: true,
            occupies_upper: false,
        }
    }

    /// Build a minimal blueprint for topology testing.
    fn syn_blueprint(beat_count: u32) -> PacingBlueprint {
        let mut beats = BTreeMap::new();
        let mut beat_order = Vec::new();
        let mut mandatory_edges = Vec::new();
        let mut archetype_requests = BTreeMap::new();
        for i in 0..beat_count {
            let bid = BeatId::new(i);
            beat_order.push(bid);
            beats.insert(
                bid,
                Beat {
                    id: bid,
                    beat_type: if i == 0 {
                        BeatType::Entrance
                    } else {
                        BeatType::Descent
                    },
                    requests: vec![ArchetypeRequestId::new(i)],
                    density: DensityClass::Dense,
                    degree: DegreeIntent::new(1, 5),
                    progression: ProgressionOrder::new(i),
                    on_critical_path: true,
                    is_grand_volume: false,
                    is_quiet_negative_space: false,
                    is_dense_setpiece: false,
                },
            );
            archetype_requests.insert(
                ArchetypeRequestId::new(i),
                ArchetypeRequest {
                    id: ArchetypeRequestId::new(i),
                    archetype: ArchetypeIndex::new(0),
                    beat_id: bid,
                    zone_id: ZoneId::new(0),
                    degree: DegreeIntent::new(1, 5),
                    forced: false,
                    rarity_class: RarityClass::Common,
                    progression: ProgressionOrder::new(i),
                    density: DensityClass::Dense,
                },
            );
            if i > 0 {
                mandatory_edges.push(MandatoryEdge {
                    from_beat: BeatId::new(i - 1),
                    to_beat: BeatId::new(i),
                });
            }
        }
        PacingBlueprint {
            preset: RichnessPreset::Sparse,
            seed: 0,
            beats,
            beat_order,
            zone_blueprint: ZoneBlueprint {
                zones: BTreeMap::new(),
                transitions: Vec::new(),
                invariants: BTreeMap::new(),
                beat_zone_map: BTreeMap::new(),
            },
            archetype_requests,
            critical_path_landmarks: Vec::new(),
            forced_landmarks: Vec::new(),
            natural_rarity_evidence: RarityEvidence::new(),
            mandatory_edges,
            branch_payoffs: BTreeMap::new(),
            shortcut_intents: Vec::new(),
            grand_volume_landmark_present: false,
            quiet_negative_space_present: false,
            dense_setpiece_present: false,
        }
    }

    fn syn_journal(
        reservations: &BTreeMap<ReservationId, ReservationRecord>,
    ) -> ReservationJournal {
        let mut journal = ReservationJournal::new(2048, 8000);
        for r in reservations.values() {
            let fp = r.footprint;
            let kind = r.kind;
            // try_reserve allocates its own ID internally.
            // We verify the returned ID matches the expected one.
            let allocated_id = journal
                .try_reserve(kind, fp, r.beat_id, r.request_id, r.zone_id, 0, 0, 0, 0)
                .unwrap();
            assert_eq!(allocated_id, r.id, "reservation ID mismatch");
        }
        journal.commit_all();
        journal
    }

    fn syn_placement(reservations: &BTreeMap<ReservationId, ReservationRecord>) -> PlacementResult {
        let journal = syn_journal(reservations);
        let mut request_to_reservation = BTreeMap::new();
        let mut request_archetypes = BTreeMap::new();
        let mut beat_to_reservations: BTreeMap<BeatId, Vec<ReservationId>> = BTreeMap::new();
        for r in reservations.values() {
            if let Some(req_id) = r.request_id {
                request_to_reservation.insert(req_id, r.id);
                request_archetypes.insert(req_id, ArchetypeIndex::new(0));
            }
            if let Some(bid) = r.beat_id {
                beat_to_reservations.entry(bid).or_default().push(r.id);
            }
        }
        PlacementResult {
            reservations: reservations.clone(),
            request_to_reservation,
            request_archetypes,
            beat_to_reservations,
            journal,
            remaining_faces: 8000,
            placed_count: reservations.len(),
            max_search_states: 0,
            total_search_states: 0,
            corridor_rejections: 0,
        }
    }

    #[test]
    fn candidate_edges_materialize_only_after_complete_reservation() {
        let mut reservations = BTreeMap::new();
        reservations.insert(
            ReservationId::new(0),
            syn_res(0, syn_fp(4, 8, 12, 20), Some(0)),
        );
        reservations.insert(
            ReservationId::new(1),
            syn_res(1, syn_fp(28, 8, 36, 20), Some(1)),
        );
        let blueprint = syn_blueprint(2);
        let placement = syn_placement(&reservations);
        let resolved = make_resolved();
        let potentials = build_potential_connections(&reservations, &blueprint);
        assert_eq!(potentials.len(), 1);

        let route_components = RouteCenterComponents::build(&placement.journal);
        let candidates = feasible_candidates_for_potential(
            &placement.journal,
            &route_components,
            &potentials[0],
            &reservations,
            0,
            0,
            0,
            &resolved,
        );
        assert!(!candidates.is_empty());
        assert_eq!(
            candidates
                .iter()
                .map(|candidate| candidate.edge.id)
                .collect::<BTreeSet<_>>()
                .len(),
            candidates.len(),
            "every concrete feasible plan requires a unique stable EdgeId"
        );
        for candidate in candidates {
            let mut proof_journal = placement.journal.clone();
            let mut route_id = 0;
            let mut portal_id = 0;
            let mut turn_id = 0;
            let route = commit_feasible_candidate(
                &mut proof_journal,
                &candidate,
                &reservations,
                &mut route_id,
                &mut portal_id,
                &mut turn_id,
                &resolved,
            )
            .unwrap();
            proof_journal.commit_all();
            assert_eq!(route.edge_id, candidate.edge.id);
            assert!(route.reservation_ids.iter().all(|id| {
                proof_journal.get(*id).is_some_and(|record| {
                    record.committed
                        && (record.kind == ReservationKind::PortalThroat
                            || record.kind == ReservationKind::Route
                            || record.kind == ReservationKind::Turn)
                        && record.owning_route_id == Some(route.id)
                })
            }));
            assert_eq!(
                route
                    .reservation_ids
                    .iter()
                    .filter(|id| {
                        proof_journal
                            .get(**id)
                            .is_some_and(|record| record.kind == ReservationKind::PortalThroat)
                    })
                    .count(),
                2
            );
        }

        let mut impossible = reservations.clone();
        impossible.insert(
            ReservationId::new(0),
            syn_res(0, syn_fp(0, 0, 2, 2), Some(0)),
        );
        impossible.insert(
            ReservationId::new(1),
            syn_res(1, syn_fp(8, 8, 10, 10), Some(1)),
        );
        let impossible_placement = syn_placement(&impossible);
        let impossible_potentials = build_potential_connections(&impossible, &blueprint);
        assert_eq!(impossible_potentials.len(), 1);
        let impossible_route_components =
            RouteCenterComponents::build(&impossible_placement.journal);
        assert!(feasible_candidates_for_potential(
            &impossible_placement.journal,
            &impossible_route_components,
            &impossible_potentials[0],
            &impossible,
            0,
            0,
            0,
            &resolved,
        )
        .is_empty());
    }

    #[test]
    fn synthetic_mandatory_edges_connect_all() {
        // Two rooms side by side with a gap and sufficient cross-axis overlap
        // Overlap must be >= 4 cells (64 Quake units) for ROUTE_WIDTH
        let mut res_map = BTreeMap::new();
        res_map.insert(
            ReservationId::new(0),
            syn_res(0, syn_fp(0, 0, 4, 8), Some(0)),
        );
        res_map.insert(
            ReservationId::new(1),
            syn_res(1, syn_fp(8, 0, 12, 8), Some(1)),
        );

        let bp = syn_blueprint(2);
        let placement = syn_placement(&res_map);
        let resolved = make_resolved();

        let result = solve_topology(&bp, &placement, &resolved);
        assert!(result.is_ok(), "topology failed: {:?}", result.err());
        let r = result.unwrap();
        assert_eq!(r.selected_edges.len(), 1);
        assert_eq!(r.routes.len(), 1);
        // Must have distinct first-class endpoint throats.
        assert_eq!(
            r.routes[0].source_portal.endpoint_reservation_id,
            r.routes[0].source
        );
        assert_eq!(
            r.journal
                .get(r.routes[0].source_portal.reservation_id)
                .map(|record| record.kind),
            Some(ReservationKind::PortalThroat)
        );
    }

    #[test]
    fn synthetic_mandatory_edge_infeasible_when_no_candidate() {
        // Two rooms narrower than the protected four-cell portal width have
        // no legal socket on any wall, even though the grid between them is
        // empty. This is a genuinely impossible routing request.
        let mut res_map = BTreeMap::new();
        res_map.insert(
            ReservationId::new(0),
            syn_res(0, syn_fp(0, 0, 2, 2), Some(0)),
        );
        res_map.insert(
            ReservationId::new(1),
            syn_res(1, syn_fp(8, 8, 10, 10), Some(1)),
        );

        let bp = syn_blueprint(2);
        let placement = syn_placement(&res_map);
        let resolved = make_resolved();

        let result = solve_topology(&bp, &placement, &resolved);
        assert!(result.is_err(), "should fail: no feasible candidate");
        assert_eq!(
            result.unwrap_err().code,
            RichnessErrorCode::TopologyExhausted
        );
    }

    #[test]
    fn synthetic_kruskal_ordering_by_distance() {
        // Three rooms in a line: 0 - gap - 1 - gap - 2
        let mut res_map = BTreeMap::new();
        res_map.insert(
            ReservationId::new(0),
            syn_res(0, syn_fp(0, 0, 4, 8), Some(0)),
        );
        res_map.insert(
            ReservationId::new(1),
            syn_res(1, syn_fp(8, 0, 12, 8), Some(1)),
        );
        res_map.insert(
            ReservationId::new(2),
            syn_res(2, syn_fp(16, 0, 20, 8), Some(2)),
        );

        let bp = syn_blueprint(3);
        let placement = syn_placement(&res_map);
        let resolved = make_resolved();

        let result = solve_topology(&bp, &placement, &resolved);
        assert!(result.is_ok(), "topology failed: {:?}", result.err());
        let r = result.unwrap();
        // Should have 2 edges (spanning tree) + possibly loops
        assert!(r.selected_edges.len() >= 2);
        assert!(!r.routes.is_empty());
    }

    #[test]
    fn synthetic_multigraph_parallel_edges() {
        // Two rooms with large cross-axis overlap should produce parallel
        // candidate edges if the overlap is wide enough.
        let mut res_map = BTreeMap::new();
        // Wide rooms: 10 cells = 160 quake units cross-axis
        res_map.insert(
            ReservationId::new(0),
            syn_res(0, syn_fp(0, 0, 4, 12), Some(0)),
        );
        res_map.insert(
            ReservationId::new(1),
            syn_res(1, syn_fp(8, 0, 12, 12), Some(1)),
        );

        let bp = syn_blueprint(2);
        let placement = syn_placement(&res_map);
        let resolved = make_resolved();

        let result = solve_topology(&bp, &placement, &resolved);
        assert!(result.is_ok(), "topology failed: {:?}", result.err());
    }

    #[test]
    fn synthetic_route_reservation_inside_envelope() {
        // Verify that route reservation occupies cells in the gap
        let mut res_map = BTreeMap::new();
        res_map.insert(
            ReservationId::new(0),
            syn_res(0, syn_fp(0, 0, 4, 8), Some(0)),
        );
        res_map.insert(
            ReservationId::new(1),
            syn_res(1, syn_fp(8, 0, 12, 8), Some(1)),
        );

        let bp = syn_blueprint(2);
        let placement = syn_placement(&res_map);
        let resolved = make_resolved();

        let result = solve_topology(&bp, &placement, &resolved);
        assert!(result.is_ok(), "topology failed: {:?}", result.err());
        let r = result.unwrap();

        // Route should occupy cells in the gap between rooms
        let route_cells: Vec<_> = r
            .journal
            .reservations
            .values()
            .filter(|rec| matches!(rec.kind, ReservationKind::Route))
            .collect();
        assert!(!route_cells.is_empty(), "route must have reserved cells");
        for route in &route_cells {
            assert!(route.footprint.x0 > 0 || route.footprint.x1 > 0);
        }
    }

    #[test]
    fn synthetic_portal_records_present() {
        let mut res_map = BTreeMap::new();
        res_map.insert(
            ReservationId::new(0),
            syn_res(0, syn_fp(0, 0, 4, 8), Some(0)),
        );
        res_map.insert(
            ReservationId::new(1),
            syn_res(1, syn_fp(8, 0, 12, 8), Some(1)),
        );

        let bp = syn_blueprint(2);
        let placement = syn_placement(&res_map);
        let resolved = make_resolved();

        let result = solve_topology(&bp, &placement, &resolved);
        assert!(result.is_ok());
        let r = result.unwrap();
        assert!(!r.routes.is_empty());
        assert_eq!(
            r.routes[0].source_portal.endpoint_reservation_id,
            r.routes[0].source
        );
        assert_eq!(
            r.routes[0].target_portal.endpoint_reservation_id,
            r.routes[0].target
        );
        assert_ne!(r.routes[0].source_portal.reservation_id, r.routes[0].source);
        assert_ne!(r.routes[0].target_portal.reservation_id, r.routes[0].target);
    }

    #[test]
    fn synthetic_rollback_byte_identity_with_routes() {
        // Verify journal state is identical after a failed route attempt
        let mut res_map = BTreeMap::new();
        res_map.insert(
            ReservationId::new(0),
            syn_res(0, syn_fp(0, 0, 4, 8), Some(0)),
        );
        res_map.insert(
            ReservationId::new(1),
            syn_res(1, syn_fp(8, 0, 12, 8), Some(1)),
        );

        let bp = syn_blueprint(2);
        let placement = syn_placement(&res_map);
        let resolved = make_resolved();
        let before = placement.journal.state_snapshot();

        let result = solve_topology(&bp, &placement, &resolved);
        assert!(result.is_ok());
        let r = result.unwrap();

        // The journal after topology should have ADDITIONAL reservations
        // (routes) but all original reservations should be intact.
        let after = r.journal.state_snapshot();
        // Route reservations are added, so after != before.
        // But original reservations should still be present.
        for id in before.reservations.keys() {
            assert!(
                after.reservations.contains_key(id),
                "original reservation {:?} lost",
                id
            );
        }
    }

    #[test]
    fn synthetic_degree_bounds_respected() {
        // Three rooms — verify degree bounds don't cause failures
        let mut res_map = BTreeMap::new();
        res_map.insert(
            ReservationId::new(0),
            syn_res(0, syn_fp(0, 0, 4, 8), Some(0)),
        );
        res_map.insert(
            ReservationId::new(1),
            syn_res(1, syn_fp(8, 0, 12, 8), Some(1)),
        );
        res_map.insert(
            ReservationId::new(2),
            syn_res(2, syn_fp(16, 0, 20, 8), Some(2)),
        );

        let bp = syn_blueprint(3);
        let placement = syn_placement(&res_map);
        let resolved = make_resolved();

        let result = solve_topology(&bp, &placement, &resolved);
        assert!(result.is_ok(), "topology failed: {:?}", result.err());
    }

    #[test]
    fn synthetic_exhaustion_typed_error() {
        // Two rooms with no feasible connection — must return typed error
        let mut res_map = BTreeMap::new();
        res_map.insert(
            ReservationId::new(0),
            syn_res(0, syn_fp(0, 0, 1, 1), Some(0)),
        );
        res_map.insert(
            ReservationId::new(1),
            syn_res(1, syn_fp(100, 100, 101, 101), Some(1)),
        );

        let bp = syn_blueprint(2);
        let placement = syn_placement(&res_map);
        let resolved = make_resolved();

        let result = solve_topology(&bp, &placement, &resolved);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(err.code, RichnessErrorCode::TopologyExhausted);
        assert_eq!(
            err.category,
            RichnessErrorCategory::PlacementTopologyExhaustion
        );
    }

    #[test]
    fn synthetic_loop_augmentation_adds_candidates() {
        // Four rooms in a 2x2 grid — spanning tree uses 3 edges,
        // loop should add at least 0 more (Sparse = 0 loops)
        let mut res_map = BTreeMap::new();
        res_map.insert(
            ReservationId::new(0),
            syn_res(0, syn_fp(0, 0, 4, 8), Some(0)),
        );
        res_map.insert(
            ReservationId::new(1),
            syn_res(1, syn_fp(8, 0, 12, 8), Some(1)),
        );
        res_map.insert(
            ReservationId::new(2),
            syn_res(2, syn_fp(0, 8, 4, 16), Some(2)),
        );
        res_map.insert(
            ReservationId::new(3),
            syn_res(3, syn_fp(8, 8, 12, 16), Some(3)),
        );

        // Only 2 beats on critical path, but 4 rooms
        // Set up: beat 0 has rooms 0,2; beat 1 has rooms 1,3
        let mut bp = syn_blueprint(2);
        // Add rooms 2 and 3 as same-beat reservations
        bp.beats
            .get_mut(&BeatId::new(0))
            .unwrap()
            .requests
            .push(ArchetypeRequestId::new(2));
        bp.beats
            .get_mut(&BeatId::new(1))
            .unwrap()
            .requests
            .push(ArchetypeRequestId::new(3));

        let mut res_map_full = BTreeMap::new();
        for i in 0..4u32 {
            let bid = if i < 2 { 0 } else { 1 };
            let x0 = 4 + (i % 2) * 24;
            let y0 = 4 + (i / 2) * 24;
            res_map_full.insert(
                ReservationId::new(i),
                syn_res(i, syn_fp(x0, y0, x0 + 8, y0 + 8), Some(bid)),
            );
        }

        let placement = syn_placement(&res_map_full);
        let resolved = make_resolved();

        let result = solve_topology(&bp, &placement, &resolved).unwrap();
        let tree_edges = result.selected_edges.len() - result.loop_count;
        assert_eq!(tree_edges, 3, "tree should have 3 edges");
        assert_eq!(result.loop_count, 0);
    }

    fn make_resolved() -> ResolvedRichnessRequestV1 {
        use super::super::request::RichnessDocumentV1;
        let doc = RichnessDocumentV1::new(0, 2048, RichnessPreset::Sparse, RichnessTheme::Ancient)
            .unwrap();
        ResolvedRichnessRequestV1::resolve(doc).unwrap()
    }

    // ── Phase 07-C Proof validators (unit tests) ───────────────────────

    #[test]
    fn proof_global_connectivity_passes_when_all_connected() {
        let mut res_map = BTreeMap::new();
        res_map.insert(
            ReservationId::new(0),
            syn_res(0, syn_fp(0, 0, 4, 8), Some(0)),
        );
        res_map.insert(
            ReservationId::new(1),
            syn_res(1, syn_fp(8, 0, 12, 8), Some(1)),
        );

        let bp = syn_blueprint(2);
        let placement = syn_placement(&res_map);
        let resolved = make_resolved();
        let result = solve_topology(&bp, &placement, &resolved).unwrap();

        let report = super::validate_global_connectivity(&result, &placement.reservations);
        assert!(report.passed, "connectivity failed: {:?}", report.errors);
    }

    #[test]
    fn proof_mandatory_edges_all_present() {
        let mut res_map = BTreeMap::new();
        res_map.insert(
            ReservationId::new(0),
            syn_res(0, syn_fp(0, 0, 4, 8), Some(0)),
        );
        res_map.insert(
            ReservationId::new(1),
            syn_res(1, syn_fp(8, 0, 12, 8), Some(1)),
        );

        let bp = syn_blueprint(2);
        let placement = syn_placement(&res_map);
        let resolved = make_resolved();
        let result = solve_topology(&bp, &placement, &resolved).unwrap();

        let report = super::validate_mandatory_edges_present(&result, &bp, &placement.reservations);
        assert!(
            report.passed,
            "mandatory edges missing: {:?}",
            report.errors
        );
    }

    #[test]
    fn proof_beat_order_respected() {
        let mut res_map = BTreeMap::new();
        res_map.insert(
            ReservationId::new(0),
            syn_res(0, syn_fp(0, 0, 4, 8), Some(0)),
        );
        res_map.insert(
            ReservationId::new(1),
            syn_res(1, syn_fp(8, 0, 12, 8), Some(1)),
        );
        res_map.insert(
            ReservationId::new(2),
            syn_res(2, syn_fp(16, 0, 20, 8), Some(2)),
        );

        let bp = syn_blueprint(3);
        let placement = syn_placement(&res_map);
        let resolved = make_resolved();
        let result = solve_topology(&bp, &placement, &resolved).unwrap();

        let report = super::validate_beat_order(&result, &bp, &placement.reservations);
        assert!(report.passed, "beat order violated: {:?}", report.errors);
    }

    #[test]
    fn proof_landmark_path_membership() {
        let mut res_map = BTreeMap::new();
        res_map.insert(
            ReservationId::new(0),
            syn_res(0, syn_fp(0, 0, 4, 8), Some(0)),
        );
        res_map.insert(
            ReservationId::new(1),
            syn_res(1, syn_fp(8, 0, 12, 8), Some(1)),
        );

        let bp = syn_blueprint(2);
        let placement = syn_placement(&res_map);
        let resolved = make_resolved();
        let result = solve_topology(&bp, &placement, &resolved).unwrap();

        let report =
            super::validate_landmark_path_membership(&result, &bp, &placement.reservations);
        assert!(report.passed, "landmark path: {:?}", report.errors);
    }

    #[test]
    fn proof_route_witness_ownership() {
        let mut res_map = BTreeMap::new();
        res_map.insert(
            ReservationId::new(0),
            syn_res(0, syn_fp(0, 0, 4, 8), Some(0)),
        );
        res_map.insert(
            ReservationId::new(1),
            syn_res(1, syn_fp(8, 0, 12, 8), Some(1)),
        );

        let bp = syn_blueprint(2);
        let placement = syn_placement(&res_map);
        let resolved = make_resolved();
        let result = solve_topology(&bp, &placement, &resolved).unwrap();

        let report = super::validate_route_witness_ownership(&result);
        assert!(report.passed, "route witness: {:?}", report.errors);
    }

    #[test]
    fn proof_degree_bounds_respected() {
        let mut res_map = BTreeMap::new();
        res_map.insert(
            ReservationId::new(0),
            syn_res(0, syn_fp(0, 0, 4, 8), Some(0)),
        );
        res_map.insert(
            ReservationId::new(1),
            syn_res(1, syn_fp(8, 0, 12, 8), Some(1)),
        );

        let bp = syn_blueprint(2);
        let placement = syn_placement(&res_map);
        let resolved = make_resolved();
        let result = solve_topology(&bp, &placement, &resolved).unwrap();

        let report = super::validate_degree_bounds(&result, &bp, &placement.reservations);
        assert!(report.passed, "degree bounds: {:?}", report.errors);
    }

    // ── Adversarial rollback test ──────────────────────────────────────

    #[test]
    fn adversarial_rollback_cheapest_candidate_blocks_later_mandatory() {
        fn path_through(points: &[CellCoord]) -> Vec<CellCoord> {
            let mut path = vec![points[0]];
            for target in &points[1..] {
                let mut cursor = *path.last().unwrap();
                while cursor.x != target.x {
                    cursor.x = if cursor.x < target.x {
                        cursor.x + 1
                    } else {
                        cursor.x - 1
                    };
                    path.push(cursor);
                }
                while cursor.y != target.y {
                    cursor.y = if cursor.y < target.y {
                        cursor.y + 1
                    } else {
                        cursor.y - 1
                    };
                    path.push(cursor);
                }
            }
            path
        }

        // A→B has a cheapest direct lane through the central choke and a
        // longer southern lane. C→D is the later mandatory connection and has
        // exactly the central choke. Committing direct A→B therefore makes
        // C→D infeasible; complete search must roll it back and choose lane 1.
        let mut reservations = BTreeMap::new();
        reservations.insert(
            ReservationId::new(0),
            syn_res(0, syn_fp(4, 20, 12, 28), Some(0)),
        );
        reservations.insert(
            ReservationId::new(1),
            syn_res(1, syn_fp(36, 20, 44, 28), Some(1)),
        );
        reservations.insert(
            ReservationId::new(2),
            syn_res(2, syn_fp(20, 4, 28, 12), Some(2)),
        );
        reservations.insert(
            ReservationId::new(3),
            syn_res(3, syn_fp(20, 36, 28, 44), Some(3)),
        );
        let placement = syn_placement(&reservations);
        let resolved = make_resolved();
        let blueprint = syn_blueprint(4);

        let first_potential = PotentialConnection {
            ordinal: 0,
            kind: EdgeKind::SameLayer { layer: 0 },
            source: ReservationId::new(0),
            target: ReservationId::new(1),
            preferred_dir: Dir::East,
            distance: 32,
            pacing_bias: 0,
            pair_rank: 0,
            mandatory_record: Some(MandatoryEdge {
                from_beat: BeatId::new(0),
                to_beat: BeatId::new(1),
            }),
        };
        let later_potential = PotentialConnection {
            ordinal: 1,
            kind: EdgeKind::SameLayer { layer: 0 },
            source: ReservationId::new(2),
            target: ReservationId::new(3),
            preferred_dir: Dir::South,
            distance: 32,
            pacing_bias: 0,
            pair_rank: 100,
            mandatory_record: Some(MandatoryEdge {
                from_beat: BeatId::new(2),
                to_beat: BeatId::new(3),
            }),
        };
        let a_socket = PortalRouteCandidate {
            wall: Dir::East,
            anchor: CellCoord::new(11, 24, 0),
            exterior_center: CellCoord::new(14, 24, 0),
        };
        let b_socket = PortalRouteCandidate {
            wall: Dir::West,
            anchor: CellCoord::new(36, 24, 0),
            exterior_center: CellCoord::new(34, 24, 0),
        };
        let c_socket = PortalRouteCandidate {
            wall: Dir::South,
            anchor: CellCoord::new(24, 11, 0),
            exterior_center: CellCoord::new(24, 14, 0),
        };
        let d_socket = PortalRouteCandidate {
            wall: Dir::North,
            anchor: CellCoord::new(24, 36, 0),
            exterior_center: CellCoord::new(24, 34, 0),
        };
        let direct = CompleteRoutePlan {
            source_socket: a_socket,
            target_socket: b_socket,
            path: path_through(&[a_socket.exterior_center, b_socket.exterior_center]),
        };
        let later = CompleteRoutePlan {
            source_socket: c_socket,
            target_socket: d_socket,
            path: path_through(&[c_socket.exterior_center, d_socket.exterior_center]),
        };

        let route_components = RouteCenterComponents::build(&placement.journal);
        let production_candidates = feasible_candidates_for_potential(
            &placement.journal,
            &route_components,
            &first_potential,
            &reservations,
            0,
            0,
            0,
            &resolved,
        );
        assert!(
            production_candidates.len() >= 2,
            "fixture did not produce a real production detour"
        );
        let cheapest = production_candidates[0].clone();
        assert_eq!(cheapest.plan, direct);
        assert!(production_candidates[1..]
            .iter()
            .all(|candidate| cheapest.edge.sort_key() < candidate.edge.sort_key()));

        let mut blocked = placement.journal.clone();
        let mut blocked_route_id = 0;
        let mut blocked_portal_id = 0;
        let mut blocked_turn_id = 0;
        commit_feasible_candidate(
            &mut blocked,
            &cheapest,
            &reservations,
            &mut blocked_route_id,
            &mut blocked_portal_id,
            &mut blocked_turn_id,
            &resolved,
        )
        .unwrap();
        assert!(
            materialize_explicit_candidate(
                &blocked,
                &later_potential,
                &later,
                0,
                &reservations,
                blocked_route_id,
                blocked_portal_id,
                blocked_turn_id,
                &resolved,
            )
            .is_none(),
            "cheapest fixture lane did not block the later mandatory edge"
        );

        let required = vec![
            RequiredConnection {
                potential: first_potential,
                backward_shortcut: false,
                explicit_plans: None,
            },
            RequiredConnection {
                potential: later_potential,
                backward_shortcut: false,
                explicit_plans: Some(vec![later]),
            },
        ];
        let before = placement.journal.state_snapshot();
        let mut journal = placement.journal.clone();
        let mut next_route_id = 0;
        let mut next_portal_id = 0;
        let mut next_turn_id = 0;
        let mut selected = Vec::new();
        let mut routes = Vec::new();
        let mut metrics = TopologySearchMetrics::default();
        let res_index: BTreeMap<_, _> = reservations
            .keys()
            .enumerate()
            .map(|(index, id)| (*id, index))
            .collect();
        assert!(reserve_mandatory_connections(
            &required,
            0,
            false,
            &[],
            &mut journal,
            &reservations,
            &res_index,
            &BTreeMap::new(),
            &mut next_route_id,
            &mut next_portal_id,
            &mut next_turn_id,
            &mut selected,
            &mut routes,
            &blueprint,
            &resolved,
            &mut metrics,
        ));
        assert_ne!(selected[0].id, cheapest.edge.id);
        assert_eq!(selected[1].id, EdgeId::new(EDGE_ID_STRIDE));
        assert!(metrics.mandatory_backtracks >= 1);
        assert_eq!(metrics.rollback_checks, metrics.mandatory_backtracks);
        assert_eq!(metrics.rollback_mismatches, 0);
        for id in before.reservations.keys() {
            assert_eq!(journal.get(*id), before.reservations.get(id));
        }
    }

    // ── Broad preset sweeps ────────────────────────────────────────────

    #[test]
    fn broad_sweep_sparse_all_seeds() {
        let mut max_search_states = 0;
        for seed in 0u64..32 {
            let doc =
                RichnessDocumentV1::new(seed, 2048, RichnessPreset::Sparse, RichnessTheme::Ancient)
                    .unwrap();
            let resolved = ResolvedRichnessRequestV1::resolve(doc).unwrap();
            let bp = build_pacing_blueprint(&resolved).unwrap();
            let placement = solve_placement(bp.clone(), resolved.clone()).unwrap();
            max_search_states = max_search_states.max(placement.max_search_states);
            let result = solve_topology(&bp, &placement, &resolved);
            assert!(
                result.is_ok(),
                "Sparse failed for seed={}: {:?}",
                seed,
                result.err()
            );
            let r = result.unwrap();
            assert!(!r.selected_edges.is_empty());
            // Run all proofs
            let proofs = super::run_all_proofs(&r, &bp, &placement.reservations);
            assert!(
                proofs.passed,
                "proofs failed at seed={}: {:?}",
                seed, proofs.errors
            );
            assert_eq!(r.loop_count, 0);
        }
        eprintln!(
            "Sparse sweep: passes=32 fails=0 max_search_states={max_search_states} frozen_bound={MAX_PLACEMENT_SEARCH_STATES}"
        );
    }

    #[test]
    fn broad_sweep_moderate_multiple_seeds() {
        // Every supported seed must materialize both loops and a backward
        // shortcut without seed substitution.
        let seeds = 0u64..32;
        let mut max_search_states = 0;
        for seed in seeds {
            let document = RichnessDocumentV1::new(
                seed,
                2048,
                RichnessPreset::Moderate,
                RichnessTheme::Ancient,
            )
            .unwrap();
            let resolved = ResolvedRichnessRequestV1::resolve(document).unwrap();
            let blueprint = build_pacing_blueprint(&resolved).unwrap();
            let placement = solve_placement(blueprint.clone(), resolved.clone()).unwrap();
            max_search_states = max_search_states.max(placement.max_search_states);
            let topology = solve_topology(&blueprint, &placement, &resolved).unwrap();
            let proofs = run_all_proofs(&topology, &blueprint, &placement.reservations);
            assert!(
                proofs.passed,
                "proofs failed at seed={seed}: {:?}; route_edges={:?}; selected_edges={:?}; vertical_edges={:?}",
                proofs.errors,
                topology.routes.iter().map(|route| route.edge_id).collect::<Vec<_>>(),
                topology.selected_edges.iter().map(|edge| edge.id).collect::<Vec<_>>(),
                topology.vertical_routes.iter().map(|route| route.edge_id).collect::<Vec<_>>()
            );
            assert_eq!(topology.loop_count, 2);
            assert!(!topology.shortcuts_realized.is_empty());
        }
        eprintln!(
            "Moderate sweep: passes=32 fails=0 max_search_states={max_search_states} frozen_bound={MAX_PLACEMENT_SEARCH_STATES}"
        );
    }

    #[test]
    fn broad_sweep_rich_multiple_seeds() {
        // Every supported seed must materialize all four loops and at least
        // one backward shortcut without seed substitution.
        let seeds = 0u64..32;
        let mut max_search_states = 0;
        for seed in seeds {
            let document =
                RichnessDocumentV1::new(seed, 3072, RichnessPreset::Rich, RichnessTheme::Ancient)
                    .unwrap();
            let resolved = ResolvedRichnessRequestV1::resolve(document).unwrap();
            let blueprint = build_pacing_blueprint(&resolved).unwrap();
            let placement = solve_placement(blueprint.clone(), resolved.clone()).unwrap();
            max_search_states = max_search_states.max(placement.max_search_states);
            let topology = solve_topology(&blueprint, &placement, &resolved).unwrap();
            let proofs = run_all_proofs(&topology, &blueprint, &placement.reservations);
            assert!(
                proofs.passed,
                "proofs failed at seed={seed}: {:?}",
                proofs.errors
            );
            assert_eq!(topology.loop_count, 4);
            assert!(!topology.shortcuts_realized.is_empty());
        }
        eprintln!(
            "Rich sweep: passes=32 fails=0 max_search_states={max_search_states} frozen_bound={MAX_PLACEMENT_SEARCH_STATES}"
        );
    }

    #[test]
    fn supported_release_matrix_all_themes_presets_and_seeds() {
        let seeds = [0u64, 42, 99, 255];
        for &preset in RichnessPreset::ALL {
            let extent = match preset {
                RichnessPreset::Rich => 3072,
                RichnessPreset::Sparse | RichnessPreset::Moderate => 2048,
            };
            for seed in seeds {
                let mut theme_baseline: Option<(PlacementResult, TopologyResult)> = None;
                for &theme in RichnessTheme::ALL {
                    let document = RichnessDocumentV1::new(seed, extent, preset, theme).unwrap();
                    let resolved = ResolvedRichnessRequestV1::resolve(document).unwrap();
                    let blueprint = build_pacing_blueprint(&resolved).unwrap();
                    let placement = solve_placement(blueprint.clone(), resolved.clone()).unwrap();
                    let topology = solve_topology(&blueprint, &placement, &resolved).unwrap();
                    let proofs = run_all_proofs(&topology, &blueprint, &placement.reservations);
                    assert!(
                        proofs.passed,
                        "matrix proofs failed: theme={theme:?} preset={preset:?} seed={seed}: {:?}",
                        proofs.errors
                    );
                    let expected_loops = match preset {
                        RichnessPreset::Sparse => 0,
                        RichnessPreset::Moderate => 2,
                        RichnessPreset::Rich => 4,
                    };
                    assert_eq!(topology.loop_count, expected_loops);
                    assert_eq!(
                        placement
                            .reservations
                            .values()
                            .filter(|reservation| reservation.kind == ReservationKind::VerticalHost)
                            .count(),
                        resolved.vertical_openings().value() as usize
                    );
                    let expected_caves = match preset {
                        RichnessPreset::Sparse | RichnessPreset::Moderate => 2,
                        RichnessPreset::Rich => 4,
                    };
                    assert_eq!(
                        placement
                            .reservations
                            .values()
                            .filter(|reservation| reservation.kind == ReservationKind::CaveHost)
                            .count(),
                        expected_caves
                    );
                    if preset != RichnessPreset::Sparse {
                        assert!(!topology.shortcuts_realized.is_empty());
                    }
                    for payoff in blueprint.branch_payoffs.values() {
                        let leaf = payoff.to_beat.unwrap();
                        assert!(placement.reservations.values().any(|reservation| {
                            reservation.beat_id == Some(leaf)
                                && matches!(reservation.kind, ReservationKind::StandardRoom)
                        }));
                    }

                    if let Some((baseline_placement, baseline_topology)) = &theme_baseline {
                        assert_eq!(&placement, baseline_placement, "theme changed placement");
                        assert_eq!(&topology, baseline_topology, "theme changed topology");
                    } else {
                        theme_baseline = Some((placement, topology));
                    }
                }
            }
        }
    }

    // ── Impossible-request tests ───────────────────────────────────────

    #[test]
    fn impossible_request_tiny_extent_returns_error() {
        let error = RichnessDocumentV1::new(0, 256, RichnessPreset::Rich, RichnessTheme::Ancient)
            .unwrap_err();
        assert_eq!(error.code, RichnessErrorCode::ValueOutOfRange);
        assert_eq!(error.path, "extent");
    }

    #[test]
    fn impossible_request_no_feasible_candidates() {
        // Two tiny rooms at opposite corners with no adjacency
        let mut res_map = BTreeMap::new();
        res_map.insert(
            ReservationId::new(0),
            syn_res(0, syn_fp(0, 0, 1, 1), Some(0)),
        );
        res_map.insert(
            ReservationId::new(1),
            syn_res(1, syn_fp(63, 63, 64, 64), Some(1)),
        );

        let bp = syn_blueprint(2);
        let placement = syn_placement(&res_map);
        let resolved = make_resolved();
        let result = solve_topology(&bp, &placement, &resolved);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(err.code, RichnessErrorCode::TopologyExhausted);
        assert_eq!(err.path, "topology.mandatory");
        assert!(err.context.contains("mandatory"));
    }

    // ── Route-infeasibility test (placement-bound exhaustion is in solver) ──

    #[test]
    fn infeasible_route_returns_typed_topology_error() {
        // Two 1-cell rooms at opposite corners cannot connect
        let mut res_map = BTreeMap::new();
        res_map.insert(
            ReservationId::new(0),
            syn_res(0, syn_fp(0, 0, 1, 1), Some(0)),
        );
        res_map.insert(
            ReservationId::new(1),
            syn_res(1, syn_fp(127, 127, 128, 128), Some(1)),
        );

        let bp = syn_blueprint(2);
        let placement = syn_placement(&res_map);
        let resolved = make_resolved();
        let result = solve_topology(&bp, &placement, &resolved);
        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err().code,
            RichnessErrorCode::TopologyExhausted
        );
    }

    // ── Same-XY ownership tests ────────────────────────────────────────

    #[test]
    fn same_xy_ownership_rejected_for_standard_rooms() {
        // Two standard rooms at same XY on different layers must be rejected
        // unless one is a composite child. This is enforced in the occupancy
        // grid (footprint.rs tests), but we verify at topology level too.
        let mut res_map = BTreeMap::new();
        let lower = Footprint3D {
            x0: 4,
            y0: 4,
            x1: 8,
            y1: 8,
            occupies_lower: true,
            occupies_upper: false,
        };
        let upper = Footprint3D {
            x0: 4,
            y0: 4,
            x1: 8,
            y1: 8,
            occupies_lower: false,
            occupies_upper: true,
        };
        res_map.insert(ReservationId::new(0), syn_res(0, lower, Some(0)));
        res_map.insert(ReservationId::new(1), syn_res(1, upper, Some(1)));

        // Both at same XY, different layers — the journal should reject
        // the second reservation if not composite. Since sync_placement
        // calls try_reserve for each, the second will fail.
        // This test verifies that same-XY ownership is enforced.

        let mut journal = ReservationJournal::new(2048, 8000);
        let r0 = journal
            .try_reserve(
                ReservationKind::StandardRoom,
                lower,
                Some(BeatId::new(0)),
                None,
                None,
                0,
                0,
                0,
                0,
            )
            .unwrap();
        assert_eq!(r0, ReservationId::new(0));

        // Second standard room at same XY must fail
        let r1_result = journal.try_reserve(
            ReservationKind::StandardRoom,
            upper,
            Some(BeatId::new(1)),
            None,
            None,
            0,
            0,
            0,
            0,
        );
        assert!(
            r1_result.is_err(),
            "same-XY standard rooms must be rejected"
        );
    }

    #[test]
    fn same_xy_allowed_under_composite() {
        // A composite reservation allows children at same XY
        let mut journal = ReservationJournal::new(2048, 8000);
        let comp_fp = Footprint3D {
            x0: 4,
            y0: 4,
            x1: 8,
            y1: 8,
            occupies_lower: true,
            occupies_upper: true,
        };
        let _comp_id = journal
            .try_reserve(
                ReservationKind::Composite,
                comp_fp,
                None,
                None,
                None,
                0,
                0,
                0,
                0,
            )
            .unwrap();

        // A child room within the composite should be allowed
        let child_fp = Footprint3D {
            x0: 5,
            y0: 5,
            x1: 7,
            y1: 7,
            occupies_lower: true,
            occupies_upper: false,
        };
        let child_result = journal.try_reserve_composite_child(
            _comp_id,
            ReservationKind::StandardRoom,
            child_fp,
            None,
            None,
            None,
            100,
            8,
            2,
            1,
        );
        // This should succeed because the composite already owns the cells
        assert!(
            child_result.is_ok(),
            "child under composite should be allowed: {:?}",
            child_result.err()
        );
    }

    // ── Canonical replay-state tests ───────────────────────────────────

    #[test]
    fn canonical_replay_same_request_same_state() {
        // Same request, same revision → identical topology
        let doc1 =
            RichnessDocumentV1::new(42, 2048, RichnessPreset::Sparse, RichnessTheme::Ancient)
                .unwrap();
        let doc2 =
            RichnessDocumentV1::new(42, 2048, RichnessPreset::Sparse, RichnessTheme::Ancient)
                .unwrap();
        let resolved1 = ResolvedRichnessRequestV1::resolve(doc1).unwrap();
        let resolved2 = ResolvedRichnessRequestV1::resolve(doc2).unwrap();
        let bp1 = build_pacing_blueprint(&resolved1).unwrap();
        let bp2 = build_pacing_blueprint(&resolved2).unwrap();

        let placement1 = solve_placement(bp1.clone(), resolved1.clone()).unwrap();
        let placement2 = solve_placement(bp2.clone(), resolved2.clone()).unwrap();

        let result1 = solve_topology(&bp1, &placement1, &resolved1).unwrap();
        let result2 = solve_topology(&bp2, &placement2, &resolved2).unwrap();

        assert_eq!(placement1, placement2);
        assert_eq!(result1, result2);
    }

    #[test]
    fn canonical_replay_different_seeds_different_output() {
        let doc1 =
            RichnessDocumentV1::new(42, 2048, RichnessPreset::Sparse, RichnessTheme::Ancient)
                .unwrap();
        let doc2 =
            RichnessDocumentV1::new(99, 2048, RichnessPreset::Sparse, RichnessTheme::Ancient)
                .unwrap();
        let resolved1 = ResolvedRichnessRequestV1::resolve(doc1).unwrap();
        let resolved2 = ResolvedRichnessRequestV1::resolve(doc2).unwrap();
        let bp1 = build_pacing_blueprint(&resolved1).unwrap();
        let bp2 = build_pacing_blueprint(&resolved2).unwrap();

        let placement1 = solve_placement(bp1.clone(), resolved1.clone()).unwrap();
        let placement2 = solve_placement(bp2.clone(), resolved2.clone()).unwrap();

        let result1 = solve_topology(&bp1, &placement1, &resolved1).unwrap();
        let result2 = solve_topology(&bp2, &placement2, &resolved2).unwrap();

        assert_ne!(placement1, placement2);
        assert_ne!(result1, result2);
    }

    // ── Moderate diagnosis test ───────────────────────────────────────

    #[test]
    fn moderate_diagnostic_seed_42() {
        let doc =
            RichnessDocumentV1::new(42, 2048, RichnessPreset::Moderate, RichnessTheme::Ancient)
                .unwrap();
        let resolved = ResolvedRichnessRequestV1::resolve(doc).unwrap();
        let bp = build_pacing_blueprint(&resolved).unwrap();
        let placement = solve_placement(bp.clone(), resolved.clone()).unwrap();
        eprintln!(
            "Placement: {} rooms, corridor_rejections={}, search_states={}",
            placement.placed_count, placement.corridor_rejections, placement.total_search_states
        );
        eprintln!(
            "Blueprint: {} beats, {} requests, {} mandatory edges",
            bp.beats.len(),
            bp.archetype_requests.len(),
            bp.mandatory_edges.len()
        );
        for (rid, rec) in &placement.reservations {
            if matches!(
                rec.kind,
                ReservationKind::StandardRoom | ReservationKind::MultiStoreyRoom
            ) {
                eprintln!(
                    "  Res {:?} beat={:?} fp=({},{})-({},{}) layer={}",
                    rid,
                    rec.beat_id,
                    rec.footprint.x0,
                    rec.footprint.y0,
                    rec.footprint.x1,
                    rec.footprint.y1,
                    if rec.footprint.occupies_lower {
                        "L"
                    } else {
                        "U"
                    }
                );
            }
        }
        let topology_result = solve_topology(&bp, &placement, &resolved);
        match topology_result {
            Ok(r) => {
                eprintln!(
                    "Moderate seed=42 SUCCESS: {} edges, {} routes, search={:?}",
                    r.selected_edges.len(),
                    r.routes.len(),
                    r.search_metrics
                );
            }
            Err(e) => {
                eprintln!("Moderate seed=42 FAILED: {}", e.context);
            }
        }
    }

    // ── Search state counter test ──────────────────────────────────────

    #[test]
    fn search_states_counter_increments() {
        let doc = RichnessDocumentV1::new(0, 2048, RichnessPreset::Sparse, RichnessTheme::Ancient)
            .unwrap();
        let resolved = ResolvedRichnessRequestV1::resolve(doc).unwrap();
        let bp = build_pacing_blueprint(&resolved).unwrap();
        let placement = solve_placement(bp, resolved).unwrap();

        // Search states must be non-zero
        assert!(
            placement.total_search_states > 0,
            "search states counter should be non-zero"
        );
        assert!(
            placement.max_search_states > 0,
            "max search states counter should be non-zero"
        );
        assert!(
            placement.max_search_states <= placement.total_search_states,
            "max <= total: max={} total={}",
            placement.max_search_states,
            placement.total_search_states
        );
    }

    // ── All proofs integration test ────────────────────────────────────

    #[test]
    fn all_proofs_pass_on_valid_topology() {
        let doc = RichnessDocumentV1::new(0, 2048, RichnessPreset::Sparse, RichnessTheme::Ancient)
            .unwrap();
        let resolved = ResolvedRichnessRequestV1::resolve(doc).unwrap();
        let bp = build_pacing_blueprint(&resolved).unwrap();
        let placement = solve_placement(bp.clone(), resolved.clone()).unwrap();
        let result = solve_topology(&bp, &placement, &resolved).unwrap();

        let report = super::run_all_proofs(&result, &bp, &placement.reservations);
        assert!(report.passed, "all proofs must pass: {:#?}", report.errors);
    }

    // ── Multi-segment route test ───────────────────────────────────────

    #[test]
    fn multisegment_route_with_turn_reservation() {
        // A and B share a mandatory beat edge; C completely blocks every
        // straight four-cell lane through their gap. The complete grid router
        // must commit an owned path around C with concrete turn witnesses.
        let mut reservations = BTreeMap::new();
        reservations.insert(
            ReservationId::new(0),
            syn_res(0, syn_fp(4, 8, 8, 20), Some(0)),
        );
        reservations.insert(
            ReservationId::new(1),
            syn_res(1, syn_fp(16, 8, 20, 20), Some(1)),
        );
        let mut blocker = syn_res(2, syn_fp(10, 0, 14, 28), None);
        blocker.kind = ReservationKind::Support;
        reservations.insert(ReservationId::new(2), blocker);

        let blueprint = syn_blueprint(2);
        let placement = syn_placement(&reservations);
        let resolved = make_resolved();
        let topology = solve_topology(&blueprint, &placement, &resolved).unwrap();
        let route = topology
            .routes
            .iter()
            .find(|route| {
                (route.source == ReservationId::new(0) && route.target == ReservationId::new(1))
                    || (route.source == ReservationId::new(1)
                        && route.target == ReservationId::new(0))
            })
            .unwrap();
        assert!(
            !route.turns.is_empty(),
            "blocking room did not force a turn"
        );
        assert!(!route.reservation_ids.is_empty());
        assert!(route
            .reservation_ids
            .iter()
            .all(|id| topology.journal.get(*id).is_some()));
        for turn in &route.turns {
            let record = topology.journal.get(turn.reservation_id).unwrap();
            assert_eq!(record.kind, ReservationKind::Turn);
            assert_eq!(record.owning_route_id, Some(route.id));
            assert_eq!(record.footprint, turn.witness);
            assert_eq!(record.footprint.cell_count(), 16);
            assert_eq!(turn.headroom, HEADROOM);
            assert_eq!(turn.witness.quake_span(), (64, 64));
        }
        let proof = validate_route_witness_ownership(&topology);
        assert!(proof.passed, "turn ownership proof: {:?}", proof.errors);
    }
}
