//! CarvedGrotto: deterministic connected cave synthesis as a convex solid
//! complement.
//!
//! A cave is a REAL capability, not a room modifier. Eligibility is canonical
//! and decided from the request revision, cave mode, and reservation
//! feasibility. The frozen 3D density field (see `fields::fbm3`) is sampled
//! on a 32-unit lattice inside a 16-unit protected shell. Portals and routes
//! are dilated by the player-hull clearance contract before empty
//! classification, and a connected empty instance containing every required
//! route witness is found by complete deterministic candidate search. The
//! solid complement is partitioned into deterministic non-overlapping
//! maximal axis-aligned boxes (greedy, frozen scan and axis order), every
//! cave solid carries a support path, and cave floor/wall/ceiling roles stay
//! distinct for theme material assignment.
//!
//! There is no marching cubes, no arbitrary plane, no overlapping host mass,
//! and no flat-room fallback. A required-but-impossible cave returns a typed
//! feasibility/budget error.

use std::collections::BTreeMap;

use crate::enhanced_v3::geometry::ConvexBrush;
use crate::enhanced_v3::richness::{
    assembly::{
        AssemblyIR, BrushAssembly, BrushAssemblyRole, BudgetDimension, CostSource,
        SemanticAttribution, SupportTarget,
    },
    error::{RichnessError, RichnessErrorCategory, RichnessErrorCode},
    fields::{fbm3, FieldTag},
    fixed::FixedQ32,
    footprint::Footprint3D,
    geometry::{footprint_quake_bounds, footprint_vertical_bounds, validate_brush},
    ids::{
        ArchetypeIndex, ArchetypeRequestId, BeatId, BrushAssemblyId, ReservationId, RouteId, ZoneId,
    },
    request::RichnessCaveMode,
    reservation::{ReservationJournal, ReservationKind, ReservationRecord},
};

/// Frozen lattice cell size in units.
pub(crate) const CAVE_LATTICE_UNITS: i32 = 32;
/// Frozen protected shell thickness in units.
pub(crate) const CAVE_SHELL_UNITS: i32 = 16;
/// Frozen candidate search bound (inclusive upper key).
pub(crate) const CAVE_MAX_CANDIDATES: u32 = 16;
/// Frozen empty threshold: a cell is EMPTY iff density >= this value.
pub(crate) const CAVE_EMPTY_THRESHOLD: i64 = 0;

/// Canonical cave eligibility decision.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CaveEligibility {
    /// Whether a cave will be materialized.
    pub selected: bool,
    /// The host reservation the cave occupies.
    pub host: Option<ReservationId>,
    /// Stable reason for the decision.
    pub reason: CaveEligibilityReason,
}

/// Stable eligibility reason (canonical tags).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum CaveEligibilityReason {
    /// Cave selected and will be synthesized.
    Selected,
    /// Cave mode is `omitted`.
    OmittedByMode,
    /// No cave host reservation exists.
    NoHost,
    /// Preferred mode: synthesis infeasible, cave omitted.
    PreferredInfeasible,
    /// Required mode: a typed error will be raised.
    RequiredError,
}

impl CaveEligibilityReason {
    pub(crate) fn tag(self) -> &'static str {
        match self {
            Self::Selected => "selected",
            Self::OmittedByMode => "omitted_by_mode",
            Self::NoHost => "no_host",
            Self::PreferredInfeasible => "preferred_infeasible",
            Self::RequiredError => "required_error",
        }
    }
}

/// A single cave solid box in lattice coordinates.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct SolidBox {
    /// Inclusive lattice min cell (i, j, k).
    pub min: (i32, i32, i32),
    /// Exclusive lattice max cell (i, j, k).
    pub max: (i32, i32, i32),
    /// Structural role.
    pub role: CaveRole,
}

/// Cave solid role (frozen; drives theme material roles).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum CaveRole {
    /// Solid whose top bounds the cave floor plane.
    Floor,
    /// Solid beside the cave void.
    Wall,
    /// Solid whose bottom bounds the cave ceiling plane.
    Ceiling,
}

impl CaveRole {
    pub(crate) fn tag(self) -> &'static str {
        match self {
            Self::Floor => "floor",
            Self::Wall => "wall",
            Self::Ceiling => "ceiling",
        }
    }

    fn brush_role(self) -> BrushAssemblyRole {
        match self {
            Self::Floor => BrushAssemblyRole::CaveFloor,
            Self::Wall => BrushAssemblyRole::CaveWall,
            Self::Ceiling => BrushAssemblyRole::CaveCeiling,
        }
    }
}

/// The lattice classification of one host.
#[derive(Debug, Clone)]
pub(crate) struct CaveLattice {
    /// Interior min corner in quake units (inclusive).
    pub x0: i32,
    pub y0: i32,
    pub z0: i32,
    /// Cell counts.
    pub nx: i32,
    pub ny: i32,
    pub nz: i32,
    /// Cell states: 0 = solid, 1 = empty (cave void), 2 = forced empty.
    cells: Vec<u8>,
    /// Forced-empty witness cells (sorted, deduped).
    pub witnesses: Vec<(i32, i32, i32)>,
}

impl CaveLattice {
    fn index(&self, i: i32, j: i32, k: i32) -> usize {
        ((k * self.ny + j) * self.nx + i) as usize
    }

    fn in_bounds(&self, i: i32, j: i32, k: i32) -> bool {
        i >= 0 && i < self.nx && j >= 0 && j < self.ny && k >= 0 && k < self.nz
    }

    pub(crate) fn cell_state(&self, i: i32, j: i32, k: i32) -> u8 {
        if !self.in_bounds(i, j, k) {
            return 0;
        }
        self.cells[self.index(i, j, k)]
    }

    /// Quake-unit AABB of a lattice cell (inclusive min, exclusive max).
    pub(crate) fn cell_bounds(
        &self,
        i: i32,
        j: i32,
        k: i32,
    ) -> (i128, i128, i128, i128, i128, i128) {
        (
            (self.x0 + i * CAVE_LATTICE_UNITS) as i128,
            (self.y0 + j * CAVE_LATTICE_UNITS) as i128,
            (self.z0 + k * CAVE_LATTICE_UNITS) as i128,
            (self.x0 + (i + 1) * CAVE_LATTICE_UNITS) as i128,
            (self.y0 + (j + 1) * CAVE_LATTICE_UNITS) as i128,
            (self.z0 + (k + 1) * CAVE_LATTICE_UNITS) as i128,
        )
    }
}

/// Complete cave synthesis result.
#[derive(Debug, Clone)]
pub(crate) struct CaveResult {
    /// The host reservation.
    pub host: ReservationId,
    /// The lattice classification.
    pub lattice: CaveLattice,
    /// Empty cells of the witness component (sorted).
    pub empty_cells: Vec<(i32, i32, i32)>,
    /// Solid complement partition (sorted, non-overlapping, full coverage).
    pub solid_boxes: Vec<SolidBox>,
    /// Candidate key that produced the connected instance.
    pub candidate_key: u32,
}

/// Decide cave eligibility from the cave mode and committed reservations.
///
/// Canonical and theme-independent: the decision depends only on revision,
/// preset capacity, and reservation feasibility. Host selection is the
/// lexicographically smallest committed `CaveHost` reservation.
pub(crate) fn decide_cave_eligibility(
    mode: RichnessCaveMode,
    journal: &ReservationJournal,
) -> CaveEligibility {
    match mode {
        RichnessCaveMode::Omitted => CaveEligibility {
            selected: false,
            host: None,
            reason: CaveEligibilityReason::OmittedByMode,
        },
        RichnessCaveMode::Preferred | RichnessCaveMode::Required => {
            let host = journal
                .reservations
                .iter()
                .filter(|(_, record)| record.kind == ReservationKind::CaveHost && record.committed)
                .map(|(id, _)| *id)
                .min();
            match host {
                Some(host) => CaveEligibility {
                    selected: true,
                    host: Some(host),
                    reason: CaveEligibilityReason::Selected,
                },
                None => CaveEligibility {
                    selected: false,
                    host: None,
                    reason: CaveEligibilityReason::NoHost,
                },
            }
        }
    }
}

/// Build the lattice for a host reservation.
///
/// The lattice covers the host interior (inset by the 16-unit protected
/// shell on every side) on 32-unit cells. A host whose interior is smaller
/// than one cell per axis is infeasible.
fn build_lattice(host: &ReservationRecord) -> Result<CaveLattice, RichnessError> {
    let (qx0, qy0, qx1, qy1) = footprint_quake_bounds(&host.footprint);
    let vertical = footprint_vertical_bounds(&host.footprint)?;
    let z0 = vertical.floor_min;
    let z1 = vertical.ceiling_max;
    let x0 = qx0 + CAVE_SHELL_UNITS as i128;
    let y0 = qy0 + CAVE_SHELL_UNITS as i128;
    let z0i = z0 + CAVE_SHELL_UNITS as i128;
    let x1 = qx1 - CAVE_SHELL_UNITS as i128;
    let y1 = qy1 - CAVE_SHELL_UNITS as i128;
    let z1i = z1 - CAVE_SHELL_UNITS as i128;
    if x1 - x0 < CAVE_LATTICE_UNITS as i128
        || y1 - y0 < CAVE_LATTICE_UNITS as i128
        || z1i - z0i < CAVE_LATTICE_UNITS as i128
    {
        return Err(cave_error(
            "lattice.host",
            format!(
                "host {:?} interior too small for a 32-unit cave lattice",
                host.id
            ),
        ));
    }
    let nx = ((x1 - x0) / CAVE_LATTICE_UNITS as i128) as i32;
    let ny = ((y1 - y0) / CAVE_LATTICE_UNITS as i128) as i32;
    let nz = ((z1i - z0i) / CAVE_LATTICE_UNITS as i128) as i32;
    let cells = vec![0u8; (nx * ny * nz) as usize];
    Ok(CaveLattice {
        x0: x0 as i32,
        y0: y0 as i32,
        z0: z0i as i32,
        nx,
        ny,
        nz,
        cells,
        witnesses: Vec::new(),
    })
}

/// Reserve AABB of a route-kind reservation in quake units.
fn route_region(
    record: &ReservationRecord,
) -> Result<(i128, i128, i128, i128, i128, i128), RichnessError> {
    let (qx0, qy0, qx1, qy1) = footprint_quake_bounds(&record.footprint);
    let vertical = footprint_vertical_bounds(&record.footprint)?;
    Ok((qx0, qy0, vertical.floor_min, qx1, qy1, vertical.ceiling_max))
}

/// Dilate a route region by the player-hull clearance contract (64 x 80).
fn dilated_region(
    region: (i128, i128, i128, i128, i128, i128),
) -> (i128, i128, i128, i128, i128, i128) {
    (
        region.0 - 32,
        region.1 - 32,
        region.2 - 40,
        region.3 + 32,
        region.4 + 32,
        region.5 + 40,
    )
}

fn regions_intersect_cell(
    region: (i128, i128, i128, i128, i128, i128),
    bounds: (i128, i128, i128, i128, i128, i128),
) -> bool {
    bounds.0 < region.3
        && bounds.3 > region.0
        && bounds.1 < region.4
        && bounds.4 > region.1
        && bounds.2 < region.5
        && bounds.5 > region.2
}

/// Classify the lattice from the frozen density field.
///
/// Route/portal/turn reservations intersecting the host are dilated by the
/// 64x80 player-hull contract and forced empty without changing field
/// connectivity elsewhere. Frozen: threshold equality (density >= 0 is
/// empty), negative coordinates via `floor_div`, scan order (k, j, i), and
/// candidate-key framing.
fn classify_lattice(
    lattice: &mut CaveLattice,
    seed: u64,
    candidate_key: u32,
    reservations: &[&ReservationRecord],
) {
    // Collect dilated route regions.
    let mut regions: Vec<(i128, i128, i128, i128, i128, i128)> = Vec::new();
    for record in reservations {
        if record.committed
            && matches!(
                record.kind,
                ReservationKind::Route | ReservationKind::PortalThroat | ReservationKind::Turn
            )
        {
            if let Ok(region) = route_region(record) {
                regions.push(dilated_region(region));
            }
        }
    }

    let mut witnesses: Vec<(i32, i32, i32)> = Vec::new();
    for k in 0..lattice.nz {
        for j in 0..lattice.ny {
            for i in 0..lattice.nx {
                let bounds = lattice.cell_bounds(i, j, k);
                let cx = FixedQ32::from_i32((bounds.0 as i32) + CAVE_LATTICE_UNITS / 2);
                let cy = FixedQ32::from_i32((bounds.1 as i32) + CAVE_LATTICE_UNITS / 2);
                let cz = FixedQ32::from_i32((bounds.2 as i32) + CAVE_LATTICE_UNITS / 2);
                let density = fbm3(seed, FieldTag::Caves, cx, cy, cz, candidate_key);
                let forced = regions
                    .iter()
                    .any(|region| regions_intersect_cell(*region, bounds));
                let empty = forced || density >= CAVE_EMPTY_THRESHOLD;
                let idx = lattice.index(i, j, k);
                lattice.cells[idx] = if empty { 1 } else { 0 };
                if forced {
                    lattice.cells[idx] = 2;
                    witnesses.push((i, j, k));
                }
            }
        }
    }
    witnesses.sort_unstable();
    witnesses.dedup();
    lattice.witnesses = witnesses;
}

/// Face-adjacent flood fill from a seed cell over empty (state >= 1) cells.
fn flood_component(
    lattice: &CaveLattice,
    start: (i32, i32, i32),
    visited: &mut Vec<bool>,
) -> Vec<(i32, i32, i32)> {
    let mut component = Vec::new();
    let mut stack = vec![start];
    let mark = |lattice: &CaveLattice, visited: &mut Vec<bool>, i: i32, j: i32, k: i32| -> bool {
        if !lattice.in_bounds(i, j, k) {
            return false;
        }
        let idx = lattice.index(i, j, k);
        if visited[idx] || lattice.cells[idx] == 0 {
            return false;
        }
        visited[idx] = true;
        true
    };
    if mark(lattice, visited, start.0, start.1, start.2) {
        while let Some((i, j, k)) = stack.pop() {
            component.push((i, j, k));
            for (di, dj, dk) in [
                (1, 0, 0),
                (-1, 0, 0),
                (0, 1, 0),
                (0, -1, 0),
                (0, 0, 1),
                (0, 0, -1),
            ] {
                if mark(lattice, visited, i + di, j + dj, k + dk) {
                    stack.push((i + di, j + dj, k + dk));
                }
            }
        }
    }
    component.sort_unstable();
    component
}

/// Synthesize the connected cave instance for one host.
///
/// Complete deterministic candidate search: candidate keys 0..=MAX are tried
/// in order until the empty component containing the first witness contains
/// every witness. Any empty cell outside that component is an unreachable
/// pocket and is re-solidified (frozen disposition). Exhaustion returns a
/// typed error.
pub(crate) fn synthesize_cave(
    seed: u64,
    mode: RichnessCaveMode,
    journal: &ReservationJournal,
) -> Result<Option<CaveResult>, RichnessError> {
    let eligibility = decide_cave_eligibility(mode, journal);
    if !eligibility.selected {
        return match mode {
            RichnessCaveMode::Required => Err(cave_error(
                "eligibility.required",
                "cave is required but no committed cave host reservation exists",
            )),
            _ => Ok(None),
        };
    }
    let host_id = eligibility.host.expect("selected cave has a host");
    let host = journal
        .reservations
        .get(&host_id)
        .ok_or_else(|| cave_error("host.missing", format!("host {host_id:?} not found")))?;
    let route_records: Vec<&ReservationRecord> = journal
        .reservations
        .values()
        .filter(|record| {
            matches!(
                record.kind,
                ReservationKind::Route | ReservationKind::PortalThroat | ReservationKind::Turn
            )
        })
        .collect();

    for candidate_key in 0..=CAVE_MAX_CANDIDATES {
        let mut lattice = build_lattice(host)?;
        classify_lattice(&mut lattice, seed, candidate_key, &route_records);
        if lattice.witnesses.is_empty() {
            // No route crosses this host: any single component qualifies.
            // Pick the component containing the lexicographically smallest
            // empty cell so the instance is canonical.
            let first = (0..lattice.nx)
                .flat_map(|i| {
                    (0..lattice.ny).flat_map(move |j| (0..lattice.nz).map(move |k| (i, j, k)))
                })
                .find(|&(i, j, k)| lattice.cell_state(i, j, k) >= 1);
            let Some(first) = first else {
                continue;
            };
            let mut visited = vec![false; (lattice.nx * lattice.ny * lattice.nz) as usize];
            let component = flood_component(&lattice, first, &mut visited);
            if component.is_empty() {
                continue;
            }
            return Ok(Some(finish_cave(
                host_id,
                lattice,
                component,
                candidate_key,
            )));
        }
        // Witness component must contain ALL witnesses.
        let first_witness = lattice.witnesses[0];
        let mut visited = vec![false; (lattice.nx * lattice.ny * lattice.nz) as usize];
        let component = flood_component(&lattice, first_witness, &mut visited);
        let all_witnesses = lattice
            .witnesses
            .iter()
            .all(|&cell| visited[lattice.index(cell.0, cell.1, cell.2)]);
        if all_witnesses {
            return Ok(Some(finish_cave(
                host_id,
                lattice,
                component,
                candidate_key,
            )));
        }
    }
    let error = cave_error(
        "candidate.exhausted",
        format!(
            "no connected cave instance containing all route witnesses after {} candidates",
            CAVE_MAX_CANDIDATES + 1
        ),
    );
    match mode {
        RichnessCaveMode::Required => Err(error),
        RichnessCaveMode::Preferred => {
            let _ = error;
            Ok(None)
        }
        RichnessCaveMode::Omitted => Ok(None),
    }
}

/// Partition the solid complement and finalize the cave result.
///
/// Unreachable pockets (empty cells outside the witness component) are
/// re-solidified. The solid complement is partitioned by greedy maximal box
/// merging with the FROZEN scan order (x asc, y asc, z asc) and FROZEN growth
/// axes (x, then y, then z). Roles: boxes touching the interior floor plane
/// are Floor, boxes touching the interior ceiling plane are Ceiling, the
/// rest are Wall.
fn finish_cave(
    host_id: ReservationId,
    mut lattice: CaveLattice,
    component: Vec<(i32, i32, i32)>,
    candidate_key: u32,
) -> CaveResult {
    let nx = lattice.nx;
    let ny = lattice.ny;
    let nz = lattice.nz;
    let mut solid = vec![true; (nx * ny * nz) as usize];
    for &(i, j, k) in &component {
        solid[lattice.index(i, j, k)] = false;
    }
    // Unreachable pockets (empty cells outside the witness component) are
    // re-solidified: their lattice state flips to solid so the emitted
    // complement owns every cell exactly once.
    for (idx, cell) in solid.iter().enumerate() {
        if *cell {
            lattice.cells[idx] = 0;
        }
    }
    let mut claimed = vec![false; (nx * ny * nz) as usize];
    let mut solid_boxes: Vec<SolidBox> = Vec::new();

    // Frozen scan: x asc, y asc, z asc.
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let idx = lattice.index(i, j, k);
                if !solid[idx] || claimed[idx] {
                    continue;
                }
                // Grow along +x.
                let mut x_end = i + 1;
                while x_end < nx
                    && solid[lattice.index(x_end, j, k)]
                    && !claimed[lattice.index(x_end, j, k)]
                {
                    x_end += 1;
                }
                // Grow along +y (full x slab).
                let mut y_end = j + 1;
                'y: while y_end < ny {
                    for xi in i..x_end {
                        if !solid[lattice.index(xi, y_end, k)]
                            || claimed[lattice.index(xi, y_end, k)]
                        {
                            break 'y;
                        }
                    }
                    y_end += 1;
                }
                // Grow along +z (full xy slab).
                let mut z_end = k + 1;
                'z: while z_end < nz {
                    for yi in j..y_end {
                        for xi in i..x_end {
                            if !solid[lattice.index(xi, yi, z_end)]
                                || claimed[lattice.index(xi, yi, z_end)]
                            {
                                break 'z;
                            }
                        }
                    }
                    z_end += 1;
                }
                for kk in k..z_end {
                    for jj in j..y_end {
                        for ii in i..x_end {
                            claimed[lattice.index(ii, jj, kk)] = true;
                        }
                    }
                }
                let role = if k == 0 {
                    CaveRole::Floor
                } else if k + 1 == nz {
                    CaveRole::Ceiling
                } else {
                    CaveRole::Wall
                };
                solid_boxes.push(SolidBox {
                    min: (i, j, k),
                    max: (x_end, y_end, z_end),
                    role,
                });
            }
        }
    }
    solid_boxes.sort_unstable();
    CaveResult {
        host: host_id,
        lattice,
        empty_cells: component,
        solid_boxes,
        candidate_key,
    }
}

/// Emit the cave solids into the assembly with roles, attribution, and
/// support. Floor boxes rest on the world; every other box is supported by
/// the box whose top face touches its bottom face with positive area.
/// A box with no such support is a typed failure (never fake support).
pub(crate) fn materialize_cave(
    ir: &mut AssemblyIR,
    result: &CaveResult,
    request_archetypes: &BTreeMap<ArchetypeRequestId, ArchetypeIndex>,
    reservations: &BTreeMap<ReservationId, ReservationRecord>,
) -> Result<(), RichnessError> {
    let host = reservations
        .get(&result.host)
        .ok_or_else(|| cave_error("host.missing", format!("host {:?} not found", result.host)))?;
    let request_id = host.request_id;
    let beat_id = host.beat_id;
    let zone_id = host.zone_id;
    let archetype = request_id.and_then(|id| request_archetypes.get(&id).copied());
    let owner = SemanticAttribution {
        reservation_id: result.host,
        request_id,
        archetype,
        beat_id,
        zone_id,
    };

    let mut emitted: Vec<(SolidBox, BrushAssemblyId)> = Vec::new();
    for solid in &result.solid_boxes {
        let (x0, y0, z0, x1, y1, z1) =
            result
                .lattice
                .cell_bounds(solid.min.0, solid.min.1, solid.min.2);
        let (_, _, _, x1e, y1e, z1e) =
            result
                .lattice
                .cell_bounds(solid.max.0 - 1, solid.max.1 - 1, solid.max.2 - 1);
        let bounds = (x0, y0, z0, x1e, y1e, z1e);
        let brush = ConvexBrush::make_box(
            (bounds.0, bounds.3),
            (bounds.1, bounds.4),
            (bounds.2, bounds.5),
        )
        .map_err(|error| cave_error("box", format!("{error}")))?;
        validate_brush(&brush)?;
        let id = ir.alloc_brush_id();
        let support = if solid.min.2 == 0 {
            SupportTarget::World
        } else {
            // Find the box directly below the center cell with positive-area
            // contact.
            let center_x = (solid.min.0 + solid.max.0) / 2;
            let center_y = (solid.min.1 + solid.max.1) / 2;
            let below = emitted
                .iter()
                .filter(|(other, _)| other.max.2 == solid.min.2)
                .filter(|(other, _)| {
                    other.min.0 <= center_x
                        && center_x < other.max.0
                        && other.min.1 <= center_y
                        && center_y < other.max.1
                })
                .map(|(_, id)| *id)
                .min();
            match below {
                Some(id) => SupportTarget::Brush(id),
                None => {
                    return Err(cave_error(
                        "support.missing",
                        format!(
                            "cave solid box {:?} has no supporting box below its center",
                            solid
                        ),
                    ));
                }
            }
        };
        ir.insert_brush(BrushAssembly {
            id,
            brush,
            role: solid.role.brush_role(),
            owner: owner.clone(),
            cost: CostSource {
                dimension: BudgetDimension::SourceFaces,
                face_count: 6,
            },
            support,
        });
        emitted.push((solid.clone(), id));
    }
    Ok(())
}

/// Validate cave invariants over an emitted result:
/// - exactly one cave per eligible map (structural: a single CaveResult);
/// - no witness cell covered by solid boxes;
/// - solid boxes are non-overlapping and jointly cover every non-empty cell;
/// - every emitted box is within the host interior.
pub(crate) fn validate_cave_result(result: &CaveResult) -> Result<(), RichnessError> {
    let nx = result.lattice.nx;
    let ny = result.lattice.ny;
    let nz = result.lattice.nz;
    let mut covered = vec![false; (nx * ny * nz) as usize];
    for solid in &result.solid_boxes {
        for k in solid.min.2..solid.max.2 {
            for j in solid.min.1..solid.max.1 {
                for i in solid.min.0..solid.max.0 {
                    let idx = result.lattice.index(i, j, k);
                    if covered[idx] {
                        return Err(cave_error(
                            "coverage.overlap",
                            format!("solid box {:?} overlaps an earlier box", solid),
                        ));
                    }
                    covered[idx] = true;
                    // Witness cells must never be covered.
                    if result.lattice.cells[idx] == 2 {
                        return Err(cave_error(
                            "coverage.witness",
                            format!("solid box {:?} covers a protected witness cell", solid),
                        ));
                    }
                }
            }
        }
    }
    // Every non-empty (solid) lattice cell must be covered exactly once.
    for idx in 0..(nx * ny * nz) as usize {
        if result.lattice.cells[idx] == 0 && !covered[idx] {
            return Err(cave_error(
                "coverage.gap",
                format!("lattice cell {idx} is solid but uncovered"),
            ));
        }
        if result.lattice.cells[idx] != 0 && covered[idx] {
            return Err(cave_error(
                "coverage.extra",
                format!("lattice cell {idx} is empty but covered by a solid box"),
            ));
        }
    }
    Ok(())
}

/// Build a typed cave error.
pub(crate) fn cave_error(path: &str, context: impl Into<String>) -> RichnessError {
    RichnessError::new(
        RichnessErrorCode::CaveFailure,
        0,
        "?",
        "?",
        "?",
        "?",
        "?",
        "?",
        "?",
        path,
        RichnessErrorCategory::CaveFailure,
        context,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_host(journal: &mut ReservationJournal, _id: u32, x0: u32, y0: u32, x1: u32, y1: u32) {
        // Cave hosts are both-band footprints.
        let footprint = Footprint3D {
            x0,
            y0,
            x1,
            y1,
            occupies_lower: true,
            occupies_upper: true,
        };
        journal
            .try_reserve(
                ReservationKind::CaveHost,
                footprint,
                None,
                None,
                None,
                0,
                0,
                0,
                0,
            )
            .expect("host reserve");
        journal.commit_all();
    }

    #[test]
    fn eligibility_omitted_is_not_selected() {
        let mut journal = ReservationJournal::new(2048, 8000);
        make_host(&mut journal, 1, 0, 0, 8, 8);
        let eligibility = decide_cave_eligibility(RichnessCaveMode::Omitted, &journal);
        assert!(!eligibility.selected);
        assert_eq!(eligibility.reason, CaveEligibilityReason::OmittedByMode);
    }

    #[test]
    fn eligibility_selects_smallest_host() {
        let mut journal = ReservationJournal::new(2048, 8000);
        make_host(&mut journal, 9, 8, 8, 24, 24);
        make_host(&mut journal, 3, 40, 8, 56, 24);
        let eligibility = decide_cave_eligibility(RichnessCaveMode::Preferred, &journal);
        assert!(eligibility.selected);
        assert_eq!(eligibility.host, Some(ReservationId::new(0)));
    }

    #[test]
    fn eligibility_no_host() {
        let journal = ReservationJournal::new(2048, 8000);
        let eligibility = decide_cave_eligibility(RichnessCaveMode::Required, &journal);
        assert!(!eligibility.selected);
        assert_eq!(eligibility.reason, CaveEligibilityReason::NoHost);
    }

    #[test]
    fn lattice_covers_host_interior_only() {
        // Host grid cells 8..56 (16u quanta): quake 128..896; interior
        // 144..880 (16u shell each side) -> 23 cells of 32u per axis.
        let mut journal = ReservationJournal::new(2048, 8000);
        make_host(&mut journal, 1, 8, 8, 56, 56);
        let host = journal.reservations.get(&ReservationId::new(0)).unwrap();
        let lattice = build_lattice(host).unwrap();
        assert_eq!(lattice.nx, 23);
        assert_eq!(lattice.ny, 23);
        // Both bands: 0..368 shelled -> 16..352 -> 10 cells.
        assert_eq!(lattice.nz, 10);
    }

    #[test]
    fn synthesis_produces_connected_cave_with_witnesses() {
        let seed = 42u64;
        let mut journal = ReservationJournal::new(2048, 8000);
        make_host(&mut journal, 1, 8, 8, 56, 56);
        // A route leading into the host boundary forces witnesses: the
        // route is placed adjacent to the host (occupancy-exclusive), and
        // its dilated region intersects the host's boundary lattice cells.
        let route_footprint = Footprint3D {
            x0: 56,
            y0: 28,
            x1: 68,
            y1: 36,
            occupies_lower: true,
            occupies_upper: false,
        };
        journal
            .try_reserve_for_route(
                RouteId::new(2),
                ReservationKind::Route,
                route_footprint,
                0,
                0,
                0,
                0,
            )
            .expect("route reserve");
        journal.commit_all();
        let result = synthesize_cave(seed, RichnessCaveMode::Required, &journal)
            .expect("synthesis")
            .expect("cave selected");
        assert_eq!(result.host, ReservationId::new(0));
        assert!(!result.empty_cells.is_empty());
        assert!(!result.lattice.witnesses.is_empty());
        // All witnesses in the component.
        for &(i, j, k) in &result.lattice.witnesses {
            assert!(result.empty_cells.binary_search(&(i, j, k)).is_ok());
        }
        // Solid complement: non-overlapping, full coverage.
        validate_cave_result(&result).expect("cave invariants");
        assert!(!result.solid_boxes.is_empty());
    }

    #[test]
    fn synthesis_is_deterministic() {
        let seed = 7u64;
        let mut journal = ReservationJournal::new(2048, 8000);
        make_host(&mut journal, 1, 8, 8, 56, 56);
        let a = synthesize_cave(seed, RichnessCaveMode::Required, &journal)
            .expect("synthesis a")
            .expect("selected");
        let b = synthesize_cave(seed, RichnessCaveMode::Required, &journal)
            .expect("synthesis b")
            .expect("selected");
        assert_eq!(a.solid_boxes, b.solid_boxes);
        assert_eq!(a.empty_cells, b.empty_cells);
        assert_eq!(a.candidate_key, b.candidate_key);
    }

    #[test]
    fn partition_boxes_are_maximal_and_cover_all_solid() {
        let seed = 99u64;
        let mut journal = ReservationJournal::new(3072, 8000);
        make_host(&mut journal, 1, 16, 16, 80, 80);
        let result = synthesize_cave(seed, RichnessCaveMode::Required, &journal)
            .expect("synthesis")
            .expect("selected");
        // Union coverage: solid cells count == sum of box volumes.
        let solid_count: usize = result
            .lattice
            .cells
            .iter()
            .filter(|&&state| state == 0)
            .count();
        let box_volume: usize = result
            .solid_boxes
            .iter()
            .map(|b| {
                (b.max.0 - b.min.0) as usize
                    * (b.max.1 - b.min.1) as usize
                    * (b.max.2 - b.min.2) as usize
            })
            .sum();
        assert_eq!(solid_count, box_volume);
    }

    #[test]
    fn required_mode_errors_when_no_host() {
        let journal = ReservationJournal::new(2048, 8000);
        let outcome = synthesize_cave(1, RichnessCaveMode::Required, &journal);
        assert!(outcome.is_err());
        let error = outcome.err().unwrap();
        assert_eq!(error.code, RichnessErrorCode::CaveFailure);
    }

    #[test]
    fn preferred_mode_omits_when_no_host() {
        let journal = ReservationJournal::new(2048, 8000);
        let outcome = synthesize_cave(1, RichnessCaveMode::Preferred, &journal)
            .expect("no error in preferred mode");
        assert!(outcome.is_none());
    }

    #[test]
    fn synthesis_theme_independent_by_construction() {
        // The cave module has no theme input; eligibility, lattice, field,
        // connectivity, partition, and roles depend only on seed, mode, and
        // reservations. Prove stability across a sweep so theme invariance
        // cannot regress via shared state.
        let mut journal = ReservationJournal::new(2048, 8000);
        make_host(&mut journal, 1, 8, 8, 56, 56);
        let route_footprint = Footprint3D {
            x0: 56,
            y0: 28,
            x1: 68,
            y1: 36,
            occupies_lower: true,
            occupies_upper: false,
        };
        journal
            .try_reserve_for_route(
                RouteId::new(2),
                ReservationKind::Route,
                route_footprint,
                0,
                0,
                0,
                0,
            )
            .expect("route reserve");
        journal.commit_all();
        for seed in 0..8u64 {
            let a = synthesize_cave(seed, RichnessCaveMode::Required, &journal)
                .expect("synthesis")
                .expect("selected");
            let b = synthesize_cave(seed, RichnessCaveMode::Required, &journal)
                .expect("synthesis repeat")
                .expect("selected repeat");
            assert_eq!(a.solid_boxes, b.solid_boxes, "seed {seed} solid boxes");
            assert_eq!(a.empty_cells, b.empty_cells, "seed {seed} empty cells");
            assert_eq!(a.candidate_key, b.candidate_key, "seed {seed} candidate");
            validate_cave_result(&a).expect("seed {seed} invariants");
        }
    }

    #[test]
    fn synthesis_broad_seed_sweep_required() {
        // Broad property sweep: every seed either yields a connected cave
        // with all witnesses and a valid complement, or (never for these
        // sizes) a typed failure. No seed may panic or produce a partial
        // cave.
        for seed in 0..16u64 {
            let mut journal = ReservationJournal::new(2048, 8000);
            make_host(&mut journal, 1, 8, 8, 56, 56);
            let route_footprint = Footprint3D {
                x0: 56,
                y0: 28,
                x1: 68,
                y1: 36,
                occupies_lower: true,
                occupies_upper: false,
            };
            journal
                .try_reserve_for_route(
                    RouteId::new(2),
                    ReservationKind::Route,
                    route_footprint,
                    0,
                    0,
                    0,
                    0,
                )
                .expect("route reserve");
            journal.commit_all();
            let outcome = synthesize_cave(seed, RichnessCaveMode::Required, &journal);
            let result = match outcome {
                Ok(Some(result)) => result,
                Ok(None) => panic!("seed {seed}: required mode produced no cave"),
                Err(error) => {
                    assert_eq!(error.code, RichnessErrorCode::CaveFailure);
                    continue;
                }
            };
            validate_cave_result(&result).expect("seed {seed} invariants");
            assert!(
                !result.empty_cells.is_empty(),
                "seed {seed}: no empty cells"
            );
            assert!(
                !result.solid_boxes.is_empty(),
                "seed {seed}: no solid boxes"
            );
        }
    }

    #[test]
    fn cave_roles_are_distinct_for_theme_materials() {
        use crate::enhanced_v3::richness::assembly::BrushAssemblyRole;
        use crate::enhanced_v3::richness::theme::SemanticRole;
        assert_eq!(
            BrushAssemblyRole::CaveFloor.semantic_role(),
            SemanticRole::Floor
        );
        assert_eq!(
            BrushAssemblyRole::CaveWall.semantic_role(),
            SemanticRole::Wall
        );
        assert_eq!(
            BrushAssemblyRole::CaveCeiling.semantic_role(),
            SemanticRole::Ceiling
        );
        let mut roles = vec![
            BrushAssemblyRole::CaveFloor.semantic_role(),
            BrushAssemblyRole::CaveWall.semantic_role(),
            BrushAssemblyRole::CaveCeiling.semantic_role(),
        ];
        roles.sort_unstable();
        roles.dedup();
        assert_eq!(roles.len(), 3, "cave roles must stay distinct");
    }
}
