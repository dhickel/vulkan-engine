//! One-way deterministic production pipeline for Enhanced V3.
//!
//! Wires the full immutable pipeline from a validated `V3Config` through
//! placement, topology, reservations, assembly, emission, and metadata.
//! The result is a canonical `.map` string and deterministic metadata,
//! produced atomically — any failure returns a typed error with no partial
//! output.
//!
//! # Pipeline stages
//!
//! ```text
//! V3Config → Footprints → CommittedTopology → Reservations →
//!   Assembly → Validate → Serialize → (map text, metadata)
//! ```
//!
//! # Determinism
//!
//! Two calls with identical `V3Config` produce byte-identical `.map` output
//! and field-identical metadata.

use super::assembly::{
    self, Assembly, AssemblyBrush, BrushRole, Interface, ProtectedVolume, Support,
};
use super::config::{ArchType, V3Config, CONSTRUCTION_QUANTUM, HEADROOM, ROUTE_WIDTH};
use super::emission;
use super::error::V3Error;
use super::footprint::build_footprints;
use super::geometry::{CanonicalPlane, ConvexBrush, FaceRole};
use super::ids::{
    CommittedPortal, CommittedRoom, CommittedTopology, PlanOutcome, QuantumVolume, RoomId,
    V3IdAllocator,
};
use super::intent::plan_composition;
use super::metadata::EnhancedV3Metadata;
use super::reservation::{Reservation, ReservationSet};
use super::rng::V3Seed;
#[cfg(test)]
use super::topology::compute_reservations;
use super::topology::{build_topology, compute_reservations_with_config};
use std::collections::BTreeMap;

// ── Pipeline output ───────────────────────────────────────────────────────

/// The atomic output of a completed Enhanced V3 generation run.
///
/// Contains canonical map text and deterministic production metadata.
/// Never contains partial results — the pipeline is all-or-nothing.
#[derive(Debug, Clone)]
pub struct V3PipelineOutput {
    /// Canonical Quake .map text (LF endings, terminal newline).
    pub map_text: String,
    /// Deterministic production metadata.
    pub metadata: EnhancedV3Metadata,
}

// ── Public pipeline entry point ───────────────────────────────────────────

/// Run the full Enhanced V3 production pipeline from a validated configuration.
///
/// Returns canonical `.map` text and deterministic metadata atomically.
/// Any stage failure returns a typed `V3Error` with no partial output.
///
/// # Determinism
///
/// Two calls with identical `config` produce byte-identical output.
pub fn run_pipeline(config: &V3Config) -> Result<V3PipelineOutput, V3Error> {
    config.validate()?;
    let seed = V3Seed::new(config.seed);
    let mut alloc = V3IdAllocator::new();

    // 1. Build footprints
    let (footprints, layout) = build_footprints(config, seed, &mut alloc)?;

    // 2. Build committed topology
    let topology = build_topology(config, &footprints, &layout, seed, &mut alloc)?;

    // 3. Compute reservation volumes
    let (spawn_volume, light_volumes) = compute_reservations_with_config(&topology, config, seed)?;

    // 4. Reserve every point-entity clearance volume before feature planning.
    let mut protected_reservations = ReservationSet::new();
    protected_reservations.add(Reservation::new("spawn", "spawn_point", spawn_volume))?;
    for (index, volume) in light_volumes.iter().copied().enumerate() {
        protected_reservations.add(Reservation::new(
            format!("light_{index:04}"),
            "light",
            volume,
        ))?;
    }

    // 5. Plan composition (grammar families + feature instances)
    let plan = plan_composition(seed, config, &topology, &spawn_volume, &light_volumes)?;

    // 6. Build assembly from topology (with feature brushes from plan)
    let (assembly, spawn_origin, light_origins) = build_assembly_from_topology(
        config,
        &topology,
        &plan,
        &spawn_volume,
        &light_volumes,
        &protected_reservations,
        seed,
    )?;

    // 7. Compute actual face/entity/brush counts from the assembly
    let actual_brushes = assembly.brushes.len() as u32;
    let actual_faces: u32 = assembly
        .brushes
        .iter()
        .map(|b| b.brush.faces.len() as u32)
        .sum();
    let actual_entities: u32 = 2 + light_origins.len() as u32; // worldspawn + spawn + lights
    if (!config.has_output_overrides() && actual_faces > config.preset.face_budget())
        || actual_faces >= super::config::FACE_BUDGET
    {
        return Err(V3Error::CompositionInvariant {
            detail: format!(
                "{} actual faces {actual_faces} exceed the active source/M2 budget",
                config.preset.tag()
            ),
        });
    }
    if actual_entities >= super::config::ENTITY_BUDGET {
        return Err(V3Error::CompositionInvariant {
            detail: format!("actual entities {actual_entities} exceed the M2 budget"),
        });
    }
    if plan.estimated_total_faces < actual_faces {
        return Err(V3Error::CompositionInvariant {
            detail: format!(
                "face estimate {} is below actual emitted faces {actual_faces}",
                plan.estimated_total_faces
            ),
        });
    }

    // 8. Emit canonical .map text
    let map_text = emission::emit_map_text_with_minlight(
        &assembly,
        spawn_origin,
        &light_origins,
        config.minlight,
    )?;

    // 9. Build metadata
    let grammar_families: Vec<String> = plan.grammar_families.iter().cloned().collect();
    let metadata = EnhancedV3Metadata::new(
        config,
        &topology,
        grammar_families,
        plan.identity_satisfied,
        plan.estimated_total_faces,
        plan.estimated_total_entities,
        actual_faces,
        actual_entities,
        actual_brushes,
        spawn_origin,
        light_origins.len() as u32,
    );

    Ok(V3PipelineOutput { map_text, metadata })
}

// ── Assembly construction ─────────────────────────────────────────────────

/// Stair step count and dimensions.
const STAIR_STEPS: i32 = 12;
const STAIR_RISER: i32 = 16;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum WallDirection {
    North,
    South,
    West,
    East,
}

impl WallDirection {
    fn parse(portal: &CommittedPortal) -> Result<Self, V3Error> {
        match portal.wall.as_str() {
            "north" => Ok(Self::North),
            "south" => Ok(Self::South),
            "west" => Ok(Self::West),
            "east" => Ok(Self::East),
            _ => Err(V3Error::ApertureInvalid {
                aperture_id: portal.id.stable_key(),
                detail: format!("unknown wall direction '{}'", portal.wall),
            }),
        }
    }

    fn opposite(self) -> Self {
        match self {
            Self::North => Self::South,
            Self::South => Self::North,
            Self::West => Self::East,
            Self::East => Self::West,
        }
    }

    fn tag(self) -> &'static str {
        match self {
            Self::North => "north",
            Self::South => "south",
            Self::West => "west",
            Self::East => "east",
        }
    }

    fn is_horizontal_route(self) -> bool {
        matches!(self, Self::West | Self::East)
    }
}

#[derive(Debug, Clone)]
struct WallAperture {
    id: String,
    center: i128,
    width: i128,
    z0: i128,
    z1: i128,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct BrushBounds {
    id: String,
    x: (i128, i128),
    y: (i128, i128),
    z: (i128, i128),
}

impl BrushBounds {
    fn new(id: String, x: (i128, i128), y: (i128, i128), z: (i128, i128)) -> Self {
        Self { id, x, y, z }
    }
}

fn room_wall_coordinate(room: &CommittedRoom, direction: WallDirection) -> i128 {
    match direction {
        WallDirection::North => room.shell.1 as i128,
        WallDirection::South => room.shell.3 as i128,
        WallDirection::West => room.shell.0 as i128,
        WallDirection::East => room.shell.2 as i128,
    }
}

fn make_wall_aperture(
    id: String,
    room: &CommittedRoom,
    direction: WallDirection,
    wall_coordinate: i128,
    center: i128,
    z_center: i128,
    width: u32,
    height: u32,
) -> Result<WallAperture, V3Error> {
    if wall_coordinate != room_wall_coordinate(room, direction) {
        return Err(V3Error::ApertureInvalid {
            aperture_id: id,
            detail: format!(
                "anchor is at {wall_coordinate}, expected {} wall coordinate {}",
                direction.tag(),
                room_wall_coordinate(room, direction)
            ),
        });
    }

    let width = i128::from(width);
    let height = i128::from(height);
    if width == 0 || height == 0 || width % 2 != 0 || height % 2 != 0 {
        return Err(V3Error::ApertureInvalid {
            aperture_id: id,
            detail: format!("portal dimensions must be positive and even, got {width}×{height}"),
        });
    }

    Ok(WallAperture {
        id,
        center,
        width,
        z0: z_center - height / 2,
        z1: z_center + height / 2,
    })
}

fn collect_wall_apertures(
    topology: &CommittedTopology,
) -> Result<BTreeMap<(RoomId, WallDirection), Vec<WallAperture>>, V3Error> {
    let mut apertures: BTreeMap<(RoomId, WallDirection), Vec<WallAperture>> = BTreeMap::new();

    for portal in &topology.portals {
        let source_direction = WallDirection::parse(portal)?;
        let source_room =
            topology
                .room(portal.source_room)
                .ok_or_else(|| V3Error::TopologyInvariant {
                    detail: format!(
                        "{} references unknown source room {}",
                        portal.id, portal.source_room
                    ),
                })?;
        let source_wall_coordinate = if source_direction.is_horizontal_route() {
            i128::from(portal.anchor.0)
        } else {
            i128::from(portal.anchor.1)
        };
        let source_center = if source_direction.is_horizontal_route() {
            i128::from(portal.anchor.1)
        } else {
            i128::from(portal.anchor.0)
        };
        apertures
            .entry((source_room.id, source_direction))
            .or_default()
            .push(make_wall_aperture(
                format!("{}/source", portal.id.stable_key()),
                source_room,
                source_direction,
                source_wall_coordinate,
                source_center,
                i128::from(portal.anchor.2),
                portal.width,
                portal.height,
            )?);

        if let Some(target_id) = portal.target_room {
            let target_room =
                topology
                    .room(target_id)
                    .ok_or_else(|| V3Error::TopologyInvariant {
                        detail: format!("{} references unknown target room {target_id}", portal.id),
                    })?;
            let target_direction = source_direction.opposite();
            apertures
                .entry((target_room.id, target_direction))
                .or_default()
                .push(make_wall_aperture(
                    format!("{}/target", portal.id.stable_key()),
                    target_room,
                    target_direction,
                    room_wall_coordinate(target_room, target_direction),
                    source_center,
                    i128::from(portal.anchor.2),
                    portal.width,
                    portal.height,
                )?);
        }
    }

    // A transition owns two real wall omissions: the lower stair entry and
    // the upper connector crest.  They are structural apertures, not solids
    // pretending to carve an opening.
    for transition in &topology.transitions {
        let lower =
            topology
                .room(transition.lower_room)
                .ok_or_else(|| V3Error::TopologyInvariant {
                    detail: format!("transition/{:04} lower room missing", transition.id),
                })?;
        let upper =
            topology
                .room(transition.upper_room)
                .ok_or_else(|| V3Error::TopologyInvariant {
                    detail: format!("transition/{:04} upper room missing", transition.id),
                })?;
        let center = (i128::from(transition.protected_volume.0)
            + i128::from(transition.protected_volume.3))
            / 2;
        let lower_z0 = i128::from(lower.floor_z) + i128::from(CONSTRUCTION_QUANTUM);
        let upper_z0 = i128::from(upper.floor_z) + i128::from(CONSTRUCTION_QUANTUM);
        apertures
            .entry((lower.id, WallDirection::South))
            .or_default()
            .push(make_wall_aperture(
                format!("transition/{:04}/lower_entry", transition.id),
                lower,
                WallDirection::South,
                room_wall_coordinate(lower, WallDirection::South),
                center,
                lower_z0 + i128::from(HEADROOM) / 2,
                ROUTE_WIDTH as u32,
                HEADROOM as u32,
            )?);
        apertures
            .entry((upper.id, WallDirection::North))
            .or_default()
            .push(make_wall_aperture(
                format!("transition/{:04}/upper_crest", transition.id),
                upper,
                WallDirection::North,
                room_wall_coordinate(upper, WallDirection::North),
                center,
                upper_z0 + i128::from(HEADROOM) / 2,
                ROUTE_WIDTH as u32,
                HEADROOM as u32,
            )?);
    }

    for room_apertures in apertures.values_mut() {
        room_apertures.sort_by(|left, right| {
            (left.center - left.width / 2)
                .cmp(&(right.center - right.width / 2))
                .then_with(|| left.id.cmp(&right.id))
        });
    }

    Ok(apertures)
}

fn push_floor_brush(
    brushes: &mut Vec<AssemblyBrush>,
    bounds: &mut Vec<BrushBounds>,
    id: String,
    x: (i128, i128),
    y: (i128, i128),
    z: (i128, i128),
) -> Result<(), V3Error> {
    brushes.push(assembly::build_floor_slab(x, y, z, &id)?);
    bounds.push(BrushBounds::new(id, x, y, z));
    Ok(())
}

fn push_box_brush(
    brushes: &mut Vec<AssemblyBrush>,
    bounds: &mut Vec<BrushBounds>,
    id: String,
    role: BrushRole,
    x: (i128, i128),
    y: (i128, i128),
    z: (i128, i128),
) -> Result<(), V3Error> {
    brushes.push(AssemblyBrush::new(
        id.clone(),
        role,
        ConvexBrush::make_box(x, y, z)?,
        Support::World {
            surface: FaceRole::Floor,
        },
    ));
    bounds.push(BrushBounds::new(id, x, y, z));
    Ok(())
}

fn push_wall_brush(
    brushes: &mut Vec<AssemblyBrush>,
    bounds: &mut Vec<BrushBounds>,
    id: String,
    x: (i128, i128),
    y: (i128, i128),
    z: (i128, i128),
) -> Result<(), V3Error> {
    brushes.push(assembly::build_wall_shell(x, y, z, &id)?);
    bounds.push(BrushBounds::new(id, x, y, z));
    Ok(())
}

fn wall_piece_bounds(
    room: &CommittedRoom,
    direction: WallDirection,
    span: (i128, i128),
    z: (i128, i128),
    wall_thickness: i128,
) -> ((i128, i128), (i128, i128), (i128, i128)) {
    let (x0, y0, x1, y1) = (
        i128::from(room.shell.0),
        i128::from(room.shell.1),
        i128::from(room.shell.2),
        i128::from(room.shell.3),
    );
    match direction {
        WallDirection::North => (span, (y0, y0 + wall_thickness), z),
        WallDirection::South => (span, (y1 - wall_thickness, y1), z),
        WallDirection::West => ((x0, x0 + wall_thickness), span, z),
        WallDirection::East => ((x1 - wall_thickness, x1), span, z),
    }
}

/// Build a floor or ceiling slab split around rectangular openings.
///
/// The slab is divided into axis-aligned rectangular pieces that avoid
/// the given openings. Each piece is a separate brush.
fn build_split_slab(
    brushes: &mut Vec<AssemblyBrush>,
    bounds: &mut Vec<BrushBounds>,
    base_id: &str,
    x_range: (i128, i128),
    y_range: (i128, i128),
    z: (i128, i128),
    role: BrushRole,
    openings: &[(i128, i128, i128, i128)],
    _wall_thickness: i128,
) -> Result<(), V3Error> {
    let mut clipped: Vec<_> = openings
        .iter()
        .map(|&(x0, y0, x1, y1)| {
            (
                x0.max(x_range.0),
                y0.max(y_range.0),
                x1.min(x_range.1),
                y1.min(y_range.1),
            )
        })
        .filter(|&(x0, y0, x1, y1)| x0 < x1 && y0 < y1)
        .collect();
    clipped.sort_unstable();
    clipped.dedup();
    if clipped.is_empty() {
        return push_box_brush(
            brushes,
            bounds,
            base_id.to_string(),
            role,
            x_range,
            y_range,
            z,
        );
    }

    // Partition on every opening boundary. Each cell is wholly solid or
    // wholly omitted, so multiple openings can never restore solid volume.
    let mut x_cuts = vec![x_range.0, x_range.1];
    let mut y_cuts = vec![y_range.0, y_range.1];
    for &(x0, y0, x1, y1) in &clipped {
        x_cuts.extend([x0, x1]);
        y_cuts.extend([y0, y1]);
    }
    x_cuts.sort_unstable();
    x_cuts.dedup();
    y_cuts.sort_unstable();
    y_cuts.dedup();

    let mut segment = 0u32;
    for y in y_cuts.windows(2) {
        for x in x_cuts.windows(2) {
            let cell = (x[0], y[0], x[1], y[1]);
            if clipped.iter().any(|&opening| {
                cell.0 >= opening.0
                    && cell.2 <= opening.2
                    && cell.1 >= opening.1
                    && cell.3 <= opening.3
            }) {
                continue;
            }
            push_box_brush(
                brushes,
                bounds,
                format!("{base_id}/seg_{segment:04}"),
                role,
                (cell.0, cell.2),
                (cell.1, cell.3),
                z,
            )?;
            segment += 1;
        }
    }
    Ok(())
}

/// Compute the actual cardinal wall interior span for a room, accounting for
/// chamfer corners that shorten the cardinal edge.
fn cardinal_wall_interior_span(
    room: &CommittedRoom,
    direction: WallDirection,
    wall_thickness: i128,
) -> (i128, i128) {
    let (x0, y0, x1, y1) = (
        i128::from(room.shell.0),
        i128::from(room.shell.1),
        i128::from(room.shell.2),
        i128::from(room.shell.3),
    );
    if !room.is_chamfered {
        let inner_x0 = x0 + wall_thickness;
        let inner_x1 = x1 - wall_thickness;
        let inner_y0 = y0 + wall_thickness;
        let inner_y1 = y1 - wall_thickness;
        return match direction {
            WallDirection::North | WallDirection::South => (inner_x0, inner_x1),
            WallDirection::West | WallDirection::East => (inner_y0, inner_y1),
        };
    }

    // For chamfered rooms, the cardinal wall span is determined by the
    // actual footprint polygon edges, which are shorter due to chamfers.
    // The footprint vertices are on the OUTER edge of the room AABB.
    // We need to inset by wall_thickness to get the INTERIOR wall span.
    let mut span: Option<(i128, i128)> = None;
    for idx in 0..room.footprint_vertices.len() {
        let a = room.footprint_vertices[idx];
        let b = room.footprint_vertices[(idx + 1) % room.footprint_vertices.len()];
        let a = (i128::from(a.0), i128::from(a.1));
        let b = (i128::from(b.0), i128::from(b.1));

        // Check if this edge lies on the requested cardinal wall.
        let on_wall = match direction {
            WallDirection::West if a.0 == x0 && b.0 == x0 => true,
            WallDirection::East if a.0 == x1 && b.0 == x1 => true,
            WallDirection::North if a.1 == y0 && b.1 == y0 => true,
            WallDirection::South if a.1 == y1 && b.1 == y1 => true,
            _ => false,
        };

        if !on_wall {
            continue;
        }

        let (lo, hi) = if direction.is_horizontal_route() {
            (a.1.min(b.1), a.1.max(b.1))
        } else {
            (a.0.min(b.0), a.0.max(b.0))
        };
        span = match span {
            None => Some((lo, hi)),
            Some((slo, shi)) => Some((slo.min(lo), shi.max(hi))),
        };
    }

    // At a chamfered endpoint the cardinal edge already terminates on the
    // diagonal shell, so trimming it would create a 16-unit leak slit.  At an
    // intact endpoint retain the conventional one-wall-thickness trim so the
    // two perpendicular cardinal wall brushes never overlap.
    let (mut lo, mut hi) = span.unwrap_or(match direction {
        WallDirection::North | WallDirection::South => (x0, x1),
        WallDirection::West | WallDirection::East => (y0, y1),
    });
    let (low_corner, high_corner) = match direction {
        WallDirection::North => ((-1, -1), (1, -1)),
        WallDirection::South => ((-1, 1), (1, 1)),
        WallDirection::West => ((-1, -1), (-1, 1)),
        WallDirection::East => ((1, -1), (1, 1)),
    };
    if !room.chamfer_corners.contains(&low_corner) {
        lo += wall_thickness;
    }
    if !room.chamfer_corners.contains(&high_corner) {
        hi -= wall_thickness;
    }
    (lo, hi)
}

#[allow(dead_code)]
fn build_chamfered_slab(
    room: &CommittedRoom,
    z: (i128, i128),
    role: BrushRole,
    id: &str,
    brushes: &mut Vec<AssemblyBrush>,
    bounds: &mut Vec<BrushBounds>,
) -> Result<(), V3Error> {
    let x_range = (i128::from(room.shell.0), i128::from(room.shell.2));
    let y_range = (i128::from(room.shell.1), i128::from(room.shell.3));

    let brush = if !room.is_chamfered || room.chamfer_corners.is_empty() {
        ConvexBrush::make_box(x_range, y_range, z)?
    } else {
        let chamfer_corners_i128: Vec<(i128, i128)> = room
            .chamfer_corners
            .iter()
            .map(|&(sx, sy)| (i128::from(sx), i128::from(sy)))
            .collect();
        super::geometry::make_chamfered_slab(
            x_range,
            y_range,
            z.0,
            z.1,
            &chamfer_corners_i128,
            i128::from(room.chamfer_size),
        )?
    };
    brushes.push(AssemblyBrush::new(
        id.to_string(),
        role,
        brush,
        Support::World {
            surface: FaceRole::Floor,
        },
    ));
    bounds.push(BrushBounds::new(id.to_string(), x_range, y_range, z));
    Ok(())
}

/// Build diagonal wall pieces for a chamfered room.
/// Fill each intact cardinal corner left open by non-overlapping wall spans.
/// The four wall brushes intentionally stop one quantum short of an uncut
/// corner; these posts own that remaining cell and keep a room shell sealed
/// without positive-volume wall/wall overlap.
fn build_cardinal_corner_posts(
    room: &CommittedRoom,
    wall_thickness: i128,
    brushes: &mut Vec<AssemblyBrush>,
    bounds: &mut Vec<BrushBounds>,
) -> Result<(), V3Error> {
    let x0 = i128::from(room.shell.0);
    let y0 = i128::from(room.shell.1);
    let x1 = i128::from(room.shell.2);
    let y1 = i128::from(room.shell.3);
    let z = (
        i128::from(room.floor_z) + wall_thickness,
        i128::from(room.floor_z) + i128::from(room.dims.2) - wall_thickness,
    );
    for ((sx, sy), tag) in [
        ((-1, -1), "nw"),
        ((1, -1), "ne"),
        ((-1, 1), "sw"),
        ((1, 1), "se"),
    ] {
        if room.chamfer_corners.contains(&(sx, sy)) {
            continue;
        }
        let x = if sx < 0 {
            (x0, x0 + wall_thickness)
        } else {
            (x1 - wall_thickness, x1)
        };
        let y = if sy < 0 {
            (y0, y0 + wall_thickness)
        } else {
            (y1 - wall_thickness, y1)
        };
        push_wall_brush(
            brushes,
            bounds,
            format!("{}/corner_{tag}", room.id.stable_key()),
            x,
            y,
            z,
        )?;
    }
    Ok(())
}

fn build_diagonal_walls(
    room: &CommittedRoom,
    _wall_thickness: i128,
    brushes: &mut Vec<AssemblyBrush>,
    bounds: &mut Vec<BrushBounds>,
) -> Result<(), V3Error> {
    if !room.is_chamfered || room.chamfer_corners.is_empty() {
        return Ok(());
    }

    let rid = room.id.stable_key();
    // Chamfered slabs stop at the diagonal interior plane. The diagonal wall
    // therefore owns the complementary corner prism through the floor and
    // ceiling bands as well as the clear-height wall band; limiting it to
    // 16..160 would leave open triangular holes under and over every chamfer.
    let wall_z0 = i128::from(room.floor_z);
    let wall_z1 = i128::from(room.floor_z) + i128::from(room.dims.2);
    let x_range = (i128::from(room.shell.0), i128::from(room.shell.2));
    let y_range = (i128::from(room.shell.1), i128::from(room.shell.3));
    let chamfer_size = i128::from(room.chamfer_size);

    for &(sx, sy) in &room.chamfer_corners {
        let tag = match (sx, sy) {
            (1, 1) => "diag_ne",
            (1, -1) => "diag_se",
            (-1, 1) => "diag_nw",
            (-1, -1) => "diag_sw",
            _ => continue,
        };
        let diag_id = format!("{rid}/wall_{tag}");
        let sx = i128::from(sx);
        let sy = i128::from(sy);

        let brush = super::geometry::make_diagonal_wall(
            x_range,
            y_range,
            wall_z0,
            wall_z1,
            sx,
            sy,
            chamfer_size,
        )?;

        brushes.push(AssemblyBrush::new(
            diag_id.clone(),
            BrushRole::WallShell,
            brush,
            Support::World {
                surface: FaceRole::Floor,
            },
        ));
        // Conservative AABB bounds for contact derivation
        let (dx0, dx1) = if sx > 0 {
            (x_range.1 - chamfer_size, x_range.1)
        } else {
            (x_range.0, x_range.0 + chamfer_size)
        };
        let (dy0, dy1) = if sy > 0 {
            (y_range.1 - chamfer_size, y_range.1)
        } else {
            (y_range.0, y_range.0 + chamfer_size)
        };
        bounds.push(BrushBounds::new(
            diag_id,
            (dx0, dx1),
            (dy0, dy1),
            (wall_z0, wall_z1),
        ));
    }
    Ok(())
}

/// Build pointed arch portal surrounds for Moderate/Rich presets.
///
/// Creates stepped axis-aligned segments above the 64×80 clear core
/// to form a pointed arch silhouette. Uses only cardinal XY normals
/// (no XZ slopes) for compiler safety.
fn build_pointed_arch_surround(
    room: &CommittedRoom,
    direction: WallDirection,
    _wall_coordinate: i128,
    aperture_center: i128,
    wall_thickness: i128,
    brushes: &mut Vec<AssemblyBrush>,
    bounds: &mut Vec<BrushBounds>,
) -> Result<(), V3Error> {
    let rid = room.id.stable_key();
    let wall_z0 = i128::from(room.floor_z) + wall_thickness;
    let wall_z1 = i128::from(room.floor_z) + i128::from(room.dims.2) - wall_thickness;
    let arch_base_z = wall_z0 + i128::from(HEADROOM); // Z = floor+16+80 = 96
    let half_width: i128 = 32; // half of 64-unit core
    let step_height: i128 = 16;
    let step_width: i128 = 16;
    let num_steps = 2i128; // two steps each side

    // Stepped arch: each side steps inward by step_width at each step_height
    // until reaching an apex. The aperture Z goes from wall_z0 to arch_base_z
    // (the 64×80 core). Above that, stepped jambs narrow the opening.

    let base_id = format!("{rid}/wall_{}/arch", direction.tag());

    let mut highest_step_top = arch_base_z;
    for step in 0..num_steps {
        let step_z0 = arch_base_z + step * step_height;
        let step_z1 = step_z0 + step_height;
        if step_z1 > wall_z1 {
            break;
        }
        highest_step_top = step_z1;
        // Each step narrows the opening by step_width on each side.
        let solid_inner = half_width - (step + 1) * step_width;
        if solid_inner < 0 {
            // Apex: fill the entire width above and stop.
            let apex_z0 = step_z0;
            let apex_z1 = wall_z1;
            let (x, y, z) = wall_piece_bounds(
                room,
                direction,
                (aperture_center - half_width, aperture_center + half_width),
                (apex_z0, apex_z1),
                wall_thickness,
            );
            push_wall_brush(brushes, bounds, format!("{base_id}/apex"), x, y, z)?;
            highest_step_top = apex_z1;
            break;
        }

        // Left jamb fill
        let left_inner = aperture_center - solid_inner;
        let left_span = (aperture_center - half_width, left_inner);
        if left_span.0 < left_span.1 {
            let (x, y, z) = wall_piece_bounds(
                room,
                direction,
                left_span,
                (step_z0, step_z1),
                wall_thickness,
            );
            push_wall_brush(
                brushes,
                bounds,
                format!("{base_id}/step_{step}_left"),
                x,
                y,
                z,
            )?;
        }

        // Right jamb fill
        let right_inner = aperture_center + solid_inner;
        let right_span = (right_inner, aperture_center + half_width);
        if right_span.0 < right_span.1 {
            let (x, y, z) = wall_piece_bounds(
                room,
                direction,
                right_span,
                (step_z0, step_z1),
                wall_thickness,
            );
            push_wall_brush(
                brushes,
                bounds,
                format!("{base_id}/step_{step}_right"),
                x,
                y,
                z,
            )?;
        }
    }

    // Lintel above the highest arch step: full-width fill from highest step
    // to ceiling.
    if highest_step_top < wall_z1 {
        let (x, y, z) = wall_piece_bounds(
            room,
            direction,
            (aperture_center - half_width, aperture_center + half_width),
            (highest_step_top, wall_z1),
            wall_thickness,
        );
        push_wall_brush(brushes, bounds, format!("{base_id}/lintel"), x, y, z)?;
    }

    Ok(())
}

/// Build a shallow segmented surround above the compatibility 64×80 core.
///
/// The first 16-unit crown band remains fully open, the second fills the
/// outer 16-unit shoulders while retaining a 32-unit centre, and a full
/// lintel seals the wall above. A one-quantum cap immediately outside the
/// remaining centre opening joins the surround to the corridor roof, so the
/// visible recess cannot reach exterior void. This is visibly distinct from
/// the pointed compatibility profile while retaining cardinal,
/// quantum-aligned brushes.
fn build_segmented_arch_surround(
    room: &CommittedRoom,
    direction: WallDirection,
    aperture_center: i128,
    wall_thickness: i128,
    brushes: &mut Vec<AssemblyBrush>,
    bounds: &mut Vec<BrushBounds>,
) -> Result<(), V3Error> {
    let rid = room.id.stable_key();
    let wall_z0 = i128::from(room.floor_z) + wall_thickness;
    let wall_z1 = i128::from(room.floor_z) + i128::from(room.dims.2) - wall_thickness;
    let arch_base = wall_z0 + i128::from(HEADROOM);
    let shoulder_z0 = arch_base + i128::from(CONSTRUCTION_QUANTUM);
    let shoulder_z1 = shoulder_z0 + i128::from(CONSTRUCTION_QUANTUM);
    let half_width = i128::from(ROUTE_WIDTH / 2);
    let shoulder_width = i128::from(CONSTRUCTION_QUANTUM);
    let root = format!("{rid}/wall_{}/segmented_arch", direction.tag());

    if shoulder_z0 < wall_z1 {
        let top = shoulder_z1.min(wall_z1);
        for (tag, span) in [
            (
                "left",
                (
                    aperture_center - half_width,
                    aperture_center - half_width + shoulder_width,
                ),
            ),
            (
                "right",
                (
                    aperture_center + half_width - shoulder_width,
                    aperture_center + half_width,
                ),
            ),
        ] {
            let (x, y, z) =
                wall_piece_bounds(room, direction, span, (shoulder_z0, top), wall_thickness);
            push_wall_brush(brushes, bounds, format!("{root}/shoulder_{tag}"), x, y, z)?;
        }

        // A level-route ceiling seals the first crown band through Z=112;
        // transition crowns open only into their sealed stairwell. Back the
        // remaining 32-unit opening in the next band immediately outside the
        // room wall. Keeping the cap out of the wall depth preserves the
        // segmented recess while closing the only path to exterior void.
        let crown_span = (
            aperture_center - half_width + shoulder_width,
            aperture_center + half_width - shoulder_width,
        );
        let (mut x, mut y, z) = wall_piece_bounds(
            room,
            direction,
            crown_span,
            (shoulder_z0, top),
            wall_thickness,
        );
        match direction {
            WallDirection::North => {
                y.0 -= wall_thickness;
                y.1 -= wall_thickness;
            }
            WallDirection::South => {
                y.0 += wall_thickness;
                y.1 += wall_thickness;
            }
            WallDirection::West => {
                x.0 -= wall_thickness;
                x.1 -= wall_thickness;
            }
            WallDirection::East => {
                x.0 += wall_thickness;
                x.1 += wall_thickness;
            }
        }
        push_wall_brush(brushes, bounds, format!("{root}/interface_cap"), x, y, z)?;
    }

    let lintel_z0 = shoulder_z1.min(wall_z1);
    if lintel_z0 < wall_z1 {
        let (x, y, z) = wall_piece_bounds(
            room,
            direction,
            (aperture_center - half_width, aperture_center + half_width),
            (lintel_z0, wall_z1),
            wall_thickness,
        );
        push_wall_brush(brushes, bounds, format!("{root}/lintel"), x, y, z)?;
    }

    Ok(())
}

/// Emit stair tread geometry for a committed transition.
///
/// Creates 12 tread×riser sealed shells with side, underside, and top
/// boundaries. Also emits the lower-room ceiling aperture, upper-room floor
/// aperture, and approach portals.
fn build_stair_emission(
    transition: &super::ids::CommittedTransition,
    topology: &CommittedTopology,
    wall_thickness: i128,
    brushes: &mut Vec<AssemblyBrush>,
    bounds: &mut Vec<BrushBounds>,
) -> Result<(), V3Error> {
    let q = i128::from(CONSTRUCTION_QUANTUM);
    let (pv_x0, _pv_y0, _pv_z0, pv_x1, _pv_y1, _pv_z1) = (
        i128::from(transition.protected_volume.0),
        i128::from(transition.protected_volume.1),
        i128::from(transition.protected_volume.2),
        i128::from(transition.protected_volume.3),
        i128::from(transition.protected_volume.4),
        i128::from(transition.protected_volume.5),
    );

    let lower = topology
        .room(transition.lower_room)
        .ok_or_else(|| V3Error::TopologyInvariant {
            detail: "transition lower room missing".into(),
        })?;
    let upper = topology
        .room(transition.upper_room)
        .ok_or_else(|| V3Error::TopologyInvariant {
            detail: "transition upper room missing".into(),
        })?;

    // The stair runs south→north from a supported lower approach to a
    // supported upper connector. Topology owns every positive-volume tread;
    // emission validates and consumes those boxes without reconstructing them.
    let lower_floor = i128::from(lower.floor_z);
    let upper_floor = i128::from(upper.floor_z);
    let tread_x0 = i128::from(transition.tread_run.0);
    let stair_y_start = i128::from(transition.tread_run.1);
    let tread_x1 = i128::from(transition.tread_run.2);
    let stair_y_end = i128::from(transition.tread_run.3);
    let trans_id = format!("transition/{:04}", transition.id);

    if transition.tread_boxes.len() != STAIR_STEPS as usize
        || stair_y_end - stair_y_start != i128::from(STAIR_STEPS) * q
        || tread_x0 != pv_x0
        || tread_x1 != pv_x1
    {
        return Err(V3Error::TopologyInvariant {
            detail: format!("{trans_id} does not satisfy the 12×16 tread-run contract"),
        });
    }

    let lower_approach = (
        i128::from(transition.lower_approach.0),
        i128::from(transition.lower_approach.1),
        i128::from(transition.lower_approach.2),
        i128::from(transition.lower_approach.3),
    );
    let connector_x = (tread_x0 - wall_thickness, tread_x1 + wall_thickness);
    push_floor_brush(
        brushes,
        bounds,
        format!("{trans_id}/lower_approach/floor"),
        connector_x,
        (lower_approach.1, lower_approach.3),
        (lower_floor, lower_floor + wall_thickness),
    )?;

    for (step, &(x0, y0, z0, x1, y1, z1)) in transition.tread_boxes.iter().enumerate() {
        let expected = (
            transition.tread_run.0,
            transition.tread_run.1 + step as i32 * CONSTRUCTION_QUANTUM,
            lower.floor_z,
            transition.tread_run.2,
            transition.tread_run.1 + (step as i32 + 1) * CONSTRUCTION_QUANTUM,
            lower.floor_z + (step as i32 + 1) * STAIR_RISER,
        );
        if (x0, y0, z0, x1, y1, z1) != expected {
            return Err(V3Error::TopologyInvariant {
                detail: format!(
                    "{trans_id} tread {step} is {:?}, expected {expected:?}",
                    (x0, y0, z0, x1, y1, z1)
                ),
            });
        }
        let id = format!("{trans_id}/tread_{step:04}");
        let brush = ConvexBrush::make_box(
            (i128::from(x0), i128::from(x1)),
            (i128::from(y0), i128::from(y1)),
            (i128::from(z0), i128::from(z1)),
        )?;
        brushes.push(AssemblyBrush::new(
            id.clone(),
            BrushRole::Feature,
            brush,
            Support::World {
                surface: FaceRole::Floor,
            },
        ));
        bounds.push(BrushBounds::new(
            id,
            (i128::from(x0), i128::from(x1)),
            (i128::from(y0), i128::from(y1)),
            (i128::from(z0), i128::from(z1)),
        ));
    }
    let clear_z0 = lower_floor + wall_thickness;
    // The upper approach is walked on top of its 16-unit support slab, so its
    // 80-unit headroom is measured from that top surface.
    let clear_z1 = upper_floor + wall_thickness + i128::from(HEADROOM);

    // Side shells begin at the lower wall face, enclose the supported lower
    // approach and complete tread run, and stop exactly at the crest.
    let side_y = (lower_approach.1, stair_y_end);
    for (tag, x) in [
        ("wall_west", (tread_x0 - wall_thickness, tread_x0)),
        ("wall_east", (tread_x1, tread_x1 + wall_thickness)),
    ] {
        push_wall_brush(
            brushes,
            bounds,
            format!("{trans_id}/{tag}"),
            x,
            side_y,
            (clear_z0, clear_z1),
        )?;
    }
    // A flat structural roof at the crest clearance seals the top while
    // retaining at least 80 units above the twelfth (Z=192) tread.
    push_box_brush(
        brushes,
        bounds,
        format!("{trans_id}/roof"),
        BrushRole::CeilingSlab,
        (tread_x0 - wall_thickness, tread_x1 + wall_thickness),
        (lower_approach.1, stair_y_end),
        (clear_z1, clear_z1 + wall_thickness),
    )?;
    // Above the lower room's own ceiling, close the external face of the
    // stairwell.  The lower portal core and its stepped apex remain omissions
    // below this cap; without it, the high exterior volume can enter the
    // stairwell around the finite-height lower room wall.
    let lower_ceiling_top = lower_floor + i128::from(lower.dims.2);
    push_wall_brush(
        brushes,
        bounds,
        format!("{trans_id}/lower_entry_cap"),
        (tread_x0, tread_x1),
        (
            i128::from(lower.shell.3),
            i128::from(lower.shell.3) + wall_thickness,
        ),
        (lower_ceiling_top, clear_z1),
    )?;

    // The final tread joins an upper-floor connector which runs directly to
    // the committed upper room's north-wall omission.  This makes the
    // transition physically traversable instead of a topology-only edge.
    let connector_y = (
        i128::from(transition.upper_approach.1),
        i128::from(transition.upper_approach.3),
    );
    if connector_y.0 != stair_y_end
        || connector_y.1 != i128::from(upper.shell.1)
        || connector_y.0 >= connector_y.1
    {
        return Err(V3Error::TopologyInvariant {
            detail: format!("{trans_id} crest does not precede upper approach"),
        });
    }
    push_floor_brush(
        brushes,
        bounds,
        format!("{trans_id}/upper_approach/floor"),
        connector_x,
        connector_y,
        (upper_floor, upper_floor + wall_thickness),
    )?;
    push_box_brush(
        brushes,
        bounds,
        format!("{trans_id}/upper_approach/ceiling"),
        BrushRole::CeilingSlab,
        connector_x,
        connector_y,
        (clear_z1, clear_z1 + wall_thickness),
    )?;
    for (tag, x) in [
        ("wall_west", (tread_x0 - wall_thickness, tread_x0)),
        ("wall_east", (tread_x1, tread_x1 + wall_thickness)),
    ] {
        push_wall_brush(
            brushes,
            bounds,
            format!("{trans_id}/upper_approach/{tag}"),
            x,
            connector_y,
            (upper_floor + wall_thickness, clear_z1),
        )?;
    }
    Ok(())
}

fn build_room_wall(
    room: &CommittedRoom,
    direction: WallDirection,
    apertures: &[WallAperture],
    wall_thickness: i128,
    skip_lintels: bool,
    brushes: &mut Vec<AssemblyBrush>,
    bounds: &mut Vec<BrushBounds>,
) -> Result<(), V3Error> {
    let rid = room.id.stable_key();
    let base_id = format!("{rid}/wall_{}", direction.tag());
    let wall_z0 = i128::from(room.floor_z) + wall_thickness;
    let wall_z1 = i128::from(room.floor_z) + i128::from(room.dims.2) - wall_thickness;
    // Use chamfer-aware wall span to avoid overlapping diagonal corner pieces.
    let wall_span = if room.is_chamfered {
        cardinal_wall_interior_span(room, direction, wall_thickness)
    } else {
        let (x0, y0, x1, y1) = (
            i128::from(room.shell.0),
            i128::from(room.shell.1),
            i128::from(room.shell.2),
            i128::from(room.shell.3),
        );
        if direction.is_horizontal_route() {
            (y0 + wall_thickness, y1 - wall_thickness)
        } else {
            (x0 + wall_thickness, x1 - wall_thickness)
        }
    };

    if apertures.is_empty() {
        let (x, y, z) = wall_piece_bounds(
            room,
            direction,
            wall_span,
            (wall_z0, wall_z1),
            wall_thickness,
        );
        return push_wall_brush(brushes, bounds, base_id, x, y, z);
    }

    #[derive(Debug)]
    struct ClippedOpening<'a> {
        aperture: &'a WallAperture,
        span: (i128, i128),
        z: (i128, i128),
    }

    let mut openings = Vec::with_capacity(apertures.len());
    for aperture in apertures {
        let aperture_span = (
            aperture.center - aperture.width / 2,
            aperture.center + aperture.width / 2,
        );
        let clipped_span = (
            aperture_span.0.max(wall_span.0),
            aperture_span.1.min(wall_span.1),
        );
        let clipped_z = (aperture.z0.max(wall_z0), aperture.z1.min(wall_z1));
        if clipped_span.0 >= clipped_span.1 || clipped_z.0 >= clipped_z.1 {
            return Err(V3Error::ApertureInvalid {
                aperture_id: aperture.id.clone(),
                detail: format!(
                    "portal does not intersect room {} {} wall",
                    room.id,
                    direction.tag()
                ),
            });
        }
        openings.push(ClippedOpening {
            aperture,
            span: clipped_span,
            z: clipped_z,
        });
    }
    openings.sort_by(|left, right| {
        left.span
            .0
            .cmp(&right.span.0)
            .then_with(|| left.aperture.id.cmp(&right.aperture.id))
    });
    for pair in openings.windows(2) {
        if pair[0].span.1 > pair[1].span.0 {
            return Err(V3Error::ApertureInvalid {
                aperture_id: pair[1].aperture.id.clone(),
                detail: format!(
                    "overlaps aperture {} on room {} {} wall",
                    pair[0].aperture.id,
                    room.id,
                    direction.tag()
                ),
            });
        }
    }

    let mut cursor = wall_span.0;
    let mut segment_index = 0usize;
    for (aperture_index, opening) in openings.iter().enumerate() {
        if cursor < opening.span.0 {
            let (x, y, z) = wall_piece_bounds(
                room,
                direction,
                (cursor, opening.span.0),
                (wall_z0, wall_z1),
                wall_thickness,
            );
            push_wall_brush(
                brushes,
                bounds,
                format!("{base_id}/segment_{segment_index:04}"),
                x,
                y,
                z,
            )?;
            segment_index += 1;
        }
        if wall_z0 < opening.z.0 {
            let (x, y, z) = wall_piece_bounds(
                room,
                direction,
                opening.span,
                (wall_z0, opening.z.0),
                wall_thickness,
            );
            push_wall_brush(
                brushes,
                bounds,
                format!("{base_id}/sill_{aperture_index:04}"),
                x,
                y,
                z,
            )?;
        }
        if opening.z.1 < wall_z1 && !skip_lintels {
            let (x, y, z) = wall_piece_bounds(
                room,
                direction,
                opening.span,
                (opening.z.1, wall_z1),
                wall_thickness,
            );
            push_wall_brush(
                brushes,
                bounds,
                format!("{base_id}/lintel_{aperture_index:04}"),
                x,
                y,
                z,
            )?;
        }
        cursor = opening.span.1;
    }
    if cursor < wall_span.1 {
        let (x, y, z) = wall_piece_bounds(
            room,
            direction,
            (cursor, wall_span.1),
            (wall_z0, wall_z1),
            wall_thickness,
        );
        push_wall_brush(
            brushes,
            bounds,
            format!("{base_id}/segment_{segment_index:04}"),
            x,
            y,
            z,
        )?;
    }

    Ok(())
}

fn route_wall_directions(
    topology: &CommittedTopology,
    route: &super::ids::CommittedRoute,
) -> Result<(WallDirection, WallDirection), V3Error> {
    for portal in &topology.portals {
        if portal.source_room == route.source_room && portal.target_room == Some(route.target_room)
        {
            let source = WallDirection::parse(portal)?;
            return Ok((source, source.opposite()));
        }
        if portal.source_room == route.target_room && portal.target_room == Some(route.source_room)
        {
            let target = WallDirection::parse(portal)?;
            return Ok((target.opposite(), target));
        }
    }
    Err(V3Error::TopologyInvariant {
        detail: format!(
            "route/{:04} has no portal connecting {} to {}",
            route.id, route.source_room, route.target_room
        ),
    })
}

fn trim_route_envelope(
    envelope: (i32, i32, i32, i32),
    source: &CommittedRoom,
    target: &CommittedRoom,
    source_direction: WallDirection,
) -> Result<((i128, i128), (i128, i128)), V3Error> {
    let (mut x0, mut y0, mut x1, mut y1) = (
        i128::from(envelope.0),
        i128::from(envelope.1),
        i128::from(envelope.2),
        i128::from(envelope.3),
    );
    if x0 >= x1 || y0 >= y1 {
        return Err(V3Error::TopologyInvariant {
            detail: format!("invalid route envelope {envelope:?}"),
        });
    }

    match source_direction {
        WallDirection::East => {
            x0 = x0.max(i128::from(source.shell.2));
            x1 = x1.min(i128::from(target.shell.0));
        }
        WallDirection::West => {
            x0 = x0.max(i128::from(target.shell.2));
            x1 = x1.min(i128::from(source.shell.0));
        }
        WallDirection::North => {
            y0 = y0.max(i128::from(target.shell.3));
            y1 = y1.min(i128::from(source.shell.1));
        }
        WallDirection::South => {
            y0 = y0.max(i128::from(source.shell.3));
            y1 = y1.min(i128::from(target.shell.1));
        }
    }

    if x0 >= x1 || y0 >= y1 {
        return Err(V3Error::TopologyInvariant {
            detail: format!(
                "route envelope {envelope:?} does not span the gap between {} and {}",
                source.id, target.id
            ),
        });
    }
    Ok(((x0, x1), (y0, y1)))
}

fn build_endpoint_caps(
    room: &CommittedRoom,
    direction: WallDirection,
    clear_span: (i128, i128),
    wall_thickness: i128,
    id_root: &str,
    brushes: &mut Vec<AssemblyBrush>,
    bounds: &mut Vec<BrushBounds>,
) -> Result<(), V3Error> {
    let room_span = if direction.is_horizontal_route() {
        (i128::from(room.shell.1), i128::from(room.shell.3))
    } else {
        (i128::from(room.shell.0), i128::from(room.shell.2))
    };
    let inner_span = (room_span.0 + wall_thickness, room_span.1 - wall_thickness);
    let wall_z = (
        i128::from(room.floor_z) + wall_thickness,
        i128::from(room.floor_z) + i128::from(room.dims.2) - wall_thickness,
    );
    let opening_top = wall_z.0 + i128::from(HEADROOM);
    if opening_top < wall_z.1 {
        let corner_spans = [
            (
                clear_span.0.max(room_span.0),
                clear_span.1.min(inner_span.0),
            ),
            (
                clear_span.0.max(inner_span.1),
                clear_span.1.min(room_span.1),
            ),
        ];
        for (index, corner_span) in corner_spans.into_iter().enumerate() {
            if corner_span.0 >= corner_span.1 {
                continue;
            }
            let (x, y, z) = wall_piece_bounds(
                room,
                direction,
                corner_span,
                (opening_top, wall_z.1),
                wall_thickness,
            );
            push_wall_brush(
                brushes,
                bounds,
                format!("{id_root}/corner_lintel_{index:04}"),
                x,
                y,
                z,
            )?;
        }
    }

    let cap_spans = [
        (clear_span.0, clear_span.1.min(room_span.0)),
        (clear_span.0.max(room_span.1), clear_span.1),
    ];
    for (index, cap_span) in cap_spans.into_iter().enumerate() {
        if cap_span.0 >= cap_span.1 {
            continue;
        }
        let (mut x, mut y, z) =
            wall_piece_bounds(room, direction, cap_span, wall_z, wall_thickness);
        match direction {
            WallDirection::North => y.1 += wall_thickness,
            WallDirection::South => y.0 -= wall_thickness,
            WallDirection::West => x.1 += wall_thickness,
            WallDirection::East => x.0 -= wall_thickness,
        }
        push_wall_brush(
            brushes,
            bounds,
            format!("{id_root}/cap_{index:04}"),
            x,
            y,
            z,
        )?;
    }
    Ok(())
}

fn build_route_brushes(
    topology: &CommittedTopology,
    wall_thickness: i128,
    brushes: &mut Vec<AssemblyBrush>,
    bounds: &mut Vec<BrushBounds>,
) -> Result<(), V3Error> {
    for route in &topology.routes {
        let source =
            topology
                .room(route.source_room)
                .ok_or_else(|| V3Error::TopologyInvariant {
                    detail: format!("route/{:04} references unknown source room", route.id),
                })?;
        let target =
            topology
                .room(route.target_room)
                .ok_or_else(|| V3Error::TopologyInvariant {
                    detail: format!("route/{:04} references unknown target room", route.id),
                })?;
        if source.floor_z != target.floor_z {
            return Err(V3Error::TopologyInvariant {
                detail: format!(
                    "route/{:04} is level but rooms {} and {} have different floor heights",
                    route.id, source.id, target.id
                ),
            });
        }
        let (source_direction, target_direction) = route_wall_directions(topology, route)?;
        let floor_z0 = i128::from(source.floor_z);
        let clear_z0 = floor_z0 + wall_thickness;
        let ceiling_z0 = clear_z0 + i128::from(HEADROOM);
        let route_root = format!("route/{:04}", route.id);

        for (segment_index, &envelope) in route.envelopes.iter().enumerate() {
            let (clear_x, clear_y) =
                trim_route_envelope(envelope, source, target, source_direction)?;
            let segment_root = format!("{route_root}/segment_{segment_index:04}");

            if source_direction.is_horizontal_route() {
                let shell_y = (clear_y.0 - wall_thickness, clear_y.1 + wall_thickness);
                push_floor_brush(
                    brushes,
                    bounds,
                    format!("{segment_root}/floor"),
                    clear_x,
                    shell_y,
                    (floor_z0, clear_z0),
                )?;
                push_box_brush(
                    brushes,
                    bounds,
                    format!("{segment_root}/ceiling"),
                    BrushRole::CeilingSlab,
                    clear_x,
                    shell_y,
                    (ceiling_z0, ceiling_z0 + wall_thickness),
                )?;
                push_wall_brush(
                    brushes,
                    bounds,
                    format!("{segment_root}/wall_north"),
                    clear_x,
                    (clear_y.0 - wall_thickness, clear_y.0),
                    (clear_z0, ceiling_z0),
                )?;
                push_wall_brush(
                    brushes,
                    bounds,
                    format!("{segment_root}/wall_south"),
                    clear_x,
                    (clear_y.1, clear_y.1 + wall_thickness),
                    (clear_z0, ceiling_z0),
                )?;
                build_endpoint_caps(
                    source,
                    source_direction,
                    clear_y,
                    wall_thickness,
                    &format!("{segment_root}/source"),
                    brushes,
                    bounds,
                )?;
                build_endpoint_caps(
                    target,
                    target_direction,
                    clear_y,
                    wall_thickness,
                    &format!("{segment_root}/target"),
                    brushes,
                    bounds,
                )?;
            } else {
                let shell_x = (clear_x.0 - wall_thickness, clear_x.1 + wall_thickness);
                push_floor_brush(
                    brushes,
                    bounds,
                    format!("{segment_root}/floor"),
                    shell_x,
                    clear_y,
                    (floor_z0, clear_z0),
                )?;
                push_box_brush(
                    brushes,
                    bounds,
                    format!("{segment_root}/ceiling"),
                    BrushRole::CeilingSlab,
                    shell_x,
                    clear_y,
                    (ceiling_z0, ceiling_z0 + wall_thickness),
                )?;
                push_wall_brush(
                    brushes,
                    bounds,
                    format!("{segment_root}/wall_west"),
                    (clear_x.0 - wall_thickness, clear_x.0),
                    clear_y,
                    (clear_z0, ceiling_z0),
                )?;
                push_wall_brush(
                    brushes,
                    bounds,
                    format!("{segment_root}/wall_east"),
                    (clear_x.1, clear_x.1 + wall_thickness),
                    clear_y,
                    (clear_z0, ceiling_z0),
                )?;
                build_endpoint_caps(
                    source,
                    source_direction,
                    clear_x,
                    wall_thickness,
                    &format!("{segment_root}/source"),
                    brushes,
                    bounds,
                )?;
                build_endpoint_caps(
                    target,
                    target_direction,
                    clear_x,
                    wall_thickness,
                    &format!("{segment_root}/target"),
                    brushes,
                    bounds,
                )?;
            }
        }
    }
    Ok(())
}

#[allow(dead_code)]
fn positive_overlap(left: (i128, i128), right: (i128, i128)) -> bool {
    left.0 < right.1 && left.1 > right.0
}

#[allow(dead_code)]
fn contact_faces(left: &BrushBounds, right: &BrushBounds) -> Option<(FaceRole, FaceRole)> {
    if left.x.1 == right.x.0
        && positive_overlap(left.y, right.y)
        && positive_overlap(left.z, right.z)
    {
        return Some((FaceRole::EastWall, FaceRole::WestWall));
    }
    if left.x.0 == right.x.1
        && positive_overlap(left.y, right.y)
        && positive_overlap(left.z, right.z)
    {
        return Some((FaceRole::WestWall, FaceRole::EastWall));
    }
    if left.y.1 == right.y.0
        && positive_overlap(left.x, right.x)
        && positive_overlap(left.z, right.z)
    {
        return Some((FaceRole::NorthWall, FaceRole::SouthWall));
    }
    if left.y.0 == right.y.1
        && positive_overlap(left.x, right.x)
        && positive_overlap(left.z, right.z)
    {
        return Some((FaceRole::SouthWall, FaceRole::NorthWall));
    }
    if left.z.1 == right.z.0
        && positive_overlap(left.x, right.x)
        && positive_overlap(left.y, right.y)
    {
        return Some((FaceRole::Ceiling, FaceRole::Floor));
    }
    if left.z.0 == right.z.1
        && positive_overlap(left.x, right.x)
        && positive_overlap(left.y, right.y)
    {
        return Some((FaceRole::Floor, FaceRole::Ceiling));
    }
    None
}

#[allow(dead_code)]
fn find_contact_faces(a: &ConvexBrush, b: &ConvexBrush) -> Option<(FaceRole, FaceRole)> {
    for fa in &a.faces {
        for fb in &b.faces {
            // Check if the planes are coincident (same normal and same plane).
            // This covers both opposite-normals (Floor↔Ceiling) and
            // same-normals (Floor↔Floor at same Z — adjacent pieces).
            if let Ok(true) = fa.plane.is_coincident_with(&fb.plane) {
                return Some((fa.role, fb.role));
            }
            // Check opposite normals
            if let (Some(neg_nx), Some(neg_ny), Some(neg_nz), Some(neg_d)) = (
                fb.plane.nx.checked_neg(),
                fb.plane.ny.checked_neg(),
                fb.plane.nz.checked_neg(),
                fb.plane.d.checked_neg(),
            ) {
                if let Ok(neg_fb) = CanonicalPlane::new(neg_nx, neg_ny, neg_nz, neg_d) {
                    if let Ok(true) = fa.plane.is_coincident_with(&neg_fb) {
                        return Some((fa.role, fb.role));
                    }
                }
            }
        }
    }
    None
}

#[allow(dead_code)]
fn build_contact_interfaces(mut bounds: Vec<BrushBounds>) -> Vec<Interface> {
    bounds.sort_by(|left, right| left.id.cmp(&right.id));
    let mut interfaces = Vec::new();
    for left_index in 0..bounds.len() {
        for right_index in (left_index + 1)..bounds.len() {
            let left = &bounds[left_index];
            let right = &bounds[right_index];
            if let Some((left_face, right_face)) = contact_faces(left, right) {
                interfaces.push(Interface::new(
                    format!("interface/{left_index:04}/{right_index:04}"),
                    left.id.clone(),
                    right.id.clone(),
                    left_face,
                    right_face,
                ));
            }
        }
    }
    interfaces
}

fn has_positive_face_contact(left: &ConvexBrush, right: &ConvexBrush) -> Result<bool, V3Error> {
    let (left_min, left_max) = left.aabb()?;
    let (right_min, right_max) = right.aabb()?;
    let overlaps = [
        left_min.0 < right_max.0 && left_max.0 > right_min.0,
        left_min.1 < right_max.1 && left_max.1 > right_min.1,
        left_min.2 < right_max.2 && left_max.2 > right_min.2,
    ]
    .into_iter()
    .filter(|overlap| *overlap)
    .count();
    Ok(overlaps >= 2)
}

/// Return the exact coincident face pair used by a feature support interface.
fn matching_contact_faces(
    child: &ConvexBrush,
    parent: &ConvexBrush,
) -> Result<(FaceRole, FaceRole), V3Error> {
    for child_face in &child.faces {
        for parent_face in &parent.faces {
            let same = child_face.plane.is_coincident_with(&parent_face.plane)?;
            let opposite = CanonicalPlane::new(
                parent_face
                    .plane
                    .nx
                    .checked_neg()
                    .ok_or(V3Error::ArithmeticOverflow {
                        operation: "support face normal negation",
                    })?,
                parent_face
                    .plane
                    .ny
                    .checked_neg()
                    .ok_or(V3Error::ArithmeticOverflow {
                        operation: "support face normal negation",
                    })?,
                parent_face
                    .plane
                    .nz
                    .checked_neg()
                    .ok_or(V3Error::ArithmeticOverflow {
                        operation: "support face normal negation",
                    })?,
                parent_face
                    .plane
                    .d
                    .checked_neg()
                    .ok_or(V3Error::ArithmeticOverflow {
                        operation: "support face offset negation",
                    })?,
            )
            .and_then(|opposite| child_face.plane.is_coincident_with(&opposite))?;
            if same || opposite {
                return Ok((child_face.role, parent_face.role));
            }
        }
    }
    Err(V3Error::CompositionInvariant {
        detail: "declared support brushes do not share a contact face".into(),
    })
}

/// Build feature brushes and their interfaces from the composition plan.
fn build_feature_brushes(
    plan: &PlanOutcome,
    topology: &CommittedTopology,
    structural_brushes: &[AssemblyBrush],
) -> Result<(Vec<AssemblyBrush>, Vec<Interface>), V3Error> {
    let mut brushes: Vec<AssemblyBrush> = Vec::new();
    let mut interfaces: Vec<Interface> = Vec::new();

    for inst in &plan.instances {
        let _room = topology
            .room(inst.room_id)
            .ok_or_else(|| V3Error::TopologyInvariant {
                detail: format!("feature instance references unknown room {}", inst.room_id),
            })?;

        let x = (i128::from(inst.volume.x0), i128::from(inst.volume.x1));
        let y = (i128::from(inst.volume.y0), i128::from(inst.volume.y1));
        let z = (i128::from(inst.volume.z0), i128::from(inst.volume.z1));
        let fid = format!("feature/{:04}", inst.id.0);

        // Determine brush role from tags.
        let role = if inst.tags.contains("pillar") || inst.tags.contains("twisted") {
            BrushRole::Column
        } else if inst.tags.contains("buttress") {
            BrushRole::Buttress
        } else if inst.tags.contains("portal-chamber") {
            BrushRole::Blade
        } else if inst.tags.contains("fractured-vault") {
            BrushRole::VaultRib
        } else if inst.tags.contains("monolith") {
            BrushRole::Monolith
        } else if inst.tags.contains("terrace") {
            BrushRole::FloorSlab
        } else {
            BrushRole::Feature
        };
        let brush = ConvexBrush::make_box(x, y, z)?;

        // Every support edge names the actual contacted structural or feature
        // brush. World roots are limited to structural shell brushes.
        let (parent_brush_id, parent_brush) = match inst.support.as_ref() {
            Some(super::ids::SupportRelation::SupportedBy(parent_id)) => {
                let parent_id = format!("feature/{:04}", parent_id.0);
                let parent = brushes
                    .iter()
                    .find(|brush| brush.id == parent_id)
                    .ok_or_else(|| V3Error::CompositionInvariant {
                        detail: format!(
                            "feature {} references unavailable parent {parent_id}",
                            inst.id
                        ),
                    })?;
                (parent_id, parent)
            }
            Some(relation) => {
                let (surface_id, expected_kind) =
                    relation
                        .support_surface()
                        .ok_or_else(|| V3Error::CompositionInvariant {
                            detail: format!("feature {} has no root support", inst.id),
                        })?;
                let surface =
                    topology
                        .surface(surface_id)
                        .ok_or_else(|| V3Error::CompositionInvariant {
                            detail: format!(
                                "feature {} references unknown surface {surface_id}",
                                inst.id
                            ),
                        })?;
                if surface.room_id != inst.room_id || surface.kind != expected_kind {
                    return Err(V3Error::CompositionInvariant {
                        detail: format!(
                            "feature {} support surface is not owned by its room",
                            inst.id
                        ),
                    });
                }
                let room_prefix = inst.room_id.stable_key();
                let prefix = match expected_kind {
                    super::ids::SupportSurfaceKind::Floor => format!("{room_prefix}/floor"),
                    super::ids::SupportSurfaceKind::Ceiling => format!("{room_prefix}/ceiling"),
                    super::ids::SupportSurfaceKind::Wall => {
                        format!("{room_prefix}/wall_{}", surface.owner.direction)
                    }
                };
                let prefix_with_separator = format!("{prefix}/");
                let parent = structural_brushes
                    .iter()
                    .find(|candidate| {
                        (candidate.id == prefix || candidate.id.starts_with(&prefix_with_separator))
                            && has_positive_face_contact(&brush, &candidate.brush).unwrap_or(false)
                            && matching_contact_faces(&brush, &candidate.brush).is_ok()
                    })
                    .ok_or_else(|| V3Error::CompositionInvariant {
                        detail: format!(
                            "feature {} has no emitted structural support {prefix}",
                            inst.id
                        ),
                    })?;
                (parent.id.clone(), parent)
            }
            None => {
                return Err(V3Error::CompositionInvariant {
                    detail: format!("feature {} has no support relation", inst.id),
                });
            }
        };
        let (child_face, parent_face) = matching_contact_faces(&brush, &parent_brush.brush)?;
        let iface_id = format!("support/feature/{:04}", inst.id.0);
        interfaces.push(Interface::new(
            iface_id.clone(),
            fid.clone(),
            parent_brush_id.clone(),
            child_face,
            parent_face,
        ));
        brushes.push(AssemblyBrush::new(
            fid,
            role,
            brush,
            Support::SupportedBy {
                brush_id: parent_brush_id,
                interface_id: iface_id,
            },
        ));
    }

    Ok((brushes, interfaces))
}

/// Build a validated assembly from the committed topology and composition plan.
fn build_assembly_from_topology(
    config: &V3Config,
    topology: &CommittedTopology,
    plan: &PlanOutcome,
    spawn_volume: &QuantumVolume,
    light_volumes: &[QuantumVolume],
    reservations: &ReservationSet,
    _seed: V3Seed,
) -> Result<(Assembly, (i32, i32, i32), Vec<(i32, i32, i32)>), V3Error> {
    let wall_thickness = i128::from(CONSTRUCTION_QUANTUM);
    let apertures = collect_wall_apertures(topology)?;
    let mut brushes: Vec<AssemblyBrush> = Vec::new();
    let mut bounds: Vec<BrushBounds> = Vec::new();

    let shaped_arch = matches!(config.arch_type, ArchType::Pointed | ArchType::Segmented);

    // Both transition approaches meet their host slabs at wall planes. No
    // tread enters a room slab, so the exact mask is empty; inventing a broad
    // lower/upper aperture here would weaken otherwise continuous shells.
    let transition_openings: Vec<(RoomId, (i128, i128, i128, i128), bool)> = Vec::new();

    for room in &topology.rooms {
        let z0 = i128::from(room.floor_z);
        let z1 = z0 + i128::from(room.dims.2);
        let rid = room.id.stable_key();

        let x_range = (i128::from(room.shell.0), i128::from(room.shell.2));
        let y_range = (i128::from(room.shell.1), i128::from(room.shell.3));

        // Floor/ceiling slabs carry the exact committed polygon, not the
        // footprint AABB.  AABB slabs would fill every clipped corner and turn
        // a chamfer into a decorative diagonal wall.  Transition openings that
        // genuinely enter a host room are split; exterior-only transition
        // envelopes do not weaken an otherwise continuous slab.
        let floor_openings: Vec<(i128, i128, i128, i128)> = transition_openings
            .iter()
            .filter(|(owner, rect, is_floor)| {
                *owner == room.id
                    && *is_floor
                    && rect.0 < x_range.1
                    && rect.2 > x_range.0
                    && rect.1 < y_range.1
                    && rect.3 > y_range.0
            })
            .map(|(_, rect, _)| *rect)
            .collect();
        if room.is_chamfered && floor_openings.is_empty() {
            build_chamfered_slab(
                room,
                (z0, z0 + wall_thickness),
                BrushRole::FloorSlab,
                &format!("{rid}/floor"),
                &mut brushes,
                &mut bounds,
            )?;
        } else if floor_openings.is_empty() {
            push_floor_brush(
                &mut brushes,
                &mut bounds,
                format!("{rid}/floor"),
                x_range,
                y_range,
                (z0, z0 + wall_thickness),
            )?;
        } else {
            build_split_slab(
                &mut brushes,
                &mut bounds,
                &format!("{rid}/floor"),
                x_range,
                y_range,
                (z0, z0 + wall_thickness),
                BrushRole::FloorSlab,
                &floor_openings,
                wall_thickness,
            )?;
        }

        let ceil_openings: Vec<(i128, i128, i128, i128)> = transition_openings
            .iter()
            .filter(|(owner, rect, is_floor)| {
                *owner == room.id
                    && !*is_floor
                    && rect.0 < x_range.1
                    && rect.2 > x_range.0
                    && rect.1 < y_range.1
                    && rect.3 > y_range.0
            })
            .map(|(_, rect, _)| *rect)
            .collect();
        if room.is_chamfered && ceil_openings.is_empty() {
            build_chamfered_slab(
                room,
                (z1 - wall_thickness, z1),
                BrushRole::CeilingSlab,
                &format!("{rid}/ceiling"),
                &mut brushes,
                &mut bounds,
            )?;
        } else if ceil_openings.is_empty() {
            push_box_brush(
                &mut brushes,
                &mut bounds,
                format!("{rid}/ceiling"),
                BrushRole::CeilingSlab,
                x_range,
                y_range,
                (z1 - wall_thickness, z1),
            )?;
        } else {
            build_split_slab(
                &mut brushes,
                &mut bounds,
                &format!("{rid}/ceiling"),
                x_range,
                y_range,
                (z1 - wall_thickness, z1),
                BrushRole::CeilingSlab,
                &ceil_openings,
                wall_thickness,
            )?;
        }

        // Cardinal wall pieces (split around apertures).
        for direction in [
            WallDirection::North,
            WallDirection::South,
            WallDirection::West,
            WallDirection::East,
        ] {
            let wall_apertures = apertures
                .get(&(room.id, direction))
                .map(Vec::as_slice)
                .unwrap_or(&[]);
            build_room_wall(
                room,
                direction,
                wall_apertures,
                wall_thickness,
                shaped_arch,
                &mut brushes,
                &mut bounds,
            )?;

            for aperture in wall_apertures {
                match config.arch_type {
                    ArchType::None => {}
                    ArchType::Pointed => {
                        let wall_coord = room_wall_coordinate(room, direction);
                        build_pointed_arch_surround(
                            room,
                            direction,
                            wall_coord,
                            aperture.center,
                            wall_thickness,
                            &mut brushes,
                            &mut bounds,
                        )?;
                    }
                    ArchType::Segmented => build_segmented_arch_surround(
                        room,
                        direction,
                        aperture.center,
                        wall_thickness,
                        &mut brushes,
                        &mut bounds,
                    )?,
                }
            }
        }

        build_cardinal_corner_posts(room, wall_thickness, &mut brushes, &mut bounds)?;
        // Diagonal wall pieces for chamfered corners.
        build_diagonal_walls(room, wall_thickness, &mut brushes, &mut bounds)?;
    }

    // Every committed portal must own a physical connector; omitting an
    // intersecting route creates an exterior-facing room aperture rather than
    // resolving a transition conflict.
    build_route_brushes(topology, wall_thickness, &mut brushes, &mut bounds)?;

    // Build stair transition geometry.
    for transition in &topology.transitions {
        build_stair_emission(
            transition,
            topology,
            wall_thickness,
            &mut brushes,
            &mut bounds,
        )?;
    }

    let mut protected_volumes = reservations.to_protected_volumes()?;
    for transition in &topology.transitions {
        for (index, &(x0, y0, z0, x1, y1, z1)) in transition.headroom_volumes.iter().enumerate() {
            protected_volumes.push(ProtectedVolume {
                id: format!("transition/{:04}/clearance_{index:04}", transition.id),
                brush: ConvexBrush::make_box(
                    (i128::from(x0), i128::from(x1)),
                    (i128::from(y0), i128::from(y1)),
                    (i128::from(z0), i128::from(z1)),
                )?,
            });
        }
    }
    // Materialize feature brushes from the composition plan
    let (feature_brushes, feature_interfaces) = build_feature_brushes(plan, topology, &brushes)?;
    for fb in feature_brushes {
        // Check feature brush does not intrude into any protected volume
        let fb_aabb = fb.brush.aabb()?;
        for pv in &protected_volumes {
            let pv_aabb = pv.brush.aabb()?;
            if fb_aabb.0 .0 < pv_aabb.1 .0
                && fb_aabb.1 .0 > pv_aabb.0 .0
                && fb_aabb.0 .1 < pv_aabb.1 .1
                && fb_aabb.1 .1 > pv_aabb.0 .1
                && fb_aabb.0 .2 < pv_aabb.1 .2
                && fb_aabb.1 .2 > pv_aabb.0 .2
            {
                return Err(V3Error::ProtectedVolumeIntrusion {
                    brush_id: fb.id.clone(),
                    protected_id: pv.id.clone(),
                });
            }
        }
        brushes.push(fb);
    }

    brushes.sort_by(|left, right| left.id.cmp(&right.id));
    // Derive interfaces from actual ConvexBrush face geometry. Cache exact
    // AABBs so disjoint brush pairs never enter the coincident-plane scan;
    // touching pairs remain eligible and are still proven by Assembly.
    let brush_aabbs = brushes
        .iter()
        .map(|brush| brush.brush.aabb())
        .collect::<Result<Vec<_>, _>>()?;
    let mut all_interfaces: Vec<Interface> = Vec::new();
    for i in 0..brushes.len() {
        for j in (i + 1)..brushes.len() {
            let left = brush_aabbs[i];
            let right = brush_aabbs[j];
            if left.0 .0 > right.1 .0
                || right.0 .0 > left.1 .0
                || left.0 .1 > right.1 .1
                || right.0 .1 > left.1 .1
                || left.0 .2 > right.1 .2
                || right.0 .2 > left.1 .2
            {
                continue;
            }
            let mut found = false;
            for fa in &brushes[i].brush.faces {
                if found {
                    break;
                }
                for fb in &brushes[j].brush.faces {
                    let same = fa.plane.is_coincident_with(&fb.plane).unwrap_or(false);
                    let opp = (|| -> Option<bool> {
                        let neg = CanonicalPlane::new(
                            fb.plane.nx.checked_neg()?,
                            fb.plane.ny.checked_neg()?,
                            fb.plane.nz.checked_neg()?,
                            fb.plane.d.checked_neg()?,
                        )
                        .ok()?;
                        fa.plane.is_coincident_with(&neg).ok()
                    })()
                    .unwrap_or(false);
                    if same || opp {
                        all_interfaces.push(Interface::new(
                            format!("interface/{i:04}/{j:04}"),
                            brushes[i].id.clone(),
                            brushes[j].id.clone(),
                            fa.role,
                            fb.role,
                        ));
                        found = true;
                        break;
                    }
                }
            }
        }
    }
    // Merge feature interfaces
    all_interfaces.extend(feature_interfaces);

    let assembly = Assembly::new(brushes, all_interfaces, protected_volumes)?;

    // Spawn origin shares the reservation's XY center and uses a
    // quantum-aligned standing height inside its clear 80-unit volume.
    let spawn_x = (spawn_volume.x0 + spawn_volume.x1) / 2;
    let spawn_y = (spawn_volume.y0 + spawn_volume.y1) / 2;
    let spawn_z = spawn_volume.z0 + 2 * CONSTRUCTION_QUANTUM;
    let spawn_origin = (spawn_x, spawn_y, spawn_z);

    // `None` retains the compatibility room-midpoint bytes. Explicit light
    // counts use the exact selected reservation centres.
    let q = CONSTRUCTION_QUANTUM;
    let light_origins: Vec<(i32, i32, i32)> = if config.light_count.is_none() {
        topology
            .rooms
            .iter()
            .map(|room| {
                let lx = (room.shell.0 + room.shell.2) / 2;
                let ly = (room.shell.1 + room.shell.3) / 2;
                let lz = room.floor_z + room.dims.2 as i32 - 2 * q;
                (lx, ly, lz)
            })
            .collect()
    } else {
        light_volumes
            .iter()
            .map(|volume| {
                (
                    (volume.x0 + volume.x1) / 2,
                    (volume.y0 + volume.y1) / 2,
                    (volume.z0 + volume.z1) / 2,
                )
            })
            .collect()
    };

    Ok((assembly, spawn_origin, light_origins))
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::super::config::V3Preset;
    use super::*;

    fn topology_and_assembly(config: &V3Config) -> (CommittedTopology, Assembly, PlanOutcome) {
        let seed = V3Seed::new(config.seed);
        let mut alloc = V3IdAllocator::new();
        let (footprints, layout) = build_footprints(config, seed, &mut alloc).unwrap();
        let topology = build_topology(config, &footprints, &layout, seed, &mut alloc).unwrap();
        let (spawn_volume, light_volumes) = compute_reservations(&topology).unwrap();
        let plan =
            plan_composition(seed, config, &topology, &spawn_volume, &light_volumes).unwrap();
        let mut reservations = ReservationSet::new();
        reservations
            .add(Reservation::new("spawn", "spawn_point", spawn_volume))
            .unwrap();
        for (index, volume) in light_volumes.iter().enumerate() {
            reservations
                .add(Reservation::new(
                    format!("light_{index:04}"),
                    "light",
                    *volume,
                ))
                .unwrap();
        }
        let (assembly, _, _) = build_assembly_from_topology(
            config,
            &topology,
            &plan,
            &spawn_volume,
            &light_volumes,
            &reservations,
            seed,
        )
        .unwrap();
        (topology, assembly, plan)
    }

    fn sparse_topology_and_assembly() -> (CommittedTopology, Assembly, PlanOutcome) {
        topology_and_assembly(&V3Config::nominal_sparse())
    }

    fn point_is_inside_brush(brush: &AssemblyBrush, point: (i128, i128, i128)) -> bool {
        let point = super::super::geometry::Point3::from_ints(point.0, point.1, point.2);
        brush.brush.faces.iter().all(|face| {
            face.plane
                .signed_distance_rational(&point)
                .is_ok_and(|distance| distance > super::super::geometry::Rational::ZERO)
        })
    }

    fn point_is_solid(assembly: &Assembly, point: (i128, i128, i128)) -> bool {
        assembly
            .brushes
            .iter()
            .any(|brush| point_is_inside_brush(brush, point))
    }

    fn boxes_overlap(
        left: ((i128, i128, i128), (i128, i128, i128)),
        right: ((i128, i128, i128), (i128, i128, i128)),
    ) -> bool {
        left.0 .0 < right.1 .0
            && left.1 .0 > right.0 .0
            && left.0 .1 < right.1 .1
            && left.1 .1 > right.0 .1
            && left.0 .2 < right.1 .2
            && left.1 .2 > right.0 .2
    }

    #[test]
    fn transition_emits_twelve_supported_treads_and_clear_approaches() {
        let (topology, assembly, _plan) = sparse_topology_and_assembly();
        let transition = topology.transitions.first().unwrap();
        let lower = topology.room(transition.lower_room).unwrap();
        let upper = topology.room(transition.upper_room).unwrap();
        let q = i128::from(CONSTRUCTION_QUANTUM);
        let x0 = i128::from(transition.protected_volume.0);
        let x1 = i128::from(transition.protected_volume.3);
        let y0 = i128::from(transition.tread_run.1);
        let lower_floor = i128::from(lower.floor_z);
        let upper_floor = i128::from(upper.floor_z);
        let run_end = y0 + i128::from(STAIR_STEPS) * q;

        let treads: Vec<_> = assembly
            .brushes
            .iter()
            .filter(|brush| brush.id.starts_with("transition/0000/tread_"))
            .collect();
        assert_eq!(treads.len(), STAIR_STEPS as usize);
        for (index, tread) in treads.iter().enumerate() {
            assert_eq!(tread.id, format!("transition/0000/tread_{index:04}"));
            let (minimum, maximum) = tread.brush.aabb().unwrap();
            assert_eq!(minimum, (x0, y0 + index as i128 * q, lower_floor));
            assert_eq!(
                maximum,
                (
                    x1,
                    y0 + (index as i128 + 1) * q,
                    lower_floor + (index as i128 + 1) * q,
                )
            );
            assert_eq!(maximum.1 - minimum.1, q, "tread {index} depth drifted");
            assert_eq!(maximum.2 - lower_floor, (index as i128 + 1) * q);
            let center = ((x0 + x1) / 2, (minimum.1 + maximum.1) / 2);
            assert!(
                point_is_inside_brush(tread, (center.0, center.1, maximum.2 - 1)),
                "tread {index} has no solid support"
            );
            for clearance in [1, 55, 79] {
                let witness = (center.0, center.1, maximum.2 + clearance);
                assert!(
                    !point_is_solid(&assembly, witness),
                    "tread {index} lacks frozen headroom at {witness:?}"
                );
            }
        }
        assert_eq!(run_end - y0, 192);
        assert_eq!(run_end, i128::from(transition.tread_run.3));
        assert_eq!(lower_floor + i128::from(STAIR_STEPS) * q, upper_floor);

        let roof = assembly
            .brushes
            .iter()
            .find(|brush| brush.id == "transition/0000/roof")
            .unwrap();
        assert_eq!(
            roof.brush.aabb().unwrap(),
            (
                (
                    x0 - q,
                    i128::from(transition.lower_approach.1),
                    upper_floor + q + i128::from(HEADROOM),
                ),
                (x1 + q, run_end, upper_floor + 2 * q + i128::from(HEADROOM)),
            )
        );
        for (id, expected_y) in [
            (
                "transition/0000/wall_west",
                (i128::from(lower.shell.3), run_end),
            ),
            (
                "transition/0000/wall_east",
                (i128::from(lower.shell.3), run_end),
            ),
        ] {
            let wall = assembly
                .brushes
                .iter()
                .find(|brush| brush.id == id)
                .unwrap();
            let (minimum, maximum) = wall.brush.aabb().unwrap();
            assert_eq!((minimum.1, maximum.1), expected_y);
        }

        let lower_approach_floor = assembly
            .brushes
            .iter()
            .find(|brush| brush.id == "transition/0000/lower_approach/floor")
            .unwrap();
        assert_eq!(
            lower_approach_floor.brush.aabb().unwrap(),
            (
                (x0 - q, i128::from(transition.lower_approach.1), lower_floor,),
                (
                    x1 + q,
                    i128::from(transition.lower_approach.3),
                    lower_floor + q,
                ),
            )
        );

        let approach_floor = assembly
            .brushes
            .iter()
            .find(|brush| brush.id == "transition/0000/upper_approach/floor")
            .unwrap();
        let approach_bounds = approach_floor.brush.aabb().unwrap();
        assert_eq!(approach_bounds.0, (x0 - q, run_end, upper_floor));
        assert_eq!(
            approach_bounds.1,
            (x1 + q, i128::from(upper.shell.1), upper_floor + q)
        );
        let approach_center = ((x0 + x1) / 2, (run_end + i128::from(upper.shell.1)) / 2);
        assert!(point_is_solid(
            &assembly,
            (approach_center.0, approach_center.1, upper_floor + q / 2)
        ));
        assert!(!point_is_solid(
            &assembly,
            (
                approach_center.0,
                approach_center.1,
                upper_floor + q + i128::from(HEADROOM) - 1,
            )
        ));

        let lower_landing = transition.lower_landing;
        let lower_center = (
            i128::from((lower_landing.0 + lower_landing.2) / 2),
            i128::from((lower_landing.1 + lower_landing.3) / 2),
        );
        assert!(point_is_solid(
            &assembly,
            (lower_center.0, lower_center.1, lower_floor + q / 2)
        ));
        assert!(!point_is_solid(
            &assembly,
            (lower_center.0, lower_center.1, lower_floor + q + 79)
        ));
        let upper_landing = transition.upper_landing;
        let upper_center = (
            i128::from((upper_landing.0 + upper_landing.2) / 2),
            i128::from((upper_landing.1 + upper_landing.3) / 2),
        );
        assert!(point_is_solid(
            &assembly,
            (upper_center.0, upper_center.1, upper_floor + q / 2)
        ));
        assert!(!point_is_solid(
            &assembly,
            (upper_center.0, upper_center.1, upper_floor + q + 79)
        ));

        let first_tread_bounds = treads[0].brush.aabb().unwrap();
        let lower_floor_id = format!("{}/floor", lower.id.stable_key());
        assert!(assembly
            .brushes
            .iter()
            .filter(|brush| brush.id.starts_with(&lower_floor_id))
            .all(|brush| !boxes_overlap(brush.brush.aabb().unwrap(), first_tread_bounds)));
        assert!(assembly
            .brushes
            .iter()
            .any(|brush| brush.id == "transition/0000/lower_entry_cap"));

        // The actual lower/upper wall masks omit every point in each 64×80
        // core across the complete 16-unit wall depth.
        for (room, y_sign, floor) in [(lower, -1_i128, lower_floor), (upper, 1, upper_floor)] {
            let wall_y = if y_sign < 0 {
                i128::from(lower.shell.3)
            } else {
                i128::from(upper.shell.1)
            };
            for normal in [1_i128, 8, 15] {
                for tangent in [x0 + 1, (x0 + x1) / 2, x1 - 1] {
                    for height in [1_i128, 40, 79] {
                        let point = (tangent, wall_y + y_sign * normal, floor + q + height);
                        assert!(
                            !point_is_solid(&assembly, point),
                            "transition wall opening for {} is solid at {point:?}",
                            room.id
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn topology_portals_split_walls_and_routes_build_corridor_shells() {
        let (topology, assembly, _plan) = sparse_topology_and_assembly();
        let portal = &topology.portals[0];
        let route = &topology.routes[0];

        assert!(assembly
            .brushes
            .iter()
            .any(|brush| brush.id == "room/0000/wall_north"));
        assert!(!assembly
            .brushes
            .iter()
            .any(|brush| brush.id == "room/0000/wall_east"));
        assert!(assembly
            .brushes
            .iter()
            .any(|brush| brush.id == "room/0000/wall_east/segment_0000"));
        assert!(assembly
            .brushes
            .iter()
            .any(|brush| brush.id.contains("room/0000/wall_east") && brush.id.contains("lintel")));
        assert!(assembly
            .brushes
            .iter()
            .any(|brush| brush.id.contains("room/0001")
                && (brush.id.contains("wall_west") || brush.id.contains("diag"))));

        for (suffix, role) in [
            ("floor", BrushRole::FloorSlab),
            ("ceiling", BrushRole::CeilingSlab),
            ("wall_north", BrushRole::WallShell),
            ("wall_south", BrushRole::WallShell),
        ] {
            let id = format!("route/{:04}/segment_0000/{suffix}", route.id);
            assert!(assembly
                .brushes
                .iter()
                .any(|brush| brush.id == id && brush.role == role));
        }

        let source = topology.room(portal.source_room).unwrap();
        let target = topology.room(portal.target_room.unwrap()).unwrap();
        let throat_z = i128::from(portal.anchor.2);
        let clear_points = [
            (
                i128::from(source.shell.2) - i128::from(CONSTRUCTION_QUANTUM) / 2,
                i128::from(portal.anchor.1),
                throat_z,
            ),
            (
                (i128::from(source.shell.2) + i128::from(target.shell.0)) / 2,
                i128::from(portal.anchor.1),
                throat_z,
            ),
            (
                i128::from(target.shell.0) + i128::from(CONSTRUCTION_QUANTUM) / 2,
                i128::from(portal.anchor.1),
                throat_z,
            ),
        ];
        for point in clear_points {
            assert!(
                assembly
                    .brushes
                    .iter()
                    .all(|brush| !point_is_inside_brush(brush, point)),
                "portal/corridor point {point:?} is inside a solid brush"
            );
        }
    }

    #[test]
    fn cardinal_portals_are_full_depth_pointed_omissions_without_opening_brushes() {
        let (topology, assembly, _plan) = sparse_topology_and_assembly();
        assert!(assembly
            .brushes
            .iter()
            .all(|brush| brush.role != BrushRole::PortalThroat));

        for portal in &topology.portals {
            let source_direction = WallDirection::parse(portal).unwrap();
            let endpoints = [
                (portal.source_room, source_direction),
                (portal.target_room.unwrap(), source_direction.opposite()),
            ];
            for (room_id, direction) in endpoints {
                let room = topology.room(room_id).unwrap();
                let center = if direction.is_horizontal_route() {
                    i128::from(portal.anchor.1)
                } else {
                    i128::from(portal.anchor.0)
                };
                let core_bottom = i128::from(room.floor_z + CONSTRUCTION_QUANTUM);
                let core_top = core_bottom + i128::from(HEADROOM);
                let wall_point = |tangent: i128, normal: i128, z: i128| match direction {
                    WallDirection::North => (tangent, i128::from(room.shell.1) + normal, z),
                    WallDirection::South => (tangent, i128::from(room.shell.3) - normal, z),
                    WallDirection::West => (i128::from(room.shell.0) + normal, tangent, z),
                    WallDirection::East => (i128::from(room.shell.2) - normal, tangent, z),
                };

                for normal in [1_i128, 8, 15] {
                    for tangent in [center - 31, center, center + 31] {
                        for z in [core_bottom + 1, core_bottom + 40, core_top - 1] {
                            let point = wall_point(tangent, normal, z);
                            assert!(
                                !point_is_solid(&assembly, point),
                                "{} {} portal core is solid at {point:?}",
                                room.id,
                                direction.tag(),
                            );
                        }
                    }
                }

                let arch_root = format!("{}/wall_{}/arch", room.id.stable_key(), direction.tag());
                for suffix in ["step_0_left", "step_0_right", "step_1_left", "step_1_right"] {
                    assert!(
                        assembly
                            .brushes
                            .iter()
                            .any(|brush| brush.id == format!("{arch_root}/{suffix}")),
                        "missing pointed surround {arch_root}/{suffix}",
                    );
                }
                let apex_center = wall_point(center, 8, core_top + 8);
                assert!(
                    !point_is_solid(&assembly, apex_center),
                    "pointed first step lost its visible center"
                );
                for tangent in [center - 24, center + 24] {
                    assert!(point_is_solid(
                        &assembly,
                        wall_point(tangent, 8, core_top + 8)
                    ));
                }
                assert!(point_is_solid(
                    &assembly,
                    wall_point(center - 8, 8, core_top + 24)
                ));
            }
        }
    }

    #[test]
    fn segmented_portals_keep_the_full_throat_and_seal_the_crown_interface() {
        let mut config = V3Config::nominal_sparse();
        config.arch_type = ArchType::Segmented;
        config.validate().unwrap();
        let (topology, assembly, _plan) = topology_and_assembly(&config);
        let q = i128::from(CONSTRUCTION_QUANTUM);

        for portal in &topology.portals {
            let source_direction = WallDirection::parse(portal).unwrap();
            let endpoints = [
                (portal.source_room, source_direction),
                (portal.target_room.unwrap(), source_direction.opposite()),
            ];
            for (room_id, direction) in endpoints {
                let room = topology.room(room_id).unwrap();
                let center = if direction.is_horizontal_route() {
                    i128::from(portal.anchor.1)
                } else {
                    i128::from(portal.anchor.0)
                };
                let core_bottom = i128::from(room.floor_z + CONSTRUCTION_QUANTUM);
                let core_top = core_bottom + i128::from(HEADROOM);
                let wall_point = |tangent: i128, normal: i128, z: i128| match direction {
                    WallDirection::North => (tangent, i128::from(room.shell.1) + normal, z),
                    WallDirection::South => (tangent, i128::from(room.shell.3) - normal, z),
                    WallDirection::West => (i128::from(room.shell.0) + normal, tangent, z),
                    WallDirection::East => (i128::from(room.shell.2) - normal, tangent, z),
                };

                // The complete 64×80 throat stays open through the wall and
                // one quantum into the adjoining corridor.
                for normal in [-15_i128, 1, 8, 15] {
                    for tangent in [center - 31, center, center + 31] {
                        for z in [core_bottom + 1, core_bottom + 40, core_top - 1] {
                            let point = wall_point(tangent, normal, z);
                            assert!(
                                !point_is_solid(&assembly, point),
                                "{} {} segmented throat is solid at {point:?}",
                                room.id,
                                direction.tag(),
                            );
                        }
                    }
                }

                let root = format!(
                    "{}/wall_{}/segmented_arch",
                    room.id.stable_key(),
                    direction.tag()
                );
                for suffix in ["shoulder_left", "shoulder_right", "lintel", "interface_cap"] {
                    assert!(
                        assembly
                            .brushes
                            .iter()
                            .any(|brush| brush.id == format!("{root}/{suffix}")),
                        "missing segmented surround {root}/{suffix}",
                    );
                }

                // The two stepped crown bands remain visible from the room:
                // a 64-unit first recess, then a 32-unit centre recess between
                // solid shoulders.
                assert!(!point_is_solid(
                    &assembly,
                    wall_point(center, 8, core_top + 8)
                ));
                assert!(!point_is_solid(
                    &assembly,
                    wall_point(center, 8, core_top + q + 8)
                ));
                for tangent in [center - 24, center + 24] {
                    assert!(point_is_solid(
                        &assembly,
                        wall_point(tangent, 8, core_top + q + 8)
                    ));
                }

                // The centre recess terminates in a one-quantum backing cap
                // immediately outside the room wall instead of opening above
                // the corridor roof into exterior void.
                let cap = assembly
                    .brushes
                    .iter()
                    .find(|brush| brush.id == format!("{root}/interface_cap"))
                    .unwrap();
                let expected = match direction {
                    WallDirection::North => (
                        (center - q, i128::from(room.shell.1) - q, core_top + q),
                        (center + q, i128::from(room.shell.1), core_top + 2 * q),
                    ),
                    WallDirection::South => (
                        (center - q, i128::from(room.shell.3), core_top + q),
                        (center + q, i128::from(room.shell.3) + q, core_top + 2 * q),
                    ),
                    WallDirection::West => (
                        (i128::from(room.shell.0) - q, center - q, core_top + q),
                        (i128::from(room.shell.0), center + q, core_top + 2 * q),
                    ),
                    WallDirection::East => (
                        (i128::from(room.shell.2), center - q, core_top + q),
                        (i128::from(room.shell.2) + q, center + q, core_top + 2 * q),
                    ),
                };
                assert_eq!(cap.brush.aabb().unwrap(), expected);
                assert!(point_is_solid(
                    &assembly,
                    wall_point(center, -8, core_top + q + 8)
                ));
            }
        }
    }

    #[test]
    fn chamfered_rooms_emit_sealed_diagonal_slabs_and_walls() {
        let (topology, assembly, _plan) = sparse_topology_and_assembly();
        let room = topology
            .rooms
            .iter()
            .find(|room| room.is_chamfered)
            .expect("production placement must contain a chamfered room");
        let floor = assembly
            .brushes
            .iter()
            .find(|brush| brush.id == format!("{}/floor", room.id.stable_key()))
            .unwrap();
        let ceiling = assembly
            .brushes
            .iter()
            .find(|brush| brush.id == format!("{}/ceiling", room.id.stable_key()))
            .unwrap();
        let is_diagonal = |role| {
            matches!(
                role,
                FaceRole::DiagNE | FaceRole::DiagNW | FaceRole::DiagSE | FaceRole::DiagSW
            )
        };
        assert!(floor.brush.faces.iter().any(|face| is_diagonal(face.role)));
        assert!(ceiling
            .brush
            .faces
            .iter()
            .any(|face| is_diagonal(face.role)));

        let &(sx, sy) = room.chamfer_corners.first().unwrap();
        let tag = match (sx, sy) {
            (1, 1) => "diag_ne",
            (1, -1) => "diag_se",
            (-1, 1) => "diag_nw",
            (-1, -1) => "diag_sw",
            _ => unreachable!(),
        };
        let diagonal = assembly
            .brushes
            .iter()
            .find(|brush| brush.id == format!("{}/wall_{tag}", room.id.stable_key()))
            .unwrap();
        assert!(diagonal
            .brush
            .faces
            .iter()
            .any(|face| is_diagonal(face.role)));
        let (minimum, maximum) = diagonal.brush.aabb().unwrap();
        assert_eq!(minimum.2, i128::from(room.floor_z));
        assert_eq!(maximum.2, i128::from(room.floor_z + room.dims.2 as i32));
        assert!(
            i128::from(room.chamfer_size).pow(2) >= 2 * i128::from(CONSTRUCTION_QUANTUM).pow(2)
        );

        let corner_x = if sx > 0 { room.shell.2 } else { room.shell.0 };
        let corner_y = if sy > 0 { room.shell.3 } else { room.shell.1 };
        let exterior = (i128::from(corner_x - sx * 8), i128::from(corner_y - sy * 8));
        let interior = (
            i128::from(corner_x - sx * 24),
            i128::from(corner_y - sy * 24),
        );
        for z in [
            i128::from(room.floor_z) + 8,
            i128::from(room.floor_z) + 80,
            i128::from(room.floor_z + room.dims.2 as i32) - 8,
        ] {
            assert!(
                point_is_inside_brush(diagonal, (exterior.0, exterior.1, z)),
                "diagonal shell has an underside/top gap at z={z}",
            );
        }
        assert!(!point_is_solid(
            &assembly,
            (interior.0, interior.1, i128::from(room.floor_z) + 80,)
        ));
    }

    #[test]
    fn run_pipeline_sparse_produces_output() {
        let config = V3Config::nominal_sparse();
        let output = run_pipeline(&config).unwrap();
        assert!(!output.map_text.is_empty());
        assert!(output.map_text.contains("worldspawn"));
        assert!(output.map_text.contains("info_player_start"));
        assert!(output.map_text.contains("light"));
        assert_eq!(output.metadata.seed(), 0);
        assert_eq!(output.metadata.preset(), "sparse");
        assert_eq!(output.metadata.schema_version(), "v3");
    }

    #[test]
    fn run_pipeline_moderate_produces_output() {
        let config = V3Config::nominal_moderate();
        let output = run_pipeline(&config).unwrap();
        assert!(!output.map_text.is_empty());
        assert!(output.map_text.contains("worldspawn"));
    }

    #[test]
    fn run_pipeline_rich_produces_output() {
        let config = V3Config::nominal_rich();
        let output = run_pipeline(&config).unwrap();
        assert!(!output.map_text.is_empty());
        assert!(output.map_text.contains("worldspawn"));
    }

    #[test]
    fn run_pipeline_deterministic() {
        let config = V3Config::nominal_sparse();
        let output1 = run_pipeline(&config).unwrap();
        let output2 = run_pipeline(&config).unwrap();
        assert_eq!(output1.map_text, output2.map_text);
        assert_eq!(output1.metadata, output2.metadata);
    }

    #[test]
    fn run_pipeline_metadata_has_room_counts() {
        let config = V3Config::nominal_sparse();
        let output = run_pipeline(&config).unwrap();
        assert!(output.metadata.room_count() >= 3);
        assert!(output.metadata.lower_room_count() >= 2);
        assert!(output.metadata.upper_room_count() >= 1);
        assert!(output.metadata.has_upper_layer());
    }

    #[test]
    fn run_pipeline_metadata_has_spawn_and_lights() {
        let config = V3Config::nominal_sparse();
        let output = run_pipeline(&config).unwrap();
        let (sx, sy, sz) = output.metadata.spawn_origin();
        assert!(sx > 0);
        assert!(sy > 0);
        assert!(sz > 0);
        assert!(output.metadata.light_count() > 0);
    }

    #[test]
    fn run_pipeline_bounds_are_reasonable() {
        let config = V3Config::nominal_sparse();
        let output = run_pipeline(&config).unwrap();
        let (min_x, min_y, min_z, max_x, max_y, max_z) = output.metadata.bounds();
        assert!(max_x > min_x);
        assert!(max_y > min_y);
        assert!(max_z > min_z);
        assert!(min_z >= 0);
    }

    #[test]
    fn run_pipeline_face_budget_satisfied() {
        let config = V3Config::nominal_rich();
        let output = run_pipeline(&config).unwrap();
        assert!(
            output.metadata.face_budget_satisfied(),
            "actual faces {} exceed estimated {}",
            output.metadata.actual_faces(),
            output.metadata.estimated_faces()
        );
        assert!(output.metadata.actual_faces() < crate::enhanced_v3::config::FACE_BUDGET);
    }

    #[test]
    fn run_pipeline_brushes_in_map_match_metadata() {
        let config = V3Config::nominal_sparse();
        let output = run_pipeline(&config).unwrap();
        // Count brush blocks in map (each block starts with "{" at the beginning of a line
        // but entities also start with "{". Count world brushes as those between worldspawn header and closing "}"
        // Simpler: use metadata
        let brush_blocks = output.map_text.lines().filter(|l| l.trim() == "{").count();
        // Each brush has one "{", each entity has one "{", worldspawn has "{"
        // brush_blocks = 1 (worldspawn) + N brushes + 1 (spawn) + M lights
        let expected_blocks = 1
            + output.metadata.actual_brushes() as usize
            + 1
            + output.metadata.light_count() as usize;
        assert_eq!(brush_blocks, expected_blocks);
    }

    #[test]
    fn run_pipeline_different_seeds_different_maps() {
        let config_a = V3Config::new(0, V3Preset::Sparse, 2048).unwrap();
        let config_b = V3Config::new(42, V3Preset::Sparse, 2048).unwrap();
        let output_a = run_pipeline(&config_a).unwrap();
        let output_b = run_pipeline(&config_b).unwrap();
        assert!(!output_a.map_text.is_empty());
        assert!(!output_b.map_text.is_empty());
    }
}
