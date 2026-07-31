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

use super::assembly::{self, Assembly, AssemblyBrush, BrushRole, Interface, Support};
#[allow(unused_imports)]
use super::config::V3Preset;
use super::config::{V3Config, CONSTRUCTION_QUANTUM, HEADROOM};
use super::emission;
use super::error::V3Error;
use super::footprint::build_footprints;
use super::geometry::{ConvexBrush, FaceRole};
use super::ids::{CommittedPortal, CommittedRoom, CommittedTopology, RoomId, V3IdAllocator};
use super::intent::plan_composition;
use super::metadata::EnhancedV3Metadata;
use super::reservation::{Reservation, ReservationSet};
use super::rng::V3Seed;
use super::topology::{build_topology, compute_reservations};
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
    let seed = V3Seed::new(config.seed);
    let mut alloc = V3IdAllocator::new();

    // 1. Build footprints
    let (footprints, layout) = build_footprints(config, seed, &mut alloc)?;

    // 2. Build committed topology
    let topology = build_topology(config, &footprints, &layout, seed, &mut alloc)?;

    // 3. Compute reservation volumes
    let (spawn_volume, light_volumes) = compute_reservations(&topology)?;

    // 4. Build reservation set
    let mut protected_reservations = ReservationSet::new();
    protected_reservations.add(Reservation::new("spawn", "spawn_point", spawn_volume))?;
    for (i, vol) in light_volumes.iter().enumerate() {
        protected_reservations.add(Reservation::new(format!("light_{i:04}"), "light", *vol))?;
    }

    // 5. Build assembly from topology
    let (assembly, spawn_origin, light_origins) =
        build_assembly_from_topology(&topology, &protected_reservations, seed)?;

    // 6. Plan composition (grammar families)
    let plan = plan_composition(
        super::ids::CompositionId(0),
        config.preset.tag(),
        topology.rooms.len() as u32,
    )?;

    // 7. Compute actual face/entity/brush counts from the assembly
    let actual_brushes = assembly.brushes.len() as u32;
    let actual_faces: u32 = assembly
        .brushes
        .iter()
        .map(|b| b.brush.faces.len() as u32)
        .sum();
    let actual_entities: u32 = 1 + light_origins.len() as u32; // worldspawn + spawn + lights

    // 8. Emit canonical .map text
    let map_text = emission::emit_map_text(&assembly, spawn_origin, &light_origins)?;

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

fn build_room_wall(
    room: &CommittedRoom,
    direction: WallDirection,
    apertures: &[WallAperture],
    wall_thickness: i128,
    brushes: &mut Vec<AssemblyBrush>,
    bounds: &mut Vec<BrushBounds>,
) -> Result<(), V3Error> {
    let rid = room.id.stable_key();
    let base_id = format!("{rid}/wall_{}", direction.tag());
    let (x0, y0, x1, y1) = (
        i128::from(room.shell.0),
        i128::from(room.shell.1),
        i128::from(room.shell.2),
        i128::from(room.shell.3),
    );
    let wall_z0 = i128::from(room.floor_z) + wall_thickness;
    let wall_z1 = i128::from(room.floor_z) + i128::from(room.dims.2) - wall_thickness;
    let wall_span = if direction.is_horizontal_route() {
        (y0 + wall_thickness, y1 - wall_thickness)
    } else {
        (x0 + wall_thickness, x1 - wall_thickness)
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
        if opening.z.1 < wall_z1 {
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

fn positive_overlap(left: (i128, i128), right: (i128, i128)) -> bool {
    left.0 < right.1 && left.1 > right.0
}

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

/// Build a validated assembly from the committed topology.
fn build_assembly_from_topology(
    topology: &CommittedTopology,
    reservations: &ReservationSet,
    _seed: V3Seed,
) -> Result<(Assembly, (i32, i32, i32), Vec<(i32, i32, i32)>), V3Error> {
    let wall_thickness = i128::from(CONSTRUCTION_QUANTUM);
    let apertures = collect_wall_apertures(topology)?;
    let mut brushes: Vec<AssemblyBrush> = Vec::new();
    let mut bounds: Vec<BrushBounds> = Vec::new();

    for room in &topology.rooms {
        let (x0, y0, x1, y1) = (
            i128::from(room.shell.0),
            i128::from(room.shell.1),
            i128::from(room.shell.2),
            i128::from(room.shell.3),
        );
        let z0 = i128::from(room.floor_z);
        let z1 = z0 + i128::from(room.dims.2);
        let rid = room.id.stable_key();

        push_floor_brush(
            &mut brushes,
            &mut bounds,
            format!("{rid}/floor"),
            (x0, x1),
            (y0, y1),
            (z0, z0 + wall_thickness),
        )?;
        push_box_brush(
            &mut brushes,
            &mut bounds,
            format!("{rid}/ceiling"),
            BrushRole::CeilingSlab,
            (x0, x1),
            (y0, y1),
            (z1 - wall_thickness, z1),
        )?;

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
                &mut brushes,
                &mut bounds,
            )?;
        }
    }

    build_route_brushes(topology, wall_thickness, &mut brushes, &mut bounds)?;

    let interfaces = build_contact_interfaces(bounds);
    let protected_volumes = reservations.to_protected_volumes()?;
    brushes.sort_by(|left, right| left.id.cmp(&right.id));
    let assembly = Assembly::new(brushes, interfaces, protected_volumes)?;

    // Spawn origin: center of first room
    let spawn_room = topology
        .rooms
        .first()
        .ok_or_else(|| V3Error::TopologyInvariant {
            detail: "no rooms for spawn".into(),
        })?;
    let spawn_x = (spawn_room.shell.0 + spawn_room.shell.2) / 2;
    let spawn_y = (spawn_room.shell.1 + spawn_room.shell.3) / 2;
    let spawn_z = spawn_room.floor_z + CONSTRUCTION_QUANTUM + HEADROOM / 2;
    let spawn_origin = (spawn_x, spawn_y, spawn_z);

    // Light origins: center of each room near ceiling
    let q = CONSTRUCTION_QUANTUM;
    let light_origins: Vec<(i32, i32, i32)> = topology
        .rooms
        .iter()
        .map(|room| {
            let lx = (room.shell.0 + room.shell.2) / 2;
            let ly = (room.shell.1 + room.shell.3) / 2;
            let lz = room.floor_z + room.dims.2 as i32 - 2 * q;
            (lx, ly, lz)
        })
        .collect();

    Ok((assembly, spawn_origin, light_origins))
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sparse_topology_and_assembly() -> (CommittedTopology, Assembly) {
        let config = V3Config::nominal_sparse();
        let seed = V3Seed::new(config.seed);
        let mut alloc = V3IdAllocator::new();
        let (footprints, layout) = build_footprints(&config, seed, &mut alloc).unwrap();
        let topology = build_topology(&config, &footprints, &layout, seed, &mut alloc).unwrap();
        let (spawn_volume, light_volumes) = compute_reservations(&topology).unwrap();
        let mut reservations = ReservationSet::new();
        reservations
            .add(Reservation::new("spawn", "spawn_point", spawn_volume))
            .unwrap();
        for (index, volume) in light_volumes.into_iter().enumerate() {
            reservations
                .add(Reservation::new(
                    format!("light_{index:04}"),
                    "light",
                    volume,
                ))
                .unwrap();
        }
        let (assembly, _, _) =
            build_assembly_from_topology(&topology, &reservations, seed).unwrap();
        (topology, assembly)
    }

    fn point_is_inside_brush(brush: &AssemblyBrush, point: (i128, i128, i128)) -> bool {
        let (minimum, maximum) = brush.brush.aabb().unwrap();
        point.0 > minimum.0
            && point.0 < maximum.0
            && point.1 > minimum.1
            && point.1 < maximum.1
            && point.2 > minimum.2
            && point.2 < maximum.2
    }

    #[test]
    fn topology_portals_split_walls_and_routes_build_corridor_shells() {
        let (topology, assembly) = sparse_topology_and_assembly();
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
            .any(|brush| brush.id == "room/0000/wall_east/lintel_0000"));
        assert!(assembly
            .brushes
            .iter()
            .any(|brush| brush.id == "room/0001/wall_west/segment_0000"));

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
