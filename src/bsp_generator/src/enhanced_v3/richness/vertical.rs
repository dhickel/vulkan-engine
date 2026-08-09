//! Richness vertical architecture.
//!
//! Every construction is integer-only, owns its slab/wall omissions through
//! live brush segments, and leaves an explicit 64x80 traversal witness.  The
//! baseline EnhancedV3 path never calls this module.

use std::collections::{BTreeMap, BTreeSet};

use crate::enhanced_v3::geometry::{self as v3_geometry, ConvexBrush};

use super::assembly::{
    AssemblyIR, BrushAssembly, BrushAssemblyRole, BudgetDimension, CostSource, EntityAssembly,
    OpeningRecord, SemanticAttribution, SupportTarget,
};
use super::content_types::VerticalRecipe;
use super::error::{RichnessError, RichnessErrorCategory, RichnessErrorCode};
use super::footprint::Footprint3D;
use super::generated_content;
use super::geometry as richness_geom;
use super::ids::{
    ArchetypeIndex, ArchetypeRequestId, BrushAssemblyId, EntityAssemblyId, OpeningAssemblyId,
    ReservationId, VerticalFeatureId,
};
use super::reservation::{ReservationKind, ReservationRecord};

const Q: i128 = 16;
const ROUTE_WIDTH: i128 = 64;
const HEADROOM: i128 = 80;
const LOWER_FLOOR_TOP: i128 = 16;
const LOWER_CEILING: (i128, i128) = (160, 176);
const UPPER_FLOOR: (i128, i128) = (192, 208);
const UPPER_CEILING_BOTTOM: i128 = 352;
const SHELL_Z: (i128, i128) = (16, 352);
const GUARD_HEIGHT: i128 = 48;

const LADDER_OUTER: i128 = 96;
const LADDER_CLEAR: i128 = 64;
const LADDER_WALL: i128 = 16;
const LADDER_UPPER_LANDING: i128 = 80;

const STAIR_TREADS: usize = 12;
const STAIR_RISE: i128 = 16;
const STAIR_RUN: i128 = 16;

const SPIRAL_ENVELOPE: i128 = 224;
const SPIRAL_COLUMN: i128 = 32;
const SPIRAL_RADIAL_DEPTH: i128 = 64;
const SPIRAL_CHAMFER: i128 = 32;

const DROP_CLEAR: i128 = 64;
const DROP_LANDING_OFFSET: i128 = 32;

const ARENA_MIN_SPAN: i128 = 384;

const CONVENTION_REVISION: &str = "enhanced-v3-richness-conventions/v1";

type Bounds = (i128, i128, i128, i128, i128, i128);

#[derive(Debug, Clone)]
pub(crate) struct VerticalFeature {
    pub id: VerticalFeatureId,
    pub composite_id: ReservationId,
    pub kind: VerticalFeatureKind,
}

#[derive(Debug, Clone)]
pub(crate) enum VerticalFeatureKind {
    MultiStoreyShell(MultiStoreyShellData),
    Balcony(BalconyData),
    Catwalk(CatwalkData),
    Overlook(OverlookData),
    PitChasm(PitChasmData),
    Stairwell(StairwellData),
    OpenStairwell(StairwellData),
    LadderShaft(LadderShaftData),
    SpiralStair(SpiralStairData),
    VerticalArena(VerticalArenaData),
}

#[derive(Debug, Clone)]
pub(crate) struct MultiStoreyShellData {
    pub shell_wall_ids: Vec<BrushAssemblyId>,
}

#[derive(Debug, Clone)]
pub(crate) struct BalconyData {
    pub slab_id: BrushAssemblyId,
    pub guard_rail_ids: Vec<BrushAssemblyId>,
    pub support_ids: Vec<BrushAssemblyId>,
}

#[derive(Debug, Clone)]
pub(crate) struct CatwalkData {
    pub deck_id: BrushAssemblyId,
    pub guard_rail_ids: Vec<BrushAssemblyId>,
    pub support_ids: Vec<BrushAssemblyId>,
    pub spans_opening_id: OpeningAssemblyId,
}

#[derive(Debug, Clone)]
pub(crate) struct OverlookData {
    pub opening_id: OpeningAssemblyId,
    pub sill_segment_ids: Vec<BrushAssemblyId>,
}

#[derive(Debug, Clone)]
pub(crate) struct PitChasmData {
    pub upper_opening_id: OpeningAssemblyId,
    pub lower_opening_id: OpeningAssemblyId,
    pub rim_ids: Vec<BrushAssemblyId>,
    pub guard_ids: Vec<BrushAssemblyId>,
    pub landing_id: BrushAssemblyId,
    pub descriptor_id: EntityAssemblyId,
}

#[derive(Debug, Clone)]
pub(crate) struct StairwellData {
    pub tread_ids: Vec<BrushAssemblyId>,
    pub landing_ids: Vec<BrushAssemblyId>,
    pub guard_ids: Vec<BrushAssemblyId>,
    pub upper_opening_id: OpeningAssemblyId,
    pub lower_opening_id: OpeningAssemblyId,
}

#[derive(Debug, Clone)]
pub(crate) struct LadderShaftData {
    pub shell_wall_ids: Vec<BrushAssemblyId>,
    pub rung_ids: Vec<BrushAssemblyId>,
    pub landing_ids: Vec<BrushAssemblyId>,
    pub lip_ids: Vec<BrushAssemblyId>,
    pub upper_opening_id: OpeningAssemblyId,
    pub lower_opening_id: OpeningAssemblyId,
    pub descriptor_id: EntityAssemblyId,
}

#[derive(Debug, Clone)]
pub(crate) struct SpiralStairData {
    pub shell_wall_ids: Vec<BrushAssemblyId>,
    pub column_id: BrushAssemblyId,
    pub tread_ids: Vec<BrushAssemblyId>,
    pub landing_id: BrushAssemblyId,
    pub upper_opening_id: OpeningAssemblyId,
    pub lower_opening_id: OpeningAssemblyId,
}

#[derive(Debug, Clone)]
pub(crate) struct VerticalArenaData {
    pub shell_wall_ids: Vec<BrushAssemblyId>,
    pub balcony_ids: Vec<BrushAssemblyId>,
    pub catwalk_ids: Vec<BrushAssemblyId>,
    pub central_mass_id: BrushAssemblyId,
    pub access_ids: Vec<BrushAssemblyId>,
    pub lower_entry_ids: Vec<OpeningAssemblyId>,
    pub upper_entry_ids: Vec<OpeningAssemblyId>,
}

#[derive(Debug, Clone, Copy)]
struct SpiralTreadTemplate {
    x0: i128,
    x1: i128,
    y0: i128,
    y1: i128,
}

// Three 16-unit treads on each side of the 32x32 column.  Every tread has
// exactly 64 units of radial depth.  Consecutive sides share a positive-width
// edge at each turn.  There is no floating-point angle reconstruction.
const SPIRAL_TREAD_TEMPLATE: [SpiralTreadTemplate; STAIR_TREADS] = [
    SpiralTreadTemplate {
        x0: 16,
        x1: 80,
        y0: -32,
        y1: -16,
    },
    SpiralTreadTemplate {
        x0: 16,
        x1: 80,
        y0: -16,
        y1: 0,
    },
    SpiralTreadTemplate {
        x0: 16,
        x1: 80,
        y0: 0,
        y1: 16,
    },
    SpiralTreadTemplate {
        x0: 16,
        x1: 32,
        y0: 16,
        y1: 80,
    },
    SpiralTreadTemplate {
        x0: 0,
        x1: 16,
        y0: 16,
        y1: 80,
    },
    SpiralTreadTemplate {
        x0: -16,
        x1: 0,
        y0: 16,
        y1: 80,
    },
    SpiralTreadTemplate {
        x0: -80,
        x1: -16,
        y0: 16,
        y1: 32,
    },
    SpiralTreadTemplate {
        x0: -80,
        x1: -16,
        y0: 0,
        y1: 16,
    },
    SpiralTreadTemplate {
        x0: -80,
        x1: -16,
        y0: -16,
        y1: 0,
    },
    SpiralTreadTemplate {
        x0: -32,
        x1: -16,
        y0: -80,
        y1: -16,
    },
    SpiralTreadTemplate {
        x0: -16,
        x1: 0,
        y0: -80,
        y1: -16,
    },
    SpiralTreadTemplate {
        x0: 0,
        x1: 16,
        y0: -80,
        y1: -16,
    },
];

/// Materialize all committed Richness vertical architecture.
pub(crate) fn materialize_vertical_features(
    ir: &mut AssemblyIR,
    reservations: &BTreeMap<ReservationId, ReservationRecord>,
    request_archetypes: &BTreeMap<ArchetypeRequestId, ArchetypeIndex>,
) -> Result<Vec<VerticalFeature>, RichnessError> {
    let mut result = Vec::new();
    let mut next_feature = 0u32;

    for composite in reservations
        .values()
        .filter(|record| record.committed && record.kind == ReservationKind::Composite)
    {
        let room = composite
            .composite_children
            .iter()
            .filter_map(|id| reservations.get(id))
            .find(|record| record.kind == ReservationKind::MultiStoreyRoom);

        if let Some(room) = room {
            let attr = attribution_for(room, request_archetypes)?;
            let shell = build_multi_storey_shell(composite, &attr, ir, &mut next_feature)?;
            result.push(shell);

            match attr.archetype_id_str() {
                Some("arena") => {
                    result.push(build_arena_balcony(
                        composite,
                        &attr,
                        ir,
                        &mut next_feature,
                    )?);
                }
                Some("bridge_crossing") => {
                    result.push(build_bridge_crossing(
                        composite,
                        &attr,
                        ir,
                        &mut next_feature,
                    )?);
                }
                Some("overlook_hall") => {
                    result.push(build_overlook_hall(
                        composite,
                        &attr,
                        ir,
                        &mut next_feature,
                    )?);
                }
                _ => {}
            }
            if attr.archetype_id_str() == Some("grand_arena") {
                result.push(build_vertical_arena(
                    composite,
                    &attr,
                    ir,
                    &mut next_feature,
                )?);
            }
            // Multi-storey rooms whose catalog contract carries a traversal
            // recipe (grand stair hall, spiral tower, ladder hub, observatory)
            // consume the COMPLETE recipe with the room as its own host.
            let recipe = attr
                .archetype
                .map(|archetype| {
                    generated_content::ARCHETYPE_VERTICAL_RECIPE
                        .get(archetype.raw() as usize)
                        .copied()
                        .unwrap_or(VerticalRecipe::None)
                })
                .unwrap_or(VerticalRecipe::None);
            let feature = match recipe {
                VerticalRecipe::Stairwell => Some(build_stairwell(
                    composite,
                    room,
                    &attr,
                    false,
                    ir,
                    &mut next_feature,
                )?),
                VerticalRecipe::OpenStairwell => Some(build_stairwell(
                    composite,
                    room,
                    &attr,
                    true,
                    ir,
                    &mut next_feature,
                )?),
                VerticalRecipe::LadderShaft => {
                    Some(build_ladder_shaft(composite, &attr, ir, &mut next_feature)?)
                }
                VerticalRecipe::SpiralStair => {
                    Some(build_spiral_stair(composite, &attr, ir, &mut next_feature)?)
                }
                _ => None,
            };
            if let Some(feature) = feature {
                result.push(feature);
            }
            continue;
        }

        for pit in composite
            .composite_children
            .iter()
            .filter_map(|id| reservations.get(id))
            .filter(|record| record.kind == ReservationKind::PitOmission)
        {
            let pair_id = pit.pit_pair_room_id.ok_or_else(|| {
                vertical_error(
                    "dispatch.pit_pair",
                    format!("pit omission {} has no committed paired room", pit.id.raw()),
                )
            })?;
            let owner = reservations.get(&pair_id).ok_or_else(|| {
                vertical_error(
                    "dispatch.pit_pair",
                    format!(
                        "pit omission {} references missing room {}",
                        pit.id.raw(),
                        pair_id.raw()
                    ),
                )
            })?;
            let attr = attribution_for(owner, request_archetypes)?;
            result.push(build_pit_chasm_pair(
                composite,
                pit,
                &attr,
                ir,
                &mut next_feature,
            )?);
        }

        let is_grand_arena = match room {
            Some(room) => {
                attribution_for(room, request_archetypes)?.archetype_id_str() == Some("grand_arena")
            }
            None => false,
        };

        for host in composite
            .composite_children
            .iter()
            .filter_map(|id| reservations.get(id))
            .filter(|record| record.kind == ReservationKind::VerticalHost)
        {
            // grand_arena consumes a committed host through its own complete
            // internal access construction.  It is not a new catalog enum.
            if is_grand_arena {
                continue;
            }
            let attr = attribution_for(host, request_archetypes)?;
            let archetype = attr.archetype.ok_or_else(|| {
                vertical_error(
                    "dispatch.recipe",
                    format!("vertical host {} has no archetype identity", host.id.raw()),
                )
            })?;
            let recipe = generated_content::ARCHETYPE_VERTICAL_RECIPE
                .get(archetype.raw() as usize)
                .copied()
                .ok_or_else(|| {
                    vertical_error(
                        "dispatch.recipe",
                        format!(
                            "archetype index {} has no generated vertical contract",
                            archetype.raw()
                        ),
                    )
                })?;
            let feature = match recipe {
                VerticalRecipe::Stairwell => {
                    build_stairwell(composite, host, &attr, false, ir, &mut next_feature)?
                }
                VerticalRecipe::OpenStairwell => {
                    build_stairwell(composite, host, &attr, true, ir, &mut next_feature)?
                }
                VerticalRecipe::LadderShaft => {
                    build_ladder_shaft(composite, &attr, ir, &mut next_feature)?
                }
                VerticalRecipe::SpiralStair => {
                    build_spiral_stair(composite, &attr, ir, &mut next_feature)?
                }
                VerticalRecipe::DropHole => {
                    return Err(vertical_error(
                        "dispatch.drop_hole",
                        format!(
                            "archetype {} requires a committed PitOmission pair, not an unreachable DropShaft host",
                            attr.archetype_id_str().unwrap_or("?")
                        ),
                    ));
                }
                // A request-level vertical host may be attached to an
                // archetype whose authored catalog recipe is intentionally
                // None.  It receives the complete generic open-stair recipe;
                // it is never silently skipped or represented as catalog data.
                VerticalRecipe::None => {
                    build_stairwell(composite, host, &attr, true, ir, &mut next_feature)?
                }
            };
            result.push(feature);
        }

        if let Some(room) = room {
            let attr = attribution_for(room, request_archetypes)?;
            if attr.archetype_id_str() == Some("grand_arena") {
                result.push(build_vertical_arena(
                    composite,
                    &attr,
                    ir,
                    &mut next_feature,
                )?);
            }
        }
    }

    // Quiet negative-space rooms may legitimately occupy both layers without
    // becoming a composite reservation. Their authored vertical recipe is
    // still mandatory, so materialize it against the room's own footprint.
    for room in reservations
        .values()
        .filter(|record| record.committed && record.kind == ReservationKind::NegativeSpace)
    {
        let attr = attribution_for(room, request_archetypes)?;
        let recipe = attr
            .archetype
            .and_then(|archetype| {
                generated_content::ARCHETYPE_VERTICAL_RECIPE
                    .get(archetype.raw() as usize)
                    .copied()
            })
            .unwrap_or(VerticalRecipe::None);
        let feature = match recipe {
            VerticalRecipe::None => continue,
            VerticalRecipe::Stairwell => {
                build_stairwell(room, room, &attr, false, ir, &mut next_feature)?
            }
            VerticalRecipe::OpenStairwell => {
                build_stairwell(room, room, &attr, true, ir, &mut next_feature)?
            }
            VerticalRecipe::LadderShaft => build_ladder_shaft(room, &attr, ir, &mut next_feature)?,
            VerticalRecipe::SpiralStair => build_spiral_stair(room, &attr, ir, &mut next_feature)?,
            VerticalRecipe::DropHole => {
                return Err(vertical_error(
                    "dispatch.drop_hole",
                    format!(
                        "negative-space room {} cannot materialize a drop-hole recipe without a paired pit omission",
                        room.id.raw()
                    ),
                ));
            }
        };
        result.push(feature);
    }

    Ok(result)
}

fn build_multi_storey_shell(
    composite: &ReservationRecord,
    attr: &SemanticAttribution,
    ir: &AssemblyIR,
    next_feature: &mut u32,
) -> Result<VerticalFeature, RichnessError> {
    let shell_wall_ids = ir
        .brushes
        .values()
        .filter(|brush| {
            brush.owner.request_id == attr.request_id
                && brush.role.is_wall()
                && brush
                    .brush
                    .aabb()
                    .is_ok_and(|(min, max)| min.2 == SHELL_Z.0 && max.2 == SHELL_Z.1)
        })
        .map(|brush| brush.id)
        .collect::<Vec<_>>();
    if shell_wall_ids.len() < 4 {
        return Err(vertical_error(
            "grand_shell",
            format!(
                "archetype {} has {} full-height shell walls; at least four are required",
                attr.archetype_id_str().unwrap_or("?"),
                shell_wall_ids.len()
            ),
        ));
    }
    Ok(VerticalFeature {
        id: alloc_feature(next_feature),
        composite_id: composite.id,
        kind: VerticalFeatureKind::MultiStoreyShell(MultiStoreyShellData { shell_wall_ids }),
    })
}

fn build_arena_balcony(
    composite: &ReservationRecord,
    attr: &SemanticAttribution,
    ir: &mut AssemblyIR,
    next_feature: &mut u32,
) -> Result<VerticalFeature, RichnessError> {
    let (x0, y0, x1, _) = room_bounds(composite);
    // Arena rooms are octagonal: the diagonal corner walls occupy the first
    // 96 units of each cardinal span. Start at their exact tangent boundary
    // so the balcony touches, but never positively overlaps, the chamfers.
    let span = (x0 + 96, x1 - 96);
    let data = build_balcony_mezzanine(
        BrushAssemblyRole::NorthWall,
        y0 + 16,
        span,
        80,
        UPPER_FLOOR.0,
        attr,
        ir,
    )?;
    Ok(VerticalFeature {
        id: alloc_feature(next_feature),
        composite_id: composite.id,
        kind: VerticalFeatureKind::Balcony(data),
    })
}

fn build_bridge_crossing(
    composite: &ReservationRecord,
    attr: &SemanticAttribution,
    ir: &mut AssemblyIR,
    next_feature: &mut u32,
) -> Result<VerticalFeature, RichnessError> {
    let (x0, y0, x1, y1) = room_bounds(composite);
    let cx = (x0 + x1) / 2;
    let cy = (y0 + y1) / 2;
    let axis = if x1 - x0 >= 256 && y1 - y0 >= 160 {
        CatwalkAxis::X
    } else if x1 - x0 >= 160 && y1 - y0 >= 256 {
        CatwalkAxis::Y
    } else {
        return Err(vertical_error(
            "bridge.envelope",
            format!(
                "{}x{} room cannot contain a protected catwalk",
                x1 - x0,
                y1 - y0
            ),
        ));
    };
    let (hole, partition) = match axis {
        CatwalkAxis::X => (
            (
                cx - 96,
                cy - 48,
                UPPER_FLOOR.0,
                cx + 96,
                cy + 48,
                UPPER_FLOOR.1,
            ),
            (
                cx - 112,
                cy - 64,
                UPPER_FLOOR.0,
                cx + 112,
                cy + 64,
                UPPER_FLOOR.1,
            ),
        ),
        CatwalkAxis::Y => (
            (
                cx - 48,
                cy - 96,
                UPPER_FLOOR.0,
                cx + 48,
                cy + 96,
                UPPER_FLOOR.1,
            ),
            (
                cx - 64,
                cy - 112,
                UPPER_FLOOR.0,
                cx + 64,
                cy + 112,
                UPPER_FLOOR.1,
            ),
        ),
    };
    require_bounds_inside(
        partition,
        (x0 + 16, y0 + 16, 0, x1 - 16, y1 - 16, 368),
        "bridge.partition",
    )?;
    let upper = carve_slab_opening(ir, partition, hole, BrushAssemblyRole::FloorSlab, attr)?;
    let lower_hole = with_z(hole, LOWER_CEILING);
    let lower_partition = with_z(partition, LOWER_CEILING);
    let _ = carve_slab_opening(
        ir,
        lower_partition,
        lower_hole,
        BrushAssemblyRole::CeilingSlab,
        attr,
    )?;
    let (span_start, span_end, cross_center) = match axis {
        CatwalkAxis::X => (cx - 80, cx + 80, cy),
        CatwalkAxis::Y => (cy - 80, cy + 80, cx),
    };
    let data = build_catwalk_bridge(
        span_start,
        span_end,
        cross_center,
        ROUTE_WIDTH,
        UPPER_FLOOR.1,
        axis,
        upper,
        attr,
        ir,
    )?;
    Ok(VerticalFeature {
        id: alloc_feature(next_feature),
        composite_id: composite.id,
        kind: VerticalFeatureKind::Catwalk(data),
    })
}

fn build_overlook_hall(
    composite: &ReservationRecord,
    attr: &SemanticAttribution,
    ir: &mut AssemblyIR,
    next_feature: &mut u32,
) -> Result<VerticalFeature, RichnessError> {
    let (x0, y0, x1, y1) = room_bounds(composite);
    let cy = snap_down((y0 + y1) / 2);
    let partition = (x0 + 32, cy, 16, x1 - 32, cy + 16, 352);
    let aperture = (
        (x0 + x1) / 2 - 64,
        cy,
        16 + 48,
        (x0 + x1) / 2 + 64,
        cy + 16,
        160,
    );
    let (segments, openings) = insert_partition_with_openings(
        ir,
        partition,
        BrushAssemblyRole::PartialWall,
        attr,
        &[aperture],
    )?;
    let sill_segment_ids = segments
        .into_iter()
        .filter(|id| {
            ir.brushes[id]
                .brush
                .aabb()
                .is_ok_and(|(_, max)| max.2 == aperture.2)
        })
        .collect();
    Ok(VerticalFeature {
        id: alloc_feature(next_feature),
        composite_id: composite.id,
        kind: VerticalFeatureKind::Overlook(OverlookData {
            opening_id: openings[0],
            sill_segment_ids,
        }),
    })
}

/// Build a supported balcony.  `wall_coordinate` is the room-side face of
/// the cardinal wall; `projection` includes a 16-unit guard band and therefore
/// must be at least 80 to retain a clear 64-unit route.
fn build_balcony_mezzanine(
    wall_role: BrushAssemblyRole,
    wall_coordinate: i128,
    wall_span: (i128, i128),
    projection: i128,
    slab_z0: i128,
    attr: &SemanticAttribution,
    ir: &mut AssemblyIR,
) -> Result<BalconyData, RichnessError> {
    if projection < ROUTE_WIDTH + Q || projection % Q != 0 {
        return Err(vertical_error(
            "balcony.projection",
            format!("projection {projection} does not preserve a 64-unit route plus guard"),
        ));
    }
    if wall_span.1 - wall_span.0 < ROUTE_WIDTH || wall_span.0 % Q != 0 || wall_span.1 % Q != 0 {
        return Err(vertical_error(
            "balcony.span",
            "balcony span is not a quantum-aligned 64-unit route",
        ));
    }
    let (sx0, sy0, sx1, sy1) = match wall_role {
        BrushAssemblyRole::NorthWall => (
            wall_span.0,
            wall_coordinate,
            wall_span.1,
            wall_coordinate + projection,
        ),
        BrushAssemblyRole::SouthWall => (
            wall_span.0,
            wall_coordinate - projection,
            wall_span.1,
            wall_coordinate,
        ),
        BrushAssemblyRole::WestWall => (
            wall_coordinate,
            wall_span.0,
            wall_coordinate + projection,
            wall_span.1,
        ),
        BrushAssemblyRole::EastWall => (
            wall_coordinate - projection,
            wall_span.0,
            wall_coordinate,
            wall_span.1,
        ),
        _ => {
            return Err(vertical_error(
                "balcony.wall",
                "balcony requires a cardinal wall",
            ))
        }
    };
    let slab_id = insert_box(
        ir,
        (sx0, sy0, slab_z0, sx1, sy1, slab_z0 + Q),
        BrushAssemblyRole::BalconySlab,
        attr,
    )?;

    let rail = match wall_role {
        BrushAssemblyRole::NorthWall => (
            sx0,
            sy1 - Q,
            slab_z0 + Q,
            sx1,
            sy1,
            slab_z0 + Q + GUARD_HEIGHT,
        ),
        BrushAssemblyRole::SouthWall => (
            sx0,
            sy0,
            slab_z0 + Q,
            sx1,
            sy0 + Q,
            slab_z0 + Q + GUARD_HEIGHT,
        ),
        BrushAssemblyRole::WestWall => (
            sx1 - Q,
            sy0,
            slab_z0 + Q,
            sx1,
            sy1,
            slab_z0 + Q + GUARD_HEIGHT,
        ),
        BrushAssemblyRole::EastWall => (
            sx0,
            sy0,
            slab_z0 + Q,
            sx0 + Q,
            sy1,
            slab_z0 + Q + GUARD_HEIGHT,
        ),
        _ => unreachable!(),
    };
    let guard_rail_ids = vec![insert_box(ir, rail, BrushAssemblyRole::GuardRail, attr)?];

    let mut support_ids = Vec::new();
    let mut pos = wall_span.0;
    while pos < wall_span.1 {
        let next = (pos + Q).min(wall_span.1);
        let support = match wall_role {
            BrushAssemblyRole::NorthWall => (pos, sy0, LOWER_FLOOR_TOP, next, sy0 + Q, slab_z0),
            BrushAssemblyRole::SouthWall => (pos, sy1 - Q, LOWER_FLOOR_TOP, next, sy1, slab_z0),
            BrushAssemblyRole::WestWall => (sx0, pos, LOWER_FLOOR_TOP, sx0 + Q, next, slab_z0),
            BrushAssemblyRole::EastWall => (sx1 - Q, pos, LOWER_FLOOR_TOP, sx1, next, slab_z0),
            _ => unreachable!(),
        };
        if !bounds_overlap_any(ir, support, &[])? {
            support_ids.push(insert_box(ir, support, BrushAssemblyRole::Corbel, attr)?);
        }
        pos += 96;
    }
    if support_ids.is_empty() {
        return Err(vertical_error(
            "balcony.support",
            "no positive-area balcony support could be placed",
        ));
    }
    Ok(BalconyData {
        slab_id,
        guard_rail_ids,
        support_ids,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CatwalkAxis {
    X,
    Y,
}

fn build_catwalk_bridge(
    span_start: i128,
    span_end: i128,
    cross_center: i128,
    clear_width: i128,
    deck_top: i128,
    axis: CatwalkAxis,
    opening_id: OpeningAssemblyId,
    attr: &SemanticAttribution,
    ir: &mut AssemblyIR,
) -> Result<CatwalkData, RichnessError> {
    if clear_width < ROUTE_WIDTH || clear_width % Q != 0 || span_end <= span_start {
        return Err(vertical_error(
            "catwalk.route",
            "catwalk requires a quantum-aligned 64-unit clear route",
        ));
    }
    let overall_width = clear_width + 2 * Q;
    let half = overall_width / 2;
    let (x0, y0, x1, y1) = match axis {
        CatwalkAxis::X => (
            span_start - Q,
            cross_center - half,
            span_end + Q,
            cross_center + half,
        ),
        CatwalkAxis::Y => (
            cross_center - half,
            span_start - Q,
            cross_center + half,
            span_end + Q,
        ),
    };
    let opening = ir.openings.get(&opening_id).ok_or_else(|| {
        vertical_error(
            "catwalk.opening",
            format!("missing committed opening {}", opening_id.raw()),
        )
    })?;
    if x0 < opening.bounds.0
        || y0 < opening.bounds.1
        || x1 > opening.bounds.3
        || y1 > opening.bounds.4
    {
        return Err(vertical_error(
            "catwalk.opening",
            "catwalk including embedded ends escapes its committed void",
        ));
    }

    let deck_bounds = (x0, y0, deck_top - Q, x1, y1, deck_top);
    split_guards_for_catwalk(ir, deck_bounds, attr)?;
    let deck_id = insert_box(ir, deck_bounds, BrushAssemblyRole::CatwalkDeck, attr)?;
    let rails = match axis {
        CatwalkAxis::X => [
            (x0, y0, deck_top, x1, y0 + Q, deck_top + GUARD_HEIGHT),
            (x0, y1 - Q, deck_top, x1, y1, deck_top + GUARD_HEIGHT),
        ],
        CatwalkAxis::Y => [
            (x0, y0, deck_top, x0 + Q, y1, deck_top + GUARD_HEIGHT),
            (x1 - Q, y0, deck_top, x1, y1, deck_top + GUARD_HEIGHT),
        ],
    };
    let mut guard_rail_ids = Vec::new();
    for rail in rails {
        guard_rail_ids.push(insert_box(ir, rail, BrushAssemblyRole::GuardRail, attr)?);
    }

    let end_footprints = match axis {
        CatwalkAxis::X => [(x0, y0, span_start, y1), (span_end, y0, x1, y1)],
        CatwalkAxis::Y => [(x0, y0, x1, span_start), (x0, span_end, x1, y1)],
    };
    let deck_bottom = deck_top - Q;
    let mut support_ids = Vec::new();
    for (sx0, sy0, sx1, sy1) in end_footprints {
        let existing_support = ir
            .brushes
            .values()
            .find(|candidate| {
                candidate.id != deck_id
                    && candidate.brush.aabb().is_ok_and(|(min, max)| {
                        max.2 == deck_bottom
                            && min.0 < sx1
                            && max.0 > sx0
                            && min.1 < sy1
                            && max.1 > sy0
                    })
            })
            .map(|candidate| candidate.id);
        if let Some(id) = existing_support {
            support_ids.push(id);
        } else {
            support_ids.push(insert_box(
                ir,
                (sx0, sy0, LOWER_FLOOR_TOP, sx1, sy1, deck_bottom),
                BrushAssemblyRole::VerticalSupport,
                attr,
            )?);
        }
    }
    Ok(CatwalkData {
        deck_id,
        guard_rail_ids,
        support_ids,
        spans_opening_id: opening_id,
    })
}

fn split_guards_for_catwalk(
    ir: &mut AssemblyIR,
    deck: Bounds,
    owner: &SemanticAttribution,
) -> Result<(), RichnessError> {
    let guard_ids = ir
        .brushes
        .values()
        .filter(|brush| brush.role == BrushAssemblyRole::GuardRail && brush.owner == *owner)
        .filter_map(|brush| {
            let bounds = brush_bounds(brush).ok()?;
            bounds_intersection(bounds, deck).map(|_| brush.id)
        })
        .collect::<Vec<_>>();
    for id in guard_ids {
        let guard = ir.remove_brush(id).ok_or_else(|| {
            vertical_error("catwalk.guard", format!("missing guard {}", id.raw()))
        })?;
        remove_brush_references(ir, id);
        let bounds = brush_bounds(&guard)?;
        let approach = (deck.0, deck.1, bounds.2, deck.3, deck.4, bounds.5);
        for piece in subtract_bounds(bounds, approach) {
            insert_box_with_cost(ir, piece, BrushAssemblyRole::GuardRail, owner, guard.cost)?;
        }
    }
    Ok(())
}

fn build_pit_chasm_pair(
    composite: &ReservationRecord,
    pit: &ReservationRecord,
    attr: &SemanticAttribution,
    ir: &mut AssemblyIR,
    next_feature: &mut u32,
) -> Result<VerticalFeature, RichnessError> {
    let (x0, y0, x1, y1) = room_bounds(composite);
    let cx = snap_down((x0 + x1) / 2);
    let cy = snap_down((y0 + y1) / 2);
    let hole_upper = (
        cx - 32,
        cy - 32,
        UPPER_FLOOR.0,
        cx + 32,
        cy + 32,
        UPPER_FLOOR.1,
    );
    let (px0, py0, px1, py1) = room_bounds(pit);
    if !pit.footprint.occupies_upper
        || hole_upper.0 < px0
        || hole_upper.1 < py0
        || hole_upper.3 > px1
        || hole_upper.4 > py1
    {
        return Err(vertical_error(
            "pit.omission",
            format!(
                "pit omission {} does not own the complete 64x64 upper hole",
                pit.id.raw()
            ),
        ));
    }
    let hole_lower = with_z(hole_upper, LOWER_CEILING);

    let upper_partition = (
        x0 + 16,
        y0 + 16,
        UPPER_FLOOR.0,
        x1 - 16,
        y1 - 16,
        UPPER_FLOOR.1,
    );
    let lower_partition = (x0, y0, LOWER_CEILING.0, x1, y1, LOWER_CEILING.1);
    let upper_opening_id = carve_slab_opening(
        ir,
        upper_partition,
        hole_upper,
        BrushAssemblyRole::FloorSlab,
        attr,
    )?;
    let lower_opening_id = carve_slab_opening(
        ir,
        lower_partition,
        hole_lower,
        BrushAssemblyRole::CeilingSlab,
        attr,
    )?;

    let mut rim_ids = Vec::new();
    let mut guard_ids = Vec::new();
    let rim_bounds = [
        (
            cx - 32,
            cy - 48,
            UPPER_FLOOR.1,
            cx + 32,
            cy - 32,
            UPPER_FLOOR.1 + 16,
        ),
        (
            cx - 32,
            cy + 32,
            UPPER_FLOOR.1,
            cx + 32,
            cy + 48,
            UPPER_FLOOR.1 + 16,
        ),
        (
            cx - 48,
            cy - 32,
            UPPER_FLOOR.1,
            cx - 32,
            cy + 32,
            UPPER_FLOOR.1 + 16,
        ),
    ];
    for rim in rim_bounds {
        let rim_id = insert_box(ir, rim, BrushAssemblyRole::PitPerimeterSlab, attr)?;
        rim_ids.push(rim_id);
        guard_ids.push(insert_box(
            ir,
            with_z(rim, (UPPER_FLOOR.1 + 16, UPPER_FLOOR.1 + 16 + GUARD_HEIGHT)),
            BrushAssemblyRole::DropEntryGuard,
            attr,
        )?);
    }

    let landing_x0 = cx - DROP_CLEAR / 2 + DROP_LANDING_OFFSET;
    let landing = (
        landing_x0,
        cy - 32,
        LOWER_FLOOR_TOP,
        landing_x0 + DROP_CLEAR,
        cy + 32,
        LOWER_FLOOR_TOP + 16,
    );
    let landing_id = insert_box(ir, landing, BrushAssemblyRole::DropLanding, attr)?;
    let descriptor_id = insert_movement_descriptor(
        ir,
        "one_way_drop",
        (
            cx - 32,
            cy - 32,
            UPPER_FLOOR.1 - HEADROOM,
            cx + 32,
            cy + 32,
            UPPER_FLOOR.1,
        ),
        (1, 0, 0),
        attr,
    )?;

    Ok(VerticalFeature {
        id: alloc_feature(next_feature),
        composite_id: composite.id,
        kind: VerticalFeatureKind::PitChasm(PitChasmData {
            upper_opening_id,
            lower_opening_id,
            rim_ids,
            guard_ids,
            landing_id,
            descriptor_id,
        }),
    })
}

#[derive(Debug, Clone, Copy)]
struct Frame2D {
    ox: i128,
    oy: i128,
    rotate: bool,
    inset_x: i128,
    inset_y: i128,
}

impl Frame2D {
    fn bounds(self, x0: i128, y0: i128, z0: i128, x1: i128, y1: i128, z1: i128) -> Bounds {
        let x0 = x0 + self.inset_x;
        let x1 = x1 + self.inset_x;
        let y0 = y0 + self.inset_y;
        let y1 = y1 + self.inset_y;
        if self.rotate {
            (
                self.ox + y0,
                self.oy + x0,
                z0,
                self.ox + y1,
                self.oy + x1,
                z1,
            )
        } else {
            (
                self.ox + x0,
                self.oy + y0,
                z0,
                self.ox + x1,
                self.oy + y1,
                z1,
            )
        }
    }
}

fn stair_frame(
    composite: &ReservationRecord,
    host: &ReservationRecord,
    ir: &AssemblyIR,
) -> Result<Frame2D, RichnessError> {
    // The authored construction occupies a 192x416 canonical frame. Its
    // frozen placement envelope contributes one additional 32-unit sealing
    // band along one axis: 192x448 for the long form or 224x416 for the
    // compact form. Either form may be rotated as a whole in map XY.
    const CANDIDATES: [(bool, i128, i128, i128, i128); 4] = [
        (false, 192, 448, 0, Q),
        (false, 224, 416, Q, 0),
        (true, 448, 192, 0, Q),
        (true, 416, 224, Q, 0),
    ];

    // Multi-storey and negative-space recipes belong to their room and must
    // never escape into the wider composite projection. A 96x96 VerticalHost
    // is only a route witness; generic hosted stairs may instead use the
    // composite room that structurally owns that witness.
    let mut containers = vec![room_bounds(host)];
    if host.kind == ReservationKind::VerticalHost && host.id != composite.id {
        containers.push(room_bounds(composite));
    }
    containers.dedup();

    for (x0, y0, x1, y1) in containers.iter().copied() {
        let interior = (x0 + Q, y0 + Q, x1 - Q, y1 - Q);
        for (rotate, envelope_width, envelope_height, inset_x, inset_y) in CANDIDATES {
            if interior.2 - interior.0 < envelope_width || interior.3 - interior.1 < envelope_height
            {
                continue;
            }
            let mut origins = Vec::new();
            for ox in (interior.0..=interior.2 - envelope_width).step_by(Q as usize) {
                for oy in (interior.1..=interior.3 - envelope_height).step_by(Q as usize) {
                    let dx = (2 * ox + envelope_width - interior.0 - interior.2).abs();
                    let dy = (2 * oy + envelope_height - interior.1 - interior.3).abs();
                    origins.push((dx + dy, dy, dx, oy, ox));
                }
            }
            origins.sort_unstable();

            for (_, _, _, envelope_y, envelope_x) in origins {
                let envelope = (
                    envelope_x,
                    envelope_y,
                    SHELL_Z.0,
                    envelope_x + envelope_width,
                    envelope_y + envelope_height,
                    SHELL_Z.1,
                );
                let envelope_brush = ConvexBrush::make_box(
                    (envelope.0, envelope.3),
                    (envelope.1, envelope.4),
                    (envelope.2, envelope.5),
                )
                .map_err(|error| vertical_error("stairwell.frame", format!("{error}")))?;
                let wall_conflict = ir
                    .brushes
                    .values()
                    .filter(|brush| brush.role.is_wall())
                    .any(|wall| {
                        richness_geom::brushes_overlap(&envelope_brush, &wall.brush).unwrap_or(true)
                    });
                if wall_conflict {
                    continue;
                }

                // Center the 192x416 physical construction inside the chosen
                // sealing envelope before applying its optional map rotation.
                return Ok(Frame2D {
                    ox: envelope_x,
                    oy: envelope_y,
                    rotate,
                    inset_x,
                    inset_y,
                });
            }
        }
    }

    let host_bounds = room_bounds(host);
    let available_x = host_bounds.2 - host_bounds.0 - 2 * Q;
    let available_y = host_bounds.3 - host_bounds.1 - 2 * Q;
    Err(vertical_error(
        "stairwell.envelope",
        format!(
            "{}x{} interior cannot contain a wall-clear 192x448, 224x416, or rotated equivalent stairwell",
            available_x, available_y
        ),
    ))
}

fn build_stairwell(
    composite: &ReservationRecord,
    host: &ReservationRecord,
    attr: &SemanticAttribution,
    open: bool,
    ir: &mut AssemblyIR,
    next_feature: &mut u32,
) -> Result<VerticalFeature, RichnessError> {
    let frame = stair_frame(composite, host, ir)?;
    // The complete route occupies y=0..416 inside the 448-unit frame:
    // lower landing (0..64), lower flight (64..128), turn (128..224), upper
    // flight (224..352), and upper landing (352..416). Keeping the slab
    // partitions to that exact envelope avoids the former unowned 16-unit
    // tail while retaining a live owner rim around every omission.
    let partition_upper = frame.bounds(16, 0, UPPER_FLOOR.0, 176, 416, UPPER_FLOOR.1);
    // The upper-floor opening clears the full upper flight and its landing.
    let hole_upper = frame.bounds(96, 224, UPPER_FLOOR.0, 176, 416, UPPER_FLOOR.1);
    // The lower ceiling is opened only where the upper flight rises through
    // it. The lower flight and turn remain beneath the intact slab with the
    // required 80 units of headroom.
    let partition_lower = with_z(
        frame.bounds(16, 0, LOWER_FLOOR_TOP, 176, 416, LOWER_FLOOR_TOP),
        LOWER_CEILING,
    );
    let hole_lower = with_z(
        frame.bounds(16, 192, LOWER_FLOOR_TOP, 176, 416, LOWER_FLOOR_TOP),
        LOWER_CEILING,
    );
    let upper_opening_id = carve_slab_opening(
        ir,
        partition_upper,
        hole_upper,
        BrushAssemblyRole::FloorSlab,
        attr,
    )?;
    let lower_opening_id = carve_slab_opening(
        ir,
        partition_lower,
        hole_lower,
        BrushAssemblyRole::CeilingSlab,
        attr,
    )?;

    // `carve_slab_opening` may add a temporary bearing column for a split
    // ceiling segment. A column inside the committed stair witnesses would
    // turn the route into a decorative-only construction, so retain support
    // only outside every lower, turn, upper-flight, and upper-landing volume.
    let stair_witnesses = [
        frame.bounds(16, 0, LOWER_FLOOR_TOP, 80, 64, LOWER_FLOOR_TOP + HEADROOM),
        frame.bounds(16, 64, LOWER_FLOOR_TOP, 80, 128, LOWER_CEILING.0),
        frame.bounds(16, 128, 80, 176, 224, LOWER_CEILING.0),
        frame.bounds(
            112,
            224,
            LOWER_FLOOR_TOP,
            176,
            352,
            UPPER_FLOOR.1 + HEADROOM,
        ),
        frame.bounds(112, 352, UPPER_FLOOR.1, 176, 416, UPPER_FLOOR.1 + HEADROOM),
    ];
    remove_stair_witness_supports(ir, &stair_witnesses)?;

    let lower_id = replace_floor_patch(
        ir,
        frame.bounds(16, 0, 0, 80, 64, LOWER_FLOOR_TOP),
        BrushAssemblyRole::StairLanding,
        attr,
    )?;

    // Lower flight: 4 steps rising 16 -> 80 (80 units of headroom below the
    // lower ceiling at 160, per the frozen controller contract).
    let mut tread_ids = Vec::new();
    for step in 0..4 {
        let y0 = 64 + step as i128 * STAIR_RUN;
        let top = LOWER_FLOOR_TOP + (step as i128 + 1) * STAIR_RISE;
        tread_ids.push(insert_box(
            ir,
            frame.bounds(16, y0, LOWER_FLOOR_TOP, 80, y0 + STAIR_RUN, top),
            BrushAssemblyRole::StairTread,
            attr,
        )?);
    }
    // Upper flight: 8 steps rising 80 -> 208 (the 12-step frozen contract)
    // through the carved ceiling opening (headroom above the upper floor).
    for step in 0..8 {
        let y0 = 224 + step as i128 * STAIR_RUN;
        let top = LOWER_FLOOR_TOP + (step as i128 + 5) * STAIR_RISE;
        tread_ids.push(insert_box(
            ir,
            frame.bounds(112, y0, LOWER_FLOOR_TOP, 176, y0 + STAIR_RUN, top),
            BrushAssemblyRole::StairTread,
            attr,
        )?);
    }

    // Composite-host frames may extend past a child room's original floor
    // partition. Materialize a lower slab patch beneath any such tread so
    // each integral riser has a real positive-area world support.
    ensure_tread_floor_supports(ir, &tread_ids, attr)?;

    let turn_id = insert_box(
        ir,
        frame.bounds(16, 128, LOWER_FLOOR_TOP, 176, 224, 80),
        BrushAssemblyRole::StairLanding,
        attr,
    )?;
    let upper_id = replace_floor_patch(
        ir,
        frame.bounds(112, 352, UPPER_FLOOR.0, 176, 416, UPPER_FLOOR.1),
        BrushAssemblyRole::StairLanding,
        attr,
    )?;

    let mut guard_ids = Vec::new();
    // Fixed guards anchor on the intact floor outside the carved stairwell
    // opening (the opening spans x 16..176, y 0..224), so their bases always
    // have a positive-area gravity contact.
    for bounds in [
        frame.bounds(0, 64, LOWER_FLOOR_TOP, 16, 160, LOWER_CEILING.1),
        frame.bounds(176, 64, LOWER_FLOOR_TOP, 192, 160, LOWER_CEILING.1),
    ] {
        for piece in split_around_interlayer(bounds) {
            guard_ids.push(insert_box(ir, piece, BrushAssemblyRole::StairGuard, attr)?);
        }
    }
    if open {
        // One guard per actual tread: four on the lower flight and eight on
        // the upper flight. Do not synthesize rails beyond the four lower
        // treads into the turn landing.
        for step in 0..4 {
            let y0 = 64 + step as i128 * STAIR_RUN;
            let base = LOWER_FLOOR_TOP + (step as i128 + 1) * STAIR_RISE;
            let top = (base + GUARD_HEIGHT).min(UPPER_FLOOR.0);
            let guard = frame.bounds(80, y0, base, 96, y0 + STAIR_RUN, top);
            insert_box(
                ir,
                (guard.0, guard.1, LOWER_FLOOR_TOP, guard.3, guard.4, base),
                BrushAssemblyRole::VerticalSupport,
                attr,
            )?;
            guard_ids.push(insert_box(ir, guard, BrushAssemblyRole::StairGuard, attr)?);
        }
        for step in 0..8 {
            let y1 = 224 + step as i128 * STAIR_RUN;
            let base = LOWER_FLOOR_TOP + (step as i128 + 5) * STAIR_RISE;
            let top = base + GUARD_HEIGHT;
            let guard = frame.bounds(96, y1, base, 112, y1 + STAIR_RUN, top);
            insert_box(
                ir,
                (guard.0, guard.1, LOWER_FLOOR_TOP, guard.3, guard.4, base),
                BrushAssemblyRole::VerticalSupport,
                attr,
            )?;
            guard_ids.push(insert_box(ir, guard, BrushAssemblyRole::StairGuard, attr)?);
        }
    } else {
        // The closed stair's central guard stops below the inter-layer slab.
        // It is a lower-flight guard, not a solid shaft through the retained
        // upper floor, so extending it to the upper ceiling would overlap the
        // live slab in swapped frames.
        let guard = frame.bounds(80, 64, LOWER_FLOOR_TOP, 112, 128, LOWER_CEILING.0);
        let guard_id = insert_box(ir, guard, BrushAssemblyRole::StairGuard, attr)?;
        ensure_tread_floor_supports(ir, &[guard_id], attr)?;
        guard_ids.push(guard_id);
    }

    // Re-establish a legal bearing for every remaining lower-ceiling segment
    // after route/landing construction has excluded support columns from the
    // protected stair witnesses.
    let ceiling_ids = ir
        .brushes
        .values()
        .filter(|brush| brush.role == BrushAssemblyRole::CeilingSlab && brush.owner == *attr)
        .map(|brush| brush.id)
        .collect::<Vec<_>>();
    ensure_ceiling_supports(ir, &ceiling_ids, attr, &stair_witnesses)?;

    let data = StairwellData {
        tread_ids,
        landing_ids: vec![lower_id, turn_id, upper_id],
        guard_ids,
        upper_opening_id,
        lower_opening_id,
    };
    Ok(VerticalFeature {
        id: alloc_feature(next_feature),
        composite_id: composite.id,
        kind: if open {
            VerticalFeatureKind::OpenStairwell(data)
        } else {
            VerticalFeatureKind::Stairwell(data)
        },
    })
}

fn remove_stair_witness_supports(
    ir: &mut AssemblyIR,
    witnesses: &[Bounds],
) -> Result<(), RichnessError> {
    let mut remove = Vec::new();
    for brush in ir.brushes.values() {
        if brush.role != BrushAssemblyRole::VerticalSupport {
            continue;
        }
        if witnesses.iter().any(|bounds| {
            ConvexBrush::make_box(
                (bounds.0, bounds.3),
                (bounds.1, bounds.4),
                (bounds.2, bounds.5),
            )
            .is_ok_and(|witness| {
                richness_geom::brushes_overlap(&witness, &brush.brush).unwrap_or(false)
            })
        }) {
            remove.push(brush.id);
        }
    }
    for id in remove {
        ir.remove_brush(id);
        remove_brush_references(ir, id);
    }
    Ok(())
}

fn ensure_tread_floor_supports(
    ir: &mut AssemblyIR,
    tread_ids: &[BrushAssemblyId],
    owner: &SemanticAttribution,
) -> Result<(), RichnessError> {
    for tread_id in tread_ids {
        let (bounds, supported) = {
            let tread = ir
                .brushes
                .get(tread_id)
                .ok_or_else(|| vertical_error("stairwell.support", "tread disappeared"))?;
            (brush_bounds(tread)?, touches_floor(ir, tread)?)
        };
        if !supported {
            insert_box(
                ir,
                (bounds.0, bounds.1, 0, bounds.3, bounds.4, LOWER_FLOOR_TOP),
                BrushAssemblyRole::FloorSlab,
                owner,
            )?;
        }
    }
    Ok(())
}

fn split_around_interlayer(bounds: Bounds) -> Vec<Bounds> {
    let mut pieces = vec![bounds];
    for band in [
        (
            i128::MIN / 4,
            i128::MIN / 4,
            LOWER_CEILING.0,
            i128::MAX / 4,
            i128::MAX / 4,
            LOWER_CEILING.1,
        ),
        (
            i128::MIN / 4,
            i128::MIN / 4,
            UPPER_FLOOR.0,
            i128::MAX / 4,
            i128::MAX / 4,
            UPPER_FLOOR.1,
        ),
    ] {
        pieces = pieces
            .into_iter()
            .flat_map(|piece| subtract_bounds(piece, band))
            .collect();
    }
    pieces
}

fn build_ladder_shaft(
    composite: &ReservationRecord,
    attr: &SemanticAttribution,
    ir: &mut AssemblyIR,
    next_feature: &mut u32,
) -> Result<VerticalFeature, RichnessError> {
    let (x0, y0, x1, y1) = room_bounds(composite);
    if x1 - x0 < 256 || y1 - y0 < 128 {
        return Err(vertical_error(
            "ladder.envelope",
            "ladder requires a 256x128 host envelope for offset landings",
        ));
    }
    let total_width = 64 + LADDER_OUTER + LADDER_UPPER_LANDING;
    let total_x0 = snap_down((x0 + x1 - total_width) / 2);
    let oy0 = snap_down((y0 + y1 - LADDER_OUTER) / 2);
    let outer_x0 = total_x0 + 64;
    let outer_x1 = outer_x0 + LADDER_OUTER;
    let outer_y0 = oy0;
    let outer_y1 = oy0 + LADDER_OUTER;
    let inner_x0 = outer_x0 + LADDER_WALL;
    let inner_x1 = outer_x1 - LADDER_WALL;
    let inner_y0 = outer_y0 + LADDER_WALL;
    let inner_y1 = outer_y1 - LADDER_WALL;

    let lower_entry = (
        outer_x0,
        inner_y0,
        LOWER_FLOOR_TOP,
        outer_x0 + 16,
        inner_y1,
        LOWER_FLOOR_TOP + HEADROOM,
    );
    let upper_entry = (
        outer_x1 - 16,
        inner_y0,
        UPPER_FLOOR.1,
        outer_x1,
        inner_y1,
        UPPER_FLOOR.1 + HEADROOM,
    );
    let mut shell_wall_ids = Vec::new();
    let west_partition = (
        outer_x0,
        outer_y0 + 16,
        SHELL_Z.0,
        outer_x0 + 16,
        outer_y1 - 16,
        SHELL_Z.1,
    );
    let (west, _) = insert_partition_with_openings(
        ir,
        west_partition,
        BrushAssemblyRole::LadderShaftWall,
        attr,
        &[lower_entry],
    )?;
    shell_wall_ids.extend(west);
    let east_partition = (
        outer_x1 - 16,
        outer_y0 + 16,
        SHELL_Z.0,
        outer_x1,
        outer_y1 - 16,
        SHELL_Z.1,
    );
    let (east, _) = insert_partition_with_openings(
        ir,
        east_partition,
        BrushAssemblyRole::LadderShaftWall,
        attr,
        &[upper_entry],
    )?;
    shell_wall_ids.extend(east);
    shell_wall_ids.push(insert_box(
        ir,
        (
            outer_x0,
            outer_y1 - 16,
            SHELL_Z.0,
            outer_x1,
            outer_y1,
            SHELL_Z.1,
        ),
        BrushAssemblyRole::LadderShaftWall,
        attr,
    )?);

    // The north wall is stacked with accent bands that are real rung brushes;
    // no rung intrudes into the 64x64 clear shaft.
    let mut rung_ids = Vec::new();
    let mut z = SHELL_Z.0;
    while z < SHELL_Z.1 {
        let z1 = (z + 16).min(SHELL_Z.1);
        let rung = z >= 32 && z < UPPER_FLOOR.0 && ((z - 32) / 16) % 2 == 0;
        let role = if rung {
            BrushAssemblyRole::LadderRung
        } else {
            BrushAssemblyRole::LadderShaftWall
        };
        let id = insert_box(
            ir,
            (outer_x0, outer_y0, z, outer_x1, outer_y0 + 16, z1),
            role,
            attr,
        )?;
        if rung {
            rung_ids.push(id);
        } else {
            shell_wall_ids.push(id);
        }
        z = z1;
    }

    // The lower landing replaces a floor patch instead of rising into the
    // exact 64x80 room-portal throat. Its top remains at the frozen floor
    // elevation and the shaft opening begins immediately above it.
    let lower_landing = replace_floor_patch(
        ir,
        (total_x0, outer_y0, 0, outer_x0, outer_y1, LOWER_FLOOR_TOP),
        BrushAssemblyRole::LadderLanding,
        attr,
    )?;
    let mut landing_ids = vec![lower_landing];
    let upper_landing = insert_box(
        ir,
        (
            outer_x1,
            outer_y0,
            UPPER_FLOOR.0,
            outer_x1 + LADDER_UPPER_LANDING,
            outer_y1,
            UPPER_FLOOR.1,
        ),
        BrushAssemblyRole::LadderLanding,
        attr,
    )?;
    landing_ids.push(upper_landing);
    let upper_supports = [
        (
            outer_x1,
            outer_y0,
            LOWER_FLOOR_TOP,
            outer_x1 + 16,
            outer_y0 + 16,
            UPPER_FLOOR.0,
        ),
        (
            outer_x1 + LADDER_UPPER_LANDING - 16,
            outer_y1 - 16,
            LOWER_FLOOR_TOP,
            outer_x1 + LADDER_UPPER_LANDING,
            outer_y1,
            UPPER_FLOOR.0,
        ),
    ];
    for support in upper_supports {
        let base_z = ir
            .brushes
            .values()
            .filter_map(|candidate| candidate.brush.aabb().ok())
            .filter(|(min, max)| {
                max.2 <= support.5
                    && min.0 < support.3
                    && max.0 > support.0
                    && min.1 < support.4
                    && max.1 > support.1
            })
            .map(|(_, max)| max.2)
            .max()
            .unwrap_or(support.2);
        if base_z < support.5 {
            landing_ids.push(insert_box(
                ir,
                (
                    support.0, support.1, base_z, support.3, support.4, support.5,
                ),
                BrushAssemblyRole::VerticalSupport,
                attr,
            )?);
        }
    }

    let paired_hole_upper = (
        outer_x0,
        outer_y0,
        UPPER_FLOOR.0,
        outer_x1 + LADDER_UPPER_LANDING,
        outer_y1,
        UPPER_FLOOR.1,
    );
    let partition_upper = (
        total_x0.max(x0 + Q),
        outer_y0,
        UPPER_FLOOR.0,
        outer_x1 + LADDER_UPPER_LANDING,
        outer_y1,
        UPPER_FLOOR.1,
    );
    let upper_opening_id = carve_slab_opening(
        ir,
        partition_upper,
        paired_hole_upper,
        BrushAssemblyRole::FloorSlab,
        attr,
    )?;
    let lower_opening_id = carve_slab_opening(
        ir,
        with_z(partition_upper, LOWER_CEILING),
        with_z(paired_hole_upper, LOWER_CEILING),
        BrushAssemblyRole::CeilingSlab,
        attr,
    )?;

    let mut lip_ids = Vec::new();
    for lip in [
        (
            outer_x1,
            outer_y0,
            UPPER_FLOOR.1,
            outer_x1 + LADDER_UPPER_LANDING,
            outer_y0 + 16,
            UPPER_FLOOR.1 + GUARD_HEIGHT,
        ),
        (
            outer_x1,
            outer_y1 - 16,
            UPPER_FLOOR.1,
            outer_x1 + LADDER_UPPER_LANDING,
            outer_y1,
            UPPER_FLOOR.1 + GUARD_HEIGHT,
        ),
        (
            outer_x1 + LADDER_UPPER_LANDING - 16,
            inner_y0,
            UPPER_FLOOR.1,
            outer_x1 + LADDER_UPPER_LANDING,
            inner_y1,
            UPPER_FLOOR.1 + GUARD_HEIGHT,
        ),
    ] {
        lip_ids.push(insert_box(ir, lip, BrushAssemblyRole::LadderLip, attr)?);
    }

    let descriptor_id = insert_movement_descriptor(
        ir,
        "climb",
        (
            inner_x0,
            inner_y0,
            LOWER_FLOOR_TOP,
            inner_x1,
            inner_y1,
            UPPER_FLOOR.1,
        ),
        (-1, 0, 0),
        attr,
    )?;

    Ok(VerticalFeature {
        id: alloc_feature(next_feature),
        composite_id: composite.id,
        kind: VerticalFeatureKind::LadderShaft(LadderShaftData {
            shell_wall_ids,
            rung_ids,
            landing_ids,
            lip_ids,
            upper_opening_id,
            lower_opening_id,
            descriptor_id,
        }),
    })
}

fn build_spiral_stair(
    composite: &ReservationRecord,
    attr: &SemanticAttribution,
    ir: &mut AssemblyIR,
    next_feature: &mut u32,
) -> Result<VerticalFeature, RichnessError> {
    validate_generated_spiral_contract()?;
    let (x0, y0, x1, y1) = room_bounds(composite);
    if x1 - x0 < SPIRAL_ENVELOPE + 32 || y1 - y0 < SPIRAL_ENVELOPE + 32 {
        return Err(vertical_error(
            "spiral.envelope",
            "spiral requires a 224x224 envelope plus room-side access",
        ));
    }
    let cx = snap_down((x0 + x1) / 2);
    let cy = snap_down((y0 + y1) / 2);
    let half = SPIRAL_ENVELOPE / 2;
    let ex0 = cx - half;
    let ey0 = cy - half;
    let ex1 = cx + half;
    let ey1 = cy + half;

    let lower_entry = (
        cx + 16,
        ey0,
        LOWER_FLOOR_TOP,
        cx + 80,
        ey0 + 16,
        LOWER_FLOOR_TOP + HEADROOM,
    );
    let upper_entry = (
        ex1 - 16,
        cy - 80,
        UPPER_FLOOR.1,
        ex1,
        cy - 16,
        UPPER_FLOOR.1 + HEADROOM,
    );
    let mut shell_wall_ids = Vec::new();
    for (partition, hole) in [
        (
            (
                ex0 + SPIRAL_CHAMFER,
                ey0,
                SHELL_Z.0,
                ex1 - SPIRAL_CHAMFER,
                ey0 + 16,
                SHELL_Z.1,
            ),
            Some(lower_entry),
        ),
        (
            (
                ex0 + SPIRAL_CHAMFER,
                ey1 - 16,
                SHELL_Z.0,
                ex1 - SPIRAL_CHAMFER,
                ey1,
                SHELL_Z.1,
            ),
            None,
        ),
        (
            (
                ex0,
                ey0 + SPIRAL_CHAMFER,
                SHELL_Z.0,
                ex0 + 16,
                ey1 - SPIRAL_CHAMFER,
                SHELL_Z.1,
            ),
            None,
        ),
        (
            (
                ex1 - 16,
                ey0 + SPIRAL_CHAMFER,
                SHELL_Z.0,
                ex1,
                ey1 - SPIRAL_CHAMFER,
                SHELL_Z.1,
            ),
            Some(upper_entry),
        ),
    ] {
        if let Some(hole) = hole {
            let (ids, _) = insert_partition_with_openings(
                ir,
                partition,
                BrushAssemblyRole::SpiralShellWall,
                attr,
                &[hole],
            )?;
            shell_wall_ids.extend(ids);
        } else {
            shell_wall_ids.push(insert_box(
                ir,
                partition,
                BrushAssemblyRole::SpiralShellWall,
                attr,
            )?);
        }
    }
    for (sx, sy) in [(-1, -1), (1, -1), (-1, 1), (1, 1)] {
        let brush = v3_geometry::make_diagonal_wall(
            (ex0, ex1),
            (ey0, ey1),
            SHELL_Z.0,
            SHELL_Z.1,
            sx,
            sy,
            SPIRAL_CHAMFER,
        )
        .map_err(|error| vertical_error("spiral.shell", format!("{error}")))?;
        richness_geom::validate_brush(&brush)?;
        let id = ir.alloc_brush_id();
        ir.insert_brush(BrushAssembly {
            id,
            brush,
            role: BrushAssemblyRole::SpiralShellWall,
            owner: attr.clone(),
            cost: brush_cost(),
            support: SupportTarget::World,
        });
        shell_wall_ids.push(id);
    }

    let column_id = insert_box(
        ir,
        (cx - 16, cy - 16, SHELL_Z.0, cx + 16, cy + 16, SHELL_Z.1),
        BrushAssemblyRole::SpiralColumn,
        attr,
    )?;
    let mut tread_ids = Vec::new();
    for (index, tread) in SPIRAL_TREAD_TEMPLATE.iter().enumerate() {
        let top = LOWER_FLOOR_TOP + (index as i128 + 1) * STAIR_RISE;
        tread_ids.push(insert_box(
            ir,
            (
                cx + tread.x0,
                cy + tread.y0,
                LOWER_FLOOR_TOP,
                cx + tread.x1,
                cy + tread.y1,
                top,
            ),
            BrushAssemblyRole::SpiralTread,
            attr,
        )?);
    }

    let landing_bounds = (
        cx + 16,
        cy - 80,
        UPPER_FLOOR.0,
        ex1 - 16,
        cy - 16,
        UPPER_FLOOR.1,
    );
    let landing_id = insert_box(ir, landing_bounds, BrushAssemblyRole::SpiralLanding, attr)?;
    for support in [
        (
            cx + 64,
            cy - 80,
            LOWER_FLOOR_TOP,
            cx + 80,
            cy - 64,
            UPPER_FLOOR.0,
        ),
        (
            cx + 80,
            cy - 32,
            LOWER_FLOOR_TOP,
            ex1 - 16,
            cy - 16,
            UPPER_FLOOR.0,
        ),
    ] {
        let base_z = ir
            .brushes
            .values()
            .filter_map(|candidate| candidate.brush.aabb().ok())
            .filter(|(min, max)| {
                max.2 <= support.5
                    && min.0 < support.3
                    && max.0 > support.0
                    && min.1 < support.4
                    && max.1 > support.1
            })
            .map(|(_, max)| max.2)
            .max()
            .unwrap_or(support.2);
        if base_z < support.5 {
            insert_box(
                ir,
                (
                    support.0, support.1, base_z, support.3, support.4, support.5,
                ),
                BrushAssemblyRole::VerticalSupport,
                attr,
            )?;
        }
    }

    let partition_upper = (cx - 96, cy - 80, UPPER_FLOOR.0, ex1, cy, UPPER_FLOOR.1);
    // Include the complete high-tread route and east shell in the omission.
    // The 16-unit west strip sits between the spiral shell and the western
    // treads, so it remains a live owner without stealing standing headroom.
    let hole_upper = (cx - 80, cy - 80, UPPER_FLOOR.0, ex1, cy, UPPER_FLOOR.1);
    let upper_opening_id = carve_slab_opening(
        ir,
        partition_upper,
        hole_upper,
        BrushAssemblyRole::FloorSlab,
        attr,
    )?;
    let lower_opening_id = carve_slab_opening(
        ir,
        with_z(partition_upper, LOWER_CEILING),
        with_z(hole_upper, LOWER_CEILING),
        BrushAssemblyRole::CeilingSlab,
        attr,
    )?;

    Ok(VerticalFeature {
        id: alloc_feature(next_feature),
        composite_id: composite.id,
        kind: VerticalFeatureKind::SpiralStair(SpiralStairData {
            shell_wall_ids,
            column_id,
            tread_ids,
            landing_id,
            upper_opening_id,
            lower_opening_id,
        }),
    })
}

fn validate_generated_spiral_contract() -> Result<(), RichnessError> {
    let valid = generated_content::SPIRAL_LAYER_OFFSET == 192
        && generated_content::SPIRAL_ENVELOPE_MIN == [224, 224]
        && generated_content::SPIRAL_STEP_COUNT == STAIR_TREADS
        && generated_content::SPIRAL_STEP_INDEX == (1u32..=12).collect::<Vec<_>>().as_slice()
        && generated_content::SPIRAL_STEP_RISE
            .iter()
            .all(|rise| *rise == 16)
        && generated_content::SPIRAL_STEP_TREAD_DEPTH
            .iter()
            .all(|depth| *depth == 64)
        && generated_content::SPIRAL_STEP_CENTER_COLUMN
            .iter()
            .all(|column| *column == [32, 32])
        && generated_content::SPIRAL_STEP_ENVELOPE
            .iter()
            .all(|envelope| *envelope == [224, 224])
        && generated_content::SPIRAL_STEP_IS_CONVEX
            .iter()
            .all(|convex| *convex);
    if valid {
        Ok(())
    } else {
        Err(vertical_error(
            "spiral.catalog",
            "generated spiral constants do not match the frozen 12x16/224/32/64 contract",
        ))
    }
}

fn build_vertical_arena(
    composite: &ReservationRecord,
    attr: &SemanticAttribution,
    ir: &mut AssemblyIR,
    next_feature: &mut u32,
) -> Result<VerticalFeature, RichnessError> {
    let (x0, y0, x1, y1) = room_bounds(composite);
    if x1 - x0 < ARENA_MIN_SPAN || y1 - y0 < ARENA_MIN_SPAN {
        return Err(vertical_error(
            "arena.envelope",
            format!("grand_arena requires at least {ARENA_MIN_SPAN}x{ARENA_MIN_SPAN}"),
        ));
    }
    let shell_wall_ids = ir
        .brushes
        .values()
        .filter(|brush| brush.owner.request_id == attr.request_id && brush.role.is_wall())
        .map(|brush| brush.id)
        .collect::<Vec<_>>();
    if shell_wall_ids.len() < 8 {
        return Err(vertical_error(
            "arena.shell",
            "grand_arena requires its existing capped octagonal shell",
        ));
    }

    let cx = snap_down((x0 + x1) / 2);
    let cy = snap_down((y0 + y1) / 2);

    // The gallery slabs sit one quantum above the inter-layer slab.  That
    // keeps the exact FloorSlab ownership partition live beneath the ring and
    // gives the two catwalks a legal, non-overlapping bearing elevation.
    let gallery_z0 = UPPER_FLOOR.1;
    for (role, coordinate, span) in [
        (BrushAssemblyRole::NorthWall, y0 + 16, (x0 + 96, x1 - 96)),
        (BrushAssemblyRole::WestWall, x0 + 16, (y0 + 96, y1 - 96)),
        (BrushAssemblyRole::EastWall, x1 - 16, (y0 + 96, y1 - 96)),
    ] {
        build_balcony_mezzanine(role, coordinate, span, 80, gallery_z0, attr, ir)?;
    }

    // The fourth ring segment is the supported upper access landing.  It is
    // kept inside the octagonal chamfers and meets the south catwalk across a
    // full 64-unit edge; the remaining south side is a stair approach rather
    // than a slab above the player's head.
    let access_x0 = cx - 128;
    let access_x1 = cx - 48;
    let access_y0 = cy + 96;
    let access_y1 = cy + 160;
    let access_balcony = (
        access_x0,
        access_y0,
        gallery_z0,
        access_x1,
        access_y1,
        gallery_z0 + Q,
    );
    insert_box(ir, access_balcony, BrushAssemblyRole::BalconySlab, attr)?;
    insert_box(
        ir,
        (
            access_x0 + Q,
            access_y0,
            LOWER_FLOOR_TOP,
            access_x0 + 2 * Q,
            access_y0 + Q,
            gallery_z0,
        ),
        BrushAssemblyRole::Corbel,
        attr,
    )?;

    let hole_x0 = (cx - 112).min(x0 + 96);
    let partition_upper = (
        hole_x0 - Q,
        cy - 128,
        UPPER_FLOOR.0,
        cx + 128,
        y1 - 16,
        UPPER_FLOOR.1,
    );
    let hole_upper = (
        hole_x0,
        cy - 112,
        UPPER_FLOOR.0,
        cx + 112,
        y1 - 16,
        UPPER_FLOOR.1,
    );
    let upper_void = carve_slab_opening(
        ir,
        partition_upper,
        hole_upper,
        BrushAssemblyRole::FloorSlab,
        attr,
    )?;
    let _ = carve_slab_opening(
        ir,
        with_z(partition_upper, LOWER_CEILING),
        with_z(hole_upper, LOWER_CEILING),
        BrushAssemblyRole::CeilingSlab,
        attr,
    )?;

    let mut catwalk_ids = Vec::new();
    for center in [cy - 48, cy + 48] {
        let catwalk = build_catwalk_bridge(
            cx - 96,
            cx + 96,
            center,
            ROUTE_WIDTH,
            gallery_z0 + 2 * Q,
            CatwalkAxis::X,
            upper_void,
            attr,
            ir,
        )?;
        catwalk_ids.push(catwalk.deck_id);
        catwalk_ids.extend(catwalk.guard_rail_ids);
        catwalk_ids.extend(catwalk.support_ids);
    }

    let central_mass_id = insert_box(
        ir,
        (
            cx - 48,
            cy - 48,
            LOWER_FLOOR_TOP,
            cx + 48,
            cy + 48,
            UPPER_FLOOR.0,
        ),
        BrushAssemblyRole::MonolithSolid,
        attr,
    )?;

    // Twelve integral 16-rise treads climb westward just south of the second
    // catwalk. Their 64-wide route ends against the supported access balcony;
    // the gallery itself supplies the final 16-unit step to its surface.
    let mut access_ids = Vec::new();
    let stair_x0 = access_x1;
    let stair_x1 = stair_x0 + STAIR_TREADS as i128 * STAIR_RUN;
    let stair_y0 = access_y0;
    let stair_y1 = access_y1;
    for step in 0..STAIR_TREADS {
        let sx1 = stair_x1 - step as i128 * STAIR_RUN;
        let sx0 = sx1 - STAIR_RUN;
        let top = LOWER_FLOOR_TOP + (step as i128 + 1) * STAIR_RISE;
        access_ids.push(insert_box(
            ir,
            (sx0, stair_y0, LOWER_FLOOR_TOP, sx1, stair_y1, top),
            BrushAssemblyRole::StairTread,
            attr,
        )?);
        if step + 1 < STAIR_TREADS {
            access_ids.push(insert_box(
                ir,
                (
                    sx0,
                    stair_y1,
                    LOWER_FLOOR_TOP,
                    sx1,
                    stair_y1 + Q,
                    top + GUARD_HEIGHT,
                ),
                BrushAssemblyRole::StairGuard,
                attr,
            )?);
        }
    }

    // Local gate partitions retain live segments while their 64x80 omissions
    // align exactly with the low and high ends of the access route.
    let lower_partition = (
        stair_x1,
        stair_y0 - Q,
        LOWER_FLOOR_TOP,
        stair_x1 + Q,
        stair_y1,
        LOWER_FLOOR_TOP + HEADROOM,
    );
    let lower_entry = (
        stair_x1,
        stair_y0,
        LOWER_FLOOR_TOP,
        stair_x1 + Q,
        stair_y1,
        LOWER_FLOOR_TOP + HEADROOM,
    );
    let (lower_gate_ids, lower_entry_ids) = insert_partition_with_openings(
        ir,
        lower_partition,
        BrushAssemblyRole::ArenaGateWall,
        attr,
        &[lower_entry],
    )?;
    ensure_tread_floor_supports(ir, &lower_gate_ids, attr)?;
    let upper_partition = (
        stair_x0,
        stair_y0,
        UPPER_FLOOR.1,
        stair_x0 + Q,
        stair_y1 + Q,
        UPPER_FLOOR.1 + HEADROOM,
    );
    let upper_entry = (
        stair_x0,
        stair_y0,
        UPPER_FLOOR.1,
        stair_x0 + Q,
        stair_y1,
        UPPER_FLOOR.1 + HEADROOM,
    );
    let (upper_gate_ids, upper_entry_ids) = insert_partition_with_openings(
        ir,
        upper_partition,
        BrushAssemblyRole::ArenaGateWall,
        attr,
        &[upper_entry],
    )?;
    for gate_id in &upper_gate_ids {
        let (bounds, supported) = {
            let gate = ir
                .brushes
                .get(gate_id)
                .ok_or_else(|| vertical_error("arena.gate", "upper gate disappeared"))?;
            (brush_bounds(gate)?, touches_floor(ir, gate)?)
        };
        if !supported {
            let extended = (
                bounds.0,
                bounds.1,
                LOWER_FLOOR_TOP,
                bounds.3,
                bounds.4,
                bounds.5,
            );
            if bounds_overlap_any(ir, extended, &[*gate_id])? {
                return Err(vertical_error(
                    "arena.gate.support",
                    format!(
                        "upper gate {} cannot extend to lower-floor support",
                        gate_id.raw()
                    ),
                ));
            }
            let brush = ConvexBrush::make_box(
                (extended.0, extended.3),
                (extended.1, extended.4),
                (extended.2, extended.5),
            )
            .map_err(|error| vertical_error("arena.gate.support", format!("{error}")))?;
            richness_geom::validate_brush(&brush)?;
            ir.brushes
                .get_mut(gate_id)
                .ok_or_else(|| vertical_error("arena.gate", "upper gate disappeared"))?
                .brush = brush;
        }
    }

    let balcony_ids = ir
        .brushes
        .values()
        .filter(|brush| {
            brush.owner == *attr
                && matches!(
                    brush.role,
                    BrushAssemblyRole::BalconySlab | BrushAssemblyRole::Corbel
                )
        })
        .map(|brush| brush.id)
        .collect();

    Ok(VerticalFeature {
        id: alloc_feature(next_feature),
        composite_id: composite.id,
        kind: VerticalFeatureKind::VerticalArena(VerticalArenaData {
            shell_wall_ids,
            balcony_ids,
            catwalk_ids,
            central_mass_id,
            access_ids,
            lower_entry_ids,
            upper_entry_ids,
        }),
    })
}

// ── Source validators ─────────────────────────────────────────────────────

pub(crate) fn validate_multi_storey_shells(ir: &AssemblyIR) -> Result<(), RichnessError> {
    let owners = ir
        .brushes
        .values()
        .filter_map(|brush| owner_key(&brush.owner))
        .filter(|key| {
            generated_content::ARCHETYPE_LAYER_OCCUPANCY
                .get(key.archetype.raw() as usize)
                .is_some_and(|occupancy| *occupancy == super::content_types::LayerOccupancy::Both)
                // A drop pair spans layers through paired omissions, not a
                // full-height multi-storey room shell.
                && owner_recipe(*key, ir) != VerticalRecipe::DropHole
        })
        .collect::<BTreeSet<_>>();
    for key in owners {
        let brushes = brushes_for_owner(ir, key);
        let shell_count = brushes
            .iter()
            .filter(|brush| {
                brush.role.is_wall()
                    && brush
                        .brush
                        .aabb()
                        .is_ok_and(|(min, max)| min.2 == SHELL_Z.0 && max.2 == SHELL_Z.1)
            })
            .count();
        if shell_count < 4 {
            return Err(missing_owner_role(
                key,
                owner_recipe(key, ir),
                &["four or more full-height multi-storey shell walls"],
            ));
        }
        let capped_floor = brushes.iter().any(|brush| {
            brush.role == BrushAssemblyRole::FloorSlab
                && brush
                    .brush
                    .aabb()
                    .is_ok_and(|(min, max)| min.2 == 0 && max.2 == LOWER_FLOOR_TOP)
        });
        let capped_ceiling = brushes.iter().any(|brush| {
            brush.role == BrushAssemblyRole::CeilingSlab
                && brush
                    .brush
                    .aabb()
                    .is_ok_and(|(min, max)| min.2 == UPPER_CEILING_BOTTOM && max.2 == 368)
        });
        if !capped_floor || !capped_ceiling {
            return Err(missing_owner_role(
                key,
                owner_recipe(key, ir),
                &["capped lower floor and upper ceiling"],
            ));
        }
    }
    Ok(())
}

pub(crate) fn validate_balcony_clearance(ir: &AssemblyIR) -> Result<(), RichnessError> {
    for balcony in ir
        .brushes
        .values()
        .filter(|brush| brush.role == BrushAssemblyRole::BalconySlab)
    {
        let bounds = brush_bounds(balcony)?;
        let width = bounds.3 - bounds.0;
        let depth = bounds.4 - bounds.1;
        if bounds.5 - bounds.2 != Q
            || width < ROUTE_WIDTH
            || depth < ROUTE_WIDTH
            || width.max(depth) < ROUTE_WIDTH + Q
        {
            return Err(owner_error(
                balcony,
                "balcony_dimensions",
                "balcony must be 16 thick with a 64-wide route and at least 80 projection/span",
            ));
        }
        if !has_clear_square(
            ir,
            balcony.owner.request_id,
            bounds.2.max(bounds.5),
            bounds,
            HEADROOM,
            &[balcony.id],
        )? {
            return Err(owner_error(
                balcony,
                "balcony_headroom",
                "balcony has no unobstructed 64x80 witness outside its guards",
            ));
        }
        if !has_support_contact_with_roles(
            ir,
            balcony,
            &[
                BrushAssemblyRole::Corbel,
                BrushAssemblyRole::VerticalSupport,
            ],
        )? {
            return Err(owner_error(
                balcony,
                "balcony_support",
                "balcony has no positive-area corbel/pilaster support",
            ));
        }
    }
    Ok(())
}

pub(crate) fn validate_catwalk_over_void_only(ir: &AssemblyIR) -> Result<(), RichnessError> {
    for deck in ir
        .brushes
        .values()
        .filter(|brush| brush.role == BrushAssemblyRole::CatwalkDeck)
    {
        let bounds = brush_bounds(deck)?;
        if bounds.5 - bounds.2 != Q
            || (bounds.3 - bounds.0).min(bounds.4 - bounds.1) < ROUTE_WIDTH + 2 * Q
        {
            return Err(owner_error(
                deck,
                "catwalk_dimensions",
                "catwalk must be 16 thick with a protected 64-wide route",
            ));
        }
        let opening = ir
            .openings
            .values()
            .find(|opening| {
                opening.owner.request_id == deck.owner.request_id
                    && matches!(opening.wall_role, BrushAssemblyRole::FloorSlab)
                    && bounds.0 >= opening.bounds.0
                    && bounds.1 >= opening.bounds.1
                    && bounds.3 <= opening.bounds.3
                    && bounds.4 <= opening.bounds.4
            })
            .ok_or_else(|| {
                owner_error(
                    deck,
                    "catwalk_void",
                    "catwalk is not contained by a committed slab void",
                )
            })?;
        if opening.wall_segment_ids.is_empty() {
            return Err(owner_error(
                deck,
                "catwalk_void",
                "catwalk opening has no live slab owners",
            ));
        }
        if !has_clear_square(
            ir,
            deck.owner.request_id,
            bounds.5,
            bounds,
            HEADROOM,
            &[deck.id],
        )? {
            return Err(owner_error(
                deck,
                "catwalk_headroom",
                "catwalk has no 64x80 clear route between guards",
            ));
        }
        let support_count = ir
            .brushes
            .values()
            .filter(|parent| {
                matches!(
                    parent.role,
                    BrushAssemblyRole::VerticalSupport
                        | BrushAssemblyRole::BalconySlab
                        | BrushAssemblyRole::MonolithSolid
                ) && super::support::compute_support_contact(deck, parent).is_some_and(|contact| {
                    contact.orientation_valid
                        && contact.contact_area_squared
                            > crate::enhanced_v3::geometry::Rational::ZERO
                })
            })
            .count();
        if support_count < 2 {
            return Err(owner_error(
                deck,
                "catwalk_support",
                format!("catwalk requires two supported ends; found {support_count}"),
            ));
        }
        let rail_count = ir
            .brushes
            .values()
            .filter(|rail| {
                rail.role == BrushAssemblyRole::GuardRail
                    && super::support::compute_support_contact(rail, deck).is_some_and(|contact| {
                        contact.orientation_valid
                            && contact.contact_area_squared
                                > crate::enhanced_v3::geometry::Rational::ZERO
                    })
            })
            .count();
        if rail_count < 2 {
            return Err(owner_error(
                deck,
                "catwalk_guards",
                format!("catwalk requires two solid guard rails; found {rail_count}"),
            ));
        }
    }
    Ok(())
}

pub(crate) fn validate_overlook_sealed(ir: &AssemblyIR) -> Result<(), RichnessError> {
    for opening in ir
        .openings
        .values()
        .filter(|opening| opening.wall_role == BrushAssemblyRole::PartialWall)
    {
        if opening.bounds.2 - opening.owner_partition_bounds.2 < 48
            || opening.bounds.2 - opening.owner_partition_bounds.2 > 64
        {
            return Err(opening_error(
                opening,
                "overlook_sill",
                "overlook sill must be 48-64 units",
            ));
        }
        let sill = opening
            .wall_segment_ids
            .iter()
            .filter_map(|id| ir.brushes.get(id))
            .any(|segment| {
                segment.brush.aabb().is_ok_and(|(min, max)| {
                    min.0 <= opening.bounds.0
                        && max.0 >= opening.bounds.3
                        && max.2 == opening.bounds.2
                })
            });
        if !sill {
            return Err(opening_error(
                opening,
                "overlook_sill",
                "overlook has no live sill segment beneath its omission",
            ));
        }
        if ir.openings.values().any(|other| {
            other.owner.request_id == opening.owner.request_id
                && other.wall_role.is_slab()
                && xy_overlap(other.bounds, opening.bounds)
        }) {
            return Err(opening_error(
                opening,
                "overlook_floor",
                "overlook incorrectly owns a floor hole",
            ));
        }
    }
    Ok(())
}

pub(crate) fn validate_pit_chasm_pairs(ir: &AssemblyIR) -> Result<(), RichnessError> {
    for key in owners_for_recipe(ir, VerticalRecipe::DropHole) {
        let openings = openings_for_owner(ir, key);
        let upper = openings
            .iter()
            .find(|opening| opening.wall_role == BrushAssemblyRole::FloorSlab);
        let lower = openings
            .iter()
            .find(|opening| opening.wall_role == BrushAssemblyRole::CeilingSlab);
        let (upper, lower) = match (upper, lower) {
            (Some(upper), Some(lower)) => (*upper, *lower),
            _ => {
                return Err(missing_owner_role(
                    key,
                    VerticalRecipe::DropHole,
                    &["floor_slab_opening", "ceiling_slab_opening"],
                ))
            }
        };
        if xy_bounds(upper.bounds) != xy_bounds(lower.bounds) {
            return Err(opening_error(
                upper,
                "pit_pair",
                "upper floor and lower ceiling holes do not match exactly",
            ));
        }
        let brushes = brushes_for_owner(ir, key);
        for role in [
            BrushAssemblyRole::PitPerimeterSlab,
            BrushAssemblyRole::DropEntryGuard,
            BrushAssemblyRole::DropLanding,
            BrushAssemblyRole::FloorSlab,
        ] {
            if !brushes.iter().any(|brush| brush.role == role) {
                return Err(missing_owner_role(
                    key,
                    VerticalRecipe::DropHole,
                    &[role.tag()],
                ));
            }
        }
        for role in [
            BrushAssemblyRole::PitPerimeterSlab,
            BrushAssemblyRole::DropEntryGuard,
        ] {
            let count = brushes.iter().filter(|brush| brush.role == role).count();
            if count < 3 {
                return Err(missing_special_role(
                    key,
                    owner_archetype_id(key, ir).unwrap_or("?"),
                    role.tag(),
                    3,
                    count,
                ));
            }
        }
        if brushes
            .iter()
            .any(|brush| brush.role == BrushAssemblyRole::DropShaftWall)
        {
            return Err(missing_owner_role(
                key,
                VerticalRecipe::DropHole,
                &["no unreachable drop_shaft_wall geometry"],
            ));
        }
        let landing = brushes
            .iter()
            .copied()
            .find(|brush| brush.role == BrushAssemblyRole::DropLanding)
            .ok_or_else(|| missing_owner_role(key, VerticalRecipe::DropHole, &["drop_landing"]))?;
        let landing_bounds = brush_bounds(landing)?;
        let landing_top = landing_bounds.5;
        if landing_bounds.3 - landing_bounds.0 != DROP_CLEAR
            || landing_bounds.4 - landing_bounds.1 != DROP_CLEAR
        {
            return Err(owner_error(
                landing,
                "drop_landing",
                "drop landing is not exactly 64x64",
            ));
        }
        let hole_center = (
            (upper.bounds.0 + upper.bounds.3) / 2,
            (upper.bounds.1 + upper.bounds.4) / 2,
        );
        let landing_center = (
            (landing_bounds.0 + landing_bounds.3) / 2,
            (landing_bounds.1 + landing_bounds.4) / 2,
        );
        let offset = (
            (landing_center.0 - hole_center.0).abs(),
            (landing_center.1 - hole_center.1).abs(),
        );
        if !matches!(offset, (DROP_LANDING_OFFSET, 0) | (0, DROP_LANDING_OFFSET)) {
            return Err(owner_error(
                landing,
                "drop_landing_offset",
                "drop landing is not offset 32 units from the shaft center",
            ));
        }
        if upper.bounds.5 - landing_top < 32 {
            return Err(opening_error(
                upper,
                "drop_nonreturn",
                "drop landing is less than 32 units below entry",
            ));
        }
        if !has_clear_square(
            ir,
            landing.owner.request_id,
            landing_top,
            landing_bounds,
            HEADROOM,
            &[landing.id],
        )? {
            return Err(owner_error(
                landing,
                "drop_landing_headroom",
                "drop landing has no 64x80 standing witness",
            ));
        }
        let descriptor = require_descriptor(ir, key, "one_way_drop")?;
        let model = descriptor.brush_model.as_ref().ok_or_else(|| {
            missing_owner_role(key, VerticalRecipe::DropHole, &["one_way_drop brush model"])
        })?;
        if ir
            .brushes
            .values()
            .any(|brush| richness_geom::brushes_overlap(model, &brush.brush).unwrap_or(true))
        {
            return Err(missing_owner_role(
                key,
                VerticalRecipe::DropHole,
                &["unobstructed one_way_drop brush model"],
            ));
        }
    }
    Ok(())
}

pub(crate) fn validate_stairwells(ir: &AssemblyIR) -> Result<(), RichnessError> {
    for key in owners_with_role(ir, BrushAssemblyRole::StairTread) {
        if owner_archetype_id(key, ir) == Some("grand_arena") {
            continue;
        }
        let mut treads = brushes_for_owner(ir, key)
            .into_iter()
            .filter(|brush| brush.role == BrushAssemblyRole::StairTread)
            .collect::<Vec<_>>();
        treads.sort_by_key(|brush| {
            brush
                .brush
                .aabb()
                .ok()
                .map(|(_, max)| max.2)
                .unwrap_or(i128::MAX)
        });
        if treads.len() != STAIR_TREADS {
            return Err(missing_owner_role(
                key,
                owner_recipe(key, ir),
                &["12 ordered stair_treads"],
            ));
        }
        for (index, tread) in treads.iter().enumerate() {
            let bounds = brush_bounds(tread)?;
            let expected_top = LOWER_FLOOR_TOP + (index as i128 + 1) * STAIR_RISE;
            if bounds.5 != expected_top
                || !matches!(
                    (bounds.3 - bounds.0, bounds.4 - bounds.1),
                    (64, 16) | (16, 64)
                )
            {
                return Err(owner_error(
                    tread,
                    "stair_tread_order",
                    format!("tread {index} is not a 64x16 tread at top z={expected_top}"),
                ));
            }
            validate_headroom_over(ir, tread, HEADROOM)?;
            if !touches_floor(ir, tread)? {
                return Err(owner_error(
                    tread,
                    "stair_support",
                    "integral tread/riser has no positive-area floor contact",
                ));
            }
        }
        let landings = brushes_for_owner(ir, key)
            .into_iter()
            .filter(|brush| brush.role == BrushAssemblyRole::StairLanding)
            .collect::<Vec<_>>();
        let lower_ok = landings.iter().any(|landing| {
            landing.brush.aabb().is_ok_and(|(min, max)| {
                max.2 == LOWER_FLOOR_TOP && max.0 - min.0 >= 64 && max.1 - min.1 >= 64
            })
        });
        let turn_ok = landings.iter().any(|landing| {
            landing
                .brush
                .aabb()
                .is_ok_and(|(min, max)| max.2 == 80 && max.0 - min.0 >= 64 && max.1 - min.1 >= 64)
        });
        let upper_ok = landings.iter().any(|landing| {
            landing.brush.aabb().is_ok_and(|(min, max)| {
                max.2 == UPPER_FLOOR.1 && max.0 - min.0 >= 64 && max.1 - min.1 >= 64
            })
        });
        if !lower_ok || !turn_ok || !upper_ok {
            let mut error = missing_owner_role(
                key,
                owner_recipe(key, ir),
                &[
                    "64x64 lower_landing",
                    "64x64 turn_landing",
                    "64x64 upper_landing",
                ],
            );
            error.context.push_str(&format!(
                "; landing checks lower={lower_ok} turn={turn_ok} upper={upper_ok}; bounds={:?}",
                landings
                    .iter()
                    .map(|landing| brush_bounds(landing))
                    .collect::<Vec<_>>()
            ));
            return Err(error);
        }
        for landing in &landings {
            let bounds = brush_bounds(landing)?;
            if !has_clear_square(
                ir,
                landing.owner.request_id,
                bounds.5,
                bounds,
                HEADROOM,
                &[landing.id],
            )? {
                return Err(owner_error(
                    landing,
                    "stair_landing_headroom",
                    format!(
                        "stair landing {:?} has no 64x80 standing witness; treads={:?}; blockers={:?}",
                        bounds,
                        treads
                            .iter()
                            .map(|tread| brush_bounds(tread))
                            .collect::<Vec<_>>(),
                        ir.brushes
                            .values()
                            .filter(|candidate| candidate.id != landing.id)
                            .filter(|candidate| {
                                ConvexBrush::make_box(
                                    (bounds.0, bounds.3),
                                    (bounds.1, bounds.4),
                                    (bounds.5, bounds.5 + HEADROOM),
                                )
                                .is_ok_and(|witness| {
                                    richness_geom::brushes_overlap(&witness, &candidate.brush)
                                        .unwrap_or(false)
                                })
                            })
                            .map(|candidate| (candidate.id.raw(), candidate.role.tag()))
                            .collect::<Vec<_>>()
                    ),
                ));
            }
        }
        validate_paired_owner_openings(ir, key)?;
    }
    Ok(())
}

pub(crate) fn validate_ladder_shafts(ir: &AssemblyIR) -> Result<(), RichnessError> {
    for key in owners_for_recipe(ir, VerticalRecipe::LadderShaft) {
        let brushes = brushes_for_owner(ir, key);
        for role in [
            BrushAssemblyRole::LadderShaftWall,
            BrushAssemblyRole::LadderRung,
            BrushAssemblyRole::LadderLanding,
            BrushAssemblyRole::LadderLip,
        ] {
            if !brushes.iter().any(|brush| brush.role == role) {
                return Err(missing_owner_role(
                    key,
                    VerticalRecipe::LadderShaft,
                    &[role.tag()],
                ));
            }
        }
        let shell = union_bounds(brushes.iter().copied().filter(|brush| {
            matches!(
                brush.role,
                BrushAssemblyRole::LadderShaftWall | BrushAssemblyRole::LadderRung
            )
        }))?;
        if shell.3 - shell.0 != LADDER_OUTER || shell.4 - shell.1 != LADDER_OUTER {
            return Err(missing_owner_role(
                key,
                VerticalRecipe::LadderShaft,
                &["exact 96x96 ladder_shell"],
            ));
        }
        let mut rungs = brushes
            .iter()
            .copied()
            .filter(|brush| brush.role == BrushAssemblyRole::LadderRung)
            .collect::<Vec<_>>();
        rungs.sort_by_key(|rung| {
            brush_bounds(rung)
                .map(|bounds| bounds.2)
                .unwrap_or(i128::MAX)
        });
        if rungs.len() < 5 {
            return Err(missing_owner_role(
                key,
                VerticalRecipe::LadderShaft,
                &["five or more ladder_rungs"],
            ));
        }
        for rung in &rungs {
            let rung_bounds = brush_bounds(rung)?;
            if rung_bounds.5 - rung_bounds.2 != Q {
                return Err(owner_error(
                    rung,
                    "ladder_rung",
                    "ladder rung is not one quantum thick",
                ));
            }
        }
        if rungs.windows(2).any(|pair| {
            let first = brush_bounds(pair[0]).ok();
            let second = brush_bounds(pair[1]).ok();
            !matches!((first, second), (Some(a), Some(b)) if b.2 - a.2 == 2 * Q)
        }) {
            return Err(missing_owner_role(
                key,
                VerticalRecipe::LadderShaft,
                &["32-unit rung spacing"],
            ));
        }
        let descriptor = require_descriptor(ir, key, "climb")?;
        let bounds = descriptor.brush_model_bounds.ok_or_else(|| {
            missing_owner_role(
                key,
                VerticalRecipe::LadderShaft,
                &["climb brush_model_bounds"],
            )
        })?;
        if bounds.3 - bounds.0 != LADDER_CLEAR
            || bounds.4 - bounds.1 != LADDER_CLEAR
            || bounds.5 - bounds.2 < HEADROOM
        {
            return Err(missing_owner_role(
                key,
                VerticalRecipe::LadderShaft,
                &["exact 64x64 climb_clearance with 80 headroom"],
            ));
        }
        let model = descriptor.brush_model.as_ref().ok_or_else(|| {
            missing_owner_role(key, VerticalRecipe::LadderShaft, &["climb brush model"])
        })?;
        if ir
            .brushes
            .values()
            .any(|brush| richness_geom::brushes_overlap(model, &brush.brush).unwrap_or(true))
        {
            return Err(missing_owner_role(
                key,
                VerticalRecipe::LadderShaft,
                &["unobstructed climb brush model"],
            ));
        }
        for landing in brushes
            .iter()
            .copied()
            .filter(|brush| brush.role == BrushAssemblyRole::LadderLanding)
        {
            let landing_bounds = brush_bounds(landing)?;
            if !has_clear_square(
                ir,
                landing.owner.request_id,
                landing_bounds.5,
                landing_bounds,
                HEADROOM,
                &[landing.id],
            )? {
                return Err(owner_error(
                    landing,
                    "ladder_landing",
                    "ladder landing has no 64x80 standing witness",
                ));
            }
        }
        validate_paired_owner_openings(ir, key)?;
    }
    Ok(())
}

pub(crate) fn validate_spiral_stairs(ir: &AssemblyIR) -> Result<(), RichnessError> {
    for key in owners_for_recipe(ir, VerticalRecipe::SpiralStair) {
        let brushes = brushes_for_owner(ir, key);
        let column = brushes
            .iter()
            .find(|brush| brush.role == BrushAssemblyRole::SpiralColumn)
            .ok_or_else(|| {
                missing_owner_role(key, VerticalRecipe::SpiralStair, &["spiral_column"])
            })?;
        let cb = brush_bounds(column)?;
        if cb.3 - cb.0 != SPIRAL_COLUMN || cb.4 - cb.1 != SPIRAL_COLUMN {
            return Err(owner_error(
                column,
                "spiral_column",
                "spiral column is not exactly 32x32",
            ));
        }
        if !touches_floor(ir, column)? {
            return Err(owner_error(
                column,
                "spiral_column_support",
                "spiral center column has no positive-area floor support",
            ));
        }
        let cx = (cb.0 + cb.3) / 2;
        let cy = (cb.1 + cb.4) / 2;
        let mut treads = brushes
            .iter()
            .copied()
            .filter(|brush| brush.role == BrushAssemblyRole::SpiralTread)
            .collect::<Vec<_>>();
        treads.sort_by_key(|brush| {
            brush
                .brush
                .aabb()
                .ok()
                .map(|(_, max)| max.2)
                .unwrap_or(i128::MAX)
        });
        if treads.len() != STAIR_TREADS {
            return Err(missing_owner_role(
                key,
                VerticalRecipe::SpiralStair,
                &["12 spiral_treads"],
            ));
        }
        for (index, tread) in treads.iter().enumerate() {
            let bounds = brush_bounds(tread)?;
            let template = SPIRAL_TREAD_TEMPLATE[index];
            let expected = (
                cx + template.x0,
                cy + template.y0,
                LOWER_FLOOR_TOP,
                cx + template.x1,
                cy + template.y1,
                LOWER_FLOOR_TOP + (index as i128 + 1) * STAIR_RISE,
            );
            if bounds != expected {
                return Err(owner_error(
                    tread,
                    "spiral_template",
                    format!("tread {index} differs from the integer template"),
                ));
            }
            validate_headroom_over(ir, tread, HEADROOM)?;
            if !touches_floor(ir, tread)? {
                return Err(owner_error(
                    tread,
                    "spiral_support",
                    "spiral tread/riser does not reach the floor",
                ));
            }
        }
        if treads
            .windows(2)
            .any(|pair| !richness_geom::has_positive_area_contact(&pair[0].brush, &pair[1].brush))
        {
            return Err(missing_owner_role(
                key,
                VerticalRecipe::SpiralStair,
                &["continuous consecutive spiral tread contacts"],
            ));
        }
        let shell = brushes
            .iter()
            .copied()
            .filter(|brush| brush.role == BrushAssemblyRole::SpiralShellWall)
            .collect::<Vec<_>>();
        if shell.len() < 8 {
            return Err(missing_owner_role(
                key,
                VerticalRecipe::SpiralStair,
                &["eight-wall chamfered spiral shell"],
            ));
        }
        let envelope = union_bounds(shell.iter().copied())?;
        if envelope.3 - envelope.0 != SPIRAL_ENVELOPE || envelope.4 - envelope.1 != SPIRAL_ENVELOPE
        {
            return Err(missing_owner_role(
                key,
                VerticalRecipe::SpiralStair,
                &["224x224 spiral_envelope"],
            ));
        }
        let diagonal_count = shell
            .iter()
            .filter(|brush| {
                brush
                    .brush
                    .faces
                    .iter()
                    .any(|face| face.plane.nx != 0 && face.plane.ny != 0)
            })
            .count();
        if diagonal_count != 4 {
            return Err(missing_owner_role(
                key,
                VerticalRecipe::SpiralStair,
                &["four exact 45-degree chamfer walls"],
            ));
        }
        let landing = brushes
            .iter()
            .copied()
            .find(|brush| brush.role == BrushAssemblyRole::SpiralLanding)
            .ok_or_else(|| {
                missing_owner_role(key, VerticalRecipe::SpiralStair, &["spiral_landing"])
            })?;
        let landing_bounds = brush_bounds(landing)?;
        if !has_clear_square(
            ir,
            landing.owner.request_id,
            landing_bounds.5,
            landing_bounds,
            HEADROOM,
            &[landing.id],
        )? {
            return Err(owner_error(
                landing,
                "spiral_landing_headroom",
                "spiral landing has no 64x80 standing witness",
            ));
        }
        if !richness_geom::has_positive_area_contact(
            &treads[STAIR_TREADS - 1].brush,
            &landing.brush,
        ) {
            return Err(missing_owner_role(
                key,
                VerticalRecipe::SpiralStair,
                &["final tread to upper landing contact"],
            ));
        }
        validate_paired_owner_openings(ir, key)?;
    }
    Ok(())
}

/// DropHole is the authored pit/chasm recipe.  It deliberately does not
/// require unreachable `DropShaftWall` roles.
pub(crate) fn validate_drop_shafts(ir: &AssemblyIR) -> Result<(), RichnessError> {
    validate_pit_chasm_pairs(ir)
}

pub(crate) fn validate_vertical_arena(ir: &AssemblyIR) -> Result<(), RichnessError> {
    for key in owners_for_archetype(ir, "grand_arena") {
        let brushes = brushes_for_owner(ir, key);
        let requirements = [
            (BrushAssemblyRole::BalconySlab, 4usize),
            (BrushAssemblyRole::GuardRail, 4),
            (BrushAssemblyRole::Corbel, 4),
            (BrushAssemblyRole::CatwalkDeck, 2),
            (BrushAssemblyRole::MonolithSolid, 1),
            (BrushAssemblyRole::StairTread, 12),
            (BrushAssemblyRole::StairGuard, 11),
            (BrushAssemblyRole::ArenaGateWall, 2),
        ];
        for (role, minimum) in requirements {
            let count = brushes.iter().filter(|brush| brush.role == role).count();
            if count < minimum {
                return Err(missing_special_role(
                    key,
                    "grand_arena",
                    role.tag(),
                    minimum,
                    count,
                ));
            }
        }
        let shell_count = brushes
            .iter()
            .filter(|brush| {
                brush.role.is_wall()
                    && brush
                        .brush
                        .aabb()
                        .is_ok_and(|(min, max)| min.2 == SHELL_Z.0 && max.2 == SHELL_Z.1)
            })
            .count();
        if shell_count < 8 {
            return Err(missing_special_role(
                key,
                "grand_arena",
                "full_height_shell_wall",
                8,
                shell_count,
            ));
        }

        let mut treads = brushes
            .iter()
            .copied()
            .filter(|brush| brush.role == BrushAssemblyRole::StairTread)
            .collect::<Vec<_>>();
        treads.sort_by_key(|brush| {
            brush
                .brush
                .aabb()
                .ok()
                .map(|(_, max)| max.2)
                .unwrap_or(i128::MAX)
        });
        for (index, tread) in treads.iter().enumerate() {
            let bounds = brush_bounds(tread)?;
            let expected_top = LOWER_FLOOR_TOP + (index as i128 + 1) * STAIR_RISE;
            if bounds.5 != expected_top
                || !matches!(
                    (bounds.3 - bounds.0, bounds.4 - bounds.1),
                    (64, 16) | (16, 64)
                )
            {
                return Err(owner_error(
                    tread,
                    "arena_access",
                    format!(
                        "arena tread {index} is not a 64x16 integral riser at z={expected_top}"
                    ),
                ));
            }
            validate_headroom_over(ir, tread, HEADROOM)?;
            if !touches_floor(ir, tread)? {
                return Err(owner_error(
                    tread,
                    "arena_access",
                    "arena tread has no positive-area floor support",
                ));
            }
        }
        validate_paired_owner_openings(ir, key)?;

        let entries = openings_for_owner(ir, key)
            .into_iter()
            .filter(|opening| opening.wall_role == BrushAssemblyRole::ArenaGateWall)
            .collect::<Vec<_>>();
        let lower = entries
            .iter()
            .filter(|opening| {
                opening.bounds.2 == LOWER_FLOOR_TOP
                    && opening.bounds.5 - opening.bounds.2 == HEADROOM
            })
            .count();
        let upper = entries
            .iter()
            .filter(|opening| {
                opening.bounds.2 == UPPER_FLOOR.1 && opening.bounds.5 - opening.bounds.2 == HEADROOM
            })
            .count();
        if lower == 0 || upper == 0 {
            return Err(missing_owner_role(
                key,
                VerticalRecipe::None,
                &[
                    "controlled lower-band arena entry",
                    "controlled upper-band arena entry",
                ],
            ));
        }
    }
    Ok(())
}

pub(crate) fn validate_vertical_features(ir: &AssemblyIR) -> Result<(), RichnessError> {
    validate_multi_storey_shells(ir)?;
    validate_slab_opening_ownership(ir)?;
    validate_balcony_clearance(ir)?;
    validate_catwalk_over_void_only(ir)?;
    validate_overlook_sealed(ir)?;
    validate_pit_chasm_pairs(ir)?;
    validate_stairwells(ir)?;
    validate_ladder_shafts(ir)?;
    validate_spiral_stairs(ir)?;
    validate_vertical_arena(ir)
}

/// Validate exact live-segment ownership for every vertical slab omission.
pub(crate) fn validate_slab_opening_ownership(ir: &AssemblyIR) -> Result<(), RichnessError> {
    let mut groups: BTreeMap<
        (BrushAssemblyRole, Bounds, SemanticAttribution),
        Vec<&OpeningRecord>,
    > = BTreeMap::new();
    for opening in ir
        .openings
        .values()
        .filter(|opening| opening.wall_role.is_slab())
    {
        groups
            .entry((
                opening.wall_role,
                opening.owner_partition_bounds,
                opening.owner.clone(),
            ))
            .or_default()
            .push(opening);
    }
    for ((role, partition, owner), openings) in groups {
        let ids = openings[0].wall_segment_ids.clone();
        if ids.is_empty() || !ids.contains(&openings[0].owner_brush_id) {
            return Err(opening_error(
                openings[0],
                "slab_owner",
                "slab opening has no live owner segments",
            ));
        }
        for opening in &openings {
            if opening.wall_segment_ids != ids || opening.owner != owner {
                return Err(opening_error(
                    opening,
                    "slab_owner",
                    "paired openings disagree about their live slab partition",
                ));
            }
        }
        for x in (partition.0..partition.3).step_by(Q as usize) {
            for y in (partition.1..partition.4).step_by(Q as usize) {
                let cell = (x, y, partition.2, x + Q, y + Q, partition.5);
                let omitted = openings
                    .iter()
                    .any(|opening| bounds_contains(opening.bounds, cell));
                // A landing may deliberately replace a portion of a slab
                // opening. It is then the sole final partition owner, rather
                // than an intruder into an otherwise empty omission.
                let covers = |brush: &super::assembly::BrushAssembly| {
                    brush.owner == owner
                        && brush.brush.aabb().is_ok_and(|(min, max)| {
                            bounds_contains((min.0, min.1, min.2, max.0, max.1, max.2), cell)
                        })
                };
                let replacement_owners = ir
                    .brushes
                    .values()
                    .filter(|brush| {
                        matches!(
                            brush.role,
                            BrushAssemblyRole::StairLanding
                                | BrushAssemblyRole::SpiralLanding
                                | BrushAssemblyRole::DropLanding
                                | BrushAssemblyRole::LadderLanding
                        ) && covers(brush)
                    })
                    .count();
                let owners = ids
                    .iter()
                    .filter(|id| {
                        ir.brushes.get(id).is_some_and(|brush| {
                            brush.role == role
                                && brush.owner == owner
                                && brush.brush.aabb().is_ok_and(|(min, max)| {
                                    bounds_contains(
                                        (min.0, min.1, min.2, max.0, max.1, max.2),
                                        cell,
                                    )
                                })
                        })
                    })
                    .count()
                    + replacement_owners;
                let final_omission = omitted && replacement_owners == 0;
                if (final_omission && owners != 0) || (!final_omission && owners != 1) {
                    return Err(opening_error(
                        openings[0],
                        "slab_omission",
                        format!("cell ({x},{y}) has {owners} owners; omitted={omitted}"),
                    ));
                }
            }
        }
    }
    Ok(())
}

// ── Partition helpers ─────────────────────────────────────────────────────

fn replace_floor_patch(
    ir: &mut AssemblyIR,
    patch: Bounds,
    replacement_role: BrushAssemblyRole,
    preferred_owner: &SemanticAttribution,
) -> Result<BrushAssemblyId, RichnessError> {
    if !bounds_quantum_aligned(patch) {
        return Err(vertical_error(
            "floor.patch",
            format!("patch {patch:?} is not quantum aligned"),
        ));
    }
    let source = ir
        .brushes
        .values()
        .filter(|brush| {
            brush.role == BrushAssemblyRole::FloorSlab
                && brush.owner.request_id == preferred_owner.request_id
        })
        .filter_map(|brush| {
            let bounds = brush_bounds(brush).ok()?;
            bounds_contains(bounds, patch).then_some((bounds_volume(bounds), brush.id))
        })
        .min_by_key(|(volume, _)| *volume)
        .map(|(_, id)| id);

    let Some(source) = source else {
        return insert_box(ir, patch, replacement_role, preferred_owner);
    };
    let old = ir.remove_brush(source).ok_or_else(|| {
        vertical_error(
            "floor.patch",
            format!("missing floor slab {}", source.raw()),
        )
    })?;
    remove_brush_references(ir, source);
    let bounds = brush_bounds(&old)?;
    for piece in subtract_bounds(bounds, patch) {
        insert_box_with_cost(
            ir,
            piece,
            BrushAssemblyRole::FloorSlab,
            &old.owner,
            old.cost,
        )?;
    }
    insert_box_with_cost(ir, patch, replacement_role, &old.owner, old.cost)
}

fn carve_slab_opening(
    ir: &mut AssemblyIR,
    partition: Bounds,
    hole: Bounds,
    role: BrushAssemblyRole,
    preferred_owner: &SemanticAttribution,
) -> Result<OpeningAssemblyId, RichnessError> {
    validate_partition_hole(partition, hole)?;
    if !role.is_slab() {
        return Err(vertical_error(
            "slab.role",
            format!("{} is not a slab role", role.tag()),
        ));
    }

    let existing_group = ir
        .openings
        .values()
        .filter(|opening| {
            opening.wall_role == role
                && opening.owner.request_id == preferred_owner.request_id
                && bounds_contains(opening.owner_partition_bounds, partition)
        })
        .min_by_key(|opening| bounds_volume(opening.owner_partition_bounds))
        .map(|opening| {
            (
                opening.owner_partition_bounds,
                opening.wall_segment_ids.clone(),
            )
        });

    let (effective_partition, mut source_ids) = if let Some(group) = existing_group {
        group
    } else {
        let containing = ir
            .brushes
            .values()
            .filter(|brush| {
                brush.role == role && brush.owner.request_id == preferred_owner.request_id
            })
            .filter_map(|brush| {
                let bounds = brush_bounds(brush).ok()?;
                (bounds.2 == partition.2
                    && bounds.5 == partition.5
                    && bounds_contains(bounds, partition))
                .then_some((bounds, brush.id))
            })
            .min_by_key(|(bounds, _)| bounds_volume(*bounds));
        if let Some((bounds, id)) = containing {
            (bounds, vec![id])
        } else {
            let ids = ir
                .brushes
                .values()
                .filter(|brush| {
                    brush.role == role
                        && brush.owner.request_id == preferred_owner.request_id
                        && brush.brush.aabb().is_ok_and(|(min, max)| {
                            let bounds = (min.0, min.1, min.2, max.0, max.1, max.2);
                            bounds.2 == partition.2
                                && bounds.5 == partition.5
                                && bounds.0 >= partition.0
                                && bounds.1 >= partition.1
                                && bounds.3 <= partition.3
                                && bounds.4 <= partition.4
                        })
                })
                .map(|brush| brush.id)
                .collect();
            (partition, ids)
        }
    };
    validate_partition_hole(effective_partition, hole)?;

    // A prior opening may have split the same semantic slab into fragments
    // whose bounds extend beyond this feature's local partition. Include
    // every overlapping live fragment before re-partitioning; otherwise a
    // later vertical feature can insert a new ceiling/floor piece through an
    // earlier live segment.
    source_ids.extend(
        ir.brushes
            .values()
            .filter(|brush| {
                brush.role == role && brush.owner.request_id == preferred_owner.request_id
            })
            .filter(|brush| {
                brush_bounds(brush)
                    .is_ok_and(|bounds| bounds_intersection(bounds, effective_partition).is_some())
            })
            .map(|brush| brush.id),
    );
    source_ids.sort_unstable();
    source_ids.dedup();

    let owner = source_ids
        .first()
        .and_then(|id| ir.brushes.get(id))
        .map_or_else(|| preferred_owner.clone(), |brush| brush.owner.clone());
    if source_ids.iter().any(|id| {
        ir.brushes
            .get(id)
            .is_none_or(|brush| brush.owner != owner || brush.role != role)
    }) {
        return Err(vertical_error(
            "slab.owner",
            "slab partition mixes semantic owners or roles",
        ));
    }

    let mut replacements = Vec::new();
    if source_ids.is_empty() {
        for piece in subtract_bounds(effective_partition, hole) {
            replacements.push(insert_box(ir, piece, role, &owner)?);
        }
    } else {
        for id in source_ids {
            let old = ir.remove_brush(id).ok_or_else(|| {
                vertical_error("slab.split", format!("missing slab segment {}", id.raw()))
            })?;
            remove_brush_references(ir, id);
            let bounds = brush_bounds(&old)?;
            for piece in subtract_bounds(bounds, hole) {
                replacements.push(insert_box_with_cost(ir, piece, role, &owner, old.cost)?);
            }
        }
    }
    replacements.sort_unstable();
    if replacements.is_empty() {
        return Err(vertical_error(
            "slab.split",
            "opening consumed the complete slab partition; a live owner segment is required",
        ));
    }

    for opening in ir.openings.values_mut().filter(|opening| {
        opening.wall_role == role
            && opening.owner_partition_bounds == effective_partition
            && opening.owner == owner
    }) {
        opening.wall_segment_ids = replacements.clone();
        opening.owner_brush_id = replacements[0];
    }

    let id = ir.alloc_opening_id();
    ir.insert_opening(OpeningRecord {
        id,
        owner_brush_id: replacements[0],
        wall_segment_ids: replacements.clone(),
        owner_partition_bounds: effective_partition,
        wall_role: role,
        owner: owner.clone(),
        bounds: hole,
        portal_id: None,
        frame_brush_ids: Vec::new(),
        portal_style: None,
    });
    if role == BrushAssemblyRole::CeilingSlab {
        ensure_ceiling_supports(ir, &replacements, &owner, &[])?;
    }
    Ok(id)
}

fn insert_partition_with_openings(
    ir: &mut AssemblyIR,
    partition: Bounds,
    role: BrushAssemblyRole,
    owner: &SemanticAttribution,
    holes: &[Bounds],
) -> Result<(Vec<BrushAssemblyId>, Vec<OpeningAssemblyId>), RichnessError> {
    let mut pieces = vec![partition];
    for hole in holes {
        validate_partition_hole(partition, *hole)?;
        pieces = pieces
            .into_iter()
            .flat_map(|piece| subtract_bounds(piece, *hole))
            .collect();
    }
    if pieces.is_empty() {
        return Err(vertical_error(
            "partition",
            "openings consumed the complete owner partition",
        ));
    }
    let mut segment_ids = Vec::new();
    for piece in pieces {
        segment_ids.push(insert_box(ir, piece, role, owner)?);
    }
    segment_ids.sort_unstable();
    let mut opening_ids = Vec::new();
    for hole in holes {
        let id = ir.alloc_opening_id();
        ir.insert_opening(OpeningRecord {
            id,
            owner_brush_id: segment_ids[0],
            wall_segment_ids: segment_ids.clone(),
            owner_partition_bounds: partition,
            wall_role: role,
            owner: owner.clone(),
            bounds: *hole,
            portal_id: None,
            frame_brush_ids: Vec::new(),
            portal_style: None,
        });
        opening_ids.push(id);
    }
    Ok((segment_ids, opening_ids))
}

fn ensure_ceiling_supports(
    ir: &mut AssemblyIR,
    segment_ids: &[BrushAssemblyId],
    owner: &SemanticAttribution,
    excluded: &[Bounds],
) -> Result<(), RichnessError> {
    for id in segment_ids {
        let (bounds, supported) = {
            let segment = ir
                .brushes
                .get(id)
                .ok_or_else(|| vertical_error("ceiling.support", "missing ceiling segment"))?;
            let supported = ir.brushes.values().any(|candidate| {
                candidate.id != *id
                    && super::support::compute_support_contact(segment, candidate).is_some_and(
                        |contact| {
                            contact.orientation_valid
                                && contact.contact_area_squared
                                    > crate::enhanced_v3::geometry::Rational::ZERO
                        },
                    )
            });
            (brush_bounds(segment)?, supported)
        };
        if supported {
            continue;
        }
        let mut placed = false;
        'cells: for x in (bounds.0..bounds.3).step_by(Q as usize) {
            for y in (bounds.1..bounds.4).step_by(Q as usize) {
                let x1 = (x + Q).min(bounds.3);
                let y1 = (y + Q).min(bounds.4);
                let base_z = ir
                    .brushes
                    .values()
                    .filter_map(|candidate| candidate.brush.aabb().ok())
                    .filter(|(min, max)| {
                        max.2 <= bounds.2 && min.0 < x1 && max.0 > x && min.1 < y1 && max.1 > y
                    })
                    .map(|(_, max)| max.2)
                    .max()
                    .unwrap_or(LOWER_FLOOR_TOP);
                let column = (x, y, base_z, x1, y1, bounds.2);
                let intersects_witness = excluded
                    .iter()
                    .any(|witness| bounds_intersection(column, *witness).is_some());
                if base_z < bounds.2
                    && !intersects_witness
                    && !bounds_overlap_any(ir, column, &[*id])?
                {
                    insert_box(ir, column, BrushAssemblyRole::VerticalSupport, owner)?;
                    placed = true;
                    break 'cells;
                }
            }
        }
        if !placed {
            return Err(vertical_error(
                "ceiling.support",
                format!(
                    "ceiling segment {} has no legal positive-area support",
                    id.raw()
                ),
            ));
        }
    }
    Ok(())
}

fn subtract_bounds(piece: Bounds, hole: Bounds) -> Vec<Bounds> {
    let Some(i) = bounds_intersection(piece, hole) else {
        return vec![piece];
    };
    let candidates = [
        (piece.0, piece.1, piece.2, i.0, piece.4, piece.5),
        (i.3, piece.1, piece.2, piece.3, piece.4, piece.5),
        (i.0, piece.1, piece.2, i.3, i.1, piece.5),
        (i.0, i.4, piece.2, i.3, piece.4, piece.5),
        (i.0, i.1, piece.2, i.3, i.4, i.2),
        (i.0, i.1, i.5, i.3, i.4, piece.5),
    ];
    candidates
        .into_iter()
        .filter(|bounds| bounds.0 < bounds.3 && bounds.1 < bounds.4 && bounds.2 < bounds.5)
        .collect()
}

fn bounds_intersection(a: Bounds, b: Bounds) -> Option<Bounds> {
    let result = (
        a.0.max(b.0),
        a.1.max(b.1),
        a.2.max(b.2),
        a.3.min(b.3),
        a.4.min(b.4),
        a.5.min(b.5),
    );
    (result.0 < result.3 && result.1 < result.4 && result.2 < result.5).then_some(result)
}

fn validate_partition_hole(partition: Bounds, hole: Bounds) -> Result<(), RichnessError> {
    if !bounds_contains(partition, hole)
        || !bounds_quantum_aligned(partition)
        || !bounds_quantum_aligned(hole)
    {
        return Err(vertical_error(
            "partition",
            format!("hole {hole:?} is not a quantum-aligned subset of {partition:?}"),
        ));
    }
    Ok(())
}

fn bounds_contains(outer: Bounds, inner: Bounds) -> bool {
    inner.0 >= outer.0
        && inner.1 >= outer.1
        && inner.2 >= outer.2
        && inner.3 <= outer.3
        && inner.4 <= outer.4
        && inner.5 <= outer.5
}

fn bounds_volume(bounds: Bounds) -> i128 {
    (bounds.3 - bounds.0)
        .saturating_mul(bounds.4 - bounds.1)
        .saturating_mul(bounds.5 - bounds.2)
}

fn bounds_quantum_aligned(bounds: Bounds) -> bool {
    [bounds.0, bounds.1, bounds.2, bounds.3, bounds.4, bounds.5]
        .into_iter()
        .all(|value| value.rem_euclid(Q) == 0)
}

fn insert_box(
    ir: &mut AssemblyIR,
    bounds: Bounds,
    role: BrushAssemblyRole,
    owner: &SemanticAttribution,
) -> Result<BrushAssemblyId, RichnessError> {
    insert_box_with_cost(ir, bounds, role, owner, brush_cost())
}

fn insert_box_with_cost(
    ir: &mut AssemblyIR,
    bounds: Bounds,
    role: BrushAssemblyRole,
    owner: &SemanticAttribution,
    cost: CostSource,
) -> Result<BrushAssemblyId, RichnessError> {
    if !bounds_quantum_aligned(bounds) {
        return Err(vertical_error(
            "box.grid",
            format!("bounds {bounds:?} are not quantum aligned"),
        ));
    }
    let brush = ConvexBrush::make_box(
        (bounds.0, bounds.3),
        (bounds.1, bounds.4),
        (bounds.2, bounds.5),
    )
    .map_err(|error| vertical_error("box", format!("{error}")))?;
    richness_geom::validate_brush(&brush)?;
    for existing in ir.brushes.values() {
        if richness_geom::brushes_overlap(&brush, &existing.brush)? {
            return Err(vertical_error(
                "box.overlap",
                format!(
                    "new {} brush {bounds:?} owner {:?} overlaps brush {} ({}, {:?}, owner {:?})",
                    role.tag(),
                    owner.reservation_id.raw(),
                    existing.id.raw(),
                    existing.role.tag(),
                    existing.brush.aabb().ok(),
                    existing.owner.reservation_id.raw(),
                ),
            ));
        }
    }
    let id = ir.alloc_brush_id();
    ir.insert_brush(BrushAssembly {
        id,
        brush,
        role,
        owner: owner.clone(),
        cost,
        support: SupportTarget::World,
    });
    Ok(id)
}

fn insert_movement_descriptor(
    ir: &mut AssemblyIR,
    kind: &str,
    bounds: Bounds,
    normal: (i128, i128, i128),
    owner: &SemanticAttribution,
) -> Result<EntityAssemblyId, RichnessError> {
    if !matches!(kind, "climb" | "one_way_drop") || !bounds_quantum_aligned(bounds) {
        return Err(vertical_error(
            "descriptor",
            "invalid movement descriptor kind or bounds",
        ));
    }
    let brush_model = ConvexBrush::make_box(
        (bounds.0, bounds.3),
        (bounds.1, bounds.4),
        (bounds.2, bounds.5),
    )
    .map_err(|error| vertical_error("descriptor.model", format!("{error}")))?;
    richness_geom::validate_brush(&brush_model)?;
    let id = ir.alloc_entity_id();
    let mut keys = BTreeMap::new();
    keys.insert(
        "convention_revision".to_string(),
        CONVENTION_REVISION.to_string(),
    );
    keys.insert("richness_volume".to_string(), kind.to_string());
    keys.insert(
        "richness_volume_id".to_string(),
        format!("vertical-{}-{}", owner.reservation_id.raw(), id.raw()),
    );
    let normal_text = format!("{} {} {}", normal.0, normal.1, normal.2);
    if kind == "climb" {
        keys.insert("climb_normal".to_string(), normal_text);
        keys.insert("climb_priority".to_string(), "1".to_string());
    } else {
        keys.insert("drop_direction".to_string(), "down".to_string());
        keys.insert("entry_normal".to_string(), normal_text);
        keys.insert("one_way".to_string(), "1".to_string());
    }
    // The brush model remains at its authored traversal volume, while qbsp
    // uses the entity origin as a fill seed.  Anchor that seed in the adjacent
    // lower landing rather than at the center of an open shaft/pit volume.
    // Both recipes therefore retain their exact runtime bounds without ever
    // introducing a point entity into the inter-layer void.
    let origin = match kind {
        "climb" => (
            bounds.0 - 48,
            (bounds.1 + bounds.4) / 2,
            LOWER_FLOOR_TOP + 24,
        ),
        "one_way_drop" => (bounds.3, (bounds.1 + bounds.4) / 2, LOWER_FLOOR_TOP + 24),
        _ => unreachable!("kind checked above"),
    };
    ir.insert_entity(EntityAssembly {
        id,
        classname: "trigger_multiple".to_string(),
        origin,
        owner: owner.clone(),
        cost: brush_cost(),
        keys,
        brush_model_bounds: Some(bounds),
        brush_model: Some(brush_model),
    });
    Ok(id)
}

fn remove_brush_references(ir: &mut AssemblyIR, id: BrushAssemblyId) {
    ir.supports
        .retain(|_, support| support.child != id && support.parent != SupportTarget::Brush(id));
    ir.interfaces
        .retain(|_, interface| interface.brush_a != id && interface.brush_b != id);
    for opening in ir.openings.values_mut() {
        opening.wall_segment_ids.retain(|segment| *segment != id);
        opening.frame_brush_ids.retain(|segment| *segment != id);
    }
}

fn bounds_overlap_any(
    ir: &AssemblyIR,
    bounds: Bounds,
    ignore: &[BrushAssemblyId],
) -> Result<bool, RichnessError> {
    let candidate = ConvexBrush::make_box(
        (bounds.0, bounds.3),
        (bounds.1, bounds.4),
        (bounds.2, bounds.5),
    )
    .map_err(|error| vertical_error("overlap", format!("{error}")))?;
    for brush in ir
        .brushes
        .values()
        .filter(|brush| !ignore.contains(&brush.id))
    {
        if richness_geom::brushes_overlap(&candidate, &brush.brush)? {
            return Ok(true);
        }
    }
    Ok(false)
}

fn brush_cost() -> CostSource {
    CostSource {
        dimension: BudgetDimension::SourceFaces,
        face_count: 6,
    }
}

fn room_bounds(record: &ReservationRecord) -> (i128, i128, i128, i128) {
    richness_geom::footprint_quake_bounds(&record.footprint)
}

fn with_z(bounds: Bounds, z: (i128, i128)) -> Bounds {
    (bounds.0, bounds.1, z.0, bounds.3, bounds.4, z.1)
}

fn snap_down(value: i128) -> i128 {
    value.div_euclid(Q) * Q
}

fn require_bounds_inside(inner: Bounds, outer: Bounds, path: &str) -> Result<(), RichnessError> {
    if bounds_contains(outer, inner) {
        Ok(())
    } else {
        Err(vertical_error(path, format!("{inner:?} escapes {outer:?}")))
    }
}

fn alloc_feature(next: &mut u32) -> VerticalFeatureId {
    let id = VerticalFeatureId::new(*next);
    *next = next.saturating_add(1);
    id
}

fn attribution_for(
    record: &ReservationRecord,
    request_archetypes: &BTreeMap<ArchetypeRequestId, ArchetypeIndex>,
) -> Result<SemanticAttribution, RichnessError> {
    let request_id = record.request_id.ok_or_else(|| {
        vertical_error(
            "owner",
            format!(
                "{} reservation {} has no request",
                record.kind.tag(),
                record.id.raw()
            ),
        )
    })?;
    let archetype = request_archetypes
        .get(&request_id)
        .copied()
        .ok_or_else(|| {
            vertical_error(
                "owner",
                format!("request {} has no archetype", request_id.raw()),
            )
        })?;
    Ok(SemanticAttribution::from_reservation(
        record.id,
        Some(request_id),
        Some(archetype),
        record.beat_id,
        record.zone_id,
    ))
}

// ── Validator helpers ─────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct OwnerKey {
    reservation: ReservationId,
    request: ArchetypeRequestId,
    archetype: ArchetypeIndex,
}

fn owner_key(owner: &SemanticAttribution) -> Option<OwnerKey> {
    Some(OwnerKey {
        reservation: owner.reservation_id,
        request: owner.request_id?,
        archetype: owner.archetype?,
    })
}

fn owners_for_recipe(ir: &AssemblyIR, recipe: VerticalRecipe) -> BTreeSet<OwnerKey> {
    ir.brushes
        .values()
        .filter_map(|brush| {
            let key = owner_key(&brush.owner)?;
            (owner_recipe(key, ir) == recipe).then_some(key)
        })
        .collect()
}

fn owners_with_role(ir: &AssemblyIR, role: BrushAssemblyRole) -> BTreeSet<OwnerKey> {
    ir.brushes
        .values()
        .filter(|brush| brush.role == role)
        .filter_map(|brush| owner_key(&brush.owner))
        .collect()
}

fn owners_for_archetype(ir: &AssemblyIR, archetype_id: &str) -> BTreeSet<OwnerKey> {
    ir.brushes
        .values()
        .filter_map(|brush| {
            let key = owner_key(&brush.owner)?;
            (generated_content::ARCHETYPE_IDS
                .get(key.archetype.raw() as usize)
                .copied()
                == Some(archetype_id))
            .then_some(key)
        })
        .collect()
}

fn owner_recipe(key: OwnerKey, _ir: &AssemblyIR) -> VerticalRecipe {
    generated_content::ARCHETYPE_VERTICAL_RECIPE
        .get(key.archetype.raw() as usize)
        .copied()
        .unwrap_or(VerticalRecipe::None)
}

fn owner_archetype_id(key: OwnerKey, _ir: &AssemblyIR) -> Option<&'static str> {
    generated_content::ARCHETYPE_IDS
        .get(key.archetype.raw() as usize)
        .copied()
}

fn brushes_for_owner(ir: &AssemblyIR, key: OwnerKey) -> Vec<&BrushAssembly> {
    ir.brushes
        .values()
        .filter(|brush| owner_key(&brush.owner) == Some(key))
        .collect()
}

fn openings_for_owner(ir: &AssemblyIR, key: OwnerKey) -> Vec<&OpeningRecord> {
    ir.openings
        .values()
        .filter(|opening| owner_key(&opening.owner) == Some(key))
        .collect()
}

fn require_descriptor<'a>(
    ir: &'a AssemblyIR,
    key: OwnerKey,
    kind: &str,
) -> Result<&'a EntityAssembly, RichnessError> {
    ir.entities
        .values()
        .find(|entity| {
            let model_matches_bounds = entity
                .brush_model
                .as_ref()
                .and_then(|model| model.aabb().ok())
                .zip(entity.brush_model_bounds)
                .is_some_and(|((min, max), bounds)| {
                    bounds == (min.0, min.1, min.2, max.0, max.1, max.2)
                });
            owner_key(&entity.owner) == Some(key)
                && entity.classname == "trigger_multiple"
                && entity
                    .keys
                    .get("richness_volume")
                    .is_some_and(|value| value == kind)
                && entity
                    .keys
                    .get("convention_revision")
                    .is_some_and(|value| value == CONVENTION_REVISION)
                && model_matches_bounds
        })
        .ok_or_else(|| {
            missing_owner_role(
                key,
                owner_recipe(key, ir),
                &[if kind == "climb" {
                    "climb trigger_multiple brush-model descriptor"
                } else {
                    "one_way_drop trigger_multiple brush-model descriptor"
                }],
            )
        })
}

fn validate_paired_owner_openings(ir: &AssemblyIR, key: OwnerKey) -> Result<(), RichnessError> {
    let openings = openings_for_owner(ir, key);
    let upper = openings
        .iter()
        .find(|opening| opening.wall_role == BrushAssemblyRole::FloorSlab);
    let lower = openings
        .iter()
        .find(|opening| opening.wall_role == BrushAssemblyRole::CeilingSlab);
    let (Some(upper), Some(lower)) = (upper, lower) else {
        return Err(missing_owner_role(
            key,
            owner_recipe(key, ir),
            &["floor_slab_opening", "ceiling_slab_opening"],
        ));
    };
    // The lower-ceiling opening may be wider than the upper-floor opening
    // where a turn or approach needs clearance below the retained upper slab.
    // It must still fully contain the upper-floor projection; pit/drop pairs
    // retain their stricter exact-match validation separately.
    let lower_contains_upper = lower.bounds.0 <= upper.bounds.0
        && lower.bounds.1 <= upper.bounds.1
        && lower.bounds.3 >= upper.bounds.3
        && lower.bounds.4 >= upper.bounds.4;
    if !lower_contains_upper {
        return Err(opening_error(
            upper,
            "paired_openings",
            "lower-ceiling omission does not contain the upper-floor omission",
        ));
    }
    Ok(())
}

fn validate_headroom_over(
    ir: &AssemblyIR,
    brush: &BrushAssembly,
    height: i128,
) -> Result<(), RichnessError> {
    let bounds = brush_bounds(brush)?;
    let witness = (
        bounds.0,
        bounds.1,
        bounds.5,
        bounds.3,
        bounds.4,
        bounds.5 + height,
    );
    if let Some(blocker) = ir.brushes.values().find(|candidate| {
        candidate.id != brush.id
            && ConvexBrush::make_box(
                (witness.0, witness.3),
                (witness.1, witness.4),
                (witness.2, witness.5),
            )
            .is_ok_and(|volume| {
                richness_geom::brushes_overlap(&volume, &candidate.brush).unwrap_or(true)
            })
    }) {
        return Err(owner_error(
            brush,
            "headroom",
            format!(
                "{} has less than {height} units of headroom due to {} {}",
                brush.role.tag(),
                blocker.id.raw(),
                blocker.role.tag(),
            ),
        ));
    }
    Ok(())
}

fn has_clear_square(
    ir: &AssemblyIR,
    owner: Option<ArchetypeRequestId>,
    surface_z: i128,
    bounds: Bounds,
    height: i128,
    ignore: &[BrushAssemblyId],
) -> Result<bool, RichnessError> {
    if bounds.3 - bounds.0 < ROUTE_WIDTH || bounds.4 - bounds.1 < ROUTE_WIDTH {
        return Ok(false);
    }
    for x in (bounds.0..=bounds.3 - ROUTE_WIDTH).step_by(Q as usize) {
        for y in (bounds.1..=bounds.4 - ROUTE_WIDTH).step_by(Q as usize) {
            let witness = (
                x,
                y,
                surface_z,
                x + ROUTE_WIDTH,
                y + ROUTE_WIDTH,
                surface_z + height,
            );
            let blocked = ir
                .brushes
                .values()
                .filter(|brush| !ignore.contains(&brush.id))
                .any(|brush| {
                    // Other owners still block physical headroom; the owner value is
                    // retained only to make this search explicitly per-owner.
                    let _same_owner = brush.owner.request_id == owner;
                    let test = ConvexBrush::make_box(
                        (witness.0, witness.3),
                        (witness.1, witness.4),
                        (witness.2, witness.5),
                    );
                    test.is_ok_and(|test| {
                        richness_geom::brushes_overlap(&test, &brush.brush).unwrap_or(true)
                    })
                });
            if !blocked {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

fn has_support_contact_with_roles(
    ir: &AssemblyIR,
    child: &BrushAssembly,
    roles: &[BrushAssemblyRole],
) -> Result<bool, RichnessError> {
    for parent in ir
        .brushes
        .values()
        .filter(|parent| roles.contains(&parent.role))
    {
        if let Some(contact) = super::support::compute_support_contact(child, parent) {
            if contact.orientation_valid
                && contact.contact_area_squared > crate::enhanced_v3::geometry::Rational::ZERO
            {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

fn touches_floor(ir: &AssemblyIR, child: &BrushAssembly) -> Result<bool, RichnessError> {
    has_support_contact_with_roles(ir, child, &[BrushAssemblyRole::FloorSlab])
}

fn union_bounds<'a>(
    brushes: impl Iterator<Item = &'a BrushAssembly>,
) -> Result<Bounds, RichnessError> {
    let mut iter = brushes;
    let first = iter
        .next()
        .ok_or_else(|| vertical_error("bounds", "empty brush set"))?;
    let mut result = brush_bounds(first)?;
    for brush in iter {
        let b = brush_bounds(brush)?;
        result = (
            result.0.min(b.0),
            result.1.min(b.1),
            result.2.min(b.2),
            result.3.max(b.3),
            result.4.max(b.4),
            result.5.max(b.5),
        );
    }
    Ok(result)
}

fn brush_bounds(brush: &BrushAssembly) -> Result<Bounds, RichnessError> {
    brush
        .brush
        .aabb()
        .map(|(min, max)| (min.0, min.1, min.2, max.0, max.1, max.2))
        .map_err(|error| vertical_error("bounds", format!("brush {}: {error}", brush.id.raw())))
}

fn xy_bounds(bounds: Bounds) -> (i128, i128, i128, i128) {
    (bounds.0, bounds.1, bounds.3, bounds.4)
}

fn xy_overlap(a: Bounds, b: Bounds) -> bool {
    a.0 < b.3 && b.0 < a.3 && a.1 < b.4 && b.1 < a.4
}

fn owner_error(brush: &BrushAssembly, tag: &str, context: impl Into<String>) -> RichnessError {
    vertical_error(
        &format!(
            "owner.{}.{}",
            brush.owner.archetype_id_str().unwrap_or("?"),
            tag
        ),
        context,
    )
}

fn opening_error(opening: &OpeningRecord, tag: &str, context: impl Into<String>) -> RichnessError {
    vertical_error(
        &format!(
            "opening.{}.{}",
            opening.owner.archetype_id_str().unwrap_or("?"),
            tag
        ),
        context,
    )
}

fn missing_owner_role(key: OwnerKey, recipe: VerticalRecipe, tags: &[&str]) -> RichnessError {
    let archetype = generated_content::ARCHETYPE_IDS
        .get(key.archetype.raw() as usize)
        .copied()
        .unwrap_or("?");
    vertical_error(
        "catalog_roles",
        format!(
            "archetype {archetype} ({recipe:?}) is missing roles/openings/entities [{}]",
            tags.join(",")
        ),
    )
}

fn missing_special_role(
    key: OwnerKey,
    archetype: &str,
    role: &str,
    expected: usize,
    actual: usize,
) -> RichnessError {
    vertical_error(
        "special_roles",
        format!("archetype {archetype} request {} is missing role {role}: expected >= {expected}, actual {actual}", key.request.raw()),
    )
}

fn vertical_error(path: &str, context: impl Into<String>) -> RichnessError {
    RichnessError::new(
        RichnessErrorCode::SemanticInfeasible,
        0,
        "?",
        "?",
        "?",
        "?",
        "?",
        "?",
        "?",
        &format!("vertical.{path}"),
        RichnessErrorCategory::SemanticInfeasibility,
        context,
    )
}

#[cfg(test)]
mod tests {
    use super::super::ids::{BeatId, ZoneId};
    use super::*;

    fn attr(archetype: u32) -> SemanticAttribution {
        SemanticAttribution::from_reservation(
            ReservationId::new(1),
            Some(ArchetypeRequestId::new(1)),
            Some(ArchetypeIndex::new(archetype)),
            Some(BeatId::new(1)),
            Some(ZoneId::new(1)),
        )
    }

    fn floor(ir: &mut AssemblyIR, owner: &SemanticAttribution, bounds: Bounds) {
        insert_box(ir, bounds, BrushAssemblyRole::FloorSlab, owner).unwrap();
    }

    fn octagonal_shell(
        ir: &mut AssemblyIR,
        owner: &SemanticAttribution,
        width: i128,
        depth: i128,
    ) -> Vec<BrushAssemblyId> {
        floor(ir, owner, (0, 0, 0, width, depth, LOWER_FLOOR_TOP));
        insert_box(
            ir,
            (0, 0, UPPER_CEILING_BOTTOM, width, depth, 368),
            BrushAssemblyRole::CeilingSlab,
            owner,
        )
        .unwrap();

        let chamfer = ((width.min(depth) / 4) & !(Q - 1)).clamp(48, 96);
        let mut walls = Vec::new();
        for (bounds, role) in [
            (
                (chamfer, 0, SHELL_Z.0, width - chamfer, Q, SHELL_Z.1),
                BrushAssemblyRole::SouthWall,
            ),
            (
                (
                    chamfer,
                    depth - Q,
                    SHELL_Z.0,
                    width - chamfer,
                    depth,
                    SHELL_Z.1,
                ),
                BrushAssemblyRole::NorthWall,
            ),
            (
                (0, chamfer, SHELL_Z.0, Q, depth - chamfer, SHELL_Z.1),
                BrushAssemblyRole::WestWall,
            ),
            (
                (
                    width - Q,
                    chamfer,
                    SHELL_Z.0,
                    width,
                    depth - chamfer,
                    SHELL_Z.1,
                ),
                BrushAssemblyRole::EastWall,
            ),
        ] {
            walls.push(insert_box(ir, bounds, role, owner).unwrap());
        }
        for (sx, sy, role) in [
            (-1, -1, BrushAssemblyRole::DiagSWWall),
            (1, -1, BrushAssemblyRole::DiagSEWall),
            (-1, 1, BrushAssemblyRole::DiagNWWall),
            (1, 1, BrushAssemblyRole::DiagNEWall),
        ] {
            let brush = v3_geometry::make_diagonal_wall(
                (0, width),
                (0, depth),
                SHELL_Z.0,
                SHELL_Z.1,
                sx,
                sy,
                chamfer,
            )
            .unwrap();
            let id = ir.alloc_brush_id();
            ir.insert_brush(BrushAssembly {
                id,
                brush,
                role,
                owner: owner.clone(),
                cost: brush_cost(),
                support: SupportTarget::World,
            });
            walls.push(id);
        }
        walls
    }

    fn record(id: u32, kind: ReservationKind, fp: Footprint3D) -> ReservationRecord {
        ReservationRecord {
            id: ReservationId::new(id),
            kind,
            footprint: fp,
            beat_id: Some(BeatId::new(1)),
            request_id: Some(ArchetypeRequestId::new(1)),
            zone_id: Some(ZoneId::new(1)),
            pit_pair_room_id: None,
            composite_children: Vec::new(),
            owning_route_id: None,
            clearance_height: None,
            committed: true,
            cost_faces: 0,
            cost_brushes: 0,
            cost_entities: 0,
            cost_lights: 0,
        }
    }

    #[test]
    fn spiral_template_is_integer_only_and_exact() {
        assert_eq!(SPIRAL_TREAD_TEMPLATE.len(), 12);
        for tread in SPIRAL_TREAD_TEMPLATE {
            assert!(matches!(
                (tread.x1 - tread.x0, tread.y1 - tread.y0),
                (SPIRAL_RADIAL_DEPTH, Q) | (Q, SPIRAL_RADIAL_DEPTH)
            ));
            assert!(bounds_quantum_aligned((
                tread.x0, tread.y0, 16, tread.x1, tread.y1, 32
            )));
        }
        validate_generated_spiral_contract().unwrap();
    }

    #[test]
    fn slab_opening_has_live_exact_owners() {
        let mut ir = AssemblyIR::new();
        let owner = attr(12);
        floor(&mut ir, &owner, (0, 0, 192, 192, 192, 208));
        let id = carve_slab_opening(
            &mut ir,
            (0, 0, 192, 192, 192, 208),
            (64, 64, 192, 128, 128, 208),
            BrushAssemblyRole::FloorSlab,
            &owner,
        )
        .unwrap();
        let opening = &ir.openings[&id];
        assert!(!opening.wall_segment_ids.is_empty());
        assert!(opening.wall_segment_ids.contains(&opening.owner_brush_id));
        validate_slab_opening_ownership(&ir).unwrap();
    }

    #[test]
    fn slab_openings_split_a_containing_live_slab_and_reuse_its_partition() {
        let mut ir = AssemblyIR::new();
        let owner = attr(12);
        floor(&mut ir, &owner, (0, 0, 192, 256, 256, 208));
        let first = carve_slab_opening(
            &mut ir,
            (48, 48, 192, 144, 144, 208),
            (64, 64, 192, 128, 128, 208),
            BrushAssemblyRole::FloorSlab,
            &owner,
        )
        .unwrap();
        let second = carve_slab_opening(
            &mut ir,
            (112, 112, 192, 208, 208, 208),
            (128, 128, 192, 192, 192, 208),
            BrushAssemblyRole::FloorSlab,
            &owner,
        )
        .unwrap();
        assert_eq!(
            ir.openings[&first].owner_partition_bounds,
            (0, 0, 192, 256, 256, 208)
        );
        assert_eq!(
            ir.openings[&first].wall_segment_ids,
            ir.openings[&second].wall_segment_ids
        );
        validate_slab_opening_ownership(&ir).unwrap();
    }

    #[test]
    fn slab_opening_rejects_full_partition_hole() {
        let mut ir = AssemblyIR::new();
        let owner = attr(12);
        let error = carve_slab_opening(
            &mut ir,
            (0, 0, 192, 64, 64, 208),
            (0, 0, 192, 64, 64, 208),
            BrushAssemblyRole::FloorSlab,
            &owner,
        )
        .unwrap_err();
        assert!(error.context.contains("live owner"));
    }

    #[test]
    fn movement_descriptor_uses_frozen_compiler_contract() {
        let mut ir = AssemblyIR::new();
        let owner = attr(17);
        let id = insert_movement_descriptor(
            &mut ir,
            "climb",
            (0, 0, 16, 64, 64, 208),
            (1, 0, 0),
            &owner,
        )
        .unwrap();
        let entity = &ir.entities[&id];
        assert_eq!(entity.classname, "trigger_multiple");
        assert_eq!(entity.keys["richness_volume"], "climb");
        assert_eq!(entity.keys["convention_revision"], CONVENTION_REVISION);
        assert_eq!(entity.brush_model_bounds, Some((0, 0, 16, 64, 64, 208)));
        assert_eq!(
            entity.brush_model.as_ref().unwrap().aabb().unwrap(),
            ((0, 0, 16), (64, 64, 208)),
        );
    }

    #[test]
    fn stairwell_has_twelve_ordered_rises_and_three_landings() {
        let fp = Footprint3D::dual_layer(0, 0, 320, 448);
        let composite = record(0, ReservationKind::Composite, fp);
        let host = record(
            1,
            ReservationKind::VerticalHost,
            Footprint3D::dual_layer(32, 32, 128, 128),
        );
        let owner = attr(12);
        let mut ir = AssemblyIR::new();
        floor(&mut ir, &owner, (0, 0, 0, 320, 448, 16));
        floor(&mut ir, &owner, (0, 0, 352, 320, 448, 368));
        let mut next = 0;
        let feature =
            build_stairwell(&composite, &host, &owner, false, &mut ir, &mut next).unwrap();
        let VerticalFeatureKind::Stairwell(data) = feature.kind else {
            panic!("wrong kind")
        };
        assert_eq!(data.tread_ids.len(), 12);
        assert_eq!(data.landing_ids.len(), 3);
        validate_slab_opening_ownership(&ir).unwrap();
        validate_stairwells(&ir).unwrap();
    }

    #[test]
    fn open_stairwell_materializes_guarded_complete_route() {
        let fp = Footprint3D::dual_layer(0, 0, 320, 448);
        let composite = record(0, ReservationKind::Composite, fp);
        let host = record(
            1,
            ReservationKind::VerticalHost,
            Footprint3D::dual_layer(32, 32, 128, 128),
        );
        let owner = attr(18);
        let mut ir = AssemblyIR::new();
        floor(&mut ir, &owner, (0, 0, 0, 320, 448, 16));
        floor(&mut ir, &owner, (0, 0, 352, 320, 448, 368));
        let mut next = 0;
        let feature = build_stairwell(&composite, &host, &owner, true, &mut ir, &mut next).unwrap();
        let VerticalFeatureKind::OpenStairwell(data) = feature.kind else {
            panic!("wrong kind")
        };
        assert_eq!(data.tread_ids.len(), STAIR_TREADS);
        assert_eq!(data.landing_ids.len(), 3);
        assert!(data.guard_ids.len() >= STAIR_TREADS);
        for id in data
            .tread_ids
            .iter()
            .chain(&data.landing_ids)
            .chain(&data.guard_ids)
        {
            assert!(bounds_contains(
                (0, 0, 0, 320, 448, 368),
                brush_bounds(&ir.brushes[id]).unwrap()
            ));
        }
        validate_slab_opening_ownership(&ir).unwrap();
        validate_stairwells(&ir).unwrap();
    }

    #[test]
    fn open_stairwell_is_wall_clear_in_both_octagonal_host_orientations() {
        for (width, depth) in [(352, 512), (512, 352)] {
            let room = record(
                0,
                ReservationKind::MultiStoreyRoom,
                Footprint3D::dual_layer(0, 0, width, depth),
            );
            let owner = attr(18);
            let mut ir = AssemblyIR::new();
            let wall_ids = octagonal_shell(&mut ir, &owner, i128::from(width), i128::from(depth));
            let mut next = 0;
            let feature = build_stairwell(&room, &room, &owner, true, &mut ir, &mut next).unwrap();
            let VerticalFeatureKind::OpenStairwell(data) = feature.kind else {
                panic!("wrong kind")
            };
            let room_bounds = (0, 0, 0, i128::from(width), i128::from(depth), 368);
            for id in data
                .tread_ids
                .iter()
                .chain(&data.landing_ids)
                .chain(&data.guard_ids)
            {
                let brush = &ir.brushes[id];
                assert!(bounds_contains(room_bounds, brush_bounds(brush).unwrap()));
                assert!(wall_ids.iter().all(|wall_id| {
                    !richness_geom::brushes_overlap(&brush.brush, &ir.brushes[wall_id].brush)
                        .unwrap()
                }));
            }
            validate_slab_opening_ownership(&ir).unwrap();
            validate_stairwells(&ir).unwrap();
        }
    }

    #[test]
    fn ladder_is_96_shell_64_clear_with_real_descriptor() {
        let fp = Footprint3D::dual_layer(0, 0, 256, 256);
        let composite = record(0, ReservationKind::Composite, fp);
        let owner = attr(17);
        let mut ir = AssemblyIR::new();
        floor(&mut ir, &owner, (0, 0, 0, 256, 256, 16));
        floor(&mut ir, &owner, (0, 0, 352, 256, 256, 368));
        let mut next = 0;
        let feature = build_ladder_shaft(&composite, &owner, &mut ir, &mut next).unwrap();
        let VerticalFeatureKind::LadderShaft(data) = feature.kind else {
            panic!("wrong kind")
        };
        assert!(!data.rung_ids.is_empty());
        assert_eq!(
            ir.entities[&data.descriptor_id].keys["richness_volume"],
            "climb"
        );
        assert_eq!(
            ir.entities[&data.descriptor_id]
                .brush_model_bounds
                .unwrap()
                .2,
            LOWER_FLOOR_TOP,
        );
        validate_slab_opening_ownership(&ir).unwrap();
        validate_ladder_shafts(&ir).unwrap();
    }

    #[test]
    fn spiral_builder_materializes_the_complete_supported_route() {
        let fp = Footprint3D::dual_layer(0, 0, 256, 256);
        let composite = record(0, ReservationKind::Composite, fp);
        let owner = attr(24);
        let mut ir = AssemblyIR::new();
        floor(&mut ir, &owner, (0, 0, 0, 256, 256, 16));
        floor(&mut ir, &owner, (0, 0, 352, 256, 256, 368));
        let mut next = 0;
        let feature = build_spiral_stair(&composite, &owner, &mut ir, &mut next).unwrap();
        let VerticalFeatureKind::SpiralStair(data) = feature.kind else {
            panic!("wrong kind")
        };
        assert_eq!(data.tread_ids.len(), STAIR_TREADS);
        assert!(data.shell_wall_ids.len() >= 8);
        for (index, id) in data.tread_ids.iter().enumerate() {
            validate_headroom_over(&ir, &ir.brushes[id], HEADROOM)
                .unwrap_or_else(|error| panic!("spiral tread {index} headroom failed: {error}"));
        }
        validate_slab_opening_ownership(&ir).unwrap();
        validate_spiral_stairs(&ir).unwrap();
    }

    #[test]
    fn bridge_builder_materializes_a_supported_guarded_catwalk() {
        for (width, depth, axis) in [(256, 160, CatwalkAxis::X), (160, 256, CatwalkAxis::Y)] {
            let fp = Footprint3D::dual_layer(0, 0, width, depth);
            let composite = record(0, ReservationKind::Composite, fp);
            let owner = attr(4);
            let mut ir = AssemblyIR::new();
            floor(
                &mut ir,
                &owner,
                (0, 0, 0, i128::from(width), i128::from(depth), 16),
            );
            floor(
                &mut ir,
                &owner,
                (0, 0, 352, i128::from(width), i128::from(depth), 368),
            );
            let mut next = 0;
            let feature = build_bridge_crossing(&composite, &owner, &mut ir, &mut next).unwrap();
            let VerticalFeatureKind::Catwalk(data) = feature.kind else {
                panic!("wrong kind")
            };
            let deck = brush_bounds(&ir.brushes[&data.deck_id]).unwrap();
            assert_eq!((deck.3 - deck.0 > deck.4 - deck.1), axis == CatwalkAxis::X,);
            assert_eq!(data.guard_rail_ids.len(), 2);
            assert_eq!(data.support_ids.len(), 2);
            validate_slab_opening_ownership(&ir).unwrap();
            validate_catwalk_over_void_only(&ir).unwrap();
        }
    }

    #[test]
    fn grand_arena_materializes_complete_non_overlapping_access() {
        for span in [384, 448] {
            let fp = Footprint3D::dual_layer(0, 0, span, span);
            let composite = record(0, ReservationKind::Composite, fp);
            let owner = attr(11);
            let mut ir = AssemblyIR::new();
            let span_q = i128::from(span);
            floor(&mut ir, &owner, (0, 0, 0, span_q, span_q, 16));
            insert_box(
                &mut ir,
                (0, 0, 352, span_q, span_q, 368),
                BrushAssemblyRole::CeilingSlab,
                &owner,
            )
            .unwrap();
            for bounds in [
                (64, 0, 16, span_q - 64, 16, 352),
                (64, span_q - 16, 16, span_q - 64, span_q, 352),
                (0, 64, 16, 16, span_q - 64, 352),
                (span_q - 16, 64, 16, span_q, span_q - 64, 352),
            ] {
                insert_box(&mut ir, bounds, BrushAssemblyRole::UpperShellWall, &owner).unwrap();
            }
            for (sx, sy) in [(-1, -1), (1, -1), (-1, 1), (1, 1)] {
                let brush = v3_geometry::make_diagonal_wall(
                    (0, span_q),
                    (0, span_q),
                    SHELL_Z.0,
                    SHELL_Z.1,
                    sx,
                    sy,
                    64,
                )
                .unwrap();
                let id = ir.alloc_brush_id();
                ir.insert_brush(BrushAssembly {
                    id,
                    brush,
                    role: BrushAssemblyRole::UpperShellWall,
                    owner: owner.clone(),
                    cost: brush_cost(),
                    support: SupportTarget::World,
                });
            }
            let mut next = 0;
            let feature = build_vertical_arena(&composite, &owner, &mut ir, &mut next).unwrap();
            let VerticalFeatureKind::VerticalArena(data) = feature.kind else {
                panic!("wrong kind")
            };
            assert_eq!(
                data.catwalk_ids
                    .iter()
                    .filter(|id| ir.brushes.contains_key(id))
                    .count(),
                data.catwalk_ids.len()
            );
            validate_multi_storey_shells(&ir).unwrap();
            validate_slab_opening_ownership(&ir).unwrap();
            validate_balcony_clearance(&ir).unwrap();
            validate_catwalk_over_void_only(&ir).unwrap();
            validate_vertical_arena(&ir).unwrap();
        }
    }

    #[test]
    fn drop_hole_materializes_pit_not_unreachable_shaft_roles() {
        let fp = Footprint3D::dual_layer(0, 0, 320, 320);
        let composite = record(0, ReservationKind::Composite, fp);
        let pit = record(
            1,
            ReservationKind::PitOmission,
            Footprint3D::single_layer(0, 0, 320, 320, 1),
        );
        let owner = attr(21);
        let mut ir = AssemblyIR::new();
        floor(&mut ir, &owner, (0, 0, 0, 320, 320, 16));
        insert_box(
            &mut ir,
            (0, 0, 160, 320, 320, 176),
            BrushAssemblyRole::CeilingSlab,
            &owner,
        )
        .unwrap();
        let mut next = 0;
        let feature = build_pit_chasm_pair(&composite, &pit, &owner, &mut ir, &mut next).unwrap();
        let VerticalFeatureKind::PitChasm(data) = feature.kind else {
            panic!("wrong kind")
        };
        let descriptor = &ir.entities[&data.descriptor_id];
        assert_eq!(descriptor.keys["drop_direction"], "down");
        assert_eq!(
            descriptor.brush_model_bounds,
            Some((128, 128, 128, 192, 192, 208)),
        );
        assert!(!ir
            .brushes
            .values()
            .any(|brush| brush.role == BrushAssemblyRole::DropShaftWall));
        validate_slab_opening_ownership(&ir).unwrap();
        validate_pit_chasm_pairs(&ir).unwrap();
    }

    #[test]
    fn overlook_is_a_wall_omission_without_floor_hole() {
        for (width, depth) in [(224, 192), (384, 320)] {
            let fp = Footprint3D::dual_layer(0, 0, width, depth);
            let composite = record(0, ReservationKind::Composite, fp);
            let owner = attr(20);
            let mut ir = AssemblyIR::new();
            let mut next = 0;
            build_overlook_hall(&composite, &owner, &mut ir, &mut next).unwrap();
            validate_overlook_sealed(&ir).unwrap();
            assert!(!ir
                .openings
                .values()
                .any(|opening| opening.wall_role.is_slab()));
        }
    }

    #[test]
    fn missing_ladder_role_returns_typed_archetype_error() {
        let owner = attr(17);
        let mut ir = AssemblyIR::new();
        insert_box(
            &mut ir,
            (0, 0, 16, 96, 16, 352),
            BrushAssemblyRole::LadderShaftWall,
            &owner,
        )
        .unwrap();
        let error = validate_ladder_shafts(&ir).unwrap_err();
        assert_eq!(error.code, RichnessErrorCode::SemanticInfeasible);
        assert!(error.context.contains("ladder_hub"));
        assert!(error.context.contains("ladder_rung"));
    }
}
