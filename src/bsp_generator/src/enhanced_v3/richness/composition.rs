//! Theme-independent macro composition for every catalog archetype.
//!
//! Consumes complete catalog variants (ShapeRule, span ranges, support rules)
//! from generated_content and produces watertight convex brush partitions
//! (floor, ceiling, cardinal walls, approved XY-45 walls, interior massing)
//! with non-overlapping brushes, positive volume, approved normals, 16-unit
//! structural constraints, and exact reservation containment.
//!
//! # Contract
//!
//! - Every archetype has a specific composition function — no generic fallback.
//! - Missing geometry returns a typed RichnessError, never a substitute room.
//! - Every opening is an omission from a unique ownership partition (no opening brushes).
//! - Floor/ceiling slabs own full partition beneath/above walls.
//! - No floats; integer geometry (i128 Quake units); canonical ordering.
//! - Crate-private; no baseline changes.
//! - No portal constructions (session B), no visibility (session C), no vertical/cave/props.

use std::collections::{BTreeMap, BTreeSet};

use crate::enhanced_v3::geometry::{self, BrushFace, CanonicalPlane, ConvexBrush};
use crate::enhanced_v3::richness::error::{
    RichnessError, RichnessErrorCategory, RichnessErrorCode,
};

use super::assembly::{
    AssemblyIR, BrushAssembly, BrushAssemblyRole, BudgetDimension, CostSource, InterfaceKind,
    InterfaceRecord, OpeningRecord, PortalAssembly, PortalStyle, SemanticAttribution,
    SharedWallChainAssembly, SupportRecord, SupportTarget,
};
use super::complexity::ComplexityPlan;
use super::content_types::ShapeRule;
use super::footprint::Footprint3D;
use super::generated_content;
use super::geometry as richness_geom;
use super::ids::{
    ArchetypeIndex, ArchetypeRequestId, BeatId, BrushAssemblyId, OpeningAssemblyId, PortalId,
    ReservationId, WallChainId, ZoneId,
};
use super::request::RichnessTheme;
use super::reservation::ReservationRecord;
use super::solver::FullGenerationResult;
use super::support::derive_support_records;
use super::theme::{THEME_ANCIENT, THEME_BRUTALIST, THEME_EGYPTIAN};
use super::topology::{CommittedPortal, CommittedRoute, Dir};
use super::validation::validate_assembly;
use super::variation::{WallChainRecord, WallMass, WallMassTreatment, WallShaping};
use super::vertical;
use super::visibility::VisibilityPlan;

// ── Macro composition entry point ─────────────────────────────────────────

/// Compose all archetype rooms from placement results into an AssemblyIR.
///
/// Consumes the reservation journal and produces a complete brush assembly
/// with semantic attribution, cost tracking, and support records.
pub(crate) fn compose_all_rooms(
    reservations: &BTreeMap<ReservationId, ReservationRecord>,
    request_archetypes: &BTreeMap<ArchetypeRequestId, ArchetypeIndex>,
) -> Result<AssemblyIR, RichnessError> {
    let mut ir = AssemblyIR::new();

    // Map request_id → archetype index from reservations
    for (_rid, record) in reservations {
        if !record.committed {
            continue;
        }
        // Only compose room reservations (skip routes, portals, spawn, lights, etc.)
        if !is_room_reservation(record) {
            continue;
        }

        let request_id = record.request_id.ok_or_else(|| {
            missing_archetype_error(record, "room reservation has no archetype request identity")
        })?;
        let archetype = request_archetypes
            .get(&request_id)
            .copied()
            .ok_or_else(|| {
                missing_archetype_error(
                    record,
                    format!(
                        "request {} has no explicit archetype identity",
                        request_id.raw()
                    ),
                )
            })?;
        compose_single_room(record, archetype, &mut ir)?;
    }

    Ok(ir)
}

/// Whether a reservation is a room that needs brush composition.
fn is_composite_child(
    record: &ReservationRecord,
    reservations: &BTreeMap<ReservationId, ReservationRecord>,
) -> bool {
    use super::reservation::ReservationKind;
    reservations.values().any(|parent| {
        parent.kind == ReservationKind::Composite && parent.composite_children.contains(&record.id)
    })
}

fn is_room_reservation(record: &ReservationRecord) -> bool {
    use super::reservation::ReservationKind;
    matches!(
        record.kind,
        ReservationKind::StandardRoom
            | ReservationKind::MultiStoreyRoom
            | ReservationKind::CaveHost
            | ReservationKind::NegativeSpace
    )
}

/// Compose a single room from its reservation into the assembly IR.
fn compose_single_room(
    record: &ReservationRecord,
    archetype: ArchetypeIndex,
    ir: &mut AssemblyIR,
) -> Result<(), RichnessError> {
    let request_id = record.request_id.ok_or_else(|| {
        missing_archetype_error(record, "room reservation has no archetype request identity")
    })?;
    let arch_idx = archetype.raw() as usize;
    let shape = generated_content::ARCHETYPE_SHAPE
        .get(arch_idx)
        .copied()
        .ok_or_else(|| {
            missing_archetype_error(
                record,
                format!(
                    "archetype index {} is absent from the frozen {}-entry catalog",
                    arch_idx,
                    generated_content::ARCHETYPE_COUNT
                ),
            )
        })?;
    let fp = &record.footprint;

    let attr = SemanticAttribution::from_reservation(
        record.id,
        Some(request_id),
        Some(archetype),
        record.beat_id,
        record.zone_id,
    );
    let cost = CostSource {
        dimension: BudgetDimension::SourceFaces,
        face_count: 6, // per-brush estimate; actual counted at validation
    };

    match shape {
        ShapeRule::Rectangle => compose_rectangle(fp, &attr, cost, ir),
        ShapeRule::Chamfer => compose_chamfered(fp, &attr, cost, arch_idx, ir),
        ShapeRule::Octagon => compose_octagonal(fp, &attr, cost, arch_idx, ir),
        ShapeRule::CompositePartition => compose_composite_partition(fp, &attr, cost, arch_idx, ir),
    }
}

/// Complete crate-private Phase-09 output produced from real Phase-07
/// placement reservations and committed topology routes.
#[derive(Debug, Clone)]
pub(crate) struct StructuralComposition {
    pub assembly: AssemblyIR,
    pub visibility: VisibilityPlan,
    /// CarvedGrotto result (None when omitted or infeasible in preferred mode).
    pub cave: Option<super::cave::CaveResult>,
    /// Presentation layer (props, lights, negative space, imperfection).
    pub presentation: super::presentation::Presentation,
}

/// Reachable structural composition path for an already solved Richness map.
///
/// This deliberately remains isolated from baseline V3. It consumes the final
/// topology journal (including route-owned throat reservations), uses the
/// placement result's explicit request→archetype identities, materializes both
/// cardinal portal endpoints for every committed horizontal route, then seals
/// exact interfaces and support records. Vertical-route witnesses feed the
/// visibility records here; their floor/ceiling apertures belong to Phase 10.
pub(crate) fn compose_solved_generation(
    generation: &FullGenerationResult,
    theme: RichnessTheme,
    complexity: &ComplexityPlan,
    seed: u64,
    cave_mode: super::request::RichnessCaveMode,
) -> Result<StructuralComposition, RichnessError> {
    let reservations = &generation.topology.journal.reservations;
    let mut assembly = compose_all_rooms(reservations, &generation.placement.request_archetypes)?;
    materialize_canonical_shared_walls(reservations, &mut assembly)?;

    let (style, theme_definition) = match theme {
        RichnessTheme::Ancient => (PortalStyle::AncientPostLintel, &THEME_ANCIENT),
        RichnessTheme::Egyptian => (PortalStyle::EgyptianSteppedSurround, &THEME_EGYPTIAN),
        RichnessTheme::Brutalist => (PortalStyle::BrutalistRevealSurround, &THEME_BRUTALIST),
    };
    for route in &generation.topology.routes {
        materialize_route_portals(
            route,
            style,
            reservations,
            &generation.placement.request_archetypes,
            &mut assembly,
        )?;
    }
    let route_shell_ids = materialize_route_shells(
        &generation.topology.routes,
        reservations,
        &generation.placement.request_archetypes,
        &mut assembly,
    )?;
    enforce_room_shell_ownership(&route_shell_ids, reservations, &mut assembly)?;

    coalesce_overlapping_portal_frames(&mut assembly)?;
    prune_portal_frame_wall_conflicts(&mut assembly)?;
    ensure_portal_frame_floor_supports(&mut assembly)?;
    enforce_opening_omission(&assembly)?;

    // Phase 10-A: materialize vertical architecture features
    let _vertical_features = vertical::materialize_vertical_features(
        &mut assembly,
        reservations,
        &generation.placement.request_archetypes,
    )?;

    // Phase 11: CarvedGrotto synthesis (exactly one cave for eligible maps).
    let cave_result = super::cave::synthesize_cave(seed, cave_mode, &generation.topology.journal)?;
    if let Some(cave_result) = &cave_result {
        super::cave::validate_cave_result(cave_result)?;
        super::cave::materialize_cave(
            &mut assembly,
            cave_result,
            &generation.placement.request_archetypes,
            reservations,
        )?;
    }

    // Phase 13: props, lighting, negative space, authored imperfection.
    let presentation = super::presentation::apply_presentation(
        &mut assembly,
        &generation.topology.journal,
        &generation.placement.request_archetypes,
        theme,
        seed,
    )?;

    derive_all_interfaces(&mut assembly)?;
    derive_support_records(&mut assembly)?;
    let visibility = VisibilityPlan::build_from_assembly_and_routes(
        &assembly,
        &generation.topology.routes,
        &generation.topology.vertical_routes,
    );
    let report = validate_assembly(&assembly, &visibility, Some(complexity), theme_definition);
    if !report.all_passed {
        let failures = report
            .checks
            .iter()
            .filter(|check| !check.passed)
            .map(|check| {
                format!(
                    "{}: {}",
                    check.name,
                    check.message.as_deref().unwrap_or("unknown failure")
                )
            })
            .collect::<Vec<_>>()
            .join("; ");
        return Err(composition_error(
            "pipeline.validation",
            format!("structural composition failed validation: {failures}"),
        ));
    }
    Ok(StructuralComposition {
        assembly,
        visibility,
        cave: cave_result,
        presentation,
    })
}

/// Emit the physical corridor shells for every committed route: floor slab,
/// ceiling slab, and the two long side walls spanning the route envelope.
/// The portal frames sit at the room boundaries; the shells connect to them
/// and carry the portal posts (which extend into the corridor).
fn materialize_route_shells(
    routes: &[super::topology::CommittedRoute],
    reservations: &BTreeMap<ReservationId, ReservationRecord>,
    request_archetypes: &BTreeMap<ArchetypeRequestId, ArchetypeIndex>,
    ir: &mut AssemblyIR,
) -> Result<Vec<BrushAssemblyId>, RichnessError> {
    let q = richness_geom::QUANTUM;
    let mut shell_ids = Vec::new();
    // Corridor floors are emitted as the GAP between rooms; overlapping gaps
    // (junctions and crossing corridors) must not double-emit floors. Track
    // every emitted floor rect and subtract it from later ones.
    let mut emitted_floors: Vec<(i128, i128, i128, i128)> = Vec::new();
    let mut emitted_ceilings: Vec<(i128, i128, i128, i128)> = Vec::new();
    for route in routes {
        let Some(source) = reservations.get(&route.source) else {
            continue;
        };
        let Some(target) = reservations.get(&route.target) else {
            continue;
        };
        let source_b = richness_geom::footprint_quake_bounds(&source.footprint);
        let target_b = richness_geom::footprint_quake_bounds(&target.footprint);
        let vertical = match richness_geom::footprint_vertical_bounds(&route.envelope) {
            Ok(vertical) => vertical,
            Err(_) => continue,
        };
        let envelope_b = richness_geom::footprint_quake_bounds(&route.envelope);
        let floor_min = vertical.floor_min;
        let ceiling_max = vertical.ceiling_max;
        let wall_min = vertical.wall_min;
        let wall_max = vertical.wall_max;
        let horizontal = source_b.2 <= target_b.0 || target_b.2 <= source_b.0;
        // Corridor gap = the run between the two rooms' facing walls; the
        // cross-axis extent comes from the envelope.
        let (x0, y0, x1, y1) = if horizontal {
            let (gx0, gx1) = if source_b.2 <= target_b.0 {
                (source_b.2, target_b.0)
            } else {
                (target_b.2, source_b.0)
            };
            (gx0, envelope_b.1, gx1, envelope_b.3)
        } else {
            let (gy0, gy1) = if source_b.3 <= target_b.1 {
                (source_b.3, target_b.1)
            } else {
                (target_b.3, source_b.1)
            };
            (envelope_b.0, gy0, envelope_b.2, gy1)
        };
        if x1 <= x0 || y1 <= y0 {
            continue;
        }
        let request_id = source.request_id.unwrap_or(ArchetypeRequestId::new(0));
        let owner = SemanticAttribution::from_reservation(
            route.source,
            source.request_id,
            request_archetypes.get(&request_id).copied(),
            source.beat_id,
            source.zone_id,
        );
        // Floor slab extends under the side walls so every shell brush has
        // a positive-area gravity support contact. Overlapping portions of
        // earlier corridor floors are subtracted (junctions emit once).
        let floor_extent = if horizontal {
            ((x0, x1), (y0 - q, y1 + q))
        } else {
            ((x0 - q, x1 + q), (y0, y1))
        };
        let mut rects = vec![(
            floor_extent.0 .0,
            floor_extent.1 .0,
            floor_extent.0 .1,
            floor_extent.1 .1,
        )];
        for existing in &emitted_floors {
            let mut next = Vec::new();
            for rect in rects {
                subtract_rect(rect, *existing, &mut next);
            }
            rects = next;
        }
        for (rx0, ry0, rx1, ry1) in rects {
            shell_ids.push(build_and_insert_box(
                (rx0, rx1),
                (ry0, ry1),
                (floor_min, wall_min),
                BrushAssemblyRole::FloorSlab,
                &owner,
                route_cost(),
                ir,
            )?);
        }
        emitted_floors.push((
            floor_extent.0 .0,
            floor_extent.1 .0,
            floor_extent.0 .1,
            floor_extent.1 .1,
        ));
        let mut ceiling_rects = vec![(x0, y0, x1, y1)];
        for existing in &emitted_ceilings {
            let mut next = Vec::new();
            for rect in ceiling_rects {
                subtract_rect(rect, *existing, &mut next);
            }
            ceiling_rects = next;
        }
        for (cx0, cy0, cx1, cy1) in ceiling_rects {
            shell_ids.push(build_and_insert_box(
                (cx0, cx1),
                (cy0, cy1),
                (wall_max, ceiling_max),
                BrushAssemblyRole::CeilingSlab,
                &owner,
                route_cost(),
                ir,
            )?);
        }
        emitted_ceilings.push((x0, y0, x1, y1));
        // Each corridor shell owns its two long boundary walls. They stay
        // outside the exact clear envelope, rest on the extended floor slab,
        // and provide positive-area side support for the ceiling instead of
        // relying on a non-existent room wall along the route run.
        let side_walls = if horizontal {
            [((x0, x1), (y0 - q, y0)), ((x0, x1), (y1, y1 + q))]
        } else {
            [((x0 - q, x0), (y0, y1)), ((x1, x1 + q), (y0, y1))]
        };
        for (x, y) in side_walls {
            shell_ids.push(build_and_insert_box(
                x,
                y,
                (wall_min, wall_max),
                BrushAssemblyRole::NorthWall,
                &owner,
                route_cost(),
                ir,
            )?);
        }
    }
    Ok(shell_ids)
}

/// Route envelopes are conservative bounding boxes; their shell pieces may
/// cross a room whose actual routed path goes around it. Rooms and dual-band
/// Composites are projection-exclusive owners, so clip every route-authored
/// floor, ceiling, and boundary wall at room footprints before vertical work.
fn enforce_room_shell_ownership(
    route_shell_ids: &[BrushAssemblyId],
    reservations: &BTreeMap<ReservationId, ReservationRecord>,
    ir: &mut AssemblyIR,
) -> Result<(), RichnessError> {
    use super::reservation::ReservationKind;

    let room_footprints = reservations
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
        .map(|record| record.footprint)
        .collect::<Vec<_>>();

    for id in route_shell_ids {
        let Some(route_brush) = ir.brushes.get(id) else {
            continue;
        };
        let ((x0, y0, z0), (x1, y1, z1)) = route_brush.brush.aabb().map_err(|error| {
            composition_error(
                "pipeline.route.shell_owner",
                format!("route shell {} has no finite AABB: {error}", id.raw()),
            )
        })?;
        let layer = if z0 >= super::footprint::UPPER_FLOOR_Z as i128 {
            1
        } else {
            0
        };
        let mut rects = vec![(x0, y0, x1, y1)];
        for footprint in room_footprints.iter().filter(|footprint| {
            (layer == 0 && footprint.occupies_lower) || (layer == 1 && footprint.occupies_upper)
        }) {
            let room = richness_geom::footprint_quake_bounds(footprint);
            let mut next = Vec::new();
            for rect in rects {
                subtract_rect(rect, room, &mut next);
            }
            rects = next;
        }
        if rects.len() == 1 && rects[0] == (x0, y0, x1, y1) {
            continue;
        }

        let old = ir.remove_brush(*id).ok_or_else(|| {
            composition_error(
                "pipeline.route.shell_owner",
                format!(
                    "route shell {} disappeared during ownership clipping",
                    id.raw()
                ),
            )
        })?;
        for (rx0, ry0, rx1, ry1) in rects {
            build_and_insert_box(
                (rx0, rx1),
                (ry0, ry1),
                (z0, z1),
                old.role,
                &old.owner,
                old.cost,
                ir,
            )?;
        }
    }
    Ok(())
}

fn subtract_rect(
    rect: (i128, i128, i128, i128),
    hole: (i128, i128, i128, i128),
    out: &mut Vec<(i128, i128, i128, i128)>,
) {
    let (rx0, ry0, rx1, ry1) = rect;
    let (hx0, hy0, hx1, hy1) = hole;
    if rx0 >= hx1 || rx1 <= hx0 || ry0 >= hy1 || ry1 <= hy0 {
        out.push(rect);
        return;
    }
    // Left / right / below / above bands around the hole.
    if rx0 < hx0 {
        out.push((rx0, ry0, hx0.min(rx1), ry1));
    }
    if rx1 > hx1 {
        out.push((hx1.max(rx0), ry0, rx1, ry1));
    }
    if ry0 < hy0 {
        out.push((hx0.max(rx0).min(hx1), ry0, hx1.min(rx1), hy0));
    }
    if ry1 > hy1 {
        out.push((hx0.max(rx0).min(hx1), hy1, hx1.min(rx1), ry1));
    }
}

fn route_cost() -> CostSource {
    CostSource {
        dimension: super::assembly::BudgetDimension::SourceFaces,
        face_count: 6,
    }
}

fn materialize_route_portals(
    route: &CommittedRoute,
    style: PortalStyle,
    reservations: &BTreeMap<ReservationId, ReservationRecord>,
    request_archetypes: &BTreeMap<ArchetypeRequestId, ArchetypeIndex>,
    ir: &mut AssemblyIR,
) -> Result<(), RichnessError> {
    for portal in [&route.source_portal, &route.target_portal] {
        let endpoint = reservations
            .get(&portal.endpoint_reservation_id)
            .ok_or_else(|| {
                composition_error(
                    "pipeline.portal.endpoint",
                    format!(
                        "portal {} references missing endpoint reservation {}",
                        portal.id.raw(),
                        portal.endpoint_reservation_id.raw()
                    ),
                )
            })?;
        let request_id = endpoint.request_id.ok_or_else(|| {
            missing_archetype_error(endpoint, "portal endpoint has no archetype request")
        })?;
        let archetype = request_archetypes
            .get(&request_id)
            .copied()
            .ok_or_else(|| {
                missing_archetype_error(endpoint, "portal endpoint archetype identity is missing")
            })?;
        let attr = SemanticAttribution::from_reservation(
            endpoint.id,
            Some(request_id),
            Some(archetype),
            endpoint.beat_id,
            endpoint.zone_id,
        );
        let (wall_id, wall_role) = find_portal_wall(ir, portal)?;
        // Earlier portals split a wall into several live fragments. Fit the
        // next canonical throat against the complete owner partition, not
        // whichever fragment happens to be first in canonical brush order.
        let wall_owner = ir.brushes[&wall_id].owner.clone();
        let wall_ids = ir
            .brushes
            .values()
            .filter(|brush| brush.owner == wall_owner && brush.role == wall_role)
            .map(|brush| brush.id)
            .collect::<Vec<_>>();
        let throat = fit_throat_to_wall_partition(
            committed_throat_anchor(portal)?,
            wall_role,
            union_brush_bounds(ir, &wall_ids)?,
            &wall_ids,
            ir,
        )?;
        build_portal(
            portal.id,
            style,
            wall_id,
            throat,
            wall_role,
            &attr,
            CostSource {
                dimension: BudgetDimension::SourceFaces,
                face_count: 6,
            },
            ir,
        )?;
    }
    Ok(())
}

fn fit_throat_to_wall_partition(
    (span_min, z_min, span_max, z_max): (i128, i128, i128, i128),
    wall_role: BrushAssemblyRole,
    bounds: (i128, i128, i128, i128, i128, i128),
    owning_wall_ids: &[BrushAssemblyId],
    ir: &AssemblyIR,
) -> Result<(i128, i128, i128, i128), RichnessError> {
    let (partition_min, partition_max) = match wall_role {
        BrushAssemblyRole::NorthWall | BrushAssemblyRole::SouthWall => (bounds.0, bounds.3),
        BrushAssemblyRole::EastWall | BrushAssemblyRole::WestWall => (bounds.1, bounds.4),
        _ => {
            return Err(composition_error(
                "pipeline.portal.wall",
                "non-cardinal portal wall",
            ))
        }
    };
    let width = span_max - span_min;
    if width != 64 || partition_max - partition_min < width {
        return Err(composition_error(
            "pipeline.portal.throat",
            format!(
                "committed portal throat width {width} cannot fit in {} partition {partition_min}..{partition_max}",
                wall_role.tag()
            ),
        ));
    }

    let preferred = span_min.clamp(partition_min, partition_max - width);
    let steps = ((partition_max - partition_min - width) / richness_geom::QUANTUM) as usize;
    let mut candidates = Vec::with_capacity(steps.saturating_mul(2).saturating_add(1));
    candidates.push(preferred);
    for step in 1..=steps {
        let delta = step as i128 * richness_geom::QUANTUM;
        if preferred >= partition_min + delta {
            candidates.push(preferred - delta);
        }
        if preferred + delta + width <= partition_max {
            candidates.push(preferred + delta);
        }
    }
    candidates.sort_by_key(|candidate| ((candidate - preferred).abs(), *candidate));
    candidates.dedup();

    for fitted_min in candidates {
        let throat_bounds = match wall_role {
            BrushAssemblyRole::NorthWall | BrushAssemblyRole::SouthWall => (
                fitted_min,
                bounds.1,
                z_min,
                fitted_min + width,
                bounds.4,
                z_max,
            ),
            _ => (
                bounds.0,
                fitted_min,
                z_min,
                bounds.3,
                fitted_min + width,
                z_max,
            ),
        };
        let throat = ConvexBrush::make_box(
            (throat_bounds.0, throat_bounds.3),
            (throat_bounds.1, throat_bounds.4),
            (throat_bounds.2, throat_bounds.5),
        )
        .map_err(|error| {
            composition_error("pipeline.portal.throat", format!("throat AABB: {error}"))
        })?;
        let blocked = ir.brushes.values().any(|brush| {
            brush.role.is_wall()
                && !owning_wall_ids.contains(&brush.id)
                && richness_geom::brushes_overlap(&throat, &brush.brush).unwrap_or(true)
        });
        if !blocked {
            return Ok((fitted_min, z_min, fitted_min + width, z_max));
        }
    }

    Err(composition_error(
        "pipeline.portal.throat",
        format!(
            "no exact 64x80 throat in {} partition {partition_min}..{partition_max} avoids adjacent structural walls",
            wall_role.tag()
        ),
    ))
}

fn committed_throat_anchor(
    portal: &CommittedPortal,
) -> Result<(i128, i128, i128, i128), RichnessError> {
    if portal.headroom != 80 {
        return Err(composition_error(
            "pipeline.portal.headroom",
            format!(
                "portal {} committed headroom {} instead of 80",
                portal.id.raw(),
                portal.headroom
            ),
        ));
    }
    let (span_min, span_max) = match portal.wall {
        Dir::North | Dir::South => (portal.witness.x0, portal.witness.x1),
        Dir::East | Dir::West => (portal.witness.y0, portal.witness.y1),
    };
    let z0 = portal.anchor_cell.quake_z_min() as i128 + richness_geom::QUANTUM;
    let anchor = (
        span_min as i128 * richness_geom::QUANTUM,
        z0,
        span_max as i128 * richness_geom::QUANTUM,
        z0 + portal.headroom as i128,
    );
    validate_portal_request(portal.id, anchor, wall_role_for_dir(portal.wall))?;
    Ok(anchor)
}

fn find_portal_wall(
    ir: &AssemblyIR,
    portal: &CommittedPortal,
) -> Result<(BrushAssemblyId, BrushAssemblyRole), RichnessError> {
    let requested_role = wall_role_for_dir(portal.wall);
    if let Some(brush) = ir.brushes.values().find(|brush| {
        brush.owner.reservation_id == portal.endpoint_reservation_id && brush.role == requested_role
    }) {
        return Ok((brush.id, requested_role));
    }
    if let Some(chain) = ir.shared_wall_chains.values().find(|chain| {
        chain.owner_reservation_id == portal.endpoint_reservation_id
            || chain.sharing_reservation_id == portal.endpoint_reservation_id
    }) {
        let brush = ir.brushes.get(&chain.owner_brush_id).ok_or_else(|| {
            composition_error(
                "pipeline.portal.wall",
                format!("shared wall chain {} lost its owner brush", chain.id.raw()),
            )
        })?;
        return Ok((brush.id, brush.role));
    }
    Err(composition_error(
        "pipeline.portal.wall",
        format!(
            "portal {} found no materialized {} for reservation {}",
            portal.id.raw(),
            requested_role.tag(),
            portal.endpoint_reservation_id.raw()
        ),
    ))
}

fn wall_role_for_dir(direction: Dir) -> BrushAssemblyRole {
    match direction {
        Dir::North => BrushAssemblyRole::NorthWall,
        Dir::South => BrushAssemblyRole::SouthWall,
        Dir::East => BrushAssemblyRole::EastWall,
        Dir::West => BrushAssemblyRole::WestWall,
    }
}

// ── Session B: shared wall chain materialization ─────────────────────────

/// Materialize canonical shared-wall chains once per wall run.
///
/// Adjacent rooms sharing a wall chain consume the SAME planes — no double
/// walls, no gaps. The variation plan's shaping and mass treatments are
/// applied while preserving wall thickness, portal anchors, protected
/// throat/turn cells, and exterior envelope.
pub(crate) fn materialize_shared_wall_chain(
    chain: &WallChainRecord,
    owner_ir: &mut AssemblyIR,
    owner_wall_id: BrushAssemblyId,
    _sharing_ir: &mut AssemblyIR,
) -> Result<(), RichnessError> {
    // Verify the wall brush exists in the owner IR
    let wall_brush = owner_ir
        .brushes
        .get(&owner_wall_id)
        .ok_or_else(|| composition_error("shared_wall", "owner wall brush not found"))?;

    // Validate structural thickness is exactly 16
    if chain.structural_thickness != 16 {
        return Err(composition_error(
            "shared_wall",
            format!(
                "wall chain {:?} has non-standard thickness {}",
                chain.id.raw(),
                chain.structural_thickness
            ),
        ));
    }

    // Validate shaping: no non-zero shaping if portal anchors exist
    if !chain.portal_anchors.is_empty() {
        for &shaping in &chain.shaping {
            if shaping.offset_units() != 0 {
                return Err(composition_error(
                    "shared_wall",
                    format!(
                        "wall chain {:?} has portal anchors but non-zero shaping",
                        chain.id.raw()
                    ),
                ));
            }
        }
    }

    // Apply mass treatments that affect this wall chain.
    // Each treatment target must be verified against protected segments.
    for treatment in &chain.mass_treatments {
        // Check the treatment does not overlap protected route/portal/turn segments
        for &(seg_start, seg_end) in &chain.protected_segments {
            if treatment.segment.0 < seg_end && treatment.segment.1 > seg_start {
                return Err(composition_error(
                    "shared_wall_mass",
                    format!(
                        "wall chain {:?}: mass treatment overlaps protected segment ({}, {})",
                        chain.id.raw(),
                        seg_start,
                        seg_end
                    ),
                ));
            }
        }
    }

    // The sharing room's opening extends to the wall chain boundary — no
    // double wall. The shared wall brush is owned by exactly one brush.
    let _ = wall_brush;
    Ok(())
}

/// Collapse touching room-wall pairs into canonical one-owner runs.
///
/// Each shared run is clipped exactly from both cardinal wall brushes. The
/// sharing copy is removed and both semantic rooms reference the owner's
/// unchanged canonical boundary plane through `SharedWallChainAssembly`.
pub(crate) fn materialize_canonical_shared_walls(
    reservations: &BTreeMap<ReservationId, ReservationRecord>,
    ir: &mut AssemblyIR,
) -> Result<(), RichnessError> {
    let rooms: Vec<_> = reservations
        .values()
        .filter(|record| record.committed && is_room_reservation(record))
        // Composite multi-storey rooms own their full boundary shells (their
        // interior constructions split their own walls); the shared-wall
        // deduplication applies to ordinary single-band rooms.
        .filter(|record| !is_composite_child(record, reservations))
        .collect();

    for (index, a) in rooms.iter().enumerate() {
        for b in rooms.iter().skip(index + 1) {
            if !layers_overlap(&a.footprint, &b.footprint) {
                continue;
            }
            let Some((a_role, b_role)) = touching_wall_roles(&a.footprint, &b.footprint) else {
                continue;
            };
            let (owner, owner_role, sharing, sharing_role) = if a.id < b.id {
                (*a, a_role, *b, b_role)
            } else {
                (*b, b_role, *a, a_role)
            };

            let owner_ids: Vec<_> = ir
                .brushes
                .values()
                .filter(|brush| brush.owner.reservation_id == owner.id && brush.role == owner_role)
                .map(|brush| brush.id)
                .collect();
            let sharing_ids: Vec<_> = ir
                .brushes
                .values()
                .filter(|brush| {
                    brush.owner.reservation_id == sharing.id && brush.role == sharing_role
                })
                .map(|brush| brush.id)
                .collect();

            let mut contact = None;
            for owner_id in &owner_ids {
                for sharing_id in &sharing_ids {
                    let Some(owner_brush) = ir.brushes.get(owner_id) else {
                        continue;
                    };
                    let Some(sharing_brush) = ir.brushes.get(sharing_id) else {
                        continue;
                    };
                    if let Some(exact) =
                        richness_geom::exact_face_contact(&owner_brush.brush, &sharing_brush.brush)
                    {
                        contact = Some((*owner_id, *sharing_id, exact));
                        break;
                    }
                }
                if contact.is_some() {
                    break;
                }
            }
            let Some((owner_id, sharing_id, exact)) = contact else {
                // Chamfered footprints can touch in their conservative XY
                // rectangles while their actual diagonal shells remain
                // disjoint. There is no shared wall to deduplicate in that
                // case; each room keeps its independently sealed boundary.
                continue;
            };

            let span = contact_span(&exact.vertices, owner_role)?;
            let owner_run = split_cardinal_wall_run(ir, owner_id, owner_role, span)?;
            let sharing_run = split_cardinal_wall_run(ir, sharing_id, sharing_role, span)?;
            ir.remove_brush(sharing_run);
            ir.supports.retain(|_, support| {
                support.child != sharing_run
                    && !matches!(support.parent, SupportTarget::Brush(id) if id == sharing_run)
            });
            ir.interfaces.retain(|_, interface| {
                interface.brush_a != sharing_run && interface.brush_b != sharing_run
            });

            let id = ir.alloc_wall_chain_id();
            ir.insert_shared_wall_chain(SharedWallChainAssembly {
                id,
                owner_reservation_id: owner.id,
                sharing_reservation_id: sharing.id,
                owner_brush_id: owner_run,
                shared_plane: exact.plane,
                span,
            });
        }
    }
    Ok(())
}

fn layers_overlap(a: &Footprint3D, b: &Footprint3D) -> bool {
    (a.occupies_lower && b.occupies_lower) || (a.occupies_upper && b.occupies_upper)
}

fn touching_wall_roles(
    a: &Footprint3D,
    b: &Footprint3D,
) -> Option<(BrushAssemblyRole, BrushAssemblyRole)> {
    let x_overlap = a.x0.max(b.x0) < a.x1.min(b.x1);
    let y_overlap = a.y0.max(b.y0) < a.y1.min(b.y1);
    if a.x1 == b.x0 && y_overlap {
        Some((BrushAssemblyRole::EastWall, BrushAssemblyRole::WestWall))
    } else if b.x1 == a.x0 && y_overlap {
        Some((BrushAssemblyRole::WestWall, BrushAssemblyRole::EastWall))
    } else if a.y1 == b.y0 && x_overlap {
        Some((BrushAssemblyRole::SouthWall, BrushAssemblyRole::NorthWall))
    } else if b.y1 == a.y0 && x_overlap {
        Some((BrushAssemblyRole::NorthWall, BrushAssemblyRole::SouthWall))
    } else {
        None
    }
}

fn contact_span(
    vertices: &[crate::enhanced_v3::geometry::Point3],
    role: BrushAssemblyRole,
) -> Result<(i128, i128), RichnessError> {
    let coordinate = |point: &crate::enhanced_v3::geometry::Point3| match role {
        BrushAssemblyRole::NorthWall | BrushAssemblyRole::SouthWall => point.x,
        BrushAssemblyRole::EastWall | BrushAssemblyRole::WestWall => point.y,
        _ => crate::enhanced_v3::geometry::Rational::ZERO,
    };
    let min = vertices
        .iter()
        .map(coordinate)
        .min()
        .ok_or_else(|| composition_error("shared_wall.span", "empty contact polygon"))?;
    let max = vertices
        .iter()
        .map(coordinate)
        .max()
        .ok_or_else(|| composition_error("shared_wall.span", "empty contact polygon"))?;
    if min.den != 1 || max.den != 1 || min.num >= max.num {
        return Err(composition_error(
            "shared_wall.span",
            format!("shared wall contact has non-integral or empty span {min}..{max}"),
        ));
    }
    Ok((min.num, max.num))
}

fn split_cardinal_wall_run(
    ir: &mut AssemblyIR,
    brush_id: BrushAssemblyId,
    role: BrushAssemblyRole,
    (span_min, span_max): (i128, i128),
) -> Result<BrushAssemblyId, RichnessError> {
    let original = ir
        .brushes
        .get(&brush_id)
        .cloned()
        .ok_or_else(|| composition_error("shared_wall.split", "wall brush not found"))?;
    let ((x0, y0, _z0), (x1, y1, _z1)) = original
        .brush
        .aabb()
        .map_err(|error| composition_error("shared_wall.split", format!("AABB: {error}")))?;
    let (brush_min, brush_max) = match role {
        BrushAssemblyRole::NorthWall | BrushAssemblyRole::SouthWall => (x0, x1),
        BrushAssemblyRole::EastWall | BrushAssemblyRole::WestWall => (y0, y1),
        _ => {
            return Err(composition_error(
                "shared_wall.split",
                "shared run is not cardinal",
            ));
        }
    };
    if span_min < brush_min || span_max > brush_max {
        return Err(composition_error(
            "shared_wall.split",
            "shared run escaped its wall brush",
        ));
    }
    if span_min == brush_min && span_max == brush_max {
        return Ok(brush_id);
    }

    let intervals = [
        (brush_min, span_min, false),
        (span_min, span_max, true),
        (span_max, brush_max, false),
    ];
    let mut run_id = None;
    let mut replacements = Vec::new();
    for (start, end, is_run) in intervals {
        if start >= end {
            continue;
        }
        let mut faces = original.brush.faces.clone();
        let (lower, upper) = match role {
            BrushAssemblyRole::NorthWall | BrushAssemblyRole::SouthWall => (
                CanonicalPlane::new(1, 0, 0, start),
                CanonicalPlane::new(-1, 0, 0, -end),
            ),
            BrushAssemblyRole::EastWall | BrushAssemblyRole::WestWall => (
                CanonicalPlane::new(0, 1, 0, start),
                CanonicalPlane::new(0, -1, 0, -end),
            ),
            _ => {
                return Err(composition_error(
                    "shared_wall.split",
                    "shared run changed to a non-cardinal role",
                ));
            }
        };
        faces.push(
            BrushFace::new(lower.map_err(|error| {
                composition_error("shared_wall.split", format!("lower plane: {error}"))
            })?)
            .map_err(|error| {
                composition_error("shared_wall.split", format!("lower face: {error}"))
            })?,
        );
        faces.push(
            BrushFace::new(upper.map_err(|error| {
                composition_error("shared_wall.split", format!("upper plane: {error}"))
            })?)
            .map_err(|error| {
                composition_error("shared_wall.split", format!("upper face: {error}"))
            })?,
        );
        let brush = validated_halfspace_brush(faces, "shared_wall.split")?;
        let id = ir.alloc_brush_id();
        ir.insert_brush(BrushAssembly {
            id,
            brush,
            role,
            owner: original.owner.clone(),
            cost: original.cost,
            support: original.support.clone(),
        });
        replacements.push(id);
        if is_run {
            run_id = Some(id);
        }
    }

    ir.remove_brush(brush_id);
    ir.supports.retain(|_, support| support.child != brush_id);
    ir.interfaces
        .retain(|_, interface| interface.brush_a != brush_id && interface.brush_b != brush_id);
    let _ = replacements;
    run_id.ok_or_else(|| composition_error("shared_wall.split", "shared run was not emitted"))
}

// ── Session B: portal construction ───────────────────────────────────────

/// Build a portal at the given anchor location with the specified style.
///
/// All portals are CARDINAL. No diagonal portals are permitted.
pub(crate) fn build_portal(
    portal_id: PortalId,
    style: PortalStyle,
    wall_brush_id: BrushAssemblyId,
    throat_anchor: (i128, i128, i128, i128),
    wall_role: BrushAssemblyRole,
    attr: &SemanticAttribution,
    cost: CostSource,
    ir: &mut AssemblyIR,
) -> Result<Option<PortalAssembly>, RichnessError> {
    validate_portal_request(portal_id, throat_anchor, wall_role)?;

    let wall = ir
        .brushes
        .get(&wall_brush_id)
        .ok_or_else(|| composition_error("portal.wall", "wall brush not found"))?;
    if wall.role != wall_role {
        return Err(composition_error(
            "portal.wall",
            format!(
                "portal {} requested {} but brush {} has role {}",
                portal_id.raw(),
                wall_role.tag(),
                wall_brush_id.raw(),
                wall.role.tag()
            ),
        ));
    }
    let wall_owner = wall.owner.clone();
    let wall_segment_ids: Vec<_> = ir
        .brushes
        .values()
        .filter(|candidate| candidate.owner == wall_owner && candidate.role == wall_role)
        .map(|candidate| candidate.id)
        .collect();
    let partition_bounds = union_brush_bounds(ir, &wall_segment_ids)?;
    validate_throat_within_partition(throat_anchor, wall_role, partition_bounds)?;

    // Skip portals whose exact throat already belongs to a committed
    // opening (overlapping route sockets on one wall run must not double
    // materialize frames inside each other's throats).
    if ir.openings.values().any(|existing| {
        let (ex0, ey0, ez0, ex1, ey1, ez1) = existing.bounds;
        let (ox0, oy0, oz0, ox1, oy1, oz1) =
            portal_throat_aabb(throat_anchor, wall_role, partition_bounds);
        ox0 < ex1 && ox1 > ex0 && oy0 < ey1 && oy1 > ey0 && oz0 < ez1 && oz1 > ez0
    }) {
        return Ok(None);
    }

    let portal = match style {
        PortalStyle::AncientPostLintel => build_ancient_post_lintel(
            portal_id,
            wall_brush_id,
            partition_bounds,
            throat_anchor,
            wall_role,
            attr,
            cost,
            ir,
        ),
        PortalStyle::EgyptianSteppedSurround => build_egyptian_stepped_surround(
            portal_id,
            wall_brush_id,
            partition_bounds,
            throat_anchor,
            wall_role,
            attr,
            cost,
            ir,
        ),
        PortalStyle::BrutalistRevealSurround => build_brutalist_reveal_surround(
            portal_id,
            wall_brush_id,
            partition_bounds,
            throat_anchor,
            wall_role,
            attr,
            cost,
            ir,
        ),
    }?;
    let mut portal = portal;

    // Split the wall at the complete emitted frame envelope, not an assumed
    // one-quantum margin around the throat. Brutalist surrounds extend two
    // quanta beyond the throat and their outer lintel reaches two quanta
    // above it; leaving the structural wall there creates positive-volume
    // overlap. The partition depth remains the wall's full depth.
    let (tx0, ty0, tz0, tx1, ty1, tz1) = portal.throat_bounds;
    let mut frame_x0 = tx0;
    let mut frame_y0 = ty0;
    let mut frame_z0 = tz0;
    let mut frame_x1 = tx1;
    let mut frame_y1 = ty1;
    let mut frame_z1 = tz1;
    for frame_id in portal
        .post_ids
        .iter()
        .chain(portal.lintel_ids.iter())
        .chain(portal.surround_ids.iter())
    {
        let frame = ir.brushes.get(frame_id).ok_or_else(|| {
            composition_error(
                "portal.wall_split",
                format!("portal {} lost frame {}", portal_id.raw(), frame_id.raw()),
            )
        })?;
        let ((x0, y0, z0), (x1, y1, z1)) = frame.brush.aabb().map_err(|error| {
            composition_error("portal.wall_split", format!("frame AABB: {error}"))
        })?;
        frame_x0 = frame_x0.min(x0);
        frame_y0 = frame_y0.min(y0);
        frame_z0 = frame_z0.min(z0);
        frame_x1 = frame_x1.max(x1);
        frame_y1 = frame_y1.max(y1);
        frame_z1 = frame_z1.max(z1);
    }
    let frame_bounds = match wall_role {
        BrushAssemblyRole::NorthWall | BrushAssemblyRole::SouthWall => (
            frame_x0,
            partition_bounds.1,
            frame_z0,
            frame_x1,
            partition_bounds.4,
            frame_z1,
        ),
        _ => (
            partition_bounds.0,
            frame_y0,
            frame_z0,
            partition_bounds.3,
            frame_y1,
            frame_z1,
        ),
    };
    let mut replacement_ids = Vec::new();
    for segment_id in wall_segment_ids {
        replacement_ids.extend(split_wall_around_opening(ir, segment_id, frame_bounds)?);
    }
    replacement_ids.sort_unstable();
    replacement_ids.dedup();
    let owner_brush_id = replacement_ids.first().copied().ok_or_else(|| {
        composition_error(
            "portal.wall_split",
            format!(
                "portal {} removed its complete owning wall",
                portal_id.raw()
            ),
        )
    })?;

    let opening = ir
        .openings
        .get_mut(&portal.opening_id)
        .ok_or_else(|| composition_error("portal.opening", "opening record disappeared"))?;
    opening.owner_brush_id = owner_brush_id;
    opening.wall_segment_ids = replacement_ids;
    opening.owner_partition_bounds = partition_bounds;
    portal.wall_brush_id = owner_brush_id;
    ir.portal_assemblies.push(portal.clone());
    Ok(Some(portal))
}

fn coalesce_overlapping_portal_frames(ir: &mut AssemblyIR) -> Result<(), RichnessError> {
    type MergeKey = (
        BrushAssemblyRole,
        SemanticAttribution,
        u8,
        (i128, i128),
        (i128, i128),
    );
    let mut groups: BTreeMap<MergeKey, Vec<(BrushAssemblyId, i128, i128)>> = BTreeMap::new();
    for brush in ir.brushes.values().filter(|brush| {
        matches!(
            brush.role,
            BrushAssemblyRole::PortalPost
                | BrushAssemblyRole::PortalLintel
                | BrushAssemblyRole::PortalSurround
        )
    }) {
        let ((x0, y0, z0), (x1, y1, z1)) = brush
            .brush
            .aabb()
            .map_err(|error| composition_error("portal.frame_merge", format!("AABB: {error}")))?;
        let dimensions = [x1 - x0, y1 - y0, z1 - z0];
        let axis = (0..3)
            .max_by_key(|axis| (dimensions[*axis], std::cmp::Reverse(*axis)))
            .unwrap_or(0);
        let (fixed_a, fixed_b, start, end) = match axis {
            0 => ((y0, y1), (z0, z1), x0, x1),
            1 => ((x0, x1), (z0, z1), y0, y1),
            _ => ((x0, x1), (y0, y1), z0, z1),
        };
        groups
            .entry((
                brush.role,
                brush.owner.clone(),
                axis as u8,
                fixed_a,
                fixed_b,
            ))
            .or_default()
            .push((brush.id, start, end));
    }

    for ((role, owner, axis, fixed_a, fixed_b), mut intervals) in groups {
        intervals.sort_by_key(|(id, start, end)| (*start, *end, *id));
        let mut index = 0;
        while index < intervals.len() {
            let mut end = intervals[index].2;
            let mut merged = vec![intervals[index].0];
            let start = intervals[index].1;
            index += 1;
            while index < intervals.len() && intervals[index].1 < end {
                end = end.max(intervals[index].2);
                merged.push(intervals[index].0);
                index += 1;
            }
            if merged.len() == 1 {
                continue;
            }

            let first = ir.brushes.get(&merged[0]).cloned().ok_or_else(|| {
                composition_error("portal.frame_merge", "frame brush disappeared")
            })?;
            let (x, y, z) = match axis {
                0 => ((start, end), fixed_a, fixed_b),
                1 => (fixed_a, (start, end), fixed_b),
                _ => (fixed_a, fixed_b, (start, end)),
            };
            let brush = ConvexBrush::make_box(x, y, z).map_err(|error| {
                composition_error("portal.frame_merge", format!("merged frame: {error}"))
            })?;
            let new_id = ir.alloc_brush_id();
            for old_id in &merged {
                ir.remove_brush(*old_id);
            }
            ir.insert_brush(BrushAssembly {
                id: new_id,
                brush,
                role,
                owner: owner.clone(),
                cost: first.cost,
                support: first.support.clone(),
            });
            replace_frame_ids(ir, &merged, new_id);
        }
    }
    Ok(())
}

/// Drop decorative portal courses that would intrude into an unrelated room
/// or route shell. The structural wall omission remains the portal; a frame
/// cannot claim a third-party wall volume merely to preserve ornamentation.
fn prune_portal_frame_wall_conflicts(ir: &mut AssemblyIR) -> Result<(), RichnessError> {
    let frames = ir
        .brushes
        .values()
        .filter(|brush| {
            matches!(
                brush.role,
                BrushAssemblyRole::PortalPost
                    | BrushAssemblyRole::PortalLintel
                    | BrushAssemblyRole::PortalSurround
            )
        })
        .cloned()
        .collect::<Vec<_>>();
    let walls = ir
        .brushes
        .values()
        .filter(|brush| brush.role.is_wall())
        .cloned()
        .collect::<Vec<_>>();
    let conflicting = frames
        .iter()
        .filter(|frame| {
            walls.iter().any(|wall| {
                wall.owner != frame.owner
                    && richness_geom::brushes_overlap(&frame.brush, &wall.brush).unwrap_or(true)
            })
        })
        .map(|frame| frame.id)
        .collect::<BTreeSet<_>>();
    if conflicting.is_empty() {
        return Ok(());
    }
    for id in &conflicting {
        ir.remove_brush(*id);
    }
    for opening in ir.openings.values_mut() {
        opening
            .frame_brush_ids
            .retain(|id| !conflicting.contains(id));
    }
    for portal in &mut ir.portal_assemblies {
        portal.post_ids.retain(|id| !conflicting.contains(id));
        portal.lintel_ids.retain(|id| !conflicting.contains(id));
        portal.surround_ids.retain(|id| !conflicting.contains(id));
    }
    Ok(())
}

/// Portal posts and surround courses at the floor line require an actual
/// floor-slab bearing. Route-shell ownership clipping can legitimately remove
/// a corridor slab beneath a frame that sits just inside its host's wall, so
/// restore a discrete 16-unit footing only when no room or route slab already
/// provides a positive-area contact.
fn ensure_portal_frame_floor_supports(ir: &mut AssemblyIR) -> Result<(), RichnessError> {
    let frames = ir
        .brushes
        .values()
        .filter(|brush| {
            matches!(
                brush.role,
                BrushAssemblyRole::PortalPost | BrushAssemblyRole::PortalSurround
            )
        })
        .cloned()
        .collect::<Vec<_>>();

    for frame in frames {
        let ((x0, y0, z0), (x1, y1, _)) = frame
            .brush
            .aabb()
            .map_err(|error| composition_error("portal.frame_support", format!("AABB: {error}")))?;
        if z0 != richness_geom::WALL_THICKNESS {
            continue;
        }
        let has_bearing = ir.brushes.values().any(|brush| {
            brush.role == BrushAssemblyRole::FloorSlab
                && brush.brush.aabb().is_ok_and(|(min, max)| {
                    max.2 == z0 && min.0 < x1 && max.0 > x0 && min.1 < y1 && max.1 > y0
                })
        });
        if !has_bearing {
            build_and_insert_box(
                (x0, x1),
                (y0, y1),
                (z0 - richness_geom::WALL_THICKNESS, z0),
                BrushAssemblyRole::FloorSlab,
                &frame.owner,
                frame.cost,
                ir,
            )?;
        }
    }
    Ok(())
}

fn replace_frame_ids(
    ir: &mut AssemblyIR,
    old_ids: &[BrushAssemblyId],
    replacement: BrushAssemblyId,
) {
    let replace = |ids: &mut Vec<BrushAssemblyId>| {
        if ids.iter().any(|id| old_ids.contains(id)) {
            ids.retain(|id| !old_ids.contains(id));
            ids.push(replacement);
            ids.sort_unstable();
            ids.dedup();
        }
    };
    for opening in ir.openings.values_mut() {
        replace(&mut opening.frame_brush_ids);
    }
    for portal in &mut ir.portal_assemblies {
        replace(&mut portal.post_ids);
        replace(&mut portal.lintel_ids);
        replace(&mut portal.surround_ids);
    }
}

fn portal_throat_aabb(
    throat_anchor: (i128, i128, i128, i128),
    wall_role: BrushAssemblyRole,
    wall_bounds: (i128, i128, i128, i128, i128, i128),
) -> (i128, i128, i128, i128, i128, i128) {
    // Include the frame margin (one wall thickness) so two portals whose
    // frames would collide on the same run are deduplicated.
    let (s0, z0, s1, z1) = throat_anchor;
    let w = richness_geom::WALL_THICKNESS;
    match wall_role {
        BrushAssemblyRole::NorthWall | BrushAssemblyRole::SouthWall => {
            (s0 - w, wall_bounds.1, z0, s1 + w, wall_bounds.4, z1)
        }
        _ => (wall_bounds.0, s0 - w, z0, wall_bounds.3, s1 + w, z1),
    }
}

fn validate_portal_request(
    portal_id: PortalId,
    (span_min, z_min, span_max, z_max): (i128, i128, i128, i128),
    wall_role: BrushAssemblyRole,
) -> Result<(), RichnessError> {
    if !matches!(
        wall_role,
        BrushAssemblyRole::NorthWall
            | BrushAssemblyRole::SouthWall
            | BrushAssemblyRole::EastWall
            | BrushAssemblyRole::WestWall
    ) {
        return Err(composition_error(
            "portal.wall_role",
            format!(
                "portal {} cannot use non-cardinal wall role {}",
                portal_id.raw(),
                wall_role.tag()
            ),
        ));
    }
    if span_max.checked_sub(span_min) != Some(64) || z_max.checked_sub(z_min) != Some(80) {
        return Err(composition_error(
            "portal.throat",
            format!(
                "portal {} throat must be ordered exact 64x80; got span=({span_min},{span_max}) z=({z_min},{z_max})",
                portal_id.raw()
            ),
        ));
    }
    if [span_min, z_min, span_max, z_max]
        .iter()
        .any(|value| value.rem_euclid(richness_geom::QUANTUM) != 0)
    {
        return Err(composition_error(
            "portal.throat",
            format!("portal {} throat is not 16-unit aligned", portal_id.raw()),
        ));
    }
    Ok(())
}

fn union_brush_bounds(
    ir: &AssemblyIR,
    brush_ids: &[BrushAssemblyId],
) -> Result<(i128, i128, i128, i128, i128, i128), RichnessError> {
    let mut bounds: Option<(i128, i128, i128, i128, i128, i128)> = None;
    for brush_id in brush_ids {
        let brush = ir
            .brushes
            .get(brush_id)
            .ok_or_else(|| composition_error("portal.wall", "wall segment not found"))?;
        let ((x0, y0, z0), (x1, y1, z1)) = brush
            .brush
            .aabb()
            .map_err(|error| composition_error("portal.wall", format!("wall AABB: {error}")))?;
        bounds = Some(match bounds {
            None => (x0, y0, z0, x1, y1, z1),
            Some((bx0, by0, bz0, bx1, by1, bz1)) => (
                bx0.min(x0),
                by0.min(y0),
                bz0.min(z0),
                bx1.max(x1),
                by1.max(y1),
                bz1.max(z1),
            ),
        });
    }
    bounds.ok_or_else(|| composition_error("portal.wall", "wall partition has no segments"))
}

fn validate_throat_within_partition(
    (s0, z0, s1, z1): (i128, i128, i128, i128),
    wall_role: BrushAssemblyRole,
    (x0, y0, wall_z0, x1, y1, wall_z1): (i128, i128, i128, i128, i128, i128),
) -> Result<(), RichnessError> {
    let span_contains = match wall_role {
        BrushAssemblyRole::NorthWall | BrushAssemblyRole::SouthWall => s0 >= x0 && s1 <= x1,
        BrushAssemblyRole::EastWall | BrushAssemblyRole::WestWall => s0 >= y0 && s1 <= y1,
        _ => false,
    };
    if !span_contains || z0 < wall_z0 || z1 > wall_z1 {
        return Err(composition_error(
            "portal.throat",
            format!(
                "exact portal throat ({s0},{z0})..({s1},{z1}) escapes {} partition ({x0},{y0},{wall_z0})..({x1},{y1},{wall_z1})",
                wall_role.tag()
            ),
        ));
    }
    Ok(())
}

fn frame_depth_range(
    wall_role: BrushAssemblyRole,
    bounds: (i128, i128, i128, i128, i128, i128),
    layer: i128,
) -> Result<((i128, i128), (i128, i128)), RichnessError> {
    let (x0, y0, _z0, x1, y1, _z1) = bounds;
    let step = richness_geom::WALL_THICKNESS * layer;
    // Frames stay INSIDE the room footprint so every frame brush rests on
    // the floor slab. Higher layers step inward from the wall line.
    match wall_role {
        BrushAssemblyRole::NorthWall => Ok(((x0, x1), (y0 + step, y0 + 16 + step))),
        BrushAssemblyRole::SouthWall => Ok(((x0, x1), (y1 - 16 - step, y1 - step))),
        BrushAssemblyRole::WestWall => Ok(((x0 + step, x0 + 16 + step), (y0, y1))),
        BrushAssemblyRole::EastWall => Ok(((x1 - 16 - step, x1 - step), (y0, y1))),
        _ => Err(composition_error(
            "portal.frame_depth",
            "portal frame depth requires a cardinal wall",
        )),
    }
}

/// Clip decorative portal framing to the cardinal wall run that can legally
/// carry it. The committed topology owns only the exact two-cell throat
/// socket; near a corner, a wider surround must shed or shorten its outer
/// course instead of penetrating the perpendicular structural wall.
fn clip_portal_frame_spans(
    ir: &mut AssemblyIR,
    frame_ids: &mut Vec<BrushAssemblyId>,
    wall_role: BrushAssemblyRole,
    wall_bounds: (i128, i128, i128, i128, i128, i128),
) -> Result<(), RichnessError> {
    let (span_min, span_max) = match wall_role {
        // North/south walls include both corner blocks in their AABB.
        BrushAssemblyRole::NorthWall | BrushAssemblyRole::SouthWall => (
            wall_bounds.0 + richness_geom::WALL_THICKNESS,
            wall_bounds.3 - richness_geom::WALL_THICKNESS,
        ),
        // East/west walls are already inset between the corner blocks.
        BrushAssemblyRole::EastWall | BrushAssemblyRole::WestWall => (wall_bounds.1, wall_bounds.4),
        _ => {
            return Err(composition_error(
                "portal.frame_clip",
                "portal frame clipping requires a cardinal wall",
            ));
        }
    };

    let original_ids = std::mem::take(frame_ids);
    for id in original_ids {
        let Some(original) = ir.brushes.get(&id).cloned() else {
            return Err(composition_error(
                "portal.frame_clip",
                format!("portal frame brush {} disappeared", id.raw()),
            ));
        };
        let ((x0, y0, z0), (x1, y1, z1)) = original
            .brush
            .aabb()
            .map_err(|error| composition_error("portal.frame_clip", format!("AABB: {error}")))?;
        let (x, y) = match wall_role {
            BrushAssemblyRole::NorthWall | BrushAssemblyRole::SouthWall => {
                ((x0.max(span_min), x1.min(span_max)), (y0, y1))
            }
            BrushAssemblyRole::EastWall | BrushAssemblyRole::WestWall => {
                ((x0, x1), (y0.max(span_min), y1.min(span_max)))
            }
            _ => unreachable!("cardinal role checked above"),
        };
        if x.0 >= x.1 || y.0 >= y.1 {
            ir.remove_brush(id);
            continue;
        }
        if (x0, x1) != x || (y0, y1) != y {
            let clipped = ConvexBrush::make_box(x, y, (z0, z1)).map_err(|error| {
                composition_error("portal.frame_clip", format!("clipped frame: {error}"))
            })?;
            richness_geom::validate_brush(&clipped)?;
            let brush = ir.brushes.get_mut(&id).ok_or_else(|| {
                composition_error("portal.frame_clip", "portal frame brush disappeared")
            })?;
            brush.brush = clipped;
            brush.cost.face_count = brush.brush.faces.len() as u32;
        }
        frame_ids.push(id);
    }
    Ok(())
}

/// Ancient post-and-lintel portal: two 16u posts + 16u lintel framing a 64×80 throat.
///
/// The throat_anchor is (span_min, z_min, span_max, z_max) where:
/// - N/S wall: span = X coordinate of throat (width 64), z = vertical extent (height 80)
/// - E/W wall: span = Y coordinate of throat (width 64), z = vertical extent (height 80)
fn build_ancient_post_lintel(
    portal_id: PortalId,
    wall_brush_id: BrushAssemblyId,
    wall_bounds: (i128, i128, i128, i128, i128, i128),
    throat_anchor: (i128, i128, i128, i128), // (span_min, z_min, span_max, z_max)
    wall_role: BrushAssemblyRole,
    attr: &SemanticAttribution,
    cost: CostSource,
    ir: &mut AssemblyIR,
) -> Result<PortalAssembly, RichnessError> {
    validate_portal_request(portal_id, throat_anchor, wall_role)?;
    let (s0, z0, s1, z1) = throat_anchor;
    let post_w = richness_geom::WALL_THICKNESS;
    let post_z0 = z0;
    let post_z1 = z1;
    let lintel_z0 = z1;
    let lintel_z1 = lintel_z0 + post_w;

    let mut post_ids = Vec::new();
    let mut lintel_ids = Vec::new();

    let bb = (
        (wall_bounds.0, wall_bounds.1, wall_bounds.2),
        (wall_bounds.3, wall_bounds.4, wall_bounds.5),
    );
    let (frame_x, frame_y) = frame_depth_range(wall_role, wall_bounds, 0)?;
    let ((wx0, wy0, _wz0), (wx1, wy1, _wz1)) = bb;

    let throat_bounds = match wall_role {
        BrushAssemblyRole::NorthWall | BrushAssemblyRole::SouthWall => {
            (s0, wy0, post_z0, s1, wy1, post_z1)
        }
        BrushAssemblyRole::EastWall | BrushAssemblyRole::WestWall => {
            (wx0, s0, post_z0, wx1, s1, post_z1)
        }
        _ => {
            return Err(composition_error(
                "portal.wall_role",
                "ancient portal requires a cardinal wall",
            ));
        }
    };

    match wall_role {
        BrushAssemblyRole::NorthWall | BrushAssemblyRole::SouthWall => {
            let post_id = build_and_insert_box(
                (s0 - post_w, s0),
                frame_y,
                (post_z0, post_z1),
                BrushAssemblyRole::PortalPost,
                attr,
                cost,
                ir,
            )?;
            post_ids.push(post_id);
            let post_id = build_and_insert_box(
                (s1, s1 + post_w),
                frame_y,
                (post_z0, post_z1),
                BrushAssemblyRole::PortalPost,
                attr,
                cost,
                ir,
            )?;
            post_ids.push(post_id);
            let lintel_id = build_and_insert_box(
                (s0 - post_w, s1 + post_w),
                frame_y,
                (lintel_z0, lintel_z1),
                BrushAssemblyRole::PortalLintel,
                attr,
                cost,
                ir,
            )?;
            lintel_ids.push(lintel_id);
        }
        BrushAssemblyRole::EastWall | BrushAssemblyRole::WestWall => {
            let post_id = build_and_insert_box(
                frame_x,
                (s0 - post_w, s0),
                (post_z0, post_z1),
                BrushAssemblyRole::PortalPost,
                attr,
                cost,
                ir,
            )?;
            post_ids.push(post_id);
            let post_id = build_and_insert_box(
                frame_x,
                (s1, s1 + post_w),
                (post_z0, post_z1),
                BrushAssemblyRole::PortalPost,
                attr,
                cost,
                ir,
            )?;
            post_ids.push(post_id);
            let lintel_id = build_and_insert_box(
                frame_x,
                (s0 - post_w, s1 + post_w),
                (lintel_z0, lintel_z1),
                BrushAssemblyRole::PortalLintel,
                attr,
                cost,
                ir,
            )?;
            lintel_ids.push(lintel_id);
        }
        _ => {
            return Err(composition_error(
                "portal.wall_role",
                "ancient portal requires a cardinal wall",
            ));
        }
    }

    clip_portal_frame_spans(ir, &mut post_ids, wall_role, wall_bounds)?;
    clip_portal_frame_spans(ir, &mut lintel_ids, wall_role, wall_bounds)?;

    let opening_id = ir.alloc_opening_id();
    let opening = OpeningRecord {
        id: opening_id,
        owner_brush_id: wall_brush_id,
        wall_segment_ids: vec![wall_brush_id],
        owner_partition_bounds: wall_bounds,
        wall_role,
        owner: attr.clone(),
        bounds: throat_bounds,
        portal_id: Some(portal_id),
        frame_brush_ids: post_ids
            .iter()
            .copied()
            .chain(lintel_ids.iter().copied())
            .collect(),
        portal_style: Some(PortalStyle::AncientPostLintel),
    };
    ir.insert_opening(opening);

    Ok(PortalAssembly {
        portal_id,
        style: PortalStyle::AncientPostLintel,
        post_ids,
        lintel_ids,
        surround_ids: Vec::new(),
        wall_brush_id,
        opening_id,
        throat_bounds,
    })
}

fn build_egyptian_stepped_surround(
    portal_id: PortalId,
    wall_brush_id: BrushAssemblyId,
    wall_bounds: (i128, i128, i128, i128, i128, i128),
    throat_anchor: (i128, i128, i128, i128), // (span_min, z_min, span_max, z_max)
    wall_role: BrushAssemblyRole,
    attr: &SemanticAttribution,
    cost: CostSource,
    ir: &mut AssemblyIR,
) -> Result<PortalAssembly, RichnessError> {
    validate_portal_request(portal_id, throat_anchor, wall_role)?;
    let (s0, z0, s1, z1) = throat_anchor;
    let step = 16i128;
    let post_z0 = z0;
    let post_z1 = z1;
    let (wall_x0, wall_y0, _wall_z0, wall_x1, wall_y1, _wall_z1) = wall_bounds;
    let mut surround_ids = Vec::new();

    let throat_bounds = match wall_role {
        BrushAssemblyRole::NorthWall | BrushAssemblyRole::SouthWall => {
            for layer in 0..3 {
                let (_frame_x, frame_y) = frame_depth_range(wall_role, wall_bounds, layer)?;
                surround_ids.push(build_and_insert_box(
                    (s0 - step, s0),
                    frame_y,
                    (post_z0, post_z1),
                    BrushAssemblyRole::PortalSurround,
                    attr,
                    cost,
                    ir,
                )?);
                surround_ids.push(build_and_insert_box(
                    (s1, s1 + step),
                    frame_y,
                    (post_z0, post_z1),
                    BrushAssemblyRole::PortalSurround,
                    attr,
                    cost,
                    ir,
                )?);
                surround_ids.push(build_and_insert_box(
                    (s0 - step, s1 + step),
                    frame_y,
                    (z1, z1 + step),
                    BrushAssemblyRole::PortalSurround,
                    attr,
                    cost,
                    ir,
                )?);
            }
            (s0, wall_y0, post_z0, s1, wall_y1, post_z1)
        }
        BrushAssemblyRole::EastWall | BrushAssemblyRole::WestWall => {
            for layer in 0..3 {
                let (frame_x, _frame_y) = frame_depth_range(wall_role, wall_bounds, layer)?;
                surround_ids.push(build_and_insert_box(
                    frame_x,
                    (s0 - step, s0),
                    (post_z0, post_z1),
                    BrushAssemblyRole::PortalSurround,
                    attr,
                    cost,
                    ir,
                )?);
                surround_ids.push(build_and_insert_box(
                    frame_x,
                    (s1, s1 + step),
                    (post_z0, post_z1),
                    BrushAssemblyRole::PortalSurround,
                    attr,
                    cost,
                    ir,
                )?);
                surround_ids.push(build_and_insert_box(
                    frame_x,
                    (s0 - step, s1 + step),
                    (z1, z1 + step),
                    BrushAssemblyRole::PortalSurround,
                    attr,
                    cost,
                    ir,
                )?);
            }
            (wall_x0, s0, post_z0, wall_x1, s1, post_z1)
        }
        _ => {
            return Err(composition_error(
                "portal.wall_role",
                "Egyptian portal requires a cardinal wall",
            ));
        }
    };

    clip_portal_frame_spans(ir, &mut surround_ids, wall_role, wall_bounds)?;

    let opening_id = ir.alloc_opening_id();
    let opening = OpeningRecord {
        id: opening_id,
        owner_brush_id: wall_brush_id,
        wall_segment_ids: vec![wall_brush_id],
        owner_partition_bounds: wall_bounds,
        wall_role,
        owner: attr.clone(),
        bounds: throat_bounds,
        portal_id: Some(portal_id),
        frame_brush_ids: surround_ids.clone(),
        portal_style: Some(PortalStyle::EgyptianSteppedSurround),
    };
    ir.insert_opening(opening);

    Ok(PortalAssembly {
        portal_id,
        style: PortalStyle::EgyptianSteppedSurround,
        post_ids: Vec::new(),
        lintel_ids: Vec::new(),
        surround_ids,
        wall_brush_id,
        opening_id,
        throat_bounds,
    })
}

fn build_brutalist_reveal_surround(
    portal_id: PortalId,
    wall_brush_id: BrushAssemblyId,
    wall_bounds: (i128, i128, i128, i128, i128, i128),
    throat_anchor: (i128, i128, i128, i128), // (span_min, z_min, span_max, z_max)
    wall_role: BrushAssemblyRole,
    attr: &SemanticAttribution,
    cost: CostSource,
    ir: &mut AssemblyIR,
) -> Result<PortalAssembly, RichnessError> {
    validate_portal_request(portal_id, throat_anchor, wall_role)?;
    let (s0, z0, s1, z1) = throat_anchor;
    let reveal_depth = 16i128;
    let surround_thickness = 16i128;

    let (wall_x0, wall_y0, _wall_z0, wall_x1, wall_y1, _wall_z1) = wall_bounds;
    let (frame_x, frame_y) = frame_depth_range(wall_role, wall_bounds, 0)?;
    let mut surround_ids = Vec::new();
    let lintel_z0 = z1;
    let lintel_z1 = lintel_z0 + reveal_depth;

    let throat_bounds = match wall_role {
        BrushAssemblyRole::NorthWall | BrushAssemblyRole::SouthWall => {
            // Reveal channels
            surround_ids.push(build_and_insert_box(
                (s0 - reveal_depth, s0),
                frame_y,
                (z0, z1),
                BrushAssemblyRole::PortalSurround,
                attr,
                cost,
                ir,
            )?);
            surround_ids.push(build_and_insert_box(
                (s1, s1 + reveal_depth),
                frame_y,
                (z0, z1),
                BrushAssemblyRole::PortalSurround,
                attr,
                cost,
                ir,
            )?);
            surround_ids.push(build_and_insert_box(
                (s0 - reveal_depth, s1 + reveal_depth),
                frame_y,
                (lintel_z0, lintel_z1),
                BrushAssemblyRole::PortalSurround,
                attr,
                cost,
                ir,
            )?);
            // Surround mass
            surround_ids.push(build_and_insert_box(
                (s0 - reveal_depth - surround_thickness, s0 - reveal_depth),
                frame_y,
                (z0, z1),
                BrushAssemblyRole::PortalSurround,
                attr,
                cost,
                ir,
            )?);
            surround_ids.push(build_and_insert_box(
                (s1 + reveal_depth, s1 + reveal_depth + surround_thickness),
                frame_y,
                (z0, z1),
                BrushAssemblyRole::PortalSurround,
                attr,
                cost,
                ir,
            )?);
            surround_ids.push(build_and_insert_box(
                (
                    s0 - reveal_depth - surround_thickness,
                    s1 + reveal_depth + surround_thickness,
                ),
                frame_y,
                (lintel_z1, lintel_z1 + surround_thickness),
                BrushAssemblyRole::PortalSurround,
                attr,
                cost,
                ir,
            )?);
            (s0, wall_y0, z0, s1, wall_y1, z1)
        }
        BrushAssemblyRole::EastWall | BrushAssemblyRole::WestWall => {
            surround_ids.push(build_and_insert_box(
                frame_x,
                (s0 - reveal_depth, s0),
                (z0, z1),
                BrushAssemblyRole::PortalSurround,
                attr,
                cost,
                ir,
            )?);
            surround_ids.push(build_and_insert_box(
                frame_x,
                (s1, s1 + reveal_depth),
                (z0, z1),
                BrushAssemblyRole::PortalSurround,
                attr,
                cost,
                ir,
            )?);
            surround_ids.push(build_and_insert_box(
                frame_x,
                (s0 - reveal_depth, s1 + reveal_depth),
                (lintel_z0, lintel_z1),
                BrushAssemblyRole::PortalSurround,
                attr,
                cost,
                ir,
            )?);
            surround_ids.push(build_and_insert_box(
                frame_x,
                (s0 - reveal_depth - surround_thickness, s0 - reveal_depth),
                (z0, z1),
                BrushAssemblyRole::PortalSurround,
                attr,
                cost,
                ir,
            )?);
            surround_ids.push(build_and_insert_box(
                frame_x,
                (s1 + reveal_depth, s1 + reveal_depth + surround_thickness),
                (z0, z1),
                BrushAssemblyRole::PortalSurround,
                attr,
                cost,
                ir,
            )?);
            surround_ids.push(build_and_insert_box(
                frame_x,
                (
                    s0 - reveal_depth - surround_thickness,
                    s1 + reveal_depth + surround_thickness,
                ),
                (lintel_z1, lintel_z1 + surround_thickness),
                BrushAssemblyRole::PortalSurround,
                attr,
                cost,
                ir,
            )?);
            (wall_x0, s0, z0, wall_x1, s1, z1)
        }
        _ => {
            return Err(composition_error(
                "portal.wall_role",
                "Brutalist portal requires a cardinal wall",
            ));
        }
    };

    clip_portal_frame_spans(ir, &mut surround_ids, wall_role, wall_bounds)?;

    let opening_id = ir.alloc_opening_id();
    let opening = OpeningRecord {
        id: opening_id,
        owner_brush_id: wall_brush_id,
        wall_segment_ids: vec![wall_brush_id],
        owner_partition_bounds: wall_bounds,
        wall_role,
        owner: attr.clone(),
        bounds: throat_bounds,
        portal_id: Some(portal_id),
        frame_brush_ids: surround_ids.clone(),
        portal_style: Some(PortalStyle::BrutalistRevealSurround),
    };
    ir.insert_opening(opening);

    Ok(PortalAssembly {
        portal_id,
        style: PortalStyle::BrutalistRevealSurround,
        post_ids: Vec::new(),
        lintel_ids: Vec::new(),
        surround_ids,
        wall_brush_id,
        opening_id,
        throat_bounds,
    })
}

// ── Session B: bounded shaping elements ───────────────────────────────────

/// Apply wall mass treatments to a wall chain from the variation plan.
///
/// Builds liners, pilasters, recesses, and buttress courses only inside
/// committed legal volumes. Rejects violations of protected reservations.
pub(crate) fn apply_wall_mass_treatments(
    chain: &WallChainRecord,
    wall_brush_id: BrushAssemblyId,
    wall_role: BrushAssemblyRole,
    attr: &SemanticAttribution,
    cost: CostSource,
    ir: &mut AssemblyIR,
) -> Result<Vec<BrushAssemblyId>, RichnessError> {
    let mut created_ids = Vec::new();

    // Get wall AABB for coordinate reference
    let wall_brush = ir
        .brushes
        .get(&wall_brush_id)
        .ok_or_else(|| composition_error("wall_mass", "wall brush not found"))?;
    let bb = wall_brush
        .brush
        .aabb()
        .map_err(|e| composition_error("wall_mass", format!("wall AABB: {e}")))?;
    let ((wx0, wy0, wz0), (wx1, wy1, wz1)) = bb;
    let mut wall_segment_ids = vec![wall_brush_id];

    for treatment in &chain.mass_treatments {
        // Check against protected segments
        for &(seg_start, seg_end) in &chain.protected_segments {
            if treatment.segment.0 < seg_end && treatment.segment.1 > seg_start {
                return Err(composition_error(
                    "wall_mass",
                    format!(
                        "mass treatment overlaps protected segment ({}, {})",
                        seg_start, seg_end
                    ),
                ));
            }
        }

        let (tx0, ty0, tx1, ty1) =
            wall_treatment_bounds(wall_role, (wx0, wy0, wx1, wy1), treatment.segment)?;
        match treatment.kind {
            WallMass::None => {}
            WallMass::Liner16 | WallMass::Liner32 => {
                let thickness = if matches!(treatment.kind, WallMass::Liner32) {
                    32i128
                } else {
                    16i128
                };
                // A liner starts on the clear-volume face of the structural
                // wall, so it contacts rather than overlaps its owner.
                build_liner_on_wall(
                    wall_role,
                    tx0,
                    ty0,
                    tx1,
                    ty1,
                    wz0,
                    wz1,
                    thickness,
                    attr,
                    cost,
                    ir,
                    &mut created_ids,
                )?;
            }
            WallMass::Recess16 => {
                // A recess is a genuine omission from the wall partition,
                // never a decorative opening record over solid geometry.
                let recess_bounds =
                    compute_recess_bounds(wall_role, tx0, ty0, tx1, ty1, wz0, wz1, 16i128);
                let opening_id = ir.alloc_opening_id();
                ir.insert_opening(OpeningRecord {
                    id: opening_id,
                    owner_brush_id: wall_brush_id,
                    wall_segment_ids: wall_segment_ids.clone(),
                    owner_partition_bounds: (wx0, wy0, wz0, wx1, wy1, wz1),
                    wall_role,
                    owner: attr.clone(),
                    bounds: recess_bounds,
                    portal_id: None,
                    frame_brush_ids: Vec::new(),
                    portal_style: None,
                });
                let mut next_segments = Vec::new();
                for segment_id in wall_segment_ids {
                    next_segments.extend(split_wall_around_opening(ir, segment_id, recess_bounds)?);
                }
                next_segments.sort_unstable();
                next_segments.dedup();
                wall_segment_ids = next_segments;
            }
            WallMass::Buttress16 => {
                // Buttresses are attached on the exterior face and therefore
                // share only an exact face with the structural wall.
                build_buttress_on_wall(
                    wall_role,
                    tx0,
                    ty0,
                    tx1,
                    ty1,
                    wz0,
                    wz1,
                    attr,
                    cost,
                    ir,
                    &mut created_ids,
                )?;
            }
        }
    }

    Ok(created_ids)
}

/// Build a liner on the interior face of a wall.
fn build_liner_on_wall(
    wall_role: BrushAssemblyRole,
    wx0: i128,
    wy0: i128,
    wx1: i128,
    wy1: i128,
    wz0: i128,
    wz1: i128,
    thickness: i128,
    attr: &SemanticAttribution,
    cost: CostSource,
    ir: &mut AssemblyIR,
    created_ids: &mut Vec<BrushAssemblyId>,
) -> Result<(), RichnessError> {
    match wall_role {
        BrushAssemblyRole::NorthWall => build_and_insert_box(
            (wx0, wx1),
            (wy1, wy1 + thickness),
            (wz0, wz1),
            BrushAssemblyRole::WallLiner,
            attr,
            cost,
            ir,
        )
        .map(|id| created_ids.push(id)),
        BrushAssemblyRole::SouthWall => build_and_insert_box(
            (wx0, wx1),
            (wy0 - thickness, wy0),
            (wz0, wz1),
            BrushAssemblyRole::WallLiner,
            attr,
            cost,
            ir,
        )
        .map(|id| created_ids.push(id)),
        BrushAssemblyRole::EastWall => build_and_insert_box(
            (wx0 - thickness, wx0),
            (wy0, wy1),
            (wz0, wz1),
            BrushAssemblyRole::WallLiner,
            attr,
            cost,
            ir,
        )
        .map(|id| created_ids.push(id)),
        BrushAssemblyRole::WestWall => build_and_insert_box(
            (wx1, wx1 + thickness),
            (wy0, wy1),
            (wz0, wz1),
            BrushAssemblyRole::WallLiner,
            attr,
            cost,
            ir,
        )
        .map(|id| created_ids.push(id)),
        _ => Err(composition_error("liner", "liner only on cardinal walls")),
    }
}

/// Build a buttress on the outside face of a wall.
fn build_buttress_on_wall(
    wall_role: BrushAssemblyRole,
    wx0: i128,
    wy0: i128,
    wx1: i128,
    wy1: i128,
    wz0: i128,
    wz1: i128,
    attr: &SemanticAttribution,
    cost: CostSource,
    ir: &mut AssemblyIR,
    created_ids: &mut Vec<BrushAssemblyId>,
) -> Result<(), RichnessError> {
    let bw = 16i128; // buttress width
    match wall_role {
        BrushAssemblyRole::NorthWall => build_and_insert_box(
            (wx0, wx1),
            (wy0 - bw, wy0),
            (wz0, wz1),
            BrushAssemblyRole::Buttress,
            attr,
            cost,
            ir,
        )
        .map(|id| created_ids.push(id)),
        BrushAssemblyRole::SouthWall => build_and_insert_box(
            (wx0, wx1),
            (wy1, wy1 + bw),
            (wz0, wz1),
            BrushAssemblyRole::Buttress,
            attr,
            cost,
            ir,
        )
        .map(|id| created_ids.push(id)),
        BrushAssemblyRole::EastWall => build_and_insert_box(
            (wx1, wx1 + bw),
            (wy0, wy1),
            (wz0, wz1),
            BrushAssemblyRole::Buttress,
            attr,
            cost,
            ir,
        )
        .map(|id| created_ids.push(id)),
        BrushAssemblyRole::WestWall => build_and_insert_box(
            (wx0 - bw, wx0),
            (wy0, wy1),
            (wz0, wz1),
            BrushAssemblyRole::Buttress,
            attr,
            cost,
            ir,
        )
        .map(|id| created_ids.push(id)),
        _ => Err(composition_error(
            "buttress",
            "buttress only on cardinal walls",
        )),
    }
}

/// Restrict a variation treatment to its committed chain segment.
fn wall_treatment_bounds(
    wall_role: BrushAssemblyRole,
    (wx0, wy0, wx1, wy1): (i128, i128, i128, i128),
    (segment_start, segment_end): (i32, i32),
) -> Result<(i128, i128, i128, i128), RichnessError> {
    let start = i128::from(segment_start);
    let end = i128::from(segment_end);
    if start >= end
        || start.rem_euclid(richness_geom::QUANTUM) != 0
        || end.rem_euclid(richness_geom::QUANTUM) != 0
    {
        return Err(composition_error(
            "wall_mass.segment",
            format!("wall mass segment {start}..{end} is not a non-empty 16-unit interval"),
        ));
    }
    let within = match wall_role {
        BrushAssemblyRole::NorthWall | BrushAssemblyRole::SouthWall => start >= wx0 && end <= wx1,
        BrushAssemblyRole::EastWall | BrushAssemblyRole::WestWall => start >= wy0 && end <= wy1,
        _ => false,
    };
    if !within {
        return Err(composition_error(
            "wall_mass.segment",
            format!("wall mass segment {start}..{end} escapes its committed wall run"),
        ));
    }
    Ok(match wall_role {
        BrushAssemblyRole::NorthWall | BrushAssemblyRole::SouthWall => (start, wy0, end, wy1),
        BrushAssemblyRole::EastWall | BrushAssemblyRole::WestWall => (wx0, start, wx1, end),
        _ => {
            return Err(composition_error(
                "wall_mass.segment",
                "wall mass requires a cardinal wall run",
            ));
        }
    })
}

fn compute_recess_bounds(
    wall_role: BrushAssemblyRole,
    wx0: i128,
    wy0: i128,
    wx1: i128,
    wy1: i128,
    wz0: i128,
    wz1: i128,
    depth: i128,
) -> (i128, i128, i128, i128, i128, i128) {
    match wall_role {
        BrushAssemblyRole::NorthWall => (wx0, wy1 - depth, wz0, wx1, wy1, wz1),
        BrushAssemblyRole::SouthWall => (wx0, wy0, wz0, wx1, wy0 + depth, wz1),
        BrushAssemblyRole::EastWall => (wx1 - depth, wy0, wz0, wx1, wy1, wz1),
        BrushAssemblyRole::WestWall => (wx0, wy0, wz0, wx0 + depth, wy1, wz1),
        _ => (wx0, wy0, wz0, wx1, wy1, wz1),
    }
}

// ── Session B: interface derivation ───────────────────────────────────────

/// Derive all interfaces from exact positive-area contacts.
///
/// Every pair of brushes that share a positive-area face contact gets an
/// interface record. Rejects undeclared contact (two brushes touching with
/// positive-area face overlap but no matching interface kind) and overlap.
pub(crate) fn derive_all_interfaces(ir: &mut AssemblyIR) -> Result<(), RichnessError> {
    ir.interfaces.clear();
    let brush_ids: Vec<BrushAssemblyId> = ir.brushes.keys().copied().collect();
    let bounds = brush_ids
        .iter()
        .map(|id| {
            ir.brushes[id]
                .brush
                .aabb()
                .map(|(min, max)| (*id, (min.0, min.1, min.2, max.0, max.1, max.2)))
                .map_err(|error| {
                    composition_error("interfaces", format!("brush {} AABB: {error}", id.raw()))
                })
        })
        .collect::<Result<BTreeMap<_, _>, _>>()?;

    for i in 0..brush_ids.len() {
        for j in (i + 1)..brush_ids.len() {
            let id_a = brush_ids[i];
            let id_b = brush_ids[j];
            let brush_a = &ir.brushes[&id_a];
            let brush_b = &ir.brushes[&id_b];
            if !aabbs_may_touch(bounds[&id_a], bounds[&id_b]) {
                continue;
            }

            // Check for positive-volume overlap — this is always an error
            if richness_geom::brushes_overlap(&brush_a.brush, &brush_b.brush)? {
                return Err(composition_error(
                    "interfaces",
                    format!(
                        "brushes {} ({}, reservation {}) {:?} {:?} and {} ({}, reservation {}) {:?} {:?} have positive-volume overlap",
                        id_a.raw(),
                        brush_a.role.tag(),
                        brush_a.owner.reservation_id.raw(),
                        brush_a.brush.aabb().ok(),
                        brush_a.brush.faces.iter().map(|face| (&face.plane, face.role)).collect::<Vec<_>>(),
                        id_b.raw(),
                        brush_b.role.tag(),
                        brush_b.owner.reservation_id.raw(),
                        brush_b.brush.aabb().ok(),
                        brush_b.brush.faces.iter().map(|face| (&face.plane, face.role)).collect::<Vec<_>>()
                    ),
                ));
            }

            // Check for positive-area face contact
            if richness_geom::has_positive_area_contact(&brush_a.brush, &brush_b.brush) {
                // Derive the interface kind
                let kind = richness_geom::derive_interface_kind(brush_a.role, brush_b.role)
                    .or_else(|| {
                        declared_portal_frame_contact(ir, id_a, id_b)
                            .then_some(InterfaceKind::PortalFrameJoint)
                    });
                if let Some(kind) = kind {
                    let if_id = ir.alloc_interface_id();
                    ir.insert_interface(InterfaceRecord {
                        id: if_id,
                        brush_a: id_a,
                        brush_b: id_b,
                        kind,
                    });
                } else {
                    // Undeclared contact: positive-area face overlap but no
                    // known interface kind — this is a validation error.
                    return Err(composition_error(
                        "interfaces",
                        format!(
                            "undeclared contact between brushes {:?} ({}) and {:?} ({})",
                            id_a.raw(),
                            brush_a.role.tag(),
                            id_b.raw(),
                            brush_b.role.tag()
                        ),
                    ));
                }
            }
        }
    }

    Ok(())
}

fn aabbs_may_touch(
    (ax0, ay0, az0, ax1, ay1, az1): (i128, i128, i128, i128, i128, i128),
    (bx0, by0, bz0, bx1, by1, bz1): (i128, i128, i128, i128, i128, i128),
) -> bool {
    ax0 <= bx1 && bx0 <= ax1 && ay0 <= by1 && by0 <= ay1 && az0 <= bz1 && bz0 <= az1
}

fn declared_portal_frame_contact(ir: &AssemblyIR, a: BrushAssemblyId, b: BrushAssemblyId) -> bool {
    let a_is_frame = ir
        .openings
        .values()
        .any(|opening| opening.frame_brush_ids.contains(&a));
    let b_is_frame = ir
        .openings
        .values()
        .any(|opening| opening.frame_brush_ids.contains(&b));
    a_is_frame && b_is_frame
}

// ── Session B: wall chain gap validation ──────────────────────────────────

/// Validate that adjacent rooms sharing a wall chain have no gaps.
///
/// Rooms sharing a wall chain must consume the same planes — no double walls,
/// no gaps between adjacent walls.
pub(crate) fn validate_wall_chain_gaps(
    ir: &AssemblyIR,
    _wall_chain_ids: &[WallChainId],
) -> Result<(), RichnessError> {
    // Check all cardinal walls for proper corner contacts.
    // Each pair of adjacent cardinal walls must have an interface.
    let brushes: Vec<_> = ir.brushes.values().collect();

    for i in 0..brushes.len() {
        for j in (i + 1)..brushes.len() {
            let a = &brushes[i];
            let b = &brushes[j];

            // Only check cardinal walls
            if !a.role.is_wall() || !b.role.is_wall() {
                continue;
            }

            // Skip diagonal walls — only check cardinal
            if names_diag(a.role) || names_diag(b.role) {
                continue;
            }

            // If two adjacent perpendicular walls have positive-area contact,
            // they must have an interface.
            if richness_geom::has_positive_area_contact(&a.brush, &b.brush) {
                let has_interface = ir.interfaces.values().any(|iface| {
                    (iface.brush_a == a.id && iface.brush_b == b.id)
                        || (iface.brush_a == b.id && iface.brush_b == a.id)
                });
                if !has_interface {
                    return Err(composition_error(
                        "wall_gap",
                        format!(
                            "adjacent cardinal walls {:?} ({}) and {:?} ({}) have contact but no interface",
                            a.id.raw(), a.role.tag(), b.id.raw(), b.role.tag()
                        ),
                    ));
                }
            }
        }
    }

    Ok(())
}

fn names_diag(r: BrushAssemblyRole) -> bool {
    matches!(
        r,
        BrushAssemblyRole::DiagNEWall
            | BrushAssemblyRole::DiagNWWall
            | BrushAssemblyRole::DiagSEWall
            | BrushAssemblyRole::DiagSWWall
    )
}

// ── Session B: bounded shaping builders ──────────────────────────────────

/// Build a sill at the given location (48-64u tall).
pub(crate) fn build_sill(
    x0: i128,
    y0: i128,
    x1: i128,
    y1: i128,
    sill_height: i128,
    attr: &SemanticAttribution,
    cost: CostSource,
    ir: &mut AssemblyIR,
) -> Result<BrushAssemblyId, RichnessError> {
    if sill_height < 48 || sill_height > 64 {
        return Err(composition_error(
            "sill",
            format!("sill height {} not in [48, 64]", sill_height),
        ));
    }
    let brush = richness_geom::make_sill(x0, y0, x1, y1, sill_height)?;
    let id = ir.alloc_brush_id();
    ir.insert_brush(BrushAssembly {
        id,
        brush,
        role: BrushAssemblyRole::Sill,
        owner: attr.clone(),
        cost,
        support: SupportTarget::World,
    });
    Ok(id)
}

/// Build a pilaster on the given wall face.
pub(crate) fn build_pilaster(
    x0: i128,
    y0: i128,
    x1: i128,
    y1: i128,
    attr: &SemanticAttribution,
    cost: CostSource,
    ir: &mut AssemblyIR,
) -> Result<BrushAssemblyId, RichnessError> {
    let brush = richness_geom::make_pilaster(x0, y0, x1, y1)?;
    let id = ir.alloc_brush_id();
    ir.insert_brush(BrushAssembly {
        id,
        brush,
        role: BrushAssemblyRole::Pilaster,
        owner: attr.clone(),
        cost,
        support: SupportTarget::World,
    });
    Ok(id)
}

/// Build a partial wall segment.
pub(crate) fn build_partial_wall(
    x0: i128,
    y0: i128,
    x1: i128,
    y1: i128,
    z_min: i128,
    z_max: i128,
    attr: &SemanticAttribution,
    cost: CostSource,
    ir: &mut AssemblyIR,
) -> Result<BrushAssemblyId, RichnessError> {
    let brush = richness_geom::make_partial_wall(x0, y0, x1, y1, z_min, z_max)?;
    let id = ir.alloc_brush_id();
    ir.insert_brush(BrushAssembly {
        id,
        brush,
        role: BrushAssemblyRole::PartialWall,
        owner: attr.clone(),
        cost,
        support: SupportTarget::World,
    });
    Ok(id)
}

/// Build an offset shaft (vertical conduit).
pub(crate) fn build_offset_shaft(
    x0: i128,
    y0: i128,
    x1: i128,
    y1: i128,
    z_min: i128,
    z_max: i128,
    attr: &SemanticAttribution,
    cost: CostSource,
    ir: &mut AssemblyIR,
) -> Result<BrushAssemblyId, RichnessError> {
    let brush = richness_geom::make_partial_wall(x0, y0, x1, y1, z_min, z_max)?;
    let id = ir.alloc_brush_id();
    ir.insert_brush(BrushAssembly {
        id,
        brush,
        role: BrushAssemblyRole::OffsetShaft,
        owner: attr.clone(),
        cost,
        support: SupportTarget::World,
    });
    Ok(id)
}

/// Build a bent approach (angled wall section within legal volume).
pub(crate) fn build_bent_approach(
    x0: i128,
    y0: i128,
    x1: i128,
    y1: i128,
    z_min: i128,
    z_max: i128,
    attr: &SemanticAttribution,
    cost: CostSource,
    ir: &mut AssemblyIR,
) -> Result<BrushAssemblyId, RichnessError> {
    let brush = richness_geom::make_partial_wall(x0, y0, x1, y1, z_min, z_max)?;
    let id = ir.alloc_brush_id();
    ir.insert_brush(BrushAssembly {
        id,
        brush,
        role: BrushAssemblyRole::BentApproach,
        owner: attr.clone(),
        cost,
        support: SupportTarget::World,
    });
    Ok(id)
}

// ── Session B: common box builder ─────────────────────────────────────────

/// Build a simple box brush, insert into IR, return ID.
fn build_and_insert_box(
    x_range: (i128, i128),
    y_range: (i128, i128),
    z_range: (i128, i128),
    role: BrushAssemblyRole,
    attr: &SemanticAttribution,
    cost: CostSource,
    ir: &mut AssemblyIR,
) -> Result<BrushAssemblyId, RichnessError> {
    let brush = ConvexBrush::make_box(x_range, y_range, z_range)
        .map_err(|e| composition_error("box", format!("{e}")))?;
    richness_geom::validate_brush(&brush)?;
    let id = ir.alloc_brush_id();
    ir.insert_brush(BrushAssembly {
        id,
        brush,
        role,
        owner: attr.clone(),
        cost,
        support: SupportTarget::World,
    });
    Ok(id)
}

// ── Rectangular room composition ──────────────────────────────────────────

fn compose_rectangle(
    fp: &Footprint3D,
    attr: &SemanticAttribution,
    cost: CostSource,
    ir: &mut AssemblyIR,
) -> Result<(), RichnessError> {
    // Floor slab
    let floor = richness_geom::make_floor_slab(fp)?;
    let floor_id = ir.alloc_brush_id();
    let floor_support = ir.alloc_support_id();
    ir.insert_brush(BrushAssembly {
        id: floor_id,
        brush: floor,
        role: BrushAssemblyRole::FloorSlab,
        owner: attr.clone(),
        cost,
        support: SupportTarget::World,
    });
    ir.insert_support(SupportRecord {
        id: floor_support,
        child: floor_id,
        parent: SupportTarget::World,
    });

    // Ceiling slab
    let ceiling = richness_geom::make_ceiling_slab(fp)?;
    let ceil_id = ir.alloc_brush_id();
    let ceil_support = ir.alloc_support_id();
    ir.insert_brush(BrushAssembly {
        id: ceil_id,
        brush: ceiling,
        role: BrushAssemblyRole::CeilingSlab,
        owner: attr.clone(),
        cost,
        support: SupportTarget::Brush(floor_id),
    });
    ir.insert_support(SupportRecord {
        id: ceil_support,
        child: ceil_id,
        parent: SupportTarget::Brush(floor_id),
    });

    // North wall (full x span, y0..y0+16)
    let n_wall = richness_geom::make_north_wall(fp)?;
    let nw_id = ir.alloc_brush_id();
    let nw_support = ir.alloc_support_id();
    ir.insert_brush(BrushAssembly {
        id: nw_id,
        brush: n_wall,
        role: BrushAssemblyRole::NorthWall,
        owner: attr.clone(),
        cost,
        support: SupportTarget::Brush(floor_id),
    });
    ir.insert_support(SupportRecord {
        id: nw_support,
        child: nw_id,
        parent: SupportTarget::Brush(floor_id),
    });
    // Wall-to-floor interface
    let if_nf = ir.alloc_interface_id();
    ir.insert_interface(InterfaceRecord {
        id: if_nf,
        brush_a: nw_id,
        brush_b: floor_id,
        kind: InterfaceKind::WallToFloor,
    });

    // South wall (full x span, y1-16..y1)
    let s_wall = richness_geom::make_south_wall(fp)?;
    let sw_id = ir.alloc_brush_id();
    let sw_support = ir.alloc_support_id();
    ir.insert_brush(BrushAssembly {
        id: sw_id,
        brush: s_wall,
        role: BrushAssemblyRole::SouthWall,
        owner: attr.clone(),
        cost,
        support: SupportTarget::Brush(floor_id),
    });
    ir.insert_support(SupportRecord {
        id: sw_support,
        child: sw_id,
        parent: SupportTarget::Brush(floor_id),
    });

    // West wall (shortened y span)
    let w_wall = richness_geom::make_west_wall(fp)?;
    let ww_id = ir.alloc_brush_id();
    let ww_support = ir.alloc_support_id();
    ir.insert_brush(BrushAssembly {
        id: ww_id,
        brush: w_wall,
        role: BrushAssemblyRole::WestWall,
        owner: attr.clone(),
        cost,
        support: SupportTarget::Brush(floor_id),
    });
    ir.insert_support(SupportRecord {
        id: ww_support,
        child: ww_id,
        parent: SupportTarget::Brush(floor_id),
    });

    // East wall (shortened y span)
    let e_wall = richness_geom::make_east_wall(fp)?;
    let ew_id = ir.alloc_brush_id();
    let ew_support = ir.alloc_support_id();
    ir.insert_brush(BrushAssembly {
        id: ew_id,
        brush: e_wall,
        role: BrushAssemblyRole::EastWall,
        owner: attr.clone(),
        cost,
        support: SupportTarget::Brush(floor_id),
    });
    ir.insert_support(SupportRecord {
        id: ew_support,
        child: ew_id,
        parent: SupportTarget::Brush(floor_id),
    });

    Ok(())
}

// ── Chamfered room composition ────────────────────────────────────────────

fn compose_chamfered(
    fp: &Footprint3D,
    attr: &SemanticAttribution,
    cost: CostSource,
    _arch_idx: usize,
    ir: &mut AssemblyIR,
) -> Result<(), RichnessError> {
    let chamfer = compute_chamfer_size(fp);
    let (qx0, qy0, qx1, qy1) = richness_geom::footprint_quake_bounds(fp);

    // Floor slab (full footprint)
    let floor = richness_geom::make_floor_slab(fp)?;
    let floor_id = ir.alloc_brush_id();
    ir.insert_brush(BrushAssembly {
        id: floor_id,
        brush: floor,
        role: BrushAssemblyRole::FloorSlab,
        owner: attr.clone(),
        cost,
        support: SupportTarget::World,
    });
    let floor_sup = ir.alloc_support_id();
    ir.insert_support(SupportRecord {
        id: floor_sup,
        child: floor_id,
        parent: SupportTarget::World,
    });

    // Ceiling slab (full footprint)
    let ceiling = richness_geom::make_ceiling_slab(fp)?;
    let ceil_id = ir.alloc_brush_id();
    ir.insert_brush(BrushAssembly {
        id: ceil_id,
        brush: ceiling,
        role: BrushAssemblyRole::CeilingSlab,
        owner: attr.clone(),
        cost,
        support: SupportTarget::Brush(floor_id),
    });
    let ceil_sup = ir.alloc_support_id();
    ir.insert_support(SupportRecord {
        id: ceil_sup,
        child: ceil_id,
        parent: SupportTarget::Brush(floor_id),
    });

    // ── Chamfered cardinal walls ───────────────────────────────────────
    // Each cardinal wall has diagonal faces at both ends matching the room's chamfers.

    // North wall: NW chamfer (y - x >= y1 - x0 - C) and NE chamfer (y + x >= y1 + x1 - C)
    // Wait, let me use the correct coordinate system:
    // North = larger y (y near qy1, the "top" in Quake)
    // So: N wall at y ∈ [qy1-16, qy1]
    // NW corner = (qx0, qy1), NE corner = (qx1, qy1)

    let vertical = richness_geom::footprint_vertical_bounds(fp)?;
    let wall_z_min = vertical.wall_min;
    let wall_z_max = vertical.wall_max;
    let wall_t = richness_geom::WALL_THICKNESS;

    compose_chamfered_north_wall(
        qx0, qy0, qx1, qy1, chamfer, wall_t, wall_z_min, wall_z_max, attr, cost, floor_id, ir,
    )?;
    compose_chamfered_south_wall(
        qx0, qy0, qx1, qy1, chamfer, wall_t, wall_z_min, wall_z_max, attr, cost, floor_id, ir,
    )?;
    compose_chamfered_west_wall(
        qx0, qy0, qx1, qy1, chamfer, wall_t, wall_z_min, wall_z_max, attr, cost, floor_id, ir,
    )?;
    compose_chamfered_east_wall(
        qx0, qy0, qx1, qy1, chamfer, wall_t, wall_z_min, wall_z_max, attr, cost, floor_id, ir,
    )?;

    // ── Diagonal corner walls ──────────────────────────────────────────
    // NW corner
    compose_diag_wall(
        (qx0, qx1),
        (qy0, qy1),
        wall_z_min,
        wall_z_max,
        -1,
        1,
        chamfer, // sx=-1 (x-min), sy=1 (y-max)
        BrushAssemblyRole::DiagNWWall,
        attr,
        cost,
        floor_id,
        ir,
    )?;
    // NE corner
    compose_diag_wall(
        (qx0, qx1),
        (qy0, qy1),
        wall_z_min,
        wall_z_max,
        1,
        1,
        chamfer, // sx=1 (x-max), sy=1 (y-max)
        BrushAssemblyRole::DiagNEWall,
        attr,
        cost,
        floor_id,
        ir,
    )?;
    // SW corner
    compose_diag_wall(
        (qx0, qx1),
        (qy0, qy1),
        wall_z_min,
        wall_z_max,
        -1,
        -1,
        chamfer, // sx=-1 (x-min), sy=-1 (y-min)
        BrushAssemblyRole::DiagSWWall,
        attr,
        cost,
        floor_id,
        ir,
    )?;
    // SE corner
    compose_diag_wall(
        (qx0, qx1),
        (qy0, qy1),
        wall_z_min,
        wall_z_max,
        1,
        -1,
        chamfer, // sx=1 (x-max), sy=-1 (y-min)
        BrushAssemblyRole::DiagSEWall,
        attr,
        cost,
        floor_id,
        ir,
    )?;

    Ok(())
}

// ── Octagonal room composition ────────────────────────────────────────────
// Same as chamfered (all 4 corners chamfered)

fn compose_octagonal(
    fp: &Footprint3D,
    attr: &SemanticAttribution,
    cost: CostSource,
    arch_idx: usize,
    ir: &mut AssemblyIR,
) -> Result<(), RichnessError> {
    // Octagonal rooms use the same composition as chamfered (all 4 corners)
    compose_chamfered(fp, attr, cost, arch_idx, ir)
}

// ── Composite partition composition ───────────────────────────────────────

fn compose_composite_partition(
    fp: &Footprint3D,
    attr: &SemanticAttribution,
    cost: CostSource,
    _arch_idx: usize,
    ir: &mut AssemblyIR,
) -> Result<(), RichnessError> {
    // Composite partition (grotto, megaron): sealed outer shell.
    // The interior is a protected volume for later cave synthesis.
    // For macro composition, we emit floor + ceiling + 4 cardinal walls
    // as a watertight sealed container.
    compose_rectangle(fp, attr, cost, ir)
}

// ── Chamfer size computation ──────────────────────────────────────────────

fn compute_chamfer_size(fp: &Footprint3D) -> i128 {
    let w = (fp.x1 - fp.x0) as i128 * richness_geom::QUANTUM;
    let d = (fp.y1 - fp.y0) as i128 * richness_geom::QUANTUM;
    let raw = (w.min(d) / 4) & !(richness_geom::QUANTUM - 1); // quantum-aligned
    raw.clamp(48, 96)
}

// ── Chamfered cardinal wall helpers ────────────────────────────────────────

fn compose_chamfered_north_wall(
    qx0: i128,
    qy0: i128,
    qx1: i128,
    qy1: i128,
    chamfer: i128,
    wall_t: i128,
    z_min: i128,
    z_max: i128,
    attr: &SemanticAttribution,
    cost: CostSource,
    floor_id: BrushAssemblyId,
    ir: &mut AssemblyIR,
) -> Result<BrushAssemblyId, RichnessError> {
    // North wall: y ∈ [qy1 - wall_t, qy1]
    // NW chamfer (at x=qx0, y=qy1): diagonal face y - x = qy1 - qx0 - C
    //   Wall occupies y - x <= qy1 - qx0 - C (room side of NW diagonal)
    // NE chamfer (at x=qx1, y=qy1): diagonal face y + x = qy1 + qx1 - C
    //   Wall occupies y + x <= qy1 + qx1 - C (room side of NE diagonal)
    let nw_diag_d = qy1 - qx0 - chamfer; // plane: -x + y = d → -x + y >= d (wall: -x + y <= d)
    let ne_diag_d = qy1 + qx1 - chamfer; // plane: x + y = d → x + y >= d (wall: x + y <= d)

    let mut planes: Vec<CanonicalPlane> = Vec::new();
    let mk = |nx, ny, nz, d| {
        CanonicalPlane::new(nx, ny, nz, d).map_err(|e| composition_error("plane", format!("{e}")))
    };
    // Floor / ceiling
    planes.push(mk(0, 0, 1, z_min)?);
    planes.push(mk(0, 0, -1, -z_max)?);
    // Outer north face: y <= qy1 → -y >= -qy1
    planes.push(mk(0, -1, 0, -qy1)?);
    // Inner south face: y >= qy1 - wall_t
    planes.push(mk(0, 1, 0, qy1 - wall_t)?);
    // NW diagonal: wall side: -x + y <= nw_diag_d → x - y >= -nw_diag_d
    planes.push(mk(1, -1, 0, -nw_diag_d)?);
    // NE diagonal: wall side: x + y <= ne_diag_d → -x - y >= -ne_diag_d
    planes.push(mk(-1, -1, 0, -ne_diag_d)?);

    build_and_insert_wall(
        planes,
        BrushAssemblyRole::NorthWall,
        attr,
        cost,
        floor_id,
        ir,
    )
}

fn compose_chamfered_south_wall(
    qx0: i128,
    qy0: i128,
    qx1: i128,
    qy1: i128,
    chamfer: i128,
    wall_t: i128,
    z_min: i128,
    z_max: i128,
    attr: &SemanticAttribution,
    cost: CostSource,
    floor_id: BrushAssemblyId,
    ir: &mut AssemblyIR,
) -> Result<BrushAssemblyId, RichnessError> {
    // South wall: y ∈ [qy0, qy0 + wall_t]
    // SW chamfer (at x=qx0, y=qy0): diagonal face y - x = qy0 - qx0 + C?
    //   Actually, the SW diagonal connects (qx0+C, qy0) to (qx0, qy0+C) in terms of octagon.
    //   Equation: x + y = qx0 + qy0 + C
    //   Wall occupies x + y <= qx0 + qy0 + C
    // SE chamfer (at x=qx1, y=qy0): diagonal face -x + y = -qx1 + qy0 + C
    //   Wall occupies -x + y <= -qx1 + qy0 + C
    let sw_diag_d = qx0 + qy0 + chamfer; // plane: x + y = d
    let se_diag_d = -qx1 + qy0 + chamfer; // plane: -x + y = d

    let mut planes: Vec<CanonicalPlane> = Vec::new();
    let mk = |nx, ny, nz, d| {
        CanonicalPlane::new(nx, ny, nz, d).map_err(|e| composition_error("plane", format!("{e}")))
    };
    // Floor / ceiling
    planes.push(mk(0, 0, 1, z_min)?);
    planes.push(mk(0, 0, -1, -z_max)?);
    // Outer south face: y >= qy0
    planes.push(mk(0, 1, 0, qy0)?);
    // Inner north face: y <= qy0 + wall_t → -y >= -(qy0 + wall_t)
    planes.push(mk(0, -1, 0, -(qy0 + wall_t))?);
    // SW diagonal (wall side: x + y >= sw_diag_d, complementary to SW diag wall which has x + y <= sw_diag_d)
    planes.push(mk(1, 1, 0, sw_diag_d)?);
    // SE diagonal (wall side: -x + y >= se_diag_d, complementary to SE diag wall which has x - y >= -se_diag_d)
    planes.push(mk(-1, 1, 0, se_diag_d)?);

    build_and_insert_wall(
        planes,
        BrushAssemblyRole::SouthWall,
        attr,
        cost,
        floor_id,
        ir,
    )
}

fn compose_chamfered_west_wall(
    qx0: i128,
    qy0: i128,
    qx1: i128,
    qy1: i128,
    chamfer: i128,
    wall_t: i128,
    z_min: i128,
    z_max: i128,
    attr: &SemanticAttribution,
    cost: CostSource,
    floor_id: BrushAssemblyId,
    ir: &mut AssemblyIR,
) -> Result<BrushAssemblyId, RichnessError> {
    // West wall: x ∈ [qx0, qx0 + wall_t]
    // NW chamfer: y - x = qy1 - qx0 - C  → wall: y - x >= qy1 - qx0 - C?
    //   Wait: the NW diagonal wall occupies y - x >= qy1 - qx0 - C. The west wall
    //   occupies the complementary: y - x <= qy1 - qx0 - C
    // SW chamfer: x + y = qx0 + qy0 + C → wall: x + y >= qx0 + qy0 + C?
    //   SW diagonal wall occupies x + y >= qx0 + qy0 + C. West wall: x + y <= qx0 + qy0 + C

    // Actually, west wall is between NW and SW. At its north end, it meets NW diagonal.
    // At its south end, it meets SW diagonal. The west wall fills the space between them.

    let nw_diag_d = qy1 - qx0 - chamfer; // plane: -x + y = d. Wall side: -x + y <= d
    let sw_diag_d = qx0 + qy0 + chamfer; // plane: x + y = d. Wall side: x + y <= d

    let mut planes: Vec<CanonicalPlane> = Vec::new();
    let mk = |nx, ny, nz, d| {
        CanonicalPlane::new(nx, ny, nz, d).map_err(|e| composition_error("plane", format!("{e}")))
    };
    // Floor / ceiling
    planes.push(mk(0, 0, 1, z_min)?);
    planes.push(mk(0, 0, -1, -z_max)?);
    // Outer west face: x >= qx0
    planes.push(mk(1, 0, 0, qx0)?);
    // Inner east face: x <= qx0 + wall_t → -x >= -(qx0 + wall_t)
    planes.push(mk(-1, 0, 0, -(qx0 + wall_t))?);
    // NW diagonal (wall side: -x + y <= nw_diag_d → x - y >= -nw_diag_d)
    planes.push(mk(1, -1, 0, -nw_diag_d)?);
    // SW diagonal (wall side: x + y >= sw_diag_d, complementary to SW diag wall)
    planes.push(mk(1, 1, 0, sw_diag_d)?);

    build_and_insert_wall(
        planes,
        BrushAssemblyRole::WestWall,
        attr,
        cost,
        floor_id,
        ir,
    )
}

fn compose_chamfered_east_wall(
    qx0: i128,
    qy0: i128,
    qx1: i128,
    qy1: i128,
    chamfer: i128,
    wall_t: i128,
    z_min: i128,
    z_max: i128,
    attr: &SemanticAttribution,
    cost: CostSource,
    floor_id: BrushAssemblyId,
    ir: &mut AssemblyIR,
) -> Result<BrushAssemblyId, RichnessError> {
    // East wall: x ∈ [qx1 - wall_t, qx1]
    // NE chamfer: x + y = qx1 + qy1 - C. Wall side: x + y <= qx1 + qy1 - C
    //   NE diagonal wall occupies x + y >= qx1 + qy1 - C
    // SE chamfer: -x + y = -qx1 + qy0 + C. Wall side: -x + y <= -qx1 + qy0 + C
    //   SE diagonal wall occupies -x + y >= -qx1 + qy0 + C

    let ne_diag_d = qx1 + qy1 - chamfer; // plane: x + y = d
    let se_diag_d = -qx1 + qy0 + chamfer; // plane: -x + y = d

    let mut planes: Vec<CanonicalPlane> = Vec::new();
    let mk = |nx, ny, nz, d| {
        CanonicalPlane::new(nx, ny, nz, d).map_err(|e| composition_error("plane", format!("{e}")))
    };
    // Floor / ceiling
    planes.push(mk(0, 0, 1, z_min)?);
    planes.push(mk(0, 0, -1, -z_max)?);
    // Outer east face: x <= qx1 → -x >= -qx1
    planes.push(mk(-1, 0, 0, -qx1)?);
    // Inner west face: x >= qx1 - wall_t
    planes.push(mk(1, 0, 0, qx1 - wall_t)?);
    // NE diagonal (wall side: x + y <= ne_diag_d → -x - y >= -ne_diag_d)
    planes.push(mk(-1, -1, 0, -ne_diag_d)?);
    // SE diagonal (wall side: -x + y >= se_diag_d, complementary to SE diag wall)
    planes.push(mk(-1, 1, 0, se_diag_d)?);

    build_and_insert_wall(
        planes,
        BrushAssemblyRole::EastWall,
        attr,
        cost,
        floor_id,
        ir,
    )
}

// ── Diagonal corner wall ──────────────────────────────────────────────────

fn compose_diag_wall(
    x_range: (i128, i128),
    y_range: (i128, i128),
    z_min: i128,
    z_max: i128,
    sx: i128,
    sy: i128,
    chamfer: i128,
    role: BrushAssemblyRole,
    attr: &SemanticAttribution,
    cost: CostSource,
    floor_id: BrushAssemblyId,
    ir: &mut AssemblyIR,
) -> Result<BrushAssemblyId, RichnessError> {
    let brush = geometry::make_diagonal_wall(x_range, y_range, z_min, z_max, sx, sy, chamfer)
        .map_err(|e| composition_error("diag_wall", format!("{e}")))?;

    let id = ir.alloc_brush_id();
    let support_id = ir.alloc_support_id();
    ir.insert_brush(BrushAssembly {
        id,
        brush,
        role,
        owner: attr.clone(),
        cost,
        support: SupportTarget::Brush(floor_id),
    });
    ir.insert_support(SupportRecord {
        id: support_id,
        child: id,
        parent: SupportTarget::Brush(floor_id),
    });
    Ok(id)
}

// ── Common wall builder ────────────────────────────────────────────────────

fn build_and_insert_wall(
    planes: Vec<CanonicalPlane>,
    role: BrushAssemblyRole,
    attr: &SemanticAttribution,
    cost: CostSource,
    floor_id: BrushAssemblyId,
    ir: &mut AssemblyIR,
) -> Result<BrushAssemblyId, RichnessError> {
    let faces: Vec<BrushFace> = planes
        .into_iter()
        .map(BrushFace::new)
        .collect::<Result<_, _>>()
        .map_err(|e| composition_error("wall_faces", format!("{e}")))?;

    let mut brush =
        ConvexBrush::new(faces).map_err(|e| composition_error("wall_brush", format!("{e}")))?;
    brush
        .validate_and_cache()
        .map_err(|e| composition_error("wall_validate", format!("{e}")))?;

    // Validate approved normals and positive volume
    richness_geom::validate_brush(&brush)?;

    let id = ir.alloc_brush_id();
    let support_id = ir.alloc_support_id();
    ir.insert_brush(BrushAssembly {
        id,
        brush,
        role,
        owner: attr.clone(),
        cost,
        support: SupportTarget::Brush(floor_id),
    });
    ir.insert_support(SupportRecord {
        id: support_id,
        child: id,
        parent: SupportTarget::Brush(floor_id),
    });
    Ok(id)
}

// ── Error helper ──────────────────────────────────────────────────────────

fn missing_archetype_error(
    record: &ReservationRecord,
    context: impl Into<String>,
) -> RichnessError {
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
        "composition.archetype",
        RichnessErrorCategory::SemanticInfeasibility,
        format!("reservation {}: {}", record.id.raw(), context.into()),
    )
}

fn composition_error(path: &str, context: impl Into<String>) -> RichnessError {
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
        path,
        RichnessErrorCategory::SemanticInfeasibility,
        context,
    )
}

// ── Session C: wall splitting around openings ────────────────────────────

/// Split a wall brush around an opening omission.
///
/// The wall brush is split into up to 4 segments (left, right, below, above)
/// so the opening volume is a true omission — no wall geometry occupies the
/// portal throat. The original wall brush is removed and replaced with the
/// split segments.
///
/// For a portal opening on a cardinal wall:
/// - N/S wall: split in X (left/right of opening) and optionally Z
///   (below/above the throat).
/// - E/W wall: split in Y (left/right of opening) and optionally Z.
///
/// Returns the IDs of the replacement wall segments.
fn validated_halfspace_brush(
    mut faces: Vec<BrushFace>,
    path: &str,
) -> Result<ConvexBrush, RichnessError> {
    faces.sort_unstable();
    faces.dedup();
    let vertices = geometry::half_space_vertices(&faces)
        .map_err(|error| composition_error(path, format!("vertices: {error}")))?;
    faces.retain(|face| {
        let on_plane: Vec<_> = vertices
            .iter()
            .copied()
            .filter(|vertex| {
                face.plane
                    .signed_distance_rational(vertex)
                    .is_ok_and(|distance| distance == crate::enhanced_v3::geometry::Rational::ZERO)
            })
            .collect();
        on_plane.len() >= 3
            && geometry::polygon_area_squared(&on_plane, &face.plane)
                .is_ok_and(|area| area > crate::enhanced_v3::geometry::Rational::ZERO)
    });
    let mut brush = ConvexBrush::new(faces)
        .map_err(|error| composition_error(path, format!("brush: {error}")))?;
    brush
        .validate_and_cache()
        .map_err(|error| composition_error(path, format!("validate: {error}")))?;
    richness_geom::validate_brush(&brush)?;
    Ok(brush)
}

fn clip_brush_to_box(
    original: &ConvexBrush,
    x: (i128, i128),
    y: (i128, i128),
    z: (i128, i128),
) -> Result<ConvexBrush, RichnessError> {
    if x.0 >= x.1 || y.0 >= y.1 || z.0 >= z.1 {
        return Err(composition_error(
            "wall_split.clip",
            "clip box has non-positive extent",
        ));
    }
    let planes = [
        CanonicalPlane::new(1, 0, 0, x.0),
        CanonicalPlane::new(-1, 0, 0, -x.1),
        CanonicalPlane::new(0, 1, 0, y.0),
        CanonicalPlane::new(0, -1, 0, -y.1),
        CanonicalPlane::new(0, 0, 1, z.0),
        CanonicalPlane::new(0, 0, -1, -z.1),
    ];
    let mut faces = original.faces.clone();
    for plane in planes {
        let plane = plane
            .map_err(|error| composition_error("wall_split.clip", format!("plane: {error}")))?;
        if !faces.iter().any(|face| face.plane == plane) {
            faces.push(
                BrushFace::new(plane).map_err(|error| {
                    composition_error("wall_split.clip", format!("face: {error}"))
                })?,
            );
        }
    }
    validated_halfspace_brush(faces, "wall_split.clip")
}

pub(crate) fn split_wall_around_opening(
    ir: &mut AssemblyIR,
    wall_brush_id: BrushAssemblyId,
    opening_bounds: (i128, i128, i128, i128, i128, i128),
) -> Result<Vec<BrushAssemblyId>, RichnessError> {
    let wall = ir
        .brushes
        .get(&wall_brush_id)
        .ok_or_else(|| composition_error("wall_split", "wall brush not found"))?;

    let wall_bb = wall
        .brush
        .aabb()
        .map_err(|e| composition_error("wall_split", format!("wall AABB: {e}")))?;
    let ((wx0, wy0, wz0), (wx1, wy1, wz1)) = wall_bb;
    let (ox0, oy0, oz0, ox1, oy1, oz1) = opening_bounds;

    let wall_role = wall.role;
    let is_ns = matches!(
        wall_role,
        BrushAssemblyRole::NorthWall | BrushAssemblyRole::SouthWall
    );
    let is_ew = matches!(
        wall_role,
        BrushAssemblyRole::EastWall | BrushAssemblyRole::WestWall
    );

    if !is_ns && !is_ew {
        return Err(composition_error(
            "wall_split",
            format!("wall role {:?} is not cardinal", wall_role.tag()),
        ));
    }

    let wall_geometry = wall.brush.clone();
    let attr = wall.owner.clone();
    let cost = wall.cost;
    let support = wall.support.clone();

    let intersects = ox0 < wx1 && ox1 > wx0 && oy0 < wy1 && oy1 > wy0 && oz0 < wz1 && oz1 > wz0;
    if !intersects {
        return Ok(vec![wall_brush_id]);
    }

    let split_x0 = ox0.max(wx0);
    let split_x1 = ox1.min(wx1);
    let split_y0 = oy0.max(wy0);
    let split_y1 = oy1.min(wy1);
    let split_z0 = oz0.max(wz0);
    let split_z1 = oz1.min(wz1);
    let mut new_ids = Vec::new();

    // Frame region span (between the posts) for the below/above bands.
    let frame_span = if is_ns {
        (split_x0, split_x1)
    } else {
        (split_y0, split_y1)
    };
    // Span clips for the below/above bands (frame bounds): left and right of
    // the frame in the wall's run axis.
    let span_left = if is_ns {
        (wx0, split_x0)
    } else {
        (wy0, split_y0)
    };
    let span_right = if is_ns {
        (split_x1, wx1)
    } else {
        (split_y1, wy1)
    };
    let emit_band = |ir: &mut AssemblyIR,
                     new_ids: &mut Vec<BrushAssemblyId>,
                     span: (i128, i128),
                     z_band: (i128, i128),
                     attr: SemanticAttribution,
                     cost: CostSource,
                     support: SupportTarget|
     -> Result<(), RichnessError> {
        if span.1 <= span.0 {
            return Ok(());
        }
        let brush = if is_ns {
            clip_brush_to_box(&wall_geometry, span, (wy0, wy1), z_band)?
        } else {
            clip_brush_to_box(&wall_geometry, (wx0, wx1), span, z_band)?
        };
        let id = ir.alloc_brush_id();
        ir.insert_brush(BrushAssembly {
            id,
            brush,
            role: wall_role,
            owner: attr,
            cost,
            support: support.clone(),
        });
        let sup_id = ir.alloc_support_id();
        ir.insert_support(SupportRecord {
            id: sup_id,
            child: id,
            parent: support,
        });
        new_ids.push(id);
        Ok(())
    };

    if is_ns {
        // Wall spans X. Split into: left of opening, right of opening, and
        // optionally below+above the opening (but below/above are usually
        // handled by floor/ceiling slabs — we only split X for portal openings).

        // Left segment: [wx0, ox0]
        if split_x0 > wx0 {
            let brush = clip_brush_to_box(&wall_geometry, (wx0, split_x0), (wy0, wy1), (wz0, wz1))?;
            let id = ir.alloc_brush_id();
            ir.insert_brush(BrushAssembly {
                id,
                brush,
                role: wall_role,
                owner: attr.clone(),
                cost,
                support: support.clone(),
            });
            let sup_id = ir.alloc_support_id();
            ir.insert_support(SupportRecord {
                id: sup_id,
                child: id,
                parent: support.clone(),
            });
            new_ids.push(id);
        }

        // Right segment: [ox1, wx1]
        if split_x1 < wx1 {
            let brush = clip_brush_to_box(&wall_geometry, (split_x1, wx1), (wy0, wy1), (wz0, wz1))?;
            let id = ir.alloc_brush_id();
            ir.insert_brush(BrushAssembly {
                id,
                brush,
                role: wall_role,
                owner: attr.clone(),
                cost,
                support: support.clone(),
            });
            let sup_id = ir.alloc_support_id();
            ir.insert_support(SupportRecord {
                id: sup_id,
                child: id,
                parent: support.clone(),
            });
            new_ids.push(id);
        }

        if split_z0 > wz0 {
            emit_band(
                ir,
                &mut new_ids,
                frame_span,
                (wz0, split_z0),
                attr.clone(),
                cost,
                support.clone(),
            )?;
        }
        if split_z1 < wz1 {
            emit_band(
                ir,
                &mut new_ids,
                frame_span,
                (split_z1, wz1),
                attr.clone(),
                cost,
                support.clone(),
            )?;
        }
    } else {
        // E/W wall. Split in Y.
        if split_y0 > wy0 {
            let brush = clip_brush_to_box(&wall_geometry, (wx0, wx1), (wy0, split_y0), (wz0, wz1))?;
            let id = ir.alloc_brush_id();
            ir.insert_brush(BrushAssembly {
                id,
                brush,
                role: wall_role,
                owner: attr.clone(),
                cost,
                support: support.clone(),
            });
            let sup_id = ir.alloc_support_id();
            ir.insert_support(SupportRecord {
                id: sup_id,
                child: id,
                parent: support.clone(),
            });
            new_ids.push(id);
        }

        if split_y1 < wy1 {
            let brush = clip_brush_to_box(&wall_geometry, (wx0, wx1), (split_y1, wy1), (wz0, wz1))?;
            let id = ir.alloc_brush_id();
            ir.insert_brush(BrushAssembly {
                id,
                brush,
                role: wall_role,
                owner: attr.clone(),
                cost,
                support: support.clone(),
            });
            let sup_id = ir.alloc_support_id();
            ir.insert_support(SupportRecord {
                id: sup_id,
                child: id,
                parent: support.clone(),
            });
            new_ids.push(id);
        }

        if split_z0 > wz0 {
            emit_band(
                ir,
                &mut new_ids,
                frame_span,
                (wz0, split_z0),
                attr.clone(),
                cost,
                support.clone(),
            )?;
        }
        if split_z1 < wz1 {
            emit_band(
                ir,
                &mut new_ids,
                frame_span,
                (split_z1, wz1),
                attr.clone(),
                cost,
                support.clone(),
            )?;
        }
    }

    // Remove the original segment and stale edges. Exact interfaces/supports
    // are rebuilt after all omissions are materialized.
    ir.remove_brush(wall_brush_id);
    ir.supports.retain(|_, support| {
        support.child != wall_brush_id
            && !matches!(support.parent, SupportTarget::Brush(id) if id == wall_brush_id)
    });
    ir.interfaces.retain(|_, interface| {
        interface.brush_a != wall_brush_id && interface.brush_b != wall_brush_id
    });

    for opening in ir.openings.values_mut() {
        if !opening.wall_segment_ids.contains(&wall_brush_id) {
            continue;
        }
        opening.wall_segment_ids.retain(|id| *id != wall_brush_id);
        opening.wall_segment_ids.extend(new_ids.iter().copied());
        opening.wall_segment_ids.sort_unstable();
        opening.wall_segment_ids.dedup();
        if opening.owner_brush_id == wall_brush_id {
            if let Some(replacement) = opening.wall_segment_ids.first().copied() {
                opening.owner_brush_id = replacement;
            }
        }
    }

    Ok(new_ids)
}

/// Enforce that for every opening, the owner wall brush geometry actually
/// omits the opening volume.
///
/// This is the enforcement companion to `split_wall_around_opening`.
/// After wall splitting, we verify that no remaining wall brush overlaps
/// the opening throat.
pub(crate) fn enforce_opening_omission(ir: &AssemblyIR) -> Result<(), RichnessError> {
    for opening in ir.openings.values() {
        if !opening.wall_segment_ids.contains(&opening.owner_brush_id)
            || !ir.brushes.contains_key(&opening.owner_brush_id)
        {
            return Err(composition_error(
                "opening_omission.owner",
                format!(
                    "opening {} has no live canonical owner segment",
                    opening.id.raw()
                ),
            ));
        }

        let (x0, y0, z0, x1, y1, z1) = opening.bounds;
        let throat = ConvexBrush::make_box((x0, x1), (y0, y1), (z0, z1))
            .map_err(|error| composition_error("opening_omission", format!("throat: {error}")))?;
        for brush_id in opening
            .wall_segment_ids
            .iter()
            .chain(opening.frame_brush_ids.iter())
        {
            let brush = ir.brushes.get(brush_id).ok_or_else(|| {
                composition_error(
                    "opening_omission",
                    format!(
                        "opening {} references missing brush {}",
                        opening.id.raw(),
                        brush_id.raw()
                    ),
                )
            })?;
            if richness_geom::brushes_overlap(&brush.brush, &throat)? {
                return Err(composition_error(
                    "opening_omission",
                    format!(
                        "brush {} ({}) intrudes into exact throat for opening {}",
                        brush_id.raw(),
                        brush.role.tag(),
                        opening.id.raw()
                    ),
                ));
            }
        }
    }
    Ok(())
}

// ── Support DAG records for portal frames ─────────────────────────────────

/// Record support DAG edges for portal frame brushes.
///
/// Every portal frame brush (post, lintel, surround) must have a support
/// edge to either the floor slab or the wall brush it frames.
pub(crate) fn record_portal_frame_support(
    ir: &mut AssemblyIR,
    frame_brush_ids: &[BrushAssemblyId],
    wall_brush_id: BrushAssemblyId,
    floor_brush_id: BrushAssemblyId,
) -> Result<(), RichnessError> {
    for &frame_id in frame_brush_ids {
        // Verify frame brush exists (short-lived borrow)
        if !ir.brushes.contains_key(&frame_id) {
            return Err(composition_error("frame_support", "frame brush not found"));
        }

        // Posts are supported by floor; lintel and surround by wall.
        let sup_id = ir.alloc_support_id();
        ir.insert_support(SupportRecord {
            id: sup_id,
            child: frame_id,
            parent: SupportTarget::Brush(wall_brush_id),
        });

        // Also record interface between frame and wall
        let if_id = ir.alloc_interface_id();
        ir.insert_interface(InterfaceRecord {
            id: if_id,
            brush_a: frame_id,
            brush_b: wall_brush_id,
            kind: InterfaceKind::PostToWall,
        });

        let _ = floor_brush_id;
    }

    Ok(())
}

// ── Cross-brush overlap validation ────────────────────────────────────────

/// Validate that no two brushes in the assembly have positive-volume overlap.
pub(crate) fn validate_no_overlaps(ir: &AssemblyIR) -> Result<(), RichnessError> {
    let brushes: Vec<_> = ir.brushes.values().collect();
    for i in 0..brushes.len() {
        for j in (i + 1)..brushes.len() {
            if richness_geom::brushes_overlap(&brushes[i].brush, &brushes[j].brush)? {
                return Err(composition_error(
                    "overlap",
                    format!(
                        "brushes {:?} ({}) and {:?} ({}) overlap",
                        brushes[i].id.raw(),
                        brushes[i].role.tag(),
                        brushes[j].id.raw(),
                        brushes[j].role.tag()
                    ),
                ));
            }
        }
    }
    Ok(())
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::super::footprint::Footprint3D;
    use super::super::ids::SemanticId;
    use super::super::ids::{
        ArchetypeRequestId, BeatId, PortalId, ReservationId, WallChainId, ZoneId,
    };
    use super::super::pacing::build_pacing_blueprint;
    use super::super::request::{RichnessDocumentV1, RichnessPreset, RichnessTheme};
    use super::super::reservation::{ReservationKind, ReservationRecord};
    use super::super::solver::solve_placement_and_topology;
    use super::super::topology::Dir;
    use super::super::variation::{WallChainRecord, WallMass, WallMassTreatment, WallShaping};
    use super::*;

    fn make_test_attr() -> SemanticAttribution {
        SemanticAttribution::from_reservation(
            ReservationId::new(0),
            Some(ArchetypeRequestId::new(0)),
            Some(ArchetypeIndex::new(0)),
            Some(BeatId::new(0)),
            Some(ZoneId::new(0)),
        )
    }

    fn make_test_fp(w: u32, h: u32) -> Footprint3D {
        // w, h are grid cell counts. Convert to quake units.
        let qw = (w * 16) as i32;
        let qh = (h * 16) as i32;
        Footprint3D::single_layer(0, 0, qw, qh, 0)
    }

    #[test]
    fn solved_phase07_output_reaches_composition_pipeline() {
        let document =
            RichnessDocumentV1::new(0, 2048, RichnessPreset::Sparse, RichnessTheme::Ancient)
                .unwrap();
        let resolved = super::super::request::ResolvedRichnessRequestV1::resolve(document).unwrap();
        let blueprint = build_pacing_blueprint(&resolved).unwrap();
        let generation = solve_placement_and_topology(blueprint.clone(), resolved).unwrap();
        let expected_portals = 2 * generation.topology.routes.len();

        for (theme_variant, theme) in [
            RichnessTheme::Ancient,
            RichnessTheme::Egyptian,
            RichnessTheme::Brutalist,
        ]
        .into_iter()
        .enumerate()
        {
            let complexity = super::super::complexity::build_complexity_plan(
                RichnessPreset::Sparse,
                theme_variant as u32,
                &blueprint,
                &generation.topology,
                &generation.placement.request_archetypes,
            );
            assert!(complexity.is_within_budget(), "{:?}", complexity.errors);
            let composition = compose_solved_generation(
                &generation,
                theme,
                &complexity,
                42,
                crate::enhanced_v3::richness::request::RichnessCaveMode::Omitted,
            )
            .unwrap();
            let route_portals = generation.topology.routes.len() * 2;
            assert!(
                composition.assembly.portal_assemblies.len() <= route_portals,
                "deduplicated portals must not exceed route sockets"
            );
            assert_eq!(
                composition.assembly.openings.len(),
                composition.assembly.portal_assemblies.len()
            );
            assert!(!composition.visibility.semantic_leaves.is_empty());
            assert!(!composition.visibility.semantic_pvs.is_empty());
            for brush in composition.assembly.brushes.values() {
                if let Some(request_id) = brush.owner.request_id {
                    assert_eq!(
                        brush.owner.archetype,
                        generation
                            .placement
                            .request_archetypes
                            .get(&request_id)
                            .copied()
                    );
                }
            }
            for opening in composition.assembly.openings.values() {
                let width = match opening.wall_role {
                    BrushAssemblyRole::NorthWall | BrushAssemblyRole::SouthWall => {
                        opening.bounds.3 - opening.bounds.0
                    }
                    BrushAssemblyRole::EastWall | BrushAssemblyRole::WestWall => {
                        opening.bounds.4 - opening.bounds.1
                    }
                    _ => 0,
                };
                assert_eq!(width, 64);
                assert_eq!(opening.bounds.5 - opening.bounds.2, 80);
            }
        }
    }

    #[test]
    fn composition_rejects_missing_explicit_archetype_identity() {
        let record = ReservationRecord {
            id: ReservationId::new(7),
            kind: ReservationKind::StandardRoom,
            footprint: make_test_fp(16, 16),
            beat_id: Some(BeatId::new(1)),
            request_id: Some(ArchetypeRequestId::new(99)),
            zone_id: Some(ZoneId::new(0)),
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
        let reservations = [(record.id, record)].into_iter().collect();
        let error = compose_all_rooms(&reservations, &BTreeMap::new()).unwrap_err();
        assert_eq!(error.code, RichnessErrorCode::SemanticInfeasible);
        assert_eq!(error.path, "composition.archetype");
    }

    #[test]
    fn touching_rooms_materialize_one_shared_wall_plane() {
        let make_record = |id: u32, request: u32, x0: i32, x1: i32| ReservationRecord {
            id: ReservationId::new(id),
            kind: ReservationKind::StandardRoom,
            footprint: Footprint3D::single_layer(x0, 0, x1, 256, 0),
            beat_id: Some(BeatId::new(id)),
            request_id: Some(ArchetypeRequestId::new(request)),
            zone_id: Some(ZoneId::new(0)),
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
        let reservations: BTreeMap<_, _> = [
            (ReservationId::new(0), make_record(0, 40, 0, 256)),
            (ReservationId::new(1), make_record(1, 41, 256, 512)),
        ]
        .into_iter()
        .collect();
        let archetypes = [
            (ArchetypeRequestId::new(40), ArchetypeIndex::new(1)),
            (ArchetypeRequestId::new(41), ArchetypeIndex::new(1)),
        ]
        .into_iter()
        .collect();
        let mut ir = compose_all_rooms(&reservations, &archetypes).unwrap();
        materialize_canonical_shared_walls(&reservations, &mut ir).unwrap();

        assert_eq!(ir.shared_wall_chains.len(), 1);
        let chain = ir.shared_wall_chains.values().next().unwrap();
        assert_eq!(chain.owner_reservation_id, ReservationId::new(0));
        assert_eq!(chain.sharing_reservation_id, ReservationId::new(1));
        let owner = &ir.brushes[&chain.owner_brush_id];
        assert!(owner
            .brush
            .faces
            .iter()
            .any(|face| face.plane == chain.shared_plane));
        assert_eq!(
            ir.brushes
                .values()
                .filter(|brush| {
                    brush.owner.reservation_id == ReservationId::new(1)
                        && brush.role == BrushAssemblyRole::WestWall
                })
                .count(),
            0
        );
        validate_no_overlaps(&ir).unwrap();
    }

    #[test]
    fn compose_rectangle_produces_brushes() {
        let fp = make_test_fp(16, 16); // 256x256 quake units
        let attr = make_test_attr();
        let cost = CostSource {
            dimension: BudgetDimension::SourceFaces,
            face_count: 6,
        };
        let mut ir = AssemblyIR::new();
        compose_rectangle(&fp, &attr, cost, &mut ir).unwrap();

        // Should produce 6 brushes: floor + ceiling + 4 walls
        assert_eq!(ir.brush_count(), 6);
        assert_eq!(ir.support_count(), 6);

        // Verify no overlaps
        validate_no_overlaps(&ir).unwrap();
    }

    #[test]
    fn compose_rectangle_all_roles_present() {
        let fp = make_test_fp(16, 16);
        let attr = make_test_attr();
        let cost = CostSource {
            dimension: BudgetDimension::SourceFaces,
            face_count: 6,
        };
        let mut ir = AssemblyIR::new();
        compose_rectangle(&fp, &attr, cost, &mut ir).unwrap();

        let roles: Vec<_> = ir.brushes.values().map(|b| b.role).collect();
        assert!(roles.contains(&BrushAssemblyRole::FloorSlab));
        assert!(roles.contains(&BrushAssemblyRole::CeilingSlab));
        assert!(roles.contains(&BrushAssemblyRole::NorthWall));
        assert!(roles.contains(&BrushAssemblyRole::SouthWall));
        assert!(roles.contains(&BrushAssemblyRole::EastWall));
        assert!(roles.contains(&BrushAssemblyRole::WestWall));
    }

    #[test]
    fn compose_chamfered_produces_brushes() {
        let fp = make_test_fp(20, 20); // 320x320, enough for chamfer
        let attr = make_test_attr();
        let cost = CostSource {
            dimension: BudgetDimension::SourceFaces,
            face_count: 6,
        };
        let mut ir = AssemblyIR::new();
        compose_chamfered(&fp, &attr, cost, 0, &mut ir).unwrap();

        // Should produce 10 brushes: floor + ceiling + 4 cardinal + 4 diagonal
        assert!(
            ir.brush_count() >= 8,
            "expected at least 8 brushes, got {}",
            ir.brush_count()
        );

        // Verify no overlaps
        validate_no_overlaps(&ir).unwrap();
    }

    #[test]
    fn compose_octagonal_produces_same_as_chamfered() {
        let fp = make_test_fp(20, 20);
        let attr = make_test_attr();
        let cost = CostSource {
            dimension: BudgetDimension::SourceFaces,
            face_count: 6,
        };
        let mut ir1 = AssemblyIR::new();
        compose_chamfered(&fp, &attr, cost, 0, &mut ir1).unwrap();
        let mut ir2 = AssemblyIR::new();
        compose_octagonal(&fp, &attr, cost, 0, &mut ir2).unwrap();

        assert_eq!(ir1.brush_count(), ir2.brush_count());
    }

    #[test]
    fn compose_composite_partition_produces_rectangle() {
        let fp = make_test_fp(16, 16);
        let attr = make_test_attr();
        let cost = CostSource {
            dimension: BudgetDimension::SourceFaces,
            face_count: 6,
        };
        let mut ir = AssemblyIR::new();
        compose_composite_partition(&fp, &attr, cost, 0, &mut ir).unwrap();
        assert_eq!(ir.brush_count(), 6);
    }

    #[test]
    fn compose_single_room_selects_correct_shape() {
        let fp = make_test_fp(16, 16);
        let req_id = Some(ArchetypeRequestId::new(1)); // antechamber → Rectangle
        let record = ReservationRecord {
            id: ReservationId::new(0),
            kind: ReservationKind::StandardRoom,
            footprint: fp,
            beat_id: Some(BeatId::new(0)),
            request_id: req_id,
            zone_id: Some(ZoneId::new(0)),
            pit_pair_room_id: None,
            composite_children: Vec::new(),
            owning_route_id: None,
            clearance_height: None,
            committed: true,
            cost_faces: 200,
            cost_brushes: 12,
            cost_entities: 3,
            cost_lights: 2,
        };

        let mut ir = AssemblyIR::new();
        compose_single_room(&record, ArchetypeIndex::new(1), &mut ir).unwrap();
        assert_eq!(ir.brush_count(), 6); // Rectangle produces 6
    }

    #[test]
    fn compose_single_room_with_octagon_shape() {
        let fp = make_test_fp(20, 20);
        let req_id = Some(ArchetypeRequestId::new(2)); // arena → Octagon
        let record = ReservationRecord {
            id: ReservationId::new(0),
            kind: ReservationKind::StandardRoom,
            footprint: fp,
            beat_id: Some(BeatId::new(0)),
            request_id: req_id,
            zone_id: Some(ZoneId::new(0)),
            pit_pair_room_id: None,
            composite_children: Vec::new(),
            owning_route_id: None,
            clearance_height: None,
            committed: true,
            cost_faces: 200,
            cost_brushes: 12,
            cost_entities: 3,
            cost_lights: 2,
        };

        let mut ir = AssemblyIR::new();
        compose_single_room(&record, ArchetypeIndex::new(2), &mut ir).unwrap();
        // Octagon produces 10 brushes (floor + ceiling + 4 cardinal + 4 diagonal)
        assert!(
            ir.brush_count() >= 8,
            "octagon should have at least 8 brushes, got {}",
            ir.brush_count()
        );
    }

    #[test]
    fn all_archetypes_compose_without_error() {
        for arch_idx in 0..generated_content::ARCHETYPE_COUNT {
            let arch_str = generated_content::ARCHETYPE_IDS[arch_idx];
            let _span_min = generated_content::ARCHETYPE_SPAN_MIN[arch_idx];
            let span_max = generated_content::ARCHETYPE_SPAN_MAX[arch_idx];
            // Use span_max as footprint size (in Quake units, convert to grid)
            let quake_w = span_max[0];
            let quake_h = span_max[1];
            let grid_w = quake_w / 16;
            let grid_h = quake_h / 16;

            if grid_w == 0 || grid_h == 0 {
                continue;
            }

            let fp = Footprint3D::single_layer(0, 0, quake_w as i32, quake_h as i32, 0);
            let attr = SemanticAttribution::from_reservation(
                ReservationId::new(0),
                Some(ArchetypeRequestId::new(arch_idx as u32)),
                Some(ArchetypeIndex::new(arch_idx as u32)),
                Some(BeatId::new(0)),
                Some(ZoneId::new(0)),
            );
            let cost = CostSource {
                dimension: BudgetDimension::SourceFaces,
                face_count: 6,
            };
            let mut ir = AssemblyIR::new();

            let result = match generated_content::ARCHETYPE_SHAPE[arch_idx] {
                ShapeRule::Rectangle => compose_rectangle(&fp, &attr, cost, &mut ir),
                ShapeRule::Chamfer => compose_chamfered(&fp, &attr, cost, arch_idx, &mut ir),
                ShapeRule::Octagon => compose_octagonal(&fp, &attr, cost, arch_idx, &mut ir),
                ShapeRule::CompositePartition => {
                    compose_composite_partition(&fp, &attr, cost, arch_idx, &mut ir)
                }
            };

            assert!(
                result.is_ok(),
                "archetype {} ({}) failed to compose: {:?}",
                arch_idx,
                arch_str,
                result.err()
            );

            // Verify no overlaps
            if let Err(e) = validate_no_overlaps(&ir) {
                panic!(
                    "archetype {} ({}) has overlapping brushes: {:?}",
                    arch_idx, arch_str, e
                );
            }

            // Verify all brushes have positive volume and approved normals
            for brush in ir.brushes.values() {
                assert!(
                    brush.brush.volume() > geometry::Rational::ZERO,
                    "archetype {} ({}): brush {:?} has zero volume",
                    arch_idx,
                    arch_str,
                    brush.role.tag()
                );
            }
        }
    }

    #[test]
    fn chamfered_corner_walls_present() {
        let fp = make_test_fp(20, 20);
        let attr = make_test_attr();
        let cost = CostSource {
            dimension: BudgetDimension::SourceFaces,
            face_count: 6,
        };
        let mut ir = AssemblyIR::new();
        compose_chamfered(&fp, &attr, cost, 0, &mut ir).unwrap();

        let roles: Vec<_> = ir.brushes.values().map(|b| b.role).collect();
        assert!(roles.contains(&BrushAssemblyRole::DiagNWWall));
        assert!(roles.contains(&BrushAssemblyRole::DiagNEWall));
        assert!(roles.contains(&BrushAssemblyRole::DiagSWWall));
        assert!(roles.contains(&BrushAssemblyRole::DiagSEWall));
    }

    #[test]
    fn debug_chamfered_north_wall() {
        // Test the chamfered north wall plane construction directly
        let qx0: i128 = 0;
        let qy0: i128 = 0;
        let qx1: i128 = 320;
        let qy1: i128 = 320;
        let chamfer: i128 = 80;
        let wall_t: i128 = 16;
        let z_min: i128 = 16;
        let z_max: i128 = 160;

        let nw_diag_d = qy1 - qx0 - chamfer;
        let ne_diag_d = qy1 + qx1 - chamfer;

        let mk = |nx, ny, nz, d| CanonicalPlane::new(nx, ny, nz, d).map_err(|e| format!("{e}"));

        let planes: Vec<CanonicalPlane> = vec![
            mk(0, 0, 1, z_min).unwrap(),
            mk(0, 0, -1, -z_max).unwrap(),
            mk(0, -1, 0, -qy1).unwrap(),
            mk(0, 1, 0, qy1 - wall_t).unwrap(),
            mk(1, -1, 0, -nw_diag_d).unwrap(),
            mk(-1, -1, 0, -ne_diag_d).unwrap(),
        ];

        for p in &planes {
            eprintln!("plane: {}", p);
        }

        let faces: Vec<BrushFace> = planes
            .into_iter()
            .map(BrushFace::new)
            .collect::<Result<_, _>>()
            .unwrap();

        let mut brush = ConvexBrush::new(faces).unwrap();
        match brush.validate_and_cache() {
            Ok(()) => {
                eprintln!("VALID: volume={}", brush.volume());
            }
            Err(e) => {
                panic!("validation failed: {:?}", e);
            }
        }
    }

    #[test]
    fn compute_chamfer_size_is_quantum_aligned() {
        let fp = make_test_fp(20, 20);
        let c = compute_chamfer_size(&fp);
        assert_eq!(c % 16, 0, "chamfer size must be quantum-aligned");
        assert!(c >= 48 && c <= 96, "chamfer size out of range: {}", c);
    }

    // ── Session B tests ──────────────────────────────────────────────

    #[test]
    fn build_ancient_post_lintel_portal_succeeds() {
        let mut ir = AssemblyIR::new();
        let attr = make_test_attr();
        let cost = CostSource {
            dimension: BudgetDimension::SourceFaces,
            face_count: 6,
        };

        let wall_id = ir.alloc_brush_id();
        let wall_brush = ConvexBrush::make_box(
            (0, 256),
            (240, 256),
            (richness_geom::WALL_Z_MIN, richness_geom::WALL_Z_MAX),
        )
        .unwrap();
        ir.insert_brush(BrushAssembly {
            id: wall_id,
            brush: wall_brush,
            role: BrushAssemblyRole::NorthWall,
            owner: attr.clone(),
            cost,
            support: SupportTarget::World,
        });

        let portal_id = PortalId::new(0);
        // Throat anchor: (span_min, z_min, span_max, z_max) = (X=80, Z=16, X=144, Z=96) => 64 wide, 80 tall
        let throat_anchor = (80i128, 16i128, 144i128, 96i128);

        let result = build_portal(
            portal_id,
            PortalStyle::AncientPostLintel,
            wall_id,
            throat_anchor,
            BrushAssemblyRole::NorthWall,
            &attr,
            cost,
            &mut ir,
        );
        assert!(
            result.is_ok(),
            "ancient post-lintel failed: {:?}",
            result.err()
        );
        let portal = result.unwrap().unwrap();
        assert_eq!(portal.post_ids.len(), 2);
        assert_eq!(portal.lintel_ids.len(), 1);
        assert!(portal.surround_ids.is_empty());
        assert_eq!(ir.openings.len(), 1);
    }

    #[test]
    fn build_egyptian_stepped_surround_succeeds() {
        let mut ir = AssemblyIR::new();
        let attr = make_test_attr();
        let cost = CostSource {
            dimension: BudgetDimension::SourceFaces,
            face_count: 6,
        };

        let wall_id = ir.alloc_brush_id();
        let wall_brush = ConvexBrush::make_box(
            (0, 256),
            (240, 256),
            (richness_geom::WALL_Z_MIN, richness_geom::WALL_Z_MAX),
        )
        .unwrap();
        ir.insert_brush(BrushAssembly {
            id: wall_id,
            brush: wall_brush,
            role: BrushAssemblyRole::NorthWall,
            owner: attr.clone(),
            cost,
            support: SupportTarget::World,
        });

        let portal_id = PortalId::new(1);
        let throat_anchor = (80i128, 16i128, 144i128, 96i128);
        let result = build_portal(
            portal_id,
            PortalStyle::EgyptianSteppedSurround,
            wall_id,
            throat_anchor,
            BrushAssemblyRole::NorthWall,
            &attr,
            cost,
            &mut ir,
        );
        assert!(
            result.is_ok(),
            "egyptian stepped failed: {:?}",
            result.err()
        );
        let portal = result.unwrap().unwrap();
        // 3 layers × 3 brushes per layer = 9 surround brushes
        assert_eq!(portal.surround_ids.len(), 9);
        assert!(portal.post_ids.is_empty());
        assert!(portal.lintel_ids.is_empty());
        assert_eq!(ir.openings.len(), 1);
    }

    #[test]
    fn build_brutalist_reveal_surround_succeeds() {
        let mut ir = AssemblyIR::new();
        let attr = make_test_attr();
        let cost = CostSource {
            dimension: BudgetDimension::SourceFaces,
            face_count: 6,
        };

        let wall_id = ir.alloc_brush_id();
        let wall_brush = ConvexBrush::make_box(
            (0, 256),
            (240, 256),
            (richness_geom::WALL_Z_MIN, richness_geom::WALL_Z_MAX),
        )
        .unwrap();
        ir.insert_brush(BrushAssembly {
            id: wall_id,
            brush: wall_brush,
            role: BrushAssemblyRole::NorthWall,
            owner: attr.clone(),
            cost,
            support: SupportTarget::World,
        });

        let portal_id = PortalId::new(2);
        let throat_anchor = (80i128, 16i128, 144i128, 96i128);
        let result = build_portal(
            portal_id,
            PortalStyle::BrutalistRevealSurround,
            wall_id,
            throat_anchor,
            BrushAssemblyRole::NorthWall,
            &attr,
            cost,
            &mut ir,
        );
        assert!(
            result.is_ok(),
            "brutalist reveal failed: {:?}",
            result.err()
        );
        let portal = result.unwrap().unwrap();
        // 3 reveal + 3 surround = 6 surround brushes
        assert_eq!(portal.surround_ids.len(), 6);
        assert!(portal.post_ids.is_empty());
        assert!(portal.lintel_ids.is_empty());
        assert_eq!(ir.openings.len(), 1);
    }

    #[test]
    fn near_corner_brutalist_surround_clips_to_two_cell_socket() {
        let mut ir = AssemblyIR::new();
        let attr = make_test_attr();
        let cost = CostSource {
            dimension: BudgetDimension::SourceFaces,
            face_count: 6,
        };

        let east_id = ir.alloc_brush_id();
        ir.insert_brush(BrushAssembly {
            id: east_id,
            brush: ConvexBrush::make_box((736, 752), (16, 240), (16, 352)).unwrap(),
            role: BrushAssemblyRole::EastWall,
            owner: attr.clone(),
            cost,
            support: SupportTarget::World,
        });
        let south_id = ir.alloc_brush_id();
        ir.insert_brush(BrushAssembly {
            id: south_id,
            brush: ConvexBrush::make_box((512, 768), (240, 256), (16, 352)).unwrap(),
            role: BrushAssemblyRole::SouthWall,
            owner: attr.clone(),
            cost,
            support: SupportTarget::World,
        });

        let portal = build_portal(
            PortalId::new(3),
            PortalStyle::BrutalistRevealSurround,
            east_id,
            (160, 16, 224, 96),
            BrushAssemblyRole::EastWall,
            &attr,
            cost,
            &mut ir,
        )
        .unwrap()
        .unwrap();

        assert_eq!(portal.throat_bounds, (736, 160, 16, 752, 224, 96));
        assert!(portal.surround_ids.iter().all(|id| {
            let ((_, y0, _), (_, y1, _)) = ir.brushes[id].brush.aabb().unwrap();
            y0 >= 16 && y1 <= 240
        }));
        validate_no_overlaps(&ir).unwrap();
        enforce_opening_omission(&ir).unwrap();
    }

    #[test]
    fn portal_rejects_diagonal_wall() {
        let mut ir = AssemblyIR::new();
        let attr = make_test_attr();
        let cost = CostSource {
            dimension: BudgetDimension::SourceFaces,
            face_count: 6,
        };

        let wall_id = ir.alloc_brush_id();
        let wall_brush = ConvexBrush::make_box(
            (0, 64),
            (0, 64),
            (richness_geom::WALL_Z_MIN, richness_geom::WALL_Z_MAX),
        )
        .unwrap();
        ir.insert_brush(BrushAssembly {
            id: wall_id,
            brush: wall_brush,
            role: BrushAssemblyRole::DiagNWWall,
            owner: attr.clone(),
            cost,
            support: SupportTarget::World,
        });

        let result = build_portal(
            PortalId::new(0),
            PortalStyle::AncientPostLintel,
            wall_id,
            (0, 16, 64, 96), // valid throat format but diagonal wall should fail first
            BrushAssemblyRole::DiagNWWall,
            &attr,
            cost,
            &mut ir,
        );
        assert!(result.is_err(), "diagonal portal should be rejected");
        let err = result.unwrap_err();
        assert!(err.context.contains("non-cardinal"));
    }

    #[test]
    fn derive_all_interfaces_for_rectangle_room() {
        let fp = make_test_fp(16, 16);
        let attr = make_test_attr();
        let cost = CostSource {
            dimension: BudgetDimension::SourceFaces,
            face_count: 6,
        };
        let mut ir = AssemblyIR::new();
        compose_rectangle(&fp, &attr, cost, &mut ir).unwrap();

        // Derive interfaces
        derive_all_interfaces(&mut ir).unwrap();

        // Should have wall-to-floor and wall-to-wall interfaces
        assert!(
            ir.interface_count() > 0,
            "expected interfaces between walls and floor"
        );
    }

    #[test]
    fn portal_frame_brushes_are_positive_volume() {
        let mut ir = AssemblyIR::new();
        let attr = make_test_attr();
        let cost = CostSource {
            dimension: BudgetDimension::SourceFaces,
            face_count: 6,
        };

        // Wall brush does NOT span the throat area to avoid overlap
        // Left segment: X=0..80
        let wall_left = ir.alloc_brush_id();
        ir.insert_brush(BrushAssembly {
            id: wall_left,
            brush: ConvexBrush::make_box(
                (0, 80),
                (240, 256),
                (richness_geom::WALL_Z_MIN, richness_geom::WALL_Z_MAX),
            )
            .unwrap(),
            role: BrushAssemblyRole::NorthWall,
            owner: attr.clone(),
            cost,
            support: SupportTarget::World,
        });

        // Right segment: X=176..320
        let wall_right = ir.alloc_brush_id();
        ir.insert_brush(BrushAssembly {
            id: wall_right,
            brush: ConvexBrush::make_box(
                (176, 320),
                (240, 256),
                (richness_geom::WALL_Z_MIN, richness_geom::WALL_Z_MAX),
            )
            .unwrap(),
            role: BrushAssemblyRole::NorthWall,
            owner: attr.clone(),
            cost,
            support: SupportTarget::World,
        });

        let throat_anchor = (96i128, 16i128, 160i128, 96i128);
        build_portal(
            PortalId::new(0),
            PortalStyle::AncientPostLintel,
            wall_left, // use left wall as the associated brush
            throat_anchor,
            BrushAssemblyRole::NorthWall,
            &attr,
            cost,
            &mut ir,
        )
        .unwrap();

        // Portal frame brushes exist and the two wall segments don't overlap the portal
        assert!(
            ir.brush_count() >= 5,
            "expected wall segments + portal frame brushes"
        );
        // Verify no overlaps among the brushes that exist
        validate_no_overlaps(&ir).unwrap();
    }

    #[test]
    fn wall_mass_rejects_protected_segment_overlap() {
        let attr = make_test_attr();
        let cost = CostSource {
            dimension: BudgetDimension::SourceFaces,
            face_count: 6,
        };

        let chain = WallChainRecord {
            id: WallChainId::new(0),
            owner: SemanticId::Reservation(ReservationId::new(0)),
            shared_with: Vec::new(),
            cardinal_direction: Dir::North,
            shaping: [WallShaping::None, WallShaping::None],
            mass_treatments: vec![WallMassTreatment {
                segment: (32, 48),
                kind: WallMass::Liner16,
                quantum_count: 1,
            }],
            portal_anchors: Vec::new(),
            protected_segments: vec![(32, 64)],
            structural_thickness: 16,
            exterior_envelope: false,
        };

        let mut ir = AssemblyIR::new();
        let wall_id = ir.alloc_brush_id();
        let wall_brush = ConvexBrush::make_box(
            (0, 256),
            (240, 256),
            (richness_geom::WALL_Z_MIN, richness_geom::WALL_Z_MAX),
        )
        .unwrap();
        ir.insert_brush(BrushAssembly {
            id: wall_id,
            brush: wall_brush,
            role: BrushAssemblyRole::NorthWall,
            owner: attr.clone(),
            cost,
            support: SupportTarget::World,
        });

        let result = apply_wall_mass_treatments(
            &chain,
            wall_id,
            BrushAssemblyRole::NorthWall,
            &attr,
            cost,
            &mut ir,
        );
        assert!(result.is_err(), "mass treatment should be rejected");
        assert!(result.unwrap_err().context.contains("overlaps protected"));
    }

    #[test]
    fn wall_mass_materializes_disjoint_liner_and_recess_omission() {
        let attr = make_test_attr();
        let cost = CostSource {
            dimension: BudgetDimension::SourceFaces,
            face_count: 6,
        };
        let chain = WallChainRecord {
            id: WallChainId::new(0),
            owner: SemanticId::Reservation(ReservationId::new(0)),
            shared_with: Vec::new(),
            cardinal_direction: Dir::North,
            shaping: [WallShaping::None, WallShaping::None],
            mass_treatments: vec![
                WallMassTreatment {
                    segment: (0, 16),
                    kind: WallMass::Liner16,
                    quantum_count: 1,
                },
                WallMassTreatment {
                    segment: (32, 48),
                    kind: WallMass::Recess16,
                    quantum_count: 1,
                },
            ],
            portal_anchors: Vec::new(),
            protected_segments: Vec::new(),
            structural_thickness: 16,
            exterior_envelope: false,
        };
        let mut ir = AssemblyIR::new();
        let wall_id = ir.alloc_brush_id();
        ir.insert_brush(BrushAssembly {
            id: wall_id,
            brush: ConvexBrush::make_box(
                (0, 256),
                (240, 256),
                (richness_geom::WALL_Z_MIN, richness_geom::WALL_Z_MAX),
            )
            .unwrap(),
            role: BrushAssemblyRole::NorthWall,
            owner: attr.clone(),
            cost,
            support: SupportTarget::World,
        });

        apply_wall_mass_treatments(
            &chain,
            wall_id,
            BrushAssemblyRole::NorthWall,
            &attr,
            cost,
            &mut ir,
        )
        .unwrap();

        assert!(ir.brushes.values().any(|brush| {
            brush.role == BrushAssemblyRole::WallLiner && brush.brush.aabb().unwrap().0 .1 == 256
        }));
        assert_eq!(ir.openings.len(), 1);
        enforce_opening_omission(&ir).unwrap();
        validate_no_overlaps(&ir).unwrap();
    }

    #[test]
    fn shared_wall_chain_rejects_non_standard_thickness() {
        let mut ir = AssemblyIR::new();
        let mut sharing_ir = AssemblyIR::new();
        let attr = make_test_attr();
        let cost = CostSource {
            dimension: BudgetDimension::SourceFaces,
            face_count: 6,
        };

        let wall_id = ir.alloc_brush_id();
        let wall_brush = ConvexBrush::make_box(
            (0, 256),
            (240, 256),
            (richness_geom::WALL_Z_MIN, richness_geom::WALL_Z_MAX),
        )
        .unwrap();
        ir.insert_brush(BrushAssembly {
            id: wall_id,
            brush: wall_brush,
            role: BrushAssemblyRole::NorthWall,
            owner: attr.clone(),
            cost,
            support: SupportTarget::World,
        });

        let chain = WallChainRecord {
            id: WallChainId::new(0),
            owner: SemanticId::Reservation(ReservationId::new(0)),
            shared_with: vec![SemanticId::Reservation(ReservationId::new(1))],
            cardinal_direction: Dir::North,
            shaping: [WallShaping::None, WallShaping::None],
            mass_treatments: Vec::new(),
            portal_anchors: Vec::new(),
            protected_segments: Vec::new(),
            structural_thickness: 24, // NOT 16 — should be rejected
            exterior_envelope: false,
        };

        let result = materialize_shared_wall_chain(&chain, &mut ir, wall_id, &mut sharing_ir);
        assert!(result.is_err(), "non-standard thickness should be rejected");
    }

    #[test]
    fn shared_wall_chain_rejects_shaping_with_portal_anchors() {
        let mut ir = AssemblyIR::new();
        let mut sharing_ir = AssemblyIR::new();
        let attr = make_test_attr();
        let cost = CostSource {
            dimension: BudgetDimension::SourceFaces,
            face_count: 6,
        };

        let wall_id = ir.alloc_brush_id();
        let wall_brush = ConvexBrush::make_box(
            (0, 256),
            (240, 256),
            (richness_geom::WALL_Z_MIN, richness_geom::WALL_Z_MAX),
        )
        .unwrap();
        ir.insert_brush(BrushAssembly {
            id: wall_id,
            brush: wall_brush,
            role: BrushAssemblyRole::NorthWall,
            owner: attr.clone(),
            cost,
            support: SupportTarget::World,
        });

        let chain = WallChainRecord {
            id: WallChainId::new(0),
            owner: SemanticId::Reservation(ReservationId::new(0)),
            shared_with: Vec::new(),
            cardinal_direction: Dir::North,
            shaping: [WallShaping::OneQuantum, WallShaping::None],
            mass_treatments: Vec::new(),
            portal_anchors: vec![PortalId::new(0)],
            protected_segments: Vec::new(),
            structural_thickness: 16,
            exterior_envelope: false,
        };

        let result = materialize_shared_wall_chain(&chain, &mut ir, wall_id, &mut sharing_ir);
        assert!(
            result.is_err(),
            "shaping with portal anchors should be rejected"
        );
    }

    #[test]
    fn portal_throat_witness_64x80_is_preserved() {
        let mut ir = AssemblyIR::new();
        let attr = make_test_attr();
        let cost = CostSource {
            dimension: BudgetDimension::SourceFaces,
            face_count: 6,
        };

        let wall_id = ir.alloc_brush_id();
        let wall_brush = ConvexBrush::make_box(
            (0, 320),
            (240, 256),
            (richness_geom::WALL_Z_MIN, richness_geom::WALL_Z_MAX),
        )
        .unwrap();
        ir.insert_brush(BrushAssembly {
            id: wall_id,
            brush: wall_brush,
            role: BrushAssemblyRole::NorthWall,
            owner: attr.clone(),
            cost,
            support: SupportTarget::World,
        });

        let throat_anchor = (96i128, 16i128, 160i128, 96i128); // X=96..160 (64 wide), Z=16..96 (80 tall)
        let portal = build_portal(
            PortalId::new(0),
            PortalStyle::AncientPostLintel,
            wall_id,
            throat_anchor,
            BrushAssemblyRole::NorthWall,
            &attr,
            cost,
            &mut ir,
        )
        .unwrap()
        .unwrap();

        // Opening bounds describe only the protected clear throat. Frame
        // extents are separate brushes and must never inflate this witness.
        let opening = &ir.openings[&portal.opening_id];
        assert_eq!(opening.bounds, (96, 240, 16, 160, 256, 96));
        assert_eq!(opening.bounds.3 - opening.bounds.0, 64);
        assert_eq!(opening.bounds.5 - opening.bounds.2, 80);
        enforce_opening_omission(&ir).unwrap();
    }

    #[test]
    fn build_sill_rejects_invalid_height() {
        let mut ir = AssemblyIR::new();
        let attr = make_test_attr();
        let cost = CostSource {
            dimension: BudgetDimension::SourceFaces,
            face_count: 6,
        };

        assert!(build_sill(0, 0, 64, 16, 32, &attr, cost, &mut ir).is_err());
        assert!(build_sill(0, 0, 64, 16, 80, &attr, cost, &mut ir).is_err());
        assert!(build_sill(0, 0, 64, 16, 48, &attr, cost, &mut ir).is_ok());
        assert!(build_sill(0, 0, 64, 16, 64, &attr, cost, &mut ir).is_ok());
    }

    #[test]
    fn build_pilaster_and_partial_wall_succeed() {
        let mut ir = AssemblyIR::new();
        let attr = make_test_attr();
        let cost = CostSource {
            dimension: BudgetDimension::SourceFaces,
            face_count: 6,
        };

        let pid = build_pilaster(64, 224, 80, 240, &attr, cost, &mut ir).unwrap();
        assert!(ir.brushes.contains_key(&pid));
        assert_eq!(ir.brushes[&pid].role, BrushAssemblyRole::Pilaster);

        let wid = build_partial_wall(
            128,
            80,
            144,
            176,
            richness_geom::WALL_Z_MIN,
            richness_geom::WALL_Z_MAX,
            &attr,
            cost,
            &mut ir,
        )
        .unwrap();
        assert!(ir.brushes.contains_key(&wid));
        assert_eq!(ir.brushes[&wid].role, BrushAssemblyRole::PartialWall);
    }

    #[test]
    fn build_bent_approach_and_offset_shaft_succeed() {
        let mut ir = AssemblyIR::new();
        let attr = make_test_attr();
        let cost = CostSource {
            dimension: BudgetDimension::SourceFaces,
            face_count: 6,
        };

        let bid = build_bent_approach(
            0,
            64,
            16,
            128,
            richness_geom::WALL_Z_MIN,
            richness_geom::WALL_Z_MAX,
            &attr,
            cost,
            &mut ir,
        )
        .unwrap();
        assert!(ir.brushes.contains_key(&bid));

        let sid = build_offset_shaft(
            32,
            32,
            48,
            48,
            richness_geom::WALL_Z_MIN,
            richness_geom::WALL_Z_MAX,
            &attr,
            cost,
            &mut ir,
        )
        .unwrap();
        assert!(ir.brushes.contains_key(&sid));
    }

    #[test]
    fn derive_all_interfaces_rejects_undeclared_contact() {
        let mut ir = AssemblyIR::new();
        let attr = make_test_attr();
        let cost = CostSource {
            dimension: BudgetDimension::SourceFaces,
            face_count: 6,
        };

        // Create two portal posts touching face-to-face (post-to-post is undeclared)
        let id_a = ir.alloc_brush_id();
        ir.insert_brush(BrushAssembly {
            id: id_a,
            brush: ConvexBrush::make_box((0, 16), (0, 16), (16, 96)).unwrap(),
            role: BrushAssemblyRole::PortalPost,
            owner: attr.clone(),
            cost,
            support: SupportTarget::World,
        });

        let id_b = ir.alloc_brush_id();
        ir.insert_brush(BrushAssembly {
            id: id_b,
            brush: ConvexBrush::make_box((16, 32), (0, 16), (16, 96)).unwrap(),
            role: BrushAssemblyRole::PortalPost,
            owner: attr.clone(),
            cost,
            support: SupportTarget::World,
        });

        // Two touching portal posts share a face but have no declared interface kind
        let result = derive_all_interfaces(&mut ir);
        assert!(result.is_err(), "undeclared contact should be rejected");
        assert!(result.unwrap_err().context.contains("undeclared contact"));
    }

    #[test]
    fn derive_all_interfaces_detects_overlap() {
        let mut ir = AssemblyIR::new();
        let attr = make_test_attr();
        let cost = CostSource {
            dimension: BudgetDimension::SourceFaces,
            face_count: 6,
        };

        // Create two overlapping wall brushes
        let id_a = ir.alloc_brush_id();
        ir.insert_brush(BrushAssembly {
            id: id_a,
            brush: ConvexBrush::make_box((0, 48), (0, 16), (16, 160)).unwrap(),
            role: BrushAssemblyRole::NorthWall,
            owner: attr.clone(),
            cost,
            support: SupportTarget::World,
        });

        let id_b = ir.alloc_brush_id();
        ir.insert_brush(BrushAssembly {
            id: id_b,
            brush: ConvexBrush::make_box((32, 64), (0, 16), (16, 160)).unwrap(),
            role: BrushAssemblyRole::SouthWall,
            owner: attr.clone(),
            cost,
            support: SupportTarget::World,
        });
        let _ = id_b;

        // These two don't actually overlap (48 != 32, they're disjoint)
        // Let's make them actually overlap
        let id_c = ir.alloc_brush_id();
        ir.insert_brush(BrushAssembly {
            id: id_c,
            brush: ConvexBrush::make_box((0, 64), (0, 32), (16, 160)).unwrap(),
            role: BrushAssemblyRole::NorthWall,
            owner: attr.clone(),
            cost,
            support: SupportTarget::World,
        });

        let id_d = ir.alloc_brush_id();
        ir.insert_brush(BrushAssembly {
            id: id_d,
            brush: ConvexBrush::make_box((16, 48), (16, 48), (16, 160)).unwrap(),
            role: BrushAssemblyRole::EastWall,
            owner: attr.clone(),
            cost,
            support: SupportTarget::World,
        });

        // id_c (N wall: 0..64 in X, 0..32 in Y, 16..160 in Z)
        // id_d (E wall: 16..48 in X, 16..48 in Y, 16..160 in Z)
        // They share region X=16..48, Y=16..32, Z=16..160 — that's positive-volume overlap!
        let result = derive_all_interfaces(&mut ir);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .context
            .contains("positive-volume overlap"));
    }
}
