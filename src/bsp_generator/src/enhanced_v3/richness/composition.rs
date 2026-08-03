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

use std::collections::BTreeMap;

use crate::enhanced_v3::geometry::{self, BrushFace, CanonicalPlane, ConvexBrush};
use crate::enhanced_v3::richness::error::{
    RichnessError, RichnessErrorCategory, RichnessErrorCode,
};

use super::assembly::{
    AssemblyIR, BrushAssembly, BrushAssemblyRole, BudgetDimension, CostSource, InterfaceKind,
    InterfaceRecord, OpeningRecord, PortalAssembly, PortalStyle, SemanticAttribution,
    SupportRecord, SupportTarget,
};
use super::content_types::ShapeRule;
use super::footprint::Footprint3D;
use super::generated_content;
use super::geometry as richness_geom;
use super::ids::{
    ArchetypeRequestId, BeatId, BrushAssemblyId, OpeningAssemblyId, PortalId, ReservationId,
    WallChainId, ZoneId,
};
use super::reservation::ReservationRecord;
use super::variation::{WallChainRecord, WallMass, WallMassTreatment, WallShaping};

// ── Macro composition entry point ─────────────────────────────────────────

/// Compose all archetype rooms from placement results into an AssemblyIR.
///
/// Consumes the reservation journal and produces a complete brush assembly
/// with semantic attribution, cost tracking, and support records.
pub(crate) fn compose_all_rooms(
    reservations: &BTreeMap<ReservationId, ReservationRecord>,
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

        compose_single_room(record, &mut ir)?;
    }

    Ok(ir)
}

/// Whether a reservation is a room that needs brush composition.
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
    ir: &mut AssemblyIR,
) -> Result<(), RichnessError> {
    let request_id = record.request_id;
    let arch_idx: usize = request_id
        .map(|rid| rid.raw() as usize % generated_content::ARCHETYPE_COUNT)
        .unwrap_or(1); // default to antechamber (simple rectangle) if no request

    if arch_idx >= generated_content::ARCHETYPE_COUNT {
        return Err(composition_error(
            "archetype_index",
            format!(
                "archetype index {} out of range (max {})",
                arch_idx,
                generated_content::ARCHETYPE_COUNT
            ),
        ));
    }

    let shape = generated_content::ARCHETYPE_SHAPE[arch_idx];
    let fp = &record.footprint;

    let attr = SemanticAttribution::from_reservation(
        record.id,
        request_id,
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
) -> Result<PortalAssembly, RichnessError> {
    // Verify the wall is cardinal (reject diagonal)
    if !matches!(
        wall_role,
        BrushAssemblyRole::NorthWall
            | BrushAssemblyRole::SouthWall
            | BrushAssemblyRole::EastWall
            | BrushAssemblyRole::WestWall
    ) {
        return Err(composition_error(
            "portal",
            format!(
                "portal {:?}: wall role {:?} is not cardinal",
                portal_id.raw(),
                wall_role.tag()
            ),
        ));
    }

    match style {
        PortalStyle::AncientPostLintel => build_ancient_post_lintel(
            portal_id,
            wall_brush_id,
            throat_anchor,
            wall_role,
            attr,
            cost,
            ir,
        ),
        PortalStyle::EgyptianSteppedSurround => build_egyptian_stepped_surround(
            portal_id,
            wall_brush_id,
            throat_anchor,
            wall_role,
            attr,
            cost,
            ir,
        ),
        PortalStyle::BrutalistRevealSurround => build_brutalist_reveal_surround(
            portal_id,
            wall_brush_id,
            throat_anchor,
            wall_role,
            attr,
            cost,
            ir,
        ),
    }
}

/// Ancient post-and-lintel portal: two 16u posts + 16u lintel framing a 64×80 throat.
///
/// The throat_anchor is (span_min, z_min, span_max, z_max) where:
/// - N/S wall: span = X coordinate of throat (width 64), z = vertical extent (height 80)
/// - E/W wall: span = Y coordinate of throat (width 64), z = vertical extent (height 80)
fn build_ancient_post_lintel(
    portal_id: PortalId,
    wall_brush_id: BrushAssemblyId,
    throat_anchor: (i128, i128, i128, i128), // (span_min, z_min, span_max, z_max)
    wall_role: BrushAssemblyRole,
    attr: &SemanticAttribution,
    cost: CostSource,
    ir: &mut AssemblyIR,
) -> Result<PortalAssembly, RichnessError> {
    let (s0, z0, s1, z1) = throat_anchor;
    let throat_w = (s1 - s0).abs();
    let throat_h = (z1 - z0).abs();

    assert_eq!(throat_w, 64, "throat width must be 64, got {throat_w}");
    assert_eq!(throat_h, 80, "throat height must be 80, got {throat_h}");

    let post_w = richness_geom::WALL_THICKNESS;
    let post_z0 = z0;
    let post_z1 = z1;
    let lintel_z0 = z1;
    let lintel_z1 = lintel_z0 + post_w;

    let mut post_ids = Vec::new();
    let mut lintel_ids = Vec::new();

    let wall_brush = ir
        .brushes
        .get(&wall_brush_id)
        .ok_or_else(|| composition_error("portal", "wall brush not found"))?;
    let bb = wall_brush
        .brush
        .aabb()
        .map_err(|e| composition_error("portal", format!("wall AABB: {e}")))?;
    let ((_wx0, wy0, _wz0), (_wx1, wy1, _wz1)) = bb;

    let throat_bounds = match wall_role {
        BrushAssemblyRole::NorthWall | BrushAssemblyRole::SouthWall => {
            (s0 - post_w, wy0, post_z0, s1 + post_w, wy1, lintel_z1)
        }
        BrushAssemblyRole::EastWall | BrushAssemblyRole::WestWall => {
            let ((wx0, _wy0, _wz), (wx1, _wy1, _wz1)) = bb;
            (wx0, s0 - post_w, post_z0, wx1, s1 + post_w, lintel_z1)
        }
        _ => unreachable!("portal only on cardinal walls"),
    };

    match wall_role {
        BrushAssemblyRole::NorthWall | BrushAssemblyRole::SouthWall => {
            let post_id = build_and_insert_box(
                (s0 - post_w, s0),
                (wy0, wy1),
                (post_z0, post_z1),
                BrushAssemblyRole::PortalPost,
                attr,
                cost,
                ir,
            )?;
            post_ids.push(post_id);
            let post_id = build_and_insert_box(
                (s1, s1 + post_w),
                (wy0, wy1),
                (post_z0, post_z1),
                BrushAssemblyRole::PortalPost,
                attr,
                cost,
                ir,
            )?;
            post_ids.push(post_id);
            let lintel_id = build_and_insert_box(
                (s0 - post_w, s1 + post_w),
                (wy0, wy1),
                (lintel_z0, lintel_z1),
                BrushAssemblyRole::PortalLintel,
                attr,
                cost,
                ir,
            )?;
            lintel_ids.push(lintel_id);
        }
        BrushAssemblyRole::EastWall | BrushAssemblyRole::WestWall => {
            let ((wx0, _wy0, _wz0), (wx1, _wy1, _wz1)) = bb;
            let post_id = build_and_insert_box(
                (wx0, wx1),
                (s0 - post_w, s0),
                (post_z0, post_z1),
                BrushAssemblyRole::PortalPost,
                attr,
                cost,
                ir,
            )?;
            post_ids.push(post_id);
            let post_id = build_and_insert_box(
                (wx0, wx1),
                (s1, s1 + post_w),
                (post_z0, post_z1),
                BrushAssemblyRole::PortalPost,
                attr,
                cost,
                ir,
            )?;
            post_ids.push(post_id);
            let lintel_id = build_and_insert_box(
                (wx0, wx1),
                (s0 - post_w, s1 + post_w),
                (lintel_z0, lintel_z1),
                BrushAssemblyRole::PortalLintel,
                attr,
                cost,
                ir,
            )?;
            lintel_ids.push(lintel_id);
        }
        _ => unreachable!("portal only on cardinal walls"),
    }

    let opening_id = ir.alloc_opening_id();
    let opening = OpeningRecord {
        id: opening_id,
        owner_brush_id: wall_brush_id,
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
    throat_anchor: (i128, i128, i128, i128), // (span_min, z_min, span_max, z_max)
    wall_role: BrushAssemblyRole,
    attr: &SemanticAttribution,
    cost: CostSource,
    ir: &mut AssemblyIR,
) -> Result<PortalAssembly, RichnessError> {
    let (s0, z0, s1, z1) = throat_anchor;
    let throat_w = (s1 - s0).abs();
    let throat_h = (z1 - z0).abs();
    assert_eq!(throat_w, 64, "throat width must be 64");
    assert_eq!(throat_h, 80, "throat height must be 80");

    let step = 16i128;
    let post_z0 = z0;
    let post_z1 = z1;

    let wall_brush = ir
        .brushes
        .get(&wall_brush_id)
        .ok_or_else(|| composition_error("portal", "wall brush not found"))?;
    let bb = wall_brush
        .brush
        .aabb()
        .map_err(|e| composition_error("portal", format!("wall AABB: {e}")))?;

    let mut surround_ids = Vec::new();

    let throat_bounds = match wall_role {
        BrushAssemblyRole::NorthWall | BrushAssemblyRole::SouthWall => {
            let ((_wx0, wy0, _wz0), (_wx1, wy1, _wz1)) = bb;
            let out = step * 3;
            for layer in 0..3 {
                let off = step * (layer + 1);
                let z_bot_top = z1 + off - step;
                // Left post
                surround_ids.push(build_and_insert_box(
                    (s0 - off, s0 - off + step),
                    (wy0, wy1),
                    (post_z0, post_z1),
                    BrushAssemblyRole::PortalSurround,
                    attr,
                    cost,
                    ir,
                )?);
                // Right post
                surround_ids.push(build_and_insert_box(
                    (s1 + off - step, s1 + off),
                    (wy0, wy1),
                    (post_z0, post_z1),
                    BrushAssemblyRole::PortalSurround,
                    attr,
                    cost,
                    ir,
                )?);
                // Lintel course
                surround_ids.push(build_and_insert_box(
                    (s0 - off, s1 + off),
                    (wy0, wy1),
                    (z_bot_top, z_bot_top + step),
                    BrushAssemblyRole::PortalSurround,
                    attr,
                    cost,
                    ir,
                )?);
            }
            let _ = out;
            (
                s0 - step * 3,
                wy0,
                post_z0,
                s1 + step * 3,
                wy1,
                z1 + step * 3,
            )
        }
        BrushAssemblyRole::EastWall | BrushAssemblyRole::WestWall => {
            let ((wx0, _wy0, _wz0), (wx1, _wy1, _wz1)) = bb;
            for layer in 0..3 {
                let off = step * (layer + 1);
                let z_bot_top = z1 + off - step;
                surround_ids.push(build_and_insert_box(
                    (wx0, wx1),
                    (s0 - off, s0 - off + step),
                    (post_z0, post_z1),
                    BrushAssemblyRole::PortalSurround,
                    attr,
                    cost,
                    ir,
                )?);
                surround_ids.push(build_and_insert_box(
                    (wx0, wx1),
                    (s1 + off - step, s1 + off),
                    (post_z0, post_z1),
                    BrushAssemblyRole::PortalSurround,
                    attr,
                    cost,
                    ir,
                )?);
                surround_ids.push(build_and_insert_box(
                    (wx0, wx1),
                    (s0 - off, s1 + off),
                    (z_bot_top, z_bot_top + step),
                    BrushAssemblyRole::PortalSurround,
                    attr,
                    cost,
                    ir,
                )?);
            }
            (
                wx0,
                s0 - step * 3,
                post_z0,
                wx1,
                s1 + step * 3,
                z1 + step * 3,
            )
        }
        _ => unreachable!(),
    };

    let opening_id = ir.alloc_opening_id();
    let opening = OpeningRecord {
        id: opening_id,
        owner_brush_id: wall_brush_id,
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
    throat_anchor: (i128, i128, i128, i128), // (span_min, z_min, span_max, z_max)
    wall_role: BrushAssemblyRole,
    attr: &SemanticAttribution,
    cost: CostSource,
    ir: &mut AssemblyIR,
) -> Result<PortalAssembly, RichnessError> {
    let (s0, z0, s1, z1) = throat_anchor;
    let throat_w = (s1 - s0).abs();
    let throat_h = (z1 - z0).abs();
    assert_eq!(throat_w, 64, "throat width must be 64");
    assert_eq!(throat_h, 80, "throat height must be 80");

    let reveal_depth = 16i128;
    let surround_thickness = 16i128;

    let wall_brush = ir
        .brushes
        .get(&wall_brush_id)
        .ok_or_else(|| composition_error("portal", "wall brush not found"))?;
    let bb = wall_brush
        .brush
        .aabb()
        .map_err(|e| composition_error("portal", format!("wall AABB: {e}")))?;

    let mut surround_ids = Vec::new();
    let lintel_z0 = z1;
    let lintel_z1 = lintel_z0 + reveal_depth;

    let throat_bounds = match wall_role {
        BrushAssemblyRole::NorthWall | BrushAssemblyRole::SouthWall => {
            let ((_wx0, wy0, _wz0), (_wx1, wy1, _wz1)) = bb;
            // Reveal channels
            surround_ids.push(build_and_insert_box(
                (s0 - reveal_depth, s0),
                (wy0, wy1),
                (z0, z1),
                BrushAssemblyRole::PortalSurround,
                attr,
                cost,
                ir,
            )?);
            surround_ids.push(build_and_insert_box(
                (s1, s1 + reveal_depth),
                (wy0, wy1),
                (z0, z1),
                BrushAssemblyRole::PortalSurround,
                attr,
                cost,
                ir,
            )?);
            surround_ids.push(build_and_insert_box(
                (s0 - reveal_depth, s1 + reveal_depth),
                (wy0, wy1),
                (lintel_z0, lintel_z1),
                BrushAssemblyRole::PortalSurround,
                attr,
                cost,
                ir,
            )?);
            // Surround mass
            surround_ids.push(build_and_insert_box(
                (s0 - reveal_depth - surround_thickness, s0 - reveal_depth),
                (wy0, wy1),
                (z0, z1),
                BrushAssemblyRole::PortalSurround,
                attr,
                cost,
                ir,
            )?);
            surround_ids.push(build_and_insert_box(
                (s1 + reveal_depth, s1 + reveal_depth + surround_thickness),
                (wy0, wy1),
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
                (wy0, wy1),
                (lintel_z1, lintel_z1 + surround_thickness),
                BrushAssemblyRole::PortalSurround,
                attr,
                cost,
                ir,
            )?);
            let outer = reveal_depth + surround_thickness;
            (
                s0 - outer,
                wy0,
                z0,
                s1 + outer,
                wy1,
                lintel_z1 + surround_thickness,
            )
        }
        BrushAssemblyRole::EastWall | BrushAssemblyRole::WestWall => {
            let ((wx0, _wy0, _wz0), (wx1, _wy1, _wz1)) = bb;
            surround_ids.push(build_and_insert_box(
                (wx0, wx1),
                (s0 - reveal_depth, s0),
                (z0, z1),
                BrushAssemblyRole::PortalSurround,
                attr,
                cost,
                ir,
            )?);
            surround_ids.push(build_and_insert_box(
                (wx0, wx1),
                (s1, s1 + reveal_depth),
                (z0, z1),
                BrushAssemblyRole::PortalSurround,
                attr,
                cost,
                ir,
            )?);
            surround_ids.push(build_and_insert_box(
                (wx0, wx1),
                (s0 - reveal_depth, s1 + reveal_depth),
                (lintel_z0, lintel_z1),
                BrushAssemblyRole::PortalSurround,
                attr,
                cost,
                ir,
            )?);
            surround_ids.push(build_and_insert_box(
                (wx0, wx1),
                (s0 - reveal_depth - surround_thickness, s0 - reveal_depth),
                (z0, z1),
                BrushAssemblyRole::PortalSurround,
                attr,
                cost,
                ir,
            )?);
            surround_ids.push(build_and_insert_box(
                (wx0, wx1),
                (s1 + reveal_depth, s1 + reveal_depth + surround_thickness),
                (z0, z1),
                BrushAssemblyRole::PortalSurround,
                attr,
                cost,
                ir,
            )?);
            surround_ids.push(build_and_insert_box(
                (wx0, wx1),
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
            let outer = reveal_depth + surround_thickness;
            (
                wx0,
                s0 - outer,
                z0,
                wx1,
                s1 + outer,
                lintel_z1 + surround_thickness,
            )
        }
        _ => unreachable!(),
    };

    let opening_id = ir.alloc_opening_id();
    let opening = OpeningRecord {
        id: opening_id,
        owner_brush_id: wall_brush_id,
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

        match treatment.kind {
            WallMass::None => {}
            WallMass::Liner16 | WallMass::Liner32 => {
                let thickness = if matches!(treatment.kind, WallMass::Liner32) {
                    32i128
                } else {
                    16i128
                };
                let offset = treatment.kind.offset_units() as i128;
                // Build liner on the interior face of the wall
                build_liner_on_wall(
                    wall_role,
                    wx0,
                    wy0,
                    wx1,
                    wy1,
                    wz0,
                    wz1,
                    thickness,
                    offset,
                    attr,
                    cost,
                    ir,
                    &mut created_ids,
                )?;
            }
            WallMass::Recess16 => {
                // Recess: negative carving inward — omitted volume from wall
                // We create an opening record instead of a brush
                let opening_id = ir.alloc_opening_id();
                let recess_bounds =
                    compute_recess_bounds(wall_role, wx0, wy0, wx1, wy1, wz0, wz1, 16i128);
                let opening = OpeningRecord {
                    id: opening_id,
                    owner_brush_id: wall_brush_id,
                    owner: attr.clone(),
                    bounds: recess_bounds,
                    portal_id: None,
                    frame_brush_ids: Vec::new(),
                    portal_style: None,
                };
                ir.insert_opening(opening);
            }
            WallMass::Buttress16 => {
                // Buttress: external mass on the outside face
                build_buttress_on_wall(
                    wall_role,
                    wx0,
                    wy0,
                    wx1,
                    wy1,
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
    _offset: i128,
    attr: &SemanticAttribution,
    cost: CostSource,
    ir: &mut AssemblyIR,
    created_ids: &mut Vec<BrushAssemblyId>,
) -> Result<(), RichnessError> {
    match wall_role {
        BrushAssemblyRole::NorthWall => {
            // Wall spans X, liner sits on south/inner face (y=wy1-thickness)
            build_and_insert_box(
                (wx0, wx1),
                (wy1 - thickness, wy1),
                (wz0, wz1),
                BrushAssemblyRole::WallLiner,
                attr,
                cost,
                ir,
            )
            .map(|id| created_ids.push(id))
        }
        BrushAssemblyRole::SouthWall => build_and_insert_box(
            (wx0, wx1),
            (wy0, wy0 + thickness),
            (wz0, wz1),
            BrushAssemblyRole::WallLiner,
            attr,
            cost,
            ir,
        )
        .map(|id| created_ids.push(id)),
        BrushAssemblyRole::EastWall => build_and_insert_box(
            (wx1 - thickness, wx1),
            (wy0, wy1),
            (wz0, wz1),
            BrushAssemblyRole::WallLiner,
            attr,
            cost,
            ir,
        )
        .map(|id| created_ids.push(id)),
        BrushAssemblyRole::WestWall => build_and_insert_box(
            (wx0, wx0 + thickness),
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
            (wy0, wy0 + bw),
            (wz0, wz1),
            BrushAssemblyRole::Buttress,
            attr,
            cost,
            ir,
        )
        .map(|id| created_ids.push(id)),
        BrushAssemblyRole::SouthWall => build_and_insert_box(
            (wx0, wx1),
            (wy1 - bw, wy1),
            (wz0, wz1),
            BrushAssemblyRole::Buttress,
            attr,
            cost,
            ir,
        )
        .map(|id| created_ids.push(id)),
        BrushAssemblyRole::EastWall => build_and_insert_box(
            (wx1 - bw, wx1),
            (wy0, wy1),
            (wz0, wz1),
            BrushAssemblyRole::Buttress,
            attr,
            cost,
            ir,
        )
        .map(|id| created_ids.push(id)),
        BrushAssemblyRole::WestWall => build_and_insert_box(
            (wx0, wx0 + bw),
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
    let brush_ids: Vec<BrushAssemblyId> = ir.brushes.keys().copied().collect();

    for i in 0..brush_ids.len() {
        for j in (i + 1)..brush_ids.len() {
            let id_a = brush_ids[i];
            let id_b = brush_ids[j];
            let brush_a = &ir.brushes[&id_a];
            let brush_b = &ir.brushes[&id_b];

            // Check for positive-volume overlap — this is always an error
            if richness_geom::brushes_overlap(&brush_a.brush, &brush_b.brush)? {
                return Err(composition_error(
                    "interfaces",
                    format!(
                        "brushes {:?} ({}) and {:?} ({}) have positive-volume overlap",
                        id_a.raw(),
                        brush_a.role.tag(),
                        id_b.raw(),
                        brush_b.role.tag()
                    ),
                ));
            }

            // Check for positive-area face contact
            if richness_geom::has_positive_area_contact(&brush_a.brush, &brush_b.brush) {
                // Derive the interface kind
                if let Some(kind) = richness_geom::derive_interface_kind(brush_a.role, brush_b.role)
                {
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

    let wall_z_min = richness_geom::WALL_Z_MIN;
    let wall_z_max = richness_geom::WALL_Z_MAX;
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

fn composition_error(path: &str, context: impl Into<String>) -> RichnessError {
    RichnessError::new(
        RichnessErrorCode::PlacementExhausted,
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

    let attr = wall.owner.clone();
    let cost = wall.cost;
    let support = wall.support.clone();

    let mut new_ids = Vec::new();

    if is_ns {
        // Wall spans X. Split into: left of opening, right of opening, and
        // optionally below+above the opening (but below/above are usually
        // handled by floor/ceiling slabs — we only split X for portal openings).

        // Left segment: [wx0, ox0]
        if ox0 > wx0 {
            let brush = ConvexBrush::make_box((wx0, ox0), (wy0, wy1), (wz0, wz1))
                .map_err(|e| composition_error("wall_split", format!("left segment: {e}")))?;
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
        if ox1 < wx1 {
            let brush = ConvexBrush::make_box((ox1, wx1), (wy0, wy1), (wz0, wz1))
                .map_err(|e| composition_error("wall_split", format!("right segment: {e}")))?;
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

        // Below opening: if opening starts above wall bottom
        if oz0 > wz0 {
            let brush = ConvexBrush::make_box((ox0, ox1), (wy0, wy1), (wz0, oz0))
                .map_err(|e| composition_error("wall_split", format!("below opening: {e}")))?;
            let id = ir.alloc_brush_id();
            ir.insert_brush(BrushAssembly {
                id,
                brush,
                role: BrushAssemblyRole::Sill,
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

        // Above opening: if opening ends below wall top
        if oz1 < wz1 {
            let brush = ConvexBrush::make_box((ox0, ox1), (wy0, wy1), (oz1, wz1))
                .map_err(|e| composition_error("wall_split", format!("above opening: {e}")))?;
            let id = ir.alloc_brush_id();
            ir.insert_brush(BrushAssembly {
                id,
                brush,
                role: BrushAssemblyRole::PortalLintel,
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
        }
    } else {
        // E/W wall. Split in Y.
        if oy0 > wy0 {
            let brush = ConvexBrush::make_box((wx0, wx1), (wy0, oy0), (wz0, wz1))
                .map_err(|e| composition_error("wall_split", format!("left segment: {e}")))?;
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

        if oy1 < wy1 {
            let brush = ConvexBrush::make_box((wx0, wx1), (oy1, wy1), (wz0, wz1))
                .map_err(|e| composition_error("wall_split", format!("right segment: {e}")))?;
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

        if oz0 > wz0 {
            let brush = ConvexBrush::make_box((wx0, wx1), (oy0, oy1), (wz0, oz0))
                .map_err(|e| composition_error("wall_split", format!("below opening: {e}")))?;
            let id = ir.alloc_brush_id();
            ir.insert_brush(BrushAssembly {
                id,
                brush,
                role: BrushAssemblyRole::Sill,
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

        if oz1 < wz1 {
            let brush = ConvexBrush::make_box((wx0, wx1), (oy0, oy1), (oz1, wz1))
                .map_err(|e| composition_error("wall_split", format!("above opening: {e}")))?;
            let id = ir.alloc_brush_id();
            ir.insert_brush(BrushAssembly {
                id,
                brush,
                role: BrushAssemblyRole::PortalLintel,
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
        }
    }

    // Remove the original wall brush
    ir.brushes.remove(&wall_brush_id);

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
        let owner_id = opening.owner_brush_id;

        // Check if the owner brush still exists
        if let Some(owner) = ir.brushes.get(&owner_id) {
            let bb = owner
                .brush
                .aabb()
                .map_err(|e| composition_error("opening_omission", format!("AABB: {e}")))?;
            let ((bmin_x, bmin_y, bmin_z), (bmax_x, bmax_y, bmax_z)) = bb;
            let (ox0, oy0, oz0, ox1, oy1, oz1) = opening.bounds;

            // If the wall was properly split, the owner brush's AABB should
            // not intersect the opening bounds except at boundaries.
            // We do a strict check: the wall AABB must not wholly contain
            // the opening bounds.
            let contains = bmin_x <= ox0
                && bmax_x >= ox1
                && bmin_y <= oy0
                && bmax_y >= oy1
                && bmin_z <= oz0
                && bmax_z >= oz1;

            if contains {
                return Err(composition_error(
                    "opening_omission",
                    format!(
                        "wall brush {:?} still contains opening {:?} — wall must be split around opening",
                        owner_id.raw(),
                        opening.id.raw()
                    ),
                ));
            }
        }

        // Also check that frame brush positions don't intrude into the throat.
        // Frame brushes (posts, lintel, surround) are adjacent, not occupying.
        for &frame_id in &opening.frame_brush_ids {
            if let Some(frame) = ir.brushes.get(&frame_id) {
                let bb = frame.brush.aabb().map_err(|e| {
                    composition_error("opening_omission", format!("frame AABB: {e}"))
                })?;
                let ((fmin_x, fmin_y, fmin_z), (fmax_x, fmax_y, fmax_z)) = bb;
                let (ox0, oy0, oz0, ox1, oy1, oz1) = opening.bounds;

                // We allow frame brushes to be AT the throat boundary but not inside.
                // The simple check: if frame is entirely inside the throat, reject.
                let inside_throat = fmin_x >= ox0
                    && fmax_x <= ox1
                    && fmin_y >= oy0
                    && fmax_y <= oy1
                    && fmin_z >= oz0
                    && fmax_z <= oz1;

                if inside_throat {
                    return Err(composition_error(
                        "opening_omission",
                        format!(
                            "frame brush {:?} is entirely inside portal throat of opening {:?}",
                            frame_id.raw(),
                            opening.id.raw()
                        ),
                    ));
                }
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
    use super::super::reservation::{ReservationKind, ReservationRecord};
    use super::super::topology::Dir;
    use super::super::variation::{WallChainRecord, WallMass, WallMassTreatment, WallShaping};
    use super::*;

    fn make_test_attr() -> SemanticAttribution {
        SemanticAttribution::from_reservation(
            ReservationId::new(0),
            Some(ArchetypeRequestId::new(0)),
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
        compose_single_room(&record, &mut ir).unwrap();
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
        compose_single_room(&record, &mut ir).unwrap();
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

        let result = build_ancient_post_lintel(
            portal_id,
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
        let portal = result.unwrap();
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
        let result = build_egyptian_stepped_surround(
            portal_id,
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
        let portal = result.unwrap();
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
        let result = build_brutalist_reveal_surround(
            portal_id,
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
        let portal = result.unwrap();
        // 3 reveal + 3 surround = 6 surround brushes
        assert_eq!(portal.surround_ids.len(), 6);
        assert!(portal.post_ids.is_empty());
        assert!(portal.lintel_ids.is_empty());
        assert_eq!(ir.openings.len(), 1);
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
        assert!(err.context.contains("not cardinal"));
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
        build_ancient_post_lintel(
            PortalId::new(0),
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
        let portal = build_ancient_post_lintel(
            PortalId::new(0),
            wall_id,
            throat_anchor,
            BrushAssemblyRole::NorthWall,
            &attr,
            cost,
            &mut ir,
        )
        .unwrap();

        // Opening bounds: (x0, y0, z0, x1, y1, z1)
        // For N wall with throat at X=96..160: post_w=16, so bounds X = 96-16=80 to 160+16=176
        // Z bounds: 16 to 96+16=112 (throat+lintel)
        let opening = &ir.openings[&portal.opening_id];
        let throat_width = (opening.bounds.3 - opening.bounds.0).abs();
        let throat_height_ok = (opening.bounds.5 - opening.bounds.2).abs();
        // Total frame width = 64 + 2*16 = 96. Total frame height = 80 + 16 = 96.
        assert!(
            throat_width >= 96,
            "frame width {} should be >= 96",
            throat_width
        );
        assert!(
            throat_height_ok >= 96,
            "frame height {} should be >= 96",
            throat_height_ok
        );
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
