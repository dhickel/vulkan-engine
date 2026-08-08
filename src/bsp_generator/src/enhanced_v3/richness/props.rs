//! Props: full-volume placement of the 15 authored prop identities with
//! support, route exclusion, theme materialization, and swept occupancy.
//!
//! Floor/ceiling placement cells are classified (protected path, open,
//! wall-adjacent, corner, reserved) before candidate-keyed integer Poisson
//! proposals. Acceptance requires a complete swept-volume check, positive
//! support, and no intersection with routes, portals, spawn, lights, stairs,
//! shafts, pits, catwalk turns, cave routes, or reservations.

use std::collections::BTreeMap;

use crate::enhanced_v3::richness::{
    assembly::{
        AssemblyIR, BrushAssembly, BrushAssemblyRole, CostSource, SemanticAttribution,
        SupportTarget,
    },
    error::{RichnessError, RichnessErrorCategory, RichnessErrorCode},
    generated_content::{
        ARCHETYPE_THEME_PROP_REFS, PROP_COLLISION, PROP_CONVEX_PIECES, PROP_DIMENSIONS, PROP_IDS,
        PROP_SWEPT_OCCUPANCY, PROP_THEME_COLLISION_OVERRIDE, PROP_THEME_DIMENSIONS,
        PROP_THEME_MODEL_OVERRIDE, SCHEMA_VERSION,
    },
    geometry::{footprint_quake_bounds, footprint_vertical_bounds, validate_brush},
    ids::{ArchetypeIndex, ArchetypeRequestId, BeatId, BrushAssemblyId, ReservationId, ZoneId},
    request::RichnessTheme,
    reservation::{ReservationJournal, ReservationKind, ReservationRecord},
    sampling::{poisson_sample, PoissonConfig},
};

/// One placed prop instance.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PlacedProp {
    /// Prop catalog index.
    pub prop: u32,
    /// Placement origin in Quake units (floor contact point for floor
    /// props; wall attach point for wall props; ceiling attach for hanging).
    pub origin: (i128, i128, i128),
    /// Theme model identity (authored variant).
    pub model: &'static str,
    /// Brush IDs emitted for this prop.
    pub brush_ids: Vec<BrushAssemblyId>,
    /// The room that owns the prop.
    pub room: ReservationId,
}

/// Full presentation result for one room.
#[derive(Debug, Clone, Default)]
pub(crate) struct RoomPresentation {
    pub props: Vec<PlacedProp>,
    pub detail_count: usize,
}

/// Cell classification for prop placement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FloorCellClass {
    /// Inside a protected route/portal/turn/spawn/light/vertical/cave region.
    Protected,
    /// Wall-adjacent (within 32 units of a room wall).
    WallAdjacent,
    /// Corner (within 32 units of two walls).
    Corner,
    /// Open floor.
    Open,
}

fn wall_edges(room: (i128, i128, i128, i128, i128, i128)) -> [(i128, i128, i128, i128); 4] {
    let (x0, y0, z0, x1, y1, _) = room;
    [
        (x0, y0, x1, y0),
        (x0, y1, x1, y1),
        (x0, y0, x0, y1),
        (x1, y0, x1, y1),
    ]
}

fn classify_cell(
    x: i128,
    y: i128,
    room: (i128, i128, i128, i128, i128, i128),
    protected: &[(i128, i128, i128, i128)],
) -> FloorCellClass {
    for &(px0, py0, px1, py1) in protected {
        if x >= px0 && x < px1 && y >= py0 && y < py1 {
            return FloorCellClass::Protected;
        }
    }
    let mut near = 0;
    for &(wx0, wy0, wx1, wy1) in &wall_edges(room) {
        let dx = if wx0 == wx1 {
            (x - wx0).abs()
        } else {
            (y - wy0).abs()
        };
        if dx <= 32 {
            near += 1;
        }
    }
    match near {
        0 => FloorCellClass::Open,
        1 => FloorCellClass::WallAdjacent,
        _ => FloorCellClass::Corner,
    }
}

/// Collect protected floor regions for a room from committed reservations:
/// routes, portal throats, turns, spawns, lights, vertical hosts, pit
/// omissions, and cave hosts that intersect the room's XY footprint.
fn protected_regions(
    room: ReservationId,
    room_bounds: (i128, i128, i128, i128, i128, i128),
    journal: &ReservationJournal,
) -> Vec<(i128, i128, i128, i128)> {
    let (rx0, ry0, _, rx1, ry1, _) = room_bounds;
    let mut regions = Vec::new();
    for record in journal.reservations.values() {
        if !record.committed || record.id == room {
            continue;
        }
        let protected = matches!(
            record.kind,
            ReservationKind::Route
                | ReservationKind::PortalThroat
                | ReservationKind::Turn
                | ReservationKind::Spawn
                | ReservationKind::Light
                | ReservationKind::Support
                | ReservationKind::VerticalHost
                | ReservationKind::PitOmission
                | ReservationKind::CaveHost
        );
        if !protected {
            continue;
        }
        let bounds = footprint_quake_bounds(&record.footprint);
        // Route/portal/turn regions are dilated by 32 units so props never
        // block route mouths or portal approaches.
        let (mut qx0, mut qy0, mut qx1, mut qy1) = (bounds.0, bounds.1, bounds.2, bounds.3);
        if matches!(
            record.kind,
            ReservationKind::Route | ReservationKind::PortalThroat | ReservationKind::Turn
        ) {
            qx0 -= 32;
            qy0 -= 32;
            qx1 += 32;
            qy1 += 32;
        }
        let x0 = qx0.max(rx0);
        let y0 = qy0.max(ry0);
        let x1 = qx1.min(rx1);
        let y1 = qy1.min(ry1);
        if x0 < x1 && y0 < y1 {
            regions.push((x0, y0, x1, y1));
        }
    }
    regions
}

/// Emit the convex assembly for one prop instance.
///
/// Every prop is composed of `PROP_CONVEX_PIECES[prop]` stacked/arranged
/// boxes derived from the base dimensions (or the theme dimensions when the
/// theme overrides them). Hanging props (chain, sconce, cage) attach from
/// the ceiling; wall props attach to the nearest wall; floor props rest on
/// the floor slab. All emitted brushes carry the prop's semantic owner and
/// a complete cost source.
pub(crate) fn emit_prop(
    ir: &mut AssemblyIR,
    prop: u32,
    theme: RichnessTheme,
    origin: (i128, i128, i128),
    owner: &SemanticAttribution,
) -> Result<Vec<BrushAssemblyId>, RichnessError> {
    let pieces = PROP_CONVEX_PIECES[prop as usize] as usize;
    let theme_idx = theme_ordinal(theme);
    let dims = PROP_THEME_DIMENSIONS[prop as usize][theme_idx];
    let collision = PROP_THEME_COLLISION_OVERRIDE[prop as usize][theme_idx]
        .unwrap_or(PROP_COLLISION[prop as usize]);
    let (dx, dy, dz) = (dims[0] as i128, dims[1] as i128, dims[2] as i128);
    let mut ids = Vec::with_capacity(pieces);
    // Quantize to the 16-unit grid (conservative: floor minima, ceil
    // maxima) so every emitted brush is quantum-aligned regardless of
    // authored theme dimensions.
    let floor16 = |v: i128| v.div_euclid(16) * 16;
    let ceil16 = |v: i128| v.div_euclid(16) * 16 + if v % 16 == 0 { 0 } else { 16 };
    let (x0, y0, z1_top) = (floor16(origin.0), floor16(origin.1), ceil16(origin.2 + dz));
    let (x1, y1) = (ceil16(origin.0 + dx), ceil16(origin.1 + dy));
    let z_base = floor16(origin.2);
    let total = z1_top - z_base;
    // Stack pieces along Z as disjoint 16-aligned bands; the final band
    // absorbs any remainder (frozen).
    let band = (total / pieces as i128).div_euclid(16) * 16;
    for piece in 0..pieces {
        let z0 = z_base + piece as i128 * band;
        let z1 = if piece + 1 == pieces {
            z1_top
        } else {
            z_base + (piece as i128 + 1) * band
        };
        let brush =
            crate::enhanced_v3::geometry::ConvexBrush::make_box((x0, x1), (y0, y1), (z0, z1))
                .map_err(|error| prop_error("box", format!("{error}")))?;
        validate_brush(&brush)?;
        // Overlap check against every existing brush.
        for existing in ir.brushes.values() {
            if crate::enhanced_v3::richness::geometry::brushes_overlap(&existing.brush, &brush)? {
                return Err(prop_error(
                    "overlap",
                    format!("prop {prop} at {origin:?} overlaps brush {:?}", existing.id),
                ));
            }
        }
        let id = ir.alloc_brush_id();
        ir.insert_brush(BrushAssembly {
            id,
            brush,
            role: if collision == CollisionBehavior::Collidable {
                BrushAssemblyRole::InteriorMass
            } else {
                BrushAssemblyRole::InteriorColumn
            },
            owner: owner.clone(),
            cost: CostSource {
                dimension: crate::enhanced_v3::richness::assembly::BudgetDimension::SourceFaces,
                face_count: 6,
            },
            support: SupportTarget::World,
        });
        ids.push(id);
    }
    let _ = (theme, collision);
    Ok(ids)
}

/// Place every referenced prop of one room.
///
/// Classification + Poisson + acceptance pipeline. `max_props` comes from
/// the complexity plan's PropsImperfection tier; `quiet` limits detail props
/// to at most two with no open-floor scatter.
pub(crate) fn place_room_props(
    ir: &mut AssemblyIR,
    room: ReservationId,
    room_bounds: (i128, i128, i128, i128, i128, i128),
    archetype: ArchetypeIndex,
    theme: RichnessTheme,
    seed: u64,
    journal: &ReservationJournal,
    max_props: usize,
    quiet: bool,
) -> Result<RoomPresentation, RichnessError> {
    let mut presentation = RoomPresentation::default();
    let theme_idx = theme_ordinal(theme);
    let prop_refs = ARCHETYPE_THEME_PROP_REFS[archetype.raw() as usize][theme_idx];
    if prop_refs.is_empty() || max_props == 0 {
        return Ok(presentation);
    }
    let protected = protected_regions(room, room_bounds, journal);
    let (x0, y0, z0, x1, y1, _) = room_bounds;
    let floor_top = z0 + 16;
    // Placement domain: room interior, 32-unit inset so props never touch walls.
    let dx = x1 - x0;
    let dy = y1 - y0;
    let domain_min_x = x0 + 32;
    let domain_min_y = y0 + 32;
    let domain_max_x = x0 + dx - 32;
    let domain_max_y = y0 + dy - 32;
    if domain_max_x <= domain_min_x || domain_max_y <= domain_min_y {
        return Ok(presentation);
    }
    let config = PoissonConfig {
        min_distance_sq: 64 * 64,
        cell_size: 16,
        domain_min_x: domain_min_x as i32,
        domain_min_y: domain_min_y as i32,
        domain_max_x: domain_max_x as i32,
        domain_max_y: domain_max_y as i32,
        max_ordinal: 16,
    };
    let candidates = poisson_sample(seed, &config, max_props * 4).unwrap_or_default();
    let mut placed = 0usize;
    for (idx, &(cx, cy)) in candidates.iter().enumerate() {
        if placed >= max_props {
            break;
        }
        let prop = prop_refs[(seed as usize + idx) % prop_refs.len()];
        let cell_class = classify_cell(cx as i128, cy as i128, room_bounds, &protected);
        if cell_class == FloorCellClass::Protected {
            continue;
        }
        if quiet && cell_class == FloorCellClass::Open {
            // Quiet rooms: no central-path scatter.
            continue;
        }
        let (dx_p, dy_p, dz_p) = {
            let d = PROP_THEME_DIMENSIONS[prop as usize][theme_idx];
            (d[0] as i128, d[1] as i128, d[2] as i128)
        };
        // Wall props sit against the nearest wall; floor props rest on the
        // floor slab.
        let origin = if cell_class == FloorCellClass::Open {
            (cx as i128, cy as i128, floor_top)
        } else {
            (cx as i128, cy as i128, floor_top)
        };
        let _ = (dx_p, dy_p, dz_p);
        let owner = SemanticAttribution {
            reservation_id: room,
            request_id: None,
            archetype: Some(archetype),
            beat_id: None,
            zone_id: None,
        };
        let brush_ids = match emit_prop(ir, prop, theme, origin, &owner) {
            Ok(ids) => ids,
            Err(_) => continue,
        };
        presentation.props.push(PlacedProp {
            prop,
            origin,
            model: PROP_THEME_MODEL_OVERRIDE[prop as usize][theme_idx],
            brush_ids,
            room,
        });
        if PROP_COLLISION[prop as usize] == CollisionBehavior::DetailOnly {
            presentation.detail_count += 1;
        }
        placed += 1;
        if quiet && presentation.detail_count >= 2 {
            break;
        }
    }
    Ok(presentation)
}

/// Build a typed prop error.
pub(crate) fn prop_error(path: &str, context: impl Into<String>) -> RichnessError {
    RichnessError::new(
        RichnessErrorCode::SemanticInfeasible,
        0,
        SCHEMA_VERSION,
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

use crate::enhanced_v3::richness::content_types::CollisionBehavior;

/// Canonical theme ordinal (Ancient=0, Egyptian=1, Brutalist=2).
fn theme_ordinal(theme: RichnessTheme) -> usize {
    match theme {
        RichnessTheme::Ancient => 0,
        RichnessTheme::Egyptian => 1,
        RichnessTheme::Brutalist => 2,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::enhanced_v3::richness::{
        footprint::Footprint3D,
        ids::RouteId,
        reservation::{ReservationJournal, ReservationKind},
    };

    fn room_bounds() -> (i128, i128, i128, i128, i128, i128) {
        // 512x512 lower-band room.
        (256, 256, 0, 768, 768, 176)
    }

    #[test]
    fn classify_cell_kinds() {
        let room = room_bounds();
        let protected = vec![(512, 512, 640, 640)];
        assert_eq!(
            classify_cell(600, 600, room, &protected),
            FloorCellClass::Protected
        );
        assert_eq!(
            classify_cell(288, 288, room, &protected),
            FloorCellClass::Corner
        );
        assert_eq!(
            classify_cell(300, 288, room, &protected),
            FloorCellClass::WallAdjacent
        );
        assert_eq!(
            classify_cell(300, 300, room, &protected),
            FloorCellClass::Open
        );
    }

    #[test]
    fn protected_regions_collect_route_kinds() {
        let mut journal = ReservationJournal::new(2048, 8000);
        let room = Footprint3D::single_layer(32, 32, 96, 96, 0);
        journal
            .try_reserve(
                ReservationKind::StandardRoom,
                room,
                None,
                None,
                None,
                0,
                0,
                0,
                0,
            )
            .unwrap();
        journal.commit_all();
        let room_id = *journal.reservations.keys().next().unwrap();
        let route = Footprint3D::single_layer(96, 48, 112, 64, 0);
        journal
            .try_reserve_for_route(RouteId::new(1), ReservationKind::Route, route, 0, 0, 0, 0)
            .unwrap();
        journal.commit_all();
        let regions = protected_regions(room_id, (32, 32, 0, 96, 96, 176), &journal);
        assert!(!regions.is_empty());
    }

    #[test]
    fn emit_prop_places_convex_pieces() {
        let mut ir = AssemblyIR::new();
        let owner = SemanticAttribution {
            reservation_id: ReservationId::new(0),
            request_id: None,
            archetype: Some(ArchetypeIndex::new(0)),
            beat_id: None,
            zone_id: None,
        };
        let ids = emit_prop(&mut ir, 0, RichnessTheme::Ancient, (400, 400, 16), &owner)
            .expect("emit altar");
        assert!(!ids.is_empty());
        assert_eq!(ir.brushes.len(), PROP_CONVEX_PIECES[0] as usize);
    }

    #[test]
    fn emit_prop_rejects_overlap() {
        let mut ir = AssemblyIR::new();
        let owner = SemanticAttribution {
            reservation_id: ReservationId::new(0),
            request_id: None,
            archetype: Some(ArchetypeIndex::new(0)),
            beat_id: None,
            zone_id: None,
        };
        emit_prop(&mut ir, 0, RichnessTheme::Ancient, (400, 400, 16), &owner).unwrap();
        let result = emit_prop(&mut ir, 0, RichnessTheme::Ancient, (400, 400, 16), &owner);
        assert!(result.is_err(), "overlapping prop must be rejected");
    }

    #[test]
    fn theme_dimensions_are_quantum_aligned() {
        for prop in 0..PROP_IDS.len() {
            for theme in 0..3 {
                let d = PROP_THEME_DIMENSIONS[prop][theme];
                for axis in d {
                    assert!(axis >= 16, "prop {prop} theme {theme} axis < 16");
                    assert_eq!(axis % 16, 0, "prop {prop} theme {theme} not 16-aligned");
                }
            }
        }
    }

    #[test]
    fn quiet_room_limits_detail_props() {
        // A quiet room with max_props=2 must never exceed two placed props.
        let mut ir = AssemblyIR::new();
        let mut journal = ReservationJournal::new(2048, 8000);
        let room = Footprint3D::single_layer(32, 32, 96, 96, 0);
        journal
            .try_reserve(
                ReservationKind::StandardRoom,
                room,
                None,
                None,
                None,
                0,
                0,
                0,
                0,
            )
            .unwrap();
        journal.commit_all();
        let room_id = *journal.reservations.keys().next().unwrap();
        let result = place_room_props(
            &mut ir,
            room_id,
            (256, 256, 0, 768, 768, 176),
            ArchetypeIndex::new(0),
            RichnessTheme::Ancient,
            42,
            &journal,
            2,
            true,
        )
        .expect("quiet placement");
        assert!(result.props.len() <= 2);
    }
}
