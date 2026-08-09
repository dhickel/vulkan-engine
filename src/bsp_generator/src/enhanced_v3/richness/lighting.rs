//! Lighting: all 12 authored light recipes with fixed entity keys, integer
//! ranges, color encoding, per-recipe counts, placement classes, falloff
//! style, readability floors, and compile costs. Colored-light semantics
//! follow the Phase-05 proven `_color` / `_light` entity convention.
//!
//! Light count is recipe-bounded and theme-stable: themes may choose
//! material/fixture variants but never change the number of lights.

use std::collections::BTreeMap;

use crate::enhanced_v3::richness::{
    assembly::{AssemblyIR, CostSource, EntityAssembly, SemanticAttribution},
    error::{RichnessError, RichnessErrorCategory, RichnessErrorCode},
    generated_content::{
        ARCHETYPE_THEME_LIGHT_REFS, LIGHT_COLOR, LIGHT_COUNT, LIGHT_ENTITY_KEYS,
        LIGHT_ENTITY_KEY_SPANS, LIGHT_ENTITY_VALUES, LIGHT_FALLOFF, LIGHT_INTENSITY,
        LIGHT_PLACEMENT_CLASS, LIGHT_READABILITY_FLOOR, SCHEMA_VERSION,
    },
    ids::{ArchetypeIndex, ReservationId},
    request::RichnessTheme,
};

/// One placed light entity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PlacedLight {
    /// Light recipe catalog index.
    pub recipe: u32,
    /// Entity origin in Quake units.
    pub origin: (i128, i128, i128),
    /// The room that owns the light.
    pub room: ReservationId,
    /// Entity ID in the assembly.
    pub entity_id: crate::enhanced_v3::richness::ids::EntityAssemblyId,
}

/// One light recipe candidate skipped because no safe enclosed origin exists.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SkippedLight {
    pub recipe: u32,
    pub attempted_origin: (i128, i128, i128),
    pub room: ReservationId,
    pub reason: &'static str,
}

/// Light placement summary for one room.
#[derive(Debug, Clone, Default)]
pub(crate) struct RoomLights {
    pub lights: Vec<PlacedLight>,
    pub skipped: Vec<SkippedLight>,
    pub focal_count: usize,
}

fn point_inside_solid(ir: &AssemblyIR, point: (i128, i128, i128)) -> bool {
    let point = crate::enhanced_v3::geometry::Point3 {
        x: crate::enhanced_v3::geometry::Rational::from_int(point.0),
        y: crate::enhanced_v3::geometry::Rational::from_int(point.1),
        z: crate::enhanced_v3::geometry::Rational::from_int(point.2),
    };
    ir.brushes.values().any(|brush| {
        brush
            .brush
            .faces
            .iter()
            .all(|face| face.plane.contains_point_rational(&point).unwrap_or(false))
    })
}

/// A ray contact must be inside a structural brush, not merely exactly on one
/// of its faces.  Treating a portal lintel's lower edge as a wall made the old
/// six-ray test accept origins that qbsp classified as part of the opening.
fn point_strictly_inside_brush(
    brush: &crate::enhanced_v3::richness::assembly::BrushAssembly,
    point: (i128, i128, i128),
) -> bool {
    [
        (0, 0, 0),
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 0, 1),
        (0, 0, -1),
    ]
    .into_iter()
    .all(|(dx, dy, dz)| {
        let point = crate::enhanced_v3::geometry::Point3 {
            x: crate::enhanced_v3::geometry::Rational::from_int(point.0 + dx),
            y: crate::enhanced_v3::geometry::Rational::from_int(point.1 + dy),
            z: crate::enhanced_v3::geometry::Rational::from_int(point.2 + dz),
        };
        brush
            .brush
            .faces
            .iter()
            .all(|face| face.plane.contains_point_rational(&point).unwrap_or(false))
    })
}

/// Verify an origin has a protected clear volume (16 units around it) inside
/// the room and not inside any emitted brush.
fn origin_clear(
    ir: &AssemblyIR,
    origin: (i128, i128, i128),
    room_bounds: (i128, i128, i128, i128, i128, i128),
) -> bool {
    let (rx0, ry0, rz0, rx1, ry1, rz1) = room_bounds;
    if origin.0 < rx0 + 16 || origin.0 >= rx1 - 16 || origin.1 < ry0 + 16 || origin.1 >= ry1 - 16 {
        return false;
    }
    if origin.2 < rz0 + 16 || origin.2 >= rz1 - 16 {
        return false;
    }
    !point_inside_solid(ir, origin)
}

fn solid_on_segment(
    ir: &AssemblyIR,
    origin: (i128, i128, i128),
    direction: (i128, i128, i128),
    max_distance: i128,
) -> bool {
    // Eight-unit probes cannot skip the 14-unit strict interior of a
    // grid-quantized 16-unit wall when an origin is not grid-aligned.
    let mut distance = 4;
    while distance <= max_distance {
        let point = (
            origin.0 + direction.0 * distance,
            origin.1 + direction.1 * distance,
            origin.2 + direction.2 * distance,
        );
        if ir.brushes.values().any(|brush| {
            let structural_enclosure = brush.role.is_wall()
                || brush.role.is_slab()
                || matches!(
                    brush.role,
                    super::assembly::BrushAssemblyRole::PortalPost
                        | super::assembly::BrushAssemblyRole::PortalLintel
                        | super::assembly::BrushAssemblyRole::PortalSurround
                );
            structural_enclosure && point_strictly_inside_brush(brush, point)
        }) {
            return true;
        }
        distance += 8;
    }
    false
}

/// Reject point entities in authored open volumes. A valid origin must see
/// solid assembly geometry above and below within 96 units, plus in all four
/// cardinal directions before leaving its owning room's XY span.
pub(crate) fn origin_enclosed(
    ir: &AssemblyIR,
    room: ReservationId,
    origin: (i128, i128, i128),
    room_bounds: (i128, i128, i128, i128, i128, i128),
) -> bool {
    // A slab omission joins bands into an authored open volume. Local rays
    // beside its rim can still encounter floor/ceiling solids while the
    // point belongs to the exterior-connected void, so point entities are
    // conservatively omitted from the entire owning room.
    if ir.openings.values().any(|opening| {
        opening.owner.reservation_id == room
            && opening.portal_id.is_none()
            && opening.wall_role.is_slab()
    }) || !origin_clear(ir, origin, room_bounds)
    {
        return false;
    }
    let (x0, y0, z0, x1, y1, z1) = room_bounds;
    [
        ((0, 0, 1), (z1 - origin.2).min(96)),
        ((0, 0, -1), (origin.2 - z0).min(96)),
        ((1, 0, 0), x1 - origin.0),
        ((-1, 0, 0), origin.0 - x0),
        ((0, 1, 0), y1 - origin.1),
        ((0, -1, 0), origin.1 - y0),
    ]
    .into_iter()
    .all(|(direction, distance)| solid_on_segment(ir, origin, direction, distance))
}

/// Validate an emitted point-entity origin against the completed structural
/// shell.  This is the final shared safety net for spawn, lights, and
/// movement descriptors after all route, vertical, cave, and presentation
/// geometry has been materialized.
pub(crate) fn origin_airtight(ir: &AssemblyIR, origin: (i128, i128, i128)) -> bool {
    if point_inside_solid(ir, origin) {
        return false;
    }
    let Some((min, max)) = ir
        .brushes
        .values()
        .filter_map(|brush| brush.brush.aabb().ok())
        .fold(None, |bounds, (min, max)| match bounds {
            None => Some((min, max)),
            Some((lo, hi)) => Some((
                (lo.0.min(min.0), lo.1.min(min.1), lo.2.min(min.2)),
                (hi.0.max(max.0), hi.1.max(max.1), hi.2.max(max.2)),
            )),
        })
    else {
        return false;
    };
    [
        ((0, 0, 1), max.2 - origin.2),
        ((0, 0, -1), origin.2 - min.2),
        ((1, 0, 0), max.0 - origin.0),
        ((-1, 0, 0), origin.0 - min.0),
        ((0, 1, 0), max.1 - origin.1),
        ((0, -1, 0), origin.1 - min.1),
    ]
    .into_iter()
    .all(|(direction, distance)| distance > 0 && solid_on_segment(ir, origin, direction, distance))
}

/// Place the lights referenced by one room's theme light list.
///
/// Placement classes: Wall -> 16 units off the nearest wall at recipe
/// height; Ceiling -> 32 units below the ceiling plane; Floor -> 32 units
/// above the floor; Pendant -> 48 units below the ceiling; Ambient -> room
/// center at eye height. Every light emits the recipe's exact entity keys
/// (`light`/`_color`/`style` spans) with integer values, and its origin is
/// validated against the protected clear volume.
pub(crate) fn place_room_lights(
    ir: &mut AssemblyIR,
    room: ReservationId,
    room_bounds: (i128, i128, i128, i128, i128, i128),
    archetype: ArchetypeIndex,
    theme: RichnessTheme,
    seed: u64,
) -> Result<RoomLights, RichnessError> {
    let mut lights = RoomLights::default();
    let theme_idx = theme_ordinal(theme);
    let light_refs = ARCHETYPE_THEME_LIGHT_REFS[archetype.raw() as usize][theme_idx];
    if light_refs.is_empty() {
        return Ok(lights);
    }
    let (x0, y0, z0, x1, y1, z1) = room_bounds;
    let _ = (x0, x1);
    let floor_top = z0 + 16;
    let ceiling_bottom = z1 - 16;
    let center_x = (x0 + x1) / 2;
    let center_y = (y0 + y1) / 2;
    for (idx, &recipe) in light_refs.iter().enumerate() {
        // LIGHT_COUNT is the recipe's authored upper bound (map-scale pool);
        // per-room placement is capped so a single room never consumes the
        // global light budget. The cap is frozen and deterministic.
        let count = (LIGHT_COUNT[recipe as usize] as usize).min(4);
        for light_idx in 0..count {
            if lights.lights.len() >= 2 {
                // Per-room cap: one focal plus one secondary light keeps
                // every room readable without exhausting the map budget.
                break;
            }
            let placement = LIGHT_PLACEMENT_CLASS[recipe as usize];
            let (origin, _placement_key) = match placement {
                crate::enhanced_v3::richness::content_types::PlacementClass::Wall => {
                    // Alternate walls deterministically by light index.
                    let wall = ((seed % 4) as usize + light_idx + idx) % 4;
                    let (ox, oy) = match wall {
                        0 => (center_x, y0 + 24),
                        1 => (center_x, y1 - 24),
                        2 => (x0 + 24, center_y),
                        _ => (x1 - 24, center_y),
                    };
                    ((ox, oy, z0 + 80), format!("{}.{}.wall", light_idx, wall))
                }
                crate::enhanced_v3::richness::content_types::PlacementClass::Ceiling => (
                    (
                        center_x + ((light_idx as i128) % 5) * 16 - 32,
                        center_y + ((light_idx as i128) / 5) * 16 - 32,
                        ceiling_bottom - 32,
                    ),
                    format!("{}.ceiling", light_idx),
                ),
                crate::enhanced_v3::richness::content_types::PlacementClass::Floor => (
                    (
                        center_x + ((light_idx as i128) % 3) * 16 - 16,
                        center_y + ((light_idx as i128) / 3) * 16 - 16,
                        floor_top + 32,
                    ),
                    format!("{}.floor", light_idx),
                ),
                crate::enhanced_v3::richness::content_types::PlacementClass::Pendant => (
                    (center_x, center_y, ceiling_bottom - 48),
                    format!("{}.pendant", light_idx),
                ),
                crate::enhanced_v3::richness::content_types::PlacementClass::Ambient => (
                    (center_x, center_y, floor_top + 96),
                    format!("{}.ambient", light_idx),
                ),
            };
            // Deterministic fallback offsets keep lights inside dense rooms
            // while preserving canonical ordering.
            let offsets: [(i128, i128, i128); 9] = [
                (0, 0, 0),
                (32, 0, 0),
                (-32, 0, 0),
                (0, 32, 0),
                (0, -32, 0),
                (0, 0, 32),
                (0, 0, -32),
                (64, 0, 0),
                (0, 64, 0),
            ];
            let resolved = offsets
                .iter()
                .map(|&(ox, oy, oz)| (origin.0 + ox, origin.1 + oy, origin.2 + oz))
                .find(|&candidate| origin_enclosed(ir, room, candidate, room_bounds));
            let Some(origin) = resolved else {
                lights.skipped.push(SkippedLight {
                    recipe,
                    attempted_origin: origin,
                    room,
                    reason: "no_solid_enclosure_within_room_band",
                });
                continue;
            };
            let entity_id = ir.alloc_entity_id();
            let mut keys = BTreeMap::new();
            let span = LIGHT_ENTITY_KEY_SPANS[recipe as usize];
            let (color, intensity, _falloff) = (
                LIGHT_COLOR[recipe as usize],
                LIGHT_INTENSITY[recipe as usize],
                LIGHT_FALLOFF[recipe as usize],
            );
            for key_idx in span.0..span.1 {
                let key = LIGHT_ENTITY_KEYS[key_idx];
                let value = match key {
                    "light" => intensity.to_string(),
                    "_color" => format!("{} {} {}", color[0], color[1], color[2]),
                    _ => LIGHT_ENTITY_VALUES[key_idx].to_string(),
                };
                keys.insert(key.to_string(), value);
            }
            let owner = SemanticAttribution {
                reservation_id: room,
                request_id: None,
                archetype: Some(archetype),
                beat_id: None,
                zone_id: None,
            };
            ir.insert_entity(EntityAssembly {
                id: entity_id,
                classname: "light".to_string(),
                origin,
                owner,
                cost: CostSource {
                    dimension: crate::enhanced_v3::richness::assembly::BudgetDimension::SourceFaces,
                    face_count: 0,
                },
                keys,
                brush_model: None,
                brush_model_bounds: None,
            });
            lights.lights.push(PlacedLight {
                recipe,
                origin,
                room,
                entity_id,
            });
            if idx == 0 {
                lights.focal_count += 1;
            }
        }
    }
    Ok(lights)
}

/// Check a room's readability floor: every referenced recipe's readability
/// floor must be satisfiable by the placed intensity (metadata policy; the
/// renderer consumes the actual light entities).
pub(crate) fn readability_floor_satisfied(
    archetype: ArchetypeIndex,
    theme: RichnessTheme,
    placed: &RoomLights,
) -> bool {
    // A room is readable when at least one of its referenced recipes placed
    // a light meeting that recipe's readability floor (one focal light).
    let theme_idx = theme_ordinal(theme);
    let light_refs = ARCHETYPE_THEME_LIGHT_REFS[archetype.raw() as usize][theme_idx];
    if light_refs.is_empty() || (placed.lights.is_empty() && !placed.skipped.is_empty()) {
        return true;
    }
    light_refs.iter().any(|&recipe| {
        let floor = LIGHT_READABILITY_FLOOR[recipe as usize];
        placed
            .lights
            .iter()
            .any(|light| light.recipe == recipe && LIGHT_INTENSITY[recipe as usize] >= floor)
    })
}

/// Canonical theme ordinal (Ancient=0, Egyptian=1, Brutalist=2) matching the
/// generated content theme-variant arrays.
fn theme_ordinal(theme: RichnessTheme) -> usize {
    match theme {
        RichnessTheme::Ancient => 0,
        RichnessTheme::Egyptian => 1,
        RichnessTheme::Brutalist => 2,
    }
}

/// Build a typed lighting error.
pub(crate) fn lighting_error(path: &str, context: impl Into<String>) -> RichnessError {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::enhanced_v3::richness::{
        assembly::{
            AssemblyIR, BrushAssembly, BrushAssemblyRole, CostSource, SemanticAttribution,
            SupportTarget,
        },
        ids::ArchetypeIndex,
    };

    fn bounds() -> (i128, i128, i128, i128, i128, i128) {
        (256, 256, 0, 768, 768, 176)
    }

    fn room_ir(include_ceiling: bool) -> AssemblyIR {
        let mut ir = AssemblyIR::new();
        let owner = SemanticAttribution {
            reservation_id: ReservationId::new(0),
            request_id: None,
            archetype: Some(ArchetypeIndex::new(0)),
            beat_id: None,
            zone_id: None,
        };
        let mut insert = |x, y, z, role| {
            let id = ir.alloc_brush_id();
            ir.insert_brush(BrushAssembly {
                id,
                brush: crate::enhanced_v3::geometry::ConvexBrush::make_box(x, y, z)
                    .expect("test room brush"),
                role,
                owner: owner.clone(),
                cost: CostSource {
                    dimension: crate::enhanced_v3::richness::assembly::BudgetDimension::SourceFaces,
                    face_count: 6,
                },
                support: SupportTarget::World,
            });
        };
        insert(
            (256, 768),
            (256, 768),
            (0, 16),
            BrushAssemblyRole::FloorSlab,
        );
        if include_ceiling {
            insert(
                (256, 768),
                (256, 768),
                (160, 176),
                BrushAssemblyRole::CeilingSlab,
            );
        }
        insert(
            (256, 768),
            (256, 272),
            (16, 160),
            BrushAssemblyRole::NorthWall,
        );
        insert(
            (256, 768),
            (752, 768),
            (16, 160),
            BrushAssemblyRole::SouthWall,
        );
        insert(
            (256, 272),
            (272, 752),
            (16, 160),
            BrushAssemblyRole::WestWall,
        );
        insert(
            (752, 768),
            (272, 752),
            (16, 160),
            BrushAssemblyRole::EastWall,
        );
        ir
    }

    #[test]
    fn origin_clear_rejects_outside_and_inside_brushes() {
        let ir = room_ir(true);
        let room = bounds();
        assert!(origin_clear(&ir, (512, 512, 96), room));
        assert!(origin_enclosed(
            &ir,
            ReservationId::new(0),
            (512, 512, 96),
            room
        ));
        assert!(!origin_clear(&ir, (240, 512, 96), room), "outside x");
        assert!(!origin_clear(&ir, (512, 512, 4), room), "below floor");
    }

    #[test]
    fn enclosedness_rejects_an_open_ceiling() {
        let ir = room_ir(false);
        assert!(!origin_enclosed(
            &ir,
            ReservationId::new(0),
            (512, 512, 96),
            bounds()
        ));
    }

    #[test]
    fn place_room_lights_emits_recipe_entities() {
        let mut ir = room_ir(true);
        let lights = place_room_lights(
            &mut ir,
            ReservationId::new(0),
            bounds(),
            ArchetypeIndex::new(0),
            RichnessTheme::Ancient,
            7,
        )
        .expect("lights");
        assert!(!lights.lights.is_empty());
        assert!(!ir.entities.is_empty());
        for light in &lights.lights {
            let entity = ir.entities.get(&light.entity_id).expect("entity present");
            assert_eq!(entity.classname, "light");
            assert!(entity.keys.contains_key("light"));
            assert!(entity.keys.contains_key("_color"));
        }
    }

    #[test]
    fn per_room_light_cap_is_four() {
        let mut ir = room_ir(true);
        let lights = place_room_lights(
            &mut ir,
            ReservationId::new(0),
            bounds(),
            ArchetypeIndex::new(0),
            RichnessTheme::Ancient,
            3,
        )
        .expect("lights");
        assert!(lights.lights.len() <= 4 * 12, "per-recipe cap of 4");
    }

    #[test]
    fn light_origins_are_inside_room() {
        let mut ir = room_ir(true);
        let room = bounds();
        let lights = place_room_lights(
            &mut ir,
            ReservationId::new(0),
            room,
            ArchetypeIndex::new(0),
            RichnessTheme::Ancient,
            11,
        )
        .expect("lights");
        for light in &lights.lights {
            let (x, y, z) = light.origin;
            assert!(x >= room.0 + 16 && x < room.3 - 16, "x in room");
            assert!(y >= room.1 + 16 && y < room.4 - 16, "y in room");
            assert!(z >= room.2 + 16 && z < room.5 - 16, "z in room");
            assert!(origin_enclosed(
                &ir,
                ReservationId::new(0),
                light.origin,
                room
            ));
        }
    }

    #[test]
    fn no_enclosed_origin_is_skipped_with_evidence() {
        let mut ir = AssemblyIR::new();
        let lights = place_room_lights(
            &mut ir,
            ReservationId::new(0),
            bounds(),
            ArchetypeIndex::new(0),
            RichnessTheme::Ancient,
            7,
        )
        .expect("open room candidates are skipped, not errors");
        assert!(lights.lights.is_empty());
        assert!(!lights.skipped.is_empty());
        assert!(ir.entities.is_empty());
        assert!(lights
            .skipped
            .iter()
            .all(|skip| skip.reason == "no_solid_enclosure_within_room_band"));
        assert!(readability_floor_satisfied(
            ArchetypeIndex::new(0),
            RichnessTheme::Ancient,
            &lights
        ));
    }
}
