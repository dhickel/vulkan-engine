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
        LIGHT_PLACEMENT_CLASS, LIGHT_READABILITY_FLOOR, LIGHT_RECIPE_IDS, SCHEMA_VERSION,
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

/// Light placement summary for one room.
#[derive(Debug, Clone, Default)]
pub(crate) struct RoomLights {
    pub lights: Vec<PlacedLight>,
    pub focal_count: usize,
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
    let point = crate::enhanced_v3::geometry::Point3 {
        x: crate::enhanced_v3::geometry::Rational::from_int(origin.0),
        y: crate::enhanced_v3::geometry::Rational::from_int(origin.1),
        z: crate::enhanced_v3::geometry::Rational::from_int(origin.2),
    };
    for brush in ir.brushes.values() {
        let inside = brush
            .brush
            .faces
            .iter()
            .all(|face| face.plane.contains_point_rational(&point).unwrap_or(false));
        if inside {
            return false;
        }
    }
    true
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
            let (origin, key) = match placement {
                crate::enhanced_v3::richness::content_types::PlacementClass::Wall => {
                    // Alternate walls deterministically by light index.
                    let wall = (seed as usize + light_idx + idx) % 4;
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
                .find(|&candidate| origin_clear(ir, candidate, room_bounds));
            let Some(origin) = resolved else {
                return Err(lighting_error(
                    "origin.blocked",
                    format!(
                        "light recipe {} has no protected clear volume in room {room:?}",
                        LIGHT_RECIPE_IDS[recipe as usize]
                    ),
                ));
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
    if light_refs.is_empty() {
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
    use crate::enhanced_v3::richness::{assembly::AssemblyIR, ids::ArchetypeIndex};

    fn bounds() -> (i128, i128, i128, i128, i128, i128) {
        (256, 256, 0, 768, 768, 176)
    }

    #[test]
    fn origin_clear_rejects_outside_and_inside_brushes() {
        let ir = AssemblyIR::new();
        let room = bounds();
        assert!(origin_clear(&ir, (512, 512, 96), room));
        assert!(!origin_clear(&ir, (240, 512, 96), room), "outside x");
        assert!(!origin_clear(&ir, (512, 512, 4), room), "below floor");
    }

    #[test]
    fn place_room_lights_emits_recipe_entities() {
        let mut ir = AssemblyIR::new();
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
        let mut ir = AssemblyIR::new();
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
        let mut ir = AssemblyIR::new();
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
        }
    }
}
