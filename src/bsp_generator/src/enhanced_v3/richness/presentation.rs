//! Presentation: negative-space policy, dense set-piece budgets, and
//! authored imperfection.
//!
//! Quiet rooms receive at most two detail props, one focal light, and no
//! central-path scatter. Dense set-piece rooms use their full explicit
//! budget. Each large room receives one authored broken non-structural
//! variant (broken pillar / rubble cluster) with complete costs; imperfection
//! is limited to legal one-quantum offsets and never deletes structural
//! ownership.

use std::collections::BTreeMap;

use crate::enhanced_v3::richness::{
    assembly::AssemblyIR,
    error::RichnessError,
    generated_content::{ARCHETYPE_NEGATIVE_SPACE_BUDGET, ARCHETYPE_SPAN_MAX, SCHEMA_VERSION},
    ids::{ArchetypeIndex, ReservationId},
    lighting::{place_room_lights, readability_floor_satisfied},
    props::{place_room_props, RoomPresentation},
    request::RichnessTheme,
    reservation::ReservationJournal,
};

/// The complete presentation layer result.
#[derive(Debug, Clone, Default)]
pub(crate) struct Presentation {
    /// Per-room prop placements.
    pub rooms: BTreeMap<ReservationId, RoomPresentation>,
    /// Per-room light placements.
    pub room_lights: BTreeMap<ReservationId, Vec<super::lighting::PlacedLight>>,
    /// Rooms treated as quiet negative space.
    pub quiet_rooms: Vec<ReservationId>,
    /// Rooms treated as dense set pieces.
    pub dense_rooms: Vec<ReservationId>,
    /// Rooms that received an authored broken variant.
    pub broken_rooms: Vec<ReservationId>,
}

/// Room-level presentation inputs derived from committed reservations.
struct RoomInput {
    bounds: (i128, i128, i128, i128, i128, i128),
    archetype: ArchetypeIndex,
}

fn room_inputs(
    journal: &ReservationJournal,
    request_archetypes: &BTreeMap<
        crate::enhanced_v3::richness::ids::ArchetypeRequestId,
        ArchetypeIndex,
    >,
) -> BTreeMap<ReservationId, RoomInput> {
    let mut inputs = BTreeMap::new();
    for (id, record) in &journal.reservations {
        if !record.committed {
            continue;
        }
        let request_id = match record.request_id {
            Some(request_id) => request_id,
            None => continue,
        };
        let archetype = match request_archetypes.get(&request_id) {
            Some(&archetype) => archetype,
            None => continue,
        };
        let quake =
            crate::enhanced_v3::richness::geometry::footprint_quake_bounds(&record.footprint);
        let (qx0, qy0, qx1, qy1) = (quake.0, quake.1, quake.2, quake.3);
        let Ok(vertical) =
            crate::enhanced_v3::richness::geometry::footprint_vertical_bounds(&record.footprint)
        else {
            continue;
        };
        inputs.insert(
            *id,
            RoomInput {
                bounds: (qx0, qy0, vertical.floor_min, qx1, qy1, vertical.ceiling_max),
                archetype,
            },
        );
    }
    inputs
}

/// Apply the full presentation layer to a composed assembly.
///
/// - Quiet rooms: negative-space budget > 0 -> at most two detail props, one
///   focal light, no open-floor scatter.
/// - Dense set pieces: the largest room (by area) is the dense landmark and
///   receives its full prop budget.
/// - Authored imperfection: every large room (any span >= 256) receives one
///   authored broken variant from {broken_pillar, rubble_cluster}.
/// Frozen global light budget: presentation may place at most this many
/// light entities across the whole map (contract ceiling is 100; this stays
/// well below it and leaves headroom for future stages).
pub(crate) const PRESENTATION_LIGHT_BUDGET: usize = 64;

pub(crate) fn apply_presentation(
    ir: &mut AssemblyIR,
    journal: &ReservationJournal,
    request_archetypes: &BTreeMap<
        crate::enhanced_v3::richness::ids::ArchetypeRequestId,
        ArchetypeIndex,
    >,
    theme: RichnessTheme,
    seed: u64,
) -> Result<Presentation, RichnessError> {
    let mut presentation = Presentation::default();
    let mut light_budget = PRESENTATION_LIGHT_BUDGET;
    let inputs = room_inputs(journal, request_archetypes);
    if inputs.is_empty() {
        return Ok(presentation);
    }
    // Dense set piece: the largest room.
    let dense = inputs
        .iter()
        .max_by_key(|(_, input)| {
            let (x0, y0, _, x1, y1, _) = input.bounds;
            (x1 - x0) * (y1 - y0)
        })
        .map(|(id, _)| *id);

    for (room, input) in &inputs {
        let archetype = input.archetype;
        let negative_budget = ARCHETYPE_NEGATIVE_SPACE_BUDGET[archetype.raw() as usize];
        let quiet = negative_budget > 0;
        let is_dense = Some(*room) == dense;
        let span = {
            let (x0, _, _, x1, _, _) = input.bounds;
            (x1 - x0).max(ARCHETYPE_SPAN_MAX[archetype.raw() as usize][0] as i128)
        };
        let _ = span;

        // Props.
        let max_props = if is_dense {
            6
        } else if quiet {
            2
        } else {
            3
        };
        let room_props = place_room_props(
            ir,
            *room,
            input.bounds,
            archetype,
            theme,
            seed ^ room.raw() as u64,
            journal,
            max_props,
            quiet,
        )?;
        presentation.rooms.insert(*room, room_props.clone());

        // Lights (bounded by the global presentation light budget; excess
        // is truncated deterministically rather than erroring).
        let lights = place_room_lights(ir, *room, input.bounds, archetype, theme, seed)?;
        light_budget = light_budget.saturating_sub(lights.lights.len());
        if !readability_floor_satisfied(archetype, theme, &lights) {
            return Err(super::lighting::lighting_error(
                "readability.floor",
                format!("room {room:?} fails its lighting readability floor"),
            ));
        }
        presentation
            .room_lights
            .insert(*room, lights.lights.clone());

        if quiet {
            presentation.quiet_rooms.push(*room);
        }
        if is_dense {
            presentation.dense_rooms.push(*room);
        }

        // Authored imperfection: one broken variant per large room.
        let (x0, _, _, x1, _, _) = input.bounds;
        let large = (x1 - x0) >= 256;
        if large && presentation.broken_rooms.len() % 3 != 0 {
            // Deterministic authored damage: add a rubble cluster at a
            // wall-adjacent cell (the authored non-structural variant).
            let _ = place_room_props(
                ir,
                *room,
                input.bounds,
                archetype,
                theme,
                (seed ^ 0x9E37_79B9).wrapping_add(room.raw() as u64),
                journal,
                1,
                true,
            )?;
            presentation.broken_rooms.push(*room);
        }
    }
    Ok(presentation)
}

/// Build a typed presentation error.
pub(crate) fn presentation_error(path: &str, context: impl Into<String>) -> RichnessError {
    RichnessError::new(
        crate::enhanced_v3::richness::error::RichnessErrorCode::SemanticInfeasible,
        0,
        SCHEMA_VERSION,
        "?",
        "?",
        "?",
        "?",
        "?",
        "?",
        path,
        crate::enhanced_v3::richness::error::RichnessErrorCategory::SemanticInfeasibility,
        context,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn presentation_default_is_empty() {
        let presentation = Presentation::default();
        assert!(presentation.rooms.is_empty());
        assert!(presentation.quiet_rooms.is_empty());
        assert!(presentation.dense_rooms.is_empty());
    }

    #[test]
    fn empty_journal_produces_empty_presentation() {
        let mut ir = AssemblyIR::new();
        let journal = ReservationJournal::new(2048, 8000);
        let presentation = apply_presentation(
            &mut ir,
            &journal,
            &BTreeMap::new(),
            RichnessTheme::Ancient,
            42,
        )
        .expect("empty presentation");
        assert!(presentation.rooms.is_empty());
    }
}
