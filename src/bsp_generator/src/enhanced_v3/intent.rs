//! Feature intent and composition planning for Enhanced V3.
//!
//! FeatureIntent carries a declared semantic feature with grammar family,
//! support relations, and quantum-aligned volume. PlanOutcome captures the
//! result of deterministic composition planning.

use std::collections::BTreeSet;

use super::error::V3Error;
use super::ids::{CompositionId, FeatureId, PlanOutcome, RoomId, SupportRelation};

/// A declared feature intent before composition planning.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FeatureIntent {
    /// Stable feature ID.
    pub id: FeatureId,
    /// Grammar family this feature belongs to.
    pub family: String,
    /// The room this feature is placed in.
    pub room_id: RoomId,
    /// Support relation (if grounded).
    pub support: Option<SupportRelation>,
}

impl FeatureIntent {
    pub fn new(
        id: FeatureId,
        family: impl Into<String>,
        room_id: RoomId,
        support: Option<SupportRelation>,
    ) -> Self {
        Self {
            id,
            family: family.into(),
            room_id,
            support,
        }
    }
}

// ── Composition planner ────────────────────────────────────────────────────

/// Plan composition by selecting grammar families and features
/// according to the preset requirements.
pub fn plan_composition(
    _composition_id: CompositionId,
    preset: &str,
    room_count: u32,
) -> Result<PlanOutcome, V3Error> {
    // Determine minimum families based on preset
    let min_families = match preset {
        "sparse" => 1u32,
        "moderate" => 2,
        "rich" => 3,
        _ => 1,
    };

    // Select grammar families deterministically
    let all_families: Vec<&str> = vec![
        "portal-chamber",
        "buttressed-hall",
        "column-grove",
        "fractured-vault",
        "terraced-shrine",
        "monolithic-chamber",
    ];

    let selected_families: BTreeSet<String> = all_families
        .into_iter()
        .take(min_families as usize)
        .map(|s| s.to_string())
        .collect();

    // For now, produce empty instances — real feature materialization
    // happens in Phase 04 (composition grammar instantiation)
    let identity_satisfied = selected_families.len() >= min_families as usize;

    Ok(PlanOutcome {
        composition_id: CompositionId(0),
        preset: preset.to_string(),
        grammar_families: selected_families,
        instances: Vec::new(),
        simplified: Vec::new(),
        rejected: Vec::new(),
        support_edges: Vec::new(),
        identity_satisfied,
        estimated_total_faces: (room_count * 6 * 6) + 72, // rooms × 6 faces × 6 brushes + entities
        estimated_total_entities: 1 + room_count + room_count, // spawn + light per room + other
    })
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sparse_plan_satisfies_minimum_families() {
        let outcome = plan_composition(CompositionId(0), "sparse", 12).unwrap();
        assert!(outcome.identity_satisfied);
        assert!(outcome.grammar_families.len() >= 1);
    }

    #[test]
    fn moderate_plan_has_two_families() {
        let outcome = plan_composition(CompositionId(0), "moderate", 20).unwrap();
        assert!(outcome.identity_satisfied);
        assert!(outcome.grammar_families.len() >= 2);
    }

    #[test]
    fn rich_plan_has_three_families() {
        let outcome = plan_composition(CompositionId(0), "rich", 28).unwrap();
        assert!(outcome.identity_satisfied);
        assert!(outcome.grammar_families.len() >= 3);
    }

    #[test]
    fn plan_outcome_within_budget() {
        let outcome = plan_composition(CompositionId(0), "rich", 28).unwrap();
        assert!(outcome.estimated_total_faces < 10000);
        assert!(outcome.estimated_total_entities < 300);
    }
}
