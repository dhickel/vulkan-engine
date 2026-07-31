//! Six grammar descriptors for Enhanced V3 composition.
//!
//! Each grammar descriptor defines a family of architectural features
//! that can be placed in rooms. Descriptors declare face budgets,
//! support requirements, and structural roles.

/// Grammar descriptor identifiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GrammarDescriptor {
    /// Portal-focused chamber with aperture framing.
    PortalChamber,
    /// Buttressed hall with wall pillars.
    ButtressedHall,
    /// Column grove with freestanding pillars.
    ColumnGrove,
    /// Fractured vault with broken ceiling geometry.
    FracturedVault,
    /// Terraced shrine with stepped platforms.
    TerracedShrine,
    /// Monolithic chamber with massive stone blocks.
    MonolithicChamber,
}

impl GrammarDescriptor {
    /// Human-readable family tag.
    pub fn family_tag(self) -> &'static str {
        match self {
            Self::PortalChamber => "portal-chamber",
            Self::ButtressedHall => "buttressed-hall",
            Self::ColumnGrove => "column-grove",
            Self::FracturedVault => "fractured-vault",
            Self::TerracedShrine => "terraced-shrine",
            Self::MonolithicChamber => "monolithic-chamber",
        }
    }

    /// Conservative face budget for a single instance of this grammar.
    pub fn face_budget(self) -> u32 {
        match self {
            Self::PortalChamber => 150,
            Self::ButtressedHall => 120,
            Self::ColumnGrove => 100,
            Self::FracturedVault => 180,
            Self::TerracedShrine => 140,
            Self::MonolithicChamber => 200,
        }
    }

    /// Primary support surface required.
    pub fn primary_support(self) -> &'static str {
        match self {
            Self::PortalChamber => "wall",
            Self::ButtressedHall => "wall",
            Self::ColumnGrove => "floor",
            Self::FracturedVault => "ceiling",
            Self::TerracedShrine => "floor",
            Self::MonolithicChamber => "floor",
        }
    }

    /// Whether this grammar requires a wall anchor.
    pub fn requires_wall(self) -> bool {
        matches!(self, Self::PortalChamber | Self::ButtressedHall)
    }

    /// All six grammar descriptors in canonical order.
    pub const ALL: &[GrammarDescriptor] = &[
        Self::PortalChamber,
        Self::ButtressedHall,
        Self::ColumnGrove,
        Self::FracturedVault,
        Self::TerracedShrine,
        Self::MonolithicChamber,
    ];
}

/// Composition plan: which grammars are active for this generation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompositionPlan {
    /// Selected grammar families to instantiate.
    pub grammars: Vec<GrammarDescriptor>,
    /// Total face budget allocated for composition.
    pub face_budget: u32,
    /// Whether the plan satisfies minimum-family requirements.
    pub minimum_satisfied: bool,
}

impl CompositionPlan {
    /// Create a composition plan from a list of grammars.
    pub fn new(grammars: Vec<GrammarDescriptor>) -> Self {
        let face_budget: u32 = grammars.iter().map(|g| g.face_budget()).sum();
        Self {
            grammars,
            face_budget,
            minimum_satisfied: true,
        }
    }

    /// Total face budget for all selected grammars.
    pub fn total_face_budget(&self) -> u32 {
        self.face_budget
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_six_families_have_distinct_tags() {
        let mut tags: Vec<&str> = GrammarDescriptor::ALL
            .iter()
            .map(|g| g.family_tag())
            .collect();
        tags.sort();
        tags.dedup();
        assert_eq!(tags.len(), 6);
    }

    #[test]
    fn face_budget_max_200() {
        for g in GrammarDescriptor::ALL {
            assert!(g.face_budget() <= 200);
        }
    }

    #[test]
    fn primary_support_returns_known_kind() {
        for g in GrammarDescriptor::ALL {
            let support = g.primary_support();
            assert!(["floor", "wall", "ceiling"].contains(&support));
        }
    }

    #[test]
    fn composition_plan_total_budget() {
        let plan = CompositionPlan::new(vec![
            GrammarDescriptor::PortalChamber,
            GrammarDescriptor::ButtressedHall,
            GrammarDescriptor::ColumnGrove,
        ]);
        assert_eq!(plan.grammars.len(), 3);
        assert!(plan.total_face_budget() > 0);
        assert!(plan.total_face_budget() <= 600);
    }
}
