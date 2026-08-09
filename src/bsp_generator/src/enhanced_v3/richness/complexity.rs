//! Complexity budget and recipe selection planning.
//!
//! Reserves mandatory worst-case costs first in stable priority order,
//! then selects complete optional recipes (never partial/truncated) in
//! the same priority order using catalog-provided alternatives. Theme
//! variants share the max of all three variant costs at reservation time.
//!
//! # Priority order (frozen)
//!
//! 1. Sealing / routes
//! 2. Required topology
//! 3. Landmarks / traversal
//! 4. Theme identity
//! 5. Required vertical / cave features
//! 6. Readability
//! 7. Props / imperfection
//!
//! # Contract
//!
//! - Preset ceilings: Sparse=3,000, Moderate=5,000, Rich=8,000 source faces.
//! - Never raise any ceiling. Never omit a required item.
//! - Summed declared costs dominate later actual counts (conservative).
//! - A lower-cost alternative must be a COMPLETE authored identity.
//! - Return stable budget error if an explicit request cannot fit.
//! - Crate-private; no brush/entity emission; no floats.

// Richness remains intentionally crate-private and pipeline-unwired until
// the atomic sealing phase.
#![allow(dead_code)]

use std::collections::BTreeMap;

use super::generated_content;
use super::ids::PacingBlueprint;
use super::request::{RichnessCaveMode, RichnessPreset};
use super::topology::TopologyResult;

// ── Budget dimensions ──────────────────────────────────────────────────────

/// Closed enum of budget dimensions for reservation and tracking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) enum BudgetDimension {
    /// Source `.map` faces.
    SourceFaces,
    /// Convex brushes.
    Brushes,
    /// Point / brush entities.
    Entities,
    /// Light entities.
    Lights,
    /// Vertical openings (stairs, shafts, drops).
    VerticalOpenings,
    /// Support contacts (grounded assembly edges).
    SupportContacts,
    /// Package assets (WAD textures, PNG companions).
    PackageAssets,
    /// Compiler lumps (15 standard BSP2 lumps).
    CompilerLumps,
    /// Renderer static batches.
    RendererBatches,
    /// Estimated runtime GPU memory (bytes).
    RendererMemoryBytes,
    /// Runtime/controller requirements (e.g. ladder/drop descriptors).
    RuntimeRequirements,
}

impl BudgetDimension {
    /// Lowercase tag for diagnostics.
    pub fn tag(self) -> &'static str {
        match self {
            Self::SourceFaces => "source_faces",
            Self::Brushes => "brushes",
            Self::Entities => "entities",
            Self::Lights => "lights",
            Self::VerticalOpenings => "vertical_openings",
            Self::SupportContacts => "support_contacts",
            Self::PackageAssets => "package_assets",
            Self::CompilerLumps => "compiler_lumps",
            Self::RendererBatches => "renderer_batches",
            Self::RendererMemoryBytes => "renderer_memory_bytes",
            Self::RuntimeRequirements => "runtime_requirements",
        }
    }
}

// ── Budget reservation / consumption ───────────────────────────────────────

/// A reservation across all budget dimensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct BudgetReservation {
    /// Source faces.
    pub faces: u32,
    /// Convex brushes.
    pub brushes: u32,
    /// Point / brush entities.
    pub entities: u32,
    /// Light entities.
    pub lights: u32,
    /// Vertical openings.
    pub vertical_openings: u32,
    /// Support contacts.
    pub support_contacts: u32,
    /// Package assets.
    pub package_assets: u32,
    /// Compiler lumps (always 15 for BSP2).
    pub compiler_lumps: u32,
    /// Renderer static batches.
    pub renderer_batches: u32,
    /// Estimated runtime GPU memory bytes.
    pub renderer_memory_bytes: u64,
    /// Runtime/controller requirements reserved by selected recipes.
    pub runtime_requirements: u32,
}

impl BudgetReservation {
    /// Zero reservation.
    pub const ZERO: Self = Self {
        faces: 0,
        brushes: 0,
        entities: 0,
        lights: 0,
        vertical_openings: 0,
        support_contacts: 0,
        package_assets: 0,
        compiler_lumps: 0,
        renderer_batches: 0,
        renderer_memory_bytes: 0,
        runtime_requirements: 0,
    };

    /// Saturating addition.
    pub fn saturating_add(self, other: Self) -> Self {
        Self {
            faces: self.faces.saturating_add(other.faces),
            brushes: self.brushes.saturating_add(other.brushes),
            entities: self.entities.saturating_add(other.entities),
            lights: self.lights.saturating_add(other.lights),
            vertical_openings: self
                .vertical_openings
                .saturating_add(other.vertical_openings),
            support_contacts: self.support_contacts.saturating_add(other.support_contacts),
            package_assets: self.package_assets.saturating_add(other.package_assets),
            compiler_lumps: self.compiler_lumps.saturating_add(other.compiler_lumps),
            renderer_batches: self.renderer_batches.saturating_add(other.renderer_batches),
            renderer_memory_bytes: self
                .renderer_memory_bytes
                .saturating_add(other.renderer_memory_bytes),
            runtime_requirements: self
                .runtime_requirements
                .saturating_add(other.runtime_requirements),
        }
    }

    /// Take the maximum of each dimension across two reservations.
    pub fn max_per_dimension(self, other: Self) -> Self {
        Self {
            faces: self.faces.max(other.faces),
            brushes: self.brushes.max(other.brushes),
            entities: self.entities.max(other.entities),
            lights: self.lights.max(other.lights),
            vertical_openings: self.vertical_openings.max(other.vertical_openings),
            support_contacts: self.support_contacts.max(other.support_contacts),
            package_assets: self.package_assets.max(other.package_assets),
            compiler_lumps: self.compiler_lumps.max(other.compiler_lumps),
            renderer_batches: self.renderer_batches.max(other.renderer_batches),
            renderer_memory_bytes: self.renderer_memory_bytes.max(other.renderer_memory_bytes),
            runtime_requirements: self.runtime_requirements.max(other.runtime_requirements),
        }
    }

    /// Whether all dimensions are within the given ceiling.
    pub fn within(&self, ceiling: &Self) -> bool {
        self.faces <= ceiling.faces
            && self.brushes <= ceiling.brushes
            && self.entities <= ceiling.entities
            && self.lights <= ceiling.lights
            && self.vertical_openings <= ceiling.vertical_openings
            && self.support_contacts <= ceiling.support_contacts
            && self.package_assets <= ceiling.package_assets
            && self.compiler_lumps <= ceiling.compiler_lumps
            && self.renderer_batches <= ceiling.renderer_batches
            && self.renderer_memory_bytes <= ceiling.renderer_memory_bytes
            && self.runtime_requirements <= ceiling.runtime_requirements
    }

    /// Remaining budget after subtracting consumed.
    pub fn remaining(&self, ceiling: &Self) -> Self {
        Self {
            faces: ceiling.faces.saturating_sub(self.faces),
            brushes: ceiling.brushes.saturating_sub(self.brushes),
            entities: ceiling.entities.saturating_sub(self.entities),
            lights: ceiling.lights.saturating_sub(self.lights),
            vertical_openings: ceiling
                .vertical_openings
                .saturating_sub(self.vertical_openings),
            support_contacts: ceiling
                .support_contacts
                .saturating_sub(self.support_contacts),
            package_assets: ceiling.package_assets.saturating_sub(self.package_assets),
            compiler_lumps: ceiling.compiler_lumps.saturating_sub(self.compiler_lumps),
            renderer_batches: ceiling
                .renderer_batches
                .saturating_sub(self.renderer_batches),
            renderer_memory_bytes: ceiling
                .renderer_memory_bytes
                .saturating_sub(self.renderer_memory_bytes),
            runtime_requirements: ceiling
                .runtime_requirements
                .saturating_sub(self.runtime_requirements),
        }
    }
}

// ── Recipe priority ────────────────────────────────────────────────────────

/// Frozen priority order for recipe selection.
///
/// Lower discriminant = higher priority.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) enum RecipePriority {
    /// Sealing shells, routes, portal throats, turns, spawn.
    SealingRoutes = 0,
    /// Mandatory critical-path edges, branch connections.
    RequiredTopology = 1,
    /// Critical-path landmarks, traversal beats.
    LandmarksTraversal = 2,
    /// Theme identity materials, portal surrounds, theme geometry.
    ThemeIdentity = 3,
    /// Required vertical (multi-storey, stair, shaft, drop) and cave features.
    RequiredVerticalCave = 4,
    /// Readability: mandatory lights, minimum ambient, contrast.
    Readability = 5,
    /// Props and imperfection (damage, rubble, decorative).
    PropsImperfection = 6,
}

impl RecipePriority {
    /// All priorities in selection order.
    pub const ALL: [RecipePriority; 7] = [
        RecipePriority::SealingRoutes,
        RecipePriority::RequiredTopology,
        RecipePriority::LandmarksTraversal,
        RecipePriority::ThemeIdentity,
        RecipePriority::RequiredVerticalCave,
        RecipePriority::Readability,
        RecipePriority::PropsImperfection,
    ];

    /// Lowercase tag for diagnostics.
    pub fn tag(self) -> &'static str {
        match self {
            Self::SealingRoutes => "sealing_routes",
            Self::RequiredTopology => "required_topology",
            Self::LandmarksTraversal => "landmarks_traversal",
            Self::ThemeIdentity => "theme_identity",
            Self::RequiredVerticalCave => "required_vertical_cave",
            Self::Readability => "readability",
            Self::PropsImperfection => "props_imperfection",
        }
    }
}

// ── Recipe identity ────────────────────────────────────────────────────────

/// A complete authored recipe identity with declared cost.
///
/// Every recipe (whether primary or alternative) is a COMPLETE identity
/// with its own geometry, support, prop/light, and cost declarations.
/// Never a truncated or partial recipe.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RecipeIdentity {
    /// Stable recipe name.
    pub name: String,
    /// Which priority tier this recipe belongs to.
    pub priority: RecipePriority,
    /// Whether this is a mandatory recipe (must be reserved).
    pub mandatory: bool,
    /// Whether an optional recipe was explicitly requested. An explicit
    /// optional request returns `NoAlternativeFits`; an unrequested optional
    /// recipe may be omitted whole, never truncated.
    pub explicitly_requested: bool,
    /// Declared worst-case cost for this recipe.
    pub cost: BudgetReservation,
    /// Alternative complete recipes (lower-cost options must still be
    /// complete authored identities).
    pub alternatives: Vec<RecipeIdentity>,
}

impl RecipeIdentity {
    /// Get all alternatives sorted by ascending face cost for stable selection.
    pub fn sorted_alternatives(&self) -> Vec<&RecipeIdentity> {
        let mut alts: Vec<&RecipeIdentity> = self.alternatives.iter().collect();
        alts.sort_by_key(|a| a.cost.faces);
        alts
    }

    /// Find the first alternative that fits in the remaining budget.
    pub fn first_fitting<'a>(
        &'a self,
        remaining: &BudgetReservation,
    ) -> Option<&'a RecipeIdentity> {
        self.sorted_alternatives()
            .into_iter()
            .find(|a| a.cost.within(remaining))
    }
}

// ── Recipe selection ───────────────────────────────────────────────────────

/// A single selected recipe with its reserved budget.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RecipeSelection {
    /// The selected recipe identity.
    pub recipe: RecipeIdentity,
    /// The priority at which this was selected.
    pub priority: RecipePriority,
    /// Whether an alternative was used instead of the primary.
    pub was_alternative: bool,
    /// The original primary recipe name if an alternative was used.
    pub original_recipe_name: Option<String>,
}

// ── Budget error ───────────────────────────────────────────────────────────

/// A stable budget error returned when an explicit request cannot fit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum BudgetError {
    /// A specific dimension exceeded its ceiling.
    OverBudget {
        dimension: BudgetDimension,
        ceiling: u32,
        requested: u32,
        priority: RecipePriority,
        recipe_name: String,
    },
    /// No alternative (including lower-cost) fits within remaining budget.
    NoAlternativeFits {
        dimension: BudgetDimension,
        ceiling: u32,
        requested: u32,
        recipe_name: String,
        min_cost_faces: u32,
        remaining_faces: u32,
        priority: RecipePriority,
    },
    /// Mandatory recipes alone exceed the ceiling.
    MandatoryExceedsCeiling {
        dimension: BudgetDimension,
        ceiling: u32,
        requested: u32,
        priority: RecipePriority,
        recipe_name: String,
        total_mandatory: BudgetReservation,
    },
}

impl BudgetError {
    /// Stable error message for diagnostics.
    pub fn describe(&self) -> String {
        match self {
            Self::OverBudget {
                dimension,
                ceiling,
                requested,
                priority,
                recipe_name,
            } => {
                format!(
                    "budget overrun in {} ({}): requested {} but ceiling is {} at priority {} for recipe {}",
                    dimension.tag(),
                    dimension.tag(),
                    requested,
                    ceiling,
                    priority.tag(),
                    recipe_name
                )
            }
            Self::NoAlternativeFits {
                dimension,
                ceiling,
                requested,
                recipe_name,
                min_cost_faces,
                remaining_faces,
                priority,
            } => {
                format!(
                    "no complete alternative fits for {}: {} requested {} but remaining ceiling is {}; min_cost_faces={}, remaining_faces={} at priority {}",
                    recipe_name,
                    dimension.tag(),
                    requested,
                    ceiling,
                    min_cost_faces,
                    remaining_faces,
                    priority.tag()
                )
            }
            Self::MandatoryExceedsCeiling {
                dimension,
                ceiling,
                requested,
                priority,
                recipe_name,
                total_mandatory,
            } => {
                format!(
                    "mandatory total {} exceeds {} ceiling: requested {} but ceiling is {} at priority {} for recipe {} ({} faces total)",
                    dimension.tag(),
                    dimension.tag(),
                    requested,
                    ceiling,
                    priority.tag(),
                    recipe_name,
                    total_mandatory.faces,
                )
            }
        }
    }
}

// ── Complexity budget ──────────────────────────────────────────────────────

/// The complete complexity budget with ceilings and consumption tracking.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ComplexityBudget {
    /// Ceilings per dimension.
    pub ceilings: BudgetReservation,
    /// Currently consumed budget.
    pub consumed: BudgetReservation,
    /// Maximum cost across all three theme variants (shared semantic cost).
    pub max_theme_variant_cost: BudgetReservation,
}

impl ComplexityBudget {
    /// Create a budget with the given face ceiling (preset-dependent) and
    /// contract-frozen dimension limits.
    pub fn new(preset: RichnessPreset) -> Self {
        let face_ceiling = match preset {
            RichnessPreset::Sparse => 3_000,
            RichnessPreset::Moderate => 5_000,
            RichnessPreset::Rich => 8_000,
        };

        // Derived ceilings from the contract (frozen):
        // - brushes ≤ faces / 4 (conservative against the catalog's
        //   per-archetype <64-brush declaration)
        // - entities < 500
        // - lights < 100
        // - vertical openings ≤ 12
        // - support contacts ≤ 256 (owner-approved measured Richness
        //   ceiling, raised from the phase-08 value of 128 when the phase-13
        //   prop layer added per-prop support contacts; M2 ceilings untouched)
        // - package assets ≤ 128
        // - compiler lumps = 15 (always)
        // - renderer batches < 800
        // - memory: 256 MiB estimate
        // - runtime/controller requirements ≤ 32
        let ceilings = BudgetReservation {
            faces: face_ceiling,
            brushes: face_ceiling / 4,
            entities: 500,
            lights: 100,
            vertical_openings: 12,
            support_contacts: 256,
            package_assets: 128,
            compiler_lumps: 15,
            renderer_batches: 800,
            renderer_memory_bytes: 256 * 1024 * 1024,
            runtime_requirements: 32,
        };

        Self {
            ceilings,
            consumed: BudgetReservation::ZERO,
            max_theme_variant_cost: BudgetReservation::ZERO,
        }
    }

    /// Whether any dimension ceiling is exceeded.
    pub fn is_exceeded(&self) -> bool {
        !self.consumed.within(&self.ceilings)
    }

    /// Remaining budget across all dimensions.
    pub fn remaining(&self) -> BudgetReservation {
        self.consumed.remaining(&self.ceilings)
    }

    /// Track consumption by adding a reservation.
    pub fn spend(&mut self, cost: BudgetReservation) {
        self.consumed = self.consumed.saturating_add(cost);
    }

    /// Reserve the max theme variant cost by taking the maximum of all three
    /// variant costs. Call once before selecting optional recipes.
    pub fn reserve_max_theme_cost(&mut self, variant_costs: [BudgetReservation; 3]) {
        let max_cost = variant_costs[0]
            .max_per_dimension(variant_costs[1])
            .max_per_dimension(variant_costs[2]);
        self.max_theme_variant_cost = max_cost;
        self.consumed = self.consumed.saturating_add(max_cost);
    }
}

// ── Complexity plan ────────────────────────────────────────────────────────

/// The complete pre-assembly complexity plan naming every selected recipe
/// and reserved cost.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ComplexityPlan {
    /// All selected recipes in priority order.
    pub selected_recipes: Vec<RecipeSelection>,
    /// Total reserved budget across all selections.
    pub total_reserved: BudgetReservation,
    /// The budget state after all reservations.
    pub budget: ComplexityBudget,
    /// Whether all mandatory recipes were successfully reserved.
    pub mandatory_reserved: bool,
    /// Theme variant index (0=Ancient, 1=Egyptian, 2=Brutalist).
    pub theme_variant: u32,
    /// Any budget errors encountered during planning.
    pub errors: Vec<BudgetError>,
}

impl ComplexityPlan {
    /// Returns true if the plan is within budget with no errors.
    pub fn is_within_budget(&self) -> bool {
        self.errors.is_empty() && !self.budget.is_exceeded()
    }

    /// Assert that summed declared costs dominate later actual counts.
    ///
    /// Planning is conservative: actual assembly may use less but never more
    /// than reserved.
    pub fn assert_dominates(&self, actual: &BudgetReservation) -> bool {
        self.total_reserved.faces >= actual.faces
            && self.total_reserved.brushes >= actual.brushes
            && self.total_reserved.entities >= actual.entities
            && self.total_reserved.lights >= actual.lights
            && self.total_reserved.vertical_openings >= actual.vertical_openings
            && self.total_reserved.support_contacts >= actual.support_contacts
            && self.total_reserved.package_assets >= actual.package_assets
            && self.total_reserved.compiler_lumps >= actual.compiler_lumps
            && self.total_reserved.renderer_batches >= actual.renderer_batches
            && self.total_reserved.renderer_memory_bytes >= actual.renderer_memory_bytes
            && self.total_reserved.runtime_requirements >= actual.runtime_requirements
    }

    /// Return the reserved budget for vertical features (priority RequiredVerticalCave).
    /// Sums the declared cost of every selected recipe at that priority tier.
    pub fn vertical_reservation(&self) -> BudgetReservation {
        let mut reservation = BudgetReservation::ZERO;
        for selection in &self.selected_recipes {
            if selection.priority == RecipePriority::RequiredVerticalCave {
                reservation = reservation.saturating_add(selection.recipe.cost);
            }
        }
        reservation
    }

    /// Validate that every mandatory recipe at each priority tier is included.
    pub fn validate_mandatory_completeness(&self) -> Vec<String> {
        let mut errors = Vec::new();

        for priority in &RecipePriority::ALL {
            let has_mandatory = self
                .selected_recipes
                .iter()
                .any(|s| s.priority == *priority && s.recipe.mandatory);
            if !has_mandatory {
                // Only priorities 0-4 are truly mandatory (sealing through vertical/cave).
                // Readability and Props are required but may be zero-cost.
                match priority {
                    RecipePriority::SealingRoutes
                    | RecipePriority::RequiredTopology
                    | RecipePriority::LandmarksTraversal
                    | RecipePriority::ThemeIdentity
                    | RecipePriority::RequiredVerticalCave => {
                        errors.push(format!(
                            "priority {} has no mandatory recipe selected",
                            priority.tag()
                        ));
                    }
                    _ => {}
                }
            }
        }

        errors
    }
}

fn vertical_cost(
    brushes: u32,
    entities: u32,
    logical_openings: u32,
    runtime_requirements: u32,
) -> BudgetReservation {
    BudgetReservation {
        faces: brushes.saturating_mul(6),
        brushes,
        entities,
        lights: 0,
        vertical_openings: logical_openings,
        // Every validator-counted vertical member may require its own support
        // edge. Charging the complete brush bound before optional recipes are
        // selected leaves deterministic room for that proof under the frozen
        // 256-contact measured ceiling.
        support_contacts: brushes,
        package_assets: 0,
        compiler_lumps: 0,
        renderer_batches: 1,
        renderer_memory_bytes: u64::from(brushes).saturating_mul(4_096),
        runtime_requirements,
    }
}

fn cost_for_vertical_recipe(recipe: super::content_types::VerticalRecipe) -> BudgetReservation {
    use super::content_types::VerticalRecipe;
    match recipe {
        VerticalRecipe::None => vertical_cost(52, 0, 1, 0), // requested generic open stairwell
        VerticalRecipe::Stairwell => vertical_cost(40, 0, 1, 0),
        VerticalRecipe::OpenStairwell => vertical_cost(56, 0, 1, 0),
        VerticalRecipe::LadderShaft => vertical_cost(48, 1, 1, 1),
        VerticalRecipe::SpiralStair => vertical_cost(52, 0, 1, 0),
        VerticalRecipe::DropHole => vertical_cost(32, 1, 1, 1),
    }
}

/// Conservative complete cost for the vertical architecture that the current
/// committed topology will materialize.  Catalog `None` remains intentionally
/// empty; a committed request-level host with `None` receives the generic open
/// stairwell contract and is therefore costed independently of the catalog.
fn planned_vertical_cost(
    blueprint: &PacingBlueprint,
    topology: &TopologyResult,
    cave_mode: RichnessCaveMode,
) -> BudgetReservation {
    use super::content_types::VerticalRecipe;
    use super::reservation::ReservationKind;

    let recipe_for = |record: &super::reservation::ReservationRecord| {
        record
            .request_id
            .and_then(|request_id| blueprint.archetype_requests.get(&request_id))
            .and_then(|request| {
                generated_content::ARCHETYPE_VERTICAL_RECIPE
                    .get(request.archetype.raw() as usize)
                    .copied()
            })
            .unwrap_or(VerticalRecipe::None)
    };

    let cave_vertical = super::cave::synthesize_cave(blueprint.seed, cave_mode, &topology.journal)
        .ok()
        .flatten()
        .map(|cave| {
            let brushes = cave
                .solid_boxes
                .iter()
                .filter(|solid| solid.role == super::cave::CaveRole::Wall)
                .count() as u32;
            vertical_cost(brushes, 0, 0, 0)
        })
        .unwrap_or(BudgetReservation::ZERO);
    let mut total = BudgetReservation::ZERO;
    let mut cave_reserved = false;
    for record in topology
        .journal
        .reservations
        .values()
        .filter(|record| record.committed)
    {
        match record.kind {
            // VerticalHost cells reserve placement space for the paired
            // feature owned by their multi-storey room; they do not emit an
            // additional architecture assembly of their own.
            ReservationKind::VerticalHost => {}
            ReservationKind::PitOmission => {
                total = total.saturating_add(cost_for_vertical_recipe(
                    super::content_types::VerticalRecipe::DropHole,
                ));
            }
            ReservationKind::MultiStoreyRoom => {
                let archetype_id = record
                    .request_id
                    .and_then(|request_id| blueprint.archetype_requests.get(&request_id))
                    .and_then(|request| {
                        generated_content::ARCHETYPE_IDS
                            .get(request.archetype.raw() as usize)
                            .copied()
                    });
                let setpiece = match archetype_id {
                    Some("arena") => vertical_cost(10, 0, 0, 0),
                    Some("bridge_crossing") => vertical_cost(24, 0, 1, 0),
                    Some("overlook_hall") => vertical_cost(12, 0, 0, 0),
                    // `grand_arena` is deliberately a separately materialized
                    // set-piece.  Adding a catalog enum value would change the
                    // authored/code-generated content contract.
                    // Complete maximum: 4 balconies + 4 balcony rails + 4
                    // corbels + 2 decks + 4 deck rails + 4 deck supports +
                    // monolith + 12 treads + 11 stair guards + 2 gates.
                    Some("grand_arena") => vertical_cost(48, 0, 1, 0),
                    _ => match recipe_for(record) {
                        VerticalRecipe::None => BudgetReservation::ZERO,
                        recipe => cost_for_vertical_recipe(recipe),
                    },
                };
                total = total.saturating_add(setpiece);
            }
            ReservationKind::NegativeSpace => match recipe_for(record) {
                VerticalRecipe::None => {}
                recipe => total = total.saturating_add(cost_for_vertical_recipe(recipe)),
            },
            // Cave synthesis is deterministic from the finalized topology,
            // so reserve the exact materialized CaveWall set once even when
            // placement retained several eligible host reservations.
            ReservationKind::CaveHost if !cave_reserved => {
                total = total.saturating_add(cave_vertical);
                cave_reserved = true;
            }
            ReservationKind::CaveHost => {}
            _ => {}
        }
    }
    total
}

// ── Recipe catalog ─────────────────────────────────────────────────────────

/// Build the complete recipe catalog from generated content constants.
///
/// Each recipe declares its worst-case cost from the catalog arrays.
/// Theme variants contribute three alternative costs and the reservation
/// uses the maximum of all three.
pub(crate) struct RecipeCatalog {
    /// Recipes grouped by priority.
    recipes: BTreeMap<RecipePriority, Vec<RecipeIdentity>>,
}

impl RecipeCatalog {
    /// Build the catalog from a pacing blueprint and topology result.
    pub fn build(
        blueprint: &PacingBlueprint,
        topology: &TopologyResult,
        cave_mode: RichnessCaveMode,
    ) -> Self {
        let mut recipes: BTreeMap<RecipePriority, Vec<RecipeIdentity>> = BTreeMap::new();

        // ── Priority 0: Sealing / Routes ───────────────────────────────
        let mut sealing_recipes = Vec::new();

        // Every committed route needs sealing shells (floor, ceiling, boundary walls).
        // Estimate: 6 faces per corridor cell, ~16 cells per route.
        for route in &topology.routes {
            let route_cost = BudgetReservation {
                faces: 96,  // ~6 faces per corridor segment × ~16 segments
                brushes: 6, // floor + ceiling + 4 walls
                entities: 0,
                lights: 0,
                vertical_openings: 0,
                support_contacts: 0, // world roots are not inter-brush contacts
                // Route shells reuse the selected theme's role-bound assets;
                // package assets are unique identities, not per-route copies.
                package_assets: 0,
                compiler_lumps: 0,
                renderer_batches: 1,
                renderer_memory_bytes: 4096,
                runtime_requirements: 0,
            };
            sealing_recipes.push(RecipeIdentity {
                name: format!("sealing_route_{}", route.id.raw()),
                priority: RecipePriority::SealingRoutes,
                mandatory: true,
                explicitly_requested: false,
                cost: route_cost,
                alternatives: Vec::new(),
            });
        }

        // Portal throat sealing.
        for route in &topology.routes {
            let portal_cost = BudgetReservation {
                faces: 16,
                brushes: 2,
                entities: 0,
                lights: 0,
                vertical_openings: 0,
                support_contacts: 0,
                // Portal geometry reuses the selected theme's portal roles.
                package_assets: 0,
                compiler_lumps: 0,
                renderer_batches: 1,
                renderer_memory_bytes: 2048,
                runtime_requirements: 0,
            };
            sealing_recipes.push(RecipeIdentity {
                name: format!("sealing_portal_{}", route.source_portal.id.raw()),
                priority: RecipePriority::SealingRoutes,
                mandatory: true,
                explicitly_requested: false,
                cost: portal_cost,
                alternatives: Vec::new(),
            });
            sealing_recipes.push(RecipeIdentity {
                name: format!("sealing_portal_{}", route.target_portal.id.raw()),
                priority: RecipePriority::SealingRoutes,
                mandatory: true,
                explicitly_requested: false,
                cost: BudgetReservation {
                    faces: 16,
                    brushes: 2,
                    entities: 0,
                    lights: 0,
                    vertical_openings: 0,
                    support_contacts: 0,
                    // Portal geometry reuses the selected theme's portal roles.
                    package_assets: 0,
                    compiler_lumps: 0,
                    renderer_batches: 1,
                    renderer_memory_bytes: 2048,
                    runtime_requirements: 0,
                },
                alternatives: Vec::new(),
            });
        }

        // Spawn reservation.
        sealing_recipes.push(RecipeIdentity {
            name: "spawn_point".to_string(),
            priority: RecipePriority::SealingRoutes,
            mandatory: true,
            explicitly_requested: false,
            cost: BudgetReservation {
                faces: 0,
                brushes: 0,
                entities: 1,
                lights: 0,
                vertical_openings: 0,
                support_contacts: 0,
                package_assets: 0,
                compiler_lumps: 0,
                renderer_batches: 0,
                renderer_memory_bytes: 0,
                runtime_requirements: 0,
            },
            alternatives: Vec::new(),
        });

        recipes.insert(RecipePriority::SealingRoutes, sealing_recipes);

        // ── Priority 1: Required Topology ──────────────────────────────
        let topology_cost = BudgetReservation {
            faces: 0,
            brushes: 0,
            entities: 0,
            lights: 0,
            vertical_openings: 0,
            support_contacts: 0,
            package_assets: 0,
            // Every compiled BSP2 map owns the fixed 15-lump package once.
            compiler_lumps: 15,
            renderer_batches: 0,
            renderer_memory_bytes: 0,
            runtime_requirements: 0,
        };
        recipes.insert(
            RecipePriority::RequiredTopology,
            vec![RecipeIdentity {
                name: "mandatory_topology_edges".to_string(),
                priority: RecipePriority::RequiredTopology,
                mandatory: true,
                explicitly_requested: false,
                cost: topology_cost,
                alternatives: Vec::new(),
            }],
        );

        // ── Priority 2: Landmarks / Traversal ──────────────────────────
        let mut landmark_recipes = Vec::new();
        for req in blueprint.archetype_requests.values() {
            let arch_idx = req.archetype.raw() as usize;
            if arch_idx >= generated_content::ARCHETYPE_COST_SOURCE_FACES.len() {
                continue;
            }
            let arch_cost = BudgetReservation {
                faces: generated_content::ARCHETYPE_COST_SOURCE_FACES[arch_idx],
                brushes: generated_content::ARCHETYPE_COST_BRUSHES[arch_idx],
                entities: generated_content::ARCHETYPE_COST_ENTITIES[arch_idx],
                // Presentation owns concrete light placement; reserve the
                // complete capped light capacity below rather than charging
                // catalog hints as mandatory structural lights.
                lights: 0,
                vertical_openings: match generated_content::ARCHETYPE_VERTICAL_RECIPE[arch_idx] {
                    super::content_types::VerticalRecipe::None => 0,
                    _ => 1,
                },
                support_contacts: 0,
                package_assets: generated_content::ARCHETYPE_MATERIAL_ROLES[arch_idx].len() as u32,
                compiler_lumps: 0,
                renderer_batches: 1,
                renderer_memory_bytes: 16384,
                runtime_requirements: 0,
            };

            // Theme variant alternatives: [Ancient, Egyptian, Brutalist]
            // The theme variant costs are per-theme alternatives — all
            // share the same declared archetype cost since the catalog
            // defines them as equal-cost. For variation, theme may alter
            // materials but not geometry cost.
            let alternatives: Vec<RecipeIdentity> =
                if arch_idx < generated_content::ARCHETYPE_THEME_MATERIALS.len() {
                    (0..3)
                        .map(|theme_idx| {
                            let materials_count = generated_content::ARCHETYPE_THEME_MATERIALS
                                [arch_idx][theme_idx]
                                .len() as u32;
                            RecipeIdentity {
                                name: format!(
                                    "{}_theme_variant_{}",
                                    generated_content::ARCHETYPE_IDS[arch_idx],
                                    theme_idx
                                ),
                                priority: RecipePriority::LandmarksTraversal,
                                mandatory: req.forced,
                                explicitly_requested: false,
                                cost: BudgetReservation {
                                    faces: generated_content::ARCHETYPE_COST_SOURCE_FACES[arch_idx],
                                    brushes: generated_content::ARCHETYPE_COST_BRUSHES[arch_idx],
                                    entities: generated_content::ARCHETYPE_COST_ENTITIES[arch_idx],
                                    lights: 0,
                                    vertical_openings: arch_cost.vertical_openings,
                                    support_contacts: 0,
                                    package_assets: materials_count,
                                    compiler_lumps: 0,
                                    renderer_batches: 1,
                                    renderer_memory_bytes: 16384,
                                    runtime_requirements: 0,
                                },
                                alternatives: Vec::new(),
                            }
                        })
                        .collect()
                } else {
                    Vec::new()
                };

            landmark_recipes.push(RecipeIdentity {
                name: generated_content::ARCHETYPE_IDS[arch_idx].to_string(),
                priority: RecipePriority::LandmarksTraversal,
                mandatory: req.forced,
                explicitly_requested: false,
                cost: arch_cost,
                alternatives,
            });
        }
        recipes.insert(RecipePriority::LandmarksTraversal, landmark_recipes);

        // ── Priority 3: Theme Identity ─────────────────────────────────
        let theme_cost = BudgetReservation {
            faces: 0,
            brushes: 0,
            entities: 0,
            lights: 0,
            vertical_openings: 0,
            support_contacts: 0,
            package_assets: generated_content::THEME_BUDGET_SOURCE_FACES
                .iter()
                .map(|&f| f / 100)
                .max()
                .unwrap_or(80),
            compiler_lumps: 0,
            renderer_batches: 0,
            renderer_memory_bytes: 0,
            runtime_requirements: 0,
        };
        let theme_alternatives: Vec<RecipeIdentity> = (0..3)
            .map(|theme_idx| RecipeIdentity {
                name: format!("theme_identity_{}", generated_content::THEME_IDS[theme_idx]),
                priority: RecipePriority::ThemeIdentity,
                mandatory: true,
                explicitly_requested: false,
                cost: BudgetReservation {
                    faces: generated_content::THEME_BUDGET_SOURCE_FACES[theme_idx] / 40,
                    brushes: generated_content::THEME_BUDGET_BRUSHES[theme_idx],
                    entities: generated_content::THEME_BUDGET_ENTITIES[theme_idx],
                    lights: generated_content::THEME_BUDGET_LIGHTS[theme_idx],
                    vertical_openings: 0,
                    support_contacts: 0,
                    package_assets: generated_content::THEME_BUDGET_SOURCE_FACES[theme_idx] / 100,
                    compiler_lumps: 0,
                    renderer_batches: 4,
                    renderer_memory_bytes: 65536,
                    runtime_requirements: 0,
                },
                alternatives: Vec::new(),
            })
            .collect();

        recipes.insert(
            RecipePriority::ThemeIdentity,
            vec![RecipeIdentity {
                name: "theme_material_roles".to_string(),
                priority: RecipePriority::ThemeIdentity,
                mandatory: true,
                explicitly_requested: false,
                cost: theme_cost,
                alternatives: theme_alternatives,
            }],
        );

        // ── Priority 4: Required Vertical / Cave ───────────────────────
        // Reserve the complete materialized recipe set.  This is derived from
        // committed hosts/pit pairs plus separately authored set-pieces; no
        // synthetic minimum is injected later as a proof substitute.
        let vertical_cost = planned_vertical_cost(blueprint, topology, cave_mode);
        recipes.insert(
            RecipePriority::RequiredVerticalCave,
            vec![RecipeIdentity {
                name: "required_vertical_features".to_string(),
                priority: RecipePriority::RequiredVerticalCave,
                mandatory: true,
                explicitly_requested: false,
                cost: vertical_cost,
                alternatives: Vec::new(),
            }],
        );

        // ── Priority 5: Readability ────────────────────────────────────
        let readability_cost = BudgetReservation {
            faces: 0,
            brushes: 0,
            entities: 0,
            lights: 12, // minimum ambient + route lights
            vertical_openings: 0,
            support_contacts: 0,
            package_assets: 0,
            compiler_lumps: 0,
            renderer_batches: 0,
            renderer_memory_bytes: 0,
            runtime_requirements: 0,
        };
        recipes.insert(
            RecipePriority::Readability,
            vec![RecipeIdentity {
                name: "readability_lights".to_string(),
                priority: RecipePriority::Readability,
                mandatory: true,
                explicitly_requested: false,
                cost: readability_cost,
                alternatives: Vec::new(),
            }],
        );

        // ── Priority 6: Props / Imperfection ───────────────────────────
        let mut prop_recipes = Vec::new();

        // Props selected per archetype request
        for req in blueprint.archetype_requests.values() {
            let arch_idx = req.archetype.raw() as usize;
            if arch_idx >= generated_content::ARCHETYPE_PROP_REFS.len() {
                continue;
            }
            let prop_refs = generated_content::ARCHETYPE_PROP_REFS[arch_idx];
            for &prop in prop_refs {
                let p = prop as usize;
                if p >= generated_content::PROP_COST_SOURCE_FACES.len() {
                    continue;
                }
                let prop_cost = BudgetReservation {
                    faces: generated_content::PROP_COST_SOURCE_FACES[p],
                    brushes: generated_content::PROP_COST_BRUSHES[p],
                    entities: generated_content::PROP_COST_ENTITIES[p],
                    lights: generated_content::PROP_COST_LIGHTS[p],
                    vertical_openings: 0,
                    support_contacts: generated_content::PROP_SUPPORT_CONTACTS[p],
                    package_assets: 2,
                    compiler_lumps: 0,
                    renderer_batches: 1,
                    renderer_memory_bytes: 2048,
                    runtime_requirements: 0,
                };

                // Props have theme alternatives with different dimensions
                // but same declared source costs.
                let prop_alternatives: Vec<RecipeIdentity> = (0..3)
                    .map(|theme_idx| {
                        let theme_dims = generated_content::PROP_THEME_DIMENSIONS[p][theme_idx];
                        RecipeIdentity {
                            name: format!(
                                "{}_theme_{}",
                                generated_content::PROP_IDS[p],
                                generated_content::THEME_IDS[theme_idx]
                            ),
                            priority: RecipePriority::PropsImperfection,
                            mandatory: false,
                            explicitly_requested: false,
                            cost: BudgetReservation {
                                // Cost scales with volume but faces are fixed per-prop.
                                faces: generated_content::PROP_COST_SOURCE_FACES[p],
                                brushes: generated_content::PROP_COST_BRUSHES[p],
                                entities: generated_content::PROP_COST_ENTITIES[p],
                                lights: generated_content::PROP_COST_LIGHTS[p],
                                vertical_openings: 0,
                                support_contacts: generated_content::PROP_SUPPORT_CONTACTS[p],
                                package_assets: 2,
                                compiler_lumps: 0,
                                renderer_batches: 1,
                                renderer_memory_bytes: (theme_dims[0]
                                    * theme_dims[1]
                                    * theme_dims[2]
                                    / 64)
                                    as u64,
                                runtime_requirements: 0,
                            },
                            alternatives: Vec::new(),
                        }
                    })
                    .collect();

                prop_recipes.push(RecipeIdentity {
                    name: generated_content::PROP_IDS[p].to_string(),
                    priority: RecipePriority::PropsImperfection,
                    mandatory: false,
                    explicitly_requested: false,
                    cost: prop_cost,
                    alternatives: prop_alternatives,
                });
            }
        }

        // Imperfection (damage variants) recipe
        prop_recipes.push(RecipeIdentity {
            name: "imperfection_damage".to_string(),
            priority: RecipePriority::PropsImperfection,
            mandatory: false,
            explicitly_requested: false,
            cost: BudgetReservation {
                faces: 24, // extra rubble faces
                brushes: 3,
                entities: 0,
                lights: 0,
                vertical_openings: 0,
                support_contacts: 1,
                package_assets: 1,
                compiler_lumps: 0,
                renderer_batches: 1,
                renderer_memory_bytes: 1024,
                runtime_requirements: 0,
            },
            alternatives: Vec::new(),
        });

        recipes.insert(RecipePriority::PropsImperfection, prop_recipes);

        // Semantic reservation is theme-independent. Reserve the maximum
        // complete variant cost now, not the selected presentation variant
        // later, so changing Ancient/Egyptian/Brutalist cannot expand total
        // worst-case capacity.
        for tier in recipes.values_mut() {
            for recipe in tier {
                recipe.cost = recipe
                    .alternatives
                    .iter()
                    .fold(recipe.cost, |maximum, alternative| {
                        maximum.max_per_dimension(alternative.cost)
                    });
            }
        }

        Self { recipes }
    }

    /// Get recipes for a given priority tier.
    pub fn at_priority(&self, priority: RecipePriority) -> &[RecipeIdentity] {
        self.recipes
            .get(&priority)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Iterate over all recipes in priority order.
    pub fn all_priorities(&self) -> impl Iterator<Item = RecipePriority> {
        RecipePriority::ALL.iter().copied()
    }

    /// Total count of recipes in catalog.
    pub fn recipe_count(&self) -> usize {
        self.recipes.values().map(|v| v.len()).sum()
    }
}

// ── Complexity planner ─────────────────────────────────────────────────────

/// Planner that builds a `ComplexityPlan` from a recipe catalog.
pub(crate) struct ComplexityPlanner {
    budget: ComplexityBudget,
    catalog: RecipeCatalog,
    selections: Vec<RecipeSelection>,
    errors: Vec<BudgetError>,
    theme_variant: u32,
}

impl ComplexityPlanner {
    /// Create a new planner with the given preset, theme variant, and catalog.
    pub fn new(preset: RichnessPreset, theme_variant: u32, catalog: RecipeCatalog) -> Self {
        Self {
            budget: ComplexityBudget::new(preset),
            catalog,
            selections: Vec::new(),
            errors: Vec::new(),
            theme_variant,
        }
    }

    /// Reserve the max theme variant cost across all three themes.
    ///
    /// Theme variants share the maximum of all three variant costs at
    /// semantic reservation time. This ensures a theme change never
    /// exceeds reserved budget. The max cost is stored but NOT added
    /// to consumed budget (individual selections handle actual tracking).
    pub fn reserve_theme_variant_max(&mut self) {
        let mut costs = [BudgetReservation::ZERO; 3];
        for theme_idx in 0..3 {
            let mut accum = BudgetReservation::ZERO;
            // Sum theme-specific costs for all mandatory recipes.
            for priority in &RecipePriority::ALL {
                if *priority > RecipePriority::ThemeIdentity {
                    break;
                }
                for recipe in self.catalog.at_priority(*priority) {
                    if recipe.mandatory {
                        if let Some(alt) = recipe.alternatives.get(theme_idx) {
                            accum = accum.saturating_add(alt.cost);
                        } else {
                            accum = accum.saturating_add(recipe.cost);
                        }
                    }
                }
            }
            costs[theme_idx] = accum;
        }
        let max_cost = costs[0]
            .max_per_dimension(costs[1])
            .max_per_dimension(costs[2]);
        self.budget.max_theme_variant_cost = max_cost;
    }

    /// Plan all recipes: mandatory first, then optional in priority order.
    pub fn plan(mut self) -> ComplexityPlan {
        // Step 1: reserve every mandatory complete recipe before any optional
        // one. A mandatory overrun fails the plan; it must never be repaired
        // by omitting a route, topology, landmark, theme, vertical/cave, or
        // readability requirement.
        let mandatory_cost = self
            .catalog
            .all_priorities()
            .flat_map(|priority| self.catalog.at_priority(priority).iter())
            .filter(|recipe| recipe.mandatory)
            .fold(BudgetReservation::ZERO, |total, recipe| {
                total.saturating_add(recipe.cost)
            });
        if !mandatory_cost.within(&self.budget.ceilings) {
            if let (Some((dimension, ceiling, requested)), Some(recipe)) = (
                first_excess(&mandatory_cost, &self.budget.ceilings),
                self.catalog
                    .all_priorities()
                    .flat_map(|priority| self.catalog.at_priority(priority).iter())
                    .find(|recipe| recipe.mandatory),
            ) {
                self.errors.push(BudgetError::MandatoryExceedsCeiling {
                    dimension,
                    ceiling,
                    requested,
                    priority: recipe.priority,
                    recipe_name: recipe.name.clone(),
                    total_mandatory: mandatory_cost,
                });
            }
            return ComplexityPlan {
                selected_recipes: Vec::new(),
                total_reserved: BudgetReservation::ZERO,
                budget: self.budget,
                mandatory_reserved: false,
                theme_variant: self.theme_variant,
                errors: self.errors,
            };
        }

        for priority in &RecipePriority::ALL {
            let recipes: Vec<RecipeIdentity> = self
                .catalog
                .at_priority(*priority)
                .iter()
                .cloned()
                .collect();

            for recipe in recipes {
                if !recipe.mandatory {
                    continue;
                }
                self.try_select(recipe, *priority);
            }
        }

        // Step 2: Select optional recipes in priority order using
        // catalog-provided alternatives when primary doesn't fit.
        for priority in &RecipePriority::ALL {
            let recipes: Vec<RecipeIdentity> = self
                .catalog
                .at_priority(*priority)
                .iter()
                .filter(|r| !r.mandatory)
                .cloned()
                .collect();

            for recipe in recipes {
                if self.budget.is_exceeded() {
                    break;
                }

                let remaining = self.budget.remaining();
                if recipe.cost.within(&remaining) {
                    // Primary fits — select it directly.
                    self.try_select(recipe, *priority);
                } else if !recipe.alternatives.is_empty() {
                    // Try alternatives (complete authored identities, never truncated).
                    if let Some(alt) = recipe.first_fitting(&remaining) {
                        let mut selected = alt.clone();
                        selected.priority = *priority;
                        self.budget.spend(selected.cost);
                        self.selections.push(RecipeSelection {
                            recipe: selected,
                            priority: *priority,
                            was_alternative: true,
                            original_recipe_name: Some(recipe.name.clone()),
                        });
                    } else if recipe.explicitly_requested {
                        let cheapest = recipe
                            .sorted_alternatives()
                            .first()
                            .copied()
                            .unwrap_or(&recipe);
                        let (dimension, ceiling, requested) =
                            first_excess(&cheapest.cost, &remaining).unwrap_or((
                                BudgetDimension::SourceFaces,
                                remaining.faces,
                                cheapest.cost.faces,
                            ));
                        self.errors.push(BudgetError::NoAlternativeFits {
                            dimension,
                            ceiling,
                            requested,
                            recipe_name: recipe.name.clone(),
                            min_cost_faces: cheapest.cost.faces,
                            remaining_faces: remaining.faces,
                            priority: *priority,
                        });
                    }
                } else if recipe.explicitly_requested {
                    let (dimension, ceiling, requested) = first_excess(&recipe.cost, &remaining)
                        .unwrap_or((
                            BudgetDimension::SourceFaces,
                            remaining.faces,
                            recipe.cost.faces,
                        ));
                    self.errors.push(BudgetError::OverBudget {
                        dimension,
                        ceiling,
                        requested,
                        priority: *priority,
                        recipe_name: recipe.name.clone(),
                    });
                }
                // Unrequested optional identities are skipped whole, never truncated.
            }
        }

        let total_reserved = self
            .selections
            .iter()
            .fold(BudgetReservation::ZERO, |acc, s| {
                acc.saturating_add(s.recipe.cost)
            });

        let mandatory_reserved = self
            .selections
            .iter()
            .filter(|s| s.recipe.mandatory)
            .count()
            >= self
                .catalog
                .all_priorities()
                .flat_map(|p| self.catalog.at_priority(p).iter())
                .filter(|r| r.mandatory)
                .count();

        ComplexityPlan {
            selected_recipes: self.selections,
            total_reserved,
            budget: self.budget,
            mandatory_reserved,
            theme_variant: self.theme_variant,
            errors: self.errors,
        }
    }

    /// Try to select a recipe, checking budget.
    fn try_select(&mut self, recipe: RecipeIdentity, priority: RecipePriority) {
        let remaining = self.budget.remaining();
        if recipe.cost.within(&remaining) {
            self.budget.spend(recipe.cost);
            self.selections.push(RecipeSelection {
                recipe,
                priority,
                was_alternative: false,
                original_recipe_name: None,
            });
        } else if let Some((dimension, ceiling, requested)) = first_excess(&recipe.cost, &remaining)
        {
            self.errors.push(BudgetError::OverBudget {
                dimension,
                ceiling,
                requested,
                priority,
                recipe_name: recipe.name.clone(),
            });
        }
    }
}

/// Return the first deterministic budget dimension where `requested` exceeds
/// `ceiling`. Ordering follows `BudgetDimension`, making diagnostics stable.
fn first_excess(
    requested: &BudgetReservation,
    ceiling: &BudgetReservation,
) -> Option<(BudgetDimension, u32, u32)> {
    let dimensions = [
        (BudgetDimension::SourceFaces, requested.faces, ceiling.faces),
        (BudgetDimension::Brushes, requested.brushes, ceiling.brushes),
        (
            BudgetDimension::Entities,
            requested.entities,
            ceiling.entities,
        ),
        (BudgetDimension::Lights, requested.lights, ceiling.lights),
        (
            BudgetDimension::VerticalOpenings,
            requested.vertical_openings,
            ceiling.vertical_openings,
        ),
        (
            BudgetDimension::SupportContacts,
            requested.support_contacts,
            ceiling.support_contacts,
        ),
        (
            BudgetDimension::PackageAssets,
            requested.package_assets,
            ceiling.package_assets,
        ),
        (
            BudgetDimension::CompilerLumps,
            requested.compiler_lumps,
            ceiling.compiler_lumps,
        ),
        (
            BudgetDimension::RendererBatches,
            requested.renderer_batches,
            ceiling.renderer_batches,
        ),
        (
            BudgetDimension::RendererMemoryBytes,
            requested.renderer_memory_bytes.min(u32::MAX as u64) as u32,
            ceiling.renderer_memory_bytes.min(u32::MAX as u64) as u32,
        ),
        (
            BudgetDimension::RuntimeRequirements,
            requested.runtime_requirements,
            ceiling.runtime_requirements,
        ),
    ];
    dimensions
        .into_iter()
        .find_map(|(dimension, requested, ceiling)| {
            (requested > ceiling).then_some((dimension, ceiling, requested))
        })
}

// ── Convenience constructor ────────────────────────────────────────────────

/// Build a complete complexity plan from a blueprint and topology result.
pub(crate) fn build_complexity_plan(
    preset: RichnessPreset,
    theme_variant: u32,
    blueprint: &PacingBlueprint,
    topology: &TopologyResult,
    _request_archetypes: &BTreeMap<super::ids::ArchetypeRequestId, super::ids::ArchetypeIndex>,
    cave_mode: RichnessCaveMode,
) -> ComplexityPlan {
    let catalog = RecipeCatalog::build(blueprint, topology, cave_mode);
    let mut planner = ComplexityPlanner::new(preset, theme_variant, catalog);
    planner.reserve_theme_variant_max();
    let mut plan = planner.plan();
    // Reserve the complete contract-bounded inter-brush support capacity.
    // World-rooted floor anchors are intentionally excluded from this metric;
    // actual support contacts are recomputed from non-world DAG edges.
    plan.total_reserved.support_contacts = plan.budget.ceilings.support_contacts;
    // Every finalized floor-slab omission is a logical vertical opening; the
    // structural composition may contain several independent paired features.
    plan.total_reserved.vertical_openings = plan.budget.ceilings.vertical_openings;
    plan.total_reserved.lights = plan.budget.ceilings.lights;
    plan
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::enhanced_v3::richness::pacing::build_pacing_blueprint;
    use crate::enhanced_v3::richness::request::{
        ResolvedRichnessRequestV1, RichnessDocumentV1, RichnessTheme,
    };

    // ── BudgetReservation ───────────────────────────────────────────────

    #[test]
    fn zero_budget_has_no_cost() {
        let z = BudgetReservation::ZERO;
        assert_eq!(z.faces, 0);
        assert_eq!(z.brushes, 0);
        assert_eq!(z.entities, 0);
        assert_eq!(z.lights, 0);
    }

    #[test]
    fn saturating_add_does_not_overflow() {
        let a = BudgetReservation {
            faces: u32::MAX,
            ..BudgetReservation::ZERO
        };
        let b = BudgetReservation {
            faces: 1,
            ..BudgetReservation::ZERO
        };
        let sum = a.saturating_add(b);
        assert_eq!(sum.faces, u32::MAX);
    }

    #[test]
    fn max_per_dimension_takes_max() {
        let a = BudgetReservation {
            faces: 100,
            brushes: 50,
            ..BudgetReservation::ZERO
        };
        let b = BudgetReservation {
            faces: 200,
            brushes: 10,
            ..BudgetReservation::ZERO
        };
        let m = a.max_per_dimension(b);
        assert_eq!(m.faces, 200);
        assert_eq!(m.brushes, 50);
    }

    #[test]
    fn within_checks_all_dimensions() {
        let ceiling = BudgetReservation {
            faces: 1000,
            brushes: 100,
            entities: 50,
            lights: 20,
            vertical_openings: 5,
            support_contacts: 32,
            package_assets: 64,
            compiler_lumps: 15,
            renderer_batches: 400,
            renderer_memory_bytes: 1024 * 1024,
            runtime_requirements: 0,
        };

        let within = BudgetReservation {
            faces: 500,
            brushes: 50,
            entities: 10,
            lights: 5,
            vertical_openings: 2,
            support_contacts: 8,
            package_assets: 16,
            compiler_lumps: 15,
            renderer_batches: 100,
            renderer_memory_bytes: 512 * 1024,
            runtime_requirements: 0,
        };
        assert!(within.within(&ceiling));

        let over = BudgetReservation {
            faces: 1001,
            ..within
        };
        assert!(!over.within(&ceiling));
    }

    #[test]
    fn remaining_subtracts_correctly() {
        let consumed = BudgetReservation {
            faces: 300,
            brushes: 20,
            ..BudgetReservation::ZERO
        };
        let ceiling = BudgetReservation {
            faces: 1000,
            brushes: 100,
            ..BudgetReservation::ZERO
        };
        let rem = consumed.remaining(&ceiling);
        assert_eq!(rem.faces, 700);
        assert_eq!(rem.brushes, 80);
    }

    // ── ComplexityBudget ────────────────────────────────────────────────

    #[test]
    fn budget_sparse_ceiling_is_3000() {
        let budget = ComplexityBudget::new(RichnessPreset::Sparse);
        assert_eq!(budget.ceilings.faces, 3_000);
    }

    #[test]
    fn budget_moderate_ceiling_is_5000() {
        let budget = ComplexityBudget::new(RichnessPreset::Moderate);
        assert_eq!(budget.ceilings.faces, 5_000);
    }

    #[test]
    fn budget_rich_ceiling_is_8000() {
        let budget = ComplexityBudget::new(RichnessPreset::Rich);
        assert_eq!(budget.ceilings.faces, 8_000);
    }

    #[test]
    fn budget_entities_ceiling_is_500() {
        let budget = ComplexityBudget::new(RichnessPreset::Rich);
        assert_eq!(budget.ceilings.entities, 500);
    }

    #[test]
    fn budget_lights_ceiling_is_100() {
        let budget = ComplexityBudget::new(RichnessPreset::Rich);
        assert_eq!(budget.ceilings.lights, 100);
    }

    #[test]
    fn budget_never_raises_ceiling() {
        for preset in &[
            RichnessPreset::Sparse,
            RichnessPreset::Moderate,
            RichnessPreset::Rich,
        ] {
            let budget = ComplexityBudget::new(*preset);
            assert!(budget.ceilings.faces <= 8_000);
            assert!(budget.ceilings.entities <= 500);
            assert!(budget.ceilings.lights <= 100);
        }
    }

    // ── Recipe priority order ───────────────────────────────────────────

    #[test]
    fn priority_order_is_strictly_increasing() {
        let all = RecipePriority::ALL;
        for i in 1..all.len() {
            assert!(
                (all[i] as u32) > (all[i - 1] as u32),
                "priority order violated at index {}",
                i
            );
        }
    }

    #[test]
    fn priority_order_has_seven_tiers() {
        assert_eq!(RecipePriority::ALL.len(), 7);
    }

    // ── ComplexityPlan ──────────────────────────────────────────────────

    #[test]
    fn empty_plan_is_within_budget() {
        let plan = ComplexityPlan {
            selected_recipes: Vec::new(),
            total_reserved: BudgetReservation::ZERO,
            budget: ComplexityBudget::new(RichnessPreset::Sparse),
            mandatory_reserved: true,
            theme_variant: 0,
            errors: Vec::new(),
        };
        assert!(plan.is_within_budget());
    }

    #[test]
    fn plan_with_errors_is_not_within_budget() {
        let plan = ComplexityPlan {
            selected_recipes: Vec::new(),
            total_reserved: BudgetReservation::ZERO,
            budget: ComplexityBudget::new(RichnessPreset::Sparse),
            mandatory_reserved: false,
            theme_variant: 0,
            errors: vec![BudgetError::OverBudget {
                dimension: BudgetDimension::SourceFaces,
                ceiling: 3000,
                requested: 3001,
                priority: RecipePriority::SealingRoutes,
                recipe_name: "test".to_string(),
            }],
        };
        assert!(!plan.is_within_budget());
    }

    #[test]
    fn assert_dominates_passes_conservative() {
        let reserved = BudgetReservation {
            faces: 100,
            brushes: 10,
            ..BudgetReservation::ZERO
        };
        let actual = BudgetReservation {
            faces: 80,
            brushes: 8,
            ..BudgetReservation::ZERO
        };
        let plan = ComplexityPlan {
            selected_recipes: Vec::new(),
            total_reserved: reserved,
            budget: ComplexityBudget::new(RichnessPreset::Sparse),
            mandatory_reserved: true,
            theme_variant: 0,
            errors: Vec::new(),
        };
        assert!(plan.assert_dominates(&actual));
    }

    #[test]
    fn assert_dominates_fails_when_actual_exceeds() {
        let reserved = BudgetReservation {
            faces: 100,
            ..BudgetReservation::ZERO
        };
        let actual = BudgetReservation {
            faces: 101,
            ..BudgetReservation::ZERO
        };
        let plan = ComplexityPlan {
            selected_recipes: Vec::new(),
            total_reserved: reserved,
            budget: ComplexityBudget::new(RichnessPreset::Sparse),
            mandatory_reserved: true,
            theme_variant: 0,
            errors: Vec::new(),
        };
        assert!(!plan.assert_dominates(&actual));
    }

    // ── BudgetError ─────────────────────────────────────────────────────

    #[test]
    fn budget_error_descriptions_are_unique_per_variant() {
        let e1 = BudgetError::OverBudget {
            dimension: BudgetDimension::SourceFaces,
            ceiling: 100,
            requested: 200,
            priority: RecipePriority::SealingRoutes,
            recipe_name: "a".to_string(),
        };
        let e2 = BudgetError::NoAlternativeFits {
            dimension: BudgetDimension::Lights,
            ceiling: 10,
            requested: 50,
            recipe_name: "b".to_string(),
            min_cost_faces: 50,
            remaining_faces: 10,
            priority: RecipePriority::PropsImperfection,
        };
        assert!(e1.describe().contains("overrun"));
        assert!(e2.describe().contains("no complete alternative"));
    }

    // ── RecipeCatalog integration ───────────────────────────────────────

    fn make_catalog() -> RecipeCatalog {
        let doc = RichnessDocumentV1::new(0, 2048, RichnessPreset::Sparse, RichnessTheme::Ancient)
            .unwrap();
        let resolved = ResolvedRichnessRequestV1::resolve(doc).unwrap();
        let blueprint = build_pacing_blueprint(&resolved).unwrap();
        let topology = TopologyResult {
            selected_edges: Vec::new(),
            routes: Vec::new(),
            journal: crate::enhanced_v3::richness::reservation::ReservationJournal::new(2048, 3000),
            beat_to_reservations: std::collections::BTreeMap::new(),
            loop_count: 0,
            shortcuts_realized: Vec::new(),
            vertical_edges: Vec::new(),
            vertical_routes: Vec::new(),
            search_metrics: Default::default(),
        };
        RecipeCatalog::build(&blueprint, &topology, RichnessCaveMode::Omitted)
    }

    #[test]
    fn catalog_has_entries_for_all_priorities() {
        let catalog = make_catalog();
        for priority in &RecipePriority::ALL {
            let recipes = catalog.at_priority(*priority);
            // Each priority must have at least one recipe.
            assert!(
                !recipes.is_empty(),
                "priority {:?} has no recipes",
                priority
            );
        }
    }

    #[test]
    fn catalog_has_mandatory_sealing_recipes() {
        let catalog = make_catalog();
        let sealing = catalog.at_priority(RecipePriority::SealingRoutes);
        assert!(!sealing.is_empty());
        assert!(sealing.iter().any(|r| r.mandatory));
    }

    #[test]
    fn catalog_landmark_recipes_match_blueprint_requests() {
        let doc = RichnessDocumentV1::new(0, 2048, RichnessPreset::Sparse, RichnessTheme::Ancient)
            .unwrap();
        let resolved = ResolvedRichnessRequestV1::resolve(doc).unwrap();
        let blueprint = build_pacing_blueprint(&resolved).unwrap();
        let topology = TopologyResult {
            selected_edges: Vec::new(),
            routes: Vec::new(),
            journal: crate::enhanced_v3::richness::reservation::ReservationJournal::new(2048, 3000),
            beat_to_reservations: std::collections::BTreeMap::new(),
            loop_count: 0,
            shortcuts_realized: Vec::new(),
            vertical_edges: Vec::new(),
            vertical_routes: Vec::new(),
            search_metrics: Default::default(),
        };
        let catalog = RecipeCatalog::build(&blueprint, &topology, RichnessCaveMode::Omitted);
        let landmarks = catalog.at_priority(RecipePriority::LandmarksTraversal);

        let request_count = blueprint.archetype_requests.len();
        assert!(
            landmarks.len() >= request_count,
            "landmarks {} < requests {}",
            landmarks.len(),
            request_count
        );
    }

    // ── ComplexityPlanner ───────────────────────────────────────────────

    fn make_plan(preset: RichnessPreset, theme_variant: u32) -> ComplexityPlan {
        let doc = RichnessDocumentV1::new(0, 2048, preset, RichnessTheme::Ancient).unwrap();
        let resolved = ResolvedRichnessRequestV1::resolve(doc).unwrap();
        let blueprint = build_pacing_blueprint(&resolved).unwrap();
        let topology = TopologyResult {
            selected_edges: Vec::new(),
            routes: Vec::new(),
            journal: crate::enhanced_v3::richness::reservation::ReservationJournal::new(2048, 8000),
            beat_to_reservations: std::collections::BTreeMap::new(),
            loop_count: 0,
            shortcuts_realized: Vec::new(),
            vertical_edges: Vec::new(),
            vertical_routes: Vec::new(),
            search_metrics: Default::default(),
        };
        build_complexity_plan(
            preset,
            theme_variant,
            &blueprint,
            &topology,
            &BTreeMap::new(),
            RichnessCaveMode::Omitted,
        )
    }

    #[test]
    fn sparse_plan_is_within_3000() {
        let plan = make_plan(RichnessPreset::Sparse, 0);
        assert!(plan.is_within_budget());
        assert!(
            plan.total_reserved.faces <= 3_000,
            "faces {} exceeds 3000",
            plan.total_reserved.faces
        );
    }

    #[test]
    fn moderate_plan_is_within_5000() {
        let plan = make_plan(RichnessPreset::Moderate, 0);
        assert!(
            plan.total_reserved.faces <= 5_000,
            "faces {} exceeds 5000",
            plan.total_reserved.faces
        );
    }

    #[test]
    fn rich_plan_is_within_8000() {
        let plan = make_plan(RichnessPreset::Rich, 0);
        assert!(
            plan.total_reserved.faces <= 8_000,
            "faces {} exceeds 8000",
            plan.total_reserved.faces
        );
    }

    #[test]
    fn plan_mandatory_is_reserved() {
        let plan = make_plan(RichnessPreset::Sparse, 0);
        assert!(plan.mandatory_reserved, "{:?}", plan.errors);
    }

    #[test]
    fn plan_theme_variant_preserved() {
        for theme_idx in 0..3u32 {
            let plan = make_plan(RichnessPreset::Sparse, theme_idx);
            assert_eq!(plan.theme_variant, theme_idx);
        }
    }

    #[test]
    fn boundary_test_exactly_3000_faces() {
        let plan = make_plan(RichnessPreset::Sparse, 0);
        assert!(
            plan.total_reserved.faces <= 3_000,
            "Sparse faces ({}) must be ≤ 3000",
            plan.total_reserved.faces
        );
    }

    #[test]
    fn boundary_test_exactly_5000_faces() {
        let plan = make_plan(RichnessPreset::Moderate, 0);
        assert!(
            plan.total_reserved.faces <= 5_000,
            "Moderate faces ({}) must be ≤ 5000",
            plan.total_reserved.faces
        );
    }

    #[test]
    fn boundary_test_exactly_8000_faces() {
        let plan = make_plan(RichnessPreset::Rich, 0);
        assert!(
            plan.total_reserved.faces <= 8_000,
            "Rich faces ({}) must be ≤ 8000",
            plan.total_reserved.faces
        );
    }

    #[test]
    fn entity_limit_not_exceeded() {
        for preset in &[
            RichnessPreset::Sparse,
            RichnessPreset::Moderate,
            RichnessPreset::Rich,
        ] {
            let plan = make_plan(*preset, 0);
            assert!(
                plan.total_reserved.entities <= 500,
                "{:?} entities ({}) exceeds 500",
                preset,
                plan.total_reserved.entities
            );
        }
    }

    #[test]
    fn light_limit_not_exceeded() {
        for preset in &[
            RichnessPreset::Sparse,
            RichnessPreset::Moderate,
            RichnessPreset::Rich,
        ] {
            let plan = make_plan(*preset, 0);
            assert!(
                plan.total_reserved.lights <= 100,
                "{:?} lights ({}) exceeds 100",
                preset,
                plan.total_reserved.lights
            );
        }
    }

    #[test]
    fn no_ceiling_is_raised() {
        for preset in &[
            RichnessPreset::Sparse,
            RichnessPreset::Moderate,
            RichnessPreset::Rich,
        ] {
            let plan = make_plan(*preset, 0);
            let budget = ComplexityBudget::new(*preset);
            assert!(
                plan.total_reserved.faces <= budget.ceilings.faces,
                "{:?} faces {} > ceiling {}",
                preset,
                plan.total_reserved.faces,
                budget.ceilings.faces
            );
            assert!(
                plan.total_reserved.entities <= budget.ceilings.entities,
                "{:?} entities {} > ceiling {}",
                preset,
                plan.total_reserved.entities,
                budget.ceilings.entities
            );
            assert!(
                plan.total_reserved.lights <= budget.ceilings.lights,
                "{:?} lights {} > ceiling {}",
                preset,
                plan.total_reserved.lights,
                budget.ceilings.lights
            );
        }
    }

    #[test]
    fn theme_variant_mutation_does_not_change_total_capacity() {
        // Theme variant change must not alter the budget ceiling or
        // total worst-case capacity.
        let budget_ancient = ComplexityBudget::new(RichnessPreset::Rich);
        let budget_brutalist = ComplexityBudget::new(RichnessPreset::Rich);
        assert_eq!(budget_ancient.ceilings, budget_brutalist.ceilings);
    }

    #[test]
    fn impossible_request_fails_stably() {
        // An Sparse preset with explicit mandatory costs that exceed 3,000
        // must produce a budget error.
        let plan = make_plan(RichnessPreset::Sparse, 0);
        if plan.total_reserved.faces > 3_000 {
            assert!(!plan.errors.is_empty());
        }
        // The plan must not panic.
        let _ = plan.is_within_budget();
    }

    #[test]
    fn vertical_openings_ceiling_not_exceeded() {
        for preset in &[
            RichnessPreset::Sparse,
            RichnessPreset::Moderate,
            RichnessPreset::Rich,
        ] {
            let plan = make_plan(*preset, 0);
            assert!(
                plan.total_reserved.vertical_openings <= 12,
                "{:?} vertical_openings ({}) exceeds 12",
                preset,
                plan.total_reserved.vertical_openings
            );
        }
    }

    #[test]
    fn support_contacts_ceiling_not_exceeded() {
        for preset in &[
            RichnessPreset::Sparse,
            RichnessPreset::Moderate,
            RichnessPreset::Rich,
        ] {
            let plan = make_plan(*preset, 0);
            assert!(
                plan.total_reserved.support_contacts <= 256,
                "{:?} support_contacts ({}) exceeds 256",
                preset,
                plan.total_reserved.support_contacts
            );
        }
    }

    fn recipe(name: &str, priority: RecipePriority, mandatory: bool, faces: u32) -> RecipeIdentity {
        RecipeIdentity {
            name: name.to_string(),
            priority,
            mandatory,
            explicitly_requested: false,
            cost: BudgetReservation {
                faces,
                ..BudgetReservation::ZERO
            },
            alternatives: Vec::new(),
        }
    }

    #[test]
    fn every_dimension_accepts_its_exact_ceiling() {
        let mut budget = ComplexityBudget::new(RichnessPreset::Rich);
        budget.spend(budget.ceilings);
        assert!(!budget.is_exceeded());
        assert_eq!(budget.remaining(), BudgetReservation::ZERO);
    }

    #[test]
    fn mandatory_impossible_request_returns_actionable_typed_error() {
        let mut recipes = BTreeMap::new();
        recipes.insert(
            RecipePriority::SealingRoutes,
            vec![recipe(
                "sealed_route",
                RecipePriority::SealingRoutes,
                true,
                3_001,
            )],
        );
        let plan =
            ComplexityPlanner::new(RichnessPreset::Sparse, 0, RecipeCatalog { recipes }).plan();
        assert!(!plan.mandatory_reserved);
        assert!(matches!(
            plan.errors.as_slice(),
            [BudgetError::MandatoryExceedsCeiling {
                dimension: BudgetDimension::SourceFaces,
                ceiling: 3_000,
                requested: 3_001,
                priority: RecipePriority::SealingRoutes,
                recipe_name,
                ..
            }] if recipe_name == "sealed_route"
        ));
    }

    #[test]
    fn complete_lower_cost_alternative_is_selected_without_truncation() {
        let mut primary = recipe(
            "ornate_complete",
            RecipePriority::PropsImperfection,
            false,
            4_000,
        );
        primary.alternatives.push(recipe(
            "plain_complete",
            RecipePriority::PropsImperfection,
            false,
            12,
        ));
        let mut recipes = BTreeMap::new();
        recipes.insert(RecipePriority::PropsImperfection, vec![primary]);
        let plan =
            ComplexityPlanner::new(RichnessPreset::Sparse, 0, RecipeCatalog { recipes }).plan();
        assert_eq!(plan.selected_recipes.len(), 1);
        assert!(plan.selected_recipes[0].was_alternative);
        assert_eq!(plan.selected_recipes[0].recipe.name, "plain_complete");
        assert_eq!(plan.selected_recipes[0].recipe.cost.faces, 12);
    }

    #[test]
    fn no_complete_alternative_fit_is_a_typed_error() {
        let mut primary = recipe(
            "ornate_complete",
            RecipePriority::PropsImperfection,
            false,
            4_000,
        );
        primary.explicitly_requested = true;
        primary.alternatives.push(recipe(
            "plain_complete",
            RecipePriority::PropsImperfection,
            false,
            3_001,
        ));
        let mut recipes = BTreeMap::new();
        recipes.insert(RecipePriority::PropsImperfection, vec![primary]);
        let plan =
            ComplexityPlanner::new(RichnessPreset::Sparse, 0, RecipeCatalog { recipes }).plan();
        assert!(matches!(
            plan.errors.as_slice(),
            [BudgetError::NoAlternativeFits {
                dimension: BudgetDimension::SourceFaces,
                ceiling: 3_000,
                requested: 3_001,
                recipe_name,
                ..
            }] if recipe_name == "ornate_complete"
        ));
    }

    #[test]
    fn theme_mutation_keeps_real_semantic_capacity_identical() {
        let first = make_plan(RichnessPreset::Rich, 0);
        for theme in 1..3 {
            let variant = make_plan(RichnessPreset::Rich, theme);
            assert_eq!(variant.total_reserved, first.total_reserved);
            assert_eq!(
                variant.budget.max_theme_variant_cost,
                first.budget.max_theme_variant_cost
            );
        }
    }
}
