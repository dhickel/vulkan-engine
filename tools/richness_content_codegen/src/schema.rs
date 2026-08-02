//! Closed schema types for EnhancedV3 Richness V1 authored content.
//!
//! These types define the RON deserialization format and serve as the
//! internal representation for validation and code generation. Every
//! struct is `Serialize + Deserialize` so the tool can read authored
//! RON catalogs.
//!
//! # Contract
//!
//! - All dimensions are integer and quantum-aligned where structural.
//! - Every declared dimension >= 16 Quake units.
//! - IDs are stable, unique, lowercase snake_case strings.
//! - Cross-references are validated before code generation proceeds.
//! - No undeclared fields survive deserialization (deny_unknown_fields on
//!   every top-level catalog container).

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

// ── Schema version ─────────────────────────────────────────────────────────

/// Frozen schema version emitted into generated code.
pub const SCHEMA_VERSION: &str = "enhanced-v3-richness-content/v1";

/// Construction quantum — every structural dimension must be a multiple.
pub const CONSTRUCTION_QUANTUM: u32 = 16;

/// Minimum legal dimension for any declared value.
pub const MIN_DIMENSION: u32 = 16;

const REQUIRED_ARCHETYPE_IDS: [&str; 30] = [
    "ambush_cross",
    "antechamber",
    "arena",
    "barracks",
    "bridge_crossing",
    "cistern",
    "crossroads",
    "entrance_hall",
    "flooded_crypt",
    "foundry",
    "gallery",
    "grand_arena",
    "grand_stair_hall",
    "grotto",
    "guard_hall",
    "hypostyle_hall",
    "kill_court",
    "ladder_hub",
    "observatory",
    "ossuary",
    "overlook_hall",
    "pit_room",
    "reliquary",
    "shrine",
    "spiral_tower",
    "throne_hall",
    "trapped_gallery",
    "treasury",
    "vault",
    "vestibule",
];
const REQUIRED_PROP_IDS: [&str; 15] = [
    "altar",
    "bench",
    "brazier",
    "broken_pillar",
    "cage",
    "canopic_cluster",
    "chain",
    "chest",
    "fountain_rim",
    "hearth",
    "rubble_cluster",
    "sarcophagus",
    "sconce",
    "shelf",
    "urn_block",
];
const REQUIRED_LIGHT_IDS: [&str; 12] = [
    "brutalist_flood",
    "cavern_gloom",
    "cistern_cool",
    "cold_crypt",
    "dim_beam",
    "egyptian_amber",
    "entrance_torch",
    "foundry_fire",
    "grand_hall_grid",
    "shrine_focus",
    "treasury_glint",
    "warm_hall",
];

// ── Resource costs ─────────────────────────────────────────────────────────

/// Worst-case resource costs for an archetype, prop, light recipe, or theme.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResourceCosts {
    /// Estimated/measured source brush faces.
    pub source_faces: u32,
    /// Number of brushes.
    pub brushes: u32,
    /// Number of point entities.
    pub entities: u32,
    /// Number of light entities.
    pub lights: u32,
}

// ── Shape rule ─────────────────────────────────────────────────────────────

/// The geometric footprint family for an archetype room.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ShapeRule {
    /// Axis-aligned rectangular room.
    Rectangle,
    /// Cardinal + 45° chamfered octagonal footprint.
    Octagon,
    /// Single chamfer cut on one or more corners.
    Chamfer,
    /// Composite partitioned footprint (multiple sub-volumes).
    #[serde(rename = "CompositePartition")]
    CompositePartition,
}

impl ShapeRule {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Rectangle => "Rectangle",
            Self::Octagon => "Octagon",
            Self::Chamfer => "Chamfer",
            Self::CompositePartition => "CompositePartition",
        }
    }
}

// ── Layer occupancy ────────────────────────────────────────────────────────

/// Which Z layer(s) this archetype may occupy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LayerOccupancy {
    /// Lower layer only (Z = 0..176).
    Lower,
    /// Upper layer only (Z = 192..368).
    Upper,
    /// Both layers (composite multi-storey reservation).
    Both,
}

impl LayerOccupancy {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Lower => "Lower",
            Self::Upper => "Upper",
            Self::Both => "Both",
        }
    }
}

// ── Rarity tier ────────────────────────────────────────────────────────────

/// Normalized rarity with approximate selection weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RarityTier {
    /// ~70% of selections, repeatable after exhaustion.
    Common,
    /// ~25% of selections.
    Uncommon,
    /// ~5% of selections, no-repeat, one-per-map cap.
    Rare,
    /// ~1% of selections, no-repeat, one-per-map cap.
    Legendary,
}

impl RarityTier {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Common => "Common",
            Self::Uncommon => "Uncommon",
            Self::Rare => "Rare",
            Self::Legendary => "Legendary",
        }
    }
}

// ── Vertical recipe ────────────────────────────────────────────────────────

/// Vertical feature attached to (or integral with) an archetype.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum VerticalRecipe {
    /// No vertical feature.
    None,
    /// Type A or B 12-tread stairwell.
    Stairwell,
    /// Ladder shaft for climb traversal.
    #[serde(rename = "LadderShaft")]
    LadderShaft,
    /// One-way drop hole.
    #[serde(rename = "DropHole")]
    DropHole,
    /// Open (un-laddered) stairwell opening.
    #[serde(rename = "OpenStairwell")]
    OpenStairwell,
    /// 12-step spiral stair.
    #[serde(rename = "SpiralStair")]
    SpiralStair,
}

impl VerticalRecipe {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::None => "None",
            Self::Stairwell => "Stairwell",
            Self::LadderShaft => "LadderShaft",
            Self::DropHole => "DropHole",
            Self::OpenStairwell => "OpenStairwell",
            Self::SpiralStair => "SpiralStair",
        }
    }
}

// ── Collision behavior ─────────────────────────────────────────────────────

/// Whether a prop is collidable or detail-only.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CollisionBehavior {
    /// Collidable: emits solid brushes / clip hull.
    Collidable,
    /// Detail-only: visual only, no collision.
    #[serde(rename = "DetailOnly")]
    DetailOnly,
}

impl CollisionBehavior {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Collidable => "Collidable",
            Self::DetailOnly => "DetailOnly",
        }
    }
}

// ── Placement class ────────────────────────────────────────────────────────

/// Where a light entity is placed relative to room geometry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PlacementClass {
    /// Wall-mounted sconce/torch light.
    Wall,
    /// Ceiling-mounted light.
    Ceiling,
    /// Floor-standing light.
    Floor,
    /// Hanging pendant light.
    Pendant,
    /// Free-floating ambient volume light.
    Ambient,
}

impl PlacementClass {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Wall => "Wall",
            Self::Ceiling => "Ceiling",
            Self::Floor => "Floor",
            Self::Pendant => "Pendant",
            Self::Ambient => "Ambient",
        }
    }
}

// ── Light falloff style ────────────────────────────────────────────────────

/// Light attenuation model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FalloffStyle {
    /// Standard Quake linear falloff.
    Linear,
    /// Inverse-square falloff.
    #[serde(rename = "InverseSquare")]
    InverseSquare,
}

impl FalloffStyle {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Linear => "Linear",
            Self::InverseSquare => "InverseSquare",
        }
    }
}

// ── Theme geometry variant (per-archetype) ─────────────────────────────────

/// A per-theme geometry choice within a single archetype's contract.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ThemeGeometryVariant {
    /// Theme identifier: "ancient", "egyptian", or "brutalist".
    pub theme: String,
    /// Semantic massing descriptor.
    pub massing: String,
    /// Material role names used by this variant.
    pub materials: Vec<String>,
    /// Prop references specific to this theme variant.
    pub props: Vec<String>,
    /// Light references specific to this theme variant.
    pub lights: Vec<String>,
    /// Support rule descriptor (e.g. "grounded_floor_wall").
    pub support_data: String,
}

// ── Material role ──────────────────────────────────────────────────────────

/// A named material role with associated texture identity.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MaterialRole {
    /// Semantic role name (e.g. "wall_stone", "floor_slab").
    pub role: String,
    /// Texture basename referenced in the WAD.
    pub texture: String,
}

// ── Prop theme variant ─────────────────────────────────────────────────────

/// Per-theme overrides for a prop.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PropThemeVariant {
    /// Theme identifier.
    pub theme: String,
    /// Per-theme model/presentation override.
    pub model_override: String,
    /// Whether collision behavior changes per theme.
    pub collision_behavior: Option<CollisionBehavior>,
    /// Per-theme dimension override (if any).
    pub dimensions_override: Option<[u32; 3]>,
}

// ── Entity key-value pair ──────────────────────────────────────────────────

/// A typed entity key-value pair for light recipes.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EntityKeyValue {
    pub key: String,
    pub value: String,
}

// ── Archetype ──────────────────────────────────────────────────────────────

/// A single archetype: a gameplay-bearing room prefab with constraints,
/// theme variants, and worst-case resource costs.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Archetype {
    /// Stable unique identifier (snake_case).
    pub id: String,

    // ── Span constraints ───────────────────────────────────────────────
    /// Minimum outer XY span in Quake units [x, y].
    /// Each component >= 112 (7 quanta; 80-unit clear interior after walls).
    pub span_min: [u32; 2],
    /// Maximum outer XY span in Quake units [x, y].
    /// Each component <= 448 for EnhancedV3.
    pub span_max: [u32; 2],

    // ── Shape rule ─────────────────────────────────────────────────────
    /// Footprint geometry family.
    pub shape: ShapeRule,

    // ── Exit degree ────────────────────────────────────────────────────
    /// Minimum number of portal exits from this room.
    pub exit_degree_min: u32,
    /// Maximum number of portal exits from this room.
    pub exit_degree_max: u32,

    // ── Layer occupancy ────────────────────────────────────────────────
    /// Which Z layer(s) this archetype may occupy.
    pub layer_occupancy: LayerOccupancy,

    // ── Route witness envelope ─────────────────────────────────────────
    /// Minimum clear width and height through this room's portals.
    /// [width, height] — typically [64, 80].
    pub route_witness_envelope: [u32; 2],

    // ── Vertical recipe ────────────────────────────────────────────────
    /// Vertical feature type, if any.
    pub vertical_recipe: VerticalRecipe,

    // ── Rarity ─────────────────────────────────────────────────────────
    /// Selection rarity tier.
    pub rarity: RarityTier,

    // ── Zone compatibility ─────────────────────────────────────────────
    /// Zone names this archetype can be placed in.
    pub zone_compatibility: Vec<String>,

    // ── Grammar compatibility ──────────────────────────────────────────
    /// Grammar family names compatible with this archetype.
    pub grammar_compatibility: Vec<String>,

    // ── Negative-space budget ──────────────────────────────────────────
    /// Maximum negative-space (void) faces allowed within this room.
    pub negative_space_budget: u32,

    // ── Prop references ────────────────────────────────────────────────
    /// Prop IDs that may be placed in this archetype.
    pub prop_references: Vec<String>,

    // ── Light references ───────────────────────────────────────────────
    /// Light recipe IDs that may be placed in this archetype.
    pub light_references: Vec<String>,

    // ── Support rules ──────────────────────────────────────────────────
    /// Descriptive support rule (e.g. "all_brushes_grounded").
    pub support_rules: String,

    // ── Theme geometry variants ────────────────────────────────────────
    /// Exactly 3 entries: one per theme (ancient, egyptian, brutalist).
    pub theme_variants: Vec<ThemeGeometryVariant>,

    // ── Material roles ─────────────────────────────────────────────────
    /// Semantic material roles used by this archetype.
    pub material_roles: Vec<MaterialRole>,

    // ── Resource costs ─────────────────────────────────────────────────
    /// Worst-case resource cost estimate.
    pub costs: ResourceCosts,
}

// ── Prop ───────────────────────────────────────────────────────────────────

/// A single authored prop: decorative entity with defined dimensions,
/// collision behavior, theme variants, and cost accounting.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Prop {
    /// Stable unique identifier (snake_case).
    pub id: String,

    /// Number of exact convex pieces that compose this prop.
    pub convex_pieces: u32,

    /// Bounding dimensions in Quake units [x, y, z].
    pub dimensions: [u32; 3],

    /// Collision behavior: collidable or detail-only.
    pub collision_behavior: CollisionBehavior,

    /// Theme-specific overrides.
    pub theme_variants: Vec<PropThemeVariant>,

    /// Swept occupancy volume [x, y, z] for placement validation.
    pub swept_occupancy: [u32; 3],

    /// Minimum number of support contacts with world geometry.
    pub support_contacts: u32,

    /// Light recipe IDs this prop may couple with (emit near).
    pub light_coupling: Vec<String>,

    /// Worst-case resource costs.
    pub costs: ResourceCosts,
}

// ── Light recipe ───────────────────────────────────────────────────────────

/// A single authored lighting recipe: entity key/value pairs, bounded
/// numeric/color values, placement class, style, falloff, readability
/// floor, count, and cost.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LightRecipe {
    /// Stable unique identifier (snake_case).
    pub id: String,

    /// Entity key/value pairs for the light entity definition.
    pub entity_keys: Vec<EntityKeyValue>,

    /// RGB color [r, g, b] in 0..255.
    pub color: [u8; 3],

    /// Intensity/brightness value (Quake `light` key).
    pub intensity: u32,

    /// Placement class.
    pub placement_class: PlacementClass,

    /// Falloff attenuation model.
    pub falloff: FalloffStyle,

    /// Minimum light level (readability floor) at the farthest point.
    pub readability_floor: u32,

    /// Maximum count of this light type per map.
    pub count: u32,

    /// Worst-case resource costs (per instance).
    pub costs: ResourceCosts,
}

// ── Theme ──────────────────────────────────────────────────────────────────

/// A single authored theme: semantic roles, transitions, geometry
/// vocabulary, material roles, prop/light compatibility, and shared
/// worst-case budgets.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Theme {
    /// Stable unique identifier (snake_case).
    pub id: String,

    /// Semantic role names (e.g. "wall", "floor", "ceiling", "accent").
    pub semantic_roles: Vec<String>,

    /// Transition role names (e.g. "portal_surround", "corridor_trim").
    pub transitions: Vec<String>,

    /// Geometry vocabulary tokens available to this theme.
    pub geometry_vocabulary: Vec<String>,

    /// Named material roles with texture assignments.
    pub material_roles: Vec<MaterialRole>,

    /// Prop IDs compatible with this theme.
    pub prop_compatibility: Vec<String>,

    /// Light recipe IDs compatible with this theme.
    pub light_compatibility: Vec<String>,

    /// Shared worst-case budget ceiling for this theme.
    pub budget: ResourceCosts,
}

// ── Spiral step ────────────────────────────────────────────────────────────

/// One tread of the 12-step spiral template.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpiralStep {
    /// Step index (1–12, bottom to top).
    pub step_index: u32,

    /// Rise in Quake units (exactly 16 per step).
    pub rise: u32,

    /// Per-step XY envelope [x, y] within the 224×224 minimum.
    pub envelope: [u32; 2],

    /// Center column dimensions [x, y] (32×32).
    pub center_column: [u32; 2],

    /// Radial tread depth (64 units).
    pub tread_depth: u32,

    /// True if this tread is a supported convex recipe.
    pub is_convex_recipe: bool,
}

// ── Spiral template ────────────────────────────────────────────────────────

/// Complete 12-step spiral template with global constraints.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpiralTemplate {
    /// Exactly 12 steps, bottom to top.
    pub steps: Vec<SpiralStep>,

    /// Layer offset between lower and upper Z planes (192 units).
    pub layer_offset: u32,

    /// Minimum chamfered envelope for the entire spiral [x, y].
    pub envelope_min: [u32; 2],
}

// ── Full catalog container ─────────────────────────────────────────────────

/// The complete authored catalog read from RON files.
///
/// This is the deserialization target for each individual RON file.
/// The tool reads all files, merges into one catalog, and validates.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RichnessCatalog {
    /// Schema version declaration.
    pub schema_version: String,
    /// Archetype list (exactly 30).
    pub archetypes: Vec<Archetype>,
    /// Prop list (exactly 15).
    pub props: Vec<Prop>,
    /// Lighting recipe list (exactly 12).
    pub lighting: Vec<LightRecipe>,
    /// Theme list (exactly 3).
    pub themes: Vec<Theme>,
    /// Spiral template.
    pub spiral_template: SpiralTemplate,
}

/// Partial catalog for a single RON file.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArchetypesFile {
    pub schema_version: String,
    pub archetypes: Vec<Archetype>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PropsFile {
    pub schema_version: String,
    pub props: Vec<Prop>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LightingFile {
    pub schema_version: String,
    pub lighting: Vec<LightRecipe>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ThemesFile {
    pub schema_version: String,
    pub themes: Vec<Theme>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpiralFile {
    pub schema_version: String,
    pub spiral_template: SpiralTemplate,
}

// ── Validation ─────────────────────────────────────────────────────────────

/// All validation errors collected during catalog validation.
#[derive(Debug, Clone)]
pub struct ValidationErrors {
    pub errors: Vec<String>,
}

impl ValidationErrors {
    pub fn new() -> Self {
        Self { errors: Vec::new() }
    }

    pub fn add(&mut self, msg: impl Into<String>) {
        self.errors.push(msg.into());
    }

    pub fn is_empty(&self) -> bool {
        self.errors.is_empty()
    }
}

impl std::fmt::Display for ValidationErrors {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (i, err) in self.errors.iter().enumerate() {
            if i > 0 {
                writeln!(f)?;
            }
            write!(f, "  - {err}")?;
        }
        Ok(())
    }
}

impl RichnessCatalog {
    /// Validate the complete catalog against all rules.
    pub fn validate(&self) -> Result<(), ValidationErrors> {
        let mut errors = ValidationErrors::new();

        // Schema version check
        if self.schema_version != SCHEMA_VERSION {
            errors.add(format!(
                "schema_version '{}' does not match expected '{}'",
                self.schema_version, SCHEMA_VERSION
            ));
        }

        // ── Exact counts ───────────────────────────────────────────────
        self.validate_counts(&mut errors);

        // ── Unique stable IDs and required catalog closure ─────────────
        self.validate_unique_ids(&mut errors);
        self.validate_required_ids(&mut errors);

        // ── Canonical ordering ─────────────────────────────────────────
        self.validate_ordering(&mut errors);

        // ── Legal dimensions (>=16, quantum-aligned where structural) ──
        self.validate_dimensions(&mut errors);

        // ── Complete cost vectors ──────────────────────────────────────
        self.validate_costs(&mut errors);

        // ── Cross-references resolve ───────────────────────────────────
        self.validate_cross_references(&mut errors);

        // ── All-three-theme completeness and authored variation ────────
        self.validate_theme_completeness(&mut errors);
        self.validate_theme_variation(&mut errors);

        // ── Spiral template specific checks ────────────────────────────
        self.validate_spiral_template(&mut errors);

        // ── No undeclared/empty fields ─────────────────────────────────
        self.validate_non_empty_fields(&mut errors);

        if !errors.is_empty() {
            return Err(errors);
        }
        Ok(())
    }

    fn validate_counts(&self, errors: &mut ValidationErrors) {
        let expected = [
            ("archetypes", self.archetypes.len(), 30),
            ("props", self.props.len(), 15),
            ("lighting", self.lighting.len(), 12),
            ("themes", self.themes.len(), 3),
        ];
        for (name, actual, expected) in &expected {
            if *actual != *expected {
                errors.add(format!(
                    "{name}: expected exactly {expected} entries, got {actual}"
                ));
            }
        }
    }

    fn validate_unique_ids(&self, errors: &mut ValidationErrors) {
        // Archetypes
        {
            let mut seen = BTreeSet::new();
            for a in &self.archetypes {
                if !seen.insert(&a.id) {
                    errors.add(format!("duplicate archetype ID: '{}'", a.id));
                }
            }
        }
        // Props
        {
            let mut seen = BTreeSet::new();
            for p in &self.props {
                if !seen.insert(&p.id) {
                    errors.add(format!("duplicate prop ID: '{}'", p.id));
                }
            }
        }
        // Lighting
        {
            let mut seen = BTreeSet::new();
            for l in &self.lighting {
                if !seen.insert(&l.id) {
                    errors.add(format!("duplicate light recipe ID: '{}'", l.id));
                }
            }
        }
        // Themes
        {
            let mut seen = BTreeSet::new();
            for t in &self.themes {
                if !seen.insert(&t.id) {
                    errors.add(format!("duplicate theme ID: '{}'", t.id));
                }
            }
        }
    }

    fn validate_required_ids(&self, errors: &mut ValidationErrors) {
        self.validate_required_id_set(
            "archetypes",
            self.archetypes.iter().map(|item| item.id.as_str()),
            &REQUIRED_ARCHETYPE_IDS,
            errors,
        );
        self.validate_required_id_set(
            "props",
            self.props.iter().map(|item| item.id.as_str()),
            &REQUIRED_PROP_IDS,
            errors,
        );
        self.validate_required_id_set(
            "lighting",
            self.lighting.iter().map(|item| item.id.as_str()),
            &REQUIRED_LIGHT_IDS,
            errors,
        );
    }

    fn validate_required_id_set<'a>(
        &self,
        catalog_name: &str,
        actual: impl Iterator<Item = &'a str>,
        required: &[&str],
        errors: &mut ValidationErrors,
    ) {
        let actual: BTreeSet<_> = actual.collect();
        let required: BTreeSet<_> = required.iter().copied().collect();
        if actual != required {
            let missing: Vec<_> = required.difference(&actual).copied().collect();
            let unexpected: Vec<_> = actual.difference(&required).copied().collect();
            errors.add(format!(
                "{catalog_name}: required stable IDs mismatch; missing {:?}, unexpected {:?}",
                missing, unexpected
            ));
        }
    }

    fn validate_ordering(&self, errors: &mut ValidationErrors) {
        // Archetypes must be in lexical order by ID
        for i in 1..self.archetypes.len() {
            if self.archetypes[i].id <= self.archetypes[i - 1].id {
                errors.add(format!(
                    "archetypes not in lexical order: '{}' appears after '{}'",
                    self.archetypes[i].id,
                    self.archetypes[i - 1].id
                ));
                break;
            }
        }
        for i in 1..self.props.len() {
            if self.props[i].id <= self.props[i - 1].id {
                errors.add(format!(
                    "props not in lexical order: '{}' appears after '{}'",
                    self.props[i].id,
                    self.props[i - 1].id
                ));
                break;
            }
        }
        for i in 1..self.lighting.len() {
            if self.lighting[i].id <= self.lighting[i - 1].id {
                errors.add(format!(
                    "lighting not in lexical order: '{}' appears after '{}'",
                    self.lighting[i].id,
                    self.lighting[i - 1].id
                ));
                break;
            }
        }
        for i in 1..self.themes.len() {
            if self.themes[i].id <= self.themes[i - 1].id {
                errors.add(format!(
                    "themes not in lexical order: '{}' appears after '{}'",
                    self.themes[i].id,
                    self.themes[i - 1].id
                ));
                break;
            }
        }
    }

    fn validate_dimensions(&self, errors: &mut ValidationErrors) {
        let quantum = CONSTRUCTION_QUANTUM;

        // Check archetype span ranges
        for a in &self.archetypes {
            for (axis, dim) in [("x", a.span_min[0]), ("y", a.span_min[1])] {
                if dim < MIN_DIMENSION {
                    errors.add(format!(
                        "archetype '{}': span_min.{} = {} is below minimum {}",
                        a.id, axis, dim, MIN_DIMENSION
                    ));
                }
                if dim % quantum != 0 {
                    errors.add(format!(
                        "archetype '{}': span_min.{} = {} is not quantum-aligned (quantum: {})",
                        a.id, axis, dim, quantum
                    ));
                }
            }
            for (axis, dim) in [("x", a.span_max[0]), ("y", a.span_max[1])] {
                if dim < MIN_DIMENSION {
                    errors.add(format!(
                        "archetype '{}': span_max.{} = {} is below minimum {}",
                        a.id, axis, dim, MIN_DIMENSION
                    ));
                }
                if dim % quantum != 0 {
                    errors.add(format!(
                        "archetype '{}': span_max.{} = {} is not quantum-aligned (quantum: {})",
                        a.id, axis, dim, quantum
                    ));
                }
            }
            if a.span_min[0] > a.span_max[0] || a.span_min[1] > a.span_max[1] {
                errors.add(format!(
                    "archetype '{}': span_min {:?} exceeds span_max {:?}",
                    a.id, a.span_min, a.span_max
                ));
            }
            // Route witness envelope
            for (axis, dim) in [
                ("width", a.route_witness_envelope[0]),
                ("height", a.route_witness_envelope[1]),
            ] {
                if dim < MIN_DIMENSION {
                    errors.add(format!(
                        "archetype '{}': route_witness_envelope.{} = {} is below minimum {}",
                        a.id, axis, dim, MIN_DIMENSION
                    ));
                }
                if dim % quantum != 0 {
                    errors.add(format!(
                        "archetype '{}': route_witness_envelope.{} = {} is not quantum-aligned (quantum: {})",
                        a.id, axis, dim, quantum
                    ));
                }
            }
        }

        // Check prop dimensions
        for p in &self.props {
            for (axis, dim) in [
                ("x", p.dimensions[0]),
                ("y", p.dimensions[1]),
                ("z", p.dimensions[2]),
            ] {
                if dim < MIN_DIMENSION {
                    errors.add(format!(
                        "prop '{}': dimensions.{} = {} is below minimum {}",
                        p.id, axis, dim, MIN_DIMENSION
                    ));
                }
                if dim % quantum != 0 {
                    errors.add(format!(
                        "prop '{}': dimensions.{} = {} is not quantum-aligned (quantum: {})",
                        p.id, axis, dim, quantum
                    ));
                }
            }
            for (axis, dim) in [
                ("x", p.swept_occupancy[0]),
                ("y", p.swept_occupancy[1]),
                ("z", p.swept_occupancy[2]),
            ] {
                if dim < MIN_DIMENSION || dim % quantum != 0 {
                    errors.add(format!(
                        "prop '{}': swept_occupancy.{} = {} must be >= {} and quantum-aligned",
                        p.id, axis, dim, MIN_DIMENSION
                    ));
                }
            }
            for variant in &p.theme_variants {
                let Some(dimensions) = variant.dimensions_override else {
                    errors.add(format!(
                        "prop '{}' theme '{}': dimensions_override is required for concrete theme massing",
                        p.id, variant.theme
                    ));
                    continue;
                };
                for (axis, dim, swept) in [
                    ("x", dimensions[0], p.swept_occupancy[0]),
                    ("y", dimensions[1], p.swept_occupancy[1]),
                    ("z", dimensions[2], p.swept_occupancy[2]),
                ] {
                    if dim < MIN_DIMENSION || dim % quantum != 0 || dim > swept {
                        errors.add(format!(
                            "prop '{}' theme '{}': dimensions_override.{} = {} must be quantum-aligned, >= {}, and within swept occupancy {}",
                            p.id, variant.theme, axis, dim, MIN_DIMENSION, swept
                        ));
                    }
                }
            }
        }

        // Check spiral template dimensions
        let st = &self.spiral_template;
        if st.layer_offset < MIN_DIMENSION || st.layer_offset % quantum != 0 {
            errors.add(format!(
                "spiral_template: layer_offset {} must be >= {} and quantum-aligned",
                st.layer_offset, MIN_DIMENSION
            ));
        }
        if st.envelope_min[0] < MIN_DIMENSION || st.envelope_min[1] < MIN_DIMENSION {
            errors.add(format!(
                "spiral_template: envelope_min {:?} must be >= {}",
                st.envelope_min, MIN_DIMENSION
            ));
        }
        if st.envelope_min[0] % quantum != 0 || st.envelope_min[1] % quantum != 0 {
            errors.add(format!(
                "spiral_template: envelope_min {:?} must be quantum-aligned",
                st.envelope_min
            ));
        }
    }

    fn validate_costs(&self, errors: &mut ValidationErrors) {
        for a in &self.archetypes {
            self.check_cost_nonzero(&a.costs, &format!("archetype '{}'", a.id), errors);
        }
        for p in &self.props {
            self.check_cost_nonzero(&p.costs, &format!("prop '{}'", p.id), errors);
        }
        for l in &self.lighting {
            self.check_cost_nonzero(&l.costs, &format!("light recipe '{}'", l.id), errors);
        }
        for t in &self.themes {
            self.check_cost_nonzero(&t.budget, &format!("theme '{}'", t.id), errors);
        }
    }

    fn check_cost_nonzero(
        &self,
        costs: &ResourceCosts,
        label: &str,
        errors: &mut ValidationErrors,
    ) {
        if costs.source_faces == 0 && costs.brushes == 0 && costs.entities == 0 && costs.lights == 0
        {
            errors.add(format!(
                "{label}: cost vector is all-zeros — every item must have declared costs"
            ));
        }
        if costs.source_faces > 8_000
            || costs.brushes > 480
            || costs.entities > 300
            || costs.lights > 80
        {
            errors.add(format!(
                "{label}: cost vector exceeds Richness V1 sanity ceilings"
            ));
        }
        if costs.source_faces > 0 && costs.brushes == 0 {
            errors.add(format!(
                "{label}: source_faces requires a non-zero brush cost"
            ));
        }
        if costs.lights > 0 && costs.entities == 0 {
            errors.add(format!(
                "{label}: light cost requires a non-zero entity cost"
            ));
        }
    }

    fn validate_cross_references(&self, errors: &mut ValidationErrors) {
        // Build lookup sets
        let prop_ids: BTreeSet<&str> = self.props.iter().map(|p| p.id.as_str()).collect();
        let light_ids: BTreeSet<&str> = self.lighting.iter().map(|l| l.id.as_str()).collect();
        let theme_ids: BTreeSet<&str> = self.themes.iter().map(|t| t.id.as_str()).collect();

        let valid_theme_ids: BTreeSet<&str> = ["ancient", "egyptian", "brutalist"]
            .iter()
            .copied()
            .collect();

        // Validate archetype cross-references
        for a in &self.archetypes {
            for prop_ref in &a.prop_references {
                if !prop_ids.contains(prop_ref.as_str()) {
                    errors.add(format!(
                        "archetype '{}': references unknown prop '{}'",
                        a.id, prop_ref
                    ));
                }
            }
            for light_ref in &a.light_references {
                if !light_ids.contains(light_ref.as_str()) {
                    errors.add(format!(
                        "archetype '{}': references unknown light recipe '{}'",
                        a.id, light_ref
                    ));
                }
            }
            // Validate theme variants completeness
            let mut variant_themes: BTreeSet<&str> = BTreeSet::new();
            for v in &a.theme_variants {
                if !valid_theme_ids.contains(v.theme.as_str()) {
                    errors.add(format!(
                        "archetype '{}': theme variant references unknown theme '{}'",
                        a.id, v.theme
                    ));
                }
                if !variant_themes.insert(&v.theme) {
                    errors.add(format!(
                        "archetype '{}': duplicate theme variant for '{}'",
                        a.id, v.theme
                    ));
                }
                let Some(theme) = self.themes.iter().find(|theme| theme.id == v.theme) else {
                    continue;
                };
                let theme_materials: BTreeSet<&str> = theme
                    .material_roles
                    .iter()
                    .map(|role| role.texture.as_str())
                    .collect();
                for material in &v.materials {
                    if !theme_materials.contains(material.as_str()) {
                        errors.add(format!(
                            "archetype '{}' theme '{}': material '{}' is not declared by that theme",
                            a.id, v.theme, material
                        ));
                    }
                }
                // Validate variant prop/light refs
                for prop_ref in &v.props {
                    if !prop_ids.contains(prop_ref.as_str()) {
                        errors.add(format!(
                            "archetype '{}' theme '{}': references unknown prop '{}'",
                            a.id, v.theme, prop_ref
                        ));
                    }
                }
                for light_ref in &v.lights {
                    if !light_ids.contains(light_ref.as_str()) {
                        errors.add(format!(
                            "archetype '{}' theme '{}': references unknown light recipe '{}'",
                            a.id, v.theme, light_ref
                        ));
                    }
                }
            }
        }

        // Validate prop cross-references
        for p in &self.props {
            for light_ref in &p.light_coupling {
                if !light_ids.contains(light_ref.as_str()) {
                    errors.add(format!(
                        "prop '{}': light_coupling references unknown light recipe '{}'",
                        p.id, light_ref
                    ));
                }
            }
            // Validate theme variants
            let mut variant_themes: BTreeSet<&str> = BTreeSet::new();
            for v in &p.theme_variants {
                if !valid_theme_ids.contains(v.theme.as_str()) {
                    errors.add(format!(
                        "prop '{}': theme variant references unknown theme '{}'",
                        p.id, v.theme
                    ));
                }
                if !variant_themes.insert(&v.theme) {
                    errors.add(format!(
                        "prop '{}': duplicate theme variant for '{}'",
                        p.id, v.theme
                    ));
                }
            }
        }

        // Validate theme cross-references
        for t in &self.themes {
            for prop_ref in &t.prop_compatibility {
                if !prop_ids.contains(prop_ref.as_str()) {
                    errors.add(format!(
                        "theme '{}': prop_compatibility references unknown prop '{}'",
                        t.id, prop_ref
                    ));
                }
            }
            for light_ref in &t.light_compatibility {
                if !light_ids.contains(light_ref.as_str()) {
                    errors.add(format!(
                        "theme '{}': light_compatibility references unknown light recipe '{}'",
                        t.id, light_ref
                    ));
                }
            }
        }

        // Validate theme IDs are the canonical three
        let expected_themes: BTreeSet<&str> = ["ancient", "egyptian", "brutalist"]
            .iter()
            .copied()
            .collect();
        if theme_ids != expected_themes {
            errors.add(format!(
                "themes must be exactly [ancient, egyptian, brutalist], got {:?}",
                theme_ids.iter().collect::<Vec<_>>()
            ));
        }
    }

    fn validate_theme_completeness(&self, errors: &mut ValidationErrors) {
        let required_themes: BTreeSet<&str> = ["ancient", "egyptian", "brutalist"]
            .iter()
            .copied()
            .collect();

        // Every archetype must have all three theme variants
        for a in &self.archetypes {
            let variant_themes: BTreeSet<&str> =
                a.theme_variants.iter().map(|v| v.theme.as_str()).collect();
            if variant_themes != required_themes {
                let missing: Vec<_> = required_themes
                    .difference(&variant_themes)
                    .copied()
                    .collect();
                errors.add(format!(
                    "archetype '{}': missing theme variants for {:?}",
                    a.id, missing
                ));
            }
        }

        // Every prop must have all three theme variants
        for p in &self.props {
            let variant_themes: BTreeSet<&str> =
                p.theme_variants.iter().map(|v| v.theme.as_str()).collect();
            if variant_themes != required_themes {
                let missing: Vec<_> = required_themes
                    .difference(&variant_themes)
                    .copied()
                    .collect();
                errors.add(format!(
                    "prop '{}': missing theme variants for {:?}",
                    p.id, missing
                ));
            }
        }
    }

    fn validate_theme_variation(&self, errors: &mut ValidationErrors) {
        for archetype in &self.archetypes {
            for left in 0..archetype.theme_variants.len() {
                for right in left + 1..archetype.theme_variants.len() {
                    let a = &archetype.theme_variants[left];
                    let b = &archetype.theme_variants[right];
                    for (field, equal) in [
                        ("massing", a.massing == b.massing),
                        ("materials", a.materials == b.materials),
                        ("props", a.props == b.props),
                        ("lights", a.lights == b.lights),
                        ("support_data", a.support_data == b.support_data),
                    ] {
                        if equal {
                            errors.add(format!(
                                "archetype '{}' themes '{}' and '{}': {field} must be distinct authored data",
                                archetype.id, a.theme, b.theme
                            ));
                        }
                    }
                }
            }
        }
        for prop in &self.props {
            for left in 0..prop.theme_variants.len() {
                for right in left + 1..prop.theme_variants.len() {
                    let a = &prop.theme_variants[left];
                    let b = &prop.theme_variants[right];
                    if a.dimensions_override == b.dimensions_override
                        && a.collision_behavior == b.collision_behavior
                    {
                        errors.add(format!(
                            "prop '{}' themes '{}' and '{}': concrete massing/collision data must be distinct",
                            prop.id, a.theme, b.theme
                        ));
                    }
                }
            }
        }
    }

    fn validate_spiral_template(&self, errors: &mut ValidationErrors) {
        let st = &self.spiral_template;

        // Exactly 12 steps
        if st.steps.len() != 12 {
            errors.add(format!(
                "spiral_template: expected exactly 12 steps, got {}",
                st.steps.len()
            ));
        }

        // Layer offset must be 192
        if st.layer_offset != 192 {
            errors.add(format!(
                "spiral_template: layer_offset must be 192, got {}",
                st.layer_offset
            ));
        }

        // Minimum envelope 224×224
        if st.envelope_min[0] < 224 || st.envelope_min[1] < 224 {
            errors.add(format!(
                "spiral_template: envelope_min {:?} must be at least [224, 224]",
                st.envelope_min
            ));
        }

        // Validate each step
        let mut indices_seen = BTreeSet::new();
        for step in &st.steps {
            if !indices_seen.insert(step.step_index) {
                errors.add(format!(
                    "spiral_template: duplicate step_index {}",
                    step.step_index
                ));
            }

            // Rise must be 16
            if step.rise != 16 {
                errors.add(format!(
                    "spiral_template step {}: rise must be 16, got {}",
                    step.step_index, step.rise
                ));
            }

            // Center column 32×32
            if step.center_column[0] != 32 || step.center_column[1] != 32 {
                errors.add(format!(
                    "spiral_template step {}: center_column must be [32, 32], got {:?}",
                    step.step_index, step.center_column
                ));
            }

            // Tread depth must be 64
            if step.tread_depth != 64 {
                errors.add(format!(
                    "spiral_template step {}: tread_depth must be 64, got {}",
                    step.step_index, step.tread_depth
                ));
            }

            // Must be a convex recipe
            if !step.is_convex_recipe {
                errors.add(format!(
                    "spiral_template step {}: is_convex_recipe must be true",
                    step.step_index
                ));
            }

            // Envelope within global minimum
            if step.envelope[0] < st.envelope_min[0] || step.envelope[1] < st.envelope_min[1] {
                errors.add(format!(
                    "spiral_template step {}: per-step envelope {:?} must be at least {:?}",
                    step.step_index, step.envelope, st.envelope_min
                ));
            }
        }

        // Ensure step indices 1..=12 are all present
        for i in 1..=12 {
            if !indices_seen.contains(&i) {
                errors.add(format!("spiral_template: missing step_index {i}"));
            }
        }
    }

    fn validate_non_empty_fields(&self, errors: &mut ValidationErrors) {
        for a in &self.archetypes {
            if a.support_rules.is_empty() {
                errors.add(format!("archetype '{}': support_rules is empty", a.id));
            }
            if a.material_roles.is_empty() {
                errors.add(format!("archetype '{}': material_roles is empty", a.id));
            }
            if a.zone_compatibility.is_empty() {
                errors.add(format!("archetype '{}': zone_compatibility is empty", a.id));
            }
            if a.grammar_compatibility.is_empty() {
                errors.add(format!(
                    "archetype '{}': grammar_compatibility is empty",
                    a.id
                ));
            }
            // Verify every theme variant has non-empty massing
            for v in &a.theme_variants {
                if v.massing.is_empty() {
                    errors.add(format!(
                        "archetype '{}' theme '{}': massing is empty",
                        a.id, v.theme
                    ));
                }
                if v.support_data.is_empty() {
                    errors.add(format!(
                        "archetype '{}' theme '{}': support_data is empty",
                        a.id, v.theme
                    ));
                }
                if v.materials.is_empty() || v.props.is_empty() || v.lights.is_empty() {
                    errors.add(format!(
                        "archetype '{}' theme '{}': materials, props, and lights must be concrete selections",
                        a.id, v.theme
                    ));
                }
            }
        }

        for p in &self.props {
            if p.convex_pieces == 0 {
                errors.add(format!("prop '{}': convex_pieces must be >= 1", p.id));
            }
            for variant in &p.theme_variants {
                if variant.model_override.is_empty() {
                    errors.add(format!(
                        "prop '{}' theme '{}': model_override is empty",
                        p.id, variant.theme
                    ));
                }
            }
        }

        for l in &self.lighting {
            if l.entity_keys.is_empty() {
                errors.add(format!("light recipe '{}': entity_keys is empty", l.id));
            }
            if l.intensity == 0 || l.intensity > 4_096 {
                errors.add(format!(
                    "light recipe '{}': intensity is outside 1..=4096",
                    l.id
                ));
            }
            if l.readability_floor == 0 || l.readability_floor > l.intensity {
                errors.add(format!(
                    "light recipe '{}': readability_floor must be in 1..=intensity",
                    l.id
                ));
            }
            if l.count == 0 || l.count > 300 {
                errors.add(format!("light recipe '{}': count is outside 1..=300", l.id));
            }
        }

        for t in &self.themes {
            if t.semantic_roles.is_empty() {
                errors.add(format!("theme '{}': semantic_roles is empty", t.id));
            }
            if t.material_roles.is_empty() {
                errors.add(format!("theme '{}': material_roles is empty", t.id));
            }
        }
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal valid catalog for testing validation negatives.
    fn minimal_valid_catalog() -> RichnessCatalog {
        let theme_ids = ["ancient", "egyptian", "brutalist"];
        let make_variants = |massing: &str| -> Vec<ThemeGeometryVariant> {
            theme_ids
                .iter()
                .enumerate()
                .map(|(index, theme)| ThemeGeometryVariant {
                    theme: (*theme).to_string(),
                    massing: format!("{massing}_{theme}"),
                    materials: vec![format!("{theme}_wall")],
                    props: vec![REQUIRED_PROP_IDS[index].to_string()],
                    lights: vec![REQUIRED_LIGHT_IDS[index].to_string()],
                    support_data: format!("grounded_{theme}"),
                })
                .collect()
        };

        let make_prop_variants = || -> Vec<PropThemeVariant> {
            theme_ids
                .iter()
                .enumerate()
                .map(|(index, theme)| PropThemeVariant {
                    theme: (*theme).to_string(),
                    model_override: format!("{theme}_model"),
                    collision_behavior: None,
                    dimensions_override: Some(match index {
                        0 => [32, 32, 32],
                        1 => [48, 32, 32],
                        _ => [32, 48, 32],
                    }),
                })
                .collect()
        };

        RichnessCatalog {
            schema_version: SCHEMA_VERSION.to_string(),
            archetypes: REQUIRED_ARCHETYPE_IDS
                .iter()
                .map(|id| Archetype {
                    id: (*id).to_string(),
                    span_min: [112, 112],
                    span_max: [448, 448],
                    shape: ShapeRule::Rectangle,
                    exit_degree_min: 1,
                    exit_degree_max: 4,
                    layer_occupancy: LayerOccupancy::Lower,
                    route_witness_envelope: [64, 80],
                    vertical_recipe: VerticalRecipe::None,
                    rarity: RarityTier::Common,
                    zone_compatibility: vec!["all".to_string()],
                    grammar_compatibility: vec!["default".to_string()],
                    negative_space_budget: 100,
                    prop_references: vec![],
                    light_references: vec![],
                    support_rules: "grounded".to_string(),
                    theme_variants: make_variants("default_massing"),
                    material_roles: vec![MaterialRole {
                        role: "wall".to_string(),
                        texture: "stone".to_string(),
                    }],
                    costs: ResourceCosts {
                        source_faces: 100,
                        brushes: 10,
                        entities: 1,
                        lights: 2,
                    },
                })
                .collect(),
            props: REQUIRED_PROP_IDS
                .iter()
                .map(|id| Prop {
                    id: (*id).to_string(),
                    convex_pieces: 1,
                    dimensions: [32, 32, 32],
                    collision_behavior: CollisionBehavior::Collidable,
                    theme_variants: make_prop_variants(),
                    swept_occupancy: [64, 64, 64],
                    support_contacts: 1,
                    light_coupling: vec![],
                    costs: ResourceCosts {
                        source_faces: 6,
                        brushes: 1,
                        entities: 1,
                        lights: 0,
                    },
                })
                .collect(),
            lighting: REQUIRED_LIGHT_IDS
                .iter()
                .map(|id| LightRecipe {
                    id: (*id).to_string(),
                    entity_keys: vec![EntityKeyValue {
                        key: "light".to_string(),
                        value: "200".to_string(),
                    }],
                    color: [255, 200, 150],
                    intensity: 200,
                    placement_class: PlacementClass::Wall,
                    falloff: FalloffStyle::Linear,
                    readability_floor: 10,
                    count: 50,
                    costs: ResourceCosts {
                        source_faces: 0,
                        brushes: 0,
                        entities: 1,
                        lights: 1,
                    },
                })
                .collect(),
            // Themes must be in lexical order: ancient, brutalist, egyptian
            themes: ["ancient", "brutalist", "egyptian"]
                .iter()
                .map(|t| Theme {
                    id: t.to_string(),
                    semantic_roles: vec!["wall".to_string()],
                    transitions: vec!["portal".to_string()],
                    geometry_vocabulary: vec!["rectangle".to_string()],
                    material_roles: vec![MaterialRole {
                        role: "wall".to_string(),
                        texture: format!("{t}_wall"),
                    }],
                    prop_compatibility: vec![],
                    light_compatibility: vec![],
                    budget: ResourceCosts {
                        source_faces: 5000,
                        brushes: 480,
                        entities: 0,
                        lights: 0,
                    },
                })
                .collect(),
            spiral_template: SpiralTemplate {
                steps: (1..=12)
                    .map(|i| SpiralStep {
                        step_index: i,
                        rise: 16,
                        envelope: [224, 224],
                        center_column: [32, 32],
                        tread_depth: 64,
                        is_convex_recipe: true,
                    })
                    .collect(),
                layer_offset: 192,
                envelope_min: [224, 224],
            },
        }
    }

    #[test]
    fn minimal_valid_catalog_passes() {
        let cat = minimal_valid_catalog();
        assert!(cat.validate().is_ok());
    }

    // ── Count tests ────────────────────────────────────────────────────

    #[test]
    fn rejects_wrong_archetype_count() {
        let mut cat = minimal_valid_catalog();
        cat.archetypes.pop();
        let err = cat.validate().unwrap_err();
        assert!(err.errors.iter().any(|e| e.contains("archetypes")));
    }

    #[test]
    fn rejects_wrong_prop_count() {
        let mut cat = minimal_valid_catalog();
        cat.props.pop();
        let err = cat.validate().unwrap_err();
        assert!(err.errors.iter().any(|e| e.contains("props")));
    }

    #[test]
    fn rejects_wrong_light_count() {
        let mut cat = minimal_valid_catalog();
        cat.lighting.pop();
        let err = cat.validate().unwrap_err();
        assert!(err.errors.iter().any(|e| e.contains("lighting")));
    }

    #[test]
    fn rejects_wrong_theme_count() {
        let mut cat = minimal_valid_catalog();
        cat.themes.pop();
        let err = cat.validate().unwrap_err();
        assert!(err.errors.iter().any(|e| e.contains("themes")));
    }

    // ── Duplicate ID tests ─────────────────────────────────────────────

    #[test]
    fn rejects_duplicate_archetype_ids() {
        let mut cat = minimal_valid_catalog();
        cat.archetypes[1].id = cat.archetypes[0].id.clone();
        let err = cat.validate().unwrap_err();
        assert!(err.errors.iter().any(|e| e.contains("duplicate archetype")));
    }

    #[test]
    fn rejects_duplicate_prop_ids() {
        let mut cat = minimal_valid_catalog();
        cat.props[1].id = cat.props[0].id.clone();
        let err = cat.validate().unwrap_err();
        assert!(err.errors.iter().any(|e| e.contains("duplicate prop")));
    }

    #[test]
    fn rejects_duplicate_light_ids() {
        let mut cat = minimal_valid_catalog();
        cat.lighting[1].id = cat.lighting[0].id.clone();
        let err = cat.validate().unwrap_err();
        assert!(err.errors.iter().any(|e| e.contains("duplicate light")));
    }

    #[test]
    fn rejects_duplicate_theme_ids() {
        let mut cat = minimal_valid_catalog();
        cat.themes[1].id = cat.themes[0].id.clone();
        let err = cat.validate().unwrap_err();
        assert!(err.errors.iter().any(|e| e.contains("duplicate theme")));
    }

    // ── Ordering tests ─────────────────────────────────────────────────

    #[test]
    fn rejects_non_lexical_archetype_order() {
        let mut cat = minimal_valid_catalog();
        cat.archetypes.swap(0, 1);
        let err = cat.validate().unwrap_err();
        assert!(err
            .errors
            .iter()
            .any(|e| e.contains("not in lexical order")));
    }

    // ── Dimension tests ────────────────────────────────────────────────

    #[test]
    fn rejects_below_min_span() {
        let mut cat = minimal_valid_catalog();
        cat.archetypes[0].span_min = [15, 112];
        let err = cat.validate().unwrap_err();
        assert!(err.errors.iter().any(|e| e.contains("below minimum")));
    }

    #[test]
    fn rejects_non_quantum_span() {
        let mut cat = minimal_valid_catalog();
        cat.archetypes[0].span_min = [113, 112];
        let err = cat.validate().unwrap_err();
        assert!(err.errors.iter().any(|e| e.contains("not quantum-aligned")));
    }

    #[test]
    fn rejects_below_min_prop_dimension() {
        let mut cat = minimal_valid_catalog();
        cat.props[0].dimensions = [15, 32, 32];
        let err = cat.validate().unwrap_err();
        assert!(err.errors.iter().any(|e| e.contains("below minimum")));
    }

    // ── Cost tests ─────────────────────────────────────────────────────

    #[test]
    fn rejects_zero_cost_archetype() {
        let mut cat = minimal_valid_catalog();
        cat.archetypes[0].costs = ResourceCosts {
            source_faces: 0,
            brushes: 0,
            entities: 0,
            lights: 0,
        };
        let err = cat.validate().unwrap_err();
        assert!(err.errors.iter().any(|e| e.contains("all-zeros")));
    }

    // ── Cross-reference tests ──────────────────────────────────────────

    #[test]
    fn rejects_unknown_prop_ref_in_archetype() {
        let mut cat = minimal_valid_catalog();
        cat.archetypes[0]
            .prop_references
            .push("nonexistent_prop".to_string());
        let err = cat.validate().unwrap_err();
        assert!(err.errors.iter().any(|e| e.contains("unknown prop")));
    }

    #[test]
    fn rejects_unknown_light_ref_in_archetype() {
        let mut cat = minimal_valid_catalog();
        cat.archetypes[0]
            .light_references
            .push("nonexistent_light".to_string());
        let err = cat.validate().unwrap_err();
        assert!(err
            .errors
            .iter()
            .any(|e| e.contains("unknown light recipe")));
    }

    #[test]
    fn rejects_bad_theme_variant_theme_ref() {
        let mut cat = minimal_valid_catalog();
        cat.archetypes[0].theme_variants[0].theme = "gothic".to_string();
        let err = cat.validate().unwrap_err();
        assert!(err.errors.iter().any(|e| e.contains("unknown theme")));
    }

    #[test]
    fn rejects_unknown_prop_ref_in_theme() {
        let mut cat = minimal_valid_catalog();
        cat.themes[0]
            .prop_compatibility
            .push("nonexistent".to_string());
        let err = cat.validate().unwrap_err();
        assert!(err.errors.iter().any(|e| e.contains("unknown prop")));
    }

    // ── Theme completeness tests ───────────────────────────────────────

    #[test]
    fn rejects_missing_theme_variant() {
        let mut cat = minimal_valid_catalog();
        cat.archetypes[0].theme_variants.pop();
        let err = cat.validate().unwrap_err();
        assert!(err
            .errors
            .iter()
            .any(|e| e.contains("missing theme variants")));
    }

    #[test]
    fn rejects_duplicate_theme_variant_per_archetype() {
        let mut cat = minimal_valid_catalog();
        cat.archetypes[0].theme_variants.pop();
        cat.archetypes[0].theme_variants.push(ThemeGeometryVariant {
            theme: "ancient".to_string(),
            massing: "dup".to_string(),
            materials: vec![],
            props: vec![],
            lights: vec![],
            support_data: "grounded".to_string(),
        });
        let err = cat.validate().unwrap_err();
        assert!(err
            .errors
            .iter()
            .any(|e| e.contains("duplicate theme variant")));
    }

    #[test]
    fn rejects_label_only_archetype_theme_variant() {
        let mut cat = minimal_valid_catalog();
        cat.archetypes[0].theme_variants[1].materials =
            cat.archetypes[0].theme_variants[0].materials.clone();
        let err = cat.validate().unwrap_err();
        assert!(err
            .errors
            .iter()
            .any(|e| e.contains("materials must be distinct")));
    }

    #[test]
    fn rejects_label_only_prop_theme_variant() {
        let mut cat = minimal_valid_catalog();
        cat.props[0].theme_variants[1].dimensions_override =
            cat.props[0].theme_variants[0].dimensions_override;
        let err = cat.validate().unwrap_err();
        assert!(err
            .errors
            .iter()
            .any(|e| e.contains("concrete massing/collision data must be distinct")));
    }

    #[test]
    fn rejects_cost_above_richness_ceiling() {
        let mut cat = minimal_valid_catalog();
        cat.archetypes[0].costs.source_faces = 8_001;
        let err = cat.validate().unwrap_err();
        assert!(err.errors.iter().any(|e| e.contains("sanity ceilings")));
    }

    // ── Spiral template tests ──────────────────────────────────────────

    #[test]
    fn rejects_wrong_step_count() {
        let mut cat = minimal_valid_catalog();
        cat.spiral_template.steps.pop();
        let err = cat.validate().unwrap_err();
        assert!(err
            .errors
            .iter()
            .any(|e| e.contains("expected exactly 12 steps")));
    }

    #[test]
    fn rejects_wrong_layer_offset() {
        let mut cat = minimal_valid_catalog();
        cat.spiral_template.layer_offset = 200;
        let err = cat.validate().unwrap_err();
        assert!(err
            .errors
            .iter()
            .any(|e| e.contains("layer_offset must be 192")));
    }

    #[test]
    fn rejects_small_envelope() {
        let mut cat = minimal_valid_catalog();
        cat.spiral_template.envelope_min = [128, 128];
        let err = cat.validate().unwrap_err();
        assert!(err.errors.iter().any(|e| e.contains("envelope_min")));
    }

    #[test]
    fn rejects_wrong_rise() {
        let mut cat = minimal_valid_catalog();
        cat.spiral_template.steps[0].rise = 32;
        let err = cat.validate().unwrap_err();
        assert!(err.errors.iter().any(|e| e.contains("rise must be 16")));
    }

    #[test]
    fn rejects_wrong_center_column() {
        let mut cat = minimal_valid_catalog();
        cat.spiral_template.steps[0].center_column = [64, 64];
        let err = cat.validate().unwrap_err();
        assert!(err.errors.iter().any(|e| e.contains("center_column")));
    }

    #[test]
    fn rejects_wrong_tread_depth() {
        let mut cat = minimal_valid_catalog();
        cat.spiral_template.steps[0].tread_depth = 32;
        let err = cat.validate().unwrap_err();
        assert!(err
            .errors
            .iter()
            .any(|e| e.contains("tread_depth must be 64")));
    }

    #[test]
    fn rejects_non_convex_step() {
        let mut cat = minimal_valid_catalog();
        cat.spiral_template.steps[0].is_convex_recipe = false;
        let err = cat.validate().unwrap_err();
        assert!(err
            .errors
            .iter()
            .any(|e| e.contains("is_convex_recipe must be true")));
    }

    #[test]
    fn rejects_duplicate_step_index() {
        let mut cat = minimal_valid_catalog();
        cat.spiral_template.steps[11].step_index = 1;
        let err = cat.validate().unwrap_err();
        assert!(err
            .errors
            .iter()
            .any(|e| e.contains("duplicate step_index")));
    }

    #[test]
    fn rejects_missing_step_index() {
        let mut cat = minimal_valid_catalog();
        cat.spiral_template.steps[11].step_index = 13;
        let err = cat.validate().unwrap_err();
        assert!(err.errors.iter().any(|e| e.contains("missing step_index")));
    }

    // ── Non-empty field tests ──────────────────────────────────────────

    #[test]
    fn rejects_empty_support_rules() {
        let mut cat = minimal_valid_catalog();
        cat.archetypes[0].support_rules = String::new();
        let err = cat.validate().unwrap_err();
        assert!(err
            .errors
            .iter()
            .any(|e| e.contains("support_rules is empty")));
    }

    #[test]
    fn rejects_empty_material_roles() {
        let mut cat = minimal_valid_catalog();
        cat.archetypes[0].material_roles.clear();
        let err = cat.validate().unwrap_err();
        assert!(err
            .errors
            .iter()
            .any(|e| e.contains("material_roles is empty")));
    }

    #[test]
    fn rejects_zero_convex_pieces() {
        let mut cat = minimal_valid_catalog();
        cat.props[0].convex_pieces = 0;
        let err = cat.validate().unwrap_err();
        assert!(err
            .errors
            .iter()
            .any(|e| e.contains("convex_pieces must be >= 1")));
    }

    #[test]
    fn rejects_empty_entity_keys() {
        let mut cat = minimal_valid_catalog();
        cat.lighting[0].entity_keys.clear();
        let err = cat.validate().unwrap_err();
        assert!(err
            .errors
            .iter()
            .any(|e| e.contains("entity_keys is empty")));
    }

    #[test]
    fn rejects_zero_intensity() {
        let mut cat = minimal_valid_catalog();
        cat.lighting[0].intensity = 0;
        let err = cat.validate().unwrap_err();
        assert!(err
            .errors
            .iter()
            .any(|e| e.contains("intensity is outside")));
    }

    #[test]
    fn rejects_bad_schema_version() {
        let mut cat = minimal_valid_catalog();
        cat.schema_version = "wrong/v1".to_string();
        let err = cat.validate().unwrap_err();
        assert!(err.errors.iter().any(|e| e.contains("schema_version")));
    }

    #[test]
    fn rejects_empty_theme_roles() {
        let mut cat = minimal_valid_catalog();
        cat.themes[0].semantic_roles.clear();
        let err = cat.validate().unwrap_err();
        assert!(err
            .errors
            .iter()
            .any(|e| e.contains("semantic_roles is empty")));
    }

    // ── Schema version constant ────────────────────────────────────────

    #[test]
    fn schema_version_is_frozen() {
        assert_eq!(SCHEMA_VERSION, "enhanced-v3-richness-content/v1");
    }

    #[test]
    fn construction_quantum_is_16() {
        assert_eq!(CONSTRUCTION_QUANTUM, 16);
    }
}
