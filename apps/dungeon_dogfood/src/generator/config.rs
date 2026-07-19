use serde::Serialize;
use sha2::{Digest, Sha256};

use super::determinism::lowercase_hex;
use super::error::{ErrorStage, GeneratorError};

const HARD_MAX_TILES: u64 = 65_536;
const HARD_MAX_LIGHTS: u32 = 16;
const HARD_MAX_CHUNKS: u32 = 512;
const HARD_MAX_STATIC_BODIES: u32 = 512;
const HARD_MAX_TOTAL_BODIES: u32 = 513;
const HARD_MAX_VERTICES: u32 = 1_600_000;
const HARD_MAX_INDICES: u32 = 2_400_000;
const CONFIG_FORMAT_TAG: &[u8] = b"dungeon-generator/config/v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum QualifiedProfile {
    Minimum,
    Primary,
    Maximum,
}

impl QualifiedProfile {
    const fn code(self) -> u8 {
        match self {
            Self::Minimum => 0,
            Self::Primary => 1,
            Self::Maximum => 2,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum Qualification {
    Qualified,
    Custom,
    SingleBottleneck,
}

impl Qualification {
    const fn code(self) -> u8 {
        match self {
            Self::Qualified => 0,
            Self::Custom => 1,
            Self::SingleBottleneck => 2,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub(super) struct GeneratorConfig {
    pub(super) profile: Option<QualifiedProfile>,
    pub(super) width: Option<u64>,
    pub(super) height: Option<u64>,
    pub(super) layers: Option<u64>,
    pub(super) single_bottleneck: bool,
    pub(super) relax_route_redundancy: bool,
    pub(super) relax_transition_redundancy: bool,
    pub(super) region_min: Option<u32>,
    pub(super) region_max: Option<u32>,
    pub(super) required_route_min: Option<u32>,
    pub(super) required_route_max: Option<u32>,
    pub(super) per_layer_cycles_min: Option<u32>,
    pub(super) per_layer_cycles_max: Option<u32>,
    pub(super) branch_depth_min: Option<u32>,
    pub(super) branch_depth_max: Option<u32>,
    pub(super) articulation_max: Option<u32>,
    pub(super) intentional_dead_ends_min: Option<u32>,
    pub(super) intentional_dead_ends_max: Option<u32>,
    pub(super) optional_mergers_max: Option<u32>,
    pub(super) optional_shortcuts_max: Option<u32>,
    pub(super) crossings_max: Option<u32>,
    pub(super) components_max: Option<u32>,
    pub(super) edge_disjoint_routes: Option<u32>,
    pub(super) transitions_per_adjacent_pair: Option<u32>,
    pub(super) corridor_width: Option<u32>,
    pub(super) hall_width: Option<u32>,
    pub(super) spacing: Option<u32>,
    pub(super) placement_attempts: Option<u32>,
    pub(super) routing_attempts: Option<u32>,
    pub(super) generation_attempts: Option<u32>,
    pub(super) reroute_budget: Option<u32>,
    pub(super) marker_relocation_budget: Option<u32>,
    pub(super) optional_edge_removal_budget: Option<u32>,
    pub(super) ordinary_prefab_ratio_numerator: Option<u32>,
    pub(super) ordinary_prefab_ratio_denominator: Option<u32>,
    pub(super) model_marker_cap: Option<u32>,
    pub(super) max_lights: Option<u32>,
    pub(super) max_chunks: Option<u32>,
    pub(super) max_static_bodies: Option<u32>,
    pub(super) max_total_bodies: Option<u32>,
    pub(super) max_vertices: Option<u32>,
    pub(super) max_indices: Option<u32>,
    pub(super) max_tiles: Option<u32>,
}

impl GeneratorConfig {
    pub(super) fn qualified(profile: QualifiedProfile) -> Self {
        Self {
            profile: Some(profile),
            ..Self::default()
        }
    }

    pub(super) fn custom(width: u64, height: u64, layers: u64) -> Self {
        Self {
            width: Some(width),
            height: Some(height),
            layers: Some(layers),
            ..Self::default()
        }
    }

    pub(super) fn normalize(&self) -> Result<NormalizedGeneratorConfig, GeneratorError> {
        NormalizedGeneratorConfig::from_raw(self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ProfileValues {
    width: u16,
    height: u16,
    layers: u16,
    region_min: u32,
    region_max: u32,
    required_route_min: u32,
    required_route_max: u32,
    per_layer_cycles_min: u32,
    per_layer_cycles_max: u32,
    branch_depth_min: u32,
    branch_depth_max: u32,
    articulation_max: u32,
    intentional_dead_ends_min: u32,
    intentional_dead_ends_max: u32,
    optional_mergers_max: u32,
    optional_shortcuts_max: u32,
    crossings_max: u32,
    components_max: u32,
    edge_disjoint_routes: u32,
    transitions_per_adjacent_pair: u32,
    corridor_width: u32,
    hall_width: u32,
    spacing: u32,
    placement_attempts: u32,
    routing_attempts: u32,
    generation_attempts: u32,
    reroute_budget: u32,
    marker_relocation_budget: u32,
    optional_edge_removal_budget: u32,
    ordinary_prefab_ratio_numerator: u32,
    ordinary_prefab_ratio_denominator: u32,
    model_marker_cap: u32,
    max_lights: u32,
    max_chunks: u32,
    max_static_bodies: u32,
    max_total_bodies: u32,
    max_vertices: u32,
    max_indices: u32,
    max_tiles: u32,
}

const PRESETS: [ProfileValues; 3] = [
    ProfileValues {
        width: 64, height: 64, layers: 2, region_min: 14, region_max: 24,
        required_route_min: 70, required_route_max: 180, per_layer_cycles_min: 1,
        per_layer_cycles_max: 6, branch_depth_min: 4, branch_depth_max: 16,
        articulation_max: 10, intentional_dead_ends_min: 2, intentional_dead_ends_max: 10,
        optional_mergers_max: 8, optional_shortcuts_max: 8, crossings_max: 6,
        components_max: 1, edge_disjoint_routes: 2, transitions_per_adjacent_pair: 2,
        corridor_width: 1, hall_width: 2, spacing: 2, placement_attempts: 256,
        routing_attempts: 128, generation_attempts: 64, reroute_budget: 32,
        marker_relocation_budget: 24, optional_edge_removal_budget: 16,
        ordinary_prefab_ratio_numerator: 3, ordinary_prefab_ratio_denominator: 5,
        model_marker_cap: 64, max_lights: 16, max_chunks: 128, max_static_bodies: 128,
        max_total_bodies: 129, max_vertices: 400_000, max_indices: 600_000,
        max_tiles: 65_536,
    },
    ProfileValues {
        width: 96, height: 96, layers: 3, region_min: 24, region_max: 40,
        required_route_min: 140, required_route_max: 320, per_layer_cycles_min: 1,
        per_layer_cycles_max: 10, branch_depth_min: 6, branch_depth_max: 24,
        articulation_max: 16, intentional_dead_ends_min: 3, intentional_dead_ends_max: 16,
        optional_mergers_max: 12, optional_shortcuts_max: 12, crossings_max: 10,
        components_max: 1, edge_disjoint_routes: 2, transitions_per_adjacent_pair: 2,
        corridor_width: 1, hall_width: 2, spacing: 2, placement_attempts: 512,
        routing_attempts: 256, generation_attempts: 100, reroute_budget: 64,
        marker_relocation_budget: 48, optional_edge_removal_budget: 32,
        ordinary_prefab_ratio_numerator: 3, ordinary_prefab_ratio_denominator: 5,
        model_marker_cap: 128, max_lights: 16, max_chunks: 256, max_static_bodies: 256,
        max_total_bodies: 257, max_vertices: 800_000, max_indices: 1_200_000,
        max_tiles: 65_536,
    },
    ProfileValues {
        width: 128, height: 128, layers: 4, region_min: 40, region_max: 64,
        required_route_min: 220, required_route_max: 520, per_layer_cycles_min: 1,
        per_layer_cycles_max: 16, branch_depth_min: 8, branch_depth_max: 32,
        articulation_max: 24, intentional_dead_ends_min: 4, intentional_dead_ends_max: 24,
        optional_mergers_max: 20, optional_shortcuts_max: 20, crossings_max: 16,
        components_max: 1, edge_disjoint_routes: 2, transitions_per_adjacent_pair: 2,
        corridor_width: 1, hall_width: 2, spacing: 3, placement_attempts: 1_024,
        routing_attempts: 512, generation_attempts: 128, reroute_budget: 96,
        marker_relocation_budget: 72, optional_edge_removal_budget: 48,
        ordinary_prefab_ratio_numerator: 3, ordinary_prefab_ratio_denominator: 5,
        model_marker_cap: 256, max_lights: 16, max_chunks: 512, max_static_bodies: 512,
        max_total_bodies: 513, max_vertices: 1_600_000, max_indices: 2_400_000,
        max_tiles: 65_536,
    },
];

fn preset(profile: QualifiedProfile) -> ProfileValues {
    PRESETS[profile.code() as usize]
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(super) struct NormalizedGeneratorConfig {
    width: u16,
    height: u16,
    layers: u16,
    qualification: Qualification,
    source_profile: Option<QualifiedProfile>,
    relax_route_redundancy: bool,
    relax_transition_redundancy: bool,
    region_min: u32,
    region_max: u32,
    required_route_min: u32,
    required_route_max: u32,
    per_layer_cycles_min: u32,
    per_layer_cycles_max: u32,
    branch_depth_min: u32,
    branch_depth_max: u32,
    articulation_max: u32,
    intentional_dead_ends_min: u32,
    intentional_dead_ends_max: u32,
    optional_mergers_max: u32,
    optional_shortcuts_max: u32,
    crossings_max: u32,
    components_max: u32,
    edge_disjoint_routes: u32,
    transitions_per_adjacent_pair: u32,
    corridor_width: u32,
    hall_width: u32,
    spacing: u32,
    placement_attempts: u32,
    routing_attempts: u32,
    generation_attempts: u32,
    reroute_budget: u32,
    marker_relocation_budget: u32,
    optional_edge_removal_budget: u32,
    ordinary_prefab_ratio_numerator: u32,
    ordinary_prefab_ratio_denominator: u32,
    model_marker_cap: u32,
    max_lights: u32,
    max_chunks: u32,
    max_static_bodies: u32,
    max_total_bodies: u32,
    max_vertices: u32,
    max_indices: u32,
    max_tiles: u32,
}

impl NormalizedGeneratorConfig {
    fn from_raw(raw: &GeneratorConfig) -> Result<Self, GeneratorError> {
        validate_relaxation_mode(raw)?;
        let selected = raw.profile.unwrap_or(QualifiedProfile::Primary);
        let selected_values = preset(selected);
        let width = raw.width.unwrap_or(u64::from(selected_values.width));
        let height = raw.height.unwrap_or(u64::from(selected_values.height));
        let layers = raw.layers.unwrap_or(u64::from(selected_values.layers));
        let tile_count = width
            .checked_mul(height)
            .and_then(|value| value.checked_mul(layers))
            .ok_or(GeneratorError::ArithmeticOverflow {
                stage: ErrorStage::Configuration,
                operation: "tile_count",
            })?;
        validate_dimensions(width, height, layers, tile_count)?;
        let exact_dimensions = width == u64::from(selected_values.width)
            && height == u64::from(selected_values.height)
            && layers == u64::from(selected_values.layers);
        let has_bound_override = raw_has_bound_override(raw);
        if raw.profile.is_some() && (!exact_dimensions || has_bound_override) {
            return Err(GeneratorError::UnsupportedConfiguration {
                stage: ErrorStage::Configuration,
                reason: "qualified_profile_override",
                value: selected.code() as u64,
            });
        }
        let mut values = if raw.profile.is_some() && exact_dimensions {
            selected_values
        } else {
            interpolate_custom(width, height, layers)?
        };
        apply_overrides(&mut values, raw);
        if raw.relax_route_redundancy {
            values.edge_disjoint_routes = 1;
        }
        if raw.relax_transition_redundancy {
            values.transitions_per_adjacent_pair = 1;
        }
        values.width = width as u16;
        values.height = height as u16;
        values.layers = layers as u16;
        validate_values(
            &values,
            tile_count,
            raw.relax_route_redundancy,
            raw.relax_transition_redundancy,
        )?;
        let qualification = if raw.single_bottleneck {
            Qualification::SingleBottleneck
        } else if raw.profile.is_some() && exact_dimensions && !has_bound_override {
            Qualification::Qualified
        } else {
            Qualification::Custom
        };
        Ok(Self::from_values(
            values,
            qualification,
            raw.profile,
            raw.relax_route_redundancy,
            raw.relax_transition_redundancy,
        ))
    }

    fn from_values(
        v: ProfileValues,
        qualification: Qualification,
        source_profile: Option<QualifiedProfile>,
        relax_route_redundancy: bool,
        relax_transition_redundancy: bool,
    ) -> Self {
        Self {
            width: v.width, height: v.height, layers: v.layers, qualification, source_profile,
            relax_route_redundancy, relax_transition_redundancy, region_min: v.region_min,
            region_max: v.region_max, required_route_min: v.required_route_min,
            required_route_max: v.required_route_max, per_layer_cycles_min: v.per_layer_cycles_min,
            per_layer_cycles_max: v.per_layer_cycles_max, branch_depth_min: v.branch_depth_min,
            branch_depth_max: v.branch_depth_max, articulation_max: v.articulation_max,
            intentional_dead_ends_min: v.intentional_dead_ends_min,
            intentional_dead_ends_max: v.intentional_dead_ends_max,
            optional_mergers_max: v.optional_mergers_max,
            optional_shortcuts_max: v.optional_shortcuts_max, crossings_max: v.crossings_max,
            components_max: v.components_max, edge_disjoint_routes: v.edge_disjoint_routes,
            transitions_per_adjacent_pair: v.transitions_per_adjacent_pair,
            corridor_width: v.corridor_width, hall_width: v.hall_width, spacing: v.spacing,
            placement_attempts: v.placement_attempts, routing_attempts: v.routing_attempts,
            generation_attempts: v.generation_attempts, reroute_budget: v.reroute_budget,
            marker_relocation_budget: v.marker_relocation_budget,
            optional_edge_removal_budget: v.optional_edge_removal_budget,
            ordinary_prefab_ratio_numerator: v.ordinary_prefab_ratio_numerator,
            ordinary_prefab_ratio_denominator: v.ordinary_prefab_ratio_denominator,
            model_marker_cap: v.model_marker_cap, max_lights: v.max_lights,
            max_chunks: v.max_chunks, max_static_bodies: v.max_static_bodies,
            max_total_bodies: v.max_total_bodies, max_vertices: v.max_vertices,
            max_indices: v.max_indices, max_tiles: v.max_tiles,
        }
    }

    pub(super) const fn dimensions(&self) -> (u16, u16, u16) {
        (self.width, self.height, self.layers)
    }

    pub(super) const fn width(&self) -> u16 {
        self.width
    }

    pub(super) const fn height(&self) -> u16 {
        self.height
    }

    pub(super) const fn layers(&self) -> (u16, u16, u16) {
        (self.width, self.height, self.layers)
    }

    pub(super) const fn qualification(&self) -> Qualification {
        self.qualification
    }

    pub(super) const fn region_min(&self) -> u32 {
        self.region_min
    }

    pub(super) const fn region_max(&self) -> u32 {
        self.region_max
    }

    pub(super) const fn required_route_min(&self) -> u32 {
        self.required_route_min
    }

    pub(super) const fn required_route_max(&self) -> u32 {
        self.required_route_max
    }

    pub(super) const fn per_layer_cycles_min(&self) -> u32 {
        self.per_layer_cycles_min
    }

    pub(super) const fn per_layer_cycles_max(&self) -> u32 {
        self.per_layer_cycles_max
    }

    pub(super) const fn branch_depth_min(&self) -> u32 {
        self.branch_depth_min
    }

    pub(super) const fn branch_depth_max(&self) -> u32 {
        self.branch_depth_max
    }

    pub(super) const fn articulation_max(&self) -> u32 {
        self.articulation_max
    }

    pub(super) const fn intentional_dead_ends_min(&self) -> u32 {
        self.intentional_dead_ends_min
    }

    pub(super) const fn intentional_dead_ends_max(&self) -> u32 {
        self.intentional_dead_ends_max
    }

    pub(super) const fn optional_mergers_max(&self) -> u32 {
        self.optional_mergers_max
    }

    pub(super) const fn optional_shortcuts_max(&self) -> u32 {
        self.optional_shortcuts_max
    }

    pub(super) const fn crossings_max(&self) -> u32 {
        self.crossings_max
    }

    pub(super) const fn components_max(&self) -> u32 {
        self.components_max
    }

    pub(super) const fn edge_disjoint_routes(&self) -> u32 {
        self.edge_disjoint_routes
    }

    pub(super) const fn transitions_per_adjacent_pair(&self) -> u32 {
        self.transitions_per_adjacent_pair
    }

    pub(super) const fn corridor_width(&self) -> u32 {
        self.corridor_width
    }

    pub(super) const fn hall_width(&self) -> u32 {
        self.hall_width
    }

    pub(super) const fn spacing(&self) -> u32 {
        self.spacing
    }

    pub(super) const fn placement_attempts(&self) -> u32 {
        self.placement_attempts
    }

    pub(super) const fn routing_attempts(&self) -> u32 {
        self.routing_attempts
    }

    pub(super) const fn generation_attempts(&self) -> u32 {
        self.generation_attempts
    }

    pub(super) const fn reroute_budget(&self) -> u32 {
        self.reroute_budget
    }

    pub(super) const fn marker_relocation_budget(&self) -> u32 {
        self.marker_relocation_budget
    }

    pub(super) const fn optional_edge_removal_budget(&self) -> u32 {
        self.optional_edge_removal_budget
    }

    pub(super) const fn ordinary_prefab_ratio_numerator(&self) -> u32 {
        self.ordinary_prefab_ratio_numerator
    }

    pub(super) const fn ordinary_prefab_ratio_denominator(&self) -> u32 {
        self.ordinary_prefab_ratio_denominator
    }

    pub(super) const fn model_marker_cap(&self) -> u32 {
        self.model_marker_cap
    }

    pub(super) fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(192);
        bytes.extend_from_slice(&(CONFIG_FORMAT_TAG.len() as u64).to_be_bytes());
        bytes.extend_from_slice(CONFIG_FORMAT_TAG);
        bytes.extend_from_slice(&self.width.to_be_bytes());
        bytes.extend_from_slice(&self.height.to_be_bytes());
        bytes.extend_from_slice(&self.layers.to_be_bytes());
        bytes.push(self.qualification.code());
        bytes.push(self.source_profile.map_or(255, QualifiedProfile::code));
        bytes.push(u8::from(self.relax_route_redundancy));
        bytes.push(u8::from(self.relax_transition_redundancy));
        macro_rules! u32_fields {
            ($($field:ident),+ $(,)?) => {$(
                bytes.extend_from_slice(&self.$field.to_be_bytes());
            )+};
        }
        u32_fields!(
            region_min, region_max, required_route_min, required_route_max,
            per_layer_cycles_min, per_layer_cycles_max, branch_depth_min, branch_depth_max,
            articulation_max, intentional_dead_ends_min, intentional_dead_ends_max,
            optional_mergers_max, optional_shortcuts_max, crossings_max, components_max,
            edge_disjoint_routes, transitions_per_adjacent_pair, corridor_width, hall_width,
            spacing, placement_attempts, routing_attempts, generation_attempts, reroute_budget,
            marker_relocation_budget, optional_edge_removal_budget,
            ordinary_prefab_ratio_numerator, ordinary_prefab_ratio_denominator,
            model_marker_cap, max_lights, max_chunks, max_static_bodies, max_total_bodies,
            max_vertices, max_indices, max_tiles,
        );
        bytes
    }

    pub(super) fn canonical_hash(&self) -> String {
        lowercase_hex(&Sha256::digest(self.canonical_bytes()))
    }
}

fn validate_relaxation_mode(raw: &GeneratorConfig) -> Result<(), GeneratorError> {
    if (raw.relax_route_redundancy || raw.relax_transition_redundancy) && !raw.single_bottleneck {
        return Err(GeneratorError::UnsupportedConfiguration {
            stage: ErrorStage::Configuration,
            reason: "relaxation_requires_single_bottleneck",
            value: 0,
        });
    }
    if raw.single_bottleneck && !raw.relax_route_redundancy && !raw.relax_transition_redundancy {
        return Err(GeneratorError::UnsupportedConfiguration {
            stage: ErrorStage::Configuration,
            reason: "single_bottleneck_requires_named_relaxation",
            value: 0,
        });
    }
    Ok(())
}

fn validate_dimensions(w: u64, h: u64, l: u64, tiles: u64) -> Result<(), GeneratorError> {
    for (value, reason, min, max) in [
        (w, "width_out_of_range", 64, 128),
        (h, "height_out_of_range", 64, 128),
        (l, "layers_out_of_range", 2, 4),
    ] {
        if !(min..=max).contains(&value) {
            return Err(GeneratorError::UnsupportedConfiguration {
                stage: ErrorStage::Configuration,
                reason,
                value,
            });
        }
    }
    if tiles > HARD_MAX_TILES {
        return Err(GeneratorError::UnsupportedConfiguration {
            stage: ErrorStage::Configuration,
            reason: "tile_count_exceeds_limit",
            value: tiles,
        });
    }
    Ok(())
}

fn interpolate_custom(w: u64, h: u64, l: u64) -> Result<ProfileValues, GeneratorError> {
    let tiles = w.checked_mul(h).and_then(|n| n.checked_mul(l)).ok_or(
        GeneratorError::ArithmeticOverflow { stage: ErrorStage::Configuration, operation: "custom_tile_count" }
    )?;
    let low = PRESETS[0];
    let high = PRESETS[2];
    let low_tiles = u64::from(low.width) * u64::from(low.height) * u64::from(low.layers);
    let high_tiles = u64::from(high.width) * u64::from(high.height) * u64::from(high.layers);
    let offset = tiles.saturating_sub(low_tiles).min(high_tiles - low_tiles);
    let range = high_tiles - low_tiles;
    let interp = |a: u32, b: u32| -> u32 {
        a + ((u64::from(b - a) * offset + range - 1) / range) as u32
    };
    Ok(ProfileValues {
        width: w as u16, height: h as u16, layers: l as u16,
        region_min: interp(low.region_min, high.region_min), region_max: interp(low.region_max, high.region_max),
        required_route_min: interp(low.required_route_min, high.required_route_min),
        required_route_max: interp(low.required_route_max, high.required_route_max),
        per_layer_cycles_min: interp(low.per_layer_cycles_min, high.per_layer_cycles_min),
        per_layer_cycles_max: interp(low.per_layer_cycles_max, high.per_layer_cycles_max),
        branch_depth_min: interp(low.branch_depth_min, high.branch_depth_min),
        branch_depth_max: interp(low.branch_depth_max, high.branch_depth_max),
        articulation_max: interp(low.articulation_max, high.articulation_max),
        intentional_dead_ends_min: interp(low.intentional_dead_ends_min, high.intentional_dead_ends_min),
        intentional_dead_ends_max: interp(low.intentional_dead_ends_max, high.intentional_dead_ends_max),
        optional_mergers_max: interp(low.optional_mergers_max, high.optional_mergers_max),
        optional_shortcuts_max: interp(low.optional_shortcuts_max, high.optional_shortcuts_max),
        crossings_max: interp(low.crossings_max, high.crossings_max), components_max: 1,
        edge_disjoint_routes: 2, transitions_per_adjacent_pair: 2,
        corridor_width: interp(low.corridor_width, high.corridor_width),
        hall_width: interp(low.hall_width, high.hall_width), spacing: interp(low.spacing, high.spacing),
        placement_attempts: interp(low.placement_attempts, high.placement_attempts),
        routing_attempts: interp(low.routing_attempts, high.routing_attempts),
        generation_attempts: interp(low.generation_attempts, high.generation_attempts),
        reroute_budget: interp(low.reroute_budget, high.reroute_budget),
        marker_relocation_budget: interp(low.marker_relocation_budget, high.marker_relocation_budget),
        optional_edge_removal_budget: interp(low.optional_edge_removal_budget, high.optional_edge_removal_budget),
        ordinary_prefab_ratio_numerator: 3, ordinary_prefab_ratio_denominator: 5,
        model_marker_cap: interp(low.model_marker_cap, high.model_marker_cap), max_lights: 16,
        max_chunks: interp(low.max_chunks, high.max_chunks),
        max_static_bodies: interp(low.max_static_bodies, high.max_static_bodies),
        max_total_bodies: interp(low.max_total_bodies, high.max_total_bodies),
        max_vertices: interp(low.max_vertices, high.max_vertices),
        max_indices: interp(low.max_indices, high.max_indices), max_tiles: 65_536,
    })
}

fn raw_has_bound_override(r: &GeneratorConfig) -> bool {
    r.region_min.is_some() || r.region_max.is_some() || r.required_route_min.is_some()
        || r.required_route_max.is_some() || r.per_layer_cycles_min.is_some()
        || r.per_layer_cycles_max.is_some() || r.branch_depth_min.is_some()
        || r.branch_depth_max.is_some() || r.articulation_max.is_some()
        || r.intentional_dead_ends_min.is_some() || r.intentional_dead_ends_max.is_some()
        || r.optional_mergers_max.is_some() || r.optional_shortcuts_max.is_some()
        || r.crossings_max.is_some() || r.components_max.is_some()
        || r.edge_disjoint_routes.is_some() || r.transitions_per_adjacent_pair.is_some()
        || r.corridor_width.is_some() || r.hall_width.is_some() || r.spacing.is_some()
        || r.placement_attempts.is_some() || r.routing_attempts.is_some()
        || r.generation_attempts.is_some() || r.reroute_budget.is_some()
        || r.marker_relocation_budget.is_some() || r.optional_edge_removal_budget.is_some()
        || r.ordinary_prefab_ratio_numerator.is_some()
        || r.ordinary_prefab_ratio_denominator.is_some() || r.model_marker_cap.is_some()
        || r.max_lights.is_some() || r.max_chunks.is_some() || r.max_static_bodies.is_some()
        || r.max_total_bodies.is_some() || r.max_vertices.is_some() || r.max_indices.is_some()
        || r.max_tiles.is_some()
}

fn apply_overrides(v: &mut ProfileValues, r: &GeneratorConfig) {
    macro_rules! apply { ($($field:ident),+ $(,)?) => {$(if let Some(value) = r.$field { v.$field = value; })+}; }
    apply!(
        region_min, region_max, required_route_min, required_route_max, per_layer_cycles_min,
        per_layer_cycles_max, branch_depth_min, branch_depth_max, articulation_max,
        intentional_dead_ends_min, intentional_dead_ends_max, optional_mergers_max,
        optional_shortcuts_max, crossings_max, components_max, edge_disjoint_routes,
        transitions_per_adjacent_pair, corridor_width, hall_width, spacing, placement_attempts,
        routing_attempts, generation_attempts, reroute_budget, marker_relocation_budget,
        optional_edge_removal_budget, ordinary_prefab_ratio_numerator,
        ordinary_prefab_ratio_denominator, model_marker_cap, max_lights, max_chunks,
        max_static_bodies, max_total_bodies, max_vertices, max_indices, max_tiles,
    );
}

fn validate_values(
    v: &ProfileValues,
    tile_count: u64,
    relax_route_redundancy: bool,
    relax_transition_redundancy: bool,
) -> Result<(), GeneratorError> {
    let relations = [
        (v.region_min, v.region_max, "region_bounds"),
        (v.required_route_min, v.required_route_max, "route_bounds"),
        (v.per_layer_cycles_min, v.per_layer_cycles_max, "cycle_bounds"),
        (v.branch_depth_min, v.branch_depth_max, "branch_bounds"),
        (v.intentional_dead_ends_min, v.intentional_dead_ends_max, "dead_end_bounds"),
    ];
    for (lower, upper, constraint) in relations {
        if lower == 0 || upper < lower {
            return Err(GeneratorError::MandatoryInfeasibility {
                stage: ErrorStage::Configuration, constraint,
                required: u64::from(lower.max(1)), available: u64::from(upper),
            });
        }
    }
    if v.region_min < 6 || u64::from(v.region_min) < u64::from(v.layers) + 4 {
        return Err(GeneratorError::MandatoryInfeasibility {
            stage: ErrorStage::Configuration, constraint: "mandatory_role_count",
            required: u64::from(v.layers) + 4, available: u64::from(v.region_min),
        });
    }
    if u64::from(v.required_route_min) > tile_count {
        return Err(GeneratorError::MandatoryInfeasibility {
            stage: ErrorStage::Configuration, constraint: "required_route_capacity",
            required: u64::from(v.required_route_min), available: tile_count,
        });
    }
    let regions_per_layer = v.region_max / u32::from(v.layers);
    if v.per_layer_cycles_min >= regions_per_layer {
        return Err(GeneratorError::MandatoryInfeasibility {
            stage: ErrorStage::Configuration,
            constraint: "per_layer_cycle_capacity",
            required: u64::from(v.per_layer_cycles_min) + 1,
            available: u64::from(regions_per_layer),
        });
    }
    let transition_endpoints = u64::from(v.transitions_per_adjacent_pair)
        * u64::from(v.layers.saturating_sub(1))
        * 2;
    let mandatory_regions = 6 + transition_endpoints;
    if mandatory_regions > u64::from(v.region_min) {
        return Err(GeneratorError::MandatoryInfeasibility {
            stage: ErrorStage::Configuration,
            constraint: "mandatory_role_and_transition_capacity",
            required: mandatory_regions,
            available: u64::from(v.region_min),
        });
    }
    if v.components_max != 1 || v.corridor_width == 0 || v.hall_width < v.corridor_width
        || v.spacing == 0
    {
        return Err(GeneratorError::UnsupportedConfiguration {
            stage: ErrorStage::Configuration, reason: "invalid_width_spacing_or_components", value: 0,
        });
    }
    if [v.placement_attempts, v.routing_attempts, v.generation_attempts]
        .contains(&0)
    {
        return Err(GeneratorError::UnsupportedConfiguration {
            stage: ErrorStage::Configuration, reason: "zero_attempt_budget", value: 0,
        });
    }
    if v.ordinary_prefab_ratio_denominator == 0
        || v.ordinary_prefab_ratio_numerator > v.ordinary_prefab_ratio_denominator
    {
        return Err(GeneratorError::UnsupportedConfiguration {
            stage: ErrorStage::Configuration, reason: "invalid_prefab_ratio", value: 0,
        });
    }
    let required_routes = if relax_route_redundancy { 1 } else { 2 };
    let required_transitions = if relax_transition_redundancy { 1 } else { 2 };
    if v.edge_disjoint_routes < required_routes {
        return Err(GeneratorError::MandatoryInfeasibility {
            stage: ErrorStage::Configuration,
            constraint: "route_redundancy",
            required: u64::from(required_routes),
            available: u64::from(v.edge_disjoint_routes),
        });
    }
    if v.transitions_per_adjacent_pair < required_transitions {
        return Err(GeneratorError::MandatoryInfeasibility {
            stage: ErrorStage::Configuration,
            constraint: "transition_redundancy",
            required: u64::from(required_transitions),
            available: u64::from(v.transitions_per_adjacent_pair),
        });
    }
    if v.max_lights == 0 || v.max_lights > HARD_MAX_LIGHTS
        || v.max_chunks < u32::from(v.layers) || v.max_chunks > HARD_MAX_CHUNKS
        || v.max_static_bodies < v.max_chunks || v.max_static_bodies > HARD_MAX_STATIC_BODIES
        || v.max_total_bodies < v.max_static_bodies.saturating_add(1)
        || v.max_total_bodies > HARD_MAX_TOTAL_BODIES || v.max_vertices == 0
        || v.max_vertices > HARD_MAX_VERTICES || v.max_indices == 0
        || v.max_indices > HARD_MAX_INDICES || u64::from(v.max_tiles) < tile_count
        || u64::from(v.max_tiles) > HARD_MAX_TILES || v.model_marker_cap == 0
    {
        return Err(GeneratorError::MandatoryInfeasibility {
            stage: ErrorStage::Configuration, constraint: "resource_ceiling",
            required: tile_count, available: u64::from(v.max_tiles),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_presets_and_table_snapshot() {
        assert_eq!(PRESETS.len(), 3);
        for (profile, expected) in [
            (QualifiedProfile::Minimum, (64, 64, 2, 14, 24, 70, 4)),
            (QualifiedProfile::Primary, (96, 96, 3, 24, 40, 140, 6)),
            (QualifiedProfile::Maximum, (128, 128, 4, 40, 64, 220, 8)),
        ] {
            let config = GeneratorConfig::qualified(profile).normalize().unwrap();
            assert_eq!(config.dimensions(), (expected.0, expected.1, expected.2));
            assert_eq!(config.qualification(), Qualification::Qualified);
            assert_eq!((config.region_min, config.region_max), (expected.3, expected.4));
            assert_eq!(config.required_route_min, expected.5);
            assert_eq!(config.branch_depth_min, expected.6);
            assert_eq!(config.edge_disjoint_routes, 2);
            assert_eq!(config.transitions_per_adjacent_pair, 2);
            assert_eq!(config.per_layer_cycles_min, 1);
            assert_eq!((config.max_lights, config.max_tiles), (16, 65_536));
        }
        assert_eq!(PRESETS[2].max_chunks, 512);
        assert_eq!(PRESETS[2].max_static_bodies, 512);
        assert_eq!(PRESETS[2].max_total_bodies, 513);
        assert_eq!(PRESETS[2].max_vertices, 1_600_000);
        assert_eq!(PRESETS[2].max_indices, 2_400_000);
    }

    #[test]
    fn custom_interpolation_is_deterministic_and_custom() {
        let a = GeneratorConfig::custom(80, 72, 3).normalize().unwrap();
        let b = GeneratorConfig::custom(80, 72, 3).normalize().unwrap();
        assert_eq!(a, b);
        assert_eq!(a.qualification(), Qualification::Custom);
        assert!((14..=40).contains(&a.region_min));
        assert_eq!(a.edge_disjoint_routes, 2);
    }

    #[test]
    fn single_bottleneck_only_relaxes_named_criteria() {
        let mut raw = GeneratorConfig::qualified(QualifiedProfile::Primary);
        raw.single_bottleneck = true;
        raw.relax_route_redundancy = true;
        let relaxed = raw.normalize().unwrap();
        let qualified = GeneratorConfig::qualified(QualifiedProfile::Primary).normalize().unwrap();
        assert_eq!(relaxed.qualification(), Qualification::SingleBottleneck);
        assert_eq!(relaxed.edge_disjoint_routes, 1);
        assert_eq!(relaxed.transitions_per_adjacent_pair, 2);
        assert_eq!(relaxed.region_min, qualified.region_min);
        assert_eq!(relaxed.required_route_min, qualified.required_route_min);
        let mut invalid = GeneratorConfig::default();
        invalid.relax_route_redundancy = true;
        assert!(invalid.normalize().is_err());
    }

    #[test]
    fn dimension_tile_and_overflow_boundaries() {
        for (w, h, l) in [(64, 64, 2), (128, 128, 4)] {
            assert!(GeneratorConfig::custom(w, h, l).normalize().is_ok());
        }
        for (w, h, l) in [(0, 64, 2), (63, 64, 2), (129, 64, 2), (64, 63, 2), (64, 129, 2), (64, 64, 1), (64, 64, 5)] {
            assert!(GeneratorConfig::custom(w, h, l).normalize().is_err());
        }
        assert!(GeneratorConfig::custom(u64::MAX, 2, 2)
            .normalize()
            .is_err_and(|e| matches!(e, GeneratorError::ArithmeticOverflow { .. })));
        assert!(GeneratorConfig::custom(128, 128, 4).normalize().is_ok());
    }

    #[test]
    fn rejects_infeasibility_zero_budgets_and_bad_resources() {
        let mut raw = GeneratorConfig::custom(64, 64, 2);
        raw.region_min = Some(3);
        assert!(matches!(raw.normalize(), Err(GeneratorError::MandatoryInfeasibility { .. })));
        let mut raw = GeneratorConfig::custom(64, 64, 2);
        raw.placement_attempts = Some(0);
        assert!(raw.normalize().is_err());
        let mut raw = GeneratorConfig::custom(64, 64, 2);
        raw.max_total_bodies = Some(1);
        assert!(raw.normalize().is_err());
        let mut raw = GeneratorConfig::custom(64, 64, 2);
        raw.hall_width = Some(1);
        raw.corridor_width = Some(2);
        assert!(raw.normalize().is_err());
        let mut raw = GeneratorConfig::custom(64, 64, 2);
        raw.per_layer_cycles_min = Some(20);
        assert!(raw.normalize().is_err());
        let mut raw = GeneratorConfig::custom(64, 64, 4);
        raw.region_min = Some(10);
        raw.transitions_per_adjacent_pair = Some(2);
        assert!(raw.normalize().is_err());
    }

    #[test]
    fn canonical_bytes_are_big_endian_tagged_and_golden() {
        let minimum = GeneratorConfig::qualified(QualifiedProfile::Minimum).normalize().unwrap();
        let bytes = minimum.canonical_bytes();
        assert_eq!(&bytes[0..8], &(CONFIG_FORMAT_TAG.len() as u64).to_be_bytes());
        assert_eq!(&bytes[8..8 + CONFIG_FORMAT_TAG.len()], CONFIG_FORMAT_TAG);
        let dimension_offset = 8 + CONFIG_FORMAT_TAG.len();
        assert_eq!(&bytes[dimension_offset..dimension_offset + 2], &64u16.to_be_bytes());
        assert_eq!(bytes.len(), 8 + CONFIG_FORMAT_TAG.len() + 6 + 4 + 36 * 4);
        assert_eq!(minimum.canonical_hash(), "0e98ab40cd55867b8b6ee401d6c6a58489cf40f7b1808674cb8c78691e2d67cf");
        assert_eq!(GeneratorConfig::qualified(QualifiedProfile::Primary).normalize().unwrap().canonical_hash(), "486977b72b2d864be9f64a043950fd328705fef9879f497fe3fb521f9758b91f");
        assert_eq!(GeneratorConfig::qualified(QualifiedProfile::Maximum).normalize().unwrap().canonical_hash(), "94b9c4e3b5c243cd9ccf952777bc968289565c90dd7e98606eac2c00389b6ffc");
    }

    #[test]
    fn normalized_fields_are_constructor_only_and_changes_affect_hash() {
        let baseline = GeneratorConfig::custom(80, 80, 2).normalize().unwrap();
        let mut raw = GeneratorConfig::custom(80, 80, 2);
        raw.marker_relocation_budget = Some(baseline.marker_relocation_budget + 1);
        let changed = raw.normalize().unwrap();
        assert_ne!(baseline.canonical_bytes(), changed.canonical_bytes());
        assert_ne!(baseline.canonical_hash(), changed.canonical_hash());
    }
}
