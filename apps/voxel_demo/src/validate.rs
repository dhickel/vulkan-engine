//! Pure config validation with path-addressed errors and checked arithmetic.
//!
//! All validators are pure functions callable without renderer or presentation
//! state. Validation rejects invalid input before allocation or worker launch;
//! it never clamps values into validity.
//!
//! # v1 validation
//! `validate_normalized` — preserved unchanged for legacy v1 one-shot path.

use crate::config::{
    AssetRef, GeneratorSection, MaterialsSection, NormalizedConfig, PresetDocument,
    CAVERN_COUNT_MAX, CAVERN_COUNT_MIN, LIGHT_BUDGET_MAX, LIGHT_BUDGET_MIN, SHELL_THICKNESS_MIN,
    SUPPORTED_GENERATOR_VERSIONS, SUPPORTED_RNG_VERSIONS, SUPPORTED_SCHEMA_VERSIONS,
    VALID_RESOLUTIONS,
};

const MAX_MAZE_WORK: u64 = 100_000_000;
const MAX_MC33_STORAGE_BYTES: u64 = 2 * 1024 * 1024 * 1024;

// ─── v1 Validation (preserved unchanged) ───────────────────────────────────

/// Allowed resolution values.
pub const V1_VALID_RESOLUTIONS: &[u32] = &[64, 96, 128];

/// Maximum allowed light budget (v1).
pub const V1_MAX_LIGHT_BUDGET: u32 = 16;

/// Minimum shell thickness (v1, allows 0).
pub const V1_MIN_SHELL_THICKNESS: u32 = 0;

/// Maximum shell thickness as a fraction of resolution.
pub const V1_MAX_SHELL_THICKNESS_RATIO: f32 = 0.4;

/// Validation error variants (v1).
#[derive(Debug, Clone, PartialEq)]
pub enum ConfigError {
    /// Resolution is not in the allowed set.
    InvalidResolution { got: u32, allowed: &'static [u32] },
    /// Light budget exceeds the maximum.
    LightBudgetExceeded { got: u32, max: u32 },
    /// Shell thickness is too small.
    ShellTooThin { got: u32, min: u32 },
    /// Shell thickness exceeds maximum ratio of resolution.
    ShellTooThick {
        got: u32,
        resolution: u32,
        max_ratio: f32,
    },
    /// Resolution is zero (empty lattice).
    ResolutionZero,
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidResolution { got, allowed: _ } => {
                write!(
                    f,
                    "invalid resolution {got}: must be one of {:?}",
                    V1_VALID_RESOLUTIONS
                )
            }
            Self::LightBudgetExceeded { got, max } => {
                write!(f, "light budget {got} exceeds maximum {max}")
            }
            Self::ShellTooThin { got, min } => {
                write!(f, "shell thickness {got} is below minimum {min}")
            }
            Self::ShellTooThick {
                got,
                resolution,
                max_ratio,
            } => {
                write!(
                    f,
                    "shell thickness {got} exceeds {:.0}% of resolution {resolution} (max {})",
                    max_ratio * 100.0,
                    (*resolution as f32 * max_ratio) as u32
                )
            }
            Self::ResolutionZero => {
                write!(f, "resolution must be positive")
            }
        }
    }
}

/// Validate a `NormalizedConfig` (v1). Returns `Ok(())` or the first error.
pub fn validate_normalized(config: &NormalizedConfig) -> Result<(), ConfigError> {
    validate_v1_resolution(config.resolution)?;
    validate_v1_light_budget(config.light_budget)?;
    validate_v1_shell_thickness(config.shell_thickness, config.resolution)?;
    Ok(())
}

fn validate_v1_resolution(resolution: u32) -> Result<(), ConfigError> {
    if resolution == 0 {
        return Err(ConfigError::ResolutionZero);
    }
    if !V1_VALID_RESOLUTIONS.contains(&resolution) {
        return Err(ConfigError::InvalidResolution {
            got: resolution,
            allowed: V1_VALID_RESOLUTIONS,
        });
    }
    Ok(())
}

fn validate_v1_light_budget(budget: u32) -> Result<(), ConfigError> {
    if budget > V1_MAX_LIGHT_BUDGET {
        return Err(ConfigError::LightBudgetExceeded {
            got: budget,
            max: V1_MAX_LIGHT_BUDGET,
        });
    }
    Ok(())
}

fn validate_v1_shell_thickness(thickness: u32, resolution: u32) -> Result<(), ConfigError> {
    if thickness < V1_MIN_SHELL_THICKNESS {
        return Err(ConfigError::ShellTooThin {
            got: thickness,
            min: V1_MIN_SHELL_THICKNESS,
        });
    }
    let max_thickness = (resolution as f32 * V1_MAX_SHELL_THICKNESS_RATIO) as u32;
    if thickness > max_thickness {
        return Err(ConfigError::ShellTooThick {
            got: thickness,
            resolution,
            max_ratio: V1_MAX_SHELL_THICKNESS_RATIO,
        });
    }
    Ok(())
}

// ─── v2 Typed Validation Errors ────────────────────────────────────────────

/// A field-path-addressed validation error.
#[derive(Debug, Clone, PartialEq)]
pub struct ValidationError {
    /// Dotted path to the offending field (e.g. "generator.resolution").
    pub field: String,
    /// Error category.
    pub category: ValidationCategory,
    /// Human-readable constraint description.
    pub constraint: String,
    /// The offending value (if safe to display).
    pub value: Option<String>,
}

/// Categories of validation failures.
#[derive(Debug, Clone, PartialEq)]
pub enum ValidationCategory {
    /// Unsupported version.
    UnsupportedVersion,
    /// Value out of allowed range.
    OutOfRange,
    /// Invalid product (overflow or impossible topology).
    InvalidProduct,
    /// Impossible interior/topology combination.
    ImpossibleTopology,
    /// Malformed reference.
    MalformedReference,
    /// Non-finite value.
    NonFiniteFloat,
}

impl ValidationError {
    pub fn new(
        field: impl Into<String>,
        category: ValidationCategory,
        constraint: impl Into<String>,
    ) -> Self {
        Self {
            field: field.into(),
            category,
            constraint: constraint.into(),
            value: None,
        }
    }

    pub fn with_value(mut self, value: impl std::fmt::Display) -> Self {
        self.value = Some(value.to_string());
        self
    }
}

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.field, self.constraint)?;
        if let Some(ref v) = self.value {
            write!(f, " (got: {v})")?;
        }
        Ok(())
    }
}

// ─── v2 Document Validation ────────────────────────────────────────────────

/// Validate a complete `PresetDocument`. Returns all errors found.
pub fn validate_preset_document(doc: &PresetDocument) -> Vec<ValidationError> {
    let mut errors = Vec::new();

    // Schema/generator/RNG versions
    if !SUPPORTED_SCHEMA_VERSIONS.contains(&doc.schema_version) {
        errors.push(
            ValidationError::new(
                "schema_version",
                ValidationCategory::UnsupportedVersion,
                format!(
                    "unsupported schema version: must be one of {:?}",
                    SUPPORTED_SCHEMA_VERSIONS
                ),
            )
            .with_value(doc.schema_version),
        );
    }
    if !SUPPORTED_GENERATOR_VERSIONS.contains(&doc.generator_version) {
        errors.push(
            ValidationError::new(
                "generator_version",
                ValidationCategory::UnsupportedVersion,
                format!(
                    "unsupported generator version: must be one of {:?}",
                    SUPPORTED_GENERATOR_VERSIONS
                ),
            )
            .with_value(doc.generator_version),
        );
    }
    if !SUPPORTED_RNG_VERSIONS.contains(&doc.rng_version) {
        errors.push(
            ValidationError::new(
                "rng_version",
                ValidationCategory::UnsupportedVersion,
                format!(
                    "unsupported RNG version: must be one of {:?}",
                    SUPPORTED_RNG_VERSIONS
                ),
            )
            .with_value(doc.rng_version),
        );
    }

    let supported_combination = matches!(
        (doc.schema_version, doc.generator_version, doc.rng_version),
        (
            1,
            crate::config::V1_GENERATOR_VERSION,
            crate::config::V1_RNG_VERSION
        ) | (
            2,
            crate::config::V2_GENERATOR_VERSION,
            crate::config::V2_RNG_VERSION
        )
    );
    if !supported_combination {
        errors.push(
            ValidationError::new(
                "versions",
                ValidationCategory::UnsupportedVersion,
                "supported combinations are schema/generator/RNG 1/1/1 and 2/2/2",
            )
            .with_value(format!(
                "{}/{}/{}",
                doc.schema_version, doc.generator_version, doc.rng_version
            )),
        );
    }

    validate_generator_fields(&doc.generator, doc.generator_version == 2, &mut errors);
    validate_materials_present(&doc.materials, &mut errors);

    errors
}

/// Validate generator fields with checked arithmetic.
fn validate_generator_fields(
    gen: &GeneratorSection,
    require_positive_shell: bool,
    errors: &mut Vec<ValidationError>,
) {
    let g = "generator";

    // Resolution
    if !VALID_RESOLUTIONS.contains(&gen.resolution) {
        errors.push(
            ValidationError::new(
                format!("{g}.resolution"),
                ValidationCategory::OutOfRange,
                format!("resolution must be one of {:?}", VALID_RESOLUTIONS),
            )
            .with_value(gen.resolution),
        );
    }

    validate_finite_generator_fields(gen, errors);

    // Shell thickness (>= 1 for v2; v1 preserves the legacy zero-shell rule)
    if require_positive_shell && gen.shell_thickness < SHELL_THICKNESS_MIN {
        errors.push(
            ValidationError::new(
                format!("{g}.shell_thickness"),
                ValidationCategory::OutOfRange,
                format!("shell_thickness must be >= {SHELL_THICKNESS_MIN} for v2"),
            )
            .with_value(gen.shell_thickness),
        );
    }

    // Cavern count [5, 12]
    if gen.cavern_count < CAVERN_COUNT_MIN || gen.cavern_count > CAVERN_COUNT_MAX {
        errors.push(
            ValidationError::new(
                format!("{g}.cavern_count"),
                ValidationCategory::OutOfRange,
                format!("cavern_count must be in [{CAVERN_COUNT_MIN}, {CAVERN_COUNT_MAX}]"),
            )
            .with_value(gen.cavern_count),
        );
    }

    // Tunnel count validation
    validate_tunnel_count(gen, errors);

    // Radius ordering
    if gen.tunnel_radius_min > gen.tunnel_radius_max {
        errors.push(
            ValidationError::new(
                format!("{g}.tunnel_radius_min"),
                ValidationCategory::OutOfRange,
                "tunnel_radius_min must be <= tunnel_radius_max",
            )
            .with_value(format!(
                "min={}, max={}",
                gen.tunnel_radius_min, gen.tunnel_radius_max
            )),
        );
    }
    if gen.cavern_radius_min > gen.cavern_radius_max {
        errors.push(
            ValidationError::new(
                format!("{g}.cavern_radius_min"),
                ValidationCategory::OutOfRange,
                "cavern_radius_min must be <= cavern_radius_max",
            )
            .with_value(format!(
                "min={}, max={}",
                gen.cavern_radius_min, gen.cavern_radius_max
            )),
        );
    }

    // Radius positivity
    if gen.tunnel_radius_min <= 0.0 {
        errors.push(ValidationError::new(
            format!("{g}.tunnel_radius_min"),
            ValidationCategory::OutOfRange,
            "tunnel_radius_min must be positive",
        ));
    }
    if gen.cavern_radius_min <= 0.0 {
        errors.push(ValidationError::new(
            format!("{g}.cavern_radius_min"),
            ValidationCategory::OutOfRange,
            "cavern_radius_min must be positive",
        ));
    }

    // Spline tension [0.0, 1.0]
    if gen.spline_tension < 0.0 || gen.spline_tension > 1.0 {
        errors.push(
            ValidationError::new(
                format!("{g}.spline_tension"),
                ValidationCategory::OutOfRange,
                "spline_tension must be in [0.0, 1.0]",
            )
            .with_value(gen.spline_tension),
        );
    }

    // Roughness >= 0.0
    if gen.roughness < 0.0 {
        errors.push(
            ValidationError::new(
                format!("{g}.roughness"),
                ValidationCategory::OutOfRange,
                "roughness must be non-negative",
            )
            .with_value(gen.roughness),
        );
    }

    // Maze density [0.0, 1.0]
    if gen.maze_density < 0.0 || gen.maze_density > 1.0 {
        errors.push(
            ValidationError::new(
                format!("{g}.maze_density"),
                ValidationCategory::OutOfRange,
                "maze_density must be in [0.0, 1.0]",
            )
            .with_value(gen.maze_density),
        );
    }

    // Maze twistiness [0.0, 1.0]
    if gen.maze_twistiness < 0.0 || gen.maze_twistiness > 1.0 {
        errors.push(
            ValidationError::new(
                format!("{g}.maze_twistiness"),
                ValidationCategory::OutOfRange,
                "maze_twistiness must be in [0.0, 1.0]",
            )
            .with_value(gen.maze_twistiness),
        );
    }

    // Maze radius > 0
    if gen.maze_radius <= 0.0 {
        errors.push(ValidationError::new(
            format!("{g}.maze_radius"),
            ValidationCategory::OutOfRange,
            "maze_radius must be positive",
        ));
    }

    // Maze retries > 0
    if gen.maze_retries == 0 {
        errors.push(ValidationError::new(
            format!("{g}.maze_retries"),
            ValidationCategory::OutOfRange,
            "maze_retries must be positive",
        ));
    }

    // Maze search budget > 0
    if gen.maze_search_budget == 0 {
        errors.push(ValidationError::new(
            format!("{g}.maze_search_budget"),
            ValidationCategory::OutOfRange,
            "maze_search_budget must be positive",
        ));
    }

    // Floor threshold [0.0, 1.0]
    if gen.floor_threshold < 0.0 || gen.floor_threshold > 1.0 {
        errors.push(
            ValidationError::new(
                format!("{g}.floor_threshold"),
                ValidationCategory::OutOfRange,
                "floor_threshold must be in [0.0, 1.0]",
            )
            .with_value(gen.floor_threshold),
        );
    }

    // UV scales > 0
    if gen.wall_uv_scale <= 0.0 {
        errors.push(ValidationError::new(
            format!("{g}.wall_uv_scale"),
            ValidationCategory::OutOfRange,
            "wall_uv_scale must be positive",
        ));
    }
    if gen.floor_uv_scale <= 0.0 {
        errors.push(ValidationError::new(
            format!("{g}.floor_uv_scale"),
            ValidationCategory::OutOfRange,
            "floor_uv_scale must be positive",
        ));
    }

    // Checked arithmetic remains independent of earlier failures so malformed
    // input cannot hide overflow or excessive-work diagnostics.
    validate_interior_bounds(gen, errors);
    validate_checked_capacities(gen, errors);
}

fn validate_finite_generator_fields(gen: &GeneratorSection, errors: &mut Vec<ValidationError>) {
    for (name, value) in [
        ("tunnel_radius_min", gen.tunnel_radius_min),
        ("tunnel_radius_max", gen.tunnel_radius_max),
        ("cavern_radius_min", gen.cavern_radius_min),
        ("cavern_radius_max", gen.cavern_radius_max),
        ("spline_tension", gen.spline_tension),
        ("roughness", gen.roughness),
        ("maze_density", gen.maze_density),
        ("maze_twistiness", gen.maze_twistiness),
        ("maze_radius", gen.maze_radius),
        ("floor_threshold", gen.floor_threshold),
        ("wall_uv_scale", gen.wall_uv_scale),
        ("floor_uv_scale", gen.floor_uv_scale),
    ] {
        if !value.is_finite() {
            errors.push(
                ValidationError::new(
                    format!("generator.{name}"),
                    ValidationCategory::NonFiniteFloat,
                    "value must be finite",
                )
                .with_value(value),
            );
        }
    }
}

fn validate_finite_range(
    field: &str,
    value: f32,
    min: f32,
    max: f32,
    errors: &mut Vec<ValidationError>,
) {
    if !value.is_finite() {
        errors.push(
            ValidationError::new(
                field,
                ValidationCategory::NonFiniteFloat,
                "value must be finite",
            )
            .with_value(value),
        );
    } else if value < min || value > max {
        errors.push(
            ValidationError::new(
                field,
                ValidationCategory::OutOfRange,
                format!("value must be in [{min}, {max}]"),
            )
            .with_value(value),
        );
    }
}

/// Validate tunnel count vs site pair count and tree connectivity.
fn validate_tunnel_count(gen: &GeneratorSection, errors: &mut Vec<ValidationError>) {
    let n = gen.cavern_count as usize;
    // Minimum edges for a connected tree: n - 1
    let min_tree_edges = n.saturating_sub(1);
    // Maximum unique site pairs: n * (n - 1) / 2
    let max_pairs = if let Some(p) = (n as u64)
        .checked_mul((n.saturating_sub(1)) as u64)
        .map(|p| p / 2)
    {
        p as u32
    } else {
        u32::MAX
    };

    if gen.tunnel_count < min_tree_edges as u32 {
        errors.push(
            ValidationError::new(
                "generator.tunnel_count",
                ValidationCategory::OutOfRange,
                format!(
                    "tunnel_count ({}) must be at least {} for a connected tree with {} sites",
                    gen.tunnel_count, min_tree_edges, n
                ),
            )
            .with_value(gen.tunnel_count),
        );
    }

    if gen.tunnel_count > max_pairs {
        errors.push(
            ValidationError::new(
                "generator.tunnel_count",
                ValidationCategory::OutOfRange,
                format!(
                    "tunnel_count ({}) exceeds maximum unique site pairs ({})",
                    gen.tunnel_count, max_pairs
                ),
            )
            .with_value(gen.tunnel_count),
        );
    }
}

/// Validate interior bounds: the carved region must fit required features.
fn validate_interior_bounds(gen: &GeneratorSection, errors: &mut Vec<ValidationError>) {
    let res = gen.resolution;
    let shell = gen.shell_thickness;

    // Interior size after shell on all 6 faces
    let interior = if let Some(shell2) = shell.checked_mul(2) {
        if shell2 >= res {
            // Shell consumes entire lattice — no interior
            errors.push(ValidationError::new(
                "generator.shell_thickness",
                ValidationCategory::ImpossibleTopology,
                format!("shell thickness ({shell}) * 2 >= resolution ({res}): no interior space"),
            ));
            return;
        }
        res - shell2
    } else {
        errors.push(ValidationError::new(
            "generator.shell_thickness",
            ValidationCategory::InvalidProduct,
            "shell thickness overflow in 2*shell".to_string(),
        ));
        return;
    };

    // Check that interior can contain caverns, tunnels, maze, roughness, and clearance
    let max_feature_radius = gen
        .cavern_radius_max
        .max(gen.tunnel_radius_max)
        .max(gen.maze_radius)
        + gen.roughness;
    // Radius plus roughness displacement, two cells of route/site clearance,
    // and one extraction-safety cell on each side.
    let required_span = (max_feature_radius + 2.0) * 2.0 + 1.0;

    if required_span > interior as f32 {
        errors.push(ValidationError::new(
            "generator.resolution",
            ValidationCategory::ImpossibleTopology,
            format!(
                "interior {interior} voxels cannot contain features requiring {required_span:.1} voxel span \
                 (max_radius={max_feature_radius:.1}, roughness={roughness}, shell={shell})",
                max_feature_radius = max_feature_radius,
                roughness = gen.roughness,
            ),
        ));
    }

    // Check that all cavern sites fit within interior
    if gen.cavern_radius_max * 2.0 > interior as f32 {
        errors.push(ValidationError::new(
            "generator.cavern_radius_max",
            ValidationCategory::ImpossibleTopology,
            format!(
                "cavern diameter ({diam:.1}) exceeds interior size ({interior})",
                diam = gen.cavern_radius_max * 2.0,
            ),
        ));
    }

    if gen.cavern_radius_min.is_finite()
        && gen.cavern_radius_max.is_finite()
        && gen.roughness.is_finite()
    {
        // A conservative aggregate occupancy proof: one mandatory largest
        // cavern plus the remaining sites at the configured minimum radius,
        // each expanded by roughness and two cells of clearance.
        let expanded_volume = |radius: f32| {
            let diameter = (radius + gen.roughness + 2.0) * 2.0;
            f64::from(diameter).powi(3)
        };
        let required_volume = expanded_volume(gen.cavern_radius_max)
            + expanded_volume(gen.cavern_radius_min)
                * f64::from(gen.cavern_count.saturating_sub(1));
        let interior_volume = f64::from(interior).powi(3);
        if !required_volume.is_finite() || required_volume > interior_volume {
            errors.push(ValidationError::new(
                "generator.cavern_count",
                ValidationCategory::ImpossibleTopology,
                format!(
                    "interior volume {interior_volume:.0} cannot contain mandatory site envelope {required_volume:.0}"
                ),
            ));
        }
    }
}

/// Validate checked capacities: voxel counts, byte estimates, MC33 worst-case.
fn validate_checked_capacities(gen: &GeneratorSection, errors: &mut Vec<ValidationError>) {
    let res = gen.resolution as u64;

    // Voxel count: res^3
    let voxel_count = match res.checked_mul(res).and_then(|r2| r2.checked_mul(res)) {
        Some(v) => v,
        None => {
            errors.push(ValidationError::new(
                "generator.resolution",
                ValidationCategory::InvalidProduct,
                format!("resolution^3 overflow: {res}^3"),
            ));
            return;
        }
    };

    // Density + material byte estimate: 2 * res^3
    if voxel_count.checked_mul(2).is_none() {
        errors.push(ValidationError::new(
            "generator.resolution",
            ValidationCategory::InvalidProduct,
            format!("byte estimate overflow: 2 * {res}^3"),
        ));
    }

    // MC33 can emit up to five triangles for every lattice cell, not merely
    // boundary cells. Check triangle/index/vertex capacities and a conservative
    // storage estimate (32-byte vertex + 4-byte index).
    let cells_per_axis = match res.checked_sub(1) {
        Some(value) => value,
        None => return,
    };
    let mc33_cells = cells_per_axis
        .checked_mul(cells_per_axis)
        .and_then(|value| value.checked_mul(cells_per_axis));
    let mc33_storage = mc33_cells
        .and_then(|cells| cells.checked_mul(5))
        .and_then(|triangles| triangles.checked_mul(3))
        .and_then(|vertices_or_indices| {
            vertices_or_indices
                .checked_mul(32)
                .and_then(|vertex_bytes| {
                    vertices_or_indices
                        .checked_mul(4)
                        .and_then(|index_bytes| vertex_bytes.checked_add(index_bytes))
                })
        });
    match mc33_storage {
        None => errors.push(ValidationError::new(
            "generator.resolution",
            ValidationCategory::InvalidProduct,
            "MC33 worst-case capacity or byte estimate overflow",
        )),
        Some(bytes) if bytes > MAX_MC33_STORAGE_BYTES => errors.push(
            ValidationError::new(
                "generator.resolution",
                ValidationCategory::InvalidProduct,
                format!(
                    "MC33 worst-case storage ({bytes} bytes) exceeds limit ({MAX_MC33_STORAGE_BYTES})"
                ),
            )
            .with_value(gen.resolution),
        ),
        Some(_) => {}
    }

    // Site pair count: n*(n-1)/2, checked even though validated n is small.
    if max_site_pairs(gen.cavern_count).is_none() {
        errors.push(ValidationError::new(
            "generator.cavern_count",
            ValidationCategory::InvalidProduct,
            "site pair count overflow",
        ));
    }

    // Retry * search budget overflow check
    if let Some(retry_search) = (gen.maze_retries as u64).checked_mul(gen.maze_search_budget as u64)
    {
        if retry_search > MAX_MAZE_WORK {
            errors.push(ValidationError::new(
                "generator.maze_search_budget",
                ValidationCategory::InvalidProduct,
                format!(
                    "maze_retries * maze_search_budget ({retry_search}) exceeds limit ({MAX_MAZE_WORK})"
                ),
            ));
        }
    } else {
        errors.push(ValidationError::new(
            "generator.maze_search_budget",
            ValidationCategory::InvalidProduct,
            "maze_retries * maze_search_budget overflow".to_string(),
        ));
    }
}

/// Validate material factors and reference shape without filesystem I/O.
fn validate_materials_present(materials: &MaterialsSection, errors: &mut Vec<ValidationError>) {
    for (surface, material) in [("wall", &materials.wall), ("floor", &materials.floor)] {
        for (name, value) in [
            ("base_color_r", material.base_color_r),
            ("base_color_g", material.base_color_g),
            ("base_color_b", material.base_color_b),
        ] {
            validate_finite_range(
                &format!("materials.{surface}.{name}"),
                value,
                0.0,
                1.0,
                errors,
            );
        }
        validate_finite_range(
            &format!("materials.{surface}.roughness_factor"),
            material.roughness_factor,
            0.0,
            2.0,
            errors,
        );
        validate_finite_range(
            &format!("materials.{surface}.metallic_factor"),
            material.metallic_factor,
            0.0,
            0.0,
            errors,
        );
        for (name, reference) in [
            ("albedo", &material.albedo),
            ("normal", &material.normal),
            ("roughness", &material.roughness),
            ("ao", &material.ao),
        ] {
            let field = format!("materials.{surface}.{name}");
            match reference {
                AssetRef::Catalog { id } if id.trim().is_empty() => {
                    errors.push(ValidationError::new(
                        field,
                        ValidationCategory::MalformedReference,
                        "catalog ID must not be empty",
                    ))
                }
                AssetRef::Filesystem { path, non_portable }
                    if path.as_os_str().is_empty() || path.is_absolute() != *non_portable =>
                {
                    errors.push(ValidationError::new(
                        field,
                        ValidationCategory::MalformedReference,
                        "filesystem path must be non-empty and non_portable must exactly match absolute-path status",
                    ));
                }
                _ => {}
            }
        }
    }
}

// ─── Runtime Validation ────────────────────────────────────────────────────

/// Validate runtime options (light_budget 9-16).
pub fn validate_runtime_light_budget(light_budget: u32) -> Result<(), ValidationError> {
    if light_budget < LIGHT_BUDGET_MIN || light_budget > LIGHT_BUDGET_MAX {
        return Err(ValidationError::new(
            "light_budget",
            ValidationCategory::OutOfRange,
            format!("light_budget must be in [{LIGHT_BUDGET_MIN}, {LIGHT_BUDGET_MAX}]"),
        )
        .with_value(light_budget));
    }
    if light_budget < 9 {
        return Err(ValidationError::new(
            "light_budget",
            ValidationCategory::OutOfRange,
            "light_budget must be at least 9 to admit the fixed nine app-owned lights".to_string(),
        )
        .with_value(light_budget));
    }
    Ok(())
}

// ─── Checked Arithmetic Helpers ────────────────────────────────────────────

/// Compute interior length after subtracting shell from both sides.
/// Returns `None` if shell*2 >= resolution.
#[allow(dead_code)] // Validated derived fact consumed by the Phase 02 generator.
pub fn interior_size(resolution: u32, shell_thickness: u32) -> Option<u32> {
    let shell2 = shell_thickness.checked_mul(2)?;
    if shell2 >= resolution {
        return None;
    }
    Some(resolution - shell2)
}

/// Compute total voxel count: resolution^3.
#[allow(dead_code)] // Validated derived fact consumed by the Phase 02 generator.
pub fn voxel_count(resolution: u32) -> Option<u64> {
    let r = resolution as u64;
    r.checked_mul(r)?.checked_mul(r)
}

/// Compute density/material byte estimate: 2 * resolution^3.
#[allow(dead_code)] // Validated derived fact consumed by the Phase 02 generator.
pub fn byte_estimate(resolution: u32) -> Option<u64> {
    voxel_count(resolution)?.checked_mul(2)
}

/// Compute maximum unique site pairs for n sites: n*(n-1)/2.
pub fn max_site_pairs(n: u32) -> Option<u32> {
    let n64 = n as u64;
    let pairs = n64.checked_mul(n64.checked_sub(1)?)? / 2;
    u32::try_from(pairs).ok()
}

/// Minimum edges for a connected tree: n - 1.
#[allow(dead_code)] // Validated derived fact consumed by the Phase 02 generator.
pub fn min_tree_edges(n: u32) -> u32 {
    n.saturating_sub(1)
}

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{
        AssetRef, GeneratorSection, MaterialTheme, MaterialsSection, PresetDocument,
    };

    fn valid_v2_doc() -> PresetDocument {
        PresetDocument {
            schema_version: 2,
            generator_version: 2,
            rng_version: 2,
            generator: GeneratorSection {
                seed: 0,
                resolution: 64,
                shell_thickness: 2,
                cavern_count: 7,
                tunnel_count: 8,
                tunnel_radius_min: 1.5,
                tunnel_radius_max: 3.0,
                cavern_radius_min: 4.0,
                cavern_radius_max: 8.0,
                spline_tension: 0.5,
                roughness: 0.3,
                maze_density: 0.15,
                maze_twistiness: 0.4,
                maze_radius: 1.2,
                maze_retries: 50,
                maze_search_budget: 5000,
                floor_threshold: 0.3,
                wall_uv_scale: 1.0,
                floor_uv_scale: 2.0,
            },
            materials: MaterialsSection {
                wall: MaterialTheme {
                    albedo: AssetRef::Catalog {
                        id: "kb3d/rock_wall_01".into(),
                    },
                    normal: AssetRef::Catalog {
                        id: "kb3d/rock_wall_01".into(),
                    },
                    roughness: AssetRef::Catalog {
                        id: "kb3d/rock_wall_01".into(),
                    },
                    ao: AssetRef::Catalog {
                        id: "kb3d/rock_wall_01".into(),
                    },
                    base_color_r: 0.8,
                    base_color_g: 0.7,
                    base_color_b: 0.6,
                    roughness_factor: 1.0,
                    metallic_factor: 0.0,
                },
                floor: MaterialTheme {
                    albedo: AssetRef::Catalog {
                        id: "kb3d/rock_floor_01".into(),
                    },
                    normal: AssetRef::Catalog {
                        id: "kb3d/rock_floor_01".into(),
                    },
                    roughness: AssetRef::Catalog {
                        id: "kb3d/rock_floor_01".into(),
                    },
                    ao: AssetRef::Catalog {
                        id: "kb3d/rock_floor_01".into(),
                    },
                    base_color_r: 0.6,
                    base_color_g: 0.55,
                    base_color_b: 0.5,
                    roughness_factor: 0.9,
                    metallic_factor: 0.0,
                },
            },
        }
    }

    // ── v1 tests (preserved) ────────────────────────────────────────────

    fn valid_v1_config() -> NormalizedConfig {
        NormalizedConfig {
            seed: 0,
            resolution: 64,
            shell_thickness: 2,
            light_budget: 4,
        }
    }

    #[test]
    fn v1_valid_config_passes() {
        assert_eq!(validate_normalized(&valid_v1_config()), Ok(()));
    }

    #[test]
    fn v1_resolution_zero_rejected() {
        let cfg = NormalizedConfig {
            resolution: 0,
            ..valid_v1_config()
        };
        assert!(matches!(
            validate_normalized(&cfg),
            Err(ConfigError::ResolutionZero)
        ));
    }

    #[test]
    fn v1_invalid_resolution_rejected() {
        let cfg = NormalizedConfig {
            resolution: 50,
            ..valid_v1_config()
        };
        assert!(matches!(
            validate_normalized(&cfg),
            Err(ConfigError::InvalidResolution { .. })
        ));
    }

    #[test]
    fn v1_light_budget_exceeded_rejected() {
        let cfg = NormalizedConfig {
            light_budget: V1_MAX_LIGHT_BUDGET + 1,
            ..valid_v1_config()
        };
        assert!(matches!(
            validate_normalized(&cfg),
            Err(ConfigError::LightBudgetExceeded { .. })
        ));
    }

    #[test]
    fn v1_shell_zero_passes() {
        let cfg = NormalizedConfig {
            shell_thickness: 0,
            ..valid_v1_config()
        };
        assert_eq!(validate_normalized(&cfg), Ok(()));
    }

    // ── v2 schema version ───────────────────────────────────────────────

    #[test]
    fn v2_valid_doc_passes() {
        let errors = validate_preset_document(&valid_v2_doc());
        assert!(errors.is_empty(), "expected no errors, got: {:?}", errors);
    }

    #[test]
    fn v2_reject_unsupported_schema_version() {
        let mut doc = valid_v2_doc();
        doc.schema_version = 99;
        let errors = validate_preset_document(&doc);
        assert!(!errors.is_empty());
        assert!(errors.iter().any(|e| e.field == "schema_version"));
    }

    #[test]
    fn v2_reject_unsupported_generator_version() {
        let mut doc = valid_v2_doc();
        doc.generator_version = 99;
        let errors = validate_preset_document(&doc);
        assert!(errors.iter().any(|e| e.field == "generator_version"));
    }

    #[test]
    fn v2_reject_unsupported_rng_version() {
        let mut doc = valid_v2_doc();
        doc.rng_version = 99;
        let errors = validate_preset_document(&doc);
        assert!(errors.iter().any(|e| e.field == "rng_version"));
    }

    // ── Resolution ──────────────────────────────────────────────────────

    #[test]
    fn v2_all_valid_resolutions_accepted() {
        for &r in VALID_RESOLUTIONS {
            let mut doc = valid_v2_doc();
            doc.generator.resolution = r;
            let errors = validate_preset_document(&doc);
            assert!(
                errors.is_empty(),
                "resolution {r} should be valid, got: {:?}",
                errors
            );
        }
    }

    #[test]
    fn v2_invalid_resolution_rejected() {
        let mut doc = valid_v2_doc();
        doc.generator.resolution = 50;
        let errors = validate_preset_document(&doc);
        assert!(errors.iter().any(|e| e.field == "generator.resolution"));
    }

    // ── Shell thickness ─────────────────────────────────────────────────

    #[test]
    fn v2_shell_zero_rejected() {
        let mut doc = valid_v2_doc();
        doc.generator.shell_thickness = 0;
        let errors = validate_preset_document(&doc);
        assert!(
            errors
                .iter()
                .any(|e| e.field == "generator.shell_thickness"),
            "shell 0 must be rejected for v2"
        );
    }

    #[test]
    fn v2_shell_one_accepted() {
        let mut doc = valid_v2_doc();
        doc.generator.shell_thickness = 1;
        let errors = validate_preset_document(&doc);
        assert!(errors.is_empty(), "shell 1 must be valid");
    }

    #[test]
    fn v2_shell_consuming_lattice_rejected() {
        let mut doc = valid_v2_doc();
        doc.generator.resolution = 64;
        doc.generator.shell_thickness = 32; // 2*32 = 64, no interior
        let errors = validate_preset_document(&doc);
        assert!(
            errors
                .iter()
                .any(|e| e.category == ValidationCategory::ImpossibleTopology),
            "shell consuming entire lattice must be rejected"
        );
    }

    // ── Cavern count ────────────────────────────────────────────────────

    #[test]
    fn v2_cavern_count_below_min_rejected() {
        let mut doc = valid_v2_doc();
        doc.generator.cavern_count = 4;
        let errors = validate_preset_document(&doc);
        assert!(errors.iter().any(|e| e.field == "generator.cavern_count"));
    }

    #[test]
    fn v2_cavern_count_above_max_rejected() {
        let mut doc = valid_v2_doc();
        doc.generator.cavern_count = 13;
        let errors = validate_preset_document(&doc);
        assert!(errors.iter().any(|e| e.field == "generator.cavern_count"));
    }

    #[test]
    fn v2_cavern_count_boundaries_accepted() {
        for &(c, t) in &[(5, 6), (12, 20)] {
            let mut doc = valid_v2_doc();
            doc.generator.cavern_count = c;
            doc.generator.tunnel_count = t; // must satisfy tree connectivity and pair limits
            let errors = validate_preset_document(&doc);
            assert!(
                errors.is_empty(),
                "cavern_count {c} should be valid, got: {:?}",
                errors
            );
        }
    }

    // ── Tunnel count ────────────────────────────────────────────────────

    #[test]
    fn v2_tunnel_count_below_min_tree_rejected() {
        let mut doc = valid_v2_doc();
        doc.generator.cavern_count = 7;
        doc.generator.tunnel_count = 5; // min tree edges = 6
        let errors = validate_preset_document(&doc);
        assert!(errors.iter().any(|e| e.field == "generator.tunnel_count"));
    }

    #[test]
    fn v2_tunnel_count_above_max_pairs_rejected() {
        let mut doc = valid_v2_doc();
        doc.generator.cavern_count = 5;
        doc.generator.tunnel_count = 11; // max pairs = 10
        let errors = validate_preset_document(&doc);
        assert!(errors.iter().any(|e| e.field == "generator.tunnel_count"));
    }

    // ── Radius ordering ─────────────────────────────────────────────────

    #[test]
    fn v2_tunnel_radius_inverted_rejected() {
        let mut doc = valid_v2_doc();
        doc.generator.tunnel_radius_min = 5.0;
        doc.generator.tunnel_radius_max = 3.0;
        let errors = validate_preset_document(&doc);
        assert!(errors
            .iter()
            .any(|e| e.field == "generator.tunnel_radius_min"));
    }

    #[test]
    fn v2_cavern_radius_inverted_rejected() {
        let mut doc = valid_v2_doc();
        doc.generator.cavern_radius_min = 10.0;
        doc.generator.cavern_radius_max = 5.0;
        let errors = validate_preset_document(&doc);
        assert!(errors
            .iter()
            .any(|e| e.field == "generator.cavern_radius_min"));
    }

    // ── Non-finite floats ───────────────────────────────────────────────

    #[test]
    fn v2_non_finite_radius_rejected_in_normalize() {
        let mut doc = valid_v2_doc();
        doc.generator.tunnel_radius_min = f32::INFINITY;
        let result = crate::config::normalize_document(&mut doc);
        assert!(result.is_err());
    }

    // ── Runtime light budget ────────────────────────────────────────────

    #[test]
    fn runtime_light_budget_valid() {
        assert!(validate_runtime_light_budget(9).is_ok());
        assert!(validate_runtime_light_budget(12).is_ok());
        assert!(validate_runtime_light_budget(16).is_ok());
    }

    #[test]
    fn runtime_light_budget_below_min_rejected() {
        let err = validate_runtime_light_budget(8).unwrap_err();
        assert!(err.field == "light_budget");
    }

    #[test]
    fn runtime_light_budget_above_max_rejected() {
        let err = validate_runtime_light_budget(17).unwrap_err();
        assert!(err.field == "light_budget");
    }

    // ── Interior bounds ─────────────────────────────────────────────────

    #[test]
    fn interior_bounds_rejects_large_features_small_interior() {
        let mut doc = valid_v2_doc();
        doc.generator.resolution = 64;
        doc.generator.shell_thickness = 2;
        doc.generator.cavern_radius_max = 35.0; // diameter 70 > interior 60
        let errors = validate_preset_document(&doc);
        assert!(
            errors
                .iter()
                .any(|e| e.category == ValidationCategory::ImpossibleTopology),
            "cavern too large for interior: {:?}",
            errors
        );
    }

    // ── Checked arithmetic ──────────────────────────────────────────────

    #[test]
    fn checked_voxel_count() {
        assert_eq!(voxel_count(64), Some(64 * 64 * 64));
        assert_eq!(voxel_count(128), Some(128 * 128 * 128));
        // u32::MAX^3 overflows u64
        assert!(voxel_count(u32::MAX).is_none());
    }

    #[test]
    fn checked_byte_estimate() {
        assert_eq!(byte_estimate(64), Some(2 * 64 * 64 * 64));
    }

    #[test]
    fn checked_interior_size() {
        assert_eq!(interior_size(64, 2), Some(60));
        assert_eq!(interior_size(64, 32), None); // 2*32 = 64, no interior
        assert_eq!(interior_size(64, 33), None);
    }

    #[test]
    fn checked_max_site_pairs() {
        assert_eq!(max_site_pairs(5), Some(10)); // 5*4/2
        assert_eq!(max_site_pairs(7), Some(21)); // 7*6/2
        assert_eq!(max_site_pairs(12), Some(66)); // 12*11/2
    }

    #[test]
    fn checked_min_tree_edges() {
        assert_eq!(min_tree_edges(5), 4);
        assert_eq!(min_tree_edges(7), 6);
        assert_eq!(min_tree_edges(1), 0);
        assert_eq!(min_tree_edges(0), 0);
    }

    #[test]
    fn unsupported_mixed_version_combination_is_rejected() {
        let mut doc = valid_v2_doc();
        doc.rng_version = 1;
        let errors = validate_preset_document(&doc);
        assert!(errors.iter().any(|error| error.field == "versions"));
    }

    #[test]
    fn legacy_document_combination_preserves_zero_shell_rule() {
        let mut doc = valid_v2_doc();
        doc.schema_version = 1;
        doc.generator_version = 1;
        doc.rng_version = 1;
        doc.generator.shell_thickness = 0;
        let errors = validate_preset_document(&doc);
        assert!(!errors
            .iter()
            .any(|error| error.field == "generator.shell_thickness"
                && error.category == ValidationCategory::OutOfRange));
    }

    #[test]
    fn pure_validation_rejects_non_finite_generator_and_material_values() {
        let mut doc = valid_v2_doc();
        doc.generator.maze_density = f32::NAN;
        doc.materials.wall.base_color_r = f32::INFINITY;
        let errors = validate_preset_document(&doc);
        assert!(
            errors
                .iter()
                .filter(|error| error.category == ValidationCategory::NonFiniteFloat)
                .count()
                >= 2
        );
    }

    #[test]
    fn validation_rejects_excessive_work_and_capacity_estimates() {
        let mut work = valid_v2_doc();
        work.generator.maze_retries = u32::MAX;
        work.generator.maze_search_budget = u32::MAX;
        assert!(validate_preset_document(&work).iter().any(|error| {
            error.field == "generator.maze_search_budget"
                && error.category == ValidationCategory::InvalidProduct
        }));

        let mut capacity = valid_v2_doc();
        capacity.generator.resolution = u32::MAX;
        assert!(validate_preset_document(&capacity).iter().any(|error| {
            error.field == "generator.resolution"
                && error.category == ValidationCategory::InvalidProduct
        }));
    }

    #[test]
    fn malformed_asset_shape_and_material_factors_are_rejected() {
        let mut doc = valid_v2_doc();
        doc.materials.wall.albedo = AssetRef::Filesystem {
            path: "/tmp/wall.png".into(),
            non_portable: false,
        };
        doc.materials.floor.metallic_factor = 0.5;
        let errors = validate_preset_document(&doc);
        assert!(errors
            .iter()
            .any(|error| error.category == ValidationCategory::MalformedReference));
        assert!(errors
            .iter()
            .any(|error| error.field == "materials.floor.metallic_factor"));
    }
}
