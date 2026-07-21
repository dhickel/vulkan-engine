//! Configuration types with generation/presentation separation.
//!
//! # v1 (legacy)
//! `NormalizedConfig` captures every input that affects generation output.
//! `PresentationConfig` captures runtime-only options (headless, capture, env).
//! These types are preserved unchanged for generator/RNG v1 fixtures.
//!
//! # v2
//! `PresetDocument` is the strict, versioned, complete TOML document.
//! `ResolvedAppConfig` carries the resolved document, runtime options, source,
//! identities, and empty digest/reproduction slots.
//! `CanonicalHasher` computes stable SHA-256 identities from typed fields.

use std::path::{Path, PathBuf};

use serde::{de, Deserialize, Deserializer, Serialize, Serializer};
use sha2::{Digest, Sha256};

// ─── v1 Types (preserved unchanged) ────────────────────────────────────────

/// Generation-affecting configuration (v1).
///
/// All fields in this struct contribute to deterministic output. Two runs with
/// the same `NormalizedConfig` and the same RNG seed must produce byte-identical
/// results. Never add presentation-only fields (e.g. `--headless`, `--capture_dir`)
/// to this struct.
#[derive(Debug, Clone, PartialEq)]
pub struct NormalizedConfig {
    /// RNG seed for deterministic generation.
    pub seed: u64,
    /// Cubic lattice resolution. Must be one of 64, 96, or 128.
    pub resolution: u32,
    /// Thickness of the solid shell around the cave boundary (voxels).
    pub shell_thickness: u32,
    /// Maximum point lights allowed in the generated scene.
    pub light_budget: u32,
}

impl NormalizedConfig {
    /// Canonical byte representation for deterministic hashing.
    #[allow(dead_code)]
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&self.seed.to_be_bytes());
        bytes.extend_from_slice(&self.resolution.to_be_bytes());
        bytes.extend_from_slice(&self.shell_thickness.to_be_bytes());
        bytes.extend_from_slice(&self.light_budget.to_be_bytes());
        bytes
    }
}

/// Presentation-only configuration (v1).
///
/// These options affect how the result is displayed or captured, but never
/// the generated content itself.
#[derive(Debug, Clone, PartialEq)]
pub struct PresentationConfig {
    /// Run headless (no window).
    pub headless: bool,
    /// Output directory for frame captures.
    pub capture_dir: Option<PathBuf>,
    /// Environment map path for IBL.
    pub env_path: Option<PathBuf>,
}

// ─── v2 Constants ──────────────────────────────────────────────────────────

/// Supported schema versions.
pub const SUPPORTED_SCHEMA_VERSIONS: &[u32] = &[1, 2];

/// Supported generator versions.
pub const SUPPORTED_GENERATOR_VERSIONS: &[u32] = &[1, 2];

/// Supported RNG versions.
pub const SUPPORTED_RNG_VERSIONS: &[u32] = &[1, 2];

/// v1 generator/RNG version selects the unchanged legacy route.
pub const V1_GENERATOR_VERSION: u32 = 1;

/// v1 RNG version selects the unchanged legacy stream.
pub const V1_RNG_VERSION: u32 = 1;

/// v2 generator version selects the v2 route.
pub const V2_GENERATOR_VERSION: u32 = 2;

/// v2 named-stream RNG version.
pub const V2_RNG_VERSION: u32 = 2;

/// Allowed cubic resolutions.
pub const VALID_RESOLUTIONS: &[u32] = &[64, 96, 128];

/// Valid cavern count range (inclusive). Minimum 5 = five core roles.
pub const CAVERN_COUNT_MIN: u32 = 5;
pub const CAVERN_COUNT_MAX: u32 = 12;

/// Shell thickness must be >= 1 for v2.
pub const SHELL_THICKNESS_MIN: u32 = 1;

/// Runtime light budget range (inclusive). Excluded from saves and identities.
pub const LIGHT_BUDGET_MIN: u32 = 9;
pub const LIGHT_BUDGET_MAX: u32 = 16;

/// Fixed nine-light policy version for scene-config identity.
pub const NINE_LIGHT_POLICY_VERSION: u32 = 1;

// ─── Asset References ──────────────────────────────────────────────────────

/// An asset reference before resolution.
///
/// Catalog IDs are resolved through the fixed repository asset-root catalog.
/// Filesystem paths may be relative (resolved from the source document directory)
/// or absolute. Equivalent path spellings are normalized before use.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", deny_unknown_fields)]
pub enum AssetRef {
    /// A catalog ID resolved through the repository asset-root catalog.
    #[serde(rename = "catalog")]
    Catalog {
        /// Catalog identifier (e.g. "kb3d/rock_wall_01").
        id: String,
    },
    /// A filesystem path. Relative paths resolve against the source document
    /// directory; absolute paths are used directly.
    #[serde(rename = "filesystem")]
    Filesystem {
        /// Filesystem path.
        path: PathBuf,
        /// `true` identifies normalized absolute references emitted by saves;
        /// relative source references must state `false` explicitly.
        non_portable: bool,
    },
}

/// A resolved asset reference for identity computation and internal use.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ResolvedAssetRef {
    /// Resolved catalog identity.
    Catalog(String),
    /// Normalized absolute filesystem path.
    Filesystem(PathBuf),
}

// ─── Material Theme ────────────────────────────────────────────────────────

/// PBR material theme for a single surface (wall or floor).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MaterialTheme {
    /// Albedo/base-color texture reference.
    pub albedo: AssetRef,
    /// Normal-map texture reference.
    pub normal: AssetRef,
    /// Roughness texture reference (grayscale, channel R).
    pub roughness: AssetRef,
    /// Ambient-occlusion texture reference.
    pub ao: AssetRef,
    /// Base color factor (sRGB, [0,1] per channel).
    pub base_color_r: f32,
    pub base_color_g: f32,
    pub base_color_b: f32,
    /// Roughness multiplier.
    pub roughness_factor: f32,
    /// Metallic factor (must be 0.0 for the fixed cave material policy).
    pub metallic_factor: f32,
}

/// Wall and floor material themes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MaterialsSection {
    pub wall: MaterialTheme,
    pub floor: MaterialTheme,
}

// ─── Generator Section ─────────────────────────────────────────────────────

/// Generator-stage fields that affect cave topology and geometry.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeneratorSection {
    /// RNG seed (0 is valid deterministic). Values above TOML's signed
    /// integer range use a decimal string representation.
    #[serde(
        deserialize_with = "deserialize_u64_toml",
        serialize_with = "serialize_u64_toml"
    )]
    pub seed: u64,
    /// Cubic lattice resolution (64, 96, or 128).
    pub resolution: u32,
    /// Thickness of the solid shell in voxels (>= 1 for v2).
    pub shell_thickness: u32,
    /// Number of cavern sites [5, 12].
    pub cavern_count: u32,
    /// Number of tunnel edges.
    pub tunnel_count: u32,
    /// Minimum tunnel radius.
    pub tunnel_radius_min: f32,
    /// Maximum tunnel radius.
    pub tunnel_radius_max: f32,
    /// Minimum cavern radius.
    pub cavern_radius_min: f32,
    /// Maximum cavern radius.
    pub cavern_radius_max: f32,
    /// Spline tension for tunnel paths [0.0, 1.0].
    pub spline_tension: f32,
    /// Surface roughness amplitude.
    pub roughness: f32,
    /// Maze link density [0.0, 1.0].
    pub maze_density: f32,
    /// Maze path twistiness [0.0, 1.0].
    pub maze_twistiness: f32,
    /// Maze tunnel radius.
    pub maze_radius: f32,
    /// Maze carving retry limit.
    pub maze_retries: u32,
    /// Maze search budget (nodes explored).
    pub maze_search_budget: u32,
    /// Floor threshold for triangle classification (density above this = floor).
    pub floor_threshold: f32,
    /// UV scale for wall surfaces.
    pub wall_uv_scale: f32,
    /// UV scale for floor surfaces.
    pub floor_uv_scale: f32,
}

fn deserialize_u64_toml<'de, D>(deserializer: D) -> Result<u64, D::Error>
where
    D: Deserializer<'de>,
{
    struct U64Visitor;

    impl<'de> de::Visitor<'de> for U64Visitor {
        type Value = u64;

        fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            formatter.write_str("a non-negative TOML integer or decimal u64 string")
        }

        fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            u64::try_from(value).map_err(|_| E::custom("seed must be non-negative"))
        }

        fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            Ok(value)
        }

        fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            value
                .parse()
                .map_err(|_| E::custom("seed string must contain a decimal u64"))
        }
    }

    deserializer.deserialize_any(U64Visitor)
}

fn serialize_u64_toml<S>(value: &u64, serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    if let Ok(value) = i64::try_from(*value) {
        serializer.serialize_i64(value)
    } else {
        serializer.serialize_str(&value.to_string())
    }
}

// ─── PresetDocument ────────────────────────────────────────────────────────

/// The complete, strict, versioned TOML document.
///
/// Contains only schema/generator/RNG versions plus generator and material/theme
/// data. Runtime/presentation values (`light_budget`, headless, capture, env,
/// panel state) never enter this document or saves.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PresetDocument {
    /// Schema version of this document format.
    pub schema_version: u32,
    /// Generator version (1 = v1 legacy, 2 = v2 named RNG).
    pub generator_version: u32,
    /// RNG version (1 = v1 PCG, 2 = v2 named streams).
    pub rng_version: u32,
    /// Generator-stage fields.
    pub generator: GeneratorSection,
    /// Wall and floor material themes.
    pub materials: MaterialsSection,
}

// ─── Source Location ───────────────────────────────────────────────────────

/// Where a loaded document came from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DocumentSource {
    /// Built-in preset loaded via `include_str!`.
    Embedded { name: String },
    /// External preset selected by name.
    Preset { name: String },
    /// External config file.
    ConfigFile { path: PathBuf },
}

/// A loaded document with its source context.
#[derive(Debug, Clone)]
pub struct LoadedDocument {
    /// The parsed preset document.
    pub document: PresetDocument,
    /// Where this document was loaded from.
    pub source: DocumentSource,
    /// Directory to resolve relative filesystem asset references against.
    /// For embedded and preset sources, this is the repository asset root.
    /// For config file sources, this is the file's parent directory.
    pub source_dir: PathBuf,
}

// ─── Runtime Options ───────────────────────────────────────────────────────

/// Runtime/presentation options. Never serialized into documents or saves.
#[derive(Debug, Clone)]
pub struct RuntimeOptions {
    /// Light budget [9, 16]. Excluded from saves and identities.
    pub light_budget: u32,
    /// Run headless (no window).
    pub headless: bool,
    /// Output directory for frame captures.
    pub capture_dir: Option<PathBuf>,
    /// Environment map path for IBL.
    pub env_path: Option<PathBuf>,
}

// ─── Identities ────────────────────────────────────────────────────────────

/// SHA-256 geometry identity: all geometry-affecting generator fields
/// plus generator and RNG versions.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GeometryIdentity(pub [u8; 32]);

/// SHA-256 scene-config identity: geometry identity plus classifier,
/// materials, UV transforms, resolved asset references, PBR factors,
/// and the fixed nine-light-policy version/fields.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SceneConfigIdentity(pub [u8; 32]);

/// Asset content digest (SHA-256 of asset file bytes). Separate metadata,
/// never part of semantic identities.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AssetDigest(pub [u8; 32]);

// ─── ResolvedAppConfig ─────────────────────────────────────────────────────

/// The fully resolved application configuration.
#[derive(Debug, Clone)]
pub struct ResolvedAppConfig {
    /// The resolved document (post-merge with CLI overrides).
    pub document: PresetDocument,
    /// Runtime/presentation options.
    pub runtime: RuntimeOptions,
    /// Where the base document came from.
    pub source: DocumentSource,
    /// Resolved semantic asset references (for identities).
    pub resolved_wall_albedo: ResolvedAssetRef,
    pub resolved_wall_normal: ResolvedAssetRef,
    pub resolved_wall_roughness: ResolvedAssetRef,
    pub resolved_wall_ao: ResolvedAssetRef,
    pub resolved_floor_albedo: ResolvedAssetRef,
    pub resolved_floor_normal: ResolvedAssetRef,
    pub resolved_floor_roughness: ResolvedAssetRef,
    pub resolved_floor_ao: ResolvedAssetRef,
    /// Geometry identity.
    pub geometry_identity: GeometryIdentity,
    /// Scene-config identity.
    pub scene_config_identity: SceneConfigIdentity,
    /// Asset digests (populated in later phases).
    pub asset_digests: Vec<(ResolvedAssetRef, AssetDigest)>,
}

// ─── Canonical Hasher ──────────────────────────────────────────────────────

/// Domain-separated, versioned SHA-256 writer for canonical identities.
///
/// Writes every field with a field tag (u16) and canonical bytes. Floats are
/// normalized (-0.0 → +0.0) and non-finite values are rejected. Integers use
/// big-endian. Text/bytes are length-framed (u32 length prefix).
pub struct CanonicalHasher {
    hasher: Sha256,
}

impl CanonicalHasher {
    /// Create a new hasher framed with a domain tag and version.
    pub fn new(domain: &str, version: u32) -> Self {
        let mut hasher = Sha256::new();
        // Domain-separate: write domain length + domain bytes
        hasher.update(&(domain.len() as u32).to_be_bytes());
        hasher.update(domain.as_bytes());
        // Version
        hasher.update(&version.to_be_bytes());
        Self { hasher }
    }

    /// Write a u16 field tag.
    pub fn tag(&mut self, tag: u16) -> &mut Self {
        self.hasher.update(&tag.to_be_bytes());
        self
    }

    /// Write a u32.
    pub fn u32(&mut self, value: u32) -> &mut Self {
        self.hasher.update(&value.to_be_bytes());
        self
    }

    /// Write a u64.
    pub fn u64(&mut self, value: u64) -> &mut Self {
        self.hasher.update(&value.to_be_bytes());
        self
    }

    /// Write a normalized finite f32. Panics on NaN/inf.
    pub fn f32(&mut self, value: f32) -> &mut Self {
        assert!(value.is_finite(), "non-finite float in canonical hash");
        // Normalize -0.0 → +0.0
        let normalized = if value == 0.0 { 0.0f32 } else { value };
        self.hasher.update(&normalized.to_bits().to_be_bytes());
        self
    }

    /// Write a length-framed string (u32 length + UTF-8 bytes).
    pub fn str(&mut self, value: &str) -> &mut Self {
        self.hasher.update(&(value.len() as u32).to_be_bytes());
        self.hasher.update(value.as_bytes());
        self
    }

    /// Finalize into a 32-byte digest.
    pub fn finish(self) -> [u8; 32] {
        self.hasher.finalize().into()
    }
}

// ─── Identity Computation ──────────────────────────────────────────────────

const GEOMETRY_IDENTITY_DOMAIN: &str = "voxel-demo/geometry-identity/v1";
const SCENE_CONFIG_IDENTITY_DOMAIN: &str = "voxel-demo/scene-config-identity/v1";

/// Field tags for geometry identity.
mod geom_tag {
    pub const GENERATOR_VERSION: u16 = 1;
    pub const RNG_VERSION: u16 = 2;
    pub const SEED: u16 = 3;
    pub const RESOLUTION: u16 = 4;
    pub const SHELL_THICKNESS: u16 = 5;
    pub const CAVERN_COUNT: u16 = 6;
    pub const TUNNEL_COUNT: u16 = 7;
    pub const TUNNEL_RADIUS_MIN: u16 = 8;
    pub const TUNNEL_RADIUS_MAX: u16 = 9;
    pub const CAVERN_RADIUS_MIN: u16 = 10;
    pub const CAVERN_RADIUS_MAX: u16 = 11;
    pub const SPLINE_TENSION: u16 = 12;
    pub const ROUGHNESS: u16 = 13;
    pub const MAZE_DENSITY: u16 = 14;
    pub const MAZE_TWISTINESS: u16 = 15;
    pub const MAZE_RADIUS: u16 = 16;
    pub const MAZE_RETRIES: u16 = 17;
    pub const MAZE_SEARCH_BUDGET: u16 = 18;
}

/// Field tags for scene-config identity (extends geometry).
mod scene_tag {
    pub const GEOMETRY_IDENTITY: u16 = 100;
    pub const FLOOR_THRESHOLD: u16 = 101;
    pub const WALL_ALBEDO: u16 = 102;
    pub const WALL_NORMAL: u16 = 103;
    pub const WALL_ROUGHNESS: u16 = 104;
    pub const WALL_AO: u16 = 105;
    pub const FLOOR_ALBEDO: u16 = 106;
    pub const FLOOR_NORMAL: u16 = 107;
    pub const FLOOR_ROUGHNESS: u16 = 108;
    pub const FLOOR_AO: u16 = 109;
    pub const WALL_BASE_COLOR_R: u16 = 110;
    pub const WALL_BASE_COLOR_G: u16 = 111;
    pub const WALL_BASE_COLOR_B: u16 = 112;
    pub const WALL_ROUGHNESS_FACTOR: u16 = 113;
    pub const WALL_METALLIC_FACTOR: u16 = 114;
    pub const FLOOR_BASE_COLOR_R: u16 = 115;
    pub const FLOOR_BASE_COLOR_G: u16 = 116;
    pub const FLOOR_BASE_COLOR_B: u16 = 117;
    pub const FLOOR_ROUGHNESS_FACTOR: u16 = 118;
    pub const FLOOR_METALLIC_FACTOR: u16 = 119;
    pub const WALL_UV_SCALE: u16 = 120;
    pub const FLOOR_UV_SCALE: u16 = 121;
    pub const NINE_LIGHT_POLICY: u16 = 200;
}

/// Compute the geometry identity from typed generator fields.
pub fn compute_geometry_identity(
    generator_version: u32,
    rng_version: u32,
    gen: &GeneratorSection,
) -> GeometryIdentity {
    let mut h = CanonicalHasher::new(GEOMETRY_IDENTITY_DOMAIN, 1);
    h.tag(geom_tag::GENERATOR_VERSION).u32(generator_version);
    h.tag(geom_tag::RNG_VERSION).u32(rng_version);
    h.tag(geom_tag::SEED).u64(gen.seed);
    h.tag(geom_tag::RESOLUTION).u32(gen.resolution);
    h.tag(geom_tag::SHELL_THICKNESS).u32(gen.shell_thickness);
    h.tag(geom_tag::CAVERN_COUNT).u32(gen.cavern_count);
    h.tag(geom_tag::TUNNEL_COUNT).u32(gen.tunnel_count);
    h.tag(geom_tag::TUNNEL_RADIUS_MIN)
        .f32(gen.tunnel_radius_min);
    h.tag(geom_tag::TUNNEL_RADIUS_MAX)
        .f32(gen.tunnel_radius_max);
    h.tag(geom_tag::CAVERN_RADIUS_MIN)
        .f32(gen.cavern_radius_min);
    h.tag(geom_tag::CAVERN_RADIUS_MAX)
        .f32(gen.cavern_radius_max);
    h.tag(geom_tag::SPLINE_TENSION).f32(gen.spline_tension);
    h.tag(geom_tag::ROUGHNESS).f32(gen.roughness);
    h.tag(geom_tag::MAZE_DENSITY).f32(gen.maze_density);
    h.tag(geom_tag::MAZE_TWISTINESS).f32(gen.maze_twistiness);
    h.tag(geom_tag::MAZE_RADIUS).f32(gen.maze_radius);
    h.tag(geom_tag::MAZE_RETRIES).u32(gen.maze_retries);
    h.tag(geom_tag::MAZE_SEARCH_BUDGET)
        .u32(gen.maze_search_budget);
    GeometryIdentity(h.finish())
}

fn write_asset_ref_to_hasher(h: &mut CanonicalHasher, tag: u16, resolved: &ResolvedAssetRef) {
    h.tag(tag);
    match resolved {
        ResolvedAssetRef::Catalog(id) => {
            h.u32(0); // variant tag: catalog
            h.str(id);
        }
        ResolvedAssetRef::Filesystem(path) => {
            h.u32(1); // variant tag: filesystem
            h.str(
                path.to_str()
                    .expect("resolved filesystem identity must be UTF-8"),
            );
        }
    }
}

fn write_nine_light_policy(h: &mut CanonicalHasher, policy_version: u32) {
    // Fixed nine-light policy: 5 site lights + 4 edge lights.
    // The exact positions vary by geometry, but the policy itself is fixed.
    // We write the policy version and the fixed counts.
    h.tag(scene_tag::NINE_LIGHT_POLICY);
    h.u32(policy_version);
    h.u32(5); // site light count
    h.u32(4); // edge light count

    // Fixed site light colors and intensities (canonical order: spawn, junction, grand_cavern, shaft, destination)
    let site_colors: [(f32, f32, f32); 5] = [
        (1.0, 0.85, 0.6), // spawn: warm orange
        (0.9, 0.7, 0.5),  // junction: amber
        (0.6, 0.75, 1.0), // grand_cavern: cool blue
        (0.8, 0.9, 0.7),  // shaft: pale green
        (1.0, 0.65, 0.4), // destination: warm orange
    ];
    let site_intensities: [f32; 5] = [25.0, 18.0, 40.0, 18.0, 25.0];
    let site_ranges: [f32; 5] = [20.0, 20.0, 20.0, 20.0, 20.0];

    for i in 0..5 {
        h.f32(site_colors[i].0)
            .f32(site_colors[i].1)
            .f32(site_colors[i].2);
        h.f32(site_intensities[i]);
        h.f32(site_ranges[i]);
    }

    // Fixed edge light colors (canonical order: spawn→junction, junction→grand_cavern, grand_cavern→destination, junction→shaft)
    let edge_colors: [(f32, f32, f32); 4] = [
        (1.0, 0.3, 0.15),
        (0.5, 0.5, 0.8),
        (0.8, 0.6, 0.3),
        (0.4, 0.7, 0.4),
    ];
    let edge_intensities: [f32; 4] = [12.0, 10.0, 10.0, 8.0];
    let edge_ranges: [f32; 4] = [15.0, 15.0, 15.0, 15.0];

    for i in 0..4 {
        h.f32(edge_colors[i].0)
            .f32(edge_colors[i].1)
            .f32(edge_colors[i].2);
        h.f32(edge_intensities[i]);
        h.f32(edge_ranges[i]);
    }
}

/// Compute the scene-config identity from geometry identity plus material/theme fields.
pub fn compute_scene_config_identity(
    geometry_identity: &GeometryIdentity,
    gen: &GeneratorSection,
    wall: &MaterialTheme,
    floor: &MaterialTheme,
    resolved_wall_albedo: &ResolvedAssetRef,
    resolved_wall_normal: &ResolvedAssetRef,
    resolved_wall_roughness: &ResolvedAssetRef,
    resolved_wall_ao: &ResolvedAssetRef,
    resolved_floor_albedo: &ResolvedAssetRef,
    resolved_floor_normal: &ResolvedAssetRef,
    resolved_floor_roughness: &ResolvedAssetRef,
    resolved_floor_ao: &ResolvedAssetRef,
) -> SceneConfigIdentity {
    let mut h = CanonicalHasher::new(SCENE_CONFIG_IDENTITY_DOMAIN, 1);

    // Geometry identity digest
    h.tag(scene_tag::GEOMETRY_IDENTITY);
    h.hasher.update(&geometry_identity.0);

    // Classifier threshold
    h.tag(scene_tag::FLOOR_THRESHOLD).f32(gen.floor_threshold);

    // Wall material theme
    write_asset_ref_to_hasher(&mut h, scene_tag::WALL_ALBEDO, resolved_wall_albedo);
    write_asset_ref_to_hasher(&mut h, scene_tag::WALL_NORMAL, resolved_wall_normal);
    write_asset_ref_to_hasher(&mut h, scene_tag::WALL_ROUGHNESS, resolved_wall_roughness);
    write_asset_ref_to_hasher(&mut h, scene_tag::WALL_AO, resolved_wall_ao);
    h.tag(scene_tag::WALL_BASE_COLOR_R).f32(wall.base_color_r);
    h.tag(scene_tag::WALL_BASE_COLOR_G).f32(wall.base_color_g);
    h.tag(scene_tag::WALL_BASE_COLOR_B).f32(wall.base_color_b);
    h.tag(scene_tag::WALL_ROUGHNESS_FACTOR)
        .f32(wall.roughness_factor);
    h.tag(scene_tag::WALL_METALLIC_FACTOR)
        .f32(wall.metallic_factor);

    // Floor material theme
    write_asset_ref_to_hasher(&mut h, scene_tag::FLOOR_ALBEDO, resolved_floor_albedo);
    write_asset_ref_to_hasher(&mut h, scene_tag::FLOOR_NORMAL, resolved_floor_normal);
    write_asset_ref_to_hasher(&mut h, scene_tag::FLOOR_ROUGHNESS, resolved_floor_roughness);
    write_asset_ref_to_hasher(&mut h, scene_tag::FLOOR_AO, resolved_floor_ao);
    h.tag(scene_tag::FLOOR_BASE_COLOR_R).f32(floor.base_color_r);
    h.tag(scene_tag::FLOOR_BASE_COLOR_G).f32(floor.base_color_g);
    h.tag(scene_tag::FLOOR_BASE_COLOR_B).f32(floor.base_color_b);
    h.tag(scene_tag::FLOOR_ROUGHNESS_FACTOR)
        .f32(floor.roughness_factor);
    h.tag(scene_tag::FLOOR_METALLIC_FACTOR)
        .f32(floor.metallic_factor);

    // UV scales
    h.tag(scene_tag::WALL_UV_SCALE).f32(gen.wall_uv_scale);
    h.tag(scene_tag::FLOOR_UV_SCALE).f32(gen.floor_uv_scale);

    // Fixed nine-light policy
    write_nine_light_policy(&mut h, NINE_LIGHT_POLICY_VERSION);

    SceneConfigIdentity(h.finish())
}

// ─── Float Normalization ───────────────────────────────────────────────────

/// Normalize -0.0 → +0.0. Returns `None` if the value is NaN or infinite.
pub fn normalize_f32(value: f32) -> Option<f32> {
    if !value.is_finite() {
        return None;
    }
    if value == 0.0 {
        Some(0.0f32)
    } else {
        Some(value)
    }
}

/// Recursively normalize all floats in a PresetDocument in place.
/// Returns Ok(()) or an error describing the first non-finite value.
pub fn normalize_document(doc: &mut PresetDocument) -> Result<(), String> {
    let gen = &mut doc.generator;
    gen.tunnel_radius_min = normalize_f32(gen.tunnel_radius_min)
        .ok_or_else(|| "tunnel_radius_min is non-finite".to_string())?;
    gen.tunnel_radius_max = normalize_f32(gen.tunnel_radius_max)
        .ok_or_else(|| "tunnel_radius_max is non-finite".to_string())?;
    gen.cavern_radius_min = normalize_f32(gen.cavern_radius_min)
        .ok_or_else(|| "cavern_radius_min is non-finite".to_string())?;
    gen.cavern_radius_max = normalize_f32(gen.cavern_radius_max)
        .ok_or_else(|| "cavern_radius_max is non-finite".to_string())?;
    gen.spline_tension = normalize_f32(gen.spline_tension)
        .ok_or_else(|| "spline_tension is non-finite".to_string())?;
    gen.roughness =
        normalize_f32(gen.roughness).ok_or_else(|| "roughness is non-finite".to_string())?;
    gen.maze_density =
        normalize_f32(gen.maze_density).ok_or_else(|| "maze_density is non-finite".to_string())?;
    gen.maze_twistiness = normalize_f32(gen.maze_twistiness)
        .ok_or_else(|| "maze_twistiness is non-finite".to_string())?;
    gen.maze_radius =
        normalize_f32(gen.maze_radius).ok_or_else(|| "maze_radius is non-finite".to_string())?;
    gen.floor_threshold = normalize_f32(gen.floor_threshold)
        .ok_or_else(|| "floor_threshold is non-finite".to_string())?;
    gen.wall_uv_scale = normalize_f32(gen.wall_uv_scale)
        .ok_or_else(|| "wall_uv_scale is non-finite".to_string())?;
    gen.floor_uv_scale = normalize_f32(gen.floor_uv_scale)
        .ok_or_else(|| "floor_uv_scale is non-finite".to_string())?;

    normalize_material_theme(&mut doc.materials.wall)?;
    normalize_material_theme(&mut doc.materials.floor)?;
    Ok(())
}

fn normalize_material_theme(m: &mut MaterialTheme) -> Result<(), String> {
    m.base_color_r =
        normalize_f32(m.base_color_r).ok_or_else(|| "base_color_r is non-finite".to_string())?;
    m.base_color_g =
        normalize_f32(m.base_color_g).ok_or_else(|| "base_color_g is non-finite".to_string())?;
    m.base_color_b =
        normalize_f32(m.base_color_b).ok_or_else(|| "base_color_b is non-finite".to_string())?;
    m.roughness_factor = normalize_f32(m.roughness_factor)
        .ok_or_else(|| "roughness_factor is non-finite".to_string())?;
    m.metallic_factor = normalize_f32(m.metallic_factor)
        .ok_or_else(|| "metallic_factor is non-finite".to_string())?;
    Ok(())
}

// ─── Asset Reference Resolution ────────────────────────────────────────────

/// Resolve an `AssetRef` against a source directory.
///
/// - Catalog IDs are returned as-is (resolution against the asset-root catalog
///   happens in later phases).
/// - Filesystem references: relative paths are resolved against `source_dir`;
///   absolute paths are used directly. All paths are normalized.
pub fn resolve_asset_ref(
    asset_ref: &AssetRef,
    source_dir: &Path,
    known_catalog_ids: &[&str],
) -> Result<ResolvedAssetRef, String> {
    match asset_ref {
        AssetRef::Catalog { id } => {
            if id.is_empty() {
                return Err("catalog ID must not be empty".to_string());
            }
            if !known_catalog_ids.contains(&id.as_str()) {
                return Err(format!("unknown catalog ID: '{id}'"));
            }
            Ok(ResolvedAssetRef::Catalog(id.clone()))
        }
        AssetRef::Filesystem { path, non_portable } => {
            if path.as_os_str().is_empty() {
                return Err("filesystem asset path must not be empty".to_string());
            }
            if path.is_absolute() != *non_portable {
                return Err(
                    "filesystem reference must use non_portable=true exactly when path is absolute"
                        .to_string(),
                );
            }
            let resolved = resolve_filesystem_path(path, source_dir)?;
            if resolved.to_str().is_none() {
                return Err("filesystem asset path must be valid UTF-8".to_string());
            }
            let metadata = std::fs::metadata(&resolved).map_err(|error| {
                format!(
                    "filesystem asset does not resolve to a readable file '{}': {error}",
                    resolved.display()
                )
            })?;
            if !metadata.is_file() {
                return Err(format!(
                    "filesystem asset is not a regular file: '{}'",
                    resolved.display()
                ));
            }
            Ok(ResolvedAssetRef::Filesystem(resolved))
        }
    }
}

fn resolve_filesystem_path(path: &Path, source_dir: &Path) -> Result<PathBuf, String> {
    let base = if source_dir.is_absolute() {
        source_dir.to_path_buf()
    } else {
        std::env::current_dir()
            .map_err(|e| format!("cannot resolve current directory: {e}"))?
            .join(source_dir)
    };
    let joined;
    let candidate = if path.is_absolute() {
        path
    } else {
        joined = base.join(path);
        &joined
    };
    normalize_path(candidate)
}

/// Lexically normalize an absolute path without consulting the filesystem.
fn normalize_path(path: &Path) -> Result<PathBuf, String> {
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            std::path::Component::ParentDir => {
                if !out.pop() {
                    return Err(format!("path escapes filesystem root: {}", path.display()));
                }
            }
            std::path::Component::CurDir => {}
            component => out.push(component),
        }
    }
    if !out.is_absolute() {
        return Err(format!(
            "normalized path is not absolute: {}",
            out.display()
        ));
    }
    Ok(out)
}

// ─── Save & Load ───────────────────────────────────────────────────────────

/// Save a canonical complete `PresetDocument` to a TOML file.
///
/// The save boundary normalizes all floats and emits filesystem references as
/// normalized absolute paths with `non_portable = true`.
#[allow(dead_code)] // Public save boundary is consumed by the Phase 06 editor.
pub fn save_preset(doc: &PresetDocument, source_dir: &Path, path: &Path) -> Result<(), String> {
    let save_doc = prepare_document_for_save(doc, source_dir)?;
    let toml_str = save_preset_canonical(&save_doc)?;
    std::fs::write(path, toml_str)
        .map_err(|e| format!("failed to write {path}: {e}", path = path.display()))
}

/// Load a `PresetDocument` from a TOML file.
///
/// Returns the parsed document. The caller must normalize, validate, and
/// resolve references separately.
pub fn load_preset(path: &Path) -> Result<PresetDocument, String> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("failed to read {path}: {e}", path = path.display()))?;
    let doc: PresetDocument = toml::from_str(&content)
        .map_err(|e| format!("TOML parse error in {path}: {e}", path = path.display()))?;
    Ok(doc)
}

/// Load a `PresetDocument` from raw TOML bytes.
pub fn load_preset_from_bytes(bytes: &[u8]) -> Result<PresetDocument, String> {
    let s = std::str::from_utf8(bytes).map_err(|e| format!("invalid UTF-8: {e}"))?;
    let doc: PresetDocument = toml::from_str(s).map_err(|e| format!("TOML parse error: {e}"))?;
    Ok(doc)
}

/// Canonical save: serialize the document with asserted canonical TOML.
/// For byte-stable saves, we use `toml::to_string` (not pretty).
#[allow(dead_code)] // Shared by save_preset and deterministic contract tests.
pub fn save_preset_canonical(doc: &PresetDocument) -> Result<String, String> {
    let mut normalized = doc.clone();
    normalize_document(&mut normalized)?;
    toml::to_string(&normalized).map_err(|e| format!("TOML serialization failed: {e}"))
}

/// Prepare an `AssetRef` for saving: filesystem refs become absolute and non-portable.
#[allow(dead_code)] // Shared by the future editor save action.
pub fn asset_ref_for_save(asset_ref: &AssetRef, source_dir: &Path) -> Result<AssetRef, String> {
    Ok(match asset_ref {
        AssetRef::Catalog { id } => AssetRef::Catalog { id: id.clone() },
        AssetRef::Filesystem { path, .. } => AssetRef::Filesystem {
            path: resolve_filesystem_path(path, source_dir)?,
            non_portable: true,
        },
    })
}

// ─── PresetDocument normalization for save ─────────────────────────────────

/// Prepare a document for saving by normalizing all floats and converting
/// filesystem references to absolute non-portable form.
#[allow(dead_code)] // Shared by the future editor save action.
pub fn prepare_document_for_save(
    doc: &PresetDocument,
    source_dir: &Path,
) -> Result<PresetDocument, String> {
    let mut save_doc = doc.clone();
    normalize_document(&mut save_doc)?;

    // Convert filesystem asset refs to absolute non-portable form
    save_doc.materials.wall.albedo = asset_ref_for_save(&doc.materials.wall.albedo, source_dir)?;
    save_doc.materials.wall.normal = asset_ref_for_save(&doc.materials.wall.normal, source_dir)?;
    save_doc.materials.wall.roughness =
        asset_ref_for_save(&doc.materials.wall.roughness, source_dir)?;
    save_doc.materials.wall.ao = asset_ref_for_save(&doc.materials.wall.ao, source_dir)?;
    save_doc.materials.floor.albedo = asset_ref_for_save(&doc.materials.floor.albedo, source_dir)?;
    save_doc.materials.floor.normal = asset_ref_for_save(&doc.materials.floor.normal, source_dir)?;
    save_doc.materials.floor.roughness =
        asset_ref_for_save(&doc.materials.floor.roughness, source_dir)?;
    save_doc.materials.floor.ao = asset_ref_for_save(&doc.materials.floor.ao, source_dir)?;

    Ok(save_doc)
}

// ─── Embedded Presets ──────────────────────────────────────────────────────

/// Load an embedded `PresetDocument` from `include_str!` bytes.
/// Panics at compile time if the source file is missing or malformed.
macro_rules! embed_preset {
    ($path:literal) => {{
        let bytes = include_str!(concat!("../presets/", $path));
        match crate::config::load_preset_from_bytes(bytes.as_bytes()) {
            Ok(doc) => doc,
            Err(e) => {
                panic!(
                    "invalid embedded preset {}: {e}",
                    concat!("../presets/", $path)
                );
            }
        }
    }};
}

/// Get the four known preset names.
#[allow(dead_code)] // Used by tests now and the Phase 06 preset picker later.
pub fn known_preset_names() -> &'static [&'static str] {
    &["default", "cavernous", "mazy", "tight"]
}

/// Get an embedded preset by name. Returns `None` if not a known preset.
pub fn get_embedded_preset(name: &str) -> Option<(&'static str, PresetDocument)> {
    match name {
        "default" => Some(("default", embed_preset!("default.toml"))),
        "cavernous" => Some(("cavernous", embed_preset!("cavernous.toml"))),
        "mazy" => Some(("mazy", embed_preset!("mazy.toml"))),
        "tight" => Some(("tight", embed_preset!("tight.toml"))),
        _ => None,
    }
}

/// Get the raw embedded preset bytes for byte-equality testing.
#[allow(dead_code)] // Exposes the include_str source bytes for contract tests.
pub fn get_embedded_preset_bytes(name: &str) -> Option<&'static [u8]> {
    match name {
        "default" => Some(include_bytes!("../presets/default.toml")),
        "cavernous" => Some(include_bytes!("../presets/cavernous.toml")),
        "mazy" => Some(include_bytes!("../presets/mazy.toml")),
        "tight" => Some(include_bytes!("../presets/tight.toml")),
        _ => None,
    }
}

/// Known catalog IDs for asset reference resolution.
pub fn known_catalog_ids() -> &'static [&'static str] {
    &["kb3d/rock_wall_01", "kb3d/rock_floor_01"]
}

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Float normalization ────────────────────────────────────────────

    #[test]
    fn normalize_negative_zero() {
        assert_eq!(normalize_f32(-0.0f32), Some(0.0f32));
        assert_eq!(normalize_f32(0.0f32), Some(0.0f32));
    }

    #[test]
    fn normalize_rejects_nan_inf() {
        assert_eq!(normalize_f32(f32::NAN), None);
        assert_eq!(normalize_f32(f32::INFINITY), None);
        assert_eq!(normalize_f32(f32::NEG_INFINITY), None);
    }

    #[test]
    fn normalize_preserves_finite() {
        assert_eq!(normalize_f32(3.14), Some(3.14));
        assert_eq!(normalize_f32(-2.5), Some(-2.5));
    }

    // ── CanonicalHasher ────────────────────────────────────────────────

    #[test]
    fn hasher_domain_separation() {
        let mut h1 = CanonicalHasher::new("test/a", 1);
        h1.u32(42);
        let d1 = h1.finish();

        let mut h2 = CanonicalHasher::new("test/b", 1);
        h2.u32(42);
        let d2 = h2.finish();

        assert_ne!(d1, d2);
    }

    #[test]
    fn hasher_version_separation() {
        let mut h1 = CanonicalHasher::new("test/v", 1);
        h1.u32(42);
        let d1 = h1.finish();

        let mut h2 = CanonicalHasher::new("test/v", 2);
        h2.u32(42);
        let d2 = h2.finish();

        assert_ne!(d1, d2);
    }

    #[test]
    fn hasher_deterministic() {
        let mut h1 = CanonicalHasher::new("test/det", 1);
        h1.u32(1).f32(2.0).str("hello");

        let mut h2 = CanonicalHasher::new("test/det", 1);
        h2.u32(1).f32(2.0).str("hello");

        assert_eq!(h1.finish(), h2.finish());
    }

    #[test]
    fn hasher_normalizes_negative_zero() {
        let mut h1 = CanonicalHasher::new("test/nz", 1);
        h1.f32(-0.0f32);
        let d1 = h1.finish();

        let mut h2 = CanonicalHasher::new("test/nz", 1);
        h2.f32(0.0f32);
        let d2 = h2.finish();

        assert_eq!(d1, d2);
    }

    #[test]
    #[should_panic(expected = "non-finite float")]
    fn hasher_rejects_nan() {
        let mut h = CanonicalHasher::new("test", 1);
        h.f32(f32::NAN);
    }

    #[test]
    fn hasher_tag_separation() {
        let mut h1 = CanonicalHasher::new("test/tag", 1);
        h1.tag(1).u32(42);
        let d1 = h1.finish();

        let mut h2 = CanonicalHasher::new("test/tag", 1);
        h2.tag(2).u32(42);
        let d2 = h2.finish();

        assert_ne!(d1, d2);
    }

    // ── Identity stability ─────────────────────────────────────────────

    fn make_test_doc() -> PresetDocument {
        PresetDocument {
            schema_version: 2,
            generator_version: 2,
            rng_version: 2,
            generator: GeneratorSection {
                seed: 42,
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

    fn make_test_resolved_refs() -> (
        ResolvedAssetRef,
        ResolvedAssetRef,
        ResolvedAssetRef,
        ResolvedAssetRef,
        ResolvedAssetRef,
        ResolvedAssetRef,
        ResolvedAssetRef,
        ResolvedAssetRef,
    ) {
        (
            ResolvedAssetRef::Catalog("kb3d/rock_wall_01".into()),
            ResolvedAssetRef::Catalog("kb3d/rock_wall_01".into()),
            ResolvedAssetRef::Catalog("kb3d/rock_wall_01".into()),
            ResolvedAssetRef::Catalog("kb3d/rock_wall_01".into()),
            ResolvedAssetRef::Catalog("kb3d/rock_floor_01".into()),
            ResolvedAssetRef::Catalog("kb3d/rock_floor_01".into()),
            ResolvedAssetRef::Catalog("kb3d/rock_floor_01".into()),
            ResolvedAssetRef::Catalog("kb3d/rock_floor_01".into()),
        )
    }

    #[test]
    fn geometry_identity_is_deterministic() {
        let doc = make_test_doc();
        let id1 = compute_geometry_identity(doc.generator_version, doc.rng_version, &doc.generator);
        let id2 = compute_geometry_identity(doc.generator_version, doc.rng_version, &doc.generator);
        assert_eq!(id1, id2);
    }

    #[test]
    fn geometry_identity_changes_with_seed() {
        let mut doc = make_test_doc();
        let id1 = compute_geometry_identity(doc.generator_version, doc.rng_version, &doc.generator);
        doc.generator.seed = 99;
        let id2 = compute_geometry_identity(doc.generator_version, doc.rng_version, &doc.generator);
        assert_ne!(id1, id2);
    }

    #[test]
    fn geometry_identity_changes_with_resolution() {
        let mut doc = make_test_doc();
        let id1 = compute_geometry_identity(doc.generator_version, doc.rng_version, &doc.generator);
        doc.generator.resolution = 128;
        let id2 = compute_geometry_identity(doc.generator_version, doc.rng_version, &doc.generator);
        assert_ne!(id1, id2);
    }

    #[test]
    fn geometry_identity_ignores_materials() {
        let mut doc = make_test_doc();
        let id1 = compute_geometry_identity(doc.generator_version, doc.rng_version, &doc.generator);
        doc.materials.wall.base_color_r = 0.1;
        doc.materials.floor.base_color_g = 0.2;
        let id2 = compute_geometry_identity(doc.generator_version, doc.rng_version, &doc.generator);
        assert_eq!(
            id1, id2,
            "geometry identity must be unaffected by materials"
        );
    }

    #[test]
    fn geometry_identity_ignores_classifier_fields() {
        let mut doc = make_test_doc();
        let id1 = compute_geometry_identity(doc.generator_version, doc.rng_version, &doc.generator);
        doc.generator.floor_threshold = 0.9;
        doc.generator.wall_uv_scale = 5.0;
        doc.generator.floor_uv_scale = 5.0;
        let id2 = compute_geometry_identity(doc.generator_version, doc.rng_version, &doc.generator);
        assert_eq!(
            id1, id2,
            "geometry identity must be unaffected by classifier/UV fields"
        );
    }

    #[test]
    fn scene_config_identity_includes_geometry() {
        let mut doc = make_test_doc();
        let (wa, wn, wr, wo, fa, fn_, fr, fo) = make_test_resolved_refs();
        let id1 = compute_scene_config_identity(
            &compute_geometry_identity(doc.generator_version, doc.rng_version, &doc.generator),
            &doc.generator,
            &doc.materials.wall,
            &doc.materials.floor,
            &wa,
            &wn,
            &wr,
            &wo,
            &fa,
            &fn_,
            &fr,
            &fo,
        );
        doc.generator.seed = 99;
        let id2 = compute_scene_config_identity(
            &compute_geometry_identity(doc.generator_version, doc.rng_version, &doc.generator),
            &doc.generator,
            &doc.materials.wall,
            &doc.materials.floor,
            &wa,
            &wn,
            &wr,
            &wo,
            &fa,
            &fn_,
            &fr,
            &fo,
        );
        assert_ne!(
            id1, id2,
            "scene-config identity must change when geometry changes"
        );
    }

    #[test]
    fn scene_config_identity_changes_with_materials() {
        let mut doc = make_test_doc();
        let (wa, wn, wr, wo, fa, fn_, fr, fo) = make_test_resolved_refs();
        let geo_id =
            compute_geometry_identity(doc.generator_version, doc.rng_version, &doc.generator);
        let id1 = compute_scene_config_identity(
            &geo_id,
            &doc.generator,
            &doc.materials.wall,
            &doc.materials.floor,
            &wa,
            &wn,
            &wr,
            &wo,
            &fa,
            &fn_,
            &fr,
            &fo,
        );
        doc.materials.wall.base_color_r = 0.123;
        let id2 = compute_scene_config_identity(
            &geo_id,
            &doc.generator,
            &doc.materials.wall,
            &doc.materials.floor,
            &wa,
            &wn,
            &wr,
            &wo,
            &fa,
            &fn_,
            &fr,
            &fo,
        );
        assert_ne!(
            id1, id2,
            "scene-config identity must change with material fields"
        );
    }

    #[test]
    fn scene_config_identity_changes_with_uv_scale() {
        let mut doc = make_test_doc();
        let (wa, wn, wr, wo, fa, fn_, fr, fo) = make_test_resolved_refs();
        let geo_id =
            compute_geometry_identity(doc.generator_version, doc.rng_version, &doc.generator);
        let id1 = compute_scene_config_identity(
            &geo_id,
            &doc.generator,
            &doc.materials.wall,
            &doc.materials.floor,
            &wa,
            &wn,
            &wr,
            &wo,
            &fa,
            &fn_,
            &fr,
            &fo,
        );
        doc.generator.wall_uv_scale = 3.0;
        let id2 = compute_scene_config_identity(
            &geo_id,
            &doc.generator,
            &doc.materials.wall,
            &doc.materials.floor,
            &wa,
            &wn,
            &wr,
            &wo,
            &fa,
            &fn_,
            &fr,
            &fo,
        );
        assert_ne!(id1, id2);
    }

    #[test]
    fn scene_config_identity_changes_with_floor_threshold() {
        let mut doc = make_test_doc();
        let (wa, wn, wr, wo, fa, fn_, fr, fo) = make_test_resolved_refs();
        let geo_id =
            compute_geometry_identity(doc.generator_version, doc.rng_version, &doc.generator);
        let id1 = compute_scene_config_identity(
            &geo_id,
            &doc.generator,
            &doc.materials.wall,
            &doc.materials.floor,
            &wa,
            &wn,
            &wr,
            &wo,
            &fa,
            &fn_,
            &fr,
            &fo,
        );
        doc.generator.floor_threshold = 0.8;
        let id2 = compute_scene_config_identity(
            &geo_id,
            &doc.generator,
            &doc.materials.wall,
            &doc.materials.floor,
            &wa,
            &wn,
            &wr,
            &wo,
            &fa,
            &fn_,
            &fr,
            &fo,
        );
        assert_ne!(id1, id2);
    }

    // ── Asset ref resolution ───────────────────────────────────────────

    #[test]
    fn resolve_catalog_ref() {
        let resolved = resolve_asset_ref(
            &AssetRef::Catalog {
                id: "kb3d/rock_wall_01".into(),
            },
            Path::new("/tmp"),
            known_catalog_ids(),
        )
        .unwrap();
        assert_eq!(
            resolved,
            ResolvedAssetRef::Catalog("kb3d/rock_wall_01".into())
        );
    }

    #[test]
    fn resolve_catalog_ref_unknown_id_rejected() {
        let err = resolve_asset_ref(
            &AssetRef::Catalog {
                id: "unknown/asset".into(),
            },
            Path::new("/tmp"),
            known_catalog_ids(),
        )
        .unwrap_err();
        assert!(err.contains("unknown catalog ID"));
    }

    #[test]
    fn resolve_catalog_ref_empty_id_rejected() {
        let err = resolve_asset_ref(
            &AssetRef::Catalog { id: "".into() },
            Path::new("/tmp"),
            known_catalog_ids(),
        )
        .unwrap_err();
        assert!(err.contains("empty"));
    }

    fn temp_asset(relative: &str) -> PathBuf {
        let path = std::env::temp_dir()
            .join(format!("voxel-demo-config-tests-{}", std::process::id()))
            .join(relative);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, b"asset").unwrap();
        path
    }

    #[test]
    fn resolve_filesystem_relative() {
        let absolute = temp_asset("relative/textures/wall.png");
        let source_dir = absolute.parent().unwrap().parent().unwrap();
        let resolved = resolve_asset_ref(
            &AssetRef::Filesystem {
                path: "textures/wall.png".into(),
                non_portable: false,
            },
            source_dir,
            known_catalog_ids(),
        )
        .unwrap();
        assert_eq!(resolved, ResolvedAssetRef::Filesystem(absolute));
    }

    #[test]
    fn resolve_filesystem_absolute() {
        let absolute = temp_asset("absolute/wall.png");
        let resolved = resolve_asset_ref(
            &AssetRef::Filesystem {
                path: absolute.clone(),
                non_portable: true,
            },
            Path::new("/ignored"),
            known_catalog_ids(),
        )
        .unwrap();
        assert_eq!(resolved, ResolvedAssetRef::Filesystem(absolute));
    }

    #[test]
    fn resolve_filesystem_missing_file_is_rejected() {
        let missing = std::env::temp_dir()
            .join(format!("voxel-demo-config-tests-{}", std::process::id()))
            .join("missing.png");
        let error = resolve_asset_ref(
            &AssetRef::Filesystem {
                path: missing,
                non_portable: true,
            },
            Path::new("/ignored"),
            known_catalog_ids(),
        )
        .unwrap_err();
        assert!(error.contains("does not resolve"));
    }

    #[test]
    fn resolve_filesystem_empty_path_rejected() {
        let err = resolve_asset_ref(
            &AssetRef::Filesystem {
                path: PathBuf::new(),
                non_portable: false,
            },
            Path::new("/tmp"),
            known_catalog_ids(),
        )
        .unwrap_err();
        assert!(err.contains("empty"));
    }

    // ── Save/load round-trip ───────────────────────────────────────────

    #[test]
    fn save_load_typed_roundtrip() {
        let doc = make_test_doc();
        let canonical = save_preset_canonical(&doc).unwrap();
        let reloaded = load_preset_from_bytes(canonical.as_bytes()).unwrap();

        // Typed-value equality
        assert_eq!(doc.schema_version, reloaded.schema_version);
        assert_eq!(doc.generator_version, reloaded.generator_version);
        assert_eq!(doc.rng_version, reloaded.rng_version);
        assert_eq!(doc.generator.seed, reloaded.generator.seed);
        assert_eq!(doc.generator.resolution, reloaded.generator.resolution);
        assert_eq!(doc.generator.cavern_count, reloaded.generator.cavern_count);
        assert_eq!(
            doc.generator.tunnel_radius_min,
            reloaded.generator.tunnel_radius_min
        );
    }

    #[test]
    fn save_load_identity_stable() {
        let doc = make_test_doc();
        let canonical = save_preset_canonical(&doc).unwrap();
        let reloaded = load_preset_from_bytes(canonical.as_bytes()).unwrap();

        let id1 = compute_geometry_identity(doc.generator_version, doc.rng_version, &doc.generator);
        let id2 = compute_geometry_identity(
            reloaded.generator_version,
            reloaded.rng_version,
            &reloaded.generator,
        );
        assert_eq!(id1, id2);
    }

    #[test]
    fn save_load_save_byte_stable() {
        let doc = make_test_doc();
        let first = save_preset_canonical(&doc).unwrap();
        let reloaded = load_preset_from_bytes(first.as_bytes()).unwrap();
        let second = save_preset_canonical(&reloaded).unwrap();
        assert_eq!(first, second, "canonical save must be byte-stable");
    }

    #[test]
    fn save_load_normalizes_negative_zero() {
        let mut doc = make_test_doc();
        doc.generator.tunnel_radius_min = -0.0f32; // negative zero
                                                   // normalize before save
        normalize_document(&mut doc).unwrap();
        let canonical = save_preset_canonical(&doc).unwrap();
        let reloaded = load_preset_from_bytes(canonical.as_bytes()).unwrap();
        // After load, -0.0 and +0.0 are the same float
        assert_eq!(reloaded.generator.tunnel_radius_min, 0.0);
        // But canonical bytes must be stable
        let second = save_preset_canonical(&reloaded).unwrap();
        assert_eq!(canonical, second);
    }

    #[test]
    fn normalize_document_rejects_non_finite() {
        let mut doc = make_test_doc();
        doc.generator.tunnel_radius_min = f32::NAN;
        assert!(normalize_document(&mut doc).is_err());
    }

    // ── Version dispatch ───────────────────────────────────────────────

    #[test]
    fn v1_version_recognized() {
        assert_eq!(V1_GENERATOR_VERSION, 1);
        assert!(SUPPORTED_GENERATOR_VERSIONS.contains(&V1_GENERATOR_VERSION));
    }

    #[test]
    fn v2_version_recognized() {
        assert_eq!(V2_GENERATOR_VERSION, 2);
        assert!(SUPPORTED_GENERATOR_VERSIONS.contains(&V2_GENERATOR_VERSION));
    }

    #[test]
    fn unsupported_version_not_in_list() {
        assert!(!SUPPORTED_GENERATOR_VERSIONS.contains(&0));
        assert!(!SUPPORTED_GENERATOR_VERSIONS.contains(&3));
    }

    // ── Strict deserialization (unknown fields rejected) ───────────────

    #[test]
    fn reject_unknown_top_level_field() {
        let toml_str = r#"
            schema_version = 2
            generator_version = 2
            rng_version = 2
            unknown_field = 42

            [generator]
            seed = 0
            resolution = 64
            shell_thickness = 2
            cavern_count = 7
            tunnel_count = 8
            tunnel_radius_min = 1.5
            tunnel_radius_max = 3.0
            cavern_radius_min = 4.0
            cavern_radius_max = 8.0
            spline_tension = 0.5
            roughness = 0.3

            [materials.wall]
            albedo = { type = "catalog", id = "kb3d/rock_wall_01" }
            normal = { type = "catalog", id = "kb3d/rock_wall_01" }
            roughness = { type = "catalog", id = "kb3d/rock_wall_01" }
            ao = { type = "catalog", id = "kb3d/rock_wall_01" }

            [materials.floor]
            albedo = { type = "catalog", id = "kb3d/rock_floor_01" }
            normal = { type = "catalog", id = "kb3d/rock_floor_01" }
            roughness = { type = "catalog", id = "kb3d/rock_floor_01" }
            ao = { type = "catalog", id = "kb3d/rock_floor_01" }
        "#;
        let result: Result<PresetDocument, _> = toml::from_str(toml_str);
        assert!(result.is_err(), "unknown top-level field must be rejected");
    }

    #[test]
    fn reject_unknown_nested_field() {
        let toml_str = r#"
            schema_version = 2
            generator_version = 2
            rng_version = 2

            [generator]
            seed = 0
            resolution = 64
            shell_thickness = 2
            cavern_count = 7
            tunnel_count = 8
            tunnel_radius_min = 1.5
            tunnel_radius_max = 3.0
            cavern_radius_min = 4.0
            cavern_radius_max = 8.0
            spline_tension = 0.5
            roughness = 0.3
            extra_generator_field = true

            [materials.wall]
            albedo = { type = "catalog", id = "kb3d/rock_wall_01" }
            normal = { type = "catalog", id = "kb3d/rock_wall_01" }
            roughness = { type = "catalog", id = "kb3d/rock_wall_01" }
            ao = { type = "catalog", id = "kb3d/rock_wall_01" }

            [materials.floor]
            albedo = { type = "catalog", id = "kb3d/rock_floor_01" }
            normal = { type = "catalog", id = "kb3d/rock_floor_01" }
            roughness = { type = "catalog", id = "kb3d/rock_floor_01" }
            ao = { type = "catalog", id = "kb3d/rock_floor_01" }
        "#;
        let result: Result<PresetDocument, _> = toml::from_str(toml_str);
        assert!(result.is_err(), "unknown nested field must be rejected");
    }

    #[test]
    fn formatting_comments_and_top_level_field_order_do_not_change_identities() {
        let doc = make_test_doc();
        let canonical = save_preset_canonical(&doc).unwrap();
        let reordered = canonical.replacen(
            "schema_version = 2\ngenerator_version = 2\nrng_version = 2",
            "# formatting is not semantic\nrng_version = 2\n\ngenerator_version = 2\nschema_version = 2",
            1,
        );
        let parsed = load_preset_from_bytes(reordered.as_bytes()).unwrap();
        assert_eq!(
            compute_geometry_identity(doc.generator_version, doc.rng_version, &doc.generator),
            compute_geometry_identity(
                parsed.generator_version,
                parsed.rng_version,
                &parsed.generator
            )
        );
        assert_eq!(
            scene_identity(&doc, &make_test_resolved_refs()),
            scene_identity(&parsed, &make_test_resolved_refs())
        );
    }

    #[test]
    fn strict_documents_reject_missing_and_duplicate_fields() {
        let complete = save_preset_canonical(&make_test_doc()).unwrap();
        let missing = complete
            .lines()
            .filter(|line| !line.starts_with("maze_density ="))
            .collect::<Vec<_>>()
            .join("\n");
        assert!(load_preset_from_bytes(missing.as_bytes()).is_err());

        let duplicate = complete.replacen(
            "schema_version = 2",
            "schema_version = 2\nschema_version = 2",
            1,
        );
        assert!(load_preset_from_bytes(duplicate.as_bytes()).is_err());

        let missing_material = complete
            .lines()
            .filter(|line| !line.starts_with("base_color_r ="))
            .collect::<Vec<_>>()
            .join("\n");
        assert!(load_preset_from_bytes(missing_material.as_bytes()).is_err());

        let unknown_material = complete.replacen(
            "[materials.wall]",
            "[materials.wall]\nunknown_material_field = 1",
            1,
        );
        assert!(load_preset_from_bytes(unknown_material.as_bytes()).is_err());

        assert!(toml::from_str::<AssetRef>(
            "type = \"catalog\"\nid = \"kb3d/rock_wall_01\"\nunknown = true"
        )
        .is_err());
    }

    #[test]
    fn filesystem_reference_requires_complete_portability_marker() {
        let missing_marker = r#"type = "filesystem"\npath = "textures/wall.png""#;
        assert!(toml::from_str::<AssetRef>(missing_marker).is_err());
        assert!(resolve_asset_ref(
            &AssetRef::Filesystem {
                path: "/tmp/wall.png".into(),
                non_portable: false,
            },
            Path::new("/tmp"),
            known_catalog_ids(),
        )
        .is_err());
    }

    #[test]
    fn equivalent_filesystem_spellings_resolve_identically() {
        let absolute = temp_asset("equivalent/config/textures/wall.png");
        let source_dir = absolute.parent().unwrap().parent().unwrap();
        let a = AssetRef::Filesystem {
            path: "textures/./stone/../wall.png".into(),
            non_portable: false,
        };
        let b = AssetRef::Filesystem {
            path: "textures/wall.png".into(),
            non_portable: false,
        };
        assert_eq!(
            resolve_asset_ref(&a, source_dir, known_catalog_ids()).unwrap(),
            resolve_asset_ref(&b, source_dir, known_catalog_ids()).unwrap()
        );
    }

    #[test]
    fn canonical_save_normalizes_and_rejects_non_finite() {
        let mut negative_zero = make_test_doc();
        negative_zero.generator.roughness = -0.0;
        let saved = save_preset_canonical(&negative_zero).unwrap();
        let loaded = load_preset_from_bytes(saved.as_bytes()).unwrap();
        assert_eq!(loaded.generator.roughness.to_bits(), 0.0f32.to_bits());

        negative_zero.generator.roughness = f32::INFINITY;
        assert!(save_preset_canonical(&negative_zero).is_err());
    }

    #[test]
    fn prepared_filesystem_save_is_absolute_non_portable_and_stable() {
        let mut doc = make_test_doc();
        doc.materials.wall.albedo = AssetRef::Filesystem {
            path: "textures/../wall.png".into(),
            non_portable: false,
        };
        let prepared = prepare_document_for_save(&doc, Path::new("/tmp/config")).unwrap();
        assert_eq!(
            prepared.materials.wall.albedo,
            AssetRef::Filesystem {
                path: "/tmp/config/wall.png".into(),
                non_portable: true,
            }
        );
        let once = save_preset_canonical(&prepared).unwrap();
        let twice =
            save_preset_canonical(&load_preset_from_bytes(once.as_bytes()).unwrap()).unwrap();
        assert_eq!(once, twice);
    }

    #[test]
    fn every_geometry_semantic_field_changes_geometry_identity() {
        let doc = make_test_doc();
        let base =
            compute_geometry_identity(doc.generator_version, doc.rng_version, &doc.generator);
        macro_rules! changed {
            ($mutation:expr) => {{
                let mut candidate = doc.clone();
                $mutation(&mut candidate);
                assert_ne!(
                    base,
                    compute_geometry_identity(
                        candidate.generator_version,
                        candidate.rng_version,
                        &candidate.generator,
                    ),
                    "geometry mutation did not change identity"
                );
            }};
        }
        changed!(|d: &mut PresetDocument| d.generator_version += 1);
        changed!(|d: &mut PresetDocument| d.rng_version += 1);
        changed!(|d: &mut PresetDocument| d.generator.seed += 1);
        changed!(|d: &mut PresetDocument| d.generator.resolution += 1);
        changed!(|d: &mut PresetDocument| d.generator.shell_thickness += 1);
        changed!(|d: &mut PresetDocument| d.generator.cavern_count += 1);
        changed!(|d: &mut PresetDocument| d.generator.tunnel_count += 1);
        changed!(|d: &mut PresetDocument| d.generator.tunnel_radius_min += 0.25);
        changed!(|d: &mut PresetDocument| d.generator.tunnel_radius_max += 0.25);
        changed!(|d: &mut PresetDocument| d.generator.cavern_radius_min += 0.25);
        changed!(|d: &mut PresetDocument| d.generator.cavern_radius_max += 0.25);
        changed!(|d: &mut PresetDocument| d.generator.spline_tension += 0.1);
        changed!(|d: &mut PresetDocument| d.generator.roughness += 0.1);
        changed!(|d: &mut PresetDocument| d.generator.maze_density += 0.1);
        changed!(|d: &mut PresetDocument| d.generator.maze_twistiness += 0.1);
        changed!(|d: &mut PresetDocument| d.generator.maze_radius += 0.1);
        changed!(|d: &mut PresetDocument| d.generator.maze_retries += 1);
        changed!(|d: &mut PresetDocument| d.generator.maze_search_budget += 1);
    }

    fn scene_identity(
        doc: &PresetDocument,
        refs: &(
            ResolvedAssetRef,
            ResolvedAssetRef,
            ResolvedAssetRef,
            ResolvedAssetRef,
            ResolvedAssetRef,
            ResolvedAssetRef,
            ResolvedAssetRef,
            ResolvedAssetRef,
        ),
    ) -> SceneConfigIdentity {
        compute_scene_config_identity(
            &compute_geometry_identity(doc.generator_version, doc.rng_version, &doc.generator),
            &doc.generator,
            &doc.materials.wall,
            &doc.materials.floor,
            &refs.0,
            &refs.1,
            &refs.2,
            &refs.3,
            &refs.4,
            &refs.5,
            &refs.6,
            &refs.7,
        )
    }

    #[test]
    fn fixed_nine_light_policy_version_is_identity_sensitive() {
        let mut first = CanonicalHasher::new(SCENE_CONFIG_IDENTITY_DOMAIN, 1);
        write_nine_light_policy(&mut first, NINE_LIGHT_POLICY_VERSION);
        let mut second = CanonicalHasher::new(SCENE_CONFIG_IDENTITY_DOMAIN, 1);
        write_nine_light_policy(&mut second, NINE_LIGHT_POLICY_VERSION + 1);
        assert_ne!(first.finish(), second.finish());
    }

    #[test]
    fn every_scene_semantic_field_changes_scene_identity() {
        let doc = make_test_doc();
        let refs = make_test_resolved_refs();
        let base = scene_identity(&doc, &refs);
        macro_rules! changed_doc {
            ($mutation:expr) => {{
                let mut candidate = doc.clone();
                $mutation(&mut candidate);
                assert_ne!(base, scene_identity(&candidate, &refs));
            }};
        }
        changed_doc!(|d: &mut PresetDocument| d.generator.floor_threshold += 0.1);
        changed_doc!(|d: &mut PresetDocument| d.generator.wall_uv_scale += 0.1);
        changed_doc!(|d: &mut PresetDocument| d.generator.floor_uv_scale += 0.1);
        changed_doc!(|d: &mut PresetDocument| d.materials.wall.base_color_r += 0.1);
        changed_doc!(|d: &mut PresetDocument| d.materials.wall.base_color_g += 0.1);
        changed_doc!(|d: &mut PresetDocument| d.materials.wall.base_color_b += 0.1);
        changed_doc!(|d: &mut PresetDocument| d.materials.wall.roughness_factor += 0.1);
        changed_doc!(|d: &mut PresetDocument| d.materials.wall.metallic_factor += 0.1);
        changed_doc!(|d: &mut PresetDocument| d.materials.floor.base_color_r += 0.1);
        changed_doc!(|d: &mut PresetDocument| d.materials.floor.base_color_g += 0.1);
        changed_doc!(|d: &mut PresetDocument| d.materials.floor.base_color_b += 0.1);
        changed_doc!(|d: &mut PresetDocument| d.materials.floor.roughness_factor += 0.1);
        changed_doc!(|d: &mut PresetDocument| d.materials.floor.metallic_factor += 0.1);

        for index in 0..8 {
            let mut changed_refs = refs.clone();
            let replacement = ResolvedAssetRef::Filesystem(PathBuf::from(format!("/tmp/{index}")));
            match index {
                0 => changed_refs.0 = replacement,
                1 => changed_refs.1 = replacement,
                2 => changed_refs.2 = replacement,
                3 => changed_refs.3 = replacement,
                4 => changed_refs.4 = replacement,
                5 => changed_refs.5 = replacement,
                6 => changed_refs.6 = replacement,
                _ => changed_refs.7 = replacement,
            }
            assert_ne!(base, scene_identity(&doc, &changed_refs));
        }
    }

    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct FixtureSchema {
        version: u32,
        description: String,
    }

    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct SeedCorpusDocument {
        schema: FixtureSchema,
        corpus: Vec<SeedCorpusEntry>,
    }

    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct SeedCorpusEntry {
        label: String,
        #[serde(deserialize_with = "deserialize_u64_toml")]
        seed: u64,
        resolutions: Vec<u32>,
        description: String,
    }

    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct PresetGatesDocument {
        schema: FixtureSchema,
        gates: Vec<PresetGate>,
    }

    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct PresetGate {
        preset: String,
        resolutions: Vec<u32>,
        min_interior: u32,
        max_mc33_triangles: u64,
        max_byte_estimate: u64,
    }

    #[test]
    fn runtime_light_budget_is_absent_from_saves_and_identities() {
        let doc = make_test_doc();
        let refs = make_test_resolved_refs();
        let geometry =
            compute_geometry_identity(doc.generator_version, doc.rng_version, &doc.generator);
        let scene = scene_identity(&doc, &refs);
        let saved = save_preset_canonical(&doc).unwrap();
        assert!(!saved.contains("light_budget"));

        for light_budget in [9, 16] {
            let runtime = RuntimeOptions {
                light_budget,
                headless: light_budget == 16,
                capture_dir: Some(PathBuf::from(format!("/tmp/{light_budget}"))),
                env_path: None,
            };
            assert_eq!(
                geometry,
                compute_geometry_identity(doc.generator_version, doc.rng_version, &doc.generator)
            );
            assert_eq!(scene, scene_identity(&doc, &refs));
            assert_eq!(runtime.light_budget, light_budget);
        }
    }

    #[test]
    fn v2_seed_and_gate_inputs_are_strict_complete_and_canonical() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("test_data/v2");
        let seeds: SeedCorpusDocument =
            toml::from_str(&std::fs::read_to_string(root.join("seed-corpus.toml")).unwrap())
                .unwrap();
        assert_eq!(seeds.schema.version, 1);
        assert!(!seeds.schema.description.is_empty());
        assert!(!seeds.corpus.is_empty());
        for entry in seeds.corpus {
            assert!(!entry.label.is_empty());
            assert!(!entry.description.is_empty());
            assert_eq!(entry.resolutions, VALID_RESOLUTIONS);
            let _ = entry.seed;
        }

        let gates: PresetGatesDocument =
            toml::from_str(&std::fs::read_to_string(root.join("preset-gates.toml")).unwrap())
                .unwrap();
        assert_eq!(gates.schema.version, 1);
        assert!(!gates.schema.description.is_empty());
        assert_eq!(gates.gates.len(), known_preset_names().len());
        for gate in gates.gates {
            assert!(known_preset_names().contains(&gate.preset.as_str()));
            assert_eq!(gate.resolutions, VALID_RESOLUTIONS);
            assert!(gate.min_interior > 0);
            assert!(gate.max_mc33_triangles > 0);
            assert!(gate.max_byte_estimate > 0);
        }
    }

    #[test]
    fn all_embedded_presets_are_complete_valid_distinct_source_bytes() {
        let mut geometry = std::collections::HashSet::new();
        let mut scene = std::collections::HashSet::new();
        for name in known_preset_names() {
            let bytes = get_embedded_preset_bytes(name).unwrap();
            let (loaded_name, mut doc) = get_embedded_preset(name).unwrap();
            assert_eq!(*name, loaded_name);
            assert_eq!(
                bytes,
                std::fs::read(
                    Path::new(env!("CARGO_MANIFEST_DIR"))
                        .join("presets")
                        .join(format!("{name}.toml")),
                )
                .unwrap(),
            );
            normalize_document(&mut doc).unwrap();
            assert!(crate::validate::validate_preset_document(&doc).is_empty());
            for &resolution in VALID_RESOLUTIONS {
                let mut gated = doc.clone();
                gated.generator.resolution = resolution;
                let errors = crate::validate::validate_preset_document(&gated);
                assert!(
                    errors.is_empty(),
                    "preset {name} failed resolution gate {resolution}: {errors:?}"
                );
            }
            let refs = make_test_resolved_refs();
            geometry.insert(compute_geometry_identity(
                doc.generator_version,
                doc.rng_version,
                &doc.generator,
            ));
            scene.insert(scene_identity(&doc, &refs));
        }
        assert_eq!(geometry.len(), 4);
        assert_eq!(scene.len(), 4);
    }
}
