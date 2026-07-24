//! Deterministic BSP dungeon generator — pure-Rust offline pipeline.
//!
//! This crate depends only on [`bsp`] for format types and [`sha2`] for
//! deterministic seeded RNG derivation. It has **zero** renderer, Vulkan,
//! windowing, audio, physics, scripting, `bsp_runtime`, or `engine_pack`
//! dependencies.
//!
//! # Architecture
//!
//! The generator is a pure function from `(u64 seed, DungeonConfig)` to
//! canonical `.map` bytes, implemented as an immutable intent pipeline:
//!
//! ```text
//! Config → PlacedLayout → RoutedIntent → EmissionIntent → bytes
//! ```
//!
//! Each stage is a validated data structure with typed construction.
//!
//! # Themes
//!
//! The CC0 Stone Beta theme at [`CC0_STONE_BETA_THEME_DIR`] provides
//! deterministic, license-clean textures and a WAD2 archive. Run
//! `build.py` in that directory to regenerate all theme assets.
//!
//! # Entry Point
//!
//! ```
//! use bsp_generator::{generate, DungeonConfig};
//!
//! let cfg = DungeonConfig::nominal_m1();
//! // Use seed 0 which routes successfully for nominal M1.
//! // (Seed-dependent routing exhaust is a pre-existing issue tracked
//! // in phase-07-generated-sprawl-topology-infeasible.)
//! let (map_bytes, meta) = generate(0, cfg).expect("generation failed");
//! assert!(!map_bytes.is_empty());
//! assert_eq!(meta.room_count, 12);
//! ```

pub mod config;
pub mod emission;
pub mod error;
pub mod geometry;
pub mod intent;
pub mod junction;
pub mod placement;
pub mod routing;
pub mod seed;
pub mod serialize;
pub mod topology;

// ── Re-exports ────────────────────────────────────────────────────────────

/// Relative path (from crate root) to the CC0 Stone Beta theme directory.
///
/// The theme directory contains `build.py` (deterministic asset generator),
/// `theme.toml` (texture role bindings), `palette.lmp`, `cc0_stone_beta.wad`,
/// `textures/` (PNG companions), and `LICENSE` (CC0 dedication).
pub const CC0_STONE_BETA_THEME_DIR: &str = "themes/cc0_stone_beta";

pub use config::{DungeonConfig, MapClass, ValidatedConfig, CONSTRUCTION_QUANTUM};
pub use emission::build_emission;
pub use error::GeneratorError;
pub use intent::{
    Brush, BrushFace, Corridor, EmissionIntent, EntityIntent, Junction, LayoutIntent, RoomIntent,
    RoutedIntent,
};
pub use junction::{
    build_junction_closures, build_l_junction, build_room_portal, build_t_junction,
    build_x_junction, make_brush,
};
pub use placement::place_rooms;
pub use routing::{route_all_edges, route_edge, CORRIDOR_HEIGHT, CORRIDOR_WIDTH};
pub use seed::{Seed, StageRng, StageSeed};
pub use serialize::serialize;
pub use topology::build_topology;

// ── Generation metadata ───────────────────────────────────────────────────

/// Descriptive metadata for a completed generation run.
///
/// Returned alongside the canonical `.map` bytes by [`generate`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GenerationMetadata {
    /// Number of placed rooms.
    pub room_count: u32,
    /// Number of corridor segments.
    pub corridor_count: u32,
    /// Number of non-worldspawn entities (spawn + lights).
    pub entity_count: u32,
    /// Estimated total face count across all brushes.
    pub face_count_estimate: u32,
    /// Axis-aligned bounding box of all rooms:
    /// `(min_x, min_y, min_z, max_x, max_y, max_z)` in Quake units.
    pub bounds: (i32, i32, i32, i32, i32, i32),
    /// The master seed used for this generation run.
    pub seed: u64,
    /// A deterministic hash of the validated configuration.
    pub config_hash: u64,
}

// ── Public entry point ────────────────────────────────────────────────────

/// Generate a canonical `.map` string and metadata from a seed and
/// configuration.
///
/// This is the sole public entry point for dungeon generation. It wires the
/// full immutable-intent pipeline:
///
/// ```text
/// config validation → seed → place rooms → build topology →
/// route edges → build emission → serialize → (bytes, metadata)
/// ```
///
/// # Errors
///
/// Returns [`GeneratorError`] if configuration validation, room placement,
/// topology construction, or corridor routing fails.
///
/// # Determinism
///
/// Two calls with identical `(seed, config)` arguments produce byte-identical
/// `.map` output. See `DECISION-20260724-08` for the serialization contract.
pub fn generate(
    seed: u64,
    config: DungeonConfig,
) -> Result<(String, GenerationMetadata), GeneratorError> {
    // 1. Validate configuration
    let validated = config.validate()?;

    // 2. Create master seed
    let master = Seed::new(seed);

    // 3. Place rooms
    let mut placement_rng = master.stage_seed("room-placement").rng();
    let rooms = place_rooms(&validated, &mut placement_rng)?;

    // 4. Build topology (connectivity graph)
    //    Shares the "corridor-routing" RNG with the routing stage — both
    //    consume from the same stream so the pipeline is deterministic.
    let mut routing_rng = master.stage_seed("corridor-routing").rng();
    let layout = build_topology(rooms, &validated, &mut routing_rng)?;

    // 5. Route all edges into corridor segments
    let routed = route_all_edges(&layout.rooms, &layout.edges, &validated, &mut routing_rng)?;

    // 6. Build emission intent (brushes + entities)
    let emission = build_emission(&layout, &routed);

    // 7. Serialize to canonical .map bytes
    let map_text = serialize(&emission);

    // 8. Compute metadata
    let room_count = layout.rooms.len() as u32;
    let corridor_count = routed.corridors.len() as u32;
    let entity_count = 1 + room_count; // spawn + one light per room

    // Face estimate: 6 faces per brush
    let face_count_estimate = emission.brushes.len() as u32 * 6;

    // Bounding box from placed rooms
    let bounds = compute_bounds(&layout);

    let config_hash = compute_config_hash(&validated);

    let metadata = GenerationMetadata {
        room_count,
        corridor_count,
        entity_count,
        face_count_estimate,
        bounds,
        seed,
        config_hash,
    };

    Ok((map_text, metadata))
}

// ── Helpers ───────────────────────────────────────────────────────────────

/// Compute the axis-aligned bounding box enclosing all placed rooms.
fn compute_bounds(layout: &LayoutIntent) -> (i32, i32, i32, i32, i32, i32) {
    if layout.rooms.is_empty() {
        return (0, 0, 0, 0, 0, 0);
    }
    let mut min_x = i32::MAX;
    let mut min_y = i32::MAX;
    let mut min_z = i32::MAX;
    let mut max_x = i32::MIN;
    let mut max_y = i32::MIN;
    let mut max_z = i32::MIN;

    for room in &layout.rooms {
        let x0 = room.position.0;
        let y0 = room.position.1;
        let z0 = room.position.2;
        let x1 = x0 + room.dimensions.0 as i32;
        let y1 = y0 + room.dimensions.1 as i32;
        let z1 = z0 + room.dimensions.2 as i32;

        min_x = min_x.min(x0);
        min_y = min_y.min(y0);
        min_z = min_z.min(z0);
        max_x = max_x.max(x1);
        max_y = max_y.max(y1);
        max_z = max_z.max(z1);
    }

    (min_x, min_y, min_z, max_x, max_y, max_z)
}

/// Compute a deterministic hash of the validated configuration.
fn compute_config_hash(config: &ValidatedConfig) -> u64 {
    use sha2::{Digest, Sha256};

    let class_byte: u8 = match config.class {
        MapClass::M1 => 1,
        MapClass::M2 => 2,
    };

    let mut hasher = Sha256::new();
    hasher.update(b"dungeon-config/v1");
    hasher.update(&[class_byte]);
    hasher.update(&config.room_count.to_le_bytes());
    hasher.update(&config.loop_count.to_le_bytes());
    hasher.update(&config.xy_bounds.0.to_le_bytes());
    hasher.update(&config.xy_bounds.1.to_le_bytes());
    hasher.update(&config.z_span.to_le_bytes());
    let digest: [u8; 32] = hasher.finalize().into();
    u64::from_le_bytes(digest[0..8].try_into().unwrap())
}
