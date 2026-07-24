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
//! Generation entrypoints are **not yet exposed** — this crate currently
//! provides only the foundation types, configuration, errors, and seed
//! infrastructure.
//!
//! # Entry Points (future)
//!
//! ```ignore
//! let cfg = DungeonConfig::nominal_m2();
//! let valid = cfg.validate()?;               // future: generate(&valid, seed)
//! ```

pub mod config;
pub mod error;
pub mod geometry;
pub mod intent;
pub mod junction;
pub mod placement;
pub mod routing;
pub mod seed;
pub mod topology;

// ── Re-exports ────────────────────────────────────────────────────────────

pub use config::{DungeonConfig, MapClass, ValidatedConfig, CONSTRUCTION_QUANTUM};
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
pub use topology::build_topology;
