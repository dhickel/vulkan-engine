//! Enhanced v2 BSP dungeon profile — additive, M2-only, two-layer with stairs.
//!
//! This module is structurally disjoint from Legacy v1. It introduces typed
//! profile identification, validated Enhanced configuration, deterministic
//! SHA-256-framed randomness with isolated domain/tag contracts, immutable
//! intent declarations, and closed typed errors. No placement, routing,
//! emission, or execution path is provided in this phase.

pub mod config;
pub mod emission;
pub mod error;
pub mod intent;
pub mod occupancy;
pub mod pipeline;
pub mod placement;
pub mod profile;
pub mod reservation;
pub mod routing;
pub mod seed;
pub mod theme;
pub mod topology;
pub mod transition;
