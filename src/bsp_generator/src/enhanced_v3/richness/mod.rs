//! EnhancedV3 Richness V1 request contract — crate-private.
//!
//! This module defines the immutable revisioned request, canonical
//! provenance contract, stable errors, and deterministic identity
//! hashing for the Richness V1 product.
//!
//! # Isolation rule
//!
//! Richness V1 is structurally isolated from baseline V3. It:
//! - Never mutates `V3Config`, `V3Preset`, or baseline `V3Error`.
//! - Never routes v1, v2, or baseline-v3 RNG, serialization, ordering,
//!   geometry, metadata, CLI, explorer, or packaging through richness code.
//! - Uses its own domain `dungeon-gen/v3-richness/v1/request` for identity
//!   hashes, independent of all generation domains.
//! - Keeps all types crate-private until the atomic release phase.
//!
//! # No-runtime-RON rule
//!
//! The generator performs NO runtime parsing, filesystem access, network
//! access, or platform-sensitive randomization. Authored RON is an offline
//! source format compiled by `tools/richness_content_codegen`; the generated
//! Rust file is checked in and byte-compared with a fresh code-generation
//! run. Runtime generation reads only checked-in Rust constants.

pub(crate) mod canonical;
pub(crate) mod content_types;
pub(crate) mod error;
pub(crate) mod fields;
pub(crate) mod fixed;
pub(crate) mod footprint;
pub(crate) mod generated_content;
pub(crate) mod ids;
pub(crate) mod metadata;
pub(crate) mod pacing;
pub(crate) mod request;
pub(crate) mod reservation;
pub(crate) mod sampling;
pub(crate) mod solver;
pub(crate) mod topology;
pub(crate) mod zones;

#[cfg(test)]
mod vectors;

// No public re-exports — richness parent module stays crate-private until
// the atomic release phase.
