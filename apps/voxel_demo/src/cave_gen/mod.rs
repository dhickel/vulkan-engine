//! Cave generation module — ported from the voxel-cave spike.
//!
//! This module is pure computation with no Vulkan or winit dependency.
//! It provides:
//! - `lattice`  — DenseLattice<i8> density + DenseLattice<u8> material tags
//! - `rng`      — PCG32 V1 RNG with phase-tagged streams
//! - `noise`    — 3D Perlin noise with FBM
//! - `generators` — Generator trait, topology-first generator, carving helpers
//! - `metrics`  — Route quality metrics and camera-pose derivation

pub mod generators;
pub mod lattice;
pub mod metrics;
pub mod noise;
pub mod rng;
