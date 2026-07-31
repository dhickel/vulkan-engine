//! Enhanced v3 proof support for the compiler-specific integration target.
//!
//! The compiler gate does not consume the proof corpus. Keeping that module
//! out of this target preserves the Phase 06 test boundary while all shared
//! compiler, geometry, and assembly invariants remain covered.

pub mod assembly;
pub mod compiler;
pub mod contract;
pub mod emission;
pub mod fixtures;
pub mod footprint;
pub mod geometry;
pub mod ir;
pub mod metadata;
pub mod pipeline;
pub mod planner;
pub mod portal;
pub mod prefab;
pub mod seed;
pub mod topology;
