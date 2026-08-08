//! BSP beta dogfood application.
//!
//! Provides app-owned Rapier physics integration and structural behavior
//! adapters for BSP map entities. This crate owns the physics dependency;
//! `bsp` and `bsp_runtime` remain physics-free.
//!
//! # Architecture
//!
//! - [`PhysicsBridge`](physics_bridge::PhysicsBridge): `AppBridge` implementation
//!   that creates Rapier static/dynamic/kinematic bodies and colliders from
//!   BSP collision recipes.
//! - [`RuntimeBridge`](runtime_bridge::RuntimeBridge): `AppBridge` implementation
//!   that wires structural behaviors (doors, buttons, platforms, triggers,
//!   targets, light styles).

pub mod cli;
pub mod generation;
pub mod m3_gui;
pub mod physics_bridge;
pub mod player_navigation;
pub mod richness_generation;
pub mod richness_gui;
pub mod runtime_bridge;
pub mod scene_sync;
pub mod snapshot;
