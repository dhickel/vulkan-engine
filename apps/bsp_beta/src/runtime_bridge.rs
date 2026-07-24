//! App-owned runtime bridge for BSP structural behavior wiring.
//!
//! Implements the [`AppBridge`] trait to manage door, button, platform,
//! trigger, target, and light-style state machines. This bridge receives
//! entity descriptors during prepare, initializes behavior state machines,
//! and wires trigger→target activation chains.
//!
//! # State Machine Ownership
//!
//! The runtime bridge owns the [`StructuralBehaviorAdapter`] and advances
//! it each frame. Moving entities produce position updates that the physics
//! bridge syncs with Rapier kinematic bodies.
//!
//! # Activation Flow
//!
//! ```text
//! Trigger enter → TriggerState::update_occupants → TriggerEvent::Fired
//!     → queue_target_activation(target) → Door/Button/Platform::activate
//!     → door.update(dt) → new position → scene/collider sync
//! ```

use std::collections::HashSet;

use bsp_runtime::behavior::{BehaviorEntityInfo, StructuralBehaviorAdapter, TriggerEvent};
use bsp_runtime::bridge::{
    AppBridge, BehaviorEntityRecipe, BridgeToken, EntityCollisionRecipe, LightEntityRecipe,
    WorldCollisionRecipe,
};
use bsp_runtime::Activation;
use log;

/// Runtime bridge that manages BSP structural behaviors.
///
/// Owns the `StructuralBehaviorAdapter` and advances it each frame.
/// Exposes trigger occupant tracking and entity position queries for
/// integration with the physics bridge and scene graph.
pub struct RuntimeBridge {
    /// The structural behavior adapter.
    pub adapter: StructuralBehaviorAdapter,
    /// Staged entity info from prepare, registered on commit.
    staged_infos: Option<Vec<BehaviorEntityInfo>>,
    /// Whether we've been committed.
    committed: bool,
}

impl RuntimeBridge {
    /// Create a new runtime bridge.
    pub fn new() -> Self {
        Self {
            adapter: StructuralBehaviorAdapter::new(),
            staged_infos: None,
            committed: false,
        }
    }

    /// Advance all behavior state machines by `dt` seconds.
    ///
    /// Returns position updates for moving entities as `(entity_index, new_position)`.
    pub fn update(&mut self, dt: f32) -> Vec<(u32, [f32; 3])> {
        self.adapter.update(dt)
    }

    /// Register a trigger occupant change and return any firing events.
    ///
    /// `occupants` is the set of entity indices currently inside the trigger volume.
    pub fn update_trigger(
        &mut self,
        trigger_entity: u32,
        occupants: HashSet<u32>,
    ) -> Option<TriggerEvent> {
        self.adapter
            .update_trigger_occupants(trigger_entity, occupants)
    }

    /// Activate an entity by name.
    pub fn activate_entity(&mut self, target: &str, activation: Activation) {
        self.adapter.activate_by_target(target, activation);
    }

    /// Get the current world-space position of an entity.
    pub fn entity_position(&self, entity_index: u32) -> Option<[f32; 3]> {
        self.adapter.entity_position(entity_index)
    }

    /// Check if an entity is currently in motion.
    pub fn is_moving(&self, entity_index: u32) -> bool {
        self.adapter.is_moving(entity_index)
    }

    /// Get light style active state.
    pub fn light_style_active(&self, style: &str) -> bool {
        self.adapter.light_style_active(style)
    }

    /// Set light style.
    pub fn set_light_style(&mut self, style: &str, active: bool) {
        self.adapter.set_light_style(style, active);
    }

    /// Export current mutable behavior state for persistence.
    pub fn export_state(&self) -> bsp_runtime::MutableBehaviorState {
        self.adapter.export_state()
    }

    /// Import mutable behavior state from a persistence payload.
    pub fn import_state(&mut self, state: &bsp_runtime::MutableBehaviorState) {
        self.adapter.import_state(state);
    }
}

impl Default for RuntimeBridge {
    fn default() -> Self {
        Self::new()
    }
}

impl AppBridge for RuntimeBridge {
    fn name(&self) -> &str {
        "runtime"
    }

    fn prepare(
        &mut self,
        _world_collider: &WorldCollisionRecipe,
        _entity_colliders: &[EntityCollisionRecipe],
        lights: &[LightEntityRecipe],
        behaviors: &[BehaviorEntityRecipe],
    ) -> Result<BridgeToken, String> {
        // Convert BehaviorEntityRecipe to BehaviorEntityInfo for adapter registration.
        let mut entity_infos: Vec<BehaviorEntityInfo> = behaviors
            .iter()
            .map(|b| BehaviorEntityInfo {
                entity_index: b.entity_index,
                classname: b.classname.clone(),
                targetname: b.targetname.clone(),
                target: b.target.clone(),
                killtarget: b.killtarget.clone(),
                origin: [b.origin.x, b.origin.y, b.origin.z],
                movedir: b.movedir,
                speed: b.speed,
                wait: b.wait,
                lip: b.lip,
                height: b.height,
                light_style: b.light_style.clone(),
            })
            .collect();

        entity_infos.extend(lights.iter().filter_map(|light| {
            light.style.as_ref().map(|style| BehaviorEntityInfo {
                entity_index: light.entity_index,
                classname: "light".to_string(),
                targetname: None,
                target: None,
                killtarget: None,
                origin: [light.origin.x, light.origin.y, light.origin.z],
                movedir: None,
                speed: None,
                wait: None,
                lip: None,
                height: None,
                light_style: Some(style.clone()),
            })
        }));

        let count = entity_infos.len();
        self.staged_infos = Some(entity_infos);

        log::debug!("Runtime bridge prepare: {} behavior entities staged", count);
        Ok(BridgeToken::new(vec![1u8])) // generation marker
    }

    fn validate(&self, token: &BridgeToken) -> Result<(), String> {
        if token.payload.is_empty() {
            return Err("empty bridge token".to_string());
        }

        let staged = self
            .staged_infos
            .as_ref()
            .ok_or_else(|| "no staged behavior state".to_string())?;

        if staged.is_empty() {
            log::debug!("Runtime bridge: no behavior entities to validate");
        }

        Ok(())
    }

    fn commit(&mut self, token: BridgeToken) -> Result<(), String> {
        if token.payload.is_empty() {
            return Err("empty bridge token".to_string());
        }

        // Allow re-commit: reset prior committed state before accepting new batch.
        if self.committed {
            log::debug!("Runtime bridge: resetting prior committed state for new mount");
            self.committed = false;
            self.adapter.reset();
        }

        let staged = self
            .staged_infos
            .take()
            .ok_or_else(|| "no staged behavior state to commit".to_string())?;

        // Register all entities into the adapter
        self.adapter.register_entities(staged);

        self.committed = true;
        log::debug!(
            "Runtime bridge commit: {} doors, {} buttons, {} platforms, {} triggers, {} light styles",
            self.adapter.doors.len(),
            self.adapter.buttons.len(),
            self.adapter.platforms.len(),
            self.adapter.triggers.len(),
            self.adapter.light_styles.len(),
        );

        Ok(())
    }

    fn rollback(&mut self, _token: BridgeToken) {
        self.staged_infos = None;
        self.adapter.reset();
        self.committed = false;
    }
}
