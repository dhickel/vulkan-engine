//! App-owned runtime bridge for BSP structural behavior wiring.
//!
//! Implements the [`AppBridge`] trait to manage door, button, platform,
//! trigger, target, and light-style state machines. This bridge receives
//! entity descriptors during prepare, initializes behavior state machines,
//! and wires trigger→target activation chains.
//!
//! # Phase 05: Active Bridge Receipts
//!
//! - **Prepare**: Creates a separate adapter state from entity recipes.
//! - **Validate**: Confirms entity count and consistency.
//! - **Activate**: Moves prevalidated adapter state into an active receipt.
//!   Does not reset the previously active A.
//! - **Teardown**: Resets the adapter, generation-specific cleanup.

use bsp_runtime::behavior::{BehaviorEntityInfo, StructuralBehaviorAdapter, TriggerEvent};
use bsp_runtime::bridge::{
    ActiveBridgeState, AppBridge, BehaviorEntityRecipe, EntityCollisionRecipe, LightEntityRecipe,
    PreparedBridgeState, WorldCollisionRecipe,
};
use bsp_runtime::Activation;
use log;

// ── Prepared Runtime State ─────────────────────────────────────────────

/// Prepared behavior state built from entity recipes during prepare.
#[derive(Debug)]
pub struct RuntimePreparedState {
    /// Staged entity infos to register during activation.
    pub entity_infos: Vec<BehaviorEntityInfo>,
}

impl PreparedBridgeState for RuntimePreparedState {
    fn registration_name(&self) -> &str {
        "runtime"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

// ── Active Runtime State ───────────────────────────────────────────────

/// Published behavior adapter moved into the active receipt during activation.
#[derive(Debug)]
pub struct RuntimeActiveState {
    /// The active behavior adapter.
    pub adapter: StructuralBehaviorAdapter,
}

impl ActiveBridgeState for RuntimeActiveState {
    fn registration_name(&self) -> &str {
        "runtime"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

// ── Runtime Bridge ─────────────────────────────────────────────────────

/// Runtime bridge that manages BSP structural behaviors.
///
/// Owns the `StructuralBehaviorAdapter` and advances it each frame.
/// Exposes trigger occupant tracking and entity position queries for
/// integration with the physics bridge and scene graph.
pub struct RuntimeBridge {
    /// The structural behavior adapter (populated from active receipt).
    pub adapter: StructuralBehaviorAdapter,
}

impl RuntimeBridge {
    /// Create a new runtime bridge.
    pub fn new() -> Self {
        Self {
            adapter: StructuralBehaviorAdapter::new(),
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
        occupants: std::collections::HashSet<u32>,
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
    ) -> Result<Box<dyn PreparedBridgeState>, String> {
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
        log::debug!("Runtime bridge prepare: {} behavior entities staged", count);
        Ok(Box::new(RuntimePreparedState { entity_infos }))
    }

    fn validate(&self, prepared: &dyn PreparedBridgeState) -> Result<(), String> {
        let state: &RuntimePreparedState = prepared
            .as_any()
            .downcast_ref::<RuntimePreparedState>()
            .ok_or_else(|| "runtime bridge received non-runtime prepared state".to_string())?;

        if state.entity_infos.is_empty() {
            log::debug!("Runtime bridge: no behavior entities to validate");
        }

        Ok(())
    }

    fn activate(&mut self, prepared: &mut dyn PreparedBridgeState) -> Box<dyn ActiveBridgeState> {
        let state: &mut RuntimePreparedState = prepared
            .as_any_mut()
            .downcast_mut::<RuntimePreparedState>()
            .expect("runtime bridge received non-runtime prepared state");

        let mut adapter = StructuralBehaviorAdapter::new();
        adapter.register_entities(std::mem::take(&mut state.entity_infos));

        // Clone into the active receipt; the receipt owns the teardown target.
        let receipt_adapter = adapter.clone();
        // The bridge's own adapter is the live one the app drives.
        self.adapter = adapter;

        log::debug!(
            "Runtime bridge activate: {} doors, {} buttons, {} platforms, {} triggers, {} light styles",
            receipt_adapter.doors.len(),
            receipt_adapter.buttons.len(),
            receipt_adapter.platforms.len(),
            receipt_adapter.triggers.len(),
            receipt_adapter.light_styles.len(),
        );

        Box::new(RuntimeActiveState {
            adapter: receipt_adapter,
        })
    }

    fn teardown(&mut self, active: &mut dyn ActiveBridgeState) -> Result<(), String> {
        let state: &mut RuntimeActiveState = active
            .as_any_mut()
            .downcast_mut::<RuntimeActiveState>()
            .ok_or_else(|| "runtime bridge received non-runtime active state".to_string())?;

        // Reset the receipt's adapter (generation-specific cleanup)
        state.adapter.reset();

        // Also reset the bridge's own adapter
        self.adapter.reset();

        Ok(())
    }

    fn rollback(&mut self, prepared: &mut dyn PreparedBridgeState) {
        let state: &mut RuntimePreparedState = prepared
            .as_any_mut()
            .downcast_mut::<RuntimePreparedState>()
            .expect("runtime bridge received non-runtime prepared state");

        // Clear staged entity infos
        state.entity_infos.clear();
    }
}
